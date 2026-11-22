#!/usr/bin/env python3
"""
Assemble per-VCV stimuli (animation + audio) from ONE full recording.

For every VCV it builds:

    [lead rest] [I said] [gap rest] [VCV] [tail rest]

The animation is assembled in BLENDSHAPE space with a short open/close ease at
every rest<->speech boundary; the eases are carved OUT of the rest durations,
so total animation length == lead + isaid + gap + vcv + tail. The audio is built
with the same three gap durations, so animation and audio stay frame-for-sample
in sync. The "I said" segment is cut once and reused for every VCV.

Timestamps are in audio seconds, measured on the SAME recording the animation
came from (so audio time == animation time). CSV format (header optional,
'#' lines ignored), with ONE carrier row whose label matches --isaid_label:

    label,start,end
    I_SAID,0.330,1.011
    ABA,1.250,1.980
    ACHA,2.310,3.090
    ...

Run:
    python assemble_samples.py \
        --animation data/all_A_A_sanity.npz \
        --audio     data/all_A_A_sanity.wav \
        --timestamps data/A_A_marks.csv \
        --out_dir   results/A_A_samples \
        --lead_ms 500 --gap_ms 300 --tail_ms 500 --ease_ms 120
"""
import argparse
import csv
from pathlib import Path

import numpy as np

MIN_FLOOR = 1e-4   # amplitude floor for the audio silence (~ -80 dBFS)


# ---------------------------------------------------------------------
# Animation assembly (blendshape space)
# ---------------------------------------------------------------------
def _ramp(n):
    """Cosine ramp 0 -> 1, shape (n, 1)."""
    return (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n)))[:, None]


def build_rest(n_rest, before, after, neutral, ease_frames):
    """One rest block of EXACTLY n_rest frames:
    close from `before` to neutral, hold neutral, open into `after`.
    `before`/`after` are the adjacent speech segments (or None at the edges)."""
    ends = int(before is not None) + int(after is not None)
    n_e = min(ease_frames, n_rest // ends) if ends else 0
    n_e = max(0, n_e)

    parts = []
    if before is not None and n_e > 0:
        w = _ramp(n_e)
        parts.append((1 - w) * before[-1] + w * neutral)     # close to neutral
    hold = max(0, n_rest - n_e * ends)
    if hold > 0:
        parts.append(np.repeat(neutral[None], hold, axis=0))  # hold closed
    if after is not None and n_e > 0:
        w = _ramp(n_e)
        parts.append((1 - w) * neutral + w * after[0])        # open into after

    if not parts:
        return np.zeros((0, neutral.shape[0]), np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def assemble_animation(segments, rests, fps, ease_ms, neutral=None):
    """segments: list of speech blendshape arrays (T_i, D).
       rests: list of rest durations in FRAMES, length len(segments)+1.
       Returns (assembled, offsets) where offsets[i] is the start frame of
       segment i inside the assembled sequence."""
    D = segments[0].shape[1]
    neutral = np.zeros(D, np.float32) if neutral is None else np.asarray(neutral, np.float32)
    ease_frames = int(ease_ms / 1000 * fps)

    out, offsets, cur = [], [], 0
    for i in range(len(segments) + 1):
        before = segments[i - 1] if i > 0 else None
        after = segments[i] if i < len(segments) else None
        rb = build_rest(rests[i], before, after, neutral, ease_frames)
        out.append(rb)
        cur += rb.shape[0]
        if i < len(segments):
            offsets.append(cur)
            out.append(segments[i])
            cur += segments[i].shape[0]
    return np.concatenate(out, axis=0).astype(np.float32), offsets


# ---------------------------------------------------------------------
# Audio assembly
# ---------------------------------------------------------------------
def silence(ms, sr, floor):
    n = int(ms / 1000 * sr)
    return (np.random.randn(n).astype(np.float32) * floor) if n > 0 else np.zeros(0, np.float32)


def assemble_audio(isaid_a, vcv_a, sr, lead_ms, gap_ms, tail_ms, floor):
    return np.concatenate([
        silence(lead_ms, sr, floor),
        isaid_a,
        silence(gap_ms, sr, floor),
        vcv_a,
        silence(tail_ms, sr, floor),
    ]).astype(np.float32)


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def read_timestamps(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = next(csv.reader([line]))
            if len(parts) < 3:
                raise ValueError(f"Expected 'label,start,end', got: {raw!r}")
            label = parts[0].strip()
            try:
                start, end = float(parts[1]), float(parts[2])
            except ValueError:
                continue  # header row -> skip
            if end <= start:
                raise ValueError(f"end <= start for '{label}' ({start}, {end})")
            rows.append((label, start, end))
    if not rows:
        raise SystemExit(f"No valid rows in {csv_path}")
    return rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Assemble per-VCV samples (animation + audio).")
    ap.add_argument('--animation', required=True, help="Full inference animation .npz.")
    ap.add_argument('--audio', required=True, help="Full recording .wav (same one the animation came from).")
    ap.add_argument('--timestamps', required=True, help="CSV label,start,end (seconds); one carrier row.")
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--isaid_label', default='I_SAID', help="Label of the carrier row to reuse (default: I_SAID).")
    ap.add_argument('--lead_ms', type=int, default=500, help="Rest before 'I said'.")
    ap.add_argument('--gap_ms', type=int, default=300, help="Gap between 'I said' and the VCV.")
    ap.add_argument('--tail_ms', type=int, default=500, help="Rest after the VCV.")
    ap.add_argument('--ease_ms', type=int, default=120, help="Open/close transition at each boundary.")
    ap.add_argument('--margin_ms', type=int, default=0, help="Extra context kept on each side of every cut (ms).")
    args = ap.parse_args()

    import torchaudio  # lazy: keeps the assembler importable without torchaudio

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load animation ----
    data = np.load(args.animation, allow_pickle=True)
    bs_full = np.asarray(data['arkit_blendshapes'], dtype=np.float32)
    fps = int(data['fps']) if 'fps' in data else 30
    T = bs_full.shape[0]

    # ---- load audio (mono) ----
    wav, sr = torchaudio.load(args.audio)
    audio_full = wav.mean(0).numpy().astype(np.float32) if wav.shape[0] > 1 else wav[0].numpy().astype(np.float32)
    floor = max(MIN_FLOOR, float(np.quantile(np.abs(audio_full), 0.10)))

    m_f = int(args.margin_ms / 1000 * fps)   # margin in frames
    m_s = int(args.margin_ms / 1000 * sr)    # margin in samples

    def cut(start_s, end_s):
        af = bs_full[max(0, round(start_s * fps) - m_f): min(T, round(end_s * fps) + m_f)]
        aa = audio_full[max(0, round(start_s * sr) - m_s): min(len(audio_full), round(end_s * sr) + m_s)]
        return af, aa

    # ---- rests in frames ----
    lead_f = round(args.lead_ms / 1000 * fps)
    gap_f = round(args.gap_ms / 1000 * fps)
    tail_f = round(args.tail_ms / 1000 * fps)

    # ---- timestamps ----
    rows = read_timestamps(args.timestamps)
    carrier = [r for r in rows if r[0] == args.isaid_label]
    if not carrier:
        raise SystemExit(f"No carrier row labelled '{args.isaid_label}' found in {args.timestamps}")
    _, c_start, c_end = carrier[0]
    isaid_anim, isaid_audio = cut(c_start, c_end)

    vcv_rows = [r for r in rows if r[0] != args.isaid_label]
    print(f"Animation {T} fr @ {fps} fps | audio {len(audio_full)/sr:.2f}s @ {sr} Hz | "
          f"'I said' = {isaid_anim.shape[0]} fr | {len(vcv_rows)} VCVs")

    for label, s, e in vcv_rows:
        vcv_anim, vcv_audio = cut(s, e)

        anim, offsets = assemble_animation(
            [isaid_anim, vcv_anim], [lead_f, gap_f, tail_f], fps, args.ease_ms
        )
        aud = assemble_audio(
            isaid_audio, vcv_audio, sr, args.lead_ms, args.gap_ms, args.tail_ms, floor
        )

        npz_path = out_dir / f"{label}.npz"
        wav_path = out_dir / f"{label}.wav"

        np.savez(npz_path, **{
            'arkit_blendshapes': anim,
            'fps': fps,
            'utterance_id': label,
            'word': label,
            'num_frames': anim.shape[0],
            'isaid_start_frame': offsets[0],
            'vcv_start_frame': offsets[1],
            'lead_frames': lead_f,
            'gap_frames': gap_f,
            'tail_frames': tail_f,
            'ease_frames': int(args.ease_ms / 1000 * fps),
            'vcv_src_sec': (s, e),
        })
        torchaudio.save(str(wav_path), __import__('torch').from_numpy(aud).unsqueeze(0), sr)

        print(f"  {label:8s} anim {anim.shape[0]:4d} fr ({anim.shape[0]/fps:5.2f}s) | "
              f"audio {len(aud)/sr:5.2f}s -> {npz_path.name}, {wav_path.name}")

    print(f"Done. Output in {out_dir}")


if __name__ == '__main__':
    main()


'''

python auto_crop_animation_audio.py \
    --animation data/all_U_U_sanity.npz \
    --audio     data/all_U_U_sanity.wav \
    --timestamps data/U_U_timestamps.csv \
    --out_dir   results/U_U_samples \
    --lead_ms 500 --gap_ms 400 --tail_ms 500 --ease_ms 200
    
'''