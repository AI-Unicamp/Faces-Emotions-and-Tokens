#!/usr/bin/env python3
"""
Condition already-cut VCV clips for the BlendshapeDecoder.

Use this AFTER your normal CosyVoice synthesis + cutting (keep those exactly as
they are -- they pronounce the pseudo-words correctly because the full-list
context keeps CosyVoice in phonetic-spelling mode). This pass only fixes the
abrupt start/end by trimming to the speech boundaries and re-padding low-level
room-tone, so the model has silence to render a closed mouth at start and end.

It does NOT re-synthesize anything, so pronunciation is untouched.

Run:
    python condition_clips.py --in_dir results/cut_clips --out_dir results/vcv_stimuli
    python condition_clips.py --in_dir results/cut_clips --out_dir results/vcv_stimuli --lead_ms 500 --tail_ms 300
    python condition_clips.py --in_dir results/cut_clips --out_dir results/vcv_stimuli --lead_ms 400 --tail_ms 400 --no_trim
"""
import argparse
from pathlib import Path

import torch
import torchaudio

DEFAULT_LEAD_MS = 500    # silence padding added BEFORE the clip
DEFAULT_TAIL_MS = 300    # silence padding added AFTER the clip
TRIM_DB         = 35    # frames quieter than (peak - TRIM_DB) are trimmed
MIN_FLOOR       = 1e-4  # amplitude floor for padding noise (~ -80 dBFS)


def to_mono(wav):
    return wav.mean(0, keepdim=True) if wav.shape[0] > 1 else wav


def estimate_noise_floor(wav, sr, frame_ms=20, pct=10):
    """Low percentile of frame energy -> robust even if the clip edges are
    already speech (which manually-cut clips often are)."""
    x = wav.abs().mean(0)
    frame = max(1, int(sr * frame_ms / 1000))
    nf = x.shape[0] // frame
    if nf < 2:
        return MIN_FLOOR
    rms = x[:nf * frame].reshape(nf, frame).pow(2).mean(1).sqrt()
    return max(MIN_FLOOR, float(torch.quantile(rms, pct / 100.0)))


def trim_silence(wav, sr, thresh_db=TRIM_DB, frame_ms=20):
    x = wav.abs().mean(0)
    frame = max(1, int(sr * frame_ms / 1000))
    nf = x.shape[0] // frame
    if nf < 2:
        return wav
    rms = x[:nf * frame].reshape(nf, frame).pow(2).mean(1).sqrt()
    keep = rms > rms.max().clamp(min=1e-9) * (10 ** (-thresh_db / 20))
    idx = torch.where(keep)[0]
    if len(idx) == 0:
        return wav
    start = int(idx[0]) * frame
    end = min((int(idx[-1]) + 1) * frame, wav.shape[1])
    return wav[:, start:end]


def pad_silence(wav, sr, lead_ms, tail_ms, noise_floor):
    n_lead = int(sr * lead_ms / 1000)
    n_tail = int(sr * tail_ms / 1000)
    lead = torch.randn(wav.shape[0], n_lead) * noise_floor
    tail = torch.randn(wav.shape[0], n_tail) * noise_floor
    return torch.cat([lead, wav, tail], dim=1)


def condition(wav, sr, lead_ms, tail_ms, do_trim=True):
    wav = to_mono(wav)
    floor = estimate_noise_floor(wav, sr)
    if do_trim:
        wav = trim_silence(wav, sr)
    wav = pad_silence(wav, sr, lead_ms, tail_ms, floor)
    peak = wav.abs().max().clamp(min=1e-9)
    if peak > 1.0:
        wav = wav / peak
    return wav


def main():
    ap = argparse.ArgumentParser(description="Trim + pad already-cut VCV clips.")
    ap.add_argument('--in_dir', default='new_approach/I_I_wavs_v2', help="Folder of cut .wav clips.")
    ap.add_argument('--out_dir', default='data_clean/I_I_wavs_padded_v2', help="Where to write conditioned clips.")
    ap.add_argument('--lead_ms', type=int, default=DEFAULT_LEAD_MS, help="Silence before the clip (ms).")
    ap.add_argument('--tail_ms', type=int, default=DEFAULT_TAIL_MS, help="Silence after the clip (ms).")
    ap.add_argument('--no_trim', action='store_true',
                    help="Skip silence trimming (just pad). Use if your cuts are already tight.")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(in_dir.rglob('*.wav'))
    if not wavs:
        raise SystemExit(f"No .wav files found under {in_dir}")

    for p in wavs:
        wav, sr = torchaudio.load(str(p))
        out = condition(wav, sr, args.lead_ms, args.tail_ms, do_trim=not args.no_trim)
        # preserve any sub-folder structure (e.g. A/, I/, O/, U/)
        rel = p.relative_to(in_dir)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(dst), out, sr)
        print(f'{rel}  {out.shape[1] / sr:5.2f}s -> {dst}')


if __name__ == '__main__':
    main()


'''

python audio_padding.py --in_dir new_approach/results/ --out_dir new_approach/results/ --lead_ms --tail_ms --no_trim

'''