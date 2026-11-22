#!/usr/bin/env python3
"""
Return the blendshape sequence for one timestamp range, sliced from a full
inference animation .npz, with optional rest-frame padding in blendshape space.

Padding adds held-neutral frames before/after the slice (lead/tail), with a
short ease so the mouth opens into the first frame and closes after the last.
Set --lead_ms 0 --tail_ms 0 for the bare slice.

Run:
    # bare slice
    python get_slice.py --animation results/I_I_full.npz --start 1.20 --end 1.95 --out results/IZI.npz

    # with rest padding
    python get_slice.py --animation results/I_I_full.npz --start 1.20 --end 1.95 --out results/IZI.npz \
        --lead_ms 500 --tail_ms 500 --ease_ms 120
"""
import argparse
from pathlib import Path
import numpy as np


def pad_animation(bs, fps, lead_ms, tail_ms, ease_ms, neutral=None):
    """Prepend/append held-neutral rest frames with a cosine ease into the first
    / out of the last frame. Returns (padded_bs, n_lead, n_tail)."""
    bs = np.asarray(bs, dtype=np.float32)
    if lead_ms <= 0 and tail_ms <= 0:
        return bs, 0, 0

    D = bs.shape[1]
    neutral = np.zeros(D, np.float32) if neutral is None else np.asarray(neutral, np.float32)

    n_lead = int(lead_ms / 1000 * fps)
    n_tail = int(tail_ms / 1000 * fps)
    n_ease = max(1, min(int(ease_ms / 1000 * fps), max(n_lead, 1), max(n_tail, 1)))
    w = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n_ease)))[:, None]   # 0 -> 1

    parts = []
    if n_lead > 0:
        parts.append(np.repeat(neutral[None], max(0, n_lead - n_ease), axis=0))  # hold closed
        parts.append((1 - w) * neutral + w * bs[0])                              # open into first frame
    parts.append(bs)
    if n_tail > 0:
        parts.append((1 - w) * bs[-1] + w * neutral)                             # close from last frame
        parts.append(np.repeat(neutral[None], max(0, n_tail - n_ease), axis=0))  # hold closed
    return np.concatenate(parts, axis=0), n_lead, n_tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--animation', required=True, help="Full animation .npz")
    ap.add_argument('--start', type=float, required=True, help="Start time (seconds)")
    ap.add_argument('--end', type=float, required=True, help="End time (seconds)")
    ap.add_argument('--out', default=None, help="Optional .npz to save the slice")
    ap.add_argument('--lead_ms', type=int, default=0, help="Rest frames before the slice (ms).")
    ap.add_argument('--tail_ms', type=int, default=0, help="Rest frames after the slice (ms).")
    ap.add_argument('--ease_ms', type=int, default=120, help="Open/close transition length (ms).")
    args = ap.parse_args()

    data = np.load(args.animation, allow_pickle=True)
    bs = np.asarray(data['arkit_blendshapes'], dtype=np.float32)
    fps = int(data['fps']) if 'fps' in data else 30

    s = max(0, round(args.start * fps))
    e = min(bs.shape[0], round(args.end * fps))
    sliced = bs[s:e]

    padded, n_lead, n_tail = pad_animation(
        sliced, fps, args.lead_ms, args.tail_ms, args.ease_ms
    )

    print(f"frames {s}:{e}  slice {sliced.shape[0]} fr  +pad({n_lead},{n_tail})  "
          f"-> {padded.shape}  ({padded.shape[0] / fps:.2f}s @ {fps} fps)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, **{
            'arkit_blendshapes': padded,
            'fps': fps,
            'num_frames': padded.shape[0],
            'lead_frames': n_lead,   # prepend this many frames of silence to playback audio
            'tail_frames': n_tail,   # append this many frames of silence to playback audio
        })
        print(f"saved -> {args.out}")


if __name__ == '__main__':
    main()


'''

python get_slice.py --animation data/all_A_A_sanity.npz --start 0.330 --end 1.011 --out ./results/i_said.npz \
    --lead_ms 500 --tail_ms 300 --ease_ms 120 
'''
