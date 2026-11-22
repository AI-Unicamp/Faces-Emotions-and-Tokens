#!/usr/bin/env python3
"""
Batch fixed-emotion insertion.

For every neutral .npz in a folder (ABA.npz, ACHA.npz, ...) apply EACH requested
emotion across the whole clip, writing one output per (sample x emotion):

    ABA.npz  ->  ABA_happy.npz, ABA_sad.npz, ABA_angry.npz
    ACHA.npz ->  ACHA_happy.npz, ...

Self-contained: only needs numpy (no torch / librosa / inference imports), since
it just loads existing neutral blendshapes and overlays emotion vectors.

Run:
    python batch_emotion.py \
        --in_dir  new_approach/results/A_A_samples_v2 \
        --out_dir new_approach/results/A_A_emotions \
        --emotion_vectors data/emotion_vectors.json \
        --emotions happy sad angry \
        --transition_frames 0
"""
import argparse
import json
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------------
ARKIT_BLENDSHAPE_NAMES = [
    'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
    'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight',
    'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
    'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
    'eyeWideLeft', 'eyeWideRight',
    'jawForward', 'jawLeft', 'jawOpen', 'jawRight',
    'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight',
    'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel',
    'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight',
    'mouthPressLeft', 'mouthPressRight', 'mouthPucker',
    'mouthRight', 'mouthRollLower', 'mouthRollUpper',
    'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight',
    'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight',
    'noseSneerLeft', 'noseSneerRight',
]
NAME_TO_INDEX = {name: i for i, name in enumerate(ARKIT_BLENDSHAPE_NAMES)}

FPS = 30
EMOTION_ALIASES = {'happy': 'h', 'sad': 's', 'angry': 'a', 'anger': 'a', 'neutral': 'n'}


# -----------------------------------------------------------------------------
def load_emotion_vectors(json_path):
    with open(json_path, 'r') as f:
        raw = json.load(f)
    vectors = {}
    for emotion_key, name_to_value in raw.items():
        if emotion_key.startswith('_'):
            continue
        if not isinstance(name_to_value, dict):
            raise ValueError(f"Emotion '{emotion_key}' must map to a blendshape-name -> value object.")
        vec = np.zeros(len(ARKIT_BLENDSHAPE_NAMES), dtype=np.float32)
        for bs_name, value in name_to_value.items():
            if bs_name.startswith('_'):
                continue
            if bs_name not in NAME_TO_INDEX:
                raise KeyError(f"Unknown blendshape '{bs_name}' for emotion '{emotion_key}'.")
            vec[NAME_TO_INDEX[bs_name]] = float(value)
        vectors[emotion_key] = vec
    return vectors


def resolve_emotion(label, emotion_vectors):
    if label in emotion_vectors:
        return label
    alias = EMOTION_ALIASES.get(label.lower())
    if alias and alias in emotion_vectors:
        return alias
    raise SystemExit(
        f"Emotion '{label}' not found in the JSON. Available keys: {list(emotion_vectors)}. "
        f"Pass one of those, or a name (happy/sad/angry) that maps to them."
    )


def load_neutral_from_file(path):
    ext = Path(path).suffix.lower()
    if ext == '.npz':
        data = np.load(path, allow_pickle=True)
        if 'arkit_blendshapes' not in data.files:
            raise KeyError(f"{path}: .npz missing 'arkit_blendshapes'. Keys: {data.files}")
        arr = np.asarray(data['arkit_blendshapes'], dtype=np.float32)
        extras = {k: data[k] for k in data.files if k != 'arkit_blendshapes'}
        return arr, extras
    elif ext == '.npy':
        return np.asarray(np.load(path), dtype=np.float32), {}
    raise ValueError(f"Unsupported extension {ext}; expected .npy or .npz")


# -----------------------------------------------------------------------------
def create_transition_weights(total_frames, transition_frames):
    """Smoothstep fade-in/out across the single whole-clip segment."""
    weights = np.zeros(total_frames)
    seg_len = total_frames
    actual = min(transition_frames, seg_len // 4) if transition_frames > 0 else 0
    for j in range(seg_len):
        d_start, d_end = j, seg_len - j
        if actual == 0:
            weights[j] = 1.0
        elif d_start < actual:
            t = d_start / actual
            weights[j] = t * t * (3 - 2 * t)
        elif d_end < actual:
            t = d_end / actual
            weights[j] = t * t * (3 - 2 * t)
        else:
            weights[j] = 1.0
    return weights


def apply_fixed_emotion(neutral, emotion_vec, transition_frames):
    """Overlay one emotion vector across the whole clip, on top of lipsync."""
    total_frames = neutral.shape[0]
    vector_dim = neutral.shape[1] if neutral.ndim > 1 else 1
    vec = emotion_vec
    if len(vec) != vector_dim:
        vec = np.pad(vec, (0, vector_dim - len(vec)), 'constant') if len(vec) < vector_dim else vec[:vector_dim]

    w = create_transition_weights(total_frames, transition_frames)
    enhanced = neutral.copy()
    enhanced += vec[None, :] * w[:, None]      # broadcast emotion over all frames
    return np.clip(enhanced, 0.0, 1.0)


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Apply multiple fixed emotions to every neutral .npz in a folder.",
        allow_abbrev=False,   # avoids the --emotion / --emotion_vectors ambiguity
    )
    ap.add_argument('--in_dir', required=True, help="Folder of neutral .npz (or .npy) samples.")
    ap.add_argument('--out_dir', required=True, help="Where to write <sample>_<emotion>.npz.")
    ap.add_argument('--emotion_vectors', default='data/emotion_vectors.json',
                    help="JSON of per-emotion blendshape activations.")
    ap.add_argument('--emotions', nargs='+', default=['happy', 'sad', 'angry'],
                    help="Emotions to apply to each sample (names or JSON keys). Default: happy sad angry.")
    ap.add_argument('--transition_frames', type=int, default=0,
                    help="Smoothstep fade frames at clip start/end. 0 (default) = constant, "
                         "full-strength emotion. Negative -> fps/2.")
    ap.add_argument('--pattern', default='*.npz', help="Glob for input samples (default: *.npz).")
    ap.add_argument('--overwrite', action='store_true', help="Re-run even if the output exists.")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emotion_vectors = load_emotion_vectors(args.emotion_vectors)
    if not emotion_vectors:
        raise SystemExit("No emotion vectors loaded; JSON had no usable emotion keys.")

    # Resolve each requested emotion up front (fail fast on a bad name)
    resolved = {name: resolve_emotion(name, emotion_vectors) for name in args.emotions}
    print("Emotions:", ", ".join(f"{n}->{k}" for n, k in resolved.items()))

    transition_frames = int(FPS / 2) if args.transition_frames < 0 else args.transition_frames
    print(f"transition_frames = {transition_frames} ({'constant' if transition_frames == 0 else 'faded'})")

    samples = sorted(in_dir.glob(args.pattern))
    if not samples:
        raise SystemExit(f"No files matching '{args.pattern}' in {in_dir}")
    print(f"Found {len(samples)} samples -> {len(samples) * len(resolved)} outputs\n")

    n_ok, n_skip, n_fail = 0, 0, 0
    for path in samples:
        try:
            neutral, extras = load_neutral_from_file(path)
        except Exception as e:
            print(f"FAIL load {path.name}: {e}")
            n_fail += 1
            continue

        for name, key in resolved.items():
            out_path = out_dir / f"{path.stem}_{name}.npz"
            if out_path.exists() and not args.overwrite:
                print(f"  skip (exists): {out_path.name}")
                n_skip += 1
                continue

            enhanced = apply_fixed_emotion(neutral, emotion_vectors[key], transition_frames)

            meta = dict(extras)
            meta['num_frames'] = enhanced.shape[0]
            meta['fps'] = int(extras['fps']) if 'fps' in extras else FPS
            meta['emotion_source'] = 'fixed'
            meta['emotion_label'] = name
            meta['fixed_emotion'] = key
            np.savez(out_path, arkit_blendshapes=enhanced, **meta)
            print(f"  {path.stem:10s} + {name:6s} -> {out_path.name}  ({enhanced.shape[0]} fr)")
            n_ok += 1

    print(f"\nDone. {n_ok} written, {n_skip} skipped, {n_fail} failed. Output in {out_dir}")


if __name__ == '__main__':
    main()

'''

python add_emotion_inteli_v2.py \
    --in_dir  new_approach/results/A_A_samples_to_emote \
    --out_dir new_approach/results/A_A_emotions \
    --emotion_vectors data/emotion_vectors.json \
    --emotions happy sad angry \
    --transition_frames 0
    
'''