#!/usr/bin/env python3
"""
All-in-one: run BlendshapeDecoder inference from audio, then insert emotion
blendshape vectors on top of the lipsync sequence based on speech emotion
diarization.

Two entry points:
  1. Full pipeline (audio -> neutral blendshapes -> emotion-enhanced output):
       --audio speech.wav --model model.pth jaw mouth --output enhanced.npz
     Pass --model once per region-specialist; their region outputs are combined
     into one full-face 51-D neutral sequence before emotion is inserted.
  2. Legacy path (already have a neutral .npz/.npy, just add emotion):
       --blendshapes_path neutral.npz --audio_path speech.wav --output enhanced.npz

Two emotion sources (selectable via --emotion_source):
  - audio (default): SpeechBrain Speech_Emotion_Diarization on the waveform
  - text:            WhisperX transcript -> sentence-level text classifier

Optional: save the intermediate neutral sequence with --save_neutral for A/B comparison.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import argparse
import numpy as np
import librosa
import torch


sys.path.append(os.path.abspath('neutral_sequence'))
from inference_from_audio import (
    find_config_file,
    load_config,
    load_model_from_checkpoint,
    load_tokenizer,
    tokenize_audio,
    calculate_target_length,
)

sys.path.append(os.path.abspath('../speechbrain'))


# BlendshapeDecoder is needed for the checkpoint-embedded config path (the way
# the multi-decoder inference script loads its models). Import is tolerant of
# where model.py lives relative to this script.
try:
    from model import BlendshapeDecoder
except Exception:
    try:
        from neutral_sequence.model import BlendshapeDecoder
    except Exception:
        BlendshapeDecoder = None

# Fallback architecture config, matching the multi-decoder script's defaults.
# Used only when a checkpoint has no embedded 'model_config' and no YAML.
DEFAULT_MODEL_CONFIG = {
    'vocab_size': 6561,
    'd_model': 512,
    'nhead': 8,
    'num_layers': 6,
    'dim_feedforward': 2048,
    'dropout': 0.1,
    'num_blendshapes': 51,
}


# -----------------------------------------------------------------------------
# ARKit blendshape names (index order matches the model's output).
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

FPS = 30  # fixed for this project


# -----------------------------------------------------------------------------
# ARKit region definitions (index ranges match ARKIT_BLENDSHAPE_NAMES above).
# Used to combine several region-specialist BlendshapeDecoders into one full-face
# sequence, the same way the multi-model inference script does.
# -----------------------------------------------------------------------------
ARKIT_REGIONS = {
    'brow':      [0, 1, 2, 3, 4],
    'cheek':     [5, 6, 7],
    'eye_blink': [8, 9],
    'eye_look':  [10, 11, 12, 13, 14, 15, 16, 17],
    'eye_shape': [18, 19, 20, 21],
    'jaw':       [22, 23, 24, 25],
    'mouth':     list(range(26, 49)),
    'nose':      [49, 50],
}


def create_region_mask(regions):
    """Boolean (51,) mask selecting the blendshapes owned by `regions`."""
    mask = np.zeros(51, dtype=bool)
    for region in regions:
        if region not in ARKIT_REGIONS:
            print(f"  WARNING: unknown region '{region}' (ignored). "
                  f"Valid: {sorted(ARKIT_REGIONS)}")
            continue
        mask[ARKIT_REGIONS[region]] = True
    return mask


def report_region_coverage(model_configs):
    """Warn about blendshapes covered by >1 model (averaged) or by none (stay 0)."""
    coverage = np.zeros(51, dtype=int)
    for spec in model_configs:
        coverage += create_region_mask(spec['regions']).astype(int)
    overlapping = np.where(coverage > 1)[0]
    uncovered = np.where(coverage == 0)[0]
    print(f"  Coverage: {int((coverage == 1).sum())}/51 unique, "
          f"{len(overlapping)} overlapping, {len(uncovered)} uncovered")
    if len(overlapping):
        print(f"  NOTE: overlapping blendshapes (weighted-averaged): {overlapping.tolist()}")
    if len(uncovered):
        print(f"  NOTE: uncovered blendshapes (remain 0): {uncovered.tolist()}")


def load_decoder(ckpt, device, shared_config=None):
    """
    Resolve one BlendshapeDecoder. This is where the multi-model path differs
    from the original single-model path: instead of *requiring* a config YAML,
    config is resolved in this order:

      1. A YAML config -- shared --config, or one auto-found next to the
         checkpoint -- loaded via the existing load_model_from_checkpoint().
      2. Otherwise, the architecture config embedded INSIDE the checkpoint
         ('model_config'), exactly like the multi-decoder script, falling back
         to DEFAULT_MODEL_CONFIG. No YAML required.

    Different decoders can therefore use different (embedded) configs, which is
    how the two-decoder script handled jaw/mouth and upper-face models that were
    trained separately.

    Returns (model, epoch).
    """
    config_path = shared_config
    if config_path is None:
        try:
            config_path = find_config_file(ckpt)
        except FileNotFoundError:
            config_path = None

    # Path 1: a YAML config exists -> use the original loader unchanged.
    if config_path is not None:
        config = load_config(config_path)
        model, info = load_model_from_checkpoint(ckpt, config, device)
        return model, info.get('epoch', 'unknown')

    # Path 2: no YAML -> read architecture from the checkpoint itself.
    if BlendshapeDecoder is None:
        raise ImportError(
            f"No config YAML found for {ckpt}, and BlendshapeDecoder could not be "
            "imported for the checkpoint-embedded config path. Either pass "
            "--config, drop a config YAML next to the checkpoint, or make model.py "
            "importable.")
    checkpoint = torch.load(ckpt, map_location=device)
    model_config = checkpoint.get('model_config', DEFAULT_MODEL_CONFIG)
    model = BlendshapeDecoder(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('epoch', 'unknown')


# -----------------------------------------------------------------------------
# Emotion vector loading
# -----------------------------------------------------------------------------
def load_emotion_vectors(json_path):
    """
    Load per-emotion blendshape activations from JSON.
    Format: {"h": {"mouthSmileLeft": 0.7, ...}, "s": {...}, "a": {...}}
    Keys starting with "_" are ignored (for comments).
    """
    with open(json_path, 'r') as f:
        raw = json.load(f)

    vectors = {}
    for emotion_key, name_to_value in raw.items():
        if emotion_key.startswith('_'):
            continue
        if not isinstance(name_to_value, dict):
            raise ValueError(f"Emotion '{emotion_key}' must map to an object of "
                             f"blendshape-name -> value; got {type(name_to_value).__name__}")

        vec = np.zeros(len(ARKIT_BLENDSHAPE_NAMES), dtype=np.float32)
        for bs_name, value in name_to_value.items():
            if bs_name.startswith('_'):
                continue
            if bs_name not in NAME_TO_INDEX:
                raise KeyError(f"Unknown blendshape name '{bs_name}' for emotion "
                               f"'{emotion_key}'. Must be one of ARKIT_BLENDSHAPE_NAMES.")
            vec[NAME_TO_INDEX[bs_name]] = float(value)
        vectors[emotion_key] = vec
        nonzero = int(np.count_nonzero(vec))
        print(f"  loaded emotion '{emotion_key}': {nonzero} non-zero blendshapes, "
              f"max value {vec.max():.2f}")

    return vectors


# -----------------------------------------------------------------------------
# Neutral blendshape acquisition (two paths)
# -----------------------------------------------------------------------------
def generate_neutral_from_audio(args):
    """
    Run one or more region-specialist BlendshapeDecoders on the audio and combine
    their predictions by ARKit region into a single full-face neutral sequence.

    Each model in args._model_configs handles a subset of regions (e.g. a
    jaw/mouth specialist plus an upper-face specialist for brow/cheek/eyes/nose);
    only the blendshapes a model owns are taken from its prediction. Overlapping
    regions are weight-averaged, uncovered ones stay 0.

    Returns (blendshapes_np [T, 51], npz_metadata_dict).
    """
    device = torch.device(args.device)

    # Tokenize audio ONCE — every model consumes the same speech tokens.
    ort_session = load_tokenizer(args.tokenizer, device=args.device)
    speech_tokens = tokenize_audio(args.audio, ort_session).to(device)
    num_tokens = speech_tokens.shape[0]
    target_length = calculate_target_length(num_tokens, FPS, token_fps=args.token_fps)

    combined = None
    weight_sum = np.zeros(51, dtype=np.float64)
    per_model_meta = []

    for spec in args._model_configs:
        ckpt = spec['checkpoint_path']
        regions = spec['regions']
        weight = spec['weight']

        # Config resolved per checkpoint: shared --config, else a YAML next to
        # the checkpoint, else the architecture embedded in the checkpoint.
        model, epoch = load_decoder(ckpt, device, shared_config=args.config)

        print(f"Running BlendshapeDecoder prediction "
              f"({os.path.basename(ckpt)} -> regions {regions}, weight {weight})...")
        predicted = model.predict(speech_tokens, target_length=target_length)
        pred_np = predicted.cpu().numpy()  # (T, 51)

        if combined is None:
            combined = np.zeros((pred_np.shape[0], 51), dtype=np.float64)

        mask = create_region_mask(regions)              # (51,) bool
        combined += pred_np * mask[None, :] * weight     # keep only owned regions
        weight_sum += mask.astype(np.float64) * weight

        per_model_meta.append({
            'checkpoint': os.path.basename(ckpt),
            'regions': regions,
            'weight': weight,
            'epoch': epoch,
            'active_blendshapes': int(mask.sum()),
        })

    report_region_coverage(args._model_configs)

    weight_sum = np.maximum(weight_sum, 1e-8)
    combined = (combined / weight_sum[None, :]).astype(np.float32)
    print(f"Generated combined neutral blendshapes: {combined.shape}")

    utterance_id = os.path.splitext(os.path.basename(args.audio))[0]
    metadata = {
        'fps': FPS,
        'utterance_id': utterance_id,
        'selected_key': utterance_id,
        'num_tokens': num_tokens,
        'num_frames': combined.shape[0],
        'num_models': len(per_model_meta),
        'models_json': json.dumps(per_model_meta),  # plain string -> safe in .npz
    }
    return combined, metadata


def load_neutral_from_file(path):
    """Load a pre-computed neutral blendshape sequence from .npy or .npz."""
    print(f"Loading neutral blendshapes from {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npz':
        data = np.load(path, allow_pickle=True)
        if 'arkit_blendshapes' not in data.files:
            raise KeyError(f".npz missing 'arkit_blendshapes'. Keys: {data.files}")
        arr = data['arkit_blendshapes']
        extras = {k: data[k] for k in data.files if k != 'arkit_blendshapes'}
        print(f"  loaded 'arkit_blendshapes' + {len(extras)} metadata keys")
        return arr, extras
    elif ext == '.npy':
        return np.load(path), {}
    else:
        raise ValueError(f"Unsupported extension {ext}; expected .npy or .npz")


# -----------------------------------------------------------------------------
# Diarization sources
# -----------------------------------------------------------------------------
def diarize_with_audio_source(audio_path):
    """Use SpeechBrain's Speech_Emotion_Diarization on the audio waveform."""
    from speechbrain.inference.diarization import Speech_Emotion_Diarization
    print("Running speech emotion diarization (audio source: SpeechBrain WavLM)...")
    classifier = Speech_Emotion_Diarization.from_hparams(
        source="speechbrain/emotion-diarization-wavlm-large"
    )
    emotion_segments = classifier.diarize_file(audio_path)
    file_key = list(emotion_segments.keys())[0]
    return emotion_segments[file_key]


def diarize_with_text_source(audio_path, device, whisper_model_name, classifier_name, language):
    """Use WhisperX transcript + sentence-level text classifier."""
    from add_emotion_text import diarize_from_text
    print("Running speech emotion diarization (text source: WhisperX + text classifier)...")
    segments, sentences = diarize_from_text(
        audio_path,
        whisper_model_name=whisper_model_name,
        classifier_name=classifier_name,
        device=device,
        language=language,
    )
    # Brief per-sentence summary so the user can see what the text classifier produced.
    print(f"  classified {len(sentences)} sentences:")
    for i, s in enumerate(sentences):
        print(f"    [{i:2d}] {s['start']:6.2f}-{s['end']:6.2f}s  "
              f"{s['raw_label']:>10s} -> {s['emotion']}  (conf {s['score']:.2f})  "
              f"\"{s['text']}\"")
    return segments


def aggregate_segments(segments, window_sec, total_duration,
                       ignore_neutral=False, neutral_labels=("n", "neutral")):
    """Re-bin SED segments into fixed windows of `window_sec` seconds.

    Within each window the emotion covering the most time wins. This coarsens
    the SED's native fine-grained output toward utterance-level without touching
    SpeechBrain internals: it post-processes the {start, end, emotion} list.

    window_sec <= 0 or >= total_duration  -> a single window (utterance-level).
    ignore_neutral=True                   -> the dominant NON-neutral emotion
                                             wins each window (neutral only if no
                                             emotion is present), i.e. a constant-
                                             EMOTION baseline rather than neutral.
    """
    if window_sec is None or window_sec <= 0 or window_sec >= total_duration:
        bounds = [0.0, total_duration]
    else:
        bounds, t = [], 0.0
        while t < total_duration - 1e-9:
            bounds.append(round(t, 4)); t += window_sec
        bounds.append(round(total_duration, 4))

    windows = []
    for i in range(len(bounds) - 1):
        w0, w1 = bounds[i], bounds[i + 1]
        cover = {}
        for seg in segments:
            ov = max(0.0, min(seg["end"], w1) - max(seg["start"], w0))
            if ov > 0:
                cover[seg["emotion"]] = cover.get(seg["emotion"], 0.0) + ov
        if not cover:
            label = "n"
        elif ignore_neutral:
            non_neu = {e: v for e, v in cover.items() if e not in neutral_labels}
            label = max(non_neu, key=non_neu.get) if non_neu else max(cover, key=cover.get)
        else:
            label = max(cover, key=cover.get)
        windows.append({"start": w0, "end": w1, "emotion": label})

    # merge adjacent windows that ended up with the same emotion
    merged = [windows[0]]
    for w in windows[1:]:
        if w["emotion"] == merged[-1]["emotion"]:
            merged[-1]["end"] = w["end"]
        else:
            merged.append(w)
    return merged


# -----------------------------------------------------------------------------
# Emotion insertion
# -----------------------------------------------------------------------------
def fix_segment_overlaps(segments):
    segments = sorted(segments, key=lambda x: x["start"])
    fixed = []
    for i, seg in enumerate(segments):
        if i == 0:
            fixed.append(seg)
            continue
        prev = fixed[-1]
        if seg["start"] < prev["end"]:
            mid = (seg["start"] + prev["end"]) / 2
            print(f"  Overlap: prev ends {prev['end']:.2f}s, curr starts {seg['start']:.2f}s "
                  f"-> boundary at {mid:.2f}s")
            fixed[-1] = {"start": prev["start"], "end": mid, "emotion": prev["emotion"]}
            fixed.append({"start": mid, "end": seg["end"], "emotion": seg["emotion"]})
        else:
            fixed.append(seg)
    return fixed


def process_segment(segment, total_frames, emotion_vectors):
    start_sec, end_sec, emotion = segment["start"], segment["end"], segment["emotion"]
    start_frame = int(start_sec * FPS)
    end_frame = min(int(end_sec * FPS), total_frames)

    if end_frame <= start_frame:
        print(f"  Skipping zero-length segment {start_sec:.2f}-{end_sec:.2f}s ({emotion})")
        return None

    if emotion not in emotion_vectors:
        print(f"  {start_sec:.2f}-{end_sec:.2f}s  emotion={emotion}  (neutral, preserving lipsync)")
        return {'start_frame': start_frame, 'end_frame': end_frame,
                'emotion': emotion, 'blendshape': None, 'is_neutral': True}

    print(f"  {start_sec:.2f}-{end_sec:.2f}s  emotion={emotion}  frames {start_frame}-{end_frame}")
    return {'start_frame': start_frame, 'end_frame': end_frame,
            'emotion': emotion, 'blendshape': emotion_vectors[emotion], 'is_neutral': False}


def create_transition_weights(boundaries, total_frames, transition_frames):
    """Smoothstep fade-in/fade-out within each segment."""
    weights = np.zeros(total_frames)
    boundaries = sorted(set(boundaries))

    for i in range(len(boundaries) - 1):
        start_frame = boundaries[i]
        end_frame = boundaries[i + 1]
        seg_len = end_frame - start_frame
        if seg_len <= 0:
            continue

        actual_transition = min(transition_frames, seg_len // 4) if transition_frames > 0 else 0

        for j in range(start_frame, end_frame):
            d_start = j - start_frame
            d_end = end_frame - j
            if actual_transition == 0:
                weights[j] = 1.0
            elif d_start < actual_transition:
                t = d_start / actual_transition
                weights[j] = t * t * (3 - 2 * t)
            elif d_end < actual_transition:
                t = d_end / actual_transition
                weights[j] = t * t * (3 - 2 * t)
            else:
                weights[j] = 1.0
    return weights


def insert_emotions(neutral_blendshapes, segments, emotion_vectors, transition_frames):
    """
    Add emotion vectors on top of a neutral blendshape sequence.
    Takes pre-computed segments from whichever diarization source the caller chose.
    Returns (enhanced_blendshapes, segment_results, transition_weights, segments, boundaries).
    """
    total_frames = len(neutral_blendshapes)
    vector_dim = neutral_blendshapes.shape[1] if neutral_blendshapes.ndim > 1 else 1

    if vector_dim != 51:
        print(f"  WARNING: expected 51 blendshape dims, got {vector_dim}. "
              f"Emotion vectors will be truncated/padded to match.")

    print(f"  found {len(segments)} raw segments")
    print("Fixing overlaps...")
    segments = fix_segment_overlaps(segments)
    print(f"  {len(segments)} segments after overlap fix")

    # Process segments
    print("Processing segments:")
    segment_results = []
    frame_to_emotion = {}
    all_processed = set()
    for seg in segments:
        result = process_segment(seg, total_frames, emotion_vectors)
        if result is None:
            continue
        segment_results.append(result)
        all_processed.update(range(result['start_frame'], min(result['end_frame'], total_frames)))
        if not result['is_neutral']:
            vec = result['blendshape']
            if len(vec) != vector_dim:
                vec = np.pad(vec, (0, vector_dim - len(vec)), 'constant') if len(vec) < vector_dim else vec[:vector_dim]
            for f in range(result['start_frame'], min(result['end_frame'], total_frames)):
                frame_to_emotion[f] = vec

    missing = set(range(total_frames)) - all_processed
    if missing:
        print(f"  {len(missing)} frames not covered by any segment (will keep neutral lipsync)")
    print(f"  {len(frame_to_emotion)} frames will receive an emotion vector")

    # Transition weights
    boundaries = set()
    for r in segment_results:
        boundaries.add(r['start_frame'])
        boundaries.add(r['end_frame'])
    boundaries.add(0)
    boundaries.add(total_frames)
    transition_weights = create_transition_weights(boundaries, total_frames, transition_frames)

    # Apply
    print("Adding emotion vectors on top of neutral lipsync...")
    enhanced = neutral_blendshapes.copy()
    for f in range(total_frames):
        if f in frame_to_emotion:
            enhanced[f] = neutral_blendshapes[f] + frame_to_emotion[f] * transition_weights[f]
    enhanced = np.clip(enhanced, 0.0, 1.0)

    return enhanced, segment_results, transition_weights, segments, boundaries


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate emotion-enhanced ARKit blendshapes from audio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline, full-face (jaw/mouth + upper-face specialists), audio diarization:
  python add_emotion_v2.py \\
      --audio speech.wav \\
      --model checkpoints/jaw_mouth.pth jaw mouth 1.0 \\
      --model checkpoints/upper.pth eye_blink eye_look eye_shape brow cheek nose 1.0 \\
      --emotion_vectors emotion_vectors.json \\
      --output enhanced.npz

  # Same, but use text-based diarization (WhisperX + text classifier):
  python add_emotion_v2.py \\
      --audio speech.wav \\
      --model checkpoints/jaw_mouth.pth jaw mouth \\
      --model checkpoints/upper.pth eye_blink eye_look eye_shape brow cheek nose \\
      --emotion_vectors emotion_vectors.json --output enhanced.npz \\
      --emotion_source text

  # Mouth-only (single specialist), with intermediate neutral saved for A/B:
  python add_emotion_v2.py \\
      --audio speech.wav --model checkpoints/jaw_mouth.pth jaw mouth \\
      --output enhanced.npz --save_neutral neutral.npz

  # Legacy: reuse an existing neutral sequence, with text-based emotion:
  python add_emotion_v2.py \\
      --blendshapes_path neutral.npz --audio_path speech.wav \\
      --emotion_vectors emotion_vectors.json --output enhanced.npz \\
      --emotion_source text
""",
    )

    # Full pipeline inputs
    full = parser.add_argument_group("Full pipeline (audio -> neutral -> enhanced)")
    full.add_argument('--audio', type=str, default=None,
                      help="Path to .wav audio file for inference + diarization.")
    full.add_argument('--model', action='append', nargs='+', default=None,
                      metavar=('CHECKPOINT', 'REGION'),
                      help="Region-specialist model: CHECKPOINT REGION [REGION ...] [WEIGHT]. "
                           "Regions: jaw, mouth, brow, cheek, nose, eye_blink, eye_look, "
                           "eye_shape. Weight optional (default 1.0). Repeat --model per "
                           "specialist, e.g. --model jaw_mouth.pth jaw mouth "
                           "--model upper.pth brow cheek nose eye_blink eye_look eye_shape.")
    full.add_argument('--config', type=str, default=None,
                      help="Config YAML applied to all models. Auto-detected per "
                           "checkpoint dir if omitted.")
    full.add_argument('--tokenizer', type=str,
                      default='neutral_sequence/pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx',
                      help="CosyVoice2 ONNX tokenizer path.")
    full.add_argument('--token_fps', type=int, default=25,
                      help="Token rate in Hz (default 25 for CosyVoice2).")
    full.add_argument('--device', type=str,
                      default='cuda' if torch.cuda.is_available() else 'cpu',
                      choices=['cuda', 'cpu'])
    full.add_argument('--save_neutral', type=str, default=None,
                      help="Optional: save the neutral (pre-emotion) blendshapes to this path "
                           "for A/B comparison.")

    # Legacy path
    legacy = parser.add_argument_group("Legacy path (reuse existing neutral blendshapes)")
    legacy.add_argument('--blendshapes_path', type=str, default=None,
                        help="Path to pre-computed neutral .npy or .npz. "
                             "If provided, skip inference and just add emotion.")
    legacy.add_argument('--audio_path', type=str, default=None,
                        help="Audio file used for diarization (legacy mode).")

    # Emotion-source selection + text-source params
    src = parser.add_argument_group("Emotion source")
    src.add_argument('--emotion_source', type=str, default='audio',
                     choices=['audio', 'text'],
                     help="How to detect emotion. 'audio' = SpeechBrain WavLM diarizer "
                          "(default). 'text' = WhisperX transcript + sentence classifier.")
    src.add_argument('--sed_window', type=float, default=None,
                     help="If set, re-aggregate SED segments into fixed windows of "
                          "this many seconds (dominant emotion per window). Use 0 "
                          "(or a value >= clip length) for one utterance-level label. "
                          "Omit to keep the SED's native fine-grained segments.")
    src.add_argument('--sed_window_ignore_neutral', action='store_true',
                     help="When aggregating, let the dominant NON-neutral emotion win "
                          "each window (neutral only if no emotion present). Builds a "
                          "constant-EMOTION baseline instead of a constant-neutral one.")
    src.add_argument('--whisper_model', type=str, default='large-v3',
                     help="WhisperX model size (only used when --emotion_source=text).")
    src.add_argument('--text_classifier', type=str,
                     default='j-hartmann/emotion-english-distilroberta-base',
                     help="HF text-classification model (only used when --emotion_source=text).")
    src.add_argument('--language', type=str, default=None,
                     help="Force language code for WhisperX (e.g. 'en'). Default: auto-detect.")

    # Common
    common = parser.add_argument_group("Emotion insertion (both modes)")
    common.add_argument('--emotion_vectors', type=str, default='emotion_vectors.json',
                        help="JSON file with per-emotion blendshape activations.")
    common.add_argument('--output', type=str, required=True,
                        help="Path to save emotion-enhanced output.")
    common.add_argument('--transition_frames', type=int, default=-1,
                        help="Frames for smoothstep fade on each side of an emotion segment. "
                             "Default: -1 -> fps/2 (half a second). Use 0 for hard switches.")
    common.add_argument('--visualize_transitions', action='store_true',
                        help="Save a plot of transition weights.")

    args = parser.parse_args()

    # Route: either (audio + model) or (blendshapes_path + audio_path) must be set.
    full_mode = bool(args.audio and args.model)
    legacy_mode = bool(args.blendshapes_path and args.audio_path)

    if full_mode and legacy_mode:
        parser.error("Choose either full pipeline (--audio + --model) OR "
                     "legacy mode (--blendshapes_path + --audio_path), not both.")
    if not full_mode and not legacy_mode:
        parser.error("Must provide either (--audio + --model) for full pipeline "
                     "OR (--blendshapes_path + --audio_path) for legacy mode.")

    # Parse each --model spec into {checkpoint_path, regions, weight}.
    if full_mode:
        model_configs = []
        for spec in args.model:
            ckpt = spec[0]
            regions, weight = [], 1.0
            for item in spec[1:]:
                try:
                    weight = float(item)        # trailing numeric token = weight
                except ValueError:
                    regions.append(item)
            if not regions:
                parser.error(f"No regions specified for model: {ckpt}")
            model_configs.append({'checkpoint_path': ckpt,
                                  'regions': regions, 'weight': weight})
        args._model_configs = model_configs

    args._full_mode = full_mode
    return args


def main():
    args = parse_arguments()

    # Load emotion vectors
    print(f"Loading emotion vectors from {args.emotion_vectors}")
    emotion_vectors = load_emotion_vectors(args.emotion_vectors)
    if not emotion_vectors:
        raise ValueError("No emotion vectors loaded; JSON had no usable emotion keys.")

    # Get neutral blendshapes — either by running inference or by loading from file
    if args._full_mode:
        print("\n=== Full pipeline: running BlendshapeDecoder inference ===")
        neutral_blendshapes, neutral_metadata = generate_neutral_from_audio(args)
        audio_for_diar = args.audio
    else:
        print("\n=== Legacy mode: loading pre-computed neutral blendshapes ===")
        neutral_blendshapes, neutral_metadata = load_neutral_from_file(args.blendshapes_path)
        audio_for_diar = args.audio_path

    # Optional: save neutral intermediate
    if args.save_neutral:
        print(f"\nSaving neutral (pre-emotion) blendshapes to {args.save_neutral}")
        save_ext = os.path.splitext(args.save_neutral)[1].lower()
        if save_ext == '.npz':
            np.savez(args.save_neutral, arkit_blendshapes=neutral_blendshapes, **neutral_metadata)
        else:
            np.save(args.save_neutral, neutral_blendshapes)

    # Sanity-check duration
    audio, sr = librosa.load(audio_for_diar, sr=16000)
    audio_duration = len(audio) / sr
    animation_duration = len(neutral_blendshapes) / FPS
    print(f"\nAudio: {audio_duration:.2f}s | Neutral animation: {animation_duration:.2f}s @ {FPS} fps")
    if abs(audio_duration - animation_duration) > 1.0:
        print(f"  WARNING: audio/animation duration mismatch > 1s")

    # Resolve transition_frames
    transition_frames = int(FPS / 2) if args.transition_frames < 0 else args.transition_frames
    print(f"Using transition_frames = {transition_frames}")

    # Diarize using the selected source
    print(f"\n=== Diarization (source: {args.emotion_source}) ===")
    if args.emotion_source == 'audio':
        segments = diarize_with_audio_source(audio_for_diar)
        if args.sed_window is not None:
            print(f"Re-aggregating SED into {args.sed_window}s windows "
                f"(ignore_neutral={args.sed_window_ignore_neutral})...")
            segments = aggregate_segments(segments, args.sed_window, audio_duration,
                                        ignore_neutral=args.sed_window_ignore_neutral)
            for s in segments:
                print(f"  {s['start']:.2f}-{s['end']:.2f}s -> {s['emotion']}")
    else:
        segments = diarize_with_text_source(
            audio_for_diar,
            device=args.device,
            whisper_model_name=args.whisper_model,
            classifier_name=args.text_classifier,
            language=args.language,
        )

    # Insert emotions
    print("\n=== Inserting emotions ===")
    enhanced, segment_results, transition_weights, segments, boundaries = insert_emotions(
        neutral_blendshapes, segments, emotion_vectors, transition_frames
    )

    # Save enhanced output
    output_ext = os.path.splitext(args.output)[1].lower()
    print(f"\nSaving enhanced blendshapes to {args.output}")
    if output_ext == '.npz':
        # Preserve (and potentially update) metadata
        out_metadata = dict(neutral_metadata)
        out_metadata['num_frames'] = enhanced.shape[0]
        np.savez(args.output, arkit_blendshapes=enhanced, **out_metadata)
    else:
        np.save(args.output, enhanced)

    # Mapping JSON
    mapping_file = os.path.splitext(args.output)[0] + '_mapping.json'
    mapping_data = {
        'fps': FPS,
        'total_frames': len(neutral_blendshapes),
        'audio_duration': audio_duration,
        'transition_frames': transition_frames,
        'emotion_source': args.emotion_source,
        'modified_frames': int(np.count_nonzero(transition_weights > 0)),
        'segments': [
            {
                'start_time': seg['start'],
                'end_time': seg['end'],
                'start_frame': res['start_frame'],
                'end_frame': res['end_frame'],
                'emotion': res['emotion'],
                'is_neutral': res['is_neutral'],
            }
            for seg, res in zip(segments, segment_results)
        ],
    }
    with open(mapping_file, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    print(f"Saved mapping to {mapping_file}")

    # Optional transition plot
    if args.visualize_transitions:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.plot(transition_weights)
            plt.title('Transition Weights')
            plt.xlabel('Frame')
            plt.ylabel('Weight')
            plt.grid(True)
            for b in sorted(boundaries):
                plt.axvline(x=b, color='r', linestyle='--', alpha=0.5)
            viz_file = os.path.splitext(args.output)[0] + '_transitions.png'
            plt.savefig(viz_file)
            plt.close()
            print(f"Saved transition visualization to {viz_file}")
        except ImportError:
            print("matplotlib not available, skipping visualization")

    print("\nDone.")


if __name__ == "__main__":
    main()


''' 

# Constant emotion with large diarization aggregation
python add_emotion_v4.py \
  --audio data/id02_p01_e02_f45.wav \
  --model neutral_sequence/checkpoints/mouth_jaw_1000ep_4bs_no_overlap/jaw_mouth_model_epoch_999.pth jaw mouth 1.0 \
  --model neutral_sequence/checkpoints/cben_1000ep_32bs_no_overlap/eye_blink_eye_look_eye_shape_brow_cheek_nose_model_epoch_999.pth eye_blink eye_look eye_shape brow cheek nose 1.0 \
  --emotion_vectors data/emotion_vectors.json \
  --output results/emotional_10s/id02_p01_e02_f45_constant.npz \
  --sed_window 0 \
  --sed_window_ignore_neutral


#Full-face (jaw/mouth + upper-face), audio-based diarization:
python add_emotion_v4.py \
  --audio data/id02_p01_e02_f45.wav \
  --model neutral_sequence/checkpoints/mouth_jaw_1000ep_4bs_no_overlap/jaw_mouth_model_epoch_999.pth jaw mouth 1.0 \
  --model neutral_sequence/checkpoints/cben_1000ep_32bs_no_overlap/eye_blink_eye_look_eye_shape_brow_cheek_nose_model_epoch_999.pth eye_blink eye_look eye_shape brow cheek nose 1.0 \
  --emotion_vectors data/emotion_vectors.json \
  --output results/emotional_10s/id02_p01_e02_f45.npz 

'''
