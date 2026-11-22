#!/usr/bin/env python3
"""
Inference script for the BlendshapeDecoder model.

Generates ARKit blendshape sequences directly from .wav audio files.
Handles CosyVoice2 speech tokenization internally.
"""
import argparse
import yaml
import torch
import numpy as np
import torchaudio
import whisper
import onnxruntime
from pathlib import Path
import logging

# Import the model definition from your model.py file
from model import BlendshapeDecoder, BlendshapeEncoderDecoder, BlendshapeDecoderConv1D

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ─────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────

def load_config(config_path):
    """Load configuration from YAML file"""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def find_config_file(checkpoint_path):
    """Find the config file associated with a checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_dir = checkpoint_path.parent

    config_candidates = [
        checkpoint_dir / 'training_config2.yaml',
        checkpoint_dir / 'config.yaml',
        checkpoint_dir / 'config2.yaml',
        checkpoint_dir / 'config7.yaml',
    ]

    for config_path in config_candidates:
        if config_path.exists():
            logging.info(f"Found config file: {config_path}")
            return config_path

    raise FileNotFoundError(
        f"Could not find config file in {checkpoint_dir}. "
        f"Tried: {[c.name for c in config_candidates]}"
    )


# ─────────────────────────────────────────────────────────────────────
# CosyVoice2 tokenization
# ─────────────────────────────────────────────────────────────────────

def load_tokenizer(onnx_path, device='cpu'):
    """
    Load the CosyVoice2 ONNX speech tokenizer.

    Args:
        onnx_path: Path to speech_tokenizer_v2.onnx
        device: 'cuda' or 'cpu'

    Returns:
        onnxruntime.InferenceSession
    """
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX tokenizer not found: {onnx_path}")

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1

    providers = ["CUDAExecutionProvider"] if device == 'cuda' else ["CPUExecutionProvider"]

    session = onnxruntime.InferenceSession(
        str(onnx_path), sess_options=option, providers=providers
    )
    logging.info(f"Loaded ONNX tokenizer from: {onnx_path}")
    return session


def tokenize_audio(wav_path, ort_session, max_duration=30.0):
    """
    Extract CosyVoice2 speech tokens from a .wav file.

    Pipeline: load wav → resample to 16 kHz → mono → 128-mel log-mel
              spectrogram (whisper) → ONNX tokenizer → token indices.

    Args:
        wav_path: Path to .wav file.
        ort_session: Loaded ONNX inference session.
        max_duration: Maximum audio duration in seconds (default 30 s).

    Returns:
        torch.LongTensor of shape (num_tokens,)
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    # Load and preprocess audio
    audio, sample_rate = torchaudio.load(str(wav_path))

    if sample_rate != 16000:
        logging.info(f"Resampling from {sample_rate} Hz to 16000 Hz")
        audio = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )(audio)
        sample_rate = 16000

    # Convert to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    duration = audio.shape[1] / sample_rate
    logging.info(f"Audio duration: {duration:.2f} s")

    if duration > max_duration:
        logging.warning(
            f"Audio is {duration:.1f}s, which exceeds {max_duration}s. "
            f"Truncating to {max_duration}s."
        )
        max_samples = int(max_duration * sample_rate)
        audio = audio[:, :max_samples]
        duration = max_duration

    # Compute 128-mel log-mel spectrogram (same as CosyVoice2 pipeline)
    feat = whisper.log_mel_spectrogram(audio, n_mels=128)

    # Run ONNX tokenizer
    input_names = [inp.name for inp in ort_session.get_inputs()]
    feed = {
        input_names[0]: feat.detach().cpu().numpy(),
        input_names[1]: np.array([feat.shape[2]], dtype=np.int32),
    }
    tokens = ort_session.run(None, feed)[0].flatten()

    speech_tokens = torch.tensor(tokens, dtype=torch.long)
    logging.info(f"Extracted {len(speech_tokens)} speech tokens ({len(speech_tokens)/25:.2f}s at 25 Hz)")

    return speech_tokens


# ─────────────────────────────────────────────────────────────────────
# Legacy token-file loader (kept for backward compatibility)
# ─────────────────────────────────────────────────────────────────────

def load_speech_tokens(tokens_path, utt_id=None, chunk_id=None):
    """Load speech tokens from a pre-extracted .pt file."""
    logging.info(f"Loading speech tokens from: {tokens_path}")
    tokens_path = Path(tokens_path)

    if not tokens_path.exists():
        raise FileNotFoundError(f"Token file not found: {tokens_path}")

    tokens_data = torch.load(tokens_path, map_location='cpu')

    if not isinstance(tokens_data, dict):
        raise TypeError(f"Expected dict, got {type(tokens_data)}")

    # Key selection logic (unchanged)
    selected_key = None
    if utt_id and chunk_id is not None:
        selected_key = f"{utt_id}_chunk_{chunk_id}"
        if selected_key not in tokens_data:
            raise KeyError(f"Key '{selected_key}' not found.")
    elif utt_id:
        if utt_id in tokens_data:
            selected_key = utt_id
        else:
            matching = [k for k in tokens_data if k.startswith(f"{utt_id}_chunk_")]
            if matching:
                selected_key = matching[0]
                logging.warning(f"Using chunk: '{selected_key}'")
            else:
                raise KeyError(f"'{utt_id}' not found. Available: {list(tokens_data.keys())[:10]}")
    else:
        selected_key = next(iter(tokens_data))
        logging.warning(f"No --utt_id provided. Using: '{selected_key}'")

    speech_tokens = tokens_data[selected_key]
    if isinstance(speech_tokens, list):
        speech_tokens = torch.tensor(speech_tokens, dtype=torch.long)
    speech_tokens = speech_tokens.long()

    metadata = {
        'key': selected_key,
        'num_tokens': speech_tokens.shape[0],
        'utterance_id': selected_key.split('_chunk_')[0] if '_chunk_' in selected_key else selected_key,
        'is_chunked': '_chunk_' in selected_key,
    }
    logging.info(f"Loaded {metadata['num_tokens']} tokens from '{selected_key}'")
    return speech_tokens, selected_key, metadata


# ─────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────

def load_model_from_checkpoint(checkpoint_path, config, device):
    """Load model and weights from a checkpoint file."""
    logging.info("Initializing model from configuration...")

    model_config = config.get('training', config)

    model = BlendshapeDecoder(
        vocab_size=model_config.get('vocab_size', 6561),
        d_model=model_config.get('d_model', 512),
        nhead=model_config.get('nhead', 8),
        num_layers=model_config.get('num_layers', 6),
        dim_feedforward=model_config.get('dim_feedforward', 2048),
        dropout=model_config.get('dropout', 0.1),
    )

    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'best_val_loss': checkpoint.get('best_val_loss', 'unknown'),
        'global_step': checkpoint.get('global_step', 'unknown'),
    }
    logging.info(f"Model loaded (epoch {info['epoch']}, val_loss {info['best_val_loss']})")
    return model, info


# ─────────────────────────────────────────────────────────────────────
# Target-length calculation
# ─────────────────────────────────────────────────────────────────────

def calculate_target_length(num_tokens, target_fps, token_fps=25):
    """Calculate output frame count from token count."""
    target_length = int(num_tokens * (target_fps / token_fps))
    logging.info(
        f"Input: {num_tokens} tokens @ {token_fps} Hz → "
        f"Output: {target_length} frames @ {target_fps} FPS "
        f"({num_tokens / token_fps:.2f}s)"
    )
    return target_length


# ─────────────────────────────────────────────────────────────────────
# Main inference
# ─────────────────────────────────────────────────────────────────────

def run_inference(args):
    """Main inference pipeline."""
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # ── Config ──
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = find_config_file(checkpoint_path)
    config = load_config(config_path)

    # ── Model ──
    model, checkpoint_info = load_model_from_checkpoint(checkpoint_path, config, device)

    # ── Tokens ──
    if args.audio:
        # ===== NEW: tokenize directly from audio =====
        ort_session = load_tokenizer(args.tokenizer, device=args.device)
        speech_tokens = tokenize_audio(args.audio, ort_session)
        utterance_id = Path(args.audio).stem
        selected_key = utterance_id
        num_tokens = speech_tokens.shape[0]
    elif args.tokens:
        # Legacy path: load pre-extracted tokens
        speech_tokens, selected_key, meta = load_speech_tokens(
            args.tokens, utt_id=args.utt_id, chunk_id=args.chunk_id
        )
        utterance_id = meta['utterance_id']
        num_tokens = meta['num_tokens']
    else:
        raise ValueError("Provide either --audio (wav file) or --tokens (pre-extracted .pt)")

    speech_tokens = speech_tokens.to(device)

    # ── Target length ──
    target_length = calculate_target_length(num_tokens, args.target_fps, token_fps=args.token_fps)

    # ── Predict ──
    logging.info("Running model prediction...")
    predicted = model.predict(speech_tokens, target_length=target_length)
    predicted_np = predicted.cpu().numpy()

    logging.info(f"Generated blendshapes: {predicted_np.shape} (frames × blendshapes)")

    # ── Save ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'arkit_blendshapes': predicted_np,
        'fps': args.target_fps,
        'utterance_id': utterance_id,
        'selected_key': selected_key,
        'num_tokens': num_tokens,
        'num_frames': predicted_np.shape[0],
        'checkpoint_epoch': checkpoint_info['epoch'],
        'mask_categories': config.get('training', config).get('mask_categories', ['jaw', 'mouth']),
    }

    np.savez(output_path, **output_data)
    logging.info(f"Saved to: {output_path}")


# ─────────────────────────────────────────────────────────────────────
# Utility: list utterances in a token file
# ─────────────────────────────────────────────────────────────────────

def list_available_utterances(tokens_path, limit=20):
    """List available utterances in a token file."""
    tokens_data = torch.load(tokens_path, map_location='cpu')
    if not isinstance(tokens_data, dict):
        logging.error("Token file is not a dictionary")
        return

    all_keys = list(tokens_data.keys())
    print("\n" + "=" * 70)
    print(f"Available Utterances in {Path(tokens_path).name}")
    print("=" * 70)
    print(f"Total: {len(all_keys)}")

    groups = {}
    for key in all_keys:
        base = key.split('_chunk_')[0] if '_chunk_' in key else key
        groups.setdefault(base, []).append(key)

    for i, (base_id, keys) in enumerate(list(groups.items())[:limit]):
        n = sum(len(tokens_data[k]) for k in keys)
        chunks = f"{len(keys)} chunks, " if len(keys) > 1 else ""
        print(f"  {base_id}: {chunks}{n} tokens")
    if len(groups) > limit:
        print(f"  ... and {len(groups) - limit} more")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ARKit blendshapes from audio or speech tokens.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From .wav audio (recommended):
  python inference_from_audio.py \\
      --audio my_recording.wav \\
      --tokenizer pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx \\
      --checkpoint checkpoints/best_model.pth \\
      --output results/my_recording.npz

  # From pre-extracted tokens (legacy):
  python inference_from_audio.py \\
      --tokens utt2speech_token.pt \\
      --utt_id 2_scott_0_1_1 \\
      --checkpoint checkpoints/best_model.pth \\
      --output results/output.npz

  # List available utterances in a token file:
  python inference_from_audio.py \\
      --tokens utt2speech_token.pt \\
      --list_utterances
""",
    )

    # ── Input (mutually preferred: --audio or --tokens) ──
    input_group = parser.add_argument_group("Input (choose one)")
    input_group.add_argument(
        '--audio', type=str, default=None,
        help="Path to .wav audio file. Requires --tokenizer."
    )
    input_group.add_argument(
        '--tokenizer', type=str,
        default='../pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx',
        help="Path to CosyVoice2 ONNX tokenizer (default: pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx)."
    )
    input_group.add_argument(
        '--tokens', type=str, default=None,
        help="Path to pre-extracted speech tokens (.pt file). Legacy mode."
    )
    input_group.add_argument(
        '--utt_id', type=str, default=None,
        help="Utterance ID (for --tokens mode)."
    )
    input_group.add_argument(
        '--chunk_id', type=int, default=None,
        help="Chunk ID (for --tokens mode)."
    )

    # ── Model ──
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        '--checkpoint', type=str,
        help="Path to trained model checkpoint (.pth)."
    )
    model_group.add_argument(
        '--config', type=str, default=None,
        help="Path to config YAML. Auto-detected from checkpoint dir if omitted."
    )

    # ── Output ──
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        '--output', type=str,
        help="Path to save output blendshapes (.npz)."
    )
    output_group.add_argument(
        '--target_fps', type=int, default=30,
        help="Output animation FPS (default: 30)."
    )
    output_group.add_argument(
        '--token_fps', type=int, default=25,
        help="Token rate in Hz (default: 25 for CosyVoice2)."
    )

    # ── Misc ──
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help="Device for inference."
    )
    parser.add_argument(
        '--list_utterances', action='store_true',
        help="List utterances in --tokens file and exit."
    )

    args = parser.parse_args()

    if args.list_utterances:
        if not args.tokens:
            parser.error("--tokens is required for --list_utterances")
        list_available_utterances(args.tokens)
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required for inference")
        if not args.output:
            parser.error("--output is required for inference")
        if not args.audio and not args.tokens:
            parser.error("Provide either --audio or --tokens")
        run_inference(args)


'''
python inference_from_audio.py \
    --audio ref_audios/three_emotions.wav \
    --checkpoint checkpoints/mouth_jaw_1000ep_4bs_no_overlap/jaw_mouth_model_epoch_999.pth \
    --output results/three_emotions.npz

'''