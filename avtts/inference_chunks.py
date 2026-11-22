#!/usr/bin/env python3
"""
Text-to-Speech + Facial Animation (TTSF) Inference Pipeline
=============================================================

Synthesize speech from input text using CosyVoice2 zero-shot cloning,
then predict ARKit blendshape sequences with trained decoder models.

Long texts are automatically split into 10s chunks for stable synthesis.

Inputs:
  --text              Text to synthesize (or --text_file for long texts)
  --ref_audio         Reference audio for voice cloning (.wav)
  --ref_text          Transcription of the reference audio
  --model             Blendshape decoder checkpoint(s) + region specs

Outputs per chunk:
  audio/{prefix}_chunk_{N}.wav
  blendshapes/{prefix}_chunk_{N}.npz
  tokens/{prefix}_chunk_{N}_tokens.pt
  metadata/{prefix}_chunk_{N}.json

Usage:
    python ttsf_inference.py \\
        --text "Hello, this is a test of the speech synthesis system." \\
        --ref_audio reference.wav \\
        --ref_text "This is the transcription of the reference audio." \\
        --model checkpoints/jaw_mouth.pth jaw mouth 1.0 \\
        --model checkpoints/rest.pth eye_blink eye_look eye_shape brow cheek nose 1.0 \\
        --output_dir results/ttsf_output/

    # Or with a text file:
    python ttsf_inference.py \\
        --text_file my_script.txt \\
        --ref_audio reference.wav \\
        --ref_text "Reference transcription." \\
        --model checkpoints/jaw_mouth.pth jaw mouth 1.0 \\
        --output_dir results/ttsf_output/
"""

import sys
from unittest.mock import MagicMock

# Mock wetext for offline mode
class MockNormalizer:
    def __init__(self, *args, **kwargs): pass
    def normalize(self, text): return text

wetext_mock = MagicMock()
wetext_mock.Normalizer = MockNormalizer
sys.modules['wetext'] = wetext_mock

import os
import re
import json
import logging
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torchaudio

sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

from model import BlendshapeDecoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# ARKit blendshape region definitions
# =============================================================================

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


# =============================================================================
# Blendshape model helpers
# =============================================================================

def create_region_mask(regions: List[str]) -> np.ndarray:
    mask = np.zeros(51, dtype=bool)
    for region in regions:
        if region in ARKIT_REGIONS:
            mask[ARKIT_REGIONS[region]] = True
        else:
            logger.warning(f"Unknown region: {region}")
    return mask


def load_blendshape_models(model_configs: List[Dict], device: str) -> List[Dict]:
    """Load all blendshape decoder models."""
    models = []
    for i, config in enumerate(model_configs):
        checkpoint_path = config['checkpoint_path']
        regions = config.get('regions', [])
        weight = config.get('weight', 1.0)

        logger.info(f"  Loading model {i+1}: {Path(checkpoint_path).name}  "
                     f"regions={regions}  weight={weight}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_cfg = checkpoint.get('model_config', {
            'vocab_size': 6561, 'd_model': 512, 'nhead': 8,
            'num_layers': 6, 'dim_feedforward': 2048,
            'dropout': 0.1, 'num_blendshapes': 51
        })

        model = BlendshapeDecoder(**model_cfg).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        mask = create_region_mask(regions)
        logger.info(f"    ✓ Loaded (epoch {checkpoint.get('epoch', '?')})  "
                     f"active blendshapes: {mask.sum()}/51")

        models.append({
            'model': model,
            'regions': regions,
            'weight': weight,
            'mask': mask,
            'checkpoint_path': checkpoint_path,
        })
    return models


def predict_blendshapes(
    models: List[Dict],
    speech_tokens: torch.Tensor,
    audio_duration: float,
    fps: int = 30,
    device: str = "cuda",
) -> torch.Tensor:
    """Run all blendshape models on speech tokens and combine by region mask."""
    num_frames = int(audio_duration * fps)

    if speech_tokens.dim() == 1:
        speech_tokens = speech_tokens.unsqueeze(0)
    speech_tokens = speech_tokens.to(device)

    all_preds = []
    for info in models:
        with torch.no_grad():
            pred = info['model'](speech_tokens, target_length=num_frames)
        all_preds.append({
            'prediction': pred.squeeze(0),
            'mask':       info['mask'],
            'weight':     info['weight'],
        })

    combined   = torch.zeros((num_frames, 51), device=device)
    weight_sum = np.zeros(51)

    for p in all_preds:
        mask_t = torch.from_numpy(p['mask']).float().to(device)
        combined   += p['prediction'] * mask_t * p['weight']
        weight_sum += p['mask'].astype(float) * p['weight']

    weight_sum = np.maximum(weight_sum, 1e-8)
    combined   = combined / torch.from_numpy(weight_sum).float().to(device)

    return combined


# =============================================================================
# Text chunking utilities
# =============================================================================

def split_into_sentences(text: str, max_words: int = 20) -> List[str]:
    """Split text into short sub-chunks at punctuation boundaries."""
    sentences = re.split(r'(?<=[.!?;:])\s+', text.strip())
    chunks = []
    for sent in sentences:
        words_list = sent.split()
        if len(words_list) <= max_words:
            chunks.append(sent)
        else:
            while len(words_list) > max_words:
                break_at = max_words
                for i in range(max_words - 1, max_words // 2, -1):
                    if words_list[i].lower() in {'and', 'but', 'or', 'so', 'because',
                                                  'when', 'if', 'that', 'which', 'the'}:
                        break_at = i
                        break
                chunks.append(" ".join(words_list[:break_at]))
                words_list = words_list[break_at:]
            if words_list:
                chunks.append(" ".join(words_list))
    return chunks


def estimate_words_per_chunk(text: str, chunk_duration: float, speaking_rate_wpm: float = 150.0) -> int:
    """
    Estimate how many words fit in one chunk based on speaking rate.
    Default speaking rate: ~150 words per minute (2.5 words/sec).
    """
    words_per_sec = speaking_rate_wpm / 60.0
    return int(words_per_sec * chunk_duration)


def split_text_into_chunks(text: str, chunk_duration: float = 10.0, speaking_rate_wpm: float = 150.0) -> List[str]:
    """
    Split input text into chunks that roughly correspond to chunk_duration seconds of speech.

    Uses sentence boundaries when possible, otherwise splits at word boundaries.
    """
    words = text.split()
    total_words = len(words)
    words_per_chunk = estimate_words_per_chunk(text, chunk_duration, speaking_rate_wpm)

    if total_words <= words_per_chunk:
        return [text]

    # First split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_chunk_words = []
    current_word_count = 0

    for sentence in sentences:
        sent_words = sentence.split()
        sent_len = len(sent_words)

        # If adding this sentence exceeds the limit, finalize current chunk
        if current_word_count + sent_len > words_per_chunk and current_chunk_words:
            chunks.append(" ".join(current_chunk_words))
            current_chunk_words = []
            current_word_count = 0

        # If a single sentence is longer than the limit, split it
        if sent_len > words_per_chunk:
            # Flush current chunk first
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
                current_chunk_words = []
                current_word_count = 0

            # Split the long sentence
            remaining = sent_words
            while len(remaining) > words_per_chunk:
                chunks.append(" ".join(remaining[:words_per_chunk]))
                remaining = remaining[words_per_chunk:]
            if remaining:
                current_chunk_words = remaining
                current_word_count = len(remaining)
        else:
            current_chunk_words.extend(sent_words)
            current_word_count += sent_len

    # Don't forget the last chunk
    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    return chunks


# =============================================================================
# Main inference
# =============================================================================

'''def run_inference(
    text: str,
    ref_audio_path: str,
    ref_text: str,
    model_configs: List[Dict],
    output_dir: str = "results/ttsf_output/",
    output_prefix: str = "ttsf",
    cosyvoice_model_path: str = "../pretrained_models/CosyVoice2-0.5B",
    chunk_duration: float = 10.0,
    speaking_rate_wpm: float = 150.0,
    ref_max_duration: float = 10.0,
    fps: int = 30,
    device: str = "cuda",
    start_sec: float = None,
    end_sec: float = None,
):
    """
    Run TTS + blendshape inference on input text.

    Args:
        text:                Full text to synthesize
        ref_audio_path:      Path to reference audio for voice cloning
        ref_text:            Transcription of reference audio
        model_configs:       Blendshape model configurations
        output_dir:          Output directory
        output_prefix:       Prefix for output filenames
        cosyvoice_model_path: Path to CosyVoice2 model
        chunk_duration:      Target duration per chunk (seconds)
        speaking_rate_wpm:   Estimated speaking rate (words per minute)
        ref_max_duration:    Max duration of reference audio to use (seconds)
        fps:                 Blendshape frame rate
        device:              torch device
    """
    output_dir = Path(output_dir)

    # Create output dirs
    audio_dir = output_dir / "audio"
    blend_dir = output_dir / "blendshapes"
    token_dir = output_dir / "tokens"
    meta_dir  = output_dir / "metadata"
    for d in [audio_dir, blend_dir, token_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- Load CosyVoice2 ----
    logger.info(f"Loading CosyVoice2 from {cosyvoice_model_path} ...")
    os.environ['MODELSCOPE_CACHE'] = os.path.expanduser('~/.cache/modelscope')
    cosyvoice = CosyVoice2(cosyvoice_model_path, load_jit=False, load_trt=False, fp16=False)
    sample_rate = cosyvoice.sample_rate
    logger.info(f"  ✓ CosyVoice2 loaded (sample_rate={sample_rate})")

    # ---- Load blendshape models ----
    logger.info(f"Loading {len(model_configs)} blendshape model(s)...")
    models = load_blendshape_models(model_configs, device)

    # ---- Load reference audio ----
    logger.info(f"Loading reference audio: {ref_audio_path}")
    ref_speech = load_wav(ref_audio_path, 16000)
    ref_max_samples = int(ref_max_duration * 16000)
    if ref_speech.shape[1] > ref_max_samples:
        ref_speech = ref_speech[:, :ref_max_samples]
        logger.info(f"  Trimmed reference to {ref_max_duration}s")
    logger.info(f"  ✓ Reference audio: {ref_speech.shape[1]/16000:.2f}s")
    logger.info(f"  Prompt text: \"{ref_text[:80]}{'...' if len(ref_text)>80 else ''}\"")

    # ---- Split text into chunks ----
    text_chunks = split_text_into_chunks(text, chunk_duration, speaking_rate_wpm)
    logger.info(f"\nInput text: {len(text.split())} words → {len(text_chunks)} chunk(s)")
    for i, chunk in enumerate(text_chunks):
        logger.info(f"  Chunk {i}: \"{chunk[:80]}{'...' if len(chunk)>80 else ''}\" ({len(chunk.split())} words)")

    # ---- Process each chunk ----
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing {len(text_chunks)} chunks")
    logger.info(f"{'='*70}\n")

    all_audio = []
    all_blendshapes = []
    stats = {
        'total_chunks': len(text_chunks),
        'processed': 0,
        'total_audio_duration': 0.0,
        'total_frames': 0,
    }

    for chunk_idx, chunk_text in enumerate(text_chunks):
        chunk_name = f"{output_prefix}_chunk_{chunk_idx}"

        logger.info(f"\n--- Chunk {chunk_idx}/{len(text_chunks)-1} ---")
        logger.info(f"  Text: \"{chunk_text[:80]}{'...' if len(chunk_text)>80 else ''}\"")

        # Split into sub-sentences for CosyVoice2
        sub_chunks = split_into_sentences(chunk_text, max_words=20)

        audio_list, token_list = [], []
        t_synth = time.time()

        for sc in sub_chunks:
            generator = cosyvoice.inference_zero_shot(
                sc, ref_text, ref_speech, stream=False
            )
            for result in generator:
                if 'tts_speech' in result:
                    audio_list.append(result['tts_speech'])
                if 'tts_token' in result:
                    token_list.append(result['tts_token'])

        if not audio_list or not token_list:
            logger.warning(f"  Chunk {chunk_idx}: no audio/tokens generated, skipping")
            continue

        audio_waveform = torch.cat(audio_list, dim=1)
        speech_tokens  = torch.cat(token_list,  dim=1)
        audio_duration = audio_waveform.shape[1] / sample_rate
        synth_time     = time.time() - t_synth

        logger.info(f"  ✓ Synthesized: {audio_duration:.2f}s in {synth_time:.1f}s  "
                     f"tokens={speech_tokens.shape}")

        # Predict blendshapes
        t_blend = time.time()
        blendshapes = predict_blendshapes(
            models, speech_tokens, audio_duration, fps=fps, device=device
        )
        blend_time = time.time() - t_blend
        logger.info(f"  ✓ Blendshapes: {blendshapes.shape} in {blend_time:.2f}s")

        # ---- Trim if requested ----
        if start_sec is not None or end_sec is not None:
            s = start_sec or 0.0
            e = end_sec or audio_duration

            # Trim audio
            s_sample = int(s * sample_rate)
            e_sample = int(e * sample_rate)
            audio_waveform = audio_waveform[:, s_sample:e_sample]

            # Trim blendshapes
            s_frame = int(s * fps)
            e_frame = int(e * fps)
            blendshapes = blendshapes[s_frame:e_frame]

            audio_duration = audio_waveform.shape[1] / sample_rate
            logger.info(f"  ✓ Trimmed to {s:.2f}s-{e:.2f}s → audio={audio_duration:.2f}s, frames={blendshapes.shape[0]}")

        # ---- Save chunk outputs ----
        audio_path = audio_dir / f"{chunk_name}.wav"
        torchaudio.save(str(audio_path), audio_waveform.cpu(), sample_rate)

        blendshapes_np = blendshapes.cpu().numpy()
        blend_path = blend_dir / f"{chunk_name}.npz"
        np.savez_compressed(str(blend_path), arkit_blendshapes=blendshapes_np)

        blend_pt_path = blend_dir / f"{chunk_name}.pt"
        torch.save(blendshapes.cpu(), blend_pt_path)

        tokens_path = token_dir / f"{chunk_name}_tokens.pt"
        torch.save(speech_tokens.cpu(), tokens_path)

        metadata = {
            'chunk_idx':           chunk_idx,
            'chunk_name':          chunk_name,
            'chunk_text':          chunk_text,
            'num_words':           len(chunk_text.split()),
            'ref_text':            ref_text,
            'ref_audio':           ref_audio_path,
            'synth_duration_s':    audio_duration,
            'num_sub_chunks':      len(sub_chunks),
            'speech_tokens_shape': list(speech_tokens.shape),
            'blendshapes_shape':   list(blendshapes.shape),
            'fps':                 fps,
            'sample_rate':         sample_rate,
            'timing': {
                'synthesis_s':  synth_time,
                'blendshape_s': blend_time,
            },
            'outputs': {
                'audio':       str(audio_path),
                'blendshapes': str(blend_path),
                'tokens':      str(tokens_path),
            },
        }
        meta_path = meta_dir / f"{chunk_name}.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"  ✓ Saved: {chunk_name}")

        all_audio.append(audio_waveform)
        all_blendshapes.append(blendshapes)
        stats['processed'] += 1
        stats['total_audio_duration'] += audio_duration
        stats['total_frames'] += blendshapes.shape[0]

    # ---- Concatenate all chunks ----
    if all_audio:
        # Concatenated audio
        full_audio = torch.cat(all_audio, dim=1)
        full_audio_path = audio_dir / f"{output_prefix}_full.wav"
        torchaudio.save(str(full_audio_path), full_audio.cpu(), sample_rate)
        logger.info(f"\n✓ Full audio: {full_audio.shape[1]/sample_rate:.2f}s → {full_audio_path}")

        # Concatenated blendshapes
        full_blendshapes = torch.cat(all_blendshapes, dim=0)
        full_blend_path = blend_dir / f"{output_prefix}_full.npz"
        np.savez_compressed(str(full_blend_path), arkit_blendshapes=full_blendshapes.cpu().numpy())
        logger.info(f"✓ Full blendshapes: {full_blendshapes.shape} → {full_blend_path}")

        # Concatenated tokens
        all_token_paths = sorted(
            token_dir.glob(f"{output_prefix}_chunk_*_tokens.pt"),
            key=lambda p: int(p.stem.split('_chunk_')[1].replace('_tokens', ''))
        )
        if all_token_paths:
            all_tokens = [torch.load(tp, map_location='cpu') for tp in all_token_paths]
            all_tokens = [t.unsqueeze(0) if t.dim() == 1 else t for t in all_tokens]
            full_tokens = torch.cat(all_tokens, dim=1)
            full_tokens_path = token_dir / f"{output_prefix}_all_tokens.pt"
            torch.save(full_tokens, full_tokens_path)
            logger.info(f"✓ Full tokens: {full_tokens.shape} → {full_tokens_path}")

    # ---- Summary ----
    logger.info(f"\n{'='*70}")
    logger.info("INFERENCE COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"  Chunks processed:     {stats['processed']}/{stats['total_chunks']}")
    logger.info(f"  Total audio duration: {stats['total_audio_duration']:.2f}s")
    logger.info(f"  Total frames:         {stats['total_frames']}")
    logger.info(f"  Output dir:           {output_dir}")
    logger.info(f"{'='*70}")

    summary_path = output_dir / "run_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(stats, f, indent=2)'''

def run_inference(
    text: str,
    ref_audio_path: str,
    ref_text: str,
    model_configs: List[Dict],
    output_dir: str = "results/ttsf_output/",
    output_prefix: str = "ttsf",
    cosyvoice_model_path: str = "../pretrained_models/CosyVoice2-0.5B",
    ref_max_duration: float = 10.0,
    fps: int = 30,
    device: str = "cuda",
    start_sec: float = None,
    end_sec: float = None,
):
    output_dir = Path(output_dir)

    audio_dir = output_dir / "audio"
    blend_dir = output_dir / "blendshapes"
    token_dir = output_dir / "tokens"
    meta_dir  = output_dir / "metadata"
    furhat_dir = output_dir / "furhat"
    for d in [audio_dir, blend_dir, token_dir, meta_dir, furhat_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- Load CosyVoice2 ----
    logger.info(f"Loading CosyVoice2 from {cosyvoice_model_path} ...")
    os.environ['MODELSCOPE_CACHE'] = os.path.expanduser('~/.cache/modelscope')
    cosyvoice = CosyVoice2(cosyvoice_model_path, load_jit=False, load_trt=False, fp16=False)
    sample_rate = cosyvoice.sample_rate
    logger.info(f"  ✓ CosyVoice2 loaded (sample_rate={sample_rate})")

    # ---- Load blendshape models ----
    logger.info(f"Loading {len(model_configs)} blendshape model(s)...")
    models = load_blendshape_models(model_configs, device)

    # ---- Load reference audio ----
    logger.info(f"Loading reference audio: {ref_audio_path}")
    ref_speech = load_wav(ref_audio_path, 16000)
    ref_max_samples = int(ref_max_duration * 16000)
    if ref_speech.shape[1] > ref_max_samples:
        ref_speech = ref_speech[:, :ref_max_samples]
        logger.info(f"  Trimmed reference to {ref_max_duration}s")
    logger.info(f"  ✓ Reference audio: {ref_speech.shape[1]/16000:.2f}s")
    logger.info(f"  Prompt text: \"{ref_text[:80]}{'...' if len(ref_text)>80 else ''}\"")

    # ---- Synthesize ----
    logger.info(f"\nSynthesizing: \"{text}\"")
    t_synth = time.time()

    audio_list, token_list = [], []
    generator = cosyvoice.inference_zero_shot(
        text, ref_text, ref_speech, stream=False
    )
    for result in generator:
        if 'tts_speech' in result:
            audio_list.append(result['tts_speech'])
        if 'tts_token' in result:
            token_list.append(result['tts_token'])

    if not audio_list or not token_list:
        logger.error("No audio/tokens generated!")
        return

    audio_waveform = torch.cat(audio_list, dim=1)
    speech_tokens  = torch.cat(token_list,  dim=1)
    audio_duration = audio_waveform.shape[1] / sample_rate
    synth_time     = time.time() - t_synth

    logger.info(f"  ✓ Synthesized: {audio_duration:.2f}s in {synth_time:.1f}s  "
                 f"tokens={speech_tokens.shape}")

    # ---- Predict blendshapes ----
    t_blend = time.time()
    blendshapes = predict_blendshapes(
        models, speech_tokens, audio_duration, fps=fps, device=device
    )
    blend_time = time.time() - t_blend
    logger.info(f"  ✓ Blendshapes: {blendshapes.shape} in {blend_time:.2f}s")

    # ---- Trim if requested ----
    if start_sec is not None or end_sec is not None:
        s = start_sec or 0.0
        e = end_sec or audio_duration

        audio_waveform = audio_waveform[:, int(s * sample_rate):int(e * sample_rate)]
        blendshapes = blendshapes[int(s * fps):int(e * fps)]
        audio_duration = audio_waveform.shape[1] / sample_rate
        logger.info(f"  ✓ Trimmed to {s:.2f}s-{e:.2f}s → audio={audio_duration:.2f}s, frames={blendshapes.shape[0]}")

    # ---- Save outputs ----
    audio_path = audio_dir / f"{output_prefix}.wav"
    torchaudio.save(str(audio_path), audio_waveform.cpu(), sample_rate)

    blendshapes_np = blendshapes.cpu().numpy()
    blend_path = blend_dir / f"{output_prefix}.npz"
    np.savez_compressed(str(blend_path), arkit_blendshapes=blendshapes_np)

    tokens_path = token_dir / f"{output_prefix}_tokens.pt"
    torch.save(speech_tokens.cpu(), tokens_path)

    # Furhat .txt
    # furhat_txt_path = furhat_dir / f"{output_prefix}.txt"
    # save_furhat_txt(blendshapes_np, furhat_txt_path, fps=fps)

    metadata = {
        'text':                text,
        'ref_text':            ref_text,
        'ref_audio':           ref_audio_path,
        'synth_duration_s':    audio_duration,
        'speech_tokens_shape': list(speech_tokens.shape),
        'blendshapes_shape':   list(blendshapes.shape),
        'fps':                 fps,
        'sample_rate':         sample_rate,
        'start_sec':           start_sec,
        'end_sec':             end_sec,
        'timing': {
            'synthesis_s':  synth_time,
            'blendshape_s': blend_time,
        },
    }
    meta_path = meta_dir / f"{output_prefix}.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\n✓ Saved:")
    logger.info(f"  Audio:      {audio_path}")
    logger.info(f"  Blendshapes: {blend_path}")
    # logger.info(f"  Furhat txt: {furhat_txt_path}")
    logger.info(f"  Tokens:     {tokens_path}")
    logger.info(f"  Metadata:   {meta_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Text-to-Speech + Facial Animation inference with CosyVoice2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        )

    # Input text (one of these required)
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument('--text', type=str, help='Text to synthesize')
    text_group.add_argument('--text_file', type=str, help='Path to text file to synthesize')

    # Reference audio
    parser.add_argument('--ref_audio', type=str, required=True,
                        help='Path to reference audio for voice cloning (.wav)')
    parser.add_argument('--ref_text', type=str, required=True,
                        help='Transcription of the reference audio')
    parser.add_argument('--ref_max_duration', type=float, default=5.0,
                        help='Max reference audio duration in seconds (default: 10)')
    parser.add_argument('--start_sec', type=float, default=None, help='Trim output from this time')
    parser.add_argument('--end_sec', type=float, default=None, help='Trim output to this time')

    # Models
    parser.add_argument(
        '--model', action='append', nargs='+', required=True,
        metavar=('CHECKPOINT', 'REGION'),
        help='checkpoint_path region1 region2 ... [weight]. Repeat for each model.'
    )

    # Chunking
    parser.add_argument('--chunk_duration', type=float, default=5.0,
                        help='Target chunk duration in seconds (default: 10)')
    parser.add_argument('--speaking_rate', type=float, default=150.0,
                        help='Estimated speaking rate in words/minute (default: 150)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Blendshape frame rate (default: 30)')

    # Output
    parser.add_argument('--output_prefix', type=str, default='ttsf',
                        help='Prefix for output filenames (default: ttsf)')
    parser.add_argument('--output_dir', type=str, default='results/ttsf_output/')

    # System
    parser.add_argument('--cosyvoice_model', type=str,
                        default='../cosyvoice2_decode_v3/pretrained_models/CosyVoice2-0.5B')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Read text
    if args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        logger.info(f"Read text from {args.text_file}: {len(text.split())} words")
    else:
        text = args.text

    # Parse model configs
    model_configs = []
    for model_spec in args.model:
        checkpoint_path = model_spec[0]
        regions = []
        weight = 1.0
        for item in model_spec[1:]:
            try:
                weight = float(item)
            except ValueError:
                regions.append(item)
        if not regions:
            parser.error(f"No regions specified for model: {checkpoint_path}")
        model_configs.append({
            'checkpoint_path': checkpoint_path,
            'regions': regions,
            'weight': weight,
        })

    # Print config
    print(f"\n{'='*70}")
    print("TTSF INFERENCE CONFIG")
    print(f"{'='*70}")
    print(f"  Text:             \"{text[:100]}{'...' if len(text)>100 else ''}\"")
    print(f"  Text length:      {len(text.split())} words")
    print(f"  Ref audio:        {args.ref_audio}")
    print(f"  Ref text:         \"{args.ref_text[:80]}{'...' if len(args.ref_text)>80 else ''}\"")
    print(f"  Chunk duration:   {args.chunk_duration}s")
    print(f"  Speaking rate:    {args.speaking_rate} wpm")
    print(f"  FPS:              {args.fps}")
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Output prefix:    {args.output_prefix}")
    for i, cfg in enumerate(model_configs):
        print(f"  Model {i+1}: {Path(cfg['checkpoint_path']).name}  "
              f"regions={cfg['regions']}  weight={cfg['weight']}")
    print(f"{'='*70}\n")

    # run_inference(
    #     text=text,
    #     ref_audio_path=args.ref_audio,
    #     ref_text=args.ref_text,
    #     model_configs=model_configs,
    #     output_dir=args.output_dir,
    #     output_prefix=args.output_prefix,
    #     cosyvoice_model_path=args.cosyvoice_model,
    #     chunk_duration=args.chunk_duration,
    #     speaking_rate_wpm=args.speaking_rate,
    #     ref_max_duration=args.ref_max_duration,
    #     fps=args.fps,
    #     device=args.device,
    #     start_sec=args.start_sec,
    #     end_sec=args.end_sec,
    # )

    run_inference(
        text=text,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        model_configs=model_configs,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        cosyvoice_model_path=args.cosyvoice_model,
        ref_max_duration=args.ref_max_duration,
        fps=args.fps,
        device=args.device,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
    )

'''

# Full setup with two models
python inference_chunks.py \
    --text 'I, said, aba.' \
    --ref_audio ref_audios/ref_george_v2.wav \
    --ref_text 'My brother found a very sharp fishing blade on the beach.' \
    --model ../cosyvoice2_decode_v3/modular/checkpoints/mouth_jaw_1000ep_4bs_no_overlap/jaw_mouth_model_epoch_999.pth jaw mouth 1.0 \
    --model ../cosyvoice2_decode_v3/modular/checkpoints/cben_1000ep_32bs_no_overlap/eye_blink_eye_look_eye_shape_brow_cheek_nose_model_epoch_999.pth eye_blink eye_look eye_shape brow cheek nose 1.0 \
    --chunk_duration 2.0  \
    --output_prefix demo \
    --ref_max_duration  3.5 \
    --output_dir results_furhat/dummy_test/

python inference_chunks.py \
    --text 'I, said, ahba, afa, ahda, asha.' \
    --ref_audio ref_audios/ref_george_v2.wav \
    --ref_text 'My brother found a very sharp fishing blade on the beach.' \
    --model ../cosyvoice2_decode_v3/modular/checkpoints/mouth_jaw_1000ep_4bs_no_overlap/jaw_mouth_model_epoch_999.pth jaw mouth 1.0 \
    --model ../cosyvoice2_decode_v3/modular/checkpoints/cben_1000ep_32bs_no_overlap/eye_blink_eye_look_eye_shape_brow_cheek_nose_model_epoch_999.pth eye_blink eye_look eye_shape brow cheek nose 1.0 \
    --chunk_duration 3.0  \
    --output_prefix demo \
    --ref_max_duration  3.5 \
    --start_sec 4.35 --end_sec 4.86\
    --output_dir results_furhat_2/george_acha/

python inference_chunks.py \
    --text 'I, said, ahba, afa, ahda, asha.' \
    --ref_audio ref_audios/ref_rachel_v2.wav \
    --ref_text 'My brother found a very sharp fishing blade on the beach.' \
    --model ../cosyvoice2_decode_v3/modular/checkpoints/mouth_jaw_1000ep_4bs_no_overlap/jaw_mouth_model_epoch_999.pth jaw mouth 1.0 \
    --model ../cosyvoice2_decode_v3/modular/checkpoints/cben_1000ep_32bs_no_overlap/eye_blink_eye_look_eye_shape_brow_cheek_nose_model_epoch_999.pth eye_blink eye_look eye_shape brow cheek nose 1.0 \
    --chunk_duration 4.0 \
    --output_prefix demo \
    --ref_max_duration  4.0 \
    --start_sec 5.10 --end_sec 5.72\
    --output_dir results_furhat_2/rachel_ada/


python inference_chunks.py \
    --text 'I said, ahba. I said, afa. I said, ahda. I said, asha.' \
    --ref_audio ref_audios/ref_rachel_v2.wav \
    --ref_text 'My brother found a very sharp fishing blade on the beach.' \
    --model ../cosyvoice2_decode_v3/modular/checkpoints/mouth_jaw_1000ep_4bs_no_overlap/jaw_mouth_model_epoch_999.pth jaw mouth 1.0 \
    --model ../cosyvoice2_decode_v3/modular/checkpoints/cben_1000ep_32bs_no_overlap/eye_blink_eye_look_eye_shape_brow_cheek_nose_model_epoch_999.pth eye_blink eye_look eye_shape brow cheek nose 1.0 \
    --output_prefix rachel_acha \
    --ref_max_duration  4.0 \
    --start_sec 6.483 --end_sec 8.320\
    --output_dir results_furhat_3/

python inference_chunks.py \
    --text 'I said, obo, I said, aufo, I said, odo, I said, osho.' \
    --ref_audio ref_audios/ref_george_v2.wav \
    --ref_text 'My brother found a very sharp fishing blade on the beach.' \
    --model ../cosyvoice2_decode_v3/modular/checkpoints/mouth_jaw_1000ep_4bs_no_overlap/jaw_mouth_model_epoch_999.pth jaw mouth 1.0 \
    --model ../cosyvoice2_decode_v3/modular/checkpoints/cben_1000ep_32bs_no_overlap/eye_blink_eye_look_eye_shape_brow_cheek_nose_model_epoch_999.pth eye_blink eye_look eye_shape brow cheek nose 1.0 \
    --output_prefix george_ocho\
    --ref_max_duration 4.0 \
    --start_sec 5.463 --end_sec 7.120\
    --output_dir results_furhat_3

'''