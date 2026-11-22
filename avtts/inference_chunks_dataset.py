#!/usr/bin/env python3
"""
BEAT2 Test Set Inference Pipeline (10s Chunked, Separate Files)
================================================================

Batch inference over the BEAT2 test split with 10-second chunked synthesis.

For each sample:
  1. Parse TextGrid → extract text + word timings
  2. Fixed reference: first 10s of GT audio + its transcription as prompt_text
  3. Divide recording into 10s windows; discard last chunk if < 10s remaining
  4. For each chunk: synthesize with CosyVoice2 zero-shot → extract tokens
  5. Feed tokens into multi-model blendshape decoder
  6. Save each chunk as a separate file: {sample_id}_chunk_{N}

Outputs per chunk:
  audio/{sample_id}_chunk_{N}.wav
  blendshapes/{sample_id}_chunk_{N}.npz
  blendshapes/{sample_id}_chunk_{N}.pt
  tokens/{sample_id}_chunk_{N}_tokens.pt
  metadata/{sample_id}_chunk_{N}.json

Usage:
    python beat2_test_inference.py \\
        --split_csv    ../BEAT2/beat_english_v2.0.0/train_test_split.csv \\
        --textgrid_dir ../BEAT2/beat_english_v2.0.0/textgrid \\
        --wave_dir     ../BEAT2/beat_english_v2.0.0/wave16k \\
        --split test \\
        --model checkpoints/jaw_mouth.pth jaw mouth 1.0 \\
        --model checkpoints/rest.pth eye_blink eye_look eye_shape brow cheek nose 1.0 \\
        --chunk_duration 10 \\
        --output_dir results/beat2_test_synth/
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
import csv
import json
import logging
import time
import math
from pathlib import Path
from typing import List, Optional, Tuple, Dict

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
# CSV split loader
# =============================================================================

def load_split_csv(csv_path: str, target_split: str = "test") -> List[str]:
    """Load sample IDs for the given split from the BEAT2 CSV."""
    sample_ids = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)

        # Handle headerless CSV (e.g., 18_daiki_0_10_10,val)
        has_header = header is not None and any(
            h.strip().lower() in ('id', 'type', 'split', 'name') for h in header
        )
        if not has_header and header is not None:
            row_id, row_type = header[0].strip(), header[1].strip()
            if row_type == target_split:
                sample_ids.append(row_id)

        for row in reader:
            if len(row) >= 2:
                sample_id, split_type = row[0].strip(), row[1].strip()
                if split_type == target_split:
                    sample_ids.append(sample_id)

    logger.info(f"Loaded {len(sample_ids)} samples for split '{target_split}'")
    return sample_ids


# =============================================================================
# TextGrid parser
# =============================================================================

def parse_textgrid(path: str) -> Tuple[str, float, List[dict]]:
    """
    Parse a Praat TextGrid file.

    Returns:
        full_text:   all words joined by spaces
        gt_duration: total recording duration (xmax from file header)
        words:       list of {'text', 't_start', 't_end'}
    """
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    xmax_match = re.search(r'^xmax = ([\d.]+)', content, re.MULTILINE)
    if not xmax_match:
        raise ValueError(f"Could not find xmax in TextGrid header: {path}")
    gt_duration = float(xmax_match.group(1))

    tier_match = re.search(
        r'name = "words".*?intervals: size = \d+(.*?)(?=item \[\d+\]:|$)',
        content, re.DOTALL
    )
    if not tier_match:
        raise ValueError(f"Could not find 'words' tier in TextGrid: {path}")

    words = []
    for m in re.finditer(
        r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"',
        tier_match.group(1)
    ):
        text = m.group(3).strip()
        if text:
            words.append({
                'text':    text,
                't_start': float(m.group(1)),
                't_end':   float(m.group(2)),
            })

    full_text = " ".join(w['text'] for w in words)
    return full_text, gt_duration, words


def extract_words_in_window(words: List[dict], t_start: float, t_end: float) -> List[dict]:
    """Return words whose midpoint falls within [t_start, t_end)."""
    selected = []
    for w in words:
        mid = (w['t_start'] + w['t_end']) / 2.0
        if t_start <= mid < t_end:
            selected.append(w)
    return selected


def words_to_text(words: List[dict]) -> str:
    return " ".join(w['text'] for w in words)


# =============================================================================
# File finding
# =============================================================================

def find_file(base_dir: Path, sample_id: str, extension: str) -> Optional[Path]:
    """Try flat path first, then speaker subfolder."""
    flat = base_dir / f"{sample_id}{extension}"
    if flat.exists():
        return flat

    parts = sample_id.split('_')
    if len(parts) >= 2:
        speaker = f"{parts[0]}_{parts[1]}"
        nested = base_dir / speaker / f"{sample_id}{extension}"
        if nested.exists():
            return nested

    return None


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
    """
    Run all blendshape models on speech tokens and combine by region mask.

    Args:
        models:         list of model dicts from load_blendshape_models
        speech_tokens:  (1, T_tokens) long tensor — same tokens used by CosyVoice2 vocoder
        audio_duration: duration in seconds (to compute target frame count)
        fps:            blendshape frame rate

    Returns:
        combined: (num_frames, 51) float tensor
    """
    num_frames = int(audio_duration * fps)

    if speech_tokens.dim() == 1:
        speech_tokens = speech_tokens.unsqueeze(0)
    speech_tokens = speech_tokens.to(device)

    all_preds = []
    for info in models:
        with torch.no_grad():
            pred = info['model'](speech_tokens, target_length=num_frames)
        all_preds.append({
            'prediction': pred.squeeze(0),  # (T, 51)
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
# Text chunking for CosyVoice2 (sub-sentence splitting within a chunk)
# =============================================================================

def split_into_sentences(text: str, max_words: int = 20) -> List[str]:
    """Split text into short sub-chunks at punctuation boundaries for CosyVoice2."""
    sentences = re.split(r'(?<=[.!?,])\s+', text.strip())
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


# =============================================================================
# Main batch inference
# =============================================================================

def run_batch_inference(
    split_csv: str,
    textgrid_dir: str,
    wave_dir: str,
    model_configs: List[Dict],
    split: str = "test",
    output_dir: str = "results/beat2_test_synth/",
    cosyvoice_model_path: str = "../pretrained_models/CosyVoice2-0.5B",
    chunk_duration: float = 10.0,
    fps: int = 30,
    device: str = "cuda",
    resume: bool = True,
):
    """
    Run batch inference on the BEAT2 dataset.

    Each sample is divided into non-overlapping chunks of chunk_duration seconds.
    The last chunk is discarded if it has less than chunk_duration of content.

    Reference audio and prompt_text are always taken from the first chunk_duration
    seconds of the GT recording (fixed across all chunks of a sample).

    Each chunk produces separate output files: {sample_id}_chunk_{N}.{ext}
    """
    output_dir   = Path(output_dir)
    textgrid_dir = Path(textgrid_dir)
    wave_dir     = Path(wave_dir)

    # Create output dirs
    audio_dir = output_dir / "audio"
    blend_dir = output_dir / "blendshapes"
    token_dir = output_dir / "tokens"
    meta_dir  = output_dir / "metadata"
    for d in [audio_dir, blend_dir, token_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- Load split ----
    sample_ids = load_split_csv(split_csv, split)
    if not sample_ids:
        logger.error(f"No samples found for split '{split}'")
        return

    # ---- Load CosyVoice2 ----
    logger.info(f"\nLoading CosyVoice2 from {cosyvoice_model_path} ...")
    os.environ['MODELSCOPE_CACHE'] = os.path.expanduser('~/.cache/modelscope')
    cosyvoice = CosyVoice2(cosyvoice_model_path, load_jit=False, load_trt=False, fp16=False)
    sample_rate = cosyvoice.sample_rate
    logger.info(f"  ✓ CosyVoice2 loaded (sample_rate={sample_rate})")

    # ---- Load blendshape models ----
    logger.info(f"\nLoading {len(model_configs)} blendshape model(s)...")
    models = load_blendshape_models(model_configs, device)

    # ---- Process samples ----
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing {len(sample_ids)} samples from split '{split}'")
    logger.info(f"  chunk_duration: {chunk_duration}s")
    logger.info(f"  fps:            {fps}")
    logger.info(f"  resume:         {resume}")
    logger.info(f"{'='*70}\n")

    stats = {
        'total_samples':    len(sample_ids),
        'processed':        0,
        'total_chunks':     0,
        'skipped_existing': 0,
        'skipped_missing':  0,
        'failed':           0,
        'errors':           [],
    }

    for idx, sample_id in enumerate(sample_ids):
        logger.info(f"\n{'='*70}")
        logger.info(f"[{idx+1}/{len(sample_ids)}] {sample_id}")
        logger.info(f"{'='*70}")

        # ---- Find files ----
        tg_path  = find_file(textgrid_dir, sample_id, ".TextGrid")
        wav_path = find_file(wave_dir, sample_id, ".wav")

        if tg_path is None:
            logger.warning(f"  ✗ TextGrid not found, skipping")
            stats['skipped_missing'] += 1
            continue
        if wav_path is None:
            logger.warning(f"  ✗ WAV not found, skipping")
            stats['skipped_missing'] += 1
            continue

        try:
            t_sample_start = time.time()

            # ---- 1. Parse TextGrid ----
            full_text, gt_duration, words = parse_textgrid(str(tg_path))
            logger.info(f"  TextGrid: {len(words)} words, {gt_duration:.1f}s")

            if not words:
                logger.warning(f"  ✗ No words in TextGrid, skipping")
                stats['skipped_missing'] += 1
                continue

            # ---- 2. Fixed reference: first chunk_duration seconds ----
            ref_words   = extract_words_in_window(words, 0.0, chunk_duration)
            prompt_text = words_to_text(ref_words)

            if not prompt_text.strip():
                # Fallback: first few words regardless of timing
                prompt_text = " ".join(w['text'] for w in words[:10])

            logger.info(f"  Prompt text: \"{prompt_text[:80]}{'...' if len(prompt_text)>80 else ''}\"")

            # Load full GT audio, extract first chunk_duration as reference
            full_gt_audio   = load_wav(str(wav_path), 16000)
            ref_max_samples = int(chunk_duration * 16000)
            ref_speech      = full_gt_audio[:, :ref_max_samples]
            logger.info(f"  Reference audio: {ref_speech.shape[1]/16000:.2f}s")

            # ---- 3. Determine chunks ----
            num_full_chunks = int(gt_duration // chunk_duration)

            if num_full_chunks == 0:
                logger.warning(f"  ✗ Recording shorter than {chunk_duration}s "
                               f"({gt_duration:.1f}s), skipping")
                stats['skipped_missing'] += 1
                continue

            remainder = gt_duration - num_full_chunks * chunk_duration
            logger.info(f"  Chunks: {num_full_chunks} × {chunk_duration}s "
                        f"(discarding {remainder:.1f}s remainder)")

            # ---- 4. Process each chunk ----
            sample_chunk_count = 0

            for chunk_idx in range(num_full_chunks):
                t_start_window = chunk_idx * chunk_duration
                t_end_window   = (chunk_idx + 1) * chunk_duration

                chunk_name = f"{sample_id}_chunk_{chunk_idx}"

                # Check resume
                blend_out = blend_dir / f"{chunk_name}.npz"
                if resume and blend_out.exists():
                    logger.info(f"  Chunk {chunk_idx}: ⏭ exists, skipping")
                    stats['skipped_existing'] += 1
                    sample_chunk_count += 1
                    continue

                # Extract text for this window
                chunk_words = extract_words_in_window(words, t_start_window, t_end_window)
                chunk_text  = words_to_text(chunk_words)

                if not chunk_text.strip():
                    logger.warning(f"  Chunk {chunk_idx} ({t_start_window:.0f}-{t_end_window:.0f}s): "
                                   f"no words, skipping")
                    continue

                logger.info(f"\n  --- Chunk {chunk_idx} ({t_start_window:.0f}-{t_end_window:.0f}s) ---")
                logger.info(f"  Text ({len(chunk_words)} words): "
                            f"\"{chunk_text[:70]}{'...' if len(chunk_text)>70 else ''}\"")

                # Split chunk text into sub-sentences if too long for CosyVoice2
                sub_chunks = split_into_sentences(chunk_text, max_words=20)

                audio_list, token_list = [], []
                t_synth = time.time()

                for sc in sub_chunks:
                    generator = cosyvoice.inference_zero_shot(
                        sc, prompt_text, ref_speech, stream=False
                    )
                    for result in generator:
                        if 'tts_speech' in result:
                            audio_list.append(result['tts_speech'])
                        if 'tts_token' in result:
                            token_list.append(result['tts_token'])

                if not audio_list or not token_list:
                    logger.warning(f"  Chunk {chunk_idx}: no audio/tokens generated, skipping")
                    continue

                audio_waveform = torch.cat(audio_list, dim=1)   # (1, N)
                speech_tokens  = torch.cat(token_list,  dim=1)  # (1, T_tok)
                audio_duration = audio_waveform.shape[1] / sample_rate
                synth_time     = time.time() - t_synth

                logger.info(f"  ✓ Synthesized: {audio_duration:.2f}s in {synth_time:.1f}s  "
                             f"tokens={speech_tokens.shape}")

                # Predict blendshapes from tokens
                t_blend = time.time()
                blendshapes = predict_blendshapes(
                    models, speech_tokens, audio_duration, fps=fps, device=device
                )
                blend_time = time.time() - t_blend
                logger.info(f"  ✓ Blendshapes: {blendshapes.shape} in {blend_time:.2f}s")

                # ---- Save outputs ----
                audio_path = audio_dir / f"{chunk_name}.wav"
                torchaudio.save(str(audio_path), audio_waveform.cpu(), sample_rate)

                blendshapes_np = blendshapes.cpu().numpy()
                np.savez_compressed(str(blend_out), arkit_blendshapes=blendshapes_np)

                blend_pt_path = blend_dir / f"{chunk_name}.pt"
                torch.save(blendshapes.cpu(), blend_pt_path)

                tokens_path = token_dir / f"{chunk_name}_tokens.pt"
                torch.save(speech_tokens.cpu(), tokens_path)

                metadata = {
                    'sample_id':           sample_id,
                    'chunk_idx':           chunk_idx,
                    'chunk_name':          chunk_name,
                    'window_start_s':      t_start_window,
                    'window_end_s':        t_end_window,
                    'gt_duration_s':       gt_duration,
                    'chunk_text':          chunk_text,
                    'num_words':           len(chunk_words),
                    'prompt_text':         prompt_text,
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
                        'blendshapes': str(blend_out),
                        'tokens':      str(tokens_path),
                    },
                }
                meta_path = meta_dir / f"{chunk_name}.json"
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                logger.info(f"  ✓ Saved: {chunk_name}")
                sample_chunk_count += 1
                stats['total_chunks'] += 1

            # ---- 5. Concatenate all chunk tokens into a single file ----
            if sample_chunk_count > 0:
                all_token_paths = sorted(
                    token_dir.glob(f"{sample_id}_chunk_*_tokens.pt"),
                    key=lambda p: int(p.stem.split('_chunk_')[1].replace('_tokens', ''))
                )
                if all_token_paths:
                    all_tokens = []
                    for tp in all_token_paths:
                        t = torch.load(tp, map_location='cpu')
                        if t.dim() == 1:
                            t = t.unsqueeze(0)
                        all_tokens.append(t)
                    concatenated = torch.cat(all_tokens, dim=1)  # (1, total_tokens)
                    concat_path = token_dir / f"{sample_id}_all_tokens.pt"
                    torch.save(concatenated, concat_path)
                    logger.info(f"  ✓ Concatenated tokens: {concatenated.shape} → {concat_path}")

            if sample_chunk_count > 0:
                stats['processed'] += 1
                total_sample_time = time.time() - t_sample_start
                logger.info(f"\n  ✓ {sample_id}: {sample_chunk_count} chunks in {total_sample_time:.1f}s")

        except Exception as e:
            logger.error(f"  ✗ FAILED: {e}", exc_info=True)
            stats['failed'] += 1
            stats['errors'].append({'sample_id': sample_id, 'error': str(e)})

    # ---- Final summary ----
    logger.info(f"\n{'='*70}")
    logger.info("BATCH INFERENCE COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"  Split:              {split}")
    logger.info(f"  Total samples:      {stats['total_samples']}")
    logger.info(f"  Processed:          {stats['processed']}")
    logger.info(f"  Total chunks saved: {stats['total_chunks']}")
    logger.info(f"  Skipped (existing): {stats['skipped_existing']}")
    logger.info(f"  Skipped (missing):  {stats['skipped_missing']}")
    logger.info(f"  Failed:             {stats['failed']}")
    logger.info(f"{'='*70}")

    summary_path = output_dir / "run_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  Summary saved: {summary_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='BEAT2 test set batch inference: 10s chunked CosyVoice2 + blendshapes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python beat2_test_inference.py \\
        --split_csv    ../BEAT2/beat_english_v2.0.0/train_test_split.csv \\
        --textgrid_dir ../BEAT2/beat_english_v2.0.0/textgrid \\
        --wave_dir     ../BEAT2/beat_english_v2.0.0/wave16k \\
        --split test \\
        --model checkpoints/jaw_mouth.pth jaw mouth 1.0 \\
        --model checkpoints/rest.pth eye_blink eye_look eye_shape brow cheek nose 1.0 \\
        --chunk_duration 10 \\
        --output_dir results/beat2_test_synth/
"""
    )

    parser.add_argument('--split_csv',    type=str, default='../BEAT2/beat_english_v2.0.0/train_test_split.csv',
                        help='Path to BEAT2 train_test_split.csv')
    parser.add_argument('--textgrid_dir', type=str, default='../BEAT2/beat_english_v2.0.0/textgrid',
                        help='Directory containing .TextGrid files')
    parser.add_argument('--wave_dir',     type=str, default='../BEAT2/beat_english_v2.0.0/wave16k',
                        help='Directory containing .wav files (flat or speaker subfolders)')
    parser.add_argument('--split',        type=str, default='test',
                        choices=['test', 'val', 'train', 'additional'],
                        help='Which split to process (default: test)')

    parser.add_argument(
        '--model', action='append', nargs='+', required=True,
        metavar=('CHECKPOINT', 'REGION'),
        help='checkpoint_path region1 region2 ... [weight]. Repeat for each model.'
    )

    parser.add_argument('--chunk_duration', type=float, default=10.0,
                        help='Duration of each chunk in seconds (default: 10)')
    parser.add_argument('--fps',            type=int,   default=30,
                        help='Blendshape frame rate (default: 30)')

    parser.add_argument('--cosyvoice_model', type=str,
                        default='../pretrained_models/CosyVoice2-0.5B')
    parser.add_argument('--output_dir', type=str, default='results/beat2_test_synth/')
    parser.add_argument('--device',     type=str, default='cuda')
    parser.add_argument('--no_resume',  action='store_true',
                        help='Re-process all chunks even if outputs exist')

    args = parser.parse_args()

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
    print("BEAT2 CHUNKED INFERENCE CONFIG")
    print(f"{'='*70}")
    print(f"  Split CSV:       {args.split_csv}")
    print(f"  TextGrid dir:    {args.textgrid_dir}")
    print(f"  Wave dir:        {args.wave_dir}")
    print(f"  Split:           {args.split}")
    print(f"  Chunk duration:  {args.chunk_duration}s")
    print(f"  FPS:             {args.fps}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Resume:          {not args.no_resume}")
    for i, cfg in enumerate(model_configs):
        print(f"  Model {i+1}: {Path(cfg['checkpoint_path']).name}  "
              f"regions={cfg['regions']}  weight={cfg['weight']}")
    print(f"{'='*70}\n")

    run_batch_inference(
        split_csv=args.split_csv,
        textgrid_dir=args.textgrid_dir,
        wave_dir=args.wave_dir,
        model_configs=model_configs,
        split=args.split,
        output_dir=args.output_dir,
        cosyvoice_model_path=args.cosyvoice_model,
        chunk_duration=args.chunk_duration,
        fps=args.fps,
        device=args.device,
        resume=not args.no_resume,
    )