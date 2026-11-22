#!/usr/bin/env python3
"""
BEAT2 Test Set Inference Pipeline
==================================

Batch inference over the BEAT2 test split:
  1. Parse CSV split file → select test samples
  2. For each sample: parse TextGrid → extract text + word timings
  3. Use first N seconds of GT audio as reference (prompt_text from words in that window)
  4. Synthesize speech with CosyVoice2 zero-shot
  5. Extract speech tokens → predict blendshapes via multi-model decoder
  6. Save audio + blendshapes + tokens + metadata

Outputs per sample:
  {sample_id}.wav           — synthesized audio
  {sample_id}.npz           — predicted blendshapes (arkit_blendshapes key)
  {sample_id}.pt            — predicted blendshapes (torch format)
  {sample_id}_tokens.pt     — CosyVoice2 speech tokens
  {sample_id}_metadata.json — full metadata with timing info

Usage:
    python beat2_test_inference.py \
        --split_csv   ../BEAT2/beat_english_v2.0.0/train_test_split.csv \
        --textgrid_dir ../BEAT2/beat_english_v2.0.0/textgrid \
        --wave_dir     ../BEAT2/beat_english_v2.0.0/wave16k \
        --split test \
        --model checkpoints/jaw_mouth.pth jaw mouth 1.0 \
        --model checkpoints/rest.pth eye_blink eye_look eye_shape brow cheek nose 1.0 \
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

        # Handle headerless CSV (like: 18_daiki_0_10_10,val)
        has_header = header is not None and any(
            h.strip().lower() in ('id', 'type', 'split', 'name') for h in header
        )
        if not has_header and header is not None:
            # First row is data, not a header
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


def extract_text_in_window(words: List[dict], start: float, end: float) -> str:
    """Extract the text of all words whose midpoint falls within [start, end]."""
    selected = []
    for w in words:
        mid = (w['t_start'] + w['t_end']) / 2.0
        if start <= mid <= end:
            selected.append(w['text'])
    return " ".join(selected)


def extract_text_up_to_duration(words: List[dict], max_duration: float) -> str:
    """Extract text for words that end before max_duration seconds."""
    selected = []
    for w in words:
        if w['t_end'] <= max_duration:
            selected.append(w['text'])
        else:
            # Include partial — if the word starts before the cutoff
            if w['t_start'] < max_duration:
                selected.append(w['text'])
            break
    return " ".join(selected)


# =============================================================================
# File finding utilities
# =============================================================================

def find_file(base_dir: Path, sample_id: str, extension: str) -> Optional[Path]:
    """
    Find a file for a sample ID, trying:
      1. base_dir/{sample_id}.ext           (flat)
      2. base_dir/{speaker}/{sample_id}.ext (speaker subfolder)
    """
    # Flat
    flat = base_dir / f"{sample_id}{extension}"
    if flat.exists():
        return flat

    # Speaker subfolder: extract speaker from ID (e.g., "1_wayne" from "1_wayne_0_2_2")
    parts = sample_id.split('_')
    if len(parts) >= 2:
        speaker = f"{parts[0]}_{parts[1]}"
        nested = base_dir / speaker / f"{sample_id}{extension}"
        if nested.exists():
            return nested

    return None


# =============================================================================
# Region mask + model loading helpers
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
    Run all blendshape models and combine by region mask.

    Args:
        models:         list of model dicts from load_blendshape_models
        speech_tokens:  (1, T_tokens) long tensor
        audio_duration: duration in seconds
        fps:            target blendshape frame rate

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
            pred = info['model'](speech_tokens, target_length=num_frames)  # (1, T, 51)
        all_preds.append({
            'prediction': pred.squeeze(0),  # (T, 51)
            'mask':       info['mask'],
            'weight':     info['weight'],
        })

    # Combine by region
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
# Text chunking for CosyVoice2
# =============================================================================

def split_into_sentences(text: str, max_words: int = 20) -> List[str]:
    """Split text into short chunks at punctuation boundaries."""
    sentences = re.split(r'(?<=[.!?,])\s+', text.strip())
    chunks = []
    for sent in sentences:
        words = sent.split()
        if len(words) <= max_words:
            chunks.append(sent)
        else:
            while len(words) > max_words:
                break_at = max_words
                for i in range(max_words - 1, max_words // 2, -1):
                    if words[i].lower() in {'and', 'but', 'or', 'so', 'because',
                                             'when', 'if', 'that', 'which', 'the'}:
                        break_at = i
                        break
                chunks.append(" ".join(words[:break_at]))
                words = words[break_at:]
            if words:
                chunks.append(" ".join(words))
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
    ref_duration: float = 10.0,
    max_synth_duration: float = 30.0,
    fps: int = 30,
    device: str = "cuda",
    resume: bool = True,
):
    """
    Run batch inference on the BEAT2 test set.

    Args:
        split_csv:           Path to train_test_split.csv
        textgrid_dir:        Directory with .TextGrid files
        wave_dir:            Directory with .wav files (flat structure)
        model_configs:       List of blendshape model configs
        split:               Which split to process ('test', 'val', 'train')
        output_dir:          Where to save results
        cosyvoice_model_path: Path to CosyVoice2 model
        ref_duration:        Seconds of GT audio to use as reference for zero-shot
        max_synth_duration:  Max seconds of text to synthesize (words beyond this are dropped)
        fps:                 Blendshape frame rate
        device:              'cuda' or 'cpu'
        resume:              Skip samples that already have outputs
    """
    output_dir = Path(output_dir)
    textgrid_dir = Path(textgrid_dir)
    wave_dir = Path(wave_dir)

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

    # ---- Process each sample ----
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing {len(sample_ids)} samples from split '{split}'")
    logger.info(f"  ref_duration:       {ref_duration}s")
    logger.info(f"  max_synth_duration: {max_synth_duration}s")
    logger.info(f"  fps:                {fps}")
    logger.info(f"  resume:             {resume}")
    logger.info(f"{'='*70}\n")

    results_summary = {
        'processed': 0,
        'skipped_existing': 0,
        'skipped_missing': 0,
        'failed': 0,
        'errors': [],
    }

    for idx, sample_id in enumerate(sample_ids):
        logger.info(f"\n[{idx+1}/{len(sample_ids)}] {sample_id}")

        # Check if already done
        blend_out = blend_dir / f"{sample_id}.npz"
        if resume and blend_out.exists():
            logger.info(f"  ⏭ Already exists, skipping")
            results_summary['skipped_existing'] += 1
            continue

        # ---- Find files ----
        tg_path = find_file(textgrid_dir, sample_id, ".TextGrid")
        wav_path = find_file(wave_dir, sample_id, ".wav")

        if tg_path is None:
            logger.warning(f"  ✗ TextGrid not found, skipping")
            results_summary['skipped_missing'] += 1
            continue
        if wav_path is None:
            logger.warning(f"  ✗ WAV not found, skipping")
            results_summary['skipped_missing'] += 1
            continue

        try:
            t_sample_start = time.time()

            # ---- 1. Parse TextGrid ----
            full_text, gt_duration, words = parse_textgrid(str(tg_path))
            logger.info(f"  TextGrid: {len(words)} words, {gt_duration:.1f}s")

            if not words:
                logger.warning(f"  ✗ No words in TextGrid, skipping")
                results_summary['skipped_missing'] += 1
                continue

            # ---- 2. Extract prompt_text from first ref_duration seconds ----
            prompt_text = extract_text_in_window(words, 0.0, ref_duration)
            if not prompt_text.strip():
                # Fallback: use the first few words regardless of timing
                prompt_text = " ".join(w['text'] for w in words[:10])
            logger.info(f"  Prompt text ({ref_duration}s window): \"{prompt_text[:80]}...\"")

            # ---- 3. Extract synthesis text (up to max_synth_duration) ----
            if gt_duration <= max_synth_duration:
                synth_text = full_text
            else:
                synth_text = extract_text_up_to_duration(words, max_synth_duration)
                logger.info(f"  Text truncated from {gt_duration:.1f}s to ~{max_synth_duration}s")

            if not synth_text.strip():
                logger.warning(f"  ✗ Empty synthesis text, skipping")
                results_summary['skipped_missing'] += 1
                continue

            logger.info(f"  Synth text: \"{synth_text[:80]}{'...' if len(synth_text)>80 else ''}\"")

            # ---- 4. Load reference audio (first ref_duration seconds) ----
            ref_speech = load_wav(str(wav_path), 16000)
            max_ref_samples = int(ref_duration * 16000)
            if ref_speech.shape[1] > max_ref_samples:
                ref_speech = ref_speech[:, :max_ref_samples]

            # ---- 5. Synthesize with CosyVoice2 zero-shot ----
            chunks = split_into_sentences(synth_text, max_words=20)
            logger.info(f"  Synthesizing {len(chunks)} chunk(s)...")

            audio_list, token_list = [], []
            t_synth = time.time()

            for chunk in chunks:
                generator = cosyvoice.inference_zero_shot(
                    chunk, prompt_text, ref_speech, stream=False
                )
                for result in generator:
                    if 'tts_speech' in result:
                        audio_list.append(result['tts_speech'])
                    if 'tts_token' in result:
                        token_list.append(result['tts_token'])

            if not audio_list:
                logger.warning(f"  ✗ No audio generated, skipping")
                results_summary['failed'] += 1
                continue
            if not token_list:
                logger.warning(f"  ✗ No tokens generated, skipping")
                results_summary['failed'] += 1
                continue

            audio_waveform = torch.cat(audio_list, dim=1)   # (1, N)
            speech_tokens  = torch.cat(token_list,  dim=1)   # (1, T_tok)
            audio_duration = audio_waveform.shape[1] / sample_rate
            synth_time = time.time() - t_synth

            logger.info(f"  ✓ Synthesized: {audio_duration:.2f}s in {synth_time:.1f}s  "
                         f"tokens={speech_tokens.shape}")

            # ---- 6. Predict blendshapes from tokens ----
            t_blend = time.time()
            blendshapes = predict_blendshapes(
                models, speech_tokens, audio_duration, fps=fps, device=device
            )
            blend_time = time.time() - t_blend
            logger.info(f"  ✓ Blendshapes: {blendshapes.shape} in {blend_time:.2f}s")

            # ---- 7. Save outputs ----
            # Audio
            audio_path = audio_dir / f"{sample_id}.wav"
            torchaudio.save(str(audio_path), audio_waveform.cpu(), sample_rate)

            # Blendshapes (npz for evaluation compatibility)
            blendshapes_np = blendshapes.cpu().numpy()
            np.savez_compressed(str(blend_out), arkit_blendshapes=blendshapes_np)

            # Blendshapes (pt)
            blend_pt_path = blend_dir / f"{sample_id}.pt"
            torch.save(blendshapes.cpu(), blend_pt_path)

            # Tokens
            tokens_path = token_dir / f"{sample_id}_tokens.pt"
            torch.save(speech_tokens.cpu(), tokens_path)

            # Metadata
            total_time = time.time() - t_sample_start
            metadata = {
                'sample_id':            sample_id,
                'gt_duration_s':        gt_duration,
                'synth_duration_s':     audio_duration,
                'prompt_text':          prompt_text,
                'synth_text':           synth_text,
                'full_text':            full_text,
                'num_words':            len(words),
                'ref_duration_s':       ref_duration,
                'max_synth_duration_s': max_synth_duration,
                'num_chunks':           len(chunks),
                'speech_tokens_shape':  list(speech_tokens.shape),
                'blendshapes_shape':    list(blendshapes.shape),
                'fps':                  fps,
                'sample_rate':          sample_rate,
                'timing': {
                    'total_s':      total_time,
                    'synthesis_s':  synth_time,
                    'blendshape_s': blend_time,
                },
                'outputs': {
                    'audio':       str(audio_path),
                    'blendshapes': str(blend_out),
                    'tokens':      str(tokens_path),
                },
            }
            meta_path = meta_dir / f"{sample_id}.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"  ✓ Saved all outputs ({total_time:.1f}s total)")
            results_summary['processed'] += 1

        except Exception as e:
            logger.error(f"  ✗ FAILED: {e}", exc_info=True)
            results_summary['failed'] += 1
            results_summary['errors'].append({'sample_id': sample_id, 'error': str(e)})

    # ---- Final summary ----
    logger.info(f"\n{'='*70}")
    logger.info("BATCH INFERENCE COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"  Split:            {split}")
    logger.info(f"  Total samples:    {len(sample_ids)}")
    logger.info(f"  Processed:        {results_summary['processed']}")
    logger.info(f"  Skipped (exist):  {results_summary['skipped_existing']}")
    logger.info(f"  Skipped (missing):{results_summary['skipped_missing']}")
    logger.info(f"  Failed:           {results_summary['failed']}")
    logger.info(f"{'='*70}")

    # Save summary
    summary_path = output_dir / "run_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"  Summary saved: {summary_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='BEAT2 test set batch inference: CosyVoice2 + multi-model blendshapes',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Data paths
    parser.add_argument('--split_csv',    type=str, default='../BEAT2/beat_english_v2.0.0/train_test_split.csv',
                        help='Path to BEAT2 train_test_split.csv')
    parser.add_argument('--textgrid_dir', type=str, default='../BEAT2/beat_english_v2.0.0/textgrid',
                        help='Directory containing .TextGrid files')
    parser.add_argument('--wave_dir',     type=str, default='../BEAT2/beat_english_v2.0.0/wave16k',
                        help='Directory containing .wav files (flat or speaker subfolders)')
    parser.add_argument('--split',        type=str, default='test',
                        choices=['test', 'val', 'train', 'additional'],
                        help='Which split to process (default: test)')

    # Model configs
    parser.add_argument(
        '--model', action='append', nargs='+', required=True,
        metavar=('CHECKPOINT', 'REGION'),
        help='Model config: checkpoint_path region1 region2 ... [weight]. '
             'Regions: jaw, mouth, cheek, nose, brow, eye_blink, eye_look, eye_shape. '
             'Weight defaults to 1.0. Repeat --model for each specialist model.'
    )

    # Synthesis parameters
    parser.add_argument('--ref_duration',       type=float, default=10.0,
                        help='Max reference audio duration in seconds (default: 10)')
    parser.add_argument('--max_synth_duration', type=float, default=30.0,
                        help='Max duration of text to synthesize in seconds (default: 30)')
    parser.add_argument('--fps',                type=int,   default=30,
                        help='Blendshape frame rate (default: 30)')

    # Paths and infra
    parser.add_argument('--cosyvoice_model', type=str,
                        default='../pretrained_models/CosyVoice2-0.5B')
    parser.add_argument('--output_dir',      type=str,
                        default='results/beat2_test_synth/')
    parser.add_argument('--device',          type=str, default='cuda')
    parser.add_argument('--no_resume',       action='store_true',
                        help='Re-process all samples even if outputs exist')

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
    print("BEAT2 TEST SET INFERENCE CONFIG")
    print(f"{'='*70}")
    print(f"  Split CSV:          {args.split_csv}")
    print(f"  TextGrid dir:       {args.textgrid_dir}")
    print(f"  Wave dir:           {args.wave_dir}")
    print(f"  Split:              {args.split}")
    print(f"  Ref duration:       {args.ref_duration}s")
    print(f"  Max synth duration: {args.max_synth_duration}s")
    print(f"  FPS:                {args.fps}")
    print(f"  Output dir:         {args.output_dir}")
    print(f"  Resume:             {not args.no_resume}")
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
        ref_duration=args.ref_duration,
        max_synth_duration=args.max_synth_duration,
        fps=args.fps,
        device=args.device,
        resume=not args.no_resume,
    )