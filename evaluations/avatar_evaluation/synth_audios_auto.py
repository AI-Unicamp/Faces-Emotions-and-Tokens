#!/usr/bin/env python3
"""
Synthesize VCV intelligibility stimuli with CosyVoice3 and condition each clip
so it feeds cleanly into the BlendshapeDecoder (inference_from_audio.py).

What it does differently from the old flow:
  1. Synthesizes EACH "I said aXa." as its own utterance instead of one long
     sentence  ->  no manual cutting, consistent onset/offset per stimulus.
  2. Keeps the "I said" carrier in the clip  ->  the avatar has a natural
     run-up into the target word, so the target onset is never abrupt, and
     short single-syllable TTS (which CosyVoice handles poorly) is avoided.
  3. Trims to the speech boundaries, then re-pads a FIXED amount of low-level
     room-tone on both sides  ->  every stimulus has identical structure and
     the model gets silence to render a closed/neutral mouth at start and end.
  4. Pads with noise matched to the clip's own floor (estimated before trim),
     not pure digital zeros, which are out-of-distribution for the tokenizer.

Run:
    python synthesize_vcv_stimuli.py                 # all vowel contexts
    python synthesize_vcv_stimuli.py --vowels A I     # subset
    python synthesize_vcv_stimuli.py --pad_ms 400     # change padding
Output:
    results/vcv_stimuli/<VOWEL>/<word>.wav            # one clean clip per VCV
"""
import sys
sys.path.append('third_party/Matcha-TTS')

import argparse
from pathlib import Path

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
MODEL_DIR = '../CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B'
REF_WAV   = '../CosyVoice/asset/wayne_ref.wav'
INSTRUCT  = 'You are a helpful assistant. Please, say this sentence very slowly.<|endofprompt|>'
OUT_ROOT  = Path('results/vcv_stimuli')

CARRIER        = 'I said {w}.'   # set to '{w}.' for bare VCV (less stable in TTS)
DEFAULT_PAD_MS = 300            # fixed silence padding added to EACH side
TRIM_DB        = 35            # frames quieter than (peak - TRIM_DB) are trimmed
MIN_FLOOR      = 1e-4          # amplitude floor for padding noise (~ -80 dBFS)

# Orthographic spellings tuned to coax the right CosyVoice3 pronunciation.
VCV = {
    'A': ['aza', 'ava', 'atah', 'assa', 'apa', 'ajah', 'aga', 'afa', 'ada', 'asha', 'akah', 'aba'],
    'I': ['izi', 'ivi', 'iti', 'issi', 'ipi', 'ijih', 'igui', 'ifi', 'idi', 'ishi', 'iki', 'ibi'],
    'U': ['uzu', 'uvu', 'utu', 'ussu', 'upu', 'ujuh', 'ugu', 'ufu', 'udu', 'ushu', 'uku', 'ubu'],
}


# ---------------------------------------------------------------------
# Audio conditioning helpers  (tensors are (channels, samples))
# ---------------------------------------------------------------------
def to_mono(wav):
    return wav.mean(0, keepdim=True) if wav.shape[0] > 1 else wav


def estimate_noise_floor(wav, sr, edge_ms=50):
    """Mean abs amplitude of the clip edges (CosyVoice's natural leading/
    trailing silence). Call this BEFORE trimming, while the edges are silent."""
    n = max(1, int(sr * edge_ms / 1000))
    edges = torch.cat([wav[:, :n], wav[:, -n:]], dim=1)
    return max(MIN_FLOOR, float(edges.abs().mean()))


def trim_silence(wav, sr, thresh_db=TRIM_DB, frame_ms=20):
    """Energy-based trim of leading/trailing near-silence."""
    x = wav.abs().mean(0)
    frame = max(1, int(sr * frame_ms / 1000))
    n_frames = x.shape[0] // frame
    if n_frames < 2:
        return wav
    rms = x[:n_frames * frame].reshape(n_frames, frame).pow(2).mean(1).sqrt()
    keep = rms > rms.max().clamp(min=1e-9) * (10 ** (-thresh_db / 20))
    idx = torch.where(keep)[0]
    if len(idx) == 0:
        return wav
    start = int(idx[0]) * frame
    end = min((int(idx[-1]) + 1) * frame, wav.shape[1])
    return wav[:, start:end]


def pad_silence(wav, sr, pad_ms, noise_floor):
    n = int(sr * pad_ms / 1000)
    pad = torch.randn(wav.shape[0], n) * noise_floor
    return torch.cat([pad, wav, pad], dim=1)


def condition(wav, sr, pad_ms):
    """Full conditioning chain for one synthesized clip."""
    wav = to_mono(wav)
    floor = estimate_noise_floor(wav, sr)   # estimate before trimming
    wav = trim_silence(wav, sr)
    wav = pad_silence(wav, sr, pad_ms, floor)
    peak = wav.abs().max().clamp(min=1e-9)
    if peak > 1.0:                           # guard against clipping
        wav = wav / peak
    return wav


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Synthesize + condition VCV stimuli.")
    ap.add_argument('--vowels', nargs='+', default=list(VCV.keys()),
                    choices=list(VCV.keys()),
                    help="Vowel contexts to generate (default: all).")
    ap.add_argument('--pad_ms', type=int, default=DEFAULT_PAD_MS,
                    help=f"Silence padding per side, ms (default: {DEFAULT_PAD_MS}).")
    args = ap.parse_args()

    cosyvoice = AutoModel(model_dir=MODEL_DIR)
    sr = cosyvoice.sample_rate

    for vowel in args.vowels:
        out_dir = OUT_ROOT / vowel
        out_dir.mkdir(parents=True, exist_ok=True)

        for w in VCV[vowel]:
            text = CARRIER.format(w=w)

            # stream=False usually yields one chunk; concat to be safe
            chunks = [out['tts_speech'] for out in
                      cosyvoice.inference_instruct2(text, INSTRUCT, REF_WAV, stream=False)]
            wav = torch.cat(chunks, dim=1)

            wav = condition(wav, sr, args.pad_ms)

            out_path = out_dir / f'{w}.wav'
            torchaudio.save(str(out_path), wav, sr)
            print(f'[{vowel}] {w:6s} {wav.shape[1] / sr:5.2f}s -> {out_path}')


if __name__ == '__main__':
    main()