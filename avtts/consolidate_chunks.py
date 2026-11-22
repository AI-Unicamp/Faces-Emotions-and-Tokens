#!/usr/bin/env python3
"""
Consolidate per-chunk token files into a single {utt_id: tensor} dict.

Input:  tokens_dir with files like 1_wayne_0_1_1_chunk_0_tokens.pt
Output: single .pt file with {\"1_wayne_0_1_1_chunk_0\": tensor, ...}
"""

import torch
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens_dir", type=str, required=True,
                        help="Directory with *_tokens.pt files")
    parser.add_argument("--output", type=str, default="utt2speech_token_synth.pt",
                        help="Output .pt file")
    args = parser.parse_args()

    tokens_dir = Path(args.tokens_dir)
    token_files = sorted(tokens_dir.glob("*_tokens.pt"))

    # Exclude _all_tokens.pt files (concatenated per-sample files)
    token_files = [f for f in token_files if "_all_tokens" not in f.stem]

    print(f"Found {len(token_files)} token files in {tokens_dir}")

    utt2tok = {}
    for f in token_files:
        # Remove _tokens suffix to get utterance ID
        utt_id = f.stem.replace("_tokens", "")
        t = torch.load(f, map_location="cpu")
        # Flatten to 1D if (1, T)
        if t.dim() == 2:
            t = t.squeeze(0)
        utt2tok[utt_id] = t
        
    print(f"Consolidated {len(utt2tok)} utterances")
    if utt2tok:
        sample_key = next(iter(utt2tok))
        print(f"  Sample: {sample_key} -> shape {utt2tok[sample_key].shape}")

    torch.save(utt2tok, args.output)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()