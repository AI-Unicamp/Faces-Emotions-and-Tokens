#!/bin/bash
#SBATCH --job-name=probing_wavt_speecht
#SBATCH --output=/home/pedro.correa/slurm/probing_wavt_speecht_%j.out
#SBATCH --error=/home/pedro.correa/slurm/probing_wavt_speecht__%j.err
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --cpus-per-task=20

source ~/miniconda3/bin/activate
conda activate cosyvoice

cd /home/pedro.correa/GIPSA_internship/probing

# python merge_tokens.py

# SpeechTokenizer
python general_analysis_v4.py \
    --tokens_file extracted_tokens/speechtokenizer/utt2speech_token_all.pt \
    --output_dir results/speechtokenizer_test_set_semantic \
    --token_fps 50 \
    --split test

# WavTokenizer
# python general_analysis_v4.py \
#     --tokens_file extracted_tokens/wavtokenizer/utt2speech_token_all.pt \
#     --output_dir results/wavtokenizer_test_set \
#     --token_fps 75 \
#     --split test


# # Coysvoice2
# python general_analysis_v4.py \
#     --tokens_file ../cosyvoice2_decode_v3/all_speakers/splits/utt2speech_token_test.pt \
#     --output_dir results/cosyvoice_test_set \
#     --token_fps 25