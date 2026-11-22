#!/bin/bash
#SBATCH --job-name=inference_facediff_wavtokenizer_100ep_mj
#SBATCH --output=/home/pedro.correa/slurm/inference_facediff_wavtokenizer_100ep_mj_%j.out
#SBATCH --error=/home/pedro.correa/slurm/inference_facediff_wavtokenizer_100ep_mj_%j.err
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=a5000
#SBATCH --cpus-per-task=20

source ~/miniconda3/bin/activate
conda activate face_diffuser

cd /home/pedro.correa/GIPSA_internship/FaceDiffuser/wavtokenizer_adaptation

# Infer mouth/jaw model
python inference_batch.py \
    --tokens wavtokenizer_tokens/test_utt2speech_token.pt \
    --output_dir results/test_set_results_100ep_GRU_WAV_mj \
    --save_path save/mj_100ep_GRU_WAV \
    --epoch 100

# # Infer cben model
# python inference_batch.py \
#     --tokens wavtokenizer_tokens/test_utt2speech_token.pt \
#     --output_dir results/test_set_results_100ep_GRU_WAV_cben \
#     --save_path save/cben_100ep_GRU_WAV \
#     --epoch 100



