#!/bin/bash
#SBATCH --job-name=facediff_wavtokenizer_100ep_v1
#SBATCH --output=/home/pedro.correa/slurm/facediff_wavtokenizer_100ep_v1_%j.out
#SBATCH --error=/home/pedro.correa/slurm/facediff_wavtokenizer_100ep_v1_%j.err
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --cpus-per-task=20

source ~/miniconda3/bin/activate
conda activate face_diffuser

cd /home/pedro.correa/GIPSA_internship/FaceDiffuser/cosyvoice_adaptation

# # Infer mouth/jaw model
# python inference_batch.py \
#     --tokens ../data/beat2/train_split_tokens/splits/utt2speech_token_test.pt \
#     --output_dir results/test_set_results_100ep_GRU_CV2_mj \
#     --save_path save/mj_100ep_GRU_COSY \
#     --epoch 100

# Infer cben model
python inference_batch.py \
    --tokens ../data/beat2/train_split_tokens/splits/utt2speech_token_test.pt \
    --output_dir results/test_set_results_100ep_GRU_CV2_all \
    --save_path save/all_100ep_GRU_COSY \
    --epoch 100




