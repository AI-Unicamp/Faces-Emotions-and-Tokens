#!/bin/bash
#SBATCH --job-name=avhubert_distil_mj
#SBATCH --output=/home/pedro.correa/slurm/avhubert_distil_mj_%j.out
#SBATCH --error=/home/pedro.correa/slurm/avhubert_distil_mj__%j.err
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --cpus-per-task=20

source ~/miniconda3/bin/activate
conda activate speechtokenizer

cd /home/pedro.correa/GIPSA_internship/SpeechTokenizer/modular

# Train mouth/jaw model
python main.py --config configs.yaml \
    --mask_categories jaw mouth \
    --checkpoint_dir checkpoints/mj_32bs_avhubert \
    --num_epochs 1000 \
    --train_tokens ../speechtokenizer_tokens/avhubert_768D_proj_RVQ8/train_utt2speech_token.pt \
    --val_tokens ../speechtokenizer_tokens/avhubert_768D_proj_RVQ8/val_utt2speech_token.pt \
    --batch_size 32 \
    --log_dir 'logs_speechtokenizer/mj_32bs_avhubert'
        

# # Train /cheek/brow/eyes/nose
# python main.py --config configs.yaml \
#     --mask_categories eye_blink eye_look eye_shape brow cheek nose \
#     --checkpoint_dir checkpoints/cben_32bs_avhubert \
#     --num_epochs 1000 \
#     --train_tokens ../speechtokenizer_tokens/avhubert_768D_proj_RVQ8/train_utt2speech_token.pt \
#     --val_tokens ../speechtokenizer_tokens/avhubert_768D_proj_RVQ8/val_utt2speech_token.pt \
#     --batch_size 32 \
#     --log_dir 'logs_speechtokenizer/cben_32bs_avhubert'