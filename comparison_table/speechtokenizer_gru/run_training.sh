#!/bin/bash
#SBATCH --job-name=cben_blendshape_distill_GRU_100ep
#SBATCH --output=/home/pedro.correa/slurm/cben_blendshape_distill_GRU_100ep_%j.out
#SBATCH --error=/home/pedro.correa/slurm/cben_blendshape_distill_GRU_100ep_%j.err
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --cpus-per-task=20

source ~/miniconda3/bin/activate
conda activate speechtokenizer

cd /home/pedro.correa/GIPSA_internship/FaceDiffuser/speechtokenizer_adaptation

# # Train mouth/jaw model
# python main.py --use_speechtokenizer \
#     --use_masking \
#     --token_path speechtokenizer_tokens/blendshape_distill/ \
#     --mask_categories mouth jaw \
#     --max_epoch 100 \
#     --save_path save/blendshape_distill/mj_blendshape_distill_GRU_100ep

# Train cben model
python main.py --use_speechtokenizer \
    --use_masking \
    --token_path speechtokenizer_tokens/blendshape_distill/ \
    --mask_categories eye_blink eye_look eye_shape brow cheek nose \
    --max_epoch 100 \
    --save_path save/blendshape_distill/cben_blendshape_distill_GRU_100ep



