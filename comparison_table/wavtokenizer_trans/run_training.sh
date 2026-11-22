#!/bin/bash
#SBATCH --job-name=cben_wavtokenizer_1000ep
#SBATCH --output=/home/pedro.correa/slurm/cben_wavtokenizer_1000ep_%j.out
#SBATCH --error=/home/pedro.correa/slurm/cben_wavtokenizer_1000ep_%j.err
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --cpus-per-task=20


source ~/miniconda3/bin/activate
conda activate wavtokenizer

cd /home/pedro.correa/GIPSA_internship/wavtokenizer_decoder/modular

# python main.py

# python inference.py \
#     --checkpoint checkpoints/test_7_checkpoint_epoch_999.pth \
#     --tokens ../wayne/wayne_tokens/utt2speech_token_test.pt \
#     --output results/test_wayne7_1000.npz \
#     --target_fps 30 \
#     --device cuda

# # Train mouth/jaw model
# python main.py --config configs.yaml \
#     --mask_categories jaw mouth \
#     --checkpoint_dir checkpoints/mouth_jaw_wavtokenizer_1000ep

# Train /cheek/brow/eyes/nose
python main.py --config configs.yaml \
    --mask_categories eye_blink eye_look eye_shape brow cheek nose \
    --checkpoint_dir checkpoints/cben_wavtokenizer_1000ep

# # Train all regions
# python main.py --config configs.yaml \
#     --mask_categories all \
#     --checkpoint_dir checkpoints/all_1000ep_32bs_no_overlap
