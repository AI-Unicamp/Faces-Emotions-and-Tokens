#!/bin/bash
#SBATCH --job-name=all_1000ep_hubert_trans
#SBATCH --output=/home/pedro.correa/slurm/all_1000ep_hubert_trans_%j.out
#SBATCH --error=/home/pedro.correa/slurm/all_1000ep_hubert_trans_%j.err
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate cosyvoice_hubert

cd /home/pedro.correa/GIPSA_internship/cosyvoice2_decode_v3/modular_hubert

# python main.py

# python inference.py \
#     --checkpoint checkpoints/test_7_checkpoint_epoch_999.pth \
#     --tokens ../wayne/wayne_tokens/utt2speech_token_test.pt \
#     --output results/test_wayne7_1000.npz \
#     --target_fps 30 \
#     --device cuda

# # Train mouth/jaw model
# python main.py --config configs.yaml \
#     --resume mouth_jaw_1000ep_4bs_no_overlap_hubert_encode/jaw_mouth_model_epoch_499.pth \
#     --mask_categories jaw mouth \
#     --checkpoint_dir checkpoints \
#     --num_epochs 1000 \
#     --checkpoint_prefix jaw_mouth_model

# # Train /cheek/brow/eyes/nose
# python main.py --config configs.yaml \
#     --resume cben_1000ep_8bs_no_overlap_hubert_encode/eye_blink_eye_look_eye_shape_brow_cheek_nose_model_epoch_499.pth \
#     --mask_categories eye_blink eye_look eye_shape brow cheek nose \
#     --checkpoint_dir checkpoints \
#     --num_epochs 1000 \
#     --checkpoint_prefix eye_blink_eye_look_eye_shape_brow_cheek_nose_model

# # Train all regions (4bs e 500ep)
# python main.py --config configs.yaml \
#     --mask_categories all \
#     --checkpoint_dir checkpoints/all_500ep_4bs_no_overlap_hubert_encode

# Train all regions (8bs e 1000ep, igual do combined)
python main.py \
    --config configs.yaml \
    --resume all_model_epoch_199.pth \
    --mask_categories all \
    --num_epochs 1000 \
    --checkpoint_dir checkpoints/all_1000ep_8bs_no_overlap_hubert_encode/ \
    --checkpoint_prefix all_model
