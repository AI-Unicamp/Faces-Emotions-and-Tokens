#!/bin/bash
#SBATCH --job-name=ttsf_synthetic_beat2_data
#SBATCH --output=/home/pedro.correa/slurm/ttsf_synthetic_beat2_data_%j.out
#SBATCH --error=/home/pedro.correa/slurm/ttsf_synthetic_beat2_data_%j.err
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --cpus-per-task=20

source ~/miniconda3/bin/activate
conda activate cosyvoice

cd /home/pedro.correa/GIPSA_internship/tts_pipeline

# python inference.py \
#     --model ../cosyvoice2_decode_v3/modular/checkpoints/mouth_jaw_1000ep_4bs_no_overlap/jaw_mouth_model_epoch_999.pth jaw mouth 1.0 \
#     --model ../cosyvoice2_decode_v3/modular/checkpoints/cben_1000ep_32bs_no_overlap/eye_blink_eye_look_eye_shape_brow_cheek_nose_model_epoch_999.pth eye_blink eye_look eye_shape brow cheek nose 1.0 \
#     --split test \
#     --ref_duration 10.0 \
#     --max_synth_duration 30.0 \
#     --cosyvoice_model ../cosyvoice2_decode_v3/pretrained_models/CosyVoice2-0.5B \
#     --output_dir results/beat2_test_synth/

python inference_chunks.py \
    --model ../cosyvoice2_decode_v3/modular/checkpoints/mouth_jaw_1000ep_4bs_no_overlap/jaw_mouth_model_epoch_999.pth jaw mouth 1.0 \
    --model ../cosyvoice2_decode_v3/modular/checkpoints/cben_1000ep_32bs_no_overlap/eye_blink_eye_look_eye_shape_brow_cheek_nose_model_epoch_999.pth eye_blink eye_look eye_shape brow cheek nose 1.0 \
    --split test \
    --chunk_duration 10.0 \
    --cosyvoice_model ../cosyvoice2_decode_v3/pretrained_models/CosyVoice2-0.5B \
    --output_dir results_10s_chunks/beat2_test_synth/