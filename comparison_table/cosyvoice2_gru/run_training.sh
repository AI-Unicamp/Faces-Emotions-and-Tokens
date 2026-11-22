#!/bin/bash
#SBATCH --job-name=cv2_gru_all
#SBATCH --output=/home/pedro.correa/slurm/cv2_gru_all_%j.out
#SBATCH --error=/home/pedro.correa/slurm/cv2_gru_all_%j.err
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate face_diffuser

cd /home/pedro.correa/GIPSA_internship/FaceDiffuser/cosyvoice_adaptation

# # Train mouth/jaw model
# python main_adapt.py --dataset beat2 --use_cosyvoice_tokens --use_masking --mask_categories eye_blink eye_look eye_shape brow cheek nose

# Train mouth/jaw model
python main_adapt.py --dataset beat2 --use_cosyvoice_tokens --max_epoch 100 --save_path save/all_100ep_GRU_COSY



