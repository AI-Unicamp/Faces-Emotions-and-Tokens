#!/bin/bash
#SBATCH --job-name=facediff_wavtokenizer_100ep_cben
#SBATCH --output=/home/pedro.correa/slurm/facediff_wavtokenizer_100ep_cben_%j.out
#SBATCH --error=/home/pedro.correa/slurm/facediff_wavtokenizer_100ep_cben_%j.err
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=a5000
#SBATCH --cpus-per-task=20

source ~/miniconda3/bin/activate
conda activate face_diffuser

cd /home/pedro.correa/GIPSA_internship/FaceDiffuser/wavtokenizer_adaptation

# Train mouth/jaw model
python main_adapt.py --use_wavtokenizer --use_masking --mask_categories eye_blink eye_look eye_shape brow cheek nose



