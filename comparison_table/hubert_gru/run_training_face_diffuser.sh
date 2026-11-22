#!/bin/bash
#SBATCH --job-name=cben_face_diffuser_beat2
#SBATCH --output=/home/pedro.correa/slurm/cben_face_diffuser_beat2_%j.out
#SBATCH --error=/home/pedro.correa/slurm/cben_face_diffuser_beat2_%j.err
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate face_diffuser

cd /home/pedro.correa/GIPSA_internship/FaceDiffuser

# python main.py --dataset beat2 --save_path sanity_check_split/face_diffuser_beat2 --max_epoch 100

# python main.py --dataset beat2 --use_masking --mask_categories jaw mouth --save_path save_region_specific/mj_face_diffuser_beat2 --max_epoch 100

python main.py --dataset beat2 --use_masking --mask_categories eye_blink eye_look eye_shape brow cheek nose --save_path save_region_specific/cben_face_diffuser_beat2 --max_epoch 100