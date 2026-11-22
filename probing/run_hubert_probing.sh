#!/bin/bash
#SBATCH --job-name=hubert_probing
#SBATCH --output=/home/pedro.correa/slurm/hubert_probing_%j.out
#SBATCH --error=/home/pedro.correa/slurm/hubert_probing_%j.err
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --cpus-per-task=20

source ~/miniconda3/bin/activate
conda activate cosyvoice

cd /home/pedro.correa/GIPSA_internship/probing/hubert_probing

python hubert_probing_analysis.py \
    --features_file hubert_features/ \
    --output_dir results/hubert_probing_test_set \
    --split test
    