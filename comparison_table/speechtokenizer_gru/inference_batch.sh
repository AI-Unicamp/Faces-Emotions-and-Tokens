#!/bin/bash
#SBATCH --job-name=inf_cben_GRU_frozen
#SBATCH --output=/home/pedro.correa/slurm/inf_cben_GRU_frozen_%j.out
#SBATCH --error=/home/pedro.correa/slurm/inf_cben_GRU_frozen_%j.err
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --cpus-per-task=20

source ~/miniconda3/bin/activate
conda activate speechtokenizer

cd /home/pedro.correa/GIPSA_internship/FaceDiffuser/speechtokenizer_adaptation

# Infer mouth/jaw model
# python inference_batch.py \
#     --save_path save/frozen/mj_100ep_GRU \
#     --tokens speechtokenizer_tokens/frozen/test_utt2speech_token.pt \
#     --output_dir results/test_set_results_100ep_GRU_frozen_mj \
#     --epoch 100

# Infer cben model
python inference_batch.py \
    --save_path save/frozen/cben_100ep_GRU \
    --tokens speechtokenizer_tokens/frozen/test_utt2speech_token.pt \
    --output_dir results/test_set_results_100ep_GRU_frozen_cben \
    --epoch 100



