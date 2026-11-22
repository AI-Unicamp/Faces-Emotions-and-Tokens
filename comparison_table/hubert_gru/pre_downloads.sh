#!/bin/bash
# Download using git lfs

MODELS_DIR="./pretrained_models/hubert"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

echo "Installing git-lfs if needed..."
git lfs install  # Uncomment if not installed

echo ""
echo "1. Cloning HuBERT base model..."
git clone https://huggingface.co/facebook/hubert-base-ls960
cd hubert-base-ls960
git lfs pull
cd ..

echo ""
echo "2. Cloning Wav2Vec2 processor..."
git clone https://huggingface.co/facebook/hubert-xlarge-ls960-ft
cd hubert-xlarge-ls960-ft
git lfs pull
cd ..

echo ""
echo "✅ Download complete!"