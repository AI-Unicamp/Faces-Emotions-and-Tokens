#!/usr/bin/env python3
"""
Download HuBERT model for offline use
Run this on the head node with internet access
"""
from transformers import HubertModel, Wav2Vec2Processor
import os

# Directory to save models
MODELS_DIR = "pretrained_models/hubert"
os.makedirs(MODELS_DIR, exist_ok=True)

print("Downloading HuBERT model...")
model = HubertModel.from_pretrained(
    "facebook/hubert-base-ls960",
    cache_dir=MODELS_DIR
)
print(f"✓ Model saved to {MODELS_DIR}")

print("\nDownloading Wav2Vec2 Processor (used for audio preprocessing)...")
processor = Wav2Vec2Processor.from_pretrained(
    "facebook/hubert-xlarge-ls960-ft",
    cache_dir=MODELS_DIR
)
print(f"✓ Processor saved to {MODELS_DIR}")

print("\nAlternative: Save models directly")
model_path = os.path.join(MODELS_DIR, "hubert-base-ls960")
processor_path = os.path.join(MODELS_DIR, "hubert-xlarge-ls960-ft")

model.save_pretrained(model_path)
processor.save_pretrained(processor_path)

print(f"\n✅ All models downloaded!")
print(f"Model location: {model_path}")
print(f"Processor location: {processor_path}")