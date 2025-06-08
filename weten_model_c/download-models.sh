#!/bin/bash

echo "📥 Downloading AI models..."

# Create model directories
mkdir -p /app/comfyui/models/checkpoints
mkdir -p /app/comfyui/models/vae

# Download Stable Diffusion 1.5 (lighter model for MacBook)
echo "⬇️  Downloading Stable Diffusion 1.5..."
wget -O /app/comfyui/models/checkpoints/v1-5-pruned-emaonly.ckpt \
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"

# Download VAE (optional but recommended)
echo "⬇️  Downloading VAE..."
wget -O /app/comfyui/models/vae/vae-ft-mse-840000-ema-pruned.ckpt \
    "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt"

echo "✅ Models downloaded successfully!"