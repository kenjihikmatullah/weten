# Minimal ComfyUI setup for debugging
FROM python:3.10-slim

# Install minimal dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git

# Install Python deps
WORKDIR /app/ComfyUI
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Create model directory
RUN mkdir -p models/checkpoints

# Download minimal model
RUN wget -O models/checkpoints/v1-5-pruned-emaonly.ckpt \
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"

# Expose port
EXPOSE 8188

# Start ComfyUI
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--cpu"]