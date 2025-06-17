# Use NVIDIA CUDA development base image instead of runtime
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    SUPABASE_URL="" \
    SUPABASE_ANON_KEY=""

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy gen-worker files
COPY gen-worker/ .

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir \
    runpod \
    numpy \
    imageio \
    supabase \
    modelscope \
    opencv-python-headless \
    addict \
    future \
    yapf \
    einops \
    timm \
    safetensors \
    sentencepiece

# Clone and setup Wan2.1
RUN git clone https://github.com/Wan-Video/Wan2.1.git /app/Wan2.1 \
    && cd /app/Wan2.1 \
    && pip3 install --no-cache-dir -r requirements.txt

# Create outputs directory
RUN mkdir -p /app/outputs && chmod -R 777 /app/outputs

ENV PYTHONPATH=/app/Wan2.1:$PYTHONPATH

# Expose port (optional, RunPod handles this)
EXPOSE 8000

# Set the entry point
CMD ["python3", "handler.py"]