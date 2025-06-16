# Use NVIDIA CUDA base image with multi-platform support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    curl \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy gen-worker directory contents
COPY gen-worker/ .

# Install Python dependencies with specific versions for better compatibility
RUN pip3 install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir -r requirements.txt

# Create necessary directories with appropriate permissions
RUN mkdir -p /app/outputs /app/cache /app/tmp \
    && chmod -R 777 /app/outputs /app/cache /app/tmp

# Set environment variables for optimization
ENV TRANSFORMERS_CACHE=/app/cache \
    HF_HOME=/app/cache \
    TORCH_HOME=/app/cache \
    TMPDIR=/app/tmp \
    OMP_NUM_THREADS=4 \
    CUDA_LAUNCH_BLOCKING=0

# Expose port (optional, RunPod handles this)
EXPOSE 8000

# Set the entry point
CMD ["python3", "handler.py"]