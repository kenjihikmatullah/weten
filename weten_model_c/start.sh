#!/bin/bash

echo "🚀 Starting AI Image Generator Stack..."

# Check if models exist, download if not
if [ ! -f "/app/comfyui/models/checkpoints/v1-5-pruned-emaonly.ckpt" ]; then
    echo "📥 Downloading Stable Diffusion 1.5 model..."
    /app/download-models.sh
fi

# Wait a moment for Redis to start
sleep 2

# Start all services via supervisor
echo "🔧 Starting services..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf