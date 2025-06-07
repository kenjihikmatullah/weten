FROM node:18-slim

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-pip python3-packaging git \
      ffmpeg build-essential libsndfile1 libsndfile1-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN npm install -g n8n

RUN pip3 install --break-system-packages --no-cache-dir \
      torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN git clone https://github.com/Wan-Video/Wan2.1.git /opt/Wan2.1 && \
    sed -i '/flash_attn/d' /opt/Wan2.1/requirements.txt && \
    pip3 install --break-system-packages --no-cache-dir -r /opt/Wan2.1/requirements.txt

USER node
WORKDIR /home/node
EXPOSE 5678
ENTRYPOINT ["n8n"]
CMD ["start"]
