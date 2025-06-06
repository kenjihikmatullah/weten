FROM node:18-slim

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      python3 \
      python3-pip \
      git \
      build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN npm install -g n8n

RUN pip3 install --break-system-packages --no-cache-dir TTS

USER node

WORKDIR /home/node

EXPOSE 5678

ENTRYPOINT ["n8n"]
CMD ["start"]
