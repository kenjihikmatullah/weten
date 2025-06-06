FROM n8nio/n8n:latest

# Install ffmpeg
USER root
RUN apk update && \
    apk add ffmpeg espeak-ng && \
    rm -rf /var/cache/apk/*

USER node
