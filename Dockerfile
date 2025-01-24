# Use the full development CUDA base image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/app/torch \
    DEMUCS_HOME=/app/models

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    git \
    libffi-dev \
    libssl-dev \
    ffmpeg \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy application files
COPY whisper-demucs-v2-serverless.py requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache models
RUN python3 -c "\
import torch; \
from demucs.pretrained import get_model; \
import whisper; \
print('Downloading and caching models...'); \
get_model('htdemucs_ft'); \
whisper.load_model('large-v3'); \
print('Model caching complete.')"

# Run the serverless handler
CMD ["python3", "whisper-demucs-v2-serverless.py"]
