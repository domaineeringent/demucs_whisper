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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy application files
COPY app.py requirements.txt ./
COPY restart.sh /usr/local/bin/restart.sh

# Make restart script executable
RUN chmod +x /usr/local/bin/restart.sh

# Create directories for models
RUN mkdir -p /app/torch /app/models

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache models
RUN python3 -c "import torch; from demucs.pretrained import get_model; model = get_model('htdemucs_ft'); import whisper; model = whisper.load_model('large-v3')"

# Expose the FastAPI port
EXPOSE 8000

# Set the entrypoint to the restart script
ENTRYPOINT ["/usr/local/bin/restart.sh"] 
