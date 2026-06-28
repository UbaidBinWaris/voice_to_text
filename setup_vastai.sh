#!/bin/bash
echo "=================================================="
echo " Setting up Low-Latency Voice AI on Vast.ai GPU"
echo "=================================================="

# Update and install system audio dependencies
apt-get update && apt-get install -y ffmpeg libasound2-dev portaudio19-dev python3-pip git

# Upgrade pip
pip3 install --upgrade pip

# Install PyTorch with CUDA support if not present
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project requirements
pip3 install faster-whisper piper-tts webrtcvad sounddevice requests numpy python-dotenv

# Copy GPU environment configuration
if [ ! -f .env ]; then
    echo "Copying .env.gpu to .env..."
    cp .env.gpu .env
fi

echo "=================================================="
echo " ✅ Setup Complete! Run 'python3 ai_caller.py'"
echo "=================================================="
