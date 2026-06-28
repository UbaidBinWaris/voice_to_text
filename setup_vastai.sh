#!/bin/bash
echo "=================================================="
echo " Setting up Low-Latency Voice AI on Vast.ai GPU"
echo "=================================================="

# Update and install system dependencies including python3-dev for C compilation
apt-get update && apt-get install -y ffmpeg libasound2-dev portaudio19-dev python3-pip python3-dev git

# Upgrade pip and build tools
pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support if not present
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project requirements (using webrtcvad-wheels for pre-built binaries, falling back to webrtcvad)
pip3 install faster-whisper piper-tts sounddevice requests numpy python-dotenv
pip3 install webrtcvad-wheels || pip3 install webrtcvad

# Copy GPU environment configuration if no .env exists
if [ ! -f .env ]; then
    echo "Copying .env.gpu to .env..."
    cp .env.gpu .env
fi

echo "=================================================="
echo " ✅ Setup Complete! Run 'python3 ai_caller.py'"
echo "=================================================="
