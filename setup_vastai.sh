#!/bin/bash
echo "=================================================="
echo " Setting up Low-Latency Voice AI on Vast.ai GPU"
echo "=================================================="

# Update and install system dependencies including python3-dev for C compilation
apt-get update && apt-get install -y ffmpeg libasound2-dev portaudio19-dev python3-pip python3-dev git wget curl

# Upgrade pip and build tools
pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support if not present
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project requirements
pip3 install faster-whisper piper-tts sounddevice requests numpy python-dotenv
pip3 install webrtcvad-wheels || pip3 install webrtcvad

# Ensure piper_models directory exists
mkdir -p piper_models

# Check and download Piper TTS model if missing or incomplete
PIPER_MODEL="piper_models/en_US-ryan-high.onnx"
PIPER_JSON="piper_models/en_US-ryan-high.onnx.json"

if [ ! -f "$PIPER_MODEL" ] || [ $(stat -c%s "$PIPER_MODEL" 2>/dev/null || echo 0) -lt 1000000 ]; then
    echo "⬇️ Downloading Piper TTS Voice Model (Ryan High)..."
    wget -O "$PIPER_MODEL" "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx"
    wget -O "$PIPER_JSON" "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx.json"
fi

# Copy GPU environment configuration
echo "⚙️ Configuring environment for Vast.ai GPU..."
cp -f .env.gpu .env

echo "=================================================="
echo " ✅ Setup Complete! Run 'python3 ai_caller.py'"
echo "=================================================="
