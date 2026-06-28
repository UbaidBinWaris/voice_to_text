# Vast.ai RTX 4090 GPU Setup & Terminal Execution Guide

## Step 1: Connect to Ollama on Vast.ai from Local Machine

If Ollama is running on your remote Vast.ai GPU server on port 11434, open a terminal on your local machine and run SSH port forwarding:

```bash
ssh -L 11434:localhost:11434 -p <VAST_PORT> root@<VAST_IP>
```
*Replace `<VAST_PORT>` and `<VAST_IP>` with your SSH connection details from Vast.ai.*

---

## Step 2: Configure Environment Variables

### Local Machine (CPU mode testing with remote Ollama):
Edit your `.env` file:
```ini
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=your_custom_model_name
WHISPER_MODEL=tiny.en
WHISPER_DEVICE=cpu
```

### Production GPU Server (Full GPU execution on Vast.ai):
On your Vast.ai SSH terminal, clone/upload this repo and copy `.env.gpu` to `.env`:
```bash
cp .env.gpu .env
```
Edit `.env` to set `OLLAMA_MODEL` to your custom model template name!

```ini
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=your_custom_model_name
WHISPER_MODEL=distil-whisper/distil-medium.en
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
```

---

## Step 3: Run the AI Caller via Terminal

On your local machine or remote GPU server terminal, run:

```bash
python3 ai_caller.py
```

### Recommended Upgraded Whisper Models for RTX 4090:
- **`distil-whisper/distil-medium.en`** *(Lightning fast, highly accurate)*
- **`small.en`** *(Super accurate, low memory footprint)*
- **`medium.en`** *(Maximum accuracy for phone speech)*
