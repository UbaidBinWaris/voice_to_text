# Voice to Text AI Real-Time Web Server

A real-time voice AI server built with FastAPI, using WebSockets for streaming audio. It uses Faster-Whisper for Speech-to-Text (STT), Ollama for LLM generation, and Piper for Text-to-Speech (TTS).

## Features
- Real-time two-way voice communication via WebSockets
- Browser-based microphone interface
- Local STT using Faster-Whisper
- Local LLM via Ollama
- Local TTS using Piper

## Prerequisites

1. **Python 3.8+**
2. **Ollama**: Install and run [Ollama](https://ollama.com/) locally to serve the LLM.
   ```bash
   # Pull the default model
   ollama run qwen2.5:7b-instruct
   ```
3. **Piper Models**: You need to download the Piper ONNX models and place them in the `piper_models` directory. The default configuration looks for `piper_models/en_US-ryan-high.onnx`.

## Installation

1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   **Important**: Make sure you have activated the virtual environment before installing the dependencies or running the scripts.
   ```bash
   pip install -r requirements.txt
   ```

3. Configure Environment Variables:
   Copy the example environment file and customize it if needed:
   ```bash
   cp .env.example .env
   ```

## Running the Server

Start the FastAPI server using Uvicorn:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Once the server is running, open your web browser and navigate to:
[http://localhost:8000](http://localhost:8000)

## Configuration

You can configure the models and hardware settings in the `.env` file or directly in `config.py`.

- `OLLAMA_MODEL`: The LLM to use (default: `qwen2.5:7b-instruct`)
- `WHISPER_MODEL`: The Faster-Whisper model (default: `Systran/faster-distil-whisper-medium.en`)
- `WHISPER_DEVICE`: Set to `cuda` for GPU or `cpu` for CPU execution.
- `PIPER_MODEL`: Path to the Piper ONNX voice file.
