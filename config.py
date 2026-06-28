import os

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

class Config:
    # Ollama Settings
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    
    # STT (Faster-Whisper) Settings
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "Systran/faster-distil-whisper-medium.en")
    WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")  # "cuda" or "cpu"
    WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")  # "float16", "int8", "float32"
    
    # TTS (Piper) Settings
    PIPER_MODEL = os.getenv("PIPER_MODEL", "piper_models/en_US-ryan-high.onnx")
    
    # VAD & Audio Settings
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
    MAX_RESPONSE_WORDS = int(os.getenv("MAX_RESPONSE_WORDS", "15"))
    VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "3"))
    SILENCE_DURATION_SEC = float(os.getenv("SILENCE_DURATION_SEC", "0.4"))

config = Config()
