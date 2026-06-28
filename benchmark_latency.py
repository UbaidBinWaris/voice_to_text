import sys
import time
import numpy as np
import requests
import json
import os
from faster_whisper import WhisperModel
from piper.voice import PiperVoice
from config import config

print("="*50)
print("LOADING MODELS FOR BENCHMARK...")
print("="*50)
print(f"⚙️  STT Model: {config.WHISPER_MODEL} ({config.WHISPER_DEVICE.upper()}, {config.WHISPER_COMPUTE_TYPE})")
print(f"⚙️  Ollama URL: {config.OLLAMA_URL}")
print(f"⚙️  Ollama Model: {config.OLLAMA_MODEL}")
print("="*50)

# Warm up Ollama and keep model permanently loaded in GPU VRAM (keep_alive: -1)
print("🔥 Warming up LLM (loading weights into GPU VRAM)...", end="\r")
try:
    requests.post(
        config.OLLAMA_URL, 
        json={
            "model": config.OLLAMA_MODEL, 
            "prompt": "Hi", 
            "stream": False, 
            "keep_alive": -1,
            "options": {"num_predict": 1}
        }, 
        timeout=30
    )
    print("✅ LLM loaded into GPU VRAM.                          ")
except Exception as e:
    print(f"\n⚠️ Warmup note: {e}")

try:
    stt_model = WhisperModel(config.WHISPER_MODEL, device=config.WHISPER_DEVICE, compute_type=config.WHISPER_COMPUTE_TYPE)
except Exception as e:
    print(f"⚠️ Warning: Failed to load on {config.WHISPER_DEVICE} ({e}). Falling back to CPU int8...")
    stt_model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")

if not os.path.exists(config.PIPER_MODEL):
    print(f"❌ Error: Could not find {config.PIPER_MODEL}.")
    sys.exit(1)
tts_voice = PiperVoice.load(config.PIPER_MODEL)

# Create a test audio file using Piper itself to test Whisper with
print("Generating test audio...")
test_text = "Hello, how are you doing today? I am calling to ask about your services."
test_audio = []
for chunk in tts_voice.synthesize(test_text):
    test_audio.append(chunk.audio_int16_array)
audio_np = np.concatenate(test_audio).flatten().astype(np.float32) / 32768.0

print("\n" + "="*50)
print("🚀 STARTING LATENCY BENCHMARK (Warm GPU)")
print("="*50)

# 1. STT BENCHMARK
start_stt = time.time()
segments, _ = stt_model.transcribe(audio_np, beam_size=1)
user_text = " ".join([segment.text for segment in segments]).strip()
stt_time = time.time() - start_stt

print(f"[STT] Transcribed: '{user_text}'")
print(f"⏱️  STT Processing Time: {stt_time:.3f} seconds\n")

# 2. LLM BENCHMARK
payload = {
    "model": config.OLLAMA_MODEL,
    "prompt": f"You are a helpful AI. Keep your answers brief. User: {user_text}\nAI:",
    "stream": True,
    "keep_alive": -1
}

start_llm = time.time()
try:
    response = requests.post(config.OLLAMA_URL, json=payload, stream=True)
except Exception as e:
    print(f"❌ Cannot reach Ollama at {config.OLLAMA_URL}: {e}")
    sys.exit(1)

first_word_time = 0
first_sentence_time = 0
current_sentence = ""

for line in response.iter_lines():
    if line:
        try:
            chunk_data = json.loads(line)
            word = chunk_data.get("response", "")
        except:
            continue
            
        if not first_word_time and word.strip():
            first_word_time = time.time() - start_llm
            
        current_sentence += word
        
        # Detect end of first sentence
        if any(punct in word for punct in ['.', '!', '?']) and len(current_sentence.strip()) > 3:
            first_sentence_time = time.time() - start_llm
            break # Stop reading from Ollama for the benchmark

print(f"[LLM] First sentence generated: '{current_sentence.strip()}'")
print(f"⏱️  LLM Time to First Word: {first_word_time:.3f} seconds")
print(f"⏱️  LLM Time to Full Sentence: {first_sentence_time:.3f} seconds\n")

# 3. TTS BENCHMARK
start_tts = time.time()
first_audio_chunk_time = 0
for chunk in tts_voice.synthesize(current_sentence.strip()):
    first_audio_chunk_time = time.time() - start_tts
    break # We only care about how fast the first audio bites arrive

print(f"[TTS] Synthesized first chunk of audio.")
print(f"⏱️  TTS Time to First Audio Chunk: {first_audio_chunk_time:.3f} seconds\n")

print("="*50)
print("📊 TOTAL PERCEIVED LATENCY")
print("="*50)
total_hardware_latency = stt_time + first_sentence_time + first_audio_chunk_time

print(f"1. Whisper Transcription:  {stt_time:.3f}s")
print(f"2. LLM Thinking Time:      {first_sentence_time:.3f}s")
print(f"3. Piper Audio Start Time: {first_audio_chunk_time:.3f}s")
print("-" * 50)
print(f"🔥 Hardware Response Time: {total_hardware_latency:.3f}s")
