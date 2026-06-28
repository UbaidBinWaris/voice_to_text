import sys
import time
import numpy as np
import requests
import json
import os
from faster_whisper import WhisperModel
from piper.voice import PiperVoice

# CONFIGURATION
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "my_gemma"
PIPER_MODEL = "piper_models/en_US-ryan-high.onnx"

print("="*50)
print("LOADING MODELS FOR BENCHMARK...")
print("="*50)

stt_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
tts_voice = PiperVoice.load(PIPER_MODEL)

# Create a test audio file using Piper itself to test Whisper with
print("Generating test audio...")
test_text = "Hello, how are you doing today? I am calling to ask about your services."
test_audio = []
for chunk in tts_voice.synthesize(test_text):
    test_audio.append(chunk.audio_int16_array)
audio_np = np.concatenate(test_audio).flatten().astype(np.float32) / 32768.0

print("\n" + "="*50)
print("🚀 STARTING LATENCY BENCHMARK")
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
    "model": OLLAMA_MODEL,
    "prompt": f"You are a helpful AI. Keep your answers brief. User: {user_text}\nAI:",
    "stream": True
}

start_llm = time.time()
response = requests.post(OLLAMA_URL, json=payload, stream=True)

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
print(f"2. Gemma Thinking Time:    {first_sentence_time:.3f}s")
print(f"3. Piper Audio Start Time: {first_audio_chunk_time:.3f}s")
print("-" * 50)
print(f"🔥 Hardware Response Time: {total_hardware_latency:.3f}s")
print("\nNOTE: In the live caller, the VAD (Silence Detector) waits exactly 1.500s")
print("after you stop speaking before it triggers. To make it feel faster, we can")
print("reduce the VAD wait time to 0.7s or 0.8s, and trigger TTS on commas (,) as well!")
