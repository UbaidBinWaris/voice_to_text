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
print("LOADING MODELS FOR PRODUCTION STRESS TEST...")
print("="*50)
print(f"⚙️  STT Model: {config.WHISPER_MODEL} ({config.WHISPER_DEVICE.upper()}, {config.WHISPER_COMPUTE_TYPE})")
print(f"⚙️  Ollama URL: {config.OLLAMA_URL}")
print(f"⚙️  Ollama Model: {config.OLLAMA_MODEL}")
print("="*50)

try:
    stt_model = WhisperModel(config.WHISPER_MODEL, device=config.WHISPER_DEVICE, compute_type=config.WHISPER_COMPUTE_TYPE)
except Exception as e:
    print(f"⚠️ Warning: Failed to load on {config.WHISPER_DEVICE} ({e}). Falling back to CPU int8...")
    stt_model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")

if not os.path.exists(config.PIPER_MODEL):
    print(f"❌ Error: Could not find {config.PIPER_MODEL}.")
    sys.exit(1)
tts_voice = PiperVoice.load(config.PIPER_MODEL)

# We simulate a 1-minute conversation broken into 4 turns of varying lengths.
test_prompts = [
    "Hello.", # Very short (1 sec)
    "I am calling to ask about your business hours.", # Medium (3 sec)
    "I was looking at your website yesterday and I saw that you offer some kind of premium subscription, but I couldn't figure out exactly what it includes. Could you explain it?", # Long (10 sec)
    "Yes, I understand that. But what if I want to cancel my subscription after three months? Are there any hidden fees or penalties that I should be aware of, or is it a rolling monthly contract?" # Very Long (12 sec)
]

print("Generating simulated user voice audio for the test...")
simulated_audio = []
for text in test_prompts:
    audio_chunks = []
    for chunk in tts_voice.synthesize(text):
        audio_chunks.append(chunk.audio_int16_array)
    audio_np = np.concatenate(audio_chunks).flatten().astype(np.float32) / 32768.0
    simulated_audio.append(audio_np)

print("Audio generated! Starting the simulated conversation stress test...\n")

results = []

for i, audio_np in enumerate(simulated_audio):
    audio_length_seconds = len(audio_np) / 16000.0
    print("-" * 50)
    print(f"TURN {i+1} - Simulating User Speaking for {audio_length_seconds:.1f} seconds")
    print("-" * 50)
    
    # 1. STT
    start_stt = time.time()
    segments, _ = stt_model.transcribe(audio_np, beam_size=1)
    user_text = " ".join([segment.text for segment in segments]).strip()
    stt_time = time.time() - start_stt
    print(f"🗣️ Transcribed: '{user_text}'")
    
    # 2. LLM
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": f"You are a helpful AI receptionist. Keep answers brief. User: {user_text}\nAI:",
        "stream": True
    }
    start_llm = time.time()
    try:
        response = requests.post(config.OLLAMA_URL, json=payload, stream=True)
    except:
        print(f"❌ Ollama is not running at {config.OLLAMA_URL}!")
        sys.exit(1)
        
    first_sentence_time = 0
    current_sentence = ""
    for line in response.iter_lines():
        if line:
            try:
                chunk_data = json.loads(line)
                word = chunk_data.get("response", "")
            except:
                continue
            current_sentence += word
            if any(punct in word for punct in ['.', '!', '?', '\n', ',']) and len(current_sentence.strip()) > 3:
                first_sentence_time = time.time() - start_llm
                break 

    # 3. TTS
    start_tts = time.time()
    first_audio_chunk_time = 0
    for chunk in tts_voice.synthesize(current_sentence.strip()):
        first_audio_chunk_time = time.time() - start_tts
        break 

    total_latency = stt_time + first_sentence_time + first_audio_chunk_time
    print(f"⏱️ STT Time: {stt_time:.3f}s")
    print(f"⏱️ LLM Time: {first_sentence_time:.3f}s")
    print(f"⏱️ TTS Time: {first_audio_chunk_time:.3f}s")
    print(f"🔥 Hardware Latency: {total_latency:.3f}s")
    
    results.append(total_latency)

print("\n" + "="*50)
print("📈 STRESS TEST RESULTS (HARDWARE LATENCY)")
print("="*50)
if results:
    avg_latency = sum(results) / len(results)
    print(f"Average Turn Latency: {avg_latency:.3f} seconds")
