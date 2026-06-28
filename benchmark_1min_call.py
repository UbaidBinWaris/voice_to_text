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
OLLAMA_MODEL = "qwen2:0.5b"
PIPER_MODEL = "piper_models/en_US-ryan-high.onnx"

print("="*50)
print("LOADING MODELS FOR PRODUCTION STRESS TEST...")
print("="*50)

stt_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
tts_voice = PiperVoice.load(PIPER_MODEL)

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
        "model": OLLAMA_MODEL,
        "prompt": f"You are a helpful AI receptionist. Keep answers brief. User: {user_text}\nAI:",
        "stream": True
    }
    start_llm = time.time()
    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
    except:
        print("❌ Ollama is not running! Please start Ollama.")
        exit(1)
        
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
print(f"Best Latency (Shortest):  {min(results):.3f} seconds")
print(f"Worst Latency (Longest):  {max(results):.3f} seconds")
print(f"Average Latency:          {sum(results)/len(results):.3f} seconds")
print("="*50)
print("NOTE: Real-world latency = Hardware Latency + 0.6s (VAD Wait Time).")
