import sys
import time
import numpy as np
import sounddevice as sd
import wave
import requests
import json
import soundfile as sf
import os
from faster_whisper import WhisperModel
from piper.voice import PiperVoice

# CONFIGURATION
SAMPLE_RATE = 16000
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "my_gemma"
PIPER_MODEL = "piper_models/en_US-ryan-high.onnx"

print("="*50)
print("INITIALIZING AI CALLER...")
print("="*50)

# 1. Load STT
print("Loading Ears (Faster Whisper)...")
stt_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

# 2. Load TTS
print("Loading Mouth (Piper TTS)...")
if not os.path.exists(PIPER_MODEL):
    print(f"❌ Error: Could not find {PIPER_MODEL}.")
    print("Please make sure you have downloaded the Ryan voice.")
    exit(1)
tts_voice = PiperVoice.load(PIPER_MODEL)

print("✅ AI Fully Loaded and Ready.\n")

# Keep track of conversation context
conversation_history = "You are a helpful, conversational AI taking part in a live voice phone call. Keep your answers very brief, friendly, and conversational. Only answer in 1 or 2 sentences max. Do not use emojis or markdown formatting, just plain spoken text.\n\n"

def ask_ollama(prompt):
    global conversation_history
    conversation_history += f"User: {prompt}\nAI:"
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": conversation_history,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        reply = response.json()["response"].strip()
        conversation_history += f" {reply}\n"
        return reply
    except requests.exceptions.ConnectionError:
        return "Error: I cannot reach Ollama. Is it running?"
    except Exception as e:
        return f"Error connecting to brain: {e}"

def speak(text):
    output_file = "temp_response.wav"
    with wave.open(output_file, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(tts_voice.config.sample_rate)
        for chunk in tts_voice.synthesize(text):
            wav_file.writeframes(chunk.audio_int16_bytes)
            
    data, fs = sf.read(output_file, dtype='float32')
    sd.play(data, fs)
    sd.wait()

def record_audio():
    print("\n" + "-"*50)
    print("Press [ENTER] to start speaking...")
    input()
    print("🔴 RECORDING... Speak now! Press [ENTER] when finished.")
    
    audio_data = []
    recording = True
    
    def callback(indata, frames, time, status):
        if recording:
            audio_data.append(indata.copy())
            
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=1024, dtype='float32')
    with stream:
        input() # wait for enter to stop
        recording = False
        
    print("🛑 Processing...")
    if not audio_data:
        return np.array([])
    return np.concatenate(audio_data, axis=0).flatten()

# Main Loop
print("="*50)
print("STARTING CALL!")
print("Make sure you have Ollama running in the background!")
print("="*50)

while True:
    try:
        audio = record_audio()
        if len(audio) < SAMPLE_RATE * 0.5: # Ignore if too short
            continue
            
        # 1. Ears (Listen)
        segments, _ = stt_model.transcribe(audio, beam_size=1)
        user_text = " ".join([segment.text for segment in segments]).strip()
        
        if not user_text:
            continue
            
        print(f"🗣️ You: {user_text}")
        
        # 2. Brain (Think)
        print("[Thinking...] Asking Llama 3...")
        ai_reply = ask_ollama(user_text)
        print(f"🤖 AI:  {ai_reply}")
        
        # 3. Mouth (Speak)
        if "Error:" not in ai_reply:
            speak(ai_reply)
        
    except KeyboardInterrupt:
        print("\nEnding call. Goodbye!")
        break
    except Exception as e:
        print(f"\nAn error occurred: {e}")
