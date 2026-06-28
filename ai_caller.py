import sys
import time
import numpy as np
import sounddevice as sd
import webrtcvad
import requests
import json
import os
from faster_whisper import WhisperModel
from piper.voice import PiperVoice

# CONFIGURATION
SAMPLE_RATE = 16000
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2:0.5b"
PIPER_MODEL = "piper_models/en_US-ryan-high.onnx"
MAX_RESPONSE_WORDS = 20

print("="*50)
print("INITIALIZING LOW-LATENCY AI CALLER...")
print("="*50)

# 1. Load STT
print("Loading Ears (Faster Whisper)...")
stt_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

# 2. Load TTS
print("Loading Mouth (Piper TTS)...")
if not os.path.exists(PIPER_MODEL):
    print(f"❌ Error: Could not find {PIPER_MODEL}.")
    exit(1)
tts_voice = PiperVoice.load(PIPER_MODEL)

# 3. Load VAD
vad = webrtcvad.Vad(3) # 0 to 3 aggression (3 is most aggressive at filtering noise)

# 4. Warm up the LLM (Prevents slow cold-starts on the first turn)
print("Warming up Brain (Loading model into memory)...", end="\r")
try:
    requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": "Hi", "stream": False, "options": {"num_predict": 1}}, timeout=10)
except:
    pass

print("✅ AI Fully Loaded and Ready.                  \n")

SYSTEM_PROMPT = f"You are Samantha, a friendly receptionist at a highly-rated Italian restaurant called 'Bella Napoli'. You are taking a live phone call from a customer. You can help them book a table, ask about the menu, or answer questions about business hours (open 5 PM to 10 PM daily). Your responses MUST be strictly under {MAX_RESPONSE_WORDS} words. Keep your answers brief, natural, and conversational. Do not use emojis, asterisks, or markdown formatting, just plain spoken text.\n\n"
chat_history = []

def ask_ollama_streaming(prompt):
    global chat_history
    chat_history.append(("User", prompt))
    
    # Keep only the last 4 turns (8 messages) to prevent CPU slowdowns!
    if len(chat_history) > 8:
        chat_history = chat_history[-8:]
        
    full_prompt = SYSTEM_PROMPT
    for role, msg in chat_history:
        full_prompt += f"{role}: {msg}\n"
    full_prompt += "AI:"
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": True
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
        response.raise_for_status()
    except Exception as e:
        print(f"\nError: I cannot reach Ollama. Is it running? ({e})")
        return

    current_sentence = ""
    full_reply = ""
    
    # Open streaming audio output (plays instantly)
    out_stream = sd.RawOutputStream(samplerate=tts_voice.config.sample_rate, channels=1, dtype='int16')
    out_stream.start()
    
    for line in response.iter_lines():
        if line:
            try:
                chunk_data = json.loads(line)
            except:
                continue
            word = chunk_data.get("response", "")
            current_sentence += word
            full_reply += word
            
            # Print word so user sees it live
            print(word, end="", flush=True)
            
            # End of sentence detection (streams to voice)
            if any(punct in word for punct in ['.', '!', '?', '\n', ',']) and len(current_sentence.strip()) > 2:
                for audio_chunk in tts_voice.synthesize(current_sentence.strip()):
                    out_stream.write(audio_chunk.audio_int16_bytes)
                current_sentence = ""
                
    # Flush any remaining words
    if current_sentence.strip():
        for audio_chunk in tts_voice.synthesize(current_sentence.strip()):
            out_stream.write(audio_chunk.audio_int16_bytes)
            
    out_stream.stop()
    out_stream.close()
    print("\n")
    chat_history.append(("AI", full_reply.strip()))

def record_audio_vad():
    print("\n" + "-"*50)
    print("Listening... (Speak to start, pause for 0.6s to submit)")
    
    audio_data = []
    chunk_duration_ms = 30 
    chunk_size = int(SAMPLE_RATE * chunk_duration_ms / 1000) # 480 samples
    
    stream = sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=chunk_size)
    stream.start()
    
    silence_frames = 0
    max_silence_frames = int(1000 / chunk_duration_ms) * 0.6 # 0.6 seconds of silence
    max_total_frames = int(1000 / chunk_duration_ms) * 15 # 15 seconds hard limit
    total_frames = 0
    has_spoken = False
    
    while True:
        chunk, overflow = stream.read(chunk_size)
        chunk_bytes = bytes(chunk)
        
        is_speech = False
        try:
            is_speech = vad.is_speech(chunk_bytes, SAMPLE_RATE)
        except:
            pass
            
        if has_spoken:
            total_frames += 1
            if total_frames > max_total_frames:
                break # Hard cutoff after 15s to prevent infinite hanging
                
        if is_speech:
            if not has_spoken:
                print("🎤 Hearing you...", end="\r")
            has_spoken = True
            silence_frames = 0
            audio_data.append(chunk_bytes)
        else:
            if has_spoken:
                audio_data.append(chunk_bytes)
                silence_frames += 1
                if silence_frames > max_silence_frames:
                    break

    stream.stop()
    stream.close()
    print("🛑 Processing...      ")
    
    if not audio_data:
        return np.array([])
        
    raw_bytes = b''.join(audio_data)
    audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_np

print("="*50)
print("STARTING CALL!")
print("="*50)

while True:
    try:
        audio = record_audio_vad()
        if len(audio) < SAMPLE_RATE * 0.5: # Ignore if too short
            continue
            
        # 1. Ears
        segments, _ = stt_model.transcribe(audio, beam_size=1)
        user_text = " ".join([segment.text for segment in segments]).strip()
        
        if not user_text:
            continue
            
        print(f"🗣️ You: {user_text}")
        print(f"🤖 AI:  ", end="", flush=True)
        
        # 2. Brain & 3. Mouth (Streaming concurrently)
        ask_ollama_streaming(user_text)
        
    except KeyboardInterrupt:
        print("\nEnding call. Goodbye!")
        break
    except Exception as e:
        print(f"\nAn error occurred: {e}")
