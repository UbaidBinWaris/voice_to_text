import time
import os
import psutil
import librosa
import wave
import torch
import gc
import json

AUDIO_FILE = "sample.wav"

def get_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 # MB

def print_result(name, text, load_time, inf_time, memory, audio_length):
    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"Transcript: {text}")
    print(f"Load Time: {load_time:.2f} s")
    print(f"Inference Time: {inf_time:.2f} s")
    print(f"RTF (Real Time Factor): {inf_time/audio_length:.2f}x")
    print(f"Peak Memory: {memory:.2f} MB")
    print(f"{'='*50}")

def run_vosk(audio_file):
    from vosk import Model, KaldiRecognizer
    gc.collect()
    mem_before = get_memory()
    
    start_load = time.time()
    # Vosk auto-downloads small english model if specified
    model = Model(lang="en-us") 
    load_time = time.time() - start_load
    
    wf = wave.open(audio_file, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    
    start_inf = time.time()
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)
    
    res = json.loads(rec.FinalResult())
    text = res.get("text", "")
    inf_time = time.time() - start_inf
    
    mem_after = get_memory()
    
    return text, load_time, inf_time, mem_after - mem_before

def run_faster_whisper(audio_file):
    from faster_whisper import WhisperModel
    gc.collect()
    mem_before = get_memory()
    
    start_load = time.time()
    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    load_time = time.time() - start_load
    
    start_inf = time.time()
    segments, _ = model.transcribe(audio_file, beam_size=1)
    text = " ".join([segment.text for segment in segments])
    inf_time = time.time() - start_inf
    
    mem_after = get_memory()
    
    return text, load_time, inf_time, mem_after - mem_before


if __name__ == "__main__":
    if not os.path.exists(AUDIO_FILE):
        print(f"Audio file {AUDIO_FILE} not found!")
        exit(1)
        
    duration = librosa.get_duration(path=AUDIO_FILE)
    print(f"Audio duration: {duration:.2f} seconds")
    
    # Run tests
    print("Testing Vosk...")
    try:
        t, lt, it, mem = run_vosk(AUDIO_FILE)
        print_result("Vosk (en-us)", t, lt, it, mem, duration)
    except Exception as e:
        print(f"Vosk failed: {e}")
    
    print("Testing Faster Whisper (tiny.en)...")
    try:
        t, lt, it, mem = run_faster_whisper(AUDIO_FILE)
        print_result("Faster Whisper (tiny.en)", t, lt, it, mem, duration)
    except Exception as e:
        print(f"Faster Whisper failed: {e}")
    
