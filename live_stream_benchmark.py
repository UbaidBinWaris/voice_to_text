import sys
import time
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
from faster_whisper import WhisperModel
import librosa

# Import benchmark functions from our previous script
from benchmark import run_vosk, print_result

# Configuration
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.02  # Increased threshold to ignore background noise (fans, hums)
SILENCE_DURATION = 0.8    # Seconds of silence before processing chunk
MAX_CHUNK_DURATION = 7.0  # Force transcription every 7 seconds even without silence

audio_queue = queue.Queue()
recording = False
stop_app = False
full_audio_data = []

print("\nLoading Faster Whisper model for live stream...")
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
print("✅ Model loaded successfully.")

def audio_callback(indata, frames, time, status):
    if status:
        pass # Ignored status warnings to prevent terminal spam
    if recording:
        audio_queue.put(indata.copy())
        full_audio_data.append(indata.copy())

def process_audio_stream():
    chunk_buffer = []
    silence_frames = 0
    
    while not stop_app:
        try:
            # Block and wait for audio data
            data = audio_queue.get(timeout=0.1)
            chunk_buffer.append(data)
            
            # Simple VAD (Voice Activity Detection)
            rms = np.sqrt(np.mean(data**2))
            
            # Visual mic indicator: prints a dot when it hears clear audio
            if rms > SILENCE_THRESHOLD:
                sys.stdout.write('.')
                sys.stdout.flush()
                silence_frames = 0
            else:
                silence_frames += len(data) / SAMPLE_RATE
                
            current_chunk_duration = len(chunk_buffer) * (len(data) / SAMPLE_RATE)
                
            # If silence exceeds threshold or chunk is too long, process chunk
            if (silence_frames > SILENCE_DURATION and current_chunk_duration > 0.5) or (current_chunk_duration > MAX_CHUNK_DURATION):
                audio_chunk = np.concatenate(chunk_buffer, axis=0).flatten()
                chunk_buffer = []
                silence_frames = 0
                
                # Transcribe chunk
                sys.stdout.write('\r\033[K') # Clear the dots line
                sys.stdout.flush()
                
                segments, _ = model.transcribe(audio_chunk, beam_size=1)
                text = " ".join([segment.text for segment in segments]).strip()
                if text:
                    print(f"\n🎙️ Whisper Live: {text}")
                    print("Listening... ", end="", flush=True)
                    
        except queue.Empty:
            continue

# Start processing thread
processor_thread = threading.Thread(target=process_audio_stream)
processor_thread.start()

print("\n" + "="*50)
print("LIVE STREAM READY")
print("="*50)

# Check microphone and run logic
try:
    print("\nTesting microphone connection...")
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=1024, dtype='float32')
    with stream:
        print("✅ Microphone properly connected and active!")
        print("\nPress [ENTER] to start recording...")
        input() # Wait for enter
        
        recording = True
        print("\n🔴 RECORDING LIVE... Speak into your mic!")
        print("💡 (You will see dots '.' appearing when it hears your voice)")
        print("\nPress [ENTER] again at any time to STOP recording and run benchmarks.")
        print("-" * 50)
        print("Listening... ", end="", flush=True)
        
        input() # Wait for enter to stop
        
        recording = False
        stop_app = True
        print("\n🛑 Stopped recording.")
except Exception as e:
    print(f"\n❌ Error accessing microphone: {e}")
    stop_app = True

processor_thread.join()

# Save full recording
print("\nSaving live recording...")
if len(full_audio_data) > 0:
    final_audio = np.concatenate(full_audio_data, axis=0)
    sf.write("live_recording.wav", final_audio, SAMPLE_RATE)
    print("Saved as 'live_recording.wav'")
    
    duration = librosa.get_duration(path="live_recording.wav")
    print(f"\nLive Audio duration: {duration:.2f} seconds")
    
    print("\n" + "="*50)
    print("Starting Post-Recording Benchmark on saved audio...")
    print("="*50)
    
    print("\nTesting Vosk...")
    try:
        t, lt, it, mem = run_vosk("live_recording.wav")
        print_result("Vosk (en-us)", t, lt, it, mem, duration)
    except Exception as e:
        print(f"Vosk failed: {e}")


else:
    print("No audio captured.")
