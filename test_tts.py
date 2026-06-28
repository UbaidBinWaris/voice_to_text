import time
import wave
import sounddevice as sd
import soundfile as sf
import os
from piper.voice import PiperVoice

model_path = "piper_models/en_US-lessac-high.onnx"

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    print("Please make sure the download has finished.")
    exit(1)

print(f"Loading Piper TTS voice...")
start_time = time.time()
voice = PiperVoice.load(model_path)
print(f"✅ Voice loaded in {time.time() - start_time:.2f} seconds.\n")

text = "Hello! My name is Lessac. I am an ultra realistic text to speech voice running entirely on your local processor. How do I sound?"

output_file = "output_tts.wav"

print(f"Generating audio for text:\n'{text}'")
print("-" * 50)
start_time = time.time()

# Synthesize into a WAV file
with wave.open(output_file, "wb") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(voice.config.sample_rate)
    
    for chunk in voice.synthesize(text):
        wav_file.writeframes(chunk.audio_int16_bytes)

generation_time = time.time() - start_time
print(f"Audio generated in {generation_time:.2f} seconds!")
print(f"Saved audio to: {output_file}")

# Calculate RTF
data, fs = sf.read(output_file, dtype='float32')
audio_length = len(data) / fs

print("-" * 50)
print(f"Audio Length: {audio_length:.2f} seconds")
print(f"Generation Time: {generation_time:.2f} seconds")
print(f"RTF (Real Time Factor): {generation_time / audio_length:.2f}x")
print("-" * 50)

print("\nPlaying audio through your speakers...")
sd.play(data, fs)
sd.wait()
print("Done!")
