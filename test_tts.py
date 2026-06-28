import time
import wave
import sounddevice as sd
import soundfile as sf
import os
import urllib.request
from piper.voice import PiperVoice

def download_voice(voice_id, dest_dir="piper_models"):
    """Automatically downloads a Piper voice from Hugging Face if missing."""
    os.makedirs(dest_dir, exist_ok=True)
    onnx_path = os.path.join(dest_dir, f"{voice_id}.onnx")
    json_path = os.path.join(dest_dir, f"{voice_id}.onnx.json")
    
    if os.path.exists(onnx_path) and os.path.exists(json_path):
        return onnx_path
        
    print(f"\nDownloading new voice '{voice_id}'... (This may take a minute)")
    
    # Parse the voice ID (e.g., en_US-ryan-high)
    try:
        lang_code, name, quality = voice_id.split('-')
        lang_family = lang_code.split('_')[0]
    except ValueError:
        print("Invalid voice ID format. Must be like 'en_US-ryan-high'")
        exit(1)
        
    base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang_family}/{lang_code}/{name}/{quality}/{voice_id}"
    
    try:
        print(f"Downloading model file...")
        import subprocess
        subprocess.run(["wget", "-q", "--show-progress", f"{base_url}.onnx", "-O", onnx_path], check=True)
        print(f"Downloading config file...")
        subprocess.run(["wget", "-q", "--show-progress", f"{base_url}.onnx.json", "-O", json_path], check=True)
        print("Download complete!\n")
    except Exception as e:
        print(f"Failed to download voice: {e}")
        if os.path.exists(onnx_path): os.remove(onnx_path)
        if os.path.exists(json_path): os.remove(json_path)
        exit(1)
        
    return onnx_path

if __name__ == "__main__":
    print("="*50)
    print("TTS Voice Tester")
    print("="*50)
    print("Available Voice Recommendations:")
    print("1. en_US-lessac-high (American Female, clear, what you just heard)")
    print("2. en_US-ryan-high   (American Male, excellent for narration/audiobooks)")
    print("3. en_GB-alba-medium (British Female, crisp accent)")
    print("4. en_US-amy-medium  (American Female, casual)")
    print("5. en_GB-alan-medium (British Male, deep voice)")
    print("="*50)

    # Get voice choice from user
    choice = input("Enter a voice ID or number (default: 2): ").strip()
    if not choice:
        choice = "2"

    voice_map = {
        "1": "en_US-lessac-high",
        "2": "en_US-ryan-high",
        "3": "en_GB-alba-medium",
        "4": "en_US-amy-medium",
        "5": "en_GB-alan-medium"
    }

    # If user typed a number, convert it to the ID
    if choice in voice_map:
        choice = voice_map[choice]

    model_path = download_voice(choice)

    print(f"Loading Piper TTS voice ({choice})...")
    start_time = time.time()
    voice = PiperVoice.load(model_path)
    print(f"✅ Voice loaded in {time.time() - start_time:.2f} seconds.\n")

    text = f"Hello! I am {choice}. I am an ultra realistic text to speech voice running entirely on your local processor. How do I sound?"

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
