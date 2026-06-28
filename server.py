import os
import sys
import json
import asyncio
import io
import wave
import numpy as np
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
from piper.voice import PiperVoice
from config import config
try:
    import webrtcvad
except ImportError:
    try:
        import webrtcvad_wheels as webrtcvad
    except ImportError:
        import webrtcvad_fast as webrtcvad

app = FastAPI(title="Voice AI Real-Time Web Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stt_model = None
tts_voice = None
vad = None

@app.on_event("startup")
def startup_event():
    global stt_model, tts_voice, vad
    print("="*50)
    print("STARTING REAL-TIME VOICE AI SERVER...")
    print("="*50)
    print(f"⚙️ STT Model: {config.WHISPER_MODEL} ({config.WHISPER_DEVICE.upper()})")
    print(f"⚙️ Ollama URL: {config.OLLAMA_URL}")
    print(f"⚙️ Ollama Model: {config.OLLAMA_MODEL}")
    
    try:
        stt_model = WhisperModel(config.WHISPER_MODEL, device=config.WHISPER_DEVICE, compute_type=config.WHISPER_COMPUTE_TYPE)
    except Exception as e:
        print(f"⚠️ GPU load failed ({e}), falling back to CPU...")
        stt_model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")

    if os.path.exists(config.PIPER_MODEL):
        tts_voice = PiperVoice.load(config.PIPER_MODEL)
    vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)

    try:
        requests.post(
            config.OLLAMA_URL,
            json={"model": config.OLLAMA_MODEL, "prompt": "Hi", "stream": False, "keep_alive": -1},
            timeout=10
        )
        print("✅ Ollama pre-warmed in GPU VRAM.")
    except Exception as e:
        print(f"⚠️ Ollama warmup note: {e}")

@app.get("/healthz")
def health_check():
    return {"status": "ok", "gpu": config.WHISPER_DEVICE, "model": config.OLLAMA_MODEL}

@app.get("/")
def get_web_client():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-Time Voice AI Call</title>
        <style>
            body { font-family: system-ui, sans-serif; background: #121212; color: white; text-align: center; padding: 40px; margin: 0; }
            .container { max-width: 700px; margin: 0 auto; background: #1e1e1e; padding: 30px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
            h1 { margin-bottom: 20px; font-size: 28px; }
            button { padding: 16px 36px; font-size: 18px; font-weight: bold; border: none; border-radius: 50px; cursor: pointer; transition: all 0.2s; margin: 10px; }
            #startBtn { background: #007acc; color: white; }
            #startBtn:hover { background: #005999; }
            #stopBtn { background: #e53935; color: white; }
            #stopBtn:disabled, #startBtn:disabled { background: #444; color: #888; cursor: not-allowed; }
            #status { margin: 20px 0; font-size: 18px; font-weight: 500; color: #4caf50; }
            #chat { text-align: left; background: #121212; padding: 20px; border-radius: 12px; height: 280px; overflow-y: auto; font-size: 16px; line-height: 1.5; border: 1px solid #333; margin-bottom: 20px; }
            .user-msg { color: #64b5f6; margin-bottom: 12px; }
            .ai-msg { color: #81c784; margin-bottom: 12px; }
            .input-box { display: flex; gap: 10px; }
            .input-box input { flex: 1; padding: 14px; border-radius: 8px; border: 1px solid #444; background: #2a2a2a; color: white; font-size: 16px; }
            .input-box button { margin: 0; padding: 14px 24px; border-radius: 8px; background: #4caf50; }
            .notice { font-size: 14px; color: #ffca28; margin-top: 15px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎙️ Live Real-Time Voice AI Call</h1>
            <div>
                <button id="startBtn">Start Microphone Call</button>
                <button id="stopBtn" disabled>End Call</button>
            </div>
            <div id="status">Ready</div>
            <div id="chat"></div>
            
            <div class="input-box">
                <input type="text" id="textInput" placeholder="Type a prompt to test instant GPU AI voice streaming..." />
                <button id="sendBtn">Send & Speak</button>
            </div>
        </div>

        <script>
            let ws;
            let audioContext;
            let processor;
            let micStream;
            let audioQueue = [];
            let isPlaying = false;

            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const status = document.getElementById('status');
            const chat = document.getElementById('chat');
            const textInput = document.getElementById('textInput');
            const sendBtn = document.getElementById('sendBtn');

            function appendChat(role, text) {
                const div = document.createElement('div');
                div.className = role === 'User' ? 'user-msg' : 'ai-msg';
                div.innerHTML = `<strong>${role}:</strong> ${text}`;
                chat.appendChild(div);
                chat.scrollTop = chat.scrollHeight;
            }

            async function playNextAudio() {
                if (isPlaying || audioQueue.length === 0) return;
                isPlaying = true;
                const audioData = audioQueue.shift();
                try {
                    if (!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                    if (audioContext.state === 'suspended') await audioContext.resume();
                    
                    const audioBuffer = await audioContext.decodeAudioData(audioData);
                    const source = audioContext.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(audioContext.destination);
                    source.onended = () => {
                        isPlaying = false;
                        playNextAudio();
                    };
                    source.start();
                } catch(e) {
                    isPlaying = false;
                    playNextAudio();
                }
            }

            function connectWebSocket() {
                return new Promise((resolve, reject) => {
                    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                    ws = new WebSocket(`${protocol}//${location.host}/ws/call`);
                    ws.binaryType = 'arraybuffer';
                    ws.onopen = () => resolve();
                    ws.onerror = (err) => reject(err);
                    ws.onmessage = async (event) => {
                        if (typeof event.data === 'string') {
                            const data = JSON.parse(event.data);
                            if (data.type === 'user') appendChat('User', data.text);
                            if (data.type === 'ai') appendChat('AI', data.text);
                        } else {
                            audioQueue.push(event.data);
                            playNextAudio();
                        }
                    };
                    ws.onclose = () => {
                        status.innerText = '🔴 Connection Closed';
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
                    };
                });
            }

            startBtn.onclick = async () => {
                try {
                    await connectWebSocket();
                    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                    if (audioContext.state === 'suspended') await audioContext.resume();
                    
                    micStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
                    const source = audioContext.createMediaStreamSource(micStream);
                    processor = audioContext.createScriptProcessor(4096, 1, 1);

                    processor.onaudioprocess = (e) => {
                        const inputData = e.inputBuffer.getChannelData(0);
                        const int16Data = new Int16Array(inputData.length);
                        for (let i = 0; i < inputData.length; i++) {
                            int16Data[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
                        }
                        if (ws && ws.readyState === WebSocket.OPEN) {
                            ws.send(int16Data.buffer);
                        }
                    };

                    source.connect(processor);
                    processor.connect(audioContext.destination);
                    status.innerText = '🟢 Live Microphone Active - Speak Now!';
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                } catch(err) {
                    console.error("Mic error:", err);
                    status.innerText = '🟢 Text Voice Session Connected';
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                }
            };

            async function sendTextMessage() {
                const txt = textInput.value.trim();
                if (!txt) return;
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    await connectWebSocket();
                }
                appendChat('User', txt);
                ws.send(JSON.stringify({ type: 'text_prompt', text: txt }));
                textInput.value = '';
            }

            sendBtn.onclick = sendTextMessage;
            textInput.onkeypress = (e) => { if (e.key === 'Enter') sendTextMessage(); };

            stopBtn.onclick = () => {
                if (ws) ws.close();
                if (micStream) micStream.getTracks().forEach(track => track.stop());
                if (audioContext) audioContext.close();
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html_content)

SYSTEM_PROMPT = f"You are Samantha, a friendly receptionist at a restaurant called 'Bella Napoli'. Answer the caller in strictly under {config.MAX_RESPONSE_WORDS} words. Keep answers brief, natural, conversational, with no asterisks or markdown."

@app.websocket("/ws/call")
async def websocket_call(websocket: WebSocket):
    await websocket.accept()
    print("\n" + "="*50)
    print("⚡ NEW REAL-TIME VOICE SESSION CONNECTED")
    print("="*50)
    
    audio_buffer = bytearray()
    speech_accumulator = bytearray()
    chat_history = []
    chunk_size = 480 # 30ms at 16kHz
    silence_frames = 0
    max_silence_frames = int((1000 / 30) * config.SILENCE_DURATION_SEC)
    has_spoken = False
    bytes_received_counter = 0
    
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                data = message["bytes"]
                audio_buffer.extend(data)
                bytes_received_counter += len(data)
                
                if bytes_received_counter % 160000 == 0:
                    print(f"📥 Received {bytes_received_counter} bytes of audio from mic stream...", end="\r", flush=True)
                
                while len(audio_buffer) >= chunk_size:
                    frame = bytes(audio_buffer[:chunk_size])
                    audio_buffer = audio_buffer[chunk_size:]
                    
                    is_speech = False
                    try:
                        is_speech = vad.is_speech(frame, 16000)
                    except:
                        pass
                        
                    if is_speech:
                        if not has_spoken:
                            print("\n🎤 Hearing user speaking! Recording speech...", end="\r", flush=True)
                        has_spoken = True
                        silence_frames = 0
                        speech_accumulator.extend(frame)
                    elif has_spoken:
                        silence_frames += 1
                        speech_accumulator.extend(frame)
                        
                    if has_spoken and silence_frames > max_silence_frames:
                        raw_audio = bytes(speech_accumulator)
                        speech_accumulator = bytearray()
                        has_spoken = False
                        silence_frames = 0
                        
                        audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
                        if len(audio_np) > 4000:
                            print("\n🛑 Transcribing user speech with Faster-Whisper GPU...")
                            segments, _ = stt_model.transcribe(audio_np, beam_size=1)
                            user_text = " ".join([s.text for s in segments]).strip()
                            if user_text:
                                print(f"🗣️ User: '{user_text}'")
                                await websocket.send_text(json.dumps({"type": "user", "text": user_text}))
                                await process_ai_voice(user_text, chat_history, websocket)

            elif "text" in message and message["text"]:
                payload = json.loads(message["text"])
                if payload.get("type") == "text_prompt":
                    user_text = payload.get("text", "")
                    print(f"\n💬 Text Prompt Received: '{user_text}'")
                    await process_ai_voice(user_text, chat_history, websocket)

    except WebSocketDisconnect:
        print("\n🔌 Real-time voice session disconnected.\n")

async def process_ai_voice(user_text, chat_history, websocket):
    print("🧠 Querying Ollama LLM (qwen2.5:7b-instruct)...")
    chat_history.append(("User", user_text))
    if len(chat_history) > 8: chat_history = chat_history[-8:]
    
    prompt = SYSTEM_PROMPT + "\n"
    for r, m in chat_history: prompt += f"{r}: {m}\n"
    prompt += "AI:"
    
    res = requests.post(config.OLLAMA_URL, json={"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": True, "keep_alive": -1}, stream=True)
    full_ai_reply = ""
    curr_sentence = ""
    print("🤖 AI: ", end="", flush=True)
    
    for line in res.iter_lines():
        if line:
            try:
                chunk_json = json.loads(line)
                word = chunk_json.get("response", "")
                curr_sentence += word
                full_ai_reply += word
                print(word, end="", flush=True)
                
                if any(p in word for p in ['.', '!', '?', ',']) and len(curr_sentence.strip()) > 2:
                    wav_io = io.BytesIO()
                    with wave.open(wav_io, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(tts_voice.config.sample_rate)
                        for audio_chunk in tts_voice.synthesize(curr_sentence.strip()):
                            wav_file.writeframes(audio_chunk.audio_int16_bytes)
                    await websocket.send_bytes(wav_io.getvalue())
                    curr_sentence = ""
            except:
                continue
    
    if curr_sentence.strip():
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(tts_voice.config.sample_rate)
            for audio_chunk in tts_voice.synthesize(curr_sentence.strip()):
                wav_file.writeframes(audio_chunk.audio_int16_bytes)
        await websocket.send_bytes(wav_io.getvalue())
        
    print("\n👄 TTS Audio chunk sent to client.\n")
    await websocket.send_text(json.dumps({"type": "ai", "text": full_ai_reply.strip()}))
    chat_history.append(("AI", full_ai_reply.strip()))
