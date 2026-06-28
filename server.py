import os
import sys
import json
import asyncio
import io
import wave
import tempfile
import numpy as np
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
from piper.voice import PiperVoice
from config import config

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

@app.on_event("startup")
def startup_event():
    global stt_model, tts_voice
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
            window.ws = null;
            window.mediaRecorder = null;
            window.micStream = null;
            window.audioContext = null;
            let recordedChunks = [];
            let recordInterval = null;
            let audioQueue = [];
            let isPlaying = false;
            let isAISpeaking = false;

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
                if (isPlaying || audioQueue.length === 0) {
                    if (audioQueue.length === 0) isAISpeaking = false;
                    return;
                }
                isPlaying = true;
                isAISpeaking = true;
                const audioData = audioQueue.shift();
                try {
                    if (!window.audioContext) window.audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    if (window.audioContext.state === 'suspended') await window.audioContext.resume();
                    
                    const audioBuffer = await window.audioContext.decodeAudioData(audioData);
                    const source = window.audioContext.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(window.audioContext.destination);
                    source.onended = () => {
                        isPlaying = false;
                        if (audioQueue.length === 0) isAISpeaking = false;
                        playNextAudio();
                    };
                    source.start();
                } catch(e) {
                    isPlaying = false;
                    if (audioQueue.length === 0) isAISpeaking = false;
                    playNextAudio();
                }
            }

            function connectWebSocket() {
                return new Promise((resolve, reject) => {
                    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                    window.ws = new WebSocket(`${protocol}//${location.host}/ws/call`);
                    window.ws.binaryType = 'arraybuffer';
                    window.ws.onopen = () => resolve();
                    window.ws.onerror = (err) => reject(err);
                    window.ws.onmessage = async (event) => {
                        if (typeof event.data === 'string') {
                            const data = JSON.parse(event.data);
                            if (data.type === 'user') appendChat('User', data.text);
                            if (data.type === 'ai') appendChat('AI', data.text);
                        } else {
                            audioQueue.push(event.data);
                            playNextAudio();
                        }
                    };
                    window.ws.onclose = () => {
                        status.innerText = '🔴 Connection Closed';
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
                    };
                });
            }

            function startRecordingCycle(mimeType) {
                recordedChunks = [];
                window.mediaRecorder = new MediaRecorder(window.micStream, { mimeType: mimeType });
                
                window.mediaRecorder.ondataavailable = (e) => {
                    if (e.data.size > 0 && !isAISpeaking) {
                        recordedChunks.push(e.data);
                    }
                };

                window.mediaRecorder.onstop = () => {
                    if (recordedChunks.length > 0 && !isAISpeaking && window.ws && window.ws.readyState === WebSocket.OPEN) {
                        const blob = new Blob(recordedChunks, { type: mimeType });
                        blob.arrayBuffer().then(buffer => {
                            window.ws.send(buffer);
                        });
                    }
                };

                window.mediaRecorder.start();
            }

            startBtn.onclick = async () => {
                try {
                    await connectWebSocket();
                    window.micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    
                    let mimeType = 'audio/webm';
                    if (!MediaRecorder.isTypeSupported('audio/webm')) {
                        mimeType = 'audio/ogg';
                    }

                    startRecordingCycle(mimeType);

                    recordInterval = setInterval(() => {
                        if (window.mediaRecorder && window.mediaRecorder.state === 'recording') {
                            window.mediaRecorder.stop();
                            startRecordingCycle(mimeType);
                        }
                    }, 2500);

                    status.innerText = '🟢 Live Microphone Active - Speak Now!';
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                } catch(err) {
                    console.error("Mic error:", err);
                    alert("Microphone Access Error: " + err.message);
                }
            };

            async function sendTextMessage() {
                const txt = textInput.value.trim();
                if (!txt) return;
                if (!window.ws || window.ws.readyState !== WebSocket.OPEN) {
                    await connectWebSocket();
                }
                appendChat('User', txt);
                window.ws.send(JSON.stringify({ type: 'text_prompt', text: txt }));
                textInput.value = '';
            }

            sendBtn.onclick = sendTextMessage;
            textInput.onkeypress = (e) => { if (e.key === 'Enter') sendTextMessage(); };

            stopBtn.onclick = () => {
                if (recordInterval) clearInterval(recordInterval);
                if (window.ws) window.ws.close();
                if (window.mediaRecorder && window.mediaRecorder.state !== 'inactive') window.mediaRecorder.stop();
                if (window.micStream) window.micStream.getTracks().forEach(track => track.stop());
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html_content)

SYSTEM_PROMPT = f"You are Samantha, a friendly receptionist at a restaurant called 'Bella Napoli'. Answer the caller in strictly under {config.MAX_RESPONSE_WORDS} words. Keep answers brief, natural, conversational, with no asterisks or markdown."

# Common Whisper hallucinations on quiet background noise
HALLUCINATIONS = ["thank you.", "thanks for watching.", "subtitles by", "thank you!", "you", "bye.", "amara.org"]

@app.websocket("/ws/call")
async def websocket_call(websocket: WebSocket):
    await websocket.accept()
    print("\n" + "="*50)
    print("⚡ NEW REAL-TIME MEDIARECORDER SESSION CONNECTED")
    print("="*50)
    
    chat_history = []
    
    try:
        while True:
            message = await websocket.receive()
            msg_type = message.get("type")
            
            if msg_type == "websocket.disconnect":
                print("\n🔌 WebSocket disconnect message received.")
                break
                
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                
                if len(raw_bytes) > 2000:
                    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as tmp:
                        tmp.write(raw_bytes)
                        tmp.flush()
                        
                        try:
                            segments, _ = stt_model.transcribe(tmp.name, beam_size=1)
                            user_text = " ".join([s.text for s in segments]).strip()
                            
                            # Filter out Whisper silence hallucinations
                            if user_text and user_text.lower() not in HALLUCINATIONS and len(user_text) > 1:
                                print(f"\n🛑 Transcribed voice slice with Faster-Whisper GPU...")
                                print(f"🗣️ User: '{user_text}'")
                                await websocket.send_text(json.dumps({"type": "user", "text": user_text}))
                                await process_ai_voice(user_text, chat_history, websocket)
                        except Exception as stt_err:
                            pass

            elif "text" in message and message["text"]:
                payload = json.loads(message["text"])
                if payload.get("type") == "text_prompt":
                    user_text = payload.get("text", "")
                    print(f"\n💬 Text Prompt Received: '{user_text}'")
                    await process_ai_voice(user_text, chat_history, websocket)

    except WebSocketDisconnect:
        print("\n🔌 Real-time voice session disconnected.\n")
    except Exception as e:
        print(f"\n⚠️ WebSocket handling error: {e}\n")

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
