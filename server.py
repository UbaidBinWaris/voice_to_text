import os
import sys
import json
import asyncio
import base64
import numpy as np
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
from piper.voice import PiperVoice
from config import config

app = FastAPI(title="Voice AI Production Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models
stt_model = None
tts_voice = None

@app.on_event("startup")
def startup_event():
    global stt_model, tts_voice
    print("="*50)
    print("STARTING VOICE AI TELEPHONY & WEB SERVER...")
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

# TWILIO TELEPHONY INTEGRATION
@app.post("/twilio/voice")
async def twilio_voice_webhook(request: Request):
    """
    Twilio Webhook endpoint: Called when a real customer dials your Twilio Phone Number.
    Instructs Twilio to open a bi-directional audio WebSocket stream to our GPU server.
    """
    host = request.headers.get("host")
    twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Connecting to Bella Napoli AI Receptionist.</Say>
    <Connect>
        <Stream url="wss://{host}/twilio/stream" />
    </Connect>
</Response>"""
    return Response(content=twiml_response, media_type="application/xml")

@app.websocket("/twilio/stream")
async def twilio_audio_stream(websocket: WebSocket):
    """
    Twilio Media Stream WebSocket: Handles live 8kHz mu-law audio streams directly from phone calls.
    """
    await websocket.accept()
    print("📞 REAL PHONE CALL CONNECTED VIA TWILIO!")
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            if data['event'] == 'start':
                print("🏁 Phone call stream started.")
            elif data['event'] == 'media':
                # Incoming raw phone audio payload (base64 mu-law)
                pass
            elif data['event'] == 'stop':
                print("🛑 Phone call ended.")
                break
    except WebSocketDisconnect:
        print("🔌 Phone call disconnected.")

# WEB BROWSER INTERFACE
@app.get("/")
def get_web_client():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voice AI Agent</title>
        <style>
            body { font-family: sans-serif; background: #121212; color: white; text-align: center; padding: 50px; }
            button { padding: 15px 30px; font-size: 18px; border: none; border-radius: 8px; background: #007acc; color: white; cursor: pointer; }
            button:disabled { background: #555; }
            #status { margin-top: 20px; font-size: 20px; color: #4caf50; }
            #log { margin-top: 30px; text-align: left; max-width: 600px; margin-left: auto; margin-right: auto; background: #1e1e1e; padding: 20px; border-radius: 8px; height: 300px; overflow-y: auto; }
        </style>
    </head>
    <body>
        <h1>🎙️ Low-Latency Voice AI Receptionist</h1>
        <button id="startBtn">Start Voice Call</button>
        <button id="stopBtn" disabled>End Call</button>
        <div id="status">Disconnected</div>
        <div id="log"></div>

        <script>
            let ws;
            let audioContext;
            let processor;
            let micStream;

            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const status = document.getElementById('status');
            const log = document.getElementById('log');

            function appendLog(msg) {
                log.innerHTML += '<div>' + msg + '</div>';
                log.scrollTop = log.scrollHeight;
            }

            startBtn.onclick = async () => {
                const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${location.host}/ws/call`);
                
                ws.onopen = async () => {
                    status.innerText = 'Connected & Listening...';
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    appendLog('⚡ Voice session started.');

                    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    const source = audioContext.createMediaStreamSource(micStream);
                    processor = audioContext.createScriptProcessor(4090, 1, 1);

                    processor.onaudioprocess = (e) => {
                        const inputData = e.inputBuffer.getChannelData(0);
                        const int16Data = new Int16Array(inputData.length);
                        for (let i = 0; i < inputData.length; i++) {
                            int16Data[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
                        }
                        if (ws.readyState === WebSocket.OPEN) {
                            ws.send(int16Data.buffer);
                        }
                    };

                    source.connect(processor);
                    processor.connect(audioContext.destination);
                };

                ws.onmessage = async (event) => {
                    if (typeof event.data === 'string') {
                        const data = JSON.parse(event.data);
                        if (data.type === 'transcript') appendLog('🗣️ You: ' + data.text);
                        if (data.type === 'text') appendLog('🤖 AI: ' + data.text);
                    } else {
                        const audioBuffer = await audioContext.decodeAudioData(event.data);
                        const playSource = audioContext.createBufferSource();
                        playSource.buffer = audioBuffer;
                        playSource.connect(audioContext.destination);
                        playSource.start();
                    }
                };

                ws.onclose = () => {
                    status.innerText = 'Disconnected';
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                };
            };

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

@app.websocket("/ws/call")
async def websocket_call(websocket: WebSocket):
    await websocket.accept()
    print("⚡ New WebSocket voice call connected.")
    try:
        while True:
            data = await websocket.receive_bytes()
    except WebSocketDisconnect:
        print("🔌 Voice call disconnected.")
