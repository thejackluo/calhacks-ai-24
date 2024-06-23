
from flask import Flask, jsonify
from flask_cors import CORS
import traceback
import asyncio
import pyaudio
from authenticator import get_access_token
from connection import Connection

# Audio format and parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SAMPLE_WIDTH = 2

# Config
app = Flask(__name__)
CORS(app)

@app.errorhandler(500)
def handle_internal_server_error(error):
    return jsonify({"error": "Internal Server Error", "details": str(error), "trace": traceback.format_exc()}), 500

@app.route("/")
def home():
    return "If humanity successfully builds AGI, how the hell are we gonna control it? In the meantime, I'm building this with the end goal of eventually helping me do shit while I'm operating my digital world. Starting out with an AI specializing in distraction management."

@app.route("/start-stream", methods=["GET"])
def start_stream():
    try:
        asyncio.run(stream_audio())
        return jsonify({"status": "Streaming started"}), 200
    except Exception as e:
        print(f"Error starting stream: {e}")
        return handle_internal_server_error(e)

async def stream_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    access_token = get_access_token()
    socket_url = f"wss://api.hume.ai/v0/assistant/chat?access_token={access_token}"

    await Connection.connect(
        socket_url,
        stream,
        RATE,
        SAMPLE_WIDTH,
        CHANNELS,
        CHUNK
    )

    stream.stop_stream()
    stream.close()
    audio.terminate()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
