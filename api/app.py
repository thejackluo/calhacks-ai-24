
# Imports

from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import requests

import asyncio
from hume import HumeVoiceClient, MicrophoneInterface

# ENV

HUME_API_KEY = "3LpGtjH4qrFPqAq6ySRYuAA6IV8zkSpxtiYHs1nqpwvK7FWY"
HUME_SECRET_KEY = "CkpfSUeJtApvpbmrFnHTqpGgSKf9bl3jndfihMYVakHkBShv1wMlLmcwsd7yalGG"

# Config
app = Flask(__name__)
CORS(app)

client = HumeVoiceClient(HUME_API_KEY)

# Flask 

@app.errorhandler(500)
def handle_internal_server_error(error):
    return jsonify({"error": "Internal Server Error", "details": str(error), "trace": traceback.format_exc()}), 500

@app.route("/")
def home():
    return "If humanity successfully builds AGI, how the hell are we gonna control it? In the meantime, I'm building this with the end goal of eventually helping me do shit while I'm operating my digital world. Starting out with an AI specializing in distraction management."

# API

@app.route("/engine/voice", methods=["POST"])
def voice_classifier():

    voice = request.files['voice']

    pass

def analyze_hume_voice():



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)