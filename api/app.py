from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app)

# Flask 

@app.errorhandler(500)
def handle_internal_server_error(error):
    return jsonify({"error": "Internal Server Error", "details": str(error), "trace": traceback.format_exc()}), 500

@app.route("/")
def home():
    return "If humanity successfully builds AGI, how the hell are we gonna control it? In the meantime, I'm building this with the end goal of eventually helping me do shit while I'm operating my digital world. Starting out with an AI specializing in distraction management."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)