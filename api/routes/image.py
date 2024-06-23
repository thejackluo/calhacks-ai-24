from flask import Flask, request, jsonify, render_template
import requests
import base64
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PREDEFINED_PROMPT = 'Tell me if the image is AI generated or not. If it is AI generated, only return "True" and if it is not AI generated, only return "False".'
OPENAI_MODEL = 'gpt-4-turbo' 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Image is required'}), 400

    image = request.files['image']

    # Convert image to base64
    image_data = base64.b64encode(image.read()).decode('utf-8')

    # Prepare the API request
    api_url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        'messages': [PREDEFINED_PROMPT],
        'image': image_data,
        'model': OPENAI_MODEL
    }

    response = requests.post(api_url, headers=headers, json=data)
    result = response.json()

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
