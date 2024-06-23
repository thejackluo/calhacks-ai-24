from flask import Flask, request, jsonify, render_template
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

AIORNOT_API_KEY = os.getenv('AIORNOT_API_KEY')
AIORNOT_API_URL = 'https://api.aiornot.com/v1/reports/image'

@app.route('/')
def index():
    return render_template('indexv2.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Image is required'}), 400

    image = request.files['image']

    # Prepare the API request
    headers = {
        'Authorization': f'Bearer {AIORNOT_API_KEY}',
    }
    files = {
        'object': (image.filename, image, image.mimetype)
    }

    response = requests.post(AIORNOT_API_URL, headers=headers, files=files)
    result = response.json()

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
