# =======================
# Documentation
# =======================
"""
STAGE 1:
We are going to take an input and differentiate it between a post and a conversation.
A post is a single block of text with an author.
A conversation consists of multiple strings of text in a dictionary with authors.

STAGE 2:
Classify the text using a scam classifier.

STAGE 3:
Use another classifier (like GPT-4) to classify the type of scam.

STAGE 4:
Generate a response using an LLM.

STAGE 5:
Implement context for more complicated and comprehensive responses.
"""

'''
EXAMPLE 1
input = {
    "data": {
        "author": "jack",
        "text": "this is a good faith post"
    }
}

output = {
    "type": "post",
    "data": {
        "id": "123",
        "author": "jack",
        "text": "this is a good faith post"
        "result": "1% scam"
    }
}

EXAMPLE 2
input = {
    "data": [
        {
            "author": "scammer",
            "text": "wire 1 million dollars to this account"
        }, 
        {
            "author": "victim",
            "text": "ok, i will :)"
        }
    ]
}
output = {
    "type": "conversation",
    "data": {
        "id": "123",
        "conversation": [
            {
                "author": "scammer",
                "text": "wire 1 million dollars to this account"
                "result": "99% scam"
            }, 
            {
                "author": "victim",
                "text": "ok, i wilkl"
                "result": "1% scam"
            }
        ]
    }
}
'''

# =======================
# IMPORT
# =======================
"""
Import necessary libraries and modules.
"""
import os
import json
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, jsonify
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint

# =======================
# ENV (TEST SUCCESS)
# =======================
"""
Load environment variables.
"""
print("Step 1: Loading Environment Variables")
print("=====================================")
# try to load the .env file
try:
    load_dotenv(override=True)
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
    if HUGGINGFACE_API_TOKEN:
        print("Loaded Environment Variables")
        print(f"HUGGINGFACE_API_TOKEN: {HUGGINGFACE_API_TOKEN}")
    else:
        print("HUGGINGFACE_API_TOKEN not found in environment variables")
except Exception as e:
    print(f"Error loading environment variables: {e}")


# =======================
# MODEL LOADING
# =======================
"""
Load the Hugging Face model and tokenizer.
"""
print("Step 2: Loading Model")
print("====================")

try:
    tokenizer = AutoTokenizer.from_pretrained("pippinnie/scam_text_classifier")
    model = AutoModelForSequenceClassification.from_pretrained("pippinnie/scam_text_classifier", from_tf=True)
    model_loaded = True
    print("Loaded Model")
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

# =======================
# FUNCTIONS (TEST SUCCESS)
# =======================
"""
Define functions for detecting input type and classifying text.
"""

def detect_input_type(input_data):
    """
    Detect if the input is a post or a conversation.
    """
    if isinstance(input_data["data"], list):
        return "conversation"
    else:
        return "post"

def classify_text(text):
    """
    Classify the text using the scam classifier.
    """
    if not model_loaded:
        raise RuntimeError("Model not loaded")
    
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=1)
    return probs.tolist()  # Convert tensor to list

# =======================
# DRIVER CODE
# =======================
"""
Test the functions.
"""
print("Step 3: Testing Functions: detect_input_type")
print("===========================================")

input_data = {
    "data": [
        {
            "author": "scammer",
            "text": "wire 1 million dollars to this account"
        }, 
        {
            "author": "victim",
            "text": "ok, i will :)"
        }
    ]
}
print(detect_input_type(input_data))  # Expected: conversation

input_data = {
    "data": {
        "author": "jack",
        "text": "this is a good faith post"
    }
}
print(detect_input_type(input_data))  # Expected: post

print("Step 4: Testing Functions: classify_text")
print("=======================================")
try:
    text = "wire 1 million dollars to this account"
    print(classify_text(text))  # Expected output: probabilities
    text = "this is a good faith post"
    print(classify_text(text))  # Expected output: probabilities
except Exception as e:
    print(f"Error during classification: {e}")

# =======================
# API
# =======================
"""
Create a Flask API to classify input text.
"""
app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    input_type = detect_input_type(data)
    result = {"type": input_type, "data": data}

    if input_type == "post":
        text = data["data"]["text"]
        result["data"]["result"] = classify_text(text)
    elif input_type == "conversation":
        for message in data["data"]:
            message["result"] = classify_text(message["text"])

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)


# plan for part 2
"""
stage 2: langchain and type (added complexity) (EASY)
# Implement Langchain (to incorporate multiple models)
# Implement a scam classifer and ensure that langchain works

stage 3: implement context and memory to the resopnse (context)
# Implement GPT 4o API to incorporate context information
# add context to the post pipeline for better nuanced detection (the context could highlight 
and make scams look more explict) (we can implement a live demo feature that shows how the model 
gradually learns it is a scam through pyschological markesr)
# use the memory to train the model and have it have reinforcement learning

stage 2+: 
# add fine tuning to the conversation classifer

stage 3+: 
# implement pyschological feature generator from a scratch
# incorporating multiple models together
"""
