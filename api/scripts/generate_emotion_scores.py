import os
import openai
import pandas as pd
import time
import json  # Import the json module

from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

def test_openai_key():
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ],
            temperature=0.5
        )
        print("S1: OpenAI key test successful. Response: ", response.choices[0].message.content)
    except openai.OpenAIError as e:
        print(f"Error testing OpenAI API key: {e}")

def generate_emotion_scores(conversation, prompt, temperature=0.7):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an emotion detection assistant."},
                {"role": "user", "content": f"{prompt}\n\n{conversation}\n\nPlease provide a JSON response with scores between 0 and 1 for the following emotions: impatience, urgency, guilt-tripping, and fear."}
            ],
            temperature=temperature
        )
        # Ensure the response is a proper text string
        message_content = response.choices[0].message.content.strip()
        print(f"Raw response: {message_content}")  # Debugging: print raw response

        # Extracting the emotion scores from JSON response
        try:
            scores = json.loads(message_content)
            return scores
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            return None
    except openai.OpenAIError as e:
        print(f"Error processing conversation: {e}")
        return None

def add_emotion_scores_to_dataset(input_file_path, output_file_path):
    data = pd.read_csv(input_file_path)
    prompts = {
        "subtle": "Augment the following conversation to make it more subtle and hard to detect as a scam.",
        "aggressive": "Augment the following conversation to make it more aggressive and urgent.",
        "variation": "Create variations of the following scam conversation."
    }
    scores = {"impatience": [], "urgency": [], "guilt_tripping": [], "fear": []}

    for index, row in data.iterrows():
        if index >= 150:  # Limit to processing only the first 5 rows
            break
        conversation = row['conversation']
        conversation_scores = {"impatience": 0, "urgency": 0, "guilt_tripping": 0, "fear": 0}

        for prompt_type, prompt_text in prompts.items():
            response = generate_emotion_scores(conversation, prompt_text)
            if response:
                try:
                    conversation_scores["impatience"] += response.get("impatience", 0) / 3
                    conversation_scores["urgency"] += response.get("urgency", 0) / 3
                    conversation_scores["guilt_tripping"] += response.get("guilt-tripping", 0) / 3
                    conversation_scores["fear"] += response.get("fear", 0) / 3
                except Exception as e:
                    print(f"Error processing row {index}: {e}")
                    conversation_scores["impatience"] = None
                    conversation_scores["urgency"] = None
                    conversation_scores["guilt_tripping"] = None
                    conversation_scores["fear"] = None
            else:
                conversation_scores["impatience"] = None
                conversation_scores["urgency"] = None
                conversation_scores["guilt_tripping"] = None
                conversation_scores["fear"] = None

        scores["impatience"].append(conversation_scores["impatience"])
        scores["urgency"].append(conversation_scores["urgency"])
        scores["guilt_tripping"].append(conversation_scores["guilt_tripping"])
        scores["fear"].append(conversation_scores["fear"])
        time.sleep(1)

    # If the length of the scores lists is not equal to the length of the dataframe, pad the lists with None
    while len(scores["impatience"]) < len(data):
        scores["impatience"].append(None)
        scores["urgency"].append(None)
        scores["guilt_tripping"].append(None)
        scores["fear"].append(None)

    data['impatience'] = scores["impatience"]
    data['urgency'] = scores["urgency"]
    data['guilt_tripping'] = scores["guilt_tripping"]
    data['fear'] = scores["fear"]
    data.to_csv(output_file_path, index=False)
    print(f"Data with emotion scores saved to {output_file_path}")
    print("Process completed successfully.")

if __name__ == "__main__":
    test_openai_key()
    input_file_path = "../data/augmented/scammer_conversation_augmented.csv"
    output_file_path = "../data/augmented/scammer_conversation_augmented_with_emotion_scores.csv"
    add_emotion_scores_to_dataset(input_file_path, output_file_path)