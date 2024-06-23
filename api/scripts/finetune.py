import pandas as pd
import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Define file paths
RAW_DATA_PATH = '../data/raw/user_data.csv'
PROCESSED_DATA_PATH = '../data/processed/scammer_conversation_augmented_with_emotion_scores.csv'

# Function to generate emotion scores
def generate_emotion_scores(conversation, prompt_text):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes emotions in text."},
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": "The conversation has the following emotion scores:"},
        {"role": "user", "content": conversation}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7
    )
    message_content = response.choices[0].message.content.strip()
    scores = json.loads(message_content)
    return scores

# Function to process new data
def process_new_data():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"No new data found at {RAW_DATA_PATH}")
        return
    
    # Read new data
    new_data = pd.read_csv(RAW_DATA_PATH)
    
    # Add emotion scores
    new_data['impatience'] = new_data['conversation'].apply(lambda x: generate_emotion_scores(x, "Analyze the impatience level in the following conversation."))
    new_data['urgency'] = new_data['conversation'].apply(lambda x: generate_emotion_scores(x, "Analyze the urgency level in the following conversation."))
    new_data['guilt-tripping'] = new_data['conversation'].apply(lambda x: generate_emotion_scores(x, "Analyze the guilt-tripping level in the following conversation."))
    new_data['fear'] = new_data['conversation'].apply(lambda x: generate_emotion_scores(x, "Analyze the fear level in the following conversation."))
    
    # Append to the existing dataset
    if os.path.exists(PROCESSED_DATA_PATH):
        existing_data = pd.read_csv(PROCESSED_DATA_PATH)
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        combined_data = new_data
    
    # Save the combined dataset
    combined_data.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")

# Main function to run the script
if __name__ == "__main__":
    process_new_data()