# data_augmentation.py

import os
import openai
import pandas as pd
from datasets import load_dataset
import random

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def print_openai_key():
    print(f"S1: OpenAI key: {openai.api_key}")

def generate_augmented_conversation(conversation, prompt, temperature):
    full_prompt = f"{prompt}\n\n{conversation}\n\nOutput format: CSV\n\nAugmented conversation:"
    
    response = openai.Completion.create(
        model="text-davinci-003",  # Use the appropriate model for GPT-4
        prompt=full_prompt,
        max_tokens=500,
        temperature=temperature
    )
    
    augmented_conversation = response.choices[0].text.strip()
    print(f"Augmented conversation (Temperature {temperature}): {augmented_conversation}")
    return augmented_conversation

def augment_dataset(dataset_name: str, save_dir: str, min_length: int = 1):
    try:
        # Load the dataset from Hugging Face
        print(f"Loading dataset '{dataset_name}' from Hugging Face...")
        dataset = load_dataset(dataset_name)
        
        # Access the training set
        train_set = dataset['train']
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(train_set)
        
        # Preprocess the dataset
        print("Preprocessing the dataset...")
        df['conversation_length'] = df['conversation'].apply(len)
        df = df[df['conversation_length'] >= min_length]

        # Define prompts for augmentation
        prompts = {
            "subtle": "Augment the following conversation to make it more subtle and hard to detect as a scam.",
            "aggressive": "Augment the following conversation to make it more aggressive and urgent.",
            "variation": "Create variations of the following scam conversation."
        }

        # Augment the dataset
        for key, prompt in prompts.items():
            print(f"Augmenting the dataset with prompt '{key}'...")
            df[f'augmented_conversation_{key}'] = df['conversation'].apply(
                lambda conv: generate_augmented_conversation(conv, prompt, random.uniform(0.65, 0.75))
            )

            # Save the processed and augmented DataFrame to a CSV file
            augmented_save_path = os.path.join(save_dir, f'scammer_conversation_{key}_augmented.csv')
            if not os.path.exists(os.path.dirname(augmented_save_path)):
                os.makedirs(os.path.dirname(augmented_save_path))
            
            df[['conversation', f'augmented_conversation_{key}']].to_csv(augmented_save_path, index=False)
            print(f"Dataset successfully saved to {augmented_save_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

def run_tests():
    """
    Runs a series of tests
    """
    print_openai_key()
    
    # Test augmentation on a small sample
    sample_conversation = "Hello, I am in trouble and need your help urgently. Please send me $500."
    sample_prompt = "Augment the following conversation to make it more subtle and hard to detect as a scam."
    sample_temperature = 0.7
    
    augmented_sample = generate_augmented_conversation(sample_conversation, sample_prompt, sample_temperature)
    print(f"Test Augmented Sample: {augmented_sample}")

if __name__ == "__main__":
    dataset_name = "BothBosu/Scammer-Conversation"
    save_dir = "../data/augmented"
    min_length = 1
    
    run_tests()
    augment_dataset(dataset_name, save_dir, min_length)