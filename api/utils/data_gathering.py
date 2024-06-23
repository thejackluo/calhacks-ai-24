# data_gathering.py

import pandas as pd
from datasets import load_dataset
import os

def download_and_preprocess_dataset(dataset_name: str, save_path: str, min_length: int = 1):
    """
    Downloads and preprocesses a dataset from Hugging Face and saves it as a CSV file.

    Args:
    - dataset_name (str): The name of the dataset to download.
    - save_path (str): The path where the processed CSV file will be saved.
    - min_length (int): The minimum length of conversations to keep.

    Returns:
    - None
    """
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

        # Save the processed DataFrame to a CSV file
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        
        df.to_csv(save_path, index=False)
        print(f"Dataset successfully saved to {save_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    dataset_name = "BothBosu/Scammer-Conversation"
    save_path = "../data/raw/scammer_conversation_train.csv"
    min_length = 1
    
    download_and_preprocess_dataset(dataset_name, save_path, min_length)