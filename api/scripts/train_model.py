import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizerFast
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Load the CSV file with emotion scores
input_file_path = "../data/augmented/scammer_conversation_augmented_with_emotion_scores.csv"
data = pd.read_csv(input_file_path)

# Select necessary columns and rename them for clarity
data = data[['conversation', 'impatience', 'urgency', 'guilt-tripping', 'fear']]
data.columns = ['text', 'impatience', 'urgency', 'guilt_tripping', 'fear']

# Split the dataset into train, validation, and test sets
train_df, temp_df = train_test_split(data, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Convert DataFrame to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Combine into a DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

print(dataset)

# Load pre-trained model and tokenizer
model_name = "ali619/distilbert-base-uncased-finetuned-emotion-detector-from-text"
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=4)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_steps=10
)

# Define a compute metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Fine-tune the model
trainer.train()

# Evaluate the model on the test set
results = trainer.evaluate(tokenized_datasets['test'])
print(results)