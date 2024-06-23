import gc
import numpy as np
import pandas as pd
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    f1_score
)

from imblearn.over_sampling import RandomOverSampler
import accelerate
import evaluate
from datasets import Dataset, Image, ClassLabel
from transformers import (
    TrainingArguments,
    Trainer,
    ViTImageProcessor,
    ViTForImageClassification,
    DefaultDataCollator
)
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomRotation,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomAdjustSharpness,
    Resize,
    ToTensor
)

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

image_data = {}

# Import necessary modules
from pathlib import Path
from tqdm import tqdm
import os

image_files = []
image_labels = []

# Collect image file paths and labels
for img_file in sorted(Path('input/deepfake-and-real-images/Dataset/').glob('*/*/*.*')):
    label_name = str(img_file).split('/')[-2]
    image_labels.append(label_name)
    image_files.append(str(img_file))

print(len(image_files), len(image_labels))

# Create a DataFrame
dataframe = pd.DataFrame.from_dict({"image": image_files, "label": image_labels})

# Display the first few rows
dataframe.head()
dataframe['label'].unique()

# Perform random oversampling
y_labels = dataframe[['label']]
dataframe = dataframe.drop(['label'], axis=1)

oversampler = RandomOverSampler(random_state=83)
dataframe, resampled_labels = oversampler.fit_resample(dataframe, y_labels)

dataframe['label'] = resampled_labels

# Clean up
del resampled_labels
gc.collect()

# Create Dataset from the DataFrame
dataset = Dataset.from_pandas(dataframe).cast_column("image", Image())
dataset[0]["image"]

# Subset of labels
subset_labels = image_labels[:5]

label_names = ['Real', 'Fake']

label_to_id, id_to_label = {}, {}

# Create label-to-ID and ID-to-label mappings
for idx, label in enumerate(label_names):
    label_to_id[label] = idx
    id_to_label[idx] = label

# Define class labels
class_labels = ClassLabel(num_classes=len(label_names), names=label_names)

# Function to map labels to IDs
def map_label_to_id(example):
    example['label'] = class_labels.str2int(example['label'])
    return example

# Apply label mapping and split dataset
dataset = dataset.map(map_label_to_id, batched=True)
dataset = dataset.cast_column('label', class_labels)
dataset = dataset.train_test_split(test_size=0.4, shuffle=True, stratify_by_column="label")

train_data = dataset['train']
test_data = dataset['test']

model_identifier = "deepfake_vs_real_image_detection"  # Pre-trained model identifier

# Create a processor for ViT model input
image_processor = ViTImageProcessor.from_pretrained(model_identifier)
image_mean, image_std = image_processor.image_mean, image_processor.image_std

# Get the size of the ViT model's input images
input_size = image_processor.size["height"]
print("Input Size: ", input_size)

# Define normalization transformation for input images
normalize_transform = Normalize(mean=image_mean, std=image_std)

# Define training transformations
train_transforms = Compose(
    [
        Resize((input_size, input_size)),
        RandomRotation(90),
        RandomAdjustSharpness(2),
        ToTensor(),
        normalize_transform
    ]
)

# Define validation transformations
val_transforms = Compose(
    [
        Resize((input_size, input_size)),
        ToTensor(),
        normalize_transform
    ]
)

# Apply training transformations
def apply_train_transforms(examples):

    examples['pixel_values'] = [train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

# Apply validation transformations
def apply_val_transforms(examples):

    examples['pixel_values'] = [val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples



train_data.set_transform(apply_train_transforms)
test_data.set_transform(apply_val_transforms)
def custom_collate_fn(batch_samples):
    pixel_values = torch.stack([sample["pixel_values"] for sample in batch_samples])
    labels = torch.tensor([sample['label'] for sample in batch_samples])
    return {"pixel_values": pixel_values, "labels": labels}

# Load the ViT model for image classification with the specified model identifier
vit_model = ViTForImageClassification.from_pretrained(model_identifier, num_labels=len(label_names))

# Update model configuration with label mappings
vit_model.config.id2label = id_to_label
vit_model.config.label2id = label_to_id

# Print the number of trainable parameters (in millions)
print(f"Number of trainable parameters: {vit_model.num_parameters(only_trainable=True) / 1e6:.2f}M")

# Load the accuracy metric for evaluation
accuracy_metric = evaluate.load("accuracy")

# Define a function to compute metrics during evaluation
def evaluate_metrics(eval_predictions):
    preds = eval_predictions.predictions.argmax(axis=1)
    labels = eval_predictions.label_ids
    accuracy_score = accuracy_metric.compute(predictions=preds, references=labels)['accuracy']
    return {"accuracy": accuracy_score}

metric_name = "accuracy"
project_name = "deepfake_vs_real_image_detection"
num_epochs = 2

# Define training arguments for the model
training_args = TrainingArguments(
    output_dir=project_name,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    learning_rate=1e-6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=8,
    num_train_epochs=num_epochs,
    weight_decay=0.02,
    warmup_steps=50,
    remove_unused_columns=False,
    save_strategy='epoch',
    load_best_model_at_end=True,
    save_total_limit=1,
    report_to="none"
)

# Initialize the Trainer with the model, training arguments, dataset, and metrics
trainer = Trainer(
    model=vit_model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=custom_collate_fn,
    compute_metrics=evaluate_metrics
)

# Start training the model
trainer.train()

from transformers import pipeline

image_classifier = pipeline('image-classification', model=vit_model, device=0)

# Get an image from the test dataset
sample_image = test_data[1]["image"]
predicted_label = id_to_label[test_data[1]["label"]]
