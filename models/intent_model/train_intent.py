# models/intent_model/train_intent.py

import os
import torch
import pandas as pd
import json
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments
)
from transformers import Trainer
from torch.utils.data import Dataset
import numpy as np
import evaluate

# -------- Settings --------
MODEL_NAME = "bert-base-multilingual-cased"
TARGET_LANGS = ["en-US", "ar-SA", "hi-IN", "bn-BD"]
OUTPUT_DIR = "./models/intent_model/output"

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom dataset class
class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].detach().clone() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

#Load and Filter Dataset
print("Loading MASSIVE dataset from local JSONL files...")
data_dir = "../../data/datasets/massive/1.1/data"

def load_massive_data(data_dir, target_langs):
    data = []
    for lang in target_langs:
        file_path = os.path.join(data_dir, f"{lang}.jsonl")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                data.append(entry)
    return pd.DataFrame(data)

# Load the dataset from JSONL files
df = load_massive_data(data_dir, TARGET_LANGS)

# Split into train and test
train_df = df[df['partition'] == 'train'].copy()
test_df = df[df['partition'] == 'test'].copy()

#Prepare Labels
# Simplify intents: if contains "book" -> "book_appointment", else "other"
def simplify_intent(intent):
    if "book" in intent.lower():
        return 1  # book_appointment
    else:
        return 0  # other

train_df['intent_label'] = train_df['intent'].apply(simplify_intent)
test_df['intent_label'] = test_df['intent'].apply(simplify_intent)

#Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_data(df):
    utterances = df['utt'].tolist()
    return tokenizer(utterances, truncation=True, padding=True, max_length=64, return_tensors="pt")

train_encodings = tokenize_data(train_df)
test_encodings = tokenize_data(test_df)

train_labels = train_df['intent_label'].tolist()
test_labels = test_df['intent_label'].tolist()

# Create datasets
train_dataset = IntentDataset(train_encodings, train_labels)
test_dataset = IntentDataset(test_encodings, test_labels)

#Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

# Metrics 
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=preds, references=labels)
    f1_score = f1.compute(predictions=preds, references=labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}

#Training Arguments 
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU available
    resume_from_checkpoint=True  # Resume from latest checkpoint
)

#Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train
print(" Starting fine-tuning for Intent Classification...")
trainer.train()
print(" Training completed!")

#Save Model
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
print(" Intent Classification model is ready!")
