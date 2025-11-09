import os
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments
)
from transformers import Trainer
from torch.utils.data import Dataset
import numpy as np
import evaluate
import json

#Settings 
MODEL_NAME = "bert-base-multilingual-cased"
TARGET_LANGS = ["en", "hi", "bn", "ar", "sa"]
LANG_NAMES = {"en": "English", "hi": "Hindi", "bn": "Bangla", "ar": "Arabic", "sa": "saudi_arabia"}
OUTPUT_DIR = "E:/ner_model_output"

# Custom dataset class
class NERDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Load Raw NER Dataset 
print("Loading raw NER dataset from JSON files...")
data_dir = "../../data/datasets/ner_new"

def load_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {filepath}")
    return data

# Load data for target languages
train_data = []
dev_data = []
test_data = []

for lang in TARGET_LANGS:
    json_file = os.path.join(data_dir, f"{lang}_ner_1000.json")
    if os.path.exists(json_file):
        loaded = load_json_file(json_file)
        # Split into train/dev/test (80/10/10)
        n = len(loaded)
        train_split = int(0.8 * n)
        dev_split = int(0.9 * n)
        train_data.extend(loaded[:train_split])
        dev_data.extend(loaded[train_split:dev_split])
        test_data.extend(loaded[dev_split:])
        print(f"Split {lang} data: train {train_split}, dev {dev_split - train_split}, test {n - dev_split}")
    else:
        print(f"File {json_file} does not exist. Skipping {lang}.")

print(f"Total train examples: {len(train_data)}")
print(f"Total dev examples: {len(dev_data)}")
print(f"Total test examples: {len(test_data)}")

#  Prepare Labels 
# Label mapping
label2id = {"O": 0, "PERSON": 1, "DATE": 2, "TIME": 3, "ORG": 4}
id2label = {i: tag for tag, i in label2id.items()}

def prepare_data(data):
    texts = []
    labels = []
    for example in data:
        texts.append(example["text"])
        # Convert entities to BIO tags
        text = example["text"]
        entities = example.get("entities", [])
        ner_tags = [0] * len(text)  # O by default
        for ent in entities:
            start, end, label = ent["start"], ent["end"], ent["label"]
            if label in label2id:
                for i in range(start, end):
                    ner_tags[i] = label2id[label]
        labels.append(ner_tags)
    return texts, labels

train_texts, train_labels = prepare_data(train_data)
dev_texts, dev_labels = prepare_data(dev_data)
test_texts, test_labels = prepare_data(test_data)

# Tokenization 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(texts, labels):
    tokenized_inputs = tokenizer(texts, truncation=True, padding=True, max_length=64, return_offsets_mapping=True)

    aligned_labels = []
    for i, (text, label) in enumerate(zip(texts, labels)):
        offsets = tokenized_inputs["offset_mapping"][i]
        word_labels = []
        for start, end in offsets:
            if start == 0 and end == 0:  # special tokens
                word_labels.append(-100)
            else:
                # Use the label of the first character in the token span
                char_label = label[start] if start < len(label) else 0
                word_labels.append(char_label)
        aligned_labels.append(word_labels)

    tokenized_inputs["labels"] = aligned_labels
    # Remove offset_mapping as it's not needed for training
    del tokenized_inputs["offset_mapping"]
    return tokenized_inputs

train_encodings = tokenize_and_align_labels(train_texts, train_labels)
dev_encodings = tokenize_and_align_labels(dev_texts, dev_labels)
test_encodings = tokenize_and_align_labels(test_texts, test_labels)

# Create datasets
train_dataset = NERDataset(train_encodings, train_encodings["labels"])
dev_dataset = NERDataset(dev_encodings, dev_encodings["labels"])
test_dataset = NERDataset(test_encodings, test_encodings["labels"])

# Model 
if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
    print("Loading model from output dir...")
    model = AutoModelForTokenClassification.from_pretrained(OUTPUT_DIR, num_labels=len(label2id))
else:
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(label2id))

# Find last checkpoint for resuming training
checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")] if os.path.exists(OUTPUT_DIR) else []
if checkpoints:
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    last_checkpoint = os.path.join(OUTPUT_DIR, checkpoints[-1])
    print(f"Resuming training from {last_checkpoint}")
else:
    last_checkpoint = None
    print("Starting training from scratch")

#  Metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Flatten
    true_predictions = [p for sublist in true_predictions for p in sublist]
    true_labels = [l for sublist in true_labels for l in sublist]

    acc = accuracy.compute(predictions=true_predictions, references=true_labels)
    f1_score = f1.compute(predictions=true_predictions, references=true_labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}

# Training Arguments 
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    fp16=True,  # Enable fp16 for GPU training
    dataloader_num_workers=0,
    gradient_accumulation_steps=8
)

#Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

if __name__ == '__main__':
    #Train
    print(" Starting fine-tuning for NER...")
    if last_checkpoint:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
    print("Training completed!")

# Save Model 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
    print(" NER model is ready!")

    # Test with Examples
    print("Testing with provided examples...")
    test_examples = [
        {"text": "I want to book an appointment with Dr. Ahmed tomorrow", "lang": "en"},
        {"text": "मुझे कल डॉक्टर अहमद से अपॉइंटमेंट बुक करनी है", "lang": "hi"},
        {"text": "আমি আগামীকাল ডাঃ আহমেদের সঙ্গে দেখা করতে চাই", "lang": "bn"},
        {"text": "أبغى أحجز موعد مع الدكتور أحمد بكرة", "lang": "sa"}
    ]

    for example in test_examples:
        inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=2)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        pred_labels = [id2label[p.item()] for p in predictions[0] if p.item() != -100]
        print(f"Text ({example['lang']}): {example['text']}")
        print(f"Predicted: {list(zip(tokens, pred_labels))}")
        print()
