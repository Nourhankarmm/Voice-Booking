import os
import json
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
import torch
import numpy as np
import evaluate

# -------- Paths --------
DATA_DIR = r"C:\Users\DELL\voice-booking\data\datasets\ner_prepared"
OLD_MODEL_PATH = "bert-base-multilingual-cased"  # أو لو عايز تكمل تدريب على الموديل القديم: "models/ner_model/output"
OUTPUT_DIR = "E:/ner_new_checkpoints"  # Moved to E: drive due to low space on C:
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- Load tokenizer --------
tokenizer = AutoTokenizer.from_pretrained(OLD_MODEL_PATH)

# -------- Load NER JSON files and create Hugging Face Dataset --------
all_datasets = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".json"):
        with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
            data = json.load(f)
        all_datasets.append(Dataset.from_list(data))

dataset = concatenate_datasets(all_datasets)

# -------- Prepare inputs --------
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# -------- Split train/dev --------
split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

# -------- Load model --------
num_labels = len(set(label for sublist in dataset["ner_tags"] for label in sublist))
model = AutoModelForTokenClassification.from_pretrained(OLD_MODEL_PATH, num_labels=num_labels)

# -------- Metrics --------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=2)

    true_preds = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Flatten
    true_preds_flat = [p for sub in true_preds for p in sub]
    true_labels_flat = [l for sub in true_labels for l in sub]

    acc = accuracy.compute(predictions=true_preds_flat, references=true_labels_flat)
    f1_score = f1.compute(predictions=true_preds_flat, references=true_labels_flat, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}

# -------- Training Arguments --------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
)

# -------- Trainer --------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# -------- Train --------
trainer.train()

# -------- Save --------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"NER model saved to {OUTPUT_DIR}")
