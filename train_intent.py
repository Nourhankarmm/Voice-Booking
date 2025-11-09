import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

#Paths
OLD_MODEL_PATH = "models/intent_model/output"          # الموديل القديم
NEW_DATA_PATH = "data/datasets/intent_new"            # البيانات الجديدة
OUTPUT_PATH = "E:/intent_new_checkpoints"   # مكان حفظ checkpoints جديدة (moved to E: drive due to low space on C:)

os.makedirs(OUTPUT_PATH, exist_ok=True)

#Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(OLD_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(OLD_MODEL_PATH, num_labels=2)

#Load new dataset
new_dataset = load_dataset("json", data_files=f"{NEW_DATA_PATH}/*.json")["train"]

#Label mapping
label2id = {"book_appointment": 0, "other": 1}

#Tokenization
def preprocess(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tokenized["labels"] = [label2id[intent] for intent in examples["intent"]]
    return tokenized

tokenized_dataset = new_dataset.map(preprocess, batched=True)

#Training Arguments
args = TrainingArguments(
    output_dir=OUTPUT_PATH,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    eval_strategy="no",
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=False,
)

#Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer,
)

#Train
print("Starting fine-tuning for Intent Classification on new data...")
trainer.train()
print(" Training completed!")

#Save model
trainer.save_model(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)
print(f"Model saved to {OUTPUT_PATH}")
