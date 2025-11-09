from transformers import AutoModelForSequenceClassification

model_path = "E:/intent_new_checkpoints"
model = AutoModelForSequenceClassification.from_pretrained(model_path)

print(model.config.id2label)
