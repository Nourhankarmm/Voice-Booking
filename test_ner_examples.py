import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load the trained model and tokenizer
MODEL_PATH = "E:/ner_model_output"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

# Label mapping
id2label = {0: "O", 1: "PERSON", 2: "DATE", 3: "TIME", 4: "ORG"}

# Test examples
test_examples = [
    {"text": "I want to book an appointment with Dr. Ahmed tomorrow", "lang": "en"},
    {"text": "मुझे कल डॉक्टर अहमद से अपॉइंटमेंट बुक करनी है", "lang": "hi"},
    {"text": "আমি আগামীকাল ডাঃ আহমেদের সঙ্গে দেখা করতে চাই", "lang": "bn"},
    {"text": "أبغى أحجز موعد مع الدكتور أحمد بكرة", "lang": "sa"}
]

print("Testing NER model with provided examples...")
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
