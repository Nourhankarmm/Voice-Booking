from fastapi import APIRouter
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline
)
import torch
from ..utils.date_resolver import resolve_relative_date, get_date_replacements  # ✅ import date resolver and replacements

#Router
nlp_router = APIRouter(
    prefix="/nlp",
    tags=["NLP"]
)

#Paths
INTENT_MODEL_PATH = "E:/intent_new_checkpoints"
NER_MODEL_PATH = "E:/ner_model_output"

#Load Models
print(" Loading NLP models...")

intent_tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_PATH)
intent_model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_PATH)

ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)

print("Models loaded successfully")

#Pipelines
device = 0 if torch.cuda.is_available() else -1

intent_pipeline = pipeline(
    "text-classification",
    model=intent_model,
    tokenizer=intent_tokenizer,
    device=device
)

ner_pipeline = pipeline(
    "token-classification",
    model=ner_model,
    tokenizer=ner_tokenizer,
    device=device,
    aggregation_strategy="simple"
)

#Label Mapping
intent_labels = {
    "LABEL_0": "book_appointment",
    "LABEL_1": "other"
}

label_mapping = {
    "LABEL_0": "O",
    "LABEL_1": "PERSON",
    "LABEL_2": "DATE",
    "LABEL_3": "TIME",
    "LABEL_4": "ORG"
}

#Helper Functions
def predict_intent(text: str):
    intent_result = intent_pipeline(text)[0]
    intent_label = intent_labels.get(intent_result["label"], intent_result["label"])
    confidence = float(intent_result["score"])
    return {"intent": intent_label, "confidence": confidence}


def extract_entities(text: str):
    entities = ner_pipeline(text)
    mapped_entities = {}
    for entity in entities:
        label = entity["entity_group"]
        if label in label_mapping:
            mapped_label = label_mapping[label]
            if mapped_label != "O":
                word = entity["word"].replace("##", "").strip(".").strip()
                mapped_entities[mapped_label] = word

    # Check if TIME entity contains date-like expressions and move to DATE
    if "TIME" in mapped_entities:
        time_value = mapped_entities["TIME"]
        date_replacements = get_date_replacements()
        text_lower = time_value.lower()
        # Check if TIME contains date expressions
        for phrase in date_replacements.keys():
            if phrase in text_lower:
                mapped_entities["DATE"] = time_value
                del mapped_entities["TIME"]
                break
        # Also check for dynamic patterns
        if "DATE" not in mapped_entities:
            import re
            if re.search(r'بعد\s*\d+\s*أيام', text_lower) or re.search(r'after\s*\d+\s*days', text_lower):
                mapped_entities["DATE"] = time_value
                del mapped_entities["TIME"]



    # Fallback: Scan for date expressions if DATE not extracted by NER
    if "DATE" not in mapped_entities:
        date_replacements = get_date_replacements()
        text_lower = text.lower()
        # Sort phrases by length descending to prioritize longer matches
        sorted_phrases = sorted(date_replacements.keys(), key=len, reverse=True)
        for phrase in sorted_phrases:
            if phrase in text_lower:
                mapped_entities["DATE"] = phrase
                break  # Take the first (longest) match

        # If still no DATE, check for patterns like "بعد X أيام"
        if "DATE" not in mapped_entities:
            import re
            match = re.search(r'بعد\s*(\d+)\s*أيام', text_lower)
            if match:
                mapped_entities["DATE"] = f"بعد {match.group(1)} أيام"
            else:
                match = re.search(r'after\s*(\d+)\s*days', text_lower)
                if match:
                    mapped_entities["DATE"] = f"after {match.group(1)} days"

    return mapped_entities

#Request Body
class TextRequest(BaseModel):
    text: str

#Route
@nlp_router.post("/parse")
async def parse_text(request: TextRequest):
    text = request.text

    # 1️ Intent prediction
    intent_data = predict_intent(text)

    # 2️ Entity extraction
    extracted_entities = extract_entities(text)

    # 3️ Resolve relative dates
    if "DATE" in extracted_entities:
        resolved_date = resolve_relative_date(extracted_entities["DATE"])
        extracted_entities["DATE"] = resolved_date

    # 4️ Final output
    return {
        "intent": intent_data["intent"],
        "confidence": intent_data["confidence"],
        "entities": extracted_entities
    }
