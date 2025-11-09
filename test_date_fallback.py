# Test script for date fallback extraction
from app.utils.date_resolver import get_date_replacements

def test_extract_entities_fallback(text: str):
    # Mock NER pipeline - assume no DATE extracted
    mapped_entities = {"PERSON": "الدكتور أحمد"}  # Mock NER output

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

        # If still no DATE, check for regex patterns like "بعد X أيام"
        if "DATE" not in mapped_entities:
            import re
            match = re.search(r'بعد\s+(\d+)\s+أيام', text_lower)
            if match:
                mapped_entities["DATE"] = f"بعد {match.group(1)} أيام"
            else:
                match = re.search(r'after\s+(\d+)\s+days', text_lower)
                if match:
                    mapped_entities["DATE"] = f"after {match.group(1)} days"

    return mapped_entities

# Test cases
test_cases = [
    "أبغى أحجز موعد مع الدكتور أحمد بعد بكرة",
    "أريد حجز مع الطبيب غدًا",
    "موعد اليوم مع الدكتور",
    "أبغى أحجز موعد مع الدكتور أحمد بعد 3 أيام"
]

for text in test_cases:
    entities = test_extract_entities_fallback(text)
    print(f"Text: {text}")
    print("Extracted entities:", entities)
    print("DATE found:", "DATE" in entities)
    if "DATE" in entities:
        from app.utils.date_resolver import resolve_relative_date
        resolved = resolve_relative_date(entities["DATE"])
        print("Resolved DATE:", resolved)
    print("---")
