import json
import os
from transformers import AutoTokenizer

def preprocess_ner_data(input_dir, output_dir, tokenizer_name="bert-base-multilingual-cased"):
    """
    Preprocess NER data for token-level fine-tuning with BIO tagging and subword alignment.

    Args:
        input_dir (str): Directory containing raw NER JSON files (e.g., en_ner_1000.json).
        output_dir (str): Directory to save processed JSON files.
        tokenizer_name (str): Name of the tokenizer to use.

    Returns:
        None: Saves processed data to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    label_map = {
        "O": 0,
        "B-PERSON": 1, "I-PERSON": 2,
        "B-DATE": 3, "I-DATE": 4,
        "B-TIME": 5, "I-TIME": 6,
        "B-ORG": 7, "I-ORG": 8
    }

    def process_example(example):
        text = example["text"]
        entities = example.get("entities", [])
        tokenized = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        tokens = tokenized["tokens"]
        offsets = tokenized["offset_mapping"]
        ner_tags = [0] * len(tokens)  # Initialize with O

        # Sort entities by start position
        entities = sorted(entities, key=lambda x: x["start"])

        for ent in entities:
            ent_start = ent["start"]
            ent_end = ent["end"]
            ent_label = ent["label"]
            overlapping_tokens = []

            for i, (start, end) in enumerate(offsets):
                if start < ent_end and end > ent_start:
                    overlapping_tokens.append(i)

            if overlapping_tokens:
                # Sort overlapping tokens by position
                overlapping_tokens.sort()
                # B- for first, I- for rest
                for j, token_idx in enumerate(overlapping_tokens):
                    if j == 0:
                        ner_tags[token_idx] = label_map[f"B-{ent_label}"]
                    else:
                        ner_tags[token_idx] = label_map[f"I-{ent_label}"]

        return {"tokens": tokens, "ner_tags": ner_tags}

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
                data = json.load(f)

            processed = [process_example(ex) for ex in data]

            output_file = os.path.join(output_dir, filename)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(processed, f, ensure_ascii=False, indent=2)

    print("✅ All NER files processed to tokens+ner_tags with BIO tagging")

# Example usage:
# preprocess_ner_data("data/datasets/ner_new", "data/datasets/ner_prepared")
