# TODO: Adding Date Resolver to NLP Service

## Tasks
- [x] Modify `extract_entities` function in `app/routes/nlp_service.py` to add fallback extraction for date expressions if not detected by NER.
- [x] Import or define date keywords from `date_resolver.py` for scanning the text.
- [x] If a date phrase is found and "DATE" not in extracted entities, add it as "DATE" entity.
- [x] Test the updated NLP service with example input to ensure DATE is extracted and resolved correctly.
