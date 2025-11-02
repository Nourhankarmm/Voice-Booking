# Voice Booking System Documentation

This documentation outlines the step-by-step implementation of a voice-activated appointment booking system supporting English, Hindi, Bengali, and Arabic (Saudi dialect). The system uses voice input, processes it through ASR, NLP, backend logic, database, and TTS for audio replies.

## Architecture Overview

```
Voice Input
    ↓
Whisper (Multilingual ASR)
    ↓
NLP (Intent + Entity Extraction)
    ↓
FastAPI Backend (Booking Logic)
    ↓
Database (SQLite/PostgreSQL)
    ↓
gTTS (Text-to-Speech for Audio Reply)
```

## Step-by-Step Implementation

### Step 0: Preparations

1. **Create a new project and virtual environment**

   - Run: `conda create -n voicebook python=3.9`
   - Activate: `conda activate voicebook`
   - Install dependencies: `pip install -r requirements.txt`
   - Initialize Git: `git init`

2. **Create requirements.txt with core dependencies**

   - Add the following to `requirements.txt`:
     ```
     fastapi==0.110.0
     uvicorn[standard]==0.29.0
     faster-whisper==1.0.1
     sqlalchemy==1.3.24
     requests==2.31.0
     python-multipart==0.0.9
     azure-cognitiveservices-speech==1.36.0
     transformers
     torch==2.7.1+cu126
     torchvision==0.22.0+cu126
     torchaudio==2.7.1+cu126
     huggingface-hub
     scikit-learn
     evaluate
     accelerate
     datasets
     psycopg2-binary
     av
     ctranslate2
     gtts
     ```
   - Install: `pip install -r requirements.txt`

3. **Set up basic folder structure**
   ```
   voice-booking/
   ├── app/
   │   ├── routes/
   │   │   ├── asr_service.py
   │   │   ├── nlp_service.py
   │   │   ├── booking.py
   │   │   └── tts_service.py
   │   ├── utils/
   │   │   └── preprocess.py
   │   ├── db.py
   │   └── main.py
   ├── models/
   │   ├── intent_model/
   │   │   ├── train_intent.py
   │   │   └── models/intent_model/output/
   │   └── ner_model/
   │       ├── train_ner.py
   │       └── models/ner_model/output/
   ├── data/
   │   ├── datasets/
   │   │   ├── massive/
   │   │   └── multiconer_v2/
   │   └── load_multiconer.py
   ├── tmp/
   ├── requirements.txt
   ├── TODO.md
   └── voice_booking.db
   ```

### Step 1: Speech-to-Text (ASR)

**Goal:** Receive an audio file, detect its language, and convert it to text with confidence scores.

**Implementation Details:**

- Framework & Model: FastAPI endpoint with Faster Whisper for ASR. Tested with "small", "medium", and "large-v3" models.
- Supported Audio Formats: flac, m4a, mp3, mp4, mpeg, mpga, ogg, wav, webm, oa, opus.
- Temporary Storage: Audio files saved temporarily in `tmp/` folder.
- Validation: File type and extension validation with error responses for unsupported files.

**Endpoint:**

- `POST /asr/transcribe`
- Accepts: multipart/form-data with one audio file.
- Returns:
  ```json
  {
    "text": "Transcribed text",
    "language": "Detected language code",
    "confidence": 0.95
  }
  ```

**Testing:**

- Successfully transcribed valid formats like .opus.
- Rejected unsupported files and handled missing uploads.
- Works on CPU/GPU (note CUDA memory limits for large models).

### Step 2: NLP (Intent + Entity Extraction)

**Phase 1: NLP Service Design (app/routes/nlp_service.py)**

- Imported libraries: `transformers` for BERT models and pipelines.
- Created `predict_intent()`: Uses fine-tuned BERT for intent classification.
- Built NER pipeline: Uses fine-tuned BERT for entity extraction.
- Endpoint: `POST /nlp/parse`
  - Input: Text string.
  - Returns:
    ```json
    {
      "intent": "book_appointment",
      "confidence": [[0.0, 1.0]],
      "entities": {
        "PERSON": "Dr. Ahmed",
        "DATE": "tomorrow"
      }
    }
    ```

**Phase 2: Preparing for Fine-Tuning**

- Realized general BERT-base-multilingual-cased needs domain-specific fine-tuning for booking intents.

**Phase 3: Dataset Preparation (app/utils/preprocess.py)**

- Downloaded and organized MASSIVE (for intent) and MultiCoNER v2 (for NER) datasets using Hugging Face Hub.
- Set up directories for datasets.
- Target languages: English, Arabic, Hindi, Bengali.

**Phase 4: Fine-tuning the Intent Model (models/intent_model/train_intent.py)**

- Loaded MASSIVE dataset, filtered languages.
- Relabeled intents: booking-related → "book_appointment", others → "other".
- Tokenized text, used Hugging Face Trainer.
- Saved fine-tuned model in `models/intent_model/output/checkpoint-8637/`.

**Phase 5: Fine-tuning the NER Model (models/ner_model/train_ner.py)**

- Loaded MultiCoNER v2, filtered languages.
- Tokenized, trained with Trainer.
- Saved fine-tuned model in `models/ner_model/output/checkpoint-1836/`.

**Updates to nlp_service.py:**

- Switched to fine-tuned models.
- Added fallback in `predict_intent()`: If model predicts "other" but text has "book"/"appointment", override to "book_appointment".

### Step 3 & 4: FastAPI Backend (Booking Logic) & Database

**Goal:** Process booking requests, validate intents, extract entities, manage database.

**Implementation Details:**

- Framework: FastAPI with SQLAlchemy ORM.
- Database: SQLite (`voice_booking.db`), switchable to PostgreSQL via environment variable `DATABASE_URL`.
- Models: User (id, name, email, phone, created_at), Doctor (id, name, specialty, available_days, created_at), Booking (id, user_id, doctor_id, date, status, created_at).

**Phase 1: Backend Service Design (app/routes/booking.py)**

- Imported: FastAPI, SQLAlchemy, Pydantic.
- Defined `BookingRequest` model.
- Endpoint: `POST /book`
  - Input:
    ```json
    {
      "text": "Transcribed text",
      "user_name": "John Doe",
      "user_email": "john@example.com",
      "user_phone": "01100800344",
      "doctor_name": "Dr. Ahmed",
      "date": "2023-10-01T10:00:00"
    }
    ```
  - Logic: Check intent via NLP, find/create user, lookup doctor, parse date, create booking.
  - Returns:
    ```json
    {
      "message": "Booking created successfully",
      "booking_id": 1,
      "status": "pending"
    }
    ```

**Phase 2: Database Setup (app/db.py)**

- SQLAlchemy setup: Base, engine, SessionLocal.
- Defined models with relationships.
- `get_db()` dependency for sessions.

**Phase 3: Main App Integration (app/main.py)**

- FastAPI app with included routers: ASR, NLP, Booking, TTS.
- Root endpoint: `GET /` returns welcome message.


### Step 5: TTS (Text-to-Speech)

**Goal:** Generate natural spoken audio replies for booking confirmations in supported languages.

**Implementation Details:**

- Framework & Library: Used gTTS (Google Text-to-Speech) for high-quality, multilingual text-to-speech synthesis. Supports English, Hindi, Bengali, and Arabic (Saudi dialect).
- Language Mapping: Mapped language codes to gTTS supported languages:
  - en → 'en' (English)
  - hi → 'hi' (Hindi)
  - bn → 'bn' (Bengali)
  - ar → 'ar' (Arabic)
- Audio Format: Generates MP3 files for compatibility and smaller file sizes.
- Temporary Storage: Synthesized audio saved in `tmp/` directory with unique filenames to avoid conflicts.
- Error Handling: Validates language codes; defaults to 'en' if invalid. Handles gTTS exceptions gracefully.

**Endpoint:**

- `POST /tts/synthesize`
- Accepts: Query parameters `text` (string) and `lang` (string, default 'en').
- Example Request: `POST /tts/synthesize?text=Your%20appointment%20has%20been%20booked&lang=en`
- Returns:
  ```json
  {
    "audio_file": "C:\\Users\\DELL\\voice-booking\\tmp\\unique_filename.mp3",
    "message": "Audio synthesized successfully"
  }
  ```

**Integration:**

- Added `tts_router` to `app/main.py` for routing.
- Called after successful booking to generate confirmation audio.
- Tested with sample texts in all supported languages.

**Testing:**

- Successfully synthesized audio for English confirmations (e.g., "Your appointment has been booked successfully").
- Verified MP3 file generation and playback.
- Handled invalid languages by falling back to English.
- Integrated with booking flow for end-to-end voice replies.

## Testing and Verification

For detailed testing instructions, refer to the separate `TESTING.md` file.

## Running the Application

1. Activate environment: `conda activate voicebook`
2. Run server: `python -m uvicorn app.main:app --reload`
3. Access at `http://127.0.0.1:8000`

## Future Enhancements

- Switch to PostgreSQL for production.
- Improve NER for better entity extraction.
- Add more languages or TTS options.
- Implement user authentication and advanced booking features.
