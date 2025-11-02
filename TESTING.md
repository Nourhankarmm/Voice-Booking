# Voice Booking System Testing Guide

This guide provides detailed steps to test each component of the Voice Booking System using Swagger UI (accessible at `http://127.0.0.1:8000/docs`) or curl commands. Ensure the server is running with `python -m uvicorn app.main:app --reload`.

## 1. ASR (Speech-to-Text) Testing

**Endpoint:** `POST /asr/transcribe`

**Purpose:** Upload an audio file and get transcribed text with language detection.

**Steps in Swagger:**

1. Navigate to `/asr/transcribe` endpoint.
2. Click "Try it out".
3. Upload an audio file (e.g., .opus, .wav) in the "file" field.
4. Click "Execute".
5. Check response: Should return JSON with "text", "language", and "confidence".

**Curl Example (English):**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/asr/transcribe' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path/to/your/audio.opus'
```

**Expected Output (English):**

```json
{
  "text": "I want to book an appointment with Dr. Ahmed",
  "language": "en",
  "confidence": 0.95
}
```

**Curl Example (Hindi):**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/asr/transcribe' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path/to/your/hindi_audio.opus'
```

**Expected Output (Hindi):**

```json
{
  "text": "मैं डॉक्टर अहमद के साथ अपॉइंटमेंट बुक करना चाहता हूं",
  "language": "hi",
  "confidence": 0.92
}
```

**Curl Example (Bengali):**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/asr/transcribe' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path/to/your/bengali_audio.opus'
```

**Expected Output (Bengali):**

```json
{
  "text": "আমি ডাক্তার আহমেদের সাথে একটি অ্যাপয়েন্টমেন্ট বুক করতে চাই",
  "language": "bn",
  "confidence": 0.9
}
```

**Curl Example (Arabic - Saudi):**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/asr/transcribe' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path/to/your/saudi_arabic_audio.opus'
```

**Expected Output (Arabic - Saudi):**

```json
{
  "text": "أريد حجز موعد مع الدكتور أحمد",
  "language": "ar",
  "confidence": 0.93
}
```

**Verification:** Ensure transcription is accurate, language is detected correctly (en, hi, bn, ar), and confidence is high for each language.

## 2. NLP (Intent + Entity Extraction) Testing

**Endpoint:** `POST /nlp/parse`

**Purpose:** Analyze text for intent and extract entities.

**Steps in Swagger:**

1. Navigate to `/nlp/parse` endpoint.
2. Click "Try it out".
3. Enter text in the "text" field (e.g., "I want to book an appointment with Dr. Ahmed").
4. Click "Execute".
5. Check response: Should return intent, confidence, and entities.

**Curl Example (English):**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/nlp/parse' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"text": "I want to book an appointment with Dr. Ahmed"}'
```

**Expected Output (English):**

```json
{
  "intent": "book_appointment",
  "confidence": [[0.0, 1.0]],
  "entities": {
    "PERSON": "Dr. Ahmed"
  }
}
```

**Curl Example (Hindi):**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/nlp/parse' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"text": "मैं डॉक्टर अहमद के साथ अपॉइंटमेंट बुक करना चाहता हूं"}'
```

**Expected Output (Hindi):**

```json
{
  "intent": "book_appointment",
  "confidence": [[0.0, 1.0]],
  "entities": {
    "PERSON": "डॉक्टर अहमद"
  }
}
```

**Curl Example (Bengali):**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/nlp/parse' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"text": "আমি ডাক্তার আহমেদের সাথে একটি অ্যাপয়েন্টমেন্ট বুক করতে চাই"}'
```

**Expected Output (Bengali):**

```json
{
  "intent": "book_appointment",
  "confidence": [[0.0, 1.0]],
  "entities": {
    "PERSON": "ডাক্তার আহমেদ"
  }
}
```

**Curl Example (Arabic):**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/nlp/parse' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"text": "أريد حجز موعد مع الدكتور أحمد"}'
```

**Expected Output (Arabic):**

```json
{
  "intent": "book_appointment",
  "confidence": [[0.0, 1.0]],
  "entities": {
    "PERSON": "الدكتور أحمد"
  }
}
```

**Verification:** Intent should be "book_appointment" for booking text in all languages; entities should extract doctor names, dates, etc.

## 3. Booking Logic Testing

**Endpoint:** `POST /book`

**Purpose:** Create a booking after validating intent from text in any supported language.

**Steps in Swagger:**

1. Navigate to `/book` endpoint.
2. Click "Try it out".
3. Fill in the request body with text (in any language), user details, doctor name, and date.
4. Click "Execute".
5. Check response: Should create booking if intent is valid.

**Curl Example (English):**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/book' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "I want to book an appointment with Dr. Ahmed",
    "user_name": "John Doe",
    "user_email": "john@example.com",
    "user_phone": "01100800344",
    "doctor_name": "Dr. Ahmed",
    "date": "2023-10-01T10:00:00"
  }'
```

**Expected Output (Success - English):**

```json
{
  "message": "Booking created successfully",
  "booking_id": 1,
  "status": "pending"
}
```

**Curl Example (Hindi):**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/book' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "मैं डॉक्टर अहमद के साथ अपॉइंटमेंट बुक करना चाहता हूं",
    "user_name": "जॉन डो",
    "user_email": "john@example.com",
    "user_phone": "01100800344",
    "doctor_name": "डॉक्टर अहमद",
    "date": "2023-10-01T10:00:00"
  }'
```

**Expected Output (Success - Hindi):**

```json
{
  "message": "Booking created successfully",
  "booking_id": 2,
  "status": "pending"
}
```

**Curl Example (Bengali):**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/book' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "আমি ডাক্তার আহমেদের সাথে একটি অ্যাপয়েন্টমেন্ট বুক করতে চাই",
    "user_name": "জন ডো",
    "user_email": "john@example.com",
    "user_phone": "01100800344",
    "doctor_name": "ডাক্তার আহমেদ",
    "date": "2023-10-01T10:00:00"
  }'
```

**Expected Output (Success - Bengali):**

```json
{
  "message": "Booking created successfully",
  "booking_id": 3,
  "status": "pending"
}
```

**Curl Example (Arabic - Saudi):**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/book' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "أريد حجز موعد مع الدكتور أحمد",
    "user_name": "جون دو",
    "user_email": "john@example.com",
    "user_phone": "01100800344",
    "doctor_name": "الدكتور أحمد",
    "date": "2023-10-01T10:00:00"
  }'
```

**Expected Output (Success - Arabic - Saudi):**

```json
{
  "message": "Booking created successfully",
  "booking_id": 4,
  "status": "pending"
}
```

**Expected Output (Failure - Invalid Intent):**

```json
{
  "detail": "Intent is not to book an appointment"
}
```

**Verification:** For valid booking text in any language, booking should be created; for non-booking text, it should reject with 400 error. Check database for new records.

## 4. TTS (Text-to-Speech) Testing

**Endpoint:** `POST /tts/synthesize`

**Purpose:** Generate audio from text in specified language.

**Steps in Swagger:**

1. Navigate to `/tts/synthesize` endpoint.
2. Click "Try it out".
3. Enter text and lang (e.g., text="Your appointment is booked", lang="en").
4. Click "Execute".
5. Check response: Should return audio file path.

**Curl Example (English):**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/tts/synthesize?text=Your%20appointment%20has%20been%20booked&lang=en' \
  -H 'accept: application/json'
```

**Expected Output (English):**

```json
{
  "audio_file": "C:\\Users\\DELL\\voice-booking\\tmp\\unique_filename.mp3",
  "message": "Audio synthesized successfully"
}
```

**Curl Example (Hindi):**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/tts/synthesize?text=आपका%20अपॉइंटमेंट%20बुक%20कर%20दिया%20गया%20है&lang=hi' \
  -H 'accept: application/json'
```

**Expected Output (Hindi):**

```json
{
  "audio_file": "C:\\Users\\DELL\\voice-booking\\tmp\\unique_filename.mp3",
  "message": "Audio synthesized successfully"
}
```

**Curl Example (Bengali):**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/tts/synthesize?text=আপনার%20অ্যাপয়েন্টমেন্ট%20বুক%20করা%20হয়েছে&lang=bn' \
  -H 'accept: application/json'
```

**Expected Output (Bengali):**

```json
{
  "audio_file": "C:\\Users\\DELL\\voice-booking\\tmp\\unique_filename.mp3",
  "message": "Audio synthesized successfully"
}
```

**Curl Example (Arabic):**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/tts/synthesize?text=تم%20حجز%20موعدك&lang=ar' \
  -H 'accept: application/json'
```

**Expected Output (Arabic):**

```json
{
  "audio_file": "C:\\Users\\DELL\\voice-booking\\tmp\\unique_filename.mp3",
  "message": "Audio synthesized successfully"
}
```

**Verification:** Audio file should be generated in `tmp/` directory for each language; play it to confirm speech quality and correct language.

## 5. Full Flow Testing

**End-to-End Test:**

1. Use ASR to transcribe an audio file with booking request.
2. Take the transcribed text and test NLP for intent/entities.
3. Use the extracted info to test Booking endpoint.
4. On success, use TTS to generate confirmation audio.

**Example Full Flow Curl:**

- ASR: As above, get text.
- NLP: Use transcribed text.
- Booking: Use NLP output.
- TTS: Use success message.

**Verification:** Ensure the entire pipeline works: Audio → Text → Intent → Booking → Audio Reply.

## Additional Testing Tips

- **Error Handling:** Test with invalid inputs (e.g., unsupported audio format, missing fields, invalid dates).
- **Languages:** Test ASR and TTS with different languages (en, hi, bn, ar).
- **Database:** Check `voice_booking.db` for created records after booking.
- **Models:** Ensure fine-tuned models are loaded (check logs for any errors).
- **Performance:** Test with larger files or multiple requests.

If any step fails, check server logs for errors and verify model paths.
