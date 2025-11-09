from fastapi import APIRouter, HTTPException
from gtts import gTTS
import os
import uuid

tts_router = APIRouter()

TMP_DIR = "C:\\Users\\DELL\\voice-booking\\tmp"
os.makedirs(TMP_DIR, exist_ok=True)

@tts_router.post("/tts/synthesize")
async def synthesize_speech(text: str, lang: str = "en"):
    """
    Synthesize text to speech using gTTS.
    Supported langs: en, hi, bn, ar (Saudi dialect approximated as ar).
    """
    try:
        # Map to gTTS supported languages
        lang_map = {
            "en": "en",
            "hi": "hi",
            "bn": "bn",
            "ar": "ar"  # Saudi dialect
        }
        tts_lang = lang_map.get(lang, "en")

        tts = gTTS(text=text, lang=tts_lang, slow=False)
        filename = f"{TMP_DIR}\\{uuid.uuid4().hex}.mp3"
        tts.save(filename)

        # For demo, return file path; in production, return audio data or URL
        return {"audio_file": filename, "message": "Audio synthesized successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")
