from fastapi import APIRouter, File, UploadFile, HTTPException
from faster_whisper import WhisperModel
import uuid
import os

asr_router = APIRouter()
#model = WhisperModel("large-v3", device="cuda")
#model = WhisperModel("medium", device="cuda") 
model = WhisperModel("small", device="cpu")


TMP_DIR = "C:\\Users\\DELL\\voice-booking\\tmp"
os.makedirs(TMP_DIR, exist_ok=True)

SUPPORTED_FORMATS = {
    'audio/flac', 'audio/m4a', 'audio/mp3', 'audio/mp4', 'audio/mpeg', 'audio/mpga',
    'audio/ogg', 'audio/wav', 'audio/webm', 'audio/oa', 'audio/opus'
}
SUPPORTED_EXTENSIONS = {'.flac', '.m4a', '.mp3', '.mp4', '.mpeg', '.mpga', '.ogg', '.wav', '.webm', '.oa', '.opus'}

@asr_router.post("/asr/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Validate file type
    content_type = file.content_type
    filename = file.filename.lower() if file.filename else ""
    extension = os.path.splitext(filename)[1]
    
    if content_type not in SUPPORTED_FORMATS and extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type. Supported formats: flac, m4a, mp3, mp4, mpeg, mpga, ogg, wav, webm, oa, opus")
    
    temp_filename = f"{TMP_DIR}\\{uuid.uuid4().hex}_{file.filename}"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())
    segments, info = model.transcribe(temp_filename)
    text = " ".join([s.text for s in segments]).strip()
    confidence = getattr(info, "language_probability", None)
    os.remove(temp_filename)
    return {"text": text, "language": info.language, "confidence": confidence}
