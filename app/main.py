from fastapi import FastAPI
from .routes.asr_service import asr_router
from .routes.nlp_service import nlp_router
from .routes.book import book_router
from .routes.tts_service import tts_router

# FastAPI App
app = FastAPI(title="Voice Booking API")

app.include_router(asr_router)
app.include_router(nlp_router)
app.include_router(book_router)
app.include_router(tts_router)

@app.get("/")
def root():
    return {"message": "Voice Booking API is running"}
