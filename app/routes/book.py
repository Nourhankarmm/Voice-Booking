from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

from app.db import get_db, User, Doctor, Booking
from .nlp_service import predict_intent, extract_entities, resolve_relative_date

book_router = APIRouter(
    prefix="/book",
    tags=["Booking"]
)

#Request Schema
class BookingRequest(BaseModel):
    text: str
    user_name: str
    user_email: EmailStr
    user_phone: Optional[str] = None
    doctor_name: str = None
    date: str = None  # ISO format (e.g. 2023-10-01T10:00:00)

#Booking Endpoint
@book_router.post("/")
def create_booking(request: BookingRequest, db: Session = Depends(get_db)):
    """
    1. Verify intent via NLP.
    2. Find/create user & doctor.
    3. Create booking in database.
    4. Return confirmation.
    """
    # Step 1: NLP Intent + Entities
    intent_data = predict_intent(request.text)
    intent = intent_data["intent"]
    entities = extract_entities(request.text)

    # Allow booking if intent is "book_appointment" or if required fields are provided (fallback for model inaccuracies)
    is_booking_intent = intent.lower() == "book_appointment" or (request.doctor_name and request.date) or (entities.get("PERSON") and entities.get("DATE"))
    if not is_booking_intent:
        raise HTTPException(status_code=400, detail="Intent is not to book an appointment.")

    # Step 2: Extract doctor name and date (from NER or request)
    doctor_name = request.doctor_name or entities.get("PERSON")
    date_str = request.date or entities.get("DATE")

    if not doctor_name:
        raise HTTPException(status_code=400, detail="Doctor name missing.")
    if not date_str:
        raise HTTPException(status_code=400, detail="Appointment date missing.")

    # Resolve relative date if needed
    if date_str and not date_str.startswith('20'):
        date_str = resolve_relative_date(date_str)

    # Validate date format
    try:
        appointment_date = datetime.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use ISO format.")

    # Step 3: Find or create user
    user = db.query(User).filter(User.email == request.user_email).first()
    if not user:
        user = User(name=request.user_name, email=request.user_email, phone=request.user_phone)
        db.add(user)
        db.commit()
        db.refresh(user)

    # Step 4: Find or create doctor
    doctor = db.query(Doctor).filter(Doctor.name == doctor_name).first()
    if not doctor:
        doctor = Doctor(name=doctor_name, specialty="General")
        db.add(doctor)
        db.commit()
        db.refresh(doctor)

    # Step 5: Create booking
    booking = Booking(
        user_id=user.id,
        doctor_id=doctor.id,
        date=appointment_date,
        status="confirmed"
    )
    db.add(booking)
    db.commit()
    db.refresh(booking)

    # Step 6: Confirmation message
    message = f"Your appointment with {doctor_name} has been booked for {appointment_date}."
    return {
        "message": message,
        "booking_id": booking.id,
        "status": "confirmed"
    }
