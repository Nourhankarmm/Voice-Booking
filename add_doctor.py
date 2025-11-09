from app.db import SessionLocal, Doctor

# Add a sample doctor
db = SessionLocal()
doctor = Doctor(name="Dr. Ahmed", specialty="General Medicine", available_days="Monday,Tuesday,Wednesday")
db.add(doctor)
db.commit()
db.refresh(doctor)
print(f"Added doctor: {doctor.name}, ID: {doctor.id}")
db.close()
