# app/api/meetings.py
from fastapi import APIRouter, UploadFile, File
import uuid, shutil

router = APIRouter()

@router.post("/meetings/upload")
async def upload_audio(file: UploadFile = File(...)):
    meeting_id = str(uuid.uuid4())
    path = f"uploads/{meeting_id}.wav"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "meeting_id": meeting_id,
        "audio_path": path
    }
