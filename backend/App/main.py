from fastapi import FastAPI
from app.api.meetings import router as meeting_router

app = FastAPI(title="Meet Transcript AI")

app.include_router(meeting_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"status": "ok"}
