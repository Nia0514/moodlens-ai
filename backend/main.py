from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, json
from dotenv import load_dotenv
from voice_analyzer import analyze_voice
from emotion_fusion import fuse_emotions
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.5-flash"
print(f"DEBUG: Key loaded = {bool(GEMINI_API_KEY)}, starts with = {GEMINI_API_KEY[:8] if GEMINI_API_KEY else 'NONE'}")
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"

app = FastAPI(title="MoodLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Text Emotion ──────────────────────────────────────────
class TextRequest(BaseModel):
    text: str

@app.post("/analyze-text")
def analyze_text(req: TextRequest):
    prompt = f"""Analyze the emotion in this text: "{req.text}"

    Respond ONLY with a valid JSON object, no extra text, no markdown backticks:
    {{"primary_emotion": "sad", "confidence": 0.82, "secondary_emotion": "anxious", "explanation": "one short sentence"}}

    Only use these emotions: happy, sad, angry, anxious, calm, excited, neutral, frustrated"""

    response = client.models.generate_content(model=MODEL, contents=prompt)
    raw = response.text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── Voice Emotion ─────────────────────────────────────────
@app.post("/analyze-voice")
async def analyze_voice_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    return analyze_voice(contents)


# ── Emotion Fusion ────────────────────────────────────────
class FuseRequest(BaseModel):
    text_emotion: dict
    face_emotion: dict
    voice_emotion: dict

@app.post("/fuse-emotions")
def fuse(req: FuseRequest):
    return fuse_emotions(req.text_emotion, req.face_emotion, req.voice_emotion)


# ── Personalized Response ─────────────────────────────────
class ResponseRequest(BaseModel):
    emotion: str
    confidence: float

@app.post("/get-response")
def get_response(req: ResponseRequest):
    prompt = f"""The user is feeling {req.emotion} (confidence: {req.confidence}).
    Write exactly 2 sentences: one empathetic acknowledgment, one small actionable tip.
    Be warm and human, not clinical."""

    response = client.models.generate_content(model=MODEL, contents=prompt)
    return {"message": response.text.strip()}


# ── Health Check ──────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "MoodLens API is running"}