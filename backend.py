# backend.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import rag_simple, rag_retriever, llm

import os
import joblib


app = FastAPI()

# Allow frontend (HTML) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # during dev; restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Arena classifier & routing (same logic you already had) ----

MODEL_PATH = "/Users/rijuphaiju/Documents/ytrag/models/arena_classifier.joblib"
arena_classifier = joblib.load(MODEL_PATH)

def predict_arena_auto(question: str) -> str:
    return arena_classifier.predict([question])[0]


def is_generic_greeting(text: str) -> bool:
    t = text.strip().lower()
    greetings = [
        "hi", "hello", "hey", "namaste", "namastey", "नमस्ते",
        "hi there", "good morning", "good afternoon", "good evening",
        "thanks", "thank you", "dhanyabad", "धन्यवाद",
    ]
    if len(t) <= 15 and any(t == g for g in greetings):
        return True
    return False


def decide_arena(message: str, arena_from_ui: str) -> str:
    text = message.lower().strip()

    # 0) User-selected arena overrides everything
    if arena_from_ui != "All (auto)":
        return arena_from_ui

    # 1) Greetings / empty → no specific act
    if is_generic_greeting(text) or not text:
        return "All (auto)"

    # 2) Strong rules (order matters)
    if any(w in text for w in ["single women", "single woman", "एकल महिला", "ekal mahila", "विधवा"]):
        return "Single Women Act"

    if any(w in text for w in ["pharmacy", "pharmasi", "फार्मेसी"]):
        return "Pharmacy Act"

    if any(w in text for w in ["immunization", "khop", "खोप", "इम्युनाइजेशन"]):
        return "Immunization Act"

    if any(w in text for w in ["sports", "sport", "खेलकुद", "खेल"]):
        return "Sports Act"

    if any(w in text for w in ["citizenship", "nagrita", "नागरिकता", "संविधान", "constitution"]):
        return "Constitution of Nepal"

    # 3) Fallback: classifier
    return predict_arena_auto(message)


# ---- Request/Response models ----

class ChatRequest(BaseModel):
    message: str
    arena: str = "All (auto)"


class ChatResponse(BaseModel):
    answer: str
    detected_arena: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    detected_arena = decide_arena(req.message, req.arena)

    answer = rag_simple(
        req.message,
        rag_retriever,
        llm,
        top_k=6,
        arena=detected_arena,
    )

    answer_with_note = f"{answer}\n\n[Detected arena: {detected_arena}]"

    return ChatResponse(answer=answer_with_note, detected_arena=detected_arena)
from fastapi.responses import HTMLResponse

# at the top: import Path
from pathlib import Path

@app.get("/", response_class=HTMLResponse)
def root():
    index_path = Path(__file__).parent / "index.html"
    return index_path.read_text(encoding="utf-8")