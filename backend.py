# backend.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


import os
import joblib
import datetime
import json
from pathlib import Path
import csv

from rag_pipeline import rag_simple, rag_retriever, llm, rag_with_context
from rag_pipeline import llm as judge_llm

app = FastAPI()

# Allow frontend (HTML) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # during dev; restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Arena classifier & routing ----

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
    return len(t) <= 15 and any(t == g for g in greetings)


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


def judge_against_context(question: str, answer: str, context: str):
    prompt = f"""You are a STRICT evaluator for a legal RAG system for Nepali law.

Question:
{question}

Retrieved legal context (from official Nepali law PDFs):
{context}

Assistant's answer:
{answer}

Evaluate:

1. correctness (0-2):
   - 0: Answer is wrong, irrelevant, or mostly does not address the question.
   - 1: Answer is partially correct but misses important details.
   - 2: Answer correctly addresses the question in most important aspects.

2. faithfulness (0-2):
   - 0: Answer contradicts the context or clearly uses details not present in the context.
   - 1: Answer is somewhat supported by the context, but parts are vague or appear invented.
   - 2: Answer is strongly and clearly supported by the context and does NOT add new legal claims.

Be strict and DO NOT give 2 unless you are confident.

Return ONLY a JSON object, no explanation, like:
{{"correctness": 1, "faithfulness": 2}}
"""
    resp = judge_llm.invoke(prompt)
    try:
        scores = json.loads(resp.content)
    except Exception:
        scores = {"correctness": 0, "faithfulness": 0}
    return scores


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
    # Fallbacks in case of unexpected error
    fallback_answer = "माफ गर्नुहोस्, सिस्टममा एउटा समस्या आयो, कृपया पछि पुनः प्रयास गर्नुहोस्।"
    fallback_arena = "All (auto)"

    try:
        detected_arena = decide_arena(req.message, req.arena)

        # Use rag_with_context so we have context for evaluation
        answer, context = rag_with_context(
            req.message,
            rag_retriever,
            llm,
            top_k=6,
            arena=detected_arena,
        )

        answer_with_note = f"{answer}\n\n[Detected arena: {detected_arena}]"

        # ---------- A) Full model-comparison logging (all models) ----------
        # This will append to data/model_comparison_log.csv
        
        
        try:
            csv_path = Path("data/rag_evaluation_log_ui.csv")
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = csv_path.exists()

            # DEBUG: show what we send to the judge
            print("\n--- JUDGE INPUT DEBUG ---")
            print("Question:", req.message)
            print("Answer (first 200):", answer[:200])
            print("Context (first 200):", context[:200])

            if context:
                scores = judge_against_context(req.message, answer, context)
                correctness = scores.get("correctness", 0)
                faithfulness = scores.get("faithfulness", 0)
            else:
                correctness = 0
                faithfulness = 0

            ts = datetime.datetime.now().isoformat(timespec="seconds")

            with csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "timestamp",
                        "question",
                        "arena_used",
                        "answer",
                        "context",
                        "correctness",
                        "faithfulness",
                    ])
                writer.writerow([
                    ts,
                    req.message,
                    detected_arena,
                    answer,
                    context,
                    correctness,
                    faithfulness,
                ])

        except Exception as e:
            print("Eval logging error:", e)
     

        return ChatResponse(answer=answer_with_note, detected_arena=detected_arena)

    except Exception as e:
        print("chat_endpoint error:", e)
        return ChatResponse(answer=fallback_answer, detected_arena=fallback_arena)


@app.get("/", response_class=HTMLResponse)
def root():
    index_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))