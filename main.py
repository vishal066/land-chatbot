import json
from pathlib import Path
from typing import Dict, List, Tuple

from fastapi import FastAPI, HTTPException
from rapidfuzz import process, fuzz
from pydantic import ValidationError

from models import ChatRequest, ChatResponse, IntentStore, SupportedLang

# Path to intents JSON
INTENTS_PATH = Path(__file__).parent / "intents.json"

app = FastAPI(
    title="Land Records FAQ Chatbot",
    description="Lightweight intent-based chatbot for common land record queries in English and Telugu.",
    version="1.0.0",
)


def load_intents() -> IntentStore:
    if not INTENTS_PATH.exists():
        raise FileNotFoundError(f"intents.json not found at {INTENTS_PATH}")
    with INTENTS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        store = IntentStore(**data)
    except ValidationError as e:
        raise RuntimeError(f"Invalid intents.json format: {e}")
    return store


# Load intents at startup
INTENT_STORE: IntentStore = load_intents()


# Build pattern index for RapidFuzz:
# {
#   "en": [("hi", "greet"), ("hello", "greet"), ...],
#   "te": [("నమస్తే", "greet"), ...]
# }
def build_pattern_index(store: IntentStore) -> Dict[SupportedLang, List[Tuple[str, str]]]:
    index: Dict[SupportedLang, List[Tuple[str, str]]] = {"en": [], "te": []}
    for intent in store.intents:
        for pattern in intent.patterns.en:
            index["en"].append((pattern, intent.intent_id))
        for pattern in intent.patterns.te:
            index["te"].append((pattern, intent.intent_id))
    return index


PATTERN_INDEX = build_pattern_index(INTENT_STORE)


def find_best_intent(message: str, lang: SupportedLang, score_threshold: float = 60.0):
    """
    Use RapidFuzz to find the best matching intent for the user's message
    in the given language. Returns (intent, score, is_fallback).
    """
    candidates = PATTERN_INDEX.get(lang, [])
    if not candidates:
        return None, 0.0, True

    # Extract just the pattern texts for RapidFuzz
    texts = [p[0] for p in candidates]

    # process.extractOne returns (best_match_text, score, index)
    best = process.extractOne(
        message,
        texts,
        scorer=fuzz.WRatio
    )

    if best is None:
        return None, 0.0, True

    best_text, score, best_idx = best
    _, intent_id = candidates[best_idx]

    if score < score_threshold:
        # Low confidence -> use fallback
        fallback_intent = next(
            (i for i in INTENT_STORE.intents if i.intent_id == "fallback"),
            None
        )
        return fallback_intent, float(score), True

    # Find the matched intent object
    matched_intent = next(
        (i for i in INTENT_STORE.intents if i.intent_id == intent_id),
        None
    )
    if matched_intent is None:
        # Should not normally happen if data is consistent
        fallback_intent = next(
            (i for i in INTENT_STORE.intents if i.intent_id == "fallback"),
            None
        )
        return fallback_intent, float(score), True

    return matched_intent, float(score), matched_intent.intent_id == "fallback"


@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok"}

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Land Records FAQ Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development; later restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest):
    """
    Chat endpoint:
    - Input: { "message": "...", "lang": "en" | "te" }
    - Output: best matching intent, reply, score, fallback flag
    """
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    intent, score, is_fallback = find_best_intent(message, request.lang)

    if intent is None:
        # No intents loaded or something went wrong
        raise HTTPException(status_code=500, detail="Intent matching failed")

    # Pick response in requested language; if missing, fallback to English
    if request.lang == "te":
        reply = intent.responses.te or intent.responses.en
    else:
        reply = intent.responses.en

    return ChatResponse(
        reply=reply,
        intent_id=intent.intent_id,
        lang=request.lang,
        score=score,
        fallback=is_fallback,
    )
