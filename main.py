import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal

import pyodbc
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import ValidationError, BaseModel
from rapidfuzz import process, fuzz

from models import ChatRequest, ChatResponse, IntentStore, SupportedLang

# ---------- PATHS & FASTAPI APP ----------
BASE_DIR = Path(__file__).parent
INTENTS_PATH = BASE_DIR / "intents.json"

app = FastAPI(
    title="Land Records FAQ Chatbot",
    description=(
        "Lightweight intent-based chatbot for common land record queries "
        "in English and Telugu, plus SQL-based land record lookup."
    ),
    version="2.1.0",  # Updated for Azure SQL
)

# Serve static files if needed (CSS/JS later)
app.mount("/static", StaticFiles(directory=BASE_DIR, html=True), name="static")

# Serve index.html at root (frontend UI)
@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse(BASE_DIR / "index.html")

# Enable CORS for web UI (if you ever serve from another origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development; later restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- AZURE SQL CONFIG (Environment Variables) ----------
AZURE_SQL_SERVER = os.getenv("AZURE_SQL_SERVER")  # your-server.database.windows.net
AZURE_SQL_DATABASE = os.getenv("AZURE_SQL_DATABASE", "LandChatbotDemo")
AZURE_SQL_USER = os.getenv("AZURE_SQL_USER")       # user@your-server
AZURE_SQL_PASSWORD = os.getenv("AZURE_SQL_PASSWORD")

def get_connection():
    """Azure SQL connection with required encryption settings."""
    if not all([AZURE_SQL_SERVER, AZURE_SQL_USER, AZURE_SQL_PASSWORD]):
        raise ValueError("Missing Azure SQL environment variables")
    
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={AZURE_SQL_SERVER};"
        f"DATABASE={AZURE_SQL_DATABASE};"
        f"UID={AZURE_SQL_USER};"
        f"PWD={AZURE_SQL_PASSWORD};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )
    return pyodbc.connect(conn_str)

def list_states():
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT StateID, StateName FROM States ORDER BY StateID")
        return cur.fetchall()

def list_districts(state_id: int):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT DistrictID, DistrictName "
            "FROM Districts WHERE StateID = ? ORDER BY DistrictID",
            (state_id,),
        )
        return cur.fetchall()

def list_mandals(district_id: int):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT MandalID, MandalName "
            "FROM Mandals WHERE DistrictID = ? ORDER BY MandalID",
            (district_id,),
        )
        return cur.fetchall()

def get_land_record(state_id: int, district_id: int, mandal_id: int, ror_number: str):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT OwnerName, Email, Phone
            FROM LandRecords
            WHERE StateID = ? AND DistrictID = ? AND MandalID = ? AND RORNumber = ?
            """,
            (state_id, district_id, mandal_id, ror_number),
        )
        return cur.fetchone()

# ---------- INTENT JSON CONFIG ----------
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

INTENT_STORE: IntentStore = load_intents()

def build_pattern_index(
    store: IntentStore,
) -> Dict[SupportedLang, List[Tuple[str, str]]]:
    index: Dict[SupportedLang, List[Tuple[str, str]]] = {"en": [], "te": []}
    for intent in store.intents:
        for pattern in intent.patterns.en:
            index["en"].append((pattern.lower(), intent.intent_id))
        for pattern in intent.patterns.te:
            index["te"].append((pattern.lower(), intent.intent_id))
    return index

PATTERN_INDEX = build_pattern_index(INTENT_STORE)

def find_best_intent(
    message: str, lang: SupportedLang, score_threshold: float = 60.0
):
    """Use RapidFuzz to find the best matching intent for the user's message."""
    candidates = PATTERN_INDEX.get(lang, [])
    if not candidates:
        return None, 0.0, True

    texts = [p[0] for p in candidates]

    best = process.extractOne(
        message.lower(),
        texts,
        scorer=fuzz.WRatio,
    )

    if best is None:
        return None, 0.0, True

    score, best_text = best[1], best[0]
    best_idx = texts.index(best_text)
    _, intent_id = candidates[best_idx]

    if score < score_threshold:
        fallback_intent = next(
            (i for i in INTENT_STORE.intents if i.intent_id == "fallback"),
            None,
        )
        return fallback_intent, float(score), True

    matched_intent = next(
        (i for i in INTENT_STORE.intents if i.intent_id == intent_id),
        None,
    )
    if matched_intent is None:
        fallback_intent = next(
            (i for i in INTENT_STORE.intents if i.intent_id == "fallback"),
            None,
        )
        return fallback_intent, float(score), True

    return matched_intent, float(score), matched_intent.intent_id == "fallback"

# ---------- SYSTEM & TEST ROUTES ----------
@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok", "db": "azure_sql"}

@app.get("/states-test")
def states_test():
    try:
        rows = list_states()
        return [{"id": r.StateID, "name": r.StateName} for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB Error: {str(e)}")

# ---------- SIMPLE SESSION STATE FOR LAND LOOKUP ----------
class FlowState(BaseModel):
    mode: Literal["unknown", "faq", "land_lookup"] = "unknown"
    step: Literal["ask_state", "ask_district", "ask_mandal", "ask_ror", "done"] = "done"
    state_id: Optional[int] = None
    district_id: Optional[int] = None
    mandal_id: Optional[int] = None

sessions: Dict[str, FlowState] = {}

def get_or_create_session(session_id: str) -> FlowState:
    state = sessions.get(session_id)
    if state is None:
        state = FlowState()  # mode="unknown", step="done"
        sessions[session_id] = state
    return state

def format_list_with_numbers(title: str, rows, name_attr: str) -> str:
    lines = [title]
    for i, r in enumerate(rows, start=1):
        lines.append(f"{i}) {getattr(r, name_attr)}")
    return "\n".join(lines)

def handle_land_lookup(req: ChatRequest) -> ChatResponse:
    """Multi-step flow: state -> district -> mandal -> ROR lookup via SQL."""
    session_id = req.session_id or "default"
    state = get_or_create_session(session_id)

    # Start or restart flow: ask for state
    if state.step == "done":
        try:
            rows = list_states()
        except Exception as e:
            reply = (
                "డేటాబేస్ కనెక్షన్ లోపం. నేరుగా అడ్మిన్‌ని సంప్రదించండి."
                if req.lang == "te"
                else "Database connection error. Contact admin directly."
            )
            return ChatResponse(
                reply=reply,
                intent_id="land_lookup_error",
                lang=req.lang,
                score=1.0,
                fallback=True,
            )
        
        if not rows:
            reply = (
                "సిస్టంలో ఎలాంటి రాష్ట్రాలు లభ్యం కావట్లేదు."
                if req.lang == "te"
                else "No states available in the system."
            )
            return ChatResponse(
                reply=reply,
                intent_id="land_lookup_error",
                lang=req.lang,
                score=1.0,
                fallback=True,
            )
        state.step = "ask_state"
        title = (
            "దయచేసి మీ రాష్ట్రాన్ని నంబర్ ద్వారా ఎంచుకోండి:"
            if req.lang == "te"
            else "Please choose your state by number:"
        )
        text = format_list_with_numbers(title, rows, "StateName")
        return ChatResponse(
            reply=text,
            intent_id="ask_state",
            lang=req.lang,
            score=1.0,
            fallback=False,
        )

    # Existing session
    state = sessions[session_id]
    msg = req.message.strip()

    # Step 1: choose state
    if state.step == "ask_state":
        try:
            idx = int(msg)
        except ValueError:
            reply = (
                "రాష్ట్రానికి సరైన నంబర్ ఇవ్వండి."
                if req.lang == "te"
                else "Please enter a valid number for the state."
            )
            return ChatResponse(
                reply=reply,
                intent_id="ask_state_again",
                lang=req.lang,
                score=0.0,
                fallback=False,
            )

        rows = list_states()
        if idx < 1 or idx > len(rows):
            reply = (
                "తప్పు ఎంపిక. దయచేసి జాబితాలో ఉన్న నంబర్‌లలో ఒకదాన్ని ఇవ్వండి."
                if req.lang == "te"
                else "Invalid choice. Please enter a number from the list."
            )
            return ChatResponse(
                reply=reply,
                intent_id="ask_state_again",
                lang=req.lang,
                score=0.0,
                fallback=False,
            )

        chosen = rows[idx - 1]
        state.state_id = chosen.StateID
        state.step = "ask_district"

        districts = list_districts(state.state_id)
        if not districts:
            reply = (
                "ఆ రాష్ట్రానికి సంబంధించి ఎలాంటి జిల్లాలు లభ్యం కావట్లేదు."
                if req.lang == "te"
                else "No districts found for that state."
            )
            state.step = "done"
            return ChatResponse(
                reply=reply,
                intent_id="land_lookup_error",
                lang=req.lang,
                score=1.0,
                fallback=True,
            )

        title = (
            f"మీరు {chosen.StateName} ఎంచుకున్నారు. ఇప్పుడు మీ జిల్లాను నంబర్ ద్వారా ఎంచుకోండి:"
            if req.lang == "te"
            else f"You selected {chosen.StateName}. Now choose your district by number:"
        )
        text = format_list_with_numbers(title, districts, "DistrictName")
        return ChatResponse(
            reply=text,
            intent_id="ask_district",
            lang=req.lang,
            score=1.0,
            fallback=False,
        )

    # Step 2: choose district
    if state.step == "ask_district":
        try:
            idx = int(msg)
        except ValueError:
            reply = (
                "జిల్లాకు సరైన నంబర్ ఇవ్వండి."
                if req.lang == "te"
                else "Please enter a valid number for the district."
            )
            return ChatResponse(
                reply=reply,
                intent_id="ask_district_again",
                lang=req.lang,
                score=0.0,
                fallback=False,
            )

        districts = list_districts(state.state_id)
        if idx < 1 or idx > len(districts):
            reply = (
                "తప్పు ఎంపిక. దయచేసి జాబితాలో ఉన్న నంబర్‌లలో ఒకదాన్ని ఇవ్వండి."
                if req.lang == "te"
                else "Invalid choice. Please enter a number from the list."
            )
            return ChatResponse(
                reply=reply,
                intent_id="ask_district_again",
                lang=req.lang,
                score=0.0,
                fallback=False,
            )

        chosen = districts[idx - 1]
        state.district_id = chosen.DistrictID
        state.step = "ask_mandal"

        mandals = list_mandals(state.district_id)
        if not mandals:
            reply = (
                "ఆ జిల్లాకు సంబంధించి ఎలాంటి మండలాలు లభ్యం కావట్లేదు."
                if req.lang == "te"
                else "No mandals found for that district."
            )
            state.step = "done"
            return ChatResponse(
                reply=reply,
                intent_id="land_lookup_error",
                lang=req.lang,
                score=1.0,
                fallback=True,
            )

        title = (
            f"మీరు {chosen.DistrictName} ఎంచుకున్నారు. ఇప్పుడు మీ మండలాన్ని నంబర్ ద్వారా ఎంచుకోండి:"
            if req.lang == "te"
            else f"You selected {chosen.DistrictName}. Now choose your mandal by number:"
        )
        text = format_list_with_numbers(title, mandals, "MandalName")
        return ChatResponse(
            reply=text,
            intent_id="ask_mandal",
            lang=req.lang,
            score=1.0,
            fallback=False,
        )

    # Step 3: choose mandal
    if state.step == "ask_mandal":
        try:
            idx = int(msg)
        except ValueError:
            reply = (
                "మండలానికి సరైన నంబర్ ఇవ్వండి."
                if req.lang == "te"
                else "Please enter a valid number for the mandal."
            )
            return ChatResponse(
                reply=reply,
                intent_id="ask_mandal_again",
                lang=req.lang,
                score=0.0,
                fallback=False,
            )

        mandals = list_mandals(state.district_id)
        if idx < 1 or idx > len(mandals):
            reply = (
                "తప్పు ఎంపిక. దయచేసి జాబితాలో ఉన్న నంబర్‌లలో ఒకదాన్ని ఇవ్వండి."
                if req.lang == "te"
                else "Invalid choice. Please enter a number from the list."
            )
            return ChatResponse(
                reply=reply,
                intent_id="ask_mandal_again",
                lang=req.lang,
                score=0.0,
                fallback=False,
            )

        chosen = mandals[idx - 1]
        state.mandal_id = chosen.MandalID
        state.step = "ask_ror"

        reply = (
            "దయచేసి మీ ROR నంబర్‌ను నమోదు చేయండి:"
            if req.lang == "te"
            else "Please enter your ROR number:"
        )
        return ChatResponse(
            reply=reply,
            intent_id="ask_ror",
            lang=req.lang,
            score=1.0,
            fallback=False,
        )

    # Step 4: enter ROR and show record
    if state.step == "ask_ror":
        ror = msg
        try:
            record = get_land_record(
                state.state_id, state.district_id, state.mandal_id, ror
            )
        except Exception as e:
            reply = (
                "డేటాబేస్ లోపం. దయచేసి మళ్లీ ప్రయత్నించండి."
                if req.lang == "te"
                else "Database error. Please try again."
            )
            state.step = "done"
            return ChatResponse(
                reply=reply,
                intent_id="land_lookup_error",
                lang=req.lang,
                score=1.0,
                fallback=True,
            )
        state.step = "done"

        if record:
            if req.lang == "te":
                reply = (
                    f"ROR {ror} కోసం వివరాలు:\n"
                    f"పేరు: {record.OwnerName}\n"
                    f"ఇమెయిల్: {record.Email}\n"
                    f"ఫోన్: {record.Phone}"
                )
            else:
                reply = (
                    f"Details for ROR {ror}:\n"
                    f"Name: {record.OwnerName}\n"
                    f"Email: {record.Email}\n"
                    f"Phone: {record.Phone}"
                )
        else:
            reply = (
                "ఆ ROR నంబర్‌కు భూ రికార్డు లభ్యం కాలేదు."
                if req.lang == "te"
                else "No land record found for that ROR number."
            )

        return ChatResponse(
            reply=reply,
            intent_id="show_land_record",
            lang=req.lang,
            score=1.0,
            fallback=False,
        )

    # Fallback
    reply = (
        "లుకప్ ప్రక్రియలో ఏదో లోపం వచ్చింది."
        if req.lang == "te"
        else "Something went wrong in the lookup flow."
    )
    return ChatResponse(
        reply=reply,
        intent_id="land_lookup_error",
        lang=req.lang,
        score=0.0,
        fallback=True,
    )

# ---------- MAIN CHAT ENDPOINT ----------
@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest):
    """
    Single chat endpoint:
    - First asks if user wants ROR lookup.
    - If yes -> land_lookup flow via SQL.
    - If no  -> FAQ mode using intents.
    """
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    session_id = request.session_id or "default"
    state = get_or_create_session(session_id)

    # 1) If user hasn't chosen yet, ask them
    if state.mode == "unknown":
        lower = message.lower()

        # user clearly answered yes -> go to land lookup
        if lower in {"yes", "y", "ha", "haan"}:
            state.mode = "land_lookup"
            state.step = "done"  # let handle_land_lookup start fresh
            return handle_land_lookup(request)

        # user clearly answered no -> go to FAQ
        if lower in {"no", "n", "kadu", "ledhu", "nope"}:
            state.mode = "faq"
            reply = (
                "సరే, నేను భూ రికార్డులకు సంబంధించిన సాధారణ ప్రశ్నలకు సమాధానం ఇస్తాను. "
                "దయచేసి మీ ప్రశ్నను టైప్ చేయండి."
                if request.lang == "te"
                else "Okay, I will answer general land record questions. Please type your question."
            )
            return ChatResponse(
                reply=reply,
                intent_id="switch_to_faq",
                lang=request.lang,
                score=1.0,
                fallback=False,
            )

        # otherwise, ask the ROR question first
        ask_text = (
            "మీరు ROR నంబర్ వివరాలు తెలుసుకోవాలనుకుంటున్నారా? "
            "దయచేసి 'yes' లేదా 'no' అని టైప్ చేయండి."
            if request.lang == "te"
            else "Are you trying to find details of an ROR number? Please reply 'yes' or 'no'."
        )
        return ChatResponse(
            reply=ask_text,
            intent_id="ask_ror_intent",
            lang=request.lang,
            score=1.0,
            fallback=False,
        )

    # 2) User already chose land lookup -> continue SQL flow
    if state.mode == "land_lookup":
        return handle_land_lookup(request)

    # 3) User already chose FAQ -> use existing FAQ behavior
    intent, score, is_fallback = find_best_intent(message, request.lang)
    if intent is None:
        raise HTTPException(status_code=500, detail="Intent matching failed")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
