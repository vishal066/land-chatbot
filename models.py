from pydantic import BaseModel, Field
from typing import Literal, List, Optional

# Supported languages: English, Telugu
SupportedLang = Literal["en", "te"]

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User query text")
    lang: SupportedLang = Field(
        "en",
        description="User language code: 'en' for English, 'te' for Telugu"
    )
    # Only session_id now; backend decides mode per session
    session_id: Optional[str] = Field(
        None,
        description="Session identifier to keep track of multi-step conversations"
    )

class ChatResponse(BaseModel):
    reply: str
    intent_id: str
    lang: SupportedLang
    score: float
    fallback: bool

class IntentPatterns(BaseModel):
    en: List[str]
    te: List[str]

class IntentResponses(BaseModel):
    en: str
    te: str

class Intent(BaseModel):
    intent_id: str
    patterns: IntentPatterns
    responses: IntentResponses

class IntentStore(BaseModel):
    intents: List[Intent]
