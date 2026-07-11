"""
AutoMind — chat routes.
The core endpoint the frontend (Streamlit, next step) calls. Persists
AgentState per session_id in memory so that a clarifying question asked
in one request is correctly followed up on in the next request from the
same session — this is what makes the multi-turn diagnosis flow work
over HTTP instead of just in a single Python process.

NOTE: in-memory session storage means state is lost on server restart and
doesn't scale across multiple server workers. That's fine for a portfolio
demo — mention Redis as the production upgrade path if asked in interviews.
"""

from typing import Dict
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.auth import get_current_user, TokenData
from src.agent.graph import run_agent_turn

router = APIRouter()

# session_id -> AgentState dict (in-memory; swap for Redis in production)
_sessions: Dict[str, dict] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str
    vehicle_manual: str | None = None   # source_file name, e.g. "TOYOTA 2025(Corolla).pdf"; None = search all manuals


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    needs_clarification: bool
    identified_symptom: str | None = None


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, current_user: TokenData = Depends(get_current_user)):
    prior_state = _sessions.get(request.session_id)

    updated_state = run_agent_turn(
        request.message,
        prior_state=prior_state,
        vehicle_manual=request.vehicle_manual,
        session_id=request.session_id,
    )
    _sessions[request.session_id] = updated_state

    return ChatResponse(
        session_id=request.session_id,
        reply=updated_state["final_answer"],
        needs_clarification=updated_state.get("needs_clarification", False),
        identified_symptom=updated_state.get("identified_symptom"),
    )


@router.delete("/chat/{session_id}")
def reset_session(session_id: str, current_user: TokenData = Depends(get_current_user)):
    """Clears a session's conversation history — useful for demo resets."""
    _sessions.pop(session_id, None)
    return {"status": "session cleared", "session_id": session_id}