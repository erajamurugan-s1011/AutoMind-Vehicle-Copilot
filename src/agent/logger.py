"""
AutoMind — agent decision logging.

Every LangGraph node's execution gets appended as one JSON line to
logs/agent_decisions.jsonl. This is a lightweight, dependency-free
alternative to a full tracing platform (LangSmith, etc.) — good enough to
demonstrate you understand WHY observability matters for agentic systems,
and to actually debug routing decisions when something looks wrong.

Each line looks like:
{
  "timestamp": "2026-07-10T14:32:01",
  "session_id": "abc123",
  "node": "classify_intent",
  "latency_sec": 1.82,
  "input": {"user_query": "check engine light is on"},
  "output": {"intent": "symptom_diagnosis", "identified_symptom": "Engine misfire"}
}
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.config import LOGS_DIR

LOG_PATH = LOGS_DIR / "agent_decisions.jsonl"


def log_event(session_id: str, node: str, input_data: Dict[str, Any],
              output_data: Dict[str, Any], latency_sec: float) -> None:
    """Appends one structured decision record. Never raises — logging must
    never break the actual agent request if something about serialization fails."""
    try:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "node": node,
            "latency_sec": round(latency_sec, 3),
            "input": input_data,
            "output": output_data,
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        print(f"⚠️  Decision logging failed (non-fatal): {e}")


def read_recent_events(limit: int = 20, session_id: str = None) -> list:
    """Reads the most recent N logged events, optionally filtered to one session."""
    if not LOG_PATH.exists():
        return []

    events = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                event = json.loads(line)
                if session_id is None or event.get("session_id") == session_id:
                    events.append(event)
            except json.JSONDecodeError:
                continue

    return events[-limit:]