"""
AutoMind — agent state.
This TypedDict is the single object that flows through every LangGraph
node. Each node reads what it needs and returns a partial update, which
LangGraph merges into the running state.

Multi-turn diagnosis (the "check engine light -> clarifying question ->
narrowed cause -> fix" flow) works because `identified_symptom` and
`candidate_causes` persist ACROSS turns — the API layer (built in the
next step) keeps one AgentState per conversation/session and feeds it
back in on every new user message.
"""

from typing import TypedDict, List, Dict, Optional


class AgentState(TypedDict, total=False):
    # ---- conversation ----
    messages: List[Dict[str, str]]      # full chat history: [{"role": "user"/"assistant", "content": ...}]
    user_query: str                      # the latest user message being processed this turn
    vehicle_manual: Optional[str]        # source_file name of the user's selected vehicle manual, or None
    session_id: Optional[str]            # used only for decision logging, not agent logic

    # ---- routing ----
    intent: Optional[str]                # "symptom_diagnosis" | "manual_question" | "general"

    # ---- symptom diagnosis path (Neo4j) ----
    identified_symptom: Optional[str]    # matched Symptom node name, or None if unmatched/unclear
    candidate_causes: Optional[List[Dict]]  # ranked causes with fixes, from kg_query.full_diagnosis_path
    needs_clarification: bool            # True if candidate_causes are too ambiguous to answer directly
    clarification_question: Optional[str]

    # ---- manual lookup path (FAISS) ----
    manual_context: Optional[List[Dict]] # retrieved chunks with text/page/source

    # ---- output ----
    final_answer: Optional[str]