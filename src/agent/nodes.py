"""
AutoMind — LangGraph agent nodes.
Each function here is one node in the graph (graph.py wires them together).
Every node takes the current AgentState and returns a dict of fields to
merge back in.
"""

import json
from typing import Dict

from src.agent.state import AgentState
from src.agent.llm import chat, simple_prompt
from src.rag.vector_store import search as faiss_search
from src.knowledge_graph.kg_query import full_diagnosis_path, get_all_symptom_names


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# Decides whether this turn is a fault-diagnosis question, a manual lookup
# question, or general chit-chat. If diagnosis, also tries to match the
# user's wording to one of the known Symptom nodes in the graph.
# ---------------------------------------------------------------------------
def classify_intent(state: AgentState) -> Dict:
    known_symptoms = get_all_symptom_names()
    query = state["user_query"]

    prompt = f"""You are an intent classifier for a vehicle assistant.

Known vehicle fault symptoms in our database:
{json.dumps(known_symptoms)}

User message: "{query}"

Respond ONLY with valid JSON, no other text, in this exact format:
{{"intent": "symptom_diagnosis" | "manual_question" | "general",
  "matched_symptom": "<exact string from the known symptoms list, or null if none clearly match>"}}

Rules:
- Use "symptom_diagnosis" if the user is describing a problem, fault, warning light, or unusual behavior.
- Use "manual_question" if the user is asking how to operate a feature or find a procedure (e.g. "how do I sync bluetooth").
- Use "general" for greetings or anything unrelated to vehicle issues.
- Only set matched_symptom if it's a close, confident match to the list above. Otherwise null.
"""

    raw = simple_prompt(prompt, temperature=0.0)

    try:
        parsed = json.loads(raw.strip().strip("`").replace("json\n", ""))
    except (json.JSONDecodeError, ValueError):
        # LLM occasionally wraps JSON in text despite instructions — fail safe to general
        parsed = {"intent": "general", "matched_symptom": None}

    return {
        "intent": parsed.get("intent", "general"),
        "identified_symptom": parsed.get("matched_symptom"),
    }


# ---------------------------------------------------------------------------
# Node 2: query_kg
# Pulls the ranked cause/fix chain for the identified symptom and decides
# whether the top cause is confident enough to answer directly, or whether
# we need to ask a clarifying question first.
# ---------------------------------------------------------------------------
def query_kg(state: AgentState) -> Dict:
    symptom = state.get("identified_symptom")
    if not symptom:
        return {"needs_clarification": False, "candidate_causes": []}

    causes = full_diagnosis_path(symptom)

    if not causes:
        return {"needs_clarification": False, "candidate_causes": []}

    # Ask a clarifying question whenever there's more than one plausible
    # cause — a single dominant "high" cause with only "low" likelihood
    # alternatives is the only case confident enough to skip straight to
    # an answer. With today's seed data (only high/medium causes), this
    # means: 1 cause -> answer directly, 2+ causes -> always clarify first.
    needs_clarification = False
    if len(causes) > 1:
        alternatives_all_low = all(c["likelihood"] == "low" for c in causes[1:])
        confident_top = causes[0]["likelihood"] == "high" and alternatives_all_low
        needs_clarification = not confident_top

    return {"candidate_causes": causes, "needs_clarification": needs_clarification}


# ---------------------------------------------------------------------------
# Node 3: ask_clarification
# Generates a natural clarifying question that would help distinguish
# between the top candidate causes (e.g. spark plug vs ignition coil vs O2 sensor).
# ---------------------------------------------------------------------------
def ask_clarification(state: AgentState) -> Dict:
    causes = state.get("candidate_causes", [])
    symptom = state.get("identified_symptom")
    cause_names = [c["cause"] for c in causes[:3]]

    prompt = f"""You are AutoMind, an in-vehicle diagnostic assistant.
The user reported: "{symptom}"

Possible causes we're trying to narrow down between: {json.dumps(cause_names)}

Ask ONE short, specific clarifying question (max 2 sentences) that would
help a non-technical car owner distinguish between these causes. Don't
list the causes by name — ask about observable symptoms instead
(e.g. timing, sound, when it happens, dashboard lights).
"""
    question = simple_prompt(prompt, temperature=0.4)
    return {"clarification_question": question, "final_answer": question}


# ---------------------------------------------------------------------------
# Node 3b: narrow_causes
# Runs when the PREVIOUS turn asked a clarifying question and this turn is
# the user's answer to it. Uses the answer to pick the most likely cause
# out of the existing candidate_causes, instead of re-classifying from
# scratch (which would lose the diagnostic thread).
# ---------------------------------------------------------------------------
def narrow_causes(state: AgentState) -> Dict:
    causes = state.get("candidate_causes", [])
    symptom = state.get("identified_symptom")
    user_answer = state["user_query"]
    cause_names = [c["cause"] for c in causes]

    prompt = f"""You are diagnosing a vehicle fault: "{symptom}"
Candidate causes: {json.dumps(cause_names)}

The user was just asked a clarifying question and answered: "{user_answer}"

Based on this answer, respond ONLY with valid JSON:
{{"most_likely_cause": "<exact string from candidate causes list>"}}

If the answer doesn't clearly point to one cause, pick the one that was
originally highest likelihood.
"""
    raw = simple_prompt(prompt, temperature=0.0)
    try:
        parsed = json.loads(raw.strip().strip("`").replace("json\n", ""))
        picked = parsed.get("most_likely_cause")
    except (json.JSONDecodeError, ValueError):
        picked = None

    # reorder causes so the picked one is first; fall back to original order
    reordered = causes
    if picked:
        matched = [c for c in causes if c["cause"] == picked]
        rest = [c for c in causes if c["cause"] != picked]
        if matched:
            reordered = matched + rest

    return {"candidate_causes": reordered, "needs_clarification": False}


# ---------------------------------------------------------------------------
# Node 4: retrieve_manual
# FAISS search over the owner's manual chunks, optionally filtered by the
# vehicle system implied by the query.
# ---------------------------------------------------------------------------
def retrieve_manual(state: AgentState) -> Dict:
    query = state["user_query"]
    vehicle_manual = state.get("vehicle_manual")
    results = faiss_search(query, top_k=4, source_file_filter=vehicle_manual)
    return {"manual_context": results}


# ---------------------------------------------------------------------------
# Node 5: generate_response
# Combines whatever context is available (graph reasoning and/or manual
# excerpts) and asks Ollama to produce the final natural-language answer.
# ---------------------------------------------------------------------------
def generate_response(state: AgentState) -> Dict:
    query = state["user_query"]
    causes = state.get("candidate_causes") or []
    manual_chunks = state.get("manual_context") or []

    context_parts = []

    if causes:
        top = causes[0]
        context_parts.append(
            f"Most likely cause: {top['cause']} ({top['likelihood']} likelihood). "
            f"Recommended fix: {top['fix']} "
            f"({'DIY possible' if top['diy_possible'] else 'professional service recommended'}, "
            f"cost tier: {top['cost_tier']})."
        )
        if len(causes) > 1:
            others = ", ".join(c["cause"] for c in causes[1:3])
            context_parts.append(f"Other possible causes if this doesn't resolve it: {others}.")

    if manual_chunks:
        excerpt_text = "\n---\n".join(
            f"(source: {c['source_file']}, page {c['page_number']}): {c['text'][:500]}"
            for c in manual_chunks[:3]
        )
        context_parts.append(f"Relevant owner's manual excerpts:\n{excerpt_text}")

    context_block = "\n\n".join(context_parts) if context_parts else "No specific diagnostic or manual data found."

    system_prompt = """You are AutoMind, a helpful in-vehicle AI copilot. You explain
vehicle issues clearly to non-technical drivers. Be concise, practical, and
reassuring but honest about safety-critical issues (brakes, overheating).
Cite the manual page when you use manual content. Don't invent facts not
present in the context provided."""

    prompt = f"""User question: "{query}"

Context available:
{context_block}

Give a clear, friendly answer in 3-5 sentences."""

    answer = simple_prompt(prompt, system=system_prompt, temperature=0.4)
    return {"final_answer": answer}