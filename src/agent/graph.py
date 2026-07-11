"""
AutoMind — LangGraph agent orchestration.

THE FLOW:

    START
      |
      v
  classify_intent -----------------------------+
      |                                         |
      | intent == symptom_diagnosis             | intent == manual_question
      v                                         v
   query_kg                              retrieve_manual
      |                                         |
      | needs_clarification?                    v
      |                                   generate_response --> END
   ---+---
   |       |
  YES      NO
   |       |
   v       v
ask_clarification   retrieve_manual (supplement with manual excerpts)
   |                       |
   v                       v
  END              generate_response --> END

(intent == general routes straight to generate_response with no context)

Multi-turn works because the API layer (next step) persists AgentState
across turns per session — so if ask_clarification fires this turn, next
turn's classify_intent + query_kg re-run with the SAME identified_symptom
already known, now narrowing further based on the user's clarifying answer.
"""

import time
from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.logger import log_event
from src.agent.nodes import (
    classify_intent,
    query_kg,
    ask_clarification,
    narrow_causes,
    retrieve_manual,
    generate_response,
)


def with_logging(node_name: str, fn):
    """
    Wraps a node function so every execution is timed and logged via
    log_event — gives full visibility into what the agent decided and why,
    without touching the node functions themselves.
    """
    def wrapped(state: AgentState) -> dict:
        start = time.time()
        result = fn(state)
        elapsed = time.time() - start

        log_event(
            session_id=state.get("session_id", "unknown"),
            node=node_name,
            input_data={
                "user_query": state.get("user_query"),
                "identified_symptom": state.get("identified_symptom"),
                "vehicle_manual": state.get("vehicle_manual"),
            },
            output_data=result,
            latency_sec=elapsed,
        )
        return result
    return wrapped


def route_entry(state: AgentState) -> str:
    """
    If the PRIOR turn ended by asking a clarifying question (needs_clarification
    was True and we have candidate_causes waiting), this turn's user_query is
    the answer to that question — route straight to narrow_causes instead of
    re-classifying intent from scratch, which would lose the diagnostic thread.
    """
    if state.get("needs_clarification") and state.get("candidate_causes"):
        return "narrow_causes"
    return "classify_intent"


def route_after_classify(state: AgentState) -> str:
    intent = state.get("intent", "general")
    if intent == "symptom_diagnosis" and state.get("identified_symptom"):
        return "query_kg"
    elif intent == "manual_question":
        return "retrieve_manual"
    else:
        return "generate_response"


def route_after_kg(state: AgentState) -> str:
    if state.get("needs_clarification"):
        return "ask_clarification"
    else:
        return "retrieve_manual"  # supplement the KG answer with manual excerpts before responding


def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", with_logging("classify_intent", classify_intent))
    graph.add_node("query_kg", with_logging("query_kg", query_kg))
    graph.add_node("ask_clarification", with_logging("ask_clarification", ask_clarification))
    graph.add_node("narrow_causes", with_logging("narrow_causes", narrow_causes))
    graph.add_node("retrieve_manual", with_logging("retrieve_manual", retrieve_manual))
    graph.add_node("generate_response", with_logging("generate_response", generate_response))

    graph.set_conditional_entry_point(
        route_entry,
        {
            "narrow_causes": "narrow_causes",
            "classify_intent": "classify_intent",
        },
    )

    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "query_kg": "query_kg",
            "retrieve_manual": "retrieve_manual",
            "generate_response": "generate_response",
        },
    )

    graph.add_edge("narrow_causes", "retrieve_manual")

    graph.add_conditional_edges(
        "query_kg",
        route_after_kg,
        {
            "ask_clarification": "ask_clarification",
            "retrieve_manual": "retrieve_manual",
        },
    )

    graph.add_edge("ask_clarification", END)
    graph.add_edge("retrieve_manual", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()


# Compiled once, reused across requests
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent_graph()
    return _agent


def run_agent_turn(user_query: str, prior_state: dict = None, vehicle_manual: str = None,
                    session_id: str = None) -> dict:
    """
    Runs one turn of the agent. `prior_state` (if given) carries over
    identified_symptom / candidate_causes from previous turns, enabling
    multi-turn narrowing after a clarifying question. `vehicle_manual`
    (source_file name) filters manual retrieval to the user's selected
    vehicle — set it on the first call; it persists via prior_state on
    subsequent turns even if not passed again. `session_id` is used only
    for decision logging (see src/agent/logger.py), not routing logic.
    Returns the full updated state (the caller should persist it and pass
    it back in as prior_state on the next turn).
    """
    state = dict(prior_state) if prior_state else {}
    state["user_query"] = user_query
    if session_id is not None:
        state["session_id"] = session_id
    if vehicle_manual is not None:
        state["vehicle_manual"] = vehicle_manual
    else:
        state.setdefault("vehicle_manual", None)
    state.setdefault("messages", [])
    state["messages"].append({"role": "user", "content": user_query})

    agent = get_agent()
    result = agent.invoke(state)

    result["messages"].append({"role": "assistant", "content": result["final_answer"]})
    return result


if __name__ == "__main__":
    # smoke test — simulates the "check engine light" multi-turn scenario
    print("🧠 Turn 1: 'My check engine light is on and the car is misfiring'")
    state = run_agent_turn("My check engine light is on and the car is misfiring")
    print(f"Agent: {state['final_answer']}\n")

    if state.get("needs_clarification"):
        print("🧠 Turn 2 (user answers clarifying question): 'It happens mostly at idle, rough idle'")
        state = run_agent_turn("It happens mostly at idle, rough idle", prior_state=state)
        print(f"Agent: {state['final_answer']}")