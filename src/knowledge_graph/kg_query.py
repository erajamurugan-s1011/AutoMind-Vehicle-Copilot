"""
AutoMind — knowledge graph queries.
These are the exact functions the LangGraph agent will call when it decides
"this needs graph reasoning, not manual text search." Each function maps
directly to a step in the multi-turn diagnosis flow:

    symptom mentioned -> get_causes_for_symptom() [ranked by likelihood]
    -> if ambiguous -> agent asks clarifying question
    -> cause narrowed -> get_fix_for_cause()

Requires: pip install neo4j
"""

from typing import List, Dict, Optional
from neo4j import GraphDatabase

from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


def get_causes_for_symptom(symptom_name: str) -> List[Dict]:
    """
    Returns all known causes for a symptom, ranked by likelihood
    (high -> medium -> low). This is the core "narrow down" query.
    """
    likelihood_rank = {"high": 0, "medium": 1, "low": 2}

    query = """
        MATCH (s:Symptom {name: $symptom_name})-[r:CAUSED_BY]->(c:Cause)
        RETURN c.name AS cause, r.likelihood AS likelihood
    """
    with get_driver().session() as session:
        results = session.run(query, symptom_name=symptom_name)
        causes = [{"cause": r["cause"], "likelihood": r["likelihood"]} for r in results]

    causes.sort(key=lambda x: likelihood_rank.get(x["likelihood"], 3))
    return causes


def get_fix_for_cause(cause_name: str) -> Optional[Dict]:
    """Returns the recommended fix for a given cause, including DIY info."""
    query = """
        MATCH (c:Cause {name: $cause_name})-[:RESOLVED_BY]->(f:Fix)
        RETURN f.name AS fix, f.diy_possible AS diy_possible, f.cost_tier AS cost_tier
        LIMIT 1
    """
    with get_driver().session() as session:
        result = session.run(query, cause_name=cause_name).single()
        if result is None:
            return None
        return {
            "fix": result["fix"],
            "diy_possible": result["diy_possible"],
            "cost_tier": result["cost_tier"],
        }


def get_warning_light_for_symptom(symptom_name: str) -> Optional[str]:
    """Returns the warning light associated with a symptom, if any."""
    query = """
        MATCH (s:Symptom {name: $symptom_name})-[:INDICATES]->(w:WarningLight)
        RETURN w.name AS light
        LIMIT 1
    """
    with get_driver().session() as session:
        result = session.run(query, symptom_name=symptom_name).single()
        return result["light"] if result else None


def get_symptoms_for_component(component_name: str) -> List[str]:
    """Returns all symptoms a given component can exhibit."""
    query = """
        MATCH (c:Component {name: $component_name})-[:CAN_EXHIBIT]->(s:Symptom)
        RETURN s.name AS symptom
    """
    with get_driver().session() as session:
        results = session.run(query, component_name=component_name)
        return [r["symptom"] for r in results]


def full_diagnosis_path(symptom_name: str) -> List[Dict]:
    """
    Returns the full reasoning path for a symptom: every cause, ranked by
    likelihood, with its associated fix already attached. This is what the
    agent hands to the LLM to generate the final natural-language answer.
    """
    causes = get_causes_for_symptom(symptom_name)
    for c in causes:
        fix_info = get_fix_for_cause(c["cause"])
        c["fix"] = fix_info["fix"] if fix_info else None
        c["diy_possible"] = fix_info["diy_possible"] if fix_info else None
        c["cost_tier"] = fix_info["cost_tier"] if fix_info else None
    return causes


def get_all_symptom_names() -> List[str]:
    """
    Returns every symptom name known to the graph. The agent uses this list
    to ask the LLM "which of these known symptoms best matches what the
    user described?" rather than trying to hardcode every possible phrasing.
    """
    query = "MATCH (s:Symptom) RETURN s.name AS name"
    with get_driver().session() as session:
        results = session.run(query)
        return [r["name"] for r in results]


if __name__ == "__main__":
    # quick smoke test
    print("🔍 Diagnosis path for 'Engine misfire':")
    for step in full_diagnosis_path("Engine misfire"):
        print(f"  - {step['cause']} ({step['likelihood']} likelihood) "
              f"-> fix: {step['fix']} (DIY: {step['diy_possible']})")