"""
AutoMind — evaluation script.

Produces the actual numbers behind claims like "my agent correctly
identifies the fault symptom X% of the time" and "vehicle-scoped
retrieval returns the correct manual Y% of the time" — replacing
"I tested it manually a few times" with a defensible metric.

Run:
    python -m evaluation.eval_script

Requires: Ollama + Neo4j running (same as the agent normally needs).
Writes a full report to evaluation/eval_report.json.
"""

import json
import time
from pathlib import Path

from src.agent.state import AgentState
from src.agent.nodes import classify_intent
from src.rag.vector_store import search as faiss_search
from src.config import VEHICLE_MANUAL_MAP

EVAL_DIR = Path(__file__).resolve().parent


def eval_symptom_matching():
    """
    Runs every symptom_test_cases.json entry through the real classify_intent
    node (same code path the live agent uses) and checks whether it correctly
    matches the user's phrasing to the right known symptom.
    """
    test_cases = json.loads((EVAL_DIR / "symptom_test_cases.json").read_text(encoding="utf-8"))

    results = []
    correct = 0
    total_latency = 0.0

    print(f"🔍 Running symptom-matching eval ({len(test_cases)} cases) ...")
    for case in test_cases:
        state: AgentState = {"user_query": case["query"], "messages": []}

        start = time.time()
        output = classify_intent(state)
        elapsed = time.time() - start
        total_latency += elapsed

        predicted = output.get("identified_symptom")
        is_correct = predicted == case["expected_symptom"]
        correct += int(is_correct)

        results.append({
            "query": case["query"],
            "expected": case["expected_symptom"],
            "predicted": predicted,
            "correct": is_correct,
            "latency_sec": round(elapsed, 2),
        })
        status = "✅" if is_correct else "❌"
        print(f"  {status} \"{case['query'][:60]}\" -> predicted: {predicted}")

    accuracy = correct / len(test_cases) if test_cases else 0
    avg_latency = total_latency / len(test_cases) if test_cases else 0

    return {
        "accuracy": round(accuracy, 3),
        "correct": correct,
        "total": len(test_cases),
        "avg_latency_sec": round(avg_latency, 2),
        "details": results,
    }


def eval_vehicle_retrieval():
    """
    Runs every retrieval_test_cases.json entry through FAISS search with the
    vehicle filter applied, and checks whether the top result actually comes
    from the expected manual (or whether the silent fallback kicked in).
    """
    test_cases = json.loads((EVAL_DIR / "retrieval_test_cases.json").read_text(encoding="utf-8"))

    results = []
    correct = 0

    print(f"\n🔍 Running vehicle-retrieval eval ({len(test_cases)} cases) ...")
    for case in test_cases:
        vehicle_manual = VEHICLE_MANUAL_MAP.get(case["vehicle"])
        top_results = faiss_search(case["query"], top_k=1, source_file_filter=vehicle_manual)

        top_source = top_results[0]["source_file"] if top_results else None
        top_score = top_results[0]["score"] if top_results else None
        is_correct = top_source == case["expected_source_file"]
        correct += int(is_correct)

        results.append({
            "query": case["query"],
            "vehicle": case["vehicle"],
            "expected_source": case["expected_source_file"],
            "actual_source": top_source,
            "score": round(top_score, 3) if top_score else None,
            "correct": is_correct,
        })
        status = "✅" if is_correct else "⚠️ fallback triggered"
        print(f"  {status} \"{case['query'][:50]}\" ({case['vehicle']}) -> {top_source}")

    precision = correct / len(test_cases) if test_cases else 0

    return {
        "vehicle_match_precision": round(precision, 3),
        "correct": correct,
        "total": len(test_cases),
        "details": results,
    }


def main():
    symptom_results = eval_symptom_matching()
    retrieval_results = eval_vehicle_retrieval()

    report = {
        "symptom_matching": symptom_results,
        "vehicle_retrieval": retrieval_results,
    }

    out_path = EVAL_DIR / "eval_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Symptom matching accuracy:      {symptom_results['accuracy']*100:.1f}% "
          f"({symptom_results['correct']}/{symptom_results['total']})")
    print(f"Avg symptom classification time: {symptom_results['avg_latency_sec']}s")
    print(f"Vehicle-filtered retrieval:      {retrieval_results['vehicle_match_precision']*100:.1f}% "
          f"({retrieval_results['correct']}/{retrieval_results['total']})")
    print(f"\nFull report saved to: {out_path}")


if __name__ == "__main__":
    main()