"""
AutoMind — knowledge graph builder.
Reads data/kg_seed/seed_data.json and populates Neo4j with the schema
designed earlier:

    (Component)-[:PART_OF]->(System)
    (Component)-[:CAN_EXHIBIT]->(Symptom)
    (Symptom)-[:INDICATES]->(WarningLight)
    (Symptom)-[:CAUSED_BY {likelihood}]->(Cause)
    (Cause)-[:RESOLVED_BY]->(Fix)

All writes use MERGE, so this script is safe to re-run without creating
duplicates — important since you'll likely tweak seed_data.json a few
times as you refine the schema.

Requires: pip install neo4j
Requires: a running Neo4j instance (local Docker or free AuraDB tier) —
see docker/docker-compose.yml (added in the Docker step) or set
NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD in your .env for AuraDB.
"""

import json
from neo4j import GraphDatabase

from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, KG_SEED_DIR


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def build_graph():
    seed_path = KG_SEED_DIR / "seed_data.json"
    with open(seed_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    driver = get_driver()
    with driver.session() as session:

        # ---- Nodes ----
        for wl in data["warning_lights"]:
            session.run(
                "MERGE (w:WarningLight {name: $name}) "
                "SET w.color = $color, w.meaning = $meaning",
                **wl
            )

        for sys in data["systems"]:
            session.run("MERGE (s:System {name: $name})", **sys)

        for comp in data["components"]:
            session.run(
                "MERGE (c:Component {name: $name}) "
                "WITH c "
                "MATCH (s:System {name: $system}) "
                "MERGE (c)-[:PART_OF]->(s)",
                **comp
            )

        for symptom in data["symptoms"]:
            session.run(
                "MERGE (s:Symptom {name: $name}) "
                "SET s.description = $description, s.severity = $severity",
                **symptom
            )

        for cause in data["causes"]:
            session.run(
                "MERGE (c:Cause {name: $name}) "
                "WITH c "
                "MATCH (s:Symptom {name: $for_symptom}) "
                "MERGE (s)-[r:CAUSED_BY]->(c) "
                "SET r.likelihood = $likelihood",
                **cause
            )

        for fix in data["fixes"]:
            session.run(
                "MERGE (f:Fix {name: $name}) "
                "SET f.diy_possible = $diy_possible, f.cost_tier = $cost_tier "
                "WITH f "
                "MATCH (c:Cause {name: $for_cause}) "
                "MERGE (c)-[:RESOLVED_BY]->(f)",
                **fix
            )

        # ---- Relationships needing both sides to already exist ----
        for rel in data["symptom_indicates_light"]:
            session.run(
                "MATCH (s:Symptom {name: $symptom}), (w:WarningLight {name: $light}) "
                "MERGE (s)-[:INDICATES]->(w)",
                **rel
            )

        for rel in data["component_exhibits_symptom"]:
            session.run(
                "MATCH (c:Component {name: $component}), (s:Symptom {name: $symptom}) "
                "MERGE (c)-[:CAN_EXHIBIT]->(s)",
                **rel
            )

    driver.close()
    print("✅ Knowledge graph built successfully.")
    print(f"   Connected to: {NEO4J_URI}")
    print("   Open your Neo4j Browser (AuraDB console, or http://localhost:7474 if local) and run:")
    print("   MATCH (n) RETURN n LIMIT 100")


if __name__ == "__main__":
    build_graph()