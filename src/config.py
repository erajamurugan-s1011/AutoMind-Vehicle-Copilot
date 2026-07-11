"""
AutoMind — central configuration.
Keep every path, model name, and tunable constant here so the rest of the
codebase never hardcodes strings. Makes the project look production-grade
in a code review / interview walkthrough.
"""

import os
from pathlib import Path

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent          # automind/
DATA_DIR = BASE_DIR / "data"
MANUALS_DIR = DATA_DIR / "manuals"                          # raw PDFs go here
PROCESSED_DIR = DATA_DIR / "processed"                      # chunked JSON output
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
KG_SEED_DIR = DATA_DIR / "kg_seed"
LOGS_DIR = BASE_DIR / "logs"

for d in [MANUALS_DIR, PROCESSED_DIR, FAISS_INDEX_DIR, KG_SEED_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------- Embedding model ----------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # fast, 384-dim, good for CPU/GPU

# ---------- Chunking ----------
CHUNK_SIZE_TOKENS = 300        # ~300 tokens per chunk (owner manuals have short procedural sections)
CHUNK_OVERLAP_TOKENS = 50      # overlap so we don't cut a procedure mid-step

# ---------- LLM (Ollama local, or Groq cloud — switch via LLM_PROVIDER) ----------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")   # "ollama" | "groq"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = "qwen2.5:7b"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# ---------- Neo4j ----------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "automind123")   # matches docker-compose.neo4j.yml; override via .env for AuraDB

# ---------- Vehicle -> manual mapping ----------
# Maps a user-facing vehicle name to the exact source_file name stored in
# chunk metadata (must match the PDF filename exactly, including spaces).
# Used to filter FAISS retrieval to the correct owner's manual.
VEHICLE_MANUAL_MAP = {
    "Honda Civic (2025)": "2025 Civic sedan.pdf",
    "Toyota Corolla (2025)": "TOYOTA 2025(Corolla).pdf",
    "Tata Harrier BS6 (2026)": "harrier-bs6-owners-manual-april-2026.pdf",
    "Maruti Suzuki Jimny": "NEXA-Jimny-Petrol-Manual-latestpdf.pdf",
    "Hyundai Tucson (2025)": "Hyundai-Tucson.pdf",
    "Kia Sportage (2025)": "Kia-Sportage.pdf",
    "MG Hector (2025)": "MG-Hector-Apr25.pdf",
    "Nissan Rogue (2025)": "nissan-rogue.pdf",
    "Renault Kiger (2025)": "renault-Kiger.pdf",
    "Not sure / Other": None,   # None = search across all manuals, no filter
}

# Minimum similarity score to trust a vehicle-filtered result. Below this,
# we silently fall back to searching all manuals instead of returning
# weak/irrelevant matches just because they came from the "right" file.
RETRIEVAL_FALLBACK_THRESHOLD = 0.30

# ---------- FastAPI / Auth ----------
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60