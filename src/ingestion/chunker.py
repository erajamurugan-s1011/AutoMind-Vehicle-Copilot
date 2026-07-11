"""
AutoMind — chunking.
Turns raw page-level text into overlapping, token-sized chunks ready for
embedding. Each chunk also gets a heuristic "system_tag" (engine, brakes,
electrical, cooling, general) based on keyword matching — this becomes a
metadata filter so the agent can narrow FAISS search to the relevant
system before even hitting the LLM, which is both faster and a good
architecture talking point in interviews.

Requires: pip install tiktoken
"""

import json
from pathlib import Path
from typing import List, Dict

import tiktoken

from src.config import PROCESSED_DIR, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS

# cl100k_base is a good general-purpose tokenizer approximation for chunk sizing
_encoding = tiktoken.get_encoding("cl100k_base")

SYSTEM_KEYWORDS = {
    "engine": ["engine", "spark plug", "misfire", "oil", "combustion", "cylinder", "timing belt"],
    "brakes": ["brake", "abs", "brake pad", "brake fluid", "rotor"],
    "electrical": ["battery", "fuse", "alternator", "wiring", "electrical", "sensor"],
    "cooling": ["coolant", "radiator", "overheat", "thermostat", "cooling"],
    "transmission": ["transmission", "gear", "clutch", "cvt"],
    "tires_suspension": ["tire", "tyre", "suspension", "wheel", "alignment"],
}


def tag_system(text: str) -> str:
    """Assigns the most likely vehicle system to a chunk based on keyword hits."""
    text_lower = text.lower()
    scores = {system: 0 for system in SYSTEM_KEYWORDS}

    for system, keywords in SYSTEM_KEYWORDS.items():
        for kw in keywords:
            scores[system] += text_lower.count(kw)

    best_system = max(scores, key=scores.get)
    return best_system if scores[best_system] > 0 else "general"


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Splits a block of text into overlapping chunks by token count."""
    tokens = _encoding.encode(text)
    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(_encoding.decode(chunk_tokens))
        start += chunk_size - overlap  # slide window with overlap

    return chunks


def chunk_manual(pages: List[Dict], source_file: str) -> List[Dict]:
    """
    Converts a list of {page_number, text} dicts into a list of chunk dicts:
        {
          "chunk_id": str,
          "text": str,
          "page_number": int,
          "source_file": str,
          "system_tag": str
        }
    """
    all_chunks = []
    chunk_counter = 0

    for page in pages:
        page_chunks = chunk_text(page["text"], CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)

        for chunk in page_chunks:
            chunk_counter += 1
            all_chunks.append({
                "chunk_id": f"{Path(source_file).stem}_c{chunk_counter}",
                "text": chunk,
                "page_number": page["page_number"],
                "source_file": source_file,
                "system_tag": tag_system(chunk),
            })

    return all_chunks


def chunk_all_processed_manuals() -> List[Dict]:
    """
    Reads every *_raw.json in data/processed/, chunks it, and writes a
    combined data/processed/all_chunks.json used by the embedding step.
    """
    raw_files = list(PROCESSED_DIR.glob("*_raw.json"))
    if not raw_files:
        print("⚠️  No raw parsed manuals found. Run pdf_parser.py first.")
        return []

    combined_chunks = []
    for raw_file in raw_files:
        with open(raw_file, "r", encoding="utf-8") as f:
            pages = json.load(f)

        source_file = pages[0]["source_file"] if pages else raw_file.stem
        chunks = chunk_manual(pages, source_file)
        combined_chunks.extend(chunks)
        print(f"✂️  {source_file}: {len(pages)} pages -> {len(chunks)} chunks")

    out_path = PROCESSED_DIR / "all_chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined_chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Total: {len(combined_chunks)} chunks saved to {out_path}")
    return combined_chunks


if __name__ == "__main__":
    chunk_all_processed_manuals()