"""
AutoMind — FAISS vector store.
Builds an IndexFlatIP (cosine similarity via normalized dot product) over
all manual chunks, and persists both the index and the chunk metadata
(text, page number, source file, system tag) so we can map FAISS result
indices back to human-readable content.

Requires: pip install faiss-cpu   (or faiss-gpu if you have CUDA set up)
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import numpy as np

from src.config import PROCESSED_DIR, FAISS_INDEX_DIR
from src.rag.embeddings import embed_texts, embed_query

INDEX_PATH = FAISS_INDEX_DIR / "manual_index.faiss"
METADATA_PATH = FAISS_INDEX_DIR / "chunk_metadata.pkl"


def build_index():
    """
    Reads data/processed/all_chunks.json, embeds every chunk, builds a
    FAISS IndexFlatIP, and saves both the index and a metadata list
    (same order as vectors) to data/faiss_index/.
    """
    chunks_path = PROCESSED_DIR / "all_chunks.json"
    if not chunks_path.exists():
        print("⚠️  all_chunks.json not found. Run chunker.py first.")
        return

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks: List[Dict] = json.load(f)

    print(f"🔢 Embedding {len(chunks)} chunks ...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product == cosine sim since vectors are normalized
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)   # chunks[i] corresponds to vector i in the index

    print(f"✅ FAISS index built: {index.ntotal} vectors, dim={dim}")
    print(f"   -> index: {INDEX_PATH}")
    print(f"   -> metadata: {METADATA_PATH}")


def load_index() -> Tuple[faiss.Index, List[Dict]]:
    """Loads the FAISS index and chunk metadata from disk."""
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run `python -m src.rag.vector_store` first."
        )

    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def search(query: str, top_k: int = 5, system_filter: str = None,
           source_file_filter: str = None) -> List[Dict]:
    """
    Searches the FAISS index for the top_k most relevant chunks to `query`.

    source_file_filter: if given (e.g. "TOYOTA 2025(Corolla).pdf"), search
    is restricted to that manual first. If the best result scores below
    RETRIEVAL_FALLBACK_THRESHOLD (i.e. that manual doesn't actually cover
    this topic), we silently fall back to searching ALL manuals instead —
    better to answer from the wrong manual with a clear citation than to
    return a weak, barely-related match just because it's the "right" file.
    """
    from src.config import RETRIEVAL_FALLBACK_THRESHOLD

    if source_file_filter:
        results = _search_impl(query, top_k, system_filter, source_file_filter)
        if results and results[0]["score"] >= RETRIEVAL_FALLBACK_THRESHOLD:
            return results
        # silent fallback — no filter, search everything
        return _search_impl(query, top_k, system_filter, None)

    return _search_impl(query, top_k, system_filter, None)


def _search_impl(query: str, top_k: int, system_filter: str = None,
                  source_file_filter: str = None) -> List[Dict]:
    index, chunks = load_index()
    query_vec = embed_query(query)

    # over-fetch when filtering, since some results will be discarded
    active_filters = sum(f is not None for f in [system_filter, source_file_filter])
    fetch_k = top_k * 5 if active_filters else top_k
    scores, indices = index.search(query_vec, fetch_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        if system_filter and chunk["system_tag"] != system_filter:
            continue
        if source_file_filter and chunk["source_file"] != source_file_filter:
            continue
        results.append({**chunk, "score": float(score)})
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    build_index()

    # quick smoke test
    print("\n🔍 Test query: 'check engine light causes'")
    for r in search("check engine light causes", top_k=3):
        print(f"  [{r['score']:.3f}] {r['source_file']} p.{r['page_number']} "
              f"({r['system_tag']}): {r['text'][:100]}...")