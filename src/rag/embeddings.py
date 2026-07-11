"""
AutoMind — embeddings.
Thin wrapper around Sentence-Transformers so the rest of the codebase
never touches the model directly. Batches encoding for speed on your GPU.

Requires: pip install sentence-transformers
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME

_model = None  # lazy-loaded singleton so we don't reload the model on every import


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"🔧 Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Encodes a list of strings into a (n, dim) float32 numpy array.
    Normalizes embeddings so we can use cosine similarity via inner product
    in FAISS (IndexFlatIP), which is faster than L2 for normalized vectors.
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # critical: enables cosine-sim via dot product
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """Encodes a single query string. Returns shape (1, dim)."""
    return embed_texts([query])