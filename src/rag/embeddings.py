"""
AutoMind — embeddings.
Uses fastembed (ONNX runtime) instead of sentence-transformers/torch. Same
underlying MiniLM model and same 384-dim output, but without pulling in the
full PyTorch library — critical for fitting inside constrained free-tier
hosting (e.g. Render's 512MB RAM limit), where torch + transformers alone
can exceed the memory budget before the app even finishes starting up.

Requires: pip install fastembed
"""

from typing import List
import numpy as np
from fastembed import TextEmbedding

from src.config import EMBEDDING_MODEL_NAME

_model = None  # lazy-loaded singleton so we don't reload the model on every import


def get_embedding_model() -> TextEmbedding:
    global _model
    if _model is None:
        print(f"🔧 Loading embedding model (fastembed/ONNX): {EMBEDDING_MODEL_NAME}")
        _model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Encodes a list of strings into a (n, dim) float32 numpy array.
    fastembed's all-MiniLM-L6-v2 output is already L2-normalized, so cosine
    similarity via inner product (FAISS IndexFlatIP) works the same as before.
    """
    model = get_embedding_model()
    embeddings = list(model.embed(texts, batch_size=batch_size))
    return np.array(embeddings).astype("float32")


def embed_query(query: str) -> np.ndarray:
    """Encodes a single query string. Returns shape (1, dim)."""
    return embed_texts([query])