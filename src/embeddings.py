"""Thin wrapper around a sentence-transformers bi-encoder.

We normalize embeddings so that inner-product search in FAISS is equivalent to
cosine similarity — that lets us use the fast IndexFlatIP index.
"""
from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Return L2-normalized float32 embeddings, shape (n, dim)."""
        if isinstance(texts, str):
            texts = [texts]
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,      # -> cosine == dot product
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype("float32")
