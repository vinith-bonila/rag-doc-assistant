"""Cross-encoder reranking.

The bi-encoder (embeddings) is fast but scores query and passage independently.
A cross-encoder reads the (query, passage) pair *together*, so it judges
relevance far more accurately — at the cost of speed. The standard pattern is:
retrieve many candidates cheaply with the bi-encoder, then rerank a small
shortlist with the cross-encoder. That two-stage design is exactly what the JD
means by "retrieval + reranking".
"""
from __future__ import annotations

from typing import List, Tuple

from sentence_transformers import CrossEncoder

from .chunking import Chunk


class Reranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Tuple[Chunk, float]],
               top_n: int) -> List[Tuple[Chunk, float]]:
        if not candidates:
            return []
        pairs = [[query, c.text] for c, _ in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip((c for c, _ in candidates), scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(chunk, float(score)) for chunk, score in ranked[:top_n]]
