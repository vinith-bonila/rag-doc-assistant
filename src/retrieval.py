"""Retrieval layer: dense, sparse, and a hybrid fusion of the two.

Every retriever implements the same tiny interface — ``retrieve(query, k) ->
[(Chunk, score)]`` — so they compose. The pipeline stacks a hybrid retriever
(dense FAISS + sparse BM25) and then a cross-encoder reranker on top.

Why hybrid? Dense (embedding) search captures meaning and paraphrase but can
miss exact keywords, rare identifiers, or acronyms; BM25 nails exact lexical
matches but is blind to synonyms. Fusing them recovers both.

Why Reciprocal Rank Fusion (RRF) instead of adding the raw scores? FAISS cosine
scores and BM25 scores live on completely different, unnormalized scales, so
summing them lets one dominate arbitrarily. RRF ignores magnitudes and fuses on
*rank position* — robust and parameter-light:

    fused(d) = sum_r  1 / (rrf_k + rank_r(d))
"""
from __future__ import annotations

import re
from typing import Dict, List, Protocol, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk
from .embeddings import Embedder
from .vectorstore import FaissStore

Scored = Tuple[Chunk, float]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class Retriever(Protocol):
    """Anything that turns a query into ranked (chunk, score) candidates."""

    def retrieve(self, query: str, k: int) -> List[Scored]:
        ...


class DenseRetriever:
    """Bi-encoder + FAISS cosine search."""

    def __init__(self, store: FaissStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, k: int) -> List[Scored]:
        q = self.embedder.encode([query])
        return self.store.search(q, k)


class BM25Retriever:
    """Sparse lexical retrieval over the same chunk set."""

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks]) if chunks else None

    def retrieve(self, query: str, k: int) -> List[Scored]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        k = min(k, len(self.chunks))
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


class HybridRetriever:
    """Fuse several retrievers by Reciprocal Rank Fusion.

    The fused score is order-preserving but not a similarity; the downstream
    cross-encoder reranker assigns the scores that are actually shown to users.
    """

    def __init__(self, retrievers: List[Retriever], rrf_k: int = 60):
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self.retrievers = retrievers
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int) -> List[Scored]:
        fused: Dict[int, float] = {}
        by_id: Dict[int, Chunk] = {}
        for retriever in self.retrievers:
            # Pull a few extra candidates per retriever so fusion has overlap
            # to work with before we trim back to k.
            ranked = retriever.retrieve(query, k)
            for rank, (chunk, _score) in enumerate(ranked, start=1):
                by_id[chunk.chunk_id] = chunk
                fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [(by_id[cid], score) for cid, score in ordered]


def build_retriever(store: FaissStore, embedder: Embedder, *, hybrid: bool,
                    rrf_k: int) -> Retriever:
    """Compose the configured retrieval stack (before reranking)."""
    dense = DenseRetriever(store, embedder)
    if not hybrid:
        return dense
    sparse = BM25Retriever(store.chunks)
    return HybridRetriever([dense, sparse], rrf_k=rrf_k)
