"""FAISS-backed vector store.

We keep the FAISS index (vectors) and a parallel list of chunk metadata
(text + source) together so a search returns not just IDs but the actual
passages we need for reranking, answering and citations.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import List, Tuple

import faiss
import numpy as np

from .chunking import Chunk


class FaissStore:
    def __init__(self, dim: int):
        # Inner-product index; combined with normalized vectors == cosine sim.
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []

    @property
    def dim(self) -> int:
        return self.index.d

    def add(self, vectors: np.ndarray, chunks: List[Chunk]) -> None:
        assert vectors.shape[0] == len(chunks), "vectors/chunks length mismatch"
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def reconstruct_all(self) -> np.ndarray:
        """Return the stored vectors as a fresh (n, dim) array. Lets callers
        rebuild the index without re-embedding (IndexFlatIP is exact-storage)."""
        n = self.index.ntotal
        if n == 0:
            return np.zeros((0, self.index.d), dtype="float32")
        return self.index.reconstruct_n(0, n)

    def rebuild(self, vectors: np.ndarray, chunks: List[Chunk]) -> None:
        """Replace the entire contents with the given vectors + chunks.

        A brand-new IndexFlatIP is created and the old one is dropped, so the
        memory backing any removed/replaced vectors is released (not just the
        Python wrapper — FAISS frees the C++ buffer when the index is GC'd)."""
        self.index = faiss.IndexFlatIP(self.index.d)
        self.chunks = []
        if len(chunks):
            self.add(vectors, chunks)

    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[Chunk, float]]:
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        k = min(k, len(self.chunks))
        scores, idxs = self.index.search(query_vec, k)
        out: List[Tuple[Chunk, float]] = []
        for i, score in zip(idxs[0], scores[0]):
            if i == -1:
                continue
            out.append((self.chunks[i], float(score)))
        return out

    # --- persistence ------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "FaissStore":
        index = faiss.read_index(os.path.join(path, "index.faiss"))
        store = cls.__new__(cls)          # bypass __init__ (we already have an index)
        store.index = index
        with open(os.path.join(path, "chunks.json"), "r", encoding="utf-8") as f:
            store.chunks = [Chunk(**c) for c in json.load(f)]
        return store
