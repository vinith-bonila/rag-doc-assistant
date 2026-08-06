"""Document chunking.

Two strategies:

1. `recursive`  - LangChain's RecursiveCharacterTextSplitter. Robust default:
   splits on paragraph/sentence boundaries and packs to a target size.

2. `semantic`   - Our own embedding-based splitter. We embed each sentence,
   walk through consecutive sentences, and start a new chunk whenever the
   similarity between neighbouring sentences drops below a percentile
   threshold (i.e. the topic shifts). This keeps semantically-coherent text
   together, which improves retrieval precision. Being able to explain *why*
   semantic chunking helps is a good interview talking point.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

# NOTE: Embedder is imported lazily inside semantic_chunk so that callers who
# only use recursive chunking don't need the (heavy) embedding stack loaded.
# The TYPE_CHECKING import exists only for annotations / linters (no runtime cost).
if TYPE_CHECKING:
    from .embeddings import Embedder

@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int
    # 1-indexed page for PDFs; None for formats without a reliable page model
    # (DOCX, TXT, Markdown). Carried through to citations so the UI can show
    # "filename p.3".
    page: Optional[int] = None
    meta: dict = field(default_factory=dict)


def _split_sentences(text: str) -> List[str]:
    # Simple, dependency-free sentence splitter. Good enough for chunking.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def recursive_chunk(text: str, source: str, chunk_size: int, chunk_overlap: int,
                    start_id: int = 0, page: Optional[int] = None) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [
        Chunk(text=t, source=source, chunk_id=start_id + i, page=page)
        for i, t in enumerate(splitter.split_text(text))
    ]


def semantic_chunk(text: str, source: str, embedder: "Embedder",
                   max_chars: int = 1000, breakpoint_percentile: int = 25,
                   start_id: int = 0, page: Optional[int] = None) -> List[Chunk]:
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return ([Chunk(text=text.strip(), source=source, chunk_id=start_id, page=page)]
                if text.strip() else [])

    embs = embedder.encode(sentences)
    # Cosine similarity between each sentence and the next (embeddings are
    # already normalized, so a dot product gives cosine).
    sims = np.sum(embs[:-1] * embs[1:], axis=1)
    # A "breakpoint" is a spot where similarity is unusually low.
    threshold = np.percentile(sims, breakpoint_percentile)

    chunks: List[Chunk] = []
    current: List[str] = [sentences[0]]
    cid = start_id
    for i in range(1, len(sentences)):
        too_long = sum(len(s) for s in current) + len(sentences[i]) > max_chars
        topic_shift = sims[i - 1] < threshold
        if too_long or topic_shift:
            chunks.append(Chunk(text=" ".join(current), source=source, chunk_id=cid, page=page))
            cid += 1
            current = []
        current.append(sentences[i])
    if current:
        chunks.append(Chunk(text=" ".join(current), source=source, chunk_id=cid, page=page))
    return chunks
