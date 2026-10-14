"""Chunk metadata schema + a retrieval filter (rag.pdf pp.8, 10).

Metadata is stored on ``Chunk.meta`` (a plain dict that already persists to
chunks.json), so no schema migration is needed. ``MetaFilter`` is an extensible,
authorization-ready predicate: today it filters by source filename, document type
and page range; an ``access``/``owner`` field can be added later without changing
the retrieval interface.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Optional, Set

# Versions stamped onto every chunk so an index can be reasoned about and
# re-built deterministically (rag.pdf p.8: parser/chunker/embedding versions).
PARSER_VERSION = "pypdf+pdfplumber-1"
CHUNKER_VERSION = "1"


def doc_type(filename: str) -> str:
    """Lowercase extension without the dot, e.g. 'pdf'. '' if none."""
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    return ext


def content_hash(data: bytes) -> str:
    """Stable short hash of raw file bytes — used to detect unchanged re-uploads."""
    return hashlib.sha256(data).hexdigest()[:16]


@dataclass
class MetaFilter:
    """Restrict retrieval to chunks matching these attributes.

    All fields are optional; ``None`` means "don't filter on this". A chunk must
    satisfy every set constraint (logical AND).
    """
    sources: Optional[Set[str]] = None     # filename allow-list
    doc_types: Optional[Set[str]] = None   # e.g. {"pdf", "md"}
    min_page: Optional[int] = None
    max_page: Optional[int] = None

    def is_empty(self) -> bool:
        return (self.sources is None and self.doc_types is None
                and self.min_page is None and self.max_page is None)

    def allows(self, chunk) -> bool:
        if self.sources is not None and chunk.source not in self.sources:
            return False
        if self.doc_types is not None:
            if (chunk.meta or {}).get("doc_type", doc_type(chunk.source)) not in self.doc_types:
                return False
        if self.min_page is not None and (chunk.page is None or chunk.page < self.min_page):
            return False
        if self.max_page is not None and (chunk.page is None or chunk.page > self.max_page):
            return False
        return True

    @classmethod
    def from_sources(cls, allowed_sources: Optional[Set[str]]) -> "Optional[MetaFilter]":
        """Backward-compatible bridge from the old ``allowed_sources`` scoping."""
        if not allowed_sources:
            return None
        return cls(sources=set(allowed_sources))


def stamp_metadata(chunks, filename: str, data: bytes, embed_model: str) -> None:
    """Attach the rich metadata schema to every chunk of a document (in place)."""
    ch = content_hash(data)
    dt = doc_type(filename)
    title = os.path.splitext(os.path.basename(filename))[0]
    for c in chunks:
        c.meta.update({
            "doc_type": dt,
            "content_hash": ch,
            "title": title,
            "parser_version": PARSER_VERSION,
            "chunker_version": CHUNKER_VERSION,
            "embedding_model": embed_model,
            "ingestion_status": "indexed",
        })
