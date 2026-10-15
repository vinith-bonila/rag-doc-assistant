"""Chunk metadata schema + a retrieval filter (rag.pdf pp.8, 10).

Metadata is stored on ``Chunk.meta`` (a plain dict that already persists to
chunks.json), so no schema migration is needed. ``MetaFilter`` is an extensible
predicate: it filters by owner, source filename, document type and page range.

``owner`` scoping note
----------------------
``owner`` is an *isolation* field, not an authenticated identity. It separates
the bundled public corpus (``OWNER_PUBLIC``) from documents a particular browser
session uploaded, so one visitor's upload is not retrieved for another visitor
on the same shared process. It is **not** authentication or authorization: the
owner value is assigned by the server to a session, but there is no verified
login, so it must not be relied on to protect confidential material. See
docs/SECURITY_AUDIT.md (DM-01).
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

# Owner value for the bundled sample corpus: readable by every session.
OWNER_PUBLIC = "public"
# Sentinel for a chunk carrying no owner at all. It matches no owner filter, so
# an unstamped chunk is excluded rather than silently treated as public
# (fail-closed; docs/SECURITY_AUDIT.md DM-01). The persisted public corpus is
# backfilled explicitly at load time, so this only ever applies to a chunk that
# bypassed ``stamp_metadata``.
OWNER_UNKNOWN = ""


def doc_type(filename: str) -> str:
    """Lowercase extension without the dot, e.g. 'pdf'. '' if none."""
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    return ext


def content_hash(data: bytes) -> str:
    """Stable short hash of raw file bytes — used to detect unchanged re-uploads."""
    return hashlib.sha256(data).hexdigest()[:16]


def chunk_owner(chunk) -> str:
    """Owner of a chunk, or ``OWNER_UNKNOWN`` when the field is absent.

    Fail-closed: an unstamped chunk matches no owner filter, so a metadata gap
    narrows access instead of widening it. Indexes written before the field
    existed are backfilled to ``OWNER_PUBLIC`` when the pipeline loads them
    (``RAGPipeline._backfill_owners``), which is where backwards compatibility
    is handled.
    """
    return (chunk.meta or {}).get("owner", OWNER_UNKNOWN)


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
    owners: Optional[Set[str]] = None      # isolation scope, NOT authorization

    def is_empty(self) -> bool:
        return (self.sources is None and self.doc_types is None
                and self.min_page is None and self.max_page is None
                and self.owners is None)

    def allows(self, chunk) -> bool:
        if self.owners is not None and chunk_owner(chunk) not in self.owners:
            return False
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

    def narrowed_to(self, sources: Optional[Set[str]]) -> "MetaFilter":
        """A copy additionally restricted to ``sources`` (intersecting any
        existing source allow-list, so narrowing can never *widen* scope)."""
        if not sources:
            return self
        new = set(sources)
        if self.sources is not None:
            new &= self.sources
        return MetaFilter(sources=new, doc_types=self.doc_types,
                          min_page=self.min_page, max_page=self.max_page,
                          owners=self.owners)

    @classmethod
    def from_sources(cls, allowed_sources: Optional[Set[str]]) -> "Optional[MetaFilter]":
        """Backward-compatible bridge from the old ``allowed_sources`` scoping."""
        if not allowed_sources:
            return None
        return cls(sources=set(allowed_sources))


def stamp_metadata(chunks, filename: str, data: bytes, embed_model: str,
                   owner: str = OWNER_PUBLIC) -> None:
    """Attach the rich metadata schema to every chunk of a document (in place)."""
    ch = content_hash(data)
    dt = doc_type(filename)
    title = os.path.splitext(os.path.basename(filename))[0]
    for c in chunks:
        c.meta.update({
            "doc_type": dt,
            "content_hash": ch,
            "title": title,
            "owner": owner,
            "parser_version": PARSER_VERSION,
            "chunker_version": CHUNKER_VERSION,
            "embedding_model": embed_model,
            "ingestion_status": "indexed",
        })
