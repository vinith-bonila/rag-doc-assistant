"""Per-request observability trace (rag.pdf p.16).

A single ``RAGTrace`` records one query end-to-end: a span (duration) per pipeline
stage plus counts/scores worth debugging (candidates retrieved, reranked, context
tokens, citation status). It is in-process and dependency-free — it degrades to a
no-op if not passed — and never stores secrets or raw document text.
"""
from __future__ import annotations

import contextlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RAGTrace:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    stages: Dict[str, float] = field(default_factory=dict)   # name -> ms
    meta: Dict[str, object] = field(default_factory=dict)    # counts/scores

    @contextlib.contextmanager
    def span(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = round((time.perf_counter() - t0) * 1000, 1)

    def set(self, key: str, value):
        self.meta[key] = value
        return value

    @property
    def total_ms(self) -> float:
        return round(sum(self.stages.values()), 1)

    def as_dict(self) -> dict:
        return {"trace_id": self.trace_id, "total_ms": self.total_ms,
                "stages_ms": self.stages, **self.meta}

    def log(self, logger: logging.Logger) -> None:
        try:
            logger.info("rag_trace %s", json.dumps(self.as_dict(), default=str))
        except Exception:  # noqa: BLE001 — logging must never break a request
            pass
