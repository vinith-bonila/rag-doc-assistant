"""Context construction (rag.pdf p.12).

Reranked chunks are not dumped straight into the prompt. This stage:
  1. drops near-duplicate / contained passages (overlapping windows, re-uploaded
     versions) so evidence isn't repeated;
  2. trims to an approximate token budget so the context stays focused and the
     request stays within the model window.

Deterministic and dependency-free (token-set Jaccard + a char/4 token estimate),
so it is fully unit-testable without any model or API.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from .chunking import Chunk

Scored = Tuple[Chunk, float]
_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_WORD.findall(text.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token), floored at the word count."""
    return max(len(text) // 4, len(text.split()))


def build_context(reranked: List[Scored], token_budget: int,
                  dedup_jaccard: float) -> Tuple[List[Scored], dict]:
    """Return the deduplicated, budget-trimmed context (rank order preserved)
    plus a small report for tracing/observability."""
    kept: List[Scored] = []
    kept_tokens: List[set] = []
    dropped_dup = 0
    dropped_budget = 0
    used = 0

    for chunk, score in reranked:
        toks = _tokens(chunk.text)
        # near-duplicate of something already kept?
        if any(_jaccard(toks, kt) >= dedup_jaccard or toks and toks <= kt
               for kt in kept_tokens):
            dropped_dup += 1
            continue
        est = estimate_tokens(chunk.text)
        if kept and used + est > token_budget:   # keep at least one chunk
            dropped_budget += 1
            continue
        kept.append((chunk, score))
        kept_tokens.append(toks)
        used += est

    report = {
        "input": len(reranked),
        "kept": len(kept),
        "dropped_duplicates": dropped_dup,
        "dropped_over_budget": dropped_budget,
        "context_tokens_est": used,
    }
    return kept, report
