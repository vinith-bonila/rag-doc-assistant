"""Live answer validation (rag.pdf p.14).

Independent of the offline evaluation harness: this runs *per answer*, after
generation, as a cheap deterministic gate. It checks that every inline ``[n]``
citation refers to a real retrieved source — catching fabricated source numbers
before the user trusts them. Semantic groundedness (NLI / LLM-judge) is handled
by the evaluation harness; here we do the fast, dependency-free structural check
the guide recommends running first.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

_CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class CitationReport:
    n_citations: int = 0
    valid_sources: List[int] = field(default_factory=list)
    invalid_markers: List[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.invalid_markers

    @property
    def status(self) -> str:
        if self.n_citations == 0:
            return "no-citations"
        return "valid" if self.ok else "invalid-citations"


def validate_citations(answer: str, n_sources: int) -> CitationReport:
    """Verify each ``[n]`` marker in ``answer`` maps to a retrieved source
    (1 <= n <= n_sources)."""
    markers = [int(m) for m in _CITATION.findall(answer)]
    valid = sorted({m for m in markers if 1 <= m <= n_sources})
    invalid = sorted({m for m in markers if m < 1 or m > n_sources})
    return CitationReport(n_citations=len(markers), valid_sources=valid,
                          invalid_markers=invalid)


def flag_invalid_citations(answer: str, n_sources: int) -> str:
    """Replace out-of-range ``[n]`` markers with ``[?]`` so the UI never shows a
    citation that points nowhere. Valid markers are left untouched."""
    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        return m.group(0) if 1 <= n <= n_sources else "[?]"
    return _CITATION.sub(repl, answer)
