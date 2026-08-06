"""Citation correctness: the sources returned with an answer must be real,
carry their metadata, and (for PDFs) preserve the page number so the UI can
render a clickable, verifiable citation."""
from __future__ import annotations

import re


def test_sources_are_wellformed_and_ranked(pipeline):
    result = pipeline.answer("Why is a cross-encoder reranker more accurate?",
                             top_n=5)
    assert result.sources, "expected at least one source"
    # scores are descending after reranking
    scores = [s.score for s in result.sources]
    assert scores == sorted(scores, reverse=True)
    for s in result.sources:
        assert s.text.strip()            # non-empty passage
        assert s.source.endswith((".md", ".pdf", ".docx", ".txt"))
        assert s.chunk_id is not None    # traceable back to the index


def test_pdf_citation_carries_page_number(pipeline):
    result = pipeline.answer("How does HNSW approximate nearest neighbour search?",
                             top_n=5)
    pdf_sources = [s for s in result.sources if s.source.endswith(".pdf")]
    assert pdf_sources, "the PDF should be retrieved for this query"
    assert any(s.page is not None for s in pdf_sources)


def test_inline_citation_markers_map_to_real_sources(pipeline):
    # Any [n] marker the model emits must reference a source that exists.
    result = pipeline.answer("What does recall@k measure?", top_n=5)
    markers = {int(m) for m in re.findall(r"\[(\d+)\]", result.answer)}
    if markers:  # only assert when an LLM produced grounded citations
        assert max(markers) <= len(result.sources)
        assert min(markers) >= 1
