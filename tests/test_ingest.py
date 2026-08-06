"""Ingestion/parsing: PDF page extraction, DOCX parsing, and chunk id/page
propagation. These need no models (recursive chunking is the default)."""
from __future__ import annotations

import os

from src.ingest import chunk_document, clean_pdf_text, parse_file

DOCS = os.path.join("data", "sample_docs")


def test_clean_pdf_text_repairs_extraction_artifacts():
    # Simulate the broken glyphs real PDFs emit (built from codepoints so the
    # test source stays ASCII): a "ti" barred-O, a "fi" ligature, a private-use
    # bullet, and a replacement char.
    raw = ("opera" + chr(0x019F) + "ons, " + chr(0xFB01) + "eld work "
           + chr(0xF0B7) + " item " + chr(0xFFFD))
    out = clean_pdf_text(raw)
    assert "operations" in out and "field" in out
    # none of the artifact codepoints survive
    for cp in (0x019F, 0x0275, 0xFB01, 0xF0B7, 0xFFFD):
        assert chr(cp) not in out


def test_parse_pdf_preserves_page_numbers():
    segments = parse_file(os.path.join(DOCS, "vector_databases.pdf"))
    pages = [page for page, _ in segments]
    assert pages == [1, 2]                      # a real 2-page PDF
    assert all(text.strip() for _, text in segments)


def test_parse_docx_is_single_pageless_segment():
    segments = parse_file(os.path.join(DOCS, "model_monitoring.docx"))
    assert len(segments) == 1
    page, text = segments[0]
    assert page is None                          # DOCX has no reliable page model
    assert "drift" in text.lower()


def test_chunk_document_ids_are_unique_and_pages_propagate():
    segments = [(1, "Alpha sentence. " * 60), (2, "Beta sentence. " * 60)]
    chunks = chunk_document("doc.pdf", segments, embedder=None, start_id=0)
    ids = [c.chunk_id for c in chunks]
    assert ids == list(range(len(chunks)))       # contiguous & unique
    assert {c.page for c in chunks} == {1, 2}    # each chunk keeps its page
