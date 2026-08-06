"""Chunking correctness: sizing, id continuity, and page-metadata propagation."""
from __future__ import annotations

from src.chunking import recursive_chunk
from src.ingest import parse_bytes


def test_recursive_chunk_ids_and_page_propagate():
    text = "Sentence one. " * 200  # long enough to force multiple chunks
    chunks = recursive_chunk(text, source="doc.pdf", chunk_size=200,
                             chunk_overlap=40, start_id=5, page=3)
    assert len(chunks) > 1
    # ids are contiguous starting from start_id
    assert [c.chunk_id for c in chunks] == list(range(5, 5 + len(chunks)))
    # the page label is carried onto every chunk of this segment
    assert all(c.page == 3 for c in chunks)
    assert all(c.source == "doc.pdf" for c in chunks)


def test_short_text_single_chunk_no_page():
    chunks = recursive_chunk("Just a little text.", source="note.md",
                             chunk_size=800, chunk_overlap=120)
    assert len(chunks) == 1
    assert chunks[0].page is None


def test_parse_markdown_bytes_is_pageless():
    segments = parse_bytes("readme.md", b"# Title\n\nBody text here.")
    assert segments == [(None, "# Title\n\nBody text here.")]


def test_parse_rejects_unknown_extension():
    import pytest
    with pytest.raises(ValueError):
        parse_bytes("mystery.xyz", b"data")
