"""Table-aware PDF parsing (rag.pdf p.6): a numeric value must stay connected to
its row/column headers instead of flattening into a bag of numbers.

Reads a committed fixture PDF (a gridded revenue table) so the test needs no PDF
authoring library at run time and works in CI."""
from __future__ import annotations

import os

from src.ingest import _table_to_markdown, parse_file

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "table_sample.pdf")


def test_table_to_markdown_serialization():
    md = _table_to_markdown([["Region", "Q2", "Q3"], ["EMEA", "18.7", "21.3"]])
    assert "| Region | Q2 | Q3 |" in md
    assert "| --- | --- | --- |" in md
    assert "| EMEA | 18.7 | 21.3 |" in md


def test_pdf_table_extracted_to_markdown():
    assert os.path.exists(FIXTURE), "table fixture PDF missing"
    text = "\n".join(t for _, t in parse_file(FIXTURE))
    # The value is recoverable *with* its labels, as a Markdown table row.
    assert "| Region | Q2 | Q3 |" in text
    assert "| EMEA | 18.7 | 21.3 |" in text
