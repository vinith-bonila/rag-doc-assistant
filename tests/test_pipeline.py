"""Pipeline-level behaviour: streaming shape and live upload indexing."""
from __future__ import annotations

from src.pipeline import RAGPipeline


def test_answer_stream_returns_sources_then_tokens(pipeline):
    sources, tokens = pipeline.answer_stream("What does recall@k measure?")
    # Citations are available up front (the UI paints them before streaming).
    assert sources and all(s.text.strip() for s in sources)
    # Tokens form a non-empty answer (real stream with a key; fallback without).
    text = "".join(list(tokens))
    assert text.strip()


def test_add_documents_indexes_and_retrieves_new_file(tmp_path):
    # Fresh pipeline on a temp index so the shared session index isn't mutated.
    pipe = RAGPipeline(index_dir=str(tmp_path / "idx"))
    before = len(pipe.store.chunks)

    summary = pipe.add_documents(
        [("note.txt", b"Zorptastic gadgets hum quietly inside the calibration lab.")]
    )
    assert summary["total_chunks"] == before + 1
    assert summary["files"][0]["chunks_added"] == 1

    # The freshly indexed doc is retrievable (BM25 nails the rare keyword).
    got = pipe.retrieve("What hums quietly in the calibration lab?", k=5)
    assert any(c.source == "note.txt" for c, _ in got)


def test_add_documents_rejects_unsupported_type(tmp_path):
    import pytest
    pipe = RAGPipeline(index_dir=str(tmp_path / "idx2"))
    with pytest.raises(ValueError):
        pipe.add_documents([("malware.exe", b"MZ...")])
