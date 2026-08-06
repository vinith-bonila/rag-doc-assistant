"""Pipeline-level behaviour: streaming shape and live upload indexing."""
from __future__ import annotations

from src.pipeline import RAGPipeline, _stop_after_refusal


def test_stop_after_refusal_truncates_trailing_sections():
    def fake_stream():
        for t in ["📄 **According to the document**\n",
                  "The uploaded document does not contain ",
                  "enough information to answer this question.\n",
                  "💡 **AI explanation**\n",
                  "Some general knowledge that must NOT appear."]:
            yield t
    out = "".join(_stop_after_refusal(fake_stream()))
    assert "does not contain enough information" in out
    assert "💡" not in out and "must NOT appear" not in out


def test_stop_after_refusal_passes_through_normal_answers():
    def fake_stream():
        for t in ["📄 grounded [1]\n", "💡 explanation\n", "🌍 example"]:
            yield t
    out = "".join(_stop_after_refusal(fake_stream()))
    assert "💡 explanation" in out and "🌍 example" in out


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


def test_retrieval_scopes_to_uploaded_sources(tmp_path):
    # After uploading their own doc, a user's query must not pull from the other
    # documents in the shared index (the "summarize this document" bug).
    pipe = RAGPipeline(index_dir=str(tmp_path / "idx_scope"))
    pipe.add_documents([("mine.txt", b"Quarterly revenue rose on strong CPF throughput.")])

    got = pipe.retrieve("summary", k=10, allowed_sources={"mine.txt"})
    assert got, "expected at least one scoped result"
    assert {c.source for c, _ in got} == {"mine.txt"}  # nothing from the sample corpus


def test_reupload_replaces_without_growth(tmp_path):
    # Re-uploading the same filename must replace its chunks (releasing the old
    # vectors), not duplicate them — otherwise repeated uploads leak memory.
    pipe = RAGPipeline(index_dir=str(tmp_path / "idx3"))
    base = len(pipe.store.chunks)

    pipe.add_documents([("a.txt", b"The quokka is a small marsupial from Australia.")])
    assert len(pipe.store.chunks) == base + 1

    summary = pipe.add_documents([("a.txt", b"The axolotl is a regenerating amphibian.")])
    assert len(pipe.store.chunks) == base + 1          # no duplicate growth
    assert summary["total_chunks"] == base + 1

    # ids stay unique after the rebuild, and the index size matches the chunks.
    ids = [c.chunk_id for c in pipe.store.chunks]
    assert len(ids) == len(set(ids))
    assert pipe.store.index.ntotal == len(pipe.store.chunks)

    # The old content is gone; the new content is retrievable.
    text = " ".join(c.text for c, _ in pipe.retrieve("axolotl amphibian", k=5)).lower()
    assert "axolotl" in text and "quokka" not in text
