"""Retrieval quality: BM25 lexical matching, RRF fusion math, and end-to-end
recall of the hybrid+rerank stack on the sample corpus."""
from __future__ import annotations

from src.chunking import Chunk
from src.retrieval import BM25Retriever, HybridRetriever


def _chunk(cid: int, text: str) -> Chunk:
    return Chunk(text=text, source=f"s{cid}.md", chunk_id=cid)


def test_bm25_finds_exact_keyword():
    chunks = [
        _chunk(0, "The cat sat on the mat."),
        _chunk(1, "Population stability index detects data drift in production."),
        _chunk(2, "Bananas are yellow and rich in potassium."),
    ]
    bm25 = BM25Retriever(chunks)
    ranked = bm25.retrieve("population stability index drift", k=3)
    assert ranked[0][0].chunk_id == 1  # the lexically matching chunk ranks first


def test_rrf_fusion_rewards_cross_list_agreement():
    # The point of fusion: a chunk both retrievers surface should beat a chunk
    # only one retriever surfaces, even at rank #1. Here c_common is #2 in BOTH
    # lists; c_solo and c_extra are each #1 in just one list. For any rrf_k>0,
    # 2/(k+2) > 1/(k+1), so the agreed-upon chunk wins.
    c_solo, c_common, c_extra = _chunk(1, "a"), _chunk(2, "b"), _chunk(3, "c")

    class Fake:
        def __init__(self, order):
            self._order = order

        def retrieve(self, query, k):
            return [(c, 1.0) for c in self._order][:k]

    dense = Fake([c_solo, c_common])    # c_solo #1, c_common #2
    sparse = Fake([c_extra, c_common])  # c_extra #1, c_common #2
    fused = HybridRetriever([dense, sparse], rrf_k=60).retrieve("q", k=3)
    assert fused[0][0].chunk_id == 2


def test_hybrid_retrieve_respects_k(pipeline):
    got = pipeline.retrieve("What is reranking?", k=4)
    assert 0 < len(got) <= 4
    assert all(isinstance(score, float) for _, score in got)


def test_end_to_end_recall_hits_expected_source(pipeline):
    # A keyword-heavy query that dense-only can miss but hybrid should catch.
    got = pipeline.retrieve("population stability index concept drift", k=10)
    sources = {c.source for c, _ in got}
    assert "model_monitoring.docx" in sources
