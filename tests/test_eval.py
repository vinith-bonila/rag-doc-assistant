"""Evaluation harness: deterministic retrieval metrics (a CI gate — no LLM) and
the full report shape (LLM when a key is present, else embedding fallback)."""
from __future__ import annotations

from eval.evaluate import retrieval_eval, run_eval

SUBSET = [
    {"question": "What does recall@k measure in a RAG system?",
     "expected_source": "rag_concepts.md"},
    {"question": "How does HNSW achieve fast approximate nearest neighbour search?",
     "expected_source": "vector_databases.pdf"},
]


def test_retrieval_eval_is_deterministic(pipeline):
    r = retrieval_eval(pipeline, questions=SUBSET, k=5)
    assert r["recall@5"] == 1.0              # both subset questions retrievable
    assert 0.0 <= r["precision@5"] <= 1.0
    assert 0.0 <= r["ndcg@5"] <= 1.0
    assert r["zero_result_rate"] == 0.0


def test_retrieval_quality_gate_full_set(pipeline):
    """CI quality gate (rag.pdf p.3/15): retrieval must not regress. No LLM."""
    r = retrieval_eval(pipeline, k=5)        # full labelled eval set
    assert r["recall@5"] >= 0.9, r


def test_run_eval_reports_expanded_metrics(pipeline):
    report = run_eval(pipeline, questions=SUBSET)
    names = {m["name"] for m in report["metrics"]}
    assert {"recall@5", "precision@5", "MRR", "NDCG@5", "faithfulness",
            "answer relevance", "citation validity", "zero-result rate"} <= names
    assert report["n_questions"] == 2
    by = {m["name"]: m["value"] for m in report["metrics"]}
    assert 0.0 <= by["recall@5"] <= 1.0
