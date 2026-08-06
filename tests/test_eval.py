"""Evaluation harness: the report exposes all four metrics with sane values.

Runs on a 2-question subset to stay fast (the full set makes many LLM calls).
recall and MRR are deterministic (no LLM), so they must be valid in [0, 1].
"""
from __future__ import annotations

from eval.evaluate import run_eval

SUBSET = [
    {"question": "What does recall@k measure in a RAG system?",
     "expected_source": "rag_concepts.md"},
    {"question": "How does HNSW achieve fast approximate nearest neighbour search?",
     "expected_source": "vector_databases.pdf"},
]


def test_run_eval_reports_all_metrics(pipeline):
    report = run_eval(pipeline, questions=SUBSET)
    names = [m["name"] for m in report["metrics"]]
    assert names == ["recall@5", "MRR", "faithfulness", "answer relevance"]
    assert report["n_questions"] == 2
    assert isinstance(report["llm_judge"], bool)

    by = {m["name"]: m["value"] for m in report["metrics"]}
    # Retrieval metrics don't depend on the LLM and must be valid probabilities.
    assert 0.0 <= by["recall@5"] <= 1.0
    assert 0.0 <= by["MRR"] <= 1.0
    # Both subset questions are clearly answerable from the corpus.
    assert by["recall@5"] == 1.0
