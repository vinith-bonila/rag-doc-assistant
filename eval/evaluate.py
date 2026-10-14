"""Evaluation harness — what separates a real RAG project from a tutorial.

Four metrics, each isolating a different failure mode:

1. recall@k     — Does the gold source document appear among the top-k RETRIEVED
                  chunks? Measures the retriever alone: if it's low, the LLM
                  never had a chance. (Retrieval-stage failure.)

2. MRR          — Mean Reciprocal Rank of the first correct passage AFTER
                  reranking (1/rank, averaged; 1.0 = the right passage is always
                  first). Measures how well the cross-encoder orders results.

3. faithfulness — Are the answer's claims actually supported by the retrieved
                  context (not hallucinated)? LLM-as-judge when a key is set,
                  else an embedding-overlap fallback. (Generation-stage failure.)

4. answer relevance — Is the answer semantically on-topic for the question?
                  Cosine similarity between question and answer embeddings.
                  Catches evasive, empty, or off-topic answers.

`run_eval(pipeline)` returns a structured report consumed by both this CLI and
the /eval API + Streamlit dashboard.

Run:
    python -m eval.evaluate
"""
from __future__ import annotations

import json
import logging
import math
import os
from typing import Dict, List, Optional

import numpy as np

from src.pipeline import RAGPipeline
from src.validation import validate_citations

logger = logging.getLogger("docmind.eval")

EVAL_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
RECALL_K = 5


def load_questions() -> List[dict]:
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# --- individual metrics -------------------------------------------------------
def _reciprocal_rank(sources: List, expected: str) -> float:
    for rank, s in enumerate(sources, start=1):
        if s.source == expected:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(rels: List[int], k: int) -> float:
    """Normalised discounted cumulative gain for binary relevance labels."""
    rels = rels[:k]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(sorted(rels, reverse=True)))
    return dcg / idcg if idcg else 0.0


def retrieval_eval(pipeline: RAGPipeline, questions: Optional[List[dict]] = None,
                   k: int = RECALL_K) -> Dict:
    """Deterministic retrieval-only metrics (no LLM) — safe to run as a CI gate.
    Relevance is at the source-document level (a chunk is relevant if it comes
    from the question's expected source)."""
    questions = questions or load_questions()
    n = len(questions)
    recall = precision = ndcg = mrr = 0.0
    zero = 0
    for q in questions:
        expected = q["expected_source"]
        srcs = [c.source for c, _ in pipeline.retrieve(q["question"], k=k)]
        if not srcs:
            zero += 1
        rels = [1 if sdoc == expected else 0 for sdoc in srcs]
        if any(rels):
            recall += 1
        precision += (sum(rels) / len(rels)) if rels else 0.0
        ndcg += _ndcg_at_k(rels, k)
        for rank, sdoc in enumerate(srcs, start=1):
            if sdoc == expected:
                mrr += 1.0 / rank
                break
    return {f"recall@{k}": recall / n, f"precision@{k}": precision / n,
            f"ndcg@{k}": ndcg / n, "retrieval_mrr": mrr / n,
            "zero_result_rate": zero / n, "k": k, "n": n}


def faithfulness_llm(pipeline: RAGPipeline, answer: str, contexts: List[str]) -> float:
    """LLM-as-judge: what fraction of the answer's claims are supported?"""
    context = "\n\n".join(contexts)
    prompt = (
        "You are grading whether an ANSWER is fully supported by the CONTEXT.\n"
        "Return ONLY a number between 0 and 1 = the fraction of the answer's "
        "factual claims that are directly supported by the context. "
        "1 means fully grounded, 0 means unsupported/hallucinated.\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nScore:"
    )
    resp = pipeline.generator.client.chat.completions.create(
        model=pipeline.generator.model,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return float(raw.split()[0])
    except (ValueError, IndexError):
        return float("nan")


def faithfulness_embedding(pipeline: RAGPipeline, answer: str,
                           contexts: List[str]) -> float:
    """Fallback: max cosine similarity between answer and any context passage."""
    if not contexts:
        return 0.0
    a = pipeline.embedder.encode([answer])
    c = pipeline.embedder.encode(contexts)
    return float(np.max(a @ c.T))


def answer_relevance(pipeline: RAGPipeline, question: str, answer: str) -> float:
    """Cosine similarity between question and answer embeddings (both are
    L2-normalized, so the dot product is cosine). Clamped to [0, 1]."""
    if not answer.strip():
        return 0.0
    qa = pipeline.embedder.encode([question, answer])
    return float(max(0.0, min(1.0, qa[0] @ qa[1])))


# --- harness ------------------------------------------------------------------
def run_eval(pipeline: RAGPipeline, k: int = RECALL_K,
             questions: Optional[List[dict]] = None) -> Dict:
    """Run all metrics over the eval set and return a structured report."""
    questions = questions or load_questions()
    n = len(questions)
    use_llm = pipeline.generator is not None

    # Deterministic retrieval metrics (no LLM).
    ret = retrieval_eval(pipeline, questions, k)

    rr_sum = 0.0
    faith_scores: List[float] = []
    rel_scores: List[float] = []
    cite_scores: List[float] = []   # per answer with >=1 citation: all valid?

    for q in questions:
        expected = q["expected_source"]
        # Guard the LLM-dependent work so one transient error (e.g. a rate limit)
        # degrades to NaN for that question instead of aborting the whole run.
        try:
            result = pipeline.answer(q["question"])
            rr_sum += _reciprocal_rank(result.sources, expected)
            contexts = [s.text for s in result.sources]
            if use_llm:
                faith_scores.append(faithfulness_llm(pipeline, result.answer, contexts))
            else:
                faith_scores.append(faithfulness_embedding(pipeline, result.answer, contexts))
            rel_scores.append(answer_relevance(pipeline, q["question"], result.answer))
            rep = validate_citations(result.answer, len(result.sources))
            if rep.n_citations:
                cite_scores.append(1.0 if rep.ok else 0.0)
        except Exception:  # noqa: BLE001
            logger.exception("eval failed for question: %r", q["question"])
            faith_scores.append(float("nan"))
            rel_scores.append(float("nan"))

    faith_valid = [s for s in faith_scores if not np.isnan(s)]
    rel_valid = [s for s in rel_scores if not np.isnan(s)]

    def _mean(xs):
        return round(float(np.mean(xs)), 3) if xs else float("nan")

    metrics = [
        {"name": f"recall@{k}", "value": round(ret[f"recall@{k}"], 3),
         "explanation": f"Share of questions whose correct source is in the top-{k} "
                        "retrieved chunks — measures the retriever."},
        {"name": f"precision@{k}", "value": round(ret[f"precision@{k}"], 3),
         "explanation": f"Share of the top-{k} retrieved chunks that are relevant."},
        {"name": "MRR", "value": round(rr_sum / n, 3),
         "explanation": "Mean reciprocal rank of the first correct passage after "
                        "reranking (1.0 = always first) — measures ordering."},
        {"name": f"NDCG@{k}", "value": round(ret[f"ndcg@{k}"], 3),
         "explanation": "Rank quality with a log position discount, normalised to the "
                        "ideal ordering."},
        {"name": "faithfulness", "value": _mean(faith_valid),
         "explanation": "Fraction of answer claims supported by the retrieved context "
                        "— less hallucination." + ("" if use_llm else " (embedding fallback).")},
        {"name": "answer relevance", "value": _mean(rel_valid),
         "explanation": "Semantic similarity between question and answer — catches "
                        "evasive or off-topic answers."},
        {"name": "citation validity", "value": _mean(cite_scores),
         "explanation": "Share of cited answers whose every [n] marker maps to a real "
                        "retrieved source (deterministic check)."},
        {"name": "zero-result rate", "value": round(ret["zero_result_rate"], 3),
         "explanation": "Share of questions that retrieved nothing (lower is better)."},
    ]
    return {"metrics": metrics, "n_questions": n, "llm_judge": use_llm}


def main() -> None:
    pipeline = RAGPipeline()
    report = run_eval(pipeline)
    print(f"Evaluation over {report['n_questions']} questions "
          f"(faithfulness via {'LLM-as-judge' if report['llm_judge'] else 'embedding fallback'})\n")
    for m in report["metrics"]:
        print(f"  {m['name']:<16} {m['value']}")
        print(f"  {'':<16} {m['explanation']}")


if __name__ == "__main__":
    main()
