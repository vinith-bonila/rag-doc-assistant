# Evaluation — DocMind

Quality is measured per **stage** so a regression points at the component that
caused it (rag.pdf p.15). Relevance labels are at the **source-document** level
(a chunk is relevant if it comes from the question's expected source).

## Metrics

**Retrieval (deterministic — no LLM):**
- `recall@k` — correct source appears in the top-k retrieved chunks.
- `precision@k` — share of the top-k that is relevant.
- `NDCG@k` — rank quality with a log position discount, normalised to the ideal.
- `retrieval MRR` / `MRR` (after rerank) — 1/rank of the first correct passage.
- `zero-result rate` — share of questions that retrieved nothing.

**Generation:**
- `faithfulness` — fraction of answer claims supported by context (LLM-as-judge
  when a key is set, else an embedding-overlap fallback).
- `answer relevance` — question↔answer semantic similarity.
- `citation validity` — share of cited answers whose every `[n]` maps to a real
  source (deterministic).

> Model-judged metrics (faithfulness) are **not** human-verified facts; calibrate
> against human labels before trusting absolute values.

## Dataset

`eval/eval_dataset.json` — a small versioned golden set of questions with the
expected source document. Extend it with paraphrases, exact-ID lookups,
table-based and multi-page questions, unanswerable cases, and adversarial
(injected) documents. Do not tune on the test set.

## Commands

```bash
python -m eval.evaluate          # full report (uses the LLM if configured)
```

In code:

```python
from eval.evaluate import run_eval, retrieval_eval
retrieval_eval(pipeline)         # deterministic retrieval metrics only (no LLM)
run_eval(pipeline)               # full report (retrieval + generation)
```

The **Analytics** tab in the app runs `run_eval` and shows the metric cards + chart.

## CI quality gate

`tests/test_eval.py::test_retrieval_quality_gate_full_set` asserts
`recall@5 >= 0.9` on the full golden set. It is **deterministic (no LLM / no
network)**, so it runs on every push as a regression gate without needing a paid
API key. Raise the threshold as the dataset grows; never silently lower it to go
green.

## Reference results

On the bundled sample corpus (12 chunks) with `openai/gpt-oss-20b`:
`recall@5 = 1.00`, `MRR ≈ 0.96`, `faithfulness ≈ 0.78`, `answer relevance ≈ 0.82`.
Reproduce with `python -m eval.evaluate`.
