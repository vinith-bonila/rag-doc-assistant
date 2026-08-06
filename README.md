# DocMind — RAG Document Assistant

> **Live demo:** _<!-- add your Cloud Run URL here after deploying, e.g. https://rag-assistant-xxxx.a.run.app -->_ **← paste the live link here**

Upload your PDFs, DOCX, TXT, or Markdown, ask questions, and get **streaming,
grounded answers with clickable source citations** — plus a live **evaluation
dashboard** that scores retrieval and answer quality. Built as a production-shaped
RAG system: **hybrid dense + BM25 retrieval**, **cross-encoder reranking**, and a
real metrics harness, behind a **FastAPI** backend and a **Streamlit** UI.

## Metrics (real, on the bundled eval set)

Measured by the built-in harness over 12 labelled questions — reproduce with
`python -m eval.evaluate` or the **Evaluation** tab in the app.

| Metric | Score | What it measures |
|---|---|---|
| **recall@5** | **1.00** | Correct source document is in the top-5 retrieved chunks (retriever quality) |
| **MRR** | **0.96** | How near the top the first correct passage lands after reranking (ordering) |
| **faithfulness** | **0.78** | Fraction of answer claims supported by context, via LLM-as-judge (anti-hallucination) |
| **answer relevance** | **0.82** | Answer is semantically on-topic for the question |

## What makes it more than a "chat with PDF" clone

- **Hybrid retrieval** — dense FAISS (meaning) **+** BM25 (exact keywords), fused
  with Reciprocal Rank Fusion so neither score scale dominates. Toggleable for
  ablation (`HYBRID=false`).
- **Cross-encoder reranking** — a shortlist re-scored by a model that reads
  query and passage together; its scores are shown next to each citation.
- **Grounded, cited, streaming answers** — the model answers only from retrieved
  context, cites passages inline as `[n]`, admits when it doesn't know, and
  streams token-by-token over **SSE**.
- **Clickable citations** — every answer's sources appear in a panel; click one
  to read the exact retrieved passage (with PDF page number).
- **Evaluation dashboard** — recall@k, MRR, faithfulness, and answer relevance
  as metric cards + a chart, each explained in one line, re-runnable in the UI.
- **Production touches** — Pydantic validation, structured logging, rate
  limiting, a typed & dependency-injected pipeline, pytest, Docker, and CI.

## Screenshots

_Add three screenshots from the running app to `docs/screenshots/` (streaming
chat with citations, an expanded citation, and the evaluation dashboard), then
they'll render here:_

<!--
![Streaming chat with citations](docs/screenshots/chat.png)
![Expanded citation panel](docs/screenshots/citation.png)
![Evaluation dashboard](docs/screenshots/eval.png)
-->

## Quickstart (local)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure an LLM key (Groq's free tier works great)
cp .env.example .env        # then edit LLM_API_KEY (see .env.example for Groq)

# 3. Build the index from the sample docs
python -m src.ingest --docs data/sample_docs

# 4. Run the backend (terminal 1)
uvicorn api.main:app --port 8000

# 5. Run the UI (terminal 2)
API_URL=http://localhost:8000 streamlit run ui/streamlit_app.py
#    open http://localhost:8501
```

Without an LLM key the app still runs retrieval + reranking (answers show the
top retrieved passage), so you can demo the retrieval half offline.

### Or run the whole stack in one container

```bash
docker build -t docmind .
docker run -p 8080:8080 -e LLM_API_KEY=... -e LLM_BASE_URL=https://api.groq.com/openai/v1 -e LLM_MODEL=llama-3.1-8b-instant docmind
#    open http://localhost:8080
```

## API

| Method & path | Purpose |
|---|---|
| `GET /health` | Liveness + index/LLM status |
| `POST /ask` | Grounded answer with citations (non-streaming) |
| `POST /ask/stream` | Same, streamed token-by-token over SSE |
| `POST /upload` | Add PDF/DOCX/TXT/MD files to the live index |
| `POST /eval` | Run the evaluation harness, return metrics |

Interactive docs at `http://localhost:8000/docs`.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What causes edge-ring defects on wafers?"}'
```

## Tests

```bash
pytest
```

Covers chunking + page-metadata propagation, BM25 lexical matching, RRF fusion
math, hybrid-retrieval recall, and citation correctness (sources are real,
ranked, and carry page numbers). CI (`.github/workflows/ci.yml`) runs lint +
tests on every push.

## Configuration

All settings live in `config.py`, overridable via environment variables (`.env`):

| Variable | Default | Meaning |
|---|---|---|
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | bi-encoder embedding model |
| `RERANKER_MODEL` | `ms-marco-MiniLM-L-6-v2` | cross-encoder reranker |
| `HYBRID` | `true` | fuse dense + BM25 (set `false` for dense-only) |
| `RRF_K` | `60` | reciprocal-rank-fusion constant |
| `RETRIEVE_K` | `20` | candidates per retriever before reranking |
| `RERANK_TOP_N` | `5` | passages kept after reranking |
| `CHUNK_STRATEGY` | `recursive` | `recursive` or `semantic` |
| `LLM_MODEL` | `gpt-4o-mini` | any OpenAI-compatible model |

## Project layout

```
config.py            all tunable settings
src/
  chunking.py        recursive + semantic chunking (page-aware)
  embeddings.py      sentence-transformers bi-encoder
  vectorstore.py     FAISS index (build/save/load/search)
  retrieval.py       dense + BM25 + RRF hybrid (composable Retriever interface)
  reranker.py        cross-encoder reranking
  llm.py             grounded generation + streaming, with citations
  pipeline.py        retrieve → rerank → generate; live upload indexing
  ingest.py          parse (pdf/docx/txt/md) → chunk → embed → FAISS
api/
  main.py            FastAPI: /health /ask /ask/stream /upload /eval
  schemas.py         Pydantic request/response models
ui/streamlit_app.py  upload · streaming chat · citations · eval dashboard
eval/                recall@k · MRR · faithfulness · answer relevance
tests/               pytest: chunking, retrieval, citations
docs/                ARCHITECTURE.md · DEPLOY.md
Dockerfile · run.sh  single-container backend + UI (Cloud Run ready)
```

## How it works

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full request flow, the
two-stage retrieval design, the SSE streaming contract, and the evaluation
rationale. Deploy steps are in [`docs/DEPLOY.md`](docs/DEPLOY.md).

## Tech stack

Python · FastAPI · FAISS · sentence-transformers · cross-encoders · rank-bm25 ·
Streamlit · OpenAI-compatible LLMs (Groq / OpenAI) · Docker · GitHub Actions
