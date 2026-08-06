# Architecture

DocMind is built around one typed, modular `retrieve → rerank → generate`
pipeline. It runs in **two interchangeable front doors**, both calling the same
`RAGPipeline`:

- **Streamlit app (`ui/`)** — the deployed product. Calls the pipeline
  **in-process** (no network hop), so the whole thing fits one free Streamlit
  Community Cloud service.
- **FastAPI service (`api/`)** — the same pipeline exposed as REST + SSE, with
  Pydantic validation, structured logging, and rate limiting. Preserved for API
  use and local dev.

## Request flow (deployed: Streamlit, in-process)

```
        ┌──────────────────────────── Streamlit UI (ui/streamlit_app.py) ───────────────────────────┐
        │  upload area   ·   streaming chat   ·   citations panel   ·   evaluation dashboard          │
        │  one cached RAGPipeline (st.cache_resource) · models lazy-loaded on first use               │
        └───────┬────────────────────────┬───────────────────────────────────────┬───────────────────┘
         add_documents()           answer_stream()                            run_eval()
                │                        │                                        │
   ┌────────────▼─────────┐   ┌──────────▼───────────────────────────┐   ┌────────▼─────────┐
   │ ingest.py            │   │ pipeline.py (RAGPipeline)            │   │ evaluate.py      │
   │ parse pdf/docx/txt/md│   │  1. retrieval.py  hybrid search      │   │ recall@k · MRR   │
   │ → chunk (+page meta) │   │       dense (FAISS) + BM25 (RRF)     │   │ faithfulness     │
   │ → embed → FAISS      │   │  2. reranker.py   cross-encoder      │   │ answer relevance │
   └──────────────────────┘   │  3. llm.py        grounded, cited,   │   └──────────────────┘
                              │                   streaming answer   │
                              └──────────────────────────────────────┘

The FastAPI service (api/main.py) wraps the SAME pipeline methods as HTTP:
   POST /upload → add_documents()   POST /ask/stream (SSE) → answer_stream()   POST /eval → run_eval()
```

## Memory & lazy loading

The pipeline is deliberately cheap to construct and light to hold:

- **Lazy models.** `embedder`, `reranker`, and the LLM `generator` are `@property`
  singletons built on first access — booting the app only loads the (tiny) FAISS
  index. The first query is when models actually load.
- **Single instance, no duplicates.** One embedder is reused for indexing,
  retrieval, and the embedding-based eval metrics. `st.cache_resource` keeps a
  single pipeline per Streamlit server across reruns and users.
- **Self-healing index.** If no persisted index exists (fresh Streamlit Cloud
  container), the pipeline builds one from the bundled sample docs at startup,
  reusing its own embedder.

## Retrieval, in two stages

**Stage 1 — Hybrid candidate generation (`retrieval.py`).**
Two retrievers run in parallel over the same chunk set:

- **Dense** (`DenseRetriever`): the query is embedded with a MiniLM bi-encoder
  and matched against a FAISS inner-product index of normalized vectors (= cosine
  similarity). Captures meaning and paraphrase.
- **Sparse** (`BM25Retriever`): classic lexical BM25 over tokenized chunks.
  Captures exact keywords, identifiers, and acronyms that embeddings blur.

Their ranked lists are fused with **Reciprocal Rank Fusion**:

```
fused(d) = Σ_r  1 / (rrf_k + rank_r(d))
```

RRF fuses on *rank position*, not raw scores, so FAISS's cosine scale and BM25's
unbounded scale can't dominate each other. Toggle `HYBRID=false` for a dense-only
ablation.

**Stage 2 — Cross-encoder reranking (`reranker.py`).**
The bi-encoder scores query and passage independently and is fast; a
cross-encoder reads the `(query, passage)` pair *together* and scores relevance
far more accurately, but is slow. So we retrieve many candidates cheaply, then
rerank a small shortlist. The reranker's scores are the ones shown next to each
citation in the UI.

## Generation (`llm.py`)

The reranked passages are numbered and injected into the prompt. The model is
instructed to answer **only** from that context, to cite passages inline as
`[1]`, `[2]`, and to say it doesn't know when the context is insufficient — the
first line of defense against hallucination. `stream_answer` yields content
deltas so the API can forward tokens over SSE as they're produced.

## Streaming contract (SSE)

`POST /ask/stream` emits Server-Sent Events in this order:

1. one `sources` event — the reranked citations, so the UI paints the panel
   immediately;
2. many `token` events — `{"delta": "..."}` chunks of the answer;
3. a terminal `done` event.

The frontend renders sources first, then streams tokens into the chat bubble.

## Evaluation (`evaluate.py`)

Four metrics, each isolating a different failure mode:

| Metric | Stage measured | Question it answers |
|---|---|---|
| **recall@k** | retriever | Did the gold document reach the top-k candidates at all? |
| **MRR** | reranker | How near the top is the first correct passage after reranking? |
| **faithfulness** | generation | Are the answer's claims supported by the context (not hallucinated)? |
| **answer relevance** | generation | Is the answer actually on-topic for the question? |

Separating retrieval-stage from generation-stage failure is the first thing you
want to know when a RAG answer is wrong. Faithfulness uses LLM-as-judge when a
key is configured, else an embedding-overlap fallback.

## Design choices worth calling out

- **Composable `Retriever` interface.** Dense, sparse, and hybrid all implement
  `retrieve(query, k) -> [(Chunk, score)]`, so the stack is built by composition
  (`build_retriever`) and each piece is unit-testable in isolation.
- **Dependency injection.** The embedder, reranker, and LLM generator are
  constructed once and injected into `RAGPipeline`; tests swap in fakes (see the
  RRF fusion test, which uses fake retrievers and no models).
- **Page-aware citations.** Parsing yields `(page, text)` segments, so PDF page
  numbers survive all the way into the clickable citation.
- **One index, live updates.** Uploads are parsed, chunked, embedded, and added
  to the store at runtime; the BM25 side is rebuilt over the full set and the
  index is persisted so new docs survive a restart.
