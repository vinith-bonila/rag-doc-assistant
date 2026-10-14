# Production RAG Audit — DocMind vs. *RAG in Production*

Baseline commit: `2de141a` · 24 tests passing · deployed (Streamlit Community Cloud).
Reference: `rag.pdf — RAG in Production: From Documents to Reliable AI Systems` (21 pp).

This audit records the **existing** architecture, maps it feature-by-feature against
the production guide, and states what this upgrade implements vs. what is
deliberately deferred and why.

## 1. Deployment model (sets the scope)

DocMind is a **single-process, single-user, free public demo**: one Streamlit app
runs the RAG pipeline **in-process** (`st.cache_resource`), with an in-memory FAISS
index + BM25 rebuilt at load, **no database**, and no per-user accounts. A FastAPI
service lives in `api/` (REST + SSE) for local/API use but is not the deployed
surface.

This matters: the guide's enterprise controls (multi-tenant auth, ACL-filtered
retrieval, tenant isolation, distributed tracing backends) assume a multi-user
server with a datastore. On a single-process public demo they are **not
applicable as built** — the guide itself says *"add a component when evaluation
shows a gap it would close."* They are documented here and in `SECURITY.md` with an
extensible interface, not force-fit.

## 2. Existing architecture (verified)

- **Ingestion** (`src/ingest.py`): PDF/DOCX/TXT/MD → `clean_pdf_text` (ligature/PUA
  repair) → recursive/semantic chunking with **page metadata** → FAISS + BM25.
  Re-upload **replaces** a file's chunks (no duplicate growth).
- **Retrieval** (`src/retrieval.py`): composable `Retriever` — dense (FAISS exact
  `IndexFlatIP`) + sparse (BM25) fused by **RRF (k=60)**; `allowed_sources` scoping
  already supported.
- **Rerank** (`src/reranker.py`): cross-encoder `ms-marco-MiniLM-L-6-v2`, two-stage.
- **Generation** (`src/llm.py`): grounded + cited (`[n]`) with a refusal path;
  structured assistant answer; streaming; conversation memory + query rewrite.
- **Eval** (`eval/evaluate.py`): recall@k, MRR, faithfulness (LLM-judge), answer
  relevance, golden set, dashboard.
- **Engineering**: FastAPI + Pydantic + slowapi + structured logging; lazy models;
  thread-locked shared pipeline; 24 pytest; ruff; GitHub Actions CI; Docker;
  deployed. LLM via Groq (`openai/gpt-oss-20b`).

## 3. Feature-by-feature vs. the guide

Legend: ✅ full · partial partial · ❌ not built · ⏭ deferred (documented)

| Guide area (pp.) | Status before | This upgrade |
|---|---|---|
| Ingestion (5) | partial multi-format, no connectors/OCR | hardening + content-hash dedupe |
| **Parsing / tables (6)** | ❌ naive text (tables flatten) | ✅ **table-aware (pdfplumber → Markdown)** |
| Chunking (7) | ✅ recursive + semantic | metadata carried through |
| **Metadata (8)** | ❌ source+page+chunk_id only | ✅ **rich meta (type, hash, versions, title)** |
| Embeddings/Index (9) | ✅ MiniLM; FAISS exact (ANN ⏭ — not needed at scale) | version tagging |
| Retrieval (10) | ✅ dense+BM25+RRF | **metadata filtering** |
| Reranking (11) | ✅ cross-encoder two-stage | (verified) |
| **Context construction (12)** | partial order+label only | ✅ **dedup + token budget + insufficient-context guard** |
| Generation (13) | ✅ grounded/cited/refuse | **prompt-injection hardening** |
| **Validation/guardrails (14)** | partial refusal guard only | ✅ **live citation validation** + injection defense |
| Evaluation (15) | ✅ 4 metrics | ✅ **+ Precision@K, NDCG@K, zero-result, citation-validity** |
| **Observability (16)** | partial logging + latency | ✅ **per-query RAGTrace (per-stage timings/counts)** |
| Latency/Cost (17) | partial streaming, small models | (caching/parallel ⏭) |
| **Security (18)** | partial rate-limit; per-session scope | content file-type check, injection defense; multi-user auth/ACL ⏭ (documented) |
| Freshness/versioning (5,8,19) | partial replace-on-reupload | ✅ **content-hash skip + version metadata** |
| CI gates (3,15,20) | partial tests | ✅ **retrieval-eval threshold test** |

## 4. Deferred (with rationale)

- **Multi-user auth + tenant isolation + ACL-filtered retrieval** — requires a
  datastore and a multi-user server; out of scope for a single-process public demo.
  An extensible `MetaFilter` (sources/types, and an `access` field in the metadata
  schema) is added so authorization can be layered later without a redesign. See
  `SECURITY.md`.
- **OCR / vision parsing** — heavy; image-only pages are *detected and flagged*
  rather than silently indexed; full OCR deferred.
- **ANN index (HNSW/IVF)** — exact FAISS is correct for this corpus size; the guide
  says use ANN *"at scale."* Kept exact; documented.
- **OpenTelemetry backend / cross-tenant caching / parallel retrieval** — the
  in-process `RAGTrace` + structured logs cover observability for one process;
  caching/parallelism are marginal at this scale. Documented as next steps.

## 5. Acceptance criteria for this upgrade

- All existing behaviour preserved; existing + new tests green; Docker builds.
- New capabilities deterministic and testable **without a paid LLM**.
- No secrets committed; no user-data loss; backward-compatible APIs.
