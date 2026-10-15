# Production RAG Audit — DocMind vs. *RAG in Production*

Baseline commit: `2de141a` · 24 tests passing · deployed (Streamlit Community Cloud).
Reference: `rag.pdf — RAG in Production: From Documents to Reliable AI Systems` (21 pp).

> **Status update — security hardening and UI rebuild.**
> A later pass audited this implementation rather than the plan, and found that
> several controls described below as "enforced" were not actually enforced in
> code. The most serious: the shared in-process index meant **one visitor's
> uploads were retrievable, listable and readable by every other visitor**, and
> the advertised `MAX_UPLOAD_MB` limit was read by no code path. Findings,
> evidence, fixes and regression tests are in
> [SECURITY_AUDIT.md](SECURITY_AUDIT.md); the current posture is in
> [SECURITY.md](SECURITY.md). Suite: **90 tests**. Read those two documents in
> preference to this one wherever they disagree — they describe the code as it
> is, this one describes the upgrade as it was planned.

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
| **Security (18)** | partial rate-limit; per-session scope | ✅ **owner-scoped retrieval, server-side upload validation (type/size/magic bytes), fence-escaping injection defense, LLM timeouts, restricted CORS**; multi-user auth/ACL ⏭ (documented — see DM-12) |
| Freshness/versioning (5,8,19) | partial replace-on-reupload | ✅ **content-hash skip + version metadata** |
| CI gates (3,15,20) | partial tests | ✅ **retrieval-eval threshold test** |

## 4. Deferred (with rationale)

- **Multi-user auth + tenant isolation + ACL-filtered retrieval** — requires a
  datastore and a multi-user server; out of scope for a single-process public demo.
  `MetaFilter` now carries an `owners` field that **is** enforced inside retrieval,
  and each browser session is given an opaque owner id, so one visitor's uploads
  stay out of another's answers. That is *isolation*, not authorization: there is
  no verified identity behind the id. Layering real auth means supplying
  `MetaFilter(owners=…)` from an authenticated principal instead of a session id —
  the retrieval side needs no redesign. See `SECURITY.md` and DM-12.
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

## 6. Corrections made by the security pass

| Claimed above | Reality in code | Now |
|---|---|---|
| "per-session scope" | scope came from session state that was empty for a new visitor, so retrieval fell back to *no filter* | owner stamped on every chunk; filter enforced inside each retriever; fail-closed on missing metadata |
| "size limit (`MAX_UPLOAD_MB`)" | the setting existed but no code read it | enforced in `src/uploads.py`, shared by every upload path |
| "content file-type check" | extension only, client-side | extension allow-list + magic bytes + NUL scan, server-side |
| "injection defense" via delimiters | a document could emit the closing delimiter and escape the fence | fence sequences escaped in both passage body and source label |
| "citation validation" | correct, but the UI labelled it "verified" | same check, honest labelling — existence, not factual support |
