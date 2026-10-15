# Security posture — DocMind

DocMind is a **public single-user demo**: one Streamlit process, an in-memory
FAISS/BM25 index, no database, **no user accounts**. This document states what
is enforced in code, what is deliberately out of scope for that model, and what
would be required before confidential or multi-user use.

Every control listed under "Enforced today" is backed by a test in `tests/`.
Findings, evidence and fixes are in [SECURITY_AUDIT.md](SECURITY_AUDIT.md).

## The one thing to know first

**Metadata scoping is not authentication.** DocMind separates one browser
session's uploads from another's, which stops casual cross-visitor exposure. It
does not verify who anyone is. There is no login, no credential, no audit trail
and no encryption at rest.

> Do not upload confidential material to a public deployment.

The UI says this on the first-run screen and on the Settings page.

## Enforced today

### Session isolation (not authorization)

- Every chunk carries an `owner` in its metadata (`src/metadata.py`). The
  bundled corpus is `public`; each browser session gets a random opaque id.
- Retrieval always composes `MetaFilter(owners={session, "public"})`, applied
  **inside** each retriever (`src/retrieval.py`) — before rank fusion and before
  reranking — so no later stage can reintroduce an excluded chunk.
- The document list and the passage inspector are scoped the same way, so
  another session's filename is never even displayed.
- Session uploads are held in memory and are **not** written to the shared
  on-disk index, so they do not outlive the process or reach the next visitor.
- **Fail-closed:** a chunk with no `owner` matches no owner filter. Indexes
  built before the field existed are backfilled to `public` explicitly at load
  (`RAGPipeline._backfill_owners`), which is the only place that is known to be
  true.

### Upload safety

One authoritative check, in `src/uploads.py`, called from
`RAGPipeline.add_documents`, so the Streamlit uploader and the API cannot drift:

| Check | Rule |
|---|---|
| Filename | directory components stripped (both separators); `.`, `..`, NUL rejected |
| Type | allow-list: `.pdf`, `.docx`, `.txt`, `.md` |
| Size | `MAX_UPLOAD_MB` (default 20 MB) per file |
| Content | magic bytes (`%PDF-`, `PK\x03\x04`); NUL scan for text formats |
| Empty | rejected |

The Streamlit file picker's type filter and `maxUploadSize` in
`.streamlit/config.toml` are convenience hints, **not** the security boundary.
Uploaded names are only ever used as display/citation labels, never as paths.
DocMind writes no temporary files — parsing runs from `io.BytesIO` throughout.

### Prompt injection

Retrieved passages are untrusted data and are treated as such:

- each passage is fenced as `<<<SOURCE [n] …>>> … <<<END SOURCE [n]>>>`;
- the passage body **and** the source label are escaped against that fence
  (`neutralize_delimiters`), so a document cannot close its own block, inject
  prompt-level text, or forge a citation target;
- both system prompts instruct the model never to follow instructions found
  inside sources and never to reveal system instructions or secrets.

This is defence in depth, not a guarantee — prompt wording is not a security
boundary. The controls that do **not** depend on the model's cooperation are
owner-scoped retrieval and citation validation.

### Citation validation — and its limits

After generation, every inline `[n]` marker is checked against the retrieved
source list; out-of-range markers are rewritten to `[?]`
(`src/validation.py`). Be precise about what that proves:

| Question | Checked? |
|---|---|
| Does the citation id exist? | Yes |
| Does it map to a real retrieved source? | Yes |
| Is that source within the permitted scope? | Yes (retrieval is scoped) |
| **Does the cited passage support the claim?** | **No** |

The UI therefore says "Citation checks passed", never "verified" or
"hallucination-free". Semantic support is measured offline by the faithfulness
metric in `eval/`, not per answer.

### LLM and configuration

- Explicit request `timeout` (60s) and `max_retries` (2) — `LLM_TIMEOUT_S`,
  `LLM_MAX_RETRIES`.
- Missing/`None` response content is tolerated, not dereferenced.
- Secrets come only from the environment or Streamlit secrets. `.env` and
  `.streamlit/secrets.toml` are gitignored; no key is displayed in the UI,
  written into the page, or included in an error message.
- Provider errors show a short explanation, with the exception type and a
  truncated message behind a collapsed panel.

### Observability

`RAGTrace` records stage timings and counts only — never secrets, never raw
document text, never another session's filenames. Asserted by
`test_trace_scope_respects_owner_and_leaks_no_text`.

### FastAPI service

`api/` is **not** part of the Streamlit deployment; Streamlit Community Cloud
runs `ui/streamlit_app.py` only. It is kept for local use and is hardened
accordingly: Pydantic validation, slowapi rate limiting, a per-request file cap,
bounded chunked reads that abort at the size limit, CORS restricted via
`API_CORS_ORIGINS` (localhost by default), request shape logged instead of
question text, and `persist=False` on upload so an exposed instance cannot write
the shared on-disk index.

**It has no authentication. Do not expose it publicly.**

## Out of scope for this deployment model

| Control | Why deferred | What it would take |
|---|---|---|
| Authentication / sessions | No accounts in a public demo | Auth provider; gate uploads and eval |
| Tenant isolation with real ACLs | One shared process, no DB | Per-owner persistent store; `MetaFilter(owners=…)` from a verified identity |
| Encryption at rest, audit trail | Ephemeral in-memory storage | Managed vector store + encrypted object storage + audit sink |
| Revocation / deletion guarantees | No identity to revoke | Tied to the auth work above |
| Rate limiting on the Streamlit app | Community Cloud fronts it | Reverse proxy or host-level limits |

The retrieval layer already enforces the filter in the correct place, so the
remaining work is authentication and storage — not retrieval.

## Deployment guidance

**Safe:** a public demo whose users understand uploads are not confidential.

**Not safe without the work above:** confidential documents, multiple tenants,
regulated data, or exposing the FastAPI service.

## Reporting

This is a portfolio project; open a GitHub issue for security concerns.
