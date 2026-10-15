# DocMind — security and correctness audit

Audit of the DocMind repository, with fixes applied and regression tests added.
Every finding below was confirmed by reading the implementation; the ones marked
**Reproduced** were additionally demonstrated against the running code before
being fixed. Nothing here is a hypothetical.

- **Scope:** the Streamlit app (`ui/`), the RAG core (`src/`), the FastAPI
  service (`api/`), configuration and deployment files.
- **Deployment model assumed:** one Streamlit Community Cloud process, shared by
  every visitor, no user accounts. See `docs/SECURITY.md`.
- **Verification commands:** see [Reproducing the verification](#reproducing-the-verification).

| Status | Count |
|---|---|
| Fixed, with regression test | 16 |
| Mitigated + documented (cannot be fully fixed without auth) | 1 (DM-12) |
| Upstream, worked around | 1 (UI-01) |

Suite: **90 tests**.

---

## Severity summary

| ID | Severity | Component | Finding | Status |
|---|---|---|---|---|
| DM-01 | **Critical** | `ui/`, `src/pipeline.py` | One visitor's uploads were retrievable, listable and readable by every other visitor | Fixed |
| DM-02 | **High** | `ui/streamlit_app.py` | Document filenames interpolated into `unsafe_allow_html` markup | Fixed |
| DM-03 | **High** | `src/llm.py` | Document text could close its own source fence and inject prompt-level instructions | Fixed (Reproduced) |
| DM-04 | **High** | `src/llm.py` | No LLM request timeout — a stalled provider hung the session for up to 10 minutes | Fixed |
| DM-05 | Medium | `ui/streamlit_app.py` | "chunks retrieved" was a computed guess, not the real count | Fixed |
| DM-06 | Medium | `ui/streamlit_app.py` | Overconfident "✓ Grounded" / "Citations verified" labels | Fixed |
| DM-07 | Medium | `src/llm.py`, `eval/` | `message.content` is Optional; dereferencing it raised `AttributeError` | Fixed |
| DM-08 | Medium | `src/pipeline.py` | `MAX_UPLOAD_MB` was never enforced anywhere; type checking was client-side only | Fixed |
| DM-09 | Medium | `api/main.py` | Oversized uploads were fully buffered in memory before rejection | Fixed |
| DM-10 | Medium | `api/main.py` | `allow_origins=["*"]` on an unauthenticated service | Fixed |
| DM-11 | Low | `api/main.py` | Full user questions written to logs | Fixed |
| DM-12 | **High** | architecture | No authentication; metadata scoping is not authorization | **Mitigated + documented** |
| DM-13 | Low | `api/main.py` | `UploadFile.filename` may be `None` → 500 | Fixed |
| DM-14 | Low | `ui/streamlit_app.py` | Navigation items that led to placeholder pages | Fixed |
| DM-15 | Low | `ui/streamlit_app.py` | Raw exception text rendered in the main flow | Fixed |
| DM-16 | Low | `ui/streamlit_app.py` | Settings surface could expose provider configuration | Fixed |
| DM-17 | Medium | `api/main.py` | Unauthenticated API uploads were persisted into the shared on-disk index | Fixed |
| UI-01 | Medium | `ui/streamlit_app.py` | Document filter used a control that renders no options in the pinned Streamlit | Fixed (Reproduced) |
| UI-02 | Medium | `ui/`, `.streamlit/` | A runtime theme toggle could not restyle Streamlit's own widgets | Fixed |
| UI-03 | Low | `ui/theme.py` | Small copy fell below WCAG AA contrast in both palettes | Fixed |
| UI-04 | Medium | `ui/theme.py` | Page titles rendered underneath the opaque app header and were clipped | Fixed (Reproduced) |

---

## DM-01 — Cross-session document exposure (Critical)

**Where:** `ui/streamlit_app.py` (pipeline caching + scope), `src/pipeline.py`
(`add_documents`).

**What was wrong.** The pipeline was built once with `@st.cache_resource`, so a
single `RAGPipeline` — and a single FAISS index — was shared by every visitor of
the deployed app. `add_documents` appended uploads to that shared index *and*
persisted them with `store.save(self.index_dir)`.

Scope was derived from `st.session_state["uploaded_sources"]`, which is empty for
a visitor who has not uploaded anything, and the code fell back to
`scope_sources = None`, i.e. *no filter*. The consequences, for a second visitor
arriving after someone uploaded a private document:

1. their questions retrieved passages from that document and quoted them in
   answers;
2. the Documents view listed its filename;
3. expanding it printed its full text.

Uploads also survived process restarts, so exposure was permanent, and the index
grew without bound.

**Attack scenario.** Visitor A uploads a contract to the public demo. Visitor B
opens the same URL, asks "summarize the documents", and receives its contents —
no interaction with A required.

**Fix.**

- Every chunk carries an `owner` in its metadata (`src/metadata.py`,
  `stamp_metadata`). The bundled corpus is `OWNER_PUBLIC`; each browser session
  gets a random opaque id (`ui/streamlit_app.py`, `st.session_state.owner`).
- `MetaFilter` gained an `owners` field, and `_build_filter` always composes
  `{session_owner, "public"}`. The filter is applied **inside** each retriever
  (`src/retrieval.py`), before RRF fusion and before reranking, so no later
  stage can reintroduce an excluded chunk.
- `visible_documents(owner)` and `chunks_for(source, owner)` scope the document
  list and the passage inspector, so another session's file is never even named.
- `add_documents(..., persist=None)` defaults to persisting **only** the public
  corpus. A visitor's upload lives in memory for the life of the process and is
  never written to the shared on-disk index.
- Per-owner keying of the content-hash map and of the replace-on-reupload set,
  so one session's `notes.txt` cannot shadow or evict another's.
- **Fail-closed:** `chunk_owner()` returns `OWNER_UNKNOWN` (`""`) when the field
  is missing, which matches no owner filter, so a metadata gap *narrows* access.
  Backwards compatibility for pre-`owner` indexes is handled explicitly by
  `RAGPipeline._backfill_owners()`, which stamps `public` on chunks loaded from
  disk — the only place where "this is public" is actually known to be true.

**Regression tests** (`tests/test_security.py`):
`test_uploads_are_isolated_per_owner`,
`test_answer_stream_sources_respect_owner_scope`,
`test_answer_respects_owner_scope`,
`test_session_uploads_are_not_written_to_the_shared_index`,
`test_same_filename_from_two_owners_does_not_collide`,
`test_reindex_after_second_upload_preserves_isolation`,
`test_missing_owner_metadata_fails_closed`,
`test_legacy_persisted_index_is_backfilled_to_public`,
`test_trace_scope_respects_owner_and_leaks_no_text`,
`test_metafilter_owner_excludes_other_sessions`.

**Remaining risk.** This is *isolation*, not authentication — see DM-12.

---

## DM-02 — HTML injection through document filenames (High)

**Where:** `ui/streamlit_app.py`, document rows, scope labels, inspector
headings.

**What was wrong.** Filenames were interpolated directly into strings rendered
with `st.markdown(..., unsafe_allow_html=True)`. Streamlit renders that markup
verbatim, and React's `dangerouslySetInnerHTML` executes event-handler
attributes, so a file named `x<img src=x onerror=...>.pdf` injected live markup.
Combined with DM-01 (filenames shown to other visitors) this was stored XSS
rather than self-XSS.

**Fix.** `ui/theme.py` exposes `esc()` (`html.escape(..., quote=True)`), and
every builder — `doc_row`, `badge`, `metric`, `kv_rows`, `status_pill` — escapes
the values it interpolates. The app escapes page headings and scope labels at
the call site. The only unescaped markup is the static flow chart, which
contains no user data.

**Regression tests** (`tests/test_ui_theme.py`): `test_esc_neutralizes_markup`,
`test_builders_escape_every_interpolation`, `test_kv_rows_escapes_both_sides`.

---

## DM-03 — Prompt-injection fence escape (High, **Reproduced**)

**Where:** `src/llm.py`, `_format_context`.

**What was wrong.** Retrieved passages were wrapped as
`<<<SOURCE [n] source=...>>> … <<<END SOURCE [n]>>>`, and both system prompts
told the model that everything between those markers is untrusted data. The
passage body was inserted verbatim, so a document containing the literal closing
marker ended its own block early — everything after it then read as
prompt-level text rather than quoted data — and could also open a forged
`<<<SOURCE [2] source=fake.pdf>>>` block to manufacture a citation target.

**Reproduction (before the fix).** A chunk whose text contained
`<<<END SOURCE [1]>>>` followed by an instruction produced a context with **two**
closing fences, the injected instruction sitting outside any fence.

**Fix.** `neutralize_delimiters()` breaks the `<<<` / `>>>` runs with zero-width
spaces before formatting; it is applied to both the passage body and the source
label (so a crafted *filename* cannot forge a fence either). The text reads
identically to a human and to the model, but can no longer be parsed as a
delimiter. Verified after the fix: exactly one opening and one closing fence
survive, and the injected instruction remains present as quotable data.

**Regression tests** (`tests/test_security.py`):
`test_document_cannot_close_its_own_source_fence`,
`test_crafted_filename_cannot_forge_a_fence`,
`test_neutralize_delimiters_preserves_readable_text`.

**Remaining risk.** Delimiting plus instructions is defence in depth, not a
guarantee. A sufficiently persuasive document can still influence an answer. The
structural controls that do not depend on the model's cooperation are citation
validation (DM-06) and owner-scoped retrieval (DM-01).

---

## DM-04 — No LLM timeout (High)

**Where:** `src/llm.py`, `Generator.__init__`.

**What was wrong.** `OpenAI(api_key=..., base_url=...)` was constructed without
`timeout` or `max_retries`. The SDK default is a **600-second** timeout, so a
stalled provider left the Streamlit session spinning for up to ten minutes with
no feedback and no way to recover.

**Fix.** Explicit `timeout` (default 60s) and `max_retries` (default 2), wired
from `settings.llm_timeout_s` / `settings.llm_max_retries` (`LLM_TIMEOUT_S`,
`LLM_MAX_RETRIES`).

**Regression test:** `test_generator_sets_explicit_timeout_and_retries`.

---

## DM-05 — Fabricated retrieval count (Medium)

**Where:** `ui/streamlit_app.py`, answer metadata.

**What was wrong.** The UI displayed `min(settings.retrieve_k, scope_chunks)` as
"**N** chunks retrieved". That is an upper bound computed from configuration,
not an observation — it disagrees with reality whenever retrieval returns fewer
candidates. The real count was already recorded in the trace.

**Fix.** The count is read from `RAGTrace.meta["candidates_retrieved"]`, set in
`RAGPipeline._rerank`, and is shown only when present.

**Verified in the running app:** an unfiltered question reported
`candidates retrieved = 12`; the same question scoped to one document reported
`candidates retrieved = 5`, matching that document's passage count.

---

## DM-06 — Overconfident validation labels (Medium)

**Where:** `ui/streamlit_app.py`.

**What was wrong.** Every answer was labelled "✓ Grounded" unconditionally, and
a passing citation check was labelled "✓ Citations verified". The implemented
check (`src/validation.py`) only verifies that each `[n]` marker falls within
the range of retrieved sources. It does **not** check that the cited passage
supports the claim. The labels implied verification the code does not perform.

**Fix.** `validation_badges()` emits precise, condition-matched wording:

| Condition | Label |
|---|---|
| sources attached | "N source passage(s) linked" |
| every `[n]` resolves | "Citation checks passed" |
| some markers out of range | "Some citations need review" |
| markers present, none valid | "Insufficient supporting evidence" |
| no markers at all | "No citations in this answer" |

The Technical page states explicitly what is and is not checked. Each badge
carries a glyph and wording, so status never depends on colour alone.

**Verified in the running app:** a question scoped to a document that cannot
answer it produced the model's refusal plus "! No citations in this answer" and
the caption "The answer did not cite specific passages."

---

## DM-07 — `None` response content (Medium)

**Where:** `src/llm.py` (`answer`, `condense_query`), `eval/evaluate.py`.

`resp.choices[0].message.content` is `Optional[str]` — a filtered or tool-only
reply returns `None`, and `.strip()` on it raises `AttributeError`. Replaced
with `_content_of()`, which tolerates a missing choice list, a missing message
and `None` content.

**Regression test:** `test_missing_response_content_does_not_raise`.

---

## DM-08 — Upload limits were never enforced (Medium)

**Where:** `src/pipeline.py` / new `src/uploads.py`.

**What was wrong.** `settings.max_upload_mb` existed in `config.py` and was
advertised in `docs/SECURITY.md`, but `grep` confirmed **no code read it**. The
only checks were `st.file_uploader(type=[...])` (client-side convenience) and
`maxUploadSize` in `.streamlit/config.toml`. The API had its own hard-coded and
inconsistent 15 MB constant. A renamed binary (`malware.exe` → `report.pdf`) was
handed straight to `pypdf`.

**Fix.** New `src/uploads.py`, called from `RAGPipeline.add_documents`, so every
entry path (Streamlit, API, tests) shares one authoritative check:

1. `safe_filename()` — strips directory components on both separators, rejects
   `.`/`..` and NUL. Uploaded names are only ever used as display/citation
   labels, never as paths; normalising here means that stays true.
2. extension allow-list (`.pdf/.docx/.txt/.md`);
3. per-file byte limit from `settings.max_upload_mb`;
4. magic-byte check (`%PDF-`, `PK\x03\x04`) for the binary formats, and a NUL
   scan for the text formats, so a mislabelled file is rejected before any
   parser touches it;
5. empty-file rejection.

Rejections raise `UploadRejected` (a `ValueError`, so existing handlers keep
working) with a message safe to show a user. A rejected batch is validated
before anything mutates the index.

**Regression tests:** `test_safe_filename_strips_traversal`,
`test_validate_upload_rejects_disallowed_extension`,
`test_validate_upload_enforces_size_limit`,
`test_validate_upload_rejects_renamed_binary`,
`test_validate_upload_accepts_real_formats`,
`test_validate_upload_rejects_empty_file`,
`test_add_documents_enforces_limits`.

**Note on temporary files.** DocMind writes no temporary files: parsing works
from `io.BytesIO` throughout. The only disk write is the index save in
`src/vectorstore.py`. There is therefore no temp-file cleanup path to fail.

---

## DM-09 — Unbounded upload buffering (Medium)

**Where:** `api/main.py`, `/upload`.

`data = await f.read()` read the whole file into memory and *then* compared its
length against the limit, so a multi-gigabyte upload was fully buffered before
being rejected. Replaced with a bounded read loop that aborts with HTTP 413 as
soon as the limit is passed, plus a `MAX_UPLOAD_FILES` cap per request.

**Regression test:**
`test_api_upload_rejects_oversize_without_buffering_whole_file` — drives an
endless upload stream and asserts the handler aborts within one block of the
limit instead of reading indefinitely.

---

## DM-10 — Wildcard CORS (Medium)

`api/main.py` used `allow_origins=["*"]` with `allow_methods=["*"]` on a service
with no authentication, so any page open in a visitor's browser could drive it.
Now driven by `API_CORS_ORIGINS` (default `http://localhost:8501,http://localhost:3000`),
with `allow_credentials=False` and explicit method/header lists.

**Regression test:** `test_api_cors_is_not_wildcard`.

---

## DM-11 / DM-13 / DM-15 / DM-16 — Information disclosure and robustness (Low)

- **DM-11:** `/ask` and `/ask/stream` logged the full question text, which can
  quote private documents. They now log its length.
- **DM-13:** `"." in f.filename` raised `AttributeError` when a client omitted
  the filename. Now routed through `safe_filename()` and answered with HTTP 400.
- **DM-15:** the UI printed `str(exc)` in the main flow. Provider errors can
  carry the endpoint and response body, so the message is now a short, plain
  explanation, with `type(exc).__name__` plus a truncated message behind a
  collapsed "Technical detail" panel.
- **DM-16:** the Settings page reports *whether* a model is configured and the
  model name, never the key. Asserted by
  `test_settings_never_expose_the_api_key_in_ui_surfaces`, which fails if
  `api_key` appears anywhere in the UI module.

**Regression tests:** `test_provider_failure_message_contains_no_secret`,
`test_condense_query_swallows_provider_errors`.

---

## DM-12 — No authentication (High) — mitigated, not fixed

**This is the one finding that is not closed, and it cannot be closed without
adding real accounts.**

`MetaFilter` is a retrieval predicate, not an access-control list. The `owner`
field added for DM-01 is assigned by the server to a browser session; there is
no login, no verified identity, and no credential behind it. Specifically:

- a session id is the only thing separating two visitors' documents;
- anyone who obtains a session's state obtains its scope;
- there is no audit trail, no revocation and no encryption at rest;
- the index lives in one process's memory, readable by anything with access to
  that process or its host.

**What this means.** DocMind is safe to run as a **public single-user demo**,
where the honest expectation is that uploads are not confidential. It is **not**
safe for confidential or multi-tenant use. The UI states this on the empty state
and the Settings page rather than hiding it.

**What closing it would require:** authenticated sessions, per-owner persistent
storage, `MetaFilter(owners=...)` derived from the verified identity rather than
a session id, encryption at rest, and an audit log. The retrieval layer already
enforces the filter in the right place, so the work is authentication and
storage, not retrieval.

---

## DM-14 — Navigation to placeholder pages (Low)

The sidebar offered Knowledge Base, Collections and Playground; two routed to a
"on the roadmap" placeholder. Navigation is now Ask / Documents / Technical
details / Settings, all of which render live data.

---

## DM-17 — Unauthenticated API writes to the shared index (Medium)

`/upload` called `add_documents(payloads)` with the default `persist=True` and
the default public owner, so an exposed instance of the API could permanently
add documents that every future visitor of the Streamlit app would retrieve.
The endpoint now passes `persist=False`: API uploads are indexed in memory for
that process only. Building a persistent index is the CLI's job
(`python -m src.ingest`).

**Regression test:** `test_api_upload_does_not_persist_to_shared_index`.

---

## UI findings (from browser-based QA)

### UI-01 — Document filter rendered no options (Medium, **Reproduced**)

The filter used `st.multiselect`. In the pinned Streamlit (1.61.1) its
virtualised dropdown lays the option list out at `width: 0px`, so no option is
reachable: the listbox correctly reported `aria-setsize="5"` (4 documents +
"Select all") while rendering only the first row. **Confirmed upstream** by
disabling this app's entire stylesheet and reproducing the same zero-width
dropdown — so it is not caused by the design system.

Replaced with one checkbox per document: plain DOM, keyboard reachable, real
labels, no virtualisation. `st.session_state.doc_filter` is a plain mirror of
the checkbox states so the scope survives navigation to pages where those
widgets do not exist. The duplicate "limit to this document" checkbox inside
each inspector was removed, so scope is set in exactly one place and the
displayed filter cannot disagree with the request.

**Verified in the running app:** ticking `wafer_defects.md` scoped a question
about `recall@k` (which is documented in a *different* file) to that document
only — all five returned sources were from `wafer_defects.md`, the trace read
`document scope = wafer_defects.md`, and the model correctly refused.

**Regression test:** `test_document_filter_avoids_multiselect`.

### UI-02 — Runtime theme toggle could not work (Medium)

A sidebar light/dark toggle was added, then removed. Streamlit paints its own
widget chrome from `.streamlit/config.toml`'s `base` and exposes no DOM hook or
CSS custom properties for its active theme, so a runtime toggle restyled only
the custom components: measured in light mode, the active nav item still had a
dark `rgb(35,34,70)` background and a secondary button rendered dark text on a
dark surface (contrast **1.09:1**).

Shipping a control that cannot do what it claims is worse than not shipping it.
The palette is now a deploy-time setting, `DOCMIND_THEME` (default `dark`), and
`test_app_theme_matches_streamlit_config_base` fails if it disagrees with
`config.toml`. Both palettes remain fully supported.

### UI-04 — Page title clipped by the app header (Medium, **Reproduced**)

Reported from a real browser: the page heading was sliced in half along its top
edge on every page.

**Cause.** Streamlit's app header is `position: absolute`, **56px** tall,
opaque, and at `z-index: 999990`. Streamlit's own `padding-top` on the main
block exists to clear it. The design system overrode that with a flat
`padding-top: 1.6rem` (24px), so content started 32px too high and the header
painted over it.

**Measured before the fix** (1280×800): header bottom `56px`, title top `39px`
— **17px of the heading covered**.

**Fix.** The header height is now a token (`--dm-header-h: 56px`) and the main
block derives its padding from it:
`padding-top: calc(var(--dm-header-h) + 1.25rem)`. The rule also targets
`[data-testid="stMainBlockContainer"]`, which is the element Streamlit actually
uses in this version.

**Measured after the fix:** padding `74.75px`, title top `90px`, a **34px clear
gap**, and an enumeration of every element inside the main block confirms
**none** intersects the header — on all four pages, at 1280px and at 375px.

**Regression test:** `test_main_content_clears_the_fixed_app_header` asserts the
padding is derived from the header token rather than a flat value. Verified to
fail when the old `1.6rem` is restored, so it is not a vacuous assertion.

### UI-03 — Contrast below WCAG AA (Low)

An automated contrast sweep of the rendered pages found small copy (12.5px —
normal-size for WCAG) failing AA:

| Token | Before | After |
|---|---|---|
| dark `faint` on surface | 3.99:1 | 5.12:1 |
| light `faint` on white | 2.56:1 | 5.36:1 |
| light `muted` on alt surface | 4.34:1 | 5.12:1 |

`test_body_and_small_text_meet_aa`, `test_text_on_accent_fill_meets_aa` and
`test_status_colours_are_legible_on_their_own_soft_backgrounds` now assert the
whole palette in both modes, so a future colour tweak cannot silently regress it.

The flow chart also forced a 540px-wide side-scrolling box on phones; it now has
a vertical list rendering below 720px with the same six steps.

---

## Reproducing the verification

```bash
python -m pytest -q                 # full suite
python -m ruff check .              # lint
python -m pytest tests/test_security.py tests/test_ui_theme.py -q
python -m pytest tests/test_eval.py::test_retrieval_quality_gate_full_set -q
python -m streamlit run ui/streamlit_app.py
```

The retrieval quality gate asserts `recall@5 >= 0.9` on the labelled evaluation
set and was not relaxed during this work.

## What was verified in a browser, and what was not

Verified against the running app at `http://localhost:8512` by inspecting the
live DOM (computed styles, geometry, accessibility tree, rendered text):

- empty state and the six-step flow chart;
- a real question, streamed answer, inline citations, source passages;
- citation-validation badges in both the passing and the no-citation case;
- the insufficient-evidence refusal;
- the document filter changing retrieval scope, confirmed in the trace;
- the per-query trace values;
- all four pages, the sidebar, conversation history;
- contrast and layout sweeps at 1280px and 375px.

**Not verified:** screenshot capture in the available tooling returned blank
images for all but one attempt, so the layout evidence in this document comes
from DOM measurement rather than from inspecting rendered images. UI upload was
not driven through the browser's file picker; the upload path is covered by
`tests/test_security.py` and `tests/test_pipeline.py` instead.
