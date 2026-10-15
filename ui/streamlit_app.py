"""DocMind — AI document workspace (Streamlit UI).

A presentation layer over the unchanged RAG backend: upload, parsing, indexing,
hybrid retrieval, cross-encoder reranking, grounded streaming generation,
citations, conversation memory, document scoping and evaluation all live in
``src/`` and ``eval/``.

Design intent: the default path is **upload -> ask -> read -> check the source**,
with no retrieval jargon in the way. Everything an engineer wants (retrieval
metadata, validation detail, per-query trace, evaluation) is one expander away
rather than spread across the main surface.

Two rules this file keeps:

* Nothing displayed is invented. Counts come from the live pipeline and the
  per-query ``RAGTrace``; a number we don't actually have is not shown.
* Every interpolation into markup goes through ``theme.esc`` — document names
  are attacker-controlled (docs/SECURITY_AUDIT.md DM-02).

Crafted by Vinith Bonila.
"""
from __future__ import annotations

import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

import theme  # ui/theme.py (ui/ is on sys.path when Streamlit runs this file)
from theme import esc
from src.trace import RAGTrace            # light modules (no torch/config import)
from src.validation import validate_citations

st.set_page_config(page_title="DocMind — AI document workspace", page_icon="◆",
                   layout="wide", initial_sidebar_state="expanded")

MAX_HISTORY_TURNS = 25
MAX_CONVERSATIONS = 12
SUPPORTED_LABEL = "PDF, DOCX, TXT, MD"

# Only secrets the app actually reads are bridged into the environment.
_SECRET_KEYS = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_TIMEOUT_S",
                "EMBED_MODEL", "RERANKER_MODEL", "HYBRID")

STARTER_PROMPTS = [
    "Summarize the key points.",
    "Explain the main concepts simply.",
    "Compare the important findings.",
    "What evidence supports the main conclusion?",
]


def _bridge_secrets() -> None:
    """Copy Streamlit Cloud secrets into the environment for ``config.py``."""
    try:
        secrets = st.secrets
    except Exception:  # noqa: BLE001 — no secrets file locally is normal
        return
    for key in _SECRET_KEYS:
        try:
            if key in secrets and secrets[key]:
                os.environ.setdefault(key, str(secrets[key]))
        except Exception:  # noqa: BLE001
            continue


_bridge_secrets()


@st.cache_resource(show_spinner="Setting up DocMind (first run only)…")
def load_pipeline():
    from src.pipeline import RAGPipeline
    return RAGPipeline()


def _settings():
    from config import settings
    return settings


def M(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


# ============================================================================
# SESSION STATE
# ============================================================================
def _init_state() -> None:
    # The owner id isolates this browser session's uploads from every other
    # session sharing the process (docs/SECURITY_AUDIT.md DM-01). It is an
    # isolation scope, not an authenticated identity.
    st.session_state.setdefault("owner", "s-" + uuid.uuid4().hex[:12])
    st.session_state.setdefault("page", "Chat")
    st.session_state.setdefault("conversations", {})   # id -> {title, turns}
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("doc_filter", [])

    if st.session_state.conversation_id is None:
        new_conversation()


def new_conversation() -> None:
    cid = uuid.uuid4().hex[:8]
    st.session_state.conversations[cid] = {"title": "New conversation", "turns": []}
    st.session_state.conversation_id = cid
    # Keep the sidebar list (and memory) bounded.
    extra = list(st.session_state.conversations)[:-MAX_CONVERSATIONS]
    for old in extra:
        st.session_state.conversations.pop(old, None)


def conversation() -> dict:
    return st.session_state.conversations[st.session_state.conversation_id]


def turns() -> list:
    return conversation()["turns"]


_init_state()

# The palette is a deploy-time setting (DOCMIND_THEME), not a runtime toggle:
# Streamlit paints its own widget chrome from .streamlit/config.toml's `base`
# and cannot be re-themed at runtime, so a runtime switch would leave native
# widgets in the other palette (docs/SECURITY_AUDIT.md UI-02).
UI_MODE = _settings().ui_theme
M(theme.css(UI_MODE))
FLOW_DIAGRAM = theme.flow_diagram(UI_MODE)
pipe = load_pipeline()
s = _settings()
OWNER = st.session_state.owner

# Documents this session may see: the bundled sample corpus plus its own
# uploads. Another session's upload is never listed here.
visible_docs: dict = pipe.visible_documents(owner=OWNER)
visible_chunks = sum(visible_docs.values())

# The document filter is the single source of truth for scope, and the exact
# value sent to retrieval — so the UI can never show a filter that isn't applied.
active_filter = [d for d in st.session_state.doc_filter if d in visible_docs]
scope_sources = set(active_filter) or None


# ============================================================================
# SIDEBAR
# ============================================================================
def render_sidebar() -> None:
    with st.sidebar:
        M('<div class="dm-brand"><span class="dm-brand-mark">D</span>'
          '<span class="dm-brand-name">DocMind</span></div>'
          '<div class="dm-brand-sub">AI document workspace</div>')

        if st.button("＋  New conversation", key="nav_new",
                     use_container_width=True):
            new_conversation()
            st.session_state.page = "Chat"
            st.rerun()

        M('<div class="dm-sidebar-heading">Workspace</div>')
        pages = [("Chat", "Ask"), ("Documents", "Documents"),
                 ("Technical", "Technical details"), ("Settings", "Settings")]
        for key, label in pages:
            active = st.session_state.page == key
            if st.button(label, key=f"nav_{key}", use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.page = key
                st.rerun()

        # Recent conversations — real session history, newest first.
        named = [(cid, c) for cid, c in st.session_state.conversations.items()
                 if c["turns"]]
        if named:
            M('<div class="dm-sidebar-heading">Recent</div>')
            for cid, convo in reversed(named):
                title = convo["title"]
                short = title if len(title) <= 34 else title[:33] + "…"
                if st.button(short, key=f"conv_{cid}", use_container_width=True,
                             type="primary" if cid == st.session_state.conversation_id
                             else "secondary"):
                    st.session_state.conversation_id = cid
                    st.session_state.page = "Chat"
                    st.rerun()

        M('<div class="dm-sidebar-heading">Status</div>')
        M(theme.status_pill("Answer model", s.has_llm))
        M(f'<div class="dm-metaline" style="margin-top:.5rem">'
          f'{len(visible_docs)} document(s) · {visible_chunks:,} passages indexed</div>')



# ============================================================================
# SHARED PIECES
# ============================================================================
def page_header(title: str, subtitle: str) -> None:
    M(f'<div class="dm-title">{esc(title)}</div>'
      f'<div class="dm-subtitle">{esc(subtitle)}</div>')


def scope_caption() -> None:
    """Always state, in words, which documents the next answer can use."""
    if active_filter:
        names = ", ".join(active_filter)
        M(theme.badge_row([theme.badge("neutral", "⌕", f"Answering from: {names}")]))
    elif visible_docs:
        M(theme.badge_row([theme.badge(
            "neutral", "⌕", f"Answering from all {len(visible_docs)} of your documents")]))


def render_uploader(key: str) -> None:
    limit = s.max_upload_mb
    files = st.file_uploader(
        f"Add documents — {SUPPORTED_LABEL}, up to {limit} MB each",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True, key=f"uploader_{key}")
    if st.button("Upload and index", key=f"upload_btn_{key}", type="primary",
                 disabled=not files, use_container_width=True):
        with st.status("Reading and indexing your documents…", expanded=False) as box:
            try:
                payloads = [(f.name, f.getvalue()) for f in files]
                result = pipe.add_documents(payloads, owner=OWNER)
                added = sum(f["chunks_added"] for f in result["files"])
                unchanged = [f["filename"] for f in result["files"]
                             if f["status"] == "unchanged"]
                empty = [f["filename"] for f in result["files"]
                         if f["status"] == "no-text"]
                box.update(label=f"Indexed {added} new passage(s).", state="complete")
                if unchanged:
                    st.caption("Already indexed, unchanged: " + ", ".join(unchanged))
                if empty:
                    st.warning("No readable text found in: " + ", ".join(empty)
                               + ". Scanned/image-only files need OCR, which DocMind "
                                 "does not do.")
                time.sleep(0.3)
                st.rerun()
            except ValueError as exc:
                # Upload validation failures are the user's to fix, so show them.
                box.update(label="Upload rejected", state="error")
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                box.update(label="Indexing failed", state="error")
                render_error(exc)


# Widget-state keys for the per-document filter checkboxes.
def _pick_key(name: str) -> str:
    return f"docpick::{name}"


def render_doc_filter(key: str) -> None:
    """The document filter. Its value *is* what retrieval receives.

    Deliberately checkboxes rather than ``st.multiselect``: in Streamlit 1.61
    the multiselect's virtualised dropdown lays its option list out at zero
    width, so the options are unreachable (reproduced with this app's CSS
    disabled, so it is upstream — see docs/SECURITY_AUDIT.md UI-01). Checkboxes
    are plain DOM, keyboard-reachable, and carry real labels.

    ``st.session_state.doc_filter`` is a plain mirror of the checkbox states, so
    the scope survives navigating to another page, where these widgets do not
    exist and Streamlit discards their state.
    """
    if len(visible_docs) <= 1:
        return
    st.caption("Tick the documents to search. All of them are searched when "
               "nothing is ticked.")
    chosen = []
    for name in sorted(visible_docs):
        if st.checkbox(name, value=name in st.session_state.doc_filter,
                       key=_pick_key(name)):
            chosen.append(name)
    if chosen != st.session_state.doc_filter:
        st.session_state.doc_filter = chosen
        st.rerun()
    if chosen and st.button("Search all documents again", key=f"reset_{key}"):
        for name in visible_docs:          # clear the widgets, then the mirror
            st.session_state[_pick_key(name)] = False
        st.session_state.doc_filter = []
        st.rerun()


def render_error(exc: Exception) -> None:
    """Show a useful, non-leaky failure message.

    The exception string can carry the provider endpoint and response body, so
    it goes in a collapsed panel rather than the main flow, and secrets are
    never interpolated anywhere (docs/SECURITY_AUDIT.md DM-15).
    """
    st.error("Something went wrong while answering. This is usually temporary — "
             "a rate limit or a slow model response. Try asking again.")
    with st.expander("Technical detail"):
        st.caption(f"{type(exc).__name__}: {str(exc)[:600]}")


# --- validation presentation --------------------------------------------------
def validation_badges(report, n_sources: int) -> list:
    """Precise, non-overstated status chips.

    What the code actually checks is **structural**: that each inline ``[n]``
    marker refers to a retrieved source. It does not check that the cited
    passage supports the claim, so no label here claims verified facts.
    """
    badges = []
    if n_sources:
        badges.append(theme.badge("neutral", "◆",
                                  f"{n_sources} source passage(s) linked"))
    if report.n_citations == 0:
        badges.append(theme.badge("warn", "!", "No citations in this answer"))
    elif report.ok:
        badges.append(theme.badge("ok", "✓", "Citation checks passed"))
    elif report.valid_sources:
        badges.append(theme.badge("warn", "!", "Some citations need review"))
    else:
        badges.append(theme.badge("err", "×", "Insufficient supporting evidence"))
    return badges


def render_answer_footer(answer: str, sources, trace: RAGTrace,
                         latency: float) -> None:
    """Citations, evidence, and the technical detail — in that order."""
    report = validate_citations(answer, len(sources))
    M(theme.badge_row(validation_badges(report, len(sources))))

    if report.n_citations and not report.ok:
        bad = ", ".join(f"[{m}]" for m in report.invalid_markers)
        st.caption(f"{bad} does not match any retrieved source and was shown as "
                   "[?]. Treat that part of the answer as unsupported.")
    elif report.n_citations == 0 and sources:
        st.caption("The answer did not cite specific passages. Check it against "
                   "the sources below before relying on it.")

    render_sources(sources)

    with st.expander("How this answer was found"):
        d = trace.as_dict()
        st.markdown(
            "1. **Your question was searched** against every passage of the "
            "documents in scope.\n"
            "2. **The closest passages were re-read and re-ranked** so the most "
            "relevant ones come first.\n"
            "3. **The answer was written from those passages only** and asked to "
            "cite them inline as `[1]`, `[2]`.\n"
            "4. **Each citation was checked** against the retrieved list — "
            "structurally, not for factual support.")
        st.caption(f"Took {latency:.1f}s end to end"
                   + (f" · {d['candidates_retrieved']} candidate passages considered"
                      if "candidates_retrieved" in d else ""))

    render_trace(trace, latency, len(sources))


def render_sources(sources) -> None:
    if not sources:
        return
    with st.expander(f"Sources used ({len(sources)})", expanded=False):
        for i, src in enumerate(sources, start=1):
            where = f" · page {src.page}" if src.page is not None else ""
            M(f'<div class="dm-h3">[{i}] {esc(src.source)}{esc(where)}</div>')
            st.caption("Passage as retrieved from your document:")
            st.text(src.text)
            if i < len(sources):
                M('<div class="dm-divider"></div>')


def render_trace(trace: RAGTrace, latency: float, n_sources: int) -> None:
    """Per-query trace as a compact stage table. Timings/counts only — the trace
    never carries secrets or whole documents."""
    with st.expander("Retrieval diagnostics and trace"):
        d = trace.as_dict()
        rows = [("end-to-end", f"{latency * 1000:.0f} ms")]
        rows += [(f"stage · {k}", f"{v:.0f} ms") for k, v in d.get("stages_ms", {}).items()]
        if "candidates_retrieved" in d:
            rows.append(("candidates retrieved", str(d["candidates_retrieved"])))
        rows.append(("passages sent to the model", str(n_sources)))
        ctx = d.get("context") or {}
        if ctx:
            rows.append(("context kept / in",
                         f"{ctx.get('kept')} / {ctx.get('input')}"))
            rows.append(("context size (estimate)",
                         f"~{ctx.get('context_tokens_est')} tokens"))
            if ctx.get("dropped_duplicates"):
                rows.append(("near-duplicates dropped", str(ctx["dropped_duplicates"])))
            if ctx.get("dropped_over_budget"):
                rows.append(("dropped over budget", str(ctx["dropped_over_budget"])))
        scope = d.get("scope") or {}
        if scope:
            docs = scope.get("documents")
            rows.append(("document scope",
                         ", ".join(docs) if isinstance(docs, list) else str(docs)))
            rows.append(("session-isolated", "yes" if scope.get("owner_scoped") else "no"))
        cites = d.get("citations") or {}
        if cites:
            rows.append(("citation check", str(cites.get("status"))))
        rows.append(("trace id", d.get("trace_id", "")))
        M(theme.kv_rows(rows))


# ============================================================================
# QUERY
# ============================================================================
def run_query(question: str) -> None:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            history = [(q, a) for q, a, *_ in turns()]
            trace = RAGTrace()
            t0 = time.time()
            with st.spinner("Searching your documents…"):
                sources, token_gen = pipe.answer_stream(
                    question, history=history, allowed_sources=scope_sources,
                    owner=OWNER, trace=trace)
            answer = st.write_stream(token_gen)
            latency = time.time() - t0
            render_answer_footer(answer, sources, trace, latency)

            convo = conversation()
            if not convo["turns"]:
                convo["title"] = question[:60]
            convo["turns"].append((question, answer, sources, trace.as_dict(), latency))
            del convo["turns"][:-MAX_HISTORY_TURNS]
        except Exception as exc:  # noqa: BLE001
            render_error(exc)
            if st.button("Try again", key="retry", type="primary"):
                queue(question)


def queue(text: str) -> None:
    st.session_state.pending_query = text
    st.rerun()


# ============================================================================
# PAGE: CHAT
# ============================================================================
def render_empty_state() -> None:
    """First-run guidance: what this is, what it takes, what happens next."""
    M('<div class="dm-empty">'
      '<h2>Start by adding a document</h2>'
      '<div class="dm-help">DocMind answers questions about documents you give '
      'it — and shows you the passage behind every claim, so you can check it.'
      '</div>' + FLOW_DIAGRAM +
      '<div class="dm-help" style="margin-top:.6rem">'
      f'<b>Supported files:</b> {esc(SUPPORTED_LABEL)} · up to '
      f'{esc(s.max_upload_mb)} MB each. Scanned or image-only PDFs need OCR, '
      'which DocMind does not do.</div>'
      '<ol class="dm-steps">'
      '<li>Upload a file and wait for indexing to finish.</li>'
      '<li>Ask a question in the box at the bottom of the page.</li>'
      '<li>Read the answer, then open <b>Sources used</b> to check it.</li>'
      '</ol>'
      '<div class="dm-privacy"><b>What happens to your file:</b> it is read into '
      'memory in this app, split into passages and indexed for search. Only this '
      'browser session can retrieve or read your uploads, and they are discarded '
      'when the app restarts. Your question and the matching passages are sent to '
      'the configured answer model. This is a public demo without user accounts — '
      'do not upload confidential material.</div>'
      '</div>')


def page_chat() -> None:
    queued = st.session_state.pop("pending_query", None)
    typed = st.chat_input("Ask a question about your documents…")
    question = queued or typed
    history = turns()

    page_header("Ask your documents",
                "Ask a question in plain English. Every answer shows the passages "
                "it came from.")

    top_left, top_right = st.columns([3, 1], gap="medium")
    with top_left:
        scope_caption()
    with top_right:
        if st.button("Manage documents", key="goto_docs", use_container_width=True):
            st.session_state.page = "Documents"
            st.rerun()

    if not visible_docs:
        render_empty_state()
        with st.container(border=True):
            render_uploader("empty")
        return

    if not history:
        with st.expander("New here? How DocMind works", expanded=False):
            M(FLOW_DIAGRAM)
            st.caption("DocMind only answers from the documents in scope. If they "
                       "don't contain the answer, it says so instead of guessing.")

    # transcript
    for turn in history:
        tq, ta, tsrc, ttrace, tlat = turn
        with st.chat_message("user"):
            st.write(tq)
        with st.chat_message("assistant"):
            st.markdown(ta)
            replay = RAGTrace()
            replay.trace_id = ttrace.get("trace_id", "")
            replay.stages = ttrace.get("stages_ms", {})
            replay.meta = {k: v for k, v in ttrace.items()
                           if k not in ("trace_id", "stages_ms", "total_ms")}
            render_answer_footer(ta, tsrc, replay, tlat)

    if not history and not question:
        M('<div class="dm-label" style="margin-top:1.2rem">Try one of these</div>')
        cols = st.columns(2)
        for i, prompt in enumerate(STARTER_PROMPTS):
            if cols[i % 2].button(prompt, key=f"starter_{i}",
                                  use_container_width=True):
                queue(prompt)

    if question:
        run_query(question)


# ============================================================================
# PAGE: DOCUMENTS
# ============================================================================
def page_documents() -> None:
    page_header("Documents",
                "The documents DocMind can search for you. Uploads stay private "
                "to this browser session.")

    left, right = st.columns([1.55, 1], gap="large")
    with right:
        with st.container(border=True):
            M('<div class="dm-h3">Add documents</div>')
            render_uploader("docs")
        with st.container(border=True):
            M('<div class="dm-h3">Scope</div>')
            if len(visible_docs) <= 1:
                st.caption("Add a second document to filter by document.")
            render_doc_filter("docs")

    with left:
        if not visible_docs:
            st.info("No documents yet. Upload one on the right to get started.")
            return

        c1, c2 = st.columns(2)
        c1.markdown(theme.metric(str(len(visible_docs)), "Documents"),
                    unsafe_allow_html=True)
        c2.markdown(theme.metric(f"{visible_chunks:,}", "Searchable passages"),
                    unsafe_allow_html=True)
        M('<div style="height:.9rem"></div>')

        for name, count in sorted(visible_docs.items(), key=lambda kv: kv[0].lower()):
            chunks = pipe.chunks_for(name, owner=OWNER)
            meta = (chunks[0].meta or {}) if chunks else {}
            pages = sorted({c.page for c in chunks if c.page is not None})
            bits = [meta.get("doc_type", "—").upper() or "—",
                    f"{count} passage{'s' if count != 1 else ''}"]
            if pages:
                bits.append(f"{len(pages)} page{'s' if len(pages) != 1 else ''}")
            bits.append("indexed" if meta.get("ingestion_status") == "indexed"
                        else "unknown status")
            bits.append("your upload" if meta.get("owner") == OWNER
                        else "bundled sample")
            M(theme.doc_row(name, " · ".join(bits)))

            with st.expander(f"Inspect passages — {name}"):
                # Scope is set in one place (the Scope panel) so the displayed
                # filter can never disagree with what retrieval receives.
                for c in chunks[:25]:
                    where = f"page {c.page} · " if c.page is not None else ""
                    st.caption(f"{where}passage #{c.chunk_id}")
                    st.text(c.text)
                if len(chunks) > 25:
                    st.caption(f"… and {len(chunks) - 25} more passages")


# ============================================================================
# PAGE: TECHNICAL
# ============================================================================
def page_technical() -> None:
    page_header("Technical details",
                "How retrieval is configured, and how it scores on a labelled "
                "evaluation set. You never need this page to use DocMind.")

    with st.expander("Retrieval configuration", expanded=True):
        M(theme.kv_rows([
            ("Embedding model", s.embed_model),
            ("Reranker (cross-encoder)", s.reranker_model),
            ("Hybrid search (dense + BM25)", "on" if s.hybrid else "off"),
            ("Rank-fusion constant (RRF k)", str(s.rrf_k)),
            ("Candidates retrieved per query", str(s.retrieve_k)),
            ("Passages kept after reranking", str(s.rerank_top_n)),
            ("Chunk size / overlap", f"{s.chunk_size} / {s.chunk_overlap} chars"),
            ("Chunking strategy", s.chunk_strategy),
            ("Context token budget", str(s.context_token_budget)),
            ("Near-duplicate threshold", str(s.dedup_jaccard)),
            ("Answer model", s.llm_model if s.has_llm else "not configured"),
        ]))

    with st.expander("What is and isn't verified", expanded=True):
        st.markdown(
            "- **Checked automatically:** every `[n]` marker in an answer points "
            "to a passage that was actually retrieved. Markers that don't are "
            "rewritten to `[?]`.\n"
            "- **Not checked automatically:** whether the cited passage really "
            "supports the sentence it is attached to. That needs a human, or the "
            "offline faithfulness metric below.\n"
            "- **No user accounts.** Uploads are separated per browser session, "
            "which keeps one visitor's files out of another's answers. That is "
            "isolation, not authentication — see `docs/SECURITY_AUDIT.md`.")

    M('<div class="dm-h2" style="margin-top:1.4rem">Evaluation</div>'
      '<div class="dm-help">Runs the labelled evaluation set through the live '
      'pipeline. Each metric isolates a different failure mode — retrieval, '
      'ranking, or generation. This calls the answer model once per question.'
      '</div><div style="height:.7rem"></div>')

    if st.button("Run evaluation", type="primary", key="run_eval"):
        with st.status("Running the evaluation harness…", expanded=False) as box:
            try:
                from eval.evaluate import run_eval
                st.session_state.eval = run_eval(pipe)
                box.update(label="Evaluation complete.", state="complete")
            except Exception as exc:  # noqa: BLE001
                box.update(label="Evaluation failed", state="error")
                render_error(exc)

    report = st.session_state.get("eval")
    if not report:
        st.caption("No results yet.")
        return

    metrics = report["metrics"]
    for row_start in range(0, len(metrics), 4):
        cols = st.columns(4)
        for col, m in zip(cols, metrics[row_start:row_start + 4]):
            v = m["value"]
            col.markdown(theme.metric("—" if v != v else f"{v:.2f}", m["name"]),
                         unsafe_allow_html=True)
    st.caption(f"{report['n_questions']} labelled questions · faithfulness via "
               f"{'LLM-as-judge' if report['llm_judge'] else 'embedding fallback'}")
    with st.expander("What each metric means"):
        for m in metrics:
            st.markdown(f"**{m['name']}** — {m['explanation']}")


# ============================================================================
# PAGE: SETTINGS
# ============================================================================
def page_settings() -> None:
    page_header("Settings",
                "Configuration is read from environment variables or Streamlit "
                "secrets at startup. Change them there, then restart the app.")

    M(theme.kv_rows([
        ("Answer model", s.llm_model if s.has_llm else "not configured"),
        ("Model endpoint configured", "yes" if s.has_llm else "no"),
        ("Request timeout", f"{s.llm_timeout_s:.0f}s (retries: {s.llm_max_retries})"),
        ("Max upload size", f"{s.max_upload_mb} MB per file"),
        ("Supported file types", SUPPORTED_LABEL),
        ("Citation checking", "on" if s.validate_citations else "off"),
        ("PDF table extraction", "on" if s.parse_tables else "off"),
        ("Appearance", f"{UI_MODE} (DOCMIND_THEME)"),
    ]))
    # The API key itself is never displayed, logged or echoed — only whether one
    # is present (docs/SECURITY_AUDIT.md DM-16).
    st.caption("API keys are never displayed here, written to the page, or "
               "included in error messages.")

    if not s.has_llm:
        st.warning("No answer model is configured, so DocMind will show the "
                   "closest retrieved passage instead of a written answer. Set "
                   "`LLM_API_KEY` to enable generated answers.")

    M('<div class="dm-h2" style="margin-top:1.4rem">Privacy and data handling</div>')
    st.markdown(
        "- Uploads are held **in memory** for this app process and indexed for "
        "search. They are not written into the shared on-disk index.\n"
        "- Only **your browser session** can list, read or retrieve your "
        "uploads.\n"
        "- Your question and the matching passages are sent to the configured "
        "answer model. Nothing else is sent anywhere.\n"
        "- There are **no user accounts**. Do not upload confidential material "
        "to a public deployment.")

    M('<div class="dm-h2" style="margin-top:1.4rem">This session</div>')
    if st.button("Clear my conversations", key="clear_convos"):
        st.session_state.conversations = {}
        st.session_state.conversation_id = None
        _init_state()
        st.rerun()


# ============================================================================
# ROUTER
# ============================================================================
render_sidebar()

{"Chat": page_chat, "Documents": page_documents,
 "Technical": page_technical, "Settings": page_settings}[st.session_state.page]()

M('<div class="dm-divider"></div>'
  '<div class="dm-metaline" style="text-align:center">DocMind · answers grounded '
  'in your documents, with the evidence attached · built by Vinith Bonila</div>')
