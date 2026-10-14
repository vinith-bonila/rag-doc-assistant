"""DocMind — a premium, spatial RAG "knowledge OS" front-end (Streamlit).

A pure front-end/UX layer around the real RAG backend: upload, parsing, indexing,
hybrid retrieval, cross-encoder reranking, grounded streaming generation,
citations, conversation memory, document scoping and evaluation are all unchanged.
No metrics or documents are faked — every value shown is derived from the live
pipeline. All motion is in-app CSS (transforms + opacity), GPU-friendly, and does
not reload on Streamlit reruns.

Crafted by Vinith Bonila.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

import theme  # ui/theme.py (ui/ is on sys.path when Streamlit runs this file)
from src.trace import RAGTrace            # light modules (no torch/config import)
from src.validation import validate_citations

st.set_page_config(page_title="DocMind — AI Document Intelligence", page_icon="◆",
                   layout="wide", initial_sidebar_state="expanded")

MAX_HISTORY_TURNS = 25
_SECRET_KEYS = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL",
                "EMBED_MODEL", "RERANKER_MODEL", "HYBRID")
_FILE_COLORS = {"pdf": "#D946EF", "docx": "#6366F1", "md": "#22D3EE",
                "txt": "#8B5CF6", "doc": "#6366F1"}


def _bridge_secrets() -> None:
    try:
        secrets = st.secrets
    except Exception:
        return
    for key in _SECRET_KEYS:
        try:
            if key in secrets and secrets[key]:
                os.environ.setdefault(key, str(secrets[key]))
        except Exception:
            continue


_bridge_secrets()


@st.cache_resource(show_spinner="Booting the knowledge engine (first run only)…")
def load_pipeline():
    from src.pipeline import RAGPipeline
    return RAGPipeline()


def _settings():
    from config import settings
    return settings


def _relevances(scores: list) -> list:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.99 for _ in scores]
    return [0.45 + 0.54 * (sc - lo) / (hi - lo) for sc in scores]


def M(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


# ============================================================================
# BOOT
# ============================================================================
M(theme.CSS)
pipe = load_pipeline()
s = _settings()
store_map: dict = {}
for c in pipe.store.chunks:
    store_map[c.source] = store_map.get(c.source, 0) + 1
uploaded = st.session_state.get("uploaded_sources", set()) & set(store_map)
# An explicit document filter (right panel) overrides the auto-scope.
_filter_sel = set(st.session_state.get("doc_filter") or []) & set(store_map)
scope_sources = _filter_sel or (uploaded or None)
st.session_state.setdefault("section", "Chat")
st.session_state.setdefault("history", [])
NAV = ["Chat", "Documents", "Knowledge Base", "Collections",
       "Analytics", "Playground", "Settings"]


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    M('<div class="dm-logo">DOC<span class="g">MIND</span></div>'
      '<div class="dm-logo-sub">RAG ASSISTANT</div>')
    M('<div class="dm-side-label">WORKSPACE</div>')
    for item in NAV:
        if item == st.session_state.section:
            M(f'<div class="dm-navactive"><span class="d"></span>{item}</div>')
        elif st.button(item, key=f"nav_{item}"):
            st.session_state.section = item
            st.rerun()

    M('<div class="dm-side-label">SYSTEM</div>')
    M(f'<div class="dm-engine"><div class="top">'
      f'<span class="name">RAG ENGINE</span>'
      f'<span class="dm-online"><span class="p"></span>ONLINE</span></div>'
      f'<div class="row">Hybrid Retrieval<b>{"ON" if s.hybrid else "OFF"}</b></div>'
      f'<div class="row">Cross-Encoder<b>ON</b></div>'
      f'<div class="row">LLM<b>{"ON" if s.has_llm else "OFF"}</b></div></div>')

    with st.expander("About · how it works"):
        st.markdown(
            "**DocMind** uses Retrieval-Augmented Generation to retrieve relevant "
            "information from your indexed documents *before* generating an answer.\n\n"
            "**Answers are grounded** in retrieved content and shown with source "
            "citations.\n\n"
            "*Like any AI system, verify answers against the cited sources for "
            "high-stakes decisions.*")

    M('<div style="height:.6rem"></div>'
      '<div class="dm-profile"><div class="dm-avatar"></div><div>'
      '<div class="who">Vinith Bonila</div><div class="plan">FREE · STREAMLIT CLOUD</div>'
      '</div></div>')


# ============================================================================
# SHARED PANEL RENDERERS
# ============================================================================
def render_upload_panel() -> None:
    M('<div class="dm-eyebrow">ADD DOCUMENTS</div>')
    files = st.file_uploader("Drop documents — PDF · DOCX · TXT · MD (max 20MB)",
                             type=["pdf", "docx", "txt", "md"],
                             accept_multiple_files=True, key="uploader")
    if st.button("Upload & Index", type="primary", disabled=not files,
                 use_container_width=True):
        anim = st.empty()
        with anim.container():
            M(theme.upload_animation())
        try:
            payloads = [(f.name, f.getvalue()) for f in files]
            result = pipe.add_documents(payloads)   # real work (animation plays meanwhile)
            added = sum(f["chunks_added"] for f in result["files"])
            st.session_state.setdefault("uploaded_sources", set()).update(
                f.name for f in files)
            anim.empty()
            M(f'<div class="dm-meta"><span class="ok">✓ DOCUMENT INDEXED</span>'
              f'<span><b>{added}</b> new chunks</span>'
              f'<span><b>{result["total_chunks"]}</b> total</span></div>')
            time.sleep(0.6)
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            anim.empty()
            render_error(exc)


def render_recent_docs(limit=6) -> None:
    M('<div class="dm-eyebrow">DOCUMENTS</div>')
    if not store_map:
        st.caption("No documents indexed yet.")
        return
    items = sorted(store_map.items(), key=lambda kv: (-kv[1], kv[0]))
    for name, n in (items[:limit] if limit else items):
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else "txt"
        color = _FILE_COLORS.get(ext, "#6366F1")
        mine = ' · <span class="yours">yours</span>' if name in uploaded else ""
        M(f'<div class="dm-doc"><div class="ic" style="background:{color}">{ext.upper()[:3]}</div>'
          f'<div><div class="fn">{name}</div>'
          f'<div class="mt"><span class="st"></span>{ext.upper()} · {n} chunks · Indexed{mine}</div>'
          f'</div></div>')
    if limit and len(items) > limit and st.button(f"View all {len(items)} →", key="viewall"):
        st.session_state.section = "Documents"
        st.rerun()


def render_stats() -> None:
    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="dm-stat"><div class="n">{len(store_map)}</div>'
                f'<div class="l">DOCUMENTS</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="dm-stat"><div class="n">{len(pipe.store.chunks):,}</div>'
                f'<div class="l">CHUNKS INDEXED</div></div>', unsafe_allow_html=True)


def render_doc_filter() -> None:
    """Metadata filter: restrict retrieval to chosen documents (rag.pdf p.8/10).
    The selection is read back at boot and enforced inside retrieval."""
    if len(store_map) <= 1:
        return
    M('<div class="dm-eyebrow">FILTER TO DOCUMENTS</div>')
    st.multiselect("Restrict answers to", options=sorted(store_map),
                   key="doc_filter", label_visibility="collapsed",
                   placeholder="All documents")


def render_error(exc: Exception) -> None:
    M('<div class="dm-answerlabel" style="color:var(--magenta)">SOMETHING INTERRUPTED '
      'THE KNOWLEDGE FLOW</div>')
    st.markdown("The engine hit a snag while composing your answer. This is usually "
                "transient (model/rate limit). You can ask again.")
    with st.expander("View details"):
        st.code(str(exc))


def render_meta(sources, retrieved, latency) -> None:
    used = len({x.source for x in sources})
    M(f'<div class="dm-meta"><span class="ok">✓ Grounded</span>'
      f'<span><b>{retrieved}</b> chunks retrieved</span>'
      f'<span><b>{len(sources)}</b> reranked</span>'
      f'<span><b>{used}</b> source{"s" if used != 1 else ""}</span>'
      f'<span><b>{latency:.1f}s</b></span></div>')


def render_sources(sources) -> None:
    if not sources:
        return
    rels = _relevances([x.score for x in sources])
    M('<div class="dm-srchead">SOURCES</div>')
    for i, (x, rel) in enumerate(zip(sources, rels), start=1):
        page = f"Page {x.page} · " if x.page is not None else ""
        with st.expander(f"{i:02d}   {x.source}    ·    {page}relevance {rel:.2f}"):
            M(f'<div class="dm-relbar"><i style="width:{rel*100:.0f}%"></i></div>')
            st.write(x.text)


def how_answer_found(retrieved, reranked) -> None:
    with st.expander("How this answer was found"):
        steps = [
            ("Query", "understood (follow-ups rewritten to a standalone query)"),
            ("Hybrid retrieval", f"dense FAISS + BM25, RRF-fused → {retrieved} candidates"),
            ("Cross-encoder rerank", f"{s.reranker_model.split('/')[-1]} → top {reranked}"),
            ("Grounded generation", f"{s.llm_model} — answers only from retrieved context"),
            ("Citation", "each claim linked to its source passage"),
        ]
        M("".join(
            f'<div style="display:flex;gap:.7rem;padding:.35rem 0;border-top:1px solid var(--line)">'
            f'<span style="font-family:var(--mono);color:var(--violet);font-size:.72rem">{i+1:02d}</span>'
            f'<div><b style="font-size:.85rem">{t}</b>'
            f'<div style="color:var(--muted);font-size:.76rem">{d}</div></div></div>'
            for i, (t, d) in enumerate(steps)))


def render_validation(rep) -> None:
    """Live citation-validation badge (rag.pdf p.14)."""
    if rep.n_citations == 0:
        return
    if rep.ok:
        M('<div class="dm-meta"><span class="ok">✓ Citations verified</span>'
          f'<span>{rep.n_citations} inline citation(s), all map to a source</span></div>')
    else:
        bad = ", ".join(f"[{m}]" for m in rep.invalid_markers)
        M('<div class="dm-meta"><span style="color:var(--magenta)">⚠ '
          f'{len(rep.invalid_markers)} citation(s) flagged</span>'
          f'<span>{bad} point(s) to no source</span></div>')


def render_trace(trace: RAGTrace, latency: float) -> None:
    with st.expander("Trace · observability"):
        d = trace.as_dict()
        rows = [("total (end-to-end)", f"{latency*1000:.0f} ms")]
        rows += [(k, f"{v:.0f} ms") for k, v in d.get("stages_ms", {}).items()]
        if "candidates_retrieved" in d:
            rows.append(("candidates retrieved", str(d["candidates_retrieved"])))
        ctx = d.get("context", {})
        if ctx:
            rows.append(("context (kept/in)", f"{ctx.get('kept')}/{ctx.get('input')}"
                                              f"  ·  ~{ctx.get('context_tokens_est')} tok"))
            if ctx.get("dropped_duplicates"):
                rows.append(("deduped", str(ctx["dropped_duplicates"])))
        rows.append(("trace id", d.get("trace_id", "")))
        M("".join(
            f'<div class="dm-engine" style="margin-bottom:.3rem"><div class="row" '
            f'style="border:none;margin:0;padding:.1rem 0">{k}<b>{v}</b></div></div>'
            for k, v in rows))


# ============================================================================
# QUERY
# ============================================================================
def run_query(question: str, workspace) -> None:
    with workspace:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            anim = st.empty()
            with anim.container():
                M(theme.rag_pipeline())
            try:
                turns = [(q, a) for q, a, *_ in st.session_state.history]
                trace = RAGTrace()
                t0 = time.time()
                sources, token_gen = pipe.answer_stream(
                    question, history=turns, allowed_sources=scope_sources, trace=trace)
                M('<div class="dm-answerlabel">AI ANSWER</div>')
                answer = st.write_stream(token_gen)
                latency = time.time() - t0
                anim.empty()
                scope_n = (sum(store_map[x] for x in scope_sources)
                           if scope_sources else len(pipe.store.chunks))
                retrieved = min(s.retrieve_k, scope_n)
                render_meta(sources, retrieved, latency)
                render_validation(validate_citations(answer, len(sources)))
                render_sources(sources)
                how_answer_found(retrieved, len(sources))
                render_trace(trace, latency)
                st.session_state.history.append(
                    (question, answer, sources, (retrieved, latency)))
                del st.session_state.history[:-MAX_HISTORY_TURNS]
            except Exception as exc:  # noqa: BLE001
                anim.empty()
                render_error(exc)


def queue(text: str) -> None:
    st.session_state.pending_query = text
    st.rerun()


# ============================================================================
# WELCOME (intro + hero + core)
# ============================================================================
def render_welcome() -> None:
    M('<div class="dm-intro">'
      '<h3>What is DocMind?</h3>'
      '<div class="sub">Your AI-powered document intelligence workspace.</div>'
      '<p>DocMind is a Retrieval-Augmented Generation (RAG) assistant that lets you '
      'upload your documents, retrieve the relevant information, and ask questions '
      'with grounded, citation-backed answers.</p>'
      '<div class="dm-flow">'
      '<span class="s">Documents</span><span class="a">→</span>'
      '<span class="s">Chunking</span><span class="a">→</span>'
      '<span class="s">Hybrid Retrieval</span><span class="a">→</span>'
      '<span class="s">Cross-Encoder</span><span class="a">→</span>'
      '<span class="s">LLM</span><span class="a">→</span>'
      '<span class="s">Grounded Answer</span><span class="a">→</span>'
      '<span class="s">Citations</span></div>'
      '<div class="dm-note">DocMind answers from your indexed documents, grounding '
      'responses in retrieved source content to reduce unsupported answers.</div>'
      '<div class="dm-builtfor">Built for &nbsp;<b>Research</b> · <b>Reports</b> · '
      '<b>Technical Documents</b> · <b>Policies</b> · <b>Knowledge Bases</b></div>'
      '</div>')

    hero, core = st.columns([1.05, 1], gap="medium")
    with hero:
        M('<div style="height:1rem"></div>'
          '<div class="dm-kicker">AI DOCUMENT INTELLIGENCE</div>'
          '<div class="dm-h1">Understand your<br><span class="g">documents.</span> Faster.</div>'
          '<div class="dm-lede">Upload your knowledge. Ask questions. Get grounded '
          'answers with citations.</div>'
          '<div class="dm-chips">'
          '<span class="dm-chip"><span class="d">◆</span> Grounded</span>'
          '<span class="dm-chip"><span class="d">◆</span> Cited</span>'
          '<span class="dm-chip"><span class="d">◆</span> Hybrid Retrieval</span>'
          '<span class="dm-chip"><span class="d">◆</span> Streaming</span></div>')
    with core:
        M(theme.knowledge_core())


def render_suggestions() -> None:
    M('<div style="height:.4rem"></div>'
      '<div class="dm-emptybig">Ask your knowledge base</div>'
      '<div class="dm-emptysub">Upload a document or ask a question to begin.</div>'
      '<div style="height:.6rem"></div>')
    cols = st.columns(2)
    for i, sug in enumerate([
            "Summarize this document", "What are the key findings?",
            "Explain it like I'm a beginner", "Find potential risks",
            "List action items", "Compare the sections"]):
        if cols[i % 2].button(sug, key=f"sug_{i}", use_container_width=True):
            queue(sug)


# ============================================================================
# SECTION: CHAT
# ============================================================================
def section_chat() -> None:
    queued = st.session_state.pop("pending_query", None)
    typed = st.chat_input("Ask a question about your documents…")
    q = queued or typed
    chatting = bool(st.session_state.history) or bool(q)

    left, right = st.columns([2.1, 1], gap="large")
    with left:
        if not chatting:
            render_welcome()

        # workspace tools -> queued intents
        M('<div class="dm-eyebrow" style="margin-top:.8rem">WORKSPACE TOOLS</div>')
        tools = {"Summarize": "Summarize this document",
                 "Compare": "Compare the main sections of this document",
                 "Translate": "Translate the key points into simple English",
                 "Find Risks": "What are the risks or concerns?",
                 "Action Items": "List the action items"}
        tcols = st.columns(len(tools))
        for col, (label, prompt) in zip(tcols, tools.items()):
            if col.button(label, key=f"tool_{label}"):
                queue(prompt)

        if scope_sources:
            M('<div class="dm-answerlabel" style="margin-top:.6rem">ANSWERING FROM · '
              + " · ".join(sorted(scope_sources)) + '</div>')

        # transcript
        for turn in st.session_state.history:
            tq, ta, tsrcs, *rest = turn
            with st.chat_message("user"):
                st.write(tq)
            with st.chat_message("assistant"):
                M('<div class="dm-answerlabel">AI ANSWER</div>')
                st.write(ta)
                if rest and rest[0]:
                    render_meta(tsrcs, rest[0][0], rest[0][1])
                render_sources(tsrcs)
                if rest and rest[0]:
                    how_answer_found(rest[0][0], len(tsrcs))

        if not chatting:
            render_suggestions()
        workspace = st.container()

    with right:
        with st.container(border=True):
            render_upload_panel()
            M('<div style="height:.5rem"></div>')
            render_doc_filter()
            M('<div style="height:.5rem"></div>')
            render_recent_docs(6)
            M('<div style="height:.5rem"></div>')
            render_stats()

    if q:
        run_query(q, workspace)


# ============================================================================
# SECTION: DOCUMENTS
# ============================================================================
def section_documents() -> None:
    M('<div class="dm-kicker">DOCUMENT LIBRARY</div>'
      '<div class="dm-h1" style="font-size:2rem">Documents</div>')
    left, right = st.columns([2.1, 1], gap="large")
    with right:
        with st.container(border=True):
            render_upload_panel()
            M('<div style="height:.5rem"></div>')
            render_stats()
    with left:
        if not store_map:
            st.info("No documents indexed yet — upload one to begin.")
            return
        M('<div class="dm-eyebrow">INDEXED DOCUMENTS · browse chunks</div>')
        for name, n in sorted(store_map.items(), key=lambda kv: (-kv[1], kv[0])):
            mine = " · yours" if name in uploaded else ""
            with st.expander(f"{name}   ·   {n} chunks · Indexed{mine}"):
                chunks = [c for c in pipe.store.chunks if c.source == name]
                for c in chunks[:40]:
                    pg = f"Page {c.page} · " if c.page is not None else ""
                    M(f'<div class="dm-srchead" style="margin:.3rem 0 .1rem">'
                      f'{pg}CHUNK #{c.chunk_id}</div>')
                    st.write(c.text)
                if len(chunks) > 40:
                    st.caption(f"… and {len(chunks) - 40} more chunks")


def section_knowledge_base() -> None:
    M('<div class="dm-kicker">KNOWLEDGE BASE</div>'
      '<div class="dm-h1" style="font-size:2rem">Index Overview</div>')
    render_stats()
    M('<div style="height:1rem"></div><div class="dm-eyebrow">RETRIEVAL ENGINE</div>')
    cfg = [("Embedding model", s.embed_model.split("/")[-1]),
           ("Reranker (cross-encoder)", s.reranker_model.split("/")[-1]),
           ("Hybrid (dense + BM25)", "ON" if s.hybrid else "OFF"),
           ("RRF constant", str(s.rrf_k)),
           ("Candidates retrieved (k)", str(s.retrieve_k)),
           ("Reranked to (top-n)", str(s.rerank_top_n)),
           ("Chunk size / overlap", f"{s.chunk_size} / {s.chunk_overlap}"),
           ("LLM", s.llm_model if s.has_llm else "not configured")]
    M("".join(f'<div class="dm-engine" style="margin-bottom:.4rem"><div class="row" '
              f'style="border:none;margin:0;padding:.15rem 0">{k}<b>{v}</b></div></div>'
              for k, v in cfg))


def section_analytics() -> None:
    M('<div class="dm-kicker">EVALUATION &amp; ANALYTICS</div>'
      '<div class="dm-h1" style="font-size:2rem">Retrieval &amp; Answer Quality</div>'
      '<div class="dm-lede">Each metric isolates a different failure mode — retrieval '
      'vs. reranking vs. generation — on a labelled evaluation set.</div>'
      '<div style="height:.8rem"></div>')
    if st.button("▶  Run evaluation", type="primary"):
        with st.status("Running the evaluation harness (calls the LLM per question)…"):
            try:
                from eval.evaluate import run_eval
                st.session_state.eval = run_eval(pipe)
            except Exception as exc:  # noqa: BLE001
                render_error(exc)
    report = st.session_state.get("eval")
    if report:
        metrics = report["metrics"]
        cols = st.columns(len(metrics))
        for col, m in zip(cols, metrics):
            v = m["value"]
            col.markdown(f'<div class="dm-stat"><div class="n">'
                         f'{"—" if v != v else f"{v:.2f}"}</div>'
                         f'<div class="l">{m["name"].upper()}</div></div>',
                         unsafe_allow_html=True)
        M('<div style="height:1rem"></div>')
        st.bar_chart(pd.DataFrame(
            {"score": [m["value"] if m["value"] == m["value"] else 0.0 for m in metrics]},
            index=[m["name"] for m in metrics]), height=280, color="#8B5CF6")
        st.caption(f"Evaluated on {report['n_questions']} questions · faithfulness via "
                   f"{'LLM-as-judge' if report['llm_judge'] else 'embedding fallback'}")
        st.divider()
        for m in metrics:
            st.markdown(f"**{m['name']}** — {m['explanation']}")
    else:
        st.info("Click **Run evaluation** to compute recall@k, MRR, faithfulness "
                "and answer relevance.")


def section_settings() -> None:
    M('<div class="dm-kicker">SETTINGS</div>'
      '<div class="dm-h1" style="font-size:2rem">Configuration</div>'
      '<div class="dm-lede">From <code>config.py</code> + environment/secrets. Change '
      'via environment variables or Streamlit secrets, then reboot.</div>'
      '<div style="height:.8rem"></div>')
    rows = [("EMBED_MODEL", s.embed_model), ("RERANKER_MODEL", s.reranker_model),
            ("HYBRID", "true" if s.hybrid else "false"), ("RRF_K", str(s.rrf_k)),
            ("RETRIEVE_K", str(s.retrieve_k)), ("RERANK_TOP_N", str(s.rerank_top_n)),
            ("CHUNK_STRATEGY", s.chunk_strategy),
            ("CHUNK_SIZE / OVERLAP", f"{s.chunk_size} / {s.chunk_overlap}"),
            ("LLM_MODEL", s.llm_model),
            ("LLM configured", "yes" if s.has_llm else "no (retrieval-only)")]
    st.dataframe(pd.DataFrame(rows, columns=["Setting", "Value"]),
                 use_container_width=True, hide_index=True)


def section_roadmap(name: str) -> None:
    M(f'<div class="dm-kicker">{name.upper()}</div>'
      f'<div class="dm-h1" style="font-size:2rem">{name}</div>')
    with st.container(border=True):
        M('<div class="dm-emptybig">On the roadmap</div>'
          '<div class="dm-emptysub">This module isn\'t part of the deployed build yet. '
          'Chat, Documents, Knowledge Base and Analytics all run on live data.</div>')


# ============================================================================
# ROUTER + FOOTER
# ============================================================================
sec = st.session_state.section
{"Chat": section_chat, "Documents": section_documents,
 "Knowledge Base": section_knowledge_base, "Analytics": section_analytics,
 "Settings": section_settings}.get(sec, lambda: section_roadmap(sec))()

M('<div style="height:2.4rem"></div>'
  '<div style="text-align:center;font-family:var(--mono);font-size:.66rem;'
  'letter-spacing:.14em;color:var(--muted2)">'
  'DOCMIND · Documents today. Insights tomorrow. &nbsp;·&nbsp; '
  'Crafted by <span style="color:var(--violet)">VINITH BONILA</span></div>')
