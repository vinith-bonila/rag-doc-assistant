"""DocMind — Streamlit front-end for the RAG document assistant.

This is the single deployable app for Streamlit Community Cloud. It runs the RAG
pipeline **in-process** (no separate backend to keep alive), so the whole thing
— upload, streaming chat, citations, evaluation — works from one Streamlit
interface within the platform's free RAM.

Memory notes:
  * The pipeline (and its models) is built once via ``st.cache_resource`` and
    reused across reruns and users — Streamlit reruns the script top-to-bottom
    on every interaction, so without this the models would reload constantly.
  * Models are lazy-loaded inside the pipeline; the single embedder instance is
    reused for indexing, retrieval and the embedding-based eval metrics.

The FastAPI service in ``api/`` is preserved for local/API use and is not needed
here — see the README.
"""
from __future__ import annotations

import os
import sys

# Streamlit puts THIS file's directory (ui/) on sys.path, not the repo root, so
# make the repo root importable for `src`, `config`, and `eval`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="DocMind — RAG Assistant", page_icon="📄", layout="wide")

# --- secrets bridge -----------------------------------------------------------
# Copy Streamlit secrets into the environment BEFORE importing config/pipeline,
# because config.py reads settings from env at import time. On Streamlit Cloud
# you set these under the app's "Secrets"; locally they come from a .env.
_SECRET_KEYS = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL",
                "EMBED_MODEL", "RERANKER_MODEL", "HYBRID")


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


# --- cached pipeline ----------------------------------------------------------
@st.cache_resource(show_spinner="Loading models and index (first run only)…")
def load_pipeline():
    # Imported here (not at module top) so the secrets bridge above has already
    # populated the environment that config.py reads.
    from src.pipeline import RAGPipeline
    return RAGPipeline()


def status(pipe) -> dict:
    return {
        "chunks": len(pipe.store.chunks),
        "hybrid": _settings().hybrid,
        "llm": _settings().has_llm,
    }


def _settings():
    from config import settings
    return settings


def render_citations(sources) -> None:
    st.caption("Sources — click to open the exact retrieved passage")
    for i, s in enumerate(sources, start=1):
        page = f" · p.{s.page}" if s.page is not None else ""
        label = f"[{i}] {s.source}{page}  ·  score {s.score:.2f}"
        with st.expander(label):
            st.write(s.text)


# --- sidebar ------------------------------------------------------------------
st.sidebar.title("📄 DocMind")
st.sidebar.caption("Hybrid retrieval · cross-encoder rerank · grounded, cited, streaming answers")

pipe = load_pipeline()
stat = status(pipe)
st.sidebar.success(
    f"Ready · {stat['chunks']} chunks\n\n"
    f"hybrid: {'on' if stat['hybrid'] else 'off'} · "
    f"LLM: {'on' if stat['llm'] else 'retrieval-only'}"
)
if not stat["llm"]:
    st.sidebar.info("No LLM key set — answers show the top retrieved passage. "
                    "Add `LLM_API_KEY` in Secrets to enable grounded generation.")

st.sidebar.subheader("Add documents")
uploaded = st.sidebar.file_uploader(
    "PDF, DOCX, TXT, or Markdown",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)
if st.sidebar.button("Index uploaded files", disabled=not uploaded):
    with st.sidebar.status("Indexing…"):
        try:
            payloads = [(f.name, f.getvalue()) for f in uploaded]
            result = pipe.add_documents(payloads)
            added = sum(f["chunks_added"] for f in result["files"])
            st.sidebar.success(f"Added {added} chunks · {result['total_chunks']} total")
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(str(exc))


# --- main ---------------------------------------------------------------------
st.title("DocMind")
st.caption("Ask questions across your documents and get streaming answers grounded in — and citing — the source passages.")

tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation"])

with tab_chat:
    if "history" not in st.session_state:
        st.session_state.history = []  # list of (question, answer, sources)

    for q, a, srcs in st.session_state.history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
            if srcs:
                render_citations(srcs)

    question = st.chat_input("Ask a question about your documents…")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            try:
                # answer_stream returns citations immediately + a token generator.
                sources, token_gen = pipe.answer_stream(question)
                answer = st.write_stream(token_gen)
                if sources:
                    render_citations(sources)
                st.session_state.history.append((question, answer, sources))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Generation failed: {exc}")

with tab_eval:
    st.subheader("Retrieval & answer quality")
    st.caption(
        "Each metric isolates a different failure mode — retrieval vs. reranking "
        "vs. generation — on a small labelled evaluation set."
    )
    if st.button("Run evaluation"):
        with st.status("Running the evaluation harness (calls the LLM per question)…"):
            try:
                from eval.evaluate import run_eval
                st.session_state.eval = run_eval(pipe)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Eval failed: {exc}")

    report = st.session_state.get("eval")
    if report:
        metrics = report["metrics"]
        cols = st.columns(len(metrics))
        for col, m in zip(cols, metrics):
            val = m["value"]
            col.metric(m["name"], "—" if val != val else f"{val:.2f}")  # val!=val => NaN
        chart_df = pd.DataFrame(
            {"score": [m["value"] if m["value"] == m["value"] else 0.0 for m in metrics]},
            index=[m["name"] for m in metrics],
        )
        st.bar_chart(chart_df, height=280, y_label="score (0–1, higher is better)")
        st.caption(
            f"Evaluated on {report['n_questions']} questions · "
            f"faithfulness via {'LLM-as-judge' if report['llm_judge'] else 'embedding fallback'}"
        )
        st.divider()
        for m in metrics:
            st.markdown(f"**{m['name']}** — {m['explanation']}")
    else:
        st.info("Click **Run evaluation** to compute recall@k, MRR, faithfulness and answer relevance.")
