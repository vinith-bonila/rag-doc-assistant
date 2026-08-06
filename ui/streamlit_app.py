"""DocMind — Streamlit front-end for the RAG document assistant.

Talks to the FastAPI backend over HTTP:
  * uploads documents            -> POST /upload
  * streams grounded answers     -> POST /ask/stream  (Server-Sent Events)
  * runs the evaluation harness  -> POST /eval

Set the backend location with the API_URL environment variable
(default http://localhost:8000). When frontend and backend run in the same
process/host — e.g. one Hugging Face Space — the default is correct.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, Tuple

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 120

st.set_page_config(page_title="DocMind — RAG Assistant", page_icon="📄", layout="wide")


# --- backend helpers ----------------------------------------------------------
def get_health() -> Dict:
    try:
        return requests.get(f"{API_URL}/health", timeout=10).json()
    except requests.RequestException:
        return {}


def upload_files(files) -> Dict:
    payload = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
               for f in files]
    r = requests.post(f"{API_URL}/upload", files=payload, timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(r.json().get("detail", r.text))
    return r.json()


def stream_answer(question: str) -> Tuple[Iterator[str], List[dict]]:
    """Return a token generator plus a list that fills with sources as the
    first SSE event arrives. Streamlit's st.write_stream consumes the generator;
    afterwards the sources list is populated."""
    sources: List[dict] = []
    resp = requests.post(f"{API_URL}/ask/stream", json={"question": question},
                         stream=True, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    def tokens() -> Iterator[str]:
        event = None
        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None or raw == "":
                continue
            if raw.startswith("event:"):
                event = raw[len("event:"):].strip()
            elif raw.startswith("data:"):
                data = raw[len("data:"):].strip()
                if event == "sources":
                    sources.extend(json.loads(data))
                elif event == "token":
                    yield json.loads(data)["delta"]
                elif event == "error":
                    yield f"\n\n⚠️ {json.loads(data).get('message', 'generation error')}"

    return tokens(), sources


def run_eval() -> Dict:
    r = requests.post(f"{API_URL}/eval", timeout=600)
    r.raise_for_status()
    return r.json()


def render_citations(sources: List[dict]) -> None:
    st.caption("Sources — click to open the exact retrieved passage")
    for i, s in enumerate(sources, start=1):
        page = f" · p.{s['page']}" if s.get("page") is not None else ""
        label = f"[{i}] {s['source']}{page}  ·  score {s['score']:.2f}"
        with st.expander(label):
            st.write(s["text"])


# --- sidebar ------------------------------------------------------------------
st.sidebar.title("📄 DocMind")
st.sidebar.caption("Hybrid retrieval · cross-encoder rerank · grounded, cited, streaming answers")

health = get_health()
if health:
    st.sidebar.success(
        f"Backend online · {health.get('chunks', 0)} chunks\n\n"
        f"hybrid: {'on' if health.get('hybrid') else 'off'} · "
        f"LLM: {'on' if health.get('llm') else 'retrieval-only'}"
    )
else:
    st.sidebar.error(f"Backend unreachable at {API_URL}")

st.sidebar.subheader("Add documents")
uploaded = st.sidebar.file_uploader(
    "PDF, DOCX, TXT, or Markdown",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)
if st.sidebar.button("Index uploaded files", disabled=not uploaded):
    with st.sidebar.status("Indexing..."):
        try:
            result = upload_files(uploaded)
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
                token_gen, sources = stream_answer(question)
                answer = st.write_stream(token_gen)
                if sources:
                    render_citations(sources)
                st.session_state.history.append((question, answer, sources))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Request failed: {exc}")

with tab_eval:
    st.subheader("Retrieval & answer quality")
    st.caption(
        "Each metric isolates a different failure mode — retrieval vs. reranking "
        "vs. generation — on a small labelled evaluation set."
    )
    if st.button("Run evaluation"):
        with st.status("Running the evaluation harness (this calls the LLM per question)…"):
            try:
                st.session_state.eval = run_eval()
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
