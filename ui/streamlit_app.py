"""DocMind — a cinematic RAG "knowledge OS" front-end (Streamlit).

This is the single deployable app for Streamlit Community Cloud. It runs the RAG
pipeline **in-process** and is a pure front-end/UX transformation around the real
backend: upload, parsing, indexing, hybrid retrieval, cross-encoder reranking,
grounded streaming generation, citations, evaluation and session state are all
unchanged. No metrics or documents are faked — every number shown is derived from
the live application.

Memory notes:
  * The pipeline (and its models) is built once via ``st.cache_resource`` and
    reused across reruns/users; models are lazy-loaded inside the pipeline.
"""
from __future__ import annotations

import os
import sys
import time

# Streamlit puts THIS file's directory (ui/) on sys.path, not the repo root, so
# make the repo root importable for `src`, `config`, and `eval`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="DocMind — AI Knowledge Engine", page_icon="◆",
                   layout="wide", initial_sidebar_state="expanded")

MAX_HISTORY_TURNS = 25

# --- secrets bridge -----------------------------------------------------------
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


@st.cache_resource(show_spinner="Booting the knowledge engine (first run only)…")
def load_pipeline():
    from src.pipeline import RAGPipeline
    return RAGPipeline()


def _settings():
    from config import settings
    return settings


def _relevances(scores: list) -> list:
    """Map raw cross-encoder scores to a readable 0-1 *relative* relevance within
    the returned set (top passage ~ full bar). Honest: it ranks the passages
    against each other, which is exactly what the reranker score means here."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.99 for _ in scores]
    return [0.45 + 0.54 * (sc - lo) / (hi - lo) for sc in scores]


# ============================================================================
# THEME
# ============================================================================
def inject_theme() -> None:
    st.markdown(
        """<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --bg:#05070E; --bg2:#0A0F1C; --bg3:#0E1626; --raised:#111B2E;
  --line:rgba(148,163,184,.12); --line2:rgba(56,189,248,.28);
  --cyan:#38BDF8; --cyan2:#22D3EE; --violet:#8B5CF6;
  --text:#E7EEF8; --muted:#8A97AB; --muted2:#586275;
  --glow:rgba(34,211,238,.30);
  --mono:'JetBrains Mono',monospace; --disp:'Space Grotesk',sans-serif; --body:'Inter',sans-serif;
}
html,body,.stApp,[data-testid="stAppViewContainer"]{
  background:var(--bg); color:var(--text); font-family:var(--body);
}
/* atmospheric, static (no continuous motion) background */
.stApp{
  background:
    radial-gradient(1100px 520px at 78% -8%, rgba(56,189,248,.10), transparent 60%),
    radial-gradient(900px 500px at 8% 108%, rgba(139,92,246,.08), transparent 60%),
    linear-gradient(rgba(148,163,184,.030) 1px, transparent 1px) 0 0/34px 34px,
    linear-gradient(90deg, rgba(148,163,184,.030) 1px, transparent 1px) 0 0/34px 34px,
    var(--bg);
}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stToolbar"]{display:none;}
footer{display:none;}
.block-container{padding-top:2.1rem; padding-bottom:6.5rem; max-width:1320px;}
h1,h2,h3,h4{font-family:var(--disp); letter-spacing:-.01em; color:var(--text);}
a{color:var(--cyan);}
::selection{background:rgba(56,189,248,.28);}

/* ---- sidebar ---- */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#070B15, #060910);
  border-right:1px solid var(--line);
}
[data-testid="stSidebar"] .block-container{padding-top:1.3rem;}
.dm-logo{font-family:var(--disp); font-weight:700; font-size:1.35rem; letter-spacing:.14em;
  color:var(--text); line-height:1;}
.dm-logo .dot{color:var(--cyan);}
.dm-logo-sub{font-family:var(--mono); font-size:.62rem; letter-spacing:.42em;
  color:var(--muted2); margin-top:.35rem;}
.dm-navactive{
  display:flex; align-items:center; gap:.6rem; padding:.55rem .7rem; margin:.12rem 0;
  border-radius:9px; color:var(--text); font-weight:600; font-size:.9rem;
  background:linear-gradient(90deg, rgba(56,189,248,.14), rgba(56,189,248,.02));
  border:1px solid var(--line); box-shadow:inset 2px 0 0 var(--cyan), 0 0 24px -8px var(--glow);
}
.dm-navactive .g{width:5px;height:5px;border-radius:50%;background:var(--cyan);
  box-shadow:0 0 10px 2px var(--glow);}
/* inactive nav = sidebar buttons */
[data-testid="stSidebar"] .stButton>button{
  width:100%; text-align:left; justify-content:flex-start;
  background:transparent; border:1px solid transparent; color:var(--muted);
  font-weight:500; font-size:.9rem; padding:.5rem .7rem; border-radius:9px;
  transition:all .16s ease;
}
[data-testid="stSidebar"] .stButton>button:hover{
  color:var(--text); background:rgba(148,163,184,.06); border-color:var(--line);
  transform:translateX(2px);
}
.dm-side-label{font-family:var(--mono); font-size:.6rem; letter-spacing:.28em;
  color:var(--muted2); margin:1.1rem 0 .5rem;}
.dm-engine{border:1px solid var(--line); border-radius:12px; padding:.8rem .85rem;
  background:linear-gradient(180deg, rgba(56,189,248,.05), transparent);}
.dm-engine .top{display:flex; align-items:center; justify-content:space-between;}
.dm-engine .name{font-family:var(--mono); font-size:.68rem; letter-spacing:.2em; color:var(--muted);}
.dm-online{display:flex; align-items:center; gap:.4rem; font-family:var(--mono);
  font-size:.68rem; color:var(--cyan2); letter-spacing:.08em;}
.dm-online .pulse{width:7px;height:7px;border-radius:50%;background:var(--cyan2);
  box-shadow:0 0 0 0 rgba(34,211,238,.55); animation:pulse 2.4s infinite;}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(34,211,238,.5)}70%{box-shadow:0 0 0 7px rgba(34,211,238,0)}100%{box-shadow:0 0 0 0 rgba(34,211,238,0)}}
.dm-engine .row{display:flex; align-items:center; justify-content:space-between;
  font-size:.78rem; color:var(--muted); padding:.28rem 0; border-top:1px solid var(--line); margin-top:.5rem;}
.dm-engine .row b{color:var(--text); font-weight:500; font-family:var(--mono); font-size:.72rem;}
.dm-profile{display:flex; align-items:center; gap:.6rem; padding-top:.4rem;}
.dm-avatar{width:32px;height:32px;border-radius:9px;
  background:linear-gradient(135deg,var(--cyan),var(--violet)); flex:0 0 auto;}
.dm-profile .who{font-size:.82rem; color:var(--text); font-weight:600;}
.dm-profile .plan{font-family:var(--mono); font-size:.6rem; letter-spacing:.14em; color:var(--muted2);}

/* ---- hero text ---- */
.dm-kicker{font-family:var(--mono); font-size:.68rem; letter-spacing:.38em; color:var(--cyan2);
  display:flex; align-items:center; gap:.6rem;}
.dm-kicker::before{content:""; width:26px; height:1px; background:linear-gradient(90deg,var(--cyan),transparent);}
.dm-h1{font-family:var(--disp); font-weight:700; font-size:3.15rem; line-height:1.02;
  letter-spacing:-.02em; margin:.5rem 0 .2rem;}
.dm-h1 .amp{background:linear-gradient(120deg,var(--cyan),var(--violet));
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;}
.dm-lede{color:var(--muted); font-size:1.02rem; max-width:34rem; line-height:1.6;}
.dm-chips{display:flex; flex-wrap:wrap; gap:.45rem; margin-top:1rem;}
.dm-chip{font-family:var(--mono); font-size:.68rem; letter-spacing:.06em; color:var(--muted);
  border:1px solid var(--line); border-radius:999px; padding:.32rem .7rem; background:rgba(148,163,184,.04);}
.dm-chip .d{color:var(--cyan2);}

/* ---- generic panels & labels ---- */
.dm-panel{border:1px solid var(--line); border-radius:14px; background:rgba(10,15,28,.6);
  padding:1rem 1.05rem;}
.dm-eyebrow{font-family:var(--mono); font-size:.62rem; letter-spacing:.28em; color:var(--muted2);
  margin-bottom:.55rem;}
.dm-emptybig{font-family:var(--disp); font-size:1.5rem; font-weight:600; color:var(--text);}
.dm-emptysub{color:var(--muted); font-size:.95rem; margin-top:.25rem;}
/* real bordered containers used as contextual panels */
[data-testid="stVerticalBlockBorderWrapper"]{
  border:1px solid var(--line) !important; border-radius:16px !important;
  background:linear-gradient(180deg, rgba(12,18,32,.65), rgba(8,12,22,.5)) !important;
  padding:.35rem !important;}

/* ---- suggested command prompts (buttons) ---- */
.dm-suggests .stButton>button{
  width:100%; text-align:left; justify-content:space-between;
  background:linear-gradient(180deg, rgba(17,27,46,.7), rgba(10,15,28,.5));
  border:1px solid var(--line); color:var(--text); font-weight:500; font-size:.9rem;
  padding:.8rem .95rem; border-radius:12px; transition:all .18s ease;
}
.dm-suggests .stButton>button:hover{
  border-color:var(--line2); transform:translateY(-2px);
  box-shadow:0 10px 30px -14px var(--glow); color:var(--text);
}
.dm-suggests .stButton>button::after{content:"→"; color:var(--cyan2); margin-left:.6rem;}

/* ---- tool tabs (buttons row) ---- */
.dm-tools .stButton>button{
  background:transparent; border:1px solid var(--line); color:var(--muted);
  font-family:var(--mono); font-size:.72rem; letter-spacing:.04em; font-weight:500;
  padding:.42rem .7rem; border-radius:999px; transition:all .16s ease;
}
.dm-tools .stButton>button:hover{color:var(--text); border-color:var(--line2);
  box-shadow:0 0 22px -10px var(--glow);}

/* ---- generic buttons ---- */
.stButton>button{border-radius:11px; font-weight:600; transition:all .16s ease;}
.stButton>button[kind="primary"], .stFormSubmitButton>button{
  background:linear-gradient(120deg,var(--cyan),#2b9fe0); color:#04121e; border:none;
  box-shadow:0 10px 30px -12px var(--glow);}
.stButton>button[kind="primary"]:hover{transform:translateY(-1px); filter:brightness(1.06);}

/* ---- chat ---- */
[data-testid="stChatMessage"]{background:transparent; padding:.2rem 0;}
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"]{
  background:linear-gradient(135deg,var(--cyan),var(--violet)) !important;}
[data-testid="stChatInput"]{background:transparent;}
[data-testid="stChatInput"] textarea{color:var(--text) !important;}
div[data-baseweb="textarea"], [data-testid="stChatInput"] > div{
  background:var(--bg3) !important; border:1px solid var(--line) !important; border-radius:14px !important;
  transition:border-color .2s ease, box-shadow .2s ease;}
div[data-baseweb="textarea"]:focus-within, [data-testid="stChatInput"] > div:focus-within{
  border-color:var(--line2) !important; box-shadow:0 0 0 1px var(--line2), 0 0 34px -12px var(--glow) !important;}

/* ---- expanders ---- */
[data-testid="stExpander"]{border:1px solid var(--line); border-radius:12px;
  background:rgba(10,15,28,.5); overflow:hidden;}
[data-testid="stExpander"] summary{color:var(--muted); font-family:var(--mono); font-size:.78rem;}
[data-testid="stExpander"] summary:hover{color:var(--text);}

/* ---- file uploader dropzone ---- */
[data-testid="stFileUploaderDropzone"]{
  background:linear-gradient(180deg, rgba(56,189,248,.05), rgba(10,15,28,.4));
  border:1.4px dashed var(--line2); border-radius:14px; color:var(--muted);}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--cyan); background:rgba(56,189,248,.07);}

/* ---- answer + sources ---- */
.dm-answerlabel{font-family:var(--mono); font-size:.62rem; letter-spacing:.3em; color:var(--cyan2); margin:.2rem 0 .1rem;}
.dm-meta{display:flex; flex-wrap:wrap; gap:1.2rem; font-family:var(--mono); font-size:.7rem;
  color:var(--muted); margin:.6rem 0 .3rem;}
.dm-meta b{color:var(--cyan2); font-weight:500;}
.dm-srchead{font-family:var(--mono); font-size:.62rem; letter-spacing:.3em; color:var(--muted2); margin:.4rem 0 .5rem;}
.dm-src{display:flex; gap:.75rem; align-items:flex-start; padding:.55rem .2rem; border-top:1px solid var(--line);}
.dm-src .idx{font-family:var(--mono); font-size:.72rem; color:var(--cyan2); min-width:1.6rem; padding-top:.1rem;}
.dm-src .name{color:var(--text); font-size:.88rem; font-weight:500;}
.dm-src .sub{font-family:var(--mono); font-size:.68rem; color:var(--muted); margin-top:.15rem;}
.dm-relbar{height:3px; border-radius:2px; background:rgba(148,163,184,.14); margin-top:.4rem; overflow:hidden;}
.dm-relbar > i{display:block; height:100%; background:linear-gradient(90deg,var(--cyan),var(--violet));}

/* ---- rag pipeline strip ---- */
.dm-pipe{display:flex; align-items:center; gap:.35rem; flex-wrap:wrap; margin:.3rem 0 .7rem;}
.dm-node{display:flex; align-items:center; gap:.4rem; font-family:var(--mono); font-size:.64rem;
  letter-spacing:.12em; padding:.3rem .6rem; border-radius:999px; border:1px solid var(--line);
  color:var(--muted2); background:rgba(148,163,184,.03);}
.dm-node .dt{width:6px;height:6px;border-radius:50%; background:var(--muted2);}
.dm-node.done{color:var(--cyan2); border-color:var(--line2);}
.dm-node.done .dt{background:var(--cyan2); box-shadow:0 0 8px 1px var(--glow);}
.dm-node.active{color:var(--text); border-color:var(--cyan); box-shadow:0 0 26px -10px var(--glow);}
.dm-node.active .dt{background:var(--cyan); animation:pulse 1.3s infinite;}
.dm-arrow{color:var(--muted2); font-size:.7rem;}

/* ---- doc rows / stats ---- */
.dm-doc{display:flex; align-items:center; gap:.7rem; padding:.6rem .3rem; border-top:1px solid var(--line);}
.dm-doc .ic{width:30px;height:30px;border-radius:8px; flex:0 0 auto;
  background:linear-gradient(135deg, rgba(56,189,248,.2), rgba(139,92,246,.15));
  border:1px solid var(--line);}
.dm-doc .fn{color:var(--text); font-size:.85rem; font-weight:500; word-break:break-all;}
.dm-doc .mt{font-family:var(--mono); font-size:.66rem; color:var(--muted); margin-top:.12rem;}
.dm-doc .yours{color:var(--violet);}
.dm-stat{border:1px solid var(--line); border-radius:12px; padding:.8rem .9rem;
  background:linear-gradient(180deg, rgba(17,27,46,.6), transparent);}
.dm-stat .n{font-family:var(--disp); font-weight:700; font-size:1.5rem; color:var(--text); line-height:1;}
.dm-stat .l{font-family:var(--mono); font-size:.6rem; letter-spacing:.18em; color:var(--muted2); margin-top:.35rem;}

@media (prefers-reduced-motion: reduce){
  .dm-online .pulse, .dm-node.active .dt{animation:none !important;}
  .dm-suggests .stButton>button:hover{transform:none;}
}
</style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# HERO — animated knowledge-engine visualization (self-contained iframe)
# ============================================================================
def render_hero_viz(height: int = 300) -> None:
    html = """
<!doctype html><html><head><meta charset="utf-8"><style>
  :root{--cyan:#38BDF8;--cyan2:#22D3EE;--violet:#8B5CF6;--muted:#586275;--line:rgba(148,163,184,.14);}
  html,body{margin:0;background:transparent;overflow:hidden;font-family:'JetBrains Mono',monospace;}
  .wrap{position:relative;width:100%;height:%HEIGHT%px;}
  svg{width:100%;height:100%;display:block;}
  .sheet{opacity:.0; animation:flowIn 7s ease-in infinite;}
  @keyframes flowIn{0%{opacity:0;transform:translateX(0) scale(.9)}
    12%{opacity:.9}55%{opacity:.55}100%{opacity:0;transform:translateX(120px) scale(.6)}}
  .ring{transform-origin:center; animation:spin 26s linear infinite;}
  .ring.rev{animation:spin 34s linear infinite reverse;}
  @keyframes spin{to{transform:rotate(360deg)}}
  .core{animation:breathe 4s ease-in-out infinite;transform-origin:center;}
  @keyframes breathe{0%,100%{opacity:.85}50%{opacity:1}}
  .particle{animation:travel 3.4s ease-in infinite;}
  @keyframes travel{0%{opacity:0}20%{opacity:1}100%{opacity:0;transform:translate(var(--dx),var(--dy))}}
  .pnode circle{animation:nodeglow 6s ease-in-out infinite;}
  @keyframes nodeglow{0%,100%{fill:#0E1626}50%{fill:#123049}}
  .n0{animation-delay:0s}.n1{animation-delay:1s}.n2{animation-delay:2s}
  .n3{animation-delay:3s}.n4{animation-delay:4s}.n5{animation-delay:5s}
  .lab{fill:#586275;font-size:8px;letter-spacing:1.5px;}
  @media (prefers-reduced-motion: reduce){*{animation:none !important;}.sheet{opacity:.5;}}
</style></head><body><div class="wrap">
<svg viewBox="0 0 640 300" preserveAspectRatio="xMidYMid meet">
  <defs>
    <radialGradient id="g" cx="50%" cy="50%"><stop offset="0%" stop-color="#22D3EE" stop-opacity=".55"/>
      <stop offset="55%" stop-color="#38BDF8" stop-opacity=".12"/><stop offset="100%" stop-color="#38BDF8" stop-opacity="0"/></radialGradient>
    <linearGradient id="edge" x1="0" x2="1"><stop offset="0" stop-color="#38BDF8"/><stop offset="1" stop-color="#8B5CF6"/></linearGradient>
  </defs>
  <!-- soft core glow -->
  <circle cx="215" cy="150" r="120" fill="url(#g)"/>
  <!-- orbital rings -->
  <g class="ring"><ellipse cx="215" cy="150" rx="96" ry="40" fill="none" stroke="url(#edge)" stroke-opacity=".35" stroke-width="1"/></g>
  <g class="ring rev"><ellipse cx="215" cy="150" rx="60" ry="92" fill="none" stroke="#38BDF8" stroke-opacity=".2" stroke-width="1"/></g>
  <!-- knowledge core -->
  <g class="core">
    <circle cx="215" cy="150" r="30" fill="#0A1424" stroke="url(#edge)" stroke-width="1.4"/>
    <circle cx="215" cy="150" r="16" fill="none" stroke="#22D3EE" stroke-opacity=".7" stroke-width="1"/>
    <circle cx="215" cy="150" r="4" fill="#22D3EE"/>
  </g>
  <!-- incoming document sheets -->
  <g fill="#0E1626" stroke="#38BDF8" stroke-opacity=".5">
    <rect class="sheet" style="animation-delay:0s"  x="40"  y="60"  width="34" height="44" rx="4"/>
    <rect class="sheet" style="animation-delay:1.6s" x="30"  y="150" width="34" height="44" rx="4"/>
    <rect class="sheet" style="animation-delay:3.1s" x="55"  y="220" width="34" height="44" rx="4"/>
    <rect class="sheet" style="animation-delay:4.6s" x="120" y="40"  width="34" height="44" rx="4"/>
  </g>
  <!-- travelling data particles -->
  <g fill="#22D3EE">
    <circle class="particle" style="--dx:150px;--dy:70px;animation-delay:.2s"  cx="70" cy="82"  r="2"/>
    <circle class="particle" style="--dx:170px;--dy:0px;animation-delay:1.3s"  cx="60" cy="172" r="2"/>
    <circle class="particle" style="--dx:150px;--dy:-70px;animation-delay:2.4s" cx="85" cy="242" r="2"/>
  </g>
  <!-- pipeline nodes -->
  <g class="pnode">
    %NODES%
  </g>
</svg></div></body></html>
"""
    stages = ["DOCS", "RETRIEVE", "RERANK", "GENERATE", "CITE", "ANSWER"]
    nodes = []
    x0, y = 360, 150
    for i, s in enumerate(stages):
        cx = x0 + i * 46
        nodes.append(
            f'<g class="n{i}"><circle cx="{cx}" cy="{y}" r="8" stroke="#38BDF8" '
            f'stroke-opacity=".45" stroke-width="1" fill="#0E1626"/></g>')
        if i < len(stages) - 1:
            nodes.append(f'<line x1="{cx+8}" y1="{y}" x2="{cx+38}" y2="{y}" '
                         f'stroke="#38BDF8" stroke-opacity=".22" stroke-width="1"/>')
        nodes.append(f'<text class="lab" x="{cx}" y="{y+22}" text-anchor="middle">{s}</text>')
    html = html.replace("%HEIGHT%", str(height)).replace("%NODES%", "".join(nodes))
    components.html(html, height=height, scrolling=False)


# ============================================================================
# RAG PIPELINE STRIP (in-app, real staged progress)
# ============================================================================
_STAGES = ["QUERY", "RETRIEVE", "RERANK", "GENERATE", "CITE"]


def pipeline_strip(active: int) -> str:
    """active = index of the currently-running stage; >=len means all done."""
    parts = []
    for i, s in enumerate(_STAGES):
        cls = "done" if i < active else ("active" if i == active else "")
        parts.append(f'<span class="dm-node {cls}"><span class="dt"></span>{s}</span>')
        if i < len(_STAGES) - 1:
            parts.append('<span class="dm-arrow">→</span>')
    return f'<div class="dm-pipe">{"".join(parts)}</div>'


# ============================================================================
# ANSWER SOURCES + METADATA
# ============================================================================
def render_answer_meta(sources, retrieved: int, latency: float) -> None:
    used = len({s.source for s in sources})
    reranked = len(sources)
    st.markdown(
        f'<div class="dm-meta">'
        f'<span><b>{retrieved}</b> candidates retrieved</span>'
        f'<span><b>{reranked}</b> reranked</span>'
        f'<span><b>{used}</b> source{"s" if used != 1 else ""} used</span>'
        f'<span><b>{latency:.2f}s</b> latency</span>'
        f'</div>', unsafe_allow_html=True)


def render_sources(sources) -> None:
    if not sources:
        return
    rels = _relevances([s.score for s in sources])
    st.markdown('<div class="dm-srchead">SOURCES</div>', unsafe_allow_html=True)
    for i, (s, rel) in enumerate(zip(sources, rels), start=1):
        page = f"Page {s.page} · " if s.page is not None else ""
        with st.expander(f"{i:02d}   {s.source}    ·    {page}relevance {rel:.2f}"):
            st.markdown(
                f'<div class="dm-relbar"><i style="width:{rel*100:.0f}%"></i></div>',
                unsafe_allow_html=True)
            st.write(s.text)


def how_answer_found(retrieved: int, reranked: int) -> None:
    s = _settings()
    with st.expander("HOW THIS ANSWER WAS FOUND"):
        steps = [
            ("Query", "your question" + (" (rewritten to a standalone search query on follow-ups)" if True else "")),
            ("Hybrid retrieval", f"dense (FAISS) + BM25, RRF-fused → {retrieved} candidate chunks"),
            ("Cross-encoder rerank", f"{s.reranker_model.split('/')[-1]} re-scores → top {reranked}"),
            ("Top context", "highest-relevance passages sent to the model"),
            ("LLM", f"{s.llm_model} — grounded, cited"),
            ("Grounded answer", "answered only from the retrieved passages"),
        ]
        rows = "".join(
            f'<div class="dm-src"><div class="idx">{i+1:02d}</div><div>'
            f'<div class="name">{t}</div><div class="sub">{d}</div></div></div>'
            for i, (t, d) in enumerate(steps))
        st.markdown(rows, unsafe_allow_html=True)


# ============================================================================
# BOOT
# ============================================================================
inject_theme()
pipe = load_pipeline()
s = _settings()
store_sources_map: dict = {}
for c in pipe.store.chunks:
    store_sources_map[c.source] = store_sources_map.get(c.source, 0) + 1
store_sources = set(store_sources_map)
uploaded = st.session_state.get("uploaded_sources", set()) & store_sources
scope_sources = uploaded or None

if "section" not in st.session_state:
    st.session_state.section = "Chat"
if "history" not in st.session_state:
    st.session_state.history = []

NAV = ["Chat", "Documents", "Knowledge Base", "Collections",
       "Analytics", "Playground", "Settings"]


# ============================================================================
# SIDEBAR — command center
# ============================================================================
with st.sidebar:
    st.markdown('<div class="dm-logo">DOC<span class="dot">MIND</span></div>'
                '<div class="dm-logo-sub">RAG ASSISTANT</div>', unsafe_allow_html=True)
    st.markdown('<div class="dm-side-label">WORKSPACE</div>', unsafe_allow_html=True)
    for item in NAV:
        if item == st.session_state.section:
            st.markdown(f'<div class="dm-navactive"><span class="g"></span>{item}</div>',
                        unsafe_allow_html=True)
        else:
            if st.button(item, key=f"nav_{item}"):
                st.session_state.section = item
                st.rerun()

    st.markdown('<div class="dm-side-label">SYSTEM</div>', unsafe_allow_html=True)
    llm_state = "llama-3.1-8b" if s.has_llm else "offline"
    st.markdown(
        f'<div class="dm-engine"><div class="top">'
        f'<span class="name">RAG ENGINE</span>'
        f'<span class="dm-online"><span class="pulse"></span>ONLINE</span></div>'
        f'<div class="row">Hybrid Retrieval<b>{"ON" if s.hybrid else "OFF"}</b></div>'
        f'<div class="row">Cross-Encoder<b>ON</b></div>'
        f'<div class="row">LLM<b>{"ON" if s.has_llm else "OFF"}</b></div>'
        f'</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dm-profile"><div class="dm-avatar"></div><div>'
        '<div class="who">You</div><div class="plan">FREE · STREAMLIT CLOUD</div>'
        '</div></div>', unsafe_allow_html=True)


# ============================================================================
# SHARED: upload + documents + stats (contextual panel / documents section)
# ============================================================================
def render_upload_panel() -> None:
    st.markdown('<div class="dm-eyebrow">ADD DOCUMENTS</div>', unsafe_allow_html=True)
    files = st.file_uploader("Drop documents here — PDF · DOCX · TXT · MD (max 20MB)",
                             type=["pdf", "docx", "txt", "md"],
                             accept_multiple_files=True, key="uploader")
    if st.button("Upload & Index", type="primary", disabled=not files,
                 use_container_width=True):
        prog = st.progress(0, text="Uploading…")
        try:
            payloads = [(f.name, f.getvalue()) for f in files]
            prog.progress(35, text="Parsing & chunking…")
            result = pipe.add_documents(payloads)
            prog.progress(85, text="Indexing…")
            added = sum(f["chunks_added"] for f in result["files"])
            st.session_state.setdefault("uploaded_sources", set()).update(
                f.name for f in files)
            prog.progress(100, text="Ready")
            st.success(f"Indexed {added} new chunks · {result['total_chunks']} total")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            prog.empty()
            st.error(str(exc))


def render_recent_docs(limit: int | None = 6) -> None:
    st.markdown('<div class="dm-eyebrow">RECENT DOCUMENTS</div>', unsafe_allow_html=True)
    if not store_sources_map:
        st.caption("No documents indexed yet.")
        return
    items = sorted(store_sources_map.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = items if limit is None else items[:limit]
    rows = []
    for name, n in shown:
        mine = ' · <span class="yours">yours</span>' if name in uploaded else ""
        rows.append(
            f'<div class="dm-doc"><div class="ic"></div><div>'
            f'<div class="fn">{name}</div>'
            f'<div class="mt">{n} chunks · Indexed{mine}</div></div></div>')
    st.markdown("".join(rows), unsafe_allow_html=True)
    if limit is not None and len(items) > limit:
        if st.button(f"View all {len(items)} →", key="viewall"):
            st.session_state.section = "Documents"
            st.rerun()


def render_stats() -> None:
    ndocs, nchunks = len(store_sources_map), len(pipe.store.chunks)
    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="dm-stat"><div class="n">{ndocs}</div>'
                f'<div class="l">DOCUMENTS</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="dm-stat"><div class="n">{nchunks:,}</div>'
                f'<div class="l">CHUNKS INDEXED</div></div>', unsafe_allow_html=True)


# ============================================================================
# QUERY HANDLER
# ============================================================================
def run_query(question: str, workspace) -> None:
    with workspace:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            strip = st.empty()
            try:
                strip.markdown(pipeline_strip(1), unsafe_allow_html=True)  # RETRIEVE
                turns = [(q, a) for q, a, *_ in st.session_state.history]
                t0 = time.time()
                sources, token_gen = pipe.answer_stream(
                    question, history=turns, allowed_sources=scope_sources)
                strip.markdown(pipeline_strip(3), unsafe_allow_html=True)  # GENERATE
                st.markdown('<div class="dm-answerlabel">AI ANSWER</div>',
                            unsafe_allow_html=True)
                answer = st.write_stream(token_gen)
                latency = time.time() - t0
                strip.markdown(pipeline_strip(len(_STAGES)), unsafe_allow_html=True)  # done
                scope_n = (sum(store_sources_map[x] for x in scope_sources)
                           if scope_sources else len(pipe.store.chunks))
                retrieved = min(s.retrieve_k, scope_n)
                st.session_state.history.append(
                    (question, answer, sources, (retrieved, latency)))
                del st.session_state.history[:-MAX_HISTORY_TURNS]
                st.rerun()  # fold the turn into the transcript for a clean layout
            except Exception as exc:  # noqa: BLE001
                strip.empty()
                st.error(f"Generation failed: {exc}")


# ============================================================================
# SECTION: CHAT (the workspace)
# ============================================================================
def section_chat() -> None:
    left, right = st.columns([2.15, 1], gap="large")

    with left:
        first_time = not st.session_state.history
        if first_time:
            st.markdown('<div class="dm-kicker">AI-POWERED DOCUMENT INTELLIGENCE</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="dm-h1">YOUR KNOWLEDGE<br><span class="amp">AMPLIFIED.</span></div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="dm-lede">Upload. Explore. Ask. Get grounded, '
                        'cited answers from your own documents.</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="dm-chips">'
                '<span class="dm-chip"><span class="d">◆</span> Grounded</span>'
                '<span class="dm-chip"><span class="d">◆</span> Cited</span>'
                '<span class="dm-chip"><span class="d">◆</span> Hybrid Retrieval</span>'
                '<span class="dm-chip"><span class="d">◆</span> Cross-Encoder</span>'
                '<span class="dm-chip"><span class="d">◆</span> Streaming</span>'
                '</div>', unsafe_allow_html=True)
            render_hero_viz(280)

        # scope indicator
        if scope_sources:
            st.markdown('<div class="dm-answerlabel">ANSWERING FROM YOUR DOCUMENTS · '
                        + " · ".join(sorted(scope_sources)) + '</div>', unsafe_allow_html=True)

        # tool tabs -> intent presets
        st.markdown('<div class="dm-eyebrow" style="margin-top:.8rem">WORKSPACE TOOLS</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="dm-tools">', unsafe_allow_html=True)
        tools = {
            "Summarize": "Summarize this document",
            "Compare": "Compare the main sections of this document",
            "Translate": "Translate the key points into simple English",
            "Find Risks": "What are the risks or concerns?",
            "Action Items": "List the action items",
        }
        picked = None
        tcols = st.columns(len(tools))
        for col, (label, prompt) in zip(tcols, tools.items()):
            if col.button(label, key=f"tool_{label}"):
                picked = prompt
        st.markdown('</div>', unsafe_allow_html=True)

        # transcript (with persisted metadata)
        for q, a, srcs, *rest in st.session_state.history:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.markdown('<div class="dm-answerlabel">AI ANSWER</div>', unsafe_allow_html=True)
                st.write(a)
                if rest and rest[0]:
                    retrieved, latency = rest[0]
                    render_answer_meta(srcs, retrieved, latency)
                render_sources(srcs)
                if rest and rest[0]:
                    how_answer_found(rest[0][0], len(srcs))

        # empty state suggestions
        if first_time:
            st.markdown('<div class="dm-emptybig">Ask anything about your documents</div>'
                        '<div class="dm-emptysub">Get accurate, cited answers from your own '
                        'knowledge base.</div>', unsafe_allow_html=True)
            st.markdown('<div class="dm-suggests">', unsafe_allow_html=True)
            scols = st.columns(2)
            for i, sug in enumerate([
                "Summarize this document", "What are the key findings?",
                "Explain it like I'm a beginner", "Find potential risks",
                "List action items", "Compare the sections"]):
                if scols[i % 2].button(sug, key=f"sug_{i}", use_container_width=True):
                    picked = sug
            st.markdown('</div>', unsafe_allow_html=True)

        workspace = st.container()

    with right:
        with st.container(border=True):
            render_upload_panel()
            st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)
            render_recent_docs(limit=6)
            st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)
            render_stats()

    # command-center input (global bottom bar)
    hint = "Hybrid RAG · ON" if s.hybrid else "Dense RAG"
    typed = st.chat_input(f"Ask a question about your documents…   ({hint})")
    question = typed or picked
    if question:
        run_query(question, workspace)


# ============================================================================
# SECTION: DOCUMENTS / KNOWLEDGE BASE
# ============================================================================
def section_documents() -> None:
    st.markdown('<div class="dm-kicker">DOCUMENT LIBRARY</div>', unsafe_allow_html=True)
    st.markdown('<div class="dm-h1" style="font-size:2.2rem">Documents</div>',
                unsafe_allow_html=True)
    left, right = st.columns([2.15, 1], gap="large")
    with right:
        with st.container(border=True):
            render_upload_panel()
            st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)
            render_stats()
    with left:
        if not store_sources_map:
            st.info("No documents indexed yet — upload one to begin.")
            return
        st.markdown('<div class="dm-eyebrow">INDEXED DOCUMENTS · browse chunks</div>',
                    unsafe_allow_html=True)
        for name, n in sorted(store_sources_map.items(), key=lambda kv: (-kv[1], kv[0])):
            mine = " · yours" if name in uploaded else ""
            with st.expander(f"{name}   ·   {n} chunks · Indexed{mine}"):
                chunks = [c for c in pipe.store.chunks if c.source == name]
                for c in chunks[:40]:
                    pg = f"Page {c.page} · " if c.page is not None else ""
                    st.markdown(f'<div class="dm-src"><div class="idx">#{c.chunk_id}</div>'
                                f'<div><div class="sub">{pg}chunk {c.chunk_id}</div></div></div>',
                                unsafe_allow_html=True)
                    st.write(c.text)
                if len(chunks) > 40:
                    st.caption(f"… and {len(chunks) - 40} more chunks")


def section_knowledge_base() -> None:
    st.markdown('<div class="dm-kicker">KNOWLEDGE BASE</div>', unsafe_allow_html=True)
    st.markdown('<div class="dm-h1" style="font-size:2.2rem">Index Overview</div>',
                unsafe_allow_html=True)
    render_stats()
    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="dm-eyebrow">RETRIEVAL ENGINE CONFIGURATION</div>',
                unsafe_allow_html=True)
    cfg = [
        ("Embedding model", s.embed_model.split("/")[-1]),
        ("Reranker (cross-encoder)", s.reranker_model.split("/")[-1]),
        ("Hybrid (dense + BM25)", "ON" if s.hybrid else "OFF"),
        ("RRF constant", str(s.rrf_k)),
        ("Candidates retrieved (k)", str(s.retrieve_k)),
        ("Reranked to (top-n)", str(s.rerank_top_n)),
        ("Chunk size / overlap", f"{s.chunk_size} / {s.chunk_overlap}"),
        ("LLM", s.llm_model if s.has_llm else "not configured"),
    ]
    rows = "".join(f'<div class="dm-engine" style="margin-bottom:.4rem"><div class="row" '
                   f'style="border:none;margin:0;padding:.15rem 0">{k}<b>{v}</b></div></div>'
                   for k, v in cfg)
    st.markdown(rows, unsafe_allow_html=True)


# ============================================================================
# SECTION: ANALYTICS (real evaluation dashboard)
# ============================================================================
def section_analytics() -> None:
    st.markdown('<div class="dm-kicker">EVALUATION &amp; ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="dm-h1" style="font-size:2.2rem">Retrieval &amp; Answer Quality</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="dm-lede">Each metric isolates a different failure mode — '
                'retrieval vs. reranking vs. generation — on a labelled evaluation set.</div>',
                unsafe_allow_html=True)
    st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)
    if st.button("▶  Run evaluation", type="primary"):
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
            disp = "—" if val != val else f"{val:.2f}"
            col.markdown(f'<div class="dm-stat"><div class="n">{disp}</div>'
                         f'<div class="l">{m["name"].upper()}</div></div>',
                         unsafe_allow_html=True)
        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        chart_df = pd.DataFrame(
            {"score": [m["value"] if m["value"] == m["value"] else 0.0 for m in metrics]},
            index=[m["name"] for m in metrics])
        st.bar_chart(chart_df, height=280, color="#38BDF8")
        st.caption(f"Evaluated on {report['n_questions']} questions · faithfulness via "
                   f"{'LLM-as-judge' if report['llm_judge'] else 'embedding fallback'}")
        st.divider()
        for m in metrics:
            st.markdown(f"**{m['name']}** — {m['explanation']}")
    else:
        st.info("Click **Run evaluation** to compute recall@k, MRR, faithfulness "
                "and answer relevance on the labelled set.")


# ============================================================================
# SECTION: SETTINGS (real, read-only config)
# ============================================================================
def section_settings() -> None:
    st.markdown('<div class="dm-kicker">SETTINGS</div>', unsafe_allow_html=True)
    st.markdown('<div class="dm-h1" style="font-size:2.2rem">Configuration</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="dm-lede">These values come from <code>config.py</code> and '
                'environment/secrets. Change them via environment variables or Streamlit '
                'secrets, then reboot the app.</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)
    section_knowledge_base_config = [
        ("EMBED_MODEL", s.embed_model),
        ("RERANKER_MODEL", s.reranker_model),
        ("HYBRID", "true" if s.hybrid else "false"),
        ("RRF_K", str(s.rrf_k)),
        ("RETRIEVE_K", str(s.retrieve_k)),
        ("RERANK_TOP_N", str(s.rerank_top_n)),
        ("CHUNK_STRATEGY", s.chunk_strategy),
        ("CHUNK_SIZE / OVERLAP", f"{s.chunk_size} / {s.chunk_overlap}"),
        ("LLM_MODEL", s.llm_model),
        ("LLM configured", "yes" if s.has_llm else "no (retrieval-only)"),
    ]
    df = pd.DataFrame(section_knowledge_base_config, columns=["Setting", "Value"])
    st.dataframe(df, use_container_width=True, hide_index=True)


def section_roadmap(name: str) -> None:
    st.markdown(f'<div class="dm-kicker">{name.upper()}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="dm-h1" style="font-size:2.2rem">{name}</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="dm-panel"><div class="dm-emptybig">On the roadmap</div>'
                '<div class="dm-emptysub">This module isn\'t part of the deployed build yet. '
                'The engine below already powers Chat, Documents, Knowledge Base and Analytics '
                '— all with live data, no placeholders.</div></div>', unsafe_allow_html=True)


# ============================================================================
# ROUTER
# ============================================================================
sec = st.session_state.section
if sec == "Chat":
    section_chat()
elif sec == "Documents":
    section_documents()
elif sec == "Knowledge Base":
    section_knowledge_base()
elif sec == "Analytics":
    section_analytics()
elif sec == "Settings":
    section_settings()
else:  # Collections, Playground
    section_roadmap(sec)
