"""Visual system for DocMind: colour tokens, motion system, 3D knowledge core,
and the CSS-animated upload / RAG-pipeline sequences.

All motion is in-app CSS (transforms + opacity keyframes) — no iframes — so it is
GPU-friendly and never reloads/flashes on a Streamlit rerun.
"""
from __future__ import annotations

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --bg:#07080D; --bg2:#0B0D14; --surface:#0E1018; --surface2:#12141E; --raised:#161926;
  --line:rgba(148,163,184,.09); --line2:rgba(99,102,241,.28);
  --indigo:#6366F1; --violet:#8B5CF6; --cyan:#22D3EE; --magenta:#D946EF; --success:#34D399;
  --text:#F8FAFC; --muted:#94A3B8; --muted2:#5B6577;
  --g-iv:linear-gradient(120deg,#6366F1,#8B5CF6);
  --g-vm:linear-gradient(120deg,#8B5CF6,#D946EF);
  --g-ci:linear-gradient(120deg,#22D3EE,#6366F1);
  --mono:'JetBrains Mono',monospace; --disp:'Space Grotesk',sans-serif; --body:'Inter',sans-serif;
  --fast:140ms; --normal:240ms; --slow:420ms;
  --ease:cubic-bezier(.22,.61,.36,1); --spring:cubic-bezier(.34,1.4,.64,1);
}
html,body,.stApp,[data-testid="stAppViewContainer"]{background:var(--bg); color:var(--text); font-family:var(--body);}
.stApp{
  background:
    radial-gradient(920px 480px at 82% -10%, rgba(99,102,241,.10), transparent 62%),
    radial-gradient(760px 460px at 4% 106%, rgba(217,70,239,.06), transparent 60%),
    var(--bg);
}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stToolbar"]{display:none;}
footer{display:none;}
.block-container{padding-top:2rem; padding-bottom:6.5rem; max-width:1340px;}
h1,h2,h3,h4{font-family:var(--disp); letter-spacing:-.015em; color:var(--text);}
a{color:var(--indigo);} ::selection{background:rgba(139,92,246,.30);}
* { -webkit-font-smoothing:antialiased; }

/* soft entrance for streamed blocks so nothing snaps in */
[data-testid="stChatMessage"]{animation:rise var(--slow) var(--ease) both;}
@keyframes rise{from{opacity:0; transform:translateY(10px);} to{opacity:1; transform:none;}}
@keyframes fade{from{opacity:0;} to{opacity:1;}}
@keyframes riseSm{from{opacity:0; transform:translateY(6px);} to{opacity:1; transform:none;}}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#090A11,#07080D); border-right:1px solid var(--line);}
[data-testid="stSidebar"] .block-container{padding-top:1.25rem;}
.dm-logo{font-family:var(--disp); font-weight:700; font-size:1.32rem; letter-spacing:.12em; color:var(--text);}
.dm-logo .g{background:var(--g-vm); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;}
.dm-logo-sub{font-family:var(--mono); font-size:.6rem; letter-spacing:.4em; color:var(--muted2); margin-top:.35rem;}
.dm-side-label{font-family:var(--mono); font-size:.58rem; letter-spacing:.26em; color:var(--muted2); margin:1.1rem 0 .5rem;}
.dm-navactive{display:flex; align-items:center; gap:.6rem; padding:.55rem .7rem; margin:.12rem 0; border-radius:10px;
  color:var(--text); font-weight:600; font-size:.9rem; background:linear-gradient(90deg, rgba(99,102,241,.16), rgba(99,102,241,.02));
  border:1px solid var(--line); box-shadow:inset 2px 0 0 var(--indigo);}
.dm-navactive .d{width:5px;height:5px;border-radius:50%;background:var(--indigo); box-shadow:0 0 10px 2px rgba(99,102,241,.55);}
[data-testid="stSidebar"] .stButton>button{width:100%; text-align:left; justify-content:flex-start; background:transparent;
  border:1px solid transparent; color:var(--muted); font-weight:500; font-size:.9rem; padding:.5rem .7rem; border-radius:10px;
  transition:all var(--fast) var(--ease);}
[data-testid="stSidebar"] .stButton>button:hover{color:var(--text); background:rgba(148,163,184,.06); transform:translateX(3px);}
.dm-engine{border:1px solid var(--line); border-radius:13px; padding:.8rem .85rem; background:linear-gradient(180deg, rgba(99,102,241,.05), transparent);}
.dm-engine .top{display:flex; align-items:center; justify-content:space-between;}
.dm-engine .name{font-family:var(--mono); font-size:.66rem; letter-spacing:.2em; color:var(--muted);}
.dm-online{display:flex; align-items:center; gap:.4rem; font-family:var(--mono); font-size:.66rem; color:var(--success); letter-spacing:.06em;}
.dm-online .p{width:7px;height:7px;border-radius:50%;background:var(--success); animation:beat 2.4s var(--ease) infinite;}
@keyframes beat{0%,100%{box-shadow:0 0 0 0 rgba(52,211,153,.5);}70%{box-shadow:0 0 0 7px rgba(52,211,153,0);}}
.dm-engine .row{display:flex; justify-content:space-between; font-size:.78rem; color:var(--muted); padding:.28rem 0;
  border-top:1px solid var(--line); margin-top:.5rem;}
.dm-engine .row b{color:var(--text); font-weight:500; font-family:var(--mono); font-size:.72rem;}
.dm-profile{display:flex; align-items:center; gap:.6rem; padding-top:.3rem;}
.dm-avatar{width:32px;height:32px;border-radius:9px; background:var(--g-vm); flex:0 0 auto; box-shadow:0 0 18px -6px rgba(217,70,239,.6);}
.dm-profile .who{font-size:.82rem; color:var(--text); font-weight:600;}
.dm-profile .plan{font-family:var(--mono); font-size:.58rem; letter-spacing:.14em; color:var(--muted2);}

/* ---------- intro / hero ---------- */
.dm-intro{border:1px solid var(--line); border-radius:18px; padding:1.15rem 1.3rem;
  background:linear-gradient(180deg, rgba(139,92,246,.06), rgba(14,16,24,.4)); animation:riseSm var(--slow) var(--ease) both;}
.dm-intro h3{font-size:1.02rem; margin:0 0 .2rem;}
.dm-intro .sub{color:var(--muted); font-size:.9rem; margin-bottom:.55rem;}
.dm-intro p{color:var(--muted); font-size:.86rem; line-height:1.55; margin:.1rem 0;}
.dm-flow{display:flex; flex-wrap:wrap; align-items:center; gap:.35rem; margin:.7rem 0 .2rem;}
.dm-flow .s{font-family:var(--mono); font-size:.64rem; letter-spacing:.04em; color:var(--text); padding:.28rem .55rem;
  border:1px solid var(--line); border-radius:8px; background:rgba(99,102,241,.06);}
.dm-flow .a{color:var(--muted2);}
.dm-note{font-size:.78rem; color:var(--muted2); border-left:2px solid var(--violet); padding-left:.7rem; margin-top:.7rem; line-height:1.5;}
.dm-builtfor{font-family:var(--mono); font-size:.68rem; letter-spacing:.05em; color:var(--muted); margin-top:.6rem;}
.dm-builtfor b{color:var(--text); font-weight:500;}

.dm-kicker{font-family:var(--mono); font-size:.66rem; letter-spacing:.36em; color:var(--violet); display:flex; align-items:center; gap:.55rem;}
.dm-kicker::before{content:""; width:24px; height:1px; background:var(--g-vm);}
.dm-h1{font-family:var(--disp); font-weight:700; font-size:2.75rem; line-height:1.03; letter-spacing:-.025em; margin:.45rem 0 .25rem;}
.dm-h1 .g{background:var(--g-iv); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;}
.dm-lede{color:var(--muted); font-size:.98rem; max-width:30rem; line-height:1.55;}

/* ---------- 3D KNOWLEDGE CORE ---------- */
.kc-scene{width:100%; height:270px; display:flex; align-items:center; justify-content:center; perspective:1000px; overflow:hidden;}
.kc{position:relative; width:210px; height:210px; transform-style:preserve-3d; animation:kcFloat 9s var(--ease) infinite;}
@keyframes kcFloat{0%,100%{transform:rotateX(6deg) translateY(-4px);}50%{transform:rotateX(6deg) translateY(6px);}}
.kc-orb{position:absolute; inset:64px; border-radius:50%;
  background:radial-gradient(circle at 36% 30%, rgba(167,139,250,.95), rgba(99,102,241,.55) 42%, rgba(11,13,20,.15) 72%);
  box-shadow:0 0 70px -12px rgba(139,92,246,.7), inset 0 0 46px rgba(217,70,239,.28); animation:kcBreathe 4.5s var(--ease) infinite;}
@keyframes kcBreathe{0%,100%{opacity:.9; transform:scale(1);}50%{opacity:1; transform:scale(1.05);}}
.kc-core{position:absolute; inset:92px; border-radius:50%; background:radial-gradient(circle,#F5F3FF,#8B5CF6 60%,transparent);
  box-shadow:0 0 26px 4px rgba(217,70,239,.6);}
.kc-wrap{position:absolute; inset:0; transform-style:preserve-3d;}
.kc-wrap.w1{transform:rotateX(74deg);} .kc-wrap.w2{transform:rotateX(74deg) rotateY(60deg);} .kc-wrap.w3{transform:rotateX(26deg) rotateY(-42deg);}
.kc-ring{position:absolute; inset:0; border-radius:50%; border:1px solid rgba(99,102,241,.4); animation:spinZ 16s linear infinite;}
.kc-ring.c{border-color:rgba(34,211,238,.3); animation-duration:22s; animation-direction:reverse;}
.kc-ring.m{border-color:rgba(217,70,239,.28); animation-duration:26s;}
@keyframes spinZ{to{transform:rotateZ(360deg);}}
.kc-dot{position:absolute; top:-3px; left:50%; width:6px; height:6px; border-radius:50%; background:var(--cyan); box-shadow:0 0 10px 2px rgba(34,211,238,.7);}
.kc-frag{position:absolute; width:16px; height:20px; border-radius:3px; background:rgba(99,102,241,.14); border:1px solid rgba(139,92,246,.5);
  box-shadow:0 0 12px -3px rgba(139,92,246,.6); animation:fragFloat 6s var(--ease) infinite;}
@keyframes fragFloat{0%,100%{transform:translateZ(40px) translateY(0);}50%{transform:translateZ(40px) translateY(-10px);}}
.kc-particle{position:absolute; width:3px; height:3px; border-radius:50%; background:var(--cyan); opacity:0; animation:pIn 3.6s linear infinite;}
@keyframes pIn{0%{opacity:0; transform:translate(var(--fx),var(--fy)) scale(.6);}25%{opacity:1;}100%{opacity:0; transform:translate(0,0) scale(1);}}

/* ---------- chips / tools / suggestions ---------- */
.dm-chips{display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.7rem;}
.dm-chip{font-family:var(--mono); font-size:.66rem; color:var(--muted); border:1px solid var(--line); border-radius:999px;
  padding:.3rem .65rem; background:rgba(148,163,184,.03); transition:all var(--fast) var(--ease);}
.dm-chip .d{color:var(--violet);}
.dm-eyebrow{font-family:var(--mono); font-size:.6rem; letter-spacing:.26em; color:var(--muted2); margin:.2rem 0 .5rem;}
.dm-emptybig{font-family:var(--disp); font-size:1.32rem; font-weight:600;}
.dm-emptysub{color:var(--muted); font-size:.9rem; margin-top:.2rem;}

/* generic buttons (magnetic hover, glow when primary) */
.stButton>button{border-radius:11px; font-weight:600; transition:transform var(--fast) var(--ease), box-shadow var(--normal) var(--ease), border-color var(--fast), background var(--fast);}
.stButton>button:hover{transform:translateY(-2px);}
.stButton>button[kind="primary"]{background:var(--g-iv); color:#0A0B12; border:none; box-shadow:0 12px 30px -14px rgba(99,102,241,.8);}
.stButton>button[kind="primary"]:hover{box-shadow:0 16px 40px -12px rgba(139,92,246,.85); filter:brightness(1.05);}
/* suggestion / tool buttons pick up the global secondary style */
.stButton>button:not([kind="primary"]){background:linear-gradient(180deg, rgba(22,25,38,.7), rgba(14,16,24,.5)); border:1px solid var(--line); color:var(--text);}
.stButton>button:not([kind="primary"]):hover{border-color:var(--line2); box-shadow:0 10px 30px -16px rgba(139,92,246,.7);}

/* chat input — animated focus ring */
[data-testid="stChatInput"] textarea{color:var(--text) !important;}
div[data-baseweb="textarea"], [data-testid="stChatInput"] > div{background:var(--surface2) !important; border:1px solid var(--line) !important;
  border-radius:14px !important; transition:border-color var(--normal) var(--ease), box-shadow var(--normal) var(--ease);}
div[data-baseweb="textarea"]:focus-within, [data-testid="stChatInput"] > div:focus-within{border-color:var(--line2) !important;
  box-shadow:0 0 0 1px rgba(99,102,241,.4), 0 0 40px -14px rgba(139,92,246,.9) !important;}

/* expanders / uploader / containers */
[data-testid="stExpander"]{border:1px solid var(--line); border-radius:12px; background:rgba(14,16,24,.5); transition:border-color var(--fast);}
[data-testid="stExpander"]:hover{border-color:var(--line2);}
[data-testid="stExpander"] summary{color:var(--muted); font-family:var(--mono); font-size:.76rem;}
[data-testid="stFileUploaderDropzone"]{background:linear-gradient(180deg, rgba(99,102,241,.05), rgba(14,16,24,.4));
  border:1.4px dashed var(--line2); border-radius:14px; color:var(--muted); transition:all var(--normal) var(--ease);}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--violet); background:rgba(139,92,246,.07);}
[data-testid="stVerticalBlockBorderWrapper"]{border:1px solid var(--line) !important; border-radius:16px !important;
  background:linear-gradient(180deg, rgba(16,18,28,.6), rgba(9,11,18,.45)) !important; padding:.35rem !important;}

/* answer + sources */
.dm-answerlabel{font-family:var(--mono); font-size:.6rem; letter-spacing:.28em; color:var(--violet); margin:.2rem 0 .15rem;}
.dm-meta{display:flex; flex-wrap:wrap; gap:1.1rem; font-family:var(--mono); font-size:.7rem; color:var(--muted);
  margin:.7rem 0 .3rem; animation:fade var(--slow) var(--ease) both;}
.dm-meta b{color:var(--cyan); font-weight:500;} .dm-meta .ok{color:var(--success);}
.dm-srchead{font-family:var(--mono); font-size:.6rem; letter-spacing:.28em; color:var(--muted2); margin:.4rem 0 .5rem;}
.dm-relbar{height:3px; border-radius:2px; background:rgba(148,163,184,.14); margin-top:.4rem; overflow:hidden;}
.dm-relbar > i{display:block; height:100%; background:var(--g-vm); animation:grow var(--slow) var(--ease) both;}
@keyframes grow{from{width:0 !important;}}

/* doc rows / stats */
.dm-doc{display:flex; align-items:center; gap:.7rem; padding:.55rem .35rem; border-top:1px solid var(--line);
  border-radius:9px; transition:transform var(--fast) var(--ease), background var(--fast);}
.dm-doc:hover{transform:translateY(-2px); background:rgba(99,102,241,.05);}
.dm-doc .ic{width:30px;height:30px;border-radius:8px; flex:0 0 auto; border:1px solid var(--line);
  display:flex; align-items:center; justify-content:center; font-family:var(--mono); font-size:.52rem; font-weight:600; color:#0A0B12;}
.dm-doc .fn{color:var(--text); font-size:.85rem; font-weight:500; word-break:break-all;}
.dm-doc .mt{font-family:var(--mono); font-size:.64rem; color:var(--muted); margin-top:.12rem; display:flex; align-items:center; gap:.35rem;}
.dm-doc .st{width:6px;height:6px;border-radius:50%; background:var(--success); box-shadow:0 0 8px 1px rgba(52,211,153,.6);}
.dm-doc .yours{color:var(--magenta);}
.dm-stat{border:1px solid var(--line); border-radius:12px; padding:.75rem .9rem; background:linear-gradient(180deg, rgba(22,25,38,.55), transparent);}
.dm-stat .n{font-family:var(--disp); font-weight:700; font-size:1.55rem; line-height:1; background:var(--g-iv);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;}
.dm-stat .l{font-family:var(--mono); font-size:.58rem; letter-spacing:.16em; color:var(--muted2); margin-top:.35rem;}

/* ---------- pipeline (answering) — vertical animated nodes ---------- */
.dm-pipe{display:flex; flex-direction:column; gap:0; margin:.2rem 0 .6rem; padding-left:.2rem;}
.dm-pn{display:flex; align-items:center; gap:.6rem; opacity:.3; animation:pnOn .45s var(--ease) forwards;}
.dm-pn .o{width:12px;height:12px;border-radius:50%; border:1px solid var(--indigo); background:var(--surface); position:relative; flex:0 0 auto;}
.dm-pn .o::after{content:""; position:absolute; inset:3px; border-radius:50%; background:var(--indigo); transform:scale(0); animation:dotOn .4s var(--ease) forwards; animation-delay:inherit;}
.dm-pn .t{font-family:var(--mono); font-size:.66rem; letter-spacing:.1em; color:var(--text);}
.dm-pn .sub{font-family:var(--mono); font-size:.6rem; color:var(--muted2); margin-left:.4rem;}
.dm-pl{width:1px; height:12px; margin-left:6px; background:linear-gradient(var(--indigo),transparent); opacity:0; animation:fade .4s var(--ease) forwards;}
@keyframes pnOn{to{opacity:1;}}
@keyframes dotOn{to{transform:scale(1);}}

/* ---------- upload animation ---------- */
.dm-up{border:1px solid var(--line2); border-radius:14px; padding:1rem; background:radial-gradient(500px 200px at 50% 120%, rgba(139,92,246,.14), transparent);
  position:relative; overflow:hidden;}
.dm-up .sheet{width:34px;height:44px;border-radius:5px; background:rgba(99,102,241,.16); border:1px solid rgba(139,92,246,.6);
  box-shadow:0 0 18px -4px rgba(139,92,246,.7); margin:0 auto; animation:sheetIn 1.4s var(--ease) forwards;}
@keyframes sheetIn{0%{transform:translateY(30px) scale(.8); opacity:0;}40%{opacity:1;}100%{transform:translateY(-4px) scale(1); opacity:1;}}
.dm-up .steps{display:flex; justify-content:center; gap:.35rem; margin-top:.9rem; flex-wrap:wrap;}
.dm-up .stp{font-family:var(--mono); font-size:.62rem; letter-spacing:.08em; color:var(--muted2); padding:.28rem .55rem;
  border:1px solid var(--line); border-radius:8px; opacity:.35; animation:stpOn .5s var(--ease) forwards;}
.dm-up .stp b{color:var(--cyan);}
@keyframes stpOn{to{opacity:1; border-color:var(--line2); color:var(--text);}}

@media (prefers-reduced-motion: reduce){
  *{animation-duration:.001ms !important; animation-iteration-count:1 !important;}
  .kc-ring,.kc-orb,.kc,.dm-online .p{animation:none !important;}
}
</style>"""


def knowledge_core() -> str:
    """Idle 3D knowledge core (pure CSS 3D, continuous subtle motion)."""
    frags = "".join(
        f'<div class="kc-frag" style="top:{t}px;left:{l}px;animation-delay:{d}s"></div>'
        for t, l, d in [(30, 40, 0), (48, 150, 1.4), (150, 30, 2.1), (150, 160, 3.0)])
    parts = "".join(
        f'<div class="kc-particle" style="top:{t}px;left:{l}px;--fx:{fx}px;--fy:{fy}px;animation-delay:{d}s"></div>'
        for t, l, fx, fy, d in [(40, 20, 60, 55, .2), (170, 30, 55, -55, 1.1),
                                (30, 180, -60, 60, 1.9), (180, 175, -55, -55, 2.7)])
    return f"""
<div class="kc-scene"><div class="kc">
  <div class="kc-wrap w1"><div class="kc-ring"><div class="kc-dot"></div></div></div>
  <div class="kc-wrap w2"><div class="kc-ring c"><div class="kc-dot" style="background:var(--magenta);box-shadow:0 0 10px 2px rgba(217,70,239,.7)"></div></div></div>
  <div class="kc-wrap w3"><div class="kc-ring m"></div></div>
  <div class="kc-orb"></div><div class="kc-core"></div>
  {frags}{parts}
</div></div>"""


def upload_animation() -> str:
    """One-shot CSS sequence played while the real add_documents() runs."""
    steps = [("READING", 0.2), ("PARSE", 0.6), ("CHUNK", 1.0), ("INDEX", 1.4), ("READY", 1.9)]
    chips = "".join(
        f'<span class="stp" style="animation-delay:{d}s">{"<b>"+t+"</b>" if t=="READY" else t}</span>'
        for t, d in steps)
    return (f'<div class="dm-up"><div class="sheet"></div>'
            f'<div class="steps">{chips}</div></div>')


_STAGES = [("QUERY", "your question"), ("RETRIEVE", "hybrid dense + BM25"),
           ("RERANK", "cross-encoder"), ("GENERATE", "grounded LLM"),
           ("CITE", "source passages")]


def rag_pipeline() -> str:
    """One-shot vertical animated pipeline played while the real query runs."""
    rows = []
    for i, (name, sub) in enumerate(_STAGES):
        d = i * 0.55
        rows.append(
            f'<div class="dm-pn" style="animation-delay:{d}s">'
            f'<span class="o" style="animation-delay:{d}s"></span>'
            f'<span class="t">{name}</span><span class="sub">{sub}</span></div>')
        if i < len(_STAGES) - 1:
            rows.append(f'<div class="dm-pl" style="animation-delay:{d+.3}s"></div>')
    return f'<div class="dm-pipe">{"".join(rows)}</div>'
