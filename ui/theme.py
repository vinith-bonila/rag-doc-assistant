"""DocMind design system — dual palette (dark default, light option).

One place for the visual language, so presentation never leaks into retrieval
logic. Exposes:

* ``css(mode)``           — design tokens + scoped component styles.
* ``flow_diagram(mode)``  — the "how DocMind works" chart (inline SVG).
* small builders for the pieces that need structure (badges, doc rows, metrics).

Why the app owns the theme
--------------------------
Streamlit applies its own light/dark choice without exposing a DOM marker or CSS
custom properties, so stylesheet-only theming cannot follow it. Instead the app
keeps the mode in session state and this module renders the matching token
values; ``.streamlit/config.toml`` carries equivalent ``[theme.dark]`` /
``[theme.light]`` palettes so Streamlit's own chrome agrees on first paint.

Rules this file follows (see docs/SECURITY_AUDIT.md DM-02):

* Every builder escapes the values it interpolates via ``esc``. Nothing puts a
  filename or document text into markup unescaped.
* Selectors are prefixed ``dm-`` or scoped to a stable Streamlit test id, never
  to a generated class name.
* The layout works with the webfont unavailable: Inter is requested, but every
  rule falls back to a system sans-serif stack.
* Motion is minimal and disabled under ``prefers-reduced-motion``.
"""
from __future__ import annotations

from html import escape as _escape

MODES = ("dark", "light")
DEFAULT_MODE = "dark"

# --- palettes -----------------------------------------------------------------
# The violet accent is shared so the brand reads the same in both modes; it is
# lightened in dark mode to hold contrast against a near-black surface.
PALETTES = {
    "light": {
        "bg": "#F8FAFC",
        "surface": "#FFFFFF",
        "surface_alt": "#F1F5F9",
        "text": "#172033",
        # 5.6:1 on white, 5.1:1 on the alt surface.
        "muted": "#5C687F",
        # 5.1:1 on white, 4.7:1 on the alt surface — the lightest tone that
        # still clears AA for the 12.5px copy it is used on.
        "faint": "#616F86",
        "accent": "#635BFF",
        "accent_hover": "#4F46E5",
        "accent_soft": "#EEEDFF",
        "accent_text": "#635BFF",
        "border": "#E2E8F0",
        "border_strong": "#CBD5E1",
        "success": "#0F766E",
        "success_soft": "#ECFDF5",
        "success_border": "#A7F3D0",
        "warning": "#B45309",
        "warning_soft": "#FFFBEB",
        "warning_border": "#FDE68A",
        "error": "#B42318",
        "error_soft": "#FEF2F2",
        "error_border": "#FECACA",
        "code": "#3730A3",
        "shadow": "0 1px 2px rgba(15,23,42,.04), 0 1px 3px rgba(15,23,42,.06)",
        # flow diagram
        "svg_box": "#FFFFFF",
        "svg_stroke": "#E2E8F0",
        "svg_line": "#CBD5E1",
        "svg_title": "#172033",
        "svg_sub": "#64748B",
        "svg_badge": "#EEEDFF",
        "svg_badge_text": "#635BFF",
    },
    "dark": {
        "bg": "#0B0F19",
        "surface": "#121826",
        "surface_alt": "#1A2234",
        "text": "#E8ECF5",
        "muted": "#9AA6BF",
        # 5.1:1 on the dark surface, 4.6:1 on the alt surface — AA for 12.5px copy.
        "faint": "#7D8AA6",
        "accent": "#8B85FF",
        "accent_hover": "#A29BFF",
        "accent_soft": "#232246",
        "accent_text": "#A9A3FF",
        "border": "#232B3D",
        "border_strong": "#344056",
        "success": "#5EEAD4",
        "success_soft": "#0C2B2A",
        "success_border": "#1B4F4A",
        "warning": "#FBBF68",
        "warning_soft": "#2A2010",
        "warning_border": "#4D3A17",
        "error": "#FB8A82",
        "error_soft": "#2B1413",
        "error_border": "#542422",
        "code": "#B9B2FF",
        "shadow": "0 1px 2px rgba(0,0,0,.35), 0 1px 3px rgba(0,0,0,.25)",
        # flow diagram
        "svg_box": "#161D2E",
        "svg_stroke": "#2A344A",
        "svg_line": "#3B475F",
        "svg_title": "#E8ECF5",
        "svg_sub": "#9AA6BF",
        "svg_badge": "#262552",
        "svg_badge_text": "#A9A3FF",
    },
}


def esc(value: object) -> str:
    """HTML-escape any value destined for markup, quotes included.

    Document names and metadata are attacker-controlled (a file can be named
    ``<img src=x onerror=...>.pdf``), and Streamlit's ``unsafe_allow_html``
    renders markup verbatim, so every interpolation goes through this.
    """
    return _escape(str(value), quote=True)


def _normalize(mode: str) -> str:
    return mode if mode in PALETTES else DEFAULT_MODE


_TEMPLATE = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root{{
  --dm-bg:{bg};
  --dm-surface:{surface};
  --dm-surface-alt:{surface_alt};
  --dm-text:{text};
  --dm-muted:{muted};
  --dm-faint:{faint};
  --dm-accent:{accent};
  --dm-accent-hover:{accent_hover};
  --dm-accent-soft:{accent_soft};
  --dm-accent-text:{accent_text};
  --dm-border:{border};
  --dm-border-strong:{border_strong};
  --dm-success:{success};
  --dm-success-soft:{success_soft};
  --dm-success-border:{success_border};
  --dm-warning:{warning};
  --dm-warning-soft:{warning_soft};
  --dm-warning-border:{warning_border};
  --dm-error:{error};
  --dm-error-soft:{error_soft};
  --dm-error-border:{error_border};
  --dm-code:{code};
  --dm-on-accent:{on_accent};
  --dm-shadow:{shadow};
  --dm-sans:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;
  --dm-mono:ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,monospace;
  --dm-radius:10px;
  /* Streamlit's app header is absolutely positioned, opaque and sits at
     z-index 999990, so main content must start below it. Measured at 56px. */
  --dm-header-h:56px;
}}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"]{{
  background:var(--dm-bg);
  color:var(--dm-text);
  font-family:var(--dm-sans);
  font-size:15px;
  line-height:1.6;
}}
[data-testid="stHeader"]{{background:var(--dm-bg); border-bottom:1px solid var(--dm-border);}}
footer{{display:none;}}

/* Readable measure: wide enough for tables, narrow enough to read comfortably.
   padding-top must clear the fixed header: overriding Streamlit's own
   header-clearing padding with a flat value painted the page title underneath
   it (docs/SECURITY_AUDIT.md UI-04). */
.block-container, [data-testid="stMainBlockContainer"]{{
  max-width:1080px;
  padding-top:calc(var(--dm-header-h) + 1.25rem);
  padding-bottom:5rem;
}}

h1,h2,h3,h4,h5,p,li,label,span,div{{color:inherit;}}
h1,h2,h3,h4,h5{{font-family:var(--dm-sans); color:var(--dm-text); letter-spacing:-.015em;}}
a{{color:var(--dm-accent-text); text-decoration:none;}}
a:hover{{color:var(--dm-accent-hover); text-decoration:underline;}}
hr{{border-color:var(--dm-border);}}

/* ---------- typography scale ---------- */
.dm-title{{font-size:30px; font-weight:700; line-height:1.2; margin:0 0 .35rem; letter-spacing:-.02em;}}
.dm-subtitle{{font-size:15px; color:var(--dm-muted); margin:0 0 1.1rem; max-width:62ch;}}
.dm-h2{{font-size:20px; font-weight:600; margin:0 0 .3rem;}}
.dm-h3{{font-size:16px; font-weight:600; margin:0 0 .25rem;}}
.dm-help{{font-size:13.5px; color:var(--dm-muted); line-height:1.55;}}
.dm-metaline{{font-size:12.5px; color:var(--dm-faint);}}
.dm-label{{font-size:12.5px; font-weight:600; color:var(--dm-muted); margin:0 0 .4rem;}}

/* ---------- surfaces ---------- */
.dm-card{{background:var(--dm-surface); border:1px solid var(--dm-border);
  border-radius:var(--dm-radius); padding:1.1rem 1.25rem; box-shadow:var(--dm-shadow);}}
.dm-divider{{height:1px; background:var(--dm-border); margin:1.3rem 0;}}
[data-testid="stVerticalBlockBorderWrapper"]{{border-color:var(--dm-border) !important;}}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{{background:var(--dm-surface); border-right:1px solid var(--dm-border);}}
[data-testid="stSidebar"] .block-container,
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{{padding-top:1.4rem;}}
[data-testid="stSidebarCollapseButton"] button{{color:var(--dm-muted);}}
.dm-brand{{display:flex; align-items:center; gap:.55rem; margin-bottom:.15rem;}}
.dm-brand-mark{{width:26px; height:26px; border-radius:7px; background:var(--dm-accent);
  color:var(--dm-on-accent); font-size:14px; font-weight:700; display:flex; align-items:center;
  justify-content:center; flex:0 0 auto;}}
.dm-brand-name{{font-size:17px; font-weight:700; letter-spacing:-.01em; color:var(--dm-text);}}
.dm-brand-sub{{font-size:12.5px; color:var(--dm-muted); margin:0 0 1rem .05rem;}}
.dm-sidebar-heading{{font-size:11.5px; font-weight:600; letter-spacing:.07em;
  text-transform:uppercase; color:var(--dm-faint); margin:1.1rem 0 .45rem;}}

/* ---------- status pill ---------- */
.dm-status{{display:inline-flex; align-items:center; gap:.4rem; font-size:12.5px;
  color:var(--dm-muted); border:1px solid var(--dm-border); border-radius:999px;
  padding:.2rem .6rem; background:var(--dm-surface-alt);}}
.dm-status .dot{{width:7px; height:7px; border-radius:50%; flex:0 0 auto;}}
.dm-status .dot.on{{background:var(--dm-success);}}
.dm-status .dot.off{{background:var(--dm-faint);}}

/* ---------- badges: status is never colour-only (each carries a glyph + word) ---- */
.dm-badge{{display:inline-flex; align-items:center; gap:.35rem; font-size:12.5px;
  font-weight:500; border-radius:6px; padding:.2rem .5rem; border:1px solid transparent;}}
.dm-badge .g{{font-weight:700; font-size:12px;}}
.dm-badge.ok{{background:var(--dm-success-soft); color:var(--dm-success); border-color:var(--dm-success-border);}}
.dm-badge.warn{{background:var(--dm-warning-soft); color:var(--dm-warning); border-color:var(--dm-warning-border);}}
.dm-badge.err{{background:var(--dm-error-soft); color:var(--dm-error); border-color:var(--dm-error-border);}}
.dm-badge.neutral{{background:var(--dm-surface-alt); color:var(--dm-muted); border-color:var(--dm-border);}}
.dm-badgerow{{display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; margin:.5rem 0 .2rem;}}

/* ---------- answer ---------- */
.dm-answer-meta{{display:flex; flex-wrap:wrap; gap:.3rem .9rem; font-size:12.5px;
  color:var(--dm-muted); margin:.1rem 0 .5rem;}}
.dm-answer-meta b{{color:var(--dm-text); font-weight:600;}}
[data-testid="stChatMessage"]{{background:transparent; padding:.35rem 0;}}
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"]{{
  background:var(--dm-surface-alt); color:var(--dm-muted);}}

/* ---------- document rows ---------- */
.dm-doc{{display:flex; align-items:center; gap:.75rem; padding:.7rem .85rem;
  border:1px solid var(--dm-border); border-radius:var(--dm-radius);
  background:var(--dm-surface); margin-bottom:.45rem;}}
.dm-doc .type{{font-size:10.5px; font-weight:700; letter-spacing:.04em; color:var(--dm-muted);
  background:var(--dm-surface-alt); border:1px solid var(--dm-border); border-radius:5px;
  padding:.25rem .4rem; min-width:42px; text-align:center; flex:0 0 auto;}}
.dm-doc .body{{min-width:0; flex:1 1 auto; display:block;}}
/* Long names truncate instead of pushing the row wide. */
.dm-doc .name{{display:block; font-size:14px; font-weight:500; color:var(--dm-text);
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}}
.dm-doc .sub{{display:block; font-size:12.5px; color:var(--dm-muted); overflow:hidden;
  text-overflow:ellipsis; white-space:nowrap;}}

/* ---------- metric tiles ---------- */
.dm-metric{{background:var(--dm-surface); border:1px solid var(--dm-border);
  border-radius:var(--dm-radius); padding:.8rem .9rem;}}
.dm-metric .v{{font-size:22px; font-weight:700; line-height:1.2; color:var(--dm-text);}}
.dm-metric .k{{font-size:12.5px; color:var(--dm-muted); margin-top:.1rem;}}

/* ---------- key/value rows (technical detail) ---------- */
.dm-kv{{display:flex; justify-content:space-between; gap:1rem; font-size:13.5px;
  padding:.4rem 0; border-bottom:1px solid var(--dm-border);}}
.dm-kv:last-child{{border-bottom:none;}}
.dm-kv .k{{color:var(--dm-muted);}}
.dm-kv .v{{color:var(--dm-text); font-family:var(--dm-mono); font-size:12.5px;
  text-align:right; word-break:break-word;}}

/* ---------- empty state ---------- */
.dm-empty{{background:var(--dm-surface); border:1px solid var(--dm-border);
  border-radius:12px; padding:1.5rem 1.6rem;}}
.dm-empty h2{{font-size:20px; font-weight:600; margin:0 0 .35rem; color:var(--dm-text);}}
.dm-steps{{margin:.9rem 0 0; padding-left:1.1rem; color:var(--dm-muted); font-size:13.5px;}}
.dm-steps li{{margin:.25rem 0;}}
.dm-privacy{{font-size:13px; color:var(--dm-muted); background:var(--dm-surface-alt);
  border-left:3px solid var(--dm-accent); border-radius:0 6px 6px 0;
  padding:.65rem .8rem; margin-top:1rem;}}
.dm-privacy b{{color:var(--dm-text);}}

/* ---------- flow diagram ----------
   Two renderings of the same six steps: the chart on screens wide enough to
   read it, and a plain numbered list below that, so a phone gets the content
   upright instead of a side-scrolling box. */
.dm-flow{{width:100%; padding:.2rem 0;}}
.dm-flow svg{{display:block; width:100%; height:auto;}}
.dm-flow-list{{display:none; margin:.2rem 0 0; padding:0; list-style:none; counter-reset:dmstep;}}
.dm-flow-list li{{position:relative; padding:.5rem 0 .5rem 2.1rem; border-bottom:1px solid var(--dm-border);}}
.dm-flow-list li:last-child{{border-bottom:none;}}
.dm-flow-list li::before{{counter-increment:dmstep; content:counter(dmstep);
  position:absolute; left:0; top:.55rem; width:1.4rem; height:1.4rem; border-radius:50%;
  background:var(--dm-accent-soft); color:var(--dm-accent-text); font-size:11.5px;
  font-weight:700; display:flex; align-items:center; justify-content:center;}}
.dm-flow-list b{{display:block; font-size:14px; font-weight:600; color:var(--dm-text);}}
.dm-flow-list span{{font-size:13px; color:var(--dm-muted);}}
@media (max-width:720px){{
  .dm-flow{{display:none;}}
  .dm-flow-list{{display:block;}}
}}

/* ---------- Streamlit widget restyling (stable test ids only) ---------- */
.stButton>button{{border-radius:8px; font-size:14px; font-weight:500;
  border:1px solid var(--dm-border); background:var(--dm-surface); color:var(--dm-text);
  transition:background 120ms ease, border-color 120ms ease;}}
.stButton>button:hover{{border-color:var(--dm-border-strong); background:var(--dm-surface-alt);
  color:var(--dm-text);}}
.stButton>button[kind="primary"]{{background:var(--dm-accent); border-color:var(--dm-accent);
  color:{on_accent};}}
.stButton>button[kind="primary"]:hover{{background:var(--dm-accent-hover);
  border-color:var(--dm-accent-hover); color:{on_accent};}}
.stButton>button:disabled, .stButton>button:disabled:hover{{opacity:.5;}}
/* Keep focus visible for keyboard users. */
.stButton>button:focus-visible, summary:focus-visible, input:focus-visible,
textarea:focus-visible, [role="option"]:focus-visible, a:focus-visible{{
  outline:2px solid var(--dm-accent); outline-offset:2px;}}

[data-testid="stSidebar"] .stButton>button{{width:100%; justify-content:flex-start;
  text-align:left; border-color:transparent; background:transparent; color:var(--dm-muted);
  font-weight:500; padding:.42rem .6rem;}}
[data-testid="stSidebar"] .stButton>button:hover{{background:var(--dm-surface-alt); color:var(--dm-text);}}
[data-testid="stSidebar"] .stButton>button[kind="primary"]{{
  background:var(--dm-accent-soft); color:var(--dm-accent-text); border-color:transparent;
  font-weight:600;}}
[data-testid="stSidebar"] .stButton>button[kind="primary"]:hover{{
  background:var(--dm-accent-soft); color:var(--dm-accent-text);}}

[data-testid="stExpander"]{{border:1px solid var(--dm-border); border-radius:var(--dm-radius);
  background:var(--dm-surface); margin-bottom:.4rem;}}
[data-testid="stExpander"] summary{{font-size:13.5px; font-weight:500; color:var(--dm-text);}}
[data-testid="stExpander"] summary:hover{{color:var(--dm-accent-text);}}
[data-testid="stExpander"] svg{{fill:var(--dm-muted);}}

[data-testid="stFileUploaderDropzone"]{{background:var(--dm-surface-alt);
  border:1px dashed var(--dm-border-strong); border-radius:var(--dm-radius);
  color:var(--dm-muted);}}
[data-testid="stFileUploaderDropzone"] button{{background:var(--dm-surface);
  border:1px solid var(--dm-border-strong); color:var(--dm-text);}}
[data-testid="stFileUploaderFile"]{{color:var(--dm-muted);}}

[data-testid="stChatInput"]{{background:var(--dm-surface); border:1px solid var(--dm-border);}}
[data-testid="stChatInput"] textarea{{font-size:15px; color:var(--dm-text);}}
[data-testid="stChatInput"] textarea::placeholder{{color:var(--dm-faint);}}

/* text inputs, multiselect, select */
[data-baseweb="input"], [data-baseweb="select"]>div, [data-baseweb="popover"] li{{
  background:var(--dm-surface) !important; color:var(--dm-text) !important;
  border-color:var(--dm-border) !important;}}
[data-baseweb="tag"]{{background:var(--dm-accent-soft) !important;
  color:var(--dm-accent-text) !important;}}
[data-testid="stWidgetLabel"] p{{color:var(--dm-muted); font-size:13.5px;}}

/* st.text / st.code blocks used to show retrieved passages */
pre, code, [data-testid="stCode"], .stCodeBlock pre{{
  background:var(--dm-surface-alt) !important; color:var(--dm-text) !important;
  border-radius:6px;}}
code{{color:var(--dm-code) !important; font-size:13px; padding:.08em .32em;}}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p{{color:var(--dm-muted);}}
[data-testid="stMarkdownContainer"] table{{font-size:13.5px;}}
[data-testid="stMarkdownContainer"] th, [data-testid="stMarkdownContainer"] td{{
  border-color:var(--dm-border);}}
[data-testid="stStatusWidget"], [data-testid="stStatus"]{{background:var(--dm-surface);}}

/* ---------- responsive: tablet and narrow screens ---------- */
@media (max-width:1100px){{
  .block-container{{max-width:100%; padding-left:1.1rem; padding-right:1.1rem;}}
}}
@media (max-width:760px){{
  .block-container{{padding-left:.8rem; padding-right:.8rem;}}
  .dm-title{{font-size:24px;}}
  .dm-card, .dm-empty{{padding:1rem;}}
  .dm-kv{{flex-direction:column; gap:.1rem;}}
  .dm-kv .v{{text-align:left;}}
  /* Columns stack; make sure nothing forces a horizontal page scroll. */
  [data-testid="stHorizontalBlock"]{{flex-wrap:wrap;}}
  [data-testid="stHorizontalBlock"]>div{{min-width:min(100%, 260px);}}
}}

@media (prefers-reduced-motion:reduce){{
  *, *::before, *::after{{animation-duration:.001ms !important; transition-duration:.001ms !important;}}
}}
</style>"""


def css(mode: str = DEFAULT_MODE) -> str:
    """The full stylesheet for ``mode`` ("dark" or "light")."""
    mode = _normalize(mode)
    palette = dict(PALETTES[mode])
    # Text drawn on top of the accent fill: near-black on the light violet used
    # in dark mode, white on the saturated violet used in light mode.
    # Text drawn on the accent fill. Checked for AA: #0B0F19 on #8B85FF is
    # 8.9:1; #FFFFFF on #635BFF is 4.9:1.
    palette["on_accent"] = "#0B0F19" if mode == "dark" else "#FFFFFF"
    return _TEMPLATE.format(**palette)


# --- the "how it works" flow chart -------------------------------------------
# Static inline SVG: no user data reaches it, so it is safe under
# unsafe_allow_html. Colours come from the active palette.
_FLOW_STEPS = [
    ("1", "You upload", "a PDF, DOCX, TXT or MD file"),
    ("2", "DocMind reads it", "split into small passages"),
    ("3", "You ask", "a question in plain English"),
    ("4", "It finds evidence", "the passages most likely to answer"),
    ("5", "It writes an answer", "using only those passages"),
    ("6", "You verify", "each claim shows its source"),
]

_BOX_W, _BOX_H, _GAP, _ROW_GAP, _COLS = 172, 94, 24, 50, 3


def _wrap(text: str, width: int = 26) -> list:
    words, line, lines = text.split(), "", []
    for w in words:
        if line and len(line) + len(w) + 1 > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        lines.append(line)
    return lines


def flow_diagram(mode: str = DEFAULT_MODE) -> str:
    """The six-step "how DocMind works" chart as a responsive inline SVG.

    Laid out serpentine (left-to-right, then right-to-left) so it fits a normal
    content width without a tiny single row, and scales down on narrow screens.
    """
    p = PALETTES[_normalize(mode)]
    rows = -(-len(_FLOW_STEPS) // _COLS)
    pad_x, pad_y = 4, 8
    width = pad_x * 2 + _COLS * _BOX_W + (_COLS - 1) * _GAP
    height = pad_y * 2 + rows * _BOX_H + (rows - 1) * _ROW_GAP

    label = ("How DocMind works, in six steps: "
             + "; ".join(f"{n}. {t} — {s}" for n, t, s in _FLOW_STEPS) + ".")
    # Marker ids are mode-suffixed: both variants can coexist on one page
    # without the second one inheriting the first one's arrow colour.
    marker = f"dmArrow-{_normalize(mode)}"
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(label)}" '
        'xmlns="http://www.w3.org/2000/svg">',
        f'<defs><marker id="{marker}" viewBox="0 0 10 10" refX="8" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,1 L9,5 L0,9 z" fill="{p["svg_line"]}"/></marker></defs>',
    ]

    pos = []
    for i in range(len(_FLOW_STEPS)):
        r, c = divmod(i, _COLS)
        if r % 2:                       # serpentine: odd rows read right-to-left
            c = _COLS - 1 - c
        pos.append((pad_x + c * (_BOX_W + _GAP), pad_y + r * (_BOX_H + _ROW_GAP)))

    # Connectors first so the boxes paint over the line ends.
    for i in range(len(_FLOW_STEPS) - 1):
        (x1, y1), (x2, y2) = pos[i], pos[i + 1]
        mid_y = y1 + _BOX_H / 2
        if y1 == y2:
            forward = x2 > x1
            sx = x1 + _BOX_W + 5 if forward else x1 - 5
            ex = x2 - 7 if forward else x2 + _BOX_W + 7
            parts.append(f'<line x1="{sx}" y1="{mid_y}" x2="{ex}" y2="{mid_y}" '
                         f'stroke="{p["svg_line"]}" stroke-width="1.5" '
                         f'marker-end="url(#{marker})"/>')
        else:
            cx = x1 + _BOX_W / 2
            parts.append(f'<path d="M{cx},{y1 + _BOX_H + 5} L{cx},{y2 - 9}" fill="none" '
                         f'stroke="{p["svg_line"]}" stroke-width="1.5" '
                         f'marker-end="url(#{marker})"/>')

    for (x, y), (num, title, sub) in zip(pos, _FLOW_STEPS):
        parts.append(
            f'<rect x="{x}" y="{y}" width="{_BOX_W}" height="{_BOX_H}" rx="10" '
            f'fill="{p["svg_box"]}" stroke="{p["svg_stroke"]}"/>'
            f'<circle cx="{x + 25}" cy="{y + 26}" r="12.5" fill="{p["svg_badge"]}"/>'
            f'<text x="{x + 25}" y="{y + 30}" text-anchor="middle" '
            f'fill="{p["svg_badge_text"]}" font-size="12" font-weight="700" '
            f'font-family="Inter,sans-serif">{num}</text>'
            f'<text x="{x + 46}" y="{y + 31}" fill="{p["svg_title"]}" font-size="13.5" '
            f'font-weight="600" font-family="Inter,sans-serif">{esc(title)}</text>'
        )
        for n, line in enumerate(_wrap(sub)[:2]):
            parts.append(f'<text x="{x + 16}" y="{y + 57 + n * 17}" fill="{p["svg_sub"]}" '
                         f'font-size="12" font-family="Inter,sans-serif">{esc(line)}</text>')

    parts.append("</svg>")
    # The list carries the same six steps for narrow screens; CSS shows exactly
    # one of the two, so the content is never duplicated to a reader.
    items = "".join(
        f'<li><b>{esc(title)}</b><span>{esc(sub_)}</span></li>'
        for _, title, sub_ in _FLOW_STEPS)
    return ('<div class="dm-flow" aria-hidden="false">' + "".join(parts) + "</div>"
            f'<ol class="dm-flow-list">{items}</ol>')


# --- small builders -----------------------------------------------------------
def badge(kind: str, glyph: str, text: str) -> str:
    """A status chip. ``kind`` is one of ok/warn/err/neutral. The glyph plus the
    wording carry the meaning, so status never depends on colour alone."""
    return (f'<span class="dm-badge {esc(kind)}"><span class="g" aria-hidden="true">'
            f'{esc(glyph)}</span>{esc(text)}</span>')


def badge_row(badges: list) -> str:
    return '<div class="dm-badgerow">' + "".join(badges) + "</div>"


def doc_row(name: str, subtitle: str) -> str:
    ext = name.rsplit(".", 1)[-1].upper()[:4] if "." in name else "DOC"
    return ('<div class="dm-doc">'
            f'<span class="type">{esc(ext)}</span>'
            f'<span class="body"><span class="name" title="{esc(name)}">{esc(name)}</span>'
            f'<span class="sub">{esc(subtitle)}</span></span></div>')


def metric(value: str, label: str) -> str:
    return (f'<div class="dm-metric"><div class="v">{esc(value)}</div>'
            f'<div class="k">{esc(label)}</div></div>')


def kv_rows(rows: list) -> str:
    """Key/value list for the technical panels. Both sides are escaped."""
    return '<div class="dm-card">' + "".join(
        f'<div class="dm-kv"><span class="k">{esc(k)}</span>'
        f'<span class="v">{esc(v)}</span></div>' for k, v in rows) + "</div>"


def status_pill(label: str, online: bool) -> str:
    state = "on" if online else "off"
    word = "connected" if online else "not configured"
    return (f'<span class="dm-status"><span class="dot {state}" aria-hidden="true"></span>'
            f'{esc(label)}: {esc(word)}</span>')
