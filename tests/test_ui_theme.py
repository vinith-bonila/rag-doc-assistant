"""Design-system guarantees that are cheap to assert and easy to regress.

These cover the UI defects found during visual QA (docs/SECURITY_AUDIT.md
UI-01..UI-03) plus the escaping rule the whole UI depends on.
"""
from __future__ import annotations

import os
import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ui"))
import theme  # noqa: E402

CONFIG = pathlib.Path(".streamlit/config.toml")
APP = pathlib.Path("ui/streamlit_app.py")


def _srgb(hex_colour: str):
    h = hex_colour.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def _luminance(hex_colour: str) -> float:
    out = []
    for v in _srgb(hex_colour):
        v /= 255
        out.append(v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4)
    return 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2]


def contrast(a: str, b: str) -> float:
    la, lb = _luminance(a), _luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


# --- UI-03: escaping is the UI's only defence against hostile filenames -------
def test_esc_neutralizes_markup():
    out = theme.esc('<img src=x onerror="alert(1)">.pdf')
    # The point is that no live markup survives: the angle brackets and quotes
    # that would start a tag or an attribute are entity-encoded. The words
    # themselves remaining as inert text is fine.
    assert "<" not in out and ">" not in out and '"' not in out
    assert "&lt;img" in out and "&quot;" in out and "&gt;" in out


@pytest.mark.parametrize("builder,args", [
    (theme.doc_row, ('<b>x</b>.pdf', '<i>meta</i>')),
    (theme.badge, ('ok', '<svg>', '<script>alert(1)</script>')),
    (theme.metric, ('<b>3</b>', '<i>Docs</i>')),
    (theme.status_pill, ('<b>Model</b>', True)),
])
def test_builders_escape_every_interpolation(builder, args):
    html = builder(*args)
    assert "<b>" not in html and "<i>" not in html
    assert "<script>" not in html and "<svg>" not in html


def test_kv_rows_escapes_both_sides():
    html = theme.kv_rows([("<b>k</b>", "<i>v</i>")])
    assert "<b>" not in html and "<i>" not in html


# --- contrast: small copy must clear WCAG AA in both palettes -----------------
@pytest.mark.parametrize("mode", theme.MODES)
def test_body_and_small_text_meet_aa(mode):
    p = theme.PALETTES[mode]
    for surface in ("bg", "surface", "surface_alt"):
        assert contrast(p["text"], p[surface]) >= 4.5, (mode, "text", surface)
        # "muted" and "faint" are used at 12.5-13.5px, i.e. normal-size text.
        assert contrast(p["muted"], p[surface]) >= 4.5, (mode, "muted", surface)
        assert contrast(p["faint"], p[surface]) >= 4.5, (mode, "faint", surface)


@pytest.mark.parametrize("mode", theme.MODES)
def test_text_on_accent_fill_meets_aa(mode):
    on_accent = "#0B0F19" if mode == "dark" else "#FFFFFF"
    assert contrast(on_accent, theme.PALETTES[mode]["accent"]) >= 4.5, mode


@pytest.mark.parametrize("mode", theme.MODES)
def test_status_colours_are_legible_on_their_own_soft_backgrounds(mode):
    p = theme.PALETTES[mode]
    for tone in ("success", "warning", "error"):
        assert contrast(p[tone], p[f"{tone}_soft"]) >= 4.5, (mode, tone)


# --- UI-02: the app palette and Streamlit's own chrome must not disagree ------
def test_app_theme_matches_streamlit_config_base():
    """Streamlit paints its native widgets from config.toml and cannot be
    re-themed at runtime, so the two must be configured to the same palette."""
    from config import settings
    text = CONFIG.read_text(encoding="utf-8")
    base = re.search(r'^base\s*=\s*"(\w+)"', text, re.MULTILINE)
    assert base, "config.toml must declare a [theme] base"
    assert base.group(1) == settings.ui_theme, (
        "DOCMIND_THEME and .streamlit/config.toml [theme] base disagree")
    assert settings.ui_theme in theme.MODES


def test_config_defines_both_palettes():
    text = CONFIG.read_text(encoding="utf-8")
    assert "[theme.dark]" in text and "[theme.light]" in text


def test_no_runtime_theme_toggle_is_offered():
    """A runtime toggle cannot restyle Streamlit's own widgets, so the UI must
    not advertise one (every visible control has to do what it says)."""
    src = APP.read_text(encoding="utf-8")
    assert "Switch to light mode" not in src
    assert "Switch to dark mode" not in src


# --- UI-01: the document filter must not use the broken virtualised dropdown --
def test_document_filter_avoids_multiselect():
    """st.multiselect's virtualised dropdown renders its option list at zero
    width in the pinned Streamlit version, making options unreachable."""
    src = APP.read_text(encoding="utf-8")
    assert "st.multiselect(" not in src      # the call, not the comment
    assert "st.checkbox(" in src


# --- the flow chart renders as well-formed SVG in both palettes ---------------
@pytest.mark.parametrize("mode", theme.MODES)
def test_flow_diagram_is_wellformed_and_labelled(mode):
    import xml.dom.minidom as minidom
    html = theme.flow_diagram(mode)
    svg = html[html.index("<svg"):html.rindex("</svg>") + 6]
    minidom.parseString(svg)                      # raises if malformed
    assert 'role="img"' in svg and "aria-label=" in svg
    assert "six steps" in svg
    # every step is drawn
    for n in "123456":
        assert f">{n}</text>" in svg


@pytest.mark.parametrize("mode", theme.MODES)
def test_main_content_clears_the_fixed_app_header(mode):
    """UI-04: Streamlit's app header is opaque, absolutely positioned and sits
    at z-index 999990. Overriding the main block's padding-top with a flat value
    smaller than the header painted the page title underneath it, clipping it.

    The padding must therefore be derived from the header height, not guessed.
    """
    css = theme.css(mode)
    assert "--dm-header-h:" in css, "header height must be a token, not a magic number"
    header_px = int(re.search(r"--dm-header-h:\s*(\d+)px", css).group(1))
    assert header_px >= 56, "token must not understate Streamlit's header"

    pad = re.search(
        r"\.block-container[^{]*\{[^}]*padding-top:\s*([^;]+);", css, re.DOTALL)
    assert pad, "the main block must set padding-top"
    value = pad.group(1).strip()
    assert "var(--dm-header-h)" in value, (
        f"padding-top is {value!r}: it must be derived from --dm-header-h, "
        "otherwise the page title renders under the header")


def test_css_renders_for_every_mode_without_stray_placeholders():
    for mode in theme.MODES:
        css = theme.css(mode)
        assert "{bg}" not in css and "{accent}" not in css
        assert "--dm-on-accent:" in css
        assert "prefers-reduced-motion" in css
    # an unknown mode falls back rather than raising
    assert theme.css("chartreuse") == theme.css(theme.DEFAULT_MODE)


def test_env_override_selects_the_light_palette(monkeypatch):
    monkeypatch.setenv("DOCMIND_THEME", "light")
    assert os.getenv("DOCMIND_THEME") == "light"
    assert "#F8FAFC" in theme.css("light")
    assert "#0B0F19" in theme.css("dark")
