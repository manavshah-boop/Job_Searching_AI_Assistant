from __future__ import annotations

import html
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

PAGE_TITLE = "Beacon"

# Beacon design tokens — editorial monochrome palette.
# Default theme is "mono" (light editorial). theme-dark overrides live in the
# CSS below; the toggle UI ships in a later chunk.
COLOR_BG = "#f6f5f1"
COLOR_BG_2 = "#efeee9"
COLOR_BG_3 = "#e9e8e2"
COLOR_SURFACE = "#ffffff"
COLOR_INK = "#16170f"
COLOR_INK_2 = "#3b3c34"
COLOR_INK_3 = "#5a5b51"
COLOR_MUTED = "#807f73"

COLOR_LINE = "rgba(22,23,15,0.08)"
COLOR_LINE_2 = "rgba(22,23,15,0.14)"
COLOR_LINE_3 = "rgba(22,23,15,0.20)"

COLOR_ACCENT = "#16170f"
COLOR_ACCENT_INK = "#f6f5f1"

COLOR_SIGNAL = "#5b6c2e"          # success / good fit
COLOR_SIGNAL_2 = "#7a8c45"
COLOR_SIGNAL_SOFT = "#e7ecd7"
COLOR_WARN = "#a05a14"             # not relevant
COLOR_WARN_SOFT = "#f4e6cf"
COLOR_DANGER = "#8a2a1f"
COLOR_DANGER_SOFT = "#f0d8d2"
COLOR_POP = "#1f3d8a"               # info / agent activity
COLOR_POP_SOFT = "#dde4f4"

COLOR_FOCUS = "rgba(22,23,15,0.18)"

RADIUS_XS = "6px"
RADIUS_SM = "8px"
RADIUS_MD = "12px"
RADIUS_LG = "18px"
RADIUS_XL = "22px"

SHADOW_1 = "0 1px 0 rgba(22,23,15,0.04)"
SHADOW_2 = "0 1px 0 rgba(22,23,15,0.04), 0 8px 24px rgba(22,23,15,0.05)"
SHADOW_POP = "0 16px 44px rgba(22,23,15,0.14)"

SPACE_1 = "4px"
SPACE_2 = "8px"
SPACE_3 = "12px"
SPACE_4 = "16px"
SPACE_5 = "20px"
SPACE_6 = "24px"
SPACE_7 = "32px"
SPACE_8 = "48px"

FONT_DISPLAY = "'Inter Tight', ui-sans-serif, system-ui, sans-serif"
FONT_BODY = "'Inter', ui-sans-serif, system-ui, sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, monospace"

_PAGE_CONFIG_APPLIED = False


def _global_css() -> str:
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Beacon design tokens (mono = default, dark = override) ───────────── */
    :root {{
        /* Beacon raw tokens */
        --bg: {COLOR_BG};
        --bg-2: {COLOR_BG_2};
        --bg-3: {COLOR_BG_3};
        --surface: {COLOR_SURFACE};
        --ink: {COLOR_INK};
        --ink-2: {COLOR_INK_2};
        --ink-3: {COLOR_INK_3};
        --muted: {COLOR_MUTED};
        --line: {COLOR_LINE};
        --line-2: {COLOR_LINE_2};
        --line-3: {COLOR_LINE_3};
        --accent: {COLOR_ACCENT};
        --accent-ink: {COLOR_ACCENT_INK};
        --signal: {COLOR_SIGNAL};
        --signal-2: {COLOR_SIGNAL_2};
        --signal-soft: {COLOR_SIGNAL_SOFT};
        --warn: {COLOR_WARN};
        --warn-soft: {COLOR_WARN_SOFT};
        --danger: {COLOR_DANGER};
        --danger-soft: {COLOR_DANGER_SOFT};
        --pop: {COLOR_POP};
        --pop-soft: {COLOR_POP_SOFT};
        --r-xs: {RADIUS_XS};
        --r-sm: {RADIUS_SM};
        --r-md: {RADIUS_MD};
        --r-lg: {RADIUS_LG};
        --r-xl: {RADIUS_XL};
        --shadow-1: {SHADOW_1};
        --shadow-2: {SHADOW_2};
        --shadow-pop: {SHADOW_POP};
        --focus: 0 0 0 3px {COLOR_FOCUS};
        --font-display: {FONT_DISPLAY};
        --font-body: {FONT_BODY};
        --font-mono: {FONT_MONO};
        --sp-1: {SPACE_1};
        --sp-2: {SPACE_2};
        --sp-3: {SPACE_3};
        --sp-4: {SPACE_4};
        --sp-5: {SPACE_5};
        --sp-6: {SPACE_6};
        --sp-7: {SPACE_7};
        --sp-8: {SPACE_8};

        /* Legacy --shell-* aliases — keep existing shell-* classes working
           by pointing them at Beacon tokens. */
        --shell-bg: var(--bg);
        --shell-bg-accent: var(--bg-2);
        --shell-surface: var(--surface);
        --shell-surface-strong: var(--surface);
        --shell-surface-muted: var(--bg-2);
        --shell-surface-inverse: var(--ink);
        --shell-text: var(--ink);
        --shell-muted: var(--muted);
        --shell-border: var(--line);
        --shell-border-strong: var(--line-2);
        --shell-accent: var(--ink);
        --shell-accent-soft: var(--bg-2);
        --shell-success: var(--signal);
        --shell-success-soft: var(--signal-soft);
        --shell-warning: var(--warn);
        --shell-warning-soft: var(--warn-soft);
        --shell-danger: var(--danger);
        --shell-danger-soft: var(--danger-soft);
        --shell-info: var(--pop);
        --shell-info-soft: var(--pop-soft);
        --shell-focus: {COLOR_FOCUS};
        --shell-radius-sm: var(--r-sm);
        --shell-radius-md: var(--r-md);
        --shell-radius-lg: var(--r-lg);
        --shell-radius-xl: var(--r-xl);
        --shell-sidebar-top-offset: 0.9rem;
        --shell-shadow: var(--shadow-1);
        --shell-shadow-panel: var(--shadow-2);
        --shell-space-2xs: var(--sp-1);
        --shell-space-xs: var(--sp-2);
        --shell-space-sm: var(--sp-3);
        --shell-space-md: var(--sp-4);
        --shell-space-lg: var(--sp-5);
        --shell-space-xl: var(--sp-6);
        --shell-space-2xl: var(--sp-7);
        --shell-font-display: var(--font-display);
        --shell-font-body: var(--font-body);
    }}

    /* Theme: dark — overrides triggered by body.theme-dark. The mono
       defaults above ARE the editorial light theme; no body.theme-mono
       override needed unless future variants are introduced. */
    body.theme-dark {{
        --bg: #0f100c;
        --bg-2: #16170f;
        --bg-3: #1d1e15;
        --surface: #1b1c14;
        --ink: #f1efe6;
        --ink-2: #cfcdc1;
        --ink-3: #9d9c8e;
        --muted: #7d7c70;
        --line: rgba(241,239,230,0.08);
        --line-2: rgba(241,239,230,0.16);
        --line-3: rgba(241,239,230,0.24);
        --accent: #f1efe6;
        --accent-ink: #0f100c;
        --signal: #b9cf7c;
        --signal-2: #a4ba62;
        --signal-soft: rgba(185,207,124,0.13);
        --warn: #e6b070;
        --warn-soft: rgba(230,176,112,0.13);
        --danger: #e58072;
        --danger-soft: rgba(229,128,114,0.13);
        --pop: #9ab0e8;
        --pop-soft: rgba(154,176,232,0.13);
        --shadow-2: 0 1px 0 rgba(0,0,0,0.4), 0 8px 24px rgba(0,0,0,0.4);
        --focus: 0 0 0 3px rgba(241,239,230,0.18);
    }}

    html, body, [class*="css"] {{
        font-family: var(--font-body);
        color: var(--ink);
        font-size: 14px;
        line-height: 1.5;
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
    }}

    body {{
        background: var(--bg);
    }}

    .stApp {{
        background: var(--bg);
        color: var(--ink);
    }}

    header[data-testid="stHeader"] {{
        background: transparent;
        height: 0;
        border: 0;
    }}

    header[data-testid="stHeader"] > div {{
        height: 0;
    }}

    div[data-testid="stToolbar"] {{
        position: fixed;
        top: 0.9rem !important;
        left: 0.55rem !important;
        z-index: 1001;
        background: transparent;
    }}

    div[data-testid="collapsedControl"] {{
        position: fixed !important;
        top: 0.9rem !important;
        left: 0.55rem !important;
        z-index: 1001 !important;
        background: transparent !important;
    }}

    div[data-testid="stToolbar"] > div {{
        background: transparent;
        margin: 0 !important;
    }}

    div[data-testid="stDecoration"] {{
        display: none;
    }}

    header[data-testid="stHeader"] button[kind="header"] {{
        position: fixed;
        top: 0.9rem !important;
        left: 0.55rem !important;
        width: 2.5rem;
        height: 2.5rem;
        min-height: 2.5rem;
        padding: 0;
        border-radius: 14px;
        border: 1px solid var(--shell-border);
        background: rgba(255, 255, 255, 0.92);
        color: var(--shell-text);
        box-shadow: 0 10px 24px rgba(17, 34, 39, 0.08);
        z-index: 1001;
    }}

    div[data-testid="stToolbar"] button,
    div[data-testid="collapsedControl"] button {{
        width: 2.5rem;
        height: 2.5rem;
        min-height: 2.5rem;
        padding: 0;
        border-radius: 14px;
        border: 1px solid var(--shell-border);
        background: rgba(255, 255, 255, 0.92);
        color: var(--shell-text);
        box-shadow: 0 10px 24px rgba(17, 34, 39, 0.08);
    }}

    div[data-testid="collapsedControl"] svg,
    div[data-testid="stToolbar"] svg {{
        width: 1.05rem;
        height: 1.05rem;
    }}

    header[data-testid="stHeader"] button[kind="header"]:hover {{
        border-color: rgba(15, 118, 110, 0.28);
        background: rgba(255, 255, 255, 0.98);
    }}

    div[data-testid="stToolbar"] button:hover,
    div[data-testid="collapsedControl"] button:hover {{
        border-color: rgba(15, 118, 110, 0.28);
        background: rgba(255, 255, 255, 0.98);
    }}

    .block-container {{
        max-width: 1400px;
        padding-top: 0.32rem;
        padding-bottom: 2.25rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }}

    .main .block-container > div[data-testid="stVerticalBlock"] {{
        gap: 1.5rem;
    }}

    [data-testid="stSidebar"] {{
        min-width: 292px !important;
        max-width: 304px !important;
    }}

    [data-testid="stSidebar"] > div:first-child {{
        background: var(--bg-2);
        border-right: 1px solid var(--line);
        box-shadow: none;
        box-sizing: border-box;
        max-height: 100dvh;
        overflow-y: hidden;
        overflow-x: hidden;
        scrollbar-width: none;
        -ms-overflow-style: none;
    }}

    @media (max-height: 780px) {{
        [data-testid="stSidebar"] > div:first-child {{
            overflow-y: auto;
        }}
    }}

    [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar {{
        width: 0;
        height: 0;
        display: none;
    }}

    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] *::before,
    [data-testid="stSidebar"] *::after {{
        scrollbar-width: none;
        -ms-overflow-style: none;
    }}

    [data-testid="stSidebar"] *::-webkit-scrollbar {{
        width: 0 !important;
        height: 0 !important;
        display: none !important;
        background: transparent !important;
    }}

    [data-testid="stSidebar"] .block-container {{
        padding-top: var(--shell-sidebar-top-offset);
        padding-bottom: 0.55rem;
        padding-left: 1rem;
        padding-right: 1rem;
        min-height: auto;
        box-sizing: border-box;
    }}

    [data-testid="stSidebar"] .block-container > div[data-testid="stVerticalBlock"] {{
        display: flex;
        flex-direction: column;
        gap: 0.16rem;
        min-height: auto;
    }}

    [data-testid="stSidebar"] div[data-testid="stButton"],
    [data-testid="stSidebar"] div[data-testid="stLinkButton"] {{
        margin-top: 0;
        margin-bottom: 0;
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        margin: 0;
    }}

    [data-testid="stSidebarNav"] {{
        display: none;
    }}

    h1, h2, h3, h4 {{
        font-family: var(--shell-font-display);
        letter-spacing: -0.03em;
        color: var(--shell-text);
    }}

    p, li, label, span {{
        color: inherit;
    }}

    [data-testid="stMarkdownContainer"] p {{
        line-height: 1.55;
    }}

    .shell-app-frame {{
        position: relative;
    }}

    .shell-app-bar {{
        display: inline-flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        width: min(100%, 760px);
        padding: 0.5rem 0.85rem;
        border: 1px solid var(--shell-border-strong);
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(12px);
        box-shadow: var(--shell-shadow-panel);
        margin-bottom: 0.45rem;
    }}

    .shell-app-bar-title {{
        font-family: var(--shell-font-display);
        font-size: 0.92rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }}

    .shell-app-bar-copy {{
        color: var(--shell-muted);
        font-size: 0.88rem;
        line-height: 1.4;
        max-width: 40rem;
    }}

    .shell-page-header {{
        display: block;
        padding: 0.02rem 0 0.02rem 0;
        margin-bottom: 0;
    }}

    .shell-page-header-main {{
        display: grid;
        gap: 0.16rem;
        max-width: 42rem;
    }}

    .shell-page-header-identity {{
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }}

    .shell-page-header-avatar {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2.5rem;
        height: 2.5rem;
        flex: 0 0 2.5rem;
        border-radius: 999px;
        background: #e2e8f0;
        color: #475569;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        margin-top: 0.05rem;
    }}

    .shell-page-header-copy {{
        display: grid;
        gap: 0;
        min-width: 0;
    }}

    .shell-page-header-eyebrow,
    .shell-section-eyebrow,
    .shell-sidebar-kicker,
    .shell-app-header-kicker {{
        color: var(--shell-muted);
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }}

    .shell-page-header-title {{
        margin: 0;
        font-size: clamp(1.36rem, 1.82vw, 1.96rem);
        line-height: 1.01;
        overflow-wrap: anywhere;
        margin-bottom: 0.5rem;
    }}

    .shell-page-header-subtitle {{
        margin: 0;
        max-width: 56ch;
        color: #64748b;
        font-size: 0.9rem;
        line-height: 1.5;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    .shell-page-header-chips {{
        margin-top: 0.75rem;
    }}

    .shell-section {{
        display: block;
        margin-top: 0.15rem;
    }}

    .shell-section-header {{
        display: grid;
        gap: 0.14rem;
        margin-bottom: 0.72rem;
        max-width: none;
    }}

    .shell-section-title {{
        margin: 0;
        font-size: 1.32rem;
        line-height: 1.08;
        overflow-wrap: anywhere;
    }}

    .shell-panel-title {{
        margin: 0;
        font-size: 1.02rem;
        line-height: 1.08;
        overflow-wrap: anywhere;
    }}

    .shell-section-copy,
    .shell-panel-subtitle,
    .shell-section-subtitle,
    .shell-breadcrumb {{
        margin: 0;
        color: var(--shell-muted);
        font-size: 0.92rem;
        line-height: 1.44;
    }}

    .shell-section-copy {{
        max-width: 78ch;
    }}

    .shell-selection-header {{
        display: grid;
        gap: 0.375rem;
        margin-bottom: 1rem;
    }}

    .shell-selection-title {{
        margin: 0;
        font-family: var(--shell-font-display);
        font-size: clamp(1.5rem, 2.1vw, 2.05rem);
        font-weight: 700;
        line-height: 1.04;
        letter-spacing: -0.03em;
    }}

    .shell-selection-copy {{
        margin: 0;
        max-width: 56ch;
        color: #64748b;
        font-size: 0.94rem;
        line-height: 1.5;
    }}

    .shell-selection-label {{
        margin: 0 0 0.375rem 0;
        color: var(--shell-muted);
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }}

    .shell-workspace-card {{
        display: grid;
        gap: 0.375rem;
        margin-bottom: 0.625rem;
    }}

    .shell-workspace-card-title {{
        margin: 0;
        font-family: var(--shell-font-display);
        font-size: 1.18rem;
        font-weight: 600;
        line-height: 1.08;
        color: var(--shell-text);
        overflow-wrap: anywhere;
    }}

    .shell-workspace-card-copy {{
        margin: 0;
        color: #64748b;
        font-size: 0.9rem;
        line-height: 1.45;
        overflow-wrap: anywhere;
    }}

    .shell-workspace-card-action {{
        margin-top: 0.25rem;
    }}

    .shell-selection-footer {{
        height: 1.25rem;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.shell-workspace-card) {{
        padding: 1.25rem;
        margin-bottom: 1rem;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.shell-workspace-card) div[data-testid="stButton"] {{
        margin-top: 0;
        margin-bottom: 0;
    }}

    .shell-panel,
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        display: block;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--shell-border);
        background: var(--shell-surface);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 0.75rem;
    }}

    .shell-panel-marker {{
        display: none;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.shell-panel-marker--primary) {{
        border-color: rgba(15, 118, 110, 0.18);
        background: linear-gradient(180deg, rgba(250, 253, 252, 0.98) 0%, rgba(255, 255, 255, 0.98) 100%);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }}

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.shell-panel-marker--supporting) {{
        background: rgba(252, 253, 253, 0.94);
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] > div {{
        gap: 0.7rem;
    }}

    .shell-panel-head {{
        display: grid;
        gap: 0.28rem;
        margin-bottom: 0.72rem;
    }}

    .shell-sidebar-card {{
        display: grid;
        gap: 0.55rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: var(--shell-radius-md);
        border: 1px solid rgba(20, 35, 40, 0.1);
        background:
            linear-gradient(180deg, rgba(255,255,255,0.99) 0%, rgba(244,248,247,0.99) 100%);
        box-shadow: 0 10px 24px rgba(17, 34, 39, 0.06);
    }}

    .shell-sidebar-profile {{
        width: 100%;
        max-width: none;
        box-sizing: border-box;
        padding: 0.02rem 0 0 0;
        margin: 0;
    }}

    .shell-sidebar-identity {{
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        width: 100%;
        min-width: 0;
    }}

    .shell-sidebar-avatar {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2.5rem;
        height: 2.5rem;
        flex: 0 0 2.5rem;
        border-radius: 999px;
        background: #e2e8f0;
        color: #475569;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }}

    .shell-sidebar-body {{
        display: grid;
        gap: 0.12rem;
        width: 100%;
        min-width: 0;
    }}

    .shell-sidebar-title {{
        margin: 0 0 0.02rem 0;
        color: rgba(20, 35, 40, 0.94);
        font-family: var(--shell-font-display);
        font-size: 1.08rem;
        font-weight: 600;
        line-height: 1.15;
        overflow-wrap: anywhere;
        letter-spacing: -0.015em;
    }}

    .shell-sidebar-divider {{
        width: 100%;
        height: 1px;
        margin: 0.18rem 0 0.2rem 0;
        background: rgba(20, 35, 40, 0.11);
    }}

    .shell-sidebar-divider--section {{
        margin-top: 0.24rem;
        margin-bottom: 0.22rem;
    }}

    .shell-sidebar-section {{
        display: grid;
        gap: 0.08rem;
        margin: 0;
    }}

    .shell-sidebar-copy {{
        margin: 0;
        color: #64748b;
        font-size: 0.9rem;
        line-height: 1.5;
        overflow-wrap: anywhere;
        word-break: break-word;
        max-width: none;
        white-space: normal;
    }}

    .shell-sidebar-meta {{
        margin-top: 0.35rem;
        color: var(--shell-muted);
        font-size: 0.82rem;
        line-height: 1.4;
    }}

    .shell-chip-row--sidebar {{
        margin-top: 0.75rem;
        justify-content: flex-start;
    }}

    .shell-chip-row--sidebar .shell-chip {{
        min-height: 1.4rem;
        padding: 0.1rem 0.52rem;
        border-color: transparent;
        background: #f1f5f9;
        color: #334155;
        font-size: 0.72rem;
        font-weight: 600;
    }}

    .shell-chip-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.3rem;
        align-items: center;
    }}

    .shell-chip {{
        display: inline-flex;
        align-items: center;
        min-height: 1.48rem;
        padding: 0.11rem 0.44rem;
        border-radius: 999px;
        border: 1px solid rgba(20, 35, 40, 0.045);
        background: rgba(255, 255, 255, 0.52);
        color: var(--shell-muted);
        font-size: 0.7rem;
        font-weight: 500;
        line-height: 1.2;
        max-width: 100%;
        overflow-wrap: anywhere;
    }}

    .shell-page-header-chips .shell-chip {{
        min-height: 1.4rem;
        padding: 0.1rem 0.52rem;
        border-color: transparent;
        background: #f1f5f9;
        color: #334155;
        font-size: 0.72rem;
        font-weight: 600;
    }}

    .shell-toolbar {{
        margin-top: 0.45rem;
        margin-bottom: 0.15rem;
    }}

    .shell-toolbar--header {{
        margin-top: 0;
        margin-bottom: 0.25rem;
    }}

    .shell-page-header-actions {{
        display: flex;
        justify-content: flex-end;
        align-items: flex-start;
        padding-top: 0.42rem;
    }}

    .shell-page-header-actions .shell-toolbar {{
        width: 100%;
        margin-top: 0;
        margin-bottom: 0;
    }}

    .shell-page-header-actions .shell-toolbar div[data-testid="stButton"] > button,
    .shell-page-header-actions .shell-toolbar div[data-testid="stLinkButton"] > a {{
        min-height: 2.2rem;
        width: auto;
        margin-left: auto;
    }}

    .shell-toolbar--compact div[data-testid="stButton"] > button,
    .shell-toolbar--compact div[data-testid="stLinkButton"] > a {{
        min-height: 2.45rem;
    }}

    .shell-toolbar--sidebar-primary {{
        margin-top: 0.08rem;
        margin-bottom: 0;
    }}

    .shell-toolbar--sidebar-primary div[data-testid="stButton"] > button[kind="primary"] {{
        min-height: 2.45rem;
        border-radius: 10px;
        box-shadow: 0 8px 18px rgba(17, 34, 39, 0.06);
    }}

    .shell-toolbar--sidebar-subtle {{
        margin-top: 0;
        margin-bottom: 0;
    }}

    .shell-toolbar--sidebar-subtle div[data-testid="stButton"] > button[kind="secondary"] {{
        min-height: 1.92rem;
        justify-content: flex-start;
        border-radius: 8px;
        border: 0;
        background: transparent;
        box-shadow: none;
        padding-left: 0.12rem;
        padding-right: 0.12rem;
        color: var(--shell-muted);
        font-weight: 500;
    }}

    .shell-toolbar--sidebar-subtle div[data-testid="stButton"] > button[kind="secondary"]:hover {{
        background: rgba(20, 35, 40, 0.045);
        border-color: transparent;
        color: var(--shell-text);
        transform: none;
    }}

    .shell-toolbar-meta {{
        color: var(--shell-muted);
        font-size: 0.86rem;
        margin-bottom: 0.45rem;
    }}

    .shell-stat-card {{
        display: grid;
        gap: 0.38rem;
        min-height: 98px;
        padding: 1rem 1.05rem;
        border-radius: 12px;
        border: 1px solid rgba(20, 35, 40, 0.1);
        background:
            linear-gradient(180deg, rgba(244,248,247,0.98) 0%, rgba(251,253,253,0.98) 100%);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }}

    .shell-stat-row-gap {{
        height: 0.85rem;
    }}

    .shell-stat-label {{
        color: var(--shell-muted);
        font-size: 0.74rem;
        font-weight: 500;
        line-height: 1.35;
        overflow-wrap: anywhere;
    }}

    .shell-stat-value {{
        font-family: var(--shell-font-display);
        font-size: clamp(1.55rem, 1.8vw, 2.15rem);
        line-height: 1;
        overflow-wrap: anywhere;
    }}

    .shell-stat-delta {{
        color: var(--shell-muted);
        font-size: 0.83rem;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }}

    .shell-callout {{
        border-radius: calc(var(--shell-radius-md) - 6px);
        border: 1px solid var(--shell-border);
        padding: 0.85rem 0.95rem;
        margin: 0.4rem 0;
    }}

    .shell-callout-title {{
        font-weight: 700;
        margin-bottom: 0.25rem;
    }}

    .shell-callout--info {{
        background: var(--shell-info-soft);
        border-color: rgba(36, 87, 166, 0.16);
    }}

    .shell-callout--success {{
        background: var(--shell-success-soft);
        border-color: rgba(19, 121, 91, 0.14);
    }}

    .shell-callout--warning {{
        background: var(--shell-warning-soft);
        border-color: rgba(154, 91, 0, 0.16);
    }}

    .shell-callout--error,
    .shell-callout--danger {{
        background: var(--shell-danger-soft);
        border-color: rgba(180, 35, 24, 0.16);
    }}

    .shell-empty-state {{
        display: grid;
        justify-items: center;
        gap: 0.38rem;
        text-align: center;
        padding: 1rem 0.9rem 0.95rem;
        border: 1px dashed rgba(20, 35, 40, 0.12);
        border-radius: calc(var(--shell-radius-md) - 4px);
        background: rgba(247, 250, 249, 0.7);
    }}

    .shell-empty-state-mark {{
        display: inline-flex;
        align-items: center;
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        background: rgba(20, 35, 40, 0.04);
        color: var(--shell-muted);
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }}

    .shell-empty-state h3 {{
        margin: 0;
        font-size: 1.05rem;
    }}

    .shell-empty-state-icon {{
        position: relative;
        width: 2.45rem;
        height: 2.95rem;
        border-radius: 8px;
        border: 1px solid rgba(20, 35, 40, 0.12);
        background: rgba(255, 255, 255, 0.92);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }}

    .shell-empty-state-icon--document::before {{
        content: "";
        position: absolute;
        left: 0.52rem;
        right: 0.52rem;
        top: 0.72rem;
        height: 1px;
        background: rgba(20, 35, 40, 0.22);
        box-shadow:
            0 0.48rem 0 rgba(20, 35, 40, 0.18),
            0 0.96rem 0 rgba(20, 35, 40, 0.14);
    }}

    .shell-empty-state-icon--document::after {{
        content: "";
        position: absolute;
        top: 0.28rem;
        right: 0.3rem;
        width: 0.5rem;
        height: 0.5rem;
        border-top: 1px solid rgba(20, 35, 40, 0.14);
        border-right: 1px solid rgba(20, 35, 40, 0.14);
        background: rgba(247, 250, 249, 0.96);
        transform: rotate(45deg);
    }}

    .shell-empty-state p {{
        margin: 0;
        max-width: 32rem;
        color: var(--shell-muted);
        font-size: 0.92rem;
        line-height: 1.45;
    }}

    .shell-badge {{
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.25rem 0.55rem;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid transparent;
    }}

    .shell-badge--neutral {{
        background: rgba(20, 35, 40, 0.06);
        border-color: rgba(20, 35, 40, 0.08);
        color: var(--shell-text);
    }}

    .shell-badge--info {{
        background: var(--shell-info-soft);
        border-color: rgba(36, 87, 166, 0.14);
        color: var(--shell-info);
    }}

    .shell-badge--success {{
        background: var(--shell-success-soft);
        border-color: rgba(19, 121, 91, 0.14);
        color: var(--shell-success);
    }}

    .shell-badge--warning {{
        background: var(--shell-warning-soft);
        border-color: rgba(154, 91, 0, 0.16);
        color: var(--shell-warning);
    }}

    .shell-badge--danger {{
        background: var(--shell-danger-soft);
        border-color: rgba(180, 35, 24, 0.16);
        color: var(--shell-danger);
    }}

    .match-card {{
        background: var(--shell-surface);
        border: 1px solid var(--shell-border);
        border-radius: calc(var(--shell-radius-md) - 4px);
        padding: 1rem;
        margin-bottom: 0.75rem;
    }}

    .match-title {{
        font-weight: 700;
        font-size: 1.03rem;
        margin-bottom: 0.2rem;
        overflow-wrap: anywhere;
    }}

    .match-meta {{
        color: var(--shell-muted);
        font-size: 0.92rem;
        margin-bottom: 0.45rem;
        overflow-wrap: anywhere;
    }}

    .match-summary {{
        color: var(--shell-text);
        line-height: 1.55;
        overflow-wrap: anywhere;
    }}

    .badge-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.55rem;
    }}

    .badge {{
        display: inline-block;
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        background: var(--shell-accent-soft);
        color: var(--shell-accent);
        font-size: 0.84rem;
        border: 1px solid rgba(15, 118, 110, 0.18);
    }}

    .badge.warn {{
        color: var(--shell-warning);
        background: var(--shell-warning-soft);
        border-color: rgba(154, 91, 0, 0.16);
    }}

    .badge.fail {{
        color: var(--shell-danger);
        background: var(--shell-danger-soft);
        border-color: rgba(180, 35, 24, 0.16);
    }}

    .ops-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.75rem;
    }}

    .ops-grid--compact {{
        grid-template-columns: 1fr;
        margin-top: 0.75rem;
    }}

    .ops-item {{
        padding: 0.85rem 0.9rem;
        border-radius: 16px;
        border: 1px solid rgba(20, 35, 40, 0.08);
        background: rgba(245, 248, 247, 0.88);
    }}

    .ops-label,
    .filter-chip-title,
    .job-detail-kicker {{
        color: var(--shell-muted);
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }}

    .ops-value {{
        margin-top: 0.32rem;
        font-weight: 600;
        line-height: 1.45;
        overflow-wrap: anywhere;
    }}

    .ops-footnote {{
        margin-top: 0.75rem;
        color: var(--shell-muted);
        font-size: 0.9rem;
    }}

    .ops-hero {{
        display: grid;
        gap: 0.22rem;
    }}

    .ops-hero-label,
    .overview-guidance-kicker,
    .last-run-title {{
        color: var(--shell-muted);
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }}

    .ops-hero-title {{
        font-family: var(--shell-font-display);
        font-size: 1.18rem;
        line-height: 1.08;
        overflow-wrap: anywhere;
    }}

    .overview-guidance-title {{
        font-family: var(--shell-font-display);
        font-size: 1.35rem;
        line-height: 1.06;
        overflow-wrap: anywhere;
    }}

    .ops-hero-copy {{
        color: var(--shell-muted);
        font-size: 0.92rem;
        line-height: 1.45;
    }}

    .review-status-card {{
        display: grid;
        gap: 0.22rem;
    }}

    .review-status-title {{
        font-family: var(--shell-font-display);
        font-size: 1.08rem;
        line-height: 1.08;
        overflow-wrap: anywhere;
    }}

    .review-status-count {{
        font-family: var(--shell-font-display);
        font-size: clamp(1.95rem, 2.2vw, 2.8rem);
        line-height: 0.95;
        color: var(--shell-text);
    }}

    .review-status-copy {{
        color: var(--shell-text);
        font-size: 0.95rem;
        line-height: 1.42;
    }}

    .review-status-meta {{
        color: var(--shell-muted);
        font-size: 0.84rem;
        line-height: 1.38;
        margin-top: 0.1rem;
    }}

    .overview-guidance-copy,
    .last-run-copy {{
        color: var(--shell-muted);
        font-size: 0.95rem;
        line-height: 1.5;
    }}

    .ops-inline-note {{
        color: var(--shell-muted);
        font-size: 0.84rem;
        line-height: 1.4;
    }}

    .overview-guidance {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        padding: 1.05rem 1.12rem;
        border-radius: 12px;
        border: 1px solid rgba(15, 118, 110, 0.2);
        background: linear-gradient(180deg, rgba(247, 251, 250, 0.98) 0%, rgba(255, 255, 255, 0.98) 100%);
        box-shadow: 0 10px 22px rgba(17, 34, 39, 0.06);
        margin-bottom: 1rem;
    }}

    .overview-guidance-status {{
        display: flex;
        justify-content: flex-end;
        min-width: 120px;
    }}

    .shell-panel-gap {{
        height: 0.55rem;
    }}

    .overview-filter-label {{
        color: var(--shell-muted);
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin: 0.05rem 0 0.35rem;
    }}

    .overview-action-toolbar {{
        margin-top: 0.75rem;
        margin-bottom: 0;
    }}

    .overview-run-bar {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.85rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(20, 35, 40, 0.08);
        background: rgba(248, 250, 249, 0.92);
        box-shadow: 0 4px 12px rgba(17, 34, 39, 0.03);
    }}

    .overview-run-bar-label {{
        color: var(--shell-muted);
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        flex: 0 0 auto;
    }}

    .overview-run-bar-copy {{
        color: var(--shell-text);
        font-size: 0.92rem;
        line-height: 1.42;
        overflow-wrap: anywhere;
    }}

    .summary-list {{
        display: grid;
        gap: 0.52rem;
    }}

    .summary-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.68rem 0.82rem;
        border-radius: 16px;
        background: rgba(245, 248, 247, 0.8);
        border: 1px solid rgba(20, 35, 40, 0.08);
    }}

    .summary-row span {{
        color: var(--shell-muted);
        font-size: 0.86rem;
    }}

    .summary-row strong {{
        text-align: right;
        max-width: 62%;
        overflow-wrap: anywhere;
    }}

    .resume-link-row {{
        margin-top: 0.55rem;
    }}

    .resume-link {{
        color: var(--shell-accent);
        font-size: 0.9rem;
        font-weight: 600;
        text-decoration: none;
    }}

    .resume-link:hover {{
        text-decoration: underline;
    }}

    .run-card {{
        padding: 0.95rem 1rem;
        border-radius: 18px;
        border: 1px solid rgba(20, 35, 40, 0.1);
        background: rgba(244, 248, 247, 0.84);
        margin-bottom: 0.75rem;
    }}

    .run-card-head,
    .jobs-workspace-summary,
    .selection-banner {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 0.9rem;
    }}

    .run-card-title,
    .job-detail-title {{
        font-family: var(--shell-font-display);
        font-size: 1.05rem;
        font-weight: 700;
        line-height: 1.08;
        overflow-wrap: anywhere;
    }}

    .run-card-meta,
    .job-detail-meta {{
        margin-top: 0.18rem;
        color: var(--shell-muted);
        font-size: 0.9rem;
        overflow-wrap: anywhere;
    }}

    .run-card-badges,
    .job-detail-badges {{
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 0.35rem;
    }}

    .run-card-stats {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin-top: 0.8rem;
    }}

    .run-card-stats div,
    .jobs-workspace-summary div {{
        display: grid;
        gap: 0.18rem;
    }}

    .run-card-stats span,
    .jobs-workspace-summary span,
    .selection-banner div:last-child {{
        color: var(--shell-muted);
        font-size: 0.84rem;
        overflow-wrap: anywhere;
    }}

    .run-card-issue {{
        margin-top: 0.8rem;
        color: var(--shell-muted);
        font-size: 0.92rem;
    }}

    .jobs-workspace-summary {{
        padding: 0.78rem 0.9rem;
        margin-bottom: 0.8rem;
        border-radius: 18px;
        border: 1px solid rgba(20, 35, 40, 0.1);
        background: rgba(252, 253, 253, 0.96);
    }}

    .selection-banner {{
        padding: 0.72rem 0.82rem;
        margin-top: 0.75rem;
        margin-bottom: 0.45rem;
        border-radius: 16px;
        border: 1px solid rgba(20, 35, 40, 0.1);
        background: rgba(244, 248, 247, 0.9);
    }}

    .job-detail-header {{
        display: grid;
        gap: 0.38rem;
        padding: 0.2rem 0 0.7rem 0;
    }}

    .run-summary-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.8rem;
        padding: 0.05rem 0;
    }}

    .run-summary-main,
    .run-summary-empty {{
        display: grid;
        gap: 0.18rem;
        min-width: 0;
    }}

    .run-summary-status {{
        font-weight: 700;
        font-size: 1rem;
        line-height: 1.24;
        overflow-wrap: anywhere;
    }}

    .run-summary-copy {{
        color: var(--shell-muted);
        font-size: 0.9rem;
        line-height: 1.42;
        overflow-wrap: anywhere;
    }}

    .run-summary-side {{
        display: flex;
        justify-content: flex-end;
        flex: 0 0 auto;
    }}

    .shell-muted-note {{
        color: var(--shell-muted);
        font-size: 0.88rem;
        line-height: 1.45;
        margin: 0.1rem 0 0 0;
    }}

    .shell-inline-section-label {{
        color: var(--shell-muted);
        font-size: 0.77rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin: 0;
    }}

    .shell-inline-section-label--sidebar {{
        margin: 0 0 0.04rem 0;
    }}

    .shell-sidebar-spacer {{
        flex: 0 0 auto;
        min-height: 0.35rem;
    }}

    .shell-sidebar-actions {{
        display: grid;
        gap: 0;
        padding-top: 0;
        border-top: 0;
    }}

    .shell-sidebar-actions .shell-toolbar--sidebar-primary {{
        margin-top: 0;
        margin-bottom: 0.1rem;
    }}

    .shell-sidebar-actions .shell-toolbar--sidebar-subtle {{
        margin-top: 0;
        margin-bottom: 0;
    }}

    .shell-sidebar-actions div[data-testid="stVerticalBlock"] {{
        gap: 0.08rem;
    }}

    .shell-sidebar-actions div[data-testid="stButton"] {{
        margin-bottom: 0;
    }}

    .workspace-grid {{
        display: grid;
        gap: 0.85rem;
    }}

    .job-detail-title {{
        margin: 0;
        font-size: 1.35rem;
    }}

    .score-card {{
        display: grid;
        gap: 0.4rem;
        padding: 0.95rem 1rem;
        border-radius: 18px;
        border: 1px solid rgba(20, 35, 40, 0.08);
        background: rgba(255, 255, 255, 0.8);
        margin-bottom: 0.7rem;
    }}

    .score-card--primary {{
        background: linear-gradient(180deg, rgba(225, 243, 240, 0.95) 0%, rgba(255, 255, 255, 0.92) 100%);
        border-color: rgba(15, 118, 110, 0.16);
    }}

    .score-card-label {{
        color: var(--shell-muted);
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    .score-card-value {{
        font-family: var(--shell-font-display);
        font-size: 2rem;
        line-height: 1;
    }}

    .score-card-copy {{
        color: var(--shell-muted);
        font-size: 0.9rem;
        line-height: 1.45;
    }}

    .setup-step-grid {{
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.7rem;
        margin-bottom: 0.6rem;
    }}

    .setup-progress-rail {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.36rem;
        margin: 0.1rem 0 0.35rem;
    }}

    .setup-step-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        min-height: 1.85rem;
        padding: 0.22rem 0.62rem;
        border-radius: 999px;
        border: 1px solid rgba(20, 35, 40, 0.1);
        background: rgba(255, 255, 255, 0.56);
        color: var(--shell-muted);
        font-size: 0.79rem;
        font-weight: 500;
    }}

    .setup-step-pill--active {{
        border-color: rgba(15, 118, 110, 0.24);
        background: linear-gradient(180deg, rgba(225, 243, 240, 0.92) 0%, rgba(255,255,255,0.82) 100%);
        color: var(--shell-text);
    }}

    .setup-step-pill--done {{
        color: var(--shell-muted);
        background: rgba(245, 248, 247, 0.72);
    }}

    .setup-step-pill-index {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.3rem;
        height: 1.3rem;
        border-radius: 999px;
        background: var(--shell-surface-inverse);
        color: #f8fcfb;
        font-size: 0.72rem;
        font-weight: 700;
        flex: 0 0 auto;
    }}

    .setup-step {{
        display: flex;
        gap: 0.75rem;
        align-items: flex-start;
        padding: 0.65rem 0.8rem;
        border-radius: 18px;
        border: 1px solid rgba(20, 35, 40, 0.08);
        background: rgba(255, 255, 255, 0.72);
    }}

    .setup-step--active {{
        border-color: rgba(15, 118, 110, 0.24);
        background: linear-gradient(180deg, rgba(225, 243, 240, 0.92) 0%, rgba(255,255,255,0.82) 100%);
    }}

    .setup-step--done {{
        background: rgba(245, 248, 247, 0.92);
    }}

    .setup-step-index {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.8rem;
        height: 1.8rem;
        border-radius: 999px;
        background: var(--shell-surface-inverse);
        color: #f8fcfb;
        font-size: 0.85rem;
        font-weight: 700;
        flex: 0 0 auto;
    }}

    .setup-step-title {{
        font-weight: 700;
        font-size: 0.92rem;
    }}

    .setup-step-copy {{
        color: var(--shell-muted);
        font-size: 0.84rem;
        line-height: 1.4;
        margin-top: 0.15rem;
    }}

    div[data-testid="stButton"],
    div[data-testid="stLinkButton"] {{
        margin-top: 0.1rem;
        margin-bottom: 0.2rem;
    }}

    div[data-testid="stButton"] > button,
    div[data-testid="stLinkButton"] > a {{
        min-height: 2.75rem;
        border-radius: 12px;
        font-weight: 600;
    }}

    div[data-testid="stButton"] > button[kind="primary"] {{
        background: var(--shell-surface-inverse);
        border-color: var(--shell-surface-inverse);
        color: #f8fcfb;
        box-shadow: 0 8px 18px rgba(17, 34, 39, 0.08);
    }}

    div[data-testid="stButton"] > button[kind="secondary"],
    div[data-testid="stLinkButton"] > a {{
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(20, 35, 40, 0.12);
        color: var(--shell-text);
        box-shadow: none;
    }}

    div[data-testid="stButton"] > button:hover,
    div[data-testid="stLinkButton"] > a:hover {{
        border-color: rgba(15, 118, 110, 0.28);
        transform: translateY(-1px);
    }}

    div[data-baseweb="tab-list"] {{
        gap: 0.45rem;
        margin-bottom: 0.5rem;
    }}

    button[data-baseweb="tab"] {{
        border-radius: 999px;
        border: 1px solid rgba(20, 35, 40, 0.10);
        background: rgba(255, 255, 255, 0.8);
        padding: 0.42rem 0.92rem;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        background: var(--shell-accent-soft);
        border-color: rgba(15, 118, 110, 0.22);
        color: var(--shell-accent);
    }}

    div[data-testid="stRadio"] > label {{
        color: var(--shell-muted);
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 0.45rem;
    }}

    div[data-testid="stRadio"] [role="radiogroup"] {{
        gap: 0.42rem;
    }}

    div[data-testid="stRadio"] [role="radiogroup"] label {{
        border: 1px solid rgba(20, 35, 40, 0.10);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.70);
        padding: 0.7rem 0.85rem;
        min-height: 46px;
    }}

    div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {{
        background: var(--shell-surface-inverse);
        border-color: var(--shell-surface-inverse);
        color: #f8fcfb;
        box-shadow: 0 10px 20px rgba(17, 34, 39, 0.10);
    }}

    div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p,
    div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) span,
    div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) div,
    div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) * {{
        color: #f8fcfb !important;
        fill: #f8fcfb !important;
    }}

    [data-testid="stSidebar"] div[data-testid="stRadio"] {{
        margin-top: 0;
        margin-bottom: 0;
    }}

    [data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] {{
        display: grid;
        gap: 0;
    }}

    [data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label {{
        position: relative;
        width: 100%;
        border-radius: 8px;
        border: 1px solid transparent;
        background: transparent;
        padding: 0.48rem 0.72rem 0.48rem 0.96rem;
        min-height: 36px;
        box-shadow: none;
        margin: 0;
    }}

    [data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label:hover {{
        background: rgba(20, 35, 40, 0.04);
        border-color: rgba(20, 35, 40, 0.04);
    }}

    [data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label > div:first-of-type {{
        display: none !important;
    }}

    [data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label input {{
        position: absolute;
        opacity: 0;
        pointer-events: none;
    }}

    [data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label p {{
        margin: 0;
        font-size: 0.94rem;
        font-weight: 500;
        line-height: 1.35;
        color: rgba(20, 35, 40, 0.84);
    }}

    [data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {{
        background: rgba(225, 243, 240, 0.9) !important;
        border-color: rgba(15, 118, 110, 0.18) !important;
        box-shadow: none !important;
    }}

    [data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked)::before {{
        content: "";
        position: absolute;
        left: 0.38rem;
        top: 50%;
        transform: translateY(-50%);
        width: 0.18rem;
        height: 1.05rem;
        border-radius: 999px;
        background: var(--shell-accent);
    }}

    [data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p {{
        color: rgba(20, 35, 40, 0.98) !important;
        font-weight: 600;
    }}

    [data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) span,
    [data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) div,
    [data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) * {{
        color: rgba(20, 35, 40, 0.98) !important;
    }}

    [data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) svg,
    [data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) [data-testid="stMarkdownContainer"] * {{
        color: rgba(20, 35, 40, 0.98) !important;
        fill: rgba(20, 35, 40, 0.98) !important;
    }}

    div[data-testid="stDataFrame"] {{
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid var(--shell-border);
        border-radius: var(--shell-radius-md);
        box-shadow: var(--shell-shadow-panel);
        overflow: hidden;
    }}

    div[data-testid="stDataFrame"] [aria-selected="true"] {{
        background: rgba(15, 118, 110, 0.10) !important;
        box-shadow: inset 3px 0 0 var(--shell-accent);
    }}

    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stSelectbox"] [data-baseweb="select"],
    div[data-testid="stMultiSelect"] [data-baseweb="select"],
    div[data-testid="stNumberInput"] input {{
        border-radius: 14px;
        border-width: 1.5px;
    }}

    div[data-testid="stSlider"] {{
        padding-top: 0.25rem;
    }}

    hr {{
        border-color: rgba(20, 35, 40, 0.08);
        margin: 0.7rem 0 0.9rem 0;
    }}

    .stApp a:focus-visible,
    .stApp button:focus-visible,
    .stApp [role="button"]:focus-visible,
    .stApp input:focus-visible,
    .stApp textarea:focus-visible,
    .stApp [tabindex]:focus-visible,
    .stApp select:focus-visible {{
        outline: 3px solid var(--shell-focus);
        outline-offset: 3px;
    }}

    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stTextArea"] textarea:focus,
    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within,
    div[data-testid="stMultiSelect"] [data-baseweb="select"]:focus-within {{
        border-color: rgba(15, 118, 110, 0.45) !important;
        box-shadow: 0 0 0 1px rgba(15, 118, 110, 0.28);
    }}

    @media (max-width: 960px) {{
        :root {{
            --shell-sidebar-top-offset: 1rem;
        }}

        .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}

        .shell-app-bar {{
            width: 100%;
        }}

        .shell-page-header-title {{
            font-size: 1.45rem;
        }}

        .ops-grid,
        .run-card-stats,
        .setup-step-grid {{
            grid-template-columns: 1fr;
        }}

        .run-card-head,
        .jobs-workspace-summary,
        .selection-banner,
        .overview-guidance,
        .run-summary-row {{
            flex-direction: column;
        }}

        .run-card-badges,
        .job-detail-badges,
        .run-summary-side {{
            justify-content: flex-start;
        }}
    }}

    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
            scroll-behavior: auto !important;
        }}
    }}

    /* ─── Beacon raw classes ────────────────────────────────────────────────
       These mirror the reference Beacon mock: Brand mark, ProfileMenu, NavItem,
       RunCard pulse, Rating control + compact variant, status pills, train
       card. Used by ui_shell sidebar primitives and dashboard tabs. */

    h1, h2, h3, h4, h5 {{
        font-family: var(--font-display);
        letter-spacing: -0.022em;
        color: var(--ink);
        font-weight: 600;
        line-height: 1.1;
    }}

    /* ── Brand mark ── */
    .beacon-brand {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 6px 8px;
    }}
    .beacon-brand-mark {{
        width: 28px;
        height: 28px;
        border-radius: 8px;
        background: var(--ink);
        color: var(--bg);
        display: grid;
        place-items: center;
        font-family: var(--font-display);
        font-weight: 700;
        font-size: 14px;
        letter-spacing: -0.04em;
    }}
    .beacon-brand-text {{
        display: flex;
        flex-direction: column;
        gap: 1px;
    }}
    .beacon-brand-name {{
        font-family: var(--font-display);
        font-weight: 600;
        letter-spacing: -0.02em;
        font-size: 15.5px;
        color: var(--ink);
    }}
    .beacon-brand-sub {{
        font-family: var(--font-mono);
        font-size: 10px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    /* ── Profile pill (sidebar identity card, opens popover) ── */
    .beacon-who {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 10px;
        border-radius: var(--r-sm);
        border: 1px solid var(--line);
        background: transparent;
        width: 100%;
        text-align: left;
    }}
    .beacon-who-av {{
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: var(--ink);
        color: var(--bg);
        display: grid;
        place-items: center;
        font-weight: 600;
        font-size: 12px;
        font-family: var(--font-display);
        flex: 0 0 auto;
    }}
    .beacon-who-meta {{
        display: flex;
        flex-direction: column;
        gap: 2px;
        flex: 1;
        min-width: 0;
    }}
    .beacon-who-name {{
        font-size: 13px;
        font-weight: 600;
        line-height: 1.1;
        color: var(--ink);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .beacon-who-sub {{
        font-size: 11px;
        color: var(--muted);
        font-family: var(--font-mono);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}

    /* ── NavGroup (sidebar grouped sections) ── */
    .beacon-nav-kicker {{
        font-family: var(--font-mono);
        font-size: 10px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 8px 8px 4px;
        font-weight: 500;
    }}

    /* Style sidebar buttons that sit inside a `.beacon-nav-group-marker`
       container as Beacon nav-items. The marker is a sibling rendered
       inside the same Streamlit container as the nav buttons. */
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.beacon-nav-group-marker) [data-testid="stButton"] > button,
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> [data-testid="stMarkdownContainer"] .beacon-nav-group-marker) [data-testid="stButton"] > button {{
        display: flex;
        align-items: center;
        justify-content: flex-start;
        gap: 10px;
        height: auto;
        min-height: 34px;
        padding: 7px 10px;
        border-radius: var(--r-xs);
        font-size: 13px;
        font-weight: 500;
        font-family: var(--font-body);
        color: var(--ink-2);
        text-align: left;
        background: transparent;
        border: 1px solid transparent;
        box-shadow: none;
    }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.beacon-nav-group-marker) [data-testid="stButton"] > button:hover,
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> [data-testid="stMarkdownContainer"] .beacon-nav-group-marker) [data-testid="stButton"] > button:hover {{
        background: rgba(22,23,15,0.04);
        color: var(--ink);
        transform: none;
    }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.beacon-nav-group-marker) [data-testid="stButton"] > button[kind="primary"],
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> [data-testid="stMarkdownContainer"] .beacon-nav-group-marker) [data-testid="stButton"] > button[kind="primary"] {{
        background: var(--surface);
        color: var(--ink);
        border-color: var(--line);
        font-weight: 600;
        box-shadow: var(--shadow-1);
    }}
    body.theme-dark [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.beacon-nav-group-marker) [data-testid="stButton"] > button[kind="primary"] {{
        background: var(--bg-3);
    }}

    /* ── Run-card (live pipeline preview pinned to sidebar bottom) ── */
    .beacon-run-card {{
        padding: 11px 12px;
        border-radius: var(--r-sm);
        background: var(--surface);
        border: 1px solid var(--line);
        display: flex;
        flex-direction: column;
        gap: 9px;
        margin-top: 6px;
    }}
    .beacon-run-status {{
        display: flex;
        align-items: center;
        gap: 6px;
        font-family: var(--font-mono);
        font-size: 10px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    .beacon-pulse {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--signal);
        box-shadow: 0 0 0 0 rgba(91,108,46,0.55);
        animation: beaconPulse 1.6s infinite;
    }}
    .beacon-pulse.warn {{ background: var(--warn); box-shadow: 0 0 0 0 rgba(160,90,20,0.55); }}
    .beacon-pulse.danger {{ background: var(--danger); box-shadow: 0 0 0 0 rgba(138,42,31,0.55); }}
    .beacon-pulse.muted {{ background: var(--muted); animation: none; }}
    @keyframes beaconPulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(91,108,46,0.55); }}
        70% {{ box-shadow: 0 0 0 7px rgba(91,108,46,0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(91,108,46,0); }}
    }}
    .beacon-run-line {{
        font-size: 13px;
        font-weight: 500;
        line-height: 1.35;
        color: var(--ink);
    }}
    .beacon-run-line .muted {{ color: var(--muted); font-weight: 400; }}
    .beacon-run-meta {{
        font-family: var(--font-mono);
        font-size: 10.5px;
        color: var(--muted);
    }}
    .beacon-progress {{
        height: 5px;
        border-radius: 3px;
        background: rgba(22,23,15,0.06);
        overflow: hidden;
    }}
    body.theme-dark .beacon-progress {{
        background: rgba(241,239,230,0.07);
    }}
    .beacon-progress > div {{
        height: 100%;
        background: var(--signal);
        border-radius: 3px;
        transition: width .4s ease;
    }}

    /* "Re-run setup" footer link inside sidebar */
    .beacon-aside-foot-marker {{ display: none; }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.beacon-aside-foot-marker) [data-testid="stButton"] > button {{
        height: auto;
        min-height: 28px;
        padding: 6px 10px;
        font-family: var(--font-mono);
        font-size: 10.5px;
        color: var(--muted);
        background: transparent;
        border: 0;
        border-radius: 5px;
        box-shadow: none;
        justify-content: flex-start;
        text-align: left;
        font-weight: 500;
        text-transform: none;
        letter-spacing: 0;
    }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.beacon-aside-foot-marker) [data-testid="stButton"] > button:hover {{
        color: var(--ink);
        background: rgba(22,23,15,0.04);
        border: 0;
        transform: none;
    }}

    /* ── Status pills (used in jobs table + match rows) ── */
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 2px 9px;
        border-radius: 99px;
        font-size: 11px;
        font-weight: 500;
        font-family: var(--font-mono);
        background: var(--bg-2);
        color: var(--ink-2);
        border: 1px solid var(--line);
        line-height: 1.4;
        text-transform: lowercase;
    }}
    .status-pill.applied   {{ background: var(--pop-soft);    color: var(--pop);    border-color: transparent; }}
    .status-pill.interest  {{ background: var(--signal-soft); color: var(--signal); border-color: transparent; }}
    .status-pill.reply     {{ background: var(--warn-soft);   color: var(--warn);   border-color: transparent; }}
    .status-pill.skip      {{ color: var(--muted); background: transparent; border-color: var(--line); }}
    .status-pill.new       {{ background: var(--bg-2); color: var(--ink-2); }}

    /* ── Rating segmented control (5-button labeled, with compact variant) ── */
    .rating {{
        display: inline-flex;
        align-items: stretch;
        flex-wrap: wrap;
        background: var(--bg-2);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 3px;
        gap: 0;
    }}
    .rating .rt {{
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 6px 11px;
        border-radius: 5px;
        font-size: 11.5px;
        font-weight: 500;
        color: var(--ink-2);
        background: transparent;
        border: 1px solid transparent;
        cursor: pointer;
        font-family: var(--font-body);
        line-height: 1.2;
        transition: background .14s ease, color .14s ease, border-color .14s ease, box-shadow .14s ease, transform .08s ease;
    }}
    .rating .rt:hover {{ background: rgba(22,23,15,0.06); color: var(--ink); }}
    body.theme-dark .rating .rt:hover {{ background: rgba(241,239,230,0.07); }}
    .rating .rt-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--line-3);
        transition: background .12s ease, transform .12s ease, box-shadow .14s ease;
        flex: 0 0 auto;
    }}
    .rating .rt-1 .rt-dot {{ background: var(--signal); }}
    .rating .rt-2 .rt-dot {{ background: var(--signal-2); }}
    .rating .rt-3 .rt-dot {{ background: var(--ink-3); opacity: .55; }}
    .rating .rt-4 .rt-dot {{ background: var(--warn); }}
    .rating .rt-5 .rt-dot {{ background: var(--muted); opacity: .65; }}
    .rating .rt .k {{
        font-family: var(--font-mono);
        font-size: 9.5px;
        color: var(--muted);
        margin-left: 2px;
        letter-spacing: .04em;
    }}
    .rating .rt.on {{
        background: var(--surface);
        border-color: var(--line);
        box-shadow: 0 1px 0 rgba(22,23,15,0.04);
        color: var(--ink);
        font-weight: 600;
    }}
    .rating .rt-1.on {{ background: var(--signal-soft); border-color: transparent; color: var(--signal); }}
    .rating .rt-2.on {{ background: var(--signal-soft); border-color: transparent; color: var(--signal); }}
    .rating .rt-3.on {{ background: var(--bg); border-color: var(--line-2); color: var(--ink); }}
    .rating .rt-4.on {{ background: var(--warn-soft); border-color: transparent; color: var(--warn); }}
    .rating .rt-5.on {{ background: transparent; border-color: var(--line-2); color: var(--muted); }}
    .rating .rt-5.on .rt-l {{ text-decoration: line-through; text-decoration-color: var(--line-3); text-underline-offset: 2px; }}

    /* compact: dot-only chips, label appears only when selected */
    .rating.compact {{ padding: 0; background: transparent; border: 0; gap: 1px; flex-wrap: nowrap; }}
    .rating.compact .rt {{ padding: 4px 6px; gap: 0; border-radius: 99px; border: 0; background: transparent; }}
    .rating.compact .rt .rt-l, .rating.compact .rt .k {{ display: none; }}
    .rating.compact .rt-dot {{ width: 9px; height: 9px; }}
    .rating.compact .rt:hover {{ background: transparent; }}
    .rating.compact .rt:hover .rt-dot {{ transform: scale(1.45); }}
    .rating.compact .rt.on {{ padding: 3px 9px 3px 6px; background: var(--bg-2); border: 1px solid var(--line); box-shadow: none; }}
    .rating.compact .rt.on .rt-l {{
        display: inline;
        font-size: 10.5px;
        font-family: var(--font-mono);
        margin-left: 7px;
        letter-spacing: .02em;
        text-transform: uppercase;
    }}
    .rating.compact .rt-1.on {{ background: var(--signal-soft); border-color: transparent; }}
    .rating.compact .rt-2.on {{ background: var(--signal-soft); border-color: transparent; }}
    .rating.compact .rt-4.on {{ background: var(--warn-soft); border-color: transparent; }}

    /* ── Rating notes textarea (rendered next to rating buttons) ── */
    .rating-notes-label {{
        font-family: var(--font-mono);
        font-size: 10px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: .08em;
        font-weight: 500;
        margin-top: 8px;
    }}
    .rating-notes-confirm {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: var(--font-mono);
        font-size: 11px;
        color: var(--signal);
        margin-top: 4px;
    }}

    /* ── Eyebrow + display headers (page-header in Beacon vocabulary) ── */
    .shell-page-header-eyebrow,
    .shell-section-eyebrow,
    .shell-sidebar-kicker,
    .shell-app-header-kicker {{
        font-family: var(--font-mono);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 10.5px;
    }}

    /* Drop the heavy avatar circle from page-header (Beacon doesn't use one). */
    .shell-page-header-avatar {{ display: none; }}
    .shell-page-header-identity {{ gap: 0; }}

    /* ── Chunk 2 Beacon components ───────────────────────────────────────── */

    /* Page header */
    .beacon-ph {{
        display: flex;
        flex-direction: column;
        gap: 6px;
        margin-bottom: 18px;
    }}
    .beacon-ph .eyebrow {{
        font-family: var(--font-mono);
        font-size: 10.5px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 500;
    }}
    .beacon-ph h1 {{
        font-family: var(--font-display);
        font-size: 30px;
        line-height: 1.05;
        letter-spacing: -0.03em;
        font-weight: 600;
        color: var(--ink);
        margin: 0;
    }}
    .beacon-ph .sub {{
        margin-top: 4px;
        color: var(--ink-2);
        font-size: 14px;
        max-width: 64ch;
        line-height: 1.5;
    }}

    /* Run banner */
    .runban {{
        display: flex;
        flex-direction: column;
        gap: 6px;
        padding: 14px 18px;
        border-radius: var(--r-md);
        background: var(--surface);
        border: 1px solid var(--line);
        border-left: 3px solid var(--signal);
        box-shadow: var(--shadow-1);
    }}
    .runban.fail {{
        border-left-color: var(--danger);
        background: linear-gradient(0deg, var(--danger-soft), var(--surface) 70%);
    }}
    .runban.partial {{ border-left-color: var(--warn); }}
    .runban.running {{ border-left-color: var(--pop); }}
    .runban .rb-state {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: var(--font-mono);
        font-size: 10.5px;
        text-transform: uppercase;
        letter-spacing: .1em;
        color: var(--signal);
    }}
    .runban.fail .rb-state {{ color: var(--danger); }}
    .runban.partial .rb-state {{ color: var(--warn); }}
    .runban.running .rb-state {{ color: var(--pop); }}
    .runban .rb-dot {{ width: 8px; height: 8px; border-radius: 50%; background: currentColor; }}
    @keyframes beacon-pulse {{
        0%   {{ box-shadow: 0 0 0 0 rgba(31,61,138,0.55); }}
        70%  {{ box-shadow: 0 0 0 7px rgba(31,61,138,0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(31,61,138,0); }}
    }}
    .runban.running .rb-dot {{ animation: beacon-pulse 1.6s infinite; }}
    .runban .rb-line {{
        font-family: var(--font-display);
        font-size: 15px;
        font-weight: 600;
        letter-spacing: -0.01em;
        line-height: 1.35;
        color: var(--ink);
    }}

    /* Verdict badge */
    .vbadge {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 9px 3px 7px;
        border-radius: 99px;
        font-family: var(--font-mono);
        font-size: 10.5px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .06em;
        line-height: 1;
    }}
    .vbadge .vd {{ width: 6px; height: 6px; border-radius: 50%; background: currentColor; }}
    .vbadge.apply {{ background: var(--signal-soft); color: var(--signal); }}
    .vbadge.maybe {{ background: var(--warn-soft); color: var(--warn); }}
    .vbadge.skip  {{ color: var(--muted); background: transparent; border: 1px solid var(--line-2); }}

    /* Match row (used in Overview top picks) */
    .beacon-match-row {{
        display: grid;
        grid-template-columns: 60px 1fr;
        gap: 16px;
        padding: 18px 4px 8px;
        border-bottom: 1px solid var(--line);
        align-items: flex-start;
    }}
    .beacon-match-row.focus {{ background: var(--bg-2); box-shadow: inset 3px 0 0 var(--ink); }}
    .beacon-match-fit {{ display: flex; flex-direction: column; align-items: center; gap: 2px; padding-top: 1px; }}
    .fit-num {{
        font-family: var(--font-display);
        font-weight: 700;
        font-size: 26px;
        line-height: 1;
        letter-spacing: -0.03em;
        color: var(--signal);
    }}
    .fit-lbl {{
        font-family: var(--font-mono);
        font-size: 9px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 1px;
    }}
    .beacon-match-row.mid .fit-num {{ color: var(--ink-2); }}
    .beacon-match-row.low .fit-num {{ color: var(--muted); }}
    .beacon-match-title-row {{
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }}
    .m-title {{
        font-family: var(--font-display);
        font-weight: 600;
        font-size: 16px;
        line-height: 1.25;
        letter-spacing: -0.012em;
        color: var(--ink);
        margin: 0;
    }}
    .m-meta {{
        font-family: var(--font-mono);
        font-size: 11px;
        color: var(--muted);
        margin-top: 5px;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        align-items: center;
        line-height: 1.6;
    }}
    .m-meta .dot {{ width: 2.5px; height: 2.5px; border-radius: 50%; background: var(--muted); }}
    .m-meta .co {{ color: var(--ink-2); font-weight: 500; }}
    .m-reason {{
        font-size: 13.5px;
        color: var(--ink-2);
        line-height: 1.55;
        margin-top: 8px;
        margin-bottom: 0;
        max-width: 64ch;
    }}

    /* Tags + rated tag */
    .tag {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2.5px 8px;
        border-radius: 99px;
        font-size: 11px;
        font-weight: 500;
        background: var(--bg-2);
        color: var(--ink-2);
        border: 1px solid var(--line);
        font-family: var(--font-mono);
    }}
    .tag.pos {{ background: var(--signal-soft); color: var(--signal); border-color: transparent; }}
    .tag.warn {{ background: var(--warn-soft); color: var(--warn); border-color: transparent; }}
    .tag.pop {{ background: var(--pop-soft); color: var(--pop); border-color: transparent; }}
    .rated-tag {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 2.5px 8px 2.5px 7px;
        border-radius: 99px;
        font-family: var(--font-mono);
        font-size: 10px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: .06em;
        line-height: 1;
        background: var(--bg-2);
        color: var(--ink-2);
        border: 1px solid var(--line);
    }}
    .rated-tag .rt-dot {{ width: 6px; height: 6px; border-radius: 50%; background: currentColor; }}

    /* Drawer (st.dialog) — train card, reasoning, dim grid */
    .train-card {{
        background: linear-gradient(180deg, var(--surface) 0%, var(--surface) 60%, var(--bg-2) 100%);
        border: 1px solid var(--line);
        border-radius: var(--r-md);
        padding: 18px 20px;
        display: flex;
        flex-direction: column;
        gap: 12px;
        box-shadow: var(--shadow-1);
    }}
    .train-title {{
        font-family: var(--font-display);
        font-size: 15px;
        font-weight: 600;
        letter-spacing: -0.012em;
        line-height: 1.25;
    }}
    .train-title .train-ic {{ display: inline-block; margin-right: 6px; color: var(--signal); }}
    .train-sub {{
        font-size: 12.5px;
        color: var(--muted);
        line-height: 1.5;
        margin-top: 3px;
        max-width: 50ch;
    }}
    .reasoning {{
        display: flex;
        flex-direction: column;
        gap: 8px;
        padding: 0;
        margin: 0;
        list-style: none;
    }}
    .reasoning li {{
        padding: 11px 13px;
        border: 1px solid var(--line);
        border-radius: var(--r-sm);
        background: var(--surface);
        font-size: 13.5px;
        color: var(--ink-2);
        line-height: 1.55;
        display: flex;
        gap: 11px;
    }}
    .reasoning li::before {{
        content: "";
        flex: 0 0 4px;
        background: var(--signal);
        border-radius: 2px;
    }}
    .reasoning li.warn::before {{ background: var(--warn); }}
    .reasoning li.neg::before {{ background: var(--danger); }}

    .dim-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    @media (max-width: 480px) {{ .dim-grid {{ grid-template-columns: 1fr; }} }}
    .dim {{
        border: 1px solid var(--line);
        border-radius: var(--r-sm);
        background: var(--surface);
        padding: 11px 13px;
        display: flex;
        flex-direction: column;
        gap: 6px;
    }}
    .dim .dim-row {{ display: flex; justify-content: space-between; align-items: baseline; }}
    .dim .nm {{ font-size: 12px; font-weight: 600; color: var(--ink-2); }}
    .dim .v {{ font-family: var(--font-display); font-weight: 600; font-size: 15px; letter-spacing: -0.02em; }}
    .dim .v small {{ font-family: var(--font-mono); font-size: 10px; color: var(--muted); margin-left: 3px; font-weight: 500; }}
    .dim .w {{ font-family: var(--font-mono); font-size: 10px; color: var(--muted); }}
    .dim .bar {{ height: 5px; border-radius: 3px; background: rgba(22,23,15,0.06); overflow: hidden; }}
    .dim .bar > div {{ height: 100%; background: var(--signal); border-radius: 3px; }}
    .dim.low .bar > div {{ background: var(--warn); }}

    /* Drawer header HTML used by st.dialog body */
    .drawer-head {{
        display: flex;
        align-items: flex-start;
        gap: 14px;
        padding-bottom: 14px;
        border-bottom: 1px solid var(--line);
    }}
    .drawer-head .fit-num {{ font-size: 32px; }}

    /* Jobs toolbar (search + filter chips + counter) */
    .beacon-toolbar {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 14px;
        border-bottom: 1px solid var(--line);
        background: var(--bg-2);
        border-radius: var(--r-md) var(--r-md) 0 0;
        flex-wrap: wrap;
    }}
    .beacon-toolbar .right {{
        margin-left: auto;
        display: flex;
        gap: 8px;
        align-items: center;
        font-family: var(--font-mono);
        font-size: 11px;
        color: var(--muted);
    }}

    /* Filter chip-buttons (rendered via st.button targeting) */
    .beacon-filter-row [data-testid="stButton"] > button {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 10px;
        border-radius: 6px;
        background: var(--surface);
        border: 1px solid var(--line);
        font-family: var(--font-body);
        font-size: 12px;
        color: var(--ink-2);
        line-height: 1;
        font-weight: 500;
        min-height: 30px;
        transition: background .12s ease, border-color .12s ease, color .12s ease;
    }}
    .beacon-filter-row [data-testid="stButton"] > button:hover {{
        border-color: var(--line-2);
        color: var(--ink);
    }}
    .beacon-filter-row.on [data-testid="stButton"] > button,
    .beacon-filter-row [data-testid="stButton"] > button[kind="primary"] {{
        background: var(--ink);
        color: var(--accent-ink);
        border-color: var(--ink);
    }}

    /* Bulk-action bar (appears when rows selected) */
    .bulk-bar {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 9px 14px;
        background: var(--ink);
        color: var(--accent-ink);
        border-bottom: 1px solid var(--ink);
        font-size: 12.5px;
    }}
    .bulk-bar .ct {{ font-family: var(--font-mono); }}

    /* Beacon empty-state (used in Jobs no-match) */
    .beacon-empty {{
        padding: 48px 32px;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 8px;
    }}
    .beacon-empty .ic {{
        width: 42px;
        height: 42px;
        border-radius: 50%;
        background: var(--bg-2);
        display: grid;
        place-items: center;
        margin-bottom: 6px;
        font-family: var(--font-mono);
        color: var(--muted);
        font-size: 18px;
    }}
    .beacon-empty .h {{ font-family: var(--font-display); font-size: 16px; font-weight: 600; }}
    .beacon-empty .s {{ font-size: 13px; color: var(--muted); max-width: 42ch; line-height: 1.5; }}

    /* Keyboard hints strip */
    .kbd-hints {{
        display: flex;
        gap: 14px;
        align-items: center;
        padding: 9px 18px;
        flex-wrap: wrap;
        font-family: var(--font-mono);
        font-size: 10.5px;
        color: var(--muted);
        border-top: 1px solid var(--line);
        background: var(--bg-2);
        border-radius: 0 0 var(--r-md) var(--r-md);
    }}
    .kbd-hints .ki {{ display: inline-flex; align-items: center; gap: 5px; }}
    .kbd-hints .ki kbd {{
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 3px;
        padding: 1px 5px;
        font: inherit;
        color: var(--ink);
        font-size: 10px;
    }}

    /* Live pipeline card on Overview side column */
    .pipeline-side-card {{
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: var(--r-md);
        padding: 14px 16px;
        box-shadow: var(--shadow-1);
        display: flex;
        flex-direction: column;
        gap: 10px;
    }}
    .pipeline-side-card .ag-head {{
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .pipeline-side-card .ag-pulse {{
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: var(--signal);
        box-shadow: 0 0 0 0 rgba(91,108,46,0.5);
        animation: beacon-pulse-signal 1.6s infinite;
    }}
    @keyframes beacon-pulse-signal {{
        0%   {{ box-shadow: 0 0 0 0 rgba(91,108,46,0.5); }}
        70%  {{ box-shadow: 0 0 0 7px rgba(91,108,46,0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(91,108,46,0); }}
    }}
    .pipeline-side-card .ag-title {{
        font-family: var(--font-display);
        font-size: 13.5px;
        font-weight: 600;
        letter-spacing: -0.01em;
    }}
    .pipeline-side-card .ag-elapsed {{
        margin-left: auto;
        font-family: var(--font-mono);
        font-size: 11px;
        color: var(--muted);
    }}
    .pipeline-side-card .progress {{
        height: 5px;
        border-radius: 3px;
        background: rgba(22,23,15,0.06);
        overflow: hidden;
    }}
    .pipeline-side-card .progress > div {{
        height: 100%;
        background: var(--signal);
        border-radius: 3px;
        transition: width .4s ease;
    }}
    .pipeline-side-card .stages {{ display: flex; flex-direction: column; gap: 6px; }}
    .pipeline-side-card .stage {{
        display: grid;
        grid-template-columns: 14px 1fr auto;
        gap: 10px;
        align-items: center;
        font-size: 12.5px;
    }}
    .pipeline-side-card .stg-icon {{
        width: 14px;
        height: 14px;
        border-radius: 50%;
        border: 1.5px solid var(--line-2);
        background: var(--surface);
        flex: 0 0 auto;
    }}
    .pipeline-side-card .stage.done .stg-icon {{ background: var(--ink); border-color: var(--ink); }}
    .pipeline-side-card .stage.run .stg-icon  {{ border-color: var(--signal); background: var(--signal); }}
    .pipeline-side-card .stage.queue {{ color: var(--muted); }}
    .pipeline-side-card .stg-name {{ font-weight: 500; color: var(--ink-2); }}
    .pipeline-side-card .stage.run .stg-name {{ color: var(--ink); font-weight: 600; }}
    .pipeline-side-card .stg-count {{
        font-family: var(--font-mono);
        font-size: 10.5px;
        color: var(--muted);
    }}

    /* Card wrapper for the top-picks + jobs sections */
    .beacon-card {{
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: var(--r-md);
        box-shadow: var(--shadow-1);
        overflow: hidden;
    }}
    .beacon-card-head {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 14px 18px;
        border-bottom: 1px solid var(--line);
    }}
    .beacon-card-title {{
        font-family: var(--font-display);
        font-size: 14px;
        font-weight: 600;
        letter-spacing: -0.012em;
        color: var(--ink);
    }}
    .beacon-card-sub {{
        font-size: 12.5px;
        color: var(--muted);
        margin-top: 2px;
        line-height: 1.4;
    }}
    .beacon-card-body {{ padding: 4px 18px 8px; }}

    /* ── Activity feed (chunk 3) ── */
    .feed {{ display: flex; flex-direction: column; }}
    .feed-row {{
        display: grid;
        grid-template-columns: 90px 14px 1fr auto;
        gap: 14px;
        padding: 13px 18px;
        border-bottom: 1px solid var(--line);
        align-items: flex-start;
        font-size: 13px;
    }}
    .feed-row:last-child {{ border-bottom: 0; }}
    .feed-row .ts {{ font-family: var(--font-mono); font-size: 11px; color: var(--muted); padding-top: 1px; }}
    .feed-row .dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--ink-2); margin-top: 6px; }}
    .feed-row.info .dot    {{ background: var(--pop); }}
    .feed-row.warn .dot    {{ background: var(--warn); }}
    .feed-row.danger .dot  {{ background: var(--danger); }}
    .feed-row.success .dot {{ background: var(--signal); }}
    .feed-row .body {{ line-height: 1.5; color: var(--ink-2); }}
    .feed-row .body b {{ color: var(--ink); font-weight: 600; }}
    .feed-row .right {{ font-family: var(--font-mono); font-size: 11px; color: var(--muted); }}

    /* Activity run header — 2-col Stages | Sources */
    .activity-run-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 24px;
        padding: 18px 20px;
    }}
    @media (max-width: 980px) {{ .activity-run-grid {{ grid-template-columns: 1fr; }} }}
    .activity-run-grid .col-eyebrow {{
        font-family: var(--font-mono);
        font-size: 10.5px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        margin-bottom: 10px;
    }}
    .activity-stage-list {{ display: flex; flex-direction: column; gap: 6px; font-size: 12.5px; }}
    .activity-stage {{
        display: grid;
        grid-template-columns: 14px 1fr auto;
        gap: 10px;
        align-items: center;
    }}
    .activity-stage .stg-icon {{
        width: 14px; height: 14px; border-radius: 50%;
        border: 1.5px solid var(--line-2);
        background: var(--surface);
    }}
    .activity-stage.done .stg-icon  {{ background: var(--ink); border-color: var(--ink); }}
    .activity-stage.run .stg-icon   {{ border-color: var(--signal); background: var(--signal); }}
    .activity-stage.fail .stg-icon  {{ border-color: var(--danger); background: var(--danger); }}
    .activity-stage.queue {{ color: var(--muted); }}
    .activity-stage .stg-name {{ font-weight: 500; color: var(--ink-2); }}
    .activity-stage.run .stg-name  {{ color: var(--ink); font-weight: 600; }}
    .activity-stage .stg-count {{ font-family: var(--font-mono); font-size: 10.5px; color: var(--muted); }}

    .activity-source-list {{ display: flex; flex-direction: column; gap: 8px; }}
    .activity-source-row {{
        display: grid;
        grid-template-columns: 96px 1fr 56px;
        gap: 10px;
        align-items: center;
        font-size: 12px;
    }}
    .activity-source-row .nm {{ font-family: var(--font-mono); color: var(--ink-2); font-size: 11px; }}
    .activity-source-row .progress {{
        height: 5px; border-radius: 3px;
        background: rgba(22,23,15,0.06); overflow: hidden;
    }}
    body.theme-dark .activity-source-row .progress {{ background: rgba(241,239,230,0.10); }}
    .activity-source-row .progress > div {{ height: 100%; border-radius: 3px; transition: width .25s ease; }}
    .activity-source-row .ct {{
        font-family: var(--font-mono); font-size: 11px;
        color: var(--muted); text-align: right;
    }}

    /* 14-day run history bar chart */
    .history-chart {{
        padding: 22px 26px;
        display: grid;
        grid-template-columns: repeat(14, 1fr);
        gap: 8px;
        align-items: end;
        height: 120px;
    }}
    .history-chart .bar {{
        background: var(--ink-2);
        opacity: 0.3;
        border-radius: 4px;
        min-height: 6px;
        transition: opacity .15s ease;
    }}
    .history-chart .bar:hover {{ opacity: 0.55; }}
    .history-chart .bar.today {{ background: var(--ink); opacity: 1; }}
    .history-axis {{
        padding: 0 26px 20px;
        display: flex;
        justify-content: space-between;
        font-family: var(--font-mono);
        font-size: 10px;
        color: var(--muted);
    }}

    /* ── Profile / Settings shared (chunk 3) ── */
    .panel-grid {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 18px;
    }}
    @media (max-width: 980px) {{ .panel-grid {{ grid-template-columns: 1fr; }} }}

    .kv {{
        display: grid;
        grid-template-columns: 140px 1fr;
        gap: 8px;
        padding: 9px 0;
        border-bottom: 1px dashed var(--line);
        font-size: 13px;
    }}
    .kv:last-child {{ border-bottom: 0; }}
    .kv .k {{
        font-family: var(--font-mono);
        font-size: 10.5px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding-top: 2px;
    }}
    .kv .v {{ color: var(--ink-2); }}

    .pill-grid {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 99px;
        background: var(--bg-2);
        border: 1px solid var(--line);
        font-size: 12.5px;
        color: var(--ink-2);
    }}
    .pill.on {{ background: var(--ink); color: var(--accent-ink); border-color: var(--ink); }}
    .pill .x {{ opacity: 0.55; font-family: var(--font-mono); font-size: 10px; }}

    /* Resume preview thumb tile */
    .resume-thumb {{
        width: 44px;
        height: 56px;
        border: 1px solid var(--line);
        border-radius: 4px;
        display: grid;
        place-items: center;
        font-family: var(--font-mono);
        font-size: 10.5px;
        color: var(--muted);
        flex: 0 0 auto;
    }}
    .resume-row {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .resume-row .grow {{ flex: 1 1 auto; min-width: 0; }}
    .resume-row .grow .name {{ font-weight: 600; font-size: 13.5px; }}
    .resume-row .grow .meta {{ font-family: var(--font-mono); font-size: 11px; color: var(--muted); margin-top: 2px; }}

    /* Settings — sources / weights / switch (chunk 3) */
    .source-tile {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        padding: 12px 14px;
        border: 1px solid var(--line);
        border-radius: var(--r-sm);
        background: var(--surface);
    }}
    .source-tile .nm {{ font-weight: 600; font-size: 13px; display: flex; align-items: center; gap: 8px; }}
    .source-tile .nm .d {{ width: 8px; height: 8px; border-radius: 50%; background: var(--signal); }}
    .source-tile.off .nm .d {{ background: var(--muted); }}
    .source-tile .ct {{ font-family: var(--font-mono); font-size: 11px; color: var(--muted); }}

    .switch {{
        position: relative;
        width: 32px;
        height: 18px;
        border-radius: 99px;
        background: rgba(22,23,15,0.18);
        flex: 0 0 auto;
    }}
    .switch::after {{
        content: "";
        position: absolute;
        left: 2px; top: 2px;
        width: 14px; height: 14px;
        border-radius: 50%;
        background: #fff;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2);
        transition: left .15s;
    }}
    .switch.on {{ background: var(--ink); }}
    .switch.on::after {{ left: 16px; }}
    body.theme-dark .switch {{ background: rgba(241,239,230,0.18); }}
    body.theme-dark .switch.on {{ background: var(--accent); }}
    body.theme-dark .switch.on::after {{ background: var(--bg); }}

    .weight-row {{
        display: grid;
        grid-template-columns: 120px 1fr 50px;
        gap: 10px;
        align-items: center;
        padding: 7px 0;
        font-size: 13px;
    }}
    .weight-row .nm {{ font-weight: 500; }}
    .weight-row .v {{ font-family: var(--font-mono); font-size: 12px; color: var(--muted); text-align: right; }}

    /* Notification rows */
    .notif-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 8px 0;
        border-bottom: 1px dashed var(--line);
    }}
    .notif-row:last-child {{ border-bottom: 0; }}
    .notif-row .label {{ font-weight: 600; font-size: 13.5px; }}
    .notif-row .sub   {{ color: var(--muted); font-size: 12px; margin-top: 3px; }}
    .notif-row .soon  {{
        font-family: var(--font-mono);
        font-size: 10px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 5px;
    }}

    /* ── Onboarding (chunk 4) ───────────────────────────────────────────── */
    .onb-wrap {{
        max-width: 980px;
        margin: 8px auto 24px;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: var(--r-lg);
        box-shadow: var(--shadow-2);
        overflow: hidden;
    }}
    .onb-side {{
        padding: 6px 4px 12px;
    }}
    .onb-side .onb-side-takes {{
        font-family: var(--font-mono);
        font-size: 10.5px;
        color: var(--muted);
        margin-top: 18px;
        letter-spacing: 0.04em;
    }}
    .onb-steps {{ display: flex; flex-direction: column; gap: 2px; margin-top: 8px; }}
    .onb-step {{
        display: grid;
        grid-template-columns: 22px 1fr;
        gap: 10px;
        align-items: flex-start;
        padding: 10px 8px;
        border-radius: 8px;
    }}
    .onb-step .num {{
        width: 22px; height: 22px;
        border-radius: 50%;
        border: 1.5px solid var(--line-2);
        background: var(--surface);
        display: grid; place-items: center;
        font-family: var(--font-mono); font-size: 10px; color: var(--muted);
    }}
    .onb-step.done .num {{ background: var(--ink); color: var(--accent-ink); border-color: var(--ink); }}
    .onb-step.active {{ background: var(--bg-2); border: 1px solid var(--line); }}
    .onb-step.active .num {{ border-color: var(--ink); color: var(--ink); background: var(--surface); font-weight: 700; }}
    .onb-step .nm {{ font-size: 13px; font-weight: 600; line-height: 1.2; }}
    .onb-step .ds {{ font-size: 11.5px; color: var(--muted); margin-top: 3px; line-height: 1.4; }}
    .onb-step:not(.active) .nm {{ color: var(--ink-2); }}

    .onb-eyebrow {{
        font-family: var(--font-mono);
        font-size: 10.5px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    .onb-h2 {{
        font-family: var(--font-display);
        font-size: 26px;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin: 4px 0 6px;
        color: var(--ink);
    }}
    .onb-hint {{
        color: var(--muted);
        font-size: 13.5px;
        line-height: 1.55;
        max-width: 56ch;
    }}
    .onb-foot-pg {{
        font-family: var(--font-mono);
        font-size: 11px;
        color: var(--muted);
        text-align: center;
        padding-top: 10px;
    }}

    /* Value-prop cards (Welcome step) */
    .onb-value-card {{
        border: 1px solid var(--line);
        background: var(--bg-2);
        border-radius: var(--r-md);
        padding: 14px 16px;
        height: 100%;
    }}
    .onb-value-card .t {{ font-weight: 600; font-size: 13.5px; color: var(--ink); }}
    .onb-value-card .d {{ color: var(--muted); font-size: 12.5px; line-height: 1.55; margin-top: 6px; }}

    /* Choice cards (Sources / Cadence steps) */
    .choice {{
        display: block;
        border: 1px solid var(--line-2);
        border-radius: var(--r-md);
        padding: 14px 16px;
        background: var(--surface);
    }}
    .choice.on {{ border-color: var(--ink); background: var(--bg-2); box-shadow: 0 0 0 3px rgba(22,23,15,0.05); }}
    body.theme-dark .choice.on {{ box-shadow: 0 0 0 3px rgba(241,239,230,0.06); }}
    .choice .top {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; }}
    .choice h4 {{ font-size: 14px; font-weight: 600; margin: 0; color: var(--ink); }}
    .choice p {{ margin: 4px 0 0; color: var(--muted); font-size: 12.5px; line-height: 1.5; }}
    .choice .meta {{
        font-family: var(--font-mono);
        font-size: 10.5px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    /* Drop-zone for resume (Resume step) */
    .onb-dropzone {{
        border: 1.5px dashed var(--line-2);
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        background: var(--bg-2);
    }}
    .onb-dropzone .lbl {{
        font-family: var(--font-mono);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--muted);
    }}
    .onb-dropzone .h {{
        font-family: var(--font-display);
        font-size: 18px;
        font-weight: 600;
        margin-top: 6px;
        color: var(--ink);
    }}
    .onb-dropzone .s {{ color: var(--muted); font-size: 12.5px; margin-top: 4px; }}

    /* Theme toggle (segmented Day / Night) */
    .theme-toggle {{
        display: inline-flex;
        border: 1px solid var(--line);
        border-radius: 99px;
        background: var(--surface);
        overflow: hidden;
        padding: 2px;
        gap: 1px;
    }}
    .theme-toggle button {{
        padding: 4px 10px;
        font-family: var(--font-mono);
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--muted);
        border-radius: 99px;
        line-height: 1.4;
        background: transparent;
        border: 0;
    }}
    .theme-toggle button.on {{ background: var(--ink); color: var(--accent-ink); }}
    body.theme-dark .theme-toggle button.on {{ background: var(--accent); color: var(--accent-ink); }}
    </style>
    """


def inject_global_css() -> None:
    st.markdown(_global_css(), unsafe_allow_html=True)


def inject_theme_class(theme: str = "mono") -> None:
    """Set the active theme class on the parent document body.

    Beacon's design tokens default to the editorial light "mono" theme via
    `:root`. The `body.theme-dark` override switches the palette in place.
    The `render_theme_toggle()` widget below drives the active theme via
    `st.session_state["beacon_theme"]`; this function just stamps the class.
    """
    safe = "dark" if theme == "dark" else "mono"
    components.html(
        f"""
        <script>
        (function () {{
          const doc = window.parent && window.parent.document;
          if (!doc || !doc.body) return;
          doc.body.classList.remove('theme-mono', 'theme-dark');
          doc.body.classList.add('theme-{safe}');
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def render_theme_toggle(*, key: str = "beacon_theme_toggle") -> None:
    """Render the Day / Night segmented control. Persists to st.session_state["beacon_theme"].

    Apply the resulting class on the next rerun via `inject_theme_class()`,
    which `apply_page_scaffold()` already calls. The toggle is intentionally
    minimal — Streamlit doesn't give us a true topbar to dock it into, so the
    caller decides where it appears (typically the right edge of a thin row at
    the top of the main panel).
    """
    current = st.session_state.get("beacon_theme", "mono")
    label_to_value = {"Day": "mono", "Night": "dark"}
    value_to_label = {v: k for k, v in label_to_value.items()}

    st.markdown(
        "<div class='theme-toggle-host'>"
        "<style>.theme-toggle-host [data-testid='stHorizontalBlock'] {gap:0;}"
        ".theme-toggle-host div[role='radiogroup'] {gap:0 !important;}"
        "</style>",
        unsafe_allow_html=True,
    )
    new_label = st.segmented_control(
        "Theme",
        list(label_to_value.keys()),
        default=value_to_label.get(current, "Day"),
        key=key,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    new_value = label_to_value.get(new_label or "Day", "mono")
    if new_value != current:
        st.session_state["beacon_theme"] = new_value
        st.rerun()


def inject_sidebar_scroll_guard() -> None:
    components.html(
        """
        <script>
        (function () {
          const parentWindow = window.parent;
          const parentDoc = parentWindow.document;
          if (!parentWindow || !parentDoc) return;

          if (!parentWindow.__shellSidebarScrollGuard) {
            const state = {};

            state.getSidebar = function () {
              return parentDoc.querySelector('[data-testid="stSidebar"]');
            };

            state.getShell = function (sidebar) {
              return sidebar && sidebar.firstElementChild ? sidebar.firstElementChild : sidebar;
            };

            state.getContent = function (sidebar) {
              return sidebar ? sidebar.querySelector('.block-container') : null;
            };

            state.getPointerPoint = function (event) {
              if (event.touches && event.touches.length > 0) {
                return {
                  x: event.touches[0].clientX,
                  y: event.touches[0].clientY,
                };
              }
              if (event.changedTouches && event.changedTouches.length > 0) {
                return {
                  x: event.changedTouches[0].clientX,
                  y: event.changedTouches[0].clientY,
                };
              }
              if (typeof event.clientX === 'number' && typeof event.clientY === 'number') {
                return {
                  x: event.clientX,
                  y: event.clientY,
                };
              }
              return null;
            };

            state.isPointerInsideSidebar = function (event, sidebar) {
              const point = state.getPointerPoint(event);
              if (!point || !sidebar) return false;
              const rect = sidebar.getBoundingClientRect();
              return (
                point.x >= rect.left &&
                point.x <= rect.right &&
                point.y >= rect.top &&
                point.y <= rect.bottom
              );
            };

            state.resetScrollPositions = function (sidebar, shell, content) {
              if (shell) shell.scrollTop = 0;
              if (content) content.scrollTop = 0;
              if (sidebar) {
                sidebar.querySelectorAll('*').forEach(function (node) {
                  if (node && typeof node.scrollTop === 'number' && node.scrollTop !== 0) {
                    node.scrollTop = 0;
                  }
                });
              }
            };

            state.onLockedScroll = function () {
              const sidebar = state.getSidebar();
              if (!sidebar || sidebar.dataset.shellSidebarFits !== 'true') return;
              state.resetScrollPositions(
                sidebar,
                state.getShell(sidebar),
                state.getContent(sidebar)
              );
            };

            state.update = function () {
              const sidebar = state.getSidebar();
              const shell = state.getShell(sidebar);
              const content = state.getContent(sidebar);
              if (!sidebar || !shell || !content) return;

              const fits = content.scrollHeight <= shell.clientHeight + 2;
              sidebar.dataset.shellSidebarFits = fits ? 'true' : 'false';

              if (fits) {
                shell.style.overflowY = 'hidden';
                shell.style.overscrollBehavior = 'none';
                content.style.overflowY = 'visible';
                content.style.overscrollBehavior = 'none';
                state.resetScrollPositions(sidebar, shell, content);
              } else {
                shell.style.overflowY = 'auto';
                shell.style.overscrollBehavior = 'contain';
                content.style.overflowY = '';
                content.style.overscrollBehavior = '';
              }
            };

            state.stopIfLocked = function (event) {
              const sidebar = state.getSidebar();
              if (!sidebar) return;
              if (sidebar.dataset.shellSidebarFits !== 'true') return;
              if (!state.isPointerInsideSidebar(event, sidebar)) return;

              event.preventDefault();
              event.stopPropagation();
              if (typeof event.stopImmediatePropagation === 'function') {
                event.stopImmediatePropagation();
              }

              state.resetScrollPositions(
                sidebar,
                state.getShell(sidebar),
                state.getContent(sidebar)
              );
            };

            state.observer = new parentWindow.MutationObserver(function () {
              parentWindow.requestAnimationFrame(state.update);
            });

            if (parentDoc.body) {
              state.observer.observe(parentDoc.body, {
                subtree: true,
                childList: true,
                attributes: true,
              });
            }

            parentDoc.addEventListener('wheel', state.stopIfLocked, {
              passive: false,
              capture: true,
            });
            parentDoc.addEventListener('touchmove', state.stopIfLocked, {
              passive: false,
              capture: true,
            });
            parentDoc.addEventListener('scroll', state.onLockedScroll, {
              passive: true,
              capture: true,
            });
            parentWindow.addEventListener('resize', function () {
              parentWindow.requestAnimationFrame(state.update);
            });

            parentWindow.__shellSidebarScrollGuard = state;
          }

          parentWindow.requestAnimationFrame(function () {
            parentWindow.__shellSidebarScrollGuard.update();
          });
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def apply_page_scaffold(
    page_title: str = PAGE_TITLE,
    *,
    header_title: Optional[str] = None,
    header_subtitle: Optional[str] = None,
    header_kicker: str = "Job Search Dashboard",
) -> None:
    global _PAGE_CONFIG_APPLIED
    if not _PAGE_CONFIG_APPLIED:
        st.set_page_config(
            page_title=page_title,
            layout="wide",
            initial_sidebar_state="expanded",
        )
        _PAGE_CONFIG_APPLIED = True

    inject_global_css()
    inject_theme_class(st.session_state.get("beacon_theme", "mono"))
    inject_sidebar_scroll_guard()

    if header_title:
        subtitle_html = ""
        if header_subtitle:
            subtitle_html = f"<div class='shell-app-bar-copy'>{html.escape(header_subtitle)}</div>"
        st.markdown(
            (
                "<section class='shell-app-bar'>"
                "<div>"
                f"<div class='shell-app-header-kicker'>{html.escape(header_kicker)}</div>"
                f"<div class='shell-app-bar-title'>{html.escape(header_title)}</div>"
                f"{subtitle_html}"
                "</div>"
                "</section>"
            ),
            unsafe_allow_html=True,
        )
