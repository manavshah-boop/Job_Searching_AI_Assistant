from __future__ import annotations

import html
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

PAGE_TITLE = "Job Search Dashboard"

COLOR_BG = "#e5edeb"
COLOR_BG_ACCENT = "#dfeae8"
COLOR_SURFACE = "rgba(255, 255, 255, 0.97)"
COLOR_SURFACE_STRONG = "#ffffff"
COLOR_SURFACE_MUTED = "#f2f6f4"
COLOR_SURFACE_INVERSE = "#122226"
COLOR_TEXT = "#142328"
COLOR_MUTED = "#5d7077"
COLOR_BORDER = "rgba(20, 35, 40, 0.13)"
COLOR_BORDER_STRONG = "rgba(20, 35, 40, 0.19)"
COLOR_ACCENT = "#0f766e"
COLOR_ACCENT_SOFT = "#e1f3f0"
COLOR_SUCCESS = "#13795b"
COLOR_SUCCESS_SOFT = "#e8f7f1"
COLOR_WARNING = "#9a5b00"
COLOR_WARNING_SOFT = "#fff3dd"
COLOR_DANGER = "#b42318"
COLOR_DANGER_SOFT = "#fdecea"
COLOR_INFO = "#2457a6"
COLOR_INFO_SOFT = "#eaf1fc"
COLOR_FOCUS = "rgba(15, 118, 110, 0.42)"

RADIUS_SM = "14px"
RADIUS_MD = "20px"
RADIUS_LG = "28px"
RADIUS_XL = "36px"
SHADOW_SOFT = "0 18px 36px rgba(17, 34, 39, 0.07)"
SHADOW_PANEL = "0 12px 28px rgba(17, 34, 39, 0.07)"
SPACE_2XS = "0.35rem"
SPACE_XS = "0.55rem"
SPACE_SM = "0.8rem"
SPACE_MD = "1.1rem"
SPACE_LG = "1.55rem"
SPACE_XL = "2.1rem"
SPACE_2XL = "2.8rem"
FONT_DISPLAY = "'Space Grotesk', 'Segoe UI', sans-serif"
FONT_BODY = "'IBM Plex Sans', 'Aptos', 'Segoe UI', sans-serif"

_PAGE_CONFIG_APPLIED = False


def _global_css() -> str:
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

    :root {{
        --shell-bg: {COLOR_BG};
        --shell-bg-accent: {COLOR_BG_ACCENT};
        --shell-surface: {COLOR_SURFACE};
        --shell-surface-strong: {COLOR_SURFACE_STRONG};
        --shell-surface-muted: {COLOR_SURFACE_MUTED};
        --shell-surface-inverse: {COLOR_SURFACE_INVERSE};
        --shell-text: {COLOR_TEXT};
        --shell-muted: {COLOR_MUTED};
        --shell-border: {COLOR_BORDER};
        --shell-border-strong: {COLOR_BORDER_STRONG};
        --shell-accent: {COLOR_ACCENT};
        --shell-accent-soft: {COLOR_ACCENT_SOFT};
        --shell-success: {COLOR_SUCCESS};
        --shell-success-soft: {COLOR_SUCCESS_SOFT};
        --shell-warning: {COLOR_WARNING};
        --shell-warning-soft: {COLOR_WARNING_SOFT};
        --shell-danger: {COLOR_DANGER};
        --shell-danger-soft: {COLOR_DANGER_SOFT};
        --shell-info: {COLOR_INFO};
        --shell-info-soft: {COLOR_INFO_SOFT};
        --shell-focus: {COLOR_FOCUS};
        --shell-radius-sm: {RADIUS_SM};
        --shell-radius-md: {RADIUS_MD};
        --shell-radius-lg: {RADIUS_LG};
        --shell-radius-xl: {RADIUS_XL};
        --shell-sidebar-top-offset: 1.9rem;
        --shell-shadow: {SHADOW_SOFT};
        --shell-shadow-panel: {SHADOW_PANEL};
        --shell-space-2xs: {SPACE_2XS};
        --shell-space-xs: {SPACE_XS};
        --shell-space-sm: {SPACE_SM};
        --shell-space-md: {SPACE_MD};
        --shell-space-lg: {SPACE_LG};
        --shell-space-xl: {SPACE_XL};
        --shell-space-2xl: {SPACE_2XL};
        --shell-font-display: {FONT_DISPLAY};
        --shell-font-body: {FONT_BODY};
    }}

    html, body, [class*="css"] {{
        font-family: var(--shell-font-body);
        color: var(--shell-text);
    }}

    body {{
        background: var(--shell-bg);
    }}

    .stApp {{
        background:
            radial-gradient(circle at top left, rgba(15, 118, 110, 0.07), transparent 25%),
            radial-gradient(circle at top right, rgba(36, 87, 166, 0.05), transparent 22%),
            linear-gradient(180deg, #f8fbfa 0%, var(--shell-bg) 100%);
        color: var(--shell-text);
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
        background:
            linear-gradient(180deg, rgba(250, 252, 252, 0.98) 0%, rgba(243, 248, 247, 0.98) 100%);
        border-right: 1px solid var(--shell-border-strong);
        box-shadow: 8px 0 24px rgba(17, 34, 39, 0.04);
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
        margin-top: 0.18rem;
        margin-bottom: 0;
    }}

    .shell-toolbar--sidebar-primary div[data-testid="stButton"] > button[kind="primary"] {{
        min-height: 2.55rem;
        border-radius: 10px;
        box-shadow: 0 8px 18px rgba(17, 34, 39, 0.06);
    }}

    .shell-toolbar--sidebar-subtle {{
        margin-top: 0;
        margin-bottom: 0;
    }}

    .shell-toolbar--sidebar-subtle div[data-testid="stButton"] > button[kind="secondary"] {{
        min-height: 2rem;
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
        margin: 0 0 0.08rem 0;
    }}

    .shell-sidebar-spacer {{
        flex: 0 0 auto;
        min-height: 0.35rem;
    }}

    .shell-sidebar-actions {{
        display: grid;
        gap: 0.04rem;
        padding-top: 0;
        border-top: 0;
    }}

    .shell-sidebar-actions .shell-toolbar--sidebar-primary {{
        margin-top: 0;
        margin-bottom: 0.22rem;
    }}

    .shell-sidebar-actions .shell-toolbar--sidebar-subtle {{
        margin-top: 0;
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
    </style>
    """


def inject_global_css() -> None:
    st.markdown(_global_css(), unsafe_allow_html=True)


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
