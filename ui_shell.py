from __future__ import annotations

from contextlib import contextmanager
import html
from typing import Any, Iterator, Sequence

import streamlit as st

MetricItem = tuple[str, Any] | tuple[str, Any, Any | None]
ActionSpec = dict[str, Any]


@contextmanager
def panel(
    title: str,
    subtitle: str | None = None,
    icon: str | None = None,
    *,
    tone: str | None = None,
) -> Iterator[None]:
    """Render a consistent panel shell with an optional header and subtitle."""
    with st.container(border=True):
        heading = " ".join(part for part in (icon, title) if part).strip()
        subtitle_html = ""
        if subtitle:
            subtitle_html = f"<div class='shell-panel-subtitle'>{html.escape(subtitle)}</div>"
        marker_html = ""
        if tone:
            safe_tone = html.escape(tone)
            marker_html = f"<div class='shell-panel-marker shell-panel-marker--{safe_tone}'></div>"
        st.markdown(
            (
                f"{marker_html}"
                "<div class='shell-panel-head'>"
                f"<div class='shell-panel-title'>{html.escape(heading)}</div>"
                f"{subtitle_html}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        yield


def page_header(
    title: str,
    subtitle: str,
    *,
    eyebrow: str | None = None,
    chips: Sequence[str] | None = None,
    primary_actions: Sequence[ActionSpec] | None = None,
    secondary_actions: Sequence[ActionSpec] | None = None,
) -> str | None:
    """Render a compact page header with metadata chips and optional actions."""
    initials = "".join(part[:1] for part in str(title).split()[:2]).upper() or "?"
    avatar_html = f"<div class='shell-page-header-avatar' aria-hidden='true'>{html.escape(initials)}</div>"
    eyebrow_html = (
        f"<div class='shell-page-header-eyebrow'>{html.escape(eyebrow)}</div>"
        if eyebrow
        else ""
    )
    chips_html = ""
    if chips:
        rendered = "".join(
            f"<span class='shell-chip'>{html.escape(str(chip))}</span>"
            for chip in chips
            if str(chip).strip()
        )
        if rendered:
            chips_html = f"<div class='shell-chip-row shell-page-header-chips'>{rendered}</div>"

    actions = list(primary_actions or []) + list(secondary_actions or [])
    if actions:
        main_col, action_col = st.columns([4.7, 1.15], gap="medium")
        with main_col:
            st.markdown(
                (
                    "<section class='shell-page-header'>"
                    "<div class='shell-page-header-main'>"
                    "<div class='shell-page-header-identity'>"
                    f"{avatar_html}"
                    "<div class='shell-page-header-copy'>"
                    f"{eyebrow_html}"
                    f"<h1 class='shell-page-header-title'>{html.escape(title)}</h1>"
                    f"<p class='shell-page-header-subtitle' title='{html.escape(subtitle)}'>{html.escape(subtitle)}</p>"
                    f"{chips_html}"
                    "</div>"
                    "</div>"
                    "</div>"
                    "</section>"
                ),
                unsafe_allow_html=True,
            )
        with action_col:
            st.markdown("<div class='shell-page-header-actions'>", unsafe_allow_html=True)
            clicked = toolbar(
                primary_actions=primary_actions,
                secondary_actions=secondary_actions,
                class_name="shell-toolbar shell-toolbar--header shell-toolbar--compact",
            )
            st.markdown("</div>", unsafe_allow_html=True)
            return clicked

    st.markdown(
        (
            "<section class='shell-page-header'>"
            "<div class='shell-page-header-main'>"
            "<div class='shell-page-header-identity'>"
            f"{avatar_html}"
            "<div class='shell-page-header-copy'>"
            f"{eyebrow_html}"
            f"<h1 class='shell-page-header-title'>{html.escape(title)}</h1>"
            f"<p class='shell-page-header-subtitle' title='{html.escape(subtitle)}'>{html.escape(subtitle)}</p>"
            f"{chips_html}"
            "</div>"
            "</div>"
            "</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )
    return None


@contextmanager
def section_shell(
    title: str,
    subtitle: str | None = None,
    *,
    eyebrow: str | None = None,
) -> Iterator[None]:
    """Render a section wrapper with a light heading band and body area."""
    eyebrow_html = (
        f"<div class='shell-section-eyebrow'>{html.escape(eyebrow)}</div>"
        if eyebrow
        else ""
    )
    subtitle_html = (
        f"<p class='shell-section-copy'>{html.escape(subtitle)}</p>"
        if subtitle
        else ""
    )
    st.markdown(
        (
            "<section class='shell-section'>"
            "<div class='shell-section-header'>"
            f"{eyebrow_html}"
            f"<h2 class='shell-section-title'>{html.escape(title)}</h2>"
            f"{subtitle_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        st.markdown("</section>", unsafe_allow_html=True)


def stat_row(items: list[MetricItem], *, columns_count: int | None = None) -> None:
    """Render a compact metric grid with optional deltas."""
    if not items:
        return

    per_row = max(1, columns_count or len(items))
    row_starts = list(range(0, len(items), per_row))
    for row_index, start in enumerate(row_starts):
        chunk = items[start : start + per_row]
        cols = st.columns(len(chunk), gap="medium")
        for col, item in zip(cols, chunk):
            label = item[0]
            value = item[1]
            delta = item[2] if len(item) > 2 else None
            with col:
                delta_html = ""
                if delta is not None and str(delta).strip():
                    delta_html = f"<div class='shell-stat-delta'>{html.escape(str(delta))}</div>"
                st.markdown(
                    (
                        "<article class='shell-stat-card'>"
                        f"<div class='shell-stat-label'>{html.escape(str(label))}</div>"
                        f"<div class='shell-stat-value'>{html.escape(str(value))}</div>"
                        f"{delta_html}"
                        "</article>"
                    ),
                    unsafe_allow_html=True,
                )
        if row_index < len(row_starts) - 1:
            st.markdown("<div class='shell-stat-row-gap' aria-hidden='true'></div>", unsafe_allow_html=True)


def chip_row(chips: list[str]) -> None:
    """Render small, wrapped chips for metadata and active filters."""
    if not chips:
        return

    rendered = "".join(
        f"<span class='shell-chip'>{html.escape(str(chip))}</span>"
        for chip in chips
        if str(chip).strip()
    )
    if not rendered:
        return

    st.markdown(f"<div class='shell-chip-row'>{rendered}</div>", unsafe_allow_html=True)


def badge(text: str, tone: str = "neutral") -> str:
    """Return a styled badge HTML snippet for inline use in markdown blocks."""
    safe_tone = tone if tone in {"neutral", "info", "success", "warning", "danger"} else "neutral"
    return (
        f"<span class='shell-badge shell-badge--{safe_tone}'>"
        f"{html.escape(str(text))}"
        "</span>"
    )


def callout(kind: str, title: str, body: str) -> None:
    """Render a consistent alert block with a heading."""
    tone = {
        "info": "info",
        "success": "success",
        "warning": "warning",
        "error": "danger",
        "danger": "danger",
    }.get(kind, "info")
    st.markdown(
        (
            f"<div class='shell-callout shell-callout--{tone}'>"
            f"<div class='shell-callout-title'>{html.escape(title)}</div>"
            f"<div>{html.escape(body)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def empty_state(
    title: str,
    body: str,
    actions: Sequence[ActionSpec] | None = None,
    *,
    mark: str = "No results",
    icon: str | None = None,
) -> str | None:
    """Render an empty state and optional action buttons. Returns clicked action id, if any."""
    icon_html = ""
    if icon:
        safe_icon = html.escape(icon)
        icon_html = f"<div class='shell-empty-state-icon shell-empty-state-icon--{safe_icon}' aria-hidden='true'></div>"
    st.markdown(
        (
            "<div class='shell-empty-state'>"
            f"<div class='shell-empty-state-mark'>{html.escape(mark)}</div>"
            f"{icon_html}"
            f"<h3>{html.escape(title)}</h3>"
            f"<p>{html.escape(body)}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    if not actions:
        return None

    cols = st.columns(len(actions), gap="small")
    clicked: str | None = None
    for col, action in zip(cols, actions):
        with col:
            action_id = str(action.get("id") or action.get("label") or "")
            label = str(action.get("label") or action_id)
            key = str(action.get("key") or f"empty_state_{action_id}")
            if action.get("url"):
                st.link_button(label, str(action["url"]), use_container_width=True)
            elif st.button(
                label,
                key=key,
                type=action.get("type", "secondary"),
                use_container_width=True,
                disabled=bool(action.get("disabled", False)),
            ):
                clicked = action_id
    return clicked


def toolbar(
    primary_actions: Sequence[ActionSpec] | None = None,
    secondary_actions: Sequence[ActionSpec] | None = None,
    meta: str | None = None,
    class_name: str = "shell-toolbar",
) -> str | None:
    """Render a consistent toolbar and return the clicked action id."""
    st.markdown(f"<div class='{html.escape(class_name)}'>", unsafe_allow_html=True)
    if meta:
        st.markdown(f"<div class='shell-toolbar-meta'>{html.escape(meta)}</div>", unsafe_allow_html=True)

    actions = list(primary_actions or []) + list(secondary_actions or [])
    if not actions:
        st.markdown("</div>", unsafe_allow_html=True)
        return None

    cols = st.columns(len(actions), gap="small")
    clicked: str | None = None
    primary_count = len(primary_actions or [])
    for index, (col, action) in enumerate(zip(cols, actions)):
        with col:
            action_id = str(action.get("id") or action.get("label") or index)
            label = str(action.get("label") or action_id)
            key = str(action.get("key") or f"toolbar_{action_id}")
            button_type = action.get("type")
            if button_type is None:
                button_type = "primary" if index < primary_count else "secondary"

            if action.get("url"):
                st.link_button(label, str(action["url"]), use_container_width=True)
            elif st.button(
                label,
                key=key,
                type=button_type,
                use_container_width=bool(action.get("use_container_width", True)),
                disabled=bool(action.get("disabled", False)),
                help=action.get("help"),
            ):
                clicked = action_id
    st.markdown("</div>", unsafe_allow_html=True)
    return clicked


def help_tip(label: str, markdown: str) -> None:
    """Render inline help using popover when available, with a safe fallback."""
    popover = getattr(st, "popover", None)
    if callable(popover):
        with popover(label):
            st.markdown(markdown)
        return

    with st.expander(label):
        st.markdown(markdown)


def sidebar_profile_summary(
    profile_name: str,
    *,
    subtitle: str | None = None,
    meta: str | None = None,
    chips: Sequence[str] | None = None,
) -> None:
    """Render a compact sidebar identity block."""
    subtitle_html = (
        f"<p class='shell-sidebar-copy'>{html.escape(subtitle)}</p>"
        if subtitle
        else ""
    )
    meta_html = (
        f"<div class='shell-sidebar-meta'>{html.escape(meta)}</div>"
        if meta
        else ""
    )
    chips_html = ""
    if chips:
        rendered = "".join(
            f"<span class='shell-chip'>{html.escape(str(chip))}</span>"
            for chip in chips
            if str(chip).strip()
        )
        if rendered:
            chips_html = f"<div class='shell-chip-row shell-chip-row--sidebar'>{rendered}</div>"

    st.markdown(
        (
            "<section class='shell-sidebar-profile'>"
            "<div class='shell-sidebar-body'>"
            f"<div class='shell-sidebar-title'>{html.escape(profile_name)}</div>"
            f"{subtitle_html}"
            f"{meta_html}"
            f"{chips_html}"
            "</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )
