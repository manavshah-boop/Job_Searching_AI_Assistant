from __future__ import annotations

from contextlib import contextmanager
import html
from typing import Any, Iterator

import streamlit as st

MetricItem = tuple[str, Any] | tuple[str, Any, Any | None]


@contextmanager
def panel(title: str, subtitle: str | None = None, icon: str | None = None) -> Iterator[None]:
    """Render a consistent bordered panel with an optional header and subtitle."""
    with st.container(border=True):
        heading = " ".join(part for part in (icon, title) if part).strip()
        subtitle_html = ""
        if subtitle:
            subtitle_html = f"<div class='shell-panel-subtitle'>{html.escape(subtitle)}</div>"
        st.markdown(
            (
                "<div class='shell-panel-head'>"
                f"<div class='shell-panel-title'>{html.escape(heading)}</div>"
                f"{subtitle_html}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        yield


def stat_row(items: list[MetricItem]) -> None:
    """Render a compact metric row with optional deltas."""
    if not items:
        return

    cols = st.columns(len(items), gap="medium")
    for col, item in zip(cols, items):
        label = item[0]
        value = item[1]
        delta = item[2] if len(item) > 2 else None
        with col:
            if delta is None:
                st.metric(label, value)
            else:
                st.metric(label, value, delta=delta)


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


def callout(kind: str, title: str, body: str) -> None:
    """Render a consistent alert block with a heading."""
    renderer = getattr(st, kind, st.info)
    renderer(f"**{title}**\n\n{body}")
