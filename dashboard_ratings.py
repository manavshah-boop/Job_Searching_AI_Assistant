"""dashboard_ratings.py — Rating-button UI + factor-source badges.

Bridges `user_ratings` (storage) and the Streamlit dashboards (rendering). Kept
in its own module so neither dashboard.py nor dashboard_semantic.py needs to
duplicate the rating-button block. The storage layer stays Streamlit-free.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

from match_explainer import FactorExplanation
from user_ratings import (
    RATING_OPTIONS,
    UserRating,
    clear_user_rating,
    get_user_rating,
    set_user_rating,
)


# Plain-text labels — no emojis, per the project's UI guidelines.
SOURCE_BADGE_LABEL = {
    "llm": "AI-verified",
    "semantic": "Semantic estimate",
}
SOURCE_BADGE_HELP = {
    "llm": "Backed by an LLM scoring pass over the full job description.",
    "semantic": "Inferred from semantic retrieval; the LLM did not verify this factor.",
}


def factor_source_label(factor: FactorExplanation) -> str:
    """Short label suitable for display next to a factor name."""
    return SOURCE_BADGE_LABEL.get(factor.source, "")


def factor_with_badge(factor: FactorExplanation) -> str:
    """Inline string: '<name> · <badge>' — drop-in for st.write/markdown lines."""
    label = factor_source_label(factor)
    return f"{factor.name} · _{label}_" if label else factor.name


def render_rating_panel(
    profile_slug: str,
    job_id: str,
    *,
    role_family: Optional[str] = None,
    key_prefix: str = "rate",
    show_helper: bool = True,
) -> Optional[UserRating]:
    """Render the rate-this-job button row. Returns the current rating after any change.

    Streamlit's button-as-radio pattern: each option is a button; clicking the
    one already selected clears the rating. We use type="primary" to highlight
    the current selection.
    """
    current = get_user_rating(profile_slug, job_id)
    current_label = current.label if current else None

    if show_helper:
        st.caption(
            "Your rating becomes ground-truth for the eval suite. Pick the option "
            "that best describes this match."
        )

    cols = st.columns(len(RATING_OPTIONS), gap="small")
    new_rating = current
    for column, (label, display, helper) in zip(cols, RATING_OPTIONS):
        is_current = current_label == label
        button_kwargs = {
            "key": f"{key_prefix}_{job_id}_{label}",
            "help": helper,
            "use_container_width": True,
        }
        if is_current:
            button_kwargs["type"] = "primary"
        if column.button(display, **button_kwargs):
            if is_current:
                clear_user_rating(profile_slug, job_id)
                new_rating = None
            else:
                new_rating = set_user_rating(
                    profile_slug,
                    job_id,
                    label,
                    notes="rated via dashboard",
                    role_family=role_family,
                )
            st.rerun()

    if new_rating is not None:
        st.caption(
            f"Current rating: **{_display_for(new_rating.label)}**. "
            f"Click the same button again to clear it."
        )
    return new_rating


def _display_for(label: str) -> str:
    for raw, display, _ in RATING_OPTIONS:
        if raw == label:
            return display
    return label
