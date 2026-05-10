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
    notes_default: str = "",
    compact: bool = False,
) -> Optional[UserRating]:
    """Render the rate-this-job control. Returns the current rating after any change.

    Two visual modes:
      • Default — full-width button row with helper text and a Beacon-style
        "Tell Beacon why (optional)" textarea that appears once a rating is
        set. The textarea persists through to `set_user_rating(notes=...)`
        on each rerun (no Save click required).
      • compact=True — dot-only segmented chips suitable for inline use in
        Jobs-table rows. Only the selected option shows a label. Notes are
        not surfaced in compact mode.

    Args:
        notes_default: Used as the textarea seed when no notes have been
            saved yet for this profile/job pair.
        compact: When True, render the dot-only chip variant matching
            Beacon's `.rating.compact` styling.
    """
    current = get_user_rating(profile_slug, job_id)
    current_label = current.label if current else None
    current_notes = current.notes if current else ""

    if compact:
        return _render_rating_compact(
            profile_slug,
            job_id,
            current=current,
            role_family=role_family,
            key_prefix=key_prefix,
        )

    if show_helper:
        st.caption(
            "Your rating becomes ground-truth for the eval suite. Pick the option "
            "that best describes this match."
        )

    cols = st.columns(len(RATING_OPTIONS), gap="small")
    new_rating: Optional[UserRating] = current
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
                    # Preserve existing notes when toggling between rating
                    # values; only clear the notes when the rating itself
                    # is cleared (handled in the if-branch above).
                    notes=current_notes or notes_default,
                    role_family=role_family,
                )
            st.rerun()

    # Once a rating exists we expose Beacon's "Tell Beacon why" textarea.
    # Persisted on every rerun (i.e. on blur) — no Save click required.
    if new_rating is not None:
        st.markdown(
            "<div class='rating-notes-label'>Tell Beacon why (optional)</div>",
            unsafe_allow_html=True,
        )
        notes_key = f"{key_prefix}_{job_id}_notes"
        seed = new_rating.notes or notes_default or ""
        if notes_key not in st.session_state:
            st.session_state[notes_key] = seed
        notes_value = st.text_area(
            "Tell Beacon why (optional)",
            key=notes_key,
            placeholder=(
                "Stack mismatch? Wrong seniority? Geo? — be specific so I avoid this pattern."
                if new_rating.label in {"bad_match", "should_skip"}
                else "Anything specific that hooked you — I'll weight it heavier next run."
            ),
            label_visibility="collapsed",
            height=72,
        )
        normalized_notes = (notes_value or "").strip()
        if normalized_notes != (new_rating.notes or "").strip():
            new_rating = set_user_rating(
                profile_slug,
                job_id,
                new_rating.label,
                notes=normalized_notes,
                role_family=role_family,
            )
        st.markdown(
            (
                "<div class='rating-notes-confirm'>"
                f"Saved as <b style='margin-left:4px'>{_display_for(new_rating.label)}</b>"
                " · applies on next run"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    return new_rating


def _render_rating_compact(
    profile_slug: str,
    job_id: str,
    *,
    current: Optional[UserRating],
    role_family: Optional[str],
    key_prefix: str,
) -> Optional[UserRating]:
    """Inline dot-only segmented control for dense surfaces (jobs table rows).

    Streamlit doesn't let us emit truly inline interactive HTML, so we render
    the dots as a column of small buttons that share the `.rating.compact`
    class via wrapper markup. Visually it's tight; functionally each dot
    toggles a rating.
    """
    current_label = current.label if current else None
    new_rating: Optional[UserRating] = current

    cols = st.columns(len(RATING_OPTIONS), gap="small")
    for column, (label, display, helper) in zip(cols, RATING_OPTIONS):
        is_current = current_label == label
        # The compact button is a single character indicator. The visible
        # label only appears on the selected option (Beacon convention).
        marker = "●" if is_current else "○"
        button_label = f"{marker} {display}" if is_current else marker
        if column.button(
            button_label,
            key=f"{key_prefix}_compact_{job_id}_{label}",
            help=f"{display} — {helper}",
            type="primary" if is_current else "secondary",
            use_container_width=True,
        ):
            if is_current:
                clear_user_rating(profile_slug, job_id)
                new_rating = None
            else:
                new_rating = set_user_rating(
                    profile_slug,
                    job_id,
                    label,
                    notes=(current.notes if current else ""),
                    role_family=role_family,
                )
            st.rerun()
    return new_rating


def _display_for(label: str) -> str:
    for raw, display, _ in RATING_OPTIONS:
        if raw == label:
            return display
    return label
