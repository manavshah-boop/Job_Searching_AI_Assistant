"""
dashboard_semantic.py - Semantic match UI panel for the dashboard.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from db import get_job_with_score
from match_explainer import build_match_explanation
from profile_intent import normalize_profile_intent
from reranker import build_profile_match_query, reranking_enabled, semantic_match_jobs
from ui_shell import panel
from vector_store import query_similar_jobs, vector_store_enabled, vector_top_k_chunks, vector_top_k_jobs


@st.cache_data(ttl=20, show_spinner=False)
def _cached_vector_search(
    slug: str,
    query: str,
    top_k_chunks: int,
    top_k_jobs: int,
) -> list:
    return query_similar_jobs(
        slug,
        query,
        top_k_chunks=top_k_chunks,
        top_k_jobs=top_k_jobs,
    )


@st.cache_data(ttl=20, show_spinner=False)
def _cached_semantic_match(
    slug: str,
    config: dict[str, Any],
    query: str | None,
) -> list:
    return semantic_match_jobs(slug, config, user_query=query)


def clear_semantic_panel_caches() -> None:
    _cached_vector_search.clear()
    _cached_semantic_match.clear()


def _render_explanation_card(explanation) -> None:
    st.caption(f"Recommended action: {explanation.recommended_action}")
    st.write(explanation.summary)
    score_parts: list[str] = []
    if explanation.overall_score is not None:
        score_parts.append(f"Final score {round(explanation.overall_score)}%")
    if explanation.llm_fit_score is not None:
        score_parts.append(f"LLM fit {explanation.llm_fit_score}")
    if explanation.ats_score is not None:
        score_parts.append(f"ATS {explanation.ats_score}")
    if score_parts:
        st.caption(" | ".join(score_parts))

    if explanation.strengths:
        st.caption(
            "**Top strengths:** "
            + " · ".join(factor.evidence[0] if factor.evidence else factor.name for factor in explanation.strengths[:3])
        )
    if explanation.concerns:
        st.caption(
            "**Top concerns:** "
            + " · ".join(factor.evidence[0] if factor.evidence else factor.name for factor in explanation.concerns[:3])
        )
    elif explanation.unknowns:
        st.caption("**Needs clarification:** " + " · ".join(factor.name for factor in explanation.unknowns[:2]))

    with st.expander("Factor-wise explanation", expanded=False):
        if explanation.matched_sections:
            st.caption("Matched sections: " + ", ".join(explanation.matched_sections))
        if explanation.evidence_snippets:
            st.caption("Evidence snippets")
            for snippet in explanation.evidence_snippets[:2]:
                st.caption(f"_{snippet}_")
        if explanation.strengths:
            st.write("Strengths")
            for factor in explanation.strengths[:4]:
                details = f" Evidence: {' | '.join(factor.evidence[:2])}." if factor.evidence else ""
                st.write(f"- {factor.name}: {factor.explanation}{details}")
        if explanation.concerns:
            st.write("Concerns")
            for factor in explanation.concerns[:4]:
                details = f" Evidence: {' | '.join(factor.evidence[:2])}." if factor.evidence else ""
                st.write(f"- {factor.name}: {factor.explanation}{details}")
        if explanation.unknowns:
            st.write("Unknowns")
            for factor in explanation.unknowns[:3]:
                details = f" Evidence: {' | '.join(factor.evidence[:2])}." if factor.evidence else ""
                st.write(f"- {factor.name}: {factor.explanation}{details}")


def render_semantic_match_panel(slug: str, config: dict[str, Any]) -> None:
    if not vector_store_enabled(config):
        st.caption("Semantic retrieval is disabled for this profile.")
        return

    query_key = f"semantic_query_{slug}"
    run_key = f"semantic_run_{slug}"
    results_key = f"semantic_results_{slug}"
    mode_key = f"semantic_rerank_{slug}"
    meta_key = f"semantic_meta_{slug}"

    search_cols = st.columns([1.8, 0.7], gap="medium")
    search_cols[0].text_input(
        "Semantic query",
        key=query_key,
        placeholder="Optional: backend AI platform role with Python and AWS",
        help="Leave blank to run a profile-aware match, or add a query to steer the search.",
    )
    search_cols[0].checkbox(
        "Use cross-encoder reranking",
        key=mode_key,
        value=reranking_enabled(config),
        help="Add a profile-aware precision pass after vector retrieval.",
    )
    run_search = search_cols[1].button("Run semantic match", key=run_key, use_container_width=True)

    if run_search:
        query = st.session_state.get(query_key, "").strip()
        use_reranker = bool(st.session_state.get(mode_key, reranking_enabled(config)))
        if use_reranker:
            st.session_state[results_key] = _cached_semantic_match(slug, config, query or None)
            st.session_state[meta_key] = {
                "mode": "reranked" if reranking_enabled(config) else "vector-fallback",
                "query_label": query or build_profile_match_query(config),
            }
        elif query:
            st.session_state[results_key] = _cached_vector_search(
                slug,
                query,
                vector_top_k_chunks(config),
                vector_top_k_jobs(config),
            )
            st.session_state[meta_key] = {"mode": "vector", "query_label": query}
        else:
            match_query = build_profile_match_query(config)
            st.session_state[results_key] = _cached_vector_search(
                slug,
                match_query,
                vector_top_k_chunks(config),
                vector_top_k_jobs(config),
            )
            st.session_state[meta_key] = {"mode": "vector", "query_label": match_query}

    results = st.session_state.get(results_key, [])
    if not results:
        st.caption("Run a semantic match to compare your profile and optional query against embedded job sections.")
        return

    meta = st.session_state.get(meta_key, {"mode": "vector"})
    if meta.get("mode") == "vector-fallback":
        st.caption("Cross-encoder reranking is disabled in config, so this panel is showing vector-ranked fallback results.")
    elif meta.get("mode") == "vector":
        st.caption("Showing vector-ranked semantic retrieval results.")
    else:
        st.caption("Showing profile-aware reranked semantic matches.")

    profile_intent = normalize_profile_intent(config)
    for index, result in enumerate(results[:5], start=1):
        if hasattr(result, "final_score"):
            subtitle = f"{result.company} · {result.source} · Final score {round(result.final_score * 100)}%"
        else:
            subtitle = f"{result.company} · {result.source} · Similarity {round(result.aggregate_score * 100)}%"

        with panel(f"{index}. {result.title}", subtitle=subtitle):
            if hasattr(result, "final_score"):
                score_record = get_job_with_score(result.job_id, profile=slug)
                explanation = build_match_explanation(score_record, score_record, result, profile_intent)
                _render_explanation_card(explanation)
            else:
                matched = ", ".join(result.matched_chunks) or "none"
                st.caption(f"Matched sections: {matched}")
                st.write(result.retrieval_reason)
            if result.url:
                st.link_button("Open posting", result.url, key=f"semantic_result_{slug}_{result.job_id}")
