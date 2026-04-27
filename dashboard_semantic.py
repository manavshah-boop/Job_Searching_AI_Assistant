"""
dashboard_semantic.py - Semantic match UI panel for the dashboard.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

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

    for index, result in enumerate(results[:5], start=1):
        if hasattr(result, "final_score"):
            score_pct = round(result.final_score * 100)
            matched = ", ".join(result.matched_sections) or "none"
            subtitle = f"{result.company} · {result.source} · Final score {score_pct}%"
            reason = result.match_reason
            evidence_snippets = getattr(result, "evidence_snippets", [])
            ev = getattr(result, "evidence", None)
        else:
            score_pct = round(result.aggregate_score * 100)
            matched = ", ".join(result.matched_chunks) or "none"
            subtitle = f"{result.company} · {result.source} · Similarity {score_pct}%"
            reason = result.retrieval_reason
            evidence_snippets = []
            ev = None

        with panel(f"{index}. {result.title}", subtitle=subtitle):
            st.caption(f"Matched sections: {matched}")
            st.write(reason)
            if ev is not None:
                if ev.positive:
                    st.caption("**Positive signals:** " + " · ".join(ev.positive))
                if ev.concerns:
                    st.caption("**Concerns:** " + " · ".join(ev.concerns))
            for snippet in evidence_snippets[:2]:
                if snippet:
                    st.caption(f"_{snippet}_")
            if result.url:
                st.link_button("Open posting", result.url, key=f"semantic_result_{slug}_{result.job_id}")
