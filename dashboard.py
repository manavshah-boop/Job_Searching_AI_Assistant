"""
dashboard.py - Main Streamlit dashboard entrypoint.

Manual verification:
- Run from the repo root so Streamlit can pick up `.streamlit/config.toml`:
  `cd c:\\Users\\Manav Shah\\Documents\\Job_Searching_AI_Assistant`
  `streamlit run dashboard.py`
- Toggle `DEBUG_THEME` below to `True` when you want a temporary theme/bootstrap
  panel at the top of the app.
- Visual checks: confirm the hero section shows a dark teal gradient, the sidebar
  exposes profile navigation and the primary run action, jobs can be multi-selected
  for bulk status updates, pipeline runs show staged progress and an activity feed,
  and keyboard focus states show a visible teal outline.
"""

from __future__ import annotations

import base64
import copy
import html
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yaml
from dotenv import load_dotenv
from loguru import logger

from config import _resolve_resume_path, apply_config_defaults, load_config
from db import (
    clear_profile_jobs,
    count_jobs,
    finish_run,
    get_db_path,
    get_all_jobs_with_scores,
    get_job_with_score,
    get_recent_runs,
    get_routing_stats,
    init_db,
    load_discovered_slugs,
    rescore_reset,
    search_jobs_by_raw_text,
    set_active_profile,
    start_run,
    update_job_status,
)
from dashboard_semantic import clear_semantic_panel_caches, render_semantic_match_panel
from evaluation import evaluate_profile, export_eval_template, load_eval_labels, load_last_eval_result
from dashboard_ui import (
    render_activity_feed,
    render_pipeline_stages,
    render_progress_header,
    render_source_progress,
)
from logging_config import configure_logging
from dashboard_ratings import factor_with_badge, render_rating_panel
from match_explainer import build_match_explanation
from user_ratings import (
    RATING_OPTIONS,
    attach_role_family_from_config,
    get_all_user_ratings,
    rating_counts,
)
from onboarding import render_onboarding, sanitize_slug
from profile_intent import normalize_profile_intent
from progress_tracker import ProgressTracker, Stage, StageStatus
from scorer import score_all_jobs
from tracking import get_experiment_name, get_tracking_uri, mlflow_enabled
from ui_shell import (
    badge,
    beacon_aside_foot,
    beacon_brand,
    beacon_nav_group,
    beacon_nav_kicker,
    beacon_profile_card,
    beacon_run_card,
    callout,
    chip_row,
    empty_state,
    help_tip,
    page_header,
    panel,
    section_shell,
    sidebar_profile_summary,
    stat_row,
    toolbar,
)
from ui_theme import PAGE_TITLE, apply_page_scaffold

load_dotenv()

BASE_DIR = Path(__file__).parent
PROFILES_DIR = BASE_DIR / "profiles"
DEBUG_THEME = False
THEME_CONFIG_RELATIVE_PATH = Path(".streamlit") / "config.toml"

PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
}

SOURCE_LABELS = {
    "greenhouse": "Greenhouse",
    "lever": "Lever",
    "hackernews": "HN Who's Hiring",
    "ashby": "Ashby",
    "workable": "Workable",
    "himalayas": "Himalayas",
}

SECTION_ORDER = ["Overview", "Jobs", "Activity", "Profile", "Settings"]
SECTION_WIDGET_KEY = "_dashboard_section_widget"
SECTION_COPY = {
    "Overview": "Track profile health, top matches, and recent run outcomes in one place.",
    "Jobs": "Scan the pipeline quickly, refine filters, and update job status without losing context.",
    "Activity": "Review pipeline history, cached discovery state, and the latest operational signals.",
    "Profile": "Inspect candidate context, preferences, and source coverage as the scorer sees them.",
    "Settings": "Adjust search behavior, scoring thresholds, sources, and maintenance actions safely.",
}
JOB_VISIBLE_OPTIONAL_COLUMNS = ["Location", "Source", "ATS", "Summary", "Posting", "Filter reason"]
LOCATION_PICKER_OPTIONS = [
    "Remote",
    "San Francisco, CA",
    "New York, NY",
    "Seattle, WA",
    "Austin, TX",
    "Boston, MA",
    "Los Angeles, CA",
    "Chicago, IL",
    "Denver, CO",
    "Washington, DC",
]


def _init_state() -> None:
    defaults = {
        "active_profile": None,
        "show_onboarding": False,
        "onboarding_step": 1,
        "onboarding_data": {},
        "dashboard_notice": None,
        "celebrate_profile_create": False,
        "dashboard_section": "Overview",
        SECTION_WIDGET_KEY: "Overview",
        "last_status_change": None,
        "last_pipeline_result": None,
        "last_pipeline_error": None,
        "show_create_profile_dialog": False,
        "show_resume_preview_dialog": False,
        "resume_preview_name": None,
        "resume_preview_b64": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _valid_dashboard_section(value: Any) -> str:
    return value if value in SECTION_ORDER else "Overview"


def _sync_dashboard_section_from_widget() -> None:
    st.session_state.dashboard_section = _valid_dashboard_section(
        st.session_state.get(SECTION_WIDGET_KEY, "Overview")
    )


def _prepare_dashboard_section_widget() -> None:
    section = _valid_dashboard_section(st.session_state.get("dashboard_section", "Overview"))
    st.session_state.dashboard_section = section
    if st.session_state.get(SECTION_WIDGET_KEY) != section:
        st.session_state[SECTION_WIDGET_KEY] = section


_WORKER_LOCKFILE  = ".worker_running"
_WORKER_PROGRESS  = ".run_progress.json"
_WORKER_STATUS    = ".last_run"
_WORKER_STALE_SEC = 3 * 3600  # match worker/run_pipeline.py
# If the worker hasn't finished a run in this many hours, surface a banner.
# The cron fires daily at 04:00 UTC; 30h gives one missed window of slack.
_LAST_RUN_STALE_HOURS = 30


def _worker_lockfile(slug: str) -> Path:
    return PROFILES_DIR / slug / _WORKER_LOCKFILE


def _worker_progress_path(slug: str) -> Path:
    return PROFILES_DIR / slug / _WORKER_PROGRESS


def _worker_is_running(slug: str) -> bool:
    lf = _worker_lockfile(slug)
    if not lf.exists():
        return False
    return (time.time() - lf.stat().st_mtime) < _WORKER_STALE_SEC


def _launch_worker(slug: str) -> None:
    worker_script = BASE_DIR / "worker" / "run_pipeline.py"
    subprocess.Popen(
        [sys.executable, str(worker_script), "--profile", slug],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _read_progress_json(slug: str) -> dict[str, Any] | None:
    path = _worker_progress_path(slug)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _last_run_path(slug: str) -> Path:
    return PROFILES_DIR / slug / _WORKER_STATUS


def _read_last_run(slug: str) -> dict[str, Any] | None:
    path = _last_run_path(slug)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _hours_since_last_run(
    last_run: dict[str, Any] | None,
    *,
    now: datetime | None = None,
) -> float | None:
    """Hours elapsed since `finished_at` in the .last_run payload.

    Returns None when the payload is missing, has no `finished_at`, or the
    timestamp is unparsable — callers treat that as "no signal", not stale.
    """
    if not last_run:
        return None
    finished = last_run.get("finished_at")
    if not finished:
        return None
    try:
        finished_dt = datetime.fromisoformat(str(finished))
    except (TypeError, ValueError):
        return None
    if finished_dt.tzinfo is None:
        finished_dt = finished_dt.replace(tzinfo=timezone.utc)
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return (now - finished_dt).total_seconds() / 3600


def _last_run_is_stale(
    last_run: dict[str, Any] | None,
    *,
    now: datetime | None = None,
    threshold_hours: float = _LAST_RUN_STALE_HOURS,
) -> bool:
    hours = _hours_since_last_run(last_run, now=now)
    if hours is None:
        return False
    return hours >= threshold_hours


def _profile_config_path(slug: str) -> Path:
    return PROFILES_DIR / slug / "config.yaml"


def _dashboard_log_path(profile: str) -> Path:
    return BASE_DIR / "logs" / profile / "agent.log"


def _eval_labels_path(slug: str) -> Path:
    return PROFILES_DIR / slug / "eval_labels.yaml"


def _safe_count_jobs(slug: str) -> dict[str, int]:
    try:
        init_db(profile=slug)
        return count_jobs(profile=slug)
    except Exception:
        return {"total": 0, "scored": 0}


@st.cache_data(ttl=10, show_spinner=False)
def _cached_list_profiles() -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    if not PROFILES_DIR.exists():
        return profiles

    for path in PROFILES_DIR.iterdir():
        if not path.is_dir():
            continue
        config_path = path / "config.yaml"
        if not config_path.exists():
            continue

        raw_config: dict[str, Any] = {}
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                raw_config = yaml.safe_load(file) or {}
        except Exception:
            raw_config = {}

        profile_cfg = raw_config.get("profile", {})
        llm_cfg = raw_config.get("llm", {})
        counts = _safe_count_jobs(path.name)

        profiles.append(
            {
                "slug": path.name,
                "name": profile_cfg.get("name", path.name.replace("_", " ").title()),
                "job_type": profile_cfg.get("job_type", "fulltime"),
                "provider": llm_cfg.get("provider", "unknown"),
                "counts": counts,
                "updated_at": config_path.stat().st_mtime,
            }
        )

    return sorted(profiles, key=lambda item: item["updated_at"], reverse=True)


def list_profiles() -> list[dict[str, Any]]:
    return _cached_list_profiles()


@st.cache_data(ttl=10, show_spinner=False)
def _cached_fetch_job_summaries(slug: str) -> list[dict[str, Any]]:
    return _fetch_job_summaries(slug)


@st.cache_data(ttl=10, show_spinner=False)
def _cached_fetch_job_detail(slug: str, job_id: str) -> Optional[dict[str, Any]]:
    return _fetch_job_detail(slug, job_id)


@st.cache_data(ttl=10, show_spinner=False)
def _cached_recent_runs(slug: str) -> list[dict[str, Any]]:
    return get_recent_runs(limit=20, profile=slug)


def invalidate_dashboard_caches() -> None:
    _cached_list_profiles.clear()
    _cached_fetch_job_summaries.clear()
    _cached_fetch_job_detail.clear()
    _cached_recent_runs.clear()
    clear_semantic_panel_caches()


def build_jobs_table_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "id": record["id"],
                "Title": record["title"],
                "Company": record["company"],
                "Location": record["location"] or "Location not listed",
                "Source": record["source_label"],
                "Job status": record["status_label"],
                "Score state": record["score_state"],
                "Fit": record["fit_score"],
                "ATS": record["ats_score"],
                "Summary": record["one_liner"],
                "Posting": record["url"] or "",
                "Filter reason": record.get("scrape_filter_reason", ""),
            }
            for record in records
        ]
    )
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "id",
                "Title",
                "Company",
                "Location",
                "Source",
                "Job status",
                "Score state",
                "Fit",
                "ATS",
                "Summary",
                "Posting",
                "Filter reason",
            ]
        )

    for column in ("Fit", "ATS"):
        frame[column] = pd.array(frame[column], dtype="Int64")
    return frame


def summarize_run_errors(errors: list[str], preview_chars: int = 72) -> str:
    if not errors:
        return "None"

    preview = errors[0].strip()
    if len(preview) > preview_chars:
        preview = preview[: preview_chars - 1].rstrip() + "…"
    if len(errors) == 1:
        return preview
    return f"{preview} (+{len(errors) - 1} more)"


def _summarize_filter_selection(selected: list[str], all_options: list[str]) -> str:
    if not selected:
        return "None"
    if len(selected) == len(all_options):
        return "All"
    if len(selected) <= 2:
        return ", ".join(selected)
    return f"{selected[0]}, {selected[1]} +{len(selected) - 2}"


def build_jobs_filter_chips(
    selected_sources: list[str],
    source_options: list[str],
    selected_statuses: list[str],
    status_options: list[str],
    selected_score_states: list[str],
    score_state_options: list[str],
    min_fit: int,
    search: str,
    include_full_text: bool,
) -> list[str]:
    chips = [
        f"Sources: {_summarize_filter_selection(selected_sources, source_options)}",
        f"Status: {_summarize_filter_selection(selected_statuses, status_options)}",
        f"Scoring: {_summarize_filter_selection(selected_score_states, score_state_options)}",
        f"Min fit: {min_fit}+" if min_fit > 0 else "Min fit: Any",
    ]
    if search.strip():
        chips.append(f"Search: {search.strip()}")
    if include_full_text:
        chips.append("Full text search: On")
    return chips


def _render_semantic_search_results(slug: str, config: dict[str, Any]) -> None:
    if not vector_store_enabled(config):
        st.caption("Semantic retrieval is disabled for this profile.")
        return

    query_key = f"semantic_query_{slug}"
    run_key = f"semantic_run_{slug}"
    results_key = f"semantic_results_{slug}"

    search_cols = st.columns([1.8, 0.6], gap="medium")
    search_cols[0].text_input(
        "Semantic search",
        key=query_key,
        placeholder="backend AI platform role with Python and AWS",
        help="Search embedded job chunks semantically instead of keyword matching only.",
    )
    run_search = search_cols[1].button("Search", key=run_key, use_container_width=True)

    if run_search:
        query = st.session_state.get(query_key, "").strip()
        if query:
            st.session_state[results_key] = _cached_vector_search(
                slug,
                query,
                vector_top_k_chunks(config),
                vector_top_k_jobs(config),
            )
        else:
            st.session_state[results_key] = []

    results = st.session_state.get(results_key, [])
    if not results:
        st.caption("Run a semantic query to see top retrieved jobs and the sections that matched.")
        return

    for index, result in enumerate(results[:5], start=1):
        similarity_pct = round(result.aggregate_score * 100)
        matched = ", ".join(result.matched_chunks) or "none"
        subtitle = f"{result.company} · {result.source} · Similarity {similarity_pct}%"
        with panel(f"{index}. {result.title}", subtitle=subtitle):
            st.caption(f"Matched sections: {matched}")
            st.write(result.retrieval_reason)
            if result.url:
                st.link_button("Open posting", result.url, key=f"semantic_result_{slug}_{result.job_id}")


def _render_semantic_search_results(slug: str, config: dict[str, Any]) -> None:
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

    rating_role_family = attach_role_family_from_config(slug, config)
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
            st.divider()
            st.caption("**Rate this match** — feeds the eval suite.")
            render_rating_panel(
                slug,
                str(result.job_id),
                role_family=rating_role_family,
                key_prefix=f"rate_main_semantic_{slug}",
                show_helper=False,
            )


def _job_filter_state_keys(slug: str) -> dict[str, str]:
    return {
        "search": f"jobs_search_{slug}",
        "min_fit": f"jobs_min_fit_{slug}",
        "include_full_text": f"jobs_include_full_text_{slug}",
        "sources": f"jobs_sources_{slug}",
        "statuses": f"jobs_statuses_{slug}",
        "score_states": f"jobs_score_states_{slug}",
        "visible_columns": f"jobs_visible_columns_{slug}",
        "show_scrape_rejected": f"jobs_show_scrape_rejected_{slug}",
    }


def reset_job_filter_state(
    slug: str,
    source_options: list[str],
    status_options: list[str],
    score_state_options: list[str],
) -> None:
    keys = _job_filter_state_keys(slug)
    st.session_state[keys["search"]] = ""
    st.session_state[keys["min_fit"]] = 0
    st.session_state[keys["include_full_text"]] = False
    st.session_state[keys["sources"]] = list(source_options)
    st.session_state[keys["statuses"]] = list(status_options)
    st.session_state[keys["score_states"]] = list(score_state_options)
    st.session_state[keys["visible_columns"]] = [c for c in JOB_VISIBLE_OPTIONAL_COLUMNS if c != "Filter reason"]
    st.session_state[keys["show_scrape_rejected"]] = False


def resolve_job_table_columns(selected_optional_columns: list[str]) -> list[str]:
    base_columns = ["Title", "Company", "Job status", "Score state", "Fit"]
    ordered_optional = [
        column for column in JOB_VISIBLE_OPTIONAL_COLUMNS if column in selected_optional_columns
    ]
    return base_columns + ordered_optional


def effective_config_summary(config: dict[str, Any], raw_config: dict[str, Any]) -> list[str]:
    prefs = config.get("preferences", {})
    location = prefs.get("location", {})
    compensation = prefs.get("compensation", {})
    is_intern = config.get("profile", {}).get("job_type") == "internship"
    enabled_sources = [
        label
        for source, label in SOURCE_LABELS.items()
        if _enabled_sources(raw_config).get(source, False)
    ]
    summary = [
        f"Provider: {config.get('llm', {}).get('provider', 'unknown').title()}",
        f"Minimum display score: {config.get('scoring', {}).get('min_display_score', 60)}",
        "Remote OK" if location.get("remote_ok", True) else "Remote not preferred",
        f"Preferred locations: {', '.join(location.get('preferred_locations', [])) or 'None'}",
        f"Target titles: {len(prefs.get('titles', []))}",
        f"Desired skills: {len(prefs.get('desired_skills', []))}",
        f"Sources enabled: {', '.join(enabled_sources) if enabled_sources else 'None'}",
    ]
    if is_intern:
        pay_preference = str(compensation.get("intern_pay_preference", "")).strip().lower()
        if pay_preference not in {"paid_only", "unpaid_ok", "no_preference"}:
            pay_preference = "paid_only" if compensation.get("monthly_stipend") else "no_preference"
        pay_preference_label = {
            "paid_only": "Paid only",
            "unpaid_ok": "Unpaid OK",
            "no_preference": "No preference",
        }[pay_preference]
        summary.append(f"Compensation preference: {pay_preference_label}")
        if compensation.get("monthly_stipend") not in (None, "", 0):
            summary.append(f"Monthly stipend target: ${int(compensation['monthly_stipend']):,}")
    elif "min_salary" in compensation:
        summary.append(f"Minimum salary: ${int(compensation['min_salary']):,}")
    return summary


def _normalize_intern_pay_preference(compensation: dict[str, Any]) -> str:
    preference = str(compensation.get("intern_pay_preference", "")).strip().lower()
    if preference in {"paid_only", "unpaid_ok", "no_preference"}:
        return preference
    if compensation.get("monthly_stipend"):
        return "paid_only"
    return "no_preference"


def _optional_int_input_value(value: Any) -> str:
    if value in (None, "", 0):
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value).strip()


def _parse_optional_int_input(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, int):
        return value
    cleaned = str(value).strip().replace("$", "").replace(",", "")
    if not cleaned:
        return None
    return int(cleaned)


def _format_compensation_value(job_type: str, compensation: dict[str, Any]) -> str:
    if job_type == "internship":
        preference = _normalize_intern_pay_preference(compensation)
        preference_label = {
            "paid_only": "Paid only",
            "unpaid_ok": "Unpaid OK",
            "no_preference": "No preference",
        }[preference]
        stipend = compensation.get("monthly_stipend")
        stipend_value = int(stipend) if stipend not in (None, "", 0) else 0
        if preference == "paid_only" and stipend_value:
            return f"{preference_label}, target ${stipend_value:,}/mo"
        return preference_label

    salary = compensation.get("min_salary")
    if salary in (None, "", 0):
        return "Not set"
    return f"${int(salary):,}"


def _compensation_rows(job_type: str, compensation: dict[str, Any]) -> list[tuple[str, str]]:
    if job_type == "internship":
        preference = _normalize_intern_pay_preference(compensation)
        rows = [("Compensation preference", {
            "paid_only": "Paid only",
            "unpaid_ok": "Unpaid OK",
            "no_preference": "No preference",
        }[preference])]
        stipend = compensation.get("monthly_stipend")
        if stipend not in (None, "", 0):
            rows.append(("Monthly stipend target", f"${int(stipend):,}/mo"))
        return rows
    return [("Compensation", _format_compensation_value(job_type, compensation))]


def _section_heading(profile_name: str, section: str) -> None:
    st.markdown(
        (
            f"<div class='shell-breadcrumb'>{html.escape(profile_name)} workspace</div>"
            f"<h2 class='shell-section-title'>{html.escape(section)}</h2>"
            f"<p class='shell-section-subtitle'>{html.escape(SECTION_COPY[section])}</p>"
        ),
        unsafe_allow_html=True,
    )


def _store_status_change(slug: str, changes: list[dict[str, str]]) -> None:
    st.session_state.last_status_change = {"profile": slug, "changes": changes}


def _apply_status_changes(slug: str, records: list[dict[str, Any]], job_ids: list[str], new_status: str) -> int:
    record_map = {record["id"]: record for record in records}
    changes: list[dict[str, str]] = []
    for job_id in job_ids:
        record = record_map.get(job_id)
        if record is None:
            continue
        previous_status = str(record.get("status", "new"))
        if previous_status == new_status:
            continue
        update_job_status(job_id, new_status, profile=slug)
        changes.append(
            {"job_id": job_id, "previous_status": previous_status, "new_status": new_status}
        )

    if changes:
        _store_status_change(slug, changes)
        invalidate_dashboard_caches()
    return len(changes)


def _undo_last_status_change(slug: str) -> int:
    payload = st.session_state.get("last_status_change")
    if not payload or payload.get("profile") != slug:
        return 0

    changes = payload.get("changes", [])
    for change in changes:
        update_job_status(change["job_id"], change["previous_status"], profile=slug)

    st.session_state.last_status_change = None
    invalidate_dashboard_caches()
    return len(changes)


def _theme_config_path_from_cwd() -> Path:
    return Path.cwd() / THEME_CONFIG_RELATIVE_PATH


def get_theme_bootstrap_state() -> dict[str, Any]:
    theme_path = _theme_config_path_from_cwd()
    return {
        "entrypoint": "dashboard.py",
        "cwd": os.getcwd(),
        "theme_config_path": theme_path,
        "theme_config_exists": theme_path.exists(),
        "streamlit_version": st.__version__,
    }


def _render_theme_bootstrap_notice() -> None:
    state = get_theme_bootstrap_state()

    if DEBUG_THEME:
        with panel("Theme debug", subtitle="Temporary bootstrap diagnostics"):
            st.caption("Running: dashboard.py")
            st.caption(f"Working directory: {state['cwd']}")
            st.caption(
                "CWD theme config: "
                + ("found" if state["theme_config_exists"] else "missing")
                + f" ({state['theme_config_path']})"
            )
            st.caption(f"Streamlit version: {state['streamlit_version']}")

    if state["theme_config_exists"]:
        return

    callout(
        "warning",
        "Repo theme file not detected from the current working directory",
        (
            f"Streamlit did not find `{state['theme_config_path']}` relative to the current working directory. "
            "The dashboard can still run, but `.streamlit/config.toml` may not be applied. "
            f"Try `cd {BASE_DIR}` and then `streamlit run dashboard.py`."
        ),
    )


def _read_profile_config(slug: str) -> dict[str, Any]:
    with open(_profile_config_path(slug), "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _write_profile_config(slug: str, config: dict[str, Any]) -> None:
    path = _profile_config_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _enabled_sources(config: dict[str, Any]) -> dict[str, bool]:
    sources = config.get("sources", {})
    return {
        "greenhouse": sources.get("greenhouse", {}).get("enabled", True),
        "lever":      sources.get("lever",      {}).get("enabled", True),
        "hackernews": sources.get("hn",         {}).get("enabled", False),
        "ashby":      sources.get("ashby",      {}).get("enabled", False),
        "workable":   sources.get("workable",   {}).get("enabled", False),
        "himalayas":  sources.get("himalayas",  {}).get("enabled", False),
    }


def _provider_model(config: dict[str, Any]) -> str:
    llm_cfg = config.get("llm", {})
    provider = llm_cfg.get("provider", "unknown")
    models = llm_cfg.get("model", {})
    return str(models.get(provider, "unknown"))


def _check_api_key(config: dict[str, Any]) -> Optional[str]:
    provider = config.get("llm", {}).get("provider")
    env_var = PROVIDER_ENV_VARS.get(str(provider))
    if not env_var:
        return None
    profile = config.get("_active_profile")
    key = os.environ.get(f"{env_var}_{profile.upper()}") if profile else None
    if not key:
        key = os.environ.get(env_var)
    if not key:
        return f"{env_var}_{profile.upper()} or {env_var}" if profile else env_var
    return None


def _set_notice(slug: str, kind: str, message: str) -> None:
    st.session_state.dashboard_notice = {
        "profile": slug,
        "kind": kind,
        "message": message,
    }


def _render_notice(slug: str) -> None:
    notice = st.session_state.get("dashboard_notice")
    if not notice or notice.get("profile") != slug:
        return

    kind = notice.get("kind", "info")
    message = notice.get("message", "")
    callout(kind, kind.title(), message)
    st.session_state.dashboard_notice = None


def _render_staleness_banner(slug: str) -> None:
    last_run = _read_last_run(slug)
    hours = _hours_since_last_run(last_run)
    if hours is None or hours < _LAST_RUN_STALE_HOURS:
        return
    callout(
        "warning",
        "Cron may have stalled",
        (
            f"Last successful run was {int(hours)} hours ago. "
            "Check the worker journal — the daily timer may have failed or the host was offline."
        ),
    )


def _profile_initials(name: str) -> str:
    parts = [part for part in str(name or "").strip().split() if part]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][:1] + parts[1][:1]).upper()


def _profile_sub_line(profile_record: dict[str, Any]) -> str:
    job_type = str(profile_record.get("job_type") or "fulltime").replace("_", " ")
    counts = profile_record.get("counts") or {}
    pipeline = int(counts.get("total") or 0)
    return f"{job_type} · {pipeline} in pipeline"


def _format_last_run_meta(last_run: dict[str, Any]) -> str:
    finished = str(last_run.get("finished_at") or "")[:16]
    scored = last_run.get("jobs_scored")
    parts: list[str] = []
    if scored is not None:
        parts.append(f"Scored {scored}")
    if finished:
        parts.append(finished.replace("T", " "))
    return " · ".join(parts) or "Run details unavailable"


def _render_sidebar_run_card(
    slug: str,
    *,
    worker_running: bool,
) -> str | None:
    """Render the run-card pinned to the sidebar bottom.

    Returns "view_pipeline" when the user wants to jump to the Activity tab,
    or "run_search" when they want to start a fresh discovery from idle.
    """
    progress_data = _read_progress_json(slug) if worker_running else None
    last_run = _read_last_run(slug)

    if worker_running and progress_data:
        try:
            tracker = ProgressTracker.from_dict(progress_data)
            running_stage = next(
                (sp for sp in tracker.stages.values() if sp.status == StageStatus.RUNNING),
                None,
            )
            stage_label = (
                running_stage.stage.value if running_stage else "Working…"
            )
            pct = tracker.overall_progress_pct
        except Exception:
            stage_label = "Working…"
            pct = 0.0
        view_clicked = beacon_run_card(
            state="running",
            headline="Run · live",
            detail=stage_label,
            progress_pct=pct,
            action_label="View pipeline →",
            action_key=f"sidebar_view_pipeline_{slug}",
        )
        if view_clicked:
            return "view_pipeline"
        return None

    if worker_running:
        view_clicked = beacon_run_card(
            state="running",
            headline="Run · starting",
            detail="Worker is spinning up — progress will appear shortly.",
            action_label="View pipeline →",
            action_key=f"sidebar_view_pipeline_{slug}",
        )
        if view_clicked:
            return "view_pipeline"
        return None

    if last_run:
        status = str(last_run.get("status") or "").lower()
        if status in {"failed", "error", "crashed"}:
            state = "fail"
            headline = f"Last run · {status}"
        elif status in {"complete", "success", "ok", "completed"}:
            state = "ok"
            headline = "Last run · ok"
        else:
            state = "warn"
            headline = f"Last run · {status or 'partial'}"
        view_clicked = beacon_run_card(
            state=state,
            headline=headline,
            detail=_format_last_run_meta(last_run),
            action_label="View pipeline →",
            action_key=f"sidebar_view_pipeline_{slug}",
        )
        if view_clicked:
            return "view_pipeline"
        return None

    start_clicked = beacon_run_card(
        state="idle",
        headline="No runs yet",
        detail="Start discovery to populate your pipeline.",
        action_label="Start discovery",
        action_key=f"sidebar_run_search_{slug}",
    )
    if start_clicked:
        return "run_search"
    return None


def _render_sidebar_nav(
    slug: str,
    profile_name: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    *,
    worker_running: bool = False,
) -> str | None:
    """Render the Beacon-style op-aside.

    Returns one of {"run_search", "create_profile", "rerun_setup"} when the
    caller needs to act. Profile-switching, view-pipeline, and onboarding
    re-opens are handled internally via st.rerun.
    """
    available_profiles = list_profiles()
    profile_cfg = config.get("profile", {}) if isinstance(config, dict) else {}
    job_type = str(profile_cfg.get("job_type") or "fulltime").replace("_", " ")
    in_pipeline = int(metrics.get("db_total") or metrics.get("total") or 0)
    sub_line = f"{job_type} · {in_pipeline} in pipeline"
    initials = _profile_initials(profile_name)

    sidebar_action: str | None = None

    with st.sidebar:
        beacon_brand("Beacon", "job-search agent")
        beacon_profile_card(profile_name, sub_line, initials=initials)

        popover_callable = callable(getattr(st, "popover", None))
        if popover_callable and (len(available_profiles) > 1 or True):
            with st.popover("Switch profile  ▾", use_container_width=True):
                for profile in available_profiles:
                    is_current = profile["slug"] == slug
                    profile_initials = _profile_initials(profile["name"])
                    profile_sub = _profile_sub_line(profile)
                    label = f"{profile_initials} · {profile['name']}"
                    if is_current:
                        label = f"{label}  (current)"
                    if st.button(
                        label,
                        key=f"profile_pop_row_{profile['slug']}",
                        use_container_width=True,
                        disabled=is_current,
                        help=profile_sub,
                    ):
                        st.session_state.active_profile = profile["slug"]
                        set_active_profile(profile["slug"])
                        st.rerun()
                _render_html_block("<hr style='margin:6px 0;border:0;border-top:1px solid var(--line)'/>")
                if st.button("+  Add profile", key=f"profile_pop_add_{slug}", use_container_width=True):
                    sidebar_action = "create_profile"

        # Workspace nav group
        beacon_nav_kicker("Workspace")
        active_section = _valid_dashboard_section(
            st.session_state.get("dashboard_section", "Overview")
        )
        ws_clicked = beacon_nav_group(
            [
                {"id": "Overview", "label": "Overview", "icon": "◉"},
                {"id": "Jobs", "label": "Jobs", "icon": "▦", "count": in_pipeline},
                {"id": "Activity", "label": "Activity", "icon": "≡"},
            ],
            active=active_section,
            key_prefix=f"nav_ws_{slug}",
        )
        if ws_clicked and ws_clicked != active_section:
            st.session_state.dashboard_section = ws_clicked
            st.rerun()

        # You nav group
        beacon_nav_kicker("You")
        you_clicked = beacon_nav_group(
            [
                {"id": "Profile", "label": "Profile", "icon": "○"},
                {"id": "Settings", "label": "Settings", "icon": "✦"},
            ],
            active=active_section,
            key_prefix=f"nav_you_{slug}",
        )
        if you_clicked and you_clicked != active_section:
            st.session_state.dashboard_section = you_clicked
            st.rerun()

        # Run card pinned to the bottom of the aside
        run_card_action = _render_sidebar_run_card(slug, worker_running=worker_running)
        if run_card_action == "view_pipeline":
            st.session_state.dashboard_section = "Activity"
            st.rerun()
        elif run_card_action == "run_search":
            sidebar_action = sidebar_action or "run_search"

        # Footer: re-run setup (re-opens onboarding for the active profile)
        if beacon_aside_foot("Re-run setup", key=f"sidebar_rerun_setup_{slug}"):
            sidebar_action = sidebar_action or "rerun_setup"

    return sidebar_action


def _source_label(source: str) -> str:
    return SOURCE_LABELS.get(source, source.replace("_", " ").title())


def _score_state(record: dict[str, Any]) -> str:
    if record["fit_score"] is not None:
        return "Scored"
    if (record["score_attempts"] or 0) >= 3:
        return "Failed"
    if record["score_error"]:
        return "Needs retry"
    return "Pending"


def _deserialize_job_record(record: dict[str, Any]) -> dict[str, Any]:
    # record is already normalized by db.get_all_jobs_with_scores; add UI-only fields
    record["score_state"] = _score_state(record)
    record["source_label"] = _source_label(record["source"])
    record["status_label"] = str(record["status"]).title()
    return record


def _fetch_job_summaries(slug: str) -> list[dict[str, Any]]:
    init_db(profile=slug)
    return [_deserialize_job_record(r) for r in get_all_jobs_with_scores(slug)]


def _fetch_job_detail(slug: str, job_id: str) -> Optional[dict[str, Any]]:
    record = get_job_with_score(job_id, profile=slug)
    if record is None:
        return None
    return _deserialize_job_record(record)


def _search_job_ids_by_raw_text(slug: str, query: str) -> set[str]:
    return search_jobs_by_raw_text(query, profile=slug)


def _collect_metrics(slug: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    # `records` is already filtered to non-disqualified jobs by the caller.
    source_counts: dict[str, int] = {}
    scored = 0
    pending = 0
    failed = 0
    retries = 0
    applied = 0
    skipped = 0
    interest = 0
    reply = 0

    for record in records:
        source_counts[record["source_label"]] = source_counts.get(record["source_label"], 0) + 1
        status = str(record.get("status") or "").lower()
        if status == "applied":
            applied += 1
        elif status == "skip":
            skipped += 1
        elif status == "interest":
            interest += 1
        elif status == "reply":
            reply += 1

        if record["score_state"] == "Scored":
            scored += 1
        elif record["score_state"] == "Pending":
            pending += 1
        elif record["score_state"] == "Failed":
            failed += 1
        else:
            retries += 1

    # avg_fit excludes 0-score disqualified jobs automatically since records
    # is pre-filtered; we also exclude pending (None) scores.
    fit_scores = [
        record["fit_score"]
        for record in records
        if record["fit_score"] is not None and record["fit_score"] > 0
    ]
    avg_fit = round(sum(fit_scores) / len(fit_scores), 1) if fit_scores else 0.0

    return {
        "total": len(records),
        "scored": scored,
        "pending": pending,
        "failed": failed,
        "needs_retry": retries,
        "applied": applied,
        "skipped": skipped,
        "interest": interest,
        "reply": reply,
        "avg_fit": avg_fit,
        "source_counts": source_counts,
        # Callers inject disqualified_count, disqualified_by_reason, db_total,
        # scrape_rejected_count, and scrape_rejected_by_reason.
        "disqualified_count": 0,
        "disqualified_by_reason": {},
        "scrape_rejected_count": 0,
        "scrape_rejected_by_reason": {},
        "db_total": len(records),
    }


def _render_pipeline_snapshot(
    tracker: ProgressTracker,
    host: Any | None,
    *,
    summary: Optional[dict[str, Any]] = None,
    error_message: str | None = None,
    diagnostics: str | None = None,
) -> None:
    if host is None:
        return

    with host.container():
        with panel("Pipeline run", subtitle="Live progress across discovery, scraping, and scoring"):
            render_progress_header(tracker)
            left, right = st.columns([1.1, 0.9], gap="large")
            with left:
                render_pipeline_stages(tracker)
                render_source_progress(tracker)
            with right:
                render_activity_feed(tracker, limit=10)

            if summary:
                stat_row(
                    [
                        ("New jobs", summary["total_new"]),
                        ("Scraped", summary["jobs_scraped"]),
                        ("Filtered", summary["jobs_filtered"]),
                        ("Scored", summary["scored_count"]),
                        ("Avg fit", summary["avg_fit"]),
                    ]
                )

            if error_message:
                callout(
                    "error",
                    "Pipeline failed",
                    error_message,
                )
                if diagnostics:
                    st.code(diagnostics, language="text")




def _clear_profile_jobs(slug: str) -> None:
    init_db(profile=slug)
    clear_profile_jobs(slug)


def _hero(profile_name: str, config: dict[str, Any], metrics: dict[str, Any], raw_config: dict[str, Any]) -> None:
    enabled = _enabled_sources(raw_config)
    profile_cfg = config.get("profile", {})
    provider = config.get("llm", {}).get("provider", "unknown")
    bio = (profile_cfg.get("bio") or "").strip() or "No profile bio saved yet."
    chip_values = [
        profile_cfg.get("job_type", "fulltime").replace("_", " ").title(),
        provider.title(),
    ]
    available_profiles = list_profiles()
    header_action = page_header(
        profile_name,
        bio,
        chips=[
            str(value)
            for value in chip_values
            if value
        ],
        secondary_actions=(
            [
                {
                    "id": "switch_profile",
                    "label": "Change profile",
                    "key": "hero_switch_profile",
                    "use_container_width": True,
                }
            ]
            if len(available_profiles) > 1
            else None
        ),
    )
    if header_action == "switch_profile":
        st.session_state.active_profile = None
        set_active_profile(None)
        st.rerun()


def _render_top_matches(records: list[dict[str, Any]]) -> None:
    scored = [record for record in records if record["fit_score"] is not None]
    if not scored:
        callout("info", "No scored jobs yet", "Start discovery to populate your top matches.")
        return

    for record in scored[:6]:
        badges: list[str] = []
        badges.append(f"<span class='badge'>Fit {record['fit_score']}/100</span>")
        if record["ats_score"] is not None:
            badges.append(f"<span class='badge'>ATS {record['ats_score']}/100</span>")
        if record["flags"]:
            badges.append(f"<span class='badge warn'>{html.escape(record['flags'][0])}</span>")
        summary = html.escape(record["one_liner"] or "Scored and ready for review.")
        st.markdown(
            f"""
            <div class="match-card">
                <div class="match-title">{html.escape(record['title'])}</div>
                <div class="match-meta">
                    {html.escape(record['company'])} · {html.escape(record['location'] or 'Location not listed')}
                    · {html.escape(record['source_label'])}
                </div>
                <div>{summary}</div>
                <div class="badge-row">{''.join(badges)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if record["url"]:
            st.link_button("Open posting", record["url"], key=f"open_posting_{record['id']}")


def _render_run_history(runs: list[dict[str, Any]]) -> None:
    if not runs:
        callout("info", "No run history yet", "The first pipeline run will appear here.")
        return

    frame = pd.DataFrame(
        [
            {
                "Started": run.get("started_at", ""),
                "Status": str(run.get("status", "")).title(),
                "Saved": run.get("jobs_saved", 0),
                "Scored": run.get("jobs_scored", 0),
                "Avg Fit": run.get("avg_fit_score", 0) or 0,
                "Source": run.get("source", ""),
                "Issues": summarize_run_errors(run.get("errors", [])),
            }
            for run in runs
        ]
    )
    st.dataframe(frame, width="stretch", hide_index=True, placeholder="")

    if any(run.get("errors") for run in runs):
        with st.expander("View full run errors", expanded=False):
            for run in runs:
                errors = run.get("errors", [])
                if not errors:
                    continue
                st.write(
                    f"{run.get('started_at', '')} · {str(run.get('status', '')).title()} · "
                    f"{run.get('source', '')}"
                )
                for error in errors:
                    st.code(error)


def _set_job_status_and_refresh(slug: str, job_id: str, status: str) -> None:
    records = _cached_fetch_job_summaries(slug)
    changed = _apply_status_changes(slug, records, [job_id], status)
    if changed:
        _set_notice(slug, "success", f"Updated {changed} job to {status}.")
    st.rerun()


def _status_badge_tone(status: str) -> str:
    """Map a job status to the existing shell-badge tone vocabulary.

    Beacon's status pills (rendered via the .status-pill CSS class) carry
    richer semantics — interest=signal, reply=warn, skip=muted-with-line —
    but the legacy shell-badge palette only supports {info, success,
    warning, danger, neutral}. We map intent here:
        new      → info     (a fresh, unhandled posting)
        interest → success  (you're nodding at it)
        applied  → info     (in flight; tracked via Beacon's pop colour)
        reply    → warning  (recruiter touched, your move)
        skip     → neutral  (out of scope, keep it muted)
    """
    return {
        "applied": "info",
        "interest": "success",
        "reply": "warning",
        "skip": "neutral",
        "new": "info",
    }.get(str(status).lower(), "neutral")


def _score_badge_tone(score_state: str) -> str:
    return {
        "scored": "success",
        "pending": "info",
        "needs retry": "warning",
        "failed": "danger",
    }.get(score_state.lower(), "neutral")


def _run_badge_tone(status: str) -> str:
    return {
        "completed": "success",
        "success": "success",
        "running": "info",
        "started": "info",
        "failed": "danger",
        "error": "danger",
    }.get(status.lower(), "neutral")


def _render_html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def _render_summary_list(items: list[tuple[str, Any]]) -> None:
    markup = "".join(
        f"<div class='summary-row'><span>{html.escape(str(label))}</span><strong>{html.escape(str(value))}</strong></div>"
        for label, value in items
    )
    _render_html_block(f"<div class='summary-list'>{markup}</div>")


def _render_tracking_status_panel(config: dict[str, Any], profile: str) -> None:
    with panel("Experiment tracking", subtitle="Optional MLflow observability for pipeline, evaluation, and semantic runs"):
        _render_summary_list(
            [
                ("MLflow", "Enabled" if mlflow_enabled(config) else "Disabled"),
                ("Tracking URI", get_tracking_uri(config)),
                ("Experiment", get_experiment_name(config, profile)),
            ]
        )
        st.caption("Tracking is local and optional. Normal scraping, scoring, matching, and dashboard flows still work when it is off.")
        st.code("mlflow ui --backend-store-uri ./mlruns", language="bash")


def _queue_pipeline_run(slug: str) -> None:
    if not _worker_is_running(slug):
        _launch_worker(slug)
        st.session_state["_scroll_to_progress"] = True
    st.rerun()


def _open_create_profile_dialog() -> None:
    st.session_state.show_create_profile_dialog = True


def _open_resume_preview_dialog(file_name: str, pdf_bytes: bytes) -> None:
    st.session_state.resume_preview_name = file_name
    st.session_state.resume_preview_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    st.session_state.show_resume_preview_dialog = True


def _launch_create_profile_flow(profile_name: str, employment_type: str) -> None:
    clean_name = profile_name.strip()
    job_type = "internship" if employment_type.lower().startswith("intern") else "fulltime"

    onboarding_data = {
        "name": clean_name,
        "profile_slug": sanitize_slug(clean_name),
        "job_type": job_type,
        "bio": "",
    }
    if job_type == "internship":
        onboarding_data.update(
            {
                "target_season": "Summer",
                "target_year": "2026",
                "graduation_year": "2027",
            }
        )

    st.session_state.show_create_profile_dialog = False
    st.session_state.onboarding_data = onboarding_data
    st.session_state.onboarding_step = 2
    st.session_state.show_onboarding = True
    st.rerun()


def _create_profile_modal_body() -> None:
    st.caption("Start with a name and search mode, then finish the rest in guided setup.")
    with st.form("create_profile_modal_form"):
        profile_name = st.text_input(
            "Profile name",
            value=st.session_state.get("quick_create_profile_name", ""),
            placeholder="Manav Shah",
        )
        employment_type = st.selectbox(
            "Employment type",
            ["Full-time", "Internship"],
            index=0 if st.session_state.get("quick_create_employment_type", "Full-time") == "Full-time" else 1,
        )
        action_cols = st.columns(2, gap="small")
        with action_cols[0]:
            continue_clicked = st.form_submit_button("Continue", type="primary", use_container_width=True)
        with action_cols[1]:
            cancel_clicked = st.form_submit_button("Cancel", use_container_width=True)

    if cancel_clicked:
        st.session_state.show_create_profile_dialog = False
        st.rerun()

    if continue_clicked:
        clean_name = profile_name.strip()
        if not clean_name:
            callout("error", "Profile name required", "Add a profile name before continuing.")
            return
        proposed_slug = sanitize_slug(clean_name)
        if not proposed_slug:
            callout("error", "Profile name required", "Use letters or numbers so the workspace can be created safely.")
            return
        existing_profiles = {profile["slug"] for profile in list_profiles()}
        if proposed_slug in existing_profiles:
            callout("warning", "Profile already exists", f"A profile named '{proposed_slug}' already exists. Choose a different name.")
            return
        st.session_state.quick_create_profile_name = clean_name
        st.session_state.quick_create_employment_type = employment_type
        _launch_create_profile_flow(clean_name, employment_type)


if callable(getattr(st, "dialog", None)):
    @st.dialog("Create new profile")
    def _render_create_profile_dialog() -> None:
        _create_profile_modal_body()
else:
    def _render_create_profile_dialog() -> None:
        with panel("Create new profile", subtitle="Start with a few details, then finish the rest in guided setup"):
            _create_profile_modal_body()


def _render_resume_preview_body() -> None:
    file_name = st.session_state.get("resume_preview_name") or "resume.pdf"
    encoded = st.session_state.get("resume_preview_b64")
    if not encoded:
        callout("error", "Resume preview unavailable", "The PDF bytes were not available for this preview.")
        return

    st.caption("Preview uses the saved PDF bytes for this profile, so it works in deployed environments too.")
    components.html(
        f"""
        <iframe
            src="data:application/pdf;base64,{encoded}#view=FitH"
            width="100%"
            height="720"
            style="border: 1px solid rgba(20, 35, 40, 0.08); border-radius: 12px; background: white;"
        ></iframe>
        """,
        height=740,
    )
    st.download_button(
        "Download PDF",
        data=base64.b64decode(encoded),
        file_name=file_name,
        mime="application/pdf",
        use_container_width=False,
    )


if callable(getattr(st, "dialog", None)):
    @st.dialog("Resume preview")
    def _render_resume_preview_dialog() -> None:
        _render_resume_preview_body()
else:
    def _render_resume_preview_dialog() -> None:
        with panel("Resume preview", subtitle="Open or download the saved PDF resume for this profile"):
            _render_resume_preview_body()


def _overview_filter_chips(config: dict[str, Any]) -> list[str]:
    scoring = config.get("scoring", {})
    location = config.get("preferences", {}).get("location", {})
    preferred_locations = [str(value).strip() for value in location.get("preferred_locations", []) if str(value).strip()]
    if preferred_locations:
        location_label = preferred_locations[0]
        if len(preferred_locations) > 1:
            location_label = f"{location_label} +{len(preferred_locations) - 1}"
    else:
        location_label = "Anywhere"

    return [
        f"Location: {location_label}",
        "Remote OK" if location.get("remote_ok", True) else "Remote off",
        f"Min Score: {int(scoring.get('min_display_score', 60))}",
    ]


def _render_overview_action_card(slug: str, config: dict[str, Any]) -> None:
    with panel(
        "Ready to start",
        subtitle=(
            "Start discovery to generate matches from Greenhouse and Lever. "
            "Your best opportunities will appear here automatically."
        ),
        tone="primary",
    ):
        _render_html_block("<div class='overview-filter-label'>Active filters</div>")
        chip_row(_overview_filter_chips(config))
        action = toolbar(
            primary_actions=[
                {
                    "id": "overview_run_search",
                    "label": "Start discovery",
                    "key": f"overview_run_search_{slug}",
                }
            ],
            secondary_actions=[
                {
                    "id": "overview_edit_filters",
                    "label": "Edit filters",
                    "key": f"overview_edit_filters_{slug}",
                }
            ],
            class_name="shell-toolbar shell-toolbar--compact overview-action-toolbar",
        )
    if action == "overview_run_search":
        _queue_pipeline_run(slug)
    if action == "overview_edit_filters":
        st.session_state.dashboard_section = "Settings"
        st.rerun()


def _render_overview_scoreboard(metrics: dict[str, Any]) -> None:
    applied_rate = f"{round((metrics['applied'] / metrics['total']) * 100)}%" if metrics["total"] else "0%"
    stat_row(
        [
            ("In review", metrics["pending"] + metrics["scored"], "Shortlist"),
            ("Analyzed", metrics["scored"], "Scored"),
            ("Applied", metrics["applied"], f"{applied_rate} forward"),
        ],
        columns_count=3,
    )


def _render_operational_snapshot(metrics: dict[str, Any], raw_config: dict[str, Any]) -> None:
    enabled = _enabled_sources(raw_config)
    enabled_labels = [label for source, label in SOURCE_LABELS.items() if enabled.get(source)]
    if metrics["needs_retry"] or metrics["failed"]:
        health = "Needs attention"
        health_copy = "Check the latest issue before you trust the shortlist."
    elif metrics["total"] == 0:
        health = "No activity yet"
        health_copy = "Run a search to generate your first matches."
    else:
        health = "Ready to review"
        health_copy = "Your latest results are ready. Start with the strongest matches."
    coverage_copy = (
        f"Search coverage: {', '.join(enabled_labels)}"
        if enabled_labels
        else "Search coverage: No sources enabled yet"
    )
    _render_html_block(
        (
            "<div class='ops-hero'>"
            f"<div class='ops-hero-title'>{html.escape(health)}</div>"
            f"<div class='ops-hero-copy'>{html.escape(health_copy)}</div>"
            f"<div class='ops-inline-note'>{html.escape(coverage_copy)}</div>"
            "</div>"
        )
    )


def _render_review_status_card(slug: str, metrics: dict[str, Any], raw_config: dict[str, Any]) -> None:
    enabled = _enabled_sources(raw_config)
    enabled_labels = [label for source, label in SOURCE_LABELS.items() if enabled.get(source)]
    review_ready = metrics["scored"] or (metrics["pending"] + metrics["scored"])
    issue_count = metrics["needs_retry"] + metrics["failed"]

    if issue_count:
        headline = "Needs attention"
        supporting = (
            f"{review_ready} job{'s' if review_ready != 1 else ''} ready. "
            f"{issue_count} item{'s' if issue_count != 1 else ''} need follow-up."
        )
    elif review_ready:
        headline = "Review ready"
        supporting = f"{review_ready} job{'s' if review_ready != 1 else ''} ready in your review queue."
    else:
        headline = "Queue active"
        supporting = "Open Jobs to review the latest saved roles."

    coverage_copy = (
        f"Searching {' and '.join(enabled_labels)}"
        if enabled_labels
        else "No sources enabled yet"
    )

    with st.container(border=True):
        _render_html_block(
            (
                "<div class='review-status-card'>"
                f"<div class='review-status-title'>{html.escape(headline)}</div>"
                f"<div class='review-status-count'>{review_ready}</div>"
                f"<div class='review-status-copy'>{html.escape(supporting)}</div>"
                f"<div class='review-status-meta'>{html.escape(coverage_copy)}</div>"
                "</div>"
            )
        )
        if st.button("Review jobs", key=f"review_jobs_status_{slug}", use_container_width=True):
            st.session_state.dashboard_section = "Jobs"
            st.rerun()


def _render_best_opportunities_panel(slug: str, metrics: dict[str, Any], raw_config: dict[str, Any]) -> None:
    jobs_analyzed = int(metrics.get("scored", 0) or 0)
    enabled = _enabled_sources(raw_config)
    enabled_labels = [label for source, label in SOURCE_LABELS.items() if enabled.get(source)]
    coverage_copy = (
        f"Searching {' and '.join(enabled_labels)}"
        if enabled_labels
        else "No sources enabled yet"
    )

    if jobs_analyzed == 0:
        clicked = empty_state(
            "No opportunities yet",
            "Your best-matching roles will appear here after the first search finishes.",
            actions=[
                {
                    "id": "best_opportunities_start_discovery",
                    "label": "Start discovery",
                    "type": "primary",
                    "key": f"best_opportunities_start_discovery_{slug}",
                }
            ],
        )
        if clicked == "best_opportunities_start_discovery":
            _queue_pipeline_run(slug)
        return

    with st.container(border=True):
        _render_html_block(
            (
                "<div class='review-status-card'>"
                "<div class='review-status-title'>Review ready</div>"
                f"<div class='review-status-copy'>{jobs_analyzed} job{'s' if jobs_analyzed != 1 else ''} ready for review.</div>"
                f"<div class='review-status-meta'>{html.escape(coverage_copy)}</div>"
                "</div>"
            )
        )
        if st.button("Review jobs", key=f"best_opportunities_review_jobs_{slug}", use_container_width=True):
            st.session_state.dashboard_section = "Jobs"
            st.rerun()


def _render_config_summary_card(config: dict[str, Any], raw_config: dict[str, Any]) -> None:
    effective = effective_config_summary(config, raw_config)
    items = [effective[1], effective[2], effective[3], effective[6]]
    _render_summary_list(
        [
            (
                line.split(":", 1)[0],
                line.split(":", 1)[1].strip() if ":" in line else line,
            )
            for line in items
        ]
    )


def _render_match_empty_state() -> None:
    empty_state(
        "No opportunities yet",
        "Your best-matching roles will appear here after the first search finishes.",
    )


def _render_job_detail(record: dict[str, Any], slug: str) -> None:
    with panel(
        "Summary",
        subtitle=(
            f"{record['company']} · {record['location'] or 'Location not listed'} · "
            f"{record['source_label']}"
        ),
    ):
        st.markdown(f"### {record['title']}")
        st.markdown(
            badge(record["status_label"], _status_badge_tone(record["status"]))
            + badge(record["score_state"], _score_badge_tone(record["score_state"]))
            + badge(record["source_label"], "neutral"),
            unsafe_allow_html=True,
        )
        stat_row(
            [
                ("Fit", record["fit_score"] if record["fit_score"] is not None else "N/A"),
                ("ATS", record["ats_score"] if record["ats_score"] is not None else "N/A"),
                ("Attempts", record["score_attempts"] or 0),
            ]
        )
        if record["one_liner"]:
            st.write(record["one_liner"])
        else:
            st.caption("No summary was generated for this job yet.")
        toolbar(
            primary_actions=[],
            secondary_actions=[
                {
                    "id": "open_posting_detail",
                    "label": "Open posting",
                    "url": record["url"],
                }
                if record["url"]
                else {
                    "id": "posting_unavailable_detail",
                    "label": "Posting unavailable",
                    "disabled": True,
                }
            ],
            meta="Use the workflow actions below to update status after you review the posting.",
        )
        if record["score_error"]:
            callout("warning", "Last scoring issue", record["score_error"])

    dims = {key: value for key, value in record["dimension_scores"].items() if value is not None}
    if dims:
        ordered_dims = (
            pd.DataFrame(
                [
                    {"Dimension": key.replace("_", " ").title(), "Score": value}
                    for key, value in dims.items()
                ]
            )
            .sort_values("Score", ascending=False)
            .set_index("Dimension")
        )
        with panel("Scoring breakdown", subtitle="Higher bars contributed more strongly to the fit score"):
            st.caption("These are the scorer dimensions, ordered from strongest signal to weakest.")
            st.bar_chart(ordered_dims)

    with panel("Why it matched", subtitle="Top positive signals from the scoring pass"):
        if record["reasons"]:
            for reason in record["reasons"]:
                st.write(f"- {reason}")
        else:
            empty_state("No match reasons saved", "This score did not include reason bullets.")

    with panel("Watchouts", subtitle="Flags and concerns worth checking before you apply"):
        if record["flags"]:
            for flag in record["flags"]:
                st.write(f"- {flag}")
        else:
            empty_state("No watchouts", "This job does not currently have any saved warning flags.")

    with panel("Missing skills", subtitle="Resume gaps the scorer highlighted for this role"):
        if record["skill_misses"]:
            for skill in record["skill_misses"]:
                st.write(f"- {skill}")
        else:
            empty_state("No missing skills noted", "The scorer did not flag resume skill gaps here.")

    with panel("Actions", subtitle="Update workflow status without leaving the detail view"):
        clicked = toolbar(
            primary_actions=[
                {
                    "id": "mark_applied_detail",
                    "label": "Mark applied",
                    "key": f"mark_applied_{record['id']}",
                }
            ],
            secondary_actions=[
                {
                    "id": "mark_new_detail",
                    "label": "Mark new",
                    "key": f"mark_new_{record['id']}",
                },
                {
                    "id": "mark_skipped_detail",
                    "label": "Mark skipped",
                    "key": f"mark_skipped_{record['id']}",
                },
                {
                    "id": "undo_status_detail",
                    "label": "Undo last status change",
                    "key": f"undo_status_detail_{slug}",
                    "disabled": not (
                        st.session_state.get("last_status_change")
                        and st.session_state["last_status_change"].get("profile") == slug
                    ),
                },
            ],
            meta="Status changes save immediately and can be undone once.",
        )
        if clicked == "mark_applied_detail":
            _set_job_status_and_refresh(slug, record["id"], "applied")
        elif clicked == "mark_new_detail":
            _set_job_status_and_refresh(slug, record["id"], "new")
        elif clicked == "mark_skipped_detail":
            _set_job_status_and_refresh(slug, record["id"], "skip")
        elif clicked == "undo_status_detail":
            restored = _undo_last_status_change(slug)
            if restored:
                _set_notice(slug, "success", f"Restored {restored} status change(s).")
                st.rerun()

    with panel("Job text", subtitle="Expanded posting text is hidden by default for readability"):
        detail_search = st.text_input(
            "Search within this job",
            key=f"job_text_search_{record['id']}",
            placeholder="Find a skill, requirement, or keyword",
        ).strip()
        with st.expander("Show job text", expanded=False):
            raw_text = record["raw_text"] or ""
            if detail_search:
                matches = [
                    line for line in raw_text.splitlines() if detail_search.lower() in line.lower()
                ]
                if matches:
                    st.code("\n".join(matches), language="text")
                else:
                    callout("info", "No matches found", f"Nothing in the saved text matched '{detail_search}'.")
            st.text(raw_text)


def _render_overview_tab(
    slug: str,
    config: dict[str, Any],
    raw_config: dict[str, Any],
    records: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    profile_name = config.get("profile", {}).get("name", slug.replace("_", " ").title())
    with section_shell(
        "Overview",
        SECTION_COPY["Overview"],
        eyebrow=f"{profile_name} workspace",
    ):
        last_result = st.session_state.get("last_pipeline_result")
        if last_result and last_result.get("profile") == slug:
            with panel("Latest pipeline summary", subtitle="Most recent run outcome from this session"):
                stat_row(
                    [
                        ("New jobs", last_result.get("total_new", 0)),
                        ("Saved", last_result.get("jobs_saved", 0)),
                        ("Scored", last_result.get("scored_count", 0)),
                        ("Avg fit", last_result.get("avg_fit", 0)),
                    ]
                )

        stat_row(
            [
                ("Total jobs", metrics["total"]),
                ("Scored", metrics["scored"]),
                ("Pending", metrics["pending"]),
                ("Failed", metrics["failed"]),
                ("Applied", metrics["applied"]),
            ]
        )

        left, right = st.columns([1.3, 0.95], gap="large")
        with left:
            with panel("Top matches", subtitle="Highest-scoring roles in the current profile"):
                _render_top_matches(records)

            with panel(
                "Effective configuration summary",
                subtitle="The settings below currently shape what gets fetched, filtered, and scored",
            ):
                for line in effective_config_summary(config, raw_config):
                    st.write(f"- {line}")

        with right:
            with panel("Pipeline health", subtitle="Quick scan of coverage, retries, and source readiness"):
                enabled = _enabled_sources(raw_config)
                enabled_labels = [
                    label for source, label in SOURCE_LABELS.items() if enabled.get(source)
                ]
                chip_row(enabled_labels or ["No sources enabled"])
                stat_row(
                    [
                        ("Average fit", f"{metrics['avg_fit']}/100"),
                        ("Needs retry", metrics["needs_retry"]),
                        ("Skipped", metrics["skipped"]),
                    ]
                )
                st.caption(
                    f"Active provider: {config.get('llm', {}).get('provider', 'unknown').title()}"
                )

            with panel("Recent runs", subtitle="Recent pipeline activity with compact issue summaries"):
                _render_run_history(runs[:5])


def _render_jobs_tab(records: list[dict[str, Any]], slug: str) -> None:
    profile_name = st.session_state.get("active_profile", slug).replace("_", " ").title()
    _section_heading(profile_name, "Jobs")

    if not records:
        empty_state("No jobs saved yet", "Start discovery to populate this profile.")
        return

    source_options = sorted({record["source_label"] for record in records})
    status_options = sorted({record["status_label"] for record in records})
    score_state_options = sorted({record["score_state"] for record in records})

    with panel("Filters", subtitle="Keep the basics visible and tuck the heavier controls behind one click"):
        top_cols = st.columns([1.8, 1.2, 1.0], gap="medium")
        search = top_cols[0].text_input(
            "Search jobs",
            placeholder="Title, company, location, summary, or keywords",
        )
        min_fit = top_cols[1].slider(
            "Minimum match score",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            help="Only show jobs with a match score at or above this value.",
        )
        include_full_text = top_cols[2].checkbox(
            "Search full description",
            value=False,
            help="Include saved job description text in keyword search. Useful for specific skills or requirements.",
        )

        with st.expander("Edit advanced filters", expanded=False):
            filter_cols = st.columns(4, gap="medium")
            selected_sources = filter_cols[0].multiselect("Source", source_options, default=source_options)
            selected_statuses = filter_cols[1].multiselect("Job status", status_options, default=status_options)
            selected_score_states = filter_cols[2].multiselect(
                "Score state",
                score_state_options,
                default=score_state_options,
            )
            visible_columns = filter_cols[3].multiselect(
                "Visible optional columns",
                JOB_VISIBLE_OPTIONAL_COLUMNS,
                default=["Location", "Source", "ATS", "Summary", "Posting"],
                help="Choose which optional columns appear in the review table.",
            )

    filtered = records
    selected_sources = locals().get("selected_sources", source_options)
    selected_statuses = locals().get("selected_statuses", status_options)
    selected_score_states = locals().get("selected_score_states", score_state_options)
    visible_columns = locals().get("visible_columns", ["Location", "Source", "ATS", "Summary", "Posting"])

    chip_row(
        build_jobs_filter_chips(
            selected_sources,
            source_options,
            selected_statuses,
            status_options,
            selected_score_states,
            score_state_options,
            min_fit,
            search,
            include_full_text,
        )
    )

    if selected_sources:
        filtered = [record for record in filtered if record["source_label"] in selected_sources]
    if selected_statuses:
        filtered = [record for record in filtered if record["status_label"] in selected_statuses]
    if selected_score_states:
        filtered = [record for record in filtered if record["score_state"] in selected_score_states]
    if min_fit > 0:
        filtered = [
            record
            for record in filtered
            if record["fit_score"] is not None and record["fit_score"] >= min_fit
        ]
    if search.strip():
        query = search.lower().strip()
        raw_text_match_ids = _search_job_ids_by_raw_text(slug, query) if include_full_text else set()
        filtered = [
            record
            for record in filtered
            if (
                query in " ".join(
                    [
                        record["title"],
                        record["company"],
                        record["location"] or "",
                        record["source_label"],
                        record["one_liner"],
                    ]
                ).lower()
                or record["id"] in raw_text_match_ids
            )
        ]

    if not filtered:
        empty_state("No jobs match the current filters", "Broaden the search, lower the minimum fit, or clear one of the advanced filters.")
        return

    st.caption(f"Showing {len(filtered)} of {len(records)} jobs")
    table_col, detail_col = st.columns([1.35, 1.0], gap="large")
    frame = build_jobs_table_frame(filtered)
    column_order = resolve_job_table_columns(visible_columns)

    with table_col:
        with panel("Job list", subtitle="Select one or many rows, then run a bulk status update from the toolbar below"):
            selection = st.dataframe(
                frame,
                width="stretch",
                hide_index=True,
                key=f"jobs_table_{slug}",
                on_select="rerun",
                selection_mode="multi-row",
                height=560,
                placeholder="",
                column_order=column_order,
                column_config={
                    "id": None,
                    "Fit": st.column_config.ProgressColumn("Fit", min_value=0, max_value=100),
                    "ATS": st.column_config.ProgressColumn("ATS", min_value=0, max_value=100),
                    "Posting": st.column_config.LinkColumn("Posting", display_text="Open"),
                },
            )

            selected_rows = list(selection.selection.rows)
            selected_ids = [
                str(frame.iloc[row]["id"])
                for row in selected_rows
                if 0 <= row < len(frame)
            ]
            st.markdown(
                badge(f"{len(selected_ids)} selected", "info") if selected_ids else badge("Nothing selected", "neutral"),
                unsafe_allow_html=True,
            )
            bulk_clicked = toolbar(
                primary_actions=[],
                secondary_actions=[
                    {
                        "id": "bulk_applied",
                        "label": "Mark applied",
                        "key": f"bulk_applied_{slug}",
                        "disabled": not selected_ids,
                    },
                    {
                        "id": "bulk_skipped",
                        "label": "Mark skipped",
                        "key": f"bulk_skipped_{slug}",
                        "disabled": not selected_ids,
                    },
                    {
                        "id": "bulk_new",
                        "label": "Mark new",
                        "key": f"bulk_new_{slug}",
                        "disabled": not selected_ids,
                    },
                    {
                        "id": "bulk_undo",
                        "label": "Undo last status change",
                        "key": f"bulk_undo_{slug}",
                        "disabled": not (
                            st.session_state.get("last_status_change")
                            and st.session_state["last_status_change"].get("profile") == slug
                        ),
                    },
                ],
                meta="Bulk actions apply immediately to the selected rows. Undo restores the most recent status change batch.",
            )

            if bulk_clicked == "bulk_undo":
                restored = _undo_last_status_change(slug)
                if restored:
                    _set_notice(slug, "success", f"Restored {restored} status change(s).")
                    st.rerun()
            elif bulk_clicked in {"bulk_applied", "bulk_skipped", "bulk_new"}:
                target_status = {
                    "bulk_applied": "applied",
                    "bulk_skipped": "skip",
                    "bulk_new": "new",
                }[bulk_clicked]
                changed = _apply_status_changes(slug, filtered, selected_ids, target_status)
                if changed:
                    _set_notice(slug, "success", f"Updated {changed} job(s) to {target_status}.")
                st.rerun()

    selected_detail: Optional[dict[str, Any]] = None
    if selected_rows:
        selected_row = selected_rows[0]
        if 0 <= selected_row < len(frame):
            selected_id = str(frame.iloc[selected_row]["id"])
            selected_detail = _cached_fetch_job_detail(slug, selected_id)

    with detail_col:
        if selected_detail is None:
            with panel("Job detail", subtitle="Select at least one row to inspect a role in depth"):
                empty_state(
                    "Select a job",
                    "Choose one row for detailed review. Multi-select still works for bulk status updates.",
                )
        else:
            if len(selected_rows) > 1:
                with panel("Selection overview", subtitle="The first selected row is shown in detail below"):
                    st.write(f"{len(selected_rows)} jobs are selected for bulk actions.")
            _render_job_detail(selected_detail, slug)


def _render_activity_tab(slug: str, runs: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    _section_heading(slug.replace("_", " ").title(), "Activity")
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        with panel("Run log", subtitle="Operational history for this profile, newest first"):
            _render_run_history(runs)

    with right:
        with panel("Source mix", subtitle="How the current database is distributed across sources"):
            source_counts = metrics["source_counts"]
            if source_counts:
                chart_df = pd.DataFrame(
                    {"Source": list(source_counts.keys()), "Jobs": list(source_counts.values())}
                ).set_index("Source")
                st.bar_chart(chart_df)
            else:
                empty_state("No source data yet", "Start discovery to populate source-level metrics.")

        with panel("Cached ATS slugs", subtitle="Discovery cache health for Greenhouse, Lever, Ashby, and Workable"):
            gh_slugs   = load_discovered_slugs(ats="greenhouse", profile=slug)
            lv_slugs   = load_discovered_slugs(ats="lever",      profile=slug)
            ash_slugs  = load_discovered_slugs(ats="ashby",      profile=slug)
            wl_slugs   = load_discovered_slugs(ats="workable",   profile=slug)
            stat_row([
                ("Greenhouse cache", len(gh_slugs)),
                ("Lever cache",      len(lv_slugs)),
                ("Ashby cache",      len(ash_slugs)),
                ("Workable cache",   len(wl_slugs)),
            ])


def _render_top_matches(records: list[dict[str, Any]]) -> None:
    scored = [record for record in records if record["fit_score"] is not None]
    if not scored:
        _render_match_empty_state()
        return

    for record in scored[:6]:
        badges = [
            badge(f"Fit {record['fit_score']}/100", "success"),
            badge(record["status_label"], _status_badge_tone(record["status"])),
        ]
        if record["ats_score"] is not None:
            badges.append(badge(f"ATS {record['ats_score']}/100", "info"))
        if record["flags"]:
            badges.append(badge(record["flags"][0], "warning"))

        _render_html_block(
            (
                "<div class='match-card'>"
                f"<div class='match-title'>{html.escape(record['title'])}</div>"
                f"<div class='match-meta'>{html.escape(record['company'])} | {html.escape(record['location'] or 'Location not listed')} | {html.escape(record['source_label'])}</div>"
                f"<div class='match-summary'>{html.escape(record['one_liner'] or 'Scored and ready for review.')}</div>"
                f"<div class='badge-row'>{''.join(badges)}</div>"
                "</div>"
            )
        )
        if record["url"]:
            st.link_button("Open posting", record["url"], key=f"open_posting_{record['id']}")


def _render_run_history(runs: list[dict[str, Any]]) -> None:
    if not runs:
        empty_state(
            "No run history yet",
            "The first run will create an operational timeline here, including saved jobs, scoring output, and retry signals.",
        )
        return

    for run in runs:
        status = str(run.get("status", "")).title() or "Unknown"
        errors = run.get("errors", [])
        issue_count_label = f"{len(errors)} issue" + ("s" if len(errors) != 1 else "")
        _render_html_block(
            (
                "<div class='run-card'>"
                "<div class='run-card-head'>"
                f"<div><div class='run-card-title'>{html.escape(str(run.get('source', '') or 'dashboard').replace('_', ' ').title())}</div>"
                f"<div class='run-card-meta'>{html.escape(run.get('started_at', ''))}</div></div>"
                f"<div class='run-card-badges'>{badge(status, _run_badge_tone(status))}{badge(issue_count_label, 'warning' if errors else 'neutral')}</div>"
                "</div>"
                "<div class='run-card-stats'>"
                f"<div><span>Saved</span><strong>{run.get('jobs_saved', 0)}</strong></div>"
                f"<div><span>Scored</span><strong>{run.get('jobs_scored', 0)}</strong></div>"
                f"<div><span>Avg fit</span><strong>{run.get('avg_fit_score', 0) or 0}</strong></div>"
                "</div>"
                f"<div class='run-card-issue'>{html.escape(summarize_run_errors(errors))}</div>"
                "</div>"
            )
        )
        if errors:
            with st.expander(f"View issues from {run.get('started_at', '')}", expanded=False):
                for error in errors:
                    st.code(error)


def _render_last_run_summary(
    runs: list[dict[str, Any]],
    *,
    history_expander_label: str = "View run details",
) -> None:
    if not runs:
        _render_html_block(
            (
                "<div class='run-summary-empty'>"
                "<div class='run-summary-status'>No recent activity yet</div>"
                "<div class='run-summary-copy'>Run a search to create your first update.</div>"
                "</div>"
            )
        )
        return

    run = runs[0]
    status = str(run.get("status", "")).title() or "Unknown"
    errors = run.get("errors", [])
    headline = {
        "Complete": "Completed successfully",
        "Completed": "Completed successfully",
        "Success": "Completed successfully",
        "Failed": "Needs attention",
    }.get(status, status)
    summary_parts = [
        f"{run.get('jobs_saved', 0)} saved",
        f"{run.get('jobs_scored', 0)} analyzed",
    ]
    if errors:
        summary_parts.append(f"{len(errors)} issue{'s' if len(errors) != 1 else ''}")

    _render_html_block(
        (
            "<div class='run-summary-row'>"
            "<div class='run-summary-main'>"
            f"<div class='run-summary-status'>{html.escape(headline)}</div>"
            f"<div class='run-summary-copy'>{html.escape(' | '.join(summary_parts))}</div>"
            "</div>"
            f"<div class='run-summary-side'>{badge(status, _run_badge_tone(status))}</div>"
            "</div>"
        )
    )
    with st.expander(history_expander_label, expanded=False):
        _render_run_history(runs)


def _render_overview_run_bar(runs: list[dict[str, Any]]) -> None:
    if not runs:
        return

    run = runs[0]
    status = str(run.get("status", "")).title() or "Unknown"
    status_copy = {
        "Complete": "Completed",
        "Completed": "Completed",
        "Success": "Completed",
        "Failed": "Failed",
    }.get(status, status)
    saved = int(run.get("jobs_saved", 0) or 0)
    analyzed = int(run.get("jobs_scored", 0) or 0)

    row_cols = st.columns([5.4, 1.1], gap="medium")
    with row_cols[0]:
        _render_html_block(
            (
                "<div class='overview-run-bar'>"
                f"<span class='overview-run-bar-label'>Last run</span>"
                f"<span class='overview-run-bar-copy'>{html.escape(f'{status_copy} with {saved} saved and {analyzed} analyzed')}</span>"
                "</div>"
            )
        )
    with row_cols[1]:
        if st.button("View all runs", key="overview_view_all_runs", use_container_width=True):
            st.session_state.dashboard_section = "Activity"
            st.rerun()


def _render_job_detail(record: dict[str, Any], slug: str) -> None:
    if not {"source_label", "status_label", "score_state"}.issubset(record):
        record = _deserialize_job_record(dict(record))
    fit_value = record["fit_score"] if record["fit_score"] is not None else "N/A"
    ats_value = record["ats_score"] if record["ats_score"] is not None else "N/A"
    _render_html_block(
        (
            "<div class='job-detail-header'>"
            f"<div class='job-detail-kicker'>{html.escape(record['company'])}</div>"
            f"<h3 class='job-detail-title'>{html.escape(record['title'])}</h3>"
            f"<div class='job-detail-meta'>{html.escape(record['location'] or 'Location not listed')} | {html.escape(record['source_label'])} | Attempts: {record['score_attempts'] or 0}</div>"
            f"<div class='job-detail-badges'>{badge(record['status_label'], _status_badge_tone(record['status']))}{badge(record['score_state'], _score_badge_tone(record['score_state']))}{badge(record['source_label'], 'neutral')}</div>"
            "</div>"
        )
    )

    score_cols = st.columns(2, gap="medium")
    with score_cols[0]:
        _render_html_block(
            (
                "<div class='score-card score-card--primary'>"
                "<div class='score-card-label'>Fit score</div>"
                f"<div class='score-card-value'>{fit_value}</div>"
                "<div class='score-card-copy'>Weighted match across role fit, stack, seniority, location, growth, and compensation.</div>"
                "</div>"
            )
        )
    with score_cols[1]:
        _render_html_block(
            (
                "<div class='score-card'>"
                "<div class='score-card-label'>ATS score</div>"
                f"<div class='score-card-value'>{ats_value}</div>"
                "<div class='score-card-copy'>Resume-to-posting keyword overlap signal for fast triage.</div>"
                "</div>"
            )
        )

    with panel("Match summary", subtitle="Use this as the fast read before diving into raw posting text"):
        if record["one_liner"]:
            callout("info", "Scorer summary", record["one_liner"])
        else:
            empty_state("No summary generated yet", "A future scoring pass can add a one-line fit summary here.")
        if record["score_error"]:
            callout("warning", "Last scoring issue", record["score_error"])
        clicked = toolbar(
            primary_actions=[],
            secondary_actions=[
                {
                    "id": "open_posting_detail",
                    "label": "Open posting",
                    "url": record["url"],
                }
                if record["url"]
                else {
                    "id": "posting_unavailable_detail",
                    "label": "Posting unavailable",
                    "disabled": True,
                }
            ],
            meta="External posting stays secondary so review can happen here first.",
        )
        if clicked == "open_posting_detail":
            return

    profile_config_for_rating = load_config(profile=slug)
    if record.get("fit_score") is not None:
        profile_intent = normalize_profile_intent(profile_config_for_rating)
        explanation = build_match_explanation(record, record, None, profile_intent)
        with panel("Factor-wise explanation", subtitle="Deterministic reasoning built from stored score evidence"):
            st.caption(f"Recommended action: {explanation.recommended_action}")
            st.write(explanation.summary)
            if explanation.strengths:
                st.write("Top strengths")
                for factor in explanation.strengths[:3]:
                    details = f" Evidence: {' | '.join(factor.evidence[:2])}." if factor.evidence else ""
                    st.write(f"- {factor_with_badge(factor)}: {factor.explanation}{details}")
            if explanation.concerns:
                st.write("Top concerns")
                for factor in explanation.concerns[:3]:
                    details = f" Evidence: {' | '.join(factor.evidence[:2])}." if factor.evidence else ""
                    st.write(f"- {factor_with_badge(factor)}: {factor.explanation}{details}")
            elif explanation.unknowns:
                st.write("Needs more information")
                for factor in explanation.unknowns[:2]:
                    details = f" Evidence: {' | '.join(factor.evidence[:2])}." if factor.evidence else ""
                    st.write(f"- {factor_with_badge(factor)}: {factor.explanation}{details}")

    with panel(
        "Rate this match",
        subtitle="Your rating builds the eval ground-truth for this profile",
    ):
        render_rating_panel(
            slug,
            str(record["id"]),
            role_family=attach_role_family_from_config(slug, profile_config_for_rating),
            key_prefix="rate_detail",
        )

    dims = {key: value for key, value in record["dimension_scores"].items() if value is not None}
    if dims:
        ordered_dims = (
            pd.DataFrame(
                [{"Dimension": key.replace("_", " ").title(), "Score": value} for key, value in dims.items()]
            )
            .sort_values("Score", ascending=False)
            .set_index("Dimension")
        )
        with panel("Scoring breakdown", subtitle="Higher bars had more influence on the fit score"):
            st.bar_chart(ordered_dims)

    insight_cols = st.columns(2, gap="large")
    with insight_cols[0]:
        with panel("Why it matched", subtitle="Positive signals from the scoring pass"):
            if record["reasons"]:
                for reason in record["reasons"]:
                    st.write(f"- {reason}")
            else:
                empty_state("No match reasons saved", "This score did not include saved reason bullets.")
    with insight_cols[1]:
        with panel("Risks and watchouts", subtitle="Check these before deciding to apply"):
            if record["flags"]:
                for flag in record["flags"]:
                    st.write(f"- {flag}")
            else:
                empty_state("No watchouts", "This role has no saved warning flags right now.")

    with panel("Missing skills", subtitle="Resume gaps highlighted by the scorer"):
        if record["skill_misses"]:
            chip_row(record["skill_misses"])
        else:
            empty_state("No missing skills noted", "The scorer did not flag resume skill gaps for this role.")

    with panel("Workflow actions", subtitle="Status updates save immediately and can be undone once"):
        clicked = toolbar(
            primary_actions=[
                {
                    "id": "mark_applied_detail",
                    "label": "Mark applied",
                    "key": f"mark_applied_{record['id']}",
                }
            ],
            secondary_actions=[
                {
                    "id": "mark_new_detail",
                    "label": "Mark new",
                    "key": f"mark_new_{record['id']}",
                },
                {
                    "id": "mark_skipped_detail",
                    "label": "Mark skipped",
                    "key": f"mark_skipped_{record['id']}",
                },
                {
                    "id": "undo_status_detail",
                    "label": "Undo last status change",
                    "key": f"undo_status_detail_{slug}",
                    "disabled": not (
                        st.session_state.get("last_status_change")
                        and st.session_state["last_status_change"].get("profile") == slug
                    ),
                },
            ],
        )
        if clicked == "mark_applied_detail":
            _set_job_status_and_refresh(slug, record["id"], "applied")
        elif clicked == "mark_new_detail":
            _set_job_status_and_refresh(slug, record["id"], "new")
        elif clicked == "mark_skipped_detail":
            _set_job_status_and_refresh(slug, record["id"], "skip")
        elif clicked == "undo_status_detail":
            restored = _undo_last_status_change(slug)
            if restored:
                _set_notice(slug, "success", f"Restored {restored} status change(s).")
                st.rerun()

    with panel("Raw job text", subtitle="Secondary reference material kept out of the main review path"):
        detail_search = st.text_input(
            "Search within this job",
            key=f"job_text_search_{record['id']}",
            placeholder="Find a skill, requirement, or keyword",
        ).strip()
        with st.expander("Show job text", expanded=False):
            raw_text = record["raw_text"] or ""
            if detail_search:
                matches = [line for line in raw_text.splitlines() if detail_search.lower() in line.lower()]
                if matches:
                    st.code("\n".join(matches), language="text")
                else:
                    callout("info", "No matches found", f"Nothing in the saved text matched '{detail_search}'.")
            st.text(raw_text)


# ─────────────────────────────────────────────────────────────────────────────
# Beacon UI (chunk 2): Overview / Jobs / Job-drawer
# ─────────────────────────────────────────────────────────────────────────────

_DRAWER_STATE_KEY = "_beacon_drawer_job_id"
_JOBS_FILTER_STATE_KEY = "_beacon_jobs_filters"
_JOBS_SELECT_STATE_KEY = "_beacon_jobs_selected"
_JOBS_SORT_STATE_KEY = "_beacon_jobs_sort"
_JOBS_DATAFRAME_STATE_KEY = "_beacon_jobs_table"

_DIM_LABELS: tuple[tuple[str, str, str], ...] = (
    ("role_fit",     "Role fit",     "0.30"),
    ("stack_match",  "Stack match",  "0.25"),
    ("seniority",    "Seniority",    "0.20"),
    ("location",     "Location",     "0.10"),
    ("growth",       "Growth",       "0.10"),
    ("compensation", "Compensation", "0.05"),
)


def _verdict_for_score(score: Optional[int]) -> str:
    """Beacon's three-bucket verdict: apply / maybe / skip."""
    if score is None:
        return "skip"
    if score >= 80:
        return "apply"
    if score >= 65:
        return "maybe"
    return "skip"


def _verdict_label(verdict: str) -> str:
    return {"apply": "Apply", "maybe": "Maybe", "skip": "Skip"}.get(verdict, verdict.title())


def _vbadge_html(verdict: str) -> str:
    label = _verdict_label(verdict)
    return f"<span class='vbadge {verdict}'><span class='vd'></span>{html.escape(label)}</span>"


def _status_pill_html(status: str) -> str:
    s = (status or "new").lower()
    if s not in {"new", "interest", "applied", "reply", "skip"}:
        s = "new"
    return f"<span class='status-pill {s}'>{html.escape(s)}</span>"


def _parse_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_relative(value: Any) -> str:
    """Compact human-readable age (e.g. '3h', '5d'). Returns '—' on bad input."""
    dt = _parse_iso(value)
    if dt is None:
        return "—"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - dt
    seconds = max(int(delta.total_seconds()), 0)
    if seconds < 60:
        return "just now" if seconds < 5 else f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    if days < 7:
        return f"{days}d"
    if days < 30:
        return f"{days // 7}w"
    if days < 365:
        return f"{days // 30}mo"
    return f"{days // 365}y"


def _format_run_duration(started: Any, finished: Any) -> str:
    s = _parse_iso(started)
    f = _parse_iso(finished)
    if s is None or f is None:
        return "—"
    total = max(int((f - s).total_seconds()), 0)
    minutes, seconds = divmod(total, 60)
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _count_new_high_fit(records: list[dict[str, Any]], threshold: int = 80) -> int:
    return sum(
        1
        for record in records
        if (record.get("fit_score") or 0) >= threshold
        and str(record.get("status") or "new").lower() == "new"
    )


def _compute_run_state(
    runs: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    worker_running: bool,
) -> dict[str, Any]:
    """Reduce the most recent run + current records into a Beacon RunBanner payload."""
    new_high_fit = _count_new_high_fit(records)
    if worker_running:
        return {
            "state": "running",
            "run_id": None,
            "finished_ago": "in progress",
            "duration": "—",
            "scraped": 0,
            "scored": 0,
            "new_high_fit": new_high_fit,
            "errors": 0,
            "error_detail": "",
        }
    if not runs:
        return {
            "state": "ok",
            "run_id": None,
            "finished_ago": "never",
            "duration": "—",
            "scraped": 0,
            "scored": 0,
            "new_high_fit": new_high_fit,
            "errors": 0,
            "error_detail": "",
        }
    run = runs[0]
    status_raw = str(run.get("status") or "").lower()
    errors = run.get("errors") or []
    error_count = len(errors)
    if status_raw in {"failed", "error", "crashed"}:
        state = "fail"
    elif error_count > 0:
        state = "partial"
    else:
        state = "ok"
    finished_at = run.get("finished_at")
    rel = _format_relative(finished_at) if finished_at else "—"
    finished_ago = f"{rel} ago" if rel not in {"—", "just now"} else rel
    return {
        "state": state,
        "run_id": run.get("run_id"),
        "finished_ago": finished_ago,
        "duration": _format_run_duration(run.get("started_at"), finished_at),
        "scraped": int(run.get("jobs_scraped") or 0),
        "scored": int(run.get("jobs_scored") or 0),
        "new_high_fit": new_high_fit,
        "errors": error_count,
        "error_detail": str(errors[0]) if errors else "",
    }


def _render_beacon_page_header(eyebrow: str, headline: str, sub: str | None = None) -> None:
    sub_html = f"<div class='sub'>{html.escape(sub)}</div>" if sub else ""
    st.markdown(
        (
            "<header class='beacon-ph'>"
            f"<div class='eyebrow'>{html.escape(eyebrow)}</div>"
            f"<h1>{html.escape(headline)}</h1>"
            f"{sub_html}"
            "</header>"
        ),
        unsafe_allow_html=True,
    )


_RUN_STATE_LABEL = {
    "ok":      "Last run · ok",
    "running": "Run in progress",
    "fail":    "Last run · failed",
    "partial": "Last run · partial",
}


def render_run_banner(
    slug: str,
    run_state: dict[str, Any],
    *,
    key_suffix: str = "overview",
    worker_running: bool = False,
) -> None:
    """Beacon's pipeline-state banner: state line + headline + actions.

    The buttons sit in a Streamlit column to the right of the banner card —
    Streamlit can't merge widgets into raw HTML. The border-left strip
    on the card still carries the state colour.
    """
    state = run_state["state"]
    state_label = _RUN_STATE_LABEL.get(state, _RUN_STATE_LABEL["ok"])
    finished = run_state.get("finished_ago") or "—"
    duration = run_state.get("duration") or "—"
    state_meta_html = f"{html.escape(state_label)} · {html.escape(finished)} · {html.escape(duration)}"

    if state == "fail":
        line_html = html.escape(
            run_state.get("error_detail") or "A source failed mid-run. Some jobs are missing."
        )
    else:
        scraped = int(run_state.get("scraped", 0) or 0)
        scored = int(run_state.get("scored", 0) or 0)
        new_h = int(run_state.get("new_high_fit", 0) or 0)
        errors = int(run_state.get("errors", 0) or 0)
        err_html = (
            f" · <span style='color:var(--danger)'>{errors} errors</span>"
            if errors > 0
            else ""
        )
        line_html = (
            f"Scraped <b>{scraped}</b>, scored <b>{scored}</b>, "
            f"surfaced <b>{new_h} new high-fit</b>{err_html}"
        )

    cols = st.columns([6.4, 2.6], gap="medium")
    with cols[0]:
        st.markdown(
            (
                f"<div class='runban {html.escape(state)}'>"
                f"<div class='rb-state'><span class='rb-dot'></span>{state_meta_html}</div>"
                f"<div class='rb-line'>{line_html}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    with cols[1]:
        sub = st.columns(2, gap="small")
        with sub[0]:
            view_clicked = st.button(
                "View pipeline →",
                key=f"runban_view_{key_suffix}_{slug}",
                use_container_width=True,
            )
        with sub[1]:
            run_clicked = st.button(
                "Run again",
                key=f"runban_run_{key_suffix}_{slug}",
                use_container_width=True,
                type="primary",
                disabled=worker_running,
            )
    if view_clicked:
        st.session_state.dashboard_section = "Activity"
        st.rerun()
    if run_clicked:
        _queue_pipeline_run(slug)


def render_match_row(
    record: dict[str, Any],
    slug: str,
    *,
    focused: bool = False,
    key_prefix: str = "mr",
) -> None:
    """Beacon match row: 60px fit number + title/badge/meta/reason + action chips + rating."""
    fit = record.get("fit_score")
    fit_display = fit if fit is not None else "—"
    verdict = _verdict_for_score(fit)
    if isinstance(fit, (int, float)) and fit >= 80:
        tier = ""
    elif isinstance(fit, (int, float)) and fit >= 65:
        tier = "mid"
    else:
        tier = "low"

    title = html.escape(record.get("title") or "Untitled")
    company = html.escape(record.get("company") or "")
    location = html.escape(record.get("location") or "Location not listed")
    src_label = record.get("source_label") or _source_label(record.get("source") or "")
    src_html = html.escape(src_label)
    posted = html.escape(_format_relative(record.get("created_at")))
    reason = (record.get("one_liner") or "").strip()
    reason_html = (
        f"<p class='m-reason'>{html.escape(reason)}</p>" if reason else ""
    )
    focus_cls = "focus" if focused else ""

    st.markdown(
        (
            f"<div class='beacon-match-row {tier} {focus_cls}'>"
            "<div class='beacon-match-fit'>"
            f"<div class='fit-num'>{html.escape(str(fit_display))}</div>"
            "<div class='fit-lbl'>fit</div>"
            "</div>"
            "<div>"
            "<div class='beacon-match-title-row'>"
            f"<h3 class='m-title'>{title}</h3>"
            f"{_vbadge_html(verdict)}"
            "</div>"
            "<div class='m-meta'>"
            f"<span class='co'>{company}</span><span class='dot'></span>"
            f"<span>{location}</span><span class='dot'></span>"
            f"<span>{src_html}</span><span class='dot'></span>"
            f"<span>{posted} ago</span>"
            "</div>"
            f"{reason_html}"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    job_id = str(record.get("id") or "")
    cur_status = str(record.get("status") or "new").lower()

    btn_cols = st.columns([1.4, 0.95, 0.95, 1.1], gap="small")
    with btn_cols[0]:
        applied_label = "Applied" if cur_status == "applied" else "Mark applied"
        if st.button(
            applied_label,
            key=f"{key_prefix}_apply_{slug}_{job_id}",
            type="primary" if cur_status == "applied" else "secondary",
            use_container_width=True,
        ):
            update_job_status(job_id, "applied", profile=slug)
            invalidate_dashboard_caches()
            st.toast(f"Marked as applied: {record.get('title', '')}")
            st.rerun()
    with btn_cols[1]:
        if st.button(
            "Save",
            key=f"{key_prefix}_save_{slug}_{job_id}",
            use_container_width=True,
        ):
            update_job_status(job_id, "interest", profile=slug)
            invalidate_dashboard_caches()
            st.toast(f"Saved: {record.get('title', '')}")
            st.rerun()
    with btn_cols[2]:
        if st.button(
            "Skip",
            key=f"{key_prefix}_skip_{slug}_{job_id}",
            use_container_width=True,
        ):
            update_job_status(job_id, "skip", profile=slug)
            invalidate_dashboard_caches()
            st.toast(f"Skipped: {record.get('title', '')}")
            st.rerun()
    with btn_cols[3]:
        if st.button(
            "Open detail",
            key=f"{key_prefix}_open_{slug}_{job_id}",
            use_container_width=True,
        ):
            st.session_state[_DRAWER_STATE_KEY] = job_id
            st.rerun()

    profile_config_for_rating = load_config(profile=slug)
    render_rating_panel(
        slug,
        job_id,
        role_family=attach_role_family_from_config(slug, profile_config_for_rating),
        key_prefix=f"{key_prefix}_rate_{slug}",
        show_helper=False,
    )
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


def _render_pipeline_side_card(slug: str, worker_running: bool, runs: list[dict[str, Any]]) -> None:
    """Side-column live-pipeline card for the Overview screen.

    Mirrors Beacon's `.ag-card`: pulse + run id + elapsed + progress bar + stages.
    Falls back to a quiescent view when the worker isn't running.
    """
    run_label = "Live pipeline · idle"
    elapsed_label = ""
    progress_pct = 0
    stages_html = ""
    detail_label = "View pipeline →"

    progress_data = _read_progress_json(slug) if worker_running else None
    if worker_running and progress_data:
        try:
            tracker = ProgressTracker.from_dict(progress_data)
            run_label = "Live pipeline · running"
            elapsed = tracker.elapsed_time
            mins, secs = divmod(int(elapsed.total_seconds()), 60)
            elapsed_label = f"{mins:02d}:{secs:02d} elapsed"
            progress_pct = max(0, min(100, int(tracker.overall_progress_pct)))
            stage_rows = [
                (Stage.DISCOVERING, "Discovery"),
                (Stage.FETCHING, "Job board fetch"),
                (Stage.SCRAPING, "Scraping"),
                (Stage.SCORING, "Scoring & filtering"),
                (Stage.EMBEDDING, "Embeddings"),
                (Stage.FINALIZING, "Finalize"),
            ]
            rendered = []
            for stage, label in stage_rows:
                prog = tracker.stages.get(stage)
                status = prog.status if prog else None
                cls = "queue"
                if status == StageStatus.COMPLETE:
                    cls = "done"
                elif status == StageStatus.RUNNING:
                    cls = "run"
                elif status == StageStatus.FAILED:
                    cls = "fail"
                count_html = ""
                if prog and prog.metrics:
                    metric_str = " · ".join(f"{k}: {v}" for k, v in list(prog.metrics.items())[:2])
                    count_html = f"<span class='stg-count'>{html.escape(metric_str)}</span>"
                rendered.append(
                    "<div class='stage " + cls + "'>"
                    "<span class='stg-icon'></span>"
                    f"<span class='stg-name'>{html.escape(label)}</span>"
                    f"{count_html}"
                    "</div>"
                )
            stages_html = "<div class='stages'>" + "".join(rendered) + "</div>"
        except Exception:  # noqa: BLE001 — defensive: malformed progress JSON shouldn't break overview
            stages_html = ""
    elif runs:
        last = runs[0]
        run_label = f"Live pipeline · idle (last run #{html.escape(str(last.get('run_id') or ''))})"
        elapsed_label = (
            f"{_format_relative(last.get('finished_at'))} ago"
            if last.get("finished_at")
            else ""
        )
        progress_pct = 100
        detail_label = "View last run →"

    st.markdown(
        (
            "<div class='pipeline-side-card'>"
            "<div class='ag-head'>"
            "<span class='ag-pulse'></span>"
            f"<span class='ag-title'>{html.escape(run_label)}</span>"
            f"<span class='ag-elapsed'>{html.escape(elapsed_label)}</span>"
            "</div>"
            f"<div class='progress'><div style='width:{progress_pct}%'></div></div>"
            f"{stages_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    if st.button(detail_label, key=f"side_pipeline_view_{slug}", use_container_width=True):
        st.session_state.dashboard_section = "Activity"
        st.rerun()


def _build_today_headline(metrics: dict[str, Any], records: list[dict[str, Any]], runs: list[dict[str, Any]]) -> str:
    strong = sum(1 for r in records if (r.get("fit_score") or 0) >= 80)
    reply_count = int(metrics.get("reply", 0) or 0)
    error_count = 0
    if runs:
        error_count = len(runs[0].get("errors") or [])
    parts = [
        f"{strong} strong pick{'s' if strong != 1 else ''}.",
        f"{reply_count} repl{'ies' if reply_count != 1 else 'y'}.",
        f"{error_count} error{'s' if error_count != 1 else ''}.",
    ]
    return " ".join(parts)


# ── Job Drawer (st.dialog modal) ────────────────────────────────────────────

def _close_job_drawer() -> None:
    if _DRAWER_STATE_KEY in st.session_state:
        st.session_state[_DRAWER_STATE_KEY] = None


def _classify_reason_tag(reason: str) -> str:
    """Beacon's three colour bands for reasoning bullets.

    - "negative" / "block" / "disq" / "miss" → red (.neg)
    - "warn" / "concern" / "caveat" → amber (.warn)
    - everything else → green strip (default)
    """
    text = (reason or "").strip().lower()
    if not text:
        return ""
    for needle in ("negative:", "block:", "disq", "no-go:", "miss:"):
        if text.startswith(needle):
            return "neg"
    for needle in ("warn:", "warning:", "concern:", "caveat:", "watch:"):
        if text.startswith(needle):
            return "warn"
    return ""


def _strip_reason_prefix(reason: str) -> str:
    text = (reason or "").strip()
    for prefix in ("positive:", "negative:", "warn:", "warning:", "block:", "concern:"):
        if text.lower().startswith(prefix):
            return text[len(prefix):].strip()
    return text


def _render_drawer_dim_grid(record: dict[str, Any]) -> None:
    dims = record.get("dimension_scores") or {}
    cards: list[str] = []
    for key, label, weight in _DIM_LABELS:
        raw = dims.get(key)
        try:
            value = round(float(raw), 1) if raw is not None else 0.0
        except (TypeError, ValueError):
            value = 0.0
        bar_pct = max(0, min(100, int(value * 10)))
        low_cls = "low" if value < 5 else ""
        cards.append(
            "<div class='dim " + low_cls + "'>"
            "<div class='dim-row'>"
            f"<span class='nm'>{html.escape(label)}</span>"
            f"<span class='w'>w {html.escape(weight)}</span>"
            "</div>"
            "<div class='dim-row'>"
            f"<span class='v'>{value:g}<small>/10</small></span>"
            "</div>"
            f"<div class='bar'><div style='width:{bar_pct}%'></div></div>"
            "</div>"
        )
    st.markdown("<div class='dim-grid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


def _render_job_drawer_body(record: dict[str, Any], slug: str) -> None:
    fit = record.get("fit_score")
    fit_display = fit if fit is not None else "—"
    verdict = _verdict_for_score(fit)
    title = html.escape(record.get("title") or "Untitled")
    company = html.escape(record.get("company") or "")
    location = html.escape(record.get("location") or "Location not listed")
    src_label = record.get("source_label") or _source_label(record.get("source") or "")
    src_html = html.escape(src_label)
    posted = html.escape(_format_relative(record.get("created_at")))
    job_id = str(record["id"])

    head_cols = st.columns([6.5, 1.0], gap="small")
    with head_cols[0]:
        st.markdown(
            (
                "<div class='drawer-head'>"
                "<div class='beacon-match-fit'>"
                f"<div class='fit-num'>{html.escape(str(fit_display))}</div>"
                "<div class='fit-lbl'>fit</div>"
                "</div>"
                "<div style='flex:1'>"
                "<div class='beacon-match-title-row'>"
                f"<h3 class='m-title' style='font-size:18px;letter-spacing:-0.018em'>{title}</h3>"
                f"{_vbadge_html(verdict)}"
                "</div>"
                "<div class='m-meta'>"
                f"<span class='co'>{company}</span><span class='dot'></span>"
                f"<span>{location}</span><span class='dot'></span>"
                f"<span>{src_html}</span><span class='dot'></span>"
                f"<span>{posted} ago</span>"
                "</div>"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    with head_cols[1]:
        if st.button("Close", key=f"drawer_close_top_{job_id}", use_container_width=True):
            _close_job_drawer()
            st.rerun()

    # 1) Train Beacon — full rating panel + notes
    st.markdown(
        (
            "<section class='train-card'>"
            "<div>"
            "<div class='train-title'>"
            "<span class='train-ic'>●</span>How was this match?"
            "</div>"
            "<div class='train-sub'>Trains future scoring. Independent of whether you apply.</div>"
            "</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )
    profile_config_for_rating = load_config(profile=slug)
    render_rating_panel(
        slug,
        job_id,
        role_family=attach_role_family_from_config(slug, profile_config_for_rating),
        key_prefix=f"drawer_rate_{slug}",
    )

    # 2) Why the agent picked this — colour-coded reasoning bullets
    st.markdown(
        "<div class='beacon-card-title' style='margin-top:18px;margin-bottom:8px'>Why the agent picked this</div>",
        unsafe_allow_html=True,
    )
    reasons: list[str] = list(record.get("reasons") or [])
    flags: list[str] = list(record.get("flags") or [])
    items: list[str] = []
    for reason in reasons:
        cls = _classify_reason_tag(reason)
        items.append(f"<li class='{cls}'>{html.escape(_strip_reason_prefix(reason))}</li>")
    for flag in flags:
        items.append(f"<li class='warn'>{html.escape(_strip_reason_prefix(str(flag)))}</li>")
    if not items:
        items.append("<li>Solid overall fit on stack and seniority. No flags raised.</li>")
    st.markdown("<ul class='reasoning'>" + "".join(items) + "</ul>", unsafe_allow_html=True)

    # 3) Score breakdown — 6-dim grid
    st.markdown(
        "<div class='beacon-card-title' style='margin-top:18px;margin-bottom:8px'>Score breakdown</div>",
        unsafe_allow_html=True,
    )
    _render_drawer_dim_grid(record)

    # 4) Action row — placeholders + open posting
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    action_cols = st.columns([1.2, 1.2, 1.0, 1.4], gap="small")
    with action_cols[0]:
        st.button(
            "Generate cover note",
            key=f"drawer_cover_{job_id}",
            disabled=True,
            help="Coming in a later chunk.",
            use_container_width=True,
        )
    with action_cols[1]:
        st.button(
            "Tailor resume",
            key=f"drawer_tailor_{job_id}",
            disabled=True,
            help="Coming in a later chunk.",
            use_container_width=True,
        )
    with action_cols[3]:
        url = record.get("url")
        if url:
            st.link_button("Open posting ↗", url, use_container_width=True)
        else:
            st.button(
                "Posting unavailable",
                key=f"drawer_url_missing_{job_id}",
                disabled=True,
                use_container_width=True,
            )

    # Footer: prev/next stubs + Skip / Save / Mark applied
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    foot_cols = st.columns([0.9, 0.9, 4.5, 1.0, 1.0, 1.4], gap="small")
    with foot_cols[0]:
        st.button(
            "← prev",
            key=f"drawer_prev_{job_id}",
            disabled=True,
            help="Job navigation arrives in a later chunk.",
            use_container_width=True,
        )
    with foot_cols[1]:
        st.button(
            "next →",
            key=f"drawer_next_{job_id}",
            disabled=True,
            help="Job navigation arrives in a later chunk.",
            use_container_width=True,
        )
    with foot_cols[3]:
        if st.button("Skip", key=f"drawer_skip_{job_id}", use_container_width=True):
            update_job_status(job_id, "skip", profile=slug)
            invalidate_dashboard_caches()
            st.toast(f"Skipped: {record.get('title', '')}")
            _close_job_drawer()
            st.rerun()
    with foot_cols[4]:
        if st.button("Save", key=f"drawer_save_{job_id}", use_container_width=True):
            update_job_status(job_id, "interest", profile=slug)
            invalidate_dashboard_caches()
            st.toast(f"Saved: {record.get('title', '')}")
            _close_job_drawer()
            st.rerun()
    with foot_cols[5]:
        if st.button(
            "Mark applied",
            key=f"drawer_apply_{job_id}",
            type="primary",
            use_container_width=True,
        ):
            update_job_status(job_id, "applied", profile=slug)
            invalidate_dashboard_caches()
            st.toast(f"Marked as applied: {record.get('title', '')}")
            _close_job_drawer()
            st.rerun()


@st.dialog("Job detail", width="large")
def _job_drawer_dialog(slug: str) -> None:
    job_id = st.session_state.get(_DRAWER_STATE_KEY)
    if not job_id:
        return
    record = _cached_fetch_job_detail(slug, str(job_id))
    if record is None:
        st.error("That job is no longer available.")
        if st.button("Close", key="drawer_missing_close"):
            _close_job_drawer()
            st.rerun()
        return
    _render_job_drawer_body(record, slug)


def _maybe_open_drawer(slug: str) -> None:
    """Open the drawer dialog if the session-state flag is set."""
    if st.session_state.get(_DRAWER_STATE_KEY):
        _job_drawer_dialog(slug)


# ── Jobs tab helpers ────────────────────────────────────────────────────────

_JOBS_DEFAULT_FILTERS: dict[str, Any] = {
    "fit": 60,
    "verdict": "all",
    "rated": "all",
    "status": "all",
    "remote": False,
    "search": "",
}
_JOBS_DEFAULT_SORT: dict[str, str] = {"key": "fit", "dir": "desc"}

_VERDICT_CYCLE = ("all", "apply", "maybe", "skip")
_RATED_CYCLE = ("all", "rated", "unrated")
_STATUS_CYCLE = ("all", "new", "interest", "applied", "reply", "skip")
_FIT_CYCLE = (60, 80)


def _jobs_filter_state(slug: str) -> dict[str, Any]:
    key = f"{_JOBS_FILTER_STATE_KEY}_{slug}"
    if key not in st.session_state:
        st.session_state[key] = dict(_JOBS_DEFAULT_FILTERS)
    return st.session_state[key]


def _jobs_sort_state(slug: str) -> dict[str, str]:
    key = f"{_JOBS_SORT_STATE_KEY}_{slug}"
    if key not in st.session_state:
        st.session_state[key] = dict(_JOBS_DEFAULT_SORT)
    return st.session_state[key]


def _cycle_value(current: Any, options: tuple) -> Any:
    try:
        idx = options.index(current)
    except ValueError:
        return options[0]
    return options[(idx + 1) % len(options)]


def _filter_jobs(records: list[dict[str, Any]], filters: dict[str, Any], ratings_index: dict[str, Any]) -> list[dict[str, Any]]:
    fit_min = int(filters.get("fit", 60) or 0)
    verdict = filters.get("verdict", "all")
    rated = filters.get("rated", "all")
    status = filters.get("status", "all")
    remote = bool(filters.get("remote", False))
    query = (filters.get("search") or "").strip().lower()

    out: list[dict[str, Any]] = []
    for record in records:
        score = record.get("fit_score") or 0
        if score < fit_min:
            continue
        if verdict != "all" and _verdict_for_score(record.get("fit_score")) != verdict:
            continue
        if status != "all" and (record.get("status") or "new").lower() != status:
            continue
        if remote and "remote" not in (record.get("location") or "").lower():
            continue
        rating = ratings_index.get(str(record.get("id")))
        if rated == "rated" and not rating:
            continue
        if rated == "unrated" and rating:
            continue
        if query:
            haystack = " ".join(
                str(record.get(field) or "")
                for field in ("title", "company", "location", "one_liner")
            ).lower()
            if query not in haystack:
                continue
        out.append(record)
    return out


def _sort_jobs(records: list[dict[str, Any]], sort: dict[str, str], ratings_index: dict[str, Any]) -> list[dict[str, Any]]:
    key = sort.get("key", "fit")
    direction = sort.get("dir", "desc")
    reverse = direction == "desc"
    verdict_rank = {"apply": 3, "maybe": 2, "skip": 1}
    rating_rank = {"great_match": 1, "good_match": 2, "okay_match": 3, "bad_match": 4, "should_skip": 5}

    if key == "fit":
        return sorted(records, key=lambda r: r.get("fit_score") or 0, reverse=reverse)
    if key == "verdict":
        return sorted(
            records,
            key=lambda r: verdict_rank.get(_verdict_for_score(r.get("fit_score")), 0),
            reverse=reverse,
        )
    if key == "title":
        return sorted(records, key=lambda r: (r.get("title") or "").lower(), reverse=reverse)
    if key == "company":
        return sorted(records, key=lambda r: (r.get("company") or "").lower(), reverse=reverse)
    if key == "location":
        return sorted(records, key=lambda r: (r.get("location") or "").lower(), reverse=reverse)
    if key == "posted":
        def _ts(r):
            dt = _parse_iso(r.get("created_at"))
            return dt.timestamp() if dt else 0.0
        return sorted(records, key=_ts, reverse=reverse)
    if key == "status":
        return sorted(records, key=lambda r: (r.get("status") or "new").lower(), reverse=reverse)
    if key == "rating":
        return sorted(
            records,
            key=lambda r: rating_rank.get(
                (ratings_index.get(str(r.get("id"))).label if ratings_index.get(str(r.get("id"))) else None),
                9,
            ),
            reverse=not reverse,  # rated-best first when desc, unrated last
        )
    return records


def _build_jobs_dataframe(
    records: list[dict[str, Any]],
    *,
    ratings_index: dict[str, Any],
) -> pd.DataFrame:
    label_map = {opt[0]: opt[1] for opt in RATING_OPTIONS}
    rows = []
    for record in records:
        rating_obj = ratings_index.get(str(record.get("id")))
        rating_label = label_map.get(rating_obj.label, rating_obj.label) if rating_obj else ""
        rows.append(
            {
                "_id": str(record.get("id") or ""),
                "Fit": record.get("fit_score") or 0,
                "Verdict": _verdict_label(_verdict_for_score(record.get("fit_score"))),
                "Title": record.get("title") or "",
                "Company": record.get("company") or "",
                "Location": record.get("location") or "—",
                "Posted": _format_relative(record.get("created_at")),
                "Status": (record.get("status") or "new").title(),
                "Rating": rating_label or "—",
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Tab renderers (Beacon edition)
# ─────────────────────────────────────────────────────────────────────────────


def _render_overview_tab(
    slug: str,
    config: dict[str, Any],
    raw_config: dict[str, Any],
    records: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    worker_running = _worker_is_running(slug)
    now = datetime.now()
    today_label = now.strftime("Today · %a, %b ") + str(now.day)
    headline = _build_today_headline(metrics, records, runs)
    sub = "Top of feed below — clear it in under five minutes."
    _render_beacon_page_header(today_label, headline, sub)

    run_state = _compute_run_state(runs, records, worker_running=worker_running)
    render_run_banner(slug, run_state, key_suffix="overview", worker_running=worker_running)

    cols = st.columns([1.45, 1.0], gap="large")
    scored_records = [r for r in records if r.get("fit_score") is not None]
    scored_records.sort(key=lambda r: r.get("fit_score") or 0, reverse=True)
    top_records = scored_records[:3]
    with cols[0]:
        see_all_label = f"See all {len(scored_records)} →"
        head_cols = st.columns([6.5, 2.0], gap="small")
        with head_cols[0]:
            st.markdown(
                (
                    "<div class='beacon-card-head' style='border-bottom:1px solid var(--line);"
                    "background:var(--surface);border:1px solid var(--line);"
                    "border-radius:var(--r-md) var(--r-md) 0 0;padding:14px 18px'>"
                    "<div>"
                    "<div class='beacon-card-title'>Top picks for you</div>"
                    "<div class='beacon-card-sub'>Ranked by overall fit</div>"
                    "</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        with head_cols[1]:
            if st.button(see_all_label, key=f"overview_see_all_{slug}", use_container_width=True):
                st.session_state.dashboard_section = "Jobs"
                st.rerun()

        if not top_records:
            st.markdown(
                (
                    "<div class='beacon-card' style='padding:24px'>"
                    "<div class='beacon-empty'>"
                    "<div class='ic'>·</div>"
                    "<div class='h'>No scored jobs yet.</div>"
                    "<div class='s'>Run the pipeline to surface your strongest picks.</div>"
                    "</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        else:
            for record in top_records:
                render_match_row(record, slug, key_prefix="overview_match")
    with cols[1]:
        _render_pipeline_side_card(slug, worker_running, runs)

    _maybe_open_drawer(slug)


def _render_jobs_tab(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    slug: str,
    *,
    scrape_rejected_records: list[dict[str, Any]] | None = None,
) -> None:
    """Beacon Jobs screen: header + toolbar + table + keyboard hints + drawer.

    `scrape_rejected_records` is accepted for signature compatibility with the
    caller in `_render_profile_dashboard` but is no longer surfaced here —
    Beacon's Jobs screen is purely the scored review queue.
    """
    filters = _jobs_filter_state(slug)
    sort = _jobs_sort_state(slug)
    ratings_index = get_all_user_ratings(slug)

    filtered = _filter_jobs(records, filters, ratings_index)
    sorted_records = _sort_jobs(filtered, sort, ratings_index)

    _render_beacon_page_header(
        f"Jobs · {len(filtered)} of {len(records)}",
        "Your job board, scored.",
        "Filter, sort, and triage. Click a row to open detail; bulk-select to mark many at once.",
    )

    # ── Toolbar ─────────────────────────────────────────────────────────
    st.markdown("<div class='beacon-toolbar' style='margin-bottom:0'>", unsafe_allow_html=True)
    toolbar_cols = st.columns([3.4, 1.0, 1.2, 1.2, 1.2, 0.9, 1.4], gap="small")
    with toolbar_cols[0]:
        new_search = st.text_input(
            "Search jobs",
            value=filters.get("search", ""),
            key=f"beacon_jobs_search_{slug}",
            placeholder="Search title, company, keyword…",
            label_visibility="collapsed",
        )
        if new_search != filters.get("search", ""):
            filters["search"] = new_search
            st.rerun()
    with toolbar_cols[1]:
        fit_label = f"Fit ≥ {filters['fit']}"
        if st.button(
            fit_label,
            key=f"beacon_jobs_fit_{slug}",
            type="primary" if filters["fit"] != _JOBS_DEFAULT_FILTERS["fit"] else "secondary",
            use_container_width=True,
        ):
            filters["fit"] = _cycle_value(filters["fit"], _FIT_CYCLE)
            st.rerun()
    with toolbar_cols[2]:
        if st.button(
            f"Verdict · {filters['verdict']}",
            key=f"beacon_jobs_verdict_{slug}",
            type="primary" if filters["verdict"] != "all" else "secondary",
            use_container_width=True,
        ):
            filters["verdict"] = _cycle_value(filters["verdict"], _VERDICT_CYCLE)
            st.rerun()
    with toolbar_cols[3]:
        if st.button(
            f"Rated · {filters['rated']}",
            key=f"beacon_jobs_rated_{slug}",
            type="primary" if filters["rated"] != "all" else "secondary",
            use_container_width=True,
        ):
            filters["rated"] = _cycle_value(filters["rated"], _RATED_CYCLE)
            st.rerun()
    with toolbar_cols[4]:
        if st.button(
            f"Status · {filters['status']}",
            key=f"beacon_jobs_status_{slug}",
            type="primary" if filters["status"] != "all" else "secondary",
            use_container_width=True,
        ):
            filters["status"] = _cycle_value(filters["status"], _STATUS_CYCLE)
            st.rerun()
    with toolbar_cols[5]:
        if st.button(
            "Remote",
            key=f"beacon_jobs_remote_{slug}",
            type="primary" if filters["remote"] else "secondary",
            use_container_width=True,
        ):
            filters["remote"] = not filters["remote"]
            st.rerun()
    with toolbar_cols[6]:
        st.markdown(
            f"<div style='text-align:right;font-family:var(--font-mono);font-size:11px;color:var(--muted);padding-top:6px'>{len(sorted_records)} of {len(records)}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if not records:
        st.markdown(
            (
                "<div class='beacon-empty'>"
                "<div class='ic'>·</div>"
                "<div class='h'>Your review queue is empty.</div>"
                "<div class='s'>Run the pipeline to populate a curated list of scored matches.</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        if st.button("Run pipeline", key=f"jobs_empty_run_{slug}", type="primary"):
            _queue_pipeline_run(slug)
        return

    if not sorted_records:
        st.markdown(
            (
                "<div class='beacon-empty'>"
                "<div class='ic'>⌕</div>"
                "<div class='h'>No matches.</div>"
                "<div class='s'>Loosen the filters or run a fresh pipeline.</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        empty_cols = st.columns([1, 1, 6], gap="small")
        with empty_cols[0]:
            if st.button("Clear filters", key=f"jobs_empty_clear_{slug}", use_container_width=True):
                st.session_state[f"{_JOBS_FILTER_STATE_KEY}_{slug}"] = dict(_JOBS_DEFAULT_FILTERS)
                st.rerun()
        with empty_cols[1]:
            if st.button("Run pipeline", key=f"jobs_empty_run_{slug}", type="primary", use_container_width=True):
                _queue_pipeline_run(slug)
        return

    # ── Sort controls ───────────────────────────────────────────────────
    sort_cols = st.columns([0.7, 1.4, 1.0, 0.6, 8.0], gap="small")
    sort_options = {
        "fit": "Fit",
        "verdict": "Verdict",
        "title": "Title",
        "company": "Company",
        "location": "Location",
        "posted": "Posted",
        "status": "Status",
        "rating": "Rating",
    }
    with sort_cols[0]:
        st.markdown(
            "<div style='font-family:var(--font-mono);font-size:11px;color:var(--muted);padding-top:8px'>Sort by</div>",
            unsafe_allow_html=True,
        )
    with sort_cols[1]:
        keys = list(sort_options.keys())
        cur_key = sort.get("key", "fit")
        idx = keys.index(cur_key) if cur_key in keys else 0
        new_key = st.selectbox(
            "Sort by",
            keys,
            index=idx,
            key=f"beacon_jobs_sortkey_{slug}",
            format_func=lambda k: sort_options[k],
            label_visibility="collapsed",
        )
        if new_key != cur_key:
            sort["key"] = new_key
            st.rerun()
    with sort_cols[2]:
        cur_dir = sort.get("dir", "desc")
        new_dir = st.selectbox(
            "Direction",
            ["desc", "asc"],
            index=0 if cur_dir == "desc" else 1,
            key=f"beacon_jobs_sortdir_{slug}",
            format_func=lambda d: "↓ desc" if d == "desc" else "↑ asc",
            label_visibility="collapsed",
        )
        if new_dir != cur_dir:
            sort["dir"] = new_dir
            st.rerun()

    # ── Table (st.dataframe with multi-row selection) ───────────────────
    frame = _build_jobs_dataframe(sorted_records, ratings_index=ratings_index)
    selection = st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        key=f"beacon_jobs_table_{slug}",
        on_select="rerun",
        selection_mode="multi-row",
        height=520,
        placeholder="",
        column_order=["Fit", "Verdict", "Title", "Company", "Location", "Posted", "Status", "Rating"],
        column_config={
            "_id": None,
            "Fit": st.column_config.ProgressColumn("Fit", min_value=0, max_value=100, format="%d"),
            "Verdict": st.column_config.TextColumn("Verdict"),
            "Title": st.column_config.TextColumn("Title"),
            "Company": st.column_config.TextColumn("Company"),
            "Location": st.column_config.TextColumn("Location"),
            "Posted": st.column_config.TextColumn("Posted"),
            "Status": st.column_config.TextColumn("Status"),
            "Rating": st.column_config.TextColumn("Rating"),
        },
    )

    selected_rows = list(selection.selection.rows)
    selected_ids = [str(frame.iloc[r]["_id"]) for r in selected_rows if 0 <= r < len(frame)]

    # ── Bulk-action bar ─────────────────────────────────────────────────
    if selected_ids:
        st.markdown(
            f"<div class='bulk-bar'><span class='ct'>{len(selected_ids)} selected</span></div>",
            unsafe_allow_html=True,
        )
        bulk_cols = st.columns([1.3, 1.4, 1.0, 1.0, 6.0], gap="small")
        with bulk_cols[0]:
            if st.button("Mark applied", key=f"beacon_bulk_apply_{slug}", use_container_width=True):
                changed = _apply_status_changes(slug, sorted_records, selected_ids, "applied")
                if changed:
                    st.toast(f"Marked {changed} job(s) as applied")
                invalidate_dashboard_caches()
                st.rerun()
        with bulk_cols[1]:
            if st.button("Mark interested", key=f"beacon_bulk_interest_{slug}", use_container_width=True):
                changed = _apply_status_changes(slug, sorted_records, selected_ids, "interest")
                if changed:
                    st.toast(f"Saved {changed} job(s)")
                invalidate_dashboard_caches()
                st.rerun()
        with bulk_cols[2]:
            if st.button("Skip", key=f"beacon_bulk_skip_{slug}", use_container_width=True):
                changed = _apply_status_changes(slug, sorted_records, selected_ids, "skip")
                if changed:
                    st.toast(f"Skipped {changed} job(s)")
                invalidate_dashboard_caches()
                st.rerun()
        with bulk_cols[3]:
            if st.button("Clear", key=f"beacon_bulk_clear_{slug}", use_container_width=True):
                # Streamlit dataframes don't expose a clear-selection API; surface a one-rerun
                # nudge via a key bump so the table re-mounts with no selection.
                st.session_state[f"beacon_jobs_table_{slug}_nonce"] = (
                    st.session_state.get(f"beacon_jobs_table_{slug}_nonce", 0) + 1
                )
                st.rerun()

    # ── Drawer trigger: first selected row opens the modal ──────────────
    open_cols = st.columns([1.4, 6.6, 1.4], gap="small")
    with open_cols[0]:
        open_disabled = not selected_ids
        if st.button(
            "Open selected →",
            key=f"beacon_jobs_open_selected_{slug}",
            disabled=open_disabled,
            use_container_width=True,
        ):
            st.session_state[_DRAWER_STATE_KEY] = selected_ids[0]
            st.rerun()
    with open_cols[2]:
        if st.button("Run pipeline", key=f"beacon_jobs_run_{slug}", use_container_width=True, type="primary"):
            _queue_pipeline_run(slug)

    # ── Keyboard hints (visual only) ────────────────────────────────────
    st.markdown(
        (
            "<div class='kbd-hints'>"
            "<span class='ki'><kbd>↵</kbd> open selected</span>"
            "<span class='ki'><kbd>A</kbd> apply</span>"
            "<span class='ki'><kbd>S</kbd> save</span>"
            "<span class='ki'><kbd>X</kbd> skip</span>"
            "<span class='ki'><kbd>⇧1</kbd>–<kbd>⇧5</kbd> rate</span>"
            "<span class='ki'><kbd>/</kbd> search</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    _maybe_open_drawer(slug)


_ACTIVITY_STAGES: list[tuple[Stage, str]] = [
    (Stage.DISCOVERING, "Discovery"),
    (Stage.FETCHING,    "Job board fetch"),
    (Stage.SCRAPING,    "Scraping"),
    (Stage.SCORING,     "Scoring & filtering"),
    (Stage.EMBEDDING,   "Embeddings"),
    (Stage.FINALIZING,  "Finalize"),
]

_ACTIVITY_SOURCE_ORDER: list[tuple[str, str]] = [
    ("greenhouse", "greenhouse"),
    ("lever",      "lever"),
    ("ashby",      "ashby"),
    ("workable",   "workable"),
    ("hn",         "hn"),
    ("himalayas",  "himalayas"),
]


@st.cache_data(ttl=10, show_spinner=False)
def _cached_history_runs(slug: str, limit: int = 200) -> list[dict[str, Any]]:
    return get_recent_runs(limit=limit, profile=slug)


def _cancel_worker_lockfile(slug: str) -> bool:
    """Mark the running worker as cancel-requested by removing its lockfile.

    The worker process is not killed; the dashboard simply stops treating it
    as live. Next render the run banner reflects idle state. Returns True if a
    lockfile was actually removed.
    """
    lf = _worker_lockfile(slug)
    try:
        if lf.exists():
            lf.unlink()
            return True
    except OSError:
        return False
    return False


def _render_activity_run_card(
    slug: str,
    runs: list[dict[str, Any]],
    records: list[dict[str, Any]],
    raw_config: dict[str, Any],
    *,
    worker_running: bool,
) -> None:
    """Card 1: latest/current run header — title, actions, stages | sources."""
    progress_data = _read_progress_json(slug) if worker_running else None
    tracker: ProgressTracker | None = None
    if progress_data:
        try:
            tracker = ProgressTracker.from_dict(progress_data)
        except Exception:
            tracker = None

    latest_run = runs[0] if runs else None
    if worker_running:
        run_id = latest_run.get("run_id") if latest_run else None
        run_title = f"Run #{run_id} · in progress" if run_id else "Run · in progress"
        if tracker:
            running_stage = next(
                (label for stage, label in _ACTIVITY_STAGES
                 if tracker.stages.get(stage) and tracker.stages[stage].status == StageStatus.RUNNING),
                "starting",
            )
            mins, secs = divmod(int(tracker.elapsed_time.total_seconds()), 60)
            sub = f"started {mins:02d}:{secs:02d} ago · {running_stage} stage"
        else:
            sub = "worker is starting up"
    elif latest_run:
        run_id = latest_run.get("run_id") or "?"
        status = str(latest_run.get("status") or "complete").lower()
        run_title = f"Run #{run_id} · {status}"
        rel = _format_relative(latest_run.get("finished_at") or latest_run.get("started_at"))
        rel_text = f"{rel} ago" if rel not in {"—", "just now"} else rel
        duration = _format_run_duration(latest_run.get("started_at"), latest_run.get("finished_at"))
        sub = f"finished {rel_text} · {duration}"
    else:
        run_title = "No runs yet"
        sub = "Run the pipeline to populate this panel."

    head_cols = st.columns([6.4, 1.3, 1.3], gap="small")
    with head_cols[0]:
        st.markdown(
            (
                "<div class='beacon-card-head' style='border:1px solid var(--line);"
                "border-bottom:0;border-radius:var(--r-md) var(--r-md) 0 0;"
                "background:var(--surface);padding:14px 18px'>"
                "<div>"
                f"<div class='beacon-card-title'>{html.escape(run_title)}</div>"
                f"<div class='beacon-card-sub'>{html.escape(sub)}</div>"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    with head_cols[1]:
        json_disabled = latest_run is None
        run_payload = json.dumps(latest_run or {}, indent=2, default=str)
        st.download_button(
            "View JSON",
            data=run_payload.encode("utf-8"),
            file_name=f"run_{(latest_run or {}).get('run_id', 'none')}.json",
            mime="application/json",
            key=f"activity_run_json_{slug}",
            use_container_width=True,
            disabled=json_disabled,
        )
    with head_cols[2]:
        cancel_disabled = not worker_running
        if st.button(
            "Cancel",
            key=f"activity_run_cancel_{slug}",
            use_container_width=True,
            disabled=cancel_disabled,
        ):
            if _cancel_worker_lockfile(slug):
                st.toast("Cancel requested. The worker will finish its current step then stop refreshing the lockfile.")
            else:
                st.toast("No live worker to cancel.")
            st.rerun()

    stages_html_parts: list[str] = []
    for stage, label in _ACTIVITY_STAGES:
        prog = tracker.stages.get(stage) if tracker else None
        status = prog.status if prog else None
        cls = "queue"
        if status == StageStatus.COMPLETE:
            cls = "done"
        elif status == StageStatus.RUNNING:
            cls = "run"
        elif status == StageStatus.FAILED:
            cls = "fail"
        count_html = ""
        if prog and prog.metrics:
            metric_str = " · ".join(f"{k}: {v}" for k, v in list(prog.metrics.items())[:2])
            count_html = f"<span class='stg-count'>{html.escape(metric_str)}</span>"
        stages_html_parts.append(
            "<div class='activity-stage " + cls + "'>"
            "<span class='stg-icon'></span>"
            f"<span class='stg-name'>{html.escape(label)}</span>"
            f"{count_html}"
            "</div>"
        )
    stages_html = "<div class='activity-stage-list'>" + "".join(stages_html_parts) + "</div>"

    enabled_sources = _enabled_sources(raw_config)
    source_counts: dict[str, int] = {}
    for record in records:
        key = str(record.get("source") or "").lower()
        if key:
            source_counts[key] = source_counts.get(key, 0) + 1

    if worker_running and latest_run and latest_run.get("jobs_scraped"):
        max_count = max(int(latest_run.get("jobs_scraped") or 0), 1)
    else:
        max_count = max(max(source_counts.values(), default=0), 1)

    rendered_sources: list[str] = []
    for source_key, label_key in _ACTIVITY_SOURCE_ORDER:
        flag_key = "hackernews" if source_key == "hn" else source_key
        on = bool(enabled_sources.get(flag_key, False))
        count = source_counts.get(label_key, 0)
        pct = max(0, min(100, int(round(100 * count / max_count)))) if on else 0
        bar_color = "var(--signal)" if on else "var(--muted)"
        rendered_sources.append(
            "<div class='activity-source-row'>"
            f"<span class='nm'>{html.escape(label_key)}</span>"
            "<div class='progress'>"
            f"<div style='width:{pct}%;background:{bar_color}'></div>"
            "</div>"
            f"<span class='ct'>{count}</span>"
            "</div>"
        )
    sources_html = "<div class='activity-source-list'>" + "".join(rendered_sources) + "</div>"

    st.markdown(
        (
            "<div class='beacon-card' style='border-radius:0 0 var(--r-md) var(--r-md);"
            "border-top:0;margin-top:-1px'>"
            "<div class='activity-run-grid'>"
            "<div>"
            "<div class='col-eyebrow'>Stages</div>"
            f"{stages_html}"
            "</div>"
            "<div>"
            "<div class='col-eyebrow'>Sources</div>"
            f"{sources_html}"
            "</div>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _build_activity_feed_events(
    runs: list[dict[str, Any]],
    *,
    worker_running: bool,
    tracker: ProgressTracker | None,
) -> list[dict[str, Any]]:
    """Synthesize a reverse-chronological feed from runs + live tracker state."""
    events: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)

    if worker_running and tracker:
        for stage, label in _ACTIVITY_STAGES:
            prog = tracker.stages.get(stage)
            if not prog:
                continue
            if prog.status == StageStatus.RUNNING:
                metric_str = ""
                if prog.metrics:
                    metric_str = " · " + ", ".join(
                        f"{k}: {v}" for k, v in list(prog.metrics.items())[:2]
                    )
                events.append({
                    "ts_dt": _seconds_to_dt(prog.started_at) or now,
                    "kind": "info",
                    "body": f"<b>{html.escape(label)}</b> stage in progress{html.escape(metric_str)}",
                    "right": "running",
                })
            elif prog.status == StageStatus.COMPLETE and prog.completed_at:
                duration_str = _seconds_duration(prog.duration)
                events.append({
                    "ts_dt": _seconds_to_dt(prog.completed_at) or now,
                    "kind": "success",
                    "body": f"<b>{html.escape(label)}</b> stage finished",
                    "right": duration_str,
                })

    for run in runs[:14]:
        run_id = run.get("run_id") or "?"
        status = str(run.get("status") or "").lower()
        finished_at = run.get("finished_at")
        started_at = run.get("started_at")
        duration_str = _format_run_duration(started_at, finished_at)
        scraped = int(run.get("jobs_scraped") or 0)
        scored = int(run.get("jobs_scored") or 0)
        saved = int(run.get("jobs_saved") or 0)
        errors = run.get("errors") or []

        if started_at:
            events.append({
                "ts_dt": _parse_iso(started_at),
                "kind": "info",
                "body": f"Run <b>#{html.escape(str(run_id))}</b> started · source <b>{html.escape(str(run.get('source') or 'pipeline'))}</b>",
                "right": "—",
            })

        for err in errors[:3]:
            err_text = str(err).strip().splitlines()[0][:160]
            events.append({
                "ts_dt": _parse_iso(finished_at or started_at),
                "kind": "danger",
                "body": f"Run #{html.escape(str(run_id))} error: {html.escape(err_text)}",
                "right": "error",
            })

        if finished_at and status not in {"running", ""}:
            if status in {"failed", "error", "crashed"}:
                kind = "danger"
                body = f"Run <b>#{html.escape(str(run_id))}</b> failed"
            elif errors:
                kind = "warn"
                body = (
                    f"Run <b>#{html.escape(str(run_id))}</b> finished with "
                    f"<b>{len(errors)} error{'s' if len(errors) != 1 else ''}</b> · "
                    f"scraped {scraped}, scored {scored}, saved {saved}"
                )
            else:
                kind = "success"
                body = (
                    f"Run <b>#{html.escape(str(run_id))}</b> complete · "
                    f"scraped <b>{scraped}</b>, scored <b>{scored}</b>, saved <b>{saved}</b>"
                )
            events.append({
                "ts_dt": _parse_iso(finished_at),
                "kind": kind,
                "body": body,
                "right": duration_str,
            })

    events = [e for e in events if e["ts_dt"] is not None]
    events.sort(key=lambda e: e["ts_dt"], reverse=True)
    return events


def _seconds_to_dt(value: float | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(value, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _seconds_duration(value: float | None) -> str:
    if value is None:
        return "—"
    total = max(int(value), 0)
    minutes, seconds = divmod(total, 60)
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _render_activity_feed_card(
    slug: str,
    runs: list[dict[str, Any]],
    *,
    worker_running: bool,
) -> None:
    """Card 2: activity feed (most recent first)."""
    progress_data = _read_progress_json(slug) if worker_running else None
    tracker: ProgressTracker | None = None
    if progress_data:
        try:
            tracker = ProgressTracker.from_dict(progress_data)
        except Exception:
            tracker = None

    events = _build_activity_feed_events(runs, worker_running=worker_running, tracker=tracker)

    head_cols = st.columns([6.4, 2.6], gap="small")
    with head_cols[0]:
        st.markdown(
            (
                "<div class='beacon-card-head' style='border:1px solid var(--line);"
                "border-bottom:0;border-radius:var(--r-md) var(--r-md) 0 0;"
                "background:var(--surface);padding:14px 18px'>"
                "<div>"
                "<div class='beacon-card-title'>Activity feed</div>"
                "<div class='beacon-card-sub'>Most recent first</div>"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    with head_cols[1]:
        if events:
            log_lines: list[str] = []
            for ev in events:
                ts_iso = ev["ts_dt"].astimezone(timezone.utc).isoformat(timespec="seconds")
                clean_body = re.sub(r"<[^>]+>", "", ev["body"])
                log_lines.append(f"{ts_iso}  [{ev['kind']:<7}]  {clean_body}  ({ev['right']})")
            log_blob = "\n".join(log_lines).encode("utf-8")
        else:
            log_blob = b"# No activity yet."
        st.download_button(
            "Export log",
            data=log_blob,
            file_name=f"activity_{slug}.log",
            mime="text/plain",
            key=f"activity_feed_export_{slug}",
            use_container_width=True,
        )

    if not events:
        st.markdown(
            (
                "<div class='beacon-card' style='border-radius:0 0 var(--r-md) var(--r-md);"
                "border-top:0;margin-top:-1px;padding:24px'>"
                "<div class='beacon-empty'>"
                "<div class='ic'>·</div>"
                "<div class='h'>No activity yet.</div>"
                "<div class='s'>Run the pipeline once to populate the feed.</div>"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        return

    rows_html: list[str] = []
    for ev in events[:50]:
        ts_label = _format_relative(ev["ts_dt"].astimezone(timezone.utc).isoformat())
        rows_html.append(
            f"<div class='feed-row {html.escape(ev['kind'])}'>"
            f"<span class='ts'>{html.escape(ts_label)}</span>"
            "<span class='dot'></span>"
            f"<div class='body'>{ev['body']}</div>"
            f"<span class='right'>{html.escape(ev['right'])}</span>"
            "</div>"
        )
    st.markdown(
        (
            "<div class='beacon-card' style='border-radius:0 0 var(--r-md) var(--r-md);"
            "border-top:0;margin-top:-1px'>"
            "<div class='feed'>"
            + "".join(rows_html)
            + "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_run_history_chart(slug: str) -> None:
    """Card 3: 14-day vertical bar chart of new high-fit per day."""
    history = _cached_history_runs(slug)

    today_utc = datetime.now(timezone.utc).date()
    by_day: dict[Any, list[dict[str, Any]]] = {}
    for run in history:
        finished = run.get("finished_at") or run.get("started_at")
        dt = _parse_iso(finished)
        if dt is None:
            continue
        day = dt.astimezone(timezone.utc).date()
        by_day.setdefault(day, []).append(run)

    days: list[Any] = []
    counts: list[int] = []
    durations: list[int] = []
    for offset in range(13, -1, -1):
        day = today_utc - timedelta(days=offset)
        runs_for_day = by_day.get(day, [])
        day_high = sum(int(r.get("jobs_saved") or 0) for r in runs_for_day)
        days.append(day)
        counts.append(day_high)
        for r in runs_for_day:
            s = _parse_iso(r.get("started_at"))
            f = _parse_iso(r.get("finished_at"))
            if s and f:
                durations.append(max(int((f - s).total_seconds()), 0))

    total_runs_14d = sum(len(by_day.get(today_utc - timedelta(days=o), [])) for o in range(14))
    if durations:
        avg_secs = sum(durations) // len(durations)
        m, s = divmod(avg_secs, 60)
        avg_label = f"{m}m {s:02d}s"
    else:
        avg_label = "—"

    max_count = max(counts) if max(counts, default=0) > 0 else 1
    bars_html: list[str] = []
    for idx, (day, count) in enumerate(zip(days, counts)):
        is_today = idx == len(counts) - 1
        height_pct = max(5, int(round(100 * count / max_count))) if count else 5
        cls = "bar today" if is_today else "bar"
        title_attr = f"{day.isoformat()} · {count} new saved jobs"
        bars_html.append(
            f"<div class='{cls}' style='height:{height_pct}%' title='{html.escape(title_attr)}'></div>"
        )

    start_label = days[0].strftime("%b %d") if days else ""
    end_label = "today"

    st.markdown(
        (
            "<div class='beacon-card-head' style='border:1px solid var(--line);"
            "border-bottom:0;border-radius:var(--r-md) var(--r-md) 0 0;"
            "background:var(--surface);padding:14px 18px'>"
            "<div>"
            "<div class='beacon-card-title'>Run history · last 14 days</div>"
            "<div class='beacon-card-sub'>One bar per day · taller = more new saved jobs</div>"
            "</div>"
            f"<span style='font-family:var(--font-mono);font-size:11px;color:var(--muted)'>"
            f"{total_runs_14d} runs · avg {html.escape(avg_label)}"
            "</span>"
            "</div>"
            "<div class='beacon-card' style='border-radius:0 0 var(--r-md) var(--r-md);"
            "border-top:0;margin-top:-1px'>"
            "<div class='history-chart'>"
            + "".join(bars_html)
            + "</div>"
            "<div class='history-axis'>"
            f"<span>{html.escape(start_label)}</span>"
            f"<span>{html.escape(end_label)}</span>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_activity_tab(
    slug: str,
    runs: list[dict[str, Any]],
    metrics: dict[str, Any],
    raw_config: dict[str, Any],
    config: dict[str, Any],
) -> None:
    worker_running = _worker_is_running(slug)
    records = _cached_fetch_job_summaries(slug)

    head_cols = st.columns([6.4, 2.6], gap="medium")
    with head_cols[0]:
        _render_beacon_page_header(
            "Activity · Pipeline runs",
            "Everything the agent did, in order.",
            "A timeline of every fetch, score, and decision. Click any run to inspect its inputs and outputs.",
        )
    with head_cols[1]:
        action_cols = st.columns(2, gap="small")
        with action_cols[0]:
            if st.button(
                "Filter",
                key=f"activity_filter_{slug}",
                use_container_width=True,
                help="Filtering arrives in a follow-up; the feed below already shows everything in chronological order.",
            ):
                st.toast("Filter UI is coming soon — the feed below is already chronological.")
        with action_cols[1]:
            if st.button(
                "Run now ⌘R",
                key=f"activity_run_now_{slug}",
                use_container_width=True,
                type="primary",
                disabled=worker_running,
            ):
                _queue_pipeline_run(slug)

    _render_activity_run_card(slug, runs, records, raw_config, worker_running=worker_running)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    _render_activity_feed_card(slug, runs, worker_running=worker_running)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    _render_run_history_chart(slug)

    disq_count = metrics.get("disqualified_count", 0)
    scrape_rej_count = metrics.get("scrape_rejected_count", 0)
    if disq_count or scrape_rej_count:
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        with st.expander("Filtered out · breakdown", expanded=False):
            if disq_count:
                st.caption(
                    f"{disq_count} job{'s' if disq_count != 1 else ''} hidden by hard-no scoring rules. "
                    "Stored in the database but excluded from the review queue."
                )
                by_reason = metrics.get("disqualified_by_reason", {})
                if by_reason:
                    rows = sorted(by_reason.items(), key=lambda kv: -kv[1])
                    _render_summary_list([(reason, f"{count}×") for reason, count in rows])
            if scrape_rej_count:
                st.caption(
                    f"{scrape_rej_count} job{'s' if scrape_rej_count != 1 else ''} rejected pre-LLM by scrape filters."
                )
                rej_by_reason = metrics.get("scrape_rejected_by_reason", {})
                if rej_by_reason:
                    rows = sorted(rej_by_reason.items(), key=lambda kv: -kv[1])
                    _render_summary_list([(reason, f"{count}×") for reason, count in rows])

    with st.expander("Debug · cache & evaluation", expanded=False):
        st.caption("Discovery cache snapshots (read-only) and the evaluation tooling that used to live on this tab.")
        gh_slugs  = load_discovered_slugs(ats="greenhouse", profile=slug)
        lv_slugs  = load_discovered_slugs(ats="lever",      profile=slug)
        ash_slugs = load_discovered_slugs(ats="ashby",      profile=slug)
        wl_slugs  = load_discovered_slugs(ats="workable",   profile=slug)
        _render_summary_list(
            [
                ("Greenhouse cache", len(gh_slugs)),
                ("Lever cache",      len(lv_slugs)),
                ("Ashby cache",      len(ash_slugs)),
                ("Workable cache",   len(wl_slugs)),
            ]
        )
        _render_evaluation_card(slug, config)


def _render_evaluation_card(slug: str, config: dict[str, Any]) -> None:
    labels_path = _eval_labels_path(slug)
    labels = load_eval_labels(labels_path)
    last_result = load_last_eval_result(slug)
    counts = rating_counts(slug)

    with panel("Evaluation", subtitle="Measure whether semantic ranking is improving for this profile"):
        st.caption(
            "Rate jobs as Great / Good / Okay / Not relevant / Skip from any job card. "
            "Those ratings ARE the eval ground truth — the more you rate, the more reliable the metrics below."
        )
        st.write(f"Labels file: {'available' if labels_path.exists() else 'not created yet'}")
        st.write(f"Labeled jobs: {len(labels)}")
        if any(counts.values()):
            distribution = " · ".join(
                f"{label.replace('_match', '').replace('_', ' ').title()}: {count}"
                for label, count in counts.items()
                if count
            )
            st.caption(f"Rating distribution — {distribution}")

        action_cols = st.columns(2, gap="small")
        if action_cols[0].button("Export eval template", key=f"export_eval_template_{slug}", use_container_width=True):
            try:
                exported = export_eval_template(
                    slug,
                    config,
                    labels_path,
                    use_reranker=reranking_enabled(config),
                )
                _set_notice(slug, "success", f"Exported {len(exported)} eval labels to {labels_path.name}.")
                invalidate_dashboard_caches()
                st.rerun()
            except Exception as exc:
                callout("error", "Eval template export failed", str(exc))

        run_disabled = not labels_path.exists() or not labels
        if action_cols[1].button("Run evaluation", key=f"run_eval_{slug}", use_container_width=True, disabled=run_disabled):
            try:
                evaluate_profile(
                    slug,
                    config,
                    labels_path,
                    use_reranker=reranking_enabled(config),
                )
                _set_notice(slug, "success", "Evaluation complete. Metrics refreshed.")
                invalidate_dashboard_caches()
                st.rerun()
            except Exception as exc:
                callout("error", "Evaluation failed", str(exc))

        if last_result is None:
            st.caption("No evaluation has been run for this profile yet.")
            return

        stat_row(
            [
                ("P@5", f"{last_result.precision_at_5:.2f}", "Top-5 precision"),
                ("P@10", f"{last_result.precision_at_10:.2f}", "Top-10 precision"),
                ("Recall@10", f"{last_result.recall_at_10:.2f}", "Relevant labeled jobs found in top 10"),
                ("NDCG@10", f"{last_result.ndcg_at_10:.2f}", "Graded ranking quality"),
                ("MRR", f"{last_result.mrr:.2f}", "Reciprocal rank of first relevant job"),
                ("Coverage", f"{last_result.coverage:.2f}", "Labeled jobs present anywhere in the candidate set"),
            ],
            columns_count=2,
        )
        if last_result.notes:
            st.caption(last_result.notes)


_PROFILE_DRAFT_STATE_KEY = "_beacon_profile_draft"
_STRUCTURED_PROFILE_FILENAME = "structured_profile.json"


def _structured_profile_cache_path(slug: str) -> Path:
    return PROFILES_DIR / slug / _STRUCTURED_PROFILE_FILENAME


def _read_structured_profile_cache(slug: str) -> dict[str, Any] | None:
    path = _structured_profile_cache_path(slug)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_structured_profile_cache(slug: str, profile: dict[str, Any]) -> None:
    path = _structured_profile_cache_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")


def _profile_draft_key(slug: str) -> str:
    return f"{_PROFILE_DRAFT_STATE_KEY}_{slug}"


def _ensure_profile_draft(slug: str, raw_config: dict[str, Any]) -> dict[str, Any]:
    """Return the live edit-buffer for the Profile tab, seeded from raw_config."""
    key = _profile_draft_key(slug)
    if key not in st.session_state:
        profile_cfg = raw_config.get("profile", {}) or {}
        prefs = raw_config.get("preferences", {}) or {}
        location = prefs.get("location", {}) or {}
        compensation = prefs.get("compensation", {}) or {}
        st.session_state[key] = {
            "name":          str(profile_cfg.get("name") or ""),
            "email":         str(profile_cfg.get("email") or ""),
            "location":      str(profile_cfg.get("location") or ""),
            "timezone":      str(profile_cfg.get("timezone") or ""),
            "auth_status":   str(profile_cfg.get("auth_status") or ""),
            "titles":        list(prefs.get("titles") or []),
            "skills":        list(prefs.get("desired_skills") or []),
            "remote_ok":     bool(location.get("remote_ok", True)),
            "preferred_locations": list(location.get("preferred_locations") or []),
            "min_salary":    int(compensation.get("min_salary") or 0),
            "sponsorship":   str(prefs.get("sponsorship") or ""),
            "excluded":      list(prefs.get("excluded_industries") or []),
        }
    return st.session_state[key]


def _reset_profile_draft(slug: str) -> None:
    st.session_state.pop(_profile_draft_key(slug), None)


def _save_profile_draft(slug: str, raw_config: dict[str, Any], draft: dict[str, Any]) -> None:
    """Merge the draft back into raw_config and persist."""
    updated = copy.deepcopy(raw_config)
    profile_cfg = updated.setdefault("profile", {})
    profile_cfg["name"]        = draft["name"].strip() or profile_cfg.get("name", "")
    profile_cfg["email"]       = draft["email"].strip()
    profile_cfg["location"]    = draft["location"].strip()
    profile_cfg["timezone"]    = draft["timezone"].strip()
    profile_cfg["auth_status"] = draft["auth_status"].strip()
    prefs = updated.setdefault("preferences", {})
    prefs["titles"]         = list(draft["titles"])
    prefs["desired_skills"] = list(draft["skills"])
    prefs["sponsorship"]    = draft["sponsorship"].strip()
    prefs["excluded_industries"] = list(draft["excluded"])
    location = prefs.setdefault("location", {})
    location["remote_ok"] = bool(draft["remote_ok"])
    location["preferred_locations"] = list(draft["preferred_locations"])
    compensation = prefs.setdefault("compensation", {})
    if int(draft["min_salary"] or 0) > 0:
        compensation["min_salary"] = int(draft["min_salary"])
    elif "min_salary" in compensation:
        compensation["min_salary"] = 0
    _write_profile_config(slug, updated)


def _reimport_resume(slug: str, raw_config: dict[str, Any]) -> tuple[bool, str]:
    """Re-extract resume PDF text and rebuild the structured profile.

    Writes the structured profile to profiles/<slug>/structured_profile.json so
    the Resume card can display the parsed metadata next render. Returns
    (success, message).
    """
    from candidate_profile import build_structured_profile
    from config import extract_text_from_pdf
    from scorer import get_llm_client, get_instructor_client

    profile_cfg = raw_config.get("profile", {}) or {}
    resume_value = profile_cfg.get("resume_file")
    if not resume_value:
        return False, "No resume_file configured for this profile."

    profile_dir = PROFILES_DIR / slug
    try:
        resolved = _resolve_resume_path(str(resume_value), profile_dir)
    except FileNotFoundError as exc:
        return False, str(exc)

    try:
        text = extract_text_from_pdf(str(resolved))
    except Exception as exc:
        return False, f"PDF extraction failed: {exc}"

    config = copy.deepcopy(raw_config)
    config["_active_profile"] = slug
    config.setdefault("profile", {})["resume"] = text

    missing = _check_api_key(config)
    if missing:
        return False, f"Set {missing} before re-importing — the structured profile needs an LLM call."

    try:
        llm_call = get_llm_client(config)
    except SystemExit:
        return False, "LLM client could not be initialized; check API keys."
    instructor_client = None
    instructor_model = None
    instructor_temperature = None
    try:
        instructor_client, instructor_model, instructor_temperature = get_instructor_client(config)
    except Exception:
        instructor_client = None

    try:
        if instructor_client and instructor_model:
            structured = build_structured_profile(
                config,
                llm_call,
                instructor_client,
                model=instructor_model,
                temperature=instructor_temperature,
            )
        else:
            structured = build_structured_profile(config, llm_call)
    except Exception as exc:
        return False, f"Structured profile build failed: {exc}"

    _write_structured_profile_cache(slug, structured)
    return True, f"Re-imported resume: {len(structured.get('core_skills', []))} skills, {structured.get('yoe', '?')}yr."


def _render_profile_identity_card(slug: str, draft: dict[str, Any]) -> None:
    """Identity & contact card — kv rows + an inline editor toggle."""
    rows = [
        ("Name",        draft.get("name") or "Not set"),
        ("Email",       draft.get("email") or "Not set"),
        ("Location",    draft.get("location") or "Not set"),
        ("Timezone",    draft.get("timezone") or "Not set"),
        ("Auth status", draft.get("auth_status") or "Not set"),
    ]
    head_cols = st.columns([6.4, 1.6], gap="small")
    with head_cols[0]:
        st.markdown(
            (
                "<div class='beacon-card-head' style='border:1px solid var(--line);"
                "border-bottom:0;border-radius:var(--r-md) var(--r-md) 0 0;"
                "background:var(--surface);padding:14px 18px'>"
                "<div>"
                "<div class='beacon-card-title'>Identity & contact</div>"
                "<div class='beacon-card-sub'>How the agent introduces you</div>"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    edit_state_key = f"profile_identity_edit_{slug}"
    with head_cols[1]:
        editing = st.toggle(
            "Edit",
            key=edit_state_key,
            value=False,
            label_visibility="collapsed",
        )

    rows_html = "".join(
        f"<div class='kv'><span class='k'>{html.escape(label)}</span>"
        f"<span class='v'>{html.escape(str(value))}</span></div>"
        for label, value in rows
    )
    st.markdown(
        (
            "<div class='beacon-card' style='border-radius:0 0 var(--r-md) var(--r-md);"
            "border-top:0;margin-top:-1px;padding:12px 22px 16px'>"
            f"{rows_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    if editing:
        edit_cols = st.columns(2, gap="small")
        with edit_cols[0]:
            draft["name"] = st.text_input(
                "Name",
                value=draft.get("name", ""),
                key=f"profile_identity_name_{slug}",
            )
            draft["location"] = st.text_input(
                "Location",
                value=draft.get("location", ""),
                placeholder="San Francisco, CA",
                key=f"profile_identity_location_{slug}",
            )
            draft["auth_status"] = st.text_input(
                "Auth status",
                value=draft.get("auth_status", ""),
                placeholder="US citizen — no sponsorship needed",
                key=f"profile_identity_auth_{slug}",
            )
        with edit_cols[1]:
            draft["email"] = st.text_input(
                "Email",
                value=draft.get("email", ""),
                placeholder="you@example.com",
                key=f"profile_identity_email_{slug}",
            )
            draft["timezone"] = st.text_input(
                "Timezone",
                value=draft.get("timezone", ""),
                placeholder="PT (UTC−7)",
                key=f"profile_identity_tz_{slug}",
            )


def _render_profile_resume_card(
    slug: str,
    raw_config: dict[str, Any],
    profile_cfg: dict[str, Any],
    structured: dict[str, Any] | None,
) -> None:
    resume_value = profile_cfg.get("resume_file")
    resume_pdf_bytes: bytes | None = None
    resume_file_name = "resume.pdf"
    resume_size_kb = 0
    if resume_value:
        name_parts = [p for p in str(profile_cfg.get("name", "")).strip().split() if p]
        if len(name_parts) >= 2:
            resume_file_name = f"{name_parts[-1]}_{name_parts[0]}_resume.pdf"
        elif name_parts:
            resume_file_name = f"{name_parts[0]}_resume.pdf"
        try:
            profile_dir = PROFILES_DIR / slug
            resolved = _resolve_resume_path(str(resume_value), profile_dir)
            resume_pdf_bytes = resolved.read_bytes()
            resume_size_kb = max(1, len(resume_pdf_bytes) // 1024)
        except (FileNotFoundError, OSError):
            resume_pdf_bytes = None

    head_cols = st.columns([6.4, 1.6], gap="small")
    with head_cols[0]:
        st.markdown(
            (
                "<div class='beacon-card-head' style='border:1px solid var(--line);"
                "border-bottom:0;border-radius:var(--r-md) var(--r-md) 0 0;"
                "background:var(--surface);padding:14px 18px'>"
                "<div>"
                "<div class='beacon-card-title'>Resume</div>"
                "<div class='beacon-card-sub'>Source PDF + parsed metadata</div>"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    with head_cols[1]:
        if st.button(
            "Replace",
            key=f"profile_resume_replace_{slug}",
            use_container_width=True,
        ):
            st.session_state[f"profile_resume_replace_open_{slug}"] = True

    meta_pieces: list[str] = []
    if resume_pdf_bytes:
        meta_pieces.append(f"{resume_size_kb} KB")
    if structured:
        skills = structured.get("core_skills") or []
        if skills:
            meta_pieces.append(f"{len(skills)} skills extracted")

    meta_label = " · ".join(meta_pieces) if meta_pieces else "No metadata yet"

    if structured:
        yoe = structured.get("yoe")
        if yoe is None or yoe == "":
            yoe_label = "Not extracted"
        else:
            yoe_label = f"{yoe} years"
        past_roles = structured.get("past_roles") or []
        last_role = past_roles[0] if past_roles else (structured.get("current_title") or "Not extracted")
        education = structured.get("education") or "Not extracted"
    else:
        yoe_label = "Click Re-import resume above"
        last_role = "Click Re-import resume above"
        education = "Click Re-import resume above"

    rows = [
        ("Years exp", yoe_label),
        ("Last role", last_role),
        ("Education", education),
    ]
    rows_html = "".join(
        f"<div class='kv'><span class='k'>{html.escape(label)}</span>"
        f"<span class='v'>{html.escape(str(value))}</span></div>"
        for label, value in rows
    )

    st.markdown(
        (
            "<div class='beacon-card' style='border-radius:0 0 var(--r-md) var(--r-md);"
            "border-top:0;margin-top:-1px;padding:14px 22px 16px'>"
            "<div class='resume-row'>"
            "<div class='resume-thumb'>PDF</div>"
            "<div class='grow'>"
            f"<div class='name'>{html.escape(resume_file_name)}</div>"
            f"<div class='meta'>{html.escape(meta_label)}</div>"
            "</div>"
            "</div>"
            "<div style='height:14px'></div>"
            f"{rows_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    if resume_pdf_bytes:
        action_cols = st.columns([1, 1, 4], gap="small")
        with action_cols[0]:
            if st.button("View PDF", key=f"profile_resume_view_{slug}", use_container_width=True):
                _open_resume_preview_dialog(resume_file_name, resume_pdf_bytes)
        with action_cols[1]:
            st.download_button(
                "Download",
                data=resume_pdf_bytes,
                file_name=resume_file_name,
                mime="application/pdf",
                key=f"profile_resume_download_{slug}",
                use_container_width=True,
            )

    if st.session_state.get(f"profile_resume_replace_open_{slug}"):
        upload = st.file_uploader(
            "Upload a new resume PDF",
            type=["pdf"],
            key=f"profile_resume_upload_{slug}",
            accept_multiple_files=False,
        )
        cancel_cols = st.columns([1, 1, 4], gap="small")
        with cancel_cols[0]:
            if st.button("Cancel", key=f"profile_resume_replace_cancel_{slug}", use_container_width=True):
                st.session_state[f"profile_resume_replace_open_{slug}"] = False
                st.rerun()
        if upload is not None:
            target = PROFILES_DIR / slug / "resume.pdf"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(upload.read())
            updated = copy.deepcopy(raw_config)
            updated.setdefault("profile", {})["resume_file"] = "resume.pdf"
            _write_profile_config(slug, updated)
            st.session_state[f"profile_resume_replace_open_{slug}"] = False
            st.toast("Resume replaced. Click Re-import resume to refresh parsed metadata.")
            st.rerun()


def _render_profile_skills_card(
    slug: str,
    draft: dict[str, Any],
) -> None:
    skills = list(draft.get("skills") or [])
    head_cols = st.columns([6.4, 1.6], gap="small")
    with head_cols[0]:
        st.markdown(
            (
                "<div class='beacon-card-head' style='border:1px solid var(--line);"
                "border-bottom:0;border-radius:var(--r-md) var(--r-md) 0 0;"
                "background:var(--surface);padding:14px 18px'>"
                "<div>"
                "<div class='beacon-card-title'>Skills</div>"
                "<div class='beacon-card-sub'>Used by the scorer for stack-match scoring</div>"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    with head_cols[1]:
        st.markdown(
            "<div style='font-family:var(--font-mono);font-size:11px;color:var(--muted);text-align:right;padding-top:10px'>"
            f"{len(skills)} total"
            "</div>",
            unsafe_allow_html=True,
        )

    pills_html = "".join(
        f"<span class='pill on'>{html.escape(s)}</span>" for s in skills
    ) if skills else "<span class='pill'>No skills yet</span>"
    st.markdown(
        (
            "<div class='beacon-card' style='border-radius:0 0 var(--r-md) var(--r-md);"
            "border-top:0;margin-top:-1px;padding:16px 22px'>"
            f"<div class='pill-grid'>{pills_html}</div>"
            "<p style='font-size:12.5px;color:var(--muted);margin-top:14px;line-height:1.55'>"
            "Edit below — these are the skills the scorer uses for the <b>stack_match</b> dimension. "
            "Adding concrete tools (e.g. <i>FastAPI</i>, <i>Postgres</i>) gives the agent better signal "
            "than soft phrases."
            "</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    edit_cols = st.columns([3, 1], gap="small")
    with edit_cols[0]:
        new_skills = st.text_area(
            "Skills (one per line)",
            value="\n".join(skills),
            key=f"profile_skills_text_{slug}",
            label_visibility="collapsed",
            height=140,
            placeholder="Python\nFastAPI\nPostgreSQL\n...",
        )
    with edit_cols[1]:
        if st.button("Apply skills", key=f"profile_skills_apply_{slug}", use_container_width=True):
            draft["skills"] = _lines_to_list(new_skills)
            st.toast(f"{len(draft['skills'])} skill(s) staged. Click Save to persist.")
            st.rerun()


def _render_profile_target_roles_card(slug: str, draft: dict[str, Any]) -> None:
    titles = list(draft.get("titles") or [])
    pills_html = "".join(
        f"<span class='pill on'>{html.escape(t)}</span>" for t in titles
    ) if titles else "<span class='pill'>No target roles yet</span>"
    st.markdown(
        (
            "<div class='beacon-card-head' style='border:1px solid var(--line);"
            "border-bottom:0;border-radius:var(--r-md) var(--r-md) 0 0;"
            "background:var(--surface);padding:14px 18px'>"
            "<div>"
            "<div class='beacon-card-title'>Target roles</div>"
            "<div class='beacon-card-sub'>Job titles the agent prioritizes</div>"
            "</div>"
            "</div>"
            "<div class='beacon-card' style='border-radius:0 0 var(--r-md) var(--r-md);"
            "border-top:0;margin-top:-1px;padding:16px 22px'>"
            f"<div class='pill-grid'>{pills_html}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    edit_cols = st.columns([3, 1], gap="small")
    with edit_cols[0]:
        new_titles = st.text_area(
            "Target roles (one per line)",
            value="\n".join(titles),
            key=f"profile_titles_text_{slug}",
            label_visibility="collapsed",
            height=110,
            placeholder="Backend Engineer\nSoftware Engineer\n...",
        )
    with edit_cols[1]:
        if st.button("Apply roles", key=f"profile_titles_apply_{slug}", use_container_width=True):
            draft["titles"] = _lines_to_list(new_titles)
            st.toast(f"{len(draft['titles'])} role(s) staged. Click Save to persist.")
            st.rerun()


def _render_profile_hard_prefs_card(slug: str, draft: dict[str, Any]) -> None:
    salary = int(draft.get("min_salary") or 0)
    salary_label = f"${salary:,} base" if salary > 0 else "Not set"
    locations = draft.get("preferred_locations") or []
    geo_label = ("Remote (US)" if draft.get("remote_ok") else "Onsite only")
    if locations:
        geo_label = f"{geo_label} · " + ", ".join(locations)
    excluded = draft.get("excluded") or []
    excluded_label = ", ".join(excluded) if excluded else "None"
    sponsorship_label = draft.get("sponsorship") or "Not specified"

    rows = [
        ("Min comp",    salary_label),
        ("Geo",         geo_label),
        ("Sponsorship", sponsorship_label),
        ("Excluded",    excluded_label),
    ]
    rows_html = "".join(
        f"<div class='kv'><span class='k'>{html.escape(label)}</span>"
        f"<span class='v'>{html.escape(str(value))}</span></div>"
        for label, value in rows
    )
    st.markdown(
        (
            "<div class='beacon-card-head' style='border:1px solid var(--line);"
            "border-bottom:0;border-radius:var(--r-md) var(--r-md) 0 0;"
            "background:var(--surface);padding:14px 18px'>"
            "<div>"
            "<div class='beacon-card-title'>Hard preferences</div>"
            "<div class='beacon-card-sub'>Disqualifiers — soft scoring uses these too</div>"
            "</div>"
            "</div>"
            "<div class='beacon-card' style='border-radius:0 0 var(--r-md) var(--r-md);"
            "border-top:0;margin-top:-1px;padding:14px 22px 16px'>"
            f"{rows_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    with st.expander("Edit hard preferences", expanded=False):
        prefs_cols = st.columns(2, gap="medium")
        with prefs_cols[0]:
            draft["min_salary"] = int(st.number_input(
                "Minimum salary (USD/year)",
                value=int(draft.get("min_salary") or 0),
                step=5000,
                min_value=0,
                key=f"profile_hard_salary_{slug}",
            ) or 0)
            draft["sponsorship"] = st.text_input(
                "Sponsorship",
                value=draft.get("sponsorship") or "",
                placeholder="Not required / required for H-1B / open to all",
                key=f"profile_hard_sponsorship_{slug}",
            )
        with prefs_cols[1]:
            draft["remote_ok"] = st.checkbox(
                "Remote OK",
                value=bool(draft.get("remote_ok", True)),
                key=f"profile_hard_remote_{slug}",
            )
            existing_locs = draft.get("preferred_locations") or []
            location_options = sorted({*LOCATION_PICKER_OPTIONS, *existing_locs})
            draft["preferred_locations"] = list(st.multiselect(
                "Preferred locations",
                options=location_options,
                default=[loc for loc in existing_locs if loc in location_options],
                key=f"profile_hard_locations_{slug}",
            ))
        excluded_text = st.text_area(
            "Excluded industries (one per line)",
            value="\n".join(draft.get("excluded") or []),
            placeholder="Defense\nCrypto-native\nAdtech",
            key=f"profile_hard_excluded_{slug}",
            height=90,
        )
        draft["excluded"] = _lines_to_list(excluded_text)


def _render_profile_tab(slug: str, config: dict[str, Any], raw_config: dict[str, Any]) -> None:
    profile_cfg = raw_config.get("profile", {}) or {}
    draft = _ensure_profile_draft(slug, raw_config)
    structured = _read_structured_profile_cache(slug)
    name_label = profile_cfg.get("name") or slug.replace("_", " ").title()

    head_cols = st.columns([6.4, 2.6], gap="medium")
    with head_cols[0]:
        _render_beacon_page_header(
            f"Profile · {name_label}",
            "What the agent knows about you.",
            "Edit anything. The agent re-scores in the background when you save.",
        )
    with head_cols[1]:
        action_cols = st.columns(2, gap="small")
        with action_cols[0]:
            if st.button(
                "Re-import resume",
                key=f"profile_reimport_{slug}",
                use_container_width=True,
                help="Re-extracts the resume PDF and rebuilds the structured profile via one LLM call.",
            ):
                with st.spinner("Re-importing resume…"):
                    ok, msg = _reimport_resume(slug, raw_config)
                if ok:
                    _set_notice(slug, "success", msg)
                    invalidate_dashboard_caches()
                else:
                    _set_notice(slug, "error", msg)
                st.rerun()
        with action_cols[1]:
            if st.button(
                "Save",
                key=f"profile_save_{slug}",
                type="primary",
                use_container_width=True,
            ):
                _save_profile_draft(slug, raw_config, draft)
                _reset_profile_draft(slug)
                _set_notice(slug, "success", "Profile saved.")
                invalidate_dashboard_caches()
                st.rerun()

    st.markdown("<div class='panel-grid'></div>", unsafe_allow_html=True)
    grid_top = st.columns(2, gap="large")
    with grid_top[0]:
        _render_profile_identity_card(slug, draft)
    with grid_top[1]:
        _render_profile_resume_card(slug, raw_config, profile_cfg, structured)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    _render_profile_skills_card(slug, draft)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    grid_bottom = st.columns(2, gap="large")
    with grid_bottom[0]:
        _render_profile_target_roles_card(slug, draft)
    with grid_bottom[1]:
        _render_profile_hard_prefs_card(slug, draft)


def _lines_to_list(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


_SETTINGS_DRAFT_STATE_KEY = "_beacon_settings_draft"
_DEFAULT_WEIGHTS_PCT: dict[str, int] = {
    "role_fit":     30,
    "stack_match":  25,
    "seniority":    20,
    "location":     10,
    "growth":       10,
    "compensation":  5,
}
_WEIGHT_LABELS: list[tuple[str, str]] = [
    ("role_fit",     "Role fit"),
    ("stack_match",  "Stack match"),
    ("seniority",    "Seniority"),
    ("location",     "Location"),
    ("growth",       "Growth"),
    ("compensation", "Compensation"),
]
_SETTINGS_NOTIFICATION_ROWS: list[tuple[str, str, str]] = [
    ("digest",  "Daily digest email",       "Top picks delivered to your inbox at the cadence below."),
    ("slack",   "Slack DM on high-fit",     "Pings when a fit ≥ 90 lands in the queue."),
    ("replies", "Reply detection",          "Watches your inbox and flags recruiter replies."),
    ("agentic", "Auto-draft cover notes",   "Concierge writes drafts you review before sending."),
]
_SETTINGS_SOURCE_TILES: list[tuple[str, str]] = [
    ("greenhouse", "Greenhouse"),
    ("lever",      "Lever"),
    ("ashby",      "Ashby"),
    ("workable",   "Workable"),
    ("hn",         "HN Who's Hiring"),
    ("himalayas",  "Himalayas (remote)"),
]


def _settings_draft_key(slug: str) -> str:
    return f"{_SETTINGS_DRAFT_STATE_KEY}_{slug}"


def _build_settings_draft(raw_config: dict[str, Any]) -> dict[str, Any]:
    weights_raw = (raw_config.get("scoring") or {}).get("weights") or {}
    weights_pct: dict[str, int] = {}
    for key, default in _DEFAULT_WEIGHTS_PCT.items():
        v = weights_raw.get(key)
        if v is None:
            weights_pct[key] = default
        elif isinstance(v, (int, float)):
            weights_pct[key] = int(round(float(v) * 100)) if float(v) <= 1.0 else int(round(float(v)))
        else:
            weights_pct[key] = default
    sources = raw_config.get("sources") or {}
    enabled: dict[str, bool] = {}
    for key, _label in _SETTINGS_SOURCE_TILES:
        enabled[key] = bool((sources.get(key) or {}).get("enabled", False))
    llm_cfg = raw_config.get("llm") or {}
    return {
        "weights":           weights_pct,
        "sources_enabled":   enabled,
        "provider":          str(llm_cfg.get("provider", "groq")),
        "model_overrides":   dict(llm_cfg.get("model") or {}),
        "notifications": {key: False for key, _, _ in _SETTINGS_NOTIFICATION_ROWS},
    }


def _ensure_settings_draft(slug: str, raw_config: dict[str, Any]) -> dict[str, Any]:
    key = _settings_draft_key(slug)
    if key not in st.session_state:
        st.session_state[key] = _build_settings_draft(raw_config)
    return st.session_state[key]


def _reset_settings_draft(slug: str) -> None:
    st.session_state.pop(_settings_draft_key(slug), None)


def _format_provider_model(raw_config: dict[str, Any]) -> str:
    llm_cfg = raw_config.get("llm") or {}
    provider = str(llm_cfg.get("provider", "unknown"))
    model = (llm_cfg.get("model") or {}).get(provider, "unknown")
    return f"{provider} · {model}"


def _format_rate_limit(raw_config: dict[str, Any]) -> str:
    llm_cfg = raw_config.get("llm") or {}
    provider = str(llm_cfg.get("provider", ""))
    rl = (llm_cfg.get("rate_limits") or {}).get(provider) or {}
    rpm = rl.get("max_rpm")
    if rpm:
        return f"{rpm} rpm"
    return "—"


def _format_spend_cap(raw_config: dict[str, Any]) -> str:
    routing = raw_config.get("routing") or {}
    budget = routing.get("rate_limit_budget") or {}
    if not budget:
        return "Unbounded"
    pieces = []
    for provider, caps in list(budget.items())[:2]:
        rpr = caps.get("max_requests_per_run")
        if rpr:
            pieces.append(f"{provider} {rpr}/run")
    if not pieces:
        return "Unbounded"
    extra = len(budget) - len(pieces)
    suffix = f" · +{extra} more" if extra > 0 else ""
    return ", ".join(pieces) + suffix


def _format_embeddings(raw_config: dict[str, Any]) -> str:
    emb = raw_config.get("embeddings") or {}
    return str(emb.get("model") or "—")


def _settings_card_head(title: str, sub: str = "", right_html: str = "") -> str:
    return (
        "<div class='beacon-card-head' style='border:1px solid var(--line);"
        "border-bottom:0;border-radius:var(--r-md) var(--r-md) 0 0;"
        "background:var(--surface);padding:14px 18px'>"
        "<div>"
        f"<div class='beacon-card-title'>{html.escape(title)}</div>"
        + (f"<div class='beacon-card-sub'>{html.escape(sub)}</div>" if sub else "")
        + "</div>"
        + right_html
        + "</div>"
    )


def _render_settings_weights_card(slug: str, draft: dict[str, Any]) -> None:
    weights = draft["weights"]
    st.markdown(_settings_card_head(
        "Scoring weights",
        "What the agent should care about most",
    ), unsafe_allow_html=True)
    st.markdown(
        (
            "<div class='beacon-card' style='border-radius:0 0 var(--r-md) var(--r-md);"
            "border-top:0;margin-top:-1px;padding:14px 22px 12px'>"
        ),
        unsafe_allow_html=True,
    )
    for key, label in _WEIGHT_LABELS:
        cur = int(weights.get(key, _DEFAULT_WEIGHTS_PCT[key]))
        cols = st.columns([1.6, 4.2, 0.7], gap="small")
        with cols[0]:
            st.markdown(
                f"<div style='font-size:13px;font-weight:500;padding-top:8px'>{html.escape(label)}</div>",
                unsafe_allow_html=True,
            )
        with cols[1]:
            new_val = st.slider(
                label,
                min_value=0,
                max_value=100,
                value=cur,
                step=1,
                key=f"settings_weight_{slug}_{key}",
                label_visibility="collapsed",
            )
        with cols[2]:
            st.markdown(
                f"<div style='font-family:var(--font-mono);font-size:12px;color:var(--muted);"
                f"text-align:right;padding-top:8px'>{int(new_val)}%</div>",
                unsafe_allow_html=True,
            )
        weights[key] = int(new_val)
    total = sum(weights.values())
    cap_color = "var(--ink)" if total <= 100 else "var(--warn)"
    st.markdown(
        (
            f"<p style='font-size:12px;color:var(--muted);margin-top:10px;line-height:1.55'>"
            f"Total <b style='color:{cap_color}'>{total}%</b>. "
            "The agent normalizes if you go over."
            "</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_settings_sources_card(
    slug: str,
    draft: dict[str, Any],
    raw_config: dict[str, Any],
    last_run: dict[str, Any] | None,
) -> None:
    enabled = draft["sources_enabled"]
    last_run_count: dict[str, int] = {}
    if last_run:
        all_records = _cached_fetch_job_summaries(slug)
        finished = _parse_iso(last_run.get("finished_at"))
        started = _parse_iso(last_run.get("started_at"))
        if finished and started:
            for r in all_records:
                created = _parse_iso(r.get("created_at"))
                if created and started <= created <= finished:
                    s = str(r.get("source") or "").lower()
                    if s:
                        last_run_count[s] = last_run_count.get(s, 0) + 1
    head_right = ("<span style='font-family:var(--font-mono);font-size:11px;color:var(--muted)'>"
                  f"{sum(1 for v in enabled.values() if v)} active"
                  "</span>")
    st.markdown(_settings_card_head("Sources", "Where the agent looks", head_right), unsafe_allow_html=True)
    st.markdown(
        (
            "<div class='beacon-card' style='border-radius:0 0 var(--r-md) var(--r-md);"
            "border-top:0;margin-top:-1px;padding:14px 22px 16px'>"
        ),
        unsafe_allow_html=True,
    )
    for key, label in _SETTINGS_SOURCE_TILES:
        on = bool(enabled.get(key, False))
        count = last_run_count.get(key, 0)
        ct_label = f"{count} last run" if (on and count) else ("on" if on else "off")
        tile_cls = "source-tile" if on else "source-tile off"
        tile_cols = st.columns([5.4, 1.6], gap="small")
        with tile_cols[0]:
            st.markdown(
                (
                    f"<div class='{tile_cls}'>"
                    f"<div class='nm'><span class='d'></span> {html.escape(label)}</div>"
                    f"<span class='ct'>{html.escape(ct_label)}</span>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        with tile_cols[1]:
            new_val = st.toggle(
                label,
                key=f"settings_source_{slug}_{key}",
                value=on,
                label_visibility="collapsed",
            )
            if new_val != on:
                enabled[key] = bool(new_val)
                st.rerun()
    if st.button(
        "Add source",
        key=f"settings_sources_add_{slug}",
        help="Custom-source connectors are coming in a future chunk.",
    ):
        st.toast("Add-source UI is coming soon. Use the toggles above for built-in connectors.")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_settings_schedule_card(slug: str, raw_config: dict[str, Any], draft: dict[str, Any]) -> None:
    cadence = "Daily at 04:00 UTC · plus on-demand"
    st.markdown(_settings_card_head("Schedule & model"), unsafe_allow_html=True)
    rows = [
        ("Cadence",       cadence),
        ("Scoring model", _format_provider_model(raw_config)),
        ("Embeddings",    _format_embeddings(raw_config)),
        ("Rate limit",    _format_rate_limit(raw_config)),
        ("Spend cap",     _format_spend_cap(raw_config)),
    ]
    rows_html = "".join(
        f"<div class='kv'><span class='k'>{html.escape(label)}</span>"
        f"<span class='v'>{html.escape(str(value))}</span></div>"
        for label, value in rows
    )
    st.markdown(
        (
            "<div class='beacon-card' style='border-radius:0 0 var(--r-md) var(--r-md);"
            "border-top:0;margin-top:-1px;padding:14px 22px 12px'>"
            f"{rows_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    with st.expander("Edit provider & model", expanded=False):
        provider_options = ["groq", "anthropic", "gemini", "openai"]
        cur_provider = draft.get("provider", "groq")
        idx = provider_options.index(cur_provider) if cur_provider in provider_options else 0
        new_provider = st.selectbox(
            "Provider",
            options=provider_options,
            index=idx,
            key=f"settings_provider_{slug}",
            help="The default LLM provider used for scoring runs.",
        )
        draft["provider"] = new_provider
        existing_model = (draft.get("model_overrides") or {}).get(new_provider) or ""
        new_model = st.text_input(
            "Model name",
            value=str(existing_model),
            key=f"settings_model_{slug}",
            placeholder="e.g. meta-llama/llama-4-scout-17b-16e-instruct",
        )
        draft.setdefault("model_overrides", {})[new_provider] = new_model.strip()


def _render_settings_notifications_card(slug: str, draft: dict[str, Any]) -> None:
    webhook_set = bool((os.environ.get("JOBAGENT_NOTIFY_WEBHOOK") or "").strip())
    head_right = (
        "<span style='font-family:var(--font-mono);font-size:11px;color:var(--muted)'>"
        f"webhook {'configured' if webhook_set else 'not set'}"
        "</span>"
    )
    st.markdown(_settings_card_head("Notifications & autonomy", right_html=head_right), unsafe_allow_html=True)
    st.markdown(
        (
            "<div class='beacon-card' style='border-radius:0 0 var(--r-md) var(--r-md);"
            "border-top:0;margin-top:-1px;padding:6px 22px 6px'>"
        ),
        unsafe_allow_html=True,
    )
    notif = draft.setdefault("notifications", {})
    for key, title, sub in _SETTINGS_NOTIFICATION_ROWS:
        row_cols = st.columns([5.4, 1.6], gap="small")
        with row_cols[0]:
            st.markdown(
                (
                    "<div class='notif-row'>"
                    "<div>"
                    f"<div class='label'>{html.escape(title)}</div>"
                    f"<div class='sub'>{html.escape(sub)}</div>"
                    "<div class='soon'>Coming soon</div>"
                    "</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        with row_cols[1]:
            new_val = st.toggle(
                title,
                key=f"settings_notif_{slug}_{key}",
                value=bool(notif.get(key)),
                label_visibility="collapsed",
            )
            notif[key] = bool(new_val)
    st.markdown(
        (
            "<p style='font-size:12px;color:var(--muted);margin:6px 0 12px;line-height:1.55'>"
            "Toggles persist in this session only — there is no working backend yet. "
            "Failure notifications are wired separately via "
            "<code>JOBAGENT_NOTIFY_WEBHOOK</code> in the environment."
            "</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _save_settings_draft(slug: str, raw_config: dict[str, Any], draft: dict[str, Any]) -> tuple[bool, str]:
    enabled = draft["sources_enabled"]
    if not any(enabled.values()):
        return False, "Enable at least one source before saving."
    weights_pct = draft["weights"]
    total_pct = sum(weights_pct.values())
    if total_pct == 0:
        return False, "At least one scoring weight must be greater than zero."
    factor = total_pct / 100.0 if total_pct > 0 else 1.0
    weights_norm = {k: round((v / 100.0) / factor, 4) for k, v in weights_pct.items()} if factor > 0 else {
        k: round(v / 100.0, 4) for k, v in weights_pct.items()
    }
    updated = copy.deepcopy(raw_config)
    updated.setdefault("scoring", {})["weights"] = weights_norm
    sources = updated.setdefault("sources", {})
    for key, _label in _SETTINGS_SOURCE_TILES:
        block = sources.setdefault(key, {})
        block["enabled"] = bool(enabled.get(key))
    llm_cfg = updated.setdefault("llm", {})
    llm_cfg["provider"] = draft.get("provider", llm_cfg.get("provider", "groq"))
    overrides = draft.get("model_overrides") or {}
    if overrides:
        models_section = llm_cfg.setdefault("model", {})
        for prov, model_name in overrides.items():
            if model_name:
                models_section[prov] = model_name
    _write_profile_config(slug, updated)
    return True, "Settings saved for this profile."


def _render_settings_tab(slug: str, config: dict[str, Any], raw_config: dict[str, Any], metrics: dict[str, Any]) -> None:
    draft = _ensure_settings_draft(slug, raw_config)

    head_cols = st.columns([6.4, 2.6], gap="medium")
    with head_cols[0]:
        _render_beacon_page_header(
            "Settings",
            "How the agent works for you.",
            "Tune scoring weights, sources, schedule, and notifications. Changes take effect on the next run.",
        )
    with head_cols[1]:
        action_cols = st.columns(2, gap="small")
        with action_cols[0]:
            if st.button(
                "Reset",
                key=f"settings_reset_{slug}",
                use_container_width=True,
                help="Discards in-progress changes and reverts to what's currently in config.yaml.",
            ):
                _reset_settings_draft(slug)
                st.toast("Settings draft reset.")
                st.rerun()
        with action_cols[1]:
            if st.button(
                "Save",
                key=f"settings_save_{slug}",
                use_container_width=True,
                type="primary",
            ):
                ok, msg = _save_settings_draft(slug, raw_config, draft)
                if ok:
                    _set_notice(slug, "success", msg)
                    invalidate_dashboard_caches()
                    st.rerun()
                else:
                    _set_notice(slug, "error", msg)
                    st.rerun()

    runs = _cached_recent_runs(slug)
    last_run = runs[0] if runs else None

    st.markdown("<div class='panel-grid'></div>", unsafe_allow_html=True)
    grid_top = st.columns(2, gap="large")
    with grid_top[0]:
        _render_settings_weights_card(slug, draft)
    with grid_top[1]:
        _render_settings_sources_card(slug, draft, raw_config, last_run)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    grid_bottom = st.columns(2, gap="large")
    with grid_bottom[0]:
        _render_settings_schedule_card(slug, raw_config, draft)
    with grid_bottom[1]:
        _render_settings_notifications_card(slug, draft)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    with st.expander("Advanced matching", expanded=False):
        st.caption(
            "Fine-tune hard-no keywords and per-source company allowlists. Titles and "
            "skills now live on the Profile tab."
        )
        editable = apply_config_defaults(copy.deepcopy(raw_config))
        editable.setdefault("preferences", {})
        editable["preferences"].setdefault("compensation", {})
        editable.setdefault("sources", {})
        for key in ("greenhouse", "lever", "ashby", "workable"):
            editable["sources"].setdefault(key, {"enabled": True, "companies": []})
        editable["sources"].setdefault("hn",        {"enabled": False})
        editable["sources"].setdefault("himalayas", {"enabled": False})
        hard_no_default = editable["preferences"].get("hard_no_keywords", []) or []
        gh_companies_default  = editable["sources"]["greenhouse"].get("companies", [])
        lv_companies_default  = editable["sources"]["lever"].get("companies", [])
        ash_companies_default = editable["sources"]["ashby"].get("companies", [])
        wl_companies_default  = editable["sources"]["workable"].get("companies", [])
        hard_no_text  = st.text_area(
            "Hard-no keywords",
            value="\n".join(hard_no_default),
            height=100,
            help="Any posting containing these phrases will be skipped before scoring.",
            key=f"settings_advanced_hardno_{slug}",
        )
        col_a, col_b = st.columns(2, gap="medium")
        with col_a:
            gh_companies = st.text_area(
                "Greenhouse companies",
                value="\n".join(gh_companies_default),
                height=140,
                key=f"settings_advanced_gh_{slug}",
                help="One company slug per line.",
            )
            ash_companies = st.text_area(
                "Ashby companies",
                value="\n".join(ash_companies_default),
                height=140,
                key=f"settings_advanced_ash_{slug}",
            )
        with col_b:
            lv_companies = st.text_area(
                "Lever companies",
                value="\n".join(lv_companies_default),
                height=140,
                key=f"settings_advanced_lv_{slug}",
            )
            wl_companies = st.text_area(
                "Workable companies",
                value="\n".join(wl_companies_default),
                height=140,
                key=f"settings_advanced_wl_{slug}",
            )
        if st.button("Save advanced matching", key=f"settings_advanced_save_{slug}", type="primary"):
            updated = copy.deepcopy(raw_config)
            updated.setdefault("preferences", {})["hard_no_keywords"] = _lines_to_list(hard_no_text)
            updated.setdefault("sources", {})
            for key, companies_text in (
                ("greenhouse", gh_companies),
                ("lever",      lv_companies),
                ("ashby",      ash_companies),
                ("workable",   wl_companies),
            ):
                block = updated["sources"].setdefault(key, {})
                block["companies"] = _lines_to_list(companies_text)
            _write_profile_config(slug, updated)
            _set_notice(slug, "success", "Advanced matching saved.")
            invalidate_dashboard_caches()
            st.rerun()

    with st.expander("Cost optimization (selective routing)", expanded=False):
        editable = apply_config_defaults(copy.deepcopy(raw_config))
        routing_cfg = editable.get("routing", {})
        st.caption(
            "Route low-confidence jobs to a synthetic score instead of the LLM. "
            "Reduces API costs by 60–90% with minimal quality impact."
        )
        routing_enabled_val = st.toggle(
            "Enable selective routing",
            value=bool(routing_cfg.get("enabled", False)),
            key=f"routing_enabled_{slug}",
        )
        current_threshold = float(
            routing_cfg.get("llm_threshold")
            if routing_cfg.get("llm_threshold") is not None
            else routing_cfg.get("threshold", 0.18)
        )
        routing_threshold = st.slider(
            "Routing threshold",
            min_value=0.05,
            max_value=0.90,
            value=max(0.05, min(0.90, current_threshold)),
            step=0.01,
            key=f"routing_threshold_{slug}",
            disabled=not routing_enabled_val,
            help=(
                "Jobs whose cross-encoder score is below this value skip the LLM. "
                "Lower = more LLM calls. Higher = fewer."
            ),
        )
        routing_quality = st.radio(
            "Quality mode",
            options=["fast", "quality"],
            index=0 if routing_cfg.get("quality_mode", "fast") == "fast" else 1,
            horizontal=True,
            key=f"routing_quality_{slug}",
            disabled=not routing_enabled_val,
        )
        routing_log = st.checkbox(
            "Log routing decisions",
            value=bool(routing_cfg.get("log_routing_decisions", True)),
            key=f"routing_log_{slug}",
            disabled=not routing_enabled_val,
        )
        try:
            stats = get_routing_stats(profile=slug)
            if stats:
                total = sum(stats.values())
                skipped = stats.get("skipped_llm", 0)
                called = stats.get("llm_called", 0)
                savings_pct = round(skipped / total * 100) if total else 0
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("LLM calls", called)
                col_r2.metric("Skipped (synthetic)", skipped)
                col_r3.metric("Cost savings", f"{savings_pct}%")
            else:
                st.caption("No routing decisions logged yet for this profile.")
        except Exception:
            pass
        if st.button("Save routing settings", key=f"routing_save_{slug}", type="primary"):
            updated = copy.deepcopy(raw_config)
            existing_routing = dict(updated.get("routing") or {})
            existing_routing["enabled"] = routing_enabled_val
            existing_routing["llm_threshold"] = round(float(routing_threshold), 2)
            existing_routing["quality_mode"] = routing_quality
            existing_routing["log_routing_decisions"] = routing_log
            updated["routing"] = existing_routing
            _write_profile_config(slug, updated)
            _set_notice(slug, "success", "Routing settings saved.")
            invalidate_dashboard_caches()
            st.rerun()

    with st.expander("Scoring · minimum display score", expanded=False):
        editable = apply_config_defaults(copy.deepcopy(raw_config))
        editable.setdefault("scoring", {})
        cur = int(editable["scoring"].get("min_display_score", 60))
        new_min = st.slider(
            "Minimum display score",
            min_value=0,
            max_value=100,
            value=cur,
            step=5,
            key=f"settings_min_display_{slug}",
            help="Jobs below this score stay saved but are hidden from the Jobs tab.",
        )
        if st.button("Save minimum score", key=f"settings_min_display_save_{slug}"):
            updated = copy.deepcopy(raw_config)
            updated.setdefault("scoring", {})["min_display_score"] = int(new_min)
            _write_profile_config(slug, updated)
            _set_notice(slug, "success", "Minimum display score saved.")
            invalidate_dashboard_caches()
            st.rerun()

    with st.expander("Profile actions", expanded=False):
        callout("info", "Profile management", "Create a fresh workspace or refresh scores without removing saved jobs.")
        profile_action_cols = st.columns([0.95, 1.05], gap="large")
        with profile_action_cols[0]:
            if st.button("Create new profile", key=f"settings_create_profile_{slug}", type="secondary"):
                _open_create_profile_dialog()
        with profile_action_cols[1]:
            confirm_rescore = bool(st.session_state.get(f"confirm_rescore_{slug}", False))
            if st.button("Re-score profile", key=f"rescore_all_{slug}", disabled=not confirm_rescore):
                config["_active_profile"] = slug
                missing = _check_api_key(config)
                if missing:
                    callout("error", "API key missing", f"Set {missing} before running a rescore.")
                else:
                    try:
                        with st.spinner("Resetting scores and rescoring jobs..."):
                            run_id = start_run(profile=slug, source="dashboard_rescore")
                            rescore_reset(profile=slug)
                            results = score_all_jobs(config, yes=True, profile=slug)
                            scored = [item for item in results if item.get("fit_score", 0) > 0]
                            avg_fit = round(sum(item["fit_score"] for item in scored) / len(scored), 1) if scored else 0.0
                            finish_run(run_id, jobs_scraped=0, jobs_filtered=0, jobs_saved=0, jobs_scored=len(scored), avg_fit_score=avg_fit, errors=[], status="complete", profile=slug)
                        _set_notice(slug, "success", f"Re-scored {len(scored)} jobs.")
                        invalidate_dashboard_caches()
                        st.rerun()
                    except Exception as exc:
                        callout("error", "Re-score failed", str(exc))
            st.checkbox(
                f"I understand this will rescore {metrics.get('db_total', metrics['total'])} jobs and replace existing scores.",
                key=f"confirm_rescore_{slug}",
            )

    with st.expander("Tracking & evaluation", expanded=False):
        _render_tracking_status_panel(config, slug)

    with st.expander("Danger zone", expanded=False):
        callout("error", "Destructive action", "This removes the current profile database and run history. It cannot be undone from the UI.")
        confirm_clear = st.checkbox(
            f"I understand this will permanently erase {metrics.get('db_total', metrics['total'])} jobs and all run history for this profile.",
            key=f"confirm_clear_{slug}",
        )
        if st.button("Clear profile database", key=f"clear_all_{slug}", disabled=not confirm_clear):
            _clear_profile_jobs(slug)
            _set_notice(slug, "success", "Profile database cleared.")
            invalidate_dashboard_caches()
            st.rerun()


def _render_profile_dashboard(slug: str) -> None:
    set_active_profile(slug)
    configure_logging(profile=slug, debug=False)
    init_db(profile=slug)

    try:
        config = load_config(profile=slug)
        raw_config = _read_profile_config(slug)
    except Exception as exc:
        logger.exception("Failed to load profile '{}'", slug)
        callout("error", "Profile could not be loaded", f"Profile '{slug}' could not be loaded.")
        st.code(str(exc))
        if st.button("Back to profile list", key=f"broken_profile_back_{slug}"):
            st.session_state.active_profile = None
            set_active_profile(None)
            st.rerun()
        return

    all_records = _cached_fetch_job_summaries(slug)
    # Three mutually exclusive partitions:
    # - records: passed scrape filters and not LLM-disqualified → review queue
    # - disq_records: LLM-disqualified (scored with disqualified=1) → hidden
    # - scrape_rejected_records: rejected pre-LLM (scrape_qualified=0) → optionally shown
    records = [r for r in all_records if not r.get("disqualified") and r.get("scrape_qualified", 1)]
    disq_records = [r for r in all_records if r.get("disqualified")]
    scrape_rejected_records = [r for r in all_records if not r.get("scrape_qualified", 1)]
    metrics = _collect_metrics(slug, records)
    # Inject disqualified summary for the Activity tab.
    by_reason: dict[str, int] = {}
    for r in disq_records:
        reason = r.get("disqualify_reason") or "unknown"
        by_reason[reason] = by_reason.get(reason, 0) + 1
    metrics["disqualified_count"] = len(disq_records)
    metrics["disqualified_by_reason"] = by_reason
    # Inject scrape-rejected summary for the Activity tab.
    rej_by_reason: dict[str, int] = {}
    for r in scrape_rejected_records:
        reason = r.get("scrape_filter_reason") or "unknown"
        prefix = reason.split(":")[0] if ":" in reason else reason
        rej_by_reason[prefix] = rej_by_reason.get(prefix, 0) + 1
    metrics["scrape_rejected_count"] = len(scrape_rejected_records)
    metrics["scrape_rejected_by_reason"] = rej_by_reason
    # Keep DB-accurate total so settings warnings ("erase N jobs") are correct.
    metrics["db_total"] = metrics["total"] + len(disq_records) + len(scrape_rejected_records)
    runs = _cached_recent_runs(slug)
    profile_name = config.get("profile", {}).get("name", slug.replace("_", " ").title())

    worker_running = _worker_is_running(slug)
    # Detect transition: worker just finished → invalidate caches once.
    was_running_key = f"worker_was_running_{slug}"
    was_running = st.session_state.get(was_running_key, False)
    if was_running and not worker_running:
        st.session_state[was_running_key] = False
        invalidate_dashboard_caches()
        _set_notice(slug, "success", "Pipeline complete. Results updated.")
        st.rerun()
    if worker_running:
        st.session_state[was_running_key] = True

    _render_notice(slug)
    if not worker_running:
        _render_staleness_banner(slug)
    sidebar_action = _render_sidebar_nav(slug, profile_name, config, metrics, worker_running=worker_running)
    if sidebar_action == "run_search" and not worker_running:
        missing = _check_api_key(config)
        if missing:
            _set_notice(slug, "error", f"Scoring requires {missing} in your environment.")
            st.rerun()
        else:
            _launch_worker(slug)
            st.session_state["_scroll_to_progress"] = True
            st.rerun()
    if sidebar_action == "create_profile":
        _open_create_profile_dialog()
    if sidebar_action == "rerun_setup":
        st.session_state.show_onboarding = True
        st.rerun()
    if sidebar_action == "switch_profile":
        st.session_state.active_profile = None
        set_active_profile(None)
        st.rerun()

    if st.session_state.pop("_scroll_to_progress", False):
        components.html(
            """<script>
            (function() {
                var main = window.parent.document.querySelector('section[data-testid="stMain"]')
                        || window.parent.document.querySelector('.main')
                        || window.parent.document.body;
                main.scrollTo({top: 0, behavior: 'smooth'});
            })();
            </script>""",
            height=0,
        )

    progress_host = st.empty()
    if worker_running:
        progress_data = _read_progress_json(slug)
        if progress_data:
            tracker = ProgressTracker.from_dict(progress_data)
            _render_pipeline_snapshot(tracker, progress_host)
        else:
            with progress_host.container():
                st.info("Worker is starting up, please wait...")

    last_error = st.session_state.get("last_pipeline_error")
    if last_error and last_error.get("profile") == slug:
        with panel("Latest pipeline error", subtitle="Most recent failed run from this session"):
            callout("error", "Pipeline failed", last_error.get("message", "Unknown error"))
            st.code(last_error.get("diagnostics", ""), language="text")

    section = st.session_state.get("dashboard_section", "Overview")
    try:
        if section == "Overview":
            _render_overview_tab(slug, config, raw_config, records, runs, metrics)
        elif section == "Jobs":
            _render_jobs_tab(config, records, slug, scrape_rejected_records=scrape_rejected_records)
        elif section == "Activity":
            _render_activity_tab(slug, runs, metrics, raw_config, config)
        elif section == "Profile":
            _render_profile_tab(slug, config, raw_config)
        else:
            _render_settings_tab(slug, config, raw_config, metrics)
    except Exception as exc:
        logger.exception(
            "Failed to render dashboard section '{}' for profile '{}'",
            section,
            slug,
        )
        callout(
            "error",
            f"{section} could not be loaded",
            (
                "The dashboard hit an unexpected error while rendering this section. "
                f"Check `{_dashboard_log_path(slug)}` for details."
            ),
        )
        if DEBUG_THEME:
            st.code(str(exc), language="text")
        if st.button("Back to overview", key=f"section_error_back_{slug}"):
            st.session_state.dashboard_section = "Overview"
            st.rerun()

    if worker_running:
        time.sleep(2)
        st.rerun()


def _render_profile_selection() -> None:
    set_active_profile(None)
    profiles = list_profiles()
    st.session_state.dashboard_section = "Overview"

    if len(profiles) == 1:
        only_profile = profiles[0]
        st.session_state.active_profile = only_profile["slug"]
        st.session_state.show_onboarding = False
        set_active_profile(only_profile["slug"])
        st.rerun()

    _render_html_block(
        (
            "<section class='shell-selection-header'>"
            "<h1 class='shell-selection-title'>Select a workspace</h1>"
            "<p class='shell-selection-copy'>Open an existing workspace or create a new one to continue.</p>"
            "</section>"
        )
    )

    if not profiles:
        clicked = empty_state(
            "Create your first workspace to get started",
            "No workspaces found yet. Create your first workspace to start finding matching job opportunities.",
            actions=[
                {
                    "id": "create_profile_zero_state",
                    "label": "Create your first workspace",
                    "type": "primary",
                    "key": "create_profile_zero_state",
                }
            ],
            mark="Zero state",
        )
        if clicked == "create_profile_zero_state":
            _open_create_profile_dialog()
        return

    _render_html_block("<div class='shell-selection-label'>Available workspaces</div>")
    grid_count = min(2, max(1, len(profiles)))
    columns = st.columns(grid_count, gap="large")
    for index, profile in enumerate(profiles):
        with columns[index % grid_count]:
            role_line = (
                f"{profile['job_type'].replace('_', ' ').title()} search with {profile['provider'].title()}"
            )
            with st.container(border=True):
                _render_html_block(
                    (
                        "<div class='shell-workspace-card'>"
                        f"<div class='shell-workspace-card-title'>{html.escape(profile['name'])}</div>"
                        f"<div class='shell-workspace-card-copy'>{html.escape(role_line)}</div>"
                        "</div>"
                    )
                )
                _render_html_block("<div class='shell-workspace-card-action'>")
                if st.button(
                    "Open workspace",
                    key=f"open_profile_{profile['slug']}",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.active_profile = profile["slug"]
                    st.session_state.show_onboarding = False
                    set_active_profile(profile["slug"])
                    st.rerun()
                _render_html_block("</div>")

    _render_html_block("<div class='shell-selection-footer'></div>")
    if st.button("Create new workspace", key="create_profile_selection_footer", type="secondary", help="Open a quick workspace-creation dialog."):
        _open_create_profile_dialog()


def main() -> None:
    apply_page_scaffold(PAGE_TITLE)
    _init_state()
    log_profile = st.session_state.get("active_profile") or "default"
    configure_logging(profile=log_profile, debug=False)

    try:
        _render_theme_bootstrap_notice()

        if st.session_state.show_onboarding:
            render_onboarding()
            return

        if st.session_state.active_profile:
            _render_profile_dashboard(st.session_state.active_profile)
        else:
            _render_profile_selection()

        if st.session_state.get("show_create_profile_dialog"):
            _render_create_profile_dialog()
        if st.session_state.get("show_resume_preview_dialog"):
            st.session_state.show_resume_preview_dialog = False
            _render_resume_preview_dialog()
    except Exception as exc:
        logger.exception("Dashboard failed to render for profile '{}'", log_profile)
        callout(
            "error",
            "Dashboard failed to load",
            f"Check `{_dashboard_log_path(log_profile)}` for details, then refresh the page.",
        )
        if DEBUG_THEME:
            st.code(str(exc), language="text")


if __name__ == "__main__":
    main()
