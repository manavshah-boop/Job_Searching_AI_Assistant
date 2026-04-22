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
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yaml
from dotenv import load_dotenv
from loguru import logger

from config import _resolve_resume_path, load_config
from db import (
    count_jobs,
    finish_run,
    get_db_path,
    get_recent_runs,
    init_db,
    load_discovered_slugs,
    rescore_reset,
    set_active_profile,
    start_run,
    update_job_status,
)
from dashboard_ui import (
    render_activity_feed,
    render_pipeline_stages,
    render_progress_header,
    render_source_progress,
)
from logging_config import configure_logging
from onboarding import render_onboarding, sanitize_slug
from progress_tracker import ProgressTracker
from scorer import score_all_jobs
from ui_shell import (
    badge,
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


_WORKER_LOCKFILE  = ".worker_running"
_WORKER_PROGRESS  = ".run_progress.json"
_WORKER_STALE_SEC = 3 * 3600  # match worker/run_pipeline.py


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


def _profile_config_path(slug: str) -> Path:
    return PROFILES_DIR / slug / "config.yaml"


def _dashboard_log_path(profile: str) -> Path:
    return BASE_DIR / "logs" / profile / "agent.log"


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


def _render_sidebar_nav(
    slug: str,
    profile_name: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    *,
    worker_running: bool = False,
) -> str | None:
    available_profiles = list_profiles()
    secondary_actions = [
        {
            "id": "create_profile",
            "label": "Create new profile",
            "key": f"create_profile_sidebar_{slug}",
        }
    ]
    if len(available_profiles) > 1:
        secondary_actions.append(
            {
                "id": "switch_profile",
                "label": "Change profile",
                "key": f"switch_profile_sidebar_{slug}",
        }
    )
    with st.sidebar:
        sidebar_profile_summary(profile_name)
        _render_html_block("<div class='shell-sidebar-divider' aria-hidden='true'></div>")
        _render_html_block("<div class='shell-sidebar-section'>")
        _render_html_block("<div class='shell-inline-section-label shell-inline-section-label--sidebar'>Navigation</div>")
        st.radio(
            "Sections",
            SECTION_ORDER,
            key="dashboard_section",
            label_visibility="collapsed",
        )
        _render_html_block("</div>")
        _render_html_block("<div class='shell-sidebar-divider shell-sidebar-divider--section' aria-hidden='true'></div>")
        _render_html_block("<div class='shell-sidebar-actions'>")
        _render_html_block("<div class='shell-inline-section-label shell-inline-section-label--sidebar'>Actions</div>")
        run_label = "Running..." if worker_running else "Start discovery"
        primary_action = toolbar(
            primary_actions=[
                {
                    "id": "run_search",
                    "label": run_label,
                    "key": f"run_search_sidebar_{slug}",
                    "disabled": worker_running,
                }
            ],
            class_name="shell-toolbar shell-toolbar--sidebar-primary shell-toolbar--compact",
        )
        secondary_action = None
        for action in secondary_actions:
            clicked_action = toolbar(
                secondary_actions=[action],
                class_name="shell-toolbar shell-toolbar--sidebar-subtle shell-toolbar--compact",
            )
            secondary_action = secondary_action or clicked_action
        _render_html_block("</div>")
    return primary_action or secondary_action


def _source_label(source: str) -> str:
    return SOURCE_LABELS.get(source, source.replace("_", " ").title())


def _score_state(record: sqlite3.Row) -> str:
    if record["fit_score"] is not None:
        return "Scored"
    if (record["score_attempts"] or 0) >= 3:
        return "Failed"
    if record["score_error"]:
        return "Needs retry"
    return "Pending"


def _deserialize_job_record(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    record["score_state"] = _score_state(row)
    record["source_label"] = _source_label(record["source"])
    record["status_label"] = str(record["status"]).title()
    record["reasons"] = json.loads(record.get("reasons") or "[]")
    record["flags"] = json.loads(record.get("flags") or "[]")
    record["skill_misses"] = json.loads(record.get("skill_misses") or "[]")
    record["one_liner"] = record.get("one_liner") or ""
    record["raw_text"] = record.get("raw_text") or ""
    record["dimension_scores"] = {
        "role_fit": record.get("role_fit"),
        "stack_match": record.get("stack_match"),
        "seniority": record.get("seniority"),
        "location": record.get("score_location"),
        "growth": record.get("growth"),
        "compensation": record.get("compensation"),
    }
    # Canonical disqualified signal from DB column; fall back to flags for
    # jobs scored before the disqualified column existed.
    disq_from_db = bool(record.get("disqualified", 0))
    disq_reason = record.get("disqualify_reason", "") or ""
    if not disq_from_db and not disq_reason:
        for flag in record["flags"]:
            if str(flag).startswith("disqualified:"):
                disq_from_db = True
                disq_reason = str(flag).replace("disqualified:", "", 1).strip()
                break
    record["disqualified"] = disq_from_db
    record["disqualify_reason"] = disq_reason
    record["scrape_qualified"] = int(record.get("scrape_qualified") or 1)
    record["scrape_filter_reason"] = record.get("scrape_filter_reason") or ""
    return record


def _fetch_job_summaries(slug: str) -> list[dict[str, Any]]:
    init_db(profile=slug)
    conn = sqlite3.connect(get_db_path(slug))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                j.id,
                j.title,
                j.company,
                j.location,
                j.url,
                j.source,
                j.score_attempts,
                j.score_error,
                j.status,
                j.created_at,
                j.scrape_qualified,
                j.scrape_filter_reason,
                s.fit_score,
                s.ats_score,
                s.reasons,
                s.flags,
                s.skill_misses,
                s.one_liner,
                s.role_fit,
                s.stack_match,
                s.seniority,
                s.location AS score_location,
                s.growth,
                s.compensation,
                s.scored_at,
                s.disqualified,
                s.disqualify_reason
            FROM jobs j
            LEFT JOIN scores s ON j.id = s.job_id
            ORDER BY COALESCE(s.fit_score, -1) DESC, j.created_at DESC
            """
        ).fetchall()
        return [_deserialize_job_record(row) for row in rows]
    finally:
        conn.close()


def _fetch_job_detail(slug: str, job_id: str) -> Optional[dict[str, Any]]:
    init_db(profile=slug)
    conn = sqlite3.connect(get_db_path(slug))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT
                j.id,
                j.title,
                j.company,
                j.location,
                j.url,
                j.raw_text,
                j.source,
                j.score_attempts,
                j.score_error,
                j.status,
                j.created_at,
                j.scrape_qualified,
                j.scrape_filter_reason,
                s.fit_score,
                s.ats_score,
                s.reasons,
                s.flags,
                s.skill_misses,
                s.one_liner,
                s.role_fit,
                s.stack_match,
                s.seniority,
                s.location AS score_location,
                s.growth,
                s.compensation,
                s.scored_at,
                s.disqualified,
                s.disqualify_reason
            FROM jobs j
            LEFT JOIN scores s ON j.id = s.job_id
            WHERE j.id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            return None
        return _deserialize_job_record(row)
    finally:
        conn.close()


def _search_job_ids_by_raw_text(slug: str, query: str) -> set[str]:
    init_db(profile=slug)
    conn = sqlite3.connect(get_db_path(slug))
    like_query = f"%{query.lower()}%"
    try:
        rows = conn.execute(
            """
            SELECT id
            FROM jobs
            WHERE LOWER(COALESCE(raw_text, '')) LIKE ?
            """,
            (like_query,),
        ).fetchall()
        return {row[0] for row in rows}
    finally:
        conn.close()


def _collect_metrics(slug: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    # `records` is already filtered to non-disqualified jobs by the caller.
    source_counts: dict[str, int] = {}
    scored = 0
    pending = 0
    failed = 0
    retries = 0
    applied = 0
    skipped = 0

    for record in records:
        source_counts[record["source_label"]] = source_counts.get(record["source_label"], 0) + 1
        if record["status"] == "applied":
            applied += 1
        elif record["status"] == "skipped":
            skipped += 1

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
    conn = sqlite3.connect(get_db_path(slug))
    conn.execute("DELETE FROM scores")
    conn.execute("DELETE FROM jobs")
    conn.execute("DELETE FROM scrape_runs")
    conn.commit()
    conn.close()


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
    return {
        "applied": "success",
        "skipped": "warning",
        "new": "info",
    }.get(status.lower(), "neutral")


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
            _set_job_status_and_refresh(slug, record["id"], "skipped")
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
                    "bulk_skipped": "skipped",
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
            _set_job_status_and_refresh(slug, record["id"], "skipped")
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


def _render_overview_tab(
    slug: str,
    config: dict[str, Any],
    raw_config: dict[str, Any],
    records: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    with section_shell("Overview", SECTION_COPY["Overview"]):
        if metrics["total"] == 0:
            _render_overview_action_card(slug, config)
        else:
            next_action = "Review jobs"
            status_label = "Needs attention" if metrics["needs_retry"] or metrics["failed"] else "On track"
            status_copy = (
                "Some results or runs need follow-up before you trust the shortlist."
                if status_label == "Needs attention"
                else "Your latest results are ready to review."
            )
            _render_html_block(
                (
                    "<div class='overview-guidance'>"
                    f"<div><div class='overview-guidance-kicker'>What to do next</div><div class='overview-guidance-title'>{html.escape(next_action)}</div><div class='overview-guidance-copy'>{html.escape(status_copy)}</div></div>"
                    f"<div class='overview-guidance-status'>{badge(status_label, 'warning' if status_label == 'Needs attention' else 'success')}</div>"
                    "</div>"
                )
            )
        _render_overview_scoreboard(metrics)
        _render_html_block("<div class='shell-panel-gap' aria-hidden='true'></div>")

        with panel("Best opportunities", subtitle="Start here. These are the strongest roles to review next.", tone="primary"):
            _render_best_opportunities_panel(slug, metrics, raw_config)

        with panel("Search preferences", subtitle="Current role, location, and compensation focus", tone="supporting"):
            _render_config_summary_card(config, raw_config)

        _render_overview_run_bar(runs)


def _render_jobs_tab(
    records: list[dict[str, Any]],
    slug: str,
    *,
    scrape_rejected_records: list[dict[str, Any]] | None = None,
) -> None:
    profile_name = st.session_state.get("active_profile", slug).replace("_", " ").title()
    with section_shell("Jobs", SECTION_COPY["Jobs"]):
        if not records:
            clicked = empty_state(
                "Your review queue is empty",
                "Start discovery to generate a curated list of scored job matches. Once complete, you can filter, compare, and manage applications directly from this view.",
                actions=[
                    {
                        "id": "start_discovery_jobs_empty",
                        "label": "Start discovery",
                        "type": "primary",
                        "key": f"start_discovery_jobs_empty_{slug}",
                    }
                ],
                mark="Zero state",
                icon="document",
            )
            if clicked == "start_discovery_jobs_empty":
                _queue_pipeline_run(slug)
            return

        source_options = sorted({record["source_label"] for record in records})
        status_options = sorted({record["status_label"] for record in records})
        score_state_options = sorted({record["score_state"] for record in records})
        state_keys = _job_filter_state_keys(slug)

        if state_keys["sources"] not in st.session_state:
            reset_job_filter_state(slug, source_options, status_options, score_state_options)

        with st.expander("Review controls", expanded=False):
            action_cols = st.columns([1.3, 1.15, 1.0, 0.85], gap="medium")
            search = action_cols[0].text_input(
                "Search jobs",
                key=state_keys["search"],
                placeholder="Title, company, location, summary, or keywords",
            )
            min_fit = action_cols[1].slider(
                "Minimum match score",
                min_value=0,
                max_value=100,
                step=5,
                key=state_keys["min_fit"],
                help="Only show jobs with a match score at or above this value.",
            )
            include_full_text = action_cols[2].checkbox(
                "Search full description",
                key=state_keys["include_full_text"],
                help="Include saved job description text in keyword search. Useful for specific skills or requirements.",
            )
            clear_filters = action_cols[3].button("Reset filters", key=f"reset_filters_{slug}", use_container_width=True)

            if clear_filters:
                reset_job_filter_state(slug, source_options, status_options, score_state_options)
                st.rerun()

            st.markdown("##### Filters")
            filter_cols = st.columns(3, gap="medium")
            selected_sources = filter_cols[0].multiselect(
                "Sources",
                source_options,
                key=state_keys["sources"],
                help="Choose one or more job sources to include.",
            )
            selected_statuses = filter_cols[1].multiselect(
                "Statuses",
                status_options,
                key=state_keys["statuses"],
                help="Choose one or more workflow statuses to show.",
            )
            selected_score_states = filter_cols[2].multiselect(
                "Score states",
                score_state_options,
                key=state_keys["score_states"],
                help="Choose one or more scoring states to include.",
            )

            st.checkbox(
                "Show scrape-filtered jobs",
                key=state_keys["show_scrape_rejected"],
                help="Jobs rejected during scraping before any LLM scoring — usually title mismatches or YOE filters.",
            )

            with st.expander("Show/hide columns", expanded=False):
                visible_columns = st.multiselect(
                    "Visible optional columns",
                    JOB_VISIBLE_OPTIONAL_COLUMNS,
                    key=state_keys["visible_columns"],
                    help="Choose which optional columns appear in the review table.",
                )

        selected_sources = st.session_state[state_keys["sources"]]
        selected_statuses = st.session_state[state_keys["statuses"]]
        selected_score_states = st.session_state[state_keys["score_states"]]
        min_fit = st.session_state[state_keys["min_fit"]]
        search = st.session_state[state_keys["search"]]
        include_full_text = st.session_state[state_keys["include_full_text"]]
        visible_columns = st.session_state[state_keys["visible_columns"]]
        show_scrape_rejected = st.session_state.get(state_keys["show_scrape_rejected"], False)

        filter_chips = build_jobs_filter_chips(
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
        _render_html_block("<div class='filter-chip-title'>Applied filters</div>")
        chip_row(filter_chips)

        pool = list(records)
        if show_scrape_rejected and scrape_rejected_records:
            pool.extend(scrape_rejected_records)
        filtered = pool
        if selected_sources:
            filtered = [record for record in filtered if record["source_label"] in selected_sources]
        if selected_statuses:
            filtered = [record for record in filtered if record["status_label"] in selected_statuses]
        if selected_score_states:
            filtered = [record for record in filtered if record["score_state"] in selected_score_states]
        if min_fit > 0:
            filtered = [record for record in filtered if record["fit_score"] is not None and record["fit_score"] >= min_fit]
        if search.strip():
            query = search.lower().strip()
            raw_text_match_ids = _search_job_ids_by_raw_text(slug, query) if include_full_text else set()
            filtered = [
                record
                for record in filtered
                if (
                    query in " ".join(
                        [record["title"], record["company"], record["location"] or "", record["source_label"], record["one_liner"]]
                    ).lower()
                    or record["id"] in raw_text_match_ids
                )
            ]

        if not filtered:
            clicked = empty_state(
                "No jobs match the current filters",
                "Clear one or more applied filters, lower the minimum fit, or search less narrowly to rebuild the review queue.",
                actions=[
                    {
                        "id": "clear_filtered_empty",
                        "label": "Reset filters",
                        "type": "primary",
                        "key": f"clear_filtered_empty_{slug}",
                    }
                ],
            )
            if clicked == "clear_filtered_empty":
                reset_job_filter_state(slug, source_options, status_options, score_state_options)
                st.rerun()
            return

        _render_html_block(
            (
                "<div class='jobs-workspace-summary'>"
                f"<div><span>Visible results</span><strong>{len(filtered)} / {len(records)}</strong></div>"
                f"<div><span>Scored in view</span><strong>{sum(1 for record in filtered if record['fit_score'] is not None)}</strong></div>"
                f"<div><span>Needs attention</span><strong>{sum(1 for record in filtered if record['score_state'] in {'Failed', 'Needs retry'})}</strong></div>"
                "</div>"
            )
        )

        table_col, detail_col = st.columns([1.32, 1.02], gap="large")
        frame = build_jobs_table_frame(filtered)
        column_order = resolve_job_table_columns(visible_columns)

        with table_col:
            with panel("Review queue", subtitle="Select rows for bulk actions. The first selection opens in the detail workspace"):
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
                selected_ids = [str(frame.iloc[row]["id"]) for row in selected_rows if 0 <= row < len(frame)]
                _render_html_block(
                    (
                        "<div class='selection-banner'>"
                        f"<div>{selected_ids and f'{len(selected_ids)} jobs selected for bulk actions' or 'No jobs selected yet'}</div>"
                        f"<div>{'The first selected row opens in detail.' if selected_ids else 'Pick one row to inspect details, or multi-select for bulk updates.'}</div>"
                        "</div>"
                    )
                )
                bulk_clicked = toolbar(
                    primary_actions=[],
                    secondary_actions=[
                        {"id": "bulk_applied", "label": "Mark applied", "key": f"bulk_applied_{slug}", "disabled": not selected_ids},
                        {"id": "bulk_skipped", "label": "Mark skipped", "key": f"bulk_skipped_{slug}", "disabled": not selected_ids},
                        {"id": "bulk_new", "label": "Mark new", "key": f"bulk_new_{slug}", "disabled": not selected_ids},
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
                    meta="Bulk actions apply immediately to the selected rows.",
                )

                if bulk_clicked == "bulk_undo":
                    restored = _undo_last_status_change(slug)
                    if restored:
                        _set_notice(slug, "success", f"Restored {restored} status change(s).")
                        st.rerun()
                elif bulk_clicked in {"bulk_applied", "bulk_skipped", "bulk_new"}:
                    target_status = {"bulk_applied": "applied", "bulk_skipped": "skipped", "bulk_new": "new"}[bulk_clicked]
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
                with panel("Inspection workspace", subtitle="Choose one row to open a structured review view"):
                    empty_state(
                        "Select a job",
                        "The detail workspace will show scores, reasons, watchouts, workflow controls, and secondary raw text without losing the table context.",
                    )
            else:
                if len(selected_rows) > 1:
                    with panel("Selection overview", subtitle="Bulk actions apply to all selected rows while detail stays locked to the first row"):
                        callout("info", "Bulk selection active", f"{len(selected_rows)} jobs are selected. The first selected role is shown in detail below.")
                _render_job_detail(selected_detail, slug)


def _render_activity_tab(
    slug: str,
    runs: list[dict[str, Any]],
    metrics: dict[str, Any],
    raw_config: dict[str, Any],
    config: dict[str, Any],
) -> None:
    with section_shell("Activity", SECTION_COPY["Activity"]):
        left, right = st.columns([1.18, 0.82], gap="large")
        with left:
            with panel("Latest run", subtitle="Most recent activity, simplified to the essentials"):
                _render_last_run_summary(runs, history_expander_label="View all runs")

        with right:
            _render_review_status_card(slug, metrics, raw_config)

        disq_count = metrics.get("disqualified_count", 0)
        if disq_count > 0:
            disq_label = f"Filtered out ({disq_count} job{'s' if disq_count != 1 else ''} hidden by scoring rules)"
            with st.expander(disq_label, expanded=False):
                st.caption(
                    "These jobs were disqualified by hard-no rules during scoring (e.g. "
                    "intern-only, location mismatch, YOE mismatch, or title blocklist). "
                    "They are stored in the database but excluded from the review queue and metrics."
                )
                by_reason = metrics.get("disqualified_by_reason", {})
                if by_reason:
                    rows = sorted(by_reason.items(), key=lambda kv: -kv[1])
                    _render_summary_list([(reason, f"{count}×") for reason, count in rows])
                else:
                    st.write("No breakdown available.")

        scrape_rej_count = metrics.get("scrape_rejected_count", 0)
        if scrape_rej_count > 0:
            rej_label = f"Scrape-filtered ({scrape_rej_count} job{'s' if scrape_rej_count != 1 else ''} rejected before scoring)"
            with st.expander(rej_label, expanded=False):
                st.caption(
                    "These jobs were rejected during scraping by pre-LLM filters (title blocklist, "
                    "YOE limits, location check). They are stored in the database for auditability "
                    "but never sent to the LLM. Enable 'Show scrape-filtered jobs' in the Jobs tab to inspect them."
                )
                rej_by_reason = metrics.get("scrape_rejected_by_reason", {})
                if rej_by_reason:
                    rows = sorted(rej_by_reason.items(), key=lambda kv: -kv[1])
                    _render_summary_list([(reason, f"{count}×") for reason, count in rows])
                else:
                    st.write("No breakdown available.")

        with st.expander("Advanced details", expanded=False):
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


def _render_profile_tab(slug: str, config: dict[str, Any], raw_config: dict[str, Any]) -> None:
    profile_cfg = config.get("profile", {})
    prefs = config.get("preferences", {})
    loc = prefs.get("location", {})
    compensation = prefs.get("compensation", {})
    job_type = profile_cfg.get("job_type", "fulltime")
    is_intern = job_type == "internship"
    with section_shell("Profile", SECTION_COPY["Profile"]):
        stat_row(
            [
                ("Profile type", job_type.replace("_", " ").title(), "Current search mode"),
                ("Target titles", len(prefs.get("titles", [])), "Roles the system prioritizes"),
                ("Desired skills", len(prefs.get("desired_skills", [])), "Skills used for matching"),
                ("Sources enabled", sum(1 for cfg in raw_config.get("sources", {}).values() if cfg.get("enabled", False)), "Providers actively searched"),
            ],
            columns_count=2,
        )
        _render_html_block("<div class='shell-panel-gap' aria-hidden='true'></div>")

        top = st.columns(2, gap="large")
        with top[0]:
            with panel("Candidate summary", subtitle="The candidate context passed into scoring and review flows"):
                resume_display = f"{len(profile_cfg.get('resume') or '')} characters inline"
                resume_pdf_bytes: bytes | None = None
                resume_file_name: str | None = None
                resume_value = profile_cfg.get("resume_file")
                if resume_value:
                    name_parts = [part for part in str(profile_cfg.get("name", "")).strip().split() if part]
                    if len(name_parts) >= 2:
                        friendly_filename = f"{name_parts[-1]}_{name_parts[0]}_resume.pdf"
                    elif name_parts:
                        friendly_filename = f"{name_parts[0]}_resume.pdf"
                    else:
                        friendly_filename = "resume.pdf"
                    resume_display = f"Auto-renamed to {friendly_filename}"
                    resume_file_name = friendly_filename
                    try:
                        profile_dir = PROFILES_DIR / slug
                        resolved_resume = _resolve_resume_path(str(resume_value), profile_dir)
                        resume_pdf_bytes = resolved_resume.read_bytes()
                    except (FileNotFoundError, OSError):
                        resume_display = f"Configured as {resume_value}"

                callout("info", "Profile summary", profile_cfg.get("bio") or "Add a short bio in Settings to give the scorer more context.")
                summary_rows = [
                    ("Name", profile_cfg.get("name", "Unknown")),
                    ("Job type", job_type.replace("_", " ").title()),
                    ("Resume source", resume_display),
                ]
                if is_intern:
                    summary_rows.extend(
                        [
                            ("School", profile_cfg.get("school", "") or "Not set"),
                            ("Major", profile_cfg.get("major", "") or "Not set"),
                            ("Graduation year", str(profile_cfg.get("graduation_year", "") or "Not set")),
                            ("Target season", profile_cfg.get("target_season", "") or "Not set"),
                        ]
                    )
                _render_html_block(
                    (
                        "<div class='summary-list'>"
                        + "".join(
                            f"<div class='summary-row'><span>{html.escape(str(label))}</span><strong>{html.escape(str(value))}</strong></div>"
                            for label, value in summary_rows
                        )
                        + "</div>"
                    )
                )
                if resume_pdf_bytes and resume_file_name:
                    preview_cols = st.columns([0.9, 1.1], gap="small")
                    with preview_cols[0]:
                        if st.button("Preview resume PDF", key=f"preview_resume_{slug}", use_container_width=True):
                            _open_resume_preview_dialog(resume_file_name, resume_pdf_bytes)
                    with preview_cols[1]:
                        st.download_button(
                            "Download PDF",
                            data=resume_pdf_bytes,
                            file_name=resume_file_name,
                            mime="application/pdf",
                            key=f"download_resume_{slug}",
                            use_container_width=True,
                        )

        with top[1]:
            with panel("Search preferences", subtitle="Location and compensation signals that shape ranking and fit"):
                compensation_rows = [
                    ("Remote", "Open" if loc.get("remote_ok", True) else "Not preferred"),
                    ("Preferred locations", ", ".join(loc.get("preferred_locations", [])) or "None set"),
                    *_compensation_rows(job_type, compensation),
                ]
                _render_html_block(
                    (
                        "<div class='summary-list'>"
                        + "".join(
                            f"<div class='summary-row'><span>{html.escape(str(label))}</span><strong>{html.escape(str(value))}</strong></div>"
                            for label, value in compensation_rows
                        )
                        + "</div>"
                    )
                )

        bottom = st.columns(2, gap="large")
        with bottom[0]:
            with panel("Target roles and skills", subtitle="What the system is actively optimizing for"):
                st.caption("Target titles")
                chip_row(prefs.get("titles", []) or ["None saved"])
                st.caption("Desired skills")
                chip_row(prefs.get("desired_skills", []) or ["None saved"])

        with bottom[1]:
            with panel("Source configuration", subtitle="Enabled source coverage for future runs"):
                source_lines: list[str] = []
                for source_name, source_cfg in raw_config.get("sources", {}).items():
                    if source_name in {"greenhouse", "lever", "ashby", "workable"}:
                        companies = source_cfg.get("companies", [])
                        source_lines.append(
                            f"{source_name.title()}: {'Enabled' if source_cfg.get('enabled', True) else 'Off'} ({len(companies)} companies)"
                        )
                    else:
                        source_lines.append(f"{source_name.title()}: {'Enabled' if source_cfg.get('enabled', False) else 'Off'}")
                if source_lines:
                    for line in source_lines:
                        st.write(f"- {line}")
                else:
                    empty_state("No source configuration found", "This profile does not have source settings yet.")


def _lines_to_list(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _render_settings_tab(slug: str, config: dict[str, Any], raw_config: dict[str, Any], metrics: dict[str, Any]) -> None:
    editable = copy.deepcopy(raw_config)
    job_type = editable.get("profile", {}).get("job_type", "fulltime")
    is_intern = job_type == "internship"
    editable.setdefault("scoring", {})
    editable.setdefault("preferences", {})
    editable["preferences"].setdefault("location", {})
    editable["preferences"].setdefault("compensation", {})
    editable.setdefault("sources", {})
    editable["sources"].setdefault("greenhouse", {"enabled": True,  "companies": []})
    editable["sources"].setdefault("lever",      {"enabled": True,  "companies": []})
    editable["sources"].setdefault("hn",         {"enabled": False})
    editable["sources"].setdefault("ashby",      {"enabled": False, "companies": []})
    editable["sources"].setdefault("workable",   {"enabled": False, "companies": []})
    editable["sources"].setdefault("himalayas",  {"enabled": False})
    with section_shell("Settings", SECTION_COPY["Settings"]):
        location_defaults = editable["preferences"]["location"].get("preferred_locations", [])
        location_options = sorted({*LOCATION_PICKER_OPTIONS, *location_defaults})
        compensation = editable["preferences"]["compensation"]
        salary_value = int(compensation.get("min_salary", 0) or 0)
        stipend_value = int(compensation.get("monthly_stipend", 0) or 0)
        intern_pay_preference = _normalize_intern_pay_preference(compensation)

        with st.expander("Search preferences", expanded=True):
            st.caption("Keep the shortlist aligned with where and how you want to work.")
            st.markdown("##### Scoring")
            min_display = st.slider(
                "Minimum display score",
                min_value=0,
                max_value=100,
                value=int(editable["scoring"].get("min_display_score", 60)),
                step=5,
                help="Jobs below this score stay saved, but they will not appear in the main shortlist views.",
            )
            with st.expander("About display score", expanded=False):
                st.write("Lower the display score only if you want to inspect weaker-fit roles more often.")

            pref_cols = st.columns([1.0, 1.1], gap="large")
            with pref_cols[0]:
                st.markdown("##### Location")
                remote_ok = st.checkbox(
                    "Remote OK",
                    value=bool(editable["preferences"]["location"].get("remote_ok", True)),
                )
                preferred_locations = st.multiselect(
                    "Preferred locations",
                    options=location_options,
                    default=[loc for loc in location_defaults if loc in location_options],
                    help="Choose the locations you want discovery to prioritize.",
                )
            with pref_cols[1]:
                st.markdown("##### Sources")
                gh_enabled = st.checkbox(
                    "Greenhouse",
                    value=bool(editable["sources"]["greenhouse"].get("enabled", True)),
                )
                lv_enabled = st.checkbox(
                    "Lever",
                    value=bool(editable["sources"]["lever"].get("enabled", True)),
                )
                ash_enabled = st.checkbox(
                    "Ashby",
                    value=bool(editable["sources"]["ashby"].get("enabled", False)),
                )
                wl_enabled = st.checkbox(
                    "Workable",
                    value=bool(editable["sources"]["workable"].get("enabled", False)),
                )
                hn_enabled = st.checkbox(
                    "HN Who's Hiring",
                    value=bool(editable["sources"]["hn"].get("enabled", False)),
                )
                him_enabled = st.checkbox(
                    "Himalayas (remote-only)",
                    value=bool(editable["sources"]["himalayas"].get("enabled", False)),
                )

        with st.expander("Matching rules", expanded=False):
            st.caption("Fine-tune titles, skills, exclusions, and source company lists when you need more control.")
            titles_default = editable["preferences"].get("titles", [])
            skills_default = editable["preferences"].get("desired_skills", [])
            hard_no_default = editable["preferences"].get("hard_no_keywords", [])
            gh_companies_default  = editable["sources"]["greenhouse"].get("companies", [])
            lv_companies_default  = editable["sources"]["lever"].get("companies", [])
            ash_companies_default = editable["sources"]["ashby"].get("companies", [])
            wl_companies_default  = editable["sources"]["workable"].get("companies", [])
            titles_text   = "\n".join(titles_default)
            skills_text   = "\n".join(skills_default)
            hard_no_text  = "\n".join(hard_no_default)
            gh_companies  = "\n".join(gh_companies_default)
            lv_companies  = "\n".join(lv_companies_default)
            ash_companies = "\n".join(ash_companies_default)
            wl_companies  = "\n".join(wl_companies_default)
            edit_matching_rules = st.toggle(
                "Edit matching rules",
                key=f"edit_matching_rules_{slug}",
                help="Switch between a compact summary and the full editor for titles, skills, exclusions, and source company lists.",
            )
            if is_intern:
                minimum_salary = None
                monthly_stipend = stipend_value if intern_pay_preference == "paid_only" else None
                compensation_summary = _format_compensation_value(job_type, compensation)
                compensation_label = "Compensation"
            else:
                minimum_salary = salary_value
                monthly_stipend = None
                compensation_summary = f"${salary_value:,}" if salary_value else "Not set"
                compensation_label = "Minimum salary"

            if not edit_matching_rules:
                _render_summary_list(
                    [
                        ("Target titles",       len(titles_default)),
                        ("Desired skills",       len(skills_default)),
                        ("Hard-no keywords",     len(hard_no_default)),
                        (compensation_label,     compensation_summary),
                        ("Greenhouse companies", len(gh_companies_default)),
                        ("Lever companies",      len(lv_companies_default)),
                        ("Ashby companies",      len(ash_companies_default)),
                        ("Workable companies",   len(wl_companies_default)),
                    ]
                )
                st.caption("Target titles")
                chip_row(titles_default[:8] or ["None saved"])
                st.caption("Desired skills")
                chip_row(skills_default[:8] or ["None saved"])
                st.caption("Hard-no keywords")
                chip_row(hard_no_default[:6] or ["None saved"])
            else:
                titles_text = st.text_area("Target titles", value=titles_text, height=120, help="One title per line.")
                skills_text = st.text_area("Desired skills", value=skills_text, height=120, help="One skill per line.")
                hard_no_text = st.text_area("Hard-no keywords", value=hard_no_text, height=100, help="Any posting containing these phrases will be skipped early.")
                if is_intern:
                    preference_labels = {
                        "paid_only": "Paid only",
                        "unpaid_ok": "Unpaid OK",
                        "no_preference": "No preference",
                    }
                    selected_label = st.radio(
                        "Compensation preference",
                        list(preference_labels.values()),
                        index=list(preference_labels.keys()).index(intern_pay_preference),
                        horizontal=True,
                    )
                    intern_pay_preference = {label: value for value, label in preference_labels.items()}[selected_label]
                    if intern_pay_preference == "paid_only":
                        monthly_stipend = st.text_input(
                            "Monthly stipend target",
                            value=_optional_int_input_value(compensation.get("monthly_stipend")),
                            placeholder="4500",
                            help="Optional. Leave blank if the internship just needs to be paid.",
                        )
                    else:
                        monthly_stipend = None
                    minimum_salary = None
                else:
                    minimum_salary = st.number_input("Minimum salary", min_value=0, step=5000, value=salary_value, help="Used as a soft scoring preference, not a hard filter.")
                    monthly_stipend = None
                gh_companies  = st.text_area("Greenhouse companies",  value=gh_companies,  height=140, help="One company slug per line.")
                lv_companies  = st.text_area("Lever companies",        value=lv_companies,  height=140, help="One company slug per line.")
                ash_companies = st.text_area("Ashby companies",        value=ash_companies, height=140, help="One company slug per line.")
                wl_companies  = st.text_area("Workable companies",     value=wl_companies,  height=140, help="One company slug per line.")

        submitted = st.button(
            "Save settings",
            key=f"save_settings_{slug}",
            type="primary",
            help="Writes the updated config to this profile only.",
        )

        if submitted:
            if is_intern:
                try:
                    parsed_monthly_stipend = _parse_optional_int_input(monthly_stipend) if intern_pay_preference == "paid_only" else None
                except ValueError:
                    callout("error", "Settings need one more detail", "Monthly stipend target must be a whole number or left blank.")
                    return
            editable["scoring"]["min_display_score"] = int(min_display)
            editable["preferences"]["titles"] = _lines_to_list(titles_text)
            editable["preferences"]["desired_skills"] = _lines_to_list(skills_text)
            editable["preferences"]["hard_no_keywords"] = _lines_to_list(hard_no_text)
            editable["preferences"]["location"]["remote_ok"] = remote_ok
            editable["preferences"]["location"]["preferred_locations"] = list(preferred_locations)
            if is_intern:
                editable["preferences"]["compensation"] = {
                    "intern_pay_preference": intern_pay_preference,
                }
                if intern_pay_preference == "paid_only" and parsed_monthly_stipend is not None:
                    editable["preferences"]["compensation"]["monthly_stipend"] = parsed_monthly_stipend
            else:
                editable["preferences"]["compensation"] = {"min_salary": int(minimum_salary or 0)}
            editable["sources"]["greenhouse"]["enabled"]  = gh_enabled
            editable["sources"]["greenhouse"]["companies"] = _lines_to_list(gh_companies)
            editable["sources"]["lever"]["enabled"]       = lv_enabled
            editable["sources"]["lever"]["companies"]     = _lines_to_list(lv_companies)
            editable["sources"]["ashby"]["enabled"]       = ash_enabled
            editable["sources"]["ashby"]["companies"]     = _lines_to_list(ash_companies)
            editable["sources"]["workable"]["enabled"]    = wl_enabled
            editable["sources"]["workable"]["companies"]  = _lines_to_list(wl_companies)
            editable["sources"]["hn"]["enabled"]          = hn_enabled
            editable["sources"]["himalayas"]["enabled"]   = him_enabled

            if not any((gh_enabled, lv_enabled, ash_enabled, wl_enabled, hn_enabled, him_enabled)):
                callout("error", "At least one source is required", "Enable at least one source before saving.")
                return

            _write_profile_config(slug, editable)
            _set_notice(slug, "success", "Settings saved for this profile.")
            invalidate_dashboard_caches()
            st.rerun()

        with st.expander("Profile actions", expanded=False):
            callout("info", "Profile management", "Create a fresh workspace or refresh scores without removing saved jobs.")
            profile_action_cols = st.columns([0.95, 1.05], gap="large")
            with profile_action_cols[0]:
                if st.button("Create new profile", key=f"settings_create_profile_{slug}", type="secondary", help="Open a quick profile-creation dialog."):
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
            _render_jobs_tab(records, slug, scrape_rejected_records=scrape_rejected_records)
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
