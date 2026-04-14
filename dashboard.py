from __future__ import annotations

import copy
import html
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
import yaml
from dotenv import load_dotenv

from config import load_config
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
from logging_config import configure_logging
from onboarding import render_onboarding
from scraper import scrape_greenhouse, scrape_hn, scrape_lever
from scorer import RateLimitReached, score_all_jobs
from theirstack import get_or_discover_slugs

load_dotenv()

st.set_page_config(
    page_title="Job Search Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).parent
PROFILES_DIR = BASE_DIR / "profiles"

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
}


def _init_state() -> None:
    defaults = {
        "active_profile": None,
        "show_onboarding": False,
        "onboarding_step": 1,
        "onboarding_data": {},
        "dashboard_notice": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _profile_config_path(slug: str) -> Path:
    return PROFILES_DIR / slug / "config.yaml"


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
            ]
        )

    for column in ("Fit", "ATS"):
        frame[column] = pd.array(frame[column], dtype="Int64")
    return frame


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
            --bg-ink: #0f172a;
            --bg-panel: rgba(255, 255, 255, 0.88);
            --line: rgba(15, 23, 42, 0.10);
            --text: #0f172a;
            --muted: #52607a;
            --accent: #0f766e;
            --accent-soft: rgba(15, 118, 110, 0.12);
            --danger: #b42318;
            --warning: #b54708;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(14, 116, 144, 0.12), transparent 24%),
                linear-gradient(180deg, #f8fbfb 0%, #f2f6f9 100%);
            color: var(--text);
            font-family: 'IBM Plex Sans', 'Aptos', 'Segoe UI', sans-serif;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }

        h1, h2, h3 {
            font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
            letter-spacing: -0.02em;
        }

        div[data-testid="stMetric"] {
            background: var(--bg-panel);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 0.75rem 1rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.04);
        }

        .hero-shell {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(15, 118, 110, 0.92));
            border-radius: 28px;
            padding: 1.5rem 1.75rem;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.10);
            box-shadow: 0 24px 50px rgba(15, 23, 42, 0.16);
            margin-bottom: 1rem;
        }

        .hero-kicker {
            text-transform: uppercase;
            font-size: 0.78rem;
            letter-spacing: 0.12em;
            opacity: 0.78;
            margin-bottom: 0.35rem;
        }

        .hero-title {
            font-size: 2rem;
            line-height: 1.05;
            margin: 0;
        }

        .hero-copy {
            margin-top: 0.8rem;
            color: rgba(255, 255, 255, 0.86);
            max-width: 54rem;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .chip {
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 999px;
            padding: 0.35rem 0.75rem;
            font-size: 0.9rem;
        }

        .panel {
            background: var(--bg-panel);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.04);
        }

        .match-card {
            background: var(--bg-panel);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.75rem;
        }

        .match-title {
            font-weight: 600;
            font-size: 1.05rem;
            margin-bottom: 0.2rem;
            color: var(--text);
        }

        .match-meta {
            color: var(--muted);
            font-size: 0.92rem;
            margin-bottom: 0.45rem;
        }

        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.55rem;
        }

        .badge {
            display: inline-block;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.84rem;
            border: 1px solid rgba(15, 118, 110, 0.18);
        }

        .badge.warn {
            color: var(--warning);
            background: rgba(181, 71, 8, 0.08);
            border-color: rgba(181, 71, 8, 0.16);
        }

        .badge.fail {
            color: var(--danger);
            background: rgba(180, 35, 24, 0.08);
            border-color: rgba(180, 35, 24, 0.16);
        }

        .profile-card {
            background: var(--bg-panel);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.15rem;
            min-height: 220px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.04);
        }

        .profile-meta {
            color: var(--muted);
            margin-bottom: 1rem;
            font-size: 0.92rem;
        }

        .stApp a:focus-visible,
        .stApp button:focus-visible,
        .stApp [role="button"]:focus-visible,
        .stApp input:focus-visible,
        .stApp textarea:focus-visible,
        .stApp [tabindex]:focus-visible {
            outline: 3px solid rgba(15, 118, 110, 0.45);
            outline-offset: 3px;
        }

        div[data-testid="stButton"],
        div[data-testid="stLinkButton"] {
            margin-top: 0.2rem;
            margin-bottom: 0.35rem;
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
                scroll-behavior: auto !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
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
        "lever": sources.get("lever", {}).get("enabled", True),
        "hackernews": sources.get("hn", {}).get("enabled", False),
    }


def _provider_model(config: dict[str, Any]) -> str:
    llm_cfg = config.get("llm", {})
    provider = llm_cfg.get("provider", "unknown")
    models = llm_cfg.get("model", {})
    return str(models.get(provider, "unknown"))


def _check_api_key(config: dict[str, Any]) -> Optional[str]:
    provider = config.get("llm", {}).get("provider")
    env_var = PROVIDER_ENV_VARS.get(str(provider))
    if env_var and not os.environ.get(env_var):
        return env_var
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
    renderer = getattr(st, kind, st.info)
    renderer(message)
    st.session_state.dashboard_notice = None


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
                s.scored_at
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
                s.scored_at
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
    counts = count_jobs(profile=slug)
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

    fit_scores = [record["fit_score"] for record in records if record["fit_score"] is not None]
    avg_fit = round(sum(fit_scores) / len(fit_scores), 1) if fit_scores else 0.0

    return {
        "total": counts["total"],
        "scored": scored,
        "pending": pending,
        "failed": failed,
        "needs_retry": retries,
        "applied": applied,
        "skipped": skipped,
        "avg_fit": avg_fit,
        "source_counts": source_counts,
    }


def _run_pipeline(
    config: dict[str, Any],
    slug: str,
    status_box: Any | None = None,
) -> dict[str, Any]:
    enabled = _enabled_sources(config)
    if not any(enabled.values()):
        raise ValueError("No sources are enabled for this profile.")

    run_id = start_run(profile=slug, source="dashboard")
    slug_map = {"greenhouse": [], "lever": []}
    total_new = 0
    total_scraped = 0
    total_filtered = 0
    total_saved = 0
    scored_results: list[dict[str, Any]] = []

    def log(message: str) -> None:
        if status_box is not None:
            status_box.write(message)
        else:
            st.write(message)

    try:
        if enabled["greenhouse"] or enabled["lever"]:
            log("Resolving company lists and cached ATS slugs...")
            slug_map = get_or_discover_slugs(config, profile=slug)

        if enabled["greenhouse"]:
            result = scrape_greenhouse(config, slugs=slug_map["greenhouse"], profile=slug)
            total_new += result.get("new_jobs_saved", 0)
            total_scraped += result.get("jobs_scraped", 0)
            total_filtered += result.get("jobs_filtered", 0)
            total_saved += result.get("new_jobs_saved", 0)
            log(f"Greenhouse saved {result.get('new_jobs_saved', 0)} new jobs.")

        if enabled["lever"]:
            result = scrape_lever(config, slugs=slug_map["lever"], profile=slug)
            total_new += result.get("new_jobs_saved", 0)
            total_scraped += result.get("jobs_scraped", 0)
            total_filtered += result.get("jobs_filtered", 0)
            total_saved += result.get("new_jobs_saved", 0)
            log(f"Lever saved {result.get('new_jobs_saved', 0)} new jobs.")

        if enabled["hackernews"]:
            result = scrape_hn(config, profile=slug)
            total_new += result.get("new_jobs_saved", 0)
            total_scraped += result.get("jobs_scraped", 0)
            total_filtered += result.get("jobs_filtered", 0)
            total_saved += result.get("new_jobs_saved", 0)
            log(f"HN Who's Hiring saved {result.get('new_jobs_saved', 0)} new jobs.")

        log("Scoring newly discovered jobs...")
        scored_results = score_all_jobs(config, yes=True, profile=slug)
        scored_ok = [result for result in scored_results if result.get("fit_score", 0) > 0]
        avg_fit = round(
            sum(result["fit_score"] for result in scored_ok) / len(scored_ok),
            1,
        ) if scored_ok else 0.0

        finish_run(
            run_id,
            jobs_scraped=total_scraped,
            jobs_filtered=total_filtered,
            jobs_saved=total_saved,
            jobs_scored=len(scored_ok),
            avg_fit_score=avg_fit,
            errors=[],
            status="complete",
            profile=slug,
        )

        summary = {
            "total_new": total_new,
            "jobs_scraped": total_scraped,
            "jobs_filtered": total_filtered,
            "jobs_saved": total_saved,
            "scored_count": len(scored_ok),
            "avg_fit": avg_fit,
        }
        if status_box is not None:
            status_box.update(
                label=(
                    f"Pipeline complete: {summary['total_new']} new jobs, "
                    f"{summary['scored_count']} scored"
                ),
                state="complete",
                expanded=False,
            )
        return summary

    except RateLimitReached as exc:
        finish_run(
            run_id,
            jobs_scraped=total_scraped,
            jobs_filtered=total_filtered,
            jobs_saved=total_saved,
            jobs_scored=0,
            avg_fit_score=None,
            errors=[str(exc)],
            status="failed",
            profile=slug,
        )
        if status_box is not None:
            status_box.update(label="Pipeline stopped by rate limits", state="error", expanded=True)
        raise

    except Exception as exc:
        finish_run(
            run_id,
            jobs_scraped=total_scraped,
            jobs_filtered=total_filtered,
            jobs_saved=total_saved,
            jobs_scored=0,
            avg_fit_score=None,
            errors=[str(exc)],
            status="failed",
            profile=slug,
        )
        if status_box is not None:
            status_box.update(label="Pipeline failed", state="error", expanded=True)
        raise


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
    enabled_labels = [
        label
        for source, label in SOURCE_LABELS.items()
        if enabled.get(source, False)
    ]
    profile_cfg = config.get("profile", {})
    provider = config.get("llm", {}).get("provider", "unknown")
    model = _provider_model(config)
    bio = html.escape((profile_cfg.get("bio") or "").strip() or "No profile bio saved yet.")
    chip_values = [
        profile_cfg.get("job_type", "fulltime").replace("_", " ").title(),
        f"{provider.title()} / {model}",
        f"{metrics['total']} jobs tracked",
        f"{metrics['scored']} scored",
        ", ".join(enabled_labels) if enabled_labels else "No sources enabled",
    ]
    chips = "".join(
        f"<span class='chip'>{html.escape(str(value))}</span>"
        for value in chip_values
        if value
    )
    st.markdown(
        f"""
        <section class="hero-shell">
            <div class="hero-kicker">Profile Dashboard</div>
            <h1 class="hero-title">{html.escape(profile_name)}</h1>
            <p class="hero-copy">{bio}</p>
            <div class="chip-row">{chips}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_top_matches(records: list[dict[str, Any]]) -> None:
    scored = [record for record in records if record["fit_score"] is not None]
    if not scored:
        st.info("No scored jobs yet. Run the pipeline to populate your top matches.")
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
        st.info("No run history yet. The first pipeline run will appear here.")
        return

    frame = pd.DataFrame(
        [
            {
                "Started": run.get("started_at", ""),
                "Finished": run.get("finished_at", ""),
                "Status": str(run.get("status", "")).title(),
                "Saved": run.get("jobs_saved", 0),
                "Scored": run.get("jobs_scored", 0),
                "Avg Fit": run.get("avg_fit_score", 0) or 0,
                "Source": run.get("source", ""),
                "Errors": "; ".join(run.get("errors", [])),
            }
            for run in runs
        ]
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)


def _render_job_detail(record: dict[str, Any], slug: str) -> None:
    st.subheader(record["title"])
    st.caption(
        f"{record['company']} · {record['location'] or 'Location not listed'} · "
        f"{record['source_label']} · {record['status_label']}"
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Fit", record["fit_score"] if record["fit_score"] is not None else "N/A")
    metric_cols[1].metric("ATS", record["ats_score"] if record["ats_score"] is not None else "N/A")
    metric_cols[2].metric("Score state", record["score_state"])
    metric_cols[3].metric("Attempts", record["score_attempts"] or 0)

    if record["one_liner"]:
        st.write(record["one_liner"])

    if record["reasons"]:
        st.write("Why it matched")
        for reason in record["reasons"]:
            st.write(f"- {reason}")

    if record["flags"]:
        st.write("Watchouts")
        for flag in record["flags"]:
            st.write(f"- {flag}")

    if record["skill_misses"]:
        st.write("Missing from resume")
        for skill in record["skill_misses"]:
            st.write(f"- {skill}")

    if record["score_error"]:
        st.error(record["score_error"])

    dims = {key: value for key, value in record["dimension_scores"].items() if value is not None}
    if dims:
        st.bar_chart(pd.DataFrame([dims]).T.rename(columns={0: "Score"}))

    actions = st.columns(4, gap="medium")
    if actions[0].button("Mark new", key=f"mark_new_{record['id']}"):
        update_job_status(record["id"], "new", profile=slug)
        invalidate_dashboard_caches()
        st.rerun()
    if actions[1].button("Mark applied", key=f"mark_applied_{record['id']}"):
        update_job_status(record["id"], "applied", profile=slug)
        invalidate_dashboard_caches()
        st.rerun()
    if actions[2].button("Mark skipped", key=f"mark_skipped_{record['id']}"):
        update_job_status(record["id"], "skipped", profile=slug)
        invalidate_dashboard_caches()
        st.rerun()
    if record["url"]:
        actions[3].link_button("Open posting", record["url"])

    with st.expander("Job text", expanded=False):
        st.text(record["raw_text"] or "")


def _render_overview_tab(
    slug: str,
    config: dict[str, Any],
    raw_config: dict[str, Any],
    records: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    metric_cols = st.columns(5)
    metric_cols[0].metric("Total jobs", metrics["total"])
    metric_cols[1].metric("Scored", metrics["scored"])
    metric_cols[2].metric("Pending", metrics["pending"])
    metric_cols[3].metric("Failed", metrics["failed"])
    metric_cols[4].metric("Applied", metrics["applied"])

    left, right = st.columns([1.25, 1.0], gap="large")
    with left:
        st.subheader("Top matches")
        _render_top_matches(records)

    with right:
        st.subheader("Pipeline health")
        with st.container(border=True):
            enabled = _enabled_sources(raw_config)
            st.write(
                f"Enabled sources: "
                f"{', '.join(label for source, label in SOURCE_LABELS.items() if enabled.get(source)) or 'None'}"
            )
            st.write(f"Average fit across scored jobs: {metrics['avg_fit']}/100")
            st.write(f"Jobs needing retry: {metrics['needs_retry']}")
            st.write(f"Skipped jobs: {metrics['skipped']}")

        st.subheader("Recent runs")
        _render_run_history(runs[:5])


def _render_jobs_tab(records: list[dict[str, Any]], slug: str) -> None:
    if not records:
        st.info("No jobs saved yet. Run the search to populate this profile.")
        return

    source_options = sorted({record["source_label"] for record in records})
    status_options = sorted({record["status_label"] for record in records})
    score_state_options = sorted({record["score_state"] for record in records})

    with st.expander("Filters", expanded=True):
        filter_cols = st.columns(4, gap="medium")
        selected_sources = filter_cols[0].multiselect("Source", source_options, default=source_options)
        selected_statuses = filter_cols[1].multiselect("Job status", status_options, default=status_options)
        selected_score_states = filter_cols[2].multiselect(
            "Score state",
            score_state_options,
            default=score_state_options,
        )
        min_fit = filter_cols[3].slider("Minimum fit", min_value=0, max_value=100, value=0, step=5)

        search_cols = st.columns([1.8, 1.0], gap="medium")
        search = search_cols[0].text_input(
            "Search jobs",
            placeholder="Title, company, location, summary, or keywords",
        )
        include_full_text = search_cols[1].checkbox(
            "Include full text in search (slower)",
            value=False,
        )

    filtered = records
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

    st.caption(f"Showing {len(filtered)} of {len(records)} jobs")

    if not filtered:
        st.warning("No jobs match the current filters.")
        return

    table_col, detail_col = st.columns([1.35, 1.0], gap="large")
    frame = build_jobs_table_frame(filtered)

    with table_col:
        st.caption(
            "Use the table search and filters to narrow jobs. Sorting or changing filters can "
            "clear the active selection in Streamlit."
        )
        # Selection indexes map to the currently rendered dataframe, so a sort/filter change can
        # reset the detail pane until the user picks a row again.
        selection = st.dataframe(
            frame,
            use_container_width=True,
            hide_index=True,
            key=f"jobs_table_{slug}",
            on_select="rerun",
            selection_mode="single-row",
            height=520,
            column_config={
                "id": None,
                "Fit": st.column_config.ProgressColumn("Fit", min_value=0, max_value=100),
                "ATS": st.column_config.ProgressColumn("ATS", min_value=0, max_value=100),
                "Posting": st.column_config.LinkColumn("Posting", display_text="Open"),
            },
        )

    selected_detail: Optional[dict[str, Any]] = None
    selected_rows = list(selection.selection.rows)
    if selected_rows:
        selected_row = selected_rows[0]
        if 0 <= selected_row < len(frame):
            selected_id = str(frame.iloc[selected_row]["id"])
            selected_detail = _cached_fetch_job_detail(slug, selected_id)

    with detail_col:
        st.subheader("Job detail")
        if selected_detail is None:
            st.info("Select a row from the table to review match reasons, watchouts, and job text.")
        else:
            _render_job_detail(selected_detail, slug)


def _render_activity_tab(slug: str, runs: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.subheader("Run log")
        _render_run_history(runs)

    with right:
        st.subheader("Source mix")
        source_counts = metrics["source_counts"]
        if source_counts:
            chart_df = pd.DataFrame(
                {"Source": list(source_counts.keys()), "Jobs": list(source_counts.values())}
            ).set_index("Source")
            st.bar_chart(chart_df)
        else:
            st.info("No source data yet.")

        st.subheader("Cached ATS slugs")
        gh_slugs = load_discovered_slugs(ats="greenhouse", profile=slug)
        lv_slugs = load_discovered_slugs(ats="lever", profile=slug)
        st.write(f"Greenhouse cache: {len(gh_slugs)}")
        st.write(f"Lever cache: {len(lv_slugs)}")


def _render_profile_tab(config: dict[str, Any], raw_config: dict[str, Any]) -> None:
    profile_cfg = config.get("profile", {})
    prefs = config.get("preferences", {})
    loc = prefs.get("location", {})
    compensation = prefs.get("compensation", {})

    top = st.columns(2, gap="large")
    with top[0]:
        st.subheader("Candidate summary")
        with st.container(border=True):
            st.write(profile_cfg.get("bio") or "No bio saved.")
            st.write(f"Name: {profile_cfg.get('name', 'Unknown')}")
            st.write(f"Job type: {profile_cfg.get('job_type', 'fulltime')}")
            if profile_cfg.get("resume_file"):
                st.write(f"Resume file: {profile_cfg['resume_file']}")
            else:
                resume_text = profile_cfg.get("resume") or ""
                st.write(f"Resume text stored inline: {len(resume_text)} characters")

    with top[1]:
        st.subheader("Search preferences")
        with st.container(border=True):
            st.write(f"Remote OK: {loc.get('remote_ok', True)}")
            st.write(
                "Preferred locations: "
                + (", ".join(loc.get("preferred_locations", [])) or "None")
            )
            if "min_salary" in compensation:
                st.write(f"Minimum salary: ${int(compensation['min_salary']):,}")
            if "monthly_stipend" in compensation:
                st.write(f"Monthly stipend target: ${int(compensation['monthly_stipend']):,}")

    st.subheader("Target titles")
    st.write(", ".join(prefs.get("titles", [])) or "None saved")

    st.subheader("Desired skills")
    st.write(", ".join(prefs.get("desired_skills", [])) or "None saved")

    st.subheader("Source configuration")
    source_lines: list[str] = []
    for source_name, source_cfg in raw_config.get("sources", {}).items():
        if source_name in {"greenhouse", "lever"}:
            companies = source_cfg.get("companies", [])
            source_lines.append(f"{source_name}: enabled={source_cfg.get('enabled', True)} ({len(companies)} companies)")
        else:
            source_lines.append(f"{source_name}: enabled={source_cfg.get('enabled', False)}")
    if source_lines:
        for line in source_lines:
            st.write(f"- {line}")
    else:
        st.write("No source configuration found.")


def _lines_to_list(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _render_settings_tab(slug: str, config: dict[str, Any], raw_config: dict[str, Any], metrics: dict[str, Any]) -> None:
    editable = copy.deepcopy(raw_config)
    editable.setdefault("scoring", {})
    editable.setdefault("preferences", {})
    editable["preferences"].setdefault("location", {})
    editable["preferences"].setdefault("compensation", {})
    editable.setdefault("sources", {})
    editable["sources"].setdefault("greenhouse", {"enabled": True, "companies": []})
    editable["sources"].setdefault("lever", {"enabled": True, "companies": []})
    editable["sources"].setdefault("hn", {"enabled": False})

    with st.form(f"settings_form_{slug}"):
        st.subheader("Search and scoring")
        min_display = st.slider(
            "Minimum display score",
            min_value=0,
            max_value=100,
            value=int(editable["scoring"].get("min_display_score", 60)),
            step=5,
        )
        titles_text = st.text_area(
            "Target titles",
            value="\n".join(editable["preferences"].get("titles", [])),
            height=120,
        )
        skills_text = st.text_area(
            "Desired skills",
            value="\n".join(editable["preferences"].get("desired_skills", [])),
            height=120,
        )
        hard_no_text = st.text_area(
            "Hard-no keywords",
            value="\n".join(editable["preferences"].get("hard_no_keywords", [])),
            height=100,
        )

        st.subheader("Location and compensation")
        remote_ok = st.checkbox(
            "Remote OK",
            value=bool(editable["preferences"]["location"].get("remote_ok", True)),
        )
        preferred_locations_text = st.text_area(
            "Preferred locations",
            value="\n".join(editable["preferences"]["location"].get("preferred_locations", [])),
            height=90,
        )

        compensation = editable["preferences"]["compensation"]
        salary_value = int(compensation.get("min_salary", 0))
        stipend_value = int(compensation.get("monthly_stipend", 0))
        if "min_salary" in compensation or "monthly_stipend" not in compensation:
            minimum_salary = st.number_input(
                "Minimum salary",
                min_value=0,
                step=5000,
                value=salary_value,
            )
            monthly_stipend = None
        else:
            monthly_stipend = st.number_input(
                "Monthly stipend target",
                min_value=0,
                step=500,
                value=stipend_value,
            )
            minimum_salary = None

        st.subheader("Sources")
        source_cols = st.columns(3)
        gh_enabled = source_cols[0].checkbox(
            "Greenhouse",
            value=bool(editable["sources"]["greenhouse"].get("enabled", True)),
        )
        lv_enabled = source_cols[1].checkbox(
            "Lever",
            value=bool(editable["sources"]["lever"].get("enabled", True)),
        )
        hn_enabled = source_cols[2].checkbox(
            "HN Who's Hiring",
            value=bool(editable["sources"]["hn"].get("enabled", False)),
        )

        gh_companies = st.text_area(
            "Greenhouse companies",
            value="\n".join(editable["sources"]["greenhouse"].get("companies", [])),
            height=160,
        )
        lv_companies = st.text_area(
            "Lever companies",
            value="\n".join(editable["sources"]["lever"].get("companies", [])),
            height=160,
        )

        submitted = st.form_submit_button("Save settings", type="primary")

    if submitted:
        editable["scoring"]["min_display_score"] = int(min_display)
        editable["preferences"]["titles"] = _lines_to_list(titles_text)
        editable["preferences"]["desired_skills"] = _lines_to_list(skills_text)
        editable["preferences"]["hard_no_keywords"] = _lines_to_list(hard_no_text)
        editable["preferences"]["location"]["remote_ok"] = remote_ok
        editable["preferences"]["location"]["preferred_locations"] = _lines_to_list(preferred_locations_text)
        if minimum_salary is not None:
            editable["preferences"]["compensation"]["min_salary"] = int(minimum_salary)
        if monthly_stipend is not None:
            editable["preferences"]["compensation"]["monthly_stipend"] = int(monthly_stipend)
        editable["sources"]["greenhouse"]["enabled"] = gh_enabled
        editable["sources"]["greenhouse"]["companies"] = _lines_to_list(gh_companies)
        editable["sources"]["lever"]["enabled"] = lv_enabled
        editable["sources"]["lever"]["companies"] = _lines_to_list(lv_companies)
        editable["sources"]["hn"]["enabled"] = hn_enabled

        if not any((gh_enabled, lv_enabled, hn_enabled)):
            st.error("Enable at least one source before saving settings.")
            return

        _write_profile_config(slug, editable)
        _set_notice(slug, "success", "Settings saved.")
        invalidate_dashboard_caches()
        st.rerun()

    st.divider()
    st.subheader("Maintenance")

    action_cols = st.columns(2, gap="large")
    with action_cols[0]:
        st.write("Re-score everything currently in this profile.")
        confirm_rescore = st.checkbox(
            f"I understand this will delete existing scores and rescore {metrics['total']} jobs.",
            key=f"confirm_rescore_{slug}",
        )
        if st.button(
            "Reset scores and rescore",
            key=f"rescore_all_{slug}",
            disabled=not confirm_rescore,
        ):
            missing = _check_api_key(config)
            if missing:
                st.error(f"Scoring requires {missing} in your environment.")
            else:
                try:
                    with st.spinner("Resetting scores and rescoring jobs..."):
                        run_id = start_run(profile=slug, source="dashboard_rescore")
                        rescore_reset(profile=slug)
                        results = score_all_jobs(config, yes=True, profile=slug)
                        scored = [item for item in results if item.get("fit_score", 0) > 0]
                        avg_fit = round(
                            sum(item["fit_score"] for item in scored) / len(scored),
                            1,
                        ) if scored else 0.0
                        finish_run(
                            run_id,
                            jobs_scraped=0,
                            jobs_filtered=0,
                            jobs_saved=0,
                            jobs_scored=len(scored),
                            avg_fit_score=avg_fit,
                            errors=[],
                            status="complete",
                            profile=slug,
                        )
                    _set_notice(slug, "success", f"Re-scored {len(scored)} jobs.")
                    invalidate_dashboard_caches()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Re-score failed: {exc}")

    with action_cols[1]:
        st.write("Clear jobs, scores, and run history for a clean slate.")
        confirm_clear = st.checkbox(
            f"I understand this will erase the current profile database ({metrics['total']} jobs).",
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
        st.error(f"Profile '{slug}' could not be loaded.")
        st.exception(exc)
        if st.button("Back to profile list", key=f"broken_profile_back_{slug}"):
            st.session_state.active_profile = None
            set_active_profile(None)
            st.rerun()
        return

    records = _cached_fetch_job_summaries(slug)
    metrics = _collect_metrics(slug, records)
    runs = _cached_recent_runs(slug)
    profile_name = config.get("profile", {}).get("name", slug.replace("_", " ").title())

    _render_notice(slug)
    _hero(profile_name, config, metrics, raw_config)

    action_cols = st.columns([1.25, 1.0, 1.0, 1.0], gap="medium")
    with action_cols[0]:
        st.write("")
    with action_cols[1]:
        if st.button("Run search now", type="primary", use_container_width=True):
            missing = _check_api_key(config)
            if missing:
                st.error(f"Scoring requires {missing} in your environment.")
            else:
                status_box = st.status("Running pipeline...", expanded=True)
                try:
                    result = _run_pipeline(config, slug, status_box=status_box)
                    _set_notice(
                        slug,
                        "success",
                        (
                            f"Pipeline complete: {result['total_new']} new jobs, "
                            f"{result['scored_count']} scored, avg fit {result['avg_fit']}/100."
                        ),
                    )
                    invalidate_dashboard_caches()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Pipeline failed: {exc}")
    with action_cols[2]:
        if st.button("Refresh data", use_container_width=True):
            invalidate_dashboard_caches()
            st.rerun()
    with action_cols[3]:
        if st.button("Switch profile", use_container_width=True):
            st.session_state.active_profile = None
            set_active_profile(None)
            st.rerun()

    tabs = st.tabs(["Overview", "Jobs", "Activity", "Profile", "Settings"])
    with tabs[0]:
        _render_overview_tab(slug, config, raw_config, records, runs, metrics)
    with tabs[1]:
        _render_jobs_tab(records, slug)
    with tabs[2]:
        _render_activity_tab(slug, runs, metrics)
    with tabs[3]:
        _render_profile_tab(config, raw_config)
    with tabs[4]:
        _render_settings_tab(slug, config, raw_config, metrics)


def _render_profile_selection() -> None:
    set_active_profile(None)
    profiles = list_profiles()

    st.markdown(
        """
        <section class="hero-shell">
            <div class="hero-kicker">Job Search Dashboard</div>
            <h1 class="hero-title">Choose a profile and get a real picture of the search.</h1>
            <p class="hero-copy">
                Each profile keeps its own config, database, and run history. Open one to browse
                top matches, pending jobs, failed scores, and source activity in one place.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    top = st.columns(3)
    with top[0]:
        st.metric("Profiles", len(profiles))
    with top[1]:
        st.metric("Tracked jobs", sum(profile["counts"]["total"] for profile in profiles))
    with top[2]:
        st.metric("Scored jobs", sum(profile["counts"]["scored"] for profile in profiles))

    if not profiles:
        st.info("No profiles found yet. Create one to get started.")
    else:
        columns = st.columns(3, gap="large")
        for index, profile in enumerate(profiles):
            counts = profile["counts"]
            with columns[index % 3]:
                with st.container(border=True):
                    st.markdown(f"### {profile['name']}")
                    st.caption(f"{profile['slug']} · {profile['provider']} · {profile['job_type']}")
                    st.write(f"{counts['total']} jobs tracked")
                    st.write(f"{counts['scored']} jobs scored")
                    if st.button("Open profile", key=f"open_profile_{profile['slug']}"):
                        st.session_state.active_profile = profile["slug"]
                        st.session_state.show_onboarding = False
                        set_active_profile(profile["slug"])
                        st.rerun()

    st.divider()
    if st.button("Create new profile", type="primary", key="create_profile"):
        st.session_state.show_onboarding = True
        st.session_state.onboarding_step = 1
        st.session_state.onboarding_data = {}
        st.rerun()


def main() -> None:
    _init_state()
    _apply_styles()

    if st.session_state.show_onboarding:
        top_cols = st.columns([1.0, 4.0])
        with top_cols[0]:
            if st.button("Back", key="cancel_onboarding"):
                st.session_state.show_onboarding = False
                st.session_state.onboarding_step = 1
                st.session_state.onboarding_data = {}
                st.rerun()
        with top_cols[1]:
            st.write("")
        render_onboarding()
        return

    if st.session_state.active_profile:
        _render_profile_dashboard(st.session_state.active_profile)
    else:
        _render_profile_selection()


if __name__ == "__main__":
    main()
