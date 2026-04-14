"""
dashboard.py — Streamlit web UI for the Job Agent.

Run with: streamlit run dashboard.py

Routing:
  - No active profile in session state  → profile selector
  - Active profile set                  → full dashboard (4 tabs + sidebar)
  - show_onboarding flag set            → onboarding flow
"""

import json
import os
import sqlite3
import time
import yaml
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from config import load_config
from db import (
    count_jobs, get_db_path, get_recent_runs, get_top_jobs, init_db,
    load_discovered_slugs, rescore_reset, set_active_profile, start_run,
    finish_run, update_job_status,
)
from onboarding import render_onboarding
from scraper import scrape_greenhouse, scrape_lever, scrape_hn
from scorer import score_all_jobs, RateLimitReached
from theirstack import get_or_discover_slugs
from progress_tracker import ProgressTracker, Stage, StageStatus, ActivityType
from dashboard_ui import (
    render_progress_header,
    render_pipeline_stages,
    render_source_progress,
    render_activity_feed,
    render_summary,
    render_debug_logs,
)
from logging_config import configure_logging

_BASE_DIR     = Path(__file__).parent
_PROFILES_DIR = _BASE_DIR / "profiles"

_PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini":    "GEMINI_API_KEY",
    "groq":      "GROQ_API_KEY",
    "openai":    "OPENAI_API_KEY",
}

st.set_page_config(
    page_title="Job Agent",
    page_icon="🔍",
    layout="wide",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def list_profiles() -> list:
    """Return info dicts for every valid profile directory under profiles/."""
    if not _PROFILES_DIR.exists():
        return []
    profiles = []
    for folder in sorted(_PROFILES_DIR.iterdir()):
        if not folder.is_dir():
            continue
        cfg_path = folder / "config.yaml"
        if not cfg_path.exists():
            continue
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            name     = cfg.get("profile", {}).get("name", folder.name)
            job_type = cfg.get("profile", {}).get("job_type", "fulltime")
        except Exception:
            name, job_type = folder.name, "fulltime"
        profiles.append({"slug": folder.name, "name": name, "job_type": job_type})
    return profiles


def _job_counts(slug: str) -> dict:
    db_path = get_db_path(slug)
    if not db_path.exists():
        return {"total": 0, "scored": 0}
    try:
        return count_jobs(profile=slug)
    except Exception:
        return {"total": 0, "scored": 0}


def _jobs_by_status(slug: str) -> dict:
    """Return {status: count} dict for a profile's jobs table."""
    db_path = get_db_path(slug)
    if not db_path.exists():
        return {}
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM jobs GROUP BY status"
        ).fetchall()
        conn.close()
        return {r[0]: r[1] for r in rows}
    except Exception:
        return {}


def _last_scored_at(slug: str) -> Optional[str]:
    """Return the most recent scored_at timestamp from the scores table, or None."""
    db_path = get_db_path(slug)
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(db_path)
        row  = conn.execute("SELECT MAX(scored_at) FROM scores").fetchone()
        conn.close()
        return row[0] if row and row[0] else None
    except Exception:
        return None


_LOCKFILE_NAME  = ".worker_running"
_STATUS_NAME    = ".last_run"
_STALE_SECONDS  = 3 * 3600


def _worker_is_running(slug: str) -> bool:
    """Return True if a fresh (< 3h) lockfile exists for this profile."""
    lock = _PROFILES_DIR / slug / _LOCKFILE_NAME
    if not lock.exists():
        return False
    age = time.time() - lock.stat().st_mtime
    return age < _STALE_SECONDS


def _read_last_run(slug: str) -> Optional[dict]:
    """Return the parsed .last_run JSON for this profile, or None."""
    path = _PROFILES_DIR / slug / _STATUS_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _load_config(slug: str) -> Optional[dict]:
    """Load config for a profile, showing an error and returning None on failure."""
    try:
        return load_config(profile=slug)
    except Exception as exc:
        st.error(f"Cannot load config for profile '{slug}': {exc}")
        return None


def _check_api_key(config: dict) -> Optional[str]:
    """Return an error message string if the provider's API key is missing, else None."""
    provider = config.get("llm", {}).get("provider", "groq")
    env_var  = _PROVIDER_ENV_VARS.get(provider)
    if env_var and not os.environ.get(env_var):
        return (
            f"**{env_var}** is not set in your `.env` file.  \n"
            f"Add it: `{env_var}=your_key_here`"
        )
    return None


def _clear_all_jobs(slug: str) -> None:
    """Permanently delete all jobs and scores from a profile's DB."""
    db_path = get_db_path(slug)
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM scores")
        conn.execute("DELETE FROM jobs")
        conn.commit()
        conn.close()


def _score_badge(score: int) -> str:
    if score >= 85:
        return f"🟢 {score}"
    if score >= 70:
        return f"🟡 {score}"
    if score >= 55:
        return f"🟠 {score}"
    return f"🔴 {score}"


def _results_to_df(results: list) -> pd.DataFrame:
    """Convert get_top_jobs() result list to a flat display DataFrame."""
    if not results:
        return pd.DataFrame(
            columns=["id", "fit_score", "score", "ats", "title",
                     "company", "location", "source", "status", "url"]
        )
    rows = []
    for r in results:
        job = r["job"]
        rows.append({
            "id":        job.id,
            "fit_score": r["fit_score"],
            "score":     _score_badge(r["fit_score"]),
            "ats":       r["ats_score"],
            "title":     job.title,
            "company":   job.company,
            "location":  job.location,
            "source":    job.source,
            "status":    job.status,
            "url":       job.url,
        })
    return pd.DataFrame(rows)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _run_pipeline(config: dict, slug: str) -> tuple[dict, ProgressTracker]:
    """
    Run the full scrape + score pipeline for a profile.

    Uses a ProgressTracker to emit user-friendly events instead of raw logs.
    Returns (summary_dict, tracker) where tracker contains all progress events.
    """
    tracker = ProgressTracker()
    sources = config.get("sources", {})
    total_new = 0
    total_scraped = 0
    total_filtered = 0
    total_saved = 0
    jobs_scored = 0
    avg_fit = 0.0
    errors: list[str] = []
    status = "complete"

    gh_enabled = sources.get("greenhouse", {}).get("enabled", True)
    lv_enabled = sources.get("lever", {}).get("enabled", True)
    hn_enabled = sources.get("hn", {}).get("enabled", True)

    run_id = start_run(profile=slug, source="dashboard")

    try:
        # Stage 1: Discovery
        tracker.start_stage(Stage.DISCOVERING)
        if gh_enabled or lv_enabled:
            tracker.add_activity_log("Discovering company slugs from TheirStack…")
            slug_map = get_or_discover_slugs(config, profile=slug)
            tracker.stages[Stage.DISCOVERING].metrics = {
                "companies": len(slug_map.get("greenhouse", [])) + len(slug_map.get("lever", []))
            }
            tracker.add_activity_log(
                f"Found {len(slug_map.get('greenhouse', []))} Greenhouse + {len(slug_map.get('lever', []))} Lever companies"
            )
        else:
            slug_map = {"greenhouse": [], "lever": []}
        tracker.complete_stage(Stage.DISCOVERING)

        # Stage 2: Fetching
        tracker.start_stage(Stage.FETCHING)
        tracker.stages[Stage.FETCHING].metrics = {
            "sources_active": sum(
                [gh_enabled, lv_enabled, hn_enabled]
            )
        }
        tracker.complete_stage(Stage.FETCHING)

        # Stage 3: Scraping
        tracker.start_stage(Stage.SCRAPING)
        
        if gh_enabled:
            tracker.start_source("Greenhouse")
            tracker.register_source(
                "Greenhouse",
                len(slug_map.get("greenhouse", []))
            )
            r = scrape_greenhouse(config, slugs=slug_map["greenhouse"], profile=slug)
            jobs_found = r.get("new_jobs_saved", 0)
            total_scraped += r.get("jobs_scraped", 0)
            total_filtered += r.get("jobs_filtered", 0)
            total_saved += jobs_found
            tracker.complete_source("Greenhouse", jobs_found)
            total_new += jobs_found

        if lv_enabled:
            tracker.start_source("Lever")
            tracker.register_source(
                "Lever",
                len(slug_map.get("lever", []))
            )
            r = scrape_lever(config, slugs=slug_map["lever"], profile=slug)
            jobs_found = r.get("new_jobs_saved", 0)
            total_scraped += r.get("jobs_scraped", 0)
            total_filtered += r.get("jobs_filtered", 0)
            total_saved += jobs_found
            tracker.complete_source("Lever", jobs_found)
            total_new += jobs_found

        if hn_enabled:
            tracker.start_source("HN Who's Hiring")
            tracker.register_source("HN Who's Hiring", 1)
            r = scrape_hn(config, profile=slug)
            jobs_found = r.get("new_jobs_saved", 0)
            total_scraped += r.get("jobs_scraped", 0)
            total_filtered += r.get("jobs_filtered", 0)
            total_saved += jobs_found
            tracker.complete_source("HN Who's Hiring", jobs_found)
            total_new += jobs_found

        tracker.stages[Stage.SCRAPING].metrics = {"jobs_found": total_new}
        tracker.complete_stage(Stage.SCRAPING)

        # Stage 4: Scoring
        tracker.start_stage(Stage.SCORING)
        tracker.add_activity_log("Beginning LLM scoring and filtering…")
        
        try:
            scored = score_all_jobs(config, yes=True, profile=slug)
            tracker.add_activity_log(f"Scored {len(scored)} jobs")
        except RateLimitReached as exc:
            scored = []
            errors.append(str(exc))
            status = "failed"
            tracker.add_warning("Daily rate limit reached — scoring stopped early")
            tracker.fail_stage(Stage.SCORING, "Daily rate limit hit")
            return {
                "total_new": total_new,
                "jobs_scraped": total_scraped,
                "jobs_filtered": total_filtered,
                "jobs_saved": total_saved,
                "scored_count": 0,
                "avg_fit": 0.0,
                "error": "Daily rate limit reached",
            }, tracker

        scored_ok = [r for r in scored if r.get("fit_score", 0) > 0]
        jobs_scored = len(scored_ok)
        avg_fit = (
            round(sum(r["fit_score"] for r in scored_ok) / len(scored_ok), 1)
            if scored_ok
            else 0.0
        )
        
        tracker.stages[Stage.SCORING].metrics = {
            "jobs_scored": jobs_scored,
            "avg_fit": avg_fit,
        }
        tracker.add_activity_log(
            f"Scoring complete: {jobs_scored} jobs with avg fit {avg_fit}/10"
        )
        tracker.complete_stage(Stage.SCORING)

        # Stage 5: Finalizing
        tracker.start_stage(Stage.FINALIZING)
        tracker.add_activity_log("Finalizing results…")
        tracker.complete_stage(Stage.FINALIZING)

        return {
            "total_new": total_new,
            "jobs_scraped": total_scraped,
            "jobs_filtered": total_filtered,
            "jobs_saved": total_saved,
            "scored_count": jobs_scored,
            "avg_fit": avg_fit,
        }, tracker

    except Exception as e:
        errors.append(str(e))
        status = "failed"
        tracker.add_error(f"Pipeline failed: {str(e)}")
        return {
            "error": str(e),
            "total_new": total_new,
            "jobs_scraped": total_scraped,
            "jobs_filtered": total_filtered,
            "jobs_saved": total_saved,
            "scored_count": 0,
            "avg_fit": 0.0,
        }, tracker

    finally:
        if run_id:
            finish_run(
                run_id,
                jobs_scraped=total_scraped,
                jobs_filtered=total_filtered,
                jobs_saved=total_saved,
                jobs_scored=jobs_scored,
                avg_fit_score=avg_fit,
                errors=errors,
                status=status,
                profile=slug,
            )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar(config: dict, slug: str) -> None:
    with st.sidebar:
        name    = config.get("profile", {}).get("name", slug)
        initial = name[0].upper() if name else "?"

        # Avatar + name
        st.markdown(
            f"""<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
              <div style="width:40px;height:40px;border-radius:50%;background:#1f77b4;
                          display:flex;align-items:center;justify-content:center;
                          color:white;font-size:18px;font-weight:bold">{initial}</div>
              <div><strong>{name}</strong><br><small><code>{slug}</code></small></div>
            </div>""",
            unsafe_allow_html=True,
        )

        if st.button("⇄  Switch profile", use_container_width=True):
            st.session_state.active_profile  = None
            st.session_state.selected_job_id = None
            st.session_state.pipeline_running = False
            st.session_state.pipeline_result  = None
            set_active_profile(None)
            st.rerun()

        st.divider()

        # DB stats
        counts    = _job_counts(slug)
        by_status = _jobs_by_status(slug)
        last_run  = _last_scored_at(slug)

        st.markdown("**Database**")
        c1, c2 = st.columns(2)
        c1.metric("Total", counts["total"])
        c2.metric("Scored", counts["scored"])
        if by_status:
            st.caption(
                f"New: {by_status.get('new', 0)}  "
                f"· Applied: {by_status.get('applied', 0)}  "
                f"· Skipped: {by_status.get('skipped', 0)}"
            )
        if last_run:
            st.caption(f"Last scored: {last_run[:16]}")

        # ── Worker status (maintenance mode) ─────────────────────────────────
        recent_runs = get_recent_runs(profile=slug)
        last_run_data = recent_runs[0] if recent_runs else _read_last_run(slug)
        if _worker_is_running(slug):
            st.warning("⚙️ Daily update in progress — data may be stale")
        elif last_run_data:
            finished = last_run_data.get("finished_at", "")[:16]
            scored   = last_run_data.get("jobs_scored", 0)
            if last_run_data.get("status") == "failed":
                err = last_run_data.get("error_message", "unknown error")
                st.error(f"⚠️ Last run failed: {err[:80]}")
            else:
                st.caption(f"Last updated: {finished} · {scored} jobs scored")

        st.divider()

        # Companies being searched
        sources = config.get("sources", {})
        gh_companies = sources.get("greenhouse", {}).get("companies", [])
        lv_companies = sources.get("lever", {}).get("companies", [])
        
        # Load discovered companies from DB
        try:
            gh_discovered = load_discovered_slugs(ats="greenhouse", profile=slug)
            lv_discovered = load_discovered_slugs(ats="lever", profile=slug)
        except Exception:
            gh_discovered = []
            lv_discovered = []
        
        st.markdown("**Companies Searched**")
        
        # Check TheirStack status
        has_theirstack_key = bool(os.environ.get("THEIRSTACK_API_KEY"))
        if has_theirstack_key:
            st.caption("🔍 TheirStack API: Active (may discover additional companies)")
        else:
            st.caption("🔍 TheirStack API: No key set (only configured companies)")
        
        if gh_companies or gh_discovered:
            with st.expander(f"Greenhouse ({len(gh_companies)} configured, {len(gh_discovered)} discovered)"):
                if gh_companies:
                    st.markdown("**Configured:**")
                    for company in gh_companies:
                        st.write(f"• {company}")
                if gh_discovered:
                    st.markdown("**Discovered:**")
                    for company in gh_discovered:
                        st.write(f"• {company}")
                else:
                    st.caption("*No additional companies discovered yet*")
        
        if lv_companies or lv_discovered:
            with st.expander(f"Lever ({len(lv_companies)} configured, {len(lv_discovered)} discovered)"):
                if lv_companies:
                    st.markdown("**Configured:**")
                    for company in lv_companies:
                        st.write(f"• {company}")
                if lv_discovered:
                    st.markdown("**Discovered:**")
                    for company in lv_discovered:
                        st.write(f"• {company}")
                else:
                    st.caption("*No additional companies discovered yet*")

        st.divider()

        # Run job search
        key_err = _check_api_key(config)
        running = st.session_state.get("pipeline_running", False)

        if key_err:
            st.warning(key_err)
        else:
            if st.button(
                "▶  Run job search",
                type="primary",
                use_container_width=True,
                disabled=running,
                key="run_btn",
            ):
                st.session_state.pipeline_running = True
                st.session_state.pipeline_result  = None
                st.rerun()

        if running:
            st.caption("Running… see main panel for progress.")

        result = st.session_state.get("pipeline_result")
        if result and not running:
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(
                    f"Done.  \n"
                    f"New jobs: **{result['total_new']}**  \n"
                    f"Scored: **{result['scored_count']}**  \n"
                    f"Avg fit: **{result['avg_fit']}**"
                )


# ── Tab 1: Jobs ───────────────────────────────────────────────────────────────

def _render_job_detail(result: dict) -> None:
    """Detail card rendered below the table when a row is selected."""
    job  = result["job"]
    dims = result.get("dimension_scores", {})

    st.markdown(f"#### {job.title} — {job.company}")
    st.caption(f"{job.location}  ·  {job.source}")

    if result.get("one_liner"):
        st.info(result["one_liner"])

    # Dimension scores — horizontal progress bars
    dim_order = ["role_fit", "stack_match", "seniority", "location", "growth", "compensation"]
    st.markdown("**Dimension scores**")
    for dim in dim_order:
        score = int(dims.get(dim) or 0)
        label = dim.replace("_", " ").title()
        col1, col2 = st.columns([2, 5])
        with col1:
            st.write(label)
        with col2:
            st.progress(score / 10, text=f"{score}/10")

    col1, col2, col3 = st.columns(3)
    with col1:
        if result.get("reasons"):
            st.markdown("**Reasons**")
            for item in result["reasons"]:
                st.write(f"✅ {item}")
    with col2:
        if result.get("flags"):
            st.markdown("**Flags**")
            for item in result["flags"]:
                st.write(f"⚠️ {item}")
    with col3:
        if result.get("skill_misses"):
            st.markdown("**Missing from resume**")
            for item in result["skill_misses"]:
                st.write(f"❌ {item}")

    st.markdown(f"[Open job posting ↗]({job.url})")


def _render_tab_jobs(results: list, slug: str, min_score_default: int = 60) -> None:
    if not results:
        st.info("No scored jobs yet. Hit **▶ Run job search** in the sidebar to get started.")
        return

    df = _results_to_df(results)

    # ── Filters ───────────────────────────────────────────────────────────────
    fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 3])
    with fc1:
        score_range = st.slider(
            "Score", 0, 100, (min_score_default, 100), key="filter_score",
        )
    with fc2:
        source_opts = sorted(df["source"].unique().tolist())
        sources_sel = st.multiselect(
            "Source", source_opts, default=source_opts, key="filter_sources",
        )
    with fc3:
        statuses_sel = st.multiselect(
            "Status", ["new", "applied", "skipped"],
            default=["new"], key="filter_statuses",
        )
    with fc4:
        search = st.text_input("Search title / company", key="filter_search")

    # Apply filters
    sources_active  = sources_sel  if sources_sel  else source_opts
    statuses_active = statuses_sel if statuses_sel else ["new", "applied", "skipped"]

    mask = (
        df["fit_score"].between(score_range[0], score_range[1]) &
        df["source"].isin(sources_active) &
        df["status"].isin(statuses_active)
    )
    if search:
        sl = search.lower()
        mask &= (
            df["title"].str.lower().str.contains(sl, na=False) |
            df["company"].str.lower().str.contains(sl, na=False)
        )
    filtered = df[mask].reset_index(drop=True)

    st.caption(f"Showing {len(filtered)} of {len(df)} scored jobs")

    if filtered.empty:
        st.warning("No jobs match the current filters.")
        return

    # ── Table ─────────────────────────────────────────────────────────────────
    selected_id = st.session_state.get("selected_job_id")

    # Add a "👁" checkbox column so the user can select a row for the detail panel
    table_df = filtered[["id", "score", "ats", "title", "company",
                          "location", "source", "status", "url"]].copy()
    table_df.insert(0, "👁", table_df["id"] == selected_id)

    edited = st.data_editor(
        table_df,
        column_config={
            "👁":       st.column_config.CheckboxColumn("👁", width="small"),
            "id":       None,  # hidden
            "score":    st.column_config.TextColumn("Score", width="small"),
            "ats":      st.column_config.NumberColumn("ATS", width="small"),
            "title":    st.column_config.TextColumn("Title"),
            "company":  st.column_config.TextColumn("Company"),
            "location": st.column_config.TextColumn("Location"),
            "source":   st.column_config.TextColumn("Source", width="small"),
            "status":   st.column_config.SelectboxColumn(
                "Status",
                options=["new", "applied", "skipped"],
                width="small",
            ),
            "url": st.column_config.LinkColumn("Link", width="small"),
        },
        disabled=["score", "ats", "title", "company", "location", "source", "url"],
        use_container_width=True,
        hide_index=True,
        key="jobs_table",
    )

    # ── Detect changes and persist ─────────────────────────────────────────────
    needs_rerun  = False
    new_selected = selected_id  # default: keep current selection

    for i in range(len(filtered)):
        job_id = filtered.iloc[i]["id"]

        # Status change → persist immediately
        orig_status = filtered.iloc[i]["status"]
        new_status  = edited.iloc[i]["status"]
        if orig_status != new_status and new_status in ("new", "applied", "skipped"):
            update_job_status(job_id, new_status, profile=slug)
            needs_rerun = True

        # View toggle → update selected_job_id
        was_selected = (job_id == selected_id)
        is_selected  = bool(edited.iloc[i]["👁"])
        if is_selected and not was_selected:
            new_selected = job_id
            needs_rerun  = True
        elif not is_selected and was_selected:
            new_selected = None
            needs_rerun  = True

    if new_selected != selected_id:
        st.session_state.selected_job_id = new_selected

    if needs_rerun:
        st.rerun()

    # ── Detail panel ───────────────────────────────────────────────────────────
    if selected_id:
        selected_result = next(
            (r for r in results if r["job"].id == selected_id), None
        )
        if selected_result:
            st.divider()
            _render_job_detail(selected_result)


# ── Tab 2: Analytics ──────────────────────────────────────────────────────────

def _render_tab_analytics(results: list, slug: str) -> None:
    if not results:
        st.info("No scored jobs yet. Run a job search to see analytics.")
        return

    df     = _results_to_df(results)
    scores = df["fit_score"]

    st.markdown("### Pipeline run history")
    recent_runs = get_recent_runs(profile=slug)
    if recent_runs:
        history_rows = []
        for run in recent_runs:
            history_rows.append({
                "Started": run["started_at"][:19],
                "Finished": run["finished_at"][:19] if run.get("finished_at") else "running",
                "Status": "✅ Complete" if run["status"] == "complete" else "⚠️ " + run["status"],
                "Saved": run["jobs_saved"],
                "Scraped": run["jobs_scraped"],
                "Filtered": run["jobs_filtered"],
                "Scored": run["jobs_scored"],
                "Avg fit": round(run.get("avg_fit_score") or 0, 1),
                "Errors": len(run.get("errors", [])),
            })
        st.dataframe(
            pd.DataFrame(history_rows),
            use_container_width=True,
        )
    else:
        st.info("No pipeline runs recorded yet. Run the job search to populate history.")

    st.markdown("---")
    row1c1, row1c2 = st.columns(2)

    with row1c1:
        st.markdown("**Score distribution**")
        bands  = [f"{b}–{b+9}" for b in range(0, 100, 10)]
        counts = [int(((scores >= b) & (scores < b + 10)).sum()) for b in range(0, 100, 10)]
        st.bar_chart(pd.DataFrame({"Jobs": counts}, index=bands))

    with row1c2:
        st.markdown("**Jobs by source**")
        st.bar_chart(df["source"].value_counts().rename("Jobs"))

    row2c1, row2c2 = st.columns(2)

    with row2c1:
        st.markdown("**Jobs by status**")
        st.bar_chart(df["status"].value_counts().rename("Jobs"))

    with row2c2:
        st.markdown("**Top 10 companies by avg fit score**")
        top = (
            df.groupby("company")["fit_score"]
            .mean()
            .nlargest(10)
            .round(1)
            .rename("Avg Score")
        )
        st.bar_chart(top)

    st.markdown("**ATS score vs fit score**")
    st.scatter_chart(
        df[["fit_score", "ats"]].rename(columns={"fit_score": "Fit Score", "ats": "ATS Score"}),
        x="Fit Score",
        y="ATS Score",
    )


# ── Tab 3: Profile ────────────────────────────────────────────────────────────

def _render_tab_profile(config: dict, slug: str) -> None:
    prof  = config.get("profile", {})
    prefs = config.get("preferences", {})
    llm   = config.get("llm", {})
    job_type = prof.get("job_type", "fulltime")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Name:** {prof.get('name', '—')}")
        st.markdown(f"**Job type:** {'Internship' if job_type == 'internship' else 'Full-time'}")
        if job_type == "internship":
            season = prof.get("target_season", "")
            school = prof.get("school", "")
            major  = prof.get("major", "")
            st.markdown(f"**Target:** {season}  ·  {school}  ·  {major}")
        st.markdown(f"**Bio:** {prof.get('bio', '—')}")

        st.markdown("**Target titles**")
        for t in prefs.get("titles", []):
            st.write(f"· {t}")

        st.markdown("**Desired skills**")
        st.write(", ".join(prefs.get("desired_skills", [])) or "—")

    with col2:
        loc = prefs.get("location", {})
        st.markdown(f"**Remote OK:** {'Yes' if loc.get('remote_ok') else 'No'}")
        preferred = ", ".join(loc.get("preferred_locations", []))
        if preferred:
            st.markdown(f"**Preferred locations:** {preferred}")

        comp = prefs.get("compensation", {})
        if job_type == "fulltime":
            sal = comp.get("min_salary", 0)
            st.markdown(f"**Min salary:** ${sal:,}")
        else:
            stip = comp.get("monthly_stipend", 0)
            st.markdown(f"**Expected stipend:** {'${:,}/mo'.format(stip) if stip else '—'}")

        provider = llm.get("provider", "—")
        model    = llm.get("model", {}).get(provider, "—")
        st.markdown(f"**LLM:** {provider} — `{model}`")

    st.divider()

    resume_text = prof.get("resume", "")
    if resume_text:
        with st.expander("Resume (click to expand)"):
            st.text(resume_text[:3000] + ("…" if len(resume_text) > 3000 else ""))
    elif prof.get("resume_file"):
        st.caption(f"Resume PDF: `{prof.get('resume_file')}`")

    st.divider()
    st.info(
        f"To edit your profile, update `profiles/{slug}/config.yaml` directly "
        "and restart the dashboard."
    )


# ── Tab 4: Settings ───────────────────────────────────────────────────────────

def _render_tab_settings(config: dict, slug: str) -> None:
    scoring  = config.get("scoring", {})
    weights  = scoring.get("weights", {})
    cfg_path = _PROFILES_DIR / slug / "config.yaml"

    # ── Display ───────────────────────────────────────────────────────────────
    st.subheader("Display")
    min_score = st.slider(
        "Min display score",
        0, 100, int(scoring.get("min_display_score", 60)),
        key="settings_min_score",
    )

    # ── Scoring weights ───────────────────────────────────────────────────────
    st.subheader("Scoring weights")
    st.caption("Must sum to 1.0")
    dim_order = ["role_fit", "stack_match", "seniority", "location", "growth", "compensation"]
    new_weights = {}
    for dim in dim_order:
        new_weights[dim] = st.slider(
            dim.replace("_", " ").title(),
            0.0, 1.0, float(weights.get(dim, 0.0)), 0.05,
            key=f"wt_{dim}",
        )
    total = round(sum(new_weights.values()), 4)
    if abs(total - 1.0) > 0.01:
        st.warning(f"Weights sum to **{total}** — they must sum to 1.0 before saving.")
    else:
        st.success(f"Weights sum to {total} ✓")

    if st.button("💾 Save settings", type="primary"):
        if abs(total - 1.0) > 0.01:
            st.error("Fix the weights before saving.")
        else:
            try:
                with open(cfg_path, encoding="utf-8") as f:
                    raw = yaml.safe_load(f)
                raw.setdefault("scoring", {})["min_display_score"] = min_score
                raw["scoring"]["weights"] = new_weights
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.dump(raw, f, default_flow_style=False,
                              sort_keys=False, allow_unicode=True)
                st.success("Settings saved. Restart the dashboard to apply weight changes.")
            except Exception as exc:
                st.error(f"Failed to save: {exc}")

    st.divider()

    # ── Re-score all jobs ─────────────────────────────────────────────────────
    st.subheader("Re-score all jobs")
    n_jobs   = _job_counts(slug).get("total", 0)
    provider = config.get("llm", {}).get("provider", "groq")
    rpm      = config.get("llm", {}).get("rate_limits", {}).get(provider, {}).get("max_rpm", 10)
    est_min  = max(1, round(n_jobs / max(rpm, 1)))
    st.caption(f"Estimated ~{est_min} min for {n_jobs} jobs at {rpm} RPM ({provider})")

    if not st.session_state.get("rescore_confirm"):
        if st.button("🔄 Re-score all jobs"):
            st.session_state.rescore_confirm = True
            st.rerun()
    else:
        st.warning("This will delete all existing scores and re-score from scratch.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✓ Confirm re-score", type="primary", key="confirm_rescore"):
                key_err = _check_api_key(config)
                if key_err:
                    st.error(key_err)
                else:
                    with st.spinner("Re-scoring all jobs…"):
                        rescore_reset(profile=slug)
                        try:
                            score_all_jobs(config, yes=True, profile=slug)
                        except RateLimitReached:
                            st.warning("Daily rate limit hit — scoring stopped early.")
                    st.session_state.rescore_confirm = False
                    st.success("Re-scoring complete.")
                    st.rerun()
        with c2:
            if st.button("Cancel", key="cancel_rescore"):
                st.session_state.rescore_confirm = False
                st.rerun()

    st.divider()

    # ── Manage companies ──────────────────────────────────────────────────────
    st.subheader("Manage Companies")
    st.caption("Add or remove companies to search for jobs. Changes take effect on next job search.")
    
    sources = config.get("sources", {})
    gh_companies = sources.get("greenhouse", {}).get("companies", [])
    lv_companies = sources.get("lever", {}).get("companies", [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Greenhouse Companies**")
        new_gh = st.text_input("Add Greenhouse company slug", key="add_gh")
        if st.button("➕ Add Greenhouse", key="add_gh_btn") and new_gh.strip():
            if new_gh.strip() not in gh_companies:
                gh_companies.append(new_gh.strip())
                sources.setdefault("greenhouse", {})["companies"] = gh_companies
                config["sources"] = sources
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                st.success(f"Added {new_gh}")
                st.rerun()
            else:
                st.warning("Company already in list")
        
        # Remove companies
        to_remove_gh = st.multiselect("Remove Greenhouse companies", gh_companies, key="remove_gh")
        if st.button("➖ Remove Selected", key="remove_gh_btn") and to_remove_gh:
            gh_companies = [c for c in gh_companies if c not in to_remove_gh]
            sources.setdefault("greenhouse", {})["companies"] = gh_companies
            config["sources"] = sources
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            st.success(f"Removed {len(to_remove_gh)} companies")
            st.rerun()
    
    with col2:
        st.markdown("**Lever Companies**")
        new_lv = st.text_input("Add Lever company slug", key="add_lv")
        if st.button("➕ Add Lever", key="add_lv_btn") and new_lv.strip():
            if new_lv.strip() not in lv_companies:
                lv_companies.append(new_lv.strip())
                sources.setdefault("lever", {})["companies"] = lv_companies
                config["sources"] = sources
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                st.success(f"Added {new_lv}")
                st.rerun()
            else:
                st.warning("Company already in list")
        
        # Remove companies
        to_remove_lv = st.multiselect("Remove Lever companies", lv_companies, key="remove_lv")
        if st.button("➖ Remove Selected", key="remove_lv_btn") and to_remove_lv:
            lv_companies = [c for c in lv_companies if c not in to_remove_lv]
            sources.setdefault("lever", {})["companies"] = lv_companies
            config["sources"] = sources
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            st.success(f"Removed {len(to_remove_lv)} companies")
            st.rerun()

    st.divider()

    # ── Clear all jobs ─────────────────────────────────────────────────────────
    st.subheader("Danger zone")
    if not st.session_state.get("clear_confirm_1"):
        if st.button("☢️ Clear all jobs"):
            st.session_state.clear_confirm_1 = True
            st.rerun()
    elif not st.session_state.get("clear_confirm_2"):
        st.error(
            f"This will permanently delete all **{n_jobs} jobs** and their scores. "
            "This cannot be undone."
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("I understand — delete everything", type="primary",
                         key="clear_confirm_2_btn"):
                st.session_state.clear_confirm_2 = True
                st.rerun()
        with c2:
            if st.button("Cancel", key="cancel_clear_1"):
                st.session_state.clear_confirm_1 = False
                st.rerun()
    else:
        st.error("Last chance — this is permanent.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("☢️ Yes, delete everything", type="primary",
                         key="clear_final_btn"):
                _clear_all_jobs(slug)
                st.session_state.clear_confirm_1 = False
                st.session_state.clear_confirm_2 = False
                st.session_state.selected_job_id = None
                st.success("All jobs cleared.")
                st.rerun()
        with c2:
            if st.button("Cancel", key="cancel_clear_2"):
                st.session_state.clear_confirm_1 = False
                st.session_state.clear_confirm_2 = False
                st.rerun()


# ── Main dashboard ────────────────────────────────────────────────────────────

def _render_profile_dashboard(slug: str) -> None:
    """Full dashboard for the active profile."""
    # Always sync module-level DB profile with session state
    set_active_profile(slug)
    configure_logging(profile=slug, debug=False)
    init_db(profile=slug)

    config = _load_config(slug)
    if config is None:
        return  # error already shown by _load_config

    # ── Pipeline execution (triggered by sidebar button) ──────────────────────
    if st.session_state.get("pipeline_running"):
        # Render sidebar first (button shows as disabled)
        _render_sidebar(config, slug)

        key_err = _check_api_key(config)
        if key_err:
            st.error(key_err)
            st.session_state.pipeline_running = False
            st.session_state.pipeline_result  = {"error": key_err}
            st.rerun()
            return

        # NEW: Modern progress dashboard
        st.title("🚀 Job Search in Progress")
        progress_placeholder = st.empty()
        stages_placeholder = st.empty()
        sources_placeholder = st.empty()
        activities_placeholder = st.empty()

        # Run pipeline with progress tracking
        with st.spinner("Running…"):
            result, tracker = _run_pipeline(config, slug)

        # Display final result
        progress_placeholder.empty()
        stages_placeholder.empty()
        sources_placeholder.empty()
        activities_placeholder.empty()

        # Show final state
        with progress_placeholder.container():
            render_progress_header(tracker)

        with stages_placeholder.container():
            render_pipeline_stages(tracker)

        if tracker.sources:
            with sources_placeholder.container():
                render_source_progress(tracker)

        with activities_placeholder.container():
            render_activity_feed(tracker)

        # Summary
        render_summary(tracker)

        # Debug logs
        render_debug_logs(tracker)

        # Results
        st.divider()
        st.subheader("Results")
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.success(
                f"✅ Pipeline complete: Found **{result['total_new']}** new jobs, "
                f"scored **{result['scored_count']}** with avg fit **{result['avg_fit']}/10**"
            )

        st.session_state.pipeline_running = False
        st.session_state.pipeline_result = result

        # Button to return to main dashboard
        if st.button("← Return to Dashboard"):
            st.rerun()
        
        return

    # ── Normal render ─────────────────────────────────────────────────────────
    _render_sidebar(config, slug)

    # Fetch all scored jobs (used by Jobs and Analytics tabs)
    all_results = get_top_jobs(0, profile=slug)

    min_score_default = int(config.get("scoring", {}).get("min_display_score", 60))

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Jobs", "📊 Analytics", "👤 Profile", "⚙️ Settings"]
    )

    with tab1:
        _render_tab_jobs(all_results, slug, min_score_default)

    with tab2:
        _render_tab_analytics(all_results, slug)

    with tab3:
        _render_tab_profile(config, slug)

    with tab4:
        _render_tab_settings(config, slug)


# ── Profile selection ─────────────────────────────────────────────────────────

def _render_profile_selection() -> None:
    st.title("🔍 Job Agent")
    st.markdown("Select your profile to continue, or create a new one.")

    profiles = list_profiles()

    if profiles:
        st.subheader("Your profiles")
        for p in profiles:
            counts    = _job_counts(p["slug"])
            is_intern = p["job_type"] == "internship"
            badge     = "🎓 Internship" if is_intern else "💼 Full-time"
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    f"**{p['name']}** &nbsp; `{p['slug']}` &nbsp; {badge}  \n"
                    f"<small>{counts['total']} jobs · {counts['scored']} scored</small>",
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("Continue →", key=f"sel_{p['slug']}"):
                    st.session_state.active_profile  = p["slug"]
                    st.session_state.show_onboarding = False
                    set_active_profile(p["slug"])
                    st.rerun()
        st.divider()
    else:
        st.info("No profiles yet. Create your first profile to get started.")

    if st.button(
        "➕  Add new profile",
        type="primary" if not profiles else "secondary",
    ):
        st.session_state.show_onboarding = True
        st.session_state.onboarding_step = 1
        st.session_state.onboarding_data = {}
        st.rerun()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Initialise all session state keys used across the app
    _defaults = {
        "active_profile":   None,
        "show_onboarding":  False,
        "selected_job_id":  None,
        "pipeline_running": False,
        "pipeline_result":  None,
        "clear_confirm_1":  False,
        "clear_confirm_2":  False,
        "rescore_confirm":  False,
        "onboarding_step":  1,
        "onboarding_data":  {},
    }
    for key, val in _defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Configure logging for the active profile or default session
    profile = st.session_state.active_profile
    configure_logging(profile=profile or "default", debug=False)

    if st.session_state.show_onboarding:
        if st.button("✕  Cancel", key="cancel_onboarding"):
            st.session_state.show_onboarding = False
            st.session_state.onboarding_step = 1
            st.session_state.onboarding_data = {}
            st.rerun()
        render_onboarding()

    elif st.session_state.active_profile:
        _render_profile_dashboard(st.session_state.active_profile)

    else:
        _render_profile_selection()


if __name__ == "__main__":
    main()
