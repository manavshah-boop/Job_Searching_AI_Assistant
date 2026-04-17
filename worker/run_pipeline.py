"""
worker/run_pipeline.py — Out-of-process pipeline runner for a single profile.

Usage:
    python worker/run_pipeline.py --profile manav
    python worker/run_pipeline.py --profile sister

Responsibilities:
- Creates a lockfile at profiles/{profile}/.worker_running on start,
  deletes it on finish (success or failure).
- Checks for a stale lockfile (> 3 hours old) and proceeds anyway.
- Writes a JSON status file at profiles/{profile}/.last_run on finish.
- Writes live progress state to profiles/{profile}/.run_progress.json
  atomically after every stage transition and every 5 scored jobs.
- Calls scrapers and scorer directly (no sys.argv monkey-patching).
- Exits 0 on success, 1 on failure.
- Adding a third profile requires zero changes here.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Resolve project root regardless of where this script is called from.
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from logging_config import configure_logging
from loguru import logger

_LOCKFILE_NAME  = ".worker_running"
_STATUS_NAME    = ".last_run"
_PROGRESS_NAME  = ".run_progress.json"
_STALE_SECONDS  = 3 * 3600  # 3 hours


def _profiles_dir() -> Path:
    return _PROJECT_ROOT / "profiles"


def _lockfile_path(profile: str) -> Path:
    return _profiles_dir() / profile / _LOCKFILE_NAME


def _status_path(profile: str) -> Path:
    return _profiles_dir() / profile / _STATUS_NAME


def _progress_path(profile: str) -> Path:
    return _profiles_dir() / profile / _PROGRESS_NAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _acquire_lock(profile: str) -> bool:
    """
    Attempt to acquire the lockfile. Returns True if lock was acquired.
    Returns False if a fresh (< 3h) lockfile already exists.
    Removes stale lockfiles (>= 3h old) and proceeds.
    """
    lock = _lockfile_path(profile)

    if lock.exists():
        age = time.time() - lock.stat().st_mtime
        if age < _STALE_SECONDS:
            logger.warning(
                f"Lock exists and is {age / 60:.0f} min old "
                f"(< 3h) — another worker may be running. Aborting."
            )
            return False
        logger.info(
            f"Stale lockfile found ({age / 3600:.1f}h old) — "
            "treating as crashed run, proceeding."
        )
        lock.unlink()

    # Write PID + timestamp so operators can inspect
    lock.write_text(json.dumps({"pid": os.getpid(), "started_at": _now_iso()}))
    return True


def _release_lock(profile: str) -> None:
    lock = _lockfile_path(profile)
    try:
        lock.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning(f"Could not remove lockfile: {exc}")


def _write_progress(profile: str, tracker) -> None:
    """Atomically write tracker state to .run_progress.json."""
    path = _progress_path(profile)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(tracker.to_dict(), indent=2))
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning(f"Could not write progress JSON: {exc}")


def _write_status(
    profile: str,
    started_at: str,
    status: str,
    jobs_scored: int = 0,
    error_message: str = "",
) -> None:
    payload = {
        "started_at":    started_at,
        "finished_at":   _now_iso(),
        "status":        status,          # "success" | "failed"
        "jobs_scored":   jobs_scored,
        "error_message": error_message,
    }
    path = _status_path(profile)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _enabled_sources(config: dict) -> dict:
    sources = config.get("sources", {})
    return {
        "greenhouse": sources.get("greenhouse", {}).get("enabled", True),
        "lever": sources.get("lever", {}).get("enabled", True),
        "hackernews": sources.get("hn", {}).get("enabled", False),
    }


def _run(profile: str, tracker) -> int:
    """
    Run the full pipeline for the given profile.
    Updates tracker at each stage and writes progress JSON atomically.
    Returns the number of jobs successfully scored.
    """
    from config import load_config
    from db import finish_run, init_db, start_run
    from progress_tracker import ActivityType, Stage
    from scraper import scrape_greenhouse, scrape_hn, scrape_lever
    from scorer import RateLimitReached, score_all_jobs
    from theirstack import get_or_discover_slugs

    config = load_config(profile=profile)
    config["_active_profile"] = profile
    init_db(profile=profile)

    enabled = _enabled_sources(config)
    if not any(enabled.values()):
        raise ValueError("No sources are enabled for this profile.")

    run_id = start_run(profile=profile, source="worker")
    total_scraped = 0
    total_filtered = 0
    total_saved = 0
    total_new = 0

    # ── DISCOVERING ──────────────────────────────────────────────────────────
    tracker.start_stage(Stage.DISCOVERING)
    _write_progress(profile, tracker)

    slug_map = {"greenhouse": [], "lever": []}
    if enabled["greenhouse"] or enabled["lever"]:
        slug_map = get_or_discover_slugs(config, profile=profile)
        tracker.set_stage_metrics(
            Stage.DISCOVERING,
            greenhouse=len(slug_map.get("greenhouse", [])),
            lever=len(slug_map.get("lever", [])),
        )
    tracker.complete_stage(Stage.DISCOVERING)

    # ── FETCHING ─────────────────────────────────────────────────────────────
    tracker.start_stage(Stage.FETCHING)
    _write_progress(profile, tracker)

    if enabled["greenhouse"]:
        tracker.register_source(
            "Greenhouse",
            len(config.get("sources", {}).get("greenhouse", {}).get("companies", [])),
        )
        tracker.start_source("Greenhouse")
        result = scrape_greenhouse(config, slugs=slug_map["greenhouse"], profile=profile)
        total_new += result.get("new_jobs_saved", 0)
        total_scraped += result.get("jobs_scraped", 0)
        total_filtered += result.get("jobs_filtered", 0)
        total_saved += result.get("new_jobs_saved", 0)
        tracker.complete_source("Greenhouse", jobs_found=result.get("new_jobs_saved", 0))
        tracker.set_stage_metrics(Stage.FETCHING, greenhouse_new=result.get("new_jobs_saved", 0))
        _write_progress(profile, tracker)

    if enabled["lever"]:
        tracker.register_source(
            "Lever",
            len(config.get("sources", {}).get("lever", {}).get("companies", [])),
        )
        tracker.start_source("Lever")
        result = scrape_lever(config, slugs=slug_map["lever"], profile=profile)
        total_new += result.get("new_jobs_saved", 0)
        total_scraped += result.get("jobs_scraped", 0)
        total_filtered += result.get("jobs_filtered", 0)
        total_saved += result.get("new_jobs_saved", 0)
        tracker.complete_source("Lever", jobs_found=result.get("new_jobs_saved", 0))
        tracker.set_stage_metrics(Stage.FETCHING, lever_new=result.get("new_jobs_saved", 0))
        _write_progress(profile, tracker)

    if enabled["hackernews"]:
        tracker.register_source("HN Who's Hiring", 1)
        tracker.start_source("HN Who's Hiring")
        result = scrape_hn(config, profile=profile)
        total_new += result.get("new_jobs_saved", 0)
        total_scraped += result.get("jobs_scraped", 0)
        total_filtered += result.get("jobs_filtered", 0)
        total_saved += result.get("new_jobs_saved", 0)
        tracker.complete_source("HN Who's Hiring", jobs_found=result.get("new_jobs_saved", 0))
        tracker.set_stage_metrics(Stage.FETCHING, hn_new=result.get("new_jobs_saved", 0))
        _write_progress(profile, tracker)

    tracker.complete_stage(Stage.FETCHING)

    # ── SCRAPING (folded into FETCHING above) ────────────────────────────────
    tracker.start_stage(Stage.SCRAPING)
    tracker.complete_stage(Stage.SCRAPING)

    # ── SCORING ──────────────────────────────────────────────────────────────
    tracker.start_stage(Stage.SCORING)
    _write_progress(profile, tracker)

    def _on_job_scored(i: int, total: int, result) -> None:
        tracker.set_stage_metrics(Stage.SCORING, scored=i, total=total)
        if i % 5 == 0:
            _write_progress(profile, tracker)

    scored_results = score_all_jobs(
        config, yes=True, profile=profile, on_job_scored=_on_job_scored
    )
    scored_ok = [r for r in scored_results if r.get("fit_score", 0) > 0]
    avg_fit = (
        round(sum(r["fit_score"] for r in scored_ok) / len(scored_ok), 1)
        if scored_ok
        else 0.0
    )
    tracker.complete_stage(Stage.SCORING)
    tracker.set_stage_metrics(Stage.SCORING, scored=len(scored_ok))

    # ── FINALIZING ────────────────────────────────────────────────────────────
    tracker.start_stage(Stage.FINALIZING)
    finish_run(
        run_id,
        jobs_scraped=total_scraped,
        jobs_filtered=total_filtered,
        jobs_saved=total_saved,
        jobs_scored=len(scored_ok),
        avg_fit_score=avg_fit,
        errors=[],
        status="complete",
        profile=profile,
    )
    tracker.total_jobs_new = total_new
    tracker.complete_stage(Stage.FINALIZING)
    tracker.log_activity(
        f"Pipeline complete: {total_new} new jobs and {len(scored_ok)} scored.",
        ActivityType.METRIC_UPDATE,
    )
    _write_progress(profile, tracker)

    return len(scored_ok)


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily pipeline runner")
    parser.add_argument("--profile", required=True, help="Profile name (e.g. manav)")
    args = parser.parse_args()

    profile = args.profile
    configure_logging(profile=profile, debug=False)

    profile_dir = _profiles_dir() / profile

    if not profile_dir.exists():
        logger.error(f"profiles/{profile}/ does not exist.")
        sys.exit(1)

    if not (profile_dir / "config.yaml").exists():
        logger.error(f"profiles/{profile}/config.yaml not found.")
        sys.exit(1)

    if not _acquire_lock(profile):
        sys.exit(1)

    started_at = _now_iso()
    logger.info(f"Starting pipeline for profile '{profile}' at {started_at}")

    from progress_tracker import ProgressTracker
    tracker = ProgressTracker()

    try:
        jobs_scored = _run(profile, tracker)
        _write_status(profile, started_at, "success", jobs_scored=jobs_scored)
        logger.info(
            f"Pipeline complete for '{profile}'. "
            f"Jobs scored this run: {jobs_scored}"
        )
        sys.exit(0)

    except Exception as exc:
        from scorer import RateLimitReached
        if isinstance(exc, RateLimitReached):
            _write_status(profile, started_at, "failed", error_message="rate_limited")
            logger.warning(f"Pipeline rate limited for '{profile}'.")
        else:
            msg = str(exc)
            _write_status(profile, started_at, "failed", error_message=msg)
            logger.error(f"Pipeline error: {msg}")
        sys.exit(1)

    finally:
        _release_lock(profile)


if __name__ == "__main__":
    main()
