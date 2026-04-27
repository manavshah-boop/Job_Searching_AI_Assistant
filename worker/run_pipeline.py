"""
worker/run_pipeline.py - Out-of-process pipeline runner for a single profile.

Responsibilities:
- manage the worker lockfile
- write final run status JSON
- persist live progress JSON
- initialize a ProgressTracker
- delegate pipeline orchestration to pipeline.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import load_config
from db import init_db
from logging_config import configure_logging
from loguru import logger
from pipeline import PipelineOptions, run_full_pipeline
from progress_tracker import ProgressTracker

_LOCKFILE_NAME = ".worker_running"
_STATUS_NAME = ".last_run"
_PROGRESS_NAME = ".run_progress.json"
_STALE_SECONDS = 3 * 3600


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
    lock = _lockfile_path(profile)
    if lock.exists():
        age = time.time() - lock.stat().st_mtime
        if age < _STALE_SECONDS:
            logger.warning("Lock exists and is {:.0f} min old (< 3h) - another worker may be running. Aborting.", age / 60)
            return False
        logger.info("Stale lockfile found ({:.1f}h old) - treating as crashed run, proceeding.", age / 3600)
        lock.unlink()

    lock.write_text(json.dumps({"pid": os.getpid(), "started_at": _now_iso()}), encoding="utf-8")
    return True


def _release_lock(profile: str) -> None:
    try:
        _lockfile_path(profile).unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Could not remove lockfile: {}", exc)


def _write_progress(profile: str, tracker: ProgressTracker) -> None:
    path = _progress_path(profile)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(tracker.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning("Could not write progress JSON: {}", exc)


def _write_status(
    profile: str,
    started_at: str,
    status: str,
    *,
    jobs_scored: int = 0,
    error_message: str = "",
) -> None:
    payload = {
        "started_at": started_at,
        "finished_at": _now_iso(),
        "status": status,
        "jobs_scored": jobs_scored,
        "error_message": error_message,
    }
    path = _status_path(profile)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _run(profile: str, tracker: ProgressTracker) -> int:
    config = load_config(profile=profile)
    config["_active_profile"] = profile
    init_db(profile=profile)

    result = run_full_pipeline(
        config,
        profile,
        PipelineOptions(
            scrape=True,
            score=True,
            embed=True,
            yes=True,
            force_embed=False,
            rescore=False,
            run_source="worker",
            on_progress=lambda updated_tracker: _write_progress(profile, updated_tracker),
        ),
        progress_tracker=tracker,
    )
    return result.score.jobs_scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily pipeline runner")
    parser.add_argument("--profile", required=True, help="Profile name (e.g. manav)")
    args = parser.parse_args()

    profile = args.profile
    configure_logging(profile=profile, debug=False)

    profile_dir = _profiles_dir() / profile
    if not profile_dir.exists():
        logger.error("profiles/{}/ does not exist.", profile)
        sys.exit(1)
    if not (profile_dir / "config.yaml").exists():
        logger.error("profiles/{}/config.yaml not found.", profile)
        sys.exit(1)
    if not _acquire_lock(profile):
        sys.exit(1)

    started_at = _now_iso()
    logger.info("Starting pipeline for profile '{}' at {}", profile, started_at)
    tracker = ProgressTracker()

    try:
        jobs_scored = _run(profile, tracker)
        _write_status(profile, started_at, "success", jobs_scored=jobs_scored)
        logger.info("Pipeline complete for '{}'. Jobs scored this run: {}", profile, jobs_scored)
        sys.exit(0)
    except Exception as exc:
        _write_status(profile, started_at, "failed", error_message=str(exc))
        logger.error("Pipeline error: {}", exc)
        sys.exit(1)
    finally:
        _release_lock(profile)


if __name__ == "__main__":
    main()
