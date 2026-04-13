"""
worker/run_pipeline.py — Daily pipeline runner for a single profile.

Usage:
    python worker/run_pipeline.py --profile manav
    python worker/run_pipeline.py --profile sister

Responsibilities:
- Creates a lockfile at profiles/{profile}/.worker_running on start,
  deletes it on finish (success or failure).
- Checks for a stale lockfile (> 3 hours old) and proceeds anyway.
- Writes a JSON status file at profiles/{profile}/.last_run on finish.
- Calls main.py logic directly (import, not subprocess).
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

_LOCKFILE_NAME  = ".worker_running"
_STATUS_NAME    = ".last_run"
_STALE_SECONDS  = 3 * 3600  # 3 hours


def _profiles_dir() -> Path:
    return _PROJECT_ROOT / "profiles"


def _lockfile_path(profile: str) -> Path:
    return _profiles_dir() / profile / _LOCKFILE_NAME


def _status_path(profile: str) -> Path:
    return _profiles_dir() / profile / _STATUS_NAME


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
            print(
                f"[run_pipeline] Lock exists and is {age / 60:.0f} min old "
                f"(< 3h) — another worker may be running. Aborting.",
                file=sys.stderr,
            )
            return False
        print(
            f"[run_pipeline] Stale lockfile found ({age / 3600:.1f}h old) — "
            "treating as crashed run, proceeding.",
            file=sys.stderr,
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
        print(f"[run_pipeline] Warning: could not remove lockfile: {exc}", file=sys.stderr)


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
    # Write to a temp file then replace atomically so the dashboard never
    # reads a half-written file.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _run(profile: str) -> int:
    """
    Import and call main.main() with --profile and --yes flags injected.
    Returns the number of jobs scored (best-effort), raises on failure.
    """
    # Patch sys.argv so argparse inside main.py sees the flags we want.
    original_argv = sys.argv[:]
    sys.argv = ["main.py", "--profile", profile, "--yes"]

    try:
        import main as _main

        # score_all_jobs returns results list; capture length via monkey-patch.
        _scored_count = [0]
        _orig_score_all = None

        try:
            from scorer import score_all_jobs as _orig_score_all_fn
            import scorer as _scorer_mod

            def _counting_score_all(config, yes=False, profile=None):
                results = _orig_score_all_fn(config, yes=yes, profile=profile)
                _scored_count[0] = len([r for r in results if r.get("fit_score", 0) > 0])
                return results

            _scorer_mod.score_all_jobs = _counting_score_all
            _main.score_all_jobs = _counting_score_all
        except Exception:
            pass  # counting is best-effort; don't let it block the run

        _main.main()
        return _scored_count[0]

    finally:
        sys.argv = original_argv
        # Restore original function if we patched it
        try:
            if _orig_score_all is not None:
                import scorer as _scorer_mod
                _scorer_mod.score_all_jobs = _orig_score_all_fn  # type: ignore[possibly-undefined]
                _main.score_all_jobs = _orig_score_all_fn  # type: ignore[possibly-undefined]
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily pipeline runner")
    parser.add_argument("--profile", required=True, help="Profile name (e.g. manav)")
    args = parser.parse_args()

    profile    = args.profile
    profile_dir = _profiles_dir() / profile

    if not profile_dir.exists():
        print(f"[run_pipeline] ERROR: profiles/{profile}/ does not exist.", file=sys.stderr)
        sys.exit(1)

    if not (profile_dir / "config.yaml").exists():
        print(
            f"[run_pipeline] ERROR: profiles/{profile}/config.yaml not found.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not _acquire_lock(profile):
        sys.exit(1)

    started_at = _now_iso()
    print(f"[run_pipeline] Starting pipeline for profile '{profile}' at {started_at}")

    try:
        jobs_scored = _run(profile)
        _write_status(profile, started_at, "success", jobs_scored=jobs_scored)
        print(
            f"[run_pipeline] Pipeline complete for '{profile}'. "
            f"Jobs scored this run: {jobs_scored}"
        )
        sys.exit(0)

    except SystemExit as exc:
        # main.py may call sys.exit(0) normally — treat exit(0) as success.
        code = exc.code if isinstance(exc.code, int) else 1
        if code == 0:
            _write_status(profile, started_at, "success")
            print(f"[run_pipeline] Pipeline exited cleanly for '{profile}'.")
            sys.exit(0)
        else:
            _write_status(profile, started_at, "failed", error_message=f"sys.exit({code})")
            print(f"[run_pipeline] Pipeline exited with code {code} for '{profile}'.", file=sys.stderr)
            sys.exit(1)

    except Exception as exc:
        msg = str(exc)
        _write_status(profile, started_at, "failed", error_message=msg)
        print(f"[run_pipeline] ERROR: {msg}", file=sys.stderr)
        sys.exit(1)

    finally:
        _release_lock(profile)


if __name__ == "__main__":
    main()
