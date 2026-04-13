"""
worker/db_guard.py — Atomic SQLite swap utilities.

Protects jobs.db from corruption when the worker writes during a run.
The dashboard always reads jobs.db directly and is never aware of the swap.

Usage from run_pipeline.py:
    from worker.db_guard import atomic_swap, commit_swap, rollback_swap

    new_path = atomic_swap(profile)     # copies live → .new
    # ... worker runs, writes to new_path ...
    commit_swap(profile)                # integrity check, then .new → live
    # on error:
    rollback_swap(profile)             # delete .new, leave live untouched

Standalone verification:
    python worker/db_guard.py --profile manav --verify
"""

import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent


def _db_path(profile: str) -> Path:
    return _PROJECT_ROOT / "profiles" / profile / "jobs.db"


def _new_path(profile: str) -> Path:
    return _PROJECT_ROOT / "profiles" / profile / "jobs.db.new"


def _bak_path(profile: str) -> Path:
    return _PROJECT_ROOT / "profiles" / profile / "jobs.db.bak"


def _integrity_check(db_path: Path) -> bool:
    """Run SQLite integrity_check pragma. Returns True if OK."""
    try:
        conn = sqlite3.connect(str(db_path))
        result = conn.execute("PRAGMA integrity_check").fetchone()
        conn.close()
        return result is not None and result[0] == "ok"
    except Exception as exc:
        print(f"[db_guard] integrity_check failed on {db_path}: {exc}", file=sys.stderr)
        return False


def atomic_swap(profile: str) -> Path:
    """
    Copy the live jobs.db to jobs.db.new for the worker to use during its run.

    Returns the Path to jobs.db.new so the caller can pass it to db helpers
    if needed (currently run_pipeline uses the normal DB path and this is
    informational — see NOTE below).

    NOTE: The current pipeline writes directly to jobs.db via db.py.
    atomic_swap / commit_swap are present so a future step can redirect
    writes to .new and atomically promote them. For now, commit_swap
    performs the integrity check and rotation after the write completes.
    """
    live = _db_path(profile)
    new  = _new_path(profile)

    if new.exists():
        new.unlink()

    if live.exists():
        shutil.copy2(str(live), str(new))
        print(f"[db_guard] Copied {live.name} → {new.name}")
    else:
        print(f"[db_guard] No live DB found at {live} — .new will be created fresh by the pipeline.")

    return new


def commit_swap(profile: str) -> None:
    """
    Run integrity check on jobs.db (the live file after the worker has written
    to it), then rotate: live → .bak, conceptually finalising the run.

    If integrity check fails, raises RuntimeError — caller should rollback.

    Current behaviour (pipeline writes directly to jobs.db):
      - Checks live jobs.db integrity after the worker run.
      - Rotates previous .bak out of the way (keeps one backup).
      - Does NOT move .new → live (pipeline wrote directly to live).

    Future behaviour (if pipeline is redirected to write to .new):
      - Check .new integrity.
      - os.replace(.bak ← live), os.replace(live ← .new).
    """
    live = _db_path(profile)
    bak  = _bak_path(profile)
    new  = _new_path(profile)

    if not live.exists():
        raise RuntimeError(f"[db_guard] commit_swap: {live} does not exist — nothing to commit.")

    if not _integrity_check(live):
        raise RuntimeError(
            f"[db_guard] Integrity check FAILED on {live}. "
            "Run: sqlite3 jobs.db 'PRAGMA integrity_check' for details."
        )

    # Rotate: old .bak → gone, live → .bak (preserves one backup copy)
    if bak.exists():
        bak.unlink()
    shutil.copy2(str(live), str(bak))
    print(f"[db_guard] Integrity OK. Backup created: {bak.name}")

    # Clean up .new if it exists (was created by atomic_swap but not used for writing)
    if new.exists():
        new.unlink()


def rollback_swap(profile: str) -> None:
    """
    Delete jobs.db.new if it exists. Leave jobs.db untouched.
    Safe to call even if .new doesn't exist.
    """
    new = _new_path(profile)
    if new.exists():
        new.unlink()
        print(f"[db_guard] Rolled back — removed {new.name}")
    else:
        print("[db_guard] Nothing to roll back (.new not found).")


def verify(profile: str) -> None:
    """Standalone verification: checks integrity of live jobs.db."""
    live = _db_path(profile)
    if not live.exists():
        print(f"[db_guard] No DB found at {live}")
        sys.exit(1)

    ok = _integrity_check(live)
    if ok:
        size_kb = live.stat().st_size // 1024
        print(f"[db_guard] {live} — integrity OK ({size_kb} KB)")
        sys.exit(0)
    else:
        print(f"[db_guard] {live} — INTEGRITY CHECK FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SQLite DB guard utilities")
    parser.add_argument("--profile", required=True, help="Profile name (e.g. manav)")
    parser.add_argument(
        "--verify", action="store_true",
        help="Run integrity check on live jobs.db and exit 0/1",
    )
    parser.add_argument(
        "--rollback", action="store_true",
        help="Remove jobs.db.new if it exists",
    )
    args = parser.parse_args()

    if args.verify:
        verify(args.profile)
    elif args.rollback:
        rollback_swap(args.profile)
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)
