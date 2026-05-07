"""
test_db_maintenance.py — Regression tests for db.py startup maintenance:
  - HIGH-5 WAL pragma is applied on init
  - MED-7 stale 'running' scrape_runs are marked 'crashed' on startup
  - HIGH-4 set_active_profile is thread-safe
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

import db
from db import (
    _STALE_RUN_HOURS,
    finish_run,
    get_active_profile,
    init_db,
    set_active_profile,
    start_run,
)


@pytest.fixture
def isolated_profile(monkeypatch, tmp_path):
    """Point _PROFILES_DIR at a fresh tmp dir and clear the active profile."""
    monkeypatch.setattr(db, "_PROFILES_DIR", tmp_path)
    set_active_profile(None)
    yield tmp_path
    set_active_profile(None)


def test_init_db_enables_wal_mode(isolated_profile, monkeypatch):
    """HIGH-5 regression: WAL mode must be enabled after init_db()."""
    init_db(profile="walprof")
    db_path = isolated_profile / "walprof" / "jobs.db"
    assert db_path.exists()

    conn = sqlite3.connect(db_path)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    assert mode.lower() == "wal"


def test_init_db_marks_stale_running_runs_crashed(isolated_profile):
    """
    MED-7 regression: a scrape_runs row stuck in 'running' for longer than
    _STALE_RUN_HOURS must be marked 'crashed' on the next init_db() call.
    """
    init_db(profile="staleprof")
    run_id = start_run(profile="staleprof", source="test")

    # Backdate started_at past the stale threshold.
    db_path = isolated_profile / "staleprof" / "jobs.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE scrape_runs SET started_at = datetime('now', ?) WHERE run_id = ?",
        (f"-{_STALE_RUN_HOURS + 1} hours", run_id),
    )
    conn.commit()
    conn.close()

    # Re-init triggers the sweep.
    init_db(profile="staleprof")

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT status, finished_at FROM scrape_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    conn.close()
    assert row is not None
    status, finished_at = row
    assert status == "crashed"
    assert finished_at  # finish_at was populated by the sweep


def test_init_db_leaves_fresh_running_runs_alone(isolated_profile):
    """A fresh run should NOT be flagged as crashed."""
    init_db(profile="freshprof")
    run_id = start_run(profile="freshprof", source="test")

    init_db(profile="freshprof")

    db_path = isolated_profile / "freshprof" / "jobs.db"
    conn = sqlite3.connect(db_path)
    status = conn.execute(
        "SELECT status FROM scrape_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()[0]
    conn.close()
    assert status == "running"


def test_init_db_does_not_touch_finished_runs(isolated_profile):
    """Already-completed runs must not be re-marked."""
    init_db(profile="doneprof")
    run_id = start_run(profile="doneprof", source="test")
    finish_run(run_id, status="complete", profile="doneprof")

    init_db(profile="doneprof")

    db_path = isolated_profile / "doneprof" / "jobs.db"
    conn = sqlite3.connect(db_path)
    status = conn.execute(
        "SELECT status FROM scrape_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()[0]
    conn.close()
    assert status == "complete"


def test_set_active_profile_is_thread_safe():
    """
    HIGH-4 smoke test: a swarm of threads writing/reading _ACTIVE_PROFILE
    must not raise and the final read must return one of the values written.
    """
    values = ["alpha", "beta", "gamma", "delta", None]
    final_reads: list = []
    stop = threading.Event()

    def writer(name):
        for _ in range(200):
            if stop.is_set():
                return
            set_active_profile(name)

    def reader():
        for _ in range(500):
            if stop.is_set():
                return
            final_reads.append(get_active_profile())

    threads = [threading.Thread(target=writer, args=(v,)) for v in values]
    threads += [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    stop.set()

    assert final_reads, "reader threads should produce some reads"
    # Every read must be one of the writeable values (no torn reads / garbage).
    valid = set(values)
    assert all(v in valid for v in final_reads)
