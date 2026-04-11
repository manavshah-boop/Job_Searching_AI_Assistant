"""
db.py — SQLite setup, job helpers, and configuration loader.

This is the only place that touches the database.
All other files import load_config and DB helpers from here.
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# Re-export load_config so every file imports from one place.
from config import load_config  # noqa: F401

DB_PATH = Path(__file__).parent / "jobs.db"


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class Job:
    id: str              # {source}_{source_id}, used for dedup
    title: str
    company: str
    location: str
    url: str
    raw_text: str        # full text sent to Claude for scoring
    source: str          # "greenhouse" | "hackernews"
    score_attempts: int = 0
    score_error: Optional[str] = None
    status: str = "new"  # new | applied | skipped


def make_id(source: str, source_id: str) -> str:
    """Simple ID format: {source}_{source_id}"""
    return f"{source}_{source_id}"


# ── DB setup ──────────────────────────────────────────────────────────────────

def migrate_db():
    """
    Safe, idempotent schema migrations. Adds new columns/tables if missing.
    Run automatically inside init_db().
    """
    conn = sqlite3.connect(DB_PATH)

    # Add score_attempts and score_error to jobs table
    for col, col_type in [
        ("score_attempts", "INTEGER DEFAULT 0"),
        ("score_error",    "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {col_type}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

    # Create scores table (separate from jobs)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            job_id       TEXT PRIMARY KEY REFERENCES jobs(id),
            fit_score    INTEGER,
            role_fit     INTEGER,
            stack_match  INTEGER,
            seniority    INTEGER,
            location     INTEGER,
            growth       INTEGER,
            compensation INTEGER,
            reasons      TEXT,
            flags        TEXT,
            skill_misses TEXT,
            one_liner    TEXT,
            ats_score    INTEGER,
            scored_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def init_db():
    """Creates the jobs table if it doesn't exist. Safe to call on every run."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id             TEXT PRIMARY KEY,
            title          TEXT NOT NULL,
            company        TEXT NOT NULL,
            location       TEXT,
            url            TEXT,
            raw_text       TEXT,
            source         TEXT,
            score_attempts INTEGER DEFAULT 0,
            score_error    TEXT,
            status         TEXT DEFAULT 'new',
            created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    migrate_db()


# ── Write operations ──────────────────────────────────────────────────────────

def insert_job(job: Job) -> bool:
    """
    Inserts a job. Returns True if new, False if already in DB.
    Dedup is enforced via PRIMARY KEY.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""
            INSERT INTO jobs (id, title, company, location, url, raw_text, source, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (job.id, job.title, job.company, job.location,
              job.url, job.raw_text, job.source, job.status))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def save_score(
    job_id: str,
    fit_score: int,
    role_fit: int,
    stack_match: int,
    seniority: int,
    loc_score: int,
    growth: int,
    compensation: int,
    reasons: str,       # JSON array string
    flags: str,         # JSON array string
    skill_misses: str,  # JSON array string
    one_liner: str,
    ats_score: int,
):
    """Writes scoring results to the scores table (INSERT OR REPLACE)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO scores
            (job_id, fit_score, role_fit, stack_match, seniority, location,
             growth, compensation, reasons, flags, skill_misses, one_liner, ats_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (job_id, fit_score, role_fit, stack_match, seniority, loc_score,
          growth, compensation, reasons, flags, skill_misses, one_liner, ats_score))
    conn.commit()
    conn.close()


def increment_score_attempts(job_id: str):
    """Increment score_attempts before each scoring attempt."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE jobs SET score_attempts = score_attempts + 1 WHERE id = ?",
        (job_id,)
    )
    conn.commit()
    conn.close()


def write_score_error(job_id: str, error: str):
    """Record the last error message for a failed scoring attempt."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE jobs SET score_error = ? WHERE id = ?",
        (str(error)[:500], job_id)
    )
    conn.commit()
    conn.close()


def rescore_reset():
    """Clear all scores and reset attempt counters. Used with --rescore."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM scores")
    conn.execute("UPDATE jobs SET score_attempts = 0, score_error = NULL")
    conn.commit()
    conn.close()


# ── Read operations ───────────────────────────────────────────────────────────

def _row_to_job(row: sqlite3.Row) -> Job:
    """Convert a DB row to a Job, mapping only known Job fields."""
    keys = row.keys()
    # Support aliased column name from JOIN queries (job_location)
    location = row["job_location"] if "job_location" in keys else row["location"]
    return Job(
        id=row["id"],
        title=row["title"],
        company=row["company"],
        location=location,
        url=row["url"],
        raw_text=row["raw_text"],
        source=row["source"],
        score_attempts=row["score_attempts"] if "score_attempts" in keys else 0,
        score_error=row["score_error"] if "score_error" in keys else None,
        status=row["status"],
    )


def get_unscored() -> list:
    """
    Jobs that have not yet been scored successfully and have fewer than
    3 scoring attempts — so persistently failing jobs don't block every run.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT * FROM jobs
        WHERE id NOT IN (SELECT job_id FROM scores)
          AND score_attempts < 3
        ORDER BY created_at DESC
    """).fetchall()
    conn.close()
    return [_row_to_job(r) for r in rows]


def get_top_jobs(min_score: int = 60) -> list:
    """
    Scored jobs above threshold, sorted best first.
    Returns result dicts compatible with print_results().
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT j.id, j.title, j.company, j.location AS job_location, j.url,
               j.raw_text, j.source, j.score_attempts, j.score_error, j.status,
               s.fit_score, s.role_fit, s.stack_match, s.seniority,
               s.location AS dim_location, s.growth, s.compensation,
               s.reasons, s.flags, s.skill_misses, s.one_liner, s.ats_score
        FROM jobs j
        JOIN scores s ON j.id = s.job_id
        WHERE s.fit_score >= ?
        ORDER BY s.fit_score DESC
    """, (min_score,)).fetchall()
    conn.close()

    results = []
    for row in rows:
        job = _row_to_job(row)
        results.append({
            "job": job,
            "fit_score": row["fit_score"],
            "ats_score": row["ats_score"] or 0,
            "reasons":      json.loads(row["reasons"]      or "[]"),
            "flags":        json.loads(row["flags"]        or "[]"),
            "skill_misses": json.loads(row["skill_misses"] or "[]"),
            "one_liner": row["one_liner"] or "",
            "dimension_scores": {
                "role_fit":     row["role_fit"],
                "stack_match":  row["stack_match"],
                "seniority":    row["seniority"],
                "location":     row["dim_location"],
                "growth":       row["growth"],
                "compensation": row["compensation"],
            },
        })
    return results


def get_all_jobs() -> list:
    """Everything in the jobs table (no score data), unscored or not."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM jobs ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [_row_to_job(r) for r in rows]


def count_jobs() -> dict:
    """Quick stats — useful for end-of-run summary."""
    conn = sqlite3.connect(DB_PATH)
    total  = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    scored = conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
    conn.close()
    return {"total": total, "scored": scored}


# ── Discovered slugs (TheirStack cache) ──────────────────────────────────────

_VALID_ATS = {"greenhouse", "lever"}


def _ensure_discovered_slugs_table(conn: sqlite3.Connection, ats: str) -> None:
    """Creates the per-ATS slug cache table if it doesn't exist."""
    if ats not in _VALID_ATS:
        raise ValueError(f"Invalid ATS: {ats!r}. Must be one of {_VALID_ATS}")
    table = f"discovered_{ats}_slugs"
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            slug         TEXT PRIMARY KEY,
            company_name TEXT,
            discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def save_discovered_slug(slug: str, company_name: str, ats: str = "greenhouse") -> None:
    """Caches a resolved slug for the given ATS so we don't re-resolve it next run."""
    conn = sqlite3.connect(DB_PATH)
    try:
        _ensure_discovered_slugs_table(conn, ats)
        table = f"discovered_{ats}_slugs"
        conn.execute(
            f"INSERT OR IGNORE INTO {table} (slug, company_name) VALUES (?, ?)",
            (slug, company_name),
        )
        conn.commit()
    finally:
        conn.close()


def load_discovered_slugs(ats: str = "greenhouse") -> list:
    """Returns all previously cached slugs for the given ATS, newest first."""
    conn = sqlite3.connect(DB_PATH)
    try:
        _ensure_discovered_slugs_table(conn, ats)
        table = f"discovered_{ats}_slugs"
        rows = conn.execute(
            f"SELECT slug FROM {table} ORDER BY discovered_at DESC"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def clear_discovered_slugs(ats: Optional[str] = None) -> None:
    """
    Clears the slug cache.
    Pass ats='greenhouse' or 'lever' to clear one table, or None to clear both.
    """
    targets = _VALID_ATS if ats is None else {ats}
    conn = sqlite3.connect(DB_PATH)
    try:
        for target in targets:
            _ensure_discovered_slugs_table(conn, target)
            conn.execute(f"DELETE FROM discovered_{target}_slugs")
        conn.commit()
    finally:
        conn.close()


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    print(f"DB initialized at {DB_PATH}")

    test_job = Job(
        id=make_id("test", "1"),
        title="Software Engineer",
        company="TestCo",
        location="Remote",
        url="https://example.com/jobs/1",
        raw_text="We are looking for a Python engineer...",
        source="test",
    )

    inserted = insert_job(test_job)
    print(f"Test job inserted: {inserted}")

    duplicate = insert_job(test_job)
    print(f"Duplicate rejected: {not duplicate}")

    increment_score_attempts(test_job.id)
    stats = count_jobs()
    print(f"DB stats: {stats}")
