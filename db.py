"""
db.py — SQLite setup, job helpers, and configuration loader.

This is the only place that touches the database.
All other files import load_config and DB helpers from here.

Profile support:
  Call set_active_profile("manav") at startup to scope all DB operations to
  profiles/manav/jobs.db. With no active profile, falls back to root jobs.db.
  Every public function also accepts an explicit profile= kwarg for one-off
  overrides (e.g. onboarding creating a new profile's DB).
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Re-export load_config so every file imports from one place.
from config import load_config  # noqa: F401

_PROFILES_DIR = Path(__file__).parent / "profiles"
_ACTIVE_PROFILE: Optional[str] = None


def set_active_profile(profile: Optional[str]) -> None:
    """Set the module-level active profile for all subsequent DB operations."""
    global _ACTIVE_PROFILE
    _ACTIVE_PROFILE = profile


def get_db_path(profile: Optional[str] = None) -> Path:
    """
    Returns the DB path for the given profile.
    Resolution order: explicit arg → _ACTIVE_PROFILE → root jobs.db
    """
    resolved = profile or _ACTIVE_PROFILE
    if resolved:
        return _PROFILES_DIR / resolved / "jobs.db"
    return Path(__file__).parent / "jobs.db"


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class Job:
    id: str              # {source}_{source_id}, used for dedup
    title: str
    company: str
    location: str
    url: str
    raw_text: str        # full text sent to LLM for scoring
    source: str          # "greenhouse" | "lever" | "hackernews"
    score_attempts: int = 0
    score_error: Optional[str] = None
    status: str = "new"  # new | applied | skipped
    scrape_qualified: int = 1       # 1 = passed filters, 0 = rejected at scrape time
    scrape_filter_reason: str = ""  # e.g. "title_blocklist: Staff", "yoe_max: 8 > 5"


def make_id(source: str, source_id: str) -> str:
    """Simple ID format: {source}_{source_id}"""
    return f"{source}_{source_id}"


# ── DB setup ──────────────────────────────────────────────────────────────────

def _migrate_db(db_path: Path) -> None:
    """
    Safe, idempotent schema migrations. Adds new columns/tables if missing.
    Run automatically inside init_db().
    """
    conn = sqlite3.connect(db_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            job_id            TEXT PRIMARY KEY REFERENCES jobs(id),
            fit_score         INTEGER,
            role_fit          INTEGER,
            stack_match       INTEGER,
            seniority         INTEGER,
            location          INTEGER,
            growth            INTEGER,
            compensation      INTEGER,
            reasons           TEXT,
            flags             TEXT,
            skill_misses      TEXT,
            one_liner         TEXT,
            ats_score         INTEGER,
            disqualified      INTEGER DEFAULT 0,
            disqualify_reason TEXT DEFAULT '',
            scored_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    for col, col_type in [
        ("score_attempts", "INTEGER DEFAULT 0"),
        ("score_error",    "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {col_type}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

    for col, col_type in [
        ("scrape_qualified",     "INTEGER DEFAULT 1"),
        ("scrape_filter_reason", "TEXT DEFAULT ''"),
    ]:
        try:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {col_type}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

    for col, col_type in [
        ("disqualified",      "INTEGER DEFAULT 0"),
        ("disqualify_reason", "TEXT DEFAULT ''"),
    ]:
        try:
            conn.execute(f"ALTER TABLE scores ADD COLUMN {col} {col_type}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scrape_runs (
            run_id        TEXT PRIMARY KEY,
            profile       TEXT,
            started_at    TIMESTAMP NOT NULL,
            finished_at   TIMESTAMP,
            source        TEXT,
            jobs_scraped  INTEGER DEFAULT 0,
            jobs_filtered INTEGER DEFAULT 0,
            jobs_saved    INTEGER DEFAULT 0,
            jobs_scored   INTEGER DEFAULT 0,
            avg_fit_score REAL,
            errors        TEXT DEFAULT '[]',
            status        TEXT DEFAULT 'running'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS job_embeddings (
            job_id       TEXT NOT NULL REFERENCES jobs(id),
            model_name   TEXT NOT NULL,
            chunk_key    TEXT NOT NULL,
            chunk_order  INTEGER DEFAULT 0,
            chunk_text   TEXT NOT NULL,
            embedding    TEXT NOT NULL,
            dimensions   INTEGER NOT NULL,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (job_id, model_name, chunk_key)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_job_embeddings_model_job
        ON job_embeddings (model_name, job_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_job_embeddings_job
        ON job_embeddings (job_id)
    """)
    conn.commit()
    conn.close()


def init_db(profile: Optional[str] = None) -> None:
    """
    Creates the jobs table if it doesn't exist. Safe to call on every run.
    Creates the profile folder if it doesn't already exist.
    """
    db_path = get_db_path(profile)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
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
    _migrate_db(db_path)


# ── Write operations ──────────────────────────────────────────────────────────

def insert_job(job: Job, profile: Optional[str] = None) -> bool:
    """
    Inserts a job. Returns True if new, False if already in DB.
    Dedup is enforced via PRIMARY KEY.
    """
    conn = sqlite3.connect(get_db_path(profile))
    try:
        conn.execute("""
            INSERT INTO jobs
                (id, title, company, location, url, raw_text, source, status,
                 scrape_qualified, scrape_filter_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (job.id, job.title, job.company, job.location,
              job.url, job.raw_text, job.source, job.status,
              job.scrape_qualified, job.scrape_filter_reason))
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
    reasons: str,           # JSON array string
    flags: str,             # JSON array string
    skill_misses: str,      # JSON array string
    one_liner: str,
    ats_score: int,
    disqualified: int = 0,
    disqualify_reason: str = "",
    profile: Optional[str] = None,
) -> None:
    """Writes scoring results to the scores table (INSERT OR REPLACE)."""
    conn = sqlite3.connect(get_db_path(profile))
    conn.execute("""
        INSERT OR REPLACE INTO scores
            (job_id, fit_score, role_fit, stack_match, seniority, location,
             growth, compensation, reasons, flags, skill_misses, one_liner, ats_score,
             disqualified, disqualify_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (job_id, fit_score, role_fit, stack_match, seniority, loc_score,
          growth, compensation, reasons, flags, skill_misses, one_liner, ats_score,
          disqualified, disqualify_reason))
    conn.commit()
    conn.close()


def increment_score_attempts(job_id: str, profile: Optional[str] = None) -> None:
    """Increment score_attempts before each scoring attempt."""
    conn = sqlite3.connect(get_db_path(profile))
    conn.execute(
        "UPDATE jobs SET score_attempts = score_attempts + 1 WHERE id = ?",
        (job_id,)
    )
    conn.commit()
    conn.close()


def write_score_error(job_id: str, error: str, profile: Optional[str] = None) -> None:
    """Record the last error message for a failed scoring attempt."""
    conn = sqlite3.connect(get_db_path(profile))
    conn.execute(
        "UPDATE jobs SET score_error = ? WHERE id = ?",
        (str(error)[:500], job_id)
    )
    conn.commit()
    conn.close()


def rescore_reset(profile: Optional[str] = None) -> None:
    """Clear all scores and reset attempt counters. Used with --rescore."""
    conn = sqlite3.connect(get_db_path(profile))
    conn.execute("DELETE FROM scores")
    conn.execute("UPDATE jobs SET score_attempts = 0, score_error = NULL")
    conn.commit()
    conn.close()


def update_job_status(job_id: str, status: str, profile: Optional[str] = None) -> None:
    """Update a job's status (new | applied | skipped)."""
    conn = sqlite3.connect(get_db_path(profile))
    conn.execute("UPDATE jobs SET status = ? WHERE id = ?", (status, job_id))
    conn.commit()
    conn.close()


# ── Read operations ───────────────────────────────────────────────────────────

def _row_to_job(row: sqlite3.Row) -> Job:
    """Convert a DB row to a Job, mapping only known Job fields."""
    keys = row.keys()
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
        scrape_qualified=row["scrape_qualified"] if "scrape_qualified" in keys else 1,
        scrape_filter_reason=row["scrape_filter_reason"] if "scrape_filter_reason" in keys else "",
    )


def _load_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value in (None, ""):
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]


def _normalize_job_with_score_record(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    record = dict(row)
    record["reasons"] = _load_json_list(record.get("reasons"))
    record["flags"] = _load_json_list(record.get("flags"))
    record["skill_misses"] = _load_json_list(record.get("skill_misses"))
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


def get_unscored(profile: Optional[str] = None) -> list:
    """
    Jobs that have not yet been scored successfully and have fewer than
    3 scoring attempts — so persistently failing jobs don't block every run.
    """
    conn = sqlite3.connect(get_db_path(profile))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT * FROM jobs
        WHERE id NOT IN (SELECT job_id FROM scores)
          AND score_attempts < 3
          AND scrape_qualified = 1
        ORDER BY created_at DESC
    """).fetchall()
    conn.close()
    return [_row_to_job(r) for r in rows]


def get_scored_jobs_for_embedding(
    model_name: str,
    profile: Optional[str] = None,
    limit: Optional[int] = None,
    force: bool = False,
) -> list[Job]:
    """
    Return scrape-qualified jobs that completed scoring and need embeddings.

    When force=False, only jobs with no embeddings for the given model are returned.
    When force=True, all scrape-qualified scored jobs are returned for refresh runs.
    """
    init_db(profile=profile)
    conn = sqlite3.connect(get_db_path(profile))
    conn.row_factory = sqlite3.Row

    sql = """
        SELECT j.*
        FROM jobs j
        JOIN scores s ON s.job_id = j.id
        WHERE j.scrape_qualified = 1
    """
    params: list[Any] = []

    if not force:
        sql += """
          AND NOT EXISTS (
              SELECT 1
              FROM job_embeddings e
              WHERE e.job_id = j.id
                AND e.model_name = ?
          )
        """
        params.append(model_name)

    sql += " ORDER BY j.created_at DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [_row_to_job(r) for r in rows]


def replace_job_embeddings(
    job_id: str,
    model_name: str,
    rows: Sequence[Dict[str, Any]],
    profile: Optional[str] = None,
) -> None:
    """
    Replace all embeddings for one job/model pair in a single transaction.

    Each row should contain: chunk_key, chunk_order, chunk_text, embedding, dimensions.
    """
    conn = sqlite3.connect(get_db_path(profile))
    try:
        conn.execute(
            "DELETE FROM job_embeddings WHERE job_id = ? AND model_name = ?",
            (job_id, model_name),
        )
        if rows:
            conn.executemany(
                """
                INSERT INTO job_embeddings
                    (job_id, model_name, chunk_key, chunk_order, chunk_text, embedding, dimensions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        job_id,
                        model_name,
                        row["chunk_key"],
                        row["chunk_order"],
                        row["chunk_text"],
                        row["embedding"],
                        row["dimensions"],
                    )
                    for row in rows
                ],
            )
        conn.commit()
    finally:
        conn.close()


def get_job_embeddings(
    job_id: str,
    profile: Optional[str] = None,
    model_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch persisted embeddings for a job, ordered by chunk_order."""
    init_db(profile=profile)
    conn = sqlite3.connect(get_db_path(profile))
    conn.row_factory = sqlite3.Row
    sql = """
        SELECT job_id, model_name, chunk_key, chunk_order, chunk_text, embedding, dimensions, created_at
        FROM job_embeddings
        WHERE job_id = ?
    """
    params: list[Any] = [job_id]
    if model_name:
        sql += " AND model_name = ?"
        params.append(model_name)
    sql += " ORDER BY chunk_order ASC, chunk_key ASC"
    rows = conn.execute(sql, params).fetchall()
    conn.close()

    result: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["embedding"] = json.loads(item["embedding"])
        except json.JSONDecodeError:
            item["embedding"] = []
        result.append(item)
    return result


def get_embedding_index_rows(
    model_name: str,
    profile: Optional[str] = None,
    job_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Return SQLite-backed embedding rows joined with job and score metadata.

    This is the canonical source for rebuilding or refreshing the vector index.
    """
    init_db(profile=profile)
    conn = sqlite3.connect(get_db_path(profile))
    conn.row_factory = sqlite3.Row

    sql = """
        SELECT
            ? AS profile,
            e.job_id,
            e.model_name,
            e.chunk_key,
            e.chunk_order,
            e.chunk_text,
            e.embedding,
            e.dimensions,
            e.created_at,
            j.title,
            j.company,
            j.source,
            j.url,
            j.status,
            s.scored_at,
            s.fit_score,
            s.ats_score
        FROM job_embeddings e
        JOIN jobs j ON j.id = e.job_id
        LEFT JOIN scores s ON s.job_id = e.job_id
        WHERE e.model_name = ?
    """
    params: list[Any] = [profile or "", model_name]

    if job_ids:
        placeholders = ",".join("?" for _ in job_ids)
        sql += f" AND e.job_id IN ({placeholders})"
        params.extend(job_ids)

    sql += " ORDER BY e.job_id ASC, e.chunk_order ASC, e.chunk_key ASC"
    rows = conn.execute(sql, params).fetchall()
    conn.close()

    result: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["embedding"] = json.loads(item["embedding"])
        except json.JSONDecodeError:
            item["embedding"] = []
        result.append(item)
    return result


def clear_job_embeddings(
    profile: Optional[str] = None,
    model_name: Optional[str] = None,
) -> None:
    """Delete persisted embeddings, optionally scoped to a single model."""
    init_db(profile=profile)
    conn = sqlite3.connect(get_db_path(profile))
    try:
        if model_name:
            conn.execute("DELETE FROM job_embeddings WHERE model_name = ?", (model_name,))
        else:
            conn.execute("DELETE FROM job_embeddings")
        conn.commit()
    finally:
        conn.close()


def get_top_jobs(min_score: int = 60, profile: Optional[str] = None) -> list:
    """
    Scored jobs above threshold, sorted best first.
    Returns result dicts compatible with print_results().
    """
    conn = sqlite3.connect(get_db_path(profile))
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
            "reasons": _load_json_list(row["reasons"]),
            "flags": _load_json_list(row["flags"]),
            "skill_misses": _load_json_list(row["skill_misses"]),
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


def get_job_with_score(job_id: str, profile: Optional[str] = None) -> Optional[dict[str, Any]]:
    """
    Fetch one job joined with any available score fields and return a normalized dict.

    This is the shared read path for job detail views, explanations, and CLI/dashboard
    integrations that need parsed reasons/flags/skill misses plus dimension scores.
    """
    init_db(profile=profile)
    conn = sqlite3.connect(get_db_path(profile))
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
        return _normalize_job_with_score_record(row)
    finally:
        conn.close()


def get_all_jobs(profile: Optional[str] = None) -> list:
    """Everything in the jobs table (no score data), unscored or not."""
    conn = sqlite3.connect(get_db_path(profile))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM jobs ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [_row_to_job(r) for r in rows]


def count_jobs(profile: Optional[str] = None) -> dict:
    """Quick stats — useful for end-of-run summary."""
    init_db(profile=profile)
    conn = sqlite3.connect(get_db_path(profile))
    total  = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    scored = conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
    embedded = conn.execute("SELECT COUNT(DISTINCT job_id) FROM job_embeddings").fetchone()[0]
    conn.close()
    return {"total": total, "scored": scored, "embedded": embedded}


# ── Discovered slugs (TheirStack cache) ──────────────────────────────────────

_VALID_ATS = {"greenhouse", "lever", "ashby", "workable"}


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


def save_discovered_slug(
    slug: str,
    company_name: str,
    ats: str = "greenhouse",
    profile: Optional[str] = None,
) -> None:
    """Caches a resolved slug for the given ATS so we don't re-resolve it next run."""
    conn = sqlite3.connect(get_db_path(profile))
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


def load_discovered_slugs(ats: str = "greenhouse", profile: Optional[str] = None) -> list:
    """Returns all previously cached slugs for the given ATS, newest first."""
    conn = sqlite3.connect(get_db_path(profile))
    try:
        _ensure_discovered_slugs_table(conn, ats)
        table = f"discovered_{ats}_slugs"
        rows = conn.execute(
            f"SELECT slug FROM {table} ORDER BY discovered_at DESC"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def clear_discovered_slugs(ats: Optional[str] = None, profile: Optional[str] = None) -> None:
    """
    Clears the slug cache.
    Pass ats='greenhouse' or 'lever' to clear one table, or None to clear both.
    """
    targets = _VALID_ATS if ats is None else {ats}
    conn = sqlite3.connect(get_db_path(profile))
    try:
        for target in targets:
            _ensure_discovered_slugs_table(conn, target)
            conn.execute(f"DELETE FROM discovered_{target}_slugs")
        conn.commit()
    finally:
        conn.close()


# ── Pipeline run tracking ─────────────────────────────────────────────────────

def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def start_run(profile: Optional[str] = None, source: str = "pipeline") -> str:
    """
    Insert a new scrape_run record with status='running'.
    Returns the run_id (UUID4 string).
    """
    run_id = str(uuid.uuid4())
    conn   = sqlite3.connect(get_db_path(profile))
    conn.execute(
        """
        INSERT INTO scrape_runs (run_id, profile, started_at, source, status)
        VALUES (?, ?, ?, ?, 'running')
        """,
        (run_id, profile or "", _now_utc(), source),
    )
    conn.commit()
    conn.close()
    return run_id


def finish_run(
    run_id: str,
    jobs_scraped:  int = 0,
    jobs_filtered: int = 0,
    jobs_saved:    int = 0,
    jobs_scored:   int = 0,
    avg_fit_score: Optional[float] = None,
    errors:        Optional[List[str]] = None,
    status:        str = "complete",
    profile:       Optional[str] = None,
) -> None:
    """Update an existing scrape_run record with final stats."""
    conn = sqlite3.connect(get_db_path(profile))
    conn.execute(
        """
        UPDATE scrape_runs
        SET finished_at   = ?,
            jobs_scraped  = ?,
            jobs_filtered = ?,
            jobs_saved    = ?,
            jobs_scored   = ?,
            avg_fit_score = ?,
            errors        = ?,
            status        = ?
        WHERE run_id = ?
        """,
        (
            _now_utc(),
            jobs_scraped,
            jobs_filtered,
            jobs_saved,
            jobs_scored,
            avg_fit_score,
            json.dumps(errors or []),
            status,
            run_id,
        ),
    )
    conn.commit()
    conn.close()


def get_recent_runs(limit: int = 20, profile: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return the most recent pipeline runs as a list of dicts, newest first."""
    conn = sqlite3.connect(get_db_path(profile))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT * FROM scrape_runs
        ORDER BY started_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()

    result = []
    for row in rows:
        d = dict(row)
        d["errors"] = json.loads(d.get("errors") or "[]")
        result.append(d)
    return result


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    print(f"DB initialized at {get_db_path()}")

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
