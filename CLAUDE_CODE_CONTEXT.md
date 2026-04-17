# Job Agent — Project Context for Claude Code

## What we're building

A personal AI job search agent. Scrapes Greenhouse, Lever, and HN Who's Hiring,
scores every job against a user profile using an LLM, and surfaces the best
matches. Two users (Manav + sister), fully isolated profiles, deployed to GCP.

## Core philosophy

- Get signal fast — scraper + AI filter is the highest-value piece
- Incremental — each step is independently runnable
- Simple infra — SQLite, profile-isolated DBs
- No overengineering — if it works for two people, it's done

---

## Current status

| Step | Status | Description |
|---|---|---|
| 1-5 | ✅ | Skeleton, scrapers, LLM scorer, TheirStack, main.py |
| 6 | ✅ | Profile system + Streamlit onboarding |
| 7 | ✅ | Streamlit dashboard with sidebar, 5-section nav |
| 8 | ✅ | Pydantic models + instructor structured output |
| 9 | ✅ | Loguru + scrape_runs table + per-profile logging |
| 10 | ✅ | llm_utils.py recovery layer (llama-4-scout {"value":x} wrapping fix) |
| 11 | ✅ | Out-of-process dashboard pipeline (subprocess + lockfile + progress JSON) |
| 12 | ✅ | Filter disqualified jobs from UI; "Filtered out" summary in Activity tab |

---

## Actual file structure

```
job-agent/
├── config.yaml              # root config — CLI fallback when no --profile
├── db.py                    # SQLite, profile-aware, all DB logic
├── config.py                # load_config() with PyPDF2 resume extraction
├── scraper.py               # Greenhouse + Lever + HN scrapers
├── scorer.py                # LLM pipeline, RateLimiter, instructor integration
├── candidate_profile.py     # build_structured_profile(), confirm_profile()
├── llm_utils.py             # Shared LLM resilience (see below — read this first)
├── text_utils.py            # extract_job_context() smart truncation
├── theirstack.py            # company discovery, slug resolution GH + Lever
├── models.py                # Pydantic: ScoreResult, StructuredProfile, ScoringWeights
├── progress_tracker.py      # ProgressTracker + dataclasses with to_dict/from_dict
├── main.py                  # CLI orchestrator (unchanged CLI interface)
├── onboarding.py            # Streamlit 5-step profile creation flow
├── dashboard.py             # Streamlit web UI — polling-only, no inline pipeline
├── dashboard_ui.py          # render_progress_header, render_pipeline_stages, etc.
├── logging_config.py        # configure_logging(profile, debug)
├── ui_shell.py              # toolbar(), panel(), callout(), badge(), etc.
├── ui_theme.py              # PAGE_TITLE, apply_page_scaffold()
├── test_scorer_recovery.py  # 10 unit tests for llm_utils recovery layer
├── worker/
│   └── run_pipeline.py      # Out-of-process pipeline runner (the ONLY way to run a pipeline)
├── profiles/
│   └── {slug}/
│       ├── config.yaml
│       ├── {slug}.db
│       ├── .worker_running      # lockfile: PID + started_at JSON, deleted on finish
│       ├── .run_progress.json   # live tracker state, written atomically during run
│       └── .last_run            # final status JSON written on finish
└── logs/
    └── {slug}/agent.log
```

---

## llm_utils.py — the resilience layer (read before touching LLM calls)

**MODULE CONTRACT: Any code that makes structured-output LLM calls MUST go
through `safe_structured_call()`. Do NOT call instructor directly.**

### Why this exists

Some models (e.g. llama-4-scout-17b via Groq) wrap every tool-call parameter
in `{"value": x}` objects instead of returning raw scalars. Groq's API
validates the tool schema before instructor sees the response and rejects it
with a 400 "tool_use_failed". Without llm_utils:

- instructor's tenacity loop retries 2-3 times, sleeping ~10 s each → 30 s dead time per job
- the `failed_generation` payload (which has all the correct data, just wrapped)
  is silently discarded, forcing an extra API call

### Public API

```python
unwrap_value_objects(data: Any) -> Any
    # Recursively unwrap {"value": x} single-key dicts. No-op for other shapes.

parse_llm_response(raw: str) -> dict
    # Strip markdown fences, JSON parse, then unwrap_value_objects.
    # Use this in all raw-LLM fallback paths.

is_schema_error(exc) -> bool
    # True if 400 / tool_use_failed / "parameters for tool" in message.

is_rate_limit_error(exc) -> bool
    # True if 429 / RESOURCE_EXHAUSTED / rate_limit in message.

extract_from_failed_generation(exc, label="recovery") -> Optional[dict]
    # Extract the model's actual output from a Groq 400 response.
    # Walks the full exception chain via .body, ast.literal_eval, __cause__,
    # __context__, and .last_attempt (InstructorRetryException).
    # Returns unwrapped param dict on success, None on failure.

safe_structured_call(client, model, prompt, response_model, *, max_tokens=700,
                     temperature=0, label="llm", max_retries=3) -> Optional[dict]
    # The only approved way to call instructor. Sets max_retries=0 per call
    # to prevent instructor's internal tenacity delays. Short-circuits on schema
    # errors (deterministic — retrying won't help). Runs pydantic validation on
    # recovered data. Returns model_dump() dict or None (caller uses raw LLM).
```

### Exception chain walking strategy

`extract_from_failed_generation` does breadth-first search across:
1. `exc.body` dict (raw `groq.BadRequestError`)
2. `ast.literal_eval` on `"Error code: 400 - {'error':..."` in str(exc)
3. `exc.last_attempt.result()` — raises inner exc (InstructorRetryException path)
4. `exc.__cause__` and `exc.__context__`

All extraction results are logged at INFO level (visible in journalctl).

---

## Out-of-process pipeline architecture

### Why

The Streamlit process blocks during a pipeline run. If the user closes the
browser tab or Streamlit restarts, the run dies. The fix: detach the run into
a subprocess and have the dashboard poll a progress file.

### How it works

**Launching:**
- User clicks "Start discovery" in dashboard → `_queue_pipeline_run(slug)` →
  `_launch_worker(slug)` → `subprocess.Popen([sys.executable, "worker/run_pipeline.py",
  "--profile", slug], start_new_session=True, stdout/stderr=DEVNULL)`
- `start_new_session=True` detaches from the Streamlit process group

**Lockfile:** `profiles/{slug}/.worker_running` — JSON with `{pid, started_at}`.
- Written at worker start, deleted in `finally` block (success or failure)
- Stale threshold: 3 hours (same in worker and dashboard)
- Dashboard reads `_worker_is_running(slug)` before every render

**Progress JSON:** `profiles/{slug}/.run_progress.json`
- Written atomically (tmp + `os.replace`) by the worker after every stage
  transition and every 5 scored jobs
- Format: `ProgressTracker.to_dict()` — stages dict keyed by stage value,
  sources dict, activities list, start_time float, job counters
- Dashboard reads and reconstructs via `ProgressTracker.from_dict(data)`,
  passes result to `_render_pipeline_snapshot(tracker, host)`

**Polling:**
- Dashboard checks `_worker_is_running(slug)` on every render
- If running: read `.run_progress.json`, reconstruct tracker, render progress,
  `time.sleep(2)`, `st.rerun()`
- Button shows "Running..." and is `disabled=True` during run
- Worker finish detection: session state `worker_was_running_{slug}` transitions
  True → False → `invalidate_dashboard_caches()` + success notice + rerun

**Status file:** `profiles/{slug}/.last_run` — JSON with
`{started_at, finished_at, status, jobs_scored, error_message}`. Written at
the very end (or on error) and is separate from the progress JSON.

**Systemd timer:** Uses the same `worker/run_pipeline.py` entry point and CLI
interface. Progress JSON writes are transparent — exit codes unchanged.

### ProgressTracker serialization

All dataclasses and ProgressTracker itself have `to_dict()` / `from_dict()`:

```python
tracker.to_dict()     # -> JSON-serializable dict
ProgressTracker.from_dict(data)  # reconstructs from dict
```

Stage values (enum) are stored by `.value` string (e.g. "🔍 Discovering companies").

---

## Filter-skipped jobs (disqualified)

### What are they

Jobs that the LLM decides to disqualify during scoring — e.g. "intern-only",
"location mismatch", "YOE too high", "title blocklist" — get `disqualified=True`
and `fit_score=0` from `score_job()`. They are stored in the DB (score row
written) but must not pollute the review queue.

### Canonical signal

`scores.disqualified INTEGER DEFAULT 0` (added via `_migrate_db`).
`scores.disqualify_reason TEXT DEFAULT ''` — the human-readable reason.

Backwards compat: `_deserialize_job_record()` also checks `flags` for items
starting with `"disqualified:"` so pre-migration rows are caught too.

### How filtering works

In `_render_profile_dashboard()`:
```python
all_records = _cached_fetch_job_summaries(slug)
records = [r for r in all_records if not r.get("disqualified")]   # visible
disq_records = [r for r in all_records if r.get("disqualified")]  # hidden
```

`records` (non-disqualified) flows into all downstream renderers:
- `_collect_metrics()` — scored/pending/avg_fit computed from visible only
- `_render_jobs_tab()` — review queue table
- `_render_overview_tab()` — top matches, scoreboard
- `_render_top_matches()` — best opportunities panel

`disq_records` summary is injected into `metrics` as:
- `metrics["disqualified_count"]` — total hidden jobs
- `metrics["disqualified_by_reason"]` — {reason: count}
- `metrics["db_total"]` — true DB count including disqualified (used in settings warnings)

### Where disqualified jobs appear

Activity tab → "Filtered out (N jobs hidden by scoring rules)" expander shows
count grouped by reason. This confirms the filter is working without cluttering
the main views.

### Genuine scoring failures still visible

Jobs with `score_error` set and no score row (or failed attempts < 3) still
appear with `score_state = "Needs retry"` or `"Failed"`. The filter only hides
jobs where `disqualified=1` — it does NOT hide error states.

---

## scorer.py key points

- `score_all_jobs(config, yes=False, profile=None, on_job_scored=None) -> list`
  — optional `on_job_scored(i, total, result)` callback called after each
  successful score; used by worker for per-job progress updates
- `save_score()` now accepts `disqualified=int` and `disqualify_reason=str`
- `safe_structured_call` from llm_utils is used for structured output; raw
  `llm_call` + `parse_llm_response` is the fallback
- `config["_active_profile"]` must be set before any LLM call so DB writes
  scope to the correct profile

## candidate_profile.py key points

- Uses `safe_structured_call` for structured output (same recovery as scorer)
- Falls back to `llm_call` + `parse_llm_response` if instructor fails
- Returns a minimal hard-coded profile dict if both LLM paths fail

---

## Key module details

### models.py

Three Pydantic v2 models:

**ScoreResult** — LLM output for job scoring
- Fields: `disqualified`, `disqualify_reason`, `role_fit`, `stack_match`,
  `seniority`, `location`, `growth`, `compensation` (all 0-10),
  `reasons` (list, max 4), `flags` (list), `one_liner`
- Validators: clamps scores to 0-10, zeros all dimensions when disqualified,
  limits reasons to 4 items, filters empty strings

**StructuredProfile** — LLM output for profile extraction
- Fields: `name`, `yoe`, `current_title`, `core_skills`, `languages`,
  `frameworks`, `cloud`, `past_roles`, `education`, `strengths`,
  `target_roles`, `target_salary`, `remote_preference`, `preferred_locations`
- `target_salary` is compensation-shape-neutral:
  annual salary for full-time profiles, monthly stipend for paid-only internship
  profiles, and `null` when an intern is open to unpaid roles
- Validators: deduplicates all list fields, parses salary from string ($130,000),
  clamps yoe to >= 0

**ScoringWeights** — validation only, loaded from config
- Fields: all 6 dimension weights as floats
- Validator: weights must sum to 1.0 ± 0.01

### db.py

Profile resolution order: explicit `profile=` arg → `_ACTIVE_PROFILE` module var → root jobs.db

Key functions (all accept `profile=None`):
- `set_active_profile(profile)` / `get_db_path(profile=None)`
- `init_db(profile=None)` — creates profile folder if needed, runs migrations
- `insert_job(job, profile=None) -> bool`
- `save_score(job_id, ..., disqualified=0, disqualify_reason="", profile=None)`
- `get_unscored(profile=None) -> list[Job]` — jobs with score IS NULL and score_attempts < 3
- `get_top_jobs(min_score, profile=None) -> list`
- `count_jobs(profile=None) -> dict` — {total, scored} (includes disqualified in total)
- `rescore_reset(profile=None)` — clears scores, resets score_attempts to 0
- `update_job_status(job_id, status, profile=None)`
- `save_discovered_slug(slug, company_name, ats, profile=None)`
- `load_discovered_slugs(ats, profile=None) -> list[str]`
- `increment_score_attempts(job_id, profile=None)`
- `write_score_error(job_id, error, profile=None)`

### Scores table (with migrations)

```sql
CREATE TABLE scores (
    job_id            TEXT PRIMARY KEY,
    fit_score         INTEGER,
    role_fit          INTEGER,
    stack_match       INTEGER,
    seniority         INTEGER,
    location          INTEGER,
    growth            INTEGER,
    compensation      INTEGER,
    reasons           TEXT,       -- JSON array, max 4 items
    flags             TEXT,       -- JSON array
    skill_misses      TEXT,       -- JSON array, top 5 missing skills
    one_liner         TEXT,
    ats_score         INTEGER,
    disqualified      INTEGER DEFAULT 0,   -- 1 = hidden from review queue
    disqualify_reason TEXT DEFAULT '',     -- human-readable reason
    scored_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

### config.yaml structure

```yaml
profile:
  name: "..."
  job_type: "fulltime" | "internship"
  resume_file: "path/to/resume.pdf"   # OR use resume: inline text
  bio: |
    ...

llm:
  provider: "groq"
  model:
    groq: "meta-llama/llama-4-scout-17b-16e-instruct"
    gemini: "gemini-2.5-flash"
    anthropic: "claude-sonnet-4-20250514"
    openai: "gpt-4o-mini"
  temperature: 0
  rate_limits:
    groq:      {max_rpm: 28, max_tpm: 28000, max_rpd: 950}
    gemini:    {max_rpm: 9,  max_tpm: 240000, max_rpd: 250}
    anthropic: {max_rpm: 50, max_tpm: 9000000}
    openai:    {max_rpm: 50, max_tpm: 9000000}

preferences:
  titles: [...]
  desired_skills: [...]
  hard_no_keywords: [...]
  location:
    remote_ok: true
    preferred_locations: [...]
  compensation:
    min_salary: 130000
  yoe: 2
  filters:
    min_yoe: 0
    max_yoe: 5
    max_job_age_days: 30
    require_degree_filter: true
    title_blocklist: [Senior, Staff, Principal, VP, Director, Head of, Manager, Lead]
    countries_allowed: [United States, US, USA, Remote]

sources:
  greenhouse:
    enabled: true
    companies: [anthropic, stripe, figma, ...]
  lever:
    enabled: true
    companies: [stripe, linear, vercel, ...]
  hn:
    enabled: true   # key is "hn" not "hackernews" in sources

scoring:
  min_display_score: 60
  weights:
    role_fit:     0.30
    stack_match:  0.25
    seniority:    0.20
    location:     0.10
    growth:       0.10
    compensation: 0.05
```

Internship profiles use this compensation shape instead:

```yaml
profile:
  job_type: internship
  target_season: "Summer 2027"
  school: "..."
  major: "..."
  graduation_year: "2028"

preferences:
  compensation:
    intern_pay_preference: paid_only | unpaid_ok | no_preference
    monthly_stipend: 4500   # optional, only when paid_only and the user entered a target
  filters:
    max_yoe: 1
    max_job_age_days: 14
    title_blocklist: [Senior, Staff, Principal, VP, Director, Head of, Manager, Lead, Executive, Full-time only]
```

Scoring behavior:
- Full-time compensation scores against `compensation.min_salary`
- Internship compensation scores against whether pay/stipend is mentioned
- `intern_pay_preference=paid_only` penalizes unpaid internship postings
- `unpaid_ok` and `no_preference` treat pay as a soft positive only
- Internship prompts should not treat "pursuing a degree" as a master's/PhD disqualifier

---

## Environment variables

Per-profile keys (written by onboarding):
```
GROQ_API_KEY_MANAV=
GROQ_API_KEY_SISTER=
ANTHROPIC_API_KEY_MANAV=
```

Unsuffixed fallback for CLI / single-user:
```
GROQ_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
OPENAI_API_KEY=
THEIRSTACK_API_KEY=
```

Resolution order: `{KEY}_{PROFILE_UPPER}` → `{KEY}` (fallback).

---

## Known issues / things to watch

- `config.py` still uses PyPDF2 — should migrate to pdfplumber for better extraction
- HN config key is `"hn"` not `"hackernews"` — dashboard/worker check `sources.get("hn")`
- `count_jobs(profile)` returns DB total including disqualified — use `metrics["db_total"]`
  in settings warnings, `metrics["total"]` elsewhere (visible only)
- `_collect_metrics()` expects pre-filtered records (no disqualified) — the split
  happens in `_render_profile_dashboard()` before passing records downstream
- Old disqualified rows (pre-migration) are caught by the flags fallback in
  `_deserialize_job_record()` — no data migration needed
