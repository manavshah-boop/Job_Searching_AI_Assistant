# Job Agent — Project Context for Claude Code

## What we're building

A personal AI job search agent. Scrapes Greenhouse, Lever, Ashby, Workable,
Himalayas, and HN Who's Hiring, scores every job against a user profile using
an LLM, and surfaces the best matches. Two users (Manav + sister), fully
isolated profiles, deployed to GCP.

## Core philosophy

- Get signal fast — scraper + AI filter is the highest-value piece
- Incremental — each step is independently runnable
- Simple infra — SQLite, profile-isolated DBs
- No overengineering — if it works for two people, it's done

| 21 | ✅ | Cross-encoder reranking: `reranker.py`, profile-aware semantic matching, `--semantic-match`, `--semantic-search`, dashboard rerank toggle |
| 22 | ✅ | Role-agnostic profile intent + evidence matching: `profile_intent.py`, `ProfileIntent`, canonical section taxonomy, role-family section weights, `MatchEvidence` with positive/negative signals |

---

## Current status

| Step | Status | Description |
|---|---|---|
| 1-5  | ✅ | Skeleton, scrapers, LLM scorer, main.py |
| 6 | ✅ | Profile system + Streamlit onboarding |
| 7 | ✅ | Streamlit dashboard with sidebar, 5-section nav |
| 8 | ✅ | Pydantic models + instructor structured output |
| 9 | ✅ | Loguru + scrape_runs table + per-profile logging |
| 10 | ✅ | llm_utils.py recovery layer (llama-4-scout {"value":x} wrapping fix) |
| 11 | ✅ | Out-of-process dashboard pipeline (subprocess + lockfile + progress JSON) |
| 12 | ✅ | Filter disqualified jobs from UI; "Filtered out" summary in Activity tab |
| 13 | ✅ | Scrape-filter rejection persistence: scrape_qualified/scrape_filter_reason columns, auditable dashboard toggle |
| 17+20 | ✅ | Semantic job embeddings: `embedder.py`, `job_embeddings` table, embedding pipeline stage, `--embed-only` |
| 18 | ✅ | ChromaDB vector retrieval: `vector_store.py`, persistent profile-scoped ANN index, `--vector-search`, `--rebuild-vector-index`, `--clear-vector-index` |

---

## Step 22 — Role-agnostic profile intent and evidence-based matching

### profile_intent.py

New module that converts a config dict into a `ProfileIntent` dataclass. This removes the assumption that users are always searching for software engineering roles.

**ProfileIntent fields:** `profile_name`, `role_family`, `target_roles`, `seniority`, `years_experience`, `industries`, `domain_skills`, `tools`, `credentials`, `education`, `locations`, `remote_ok`, `compensation`, `dealbreakers`, `nice_to_haves`, `job_type`, `raw_keywords`

**Supported role families:**
`software_engineering`, `data_analytics`, `finance`, `product_management`, `operations`, `sales`, `marketing`, `design`, `internship`, `general`

Unknown role families fall back to `general`. Add `role_family: finance` to `profile.role_family` in config to override inference.

**Role family inference:** signal matching over titles + skills + bio/resume text. Explicit config key `profile.role_family` always wins.

**Public API:**
- `normalize_profile_intent(config) -> ProfileIntent` — main entry point
- `get_role_family_section_weights(role_family) -> dict[str, float]` — chunk-key weights per family
- `map_header_to_canonical(text) -> str | None` — maps a JD header to canonical section name

### Canonical section taxonomy

`CANONICAL_SECTION_ALIASES` maps canonical labels to lists of header phrases found in real job postings.

Canonical sections: `summary`, `company`, `responsibilities`, `requirements`, `preferred_qualifications`, `tools_and_skills`, `logistics`, `compensation`, `benefits`

`CANONICAL_TO_CHUNK_KEY` maps canonical labels to the five chunk_key values used in SQLite/Chroma:
- `preferred_qualifications` → `requirements`
- `tools_and_skills` → `requirements`
- `logistics` → `requirements`
- `company` → `summary`
- `unknown` → `summary`

Existing `chunk_key` values and DB schema are unchanged. The canonical taxonomy is used in `embedder._classify_header` for better header detection.

### Role-family section weights

`profile_intent.get_role_family_section_weights(role_family)` returns a `dict[chunk_key -> float]`. These replace the static `SECTION_WEIGHTS` dict in reranker.py per-job when the candidate's role family is known.

Examples:
- `software_engineering`: requirements=1.20, responsibilities=1.15, summary=1.00, compensation=0.90, benefits=0.80
- `finance`: requirements=1.20, responsibilities=1.15, compensation=0.95 (slightly higher than software)
- `internship`: responsibilities=1.15, compensation=1.10 (compensation matters more)
- `sales`: responsibilities=1.20, compensation=1.00 (both high signal for sales roles)

### MatchEvidence

New dataclass in `reranker.py`:

```python
@dataclass
class MatchEvidence:
    positive: list[str]       # title match, skills present, remote match, etc.
    concerns: list[str]       # seniority mismatch, clearance, missing credential, etc.
    matched_keywords: list[str]
    missing_or_unclear: list[str]
```

`RerankedJobResult` now includes an `evidence: MatchEvidence` field. Evidence is deterministic — no LLM.

**Positive signals:** title keyword overlap, skills/tools found in chunk text, location/remote match, industry match.

**Concern signals:** senior/staff/lead in job title for entry/mid profiles, clearance requirements, no-sponsorship language, credential requirements (CPA, CFA, PMP, MBA, clearance, degree) not listed in the profile, unpaid internship when `paid_only` preference is set.

### build_profile_match_query

Now delegates to `normalize_profile_intent(config)` so the generated query is always role-agnostic.

Finance example: `"Candidate seeking full-time Financial Analyst, FP&A Analyst roles. Experience: 2 years. Skills: financial modeling, budgeting, Excel, SQL. Preferences: remote or New York, NY, compensation $80,000+. Avoid: Senior/Director/VP roles."`

Marketing example: `"Candidate seeking full-time Growth Marketing Manager, Lifecycle Marketing roles. Skills: SEO, email campaigns, Google Analytics, HubSpot. Preferences: remote-friendly. Avoid: Director/VP/Head of roles."`

### Dashboard (semantic match panel)

Now shows:
- Final score, matched sections, match reason (unchanged)
- **Positive signals** line (e.g. "Title match: Backend Engineer · Skills present: Python, SQL, AWS · Remote-friendly posting")
- **Concerns** line when present (e.g. "Seniority concern: posting appears senior-level · Clearance requirement detected")
- Evidence snippets as italic captions

### CLI (--semantic-match and --semantic-search --rerank)

Now prints `Positive: ...` and `Concerns: ...` lines below the match reason for each result.

---

## Actual file structure

```
job-agent/
├── config.yaml              # root config — CLI fallback when no --profile
├── db.py                    # SQLite, profile-aware, all DB logic
├── config.py                # load_config() with PyPDF2 resume extraction
├── scraper.py               # Greenhouse + Lever + HN + Ashby + Workable + Himalayas scrapers
├── scorer.py                # LLM pipeline, RateLimiter, instructor integration
├── embedder.py              # Semantic chunking + sentence-transformers batch embeddings
├── vector_store.py          # ChromaDB wrapper: profile-scoped indexing + semantic retrieval
├── reranker.py              # Cross-encoder reranking over vector-retrieved job chunks
├── profile_intent.py        # ProfileIntent normalization, canonical section taxonomy, role-family weights
├── candidate_profile.py     # build_structured_profile(), confirm_profile()
├── llm_utils.py             # Shared LLM resilience (see below — read this first)
├── text_utils.py            # extract_job_context() smart truncation
├── theirstack.py            # slug resolution utilities for GH, Lever, Ashby, Workable; get_or_discover_slugs()
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
├── test_embedder.py         # Chunking, embedding batch shape, DB round-trip, canonical alias tests
├── test_vector_store.py     # Chroma indexing, idempotent upserts, rebuild/query tests
├── test_reranker.py         # Profile-aware query building, role-family weights, evidence tests
├── test_profile_intent.py   # Role family inference, ProfileIntent normalization, canonical taxonomy
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
Current ordered stages: `DISCOVERING -> FETCHING -> SCRAPING -> SCORING -> EMBEDDING -> FINALIZING`.
Vector indexing is reported as metrics inside the `EMBEDDING` stage rather than as a separate stage.

---

## Filter-skipped jobs (disqualified and scrape-rejected)

There are two distinct hidden-job systems. Do not confuse them.

### 1. LLM-disqualified jobs

Jobs that the LLM decides to disqualify during scoring — e.g. "intern-only",
"location mismatch", "YOE too high", "title blocklist" — get `disqualified=True`
and `fit_score=0` from `score_job()`. They are stored in the DB (score row
written) but must not pollute the review queue.

**Canonical signal:** `scores.disqualified INTEGER DEFAULT 0`, `scores.disqualify_reason TEXT DEFAULT ''`.
Backwards compat: `_deserialize_job_record()` also checks `flags` for items
starting with `"disqualified:"` so pre-migration rows are caught too.

### 2. Scrape-rejected jobs

Jobs that failed `passes_filters()` during scraping — title blocklist, YOE limits,
non-US location, no hiring intent (HN). These are inserted into the DB with
`scrape_qualified=0` and a human-readable `scrape_filter_reason` (e.g.
`"title_blocklist: Staff"`, `"yoe_max: 8 > 5"`, `"non_us_location"`).

**New DB columns (added via `_migrate_db`):**
```
jobs.scrape_qualified     INTEGER DEFAULT 1  -- 0 = rejected at scrape time
jobs.scrape_filter_reason TEXT DEFAULT ''    -- human-readable filter reason
```

`passes_filters()` now returns `tuple[bool, str]` — `(True, "")` on pass,
`(False, reason)` on reject. Scraper call sites unpack
the tuple and insert rejected jobs with `scrape_qualified=0`.

`get_unscored()` has `AND scrape_qualified = 1` in its WHERE clause so
scrape-rejected jobs are never fetched for scoring. `score_job()` also checks
`job.scrape_qualified == 0` as a safety net and returns early with
`flags=["scrape_filtered: {reason}"]` if a job somehow slips through.

### How filtering works in the dashboard

In `_render_profile_dashboard()`, three mutually exclusive partitions:
```python
records = [r for r in all_records if not r.get("disqualified") and r.get("scrape_qualified", 1)]
disq_records = [r for r in all_records if r.get("disqualified")]
scrape_rejected_records = [r for r in all_records if not r.get("scrape_qualified", 1)]
```

`records` (clean jobs) flows into all downstream renderers.

`disq_records` summary injected into `metrics`:
- `metrics["disqualified_count"]` — hidden LLM-disqualified count
- `metrics["disqualified_by_reason"]` — {reason: count}

`scrape_rejected_records` summary injected into `metrics`:
- `metrics["scrape_rejected_count"]` — hidden scrape-rejected count
- `metrics["scrape_rejected_by_reason"]` — {reason_prefix: count} (grouped by part before `:`)

`metrics["db_total"]` = `metrics["total"]` + disq + scrape_rejected (used in settings warnings).

### Where each appears

- Activity tab → "Filtered out (N jobs hidden by scoring rules)" — LLM-disqualified breakdown
- Activity tab → "Scrape-filtered (N jobs rejected before scoring)" — scrape-rejected breakdown
- Jobs tab → "Show scrape-filtered jobs" checkbox (inside Review controls expander, hidden by default):
  when checked, `scrape_rejected_records` are merged into the visible pool before filters apply.
  Jobs table shows a "Filter reason" optional column (hidden by default, toggleable via Show/hide columns).

### Genuine scoring failures still visible

Jobs with `score_error` set and no score row (or failed attempts < 3) still
appear with `score_state = "Needs retry"` or `"Failed"`. Neither filter hides
error states — only `disqualified=1` and `scrape_qualified=0` hide jobs.

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

## embedder.py key points

- Public entrypoint: `embed_jobs(config, profile=None, force=False, on_job_embedded=None) -> dict`
- Uses heuristic semantic chunking, not an LLM call
- `semantic_chunk_job(job)` emits a small named set of chunks in this order:
  `summary`, `responsibilities`, `requirements`, `compensation`, `benefits`
- Sentence-transformers model loads lazily once per process and is cached in-module
- Worker pipeline runs embeddings after scoring; CLI supports standalone refresh via `python main.py --embed-only`
- When `embeddings.vector_store.enabled` is true, the embedder indexes freshly stored SQLite embeddings into ChromaDB without recomputing vectors

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
- `get_unscored(profile=None) -> list[Job]` — jobs with score IS NULL, score_attempts < 3, AND scrape_qualified = 1
- `get_scored_jobs_for_embedding(model_name, profile=None, limit=None, force=False) -> list[Job]` — scrape-qualified jobs with score rows, optionally only the unembedded subset for one model
- `replace_job_embeddings(job_id, model_name, rows, profile=None)` — atomic delete+bulk-insert for one job/model pair
- `get_job_embeddings(job_id, profile=None, model_name=None) -> list[dict]`
- `get_embedding_index_rows(model_name, profile=None, job_ids=None) -> list[dict]` — SQLite-backed embedding rows joined with job and score metadata for Chroma indexing/rebuilds
- `get_top_jobs(min_score, profile=None) -> list`
- `count_jobs(profile=None) -> dict` — `{total, scored, embedded}` (includes disqualified in total)
- `rescore_reset(profile=None)` — clears scores, resets score_attempts to 0
- `update_job_status(job_id, status, profile=None)`
- `save_discovered_slug(slug, company_name, ats, profile=None)` — valid ats: greenhouse, lever, ashby, workable
- `load_discovered_slugs(ats, profile=None) -> list[str]`
- `increment_score_attempts(job_id, profile=None)`
- `write_score_error(job_id, error, profile=None)`

### Jobs table (with migrations)

```sql
CREATE TABLE jobs (
    id                   TEXT PRIMARY KEY,
    title                TEXT NOT NULL,
    company              TEXT NOT NULL,
    location             TEXT,
    url                  TEXT,
    raw_text             TEXT,
    source               TEXT,
    score_attempts       INTEGER DEFAULT 0,
    score_error          TEXT,
    status               TEXT DEFAULT 'new',
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- added via _migrate_db:
    scrape_qualified     INTEGER DEFAULT 1,  -- 0 = rejected by passes_filters() at scrape time
    scrape_filter_reason TEXT DEFAULT ''     -- e.g. "title_blocklist: Staff", "yoe_max: 8 > 5"
)
```

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

### Job embeddings table (with migrations)

```sql
CREATE TABLE job_embeddings (
    job_id      TEXT NOT NULL REFERENCES jobs(id),
    model_name  TEXT NOT NULL,
    chunk_key   TEXT NOT NULL,   -- summary | responsibilities | requirements | compensation | benefits
    chunk_order INTEGER DEFAULT 0,
    chunk_text  TEXT NOT NULL,   -- section text with title/company/location context
    embedding   TEXT NOT NULL,   -- compact JSON array of floats
    dimensions  INTEGER NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (job_id, model_name, chunk_key)
)
```

Supporting indexes:
- `idx_job_embeddings_model_job(model_name, job_id)` for "what still needs embeddings?" scans
- `idx_job_embeddings_job(job_id)` for job detail / retrieval lookups

### vector_store.py

Purpose:
- Wraps all ChromaDB usage behind one project-specific module
- Keeps SQLite as the source of truth while treating ChromaDB as a rebuildable ANN index
- Owns stable IDs, metadata validation, index rebuilds, and semantic retrieval aggregation

Public API:
- `get_chroma_client(profile)`
- `get_job_collection(profile)`
- `vector_store_enabled(config) -> bool`
- `upsert_job_embeddings(profile, model_name, rows) -> dict`
- `index_embedded_jobs(profile, model_name, force=False) -> dict`
- `query_similar_chunks(profile, query, top_k_chunks=50, filters=None) -> list[VectorChunkResult]`
- `query_similar_jobs(profile, query, top_k_chunks=50, top_k_jobs=30, filters=None) -> list[VectorJobResult]`
- `rebuild_vector_index(profile, model_name=None) -> dict`
- `clear_vector_index(profile) -> None`

Storage path:
- Default persist directory: `profiles/{profile}/chroma/`
- One collection per profile: `job_chunks`

Stable vector IDs:
- Format: `{job_id}:{chunk_key}:{embedding_model_slug}`
- Deterministic and idempotent across reruns

Collection metadata:
- `embedding_model`
- `embedding_dimensions`
- `distance_metric`
- `created_by = job_search_ai_assistant`

Stored Chroma metadata fields:
- `profile`
- `job_id`
- `chunk_key`
- `chunk_order`
- `title`
- `company`
- `source`
- `url`
- `model_name`
- `dimensions`
- `scored_at`
- `fit_score`
- `ats_score`
- `status`
- `created_at`

Document field:
- The semantic chunk text only. Raw `jobs.raw_text` is not duplicated into Chroma metadata.

Indexing flow:
- `score_all_jobs()`
- `embed_jobs()`
- semantic chunks + normalized sentence-transformer vectors stored in SQLite `job_embeddings`
- `vector_store.index_embedded_jobs()` reads those stored vectors back from SQLite and upserts them into ChromaDB

Query flow:
- `query_similar_chunks()` embeds the query with the same sentence-transformers model
- Chroma returns nearest chunk candidates
- `query_similar_jobs()` aggregates chunk hits into ranked job candidates using transparent section weights
- Ranking formula: `best * 0.75 + top3_avg * 0.20 + coverage_bonus * 0.05`
- Default section weights: `requirements 1.10`, `responsibilities 1.05`, `summary 1.00`, `compensation 0.90`, `benefits 0.80`
- Compensation/benefits weights are relaxed upward only when the query explicitly asks about pay/perks

Rebuild behavior:
- `clear_vector_index(profile)` deletes the Chroma collection/files for that profile
- `rebuild_vector_index(profile)` recreates the index from SQLite `job_embeddings`
- If the collection metadata does not match the active embedding model/dimensions, the collection is recreated automatically

Why Chroma is rebuildable and not source of truth:
- SQLite keeps durable job rows, score rows, run history, and chunk embeddings
- Chroma only accelerates retrieval over those persisted chunk vectors
- If Chroma is missing or corrupted, it can be rebuilt from SQLite without touching scraping or scoring state

### reranker.py

Purpose:
- Adds a reusable profile-aware precision layer after vector retrieval
- Compares a compact candidate/profile match query against the most important job chunks
- Keeps vector retrieval and reranking concerns separate so later steps can reuse reranked candidates for selective LLM scoring

Public API:
- `reranking_enabled(config) -> bool`
- `reranking_model_name(config) -> str`
- `get_cross_encoder(config)`
- `build_profile_match_query(config, user_query=None) -> str`
- `select_chunks_for_reranking(job_result, chunk_results, max_chunks_per_job, *, match_query=None) -> list[VectorChunkResult]`
- `rerank_chunks(match_query, chunk_results, config) -> list[RerankedChunkResult]`
- `rerank_jobs(match_query, vector_job_results, chunk_results, config) -> list[RerankedJobResult]`
- `semantic_match_jobs(profile, config, user_query=None) -> list[RerankedJobResult]`
- `select_jobs_for_llm_scoring(reranked_results, threshold, top_k) -> list[str]`

Profile-aware match query:
- Built deterministically from config, not from an LLM call
- Includes target titles, YOE, strongest desired/resume-backed skills, location preferences, remote preference, compensation preference, job type, and optional user query
- Stays concise so it fits the cross-encoder context budget cleanly

Section-aware chunk selection:
- Selects only a small set of high-signal chunks per job before reranking
- Prioritizes `requirements`, `responsibilities`, then `summary`
- Includes `compensation` when the query/profile signals salary or stipend intent
- Includes `benefits` only for explicit perks/benefits intent or when benefits is already a strong retrieved hit

Cross-encoder scoring:
- Uses `sentence-transformers` `CrossEncoder`
- Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Loaded lazily once per process and cached by model name
- Scored in batches on CPU: `(match_query, truncated_chunk_text)`
- Raw scores are preserved; display/aggregation uses `sigmoid(raw_score)` normalization

Job-level aggregation formula:
- Section weights:
  - `requirements: 1.15`
  - `responsibilities: 1.10`
  - `summary: 1.00`
  - `compensation: 0.90` normally, `1.05` for compensation intent
  - `benefits: 0.80` normally, `1.00` for benefits/compensation intent
- Job rerank score:
  - `best_weighted_chunk * 0.55`
  - `top3_weighted_average * 0.30`
  - `required_section_coverage_bonus * 0.10`
- Final score:
  - `rerank_score + vector_score * 0.05`
- Coverage bonus rewards strong `requirements + responsibilities` support, with a smaller summary bonus

Query flow:
- `semantic_match_jobs()` builds the profile-aware query
- Runs vector retrieval for broad recall through `vector_store.py`
- Selects high-signal chunks per job
- Reranks those chunks with the cross-encoder
- Aggregates reranked evidence back to unique job-level results with match reasons and evidence snippets

Fallback behavior:
- If reranking is disabled, `semantic_match_jobs()` returns vector-ranked fallback results in the same job-result shape
- Existing vector-only search remains available via `vector_store.query_similar_jobs()`

CLI commands:
- `python main.py --profile manav --semantic-match`
- `python main.py --profile manav --semantic-search "backend AI platform role with Python and AWS" --rerank`
- `python main.py --profile manav --semantic-search "backend AI platform role with Python and AWS"`
- `python main.py --profile manav --vector-search "backend AI platform role"`

Dashboard usage:
- Jobs tab semantic panel now supports:
  - optional free-text query
  - "Use cross-encoder reranking" toggle
  - profile-aware matching when the query is left blank
- Falls back cleanly to vector-only results when reranking is disabled

Future LLM-reduction path:
- Intended next architecture:
  - vector recall
  - cross-encoder precision
  - selective LLM scoring only for top reranked job IDs
- `select_jobs_for_llm_scoring()` is the bridge for that next step

### Retrieval Architecture

- SQLite: durable job, score, run, and embedding records
- ChromaDB: ANN retrieval index over chunk embeddings
- Cross encoder: profile-aware reranking over top vector candidates
- LLM scorer: expensive final evaluator, not used for every retrieved job

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
  ashby:
    enabled: true
    companies: [linear, vercel, retool, ...]   # YC/growth startups; same slug shape as GH/Lever
  workable:
    enabled: false
    companies: []   # public widget API slugs from apply.workable.com/{slug}
  himalayas:
    enabled: false   # remote-only public feed; no company list needed
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

embeddings:
  enabled: true
  model: sentence-transformers/all-MiniLM-L6-v2
  batch_size: 8
  vector_store:
    enabled: true
    provider: chromadb
    persist_directory: null
    collection_name: job_chunks
    top_k_chunks: 50
    top_k_jobs: 30
    strict: false

reranking:
  enabled: true
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  batch_size: 16
  top_k_vector_chunks: 80
  top_k_vector_jobs: 40
  top_k_final: 15
  max_chunks_per_job: 4
  max_chunk_chars: 900
  include_profile_context: true
  include_query_context: true
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
