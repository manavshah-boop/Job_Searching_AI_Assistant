# Job Agent — Project Context for Claude Code

## What we're building

A personal AI job search agent. Scrapes job boards, scores every job against
a user profile using an LLM, and surfaces only the relevant ones. Built for
personal use — two people (Manav and his sister) each with their own profile,
config, and jobs database. No SaaS, no multi-tenancy beyond two profiles,
no overengineering.

## Core philosophy

- **Get signal fast.** The scraper + AI filter is the highest-value piece.
- **Incremental.** Each step is independently runnable before moving to the next.
- **Simple infra.** Runs locally or on a single Railway deployment. SQLite for
  now, Postgres before multi-user deploy.
- **No overengineering.** If it works for two people, it's done.
- **Learning.** Technologies are chosen partly because they're worth knowing —
  embeddings, LangGraph, MLflow are on the roadmap specifically because they're
  relevant to agentic AI engineering roles.

---

## Full build plan

### PHASE 1 — Prototype (current focus)

| Step | Status | Description |
|---|---|---|
| 1 | ✅ | Project skeleton, config.yaml, db.py |
| 2 | ✅ | Greenhouse scraper (public API, no auth) |
| 3 | ✅ | HN Who's Hiring scraper (Algolia API) |
| 4 | ✅ | LLM scorer — multi-dimension, ATS score, multi-provider |
| 4.5 | ✅ | TheirStack company discovery + Lever scraper |
| 5 | ✅ | main.py orchestrator with CLI flags |
| 6 | ⬜ | Profile system + Streamlit onboarding |
| 7 | ⬜ | Streamlit dashboard (profile-aware) |

### PHASE 2 — Harden + Deploy

| Step | Description |
|---|---|
| 8 | Pydantic + instructor — structured output reliability, kills JSON parse errors |
| 9 | Loguru + pipeline run logging — observability foundation, scrape_runs table |
| 10 | SQLite → PostgreSQL — multi-user write safety, required before deploy |
| 11 | Deploy to Railway — web dashboard accessible from both locations |

### PHASE 3 — Make the Agent Smarter

| Step | Description |
|---|---|
| 12 | Embeddings + ChromaDB — semantic job matching, replaces keyword pre-filter |
| 13 | MLflow experiment tracking — compare prompts and models objectively |
| 14 | Feedback loop — thumbs up/down on scores builds a labeled dataset |
| 15 | APScheduler — automatic scrape + score runs, no manual trigger needed |

### PHASE 4 — Agentic + Advanced

| Step | Description |
|---|---|
| 16 | LangGraph — stateful decision graph, agent decides when to broaden search, escalate to human, re-trigger discovery |
| 17 | pgvector + FAISS — production-grade vector search replacing ChromaDB |
| 18 | Fine-tuned classifier — train on feedback data to replace or augment Claude pre-scoring |
| 19 | Celery + Redis — background pipeline, dashboard stays responsive during long runs |

### Always on the backlog
- Response rate tracking — did applications at certain sources get replies?
- Outcome-weighted scoring — jobs that got interviews retroactively score higher
- PyTorch deep dive — understand embedding models and LLMs from the ground up

---

## What exists right now

### File structure

```
job-agent/
├── config.yaml              # root config (CLI fallback when no --profile)
├── db.py                    # all SQLite logic, load_config, get_db_path
├── config.py                # PDF resume extraction via pdfplumber
├── scraper.py               # Greenhouse + Lever + HN scrapers, passes_filters()
├── scorer.py                # LLM scoring pipeline, RateLimiter, print_results()
├── candidate_profile.py     # build_structured_profile(), confirm_profile()
├── text_utils.py            # extract_job_context(), smart truncation
├── theirstack.py            # company discovery, slug resolution for GH + Lever
├── main.py                  # CLI orchestrator, glues all modules together
├── app.py                   # legacy Streamlit config editor (may be superseded)
├── requirements.txt
├── .env                     # API keys (never committed)
├── .env.example
└── jobs.db                  # root DB (CLI use, no --profile)
```

### Profile structure (Step 6 target)

```
profiles/
  manav/
    config.yaml
    jobs.db
    resume.pdf
  sister/
    config.yaml
    jobs.db
    resume.pdf
```

### config.yaml structure

```yaml
profile:
  name: "..."
  job_type: "fulltime" | "internship"
  resume_file: "..."   # path to PDF, OR
  resume: |            # inline text
    ...
  bio: |
    ...
  # internship-only fields:
  target_season: "Summer 2026"
  school: "..."
  major: "..."
  graduation_year: 2027

llm:
  provider: "groq"     # groq | anthropic | gemini | openai
  model:
    groq: "meta-llama/llama-4-scout-17b-16e-instruct"
    anthropic: "claude-sonnet-4-20250514"
    gemini: "gemini-2.5-flash"
    openai: "gpt-4o-mini"
  temperature: 0
  rate_limits:
    groq:
      max_rpm: 28
      max_tpm: 28000
      max_rpd: 950
    gemini:
      max_rpm: 9
      max_tpm: 240000
      max_rpd: 250
    anthropic:
      max_rpm: 50
      max_tpm: 9000000
    openai:
      max_rpm: 50
      max_tpm: 9000000

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
    max_job_age_days: 30   # 14 for internship profiles
    require_degree_filter: true
    title_blocklist: [...]
    countries_allowed: [...]

sources:
  greenhouse:
    enabled: true
    companies: [...]   # priority list
  lever:
    enabled: true
    companies: [...]   # priority list
  hackernews:
    enabled: true

scoring:
  min_display_score: 60
  weights:
    role_fit: 0.30
    stack_match: 0.25
    seniority: 0.20
    location: 0.10
    growth: 0.10
    compensation: 0.05
```

### db.py — key functions

- `get_db_path(profile=None) -> Path` — returns profile-scoped or root DB path
- `init_db(profile=None)` — creates tables, runs migrations, creates profile folder if needed
- `load_config(profile=None)` — loads profile-scoped or root config.yaml
- `insert_job(job, profile=None) -> bool` — True if new, False if duplicate
- `save_score(job_id, ..., profile=None)` — writes scoring results to scores table
- `get_unscored(profile=None) -> list[Job]` — jobs with no score and < 3 attempts
- `get_top_jobs(min_score, profile=None) -> list` — scored jobs above threshold
- `count_jobs(profile=None) -> dict` — total and scored counts
- `rescore_reset(profile=None)` — clears scores table, resets attempts
- `save_discovered_slug(slug, company_name, ats, profile=None)` — caches resolved slug
- `load_discovered_slugs(ats, profile=None) -> list[str]` — cached slugs for GH or Lever

### Job dataclass

```python
@dataclass
class Job:
    id: str           # "{source}_{source_id}" e.g. "greenhouse_127817"
    title: str
    company: str
    location: str
    url: str
    raw_text: str     # cleaned, capped at 2000 chars, sent to LLM
    source: str       # "greenhouse" | "lever" | "hackernews"
    score_attempts: int = 0
    score_error: Optional[str] = None
    status: str = "new"  # new | applied | skipped
```

### Scores table (separate from jobs)

```sql
CREATE TABLE scores (
    job_id       TEXT PRIMARY KEY,
    fit_score    INTEGER,
    role_fit     INTEGER,
    stack_match  INTEGER,
    seniority    INTEGER,
    location     INTEGER,
    growth       INTEGER,
    compensation INTEGER,
    reasons      TEXT,   -- JSON array
    flags        TEXT,   -- JSON array
    skill_misses TEXT,   -- JSON array
    one_liner    TEXT,
    ats_score    INTEGER,
    scored_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

### scraper.py — key functions

- `passes_filters(text, title, location, config, source, debug) -> bool`
  Pre-filter applied before insert_job(). Checks:
  - Title blocklist (word-boundary matched)
  - YOE regex filter (max_yoe from config)
  - US location filter (Greenhouse/Lever use structured field, HN uses text scan)
  - HN-only: hiring intent proximity check
  - Degree requirement filter (regex patterns for "master's required" etc.)
- `scrape_greenhouse(config, slugs) -> dict`
- `scrape_lever(config, slugs) -> dict`
- `scrape_hn(config) -> dict`
- `strip_html(html_text) -> str` — handles double-encoded HTML entities
- `_extract_yoe_numbers(text) -> list[int]` — extracts YOE from job text
- `contains_hard_no_keyword(text, keywords) -> bool`

### scorer.py — key functions

- `get_llm_client(config) -> callable` — returns `(prompt, max_tokens) -> (str, int)`
  Supports: groq, anthropic, gemini, openai. Local imports per provider.
- `RateLimiter(max_rpm, max_tpm, max_rpd)` — rolling window, tracks RPM + TPM + RPD
- `keyword_prescore(job, config) -> float` — pure Python, 0.0–1.0, skip LLM if < 0.15
- `score_dimensions(job, config, llm_call, structured_profile) -> dict`
  Single LLM call. Returns: disqualified, fit_score, dimension_scores, reasons, flags, one_liner.
  Handles disqualifier check + dimension scoring in one prompt.
  Adjusts prompt based on `config["profile"].get("job_type")` — intern vs fulltime.
- `compute_ats_score(job, config) -> dict` — pure Python keyword overlap, returns ats_score + skill_misses
- `score_all_jobs(config, yes=False) -> list` — full pipeline with cost guard + rate limiting
- `print_results(results, config)` — terminal display with score bars

### candidate_profile.py

- `build_structured_profile(config, llm_call) -> dict`
  Extracts structured profile from raw resume once at startup.
  Intern path adjusts the extracted fields (graduation_year, school, major instead of past_roles/yoe).
- `print_profile_summary(profile)` — prints profile to terminal for verification
- `confirm_profile() -> bool` — asks user y/n before scoring begins

### text_utils.py

- `extract_job_context(text, max_chars=2000) -> str`
  Smart truncation that prioritizes high-signal sections (requirements, stack,
  role description, compensation) over boilerplate. Falls back to head+tail.
- `_head_tail(text, max_chars) -> str` — fallback: 60% head + 40% tail

### theirstack.py

- `fetch_companies(config) -> list` — queries TheirStack API, filtered to US tech companies
- `resolve_greenhouse_slug(company) -> str | None` — tries slug candidates against GH API
- `resolve_lever_slug(company) -> str | None` — tries slug candidates against Lever API
- `get_or_discover_slugs(config) -> dict[str, list[str]]`
  Returns `{"greenhouse": [...], "lever": [...]}`.
  Priority list from config → cached from DB → newly discovered via TheirStack.
  Only runs TheirStack discovery if THEIRSTACK_API_KEY is set.

### main.py — CLI flags

```
python main.py                  # full run: scrape → score → display
python main.py --scrape-only    # scrape only
python main.py --score-only     # score unscored jobs in DB
python main.py --show           # display top jobs, no API calls
python main.py --rescore        # clear scores, re-score everything
python main.py --yes            # skip all confirmation prompts
python main.py --min-score 75   # override display threshold
python main.py --profile manav  # use profiles/manav/config.yaml + jobs.db
```

---

## Sources

### Greenhouse
- API: `GET https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true`
- No auth, pure JSON
- Key fields: `id` (source_id), `title`, `location.name`, `absolute_url`, `content` (HTML, double-encoded)
- Date filter: `updated_at` ISO timestamp, skip if > max_job_age_days old

### Lever
- API: `GET https://api.lever.co/v0/postings/{slug}?mode=json`
- No auth, returns flat JSON array
- Key fields: `id`, `text` (title), `categories.location`, `workplaceType`, `hostedUrl`, `description` (HTML), `lists[]` (sections), `createdAt` (milliseconds)
- Date filter: `createdAt / 1000` compared to max_job_age_days

### HN Who's Hiring
- Find thread: Algolia search for latest post by `whoishiring`
- Fetch comments: `https://hn.algolia.com/api/v1/search?tags=comment,story_{id}&hitsPerPage=200`
- Each comment = one job posting, free-form text
- Filters: hiring intent proximity check + US signal check applied to full text
- source_id: Algolia `objectID`

### TheirStack (company discovery)
- API: `POST https://api.theirstack.com/v1/companies/search`
- Requires `THEIRSTACK_API_KEY` env var — optional, scraper works without it
- Filters to: US, tech industries, 10-5000 employees, sorted by theirstack_score
- Results cached in `discovered_greenhouse_slugs` and `discovered_lever_slugs` tables

---

## Scoring pipeline (per job)

```
1. keyword_prescore()     → if < 0.15, score=0, skip LLM entirely
2. score_dimensions()     → single LLM call
   - disqualifier check (clearance, relocation, masters required, yoe > max, internship)
   - if disqualified: score=0, return early
   - dimension scoring: role_fit, stack_match, seniority, location, growth, compensation (each 0-10)
   - fit_score computed in Python from weighted dimensions × 10
3. compute_ats_score()    → pure Python keyword overlap, no LLM
4. save_score()           → write to scores table
```

Fit score formula:
```python
fit_score = (
    role_fit    * weights["role_fit"]    +
    stack_match * weights["stack_match"] +
    seniority   * weights["seniority"]   +
    location    * weights["location"]    +
    growth      * weights["growth"]      +
    compensation* weights["compensation"]
) * 10
```

---

## Profile system (Step 6 — next to build)

Two profiles: Manav (full-time) and sister (TBD).

Each profile is fully isolated:
- Own `config.yaml` with all preferences
- Own `jobs.db` with own scraped + scored jobs
- Own `resume.pdf`

Profile-aware functions accept `profile=None` and fall back to root for CLI use.

### Onboarding flow (Streamlit, `onboarding.py`)

Multi-step Streamlit form using `st.session_state` for step tracking.
Generates a complete `config.yaml` for a new profile.

**Step 1 — Job type**
- Full-time or Internship (radio)
- If Internship: target season (Summer/Fall/Spring), year (2025/2026/2027),
  school, major, GPA (optional)

**Step 2 — Basic info**
- Name
- Resume: PDF upload (saves to `profiles/{slug}/resume.pdf`) or paste text
- Bio: text area with example placeholder

**Step 3 — Job preferences (branched)**

Full-time:
- Target titles (multi-select + free text)
- Desired skills (same)
- Hard no keywords (pre-filled, editable)
- Years of experience
- Minimum salary

Internship:
- Pre-filled intern titles: "Software Engineering Intern", "SWE Intern",
  "ML Intern", "AI Intern", "Backend Intern", "Data Engineering Intern"
- Desired skills
- Hard no keywords (intern-appropriate defaults)
- Stipend expectation (optional, for scoring context only)
- Skip YOE and minimum salary

Both:
- Remote ok toggle
- Preferred locations multi-select

**Step 4 — LLM provider**

Comparison table showing RPM / RPD / accuracy rating per model.
Radio button to select. API key field with helper:
"Don't have an API key? Contact admin for assistance."

**Step 5 — Review + Create**

Summary of all collected data. Profile name slug field (auto-suggested, editable).
Confirm creates: profile folder, config.yaml, initializes DB.

### Intern profile auto-settings

When job_type = "internship", config.yaml is generated with:
```yaml
preferences:
  filters:
    max_yoe: 1
    max_job_age_days: 14
    title_blocklist: [Staff, Principal, VP, Director, Head of, Manager, Lead, Full-time only]
```

Scoring prompt adjusted to frame candidate as student targeting internship.

### Dashboard (`dashboard.py`)

Streamlit entry point. Lists existing profiles in `profiles/` folder.
"Continue as {name}" button per profile.
"Add new profile" launches onboarding.
Active profile stored in `st.session_state["active_profile"]`.

---

## Key decisions and constraints

- **No Playwright yet.** Auto-apply is out of scope.
- **SQLite for prototype.** Postgres in Phase 2 before Railway deploy.
- **Each step runs independently.** Don't couple steps before they're both built.
- **Dedup is source-based.** ID = `{source}_{source_id}`.
- **raw_text capped at 2000 chars** using smart extraction prioritizing requirements/stack/role.
- **LLM is swappable** via config.yaml — no code changes to switch models.
- **Rate limiter is mathematical** — rolling window, sleeps only as needed.
- **Profile system is additive** — CLI behavior unchanged with no --profile flag.
- **Admin manages API keys** — onboarding shows key field with "contact admin" note.

---

## Environment variables

```
ANTHROPIC_API_KEY=    # if provider: anthropic
GEMINI_API_KEY=       # if provider: gemini
GROQ_API_KEY=         # if provider: groq (current default)
OPENAI_API_KEY=       # if provider: openai
THEIRSTACK_API_KEY=   # optional — enables dynamic company discovery
```

---

## Upcoming: Step 6 implementation notes

- `db.py` and `config.py` need `profile=None` parameter threading through all functions
- `get_db_path(profile=None)` returns `profiles/{profile}/jobs.db` or root `jobs.db`
- `init_db(profile=None)` creates `profiles/{profile}/` folder if needed
- `main.py --profile` loads the real profile instead of printing stub message
- Profile slugs: lowercase, spaces → underscores, strip special chars
- Duplicate profile name → warning, don't overwrite
- `streamlit run dashboard.py` is the new primary entry point for web use
- `python main.py` remains the CLI entry point
