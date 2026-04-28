# Job Agent – Realigned Roadmap (v3.0)

**Last updated:** April 28, 2026  
**Current state:** Phases 1–3 complete (steps 1–24)  
**Goal:** A personal AI job search assistant that finds, ranks, and applies to relevant jobs with human judgment gates.

---

## Completed Phases ✅

### Phase 1: Core Pipeline (Steps 1–8)
- ✅ Skeleton, scrapers (Greenhouse, Lever, HN Who's Hiring, Ashby, Workable, Himalayas)
- ✅ LLM-based scoring with profile-aware ranking
- ✅ Streamlit dashboard with 5-section navigation
- ✅ Pydantic models + instructor structured output
- ✅ SQLite schema with job, score, run, embedding persistence

### Phase 2: Intelligence & Observability (Steps 9–16)
- ✅ Loguru structured logging + scrape_runs table
- ✅ LLM resilience layer (llm_utils.py) for Groq wrapped scalars
- ✅ Out-of-process pipeline (subprocess + lockfile)
- ✅ Disqualified job filtering + "Filtered out" audit trail
- ✅ Scrape-filter rejection persistence (scrape_qualified, scrape_filter_reason)

### Phase 3: Semantic Intelligence (Steps 17–24)
- ✅ Semantic job embeddings (sentence-transformers, ChromaDB ANN)
- ✅ Vector retrieval with profile-scoped indexing
- ✅ Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
- ✅ Role-agnostic profile intent system (software_engineering, finance, internship, etc.)
- ✅ Evidence-based matching (MatchEvidence with positive signals + concerns)
- ✅ Factor-wise explanations (match_explainer.py)
- ✅ MLflow optional observability (safe fail-safe integration)

---

## What's Ready Today

**Finding & Ranking:** You have a state-of-the-art job discovery pipeline.

- Scrapes 6 ATS platforms + open-ended job boards
- Semantic retrieval + cross-encoder precision filtering
- Profile-aware scoring with role-family section weights
- Clear match explanations (why is this job ranked #3?)
- MLflow optional experiment tracking for portfolio

**What's Missing:** Application efficiency.

- No form automation → you must manually fill every Greenhouse/Lever/etc. form
- No application history tracking → you can't remember which companies you applied to
- No audit trail of what you sent → hard to reference responses later
- No ATS field detection → you don't know what fields are coming until you see them

---

## Phase 4: Selective LLM Routing (Step 26)

**Goal:** Reduce API costs 60% while maintaining ranking quality.

| Feature | Description | Effort | Impact |
|---------|-------------|--------|--------|
| **Dynamic routing** | Reranker score ≥ threshold → LLM; else keep reranker score (no LLM call) | 2–3 days | 60% cost cut, faster daily runs |
| **Cost/Quality slider** | Streamlit toggle: cheap (GPT-4o-mini) ↔ quality (Claude/GPT-4) | 1 day | Fine-grained control |
| **Latency tracking** | Log retrieval/rerank/LLM timing; alert if any stage is slow | 1 day | Observe bottlenecks |
| **One-hour sanity check** | Post-live: compare top-5 LLM jobs vs. top-5 routed jobs; verify no great matches missed | 1 hour | Confidence in threshold quality |

**Sanity Check Protocol (post-implementation):**
```
1. Run full pipeline once (all jobs via LLM, expensive).
2. Run full pipeline again with routing enabled (cheap run).
3. Extract top-5 jobs from each run.
4. Manually review: Are any great matches missing from the routed top-5?
5. If drift detected, adjust threshold or model.
6. If all looks good, mark threshold as "verified".
7. Future: optionally log top-10 job IDs each run + monthly spot-check for drift.
```

**Deliverable:** 
- CLI flag `--selective-routing` + dashboard toggle
- LLM calls only for jobs with reranker score ≥ threshold (e.g., 0.65)
- Documented sanity check results (top-5 comparison, any misses noted)
- Vector rank still displayed for transparency, but routing uses reranker

**Status:** Not started.  
**Why next:** Unblocks daily runs without API fear. Sanity check (1 hour) gives you confidence you didn't lose quality.

---

## Phase 5: Application Efficiency (Steps 27, 28, 30)

### Step 27: Resume Intelligence Upgrade

**Goal:** Extract cleaner structured profile data for form fills.

| Feature | Description | Effort | Impact |
|---------|-------------|--------|--------|
| **pdfplumber upgrade** | Replace PyPDF2 with pdfplumber for better text extraction | 1 day | More reliable resume parsing |
| **Structured extraction** | Auto-extract: years_of_experience, education, certifications, skills | 1 day | Cleaner prefill for forms |
| **Resume caching** | Store extracted JSON in profile.db; don't re-parse on every run | 0.5 days | Faster pipeline startup |

**Deliverable:** `resume_parsed` table in profile.db with structured fields.

**Status:** Not started.  
**Why:** Better input → better form fills in Phase 6.

---

### Step 28: Shortlist CRM

**Goal:** Replace spreadsheet chaos with a simple, queryable application tracker.

**New table: `applications`**
```sql
CREATE TABLE applications (
  id TEXT PRIMARY KEY,           -- job_id
  job_title TEXT,
  company TEXT,
  job_url TEXT,
  status TEXT,                   -- 'shortlist', 'applied', 'interviewing', 'offer', 'rejected', 'withdrew'
  applied_at TIMESTAMP,
  deadline TIMESTAMP,
  referral_contact TEXT,         -- name + email if applicable
  notes TEXT,
  user_notes TEXT,
  last_updated TIMESTAMP
);
```

| Feature | Description | Effort | Impact |
|---------|-------------|--------|--------|
| **Status tracking** | UI buttons: Shortlist → Applied → Interviewing → Offer/Reject | 2 days | Single source of truth |
| **Notes + deadlines** | Persist deadline, referral name, custom notes per job | 1 day | Stop losing context |
| **Dashboard panel** | View all applications; filter by status/company/date; bulk-mark applied | 2 days | See your pipeline at a glance |
| **Analytics** | Count by status; show application→interview conversion rate | 1 day | Measure what's working |

**Deliverable:** New "Applications" tab in dashboard. Click job → "Add to shortlist" → later "Mark as applied".

**Status:** Not started.  
**Why:** You'll apply to 5–10 jobs per day. You need a way to not lose track of them.

---

### Step 30: Application Packet Storage

**Goal:** Persist all per-job materials in a durable, reproducible way.

**New table: `application_packets`**
```sql
CREATE TABLE application_packets (
  id TEXT PRIMARY KEY,           -- unique packet ID
  job_id TEXT REFERENCES jobs(id),
  created_at TIMESTAMP,
  match_explanation TEXT,        -- JSON: MatchExplanation from match_explainer.py
  prefilled_answers JSON,        -- {field_name: value, ...} for form fields we auto-filled
  user_answers JSON,             -- {field_name: custom_response} for questions user wrote
  resume_attached TEXT,          -- path/hash to resume used
  status TEXT,                   -- 'draft', 'pending_review', 'submitted', 'error'
  submission_error TEXT,         -- error message if submit failed
  submitted_at TIMESTAMP
);
```

| Feature | Description | Effort | Impact |
|---------|-------------|--------|--------|
| **Packet creation** | Before submitting, bundle: match explanation + prefilled answers + user responses | 1 day | Audit trail |
| **Draft persistence** | Save & recall packet if user doesn't submit immediately | 0.5 days | Resume interrupted flows |
| **Submission log** | Store final state: what we submitted, when, any errors | 0.5 days | Reproducibility |
| **Replay** | For similar future job, reuse packet + edit (don't re-answer everything) | 1 day | Speed up repeated applications |

**Deliverable:** `application_packets` table + UI to view/edit/submit stored packets.

**Status:** Not started.  
**Why:** You'll apply to 100+ jobs this cycle. Without this, responses are lost and each form feels brand-new.

---

## Phase 6: Human-in-the-Loop Application Automation (Steps 32, 33, 34, 35)

**Philosophy:** Playwright automates *typing*, you make the *decisions*.

### Step 32: ATS Form Field Detection

**Goal:** Understand form structure *before* you try to fill it.

| Feature | Description | Effort | Impact |
|---------|-------------|--------|--------|
| **Field type detection** | Playwright snapshot → LLM analysis: detect text, textarea, dropdown, file, yes/no | 2 days | Know what's coming |
| **Field name extraction** | Extract visible labels: "Name", "Email", "Why do you want to work here?" | 1 day | Match to profile data |
| **Field precedence** | Rank fields: standard (name, email, resume) vs. custom (interview questions) | 1 day | Prioritize automation |
| **Blockers detection** | Detect: login required, CAPTCHA, "no third-party autofill" language | 1 day | Know when to give up gracefully |

**Deliverable:** `analyze_form(url, profile) → FormAnalysis` with field list + detection confidence.

**Status:** Not started.  
**Why:** Greenhouse/Lever forms vary wildly. You need to know what you're facing before attempting fill.

---

### Step 33: Human Approval Gate

**Goal:** Show the user what the system *wants* to submit before it submits.

| Feature | Description | Effort | Impact |
|---------|-------------|--------|--------|
| **Form preview** | Render filled form in Streamlit: what we auto-filled + what we need you to write | 2 days | Final safety check |
| **Field editing** | User can edit/override any auto-filled field before approve | 1 day | Catch mistakes |
| **Question highlights** | Show "Requires custom response" for fields we can't auto-fill | 0.5 days | User focus |
| **Approve button (disabled by default)** | Checkbox "I reviewed this and approve submission" → button unlocks | 0.5 days | Friction-full safety |

**Deliverable:** Dashboard modal: form preview + edit fields + "Confirm & Submit" button.

**Status:** Not started.  
**Why:** This is your safety layer. Never auto-submit without explicit approval.

---

### Step 34: Semi-Automated Form Filling

**Goal:** Use profile data to fill known fields; pause for custom responses.

| Feature | Description | Effort | Impact |
|---------|-------------|--------|--------|
| **Standard field mapping** | name → profile.name; location → profile.location; resume → profile.resume_file | 2 days | Reduce 50% of typing |
| **Degree/education** | Extract from parsed resume; match to dropdown options if possible | 1 day | Automatable credential |
| **Years of experience** | Calculate from resume dates; fill YoE dropdowns | 0.5 days | Deterministic |
| **Custom response placeholders** | For "Why do you want to work here?", show template + LLM suggestions (user selects) | 2 days | Intelligent defaults |
| **File upload handling** | Identify resume upload field; attach resume.pdf automatically | 1 day | One less manual step |

**Deliverable:** `fill_form_with_profile(driver, form_analysis, profile, user_answers) → FormFilledResult`

**Status:** Not started.  
**Why:** 70% of form fields are just typing profile data. Automate that; leave judgment to you.

---

### Step 35: Approved Submit Automation

**Goal:** Once user approves, click submit & log result.

| Feature | Description | Effort | Impact |
|---------|-------------|--------|--------|
| **Safe submit** | User clicked "Confirm & Submit" → script clicks submit button | 1 day | No mistakes on your part |
| **Post-submit handling** | Capture success page / error page screenshot; detect if submit worked | 1 day | Know if it went through |
| **Application log** | Write application_packets record: submitted_at, final_status | 0.5 days | Audit trail |
| **Error recovery** | If submit fails, save state + suggest manual retry | 0.5 days | Don't lose data on network failure |
| **Rate limiting** | Don't submit >1 job per 5 seconds (ATS throttling) | 0.5 days | Don't trigger bot detection |

**Deliverable:** `submit_application(driver, packet_id, config) → SubmissionResult`

**Status:** Not started.  
**Why:** Final step. After approval, get out of the way and let the system finish the job.

---

## Implementation Order & Dependencies

```
Phase 4 (Step 26)
 ├─ Selective LLM Routing (2–3 days)
 └─ Unblocks: Daily runs without API fear

Phase 5 (Steps 27, 28, 30)
 ├─ Step 27: Resume Intelligence (1 day) — improves profile data quality
 ├─ Step 28: Shortlist CRM (2 days) — tracks where you've applied
 ├─ Step 30: Application Packets (2 days) — stores form responses
 └─ Parallel work (independent)

Phase 6 (Steps 32–35)
 ├─ Step 32: ATS Form Detection (3 days) — understand form structure
 ├─ Step 33: Approval Gate (2 days) — safety review before submit
 ├─ Step 34: Semi-Automated Filling (3 days) — map profile → form fields
 └─ Step 35: Submit Automation (2 days) — click submit after approval

Total effort: ~24–27 days (realistically 4–5 weeks with testing + debugging)
```

---

## What NOT to Do (Explicit Scope Exclusions)

| Exclusion | Why | Deferred to |
|-----------|-----|------------|
| **Resume generation** | You write your own resume tailored per role type | Later (Phase 7) if at all |
| **Cover letter generation** | Too template-y; you handle custom responses | Step 34 "custom response suggestions" |
| **Architecture refactor (23A)** | Code is clean enough; velocity > cleanliness for personal tool | Never (unnecessary) |
| **BM25 hybrid retrieval (25B)** | Vector + cross-encoder is already very good; only add if search fails | When search quality degrades |
| **Full evaluation dataset (25)** | Portfolio signal, not functional requirement | Phase 7 if needed |
| **Full automation without approval** | Too risky; human judgment is the safety valve | Never (by design) |
| **Application submission without Playwright safety** | Form field mismatch = spam applicant profile | Always require explicit approval + logging |

---

## Success Metrics

### Phase 4 (Step 26)
- ✅ API cost per run cut by 50%+ (track in MLflow)
- ✅ Reranker score ≥ LLM score (verify ranking stays good)
- ✅ Daily run completes in <10 min (latency target)

### Phase 5 (Steps 27, 28, 30)
- ✅ Shortlist CRM has all applications you made (100% coverage)
- ✅ You can replay an application packet in <30 seconds (edit + resubmit)
- ✅ Application→Interview conversion rate visible in dashboard

### Phase 6 (Steps 32–35)
- ✅ Form detection accuracy >90% (doesn't confuse text fields for dropdowns)
- ✅ Auto-fill covers 60%+ of form fields (name, email, resume, education, YoE)
- ✅ You apply to 5+ jobs per day without decision fatigue (form preview handles safety)
- ✅ 0 spam applications (no unreviewed submissions)
- ✅ Application time per job <2 minutes (excluding custom response writing)

---

## Timeline

- **Week 1:** Step 26 (selective routing) + Step 27 (resume intelligence)
- **Week 2:** Steps 28 + 30 (CRM + packet storage)
- **Week 3–4:** Steps 32–35 (form automation pipeline)
- **Week 5:** Integration testing + edge case handling

**Milestone at Week 2:** You can run the pipeline daily without API fear, and track all applications.  
**Milestone at Week 5:** You can apply to vetted jobs with one click + review.

---

## Technical Notes

### Why Playwright (not Selenium)?
- Modern API, better stealth (stealth plugin available)
- Built-in waits for dynamic content
- Better browser pool management for multiple jobs
- Easier debugging (native async/await)

### Why human approval gate (not fully auto)?
- ATS field detection is ~90% accurate; 10% means mistakes
- Custom response questions need your voice, not a template
- One unreviewed spam application = damage to your recruiter relationships
- Humans can read context (company values, job description nuance) in seconds

### Why application_packets table?
- Audit trail: what did you actually submit?
- Replay: similar role next week = edit packet, not re-answer
- Debugging: if interview, you can reference what you said
- Multi-user future: sister's packets stay separate

### Cost/Quality Slider Design
```yaml
# Low-cost (fast iteration)
routing:
  selective: true
  threshold: 0.60  # reranker score ≥ 60 → skip LLM
  quality_mode: "fast"  # use gpt-4o-mini or groq fast
  
# High-quality (thorough evaluation)
routing:
  selective: true
  threshold: 0.75  # higher bar for LLM
  quality_mode: "quality"  # use claude-sonnet or gpt-4
```

---

## Known Risks & Mitigation

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Playwright stealth blocked by Cloudflare | Medium | Use official `playwright_stealth` plugin; if unreliable, upgrade to **Patchright** or **Nodriver** (drop-in replacements with deeper fingerprint evasion). Monitor GitHub issues for latest. |
| Form field detection fails for custom ATS | Medium | Manual fallback: pause and show form preview for human inspection |
| Typos in auto-filled fields (name, email) | Low | Require human review before submit (Step 33 approval gate) |
| Application rate limiting | Low | Track submission timestamps; enforce 5-second minimum between submits |
| Resume PDF upload (wrong format/size) | Low | Validate file size <10MB before attempting upload |
| Login-required ATS (no public apply) | Low | Skip gracefully; alert user to manual apply |

---

## Dependencies & Environment

```
# Existing (already in requirements.txt)
streamlit>=1.52.0
pydantic>=2.7.0
instructor>=1.3.0
loguru>=0.7.0
sentence-transformers~=3.2.1
chromadb>=0.5,<0.6
mlflow>=2.16,<3.0

# New for Phase 4
# (no new dependencies; routing logic is pure Python)

# New for Phase 5
pdfplumber>=0.10.0  # upgrade from PyPDF2

# New for Phase 6
playwright>=1.40.0  # (already in requirements)
python-dotenv>=1.0.0  # for Playwright credentials if needed
```

---

## How to Track Progress

**Use GitHub Issues labeled:**
- `phase/4-selective-routing`
- `phase/5-application-efficiency`
- `phase/6-automation`

**Use MLflow to track:**
- Selective routing cost savings (API call count before/after)
- Application success rate (submitted / attempted)
- Form field detection accuracy (ground truth vs. detected)

**Use dashboard metrics to track:**
- Jobs scraped per run
- Jobs scored per run
- Application shortlist size
- Application→interview conversion rate

---

## When to Ship to Sister

Once **Phase 5 (Steps 27, 28, 30)** is stable:
1. Create profile: `jia_shah` (existing)
2. She runs `python main.py --profile jia_shah --onboard` → sets role_family=finance, targets, location
3. She can scrape, see ranked jobs, mark shortlist
4. Wait until Phase 6 before automating applies for her (different ATS landscape, more caution needed)

---

## Open Questions

1. **Should selective routing use semantic score or reranker score as the cutoff?**  
   → Reranker is profile-aware, so prefer that. Threshold ~0.65.

2. **For custom response suggestions (Step 34), should we use Claude or cheap model?**  
   → Cheap model (Groq fast) for speed; user edits anyway.

3. **Should application_packets table also store rejected/withdrawn applications, or only submitted ones?**  
   → Store all. Useful to know why you rejected a job (changed mind, better offer, etc.).

4. **How many previous applications should "replay suggestions" scan?**  
   → Last 20 submitted packets to the same company (find similar response patterns).

---

## Success Definition

**You will consider this project complete when:**

✅ You can find jobs via the dashboard  
✅ You can rank them by relevance  
✅ You can review top matches with clear explanations  
✅ **You can apply to vetted jobs with one click + human review (NEW)**  
✅ **You can track where you've applied & what you said (NEW)**  
✅ **You never spam recruiter by accident (approval gate) (NEW)**  
✅ You spend <2 min per application (typing form boilerplate) (NEW)  

And your sister can do the same for finance internships.

---

## Next Action

1. **Confirm this roadmap aligns with your goals.** (This doc)
2. **Code Step 26 (selective routing).** (~3 days)
3. **Validate: run daily for a week, track cost savings.**
4. **Code Steps 27, 28, 30 in parallel.** (~5 days)
5. **Code Steps 32–35 with careful testing.** (~8 days)

**Estimated total: 3–4 weeks to full application automation.**

Good luck. This is a real, useful tool.