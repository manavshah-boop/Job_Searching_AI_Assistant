# Step 26 – Selective LLM Routing

## What it does

Selective routing uses a cross-encoder score as a gate before each LLM call.

- Jobs whose cross-encoder score **≥ threshold** → LLM scores the job normally
- Jobs whose cross-encoder score **< threshold** → synthetic score is generated (no API call)

Result: same top-ranked jobs, 60–90% fewer API calls, tunable per profile.

---

## How to enable

In your profile's `config.yaml` (e.g. `profiles/default/config.yaml`):

```yaml
routing:
  enabled: true
  threshold: 0.65          # start here; tune after the sanity check
  quality_mode: quality    # "fast" = cheaper model, "quality" = current model
  log_routing_decisions: true
```

Or override for a single run with CLI flags:

```bash
python main.py --selective-routing          # force ON for this run
python main.py --no-selective-routing       # force OFF (current behavior)
```

You can also toggle it in the **Settings → Cost optimization** panel in the dashboard.

---

## How it works

1. Before the scoring loop, the router builds a **profile match query** (same as the reranker).
2. For each unscored job, the cross-encoder computes a **routing score** (0–1) by comparing the job's raw text (first 1800 chars) to the match query.
3. If `routing_score >= threshold` → call LLM as normal, log `routed=llm_called`.
4. If `routing_score < threshold` → build a **synthetic score** from:
   - `routing_score` → maps to `role_fit` and `stack_match` (conservative, 85% scale)
   - `matched_sections` (pure-Python keyword detection) → adjusts `stack_match` and `seniority`
   - Neutral defaults for `location`, `growth`, `compensation`
5. Both paths save to the `scores` table. Every decision is also logged to `routing_decisions`.

---

## Synthetic score design

Synthetic scores are **intentionally lower** than LLM scores so LLM-scored jobs always rank higher at equal quality:

| routing_score | role_fit (LLM est.) | role_fit (synthetic) |
|---|---|---|
| 0.80 | ~8 | ~7 (0.80 × 0.85 × 10) |
| 0.70 | ~7 | ~6 |
| 0.65 | ~6–7 | ~5–6 |

If the `requirements` section of the job matched the query, `stack_match` gets a small boost (+8%). If `responsibilities` matched, `seniority` is lifted from 5 → 6.

---

## Threshold tuning guide

| Threshold | LLM calls | Quality | Cost |
|---|---|---|---|
| 0.55 | ~60% of jobs | High | Medium |
| **0.65** | **~25–35% of jobs** | **Good** | **Low** |
| 0.75 | ~10–20% of jobs | Moderate | Very Low |

**Start at 0.65.** Run the sanity check below. Then adjust:
- **Lower** (0.55–0.60): more LLM calls, better coverage, higher cost
- **Higher** (0.70–0.80): fewer calls, lower cost, check that top-5 stays stable

---

## Quality mode

`quality_mode` controls which LLM model is used for jobs **that do** call the LLM:

| quality_mode | groq | anthropic | openai | gemini |
|---|---|---|---|---|
| `fast` | llama-3.1-8b-instant | claude-haiku-4-5 | gpt-4o-mini | gemini-2.0-flash |
| `quality` | llama-4-scout (current) | claude-sonnet-4 | gpt-4o | gemini-2.5-flash |

Set `quality_mode: quality` to keep the current model for jobs that pass the threshold.  
Set `quality_mode: fast` for maximum cost savings (cheap model + fewer calls).

---

## Sanity check protocol (run once before shipping to daily runs)

**Goal:** Confirm top-5 jobs are the same with routing on vs. off.

```bash
# Step 1: full LLM run (routing disabled)
python main.py --profile default --no-selective-routing --score-only --yes

# Step 2: note the top-5 jobs by fit_score (from dashboard or query below)
sqlite3 profiles/default/jobs.db \
  "SELECT j.title, j.company, s.fit_score FROM jobs j
   JOIN scores s ON j.id = s.job_id
   ORDER BY s.fit_score DESC LIMIT 5;"

# Step 3: clear scores and re-run with routing enabled
python main.py --profile default --rescore --selective-routing --yes

# Step 4: check top-5 again
sqlite3 profiles/default/jobs.db \
  "SELECT j.title, j.company, s.fit_score FROM jobs j
   JOIN scores s ON j.id = s.job_id
   ORDER BY s.fit_score DESC LIMIT 5;"

# Step 5: check how many were routed
sqlite3 profiles/default/jobs.db \
  "SELECT routed_to, COUNT(*) FROM routing_decisions GROUP BY routed_to;"
```

**Expected result:**
```
LLM run top-5:    [Job A 85, Job B 82, Job C 79, Job D 77, Job E 75]
Routed run top-5: [Job A 82, Job B 79, Job C 76, Job D 74, Job E 72]

✅ Same 5 jobs (order preserved)
✅ Scores slightly lower (expected — synthetic scores are conservative)
✅ API calls: ~20 instead of ~200
```

If a great match drops out of top-5, lower the threshold by 0.05 and re-run.

---

## Querying routing decisions

```sql
-- How many jobs were LLM-scored vs. synthetic?
SELECT routed_to, COUNT(*) as count
FROM routing_decisions
GROUP BY routed_to;

-- Which jobs got synthetic scores?
SELECT rd.job_id, j.title, j.company, rd.routing_score, rd.threshold
FROM routing_decisions rd
JOIN jobs j ON rd.job_id = j.id
WHERE rd.routed_to = 'skipped_llm'
ORDER BY rd.routing_score DESC;

-- Estimated cost savings
SELECT
  COUNT(*) AS total,
  SUM(CASE WHEN routed_to = 'skipped_llm' THEN 1 ELSE 0 END) AS skipped,
  ROUND(100.0 * SUM(CASE WHEN routed_to = 'skipped_llm' THEN 1 ELSE 0 END) / COUNT(*), 1)
    AS savings_pct
FROM routing_decisions;
```

---

## Logging output

When enabled, every routing decision is logged at INFO level:

```
routing | job=gh_12345 company=Stripe routed=skipped_llm routing_score=0.582 threshold=0.65 matched_sections=[requirements] synthetic_fit=42
routing | job=gh_12346 company=OpenAI routed=llm_called routing_score=0.731 reason=meets_threshold
...
routing | summary total=200 llm_called=48 skipped=152 errors=0 cost_savings=76%
```

---

## Reverting to current behavior

Set `enabled: false` in your profile config (or use `--no-selective-routing`). Zero code changes required.
