"""
scorer.py — Step 4: Score unscored jobs using a multi-stage pipeline.

Pipeline per job:
  1. keyword_prescore()  — pure Python, no API call
  2. score_dimensions()  — single LLM call (disqualifier check + dimension scoring merged)
  3. compute_ats_score() — pure Python, no API call

Active provider and model are set in config.yaml under the `llm` key.
No code changes needed to switch providers.
"""

import json
import os
import re
import sys
import time
from collections import deque
from typing import Any, Callable, Dict, Optional, Tuple

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Ensure Unicode output works on Windows terminals.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from db import (
    Job,
    get_unscored,
    increment_score_attempts,
    init_db,
    load_config,
    rescore_reset,
    save_score,
    write_score_error,
)
from candidate_profile import build_structured_profile, confirm_profile, print_profile_summary
from llm_utils import parse_llm_response, safe_structured_call
from models import ScoreResult, StructuredProfile

class RateLimitReached(Exception):
    """Raised when the provider's daily RPD limit is exhausted."""


def _resolve_api_key(env_var: str, profile: Optional[str]) -> Optional[str]:
    """Try the profile-suffixed key first, then fall back to the unsuffixed name."""
    if profile:
        suffixed = os.environ.get(f"{env_var}_{profile.upper()}")
        if suffixed:
            return suffixed
    return os.environ.get(env_var)


# LlmCall type: (prompt, max_tokens) -> (response_text, tokens_used)
LlmCall = Callable[[str, int], Tuple[str, int]]

# Providers that bill per token — show real cost estimate
_PAID_PROVIDERS = {"anthropic", "openai"}


# Provider-specific notes shown before confirmation prompt
_PROVIDER_NOTES: Dict[str, Any] = {
    "gemini": lambda n: f"Free tier — 15 RPM limit (est. {n * 60 / 14 / 60:.1f} min total runtime)",
}


class RateLimiter:
    """
    Tracks requests and tokens within a rolling 60-second window, plus a
    daily request counter. Sleeps only as long as needed to stay under
    RPM, TPM, and RPD limits.
    """

    def __init__(self, max_rpm: int, max_tpm: int, max_rpd: Optional[int] = None) -> None:
        self.max_rpm  = max_rpm
        self.max_tpm  = max_tpm
        self.max_rpd  = max_rpd
        self.requests: deque = deque()   # timestamps of recent requests (rolling 60s)
        self.tokens:   deque = deque()   # (timestamp, token_count) tuples (rolling 60s)
        self.daily_requests = 0
        self.day_start      = time.time()

    def wait_if_needed(self) -> None:
        now    = time.time()
        window = 60.0

        # Reset daily counter if more than 24 hours have passed
        if now - self.day_start > 86400:
            self.daily_requests = 0
            self.day_start = now

        # Hard stop on RPD — raise exception so callers can record status
        if self.max_rpd and self.daily_requests >= self.max_rpd:
            reset_in = 86400 - (now - self.day_start)
            msg = f"Daily request limit reached ({self.max_rpd} RPD). Resets in {reset_in / 3600:.1f} hours."
            logger.warning(msg)
            raise RateLimitReached(msg)

        # Drop entries outside the rolling window
        while self.requests and now - self.requests[0] > window:
            self.requests.popleft()
        while self.tokens and now - self.tokens[0][0] > window:
            self.tokens.popleft()

        rpm_wait = 0.0
        if len(self.requests) >= self.max_rpm:
            rpm_wait = window - (now - self.requests[0])

        tpm_used = sum(t for _, t in self.tokens)
        tpm_wait = 0.0
        if tpm_used >= self.max_tpm:
            tpm_wait = window - (now - self.tokens[0][0])

        wait = max(rpm_wait, tpm_wait)
        if wait > 0:
            logger.warning(f"Rate limit — pausing {wait:.1f}s...")
            time.sleep(wait)

    def record(self, tokens_used: int) -> None:
        now = time.time()
        self.requests.append(now)
        self.tokens.append((now, tokens_used))
        self.daily_requests += 1


# ── LLM client factory ────────────────────────────────────────────────────────

def get_llm_client(config: Dict[str, Any]) -> LlmCall:
    """
    Returns a callable: (prompt: str, max_tokens: int) -> str

    The same interface regardless of provider — the rest of the scorer
    never knows which backend it's talking to.

    SDKs are imported lazily inside each branch so a missing SDK only
    fails if that provider is actually selected.
    """
    provider    = config["llm"]["provider"]
    models      = config["llm"]["model"]
    temperature = config["llm"].get("temperature", 0)
    profile     = config.get("_active_profile")

    if provider == "anthropic":
        import anthropic
        api_key = _resolve_api_key("ANTHROPIC_API_KEY", profile)
        if not api_key:
            names = f"ANTHROPIC_API_KEY_{profile.upper()} or ANTHROPIC_API_KEY" if profile else "ANTHROPIC_API_KEY"
            logger.error("%s not set in environment or .env", names)
            sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key)

        def call_anthropic(prompt: str, max_tokens: int = 700) -> Tuple[str, int]:
            response = client.messages.create(
                model=models["anthropic"],
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if block.type == "text":
                    text = block.text
                    break
            tokens = response.usage.input_tokens + response.usage.output_tokens
            return text, tokens

        return call_anthropic

    elif provider == "gemini":
        from google import genai
        from google.genai import types as genai_types
        api_key = _resolve_api_key("GEMINI_API_KEY", profile)
        if not api_key:
            names = f"GEMINI_API_KEY_{profile.upper()} or GEMINI_API_KEY" if profile else "GEMINI_API_KEY"
            logger.error("%s not set in environment or .env", names)
            sys.exit(1)
        client = genai.Client(api_key=api_key)
        gemini_model = models["gemini"]

        def call_gemini(prompt: str, max_tokens: int = 700) -> Tuple[str, int]:
            response = client.models.generate_content(
                model=gemini_model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(  # type: ignore[call-arg]
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
            tokens = getattr(response.usage_metadata, "total_token_count", 500)
            return response.text or "", tokens

        return call_gemini

    elif provider == "groq":
        from groq import Groq
        api_key = _resolve_api_key("GROQ_API_KEY", profile)
        if not api_key:
            names = f"GROQ_API_KEY_{profile.upper()} or GROQ_API_KEY" if profile else "GROQ_API_KEY"
            logger.error("%s not set in environment or .env", names)
            sys.exit(1)
        client = Groq(api_key=api_key)

        def call_groq(prompt: str, max_tokens: int = 700) -> Tuple[str, int]:
            response = client.chat.completions.create(
                model=models["groq"],
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or "", response.usage.total_tokens

        return call_groq

    elif provider == "openai":
        from openai import OpenAI
        api_key = _resolve_api_key("OPENAI_API_KEY", profile)
        if not api_key:
            names = f"OPENAI_API_KEY_{profile.upper()} or OPENAI_API_KEY" if profile else "OPENAI_API_KEY"
            logger.error("%s not set in environment or .env", names)
            sys.exit(1)
        client = OpenAI(api_key=api_key)

        def call_openai(prompt: str, max_tokens: int = 700) -> Tuple[str, int]:
            response = client.chat.completions.create(
                model=models["openai"],
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or "", response.usage.total_tokens

        return call_openai

    else:
        logger.error(f"Unknown provider '{provider}'. Options: anthropic, gemini, groq, openai")
        sys.exit(1)


# ── Instructor client factory (for structured outputs) ────────────────────────

def get_instructor_client(config: Dict[str, Any]) -> Tuple[Any, str, float]:
    """
    Returns (instructor_client, model, temperature) for structured LLM outputs.
    Uses instructor to wrap the provider's client for reliable JSON parsing.
    
    Returns:
      - instructor_client: instructor-wrapped client ready for structured calls
      - model: model name string for that provider
      - temperature: temperature setting from config
    """
    provider    = config["llm"]["provider"]
    models      = config["llm"]["model"]
    temperature = config["llm"].get("temperature", 0)
    profile     = config.get("_active_profile")

    if provider == "anthropic":
        import anthropic
        import instructor
        api_key = _resolve_api_key("ANTHROPIC_API_KEY", profile)
        if not api_key:
            names = f"ANTHROPIC_API_KEY_{profile.upper()} or ANTHROPIC_API_KEY" if profile else "ANTHROPIC_API_KEY"
            logger.error("%s not set in environment or .env", names)
            sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key)
        client = instructor.from_anthropic(client)
        return client, models["anthropic"], temperature

    elif provider == "gemini":
        from google import genai
        import instructor
        api_key = _resolve_api_key("GEMINI_API_KEY", profile)
        if not api_key:
            names = f"GEMINI_API_KEY_{profile.upper()} or GEMINI_API_KEY" if profile else "GEMINI_API_KEY"
            logger.error("%s not set in environment or .env", names)
            sys.exit(1)
        client = genai.Client(api_key=api_key)
        # Gemini doesn't have direct instructor support yet, so we use raw client
        # Fall back to get_llm_client for gemini
        return None, models["gemini"], temperature

    elif provider == "groq":
        from groq import Groq
        import instructor
        api_key = _resolve_api_key("GROQ_API_KEY", profile)
        if not api_key:
            names = f"GROQ_API_KEY_{profile.upper()} or GROQ_API_KEY" if profile else "GROQ_API_KEY"
            logger.error("%s not set in environment or .env", names)
            sys.exit(1)
        client = Groq(api_key=api_key)
        client = instructor.from_groq(client)
        return client, models["groq"], temperature

    elif provider == "openai":
        from openai import OpenAI
        import instructor
        api_key = _resolve_api_key("OPENAI_API_KEY", profile)
        if not api_key:
            names = f"OPENAI_API_KEY_{profile.upper()} or OPENAI_API_KEY" if profile else "OPENAI_API_KEY"
            logger.error("%s not set in environment or .env", names)
            sys.exit(1)
        client = OpenAI(api_key=api_key)
        client = instructor.from_openai(client)
        return client, models["openai"], temperature

    else:
        logger.error(f"Unknown provider '{provider}'. Options: anthropic, gemini, groq, openai")
        sys.exit(1)


# ── 1. Keyword pre-score ──────────────────────────────────────────────────────

def keyword_prescore(job: Job, config: Dict[str, Any]) -> float:
    """
    Pure Python. Fraction of desired_skills present in raw_text.
    Returns 0.0–1.0. Below 0.15 → skip the LLM call entirely.

    Exception: if the job title is an exact (case-insensitive) match to any
    preferred title in preferences.titles, return 1.0 to guarantee LLM scoring.
    A user who explicitly listed a title wants every instance of it scored.
    """
    preferred_titles = [t.lower() for t in config.get("preferences", {}).get("titles", [])]
    if preferred_titles and job.title.lower() in preferred_titles:
        return 1.0

    text   = job.raw_text.lower()
    skills = config["preferences"]["desired_skills"]
    if not skills:
        return 0.0
    matches = sum(1 for s in skills if s.lower() in text)
    return matches / len(skills)


# ── 2. Merged disqualifier + dimension scoring (single LLM call) ──────────────

def _llm_call_with_retry(llm_call: LlmCall, prompt: str, max_tokens: int, retries: int = 3) -> Tuple[str, int]:
    """Calls llm_call with exponential backoff on 429 rate-limit errors."""
    for attempt in range(retries):
        try:
            return llm_call(prompt, max_tokens)
        except Exception as e:
            msg = str(e)
            is_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate_limit" in msg.lower()
            if is_rate_limit and attempt < retries - 1:
                wait = 2 ** attempt * 10  # 10s, 20s, 40s
                logger.warning(f"Rate limited. Retrying in {wait}s... (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("unreachable")


def score_dimensions(
    job: Job,
    config: Dict[str, Any],
    llm_call: LlmCall,
    structured_profile: Optional[Dict] = None,
    instructor_client: Optional[Any] = None,
) -> dict:
    """
    Single LLM call that handles both disqualifier detection and dimension
    scoring. Returns a dict with:
      disqualified, disqualify_reason, fit_score, dimension_scores,
      reasons, flags, one_liner.
    
    If instructor_client is provided, uses instructor for reliable structured output.
    Otherwise falls back to raw LLM call with JSON parsing.
    """
    prefs   = config.get("preferences", {})
    weights = config["scoring"]["weights"]
    max_yoe = prefs.get("filters", {}).get("max_yoe", 5)

    zeroed = {
        "disqualified": False,
        "disqualify_reason": "",
        "fit_score": 0,
        "tokens_used": 0,
        "dimension_scores": {
            "role_fit": 0, "stack_match": 0, "seniority": 0,
            "location": 0, "growth": 0, "compensation": 0,
        },
        "reasons": [],
        "flags": ["parse error"],
        "one_liner": "",
    }

    job_type = config.get("profile", {}).get("job_type", "fulltime")
    is_intern = job_type == "internship"
    compensation = prefs.get("compensation", {})
    intern_pay_preference = str(compensation.get("intern_pay_preference", "")).strip().lower()
    if is_intern and intern_pay_preference not in {"paid_only", "unpaid_ok", "no_preference"}:
        intern_pay_preference = "paid_only" if compensation.get("monthly_stipend") else "no_preference"

    def _format_intern_compensation_line(target_value: Any) -> str:
        stipend = None
        if target_value not in (None, "", 0):
            try:
                stipend = int(target_value)
            except (TypeError, ValueError):
                stipend = None

        if intern_pay_preference == "paid_only":
            if stipend is not None:
                return f"Compensation Preference: Paid only (target stipend: ${stipend:,}/mo)"
            return "Compensation Preference: Paid only"
        if intern_pay_preference == "unpaid_ok":
            return "Compensation: Unpaid OK"
        return "Compensation: Open to paid or unpaid"

    if structured_profile:
        sp = structured_profile
        remote_str = (
            "Remote preferred"
            if str(sp.get("remote_preference", "True")) == "True"
            else "Open to office"
        )
        profile_section = f"""Name: {sp.get("name", "")}
Experience: {sp.get("yoe", prefs.get("yoe", 0))} years
Core Skills: {", ".join(sp.get("core_skills", []))}
Languages: {", ".join(sp.get("languages", []))}
Frameworks: {", ".join(sp.get("frameworks", []))}
Cloud/Infra: {", ".join(sp.get("cloud", []))}
Past Roles: {", ".join(sp.get("past_roles", []))}
Education: {sp.get("education", "")}
Target Roles: {", ".join(sp.get("target_roles", prefs.get("titles", [])))}
Location: {remote_str}
Preferred Cities: {", ".join(sp.get("preferred_locations", []))}"""
        if is_intern:
            profile_section += f"""
{_format_intern_compensation_line(sp.get("target_salary", compensation.get("monthly_stipend")))}"""
        else:
            target_salary = sp.get("target_salary", compensation.get("min_salary"))
            if target_salary not in (None, "", 0):
                try:
                    profile_section += f"""
Min Salary: ${int(target_salary):,}"""
                except (TypeError, ValueError):
                    profile_section += """
Min Salary: Not set"""
            else:
                profile_section += """
Min Salary: Not set"""
    else:
        profile = config["profile"]
        profile_section = f"""Name: {profile.get('name', '')}
Bio: {profile.get('bio', '')}
Resume:
{profile.get('resume', '')}

Target roles: {', '.join(prefs.get('titles', []))}
Desired skills: {', '.join(prefs.get('desired_skills', []))}
Years of experience: {prefs.get('yoe', 0)}
Location preferences: remote_ok={prefs.get('location', {}).get('remote_ok', True)}, preferred={prefs.get('location', {}).get('preferred_locations', [])}"""
        if is_intern:
            profile_section += f"""
{_format_intern_compensation_line(compensation.get('monthly_stipend'))}"""
        else:
            min_salary = compensation.get("min_salary")
            if min_salary not in (None, "", 0):
                try:
                    profile_section += f"""
Minimum salary: ${int(min_salary):,}"""
                except (TypeError, ValueError):
                    profile_section += """
Minimum salary: Not set"""
            else:
                profile_section += """
Minimum salary: Not set"""

    # Append internship context when candidate is a student
    if is_intern:
        prof_cfg = config.get("profile", {})
        season_year = prof_cfg.get("target_season", "")
        school = prof_cfg.get("school", "")
        major  = prof_cfg.get("major", "")
        profile_section += f"""

IMPORTANT — INTERNSHIP CANDIDATE:
This candidate is a student targeting a {season_year} internship.
School: {school}  |  Major: {major}
They are NOT a full-time hire — score accordingly."""

    # Dimension instructions differ for interns vs full-time candidates
    if is_intern:
        seniority_desc    = "Does this posting explicitly target students, new grads, or interns? Score 10 if yes, 0 if the role is clearly for experienced hires only."
        compensation_desc = (
            "Does the posting clearly mention pay, a stipend, or an hourly rate? "
            "Score compensation according to the candidate's pay preference instead of assuming a minimum salary."
        )
        disqualifier_intern_rule = ""  # interns WANT internship postings — don't disqualify them
        if intern_pay_preference == "paid_only":
            compensation_desc += " Penalize unpaid postings or postings that strongly imply unpaid work."
        else:
            compensation_desc += " Do not penalize unpaid postings or postings with no pay details when the candidate is open to that."
    else:
        seniority_desc    = "How well does the seniority level match the candidate's experience?"
        compensation_desc = "How likely is the compensation to meet or exceed the candidate's minimum salary?"
        disqualifier_intern_rule = "\n- Is an internship or co-op position"

    prompt = f"""You are an expert technical recruiter evaluating a job posting for a candidate.

CANDIDATE PROFILE:
{profile_section}

--- JOB POSTING ---
{job.raw_text}

--- TASK ---
Step 1 — Hard disqualifier check.
Set "disqualified" to true and fill "disqualify_reason" if ANY of the following apply:
- Requires a security clearance (TS/SCI, government clearance, etc.)
- Explicitly requires on-site relocation with no remote option
- Requires a master's degree or PhD as a hard requirement (not just preferred)
- Requires more than {max_yoe} years of experience as a hard minimum{disqualifier_intern_rule}

Important interpretation notes:
- Do NOT treat "currently pursuing", "working toward", or "enrolled in" a bachelor's/master's degree as an advanced-degree disqualifier for internship candidates.
- Do NOT disqualify internship postings just because they mention students, graduation dates, or being in school.

If disqualified, set all dimension scores to 0 and skip Step 2.

Step 2 — Dimension scoring (only if NOT disqualified).
Score each dimension 0–10:
- role_fit:     How well does the job title and core responsibilities match the candidate's target roles and experience?
- stack_match:  How well do required/preferred technologies match the candidate's skills?
- seniority:    {seniority_desc}
- location:     How well does the job's location/remote policy match the candidate's preferences?
- growth:       How strong are the growth signals? (AI-native, early-stage, interesting domain)
- compensation: {compensation_desc}

Scoring rules:
- Score each dimension independently before arriving at a number. Do not let one dimension bias another.
- Required vs preferred matters — penalize missing required skills heavily, missing preferred skills lightly.
- A seniority mismatch on title alone is not a hard disqualifier if YOE and stack match well.
- Compensation below the candidate's minimum is a flag, not a disqualifier — it is often negotiable.
- You may only score based on information explicitly present in the job posting. Do not assume or invent details.
- reasons must contain exactly 2 to 4 short strings, never more than 4. If you have more than 4, combine the least important ones.

Return only valid JSON. No markdown fences. No preamble. No explanation.

{{
  "disqualified": false,
  "disqualify_reason": "",
  "role_fit":     0,
  "stack_match":  0,
  "seniority":    0,
  "location":     0,
  "growth":       0,
  "compensation": 0,
  "reasons":  ["exactly 2 to 4 short strings, never more than 4"],
  "flags":    ["<concern>"],
  "one_liner": "<one sentence summary of fit>"
}}"""

    tokens_used = 0
    dims = None

    # Structured output via instructor (preferred path)
    if instructor_client:
        provider  = config["llm"]["provider"]
        model_name = config["llm"]["model"][provider]
        logger.info(f"scorer | structured output via {provider}/{model_name}")
        dims = safe_structured_call(
            instructor_client, model_name, prompt, ScoreResult,
            max_tokens=700,
            temperature=config["llm"].get("temperature", 0),
            label="scorer",
        )
        if dims is not None:
            tokens_used = 500  # instructor doesn't expose real token counts

    # Fallback: raw LLM call + JSON parsing (also handles {"value": x} wrapping)
    if dims is None:
        try:
            raw, tokens_used = _llm_call_with_retry(llm_call, prompt, 700)
            dims = parse_llm_response(raw)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}")
            return zeroed
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise  # re-raise so score_all_jobs can record the error

    if dims.get("disqualified"):
        reason = dims.get("disqualify_reason", "unknown disqualifier")
        return {
            "disqualified": True,
            "disqualify_reason": reason,
            "fit_score": 0,
            "tokens_used": tokens_used,
            "dimension_scores": {
                "role_fit": 0, "stack_match": 0, "seniority": 0,
                "location": 0, "growth": 0, "compensation": 0,
            },
            "reasons": [],
            "flags": [f"disqualified: {reason}"],
            "one_liner": "",
        }

    # Weighted fit score computed in Python — not by the LLM
    try:
        fit_score = (
            dims["role_fit"]     * weights["role_fit"]     +
            dims["stack_match"]  * weights["stack_match"]  +
            dims["seniority"]    * weights["seniority"]    +
            dims["location"]     * weights["location"]     +
            dims["growth"]       * weights["growth"]       +
            dims["compensation"] * weights["compensation"]
        ) * 10
        fit_score = min(100, round(fit_score))
    except (KeyError, TypeError) as e:
        logger.warning(f"Fit score computation failed: {e}")
        fit_score = 0

    return {
        "disqualified": False,
        "disqualify_reason": "",
        "fit_score": fit_score,
        "tokens_used": tokens_used,
        "dimension_scores": {
            "role_fit":     dims.get("role_fit", 0),
            "stack_match":  dims.get("stack_match", 0),
            "seniority":    dims.get("seniority", 0),
            "location":     dims.get("location", 0),
            "growth":       dims.get("growth", 0),
            "compensation": dims.get("compensation", 0),
        },
        "reasons":   dims.get("reasons", []),
        "flags":     dims.get("flags", []),
        "one_liner": dims.get("one_liner", ""),
    }


# ── 3. ATS score ──────────────────────────────────────────────────────────────

def compute_ats_score(job: Job, config: Dict[str, Any]) -> dict:
    """
    Pure Python. Simulates keyword-based ATS matching.
    Measures overlap between job description words and resume text.
    """
    resume_text = (config["profile"].get("resume") or "").lower()
    job_text    = job.raw_text.lower()

    stopwords = {
        "and", "or", "the", "a", "an", "to", "of", "in",
        "for", "with", "is", "are", "we", "you", "your",
        "be", "as", "at", "by", "it", "its", "from", "this",
        "that", "have", "has", "will", "can", "our", "their",
    }

    jd_words = set(re.findall(r'\b[a-z][a-z0-9+#.\-]{2,}\b', job_text))
    jd_words -= stopwords

    matched       = {w for w in jd_words if w in resume_text}
    des_skills    = [s.lower() for s in config["preferences"].get("desired_skills", [])]
    skill_matches = [s for s in des_skills if s in job_text and s in resume_text]
    skill_misses  = [s for s in des_skills if s in job_text and s not in resume_text]

    base_score  = len(matched) / len(jd_words) if jd_words else 0
    skill_bonus = len(skill_matches) / len(des_skills) if des_skills else 0
    ats_score   = min(100, int((base_score * 0.6 + skill_bonus * 0.4) * 100))

    return {"ats_score": ats_score, "skill_misses": skill_misses[:5]}


# ── 4. Single-job orchestrator ────────────────────────────────────────────────

def score_job(
    job: Job,
    config: Dict[str, Any],
    llm_call: LlmCall,
    structured_profile: Optional[Dict] = None,
    instructor_client: Optional[Any] = None,
) -> dict:
    """
    Runs the full pipeline for one job. Returns a result dict.
    Raises on unrecoverable LLM errors (caller handles and records).
    """
    base = {
        "job": job,
        "fit_score": 0,
        "tokens_used": 0,
        "ats_score": 0,
        "reasons": [],
        "flags": [],
        "one_liner": "",
        "dimension_scores": {},
        "skill_misses": [],
    }

    # Stage 1: keyword pre-score — skip LLM call if no skill overlap
    if keyword_prescore(job, config) < 0.15:
        base["flags"] = ["no skill overlap"]
        return base

    # Stage 2: merged disqualifier + dimension scoring (single LLM call)
    result = score_dimensions(job, config, llm_call, structured_profile, instructor_client)
    base["fit_score"]        = result["fit_score"]
    base["tokens_used"]      = result["tokens_used"]
    base["reasons"]          = result["reasons"]
    base["flags"]            = result["flags"]
    base["one_liner"]        = result["one_liner"]
    base["dimension_scores"] = result["dimension_scores"]

    # Stage 3: ATS score — pure Python, always runs
    ats = compute_ats_score(job, config)
    base["ats_score"]    = ats["ats_score"]
    base["skill_misses"] = ats["skill_misses"]

    return base


# ── 5. Batch scorer ───────────────────────────────────────────────────────────

def _cost_estimate(provider: str, n_jobs: int) -> str:
    """Returns a human-readable cost estimate string."""
    if provider in _PAID_PROVIDERS:
        est = n_jobs * 0.005
        return f"~${est:.2f}"
    return "~$0.00 (free tier)"


def score_all_jobs(config: Dict[str, Any], yes: bool = False, profile: Optional[str] = None, on_job_scored=None) -> list:
    """
    Scores all eligible unscored jobs. Prompts for confirmation unless yes=True.
    Returns results sorted by fit score descending.
    Uses instructor for structured output when available.
    """
    llm_cfg  = config["llm"]
    provider = llm_cfg["provider"]
    model    = llm_cfg["model"][provider]

    llm_call = get_llm_client(config)
    
    # Try to get instructor client for structured output
    instructor_client = None
    try:
        instructor_client, _, _ = get_instructor_client(config)
    except Exception as e:
        logger.info(f"Instructor client unavailable ({provider} may not support it yet). Falling back to raw LLM with JSON parsing.")

    jobs = get_unscored(profile=profile)

    if not jobs:
        logger.info("No new jobs to score.")
        return []

    rate_config = config["llm"].get("rate_limits", {}).get(provider)
    if not rate_config:
        logger.warning(f"No rate limits configured for '{provider}' — using conservative defaults")
        rate_config = {"max_rpm": 10, "max_tpm": 50_000}

    cost_str = _cost_estimate(provider, len(jobs))
    rpd      = rate_config.get("max_rpd")
    rpd_str  = f" / {rpd:,} RPD" if rpd else ""
    logger.info(f"Provider: {provider} ({model})")
    logger.info(f"Rate limits: {rate_config['max_rpm']} RPM / {rate_config['max_tpm']:,} TPM{rpd_str}")
    if rpd:
        logger.info(f"Daily budget: {rpd:,} requests (resets midnight PT)")
    logger.info(f"Jobs to score: {len(jobs)}")
    logger.info(f"Estimated cost: {cost_str}")
    if provider in _PROVIDER_NOTES:
        logger.info(f"Note: {_PROVIDER_NOTES[provider](len(jobs))}")
    if instructor_client:
        logger.info("Using instructor for structured output (max_retries=3)")

    limiter = RateLimiter(
        max_rpm=rate_config["max_rpm"],
        max_tpm=rate_config["max_tpm"],
        max_rpd=rate_config.get("max_rpd"),
    )

    # Build structured profile once — counts as 1 API call toward rate limiter
    logger.info("Building structured profile from resume...")
    if instructor_client:
        structured_profile = build_structured_profile(
            config, llm_call, instructor_client, 
            model=model, temperature=config["llm"].get("temperature", 0)
        )
    else:
        structured_profile = build_structured_profile(config, llm_call)
    limiter.record(600)
    print_profile_summary(structured_profile)
    logger.info(
        f"Profile: {structured_profile.get('name')} | "
        f"{structured_profile.get('yoe')}yr | "
        f"{len(structured_profile.get('core_skills', []))} skills extracted"
    )

    if not yes:
        if not confirm_profile():
            logger.info("Aborted — update resume PDF or bio in config.yaml and re-run.")
            sys.exit(0)

    logger.info("Note: run `python scorer.py --rescore` to clear old scores and re-score.")

    if not yes:
        try:
            confirm = input("Proceed with scoring? [y/n]: ").strip().lower()
        except EOFError:
            confirm = "y"  # non-interactive context — proceed automatically
        if confirm != "y":
            logger.info("Aborted.")
            return []

    results = []

    for i, job in enumerate(jobs, 1):
        limiter.wait_if_needed()
        logger.debug(f"[{i}/{len(jobs)}] Scoring: {job.company} — {job.title}")

        increment_score_attempts(job.id, profile=profile)

        try:
            result = score_job(job, config, llm_call, structured_profile, instructor_client)
            limiter.record(result.get("tokens_used", 500))

            dims = result["dimension_scores"]
            save_score(
                job_id=job.id,
                fit_score=result["fit_score"],
                role_fit=dims.get("role_fit", 0),
                stack_match=dims.get("stack_match", 0),
                seniority=dims.get("seniority", 0),
                loc_score=dims.get("location", 0),
                growth=dims.get("growth", 0),
                compensation=dims.get("compensation", 0),
                reasons=json.dumps(result["reasons"]),
                flags=json.dumps(result["flags"]),
                skill_misses=json.dumps(result["skill_misses"]),
                one_liner=result["one_liner"],
                ats_score=result["ats_score"],
                disqualified=1 if result.get("disqualified") else 0,
                disqualify_reason=result.get("disqualify_reason", "") or "",
                profile=profile,
            )

            results.append(result)
            if on_job_scored is not None:
                try:
                    on_job_scored(i, len(jobs), result)
                except Exception:
                    pass
            liner = result["one_liner"][:60] if result["one_liner"] else "(skipped)"
            logger.debug(f"    fit={result['fit_score']}  ats={result['ats_score']}  {liner}")

        except RateLimitReached:
            raise  # propagate — don't treat as a per-job error
        except Exception as e:
            write_score_error(job.id, str(e), profile=profile)
            limiter.record(500)  # count failed attempts toward rate limit
            logger.error(f"Scoring failed for job {job.id}: {e}")

    scored = [r for r in results if r["fit_score"] > 0]
    if scored:
        avg_fit = sum(r["fit_score"] for r in scored) / len(scored)
        avg_ats = sum(r["ats_score"] for r in scored) / len(scored)
        logger.info(f"Done. {len(scored)}/{len(jobs)} scored  |  avg fit={avg_fit:.0f}  avg ats={avg_ats:.0f}")

    return sorted(results, key=lambda r: r["fit_score"], reverse=True)


# ── 6. Display ────────────────────────────────────────────────────────────────

def _score_bar(score: int, width: int = 14) -> str:
    filled = round(score / 100 * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def print_results(results: list, config: Dict[str, Any]):
    """Print ranked job results to terminal."""
    min_score = config["scoring"].get("min_display_score", 60)
    visible   = [r for r in results if r["fit_score"] >= min_score]

    if not visible:
        logger.info(f"No jobs scored above {min_score}. Lower min_display_score in config.yaml to see more.")
        return

    print(f"\n{'='*60}")
    print(f"  TOP JOBS  (showing {len(visible)} above score {min_score})")
    print(f"{'='*60}\n")

    for r in visible:
        job = r["job"]
        fit = r["fit_score"]
        ats = r["ats_score"]

        if fit >= 85:
            dot = "\U0001f7e2"
        elif fit >= 70:
            dot = "\U0001f7e1"
        elif fit >= 55:
            dot = "\U0001f7e0"
        else:
            dot = "\U0001f534"

        print(f"{dot} {job.title} \u2014 {job.company} ({job.location})")
        print(f"   Fit: {fit}/100  {_score_bar(fit)}  |  ATS: {ats}/100  {_score_bar(ats)}")

        if r["one_liner"]:
            print(f"   \u2192 {r['one_liner']}")

        if r["reasons"]:
            print(f"   \u2713 " + " \u00b7 ".join(r["reasons"]))

        if r["flags"]:
            print(f"   \u26a0 " + " \u00b7 ".join(r["flags"]))

        if r["skill_misses"]:
            misses = ", ".join(f'"{s}"' for s in r["skill_misses"])
            print(f"   \u2717 Missing from resume: {misses}")

        print(f"   {job.url}")
        print()


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from logging_config import configure_logging
    configure_logging(profile="default", debug="--debug" in sys.argv)

    if "--rescore" in sys.argv:
        logger.info("--rescore: clearing scores table and resetting score_attempts...")
        rescore_reset()

    config = load_config()
    init_db()
    results = score_all_jobs(config, yes="--yes" in sys.argv)
    print_results(results, config)
