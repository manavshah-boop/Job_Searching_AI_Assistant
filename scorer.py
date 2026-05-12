"""
scorer.py — Step 4: Score unscored jobs using a multi-stage pipeline.

Pipeline per job:
  1. keyword_prescore()  — pure Python, no API call
  2. score_dimensions()  — single LLM call (disqualifier check + dimension scoring merged)
  3. compute_ats_score() — pure Python, no API call

Active provider and model are set in config.yaml under the `llm` key.
No code changes needed to switch providers.
"""

import copy
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
    log_routing_decision,
    rescore_reset,
    save_score,
    write_score_error,
)
from candidate_profile import build_structured_profile, confirm_profile, print_profile_summary
from llm_utils import estimate_tokens, is_rate_limit_error, parse_llm_response, safe_structured_call
from models import ScoreResult, StructuredProfile

class RateLimitReached(Exception):
    """Raised when the provider's daily RPD limit is exhausted."""


class ProviderConfigError(RuntimeError):
    """
    Raised when a provider cannot be initialized (missing API key, unknown
    provider, missing SDK). Distinct from rate limits so the fallback layer
    can drop this provider for the rest of the run instead of retrying.
    """


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

def _require_api_key(env_var: str, profile: Optional[str]) -> str:
    """
    Resolve an API key or raise ProviderConfigError.

    Replaces the previous sys.exit(1) on missing keys so the fallback layer
    can catch and drop the provider instead of killing the worker. The primary
    provider check at startup happens earlier (main._check_api_key); by the
    time we hit this code path we are usually building a *fallback* client
    where a missing key just means "skip this provider, try the next one".
    """
    api_key = _resolve_api_key(env_var, profile)
    if api_key:
        return api_key
    names = (
        f"{env_var}_{profile.upper()} or {env_var}"
        if profile else env_var
    )
    raise ProviderConfigError(f"{names} not set in environment or .env")


def get_llm_client(config: Dict[str, Any]) -> LlmCall:
    """
    Returns a callable: (prompt: str, max_tokens: int) -> str

    The same interface regardless of provider — the rest of the scorer
    never knows which backend it's talking to.

    SDKs are imported lazily inside each branch so a missing SDK only
    fails if that provider is actually selected. Missing keys / SDKs /
    unknown providers raise ProviderConfigError (not sys.exit) so callers
    that maintain a fallback chain can drop this provider and keep going.
    """
    provider    = config["llm"]["provider"]
    models      = config["llm"]["model"]
    temperature = config["llm"].get("temperature", 0)
    profile     = config.get("_active_profile")

    if provider == "anthropic":
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderConfigError(f"anthropic SDK not installed: {exc}") from exc
        api_key = _require_api_key("ANTHROPIC_API_KEY", profile)
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
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as exc:
            raise ProviderConfigError(f"google-genai SDK not installed: {exc}") from exc
        api_key = _require_api_key("GEMINI_API_KEY", profile)
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
        try:
            from groq import Groq
        except ImportError as exc:
            raise ProviderConfigError(f"groq SDK not installed: {exc}") from exc
        api_key = _require_api_key("GROQ_API_KEY", profile)
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
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderConfigError(f"openai SDK not installed: {exc}") from exc
        api_key = _require_api_key("OPENAI_API_KEY", profile)
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

    elif provider == "cerebras":
        # Cerebras ships an OpenAI-compatible REST API. Prefer the official
        # cerebras-cloud-sdk when available; fall back to the OpenAI SDK
        # pointed at api.cerebras.ai so the fallback chain works without
        # requiring an extra package install.
        api_key = _require_api_key("CEREBRAS_API_KEY", profile)
        cerebras_model = models.get("cerebras") or "llama-3.3-70b"
        try:
            from cerebras.cloud.sdk import Cerebras  # type: ignore[import-not-found]
            client = Cerebras(api_key=api_key)

            def call_cerebras(prompt: str, max_tokens: int = 700) -> Tuple[str, int]:
                response = client.chat.completions.create(
                    model=cerebras_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                usage = getattr(response, "usage", None)
                tokens = getattr(usage, "total_tokens", 500) if usage else 500
                return response.choices[0].message.content or "", tokens

            return call_cerebras
        except ImportError:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ProviderConfigError(
                    "Neither cerebras-cloud-sdk nor openai is installed; "
                    "install one of them to use the cerebras provider"
                ) from exc
            client_oa = OpenAI(api_key=api_key, base_url="https://api.cerebras.ai/v1")

            def call_cerebras_openai(prompt: str, max_tokens: int = 700) -> Tuple[str, int]:
                response = client_oa.chat.completions.create(
                    model=cerebras_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content or "", response.usage.total_tokens

            return call_cerebras_openai

    elif provider == "openrouter":
        # OpenRouter is OpenAI-compatible, served at https://openrouter.ai/api/v1.
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderConfigError(f"openai SDK not installed: {exc}") from exc
        api_key = _require_api_key("OPENROUTER_API_KEY", profile)
        openrouter_model = models.get("openrouter") or "openrouter/auto"
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

        def call_openrouter(prompt: str, max_tokens: int = 700) -> Tuple[str, int]:
            response = client.chat.completions.create(
                model=openrouter_model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = getattr(response, "usage", None)
            tokens = getattr(usage, "total_tokens", 500) if usage else 500
            return response.choices[0].message.content or "", tokens

        return call_openrouter

    elif provider == "mistral":
        try:
            from mistralai import Mistral  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ProviderConfigError(f"mistralai SDK not installed: {exc}") from exc
        api_key = _require_api_key("MISTRAL_API_KEY", profile)
        mistral_model = models.get("mistral") or "mistral-small-latest"
        client = Mistral(api_key=api_key)

        def call_mistral(prompt: str, max_tokens: int = 700) -> Tuple[str, int]:
            response = client.chat.complete(
                model=mistral_model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = getattr(response, "usage", None)
            tokens = getattr(usage, "total_tokens", 500) if usage else 500
            return response.choices[0].message.content or "", tokens

        return call_mistral

    raise ProviderConfigError(
        f"Unknown provider {provider!r}. "
        "Options: anthropic, gemini, groq, openai, cerebras, openrouter, mistral"
    )


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
        try:
            import anthropic
            import instructor
        except ImportError as exc:
            raise ProviderConfigError(f"anthropic/instructor SDK not installed: {exc}") from exc
        api_key = _require_api_key("ANTHROPIC_API_KEY", profile)
        client = anthropic.Anthropic(api_key=api_key)
        client = instructor.from_anthropic(client)
        return client, models["anthropic"], temperature

    elif provider == "gemini":
        try:
            from google import genai
        except ImportError as exc:
            raise ProviderConfigError(f"google-genai SDK not installed: {exc}") from exc
        api_key = _require_api_key("GEMINI_API_KEY", profile)
        _ = genai.Client(api_key=api_key)  # validate key/SDK
        # Gemini doesn't have direct instructor support yet, so we use raw client
        # Fall back to get_llm_client for gemini
        return None, models["gemini"], temperature

    elif provider == "groq":
        try:
            from groq import Groq
            import instructor
        except ImportError as exc:
            raise ProviderConfigError(f"groq/instructor SDK not installed: {exc}") from exc
        api_key = _require_api_key("GROQ_API_KEY", profile)
        client = Groq(api_key=api_key)
        client = instructor.from_groq(client)
        return client, models["groq"], temperature

    elif provider == "openai":
        try:
            from openai import OpenAI
            import instructor
        except ImportError as exc:
            raise ProviderConfigError(f"openai/instructor SDK not installed: {exc}") from exc
        api_key = _require_api_key("OPENAI_API_KEY", profile)
        client = OpenAI(api_key=api_key)
        client = instructor.from_openai(client)
        return client, models["openai"], temperature

    # Cerebras / OpenRouter / Mistral are reached only via the fallback chain;
    # they don't support instructor's structured-output mode yet, so we return
    # (None, model, temperature) and the caller falls through to the raw LLM
    # path with JSON parsing (parse_llm_response handles {"value": x} wrapping).
    elif provider in ("cerebras", "openrouter", "mistral"):
        return None, models.get(provider, ""), temperature

    raise ProviderConfigError(
        f"Unknown provider {provider!r}. "
        "Options: anthropic, gemini, groq, openai, cerebras, openrouter, mistral"
    )


# ── 1. Keyword pre-score ──────────────────────────────────────────────────────

def keyword_prescore(job: Job, config: Dict[str, Any]) -> float:
    """
    Pure Python. Fraction of desired_skills present in raw_text.
    Returns 0.0–1.0. Below 0.15 → skip the LLM call entirely.

    Exception: if the job title is an exact or close normalized variant of any
    preferred title in preferences.titles, return 1.0 to guarantee LLM scoring.
    A user who explicitly listed a title wants close variants of it scored too.
    """
    def _normalize_title(title: str) -> str:
        # Collapse punctuation and spacing so title variants compare cleanly.
        return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()

    preferred_titles = [t.lower() for t in config.get("preferences", {}).get("titles", [])]
    if preferred_titles:
        job_title = job.title.lower()
        normalized_job_title = _normalize_title(job.title)
        for preferred_title in preferred_titles:
            if job_title == preferred_title:
                return 1.0

            normalized_preferred_title = _normalize_title(preferred_title)
            if normalized_preferred_title and normalized_preferred_title in normalized_job_title:
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
    max_tokens: int = 700,
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
    # Used in the disqualifier prompt — the threshold at which a hard YOE
    # minimum disqualifies. Example: max_yoe=4 → only "5+ years required"
    # (or higher) triggers a YOE disqualifier.
    max_yoe_plus_one = int(max_yoe) + 1

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
Step 1 — Hard disqualifier check. Default is disqualified=false. Only set disqualified=true if a verbatim phrase from the lists below appears in the JD. Do not paraphrase, infer, or generalize. If you cannot quote the exact disqualifying phrase from the JD, disqualified MUST be false.

Set "disqualified" to true and fill "disqualify_reason" ONLY if you can quote a verbatim match for one of:

- **Security clearance**: the JD literally contains one of "security clearance", "TS/SCI", "Top Secret", "government clearance", "DoD clearance", "ITAR", "polygraph", "active clearance". Words like "government", "federal", "defense" ALONE are NOT a disqualifier.

- **No-remote / required relocation**: the JD literally contains one of "no remote", "not remote", "fully on-site only", "must be on-site", "must be onsite", "must be in office", "must be in-office", "relocation required", "this role is not remote", "remote not available", "must relocate". A JD that lists an office location but doesn't forbid remote is NOT a disqualifier.

- **Graduate degree required (not preferred)**: the JD literally contains one of "MS required", "M.S. required", "Master's required", "Master's degree required", "PhD required", "Ph.D. required", "doctorate required", "graduate degree required", "advanced degree required", "must have a Master's", "must have a PhD". JDs that say "MS preferred", "PhD a plus", "Master's is nice to have" are NOT disqualifiers.

- **YOE > {max_yoe} years as a hard minimum**: the JD literally contains a phrase like "{max_yoe_plus_one}+ years required", "minimum {max_yoe_plus_one} years of experience", "at least {max_yoe_plus_one} years required". JDs that say "{max_yoe_plus_one}+ years preferred", "ideally {max_yoe_plus_one} years", "{max_yoe_plus_one}+ years a plus" are NOT disqualifiers. JDs that give a range starting at or below {max_yoe} (e.g. "2-5 years") are NOT disqualifiers.{disqualifier_intern_rule}

Important interpretation notes:
- Do NOT treat "currently pursuing", "working toward", or "enrolled in" a bachelor's/master's degree as an advanced-degree disqualifier for internship candidates.
- Do NOT disqualify internship postings just because they mention students, graduation dates, or being in school.
- Do NOT infer disqualifiers from the company sector, the team name, or your prior knowledge of any organization — only the JD text matters.
- When unsure, prefer disqualified=false. A bad disqualification wastes a real candidate; a missed disqualification is a small annoyance.
- If you set disqualified=true, your disqualify_reason MUST quote the verbatim phrase from the JD that triggered it. Format: 'Reason: "<quoted JD text>"'.

If disqualified, set all dimension scores to 0 and skip Step 2.

Step 2 — Dimension scoring (only if NOT disqualified).
Score each dimension 0–10:
- role_fit:     How well does the job title and core responsibilities match the candidate's target roles and experience?
- stack_match:  How well do required/preferred technologies match the candidate's skills?
- seniority:    {seniority_desc}
- location:     How well does the job's location/remote policy match the candidate's preferences? See location scoring rules below.
- growth:       How strong are the growth signals? (AI-native, early-stage, interesting domain)
- compensation: {compensation_desc} See compensation scoring rules below.

Scoring rules:
- Score each dimension independently before arriving at a number. Do not let one dimension bias another.
- Required vs preferred matters — penalize missing required skills heavily, missing preferred skills lightly.
- A seniority mismatch on title alone is not a hard disqualifier if YOE and stack match well.
- Compensation below the candidate's minimum is a flag, not a disqualifier — it is often negotiable.
- You may only score based on information explicitly present in the job posting. Do not assume or invent details.
- reasons must contain exactly 2 to 4 short strings, never more than 4. If you have more than 4, combine the least important ones.

CALIBRATION ANCHORS — use the FULL 0–10 range per dimension and AVOID bimodal output.
Past runs showed scores clustering at either 0 (false disqualifiers) or 8–9 (everything that passed). That is incorrect. The correct distribution is graded:
- 9–10 : near-perfect fit on this dimension. RARE. Reserve for "every required + preferred element present, no caveats".
- 7–8  : strong fit. Most reasonable matches land here. Some preferred items missing is OK.
- 5–6  : partial / ambiguous fit. Some signals present, others missing. Default when uncertain.
- 3–4  : weak fit. Most signals missing, but not a categorical mismatch.
- 1–2  : poor fit on this specific dimension.
- 0    : ONLY when this dimension is categorically irrelevant (e.g. role_fit for a non-engineering role being scored against an engineer).

Composite fit_score guidance (the system computes this from your dimension scores; use it as a sanity check):
- 90–100 : perfect / aspirational match. Almost never appears. Reserve for "exactly the right title, every skill matches, in-budget, in preferred location, top-tier company".
- 70–85  : strong match — most decent-fit roles for an in-target candidate should land here.
- 50–70  : decent / partial match. The candidate could reasonably apply but with caveats.
- 30–50  : marginal match. Some overlap exists but key elements are missing.
- 0–30   : not a match. Only score this low when most dimensions are genuinely poor.

CRITICAL: A target-title role at a real company in the candidate's region with overlapping stack SHOULD score 60+ even if some preferred skills are missing. Do not default to 50 just because "some things are missing" — most jobs miss some things. Score what IS there, not what isn't.

Location scoring rules:
- A job's office is in one of the candidate's preferred_locations (city, state, or region) → score 7–10. Treat US states broadly: e.g. San Francisco, Palo Alto, Mountain View all satisfy "California"; Brooklyn satisfies "New York"; Bellevue/Redmond satisfy "Seattle, WA" or "Washington".
- The job offers remote and remote_ok=True → score 8–10.
- Onsite-only in a preferred location is NOT a penalty — score the same as remote in a preferred location.
- Onsite-only outside any preferred location → score 2–4.
- Hybrid in a preferred location → score 7–9.
- Only score 0–2 when the job explicitly forbids the candidate's region (e.g. EU-only, requires relocation to a country the candidate is not eligible for).

Compensation scoring rules:
- The job posting clearly states a salary/comp range that meets or exceeds the candidate's minimum → score 8–10.
- The salary range is below the candidate's minimum → score 2–4 and flag it.
- The job posting does NOT mention compensation at all → score 5 (neutral / unknown). Do NOT penalize for missing data — top-tier US tech companies routinely pay above minimum even when the range is unstated. Add a brief note in flags such as "Compensation not stated".
- The posting hints at equity/perks but no base salary → score 5 (neutral) and note in flags.

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
            max_tokens=max_tokens,
            temperature=config["llm"].get("temperature", 0),
            label="scorer",
        )
        if dims is not None:
            tokens_used = max(500, len(prompt) // 4)

    # Fallback: raw LLM call + JSON parsing (also handles {"value": x} wrapping)
    if dims is None:
        try:
            raw, tokens_used = _llm_call_with_retry(llm_call, prompt, max_tokens)
            dims = parse_llm_response(raw)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}")
            return zeroed
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise  # re-raise so score_all_jobs can record the error

    # ── Disqualifier hallucination guards ─────────────────────────────────────
    # The 2026-05-10 run showed the cheap fast-tier models (llama-3.1-8b-instant
    # in particular) hallucinating disqualifiers on 22 of 27 LLM-scored jobs:
    #   - "Requires a security clearance" on JDs with zero clearance language
    #   - "Requires more than 4 years of experience" on JDs that say
    #     "2-4 years preferred" or list YOE in a "nice to have" section
    #   - "Requires a master's degree" on JDs that say "MS preferred"
    #   - "Requires on-site relocation with no remote option" on remote-friendly JDs
    #
    # Each guard below requires the JD to literally contain the disqualifying
    # phrase before we trust the LLM. When a guard fires, dims["disqualified"]
    # is flipped to False AND the LLM-returned zero dimensions are discarded —
    # we don't trust scores produced by an LLM that misread the JD. The caller
    # gets a neutral fallback dimension set (see _build_guarded_fallback below)
    # which lands near the synthetic-score range, signalling "AI flagged this
    # for review, but the flag wasn't supported by the JD — verify manually."
    guard_fired_reason: Optional[str] = None
    if dims.get("disqualified"):
        original_reason = str(dims.get("disqualify_reason", "unknown disqualifier") or "")
        job_text_lower = (job.raw_text or "").lower()
        reason_l = original_reason.lower()

        # Guard 1: security clearance
        clearance_phrases = (
            "security clearance", "ts/sci", "top secret", "government clearance",
            "dod clearance", "itar", "polygraph", "active clearance",
        )
        if "clearance" in reason_l or "ts/sci" in reason_l or "polygraph" in reason_l:
            if not any(p in job_text_lower for p in clearance_phrases):
                guard_fired_reason = (
                    f"clearance-hallucination (LLM claimed '{original_reason}' "
                    "but JD has no clearance keywords)"
                )

        # Guard 2: YOE — only disqualify if the JD literally specifies a hard
        # minimum exceeding max_yoe. We accept formats like "5+ years", "5 or more
        # years", "minimum 5 years", "at least 5 years of". "Preferred"/"plus"
        # language nearby is NOT a hard minimum.
        if guard_fired_reason is None and (
            "year" in reason_l and ("experience" in reason_l or "minimum" in reason_l or "yoe" in reason_l)
        ):
            yoe_threshold = int(max_yoe) + 1  # any "X+ years" where X > max_yoe disqualifies
            yoe_patterns = [
                rf"\b{yoe_threshold}\s*\+\s*years?\b",
                rf"\b{yoe_threshold}\s*or more\s*years?\b",
                rf"minimum\s+(?:of\s+)?{yoe_threshold}\s+years?\b",
                rf"at\s+least\s+{yoe_threshold}\s+years?\b",
                rf"\b{yoe_threshold}\s*-\s*\d+\s+years?\s+(?:of\s+)?(?:required|experience\s+required)\b",
            ]
            # Catch-all: ANY "N+ years of experience" where N > max_yoe disqualifies.
            # Previously this was a hard-coded \b([5-9]|[1-9]\d)\+ pattern, which
            # silently failed for intern profiles (max_yoe=2) where "3+ years"
            # should disqualify but the regex required the leading digit to be 5–9.
            # Sweep ALL "N+ years" matches and let the >max_yoe check filter them.
            any_high_yoe = None
            for match in re.finditer(
                r"\b(\d{1,2})\s*\+\s*years?\s+(?:of\s+)?(?:experience|exp)\b",
                job_text_lower,
            ):
                try:
                    found = int(match.group(1))
                except (ValueError, IndexError):
                    continue
                if found > max_yoe:
                    any_high_yoe = match
                    break

            literal_yoe_disqualifier = (
                any(re.search(p, job_text_lower) for p in yoe_patterns)
                or any_high_yoe is not None
            )
            if not literal_yoe_disqualifier:
                guard_fired_reason = (
                    f"yoe-hallucination (LLM claimed '{original_reason}' "
                    f"but JD does not literally require >{max_yoe} years)"
                )

        # Guard 3: advanced-degree requirements. The system prompt explicitly
        # tells the LLM not to disqualify on "MS preferred" or "PhD a plus",
        # but small models mishandle this regularly.
        if guard_fired_reason is None and (
            "master" in reason_l or "phd" in reason_l or "doctorate" in reason_l
            or "graduate degree" in reason_l
        ):
            hard_degree_phrases = (
                "ms required", "m.s. required", "master's required", "masters required",
                "master's degree required", "masters degree required",
                "phd required", "ph.d. required", "doctorate required",
                "graduate degree required", "advanced degree required",
                "must have a master", "must have a phd", "must have an ms",
                "must have a doctorate",
            )
            if not any(p in job_text_lower for p in hard_degree_phrases):
                guard_fired_reason = (
                    f"degree-hallucination (LLM claimed '{original_reason}' "
                    "but JD does not literally require a graduate degree)"
                )

        # Guard 4: on-site / no-remote. The system prompt's strict triggers are
        # "no remote", "must be on-site only", "relocation required". A literal
        # match is required.
        if guard_fired_reason is None and (
            "on-site" in reason_l or "onsite" in reason_l or "remote" in reason_l
            or "relocation" in reason_l
        ):
            hard_onsite_phrases = (
                "no remote", "not remote", "fully on-site only", "fully onsite only",
                "must be on-site", "must be onsite", "must be in office",
                "must be in-office", "relocation required", "relocation is required",
                "this role is not remote", "remote not available",
                "must relocate", "required to relocate",
            )
            if not any(p in job_text_lower for p in hard_onsite_phrases):
                guard_fired_reason = (
                    f"onsite-hallucination (LLM claimed '{original_reason}' "
                    "but JD does not literally forbid remote / require relocation)"
                )

        if guard_fired_reason is not None:
            logger.warning(
                "scorer | disqualifier guard fired for job={} — {}; falling back "
                "to neutral dimension estimate instead of trusting LLM zeros",
                job.id, guard_fired_reason,
            )
            dims["disqualified"] = False
            dims["disqualify_reason"] = ""
            # Discard the LLM's dimension scores — when it misread the JD enough
            # to invent a disqualifier, its dimension scoring is also untrustworthy.
            dims["__guard_fired"] = True
            dims["__guard_note"] = guard_fired_reason

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
            "reasons": [f"Hard disqualifier: {reason}"],
            "flags": [f"Disqualified — {reason}"],
            "one_liner": f"Disqualified: {reason}",
        }

    # If a hallucination guard fired, override the LLM's (likely zeroed)
    # dimension scores with a neutral estimate so the job isn't silently
    # buried at fit_score=0. The estimate uses simple heuristics that don't
    # need another LLM call:
    #   - role_fit  : 7 if title contains any preferred title, else 5
    #   - stack     : 5 + (1 per skill overlap, capped at 9)
    #   - others    : neutral 5
    if dims.get("__guard_fired"):
        target_titles = [t.lower() for t in prefs.get("titles", [])]
        job_title_l = (job.title or "").lower()
        title_hit = any(t in job_title_l for t in target_titles if t)

        skills = prefs.get("desired_skills", [])
        job_text_l = (job.raw_text or "").lower()
        skill_hits = sum(1 for s in skills if s.lower() in job_text_l)

        fallback = {
            "role_fit":     7 if title_hit else 5,
            "stack_match":  max(5, min(9, 5 + skill_hits)),
            "seniority":    5,
            "location":     5,
            "growth":       5,
            "compensation": 5,
        }
        dims.update(fallback)
        dims["reasons"] = [
            "AI-flagged disqualifier was not supported by the job description",
            "Dimension scores fall back to a neutral estimate — please verify manually",
        ]
        dims["flags"] = [
            "Disqualifier override: AI misread JD — manual review recommended",
        ]
        dims["one_liner"] = (
            "AI flagged a disqualifier the JD doesn't support; showing a "
            "neutral fallback estimate — verify manually."
        )

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

def compute_ats_score(
    job: Job,
    config: Dict[str, Any],
    structured_profile: Optional[Dict] = None,
) -> dict:
    """
    ATS keyword matching.

    When structured_profile is available (LLM-extracted skills), counts how many
    of the candidate's concrete skills appear in the job description. This produces
    realistic 20–60% scores for genuine matches instead of single-digit values.

    Falls back to word-level resume matching when no structured profile is given.
    """
    from skill_extraction import extract_skills, text_has_skill

    job_text  = job.raw_text or ""
    job_lower = job_text.lower()
    des_skills = [s for s in config["preferences"].get("desired_skills", [])]

    if structured_profile:
        # Build the candidate's canonical skill set (deduped, lowercased).
        candidate_set: set[str] = set()
        for bucket in ("core_skills", "languages", "frameworks", "cloud"):
            for item in structured_profile.get(bucket, []) or []:
                k = str(item).strip().lower()
                if k:
                    candidate_set.add(k)

        if not candidate_set:
            return {"ats_score": 0, "skill_misses": []}

        # Run the same extractor over the JD to find canonical skills it asks for.
        # This makes the ATS metric "what fraction of skills the JD wants does the
        # candidate have?" — independent of how rich the candidate's profile is.
        jd_extracted = extract_skills(job_text)
        jd_canon: list[str] = (
            jd_extracted["languages"] + jd_extracted["frameworks"]
            + jd_extracted["cloud"] + jd_extracted["databases"]
            + jd_extracted["concepts"]
        )

        if not jd_canon:
            # JD has no extractable skill tokens — fall back to coverage of the
            # configured desired_skills list so we still produce a meaningful number.
            wanted = [s for s in des_skills if text_has_skill(job_text, s)]
            covered = [s for s in wanted if text_has_skill(" ".join(candidate_set), s)]
            ats_score = min(100, int(len(covered) / max(1, len(wanted) or 1) * 100)) if wanted else 0
        else:
            covered = [s for s in jd_canon if s.lower() in candidate_set]
            ats_score = min(100, int(len(covered) / max(1, len(jd_canon)) * 100))

        # Skills the JD wants that the candidate doesn't have (canonical view).
        skill_misses = [
            s for s in (jd_canon if jd_canon else des_skills)
            if (s.lower() not in candidate_set if jd_canon
                else text_has_skill(job_text, s) and not text_has_skill(" ".join(candidate_set), s))
        ][:5]
        return {"ats_score": ats_score, "skill_misses": skill_misses}

    # Fallback: word-level overlap between job description and raw resume text.
    # Defensive .get on "profile" — tests construct configs without it, and the
    # synthetic-score path may call this before the dashboard has populated one.
    resume_text = (config.get("profile", {}).get("resume") or "").lower()
    job_text = job_lower

    # Last-resort path: no structured profile AND no resume. Score against the
    # candidate's desired_skills list using canonical text_has_skill so that
    # multi-word configured phrases (e.g. "Machine Learning") match JD aliases
    # like "ML" or "machine-learning". Without this branch ATS would silently
    # be 0 whenever no resume is loaded.
    if not resume_text:
        if not des_skills:
            return {"ats_score": 0, "skill_misses": []}
        present = [s for s in des_skills if text_has_skill(job_text, s)]
        ats_score = min(100, int(len(present) / max(1, len(des_skills)) * 100))
        # No "skill_misses" notion here — we don't know what the candidate has.
        return {"ats_score": ats_score, "skill_misses": []}

    stopwords = {
        "and", "or", "the", "a", "an", "to", "of", "in",
        "for", "with", "is", "are", "we", "you", "your",
        "be", "as", "at", "by", "it", "its", "from", "this",
        "that", "have", "has", "will", "can", "our", "their",
    }

    jd_words = set(re.findall(r'\b[a-z][a-z0-9+#.\-]{2,}\b', job_text))
    jd_words -= stopwords

    matched       = {w for w in jd_words if w in resume_text}
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
    max_tokens: int = 700,
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

    # Stage 0: scrape-filter gate — job was admitted to DB for dedup but rejected at scrape time
    if job.scrape_qualified == 0:
        base["flags"] = [f"scrape_filtered: {job.scrape_filter_reason}"]
        return base

    # Stage 1: keyword pre-score — only meaningful when selective routing is OFF.
    # When routing is enabled, the cross-encoder + title/skills boosts already make
    # a much better LLM-vs-skip decision, and this gate uses naive `s.lower() in text`
    # substring matching that mis-fires on multi-word desired_skills phrases like
    # "Machine learning infrastructure" — silently zeroing genuinely good matches
    # the router has already approved (CRIT-3 in the audit).
    if not config.get("routing", {}).get("enabled", False):
        if keyword_prescore(job, config) < 0.15:
            base["flags"] = ["no skill overlap"]
            return base

    # Stage 2: merged disqualifier + dimension scoring (single LLM call)
    result = score_dimensions(
        job, config, llm_call, structured_profile, instructor_client, max_tokens=max_tokens
    )
    base["fit_score"]        = result["fit_score"]
    base["tokens_used"]      = result["tokens_used"]
    base["reasons"]          = result["reasons"]
    base["flags"]            = result["flags"]
    base["one_liner"]        = result["one_liner"]
    base["dimension_scores"] = result["dimension_scores"]

    # Stage 3: ATS score — structured_profile skills give realistic 20-60% scores
    ats = compute_ats_score(job, config, structured_profile)
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
    When routing.enabled=true, skips the LLM for jobs below the routing threshold.
    """
    # ── Selective routing setup (must run before get_llm_client to apply model override) ──
    router = None
    _match_query = None
    _fast_config: Dict[str, Any] = config
    _quality_config: Dict[str, Any] = config
    _fast_llm_call: Optional[LlmCall] = None
    _quality_llm_call: Optional[LlmCall] = None
    _provider_budget = None
    _llm_client_cache: Dict[str, LlmCall] = {}
    _primary_provider = config.get("llm", {}).get("provider", "groq")

    if config.get("routing", {}).get("enabled", False):
        from selective_routing import SelectiveRouter, ProviderBudget
        router = SelectiveRouter(config, profile)
        config = router.apply_model_override(config)
        _match_query = router.build_match_query()
        logger.info(
            "Selective routing enabled | threshold={:.2f} | quality_mode={} | model={}",
            router.threshold, router.quality_mode, router.get_effective_llm_model(),
        )

        # Initialize per-run provider budget when rate_limit_budget is configured.
        # When fallback_chain is configured WITHOUT rate_limit_budget (which
        # silently disabled the fallback in the 2026-05-11 run), build a
        # permissive default so the fallback chain is still wired up. Without
        # this, every 429 from the primary provider kills the job instead of
        # rolling to the next provider.
        budget_cfg = config.get("routing", {}).get("rate_limit_budget")
        fallback_chain = config.get("routing", {}).get("fallback_chain") or []
        if not budget_cfg and fallback_chain:
            budget_cfg = {
                p: {"max_requests_per_run": 999_999, "max_tokens_per_run": 999_999_999}
                for p in fallback_chain
            }
            logger.warning(
                "routing | fallback_chain set without rate_limit_budget — "
                "using permissive defaults so 429s roll over to the next provider. "
                "Add a rate_limit_budget block to cap per-provider spend.",
            )
        if budget_cfg:
            _provider_budget = ProviderBudget(budget_cfg)
            logger.info(
                "Provider budget tracking enabled | providers: {}",
                ", ".join(budget_cfg.keys()),
            )

        # Pre-create LLM clients for each tier so we don't reconstruct per job.
        # These are used by the legacy quality_mode path when no budget config is set.
        try:
            _fast_config    = router.apply_model_override_for_mode(config, "fast")
            _quality_config = router.apply_model_override_for_mode(config, "quality")
            _fast_llm_call    = get_llm_client(_fast_config)
            _quality_llm_call = get_llm_client(_quality_config)
            logger.info(
                "Tier clients ready | fast={} | quality={}",
                router.get_model_for_quality_mode("fast"),
                router.get_model_for_quality_mode("quality"),
            )
        except Exception as exc:
            logger.warning("Could not pre-create tier LLM clients ({}); using default", exc)

    llm_cfg  = config["llm"]
    provider = llm_cfg["provider"]
    model    = llm_cfg["model"][provider]

    llm_call = get_llm_client(config)
    if _fast_llm_call is None:
        _fast_llm_call = llm_call
    if _quality_llm_call is None:
        _quality_llm_call = llm_call

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

    # Build structured profile once — counts as 1 API call toward rate limiter.
    # Use the quality-tier model when routing is enabled: the profile extraction is
    # a one-time call and the fast tier (e.g. llama-3.1-8b-instant) fails instructor
    # tool_use schema validation, forcing a degraded raw-text fallback that returns
    # a weak skills list.
    logger.info("Building structured profile from resume...")
    if router is not None:
        _profile_model = router.get_model_for_quality_mode("quality") or model
        _profile_cfg = router.apply_model_override_for_mode(config, "quality")
        try:
            _profile_llm_call = get_llm_client(_profile_cfg)
        except Exception:
            _profile_llm_call = llm_call
        try:
            _profile_instructor, _, _ = get_instructor_client(_profile_cfg)
        except Exception:
            _profile_instructor = instructor_client
    else:
        _profile_model = model
        _profile_llm_call = llm_call
        _profile_instructor = instructor_client

    if _profile_instructor:
        structured_profile = build_structured_profile(
            config, _profile_llm_call, _profile_instructor,
            model=_profile_model, temperature=config["llm"].get("temperature", 0)
        )
    else:
        structured_profile = build_structured_profile(config, _profile_llm_call)
    limiter.record(1000)

    # Hand the structured profile to the router so the synthetic path's _quick_ats
    # can use canonical-skill matching (otherwise multi-word skill phrases like
    # "Machine learning infrastructure" never match real JDs and ATS scores deflate).
    if router is not None:
        router.set_structured_profile(structured_profile)

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
            if router and _match_query:
                _routing_score = router.compute_routing_score(job, _match_query)
                _title_boost   = router.compute_title_boost(job)
                _skills_boost  = router.compute_skills_boost(job)
                title_matched  = _title_boost > 0
                _effective     = _routing_score + _title_boost + _skills_boost

                if not router.should_call_llm(_effective):
                    result = router.create_synthetic_score(
                        job, _routing_score, title_matched=title_matched
                    )
                    log_routing_decision(
                        job.id, _effective, router.threshold,
                        "skipped_llm", "below_threshold", profile,
                    )
                elif _provider_budget:
                    # ── Budget-aware provider fallback path ──────────────────
                    from selective_routing import select_model_with_fallback, RoutingExhaustedError

                    tier = router.compute_tier(_effective, job.raw_text or "", title_matched)

                    # Pre-estimate tokens so can_afford() has a realistic number.
                    _profile_text = json.dumps(structured_profile) if structured_profile else ""
                    _job_text     = job.raw_text or ""
                    _est_tokens   = estimate_tokens(
                        _profile_text, _job_text,
                        tier["max_tokens"],
                        tier.get("primary_model", ""),
                    )

                    result = None
                    _max_provider_attempts = len(
                        config.get("routing", {}).get("fallback_chain", [])
                    ) + 2

                    for _provider_attempt in range(_max_provider_attempts):
                        # Select provider (falls back on each iteration if previous was exhausted)
                        try:
                            _sel_provider, _sel_model = select_model_with_fallback(
                                tier, tier["tier"],
                                config.get("routing", {}),
                                _provider_budget,
                                _est_tokens,
                                job.id,
                                profile,
                            )
                        except RoutingExhaustedError as _exhausted:
                            logger.error(
                                "routing | job={} ALL providers exhausted — using synthetic score. {}",
                                job.id, _exhausted,
                            )
                            result = router.create_synthetic_score(
                                job, _routing_score, title_matched=title_matched
                            )
                            log_routing_decision(
                                job.id, _effective, router.threshold,
                                "skipped_llm", "all_providers_exhausted", profile,
                            )
                            break

                        # Build a config copy for this provider/model so score_dimensions
                        # uses the right model name in logs and instructor calls.
                        _tier_cfg = copy.deepcopy(config)
                        _tier_cfg["llm"]["provider"] = _sel_provider
                        _tier_cfg["llm"].setdefault("model", {})[_sel_provider] = _sel_model

                        # Cache LLM clients by (provider, model) to avoid re-init overhead.
                        _client_key = f"{_sel_provider}/{_sel_model}"
                        if _client_key not in _llm_client_cache:
                            try:
                                _llm_client_cache[_client_key] = get_llm_client(_tier_cfg)
                            except Exception as _ce:
                                logger.warning(
                                    "routing | could not build client for {}: {}", _client_key, _ce
                                )
                                _provider_budget.force_exhaust(_sel_provider)
                                continue
                        _tier_llm = _llm_client_cache[_client_key]

                        # Instructor is only valid for the primary configured provider.
                        _tier_instr = (
                            instructor_client
                            if _sel_provider == _primary_provider
                            else None
                        )

                        # Reserve budget before the call (optimistic spend).
                        _provider_budget.spend(_sel_provider, _est_tokens)

                        router.record_llm_call()
                        _tier_primary = tier.get("primary_model", "")
                        _tier_primary_provider = (
                            _tier_primary.split("/", 1)[0] if "/" in _tier_primary else ""
                        )
                        _is_fallback = bool(
                            _tier_primary_provider
                        ) and _sel_provider != _tier_primary_provider
                        if router.log_decisions:
                            logger.info(
                                "routing | job={} company={} tier={} model={}/{} "
                                "fallback={} routing_score={:.3f} effective={:.3f} "
                                "title_match={} est_tokens={}",
                                job.id, job.company, tier["tier"],
                                _sel_provider, _sel_model,
                                _is_fallback, _routing_score, _effective,
                                title_matched, _est_tokens,
                            )

                        try:
                            result = score_job(
                                job, _tier_cfg, _tier_llm,
                                structured_profile, _tier_instr,
                                max_tokens=tier["max_tokens"],
                            )
                            # Correct the ledger with actual token usage.
                            _actual_tokens = result.get("tokens_used", _est_tokens)
                            _provider_budget.update_actual(
                                _sel_provider, _est_tokens, _actual_tokens
                            )
                            log_routing_decision(
                                job.id, _effective, router.threshold,
                                "llm_called", f"tier:{tier['tier']}", profile,
                            )
                            break  # success — stop retrying

                        except Exception as _call_exc:
                            if is_rate_limit_error(_call_exc):
                                # Correct the optimistic spend before marking exhausted.
                                _provider_budget.update_actual(_sel_provider, _est_tokens, 0)
                                _provider_budget.force_exhaust(_sel_provider)
                                logger.warning(
                                    "routing | job={} rate-limit from {} — "
                                    "marking exhausted, trying next provider",
                                    job.id, _sel_provider,
                                )
                                continue
                            raise  # non-rate-limit error: let outer handler record it

                    if result is None:
                        # Defensive fallback — should not happen if RoutingExhaustedError is raised
                        result = router.create_synthetic_score(
                            job, _routing_score, title_matched=title_matched
                        )

                else:
                    # ── Legacy quality_mode path (no budget config) ──────────
                    tier = router.compute_tier(_effective, job.raw_text or "", title_matched)
                    if tier["quality_mode"] == "quality":
                        tier_llm    = _quality_llm_call
                        tier_config = _quality_config
                    else:
                        tier_llm    = _fast_llm_call
                        tier_config = _fast_config

                    router.record_llm_call()
                    if router.log_decisions:
                        logger.info(
                            "routing | job={} company={} tier={} model={} "
                            "routing_score={:.3f} effective={:.3f} title_match={}",
                            job.id, job.company, tier["tier"],
                            router.get_model_for_quality_mode(tier["quality_mode"]),
                            _routing_score, _effective, title_matched,
                        )
                    result = score_job(
                        job, tier_config, tier_llm,
                        structured_profile, instructor_client,
                        max_tokens=tier["max_tokens"],
                    )
                    log_routing_decision(
                        job.id, _effective, router.threshold,
                        "llm_called", f"tier:{tier['tier']}", profile,
                    )
            else:
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

    if router:
        router.log_summary()

    if _provider_budget:
        _provider_budget.log_summary()

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
