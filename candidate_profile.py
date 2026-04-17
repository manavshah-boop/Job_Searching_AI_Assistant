"""
candidate_profile.py - Structured profile extraction from resume.

Extracts a structured candidate profile once at startup using a single LLM call.
This structured profile is injected into every scoring prompt instead of the raw
resume text - ~300 tokens vs ~800+ for raw text, cleaner signal per job.

Supports both raw LLM with JSON parsing and instructor-based structured output.
"""

import json
from typing import Any, Callable, Optional, Tuple

from loguru import logger

from llm_utils import parse_llm_response, safe_structured_call

LlmCall = Callable[[str, int], Tuple[str, int]]


_INTERN_PAY_PREFERENCE_LABELS = {
    "paid_only": "Paid only",
    "unpaid_ok": "Unpaid OK",
    "no_preference": "No preference",
}


def _normalize_intern_pay_preference(compensation: dict) -> str:
    """Return a stable internship compensation preference value."""
    preference = str(compensation.get("intern_pay_preference", "")).strip().lower()
    if preference in _INTERN_PAY_PREFERENCE_LABELS:
        return preference
    if compensation.get("monthly_stipend"):
        return "paid_only"
    return "no_preference"


def _structured_profile_compensation(config: dict) -> tuple[Optional[int], str, str]:
    """Return target salary plus prompt/display text for the current profile."""
    profile = config.get("profile", {})
    prefs = config.get("preferences", {})
    compensation = prefs.get("compensation", {})
    is_intern = profile.get("job_type") == "internship"

    if is_intern:
        pay_pref = _normalize_intern_pay_preference(compensation)
        stipend = compensation.get("monthly_stipend")
        stipend_value = int(stipend) if stipend not in (None, "", 0) else None
        pref_label = _INTERN_PAY_PREFERENCE_LABELS[pay_pref]
        if pay_pref == "paid_only":
            prompt_value = str(stipend_value) if stipend_value is not None else "null"
            display_value = (
                f"{pref_label} (target monthly stipend: ${stipend_value:,}/mo)"
                if stipend_value is not None
                else pref_label
            )
        else:
            prompt_value = "null"
            display_value = pref_label
        return stipend_value if pay_pref == "paid_only" else None, prompt_value, display_value

    min_salary = compensation.get("min_salary")
    salary_value = int(min_salary) if min_salary not in (None, "") else None
    prompt_value = str(salary_value) if salary_value is not None else "null"
    display_value = f"${salary_value:,}+" if salary_value is not None else "Not set"
    return salary_value, prompt_value, display_value


def build_structured_profile(
    config: dict, 
    llm_call: LlmCall,
    instructor_client: Optional[Any] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> dict:
    """
    Extract a structured profile from the raw resume once at startup.
    This gets injected into every scoring prompt instead of the raw PDF text.
    Called once in score_all_jobs() before the scoring loop.
    
    Supports both raw LLM with JSON parsing and instructor-based structured output.
    
    Args:
      config: Full application config dict
      llm_call: Raw LLM callable for fallback (prompt, max_tokens) -> (text, tokens)
      instructor_client: Optional instructor-wrapped client for structured output
      model: Optional model name (needed for instructor calls)
      temperature: Optional temperature (needed for instructor calls)
    """
    from models import StructuredProfile
    
    profile_cfg = config.get("profile", {})
    resume_text = profile_cfg.get("resume", "")
    bio = profile_cfg.get("bio", "")
    prefs = config.get("preferences", {})
    location = prefs.get("location", {})
    is_intern = profile_cfg.get("job_type") == "internship"
    target_salary, target_salary_prompt, compensation_display = _structured_profile_compensation(config)
    compensation_rules = (
        "- target_salary should be the candidate's monthly stipend target only when they want paid internships only; otherwise return null\n"
        "- Internship compensation preference: "
        f"{compensation_display}\n"
        "- Do not invent a stipend target if none is specified"
        if is_intern
        else "- target_salary should be the candidate's minimum annual salary expectation when available; otherwise return null"
    )

    prompt = f"""Extract a structured candidate profile from this resume. Return only valid JSON, no markdown.

RESUME:
{resume_text[:3000]}

BIO:
{bio}

Return this exact structure:
{{
  "name": "...",
  "yoe": <integer years of experience>,
  "current_title": "...",
  "core_skills": ["skill1", "skill2", ...],
  "languages": ["Python", "..."],
  "frameworks": ["FastAPI", "..."],
  "cloud": ["AWS", "..."],
  "past_roles": ["Role at Company", "..."],
  "education": "BS Computer Science, Purdue University",
  "strengths": ["LLM integration", "..."],
  "target_roles": {json.dumps(prefs.get("titles", []))},
  "target_salary": {target_salary_prompt},
  "remote_preference": "{location.get("remote_ok", True)}",
  "preferred_locations": {json.dumps(location.get("preferred_locations", []))}
}}

Rules:
- yoe should reflect actual work experience, not education
- core_skills should be concrete and technical, not soft skills
- target_roles should stay aligned with the configured target titles
{compensation_rules}
- Return only JSON. No preamble. No markdown fences."""

    profile_dict = None

    # Structured output via instructor (preferred path)
    if instructor_client and model:
        profile_dict = safe_structured_call(
            instructor_client, model, prompt, StructuredProfile,
            max_tokens=600,
            temperature=temperature or 0,
            label="profile",
        )

    # Fallback: raw LLM call + JSON parsing (also handles {"value": x} wrapping)
    if profile_dict is None:
        try:
            raw, _ = llm_call(prompt, 600)
            profile_dict = parse_llm_response(raw)
        except Exception:
            profile_dict = None

    # If we still don't have a profile, build a minimal one from config
    if profile_dict is None:
        profile_dict = {
            "name": config["profile"].get("name", "Candidate"),
            "yoe": prefs.get("yoe", 2),
            "current_title": "",
            "core_skills": prefs.get("desired_skills", []),
            "languages": [],
            "frameworks": [],
            "cloud": [],
            "past_roles": [],
            "education": "",
            "strengths": [],
            "target_roles": prefs.get("titles", []),
            "target_salary": target_salary,
            "remote_preference": str(location.get("remote_ok", True)),
            "preferred_locations": location.get("preferred_locations", []),
        }

    # Override config-critical fields
    profile_dict["target_roles"] = prefs.get("titles", [])
    profile_dict["target_salary"] = target_salary
    profile_dict["remote_preference"] = str(location.get("remote_ok", True))
    profile_dict["preferred_locations"] = location.get("preferred_locations", [])
    profile_dict["job_type"] = profile_cfg.get("job_type", "fulltime")
    if is_intern:
        profile_dict["intern_pay_preference"] = _normalize_intern_pay_preference(prefs.get("compensation", {}))

    return profile_dict


def print_profile_summary(profile: dict) -> None:
    """Print structured profile to terminal for user verification."""
    skills = ", ".join(profile.get("core_skills", [])[:8])
    languages = ", ".join(profile.get("languages", []))
    past_roles = ", ".join(profile.get("past_roles", [])[:3])
    target = ", ".join(profile.get("target_roles", []))
    target_salary = profile.get("target_salary")
    is_intern = profile.get("job_type") == "internship"
    remote = str(profile.get("remote_preference", "True"))
    locs = ", ".join(profile.get("preferred_locations", []))
    remote_str = "Remote preferred" if remote == "True" else "Open to office"

    logger.info("-- Candidate Profile ------------------------------")
    logger.info(f"  Name:       {profile.get('name', '')}")
    logger.info(f"  Experience: {profile.get('yoe', '?')} years")
    logger.info(f"  Skills:     {skills}")
    if languages:
        logger.info(f"  Languages:  {languages}")
    if past_roles:
        logger.info(f"  Past Roles: {past_roles}")
    logger.info(f"  Target:     {target}")
    if is_intern:
        pay_pref = str(profile.get("intern_pay_preference", "no_preference"))
        pay_pref_label = _INTERN_PAY_PREFERENCE_LABELS.get(pay_pref, "No preference")
        if target_salary:
            logger.info(f"  Comp:       {pay_pref_label} (${int(target_salary):,}/mo target)")
        else:
            logger.info(f"  Comp:       {pay_pref_label}")
    else:
        if target_salary not in (None, ""):
            logger.info(f"  Salary:     ${int(target_salary):,}+")
        else:
            logger.info("  Salary:     Not set")
    logger.info(f"  Location:   {remote_str} ({locs})")
    logger.info("-----------------------------------------------")


def confirm_profile() -> bool:
    """Ask user to confirm the extracted profile is correct. Returns True to proceed."""
    try:
        answer = input("Is this profile correct? [y/n]: ").strip().lower()
        return answer == "y"
    except EOFError:
        return True  # Non-interactive context - proceed automatically.
