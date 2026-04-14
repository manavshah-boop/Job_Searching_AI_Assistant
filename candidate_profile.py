"""
candidate_profile.py - Structured profile extraction from resume.

Extracts a structured candidate profile once at startup using a single LLM call.
This structured profile is injected into every scoring prompt instead of the raw
resume text - ~300 tokens vs ~800+ for raw text, cleaner signal per job.

Supports both raw LLM with JSON parsing and instructor-based structured output.
"""

import json
import re
import time
from typing import Any, Callable, Optional, Tuple

from loguru import logger

LlmCall = Callable[[str, int], Tuple[str, int]]


def parse_llm_json(raw: str) -> dict:
    """Parse JSON from an LLM response, stripping markdown fences if present."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    return json.loads(raw)


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
    
    resume_text = config["profile"]["resume"]
    bio = config["profile"]["bio"]
    prefs = config["preferences"]

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
  "target_roles": {json.dumps(prefs["titles"])},
  "min_salary": {prefs["compensation"]["min_salary"]},
  "remote_preference": "{prefs["location"]["remote_ok"]}",
  "preferred_locations": {json.dumps(prefs["location"]["preferred_locations"])}
}}

Rules:
- yoe should reflect actual work experience, not education
- core_skills should be concrete and technical, not soft skills
- Return only JSON. No preamble. No markdown fences."""

    profile_dict = None

    # Try instructor-based structured output if available
    if instructor_client and model:
        try:
            for attempt in range(3):
                try:
                    result = instructor_client.messages.create(
                        model=model,
                        max_tokens=600,
                        temperature=temperature or 0,
                        messages=[{"role": "user", "content": prompt}],
                        response_model=StructuredProfile,
                    )
                    # Convert StructuredProfile model to dict
                    profile_dict = result.model_dump()
                    break
                except Exception as e:
                    if attempt < 2:
                        wait = 2 ** attempt * 10
                        logger.warning(f"Profile extraction failed. Retrying in {wait}s... (attempt {attempt + 1}/3)")
                        time.sleep(wait)
                    else:
                        raise
        except Exception as e:
            logger.info(f"Instructor profile extraction failed: {e}. Falling back to raw LLM.")
            profile_dict = None

    # Fallback to raw LLM call with JSON parsing if instructor unavailable
    if profile_dict is None:
        try:
            raw, _ = llm_call(prompt, 600)
            profile_dict = parse_llm_json(raw)
        except Exception:
            # Fallback - build a minimal profile directly from config.
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
            "min_salary": prefs["compensation"]["min_salary"],
            "remote_preference": str(prefs["location"]["remote_ok"]),
            "preferred_locations": prefs["location"]["preferred_locations"],
        }

    # Override config-critical fields
    profile_dict["target_roles"] = prefs.get("titles", [])
    profile_dict["min_salary"] = prefs["compensation"]["min_salary"]
    profile_dict["remote_preference"] = str(prefs["location"]["remote_ok"])
    profile_dict["preferred_locations"] = prefs["location"]["preferred_locations"]

    return profile_dict


def print_profile_summary(profile: dict) -> None:
    """Print structured profile to terminal for user verification."""
    skills = ", ".join(profile.get("core_skills", [])[:8])
    languages = ", ".join(profile.get("languages", []))
    past_roles = ", ".join(profile.get("past_roles", [])[:3])
    target = ", ".join(profile.get("target_roles", []))
    min_sal = profile.get("min_salary", 0)
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
    logger.info(f"  Salary:     ${min_sal:,}+")
    logger.info(f"  Location:   {remote_str} ({locs})")
    logger.info("-----------------------------------------------")


def confirm_profile() -> bool:
    """Ask user to confirm the extracted profile is correct. Returns True to proceed."""
    try:
        answer = input("Is this profile correct? [y/n]: ").strip().lower()
        return answer == "y"
    except EOFError:
        return True  # Non-interactive context - proceed automatically.
