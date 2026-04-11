"""
profile.py — Structured profile extraction from resume.

Extracts a structured candidate profile once at startup using a single LLM call.
This structured profile is injected into every scoring prompt instead of the raw
resume text — ~300 tokens vs ~800+ for raw text, cleaner signal per job.
"""

import json
import re
import sys
from typing import Callable, Dict, Tuple

LlmCall = Callable[[str, int], Tuple[str, int]]


def parse_llm_json(raw: str) -> dict:
    """Parse JSON from an LLM response, stripping markdown fences if present."""
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
    return json.loads(raw)


def build_structured_profile(config: dict, llm_call: LlmCall) -> dict:
    """
    Extracts a structured profile from the raw resume once at startup.
    This gets injected into every scoring prompt instead of the raw PDF text.
    Called once in score_all_jobs() before the scoring loop.
    """
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

    raw, _ = llm_call(prompt, 600)
    try:
        return parse_llm_json(raw)
    except Exception:
        # Fallback — build minimal structured profile from config directly
        return {
            "name": config["profile"].get("name", "Candidate"),
            "yoe": prefs.get("yoe", 2),
            "core_skills": prefs.get("desired_skills", []),
            "languages": [],
            "frameworks": [],
            "cloud": [],
            "past_roles": [],
            "education": "",
            "target_roles": prefs.get("titles", []),
            "min_salary": prefs["compensation"]["min_salary"],
            "remote_preference": str(prefs["location"]["remote_ok"]),
            "preferred_locations": prefs["location"]["preferred_locations"],
        }


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

    print("\n── Candidate Profile ─────────────────────────────")
    print(f"  Name:       {profile.get('name', '')}")
    print(f"  Experience: {profile.get('yoe', '?')} years")
    print(f"  Skills:     {skills}")
    if languages:
        print(f"  Languages:  {languages}")
    if past_roles:
        print(f"  Past Roles: {past_roles}")
    print(f"  Target:     {target}")
    print(f"  Salary:     ${min_sal:,}+")
    print(f"  Location:   {remote_str} ({locs})")
    print("──────────────────────────────────────────────────")


def confirm_profile() -> bool:
    """Ask user to confirm the extracted profile is correct. Returns True to proceed."""
    try:
        answer = input("Is this profile correct? [y/n]: ").strip().lower()
        return answer == "y"
    except EOFError:
        return True  # non-interactive context — proceed automatically
