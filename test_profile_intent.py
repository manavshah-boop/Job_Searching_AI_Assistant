"""
test_profile_intent.py - Tests for role-agnostic profile normalization.
"""

from __future__ import annotations

from profile_intent import (
    CANONICAL_SECTION_ALIASES,
    CANONICAL_TO_CHUNK_KEY,
    ProfileIntent,
    get_role_family_section_weights,
    map_header_to_canonical,
    normalize_profile_intent,
)


# ── Config fixtures ───────────────────────────────────────────────────────────


def _software_config() -> dict:
    return {
        "profile": {
            "name": "Manav Shah",
            "job_type": "fulltime",
            "resume": "Python FastAPI SQL AWS backend APIs React distributed systems",
            "bio": "Backend and AI platform engineer with 3 years experience.",
        },
        "preferences": {
            "titles": ["Backend Engineer", "AI Platform Engineer", "Software Engineer"],
            "desired_skills": ["Python", "FastAPI", "SQL", "AWS", "React", "Docker"],
            "hard_no_keywords": ["security clearance required"],
            "location": {"remote_ok": True, "preferred_locations": ["Seattle, WA"]},
            "compensation": {"min_salary": 130000},
            "yoe": 3,
            "filters": {"title_blocklist": ["Senior", "Staff", "Principal"]},
        },
    }


def _finance_config() -> dict:
    return {
        "profile": {
            "name": "Alex Chen",
            "job_type": "fulltime",
            "resume": "financial modeling Excel budgeting FP&A variance analysis GAAP CPA candidate",
            "bio": "Finance analyst with 2 years in corporate FP&A.",
        },
        "preferences": {
            "titles": ["Financial Analyst", "FP&A Analyst", "Corporate Finance Associate"],
            "desired_skills": ["Excel", "financial modeling", "budgeting", "variance analysis", "SQL", "Tableau"],
            "hard_no_keywords": [],
            "location": {"remote_ok": True, "preferred_locations": ["New York, NY"]},
            "compensation": {"min_salary": 80000},
            "yoe": 2,
            "filters": {"title_blocklist": ["Senior", "Director", "VP"]},
        },
    }


def _marketing_config() -> dict:
    return {
        "profile": {
            "name": "Jordan Lee",
            "job_type": "fulltime",
            "resume": "SEO paid search Google Analytics HubSpot email campaigns content strategy lifecycle marketing",
            "bio": "Growth marketing specialist focused on lifecycle and paid channels.",
        },
        "preferences": {
            "titles": ["Growth Marketing Manager", "Lifecycle Marketing", "Demand Generation"],
            "desired_skills": ["SEO", "Google Analytics", "HubSpot", "email campaigns", "paid search"],
            "hard_no_keywords": [],
            "location": {"remote_ok": True, "preferred_locations": []},
            "compensation": {"min_salary": 75000},
            "yoe": 2,
            "filters": {"title_blocklist": ["Director", "VP", "Head of"]},
        },
    }


def _internship_config() -> dict:
    return {
        "profile": {
            "name": "Sam Kim",
            "job_type": "internship",
            "resume": "Python SQL data analysis pandas numpy coursework statistics",
            "bio": "Junior at state university studying computer science.",
        },
        "preferences": {
            "titles": ["Software Engineering Intern", "Data Science Intern"],
            "desired_skills": ["Python", "SQL", "pandas"],
            "hard_no_keywords": [],
            "location": {"remote_ok": True, "preferred_locations": []},
            "compensation": {"intern_pay_preference": "paid_only", "monthly_stipend": 5000},
            "yoe": 0,
            "filters": {"title_blocklist": ["Senior", "Full-time only"]},
        },
    }


def _unknown_role_config() -> dict:
    return {
        "profile": {
            "name": "Pat Rivera",
            "job_type": "fulltime",
            "resume": "",
            "bio": "",
        },
        "preferences": {
            "titles": ["Consultant"],
            "desired_skills": [],
            "hard_no_keywords": [],
            "location": {"remote_ok": False, "preferred_locations": ["Chicago, IL"]},
            "compensation": {},
            "yoe": 1,
            "filters": {},
        },
    }


def _explicit_role_family_config() -> dict:
    cfg = _software_config()
    cfg["profile"]["role_family"] = "finance"
    return cfg


# ── Role family inference tests ───────────────────────────────────────────────


def test_software_profile_extracts_software_intent():
    intent = normalize_profile_intent(_software_config())

    assert intent.role_family == "software_engineering"
    assert "Backend Engineer" in intent.target_roles
    assert intent.job_type == "fulltime"
    assert intent.years_experience == 3
    assert intent.remote_ok is True
    assert "Seattle, WA" in intent.locations


def test_finance_profile_extracts_finance_intent():
    intent = normalize_profile_intent(_finance_config())

    assert intent.role_family == "finance"
    assert any("Financial Analyst" in r for r in intent.target_roles)
    assert intent.years_experience == 2
    assert "New York, NY" in intent.locations


def test_marketing_profile_extracts_marketing_intent():
    intent = normalize_profile_intent(_marketing_config())

    assert intent.role_family == "marketing"
    assert any("Marketing" in r for r in intent.target_roles)


def test_unknown_role_falls_back_to_general():
    intent = normalize_profile_intent(_unknown_role_config())

    assert intent.role_family == "general"


def test_explicit_role_family_overrides_inference():
    intent = normalize_profile_intent(_explicit_role_family_config())

    assert intent.role_family == "finance"


def test_internship_config_seniority_is_intern():
    intent = normalize_profile_intent(_internship_config())

    assert intent.seniority == "intern"
    assert intent.job_type == "internship"


# ── Skill/tool splitting tests ────────────────────────────────────────────────


def test_software_skills_split_into_domain_and_tools():
    intent = normalize_profile_intent(_software_config())

    all_skills = intent.domain_skills + intent.tools
    assert len(all_skills) > 0
    tool_names = [t.lower() for t in intent.tools]
    assert any("python" in t or "aws" in t or "sql" in t for t in tool_names)


def test_finance_skills_include_excel_as_tool():
    intent = normalize_profile_intent(_finance_config())

    tool_names = [t.lower() for t in intent.tools]
    assert "excel" in tool_names


# ── Dealbreaker extraction tests ──────────────────────────────────────────────


def test_dealbreakers_include_blocklist_and_hard_no():
    intent = normalize_profile_intent(_software_config())

    blocklist_terms = [d.lower() for d in intent.dealbreakers]
    assert any("senior" in t for t in blocklist_terms)
    assert any("clearance" in t for t in blocklist_terms)


# ── Compensation tests ────────────────────────────────────────────────────────


def test_fulltime_compensation_captured():
    intent = normalize_profile_intent(_software_config())

    assert intent.compensation.get("min_salary") == 130000


def test_internship_paid_only_preference_captured():
    intent = normalize_profile_intent(_internship_config())

    assert intent.compensation.get("intern_pay_preference") == "paid_only"
    assert intent.compensation.get("monthly_stipend") == 5000


# ── Seniority inference tests ─────────────────────────────────────────────────


def test_seniority_entry_for_low_yoe():
    cfg = _software_config()
    cfg["preferences"]["yoe"] = 1
    intent = normalize_profile_intent(cfg)
    assert intent.seniority == "entry"


def test_seniority_mid_for_moderate_yoe():
    cfg = _software_config()
    cfg["preferences"]["yoe"] = 4
    intent = normalize_profile_intent(cfg)
    assert intent.seniority == "mid"


def test_seniority_senior_for_high_yoe():
    cfg = _software_config()
    cfg["preferences"]["yoe"] = 7
    intent = normalize_profile_intent(cfg)
    assert intent.seniority == "senior"


# ── Section weights tests ─────────────────────────────────────────────────────


def test_software_section_weights_prioritize_requirements():
    weights = get_role_family_section_weights("software_engineering")

    assert weights["requirements"] > weights["benefits"]
    assert weights["requirements"] > weights["compensation"]


def test_finance_section_weights_prioritize_requirements_and_compensation():
    weights = get_role_family_section_weights("finance")

    assert weights["requirements"] >= weights["benefits"]
    assert weights["compensation"] > weights["benefits"]


def test_internship_section_weights_boost_compensation():
    sw_weights = get_role_family_section_weights("software_engineering")
    intern_weights = get_role_family_section_weights("internship")

    assert intern_weights["compensation"] >= sw_weights["compensation"]


def test_unknown_family_returns_general_weights():
    general_weights = get_role_family_section_weights("general")
    unknown_weights = get_role_family_section_weights("totally_unknown_role_family_xyz")

    assert unknown_weights == general_weights


# ── Canonical section alias tests ────────────────────────────────────────────


def test_map_header_to_canonical_requirements_aliases():
    assert map_header_to_canonical("What You Bring") == "requirements"
    assert map_header_to_canonical("Minimum Qualifications:") == "requirements"
    assert map_header_to_canonical("Basic Qualifications") == "requirements"
    assert map_header_to_canonical("Who You Are") == "requirements"


def test_map_header_to_canonical_responsibilities_aliases():
    assert map_header_to_canonical("What You'll Do") == "responsibilities"
    assert map_header_to_canonical("Day to Day") == "responsibilities"
    assert map_header_to_canonical("In This Role") == "responsibilities"


def test_map_header_to_canonical_preferred_qualifications():
    assert map_header_to_canonical("Nice to Have") == "preferred_qualifications"
    assert map_header_to_canonical("Bonus Points") == "preferred_qualifications"


def test_map_header_to_canonical_tools_and_skills():
    assert map_header_to_canonical("Tech Stack") == "tools_and_skills"
    assert map_header_to_canonical("Technologies") == "tools_and_skills"


def test_map_header_to_canonical_company():
    assert map_header_to_canonical("About Us") == "company"
    assert map_header_to_canonical("About the Company") == "company"


def test_map_header_to_canonical_logistics():
    assert map_header_to_canonical("Work Authorization") == "logistics"
    assert map_header_to_canonical("Visa Sponsorship") == "logistics"


def test_map_header_to_canonical_unknown_returns_none():
    assert map_header_to_canonical("Foobar Xyz Nonexistent") is None


def test_canonical_to_chunk_key_maps_preferred_qualifications():
    assert CANONICAL_TO_CHUNK_KEY["preferred_qualifications"] == "requirements"
    assert CANONICAL_TO_CHUNK_KEY["tools_and_skills"] == "requirements"
    assert CANONICAL_TO_CHUNK_KEY["company"] == "summary"
    assert CANONICAL_TO_CHUNK_KEY["logistics"] == "requirements"
