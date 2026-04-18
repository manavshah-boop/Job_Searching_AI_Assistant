import json

import dashboard
from candidate_profile import build_structured_profile
from db import Job
from onboarding import generate_config
from scraper import requires_advanced_degree
from scorer import keyword_prescore, score_dimensions


def test_generate_config_for_internship_uses_intern_compensation_shape():
    config = generate_config(
        {
            "profile_slug": "intern_test",
            "name": "Intern Test",
            "job_type": "internship",
            "bio": "Student seeking backend internships.",
            "resume_type": "text",
            "resume_text": "resume body",
            "titles": ["Software Engineering Intern"],
            "desired_skills": ["Python", "AWS"],
            "hard_no_keywords": ["senior", "staff", "principal"],
            "remote_ok": True,
            "preferred_locations": ["Remote"],
            "intern_pay_preference": "paid_only",
            "stipend_expectation": 4500,
            "provider": "groq",
            "model_key": "groq",
            "model_id": "test-model",
        }
    )

    compensation = config["preferences"]["compensation"]
    filters = config["preferences"]["filters"]

    assert compensation == {
        "intern_pay_preference": "paid_only",
        "monthly_stipend": 4500,
    }
    assert "min_salary" not in compensation
    assert filters["max_job_age_days"] == 14
    assert "Senior" in filters["title_blocklist"]
    assert "Intern" not in filters["title_blocklist"]


def test_generate_config_for_internship_without_paid_requirement_omits_stipend():
    config = generate_config(
        {
            "profile_slug": "intern_open",
            "name": "Intern Open",
            "job_type": "internship",
            "bio": "Student open to different internship formats.",
            "resume_type": "text",
            "resume_text": "resume body",
            "titles": ["Software Engineering Intern"],
            "desired_skills": ["Python"],
            "hard_no_keywords": ["senior"],
            "remote_ok": True,
            "preferred_locations": ["Remote"],
            "intern_pay_preference": "no_preference",
            "provider": "groq",
            "model_key": "groq",
            "model_id": "test-model",
        }
    )

    compensation = config["preferences"]["compensation"]

    assert compensation == {"intern_pay_preference": "no_preference"}
    assert "min_salary" not in compensation


def test_generate_config_for_paid_only_internship_can_omit_stipend_target():
    config = generate_config(
        {
            "profile_slug": "intern_paid_no_target",
            "name": "Intern Paid No Target",
            "job_type": "internship",
            "bio": "Student seeking paid internships.",
            "resume_type": "text",
            "resume_text": "resume body",
            "titles": ["Software Engineering Intern"],
            "desired_skills": ["Python"],
            "hard_no_keywords": ["senior"],
            "remote_ok": True,
            "preferred_locations": ["Remote"],
            "intern_pay_preference": "paid_only",
            "provider": "groq",
            "model_key": "groq",
            "model_id": "test-model",
        }
    )

    compensation = config["preferences"]["compensation"]

    assert compensation == {"intern_pay_preference": "paid_only"}
    assert "monthly_stipend" not in compensation


def test_build_structured_profile_for_intern_uses_target_salary_and_prompt_guidance():
    captured = {}
    config = {
        "profile": {
            "name": "Intern Test",
            "job_type": "internship",
            "resume": "Built APIs with Python and FastAPI.",
            "bio": "CS student targeting summer internships.",
        },
        "preferences": {
            "titles": ["Software Engineering Intern"],
            "desired_skills": ["Python", "FastAPI"],
            "location": {"remote_ok": True, "preferred_locations": ["Remote"]},
            "compensation": {"intern_pay_preference": "unpaid_ok"},
        },
    }

    def fake_llm(prompt: str, max_tokens: int):
        captured["prompt"] = prompt
        return (
            json.dumps(
                {
                    "name": "Intern Test",
                    "yoe": 0,
                    "current_title": "Student",
                    "core_skills": ["Python", "FastAPI"],
                    "languages": ["Python"],
                    "frameworks": ["FastAPI"],
                    "cloud": [],
                    "past_roles": ["Student Developer"],
                    "education": "BS Computer Science",
                    "strengths": ["Backend systems"],
                    "target_roles": ["Software Engineering Intern"],
                    "target_salary": None,
                    "remote_preference": "True",
                    "preferred_locations": ["Remote"],
                }
            ),
            42,
        )

    profile = build_structured_profile(config, fake_llm)

    assert '"target_salary": null' in captured["prompt"]
    assert "monthly stipend target only when they want paid internships only" in captured["prompt"]
    assert profile["target_salary"] is None
    assert profile["intern_pay_preference"] == "unpaid_ok"


def test_score_dimensions_for_paid_only_internship_without_stipend_target_mentions_paid_only():
    captured = {}
    config = {
        "profile": {
            "name": "Intern Test",
            "job_type": "internship",
            "resume": "resume body",
            "bio": "Student builder",
            "target_season": "Summer 2027",
            "school": "State University",
            "major": "Computer Science",
        },
        "preferences": {
            "titles": ["Software Engineering Intern"],
            "desired_skills": ["Python", "AWS"],
            "location": {"remote_ok": True, "preferred_locations": ["Remote"]},
            "compensation": {"intern_pay_preference": "paid_only"},
            "filters": {"max_yoe": 1},
            "yoe": 0,
        },
        "scoring": {
            "weights": {
                "role_fit": 0.3,
                "stack_match": 0.25,
                "seniority": 0.2,
                "location": 0.1,
                "growth": 0.1,
                "compensation": 0.05,
            }
        },
        "llm": {"provider": "groq", "model": {"groq": "test-model"}, "temperature": 0},
    }
    job = Job(
        id="intern-2",
        title="Software Engineering Intern",
        company="Acme",
        location="Remote",
        url="https://example.com/intern-2",
        raw_text="Paid software engineering internship for students.",
        source="greenhouse",
    )

    def fake_llm(prompt: str, max_tokens: int):
        captured["prompt"] = prompt
        return (
            json.dumps(
                {
                    "disqualified": False,
                    "disqualify_reason": "",
                    "role_fit": 9,
                    "stack_match": 8,
                    "seniority": 10,
                    "location": 9,
                    "growth": 7,
                    "compensation": 9,
                    "reasons": ["Intern title match", "Python stack", "Paid internship"],
                    "flags": [],
                    "one_liner": "Strong fit for a paid software engineering internship.",
                }
            ),
            100,
        )

    score_dimensions(job, config, fake_llm)

    assert "Compensation Preference: Paid only" in captured["prompt"]
    assert "target stipend:" not in captured["prompt"]


def test_score_dimensions_for_internship_mentions_paid_preference_and_avoids_min_salary():
    captured = {}
    config = {
        "profile": {
            "name": "Intern Test",
            "job_type": "internship",
            "resume": "resume body",
            "bio": "Student builder",
            "target_season": "Summer 2027",
            "school": "State University",
            "major": "Computer Science",
        },
        "preferences": {
            "titles": ["Software Engineering Intern"],
            "desired_skills": ["Python", "AWS"],
            "location": {"remote_ok": True, "preferred_locations": ["Remote"]},
            "compensation": {
                "intern_pay_preference": "paid_only",
                "monthly_stipend": 5000,
            },
            "filters": {"max_yoe": 1},
            "yoe": 0,
        },
        "scoring": {
            "weights": {
                "role_fit": 0.3,
                "stack_match": 0.25,
                "seniority": 0.2,
                "location": 0.1,
                "growth": 0.1,
                "compensation": 0.05,
            }
        },
        "llm": {"provider": "groq", "model": {"groq": "test-model"}, "temperature": 0},
    }
    job = Job(
        id="intern-1",
        title="Software Engineering Intern",
        company="Acme",
        location="Remote",
        url="https://example.com/intern-1",
        raw_text="Paid software engineering internship. $30/hour. Students pursuing a BS or MS are welcome.",
        source="greenhouse",
    )

    def fake_llm(prompt: str, max_tokens: int):
        captured["prompt"] = prompt
        return (
            json.dumps(
                {
                    "disqualified": False,
                    "disqualify_reason": "",
                    "role_fit": 9,
                    "stack_match": 8,
                    "seniority": 10,
                    "location": 9,
                    "growth": 7,
                    "compensation": 9,
                    "reasons": ["Intern title match", "Python stack", "Paid internship"],
                    "flags": [],
                    "one_liner": "Strong fit for a paid software engineering internship.",
                }
            ),
            100,
        )

    result = score_dimensions(job, config, fake_llm)

    assert "Compensation Preference: Paid only (target stipend: $5,000/mo)" in captured["prompt"]
    assert "Min Salary:" not in captured["prompt"]
    assert "Penalize unpaid postings" in captured["prompt"]
    assert "currently pursuing" in captured["prompt"]
    assert result["fit_score"] > 0


def test_requires_advanced_degree_ignores_pursuing_degree_language():
    assert not requires_advanced_degree("Students pursuing a master's degree in computer science are encouraged to apply.")
    assert requires_advanced_degree("Master's degree required for this research scientist role.")


def test_effective_config_summary_for_intern_does_not_show_min_salary_zero():
    config = {
        "profile": {"job_type": "internship"},
        "llm": {"provider": "groq"},
        "scoring": {"min_display_score": 65},
        "preferences": {
            "titles": ["Software Engineering Intern"],
            "desired_skills": ["Python", "AWS"],
            "location": {"remote_ok": True, "preferred_locations": ["Remote"]},
            "compensation": {"intern_pay_preference": "no_preference"},
        },
    }
    raw_config = {
        "sources": {
            "greenhouse": {"enabled": True},
            "lever": {"enabled": False},
            "hn": {"enabled": False},
        }
    }

    summary = dashboard.effective_config_summary(config, raw_config)

    assert "Compensation preference: No preference" in summary
    assert not any("Minimum salary" in line for line in summary)


def test_keyword_prescore_allows_close_title_variants_without_skill_phrase_overlap():
    config = {
        "preferences": {
            "titles": ["Software Engineer", "Backend Engineer"],
            "desired_skills": [
                "Machine learning infrastructure",
                "LLM systems",
                "Distributed systems",
                "CI/CD pipelines",
            ],
        }
    }
    job = Job(
        id="early-career-1",
        title="Software Engineer - Early Career",
        company="Datadog",
        location="New York, United States",
        url="https://example.com/early-career-1",
        raw_text="Build backend services with Python.",
        source="greenhouse",
    )

    assert keyword_prescore(job, config) == 1.0
