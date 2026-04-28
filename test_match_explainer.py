from __future__ import annotations

from match_explainer import build_match_explanation, recommend_action, summarize_match
from profile_intent import normalize_profile_intent
from reranker import MatchEvidence, RerankedJobResult


def _config(role_family: str, *, title: str, skills: list[str], resume: str = "", bio: str = "") -> dict:
    return {
        "profile": {
            "name": "Test Candidate",
            "job_type": "internship" if role_family == "internship" else "fulltime",
            "role_family": role_family,
            "resume": resume,
            "bio": bio,
            "industries": ["B2B SaaS"] if role_family in {"software_engineering", "marketing"} else ["Finance"],
            "education": ["BS"],
        },
        "preferences": {
            "titles": [title],
            "desired_skills": skills,
            "hard_no_keywords": [],
            "location": {"remote_ok": True, "preferred_locations": ["New York, NY"]},
            "compensation": {"min_salary": 120000},
            "yoe": 2,
            "filters": {"title_blocklist": ["Director"]},
        },
    }


def _score_record(**overrides):
    base = {
        "id": "job-1",
        "title": "Backend Engineer",
        "company": "Acme",
        "fit_score": 86,
        "ats_score": 82,
        "reasons": ["Strong backend alignment", "Python and AWS are explicit requirements"],
        "flags": [],
        "skill_misses": [],
        "dimension_scores": {
            "role_fit": 9,
            "stack_match": 8,
            "seniority": 8,
            "location": 7,
            "growth": 7,
            "compensation": 6,
        },
    }
    base.update(overrides)
    return base


def _reranked_result(
    *,
    title: str,
    final_score: float = 0.84,
    vector_score: float = 0.8,
    rerank_score: float = 0.85,
    matched_sections: list[str] | None = None,
    positive: list[str] | None = None,
    concerns: list[str] | None = None,
    matched_keywords: list[str] | None = None,
    missing_or_unclear: list[str] | None = None,
) -> RerankedJobResult:
    return RerankedJobResult(
        job_id="job-1",
        title=title,
        company="Acme",
        source="greenhouse",
        url="https://example.com/job-1",
        vector_score=vector_score,
        rerank_score=rerank_score,
        final_score=final_score,
        matched_sections=matched_sections or ["requirements", "responsibilities"],
        section_scores={"requirements": 0.78, "responsibilities": 0.74, "summary": 0.62},
        best_chunk_key="requirements",
        match_reason="Strong requirements and responsibilities match for key profile themes.",
        evidence_snippets=["Python, AWS, and APIs are called out directly in requirements."],
        evidence=MatchEvidence(
            positive=positive or ["Title match: Backend Engineer", "Skills present: Python, AWS, APIs", "Remote-friendly posting"],
            concerns=concerns or [],
            matched_keywords=matched_keywords or ["Python", "AWS", "APIs"],
            missing_or_unclear=missing_or_unclear or [],
        ),
    )


def test_strong_software_match_produces_skills_and_responsibility_strengths():
    intent = normalize_profile_intent(_config("software_engineering", title="Backend Engineer", skills=["Python", "AWS", "FastAPI"], resume="Python AWS FastAPI"))
    explanation = build_match_explanation(
        _score_record(),
        _score_record(),
        _reranked_result(title="Backend Engineer"),
        intent,
    )

    strength_names = {factor.name for factor in explanation.strengths}
    assert "Skills/tools match" in strength_names
    assert "Responsibilities match" in strength_names


def test_finance_profile_uses_finance_skills_and_credentials():
    config = _config(
        "finance",
        title="Financial Analyst",
        skills=["Excel", "financial modeling", "FP&A"],
        resume="CFA Excel financial modeling budgeting forecasting",
    )
    intent = normalize_profile_intent(config)
    explanation = build_match_explanation(
        _score_record(title="Financial Analyst"),
        _score_record(title="Financial Analyst"),
        _reranked_result(
            title="Financial Analyst",
            positive=["Title match: Financial Analyst", "Skills present: Excel, financial modeling, FP&A"],
            matched_keywords=["Excel", "financial modeling", "FP&A"],
        ),
        intent,
    )

    strength_names = {factor.name for factor in explanation.strengths}
    assert "Skills/tools match" in strength_names
    assert "Credentials/certifications" in strength_names


def test_marketing_profile_uses_marketing_skills_and_channels():
    intent = normalize_profile_intent(
        _config(
            "marketing",
            title="Growth Marketing Manager",
            skills=["SEO", "paid search", "HubSpot", "content strategy"],
            resume="SEO paid search lifecycle campaigns HubSpot content strategy",
        )
    )
    explanation = build_match_explanation(
        _score_record(title="Growth Marketing Manager"),
        _score_record(title="Growth Marketing Manager"),
        _reranked_result(
            title="Growth Marketing Manager",
            positive=["Title match: Growth Marketing Manager", "Skills present: SEO, paid search, HubSpot"],
            matched_keywords=["SEO", "paid search", "HubSpot"],
        ),
        intent,
    )

    skill_factor = next(f for f in explanation.strengths if f.name == "Skills/tools match")
    assert any("SEO" in item or "HubSpot" in item for item in skill_factor.evidence)


def test_seniority_mismatch_appears_as_concern():
    intent = normalize_profile_intent(_config("software_engineering", title="Backend Engineer", skills=["Python", "AWS"]))
    explanation = build_match_explanation(
        _score_record(),
        _score_record(),
        _reranked_result(title="Senior Backend Engineer", concerns=["Seniority concern: posting appears senior-level"]),
        intent,
    )

    assert any(factor.name == "Seniority fit" for factor in explanation.concerns)


def test_missing_credential_appears_as_concern():
    intent = normalize_profile_intent(_config("finance", title="Financial Analyst", skills=["Excel", "financial modeling"]))
    explanation = build_match_explanation(
        _score_record(title="Financial Analyst"),
        _score_record(title="Financial Analyst"),
        _reranked_result(
            title="Financial Analyst",
            concerns=["Credential may be required: CPA"],
            missing_or_unclear=["CPA"],
        ),
        intent,
    )

    assert any("CPA" in " ".join(factor.evidence) for factor in explanation.concerns)


def test_high_semantic_score_without_llm_score_returns_needs_more_info_or_strong_match():
    intent = normalize_profile_intent(_config("software_engineering", title="Backend Engineer", skills=["Python", "AWS"]))
    explanation = build_match_explanation(
        {"id": "job-1", "title": "Backend Engineer", "company": "Acme"},
        None,
        _reranked_result(title="Backend Engineer", final_score=0.83),
        intent,
    )

    assert explanation.recommended_action in {"Needs More Info", "Strong Match"}


def test_low_score_with_concerns_returns_maybe_or_skip():
    intent = normalize_profile_intent(_config("software_engineering", title="Backend Engineer", skills=["Python"]))
    explanation = build_match_explanation(
        _score_record(fit_score=48, ats_score=40, flags=["Location mismatch"], dimension_scores={"role_fit": 4, "stack_match": 4, "seniority": 3, "location": 2, "growth": 4, "compensation": 4}),
        _score_record(fit_score=48, ats_score=40, flags=["Location mismatch"], dimension_scores={"role_fit": 4, "stack_match": 4, "seniority": 3, "location": 2, "growth": 4, "compensation": 4}),
        _reranked_result(title="Backend Engineer", final_score=0.44, concerns=["Location may not match profile preferences"]),
        intent,
    )

    assert explanation.recommended_action in {"Maybe", "Skip"}


def test_explanation_handles_missing_reranked_result():
    intent = normalize_profile_intent(_config("software_engineering", title="Backend Engineer", skills=["Python", "AWS"]))
    explanation = build_match_explanation(_score_record(), _score_record(), None, intent)

    assert explanation.summary
    assert explanation.rerank_score is None
    assert any(factor.name == "Semantic evidence" for factor in explanation.unknowns)


def test_explanation_handles_missing_score_record():
    intent = normalize_profile_intent(_config("software_engineering", title="Backend Engineer", skills=["Python", "AWS"]))
    explanation = build_match_explanation(
        {"id": "job-1", "title": "Backend Engineer", "company": "Acme"},
        None,
        _reranked_result(title="Backend Engineer"),
        intent,
    )

    assert explanation.llm_fit_score is None
    assert any(factor.name == "LLM score coverage" for factor in explanation.unknowns)


def test_summary_is_deterministic_and_non_empty():
    intent = normalize_profile_intent(_config("software_engineering", title="Backend Engineer", skills=["Python", "AWS"]))
    explanation = build_match_explanation(_score_record(), _score_record(), _reranked_result(title="Backend Engineer"), intent)

    assert summarize_match(explanation) == explanation.summary
    assert explanation.summary.strip()


def test_recommended_action_is_deterministic():
    intent = normalize_profile_intent(_config("software_engineering", title="Backend Engineer", skills=["Python", "AWS"]))
    explanation = build_match_explanation(_score_record(), _score_record(), _reranked_result(title="Backend Engineer"), intent)

    assert recommend_action(explanation) == explanation.recommended_action
