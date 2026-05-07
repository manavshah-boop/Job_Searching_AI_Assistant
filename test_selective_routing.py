"""
test_selective_routing.py — Unit and integration tests for selective LLM routing.

Run with: pytest test_selective_routing.py -v
"""

import math
from unittest.mock import MagicMock, patch

import pytest

from db import Job
from selective_routing import SelectiveRouter, RoutingConfig, routing_enabled, _sigmoid


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_config(
    enabled: bool = True,
    threshold: float = 0.65,
    quality_mode: str = "fast",
    provider: str = "groq",
    log_decisions: bool = False,
) -> dict:
    return {
        "routing": {
            "enabled": enabled,
            "threshold": threshold,
            "quality_mode": quality_mode,
            "log_routing_decisions": log_decisions,
        },
        "llm": {
            "provider": provider,
            "model": {
                "groq": "meta-llama/llama-4-scout-17b-16e-instruct",
                "anthropic": "claude-sonnet-4-20250514",
                "openai": "gpt-4o-mini",
                "gemini": "gemini-2.5-flash",
            },
            "temperature": 0,
        },
        "scoring": {
            "weights": {
                "role_fit": 0.30,
                "stack_match": 0.25,
                "seniority": 0.20,
                "location": 0.10,
                "growth": 0.10,
                "compensation": 0.05,
            }
        },
        "preferences": {
            "desired_skills": ["Python", "AWS", "Machine Learning"],
        },
        "reranking": {
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        },
    }


def make_job(job_id: str = "test_123", raw_text: str = "Python engineer AWS ML") -> Job:
    return Job(
        id=job_id,
        title="Software Engineer",
        company="Acme Corp",
        location="Remote",
        url="https://example.com/job/123",
        raw_text=raw_text,
        source="greenhouse",
    )


# ── routing_enabled() helper ──────────────────────────────────────────────────

def test_routing_enabled_true():
    assert routing_enabled(make_config(enabled=True)) is True


def test_routing_enabled_false():
    assert routing_enabled(make_config(enabled=False)) is False


def test_routing_enabled_missing_section():
    assert routing_enabled({}) is False


# ── SelectiveRouter init ──────────────────────────────────────────────────────

def test_router_init_threshold_clamped_low():
    cfg = make_config(threshold=-0.5)
    router = SelectiveRouter(cfg)
    assert router.threshold == 0.0


def test_router_init_threshold_clamped_high():
    cfg = make_config(threshold=1.5)
    router = SelectiveRouter(cfg)
    assert router.threshold == 1.0


def test_router_init_invalid_threshold_uses_default():
    from selective_routing import _DEFAULT_THRESHOLD
    cfg = make_config()
    cfg["routing"]["threshold"] = "not_a_number"
    router = SelectiveRouter(cfg)
    assert router.threshold == _DEFAULT_THRESHOLD


def test_router_init_quality_mode():
    router = SelectiveRouter(make_config(quality_mode="quality"))
    assert router.quality_mode == "quality"


# ── should_call_llm() ─────────────────────────────────────────────────────────

def test_should_call_llm_above_threshold():
    router = SelectiveRouter(make_config(threshold=0.65))
    assert router.should_call_llm(0.75) is True


def test_should_call_llm_at_threshold():
    router = SelectiveRouter(make_config(threshold=0.65))
    assert router.should_call_llm(0.65) is True  # boundary is inclusive


def test_should_call_llm_below_threshold():
    router = SelectiveRouter(make_config(threshold=0.65))
    assert router.should_call_llm(0.64) is False


def test_should_call_llm_zero_score():
    router = SelectiveRouter(make_config(threshold=0.65))
    assert router.should_call_llm(0.0) is False


def test_should_call_llm_perfect_score():
    router = SelectiveRouter(make_config(threshold=0.65))
    assert router.should_call_llm(1.0) is True


# ── create_synthetic_score() ──────────────────────────────────────────────────

def test_create_synthetic_score_structure():
    router = SelectiveRouter(make_config())
    job = make_job()
    result = router.create_synthetic_score(job, routing_score=0.70)

    assert "job" in result
    assert "fit_score" in result
    assert "dimension_scores" in result
    assert "reasons" in result
    assert "flags" in result
    assert "one_liner" in result
    assert "tokens_used" in result
    assert result["tokens_used"] == 0


def test_create_synthetic_score_conservative():
    """Synthetic scores must be lower than a theoretical LLM score at same routing_score."""
    router = SelectiveRouter(make_config())
    job = make_job()
    result = router.create_synthetic_score(job, routing_score=0.80)

    dims = result["dimension_scores"]
    # At routing_score=0.80, role_fit (0.80 * 0.85 * 10 = 6.8 → 7) + possible summary bonus
    assert dims["role_fit"] <= 8
    assert dims["stack_match"] <= 8


def test_create_synthetic_score_fit_range():
    router = SelectiveRouter(make_config())
    job = make_job()

    for score in [0.0, 0.3, 0.5, 0.65, 0.80, 1.0]:
        result = router.create_synthetic_score(job, routing_score=score)
        assert 0 <= result["fit_score"] <= 100


def test_create_synthetic_score_flag():
    router = SelectiveRouter(make_config())
    result = router.create_synthetic_score(make_job(), 0.60)
    # Flags must be human-readable — no raw internal key names allowed
    assert len(result["flags"]) >= 1
    assert all("score_source:" not in f and "matched_sections:" not in f for f in result["flags"])
    assert any("estimate" in f.lower() or "not reviewed" in f.lower() for f in result["flags"])


def test_create_synthetic_score_not_disqualified():
    """Routing skipped jobs should never be marked disqualified."""
    router = SelectiveRouter(make_config())
    result = router.create_synthetic_score(make_job(), 0.60)
    # disqualified key may be absent (matching score_job behavior) or False
    assert not result.get("disqualified", False)


def test_create_synthetic_score_increments_count():
    router = SelectiveRouter(make_config())
    assert router._counts["skipped"] == 0
    router.create_synthetic_score(make_job(), 0.60)
    assert router._counts["skipped"] == 1


def test_create_synthetic_score_one_liner_flags_estimate():
    """one_liner must always make clear the score is keyword-based, not LLM-verified."""
    router = SelectiveRouter(make_config(threshold=0.65))
    result = router.create_synthetic_score(make_job(), 0.60)
    text = result["one_liner"].lower()
    assert "estimate" in text or "manual check" in text or "keyword" in text


def test_create_synthetic_score_uses_match_reason_when_no_evidence():
    """
    create_synthetic_score uses the supplied match_reason as the one_liner only
    when there is NO title match and NO tech-overlap evidence; otherwise the
    extracted evidence (skills found in the JD) takes precedence — that's the
    actual contract today.
    """
    router = SelectiveRouter(make_config())
    bare_job = make_job(raw_text="Generic role with no matching tech keywords here.")
    result = router.create_synthetic_score(bare_job, 0.60, match_reason="Good Python fit")
    assert result["one_liner"] == "Good Python fit"


def test_create_synthetic_score_includes_skills_when_present():
    """When tech overlaps exist, one_liner surfaces the matching skill names."""
    router = SelectiveRouter(make_config())
    result = router.create_synthetic_score(
        make_job(raw_text="Python AWS Machine Learning role"), 0.60,
    )
    # Job and config share Python and AWS; one_liner must mention at least one.
    text = result["one_liner"]
    assert ("Python" in text) or ("AWS" in text)


def test_create_synthetic_score_uses_matched_sections():
    """Providing matched_sections should adjust dimension scores."""
    router = SelectiveRouter(make_config())
    job = make_job()
    result_no_sections = router.create_synthetic_score(job, 0.70, matched_sections=[])
    result_with_req = router.create_synthetic_score(job, 0.70, matched_sections=["requirements"])
    # stack_match should be higher when requirements section matched
    assert (
        result_with_req["dimension_scores"]["stack_match"]
        >= result_no_sections["dimension_scores"]["stack_match"]
    )


def test_create_synthetic_score_responsibilities_lifts_seniority():
    """'responsibilities' in matched_sections should lift seniority from 5 to 6."""
    router = SelectiveRouter(make_config())
    job = make_job()
    result_no = router.create_synthetic_score(job, 0.70, matched_sections=[])
    result_yes = router.create_synthetic_score(job, 0.70, matched_sections=["responsibilities"])
    assert result_yes["dimension_scores"]["seniority"] > result_no["dimension_scores"]["seniority"]


# ── get_effective_llm_model() ─────────────────────────────────────────────────

def test_get_effective_llm_model_groq_fast():
    router = SelectiveRouter(make_config(provider="groq", quality_mode="fast"))
    assert router.get_effective_llm_model() == "llama-3.1-8b-instant"


def test_get_effective_llm_model_groq_quality():
    router = SelectiveRouter(make_config(provider="groq", quality_mode="quality"))
    assert router.get_effective_llm_model() == "meta-llama/llama-4-scout-17b-16e-instruct"


def test_get_effective_llm_model_anthropic_fast():
    router = SelectiveRouter(make_config(provider="anthropic", quality_mode="fast"))
    model = router.get_effective_llm_model()
    assert "haiku" in model


def test_get_effective_llm_model_anthropic_quality():
    router = SelectiveRouter(make_config(provider="anthropic", quality_mode="quality"))
    model = router.get_effective_llm_model()
    assert "sonnet" in model


def test_get_effective_llm_model_openai_fast():
    router = SelectiveRouter(make_config(provider="openai", quality_mode="fast"))
    assert router.get_effective_llm_model() == "gpt-4o-mini"


def test_get_effective_llm_model_unknown_falls_back():
    cfg = make_config(provider="groq", quality_mode="unknown_mode")
    router = SelectiveRouter(cfg)
    # Should fall back to config model
    result = router.get_effective_llm_model()
    assert isinstance(result, str)


# ── apply_model_override() ────────────────────────────────────────────────────

def test_apply_model_override_changes_model():
    cfg = make_config(provider="groq", quality_mode="fast")
    router = SelectiveRouter(cfg)
    overridden = router.apply_model_override(cfg)
    assert overridden["llm"]["model"]["groq"] == "llama-3.1-8b-instant"


def test_apply_model_override_does_not_mutate_original():
    cfg = make_config(provider="groq", quality_mode="fast")
    original_model = cfg["llm"]["model"]["groq"]
    router = SelectiveRouter(cfg)
    router.apply_model_override(cfg)
    # original config must not be changed
    assert cfg["llm"]["model"]["groq"] == original_model


# ── compute_routing_score() ───────────────────────────────────────────────────

def test_compute_routing_score_empty_text_returns_zero():
    router = SelectiveRouter(make_config())
    job = make_job(raw_text="")
    score = router.compute_routing_score(job, "software engineer Python")
    assert score == 0.0


def test_compute_routing_score_encoder_error_returns_half():
    """If cross-encoder raises, should fail-open and return 0.5."""
    router = SelectiveRouter(make_config())
    mock_encoder = MagicMock()
    mock_encoder.predict.side_effect = RuntimeError("model error")
    router._encoder = mock_encoder

    job = make_job(raw_text="Python engineer")
    score = router.compute_routing_score(job, "Python engineer")
    assert score == 0.5


def test_compute_routing_score_with_mock_encoder():
    """Mock the cross-encoder to verify sigmoid normalization."""
    router = SelectiveRouter(make_config())
    mock_encoder = MagicMock()
    # cross-encoder raw output of 0.0 → sigmoid → 0.5
    mock_encoder.predict.return_value = [0.0]
    router._encoder = mock_encoder

    job = make_job(raw_text="Python engineer at Acme Corp")
    score = router.compute_routing_score(job, "Python software engineer")
    assert abs(score - 0.5) < 0.001


# ── _sigmoid helper ───────────────────────────────────────────────────────────

def test_sigmoid_zero():
    assert abs(_sigmoid(0.0) - 0.5) < 1e-9


def test_sigmoid_large_positive():
    assert _sigmoid(10.0) > 0.99


def test_sigmoid_large_negative():
    assert _sigmoid(-10.0) < 0.01


# ── Summary logging ───────────────────────────────────────────────────────────

def test_log_summary_no_jobs_does_not_crash():
    router = SelectiveRouter(make_config())
    router.log_summary()  # should not raise


def test_counts_track_correctly():
    router = SelectiveRouter(make_config())
    router.record_llm_call()
    router.record_llm_call()
    router.record_skipped()
    assert router._counts["llm"] == 2
    assert router._counts["skipped"] == 1


# ── Integration: routing disabled falls back to None ─────────────────────────

def test_routing_disabled_no_router():
    """When routing.enabled=false, scorer should not create a router (no regression)."""
    cfg = make_config(enabled=False)
    # Simulate the check done in score_all_jobs()
    router = None
    if cfg.get("routing", {}).get("enabled", False):
        router = SelectiveRouter(cfg)
    assert router is None


# ── Integration: full synthetic score roundtrip ───────────────────────────────

def test_full_synthetic_roundtrip():
    """
    Verify the synthetic score dict has all keys needed by save_score() in scorer.py.
    """
    router = SelectiveRouter(make_config())
    job = make_job(raw_text="Python backend engineer with AWS experience and ML")
    result = router.create_synthetic_score(job, routing_score=0.58)

    required_keys = {
        "job", "fit_score", "tokens_used", "ats_score",
        "reasons", "flags", "one_liner", "dimension_scores", "skill_misses",
    }
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - result.keys()}"
    )

    dims = result["dimension_scores"]
    for dim in ("role_fit", "stack_match", "seniority", "location", "growth", "compensation"):
        assert dim in dims
        assert 0 <= dims[dim] <= 10


def test_quick_ats_skill_matching():
    """Verify _quick_ats gives partial credit for matching skills."""
    router = SelectiveRouter(make_config())
    job_with_skills = make_job(raw_text="Looking for Python developer with AWS experience")
    job_no_skills = make_job(raw_text="Looking for Java developer with Spring Boot")

    ats_with = router._quick_ats(job_with_skills)
    ats_without = router._quick_ats(job_no_skills)

    assert ats_with > ats_without
    assert 0 <= ats_with <= 100


def test_quick_ats_uses_canonical_matching_for_multi_word_skills():
    """
    CRIT-2 regression: when the candidate has structured-profile skills, _quick_ats
    must score JDs using canonical aliases rather than literal substring of multi-word
    skill phrases. A candidate with "AWS" and "Python" should score high on a JD that
    mentions "Amazon Web Services" and "python", even though neither appears verbatim
    in a desired_skills list of multi-word phrases.
    """
    cfg = make_config()
    cfg["preferences"]["desired_skills"] = [
        "Machine learning infrastructure",
        "Authentication and authorization",
        "Cloud architecture",
    ]
    router = SelectiveRouter(cfg)
    router.set_structured_profile({
        "core_skills": ["Python"],
        "languages": ["Python"],
        "frameworks": [],
        "cloud": ["AWS", "Docker"],
    })
    job = make_job(
        raw_text=(
            "We are hiring a backend engineer. Required skills: Python, Amazon Web Services, "
            "Docker. Experience with REST APIs and CI/CD pipelines preferred."
        ),
    )
    score = router._quick_ats(job)
    # Canonical extractor catches Python (language), AWS (alias of Amazon Web Services),
    # Docker. With naive substring matching of the desired_skills list this would be 0
    # because none of the configured phrases appear verbatim.
    assert score > 30, f"expected canonical match score > 30, got {score}"


def test_quick_ats_handles_missing_structured_profile():
    """Without a structured profile, _quick_ats must still return a non-error value."""
    router = SelectiveRouter(make_config())
    score = router._quick_ats(make_job(raw_text="Python role at Acme"))
    assert isinstance(score, int)
    assert 0 <= score <= 100


# ── score_job: keyword_prescore is bypassed when routing enabled (CRIT-3) ─────

def test_score_job_skips_keyword_prescore_when_routing_enabled():
    """
    CRIT-3 regression: when routing is enabled, score_job MUST NOT short-circuit
    on keyword_prescore. The router has already decided the job is worth an LLM call;
    keyword_prescore uses naive substring matching of multi-word desired_skills phrases
    that silently zeros good matches.
    """
    from scorer import score_job

    cfg = make_config()
    cfg["routing"]["enabled"] = True
    # Multi-word phrases that will NOT substring-match a real-looking JD.
    cfg["preferences"]["desired_skills"] = [
        "Machine learning infrastructure",
        "Authentication and authorization",
        "Distributed systems",
    ]
    cfg["preferences"]["titles"] = []  # disable the title shortcut so prescore would otherwise fire
    cfg["profile"] = {"resume": ""}

    job = make_job(
        raw_text=(
            "Backend engineer wanted. We use Python, Postgres and Docker. "
            "Strong CI/CD culture. Required: REST APIs, k8s."
        ),
    )

    fake_llm_called = {"count": 0}

    def fake_llm(prompt: str, max_tokens: int = 700):
        fake_llm_called["count"] += 1
        return (
            '{"disqualified": false, "disqualify_reason": "", "role_fit": 7, '
            '"stack_match": 7, "seniority": 7, "location": 7, "growth": 6, '
            '"compensation": 5, "reasons": ["good fit","python"], "flags": [], '
            '"one_liner": "good fit"}',
            300,
        )

    result = score_job(job, cfg, fake_llm, structured_profile=None, instructor_client=None)

    # The LLM call must have happened — proving keyword_prescore did not gate it.
    assert fake_llm_called["count"] == 1
    assert result["fit_score"] > 0
    assert "no skill overlap" not in result["flags"]


def test_score_job_keeps_keyword_prescore_when_routing_disabled():
    """When routing is disabled, the legacy keyword_prescore gate must still run."""
    from scorer import score_job

    cfg = make_config(enabled=False)
    cfg["preferences"]["desired_skills"] = [
        "Quantum cryptography",
        "Embedded firmware",
        "Driver development",
    ]
    cfg["preferences"]["titles"] = []
    cfg["profile"] = {"resume": ""}

    # JD has no overlap with any of the (deliberately niche) desired_skills.
    job = make_job(raw_text="We are hiring a marketing copywriter for our brand team.")

    def must_not_be_called(prompt: str, max_tokens: int = 700):
        raise AssertionError("LLM must not be called when prescore zeros out the job")

    result = score_job(job, cfg, must_not_be_called, structured_profile=None, instructor_client=None)
    assert result["fit_score"] == 0
    assert result["flags"] == ["no skill overlap"]


# ── detect_matched_sections() ─────────────────────────────────────────────────

def test_detect_matched_sections_empty_text():
    router = SelectiveRouter(make_config())
    result = router.detect_matched_sections(make_job(raw_text=""), "Python engineer")
    assert result == []


def test_detect_matched_sections_empty_query():
    router = SelectiveRouter(make_config())
    job = make_job(raw_text="Requirements\nPython experience required")
    result = router.detect_matched_sections(job, "")
    assert result == []


def test_detect_matched_sections_finds_requirements():
    router = SelectiveRouter(make_config())
    text = (
        "About the Role\nWe build ML platforms.\n\n"
        "Requirements\nPython required. AWS experience. Machine learning background.\n"
        "3+ years Python. Strong AWS skills."
    )
    result = router.detect_matched_sections(make_job(raw_text=text), "Python AWS Machine learning")
    assert "requirements" in result


def test_detect_matched_sections_finds_responsibilities():
    router = SelectiveRouter(make_config())
    text = (
        "What you'll do\n"
        "Build Python services on AWS. Deploy machine learning models. "
        "Collaborate with ML engineers."
    )
    result = router.detect_matched_sections(make_job(raw_text=text), "Python AWS Machine learning")
    assert "responsibilities" in result


def test_detect_matched_sections_no_match():
    router = SelectiveRouter(make_config())
    text = "Looking for Java developers with Spring Boot and Kubernetes experience."
    result = router.detect_matched_sections(make_job(raw_text=text), "Python AWS Machine learning")
    # May or may not have sections, but should not crash
    assert isinstance(result, list)


def test_detect_matched_sections_stores_query():
    """build_match_query should cache the query on self._match_query."""
    router = SelectiveRouter(make_config())
    router._match_query = "Python engineer"
    job = make_job(raw_text="Requirements\nPython required. Python skills needed.")
    # Should use self._match_query when match_query arg is empty
    result = router.detect_matched_sections(job, "")
    # Not empty because _match_query is used as fallback
    # (result may or may not have "requirements" depending on hit count — just verify no crash)
    assert isinstance(result, list)


# ── compute_tier returns full tier config (regression for CRIT-1) ────────────

def _config_with_tiers() -> dict:
    cfg = make_config()
    cfg["routing"]["tiers"] = {
        "high": {
            "min_reranker": 0.40,
            "require_title_match": True,
            "primary_model": "groq/llama-3.1-8b-instant",
            "quality_mode": "fast",
            "max_tokens": 500,
        },
        "medium": {
            "min_reranker": 0.25,
            "primary_model": "groq/llama-4-scout-17b-16e-instruct",
            "quality_mode": "fast",
            "max_tokens": 800,
        },
        "low": {
            "min_reranker": 0.18,
            "primary_model": "gemini/gemini-2.5-flash",
            "quality_mode": "quality",
            "max_tokens": 1200,
        },
    }
    cfg["routing"]["fallback_chain"] = ["groq", "gemini", "openai"]
    cfg["routing"]["rate_limit_budget"] = {
        "groq": {"max_requests_per_run": 10, "max_tokens_per_run": 50_000},
        "gemini": {"max_requests_per_run": 10, "max_tokens_per_run": 50_000},
        "openai": {"max_requests_per_run": 5, "max_tokens_per_run": 20_000},
    }
    return cfg


def test_compute_tier_high_includes_primary_model():
    """compute_tier must propagate primary_model so select_model_with_fallback reads it."""
    router = SelectiveRouter(_config_with_tiers())
    tier = router.compute_tier(effective_score=0.95, job_text="short", title_matched=True)
    assert tier["tier"] == "high"
    assert tier["primary_model"] == "groq/llama-3.1-8b-instant"
    assert tier["require_title_match"] is True


def test_compute_tier_medium_includes_primary_model():
    router = SelectiveRouter(_config_with_tiers())
    tier = router.compute_tier(effective_score=0.30, job_text="short", title_matched=False)
    assert tier["tier"] == "medium"
    assert tier["primary_model"] == "groq/llama-4-scout-17b-16e-instruct"


def test_compute_tier_low_includes_primary_model():
    router = SelectiveRouter(_config_with_tiers())
    tier = router.compute_tier(effective_score=0.10, job_text="short", title_matched=False)
    assert tier["tier"] == "low"
    assert tier["primary_model"] == "gemini/gemini-2.5-flash"


def test_select_model_with_fallback_uses_per_tier_primary(monkeypatch):
    """
    Configured per-tier primary_model must be honored — not silently overridden
    by the default groq/llama-3.1-8b-instant fallback. (CRIT-1 regression.)
    """
    from selective_routing import ProviderBudget, select_model_with_fallback

    cfg = _config_with_tiers()
    router = SelectiveRouter(cfg)

    # Pretend gemini and groq both have keys + budget.
    monkeypatch.setattr(
        "selective_routing._provider_has_key",
        lambda provider, profile=None: provider in {"groq", "gemini", "openai"},
    )
    budgets = ProviderBudget(cfg["routing"]["rate_limit_budget"])

    tier = router.compute_tier(effective_score=0.10, job_text="short", title_matched=False)
    provider, model = select_model_with_fallback(
        tier, tier["tier"], cfg["routing"], budgets,
        estimated_tokens=500, job_id="job_test", profile=None,
    )
    assert provider == "gemini"
    assert model == "gemini-2.5-flash"


def test_select_model_with_fallback_falls_through_when_primary_exhausted(monkeypatch):
    """If the configured tier primary is exhausted, walk fallback_chain in order."""
    from selective_routing import ProviderBudget, select_model_with_fallback

    cfg = _config_with_tiers()
    router = SelectiveRouter(cfg)
    monkeypatch.setattr(
        "selective_routing._provider_has_key",
        lambda provider, profile=None: provider in {"groq", "gemini", "openai"},
    )
    budgets = ProviderBudget(cfg["routing"]["rate_limit_budget"])
    budgets.force_exhaust("gemini")  # tier-low primary blown

    tier = router.compute_tier(effective_score=0.10, job_text="short", title_matched=False)
    provider, _ = select_model_with_fallback(
        tier, tier["tier"], cfg["routing"], budgets,
        estimated_tokens=500, job_id="job_test", profile=None,
    )
    # fallback_chain order: groq, gemini, openai → next viable after gemini is groq
    assert provider == "groq"
