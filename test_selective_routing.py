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
    cfg = make_config()
    cfg["routing"]["threshold"] = "not_a_number"
    router = SelectiveRouter(cfg)
    assert router.threshold == 0.65


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
    # At routing_score=0.80, role_fit should be 0.80 * 0.85 * 10 = 6.8 → 7, not 8+
    assert dims["role_fit"] <= 7
    assert dims["stack_match"] <= 7


def test_create_synthetic_score_fit_range():
    router = SelectiveRouter(make_config())
    job = make_job()

    for score in [0.0, 0.3, 0.5, 0.65, 0.80, 1.0]:
        result = router.create_synthetic_score(job, routing_score=score)
        assert 0 <= result["fit_score"] <= 100


def test_create_synthetic_score_flag():
    router = SelectiveRouter(make_config())
    result = router.create_synthetic_score(make_job(), 0.60)
    assert "score_source:reranker" in result["flags"]


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


def test_create_synthetic_score_match_reason_in_one_liner():
    router = SelectiveRouter(make_config(threshold=0.65))
    result = router.create_synthetic_score(make_job(), 0.60)
    # one_liner should reference routing score
    assert "0.60" in result["one_liner"] or "reranker" in result["one_liner"].lower()


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
