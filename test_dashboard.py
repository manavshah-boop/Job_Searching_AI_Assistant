import dashboard
import dashboard_semantic
from progress_tracker import ProgressTracker
from worker import run_pipeline


def test_run_pipeline_delegates_to_shared_pipeline(monkeypatch):
    calls = []
    config = {"sources": {"greenhouse": {"enabled": True}}}

    monkeypatch.setattr(run_pipeline, "load_config", lambda profile=None: config)
    monkeypatch.setattr(run_pipeline, "init_db", lambda profile=None: calls.append(("init_db", profile)))
    monkeypatch.setattr(
        run_pipeline,
        "run_full_pipeline",
        lambda config, profile, options, progress_tracker=None: calls.append(
            (
                "run_full_pipeline",
                profile,
                options.scrape,
                options.score,
                options.embed,
                options.yes,
                options.run_source,
                progress_tracker is not None,
            )
        ) or type("Result", (), {"score": type("Score", (), {"jobs_scored": 3})()})(),
    )

    result = run_pipeline._run("default", ProgressTracker())

    assert result == 3
    assert calls == [
        ("init_db", "default"),
        ("run_full_pipeline", "default", True, True, True, True, "worker", True),
    ]


def test_score_all_jobs_uses_profile_for_attempt_and_error_writes(monkeypatch):
    import scorer
    from db import Job

    calls = []
    job = Job(
        id="job-1",
        title="Software Engineer",
        company="Acme",
        location="Remote",
        url="https://example.com/job-1",
        raw_text="Python AWS backend",
        source="greenhouse",
    )

    monkeypatch.setattr(scorer, "get_llm_client", lambda config: object())
    monkeypatch.setattr(scorer, "get_unscored", lambda profile=None: [job])
    monkeypatch.setattr(scorer, "build_structured_profile", lambda config, llm_call: {"name": "Test"})
    monkeypatch.setattr(scorer, "print_profile_summary", lambda profile: None)
    monkeypatch.setattr(scorer.RateLimiter, "wait_if_needed", lambda self: None)
    monkeypatch.setattr(
        scorer,
        "increment_score_attempts",
        lambda job_id, profile=None: calls.append(("attempt", job_id, profile)),
    )
    monkeypatch.setattr(
        scorer,
        "score_job",
        lambda job, config, llm_call, structured_profile=None, instructor_client=None: (
            _ for _ in ()
        ).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        scorer,
        "write_score_error",
        lambda job_id, error, profile=None: calls.append(("error", job_id, profile, error)),
    )

    config = {
        "llm": {
            "provider": "groq",
            "model": {"groq": "test-model"},
            "rate_limits": {"groq": {"max_rpm": 1, "max_tpm": 1000}},
        },
        "profile": {"resume": "resume text", "bio": "bio"},
        "preferences": {
            "desired_skills": ["Python"],
            "titles": ["Software Engineer"],
            "location": {"remote_ok": True, "preferred_locations": []},
            "compensation": {"min_salary": 100000},
            "yoe": 3,
        },
        "scoring": {
            "min_display_score": 60,
            "weights": {
                "role_fit": 0.3,
                "stack_match": 0.25,
                "seniority": 0.2,
                "location": 0.1,
                "growth": 0.1,
                "compensation": 0.05,
            },
        },
    }

    scorer.score_all_jobs(config, yes=True, profile="default")

    assert calls == [
        ("attempt", "job-1", "default"),
        ("error", "job-1", "default", "boom"),
    ]


def test_list_profiles_reads_names_and_sorts_by_updated_time(monkeypatch):
    from pathlib import Path
    import shutil

    root = Path(".tmp_streamlit_test") / "dashboard_profiles"
    if root.exists():
        shutil.rmtree(root)

    profiles_dir = root / "profiles"
    alpha = profiles_dir / "alpha"
    beta = profiles_dir / "beta"
    alpha.mkdir(parents=True)
    beta.mkdir(parents=True)

    (alpha / "config.yaml").write_text(
        "profile:\n  name: Alpha Profile\n  job_type: fulltime\nllm:\n  provider: groq\n",
        encoding="utf-8",
    )
    (beta / "config.yaml").write_text(
        "profile:\n  name: Beta Profile\n  job_type: internship\nllm:\n  provider: gemini\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(dashboard, "PROFILES_DIR", profiles_dir)
    monkeypatch.setattr(
        dashboard,
        "_safe_count_jobs",
        lambda slug: {"alpha": {"total": 3, "scored": 1}, "beta": {"total": 8, "scored": 5}}[slug],
    )

    alpha_time = (alpha / "config.yaml").stat().st_mtime
    beta_time = alpha_time + 10
    import os

    os.utime(beta / "config.yaml", (beta_time, beta_time))

    profiles = dashboard.list_profiles()

    assert [profile["slug"] for profile in profiles] == ["beta", "alpha"]
    assert profiles[0]["name"] == "Beta Profile"
    assert profiles[0]["counts"] == {"total": 8, "scored": 5}

    shutil.rmtree(root)


def test_dashboard_invalidate_caches_clears_semantic_panel_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(dashboard, "clear_semantic_panel_caches", lambda: calls.append("semantic"))
    monkeypatch.setattr(dashboard._cached_list_profiles, "clear", lambda: calls.append("profiles"))
    monkeypatch.setattr(dashboard._cached_fetch_job_summaries, "clear", lambda: calls.append("summaries"))
    monkeypatch.setattr(dashboard._cached_fetch_job_detail, "clear", lambda: calls.append("detail"))
    monkeypatch.setattr(dashboard._cached_recent_runs, "clear", lambda: calls.append("runs"))

    dashboard.invalidate_dashboard_caches()

    assert calls == ["profiles", "summaries", "detail", "runs", "semantic"]


def test_render_semantic_match_panel_handles_disabled_vector_store(monkeypatch):
    captions = []
    monkeypatch.setattr(dashboard_semantic, "vector_store_enabled", lambda config: False)
    monkeypatch.setattr(dashboard_semantic.st, "caption", lambda message: captions.append(message))

    dashboard_semantic.render_semantic_match_panel("alpha", {})

    assert captions == ["Semantic retrieval is disabled for this profile."]
