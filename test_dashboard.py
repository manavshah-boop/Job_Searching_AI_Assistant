import dashboard


def test_run_pipeline_passes_profile_to_all_profile_scoped_calls(monkeypatch):
    calls = []

    monkeypatch.setattr(
        dashboard,
        "get_or_discover_slugs",
        lambda config, profile=None: calls.append(("slugs", profile)) or {
            "greenhouse": ["gh-co"],
            "lever": ["lv-co"],
        },
    )
    monkeypatch.setattr(
        dashboard,
        "scrape_greenhouse",
        lambda config, slugs=None, profile=None: calls.append(("greenhouse", tuple(slugs or ()), profile)) or {
            "new_jobs_saved": 2,
        },
    )
    monkeypatch.setattr(
        dashboard,
        "scrape_lever",
        lambda config, slugs=None, profile=None: calls.append(("lever", tuple(slugs or ()), profile)) or {
            "new_jobs_saved": 3,
        },
    )
    monkeypatch.setattr(
        dashboard,
        "scrape_hn",
        lambda config, profile=None: calls.append(("hn", profile)) or {
            "new_jobs_saved": 4,
        },
    )
    monkeypatch.setattr(
        dashboard,
        "score_all_jobs",
        lambda config, yes=False, profile=None: calls.append(("score", yes, profile)) or [
            {"fit_score": 80},
            {"fit_score": 0},
        ],
    )
    monkeypatch.setattr(dashboard.st, "write", lambda *args, **kwargs: None)
    monkeypatch.setattr(dashboard.st, "warning", lambda *args, **kwargs: None)

    config = {
        "sources": {
            "greenhouse": {"enabled": True},
            "lever": {"enabled": True},
            "hn": {"enabled": True},
        }
    }

    result = dashboard._run_pipeline(config, slug="default")

    assert result == {
        "total_new": 9,
        "jobs_scraped": 0,
        "jobs_filtered": 0,
        "jobs_saved": 9,
        "scored_count": 1,
        "avg_fit": 80.0,
    }
    assert calls == [
        ("slugs", "default"),
        ("greenhouse", ("gh-co",), "default"),
        ("lever", ("lv-co",), "default"),
        ("hn", "default"),
        ("score", True, "default"),
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
