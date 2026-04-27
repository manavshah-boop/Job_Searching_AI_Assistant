from __future__ import annotations

import argparse

import main


def test_main_cli_delegates_pipeline_modes(monkeypatch):
    calls = []
    args = argparse.Namespace(
        scrape_only=False,
        score_only=True,
        embed_only=False,
        vector_search=None,
        semantic_search=None,
        semantic_match=False,
        rebuild_vector_index=False,
        clear_vector_index=False,
        show=False,
        rescore=False,
        yes=True,
        min_score=None,
        rerank=False,
        top_k=None,
        profile="alpha",
        debug=False,
    )
    config = {
        "llm": {"provider": "groq", "model": {"groq": "test-model"}},
        "scoring": {"min_display_score": 60},
        "sources": {"greenhouse": {"enabled": True}},
        "embeddings": {"enabled": True},
    }

    monkeypatch.setattr(main, "parse_args", lambda: args)
    monkeypatch.setattr(main, "_handle_profile_arg", lambda profile: calls.append(("profile", profile)))
    monkeypatch.setattr(main, "load_config", lambda profile=None: dict(config))
    monkeypatch.setattr(main, "init_db", lambda profile=None: calls.append(("init_db", profile)))
    monkeypatch.setattr(main, "configure_logging", lambda profile, debug=False: calls.append(("logging", profile, debug)))
    monkeypatch.setattr(main, "_print_banner", lambda config: calls.append(("banner", config["llm"]["provider"])))
    monkeypatch.setattr(
        main,
        "run_full_pipeline",
        lambda config, profile, options: calls.append(
            ("pipeline", profile, options.scrape, options.score, options.embed, options.yes, options.run_source)
        ) or type(
            "Result",
            (),
            {
                "score": type("Score", (), {"results": [], "jobs_scored": 0, "avg_fit_score": 0.0, "avg_ats_score": 0.0})(),
                "scrape": type("Scrape", (), {"total_new": 0})(),
                "embed": type("Embed", (), {"jobs_total": 0, "vector_index": {}})(),
                "final_db_stats": {"total": 1, "scored": 1, "embedded": 0},
            },
        )(),
    )

    main.main()

    assert calls == [
        ("profile", "alpha"),
        ("init_db", "alpha"),
        ("logging", "alpha", False),
        ("banner", "groq"),
        ("pipeline", "alpha", False, True, True, True, "cli"),
    ]
