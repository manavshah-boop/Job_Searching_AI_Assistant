from __future__ import annotations

import argparse
from pathlib import Path

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
        eval=False,
        export_eval_template=False,
        show=False,
        rescore=False,
        yes=True,
        min_score=None,
        rerank=False,
        no_rerank=False,
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


def test_main_cli_runs_eval_mode(monkeypatch, tmp_path: Path):
    calls = []
    labels_path = tmp_path / "profiles" / "alpha" / "eval_labels.yaml"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text("labels: []\n", encoding="utf-8")

    args = argparse.Namespace(
        scrape_only=False,
        score_only=False,
        embed_only=False,
        vector_search=None,
        semantic_search=None,
        semantic_match=False,
        rebuild_vector_index=False,
        clear_vector_index=False,
        eval=True,
        export_eval_template=False,
        show=False,
        rescore=False,
        yes=True,
        min_score=None,
        rerank=False,
        no_rerank=True,
        top_k=None,
        profile="alpha",
        debug=False,
    )
    config = {
        "llm": {"provider": "groq", "model": {"groq": "test-model"}},
        "scoring": {"min_display_score": 60},
        "sources": {"greenhouse": {"enabled": True}},
        "embeddings": {"vector_store": {"enabled": True}},
    }

    monkeypatch.setattr(main, "parse_args", lambda: args)
    monkeypatch.setattr(main, "_handle_profile_arg", lambda profile: None)
    monkeypatch.setattr(main, "load_config", lambda profile=None: dict(config))
    monkeypatch.setattr(main, "init_db", lambda profile=None: None)
    monkeypatch.setattr(main, "configure_logging", lambda profile, debug=False: None)
    monkeypatch.setattr(main, "_print_banner", lambda config: calls.append("banner"))
    monkeypatch.setattr(main, "_default_eval_labels_path", lambda profile=None: labels_path)
    monkeypatch.setattr(
        main,
        "evaluate_profile",
        lambda profile_slug, config, labels_path, use_reranker=True: calls.append(
            ("evaluate", profile_slug, str(labels_path), use_reranker)
        ) or type(
            "EvalResult",
            (),
            {
                "role_family": "software_engineering",
                "total_labeled": 0,
                "precision_at_5": 0.0,
                "precision_at_10": 0.0,
                "recall_at_10": 0.0,
                "mrr": 0.0,
                "ndcg_at_10": 0.0,
                "coverage": 0.0,
                "notes": "",
            },
        )(),
    )
    monkeypatch.setattr(main, "_print_eval_summary", lambda *args, **kwargs: calls.append("summary"))
    monkeypatch.setattr(
        main,
        "safe_log_eval_result",
        lambda config, profile, result, extra_params=None: calls.append(
            ("track_eval", profile, extra_params["use_reranker"], extra_params["labels_path"])
        ),
    )

    main.main()

    assert calls == [
        "banner",
        ("evaluate", "alpha", str(labels_path), False),
        "summary",
        ("track_eval", "alpha", False, str(labels_path)),
    ]


def test_main_cli_exports_eval_template(monkeypatch, tmp_path: Path):
    calls = []
    labels_path = tmp_path / "profiles" / "alpha" / "eval_labels.yaml"

    args = argparse.Namespace(
        scrape_only=False,
        score_only=False,
        embed_only=False,
        vector_search=None,
        semantic_search=None,
        semantic_match=False,
        rebuild_vector_index=False,
        clear_vector_index=False,
        eval=False,
        export_eval_template=True,
        show=False,
        rescore=False,
        yes=True,
        min_score=None,
        rerank=False,
        no_rerank=False,
        top_k=12,
        profile="alpha",
        debug=False,
    )
    config = {
        "llm": {"provider": "groq", "model": {"groq": "test-model"}},
        "scoring": {"min_display_score": 60},
        "sources": {"greenhouse": {"enabled": True}},
        "embeddings": {"vector_store": {"enabled": True}},
    }

    monkeypatch.setattr(main, "parse_args", lambda: args)
    monkeypatch.setattr(main, "_handle_profile_arg", lambda profile: None)
    monkeypatch.setattr(main, "load_config", lambda profile=None: dict(config))
    monkeypatch.setattr(main, "init_db", lambda profile=None: None)
    monkeypatch.setattr(main, "configure_logging", lambda profile, debug=False: None)
    monkeypatch.setattr(main, "_print_banner", lambda config: calls.append("banner"))
    monkeypatch.setattr(main, "_default_eval_labels_path", lambda profile=None: labels_path)
    monkeypatch.setattr(
        main,
        "export_eval_template",
        lambda profile_slug, config, path, use_reranker=True, limit=20: calls.append(
            ("export", profile_slug, str(path), use_reranker, limit)
        ) or [],
    )

    main.main()

    assert calls == [
        "banner",
        ("export", "alpha", str(labels_path), True, 12),
    ]
