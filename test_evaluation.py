from __future__ import annotations

from pathlib import Path

import pytest

import evaluation
from evaluation import (
    EvalLabel,
    compare_eval_runs,
    evaluate_profile,
    evaluate_ranked_results,
    export_eval_template,
    label_to_relevance,
    load_eval_labels,
    save_eval_labels,
)


def _result(job_id: str):
    return type("Result", (), {"job_id": job_id})()


def _software_config() -> dict:
    return {
        "profile": {
            "name": "Manav Shah",
            "job_type": "fulltime",
            "resume": "Python FastAPI SQL AWS backend APIs",
            "bio": "Backend engineer",
        },
        "preferences": {
            "titles": ["Backend Engineer", "Software Engineer"],
            "desired_skills": ["Python", "SQL", "AWS"],
            "hard_no_keywords": [],
            "location": {"remote_ok": True, "preferred_locations": ["Seattle, WA"]},
            "compensation": {"min_salary": 130000},
            "yoe": 3,
            "filters": {"title_blocklist": ["Senior", "Staff"]},
        },
        "embeddings": {"vector_store": {"enabled": True, "top_k_chunks": 50, "top_k_jobs": 30}},
        "reranking": {"enabled": True, "top_k_final": 15, "top_k_vector_jobs": 40, "top_k_vector_chunks": 80},
    }


def _finance_config() -> dict:
    return {
        "profile": {
            "name": "Alex Chen",
            "job_type": "internship",
            "resume": "Excel financial modeling budgeting variance analysis",
            "bio": "Finance internship candidate",
        },
        "preferences": {
            "titles": ["Finance Intern", "FP&A Intern"],
            "desired_skills": ["Excel", "financial modeling", "budgeting"],
            "hard_no_keywords": [],
            "location": {"remote_ok": True, "preferred_locations": ["New York, NY"]},
            "compensation": {"intern_pay_preference": "paid_only", "monthly_stipend": 4500},
            "yoe": 0,
            "filters": {"title_blocklist": ["Senior", "Director"]},
        },
        "embeddings": {"vector_store": {"enabled": True, "top_k_chunks": 50, "top_k_jobs": 30}},
        "reranking": {"enabled": True, "top_k_final": 15, "top_k_vector_jobs": 40, "top_k_vector_chunks": 80},
    }


def _marketing_config() -> dict:
    return {
        "profile": {
            "name": "Jordan Lee",
            "job_type": "fulltime",
            "resume": "SEO HubSpot lifecycle marketing email campaigns",
            "bio": "Marketing specialist",
        },
        "preferences": {
            "titles": ["Growth Marketing Manager"],
            "desired_skills": ["SEO", "HubSpot", "email campaigns"],
            "hard_no_keywords": [],
            "location": {"remote_ok": True, "preferred_locations": []},
            "compensation": {"min_salary": 75000},
            "yoe": 2,
            "filters": {"title_blocklist": ["Director"]},
        },
        "embeddings": {"vector_store": {"enabled": True, "top_k_chunks": 50, "top_k_jobs": 30}},
        "reranking": {"enabled": True, "top_k_final": 15, "top_k_vector_jobs": 40, "top_k_vector_chunks": 80},
    }


def _general_config() -> dict:
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
        "embeddings": {"vector_store": {"enabled": True, "top_k_chunks": 50, "top_k_jobs": 30}},
        "reranking": {"enabled": True, "top_k_final": 15, "top_k_vector_jobs": 40, "top_k_vector_chunks": 80},
    }


def test_label_to_relevance_mapping():
    assert label_to_relevance("great_match") == 4
    assert label_to_relevance("good_match") == 3
    assert label_to_relevance("okay_match") == 2
    assert label_to_relevance("bad_match") == 1
    assert label_to_relevance("should_skip") == 0


def test_evaluate_ranked_results_metrics():
    labels = [
        EvalLabel("alpha", "job-a", "great_match", role_family="software_engineering"),
        EvalLabel("alpha", "job-b", "good_match", role_family="software_engineering"),
        EvalLabel("alpha", "job-c", "okay_match", role_family="software_engineering"),
        EvalLabel("alpha", "job-d", "should_skip", role_family="software_engineering"),
    ]
    ranked = ["job-c", "job-b", "job-x", "job-a", "job-y", "job-z", "job-d", "job-q", "job-r", "job-s"]

    result = evaluate_ranked_results(labels, ranked)

    assert result.total_labeled == 4
    assert result.precision_at_5 == pytest.approx(0.4)
    assert result.precision_at_10 == pytest.approx(0.2)
    assert result.recall_at_10 == pytest.approx(1.0)
    assert result.mrr == pytest.approx(0.5)
    assert result.coverage == pytest.approx(1.0)
    assert result.ndcg_at_10 == pytest.approx(0.6634, rel=1e-3)


def test_ndcg_is_one_for_perfect_ranking():
    labels = [
        EvalLabel("alpha", "job-a", "great_match"),
        EvalLabel("alpha", "job-b", "good_match"),
        EvalLabel("alpha", "job-c", "okay_match"),
    ]

    result = evaluate_ranked_results(labels, ["job-a", "job-b", "job-c"])

    assert result.ndcg_at_10 == pytest.approx(1.0)


def test_empty_labels_return_zero_metrics():
    result = evaluate_ranked_results([], ["job-a", "job-b"])

    assert result.total_labeled == 0
    assert result.precision_at_5 == 0.0
    assert result.precision_at_10 == 0.0
    assert result.recall_at_10 == 0.0
    assert result.mrr == 0.0
    assert result.ndcg_at_10 == 0.0
    assert result.coverage == 0.0


def test_no_relevant_labels_return_zero_precision_recall_and_mrr():
    labels = [
        EvalLabel("alpha", "job-a", "okay_match"),
        EvalLabel("alpha", "job-b", "bad_match"),
        EvalLabel("alpha", "job-c", "should_skip"),
    ]

    result = evaluate_ranked_results(labels, ["job-a", "job-b", "job-c"])

    assert result.precision_at_5 == 0.0
    assert result.precision_at_10 == 0.0
    assert result.recall_at_10 == 0.0
    assert result.mrr == 0.0


def test_duplicate_job_ids_are_deduped_safely():
    labels = [
        EvalLabel("alpha", "job-a", "bad_match"),
        EvalLabel("alpha", "job-a", "great_match"),
    ]

    result = evaluate_ranked_results(labels, ["job-a"])

    assert result.total_labeled == 1
    assert result.mrr == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("config_factory", "expected_role_family"),
    [
        (_software_config, "software_engineering"),
        (_finance_config, "finance"),
        (_marketing_config, "marketing"),
        (_general_config, "general"),
    ],
)
def test_evaluate_profile_works_across_role_families(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_factory, expected_role_family: str):
    profile_slug = f"profile_{expected_role_family}"
    labels_path = tmp_path / profile_slug / "eval_labels.yaml"
    save_eval_labels(
        [
            EvalLabel(profile_slug, "job-1", "great_match"),
            EvalLabel(profile_slug, "job-2", "good_match"),
            EvalLabel(profile_slug, "job-3", "should_skip"),
        ],
        labels_path,
    )
    monkeypatch.setattr(evaluation, "semantic_match_jobs", lambda *args, **kwargs: [_result("job-2"), _result("job-1"), _result("job-9")])

    result = evaluate_profile(profile_slug, config_factory(), labels_path, use_reranker=True)

    assert result.role_family == expected_role_family
    assert result.precision_at_5 == pytest.approx(0.4)
    assert result.recall_at_10 == pytest.approx(1.0)


def test_evaluate_profile_without_reranker_uses_vector_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    profile_slug = "vector_only"
    labels_path = tmp_path / profile_slug / "eval_labels.yaml"
    save_eval_labels([EvalLabel(profile_slug, "job-1", "great_match")], labels_path)
    monkeypatch.setattr(evaluation, "query_similar_jobs", lambda *args, **kwargs: [_result("job-1"), _result("job-2")])

    result = evaluate_profile(profile_slug, _software_config(), labels_path, use_reranker=False)

    assert result.mrr == pytest.approx(1.0)
    assert result.coverage == pytest.approx(1.0)


def test_save_and_load_eval_labels_yaml_and_json(tmp_path: Path):
    labels = [
        EvalLabel("alpha", "job-a", "great_match", notes="Strong fit", role_family="software_engineering"),
        EvalLabel("alpha", "job-b", "should_skip", notes="Too senior"),
    ]

    yaml_path = tmp_path / "alpha" / "eval_labels.yaml"
    json_path = tmp_path / "alpha" / "eval_labels.json"
    save_eval_labels(labels, yaml_path)
    save_eval_labels(labels, json_path)

    assert load_eval_labels(yaml_path) == labels
    assert load_eval_labels(json_path) == labels


def test_export_template_produces_valid_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    profile_slug = "alpha"
    path = tmp_path / profile_slug / "eval_labels.yaml"
    monkeypatch.setattr(evaluation, "semantic_match_jobs", lambda *args, **kwargs: [_result("job-1"), _result("job-2")])

    exported = export_eval_template(profile_slug, _software_config(), path, use_reranker=True, limit=2)
    loaded = load_eval_labels(path)

    assert [label.job_id for label in exported] == ["job-1", "job-2"]
    assert loaded == exported
    assert all(label.label == "okay_match" for label in loaded)


def test_compare_eval_runs_reports_deltas():
    previous = evaluate_ranked_results([EvalLabel("alpha", "job-a", "great_match")], ["job-b", "job-a"])
    current = evaluate_ranked_results([EvalLabel("alpha", "job-a", "great_match")], ["job-a"])

    diff = compare_eval_runs(previous, current)

    assert diff["delta"]["mrr"] > 0
    assert diff["delta"]["precision_at_5"] >= 0
