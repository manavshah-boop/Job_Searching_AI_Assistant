from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import config as config_module
import tracking


def _enabled_config() -> dict:
    return {
        "tracking": {
            "mlflow": {
                "enabled": True,
                "tracking_uri": "./mlruns",
                "experiment_name": "job-agent",
                "log_artifacts": True,
                "log_eval_labels": False,
            }
        },
        "embeddings": {
            "model": "embed-model",
            "vector_store": {"enabled": True, "top_k_chunks": 50, "top_k_jobs": 30},
        },
        "reranking": {"enabled": True, "model": "rerank-model", "top_k_final": 15},
        "llm": {"provider": "groq", "model": {"groq": "llama-test"}},
        "sources": {"greenhouse": {"enabled": True}, "lever": {"enabled": False}},
    }


def test_disabled_tracking_no_ops(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(tracking, "_get_mlflow", lambda: calls.append("mlflow") or object())

    disabled = {"tracking": {"mlflow": {"enabled": False}}}

    assert tracking.mlflow_enabled(disabled) is False
    assert tracking.start_run(disabled, "alpha", "disabled-run") is None
    tracking.safe_log_pipeline_result(disabled, "alpha", SimpleNamespace(status="complete"))
    tracking.safe_log_eval_result(disabled, "alpha", SimpleNamespace(total_labeled=0))
    tracking.safe_log_semantic_match(disabled, "alpha", [])

    assert calls == []


def test_missing_mlflow_import_does_not_crash(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(tracking, "_IMPORT_ATTEMPTED", False)
    monkeypatch.setattr(tracking, "_MLFLOW_MODULE", None)
    monkeypatch.setattr(tracking, "_WARNED_KEYS", set())
    monkeypatch.setattr(tracking.importlib, "import_module", lambda name: (_ for _ in ()).throw(ModuleNotFoundError("no mlflow")))
    monkeypatch.setattr(tracking.logger, "warning", lambda message, *args: warnings.append(message.format(*args)))

    run = tracking.start_run(_enabled_config(), "alpha", "eval-alpha")

    assert run is None
    assert warnings


def test_safe_log_eval_result_logs_expected_payload(monkeypatch, tmp_path: Path):
    calls: dict[str, list] = {"params": [], "metrics": [], "artifacts": [], "ended": []}
    labels_path = tmp_path / "profiles" / "alpha" / "eval_labels.yaml"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text("labels: []\n", encoding="utf-8")
    last_result = labels_path.parent / "eval_last_result.json"
    last_result.write_text('{"coverage": 1.0}\n', encoding="utf-8")

    monkeypatch.setattr(tracking, "start_run", lambda *args, **kwargs: object())
    monkeypatch.setattr(tracking, "log_params", lambda payload: calls["params"].append(payload))
    monkeypatch.setattr(tracking, "log_metrics", lambda payload: calls["metrics"].append(payload))
    monkeypatch.setattr(tracking, "log_artifact", lambda path: calls["artifacts"].append(Path(path).name))
    monkeypatch.setattr(tracking, "end_run", lambda status="FINISHED": calls["ended"].append(status))

    result = SimpleNamespace(
        role_family="software_engineering",
        total_labeled=12,
        precision_at_5=0.6,
        precision_at_10=0.5,
        recall_at_10=0.75,
        mrr=0.8,
        ndcg_at_10=0.7,
        coverage=0.9,
    )

    tracking.safe_log_eval_result(
        _enabled_config(),
        "alpha",
        result,
        extra_params={
            "labels_path": str(labels_path),
            "use_reranker": True,
            "top_k_chunks": 80,
            "top_k_jobs": 40,
            "top_k_final": 15,
        },
    )

    assert calls["params"] == [
        {
            "profile": "alpha",
            "role_family": "software_engineering",
            "use_reranker": True,
            "embedding_model": "embed-model",
            "reranker_model": "rerank-model",
            "top_k_chunks": 80,
            "top_k_jobs": 40,
            "top_k_final": 15,
            "labels_path": str(labels_path),
        }
    ]
    assert calls["metrics"] == [
        {
            "total_labeled": 12,
            "precision_at_5": 0.6,
            "precision_at_10": 0.5,
            "recall_at_10": 0.75,
            "mrr": 0.8,
            "ndcg_at_10": 0.7,
            "coverage": 0.9,
        }
    ]
    assert calls["artifacts"] == ["eval_last_result.json"]
    assert calls["ended"] == ["FINISHED"]


def test_safe_log_pipeline_result_handles_missing_fields(monkeypatch):
    calls: dict[str, list] = {"params": [], "metrics": [], "ended": []}
    monkeypatch.setattr(tracking, "start_run", lambda *args, **kwargs: object())
    monkeypatch.setattr(tracking, "log_params", lambda payload: calls["params"].append(payload))
    monkeypatch.setattr(tracking, "log_metrics", lambda payload: calls["metrics"].append(payload))
    monkeypatch.setattr(tracking, "log_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(tracking, "end_run", lambda status="FINISHED": calls["ended"].append(status))
    monkeypatch.setattr(tracking, "_role_family", lambda config: "general")

    result = SimpleNamespace(status="failed", errors=["boom"], final_db_stats={"total": 2})

    tracking.safe_log_pipeline_result(_enabled_config(), "alpha", result)

    assert calls["params"][0]["profile"] == "alpha"
    assert calls["metrics"][0]["pipeline_failure"] == 1
    assert calls["metrics"][0]["jobs_scraped"] == 0
    assert calls["ended"] == ["FAILED"]


def test_log_params_catches_mlflow_exceptions(monkeypatch):
    warnings: list[str] = []

    class FakeMlflow:
        def log_params(self, payload):
            raise RuntimeError("broken")

    monkeypatch.setattr(tracking, "_get_mlflow", lambda: FakeMlflow())
    monkeypatch.setattr(tracking.logger, "warning", lambda message, *args: warnings.append(message.format(*args)))

    tracking.log_params({"profile": "alpha"})

    assert warnings == ["tracking | failed to log MLflow params: broken"]


def test_load_config_applies_tracking_defaults(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm:",
                "  provider: groq",
                "  model:",
                "    groq: test-model",
                "profile:",
                "  name: Test User",
                "  resume: inline resume",
                "preferences: {}",
                "sources: {}",
                "scoring: {}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "_BASE_DIR", tmp_path)

    loaded = config_module.load_config()

    assert loaded["tracking"]["mlflow"]["enabled"] is False
    assert loaded["tracking"]["mlflow"]["tracking_uri"] == "./mlruns"
    assert loaded["tracking"]["mlflow"]["experiment_name"] == "job-agent"
