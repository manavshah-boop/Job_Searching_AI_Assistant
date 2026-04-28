"""
tracking.py - Optional, fail-safe MLflow observability helpers.

This is the only module in the repo that imports or calls MLflow.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from loguru import logger

from config import apply_config_defaults

_PROJECT_ROOT = Path(__file__).parent
_IMPORT_ATTEMPTED = False
_MLFLOW_MODULE: Any = None
_WARNED_KEYS: set[str] = set()


def _warn_once(key: str, message: str, *args: Any) -> None:
    if key in _WARNED_KEYS:
        return
    _WARNED_KEYS.add(key)
    logger.warning(message, *args)


def _tracking_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = apply_config_defaults(config)
    return dict(normalized.get("tracking", {}).get("mlflow", {}) or {})


def _artifacts_enabled(config: dict[str, Any]) -> bool:
    return bool(_tracking_config(config).get("log_artifacts", True))


def _eval_labels_enabled(config: dict[str, Any]) -> bool:
    return bool(_tracking_config(config).get("log_eval_labels", False))


def _active_profile_name(profile: str | None) -> str:
    return str(profile or "default")


def _coerce_param_value(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, int, float)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _resolve_tracking_uri(raw_uri: str) -> str:
    text = str(raw_uri or "./mlruns").strip()
    if not text:
        text = "./mlruns"
    if "://" in text or text.startswith("file:"):
        return text
    path = Path(text)
    if not path.is_absolute():
        path = (_PROJECT_ROOT / path).resolve()
    return str(path)


def _get_mlflow():
    global _IMPORT_ATTEMPTED, _MLFLOW_MODULE
    if _MLFLOW_MODULE is not None:
        return _MLFLOW_MODULE
    if _IMPORT_ATTEMPTED:
        return None

    _IMPORT_ATTEMPTED = True
    try:
        _MLFLOW_MODULE = importlib.import_module("mlflow")
    except Exception as exc:
        _warn_once("mlflow-import", "tracking | MLflow import failed; tracking disabled for this run: {}", exc)
        _MLFLOW_MODULE = None
    return _MLFLOW_MODULE


def mlflow_enabled(config: dict[str, Any]) -> bool:
    return bool(_tracking_config(config).get("enabled", False))


def get_tracking_uri(config: dict[str, Any]) -> str:
    return _resolve_tracking_uri(str(_tracking_config(config).get("tracking_uri", "./mlruns")))


def get_experiment_name(config: dict[str, Any], profile: str | None) -> str:
    raw = str(_tracking_config(config).get("experiment_name", "job-agent")).strip()
    return raw or "job-agent"


def start_run(
    config: dict[str, Any],
    profile: str | None,
    run_name: str,
    tags: dict[str, Any] | None = None,
):
    if not mlflow_enabled(config):
        return None

    mlflow = _get_mlflow()
    if mlflow is None:
        return None

    clean_tags = {
        "profile": _active_profile_name(profile),
        **{str(key): _coerce_param_value(value) for key, value in (tags or {}).items() if value is not None},
    }
    try:
        mlflow.set_tracking_uri(get_tracking_uri(config))
        mlflow.set_experiment(get_experiment_name(config, profile))
        return mlflow.start_run(run_name=run_name, tags=clean_tags)
    except Exception as exc:
        logger.warning("tracking | failed to start MLflow run '{}': {}", run_name, exc)
        return None


def log_params(params: dict[str, Any]) -> None:
    mlflow = _get_mlflow()
    if mlflow is None or not params:
        return
    try:
        clean = {
            str(key): _coerce_param_value(value)
            for key, value in params.items()
            if value is not None
        }
        if clean:
            mlflow.log_params(clean)
    except Exception as exc:
        logger.warning("tracking | failed to log MLflow params: {}", exc)


def log_metrics(metrics: dict[str, Any]) -> None:
    mlflow = _get_mlflow()
    if mlflow is None or not metrics:
        return
    try:
        clean: dict[str, float] = {}
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, bool):
                clean[str(key)] = 1.0 if value else 0.0
                continue
            try:
                clean[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        if clean:
            mlflow.log_metrics(clean)
    except Exception as exc:
        logger.warning("tracking | failed to log MLflow metrics: {}", exc)


def log_artifact(path) -> None:
    mlflow = _get_mlflow()
    if mlflow is None or path is None:
        return
    artifact_path = Path(path)
    if not artifact_path.exists():
        return
    try:
        mlflow.log_artifact(str(artifact_path))
    except Exception as exc:
        logger.warning("tracking | failed to log MLflow artifact '{}': {}", artifact_path, exc)


def log_text(name: str, text: str) -> None:
    mlflow = _get_mlflow()
    if mlflow is None or not name:
        return
    try:
        mlflow.log_text(text or "", name)
    except Exception as exc:
        logger.warning("tracking | failed to log MLflow text artifact '{}': {}", name, exc)


def end_run(status: str = "FINISHED") -> None:
    mlflow = _get_mlflow()
    if mlflow is None:
        return
    try:
        if getattr(mlflow, "active_run", lambda: None)() is not None:
            mlflow.end_run(status=status)
    except Exception as exc:
        logger.warning("tracking | failed to end MLflow run: {}", exc)


def _role_family(config: dict[str, Any]) -> str | None:
    try:
        from profile_intent import normalize_profile_intent

        return normalize_profile_intent(config).role_family
    except Exception:
        return None


def safe_log_pipeline_result(config, profile, pipeline_result) -> None:
    if not mlflow_enabled(config):
        return

    status = str(getattr(pipeline_result, "status", "complete") or "complete")
    if start_run(
        config,
        profile,
        f"pipeline-{_active_profile_name(profile)}",
        tags={"run_type": "pipeline", "status": status},
    ) is None:
        return

    scrape = getattr(pipeline_result, "scrape", None)
    score = getattr(pipeline_result, "score", None)
    embed = getattr(pipeline_result, "embed", None)
    vector_index = getattr(embed, "vector_index", {}) or {}
    stage_latencies = dict(getattr(pipeline_result, "stage_latencies", {}) or {})
    provider = str(config.get("llm", {}).get("provider", "") or "")
    llm_models = config.get("llm", {}).get("model", {}) or {}
    enabled_sources = sorted(
        source for source, payload in (config.get("sources", {}) or {}).items()
        if bool((payload or {}).get("enabled", False))
    )

    params = {
        "profile": _active_profile_name(profile),
        "role_family": _role_family(config),
        "llm_provider": provider,
        "llm_model": llm_models.get(provider),
        "embedding_model": config.get("embeddings", {}).get("model"),
        "vector_store_enabled": bool(config.get("embeddings", {}).get("vector_store", {}).get("enabled", False)),
        "reranker_enabled": bool(config.get("reranking", {}).get("enabled", True)),
        "reranker_model": config.get("reranking", {}).get("model"),
        "enabled_sources": enabled_sources,
        "top_k_chunks": config.get("embeddings", {}).get("vector_store", {}).get("top_k_chunks"),
        "top_k_jobs": config.get("embeddings", {}).get("vector_store", {}).get("top_k_jobs"),
        "top_k_final": config.get("reranking", {}).get("top_k_final"),
    }
    metrics = {
        "jobs_scraped": getattr(scrape, "jobs_scraped", 0),
        "jobs_filtered": getattr(scrape, "jobs_filtered", 0),
        "jobs_saved": getattr(scrape, "jobs_saved", 0),
        "jobs_scored": getattr(score, "jobs_scored", 0),
        "avg_fit_score": getattr(score, "avg_fit_score", 0.0),
        "avg_ats_score": getattr(score, "avg_ats_score", 0.0),
        "jobs_embedded": getattr(embed, "jobs_embedded", 0),
        "chunks_embedded": getattr(embed, "chunks_embedded", 0),
        "chunks_indexed": vector_index.get("chunks_indexed", 0),
        "vector_jobs_indexed": vector_index.get("jobs_indexed", 0),
        "pipeline_success": 1 if status == "complete" else 0,
        "pipeline_failure": 0 if status == "complete" else 1,
        "pipeline_duration_seconds": getattr(pipeline_result, "duration_seconds", 0.0),
    }
    for key, value in stage_latencies.items():
        metrics[f"latency_{key}_seconds"] = value

    try:
        log_params(params)
        log_metrics(metrics)
        if _artifacts_enabled(config):
            payload = {
                "status": status,
                "errors": list(getattr(pipeline_result, "errors", []) or []),
                "final_db_stats": dict(getattr(pipeline_result, "final_db_stats", {}) or {}),
                "stage_latencies": stage_latencies,
            }
            log_text("pipeline_summary.json", json.dumps(payload, indent=2))
    finally:
        end_run(status="FINISHED" if status == "complete" else "FAILED")


def safe_log_eval_result(config, profile, eval_result, extra_params=None) -> None:
    if not mlflow_enabled(config):
        return

    extra_params = dict(extra_params or {})
    labels_path = Path(
        extra_params.get("labels_path")
        or _PROJECT_ROOT / "profiles" / _active_profile_name(profile) / "eval_labels.yaml"
    )
    last_result_path = labels_path.parent / "eval_last_result.json"
    role_family = getattr(eval_result, "role_family", None)

    if start_run(
        config,
        profile,
        f"eval-{_active_profile_name(profile)}",
        tags={"run_type": "evaluation", "role_family": role_family or "unknown"},
    ) is None:
        return

    params = {
        "profile": _active_profile_name(profile),
        "role_family": role_family,
        "use_reranker": extra_params.get("use_reranker"),
        "embedding_model": config.get("embeddings", {}).get("model"),
        "reranker_model": config.get("reranking", {}).get("model"),
        "top_k_chunks": extra_params.get("top_k_chunks", config.get("embeddings", {}).get("vector_store", {}).get("top_k_chunks")),
        "top_k_jobs": extra_params.get("top_k_jobs", config.get("embeddings", {}).get("vector_store", {}).get("top_k_jobs")),
        "top_k_final": extra_params.get("top_k_final", config.get("reranking", {}).get("top_k_final")),
        "labels_path": str(labels_path),
    }
    metrics = {
        "total_labeled": getattr(eval_result, "total_labeled", 0),
        "precision_at_5": getattr(eval_result, "precision_at_5", 0.0),
        "precision_at_10": getattr(eval_result, "precision_at_10", 0.0),
        "recall_at_10": getattr(eval_result, "recall_at_10", 0.0),
        "mrr": getattr(eval_result, "mrr", 0.0),
        "ndcg_at_10": getattr(eval_result, "ndcg_at_10", 0.0),
        "coverage": getattr(eval_result, "coverage", 0.0),
    }

    try:
        log_params(params)
        log_metrics(metrics)
        if _artifacts_enabled(config):
            log_artifact(last_result_path)
            if _eval_labels_enabled(config):
                log_artifact(labels_path)
    finally:
        end_run()


def safe_log_semantic_match(config, profile, results, query=None) -> None:
    if not mlflow_enabled(config):
        return

    items = list(results or [])
    if start_run(
        config,
        profile,
        f"semantic-{_active_profile_name(profile)}",
        tags={"run_type": "semantic_match"},
    ) is None:
        return

    def _avg(attr: str) -> float:
        values = [float(getattr(item, attr)) for item in items if getattr(item, attr, None) is not None]
        if not values:
            return 0.0
        return sum(values) / len(values)

    params = {
        "profile": _active_profile_name(profile),
        "query_provided": bool(query and str(query).strip()),
        "result_mode": "reranked" if any(hasattr(item, "final_score") for item in items) else "vector",
    }
    metrics = {
        "num_results": len(items),
        "avg_final_score": _avg("final_score"),
        "avg_rerank_score": _avg("rerank_score"),
        "avg_vector_score": _avg("vector_score") or _avg("aggregate_score"),
    }

    try:
        log_params(params)
        log_metrics(metrics)
        if _artifacts_enabled(config):
            top_job_ids = "\n".join(
                str(getattr(item, "job_id", ""))
                for item in items[:10]
                if getattr(item, "job_id", "")
            )
            if top_job_ids:
                log_text("top_result_job_ids.txt", top_job_ids)
    finally:
        end_run()
