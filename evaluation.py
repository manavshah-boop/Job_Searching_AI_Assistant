"""
evaluation.py - Lightweight local evaluation for semantic job matching quality.

This module is intentionally deterministic, profile-scoped, and LLM-free.
It helps answer whether retrieval/reranking changes are improving ranked job
quality across multiple role families.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from profile_intent import normalize_profile_intent
from reranker import reranking_top_k_final, semantic_match_jobs
from vector_store import query_similar_jobs, vector_top_k_chunks, vector_top_k_jobs

ALLOWED_EVAL_LABELS = (
    "great_match",
    "good_match",
    "okay_match",
    "bad_match",
    "should_skip",
)

_LABEL_TO_RELEVANCE = {
    "great_match": 4,
    "good_match": 3,
    "okay_match": 2,
    "bad_match": 1,
    "should_skip": 0,
}

_DEFAULT_EXPORT_LIMIT = 20
_LAST_RESULT_FILENAME = "eval_last_result.json"


@dataclass(frozen=True)
class EvalLabel:
    profile_slug: str
    job_id: str
    label: str
    notes: str = ""
    role_family: str | None = None


@dataclass(frozen=True)
class EvalResult:
    profile_slug: str
    role_family: str | None
    total_labeled: int
    precision_at_5: float
    precision_at_10: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    coverage: float
    notes: str = ""


def label_to_relevance(label: str) -> int:
    try:
        return _LABEL_TO_RELEVANCE[str(label).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported eval label: {label!r}") from exc


def _profile_slug_from_path(path: Path) -> str:
    if path.parent.name:
        return path.parent.name
    return "default"


def _normalize_label_entry(entry: dict[str, Any], *, profile_slug: str) -> EvalLabel:
    job_id = str(entry.get("job_id", "")).strip()
    label = str(entry.get("label", "")).strip().lower()
    if not job_id:
        raise ValueError("Eval label entry is missing job_id")
    if label not in ALLOWED_EVAL_LABELS:
        raise ValueError(f"Unsupported eval label {label!r} for job_id={job_id!r}")
    role_family = entry.get("role_family")
    role_family_str = str(role_family).strip() if role_family not in (None, "") else None
    return EvalLabel(
        profile_slug=str(entry.get("profile_slug", profile_slug)).strip() or profile_slug,
        job_id=job_id,
        label=label,
        notes=str(entry.get("notes", "") or "").strip(),
        role_family=role_family_str,
    )


def _dedupe_labels(labels: Iterable[EvalLabel]) -> list[EvalLabel]:
    deduped: dict[str, EvalLabel] = {}
    for label in labels:
        deduped[label.job_id] = label
    return list(deduped.values())


def load_eval_labels(path: str | Path) -> list[EvalLabel]:
    eval_path = Path(path)
    if not eval_path.exists():
        return []

    raw_text = eval_path.read_text(encoding="utf-8")
    profile_slug = _profile_slug_from_path(eval_path)

    if eval_path.suffix.lower() == ".json":
        payload = json.loads(raw_text or "{}")
    else:
        payload = yaml.safe_load(raw_text) or {}

    raw_labels = payload.get("labels", []) if isinstance(payload, dict) else []
    if not isinstance(raw_labels, list):
        raise ValueError(f"Expected 'labels' list in {eval_path}")

    labels = [_normalize_label_entry(entry or {}, profile_slug=profile_slug) for entry in raw_labels]
    return _dedupe_labels(labels)


def save_eval_labels(labels: Sequence[EvalLabel], path: str | Path) -> None:
    eval_path = Path(path)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "labels": [
            {
                "job_id": label.job_id,
                "label": label.label,
                "notes": label.notes,
                **({"role_family": label.role_family} if label.role_family else {}),
            }
            for label in _dedupe_labels(labels)
        ]
    }

    if eval_path.suffix.lower() == ".json":
        eval_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return

    eval_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _relevant_job_ids(labels: Sequence[EvalLabel]) -> set[str]:
    return {
        label.job_id
        for label in labels
        if label_to_relevance(label.label) >= 3
    }


def _precision_at_k(relevance_by_job_id: dict[str, int], ranked_job_ids: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = list(ranked_job_ids[:k])
    if not top_k:
        return 0.0
    relevant = sum(1 for job_id in top_k if relevance_by_job_id.get(job_id, 0) >= 3)
    return relevant / k


def _recall_at_k(relevant_job_ids: set[str], ranked_job_ids: Sequence[str], k: int) -> float:
    if not relevant_job_ids:
        return 0.0
    hits = sum(1 for job_id in ranked_job_ids[:k] if job_id in relevant_job_ids)
    return hits / len(relevant_job_ids)


def _mrr(relevant_job_ids: set[str], ranked_job_ids: Sequence[str]) -> float:
    if not relevant_job_ids:
        return 0.0
    for index, job_id in enumerate(ranked_job_ids, start=1):
        if job_id in relevant_job_ids:
            return 1.0 / index
    return 0.0


def _dcg(relevance_scores: Sequence[int]) -> float:
    total = 0.0
    for index, relevance in enumerate(relevance_scores, start=1):
        if relevance <= 0:
            continue
        total += (2**relevance - 1) / math.log2(index + 1)
    return total


def _ndcg_at_k(relevance_by_job_id: dict[str, int], ranked_job_ids: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    actual = [relevance_by_job_id.get(job_id, 0) for job_id in ranked_job_ids[:k]]
    ideal = sorted(relevance_by_job_id.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg <= 0:
        return 0.0
    return _dcg(actual) / ideal_dcg


def _coverage(labels: Sequence[EvalLabel], ranked_job_ids: Sequence[str]) -> float:
    if not labels:
        return 0.0
    ranked = set(ranked_job_ids)
    covered = sum(1 for label in labels if label.job_id in ranked)
    return covered / len(labels)


def _result_notes(labels: Sequence[EvalLabel], ranked_job_ids: Sequence[str]) -> str:
    relevant_labels = [
        label for label in labels
        if label_to_relevance(label.label) >= 3 and label.job_id not in set(ranked_job_ids)
    ]
    if not relevant_labels:
        return "All labeled great/good matches appeared in the evaluated candidate set."
    formatted = ", ".join(f"{label.job_id} ({label.label})" for label in relevant_labels[:5])
    suffix = "" if len(relevant_labels) <= 5 else f", +{len(relevant_labels) - 5} more"
    return f"Missed labeled great/good matches: {formatted}{suffix}."


def evaluate_ranked_results(
    labels: Sequence[EvalLabel],
    ranked_job_ids: Sequence[str],
    k_values: Sequence[int] = (5, 10),
) -> EvalResult:
    deduped_labels = _dedupe_labels(labels)
    profile_slug = deduped_labels[0].profile_slug if deduped_labels else "default"
    role_families = sorted({label.role_family for label in deduped_labels if label.role_family})
    role_family = role_families[0] if len(role_families) == 1 else None
    relevance_by_job_id = {label.job_id: label_to_relevance(label.label) for label in deduped_labels}
    relevant_job_ids = _relevant_job_ids(deduped_labels)
    ranked_unique = list(dict.fromkeys(str(job_id) for job_id in ranked_job_ids if str(job_id).strip()))

    k_set = {int(k) for k in k_values}
    precision_at_5 = _precision_at_k(relevance_by_job_id, ranked_unique, 5 if 5 in k_set else 5)
    precision_at_10 = _precision_at_k(relevance_by_job_id, ranked_unique, 10 if 10 in k_set else 10)
    recall_at_10 = _recall_at_k(relevant_job_ids, ranked_unique, 10)

    return EvalResult(
        profile_slug=profile_slug,
        role_family=role_family,
        total_labeled=len(deduped_labels),
        precision_at_5=precision_at_5,
        precision_at_10=precision_at_10,
        recall_at_10=recall_at_10,
        mrr=_mrr(relevant_job_ids, ranked_unique),
        ndcg_at_10=_ndcg_at_k(relevance_by_job_id, ranked_unique, 10),
        coverage=_coverage(deduped_labels, ranked_unique),
        notes=_result_notes(deduped_labels, ranked_unique),
    )


def _profile_eval_paths(profile_slug: str) -> tuple[Path, Path]:
    profile_dir = Path(__file__).parent / "profiles" / profile_slug
    return (
        profile_dir / "eval_labels.yaml",
        profile_dir / _LAST_RESULT_FILENAME,
    )


def _expanded_eval_config(config: dict[str, Any], label_count: int) -> dict[str, Any]:
    expanded = json.loads(json.dumps(config))
    reranking_cfg = expanded.setdefault("reranking", {})
    embeddings_cfg = expanded.setdefault("embeddings", {}).setdefault("vector_store", {})

    expanded_limit = max(_DEFAULT_EXPORT_LIMIT, label_count + 10, reranking_top_k_final(expanded))
    reranking_cfg["top_k_final"] = expanded_limit
    reranking_cfg["top_k_vector_jobs"] = max(expanded_limit * 3, int(reranking_cfg.get("top_k_vector_jobs", 40) or 40))
    reranking_cfg["top_k_vector_chunks"] = max(expanded_limit * 4, int(reranking_cfg.get("top_k_vector_chunks", 80) or 80))
    embeddings_cfg["top_k_jobs"] = max(expanded_limit * 3, int(embeddings_cfg.get("top_k_jobs", 30) or 30))
    embeddings_cfg["top_k_chunks"] = max(expanded_limit * 4, int(embeddings_cfg.get("top_k_chunks", 50) or 50))
    return expanded


def _evaluate_ranked_job_ids(profile_slug: str, config: dict[str, Any], *, use_reranker: bool) -> list[str]:
    if use_reranker:
        return [result.job_id for result in semantic_match_jobs(profile_slug, config, user_query=None)]
    vector_results = query_similar_jobs(
        profile_slug,
        " ".join(normalize_profile_intent(config).raw_keywords) or "job match",
        top_k_chunks=vector_top_k_chunks(config),
        top_k_jobs=vector_top_k_jobs(config),
    )
    return [result.job_id for result in vector_results]


def _save_last_eval_result(profile_slug: str, result: EvalResult) -> None:
    _, last_result_path = _profile_eval_paths(profile_slug)
    last_result_path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")


def _save_last_eval_result_at_path(path: str | Path, result: EvalResult) -> None:
    last_result_path = Path(path)
    last_result_path.parent.mkdir(parents=True, exist_ok=True)
    last_result_path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")


def load_last_eval_result(profile_slug: str) -> EvalResult | None:
    _, last_result_path = _profile_eval_paths(profile_slug)
    if not last_result_path.exists():
        return None
    payload = json.loads(last_result_path.read_text(encoding="utf-8") or "{}")
    return EvalResult(
        profile_slug=str(payload.get("profile_slug", profile_slug)),
        role_family=payload.get("role_family"),
        total_labeled=int(payload.get("total_labeled", 0) or 0),
        precision_at_5=float(payload.get("precision_at_5", 0.0) or 0.0),
        precision_at_10=float(payload.get("precision_at_10", 0.0) or 0.0),
        recall_at_10=float(payload.get("recall_at_10", 0.0) or 0.0),
        mrr=float(payload.get("mrr", 0.0) or 0.0),
        ndcg_at_10=float(payload.get("ndcg_at_10", 0.0) or 0.0),
        coverage=float(payload.get("coverage", 0.0) or 0.0),
        notes=str(payload.get("notes", "") or ""),
    )


def evaluate_profile(
    profile_slug: str,
    config: dict[str, Any],
    labels_path: str | Path,
    use_reranker: bool = True,
) -> EvalResult:
    eval_labels_path = Path(labels_path)
    labels = load_eval_labels(eval_labels_path)
    profile_intent = normalize_profile_intent(config)
    with_role_family = [
        label if label.role_family else EvalLabel(
            profile_slug=label.profile_slug,
            job_id=label.job_id,
            label=label.label,
            notes=label.notes,
            role_family=profile_intent.role_family,
        )
        for label in labels
    ]
    ranked_job_ids = _evaluate_ranked_job_ids(
        profile_slug,
        _expanded_eval_config(config, len(with_role_family)),
        use_reranker=use_reranker,
    )
    result = evaluate_ranked_results(with_role_family, ranked_job_ids)
    normalized_result = EvalResult(
        profile_slug=profile_slug,
        role_family=profile_intent.role_family,
        total_labeled=result.total_labeled,
        precision_at_5=result.precision_at_5,
        precision_at_10=result.precision_at_10,
        recall_at_10=result.recall_at_10,
        mrr=result.mrr,
        ndcg_at_10=result.ndcg_at_10,
        coverage=result.coverage,
        notes=result.notes,
    )
    _save_last_eval_result_at_path(eval_labels_path.parent / _LAST_RESULT_FILENAME, normalized_result)
    return normalized_result


def compare_eval_runs(previous: EvalResult | dict[str, Any], current: EvalResult | dict[str, Any]) -> dict[str, Any]:
    previous_payload = asdict(previous) if isinstance(previous, EvalResult) else dict(previous)
    current_payload = asdict(current) if isinstance(current, EvalResult) else dict(current)
    metric_names = (
        "precision_at_5",
        "precision_at_10",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
        "coverage",
    )
    deltas = {
        metric: round(float(current_payload.get(metric, 0.0)) - float(previous_payload.get(metric, 0.0)), 4)
        for metric in metric_names
    }
    return {
        "profile_slug": current_payload.get("profile_slug") or previous_payload.get("profile_slug"),
        "role_family": current_payload.get("role_family") or previous_payload.get("role_family"),
        "previous": previous_payload,
        "current": current_payload,
        "delta": deltas,
    }


def export_eval_template(
    profile_slug: str,
    config: dict[str, Any],
    path: str | Path,
    use_reranker: bool = True,
    limit: int = _DEFAULT_EXPORT_LIMIT,
) -> list[EvalLabel]:
    eval_path = Path(path)
    existing_by_job_id = {label.job_id: label for label in load_eval_labels(eval_path)}
    profile_intent = normalize_profile_intent(config)
    expanded_config = _expanded_eval_config(config, limit)
    ranked_job_ids = _evaluate_ranked_job_ids(profile_slug, expanded_config, use_reranker=use_reranker)[:limit]

    template_labels: list[EvalLabel] = []
    for job_id in ranked_job_ids:
        existing = existing_by_job_id.get(job_id)
        if existing is not None:
            template_labels.append(existing)
            continue
        template_labels.append(
            EvalLabel(
                profile_slug=profile_slug,
                job_id=job_id,
                label="okay_match",
                notes="Starter label from current semantic matches. Review and relabel manually.",
                role_family=profile_intent.role_family,
            )
        )

    save_eval_labels(template_labels, eval_path)
    return template_labels
