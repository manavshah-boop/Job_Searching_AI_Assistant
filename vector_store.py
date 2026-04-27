"""
vector_store.py - Persistent ChromaDB retrieval index over SQLite-backed job chunks.

SQLite remains the durable source of truth for jobs, scores, and embeddings.
ChromaDB is a profile-scoped, rebuildable retrieval index that can be recreated
from the `job_embeddings` table at any time.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from loguru import logger

from config import load_config
from db import get_db_path, get_embedding_index_rows
from embedder import DEFAULT_EMBEDDING_MODEL, embed_texts, embedding_model_name

DEFAULT_COLLECTION_NAME = "job_chunks"
DEFAULT_TOP_K_CHUNKS = 50
DEFAULT_TOP_K_JOBS = 30
DEFAULT_DISTANCE_METRIC = "cosine"
MIN_CHUNK_SIMILARITY = 0.05
SECTION_WEIGHTS = {
    "requirements": 1.10,
    "responsibilities": 1.05,
    "summary": 1.00,
    "compensation": 0.90,
    "benefits": 0.80,
}
COMPENSATION_QUERY_TERMS = (
    "salary",
    "compensation",
    "pay",
    "benefits",
    "equity",
    "bonus",
    "pto",
    "401k",
    "medical",
)


@dataclass
class MatchedChunkScore:
    chunk_key: str
    chunk_order: int
    similarity: float
    weighted_similarity: float


@dataclass
class VectorChunkResult:
    chroma_id: str
    job_id: str
    chunk_key: str
    chunk_order: int
    title: str
    company: str
    source: str
    url: str
    distance: float
    similarity: float
    chunk_text: str


@dataclass
class VectorJobResult:
    job_id: str
    title: str
    company: str
    source: str
    url: str
    best_similarity: float
    aggregate_score: float
    matched_chunks: list[str] = field(default_factory=list)
    chunk_scores: list[MatchedChunkScore] = field(default_factory=list)
    retrieval_reason: str = ""


def vector_store_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("embeddings", {}).get("vector_store", {}).get("enabled", False))


def vector_store_strict(config: dict[str, Any]) -> bool:
    return bool(config.get("embeddings", {}).get("vector_store", {}).get("strict", False))


def vector_collection_name(config: dict[str, Any]) -> str:
    raw = str(
        config.get("embeddings", {})
        .get("vector_store", {})
        .get("collection_name", DEFAULT_COLLECTION_NAME)
    ).strip()
    return raw or DEFAULT_COLLECTION_NAME


def vector_top_k_chunks(config: dict[str, Any]) -> int:
    raw = config.get("embeddings", {}).get("vector_store", {}).get("top_k_chunks", DEFAULT_TOP_K_CHUNKS)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_TOP_K_CHUNKS
    return max(1, min(200, value))


def vector_top_k_jobs(config: dict[str, Any]) -> int:
    raw = config.get("embeddings", {}).get("vector_store", {}).get("top_k_jobs", DEFAULT_TOP_K_JOBS)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_TOP_K_JOBS
    return max(1, min(100, value))


def _persist_directory(profile: str, config: dict[str, Any]) -> Path:
    base = get_db_path(profile).parent
    raw = config.get("embeddings", {}).get("vector_store", {}).get("persist_directory")
    if raw:
        candidate = Path(str(raw))
        if not candidate.is_absolute():
            candidate = base / candidate
        return candidate
    return base / "chroma"


def _model_slug(model_name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in model_name).strip("_") or "model"


def _stable_chroma_id(job_id: str, chunk_key: str, model_name: str) -> str:
    return f"{job_id}:{chunk_key}:{_model_slug(model_name)}"


def _distance_to_similarity(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - float(distance)))


def _vector_store_config(profile: str) -> dict[str, Any]:
    config = load_config(profile=profile)
    config["_active_profile"] = profile
    return config


def _import_chromadb():
    import chromadb

    return chromadb


def _collection_metadata(model_name: str, dimensions: int) -> dict[str, Any]:
    return {
        "embedding_model": model_name,
        "embedding_dimensions": int(dimensions),
        "distance_metric": DEFAULT_DISTANCE_METRIC,
        "created_by": "job_search_ai_assistant",
    }


def _sanitize_metadata(row: dict[str, Any], model_name: str, dimensions: int) -> dict[str, Any]:
    def _string(value: Any) -> str:
        return "" if value is None else str(value)

    def _number(value: Any, default: float | int = 0) -> float | int:
        if value in (None, ""):
            return default
        return value

    return {
        "profile": _string(row.get("profile")),
        "job_id": _string(row.get("job_id")),
        "chunk_key": _string(row.get("chunk_key")),
        "chunk_order": int(_number(row.get("chunk_order"), 0)),
        "title": _string(row.get("title")),
        "company": _string(row.get("company")),
        "source": _string(row.get("source")),
        "url": _string(row.get("url")),
        "model_name": model_name,
        "dimensions": int(dimensions),
        "scored_at": _string(row.get("scored_at")),
        "fit_score": float(_number(row.get("fit_score"), 0.0)),
        "ats_score": float(_number(row.get("ats_score"), 0.0)),
        "status": _string(row.get("status")),
        "created_at": _string(row.get("created_at")),
    }


def _section_weights_for_query(query: str) -> dict[str, float]:
    weights = dict(SECTION_WEIGHTS)
    lowered = query.lower()
    if any(term in lowered for term in COMPENSATION_QUERY_TERMS):
        weights["compensation"] = 1.05
        weights["benefits"] = 1.00
    return weights


def _resolve_model_name(profile: str, explicit_model_name: Optional[str] = None) -> str:
    if explicit_model_name:
        return explicit_model_name
    config = _vector_store_config(profile)
    return embedding_model_name(config) or DEFAULT_EMBEDDING_MODEL


def _ensure_collection(profile: str, model_name: Optional[str] = None, dimensions: Optional[int] = None):
    config = _vector_store_config(profile)
    client = get_chroma_client(profile)
    collection_name = vector_collection_name(config)
    chromadb = _import_chromadb()

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        metadata = _collection_metadata(model_name or DEFAULT_EMBEDDING_MODEL, dimensions or 0)
        collection = client.create_collection(name=collection_name, metadata=metadata)
        return collection, False

    if model_name is None or dimensions is None:
        return collection, False

    desired = _collection_metadata(model_name, dimensions)
    current = collection.metadata or {}
    mismatched = any(current.get(key) != value for key, value in desired.items())
    if mismatched:
        logger.warning(
            "vector_store | collection metadata mismatch for profile={} collection={} current_model={} new_model={} | recreating index",
            profile,
            collection_name,
            current.get("embedding_model"),
            model_name,
        )
        client.delete_collection(collection_name)
        collection = client.create_collection(name=collection_name, metadata=desired)
        return collection, True

    return collection, False


def get_chroma_client(profile: str):
    config = _vector_store_config(profile)
    persist_dir = _persist_directory(profile, config)
    persist_dir.mkdir(parents=True, exist_ok=True)
    chromadb = _import_chromadb()
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def get_job_collection(profile: str):
    collection, _ = _ensure_collection(profile)
    return collection


def _normalize_embedding(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    raise ValueError("Embedding row is missing a decoded embedding list")


def upsert_job_embeddings(profile: str, model_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    if not rows:
        config = _vector_store_config(profile)
        return {
            "profile": profile,
            "model_name": model_name,
            "chunks_indexed": 0,
            "jobs_indexed": 0,
            "collection_name": vector_collection_name(config),
            "persist_directory": str(_persist_directory(profile, config)),
            "recreated_collection": False,
            "latency_ms": 0.0,
        }

    dimensions = int(rows[0].get("dimensions") or len(_normalize_embedding(rows[0]["embedding"])))
    collection, recreated = _ensure_collection(profile, model_name=model_name, dimensions=dimensions)
    config = _vector_store_config(profile)
    persist_dir = _persist_directory(profile, config)
    collection_name = vector_collection_name(config)

    documents: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []
    job_ids: list[str] = []

    for row in rows:
        embedding = _normalize_embedding(row["embedding"])
        document = str(row.get("chunk_text") or "").strip()
        if not document:
            continue
        job_id = str(row["job_id"])
        chunk_key = str(row["chunk_key"])
        ids.append(_stable_chroma_id(job_id, chunk_key, model_name))
        embeddings.append(embedding)
        documents.append(document)
        metadatas.append(_sanitize_metadata(row, model_name, len(embedding)))
        job_ids.append(job_id)

    if not ids:
        return {
            "profile": profile,
            "model_name": model_name,
            "chunks_indexed": 0,
            "jobs_indexed": 0,
            "collection_name": collection_name,
            "persist_directory": str(persist_dir),
            "recreated_collection": recreated,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    for job_id in sorted(set(job_ids)):
        collection.delete(where={"job_id": job_id})

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "vector_store | indexed {} chunks across {} jobs | profile={} path={} collection={} model={} latency_ms={}",
        len(ids),
        len(set(job_ids)),
        profile,
        persist_dir,
        collection_name,
        model_name,
        latency_ms,
    )
    return {
        "profile": profile,
        "model_name": model_name,
        "chunks_indexed": len(ids),
        "jobs_indexed": len(set(job_ids)),
        "collection_name": collection_name,
        "persist_directory": str(persist_dir),
        "recreated_collection": recreated,
        "latency_ms": latency_ms,
    }


def index_embedded_jobs(
    profile: str,
    model_name: str,
    force: bool = False,
    *,
    job_ids: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if force:
        clear_vector_index(profile)
    rows = get_embedding_index_rows(model_name, profile=profile, job_ids=job_ids)
    result = upsert_job_embeddings(profile, model_name, rows)
    result["force"] = force
    result["source_rows"] = len(rows)
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result


def _query_collection(profile: str, query_embedding: list[float], top_k_chunks: int, filters: dict[str, Any] | None):
    collection = get_job_collection(profile)
    count = collection.count()
    if count <= 0:
        return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k_chunks, count),
        where=filters or None,
        include=["documents", "distances", "metadatas"],
    )


def query_similar_chunks(
    profile: str,
    query: str,
    top_k_chunks: int = 50,
    filters: dict[str, Any] | None = None,
) -> list[VectorChunkResult]:
    config = _vector_store_config(profile)
    if not vector_store_enabled(config):
        logger.info("vector_store | query skipped because vector store is disabled for profile={}", profile)
        return []

    started = time.perf_counter()
    model_name = _resolve_model_name(profile)
    query_vector = embed_texts([query], model_name=model_name)[0]
    response = _query_collection(profile, query_vector, top_k_chunks, filters)

    results: list[VectorChunkResult] = []
    ids = response.get("ids", [[]])[0]
    documents = response.get("documents", [[]])[0]
    distances = response.get("distances", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]

    for chroma_id, document, distance, metadata in zip(ids, documents, distances, metadatas):
        metadata = metadata or {}
        results.append(
            VectorChunkResult(
                chroma_id=str(chroma_id),
                job_id=str(metadata.get("job_id", "")),
                chunk_key=str(metadata.get("chunk_key", "")),
                chunk_order=int(metadata.get("chunk_order", 0) or 0),
                title=str(metadata.get("title", "")),
                company=str(metadata.get("company", "")),
                source=str(metadata.get("source", "")),
                url=str(metadata.get("url", "")),
                distance=float(distance),
                similarity=_distance_to_similarity(float(distance)),
                chunk_text=str(document or ""),
            )
        )

    results = [result for result in results if result.similarity >= MIN_CHUNK_SIMILARITY]
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "vector_store | query_similar_chunks profile={} model={} top_k={} latency_ms={}",
        profile,
        model_name,
        top_k_chunks,
        latency_ms,
    )
    return results


def _coverage_bonus(weighted_chunks: Sequence[MatchedChunkScore]) -> float:
    if not weighted_chunks:
        return 0.0
    distinct = {chunk.chunk_key for chunk in weighted_chunks[:3]}
    return min(len(distinct) / 3.0, 1.0)


def _retrieval_reason(scored_chunks: Sequence[MatchedChunkScore]) -> str:
    if not scored_chunks:
        return "No semantic evidence matched this query."
    first = scored_chunks[0].chunk_key
    if len(scored_chunks) == 1:
        return f"Best semantic match came from the {first} section."
    second = scored_chunks[1].chunk_key
    return f"Best semantic match came from the {first} section, with supporting evidence in {second}."


def query_similar_jobs(
    profile: str,
    query: str,
    top_k_chunks: int = 50,
    top_k_jobs: int = 30,
    filters: dict[str, Any] | None = None,
) -> list[VectorJobResult]:
    started = time.perf_counter()
    chunk_results = query_similar_chunks(profile, query, top_k_chunks=top_k_chunks, filters=filters)
    if not chunk_results:
        return []

    weights = _section_weights_for_query(query)
    by_job: dict[str, list[VectorChunkResult]] = {}
    for chunk in chunk_results:
        by_job.setdefault(chunk.job_id, []).append(chunk)

    job_results: list[VectorJobResult] = []
    for job_id, chunks in by_job.items():
        ordered_chunks = sorted(chunks, key=lambda item: item.similarity, reverse=True)
        scored_chunks = [
            MatchedChunkScore(
                chunk_key=chunk.chunk_key,
                chunk_order=chunk.chunk_order,
                similarity=round(chunk.similarity, 4),
                weighted_similarity=round(
                    min(1.0, chunk.similarity * weights.get(chunk.chunk_key, 1.0)),
                    4,
                ),
            )
            for chunk in ordered_chunks
        ]
        weighted_values = [chunk.weighted_similarity for chunk in scored_chunks]
        best = weighted_values[0]
        top_average = sum(weighted_values[:3]) / min(3, len(weighted_values))
        coverage = _coverage_bonus(scored_chunks)
        aggregate = min(1.0, best * 0.75 + top_average * 0.20 + coverage * 0.05)
        top_chunk = ordered_chunks[0]
        if top_chunk.similarity < MIN_CHUNK_SIMILARITY:
            continue
        job_results.append(
            VectorJobResult(
                job_id=job_id,
                title=top_chunk.title,
                company=top_chunk.company,
                source=top_chunk.source,
                url=top_chunk.url,
                best_similarity=round(top_chunk.similarity, 4),
                aggregate_score=round(aggregate, 4),
                matched_chunks=[chunk.chunk_key for chunk in scored_chunks[:3]],
                chunk_scores=scored_chunks[:5],
                retrieval_reason=_retrieval_reason(scored_chunks[:3]),
            )
        )

    ranked = sorted(
        job_results,
        key=lambda item: (item.aggregate_score, item.best_similarity),
        reverse=True,
    )[:top_k_jobs]
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "vector_store | query_similar_jobs profile={} top_k_chunks={} top_k_jobs={} latency_ms={}",
        profile,
        top_k_chunks,
        top_k_jobs,
        latency_ms,
    )
    return ranked


def clear_vector_index(profile: str) -> None:
    config = _vector_store_config(profile)
    collection_name = vector_collection_name(config)
    persist_dir = _persist_directory(profile, config)
    try:
        client = get_chroma_client(profile)
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    finally:
        if persist_dir.exists():
            shutil.rmtree(persist_dir, ignore_errors=True)
        persist_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "vector_store | cleared collection profile={} path={} collection={}",
        profile,
        persist_dir,
        collection_name,
    )


def rebuild_vector_index(profile: str, model_name: str | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    resolved_model = _resolve_model_name(profile, explicit_model_name=model_name)
    logger.info(
        "vector_store | rebuild start profile={} model={}",
        profile,
        resolved_model,
    )
    clear_vector_index(profile)
    result = index_embedded_jobs(profile, resolved_model, force=False)
    result["rebuild"] = True
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "vector_store | rebuild complete profile={} model={} chunks={} jobs={} latency_ms={}",
        profile,
        resolved_model,
        result.get("chunks_indexed", 0),
        result.get("jobs_indexed", 0),
        result["latency_ms"],
    )
    return result
