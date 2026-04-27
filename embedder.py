"""
embedder.py - Semantic chunking and batch embedding generation for scored jobs.

The embedding stage is intentionally additive:
- chunk raw job text into a small set of named sections
- embed those sections in batches with a cached sentence-transformers model
- persist one embedding row per job chunk
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

from db import (
    Job,
    get_scored_jobs_for_embedding,
    replace_job_embeddings,
)
from profile_intent import CANONICAL_SECTION_ALIASES, CANONICAL_TO_CHUNK_KEY, map_header_to_canonical

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 8
_MODEL_CACHE: dict[str, Any] = {}

_SECTION_ORDER = [
    "summary",
    "responsibilities",
    "requirements",
    "compensation",
    "benefits",
]

# Built from the canonical alias map so both header classification paths stay in sync.
# canonical sections that don't have a direct chunk_key equivalent fold into the
# existing five-key set via CANONICAL_TO_CHUNK_KEY.
_SECTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "summary": (
        *CANONICAL_SECTION_ALIASES["summary"],
        *CANONICAL_SECTION_ALIASES["company"],
    ),
    "responsibilities": CANONICAL_SECTION_ALIASES["responsibilities"],
    "requirements": (
        *CANONICAL_SECTION_ALIASES["requirements"],
        *CANONICAL_SECTION_ALIASES["preferred_qualifications"],
        *CANONICAL_SECTION_ALIASES["tools_and_skills"],
        *CANONICAL_SECTION_ALIASES["logistics"],
    ),
    "compensation": CANONICAL_SECTION_ALIASES["compensation"],
    "benefits": CANONICAL_SECTION_ALIASES["benefits"],
}

_HEADER_PREFIX_RE = re.compile(r"^(?:[-*]\s+)?([A-Za-z][A-Za-z0-9 &/'()-]{1,80}):?\s*$")


@dataclass
class JobChunk:
    """Single named semantic chunk for a job description."""

    chunk_key: str
    chunk_text: str
    chunk_order: int


def embeddings_enabled(config: Dict[str, Any]) -> bool:
    """Profile-level toggle for the embedding stage."""
    return bool(config.get("embeddings", {}).get("enabled", True))


def embedding_model_name(config: Dict[str, Any]) -> str:
    """Return the configured embedding model, falling back to MiniLM."""
    return str(config.get("embeddings", {}).get("model", DEFAULT_EMBEDDING_MODEL)).strip() or DEFAULT_EMBEDDING_MODEL


def embedding_batch_size(config: Dict[str, Any]) -> int:
    """Return a conservative job-batch size suitable for small machines."""
    raw = config.get("embeddings", {}).get("batch_size", DEFAULT_BATCH_SIZE)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_BATCH_SIZE
    return max(1, min(32, value))


def _load_sentence_transformer(model_name: str) -> Any:
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        logger.info("embedder | loading model {}", model_name)
        model = SentenceTransformer(model_name, device="cpu")
        # Short, section-sized chunks fit comfortably inside this budget.
        if hasattr(model, "max_seq_length"):
            model.max_seq_length = 256
        _MODEL_CACHE[model_name] = model
    return _MODEL_CACHE[model_name]


def _extract_description_body(raw_text: str) -> str:
    marker = "\nDescription:\n"
    if marker in raw_text:
        return raw_text.split(marker, 1)[1].strip()
    return raw_text.strip()


def _extract_metadata(raw_text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for label in ("Title", "Company", "Location", "URL"):
        match = re.search(rf"^{label}:\s*(.+)$", raw_text, flags=re.MULTILINE)
        if match:
            metadata[label.lower()] = match.group(1).strip()
    return metadata


def _normalize_block_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    compact = "\n".join(line for line in lines if line)
    return re.sub(r"\n{3,}", "\n\n", compact).strip()


def _looks_like_header(text: str) -> bool:
    match = _HEADER_PREFIX_RE.match(text.strip())
    if not match:
        return False
    token_count = len(match.group(1).split())
    return 1 <= token_count <= 6


def _classify_header(text: str) -> Optional[str]:
    canonical = map_header_to_canonical(text)
    if canonical is not None:
        return CANONICAL_TO_CHUNK_KEY.get(canonical, canonical)
    return None


def _score_block_for_section(block: str, section: str) -> int:
    text = block.lower()
    keywords = _SECTION_KEYWORDS[section]
    score = sum(1 for keyword in keywords if keyword in text)
    if _looks_like_header(block.splitlines()[0]):
        header_section = _classify_header(block.splitlines()[0])
        if header_section == section:
            score += 3
    return score


def _assign_section(block: str, is_first_block: bool) -> str:
    scores = {section: _score_block_for_section(block, section) for section in _SECTION_ORDER}
    best_section = max(scores, key=scores.get)
    if scores[best_section] > 0:
        return best_section
    return "summary" if is_first_block else "requirements"


def _split_description_into_blocks(description: str) -> list[str]:
    text = description.replace("\r\n", "\n")
    parts = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    return parts


def semantic_chunk_job(job: Job, max_chunk_chars: int = 1400) -> list[JobChunk]:
    """
    Convert a job's raw text into a small set of named semantic chunks.

    The chunker intentionally prefers a handful of high-signal sections over
    many tiny fragments, which keeps retrieval cleaner and storage smaller.
    """
    metadata = _extract_metadata(job.raw_text)
    description = _extract_description_body(job.raw_text)
    blocks = _split_description_into_blocks(description)
    grouped: dict[str, list[str]] = {section: [] for section in _SECTION_ORDER}

    for index, block in enumerate(blocks):
        section = _assign_section(block, is_first_block=index == 0)
        grouped.setdefault(section, []).append(_normalize_block_text(block))

    chunks: list[JobChunk] = []
    context_prefix = "\n".join(
        line
        for line in (
            f"Title: {metadata.get('title', job.title)}",
            f"Company: {metadata.get('company', job.company)}",
            f"Location: {metadata.get('location', job.location)}",
        )
        if line.strip()
    )

    for order, section in enumerate(_SECTION_ORDER):
        if not grouped.get(section):
            continue
        body = "\n\n".join(part for part in grouped[section] if part)
        body = body[:max_chunk_chars].strip()
        if not body:
            continue
        text = f"{context_prefix}\nSection: {section}\n{body}".strip()
        chunks.append(JobChunk(chunk_key=section, chunk_text=text, chunk_order=order))

    if not chunks:
        fallback_text = f"{context_prefix}\nSection: summary\n{_normalize_block_text(description)[:max_chunk_chars]}".strip()
        chunks.append(JobChunk(chunk_key="summary", chunk_text=fallback_text, chunk_order=0))

    return chunks


def embed_texts(texts: Iterable[str], *, model_name: str = DEFAULT_EMBEDDING_MODEL) -> list[list[float]]:
    """Embed a batch of texts with a cached sentence-transformers model."""
    materialized = list(texts)
    if not materialized:
        return []

    model = _load_sentence_transformer(model_name)
    vectors = model.encode(
        materialized,
        batch_size=min(len(materialized), 32),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


def embed_jobs(
    config: Dict[str, Any],
    *,
    profile: Optional[str] = None,
    force: bool = False,
    on_job_embedded=None,
) -> dict[str, Any]:
    """
    Embed scored jobs in batches and persist one row per semantic chunk.

    Returns summary metrics suitable for CLI logs and worker progress updates.
    """
    if not embeddings_enabled(config):
        logger.info("embedder | embeddings disabled in config")
        return {
            "enabled": False,
            "jobs_embedded": 0,
            "jobs_total": 0,
            "chunks_embedded": 0,
            "model_name": embedding_model_name(config),
        }

    model_name = embedding_model_name(config)
    job_batch_size = embedding_batch_size(config)
    jobs = get_scored_jobs_for_embedding(model_name, profile=profile, force=force)

    if not jobs:
        logger.info("embedder | no jobs need embeddings")
        return {
            "enabled": True,
            "jobs_embedded": 0,
            "jobs_total": 0,
            "chunks_embedded": 0,
            "model_name": model_name,
        }

    jobs_embedded = 0
    chunks_embedded = 0
    embedded_job_ids: list[str] = []

    for start in range(0, len(jobs), job_batch_size):
        batch_jobs = jobs[start:start + job_batch_size]
        batch_chunks: list[tuple[Job, JobChunk]] = []
        for job in batch_jobs:
            batch_chunks.extend((job, chunk) for chunk in semantic_chunk_job(job))

        vectors = embed_texts(
            (chunk.chunk_text for _, chunk in batch_chunks),
            model_name=model_name,
        )

        cursor = 0
        for job in batch_jobs:
            job_chunks = semantic_chunk_job(job)
            count = len(job_chunks)
            job_vectors = vectors[cursor:cursor + count]
            cursor += count

            payload = [
                {
                    "chunk_key": chunk.chunk_key,
                    "chunk_order": chunk.chunk_order,
                    "chunk_text": chunk.chunk_text,
                    "embedding": json.dumps(vector, separators=(",", ":")),
                    "dimensions": len(vector),
                }
                for chunk, vector in zip(job_chunks, job_vectors)
            ]
            replace_job_embeddings(job.id, model_name, payload, profile=profile)
            jobs_embedded += 1
            chunks_embedded += len(payload)
            embedded_job_ids.append(job.id)
            if on_job_embedded is not None:
                try:
                    on_job_embedded(jobs_embedded, len(jobs), job, len(payload))
                except Exception:
                    pass

    vector_index_result = {
        "enabled": False,
        "status": "skipped",
        "chunks_indexed": 0,
        "jobs_indexed": 0,
        "error": "",
    }
    try:
        from vector_store import index_embedded_jobs, vector_store_enabled

        if vector_store_enabled(config):
            indexed = index_embedded_jobs(
                profile or "default",
                model_name,
                force=force,
                job_ids=embedded_job_ids,
            )
            vector_index_result = {
                "enabled": True,
                "status": "indexed",
                "chunks_indexed": indexed.get("chunks_indexed", 0),
                "jobs_indexed": indexed.get("jobs_indexed", 0),
                "collection_name": indexed.get("collection_name", ""),
                "persist_directory": indexed.get("persist_directory", ""),
                "error": "",
            }
        else:
            vector_index_result = {
                "enabled": False,
                "status": "disabled",
                "chunks_indexed": 0,
                "jobs_indexed": 0,
                "error": "",
            }
    except Exception as exc:
        logger.exception("embedder | vector indexing failed")
        from vector_store import vector_store_strict

        if vector_store_strict(config):
            raise
        vector_index_result = {
            "enabled": True,
            "status": "failed",
            "chunks_indexed": 0,
            "jobs_indexed": 0,
            "error": str(exc),
        }

    logger.info(
        "embedder | stored {} chunks across {} jobs with {}",
        chunks_embedded,
        jobs_embedded,
        model_name,
    )
    return {
        "enabled": True,
        "jobs_embedded": jobs_embedded,
        "jobs_total": len(jobs),
        "chunks_embedded": chunks_embedded,
        "model_name": model_name,
        "vector_index": vector_index_result,
    }
