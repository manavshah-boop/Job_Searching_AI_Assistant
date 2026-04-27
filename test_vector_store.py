"""
test_vector_store.py - Unit tests for the persistent ChromaDB retrieval layer.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import config as config_module
import db
from db import Job, init_db, insert_job, replace_job_embeddings, save_score
from vector_store import (
    clear_vector_index,
    get_job_collection,
    index_embedded_jobs,
    query_similar_chunks,
    query_similar_jobs,
    rebuild_vector_index,
)


def _write_profile_config(root: Path, profile: str, *, vector_enabled: bool = True) -> None:
    profile_dir = root / "profiles" / profile
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "config.yaml").write_text(
        (
            "profile:\n"
            "  name: Test Profile\n"
            "  job_type: fulltime\n"
            "  resume: test resume\n"
            "llm:\n"
            "  provider: groq\n"
            "  model:\n"
            "    groq: test-model\n"
            "embeddings:\n"
            "  enabled: true\n"
            "  model: sentence-transformers/all-MiniLM-L6-v2\n"
            "  batch_size: 8\n"
            "  vector_store:\n"
            f"    enabled: {'true' if vector_enabled else 'false'}\n"
            "    provider: chromadb\n"
            "    persist_directory: null\n"
            "    collection_name: job_chunks\n"
            "    top_k_chunks: 10\n"
            "    top_k_jobs: 5\n"
            "    strict: false\n"
        ),
        encoding="utf-8",
    )


def _configure_temp_workspace(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(db, "_PROFILES_DIR", root / "profiles")
    monkeypatch.setattr(config_module, "_BASE_DIR", root)


def _unit_vector(*values: float) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def _seed_indexable_jobs(profile: str) -> None:
    jobs = [
        Job(
            id="job-backend",
            title="Backend Platform Engineer",
            company="Acme",
            location="Remote",
            url="https://example.com/backend",
            raw_text="Backend Python AWS role",
            source="ashby",
        ),
        Job(
            id="job-frontend",
            title="Frontend UI Engineer",
            company="Beta",
            location="Remote",
            url="https://example.com/frontend",
            raw_text="Frontend React role",
            source="greenhouse",
        ),
    ]
    for job in jobs:
        assert insert_job(job, profile=profile) is True
        save_score(
            job_id=job.id,
            fit_score=82 if "backend" in job.id else 71,
            role_fit=8,
            stack_match=8,
            seniority=7,
            loc_score=10,
            growth=6,
            compensation=6,
            reasons='["strong fit"]',
            flags="[]",
            skill_misses="[]",
            one_liner="Strong fit",
            ats_score=77,
            profile=profile,
        )

    replace_job_embeddings(
        "job-backend",
        "sentence-transformers/all-MiniLM-L6-v2",
        [
            {
                "chunk_key": "requirements",
                "chunk_order": 2,
                "chunk_text": "Python backend systems with AWS and APIs",
                "embedding": json.dumps(_unit_vector(1.0, 0.0, 1.0), separators=(",", ":")),
                "dimensions": 3,
            },
            {
                "chunk_key": "responsibilities",
                "chunk_order": 1,
                "chunk_text": "Build backend services and platform tooling",
                "embedding": json.dumps(_unit_vector(1.0, 0.0, 0.0), separators=(",", ":")),
                "dimensions": 3,
            },
        ],
        profile=profile,
    )
    replace_job_embeddings(
        "job-frontend",
        "sentence-transformers/all-MiniLM-L6-v2",
        [
            {
                "chunk_key": "summary",
                "chunk_order": 0,
                "chunk_text": "React frontend interfaces and design systems",
                "embedding": json.dumps(_unit_vector(0.0, 1.0, 0.0), separators=(",", ":")),
                "dimensions": 3,
            },
            {
                "chunk_key": "benefits",
                "chunk_order": 4,
                "chunk_text": "Great UI tooling and perks",
                "embedding": json.dumps(_unit_vector(0.0, 0.8, 0.2), separators=(",", ":")),
                "dimensions": 3,
            },
        ],
        profile=profile,
    )


def _fake_embed_texts(texts, *, model_name=None):
    vectors = []
    for text in texts:
        lowered = text.lower()
        if "python" in lowered or "backend" in lowered or "aws" in lowered:
            vectors.append(_unit_vector(1.0, 0.0, 1.0))
        elif "react" in lowered or "frontend" in lowered or "ui" in lowered:
            vectors.append(_unit_vector(0.0, 1.0, 0.0))
        else:
            vectors.append(_unit_vector(1.0, 1.0, 0.0))
    return vectors


def test_profile_scoped_collection_creation(monkeypatch):
    root = Path(".tmp_streamlit_test") / "vector_store_profiles"
    if root.exists():
        shutil.rmtree(root)
    _configure_temp_workspace(monkeypatch, root)
    _write_profile_config(root, "alpha")
    _write_profile_config(root, "beta")

    clear_vector_index("alpha")
    clear_vector_index("beta")
    alpha = get_job_collection("alpha")
    beta = get_job_collection("beta")

    assert alpha.name == "job_chunks"
    assert beta.name == "job_chunks"
    assert (root / "profiles" / "alpha" / "chroma").exists()
    assert (root / "profiles" / "beta" / "chroma").exists()


def test_upsert_query_and_rebuild_round_trip(monkeypatch):
    root = Path(".tmp_streamlit_test") / "vector_store_round_trip"
    if root.exists():
        shutil.rmtree(root)
    _configure_temp_workspace(monkeypatch, root)
    _write_profile_config(root, "retrieval")
    monkeypatch.setattr("vector_store.embed_texts", _fake_embed_texts)

    profile = "retrieval"
    init_db(profile=profile)
    _seed_indexable_jobs(profile)

    first = index_embedded_jobs(profile, "sentence-transformers/all-MiniLM-L6-v2")
    second = index_embedded_jobs(profile, "sentence-transformers/all-MiniLM-L6-v2")
    collection = get_job_collection(profile)

    assert first["chunks_indexed"] == 4
    assert second["chunks_indexed"] == 4
    assert collection.count() == 4

    chunks = query_similar_chunks(profile, "python backend aws", top_k_chunks=4)
    assert chunks
    assert chunks[0].job_id == "job-backend"
    assert chunks[0].chunk_key in {"requirements", "responsibilities"}
    assert chunks[0].title == "Backend Platform Engineer"
    assert chunks[0].company == "Acme"
    assert 0.0 <= chunks[0].similarity <= 1.0

    jobs = query_similar_jobs(profile, "python backend aws", top_k_chunks=4, top_k_jobs=3)
    assert jobs
    assert len({job.job_id for job in jobs}) == len(jobs)
    assert jobs[0].job_id == "job-backend"
    assert "requirements" in jobs[0].matched_chunks


def test_rebuild_vector_index_recovers_from_missing_directory(monkeypatch):
    root = Path(".tmp_streamlit_test") / "vector_store_rebuild"
    if root.exists():
        shutil.rmtree(root)
    _configure_temp_workspace(monkeypatch, root)
    _write_profile_config(root, "rebuild")
    monkeypatch.setattr("vector_store.embed_texts", _fake_embed_texts)

    profile = "rebuild"
    init_db(profile=profile)
    _seed_indexable_jobs(profile)
    index_embedded_jobs(profile, "sentence-transformers/all-MiniLM-L6-v2")

    shutil.rmtree(root / "profiles" / profile / "chroma", ignore_errors=True)
    rebuilt = rebuild_vector_index(profile)
    jobs = query_similar_jobs(profile, "react frontend ui", top_k_chunks=4, top_k_jobs=2)

    assert rebuilt["chunks_indexed"] == 4
    assert jobs
    assert jobs[0].job_id == "job-frontend"
