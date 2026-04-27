"""
test_embedder.py - Unit tests for semantic chunking and embedding persistence.
"""

import shutil
from pathlib import Path

import db
from db import Job, get_job_embeddings, get_scored_jobs_for_embedding, init_db, insert_job, save_score
from embedder import embed_jobs, embed_texts, semantic_chunk_job


class _FakeVector(list):
    def tolist(self):
        return list(self)


class _FakeModel:
    def encode(
        self,
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        materialized = list(texts)
        return [
            _FakeVector([float(index + 1), float(len(text)), 0.5])
            for index, text in enumerate(materialized)
        ]


def _sample_job() -> Job:
    return Job(
        id="job-embed-1",
        title="Backend Engineer",
        company="Acme",
        location="Remote",
        url="https://example.com/jobs/1",
        raw_text=(
            "Title: Backend Engineer\n"
            "Company: Acme\n"
            "Location: Remote\n"
            "URL: https://example.com/jobs/1\n\n"
            "Description:\n"
            "About the role\n"
            "Build backend systems for our product.\n\n"
            "Responsibilities:\n"
            "- Own API services\n"
            "- Improve reliability\n\n"
            "Requirements:\n"
            "- 3+ years of Python\n"
            "- Experience with PostgreSQL\n\n"
            "Compensation:\n"
            "$140,000 - $170,000 base salary\n\n"
            "Benefits:\n"
            "Medical, dental, vision, and 401k."
        ),
        source="ashby",
    )


def test_semantic_chunk_job_extracts_named_sections():
    job = _sample_job()

    chunks = semantic_chunk_job(job)

    chunk_keys = [chunk.chunk_key for chunk in chunks]
    assert chunk_keys == [
        "summary",
        "responsibilities",
        "requirements",
        "compensation",
        "benefits",
    ]
    assert "Build backend systems" in chunks[0].chunk_text
    assert "Experience with PostgreSQL" in chunks[2].chunk_text
    assert "$140,000" in chunks[3].chunk_text


def test_embed_texts_returns_expected_batch_shape(monkeypatch):
    monkeypatch.setattr("embedder._load_sentence_transformer", lambda model_name: _FakeModel())

    vectors = embed_texts(["alpha", "beta", "gamma"])

    assert len(vectors) == 3
    assert all(len(vector) == 3 for vector in vectors)
    assert vectors[0][0] == 1.0
    assert vectors[1][1] == 4.0


def test_embed_jobs_persists_and_reads_back_embeddings(monkeypatch):
    root = Path(".tmp_streamlit_test") / "embedder_profiles"
    if root.exists():
        shutil.rmtree(root)
    profiles_dir = root / "profiles"
    monkeypatch.setattr(db, "_PROFILES_DIR", profiles_dir)
    monkeypatch.setattr("embedder._load_sentence_transformer", lambda model_name: _FakeModel())

    profile = "embedder_test"
    init_db(profile=profile)
    job = _sample_job()
    assert insert_job(job, profile=profile) is True
    save_score(
        job_id=job.id,
        fit_score=78,
        role_fit=8,
        stack_match=8,
        seniority=7,
        loc_score=10,
        growth=6,
        compensation=8,
        reasons='["strong python fit"]',
        flags="[]",
        skill_misses="[]",
        one_liner="Strong backend match",
        ats_score=75,
        profile=profile,
    )

    config = {
        "embeddings": {
            "enabled": True,
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 2,
        }
    }

    result = embed_jobs(config, profile=profile, force=False)

    stored = get_job_embeddings(job.id, profile=profile, model_name="sentence-transformers/all-MiniLM-L6-v2")
    pending = get_scored_jobs_for_embedding("sentence-transformers/all-MiniLM-L6-v2", profile=profile, force=False)

    assert result["jobs_embedded"] == 1
    assert result["chunks_embedded"] == len(stored)
    assert len(stored) >= 4
    assert stored[0]["job_id"] == job.id
    assert len(stored[0]["embedding"]) == 3
    assert pending == []

    shutil.rmtree(root)


def test_embed_jobs_skips_vector_index_when_disabled(monkeypatch):
    root = Path(".tmp_streamlit_test") / "embedder_vector_disabled"
    if root.exists():
        shutil.rmtree(root)
    profiles_dir = root / "profiles"
    monkeypatch.setattr(db, "_PROFILES_DIR", profiles_dir)
    monkeypatch.setattr("embedder._load_sentence_transformer", lambda model_name: _FakeModel())
    monkeypatch.setattr(
        "vector_store.index_embedded_jobs",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not index")),
    )

    profile = "embedder_disabled"
    init_db(profile=profile)
    job = _sample_job()
    assert insert_job(job, profile=profile) is True
    save_score(
        job_id=job.id,
        fit_score=78,
        role_fit=8,
        stack_match=8,
        seniority=7,
        loc_score=10,
        growth=6,
        compensation=8,
        reasons='["strong python fit"]',
        flags="[]",
        skill_misses="[]",
        one_liner="Strong backend match",
        ats_score=75,
        profile=profile,
    )

    config = {
        "embeddings": {
            "enabled": True,
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 2,
            "vector_store": {"enabled": False},
        }
    }

    result = embed_jobs(config, profile=profile, force=False)

    assert result["vector_index"]["status"] == "disabled"
    assert result["vector_index"]["chunks_indexed"] == 0

    shutil.rmtree(root)
