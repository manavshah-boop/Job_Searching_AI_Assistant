"""
test_reranker.py - Unit tests for profile-aware cross-encoder reranking.
"""

from __future__ import annotations

from reranker import (
    build_profile_match_query,
    rerank_chunks,
    rerank_jobs,
    select_chunks_for_reranking,
    select_jobs_for_llm_scoring,
    semantic_match_jobs,
)
from vector_store import MatchedChunkScore, VectorChunkResult, VectorJobResult


class _FakeCrossEncoder:
    def __init__(self, scores):
        self._scores = list(scores)
        self.calls = []

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        materialized = list(pairs)
        self.calls.append(
            {
                "pairs": materialized,
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
            }
        )
        return self._scores[: len(materialized)]


def _config(*, rerank_enabled: bool = True) -> dict:
    return {
        "profile": {
            "name": "Manav Shah",
            "job_type": "fulltime",
            "resume": "Python FastAPI SQL AWS backend APIs React platform systems",
            "bio": "Backend and AI platform engineer",
        },
        "preferences": {
            "titles": ["Backend Engineer", "AI Platform Engineer"],
            "desired_skills": ["Python", "FastAPI", "SQL", "AWS", "React", "Backend systems"],
            "hard_no_keywords": ["security clearance required", "non-US relocation"],
            "location": {"remote_ok": True, "preferred_locations": ["Seattle, WA", "New York, NY"]},
            "compensation": {"min_salary": 130000},
            "yoe": 3,
            "filters": {"title_blocklist": ["Senior", "Staff", "Principal"]},
        },
        "embeddings": {"vector_store": {"enabled": True}},
        "reranking": {
            "enabled": rerank_enabled,
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "batch_size": 16,
            "top_k_vector_chunks": 80,
            "top_k_vector_jobs": 40,
            "top_k_final": 15,
            "max_chunks_per_job": 4,
            "max_chunk_chars": 900,
            "include_profile_context": True,
            "include_query_context": True,
        },
    }


def _chunk(job_id: str, chunk_key: str, similarity: float, text: str, *, title: str = "Role", company: str = "Acme") -> VectorChunkResult:
    return VectorChunkResult(
        chroma_id=f"{job_id}:{chunk_key}",
        job_id=job_id,
        chunk_key=chunk_key,
        chunk_order={"summary": 0, "responsibilities": 1, "requirements": 2, "compensation": 3, "benefits": 4}.get(chunk_key, 0),
        title=title,
        company=company,
        source="ashby",
        url=f"https://example.com/{job_id}",
        distance=1.0 - similarity,
        similarity=similarity,
        chunk_text=text,
    )


def _job(job_id: str, aggregate: float, best: float, matched: list[str], *, title: str = "Role", company: str = "Acme") -> VectorJobResult:
    return VectorJobResult(
        job_id=job_id,
        title=title,
        company=company,
        source="ashby",
        url=f"https://example.com/{job_id}",
        best_similarity=best,
        aggregate_score=aggregate,
        matched_chunks=matched,
        chunk_scores=[
            MatchedChunkScore(
                chunk_key=chunk_key,
                chunk_order=index,
                similarity=best,
                weighted_similarity=best,
            )
            for index, chunk_key in enumerate(matched)
        ],
        retrieval_reason="Vector recall match",
    )


def test_build_profile_match_query_includes_roles_skills_locations_and_user_query():
    query = build_profile_match_query(_config(), user_query="backend AI platform role with Python and AWS")

    assert "Backend Engineer" in query
    assert "3 years" in query
    assert "Python" in query and "AWS" in query
    assert "Seattle, WA" in query
    assert "remote" in query.lower()
    assert "backend AI platform role with Python and AWS" in query


def test_select_chunks_for_reranking_prioritizes_requirements_and_responsibilities():
    job = _job("job-1", 0.82, 0.92, ["benefits", "requirements", "responsibilities"])
    chunks = [
        _chunk("job-1", "benefits", 0.98, "Benefits include PTO and meals"),
        _chunk("job-1", "requirements", 0.81, "Requirements: Python APIs and SQL"),
        _chunk("job-1", "responsibilities", 0.80, "Responsibilities: build backend systems"),
        _chunk("job-1", "summary", 0.72, "Summary: backend role"),
    ]

    selected = select_chunks_for_reranking(job, chunks, 3, match_query="backend platform role")

    assert [chunk.chunk_key for chunk in selected] == ["requirements", "responsibilities", "summary"]


def test_compensation_chunks_get_boosted_for_salary_queries(monkeypatch):
    fake_model = _FakeCrossEncoder([1.0])
    monkeypatch.setattr("reranker.get_cross_encoder", lambda config: fake_model)
    chunk = _chunk("job-1", "compensation", 0.65, "Compensation: salary range is $150,000")

    normal = rerank_chunks("backend platform role", [chunk], _config())[0]
    salary = rerank_chunks("salary and compensation for backend role", [chunk], _config())[0]

    assert salary.weighted_score > normal.weighted_score


def test_batched_reranking_returns_one_score_per_selected_chunk(monkeypatch):
    fake_model = _FakeCrossEncoder([1.0, 0.5, -0.5])
    monkeypatch.setattr("reranker.get_cross_encoder", lambda config: fake_model)
    chunks = [
        _chunk("job-1", "requirements", 0.8, "Requirements: Python and SQL"),
        _chunk("job-1", "responsibilities", 0.7, "Responsibilities: build APIs"),
        _chunk("job-2", "summary", 0.6, "Summary: frontend role"),
    ]

    reranked = rerank_chunks("python backend role", chunks, _config())

    assert len(reranked) == 3
    assert len(fake_model.calls) == 1
    assert len(fake_model.calls[0]["pairs"]) == 3
    assert fake_model.calls[0]["batch_size"] == 16


def test_job_aggregation_ranks_requirements_and_responsibilities_above_benefits_only(monkeypatch):
    fake_model = _FakeCrossEncoder([2.0, 1.8, 2.0])
    monkeypatch.setattr("reranker.get_cross_encoder", lambda config: fake_model)
    vector_jobs = [
        _job("job-strong", 0.80, 0.86, ["requirements", "responsibilities"], title="Backend Engineer"),
        _job("job-benefits", 0.82, 0.87, ["benefits"], title="Benefits Heavy Role"),
    ]
    chunks = [
        _chunk("job-strong", "requirements", 0.86, "Requirements: Python SQL AWS APIs", title="Backend Engineer"),
        _chunk("job-strong", "responsibilities", 0.82, "Responsibilities: build backend systems and APIs", title="Backend Engineer"),
        _chunk("job-benefits", "benefits", 0.87, "Benefits: PTO, medical, wellness stipend", title="Benefits Heavy Role"),
    ]

    ranked = rerank_jobs("python backend role with aws", vector_jobs, chunks, _config())

    assert ranked
    assert ranked[0].job_id == "job-strong"
    assert ranked[0].best_chunk_key in {"requirements", "responsibilities"}


def test_empty_vector_results_return_empty_rerank_results():
    assert rerank_jobs("python backend role", [], [], _config()) == []


def test_reranking_disabled_falls_back_cleanly(monkeypatch):
    config = _config(rerank_enabled=False)
    vector_results = [_job("job-1", 0.77, 0.8, ["requirements"], title="Backend Engineer")]
    monkeypatch.setattr("reranker.query_similar_jobs", lambda *args, **kwargs: vector_results)
    monkeypatch.setattr(
        "reranker.query_similar_chunks",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not rerank")),
    )

    results = semantic_match_jobs("default", config, user_query="python backend role")

    assert len(results) == 1
    assert results[0].job_id == "job-1"
    assert results[0].final_score == vector_results[0].aggregate_score


def test_semantic_match_uses_raw_user_query_for_vector_recall(monkeypatch):
    config = _config()
    captured = {"jobs_query": None, "chunks_query": None}
    vector_results = [_job("job-1", 0.77, 0.8, ["requirements"], title="Backend Engineer")]
    chunk_results = [_chunk("job-1", "requirements", 0.8, "Requirements: Python APIs and AWS", title="Backend Engineer")]

    monkeypatch.setattr(
        "reranker.query_similar_jobs",
        lambda profile, query, **kwargs: captured.__setitem__("jobs_query", query) or vector_results,
    )
    monkeypatch.setattr(
        "reranker.query_similar_chunks",
        lambda profile, query, **kwargs: captured.__setitem__("chunks_query", query) or chunk_results,
    )
    monkeypatch.setattr(
        "reranker.rerank_jobs",
        lambda match_query, vector_job_results, chunk_results, config: [],
    )

    semantic_match_jobs("default", config, user_query="workload porting kernels")

    assert captured["jobs_query"] == "workload porting kernels"
    assert captured["chunks_query"] == "workload porting kernels"


def test_semantic_match_uses_compact_profile_recall_query_without_user_query(monkeypatch):
    config = _config()
    captured = {"jobs_query": None, "chunks_query": None}
    vector_results = [_job("job-1", 0.77, 0.8, ["requirements"], title="Backend Engineer")]
    chunk_results = [_chunk("job-1", "requirements", 0.8, "Requirements: Python APIs and AWS", title="Backend Engineer")]

    monkeypatch.setattr(
        "reranker.query_similar_jobs",
        lambda profile, query, **kwargs: captured.__setitem__("jobs_query", query) or vector_results,
    )
    monkeypatch.setattr(
        "reranker.query_similar_chunks",
        lambda profile, query, **kwargs: captured.__setitem__("chunks_query", query) or chunk_results,
    )
    monkeypatch.setattr(
        "reranker.rerank_jobs",
        lambda match_query, vector_job_results, chunk_results, config: [],
    )

    semantic_match_jobs("default", config, user_query=None)

    assert "Candidate seeking" not in captured["jobs_query"]
    assert "Backend Engineer" in captured["jobs_query"]
    assert "Python" in captured["jobs_query"]
    assert captured["jobs_query"] == captured["chunks_query"]


def test_select_jobs_for_llm_scoring_returns_top_ids_by_threshold():
    ranked_jobs = [
        type(
            "Result",
            (),
            {"job_id": "job-1", "final_score": 0.91},
        )(),
        type(
            "Result",
            (),
            {"job_id": "job-2", "final_score": 0.72},
        )(),
        type(
            "Result",
            (),
            {"job_id": "job-3", "final_score": 0.66},
        )(),
    ]

    selected = select_jobs_for_llm_scoring(ranked_jobs, threshold=0.70, top_k=2)

    assert selected == ["job-1", "job-2"]
