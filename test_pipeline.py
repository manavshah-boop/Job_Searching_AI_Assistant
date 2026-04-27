from __future__ import annotations

from progress_tracker import ProgressTracker, StageStatus

import pipeline
from pipeline import PipelineOptions


def test_run_full_pipeline_orchestrates_enabled_stages(monkeypatch):
    calls = []
    tracker = ProgressTracker()

    monkeypatch.setattr(pipeline, "start_run", lambda profile=None, source=None: calls.append(("start_run", profile, source)) or "run-1")
    monkeypatch.setattr(pipeline, "finish_run", lambda run_id, **kwargs: calls.append(("finish_run", run_id, kwargs["profile"], kwargs["jobs_scored"])))
    monkeypatch.setattr(pipeline, "count_jobs", lambda profile=None: {"total": 12, "scored": 4, "embedded": 3})
    monkeypatch.setattr(
        pipeline,
        "run_scrapers",
        lambda config, profile, progress_tracker=None, on_progress=None: calls.append(("scrape", profile, progress_tracker is tracker)) or pipeline.ScrapeStats(total_new=5, jobs_scraped=9, jobs_filtered=2, jobs_saved=5),
    )
    monkeypatch.setattr(
        pipeline,
        "run_scoring",
        lambda config, profile, yes=False, on_job_scored=None: calls.append(("score", profile, yes, on_job_scored is not None)) or pipeline.ScoreStats(results=[{"fit_score": 82, "ats_score": 76}], jobs_scored=1, avg_fit_score=82.0, avg_ats_score=76.0),
    )
    monkeypatch.setattr(
        pipeline,
        "run_embedding",
        lambda config, profile, force=False, on_job_embedded=None: calls.append(("embed", profile, force, on_job_embedded is not None)) or pipeline.EmbedStats(enabled=True, jobs_embedded=1, jobs_total=1, chunks_embedded=3, vector_index={"enabled": True, "status": "indexed", "chunks_indexed": 3, "jobs_indexed": 1}),
    )

    result = pipeline.run_full_pipeline(
        {"embeddings": {"enabled": True}},
        "alpha",
        PipelineOptions(scrape=True, score=True, embed=True, yes=True, run_source="cli"),
        progress_tracker=tracker,
    )

    assert result.status == "complete"
    assert result.scrape.total_new == 5
    assert result.score.jobs_scored == 1
    assert result.embed.chunks_embedded == 3
    assert tracker.stages[pipeline.Stage.FINALIZING].status == StageStatus.COMPLETE
    assert calls == [
        ("start_run", "alpha", "cli"),
        ("scrape", "alpha", True),
        ("score", "alpha", True, True),
        ("embed", "alpha", False, True),
        ("finish_run", "run-1", "alpha", 1),
    ]


def test_run_full_pipeline_skips_disabled_embedding_stage(monkeypatch):
    monkeypatch.setattr(pipeline, "start_run", lambda profile=None, source=None: "run-2")
    monkeypatch.setattr(pipeline, "finish_run", lambda run_id, **kwargs: None)
    monkeypatch.setattr(pipeline, "count_jobs", lambda profile=None: {"total": 3, "scored": 0, "embedded": 0})
    monkeypatch.setattr(
        pipeline,
        "run_scoring",
        lambda config, profile, yes=False, on_job_scored=None: pipeline.ScoreStats(results=[], jobs_scored=0),
    )

    result = pipeline.run_full_pipeline(
        {"embeddings": {"enabled": False}},
        "beta",
        PipelineOptions(scrape=False, score=True, embed=False, yes=True, run_source="worker"),
    )

    assert result.embed.enabled is False
    assert result.final_db_stats == {"total": 3, "scored": 0, "embedded": 0}
