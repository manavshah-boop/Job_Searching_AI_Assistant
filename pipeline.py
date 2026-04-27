"""
pipeline.py - Shared orchestration for scraping, scoring, and embedding.

This module is the single source of truth for pipeline stage ordering.
It coordinates specialized modules but does not own their implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from loguru import logger

from db import count_jobs, finish_run, rescore_reset, start_run
from embedder import embed_jobs, embeddings_enabled
from progress_tracker import ActivityType, ProgressTracker, Stage
from scraper import (
    scrape_ashby,
    scrape_greenhouse,
    scrape_himalayas,
    scrape_hn,
    scrape_lever,
    scrape_workable,
)
from scorer import score_all_jobs
from theirstack import get_or_discover_slugs


@dataclass
class ScrapeStats:
    total_new: int = 0
    jobs_scraped: int = 0
    jobs_filtered: int = 0
    jobs_saved: int = 0
    discovered_slugs: dict[str, int] = field(default_factory=dict)
    source_results: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class ScoreStats:
    results: list[dict[str, Any]] = field(default_factory=list)
    jobs_scored: int = 0
    avg_fit_score: float = 0.0
    avg_ats_score: float = 0.0


@dataclass
class EmbedStats:
    enabled: bool = True
    jobs_embedded: int = 0
    jobs_total: int = 0
    chunks_embedded: int = 0
    model_name: str = ""
    vector_index: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineOptions:
    scrape: bool = True
    score: bool = True
    embed: bool = True
    yes: bool = False
    force_embed: bool = False
    rescore: bool = False
    run_source: str = "pipeline"
    on_job_scored: Optional[Callable[[int, int, dict[str, Any]], None]] = None
    on_job_embedded: Optional[Callable[[int, int, Any, int], None]] = None
    on_progress: Optional[Callable[[ProgressTracker], None]] = None


@dataclass
class PipelineResult:
    run_id: Optional[str]
    status: str
    errors: list[str]
    scrape: ScrapeStats
    score: ScoreStats
    embed: EmbedStats
    final_db_stats: dict[str, int]


@dataclass(frozen=True)
class _SourceSpec:
    key: str
    tracker_label: str
    metric_key: str
    requires_slugs: bool
    scraper: Callable[..., dict[str, Any]]


_SOURCE_SPECS: tuple[_SourceSpec, ...] = (
    _SourceSpec("greenhouse", "Greenhouse", "greenhouse_new", True, scrape_greenhouse),
    _SourceSpec("lever", "Lever", "lever_new", True, scrape_lever),
    _SourceSpec("hn", "HN Who's Hiring", "hn_new", False, scrape_hn),
    _SourceSpec("ashby", "Ashby", "ashby_new", True, scrape_ashby),
    _SourceSpec("workable", "Workable", "workable_new", True, scrape_workable),
    _SourceSpec("himalayas", "Himalayas", "himalayas_new", False, scrape_himalayas),
)


def _notify_progress(
    progress_tracker: Optional[ProgressTracker],
    callback: Optional[Callable[[ProgressTracker], None]],
) -> None:
    if progress_tracker is None or callback is None:
        return
    callback(progress_tracker)


def _enabled_sources(config: dict[str, Any]) -> dict[str, bool]:
    sources = config.get("sources", {})
    return {
        "greenhouse": sources.get("greenhouse", {}).get("enabled", True),
        "lever": sources.get("lever", {}).get("enabled", True),
        "hn": sources.get("hn", {}).get("enabled", False),
        "ashby": sources.get("ashby", {}).get("enabled", False),
        "workable": sources.get("workable", {}).get("enabled", False),
        "himalayas": sources.get("himalayas", {}).get("enabled", False),
    }


def _company_count(config: dict[str, Any], key: str) -> int:
    if key in {"hn", "himalayas"}:
        return 1
    return len(config.get("sources", {}).get(key, {}).get("companies", []))


def _result_to_source_stats(result: dict[str, Any]) -> dict[str, int]:
    return {
        "new_jobs_saved": int(result.get("new_jobs_saved", 0) or 0),
        "jobs_scraped": int(result.get("jobs_scraped", 0) or 0),
        "jobs_filtered": int(result.get("jobs_filtered", 0) or 0),
    }


def run_scrapers(
    config: dict[str, Any],
    profile: str | None,
    *,
    progress_tracker: Optional[ProgressTracker] = None,
    on_progress: Optional[Callable[[ProgressTracker], None]] = None,
) -> ScrapeStats:
    enabled = _enabled_sources(config)
    if not any(enabled.values()):
        logger.error("pipeline | no sources are enabled in config")
        raise ValueError("No sources are enabled in config.yaml. Nothing to scrape.")

    if progress_tracker is not None:
        progress_tracker.start_stage(Stage.DISCOVERING)
        _notify_progress(progress_tracker, on_progress)

    slug_map = {"greenhouse": [], "lever": [], "ashby": [], "workable": []}
    if any(enabled[key] for key in slug_map):
        slug_map = get_or_discover_slugs(config, profile=profile)

    scrape_stats = ScrapeStats(
        discovered_slugs={key: len(slug_map.get(key, [])) for key in slug_map},
    )

    if progress_tracker is not None:
        progress_tracker.set_stage_metrics(Stage.DISCOVERING, **scrape_stats.discovered_slugs)
        progress_tracker.complete_stage(Stage.DISCOVERING)
        progress_tracker.start_stage(Stage.FETCHING)
        _notify_progress(progress_tracker, on_progress)

    for spec in _SOURCE_SPECS:
        if not enabled.get(spec.key, False):
            continue

        if progress_tracker is not None:
            progress_tracker.register_source(spec.tracker_label, _company_count(config, spec.key))
            progress_tracker.start_source(spec.tracker_label)
            _notify_progress(progress_tracker, on_progress)

        if spec.requires_slugs:
            result = spec.scraper(config, slugs=slug_map.get(spec.key, []), profile=profile)
        else:
            result = spec.scraper(config, profile=profile)

        source_stats = _result_to_source_stats(result)
        scrape_stats.source_results[spec.key] = source_stats
        scrape_stats.total_new += source_stats["new_jobs_saved"]
        scrape_stats.jobs_scraped += source_stats["jobs_scraped"]
        scrape_stats.jobs_filtered += source_stats["jobs_filtered"]
        scrape_stats.jobs_saved += source_stats["new_jobs_saved"]

        logger.info(
            "pipeline | scrape summary source={} new={} scraped={} filtered={}",
            spec.key,
            source_stats["new_jobs_saved"],
            source_stats["jobs_scraped"],
            source_stats["jobs_filtered"],
        )

        if progress_tracker is not None:
            progress_tracker.complete_source(spec.tracker_label, jobs_found=source_stats["new_jobs_saved"])
            progress_tracker.set_stage_metrics(Stage.FETCHING, **{spec.metric_key: source_stats["new_jobs_saved"]})
            _notify_progress(progress_tracker, on_progress)

    if progress_tracker is not None:
        progress_tracker.complete_stage(Stage.FETCHING)
        progress_tracker.start_stage(Stage.SCRAPING)
        progress_tracker.complete_stage(Stage.SCRAPING)
        _notify_progress(progress_tracker, on_progress)

    logger.info(
        "pipeline | scrape complete new={} scraped={} filtered={} saved={}",
        scrape_stats.total_new,
        scrape_stats.jobs_scraped,
        scrape_stats.jobs_filtered,
        scrape_stats.jobs_saved,
    )
    return scrape_stats


def run_scoring(
    config: dict[str, Any],
    profile: str | None,
    *,
    yes: bool = False,
    on_job_scored: Optional[Callable[[int, int, dict[str, Any]], None]] = None,
) -> ScoreStats:
    results = score_all_jobs(config, yes=yes, profile=profile, on_job_scored=on_job_scored)
    scored = [result for result in results if result.get("fit_score", 0) > 0]
    avg_fit = sum(result["fit_score"] for result in scored) / len(scored) if scored else 0.0
    avg_ats = sum(result["ats_score"] for result in scored) / len(scored) if scored else 0.0

    logger.info(
        "pipeline | scoring complete total_results={} scored={} avg_fit={:.1f} avg_ats={:.1f}",
        len(results),
        len(scored),
        avg_fit,
        avg_ats,
    )
    return ScoreStats(
        results=results,
        jobs_scored=len(scored),
        avg_fit_score=avg_fit,
        avg_ats_score=avg_ats,
    )


def run_embedding(
    config: dict[str, Any],
    profile: str | None,
    *,
    force: bool = False,
    on_job_embedded: Optional[Callable[[int, int, Any, int], None]] = None,
) -> EmbedStats:
    if not embeddings_enabled(config):
        logger.info("pipeline | embedding skipped because embeddings are disabled")
        return EmbedStats(enabled=False)

    result = embed_jobs(
        config,
        profile=profile,
        force=force,
        on_job_embedded=on_job_embedded,
    )

    embed_stats = EmbedStats(
        enabled=bool(result.get("enabled", True)),
        jobs_embedded=int(result.get("jobs_embedded", 0) or 0),
        jobs_total=int(result.get("jobs_total", 0) or 0),
        chunks_embedded=int(result.get("chunks_embedded", 0) or 0),
        model_name=str(result.get("model_name", "") or ""),
        vector_index=dict(result.get("vector_index", {}) or {}),
    )
    logger.info(
        "pipeline | embedding complete jobs_embedded={} jobs_total={} chunks_embedded={}",
        embed_stats.jobs_embedded,
        embed_stats.jobs_total,
        embed_stats.chunks_embedded,
    )
    return embed_stats


def run_full_pipeline(
    config: dict[str, Any],
    profile: str | None,
    options: PipelineOptions,
    progress_tracker: Optional[ProgressTracker] = None,
) -> PipelineResult:
    run_id = start_run(profile=profile, source=options.run_source)
    scrape_stats = ScrapeStats()
    score_stats = ScoreStats()
    embed_stats = EmbedStats(enabled=embeddings_enabled(config))
    errors: list[str] = []
    status = "complete"
    active_stage: Stage | None = None

    logger.info(
        "pipeline | start profile={} source={} scrape={} score={} embed={} force_embed={} rescore={}",
        profile or "default",
        options.run_source,
        options.scrape,
        options.score,
        options.embed,
        options.force_embed,
        options.rescore,
    )

    try:
        if options.rescore:
            logger.info("pipeline | resetting scores before scoring")
            rescore_reset(profile=profile)

        if options.scrape:
            scrape_stats = run_scrapers(
                config,
                profile,
                progress_tracker=progress_tracker,
                on_progress=options.on_progress,
            )

        if options.score:
            active_stage = Stage.SCORING
            if progress_tracker is not None:
                progress_tracker.start_stage(Stage.SCORING)
                _notify_progress(progress_tracker, options.on_progress)

            def _wrapped_on_job_scored(i: int, total: int, result: dict[str, Any]) -> None:
                if progress_tracker is not None:
                    progress_tracker.set_stage_metrics(Stage.SCORING, scored=i, total=total)
                    _notify_progress(progress_tracker, options.on_progress)
                if options.on_job_scored is not None:
                    options.on_job_scored(i, total, result)

            score_stats = run_scoring(
                config,
                profile,
                yes=options.yes,
                on_job_scored=_wrapped_on_job_scored if (progress_tracker or options.on_job_scored) else None,
            )

            if progress_tracker is not None:
                progress_tracker.set_stage_metrics(
                    Stage.SCORING,
                    scored=score_stats.jobs_scored,
                    avg_fit=round(score_stats.avg_fit_score, 1),
                    avg_ats=round(score_stats.avg_ats_score, 1),
                )
                progress_tracker.complete_stage(Stage.SCORING)
                _notify_progress(progress_tracker, options.on_progress)

        if options.embed:
            active_stage = Stage.EMBEDDING
            if progress_tracker is not None:
                progress_tracker.start_stage(Stage.EMBEDDING)
                _notify_progress(progress_tracker, options.on_progress)

            def _wrapped_on_job_embedded(i: int, total: int, job: Any, chunk_count: int) -> None:
                if progress_tracker is not None:
                    progress_tracker.set_stage_metrics(
                        Stage.EMBEDDING,
                        embedded=i,
                        total=total,
                        last_chunks=chunk_count,
                    )
                    _notify_progress(progress_tracker, options.on_progress)
                if options.on_job_embedded is not None:
                    options.on_job_embedded(i, total, job, chunk_count)

            embed_stats = run_embedding(
                config,
                profile,
                force=options.force_embed,
                on_job_embedded=_wrapped_on_job_embedded if (progress_tracker or options.on_job_embedded) else None,
            )

            if progress_tracker is not None:
                embedding_metrics: dict[str, Any] = {
                    "embedded": embed_stats.jobs_embedded,
                    "total": embed_stats.jobs_total,
                    "chunks": embed_stats.chunks_embedded,
                }
                if not embed_stats.enabled:
                    embedding_metrics["skipped"] = True
                if embed_stats.vector_index:
                    embedding_metrics.update(
                        chunks_indexed=embed_stats.vector_index.get("chunks_indexed", 0),
                        jobs_indexed=embed_stats.vector_index.get("jobs_indexed", 0),
                        vector_index_status=embed_stats.vector_index.get("status", "disabled"),
                        vector_index_error=embed_stats.vector_index.get("error", ""),
                    )
                progress_tracker.set_stage_metrics(Stage.EMBEDDING, **embedding_metrics)
                progress_tracker.complete_stage(Stage.EMBEDDING)
                _notify_progress(progress_tracker, options.on_progress)

        final_stats = count_jobs(profile=profile)

    except Exception as exc:
        status = "failed"
        errors.append(str(exc))
        if progress_tracker is not None and active_stage is not None:
            progress_tracker.fail_stage(active_stage, str(exc))
            _notify_progress(progress_tracker, options.on_progress)
        logger.exception("pipeline | failed")
        final_stats = count_jobs(profile=profile)
        raise

    finally:
        finish_run(
            run_id,
            jobs_scraped=scrape_stats.jobs_scraped,
            jobs_filtered=scrape_stats.jobs_filtered,
            jobs_saved=scrape_stats.jobs_saved,
            jobs_scored=score_stats.jobs_scored,
            avg_fit_score=score_stats.avg_fit_score if score_stats.jobs_scored else 0.0,
            errors=errors,
            status=status,
            profile=profile,
        )
        if progress_tracker is not None:
            progress_tracker.start_stage(Stage.FINALIZING)
            progress_tracker.total_jobs_new = scrape_stats.total_new
            progress_tracker.log_activity(
                f"Pipeline complete: {scrape_stats.total_new} new jobs and {score_stats.jobs_scored} scored.",
                ActivityType.METRIC_UPDATE,
            )
            progress_tracker.complete_stage(Stage.FINALIZING)
            _notify_progress(progress_tracker, options.on_progress)

    logger.info(
        "pipeline | complete profile={} total_jobs={} scored={} embedded={} status={}",
        profile or "default",
        final_stats.get("total", 0),
        final_stats.get("scored", 0),
        final_stats.get("embedded", 0),
        status,
    )
    return PipelineResult(
        run_id=run_id,
        status=status,
        errors=errors,
        scrape=scrape_stats,
        score=score_stats,
        embed=embed_stats,
        final_db_stats=final_stats,
    )
