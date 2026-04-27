"""
main.py - CLI entry point for the Job Agent pipeline.

Usage:
  python main.py                  # full run: scrape + score + display
  python main.py --scrape-only    # scrape and save to DB, skip scoring
  python main.py --score-only     # score unscored jobs in DB, skip scraping
  python main.py --embed-only     # regenerate embeddings for scored jobs only
  python main.py --show           # print current top jobs, no scraping/scoring
  python main.py --rescore        # clear scores and re-score everything
  python main.py --yes            # skip all confirmation prompts
  python main.py --min-score 75   # override min display score for this run
  python main.py --profile sister # load from profiles/sister/config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from db import count_jobs, get_top_jobs, init_db, load_config, set_active_profile
from logging_config import configure_logging
from pipeline import PipelineOptions, run_full_pipeline
from reranker import reranking_enabled, reranking_top_k_final, semantic_match_jobs
from scorer import print_results
from vector_store import (
    clear_vector_index,
    query_similar_jobs,
    rebuild_vector_index,
    vector_store_enabled,
    vector_top_k_chunks,
    vector_top_k_jobs,
)

# Ensure Unicode output works on Windows terminals.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

load_dotenv()

_PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Job Agent - scrape, score, and display job matches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--scrape-only", action="store_true", help="Scrape and save to DB, skip scoring")
    mode.add_argument("--score-only", action="store_true", help="Score unscored jobs in DB, skip scraping")
    mode.add_argument("--embed-only", action="store_true", help="Generate embeddings for scored jobs in DB, skip scraping/scoring")
    mode.add_argument("--vector-search", type=str, metavar="QUERY", help="Semantic search over embedded job chunks, skip scraping/scoring")
    mode.add_argument("--semantic-search", type=str, metavar="QUERY", help="Profile-aware semantic search, optionally reranked with a cross-encoder")
    mode.add_argument("--semantic-match", action="store_true", help="Run profile-aware matching against indexed jobs, skip scraping/scoring")
    mode.add_argument("--rebuild-vector-index", action="store_true", help="Rebuild the ChromaDB vector index from SQLite embeddings")
    mode.add_argument("--clear-vector-index", action="store_true", help="Delete the profile-scoped ChromaDB vector index")
    mode.add_argument("--show", action="store_true", help="Print current top jobs from DB, no scraping or scoring")

    parser.add_argument("--rescore", action="store_true", help="Clear all scores and re-score everything")
    parser.add_argument("--yes", action="store_true", help="Skip all confirmation prompts (non-interactive / cron)")
    parser.add_argument("--min-score", type=int, default=None, metavar="N", help="Override min_display_score from config for this run")
    parser.add_argument("--rerank", action="store_true", help="Use the cross-encoder reranker for semantic search")
    parser.add_argument("--top-k", type=int, default=None, metavar="N", help="Override the number of semantic results to print")
    parser.add_argument("--profile", type=str, default=None, metavar="NAME", help="Load config from profiles/<NAME>/config.yaml")
    parser.add_argument("--debug", action="store_true", help="Enable debug-level console output")
    return parser.parse_args()


def _check_api_key(provider: str, profile: str | None = None) -> None:
    env_var = _PROVIDER_ENV_VARS.get(provider)
    if not env_var:
        return
    key = os.environ.get(f"{env_var}_{profile.upper()}") if profile else None
    if not key:
        key = os.environ.get(env_var)
    if not key:
        if profile:
            logger.error("Scoring requires {}_{} or {} but neither is set.", env_var, profile.upper(), env_var)
        else:
            logger.error("Scoring requires {} but it is not set.", env_var)
        logger.error("Add it to your .env file:  {}=your_key_here", env_var)
        sys.exit(1)


def _print_banner(config: dict) -> None:
    provider = config["llm"]["provider"]
    model = config["llm"]["model"].get(provider, "unknown")
    sources = config.get("sources", {})

    enabled_names = []
    for key, label, default in [
        ("greenhouse", "Greenhouse", True),
        ("lever", "Lever", True),
        ("hn", "HN", False),
        ("ashby", "Ashby", False),
        ("workable", "Workable", False),
        ("himalayas", "Himalayas", False),
    ]:
        if sources.get(key, {}).get("enabled", default):
            enabled_names.append(label)

    stats = count_jobs()
    logger.info("Job Agent")
    logger.info("Provider: {} ({})", provider, model)
    logger.info("Sources: {}", "  ".join(enabled_names) if enabled_names else "none")
    logger.info("DB: {} jobs total, {} scored, {} embedded", stats["total"], stats["scored"], stats.get("embedded", 0))


def _print_vector_results(results, query: str) -> None:
    if not results:
        logger.info("No semantic matches found for query: {}", query)
        return

    print(f"\n{'=' * 68}")
    print(f"  VECTOR SEARCH RESULTS ({len(results)} jobs)")
    print(f"{'=' * 68}\n")

    for result in results:
        similarity_pct = round(result.aggregate_score * 100)
        matched = ", ".join(result.matched_chunks) or "none"
        print(f"- {result.title} - {result.company}")
        print(f"  Similarity: {similarity_pct}%  |  Best chunk: {round(result.best_similarity * 100)}%")
        print(f"  Matched sections: {matched}")
        print(f"  Why: {result.retrieval_reason}")
        if result.url:
            print(f"  {result.url}")
        print()


def _print_reranked_results(results, query_label: str) -> None:
    if not results:
        logger.info("No semantic matches found for {}.", query_label)
        return

    print(f"\n{'=' * 72}")
    print(f"  SEMANTIC MATCH RESULTS ({len(results)} jobs)")
    print(f"{'=' * 72}\n")

    for result in results:
        final_pct = round(result.final_score * 100)
        vector_pct = round(result.vector_score * 100)
        rerank_pct = round(result.rerank_score * 100)
        matched = ", ".join(result.matched_sections) or "none"
        print(f"- {result.title} - {result.company}")
        print(f"  Final: {final_pct}%  |  Vector: {vector_pct}%  |  Rerank: {rerank_pct}%")
        print(f"  Matched sections: {matched}")
        print(f"  Why: {result.match_reason}")
        ev = getattr(result, "evidence", None)
        if ev is not None and ev.positive:
            print(f"  Positive: {' | '.join(ev.positive)}")
        if ev is not None and ev.concerns:
            print(f"  Concerns: {' | '.join(ev.concerns)}")
        for snippet in result.evidence_snippets[:2]:
            if snippet:
                print(f"  Evidence: {snippet}")
        if result.url:
            print(f"  {result.url}")
        print()


def _fallback_to_sqlite_results(config: dict, context: str, exc: Exception) -> None:
    logger.warning("{} failed: {}", context, exc)
    logger.info("Falling back to SQLite-ranked results.")
    print_results(get_top_jobs(config["scoring"].get("min_display_score", 60)), config)


def _handle_profile_arg(profile: str | None) -> None:
    if not profile:
        return
    profiles_dir = Path(__file__).parent / "profiles"
    if not (profiles_dir / profile).exists():
        logger.error("Profile '{}' not found.", profile)
        logger.error("Run the dashboard to create it: streamlit run dashboard.py")
        logger.error("Expected location: profiles/{}/config.yaml", profile)
        sys.exit(1)
    set_active_profile(profile)


def _maybe_confirm_rescore(args: argparse.Namespace) -> bool:
    if not args.rescore:
        return True
    if args.embed_only:
        logger.error("--rescore cannot be combined with --embed-only.")
        sys.exit(1)
    if args.score_only or args.yes:
        return True
    try:
        confirm = input("--rescore will delete all existing scores. Continue? [y/n]: ").strip().lower()
    except EOFError:
        confirm = "y"
    if confirm != "y":
        logger.info("Aborted rescore.")
        return False
    return True


def main() -> None:
    args = parse_args()
    _handle_profile_arg(args.profile)

    config = load_config(profile=args.profile)
    config["_active_profile"] = args.profile
    init_db(profile=args.profile)
    configure_logging(profile=args.profile or "default", debug=args.debug)

    if args.min_score is not None:
        config["scoring"]["min_display_score"] = args.min_score

    provider = config["llm"]["provider"]

    if args.show:
        _print_banner(config)
        results = get_top_jobs(config["scoring"].get("min_display_score", 60))
        if not results:
            logger.info("No scored jobs above {} in DB.", config["scoring"].get("min_display_score", 60))
        else:
            print_results(results, config)
        return

    if args.clear_vector_index:
        _print_banner(config)
        if not vector_store_enabled(config):
            logger.info("Vector store is disabled in this profile config; nothing to clear.")
            return
        clear_vector_index(args.profile or "default")
        logger.info("Cleared vector index for profile '{}'.", args.profile or "default")
        return

    if args.rebuild_vector_index:
        _print_banner(config)
        if not vector_store_enabled(config):
            logger.info("Vector store is disabled in this profile config; nothing to rebuild.")
            return
        result = rebuild_vector_index(args.profile or "default")
        logger.info("Rebuilt vector index: {} chunks across {} jobs", result.get("chunks_indexed", 0), result.get("jobs_indexed", 0))
        return

    if args.vector_search:
        _print_banner(config)
        if not vector_store_enabled(config):
            logger.info("Vector store is disabled in this profile config; semantic search is unavailable.")
            return
        try:
            results = query_similar_jobs(
                args.profile or "default",
                args.vector_search,
                top_k_chunks=vector_top_k_chunks(config),
                top_k_jobs=args.top_k or vector_top_k_jobs(config),
            )
            _print_vector_results(results, args.vector_search)
        except Exception as exc:
            _fallback_to_sqlite_results(config, "Vector search", exc)
        return

    if args.semantic_search:
        _print_banner(config)
        if not vector_store_enabled(config):
            logger.info("Vector store is disabled in this profile config; semantic search is unavailable.")
            return
        try:
            if args.rerank:
                results = semantic_match_jobs(args.profile or "default", config, user_query=args.semantic_search)
                limit = args.top_k or reranking_top_k_final(config)
                _print_reranked_results(results[:limit], f"query '{args.semantic_search}'")
            else:
                results = query_similar_jobs(
                    args.profile or "default",
                    args.semantic_search,
                    top_k_chunks=vector_top_k_chunks(config),
                    top_k_jobs=args.top_k or vector_top_k_jobs(config),
                )
                _print_vector_results(results, args.semantic_search)
        except Exception as exc:
            _fallback_to_sqlite_results(config, "Semantic search", exc)
        return

    if args.semantic_match:
        _print_banner(config)
        if not vector_store_enabled(config):
            logger.info("Vector store is disabled in this profile config; semantic matching is unavailable.")
            return
        try:
            results = semantic_match_jobs(args.profile or "default", config, user_query=None)
            limit = args.top_k or reranking_top_k_final(config)
            if not reranking_enabled(config):
                logger.info("Reranking is disabled in config; falling back to vector-only semantic matching.")
            _print_reranked_results(results[:limit], "the active profile")
        except Exception as exc:
            _fallback_to_sqlite_results(config, "Semantic match", exc)
        return

    will_score = not (
        args.scrape_only
        or args.embed_only
        or args.vector_search
        or args.semantic_search
        or args.semantic_match
        or args.rebuild_vector_index
        or args.clear_vector_index
    )
    if will_score:
        _check_api_key(provider, args.profile)

    if not _maybe_confirm_rescore(args):
        return

    _print_banner(config)

    try:
        result = run_full_pipeline(
            config,
            args.profile,
            PipelineOptions(
                scrape=not args.score_only and not args.embed_only,
                score=not args.scrape_only and not args.embed_only,
                embed=args.embed_only or (not args.scrape_only and not args.embed_only),
                yes=args.yes,
                force_embed=args.embed_only,
                rescore=args.rescore,
                run_source="cli",
            ),
        )

        if result.score.results:
            print_results(result.score.results, config)

        logger.info("Run complete")
        logger.info(
            "DB: {} jobs total, {} scored, {} embedded",
            result.final_db_stats["total"],
            result.final_db_stats["scored"],
            result.final_db_stats.get("embedded", 0),
        )
        if not args.score_only and not args.embed_only:
            logger.info("New jobs this run: {}", result.scrape.total_new)
        if result.score.jobs_scored:
            logger.info("Scored this run: {}", result.score.jobs_scored)
            logger.info("Avg fit score: {:.0f}", result.score.avg_fit_score)
            logger.info("Avg ATS score: {:.0f}", result.score.avg_ats_score)
        if result.embed.jobs_total:
            logger.info("Embedded this run: {} jobs, {} chunks", result.embed.jobs_embedded, result.embed.chunks_embedded)
        if result.embed.vector_index.get("enabled"):
            logger.info(
                "Indexed this run: {} chunks across {} jobs",
                result.embed.vector_index.get("chunks_indexed", 0),
                result.embed.vector_index.get("jobs_indexed", 0),
            )
            if result.embed.vector_index.get("status") == "failed" and result.embed.vector_index.get("error"):
                logger.warning("Vector indexing error: {}", result.embed.vector_index["error"])
        logger.info("{}", "-" * 44)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        stats = count_jobs(profile=args.profile)
        logger.info("DB state: {} jobs total, {} scored", stats["total"], stats["scored"])
        raise
    except Exception as exc:
        logger.error("Pipeline crashed: {}", exc)
        stats = count_jobs(profile=args.profile)
        logger.info("DB state: {} jobs total, {} scored", stats["total"], stats["scored"])
        raise


if __name__ == "__main__":
    main()
