"""
main.py — Single entry point for the Job Agent pipeline.

Usage:
  python main.py                  # full run: scrape + score + display
  python main.py --scrape-only    # scrape and save to DB, skip scoring
  python main.py --score-only     # score unscored jobs in DB, skip scraping
  python main.py --show           # print current top jobs, no scraping/scoring
  python main.py --rescore        # clear scores and re-score everything
  python main.py --yes            # skip all confirmation prompts
  python main.py --min-score 75   # override min display score for this run
  python main.py --profile sister # load from profiles/sister/config.yaml (Step 7)
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure Unicode output works on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from dotenv import load_dotenv
load_dotenv()

from db import count_jobs, init_db, load_config, rescore_reset, set_active_profile, start_run, finish_run
from db import get_top_jobs
from scraper import scrape_ashby, scrape_greenhouse, scrape_himalayas, scrape_hn, scrape_lever, scrape_workable
from scorer import RateLimitReached, print_results, score_all_jobs
from theirstack import get_or_discover_slugs
from logging_config import configure_logging
from loguru import logger


# Map provider name -> expected environment variable
_PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini":    "GEMINI_API_KEY",
    "groq":      "GROQ_API_KEY",
    "openai":    "OPENAI_API_KEY",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Job Agent — scrape, score, and display job matches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--scrape-only", action="store_true",
                      help="Scrape and save to DB, skip scoring")
    mode.add_argument("--score-only",  action="store_true",
                      help="Score unscored jobs in DB, skip scraping")
    mode.add_argument("--show",        action="store_true",
                      help="Print current top jobs from DB, no scraping or scoring")

    parser.add_argument("--rescore",   action="store_true",
                        help="Clear all scores and re-score everything")
    parser.add_argument("--yes",       action="store_true",
                        help="Skip all confirmation prompts (non-interactive / cron)")
    parser.add_argument("--min-score", type=int, default=None, metavar="N",
                        help="Override min_display_score from config for this run")
    parser.add_argument("--profile",   type=str, default=None, metavar="NAME",
                        help="Load config from profiles/<NAME>/config.yaml (Step 7)")
    parser.add_argument("--debug",     action="store_true",
                        help="Enable debug-level console output")
    return parser.parse_args()


def _check_api_key(provider: str, profile: str | None = None) -> None:
    """Exit with a clear message if the required API key is missing."""
    env_var = _PROVIDER_ENV_VARS.get(provider)
    if not env_var:
        return
    key = os.environ.get(f"{env_var}_{profile.upper()}") if profile else None
    if not key:
        key = os.environ.get(env_var)
    if not key:
        if profile:
            logger.error(f"Scoring requires {env_var}_{profile.upper()} or {env_var} but neither is set.")
        else:
            logger.error(f"Scoring requires {env_var} but it is not set.")
        logger.error(f"Add it to your .env file:  {env_var}=your_key_here")
        logger.error(f"Or export it:              export {env_var}=your_key_here")
        sys.exit(1)


def _print_banner(config: dict) -> None:
    """Print startup banner showing provider, enabled sources, and DB stats."""
    provider = config["llm"]["provider"]
    model    = config["llm"]["model"].get(provider, "unknown")
    sources  = config.get("sources", {})

    enabled_names = []
    for key, label, default in [
        ("greenhouse", "Greenhouse", True),
        ("lever",      "Lever",      True),
        ("hn",         "HN",         False),
        ("ashby",      "Ashby",      False),
        ("workable",   "Workable",   False),
        ("himalayas",  "Himalayas",  False),
    ]:
        if sources.get(key, {}).get("enabled", default):
            enabled_names.append(label)

    stats = count_jobs()

    logger.info("Job Agent")
    logger.info(f"Provider: {provider} ({model})")
    logger.info("Sources: %s", "  ".join(enabled_names) if enabled_names else "none")
    logger.info(f"DB: {stats['total']} jobs total, {stats['scored']} scored")


def _run_scrapers(config: dict, profile: str | None = None) -> dict:
    """Run all enabled scrapers. Returns combined stats."""
    sources = config.get("sources", {})

    gh_enabled  = sources.get("greenhouse", {}).get("enabled", True)
    lv_enabled  = sources.get("lever",      {}).get("enabled", True)
    hn_enabled  = sources.get("hn",         {}).get("enabled", False)
    ash_enabled = sources.get("ashby",      {}).get("enabled", False)
    wl_enabled  = sources.get("workable",   {}).get("enabled", False)
    him_enabled = sources.get("himalayas",  {}).get("enabled", False)

    if not any([gh_enabled, lv_enabled, hn_enabled, ash_enabled, wl_enabled, him_enabled]):
        logger.error("No sources are enabled in config.yaml. Nothing to scrape.")
        raise ValueError("No sources are enabled in config.yaml. Nothing to scrape.")

    total_new = 0
    total_scraped = 0
    total_filtered = 0
    total_saved = 0

    # Resolve slugs once for all company-list ATS sources
    if gh_enabled or lv_enabled or ash_enabled or wl_enabled:
        slug_map = get_or_discover_slugs(config, profile=profile)

    if gh_enabled:
        result = scrape_greenhouse(config, slugs=slug_map["greenhouse"], profile=profile)
        total_new += result.get("new_jobs_saved", 0)
        total_scraped += result.get("jobs_scraped", 0)
        total_filtered += result.get("jobs_filtered", 0)
        total_saved += result.get("new_jobs_saved", 0)

    if lv_enabled:
        result = scrape_lever(config, slugs=slug_map["lever"], profile=profile)
        total_new += result.get("new_jobs_saved", 0)
        total_scraped += result.get("jobs_scraped", 0)
        total_filtered += result.get("jobs_filtered", 0)
        total_saved += result.get("new_jobs_saved", 0)

    if ash_enabled:
        result = scrape_ashby(config, slugs=slug_map["ashby"], profile=profile)
        total_new += result.get("new_jobs_saved", 0)
        total_scraped += result.get("jobs_scraped", 0)
        total_filtered += result.get("jobs_filtered", 0)
        total_saved += result.get("new_jobs_saved", 0)

    if wl_enabled:
        result = scrape_workable(config, slugs=slug_map["workable"], profile=profile)
        total_new += result.get("new_jobs_saved", 0)
        total_scraped += result.get("jobs_scraped", 0)
        total_filtered += result.get("jobs_filtered", 0)
        total_saved += result.get("new_jobs_saved", 0)

    if hn_enabled:
        result = scrape_hn(config, profile=profile)
        total_new += result.get("new_jobs_saved", 0)
        total_scraped += result.get("jobs_scraped", 0)
        total_filtered += result.get("jobs_filtered", 0)
        total_saved += result.get("new_jobs_saved", 0)

    if him_enabled:
        result = scrape_himalayas(config, profile=profile)
        total_new += result.get("new_jobs_saved", 0)
        total_scraped += result.get("jobs_scraped", 0)
        total_filtered += result.get("jobs_filtered", 0)
        total_saved += result.get("new_jobs_saved", 0)

    return {
        "total_new": total_new,
        "jobs_scraped": total_scraped,
        "jobs_filtered": total_filtered,
        "jobs_saved": total_saved,
    }


def main() -> None:
    args = parse_args()

    # Profile: set active profile before any DB or config access
    if args.profile:
        profiles_dir = Path(__file__).parent / "profiles"
        if not (profiles_dir / args.profile).exists():
            logger.error("Profile '%s' not found.", args.profile)
            logger.error("Run the dashboard to create it: streamlit run dashboard.py")
            logger.error("Expected location: profiles/%s/config.yaml", args.profile)
            sys.exit(1)
        set_active_profile(args.profile)

    config = load_config(profile=args.profile)
    config["_active_profile"] = args.profile
    init_db(profile=args.profile)

    # Configure logging after profile is set
    configure_logging(profile=args.profile or "default", debug=args.debug)

    # Apply --min-score override before any display or scoring
    if args.min_score is not None:
        config["scoring"]["min_display_score"] = args.min_score

    provider = config["llm"]["provider"]

    # --show: just print current top jobs and exit
    if args.show:
        _print_banner(config)
        min_score = config["scoring"].get("min_display_score", 60)
        results   = get_top_jobs(min_score)
        if not results:
            logger.info("No scored jobs above %d in DB.", min_score)
        else:
            print_results(results, config)
        return

    # Validate API key before starting any work if scoring will run
    will_score = not args.scrape_only
    if will_score:
        _check_api_key(provider, args.profile)

    _print_banner(config)

    run_id = start_run(profile=args.profile, source="cli")
    scrape_stats = {"total_new": 0, "jobs_scraped": 0, "jobs_filtered": 0, "jobs_saved": 0}
    score_results = []
    avg_fit = 0.0
    final_status = "complete"
    final_errors: list[str] = []

    # --rescore: clear scores table before scoring
    if args.rescore:
        if not args.score_only and not args.yes:
            try:
                confirm = input("--rescore will delete all existing scores. Continue? [y/n]: ").strip().lower()
            except EOFError:
                confirm = "y"
            if confirm != "y":
                logger.info("Aborted rescore.")
                return
        logger.info("[rescore] Clearing scores table and resetting score_attempts...")
        rescore_reset()

    try:
        # ── Scrape ──────────────────────────────────────────────────────────
        if not args.score_only:
            scrape_stats = _run_scrapers(config, profile=args.profile)

        # ── Score ────────────────────────────────────────────────────────────
        if not args.scrape_only:
            score_results = score_all_jobs(config, yes=args.yes, profile=args.profile)
            if score_results:
                print_results(score_results, config)

        # ── Final summary ──────────────────────────────────────────────────
        final_stats = count_jobs(profile=args.profile)
        scored_this_run = [r for r in score_results if r.get("fit_score", 0) > 0]

        logger.info("Run complete")
        logger.info("DB: %d jobs total, %d scored", final_stats['total'], final_stats['scored'])
        if not args.score_only:
            logger.info("New jobs this run: %d", scrape_stats.get('total_new', 0))
        if scored_this_run:
            avg_fit = sum(r["fit_score"] for r in scored_this_run) / len(scored_this_run)
            avg_ats = sum(r["ats_score"] for r in scored_this_run) / len(scored_this_run)
            logger.info("Scored this run: %d", len(scored_this_run))
            logger.info("Avg fit score: %.0f", avg_fit)
            logger.info("Avg ATS score: %.0f", avg_ats)
        logger.info("%s", "─" * 44)

    except KeyboardInterrupt:
        final_status = "failed"
        final_errors = ["Interrupted by user"]
        logger.warning("Interrupted by user")
        stats = count_jobs(profile=args.profile)
        logger.info("DB state: %d jobs total, %d scored", stats['total'], stats['scored'])
        raise

    except Exception as e:
        final_status = "failed"
        final_errors = [str(e)]
        logger.error("Pipeline crashed: %s", e)
        stats = count_jobs(profile=args.profile)
        logger.info("DB state: %d jobs total, %d scored", stats['total'], stats['scored'])
        raise

    finally:
        if run_id:
            finish_run(
                run_id,
                jobs_scraped=scrape_stats.get('jobs_scraped', 0),
                jobs_filtered=scrape_stats.get('jobs_filtered', 0),
                jobs_saved=scrape_stats.get('jobs_saved', 0),
                jobs_scored=len([r for r in score_results if r.get('fit_score', 0) > 0]),
                avg_fit_score=avg_fit if scored_this_run else 0.0,
                errors=final_errors,
                status=final_status,
                profile=args.profile,
            )


if __name__ == "__main__":
    main()
