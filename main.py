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

from db import count_jobs, init_db, load_config, rescore_reset, set_active_profile
from db import get_top_jobs
from scraper import scrape_greenhouse, scrape_hn, scrape_lever
from scorer import print_results, score_all_jobs
from theirstack import get_or_discover_slugs


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
    return parser.parse_args()


def _check_api_key(provider: str) -> None:
    """Exit with a clear message if the required API key is missing."""
    env_var = _PROVIDER_ENV_VARS.get(provider)
    if env_var and not os.environ.get(env_var):
        print(f"\n[ERROR] Scoring requires {env_var} but it is not set.")
        print(f"  Add it to your .env file:  {env_var}=your_key_here")
        print(f"  Or export it:              export {env_var}=your_key_here\n")
        sys.exit(1)


def _print_banner(config: dict) -> None:
    """Print startup banner showing provider, enabled sources, and DB stats."""
    provider = config["llm"]["provider"]
    model    = config["llm"]["model"].get(provider, "unknown")
    sources  = config.get("sources", {})

    gh_enabled  = sources.get("greenhouse", {}).get("enabled", True)
    lv_enabled  = sources.get("lever",      {}).get("enabled", True)
    hn_enabled  = sources.get("hn",         {}).get("enabled", True)

    gh_mark = "\u2713" if gh_enabled else "\u2717"
    lv_mark = "\u2713" if lv_enabled else "\u2717"
    hn_mark = "\u2713" if hn_enabled else "\u2717"

    stats = count_jobs()

    width = 44
    print("\n" + "\u2550" * width)
    print("  Job Agent")
    print(f"  Provider: {provider} ({model})")
    print(f"  Sources: Greenhouse {gh_mark}  Lever {lv_mark}  HN {hn_mark}")
    print(f"  DB: {stats['total']} jobs total, {stats['scored']} scored")
    print("\u2550" * width + "\n")


def _run_scrapers(config: dict, profile: str | None = None) -> dict:
    """Run all enabled scrapers. Returns combined stats."""
    sources = config.get("sources", {})

    gh_enabled = sources.get("greenhouse", {}).get("enabled", True)
    lv_enabled = sources.get("lever",      {}).get("enabled", True)
    hn_enabled = sources.get("hn",         {}).get("enabled", True)

    if not any([gh_enabled, lv_enabled, hn_enabled]):
        print("[ERROR] No sources are enabled in config.yaml. Nothing to scrape.")
        sys.exit(1)

    total_new = 0

    # Resolve slugs once for Greenhouse + Lever (includes TheirStack discovery)
    if gh_enabled or lv_enabled:
        slug_map = get_or_discover_slugs(config, profile=profile)

    if gh_enabled:
        result = scrape_greenhouse(config, slugs=slug_map["greenhouse"], profile=profile)
        total_new += result.get("new_jobs_saved", 0)

    if lv_enabled:
        result = scrape_lever(config, slugs=slug_map["lever"], profile=profile)
        total_new += result.get("new_jobs_saved", 0)

    if hn_enabled:
        result = scrape_hn(config, profile=profile)
        total_new += result.get("new_jobs_saved", 0)

    return {"total_new": total_new}


def main() -> None:
    args = parse_args()

    # Profile: set active profile before any DB or config access
    if args.profile:
        profiles_dir = Path(__file__).parent / "profiles"
        if not (profiles_dir / args.profile).exists():
            print(f"\n[ERROR] Profile '{args.profile}' not found.")
            print(f"  Run the dashboard to create it:  streamlit run dashboard.py")
            print(f"  Expected location: profiles/{args.profile}/config.yaml\n")
            sys.exit(1)
        set_active_profile(args.profile)

    config = load_config(profile=args.profile)
    init_db(profile=args.profile)

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
            print(f"No scored jobs above {min_score} in DB.")
        else:
            print_results(results, config)
        return

    # Validate API key before starting any work if scoring will run
    will_score = not args.scrape_only
    if will_score:
        _check_api_key(provider)

    _print_banner(config)

    # --rescore: clear scores table before scoring
    if args.rescore:
        if not args.score_only and not args.yes:
            try:
                confirm = input("--rescore will delete all existing scores. Continue? [y/n]: ").strip().lower()
            except EOFError:
                confirm = "y"
            if confirm != "y":
                print("Aborted.")
                return
        print("[rescore] Clearing scores table and resetting score_attempts...")
        rescore_reset()

    try:
        scrape_stats = {"total_new": 0}
        score_results: list = []

        # ── Scrape ──────────────────────────────────────────────────────────
        if not args.score_only:
            scrape_stats = _run_scrapers(config, profile=args.profile)

        # ── Score ────────────────────────────────────────────────────────────
        if not args.scrape_only:
            score_results = score_all_jobs(config, yes=args.yes)
            if score_results:
                print_results(score_results, config)

        # ── Final summary ────────────────────────────────────────────────────
        final_stats = count_jobs()
        scored_this_run = [r for r in score_results if r.get("fit_score", 0) > 0]

        print("\n" + "─" * 44)
        print("  Run complete")
        print(f"  DB: {final_stats['total']} jobs total, {final_stats['scored']} scored")
        if not args.score_only:
            print(f"  New jobs this run: {scrape_stats['total_new']}")
        if scored_this_run:
            avg_fit = sum(r["fit_score"] for r in scored_this_run) / len(scored_this_run)
            avg_ats = sum(r["ats_score"] for r in scored_this_run) / len(scored_this_run)
            print(f"  Scored this run:   {len(scored_this_run)}")
            print(f"  Avg fit score:     {avg_fit:.0f}")
            print(f"  Avg ATS score:     {avg_ats:.0f}")
        print("─" * 44 + "\n")

    except KeyboardInterrupt:
        print("\n\n[Interrupted]")
        stats = count_jobs()
        print(f"DB state: {stats['total']} jobs total, {stats['scored']} scored")
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Pipeline crashed: {e}")
        stats = count_jobs()
        print(f"DB state: {stats['total']} jobs total, {stats['scored']} scored")
        raise


if __name__ == "__main__":
    main()
