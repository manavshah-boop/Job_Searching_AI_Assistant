"""
force_score_job.py — Step 2 diagnostic: score a single job with LLM forced, bypassing routing.

Usage:
  python force_score_job.py --job-id <job_id> [--profile default]
  python force_score_job.py --list                     # list recent jobs

This script loads the profile config, disables routing, and calls score_job() directly
on the specified job. Use it to get a baseline LLM score when routing would have skipped it.
"""

import argparse
import sys

from db import Job, init_db, load_config, get_all_jobs, get_job_with_score
from scorer import get_llm_client, get_instructor_client, score_job
from candidate_profile import build_structured_profile, print_profile_summary
from logging_config import configure_logging

configure_logging(profile="force_score", debug=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Force LLM score on a single job")
    parser.add_argument("--job-id", metavar="ID", help="Job ID to score")
    parser.add_argument("--profile", default="default", help="Profile name (default: default)")
    parser.add_argument("--list", action="store_true", help="List recently added jobs")
    args = parser.parse_args()

    config = load_config(profile=args.profile)
    init_db(profile=args.profile)

    if args.list:
        jobs = get_all_jobs(profile=args.profile)
        for j in jobs[:30]:
            print(f"  {j.id:<50}  {j.company:<20}  {j.title}")
        return

    if not args.job_id:
        parser.error("--job-id is required (use --list to find job IDs)")

    record = get_job_with_score(args.job_id, profile=args.profile)
    if record is None:
        print(f"Job not found: {args.job_id}", file=sys.stderr)
        sys.exit(1)

    job = Job(
        id=record["id"],
        title=record["title"],
        company=record["company"],
        location=record.get("location", ""),
        url=record.get("url", ""),
        raw_text=record.get("raw_text", ""),
        source=record.get("source", ""),
        scrape_qualified=record.get("scrape_qualified", 1),
        scrape_filter_reason=record.get("scrape_filter_reason", ""),
    )

    print(f"\n=== JOB ===")
    print(f"Title:   {job.title}")
    print(f"Company: {job.company}")
    print(f"ID:      {job.id}")
    print(f"Description length: {len(job.raw_text or '')} chars")
    print(f"First 300 chars: {(job.raw_text or '')[:300]!r}")

    print(f"\n=== RESUME ===")
    resume = config["profile"].get("resume") or ""
    print(f"Resume length: {len(resume)} chars")
    print(f"First 500 chars: {resume[:500]!r}")

    # Disable routing explicitly
    config.setdefault("routing", {})["enabled"] = False
    print(f"\n=== ROUTING DISABLED — FORCING LLM ===")

    llm_call = get_llm_client(config)
    instructor_client = None
    try:
        instructor_client, _, _ = get_instructor_client(config)
    except Exception:
        pass

    structured_profile = build_structured_profile(config, llm_call, instructor_client,
                                                   model=config["llm"]["model"][config["llm"]["provider"]],
                                                   temperature=config["llm"].get("temperature", 0))
    print_profile_summary(structured_profile)

    print(f"\n=== SCORING ===")
    result = score_job(job, config, llm_call, structured_profile, instructor_client)

    print(f"\n=== RESULTS ===")
    print(f"Fit score:  {result['fit_score']}/100")
    print(f"ATS score:  {result['ats_score']}/100")
    dims = result.get("dimension_scores", {})
    for k, v in dims.items():
        print(f"  {k:<20} {v}/10")
    print(f"\nOne-liner: {result['one_liner']}")
    print(f"Reasons:   {result['reasons']}")
    print(f"Flags:     {result['flags']}")
    print(f"Skill misses: {result['skill_misses']}")


if __name__ == "__main__":
    main()
