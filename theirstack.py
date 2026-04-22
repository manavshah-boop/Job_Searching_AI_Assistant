"""
theirstack.py — Slug resolution utilities for Greenhouse, Lever, Ashby, and Workable.

Resolves ATS board slugs from company name/domain, and caches results in the DB.
get_or_discover_slugs() merges the priority list from config.yaml with the DB cache —
no external discovery calls are made.
"""

import re
import time
from typing import Optional

import httpx
from loguru import logger

from db import load_discovered_slugs


def _generate_slug_candidates(name: str, domain: str) -> list:
    """Generates slug candidates from company name and domain."""
    candidates = []

    # From domain: strip TLD (varsitytutors.com -> varsitytutors)
    if domain:
        base = domain.split(".")[0].lower()
        candidates.append(base)

    # From name: various normalizations
    if name:
        clean = re.sub(r"[^\w\s-]", "", name.lower()).strip()
        candidates.append(clean.replace(" ", ""))   # "openai"
        candidates.append(clean.replace(" ", "-"))  # "open-ai"
        candidates.append(clean.replace(" ", "_"))  # "open_ai"

        # Strip common corporate suffixes
        for suffix in [" inc", " llc", " ltd", " corp", " co", " company"]:
            stripped = clean.replace(suffix, "").strip()
            if stripped != clean:
                candidates.append(stripped.replace(" ", ""))
                candidates.append(stripped.replace(" ", "-"))

    # Deduplicate while preserving order
    seen: set = set()
    return [c for c in candidates if c and not (c in seen or seen.add(c))]


def resolve_greenhouse_slug(company: dict) -> Optional[str]:
    """
    Attempts to find a valid Greenhouse board slug for a company.
    Tries several slug variations derived from company name and domain.
    Returns the first valid slug, or None if none work.
    """
    name = company.get("name", "")
    domain = company.get("domain", "")
    candidates = _generate_slug_candidates(name, domain)

    for slug in candidates:
        try:
            resp = httpx.get(
                f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs",
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("jobs") is not None:
                    return slug
        except Exception:
            continue
        time.sleep(0.2)  # be polite

    return None


def resolve_lever_slug(company: dict) -> Optional[str]:
    """
    Attempts to find a valid Lever posting slug for a company.
    Lever slugs follow api.lever.co/v0/postings/{slug}?mode=json
    """
    name = company.get("name", "")
    domain = company.get("domain", "")
    candidates = _generate_slug_candidates(name, domain)

    for slug in candidates:
        try:
            resp = httpx.get(
                f"https://api.lever.co/v0/postings/{slug}?mode=json",
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                # Lever returns an array — valid if it's a list (even empty)
                if isinstance(data, list):
                    return slug
        except Exception:
            continue
        time.sleep(0.2)  # be polite

    return None


def resolve_ashby_slug(company: dict) -> Optional[str]:
    """
    Attempts to find a valid Ashby board slug for a company.
    Returns the first valid slug, or None if none work.
    """
    name = company.get("name", "")
    domain = company.get("domain", "")
    candidates = _generate_slug_candidates(name, domain)

    for slug in candidates:
        try:
            resp = httpx.get(
                f"https://api.ashbyhq.com/posting-api/job-board/{slug}",
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("jobs") is not None or data.get("jobPostings") is not None:
                    return slug
        except Exception:
            continue
        time.sleep(0.2)  # be polite

    return None


def resolve_workable_slug(company: dict) -> Optional[str]:
    """
    Attempts to find a valid Workable posting slug for a company.
    """
    name = company.get("name", "")
    domain = company.get("domain", "")
    candidates = _generate_slug_candidates(name, domain)

    for slug in candidates:
        try:
            resp = httpx.get(
                f"https://apply.workable.com/api/v1/widget/accounts/{slug}",
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("jobs") is not None:
                    return slug
        except Exception:
            continue
        time.sleep(0.2)  # be polite

    return None


def get_or_discover_slugs(config: dict, profile: Optional[str] = None) -> dict:
    """
    Returns slug lists per ATS:
      {
          "greenhouse": [...slugs...],
          "lever":      [...slugs...],
          "ashby":      [...slugs...],
          "workable":   [...slugs...],
      }

    Each list is: priority (from config) + DB cache.
    No external discovery calls are made.
    """
    sources = config.get("sources", {})

    # Priority lists from config
    gh_priority    = sources.get("greenhouse", {}).get("companies", [])
    lv_priority    = sources.get("lever",      {}).get("companies", [])
    ashby_priority = sources.get("ashby",      {}).get("companies", [])
    wl_priority    = sources.get("workable",   {}).get("companies", [])

    # Load cached slugs from DB
    gh_cached    = load_discovered_slugs(ats="greenhouse", profile=profile)
    lv_cached    = load_discovered_slugs(ats="lever",      profile=profile)
    ashby_cached = load_discovered_slugs(ats="ashby",      profile=profile)
    wl_cached    = load_discovered_slugs(ats="workable",   profile=profile)

    gh_all    = gh_priority    + [s for s in gh_cached    if s not in gh_priority]
    lv_all    = lv_priority    + [s for s in lv_cached    if s not in lv_priority]
    ashby_all = ashby_priority + [s for s in ashby_cached if s not in ashby_priority]
    wl_all    = wl_priority    + [s for s in wl_cached    if s not in wl_priority]

    logger.info(
        "[greenhouse] %d companies (%d priority, %d cached)",
        len(gh_all), len(gh_priority), len(gh_cached),
    )
    logger.info(
        "[lever] %d companies (%d priority, %d cached)",
        len(lv_all), len(lv_priority), len(lv_cached),
    )
    logger.info(
        "[ashby] %d companies (%d priority, %d cached)",
        len(ashby_all), len(ashby_priority), len(ashby_cached),
    )
    logger.info(
        "[workable] %d companies (%d priority, %d cached)",
        len(wl_all), len(wl_priority), len(wl_cached),
    )

    return {
        "greenhouse": gh_all,
        "lever":      lv_all,
        "ashby":      ashby_all,
        "workable":   wl_all,
    }
