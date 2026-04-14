"""
theirstack.py — Dynamic company discovery via TheirStack API.

Discovers companies that use Greenhouse/Lever/Ashby, resolves their
Greenhouse board slugs, and caches results in the DB so slug resolution
(the slow part) only runs once per company.

Only runs when THEIRSTACK_API_KEY is set — the scraper works fine without it
using the priority list from config.yaml.
"""

import os
import re
import time
from typing import Optional

import httpx
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from db import load_discovered_slugs, save_discovered_slug


def fetch_companies(config: dict) -> list:
    """
    Fetches companies from TheirStack that use Greenhouse/Lever/Ashby,
    filtered to US-based, tech-adjacent, and reasonable size.
    Returns list of {name, domain, country_code, employee_count, industry}
    """
    api_key = os.environ.get("THEIRSTACK_API_KEY")
    if not api_key:
        logger.debug("No THEIRSTACK_API_KEY found, skipping discovery")
        return []

    payload = {
        "company_technology_slug_or": ["greenhouse", "lever", "ashby"],
        "company_country_code_or": ["US"],
        "min_employee_count": 10,
        "max_employee_count": 5000,
        "industry_or": [
            "Technology, Information and Internet",
            "Software Development",
            "IT Services and IT Consulting",
            "Computer and Network Security",
            "Artificial Intelligence",
            "Financial Services",
            "Internet Marketplace Platforms",
        ],
        "order_by": [{"desc": True, "field": "num_jobs_last_30_days"}],
        "limit": 25,  # free plan cap; upgrade for more
        "page": 0,
    }

    try:
        resp = httpx.post(
            "https://api.theirstack.com/v1/companies/search",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
    except Exception as e:
        logger.warning(f"TheirStack request failed: {e}")
        return []

    if resp.status_code != 200:
        logger.warning(f"TheirStack API error: {resp.status_code} — {resp.text[:200]}")
        return []

    return resp.json().get("data", [])


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


def get_or_discover_slugs(config: dict, profile: Optional[str] = None) -> dict:
    """
    Returns slug lists per ATS:
      {
          "greenhouse": [...slugs...],
          "lever":      [...slugs...],
      }

    Each list is: priority (from config) + DB cache + newly discovered from TheirStack.
    Newly discovered slugs are saved to DB so resolution only runs once per company.
    """
    # Priority lists from config
    gh_priority = (
        config.get("sources", {}).get("greenhouse", {}).get("companies", [])
    )
    lv_priority = (
        config.get("sources", {}).get("lever", {}).get("companies", [])
    )

    # Load cached slugs from DB
    gh_cached = load_discovered_slugs(ats="greenhouse", profile=profile)
    lv_cached = load_discovered_slugs(ats="lever", profile=profile)

    # Discover new slugs via TheirStack if key is set
    gh_new: list = []
    lv_new: list = []
    if os.environ.get("THEIRSTACK_API_KEY"):
        companies = fetch_companies(config)
        logger.info("[theirstack] %d companies returned, resolving slugs...", len(companies))
        for company in companies:
            name = company.get("name", "")

            gh_slug = resolve_greenhouse_slug(company)
            if gh_slug and gh_slug not in gh_priority and gh_slug not in gh_cached:
                gh_new.append(gh_slug)
                save_discovered_slug(gh_slug, name, ats="greenhouse", profile=profile)
                logger.info("  + [GH] %s -> %s", name, gh_slug)

            lv_slug = resolve_lever_slug(company)
            if lv_slug and lv_slug not in lv_priority and lv_slug not in lv_cached:
                lv_new.append(lv_slug)
                save_discovered_slug(lv_slug, name, ats="lever", profile=profile)
                logger.info("  + [LV] %s -> %s", name, lv_slug)

            if not gh_slug and not lv_slug:
                logger.warning("  x %s -> no slug found (GH or LV)", name)

    gh_all = gh_priority + [s for s in gh_cached if s not in gh_priority] + gh_new
    lv_all = lv_priority + [s for s in lv_cached if s not in lv_priority] + lv_new

    logger.info(
        "[greenhouse] %d companies (%d priority, %d cached, %d new)",
        len(gh_all), len(gh_priority), len(gh_cached), len(gh_new)
    )
    logger.info(
        "[lever] %d companies (%d priority, %d cached, %d new)",
        len(lv_all), len(lv_priority), len(lv_cached), len(lv_new)
    )
    return {"greenhouse": gh_all, "lever": lv_all}
