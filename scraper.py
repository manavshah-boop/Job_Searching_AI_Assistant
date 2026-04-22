"""
scraper.py — Scrapes job listings from Greenhouse and HN Who's Hiring.

Greenhouse: Public API, structured data
HN: Algolia API, free-form text in comments
"""

import html as html_module
import httpx
import re
import sys

# Ensure Unicode output works on Windows terminals (cp1252 -> utf-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
from loguru import logger
from db import Job, clear_discovered_slugs, init_db, insert_job, load_config, make_id
from text_utils import extract_job_context
from theirstack import get_or_discover_slugs


GREENHOUSE_API_BASE = "https://boards-api.greenhouse.io/v1/boards"
HN_ALGOLIA_URL = "https://hn.algolia.com/api/v1/search"


def strip_html(html_text: str) -> str:
    """Remove HTML tags from text. Handles HTML-escaped content (e.g. Greenhouse API)."""
    if not html_text:
        return ""
    # Greenhouse returns HTML-escaped content (&lt;div&gt; etc.).
    # Unescape once so BeautifulSoup sees real tags, not escaped text.
    unescaped = html_module.unescape(html_text)
    soup = BeautifulSoup(unescaped, 'html.parser')
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    # Get text
    text = soup.get_text()
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    # Second unescape pass handles double-encoded entities (e.g. &amp;lt; -> &lt; -> <)
    return html_module.unescape(text)



# Words that are too generic to be meaningful title-match signals.
# They appear in preferred titles (e.g. "Software Development Engineer")
# but would false-match unrelated roles like "Sales Development Representative".
_TITLE_MATCH_STOPWORDS = {
    "software", "development", "and", "of", "the", "in", "for", "with",
}


def title_matches(job_title: str, preferred_titles: List[str]) -> bool:
    """
    Loose keyword-based matching: if ANY *discriminating* word from
    preferred_titles appears in the job title, it passes.

    Discriminating words are those not in _TITLE_MATCH_STOPWORDS — e.g.
    "engineer", "developer", "backend", "ai", "sde", "swe" qualify;
    "development" and "software" do not (too common, cause false matches).
    """
    job_title_lower = job_title.lower()

    preferred_words = set()
    for title in preferred_titles:
        for word in title.lower().split():
            if word not in _TITLE_MATCH_STOPWORDS:
                preferred_words.add(word)

    for word in preferred_words:
        if word in job_title_lower:
            return True

    return False


def contains_hard_no_keyword(text: str, hard_no_keywords: List[str]) -> bool:
    """
    Check if text contains any hard no keywords using word-boundary matching.
    Word boundaries prevent false positives like 'intern' matching 'internal'.
    """
    text_lower = text.lower()
    for keyword in hard_no_keywords:
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        if re.search(pattern, text_lower):
            return True
    return False


# ---------------------------------------------------------------------------
# Unified pre-filter
# ---------------------------------------------------------------------------

# All 50 US state abbreviations + DC + territories
_US_STATE_ABBREVS = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",
}

# US state full names (lower-cased for matching)
_US_STATE_NAMES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
    "rhode island", "south carolina", "south dakota", "tennessee", "texas",
    "utah", "vermont", "virginia", "washington", "west virginia",
    "wisconsin", "wyoming", "district of columbia",
}

# Common US city shorthand that appears alone without a state
_US_CITY_SIGNALS = {
    "nyc", "sf", "new york", "san francisco", "seattle", "chicago",
    "boston", "austin", "denver", "atlanta", "los angeles", "la",
    "portland", "miami", "dallas", "houston", "phoenix", "philadelphia",
    "san jose", "san diego", "pittsburgh", "minneapolis", "detroit",
    "nashville", "charlotte", "raleigh", "salt lake city",
}

# Location placeholder strings that mean "unspecified" — let through
_LOCATION_UNKNOWNS = {"n/a", "na", "location", "tbd", "unknown", ""}

# Regex to detect ", XX" or "| XX" US state abbreviation patterns (word boundary)
_US_STATE_ABBREV_RE = re.compile(
    r'(?:,\s*|;\s*|\|\s*|\b)(' + '|'.join(_US_STATE_ABBREVS) + r')\b'
)

# Explicit non-US signals — used in both Greenhouse and HN filters.
# Checked before "remote" so "Remote, Denmark" is correctly rejected.
NON_US_SIGNALS = [
    "denmark", "germany", "france", "uk", "united kingdom", "canada",
    "australia", "india", "netherlands", "sweden", "norway", "finland",
    "spain", "italy", "portugal", "brazil", "singapore", "japan",
    "aarhus", "copenhagen", "berlin", "london", "paris", "toronto",
    "sydney", "amsterdam", "stockholm", "dublin", "zurich",
]

# Explicit non-US signals for HN free-text scanning (superset of NON_US_SIGNALS)
_NON_US_GEO_SIGNALS = [
    "canada", "uk", "united kingdom", "london", "germany", "berlin",
    "france", "paris", "australia", "sydney", "melbourne", "india",
    "bangalore", "bengaluru", "amsterdam", "netherlands", "singapore",
    "brazil", "toronto", "ontario", "british columbia", "europe",
    "latam", "apac", "emea", "dublin", "ireland", "israel", "japan",
    "tokyo", "korea", "seoul", "serbia", "sweden", "norway", "denmark",
    "finland", "new zealand", "auckland", "switzerland", "zurich",
    "aarhus", "copenhagen", "spain", "italy", "portugal", "stockholm",
]

# Hiring signal words used in the HN intent filter
_HIRING_SIGNALS = [
    "hiring", "looking for", "seeking", "we need", "join us",
    "open role", "we're looking", "we are looking",
]


def _is_us_location(location: str) -> bool:
    """
    Return True if a Greenhouse location string is US-based or unspecified.

    Handles formats like:
      "San Francisco, CA"
      "San Francisco, CA | New York City, NY | Seattle, WA"
      "Remote"
      "N/A"  (unspecified — let through)
      "LOCATION"  (template placeholder — let through)
    """
    loc = location.strip()
    loc_lower = loc.lower()

    # Empty or known placeholder -> unspecified, let through
    if loc_lower in _LOCATION_UNKNOWNS:
        return True

    # Explicit non-US signal — reject immediately (even if "remote" is also present)
    if any(signal in loc_lower for signal in NON_US_SIGNALS):
        return False

    # Explicit "remote" with no non-US signal -> assume US-eligible
    if "remote" in loc_lower:
        return True

    # Explicit US country names
    if "united states" in loc_lower or " usa" in loc_lower:
        return True

    # "US" as a whole word (avoid matching "brussels", "houston" substrings)
    if re.search(r'\bUS\b', loc):
        return True

    # US state abbreviation: ", CA" / "| NY" / "; WA" etc.
    if _US_STATE_ABBREV_RE.search(loc):
        return True

    # US state full name anywhere in string
    for state_name in _US_STATE_NAMES:
        if state_name in loc_lower:
            return True

    # Common US city shorthand
    for city in _US_CITY_SIGNALS:
        # word-boundary match to avoid "portland" matching "Portland, OR" vs "Portland, UK"
        # (Portland is only in Oregon/Maine so it's fine, but we're careful anyway)
        if re.search(r'\b' + re.escape(city) + r'\b', loc_lower):
            return True

    return False


# Degree requirement patterns — match when a master's/PhD is *required*, not just preferred
DEGREE_REQUIRED_PATTERNS = [
    re.compile(r"master'?s degree required", re.IGNORECASE),
    re.compile(r"ms or phd required", re.IGNORECASE),
    re.compile(r"phd required", re.IGNORECASE),
    re.compile(r"must have an? (ms|m\.s\.|master'?s|phd|ph\.d)", re.IGNORECASE),
    re.compile(r"requires? an? (ms|m\.s\.|master'?s|phd|ph\.d)", re.IGNORECASE),
    re.compile(r"minimum.{0,20}master'?s degree", re.IGNORECASE),
    re.compile(r"(ms|phd|master'?s).{0,20}(required|mandatory|must)", re.IGNORECASE),
]


def requires_advanced_degree(text: str) -> bool:
    """Return True if the text indicates a master's degree or higher is required."""
    for pattern in DEGREE_REQUIRED_PATTERNS:
        if pattern.search(text):
            return True
    return False


# Regex patterns for extracting YOE numbers from text
_YOE_PATTERNS = [
    re.compile(r'(\d+)\+\s*years?', re.IGNORECASE),
    re.compile(r'(\d+)\s*-\s*(\d+)\s*years?', re.IGNORECASE),
    re.compile(r'minimum\s+(\d+)\s*years?', re.IGNORECASE),
    re.compile(r'at\s+least\s+(\d+)\s*years?', re.IGNORECASE),
    re.compile(r'(\d+)\s*years?\s+of\s+experience', re.IGNORECASE),
]


def _extract_yoe_numbers(text: str) -> List[int]:
    """Return YOE numbers found in text (upper bound of each range, no overlapping matches)."""
    all_matches: List[tuple] = []  # (start, end, value)
    for pattern in _YOE_PATTERNS:
        for match in pattern.finditer(text):
            groups = [int(g) for g in match.groups() if g is not None]
            if groups:
                all_matches.append((match.start(), match.end(), max(groups)))

    # Sort by start position and drop matches that overlap a previously kept match
    all_matches.sort(key=lambda m: m[0])
    kept: List[tuple] = []
    for start, end, value in all_matches:
        if any(start < ke and end > ks for ks, ke, _ in kept):
            continue  # overlaps a kept match
        kept.append((start, end, value))

    return [v for _, _, v in kept]


def _configured_max_job_age_days(config: Dict[str, Any]) -> int:
    """Return the configured max posting age in days, defaulting to 30."""
    try:
        return int(config.get("preferences", {}).get("filters", {}).get("max_job_age_days", 30))
    except (TypeError, ValueError):
        return 30


def _preferred_titles(config: Dict[str, Any]) -> List[str]:
    profile_info = config.get("profile", {})
    preferences = config.get("preferences", {})
    return profile_info.get("titles") or preferences.get("titles", [])


def _parse_posted_at(value: Any) -> Optional[datetime]:
    """Parse ISO strings, epoch seconds, or epoch milliseconds into UTC datetimes."""
    if value in (None, ""):
        return None

    try:
        if isinstance(value, (int, float)):
            timestamp = value / 1000 if value > 10_000_000_000 else value
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)

        text = str(value).strip()
        if not text:
            return None
        if re.fullmatch(r"\d+(\.\d+)?", text):
            timestamp = float(text)
            timestamp = timestamp / 1000 if timestamp > 10_000_000_000 else timestamp
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)

        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (OSError, ValueError, TypeError):
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_posted_at_older_than(value: Any, max_age: timedelta) -> bool:
    posted_at = _parse_posted_at(value)
    if posted_at is None:
        return False
    return datetime.now(tz=timezone.utc) - posted_at > max_age


def _build_raw_text(
    *,
    title: str,
    company: str,
    location: str,
    url: str,
    description: str,
    max_chars: int = 2000,
) -> str:
    raw_text = f"""Title: {title}
Company: {company}
Location: {location}
URL: {url}

Description:
{description}""".strip()
    return extract_job_context(raw_text, max_chars=max_chars)


def _insert_scraped_job(
    *,
    source: str,
    job_id: str,
    title: str,
    company: str,
    location: str,
    url: str,
    raw_text: str,
    profile: Optional[str],
    filter_reason: str = "",
) -> bool:
    return insert_job(Job(
        id=job_id,
        title=title,
        company=company,
        location=location,
        url=url,
        raw_text=raw_text,
        source=source,
        scrape_qualified=0 if filter_reason else 1,
        scrape_filter_reason=filter_reason,
    ), profile=profile)


def passes_filters(
    text: str,
    title: str,
    location: str,
    config: Dict[str, Any],
    source: str = "",
    debug: bool = False,
) -> tuple:
    """
    Unified pre-filter applied before insert_job().

    Returns (True, "") if the job should be kept, or (False, reason) if rejected.
    reason is a short consistent string e.g. "title_blocklist: Staff", "yoe_max: 8 > 5".
    Prints a one-line skip reason when debug=True.
    """
    filters = config.get('preferences', {}).get('filters', {})
    title_blocklist = [t.lower() for t in filters.get('title_blocklist', [])]
    min_yoe = filters.get('min_yoe', 0)
    max_yoe = filters.get('max_yoe', 99)

    preferred_titles = config.get('preferences', {}).get('titles', [])
    desired_skills = config.get('preferences', {}).get('desired_skills', [])

    title_lower = title.lower()
    text_lower = text.lower()

    # ------------------------------------------------------------------
    # 1. Title blocklist (structured sources check title only; HN scans full text)
    # ------------------------------------------------------------------
    check_text = (
        title_lower
        if source in ("greenhouse", "lever", "ashby", "workable", "himalayas")
        else title_lower + " " + text_lower
    )
    for blocked in title_blocklist:
        if re.search(r'\b' + re.escape(blocked) + r'\b', check_text):
            if debug:
                logger.debug(f"[SKIP] title_blocklist matched '{blocked}' | {title!r}")
            return False, f"title_blocklist: {blocked}"

    # ------------------------------------------------------------------
    # 2. YOE filter (both sources)
    # ------------------------------------------------------------------
    yoe_numbers = _extract_yoe_numbers(text)
    if yoe_numbers:
        yoe_max_found = max(yoe_numbers)
        yoe_min_found = min(yoe_numbers)
        if yoe_max_found > max_yoe:
            if debug:
                logger.debug(f"[SKIP] yoe_max {yoe_max_found} > {max_yoe} | {title!r}")
            return False, f"yoe_max: {yoe_max_found} > {max_yoe}"
        if yoe_min_found < min_yoe:
            if debug:
                logger.debug(f"[SKIP] yoe_min {yoe_min_found} < {min_yoe} | {title!r}")
            return False, f"yoe_min: {yoe_min_found} < {min_yoe}"

    # ------------------------------------------------------------------
    # 3. HN-only: hiring intent filter
    # ------------------------------------------------------------------
    if source == "hackernews":
        words = text_lower.split()
        word_count = len(words)
        found_intent = False
        for signal in _HIRING_SIGNALS:
            # find position of signal in the word list (approximate)
            try:
                idx = text_lower.index(signal)
            except ValueError:
                continue
            # find which word index that byte offset corresponds to
            signal_word_idx = len(text_lower[:idx].split())
            # check if a title/skill word appears within 50 words
            window_start = max(0, signal_word_idx - 50)
            window_end = min(word_count, signal_word_idx + 50)
            window = " ".join(words[window_start:window_end])
            for keyword in list(preferred_titles) + list(desired_skills):
                if keyword.lower() in window:
                    found_intent = True
                    break
            if found_intent:
                break
        if not found_intent:
            if debug:
                logger.debug(f"[SKIP] no hiring intent near title/skill keyword | {title!r}")
            return False, "hiring_intent_not_found"

    # ------------------------------------------------------------------
    # 4. Degree requirement filter (both sources)
    # ------------------------------------------------------------------
    if filters.get('require_degree_filter', False) and requires_advanced_degree(text):
        if debug:
            logger.debug(f"[SKIP] requires advanced degree | {title!r}")
        return False, "degree_required"

    # ------------------------------------------------------------------
    # 5. US location filter
    # ------------------------------------------------------------------
    if source in ("greenhouse", "lever", "ashby", "workable", "himalayas"):
        # Use structured location field + comprehensive US detector
        if not _is_us_location(location):
            if debug:
                logger.debug(f"[SKIP] non-US location '{location}' | {title!r}")
            return False, "non_us_location"

    elif source == "hackernews":
        # Unstructured free text — look for US signals and absence of non-US geo
        has_us_signal = (
            "united states" in text_lower
            or " usa" in text_lower
            or bool(re.search(r'\bUS\b', text))
            or any(state in text_lower for state in _US_STATE_NAMES)
            or bool(_US_STATE_ABBREV_RE.search(text))
            or any(re.search(r'\b' + re.escape(c) + r'\b', text_lower) for c in _US_CITY_SIGNALS)
        )
        has_remote = "remote" in text_lower
        has_non_us_geo = any(geo in text_lower for geo in _NON_US_GEO_SIGNALS)

        if has_remote and not has_non_us_geo:
            pass  # remote with no non-US signals -> assume US-eligible
        elif has_us_signal:
            pass  # explicit US mention
        else:
            if debug:
                logger.debug("[SKIP] no US/remote signal in HN post | %s", title)
            return False, "non_us_location"

    return True, ""


def scrape_greenhouse(config: Dict[str, Any], slugs: Optional[List[str]] = None, profile: Optional[str] = None) -> Dict[str, Any]:
    """
    Scrape all configured Greenhouse companies.
    Returns dict with: {companies_checked, new_jobs_saved, jobs_scraped, jobs_filtered, errors}
    """
    init_db(profile=profile)

    companies = slugs if slugs is not None else []
    preferences = config.get('preferences', {})
    preferred_titles = _preferred_titles(config)
    desired_skills = profile_info.get('desired_skills') or preferences.get('desired_skills', [])
    hard_no_keywords = preferences.get('hard_no_keywords', [])
    max_job_age_days = _configured_max_job_age_days(config)

    companies_checked = 0
    new_jobs_saved = 0
    jobs_scraped = 0
    jobs_filtered = 0
    errors = []

    logger.info("Scraping %s Greenhouse companies...", len(companies))
    logger.info("Looking for: %s", ", ".join(preferred_titles))
    logger.info("Hard no keywords: %s", ", ".join(hard_no_keywords))

    for company_slug in companies:
        companies_checked += 1
        try:
            url = f"{GREENHOUSE_API_BASE}/{company_slug}/jobs?content=true"
            
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url)
            
            if response.status_code != 200:
                errors.append(f"[!] {company_slug}: HTTP {response.status_code}")
                continue

            data = response.json()
            jobs = data.get('jobs', [])

            company_new_count = 0

            for job_posting in jobs:
                # Skip postings older than the configured age window.
                updated_at_str = job_posting.get('updated_at')
                if updated_at_str:
                    try:
                        # Parse ISO timestamp (e.g. "2016-01-14T10:55:28-05:00")
                        updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
                        # Use UTC for comparison to avoid timezone issues
                        oldest_allowed = datetime.now(updated_at.tzinfo) - timedelta(days=max_job_age_days)
                        if updated_at < oldest_allowed:
                            continue  # Skip old jobs
                    except (ValueError, TypeError):
                        # If we can't parse the date, continue processing (don't skip)
                        pass
                
                job_id = job_posting.get('id')
                title = job_posting.get('title', '')
                location = job_posting.get('location', {}).get('name', 'Unknown')
                url = job_posting.get('absolute_url', '')
                
                # Skip if title doesn't match
                if not title_matches(title, preferred_titles):
                    continue

                # Build raw_text for Claude
                # Greenhouse API returns the body as 'content', not 'description'
                description = job_posting.get('content', '') or job_posting.get('description', '')
                description_clean = strip_html(description)

                # Cap raw_text at ~3500 chars
                raw_text = f"""
Title: {title}
Company: {company_slug.upper()}
Location: {location}
URL: {url}

Description:
{description_clean}
""".strip()

                raw_text = extract_job_context(raw_text, max_chars=2000)

                # Check for hard no keywords in full posting
                full_text = title + " " + description_clean
                jobs_scraped += 1
                if contains_hard_no_keyword(full_text, hard_no_keywords):
                    jobs_filtered += 1
                    continue

                # Pre-filter (title blocklist, YOE, US location)
                job_id = make_id("greenhouse", str(job_posting["id"]))
                passed, filter_reason = passes_filters(full_text, title, location, config, source="greenhouse", debug=True)
                if not passed:
                    jobs_filtered += 1
                    insert_job(Job(
                        id=job_id, title=title, company=company_slug,
                        location=location, url=url, raw_text=raw_text,
                        source="greenhouse", scrape_qualified=0,
                        scrape_filter_reason=filter_reason,
                    ), profile=profile)
                    continue

                # Insert and track
                if insert_job(Job(
                    id=job_id, title=title, company=company_slug,
                    location=location, url=url, raw_text=raw_text,
                    source="greenhouse",
                ), profile=profile):
                    new_jobs_saved += 1
                    company_new_count += 1

            status = "[+]" if company_new_count > 0 else "[-]"
            logger.info("%s %s -> %d new jobs", status, company_slug, company_new_count)

        except Exception as e:
            errors.append(f"[!] {company_slug}: {str(e)}")
            logger.error("%s %s -> ERROR: %s", "[!]", company_slug, str(e))

    logger.info("Greenhouse companies checked: %d", companies_checked)
    logger.info("New jobs saved: %d", new_jobs_saved)
    if errors:
        logger.warning("Errors (%d):", len(errors))
        for error in errors:
            logger.warning(error)

    return {
        'companies_checked': companies_checked,
        'new_jobs_saved': new_jobs_saved,
        'jobs_scraped': jobs_scraped,
        'jobs_filtered': jobs_filtered,
        'errors': errors
    }


LEVER_API_BASE = "https://api.lever.co/v0/postings"


def scrape_lever(config: Dict[str, Any], slugs: Optional[List[str]] = None, profile: Optional[str] = None) -> Dict[str, Any]:
    """
    Scrape all configured Lever companies.
    Returns dict with: {companies_checked, new_jobs_saved, jobs_scraped, jobs_filtered, errors}
    """
    init_db(profile=profile)

    companies = slugs if slugs is not None else []
    preferences = config.get('preferences', {})

    preferred_titles = _preferred_titles(config)
    hard_no_keywords = preferences.get('hard_no_keywords', [])

    companies_checked = 0
    new_jobs_saved = 0
    jobs_scraped = 0
    jobs_filtered = 0
    errors = []

    logger.info("Scraping %s Lever companies...", len(companies))
    logger.info("Looking for: %s", ", ".join(preferred_titles))
    logger.info("Hard no keywords: %s", ", ".join(hard_no_keywords))

    max_age = timedelta(days=_configured_max_job_age_days(config))

    for slug in companies:
        companies_checked += 1
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{LEVER_API_BASE}/{slug}?mode=json")

            if response.status_code != 200:
                errors.append(f"[!] {slug}: HTTP {response.status_code}")
                continue

            postings = response.json()
            if not isinstance(postings, list):
                errors.append(f"[!] {slug}: unexpected response (not a list)")
                continue

            company_new_count = 0

            for posting in postings:
                # Date filter — createdAt is milliseconds
                if _is_posted_at_older_than(posting.get('createdAt'), max_age):
                    continue

                title = posting.get('text', '')
                categories = posting.get('categories', {})
                location = categories.get('location', 'Unknown')
                workplace_type = posting.get('workplaceType', '')
                url = posting.get('hostedUrl', '')

                # Skip if title doesn't match
                if not title_matches(title, preferred_titles):
                    continue

                # Build description: strip HTML from description + all lists[].content
                description_html = posting.get('description', '') or ''
                lists_html = ' '.join(
                    item.get('content', '') for item in posting.get('lists', [])
                )
                description_clean = strip_html(description_html + ' ' + lists_html)

                workplace_note = f"\nWorkplace: {workplace_type}" if workplace_type else ""
                raw_text = _build_raw_text(
                    title=title,
                    company=slug.upper(),
                    location=f"{location}{workplace_note}",
                    url=url,
                    description=description_clean,
                )

                full_text = title + " " + description_clean
                jobs_scraped += 1

                if contains_hard_no_keyword(full_text, hard_no_keywords):
                    jobs_filtered += 1
                    continue

                # Normalize location: remote workplaceType with no non-US signal -> "Remote"
                effective_location = location
                if workplace_type == "remote" and not any(
                    signal in location.lower() for signal in NON_US_SIGNALS
                ):
                    effective_location = "Remote"

                job_id = make_id("lever", posting["id"])
                passed, filter_reason = passes_filters(full_text, title, effective_location, config, source="lever", debug=True)
                if not passed:
                    jobs_filtered += 1
                    _insert_scraped_job(
                        source="lever",
                        job_id=job_id,
                        title=title,
                        company=slug,
                        location=location,
                        url=url,
                        raw_text=raw_text,
                        profile=profile,
                        filter_reason=filter_reason,
                    )
                    continue

                if _insert_scraped_job(
                    source="lever",
                    job_id=job_id,
                    title=title,
                    company=slug,
                    location=location,
                    url=url,
                    raw_text=raw_text,
                    profile=profile,
                ):
                    new_jobs_saved += 1
                    company_new_count += 1

            status = "[+]" if company_new_count > 0 else "[-]"
            logger.info("%s %s -> %d new jobs", status, slug, company_new_count)

        except Exception as e:
            errors.append(f"[!] {slug}: {str(e)}")
            logger.error("%s %s -> ERROR: %s", "[!]", slug, str(e))

    logger.info("Lever companies checked: %d", companies_checked)
    logger.info("New jobs saved: %d", new_jobs_saved)
    if errors:
        logger.warning("Errors (%d):", len(errors))
        for error in errors:
            logger.warning(error)

    return {
        'companies_checked': companies_checked,
        'new_jobs_saved': new_jobs_saved,
        'jobs_scraped': jobs_scraped,
        'jobs_filtered': jobs_filtered,
        'errors': errors,
    }


def scrape_hn(config: Dict[str, Any], profile: Optional[str] = None) -> Dict[str, Any]:
    """
    Scrape HN Who's Hiring thread via Algolia API.
    Returns dict with: {thread_found, new_jobs_saved, jobs_scraped, jobs_filtered, errors}
    """
    init_db(profile=profile)

    preferences = config.get('preferences', {})
    preferred_titles = _preferred_titles(config)
    desired_skills = preferences.get('desired_skills', [])
    hard_no_keywords = preferences.get('hard_no_keywords', [])

    new_jobs_saved = 0
    jobs_scraped = 0
    jobs_filtered = 0
    errors = []

    logger.info("Scraping HN Who's Hiring...")
    logger.info("Looking for: %s", ", ".join(preferred_titles))
    logger.info("Hard no keywords: %s", ", ".join(hard_no_keywords))

    try:
        with httpx.Client(timeout=10.0) as client:
            # Step 1: Find the latest "Who is Hiring?" thread from last 45 days
            forty_five_days_ago = datetime.now() - timedelta(days=45)
            forty_five_days_ago_unix = int(forty_five_days_ago.timestamp())
            
            search_params = {
                "query": "Who is Hiring?",
                "tags": "story",
                "hitsPerPage": 10,  # Get a few to find the latest
                "numericFilters": f"created_at_i>{forty_five_days_ago_unix}"
            }

            response = client.get(HN_ALGOLIA_URL, params=search_params)
            
            if response.status_code != 200:
                errors.append(f"[!] HN search failed: HTTP {response.status_code}")
                return {
                    'thread_found': False,
                    'new_jobs_saved': new_jobs_saved,
                    'jobs_scraped': jobs_scraped,
                    'jobs_filtered': jobs_filtered,
                    'errors': errors,
                }

            data = response.json()
            hits = data.get('hits', [])
            
            # Filter for whoishiring posts and find the latest
            whoishiring_posts = [hit for hit in hits if hit.get('author') == 'whoishiring']
            
            if not whoishiring_posts:
                logger.warning("No 'Who is Hiring?' threads found in the last 45 days")
                logger.warning("This might mean the monthly thread hasn't been posted yet, or there could be an issue with the HN API.")
                return {
                    'thread_found': False,
                    'new_jobs_saved': new_jobs_saved,
                    'jobs_scraped': jobs_scraped,
                    'jobs_filtered': jobs_filtered,
                    'errors': errors,
                }

            # Sort by creation time (newest first) and take the first
            whoishiring_posts.sort(key=lambda x: x.get('created_at_i', 0), reverse=True)
            thread = whoishiring_posts[0]
            thread_id = thread['objectID']
            thread_title = thread['title']
            
            logger.info("Found thread: %s", thread_title)
            logger.info("Thread ID: %s", thread_id)

            # Step 2: Fetch top-level comments using HN Firebase API
            hn_api_base = "https://hacker-news.firebaseio.com/v0"
            
            # Get thread details to find comment IDs
            thread_url = f"{hn_api_base}/item/{thread_id}.json"
            response = client.get(thread_url)
            
            if response.status_code != 200:
                errors.append(f"[!] HN thread fetch failed: HTTP {response.status_code}")
                return {
                    'thread_found': True,
                    'new_jobs_saved': new_jobs_saved,
                    'jobs_scraped': jobs_scraped,
                    'jobs_filtered': jobs_filtered,
                    'errors': errors,
                }

            thread_data = response.json()
            comment_ids = thread_data.get('kids', [])
            
            logger.info("Found %d top-level comments", len(comment_ids))

            for comment_id in comment_ids[:100]:  # Limit to first 100 to avoid too many requests
                comment_url = f"{hn_api_base}/item/{comment_id}.json"
                response = client.get(comment_url)
                
                if response.status_code != 200:
                    continue  # Skip failed comment fetches
                    
                comment_data = response.json()
                comment_text = strip_html(comment_data.get('text', '').strip())

                if not comment_text:
                    continue

                # Skip if too short (probably not a real job posting)
                if len(comment_text) < 50:
                    continue

                jobs_scraped += 1

                # Extract basic info from text (Claude will do better parsing)
                # For now, just use the first line as title, rest as description
                lines = comment_text.split('\n', 1)
                title = lines[0].strip() if lines else "Job Posting"
                description = lines[1].strip() if len(lines) > 1 else comment_text

                # Skip if title doesn't match keywords
                if not title_matches(title, preferred_titles):
                    jobs_filtered += 1
                    continue

                # Build raw_text for Claude (include full comment)
                raw_text = f"""
HN Job Posting
Comment ID: {comment_id}
Thread: {thread_title}

{comment_text}
""".strip()

                raw_text = extract_job_context(raw_text, max_chars=2000)

                # Check for hard no keywords in full posting
                if contains_hard_no_keyword(comment_text, hard_no_keywords):
                    jobs_filtered += 1
                    continue

                # Pre-filter (title blocklist, YOE, hiring intent, US location)
                job_id = make_id("hackernews", str(comment_id))
                hn_url = f"https://news.ycombinator.com/item?id={comment_id}"
                passed, filter_reason = passes_filters(comment_text, title, "", config, source="hackernews", debug=True)
                if not passed:
                    jobs_filtered += 1
                    insert_job(Job(
                        id=job_id, title=title, company="Unknown",
                        location="Unknown", url=hn_url, raw_text=raw_text,
                        source="hackernews", scrape_qualified=0,
                        scrape_filter_reason=filter_reason,
                    ), profile=profile)
                    continue

                # Insert and track
                if insert_job(Job(
                    id=job_id, title=title, company="Unknown",
                    location="Unknown", url=hn_url, raw_text=raw_text,
                    source="hackernews",
                ), profile=profile):
                    new_jobs_saved += 1

        logger.info("Saved %d new HN jobs", new_jobs_saved)

    except Exception as e:
        errors.append(f"[!] HN scraping error: {str(e)}")
        logger.error("HN scraping error: %s", e)

    # Summary
    logger.info("Thread found: %s", 'Yes' if not errors else 'No')
    logger.info("New jobs saved: %d", new_jobs_saved)
    if errors:
        logger.warning("Errors (%d):", len(errors))
        for error in errors:
            logger.warning(error)

    return {
        'thread_found': len(errors) == 0,
        'new_jobs_saved': new_jobs_saved,
        'jobs_scraped': jobs_scraped,
        'jobs_filtered': jobs_filtered,
        'errors': errors,
    }


ASHBY_API_BASE = "https://api.ashbyhq.com/posting-api/job-board"


def _ashby_slug_variants(company_slug: str) -> List[str]:
    """Try the configured Ashby board name first, then common normalized variants."""
    variants = []
    for slug in (company_slug, company_slug.strip(), company_slug.strip().lower()):
        if slug and slug not in variants:
            variants.append(slug)
    return variants


def _ashby_location(posting: Dict[str, Any]) -> str:
    location = posting.get("location") or posting.get("locationName")
    if location:
        return str(location)

    secondary = posting.get("secondaryLocations") or []
    secondary_names = [
        str(item.get("location"))
        for item in secondary
        if isinstance(item, dict) and item.get("location")
    ]
    if secondary_names:
        return " | ".join(secondary_names)

    return "Unknown"


def _ashby_description(posting: Dict[str, Any]) -> str:
    description_plain = posting.get("descriptionPlain")
    if description_plain:
        return str(description_plain)
    return strip_html(posting.get("descriptionHtml", "") or "")


def _fetch_ashby_board(
    client: httpx.Client,
    company_slug: str,
) -> tuple[Optional[httpx.Response], str, Optional[str]]:
    """Fetch an Ashby board, trying exact and normalized slug variants."""
    last_response = None
    last_error = None

    for slug_variant in _ashby_slug_variants(company_slug):
        try:
            response = client.get(f"{ASHBY_API_BASE}/{slug_variant}")
        except httpx.HTTPError as exc:
            last_error = str(exc)
            continue

        if response.status_code == 200:
            return response, slug_variant, None
        last_response = response

    return last_response, company_slug, last_error


def scrape_ashby(config: Dict[str, Any], slugs: Optional[List[str]] = None, profile: Optional[str] = None) -> Dict[str, Any]:
    """
    Scrape all configured Ashby companies.
    Returns dict with: {companies_checked, new_jobs_saved, jobs_scraped, jobs_filtered, errors}
    """
    init_db(profile=profile)

    companies = slugs if slugs is not None else []
    preferences = config.get('preferences', {})
    profile_info = config.get('profile', {})

    preferred_titles = _preferred_titles(config)
    hard_no_keywords = preferences.get('hard_no_keywords', [])
    max_job_age_days = _configured_max_job_age_days(config)

    companies_checked = 0
    new_jobs_saved = 0
    jobs_scraped = 0
    jobs_filtered = 0
    errors = []

    logger.info("Scraping %s Ashby companies...", len(companies))
    logger.info("Looking for: %s", ", ".join(preferred_titles))
    logger.info("Hard no keywords: %s", ", ".join(hard_no_keywords))

    max_age = timedelta(days=max_job_age_days)

    for company_slug in companies:
        companies_checked += 1
        try:
            with httpx.Client(timeout=10.0) as client:
                response, resolved_slug, fetch_error = _fetch_ashby_board(client, company_slug)

            if response is None or response.status_code != 200:
                status_code = response.status_code if response is not None else (fetch_error or "no response")
                errors.append(f"[!] {company_slug}: HTTP {status_code}")
                continue

            data = response.json()
            postings = data.get("jobs", data.get("jobPostings", []))
            company_new_count = 0

            for posting in postings:
                if posting.get("isListed") is False:
                    continue

                published_at = posting.get("publishedAt") or posting.get("publishedDate")
                if _is_posted_at_older_than(published_at, max_age):
                    continue

                title = posting.get("title", "")
                location = _ashby_location(posting)
                job_url = posting.get("jobUrl", "")

                if not title_matches(title, preferred_titles):
                    continue

                description_clean = _ashby_description(posting)
                raw_text = _build_raw_text(
                    title=title,
                    company=resolved_slug.upper(),
                    location=location,
                    url=job_url,
                    description=description_clean,
                )
                full_text = title + " " + description_clean
                jobs_scraped += 1

                if contains_hard_no_keyword(full_text, hard_no_keywords):
                    jobs_filtered += 1
                    continue

                job_id = make_id("ashby", str(posting.get("id") or job_url))
                passed, filter_reason = passes_filters(full_text, title, location, config, source="ashby", debug=True)
                if not passed:
                    jobs_filtered += 1
                    _insert_scraped_job(
                        source="ashby",
                        job_id=job_id,
                        title=title,
                        company=resolved_slug,
                        location=location,
                        url=job_url,
                        raw_text=raw_text,
                        profile=profile,
                        filter_reason=filter_reason,
                    )
                    continue

                if _insert_scraped_job(
                    source="ashby",
                    job_id=job_id,
                    title=title,
                    company=resolved_slug,
                    location=location,
                    url=job_url,
                    raw_text=raw_text,
                    profile=profile,
                ):
                    new_jobs_saved += 1
                    company_new_count += 1

            status = "[+]" if company_new_count > 0 else "[-]"
            logger.info("%s %s -> %d new jobs", status, company_slug, company_new_count)

        except Exception as e:
            errors.append(f"[!] {company_slug}: {str(e)}")
            logger.error("%s %s -> ERROR: %s", "[!]", company_slug, str(e))

    logger.info("Ashby companies checked: %d", companies_checked)
    logger.info("New jobs saved: %d", new_jobs_saved)
    if errors:
        logger.warning("Errors (%d):", len(errors))
        for error in errors:
            logger.warning(error)

    return {
        'companies_checked': companies_checked,
        'new_jobs_saved': new_jobs_saved,
        'jobs_scraped': jobs_scraped,
        'jobs_filtered': jobs_filtered,
        'errors': errors,
    }


WORKABLE_WIDGET_API_BASE = "https://apply.workable.com/api/v1/widget/accounts"


def _workable_postings(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    postings = data.get("jobs", data.get("results", []))
    return postings if isinstance(postings, list) else []


def _workable_location(posting: Dict[str, Any]) -> str:
    if posting.get("telecommuting"):
        return "Remote"

    location_data = posting.get("location")
    if isinstance(location_data, dict):
        city = location_data.get("city", "")
        region = location_data.get("region") or location_data.get("state", "")
        country = location_data.get("country", "")
    else:
        city = posting.get("city", "")
        region = posting.get("state", "")
        country = posting.get("country", "")

    parts = [part for part in (city, region, country) if part]
    if parts:
        return ", ".join(parts)

    locations = posting.get("locations") or []
    names = []
    for item in locations:
        if not isinstance(item, dict):
            continue
        item_parts = [
            item.get("city", ""),
            item.get("region", ""),
            item.get("country", ""),
        ]
        name = ", ".join(part for part in item_parts if part)
        if name:
            names.append(name)
    if names:
        return " | ".join(names)

    return "Unknown"


def _workable_description(posting: Dict[str, Any]) -> str:
    description = strip_html(posting.get("description", "") or "")
    metadata = [
        f"{label}: {posting[key]}"
        for key, label in (
            ("department", "Department"),
            ("employment_type", "Employment type"),
            ("experience", "Experience"),
            ("education", "Education"),
            ("function", "Function"),
            ("industry", "Industry"),
        )
        if posting.get(key)
    ]
    return "\n".join([part for part in (description, "\n".join(metadata)) if part])


def _workable_job_id(posting: Dict[str, Any]) -> str:
    return str(posting.get("id") or posting.get("shortcode") or posting.get("url") or posting.get("shortlink") or "")


def scrape_workable(config: Dict[str, Any], slugs: Optional[List[str]] = None, profile: Optional[str] = None) -> Dict[str, Any]:
    """
    Scrape all configured Workable companies.
    Returns dict with: {companies_checked, new_jobs_saved, jobs_scraped, jobs_filtered, errors}
    """
    init_db(profile=profile)

    companies = slugs if slugs is not None else []
    preferences = config.get('preferences', {})

    preferred_titles = preferences.get('titles', [])
    hard_no_keywords = preferences.get('hard_no_keywords', [])
    max_job_age_days = _configured_max_job_age_days(config)
    max_age = timedelta(days=max_job_age_days)

    companies_checked = 0
    new_jobs_saved = 0
    jobs_scraped = 0
    jobs_filtered = 0
    errors = []

    logger.info("Scraping %s Workable companies...", len(companies))
    logger.info("Looking for: %s", ", ".join(preferred_titles))
    logger.info("Hard no keywords: %s", ", ".join(hard_no_keywords))

    for slug in companies:
        companies_checked += 1
        try:
            url = f"{WORKABLE_WIDGET_API_BASE}/{slug}"
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url)

            if response.status_code != 200:
                errors.append(f"[!] {slug}: HTTP {response.status_code}")
                continue

            data = response.json()
            postings = _workable_postings(data)
            if not postings and "jobs" not in data and "results" not in data:
                errors.append(f"[!] {slug}: unexpected response (not a list)")
                continue

            company_new_count = 0

            for posting in postings:
                posted_at = posting.get("published_on") or posting.get("created_at")
                if _is_posted_at_older_than(posted_at, max_age):
                    continue

                title = posting.get("title", "")
                telecommuting = bool(posting.get("telecommuting"))
                location = _workable_location(posting)

                job_url = posting.get("url") or posting.get("shortlink") or posting.get("application_url") or ""

                if not title_matches(title, preferred_titles):
                    continue

                description_clean = _workable_description(posting)

                workplace_note = "\nWorkplace: Remote" if telecommuting else ""
                raw_text = _build_raw_text(
                    title=title,
                    company=slug.upper(),
                    location=f"{location}{workplace_note}",
                    url=job_url,
                    description=description_clean,
                )

                full_text = title + " " + description_clean
                jobs_scraped += 1

                if contains_hard_no_keyword(full_text, hard_no_keywords):
                    jobs_filtered += 1
                    continue

                job_id = make_id("workable", _workable_job_id(posting))
                passed, filter_reason = passes_filters(full_text, title, location, config, source="workable", debug=True)
                if not passed:
                    jobs_filtered += 1
                    _insert_scraped_job(
                        source="workable",
                        job_id=job_id,
                        title=title,
                        company=slug,
                        location=location,
                        url=job_url,
                        raw_text=raw_text,
                        profile=profile,
                        filter_reason=filter_reason,
                    )
                    continue

                if _insert_scraped_job(
                    source="workable",
                    job_id=job_id,
                    title=title,
                    company=slug,
                    location=location,
                    url=job_url,
                    raw_text=raw_text,
                    profile=profile,
                ):
                    new_jobs_saved += 1
                    company_new_count += 1

            status = "[+]" if company_new_count > 0 else "[-]"
            logger.info("%s %s -> %d new jobs", status, slug, company_new_count)

        except Exception as e:
            errors.append(f"[!] {slug}: {str(e)}")
            logger.error("%s %s -> ERROR: %s", "[!]", slug, str(e))

    logger.info("Workable companies checked: %d", companies_checked)
    logger.info("New jobs saved: %d", new_jobs_saved)
    if errors:
        logger.warning("Errors (%d):", len(errors))
        for error in errors:
            logger.warning(error)

    return {
        'companies_checked': companies_checked,
        'new_jobs_saved': new_jobs_saved,
        'jobs_scraped': jobs_scraped,
        'jobs_filtered': jobs_filtered,
        'errors': errors,
    }


HIMALAYAS_API_URL = "https://himalayas.app/jobs/api"


def _himalayas_company(posting: Dict[str, Any]) -> str:
    if posting.get("companyName"):
        return str(posting["companyName"])

    company_info = posting.get("company", {})
    if isinstance(company_info, dict):
        return str(company_info.get("name") or "Unknown")
    if company_info:
        return str(company_info)
    return "Unknown"


def _himalayas_location(posting: Dict[str, Any]) -> str:
    location = posting.get("location")
    if location:
        return str(location)

    restrictions = posting.get("locationRestrictions") or []
    if restrictions:
        return " | ".join(str(item) for item in restrictions)

    return "Remote"


def _himalayas_url(posting: Dict[str, Any]) -> str:
    return str(posting.get("applicationLink") or posting.get("url") or posting.get("guid") or "")


def scrape_himalayas(config: Dict[str, Any], profile: Optional[str] = None) -> Dict[str, Any]:
    """
    Scrape Himalayas remote job feed (no company list — feed-based like HN).
    Returns dict with: {thread_found, new_jobs_saved, jobs_scraped, jobs_filtered, errors}
    """
    init_db(profile=profile)

    preferences = config.get('preferences', {})

    preferred_titles = _preferred_titles(config)
    hard_no_keywords = preferences.get('hard_no_keywords', [])
    max_job_age_days = _configured_max_job_age_days(config)
    max_age = timedelta(days=max_job_age_days)

    new_jobs_saved = 0
    jobs_scraped = 0
    jobs_filtered = 0
    errors = []

    logger.info("Scraping Himalayas remote jobs...")
    logger.info("Looking for: %s", ", ".join(preferred_titles))
    logger.info("Hard no keywords: %s", ", ".join(hard_no_keywords))

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(HIMALAYAS_API_URL)

        if response.status_code != 200:
            errors.append(f"[!] Himalayas: HTTP {response.status_code}")
            return {
                'thread_found': False,
                'new_jobs_saved': 0,
                'jobs_scraped': 0,
                'jobs_filtered': 0,
                'errors': errors,
            }

        data = response.json()
        postings = data.get("jobs", [])
        logger.info("Himalayas: %d jobs returned", len(postings))

        for posting in postings:
            if _is_posted_at_older_than(posting.get("pubDate"), max_age):
                continue

            title = posting.get("title", "")
            company_name = _himalayas_company(posting)
            location = _himalayas_location(posting)
            job_url = _himalayas_url(posting)

            if not title_matches(title, preferred_titles):
                jobs_filtered += 1
                continue

            description = posting.get("description", "") or ""
            description_clean = strip_html(description)

            raw_text = _build_raw_text(
                title=title,
                company=company_name,
                location=location,
                url=job_url,
                description=description_clean,
            )

            full_text = title + " " + description_clean
            jobs_scraped += 1

            if contains_hard_no_keyword(full_text, hard_no_keywords):
                jobs_filtered += 1
                continue

            job_id = make_id("himalayas", str(posting.get("id", job_url)))
            passed, filter_reason = passes_filters(full_text, title, location, config, source="himalayas", debug=True)
            if not passed:
                jobs_filtered += 1
                _insert_scraped_job(
                    source="himalayas",
                    job_id=job_id,
                    title=title,
                    company=company_name,
                    location=location,
                    url=job_url,
                    raw_text=raw_text,
                    profile=profile,
                    filter_reason=filter_reason,
                )
                continue

            if _insert_scraped_job(
                source="himalayas",
                job_id=job_id,
                title=title,
                company=company_name,
                location=location,
                url=job_url,
                raw_text=raw_text,
                profile=profile,
            ):
                new_jobs_saved += 1

    except Exception as e:
        errors.append(f"[!] Himalayas: {str(e)}")
        logger.error("Himalayas scraping error: %s", e)

    logger.info("New Himalayas jobs saved: %d", new_jobs_saved)
    if errors:
        logger.warning("Errors (%d):", len(errors))
        for error in errors:
            logger.warning(error)

    return {
        'thread_found': len(errors) == 0,
        'new_jobs_saved': new_jobs_saved,
        'jobs_scraped': jobs_scraped,
        'jobs_filtered': jobs_filtered,
        'errors': errors,
    }


if __name__ == "__main__":
    if "--refresh-slugs" in sys.argv:
        clear_discovered_slugs()  # clears all ATS slug cache tables
        print("Slug cache cleared — re-populate by running the pipeline.")

    config = load_config()
    print("Choose scraper:")
    print("1. Greenhouse")
    print("2. Lever")
    print("3. HN Who's Hiring")
    print("4. All")
    choice = input("Enter choice [1-4]: ").strip()

    if choice in ("1", "2", "4"):
        slug_map = get_or_discover_slugs(config)

    if choice == "1" or choice == "4":
        if config.get("sources", {}).get("greenhouse", {}).get("enabled", True):
            scrape_greenhouse(config, slugs=slug_map["greenhouse"])

    if choice == "2" or choice == "4":
        if config.get("sources", {}).get("lever", {}).get("enabled", True):
            scrape_lever(config, slugs=slug_map["lever"])

    if choice == "3" or choice == "4":
        scrape_hn(config)

    if choice not in ("1", "2", "3", "4"):
        print("Invalid choice")
