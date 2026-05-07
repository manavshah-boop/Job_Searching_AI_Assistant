"""
test_scrapers.py — Unit tests for scrape_ashby, scrape_workable, scrape_himalayas.

Covers: passes_filters integration, HTML stripping, date filtering,
        dedup via insert_job, and slug resolution functions.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest

import scraper
from scraper import (
    _get_with_retry,
    _parse_retry_after,
    scrape_ashby,
    scrape_greenhouse,
    scrape_himalayas,
    scrape_hn,
    scrape_lever,
    scrape_workable,
    strip_html,
)
from theirstack import (
    _generate_slug_candidates,
    resolve_ashby_slug,
    resolve_workable_slug,
)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Skip retry backoff sleeps so tests don't actually wait seconds."""
    monkeypatch.setattr("scraper._sleep", lambda _: None)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _base_config(titles=None, max_age_days=30):
    return {
        "preferences": {
            "titles": titles or ["Software Engineer", "Backend Engineer"],
            "desired_skills": ["Python"],
            "hard_no_keywords": ["clearance required"],
            "filters": {
                "min_yoe": 0,
                "max_yoe": 5,
                "max_job_age_days": max_age_days,
                "require_degree_filter": False,
                "title_blocklist": ["Staff", "Director"],
            },
        },
        "profile": {},
    }


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _old_iso(days=60):
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


# ── strip_html ─────────────────────────────────────────────────────────────────

def test_strip_html_removes_tags():
    assert strip_html("<b>Hello</b> <i>world</i>") == "Hello world"


def test_strip_html_unescapes_entities():
    assert "&amp;" not in strip_html("&amp;lt;div&amp;gt;")


def test_strip_html_empty():
    assert strip_html("") == ""
    assert strip_html(None) == ""


# ── _generate_slug_candidates ──────────────────────────────────────────────────

def test_generate_slug_candidates_from_domain():
    cands = _generate_slug_candidates("", "acme.com")
    assert "acme" in cands


def test_generate_slug_candidates_from_name():
    cands = _generate_slug_candidates("Open AI", "")
    assert "openai" in cands
    assert "open-ai" in cands


def test_generate_slug_candidates_strips_suffix():
    cands = _generate_slug_candidates("Acme Inc", "")
    assert "acme" in cands


def test_generate_slug_candidates_no_duplicates():
    cands = _generate_slug_candidates("acme", "acme.io")
    assert len(cands) == len(set(cands))


# ── resolve_ashby_slug ─────────────────────────────────────────────────────────

def test_resolve_ashby_slug_returns_valid(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"jobs": []}

    monkeypatch.setattr("theirstack.httpx.get", lambda url, timeout: mock_resp)
    monkeypatch.setattr("theirstack.time.sleep", lambda _: None)

    result = resolve_ashby_slug({"name": "Linear", "domain": "linear.app"})
    assert result is not None


def test_resolve_ashby_slug_returns_none_on_404(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.status_code = 404

    monkeypatch.setattr("theirstack.httpx.get", lambda url, timeout: mock_resp)
    monkeypatch.setattr("theirstack.time.sleep", lambda _: None)

    result = resolve_ashby_slug({"name": "Nonexistent Corp", "domain": ""})
    assert result is None


# ── resolve_workable_slug ──────────────────────────────────────────────────────

def test_resolve_workable_slug_returns_valid(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"jobs": []}

    monkeypatch.setattr("theirstack.httpx.get", lambda url, timeout: mock_resp)
    monkeypatch.setattr("theirstack.time.sleep", lambda _: None)

    result = resolve_workable_slug({"name": "Acme", "domain": "acme.com"})
    assert result is not None


def test_resolve_workable_slug_returns_none_on_missing_results(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"error": "not found"}

    monkeypatch.setattr("theirstack.httpx.get", lambda url, timeout: mock_resp)
    monkeypatch.setattr("theirstack.time.sleep", lambda _: None)

    result = resolve_workable_slug({"name": "Ghost Co", "domain": ""})
    assert result is None


# ── scrape_ashby ───────────────────────────────────────────────────────────────

def _ashby_posting(title="Software Engineer", location="Remote", days_old=1, posting_id="abc123"):
    pub = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    return {
        "id": posting_id,
        "title": title,
        "location": location,
        "jobUrl": f"https://jobs.ashbyhq.com/company/{posting_id}",
        "descriptionHtml": "<p>We need a <b>Python</b> developer.</p>",
        "descriptionPlain": "We need a Python developer.",
        "publishedAt": pub,
    }


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_ashby_saves_matching_job(mock_client, mock_insert, mock_init):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"jobs": [_ashby_posting()]}
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    result = scrape_ashby(_base_config(), slugs=["linear"])

    assert result["new_jobs_saved"] == 1
    assert result["companies_checked"] == 1
    assert result["errors"] == []


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_ashby_tries_lowercase_slug_variant(mock_client, mock_insert, mock_init):
    missing = MagicMock()
    missing.status_code = 404
    found = MagicMock()
    found.status_code = 200
    found.json.return_value = {"jobs": [_ashby_posting()]}
    get_mock = mock_client.return_value.__enter__.return_value.get
    get_mock.side_effect = [missing, found]

    result = scrape_ashby(_base_config(), slugs=["OpenAI"])

    assert result["new_jobs_saved"] == 1
    assert get_mock.call_args_list[0].args[0].endswith("/OpenAI")
    assert get_mock.call_args_list[1].args[0].endswith("/openai")


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_ashby_continues_after_slug_variant_timeout(mock_client, mock_insert, mock_init):
    found = MagicMock()
    found.status_code = 200
    found.json.return_value = {"jobs": [_ashby_posting()]}
    get_mock = mock_client.return_value.__enter__.return_value.get
    get_mock.side_effect = [httpx.ReadTimeout("timed out"), found]

    result = scrape_ashby(_base_config(), slugs=["OpenAI"])

    assert result["new_jobs_saved"] == 1
    assert result["errors"] == []


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=False)
@patch("scraper.httpx.Client")
def test_scrape_ashby_dedup(mock_client, mock_insert, mock_init):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"jobs": [_ashby_posting()]}
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    result = scrape_ashby(_base_config(), slugs=["linear"])

    assert result["new_jobs_saved"] == 0  # insert_job returned False (dupe)


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_ashby_filters_old_jobs(mock_client, mock_insert, mock_init):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"jobs": [_ashby_posting(days_old=60)]}
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    result = scrape_ashby(_base_config(max_age_days=30), slugs=["linear"])

    assert result["new_jobs_saved"] == 0
    mock_insert.assert_not_called()


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_ashby_filters_title_blocklist(mock_client, mock_insert, mock_init):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"jobs": [_ashby_posting(title="Staff Engineer")]}
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    result = scrape_ashby(_base_config(), slugs=["linear"])

    # Staff is in title_blocklist → scrape_qualified=0 insert or skipped
    assert result["jobs_filtered"] >= 1


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_ashby_filters_hard_no_keyword(mock_client, mock_insert, mock_init):
    posting = _ashby_posting()
    posting.pop("descriptionPlain")
    posting["descriptionHtml"] = "<p>Security clearance required.</p>"
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"jobs": [posting]}
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    result = scrape_ashby(_base_config(), slugs=["linear"])

    assert result["jobs_filtered"] >= 1


@patch("scraper.init_db")
@patch("scraper.httpx.Client")
def test_scrape_ashby_http_error(mock_client, mock_init):
    resp = MagicMock()
    resp.status_code = 500
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    result = scrape_ashby(_base_config(), slugs=["bad-slug"])

    assert result["errors"] != []
    assert result["new_jobs_saved"] == 0


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_ashby_strips_html_from_description(mock_client, mock_insert, mock_init):
    posting = _ashby_posting()
    posting.pop("descriptionPlain")
    posting["descriptionHtml"] = "<b>Strong</b> Python skills required."
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"jobs": [posting]}
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    scrape_ashby(_base_config(), slugs=["linear"])

    call_args = mock_insert.call_args
    job = call_args[0][0]
    assert "<b>" not in job.raw_text


# ── scrape_workable ────────────────────────────────────────────────────────────

def _workable_posting(title="Software Engineer", telecommuting=True, country="US", days_old=1, posting_id="wk1"):
    created = (datetime.now(timezone.utc) - timedelta(days=days_old)).date().isoformat()
    return {
        "shortcode": posting_id,
        "title": title,
        "city": "New York",
        "state": "NY",
        "country": country,
        "telecommuting": telecommuting,
        "url": f"https://apply.workable.com/company/jobs/{posting_id}",
        "description": "<p>We need a <b>Python</b> developer.</p>",
        "published_on": created,
        "department": "Engineering",
        "employment_type": "Full-time",
    }


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_workable_saves_matching_job(mock_client, mock_insert, mock_init):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"jobs": [_workable_posting()]}
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    result = scrape_workable(_base_config(), slugs=["acme"])

    assert result["new_jobs_saved"] == 1
    assert result["errors"] == []


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_workable_telecommuting_sets_remote_location(mock_client, mock_insert, mock_init):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"jobs": [_workable_posting(telecommuting=True)]}
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    scrape_workable(_base_config(), slugs=["acme"])

    job = mock_insert.call_args[0][0]
    assert job.location == "Remote"


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_workable_non_us_filtered(mock_client, mock_insert, mock_init):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"jobs": [_workable_posting(telecommuting=False, country="Germany")]}
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    result = scrape_workable(_base_config(), slugs=["acme"])

    assert result["jobs_filtered"] >= 1


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_workable_filters_old_jobs(mock_client, mock_insert, mock_init):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"jobs": [_workable_posting(days_old=60)]}
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    result = scrape_workable(_base_config(max_age_days=30), slugs=["acme"])

    assert result["new_jobs_saved"] == 0
    mock_insert.assert_not_called()


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=False)
@patch("scraper.httpx.Client")
def test_scrape_workable_dedup(mock_client, mock_insert, mock_init):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"jobs": [_workable_posting()]}
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    result = scrape_workable(_base_config(), slugs=["acme"])

    assert result["new_jobs_saved"] == 0


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_workable_strips_html(mock_client, mock_insert, mock_init):
    posting = _workable_posting()
    posting["description"] = "<b>Strong</b> Python skills required."
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"jobs": [posting]}
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    scrape_workable(_base_config(), slugs=["acme"])

    job = mock_insert.call_args[0][0]
    assert "<b>" not in job.raw_text


# ── scrape_himalayas ───────────────────────────────────────────────────────────

def _himalayas_posting(title="Software Engineer", days_old=1, posting_id="him1"):
    pub = int((datetime.now(timezone.utc) - timedelta(days=days_old)).timestamp())
    return {
        "guid": f"https://himalayas.app/jobs/{posting_id}",
        "title": title,
        "companyName": "RemoteCo",
        "locationRestrictions": ["United States"],
        "applicationLink": f"https://himalayas.app/jobs/{posting_id}",
        "description": "<p>We need a <b>Python</b> developer.</p>",
        "pubDate": pub,
    }


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_himalayas_saves_matching_job(mock_client, mock_insert, mock_init):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"jobs": [_himalayas_posting()]}
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    result = scrape_himalayas(_base_config())

    assert result["new_jobs_saved"] == 1
    assert result["thread_found"] is True
    assert result["errors"] == []
    job = mock_insert.call_args[0][0]
    assert job.company == "RemoteCo"
    assert job.location == "United States"
    assert job.url == "https://himalayas.app/jobs/him1"


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=False)
@patch("scraper.httpx.Client")
def test_scrape_himalayas_dedup(mock_client, mock_insert, mock_init):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"jobs": [_himalayas_posting()]}
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    result = scrape_himalayas(_base_config())

    assert result["new_jobs_saved"] == 0


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_himalayas_filters_old_jobs(mock_client, mock_insert, mock_init):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"jobs": [_himalayas_posting(days_old=60)]}
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    result = scrape_himalayas(_base_config(max_age_days=30))

    assert result["new_jobs_saved"] == 0
    mock_insert.assert_not_called()


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_himalayas_filters_hard_no(mock_client, mock_insert, mock_init):
    posting = _himalayas_posting()
    posting["description"] = "<p>Security clearance required for this role.</p>"
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"jobs": [posting]}
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    result = scrape_himalayas(_base_config())

    assert result["jobs_filtered"] >= 1


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_himalayas_strips_html(mock_client, mock_insert, mock_init):
    posting = _himalayas_posting()
    posting["description"] = "<b>Strong</b> Python skills required."
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"jobs": [posting]}
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    scrape_himalayas(_base_config())

    job = mock_insert.call_args[0][0]
    assert "<b>" not in job.raw_text


@patch("scraper.init_db")
@patch("scraper.httpx.Client")
def test_scrape_himalayas_http_error(mock_client, mock_init):
    resp = MagicMock()
    resp.status_code = 503
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    result = scrape_himalayas(_base_config())

    assert result["thread_found"] is False
    assert result["errors"] != []


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_himalayas_filters_title_mismatch(mock_client, mock_insert, mock_init):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"jobs": [_himalayas_posting(title="Marketing Manager")]}
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    result = scrape_himalayas(_base_config())

    assert result["new_jobs_saved"] == 0
    assert result["jobs_filtered"] >= 1


# ── scrape_greenhouse — required-field validation (HIGH-1, HIGH-3) ────────────

def _greenhouse_posting(posting_id="123", title="Software Engineer", url="https://example.com/jobs/123"):
    return {
        "id": posting_id,
        "title": title,
        "absolute_url": url,
        "location": {"name": "Remote"},
        "content": "<p>We need a Python developer.</p>",
        "updated_at": _now_iso(),
    }


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_greenhouse_skips_missing_id_does_not_crash(mock_client, mock_insert, mock_init):
    """HIGH-1 regression: postings missing `id` must be skipped, not crash the company."""
    bad = _greenhouse_posting()
    del bad["id"]
    good = _greenhouse_posting(posting_id="ok-456")
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"jobs": [bad, good]}
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    result = scrape_greenhouse(_base_config(), slugs=["acme"])

    # The good posting is saved; the bad row was skipped quietly.
    assert result["new_jobs_saved"] == 1
    assert result["errors"] == []


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_greenhouse_skips_empty_title(mock_client, mock_insert, mock_init):
    """HIGH-3 regression: postings with empty title must not reach the DB."""
    bad = _greenhouse_posting(title="")
    good = _greenhouse_posting(posting_id="ok-456")
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"jobs": [bad, good]}
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    result = scrape_greenhouse(_base_config(), slugs=["acme"])
    assert result["new_jobs_saved"] == 1


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_greenhouse_skips_empty_url(mock_client, mock_insert, mock_init):
    """HIGH-3 regression: postings with empty url must not reach the DB."""
    bad = _greenhouse_posting(url="")
    good = _greenhouse_posting(posting_id="ok-456")
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"jobs": [bad, good]}
    mock_client.return_value.__enter__.return_value.get.return_value = resp

    result = scrape_greenhouse(_base_config(), slugs=["acme"])
    assert result["new_jobs_saved"] == 1


# ── HIGH-2: retry-with-backoff regression coverage ────────────────────────────

def _resp(status, json_value=None, headers=None):
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    resp.json.return_value = json_value if json_value is not None else {}
    return resp


def test_get_with_retry_recovers_from_503_then_succeeds(monkeypatch):
    """HIGH-2: a transient 5xx is retried, recovering on attempt 2."""
    sleeps: list = []
    monkeypatch.setattr("scraper._sleep", lambda s: sleeps.append(s))

    transient = _resp(503)
    success = _resp(200, {"ok": True})
    client = MagicMock()
    client.get.side_effect = [transient, success]

    response = _get_with_retry(client, "https://example.com/api")

    assert response.status_code == 200
    assert client.get.call_count == 2
    assert sleeps == [1.0]  # first retry waits base_backoff


def test_get_with_retry_does_not_retry_on_404(monkeypatch):
    """HIGH-2: 404 is permanent — return immediately, never sleep."""
    sleeps: list = []
    monkeypatch.setattr("scraper._sleep", lambda s: sleeps.append(s))

    not_found = _resp(404)
    client = MagicMock()
    client.get.return_value = not_found

    response = _get_with_retry(client, "https://example.com/missing")

    assert response.status_code == 404
    assert client.get.call_count == 1
    assert sleeps == []


def test_get_with_retry_does_not_retry_on_403(monkeypatch):
    """HIGH-2: 403 is permanent — return immediately, never sleep."""
    sleeps: list = []
    monkeypatch.setattr("scraper._sleep", lambda s: sleeps.append(s))

    forbidden = _resp(403)
    client = MagicMock()
    client.get.return_value = forbidden

    response = _get_with_retry(client, "https://example.com/forbidden")

    assert response.status_code == 403
    assert client.get.call_count == 1
    assert sleeps == []


def test_get_with_retry_honors_retry_after_on_429(monkeypatch):
    """HIGH-2: 429 honors the Retry-After header (in seconds) instead of default backoff."""
    sleeps: list = []
    monkeypatch.setattr("scraper._sleep", lambda s: sleeps.append(s))

    rate_limited = _resp(429, headers={"Retry-After": "7"})
    success = _resp(200, {"ok": True})
    client = MagicMock()
    client.get.side_effect = [rate_limited, success]

    response = _get_with_retry(client, "https://example.com/api")

    assert response.status_code == 200
    # The header value (7s) should be used directly, not the default 2s doubled backoff.
    assert sleeps == [7.0]


def test_get_with_retry_429_without_header_doubles_backoff(monkeypatch):
    """HIGH-2: 429 with no Retry-After header backs off twice as long as a 5xx."""
    sleeps: list = []
    monkeypatch.setattr("scraper._sleep", lambda s: sleeps.append(s))

    rate_limited = _resp(429)
    success = _resp(200, {"ok": True})
    client = MagicMock()
    client.get.side_effect = [rate_limited, success]

    _get_with_retry(client, "https://example.com/api")

    # Doubled: base_backoff * 2**attempt instead of 2**(attempt-1) → 2.0s on 1st retry.
    assert sleeps == [2.0]


def test_get_with_retry_retries_network_errors(monkeypatch):
    """HIGH-2: httpx.ReadTimeout (network-layer) triggers retry."""
    sleeps: list = []
    monkeypatch.setattr("scraper._sleep", lambda s: sleeps.append(s))

    success = _resp(200, {"ok": True})
    client = MagicMock()
    client.get.side_effect = [httpx.ReadTimeout("slow"), success]

    response = _get_with_retry(client, "https://example.com/api")

    assert response.status_code == 200
    assert sleeps == [1.0]


def test_get_with_retry_exhausts_attempts_returns_last_response(monkeypatch):
    """HIGH-2: after max_attempts of 5xx, return the last response so the
    caller's existing 'errors.append + continue' fallthrough fires."""
    monkeypatch.setattr("scraper._sleep", lambda _: None)

    client = MagicMock()
    client.get.return_value = _resp(503)

    response = _get_with_retry(client, "https://example.com/api")

    assert response.status_code == 503
    assert client.get.call_count == 3


def test_parse_retry_after_seconds():
    assert _parse_retry_after("12") == 12.0
    assert _parse_retry_after("0") == 0.0


def test_parse_retry_after_invalid_returns_none():
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("") is None
    assert _parse_retry_after("not a date") is None


# Scraper-integration regression: a transient 503 on the company endpoint
# is retried and the company scrape ultimately succeeds.
@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_greenhouse_retries_transient_503(mock_client, mock_insert, mock_init):
    transient = _resp(503)
    success = _resp(200, {"jobs": [_greenhouse_posting()]})
    mock_client.return_value.__enter__.return_value.get.side_effect = [transient, success]

    result = scrape_greenhouse(_base_config(), slugs=["acme"])

    assert result["new_jobs_saved"] == 1
    assert result["errors"] == []


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_greenhouse_does_not_retry_on_404(mock_client, mock_insert, mock_init):
    """A 404 should be recorded as an error after a single attempt — no retry."""
    get_mock = mock_client.return_value.__enter__.return_value.get
    get_mock.return_value = _resp(404)

    result = scrape_greenhouse(_base_config(), slugs=["nonexistent"])

    assert result["new_jobs_saved"] == 0
    assert result["errors"] != []
    # Single GET — no retries fired on a permanent 4xx.
    assert get_mock.call_count == 1


# ── MED-9: HN max_comments config + cap-hit warning ──────────────────────────

def _hn_search_resp(thread_id="42", title="Ask HN: Who is hiring? (May 2026)"):
    return _resp(200, {
        "hits": [
            {"author": "whoishiring", "objectID": thread_id,
             "title": title, "created_at_i": 1714000000},
        ],
    })


def _hn_thread_resp(num_kids=120):
    return _resp(200, {"kids": list(range(1, num_kids + 1))})


def _hn_comment_resp(text="Software Engineer\nWe are hiring a Python developer in San Francisco, CA."):
    return _resp(200, {"text": text})


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_hn_reads_max_comments_from_config(mock_client, mock_insert, mock_init):
    """MED-9: sources.hn.max_comments controls how many top-level comments are fetched."""
    get_mock = mock_client.return_value.__enter__.return_value.get
    # search → thread → 5 comment fetches (cap = 5)
    get_mock.side_effect = [
        _hn_search_resp(),
        _hn_thread_resp(num_kids=120),
    ] + [_hn_comment_resp() for _ in range(5)]

    config = _base_config()
    config["sources"] = {"hn": {"max_comments": 5}}

    scrape_hn(config)

    # 1 search + 1 thread + 5 comments = 7 GETs total
    assert get_mock.call_count == 2 + 5


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_hn_warns_when_cap_hit(mock_client, mock_insert, mock_init, caplog):
    """MED-9: a warning fires when the configured cap drops some comments."""
    import logging
    get_mock = mock_client.return_value.__enter__.return_value.get
    get_mock.side_effect = [
        _hn_search_resp(),
        _hn_thread_resp(num_kids=10),  # 10 comments, cap = 3 → cap hit
    ] + [_hn_comment_resp() for _ in range(3)]

    config = _base_config()
    config["sources"] = {"hn": {"max_comments": 3}}

    # loguru → propagate to the standard logging stream so caplog can see it
    from loguru import logger
    handler_id = logger.add(
        lambda msg: logging.getLogger("scraper-test").warning(msg.rstrip()),
        level="WARNING",
        format="{message}",
    )
    try:
        with caplog.at_level(logging.WARNING, logger="scraper-test"):
            scrape_hn(config)
    finally:
        logger.remove(handler_id)

    cap_warnings = [r for r in caplog.records if "HN cap hit" in r.getMessage()]
    assert cap_warnings, f"expected cap-hit warning, got: {[r.getMessage() for r in caplog.records]}"


@patch("scraper.init_db")
@patch("scraper.insert_job", return_value=True)
@patch("scraper.httpx.Client")
def test_scrape_hn_defaults_to_100_when_config_missing(mock_client, mock_insert, mock_init):
    """MED-9: missing config falls back to the previous hard-coded cap of 100."""
    get_mock = mock_client.return_value.__enter__.return_value.get
    # 150 kids — should fetch first 100 by default.
    get_mock.side_effect = [
        _hn_search_resp(),
        _hn_thread_resp(num_kids=150),
    ] + [_hn_comment_resp() for _ in range(100)]

    config = _base_config()  # no "sources" key

    scrape_hn(config)

    # 1 search + 1 thread + 100 comments
    assert get_mock.call_count == 2 + 100
