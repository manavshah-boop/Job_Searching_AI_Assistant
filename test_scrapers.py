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

from scraper import (
    scrape_ashby,
    scrape_himalayas,
    scrape_workable,
    strip_html,
)
from theirstack import (
    _generate_slug_candidates,
    resolve_ashby_slug,
    resolve_workable_slug,
)


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
