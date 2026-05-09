from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import user_ratings
from evaluation import load_eval_labels


@pytest.fixture
def tmp_profile(tmp_path, monkeypatch) -> str:
    """Build a fresh profile dir and rebind user_ratings to use it."""
    slug = "ratingtest"
    profile_dir = tmp_path / "profiles" / slug
    profile_dir.mkdir(parents=True)
    monkeypatch.setattr(user_ratings, "_profile_dir", lambda s: tmp_path / "profiles" / s)
    return slug


def test_set_and_get_round_trip(tmp_profile):
    user_ratings.set_user_rating(tmp_profile, "job-42", "great_match", notes="loved this one")
    rating = user_ratings.get_user_rating(tmp_profile, "job-42")
    assert rating is not None
    assert rating.label == "great_match"
    assert rating.notes == "loved this one"
    assert rating.profile_slug == tmp_profile


def test_resetting_label_overwrites_in_place(tmp_profile):
    user_ratings.set_user_rating(tmp_profile, "job-1", "good_match")
    user_ratings.set_user_rating(tmp_profile, "job-1", "bad_match", notes="changed mind")
    all_ratings = user_ratings.get_all_user_ratings(tmp_profile)
    assert len(all_ratings) == 1
    assert all_ratings["job-1"].label == "bad_match"
    assert all_ratings["job-1"].notes == "changed mind"


def test_clear_rating_removes_entry(tmp_profile):
    user_ratings.set_user_rating(tmp_profile, "job-1", "good_match")
    assert user_ratings.clear_user_rating(tmp_profile, "job-1") is True
    assert user_ratings.get_user_rating(tmp_profile, "job-1") is None
    # Idempotent: clearing a missing rating returns False without raising.
    assert user_ratings.clear_user_rating(tmp_profile, "job-1") is False


def test_invalid_label_rejected(tmp_profile):
    with pytest.raises(ValueError):
        user_ratings.set_user_rating(tmp_profile, "job-1", "neutral_match")


def test_persisted_yaml_is_compatible_with_eval_loader(tmp_profile):
    user_ratings.set_user_rating(tmp_profile, "job-1", "great_match", notes="ideal")
    user_ratings.set_user_rating(tmp_profile, "job-2", "bad_match")
    path = user_ratings.eval_labels_path(tmp_profile)
    # The eval loader and the rating store must read the same file.
    loaded = load_eval_labels(path)
    by_id = {label.job_id: label for label in loaded}
    assert by_id["job-1"].label == "great_match"
    assert by_id["job-2"].label == "bad_match"


def test_rating_counts_reflect_persisted_state(tmp_profile):
    user_ratings.set_user_rating(tmp_profile, "j1", "great_match")
    user_ratings.set_user_rating(tmp_profile, "j2", "great_match")
    user_ratings.set_user_rating(tmp_profile, "j3", "okay_match")
    counts = user_ratings.rating_counts(tmp_profile)
    assert counts["great_match"] == 2
    assert counts["okay_match"] == 1
    assert counts["bad_match"] == 0


def test_yaml_keeps_all_entries_after_multiple_writes(tmp_profile):
    for i in range(5):
        user_ratings.set_user_rating(tmp_profile, f"job-{i}", "good_match")
    path = user_ratings.eval_labels_path(tmp_profile)
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert {e["job_id"] for e in raw["labels"]} == {f"job-{i}" for i in range(5)}


def test_role_family_persists_when_provided(tmp_profile):
    user_ratings.set_user_rating(
        tmp_profile, "job-9", "good_match", role_family="software_engineering"
    )
    rating = user_ratings.get_user_rating(tmp_profile, "job-9")
    assert rating is not None
    assert rating.role_family == "software_engineering"


def test_get_all_with_no_file_returns_empty(tmp_profile):
    assert user_ratings.get_all_user_ratings(tmp_profile) == {}
