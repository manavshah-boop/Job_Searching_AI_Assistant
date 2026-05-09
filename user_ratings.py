"""user_ratings.py — Per-job user feedback that doubles as eval ground truth.

The rating UI in the dashboard writes here. Storage is the same
`profiles/<slug>/eval_labels.yaml` file that `evaluate_profile()` already reads,
so user ratings flow straight into the metrics pipeline with no extra plumbing.

Design notes:
- Rating vocabulary mirrors the existing eval label set (great_match,
  good_match, okay_match, bad_match, should_skip) — single source of truth.
- This module owns the rating store as a logical concept; the YAML IO is
  delegated to `evaluation.{load_eval_labels,save_eval_labels}`.
- Idempotent: re-rating overwrites the existing label for the same job_id.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from evaluation import ALLOWED_EVAL_LABELS, EvalLabel, load_eval_labels, save_eval_labels
from profile_intent import normalize_profile_intent


# Order is the order shown in the UI; first entry is "best", last is "worst".
RATING_OPTIONS: tuple[tuple[str, str, str], ...] = (
    ("great_match", "Great fit", "Apply ASAP — exactly what I'm looking for"),
    ("good_match", "Good fit", "Worth applying"),
    ("okay_match", "Okay", "Might apply if I have time"),
    ("bad_match", "Not relevant", "Wrong role or doesn't fit"),
    ("should_skip", "Skip", "Out of scope — don't show similar"),
)


@dataclass(frozen=True)
class UserRating:
    """A user-supplied rating for a single job."""

    profile_slug: str
    job_id: str
    label: str
    notes: str = ""
    role_family: str | None = None


def _profile_dir(profile_slug: str) -> Path:
    return Path(__file__).parent / "profiles" / profile_slug


def eval_labels_path(profile_slug: str) -> Path:
    """Where ratings are persisted for a given profile."""
    return _profile_dir(profile_slug) / "eval_labels.yaml"


def _to_user_rating(label: EvalLabel) -> UserRating:
    return UserRating(
        profile_slug=label.profile_slug,
        job_id=label.job_id,
        label=label.label,
        notes=label.notes,
        role_family=label.role_family,
    )


def _to_eval_label(rating: UserRating) -> EvalLabel:
    return EvalLabel(
        profile_slug=rating.profile_slug,
        job_id=rating.job_id,
        label=rating.label,
        notes=rating.notes,
        role_family=rating.role_family,
    )


def get_all_user_ratings(profile_slug: str) -> dict[str, UserRating]:
    """Return all stored ratings for a profile, keyed by job_id."""
    path = eval_labels_path(profile_slug)
    if not path.exists():
        return {}
    return {
        label.job_id: _to_user_rating(label)
        for label in load_eval_labels(path)
    }


def get_user_rating(profile_slug: str, job_id: str) -> UserRating | None:
    return get_all_user_ratings(profile_slug).get(str(job_id))


def set_user_rating(
    profile_slug: str,
    job_id: str,
    label: str,
    *,
    notes: str = "",
    role_family: str | None = None,
) -> UserRating:
    """Create or overwrite the rating for `job_id`. Returns the stored rating."""
    label_normalized = str(label).strip().lower()
    if label_normalized not in ALLOWED_EVAL_LABELS:
        raise ValueError(
            f"Unsupported rating label {label!r}; expected one of {ALLOWED_EVAL_LABELS}"
        )
    existing = get_all_user_ratings(profile_slug)
    new_rating = UserRating(
        profile_slug=profile_slug,
        job_id=str(job_id),
        label=label_normalized,
        notes=notes,
        role_family=role_family,
    )
    existing[new_rating.job_id] = new_rating
    _persist(profile_slug, existing.values())
    return new_rating


def clear_user_rating(profile_slug: str, job_id: str) -> bool:
    """Remove the rating for `job_id` if present. Returns True if a rating was removed."""
    existing = get_all_user_ratings(profile_slug)
    if str(job_id) not in existing:
        return False
    del existing[str(job_id)]
    _persist(profile_slug, existing.values())
    return True


def _persist(profile_slug: str, ratings: Iterable[UserRating]) -> None:
    path = eval_labels_path(profile_slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_eval_labels([_to_eval_label(r) for r in ratings], path)


def rating_counts(profile_slug: str) -> dict[str, int]:
    """Counts per label (sorted by RATING_OPTIONS order). Useful for dashboards."""
    ratings = get_all_user_ratings(profile_slug)
    counts = {label: 0 for label, _, _ in RATING_OPTIONS}
    for rating in ratings.values():
        if rating.label in counts:
            counts[rating.label] += 1
    return counts


def attach_role_family_from_config(profile_slug: str, config: dict) -> str | None:
    """Look up the role_family inferred from the profile config, for tagging new ratings."""
    try:
        return normalize_profile_intent(config).role_family
    except Exception:
        return None
