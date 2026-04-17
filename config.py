"""
config.py — Configuration loader with PDF resume support.

Loads config.yaml and extracts text from resume PDF if specified.

Profile support:
  load_config(profile="manav") loads from profiles/manav/config.yaml.
  load_config() (no arg) loads from root config.yaml for CLI backwards compat.
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional

import PyPDF2
import yaml
from loguru import logger

_BASE_DIR = Path(__file__).parent


def _resolve_resume_path(
    resume_value: str,
    profile_dir: Path,
) -> Path:
    """
    Resolve a resume path in a portable way.

    Supports:
    - relative paths inside the profile directory
    - legacy absolute paths from another machine by falling back to a matching
      filename in the profile directory or project root

    When profile_dir is a profile-scoped subdirectory (not the project root),
    relative paths are only searched inside that profile directory. This prevents
    a profile from accidentally picking up a sibling profile's or the root resume.
    """
    raw_path = Path(resume_value)
    candidates: list[Path] = []

    # Profile-scoped calls (profiles/jia_shah/) must not fall back to _BASE_DIR
    # for relative paths — that would silently load another profile's resume.
    is_profile_scoped = profile_dir.resolve() != _BASE_DIR.resolve()

    if raw_path.is_absolute():
        candidates.append(raw_path)
        candidates.append(profile_dir / raw_path.name)
        if not is_profile_scoped:
            candidates.append(_BASE_DIR / raw_path.name)
    else:
        candidates.append(profile_dir / raw_path)
        candidates.append(profile_dir / raw_path.name)
        if not is_profile_scoped:
            candidates.append(_BASE_DIR / raw_path)
            candidates.append(_BASE_DIR / raw_path.name)

    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        if candidate.exists():
            return candidate

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Resume PDF not found. Tried:\n"
        f"{searched}"
    )


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF {pdf_path}: {e}")


def load_config(profile: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML, handling resume PDF extraction.

    If profile is given, loads from profiles/{profile}/config.yaml.
    Otherwise loads from root config.yaml (CLI backwards compatibility).
    """
    if profile:
        config_path = _BASE_DIR / "profiles" / profile / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Profile '{profile}' not found. "
                f"Run the dashboard to create it, or check the name in profiles/."
            )
        profile_dir = config_path.parent
    else:
        config_path = _BASE_DIR / "config.yaml"
        profile_dir = _BASE_DIR

    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # Handle resume: either inline text or PDF file
    if config['profile'].get('resume_file'):
        try:
            pdf_path = _resolve_resume_path(config['profile']['resume_file'], profile_dir)
            config['profile']['resume'] = extract_text_from_pdf(str(pdf_path))
        except FileNotFoundError:
            logger.warning(
                f"Resume file '{config['profile']['resume_file']}' not found in "
                f"{profile_dir} — scoring will use bio and preferences only. "
                f"Add the resume PDF to profiles/{profile}/ to enable full scoring."
            )
            config['profile']['resume'] = ""
    elif not config['profile'].get('resume'):
        config['profile']['resume'] = ""
        logger.warning(
            "No resume text or resume_file specified in config.yaml — "
            "scoring will use bio and preferences only."
        )

    return config


# Example usage
if __name__ == "__main__":
    config = load_config()
    print("Config loaded successfully!")
    print(f"Resume length: {len(config['profile']['resume'])} characters")
