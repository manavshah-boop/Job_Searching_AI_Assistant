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
    """
    raw_path = Path(resume_value)
    candidates: list[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
        candidates.append(profile_dir / raw_path.name)
        candidates.append(_BASE_DIR / raw_path.name)
    else:
        candidates.append(profile_dir / raw_path)
        candidates.append(_BASE_DIR / raw_path)
        candidates.append(profile_dir / raw_path.name)
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
        pdf_path = _resolve_resume_path(config['profile']['resume_file'], profile_dir)
        config['profile']['resume'] = extract_text_from_pdf(str(pdf_path))
    elif not config['profile'].get('resume'):
        raise ValueError("Either 'resume' text or 'resume_file' path must be provided in config.yaml")

    return config


# Example usage
if __name__ == "__main__":
    config = load_config()
    print("Config loaded successfully!")
    print(f"Resume length: {len(config['profile']['resume'])} characters")
