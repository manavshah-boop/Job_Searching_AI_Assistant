"""
config.py — Configuration loader with PDF resume support.

Loads config.yaml and extracts text from resume PDF if specified.

Profile support:
  load_config(profile="manav") loads from profiles/manav/config.yaml.
  load_config() (no arg) loads from root config.yaml for CLI backwards compat.
"""

import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import PyPDF2

_BASE_DIR = Path(__file__).parent


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
        pdf_path = Path(config['profile']['resume_file'])
        # Resolve relative paths against the profile dir (or project root)
        if not pdf_path.is_absolute():
            pdf_path = profile_dir / pdf_path
        if not pdf_path.exists():
            raise FileNotFoundError(f"Resume PDF not found: {pdf_path}")
        config['profile']['resume'] = extract_text_from_pdf(str(pdf_path))
    elif not config['profile'].get('resume'):
        raise ValueError("Either 'resume' text or 'resume_file' path must be provided in config.yaml")

    return config


# Example usage
if __name__ == "__main__":
    config = load_config()
    print("Config loaded successfully!")
    print(f"Resume length: {len(config['profile']['resume'])} characters")