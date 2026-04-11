"""
config.py — Configuration loader with PDF resume support.

Loads config.yaml and extracts text from resume PDF if specified.
"""

import re
import yaml
from pathlib import Path
from typing import Dict, Any
import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using pdfplumber (handles layout better than PyPDF2)."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=3, y_tolerance=3)
                if text:
                    pages.append(text.strip())
            result = "\n\n".join(pages)
            return re.sub(r'\n{3,}', '\n\n', result)
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF {pdf_path}: {e}")


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml, handling resume PDF extraction."""
    config_path = Path(__file__).parent / "config.yaml"

    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # Handle resume: either inline text or PDF file
    if 'resume_file' in config['profile'] and config['profile']['resume_file']:
        pdf_path = config['profile']['resume_file']
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"Resume PDF not found: {pdf_path}")
        config['profile']['resume'] = extract_text_from_pdf(pdf_path)
    elif 'resume' not in config['profile'] or not config['profile']['resume']:
        raise ValueError("Either 'resume' text or 'resume_file' path must be provided in config.yaml")

    return config


# Example usage
if __name__ == "__main__":
    config = load_config()
    print("Config loaded successfully!")
    print(f"Resume length: {len(config['profile']['resume'])} characters")