"""
skill_extraction.py — Deterministic tech-skill detection for resumes and job descriptions.

Used by:
  - candidate_profile.build_structured_profile() to enrich the LLM-extracted profile
    with concrete canonical skill tokens pulled directly from the resume text. The
    LLM (especially Groq tool_use) is unreliable and often returns bio-flavored
    phrases like "Cloud-native systems" instead of "AWS, Docker". This module
    guarantees the profile contains crisp tokens regardless of LLM output.
  - scorer.compute_ats_score() to match a candidate skill against a JD using the
    same aliases the resume was parsed with — so e.g. "CI/CD" in the profile finds
    "continuous integration" in the JD.

Add new entries by appending to _SKILL_REGISTRY. The registry maps a canonical
display token to a list of regex patterns that should be matched (case-insensitive,
word-boundary aware where the token is short enough to collide).
"""

from __future__ import annotations

import re
from typing import Iterable

# Categorised so we can fold each match into the right structured-profile field.
# Each entry: (canonical_token, [regex_patterns...]).
# Patterns are compiled case-insensitively. Use \b at edges where the canonical
# token is short and could substring-match unrelated words (e.g. "go" → "google").

_LANGUAGES: list[tuple[str, list[str]]] = [
    ("Python",     [r"\bpython\b"]),
    ("JavaScript", [r"\bjavascript\b"]),
    ("TypeScript", [r"\btypescript\b"]),
    ("C++",        [r"c\+\+"]),
    ("Java",       [r"\bjava\b(?!script)"]),
    ("Go",         [r"\bgolang\b"]),
    ("Rust",       [r"\brust\b"]),
    ("Ruby",       [r"\bruby\b"]),
    ("Swift",      [r"\bswift\b"]),
    ("Kotlin",     [r"\bkotlin\b"]),
    ("SQL",        [r"\bsql\b"]),
    ("Bash",       [r"\bbash\b", r"\bshell script"]),
]

_FRAMEWORKS: list[tuple[str, list[str]]] = [
    ("FastAPI",      [r"\bfastapi\b"]),
    ("Flask",        [r"\bflask\b"]),
    ("Django",       [r"\bdjango\b"]),
    ("React",        [r"\breact\b"]),
    ("React Native", [r"\breact native\b"]),
    ("Node.js",      [r"\bnode\.?js\b"]),
    ("Express",      [r"\bexpress\.?js\b"]),
    ("Spring Boot",  [r"\bspring boot\b"]),
    ("Qt",           [r"\bqt\b"]),
    ("Next.js",      [r"\bnext\.?js\b"]),
    ("Pytest",       [r"\bpytest\b"]),
    ("PyTorch",      [r"\bpytorch\b"]),
    ("TensorFlow",   [r"\btensorflow\b"]),
]

_CLOUD: list[tuple[str, list[str]]] = [
    ("AWS",            [r"\baws\b", r"\bamazon web services\b"]),
    ("S3",             [r"\bs3\b"]),
    ("ECR",            [r"\becr\b"]),
    ("RDS",            [r"\brds\b"]),
    ("Lambda",         [r"\baws lambda\b", r"\blambda functions?\b"]),
    ("EC2",            [r"\bec2\b"]),
    ("Route53",        [r"\broute\s?53\b"]),
    ("GCP",            [r"\bgcp\b", r"\bgoogle cloud\b"]),
    ("Azure",          [r"\bazure\b"]),
    ("Docker",         [r"\bdocker\b"]),
    ("Kubernetes",     [r"\bkubernetes\b", r"\bk8s\b"]),
    ("Terraform",      [r"\bterraform\b"]),
]

_DATABASES: list[tuple[str, list[str]]] = [
    ("PostgreSQL", [r"\bpostgresql\b", r"\bpostgres\b"]),
    ("MySQL",      [r"\bmysql\b"]),
    ("Redis",      [r"\bredis\b"]),
    ("MongoDB",    [r"\bmongodb\b", r"\bmongo\b"]),
    ("SQLite",     [r"\bsqlite\b"]),
    ("DynamoDB",   [r"\bdynamodb\b"]),
]

_TOOLS_AND_CONCEPTS: list[tuple[str, list[str]]] = [
    ("REST APIs",            [r"\brest\s?api", r"\brestful\b"]),
    ("GraphQL",              [r"\bgraphql\b"]),
    ("gRPC",                 [r"\bgrpc\b"]),
    ("Git",                  [r"\bgit\b(?!hub)"]),
    ("GitHub Actions",       [r"\bgithub actions\b"]),
    ("CI/CD",                [r"ci/cd", r"continuous integration", r"continuous delivery", r"continuous deployment"]),
    ("Microservices",        [r"\bmicroservice"]),
    ("Distributed systems",  [r"distributed system"]),
    ("System design",        [r"\bsystem design\b", r"\bsystems? design\b"]),
    ("API design",           [r"\bapi design\b"]),
    ("LLM integration",      [r"\bllm\b", r"large language model", r"\bgpt-?\d", r"\bgemini\b", r"\bclaude\b", r"\banthropic\b"]),
    ("Machine Learning",     [r"\bmachine learning\b", r"\bml\s+(?:model|engineer|infrastructure|pipeline)"]),
    ("Agentic systems",      [r"\bagentic\b", r"\bagent(?:ic)? system"]),
    ("OOP",                  [r"\boop\b", r"object[- ]oriented"]),
    ("Data structures",      [r"data structures? (?:and|&) algorithms?", r"\bdsa\b"]),
    ("Data pipelines",       [r"data pipelines?", r"\betl\b"]),
    ("Linux",                [r"\blinux\b"]),
]

_REGISTRY: list[tuple[str, str, list[str]]] = (
    [("language",  k, p) for k, p in _LANGUAGES]
    + [("framework", k, p) for k, p in _FRAMEWORKS]
    + [("cloud",     k, p) for k, p in _CLOUD]
    + [("database",  k, p) for k, p in _DATABASES]
    + [("concept",   k, p) for k, p in _TOOLS_AND_CONCEPTS]
)


def _compile(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


# Compile once at import.
_COMPILED: list[tuple[str, str, list[re.Pattern[str]]]] = [
    (cat, canon, _compile(pats)) for cat, canon, pats in _REGISTRY
]


def text_has_skill(text: str, canonical: str) -> bool:
    """Return True if any alias for `canonical` appears in `text`. Case-insensitive."""
    if not text:
        return False
    for cat, canon, regs in _COMPILED:
        if canon.lower() == canonical.lower():
            return any(r.search(text) for r in regs)
    # Unknown canonical → fall back to substring match for safety.
    return canonical.lower() in text.lower()


def extract_skills(text: str) -> dict[str, list[str]]:
    """
    Scan `text` for every canonical skill in the registry. Returns categorised
    lists of canonical tokens (deduped, in registry order):
      {"languages": [...], "frameworks": [...], "cloud": [...],
       "databases": [...], "concepts": [...]}
    """
    found: dict[str, list[str]] = {
        "languages": [], "frameworks": [], "cloud": [],
        "databases": [], "concepts": [],
    }
    if not text:
        return found

    bucket_for = {
        "language": "languages", "framework": "frameworks", "cloud": "cloud",
        "database": "databases", "concept": "concepts",
    }
    for cat, canon, regs in _COMPILED:
        if any(r.search(text) for r in regs):
            bucket = bucket_for[cat]
            if canon not in found[bucket]:
                found[bucket].append(canon)
    return found


def all_canonical_skills() -> list[str]:
    """Return every canonical token in the registry (used for diagnostics)."""
    return [canon for _, canon, _ in _REGISTRY]
