"""
text_utils.py — Smart job description extraction utilities.

Prioritizes high-signal sections (requirements, stack, role, comp) over
boilerplate when truncating long job descriptions for LLM scoring.
"""

import os
import re

# Sections ordered by scoring relevance — highest signal first
PRIORITY_SECTIONS = [
    # Requirements / qualifications
    r"(requirements?|qualifications?|what you.ll need|what we.re looking for|"
    r"you should have|must have|minimum qualifications?|basic qualifications?)",

    # Tech stack
    r"(tech(nology)? stack|technologies|tools|languages|frameworks|"
    r"what you.ll (use|work with)|our stack)",

    # Role description
    r"(about the role|the role|position|responsibilities|what you.ll do|"
    r"what you will do|your role|job description|overview)",

    # Compensation
    r"(compensation|salary|pay|equity|benefits|total rewards|"
    r"what we offer|perks|package)",

    # Experience level
    r"(experience|background|years|yoe|seniority)",

    # Location / remote
    r"(location|remote|hybrid|on-?site|where you.ll work|timezone)",

    # About the company
    r"(about us|about the company|who we are|our mission|our story|"
    r"what we do|company overview)",
]

# Set to True when DEBUG=1 in environment — prints extraction decisions
_DEBUG = os.environ.get("DEBUG", "").strip() == "1"
_debug_count = 0          # module-level counter; reset per scraper run
_DEBUG_LIMIT  = 3         # only print for first N jobs


def extract_job_context(text: str, max_chars: int = 2000) -> str:
    """
    Extracts the most scoring-relevant content from a job description.

    Prioritizes requirements, stack, role description, and comp over
    boilerplate. Falls back to head+tail if no sections are detected.

    Edge cases handled:
    - Already under limit: returned as-is
    - No paragraph breaks (HN single-block posts): _head_tail()
    - Single giant paragraph: included if under limit, else truncated
    - All sections over limit: highest-priority section truncated
    - Windows line endings: normalized before splitting
    - Empty / whitespace-only: returns ""
    """
    global _debug_count

    if not text or not text.strip():
        return ""

    # Normalize Windows line endings before splitting
    text = text.replace("\r\n", "\n")

    if len(text) <= max_chars:
        return text

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    if not paragraphs:
        return _head_tail(text, max_chars)

    # Score each paragraph against priority section patterns.
    # Lower index = higher priority. Default = lowest priority (no match).
    scored = []
    for para in paragraphs:
        para_lower = para.lower()
        priority = len(PRIORITY_SECTIONS)
        for i, pattern in enumerate(PRIORITY_SECTIONS):
            if re.search(pattern, para_lower):
                priority = i
                break
        scored.append((priority, para))

    # Sort by priority, preserving original order within the same priority
    scored_sorted = sorted(scored, key=lambda x: x[0])

    # Greedily fill up to max_chars
    selected = set()
    used = 0
    for priority, para in scored_sorted:
        if used + len(para) + 1 <= max_chars:
            selected.add(para)
            used += len(para) + 1
        elif priority <= 2 and used < max_chars - 200:
            # High-priority section: truncate to fit remaining space
            remaining = max_chars - used - 4
            selected.add(para[:remaining] + "...")
            used = max_chars
            break

    if not selected:
        return _head_tail(text, max_chars)

    # Re-join in original document order
    result = "\n\n".join(p for p in paragraphs if p in selected or
                         any(p.startswith(s[:-3]) for s in selected if s.endswith("...")))

    if _DEBUG and _debug_count < _DEBUG_LIMIT:
        _debug_count += 1
        _print_debug(scored_sorted, selected, used, max_chars)

    return result


def _head_tail(text: str, max_chars: int) -> str:
    """
    Fallback: first 60% + last 40% of allowed chars.

    The opening usually contains the role description;
    the end usually contains requirements. Beats naive truncation.
    """
    head = int(max_chars * 0.6)
    tail = int(max_chars * 0.4)
    return text[:head] + "\n\n...\n\n" + text[-tail:]


def _print_debug(
    scored: list,
    selected: set,
    used: int,
    max_chars: int,
) -> None:
    """Print extraction decisions to stdout for the first N jobs."""
    section_names = [
        "requirements", "tech stack", "role description",
        "compensation", "experience", "location", "about company",
    ]
    print(f"\n  [extract_job_context] {used}/{max_chars} chars used")
    for priority, para in scored[:8]:
        name = section_names[priority] if priority < len(section_names) else "unclassified"
        kept = "KEPT" if para in selected or any(
            para.startswith(s[:-3]) for s in selected if s.endswith("...")
        ) else "skip"
        preview = para[:60].replace("\n", " ")
        print(f"    [{kept}] pri={priority} ({name}): {preview!r}")
