"""
test_filters.py — Sanity-check passes_filters() against known inputs.
Run with: python test_filters.py
"""
import sys
from scraper import passes_filters, _extract_yoe_numbers

# Minimal config that mirrors config.yaml
CONFIG = {
    "preferences": {
        "titles": [
            "Software Engineer", "Software Developer",
            "Software Development Engineer", "SDE", "SWE",
            "Backend Engineer", "AI Engineer",
        ],
        "desired_skills": ["Python", "ML Infrastructure", "LLM", "backend", "AWS"],
        "filters": {
            "countries_allowed": ["United States", "US", "USA", "Remote"],
            "min_yoe": 0,
            "max_yoe": 5,
            "title_blocklist": [
                "Staff", "Principal", "VP", "Director",
                "Head of", "Manager", "Lead", "Senior",
            ],
        },
    }
}

PASS = True
FAIL = False

# (description, text, title, location, source, expected)
CASES = [
    # ── Title blocklist ──────────────────────────────────────────────────────
    ("GH: blocks 'Staff Software Engineer'",
     "We are hiring a Staff Software Engineer with 3 years experience in the US.",
     "Staff Software Engineer", "United States", "greenhouse", FAIL),

    ("GH: blocks 'Senior Software Engineer'",
     "Senior Software Engineer role, 4+ years required.",
     "Senior Software Engineer", "New York, NY", "greenhouse", FAIL),

    ("GH: blocks 'VP of Engineering'",
     "VP of Engineering at our SF office.",
     "VP of Engineering", "San Francisco, CA", "greenhouse", FAIL),

    ("GH: blocks 'Engineering Manager'",
     "Engineering Manager to lead a team of 5.",
     "Engineering Manager", "Remote", "greenhouse", FAIL),

    ("GH: does NOT block 'Machine Learning Engineer'",
     "Machine Learning Engineer, 2 years exp, remote US.",
     "Machine Learning Engineer", "Remote", "greenhouse", PASS),

    ("GH: does NOT block 'Platform Engineer'",
     "Platform Engineer, AWS, Python, backend. Based in US.",
     "Platform Engineer", "Austin, TX, United States", "greenhouse", PASS),

    # 'lead' as substring in unrelated word — must NOT block
    ("GH: 'lead' in 'upload' should NOT block",
     "Build our data upload pipeline. Python backend, US remote.",
     "Data Upload Engineer", "Remote, United States", "greenhouse", PASS),

    # ── YOE filter ───────────────────────────────────────────────────────────
    ("GH: blocks '7+ years of experience'",
     "Backend Engineer. Requires 7+ years of experience. Remote, US.",
     "Backend Engineer", "Remote", "greenhouse", FAIL),

    ("GH: allows '3+ years of experience'",
     "AI Engineer. 3+ years of experience required. Remote, US.",
     "AI Engineer", "Remote", "greenhouse", PASS),

    ("GH: allows '3-5 years'",
     "Software Engineer. 3-5 years required. US.",
     "Software Engineer", "New York, United States", "greenhouse", PASS),

    ("GH: blocks '6-8 years'",
     "Backend Engineer. 6-8 years required. US.",
     "Backend Engineer", "Remote", "greenhouse", FAIL),

    ("GH: no YOE mentioned -> let through",
     "Software Engineer. Great opportunity, Python, AWS. Remote US.",
     "Software Engineer", "Remote", "greenhouse", PASS),

    # ── US location filter (Greenhouse) ──────────────────────────────────────
    # Real failures seen in output — city+state format must be detected as US
    ("GH: allows 'San Francisco, CA | New York City, NY'",
     "Software Engineer role.", "Software Engineer",
     "San Francisco, CA | New York City, NY", "greenhouse", PASS),

    ("GH: allows 'San Francisco, CA | New York City, NY | Seattle, WA'",
     "AI Engineer role.", "AI Engineer",
     "San Francisco, CA | New York City, NY | Seattle, WA", "greenhouse", PASS),

    ("GH: allows 'Washington, DC'",
     "Backend Engineer.", "Backend Engineer",
     "Washington, DC", "greenhouse", PASS),

    ("GH: allows 'New York City, NY; San Francisco, CA'",
     "Software Engineer.", "Software Engineer",
     "New York City, NY; San Francisco, CA", "greenhouse", PASS),

    ("GH: allows 'N/A' (unspecified placeholder)",
     "Backend Engineer.", "Backend Engineer",
     "N/A", "greenhouse", PASS),

    ("GH: allows 'LOCATION' (template placeholder)",
     "Software Engineer.", "Software Engineer",
     "LOCATION", "greenhouse", PASS),

    ("GH: allows 'Seattle'",
     "Backend Engineer.", "Backend Engineer",
     "Seattle", "greenhouse", PASS),

    ("GH: allows 'New York'",
     "Software Engineer.", "Software Engineer",
     "New York", "greenhouse", PASS),

    ("GH: allows 'NYC'",
     "Backend Engineer.", "Backend Engineer",
     "NYC", "greenhouse", PASS),

    ("GH: allows 'SF'",
     "Software Engineer.", "Software Engineer",
     "SF", "greenhouse", PASS),

    ("GH: allows 'Chicago, Illinois'",
     "Backend Engineer.", "Backend Engineer",
     "Chicago, Illinois", "greenhouse", PASS),

    ("GH: allows 'Maryland; Virginia; Washington, D.C.'",
     "Software Engineer.", "Software Engineer",
     "Maryland; Virginia; Washington, D.C.", "greenhouse", PASS),

    ("GH: blocks non-US location 'London, UK'",
     "Backend Engineer role.",
     "Backend Engineer", "London, UK", "greenhouse", FAIL),

    ("GH: blocks 'Tokyo, Japan'",
     "AI Engineer.", "AI Engineer",
     "Tokyo, Japan", "greenhouse", FAIL),

    ("GH: blocks 'Bangalore, India'",
     "Software Engineer.", "Software Engineer",
     "Bangalore, India", "greenhouse", FAIL),

    ("GH: blocks 'Ontario, CAN'",
     "Backend Engineer.", "Backend Engineer",
     "Ontario, CAN", "greenhouse", FAIL),

    ("GH: allows 'Remote'",
     "Backend Engineer. Python, AWS.",
     "Backend Engineer", "Remote", "greenhouse", PASS),

    ("GH: allows 'New York, United States'",
     "Software Engineer. Python.",
     "Software Engineer", "New York, United States", "greenhouse", PASS),

    ("GH: blocks 'Toronto, Canada'",
     "Software Engineer role.",
     "Software Engineer", "Toronto, Canada", "greenhouse", FAIL),

    # ── HN hiring intent ─────────────────────────────────────────────────────
    ("HN: passes — has hiring signal + matching title",
     "Acme Corp | We are hiring a Backend Engineer | Remote, US\n"
     "We're looking for a backend engineer with Python and AWS experience. "
     "Salary $120k-$160k. Remote, United States only.",
     "Acme Corp | Hiring Backend Engineer", "", "hackernews", PASS),

    ("HN: fails — no hiring signal word",
     "Acme Corp | Backend Engineer | Remote, US\n"
     "Backend role with Python and AWS. $120k.",
     "Acme Corp | Backend Engineer", "", "hackernews", FAIL),

    ("HN: fails — has hiring signal but no matching title/skill",
     "We are hiring a Marketing Manager in the US.",
     "Marketing Manager", "", "hackernews", FAIL),

    ("HN: passes — 'seeking' + 'Software Engineer'",
     "StartupXYZ | Seeking Software Engineer | Remote\n"
     "We are seeking a software engineer. Python, LLM. US remote. $130k.",
     "StartupXYZ | Seeking SWE", "", "hackernews", PASS),

    # ── HN US location filter ─────────────────────────────────────────────────
    ("HN: fails — non-US only (Berlin, Germany)",
     "We are hiring a Software Engineer | Berlin, Germany | On-site only\n"
     "Seeking a backend engineer. Python. Based in Berlin.",
     "Hiring SWE", "", "hackernews", FAIL),

    ("HN: passes — remote with no non-US geo signal",
     "We are hiring a Backend Engineer | Remote\n"
     "Fully remote, Python, AWS, LLM. $140k.",
     "Hiring Backend Engineer", "", "hackernews", PASS),

    # ── HN title blocklist checks full text ──────────────────────────────────
    ("HN: blocks 'Staff Engineer' mentioned in post body",
     "We are hiring a Staff Engineer | Remote, US\n"
     "Looking for a staff engineer with 8+ years Python.",
     "Acme | Hiring", "", "hackernews", FAIL),
]


def run():
    passed = 0
    failed = 0
    for desc, text, title, location, source, expected in CASES:
        result = passes_filters(text, title, location, CONFIG, source=source, debug=False)
        ok = result == expected
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
            # Re-run with debug to show skip reason
            print(f"\n{'─'*60}")
            print(f"[{status}] {desc}")
            print(f"  title={title!r}  location={location!r}  source={source}")
            print(f"  expected={'KEEP' if expected else 'SKIP'}, got={'KEEP' if result else 'SKIP'}")
            passes_filters(text, title, location, CONFIG, source=source, debug=True)
        else:
            passed += 1
            print(f"[{status}] {desc}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(CASES)} cases")

    # Also show YOE extraction sanity check
    print("\n-- YOE extraction --")
    for text, expected in [
        ("requires 7+ years of experience", [7]),
        ("3-5 years required", [5]),
        ("minimum 4 years", [4]),
        ("at least 2 years of experience", [2]),  # deduplicated (two patterns overlap)
        ("2 years of experience in Python", [2]),
        ("no experience mentioned here", []),
    ]:
        result = _extract_yoe_numbers(text)
        ok = result == expected
        print(f"  {'OK' if ok else 'WRONG'}: {text!r} -> {result} (expected {expected})")

    return failed


if __name__ == "__main__":
    sys.exit(run())
