"""onboarding.py — Beacon-style multi-step profile creation flow.

Exported entry points:
    render_onboarding()   — main 5-step UI
    sanitize_slug()       — used by dashboard.py for the quick-create dialog
    generate_config()     — used by tests + create_profile()
    create_profile()      — write profile dir, config.yaml, .env key, init DB

Layout follows Beacon-final.html: 240px left rail (brand + step indicator +
"Skip setup, see demo →" link) and a right main panel that swaps between the
five Beacon steps (Welcome / Resume / Targets / Sources / Cadence). Profile
name + employment_type are collected on Welcome; LLM provider selection is
folded into Sources alongside the source picker.
"""

import html
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from config import apply_config_defaults
from db import init_db
from ui_shell import callout

_BASE_DIR    = Path(__file__).parent
_PROFILES_DIR = _BASE_DIR / "profiles"


# ── Provider metadata (Step 4 — Sources / Model) ─────────────────────────────

_PROVIDERS: List[Dict[str, Any]] = [
    {
        "label":     "Groq — Llama 4 Scout  (recommended)",
        "provider":  "groq",
        "model_key": "groq",
        "model_id":  "meta-llama/llama-4-scout-17b-16e-instruct",
        "rpm": 30, "rpd": 1_000,
        "stars": "⭐⭐⭐⭐",
        "env_var": "GROQ_API_KEY",
        "paid": False,
    },
    {
        "label":     "Groq — Llama 3.3 70B",
        "provider":  "groq",
        "model_key": "groq_balanced",
        "model_id":  "llama-3.3-70b-versatile",
        "rpm": 30, "rpd": 1_000,
        "stars": "⭐⭐⭐⭐",
        "env_var": "GROQ_API_KEY",
        "paid": False,
    },
    {
        "label":     "Groq — Llama 3.1 8B  (fastest / most requests)",
        "provider":  "groq",
        "model_key": "groq_testing",
        "model_id":  "llama-3.1-8b-instant",
        "rpm": 30, "rpd": 14_400,
        "stars": "⭐⭐⭐",
        "env_var": "GROQ_API_KEY",
        "paid": False,
    },
    {
        "label":     "Gemini — 2.5 Flash",
        "provider":  "gemini",
        "model_key": "gemini",
        "model_id":  "gemini-2.5-flash",
        "rpm": 10, "rpd": 250,
        "stars": "⭐⭐⭐⭐⭐",
        "env_var": "GEMINI_API_KEY",
        "paid": False,
    },
    {
        "label":     "Gemini — 2.5 Flash-Lite",
        "provider":  "gemini",
        "model_key": "gemini_lite",
        "model_id":  "gemini-2.5-flash-lite-preview-06-17",
        "rpm": 15, "rpd": 1_000,
        "stars": "⭐⭐⭐⭐",
        "env_var": "GEMINI_API_KEY",
        "paid": False,
    },
    {
        "label":     "Anthropic — Claude Sonnet  (paid, best accuracy)",
        "provider":  "anthropic",
        "model_key": "anthropic",
        "model_id":  "claude-sonnet-4-20250514",
        "rpm": 50, "rpd": None,
        "stars": "⭐⭐⭐⭐⭐",
        "env_var": "ANTHROPIC_API_KEY",
        "paid": True,
    },
]

_PROVIDER_LABELS = [p["label"] for p in _PROVIDERS]


# ── Defaults ─────────────────────────────────────────────────────────────────

_DEFAULT_FT_TITLES = [
    "Backend", "Platform", "AI/ML", "Infra", "Distributed",
    "Frontend", "Full-stack", "Data eng", "DevOps", "Security", "Mobile",
]

_DEFAULT_INTERN_TITLES = [
    "SWE Intern", "Backend Intern", "ML Intern", "AI Intern",
    "Data Intern", "Platform Intern", "Infra Intern", "Frontend Intern",
]

_DEFAULT_SKILLS = ["Python", "ML Infrastructure", "LLM", "backend", "AWS"]

_DEFAULT_FT_HARD_NO = [
    "security clearance required", "5+ years of experience",
    "internship", "intern",
]

_DEFAULT_INTERN_HARD_NO = [
    "security clearance required", "full-time only", "10+ years",
    "senior", "staff", "principal", "director", "manager", "lead", "executive",
]

_LOCATION_OPTIONS = [
    "Remote (US)", "San Francisco", "NYC", "Seattle",
    "Austin", "Boston", "Los Angeles", "Chicago", "Denver", "Washington, DC",
]

# Map Beacon's friendly location labels back to the structured strings the
# config and scoring code expect.
_LOCATION_VALUE_MAP: Dict[str, str] = {
    "Remote (US)":     "Remote",
    "San Francisco":   "San Francisco, CA",
    "NYC":             "New York, NY",
    "Seattle":         "Seattle, WA",
    "Austin":          "Austin, TX",
    "Boston":          "Boston, MA",
    "Los Angeles":     "Los Angeles, CA",
    "Chicago":         "Chicago, IL",
    "Denver":          "Denver, CO",
    "Washington, DC":  "Washington, DC",
}

_DEFAULT_LOCATIONS = ["Remote (US)", "San Francisco", "NYC", "Seattle"]
_DEFAULT_ROLE_TYPES_FT = ["Backend", "Platform", "AI/ML", "Infra", "Distributed"]
_DEFAULT_ROLE_TYPES_INTERN = ["SWE Intern", "Backend Intern", "ML Intern", "AI Intern"]


# ── Source choices (Step 4) ──────────────────────────────────────────────────

_SOURCES: List[Dict[str, str]] = [
    {"id": "greenhouse", "name": "Greenhouse",                   "desc": "Most YC + growth-stage companies",          "meta": "~2,400 companies"},
    {"id": "lever",      "name": "Lever",                        "desc": "Common at later-stage startups",            "meta": "~1,100 companies"},
    {"id": "ashby",      "name": "Ashby",                        "desc": "Newer ATS, growing fast",                   "meta": "~350 companies"},
    {"id": "workable",   "name": "Workable",                     "desc": "Smaller companies, EU-heavy",               "meta": "~600 companies"},
    {"id": "hn",         "name": "Hacker News 'Who's Hiring'",   "desc": "Monthly thread, parsed",                    "meta": "monthly"},
    {"id": "himalayas",  "name": "Himalayas (remote)",           "desc": "Remote-only board",                         "meta": "~800 companies"},
]

_DEFAULT_ENABLED_SOURCES = {"greenhouse", "lever"}

_DEFAULT_GH_COMPANIES = ["anthropic", "stripe", "figma", "databricks"]
_DEFAULT_LV_COMPANIES = [
    "stripe", "linear", "vercel", "notion", "retool",
    "figma", "rippling", "brex", "ramp", "scale",
]
_DEFAULT_ASHBY_COMPANIES = [
    "linear", "vercel", "retool", "notion", "rippling", "brex", "ramp",
    "scale-ai", "weights-biases", "cohere", "mistral", "perplexity",
    "cursor", "replit", "harvey", "glean", "vanta", "drata", "merge", "finch",
]


# ── Cadence options (Step 5) ─────────────────────────────────────────────────

_CADENCE_OPTIONS_FT: List[Dict[str, str]] = [
    {"id": "daily_morning", "name": "Daily morning", "desc": "Run at 09:00 in your timezone, email digest at 09:05", "meta": "recommended"},
    {"id": "twice_daily",   "name": "Twice a day",   "desc": "09:00 and 16:00 — for active searches",                 "meta": ""},
    {"id": "hourly",        "name": "Hourly",        "desc": "Aggressive. ~24 runs / day, costs ~$2.10/wk",           "meta": ""},
    {"id": "manual",        "name": "Manual only",   "desc": "I run only when you click 'Run pipeline'",              "meta": ""},
]

_CADENCE_OPTIONS_INTERN: List[Dict[str, str]] = [
    {"id": "daily_morning", "name": "Daily morning", "desc": "Run at 09:00 — fits the campus career-site refresh cadence",   "meta": "recommended"},
    {"id": "twice_daily",   "name": "Twice a day",   "desc": "09:00 and 16:00 — for active recruiting season",                "meta": ""},
    {"id": "hourly",        "name": "Hourly",        "desc": "Aggressive. ~24 runs / day, usually overkill for class hours",  "meta": ""},
    {"id": "manual",        "name": "Manual only",   "desc": "I run only when you click 'Run pipeline'",                      "meta": ""},
]


# ── Step layout (left rail) ──────────────────────────────────────────────────

_STEPS: List[tuple[str, str]] = [
    ("Welcome", "What Beacon does for you"),
    ("Resume",  "Drop in your CV"),
    ("Targets", "Roles, geos, comp"),
    ("Sources", "Pick job boards"),
    ("Cadence", "How often to run"),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def sanitize_slug(name: str) -> str:
    """Convert a name to a safe profile slug: lowercase, underscores, no specials."""
    slug = name.lower().strip()
    slug = slug.replace(" ", "_")
    slug = re.sub(r"[^\w]", "", slug)
    return slug


def _lines_to_list(text: str) -> List[str]:
    return [t.strip() for t in (text or "").strip().splitlines() if t.strip()]


def _parse_min_comp(text: str) -> int:
    """Strip non-digits from a compensation string. Empty / unparsable → 0."""
    if not text:
        return 0
    cleaned = re.sub(r"[^0-9]", "", str(text))
    return int(cleaned) if cleaned else 0


def _normalize_intern_pay_preference(data: Dict[str, Any]) -> str:
    preference = str(data.get("intern_pay_preference", "")).strip().lower()
    if preference in {"paid_only", "unpaid_ok", "no_preference"}:
        return preference
    if data.get("stipend_expectation"):
        return "paid_only"
    return "no_preference"


def _profile_dir(slug: str) -> Path:
    return _PROFILES_DIR / slug


# ── Config generator ─────────────────────────────────────────────────────────

def generate_config(data: Dict[str, Any]) -> dict:
    """Build a full config.yaml dict from collected onboarding data."""
    is_intern = data.get("job_type") == "internship"

    profile_section: Dict[str, Any] = {
        "name":      data["name"],
        "bio":       data.get("bio", ""),
        "job_type":  data.get("job_type", "fulltime"),
        "resume":    None,
        "resume_file": None,
    }
    if data.get("resume_type") == "pdf":
        profile_section["resume_file"] = "resume.pdf"
    else:
        profile_section["resume"] = data.get("resume_text", "")

    if is_intern:
        profile_section.update({
            "target_season":   f"{data.get('target_season', '')} {data.get('target_year', '')}".strip(),
            "school":          data.get("school", ""),
            "major":           data.get("major", ""),
            "gpa":             data.get("gpa", ""),
            "graduation_year": data.get("graduation_year", ""),
        })

    title_blocklist = [
        "Senior", "Staff", "Principal", "VP", "Director",
        "Head of", "Manager", "Lead", "Executive",
    ]
    default_hard_no = _DEFAULT_INTERN_HARD_NO if is_intern else _DEFAULT_FT_HARD_NO
    if is_intern:
        title_blocklist.append("Full-time only")
    else:
        title_blocklist.append("Intern")

    preferences: Dict[str, Any] = {
        "titles":          data.get("titles", _DEFAULT_INTERN_TITLES if is_intern else _DEFAULT_FT_TITLES),
        "desired_skills":  data.get("desired_skills", _DEFAULT_SKILLS),
        "hard_no_keywords": data.get("hard_no_keywords", default_hard_no),
        "location": {
            "remote_ok":           data.get("remote_ok", True),
            "preferred_locations": data.get("preferred_locations", []),
        },
        "filters": {
            "countries_allowed":    ["United States", "US", "USA", "Remote"],
            "min_yoe":              0,
            "max_yoe":              1 if is_intern else 5,
            "max_job_age_days":     14 if is_intern else 30,
            "require_degree_filter": True,
            "title_blocklist":      title_blocklist,
        },
    }

    if is_intern:
        pay_preference = _normalize_intern_pay_preference(data)
        preferences["compensation"] = {"intern_pay_preference": pay_preference}
        stipend = data.get("stipend_expectation")
        if pay_preference == "paid_only" and stipend not in (None, "", 0):
            preferences["compensation"]["monthly_stipend"] = int(stipend)
    else:
        preferences["yoe"] = data.get("yoe", 0)
        preferences["compensation"] = {"min_salary": data.get("min_salary", 100_000)}

    # LLM section — keep all providers, set the active one
    provider  = data.get("provider", "groq")
    model_key = data.get("model_key", "groq")
    model_id  = data.get("model_id", "meta-llama/llama-4-scout-17b-16e-instruct")

    llm = {
        "provider":    provider,
        "temperature": 0,
        "model": {
            "anthropic":     "claude-sonnet-4-20250514",
            "gemini":        "gemini-2.5-flash",
            "gemini_lite":   "gemini-2.5-flash-lite-preview-06-17",
            "groq":          "meta-llama/llama-4-scout-17b-16e-instruct",
            "groq_balanced": "llama-3.3-70b-versatile",
            "groq_testing":  "llama-3.1-8b-instant",
            "openai":        "gpt-4o-mini",
        },
        "rate_limits": {
            "groq":          {"max_rpm": 28, "max_tpm": 28_000,  "max_rpd": 1_000},
            "groq_balanced": {"max_rpm": 28, "max_tpm": 11_000,  "max_rpd": 1_000},
            "groq_testing":  {"max_rpm": 28, "max_tpm": 5_500,   "max_rpd": 14_400},
            "gemini":        {"max_rpm": 9,  "max_tpm": 240_000, "max_rpd": 250},
            "gemini_lite":   {"max_rpm": 14, "max_tpm": 240_000, "max_rpd": 1_000},
            "anthropic":     {"max_rpm": 50, "max_tpm": 9_000_000},
            "openai":        {"max_rpm": 50, "max_tpm": 9_000_000},
        },
    }
    llm["model"][model_key] = model_id
    if model_key != provider:
        llm["model"][provider] = model_id

    # Sources — driven by Step 4 selection
    enabled_sources: set = set(data.get("enabled_sources") or _DEFAULT_ENABLED_SOURCES)
    sources: Dict[str, Any] = {
        "greenhouse": {
            "enabled": "greenhouse" in enabled_sources,
            "companies": data.get("gh_companies", _DEFAULT_GH_COMPANIES),
        },
        "lever": {
            "enabled": "lever" in enabled_sources,
            "companies": data.get("lv_companies", _DEFAULT_LV_COMPANIES),
        },
        "ashby": {
            "enabled": "ashby" in enabled_sources,
            "companies": data.get("ashby_companies", _DEFAULT_ASHBY_COMPANIES),
        },
        "workable": {
            "enabled": "workable" in enabled_sources,
            "companies": data.get("wl_companies", []),
        },
        "himalayas": {
            "enabled": "himalayas" in enabled_sources,
        },
    }
    if "hn" in enabled_sources:
        sources["hn"] = {"enabled": True}

    scoring = {
        "min_display_score": 60,
        "weights": {
            "role_fit":     0.30,
            "stack_match":  0.25,
            "seniority":    0.20,
            "location":     0.10,
            "growth":       0.10,
            "compensation": 0.05,
        },
    }

    schedule = {"cadence": data.get("cadence", "manual")}

    return apply_config_defaults({
        "llm":         llm,
        "profile":     profile_section,
        "preferences": preferences,
        "sources":     sources,
        "scoring":     scoring,
        "schedule":    schedule,
    })


def create_profile(data: Dict[str, Any]) -> None:
    """Write config.yaml, copy resume PDF, save API key, and initialize the DB."""
    import yaml

    slug = data["profile_slug"]
    profile_dir = _profile_dir(slug)
    profile_dir.mkdir(parents=True, exist_ok=True)

    if data.get("resume_type") == "pdf" and data.get("resume_pdf_bytes"):
        (profile_dir / "resume.pdf").write_bytes(data["resume_pdf_bytes"])

    config = generate_config(data)
    with open(profile_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    api_key = (data.get("api_key") or "").strip()
    if api_key:
        env_var = data.get("env_var") or ""
        if env_var:
            env_path = _BASE_DIR / ".env"
            _upsert_env_key(env_path, f"{env_var}_{slug.upper()}", api_key)

    init_db(profile=slug)


def _upsert_env_key(env_path: Path, key: str, value: str) -> None:
    """Insert or update a key=value line in a .env file without touching other lines."""
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    prefix = f"{key}="
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            lines[i] = f"{key}={value}"
            updated = True
            break
    if not updated:
        lines.append(f"{key}={value}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Markup helpers ───────────────────────────────────────────────────────────

def _eyebrow(text: str) -> None:
    st.markdown(
        f"<div class='onb-eyebrow'>{html.escape(text)}</div>",
        unsafe_allow_html=True,
    )


def _h2(text: str) -> None:
    st.markdown(
        f"<h2 class='onb-h2'>{html.escape(text)}</h2>",
        unsafe_allow_html=True,
    )


def _hint(text: str) -> None:
    st.markdown(
        f"<p class='onb-hint'>{html.escape(text)}</p>",
        unsafe_allow_html=True,
    )


def _render_left_rail(step_idx: int) -> bool:
    """Render brand + step indicator + "Skip setup, see demo →".
    Returns True if the user clicked "Skip setup".
    """
    st.markdown(
        (
            "<div style='display:flex;align-items:center;gap:10px;padding:6px 4px 14px;"
            "border-bottom:1px solid var(--line)'>"
            "<div style='width:28px;height:28px;border-radius:8px;background:var(--ink);"
            "color:var(--accent-ink);display:grid;place-items:center;font-family:var(--font-display);"
            "font-weight:700;font-size:14px'>B</div>"
            "<div>"
            "<div style='font-family:var(--font-display);font-weight:600;font-size:14px'>Beacon</div>"
            "<div class='onb-eyebrow'>job-search agent</div>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    steps_html = ["<div class='onb-steps'>"]
    for index, (name, desc) in enumerate(_STEPS):
        state = "active" if index == step_idx else ("done" if index < step_idx else "")
        marker = "✓" if index < step_idx else str(index + 1)
        steps_html.append(
            (
                f"<div class='onb-step {state}'>"
                f"<span class='num'>{html.escape(marker)}</span>"
                f"<div><div class='nm'>{html.escape(name)}</div>"
                f"<div class='ds'>{html.escape(desc)}</div></div>"
                "</div>"
            )
        )
    steps_html.append("</div>")
    st.markdown("".join(steps_html), unsafe_allow_html=True)

    st.markdown(
        "<div class='onb-side-takes'>Takes about 2 minutes.</div>",
        unsafe_allow_html=True,
    )
    return st.button(
        "Skip setup, see demo →",
        key="onb_skip_demo",
        use_container_width=True,
        help="Cancel onboarding and return to the dashboard.",
    )


def _render_footer(
    step_idx: int,
    *,
    next_label: str,
    can_advance: bool,
    next_key: str,
) -> str | None:
    """Render Back / page indicator / Continue or Open Beacon. Returns clicked id."""
    cols = st.columns([1, 2, 1], gap="small")
    clicked: str | None = None
    with cols[0]:
        if step_idx > 0:
            if st.button("← Back", key=f"onb_back_{step_idx}", use_container_width=True):
                clicked = "back"
        else:
            st.empty()
    with cols[1]:
        st.markdown(
            f"<div class='onb-foot-pg'>{step_idx + 1} / {len(_STEPS)}</div>",
            unsafe_allow_html=True,
        )
    with cols[2]:
        if st.button(
            next_label,
            key=next_key,
            type="primary",
            use_container_width=True,
            disabled=not can_advance,
        ):
            clicked = "next"
    return clicked


# ── Step 1 — Welcome ─────────────────────────────────────────────────────────

_VALUE_PROPS = [
    ("Score, don't spam",     "Each posting gets a transparent fit score with reasoning. No black box."),
    ("Move at your pace",     "Run on a schedule or on demand. Pause anytime."),
    ("You stay in control",   "I draft, you decide. Nothing leaves your inbox without you."),
    ("Local & private",       "Your resume + preferences live on your machine. Models run in your account."),
]


def _step_welcome() -> None:
    data = st.session_state.onboarding_data
    _eyebrow(f"Step 1 of {len(_STEPS)}")
    _h2("Hi. I'll be your job-search agent.")
    _hint(
        "Every morning I'll scan the job boards you trust, score new postings against your "
        "skills and preferences, and surface the few you should actually look at. You stay "
        "in the driver's seat — I just handle the noise."
    )

    # Value-prop cards in a 2-col grid
    rows = [_VALUE_PROPS[i:i + 2] for i in range(0, len(_VALUE_PROPS), 2)]
    for row in rows:
        cols = st.columns(2, gap="medium")
        for col, (title, desc) in zip(cols, row):
            with col:
                st.markdown(
                    (
                        "<div class='onb-value-card'>"
                        f"<div class='t'>{html.escape(title)}</div>"
                        f"<div class='d'>{html.escape(desc)}</div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='onb-eyebrow'>Profile basics</div>",
        unsafe_allow_html=True,
    )
    name = st.text_input(
        "Your name",
        value=data.get("name", ""),
        placeholder="Manav Shah",
        key="onb_welcome_name",
    )

    cur_type = data.get("job_type", "fulltime")
    type_choice = st.radio(
        "Search mode",
        ["Full-time", "Internship"],
        index=1 if cur_type == "internship" else 0,
        horizontal=True,
        key="onb_welcome_type",
    )

    clicked = _render_footer(
        step_idx=0,
        next_label="Continue →",
        can_advance=bool(name.strip()),
        next_key="onb_welcome_next",
    )
    if clicked == "next":
        if not name.strip():
            callout("error", "Profile name required", "Add a profile name before continuing.")
            return
        clean = name.strip()
        data["name"] = clean
        # If the user did not pre-fill a slug (e.g. via dashboard quick-create),
        # derive one from the name. The Cadence step will let them confirm.
        data.setdefault("profile_slug", sanitize_slug(clean))
        data["job_type"] = "internship" if type_choice == "Internship" else "fulltime"
        if data["job_type"] == "internship":
            data.setdefault("target_season", "Summer")
            data.setdefault("target_year", "2026")
            data.setdefault("graduation_year", "2027")
        st.session_state.onboarding_step = 2
        st.rerun()


# ── Step 2 — Resume ──────────────────────────────────────────────────────────

def _step_resume() -> None:
    data = st.session_state.onboarding_data
    _eyebrow(f"Step 2 of {len(_STEPS)}")
    _h2("Drop in your resume.")
    _hint(
        "I'll parse roles, skills, and seniority from the PDF. You can edit anything "
        "I get wrong on the next screen."
    )

    has_pdf = bool(data.get("resume_pdf_bytes"))
    pdf_name = data.get("resume_pdf_name", "resume.pdf")

    st.markdown(
        (
            "<div class='onb-dropzone'>"
            "<div class='lbl'>Drop file</div>"
            f"<div class='h'>{html.escape('Resume on file: ' + pdf_name) if has_pdf else 'Drag your PDF or DOCX here'}</div>"
            "<div class='s'>or use the uploader below · max 10 MB</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    if has_pdf:
        callout("success", "Resume uploaded", pdf_name)
        if st.button("Replace PDF", key="onb_resume_replace"):
            data.pop("resume_pdf_bytes", None)
            data.pop("resume_pdf_name", None)
            data.pop("resume_type", None)
            st.rerun()
    else:
        uploaded = st.file_uploader(
            "Upload your resume",
            type=["pdf", "docx"],
            key="onb_resume_uploader",
            help="PDF preferred. DOCX is parsed as plain text.",
        )
        if uploaded is not None:
            data["resume_pdf_bytes"] = uploaded.getvalue()
            data["resume_pdf_name"]  = uploaded.name
            data["resume_type"]      = "pdf"
            st.rerun()

    st.markdown(
        "<div class='onb-eyebrow' style='margin-top:14px'>Or paste a LinkedIn URL — I'll extract from your public profile</div>",
        unsafe_allow_html=True,
    )
    linkedin = st.text_input(
        "LinkedIn URL",
        value=data.get("linkedin_url", ""),
        placeholder="linkedin.com/in/your-handle",
        key="onb_resume_linkedin",
        label_visibility="collapsed",
    )

    # When the user hasn't uploaded a PDF, fall through to the existing
    # paste-text path so the rest of the pipeline still has something to work
    # with. This keeps the resume pipeline intact while matching the Beacon
    # mock's drop-zone-first UX.
    if not has_pdf:
        with st.expander("Or paste resume text", expanded=False):
            resume_text = st.text_area(
                "Resume text",
                value=data.get("resume_text", ""),
                height=200,
                key="onb_resume_text",
                help="Used if you skip the upload. Plain text only.",
            )
            data["resume_text"] = resume_text
            if resume_text.strip() and not data.get("resume_pdf_bytes"):
                data["resume_type"] = "text"

    can_advance = bool(data.get("resume_pdf_bytes")) or bool((data.get("resume_text") or "").strip())

    clicked = _render_footer(
        step_idx=1,
        next_label="Continue →",
        can_advance=can_advance,
        next_key="onb_resume_next",
    )
    if clicked == "back":
        st.session_state.onboarding_step = 1
        st.rerun()
    if clicked == "next":
        if not can_advance:
            callout("error", "Resume required", "Upload a PDF or paste resume text before continuing.")
            return
        if linkedin.strip():
            data["linkedin_url"] = linkedin.strip()
        st.session_state.onboarding_step = 3
        st.rerun()


# ── Step 3 — Targets ─────────────────────────────────────────────────────────

def _step_targets() -> None:
    data = st.session_state.onboarding_data
    is_intern = data.get("job_type") == "internship"
    _eyebrow(f"Step 3 of {len(_STEPS)}")
    _h2("What are you looking for?")
    _hint("Pick role types and geos. I weight your top 3 the heaviest.")

    # Role types — multi-select via st.pills (clean Beacon-style pill grid)
    role_options = _DEFAULT_INTERN_TITLES if is_intern else _DEFAULT_FT_TITLES
    role_default = data.get("role_pills") or (
        _DEFAULT_ROLE_TYPES_INTERN if is_intern else _DEFAULT_ROLE_TYPES_FT
    )
    role_default = [r for r in role_default if r in role_options]
    pills_callable = getattr(st, "pills", None)
    if callable(pills_callable):
        selected_roles = pills_callable(
            "Role types",
            role_options,
            default=role_default,
            selection_mode="multi",
            key="onb_targets_roles",
        ) or []
    else:
        selected_roles = st.multiselect(
            "Role types",
            options=role_options,
            default=role_default,
            key="onb_targets_roles",
        )

    cols = st.columns(2, gap="medium")
    with cols[0]:
        location_options = _LOCATION_OPTIONS
        location_default = [
            loc for loc in (data.get("location_pills") or _DEFAULT_LOCATIONS)
            if loc in location_options
        ]
        if callable(pills_callable):
            selected_locations = pills_callable(
                "Locations",
                location_options,
                default=location_default,
                selection_mode="multi",
                key="onb_targets_locs",
            ) or []
        else:
            selected_locations = st.multiselect(
                "Locations",
                options=location_options,
                default=location_default,
                key="onb_targets_locs",
            )

    with cols[1]:
        if is_intern:
            stipend_value = data.get("stipend_expectation", "")
            stipend_text = st.text_input(
                "Target monthly stipend (optional)",
                value=str(stipend_value) if stipend_value else "",
                placeholder="$4,500 / mo",
                key="onb_targets_stipend",
                help="Leave blank if you only care that the role is paid.",
            )
        else:
            min_comp_default = data.get("min_salary", 160_000)
            min_comp_text = st.text_input(
                "Min base comp",
                value=f"${int(min_comp_default):,}" if min_comp_default else "$160,000",
                key="onb_targets_min_comp",
            )

    hard_no_default = "\n".join(
        data.get("hard_no_keywords",
                 _DEFAULT_INTERN_HARD_NO if is_intern else _DEFAULT_FT_HARD_NO)
    )
    hard_no_text = st.text_area(
        "Hard nos (free text — one per line)",
        value=hard_no_default,
        height=92,
        key="onb_targets_hardno",
        help="Phrases like 'security clearance required' or '5+ years of experience' that should auto-skip a role.",
    )

    # Internship-specific extras, surfaced only on this step.
    if is_intern:
        st.markdown(
            "<div class='onb-eyebrow' style='margin-top:8px'>Internship details</div>",
            unsafe_allow_html=True,
        )
        sub_cols = st.columns(2, gap="medium")
        with sub_cols[0]:
            season_opts = ["Summer", "Fall", "Spring"]
            season = st.selectbox(
                "Target season",
                season_opts,
                index=season_opts.index(data.get("target_season", "Summer")),
                key="onb_targets_season",
            )
        with sub_cols[1]:
            year_opts = ["2025", "2026", "2027"]
            year_default = str(data.get("target_year", "2026"))
            year = st.selectbox(
                "Year",
                year_opts,
                index=year_opts.index(year_default) if year_default in year_opts else 1,
                key="onb_targets_year",
            )
        school = st.text_input(
            "School",
            value=data.get("school", ""),
            placeholder="Georgia Tech",
            key="onb_targets_school",
        )
        sub_cols2 = st.columns([2, 1.4, 1], gap="medium")
        with sub_cols2[0]:
            major = st.text_input(
                "Major",
                value=data.get("major", ""),
                placeholder="Computer Science",
                key="onb_targets_major",
            )
        with sub_cols2[1]:
            grad_opts = ["2025", "2026", "2027", "2028", "2029"]
            grad_default = str(data.get("graduation_year", "2027"))
            grad_year = st.selectbox(
                "Graduation year",
                grad_opts,
                index=grad_opts.index(grad_default) if grad_default in grad_opts else 2,
                key="onb_targets_grad",
            )
        with sub_cols2[2]:
            gpa = st.text_input(
                "GPA",
                value=data.get("gpa", ""),
                placeholder="Optional",
                key="onb_targets_gpa",
            )

    clicked = _render_footer(
        step_idx=2,
        next_label="Continue →",
        can_advance=True,
        next_key="onb_targets_next",
    )
    if clicked == "back":
        st.session_state.onboarding_step = 2
        st.rerun()
    if clicked == "next":
        # Persist pill selections separately so they survive back-navigation.
        data["role_pills"] = list(selected_roles)
        data["location_pills"] = list(selected_locations)

        # Map Beacon pill labels to the shapes the rest of the pipeline expects.
        full_titles_pool = _DEFAULT_INTERN_TITLES if is_intern else _DEFAULT_FT_TITLES
        data["titles"] = list(selected_roles) if selected_roles else full_titles_pool
        data["preferred_locations"] = [
            _LOCATION_VALUE_MAP.get(loc, loc) for loc in selected_locations
        ]
        data["remote_ok"] = any(
            "Remote" in _LOCATION_VALUE_MAP.get(loc, loc)
            for loc in selected_locations
        ) or not selected_locations
        data["hard_no_keywords"] = _lines_to_list(hard_no_text)

        if is_intern:
            stipend_int = _parse_min_comp(stipend_text) if 'stipend_text' in locals() else 0
            data["intern_pay_preference"] = "paid_only" if stipend_int else "no_preference"
            if stipend_int:
                data["stipend_expectation"] = stipend_int
            else:
                data.pop("stipend_expectation", None)
            data["target_season"]   = season
            data["target_year"]     = year
            data["school"]          = school.strip()
            data["major"]           = major.strip()
            data["graduation_year"] = grad_year
            data["gpa"]             = gpa.strip()
        else:
            data["min_salary"] = _parse_min_comp(min_comp_text) or 100_000
            data["yoe"] = int(data.get("yoe", 0))

        st.session_state.onboarding_step = 4
        st.rerun()


# ── Step 4 — Sources (+ Model) ───────────────────────────────────────────────

def _step_sources() -> None:
    data = st.session_state.onboarding_data
    _eyebrow(f"Step 4 of {len(_STEPS)}")
    _h2("Where should I look?")
    _hint("I work best with company career pages on Greenhouse, Lever, and Ashby. You can add more later.")

    # Seed each toggle's session state from `data["enabled_sources"]` on first
    # render. Streamlit takes over after that, so reading the session state is
    # what tells us the user's current selection — including clicks that just
    # happened on this rerun.
    seed = set(data.get("enabled_sources") or _DEFAULT_ENABLED_SOURCES)
    for source in _SOURCES:
        key = f"onb_source_{source['id']}"
        if key not in st.session_state:
            st.session_state[key] = source["id"] in seed
    enabled = {
        s["id"] for s in _SOURCES
        if st.session_state.get(f"onb_source_{s['id']}", False)
    }

    # Render source picker as 2-col grid of choice cards. Each card has a
    # st.toggle below it for the on/off state — Streamlit can't combine the
    # visual card and the click handler in a single widget, so the card markup
    # provides the styling while the toggle provides the state. We use the
    # already-computed `enabled` set so the card "on" highlight matches the
    # toggle without a one-render lag.
    rows = [_SOURCES[i:i + 2] for i in range(0, len(_SOURCES), 2)]
    for row in rows:
        cols = st.columns(2, gap="medium")
        for col, source in zip(cols, row):
            with col:
                is_on = source["id"] in enabled
                st.markdown(
                    (
                        f"<div class='choice {'on' if is_on else ''}'>"
                        "<div class='top'><div>"
                        f"<h4>{html.escape(source['name'])}</h4>"
                        f"<p>{html.escape(source['desc'])}</p>"
                        "</div></div>"
                        f"<div class='meta' style='margin-top:10px'>{html.escape(source['meta'])}</div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
                st.toggle(
                    f"Enable {source['name']}",
                    key=f"onb_source_{source['id']}",
                    label_visibility="collapsed",
                )

    # Re-read after rendering — the toggle widgets may have updated state in
    # this same render; storing the latest snapshot makes Continue robust.
    data["enabled_sources"] = sorted(
        s["id"] for s in _SOURCES
        if st.session_state.get(f"onb_source_{s['id']}", False)
    )

    # ── Model selection (folded in here per spec) ─────────────────────────
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='onb-eyebrow'>Scoring model</div>",
        unsafe_allow_html=True,
    )
    _hint(
        "RPM = requests per minute. RPD = requests per day. Higher limits make long "
        "scoring runs smoother. You can switch later in Settings."
    )

    current_label = data.get("provider_label", _PROVIDER_LABELS[0])
    current_idx = _PROVIDER_LABELS.index(current_label) if current_label in _PROVIDER_LABELS else 0
    selected_label = st.radio(
        "Provider",
        _PROVIDER_LABELS,
        index=current_idx,
        key="onb_sources_provider",
    )
    selected = next(p for p in _PROVIDERS if p["label"] == selected_label)

    rpd_label = "Unlimited" if selected["rpd"] is None else f"{selected['rpd']:,} / day"
    st.caption(
        f"{selected['stars']}{' (paid)' if selected['paid'] else ''} · "
        f"{selected['rpm']} RPM · {rpd_label}"
    )
    if selected["paid"]:
        callout("warning", "Paid provider", "Anthropic Claude Sonnet bills per token. Verify the API key before large runs.")

    api_key = st.text_input(
        selected["env_var"],
        value=data.get("api_key", "") if data.get("provider") == selected["provider"] else "",
        type="password",
        key="onb_sources_api_key",
        help="Stored in the repo-level .env file so future runs can reuse it.",
    )

    can_advance = bool(api_key.strip()) and bool(data["enabled_sources"])

    clicked = _render_footer(
        step_idx=3,
        next_label="Continue →",
        can_advance=can_advance,
        next_key="onb_sources_next",
    )
    if clicked == "back":
        st.session_state.onboarding_step = 3
        st.rerun()
    if clicked == "next":
        if not data["enabled_sources"]:
            callout("error", "Pick at least one source", "Enable at least one source so the pipeline has somewhere to look.")
            return
        if not api_key.strip():
            callout("error", "API key required", f"Enter {selected['env_var']} before continuing.")
            return
        data["provider_label"] = selected_label
        data["provider"]       = selected["provider"]
        data["model_key"]      = selected["model_key"]
        data["model_id"]       = selected["model_id"]
        data["env_var"]        = selected["env_var"]
        data["api_key"]        = api_key.strip()
        st.session_state.onboarding_step = 5
        st.rerun()


# ── Step 5 — Cadence (final) ─────────────────────────────────────────────────

def _step_cadence() -> None:
    data = st.session_state.onboarding_data
    is_intern = data.get("job_type") == "internship"
    _eyebrow(f"Step 5 of {len(_STEPS)}")
    _h2("How should I check in?")
    _hint("Pick a cadence. You can always run on-demand from anywhere in the app.")

    options = _CADENCE_OPTIONS_INTERN if is_intern else _CADENCE_OPTIONS_FT
    cur = data.get("cadence", "daily_morning")

    rows = [options[i:i + 2] for i in range(0, len(options), 2)]
    for row in rows:
        cols = st.columns(2, gap="medium")
        for col, opt in zip(cols, row):
            with col:
                is_on = opt["id"] == cur
                st.markdown(
                    (
                        f"<div class='choice {'on' if is_on else ''}'>"
                        "<div class='top'><div>"
                        f"<h4>{html.escape(opt['name'])}</h4>"
                        f"<p>{html.escape(opt['desc'])}</p>"
                        "</div>"
                        + (f"<span class='meta'>{html.escape(opt['meta'])}</span>" if opt["meta"] else "")
                        + "</div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Selected" if is_on else "Pick this cadence",
                    key=f"onb_cadence_{opt['id']}",
                    type="primary" if is_on else "secondary",
                    use_container_width=True,
                ):
                    data["cadence"] = opt["id"]
                    st.rerun()

    # Profile slug confirmation — gives the user a chance to fix the auto-derived
    # one before the folder is created.
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='onb-eyebrow'>Profile folder</div>",
        unsafe_allow_html=True,
    )
    suggested_slug = data.get("profile_slug") or sanitize_slug(data.get("name", "profile"))
    profile_slug = sanitize_slug(
        st.text_input(
            "Profile slug",
            value=suggested_slug,
            help="Used as the folder name under profiles/. Lowercase and underscores only.",
            key="onb_cadence_slug",
        )
    )
    if profile_slug:
        st.caption(f"Will be created at: profiles/{profile_slug}/")

    summary_bits = [
        f"<b>{html.escape(data.get('name', ''))}</b> · {'Internship' if is_intern else 'Full-time'}",
        f"{len(data.get('enabled_sources') or [])} sources",
        data.get("provider_label", ""),
        next(
            (opt["name"] for opt in options if opt["id"] == data.get("cadence")),
            "Manual only",
        ),
    ]
    st.markdown(
        (
            "<div class='onb-value-card' style='margin-top:8px'>"
            "<div style='display:flex;gap:14px;align-items:center'>"
            "<span style='font-size:22px'>✶</span>"
            "<div>"
            "<div style='font-weight:600;font-size:14px'>You're all set.</div>"
            f"<div class='d' style='margin-top:3px'>{' · '.join(b for b in summary_bits if b)}</div>"
            "</div></div></div>"
        ),
        unsafe_allow_html=True,
    )

    clicked = _render_footer(
        step_idx=4,
        next_label="Open Beacon →",
        can_advance=bool(profile_slug),
        next_key="onb_cadence_create",
    )
    if clicked == "back":
        st.session_state.onboarding_step = 4
        st.rerun()
    if clicked == "next":
        if not profile_slug:
            callout("error", "Profile slug required", "Add a profile slug so the workspace can be created.")
            return
        profile_dir = _profile_dir(profile_slug)
        if profile_dir.exists():
            callout(
                "warning",
                "Profile already exists",
                f"A profile named '{profile_slug}' already exists. Choose a different slug.",
            )
            return
        data["profile_slug"] = profile_slug
        try:
            create_profile(data)
        except Exception as e:
            callout("error", "Profile creation failed", str(e))
            return

        st.session_state.onboarding_step = 1
        st.session_state.onboarding_data = {}
        st.session_state.show_onboarding = False
        st.session_state.active_profile  = profile_slug
        st.cache_data.clear()
        st.toast(f"Profile created · {profile_slug}")
        st.rerun()


# ── Public entry point ───────────────────────────────────────────────────────

_STEP_RENDERERS = [
    _step_welcome,
    _step_resume,
    _step_targets,
    _step_sources,
    _step_cadence,
]


def render_onboarding() -> None:
    """Main entry point. Manages multi-step state via st.session_state."""
    if "onboarding_step" not in st.session_state:
        st.session_state.onboarding_step = 1
    if "onboarding_data" not in st.session_state:
        st.session_state.onboarding_data = {}

    step = max(1, min(st.session_state.onboarding_step, len(_STEPS)))
    step_idx = step - 1

    st.markdown("<div class='onb-wrap'>", unsafe_allow_html=True)
    rail_col, main_col = st.columns([0.27, 0.73], gap="large")
    with rail_col:
        skip_clicked = _render_left_rail(step_idx)
        if skip_clicked:
            st.session_state.show_onboarding = False
            st.session_state.onboarding_step = 1
            st.session_state.onboarding_data = {}
            st.rerun()
    with main_col:
        _STEP_RENDERERS[step_idx]()
    st.markdown("</div>", unsafe_allow_html=True)
