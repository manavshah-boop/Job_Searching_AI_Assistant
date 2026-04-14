"""
onboarding.py — Multi-step profile creation flow.

Exported entry point: render_onboarding()
Called by dashboard.py when the user wants to create a new profile.
"""

import re
import yaml
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from db import init_db

_BASE_DIR    = Path(__file__).parent
_PROFILES_DIR = _BASE_DIR / "profiles"

# ── Provider metadata (Step 4) ────────────────────────────────────────────────

_PROVIDERS = [
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


# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_FT_TITLES = [
    "Software Engineer", "Software Developer", "Software Development Engineer",
    "SDE", "SWE", "Backend Engineer", "AI Engineer",
]

_DEFAULT_INTERN_TITLES = [
    "Software Engineering Intern", "SWE Intern", "Software Engineer Intern",
    "ML Intern", "AI Intern", "Backend Intern", "Data Engineering Intern",
]

_DEFAULT_SKILLS = ["Python", "ML Infrastructure", "LLM", "backend", "AWS"]

_DEFAULT_FT_HARD_NO = [
    "security clearance required", "5+ years of experience",
    "internship", "intern",
]

_DEFAULT_INTERN_HARD_NO = [
    "security clearance required", "full-time only", "10+ years",
    "senior", "staff",
]

_LOCATION_OPTIONS = [
    "Remote", "San Francisco, CA", "New York, NY", "Seattle, WA",
    "Austin, TX", "Boston, MA", "Los Angeles, CA", "Chicago, IL",
    "Denver, CO", "Washington, DC",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def sanitize_slug(name: str) -> str:
    """Convert a name to a safe profile slug: lowercase, underscores, no specials."""
    slug = name.lower().strip()
    slug = slug.replace(" ", "_")
    slug = re.sub(r"[^\w]", "", slug)
    return slug


def _runtime_estimate(rpm: int, rpd: int | None, n_jobs: int = 1000) -> str:
    """Human-readable estimate: how long to score n_jobs at given rate limits."""
    if rpd is not None and rpd < n_jobs:
        return f"~{rpd:,} jobs/day max on free tier ({rpd / rpm:.0f} min/day)"
    minutes = n_jobs / rpm
    return f"~{minutes:.0f} min to score {n_jobs:,} jobs"


def _lines_to_list(text: str) -> list:
    return [t.strip() for t in text.strip().splitlines() if t.strip()]


# ── Config generator ──────────────────────────────────────────────────────────

def generate_config(data: Dict[str, Any]) -> dict:
    """Build a full config.yaml dict from collected onboarding data."""
    is_intern = data.get("job_type") == "internship"
    slug = data["profile_slug"]

    # Profile section
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

    # Preferences
    title_blocklist = [
        "Staff", "Principal", "VP", "Director",
        "Head of", "Manager", "Lead", "Executive",
    ]
    if is_intern:
        title_blocklist.append("Full-time only")
    else:
        title_blocklist.append("Intern")

    preferences: Dict[str, Any] = {
        "titles":          data.get("titles", []),
        "desired_skills":  data.get("desired_skills", []),
        "hard_no_keywords": data.get("hard_no_keywords", []),
        "location": {
            "remote_ok":           data.get("remote_ok", True),
            "preferred_locations": data.get("preferred_locations", []),
        },
        "filters": {
            "countries_allowed":    ["United States", "US", "USA", "Remote"],
            "min_yoe":              0,
            "max_yoe":              1 if is_intern else 5,
            "require_degree_filter": True,
            "title_blocklist":      title_blocklist,
        },
    }

    if is_intern:
        preferences["filters"]["max_job_age_days"] = 14
        preferences["compensation"] = {
            "monthly_stipend": data.get("stipend_expectation") or 0
        }
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
            "anthropic":   "claude-sonnet-4-20250514",
            "gemini":      "gemini-2.5-flash",
            "gemini_lite": "gemini-2.5-flash-lite-preview-06-17",
            "groq":        "meta-llama/llama-4-scout-17b-16e-instruct",
            "groq_balanced": "llama-3.3-70b-versatile",
            "groq_testing":  "llama-3.1-8b-instant",
            "openai":      "gpt-4o-mini",
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
    # Point the active provider's model key to the chosen model
    llm["model"][model_key] = model_id
    # Ensure the provider key resolves to the right model_key entry
    if model_key != provider:
        llm["model"][provider] = model_id

    # Sources
    sources = {
        "greenhouse": {
            "enabled": True,
            "companies": ["anthropic", "stripe", "figma", "databricks"],
        },
        "lever": {
            "enabled": True,
            "companies": ["stripe", "linear", "vercel", "notion", "retool",
                          "figma", "rippling", "brex", "ramp", "scale"],
        },
    }

    # Scoring
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

    return {
        "llm":         llm,
        "profile":     profile_section,
        "preferences": preferences,
        "sources":     sources,
        "scoring":     scoring,
    }


def create_profile(data: Dict[str, Any]) -> None:
    """Write config.yaml, copy resume PDF, and initialize DB for a new profile."""
    slug = data["profile_slug"]
    profile_dir = _PROFILES_DIR / slug
    profile_dir.mkdir(parents=True, exist_ok=True)

    # Save resume PDF if uploaded
    if data.get("resume_type") == "pdf" and data.get("resume_pdf_bytes"):
        (profile_dir / "resume.pdf").write_bytes(data["resume_pdf_bytes"])

    # Write config.yaml
    config = generate_config(data)
    with open(profile_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # Save API key to root .env (updates existing key, adds if missing)
    api_key = data.get("api_key", "").strip()
    if api_key:
        env_var = data.get("env_var", "")
        if env_var:
            env_path = _BASE_DIR / ".env"
            _upsert_env_key(env_path, env_var, api_key)

    # Initialize the profile's DB
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


# ── Step renderers ────────────────────────────────────────────────────────────

def _step_job_type() -> None:
    st.header("What are you looking for?")

    data = st.session_state.onboarding_data
    current_idx = 1 if data.get("job_type") == "internship" else 0
    job_type_choice = st.radio(
        "Job type",
        ["Full-time", "Internship"],
        index=current_idx,
        horizontal=True,
        label_visibility="collapsed",
    )

    if job_type_choice == "Internship":
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            season_opts = ["Summer", "Fall", "Spring"]
            season = st.selectbox(
                "Target season",
                season_opts,
                index=season_opts.index(data.get("target_season", "Summer")),
            )
        with col2:
            year_opts = ["2025", "2026", "2027"]
            year = st.selectbox(
                "Year",
                year_opts,
                index=year_opts.index(data.get("target_year", "2026")),
            )
        school = st.text_input("School name", value=data.get("school", ""))
        col1, col2 = st.columns(2)
        with col1:
            major = st.text_input("Major", value=data.get("major", ""))
        with col2:
            grad_year_opts = ["2025", "2026", "2027", "2028", "2029"]
            grad_default = data.get("graduation_year", "2027")
            grad_year = st.selectbox(
                "Graduation year",
                grad_year_opts,
                index=grad_year_opts.index(grad_default) if grad_default in grad_year_opts else 2,
            )
        gpa = st.text_input("GPA (optional)", value=data.get("gpa", ""))

    st.divider()
    if st.button("Next →", type="primary", key="step1_next"):
        data["job_type"] = "internship" if job_type_choice == "Internship" else "fulltime"
        if job_type_choice == "Internship":
            data["target_season"]   = season
            data["target_year"]     = year
            data["school"]          = school
            data["major"]           = major
            data["graduation_year"] = grad_year
            data["gpa"]             = gpa
        st.session_state.onboarding_step = 2
        st.rerun()


def _step_basic_info() -> None:
    st.header("About you")

    data = st.session_state.onboarding_data
    name = st.text_input("Your name", value=data.get("name", ""))

    st.subheader("Resume")
    resume_opts = ["Upload PDF", "Paste text"]
    resume_type_idx = 0 if data.get("resume_type") == "pdf" else 1
    resume_choice = st.radio("How would you like to provide your resume?",
                             resume_opts, index=resume_type_idx, horizontal=True)

    if resume_choice == "Upload PDF":
        has_pdf = bool(data.get("resume_pdf_bytes"))
        if has_pdf:
            st.success(f"Uploaded: {data.get('resume_pdf_name', 'resume.pdf')}")
            if st.button("Replace PDF", key="replace_pdf"):
                data.pop("resume_pdf_bytes", None)
                data.pop("resume_pdf_name", None)
                st.rerun()
        else:
            uploaded = st.file_uploader("Upload your resume PDF", type=["pdf"])
            if uploaded is not None:
                data["resume_pdf_bytes"] = uploaded.getvalue()
                data["resume_pdf_name"]  = uploaded.name
                st.rerun()
    else:
        resume_text = st.text_area(
            "Resume text",
            value=data.get("resume_text", ""),
            height=220,
            help="Paste your resume as plain text",
        )

    bio_placeholder = (
        "I'm a junior CS student at Georgia Tech with experience in Python and ML, "
        "looking for a summer 2026 backend or AI engineering internship."
        if data.get("job_type") == "internship"
        else "Software engineer specializing in Python backend and LLM integrations, "
             "with 3 years of experience at startups and scale-ups."
    )
    bio = st.text_area(
        "Bio (2–3 sentences)",
        value=data.get("bio", ""),
        height=100,
        placeholder=bio_placeholder,
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back", key="step2_back"):
            st.session_state.onboarding_step = 1
            st.rerun()
    with col2:
        if st.button("Next →", type="primary", key="step2_next"):
            errors = []
            if not name.strip():
                errors.append("Name is required.")
            if resume_choice == "Upload PDF" and not data.get("resume_pdf_bytes"):
                errors.append("Please upload a PDF before continuing.")
            if resume_choice == "Paste text" and not resume_text.strip():
                errors.append("Resume text cannot be empty.")

            if errors:
                for e in errors:
                    st.error(e)
            else:
                data["name"] = name.strip()
                data["bio"]  = bio.strip()
                data["resume_type"] = "pdf" if resume_choice == "Upload PDF" else "text"
                if resume_choice == "Paste text":
                    data["resume_text"] = resume_text.strip()
                st.session_state.onboarding_step = 3
                st.rerun()


def _step_preferences() -> None:
    is_intern = st.session_state.onboarding_data.get("job_type") == "internship"
    st.header("Job preferences")

    data = st.session_state.onboarding_data

    # Titles
    default_titles = _DEFAULT_INTERN_TITLES if is_intern else _DEFAULT_FT_TITLES
    titles_default = "\n".join(data.get("titles", default_titles))
    titles_text = st.text_area(
        "Target job titles (one per line)",
        value=titles_default,
        height=160,
        help="Any job whose title contains one of these words will be fetched for scoring.",
    )

    # Skills
    skills_default = "\n".join(data.get("desired_skills", _DEFAULT_SKILLS))
    skills_text = st.text_area(
        "Desired skills (one per line)",
        value=skills_default,
        height=120,
        help="Technologies and tools you want to use.",
    )

    # Hard no keywords
    default_hard_no = _DEFAULT_INTERN_HARD_NO if is_intern else _DEFAULT_FT_HARD_NO
    hard_no_default = "\n".join(data.get("hard_no_keywords", default_hard_no))
    hard_no_text = st.text_area(
        "Hard-no keywords (one per line)",
        value=hard_no_default,
        height=100,
        help="Jobs containing any of these phrases will be skipped before scoring.",
    )

    # Full-time only: YOE + salary
    if not is_intern:
        col1, col2 = st.columns(2)
        with col1:
            yoe = st.number_input(
                "Years of experience",
                min_value=0, max_value=30,
                value=int(data.get("yoe", 0)),
            )
        with col2:
            min_salary = st.number_input(
                "Minimum salary ($)",
                min_value=0, step=5_000,
                value=int(data.get("min_salary", 130_000)),
            )
    else:
        stipend = st.number_input(
            "Expected monthly stipend (optional, $ — used for scoring context)",
            min_value=0, step=500,
            value=int(data.get("stipend_expectation", 0)),
        )

    # Location
    st.subheader("Location")
    remote_ok = st.checkbox(
        "Open to remote",
        value=data.get("remote_ok", True),
    )
    preferred_locations = st.multiselect(
        "Preferred locations",
        options=_LOCATION_OPTIONS,
        default=[loc for loc in data.get("preferred_locations", ["Remote", "San Francisco, CA", "New York, NY"])
                 if loc in _LOCATION_OPTIONS],
    )
    custom_loc = st.text_input(
        "Add a location not in the list (optional)",
        value="",
        placeholder="e.g. Austin, TX",
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back", key="step3_back"):
            st.session_state.onboarding_step = 2
            st.rerun()
    with col2:
        if st.button("Next →", type="primary", key="step3_next"):
            all_locations = preferred_locations[:]
            if custom_loc.strip() and custom_loc.strip() not in all_locations:
                all_locations.append(custom_loc.strip())

            data["titles"]            = _lines_to_list(titles_text)
            data["desired_skills"]    = _lines_to_list(skills_text)
            data["hard_no_keywords"]  = _lines_to_list(hard_no_text)
            data["remote_ok"]         = remote_ok
            data["preferred_locations"] = all_locations
            if not is_intern:
                data["yoe"]        = int(yoe)
                data["min_salary"] = int(min_salary)
            else:
                data["stipend_expectation"] = int(stipend)

            st.session_state.onboarding_step = 4
            st.rerun()


def _step_llm_provider() -> None:
    st.header("Choose your AI provider")

    data = st.session_state.onboarding_data

    comparison_frame = pd.DataFrame(
        [
            {
                "Provider": provider["provider"].title(),
                "Model": provider["model_id"],
                "Free RPM": provider["rpm"],
                "Free RPD": provider["rpd"] or "Unlimited",
                "Accuracy": f"{provider['stars']}{' (paid)' if provider['paid'] else ''}",
            }
            for provider in _PROVIDERS
        ]
    )
    st.dataframe(comparison_frame, use_container_width=True, hide_index=True)

    # Provider selection
    current_label = data.get("provider_label", _PROVIDER_LABELS[0])
    current_idx   = _PROVIDER_LABELS.index(current_label) if current_label in _PROVIDER_LABELS else 0
    selected_label = st.radio(
        "Select provider and model",
        _PROVIDER_LABELS,
        index=current_idx,
        label_visibility="collapsed",
    )

    selected = next(p for p in _PROVIDERS if p["label"] == selected_label)

    # Runtime estimate
    rpd_val = selected["rpd"]
    est = _runtime_estimate(selected["rpm"], rpd_val)
    st.caption(f"At 1,000 jobs/day, {selected['label'].split('—')[1].strip()} can score your full list in {est}.")

    # API key input
    st.subheader("API key")
    if selected["paid"]:
        st.info("Anthropic Claude Sonnet is a paid model. You will be billed per token.")
    api_key = st.text_input(
        f"{selected['env_var']}",
        value=data.get("api_key", "") if data.get("provider") == selected["provider"] else "",
        type="password",
        help="Don't have an API key? Contact admin for assistance.",
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back", key="step4_back"):
            st.session_state.onboarding_step = 3
            st.rerun()
    with col2:
        if st.button("Next →", type="primary", key="step4_next"):
            if not api_key.strip():
                st.error(f"Please enter your {selected['env_var']} to continue.")
            else:
                data["provider_label"] = selected_label
                data["provider"]   = selected["provider"]
                data["model_key"]  = selected["model_key"]
                data["model_id"]   = selected["model_id"]
                data["env_var"]    = selected["env_var"]
                data["api_key"]    = api_key.strip()
                st.session_state.onboarding_step = 5
                st.rerun()


def _step_review_create() -> None:
    st.header("Review and create profile")

    data = st.session_state.onboarding_data
    is_intern = data.get("job_type") == "internship"

    # Suggest slug from name
    suggested_slug = sanitize_slug(data.get("name", "profile"))
    profile_slug = st.text_input(
        "Profile name (used as folder slug)",
        value=data.get("profile_slug", suggested_slug),
        help="Lowercase, underscores only. This becomes the folder name under profiles/.",
    )
    profile_slug = sanitize_slug(profile_slug)
    if profile_slug:
        st.caption(f"Will be created at: profiles/{profile_slug}/")

    st.divider()
    st.subheader("Summary")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Name:** {data.get('name', '')}")
        st.markdown(f"**Type:** {'Internship' if is_intern else 'Full-time'}")
        if is_intern:
            st.markdown(f"**Target:** {data.get('target_season', '')} {data.get('target_year', '')} — {data.get('school', '')}")
        resume_label = f"PDF ({data.get('resume_pdf_name', 'resume.pdf')})" if data.get("resume_type") == "pdf" else "Text"
        st.markdown(f"**Resume:** {resume_label}")
        st.markdown(f"**Provider:** {data.get('provider_label', '')}")
    with col2:
        st.markdown(f"**Remote OK:** {'Yes' if data.get('remote_ok') else 'No'}")
        locs = ", ".join(data.get("preferred_locations", []))
        st.markdown(f"**Locations:** {locs or '—'}")
        if not is_intern:
            sal = data.get("min_salary", 0)
            st.markdown(f"**Min salary:** ${sal:,}")
        else:
            stip = data.get("stipend_expectation", 0)
            st.markdown(f"**Stipend:** ${stip:,}/mo" if stip else "**Stipend:** —")

    titles = data.get("titles", [])
    if titles:
        st.markdown("**Target titles:**")
        st.code("\n".join(titles))

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back", key="step5_back"):
            st.session_state.onboarding_step = 4
            st.rerun()
    with col2:
        if st.button("Create profile", type="primary", key="step5_create"):
            if not profile_slug:
                st.error("Profile name cannot be empty.")
                return

            profile_dir = _PROFILES_DIR / profile_slug
            if profile_dir.exists():
                st.warning(
                    f"A profile named '{profile_slug}' already exists. "
                    "Choose a different name or go back to the main page."
                )
                return

            data["profile_slug"] = profile_slug
            try:
                create_profile(data)
            except Exception as e:
                st.error(f"Failed to create profile: {e}")
                return

            # Success — clear onboarding state
            st.session_state.onboarding_step  = 1
            st.session_state.onboarding_data  = {}
            st.session_state.show_onboarding  = False
            st.session_state.active_profile   = profile_slug
            st.cache_data.clear()
            st.success(
                f"Profile '{profile_slug}' created! "
                "Your profile is ready. Go to the dashboard to start your first job search."
            )
            st.balloons()
            st.rerun()


# ── Public entry point ────────────────────────────────────────────────────────

def render_onboarding() -> None:
    """
    Main entry point. Call this from dashboard.py when the user wants to
    create a new profile. Manages its own multi-step state via st.session_state.
    """
    if "onboarding_step" not in st.session_state:
        st.session_state.onboarding_step = 1
    if "onboarding_data" not in st.session_state:
        st.session_state.onboarding_data = {}

    step = st.session_state.onboarding_step

    st.progress(step / 5)
    st.caption(f"Step {step} of 5 — New profile setup")
    st.info(
        "You will need your name, a resume, and one provider API key. Creating a profile writes "
        "`profiles/<slug>/config.yaml`, initializes a dedicated jobs database, and opens that "
        "profile in the dashboard."
    )

    steps = [
        _step_job_type,
        _step_basic_info,
        _step_preferences,
        _step_llm_provider,
        _step_review_create,
    ]
    steps[step - 1]()
