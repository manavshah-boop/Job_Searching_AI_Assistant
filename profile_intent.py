"""
profile_intent.py - Role-agnostic profile normalization for semantic matching.

Converts a raw config dict into a ProfileIntent that captures what the candidate
is looking for without assuming any specific role family (software, finance,
marketing, operations, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Role family registry ──────────────────────────────────────────────────────

SUPPORTED_ROLE_FAMILIES = {
    "software_engineering",
    "data_analytics",
    "finance",
    "product_management",
    "operations",
    "sales",
    "marketing",
    "design",
    "internship",
    "general",
}

_FAMILY_TITLE_SIGNALS: dict[str, tuple[str, ...]] = {
    "software_engineering": (
        "software engineer", "backend engineer", "frontend engineer",
        "full stack", "fullstack", "swe", "platform engineer",
        "ml engineer", "ai engineer", "devops engineer", "site reliability",
        "infrastructure engineer", "data engineer", "mobile engineer",
        "ios engineer", "android engineer", "firmware engineer",
    ),
    "data_analytics": (
        "data analyst", "data scientist", "analytics engineer",
        "business analyst", "bi engineer", "business intelligence",
        "quantitative analyst", "research analyst",
    ),
    "finance": (
        "financial analyst", "fp&a", "investment banker", "equity research",
        "accounting", "auditor", "controller", "treasury analyst",
        "credit analyst", "risk analyst", "portfolio manager",
        "corporate finance", "finance associate", "finance manager",
    ),
    "product_management": (
        "product manager", "product owner", "associate pm",
        "growth pm", "technical pm", "product lead",
    ),
    "operations": (
        "operations manager", "operations analyst", "program manager",
        "project manager", "supply chain", "logistics", "procurement",
        "vendor management", "process improvement", "operations associate",
    ),
    "sales": (
        "sales", "account executive", "account manager",
        "business development", "bdr", "sdr", "customer success",
        "revenue", "enterprise sales", "sales associate",
    ),
    "marketing": (
        "marketing", "growth marketing", "demand generation", "content",
        "seo specialist", "paid search", "email marketing", "lifecycle",
        "brand", "communications", "public relations",
    ),
    "design": (
        "designer", "ux designer", "ui designer", "product design",
        "visual design", "brand design", "graphic design", "motion design",
    ),
}

_FAMILY_SKILL_SIGNALS: dict[str, tuple[str, ...]] = {
    "software_engineering": (
        "python", "java", "javascript", "typescript", "golang", "rust",
        "react", "fastapi", "django", "aws", "gcp", "docker", "kubernetes",
        "postgresql", "redis", "kafka", "apis", "microservices",
    ),
    "data_analytics": (
        "tableau", "power bi", "looker", "dbt", "spark", "hadoop",
        "pandas", "numpy", "data modeling", "etl", "airflow",
    ),
    "finance": (
        "excel", "financial modeling", "fp&a", "gaap", "quickbooks",
        "bloomberg", "cfa", "cpa", "valuation", "variance analysis",
        "budgeting", "forecasting", "audit",
    ),
    "product_management": (
        "roadmap", "user research", "a/b testing", "experimentation",
        "stakeholder management", "jira", "product strategy",
        "metrics", "okrs",
    ),
    "operations": (
        "erp", "sap", "oracle", "procurement", "inventory", "six sigma",
        "lean", "vendor management", "process improvement",
    ),
    "sales": (
        "salesforce", "crm", "hubspot", "outbound", "pipeline",
        "quota", "discovery calls", "enterprise accounts",
    ),
    "marketing": (
        "google analytics", "marketo", "seo", "sem",
        "paid search", "google ads", "email campaigns", "content strategy",
    ),
    "design": (
        "figma", "sketch", "adobe", "illustrator", "photoshop",
        "wireframing", "prototyping", "design systems",
    ),
}

# ── Canonical section taxonomy ────────────────────────────────────────────────

# Maps canonical label -> common alias phrases found in job description headers.
# Used by embedder._classify_header for more accurate section assignment.
CANONICAL_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "summary": (
        "about the role", "role overview", "role summary", "position overview",
        "overview", "summary", "about this role", "the role", "position summary",
        "job summary", "job overview",
    ),
    "company": (
        "about us", "about the company", "about the team", "who we are",
        "our company", "our mission", "the company", "company overview",
    ),
    "responsibilities": (
        "responsibilities", "what you'll do", "what you will do",
        "in this role", "day to day", "day-to-day", "what you'll own",
        "duties", "role expectations", "you will", "key responsibilities",
        "your responsibilities", "the job",
    ),
    # preferred_qualifications must come BEFORE requirements in iteration order
    # so that "preferred qualifications" matches here before "qualifications" matches requirements.
    "preferred_qualifications": (
        "preferred qualifications", "nice to have", "bonus points",
        "preferred skills", "ideal candidate", "plus if you have",
        "preferred experience", "bonus if you have",
    ),
    "requirements": (
        "requirements", "qualifications", "must have", "minimum qualifications",
        "basic qualifications", "what we're looking for", "you have",
        "experience with", "skills", "what you bring", "who you are",
        "experience needed", "required skills", "required qualifications",
        "what you'll need", "you should have", "minimum requirements",
    ),
    "tools_and_skills": (
        "technologies", "tools", "stack", "tech stack", "systems",
        "platforms", "skills and tools", "tools and technologies",
        "technical skills", "tools we use",
    ),
    "logistics": (
        "location", "schedule", "travel", "work authorization",
        "visa", "onsite", "hybrid", "remote work", "visa sponsorship",
        "where you'll work",
    ),
    "compensation": (
        "compensation", "salary", "pay range", "salary range", "base pay",
        "base salary", "hourly rate", "ote", "equity", "total compensation",
    ),
    "benefits": (
        "benefits", "perks", "what we offer", "why you'll love",
        "healthcare", "medical", "pto", "401k", "employee benefits",
    ),
}

# Maps canonical section label -> chunk_key used in DB and Chroma.
# preferred_qualifications and tools_and_skills fold into requirements;
# company and logistics fold into summary.
CANONICAL_TO_CHUNK_KEY: dict[str, str] = {
    "summary": "summary",
    "company": "summary",
    "responsibilities": "responsibilities",
    "requirements": "requirements",
    "preferred_qualifications": "requirements",
    "tools_and_skills": "requirements",
    "logistics": "requirements",
    "compensation": "compensation",
    "benefits": "benefits",
    "unknown": "summary",
}

# ── Role-family section weights ───────────────────────────────────────────────
# Weights applied to reranked chunk scores by role family.
# Keys are chunk_key values (the 5-value set used in DB/Chroma).
_ROLE_FAMILY_SECTION_WEIGHTS: dict[str, dict[str, float]] = {
    "software_engineering": {
        "requirements": 1.20,
        "responsibilities": 1.15,
        "summary": 1.00,
        "compensation": 0.90,
        "benefits": 0.80,
    },
    "data_analytics": {
        "requirements": 1.18,
        "responsibilities": 1.12,
        "summary": 1.00,
        "compensation": 0.90,
        "benefits": 0.80,
    },
    "finance": {
        "requirements": 1.20,
        "responsibilities": 1.15,
        "summary": 1.00,
        "compensation": 0.95,
        "benefits": 0.80,
    },
    "product_management": {
        "requirements": 1.15,
        "responsibilities": 1.20,
        "summary": 1.05,
        "compensation": 0.90,
        "benefits": 0.80,
    },
    "operations": {
        "requirements": 1.15,
        "responsibilities": 1.20,
        "summary": 1.00,
        "compensation": 0.90,
        "benefits": 0.80,
    },
    "sales": {
        "responsibilities": 1.20,
        "requirements": 1.15,
        "summary": 1.00,
        "compensation": 1.00,
        "benefits": 0.85,
    },
    "marketing": {
        "requirements": 1.15,
        "responsibilities": 1.15,
        "summary": 1.05,
        "compensation": 0.90,
        "benefits": 0.80,
    },
    "design": {
        "requirements": 1.15,
        "responsibilities": 1.15,
        "summary": 1.05,
        "compensation": 0.90,
        "benefits": 0.80,
    },
    "internship": {
        "requirements": 1.10,
        "responsibilities": 1.15,
        "compensation": 1.10,
        "summary": 1.00,
        "benefits": 0.90,
    },
    "general": {
        "requirements": 1.15,
        "responsibilities": 1.10,
        "summary": 1.00,
        "compensation": 0.90,
        "benefits": 0.80,
    },
}

# Credential patterns per role family: credential label -> phrases in job text
CREDENTIAL_REQUIREMENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "CPA": ("cpa required", "cpa preferred", "must have cpa", "active cpa", "cpa license"),
    "CFA": ("cfa required", "cfa preferred", "cfa designation", "cfa level", "chartered financial analyst"),
    "PMP": ("pmp required", "pmp preferred", "pmp certification", "pmp certified"),
    "MBA": ("mba required", "mba preferred", "must have mba", "mba degree"),
    "Series 7": ("series 7 required", "series 7 preferred", "series 7 license"),
    "clearance": (
        "security clearance", "secret clearance", "top secret", "ts/sci",
        "clearance required", "active clearance",
    ),
    "degree": (
        "bachelor's degree required", "bs required", "b.s. required",
        "degree required", "must have a degree",
    ),
}

# ── ProfileIntent ─────────────────────────────────────────────────────────────


@dataclass
class ProfileIntent:
    profile_name: str
    role_family: str
    target_roles: list[str] = field(default_factory=list)
    seniority: str = "entry"
    years_experience: float = 0
    industries: list[str] = field(default_factory=list)
    domain_skills: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    credentials: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    remote_ok: bool = True
    compensation: dict = field(default_factory=dict)
    dealbreakers: list[str] = field(default_factory=list)
    nice_to_haves: list[str] = field(default_factory=list)
    job_type: str = "fulltime"
    raw_keywords: list[str] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            result.append(cleaned)
    return result


# Generic tool/platform phrases recognizable across many role families.
_TOOL_TOKENS = frozenset({
    "excel", "sql", "python", "r", "tableau", "power bi", "salesforce", "hubspot",
    "jira", "figma", "aws", "gcp", "azure", "docker", "kubernetes", "react",
    "fastapi", "django", "looker", "dbt", "airflow", "spark", "bloomberg",
    "quickbooks", "sap", "oracle", "erp", "google analytics", "marketo",
    "mailchimp", "wordpress", "airtable", "notion", "asana", "linear",
    "slack", "github", "gitlab", "terraform", "ansible", "jenkins",
    "pandas", "numpy", "pytorch", "tensorflow",
})

_CREDENTIAL_RESUME_SIGNALS: dict[str, tuple[str, ...]] = {
    "finance": ("cfa", "cpa", "frm", "series 7", "series 66", "mba"),
    "data_analytics": ("google analytics certified", "tableau certified"),
    "software_engineering": ("aws certified", "gcp certified", "azure certified"),
    "marketing": ("google ads certified", "hubspot certified", "facebook blueprint"),
    "product_management": ("cspo", "pmp", "csm"),
    "operations": ("pmp", "six sigma", "lean", "scrum master", "csm"),
}


def _infer_role_family(config: dict[str, Any]) -> str:
    """Infer role family from profile config. Explicit config key wins; falls back to signal matching."""
    explicit = str(config.get("profile", {}).get("role_family", "")).strip().lower()
    if explicit in SUPPORTED_ROLE_FAMILIES:
        return explicit

    prefs = config.get("preferences", {})
    profile_cfg = config.get("profile", {})
    titles_text = " ".join(str(t).lower() for t in prefs.get("titles", []))
    skills_text = " ".join(str(s).lower() for s in prefs.get("desired_skills", []))
    bio_text = str(profile_cfg.get("bio", "")).lower()
    resume_text = str(profile_cfg.get("resume", "")).lower()[:600]
    combined = f"{titles_text} {skills_text} {bio_text} {resume_text}"

    family_scores: dict[str, int] = {}
    for family, title_signals in _FAMILY_TITLE_SIGNALS.items():
        score = sum(2 for signal in title_signals if signal in combined)
        skill_signals = _FAMILY_SKILL_SIGNALS.get(family, ())
        score += sum(1 for signal in skill_signals if signal in combined)
        if score > 0:
            family_scores[family] = score

    if not family_scores:
        job_type = str(profile_cfg.get("job_type", "")).lower()
        return "internship" if job_type == "internship" else "general"

    best = max(family_scores, key=family_scores.get)
    if profile_cfg.get("job_type") == "internship" and family_scores.get(best, 0) <= 2:
        return "internship"
    return best


def _infer_seniority(config: dict[str, Any]) -> str:
    job_type = str(config.get("profile", {}).get("job_type", "fulltime")).lower()
    if job_type == "internship":
        return "intern"
    yoe = float(config.get("preferences", {}).get("yoe", 0))
    if yoe <= 2:
        return "entry"
    if yoe <= 5:
        return "mid"
    if yoe <= 8:
        return "senior"
    return "staff"


def _extract_credentials(config: dict[str, Any], role_family: str) -> list[str]:
    """Look for known credential signals in resume/bio text."""
    profile_cfg = config.get("profile", {})
    text = f"{profile_cfg.get('resume', '')} {profile_cfg.get('bio', '')}".lower()
    patterns = _CREDENTIAL_RESUME_SIGNALS.get(role_family, ())
    return _dedupe([p.upper() for p in patterns if p in text])


def _split_skills_and_tools(skills: list[str]) -> tuple[list[str], list[str]]:
    """Partition a skill list into domain skills and identifiable tools/platforms."""
    tools: list[str] = []
    domain: list[str] = []
    for skill in skills:
        if any(token in skill.lower() for token in _TOOL_TOKENS):
            tools.append(skill)
        else:
            domain.append(skill)
    return _dedupe(domain), _dedupe(tools)


# ── Public API ────────────────────────────────────────────────────────────────


def get_role_family_section_weights(role_family: str) -> dict[str, float]:
    """Return chunk-key section weights for the given role family."""
    return _ROLE_FAMILY_SECTION_WEIGHTS.get(role_family, _ROLE_FAMILY_SECTION_WEIGHTS["general"])


def map_header_to_canonical(text: str) -> str | None:
    """Map a job description header string to a canonical section label, or None."""
    header = text.strip().rstrip(":").lower()
    for canonical, aliases in CANONICAL_SECTION_ALIASES.items():
        if any(alias in header for alias in aliases):
            return canonical
    return None


def normalize_profile_intent(config: dict[str, Any]) -> ProfileIntent:
    """Convert a config dict into a role-agnostic ProfileIntent."""
    profile_cfg = config.get("profile", {})
    prefs = config.get("preferences", {})
    location = prefs.get("location", {})
    compensation = prefs.get("compensation", {})

    role_family = _infer_role_family(config)
    seniority = _infer_seniority(config)

    raw_skills = [str(s).strip() for s in prefs.get("desired_skills", []) if str(s).strip()]
    domain_skills, tools = _split_skills_and_tools(raw_skills)
    credentials = _extract_credentials(config, role_family)

    title_blocklist = [str(t).strip() for t in prefs.get("filters", {}).get("title_blocklist", []) if str(t).strip()]
    hard_no = [str(k).strip() for k in prefs.get("hard_no_keywords", []) if str(k).strip()]
    dealbreakers = _dedupe(title_blocklist + hard_no)[:8]

    nice_to_haves = _dedupe([str(n).strip() for n in profile_cfg.get("nice_to_haves", []) if str(n).strip()])

    raw_keywords = _dedupe(
        [str(t).strip() for t in prefs.get("titles", [])]
        + [str(s).strip() for s in prefs.get("desired_skills", [])]
    )[:20]

    return ProfileIntent(
        profile_name=str(profile_cfg.get("name", "Candidate")),
        role_family=role_family,
        target_roles=_dedupe([str(t).strip() for t in prefs.get("titles", []) if str(t).strip()]),
        seniority=seniority,
        years_experience=float(prefs.get("yoe", 0)),
        industries=_dedupe([str(i).strip() for i in profile_cfg.get("industries", []) if str(i).strip()]),
        domain_skills=domain_skills,
        tools=tools,
        credentials=credentials,
        education=_dedupe([str(e).strip() for e in profile_cfg.get("education", []) if str(e).strip()]),
        locations=_dedupe([str(loc).strip() for loc in location.get("preferred_locations", []) if str(loc).strip()]),
        remote_ok=bool(location.get("remote_ok", True)),
        compensation=dict(compensation),
        dealbreakers=dealbreakers,
        nice_to_haves=nice_to_haves,
        job_type=str(profile_cfg.get("job_type", "fulltime")).strip().lower(),
        raw_keywords=raw_keywords,
    )
