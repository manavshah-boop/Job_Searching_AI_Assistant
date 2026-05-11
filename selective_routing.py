"""
selective_routing.py — Step 26: Selective LLM routing for cost-efficient scoring.

Routes jobs through the LLM only when a cross-encoder routing score meets the
configured threshold. Below-threshold jobs get a conservative synthetic score
derived from the cross-encoder signal and lightweight section detection — no API
call needed.

Usage (config.yaml):
  routing:
    enabled: true
    threshold: 0.65
    quality_mode: quality   # "fast" or "quality" — selects LLM model tier
    log_routing_decisions: true
"""

import copy
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from loguru import logger

from db import Job

# ── Cross-encoder model cache ─────────────────────────────────────────────────
# Shared with reranker so we don't load the same ~150MB model twice on the
# 1GB e2-micro deployment. Importing reranker._MODEL_CACHE directly would
# create a hard import-time dependency; we rebind through a lazy accessor.
def _shared_model_cache() -> dict[str, Any]:
    try:
        from reranker import _MODEL_CACHE as _RERANKER_CACHE
        return _RERANKER_CACHE
    except Exception:
        # Fallback to module-local cache if reranker is unavailable
        # (e.g. tests that import selective_routing in isolation).
        return _MODEL_CACHE


_MODEL_CACHE: dict[str, Any] = {}

_DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
# BGE-reranker-base post-sigmoid scores spread across [0,1] more uniformly than
# ms-marco-MiniLM-L-6-v2 did (which compressed into ~0.2-0.35 for decent matches).
# 0.18 was originally tuned for the ms-marco compression — kept here as a
# permissive default that still routes plausible jobs to the LLM under BGE.
# Per-profile `routing.llm_threshold` (e.g. 0.30) is the right knob for re-tuning
# now that the underlying score distribution is wider.
_DEFAULT_THRESHOLD = 0.18
_DEFAULT_QUALITY_MODE = "fast"

# (provider, quality_mode) → model name
# Three modes: fast (cheapest), medium (balanced), quality (best — used for uncertain jobs)
_MODEL_MAP: dict[tuple[str, str], str] = {
    ("groq", "fast"):    "llama-3.1-8b-instant",
    ("groq", "medium"):  "llama-3.3-70b-versatile",
    ("groq", "quality"): "meta-llama/llama-4-scout-17b-16e-instruct",
    ("anthropic", "fast"):    "claude-haiku-4-5-20251001",
    ("anthropic", "medium"):  "claude-haiku-4-5-20251001",
    ("anthropic", "quality"): "claude-sonnet-4-20250514",
    ("openai", "fast"):    "gpt-4o-mini",
    ("openai", "medium"):  "gpt-4o-mini",
    ("openai", "quality"): "gpt-4o",
    ("gemini", "fast"):    "gemini-2.0-flash",
    ("gemini", "medium"):  "gemini-2.5-flash",
    ("gemini", "quality"): "gemini-2.5-flash",
}

# Heuristic section header patterns used for cheap section detection
_SECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "requirements": re.compile(
        r"(?i)(requirements?|qualifications?|must.have|you.?ll need|what you need|"
        r"basic qualifications|minimum qualifications|preferred qualifications)"
    ),
    "responsibilities": re.compile(
        r"(?i)(responsibilities?|what you.?ll do|your role|day.to.day|"
        r"key responsibilities|job duties|you will|you.?ll be)"
    ),
    "summary": re.compile(
        r"(?i)(about (?:the role|this role|you|us|the job)|job description|"
        r"overview|summary|we are looking|position overview)"
    ),
    "benefits": re.compile(
        r"(?i)(benefits?|perks?|what we offer|we offer|compensation|equity|salary range)"
    ),
}

_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "are", "with", "you", "our", "will", "have",
    "this", "that", "from", "not", "but", "your", "they", "was", "all",
})

# Rare/niche tech terms signal a complex job that warrants a quality-tier model
# even when the reranker score looks high.
_RARE_TERMS: frozenset[str] = frozenset({
    "kubernetes", "terraform", "mlops", "consensus", "cryptography",
    "zero-knowledge", "homomorphic", "formal verification", "model parallelism",
    "rlhf", "quantization", "fpga", "asic", "verilog", "cuda", "triton",
    "flash attention", "byzantine", "sharding",
})

# Default tier configuration — overridable via routing.tiers in profile config.yaml.
# High confidence (obvious match) → cheap fast model.
# Low confidence (uncertain) → quality model for deeper analysis.
_DEFAULT_TIERS: dict[str, Any] = {
    "high": {
        "min_reranker": 0.40,
        "require_title_match": True,
        "quality_mode": "fast",
        "max_tokens": 500,
    },
    "medium": {
        "min_reranker": 0.25,
        "quality_mode": "fast",
        "max_tokens": 700,
    },
    "low": {
        "min_reranker": 0.0,
        "quality_mode": "quality",
        "max_tokens": 900,
    },
}


def routing_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("routing", {}).get("enabled", False))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _load_cross_encoder(model_name: str) -> Any:
    cache = _shared_model_cache()
    if model_name not in cache:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for selective routing. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        logger.info("selective_routing | loading cross-encoder {}", model_name)
        model = CrossEncoder(model_name, device="cpu")
        if hasattr(model, "max_length"):
            model.max_length = 512
        cache[model_name] = model
    return cache[model_name]


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class RoutingConfig:
    enabled: bool
    threshold: float
    quality_mode: str
    log_routing_decisions: bool


# ── Main router class ─────────────────────────────────────────────────────────

class SelectiveRouter:
    """
    Routes individual jobs to LLM or synthetic scoring based on cross-encoder score.

    Construct once per scoring run (before the job loop), call
    compute_routing_score() + should_call_llm() per job, and
    create_synthetic_score() for jobs that skip the LLM.
    """

    def __init__(self, config: dict[str, Any], profile: Optional[str] = None) -> None:
        raw = config.get("routing", {})
        self.enabled: bool = bool(raw.get("enabled", False))

        # Per-profile llm_threshold overrides the base threshold (both fall back to code default).
        # This lets individual profiles fine-tune without changing the global default.
        raw_threshold = raw.get("llm_threshold") if raw.get("llm_threshold") is not None else raw.get("threshold", _DEFAULT_THRESHOLD)
        try:
            threshold = float(raw_threshold)
        except (TypeError, ValueError):
            threshold = _DEFAULT_THRESHOLD
        self.threshold: float = max(0.0, min(1.0, threshold))

        self.quality_mode: str = str(raw.get("quality_mode", _DEFAULT_QUALITY_MODE)).lower()
        self.log_decisions: bool = bool(raw.get("log_routing_decisions", True))
        self._config = config
        self._profile = profile
        self._encoder_model_name: str = config.get("reranking", {}).get(
            "model", _DEFAULT_RERANKER_MODEL
        )
        self._encoder: Any = None
        self._match_query: str = ""
        # Structured profile (LLM-extracted canonical skills) for canonical ATS
        # matching in _quick_ats. Populated by score_all_jobs after build_structured_profile.
        self._structured_profile: Optional[dict[str, Any]] = None
        self._counts: dict[str, int] = {"llm": 0, "skipped": 0, "error": 0}
        self._synthetic_routing_scores: list[float] = []  # for domain suitability diagnostic

    def set_structured_profile(self, profile: Optional[dict[str, Any]]) -> None:
        """Cache the LLM-extracted candidate profile so synthetic scoring can do canonical ATS."""
        self._structured_profile = profile

    # ── Encoder ───────────────────────────────────────────────────────────────

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            self._encoder = _load_cross_encoder(self._encoder_model_name)
        return self._encoder

    # ── Match query ───────────────────────────────────────────────────────────

    def build_match_query(self) -> str:
        """Build the profile query used to score each job via cross-encoder."""
        try:
            from reranker import build_profile_match_query
            query = build_profile_match_query(self._config)
        except Exception as exc:
            logger.warning("selective_routing | build_match_query failed ({}), using fallback", exc)
            prefs = self._config.get("preferences", {})
            titles = prefs.get("titles", [])
            skills = prefs.get("desired_skills", [])
            parts: list[str] = []
            if titles:
                parts.append(f"Looking for: {', '.join(titles[:4])}")
            if skills:
                parts.append(f"Skills: {', '.join(skills[:8])}")
            query = ". ".join(parts) or "software engineer"
        self._match_query = query
        return query

    # ── Routing score ─────────────────────────────────────────────────────────

    def compute_routing_score(self, job: Job, match_query: str) -> float:
        """
        Compute a cross-encoder routing score (0–1) for one job.
        Truncates job text to 1800 chars to fit the cross-encoder's 512-token window.
        Returns 0.5 on any error (fail-open → call LLM).
        """
        text = (job.raw_text or "").strip()
        if not text:
            return 0.0

        truncated = text[:1800]
        started = time.perf_counter()
        try:
            encoder = self._get_encoder()
            raw_scores = encoder.predict(
                [(match_query, truncated)],
                show_progress_bar=False,
            )
            score = round(float(_sigmoid(float(raw_scores[0]))), 4)
        except Exception as exc:
            logger.warning(
                "selective_routing | cross-encoder failed for job={} ({}), defaulting to 0.5",
                job.id, exc,
            )
            self._counts["error"] += 1
            return 0.5  # fail-open: uncertain jobs go to LLM

        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.debug(
            "selective_routing | job={} score={:.3f} latency_ms={}",
            job.id, score, latency_ms,
        )
        return score

    # ── Title boost ───────────────────────────────────────────────────────────

    def compute_title_boost(self, job: Job) -> float:
        """
        Return 0.25 when the job title contains any configured target role.
        Applied to the effective score for the routing/tier decision only —
        never stored in the final score record. The boost was originally
        sized for ms-marco's compressed output (great matches landed near
        ~0.30); it remains a useful ~+0.25 lift under BGE's wider distribution
        and reliably clears the 0.18 floor for any title-matched role.
        """
        target_roles = self._config.get("preferences", {}).get("titles", [])
        if not target_roles or not job.title:
            return 0.0
        job_title_lower = job.title.lower().strip()
        for role in target_roles:
            if role.lower().strip() in job_title_lower:
                return 0.25
        return 0.0

    # ── Skills-overlap boost ──────────────────────────────────────────────────

    def compute_skills_boost(self, job: Job) -> float:
        """
        Return a boost based on how many of the candidate's desired_skills appear
        in the job text. The reranker (ms-marco) is undertuned for resume↔JD
        matching and routinely under-scores good fits; explicit skill-keyword
        overlap is a direct, conservative signal we can layer on top.

          ≥3 desired-skill hits  → +0.20 (likely-good fit, force-route to LLM)
          ≥2 desired-skill hits  → +0.10 (plausible fit, lifts borderline scores)
          else                   → 0
        """
        skills = self._config.get("preferences", {}).get("desired_skills", [])
        if not skills or not job.raw_text:
            return 0.0
        text_lower = job.raw_text.lower()
        hits = 0
        for skill in skills:
            sl = skill.lower().strip()
            if not sl:
                continue
            if sl in text_lower:
                hits += 1
                if hits >= 3:
                    return 0.20
        if hits >= 2:
            return 0.10
        return 0.0

    # ── Routing decision ──────────────────────────────────────────────────────

    def should_call_llm(self, routing_score: float) -> bool:
        """True if routing_score >= threshold (deterministic and idempotent)."""
        return routing_score >= self.threshold

    def record_llm_call(self) -> None:
        self._counts["llm"] += 1

    def record_skipped(self) -> None:
        self._counts["skipped"] += 1

    # ── Complexity + tier ─────────────────────────────────────────────────────

    def compute_complexity_score(self, job_text: str) -> float:
        """
        Returns 0.0–1.0 complexity estimate.
        Weighted by description length and presence of rare/niche tech terms.
        Used to downgrade a high-confidence tier when the job is unexpectedly complex.
        """
        word_count = len(job_text.split())
        length_score = min(1.0, word_count / 750)
        text_lower = job_text.lower()
        rare_hits = sum(1 for term in _RARE_TERMS if term in text_lower)
        rare_score = min(1.0, rare_hits / 3)
        return round(min(1.0, length_score * 0.5 + rare_score * 0.5), 3)

    def compute_tier(
        self,
        effective_score: float,
        job_text: str,
        title_matched: bool,
    ) -> dict[str, Any]:
        """
        Assign a routing tier from effective_score (reranker + title boost) and complexity.

        Tier priority (highest first):
          high   — strong score + title match + low complexity → cheapest fast model
          medium — moderate score → standard fast model
          low    — low score or complex job → quality model for deeper analysis

        Returns the full tier config so downstream code (notably
        select_model_with_fallback) can read primary_model and require_title_match
        without re-reading raw config. Keys returned:
          tier, quality_mode, max_tokens, primary_model, require_title_match
        """
        tiers = self._config.get("routing", {}).get("tiers", _DEFAULT_TIERS)
        complexity = self.compute_complexity_score(job_text)

        def _resolved(name: str, defaults: dict[str, Any]) -> dict[str, Any]:
            cfg = tiers.get(name, defaults)
            return {
                "tier": name,
                "quality_mode": cfg.get("quality_mode", defaults.get("quality_mode", "fast")),
                "max_tokens": cfg.get("max_tokens", defaults.get("max_tokens", 500)),
                "primary_model": cfg.get("primary_model", defaults.get("primary_model", "")),
                "require_title_match": cfg.get(
                    "require_title_match", defaults.get("require_title_match", False)
                ),
            }

        high = tiers.get("high", _DEFAULT_TIERS["high"])
        if (
            effective_score >= high.get("min_reranker", 0.40)
            and (not high.get("require_title_match", True) or title_matched)
            and complexity <= 0.7
        ):
            return _resolved("high", _DEFAULT_TIERS["high"])

        medium = tiers.get("medium", _DEFAULT_TIERS["medium"])
        if effective_score >= medium.get("min_reranker", 0.25):
            return _resolved("medium", _DEFAULT_TIERS["medium"])

        return _resolved("low", _DEFAULT_TIERS["low"])

    # ── Section detection ─────────────────────────────────────────────────────

    def detect_matched_sections(self, job: Job, match_query: str = "") -> list[str]:
        """
        Detect which job-description sections contain content that overlaps with
        the match query. Pure-Python keyword matching — no model call.

        Returns a subset of: ["requirements", "responsibilities", "summary", "benefits"]
        """
        text = (job.raw_text or "").strip()
        if not text:
            return []

        query = match_query or self._match_query
        query_terms: set[str] = {
            t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9+#./-]{2,}", query)
            if t.lower() not in _STOPWORDS
        }
        if not query_terms:
            return []

        # Bucket lines into sections by scanning for header patterns
        current = "summary"
        section_lines: dict[str, list[str]] = {k: [] for k in _SECTION_PATTERNS}
        section_lines["summary"] = []

        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            matched_header = False
            if len(line) < 100:  # headers are short
                for section, pattern in _SECTION_PATTERNS.items():
                    if pattern.search(line):
                        current = section
                        matched_header = True
                        break
            if not matched_header:
                section_lines.setdefault(current, []).append(line.lower())

        # A section is "matched" if it has ≥2 query term hits
        matched: list[str] = []
        for section, lines in section_lines.items():
            body = " ".join(lines)
            hits = sum(1 for term in query_terms if term in body)
            if hits >= 2:
                matched.append(section)

        return matched

    # ── Synthetic score ───────────────────────────────────────────────────────

    def _extract_tech_overlaps(self, job_text: str) -> list[str]:
        """Profile skills that appear in the job text (for synthetic explanations)."""
        prefs = self._config.get("preferences", {})
        skills = prefs.get("desired_skills", [])
        job_lower = job_text.lower()
        return [s for s in skills if s.lower() in job_lower][:6]

    # Hard cap on synthetic fit_score. Synthetic scores reflect "router thought
    # this was a plausible match, but no LLM verified it" — they must always
    # rank below an LLM-evaluated good fit so the user can trust the display
    # threshold semantics. 55 sits just below the conventional 60 min_display,
    # signalling "needs manual review or rerun with LLM budget".
    _SYNTHETIC_FIT_CAP: int = 55

    def create_synthetic_score(
        self,
        job: Job,
        routing_score: float,
        matched_sections: Optional[list[str]] = None,
        match_reason: str = "",
        title_matched: bool = False,
    ) -> dict[str, Any]:
        """
        Create a synthetic score for jobs that bypass the LLM.

        Two invariants drive the design:

          1. **No static floor.** Previously, location/growth/compensation were
             hard-coded to neutral 5–6 regardless of routing signal, producing
             a structural floor of ~22 fit_score even for clearly-unrelated
             jobs. The 2026-05-10 run had 96% of scored jobs cluster at 22–29
             because of this. New behavior: every dimension scales with
             routing_score, so a job with routing_score 0.05 lands in single
             digits.
          2. **Capped at SYNTHETIC_FIT_CAP (55).** Even a strong routing match
             without LLM verification must rank below a true LLM-scored good
             fit, so display thresholds remain meaningful.

        Matched-section bumps refine specific dimensions:
          - "requirements" matched  → stack_match lifted
          - "responsibilities" matched → seniority lifted
          - "summary" matched → role_fit lifted
          - title matched (router-level signal) → role_fit + stack lifted

        Returns a dict matching score_job()'s return format.
        """
        weights: dict[str, float] = self._config.get("scoring", {}).get("weights", {
            "role_fit": 0.30, "stack_match": 0.25, "seniority": 0.20,
            "location": 0.10, "growth": 0.10, "compensation": 0.05,
        })

        # Detect sections if not pre-computed
        if matched_sections is None:
            matched_sections = self.detect_matched_sections(job, self._match_query)

        has_requirements = "requirements" in matched_sections
        has_responsibilities = "responsibilities" in matched_sections
        has_summary = "summary" in matched_sections

        # Direct projection of routing signal — no 0.85 dampening, no static
        # neutral floor. Bonuses widen the spread between matched-section and
        # no-section jobs so the user can still see relative ranking inside
        # the synthetic bucket.
        scale = routing_score

        # role_fit: small lift if the summary section matched (confirms general fit).
        role_fit_scale = scale * (1.05 if has_summary else 1.0)

        # stack_match: meaningful spread between "requirements matched" and not.
        # Widened from 1.08/0.90 → 1.15/0.85 so the test_uses_matched_sections
        # delta is visible at all routing scores, not just 0.7+.
        stack_scale = scale * (1.15 if has_requirements else 0.85)

        # seniority: widened similarly. Was a flat 6-vs-5 jump that ignored
        # routing signal entirely.
        seniority_scale = scale * (1.15 if has_responsibilities else 0.85)

        # location / growth / compensation: the cross-encoder genuinely can't
        # infer these. Scale them at a discount (0.70x of routing) — synthetic
        # score should not invent strong location/comp signals when there is
        # no actual signal. The previous static 5/5/6 inflated unrelated jobs
        # by ~25 points purely from these three dimensions.
        loc_scale = scale * 0.70
        growth_scale = scale * 0.70
        comp_scale = scale * 0.70

        # Title-match boost: when the router's title check fired, that's a
        # strong direct signal — bump role_fit and stack proportionally.
        if title_matched:
            role_fit_scale = min(1.0, role_fit_scale + 0.10)
            stack_scale = min(1.0, stack_scale + 0.05)

        # Tech-overlap boost: every desired_skill found in the JD text is a
        # direct, verifiable signal that the cross-encoder may have missed.
        # The previous synthetic path counted these only in `reasons` but
        # never reflected them in stack_match — so a JD that literally listed
        # half the candidate's skills could still get stack_match=1. Each hit
        # lifts stack_scale by 0.04 (max +0.20 from 5 hits).
        tech_overlap_count = len(self._extract_tech_overlaps(job.raw_text or ""))
        if tech_overlap_count > 0:
            stack_scale = min(1.0, stack_scale + 0.04 * tech_overlap_count)

        role_fit     = max(0, min(10, round(role_fit_scale * 10)))
        stack_match  = max(0, min(10, round(stack_scale * 10)))
        seniority    = max(0, min(10, round(seniority_scale * 10)))
        location     = max(0, min(10, round(loc_scale * 10)))
        growth       = max(0, min(10, round(growth_scale * 10)))
        compensation = max(0, min(10, round(comp_scale * 10)))

        dims: dict[str, int] = {
            "role_fit": role_fit,
            "stack_match": stack_match,
            "seniority": seniority,
            "location": location,
            "growth": growth,
            "compensation": compensation,
        }

        # Compute weighted score, then cap at SYNTHETIC_FIT_CAP. The cap is
        # what makes display-threshold semantics ("≥60 = good") trustworthy.
        raw_fit = int(sum(dims[k] * weights.get(k, 0) for k in dims) * 10)
        fit_score = max(0, min(self._SYNTHETIC_FIT_CAP, raw_fit))
        ats_score = self._quick_ats(job)

        # Build informative reasons with specific evidence
        tech_found = self._extract_tech_overlaps(job.raw_text or "")
        reasons: list[str] = []
        if title_matched:
            reasons.append(f"Title match: {job.title.split(',')[0].strip()}")
        if tech_found:
            reasons.append(f"Skills overlap: {', '.join(tech_found[:4])}")
        if has_requirements:
            reasons.append("Requirements section signals skill alignment")
        if has_responsibilities and len(reasons) < 4:
            reasons.append("Responsibilities section aligns with target role")
        if not reasons:
            reasons.append(f"Routing score {routing_score:.2f} — plausible match")

        # Build a human-readable one_liner with concrete evidence
        title_note = f"Title match: {job.title.split(',')[0].strip()}. " if title_matched else ""
        tech_note = f"Skills: {', '.join(tech_found[:4])}. " if tech_found else ""
        one_liner = (
            f"{title_note}{tech_note}"
            "Keyword-based estimate — AI deep review skipped to save time. "
            "Consider a manual check."
        ).strip()
        if not title_note and not tech_note and match_reason.strip():
            one_liner = match_reason.strip()

        section_str = ", ".join(matched_sections) if matched_sections else "none"

        self.record_skipped()
        self._synthetic_routing_scores.append(routing_score)
        if self.log_decisions:
            logger.info(
                "routing | job={} company={} routed=skipped_llm routing_score={:.3f} "
                "threshold={:.2f} title_match={} matched_sections=[{}] synthetic_fit={}",
                job.id, job.company, routing_score, self.threshold,
                title_matched, section_str, fit_score,
            )

        # Human-readable flags — never expose raw internal field names to the UI
        synthetic_flags: list[str] = [
            f"Keyword estimate only — not reviewed by AI. Synthetic scores are "
            f"capped at {self._SYNTHETIC_FIT_CAP}/100; rerun with LLM budget to verify."
        ]
        if not matched_sections:
            synthetic_flags.append("No strong section matches found in job description")

        return {
            "job": job,
            "fit_score": fit_score,
            "tokens_used": 0,
            "ats_score": ats_score,
            "reasons": reasons,
            "flags": synthetic_flags,
            "one_liner": one_liner,
            "dimension_scores": dims,
            "skill_misses": [],
        }

    def _quick_ats(self, job: Job) -> int:
        """
        ATS score for the synthetic path. Delegates to scorer.compute_ats_score so
        synthetic and LLM-scored jobs use the same canonical-skill matching logic.

        Falling back to a naive `s.lower() in raw_text` check would understate the
        score for any multi-word skill phrase (e.g. "Machine learning infrastructure"
        rarely appears verbatim in JDs even when the role is a perfect fit) — that
        was CRIT-2 in the audit.
        """
        if not job.raw_text:
            return 0
        try:
            from scorer import compute_ats_score  # lazy: avoid circular import at module load
        except Exception as exc:
            logger.warning(
                "_quick_ats | could not import compute_ats_score ({}); using fallback", exc,
            )
            prefs = self._config.get("preferences", {})
            skills = prefs.get("desired_skills", [])
            if not skills:
                return 0
            raw_lower = job.raw_text.lower()
            hits = sum(1 for s in skills if s.lower() in raw_lower)
            return min(100, int(hits / max(1, len(skills)) * 100))

        try:
            result = compute_ats_score(job, self._config, self._structured_profile)
            return int(result.get("ats_score", 0))
        except Exception as exc:
            logger.warning("_quick_ats | compute_ats_score failed ({}); returning 0", exc)
            return 0

    # ── Model override ────────────────────────────────────────────────────────

    def get_effective_llm_model(self) -> str:
        """Return the LLM model name for jobs that DO call the LLM (quality_mode-aware)."""
        provider = self._config.get("llm", {}).get("provider", "groq")
        key = (provider, self.quality_mode)
        candidate = _MODEL_MAP.get(key)
        if candidate:
            return candidate
        return self._config.get("llm", {}).get("model", {}).get(provider, "")

    def get_model_for_quality_mode(self, quality_mode: str) -> str:
        """Return the model name for an explicit quality_mode and the configured provider."""
        provider = self._config.get("llm", {}).get("provider", "groq")
        candidate = _MODEL_MAP.get((provider, quality_mode))
        if candidate:
            return candidate
        return self._config.get("llm", {}).get("model", {}).get(provider, "")

    def apply_model_override(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Return a deep copy of config with the LLM model replaced by the
        quality_mode-appropriate model. Call this before get_llm_client().
        """
        effective = self.get_effective_llm_model()
        if not effective:
            return config
        provider = config.get("llm", {}).get("provider", "groq")
        cfg = copy.deepcopy(config)
        cfg.setdefault("llm", {}).setdefault("model", {})[provider] = effective
        return cfg

    def apply_model_override_for_mode(
        self, config: dict[str, Any], quality_mode: str
    ) -> dict[str, Any]:
        """
        Return a deep copy of config with the LLM model set to the given quality_mode.
        Used to pre-create per-tier LLM clients before the job scoring loop.
        """
        model = self.get_model_for_quality_mode(quality_mode)
        if not model:
            return config
        provider = config.get("llm", {}).get("provider", "groq")
        cfg = copy.deepcopy(config)
        cfg.setdefault("llm", {}).setdefault("model", {})[provider] = model
        return cfg

    # ── Summary ───────────────────────────────────────────────────────────────

    def log_summary(self) -> None:
        total = self._counts["llm"] + self._counts["skipped"]
        if total == 0:
            return
        pct_saved = round(self._counts["skipped"] / total * 100)
        logger.info(
            "routing | summary total={} llm_called={} skipped={} errors={} cost_savings={}%",
            total,
            self._counts["llm"],
            self._counts["skipped"],
            self._counts["error"],
            pct_saved,
        )

        # Diagnostic: warn if many synthetically-scored jobs had very low
        # reranker scores. With BGE-reranker-base most plausible matches now
        # land at ≥0.30, so persistent sub-0.35 routing scores suggest either
        # a profile/role mismatch or a calibration issue worth investigating.
        scores = self._synthetic_routing_scores
        if len(scores) >= 5:
            below_threshold = sum(1 for s in scores[:5] if s < 0.35)
            if below_threshold >= 3:
                logger.warning(
                    "Reranker scoring many jobs <0.35 — profile may be "
                    "mismatched to scraped roles, or consider fine-tuning "
                    "the reranker on feedback labels."
                )


# ── Multi-provider fallback chain ─────────────────────────────────────────────

class RoutingExhaustedError(Exception):
    """Raised when every provider in the fallback chain has exhausted its run budget."""


# Default model for each provider when used as a fallback.
# Values are bare model names (without the "provider/" prefix).
_FALLBACK_PROVIDER_MODELS: dict[str, str] = {
    "groq":       "llama-3.1-8b-instant",
    "gemini":     "gemini-2.5-flash",
    "cerebras":   "llama-3.3-70b",
    "openai":     "gpt-4o-mini",
    "openrouter": "auto",
    "mistral":    "mistral-small-latest",
}

# For the low tier (uncertain jobs) Gemini's stronger model is preferred.
_FALLBACK_PROVIDER_MODELS_LOW: dict[str, str] = {
    **_FALLBACK_PROVIDER_MODELS,
    "gemini": "gemini-2.5-pro",
}

# Map provider name → environment variable for its API key.
_PROVIDER_API_KEY_ENV: dict[str, str] = {
    "groq":       "GROQ_API_KEY",
    "gemini":     "GEMINI_API_KEY",
    "cerebras":   "CEREBRAS_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "mistral":    "MISTRAL_API_KEY",
}


@dataclass
class ProviderBudget:
    """
    Tracks per-provider LLM usage within a single pipeline run.

    Instantiate once at the start of a run with the rate_limit_budget section
    from the routing config. Call can_afford() before every LLM request and
    spend() immediately after to keep counters accurate.
    """

    _limits: dict[str, dict[str, int]] = field(default_factory=dict)
    _requests_used: dict[str, int] = field(default_factory=dict)
    _tokens_used: dict[str, int] = field(default_factory=dict)
    _fallback_counts: dict[str, int] = field(default_factory=dict)

    def __init__(self, budget_config: dict[str, Any]) -> None:
        self._limits = {}
        self._requests_used = {}
        self._tokens_used = {}
        self._fallback_counts = {}
        for provider, limits in budget_config.items():
            self._limits[provider] = {
                "max_requests": int(limits.get("max_requests_per_run", 999)),
                "max_tokens":   int(limits.get("max_tokens_per_run",   999_999)),
            }
            self._requests_used[provider]  = 0
            self._tokens_used[provider]    = 0
            self._fallback_counts[provider] = 0

    def can_afford(self, provider: str, estimated_tokens: int) -> bool:
        """True if provider has remaining request and token headroom."""
        if provider not in self._limits:
            return True  # unconfigured provider → allow (no cap)
        lim = self._limits[provider]
        return (
            self._requests_used[provider] < lim["max_requests"]
            and self._tokens_used[provider] + estimated_tokens <= lim["max_tokens"]
        )

    def spend(self, provider: str, tokens: int) -> None:
        """Record one request and its estimated token cost."""
        if provider not in self._requests_used:
            self._requests_used[provider] = 0
            self._tokens_used[provider]   = 0
            self._fallback_counts[provider] = 0
        self._requests_used[provider] += 1
        self._tokens_used[provider]   += tokens

    def update_actual(self, provider: str, estimated: int, actual: int) -> None:
        """Correct the token ledger after a call returns its actual usage."""
        if provider in self._tokens_used:
            self._tokens_used[provider] += actual - estimated

    def force_exhaust(self, provider: str) -> None:
        """Mark provider as fully exhausted (e.g. after a 429 response)."""
        if provider in self._limits:
            lim = self._limits[provider]
            self._requests_used[provider] = lim["max_requests"]
            self._tokens_used[provider]   = lim["max_tokens"]
        else:
            # Provider not in budget config — add a sentinel so can_afford returns False.
            self._limits[provider]        = {"max_requests": 0, "max_tokens": 0}
            self._requests_used[provider] = 1
            self._tokens_used[provider]   = 1

    def record_fallback(self, provider: str) -> None:
        """Increment the fallback counter for a provider (used in summary log)."""
        self._fallback_counts[provider] = self._fallback_counts.get(provider, 0) + 1

    def remaining(self, provider: str) -> dict[str, int]:
        lim = self._limits.get(provider, {})
        return {
            "requests_remaining": max(0, lim.get("max_requests", 0) - self._requests_used.get(provider, 0)),
            "tokens_remaining":   max(0, lim.get("max_tokens",   0) - self._tokens_used.get(provider, 0)),
        }

    def log_summary(self) -> None:
        """Log a per-provider budget usage summary at the end of a run."""
        logger.info("Provider budget summary:")
        for provider in self._limits:
            req_used = self._requests_used.get(provider, 0)
            tok_used = self._tokens_used.get(provider, 0)
            max_req  = self._limits[provider]["max_requests"]
            max_tok  = self._limits[provider]["max_tokens"]
            fb_count = self._fallback_counts.get(provider, 0)
            fb_note  = f" (fallback used: {fb_count})" if fb_count else ""
            if req_used == 0:
                logger.info("  {}: 0/{} requests (not needed)", provider, max_req)
            else:
                logger.info(
                    "  {}: {}/{} requests, {:,}/{:,} tokens{}",
                    provider, req_used, max_req, tok_used, max_tok, fb_note,
                )


def _provider_has_key(provider: str, profile: Optional[str] = None) -> bool:
    """Return True if the provider's API key is present in the environment."""
    env_var = _PROVIDER_API_KEY_ENV.get(provider)
    if not env_var:
        return False
    if profile:
        if os.environ.get(f"{env_var}_{profile.upper()}"):
            return True
    return bool(os.environ.get(env_var))


def select_model_with_fallback(
    tier_config: dict[str, Any],
    tier_name: str,
    routing_config: dict[str, Any],
    budgets: ProviderBudget,
    estimated_tokens: int = 500,
    job_id: Optional[Any] = None,
    profile: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Select the best available (provider, model) for a scoring call.

    Starts with the tier's primary_model. If that provider's budget is
    exhausted, walks the fallback_chain until a provider with budget is found.
    Logs a WARNING for every fallback decision so the audit trail is clear.

    Returns:
        (provider, model_name) — bare model name without the "provider/" prefix.

    Raises:
        RoutingExhaustedError — if every provider in the chain is exhausted.
    """
    fallback_models = (
        _FALLBACK_PROVIDER_MODELS_LOW if tier_name == "low"
        else _FALLBACK_PROVIDER_MODELS
    )

    # Parse the tier's primary_model ("provider/model-name")
    primary_model_str = tier_config.get("primary_model", "")
    if "/" in primary_model_str:
        primary_provider, primary_model = primary_model_str.split("/", 1)
    else:
        primary_provider = "groq"
        primary_model    = primary_model_str or fallback_models.get("groq", "llama-3.1-8b-instant")

    # Try primary first
    if (
        _provider_has_key(primary_provider, profile)
        and budgets.can_afford(primary_provider, estimated_tokens)
    ):
        return primary_provider, primary_model

    # Walk the fallback chain
    fallback_chain: list[str] = routing_config.get("fallback_chain", list(fallback_models.keys()))
    exhausted_reason = (
        f"budget exhausted ({budgets._tokens_used.get(primary_provider, 0):,}/"
        f"{budgets._limits.get(primary_provider, {}).get('max_tokens', 0):,} tokens)"
        if budgets._limits.get(primary_provider)
        else "no API key"
    )

    for fallback_provider in fallback_chain:
        if fallback_provider == primary_provider:
            continue  # already tried
        if not _provider_has_key(fallback_provider, profile):
            logger.debug(
                "select_model | skipping {} — no API key found", fallback_provider
            )
            continue
        if not budgets.can_afford(fallback_provider, estimated_tokens):
            logger.debug(
                "select_model | skipping {} — budget exhausted", fallback_provider
            )
            continue

        fallback_model = fallback_models.get(fallback_provider, "")
        if not fallback_model:
            continue

        logger.warning(
            "Falling back from {} to {} for job {}: {} "
            "(estimated tokens for this call: {:,})",
            primary_provider, fallback_provider,
            job_id or "unknown", exhausted_reason, estimated_tokens,
        )
        budgets.record_fallback(fallback_provider)
        return fallback_provider, fallback_model

    # All providers exhausted
    raise RoutingExhaustedError(
        f"All providers in fallback chain exhausted for job {job_id}. "
        f"Primary provider: {primary_provider} ({exhausted_reason}). "
        f"Chain tried: {fallback_chain}"
    )
