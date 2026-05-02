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
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

from db import Job

# ── Cross-encoder model cache (module-level; separate from reranker._MODEL_CACHE) ─

_MODEL_CACHE: dict[str, Any] = {}

_DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# ms-marco-MiniLM-L-6-v2 outputs logits in a compressed range (0.2-0.35 for decent
# matches). 0.18 lets most plausible jobs reach the LLM; title-match boost (+0.15)
# lifts obvious roles over this floor without touching the stored score. Was 0.65.
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
    if model_name not in _MODEL_CACHE:
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
        _MODEL_CACHE[model_name] = model
    return _MODEL_CACHE[model_name]


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
        self._counts: dict[str, int] = {"llm": 0, "skipped": 0, "error": 0}
        self._synthetic_routing_scores: list[float] = []  # for domain suitability diagnostic

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
        Return 0.15 when the job title contains any configured target role.
        Applied to the effective score for the routing/tier decision only —
        never stored in the final score record.
        """
        target_roles = self._config.get("preferences", {}).get("titles", [])
        if not target_roles or not job.title:
            return 0.0
        job_title_lower = job.title.lower().strip()
        for role in target_roles:
            if role.lower().strip() in job_title_lower:
                return 0.15
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

        Returns {'tier': str, 'quality_mode': str, 'max_tokens': int}.
        """
        tiers = self._config.get("routing", {}).get("tiers", _DEFAULT_TIERS)
        complexity = self.compute_complexity_score(job_text)

        high = tiers.get("high", _DEFAULT_TIERS["high"])
        if (
            effective_score >= high.get("min_reranker", 0.40)
            and (not high.get("require_title_match", True) or title_matched)
            and complexity <= 0.7
        ):
            return {
                "tier": "high",
                "quality_mode": high.get("quality_mode", "fast"),
                "max_tokens": high.get("max_tokens", 500),
            }

        medium = tiers.get("medium", _DEFAULT_TIERS["medium"])
        if effective_score >= medium.get("min_reranker", 0.25):
            return {
                "tier": "medium",
                "quality_mode": medium.get("quality_mode", "fast"),
                "max_tokens": medium.get("max_tokens", 700),
            }

        low = tiers.get("low", _DEFAULT_TIERS["low"])
        return {
            "tier": "low",
            "quality_mode": low.get("quality_mode", "quality"),
            "max_tokens": low.get("max_tokens", 900),
        }

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

    def create_synthetic_score(
        self,
        job: Job,
        routing_score: float,
        matched_sections: Optional[list[str]] = None,
        match_reason: str = "",
        title_matched: bool = False,
    ) -> dict[str, Any]:
        """
        Create a conservative score result when the LLM is skipped.

        Uses routing_score as the primary signal and matched_sections to refine
        individual dimension scores:
          - "requirements" matched  → better stack_match estimate
          - "responsibilities" matched → better seniority estimate
          - "summary" matched → confirms role_fit signal

        Dimension scores are intentionally conservative (scaled by 0.85) so that
        LLM-scored jobs always outrank synthetic ones at equal quality.

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

        # Conservative scale: 0.85 keeps synthetic scores visibly below LLM scores
        scale = routing_score * 0.85

        # role_fit: lifted slightly if the summary section matched (confirms general fit)
        role_fit_scale = scale * (1.05 if has_summary else 1.0)
        role_fit = max(0, min(10, round(role_fit_scale * 10)))

        # stack_match: lifted if requirements section matched (skills overlap detected)
        stack_scale = scale * (1.08 if has_requirements else 0.90)
        stack_match = max(0, min(10, round(stack_scale * 10)))

        # seniority: lifted from neutral (5) if responsibilities matched (role context found)
        seniority = 6 if has_responsibilities else 5

        # location / growth / compensation: neutral defaults (cross-encoder can't infer these)
        location = 6
        growth = 5
        compensation = 5

        dims: dict[str, int] = {
            "role_fit": role_fit,
            "stack_match": stack_match,
            "seniority": seniority,
            "location": location,
            "growth": growth,
            "compensation": compensation,
        }

        fit_score = max(0, min(100, int(
            sum(dims[k] * weights.get(k, 0) for k in dims) * 10
        )))
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
            "Fast keyword estimate — not reviewed by AI. Consider a manual check."
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
        """Cheap ATS keyword count without the full compute_ats_score logic."""
        prefs = self._config.get("preferences", {})
        skills = prefs.get("desired_skills", [])
        if not skills or not job.raw_text:
            return 0
        raw_lower = job.raw_text.lower()
        hits = sum(1 for s in skills if s.lower() in raw_lower)
        return min(100, int(hits / max(1, len(skills)) * 100))

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

        # Step 5 diagnostic: warn if many synthetically-scored jobs had very low reranker scores,
        # which suggests the ms-marco model is poorly calibrated for job-matching queries.
        scores = self._synthetic_routing_scores
        if len(scores) >= 5:
            below_threshold = sum(1 for s in scores[:5] if s < 0.35)
            if below_threshold >= 3:
                logger.warning(
                    "Reranker may be undertuned for this domain. "
                    "Consider swapping to cross-encoder/stsb-roberta-base "
                    "or fine-tuning on feedback data."
                )
