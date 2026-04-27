"""
reranker.py - Profile-aware cross-encoder reranking over vector-retrieved job chunks.

This module sits after Chroma-based vector retrieval:
candidate profile/query -> vector recall -> cross-encoder precision -> ranked jobs

It is intentionally deterministic and LLM-free so it can be reused later to
reduce expensive scoring calls.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from loguru import logger

from profile_intent import (
    CREDENTIAL_REQUIREMENT_PATTERNS,
    ProfileIntent,
    get_role_family_section_weights,
    normalize_profile_intent,
)
from vector_store import (
    VectorChunkResult,
    VectorJobResult,
    query_similar_chunks,
    query_similar_jobs,
    vector_store_enabled,
)

DEFAULT_RERANKING_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_BATCH_SIZE = 16
DEFAULT_TOP_K_VECTOR_CHUNKS = 80
DEFAULT_TOP_K_VECTOR_JOBS = 40
DEFAULT_TOP_K_FINAL = 15
DEFAULT_MAX_CHUNKS_PER_JOB = 4
DEFAULT_MAX_CHUNK_CHARS = 900
DEFAULT_INCLUDE_PROFILE_CONTEXT = True
DEFAULT_INCLUDE_QUERY_CONTEXT = True

SECTION_PRIORITY = {
    "requirements": 5,
    "responsibilities": 4,
    "summary": 3,
    "compensation": 2,
    "benefits": 1,
}

# Base section weights (role-family aware weights from profile_intent override these per job).
SECTION_WEIGHTS = {
    "requirements": 1.15,
    "responsibilities": 1.10,
    "summary": 1.00,
    "compensation": 0.90,
    "benefits": 0.80,
}

COMPENSATION_TERMS = (
    "salary",
    "compensation",
    "pay",
    "equity",
    "bonus",
    "stipend",
    "base",
    "ote",
    "hourly",
)
BENEFITS_TERMS = (
    "benefits",
    "perk",
    "perks",
    "pto",
    "401k",
    "medical",
    "healthcare",
    "dental",
    "vision",
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "backend",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "job",
    "of",
    "on",
    "or",
    "role",
    "seeking",
    "systems",
    "the",
    "to",
    "with",
}

_SENIOR_TITLE_SIGNALS = (
    "senior", "sr.", "staff", "principal", "lead", "director",
    "vp ", "head of", "manager", "executive",
)

_MODEL_CACHE: dict[str, Any] = {}


@dataclass
class RerankedChunkResult:
    job_id: str
    chunk_key: str
    chunk_order: int
    title: str
    company: str
    source: str
    url: str
    vector_similarity: float
    cross_encoder_score: float
    normalized_score: float
    weighted_score: float
    chunk_text: str


@dataclass
class MatchEvidence:
    """Deterministic positive and negative signals extracted from reranked job chunks."""
    positive: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    missing_or_unclear: list[str] = field(default_factory=list)


@dataclass
class RerankedJobResult:
    job_id: str
    title: str
    company: str
    source: str
    url: str
    vector_score: float
    rerank_score: float
    final_score: float
    matched_sections: list[str] = field(default_factory=list)
    section_scores: dict[str, float] = field(default_factory=dict)
    best_chunk_key: str = ""
    match_reason: str = ""
    evidence_snippets: list[str] = field(default_factory=list)
    evidence: MatchEvidence = field(default_factory=MatchEvidence)


@dataclass
class MatchIntent:
    compensation: bool = False
    benefits: bool = False


def reranking_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("reranking", {}).get("enabled", True))


def reranking_model_name(config: dict[str, Any]) -> str:
    raw = str(config.get("reranking", {}).get("model", DEFAULT_RERANKING_MODEL)).strip()
    return raw or DEFAULT_RERANKING_MODEL


def reranking_batch_size(config: dict[str, Any]) -> int:
    raw = config.get("reranking", {}).get("batch_size", DEFAULT_BATCH_SIZE)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_BATCH_SIZE
    return max(1, min(64, value))


def reranking_top_k_vector_chunks(config: dict[str, Any]) -> int:
    raw = config.get("reranking", {}).get("top_k_vector_chunks", DEFAULT_TOP_K_VECTOR_CHUNKS)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_TOP_K_VECTOR_CHUNKS
    return max(1, min(200, value))


def reranking_top_k_vector_jobs(config: dict[str, Any]) -> int:
    raw = config.get("reranking", {}).get("top_k_vector_jobs", DEFAULT_TOP_K_VECTOR_JOBS)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_TOP_K_VECTOR_JOBS
    return max(1, min(100, value))


def reranking_top_k_final(config: dict[str, Any]) -> int:
    raw = config.get("reranking", {}).get("top_k_final", DEFAULT_TOP_K_FINAL)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_TOP_K_FINAL
    return max(1, min(100, value))


def reranking_max_chunks_per_job(config: dict[str, Any]) -> int:
    raw = config.get("reranking", {}).get("max_chunks_per_job", DEFAULT_MAX_CHUNKS_PER_JOB)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_MAX_CHUNKS_PER_JOB
    return max(1, min(8, value))


def reranking_max_chunk_chars(config: dict[str, Any]) -> int:
    raw = config.get("reranking", {}).get("max_chunk_chars", DEFAULT_MAX_CHUNK_CHARS)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_MAX_CHUNK_CHARS
    return max(200, min(4000, value))


def include_profile_context(config: dict[str, Any]) -> bool:
    return bool(
        config.get("reranking", {}).get("include_profile_context", DEFAULT_INCLUDE_PROFILE_CONTEXT)
    )


def include_query_context(config: dict[str, Any]) -> bool:
    return bool(
        config.get("reranking", {}).get("include_query_context", DEFAULT_INCLUDE_QUERY_CONTEXT)
    )


def _load_cross_encoder(model_name: str) -> Any:
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import CrossEncoder

        logger.info("reranker | loading cross-encoder {}", model_name)
        model = CrossEncoder(model_name, device="cpu")
        if hasattr(model, "max_length"):
            model.max_length = 512
        _MODEL_CACHE[model_name] = model
    return _MODEL_CACHE[model_name]


def get_cross_encoder(config: dict[str, Any]) -> Any:
    return _load_cross_encoder(reranking_model_name(config))


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        lowered = cleaned.lower()
        if not cleaned or lowered in seen:
            continue
        seen.add(lowered)
        result.append(cleaned)
    return result


def build_profile_match_query(config: dict[str, Any], user_query: str | None = None) -> str:
    """Build a concise, role-agnostic match query from the candidate's ProfileIntent."""
    intent = normalize_profile_intent(config)

    segments: list[str] = []
    if include_profile_context(config):
        role_label = "internship" if intent.job_type == "internship" else "full-time"
        role_text = ", ".join(intent.target_roles[:4]) if intent.target_roles else "relevant"
        segments.append(f"Candidate seeking {role_label} {role_text} roles.")

        if intent.years_experience:
            segments.append(f"Experience: {intent.years_experience:.0f} years.")

        all_skills = _dedupe_preserve_order(intent.domain_skills + intent.tools)[:8]
        if all_skills:
            segments.append(f"Skills: {', '.join(all_skills)}.")

        if intent.credentials:
            segments.append(f"Credentials: {', '.join(intent.credentials[:3])}.")

        preference_parts: list[str] = []
        if intent.remote_ok and intent.locations:
            preference_parts.append(f"remote or {', '.join(intent.locations[:2])}")
        elif intent.remote_ok:
            preference_parts.append("remote-friendly")
        elif intent.locations:
            preference_parts.append(", ".join(intent.locations[:2]))

        if intent.job_type == "internship":
            pay_pref = intent.compensation.get("intern_pay_preference", "")
            if pay_pref == "paid_only":
                stipend = intent.compensation.get("monthly_stipend")
                if stipend:
                    preference_parts.append(f"paid internships ${int(stipend):,}/mo+")
                else:
                    preference_parts.append("paid internships only")
        else:
            min_salary = intent.compensation.get("min_salary")
            if min_salary:
                preference_parts.append(f"compensation ${int(min_salary):,}+")

        if preference_parts:
            segments.append(f"Preferences: {', '.join(preference_parts)}.")

        if intent.dealbreakers:
            avoid_parts = _dedupe_preserve_order(
                [db.replace("required", "").strip() for db in intent.dealbreakers[:4]]
            )
            avoid_parts = [p for p in avoid_parts if p]
            if avoid_parts:
                segments.append(f"Avoid: {', '.join(avoid_parts[:4])}.")

    if include_query_context(config) and user_query and user_query.strip():
        segments.append(f"User intent: {user_query.strip()}.")

    return " ".join(segment.strip() for segment in segments if segment.strip())


def _build_vector_recall_query(config: dict[str, Any]) -> str:
    """Build a compact token-based query for initial ANN recall."""
    intent = normalize_profile_intent(config)
    tokens: list[str] = []
    tokens.extend(intent.target_roles[:4])
    tokens.extend((intent.domain_skills + intent.tools)[:8])
    if intent.remote_ok:
        tokens.append("remote")
    tokens.extend(intent.locations[:2])
    if intent.job_type == "internship":
        tokens.append("internship")
    else:
        tokens.append("full-time")
    return " ".join(_dedupe_preserve_order(tokens))


def _detect_match_intent(match_query: str) -> MatchIntent:
    lowered = match_query.lower()
    return MatchIntent(
        compensation=any(term in lowered for term in COMPENSATION_TERMS),
        benefits=any(term in lowered for term in BENEFITS_TERMS),
    )


def _section_weight(chunk_key: str, intent: MatchIntent, role_family: str = "general") -> float:
    if chunk_key == "compensation" and intent.compensation:
        return 1.05
    if chunk_key == "benefits" and (intent.compensation or intent.benefits):
        return 1.00
    family_weights = get_role_family_section_weights(role_family)
    return family_weights.get(chunk_key, SECTION_WEIGHTS.get(chunk_key, 1.0))


def _section_priority(chunk_key: str, intent: MatchIntent) -> int:
    if chunk_key == "compensation" and intent.compensation:
        return 3
    if chunk_key == "benefits" and intent.benefits:
        return 3
    return SECTION_PRIORITY.get(chunk_key, 0)


def _dedupe_chunk_results(chunk_results: Sequence[VectorChunkResult]) -> list[VectorChunkResult]:
    best_by_key: dict[tuple[str, str], VectorChunkResult] = {}
    for chunk in chunk_results:
        key = (chunk.job_id, chunk.chunk_key)
        current = best_by_key.get(key)
        if current is None or chunk.similarity > current.similarity:
            best_by_key[key] = chunk
    return list(best_by_key.values())


def select_chunks_for_reranking(
    job_result: VectorJobResult,
    chunk_results: Sequence[VectorChunkResult],
    max_chunks_per_job: int,
    *,
    match_query: str | None = None,
) -> list[VectorChunkResult]:
    if max_chunks_per_job <= 0:
        return []

    intent = _detect_match_intent(match_query or "")
    candidates = [
        chunk for chunk in _dedupe_chunk_results(chunk_results) if chunk.job_id == job_result.job_id
    ]
    if not candidates:
        return []

    sorted_candidates = sorted(
        candidates,
        key=lambda item: (
            _section_priority(item.chunk_key, intent),
            item.similarity,
            -item.chunk_order,
        ),
        reverse=True,
    )

    selected: list[VectorChunkResult] = []
    selected_keys: set[str] = set()

    def _maybe_take(chunk_key: str) -> None:
        if len(selected) >= max_chunks_per_job:
            return
        for chunk in sorted_candidates:
            if chunk.chunk_key == chunk_key and chunk.chunk_key not in selected_keys:
                selected.append(chunk)
                selected_keys.add(chunk.chunk_key)
                return

    for required_key in ("requirements", "responsibilities", "summary"):
        _maybe_take(required_key)

    if intent.compensation:
        _maybe_take("compensation")
    if intent.benefits:
        _maybe_take("benefits")

    for chunk in sorted_candidates:
        if len(selected) >= max_chunks_per_job:
            break
        if chunk.chunk_key in selected_keys:
            continue
        if chunk.chunk_key == "benefits" and not intent.benefits and chunk.similarity < 0.55:
            continue
        selected.append(chunk)
        selected_keys.add(chunk.chunk_key)

    return selected[:max_chunks_per_job]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _prepare_chunk_text(chunk: VectorChunkResult, max_chunk_chars: int) -> str:
    text = chunk.chunk_text.strip()
    if len(text) <= max_chunk_chars:
        return text

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    prefix_lines = [line for line in lines[:4] if ":" in line]
    body_lines = [line for line in lines if line not in prefix_lines]
    prefix = "\n".join(prefix_lines)
    prefix_budget = min(len(prefix), max_chunk_chars // 3)
    remaining = max_chunk_chars - prefix_budget - 1
    body = " ".join(body_lines)
    truncated_body = body[:remaining].rsplit(" ", 1)[0].strip() if remaining > 0 else ""
    if prefix:
        return f"{prefix}\n{truncated_body}".strip()
    return body[:max_chunk_chars].rsplit(" ", 1)[0].strip()


def rerank_chunks(
    match_query: str,
    chunk_results: list[VectorChunkResult],
    config: dict[str, Any],
) -> list[RerankedChunkResult]:
    if not chunk_results:
        return []

    intent = _detect_match_intent(match_query)
    role_family = normalize_profile_intent(config).role_family
    max_chunk_chars = reranking_max_chunk_chars(config)
    model = get_cross_encoder(config)
    pairs = [(match_query, _prepare_chunk_text(chunk, max_chunk_chars)) for chunk in chunk_results]
    started = time.perf_counter()
    raw_scores = model.predict(
        pairs,
        batch_size=reranking_batch_size(config),
        show_progress_bar=False,
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "reranker | rerank_chunks model={} batch_size={} chunks={} latency_ms={}",
        reranking_model_name(config),
        reranking_batch_size(config),
        len(chunk_results),
        latency_ms,
    )

    reranked: list[RerankedChunkResult] = []
    for chunk, raw_score in zip(chunk_results, raw_scores):
        raw_value = float(raw_score)
        normalized = _sigmoid(raw_value)
        weighted = min(1.0, normalized * _section_weight(chunk.chunk_key, intent, role_family))
        reranked.append(
            RerankedChunkResult(
                job_id=chunk.job_id,
                chunk_key=chunk.chunk_key,
                chunk_order=chunk.chunk_order,
                title=chunk.title,
                company=chunk.company,
                source=chunk.source,
                url=chunk.url,
                vector_similarity=round(chunk.similarity, 4),
                cross_encoder_score=round(raw_value, 4),
                normalized_score=round(normalized, 4),
                weighted_score=round(weighted, 4),
                chunk_text=chunk.chunk_text,
            )
        )

    return sorted(
        reranked,
        key=lambda item: (item.weighted_score, item.normalized_score, item.vector_similarity),
        reverse=True,
    )


def _coverage_bonus(section_scores: dict[str, float]) -> float:
    bonus = 0.0
    if section_scores.get("requirements", 0.0) >= 0.55 and section_scores.get("responsibilities", 0.0) >= 0.55:
        bonus += 0.07
    if section_scores.get("summary", 0.0) >= 0.50:
        bonus += 0.03
    return min(0.10, bonus)


def _query_terms(match_query: str) -> list[str]:
    found = re.findall(r"[A-Za-z][A-Za-z0-9+#./-]{2,}", match_query)
    return [term for term in found if term.lower() not in STOPWORDS]


def _extract_evidence_terms(match_query: str, chunks: Sequence[RerankedChunkResult]) -> list[str]:
    combined = " ".join(chunk.chunk_text.lower() for chunk in chunks[:3])
    matches: list[str] = []
    for term in _query_terms(match_query):
        if term.lower() in combined and term.lower() not in {value.lower() for value in matches}:
            matches.append(term)
        if len(matches) >= 5:
            break
    return matches


def _snippet_for_chunk(chunk_text: str, terms: Sequence[str], max_chars: int = 160) -> str:
    text = " ".join(line.strip() for line in chunk_text.splitlines() if line.strip())
    lowered = text.lower()
    for term in terms:
        idx = lowered.find(term.lower())
        if idx >= 0:
            start = max(0, idx - 45)
            end = min(len(text), idx + max_chars - 45)
            snippet = text[start:end].strip()
            return f"...{snippet}..." if start > 0 or end < len(text) else snippet
    snippet = text[:max_chars].strip()
    return f"{snippet}..." if len(text) > max_chars else snippet


def _extract_match_evidence(
    intent: ProfileIntent,
    job_title: str,
    chunks: Sequence[RerankedChunkResult],
    evidence_terms: Sequence[str],
) -> MatchEvidence:
    """Extract deterministic positive and negative signals from reranked chunks."""
    combined = " ".join(chunk.chunk_text.lower() for chunk in chunks)
    job_title_lower = job_title.lower()

    positive: list[str] = []
    concerns: list[str] = []
    matched_keywords: list[str] = list(evidence_terms[:8])
    missing_or_unclear: list[str] = []

    # Role title match
    for target_role in intent.target_roles[:3]:
        role_words = [w.lower() for w in target_role.split() if len(w) > 2]
        if role_words and any(w in job_title_lower for w in role_words):
            positive.append(f"Title match: {target_role}")
            break

    # Skill/tool match in chunk text
    all_skills = _dedupe_preserve_order(intent.domain_skills + intent.tools)[:12]
    found_skills = [s for s in all_skills if s.lower() in combined]
    if found_skills:
        positive.append(f"Skills present: {', '.join(found_skills[:5])}")
        for s in found_skills[:5]:
            if s.lower() not in {k.lower() for k in matched_keywords}:
                matched_keywords.append(s)

    # Location / remote match
    if intent.remote_ok and "remote" in combined:
        positive.append("Remote-friendly posting")
    else:
        for loc in intent.locations[:2]:
            if loc.lower() in combined:
                positive.append(f"Location match: {loc}")
                break

    # Industry match
    for industry in intent.industries[:2]:
        if industry.lower() in combined:
            positive.append(f"Industry match: {industry}")
            break

    # Seniority mismatch check
    if intent.seniority in ("entry", "mid", "intern"):
        for signal in _SENIOR_TITLE_SIGNALS:
            if signal in job_title_lower:
                concerns.append(f"Seniority concern: posting appears {signal.strip()}-level")
                break

    # Security clearance requirement
    clearance_patterns = ("security clearance", "secret clearance", "top secret", "ts/sci", "clearance required")
    if any(pat in combined for pat in clearance_patterns):
        concerns.append("Clearance requirement detected")

    # Work authorization / no sponsorship
    no_sponsor_patterns = ("no sponsorship", "no visa", "must be authorized", "us citizen only")
    if any(pat in combined for pat in no_sponsor_patterns):
        concerns.append("Work authorization requirement detected")

    # Credential requirements not in profile
    for cred_label, patterns in CREDENTIAL_REQUIREMENT_PATTERNS.items():
        if any(pat in combined for pat in patterns):
            profile_has = any(cred_label.lower() in c.lower() for c in intent.credentials)
            if not profile_has:
                concerns.append(f"Credential may be required: {cred_label}")
                missing_or_unclear.append(cred_label)

    # Unpaid internship check
    if intent.job_type == "internship":
        pay_pref = intent.compensation.get("intern_pay_preference", "no_preference")
        if pay_pref == "paid_only" and "unpaid" in combined:
            concerns.append("Posting may be unpaid (profile requires paid internship)")

    # Location mismatch for non-remote profiles
    if not intent.remote_ok and intent.locations:
        onsite_signals = ("must be located", "must reside", "in-office", "no remote")
        if any(sig in combined for sig in onsite_signals):
            if not any(loc.lower() in combined for loc in intent.locations):
                concerns.append("Location may not match profile preferences")

    return MatchEvidence(
        positive=positive[:5],
        concerns=concerns[:5],
        matched_keywords=_dedupe_preserve_order(matched_keywords)[:8],
        missing_or_unclear=missing_or_unclear[:4],
    )


def _match_reason(
    section_scores: dict[str, float],
    evidence_terms: Sequence[str],
    intent: Optional[ProfileIntent] = None,
) -> str:
    strong_sections = [
        key for key in ("requirements", "responsibilities", "summary")
        if section_scores.get(key, 0.0) >= 0.55
    ]
    if not strong_sections:
        return "Moderate semantic match from supporting job sections."

    terms_str = ", ".join(evidence_terms[:4]) if evidence_terms else ""

    if len(strong_sections) >= 2:
        section_label = f"{strong_sections[0]} and {strong_sections[1]}"
    else:
        section_label = strong_sections[0]

    if terms_str:
        return f"Strong {section_label} match for {terms_str}."
    return f"Strong match in {section_label}."


def _fallback_from_vector_results(vector_job_results: Sequence[VectorJobResult]) -> list[RerankedJobResult]:
    results: list[RerankedJobResult] = []
    for job in vector_job_results:
        section_scores = {
            chunk.chunk_key: chunk.weighted_similarity
            for chunk in job.chunk_scores[:5]
        }
        results.append(
            RerankedJobResult(
                job_id=job.job_id,
                title=job.title,
                company=job.company,
                source=job.source,
                url=job.url,
                vector_score=round(job.aggregate_score, 4),
                rerank_score=round(job.aggregate_score, 4),
                final_score=round(job.aggregate_score, 4),
                matched_sections=list(job.matched_chunks),
                section_scores=section_scores,
                best_chunk_key=job.matched_chunks[0] if job.matched_chunks else "",
                match_reason=job.retrieval_reason,
                evidence_snippets=[],
                evidence=MatchEvidence(),
            )
        )
    return results


def rerank_jobs(
    match_query: str,
    vector_job_results: list[VectorJobResult],
    chunk_results: list[VectorChunkResult],
    config: dict[str, Any],
) -> list[RerankedJobResult]:
    if not vector_job_results or not chunk_results:
        return []

    if not reranking_enabled(config):
        return _fallback_from_vector_results(vector_job_results)

    profile_intent = normalize_profile_intent(config)

    selected_chunks: list[VectorChunkResult] = []
    for job in vector_job_results:
        selected_chunks.extend(
            select_chunks_for_reranking(
                job,
                chunk_results,
                reranking_max_chunks_per_job(config),
                match_query=match_query,
            )
        )

    if not selected_chunks:
        return _fallback_from_vector_results(vector_job_results)

    reranked_chunks = rerank_chunks(match_query, selected_chunks, config)
    chunks_by_job: dict[str, list[RerankedChunkResult]] = {}
    for chunk in reranked_chunks:
        chunks_by_job.setdefault(chunk.job_id, []).append(chunk)

    ranked_jobs: list[RerankedJobResult] = []
    for vector_job in vector_job_results:
        job_chunks = chunks_by_job.get(vector_job.job_id, [])
        if not job_chunks:
            continue
        ordered = sorted(job_chunks, key=lambda item: item.weighted_score, reverse=True)
        section_scores = {chunk.chunk_key: chunk.weighted_score for chunk in ordered}
        weighted_scores = [chunk.weighted_score for chunk in ordered]
        best = weighted_scores[0]
        top_average = sum(weighted_scores[:3]) / min(3, len(weighted_scores))
        rerank_score = min(1.0, best * 0.55 + top_average * 0.30 + _coverage_bonus(section_scores))
        final_score = min(1.0, rerank_score + vector_job.aggregate_score * 0.05)
        evidence_terms = _extract_evidence_terms(match_query, ordered)
        evidence_snippets = [_snippet_for_chunk(chunk.chunk_text, evidence_terms) for chunk in ordered[:2]]
        evidence = _extract_match_evidence(profile_intent, vector_job.title, ordered, evidence_terms)
        ranked_jobs.append(
            RerankedJobResult(
                job_id=vector_job.job_id,
                title=vector_job.title,
                company=vector_job.company,
                source=vector_job.source,
                url=vector_job.url,
                vector_score=round(vector_job.aggregate_score, 4),
                rerank_score=round(rerank_score, 4),
                final_score=round(final_score, 4),
                matched_sections=[chunk.chunk_key for chunk in ordered[:4]],
                section_scores={key: round(value, 4) for key, value in section_scores.items()},
                best_chunk_key=ordered[0].chunk_key,
                match_reason=_match_reason(section_scores, evidence_terms, profile_intent),
                evidence_snippets=evidence_snippets,
                evidence=evidence,
            )
        )

    ranked_jobs = sorted(
        ranked_jobs,
        key=lambda item: (item.final_score, item.rerank_score, item.vector_score),
        reverse=True,
    )[: reranking_top_k_final(config)]
    top_ids = [job.job_id for job in ranked_jobs[:5]]
    logger.info(
        "reranker | rerank_jobs candidates={} selected_chunks={} reranked_jobs={} top_ids={}",
        len(vector_job_results),
        len(selected_chunks),
        len(ranked_jobs),
        top_ids,
    )
    return ranked_jobs


def semantic_match_jobs(
    profile: str,
    config: dict[str, Any],
    user_query: str | None = None,
) -> list[RerankedJobResult]:
    if not vector_store_enabled(config):
        logger.info("reranker | semantic match skipped because vector store is disabled for profile={}", profile)
        return []

    started = time.perf_counter()
    match_query = build_profile_match_query(config, user_query=user_query)
    vector_query = user_query.strip() if user_query and user_query.strip() else _build_vector_recall_query(config)
    vector_job_results = query_similar_jobs(
        profile,
        vector_query,
        top_k_chunks=reranking_top_k_vector_chunks(config),
        top_k_jobs=reranking_top_k_vector_jobs(config),
    )
    if not vector_job_results:
        return []

    if not reranking_enabled(config):
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "reranker | semantic match fallback profile={} query_provided={} vector_jobs={} latency_ms={}",
            profile,
            bool(user_query and user_query.strip()),
            len(vector_job_results),
            latency_ms,
        )
        return _fallback_from_vector_results(vector_job_results)[: reranking_top_k_final(config)]

    chunk_results = query_similar_chunks(
        profile,
        vector_query,
        top_k_chunks=reranking_top_k_vector_chunks(config),
    )
    reranked = rerank_jobs(match_query, vector_job_results, chunk_results, config)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "reranker | semantic match profile={} query_provided={} vector_candidates={} selected_chunks={} reranked_jobs={} model={} batch_size={} latency_ms={} top_ids={}",
        profile,
        bool(user_query and user_query.strip()),
        len(vector_job_results),
        len(chunk_results),
        len(reranked),
        reranking_model_name(config),
        reranking_batch_size(config),
        latency_ms,
        [job.job_id for job in reranked[:5]],
    )
    return reranked


def select_jobs_for_llm_scoring(
    reranked_results: Sequence[RerankedJobResult],
    threshold: float,
    top_k: int,
) -> list[str]:
    if top_k <= 0:
        return []
    eligible = [result for result in reranked_results if result.final_score >= threshold]
    ranked = sorted(eligible, key=lambda item: item.final_score, reverse=True)
    return [result.job_id for result in ranked[:top_k]]
