"""
match_explainer.py - Deterministic factor-wise explanations for semantic job matches.

This module combines existing semantic evidence, reranker outputs, and stored
LLM score dimensions into reusable explanations for both CLI and dashboard use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from profile_intent import ProfileIntent
from reranker import RerankedJobResult

FactorStatus = Literal["strong", "medium", "weak", "concern", "unknown"]

_ROLE_LABELS = {
    "software_engineering": "engineering stack",
    "data_analytics": "analytics toolkit",
    "finance": "finance toolkit",
    "product_management": "product toolkit",
    "operations": "operations toolkit",
    "sales": "sales toolkit",
    "marketing": "marketing toolkit",
    "design": "design toolkit",
    "internship": "internship fit",
    "general": "core skills",
}


@dataclass
class FactorExplanation:
    name: str
    status: FactorStatus
    score: float | int | None
    evidence: list[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class MatchExplanation:
    job_id: str
    title: str
    company: str
    overall_score: float | None
    semantic_score: float | None
    rerank_score: float | None
    llm_fit_score: int | None
    ats_score: int | None
    recommended_action: str
    summary: str
    strengths: list[FactorExplanation] = field(default_factory=list)
    concerns: list[FactorExplanation] = field(default_factory=list)
    unknowns: list[FactorExplanation] = field(default_factory=list)
    matched_sections: list[str] = field(default_factory=list)
    evidence_snippets: list[str] = field(default_factory=list)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(cleaned)
    return result


def _dimension_score(score_record: Optional[dict], key: str) -> int | None:
    if not score_record:
        return None
    dims = score_record.get("dimension_scores") or {}
    value = dims.get(key, score_record.get(key))
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _status_from_score(score: int | None, *, high: int = 8, medium: int = 6, low: int = 4) -> FactorStatus:
    if score is None:
        return "unknown"
    if score >= high:
        return "strong"
    if score >= medium:
        return "medium"
    if score >= low:
        return "weak"
    return "concern"


def _overall_score(score_record: Optional[dict], reranked_result: Optional[RerankedJobResult]) -> float | None:
    if reranked_result is not None and reranked_result.final_score is not None:
        return round(float(reranked_result.final_score) * 100, 1)
    if score_record and score_record.get("fit_score") is not None:
        return float(score_record["fit_score"])
    return None


def explain_score_dimensions(score_record: Optional[dict]) -> list[FactorExplanation]:
    if not score_record:
        return []

    reasons = list(score_record.get("reasons") or [])
    flags = list(score_record.get("flags") or [])
    skill_misses = list(score_record.get("skill_misses") or [])
    factors = [
        FactorExplanation(
            name="Role fit",
            status=_status_from_score(_dimension_score(score_record, "role_fit")),
            score=_dimension_score(score_record, "role_fit"),
            evidence=reasons[:2],
            explanation="Stored role-fit score from the LLM scoring pass.",
        ),
        FactorExplanation(
            name="Skills/tools match",
            status=_status_from_score(_dimension_score(score_record, "stack_match")),
            score=_dimension_score(score_record, "stack_match"),
            evidence=(reasons + skill_misses[:2])[:3],
            explanation="Combines the LLM stack/skills dimension with saved skill gaps.",
        ),
        FactorExplanation(
            name="Seniority fit",
            status=_status_from_score(_dimension_score(score_record, "seniority")),
            score=_dimension_score(score_record, "seniority"),
            evidence=flags[:2],
            explanation="How well the posting level lines up with the profile's target seniority.",
        ),
        FactorExplanation(
            name="Location/remote fit",
            status=_status_from_score(_dimension_score(score_record, "location")),
            score=_dimension_score(score_record, "location"),
            evidence=[flag for flag in flags if "location" in flag.lower()][:2],
            explanation="Uses the saved location dimension and any explicit location watchouts.",
        ),
        FactorExplanation(
            name="Growth/company fit",
            status=_status_from_score(_dimension_score(score_record, "growth")),
            score=_dimension_score(score_record, "growth"),
            evidence=reasons[2:4],
            explanation="Reflects the company/growth dimension from the scoring pass.",
        ),
        FactorExplanation(
            name="Compensation fit",
            status=_status_from_score(_dimension_score(score_record, "compensation")),
            score=_dimension_score(score_record, "compensation"),
            evidence=[flag for flag in flags if any(token in flag.lower() for token in ("salary", "compensation", "pay"))][:2],
            explanation="Based on the saved compensation dimension and any comp-related warnings.",
        ),
        FactorExplanation(
            name="ATS/resume keyword fit",
            status=_status_from_score(score_record.get("ats_score"), high=80, medium=65, low=45),
            score=score_record.get("ats_score"),
            evidence=skill_misses[:3],
            explanation="ATS score and saved missing-skill notes estimate resume keyword coverage.",
        ),
    ]
    return [factor for factor in factors if factor.score is not None or factor.evidence]


def explain_semantic_evidence(
    reranked_result: Optional[RerankedJobResult],
    profile_intent: ProfileIntent,
) -> list[FactorExplanation]:
    if reranked_result is None:
        return []

    factors: list[FactorExplanation] = []
    evidence = reranked_result.evidence
    matched_sections = list(reranked_result.matched_sections or [])
    matched_keywords = list(getattr(evidence, "matched_keywords", []) or [])
    positive = list(getattr(evidence, "positive", []) or [])

    role_fit_status: FactorStatus = "strong" if reranked_result.final_score >= 0.78 else "medium" if reranked_result.final_score >= 0.62 else "weak"
    factors.append(
        FactorExplanation(
            name="Role fit",
            status=role_fit_status,
            score=round(reranked_result.final_score * 100),
            evidence=_dedupe(positive[:2] + [reranked_result.match_reason]),
            explanation="Semantic retrieval and reranking both point to a close role match.",
        )
    )

    skill_label = _ROLE_LABELS.get(profile_intent.role_family, "core skills").title()
    skill_evidence = [item for item in positive if "skill" in item.lower() or "title match" in item.lower()]
    skill_evidence.extend(matched_keywords[:3])
    factors.append(
        FactorExplanation(
            name="Skills/tools match",
            status="strong" if skill_evidence else "medium" if matched_sections else "unknown",
            score=round(reranked_result.rerank_score * 100),
            evidence=_dedupe(skill_evidence)[:4],
            explanation=f"Semantic evidence lines up with the profile's {skill_label.lower()}.",
        )
    )

    if "responsibilities" in matched_sections:
        factors.append(
            FactorExplanation(
                name="Responsibilities match",
                status="strong" if reranked_result.section_scores.get("responsibilities", 0.0) >= 0.6 else "medium",
                score=round(reranked_result.section_scores.get("responsibilities", 0.0) * 100),
                evidence=[reranked_result.match_reason],
                explanation="The responsibilities section contains strong overlap with the target profile.",
            )
        )
    if "requirements" in matched_sections:
        factors.append(
            FactorExplanation(
                name="Requirements match",
                status="strong" if reranked_result.section_scores.get("requirements", 0.0) >= 0.6 else "medium",
                score=round(reranked_result.section_scores.get("requirements", 0.0) * 100),
                evidence=matched_keywords[:4],
                explanation="Requirements text matched the candidate intent closely enough to survive reranking.",
            )
        )

    if any("remote" in item.lower() or "location" in item.lower() for item in positive):
        factors.append(
            FactorExplanation(
                name="Location/remote fit",
                status="strong",
                score=None,
                evidence=[item for item in positive if "remote" in item.lower() or "location" in item.lower()][:2],
                explanation="Semantic evidence includes a direct remote or location alignment signal.",
            )
        )

    if "compensation" in matched_sections:
        factors.append(
            FactorExplanation(
                name="Compensation fit",
                status="medium",
                score=round(reranked_result.section_scores.get("compensation", 0.0) * 100),
                evidence=["Compensation section was part of the semantic match."],
                explanation="Pay details were surfaced in matched sections, which can support review decisions.",
            )
        )

    if profile_intent.credentials and not getattr(evidence, "missing_or_unclear", []):
        factors.append(
            FactorExplanation(
                name="Credentials/certifications",
                status="medium",
                score=None,
                evidence=profile_intent.credentials[:3],
                explanation="The profile includes credentials and no credential mismatch was detected in semantic evidence.",
            )
        )

    return factors


def explain_concerns(
    reranked_result: Optional[RerankedJobResult],
    profile_intent: ProfileIntent,
) -> list[FactorExplanation]:
    concerns: list[FactorExplanation] = []
    if reranked_result is None:
        return concerns

    evidence = reranked_result.evidence
    concern_items = list(getattr(evidence, "concerns", []) or [])
    missing_or_unclear = list(getattr(evidence, "missing_or_unclear", []) or [])

    for item in concern_items:
        lowered = item.lower()
        name = "Dealbreakers/concerns"
        if "senior" in lowered or "seniority" in lowered:
            name = "Seniority fit"
        elif "credential" in lowered:
            name = "Credentials/certifications"
        elif "location" in lowered or "remote" in lowered:
            name = "Location/remote fit"
        elif "clearance" in lowered or "authorization" in lowered:
            name = "Dealbreakers/concerns"
        concerns.append(
            FactorExplanation(
                name=name,
                status="concern",
                score=None,
                evidence=[item],
                explanation=item,
            )
        )

    if missing_or_unclear:
        concerns.append(
            FactorExplanation(
                name="Credentials/certifications",
                status="concern",
                score=None,
                evidence=missing_or_unclear[:3],
                explanation="The job appears to require credentials that are not confirmed in the profile.",
            )
        )

    if profile_intent.job_type == "internship" and not any("paid" in item.lower() for item in concern_items):
        pay_pref = str(profile_intent.compensation.get("intern_pay_preference", "")).lower()
        if pay_pref == "paid_only" and "compensation" not in reranked_result.matched_sections:
            concerns.append(
                FactorExplanation(
                    name="Compensation fit",
                    status="unknown",
                    score=None,
                    evidence=["Paid status was not surfaced in the matched sections."],
                    explanation="Internship compensation may need manual review before applying.",
                )
            )

    return concerns


def _build_unknowns(
    score_record: Optional[dict],
    reranked_result: Optional[RerankedJobResult],
    profile_intent: ProfileIntent,
) -> list[FactorExplanation]:
    unknowns: list[FactorExplanation] = []
    if score_record is None:
        unknowns.append(
            FactorExplanation(
                name="LLM score coverage",
                status="unknown",
                score=None,
                evidence=["No stored fit score is available for this job yet."],
                explanation="Semantic evidence exists, but the LLM scoring pass has not populated dimension scores.",
            )
        )
    if reranked_result is None:
        unknowns.append(
            FactorExplanation(
                name="Semantic evidence",
                status="unknown",
                score=None,
                evidence=["No reranked semantic result was provided."],
                explanation="Only stored score data is available, so section-level match evidence is limited.",
            )
        )
    elif profile_intent.remote_ok and "Location/remote fit" not in {factor.name for factor in explain_semantic_evidence(reranked_result, profile_intent)}:
        unknowns.append(
            FactorExplanation(
                name="Location/remote fit",
                status="unknown",
                score=None,
                evidence=["Remote or preferred location language did not appear in the strongest matched sections."],
                explanation="The role may still fit, but location specifics were not strongly evidenced.",
            )
        )
    if profile_intent.compensation and reranked_result is not None and "compensation" not in reranked_result.matched_sections:
        unknowns.append(
            FactorExplanation(
                name="Compensation fit",
                status="unknown",
                score=None,
                evidence=["Compensation details were not part of the strongest semantic match."],
                explanation="Pay fit may require opening the posting and checking the full compensation details.",
            )
        )
    return unknowns


def summarize_match(explanation: MatchExplanation) -> str:
    top_strength = explanation.strengths[0].name if explanation.strengths else "some relevant overlap"
    if explanation.concerns:
        lead_concern = explanation.concerns[0].evidence[0] if explanation.concerns[0].evidence else explanation.concerns[0].name
        return f"{top_strength} looks promising, but review {lead_concern.lower()} before deciding."
    if explanation.unknowns:
        lead_unknown = explanation.unknowns[0].name.lower()
        return f"{top_strength} is promising, though {lead_unknown} still needs confirmation."
    return f"{top_strength} looks like the clearest reason this role matched."


def recommend_action(explanation: MatchExplanation) -> str:
    overall = explanation.overall_score or 0.0
    llm_fit = explanation.llm_fit_score
    serious_concerns = sum(1 for factor in explanation.concerns if factor.status == "concern")
    unknown_count = len(explanation.unknowns)
    strong_count = sum(1 for factor in explanation.strengths if factor.status == "strong")

    if serious_concerns >= 2 and overall < 60:
        return "Skip"
    if serious_concerns >= 1 and overall < 50:
        return "Skip"
    if overall >= 82 and strong_count >= 2 and serious_concerns == 0 and llm_fit is not None and llm_fit >= 80:
        return "Apply Soon"
    if overall >= 72 and serious_concerns == 0 and (llm_fit is None or llm_fit >= 65):
        return "Strong Match"
    if overall >= 65 and serious_concerns <= 1:
        if llm_fit is None and unknown_count >= 1:
            return "Needs More Info"
        return "Worth Reviewing"
    if overall >= 55 or strong_count >= 1:
        if llm_fit is None and unknown_count >= 1:
            return "Needs More Info"
        return "Maybe"
    if unknown_count >= 2 and serious_concerns == 0:
        return "Needs More Info"
    return "Skip"


def build_match_explanation(
    job_record: Optional[dict],
    score_record: Optional[dict],
    reranked_result: Optional[RerankedJobResult],
    profile_intent: ProfileIntent,
) -> MatchExplanation:
    job = job_record or score_record or {}
    semantic_factors = explain_semantic_evidence(reranked_result, profile_intent)
    score_factors = explain_score_dimensions(score_record)
    concern_factors = explain_concerns(reranked_result, profile_intent)

    strengths = [factor for factor in semantic_factors + score_factors if factor.status in {"strong", "medium"}]
    concerns = concern_factors + [factor for factor in score_factors if factor.status == "concern"]
    unknowns = _build_unknowns(score_record, reranked_result, profile_intent) + [factor for factor in score_factors if factor.status == "unknown"]

    strengths = strengths[:6]
    concerns = concerns[:6]
    unknowns = unknowns[:4]

    explanation = MatchExplanation(
        job_id=str(job.get("id") or getattr(reranked_result, "job_id", "")),
        title=str(job.get("title") or getattr(reranked_result, "title", "Unknown role")),
        company=str(job.get("company") or getattr(reranked_result, "company", "Unknown company")),
        overall_score=_overall_score(score_record, reranked_result),
        semantic_score=round(float(reranked_result.vector_score) * 100, 1) if reranked_result is not None else None,
        rerank_score=round(float(reranked_result.rerank_score) * 100, 1) if reranked_result is not None else None,
        llm_fit_score=(int(score_record["fit_score"]) if score_record and score_record.get("fit_score") is not None else None),
        ats_score=(int(score_record["ats_score"]) if score_record and score_record.get("ats_score") is not None else None),
        recommended_action="",
        summary="",
        strengths=strengths,
        concerns=concerns,
        unknowns=unknowns,
        matched_sections=list(getattr(reranked_result, "matched_sections", []) or []),
        evidence_snippets=list(getattr(reranked_result, "evidence_snippets", []) or [])[:3],
    )
    explanation.recommended_action = recommend_action(explanation)
    explanation.summary = summarize_match(explanation)
    return explanation
