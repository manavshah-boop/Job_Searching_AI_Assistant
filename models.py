"""
models.py — Step 8: Pydantic models for structured LLM outputs.

Provides three core models:
  - ScoreResult: Dimension scores from score_dimensions()
  - StructuredProfile: Extracted candidate profile from build_structured_profile()
  - ScoringWeights: Dimension weights (validation only, used with instructor)

These replace fragile JSON parsing with guaranteed structured validation.
Uses Pydantic v2 field validators and can be used with instructor for structured LLM outputs.
"""

from typing import List, Optional
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator


class ScoreResult(BaseModel):
    """
    LLM structured output for job scoring.
    Fields must match the JSON schema used in score_dimensions() prompts.
    """
    disqualified: bool
    disqualify_reason: str = ""
    role_fit: int = 0
    stack_match: int = 0
    seniority: int = 0
    location: int = 0
    growth: int = 0
    compensation: int = 0
    reasons: List[str] = []
    flags: List[str] = []
    one_liner: str = ""

    @field_validator("role_fit", "stack_match", "seniority", "location", "growth", "compensation", mode="before")
    @classmethod
    def clamp_scores(cls, v):
        """Clamp dimension scores to 0-10 range."""
        if not isinstance(v, int):
            try:
                v = int(v)
            except (ValueError, TypeError):
                return 0
        return max(0, min(10, v))

    @field_validator("reasons", mode="before")
    @classmethod
    def limit_reasons(cls, v):
        """Limit reasons to exactly 2-4 items. If fewer, keep as-is. If more than 4, keep first 4."""
        if not isinstance(v, list):
            return []
        # Ensure all items are strings
        v = [str(item) if item else "" for item in v]
        # Keep max 4 items
        return v[:4]

    @model_validator(mode="after")
    def zero_scores_when_disqualified(self):
        """When disqualified, set all dimension scores to 0."""
        if self.disqualified:
            self.role_fit = 0
            self.stack_match = 0
            self.seniority = 0
            self.location = 0
            self.growth = 0
            self.compensation = 0
        return self


class StructuredProfile(BaseModel):
    """
    LLM structured output for candidate profile extraction.
    Fields match the JSON schema used in build_structured_profile() prompts.
    """
    name: str = ""
    yoe: int = 0
    current_title: str = ""
    core_skills: List[str] = []
    languages: List[str] = []
    frameworks: List[str] = []
    cloud: List[str] = []
    past_roles: List[str] = []
    education: str = ""
    strengths: List[str] = []
    target_roles: List[str] = []
    target_salary: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("target_salary", "min_salary"),
    )
    remote_preference: str = "True"
    preferred_locations: List[str] = []

    @field_validator(
        "core_skills", "languages", "frameworks", "cloud", 
        "past_roles", "strengths", "target_roles", "preferred_locations",
        mode="before"
    )
    @classmethod
    def ensure_list(cls, v):
        """Ensure field is a list, deduplicate, and filter empty strings."""
        if not isinstance(v, list):
            if isinstance(v, str):
                return [v] if v else []
            return []
        # Deduplicate while preserving order, filter empty strings
        seen = set()
        result = []
        for item in v:
            item_str = str(item).strip()
            if item_str and item_str not in seen:
                seen.add(item_str)
                result.append(item_str)
        return result

    @field_validator("target_salary", mode="before")
    @classmethod
    def parse_salary(cls, v):
        """Ensure target_salary is an integer when present."""
        if v in (None, ""):
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            # Strip $ and commas, convert to int
            v = v.replace("$", "").replace(",", "").strip()
            try:
                return int(v)
            except ValueError:
                return None
        return None

    @field_validator("yoe", mode="before")
    @classmethod
    def parse_yoe(cls, v):
        """Ensure yoe is an integer."""
        if isinstance(v, int):
            return max(0, v)
        try:
            return max(0, int(v))
        except (ValueError, TypeError):
            return 0


class ScoringWeights(BaseModel):
    """
    Dimension weights for the scoring pipeline.
    All weights must sum to 1.0 ± 0.01 (accounting for float precision).
    Used for validation only; typically loaded from config.yaml.
    """
    role_fit: float = 0.30
    stack_match: float = 0.25
    seniority: float = 0.20
    location: float = 0.10
    growth: float = 0.10
    compensation: float = 0.05

    @model_validator(mode="after")
    def weights_sum_to_one(self):
        """Validate that all weights sum to approximately 1.0 (within 0.01 tolerance)."""
        total = (
            self.role_fit + self.stack_match + self.seniority +
            self.location + self.growth + self.compensation
        )
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0 ± 0.01, got {total:.4f}")
        return self
