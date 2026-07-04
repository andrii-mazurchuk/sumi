"""
Sumi data models — source of truth for all data shapes.

All components read from and write to these Pydantic models.
No ad-hoc dicts anywhere in the codebase.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Scenario input models ────────────────────────────────────────────────────


class Trait(BaseModel):
    """One testable behavioral property the model is expected to exhibit."""

    name: str
    description: str
    evaluation_method: Literal["llm_judge", "pattern_match", "stylometric"]
    # For pattern_match: one or more regex patterns (any match = present)
    patterns: Optional[List[str]] = None
    # For llm_judge: criteria injected into the judge prompt
    judge_criteria: Optional[str] = None
    # Anchors for llm_judge calibration
    judge_anchors: Optional[Dict[str, str]] = None
    # Weight when aggregating trait scores (default 1.0)
    weight: float = 1.0

    @field_validator("patterns")
    @classmethod
    def patterns_only_for_pattern_match(cls, v, info):
        method = info.data.get("evaluation_method")
        if method != "pattern_match" and v is not None:
            raise ValueError("'patterns' field is only valid for evaluation_method='pattern_match'")
        return v


class TestCase(BaseModel):
    """One (prompt, expected_behavior) pair used in static coverage tests."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    prompt: str
    expected_behavior: str
    evaluation_method: Literal[
        "stylometric", "llm_judge", "pattern_match", "embedding_sim", "perplexity"
    ]
    weight: float = 1.0
    # Optional reference text for perplexity scoring or embedding anchor
    reference_text: Optional[str] = None


class PassThreshold(BaseModel):
    per_category: float = 0.75
    per_trait: Optional[float] = None  # defaults to per_category at runtime

    def effective_per_trait(self) -> float:
        return self.per_trait if self.per_trait is not None else self.per_category


class ValidationScenario(BaseModel):
    """Complete user-defined validation scenario loaded from YAML."""

    name: str
    goal: str
    traits: List[Trait]
    test_cases: List[TestCase]
    pass_threshold: PassThreshold = Field(default_factory=PassThreshold)
    temporal_turns: int = 20
    metadata: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def check_non_empty(self) -> ValidationScenario:
        if not self.traits:
            raise ValueError("scenario must define at least one trait")
        if not self.test_cases:
            raise ValueError("scenario must define at least one test_case")
        return self


# ─── Result models ────────────────────────────────────────────────────────────


class TestCaseResult(BaseModel):
    test_case_id: str
    prompt: str
    response: str
    score: float  # 0–1
    passed: bool
    evaluator: str
    explanation: Optional[str] = None


class StaticCoverageResult(BaseModel):
    test_case_results: List[TestCaseResult]
    aggregate_score: float
    passed: bool
    baseline_comparison: Optional[float] = None  # score of non-fine-tuned baseline


class TurnResult(BaseModel):
    turn: int
    prompt: str
    response: str
    consistency_score: float  # 0–1
    trait_scores: Dict[str, float] = Field(default_factory=dict)
    explanation: Optional[str] = None


class DecayCurve(BaseModel):
    turn_scores: List[float]  # index = turn number (0-based)
    breakpoint_turn: Optional[int] = None  # first turn score < threshold
    half_life: Optional[float] = None  # interpolated turn where score ≈ 0.5
    threshold_used: float


class TemporalPersistenceResult(BaseModel):
    turn_results: List[TurnResult]
    decay_curve: DecayCurve
    aggregate_score: float  # mean consistency over all turns
    passed: bool


class AdversarialTurnResult(BaseModel):
    attack_type: str
    turn: int
    attack_prompt: str
    response: str
    maintained_persona: bool
    resistance_score: float  # 0–1 for this turn
    explanation: Optional[str] = None


class AdversarialAttackResult(BaseModel):
    attack_type: Literal[
        "direct_demand", "gradual_pressure", "roleplay_injection", "logical_challenge"
    ]
    prompts_used: int
    turn_results: List[AdversarialTurnResult]
    resistance_score: float  # mean across all turns
    breakdown_turn: Optional[int] = None  # first turn persona breaks
    passed: bool


class AdversarialRobustnessResult(BaseModel):
    attack_results: List[AdversarialAttackResult]
    aggregate_score: float  # mean across all attack types
    weakest_attack_type: str
    passed: bool


class TraitProfile(BaseModel):
    trait_name: str
    static_score: Optional[float] = None
    temporal_score: Optional[float] = None
    adversarial_score: Optional[float] = None
    overall: float
    verdict: Literal["robust", "fragile", "mixed"]


class TraitDecompositionResult(BaseModel):
    trait_profiles: List[TraitProfile]
    robust_traits: List[str]
    fragile_traits: List[str]


# ─── Top-level report ─────────────────────────────────────────────────────────


class ValidationReport(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scenario_name: str
    model_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    static_coverage: Optional[StaticCoverageResult] = None
    temporal_persistence: Optional[TemporalPersistenceResult] = None
    adversarial_robustness: Optional[AdversarialRobustnessResult] = None
    trait_decomposition: Optional[TraitDecompositionResult] = None

    overall_verdict: Literal["pass", "fail", "partial"]
    confidence: float  # 0–1
    categories_run: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Run: {self.run_id[:8]}",
            f"Scenario: {self.scenario_name}",
            f"Model: {self.model_id}",
            f"Verdict: {self.overall_verdict.upper()} (confidence {self.confidence:.2f})",
        ]
        if self.static_coverage:
            lines.append(
                f"  Static coverage: {self.static_coverage.aggregate_score:.2f} "
                f"({'PASS' if self.static_coverage.passed else 'FAIL'})"
            )
        if self.temporal_persistence:
            lines.append(
                f"  Temporal persistence: {self.temporal_persistence.aggregate_score:.2f} "
                f"({'PASS' if self.temporal_persistence.passed else 'FAIL'})"
            )
        if self.adversarial_robustness:
            lines.append(
                f"  Adversarial robustness: {self.adversarial_robustness.aggregate_score:.2f} "
                f"({'PASS' if self.adversarial_robustness.passed else 'FAIL'})"
            )
        return "\n".join(lines)
