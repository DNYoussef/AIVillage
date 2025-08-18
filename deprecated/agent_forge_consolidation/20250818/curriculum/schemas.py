"""Strict JSON schemas for Frontier Curriculum Engine.

All schemas enforce the exact JSON contracts specified in the prompt templates.
Uses Pydantic for validation with strict parsing and helpful error messages.
"""

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, validator


class StrictJsonError(ValueError):
    """Raised when JSON parsing fails with truncated raw text for debugging."""

    def __init__(self, message: str, raw_text: str):
        self.raw_text = raw_text[:500] + "..." if len(raw_text) > 500 else raw_text
        super().__init__(f"{message}\nRaw text (truncated): {self.raw_text}")


def strict_load(text: str, schema_class: type) -> BaseModel:
    """Parse JSON text into schema with helpful error messages on failure.

    Args:
        text: Raw JSON text from LLM
        schema_class: Pydantic model class to parse into

    Returns:
        Validated schema instance

    Raises:
        StrictJsonError: With truncated raw text for debugging
    """
    try:
        # First parse as JSON
        data = json.loads(text.strip())
    except json.JSONDecodeError as e:
        raise StrictJsonError(f"Invalid JSON: {e}", text)

    try:
        # Then validate with Pydantic
        return schema_class(**data)
    except Exception as e:
        raise StrictJsonError(f"Schema validation failed: {e}", text)


# ===== EDGE FINDER SCHEMAS =====


class EdgeWindow(BaseModel):
    """Difficulty window for productive struggle."""

    low: float = Field(..., ge=0.0, le=1.0)
    high: float = Field(..., ge=0.0, le=1.0)

    @validator("high")
    def high_must_be_greater_than_low(cls, v, values):
        if "low" in values and v <= values["low"]:
            raise ValueError("high must be greater than low")
        return v


class TopicMix(BaseModel):
    """Topic weighting for problem generation."""

    topic: str
    weight: float = Field(..., ge=0.0, le=1.0)


class DistributionPoint(BaseModel):
    """Difficulty distribution point."""

    difficulty: float = Field(..., ge=0.0, le=1.0)
    count: int = Field(..., ge=0)


class GenerationPlan(BaseModel):
    """Plan for generating problem batch."""

    n_total: int = Field(..., ge=1)
    per_topic_min: int = Field(..., ge=1)
    variant_rate: float = Field(..., ge=0.0, le=1.0)


class TelemetryEntry(BaseModel):
    """Single telemetry data point."""

    task_id: str
    difficulty: float = Field(..., ge=0.0, le=1.0)
    correct: bool


class DifficultyScale(BaseModel):
    """Difficulty scale bounds."""

    min: float = 0.0
    max: float = 1.0


class EdgeConstraints(BaseModel):
    """Edge finding constraints."""

    target_low: float = Field(default=0.55, ge=0.0, le=1.0)
    target_high: float = Field(default=0.75, ge=0.0, le=1.0)
    problem_budget: int = Field(default=1000, ge=1)

    @validator("target_high")
    def target_high_greater_than_low(cls, v, values):
        if "target_low" in values and v <= values["target_low"]:
            raise ValueError("target_high must be greater than target_low")
        return v


class EdgeAssessmentRequest(BaseModel):
    """Input for edge finder."""

    domain: str
    telemetry: list[TelemetryEntry]
    difficulty_scale: DifficultyScale
    constraints: EdgeConstraints


class EdgeAssessmentResponse(BaseModel):
    """Output from edge finder (strict JSON only)."""

    ok: bool = True
    msg: str = "edge selected"
    edge: EdgeWindow
    topic_mix: list[TopicMix]
    distribution: list[DistributionPoint]
    generation_plan: GenerationPlan


# ===== PROBLEM GENERATOR SCHEMAS =====


class UnitTest(BaseModel):
    """Unit test for a problem."""

    test: str


class Problem(BaseModel):
    """Generated problem."""

    id: str
    topic: str
    difficulty: float = Field(..., ge=0.0, le=1.0)
    statement: str
    canonical_answer: str
    rubric: str
    unit_tests: list[str] = Field(default_factory=list)


class ProblemGenerationRequest(BaseModel):
    """Input for problem generator."""

    domain: str
    edge: EdgeWindow
    topic_mix: list[TopicMix]
    n: int = Field(..., ge=1)
    style: str = "default"


class ProblemGenerationResponse(BaseModel):
    """Output from problem generator (strict JSON only)."""

    ok: bool = True
    msg: str = "generated"
    problems: list[Problem]


# ===== VARIANT SYNTHESIZER SCHEMAS =====


class NumericJitterPolicy(BaseModel):
    """Policy for numeric jittering."""

    enabled: bool = True
    pct: float = Field(default=5, ge=0, le=50)


class VariantPolicy(BaseModel):
    """Policy for variant generation."""

    paraphrase: bool = True
    numeric_jitter: NumericJitterPolicy = Field(default_factory=NumericJitterPolicy)


class ProblemVariant(BaseModel):
    """Variant of a base problem."""

    id: str
    statement: str
    canonical_answer: str
    rubric: str
    unit_tests: list[str] = Field(default_factory=list)


class VariantRequest(BaseModel):
    """Input for variant synthesizer."""

    base_problem: Problem
    variant_policy: VariantPolicy


class VariantResponse(BaseModel):
    """Output from variant synthesizer (strict JSON only)."""

    ok: bool = True
    msg: str = "variants"
    variants: list[ProblemVariant]


# ===== GRADER SCHEMAS =====


class GradingRequest(BaseModel):
    """Input for final answer grader."""

    problem: Problem  # Has statement, canonical_answer, rubric, unit_tests
    model_answer: str


class GradingResponse(BaseModel):
    """Output from grader (strict JSON only)."""

    ok: bool = True
    msg: str = "graded"
    correct: bool
    error_tags: list[str] = Field(default_factory=list)
    normalizer_notes: str = ""


# ===== HINT SYSTEM SCHEMAS =====


class PeerSummary(BaseModel):
    """Brief peer summary for hint generation."""

    model: str
    error_tags: list[str]
    brief_rationale: str = Field(..., max_length=30 * 4)  # ~30 tokens


class HintType(str, Enum):
    """Types of hints."""

    CONCEPT = "concept"
    PROCEDURE = "procedure"
    SANITY_CHECK = "sanity-check"
    BOUNDARY = "boundary"
    UNITS = "units"


class HintRequest(BaseModel):
    """Input for hint generator."""

    problem: Problem  # Has statement, rubric
    wrong_answer: str
    peer_summaries: list[PeerSummary] = Field(default_factory=list)


class HintResponse(BaseModel):
    """Output from hint generator (strict JSON only)."""

    ok: bool = True
    msg: str = "hint"
    hint: str = Field(..., max_length=25 * 4)  # ≤25 tokens
    hint_type: HintType


# ===== MASTERY TRACKING SCHEMAS =====


class AttemptRecord(BaseModel):
    """Record of attempts on a problem."""

    problem_id: str
    attempts: int = 0
    correct_count: int = 0
    variant_ids_correct: list[str] = Field(default_factory=list)
    variant_ids_seen: list[str] = Field(default_factory=list)


class LastResult(BaseModel):
    """Last attempt result."""

    variant_id: str
    correct: bool


class MasteryStatus(str, Enum):
    """Mastery status."""

    LEARNING = "learning"
    UNDERSTOOD = "understood"
    STALLED = "stalled"


class MasteryAction(str, Enum):
    """Next action for mastery."""

    RESHUFFLE = "reshuffle"
    PROMOTE = "promote"
    INJECT_HINT_VARIANT = "inject_hint_variant"


class MasteryRequest(BaseModel):
    """Input for mastery policy."""

    record: AttemptRecord
    last_result: LastResult


class MasteryResponse(BaseModel):
    """Output from mastery policy (strict JSON only)."""

    ok: bool = True
    msg: str = "updated"
    status: MasteryStatus
    next_action: MasteryAction
    needs_hint: bool


# ===== EDGE CONTROLLER SCHEMAS =====


class EdgeDelta(BaseModel):
    """Change in edge window."""

    low: float
    high: float


class ControllerRequest(BaseModel):
    """Input for edge controller."""

    window_accuracy: float = Field(..., ge=0.0, le=1.0)
    current_edge: EdgeWindow
    constraints: EdgeConstraints


class ControllerResponse(BaseModel):
    """Output from edge controller (strict JSON only)."""

    ok: bool = True
    msg: str = "nudged"
    new_edge: EdgeWindow
    delta: EdgeDelta


# ===== CONDUCTOR SCHEMAS =====


class QueueBacklog(BaseModel):
    """Current queue backlogs."""

    fresh: int = Field(..., ge=0)
    variants: int = Field(..., ge=0)
    hint_variants: int = Field(..., ge=0)


class MasteryStats(BaseModel):
    """Mastery statistics."""

    learning: int = Field(..., ge=0)
    understood: int = Field(..., ge=0)
    stalled: int = Field(..., ge=0)


class BatchOperation(str, Enum):
    """Batch operations."""

    GENERATE = "generate"
    VARIANT = "variant"
    HINT_VARIANT = "hint_variant"
    PROMOTE = "promote"
    DROP = "drop"


class BatchItem(BaseModel):
    """Single batch item."""

    op: BatchOperation
    n: int = Field(..., ge=0)
    params: dict[str, Any] = Field(default_factory=dict)


class ConductorRequest(BaseModel):
    """Input for conductor."""

    edge: EdgeWindow
    backlog: QueueBacklog
    mastery_stats: MasteryStats
    capacity: int = Field(..., ge=1)


class ConductorResponse(BaseModel):
    """Output from conductor (strict JSON only)."""

    ok: bool = True
    msg: str = "batch plan"
    queue: list[BatchItem]
