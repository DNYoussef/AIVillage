from __future__ import annotations

"""Rule-based confidence scoring utilities."""


# Define confidence tier and evidence pack locally to avoid import issues
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConfidenceTier(Enum):
    """Confidence tiers for evidence scoring."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class EvidenceChunk:
    """Individual evidence chunk."""

    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidencePack:
    """Collection of evidence chunks."""

    chunks: list[EvidenceChunk] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


THRESHOLDS = {
    ConfidenceTier.HIGH: 0.8,
    ConfidenceTier.MEDIUM: 0.5,
    ConfidenceTier.LOW: 0.0,
}


def score_evidence(pack: EvidencePack) -> ConfidenceTier:
    """Assign a confidence tier based on average chunk score."""
    if not pack.chunks:
        return ConfidenceTier.LOW
    avg = sum(c.score for c in pack.chunks) / len(pack.chunks)
    for tier in (ConfidenceTier.HIGH, ConfidenceTier.MEDIUM, ConfidenceTier.LOW):
        if avg >= THRESHOLDS[tier]:
            return tier
    return ConfidenceTier.LOW


def assign_tier(pack: EvidencePack) -> EvidencePack:
    """Return new EvidencePack with confidence_tier set."""
    tier = score_evidence(pack)
    object.__setattr__(pack, "confidence_tier", tier)
    return pack
