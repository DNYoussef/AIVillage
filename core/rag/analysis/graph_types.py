"""
Type definitions for graph analysis components.

Centralized type definitions to reduce connascence of meaning
and provide single source of truth for data structures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid


class GapType(Enum):
    """Types of knowledge gaps that can be detected."""

    MISSING_NODE = "missing_node"
    MISSING_RELATIONSHIP = "missing_relationship"
    WEAK_CONNECTION = "weak_connection"
    ISOLATED_CLUSTER = "isolated_cluster"
    CONFLICTING_INFO = "conflicting_info"
    INCOMPLETE_PATH = "incomplete_path"
    REDUNDANT_INFO = "redundant_info"


class ConfidenceLevel(Enum):
    """Confidence levels for gap detection and proposals."""

    VERY_HIGH = 0.9
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    VERY_LOW = 0.2


@dataclass
class DetectedGap:
    """A detected gap in the knowledge graph."""

    # Core identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    gap_type: GapType = GapType.MISSING_NODE

    # Gap location and context
    source_nodes: list[str] = field(default_factory=list)
    target_nodes: list[str] = field(default_factory=list)
    context_area: str = ""

    # Gap description
    description: str = ""
    evidence: list[str] = field(default_factory=list)

    # Confidence and priority
    confidence: float = 0.5
    priority: float = 0.5
    severity: float = 0.5

    # Detection metadata
    detection_method: str = ""
    detection_confidence: float = 0.5
    detected_at: datetime = field(default_factory=datetime.now)

    # Resolution tracking
    proposed_solution: dict[str, Any] | None = None
    resolution_status: str = "detected"

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProposedNode:
    """A proposed node to fill a knowledge gap."""

    # Core identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    concept: str = ""

    # Proposal reasoning
    gap_id: str = ""
    reasoning: str = ""
    evidence_sources: list[str] = field(default_factory=list)

    # Probability and confidence
    existence_probability: float = 0.5
    utility_score: float = 0.5
    confidence: float = 0.5

    # Proposed properties
    suggested_trust_score: float = 0.5
    suggested_relationships: list[dict[str, Any]] = field(default_factory=list)

    # Validation
    validation_status: str = "proposed"
    validation_feedback: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProposedRelationship:
    """A proposed relationship to connect existing or proposed nodes."""

    # Core identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""

    # Relationship properties
    relation_type: str = "associative"
    relation_strength: float = 0.5

    # Proposal reasoning
    gap_id: str = ""
    reasoning: str = ""
    evidence_sources: list[str] = field(default_factory=list)

    # Probability and confidence
    existence_probability: float = 0.5
    utility_score: float = 0.5
    confidence: float = 0.5

    # Validation
    validation_status: str = "proposed"
    validation_feedback: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GapAnalysisResult:
    """Result from knowledge gap analysis."""

    gaps_detected: list[DetectedGap] = field(default_factory=list)
    proposed_nodes: list[ProposedNode] = field(default_factory=list)
    proposed_relationships: list[ProposedRelationship] = field(default_factory=list)

    # Analysis metadata
    analysis_time_ms: float = 0.0
    total_gaps_found: int = 0
    total_proposals: int = 0

    # Quality metrics
    avg_gap_confidence: float = 0.0
    avg_proposal_confidence: float = 0.0
    coverage_improvement: float = 0.0

    # Analysis scope
    nodes_analyzed: int = 0
    relationships_analyzed: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)
