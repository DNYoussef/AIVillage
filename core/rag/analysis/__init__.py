"""
Analysis subsystem for HyperRAG - Knowledge gap detection and repair.

This module provides graph analysis capabilities including gap detection,
node proposals, and knowledge completeness assessment.
"""

from .graph_fixer import (
    ConfidenceLevel,
    DetectedGap,
    GapAnalysisResult,
    GapType,
    GraphFixer,
    ProposedNode,
    ProposedRelationship,
)

__all__ = [
    "GraphFixer",
    "DetectedGap",
    "ProposedNode",
    "ProposedRelationship",
    "GapAnalysisResult",
    "GapType",
    "ConfidenceLevel",
]
