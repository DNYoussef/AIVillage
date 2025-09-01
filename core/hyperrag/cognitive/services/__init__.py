"""
GraphFixer Services Package

Decomposed services from the original GraphFixer god class,
following clean architecture principles and single responsibility.

Each service handles a specific aspect of knowledge graph gap detection
and resolution:

- GapDetectionService: Detects knowledge gaps using multiple algorithms
- NodeProposalService: Proposes new nodes to fill gaps
- RelationshipAnalyzerService: Analyzes and proposes relationships
- ConfidenceCalculatorService: Calculates confidence scores
- GraphAnalyticsService: Provides graph metrics and analytics
- KnowledgeValidatorService: Validates consistency and learns from feedback
"""

from .gap_detection_service import GapDetectionService
from .node_proposal_service import NodeProposalService
from .relationship_analyzer_service import RelationshipAnalyzerService
from .confidence_calculator_service import ConfidenceCalculatorService
from .graph_analytics_service import GraphAnalyticsService
from .knowledge_validator_service import KnowledgeValidatorService

__all__ = [
    "GapDetectionService",
    "NodeProposalService",
    "RelationshipAnalyzerService",
    "ConfidenceCalculatorService",
    "GraphAnalyticsService",
    "KnowledgeValidatorService",
]
