"""
Service Interfaces for GraphFixer Decomposition

Defines the contracts for all GraphFixer services, enabling
loose coupling and testability.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from ..graph_fixer import DetectedGap, ProposedNode, ProposedRelationship
from .base_service import BaseService


class IGapDetectionService(BaseService):
    """Interface for gap detection in knowledge graphs."""
    
    @abstractmethod
    async def detect_gaps(
        self, 
        query: str = None, 
        retrieved_info: List[Any] = None, 
        focus_area: str = None
    ) -> List[DetectedGap]:
        """Detect knowledge gaps using multiple analysis methods."""
        pass
    
    @abstractmethod
    async def detect_structural_gaps(self) -> List[DetectedGap]:
        """Detect structural gaps in graph connectivity."""
        pass
    
    @abstractmethod
    async def detect_semantic_gaps(
        self, 
        query: str = None, 
        focus_area: str = None
    ) -> List[DetectedGap]:
        """Detect semantic gaps using vector analysis."""
        pass
    
    @abstractmethod
    async def detect_connectivity_gaps(self) -> List[DetectedGap]:
        """Detect overall connectivity issues."""
        pass


class INodeProposalService(BaseService):
    """Interface for proposing new nodes to fill gaps."""
    
    @abstractmethod
    async def propose_nodes(self, gaps: List[DetectedGap]) -> List[ProposedNode]:
        """Generate node proposals for detected gaps."""
        pass
    
    @abstractmethod
    async def calculate_existence_probability(self, gap: DetectedGap) -> float:
        """Calculate probability that a proposed node should exist."""
        pass
    
    @abstractmethod
    async def score_utility(self, proposal: ProposedNode, gap: DetectedGap) -> float:
        """Score the utility of a proposed node."""
        pass


class IRelationshipAnalyzerService(BaseService):
    """Interface for analyzing and proposing relationships."""
    
    @abstractmethod
    async def propose_relationships(self, gaps: List[DetectedGap]) -> List[ProposedRelationship]:
        """Generate relationship proposals for detected gaps."""
        pass
    
    @abstractmethod
    async def analyze_semantic_similarity(self, node_id1: str, node_id2: str) -> float:
        """Analyze semantic similarity between two nodes."""
        pass
    
    @abstractmethod
    async def score_relationships(self, proposals: List[ProposedRelationship]) -> List[ProposedRelationship]:
        """Score and rank relationship proposals."""
        pass


class IConfidenceCalculatorService(BaseService):
    """Interface for calculating confidence scores."""
    
    @abstractmethod
    async def calculate_confidence(
        self, 
        proposal: ProposedNode | ProposedRelationship,
        gap: DetectedGap,
        evidence: List[str]
    ) -> float:
        """Calculate confidence score for a proposal."""
        pass
    
    @abstractmethod
    async def combine_evidence(self, evidence_list: List[str]) -> float:
        """Combine multiple pieces of evidence into a confidence score."""
        pass
    
    @abstractmethod
    async def validate_proposal_logic(
        self, 
        proposal: ProposedNode | ProposedRelationship
    ) -> bool:
        """Validate the logical consistency of a proposal."""
        pass


class IGraphAnalyticsService(BaseService):
    """Interface for graph metrics and analytics."""
    
    @abstractmethod
    async def compute_centrality_metrics(self) -> Dict[str, Any]:
        """Compute centrality metrics for graph nodes."""
        pass
    
    @abstractmethod
    async def analyze_clusters(self) -> Dict[str, Any]:
        """Analyze clustering patterns in the graph."""
        pass
    
    @abstractmethod
    async def measure_connectivity(self) -> Dict[str, Any]:
        """Measure overall graph connectivity."""
        pass
    
    @abstractmethod
    async def analyze_completeness(self) -> Dict[str, Any]:
        """Analyze graph completeness and coverage."""
        pass


class IKnowledgeValidatorService(BaseService):
    """Interface for validating knowledge consistency."""
    
    @abstractmethod
    async def validate_consistency(
        self, 
        proposals: List[ProposedNode | ProposedRelationship]
    ) -> Dict[str, bool]:
        """Validate consistency of proposals with existing knowledge."""
        pass
    
    @abstractmethod
    async def check_conflicts(
        self, 
        proposal: ProposedNode | ProposedRelationship
    ) -> List[str]:
        """Check for conflicts with existing knowledge."""
        pass
    
    @abstractmethod
    async def verify_logic(self, gap: DetectedGap) -> bool:
        """Verify the logical validity of a detected gap."""
        pass
    
    @abstractmethod
    async def learn_from_validation(
        self, 
        proposal: ProposedNode | ProposedRelationship, 
        is_accepted: bool
    ) -> None:
        """Learn from validation feedback to improve future proposals."""
        pass