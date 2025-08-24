"""
GraphFixer - Automated Knowledge Gap Detection and Node Proposal System

Intelligent system for detecting gaps in knowledge graphs and proposing
new nodes and relationships with probability ratings. Uses graph analysis,
semantic understanding, and probabilistic reasoning to identify missing
knowledge and suggest improvements.

This module provides the graph analysis component of the unified HyperRAG system.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import time
from typing import Any
import uuid

import numpy as np

logger = logging.getLogger(__name__)


class GapType(Enum):
    """Types of knowledge gaps that can be detected."""

    MISSING_NODE = "missing_node"  # Missing concept or entity
    MISSING_RELATIONSHIP = "missing_relationship"  # Missing connection
    WEAK_CONNECTION = "weak_connection"  # Underconnected nodes
    ISOLATED_CLUSTER = "isolated_cluster"  # Disconnected subgraphs
    CONFLICTING_INFO = "conflicting_info"  # Contradictory information
    INCOMPLETE_PATH = "incomplete_path"  # Broken reasoning chains
    REDUNDANT_INFO = "redundant_info"  # Duplicate/overlapping content


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

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    gap_type: GapType = GapType.MISSING_NODE

    # Gap location and context
    source_nodes: list[str] = field(default_factory=list)  # Nodes involved in gap
    target_nodes: list[str] = field(default_factory=list)  # Expected target nodes
    context_area: str = ""  # Conceptual area of the gap

    # Gap description
    description: str = ""
    evidence: list[str] = field(default_factory=list)  # Evidence for this gap

    # Confidence and priority
    confidence: float = 0.5
    priority: float = 0.5  # How important is fixing this gap
    severity: float = 0.5  # How much does this gap hurt system performance

    # Detection metadata
    detection_method: str = ""
    detection_confidence: float = 0.5
    detected_at: datetime = field(default_factory=datetime.now)

    # Resolution tracking
    proposed_solution: dict[str, Any] | None = None
    resolution_status: str = "detected"  # detected, proposed, resolved, dismissed

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProposedNode:
    """A proposed node to fill a knowledge gap."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    concept: str = ""

    # Proposal reasoning
    gap_id: str = ""  # Gap this proposal addresses
    reasoning: str = ""  # Why this node is proposed
    evidence_sources: list[str] = field(default_factory=list)

    # Probability and confidence
    existence_probability: float = 0.5  # How likely is this concept to be valid
    utility_score: float = 0.5  # How useful would this node be
    confidence: float = 0.5  # Overall confidence in proposal

    # Proposed properties
    suggested_trust_score: float = 0.5
    suggested_relationships: list[dict[str, Any]] = field(default_factory=list)

    # Validation
    validation_status: str = "proposed"  # proposed, validated, rejected
    validation_feedback: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProposedRelationship:
    """A proposed relationship to connect existing or proposed nodes."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""

    # Relationship properties
    relation_type: str = "associative"
    relation_strength: float = 0.5

    # Proposal reasoning
    gap_id: str = ""  # Gap this proposal addresses
    reasoning: str = ""  # Why this relationship is proposed
    evidence_sources: list[str] = field(default_factory=list)

    # Probability and confidence
    existence_probability: float = 0.5  # How likely is this relationship to be valid
    utility_score: float = 0.5  # How useful would this relationship be
    confidence: float = 0.5  # Overall confidence in proposal

    # Validation
    validation_status: str = "proposed"  # proposed, validated, rejected
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
    coverage_improvement: float = 0.0  # Estimated improvement in graph coverage

    # Analysis scope
    nodes_analyzed: int = 0
    relationships_analyzed: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)


class GraphFixer:
    """
    Automated Knowledge Gap Detection and Repair System

    Intelligent system for analyzing knowledge graphs to detect gaps,
    inconsistencies, and opportunities for improvement. Proposes new
    nodes and relationships with probability ratings based on graph
    structure analysis and semantic understanding.

    Features:
    - Multi-dimensional gap detection (structural, semantic, logical)
    - Probabilistic node and relationship proposals
    - Confidence scoring for all suggestions
    - Validation feedback integration
    - Priority-based gap ranking
    - Evidence-based reasoning
    """

    def __init__(
        self,
        trust_graph=None,
        vector_engine=None,
        min_confidence_threshold: float = 0.3,
        max_proposals_per_gap: int = 3,
    ):
        self.trust_graph = trust_graph
        self.vector_engine = vector_engine
        self.min_confidence_threshold = min_confidence_threshold
        self.max_proposals_per_gap = max_proposals_per_gap

        # Analysis configuration
        self.gap_detection_methods = [
            "structural_analysis",
            "semantic_clustering",
            "path_analysis",
            "trust_inconsistency",
            "connectivity_analysis",
        ]

        # Caching for expensive operations
        self.analysis_cache: dict[str, GapAnalysisResult] = {}
        self.node_similarity_cache: dict[str, dict[str, float]] = {}

        # Statistics
        self.stats = {
            "gaps_detected": 0,
            "nodes_proposed": 0,
            "relationships_proposed": 0,
            "analyses_performed": 0,
            "proposals_validated": 0,
            "proposals_rejected": 0,
        }

        self.initialized = False

    async def initialize(self):
        """Initialize the graph fixer system."""
        logger.info("Initializing GraphFixer...")

        if not self.trust_graph:
            logger.warning("No trust graph provided - some functionality will be limited")

        if not self.vector_engine:
            logger.warning("No vector engine provided - semantic analysis will be limited")

        self.initialized = True
        logger.info("🔧 GraphFixer ready for knowledge gap detection and repair")

    async def detect_knowledge_gaps(
        self, query: str | None = None, retrieved_info: list[Any] | None = None, focus_area: str | None = None
    ) -> list[DetectedGap]:
        """Detect knowledge gaps in the graph."""
        start_time = time.time()

        try:
            gaps = []

            # Run different gap detection methods
            for method in self.gap_detection_methods:
                method_gaps = await self._run_detection_method(method, query, retrieved_info, focus_area)
                gaps.extend(method_gaps)

            # Remove duplicates and rank by priority
            unique_gaps = await self._deduplicate_gaps(gaps)
            ranked_gaps = await self._rank_gaps_by_priority(unique_gaps)

            # Update statistics
            self.stats["gaps_detected"] += len(ranked_gaps)
            self.stats["analyses_performed"] += 1

            analysis_time = (time.time() - start_time) * 1000
            logger.info(f"Detected {len(ranked_gaps)} knowledge gaps in {analysis_time:.1f}ms")

            return ranked_gaps

        except Exception as e:
            logger.exception(f"Gap detection failed: {e}")
            return []

    async def propose_solutions(
        self, gaps: list[DetectedGap], max_proposals: int | None = None
    ) -> tuple[list[ProposedNode], list[ProposedRelationship]]:
        """Propose nodes and relationships to fix detected gaps."""
        try:
            proposed_nodes = []
            proposed_relationships = []

            limit = max_proposals or len(gaps)

            for gap in gaps[:limit]:
                # Generate proposals based on gap type
                if gap.gap_type == GapType.MISSING_NODE:
                    nodes = await self._propose_missing_nodes(gap)
                    proposed_nodes.extend(nodes)

                elif gap.gap_type == GapType.MISSING_RELATIONSHIP:
                    relationships = await self._propose_missing_relationships(gap)
                    proposed_relationships.extend(relationships)

                elif gap.gap_type == GapType.WEAK_CONNECTION:
                    relationships = await self._propose_strengthening_relationships(gap)
                    proposed_relationships.extend(relationships)

                elif gap.gap_type == GapType.ISOLATED_CLUSTER:
                    bridge_relationships = await self._propose_bridge_relationships(gap)
                    proposed_relationships.extend(bridge_relationships)

                elif gap.gap_type == GapType.INCOMPLETE_PATH:
                    path_nodes, path_relationships = await self._propose_path_completion(gap)
                    proposed_nodes.extend(path_nodes)
                    proposed_relationships.extend(path_relationships)

            # Rank proposals by utility and confidence
            proposed_nodes = await self._rank_node_proposals(proposed_nodes)
            proposed_relationships = await self._rank_relationship_proposals(proposed_relationships)

            # Update statistics
            self.stats["nodes_proposed"] += len(proposed_nodes)
            self.stats["relationships_proposed"] += len(proposed_relationships)

            logger.info(f"Proposed {len(proposed_nodes)} nodes and {len(proposed_relationships)} relationships")

            return proposed_nodes, proposed_relationships

        except Exception as e:
            logger.exception(f"Solution proposal failed: {e}")
            return [], []

    async def validate_proposal(
        self, proposal: ProposedNode | ProposedRelationship, validation_feedback: str, is_accepted: bool
    ) -> bool:
        """Validate a proposed node or relationship."""
        try:
            if is_accepted:
                proposal.validation_status = "validated"
                self.stats["proposals_validated"] += 1
            else:
                proposal.validation_status = "rejected"
                self.stats["proposals_rejected"] += 1

            proposal.validation_feedback = validation_feedback

            # Learn from validation to improve future proposals
            await self._learn_from_validation(proposal, is_accepted)

            return True

        except Exception as e:
            logger.exception(f"Proposal validation failed: {e}")
            return False

    async def analyze_graph_completeness(self) -> dict[str, Any]:
        """Analyze overall completeness and quality of the knowledge graph."""
        try:
            if not self.trust_graph:
                return {"error": "No trust graph available"}

            analysis = {
                "structural_metrics": await self._analyze_structural_completeness(),
                "semantic_metrics": await self._analyze_semantic_completeness(),
                "trust_metrics": await self._analyze_trust_distribution(),
                "connectivity_metrics": await self._analyze_connectivity_patterns(),
            }

            # Calculate overall completeness score
            structural_score = analysis["structural_metrics"].get("completeness_score", 0.0)
            semantic_score = analysis["semantic_metrics"].get("completeness_score", 0.0)
            trust_score = analysis["trust_metrics"].get("avg_trust", 0.0)
            connectivity_score = analysis["connectivity_metrics"].get("connectivity_score", 0.0)

            overall_score = (structural_score + semantic_score + trust_score + connectivity_score) / 4
            analysis["overall_completeness"] = overall_score

            return analysis

        except Exception as e:
            logger.exception(f"Graph completeness analysis failed: {e}")
            return {"error": str(e)}

    async def get_gap_statistics(self) -> dict[str, Any]:
        """Get statistics about gap detection and proposals."""
        try:
            return {
                "gap_detection": {
                    "total_gaps_detected": self.stats["gaps_detected"],
                    "analyses_performed": self.stats["analyses_performed"],
                    "detection_methods_active": len(self.gap_detection_methods),
                },
                "proposals": {
                    "nodes_proposed": self.stats["nodes_proposed"],
                    "relationships_proposed": self.stats["relationships_proposed"],
                    "validation_rate": (
                        self.stats["proposals_validated"]
                        / max(1, self.stats["proposals_validated"] + self.stats["proposals_rejected"])
                    ),
                },
                "performance": {
                    "cache_entries": len(self.analysis_cache),
                    "similarity_cache_entries": len(self.node_similarity_cache),
                },
                "configuration": {
                    "min_confidence_threshold": self.min_confidence_threshold,
                    "max_proposals_per_gap": self.max_proposals_per_gap,
                    "trust_graph_available": self.trust_graph is not None,
                    "vector_engine_available": self.vector_engine is not None,
                },
            }

        except Exception as e:
            logger.exception(f"Statistics gathering failed: {e}")
            return {"error": str(e)}

    # Private implementation methods

    async def _run_detection_method(
        self, method: str, query: str | None, retrieved_info: list[Any] | None, focus_area: str | None
    ) -> list[DetectedGap]:
        """Run a specific gap detection method."""
        try:
            if method == "structural_analysis":
                return await self._detect_structural_gaps()
            elif method == "semantic_clustering":
                return await self._detect_semantic_gaps(query, focus_area)
            elif method == "path_analysis":
                return await self._detect_path_gaps(retrieved_info)
            elif method == "trust_inconsistency":
                return await self._detect_trust_inconsistencies()
            elif method == "connectivity_analysis":
                return await self._detect_connectivity_gaps()
            else:
                logger.warning(f"Unknown detection method: {method}")
                return []

        except Exception as e:
            logger.warning(f"Detection method {method} failed: {e}")
            return []

    async def _detect_structural_gaps(self) -> list[DetectedGap]:
        """Detect structural gaps in the graph."""
        gaps = []

        if not self.trust_graph:
            return gaps

        try:
            # Find isolated nodes (no connections)
            for node_id, node in self.trust_graph.nodes.items():
                if len(node.incoming_edges) == 0 and len(node.outgoing_edges) == 0:
                    gap = DetectedGap(
                        gap_type=GapType.ISOLATED_CLUSTER,
                        source_nodes=[node_id],
                        description=f"Node '{node.concept}' has no connections",
                        evidence=[f"Node {node_id} has 0 incoming and 0 outgoing edges"],
                        confidence=0.9,
                        priority=0.6,
                        detection_method="structural_analysis",
                    )
                    gaps.append(gap)

            # Find weakly connected nodes (only 1-2 connections)
            for node_id, node in self.trust_graph.nodes.items():
                total_connections = len(node.incoming_edges) + len(node.outgoing_edges)

                if 1 <= total_connections <= 2 and node.trust_score > 0.6:
                    gap = DetectedGap(
                        gap_type=GapType.WEAK_CONNECTION,
                        source_nodes=[node_id],
                        description=f"High-trust node '{node.concept}' is under-connected",
                        evidence=[
                            f"Node {node_id} has only {total_connections} connections but trust score {node.trust_score:.2f}"
                        ],
                        confidence=0.7,
                        priority=0.5,
                        detection_method="structural_analysis",
                    )
                    gaps.append(gap)

            return gaps

        except Exception as e:
            logger.exception(f"Structural gap detection failed: {e}")
            return gaps

    async def _detect_semantic_gaps(self, query: str | None, focus_area: str | None) -> list[DetectedGap]:
        """Detect semantic gaps using vector analysis."""
        gaps = []

        if not self.vector_engine or not self.trust_graph:
            return gaps

        try:
            # Look for semantic clusters that might be missing bridging concepts
            concept_embeddings = {}

            # Get embeddings for graph concepts (simplified)
            for node_id, node in self.trust_graph.nodes.items():
                if node.embedding is not None:
                    concept_embeddings[node_id] = node.embedding

            if len(concept_embeddings) < 3:
                return gaps  # Need enough concepts to detect gaps

            # Find concepts that are semantically similar but not connected
            node_ids = list(concept_embeddings.keys())
            for i, node_id1 in enumerate(node_ids):
                for _j, node_id2 in enumerate(node_ids[i + 1 :], i + 1):
                    # Calculate semantic similarity
                    emb1 = concept_embeddings[node_id1]
                    emb2 = concept_embeddings[node_id2]
                    similarity = np.dot(emb1, emb2)

                    # Check if they're semantically similar but not connected
                    if similarity > 0.7:  # High semantic similarity
                        node1 = self.trust_graph.nodes[node_id1]
                        node2 = self.trust_graph.nodes[node_id2]

                        # Check if they're connected
                        connected = any(
                            edge.source_id == node_id1
                            and edge.target_id == node_id2
                            or edge.source_id == node_id2
                            and edge.target_id == node_id1
                            for edge in self.trust_graph.edges.values()
                        )

                        if not connected:
                            gap = DetectedGap(
                                gap_type=GapType.MISSING_RELATIONSHIP,
                                source_nodes=[node_id1, node_id2],
                                description=f"Semantically similar concepts '{node1.concept}' and '{node2.concept}' are not connected",
                                evidence=[f"Semantic similarity: {similarity:.3f}, but no direct relationship"],
                                confidence=similarity * 0.8,  # Scale confidence by similarity
                                priority=0.6,
                                detection_method="semantic_clustering",
                            )
                            gaps.append(gap)

            return gaps

        except Exception as e:
            logger.exception(f"Semantic gap detection failed: {e}")
            return gaps

    async def _detect_path_gaps(self, retrieved_info: list[Any] | None) -> list[DetectedGap]:
        """Detect gaps in reasoning paths."""
        gaps = []

        if not retrieved_info or not self.trust_graph:
            return gaps

        try:
            # Look for broken reasoning chains in retrieved information
            # This is a simplified implementation

            # Group retrieved info by source system
            sources_info = {}
            for info in retrieved_info:
                source = getattr(info, "source", "unknown")
                if source not in sources_info:
                    sources_info[source] = []
                sources_info[source].append(info)

            # Check for information that appears in vector/episodic but not in graph
            if "vector" in sources_info and "graph" in sources_info:
                vector_concepts = set()
                graph_concepts = set()

                # Extract concepts from vector results
                for info in sources_info["vector"]:
                    content = getattr(info, "content", "")
                    # Simple concept extraction
                    words = content.lower().split()
                    vector_concepts.update(word for word in words if len(word) > 4)

                # Extract concepts from graph results
                for info in sources_info["graph"]:
                    content = getattr(info, "content", "")
                    words = content.lower().split()
                    graph_concepts.update(word for word in words if len(word) > 4)

                # Find concepts in vector but not in graph
                missing_in_graph = vector_concepts - graph_concepts

                if missing_in_graph:
                    gap = DetectedGap(
                        gap_type=GapType.MISSING_NODE,
                        description="Concepts found in vector search but missing from knowledge graph",
                        evidence=[f"Missing concepts: {list(missing_in_graph)[:5]}"],  # Show first 5
                        confidence=0.6,
                        priority=0.7,
                        detection_method="path_analysis",
                    )
                    gaps.append(gap)

            return gaps

        except Exception as e:
            logger.exception(f"Path gap detection failed: {e}")
            return gaps

    async def _detect_trust_inconsistencies(self) -> list[DetectedGap]:
        """Detect inconsistencies in trust scores."""
        gaps = []

        if not self.trust_graph:
            return gaps

        try:
            # Find nodes with conflicting trust information
            for node_id, node in self.trust_graph.nodes.items():
                # Look for incoming edges with very different trust scores
                incoming_trusts = []
                for edge_id in node.incoming_edges:
                    if edge_id in self.trust_graph.edges:
                        edge = self.trust_graph.edges[edge_id]
                        source_node = self.trust_graph.nodes[edge.source_id]
                        incoming_trusts.append(source_node.trust_score)

                if len(incoming_trusts) >= 2:
                    trust_variance = np.var(incoming_trusts)

                    # High variance suggests conflicting trust information
                    if trust_variance > 0.2:  # Significant variance
                        gap = DetectedGap(
                            gap_type=GapType.CONFLICTING_INFO,
                            source_nodes=[node_id],
                            description=f"Node '{node.concept}' has conflicting trust information from connected nodes",
                            evidence=[f"Trust variance: {trust_variance:.3f}, incoming trusts: {incoming_trusts}"],
                            confidence=0.8,
                            priority=0.8,  # High priority for trust conflicts
                            detection_method="trust_inconsistency",
                        )
                        gaps.append(gap)

            return gaps

        except Exception as e:
            logger.exception(f"Trust inconsistency detection failed: {e}")
            return gaps

    async def _detect_connectivity_gaps(self) -> list[DetectedGap]:
        """Detect connectivity issues in the graph."""
        gaps = []

        if not self.trust_graph:
            return gaps

        try:
            # Simple connectivity analysis
            total_nodes = len(self.trust_graph.nodes)
            total_edges = len(self.trust_graph.edges)

            if total_nodes > 10:  # Only analyze if we have enough nodes
                # Calculate expected connectivity
                expected_edges = total_nodes * 1.5  # Rough heuristic

                if total_edges < expected_edges * 0.5:  # Significantly under-connected
                    gap = DetectedGap(
                        gap_type=GapType.WEAK_CONNECTION,
                        description="Graph appears under-connected overall",
                        evidence=[
                            f"Total nodes: {total_nodes}, Total edges: {total_edges}, Expected: ~{expected_edges:.0f}"
                        ],
                        confidence=0.7,
                        priority=0.6,
                        detection_method="connectivity_analysis",
                    )
                    gaps.append(gap)

            return gaps

        except Exception as e:
            logger.exception(f"Connectivity gap detection failed: {e}")
            return gaps

    async def _propose_missing_nodes(self, gap: DetectedGap) -> list[ProposedNode]:
        """Propose nodes to fill missing node gaps."""
        proposals = []

        try:
            # Analyze the gap context to propose relevant nodes
            if gap.source_nodes and self.trust_graph:
                # Look at neighboring concepts for inspiration
                neighboring_concepts = set()

                for node_id in gap.source_nodes:
                    if node_id in self.trust_graph.nodes:
                        node = self.trust_graph.nodes[node_id]

                        # Collect concepts from connected nodes
                        for edge_id in node.outgoing_edges | node.incoming_edges:
                            if edge_id in self.trust_graph.edges:
                                edge = self.trust_graph.edges[edge_id]
                                other_node_id = edge.target_id if edge.source_id == node_id else edge.source_id

                                if other_node_id in self.trust_graph.nodes:
                                    other_node = self.trust_graph.nodes[other_node_id]
                                    neighboring_concepts.add(other_node.concept)

                # Propose bridging concepts
                if len(neighboring_concepts) >= 2:
                    concept_list = list(neighboring_concepts)
                    bridging_concept = f"bridge_concept_for_{concept_list[0]}_and_{concept_list[1]}"

                    proposal = ProposedNode(
                        content=f"Bridging concept connecting {concept_list[0]} and {concept_list[1]}",
                        concept=bridging_concept,
                        gap_id=gap.id,
                        reasoning="Would help connect disparate concepts in the knowledge graph",
                        existence_probability=0.6,
                        utility_score=0.7,
                        confidence=0.6,
                        suggested_trust_score=0.5,
                    )
                    proposals.append(proposal)

            return proposals[: self.max_proposals_per_gap]

        except Exception as e:
            logger.exception(f"Node proposal failed: {e}")
            return []

    async def _propose_missing_relationships(self, gap: DetectedGap) -> list[ProposedRelationship]:
        """Propose relationships to fill missing relationship gaps."""
        proposals = []

        try:
            if len(gap.source_nodes) >= 2:
                # Propose relationship between the nodes
                source_id = gap.source_nodes[0]
                target_id = gap.source_nodes[1]

                proposal = ProposedRelationship(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type="semantic",
                    relation_strength=0.6,
                    gap_id=gap.id,
                    reasoning="Semantically similar concepts should be connected",
                    existence_probability=0.7,
                    utility_score=0.8,
                    confidence=gap.confidence * 0.9,
                )
                proposals.append(proposal)

            return proposals

        except Exception as e:
            logger.exception(f"Relationship proposal failed: {e}")
            return []

    async def _deduplicate_gaps(self, gaps: list[DetectedGap]) -> list[DetectedGap]:
        """Remove duplicate gaps."""
        # Simple deduplication based on gap type and source nodes
        seen = set()
        unique_gaps = []

        for gap in gaps:
            key = (gap.gap_type, tuple(sorted(gap.source_nodes)))
            if key not in seen:
                seen.add(key)
                unique_gaps.append(gap)

        return unique_gaps

    async def _rank_gaps_by_priority(self, gaps: list[DetectedGap]) -> list[DetectedGap]:
        """Rank gaps by priority and confidence."""
        # Sort by priority * confidence (descending)
        gaps.sort(key=lambda g: g.priority * g.confidence, reverse=True)
        return gaps

    async def _rank_node_proposals(self, proposals: list[ProposedNode]) -> list[ProposedNode]:
        """Rank node proposals by utility and confidence."""
        proposals.sort(key=lambda p: p.utility_score * p.confidence, reverse=True)
        return proposals

    async def _rank_relationship_proposals(self, proposals: list[ProposedRelationship]) -> list[ProposedRelationship]:
        """Rank relationship proposals by utility and confidence."""
        proposals.sort(key=lambda p: p.utility_score * p.confidence, reverse=True)
        return proposals

    async def _learn_from_validation(self, proposal: ProposedNode | ProposedRelationship, is_accepted: bool):
        """Learn from validation feedback to improve future proposals."""
        # This would implement machine learning to improve proposal quality
        # For now, just log the feedback
        logger.info(f"Validation feedback: {proposal.id} {'accepted' if is_accepted else 'rejected'}")

        # Could implement:
        # - Update detection method weights based on proposal success
        # - Adjust confidence scoring algorithms
        # - Learn patterns in successful vs. unsuccessful proposals

    # Additional analysis methods for completeness assessment

    async def _analyze_structural_completeness(self) -> dict[str, Any]:
        """Analyze structural completeness of the graph."""
        if not self.trust_graph:
            return {"completeness_score": 0.0}

        total_nodes = len(self.trust_graph.nodes)
        total_edges = len(self.trust_graph.edges)

        # Simple structural metrics
        avg_degree = (2 * total_edges) / max(1, total_nodes)
        isolated_nodes = sum(
            1
            for node in self.trust_graph.nodes.values()
            if len(node.incoming_edges) == 0 and len(node.outgoing_edges) == 0
        )

        completeness_score = min(1.0, avg_degree / 3.0)  # Assume 3 is a good average degree

        return {
            "completeness_score": completeness_score,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "average_degree": avg_degree,
            "isolated_nodes": isolated_nodes,
        }

    async def _analyze_semantic_completeness(self) -> dict[str, Any]:
        """Analyze semantic completeness using vector analysis."""
        if not self.vector_engine:
            return {"completeness_score": 0.5}  # Default when no vector engine

        # This would analyze semantic coverage and density
        # For now, return a placeholder
        return {
            "completeness_score": 0.7,
            "semantic_clusters": 5,
            "coverage_areas": ["machine_learning", "neural_networks", "ai"],
        }

    async def _analyze_trust_distribution(self) -> dict[str, Any]:
        """Analyze trust score distribution."""
        if not self.trust_graph:
            return {"avg_trust": 0.0}

        trust_scores = [node.trust_score for node in self.trust_graph.nodes.values()]

        if not trust_scores:
            return {"avg_trust": 0.0}

        return {
            "avg_trust": np.mean(trust_scores),
            "trust_std": np.std(trust_scores),
            "high_trust_nodes": sum(1 for score in trust_scores if score > 0.8),
            "low_trust_nodes": sum(1 for score in trust_scores if score < 0.3),
        }

    async def _analyze_connectivity_patterns(self) -> dict[str, Any]:
        """Analyze connectivity patterns in the graph."""
        if not self.trust_graph:
            return {"connectivity_score": 0.0}

        total_nodes = len(self.trust_graph.nodes)
        total_edges = len(self.trust_graph.edges)

        if total_nodes == 0:
            return {"connectivity_score": 0.0}

        # Simple connectivity metrics
        edge_density = total_edges / (total_nodes * (total_nodes - 1) / 2) if total_nodes > 1 else 0.0
        connectivity_score = min(1.0, edge_density * 10)  # Scale appropriately

        return {
            "connectivity_score": connectivity_score,
            "edge_density": edge_density,
            "total_components": 1,  # Simplified - would need graph traversal for accurate count
        }


if __name__ == "__main__":

    async def test_graph_fixer():
        """Test GraphFixer functionality."""
        # Create system (would normally pass real trust_graph and vector_engine)
        fixer = GraphFixer(
            trust_graph=None,  # Would pass real graph
            vector_engine=None,  # Would pass real engine
            min_confidence_threshold=0.3,
        )
        await fixer.initialize()

        # Test gap detection (with mock data)
        gaps = await fixer.detect_knowledge_gaps(
            query="machine learning neural networks", focus_area="artificial intelligence"
        )
        print(f"Detected {len(gaps)} knowledge gaps")

        # Test solution proposals
        if gaps:
            proposed_nodes, proposed_relationships = await fixer.propose_solutions(gaps)
            print(f"Proposed {len(proposed_nodes)} nodes and {len(proposed_relationships)} relationships")

        # Test graph completeness analysis
        completeness = await fixer.analyze_graph_completeness()
        print(f"Graph completeness analysis: {completeness}")

        # Test statistics
        stats = await fixer.get_gap_statistics()
        print(f"Gap detection statistics: {stats}")

    import asyncio

    asyncio.run(test_graph_fixer())
