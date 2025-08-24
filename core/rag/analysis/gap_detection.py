"""
Gap Detection Engine for Knowledge Graphs

Focused module for detecting various types of knowledge gaps using
different analysis methods. Follows single responsibility principle
and weak connascence through dependency injection.
"""

import asyncio
from dataclasses import dataclass
import logging
from typing import Any, Protocol

import numpy as np

from .graph_types import DetectedGap, GapType

logger = logging.getLogger(__name__)


class TrustGraphProtocol(Protocol):
    """Protocol for trust graph dependency - weak connascence via interface."""

    @property
    def nodes(self) -> dict[str, Any]:
        """Node collection."""
        ...

    @property
    def edges(self) -> dict[str, Any]:
        """Edge collection."""
        ...


class VectorEngineProtocol(Protocol):
    """Protocol for vector engine dependency - weak connascence via interface."""

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding for text."""
        ...


@dataclass(frozen=True)
class DetectionConfig:
    """Immutable configuration for gap detection - no magic values."""

    SIMILARITY_THRESHOLD: float = 0.7
    MIN_CONNECTIONS_FOR_HIGH_TRUST: int = 3
    HIGH_TRUST_THRESHOLD: float = 0.6
    TRUST_VARIANCE_THRESHOLD: float = 0.2
    MIN_NODES_FOR_ANALYSIS: int = 3
    EXPECTED_EDGES_MULTIPLIER: float = 1.5
    CONNECTIVITY_THRESHOLD_RATIO: float = 0.5


class GapDetectionEngine:
    """
    Engine for detecting knowledge gaps in graphs.

    Uses dependency injection and single responsibility principle.
    Each detection method is focused on one type of gap.
    """

    def __init__(
        self,
        trust_graph: TrustGraphProtocol | None = None,
        vector_engine: VectorEngineProtocol | None = None,
        config: DetectionConfig | None = None,
    ):
        self._trust_graph = trust_graph
        self._vector_engine = vector_engine
        self._config = config or DetectionConfig()

        # Method registry - single source of truth for detection methods
        self._detection_methods = {
            "structural_analysis": self._detect_structural_gaps,
            "semantic_clustering": self._detect_semantic_gaps,
            "path_analysis": self._detect_path_gaps,
            "trust_inconsistency": self._detect_trust_inconsistencies,
            "connectivity_analysis": self._detect_connectivity_gaps,
        }

    async def detect_gaps(
        self,
        query: str | None = None,
        retrieved_info: list[Any] | None = None,
        focus_area: str | None = None,
        methods: list[str] | None = None,
    ) -> list[DetectedGap]:
        """
        Detect knowledge gaps using specified methods.

        Args:
            query: Analysis query context
            retrieved_info: Information for path analysis
            focus_area: Specific area to focus on
            methods: Specific detection methods to use

        Returns:
            List of detected gaps, deduplicated and ranked
        """
        methods = methods or list(self._detection_methods.keys())

        # Run detection methods concurrently
        tasks = [
            self._run_detection_method(method, query, retrieved_info, focus_area)
            for method in methods
            if method in self._detection_methods
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect gaps, filtering out exceptions
        all_gaps = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Detection method {methods[i]} failed: {result}")
            else:
                all_gaps.extend(result)

        # Deduplicate and rank
        unique_gaps = self._deduplicate_gaps(all_gaps)
        return self._rank_gaps_by_priority(unique_gaps)

    async def _run_detection_method(
        self, method: str, query: str | None, retrieved_info: list[Any] | None, focus_area: str | None
    ) -> list[DetectedGap]:
        """Run a specific detection method."""
        try:
            detection_func = self._detection_methods[method]

            # Call with appropriate parameters based on method signature
            if method == "semantic_clustering":
                return await detection_func(query, focus_area)
            elif method == "path_analysis":
                return await detection_func(retrieved_info)
            else:
                return await detection_func()

        except Exception as e:
            logger.warning(f"Detection method {method} failed: {e}")
            return []

    async def _detect_structural_gaps(self) -> list[DetectedGap]:
        """Detect structural gaps in the graph."""
        if not self._trust_graph:
            return []

        gaps = []

        # Find isolated nodes
        for node_id, node in self._trust_graph.nodes.items():
            total_connections = len(getattr(node, "incoming_edges", [])) + len(getattr(node, "outgoing_edges", []))

            if total_connections == 0:
                gaps.append(
                    DetectedGap(
                        gap_type=GapType.ISOLATED_CLUSTER,
                        source_nodes=[node_id],
                        description=f"Node '{getattr(node, 'concept', node_id)}' has no connections",
                        evidence=[f"Node {node_id} has 0 connections"],
                        confidence=0.9,
                        priority=0.6,
                        detection_method="structural_analysis",
                    )
                )
            elif 1 <= total_connections <= 2 and getattr(node, "trust_score", 0) > self._config.HIGH_TRUST_THRESHOLD:
                gaps.append(
                    DetectedGap(
                        gap_type=GapType.WEAK_CONNECTION,
                        source_nodes=[node_id],
                        description=f"High-trust node '{getattr(node, 'concept', node_id)}' is under-connected",
                        evidence=[f"Node {node_id} has only {total_connections} connections but high trust"],
                        confidence=0.7,
                        priority=0.5,
                        detection_method="structural_analysis",
                    )
                )

        return gaps

    async def _detect_semantic_gaps(self, query: str | None, focus_area: str | None) -> list[DetectedGap]:
        """Detect semantic gaps using vector analysis."""
        if not self._vector_engine or not self._trust_graph:
            return []

        # Get concept embeddings
        concept_embeddings = {}
        for node_id, node in self._trust_graph.nodes.items():
            embedding = getattr(node, "embedding", None)
            if embedding is not None:
                concept_embeddings[node_id] = embedding

        if len(concept_embeddings) < self._config.MIN_NODES_FOR_ANALYSIS:
            return []

        gaps = []
        node_ids = list(concept_embeddings.keys())

        for i, node_id1 in enumerate(node_ids):
            for node_id2 in node_ids[i + 1 :]:
                similarity = self._calculate_semantic_similarity(
                    concept_embeddings[node_id1], concept_embeddings[node_id2]
                )

                if similarity > self._config.SIMILARITY_THRESHOLD:
                    if not self._are_nodes_connected(node_id1, node_id2):
                        node1 = self._trust_graph.nodes[node_id1]
                        node2 = self._trust_graph.nodes[node_id2]

                        gaps.append(
                            DetectedGap(
                                gap_type=GapType.MISSING_RELATIONSHIP,
                                source_nodes=[node_id1, node_id2],
                                description=(
                                    f"Semantically similar concepts "
                                    f"'{getattr(node1, 'concept', node_id1)}' and "
                                    f"'{getattr(node2, 'concept', node_id2)}' are not connected"
                                ),
                                evidence=[f"Semantic similarity: {similarity:.3f}"],
                                confidence=similarity * 0.8,
                                priority=0.6,
                                detection_method="semantic_clustering",
                            )
                        )

        return gaps

    async def _detect_path_gaps(self, retrieved_info: list[Any] | None) -> list[DetectedGap]:
        """Detect gaps in reasoning paths."""
        if not retrieved_info or not self._trust_graph:
            return []

        # Group info by source
        sources_info = self._group_info_by_source(retrieved_info)

        if "vector" not in sources_info or "graph" not in sources_info:
            return []

        # Extract concepts from different sources
        vector_concepts = self._extract_concepts_from_source(sources_info["vector"])
        graph_concepts = self._extract_concepts_from_source(sources_info["graph"])

        # Find missing concepts
        missing_in_graph = vector_concepts - graph_concepts

        if missing_in_graph:
            return [
                DetectedGap(
                    gap_type=GapType.MISSING_NODE,
                    description="Concepts found in vector search but missing from knowledge graph",
                    evidence=[f"Missing concepts: {list(missing_in_graph)[:5]}"],
                    confidence=0.6,
                    priority=0.7,
                    detection_method="path_analysis",
                )
            ]

        return []

    async def _detect_trust_inconsistencies(self) -> list[DetectedGap]:
        """Detect inconsistencies in trust scores."""
        if not self._trust_graph:
            return []

        gaps = []

        for node_id, node in self._trust_graph.nodes.items():
            incoming_trusts = self._get_incoming_trust_scores(node, node_id)

            if len(incoming_trusts) >= 2:
                trust_variance = np.var(incoming_trusts)

                if trust_variance > self._config.TRUST_VARIANCE_THRESHOLD:
                    gaps.append(
                        DetectedGap(
                            gap_type=GapType.CONFLICTING_INFO,
                            source_nodes=[node_id],
                            description=(
                                f"Node '{getattr(node, 'concept', node_id)}' " f"has conflicting trust information"
                            ),
                            evidence=[f"Trust variance: {trust_variance:.3f}"],
                            confidence=0.8,
                            priority=0.8,
                            detection_method="trust_inconsistency",
                        )
                    )

        return gaps

    async def _detect_connectivity_gaps(self) -> list[DetectedGap]:
        """Detect overall connectivity issues."""
        if not self._trust_graph:
            return []

        total_nodes = len(self._trust_graph.nodes)
        total_edges = len(self._trust_graph.edges)

        if total_nodes <= 10:
            return []

        expected_edges = total_nodes * self._config.EXPECTED_EDGES_MULTIPLIER

        if total_edges < expected_edges * self._config.CONNECTIVITY_THRESHOLD_RATIO:
            return [
                DetectedGap(
                    gap_type=GapType.WEAK_CONNECTION,
                    description="Graph appears under-connected overall",
                    evidence=[f"Nodes: {total_nodes}, Edges: {total_edges}, Expected: ~{expected_edges:.0f}"],
                    confidence=0.7,
                    priority=0.6,
                    detection_method="connectivity_analysis",
                )
            ]

        return []

    # Helper methods with single responsibility

    def _calculate_semantic_similarity(self, emb1: list[float], emb2: list[float]) -> float:
        """Calculate semantic similarity between embeddings."""
        return float(np.dot(emb1, emb2))

    def _are_nodes_connected(self, node_id1: str, node_id2: str) -> bool:
        """Check if two nodes are directly connected."""
        if not self._trust_graph:
            return False

        return any(
            (edge.source_id == node_id1 and edge.target_id == node_id2)
            or (edge.source_id == node_id2 and edge.target_id == node_id1)
            for edge in self._trust_graph.edges.values()
        )

    def _group_info_by_source(self, retrieved_info: list[Any]) -> dict[str, list[Any]]:
        """Group retrieved information by source."""
        sources_info = {}
        for info in retrieved_info:
            source = getattr(info, "source", "unknown")
            if source not in sources_info:
                sources_info[source] = []
            sources_info[source].append(info)
        return sources_info

    def _extract_concepts_from_source(self, source_info: list[Any]) -> set[str]:
        """Extract concepts from source information."""
        concepts = set()
        for info in source_info:
            content = getattr(info, "content", "")
            words = content.lower().split()
            concepts.update(word for word in words if len(word) > 4)
        return concepts

    def _get_incoming_trust_scores(self, node: Any, node_id: str) -> list[float]:
        """Get trust scores from incoming edges."""
        if not self._trust_graph:
            return []

        incoming_trusts = []
        for edge_id in getattr(node, "incoming_edges", []):
            if edge_id in self._trust_graph.edges:
                edge = self._trust_graph.edges[edge_id]
                source_node = self._trust_graph.nodes[edge.source_id]
                trust_score = getattr(source_node, "trust_score", 0.0)
                incoming_trusts.append(trust_score)

        return incoming_trusts

    def _deduplicate_gaps(self, gaps: list[DetectedGap]) -> list[DetectedGap]:
        """Remove duplicate gaps based on type and source nodes."""
        seen = set()
        unique_gaps = []

        for gap in gaps:
            key = (gap.gap_type, tuple(sorted(gap.source_nodes)))
            if key not in seen:
                seen.add(key)
                unique_gaps.append(gap)

        return unique_gaps

    def _rank_gaps_by_priority(self, gaps: list[DetectedGap]) -> list[DetectedGap]:
        """Rank gaps by priority and confidence."""
        return sorted(gaps, key=lambda g: g.priority * g.confidence, reverse=True)
