"""
Gap Detection Service

Responsible for detecting various types of knowledge gaps in the graph
including structural, semantic, and logical inconsistencies.

Extracted from GraphFixer to follow single responsibility principle.
"""

import time
from typing import Any

import numpy as np

from ..graph_fixer import DetectedGap, GapType
from ..interfaces.base_service import AsyncServiceMixin, CacheableMixin, ServiceConfig
from ..interfaces.service_interfaces import IGapDetectionService


class GapDetectionService(IGapDetectionService, CacheableMixin, AsyncServiceMixin):
    """
    Service for detecting knowledge gaps in graphs.

    Implements multiple detection algorithms:
    - Structural analysis for connectivity issues
    - Semantic analysis using vector embeddings
    - Path analysis for reasoning chains
    - Trust inconsistency detection
    - Overall connectivity assessment
    """

    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self.detection_methods = [
            "structural_analysis",
            "semantic_clustering",
            "connectivity_analysis",
            "trust_inconsistency",
        ]
        self.stats = {"gaps_detected": 0, "analyses_performed": 0, "cache_hits": 0}

    async def initialize(self) -> bool:
        """Initialize gap detection service."""
        self.logger.info("Initializing GapDetectionService...")

        # Validate minimal dependencies
        if not self.validate_dependencies(["trust_graph"]):
            self.logger.warning("Trust graph not available - some detection methods will be limited")

        self._initialized = True
        self.logger.info("✓ GapDetectionService initialized")
        return True

    async def cleanup(self) -> None:
        """Clean up service resources."""
        self.clear_cache()
        self._initialized = False

    async def detect_gaps(
        self, query: str = None, retrieved_info: list[Any] = None, focus_area: str = None
    ) -> list[DetectedGap]:
        """
        Detect knowledge gaps using multiple analysis methods.

        Args:
            query: Query context for gap detection
            retrieved_info: Information retrieved from various sources
            focus_area: Specific area to focus gap detection on

        Returns:
            List of detected gaps ranked by priority
        """
        if not self.is_initialized:
            await self.initialize()

        start_time = time.time()
        cache_key = f"gaps_{hash((query, focus_area, len(retrieved_info or [])))}"

        # Check cache first
        cached_result = self.get_from_cache(cache_key)
        if cached_result:
            self.stats["cache_hits"] += 1
            return cached_result

        try:
            gaps = []

            # Run all detection methods
            for method in self.detection_methods:
                method_gaps = await self._run_detection_method(method, query, retrieved_info, focus_area)
                gaps.extend(method_gaps)

            # Process and rank gaps
            unique_gaps = await self._deduplicate_gaps(gaps)
            ranked_gaps = await self._rank_gaps_by_priority(unique_gaps)

            # Update statistics
            self.stats["gaps_detected"] += len(ranked_gaps)
            self.stats["analyses_performed"] += 1

            # Cache results
            self.set_cache(cache_key, ranked_gaps)

            analysis_time = (time.time() - start_time) * 1000
            self.logger.info(f"Detected {len(ranked_gaps)} gaps in {analysis_time:.1f}ms")

            return ranked_gaps

        except Exception as e:
            self.logger.exception(f"Gap detection failed: {e}")
            return []

    async def detect_structural_gaps(self) -> list[DetectedGap]:
        """Detect structural gaps in graph connectivity."""
        gaps = []

        if not self.config.trust_graph:
            return gaps

        try:
            # Find isolated nodes
            gaps.extend(await self._find_isolated_nodes())

            # Find weakly connected high-trust nodes
            gaps.extend(await self._find_underconnected_nodes())

            return gaps

        except Exception as e:
            self.logger.exception(f"Structural gap detection failed: {e}")
            return gaps

    async def detect_semantic_gaps(self, query: str = None, focus_area: str = None) -> list[DetectedGap]:
        """Detect semantic gaps using vector analysis."""
        gaps = []

        if not self.config.vector_engine or not self.config.trust_graph:
            return gaps

        try:
            # Find semantically similar but unconnected concepts
            gaps.extend(await self._find_semantic_disconnects())

            # Look for missing bridging concepts
            if focus_area:
                gaps.extend(await self._find_missing_bridges(focus_area))

            return gaps

        except Exception as e:
            self.logger.exception(f"Semantic gap detection failed: {e}")
            return gaps

    async def detect_connectivity_gaps(self) -> list[DetectedGap]:
        """Detect overall connectivity issues."""
        gaps = []

        if not self.config.trust_graph:
            return gaps

        try:
            total_nodes = len(self.config.trust_graph.nodes)
            total_edges = len(self.config.trust_graph.edges)

            if total_nodes > 10:
                expected_edges = total_nodes * 1.5  # Heuristic

                if total_edges < expected_edges * 0.5:
                    gap = DetectedGap(
                        gap_type=GapType.WEAK_CONNECTION,
                        description="Graph appears under-connected overall",
                        evidence=[f"Nodes: {total_nodes}, Edges: {total_edges}, Expected: ~{expected_edges:.0f}"],
                        confidence=0.7,
                        priority=0.6,
                        detection_method="connectivity_analysis",
                    )
                    gaps.append(gap)

            return gaps

        except Exception as e:
            self.logger.exception(f"Connectivity gap detection failed: {e}")
            return gaps

    # Private implementation methods

    async def _run_detection_method(
        self, method: str, query: str, retrieved_info: list[Any], focus_area: str
    ) -> list[DetectedGap]:
        """Run a specific detection method."""
        try:
            if method == "structural_analysis":
                return await self.detect_structural_gaps()
            elif method == "semantic_clustering":
                return await self.detect_semantic_gaps(query, focus_area)
            elif method == "connectivity_analysis":
                return await self.detect_connectivity_gaps()
            elif method == "trust_inconsistency":
                return await self._detect_trust_inconsistencies()
            else:
                self.logger.warning(f"Unknown detection method: {method}")
                return []

        except Exception as e:
            self.logger.warning(f"Detection method {method} failed: {e}")
            return []

    async def _find_isolated_nodes(self) -> list[DetectedGap]:
        """Find completely isolated nodes."""
        gaps = []

        for node_id, node in self.config.trust_graph.nodes.items():
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

        return gaps

    async def _find_underconnected_nodes(self) -> list[DetectedGap]:
        """Find high-trust nodes with few connections."""
        gaps = []

        for node_id, node in self.config.trust_graph.nodes.items():
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

    async def _find_semantic_disconnects(self) -> list[DetectedGap]:
        """Find semantically similar but unconnected concepts."""
        gaps = []

        concept_embeddings = {}
        for node_id, node in self.config.trust_graph.nodes.items():
            if hasattr(node, "embedding") and node.embedding is not None:
                concept_embeddings[node_id] = node.embedding

        if len(concept_embeddings) < 3:
            return gaps

        node_ids = list(concept_embeddings.keys())
        for i, node_id1 in enumerate(node_ids):
            for j, node_id2 in enumerate(node_ids[i + 1 :], i + 1):
                # Calculate semantic similarity
                emb1 = concept_embeddings[node_id1]
                emb2 = concept_embeddings[node_id2]
                similarity = np.dot(emb1, emb2)

                if similarity > 0.7:  # High semantic similarity
                    # Check if they're connected
                    connected = await self._are_nodes_connected(node_id1, node_id2)

                    if not connected:
                        node1 = self.config.trust_graph.nodes[node_id1]
                        node2 = self.config.trust_graph.nodes[node_id2]

                        gap = DetectedGap(
                            gap_type=GapType.MISSING_RELATIONSHIP,
                            source_nodes=[node_id1, node_id2],
                            description=f"Semantically similar concepts '{node1.concept}' and '{node2.concept}' are not connected",
                            evidence=[f"Semantic similarity: {similarity:.3f}, but no direct relationship"],
                            confidence=similarity * 0.8,
                            priority=0.6,
                            detection_method="semantic_clustering",
                        )
                        gaps.append(gap)

        return gaps

    async def _find_missing_bridges(self, focus_area: str) -> list[DetectedGap]:
        """Find missing bridge concepts in a focus area."""
        gaps = []

        # This would implement more sophisticated bridge detection
        # For now, return empty list

        return gaps

    async def _detect_trust_inconsistencies(self) -> list[DetectedGap]:
        """Detect inconsistencies in trust scores."""
        gaps = []

        if not self.config.trust_graph:
            return gaps

        for node_id, node in self.config.trust_graph.nodes.items():
            incoming_trusts = []

            for edge_id in node.incoming_edges:
                if edge_id in self.config.trust_graph.edges:
                    edge = self.config.trust_graph.edges[edge_id]
                    source_node = self.config.trust_graph.nodes[edge.source_id]
                    incoming_trusts.append(source_node.trust_score)

            if len(incoming_trusts) >= 2:
                trust_variance = np.var(incoming_trusts)

                if trust_variance > 0.2:  # Significant variance
                    gap = DetectedGap(
                        gap_type=GapType.CONFLICTING_INFO,
                        source_nodes=[node_id],
                        description=f"Node '{node.concept}' has conflicting trust information",
                        evidence=[f"Trust variance: {trust_variance:.3f}, incoming trusts: {incoming_trusts}"],
                        confidence=0.8,
                        priority=0.8,
                        detection_method="trust_inconsistency",
                    )
                    gaps.append(gap)

        return gaps

    async def _are_nodes_connected(self, node_id1: str, node_id2: str) -> bool:
        """Check if two nodes are directly connected."""
        return any(
            (edge.source_id == node_id1 and edge.target_id == node_id2)
            or (edge.source_id == node_id2 and edge.target_id == node_id1)
            for edge in self.config.trust_graph.edges.values()
        )

    async def _deduplicate_gaps(self, gaps: list[DetectedGap]) -> list[DetectedGap]:
        """Remove duplicate gaps based on type and source nodes."""
        seen = set()
        unique_gaps = []

        for gap in gaps:
            key = (gap.gap_type, tuple(sorted(gap.source_nodes)))
            if key not in seen:
                seen.add(key)
                unique_gaps.append(gap)

        return unique_gaps

    async def _rank_gaps_by_priority(self, gaps: list[DetectedGap]) -> list[DetectedGap]:
        """Rank gaps by priority * confidence score."""
        gaps.sort(key=lambda g: g.priority * g.confidence, reverse=True)
        return gaps

    def get_statistics(self) -> dict:
        """Get service statistics."""
        return {
            "gaps_detected": self.stats["gaps_detected"],
            "analyses_performed": self.stats["analyses_performed"],
            "cache_hits": self.stats["cache_hits"],
            "cache_size": len(self._cache) if hasattr(self, "_cache") else 0,
            "detection_methods": self.detection_methods,
        }
