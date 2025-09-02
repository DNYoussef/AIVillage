"""
Relationship Analyzer Service

Responsible for analyzing and proposing relationships between nodes
in the knowledge graph. Handles semantic similarity analysis and
relationship scoring.

Extracted from GraphFixer to follow single responsibility principle.
"""

from typing import Any
import uuid

import numpy as np

from ..graph_fixer import DetectedGap, GapType, ProposedRelationship
from ..interfaces.base_service import AsyncServiceMixin, CacheableMixin, ServiceConfig
from ..interfaces.service_interfaces import IRelationshipAnalyzerService


class RelationshipAnalyzerService(IRelationshipAnalyzerService, CacheableMixin, AsyncServiceMixin):
    """
    Service for analyzing and proposing relationships between nodes.

    Capabilities:
    - Semantic similarity analysis
    - Relationship type detection
    - Strength scoring
    - Bridge relationship identification
    - Path completion relationships
    """

    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self.relationship_types = [
            "semantic",
            "associative",
            "causal",
            "hierarchical",
            "temporal",
            "functional",
            "bridging",
        ]
        self.stats = {"relationships_proposed": 0, "similarity_calculations": 0, "avg_relationship_strength": 0.0}

    async def initialize(self) -> bool:
        """Initialize relationship analyzer service."""
        self.logger.info("Initializing RelationshipAnalyzerService...")

        if not self.validate_dependencies(["trust_graph"]):
            self.logger.warning("Trust graph not available - analysis will be limited")

        self._initialized = True
        self.logger.info("✓ RelationshipAnalyzerService initialized")
        return True

    async def cleanup(self) -> None:
        """Clean up service resources."""
        self.clear_cache()
        self._initialized = False

    async def propose_relationships(self, gaps: list[DetectedGap]) -> list[ProposedRelationship]:
        """
        Generate relationship proposals for detected gaps.

        Args:
            gaps: List of detected knowledge gaps

        Returns:
            List of proposed relationships ranked by utility
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            all_proposals = []

            for gap in gaps:
                if gap.gap_type == GapType.MISSING_RELATIONSHIP:
                    proposals = await self._propose_missing_relationships(gap)
                    all_proposals.extend(proposals)

                elif gap.gap_type == GapType.WEAK_CONNECTION:
                    proposals = await self._propose_strengthening_relationships(gap)
                    all_proposals.extend(proposals)

                elif gap.gap_type == GapType.ISOLATED_CLUSTER:
                    proposals = await self._propose_bridge_relationships(gap)
                    all_proposals.extend(proposals)

                elif gap.gap_type == GapType.INCOMPLETE_PATH:
                    proposals = await self._propose_path_completion_relationships(gap)
                    all_proposals.extend(proposals)

            # Score and rank all proposals
            scored_proposals = await self.score_relationships(all_proposals)

            # Update statistics
            self.stats["relationships_proposed"] += len(scored_proposals)
            if scored_proposals:
                avg_strength = sum(p.relation_strength for p in scored_proposals) / len(scored_proposals)
                self.stats["avg_relationship_strength"] = avg_strength

            self.logger.info(f"Generated {len(scored_proposals)} relationship proposals")

            return scored_proposals

        except Exception as e:
            self.logger.exception(f"Relationship proposal generation failed: {e}")
            return []

    async def analyze_semantic_similarity(self, node_id1: str, node_id2: str) -> float:
        """
        Analyze semantic similarity between two nodes.

        Uses vector embeddings and concept analysis to determine
        how semantically related two nodes are.
        """
        if not self.config.trust_graph:
            return 0.0

        cache_key = f"similarity_{min(node_id1, node_id2)}_{max(node_id1, node_id2)}"
        cached_result = self.get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        try:
            similarity = 0.0

            # Get nodes
            if node_id1 not in self.config.trust_graph.nodes or node_id2 not in self.config.trust_graph.nodes:
                self.set_cache(cache_key, similarity)
                return similarity

            node1 = self.config.trust_graph.nodes[node_id1]
            node2 = self.config.trust_graph.nodes[node_id2]

            # Vector embedding similarity
            if hasattr(node1, "embedding") and hasattr(node2, "embedding"):
                if node1.embedding is not None and node2.embedding is not None:
                    similarity = np.dot(node1.embedding, node2.embedding)

            # Concept similarity (simple text-based)
            if hasattr(node1, "concept") and hasattr(node2, "concept"):
                concept_similarity = await self._calculate_concept_similarity(node1.concept, node2.concept)
                similarity = max(similarity, concept_similarity)

            # Normalize to [0, 1]
            similarity = max(0.0, min(1.0, similarity))

            self.stats["similarity_calculations"] += 1
            self.set_cache(cache_key, similarity)

            return similarity

        except Exception as e:
            self.logger.exception(f"Semantic similarity analysis failed: {e}")
            self.set_cache(cache_key, 0.0)
            return 0.0

    async def score_relationships(self, proposals: list[ProposedRelationship]) -> list[ProposedRelationship]:
        """
        Score and rank relationship proposals.

        Scoring factors:
        - Semantic similarity
        - Graph utility
        - Evidence strength
        - Integration feasibility
        """
        try:
            for proposal in proposals:
                # Calculate comprehensive score
                semantic_score = await self._score_semantic_coherence(proposal)
                utility_score = await self._score_graph_utility(proposal)
                evidence_score = await self._score_evidence_strength(proposal)
                feasibility_score = await self._score_integration_feasibility(proposal)

                # Weighted average
                weights = [0.3, 0.3, 0.2, 0.2]  # semantic, utility, evidence, feasibility
                scores = [semantic_score, utility_score, evidence_score, feasibility_score]

                proposal.utility_score = sum(w * s for w, s in zip(weights, scores))
                proposal.confidence = (proposal.existence_probability + proposal.utility_score) / 2.0

            # Sort by utility * confidence
            proposals.sort(key=lambda p: p.utility_score * p.confidence, reverse=True)

            return proposals

        except Exception as e:
            self.logger.exception(f"Relationship scoring failed: {e}")
            return proposals

    # Private implementation methods

    async def _propose_missing_relationships(self, gap: DetectedGap) -> list[ProposedRelationship]:
        """Propose relationships for missing relationship gaps."""
        proposals = []

        if len(gap.source_nodes) >= 2:
            source_id = gap.source_nodes[0]
            target_id = gap.source_nodes[1]

            # Calculate semantic similarity
            similarity = await self.analyze_semantic_similarity(source_id, target_id)

            # Determine relationship type based on similarity and context
            relation_type = await self._determine_relationship_type(source_id, target_id, similarity)

            proposal = ProposedRelationship(
                id=str(uuid.uuid4()),
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                relation_strength=similarity * 0.8,  # Scale down from similarity
                gap_id=gap.id,
                reasoning="Semantically similar concepts should be connected",
                existence_probability=similarity * 0.9,
                utility_score=0.8,  # Will be refined by scoring
                confidence=gap.confidence * 0.9,
                evidence_sources=[f"Semantic similarity: {similarity:.3f}"],
            )
            proposals.append(proposal)

        return proposals

    async def _propose_strengthening_relationships(self, gap: DetectedGap) -> list[ProposedRelationship]:
        """Propose relationships to strengthen weak connections."""
        proposals = []

        try:
            if gap.source_nodes and self.config.trust_graph:
                source_node_id = gap.source_nodes[0]
                self.config.trust_graph.nodes[source_node_id]

                # Find nodes with high trust that could strengthen connections
                high_trust_nodes = [
                    node_id
                    for node_id, node in self.config.trust_graph.nodes.items()
                    if node.trust_score > 0.7 and node_id != source_node_id
                ]

                # Propose connections to high-trust nodes
                for target_id in high_trust_nodes[:3]:  # Limit to top 3
                    similarity = await self.analyze_semantic_similarity(source_node_id, target_id)

                    if similarity > 0.5:  # Only propose if reasonably similar
                        proposal = ProposedRelationship(
                            id=str(uuid.uuid4()),
                            source_id=source_node_id,
                            target_id=target_id,
                            relation_type="associative",
                            relation_strength=similarity,
                            gap_id=gap.id,
                            reasoning="Strengthen weak node by connecting to high-trust node",
                            existence_probability=similarity * 0.7,
                            utility_score=0.7,
                            confidence=gap.confidence * 0.8,
                            evidence_sources=[
                                f"Target trust score: {self.config.trust_graph.nodes[target_id].trust_score:.2f}",
                                f"Semantic similarity: {similarity:.3f}",
                            ],
                        )
                        proposals.append(proposal)

            return proposals[: self.config.max_proposals_per_gap]

        except Exception as e:
            self.logger.exception(f"Strengthening relationship proposal failed: {e}")
            return []

    async def _propose_bridge_relationships(self, gap: DetectedGap) -> list[ProposedRelationship]:
        """Propose relationships to bridge isolated clusters."""
        proposals = []

        try:
            if gap.source_nodes and len(gap.source_nodes) >= 1:
                isolated_node_id = gap.source_nodes[0]

                # Find potential bridge targets (well-connected nodes)
                bridge_candidates = await self._find_bridge_candidates(isolated_node_id)

                for candidate_id in bridge_candidates[:2]:  # Limit to top 2
                    similarity = await self.analyze_semantic_similarity(isolated_node_id, candidate_id)

                    proposal = ProposedRelationship(
                        id=str(uuid.uuid4()),
                        source_id=isolated_node_id,
                        target_id=candidate_id,
                        relation_type="bridging",
                        relation_strength=similarity * 0.9,
                        gap_id=gap.id,
                        reasoning="Bridge isolated node to main knowledge network",
                        existence_probability=similarity * 0.8,
                        utility_score=0.9,  # High utility for bridging
                        confidence=gap.confidence * 0.8,
                        evidence_sources=[
                            f"Bridge candidate connectivity: {await self._get_node_connectivity(candidate_id)}",
                            f"Semantic similarity: {similarity:.3f}",
                        ],
                    )
                    proposals.append(proposal)

            return proposals

        except Exception as e:
            self.logger.exception(f"Bridge relationship proposal failed: {e}")
            return []

    async def _propose_path_completion_relationships(self, gap: DetectedGap) -> list[ProposedRelationship]:
        """Propose relationships to complete reasoning paths."""
        proposals = []

        try:
            if len(gap.source_nodes) >= 2:
                # For incomplete paths, propose direct connections
                source_id = gap.source_nodes[0]
                target_id = gap.source_nodes[1]

                # Check if path exists through intermediates
                path_exists = await self._check_indirect_path(source_id, target_id)

                if not path_exists:
                    similarity = await self.analyze_semantic_similarity(source_id, target_id)

                    proposal = ProposedRelationship(
                        id=str(uuid.uuid4()),
                        source_id=source_id,
                        target_id=target_id,
                        relation_type="functional",
                        relation_strength=similarity * 0.7,
                        gap_id=gap.id,
                        reasoning="Complete reasoning path between disconnected concepts",
                        existence_probability=similarity * 0.8,
                        utility_score=0.8,
                        confidence=gap.confidence * 0.9,
                        evidence_sources=["No indirect path found", f"Similarity: {similarity:.3f}"],
                    )
                    proposals.append(proposal)

            return proposals

        except Exception as e:
            self.logger.exception(f"Path completion proposal failed: {e}")
            return []

    async def _calculate_concept_similarity(self, concept1: str, concept2: str) -> float:
        """Calculate similarity between concept strings."""
        if not concept1 or not concept2:
            return 0.0

        # Simple word overlap similarity
        words1 = set(concept1.lower().split())
        words2 = set(concept2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    async def _determine_relationship_type(self, source_id: str, target_id: str, similarity: float) -> str:
        """Determine the most appropriate relationship type."""
        if similarity > 0.8:
            return "semantic"
        elif similarity > 0.6:
            return "associative"
        else:
            return "functional"

    async def _find_bridge_candidates(self, isolated_node_id: str) -> list[str]:
        """Find well-connected nodes that could serve as bridges."""
        candidates = []

        if not self.config.trust_graph:
            return candidates

        # Find nodes with high connectivity
        for node_id, node in self.config.trust_graph.nodes.items():
            if node_id != isolated_node_id:
                connectivity = len(node.incoming_edges) + len(node.outgoing_edges)
                if connectivity >= 3:  # Well-connected threshold
                    candidates.append((node_id, connectivity))

        # Sort by connectivity and return node IDs
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in candidates[:5]]

    async def _get_node_connectivity(self, node_id: str) -> int:
        """Get connectivity count for a node."""
        if not self.config.trust_graph or node_id not in self.config.trust_graph.nodes:
            return 0

        node = self.config.trust_graph.nodes[node_id]
        return len(node.incoming_edges) + len(node.outgoing_edges)

    async def _check_indirect_path(self, source_id: str, target_id: str, max_depth: int = 3) -> bool:
        """Check if an indirect path exists between two nodes."""
        if not self.config.trust_graph:
            return False

        # Simple BFS to find path (simplified implementation)
        visited = set()
        queue = [(source_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)

            if current_id == target_id:
                return True

            if depth >= max_depth or current_id in visited:
                continue

            visited.add(current_id)

            if current_id in self.config.trust_graph.nodes:
                node = self.config.trust_graph.nodes[current_id]

                # Add outgoing connections to queue
                for edge_id in node.outgoing_edges:
                    if edge_id in self.config.trust_graph.edges:
                        edge = self.config.trust_graph.edges[edge_id]
                        queue.append((edge.target_id, depth + 1))

        return False

    async def _score_semantic_coherence(self, proposal: ProposedRelationship) -> float:
        """Score semantic coherence of a relationship proposal."""
        similarity = await self.analyze_semantic_similarity(proposal.source_id, proposal.target_id)
        return similarity

    async def _score_graph_utility(self, proposal: ProposedRelationship) -> float:
        """Score the graph utility improvement from the relationship."""
        # Base score on relationship type
        type_scores = {
            "semantic": 0.8,
            "associative": 0.7,
            "bridging": 0.9,
            "functional": 0.8,
            "causal": 0.9,
            "hierarchical": 0.7,
            "temporal": 0.6,
        }
        return type_scores.get(proposal.relation_type, 0.6)

    async def _score_evidence_strength(self, proposal: ProposedRelationship) -> float:
        """Score the strength of evidence supporting the relationship."""
        if not proposal.evidence_sources:
            return 0.3

        # Simple scoring based on evidence count and quality indicators
        evidence_count = len(proposal.evidence_sources)
        base_score = min(1.0, evidence_count / 3.0)

        # Look for quality indicators in evidence
        quality_bonus = 0.0
        for evidence in proposal.evidence_sources:
            if "similarity:" in evidence.lower():
                quality_bonus += 0.1
            if "trust score:" in evidence.lower():
                quality_bonus += 0.1

        return min(1.0, base_score + quality_bonus)

    async def _score_integration_feasibility(self, proposal: ProposedRelationship) -> float:
        """Score how easily the relationship can be integrated."""
        # Check if both nodes exist
        if (
            not self.config.trust_graph
            or proposal.source_id not in self.config.trust_graph.nodes
            or proposal.target_id not in self.config.trust_graph.nodes
        ):
            return 0.2

        # Higher feasibility for nodes that already have some connections
        source_connectivity = await self._get_node_connectivity(proposal.source_id)
        target_connectivity = await self._get_node_connectivity(proposal.target_id)

        # Normalize connectivity score
        connectivity_factor = min(1.0, (source_connectivity + target_connectivity) / 10.0)

        return 0.5 + (connectivity_factor * 0.5)  # Range [0.5, 1.0]

    def get_statistics(self) -> dict[str, Any]:
        """Get service statistics."""
        return {
            "relationships_proposed": self.stats["relationships_proposed"],
            "similarity_calculations": self.stats["similarity_calculations"],
            "avg_relationship_strength": self.stats["avg_relationship_strength"],
            "relationship_types": self.relationship_types,
            "cache_size": len(self._cache) if hasattr(self, "_cache") else 0,
        }
