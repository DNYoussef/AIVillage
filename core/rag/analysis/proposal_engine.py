"""
Proposal Engine for Knowledge Graph Solutions

Focused module for generating proposals to fix detected gaps.
Uses strategy pattern to handle different proposal types.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Protocol

from .graph_types import DetectedGap, GapType, ProposedNode, ProposedRelationship

logger = logging.getLogger(__name__)


class TrustGraphProtocol(Protocol):
    """Protocol for trust graph dependency."""

    @property
    def nodes(self) -> dict[str, Any]:
        """Node collection."""
        ...

    @property
    def edges(self) -> dict[str, Any]:
        """Edge collection."""
        ...


class ProposalStrategy(ABC):
    """Abstract strategy for gap resolution proposals."""

    @abstractmethod
    async def propose_nodes(self, gap: DetectedGap) -> list[ProposedNode]:
        """Generate node proposals for the gap."""
        ...

    @abstractmethod
    async def propose_relationships(self, gap: DetectedGap) -> list[ProposedRelationship]:
        """Generate relationship proposals for the gap."""
        ...


class MissingNodeStrategy(ProposalStrategy):
    """Strategy for proposing missing nodes."""

    def __init__(self, trust_graph: TrustGraphProtocol | None = None, max_proposals: int = 3):
        self._trust_graph = trust_graph
        self._max_proposals = max_proposals

    async def propose_nodes(self, gap: DetectedGap) -> list[ProposedNode]:
        """Propose bridging nodes based on neighboring concepts."""
        if not gap.source_nodes or not self._trust_graph:
            return []

        neighboring_concepts = self._collect_neighboring_concepts(gap.source_nodes)

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
            return [proposal]

        return []

    async def propose_relationships(self, gap: DetectedGap) -> list[ProposedRelationship]:
        """Missing node strategy doesn't propose relationships directly."""
        return []

    def _collect_neighboring_concepts(self, source_nodes: list[str]) -> set[str]:
        """Collect concepts from nodes connected to source nodes."""
        if not self._trust_graph:
            return set()

        neighboring_concepts = set()

        for node_id in source_nodes:
            if node_id not in self._trust_graph.nodes:
                continue

            node = self._trust_graph.nodes[node_id]

            # Get connected node concepts
            for edge_id in getattr(node, "outgoing_edges", set()) | getattr(node, "incoming_edges", set()):
                if edge_id not in self._trust_graph.edges:
                    continue

                edge = self._trust_graph.edges[edge_id]
                other_node_id = edge.target_id if edge.source_id == node_id else edge.source_id

                if other_node_id in self._trust_graph.nodes:
                    other_node = self._trust_graph.nodes[other_node_id]
                    concept = getattr(other_node, "concept", "")
                    if concept:
                        neighboring_concepts.add(concept)

        return neighboring_concepts


class MissingRelationshipStrategy(ProposalStrategy):
    """Strategy for proposing missing relationships."""

    def __init__(self, trust_graph: TrustGraphProtocol | None = None):
        self._trust_graph = trust_graph

    async def propose_nodes(self, gap: DetectedGap) -> list[ProposedNode]:
        """Missing relationship strategy focuses on relationships."""
        return []

    async def propose_relationships(self, gap: DetectedGap) -> list[ProposedRelationship]:
        """Propose relationships between semantically similar nodes."""
        if len(gap.source_nodes) < 2:
            return []

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

        return [proposal]


class WeakConnectionStrategy(ProposalStrategy):
    """Strategy for strengthening weak connections."""

    def __init__(self, trust_graph: TrustGraphProtocol | None = None):
        self._trust_graph = trust_graph

    async def propose_nodes(self, gap: DetectedGap) -> list[ProposedNode]:
        """Weak connection strategy focuses on relationships."""
        return []

    async def propose_relationships(self, gap: DetectedGap) -> list[ProposedRelationship]:
        """Propose strengthening relationships for weakly connected nodes."""
        if not gap.source_nodes or not self._trust_graph:
            return []

        proposals = []
        source_node_id = gap.source_nodes[0]

        # Find high-trust nodes that could connect to this node
        for node_id, node in self._trust_graph.nodes.items():
            if (
                node_id != source_node_id
                and getattr(node, "trust_score", 0) > 0.7
                and not self._are_nodes_connected(source_node_id, node_id)
            ):
                proposal = ProposedRelationship(
                    source_id=source_node_id,
                    target_id=node_id,
                    relation_type="supporting",
                    relation_strength=0.5,
                    gap_id=gap.id,
                    reasoning="Connect under-connected high-trust node to well-connected high-trust node",
                    existence_probability=0.6,
                    utility_score=0.7,
                    confidence=0.6,
                )
                proposals.append(proposal)

                # Limit proposals
                if len(proposals) >= 2:
                    break

        return proposals

    def _are_nodes_connected(self, node_id1: str, node_id2: str) -> bool:
        """Check if two nodes are connected."""
        if not self._trust_graph:
            return False

        return any(
            (edge.source_id == node_id1 and edge.target_id == node_id2)
            or (edge.source_id == node_id2 and edge.target_id == node_id1)
            for edge in self._trust_graph.edges.values()
        )


class IsolatedClusterStrategy(ProposalStrategy):
    """Strategy for connecting isolated clusters."""

    def __init__(self, trust_graph: TrustGraphProtocol | None = None):
        self._trust_graph = trust_graph

    async def propose_nodes(self, gap: DetectedGap) -> list[ProposedNode]:
        """Isolated cluster strategy focuses on bridge relationships."""
        return []

    async def propose_relationships(self, gap: DetectedGap) -> list[ProposedRelationship]:
        """Propose bridge relationships to connect isolated nodes."""
        if not gap.source_nodes or not self._trust_graph:
            return []

        isolated_node_id = gap.source_nodes[0]
        proposals = []

        # Find suitable nodes to connect to
        for node_id, node in self._trust_graph.nodes.items():
            if node_id != isolated_node_id and getattr(node, "trust_score", 0) > 0.5:
                proposal = ProposedRelationship(
                    source_id=isolated_node_id,
                    target_id=node_id,
                    relation_type="bridge",
                    relation_strength=0.4,
                    gap_id=gap.id,
                    reasoning="Connect isolated node to established network",
                    existence_probability=0.5,
                    utility_score=0.8,
                    confidence=0.6,
                )
                proposals.append(proposal)

                # Limit to avoid overwhelming
                if len(proposals) >= 3:
                    break

        return proposals


class IncompletePathStrategy(ProposalStrategy):
    """Strategy for completing reasoning paths."""

    def __init__(self, trust_graph: TrustGraphProtocol | None = None):
        self._trust_graph = trust_graph

    async def propose_nodes(self, gap: DetectedGap) -> list[ProposedNode]:
        """Propose intermediate nodes for path completion."""
        if not gap.source_nodes or len(gap.source_nodes) < 2:
            return []

        start_node = gap.source_nodes[0]
        end_node = gap.source_nodes[-1]

        # Propose intermediate concept
        intermediate_concept = f"intermediate_step_{start_node}_to_{end_node}"

        proposal = ProposedNode(
            content=f"Intermediate reasoning step between {start_node} and {end_node}",
            concept=intermediate_concept,
            gap_id=gap.id,
            reasoning="Would complete broken reasoning chain",
            existence_probability=0.5,
            utility_score=0.7,
            confidence=0.5,
            suggested_trust_score=0.4,
        )

        return [proposal]

    async def propose_relationships(self, gap: DetectedGap) -> list[ProposedRelationship]:
        """Propose relationships for path completion."""
        if not gap.source_nodes or len(gap.source_nodes) < 2:
            return []

        proposals = []

        # Create sequential relationships
        for i in range(len(gap.source_nodes) - 1):
            source_id = gap.source_nodes[i]
            target_id = gap.source_nodes[i + 1]

            proposal = ProposedRelationship(
                source_id=source_id,
                target_id=target_id,
                relation_type="sequential",
                relation_strength=0.6,
                gap_id=gap.id,
                reasoning="Complete reasoning path sequence",
                existence_probability=0.6,
                utility_score=0.7,
                confidence=0.6,
            )
            proposals.append(proposal)

        return proposals


class ProposalEngine:
    """
    Engine for generating proposals to fix knowledge gaps.

    Uses strategy pattern to handle different gap types.
    Follows single responsibility and dependency injection principles.
    """

    def __init__(self, trust_graph: TrustGraphProtocol | None = None, max_proposals_per_gap: int = 3):
        self._trust_graph = trust_graph
        self._max_proposals_per_gap = max_proposals_per_gap

        # Strategy registry - single source of truth for proposal strategies
        self._strategies = {
            GapType.MISSING_NODE: MissingNodeStrategy(trust_graph, max_proposals_per_gap),
            GapType.MISSING_RELATIONSHIP: MissingRelationshipStrategy(trust_graph),
            GapType.WEAK_CONNECTION: WeakConnectionStrategy(trust_graph),
            GapType.ISOLATED_CLUSTER: IsolatedClusterStrategy(trust_graph),
            GapType.INCOMPLETE_PATH: IncompletePathStrategy(trust_graph),
        }

    async def generate_proposals(
        self, gaps: list[DetectedGap], max_proposals: int | None = None
    ) -> tuple[list[ProposedNode], list[ProposedRelationship]]:
        """
        Generate proposals to fix detected gaps.

        Args:
            gaps: List of detected gaps to fix
            max_proposals: Maximum number of gaps to process

        Returns:
            Tuple of (proposed_nodes, proposed_relationships)
        """
        limit = max_proposals or len(gaps)
        all_nodes = []
        all_relationships = []

        for gap in gaps[:limit]:
            strategy = self._strategies.get(gap.gap_type)
            if not strategy:
                logger.warning(f"No strategy for gap type: {gap.gap_type}")
                continue

            try:
                # Generate proposals using appropriate strategy
                nodes = await strategy.propose_nodes(gap)
                relationships = await strategy.propose_relationships(gap)

                all_nodes.extend(nodes)
                all_relationships.extend(relationships)

            except Exception as e:
                logger.exception(f"Proposal generation failed for gap {gap.id}: {e}")

        # Rank proposals by utility and confidence
        ranked_nodes = self._rank_node_proposals(all_nodes)
        ranked_relationships = self._rank_relationship_proposals(all_relationships)

        return ranked_nodes, ranked_relationships

    def _rank_node_proposals(self, proposals: list[ProposedNode]) -> list[ProposedNode]:
        """Rank node proposals by utility and confidence."""
        return sorted(proposals, key=lambda p: p.utility_score * p.confidence, reverse=True)

    def _rank_relationship_proposals(self, proposals: list[ProposedRelationship]) -> list[ProposedRelationship]:
        """Rank relationship proposals by utility and confidence."""
        return sorted(proposals, key=lambda p: p.utility_score * p.confidence, reverse=True)
