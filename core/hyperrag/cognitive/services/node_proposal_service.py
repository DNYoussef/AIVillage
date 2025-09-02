"""
Node Proposal Service

Responsible for generating proposed nodes to fill detected knowledge gaps.
Calculates existence probabilities and utility scores for proposals.

Extracted from GraphFixer to follow single responsibility principle.
"""

from typing import Any
import uuid

from ..graph_fixer import DetectedGap, GapType, ProposedNode
from ..interfaces.base_service import AsyncServiceMixin, CacheableMixin, ServiceConfig
from ..interfaces.service_interfaces import INodeProposalService


class NodeProposalService(INodeProposalService, CacheableMixin, AsyncServiceMixin):
    """
    Service for proposing new nodes to fill knowledge gaps.

    Generates node proposals based on:
    - Gap analysis context
    - Neighboring concept analysis
    - Semantic bridging opportunities
    - Evidence-based reasoning
    """

    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self.proposal_strategies = [
            "bridging_concepts",
            "missing_intermediates",
            "context_expansion",
            "semantic_completion",
        ]
        self.stats = {"nodes_proposed": 0, "proposals_generated": 0, "avg_confidence": 0.0}

    async def initialize(self) -> bool:
        """Initialize node proposal service."""
        self.logger.info("Initializing NodeProposalService...")

        if not self.validate_dependencies(["trust_graph"]):
            self.logger.warning("Trust graph not available - proposals will be limited")

        self._initialized = True
        self.logger.info("✓ NodeProposalService initialized")
        return True

    async def cleanup(self) -> None:
        """Clean up service resources."""
        self.clear_cache()
        self._initialized = False

    async def propose_nodes(self, gaps: list[DetectedGap]) -> list[ProposedNode]:
        """
        Generate node proposals for detected gaps.

        Args:
            gaps: List of detected knowledge gaps

        Returns:
            List of proposed nodes ranked by utility and confidence
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            all_proposals = []

            for gap in gaps:
                if gap.gap_type == GapType.MISSING_NODE:
                    proposals = await self._propose_missing_nodes(gap)
                    all_proposals.extend(proposals)

                elif gap.gap_type == GapType.ISOLATED_CLUSTER:
                    proposals = await self._propose_bridging_nodes(gap)
                    all_proposals.extend(proposals)

                elif gap.gap_type == GapType.INCOMPLETE_PATH:
                    proposals = await self._propose_path_nodes(gap)
                    all_proposals.extend(proposals)

            # Rank proposals by utility and confidence
            ranked_proposals = await self._rank_proposals(all_proposals)

            # Update statistics
            self.stats["nodes_proposed"] += len(ranked_proposals)
            self.stats["proposals_generated"] += 1
            if ranked_proposals:
                self.stats["avg_confidence"] = sum(p.confidence for p in ranked_proposals) / len(ranked_proposals)

            self.logger.info(f"Generated {len(ranked_proposals)} node proposals")

            return ranked_proposals

        except Exception as e:
            self.logger.exception(f"Node proposal generation failed: {e}")
            return []

    async def calculate_existence_probability(self, gap: DetectedGap) -> float:
        """
        Calculate probability that a proposed node should exist.

        Based on:
        - Gap confidence level
        - Context richness
        - Neighboring node evidence
        - Semantic coherence
        """
        try:
            base_probability = gap.confidence

            # Adjust based on evidence strength
            evidence_factor = min(1.0, len(gap.evidence) / 3.0)

            # Adjust based on gap type
            type_factors = {
                GapType.MISSING_NODE: 0.8,
                GapType.ISOLATED_CLUSTER: 0.6,
                GapType.INCOMPLETE_PATH: 0.7,
                GapType.WEAK_CONNECTION: 0.5,
            }
            type_factor = type_factors.get(gap.gap_type, 0.5)

            # Calculate context richness
            context_factor = await self._calculate_context_richness(gap)

            probability = base_probability * evidence_factor * type_factor * context_factor
            return min(1.0, max(0.1, probability))

        except Exception as e:
            self.logger.exception(f"Existence probability calculation failed: {e}")
            return 0.5

    async def score_utility(self, proposal: ProposedNode, gap: DetectedGap) -> float:
        """
        Score the utility of a proposed node.

        Considers:
        - Potential connectivity improvements
        - Knowledge coverage enhancement
        - Gap resolution effectiveness
        - Integration feasibility
        """
        try:
            utility_factors = []

            # Base utility from gap priority
            utility_factors.append(gap.priority)

            # Connectivity potential
            connectivity_score = await self._assess_connectivity_potential(proposal, gap)
            utility_factors.append(connectivity_score)

            # Knowledge coverage improvement
            coverage_score = await self._assess_coverage_improvement(proposal, gap)
            utility_factors.append(coverage_score)

            # Integration feasibility
            integration_score = await self._assess_integration_feasibility(proposal)
            utility_factors.append(integration_score)

            # Calculate weighted average
            if utility_factors:
                utility = sum(utility_factors) / len(utility_factors)
                return min(1.0, max(0.1, utility))

            return 0.5

        except Exception as e:
            self.logger.exception(f"Utility scoring failed: {e}")
            return 0.5

    # Private implementation methods

    async def _propose_missing_nodes(self, gap: DetectedGap) -> list[ProposedNode]:
        """Propose nodes for missing node gaps."""
        proposals = []

        if not gap.source_nodes or not self.config.trust_graph:
            return proposals

        try:
            # Analyze neighboring concepts
            neighboring_concepts = await self._get_neighboring_concepts(gap.source_nodes)

            if len(neighboring_concepts) >= 2:
                # Generate bridging concept proposal
                concept_list = list(neighboring_concepts)[:2]
                proposal = await self._create_bridging_proposal(gap, concept_list)
                proposals.append(proposal)

            # Generate context-based proposals
            context_proposals = await self._generate_context_proposals(gap)
            proposals.extend(context_proposals)

            return proposals[: self.config.max_proposals_per_gap]

        except Exception as e:
            self.logger.exception(f"Missing node proposal failed: {e}")
            return []

    async def _propose_bridging_nodes(self, gap: DetectedGap) -> list[ProposedNode]:
        """Propose nodes to bridge isolated clusters."""
        proposals = []

        try:
            if gap.source_nodes:
                for node_id in gap.source_nodes:
                    # Find potential bridging concepts
                    bridging_concept = f"bridge_for_{node_id}"

                    proposal = ProposedNode(
                        id=str(uuid.uuid4()),
                        content=f"Bridging concept to connect isolated node {node_id}",
                        concept=bridging_concept,
                        gap_id=gap.id,
                        reasoning="Connects isolated node to main knowledge network",
                        existence_probability=await self.calculate_existence_probability(gap),
                        utility_score=0.7,
                        confidence=gap.confidence * 0.8,
                        suggested_trust_score=0.6,
                    )
                    proposals.append(proposal)

            return proposals

        except Exception as e:
            self.logger.exception(f"Bridging node proposal failed: {e}")
            return []

    async def _propose_path_nodes(self, gap: DetectedGap) -> list[ProposedNode]:
        """Propose nodes to complete reasoning paths."""
        proposals = []

        try:
            if len(gap.source_nodes) >= 2:
                source_id = gap.source_nodes[0]
                target_id = gap.source_nodes[1]

                intermediate_concept = f"intermediate_{source_id}_{target_id}"

                proposal = ProposedNode(
                    id=str(uuid.uuid4()),
                    content=f"Intermediate concept between {source_id} and {target_id}",
                    concept=intermediate_concept,
                    gap_id=gap.id,
                    reasoning="Completes reasoning path between disconnected concepts",
                    existence_probability=await self.calculate_existence_probability(gap),
                    utility_score=0.8,
                    confidence=gap.confidence * 0.9,
                    suggested_trust_score=0.7,
                )
                proposals.append(proposal)

            return proposals

        except Exception as e:
            self.logger.exception(f"Path node proposal failed: {e}")
            return []

    async def _get_neighboring_concepts(self, node_ids: list[str]) -> set[str]:
        """Get concepts from nodes neighboring the given nodes."""
        neighboring_concepts = set()

        if not self.config.trust_graph:
            return neighboring_concepts

        for node_id in node_ids:
            if node_id in self.config.trust_graph.nodes:
                node = self.config.trust_graph.nodes[node_id]

                # Collect concepts from connected nodes
                for edge_id in node.outgoing_edges | node.incoming_edges:
                    if edge_id in self.config.trust_graph.edges:
                        edge = self.config.trust_graph.edges[edge_id]
                        other_node_id = edge.target_id if edge.source_id == node_id else edge.source_id

                        if other_node_id in self.config.trust_graph.nodes:
                            other_node = self.config.trust_graph.nodes[other_node_id]
                            neighboring_concepts.add(other_node.concept)

        return neighboring_concepts

    async def _create_bridging_proposal(self, gap: DetectedGap, concepts: list[str]) -> ProposedNode:
        """Create a proposal for a bridging concept."""
        bridging_concept = f"bridge_concept_for_{concepts[0]}_and_{concepts[1]}"

        proposal = ProposedNode(
            id=str(uuid.uuid4()),
            content=f"Bridging concept connecting {concepts[0]} and {concepts[1]}",
            concept=bridging_concept,
            gap_id=gap.id,
            reasoning="Would help connect disparate concepts in the knowledge graph",
            existence_probability=await self.calculate_existence_probability(gap),
            utility_score=0.7,
            confidence=gap.confidence * 0.8,
            suggested_trust_score=0.6,
            suggested_relationships=[
                {"target_concept": concepts[0], "relation_type": "related_to"},
                {"target_concept": concepts[1], "relation_type": "related_to"},
            ],
        )

        # Calculate more accurate utility score
        proposal.utility_score = await self.score_utility(proposal, gap)

        return proposal

    async def _generate_context_proposals(self, gap: DetectedGap) -> list[ProposedNode]:
        """Generate proposals based on gap context."""
        proposals = []

        if gap.context_area and gap.description:
            # Simple context-based proposal
            context_concept = f"context_concept_for_{gap.context_area}"

            proposal = ProposedNode(
                id=str(uuid.uuid4()),
                content=f"Context-relevant concept for {gap.context_area}",
                concept=context_concept,
                gap_id=gap.id,
                reasoning=f"Addresses gap in {gap.context_area}: {gap.description}",
                existence_probability=await self.calculate_existence_probability(gap),
                utility_score=0.6,
                confidence=gap.confidence * 0.7,
                suggested_trust_score=0.5,
            )

            proposal.utility_score = await self.score_utility(proposal, gap)
            proposals.append(proposal)

        return proposals

    async def _calculate_context_richness(self, gap: DetectedGap) -> float:
        """Calculate the richness of context around a gap."""
        context_factors = []

        # Evidence count factor
        if gap.evidence:
            context_factors.append(min(1.0, len(gap.evidence) / 5.0))

        # Context area specificity
        if gap.context_area:
            context_factors.append(0.8)
        else:
            context_factors.append(0.4)

        # Source nodes availability
        if gap.source_nodes:
            context_factors.append(min(1.0, len(gap.source_nodes) / 3.0))
        else:
            context_factors.append(0.2)

        return sum(context_factors) / len(context_factors) if context_factors else 0.5

    async def _assess_connectivity_potential(self, proposal: ProposedNode, gap: DetectedGap) -> float:
        """Assess how much the proposed node would improve connectivity."""
        # Simplified assessment - would be more sophisticated in practice
        if gap.gap_type in [GapType.ISOLATED_CLUSTER, GapType.WEAK_CONNECTION]:
            return 0.8
        elif gap.gap_type == GapType.MISSING_NODE:
            return 0.7
        return 0.5

    async def _assess_coverage_improvement(self, proposal: ProposedNode, gap: DetectedGap) -> float:
        """Assess knowledge coverage improvement from the proposal."""
        # Base on gap priority and confidence
        return (gap.priority + gap.confidence) / 2.0

    async def _assess_integration_feasibility(self, proposal: ProposedNode) -> float:
        """Assess how easily the proposed node could be integrated."""
        # Simplified feasibility assessment
        if proposal.suggested_relationships:
            return 0.8
        return 0.6

    async def _rank_proposals(self, proposals: list[ProposedNode]) -> list[ProposedNode]:
        """Rank proposals by utility and confidence."""
        proposals.sort(key=lambda p: (p.utility_score * p.confidence * p.existence_probability), reverse=True)
        return proposals

    def get_statistics(self) -> dict[str, Any]:
        """Get service statistics."""
        return {
            "nodes_proposed": self.stats["nodes_proposed"],
            "proposals_generated": self.stats["proposals_generated"],
            "avg_confidence": self.stats["avg_confidence"],
            "proposal_strategies": self.proposal_strategies,
        }
