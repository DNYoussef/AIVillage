"""
Unit tests for NodeProposalService

Tests node proposal generation, utility scoring, and existence probability calculation.
"""

import pytest
from unittest.mock import Mock
import uuid

from core.hyperrag.cognitive.services.node_proposal_service import NodeProposalService
from core.hyperrag.cognitive.interfaces.base_service import ServiceConfig
from core.hyperrag.cognitive.graph_fixer import DetectedGap, ProposedNode, GapType


@pytest.fixture
def service_config():
    """Create test service configuration."""
    config = ServiceConfig(
        trust_graph=Mock(),
        vector_engine=Mock(),
        min_confidence_threshold=0.3,
        max_proposals_per_gap=3,
        cache_enabled=True,
    )

    # Mock trust graph with nodes and edges
    config.trust_graph.nodes = {
        "node1": Mock(
            concept="Machine Learning", trust_score=0.8, incoming_edges=set(["edge1"]), outgoing_edges=set(["edge2"])
        ),
        "node2": Mock(
            concept="Neural Networks", trust_score=0.9, incoming_edges=set(["edge2"]), outgoing_edges=set(["edge3"])
        ),
        "node3": Mock(concept="Deep Learning", trust_score=0.7, incoming_edges=set(["edge3"]), outgoing_edges=set()),
    }

    config.trust_graph.edges = {
        "edge1": Mock(source_id="node0", target_id="node1"),
        "edge2": Mock(source_id="node1", target_id="node2"),
        "edge3": Mock(source_id="node2", target_id="node3"),
    }

    return config


@pytest.fixture
async def node_proposal_service(service_config):
    """Create and initialize node proposal service."""
    service = NodeProposalService(service_config)
    await service.initialize()
    return service


@pytest.mark.asyncio
class TestNodeProposalService:
    """Test suite for NodeProposalService."""

    async def test_initialization(self, service_config):
        """Test service initialization."""
        service = NodeProposalService(service_config)

        assert not service.is_initialized
        success = await service.initialize()

        assert success
        assert service.is_initialized
        assert len(service.proposal_strategies) > 0

    async def test_propose_nodes_missing_node_gap(self, node_proposal_service):
        """Test node proposal for missing node gaps."""
        gap = DetectedGap(
            id=str(uuid.uuid4()),
            gap_type=GapType.MISSING_NODE,
            source_nodes=["node1", "node2"],
            description="Missing bridging concept",
            evidence=["High semantic similarity", "No direct connection"],
            confidence=0.8,
            priority=0.7,
        )

        proposals = await node_proposal_service.propose_nodes([gap])

        assert len(proposals) > 0

        # Verify proposal properties
        for proposal in proposals:
            assert isinstance(proposal, ProposedNode)
            assert proposal.gap_id == gap.id
            assert 0.0 <= proposal.existence_probability <= 1.0
            assert 0.0 <= proposal.utility_score <= 1.0
            assert 0.0 <= proposal.confidence <= 1.0
            assert proposal.content
            assert proposal.concept

    async def test_propose_nodes_isolated_cluster_gap(self, node_proposal_service):
        """Test node proposal for isolated cluster gaps."""
        gap = DetectedGap(
            id=str(uuid.uuid4()),
            gap_type=GapType.ISOLATED_CLUSTER,
            source_nodes=["node3"],
            description="Isolated node needs connections",
            evidence=["Node has no connections"],
            confidence=0.9,
            priority=0.8,
        )

        proposals = await node_proposal_service.propose_nodes([gap])

        assert len(proposals) > 0

        # Should propose bridging nodes
        for proposal in proposals:
            assert "bridge" in proposal.concept.lower() or "bridge" in proposal.content.lower()
            assert proposal.gap_id == gap.id

    async def test_propose_nodes_incomplete_path_gap(self, node_proposal_service):
        """Test node proposal for incomplete path gaps."""
        gap = DetectedGap(
            id=str(uuid.uuid4()),
            gap_type=GapType.INCOMPLETE_PATH,
            source_nodes=["node1", "node3"],
            description="Missing intermediate concept",
            evidence=["Path exists but is incomplete"],
            confidence=0.7,
            priority=0.6,
        )

        proposals = await node_proposal_service.propose_nodes([gap])

        assert len(proposals) > 0

        # Should propose intermediate nodes
        for proposal in proposals:
            assert "intermediate" in proposal.concept.lower() or "intermediate" in proposal.content.lower()
            assert proposal.gap_id == gap.id

    async def test_calculate_existence_probability(self, node_proposal_service):
        """Test existence probability calculation."""
        gap = DetectedGap(
            gap_type=GapType.MISSING_NODE,
            confidence=0.8,
            evidence=["Strong evidence 1", "Strong evidence 2", "Strong evidence 3"],
            source_nodes=["node1", "node2"],
        )

        probability = await node_proposal_service.calculate_existence_probability(gap)

        assert 0.1 <= probability <= 1.0
        # Should be higher for gaps with more evidence and higher confidence
        assert probability > 0.5  # Strong evidence should yield high probability

    async def test_score_utility(self, node_proposal_service):
        """Test utility scoring."""
        gap = DetectedGap(priority=0.8, confidence=0.7, gap_type=GapType.MISSING_NODE)

        proposal = ProposedNode(
            content="Test bridging concept",
            concept="test_concept",
            gap_id=gap.id,
            existence_probability=0.7,
            utility_score=0.0,  # Will be calculated
            confidence=0.6,
        )

        utility = await node_proposal_service.score_utility(proposal, gap)

        assert 0.1 <= utility <= 1.0
        # Higher priority gaps should yield higher utility scores

    async def test_get_neighboring_concepts(self, node_proposal_service):
        """Test neighboring concept extraction."""
        node_ids = ["node1", "node2"]

        concepts = await node_proposal_service._get_neighboring_concepts(node_ids)

        assert isinstance(concepts, set)
        # Should find concepts from connected nodes

    async def test_create_bridging_proposal(self, node_proposal_service):
        """Test bridging proposal creation."""
        gap = DetectedGap(id=str(uuid.uuid4()), gap_type=GapType.MISSING_NODE, confidence=0.8, priority=0.7)

        concepts = ["Machine Learning", "Neural Networks"]

        proposal = await node_proposal_service._create_bridging_proposal(gap, concepts)

        assert isinstance(proposal, ProposedNode)
        assert proposal.gap_id == gap.id
        assert "bridge" in proposal.concept.lower()
        assert len(proposal.suggested_relationships) > 0

        # Should reference both concepts
        content_lower = proposal.content.lower()
        assert "machine learning" in content_lower
        assert "neural networks" in content_lower

    async def test_generate_context_proposals(self, node_proposal_service):
        """Test context-based proposal generation."""
        gap = DetectedGap(
            id=str(uuid.uuid4()),
            context_area="artificial intelligence",
            description="Missing AI concept",
            confidence=0.6,
        )

        proposals = await node_proposal_service._generate_context_proposals(gap)

        if proposals:  # May be empty if context is insufficient
            for proposal in proposals:
                assert isinstance(proposal, ProposedNode)
                assert proposal.gap_id == gap.id
                assert "artificial intelligence" in proposal.content.lower()

    async def test_calculate_context_richness(self, node_proposal_service):
        """Test context richness calculation."""
        # Rich context gap
        rich_gap = DetectedGap(
            evidence=["Evidence 1", "Evidence 2", "Evidence 3", "Evidence 4"],
            context_area="machine learning",
            source_nodes=["node1", "node2", "node3"],
        )

        rich_score = await node_proposal_service._calculate_context_richness(rich_gap)

        # Poor context gap
        poor_gap = DetectedGap(evidence=[], context_area="", source_nodes=[])

        poor_score = await node_proposal_service._calculate_context_richness(poor_gap)

        assert rich_score > poor_score
        assert 0.0 <= rich_score <= 1.0
        assert 0.0 <= poor_score <= 1.0

    async def test_assess_connectivity_potential(self, node_proposal_service):
        """Test connectivity potential assessment."""
        proposal = ProposedNode()

        # High connectivity gap types
        high_connectivity_gap = DetectedGap(gap_type=GapType.ISOLATED_CLUSTER)
        high_score = await node_proposal_service._assess_connectivity_potential(proposal, high_connectivity_gap)

        # Lower connectivity gap types
        low_connectivity_gap = DetectedGap(gap_type=GapType.CONFLICTING_INFO)
        low_score = await node_proposal_service._assess_connectivity_potential(proposal, low_connectivity_gap)

        assert high_score >= low_score
        assert 0.0 <= high_score <= 1.0
        assert 0.0 <= low_score <= 1.0

    async def test_rank_proposals(self, node_proposal_service):
        """Test proposal ranking."""
        proposals = [
            ProposedNode(utility_score=0.5, confidence=0.6, existence_probability=0.7),
            ProposedNode(utility_score=0.8, confidence=0.9, existence_probability=0.8),
            ProposedNode(utility_score=0.3, confidence=0.4, existence_probability=0.5),
        ]

        ranked = await node_proposal_service._rank_proposals(proposals)

        # Should be ranked by utility * confidence * existence_probability
        scores = [p.utility_score * p.confidence * p.existence_probability for p in ranked]

        assert scores == sorted(scores, reverse=True)

    async def test_max_proposals_per_gap_limit(self, node_proposal_service):
        """Test max proposals per gap limit."""
        # Set low limit
        node_proposal_service.config.max_proposals_per_gap = 2

        gap = DetectedGap(
            id=str(uuid.uuid4()),
            gap_type=GapType.MISSING_NODE,
            source_nodes=["node1", "node2", "node3", "node4"],  # Many sources
            evidence=["Evidence 1", "Evidence 2", "Evidence 3"],
        )

        proposals = await node_proposal_service.propose_nodes([gap])

        # Should respect the limit
        assert len(proposals) <= 2

    async def test_statistics_tracking(self, node_proposal_service):
        """Test statistics tracking."""
        initial_stats = node_proposal_service.get_statistics()

        # Generate some proposals
        gap = DetectedGap(id=str(uuid.uuid4()), gap_type=GapType.MISSING_NODE, source_nodes=["node1"], confidence=0.8)

        await node_proposal_service.propose_nodes([gap])

        final_stats = node_proposal_service.get_statistics()

        # Verify statistics updated
        assert final_stats["proposals_generated"] > initial_stats["proposals_generated"]
        assert final_stats["nodes_proposed"] >= initial_stats["nodes_proposed"]

    async def test_error_handling(self, service_config):
        """Test error handling in proposal generation."""
        # Create service with missing dependencies
        service_config.trust_graph = None
        service = NodeProposalService(service_config)
        await service.initialize()

        gap = DetectedGap(gap_type=GapType.MISSING_NODE, source_nodes=["nonexistent"])

        # Should handle missing trust graph gracefully
        proposals = await service.propose_nodes([gap])
        assert proposals == []  # Should return empty list, not crash

    async def test_cleanup(self, node_proposal_service):
        """Test service cleanup."""
        # Add some cache entries
        node_proposal_service.set_cache("test_key", "test_value")

        assert node_proposal_service.is_initialized
        assert len(node_proposal_service._cache) > 0

        await node_proposal_service.cleanup()

        assert not node_proposal_service.is_initialized
        assert len(node_proposal_service._cache) == 0


if __name__ == "__main__":
    pytest.main([__file__])
