"""
Unit tests for GraphFixerFacade

Tests facade coordination, backward compatibility, and service integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import uuid

from core.hyperrag.cognitive.facades.graph_fixer_facade import GraphFixerFacade
from core.hyperrag.cognitive.graph_fixer import DetectedGap, ProposedNode, ProposedRelationship, GapType


@pytest.fixture
def mock_trust_graph():
    """Create mock trust graph."""
    trust_graph = Mock()
    trust_graph.nodes = {
        "node1": Mock(
            concept="AI",
            trust_score=0.8,
            incoming_edges=set(),
            outgoing_edges=set(["edge1"])
        ),
        "node2": Mock(
            concept="ML",
            trust_score=0.9,
            incoming_edges=set(["edge1"]),
            outgoing_edges=set()
        )
    }
    trust_graph.edges = {
        "edge1": Mock(source_id="node1", target_id="node2")
    }
    return trust_graph


@pytest.fixture
def mock_vector_engine():
    """Create mock vector engine."""
    return Mock()


@pytest.fixture
async def graph_fixer_facade(mock_trust_graph, mock_vector_engine):
    """Create and initialize GraphFixer facade."""
    facade = GraphFixerFacade(
        trust_graph=mock_trust_graph,
        vector_engine=mock_vector_engine,
        min_confidence_threshold=0.3,
        max_proposals_per_gap=3
    )
    await facade.initialize()
    return facade


@pytest.mark.asyncio
class TestGraphFixerFacade:
    """Test suite for GraphFixerFacade."""
    
    async def test_initialization(self, mock_trust_graph, mock_vector_engine):
        """Test facade initialization."""
        facade = GraphFixerFacade(
            trust_graph=mock_trust_graph,
            vector_engine=mock_vector_engine
        )
        
        assert not facade.initialized
        
        await facade.initialize()
        
        assert facade.initialized
        # All services should be initialized
        assert facade.gap_detection.is_initialized
        assert facade.node_proposal.is_initialized
        assert facade.relationship_analyzer.is_initialized
        assert facade.confidence_calculator.is_initialized
        assert facade.graph_analytics.is_initialized
        assert facade.knowledge_validator.is_initialized
    
    async def test_backward_compatibility_attributes(self, graph_fixer_facade):
        """Test that original GraphFixer attributes are maintained."""
        facade = graph_fixer_facade
        
        # Original attributes should be accessible
        assert facade.trust_graph is not None
        assert facade.vector_engine is not None
        assert facade.min_confidence_threshold == 0.3
        assert facade.max_proposals_per_gap == 3
    
    async def test_detect_knowledge_gaps(self, graph_fixer_facade):
        """Test knowledge gap detection through facade."""
        with patch.object(graph_fixer_facade.gap_detection, 'detect_gaps') as mock_detect:
            mock_gaps = [
                DetectedGap(
                    id=str(uuid.uuid4()),
                    gap_type=GapType.MISSING_NODE,
                    confidence=0.8,
                    priority=0.7
                )
            ]
            mock_detect.return_value = mock_gaps
            
            gaps = await graph_fixer_facade.detect_knowledge_gaps(
                query="test query",
                retrieved_info=[],
                focus_area="AI"
            )
            
            assert len(gaps) == 1
            assert gaps[0].gap_type == GapType.MISSING_NODE
            
            # Verify service was called correctly
            mock_detect.assert_called_once_with("test query", [], "AI")
    
    async def test_propose_solutions(self, graph_fixer_facade):
        """Test solution proposal through facade."""
        gaps = [
            DetectedGap(
                id=str(uuid.uuid4()),
                gap_type=GapType.MISSING_NODE,
                confidence=0.8
            )
        ]
        
        # Mock service responses
        mock_nodes = [ProposedNode(gap_id=gaps[0].id, confidence=0.7)]
        mock_relationships = [ProposedRelationship(gap_id=gaps[0].id, confidence=0.6)]
        
        with patch.object(graph_fixer_facade.node_proposal, 'propose_nodes') as mock_propose_nodes, \
             patch.object(graph_fixer_facade.relationship_analyzer, 'propose_relationships') as mock_propose_rels:
            
            mock_propose_nodes.return_value = mock_nodes
            mock_propose_rels.return_value = mock_relationships
            
            nodes, relationships = await graph_fixer_facade.propose_solutions(gaps, max_proposals=5)
            
            assert len(nodes) == 1
            assert len(relationships) == 1
            
            # Verify services were called
            mock_propose_nodes.assert_called_once()
            mock_propose_rels.assert_called_once()
    
    async def test_validate_proposal(self, graph_fixer_facade):
        """Test proposal validation through facade."""
        proposal = ProposedNode(
            id=str(uuid.uuid4()),
            content="Test proposal",
            confidence=0.8
        )
        
        with patch.object(graph_fixer_facade.knowledge_validator, 'learn_from_validation') as mock_learn:
            mock_learn.return_value = None
            
            success = await graph_fixer_facade.validate_proposal(
                proposal, "Good proposal", True
            )
            
            assert success
            assert proposal.validation_status == "validated"
            assert proposal.validation_feedback == "Good proposal"
            
            # Verify learning was triggered
            mock_learn.assert_called_once_with(proposal, True)
    
    async def test_analyze_graph_completeness(self, graph_fixer_facade):
        """Test graph completeness analysis through facade."""
        mock_analysis = {
            "overall_completeness": 0.75,
            "structural_completeness": {"completeness_score": 0.8},
            "semantic_completeness": {"completeness_score": 0.7}
        }
        
        with patch.object(graph_fixer_facade.graph_analytics, 'analyze_completeness') as mock_analyze:
            mock_analyze.return_value = mock_analysis
            
            result = await graph_fixer_facade.analyze_graph_completeness()
            
            assert result["overall_completeness"] == 0.75
            mock_analyze.assert_called_once()
    
    async def test_get_gap_statistics(self, graph_fixer_facade):
        """Test gap statistics aggregation."""
        # Mock individual service statistics
        with patch.object(graph_fixer_facade.gap_detection, 'get_statistics') as mock_gap_stats, \
             patch.object(graph_fixer_facade.node_proposal, 'get_statistics') as mock_node_stats:
            
            mock_gap_stats.return_value = {"gaps_detected": 10}
            mock_node_stats.return_value = {"nodes_proposed": 5}
            
            stats = await graph_fixer_facade.get_gap_statistics()
            
            assert "facade" in stats
            assert "gap_detection" in stats
            assert "node_proposal" in stats
            assert "aggregate" in stats
            
            # Check aggregate calculations
            assert stats["aggregate"]["total_gaps_detected"] == 10
    
    async def test_perform_comprehensive_analysis(self, graph_fixer_facade):
        """Test comprehensive analysis workflow."""
        # Mock service responses
        mock_gaps = [DetectedGap(id=str(uuid.uuid4()), confidence=0.8)]
        mock_nodes = [ProposedNode(confidence=0.7)]
        mock_relationships = [ProposedRelationship(confidence=0.6)]
        mock_analytics = {"overall_completeness": 0.8}
        
        with patch.object(graph_fixer_facade, 'detect_knowledge_gaps') as mock_detect, \
             patch.object(graph_fixer_facade, 'propose_solutions') as mock_propose, \
             patch.object(graph_fixer_facade.graph_analytics, 'analyze_completeness') as mock_analyze:
            
            mock_detect.return_value = mock_gaps
            mock_propose.return_value = (mock_nodes, mock_relationships)
            mock_analyze.return_value = mock_analytics
            
            result = await graph_fixer_facade.perform_comprehensive_analysis(
                query="test", focus_area="AI"
            )
            
            assert len(result.gaps_detected) == 1
            assert len(result.proposed_nodes) == 1
            assert len(result.proposed_relationships) == 1
            assert result.total_gaps_found == 1
            assert result.total_proposals == 2
            assert result.coverage_improvement == 0.8
            assert result.analysis_time_ms > 0
    
    async def test_enhance_proposals_with_confidence(self, graph_fixer_facade):
        """Test proposal confidence enhancement."""
        gap = DetectedGap(
            id=str(uuid.uuid4()),
            evidence=["Strong evidence"]
        )
        
        proposal = ProposedNode(
            gap_id=gap.id,
            reasoning="Test reasoning",
            confidence=0.5
        )
        
        with patch.object(graph_fixer_facade.confidence_calculator, 'calculate_confidence') as mock_calc:
            mock_calc.return_value = 0.8
            
            await graph_fixer_facade._enhance_proposals_with_confidence([proposal], [], [gap])
            
            assert proposal.confidence == 0.8
            mock_calc.assert_called_once()
    
    async def test_validate_proposals(self, graph_fixer_facade):
        """Test proposal validation coordination."""
        proposals = [
            ProposedNode(id=str(uuid.uuid4()), confidence=0.8),
            ProposedRelationship(id=str(uuid.uuid4()), confidence=0.7)
        ]
        
        with patch.object(graph_fixer_facade.knowledge_validator, 'validate_consistency') as mock_validate, \
             patch.object(graph_fixer_facade.knowledge_validator, 'check_conflicts') as mock_conflicts:
            
            mock_validate.return_value = {proposals[0].id: True, proposals[1].id: False}
            mock_conflicts.return_value = ["Test conflict"]
            
            await graph_fixer_facade._validate_proposals(proposals)
            
            # Valid proposal should maintain confidence
            assert proposals[0].confidence == 0.8
            
            # Invalid proposal should have reduced confidence
            assert proposals[1].confidence < 0.7
            assert "Test conflict" in proposals[1].validation_feedback
    
    async def test_error_handling_in_detect_gaps(self, graph_fixer_facade):
        """Test error handling in gap detection."""
        with patch.object(graph_fixer_facade.gap_detection, 'detect_gaps') as mock_detect:
            mock_detect.side_effect = Exception("Service error")
            
            gaps = await graph_fixer_facade.detect_knowledge_gaps("test")
            
            assert gaps == []  # Should return empty list on error
            assert graph_fixer_facade.stats["service_errors"] > 0
    
    async def test_error_handling_in_propose_solutions(self, graph_fixer_facade):
        """Test error handling in solution proposal."""
        gap = DetectedGap(id=str(uuid.uuid4()))
        
        with patch.object(graph_fixer_facade.node_proposal, 'propose_nodes') as mock_propose:
            mock_propose.side_effect = Exception("Proposal error")
            
            nodes, relationships = await graph_fixer_facade.propose_solutions([gap])
            
            assert nodes == []
            assert relationships == []
            assert graph_fixer_facade.stats["service_errors"] > 0
    
    async def test_ensure_initialized(self, mock_trust_graph, mock_vector_engine):
        """Test automatic initialization."""
        facade = GraphFixerFacade(
            trust_graph=mock_trust_graph,
            vector_engine=mock_vector_engine
        )
        
        assert not facade.initialized
        
        # Should auto-initialize when calling methods
        await facade._ensure_initialized()
        
        assert facade.initialized
    
    async def test_cleanup(self, graph_fixer_facade):
        """Test facade cleanup."""
        assert graph_fixer_facade.initialized
        
        with patch.object(graph_fixer_facade.gap_detection, 'cleanup') as mock_cleanup:
            mock_cleanup.return_value = None
            
            await graph_fixer_facade.cleanup()
            
            assert not graph_fixer_facade.initialized
    
    async def test_async_context_manager(self, mock_trust_graph, mock_vector_engine):
        """Test async context manager functionality."""
        async with GraphFixerFacade(
            trust_graph=mock_trust_graph,
            vector_engine=mock_vector_engine
        ) as facade:
            assert facade.initialized
            
            # Should be able to use facade normally
            gaps = await facade.detect_knowledge_gaps("test")
            assert isinstance(gaps, list)
        
        # Should be cleaned up after context exit
        assert not facade.initialized
    
    async def test_statistics_tracking(self, graph_fixer_facade):
        """Test facade statistics tracking."""
        initial_calls = graph_fixer_facade.stats["facade_calls"]
        
        # Make some facade calls
        await graph_fixer_facade.detect_knowledge_gaps("test")
        await graph_fixer_facade.analyze_graph_completeness()
        
        final_calls = graph_fixer_facade.stats["facade_calls"]
        
        assert final_calls > initial_calls
    
    @pytest.mark.parametrize("query,focus_area", [
        ("machine learning", "AI"),
        ("neural networks", "ML"),
        (None, None),
        ("", "")
    ])
    async def test_detect_gaps_with_various_inputs(self, graph_fixer_facade, query, focus_area):
        """Test gap detection with various input combinations."""
        with patch.object(graph_fixer_facade.gap_detection, 'detect_gaps') as mock_detect:
            mock_detect.return_value = []
            
            gaps = await graph_fixer_facade.detect_knowledge_gaps(
                query=query, focus_area=focus_area
            )
            
            assert isinstance(gaps, list)
            mock_detect.assert_called_once_with(query, None, focus_area)


if __name__ == "__main__":
    pytest.main([__file__])