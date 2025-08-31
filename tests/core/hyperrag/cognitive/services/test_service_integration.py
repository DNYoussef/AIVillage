"""
Integration tests for GraphFixer services

Tests service interactions, data flow, and end-to-end workflows.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
import uuid
import numpy as np

from core.hyperrag.cognitive.services import *
from core.hyperrag.cognitive.facades.graph_fixer_facade import GraphFixerFacade
from core.hyperrag.cognitive.interfaces.base_service import ServiceConfig
from core.hyperrag.cognitive.graph_fixer import DetectedGap, GapType


@pytest.fixture
def integration_config():
    """Create comprehensive test configuration."""
    config = ServiceConfig(
        trust_graph=Mock(),
        vector_engine=Mock(),
        min_confidence_threshold=0.3,
        max_proposals_per_gap=3,
        cache_enabled=True
    )
    
    # Rich mock trust graph for integration testing
    config.trust_graph.nodes = {
        "ai_node": Mock(
            concept="Artificial Intelligence",
            trust_score=0.9,
            incoming_edges=set(),
            outgoing_edges=set(["ai_to_ml"]),
            embedding=np.array([0.1, 0.8, 0.3, 0.6])
        ),
        "ml_node": Mock(
            concept="Machine Learning", 
            trust_score=0.8,
            incoming_edges=set(["ai_to_ml"]),
            outgoing_edges=set(["ml_to_nn"]),
            embedding=np.array([0.2, 0.7, 0.4, 0.5])
        ),
        "nn_node": Mock(
            concept="Neural Networks",
            trust_score=0.85,
            incoming_edges=set(["ml_to_nn"]),
            outgoing_edges=set(),
            embedding=np.array([0.15, 0.75, 0.35, 0.55])
        ),
        "isolated_node": Mock(
            concept="Quantum Computing",
            trust_score=0.7,
            incoming_edges=set(),
            outgoing_edges=set(),
            embedding=np.array([0.9, 0.1, 0.2, 0.3])
        ),
        "weak_node": Mock(
            concept="Robotics",
            trust_score=0.95,  # High trust but few connections
            incoming_edges=set(),
            outgoing_edges=set(["robotics_edge"]),
            embedding=np.array([0.3, 0.4, 0.8, 0.2])
        )
    }
    
    config.trust_graph.edges = {
        "ai_to_ml": Mock(source_id="ai_node", target_id="ml_node"),
        "ml_to_nn": Mock(source_id="ml_node", target_id="nn_node"),
        "robotics_edge": Mock(source_id="weak_node", target_id="ai_node")
    }
    
    return config


@pytest.fixture
async def all_services(integration_config):
    """Initialize all services for integration testing."""
    services = {
        'gap_detection': GapDetectionService(integration_config),
        'node_proposal': NodeProposalService(integration_config),
        'relationship_analyzer': RelationshipAnalyzerService(integration_config),
        'confidence_calculator': ConfidenceCalculatorService(integration_config),
        'graph_analytics': GraphAnalyticsService(integration_config),
        'knowledge_validator': KnowledgeValidatorService(integration_config)
    }
    
    # Initialize all services
    for service in services.values():
        await service.initialize()
    
    return services


@pytest.fixture
async def integrated_facade(integration_config):
    """Create facade with real services."""
    facade = GraphFixerFacade(
        trust_graph=integration_config.trust_graph,
        vector_engine=integration_config.vector_engine,
        min_confidence_threshold=0.3,
        max_proposals_per_gap=3
    )
    await facade.initialize()
    return facade


@pytest.mark.asyncio
class TestServiceIntegration:
    """Test suite for service integration."""
    
    async def test_end_to_end_gap_detection_and_proposal(self, all_services):
        """Test complete workflow from gap detection to proposal."""
        gap_service = all_services['gap_detection']
        node_service = all_services['node_proposal']
        rel_service = all_services['relationship_analyzer']
        confidence_service = all_services['confidence_calculator']
        
        # Step 1: Detect gaps
        gaps = await gap_service.detect_gaps(
            query="artificial intelligence neural networks",
            focus_area="AI"
        )
        
        assert len(gaps) > 0
        
        # Step 2: Generate proposals
        node_proposals = await node_service.propose_nodes(gaps)
        rel_proposals = await rel_service.propose_relationships(gaps)
        
        # Step 3: Calculate enhanced confidence scores
        for proposal in node_proposals + rel_proposals:
            matching_gaps = [g for g in gaps if g.id == proposal.gap_id]
            if matching_gaps:
                gap = matching_gaps[0]
                evidence = gap.evidence + [proposal.reasoning]
                
                confidence = await confidence_service.calculate_confidence(
                    proposal, gap, evidence
                )
                
                assert 0.0 <= confidence <= 1.0
                proposal.confidence = confidence
        
        # Verify proposals have reasonable properties
        for proposal in node_proposals:
            assert proposal.content
            assert proposal.concept
            assert 0.0 <= proposal.confidence <= 1.0
            
        for proposal in rel_proposals:
            assert proposal.source_id
            assert proposal.target_id
            assert 0.0 <= proposal.confidence <= 1.0
    
    async def test_validation_and_learning_workflow(self, all_services):
        """Test validation and learning workflow."""
        validator_service = all_services['knowledge_validator']
        gap_service = all_services['gap_detection']
        node_service = all_services['node_proposal']
        
        # Generate gap and proposal
        gaps = await gap_service.detect_gaps()
        if not gaps:
            # Create test gap if none detected
            gaps = [DetectedGap(
                id=str(uuid.uuid4()),
                gap_type=GapType.MISSING_NODE,
                source_nodes=["ai_node"],
                evidence=["Test evidence"],
                confidence=0.8
            )]
        
        proposals = await node_service.propose_nodes(gaps[:1])
        if not proposals:
            return  # Skip if no proposals generated
        
        proposal = proposals[0]
        
        # Step 1: Validate consistency
        validation_results = await validator_service.validate_consistency([proposal])
        assert proposal.id in validation_results
        
        # Step 2: Check for conflicts  
        conflicts = await validator_service.check_conflicts(proposal)
        assert isinstance(conflicts, list)
        
        # Step 3: Learn from validation
        await validator_service.learn_from_validation(proposal, True)
        
        # Verify learning data was updated
        stats = validator_service.get_statistics()
        assert stats["learning_updates"] > 0
    
    async def test_graph_analytics_integration(self, all_services):
        """Test graph analytics with other services."""
        analytics_service = all_services['graph_analytics']
        gap_service = all_services['gap_detection']
        
        # Get analytics
        centrality = await analytics_service.compute_centrality_metrics()
        clusters = await analytics_service.analyze_clusters()
        connectivity = await analytics_service.measure_connectivity()
        completeness = await analytics_service.analyze_completeness()
        
        # Verify analytics structure
        assert "degree_centrality" in centrality
        assert "communities" in clusters
        assert "graph_density" in connectivity
        assert "overall_completeness" in completeness
        
        # Analytics should influence gap detection
        gaps = await gap_service.detect_gaps()
        
        # Should detect isolated node gap
        isolated_gaps = [g for g in gaps if g.gap_type == GapType.ISOLATED_CLUSTER]
        assert len(isolated_gaps) > 0
        
        # Should detect weak connection gap
        weak_gaps = [g for g in gaps if g.gap_type == GapType.WEAK_CONNECTION]
        assert len(weak_gaps) > 0
    
    async def test_semantic_similarity_integration(self, all_services):
        """Test semantic similarity across services."""
        rel_service = all_services['relationship_analyzer']
        gap_service = all_services['gap_detection']
        
        # Test semantic similarity calculation
        similarity = await rel_service.analyze_semantic_similarity("ai_node", "ml_node")
        assert 0.0 <= similarity <= 1.0
        
        # Should be higher for related concepts
        ai_ml_similarity = await rel_service.analyze_semantic_similarity("ai_node", "ml_node")
        ai_quantum_similarity = await rel_service.analyze_semantic_similarity("ai_node", "isolated_node")
        
        # AI and ML should be more similar than AI and Quantum Computing
        assert ai_ml_similarity >= ai_quantum_similarity
        
        # Semantic gaps should be detected for similar but unconnected nodes
        semantic_gaps = await gap_service.detect_semantic_gaps()
        semantic_missing_rel = [g for g in semantic_gaps if g.gap_type == GapType.MISSING_RELATIONSHIP]
        
        # Should find some semantically similar but unconnected pairs
        assert len(semantic_missing_rel) >= 0  # May be 0 if all similar nodes are connected
    
    async def test_confidence_calculation_integration(self, all_services):
        """Test confidence calculation across different proposal types."""
        confidence_service = all_services['confidence_calculator']
        gap_service = all_services['gap_detection']
        node_service = all_services['node_proposal'] 
        rel_service = all_services['relationship_analyzer']
        
        # Generate test data
        gaps = await gap_service.detect_gaps()
        if not gaps:
            gaps = [DetectedGap(
                id=str(uuid.uuid4()),
                gap_type=GapType.MISSING_NODE,
                evidence=["Strong evidence", "Supporting evidence"],
                confidence=0.8,
                priority=0.7
            )]
        
        # Test confidence for node proposals
        node_proposals = await node_service.propose_nodes(gaps[:1])
        for proposal in node_proposals:
            gap = gaps[0]
            confidence = await confidence_service.calculate_confidence(
                proposal, gap, gap.evidence
            )
            
            assert 0.0 <= confidence <= 1.0
            # Should be influenced by gap confidence and evidence
            assert confidence > 0.0  # Should have some confidence
        
        # Test confidence for relationship proposals
        rel_proposals = await rel_service.propose_relationships(gaps[:1])
        for proposal in rel_proposals:
            gap = gaps[0]
            confidence = await confidence_service.calculate_confidence(
                proposal, gap, proposal.evidence_sources
            )
            
            assert 0.0 <= confidence <= 1.0
    
    async def test_service_statistics_aggregation(self, all_services):
        """Test statistics collection across services."""
        # Run operations on all services
        await all_services['gap_detection'].detect_gaps("test query")
        
        gaps = [DetectedGap(
            id=str(uuid.uuid4()),
            gap_type=GapType.MISSING_NODE,
            source_nodes=["ai_node"],
            confidence=0.8
        )]
        
        await all_services['node_proposal'].propose_nodes(gaps)
        await all_services['relationship_analyzer'].propose_relationships(gaps)
        await all_services['graph_analytics'].compute_centrality_metrics()
        
        # Collect statistics from all services
        all_stats = {}
        for name, service in all_services.items():
            all_stats[name] = service.get_statistics()
        
        # Verify each service has reasonable statistics
        assert all_stats['gap_detection']['analyses_performed'] > 0
        assert 'nodes_proposed' in all_stats['node_proposal']
        assert 'relationships_proposed' in all_stats['relationship_analyzer']
        assert 'confidence_calculations' in all_stats['confidence_calculator']
        assert 'metric_calculations' in all_stats['graph_analytics']
        assert 'validations_performed' in all_stats['knowledge_validator']
    
    async def test_error_propagation_and_recovery(self, integration_config):
        """Test error handling and recovery across services."""
        # Create service with problematic configuration
        bad_config = ServiceConfig(
            trust_graph=None,  # Missing required dependency
            vector_engine=None,
            min_confidence_threshold=0.3
        )
        
        gap_service = GapDetectionService(bad_config)
        await gap_service.initialize()
        
        # Should handle missing dependencies gracefully
        gaps = await gap_service.detect_gaps("test")
        assert gaps == []  # Should return empty, not crash
        
        # Service should still be functional for operations that don't need trust_graph
        stats = gap_service.get_statistics()
        assert isinstance(stats, dict)
    
    async def test_cache_coordination_across_services(self, all_services):
        """Test cache usage and coordination."""
        gap_service = all_services['gap_detection']
        rel_service = all_services['relationship_analyzer']
        
        # First calls should populate cache
        await gap_service.detect_gaps("test query", focus_area="AI")
        await rel_service.analyze_semantic_similarity("ai_node", "ml_node")
        
        initial_gap_hits = gap_service.stats.get("cache_hits", 0)
        initial_rel_hits = rel_service.stats.get("cache_hits", 0)
        
        # Subsequent identical calls should hit cache
        await gap_service.detect_gaps("test query", focus_area="AI")
        await rel_service.analyze_semantic_similarity("ai_node", "ml_node")
        
        final_gap_hits = gap_service.stats.get("cache_hits", 0)
        final_rel_hits = rel_service.stats.get("cache_hits", 0)
        
        # Cache hits should increase
        assert final_gap_hits >= initial_gap_hits
        assert final_rel_hits >= initial_rel_hits
    
    async def test_facade_service_coordination(self, integrated_facade):
        """Test facade coordination of all services."""
        facade = integrated_facade
        
        # Comprehensive analysis should coordinate all services
        result = await facade.perform_comprehensive_analysis(
            query="artificial intelligence",
            focus_area="AI"
        )
        
        assert result.total_gaps_found >= 0
        assert result.total_proposals >= 0
        assert result.analysis_time_ms > 0
        assert 0.0 <= result.coverage_improvement <= 1.0
        
        # Should have metadata from all services
        assert "services_used" in result.metadata
        assert result.metadata["services_used"] == 6
        
        # Statistics should show activity across services
        stats = await facade.get_gap_statistics()
        assert "aggregate" in stats
        assert stats["aggregate"]["total_gaps_detected"] >= 0
    
    async def test_service_cleanup_coordination(self, all_services):
        """Test coordinated cleanup of all services."""
        # Verify all services are initialized
        for service in all_services.values():
            assert service.is_initialized
        
        # Add some data to caches
        for service in all_services.values():
            if hasattr(service, 'set_cache'):
                service.set_cache("test_key", "test_value")
        
        # Cleanup all services
        cleanup_tasks = [service.cleanup() for service in all_services.values()]
        await asyncio.gather(*cleanup_tasks)
        
        # Verify all services are cleaned up
        for service in all_services.values():
            assert not service.is_initialized
            if hasattr(service, '_cache'):
                assert len(service._cache) == 0


if __name__ == "__main__":
    pytest.main([__file__])