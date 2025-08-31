"""
Unit tests for GapDetectionService

Tests gap detection algorithms, caching, and statistical analysis.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from core.hyperrag.cognitive.services.gap_detection_service import GapDetectionService
from core.hyperrag.cognitive.interfaces.base_service import ServiceConfig
from core.hyperrag.cognitive.graph_fixer import DetectedGap, GapType


@pytest.fixture
def service_config():
    """Create test service configuration."""
    config = ServiceConfig(
        trust_graph=Mock(),
        vector_engine=Mock(),
        min_confidence_threshold=0.3,
        max_proposals_per_gap=3,
        cache_enabled=True
    )
    
    # Mock trust graph structure
    config.trust_graph.nodes = {
        "node1": Mock(
            concept="AI", 
            trust_score=0.8,
            incoming_edges=set(["edge1"]),
            outgoing_edges=set(["edge2"]),
            embedding=np.array([0.1, 0.2, 0.3])
        ),
        "node2": Mock(
            concept="ML",
            trust_score=0.6, 
            incoming_edges=set(),
            outgoing_edges=set(),
            embedding=np.array([0.2, 0.3, 0.4])
        ),
        "node3": Mock(
            concept="Neural",
            trust_score=0.9,
            incoming_edges=set(["edge2"]), 
            outgoing_edges=set(["edge3"]),
            embedding=np.array([0.15, 0.25, 0.35])
        )
    }
    
    config.trust_graph.edges = {
        "edge1": Mock(source_id="node3", target_id="node1"),
        "edge2": Mock(source_id="node1", target_id="node3"),
        "edge3": Mock(source_id="node3", target_id="node2")
    }
    
    return config


@pytest.fixture
async def gap_detection_service(service_config):
    """Create and initialize gap detection service."""
    service = GapDetectionService(service_config)
    await service.initialize()
    return service


@pytest.mark.asyncio
class TestGapDetectionService:
    """Test suite for GapDetectionService."""
    
    async def test_initialization(self, service_config):
        """Test service initialization."""
        service = GapDetectionService(service_config)
        
        assert not service.is_initialized
        success = await service.initialize()
        
        assert success
        assert service.is_initialized
        assert len(service.detection_methods) > 0
    
    async def test_detect_structural_gaps(self, gap_detection_service):
        """Test structural gap detection."""
        gaps = await gap_detection_service.detect_structural_gaps()
        
        # Should detect isolated node (node2)
        isolated_gaps = [g for g in gaps if g.gap_type == GapType.ISOLATED_CLUSTER]
        assert len(isolated_gaps) > 0
        
        # Check gap properties
        isolated_gap = isolated_gaps[0]
        assert "node2" in isolated_gap.source_nodes
        assert isolated_gap.confidence > 0.8
        assert "structural_analysis" in isolated_gap.detection_method
    
    async def test_detect_semantic_gaps(self, gap_detection_service):
        """Test semantic gap detection."""
        gaps = await gap_detection_service.detect_semantic_gaps("test query", "AI")
        
        # Should detect semantically similar but unconnected nodes
        semantic_gaps = [g for g in gaps if g.gap_type == GapType.MISSING_RELATIONSHIP]
        
        # Verify gap detection logic
        for gap in semantic_gaps:
            assert len(gap.source_nodes) >= 2
            assert gap.confidence > 0.0
            assert "semantic" in gap.detection_method
    
    async def test_detect_connectivity_gaps(self, gap_detection_service):
        """Test connectivity gap detection."""
        gaps = await gap_detection_service.detect_connectivity_gaps()
        
        # Should detect under-connectivity if graph is sparse
        connectivity_gaps = [g for g in gaps if g.gap_type == GapType.WEAK_CONNECTION]
        
        # Verify detection logic
        for gap in connectivity_gaps:
            assert gap.confidence > 0.0
            assert "connectivity_analysis" in gap.detection_method
    
    async def test_detect_gaps_with_caching(self, gap_detection_service):
        """Test gap detection with caching."""
        query = "test query"
        focus_area = "AI"
        
        # First call
        gaps1 = await gap_detection_service.detect_gaps(query, None, focus_area)
        initial_cache_hits = gap_detection_service.stats["cache_hits"]
        
        # Second call should hit cache
        gaps2 = await gap_detection_service.detect_gaps(query, None, focus_area)
        final_cache_hits = gap_detection_service.stats["cache_hits"]
        
        assert final_cache_hits > initial_cache_hits
        assert len(gaps1) == len(gaps2)
    
    async def test_deduplicate_gaps(self, gap_detection_service):
        """Test gap deduplication."""
        # Create duplicate gaps
        gap1 = DetectedGap(
            gap_type=GapType.MISSING_NODE,
            source_nodes=["node1", "node2"],
            description="Test gap"
        )
        gap2 = DetectedGap(
            gap_type=GapType.MISSING_NODE,
            source_nodes=["node2", "node1"],  # Same nodes, different order
            description="Duplicate gap"
        )
        gap3 = DetectedGap(
            gap_type=GapType.MISSING_RELATIONSHIP,
            source_nodes=["node1", "node2"],
            description="Different type"
        )
        
        gaps = [gap1, gap2, gap3]
        unique_gaps = await gap_detection_service._deduplicate_gaps(gaps)
        
        # Should remove gap2 as duplicate of gap1
        assert len(unique_gaps) == 2
        
        # Should keep gap1 and gap3 (different types)
        gap_types = [g.gap_type for g in unique_gaps]
        assert GapType.MISSING_NODE in gap_types
        assert GapType.MISSING_RELATIONSHIP in gap_types
    
    async def test_rank_gaps_by_priority(self, gap_detection_service):
        """Test gap ranking by priority."""
        gaps = [
            DetectedGap(priority=0.5, confidence=0.6),  # Score: 0.3
            DetectedGap(priority=0.8, confidence=0.9),  # Score: 0.72
            DetectedGap(priority=0.3, confidence=0.8),  # Score: 0.24
        ]
        
        ranked_gaps = await gap_detection_service._rank_gaps_by_priority(gaps)
        
        # Should be sorted by priority * confidence (descending)
        scores = [g.priority * g.confidence for g in ranked_gaps]
        assert scores == sorted(scores, reverse=True)
    
    async def test_find_isolated_nodes(self, gap_detection_service):
        """Test isolated node detection."""
        gaps = await gap_detection_service._find_isolated_nodes()
        
        # Should find node2 as isolated
        isolated_node_ids = []
        for gap in gaps:
            isolated_node_ids.extend(gap.source_nodes)
        
        assert "node2" in isolated_node_ids
    
    async def test_find_underconnected_nodes(self, gap_detection_service):
        """Test underconnected node detection."""
        gaps = await gap_detection_service._find_underconnected_nodes()
        
        # Verify logic for high-trust, low-connection nodes
        for gap in gaps:
            assert gap.gap_type == GapType.WEAK_CONNECTION
            # Should only flag high-trust nodes with few connections
    
    async def test_are_nodes_connected(self, gap_detection_service):
        """Test node connection checking."""
        # Test connected nodes
        connected = await gap_detection_service._are_nodes_connected("node1", "node3")
        assert connected  # They should be connected via edge2
        
        # Test unconnected nodes
        unconnected = await gap_detection_service._are_nodes_connected("node1", "node2")
        assert not unconnected  # No direct connection
    
    async def test_statistics_tracking(self, gap_detection_service):
        """Test statistics tracking."""
        initial_stats = gap_detection_service.get_statistics()
        
        # Perform some operations
        await gap_detection_service.detect_gaps("test")
        await gap_detection_service.detect_structural_gaps()
        
        final_stats = gap_detection_service.get_statistics()
        
        # Verify statistics updated
        assert final_stats["gaps_detected"] >= initial_stats["gaps_detected"]
        assert final_stats["analyses_performed"] > initial_stats["analyses_performed"]
    
    async def test_error_handling(self, service_config):
        """Test error handling in gap detection."""
        # Create service with invalid config
        service_config.trust_graph = None
        service = GapDetectionService(service_config)
        await service.initialize()
        
        # Should handle missing trust graph gracefully
        gaps = await service.detect_gaps("test query")
        assert gaps == []  # Should return empty list, not crash
    
    async def test_cleanup(self, gap_detection_service):
        """Test service cleanup."""
        # Add some cache entries
        gap_detection_service.set_cache("test_key", "test_value")
        
        assert gap_detection_service.is_initialized
        assert len(gap_detection_service._cache) > 0
        
        await gap_detection_service.cleanup()
        
        assert not gap_detection_service.is_initialized
        assert len(gap_detection_service._cache) == 0
    
    async def test_async_context_manager(self, service_config):
        """Test async context manager functionality."""
        async with GapDetectionService(service_config) as service:
            assert service.is_initialized
            gaps = await service.detect_gaps("test")
            assert isinstance(gaps, list)
        
        # Should be cleaned up after context exit
        assert not service.is_initialized
    
    @pytest.mark.parametrize("method", [
        "structural_analysis",
        "semantic_clustering", 
        "connectivity_analysis",
        "trust_inconsistency"
    ])
    async def test_detection_methods(self, gap_detection_service, method):
        """Test individual detection methods."""
        gaps = await gap_detection_service._run_detection_method(
            method, "test query", [], "test area"
        )
        
        assert isinstance(gaps, list)
        # Each method should return valid gaps or empty list
        for gap in gaps:
            assert isinstance(gap, DetectedGap)
            assert gap.detection_method == method


if __name__ == "__main__":
    pytest.main([__file__])