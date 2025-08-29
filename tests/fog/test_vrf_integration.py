"""
VRF Integration Tests

End-to-end integration tests demonstrating the complete VRF neighbor selection system
working with all components:
- VRF cryptographic security
- Reputation system integration
- Topology management and healing
- P2P networking integration
- Eclipse attack prevention
"""

import asyncio
import pytest
import time
from unittest.mock import MagicMock, patch, AsyncMock

# Import all VRF components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'infrastructure'))

from fog.vrf.vrf_neighbor_selection import VRFNeighborSelector, NodeInfo, VRFStatus
from fog.vrf.topology_manager import TopologyManager, TopologyHealthStatus
from fog.vrf.vrf_p2p_integration import VRFIntegrationManager
from fog.reputation.bayesian_reputation import (
    BayesianReputationEngine, 
    ReputationEvent, 
    EventType,
    ReputationTier
)


class TestCompleteVRFIntegration:
    """Test complete VRF system integration."""
    
    @pytest.fixture
    async def complete_vrf_system(self):
        """Create complete VRF system with all components."""
        # Create reputation engine
        reputation_engine = BayesianReputationEngine()
        
        # Create integration manager
        integration_manager = VRFIntegrationManager(
            node_id="integration_test_node",
            reputation_engine=reputation_engine,
            target_degree=6,
            min_degree=3,
            max_degree=10
        )
        
        # Start system
        await integration_manager.start()
        
        yield integration_manager
        
        # Cleanup
        await integration_manager.stop()
    
    async def test_end_to_end_neighbor_selection(self, complete_vrf_system):
        """Test complete end-to-end neighbor selection process."""
        manager = complete_vrf_system
        
        # Add diverse set of nodes with different characteristics
        test_nodes = [
            # High-quality nodes
            NodeInfo(
                node_id="high_quality_1",
                public_key=b"hq1_key",
                address="10.0.1.1",
                port=8000,
                reliability_score=0.95,
                trust_score=0.9,
                uptime_hours=720,  # 30 days
                latency_ms=25,
                bandwidth_mbps=150
            ),
            NodeInfo(
                node_id="high_quality_2", 
                public_key=b"hq2_key",
                address="10.0.2.1",
                port=8000,
                reliability_score=0.92,
                trust_score=0.88,
                uptime_hours=600,
                latency_ms=30,
                bandwidth_mbps=120
            ),
            # Medium-quality nodes
            NodeInfo(
                node_id="medium_quality_1",
                public_key=b"mq1_key", 
                address="10.0.3.1",
                port=8000,
                reliability_score=0.75,
                trust_score=0.7,
                uptime_hours=200,
                latency_ms=80,
                bandwidth_mbps=50
            ),
            NodeInfo(
                node_id="medium_quality_2",
                public_key=b"mq2_key",
                address="10.0.4.1", 
                port=8000,
                reliability_score=0.78,
                trust_score=0.72,
                uptime_hours=180,
                latency_ms=75,
                bandwidth_mbps=60
            ),
            # Low-quality nodes (should be deprioritized)
            NodeInfo(
                node_id="low_quality_1",
                public_key=b"lq1_key",
                address="10.0.5.1",
                port=8000,
                reliability_score=0.4,
                trust_score=0.3,
                uptime_hours=20,
                latency_ms=200,
                bandwidth_mbps=5
            ),
            # Potentially malicious nodes
            NodeInfo(
                node_id="suspicious_1",
                public_key=b"sus1_key",
                address="192.168.1.1",  # Same subnet as others below
                port=8000,
                reliability_score=0.3,
                trust_score=0.2,
                connection_count=80,  # Suspiciously high
                latency_ms=5,  # Too good to be true
                bandwidth_mbps=1000  # Unrealistic
            ),
            NodeInfo(
                node_id="suspicious_2",
                public_key=b"sus2_key",
                address="192.168.1.2",  # Same subnet
                port=8000,
                reliability_score=0.25,
                trust_score=0.15,
                connection_count=85,
                latency_ms=3,
                bandwidth_mbps=1200
            ),
        ]
        
        # Add nodes to system
        for node in test_nodes:
            success = await manager.add_discovered_node(node)
            assert success is True
        
        # Build reputation history for some nodes
        reputation_engine = manager.reputation_engine
        
        # High-quality nodes get good reputation
        for node_id in ["high_quality_1", "high_quality_2"]:
            for _ in range(50):
                event = ReputationEvent(
                    node_id=node_id,
                    event_type=EventType.TASK_SUCCESS,
                    timestamp=time.time() - 3600 * (_ + 1),  # Spread over time
                    quality_score=0.9
                )
                reputation_engine.record_event(event)
        
        # Medium-quality nodes get mixed reputation
        for node_id in ["medium_quality_1", "medium_quality_2"]:
            for i in range(30):
                event_type = EventType.TASK_SUCCESS if i % 3 != 0 else EventType.TASK_FAILURE
                event = ReputationEvent(
                    node_id=node_id,
                    event_type=event_type,
                    timestamp=time.time() - 3600 * (i + 1),
                    quality_score=0.7 if event_type == EventType.TASK_SUCCESS else 0.3
                )
                reputation_engine.record_event(event)
        
        # Suspicious nodes get bad reputation
        for node_id in ["suspicious_1", "suspicious_2"]:
            for _ in range(20):
                event = ReputationEvent(
                    node_id=node_id,
                    event_type=EventType.TASK_FAILURE,
                    timestamp=time.time() - 3600 * (_ + 1),
                    quality_score=0.1
                )
                reputation_engine.record_event(event)
        
        # Get optimal neighbors
        neighbors = await manager.get_optimal_neighbors(count=4)
        
        # Verify selection quality
        assert len(neighbors) <= 4
        
        # High-quality nodes should be preferred
        high_quality_selected = sum(1 for n in neighbors if n.startswith("high_quality"))
        suspicious_selected = sum(1 for n in neighbors if n.startswith("suspicious"))
        
        assert high_quality_selected > 0  # Should select some high-quality nodes
        assert suspicious_selected == 0   # Should not select suspicious nodes
        
        # Verify eclipse attack prevention
        # Try to force selection of only suspicious nodes
        vrf_selector = manager.vrf_selector
        suspicious_only = ["suspicious_1", "suspicious_2", "low_quality_1"]
        
        eclipse_detected = await vrf_selector._detect_eclipse_attempt(suspicious_only)
        assert eclipse_detected is True  # Should detect eclipse attempt
    
    async def test_topology_health_monitoring_integration(self, complete_vrf_system):
        """Test topology health monitoring and healing integration."""
        manager = complete_vrf_system
        topology_manager = manager.topology_manager
        
        # Add nodes to create initial topology
        nodes = []
        for i in range(12):
            node = NodeInfo(
                node_id=f"topo_node_{i}",
                public_key=f"key_{i}".encode(),
                address=f"10.1.{i//4}.{i%4}",  # Diverse subnets
                port=8000,
                reliability_score=0.8 + 0.1 * (i % 3),
                trust_score=0.75 + 0.15 * (i % 4) / 4
            )
            nodes.append(node)
            await manager.add_discovered_node(node)
        
        # Create initial connections
        initial_neighbors = await manager.get_optimal_neighbors(count=6)
        assert len(initial_neighbors) > 0
        
        # Update topology with realistic adjacency
        adjacency_updates = {}
        for i, node in enumerate(nodes):
            # Each node connects to 3-4 others
            neighbors = set()
            for j in range(3):
                neighbor_idx = (i + j + 1) % len(nodes)
                neighbors.add(f"topo_node_{neighbor_idx}")
            adjacency_updates[node.node_id] = neighbors
        
        await topology_manager.update_network_topology(adjacency_updates)
        
        # Assess initial health
        initial_health = await topology_manager.assess_topology_health()
        assert initial_health in [
            TopologyHealthStatus.GOOD,
            TopologyHealthStatus.EXCELLENT,
            TopologyHealthStatus.DEGRADED
        ]
        
        # Simulate network degradation (remove nodes)
        degraded_adjacency = {}
        remaining_nodes = nodes[:8]  # Remove 1/3 of nodes
        
        for node in remaining_nodes:
            neighbors = adjacency_updates[node.node_id]
            # Remove connections to deleted nodes
            active_neighbors = {n for n in neighbors if any(n == rn.node_id for rn in remaining_nodes)}
            degraded_adjacency[node.node_id] = active_neighbors
        
        await topology_manager.update_network_topology(degraded_adjacency)
        
        # Check if degradation was detected
        degraded_health = await topology_manager.assess_topology_health()
        
        # Plan healing if needed
        if degraded_health in [TopologyHealthStatus.DEGRADED, TopologyHealthStatus.CRITICAL]:
            healing_actions = await topology_manager.plan_healing_actions()
            assert len(healing_actions) > 0
            
            # Execute first healing action
            success = await topology_manager.execute_healing_action(healing_actions[0])
            assert isinstance(success, bool)  # Should complete
    
    async def test_reputation_feedback_loop(self, complete_vrf_system):
        """Test reputation feedback from connection experiences."""
        manager = complete_vrf_system
        
        # Add test nodes
        test_nodes = [
            NodeInfo(
                node_id=f"feedback_node_{i}",
                public_key=f"fb_key_{i}".encode(),
                address=f"10.2.1.{i}",
                port=8000,
                reliability_score=0.8,
                trust_score=0.8
            )
            for i in range(5)
        ]
        
        for node in test_nodes:
            await manager.add_discovered_node(node)
        
        # Simulate connection experiences
        for i, node in enumerate(test_nodes):
            node_id = node.node_id
            metrics = manager.connection_metrics[node_id]
            
            if i < 2:  # First 2 nodes: successful connections
                metrics["connection_attempts"] = 10
                metrics["successful_connections"] = 9
                metrics["failed_connections"] = 1
                metrics["avg_latency"] = 40.0
            else:  # Rest: poor performance
                metrics["connection_attempts"] = 8
                metrics["successful_connections"] = 3
                metrics["failed_connections"] = 5
                metrics["avg_latency"] = 150.0
            
            metrics["last_connection_attempt"] = time.time()
        
        # Manually trigger reputation update (simulate background loop)
        reputation_engine = manager.reputation_engine
        current_time = time.time()
        metrics_data = []
        
        for node_id, metrics in manager.connection_metrics.items():
            if metrics["last_connection_attempt"] > current_time - 3600:
                success_rate = 0.0
                if metrics["connection_attempts"] > 0:
                    success_rate = metrics["successful_connections"] / metrics["connection_attempts"]
                
                metrics_data.append({
                    "node_id": node_id,
                    "tasks_completed": metrics["successful_connections"],
                    "tasks_failed": metrics["failed_connections"],
                    "uptime_ratio": success_rate,
                    "avg_quality_score": min(1.0, 1.0 / (1.0 + metrics["avg_latency"] / 100))
                })
        
        reputation_engine.batch_update_from_metrics(metrics_data)
        
        # Verify reputation reflects performance
        for i, node in enumerate(test_nodes):
            trust_score = reputation_engine.get_trust_score(node.node_id)
            
            if i < 2:  # Good performers should have higher trust
                assert trust_score > 0.6
            else:  # Poor performers should have lower trust
                assert trust_score < 0.6
    
    async def test_vrf_cryptographic_consistency(self, complete_vrf_system):
        """Test VRF cryptographic consistency across operations."""
        manager = complete_vrf_system
        vrf_selector = manager.vrf_selector
        
        # Generate multiple VRF proofs with same input
        test_input = b"consistency_test_input"
        
        proofs = []
        for _ in range(5):
            proof = vrf_selector._generate_vrf_proof(test_input)
            proofs.append(proof)
        
        # All proofs should be identical (deterministic)
        for i in range(1, len(proofs)):
            assert proofs[i].beta == proofs[0].beta
            assert proofs[i].pi == proofs[0].pi
            assert proofs[i].alpha == proofs[0].alpha
        
        # All proofs should verify
        for proof in proofs:
            assert vrf_selector._verify_vrf_proof(proof) is True
        
        # Different inputs should produce different outputs
        different_input = b"different_test_input"
        different_proof = vrf_selector._generate_vrf_proof(different_input)
        
        assert different_proof.beta != proofs[0].beta
        assert different_proof.pi != proofs[0].pi
    
    async def test_system_status_reporting(self, complete_vrf_system):
        """Test comprehensive system status reporting."""
        manager = complete_vrf_system
        
        # Add some nodes
        for i in range(3):
            node = NodeInfo(
                node_id=f"status_node_{i}",
                public_key=f"status_key_{i}".encode(),
                address=f"10.3.1.{i}",
                port=8000
            )
            await manager.add_discovered_node(node)
        
        # Get comprehensive status
        status = manager.get_integration_status()
        
        # Verify all status components are present
        assert "node_id" in status
        assert "vrf_status" in status
        assert "topology_status" in status
        assert "active_connections" in status
        assert "total_known_nodes" in status
        assert "reputation_active" in status
        
        # Check VRF status details
        vrf_status = status["vrf_status"]
        assert "status" in vrf_status
        assert "current_neighbors" in vrf_status
        assert "metrics" in vrf_status
        
        # Check topology status
        topology_status = status["topology_status"]
        assert "health_status" in topology_status
        assert "node_count" in topology_status
        
        # Verify counts are reasonable
        assert status["total_known_nodes"] >= 3
        assert isinstance(status["reputation_active"], bool)


class TestVRFSystemResilience:
    """Test VRF system resilience under adverse conditions."""
    
    @pytest.fixture
    async def resilient_vrf_system(self):
        """Create VRF system configured for resilience testing."""
        reputation_engine = BayesianReputationEngine()
        
        integration_manager = VRFIntegrationManager(
            node_id="resilience_test_node",
            reputation_engine=reputation_engine,
            target_degree=8,
            min_degree=4,
            max_degree=12,
            eclipse_detection_threshold=0.7,  # Strict eclipse detection
            health_check_interval=0.1,  # Fast health checks for testing
        )
        
        await integration_manager.start()
        yield integration_manager
        await integration_manager.stop()
    
    async def test_resilience_to_eclipse_attacks(self, resilient_vrf_system):
        """Test system resilience against sophisticated eclipse attacks."""
        manager = resilient_vrf_system
        
        # Create coordinated eclipse attack scenario
        # Add many malicious nodes with similar characteristics
        malicious_nodes = []
        for i in range(20):  # Large number of attackers
            node = NodeInfo(
                node_id=f"attacker_{i}",
                public_key=f"attack_key_{i}".encode(),
                address=f"192.168.100.{i}",  # Same subnet
                port=8000,
                reliability_score=0.95,  # Fake good metrics
                trust_score=0.9,
                connection_count=100,  # Suspiciously high
                latency_ms=1,  # Too good to be true
                bandwidth_mbps=10000,  # Unrealistic
                uptime_hours=1  # Recently joined
            )
            malicious_nodes.append(node)
            await manager.add_discovered_node(node)
        
        # Add fewer legitimate nodes
        legitimate_nodes = []
        for i in range(5):
            node = NodeInfo(
                node_id=f"legitimate_{i}",
                public_key=f"legit_key_{i}".encode(),
                address=f"10.4.{i}.1",  # Different subnets
                port=8000,
                reliability_score=0.85,
                trust_score=0.8,
                connection_count=15,
                latency_ms=50,
                bandwidth_mbps=100,
                uptime_hours=720  # Long uptime
            )
            legitimate_nodes.append(node)
            await manager.add_discovered_node(node)
        
        # Give malicious nodes bad reputation
        reputation_engine = manager.reputation_engine
        for node in malicious_nodes:
            for _ in range(10):
                event = ReputationEvent(
                    node_id=node.node_id,
                    event_type=EventType.TASK_FAILURE,
                    timestamp=time.time() - 60 * _,  # Recent failures
                    quality_score=0.1
                )
                reputation_engine.record_event(event)
        
        # Give legitimate nodes good reputation
        for node in legitimate_nodes:
            for _ in range(30):
                event = ReputationEvent(
                    node_id=node.node_id,
                    event_type=EventType.TASK_SUCCESS,
                    timestamp=time.time() - 3600 * _,  # Historical success
                    quality_score=0.9
                )
                reputation_engine.record_event(event)
        
        # Attempt neighbor selection
        neighbors = await manager.get_optimal_neighbors(count=6)
        
        # Verify eclipse attack was mitigated
        attacker_count = sum(1 for n in neighbors if n.startswith("attacker_"))
        legitimate_count = sum(1 for n in neighbors if n.startswith("legitimate_"))
        
        # Should strongly prefer legitimate nodes despite being outnumbered
        assert legitimate_count > attacker_count
        assert legitimate_count >= 3  # Should get majority legitimate
    
    async def test_recovery_from_network_partition(self, resilient_vrf_system):
        """Test recovery from network partition scenarios."""
        manager = resilient_vrf_system
        
        # Create two network partitions
        partition_a = []
        partition_b = []
        
        for i in range(6):
            node_a = NodeInfo(
                node_id=f"partition_a_{i}",
                public_key=f"pa_key_{i}".encode(),
                address=f"10.5.1.{i}",
                port=8000,
                reliability_score=0.85,
                trust_score=0.8
            )
            partition_a.append(node_a)
            await manager.add_discovered_node(node_a)
            
            node_b = NodeInfo(
                node_id=f"partition_b_{i}",
                public_key=f"pb_key_{i}".encode(), 
                address=f"10.5.2.{i}",
                port=8000,
                reliability_score=0.85,
                trust_score=0.8
            )
            partition_b.append(node_b)
            await manager.add_discovered_node(node_b)
        
        # Initially, select neighbors from both partitions
        initial_neighbors = await manager.get_optimal_neighbors(count=6)
        
        # Simulate partition: remove partition B
        for node in partition_b:
            await manager.remove_node(node.node_id)
        
        # System should adapt to only partition A
        partition_neighbors = await manager.get_optimal_neighbors(count=4)
        
        # All selected neighbors should be from partition A only
        partition_a_ids = [n.node_id for n in partition_a]
        assert all(neighbor in partition_a_ids for neighbor in partition_neighbors)
        
        # Simulate partition healing: re-add partition B
        for node in partition_b:
            await manager.add_discovered_node(node)
        
        # System should rebalance across both partitions
        healed_neighbors = await manager.get_optimal_neighbors(count=6)
        
        # Should have neighbors from both partitions after healing
        a_count = sum(1 for n in healed_neighbors if n.startswith("partition_a_"))
        b_count = sum(1 for n in healed_neighbors if n.startswith("partition_b_"))
        
        assert a_count > 0
        assert b_count > 0
    
    async def test_performance_under_churn(self, resilient_vrf_system):
        """Test performance under high node churn."""
        manager = resilient_vrf_system
        
        # Add stable base nodes
        stable_nodes = []
        for i in range(8):
            node = NodeInfo(
                node_id=f"stable_{i}",
                public_key=f"stable_key_{i}".encode(),
                address=f"10.6.1.{i}",
                port=8000,
                reliability_score=0.9,
                trust_score=0.85,
                uptime_hours=2000  # Very stable
            )
            stable_nodes.append(node)
            await manager.add_discovered_node(node)
        
        # Simulate high churn: rapidly add and remove nodes
        churn_count = 0
        max_churn = 20
        
        while churn_count < max_churn:
            # Add churning nodes
            churning_nodes = []
            for i in range(3):
                node = NodeInfo(
                    node_id=f"churn_{churn_count}_{i}",
                    public_key=f"churn_key_{churn_count}_{i}".encode(),
                    address=f"10.6.2.{i}",
                    port=8000,
                    reliability_score=0.6,  # Lower reliability
                    trust_score=0.5,
                    uptime_hours=1  # Just joined
                )
                churning_nodes.append(node)
                await manager.add_discovered_node(node)
            
            # Brief pause
            await asyncio.sleep(0.01)
            
            # Remove churning nodes
            for node in churning_nodes:
                await manager.remove_node(node.node_id)
            
            churn_count += 1
        
        # System should still function and prefer stable nodes
        final_neighbors = await manager.get_optimal_neighbors(count=6)
        
        stable_selected = sum(1 for n in final_neighbors if n.startswith("stable_"))
        
        # Should prefer stable nodes despite churn
        assert stable_selected >= 4  # Majority should be stable nodes
        assert len(final_neighbors) > 0  # Should still be able to select neighbors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])