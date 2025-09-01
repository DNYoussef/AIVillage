"""
Comprehensive Security Tests for VRF Neighbor Selection

Tests security properties including:
- Eclipse attack prevention
- VRF proof verification and uniqueness
- Cryptographic security properties
- Network topology resistance
- Reputation integration security
"""

import time
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

# Import VRF components
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "infrastructure"))

from fog.vrf.vrf_neighbor_selection import VRFNeighborSelector, NodeInfo
from fog.vrf.topology_manager import TopologyManager
from fog.vrf.vrf_p2p_integration import VRFIntegrationManager
from fog.reputation.bayesian_reputation import BayesianReputationEngine, ReputationEvent, EventType, ReputationTier


class TestVRFCryptographicSecurity:
    """Test cryptographic security properties of VRF system."""

    @pytest.fixture
    def vrf_selector(self):
        """Create VRF selector for testing."""
        return VRFNeighborSelector(node_id="test_node_1", target_degree=8, min_degree=4, max_degree=16)

    def test_vrf_proof_generation_deterministic(self, vrf_selector):
        """Test that VRF proofs are deterministic for same input."""
        alpha = b"test_input_data"

        # Generate proof twice
        proof1 = vrf_selector._generate_vrf_proof(alpha)
        proof2 = vrf_selector._generate_vrf_proof(alpha)

        # Should be identical
        assert proof1.beta == proof2.beta
        assert proof1.pi == proof2.pi
        assert proof1.alpha == proof2.alpha
        assert proof1.public_key == proof2.public_key

    def test_vrf_proof_different_inputs_different_outputs(self, vrf_selector):
        """Test that different inputs produce different VRF outputs."""
        alpha1 = b"input_1"
        alpha2 = b"input_2"

        proof1 = vrf_selector._generate_vrf_proof(alpha1)
        proof2 = vrf_selector._generate_vrf_proof(alpha2)

        # Different inputs should produce different outputs
        assert proof1.beta != proof2.beta
        assert proof1.pi != proof2.pi

    def test_vrf_proof_verification_valid(self, vrf_selector):
        """Test verification of valid VRF proofs."""
        alpha = b"test_verification_input"

        proof = vrf_selector._generate_vrf_proof(alpha)
        is_valid = vrf_selector._verify_vrf_proof(proof)

        assert is_valid is True

    def test_vrf_proof_verification_invalid_modified_beta(self, vrf_selector):
        """Test verification fails with modified beta."""
        alpha = b"test_invalid_beta"

        proof = vrf_selector._generate_vrf_proof(alpha)
        # Modify beta
        modified_beta = bytearray(proof.beta)
        modified_beta[0] = (modified_beta[0] + 1) % 256
        proof.beta = bytes(modified_beta)

        is_valid = vrf_selector._verify_vrf_proof(proof)
        assert is_valid is False

    def test_vrf_proof_verification_invalid_signature(self, vrf_selector):
        """Test verification fails with invalid signature."""
        alpha = b"test_invalid_signature"

        proof = vrf_selector._generate_vrf_proof(alpha)
        # Modify signature
        modified_pi = bytearray(proof.pi)
        modified_pi[0] = (modified_pi[0] + 1) % 256
        proof.pi = bytes(modified_pi)

        is_valid = vrf_selector._verify_vrf_proof(proof)
        assert is_valid is False

    def test_vrf_proof_cross_verification_fails(self):
        """Test that VRF proofs from different keys can't be cross-verified."""
        # Create two different VRF selectors (different keys)
        vrf1 = VRFNeighborSelector(node_id="node_1")
        vrf2 = VRFNeighborSelector(node_id="node_2")

        alpha = b"cross_verification_test"

        # Generate proof with first key
        proof1 = vrf1._generate_vrf_proof(alpha)

        # Try to verify with second key
        is_valid = vrf2._verify_vrf_proof(proof1)
        assert is_valid is False

    def test_vrf_output_unpredictability(self, vrf_selector):
        """Test that VRF outputs appear random and unpredictable."""
        outputs = []

        # Generate many VRF outputs
        for i in range(100):
            alpha = f"test_input_{i}".encode()
            proof = vrf_selector._generate_vrf_proof(alpha)
            outputs.append(proof.beta)

        # Convert to integers for analysis
        output_ints = [int.from_bytes(output[:8], "big") for output in outputs]

        # Test for randomness properties
        # 1. No duplicates (very unlikely with good VRF)
        assert len(set(output_ints)) == len(output_ints)

        # 2. Mean should be around middle of range
        mean_output = np.mean(output_ints)
        expected_mean = (2**64) / 2
        # Allow 10% deviation
        assert abs(mean_output - expected_mean) / expected_mean < 0.1

        # 3. Standard deviation should be significant
        std_dev = np.std(output_ints)
        assert std_dev > (2**64) * 0.1  # At least 10% of range


class TestEclipseAttackPrevention:
    """Test eclipse attack prevention mechanisms."""

    @pytest.fixture
    def reputation_engine(self):
        """Create reputation engine for testing."""
        return BayesianReputationEngine()

    @pytest.fixture
    def vrf_selector_with_reputation(self, reputation_engine):
        """Create VRF selector with reputation engine."""
        return VRFNeighborSelector(
            node_id="test_node", reputation_engine=reputation_engine, target_degree=6, eclipse_detection_threshold=0.6
        )

    async def test_eclipse_attack_detection_suspicious_nodes(self, vrf_selector_with_reputation):
        """Test detection of eclipse attacks with suspicious nodes."""
        # Add suspicious nodes with low trust scores
        suspicious_nodes = []
        for i in range(8):
            node_id = f"suspicious_node_{i}"
            node_info = NodeInfo(
                node_id=node_id,
                public_key=b"fake_key",
                address=f"192.168.1.{i}",
                port=8000,
                trust_score=0.2,  # Low trust
                reliability_score=0.1,  # Low reliability
                connection_count=60,  # Suspiciously high
            )
            await vrf_selector_with_reputation.add_node(node_info)
            suspicious_nodes.append(node_id)

        # Test eclipse detection
        is_eclipse = await vrf_selector_with_reputation._detect_eclipse_attempt(suspicious_nodes)

        assert is_eclipse is True
        assert vrf_selector_with_reputation.eclipse_attempts > 0

    async def test_no_false_positive_with_good_nodes(self, vrf_selector_with_reputation):
        """Test no false positives with legitimate good nodes."""
        # Add legitimate nodes with good metrics
        good_nodes = []
        for i in range(6):
            node_id = f"good_node_{i}"
            node_info = NodeInfo(
                node_id=node_id,
                public_key=b"legitimate_key",
                address=f"10.0.{i//4}.{i%4}",  # Different subnets
                port=8000,
                trust_score=0.9,
                reliability_score=0.95,
                connection_count=15,  # Normal connection count
                uptime_hours=168,  # 1 week uptime
                latency_ms=50,
            )
            await vrf_selector_with_reputation.add_node(node_info)
            good_nodes.append(node_id)

        # Test no eclipse detection
        is_eclipse = await vrf_selector_with_reputation._detect_eclipse_attempt(good_nodes)

        assert is_eclipse is False

    async def test_reputation_based_eclipse_detection(self, vrf_selector_with_reputation, reputation_engine):
        """Test eclipse detection using reputation scores."""
        # Create nodes with poor reputation
        low_rep_nodes = []
        for i in range(5):
            node_id = f"low_rep_node_{i}"
            node_info = NodeInfo(
                node_id=node_id,
                public_key=b"key",
                address=f"192.168.2.{i}",
                port=8000,
                trust_score=0.8,  # Initially good
                reliability_score=0.8,
            )
            await vrf_selector_with_reputation.add_node(node_info)

            # Record bad reputation events
            for _ in range(10):
                event = ReputationEvent(node_id=node_id, event_type=EventType.TASK_FAILURE, timestamp=time.time())
                reputation_engine.record_event(event)

            low_rep_nodes.append(node_id)

        # Test eclipse detection based on reputation
        is_eclipse = await vrf_selector_with_reputation._detect_eclipse_attempt(low_rep_nodes)

        assert is_eclipse is True

    async def test_geographic_diversity_validation(self, vrf_selector_with_reputation):
        """Test that geographic diversity prevents eclipse attacks."""
        # Add nodes all from same subnet (suspicious)
        same_subnet_nodes = []
        for i in range(6):
            node_id = f"same_subnet_node_{i}"
            node_info = NodeInfo(
                node_id=node_id,
                public_key=b"key",
                address=f"192.168.1.{i}",  # Same subnet
                port=8000,
                trust_score=0.7,
                reliability_score=0.7,
            )
            await vrf_selector_with_reputation.add_node(node_info)
            same_subnet_nodes.append(node_id)

        # Test expansion validation should fail due to lack of diversity
        is_valid = await vrf_selector_with_reputation._validate_expansion_properties(same_subnet_nodes)

        assert is_valid is False


class TestTopologyProperties:
    """Test network topology maintenance and properties."""

    @pytest.fixture
    def vrf_selector(self):
        return VRFNeighborSelector(node_id="topology_test_node")

    @pytest.fixture
    def topology_manager(self, vrf_selector):
        return TopologyManager(vrf_selector=vrf_selector, target_spectral_gap=0.3, min_k_core=3)

    async def test_k_core_decomposition(self, topology_manager):
        """Test K-core decomposition functionality."""
        # Create test adjacency graph
        adjacency = {
            "node_1": {"node_2", "node_3", "node_4"},
            "node_2": {"node_1", "node_3", "node_4"},
            "node_3": {"node_1", "node_2", "node_4", "node_5"},
            "node_4": {"node_1", "node_2", "node_3", "node_5"},
            "node_5": {"node_3", "node_4"},
        }

        k_cores = topology_manager.graph_analyzer.compute_k_core_decomposition(adjacency)

        # Verify k-core structure
        assert len(k_cores) > 0
        # Nodes 1-4 should form a 3-core (each has 3+ connections within core)
        assert 3 in k_cores
        high_k_nodes = k_cores[max(k_cores.keys())]
        assert len(high_k_nodes) >= 3

    def test_spectral_gap_computation(self, topology_manager):
        """Test spectral gap computation for graph quality."""
        # Create adjacency matrix for well-connected graph
        n = 10
        adjacency_matrix = np.zeros((n, n))

        # Create ring + random edges (good expansion)
        for i in range(n):
            adjacency_matrix[i][(i + 1) % n] = 1
            adjacency_matrix[(i + 1) % n][i] = 1

            # Add random edges
            for j in range(2):
                target = (i + 3 + j) % n
                adjacency_matrix[i][target] = 1
                adjacency_matrix[target][i] = 1

        spectral_gap = topology_manager.graph_analyzer.compute_spectral_gap(adjacency_matrix)

        # Should have reasonable spectral gap for good expansion
        assert spectral_gap > 0.1
        assert spectral_gap < 1.0

    async def test_topology_healing_actions(self, topology_manager):
        """Test topology healing action planning."""
        # Initialize with poor topology metrics
        topology_manager.current_metrics = MagicMock()
        topology_manager.current_metrics.spectral_gap = 0.05  # Below target
        topology_manager.current_metrics.max_k_core = 1  # Below minimum
        topology_manager.current_metrics.expansion_ratio = 0.3  # Below target

        # Plan healing actions
        healing_actions = await topology_manager.plan_healing_actions()

        assert len(healing_actions) > 0

        # Should include spectral gap improvement
        action_types = [action.action_type for action in healing_actions]
        assert "improve_spectral_gap" in action_types
        assert "strengthen_k_core" in action_types

    async def test_network_diameter_computation(self, topology_manager):
        """Test network diameter and radius computation."""
        # Create path graph (worst case for diameter)
        adjacency = {}
        n = 8
        for i in range(n):
            adjacency[f"node_{i}"] = set()
            if i > 0:
                adjacency[f"node_{i}"].add(f"node_{i-1}")
            if i < n - 1:
                adjacency[f"node_{i}"].add(f"node_{i+1}")

        diameter, radius = topology_manager.graph_analyzer.compute_diameter(adjacency)

        # Path graph should have diameter n-1
        assert diameter == n - 1
        assert radius <= diameter


class TestVRFReputationIntegration:
    """Test integration between VRF system and reputation engine."""

    @pytest.fixture
    def reputation_engine(self):
        engine = BayesianReputationEngine()

        # Add some test reputation data
        nodes_data = [
            ("high_rep_node", 0.95, ReputationTier.PLATINUM),
            ("medium_rep_node", 0.75, ReputationTier.SILVER),
            ("low_rep_node", 0.3, ReputationTier.UNTRUSTED),
            ("new_node", 0.5, ReputationTier.BRONZE),
        ]

        for node_id, score, tier in nodes_data:
            # Simulate reputation events to achieve desired scores
            events_count = 50
            success_rate = score

            for i in range(events_count):
                event_type = EventType.TASK_SUCCESS if i < success_rate * events_count else EventType.TASK_FAILURE
                event = ReputationEvent(
                    node_id=node_id,
                    event_type=event_type,
                    timestamp=time.time() - (events_count - i) * 3600,  # Spread over time
                )
                engine.record_event(event)

        return engine

    @pytest.fixture
    def vrf_with_reputation(self, reputation_engine):
        return VRFNeighborSelector(node_id="rep_test_node", reputation_engine=reputation_engine)

    async def test_reputation_weighted_selection(self, vrf_with_reputation, reputation_engine):
        """Test that reputation affects node selection weights."""
        # Add nodes with different reputation scores
        test_nodes = [("high_rep_node", 0.95), ("medium_rep_node", 0.75), ("low_rep_node", 0.3), ("new_node", 0.5)]

        for node_id, expected_score in test_nodes:
            node_info = NodeInfo(
                node_id=node_id,
                public_key=b"key",
                address="127.0.0.1",
                port=8000,
                reliability_score=0.8,
                trust_score=0.8,
            )
            await vrf_with_reputation.add_node(node_info)

        # Test weight calculation includes reputation
        for node_id, expected_score in test_nodes:
            node_info = vrf_with_reputation.known_nodes[node_id]
            weight = vrf_with_reputation._calculate_node_weight(node_info)

            # Higher reputation nodes should have higher weights
            if expected_score > 0.8:
                assert weight > 1.0  # Should get bonus
            elif expected_score < 0.4:
                assert weight < 1.0  # Should get penalty

    async def test_tier_based_bonuses(self, vrf_with_reputation):
        """Test that reputation tiers provide appropriate bonuses."""
        # Create nodes and manually set their reputation tiers
        tier_nodes = [
            ("diamond_node", ReputationTier.DIAMOND),
            ("gold_node", ReputationTier.GOLD),
            ("bronze_node", ReputationTier.BRONZE),
            ("untrusted_node", ReputationTier.UNTRUSTED),
        ]

        weights = {}
        for node_id, tier in tier_nodes:
            node_info = NodeInfo(
                node_id=node_id,
                public_key=b"key",
                address="127.0.0.1",
                port=8000,
                reliability_score=0.8,
                trust_score=0.8,
            )
            await vrf_with_reputation.add_node(node_info)

            # Mock reputation score with specific tier
            with patch.object(vrf_with_reputation.reputation_engine, "get_reputation_score") as mock_get_rep:
                mock_score = MagicMock()
                mock_score.mean_score = 0.8
                mock_score.uncertainty = 0.1
                mock_score.tier = tier
                mock_get_rep.return_value = mock_score

                weight = vrf_with_reputation._calculate_node_weight(node_info)
                weights[tier] = weight

        # Higher tiers should have higher weights
        assert weights[ReputationTier.DIAMOND] > weights[ReputationTier.GOLD]
        assert weights[ReputationTier.GOLD] > weights[ReputationTier.BRONZE]
        assert weights[ReputationTier.BRONZE] > weights[ReputationTier.UNTRUSTED]


class TestVRFIntegrationManager:
    """Test the VRF integration manager functionality."""

    @pytest.fixture
    def integration_manager(self):
        return VRFIntegrationManager(node_id="integration_test_node")

    async def test_integration_manager_startup_shutdown(self, integration_manager):
        """Test proper startup and shutdown of integration manager."""
        # Test startup
        started = await integration_manager.start()
        assert started is True

        # Verify components are running
        status = integration_manager.get_integration_status()
        assert status["node_id"] == "integration_test_node"
        assert "vrf_status" in status
        assert "topology_status" in status

        # Test shutdown
        await integration_manager.stop()
        # Should complete without errors

    async def test_node_discovery_integration(self, integration_manager):
        """Test integration of node discovery with VRF system."""
        await integration_manager.start()

        # Add discovered nodes
        test_nodes = []
        for i in range(5):
            node_info = NodeInfo(node_id=f"discovered_node_{i}", public_key=b"key", address=f"192.168.1.{i}", port=8000)
            success = await integration_manager.add_discovered_node(node_info)
            assert success is True
            test_nodes.append(node_info.node_id)

        # Test neighbor selection includes discovered nodes
        neighbors = await integration_manager.get_optimal_neighbors(count=3)
        assert len(neighbors) <= 3
        assert all(neighbor in test_nodes for neighbor in neighbors)

        await integration_manager.stop()

    async def test_connection_metrics_tracking(self, integration_manager):
        """Test connection metrics are properly tracked."""
        await integration_manager.start()

        # Add a node
        node_info = NodeInfo(node_id="metrics_test_node", public_key=b"key", address="192.168.1.100", port=8000)
        await integration_manager.add_discovered_node(node_info)

        # Verify metrics tracking was initialized
        assert "metrics_test_node" in integration_manager.connection_metrics
        metrics = integration_manager.connection_metrics["metrics_test_node"]
        assert "connection_attempts" in metrics
        assert "successful_connections" in metrics
        assert "avg_latency" in metrics

        await integration_manager.stop()


# Performance and Stress Tests
class TestVRFPerformance:
    """Test VRF system performance and scalability."""

    def test_vrf_proof_generation_performance(self):
        """Test VRF proof generation performance."""
        vrf = VRFNeighborSelector(node_id="perf_test")

        # Measure time for batch proof generation
        start_time = time.time()
        proofs = []

        for i in range(100):
            alpha = f"performance_test_{i}".encode()
            proof = vrf._generate_vrf_proof(alpha)
            proofs.append(proof)

        end_time = time.time()
        avg_time = (end_time - start_time) / 100

        # Should be fast enough for practical use (< 10ms per proof)
        assert avg_time < 0.01
        assert len(proofs) == 100

    def test_vrf_verification_performance(self):
        """Test VRF proof verification performance."""
        vrf = VRFNeighborSelector(node_id="verify_perf_test")

        # Generate proofs to verify
        proofs = []
        for i in range(50):
            alpha = f"verify_test_{i}".encode()
            proof = vrf._generate_vrf_proof(alpha)
            proofs.append(proof)

        # Measure verification time
        start_time = time.time()

        for proof in proofs:
            is_valid = vrf._verify_vrf_proof(proof)
            assert is_valid is True

        end_time = time.time()
        avg_time = (end_time - start_time) / 50

        # Verification should be fast (< 5ms per verification)
        assert avg_time < 0.005

    async def test_large_network_selection_performance(self):
        """Test neighbor selection performance with large network."""
        vrf = VRFNeighborSelector(node_id="large_network_test", target_degree=20)

        # Add many nodes
        for i in range(1000):
            node_info = NodeInfo(
                node_id=f"large_net_node_{i}",
                public_key=b"key",
                address=f"10.{i//256}.{(i//16)%16}.{i%16}",
                port=8000,
                reliability_score=0.5 + 0.5 * (i % 100) / 100,
                trust_score=0.5 + 0.5 * ((i * 7) % 100) / 100,
            )
            await vrf.add_node(node_info)

        # Measure selection time
        start_time = time.time()
        neighbors = await vrf.select_neighbors()
        end_time = time.time()

        selection_time = end_time - start_time

        # Should complete selection quickly even with 1000 nodes
        assert selection_time < 5.0  # Less than 5 seconds
        assert len(neighbors) == vrf.target_degree


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
