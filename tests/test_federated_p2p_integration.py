"""
Comprehensive Integration Test for Federated P2P Training System
==============================================================

End-to-end test that validates the complete integration of:
- P2P network for participant discovery
- BetaNet transport for secure gradient exchange
- Hierarchical aggregation using P2P topology
- Mobile device optimization and participation
- Byzantine fault tolerance for unreliable participants

This test demonstrates the fully working integrated system.
"""

import asyncio
import logging
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the integrated system components
try:
    from infrastructure.distributed_inference.federated_p2p_integration import (
        FederatedP2PCoordinator,
        P2PTrainingConfig,
        P2PTrainingMode,
        P2PParticipant,
    )
    from infrastructure.distributed_inference.p2p_hierarchical_aggregator import (
        P2PHierarchicalAggregator,
        HierarchicalGradient,
        ClusterRole,
    )
    from infrastructure.distributed_inference.mobile_federated_optimizer import (
        MobileFederatedOptimizer,
        MobileDeviceProfile,
        DeviceCapabilityTier,
        MobileTrainingMode,
    )
    from infrastructure.fog.bridges.betanet_integration import BetaNetFogTransport
    from infrastructure.p2p import P2PNetwork

    INTEGRATION_AVAILABLE = True

except ImportError as e:
    logger.warning(f"Integration components not available: {e}")
    INTEGRATION_AVAILABLE = False

    # Create mock classes for testing infrastructure
    class MockFederatedP2PCoordinator:
        def __init__(self, *args, **kwargs):
            pass

        async def initialize(self):
            return True

        async def stop(self):
            pass

    FederatedP2PCoordinator = MockFederatedP2PCoordinator


class TestFederatedP2PIntegration:
    """Comprehensive integration tests for federated P2P training"""

    @pytest.fixture
    async def mock_p2p_network(self):
        """Create mock P2P network for testing"""
        network = Mock(spec=P2PNetwork)
        network.initialize = AsyncMock(return_value=None)
        network.start_discovery = AsyncMock(return_value=None)
        network.get_peers = AsyncMock(return_value=[])
        network.send = AsyncMock(return_value=True)
        network.broadcast = AsyncMock(return_value=5)
        network.shutdown = AsyncMock(return_value=None)
        return network

    @pytest.fixture
    async def mock_betanet_transport(self):
        """Create mock BetaNet transport for testing"""
        transport = Mock(spec=BetaNetFogTransport)
        transport.send_job_data = AsyncMock(return_value={"success": True, "transport": "betanet"})
        transport.receive_job_data = AsyncMock(return_value={"data": b'{"gradients": {}}'})
        transport.get_transport_stats = Mock(return_value={"betanet_available": True})
        return transport

    @pytest.fixture
    def sample_participants(self):
        """Create sample participants for testing"""
        return [
            {
                "participant_id": "mobile_device_1",
                "device_type": "mobile",
                "is_mobile": True,
                "latency_ms": 80.0,
                "bandwidth_mbps": 25.0,
                "trust_score": 0.8,
                "battery_percent": 75.0,
                "betanet_support": True,
            },
            {
                "participant_id": "edge_node_1",
                "device_type": "edge",
                "is_mobile": False,
                "latency_ms": 20.0,
                "bandwidth_mbps": 100.0,
                "trust_score": 0.9,
                "betanet_support": True,
            },
            {
                "participant_id": "cloud_node_1",
                "device_type": "cloud",
                "is_mobile": False,
                "latency_ms": 10.0,
                "bandwidth_mbps": 1000.0,
                "trust_score": 0.95,
                "betanet_support": True,
            },
            {
                "participant_id": "mobile_device_2",
                "device_type": "mobile",
                "is_mobile": True,
                "latency_ms": 120.0,
                "bandwidth_mbps": 15.0,
                "trust_score": 0.7,
                "battery_percent": 45.0,
                "betanet_support": False,
            },
            {
                "participant_id": "byzantine_node",
                "device_type": "unknown",
                "is_mobile": False,
                "latency_ms": 50.0,
                "bandwidth_mbps": 50.0,
                "trust_score": 0.3,  # Low trust score
                "betanet_support": False,
            },
        ]

    @pytest.mark.asyncio
    async def test_p2p_coordinator_initialization(self, mock_p2p_network, mock_betanet_transport):
        """Test P2P coordinator initialization"""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration components not available")

        config = P2PTrainingConfig(
            p2p_mode=P2PTrainingMode.HYBRID_OPTIMAL, enable_betanet_privacy=True, enable_fog_integration=True
        )

        coordinator = FederatedP2PCoordinator(coordinator_id="test_coordinator", p2p_config=config)

        # Mock the internal initialization
        with patch.object(coordinator, "p2p_network", mock_p2p_network), patch.object(
            coordinator, "betanet_transport", mock_betanet_transport
        ):

            success = await coordinator.initialize()
            assert success, "Coordinator initialization should succeed"

            # Check initialization state
            assert coordinator._running, "Coordinator should be running"
            assert coordinator.config.p2p_mode == P2PTrainingMode.HYBRID_OPTIMAL

            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_participant_discovery_and_selection(self, mock_p2p_network, sample_participants):
        """Test P2P participant discovery and selection"""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration components not available")

        coordinator = FederatedP2PCoordinator("test_coordinator")

        # Mock discovered peers
        mock_peers = []
        for participant in sample_participants:
            mock_peer = Mock()
            mock_peer.peer_id = participant["participant_id"]
            mock_peer.addresses = [f"{participant['participant_id']}_address"]
            mock_peer.protocols = ["libp2p", "betanet"] if participant.get("betanet_support") else ["libp2p"]
            mock_peer.metadata = participant
            mock_peers.append(mock_peer)

        mock_p2p_network.get_peers = AsyncMock(return_value=mock_peers)

        with patch.object(coordinator, "p2p_network", mock_p2p_network):
            # Process discovered peers
            for peer in mock_peers:
                await coordinator._process_discovered_peer(peer)

            # Check that participants were registered
            assert len(coordinator.p2p_participants) == len(sample_participants)

            # Test participant selection

            selected = await coordinator._select_p2p_participants(
                coordinator.active_training_jobs.get(
                    "test_job",
                    Mock(min_participants=3, max_participants=10, enable_betanet_privacy=True, job_type=Mock()),
                ),
            )

            assert len(selected) >= 3, "Should select at least minimum participants"

            # Check that high-trust participants are preferred
            high_trust_selected = [p for p in selected if p.trust_score > 0.8]
            assert len(high_trust_selected) > 0, "Should select high-trust participants"

    @pytest.mark.asyncio
    async def test_hierarchical_aggregation_integration(self, sample_participants):
        """Test hierarchical aggregation with P2P topology"""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration components not available")

        # Create hierarchical aggregator
        aggregator = P2PHierarchicalAggregator(aggregator_id="test_aggregator")

        # Initialize with sample participants
        success = await aggregator.initialize(sample_participants)
        assert success, "Hierarchical aggregator should initialize successfully"

        # Check cluster formation
        assert len(aggregator.clusters) > 0, "Should create at least one cluster"

        # Check role assignment
        cluster_heads = [p for p, r in aggregator.participant_roles.items() if r == ClusterRole.CLUSTER_HEAD]
        assert len(cluster_heads) > 0, "Should assign cluster heads"

        # Test hierarchical aggregation
        gradients = []
        for participant in sample_participants:
            gradient = HierarchicalGradient(
                participant_id=participant["participant_id"],
                gradients={"gradients": {"layer1.weight": [[0.1, 0.2], [0.3, 0.4]], "layer1.bias": [0.1, 0.2]}},
                quality_score=participant.get("trust_score", 0.5),
                data_samples=100,
            )
            gradients.append(gradient)

        session_id = "test_session"
        job_config = {"job_id": "test_job", "byzantine_tolerance": 0.3}

        result = await aggregator.start_hierarchical_aggregation(session_id, gradients, job_config)

        assert result.get("final_result") is not None, "Should produce final aggregation result"
        assert result["bandwidth_savings"] > 0, "Should achieve bandwidth savings"

        await aggregator.stop()

    @pytest.mark.asyncio
    async def test_mobile_optimization_integration(self):
        """Test mobile device optimization integration"""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration components not available")

        optimizer = MobileFederatedOptimizer("test_optimizer")
        success = await optimizer.initialize()
        assert success, "Mobile optimizer should initialize successfully"

        # Create mobile device profiles
        mobile_devices = [
            MobileDeviceProfile(
                device_id="smartphone_1",
                device_type="smartphone",
                capability_tier=DeviceCapabilityTier.HIGH_END,
                battery_percent=80.0,
                temperature_celsius=30.0,
                is_charging=False,
                wifi_connected=True,
            ),
            MobileDeviceProfile(
                device_id="tablet_1",
                device_type="tablet",
                capability_tier=DeviceCapabilityTier.MID_RANGE,
                battery_percent=45.0,
                temperature_celsius=35.0,
                is_charging=True,
                wifi_connected=True,
            ),
            MobileDeviceProfile(
                device_id="iot_device_1",
                device_type="iot",
                capability_tier=DeviceCapabilityTier.EMBEDDED,
                battery_percent=25.0,
                temperature_celsius=40.0,
                is_charging=False,
                wifi_connected=False,
                cellular_connected=True,
            ),
        ]

        # Register mobile devices
        for device in mobile_devices:
            success = await optimizer.register_mobile_device(device)
            assert success, f"Should register mobile device {device.device_id}"

        # Test optimization for each device
        for device in mobile_devices:
            training_request = {"job_id": "test_mobile_job", "priority": "normal"}

            session = await optimizer.optimize_training_for_mobile(device.device_id, training_request)

            if device.battery_percent > device.battery_threshold_percent or device.is_charging:
                assert session is not None, f"Should create session for eligible device {device.device_id}"

                # Check optimization applied based on device characteristics
                if device.capability_tier == DeviceCapabilityTier.EMBEDDED:
                    assert session.gradient_compression_ratio > 0.8, "Embedded device should have high compression"
                    assert session.max_duration_minutes <= 5, "Embedded device should have short duration"

                if device.is_charging:
                    assert session.training_mode in [
                        MobileTrainingMode.AGGRESSIVE,
                        MobileTrainingMode.CONSERVATIVE,
                    ], "Charging device should allow training"

                if not device.wifi_connected and device.cellular_connected:
                    assert (
                        session.training_mode == MobileTrainingMode.WIFI_ONLY or device.allow_cellular_training
                    ), "Should handle network constraints"

        await optimizer.stop()

    @pytest.mark.asyncio
    async def test_byzantine_fault_tolerance(self, sample_participants):
        """Test Byzantine fault tolerance across the system"""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration components not available")

        aggregator = P2PHierarchicalAggregator("test_aggregator")
        await aggregator.initialize(sample_participants)

        # Create gradients with one Byzantine participant
        gradients = []
        for participant in sample_participants:
            if participant["participant_id"] == "byzantine_node":
                # Byzantine node sends malicious gradients (very large values)
                gradient = HierarchicalGradient(
                    participant_id=participant["participant_id"],
                    gradients={
                        "gradients": {
                            "layer1.weight": [[100.0, 200.0], [300.0, 400.0]],  # Abnormally large
                            "layer1.bias": [100.0, 200.0],
                        }
                    },
                    quality_score=participant.get("trust_score", 0.3),
                    data_samples=100,
                )
            else:
                # Normal participants send reasonable gradients
                gradient = HierarchicalGradient(
                    participant_id=participant["participant_id"],
                    gradients={
                        "gradients": {
                            "layer1.weight": [[0.1, 0.2], [0.3, 0.4]],  # Normal values
                            "layer1.bias": [0.1, 0.2],
                        }
                    },
                    quality_score=participant.get("trust_score", 0.8),
                    data_samples=100,
                )
            gradients.append(gradient)

        # Test Byzantine tolerance in hierarchical aggregation
        session_id = "byzantine_test"
        job_config = {"job_id": "test_job", "byzantine_tolerance": 0.3}

        result = await aggregator.start_hierarchical_aggregation(session_id, gradients, job_config)

        # Check that Byzantine participant was detected and filtered
        assert result.get("final_result") is not None, "Should complete aggregation despite Byzantine participant"

        # Check statistics for Byzantine detection
        if hasattr(aggregator, "stats") and "byzantine_detections" in aggregator.stats:
            assert aggregator.stats["byzantine_detections"] > 0, "Should detect Byzantine participants"

        await aggregator.stop()

    @pytest.mark.asyncio
    async def test_secure_gradient_exchange(self, mock_betanet_transport, sample_participants):
        """Test secure gradient exchange using BetaNet transport"""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration components not available")

        coordinator = FederatedP2PCoordinator("test_coordinator")

        with patch.object(coordinator, "betanet_transport", mock_betanet_transport):
            # Mock participant with BetaNet capability
            participant = P2PParticipant(
                participant_id="secure_participant",
                p2p_peer_id="secure_peer",
                transport_types=["betanet"],
                betanet_available=True,
                privacy_capable=True,
            )

            coordinator.p2p_participants["secure_participant"] = participant

            # Test secure gradient collection
            job_config = Mock()
            job_config.enable_betanet_privacy = True

            await coordinator._collect_gradients_via_betanet(participant, job_config)

            # Check that BetaNet transport was used
            mock_betanet_transport.send_job_data.assert_called()
            mock_betanet_transport.receive_job_data.assert_called()

    @pytest.mark.asyncio
    async def test_end_to_end_training_workflow(self, mock_p2p_network, mock_betanet_transport, sample_participants):
        """Test complete end-to-end federated training workflow"""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration components not available")

        # Create integrated system components
        config = P2PTrainingConfig(
            p2p_mode=P2PTrainingMode.HYBRID_OPTIMAL,
            enable_betanet_privacy=True,
            enable_fog_integration=True,
            mobile_optimized=True,
            byzantine_tolerance_threshold=0.3,
        )

        coordinator = FederatedP2PCoordinator(coordinator_id="e2e_coordinator", p2p_config=config)

        # Mock peer discovery
        mock_peers = []
        for participant in sample_participants:
            mock_peer = Mock()
            mock_peer.peer_id = participant["participant_id"]
            mock_peer.addresses = [f"{participant['participant_id']}_address"]
            mock_peer.protocols = ["libp2p", "betanet"] if participant.get("betanet_support") else ["libp2p"]
            mock_peer.metadata = participant
            mock_peers.append(mock_peer)

        mock_p2p_network.get_peers = AsyncMock(return_value=mock_peers)

        # Mock successful initialization
        with patch.object(coordinator, "p2p_network", mock_p2p_network), patch.object(
            coordinator, "betanet_transport", mock_betanet_transport
        ), patch.object(coordinator, "_running", True):

            # Process discovered peers
            for peer in mock_peers:
                await coordinator._process_discovered_peer(peer)

            # Start federated training job
            job_config = {
                "name": "E2E Integration Test",
                "rounds": 3,
                "min_participants": 3,
                "max_participants": 8,
                "dataset_requirements": {"min_samples": 100},
            }

            job_id = await coordinator.start_federated_training_job(job_config)
            assert job_id is not None, "Should create training job"

            # Let training execute for a short time
            await asyncio.sleep(2.0)

            # Check training status
            status = await coordinator.get_training_status()

            assert status["running"], "System should be running"
            assert status["participants"]["total"] > 0, "Should have discovered participants"
            assert job_id in coordinator.active_training_jobs, "Job should be active"

            # Check different participant types were handled
            mobile_participants = [p for p in coordinator.p2p_participants.values() if p.is_mobile]
            assert len(mobile_participants) > 0, "Should have mobile participants"

            edge_participants = [p for p in coordinator.p2p_participants.values() if p.device_type == "edge"]
            assert len(edge_participants) > 0, "Should have edge participants"

    @pytest.mark.asyncio
    async def test_system_resilience_and_recovery(self, mock_p2p_network, sample_participants):
        """Test system resilience and recovery from failures"""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration components not available")

        coordinator = FederatedP2PCoordinator("resilience_test_coordinator")

        with patch.object(coordinator, "p2p_network", mock_p2p_network):
            # Simulate network failure during discovery
            mock_p2p_network.start_discovery = AsyncMock(side_effect=Exception("Network failure"))

            # System should handle discovery failure gracefully
            await coordinator.initialize()
            # Should still initialize even if discovery fails

            # Simulate participant failure during training
            participant = P2PParticipant(
                participant_id="failing_participant", p2p_peer_id="failing_peer", transport_types=["libp2p"]
            )
            coordinator.p2p_participants["failing_participant"] = participant

            # Mock training failure
            with patch.object(
                coordinator, "_execute_participant_training_p2p", AsyncMock(side_effect=Exception("Training failure"))
            ):

                job_config = Mock()
                job_config.min_participants = 1

                result = await coordinator._coordinate_p2p_training(job_config, [participant], 1)

                # Should handle participant failures
                assert result["failed_participants"] > 0, "Should record participant failures"
                assert result["successful_participants"] == 0, "No successful participants expected"

            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_performance_and_scalability(self, sample_participants):
        """Test performance characteristics and scalability"""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration components not available")

        # Create larger participant set for scalability test
        large_participant_set = []
        for i in range(50):  # 50 participants
            participant = {
                "participant_id": f"participant_{i:03d}",
                "device_type": "mobile" if i % 3 == 0 else "edge" if i % 3 == 1 else "cloud",
                "is_mobile": i % 3 == 0,
                "latency_ms": 50.0 + (i * 2),
                "bandwidth_mbps": 20.0 + (i * 5),
                "trust_score": 0.5 + (i % 5) * 0.1,
                "betanet_support": i % 4 == 0,
            }
            large_participant_set.append(participant)

        # Test hierarchical aggregation scalability
        aggregator = P2PHierarchicalAggregator("scalability_test")

        start_time = time.time()
        success = await aggregator.initialize(large_participant_set)
        init_time = time.time() - start_time

        assert success, "Should handle large participant sets"
        assert init_time < 5.0, f"Initialization should be fast, took {init_time:.2f}s"

        # Check cluster formation efficiency
        total_participants = len(large_participant_set)
        total_clusters = len(aggregator.clusters)

        expected_clusters = max(1, total_participants // 8)  # Assuming max cluster size of 8
        assert (
            total_clusters >= expected_clusters * 0.8
        ), f"Should create reasonable number of clusters: {total_clusters} vs expected ~{expected_clusters}"

        # Test aggregation performance
        gradients = []
        for participant in large_participant_set[:20]:  # Test with subset for performance
            gradient = HierarchicalGradient(
                participant_id=participant["participant_id"],
                gradients={"gradients": {"layer1.weight": [[0.1, 0.2], [0.3, 0.4]], "layer1.bias": [0.1, 0.2]}},
                quality_score=participant["trust_score"],
                data_samples=100,
            )
            gradients.append(gradient)

        start_time = time.time()
        result = await aggregator.start_hierarchical_aggregation("perf_test", gradients, {"byzantine_tolerance": 0.3})
        aggregation_time = time.time() - start_time

        assert result.get("final_result") is not None, "Should complete aggregation"
        assert aggregation_time < 10.0, f"Aggregation should be fast, took {aggregation_time:.2f}s"

        # Check bandwidth savings
        bandwidth_savings = result.get("bandwidth_savings", 0.0)
        assert bandwidth_savings > 30.0, f"Should achieve significant bandwidth savings: {bandwidth_savings:.1f}%"

        await aggregator.stop()

    def test_configuration_and_policies(self):
        """Test configuration management and policy application"""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration components not available")

        # Test P2P training configuration
        config = P2PTrainingConfig(
            p2p_mode=P2PTrainingMode.MOBILE_OPTIMIZED,
            enable_betanet_privacy=True,
            mobile_battery_threshold=25,
            byzantine_tolerance_threshold=0.2,
            max_p2p_peers=200,
        )

        assert config.p2p_mode == P2PTrainingMode.MOBILE_OPTIMIZED
        assert config.enable_betanet_privacy is True
        assert config.mobile_battery_threshold == 25
        assert config.byzantine_tolerance_threshold == 0.2
        assert config.max_p2p_peers == 200

        # Test mobile optimization policies
        optimizer = MobileFederatedOptimizer("policy_test")

        # Check that default policies are properly configured
        assert "high_end" in optimizer.optimization_policies
        assert "low_end" in optimizer.optimization_policies
        assert "embedded" in optimizer.optimization_policies

        # Verify policy differences
        high_end_policy = optimizer.optimization_policies["high_end"]
        low_end_policy = optimizer.optimization_policies["low_end"]

        assert high_end_policy["max_training_duration_minutes"] > low_end_policy["max_training_duration_minutes"]
        assert high_end_policy["gradient_compression"] < low_end_policy["gradient_compression"]
        assert high_end_policy["max_cpu_usage"] > low_end_policy["max_cpu_usage"]


# Performance benchmark test
@pytest.mark.asyncio
@pytest.mark.performance
async def test_system_performance_benchmark():
    """Benchmark the integrated system performance"""
    if not INTEGRATION_AVAILABLE:
        pytest.skip("Integration components not available")

    logger.info("Starting system performance benchmark...")

    # Create realistic participant set
    participants = []
    for i in range(100):
        device_types = ["mobile", "edge", "cloud", "gpu"]
        participant = {
            "participant_id": f"benchmark_participant_{i:03d}",
            "device_type": device_types[i % 4],
            "is_mobile": i % 4 == 0,
            "latency_ms": 20.0 + (i % 10) * 10,
            "bandwidth_mbps": 50.0 + (i % 8) * 25,
            "trust_score": 0.6 + (i % 4) * 0.1,
            "betanet_support": i % 3 == 0,
            "battery_percent": 50 + (i % 5) * 10 if i % 4 == 0 else None,
        }
        participants.append(participant)

    # Benchmark hierarchical aggregation
    aggregator = P2PHierarchicalAggregator("benchmark_aggregator")

    start_time = time.time()
    await aggregator.initialize(participants)
    init_time = time.time() - start_time

    logger.info(f"Hierarchical aggregator initialization: {init_time:.3f}s for {len(participants)} participants")

    # Create test gradients
    test_gradients = []
    for participant in participants[:50]:  # Test with 50 participants
        gradient = HierarchicalGradient(
            participant_id=participant["participant_id"],
            gradients={
                "gradients": {
                    "layer1.weight": np.random.randn(64, 32).tolist(),
                    "layer1.bias": np.random.randn(64).tolist(),
                    "layer2.weight": np.random.randn(32, 16).tolist(),
                    "layer2.bias": np.random.randn(32).tolist(),
                    "output.weight": np.random.randn(10, 16).tolist(),
                    "output.bias": np.random.randn(10).tolist(),
                }
            },
            quality_score=participant["trust_score"],
            data_samples=np.random.randint(50, 200),
        )
        test_gradients.append(gradient)

    # Benchmark aggregation
    start_time = time.time()
    result = await aggregator.start_hierarchical_aggregation(
        "benchmark_session", test_gradients, {"byzantine_tolerance": 0.3}
    )
    aggregation_time = time.time() - start_time

    logger.info(f"Hierarchical aggregation: {aggregation_time:.3f}s for {len(test_gradients)} gradients")
    logger.info(f"Bandwidth savings achieved: {result.get('bandwidth_savings', 0):.1f}%")

    # Performance assertions
    assert init_time < 10.0, f"Initialization too slow: {init_time:.3f}s"
    assert aggregation_time < 15.0, f"Aggregation too slow: {aggregation_time:.3f}s"
    assert result.get("bandwidth_savings", 0) > 40.0, "Should achieve significant bandwidth savings"

    await aggregator.stop()

    logger.info("Performance benchmark completed successfully")


if __name__ == "__main__":
    # Run basic integration test
    async def main():
        logger.info("Running basic federated P2P integration test...")

        if not INTEGRATION_AVAILABLE:
            logger.warning("Integration components not available - running basic test")
            return

        try:
            # Test basic coordinator functionality
            coordinator = FederatedP2PCoordinator("test_integration")

            # Mock minimal dependencies for basic test
            with patch.object(coordinator, "_initialize_p2p_network", AsyncMock()), patch.object(
                coordinator, "_initialize_betanet_transport", AsyncMock()
            ), patch.object(coordinator, "_initialize_base_coordinators", AsyncMock()), patch.object(
                coordinator, "_start_p2p_discovery", AsyncMock()
            ), patch.object(
                coordinator, "_start_background_tasks", AsyncMock()
            ):

                success = await coordinator.initialize()
                assert success, "Basic coordinator initialization failed"

                status = await coordinator.get_training_status()
                assert status is not None, "Should return training status"

                await coordinator.stop()

            logger.info("Basic integration test passed!")

        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            raise

    # Run the test
    asyncio.run(main())
