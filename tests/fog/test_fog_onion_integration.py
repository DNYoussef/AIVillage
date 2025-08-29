"""
Test Fog Computing Onion Routing Integration

Tests the complete integration of onion routing privacy layer
with fog computing task distribution and service hosting.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, UTC, timedelta

# Import fog components
from infrastructure.fog.integration.fog_coordinator import FogCoordinator
from infrastructure.fog.integration.fog_onion_coordinator import (
    FogOnionCoordinator,
    PrivacyAwareTask,
    PrivacyAwareService,
    PrivacyLevel,
    TaskPrivacyPolicy,
)
from infrastructure.fog.privacy.onion_routing import NodeType, OnionRouter, OnionCircuit
from infrastructure.fog.privacy.mixnet_integration import NymMixnetClient


class TestFogOnionIntegration:
    """Test fog computing onion routing integration."""

    @pytest.fixture
    async def fog_coordinator(self):
        """Create a fog coordinator for testing."""
        coordinator = FogCoordinator(
            node_id="test-fog-node",
            enable_harvesting=False,
            enable_onion_routing=True,
            enable_marketplace=False,
            enable_tokens=False,
        )

        # Mock initialization methods to avoid external dependencies
        with patch.object(coordinator, "_initialize_onion_router"):
            with patch.object(coordinator, "_initialize_onion_coordinator"):
                await coordinator.start()

        yield coordinator

        await coordinator.stop()

    @pytest.fixture
    async def onion_coordinator(self, fog_coordinator):
        """Create an onion coordinator for testing."""
        coordinator = FogOnionCoordinator(
            node_id="test-onion-coord",
            fog_coordinator=fog_coordinator,
            enable_mixnet=True,
            default_privacy_level=PrivacyLevel.PRIVATE,
            max_circuits=10,
        )

        # Mock onion router and mixnet client
        mock_onion_router = Mock()
        mock_mixnet_client = Mock()

        coordinator.onion_router = mock_onion_router
        coordinator.mixnet_client = mock_mixnet_client
        
        # Also set the fog coordinator's onion router to our mock
        # to prevent it from being overridden during start()
        fog_coordinator.onion_router = mock_onion_router

        # Mock consensus and circuit building
        mock_onion_router.consensus = {f"node_{i}": Mock() for i in range(10)}
        
        # Create mock circuit
        mock_circuit_obj = Mock()
        mock_circuit_obj.circuit_id = "test-circuit-123"
        mock_circuit_obj.state.value = "established"
        mock_circuit_obj.hops = [Mock(), Mock(), Mock()]
        mock_circuit_obj.bytes_sent = 0
        mock_circuit_obj.bytes_received = 0
        
        mock_onion_router.build_circuit = AsyncMock(return_value=mock_circuit_obj)
        
        # Mock the circuits registry in the onion router
        mock_onion_router.circuits = {"test-circuit-123": mock_circuit_obj}
        mock_onion_router.send_data = AsyncMock(return_value=True)
        mock_onion_router.close_circuit = AsyncMock(return_value=True)  # For teardown
        mock_onion_router.create_hidden_service = AsyncMock(
            return_value=Mock(
                service_id="test-service-456",
                onion_address="abc123def456ghi789.fog",
            )
        )

        mock_mixnet_client.send_anonymous_message = AsyncMock(return_value="packet-789")
        mock_mixnet_client.start = AsyncMock(return_value=True)  # For start
        mock_mixnet_client.stop = AsyncMock(return_value=True)   # For teardown
        mock_mixnet_client.get_mixnet_stats = AsyncMock(
            return_value={
                "client_id": "test-mixnet",
                "packets_sent": 10,
                "topology": {"entry": 5, "mix": 15, "exit": 8},
            }
        )

        # Pre-populate circuit pools with mock circuits (use same circuit object)
        for privacy_level in PrivacyLevel:
            if privacy_level != PrivacyLevel.PUBLIC:
                coordinator.circuit_pools[privacy_level] = [mock_circuit_obj]

        # Mock the circuit pool initialization to avoid actual circuit building
        with patch.object(coordinator, '_initialize_circuit_pools') as mock_init_pools:
            mock_init_pools.return_value = None
            with patch.object(coordinator, '_start_background_tasks') as mock_bg_tasks:
                mock_bg_tasks.return_value = None
                await coordinator.start()
        
        yield coordinator
        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_privacy_aware_task_submission(self, onion_coordinator):
        """Test submitting privacy-aware tasks."""
        task = PrivacyAwareTask(
            task_id="test-task-001",
            privacy_level=PrivacyLevel.PRIVATE,
            task_data=b"sensitive computation data",
            compute_requirements={
                "cpu_cores": 2,
                "memory_gb": 4,
                "duration_minutes": 30,
            },
            client_id="test-client-123",
            require_onion_circuit=True,
            min_circuit_hops=3,
        )

        success = await onion_coordinator.submit_privacy_aware_task(task)

        assert success is True
        assert task.task_id in onion_coordinator.privacy_tasks
        assert task.task_id in onion_coordinator.task_circuits

        # Verify circuit was used (send_data should be called for routing)
        onion_coordinator.onion_router.send_data.assert_called()
        
        # Verify task is properly tracked
        assert onion_coordinator.privacy_tasks[task.task_id] == task

    @pytest.mark.asyncio
    async def test_high_privacy_task_with_mixnet(self, onion_coordinator):
        """Test high privacy tasks using mixnet."""
        task = PrivacyAwareTask(
            task_id="test-task-secret",
            privacy_level=PrivacyLevel.SECRET,
            task_data=b"highly sensitive data",
            compute_requirements={
                "cpu_cores": 4,
                "memory_gb": 8,
                "duration_minutes": 60,
            },
            client_id="test-client-456",
            require_mixnet=True,
        )

        success = await onion_coordinator.submit_privacy_aware_task(task)

        assert success is True
        assert task.task_id in onion_coordinator.privacy_tasks

        # Verify mixnet was used for maximum privacy
        onion_coordinator.mixnet_client.send_anonymous_message.assert_called()

    @pytest.mark.asyncio
    async def test_create_privacy_aware_service(self, onion_coordinator):
        """Test creating privacy-aware fog services."""
        service = await onion_coordinator.create_privacy_aware_service(
            service_id="private-api-service",
            service_type="api",
            privacy_level=PrivacyLevel.PRIVATE,
            ports={8080: 8080, 8443: 8443},
            authentication_required=True,
        )

        assert service is not None
        assert service.service_id == "private-api-service"
        assert service.privacy_level == PrivacyLevel.PRIVATE
        assert service.onion_address is not None
        assert service.onion_address.endswith(".fog")
        assert service.ports == {8080: 8080, 8443: 8443}

        # Verify hidden service was created
        onion_coordinator.onion_router.create_hidden_service.assert_called_with(
            ports={8080: 8080, 8443: 8443},
            descriptor_cookie=None,
        )

    @pytest.mark.asyncio
    async def test_confidential_service_with_dedicated_circuit(self, onion_coordinator):
        """Test creating confidential services with dedicated circuits."""
        service = await onion_coordinator.create_privacy_aware_service(
            service_id="confidential-db-service",
            service_type="database",
            privacy_level=PrivacyLevel.CONFIDENTIAL,
            ports={5432: 5432},
        )

        assert service is not None
        assert service.privacy_level == PrivacyLevel.CONFIDENTIAL
        assert service.circuit_id is not None
        assert service.service_id in onion_coordinator.service_circuits

        # Should have built a dedicated circuit for confidential service
        assert onion_coordinator.onion_router.build_circuit.call_count >= 1

    @pytest.mark.asyncio
    async def test_private_gossip_communication(self, onion_coordinator):
        """Test private gossip between fog nodes."""
        message = b"node status update: healthy, 85% cpu, 12GB available memory"

        # Test private level gossip (onion routing)
        success = await onion_coordinator.send_private_gossip(
            recipient_id="fog-node-789",
            message=message,
            privacy_level=PrivacyLevel.PRIVATE,
        )

        assert success is True
        onion_coordinator.onion_router.send_data.assert_called()

        # Test confidential level gossip (mixnet)
        success = await onion_coordinator.send_private_gossip(
            recipient_id="fog-node-999",
            message=message,
            privacy_level=PrivacyLevel.CONFIDENTIAL,
        )

        assert success is True
        onion_coordinator.mixnet_client.send_anonymous_message.assert_called_with(
            destination="fog-node-999",
            message=message,
        )

    @pytest.mark.asyncio
    async def test_circuit_pool_management(self, onion_coordinator):
        """Test circuit pool management for different privacy levels."""
        # Submit tasks requiring different privacy levels
        tasks = [
            PrivacyAwareTask(
                task_id=f"task-{i}",
                privacy_level=PrivacyLevel.PRIVATE,
                task_data=f"task data {i}".encode(),
                compute_requirements={"cpu_cores": 1},
                client_id=f"client-{i}",
            )
            for i in range(5)
        ]

        # Submit all tasks
        for task in tasks:
            await onion_coordinator.submit_privacy_aware_task(task)

        # Verify circuit pools are being managed
        private_circuits = onion_coordinator.circuit_pools[PrivacyLevel.PRIVATE]
        assert len(private_circuits) > 0

        # Verify stats tracking
        stats = await onion_coordinator.get_coordinator_stats()
        assert stats["privacy_stats"]["tasks_processed"] == 5
        assert stats["privacy_tasks"] == 5

    @pytest.mark.asyncio
    async def test_fog_coordinator_privacy_integration(self, fog_coordinator):
        """Test privacy integration through fog coordinator API."""
        # Mock the onion coordinator
        mock_onion_coordinator = AsyncMock()
        fog_coordinator.onion_coordinator = mock_onion_coordinator

        # Test privacy task submission
        task_data = {
            "task_id": "integration-test-001",
            "privacy_level": "PRIVATE",
            "task_data": "test computation",
            "compute_requirements": {"cpu_cores": 2},
            "client_id": "integration-client",
        }

        mock_onion_coordinator.submit_privacy_aware_task = AsyncMock(return_value=True)

        result = await fog_coordinator.process_fog_request("submit_privacy_task", task_data)

        assert result["success"] is True
        assert result["task_id"] == "integration-test-001"
        mock_onion_coordinator.submit_privacy_aware_task.assert_called_once()

        # Test privacy service creation
        service_data = {
            "service_id": "integration-service-001",
            "service_type": "web",
            "privacy_level": "CONFIDENTIAL",
            "ports": {80: 8080},
        }

        mock_service = Mock()
        mock_service.service_id = "integration-service-001"
        mock_service.onion_address = "test123.fog"
        mock_service.privacy_level = PrivacyLevel.CONFIDENTIAL

        mock_onion_coordinator.create_privacy_aware_service = AsyncMock(return_value=mock_service)

        result = await fog_coordinator.process_fog_request("create_privacy_service", service_data)

        assert result["success"] is True
        assert result["service_id"] == "integration-service-001"
        assert result["onion_address"] == "test123.fog"
        assert result["privacy_level"] == "confidential"

        # Test private gossip
        gossip_data = {
            "recipient_id": "fog-node-456",
            "message": "network health status",
            "privacy_level": "PRIVATE",
        }

        mock_onion_coordinator.send_private_gossip = AsyncMock(return_value=True)

        result = await fog_coordinator.process_fog_request("send_private_gossip", gossip_data)

        assert result["success"] is True
        mock_onion_coordinator.send_private_gossip.assert_called_once()

    @pytest.mark.asyncio
    async def test_privacy_service_discovery_by_onion_address(self, onion_coordinator):
        """Test service discovery using onion addresses."""
        # Create a service
        service = await onion_coordinator.create_privacy_aware_service(
            service_id="discoverable-service",
            service_type="api",
            privacy_level=PrivacyLevel.PRIVATE,
            ports={8080: 8080},
        )

        # Find service by onion address
        found_service = await onion_coordinator.get_service_by_onion_address(service.onion_address)

        assert found_service is not None
        assert found_service.service_id == "discoverable-service"
        assert found_service.onion_address == service.onion_address

        # Test with non-existent address
        not_found = await onion_coordinator.get_service_by_onion_address("nonexistent.fog")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_privacy_level_validation(self, onion_coordinator):
        """Test privacy level validation and requirements."""
        # Test task with invalid requirements
        invalid_task = PrivacyAwareTask(
            task_id="invalid-task",
            privacy_level=PrivacyLevel.CONFIDENTIAL,
            task_data=b"test data",
            compute_requirements={},
            client_id="test-client",
            require_mixnet=True,  # Required for CONFIDENTIAL
        )

        # Mock mixnet not available
        onion_coordinator.enable_mixnet = False
        onion_coordinator.mixnet_client = None

        success = await onion_coordinator.submit_privacy_aware_task(invalid_task)
        assert success is False  # Should fail validation

        # Re-enable mixnet and try again
        onion_coordinator.enable_mixnet = True
        onion_coordinator.mixnet_client = Mock()
        onion_coordinator.mixnet_client.send_anonymous_message = AsyncMock(return_value="packet-123")

        success = await onion_coordinator.submit_privacy_aware_task(invalid_task)
        assert success is True  # Should pass validation now

    @pytest.mark.asyncio
    async def test_comprehensive_privacy_statistics(self, onion_coordinator):
        """Test comprehensive privacy and performance statistics."""
        # Create various privacy-aware components
        await onion_coordinator.submit_privacy_aware_task(
            PrivacyAwareTask(
                task_id="stats-task-1",
                privacy_level=PrivacyLevel.PRIVATE,
                task_data=b"data1",
                compute_requirements={},
                client_id="client1",
            )
        )

        await onion_coordinator.create_privacy_aware_service(
            service_id="stats-service-1",
            service_type="web",
            privacy_level=PrivacyLevel.CONFIDENTIAL,
            ports={80: 8080},
        )

        # Get comprehensive stats
        stats = await onion_coordinator.get_coordinator_stats()

        # Verify structure and data
        assert "node_id" in stats
        assert "running" in stats
        assert "privacy_stats" in stats
        assert "privacy_tasks" in stats
        assert "privacy_services" in stats
        assert "circuit_pools" in stats
        assert "onion_routing" in stats
        assert "mixnet" in stats

        # Verify metrics
        assert stats["privacy_tasks"] >= 1
        assert stats["privacy_services"] >= 1
        assert stats["privacy_stats"]["tasks_processed"] >= 1
        assert stats["privacy_stats"]["services_hosted"] >= 1


class TestFogOnionPerformance:
    """Test performance aspects of fog onion integration."""

    @pytest.mark.asyncio
    async def test_concurrent_privacy_task_processing(self):
        """Test processing multiple privacy tasks concurrently."""
        coordinator = FogOnionCoordinator(
            node_id="perf-test-node",
            fog_coordinator=Mock(),
            enable_mixnet=True,
            max_circuits=50,
        )

        # Mock dependencies
        coordinator.onion_router = Mock()
        coordinator.onion_router.consensus = {f"node_{i}": Mock() for i in range(20)}
        coordinator.onion_router.build_circuit = AsyncMock(
            return_value=Mock(
                circuit_id=f"circuit-{asyncio.current_task().get_name()}",
                state=Mock(value="established"),
                hops=[Mock(), Mock(), Mock()],
                bytes_sent=0,
                bytes_received=0,
            )
        )
        coordinator.onion_router.send_data = AsyncMock(return_value=True)
        coordinator._running = True

        # Pre-build circuit pools
        for level in PrivacyLevel:
            if level != PrivacyLevel.PUBLIC:
                for i in range(3):
                    circuit = Mock()
                    circuit.circuit_id = f"pool-{level.value}-{i}"
                    circuit.state = Mock(value="established")
                    circuit.bytes_sent = 0
                    circuit.bytes_received = 0
                    coordinator.circuit_pools[level].append(circuit)

        # Create multiple tasks
        tasks = []
        for i in range(20):
            privacy_task = PrivacyAwareTask(
                task_id=f"concurrent-task-{i}",
                privacy_level=PrivacyLevel.PRIVATE,
                task_data=f"concurrent data {i}".encode(),
                compute_requirements={"cpu_cores": 1},
                client_id=f"concurrent-client-{i}",
            )
            task_coro = coordinator.submit_privacy_aware_task(privacy_task)
            tasks.append(task_coro)

        # Submit all tasks concurrently
        start_time = datetime.now(UTC)
        results = await asyncio.gather(*tasks)
        end_time = datetime.now(UTC)

        # Verify all tasks succeeded
        assert all(results)
        assert len(coordinator.privacy_tasks) == 20

        # Verify reasonable performance (should complete within 5 seconds)
        duration = (end_time - start_time).total_seconds()
        assert duration < 5.0

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_circuit_reuse_and_rotation(self):
        """Test circuit reuse efficiency and rotation."""
        coordinator = FogOnionCoordinator(
            node_id="rotation-test-node",
            fog_coordinator=Mock(),
            enable_mixnet=False,
            max_circuits=5,
        )

        # Mock onion router
        coordinator.onion_router = Mock()
        coordinator.onion_router.consensus = {f"node_{i}": Mock() for i in range(10)}

        # Track circuit creation calls
        circuit_counter = 0

        def mock_build_circuit(*args, **kwargs):
            nonlocal circuit_counter
            circuit_counter += 1
            return Mock(
                circuit_id=f"circuit-{circuit_counter}",
                state=Mock(value="established"),
                hops=[Mock(), Mock(), Mock()],
                bytes_sent=0,
                bytes_received=0,
                created_at=datetime.now(UTC),
            )

        coordinator.onion_router.build_circuit = AsyncMock(side_effect=mock_build_circuit)
        coordinator.onion_router.send_data = AsyncMock(return_value=True)
        coordinator.onion_router.rotate_circuits = AsyncMock(return_value=0)
        coordinator._running = True

        # Submit multiple tasks to test circuit reuse
        for i in range(10):
            privacy_task = PrivacyAwareTask(
                task_id=f"reuse-task-{i}",
                privacy_level=PrivacyLevel.PRIVATE,
                task_data=f"reuse data {i}".encode(),
                compute_requirements={},
                client_id=f"reuse-client-{i}",
            )
            await coordinator.submit_privacy_aware_task(privacy_task)

        # Verify circuits were reused (should not create 10 new circuits)
        private_circuits = coordinator.circuit_pools[PrivacyLevel.PRIVATE]
        assert len(private_circuits) <= 5  # Should reuse circuits within pool limit

        await coordinator.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
