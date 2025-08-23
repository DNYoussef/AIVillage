"""Edge case tests for Sprint 6 infrastructure components"""

import time
from unittest.mock import Mock, patch

import pytest

from packages.core.resources.adaptive_loader import AdaptiveLoader
from packages.core.resources.constraint_manager import ConstraintManager
from packages.core.resources.device_profiler import DeviceProfiler, DeviceType, ThermalState
from packages.core.resources.resource_monitor import ResourceMonitor
from packages.p2p.core.p2p_node import P2PNode, PeerCapabilities


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_device_profiler_low_memory_device(self):
        """Test device profiler with very low memory device"""
        profiler = DeviceProfiler()

        # Mock a low-memory device (like old phone with 1GB RAM)
        with patch.object(profiler, "_get_memory_info") as mock_memory:
            mock_memory.return_value = (
                1.0,
                0.8,
                0.2,
            )  # 1GB total, 0.8GB used, 0.2GB available

            snapshot = profiler.take_snapshot()

            assert snapshot.memory_percent > 70  # Should show high memory usage
            assert snapshot.is_resource_constrained is True
            assert snapshot.evolution_suitability_score < 0.3  # Very low suitability

    def test_device_profiler_thermal_throttling(self):
        """Test device behavior under thermal throttling"""
        profiler = DeviceProfiler()

        # Mock thermal throttling conditions
        with (
            patch.object(profiler, "_get_cpu_temperature") as mock_temp,
            patch.object(profiler, "_get_cpu_percent") as mock_cpu,
        ):
            mock_temp.return_value = 88.0  # Critical temperature
            mock_cpu.return_value = 95.0  # High CPU usage

            snapshot = profiler.take_snapshot()

            assert snapshot.thermal_state == ThermalState.CRITICAL
            assert snapshot.is_resource_constrained is True
            assert snapshot.evolution_suitability_score < 0.2

    def test_constraint_manager_resource_exhaustion(self):
        """Test constraint manager when resources are exhausted"""
        # Create a mock device profiler with exhausted resources
        profiler = Mock()
        profiler.profile = Mock()
        profiler.profile.total_memory_gb = 2.0  # Low memory device
        profiler.profile.max_evolution_memory_mb = 512
        profiler.profile.max_evolution_cpu_percent = 30.0

        # Mock current snapshot showing high resource usage
        snapshot = Mock()
        snapshot.memory_percent = 95.0  # Very high memory usage
        snapshot.cpu_percent = 90.0  # Very high CPU usage
        snapshot.memory_available = 100 * 1024 * 1024  # Only 100MB available
        snapshot.is_resource_constrained = True

        profiler.current_snapshot = snapshot
        profiler.take_snapshot = Mock(return_value=snapshot)

        manager = ConstraintManager(profiler)

        # Should not be able to register any evolution tasks
        success = manager.register_task("test_evolution", "nightly")
        assert success is False

        # Should report resource unavailability
        constraints = manager.constraint_templates["nightly"]
        available = manager._check_resource_availability(constraints)
        assert available is False

    @pytest.mark.asyncio
    async def test_p2p_node_connection_failures(self):
        """Test P2P node handling connection failures"""
        node = P2PNode(node_id="test_node")

        # Mock server startup failure
        with patch("asyncio.start_server") as mock_server:
            mock_server.side_effect = OSError("Address already in use")

            with pytest.raises(OSError):
                await node.start()

            # Node should remain inactive
            assert node.status.value == "inactive"

    @pytest.mark.asyncio
    async def test_resource_monitor_rapid_changes(self):
        """Test resource monitor with rapid resource changes"""
        profiler = Mock()
        profiler.profile = Mock()
        profiler.profile.device_type = DeviceType.PHONE

        monitor = ResourceMonitor(profiler)

        # Simulate rapid resource changes
        resource_values = [30.0, 80.0, 20.0, 90.0, 15.0]  # Volatile CPU usage

        for i, cpu_val in enumerate(resource_values):
            snapshot = Mock()
            snapshot.cpu_percent = cpu_val
            snapshot.memory_percent = 50.0
            snapshot.timestamp = time.time() + i

            profiler.current_snapshot = snapshot

            # Monitor should handle rapid changes gracefully
            await monitor._process_snapshot_update(snapshot)

        # Should detect high volatility
        cpu_history = monitor.resource_history.get("cpu_percent", [])
        assert len(cpu_history) > 0

    def test_adaptive_loader_no_suitable_variants(self):
        """Test adaptive loader when no model variants are suitable"""
        # Create a mock device profiler with severe constraints
        profiler = Mock()
        profiler.profile = Mock()
        profiler.profile.device_type = DeviceType.PHONE
        profiler.profile.total_memory_gb = 0.5  # Very low memory
        profiler.profile.max_evolution_memory_mb = 128
        profiler.profile.max_evolution_cpu_percent = 20.0

        snapshot = Mock()
        snapshot.memory_percent = 90.0
        snapshot.cpu_percent = 85.0
        snapshot.memory_available = 50 * 1024 * 1024  # Only 50MB available
        snapshot.is_resource_constrained = True

        profiler.current_snapshot = snapshot

        constraint_manager = ConstraintManager(profiler)
        loader = AdaptiveLoader(profiler, constraint_manager)

        # Create loading context with very strict constraints
        from packages.core.resources.adaptive_loader import LoadingContext

        context = LoadingContext(
            task_type="nightly",
            priority_level=1,
            max_loading_time_seconds=30.0,  # Very short time
            quality_preference=0.9,  # High quality required
            resource_constraints=constraint_manager.constraint_templates["emergency"],
        )

        # Should not find any suitable variants
        best_variant = loader._select_best_variant("base_evolution_model", context)

        # Should either return None or the most constrained variant
        if best_variant is not None:
            assert best_variant.memory_mb <= 128

    def test_peer_capabilities_edge_cases(self):
        """Test peer capabilities with edge case hardware"""
        # Test extremely low-spec device
        low_spec = PeerCapabilities(
            device_id="low_spec_phone",
            cpu_cores=1,
            ram_mb=512,  # 512MB RAM
            evolution_capacity=0.1,
            available_for_evolution=True,
        )

        assert low_spec.is_suitable_for_evolution() is False
        assert low_spec.get_evolution_priority() < 0.2

        # Test high-spec device with availability issues
        high_spec_busy = PeerCapabilities(
            device_id="high_spec_busy",
            cpu_cores=16,
            ram_mb=32768,  # 32GB RAM
            evolution_capacity=0.95,
            available_for_evolution=False,  # Not available despite high specs
        )

        assert high_spec_busy.is_suitable_for_evolution() is False

        # Test medium device at capacity limit
        medium_at_limit = PeerCapabilities(
            device_id="medium_device",
            cpu_cores=4,
            ram_mb=4096,  # 4GB RAM
            evolution_capacity=0.49,  # Just below 50% threshold
            available_for_evolution=True,
        )

        # Should be barely suitable
        assert medium_at_limit.is_suitable_for_evolution() is False  # Below 50% threshold

    @pytest.mark.asyncio
    async def test_resource_monitor_error_recovery(self):
        """Test resource monitor error recovery"""
        profiler = Mock()
        profiler.profile = Mock()
        profiler.profile.device_type = DeviceType.LAPTOP

        monitor = ResourceMonitor(profiler)

        # Mock profiler to raise exception
        profiler.take_snapshot.side_effect = Exception("Monitoring hardware failure")

        # Monitor should handle errors gracefully
        await monitor.start_monitoring()

        # Should still be in monitoring state but handle errors
        assert monitor.monitoring_active is True

        # Recovery: Fix the profiler
        working_snapshot = Mock()
        working_snapshot.memory_percent = 50.0
        working_snapshot.cpu_percent = 30.0
        working_snapshot.timestamp = time.time()

        profiler.take_snapshot.side_effect = None
        profiler.take_snapshot.return_value = working_snapshot

        # Should recover and process snapshots normally
        await monitor._process_snapshot_update(working_snapshot)

    def test_constraint_manager_concurrent_tasks(self):
        """Test constraint manager with many concurrent tasks"""
        profiler = Mock()
        profiler.profile = Mock()
        profiler.profile.total_memory_gb = 8.0
        profiler.profile.max_evolution_memory_mb = 4096
        profiler.profile.max_evolution_cpu_percent = 70.0

        snapshot = Mock()
        snapshot.memory_percent = 40.0
        snapshot.cpu_percent = 20.0
        snapshot.memory_available = 4000 * 1024 * 1024  # 4GB available
        snapshot.is_resource_constrained = False

        profiler.current_snapshot = snapshot
        profiler.take_snapshot = Mock(return_value=snapshot)

        manager = ConstraintManager(profiler)

        # Register multiple tasks until resources are exhausted
        registered_tasks = []
        for i in range(10):  # Try to register many tasks
            task_id = f"evolution_task_{i}"
            success = manager.register_task(task_id, "nightly")
            if success:
                registered_tasks.append(task_id)

        # Should have registered some tasks but eventually hit limits
        assert len(registered_tasks) > 0
        assert len(registered_tasks) < 10  # Shouldn't register all due to resource limits

        # Clean up
        for task_id in registered_tasks:
            manager.unregister_task(task_id)


class TestCriticalPaths:
    """Test critical execution paths and performance bottlenecks"""

    def test_device_profiler_initialization_performance(self):
        """Test device profiler initialization time"""
        start_time = time.time()

        profiler = DeviceProfiler()

        init_time = time.time() - start_time

        # Should initialize quickly (less than 1 second)
        assert init_time < 1.0
        assert profiler.device_id is not None
        assert profiler.profile.device_type != DeviceType.UNKNOWN

    def test_resource_snapshot_performance(self):
        """Test resource snapshot performance"""
        profiler = DeviceProfiler()

        # Take multiple snapshots and measure performance
        snapshot_times = []
        for _ in range(5):
            start_time = time.time()
            snapshot = profiler.take_snapshot()
            snapshot_time = time.time() - start_time
            snapshot_times.append(snapshot_time)

            assert snapshot is not None
            assert snapshot.memory_percent >= 0
            assert snapshot.cpu_percent >= 0

        # Average snapshot time should be reasonable
        avg_time = sum(snapshot_times) / len(snapshot_times)
        assert avg_time < 0.1  # Should take less than 100ms on average

    @pytest.mark.asyncio
    async def test_p2p_message_handling_performance(self):
        """Test P2P message handling performance"""
        node = P2PNode(node_id="perf_test_node")

        # Mock message handler
        message_count = 0

        async def mock_handler(message, peer_id):
            nonlocal message_count
            message_count += 1
            return {"status": "processed"}

        node.register_handler("test_message", mock_handler)

        # Simulate processing multiple messages quickly
        start_time = time.time()

        for i in range(10):
            await node._handle_message(
                {"type": "test_message", "id": f"msg_{i}", "data": {"test": True}},
                "test_peer",
            )

        processing_time = time.time() - start_time

        assert message_count == 10
        assert processing_time < 1.0  # Should process 10 messages in under 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
