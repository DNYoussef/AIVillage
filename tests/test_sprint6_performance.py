"""Performance benchmark tests for Sprint 6 infrastructure"""

import asyncio
import statistics
import time
from unittest.mock import Mock, patch

import pytest

from src.core.resources.adaptive_loader import AdaptiveLoader
from src.core.resources.constraint_manager import ConstraintManager
from src.core.resources.device_profiler import DeviceProfiler, DeviceType
from src.core.resources.resource_monitor import ResourceMonitor
from src.production.communications.p2p.p2p_node import P2PNode


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests for Sprint 6 components"""

    def test_device_profiler_snapshot_benchmark(self, benchmark):
        """Benchmark device profiler snapshot performance"""
        profiler = DeviceProfiler()

        def take_snapshot():
            return profiler.take_snapshot()

        # Benchmark the snapshot operation
        result = benchmark(take_snapshot)

        # Assertions about the result
        assert result is not None
        assert result.memory_percent >= 0
        assert result.cpu_percent >= 0

        # Performance assertions (adjust based on expected performance)
        stats = benchmark.stats
        assert stats.mean < 0.1  # Should take less than 100ms on average

    def test_constraint_manager_task_registration_benchmark(self, benchmark):
        """Benchmark constraint manager task registration"""
        # Setup mock profiler
        profiler = Mock()
        profiler.profile = Mock()
        profiler.profile.total_memory_gb = 8.0
        profiler.profile.max_evolution_memory_mb = 4096
        profiler.profile.max_evolution_cpu_percent = 70.0

        snapshot = Mock()
        snapshot.memory_percent = 50.0
        snapshot.cpu_percent = 30.0
        snapshot.memory_available = 4000 * 1024 * 1024
        snapshot.is_resource_constrained = False

        profiler.current_snapshot = snapshot
        profiler.take_snapshot = Mock(return_value=snapshot)

        manager = ConstraintManager(profiler)

        task_counter = 0

        def register_task():
            nonlocal task_counter
            task_id = f"benchmark_task_{task_counter}"
            task_counter += 1
            success = manager.register_task(task_id, "nightly")
            if success:
                manager.unregister_task(task_id)  # Clean up immediately
            return success

        # Benchmark task registration
        result = benchmark(register_task)

        assert isinstance(result, bool)

        # Should be very fast
        stats = benchmark.stats
        assert stats.mean < 0.01  # Should take less than 10ms

    @pytest.mark.asyncio
    async def test_p2p_node_startup_benchmark(self, benchmark):
        """Benchmark P2P node startup performance"""

        async def startup_node():
            node = P2PNode(node_id="benchmark_node")

            # Mock server to avoid actual network binding
            with patch("asyncio.start_server") as mock_server:
                mock_server_obj = Mock()
                mock_server_obj.sockets = [Mock()]
                mock_server_obj.sockets[0].getsockname.return_value = (
                    "localhost",
                    9000,
                )
                mock_server.return_value = mock_server_obj

                start_time = time.time()
                await node.start()
                startup_time = time.time() - start_time

                await node.stop()
                return startup_time

        # Run the benchmark
        startup_time = await startup_node()

        # Should start up quickly
        assert startup_time < 1.0  # Less than 1 second

    def test_adaptive_loader_variant_selection_benchmark(self, benchmark):
        """Benchmark adaptive loader variant selection"""
        # Setup mock components
        profiler = Mock()
        profiler.profile = Mock()
        profiler.profile.device_type = DeviceType.LAPTOP
        profiler.profile.total_memory_gb = 8.0
        profiler.profile.max_evolution_memory_mb = 4096
        profiler.profile.max_evolution_cpu_percent = 70.0

        snapshot = Mock()
        snapshot.memory_percent = 40.0
        snapshot.cpu_percent = 30.0
        snapshot.memory_available = 4000 * 1024 * 1024
        snapshot.is_resource_constrained = False

        profiler.current_snapshot = snapshot

        constraint_manager = ConstraintManager(profiler)
        loader = AdaptiveLoader(profiler, constraint_manager)

        # Create loading context
        from src.core.resources.adaptive_loader import LoadingContext

        context = LoadingContext(
            task_type="nightly",
            priority_level=2,
            max_loading_time_seconds=120.0,
            quality_preference=0.7,
            resource_constraints=constraint_manager.default_constraints,
        )

        def select_variant():
            return loader._select_best_variant("base_evolution_model", context)

        # Benchmark variant selection
        result = benchmark(select_variant)

        # Should find a variant or return None
        assert result is None or hasattr(result, "memory_mb")

        # Should be fast
        stats = benchmark.stats
        assert stats.mean < 0.05  # Should take less than 50ms


class TestScalabilityTests:
    """Test system behavior under various load conditions"""

    def test_resource_monitor_high_frequency_updates(self):
        """Test resource monitor with high-frequency updates"""
        profiler = Mock()
        profiler.profile = Mock()
        profiler.profile.device_type = DeviceType.DESKTOP

        monitor = ResourceMonitor(profiler)

        # Generate many rapid snapshots
        snapshots = []
        start_time = time.time()

        for i in range(100):  # 100 rapid updates
            snapshot = Mock()
            snapshot.memory_percent = 50.0 + (i % 20)  # Varying memory
            snapshot.cpu_percent = 30.0 + (i % 30)  # Varying CPU
            snapshot.timestamp = start_time + (i * 0.01)  # 10ms intervals

            snapshots.append(snapshot)
            profiler.current_snapshot = snapshot

        # Process all snapshots and measure performance
        processing_start = time.time()

        for snapshot in snapshots:
            asyncio.run(monitor._process_snapshot_update(snapshot))

        processing_time = time.time() - processing_start

        # Should handle high-frequency updates efficiently
        assert processing_time < 2.0  # Process 100 updates in under 2 seconds

        # Should maintain reasonable history size
        cpu_history = monitor.resource_history.get("cpu_percent", [])
        assert len(cpu_history) <= monitor.max_history_size

    def test_constraint_manager_many_concurrent_tasks(self):
        """Test constraint manager with many concurrent evolution tasks"""
        profiler = Mock()
        profiler.profile = Mock()
        profiler.profile.total_memory_gb = 16.0  # High-memory system
        profiler.profile.max_evolution_memory_mb = 8192
        profiler.profile.max_evolution_cpu_percent = 80.0

        snapshot = Mock()
        snapshot.memory_percent = 30.0  # Low usage initially
        snapshot.cpu_percent = 20.0
        snapshot.memory_available = 10000 * 1024 * 1024  # 10GB available
        snapshot.is_resource_constrained = False

        profiler.current_snapshot = snapshot
        profiler.take_snapshot = Mock(return_value=snapshot)

        manager = ConstraintManager(profiler)

        # Try to register many tasks
        registered_tasks = []
        registration_times = []

        for i in range(50):  # Try 50 tasks
            task_id = f"scale_test_task_{i}"

            start_time = time.time()
            success = manager.register_task(task_id, "lightweight")  # Use lightweight constraints
            reg_time = time.time() - start_time

            registration_times.append(reg_time)

            if success:
                registered_tasks.append(task_id)
            else:
                break  # Stop when resources are exhausted

        # Should have registered multiple tasks
        assert len(registered_tasks) > 5  # At least some tasks should register

        # Registration should remain fast even with many tasks
        avg_reg_time = statistics.mean(registration_times)
        assert avg_reg_time < 0.02  # Should stay under 20ms per registration

        # Cleanup
        for task_id in registered_tasks:
            manager.unregister_task(task_id)

    @pytest.mark.asyncio
    async def test_p2p_node_message_throughput(self):
        """Test P2P node message handling throughput"""
        node = P2PNode(node_id="throughput_test_node")

        # Message processing counters
        processed_messages = []
        processing_times = []

        async def benchmark_handler(message, peer_id):
            start_time = time.time()
            # Simulate some processing
            await asyncio.sleep(0.001)  # 1ms processing time
            process_time = time.time() - start_time

            processed_messages.append(message["id"])
            processing_times.append(process_time)

            return {"status": "processed", "message_id": message["id"]}

        node.register_handler("benchmark_message", benchmark_handler)

        # Send many messages rapidly
        message_count = 20
        send_start = time.time()

        # Send messages concurrently
        tasks = []
        for i in range(message_count):
            message = {
                "type": "benchmark_message",
                "id": f"msg_{i}",
                "data": {"index": i, "timestamp": time.time()},
            }

            task = asyncio.create_task(node._handle_message(message, "benchmark_peer"))
            tasks.append(task)

        # Wait for all messages to be processed
        await asyncio.gather(*tasks)

        total_time = time.time() - send_start

        # All messages should be processed
        assert len(processed_messages) == message_count

        # Calculate throughput
        throughput = message_count / total_time
        assert throughput > 10  # Should handle at least 10 messages per second

        # Average processing time should be reasonable
        avg_processing_time = statistics.mean(processing_times)
        assert avg_processing_time < 0.1  # Should process each message in under 100ms

    def test_adaptive_loader_memory_usage_scaling(self):
        """Test adaptive loader memory usage with different device configurations"""
        device_configs = [
            # (total_memory_gb, max_evolution_mb, expected_variants)
            (1.0, 256, 1),  # Very low memory device
            (4.0, 1024, 2),  # Low memory device
            (8.0, 2048, 3),  # Medium memory device
            (16.0, 4096, 4),  # High memory device
        ]

        for total_memory, max_evolution_mb, expected_min_variants in device_configs:
            profiler = Mock()
            profiler.profile = Mock()
            profiler.profile.device_type = DeviceType.LAPTOP
            profiler.profile.total_memory_gb = total_memory
            profiler.profile.max_evolution_memory_mb = max_evolution_mb
            profiler.profile.max_evolution_cpu_percent = 70.0

            snapshot = Mock()
            snapshot.memory_percent = 40.0
            snapshot.cpu_percent = 30.0
            snapshot.memory_available = int(total_memory * 1024 * 1024 * 1024 * 0.6)  # 60% available
            snapshot.is_resource_constrained = False

            profiler.current_snapshot = snapshot

            constraint_manager = ConstraintManager(profiler)
            loader = AdaptiveLoader(profiler, constraint_manager)

            # Count available variants for this device
            variants = loader.model_variants.get("base_evolution_model", [])
            suitable_variants = []

            for variant in variants:
                if variant.memory_mb <= max_evolution_mb:
                    suitable_variants.append(variant)

            # Should have appropriate number of variants for device class
            assert (
                len(suitable_variants) >= expected_min_variants
            ), f"Device with {total_memory}GB should have at least {expected_min_variants} variants, got {len(suitable_variants)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
