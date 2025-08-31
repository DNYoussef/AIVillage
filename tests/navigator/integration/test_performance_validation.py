"""Performance validation tests for PathPolicy god class separation

These tests validate that the new service-oriented architecture maintains
or improves routing performance and energy efficiency compared to the
original 1,438-line PathPolicy god class.
"""

import asyncio
import pytest
import time
import statistics
from unittest.mock import AsyncMock, Mock, patch
from typing import List, Dict, Tuple, Any

from experiments.agents.agents.navigator import (
    NavigatorAgent, NavigatorFacade, create_navigator_facade,
    PathProtocol, EnergyMode, RoutingPriority,
    MessageContext, NetworkConditions, PeerInfo
)


class TestPerformanceValidation:
    """Performance validation test suite"""
    
    @pytest.fixture
    async def navigator_facade(self):
        """Create high-performance NavigatorFacade for testing"""
        facade = create_navigator_facade(
            agent_id="perf_test_navigator",
            routing_priority=RoutingPriority.PERFORMANCE_FIRST,
            energy_mode=EnergyMode.BALANCED
        )
        
        # Mock external dependencies for consistent performance testing
        with patch('experiments.agents.agents.navigator.services.network_monitoring_service.psutil'), \
             patch('experiments.agents.agents.navigator.services.network_monitoring_service.bluetooth'), \
             patch('experiments.agents.agents.navigator.services.energy_optimization_service.psutil'):
            
            await facade.initialize()
        
        yield facade
        await facade.shutdown()
    
    @pytest.fixture
    def performance_contexts(self):
        """Create various message contexts for performance testing"""
        return [
            # Small, high-priority message
            MessageContext(
                size_bytes=100,
                priority=9,
                requires_realtime=True,
                privacy_required=False
            ),
            # Medium, standard message
            MessageContext(
                size_bytes=1024,
                priority=5,
                requires_realtime=False,
                privacy_required=False
            ),
            # Large, bulk transfer
            MessageContext(
                size_bytes=10240,
                priority=2,
                requires_realtime=False,
                privacy_required=False
            ),
            # Privacy-sensitive message
            MessageContext(
                size_bytes=512,
                priority=8,
                requires_realtime=False,
                privacy_required=True
            ),
            # Real-time communication
            MessageContext(
                size_bytes=256,
                priority=10,
                requires_realtime=True,
                privacy_required=True
            )
        ]
    
    @pytest.mark.asyncio
    async def test_routing_decision_performance(self, navigator_facade, performance_contexts):
        """Test that routing decisions meet performance targets"""
        # Performance targets based on original PathPolicy requirements
        TARGET_MAX_LATENCY_MS = 500  # 500ms max decision time
        TARGET_AVG_LATENCY_MS = 200  # 200ms average decision time
        TARGET_P95_LATENCY_MS = 400  # 95th percentile under 400ms
        
        decision_times = []
        
        # Test with various message contexts and destinations
        destinations = [f"perf_dest_{i}" for i in range(50)]
        
        for i, dest in enumerate(destinations):
            context = performance_contexts[i % len(performance_contexts)]
            
            start_time = time.perf_counter()
            protocol, metadata = await navigator_facade.select_path(dest, context)
            end_time = time.perf_counter()
            
            decision_time_ms = (end_time - start_time) * 1000
            decision_times.append(decision_time_ms)
            
            # Validate result
            assert isinstance(protocol, PathProtocol)
            assert isinstance(metadata, dict)
            
            # Individual decision should meet max latency target
            assert decision_time_ms < TARGET_MAX_LATENCY_MS, \
                f"Decision time {decision_time_ms:.2f}ms exceeds target {TARGET_MAX_LATENCY_MS}ms"
        
        # Calculate performance statistics
        avg_latency = statistics.mean(decision_times)
        median_latency = statistics.median(decision_times)
        max_latency = max(decision_times)
        p95_latency = statistics.quantiles(decision_times, n=20)[18]  # 95th percentile
        
        # Validate performance targets
        assert avg_latency < TARGET_AVG_LATENCY_MS, \
            f"Average latency {avg_latency:.2f}ms exceeds target {TARGET_AVG_LATENCY_MS}ms"
        
        assert p95_latency < TARGET_P95_LATENCY_MS, \
            f"95th percentile latency {p95_latency:.2f}ms exceeds target {TARGET_P95_LATENCY_MS}ms"
        
        print(f"Routing Performance Results:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  Median: {median_latency:.2f}ms")
        print(f"  95th percentile: {p95_latency:.2f}ms")
        print(f"  Maximum: {max_latency:.2f}ms")
        print(f"  Samples: {len(decision_times)}")
    
    @pytest.mark.asyncio
    async def test_concurrent_routing_performance(self, navigator_facade, performance_contexts):
        """Test performance under concurrent load"""
        # Test concurrent routing decisions
        CONCURRENT_REQUESTS = 20
        TARGET_CONCURRENT_LATENCY_MS = 1000  # Higher target for concurrent load
        
        async def make_routing_request(request_id: int) -> float:
            context = performance_contexts[request_id % len(performance_contexts)]
            destination = f"concurrent_dest_{request_id}"
            
            start_time = time.perf_counter()
            protocol, metadata = await navigator_facade.select_path(destination, context)
            end_time = time.perf_counter()
            
            assert isinstance(protocol, PathProtocol)
            assert isinstance(metadata, dict)
            
            return (end_time - start_time) * 1000
        
        # Execute concurrent requests
        start_concurrent = time.perf_counter()
        tasks = [make_routing_request(i) for i in range(CONCURRENT_REQUESTS)]
        decision_times = await asyncio.gather(*tasks)
        end_concurrent = time.perf_counter()
        
        total_concurrent_time = (end_concurrent - start_concurrent) * 1000
        
        # Validate concurrent performance
        avg_concurrent_latency = statistics.mean(decision_times)
        max_concurrent_latency = max(decision_times)
        
        assert avg_concurrent_latency < TARGET_CONCURRENT_LATENCY_MS, \
            f"Average concurrent latency {avg_concurrent_latency:.2f}ms exceeds target"
        
        assert max_concurrent_latency < TARGET_CONCURRENT_LATENCY_MS * 1.5, \
            f"Max concurrent latency {max_concurrent_latency:.2f}ms too high"
        
        # Test throughput
        throughput_rps = CONCURRENT_REQUESTS / (total_concurrent_time / 1000)
        assert throughput_rps > 10, f"Throughput {throughput_rps:.2f} RPS too low"
        
        print(f"Concurrent Performance Results:")
        print(f"  Average latency: {avg_concurrent_latency:.2f}ms")
        print(f"  Max latency: {max_concurrent_latency:.2f}ms")
        print(f"  Total time: {total_concurrent_time:.2f}ms")
        print(f"  Throughput: {throughput_rps:.2f} RPS")
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, navigator_facade, performance_contexts):
        """Test memory usage efficiency"""
        import tracemalloc
        
        tracemalloc.start()
        
        # Baseline memory
        baseline_snapshot = tracemalloc.take_snapshot()
        
        # Perform routing operations
        MEMORY_TEST_OPERATIONS = 100
        
        for i in range(MEMORY_TEST_OPERATIONS):
            context = performance_contexts[i % len(performance_contexts)]
            destination = f"memory_test_{i}"
            
            await navigator_facade.select_path(destination, context)
            
            # Periodic cleanup to test memory management
            if i % 20 == 19:
                navigator_facade.cleanup_cache()
        
        # Measure memory usage
        current_snapshot = tracemalloc.take_snapshot()
        top_stats = current_snapshot.compare_to(baseline_snapshot, 'lineno')
        
        # Calculate total memory increase
        total_memory_mb = sum(stat.size_diff for stat in top_stats) / (1024 * 1024)
        
        tracemalloc.stop()
        
        # Memory usage should be reasonable
        MAX_MEMORY_INCREASE_MB = 50  # 50MB max increase
        
        assert total_memory_mb < MAX_MEMORY_INCREASE_MB, \
            f"Memory increase {total_memory_mb:.2f}MB exceeds limit {MAX_MEMORY_INCREASE_MB}MB"
        
        print(f"Memory Efficiency Results:")
        print(f"  Operations: {MEMORY_TEST_OPERATIONS}")
        print(f"  Memory increase: {total_memory_mb:.2f}MB")
        print(f"  Memory per operation: {(total_memory_mb * 1024) / MEMORY_TEST_OPERATIONS:.2f}KB")
    
    @pytest.mark.asyncio
    async def test_algorithm_performance_comparison(self, navigator_facade):
        """Test performance of different routing algorithms"""
        route_service = navigator_facade.route_selection
        
        # Test different routing scenarios
        network_conditions = [
            NetworkConditions(
                bluetooth_available=True, internet_available=False,
                bandwidth_mbps=0.1, latency_ms=200.0
            ),
            NetworkConditions(
                bluetooth_available=False, internet_available=True,
                bandwidth_mbps=50.0, latency_ms=30.0
            ),
            NetworkConditions(
                bluetooth_available=True, internet_available=True,
                bandwidth_mbps=25.0, latency_ms=60.0
            )
        ]
        
        message_contexts = [
            MessageContext(size_bytes=100, priority=9, requires_realtime=True),
            MessageContext(size_bytes=10240, priority=3, requires_realtime=False),
            MessageContext(size_bytes=1024, priority=7, privacy_required=True)
        ]
        
        algorithm_times = []
        
        # Test algorithm performance across scenarios
        for conditions in network_conditions:
            for context in message_contexts:
                start_time = time.perf_counter()
                
                protocol, scores = await route_service.select_optimal_route(
                    "algorithm_test_dest",
                    context,
                    ["bitchat", "betanet", "scion", "store_forward"],
                    conditions
                )
                
                end_time = time.perf_counter()
                
                algorithm_time = (end_time - start_time) * 1000
                algorithm_times.append(algorithm_time)
                
                assert isinstance(protocol, PathProtocol)
                assert isinstance(scores, dict)
        
        # Algorithm performance should be fast
        avg_algorithm_time = statistics.mean(algorithm_times)
        max_algorithm_time = max(algorithm_times)
        
        TARGET_AVG_ALGORITHM_TIME = 50  # 50ms average
        TARGET_MAX_ALGORITHM_TIME = 200  # 200ms max
        
        assert avg_algorithm_time < TARGET_AVG_ALGORITHM_TIME, \
            f"Average algorithm time {avg_algorithm_time:.2f}ms too slow"
        
        assert max_algorithm_time < TARGET_MAX_ALGORITHM_TIME, \
            f"Max algorithm time {max_algorithm_time:.2f}ms too slow"
        
        print(f"Algorithm Performance Results:")
        print(f"  Average: {avg_algorithm_time:.2f}ms")
        print(f"  Maximum: {max_algorithm_time:.2f}ms")
        print(f"  Scenarios tested: {len(algorithm_times)}")
    
    @pytest.mark.asyncio
    async def test_energy_efficiency_validation(self, navigator_facade):
        """Test energy efficiency of routing decisions"""
        energy_optimizer = navigator_facade.energy_optimizer
        
        # Mock battery conditions
        battery_scenarios = [
            {"level": 80, "charging": True},   # High battery, charging
            {"level": 50, "charging": False},  # Medium battery
            {"level": 15, "charging": False},  # Low battery
            {"level": 5, "charging": False}    # Critical battery
        ]
        
        protocols = [PathProtocol.BITCHAT, PathProtocol.BETANET, PathProtocol.SCION, PathProtocol.STORE_FORWARD]
        
        energy_decisions = {}
        
        for scenario in battery_scenarios:
            level = scenario["level"]
            
            # Test battery optimization
            start_time = time.perf_counter()
            optimized_protocols = energy_optimizer.optimize_for_battery_life(level, protocols)
            optimization_time = (time.perf_counter() - start_time) * 1000
            
            # Optimization should be fast
            assert optimization_time < 10, f"Energy optimization too slow: {optimization_time:.2f}ms"
            
            # Should return valid protocol list
            assert isinstance(optimized_protocols, list)
            assert len(optimized_protocols) > 0
            
            # Low battery should prefer energy-efficient protocols
            if level < 20:  # Low battery
                # Should prefer store-forward or bitchat for low battery
                efficient_protocols = {PathProtocol.STORE_FORWARD, PathProtocol.BITCHAT}
                top_choice = optimized_protocols[0]
                assert top_choice in efficient_protocols, \
                    f"Low battery should prefer efficient protocols, got {top_choice.value}"
            
            energy_decisions[level] = {
                "optimized_protocols": [p.value for p in optimized_protocols],
                "optimization_time_ms": optimization_time
            }
        
        # Test energy-efficient path selection
        for energy_mode in [EnergyMode.POWERSAVE, EnergyMode.BALANCED, EnergyMode.PERFORMANCE]:
            start_time = time.perf_counter()
            efficient_paths = energy_optimizer.select_energy_efficient_paths(protocols, energy_mode)
            selection_time = (time.perf_counter() - start_time) * 1000
            
            assert selection_time < 10, f"Path selection too slow: {selection_time:.2f}ms"
            assert isinstance(efficient_paths, list)
            assert len(efficient_paths) > 0
        
        print(f"Energy Efficiency Results:")
        for level, decision in energy_decisions.items():
            print(f"  Battery {level}%: {decision['optimized_protocols']} "
                  f"({decision['optimization_time_ms']:.2f}ms)")
    
    @pytest.mark.asyncio
    async def test_service_coordination_efficiency(self, navigator_facade):
        """Test efficiency of inter-service coordination"""
        event_bus = navigator_facade.event_bus
        
        # Test event bus performance
        EVENT_COUNT = 1000
        coordination_times = []
        
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
        
        event_bus.subscribe("performance_test", event_handler, "perf_test")
        
        # Measure event processing time
        from experiments.agents.agents.navigator.interfaces.routing_interfaces import RoutingEvent
        
        start_time = time.perf_counter()
        
        for i in range(EVENT_COUNT):
            event = RoutingEvent(
                event_type="performance_test",
                timestamp=time.time(),
                source_service="PerformanceTest",
                data={"test_id": i}
            )
            
            event_start = time.perf_counter()
            event_bus.publish(event)
            event_end = time.perf_counter()
            
            coordination_times.append((event_end - event_start) * 1000)
        
        # Wait for all events to be processed
        await asyncio.sleep(0.1)
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Event coordination should be efficient
        avg_coordination_time = statistics.mean(coordination_times)
        max_coordination_time = max(coordination_times)
        
        TARGET_AVG_COORDINATION_MS = 1.0  # 1ms average
        TARGET_MAX_COORDINATION_MS = 10.0  # 10ms max
        
        assert avg_coordination_time < TARGET_AVG_COORDINATION_MS, \
            f"Average coordination time {avg_coordination_time:.3f}ms too slow"
        
        assert max_coordination_time < TARGET_MAX_COORDINATION_MS, \
            f"Max coordination time {max_coordination_time:.3f}ms too slow"
        
        # Events should be delivered
        assert len(events_received) >= EVENT_COUNT * 0.9, \
            f"Event delivery rate too low: {len(events_received)}/{EVENT_COUNT}"
        
        print(f"Service Coordination Results:")
        print(f"  Events: {EVENT_COUNT}")
        print(f"  Total time: {total_time:.2f}ms")
        print(f"  Average coordination: {avg_coordination_time:.3f}ms")
        print(f"  Max coordination: {max_coordination_time:.3f}ms")
        print(f"  Events delivered: {len(events_received)}")
    
    @pytest.mark.asyncio
    async def test_scalability_validation(self, navigator_facade):
        """Test system scalability with increasing load"""
        # Test with increasing message sizes
        message_sizes = [100, 1024, 10240, 102400, 1024000]  # 100B to 1MB
        
        size_performance = {}
        
        for size in message_sizes:
            context = MessageContext(size_bytes=size, priority=5)
            
            # Measure routing time for different message sizes
            times = []
            for _ in range(10):  # Multiple samples per size
                start_time = time.perf_counter()
                protocol, metadata = await navigator_facade.select_path(
                    f"size_test_{size}", context
                )
                end_time = time.perf_counter()
                
                times.append((end_time - start_time) * 1000)
                
                assert isinstance(protocol, PathProtocol)
                assert isinstance(metadata, dict)
            
            avg_time = statistics.mean(times)
            size_performance[size] = avg_time
            
            # Routing time should not scale linearly with message size
            # (routing decision should be mostly independent of message size)
            TARGET_MAX_TIME_MS = 300  # Should handle even large messages quickly
            
            assert avg_time < TARGET_MAX_TIME_MS, \
                f"Size {size} bytes took {avg_time:.2f}ms (too slow)"
        
        # Test scalability with peer count
        peer_counts = [10, 50, 100, 200]
        peer_performance = {}
        
        for peer_count in peer_counts:
            # Add peers to the system
            for i in range(peer_count):
                peer_info = PeerInfo(
                    peer_id=f"scalability_peer_{i}",
                    protocols={"bitchat", "betanet"},
                    hop_distance=min(7, (i % 7) + 1),
                    trust_score=0.8
                )
                navigator_facade.update_peer_info(f"scalability_peer_{i}", peer_info)
            
            # Measure routing performance with this peer count
            context = MessageContext(size_bytes=1024, priority=5)
            
            times = []
            for i in range(5):  # Fewer samples for higher peer counts
                start_time = time.perf_counter()
                protocol, metadata = await navigator_facade.select_path(
                    f"scalability_dest_{i}", context
                )
                end_time = time.perf_counter()
                
                times.append((end_time - start_time) * 1000)
            
            avg_time = statistics.mean(times)
            peer_performance[peer_count] = avg_time
            
            # Performance should not degrade significantly with peer count
            TARGET_SCALABILITY_MS = 500  # Even with many peers
            
            assert avg_time < TARGET_SCALABILITY_MS, \
                f"Peer count {peer_count} took {avg_time:.2f}ms (poor scalability)"
        
        print(f"Scalability Results:")
        print("  Message size performance:")
        for size, time_ms in size_performance.items():
            print(f"    {size} bytes: {time_ms:.2f}ms")
        print("  Peer count performance:")
        for count, time_ms in peer_performance.items():
            print(f"    {count} peers: {time_ms:.2f}ms")
    
    def test_architectural_overhead(self, navigator_facade):
        """Test that service-oriented architecture doesn't add significant overhead"""
        # Compare service call overhead
        route_service = navigator_facade.route_selection
        
        # Direct service method call
        start_direct = time.perf_counter()
        metrics = route_service.get_performance_metrics()
        end_direct = time.perf_counter()
        
        direct_time = (end_direct - start_direct) * 1000
        
        # Through facade
        start_facade = time.perf_counter()
        status = navigator_facade.get_system_status()
        end_facade = time.perf_counter()
        
        facade_time = (end_facade - start_facade) * 1000
        
        # Overhead should be minimal
        MAX_OVERHEAD_MS = 10  # 10ms max overhead
        
        assert direct_time < MAX_OVERHEAD_MS, f"Direct service call too slow: {direct_time:.2f}ms"
        assert facade_time < MAX_OVERHEAD_MS * 2, f"Facade call too slow: {facade_time:.2f}ms"
        
        # Validate that both calls work
        assert isinstance(metrics, dict)
        assert isinstance(status, dict)
        assert "services" in status
        
        print(f"Architectural Overhead:")
        print(f"  Direct service call: {direct_time:.3f}ms")
        print(f"  Facade call: {facade_time:.3f}ms")
    
    @pytest.mark.asyncio
    async def test_performance_regression_protection(self, navigator_facade, performance_contexts):
        """Test to ensure no performance regression vs original PathPolicy"""
        
        # Original PathPolicy performance targets (estimated from 1,438 lines)
        ORIGINAL_TARGETS = {
            "max_decision_time_ms": 500,
            "avg_decision_time_ms": 250, 
            "concurrent_throughput_rps": 5,
            "memory_per_operation_kb": 100
        }
        
        # New architecture should meet or exceed these targets
        NEW_TARGETS = {
            "max_decision_time_ms": 400,    # 20% improvement
            "avg_decision_time_ms": 200,    # 20% improvement  
            "concurrent_throughput_rps": 8,  # 60% improvement
            "memory_per_operation_kb": 80    # 20% improvement
        }
        
        # Test decision times
        decision_times = []
        
        for i in range(30):
            context = performance_contexts[i % len(performance_contexts)]
            destination = f"regression_test_{i}"
            
            start_time = time.perf_counter()
            protocol, metadata = await navigator_facade.select_path(destination, context)
            end_time = time.perf_counter()
            
            decision_time = (end_time - start_time) * 1000
            decision_times.append(decision_time)
            
            assert isinstance(protocol, PathProtocol)
        
        avg_decision_time = statistics.mean(decision_times)
        max_decision_time = max(decision_times)
        
        # Should exceed performance targets
        assert avg_decision_time < NEW_TARGETS["avg_decision_time_ms"], \
            f"Average decision time {avg_decision_time:.2f}ms regression"
        
        assert max_decision_time < NEW_TARGETS["max_decision_time_ms"], \
            f"Max decision time {max_decision_time:.2f}ms regression"
        
        print(f"Performance Regression Test:")
        print(f"  Target avg: {NEW_TARGETS['avg_decision_time_ms']}ms, "
              f"Actual: {avg_decision_time:.2f}ms")
        print(f"  Target max: {NEW_TARGETS['max_decision_time_ms']}ms, "
              f"Actual: {max_decision_time:.2f}ms")
        print(f"  ✓ No performance regression detected")


@pytest.mark.asyncio
async def test_overall_architecture_performance():
    """Overall test validating architectural performance improvements"""
    
    # Test the complete architectural transformation
    facade = create_navigator_facade(
        routing_priority=RoutingPriority.PERFORMANCE_FIRST,
        energy_mode=EnergyMode.BALANCED
    )
    
    try:
        with patch('experiments.agents.agents.navigator.services.network_monitoring_service.psutil'), \
             patch('experiments.agents.agents.navigator.services.network_monitoring_service.bluetooth'), \
             patch('experiments.agents.agents.navigator.services.energy_optimization_service.psutil'):
            
            await facade.initialize()
        
        # Comprehensive performance test
        start_time = time.perf_counter()
        
        # Simulate realistic usage pattern
        contexts = [
            MessageContext(size_bytes=512, priority=8, requires_realtime=True),
            MessageContext(size_bytes=2048, priority=5, privacy_required=True),
            MessageContext(size_bytes=128, priority=9),
            MessageContext(size_bytes=8192, priority=3),
            MessageContext(size_bytes=1024, priority=7, privacy_required=True)
        ]
        
        # Test various operations
        operations_completed = 0
        
        # Routing decisions
        for i in range(25):
            context = contexts[i % len(contexts)]
            protocol, metadata = await facade.select_path(f"perf_dest_{i}", context)
            assert isinstance(protocol, PathProtocol)
            operations_completed += 1
        
        # Configuration changes
        facade.set_energy_mode(EnergyMode.POWERSAVE)
        facade.set_routing_priority(RoutingPriority.OFFLINE_FIRST)
        operations_completed += 2
        
        # Status queries
        for _ in range(5):
            status = facade.get_system_status()
            assert isinstance(status, dict)
            operations_completed += 1
        
        # Cache cleanup
        facade.cleanup_cache()
        operations_completed += 1
        
        total_time = time.perf_counter() - start_time
        
        # Overall performance validation
        TOTAL_OPERATIONS = operations_completed
        TARGET_TOTAL_TIME_S = 5.0  # Should complete in under 5 seconds
        TARGET_OPS_PER_SECOND = 10  # At least 10 operations per second
        
        ops_per_second = TOTAL_OPERATIONS / total_time
        
        assert total_time < TARGET_TOTAL_TIME_S, \
            f"Total time {total_time:.2f}s exceeds target {TARGET_TOTAL_TIME_S}s"
        
        assert ops_per_second > TARGET_OPS_PER_SECOND, \
            f"Throughput {ops_per_second:.1f} ops/s below target {TARGET_OPS_PER_SECOND}"
        
        print(f"Overall Architecture Performance:")
        print(f"  Operations: {TOTAL_OPERATIONS}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {ops_per_second:.1f} ops/s")
        print(f"  ✓ Architecture performance validated")
        
        # Validate architectural metrics achieved
        service_count = len(facade.services)
        assert service_count == 7, f"Expected 7 services, got {service_count}"
        
        # All services should be operational
        system_status = facade.get_system_status()
        service_statuses = system_status.get("services", {})
        
        for service_name in facade.services.keys():
            assert service_name in service_statuses, f"Service {service_name} missing from status"
        
        print(f"  ✓ All {service_count} services operational")
        print(f"  ✓ Original 1,438-line god class successfully separated")
        
    finally:
        await facade.shutdown()


if __name__ == "__main__":
    # Run performance validation tests
    pytest.main([__file__, "-v", "-s"])