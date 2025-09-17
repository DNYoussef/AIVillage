"""
Performance target validation tests
Verifies the system meets required performance metrics
"""

import asyncio
import json
import pytest
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, AsyncMock, patch
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from infrastructure.python.betanet_bridge import BetaNetBridge


class TestPerformanceTargets:
    """Black box performance validation tests"""

    @pytest.fixture
    async def perf_system(self):
        """Setup performance testing system"""
        system = {
            'bridge': BetaNetBridge(host='localhost', port=5680),
            'metrics': {
                'latencies': [],
                'throughput': [],
                'errors': 0,
                'success': 0
            }
        }

        await system['bridge'].start()
        yield system
        await system['bridge'].stop()

    @pytest.mark.asyncio
    async def test_p95_latency_target(self, perf_system):
        """Verify P95 latency is under 75ms target"""
        bridge = perf_system['bridge']
        latencies = []

        # Warm up the system
        for _ in range(10):
            request = {
                'jsonrpc': '2.0',
                'id': 'warmup',
                'method': 'ping',
                'params': {}
            }
            await bridge.handle_request(json.dumps(request))

        # Measure latencies for 100 requests
        for i in range(100):
            start = time.perf_counter()

            request = {
                'jsonrpc': '2.0',
                'id': f'latency-{i}',
                'method': 'processRequest',
                'params': {
                    'protocol': 'betanet',
                    'data': {'message': f'test-{i}'},
                    'privacyTier': 'Silver'
                }
            }

            response = await bridge.handle_request(json.dumps(request))

            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

            # Verify response is valid
            response_data = json.loads(response)
            assert 'result' in response_data or 'error' in response_data

        # Calculate P95
        latencies.sort()
        p95_index = int(len(latencies) * 0.95)
        p95_latency = latencies[p95_index]

        # Calculate other metrics
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p99_latency = latencies[int(len(latencies) * 0.99)]

        print(f"\nLatency Metrics:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  Median: {median_latency:.2f}ms")
        print(f"  P95: {p95_latency:.2f}ms")
        print(f"  P99: {p99_latency:.2f}ms")

        # Assert P95 target
        assert p95_latency < 75, f"P95 latency {p95_latency:.2f}ms exceeds 75ms target"

    @pytest.mark.asyncio
    async def test_throughput_target(self, perf_system):
        """Test system throughput capabilities"""
        bridge = perf_system['bridge']

        # Test different batch sizes
        batch_sizes = [1, 10, 50, 100]
        throughput_results = []

        for batch_size in batch_sizes:
            requests = []
            for i in range(batch_size):
                req = {
                    'jsonrpc': '2.0',
                    'id': f'throughput-{batch_size}-{i}',
                    'method': 'processRequest',
                    'params': {'data': {'batch': batch_size, 'index': i}}
                }
                requests.append(json.dumps(req))

            # Measure time for batch
            start = time.perf_counter()

            # Process requests concurrently
            tasks = [bridge.handle_request(req) for req in requests]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            duration = time.perf_counter() - start

            # Calculate throughput
            successful = sum(1 for r in responses if not isinstance(r, Exception))
            throughput = successful / duration if duration > 0 else 0

            throughput_results.append({
                'batch_size': batch_size,
                'duration': duration,
                'throughput': throughput,
                'success_rate': successful / batch_size
            })

            print(f"\nBatch Size {batch_size}:")
            print(f"  Duration: {duration:.3f}s")
            print(f"  Throughput: {throughput:.1f} req/s")
            print(f"  Success Rate: {successful}/{batch_size}")

        # Verify minimum throughput
        max_throughput = max(r['throughput'] for r in throughput_results)
        assert max_throughput > 100, f"Max throughput {max_throughput:.1f} req/s below 100 req/s target"

    @pytest.mark.asyncio
    async def test_concurrent_connections(self, perf_system):
        """Test handling of concurrent connections"""
        bridge = perf_system['bridge']

        # Test concurrent connection limits
        connection_counts = [10, 25, 50, 100]

        for count in connection_counts:
            start = time.perf_counter()
            errors = 0

            # Create concurrent requests
            async def make_request(index):
                try:
                    request = {
                        'jsonrpc': '2.0',
                        'id': f'concurrent-{count}-{index}',
                        'method': 'processRequest',
                        'params': {'connection': index}
                    }
                    response = await bridge.handle_request(json.dumps(request))
                    return json.loads(response)
                except Exception as e:
                    return {'error': str(e)}

            # Execute concurrently
            tasks = [make_request(i) for i in range(count)]
            responses = await asyncio.gather(*tasks)

            duration = time.perf_counter() - start

            # Count successes and errors
            successes = sum(1 for r in responses if 'result' in r)
            errors = sum(1 for r in responses if 'error' in r)

            print(f"\n{count} Concurrent Connections:")
            print(f"  Duration: {duration:.3f}s")
            print(f"  Success: {successes}/{count}")
            print(f"  Errors: {errors}/{count}")

            # Should handle at least 50 concurrent connections
            if count <= 50:
                assert successes >= count * 0.95, f"Too many failures at {count} connections"

    @pytest.mark.asyncio
    async def test_memory_efficiency(self, perf_system):
        """Test memory usage under load"""
        bridge = perf_system['bridge']

        # Baseline memory check
        baseline_request = {
            'jsonrpc': '2.0',
            'id': 'memory-baseline',
            'method': 'getMemoryUsage',
            'params': {}
        }

        baseline_response = await bridge.handle_request(json.dumps(baseline_request))
        baseline_data = json.loads(baseline_response)

        if 'result' in baseline_data:
            baseline_memory = baseline_data['result'].get('memory', 0)
        else:
            baseline_memory = 0

        # Process many requests
        for i in range(1000):
            request = {
                'jsonrpc': '2.0',
                'id': f'memory-test-{i}',
                'method': 'processRequest',
                'params': {
                    'data': {'index': i, 'payload': 'x' * 1000}  # 1KB payload
                }
            }
            await bridge.handle_request(json.dumps(request))

        # Check memory after load
        final_response = await bridge.handle_request(json.dumps(baseline_request))
        final_data = json.loads(final_response)

        if 'result' in final_data:
            final_memory = final_data['result'].get('memory', baseline_memory)
            memory_increase = final_memory - baseline_memory

            print(f"\nMemory Usage:")
            print(f"  Baseline: {baseline_memory / 1024 / 1024:.1f}MB")
            print(f"  Final: {final_memory / 1024 / 1024:.1f}MB")
            print(f"  Increase: {memory_increase / 1024 / 1024:.1f}MB")

            # Memory increase should be reasonable (< 100MB for 1000 requests)
            assert memory_increase < 100 * 1024 * 1024, "Excessive memory usage detected"

    @pytest.mark.asyncio
    async def test_circuit_breaker_performance(self, perf_system):
        """Test circuit breaker response times"""
        bridge = perf_system['bridge']

        # Trigger circuit breaker
        for i in range(6):
            failing_request = {
                'jsonrpc': '2.0',
                'id': f'cb-fail-{i}',
                'method': 'forceError',
                'params': {'error': True}
            }
            await bridge.handle_request(json.dumps(failing_request))

        # Circuit should be open - measure fast-fail time
        fast_fail_times = []

        for i in range(20):
            start = time.perf_counter()

            request = {
                'jsonrpc': '2.0',
                'id': f'cb-test-{i}',
                'method': 'processRequest',
                'params': {'data': 'test'}
            }

            response = await bridge.handle_request(json.dumps(request))

            duration_ms = (time.perf_counter() - start) * 1000
            fast_fail_times.append(duration_ms)

            # Should get circuit breaker error
            response_data = json.loads(response)
            assert 'error' in response_data

        # Fast-fail should be very quick (< 5ms)
        avg_fail_time = statistics.mean(fast_fail_times)
        print(f"\nCircuit Breaker Fast-Fail: {avg_fail_time:.2f}ms")
        assert avg_fail_time < 5, "Circuit breaker fast-fail too slow"

    @pytest.mark.asyncio
    async def test_protocol_routing_performance(self, perf_system):
        """Test performance of different protocol routes"""
        bridge = perf_system['bridge']

        protocols = ['betanet', 'bitchat', 'p2p', 'fog']
        protocol_metrics = {}

        for protocol in protocols:
            latencies = []

            for i in range(50):
                start = time.perf_counter()

                request = {
                    'jsonrpc': '2.0',
                    'id': f'{protocol}-perf-{i}',
                    'method': 'routeProtocol',
                    'params': {
                        'protocol': protocol,
                        'data': {'test': i}
                    }
                }

                response = await bridge.handle_request(json.dumps(request))

                latency_ms = (time.perf_counter() - start) * 1000
                latencies.append(latency_ms)

            # Calculate metrics for protocol
            protocol_metrics[protocol] = {
                'avg': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'p95': sorted(latencies)[int(len(latencies) * 0.95)]
            }

            print(f"\n{protocol.upper()} Protocol:")
            print(f"  Avg: {protocol_metrics[protocol]['avg']:.2f}ms")
            print(f"  P95: {protocol_metrics[protocol]['p95']:.2f}ms")

        # All protocols should meet P95 target
        for protocol, metrics in protocol_metrics.items():
            assert metrics['p95'] < 75, f"{protocol} P95 {metrics['p95']:.2f}ms exceeds target"

    @pytest.mark.asyncio
    async def test_privacy_tier_performance(self, perf_system):
        """Test performance impact of privacy tiers"""
        bridge = perf_system['bridge']

        tiers = ['Bronze', 'Silver', 'Gold', 'Platinum']
        tier_metrics = {}

        for tier in tiers:
            latencies = []

            for i in range(50):
                start = time.perf_counter()

                request = {
                    'jsonrpc': '2.0',
                    'id': f'{tier}-perf-{i}',
                    'method': 'processWithPrivacy',
                    'params': {
                        'privacyTier': tier,
                        'data': {
                            'userId': f'user-{i}',
                            'email': f'test{i}@example.com',
                            'content': 'test message'
                        }
                    }
                }

                response = await bridge.handle_request(json.dumps(request))

                latency_ms = (time.perf_counter() - start) * 1000
                latencies.append(latency_ms)

            # Calculate tier metrics
            tier_metrics[tier] = {
                'avg': statistics.mean(latencies),
                'p95': sorted(latencies)[int(len(latencies) * 0.95)]
            }

            print(f"\n{tier} Tier Performance:")
            print(f"  Avg: {tier_metrics[tier]['avg']:.2f}ms")
            print(f"  P95: {tier_metrics[tier]['p95']:.2f}ms")

        # Bronze (most transformation) should still meet target
        assert tier_metrics['Bronze']['p95'] < 100, "Bronze tier too slow"

        # Platinum (no transformation) should be fastest
        assert tier_metrics['Platinum']['avg'] < tier_metrics['Bronze']['avg'], \
            "Platinum should be faster than Bronze"

    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, perf_system):
        """Test performance under sustained load"""
        bridge = perf_system['bridge']

        # Run for 30 seconds at steady rate
        duration_seconds = 30
        target_rps = 50  # Requests per second

        start_time = time.time()
        request_times = []
        errors = 0

        while time.time() - start_time < duration_seconds:
            request_start = time.perf_counter()

            request = {
                'jsonrpc': '2.0',
                'id': f'sustained-{len(request_times)}',
                'method': 'processRequest',
                'params': {'data': 'sustained load test'}
            }

            try:
                response = await bridge.handle_request(json.dumps(request))
                response_data = json.loads(response)
                if 'error' in response_data:
                    errors += 1
            except Exception:
                errors += 1

            request_time = time.perf_counter() - request_start
            request_times.append(request_time * 1000)

            # Maintain target rate
            sleep_time = max(0, (1.0 / target_rps) - request_time)
            await asyncio.sleep(sleep_time)

        # Analyze sustained performance
        total_requests = len(request_times)
        actual_rps = total_requests / duration_seconds
        error_rate = errors / total_requests if total_requests > 0 else 0

        # Calculate percentiles
        sorted_times = sorted(request_times)
        p50 = sorted_times[int(len(sorted_times) * 0.50)]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]

        print(f"\nSustained Load Results ({duration_seconds}s at {target_rps} RPS):")
        print(f"  Total Requests: {total_requests}")
        print(f"  Actual RPS: {actual_rps:.1f}")
        print(f"  Error Rate: {error_rate:.2%}")
        print(f"  P50 Latency: {p50:.2f}ms")
        print(f"  P95 Latency: {p95:.2f}ms")
        print(f"  P99 Latency: {p99:.2f}ms")

        # Assertions
        assert actual_rps >= target_rps * 0.9, "Could not maintain target RPS"
        assert error_rate < 0.01, "Error rate too high under sustained load"
        assert p95 < 75, "P95 degraded under sustained load"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])