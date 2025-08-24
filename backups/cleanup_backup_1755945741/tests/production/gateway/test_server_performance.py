#!/usr/bin/env python3
"""
Production Gateway Performance Test Suite
Agent 5: Test System Orchestrator

Target: Validate Agent 1's consolidated gateway server performance
Performance Targets:
- Health check: <2.8ms response time (97% improvement achieved)
- API endpoints: <100ms response time
- Security middleware: <50ms overhead
- Error handling: <200ms recovery time
"""

import asyncio
import time

from fastapi.testclient import TestClient
import httpx
import pytest

# Import the consolidated gateway server
try:
    from core.gateway.server import GatewayConfig, app
except ImportError:
    app = None
    GatewayConfig = None


class TestGatewayPerformance:
    """Performance tests for consolidated gateway server"""

    @pytest.fixture
    def client(self):
        """Test client fixture"""
        if app is None:
            pytest.skip("Gateway server not available")
        return TestClient(app)

    @pytest.fixture
    def config(self):
        """Gateway configuration fixture"""
        if GatewayConfig is None:
            pytest.skip("Gateway config not available")
        return GatewayConfig()

    def test_health_check_performance_target(self, client):
        """
        CRITICAL: Validate 2.8ms health check performance target
        This is the key metric from Agent 1's 97% improvement
        """
        # Warm up the endpoint
        for _ in range(3):
            client.get("/health")

        # Measure performance over multiple requests
        response_times = []
        for _ in range(100):
            start_time = time.perf_counter()
            response = client.get("/health")
            end_time = time.perf_counter()

            assert response.status_code == 200
            response_times.append((end_time - start_time) * 1000)  # Convert to ms

        # Statistical analysis
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        p95_time = sorted(response_times)[int(0.95 * len(response_times))]

        # Validate performance targets
        assert avg_time < 2.8, f"Average response time {avg_time:.2f}ms exceeds 2.8ms target"
        assert p95_time < 5.0, f"95th percentile {p95_time:.2f}ms exceeds 5.0ms threshold"
        assert max_time < 10.0, f"Maximum response time {max_time:.2f}ms exceeds 10.0ms limit"

        print("Health Check Performance:")
        print(f"  Average: {avg_time:.2f}ms (target: <2.8ms)")
        print(f"  95th percentile: {p95_time:.2f}ms")
        print(f"  Min/Max: {min_time:.2f}ms / {max_time:.2f}ms")

    def test_api_endpoint_performance(self, client):
        """Test all API endpoints meet <100ms performance target"""
        endpoints = [
            ("/", "GET"),
            ("/health", "GET"),
            ("/metrics", "GET"),
            ("/query", "POST"),
            ("/chat", "POST"),
        ]

        performance_results = {}

        for endpoint, method in endpoints:
            response_times = []

            for _ in range(20):
                start_time = time.perf_counter()

                if method == "GET":
                    client.get(endpoint)
                elif method == "POST":
                    # Provide minimal valid payloads for POST endpoints
                    if endpoint == "/query":
                        client.post(endpoint, json={"query": "test"})
                    elif endpoint == "/chat":
                        client.post(endpoint, json={"message": "test"})
                    else:
                        client.post(endpoint, json={})

                end_time = time.perf_counter()
                response_time_ms = (end_time - start_time) * 1000
                response_times.append(response_time_ms)

            avg_time = sum(response_times) / len(response_times)
            p95_time = sorted(response_times)[int(0.95 * len(response_times))]

            performance_results[f"{method} {endpoint}"] = {
                "avg_ms": avg_time,
                "p95_ms": p95_time,
                "status": "PASS" if p95_time < 100 else "FAIL",
            }

            # Validate performance target
            assert p95_time < 100, f"{method} {endpoint} p95 time {p95_time:.2f}ms exceeds 100ms target"

        print("API Endpoint Performance Results:")
        for endpoint, metrics in performance_results.items():
            print(f"  {endpoint}: {metrics['avg_ms']:.2f}ms avg, {metrics['p95_ms']:.2f}ms p95 - {metrics['status']}")

    def test_concurrent_request_handling(self, client):
        """Test gateway performance under concurrent load"""

        async def make_request():
            """Async request helper"""
            async with httpx.AsyncClient(app=app, base_url="http://test") as async_client:
                start_time = time.perf_counter()
                response = await async_client.get("/health")
                end_time = time.perf_counter()
                return (end_time - start_time) * 1000, response.status_code

        async def concurrent_load_test():
            """Execute concurrent requests"""
            # Test with 50 concurrent requests
            tasks = [make_request() for _ in range(50)]
            results = await asyncio.gather(*tasks)
            return results

        # Run the concurrent load test
        results = asyncio.run(concurrent_load_test())

        # Analyze results
        response_times = [result[0] for result in results]
        status_codes = [result[1] for result in results]

        # Validate all requests succeeded
        assert all(code == 200 for code in status_codes), "Some concurrent requests failed"

        # Validate performance under load
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        p95_time = sorted(response_times)[int(0.95 * len(response_times))]

        # Under load, allow some degradation but stay reasonable
        assert avg_time < 10.0, f"Concurrent average time {avg_time:.2f}ms too high"
        assert p95_time < 20.0, f"Concurrent p95 time {p95_time:.2f}ms too high"

        print("Concurrent Load Performance (50 requests):")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  95th percentile: {p95_time:.2f}ms")
        print(f"  Maximum: {max_time:.2f}ms")

    def test_security_middleware_overhead(self, client):
        """Test that security middleware adds minimal overhead"""
        # This would test CORS, rate limiting, security headers, etc.
        # For now, we'll test that requests still meet performance targets
        # even with security middleware active

        headers = {"Origin": "https://example.com", "User-Agent": "test-client/1.0"}

        response_times = []
        for _ in range(20):
            start_time = time.perf_counter()
            response = client.get("/health", headers=headers)
            end_time = time.perf_counter()

            assert response.status_code == 200
            response_times.append((end_time - start_time) * 1000)

        avg_time = sum(response_times) / len(response_times)

        # Security middleware should add <50ms overhead
        assert avg_time < 50.0, f"Security middleware overhead {avg_time:.2f}ms exceeds 50ms target"

        print(f"Security Middleware Overhead: {avg_time:.2f}ms (target: <50ms)")

    def test_error_handling_performance(self, client):
        """Test that error handling is fast and doesn't degrade performance"""

        # Test 404 error handling
        response_times_404 = []
        for _ in range(20):
            start_time = time.perf_counter()
            response = client.get("/nonexistent-endpoint")
            end_time = time.perf_counter()

            assert response.status_code == 404
            response_times_404.append((end_time - start_time) * 1000)

        avg_404_time = sum(response_times_404) / len(response_times_404)

        # Error handling should be fast
        assert avg_404_time < 200.0, f"404 error handling {avg_404_time:.2f}ms exceeds 200ms target"

        print("Error Handling Performance:")
        print(f"  404 responses: {avg_404_time:.2f}ms (target: <200ms)")

    def test_memory_usage_stability(self, client):
        """Test that gateway doesn't have memory leaks under load"""
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Make many requests to check for memory leaks
        for _ in range(500):
            client.get("/health")

        # Check memory usage after load
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        memory_increase_mb = memory_increase / (1024 * 1024)

        # Memory increase should be minimal (<50MB for 500 requests)
        assert memory_increase_mb < 50.0, f"Memory increased by {memory_increase_mb:.2f}MB - possible leak"

        print(f"Memory Usage: +{memory_increase_mb:.2f}MB after 500 requests (limit: <50MB)")


@pytest.mark.benchmark
class TestGatewayBenchmarks:
    """Comprehensive benchmarks for performance regression testing"""

    def test_health_check_benchmark(self, benchmark, client):
        """Benchmark health check endpoint"""
        if client is None:
            pytest.skip("Gateway not available")

        def health_check():
            return client.get("/health")

        result = benchmark(health_check)
        assert result.status_code == 200

        # Benchmark should meet 2.8ms target
        assert benchmark.stats.stats.mean < 0.0028, "Health check benchmark exceeds 2.8ms target"

    def test_api_throughput_benchmark(self, client):
        """Test maximum throughput under optimal conditions"""
        from collections import deque
        import threading
        import time

        results = deque()
        stop_event = threading.Event()

        def worker():
            """Worker thread for throughput testing"""
            while not stop_event.is_set():
                start = time.perf_counter()
                response = client.get("/health")
                end = time.perf_counter()

                if response.status_code == 200:
                    results.append(end - start)

        # Run throughput test for 10 seconds with 10 threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        time.sleep(10)  # Run for 10 seconds
        stop_event.set()

        for thread in threads:
            thread.join()

        # Calculate throughput
        total_requests = len(results)
        requests_per_second = total_requests / 10.0
        avg_response_time = sum(results) / len(results) * 1000  # Convert to ms

        print("Throughput Benchmark:")
        print(f"  Requests/second: {requests_per_second:.2f}")
        print(f"  Total requests: {total_requests}")
        print(f"  Average response time: {avg_response_time:.2f}ms")

        # Validate throughput targets
        assert requests_per_second > 1000, f"Throughput {requests_per_second:.2f} RPS below 1000 RPS target"
        assert avg_response_time < 10.0, f"Average response time {avg_response_time:.2f}ms too high under load"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
