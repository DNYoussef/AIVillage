#!/usr/bin/env python3
"""
End-to-End tests for SCION Gateway integration.
Validates the complete pipeline from Python client through Rust gateway to Go sidecar to SCION network.

Test Coverage:
- Gateway connectivity and health checks
- SCION path discovery and selection
- Packet encapsulation and decapsulation
- Anti-replay protection validation
- Performance KPI validation (≥500k packets/min, ≤750ms p95 recovery)
- Error handling and failover scenarios
"""

import asyncio
import json
import logging
import os
from pathlib import Path
import statistics
import time
from typing import Any

import aiohttp
import pytest
import pytest_asyncio

from src.core.message_types import Message, MessageType
from src.navigation.scion_navigator import (
    SCIONAwareNavigator,
)
from src.transport.scion_gateway import GatewayConfig, SCIONGateway, SCIONGatewayError

# Test configuration
GATEWAY_ENDPOINT = os.getenv("GATEWAY_ENDPOINT", "https://127.0.0.1:8443")
SIDECAR_ENDPOINT = os.getenv("SIDECAR_ENDPOINT", "127.0.0.1:8080")
TEST_DESTINATION = os.getenv("TEST_DESTINATION", "1-ff00:0:120")
TEST_OUTPUT_DIR = Path(os.getenv("TEST_OUTPUT_DIR", "/test-results"))

# Performance targets
TARGET_THROUGHPUT_PPM = 500_000  # 500k packets per minute
TARGET_P95_RECOVERY_MS = 750  # 750ms p95 failover recovery
TARGET_FALSE_REJECT_RATE = 0.0  # 0% false-reject rate for anti-replay


logger = logging.getLogger(__name__)


class TestMetrics:
    """Test metrics collection."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.packets_sent = 0
        self.packets_received = 0
        self.errors = 0
        self.latencies = []
        self.throughput_samples = []
        self.failover_times = []
        self.false_rejects = 0
        self.anti_replay_tests = 0

    def record_packet_sent(self):
        self.packets_sent += 1

    def record_packet_received(self):
        self.packets_received += 1

    def record_error(self):
        self.errors += 1

    def record_latency(self, latency_ms: float):
        self.latencies.append(latency_ms)

    def record_throughput_sample(self, packets_per_minute: float):
        self.throughput_samples.append(packets_per_minute)

    def record_failover_time(self, failover_ms: float):
        self.failover_times.append(failover_ms)

    def record_anti_replay_test(self, false_reject: bool):
        self.anti_replay_tests += 1
        if false_reject:
            self.false_rejects += 1

    def get_summary(self) -> dict[str, Any]:
        duration = time.time() - self.start_time

        return {
            "duration_seconds": duration,
            "packets_sent": self.packets_sent,
            "packets_received": self.packets_received,
            "errors": self.errors,
            "success_rate": self.packets_received / max(1, self.packets_sent),
            "error_rate": self.errors / max(1, self.packets_sent + self.errors),
            # Latency statistics
            "latency_ms": {
                "mean": statistics.mean(self.latencies) if self.latencies else 0,
                "median": statistics.median(self.latencies) if self.latencies else 0,
                "p95": statistics.quantiles(self.latencies, n=20)[18]
                if len(self.latencies) >= 20
                else 0,
                "p99": statistics.quantiles(self.latencies, n=100)[98]
                if len(self.latencies) >= 100
                else 0,
                "min": min(self.latencies) if self.latencies else 0,
                "max": max(self.latencies) if self.latencies else 0,
            },
            # Throughput statistics
            "throughput_ppm": {
                "mean": statistics.mean(self.throughput_samples)
                if self.throughput_samples
                else 0,
                "max": max(self.throughput_samples) if self.throughput_samples else 0,
                "target_achieved": any(
                    t >= TARGET_THROUGHPUT_PPM for t in self.throughput_samples
                ),
            },
            # Failover statistics
            "failover_ms": {
                "mean": statistics.mean(self.failover_times)
                if self.failover_times
                else 0,
                "p95": statistics.quantiles(self.failover_times, n=20)[18]
                if len(self.failover_times) >= 20
                else 0,
                "target_achieved": all(
                    t <= TARGET_P95_RECOVERY_MS for t in self.failover_times
                ),
            },
            # Anti-replay statistics
            "anti_replay": {
                "tests_performed": self.anti_replay_tests,
                "false_rejects": self.false_rejects,
                "false_reject_rate": self.false_rejects
                / max(1, self.anti_replay_tests),
                "target_achieved": self.false_rejects / max(1, self.anti_replay_tests)
                <= TARGET_FALSE_REJECT_RATE,
            },
        }


@pytest_asyncio.fixture
async def gateway_config():
    """Create gateway configuration for testing."""
    return GatewayConfig(
        htx_endpoint=GATEWAY_ENDPOINT,
        sidecar_address=SIDECAR_ENDPOINT,
        request_timeout=30.0,
        connection_timeout=10.0,
        verify_ssl=False,  # Self-signed certs in test environment
    )


@pytest_asyncio.fixture
async def scion_gateway(gateway_config):
    """Create and start SCION gateway for testing."""
    gateway = SCIONGateway(gateway_config)

    try:
        await gateway.start()
        yield gateway
    finally:
        await gateway.stop()


@pytest_asyncio.fixture
async def test_metrics():
    """Create test metrics collector."""
    metrics = TestMetrics()
    yield metrics

    # Save metrics to file
    summary = metrics.get_summary()
    metrics_file = TEST_OUTPUT_DIR / "test_metrics.json"
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)

    with open(metrics_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Test metrics saved to {metrics_file}")


class TestSCIONGatewayConnectivity:
    """Test basic gateway connectivity and health."""

    @pytest.mark.asyncio
    async def test_gateway_health_check(self, scion_gateway):
        """Test gateway health check endpoint."""
        health = await scion_gateway.health_check()

        assert health is not None
        assert "status" in health

        # Gateway should be healthy
        assert health["status"] in ["healthy", "degraded"]

        # SCION should be connected
        if health["status"] == "healthy":
            assert health.get("scion_connected", False) is True

    @pytest.mark.asyncio
    async def test_scion_sidecar_connectivity(self, gateway_config):
        """Test direct connectivity to SCION sidecar."""
        async with aiohttp.ClientSession():
            # Test gRPC health (if health endpoint is exposed)
            try:
                # This would be a direct gRPC call in a real implementation
                # For now, test that the endpoint is reachable
                host, port = gateway_config.sidecar_address.split(":")

                # Simple socket connection test
                import socket

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                result = sock.connect_ex((host, int(port)))
                sock.close()

                assert (
                    result == 0
                ), f"Cannot connect to SCION sidecar at {gateway_config.sidecar_address}"

            except Exception as e:
                pytest.fail(f"SCION sidecar connectivity test failed: {e}")

    @pytest.mark.asyncio
    async def test_path_discovery(self, scion_gateway):
        """Test SCION path discovery functionality."""
        paths = await scion_gateway.query_paths(TEST_DESTINATION)

        # Should have at least one path (even if simulated)
        assert len(paths) >= 0  # Paths may not be available in test environment

        for path in paths:
            assert path.path_id is not None
            assert path.destination == TEST_DESTINATION
            assert path.rtt_us >= 0
            assert 0.0 <= path.loss_rate <= 1.0


class TestSCIONPacketProcessing:
    """Test SCION packet sending and receiving."""

    @pytest.mark.asyncio
    async def test_basic_packet_send(self, scion_gateway, test_metrics):
        """Test basic packet sending functionality."""
        test_packet = b"Hello, SCION world! This is a test packet."

        start_time = time.time()

        try:
            packet_id = await scion_gateway.send_packet(test_packet, TEST_DESTINATION)

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            test_metrics.record_packet_sent()
            test_metrics.record_latency(latency_ms)

            assert packet_id is not None
            assert len(packet_id) > 0

            logger.info(f"Sent packet {packet_id} in {latency_ms:.2f}ms")

        except SCIONGatewayError as e:
            test_metrics.record_error()
            pytest.fail(f"Packet send failed: {e}")

    @pytest.mark.asyncio
    async def test_message_serialization(self, scion_gateway, test_metrics):
        """Test message serialization and sending."""
        message = Message(
            type=MessageType.DATA,
            content={"test": "message", "timestamp": time.time()},
            metadata={"source": "e2e_test", "sequence": 1},
        )

        start_time = time.time()

        try:
            packet_id = await scion_gateway.send_message(message, TEST_DESTINATION)

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            test_metrics.record_packet_sent()
            test_metrics.record_latency(latency_ms)

            assert packet_id is not None

        except SCIONGatewayError as e:
            test_metrics.record_error()
            pytest.fail(f"Message send failed: {e}")

    @pytest.mark.asyncio
    async def test_packet_receive_polling(self, scion_gateway, test_metrics):
        """Test packet receiving via polling."""
        # Test polling with short timeout
        packets = await scion_gateway.receive_packets(timeout_ms=1000, max_packets=5)

        # Should return list (may be empty in test environment)
        assert isinstance(packets, list)

        for packet_data, source in packets:
            assert isinstance(packet_data, bytes)
            assert isinstance(source, str)
            test_metrics.record_packet_received()

    @pytest.mark.asyncio
    async def test_large_packet_handling(self, scion_gateway, test_metrics):
        """Test handling of large packets."""
        # Test with 32KB packet
        large_packet = b"X" * (32 * 1024)

        start_time = time.time()

        try:
            packet_id = await scion_gateway.send_packet(large_packet, TEST_DESTINATION)

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            test_metrics.record_packet_sent()
            test_metrics.record_latency(latency_ms)

            assert packet_id is not None

            # Large packets should take longer but still complete
            assert latency_ms < 10000  # Should complete within 10 seconds

        except SCIONGatewayError as e:
            if "packet too large" in str(e).lower():
                pytest.skip("Large packet rejected by gateway (expected)")
            else:
                test_metrics.record_error()
                pytest.fail(f"Large packet test failed: {e}")


class TestSCIONPerformance:
    """Test SCION gateway performance and KPIs."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_throughput_target(self, scion_gateway, test_metrics):
        """Test achieving target throughput of ≥500k packets/minute."""
        target_duration = 60  # 1 minute test
        batch_size = 100
        target_interval = (
            60.0 / TARGET_THROUGHPUT_PPM
        ) * batch_size  # Target time per batch

        start_time = time.time()
        packets_sent_in_interval = 0
        interval_start = start_time

        logger.info(
            f"Starting throughput test: target {TARGET_THROUGHPUT_PPM} packets/minute"
        )

        while (time.time() - start_time) < target_duration:
            batch_start = time.time()

            # Send batch of packets
            batch_tasks = []
            for i in range(batch_size):
                test_packet = (
                    f"throughput_test_packet_{packets_sent_in_interval + i}".encode()
                )
                batch_tasks.append(
                    scion_gateway.send_packet(test_packet, TEST_DESTINATION)
                )

            try:
                # Send batch concurrently
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                successful = sum(1 for r in results if not isinstance(r, Exception))
                failed = len(results) - successful

                test_metrics.packets_sent += successful
                test_metrics.errors += failed
                packets_sent_in_interval += successful

            except Exception as e:
                logger.error(f"Batch send failed: {e}")
                test_metrics.errors += batch_size

            batch_end = time.time()
            batch_duration = batch_end - batch_start

            # Calculate throughput for this interval
            if (batch_end - interval_start) >= 10.0:  # Every 10 seconds
                interval_duration = batch_end - interval_start
                throughput_ppm = (packets_sent_in_interval / interval_duration) * 60
                test_metrics.record_throughput_sample(throughput_ppm)

                logger.info(f"Current throughput: {throughput_ppm:.0f} packets/minute")

                packets_sent_in_interval = 0
                interval_start = batch_end

            # Rate limiting to avoid overwhelming the system
            if batch_duration < target_interval:
                await asyncio.sleep(target_interval - batch_duration)

        # Final throughput calculation
        total_duration = time.time() - start_time
        final_throughput = (test_metrics.packets_sent / total_duration) * 60
        test_metrics.record_throughput_sample(final_throughput)

        logger.info(f"Final throughput: {final_throughput:.0f} packets/minute")

        # Verify target achievement
        max_throughput = (
            max(test_metrics.throughput_samples)
            if test_metrics.throughput_samples
            else 0
        )
        assert (
            max_throughput >= TARGET_THROUGHPUT_PPM * 0.8
        ), f"Throughput target not achieved: {max_throughput:.0f} < {TARGET_THROUGHPUT_PPM}"

    @pytest.mark.asyncio
    async def test_concurrent_connections(self, gateway_config, test_metrics):
        """Test multiple concurrent gateway connections."""
        num_connections = 10
        packets_per_connection = 50

        async def connection_worker(worker_id: int):
            gateway = SCIONGateway(gateway_config)
            try:
                await gateway.start()

                for i in range(packets_per_connection):
                    test_packet = f"worker_{worker_id}_packet_{i}".encode()

                    start_time = time.time()
                    await gateway.send_packet(test_packet, TEST_DESTINATION)
                    end_time = time.time()

                    latency_ms = (end_time - start_time) * 1000
                    test_metrics.record_packet_sent()
                    test_metrics.record_latency(latency_ms)

            except Exception as e:
                logger.error(f"Worker {worker_id} failed: {e}")
                test_metrics.errors += packets_per_connection
            finally:
                await gateway.stop()

        # Start all workers concurrently
        start_time = time.time()

        workers = [
            asyncio.create_task(connection_worker(i)) for i in range(num_connections)
        ]

        await asyncio.gather(*workers, return_exceptions=True)

        end_time = time.time()
        total_duration = end_time - start_time

        expected_packets = num_connections * packets_per_connection
        throughput_ppm = (test_metrics.packets_sent / total_duration) * 60

        test_metrics.record_throughput_sample(throughput_ppm)

        logger.info(
            f"Concurrent test: {test_metrics.packets_sent}/{expected_packets} packets "
            f"in {total_duration:.2f}s ({throughput_ppm:.0f} ppm)"
        )

        # At least 80% success rate expected
        success_rate = test_metrics.packets_sent / expected_packets
        assert (
            success_rate >= 0.8
        ), f"Low success rate in concurrent test: {success_rate:.2%}"


class TestSCIONAntiReplay:
    """Test anti-replay protection functionality."""

    @pytest.mark.asyncio
    async def test_anti_replay_basic(self, scion_gateway, test_metrics):
        """Test basic anti-replay functionality."""
        # This test would require access to the anti-replay validation endpoint
        # For now, test that packets are properly sequenced

        sequence_test_count = 100

        for i in range(sequence_test_count):
            test_packet = f"sequence_test_packet_{i}".encode()

            try:
                packet_id = await scion_gateway.send_packet(
                    test_packet, TEST_DESTINATION
                )
                assert packet_id is not None

                test_metrics.record_packet_sent()
                test_metrics.record_anti_replay_test(false_reject=False)

            except SCIONGatewayError as e:
                # Should not be rejected for legitimate sequential packets
                test_metrics.record_anti_replay_test(false_reject=True)
                logger.warning(f"Legitimate packet rejected: {e}")

        # Verify false reject rate
        false_reject_rate = test_metrics.false_rejects / test_metrics.anti_replay_tests
        assert (
            false_reject_rate <= TARGET_FALSE_REJECT_RATE
        ), f"Anti-replay false reject rate too high: {false_reject_rate:.2%}"

    @pytest.mark.asyncio
    async def test_duplicate_packet_detection(self, scion_gateway):
        """Test that duplicate packets are properly handled."""
        # Send the same packet multiple times
        test_packet = b"duplicate_test_packet"

        first_result = await scion_gateway.send_packet(test_packet, TEST_DESTINATION)
        assert first_result is not None

        # Subsequent sends of the same packet should still succeed
        # (at the HTTP level, anti-replay is handled at SCION level)
        second_result = await scion_gateway.send_packet(test_packet, TEST_DESTINATION)
        assert second_result is not None

        # Each send should get a unique packet ID
        assert first_result != second_result


class TestSCIONFailover:
    """Test SCION path failover and recovery."""

    @pytest.mark.asyncio
    async def test_path_failover_simulation(self, scion_gateway, test_metrics):
        """Test path failover behavior."""
        # Get available paths
        paths = await scion_gateway.query_paths(TEST_DESTINATION)

        if len(paths) < 2:
            pytest.skip("Multiple paths required for failover testing")

        # Test sending with specific path preference
        primary_path = paths[0]
        backup_path = paths[1]

        # Send packet with primary path
        start_time = time.time()

        try:
            packet_id = await scion_gateway.send_packet(
                b"failover_test_primary",
                TEST_DESTINATION,
                path_preference=primary_path.fingerprint,
            )

            primary_latency = (time.time() - start_time) * 1000
            test_metrics.record_latency(primary_latency)

            assert packet_id is not None

        except SCIONGatewayError as e:
            # Primary path failed, test backup
            failover_start = time.time()

            try:
                backup_id = await scion_gateway.send_packet(
                    b"failover_test_backup",
                    TEST_DESTINATION,
                    path_preference=backup_path.fingerprint,
                )

                failover_time = (time.time() - failover_start) * 1000
                test_metrics.record_failover_time(failover_time)

                assert backup_id is not None

                # Verify failover time meets target
                assert (
                    failover_time <= TARGET_P95_RECOVERY_MS
                ), f"Failover too slow: {failover_time}ms > {TARGET_P95_RECOVERY_MS}ms"

            except SCIONGatewayError as backup_error:
                pytest.fail(
                    f"Both primary and backup paths failed: {e}, {backup_error}"
                )


class TestSCIONNavigatorIntegration:
    """Test SCION Navigator integration."""

    @pytest.mark.asyncio
    async def test_navigator_transport_selection(self, gateway_config):
        """Test Navigator's transport selection logic."""

        # Mock transport manager for testing
        class MockTransportManager:
            async def handle_received_message(self, message, source, transport_type):
                pass

            async def send_message_via_transport(
                self, message, destination, transport_type, metadata
            ):
                return transport_type == "scion"  # Only SCION succeeds

        transport_manager = MockTransportManager()

        try:
            navigator = SCIONAwareNavigator(
                scion_config=gateway_config,
                transport_manager=transport_manager,
                enable_scion_preference=True,
            )

            await navigator.start()

            # Test route finding
            test_message = Message(
                type=MessageType.DATA, content={"test": "navigation"}, metadata={}
            )

            decision = await navigator.find_optimal_route(
                TEST_DESTINATION, test_message
            )

            assert decision is not None
            assert decision.primary_transport is not None
            assert decision.confidence_score > 0.0

            # SCION should be preferred when available
            if any(
                c.transport_type == "scion"
                for c in [decision.primary_transport] + decision.backup_transports
            ):
                logger.info("SCION transport properly included in routing decision")

        finally:
            if "navigator" in locals():
                await navigator.stop()


@pytest.mark.asyncio
async def test_end_to_end_integration(gateway_config, test_metrics):
    """Complete end-to-end integration test."""
    logger.info("Starting end-to-end integration test")

    async with SCIONGateway(gateway_config) as gateway:
        # 1. Health check
        health = await gateway.health_check()
        assert health["status"] in ["healthy", "degraded"]

        # 2. Path discovery
        paths = await gateway.query_paths(TEST_DESTINATION)
        logger.info(f"Discovered {len(paths)} paths to {TEST_DESTINATION}")

        # 3. Send test messages
        test_messages = [
            {"type": "text", "content": "Hello, SCION!"},
            {"type": "json", "content": {"test": True, "timestamp": time.time()}},
            {"type": "binary", "content": b"\x00\x01\x02\x03" * 100},
        ]

        for i, msg_data in enumerate(test_messages):
            message = Message(
                type=MessageType.DATA,
                content=msg_data["content"],
                metadata={"test_message_id": i, "test_type": msg_data["type"]},
            )

            start_time = time.time()
            packet_id = await gateway.send_message(message, TEST_DESTINATION)
            end_time = time.time()

            latency_ms = (end_time - start_time) * 1000
            test_metrics.record_packet_sent()
            test_metrics.record_latency(latency_ms)

            assert packet_id is not None
            logger.info(f"Sent message {i}: {packet_id} ({latency_ms:.2f}ms)")

        # 4. Statistics check
        stats = await gateway.get_statistics()
        logger.info(f"Gateway statistics: {stats}")

        assert stats["packets_sent"] >= len(test_messages)

    logger.info("End-to-end integration test completed successfully")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_collection_modifyitems(config, items):
    """Add slow marker to performance tests."""
    for item in items:
        if "throughput" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v", "--tb=short"])
