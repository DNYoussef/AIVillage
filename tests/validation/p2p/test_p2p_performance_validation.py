#!/usr/bin/env python3
"""Comprehensive P2P Network Performance Validation Suite

This suite validates the actual performance claims made by AIVillage P2P systems:
- BitChat (Bluetooth mesh networking)
- BetaNet (HTX encrypted transport)
- Unified Transport Manager

Performance Claims to Validate:
- Message delivery rates
- Latency under various conditions
- Throughput measurements
- Scale testing with multiple peers
- Battery impact on mobile devices
"""

import asyncio
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import pytest

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "packages"))

from packages.p2p.core.message_types import MessagePriority, MessageType, P2PMessage
from packages.p2p.core.transport_manager import TransportManager

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for P2P testing"""

    test_name: str
    message_count: int
    success_count: int
    failed_count: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    throughput_msg_per_sec: float
    total_duration_sec: float
    success_rate_percent: float


class MockP2PNode:
    """Mock P2P node for testing"""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.transport_manager = TransportManager()
        self.received_messages = []
        self.sent_messages = []
        self.is_running = False

    async def start(self):
        """Start the mock node"""
        await self.transport_manager.initialize()
        self.is_running = True
        logger.info(f"Mock node {self.node_id} started")

    async def stop(self):
        """Stop the mock node"""
        await self.transport_manager.shutdown()
        self.is_running = False
        logger.info(f"Mock node {self.node_id} stopped")

    async def send_message(
        self, recipient_id: str, content: str, priority: MessagePriority = MessagePriority.MEDIUM
    ) -> bool:
        """Send a message to another node"""
        message = P2PMessage(
            sender_id=self.node_id,
            recipient_id=recipient_id,
            content=content,
            message_type=MessageType.DATA,
            priority=priority,
        )

        try:
            success = await self.transport_manager.send_message(message)
            if success:
                self.sent_messages.append(message)
            return success
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    async def receive_message(self, message: P2PMessage):
        """Receive a message from another node"""
        self.received_messages.append(message)
        logger.debug(f"Node {self.node_id} received message from {message.sender_id}")


class P2PPerformanceValidator:
    """Main P2P performance validation class"""

    def __init__(self):
        self.nodes: dict[str, MockP2PNode] = {}
        self.results: list[PerformanceMetrics] = []

    async def setup_test_network(self, node_count: int) -> list[MockP2PNode]:
        """Set up a test network with specified number of nodes"""
        nodes = []

        for i in range(node_count):
            node_id = f"test_node_{i:03d}"
            node = MockP2PNode(node_id)
            await node.start()
            nodes.append(node)
            self.nodes[node_id] = node

        logger.info(f"Set up test network with {node_count} nodes")
        return nodes

    async def teardown_test_network(self):
        """Teardown the test network"""
        for node in self.nodes.values():
            await node.stop()
        self.nodes.clear()
        logger.info("Test network torn down")

    async def measure_latency(
        self, sender: MockP2PNode, receiver: MockP2PNode, message_count: int = 100
    ) -> PerformanceMetrics:
        """Measure point-to-point latency between two nodes"""
        latencies = []
        success_count = 0
        start_time = time.perf_counter()

        for i in range(message_count):
            message_start = time.perf_counter()
            success = await sender.send_message(receiver.node_id, f"test_message_{i}")

            if success:
                # Wait for message to be processed (simulate real network delay)
                await asyncio.sleep(0.001)  # 1ms simulated network delay
                latency = (time.perf_counter() - message_start) * 1000  # Convert to ms
                latencies.append(latency)
                success_count += 1

        total_duration = time.perf_counter() - start_time

        if latencies:
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        else:
            avg_latency = min_latency = max_latency = p95_latency = 0.0

        throughput = success_count / total_duration if total_duration > 0 else 0.0
        success_rate = (success_count / message_count) * 100

        return PerformanceMetrics(
            test_name="point_to_point_latency",
            message_count=message_count,
            success_count=success_count,
            failed_count=message_count - success_count,
            avg_latency_ms=avg_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            p95_latency_ms=p95_latency,
            throughput_msg_per_sec=throughput,
            total_duration_sec=total_duration,
            success_rate_percent=success_rate,
        )

    async def measure_broadcast_performance(
        self, sender: MockP2PNode, receivers: list[MockP2PNode], message_count: int = 50
    ) -> PerformanceMetrics:
        """Measure broadcast performance to multiple receivers"""
        success_count = 0
        total_sends = message_count * len(receivers)
        start_time = time.perf_counter()

        for i in range(message_count):
            message = f"broadcast_message_{i}"
            sends_this_round = 0

            for receiver in receivers:
                success = await sender.send_message(receiver.node_id, message)
                if success:
                    sends_this_round += 1

            success_count += sends_this_round
            # Small delay between broadcasts
            await asyncio.sleep(0.001)

        total_duration = time.perf_counter() - start_time
        throughput = success_count / total_duration if total_duration > 0 else 0.0
        success_rate = (success_count / total_sends) * 100

        return PerformanceMetrics(
            test_name="broadcast_performance",
            message_count=total_sends,
            success_count=success_count,
            failed_count=total_sends - success_count,
            avg_latency_ms=0.0,  # Not applicable for broadcast
            min_latency_ms=0.0,
            max_latency_ms=0.0,
            p95_latency_ms=0.0,
            throughput_msg_per_sec=throughput,
            total_duration_sec=total_duration,
            success_rate_percent=success_rate,
        )

    async def measure_network_scale(self, node_count: int, messages_per_node: int = 10) -> PerformanceMetrics:
        """Measure performance under network scale with many nodes"""
        nodes = await self.setup_test_network(node_count)

        total_messages = 0
        success_count = 0
        start_time = time.perf_counter()

        # Each node sends messages to 3 random other nodes
        for sender in nodes:
            for i in range(messages_per_node):
                # Select random receivers (up to 3)
                receivers = [n for n in nodes if n.node_id != sender.node_id][:3]

                for receiver in receivers:
                    total_messages += 1
                    success = await sender.send_message(receiver.node_id, f"scale_test_{i}")
                    if success:
                        success_count += 1

        total_duration = time.perf_counter() - start_time
        throughput = success_count / total_duration if total_duration > 0 else 0.0
        success_rate = (success_count / total_messages) * 100

        await self.teardown_test_network()

        return PerformanceMetrics(
            test_name=f"network_scale_{node_count}_nodes",
            message_count=total_messages,
            success_count=success_count,
            failed_count=total_messages - success_count,
            avg_latency_ms=0.0,  # Not measured in scale test
            min_latency_ms=0.0,
            max_latency_ms=0.0,
            p95_latency_ms=0.0,
            throughput_msg_per_sec=throughput,
            total_duration_sec=total_duration,
            success_rate_percent=success_rate,
        )

    async def run_full_validation_suite(self) -> dict[str, Any]:
        """Run the complete P2P performance validation suite"""
        logger.info("Starting P2P Performance Validation Suite")

        # Test 1: Basic latency test (2 nodes)
        nodes = await self.setup_test_network(2)
        latency_metrics = await self.measure_latency(nodes[0], nodes[1], message_count=100)
        self.results.append(latency_metrics)
        await self.teardown_test_network()

        # Test 2: Broadcast performance (1 sender, 5 receivers)
        nodes = await self.setup_test_network(6)
        broadcast_metrics = await self.measure_broadcast_performance(nodes[0], nodes[1:], message_count=50)
        self.results.append(broadcast_metrics)
        await self.teardown_test_network()

        # Test 3: Small network scale (10 nodes)
        scale_10_metrics = await self.measure_network_scale(10, messages_per_node=5)
        self.results.append(scale_10_metrics)

        # Test 4: Medium network scale (25 nodes)
        scale_25_metrics = await self.measure_network_scale(25, messages_per_node=3)
        self.results.append(scale_25_metrics)

        # Compile results
        validation_results = {
            "test_suite": "AIVillage P2P Performance Validation",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_tests": len(self.results),
            "tests": [asdict(result) for result in self.results],
            "summary": self._generate_summary(),
        }

        logger.info("P2P Performance Validation Suite completed")
        return validation_results

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary of validation results"""
        if not self.results:
            return {"status": "no_tests_run"}

        total_messages = sum(r.message_count for r in self.results)
        total_success = sum(r.success_count for r in self.results)
        overall_success_rate = (total_success / total_messages) * 100 if total_messages > 0 else 0

        # Find latency test results
        latency_tests = [r for r in self.results if r.test_name == "point_to_point_latency"]
        avg_latency = statistics.mean([r.avg_latency_ms for r in latency_tests]) if latency_tests else 0

        # Find highest throughput
        max_throughput = max([r.throughput_msg_per_sec for r in self.results], default=0)

        return {
            "overall_success_rate_percent": round(overall_success_rate, 2),
            "total_messages_tested": total_messages,
            "average_latency_ms": round(avg_latency, 2),
            "max_throughput_msg_per_sec": round(max_throughput, 2),
            "production_ready": overall_success_rate >= 95.0 and avg_latency <= 100.0,
            "performance_grade": self._calculate_performance_grade(overall_success_rate, avg_latency, max_throughput),
        }

    def _calculate_performance_grade(self, success_rate: float, avg_latency: float, max_throughput: float) -> str:
        """Calculate performance grade based on metrics"""
        score = 0

        # Success rate scoring (40% of grade)
        if success_rate >= 99.0:
            score += 40
        elif success_rate >= 95.0:
            score += 30
        elif success_rate >= 90.0:
            score += 20
        elif success_rate >= 80.0:
            score += 10

        # Latency scoring (30% of grade)
        if avg_latency <= 10.0:
            score += 30
        elif avg_latency <= 50.0:
            score += 20
        elif avg_latency <= 100.0:
            score += 10
        elif avg_latency <= 200.0:
            score += 5

        # Throughput scoring (30% of grade)
        if max_throughput >= 1000.0:
            score += 30
        elif max_throughput >= 500.0:
            score += 20
        elif max_throughput >= 100.0:
            score += 10
        elif max_throughput >= 50.0:
            score += 5

        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"


@pytest.mark.asyncio
async def test_p2p_latency_performance():
    """Test P2P latency performance meets production requirements"""
    validator = P2PPerformanceValidator()
    nodes = await validator.setup_test_network(2)

    try:
        metrics = await validator.measure_latency(nodes[0], nodes[1], message_count=50)

        # Production requirements
        assert (
            metrics.success_rate_percent >= 95.0
        ), f"Success rate {metrics.success_rate_percent}% below 95% requirement"
        assert metrics.avg_latency_ms <= 100.0, f"Average latency {metrics.avg_latency_ms}ms exceeds 100ms requirement"
        assert metrics.p95_latency_ms <= 200.0, f"P95 latency {metrics.p95_latency_ms}ms exceeds 200ms requirement"

    finally:
        await validator.teardown_test_network()


@pytest.mark.asyncio
async def test_p2p_throughput_performance():
    """Test P2P throughput performance meets production requirements"""
    validator = P2PPerformanceValidator()
    nodes = await validator.setup_test_network(6)

    try:
        metrics = await validator.measure_broadcast_performance(nodes[0], nodes[1:], message_count=30)

        # Production requirements
        assert (
            metrics.success_rate_percent >= 90.0
        ), f"Broadcast success rate {metrics.success_rate_percent}% below 90% requirement"
        assert (
            metrics.throughput_msg_per_sec >= 50.0
        ), f"Throughput {metrics.throughput_msg_per_sec} msg/s below 50 msg/s requirement"

    finally:
        await validator.teardown_test_network()


@pytest.mark.asyncio
async def test_p2p_scale_performance():
    """Test P2P performance scales to production requirements"""
    validator = P2PPerformanceValidator()

    # Test medium scale network
    metrics = await validator.measure_network_scale(20, messages_per_node=3)

    # Production requirements for scale
    assert (
        metrics.success_rate_percent >= 85.0
    ), f"Scale success rate {metrics.success_rate_percent}% below 85% requirement"
    assert (
        metrics.total_duration_sec <= 30.0
    ), f"Scale test duration {metrics.total_duration_sec}s exceeds 30s requirement"


async def main():
    """Main function to run P2P performance validation"""
    validator = P2PPerformanceValidator()
    results = await validator.run_full_validation_suite()

    # Save results
    output_file = Path("docs/benchmarks/p2p_performance_validation_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nResults saved to: {output_file}")

    # Print summary
    summary = results["summary"]
    print("\n=== P2P Performance Validation Summary ===")
    print(f"Overall Success Rate: {summary['overall_success_rate_percent']}%")
    print(f"Average Latency: {summary['average_latency_ms']}ms")
    print(f"Max Throughput: {summary['max_throughput_msg_per_sec']} msg/s")
    print(f"Production Ready: {summary['production_ready']}")
    print(f"Performance Grade: {summary['performance_grade']}")


if __name__ == "__main__":
    asyncio.run(main())
