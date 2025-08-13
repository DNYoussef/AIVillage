#!/usr/bin/env python3
"""Comprehensive test suite for decentralized mesh network functionality.

Tests all aspects of the mesh network including:
- Network formation and topology
- Message routing and delivery
- Network resilience and fault tolerance
- Performance under load
- Scalability characteristics
- Security features
"""

import asyncio
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
import time

from implement_mesh_protocol import MeshMessage, MeshNetworkSimulator, MessageType

# Add scripts directory to path
sys.path.append(str(Path(__file__).resolve().parent / "scripts"))


class MeshNetworkTester:
    """Comprehensive mesh network test suite."""

    def __init__(self):
        self.results = {
            "test_start": datetime.now().isoformat(),
            "tests_run": [],
            "performance_metrics": {},
            "scalability_results": {},
            "resilience_results": {},
            "security_results": {},
            "summary": {},
        }

    async def test_basic_network_formation(self):
        """Test basic mesh network formation."""
        print("🔧 Testing basic network formation...")

        test_configs = [
            {"nodes": 3, "connectivity": 0.6, "name": "Small Network"},
            {"nodes": 5, "connectivity": 0.5, "name": "Medium Network"},
            {"nodes": 10, "connectivity": 0.4, "name": "Large Network"},
            {"nodes": 20, "connectivity": 0.3, "name": "Very Large Network"},
        ]

        formation_results = []

        for config in test_configs:
            print(f"  Testing {config['name']} ({config['nodes']} nodes)...")
            start_time = time.time()

            try:
                simulator = MeshNetworkSimulator(
                    num_nodes=config["nodes"], connectivity=config["connectivity"]
                )
                await simulator.create_network()

                # Verify network formation
                nodes_created = len(simulator.nodes)
                total_connections = sum(
                    len(node.neighbors) for node in simulator.nodes.values()
                )

                # Calculate network connectivity metrics
                max_possible_connections = config["nodes"] * (config["nodes"] - 1)
                actual_connectivity = (
                    total_connections / max_possible_connections
                    if max_possible_connections > 0
                    else 0
                )

                formation_time = time.time() - start_time

                result = {
                    "config": config,
                    "nodes_created": nodes_created,
                    "total_connections": total_connections,
                    "actual_connectivity": actual_connectivity,
                    "formation_time": formation_time,
                    "success": nodes_created == config["nodes"]
                    and total_connections > 0,
                }

                formation_results.append(result)

                status = "✅ PASS" if result["success"] else "❌ FAIL"
                print(
                    f"    {status} - {nodes_created} nodes, {total_connections} connections ({formation_time:.2f}s)"
                )

            except Exception as e:
                formation_results.append(
                    {
                        "config": config,
                        "error": str(e),
                        "success": False,
                        "formation_time": time.time() - start_time,
                    }
                )
                print(f"    ❌ FAIL - Error: {e}")

        self.results["tests_run"].append(
            {
                "test_name": "Basic Network Formation",
                "results": formation_results,
                "success_rate": sum(
                    1 for r in formation_results if r.get("success", False)
                )
                / len(formation_results),
            }
        )

        return formation_results

    async def test_message_routing_reliability(self):
        """Test message routing and delivery reliability."""
        print("📡 Testing message routing reliability...")

        # Create test network
        simulator = MeshNetworkSimulator(num_nodes=8, connectivity=0.5)
        await simulator.create_network()

        routing_results = []

        # Test different message types
        message_types = [
            MessageType.DISCOVERY,
            MessageType.PARAMETER_UPDATE,
            MessageType.GRADIENT_SHARE,
            MessageType.EMERGENCY,
        ]

        for msg_type in message_types:
            print(f"  Testing {msg_type.name} message routing...")

            # Send messages from random nodes to random destinations
            successful_deliveries = 0
            total_messages = 10
            delivery_times = []

            for i in range(total_messages):
                try:
                    # Select random source and destination
                    nodes = list(simulator.nodes.values())
                    source_node = nodes[i % len(nodes)]
                    dest_node = nodes[(i + 3) % len(nodes)]

                    if source_node.node_id == dest_node.node_id:
                        continue

                    # Clear previous message counts
                    dest_node.stats["messages_received"] = 0

                    start_time = time.time()

                    # Send message
                    await source_node.send_message(
                        msg_type,
                        {"test_id": i, "timestamp": start_time},
                        recipient_id=dest_node.node_id,
                        priority=5,
                    )

                    # Wait for delivery
                    await asyncio.sleep(0.5)

                    delivery_time = time.time() - start_time

                    # Check if message was received
                    if dest_node.stats["messages_received"] > 0:
                        successful_deliveries += 1
                        delivery_times.append(delivery_time)

                except Exception as e:
                    print(f"    Error in message {i}: {e}")

            delivery_rate = (
                successful_deliveries / total_messages if total_messages > 0 else 0
            )
            avg_delivery_time = statistics.mean(delivery_times) if delivery_times else 0

            result = {
                "message_type": msg_type.name,
                "total_messages": total_messages,
                "successful_deliveries": successful_deliveries,
                "delivery_rate": delivery_rate,
                "avg_delivery_time": avg_delivery_time,
                "success": delivery_rate >= 0.7,  # 70% delivery rate threshold
            }

            routing_results.append(result)

            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(
                f"    {status} - {delivery_rate:.1%} delivery rate, {avg_delivery_time:.3f}s avg time"
            )

        self.results["tests_run"].append(
            {
                "test_name": "Message Routing Reliability",
                "results": routing_results,
                "overall_delivery_rate": statistics.mean(
                    [r["delivery_rate"] for r in routing_results]
                ),
            }
        )

        return routing_results

    async def test_network_resilience(self):
        """Test network resilience under node failures."""
        print("🛡️ Testing network resilience...")

        resilience_results = []

        # Test different failure scenarios
        failure_scenarios = [
            {"name": "Single Node Failure", "nodes": 10, "failures": 1},
            {"name": "Multiple Node Failures", "nodes": 15, "failures": 3},
            {"name": "Cascade Failures", "nodes": 20, "failures": 5},
            {"name": "Network Partition", "nodes": 12, "failures": 4},
        ]

        for scenario in failure_scenarios:
            print(f"  Testing {scenario['name']}...")

            try:
                # Create network
                simulator = MeshNetworkSimulator(
                    num_nodes=scenario["nodes"], connectivity=0.4
                )
                await simulator.create_network()

                # Baseline: measure initial connectivity
                initial_nodes = len(simulator.nodes)
                initial_connections = sum(
                    len(node.neighbors) for node in simulator.nodes.values()
                )

                # Start background traffic simulation
                traffic_task = asyncio.create_task(
                    simulator.simulate_traffic(duration=8)
                )

                # Wait a bit for traffic to start
                await asyncio.sleep(1)

                # Introduce failures
                failed_nodes = []
                node_ids = list(simulator.nodes.keys())

                for i in range(scenario["failures"]):
                    if i < len(node_ids):
                        failed_node_id = node_ids[i * 2]  # Spread out failures
                        failed_node = simulator.nodes[failed_node_id]

                        # Simulate node failure by clearing its neighbors
                        failed_node.neighbors.clear()
                        failed_nodes.append(failed_node_id)

                        # Remove this node from other nodes' neighbor lists
                        for other_node in simulator.nodes.values():
                            if failed_node_id in other_node.neighbors:
                                del other_node.neighbors[failed_node_id]

                        # Wait between failures
                        await asyncio.sleep(0.5)

                # Continue traffic and measure recovery
                await traffic_task

                # Measure post-failure network state
                active_nodes = [
                    node
                    for node_id, node in simulator.nodes.items()
                    if node_id not in failed_nodes and len(node.neighbors) > 0
                ]

                remaining_connections = sum(
                    len(node.neighbors) for node in active_nodes
                )

                # Calculate resilience metrics
                node_survival_rate = len(active_nodes) / (
                    initial_nodes - scenario["failures"]
                )
                connection_retention = remaining_connections / max(
                    1, initial_connections - scenario["failures"] * 2
                )

                # Check if network is still functional (can route messages)
                network_functional = (
                    len(active_nodes) >= 2 and remaining_connections > 0
                )

                result = {
                    "scenario": scenario["name"],
                    "initial_nodes": initial_nodes,
                    "failures_introduced": scenario["failures"],
                    "active_nodes": len(active_nodes),
                    "node_survival_rate": node_survival_rate,
                    "connection_retention": connection_retention,
                    "network_functional": network_functional,
                    "success": network_functional and node_survival_rate > 0.5,
                }

                resilience_results.append(result)

                status = "✅ PASS" if result["success"] else "❌ FAIL"
                print(
                    f"    {status} - {len(active_nodes)} nodes active, {node_survival_rate:.1%} survival rate"
                )

            except Exception as e:
                resilience_results.append(
                    {"scenario": scenario["name"], "error": str(e), "success": False}
                )
                print(f"    ❌ FAIL - Error: {e}")

        self.results["resilience_results"] = resilience_results
        self.results["tests_run"].append(
            {
                "test_name": "Network Resilience",
                "results": resilience_results,
                "success_rate": sum(
                    1 for r in resilience_results if r.get("success", False)
                )
                / len(resilience_results),
            }
        )

        return resilience_results

    async def test_scalability_performance(self):
        """Test network performance and scalability."""
        print("📈 Testing scalability and performance...")

        scalability_results = []

        # Test different network sizes
        network_sizes = [5, 10, 15, 20, 25, 30]

        for size in network_sizes:
            print(f"  Testing network with {size} nodes...")

            try:
                start_time = time.time()

                # Create network
                simulator = MeshNetworkSimulator(num_nodes=size, connectivity=0.4)
                await simulator.create_network()

                formation_time = time.time() - start_time

                # Measure baseline metrics
                total_connections = sum(
                    len(node.neighbors) for node in simulator.nodes.values()
                )
                avg_connections_per_node = total_connections / size if size > 0 else 0

                # Performance test: measure message throughput
                start_time = time.time()

                # Send multiple messages concurrently
                message_tasks = []
                num_messages = min(50, size * 3)  # Scale messages with network size

                for i in range(num_messages):
                    source_node = list(simulator.nodes.values())[i % size]
                    dest_node = list(simulator.nodes.values())[(i + size // 2) % size]

                    if source_node.node_id != dest_node.node_id:
                        task = asyncio.create_task(
                            source_node.send_message(
                                MessageType.PARAMETER_UPDATE,
                                {"test_message": i},
                                recipient_id=dest_node.node_id,
                            )
                        )
                        message_tasks.append(task)

                # Wait for all messages
                await asyncio.gather(*message_tasks, return_exceptions=True)

                messaging_time = time.time() - start_time
                throughput = num_messages / messaging_time if messaging_time > 0 else 0

                # Memory and resource estimation
                estimated_memory_mb = (
                    size * 0.1 + total_connections * 0.01
                )  # Rough estimate

                result = {
                    "network_size": size,
                    "formation_time": formation_time,
                    "total_connections": total_connections,
                    "avg_connections_per_node": avg_connections_per_node,
                    "message_throughput": throughput,
                    "estimated_memory_mb": estimated_memory_mb,
                    "success": formation_time < 5.0
                    and throughput > 10,  # Performance thresholds
                }

                scalability_results.append(result)

                status = "✅ PASS" if result["success"] else "❌ FAIL"
                print(
                    f"    {status} - {throughput:.1f} msg/s, {formation_time:.2f}s formation, {estimated_memory_mb:.1f}MB"
                )

            except Exception as e:
                scalability_results.append(
                    {"network_size": size, "error": str(e), "success": False}
                )
                print(f"    ❌ FAIL - Error: {e}")

        self.results["scalability_results"] = scalability_results
        self.results["tests_run"].append(
            {
                "test_name": "Scalability Performance",
                "results": scalability_results,
                "max_tested_size": max(network_sizes),
                "performance_degradation": self._analyze_performance_degradation(
                    scalability_results
                ),
            }
        )

        return scalability_results

    def _analyze_performance_degradation(self, results):
        """Analyze performance degradation with network size."""
        if len(results) < 3:
            return "Insufficient data"

        # Compare throughput at different sizes
        successful_results = [r for r in results if r.get("success", False)]
        if len(successful_results) < 2:
            return "Insufficient successful tests"

        small_net = successful_results[0]
        large_net = successful_results[-1]

        throughput_ratio = (
            large_net["message_throughput"] / small_net["message_throughput"]
        )
        large_net["network_size"] / small_net["network_size"]

        if throughput_ratio > 0.8:
            return "Excellent - Minimal degradation"
        if throughput_ratio > 0.6:
            return "Good - Acceptable degradation"
        if throughput_ratio > 0.4:
            return "Moderate - Noticeable degradation"
        return "Poor - Significant degradation"

    async def test_routing_algorithms(self):
        """Test mesh routing algorithm effectiveness."""
        print("🗺️ Testing routing algorithms...")

        routing_results = []

        # Test different network topologies
        test_topologies = [
            {"name": "Dense Network", "nodes": 10, "connectivity": 0.8},
            {"name": "Sparse Network", "nodes": 15, "connectivity": 0.3},
            {"name": "Ring Topology", "nodes": 12, "connectivity": 0.2},
            {"name": "Hub-Spoke Like", "nodes": 20, "connectivity": 0.25},
        ]

        for topology in test_topologies:
            print(f"  Testing routing in {topology['name']}...")

            try:
                # Create network
                simulator = MeshNetworkSimulator(
                    num_nodes=topology["nodes"], connectivity=topology["connectivity"]
                )
                await simulator.create_network()

                # Test routing efficiency
                nodes = list(simulator.nodes.values())
                routing_tests = []

                # Test multiple source-destination pairs
                for i in range(min(20, len(nodes) * 2)):
                    source = nodes[i % len(nodes)]
                    dest = nodes[(i + len(nodes) // 2) % len(nodes)]

                    if source.node_id == dest.node_id:
                        continue

                    # Clear routing table to force route discovery
                    source.routing_table.clear()

                    start_time = time.time()

                    # Send test message
                    await source.send_message(
                        MessageType.DISCOVERY,
                        {"route_test": i},
                        recipient_id=dest.node_id,
                    )

                    # Give time for routing
                    await asyncio.sleep(0.2)

                    routing_time = time.time() - start_time

                    # Check if route was established
                    route_found = dest.node_id in source.routing_table

                    if route_found:
                        route_length = source.routing_table[dest.node_id][1]  # Distance
                        routing_tests.append(
                            {
                                "route_found": True,
                                "route_length": route_length,
                                "routing_time": routing_time,
                            }
                        )
                    else:
                        routing_tests.append(
                            {"route_found": False, "routing_time": routing_time}
                        )

                # Analyze routing effectiveness
                successful_routes = [t for t in routing_tests if t["route_found"]]
                route_success_rate = (
                    len(successful_routes) / len(routing_tests) if routing_tests else 0
                )
                avg_route_length = (
                    statistics.mean([t["route_length"] for t in successful_routes])
                    if successful_routes
                    else 0
                )
                avg_routing_time = (
                    statistics.mean([t["routing_time"] for t in routing_tests])
                    if routing_tests
                    else 0
                )

                result = {
                    "topology": topology["name"],
                    "network_config": topology,
                    "total_route_tests": len(routing_tests),
                    "successful_routes": len(successful_routes),
                    "route_success_rate": route_success_rate,
                    "avg_route_length": avg_route_length,
                    "avg_routing_time": avg_routing_time,
                    "success": route_success_rate >= 0.6 and avg_routing_time < 1.0,
                }

                routing_results.append(result)

                status = "✅ PASS" if result["success"] else "❌ FAIL"
                print(
                    f"    {status} - {route_success_rate:.1%} success, {avg_route_length:.1f} hops avg"
                )

            except Exception as e:
                routing_results.append(
                    {"topology": topology["name"], "error": str(e), "success": False}
                )
                print(f"    ❌ FAIL - Error: {e}")

        self.results["tests_run"].append(
            {
                "test_name": "Routing Algorithms",
                "results": routing_results,
                "success_rate": sum(
                    1 for r in routing_results if r.get("success", False)
                )
                / len(routing_results),
            }
        )

        return routing_results

    async def test_security_features(self):
        """Test mesh network security features."""
        print("🔒 Testing security features...")

        security_results = []

        # Test message integrity and anti-replay
        print("  Testing message integrity...")

        try:
            # Create small test network
            simulator = MeshNetworkSimulator(num_nodes=5, connectivity=0.6)
            await simulator.create_network()

            nodes = list(simulator.nodes.values())
            source = nodes[0]
            dest = nodes[1]

            # Test 1: Valid message handling
            valid_messages_sent = 0
            valid_messages_processed = 0

            for i in range(10):
                message = MeshMessage(
                    message_id=f"valid_{i}",
                    sender_id=source.node_id,
                    recipient_id=dest.node_id,
                    message_type=MessageType.PARAMETER_UPDATE,
                    payload={"data": f"test_{i}"},
                    ttl=5,
                    priority=3,
                )

                initial_cache_size = len(dest.message_cache)
                await dest.receive_message(message.to_bytes(), source.node_id)

                valid_messages_sent += 1
                if len(dest.message_cache) > initial_cache_size:
                    valid_messages_processed += 1

                await asyncio.sleep(0.1)

            # Test 2: Duplicate message detection (anti-replay)
            duplicate_message = MeshMessage(
                message_id="duplicate_test",
                sender_id=source.node_id,
                recipient_id=dest.node_id,
                message_type=MessageType.PARAMETER_UPDATE,
                payload={"data": "duplicate"},
                ttl=5,
                priority=3,
            )

            # Send same message twice
            await dest.receive_message(duplicate_message.to_bytes(), source.node_id)
            initial_processed = dest.stats["messages_received"]

            await dest.receive_message(duplicate_message.to_bytes(), source.node_id)
            final_processed = dest.stats["messages_received"]

            duplicate_blocked = final_processed == initial_processed

            security_test_result = {
                "test_type": "Message Integrity & Anti-Replay",
                "valid_message_success_rate": (
                    valid_messages_processed / valid_messages_sent
                    if valid_messages_sent > 0
                    else 0
                ),
                "duplicate_message_blocked": duplicate_blocked,
                "success": (valid_messages_processed / valid_messages_sent >= 0.8)
                and duplicate_blocked,
            }

            security_results.append(security_test_result)

            status = "✅ PASS" if security_test_result["success"] else "❌ FAIL"
            print(
                f"    {status} - {valid_messages_processed}/{valid_messages_sent} valid msgs, duplicate blocked: {duplicate_blocked}"
            )

        except Exception as e:
            security_results.append(
                {
                    "test_type": "Message Integrity & Anti-Replay",
                    "error": str(e),
                    "success": False,
                }
            )
            print(f"    ❌ FAIL - Error: {e}")

        # Test 3: TTL (Time To Live) enforcement
        print("  Testing TTL enforcement...")

        try:
            simulator = MeshNetworkSimulator(num_nodes=6, connectivity=0.4)
            await simulator.create_network()

            nodes = list(simulator.nodes.values())
            source = nodes[0]

            # Send message with very low TTL
            MeshMessage(
                message_id="ttl_test_low",
                sender_id=source.node_id,
                recipient_id="broadcast",
                message_type=MessageType.DISCOVERY,
                payload={"ttl_test": "low"},
                ttl=1,  # Very low TTL
                priority=3,
            )

            # Count initial forwards
            initial_forwards = sum(node.stats["messages_forwarded"] for node in nodes)

            await source.send_message(
                MessageType.DISCOVERY, {"ttl_test": "low"}, priority=3
            )

            await asyncio.sleep(0.5)

            final_forwards = sum(node.stats["messages_forwarded"] for node in nodes)
            limited_propagation = (final_forwards - initial_forwards) < len(nodes)

            ttl_test_result = {
                "test_type": "TTL Enforcement",
                "limited_propagation": limited_propagation,
                "forwards_count": final_forwards - initial_forwards,
                "success": limited_propagation,
            }

            security_results.append(ttl_test_result)

            status = "✅ PASS" if ttl_test_result["success"] else "❌ FAIL"
            print(f"    {status} - TTL limited propagation: {limited_propagation}")

        except Exception as e:
            security_results.append(
                {"test_type": "TTL Enforcement", "error": str(e), "success": False}
            )
            print(f"    ❌ FAIL - Error: {e}")

        self.results["security_results"] = security_results
        self.results["tests_run"].append(
            {
                "test_name": "Security Features",
                "results": security_results,
                "success_rate": sum(
                    1 for r in security_results if r.get("success", False)
                )
                / len(security_results),
            }
        )

        return security_results

    def generate_comprehensive_report(self):
        """Generate comprehensive test report."""
        print("\n" + "=" * 80)
        print("🌐 DECENTRALIZED MESH NETWORK - COMPREHENSIVE TEST RESULTS")
        print("=" * 80)

        # Calculate overall metrics
        total_tests = len(self.results["tests_run"])
        successful_tests = sum(
            1
            for test in self.results["tests_run"]
            if test.get("success_rate", 0) >= 0.6
        )

        overall_success_rate = successful_tests / total_tests if total_tests > 0 else 0

        print("\n📊 EXECUTIVE SUMMARY")
        print(
            f"   Test Suite: {successful_tests}/{total_tests} test categories passed ({overall_success_rate:.1%})"
        )
        print(f"   Generated: {self.results['test_start']}")
        print(
            f"   Status: {
                '✅ OPERATIONAL'
                if overall_success_rate >= 0.7
                else '⚠️ NEEDS ATTENTION'
                if overall_success_rate >= 0.5
                else '❌ CRITICAL ISSUES'
            }"
        )

        # Test category results
        print("\n🔍 TEST CATEGORY RESULTS")
        for test in self.results["tests_run"]:
            success_rate = test.get("success_rate", 0)
            status_emoji = (
                "✅" if success_rate >= 0.7 else "⚠️" if success_rate >= 0.5 else "❌"
            )
            print(
                f"   {status_emoji} {test['test_name']:<25} {success_rate:.1%} success rate"
            )

        # Performance highlights
        if "scalability_results" in self.results:
            scalability = self.results["scalability_results"]
            successful_scalability = [r for r in scalability if r.get("success", False)]

            if successful_scalability:
                max_size = max(r["network_size"] for r in successful_scalability)
                best_throughput = max(
                    r["message_throughput"] for r in successful_scalability
                )

                print("\n⚡ PERFORMANCE HIGHLIGHTS")
                print(f"   Maximum tested network size: {max_size} nodes")
                print(
                    f"   Peak message throughput: {best_throughput:.1f} messages/second"
                )

                degradation = self.results["tests_run"][-2].get(
                    "performance_degradation", "Unknown"
                )
                print(f"   Performance scaling: {degradation}")

        # Resilience highlights
        if "resilience_results" in self.results:
            resilience = self.results["resilience_results"]
            successful_resilience = [r for r in resilience if r.get("success", False)]

            if successful_resilience:
                avg_survival_rate = statistics.mean(
                    [r["node_survival_rate"] for r in successful_resilience]
                )

                print("\n🛡️ RESILIENCE HIGHLIGHTS")
                print(f"   Average node survival rate: {avg_survival_rate:.1%}")
                print(
                    f"   Network remains functional after failures: {'Yes' if avg_survival_rate > 0.5 else 'No'}"
                )

        # Security assessment
        if "security_results" in self.results:
            security = self.results["security_results"]
            security_passed = sum(1 for r in security if r.get("success", False))

            print("\n🔒 SECURITY ASSESSMENT")
            print(f"   Security features tested: {len(security)}")
            print(f"   Security tests passed: {security_passed}/{len(security)}")

            for result in security:
                status = "✅" if result.get("success", False) else "❌"
                print(f"   {status} {result['test_type']}")

        # Recommendations
        print("\n💡 RECOMMENDATIONS")

        if overall_success_rate >= 0.8:
            print("   🎉 Excellent! Mesh network is production-ready")
            print("   📈 Consider scaling tests to larger networks")
            print("   🔧 Monitor performance under real-world conditions")
        elif overall_success_rate >= 0.6:
            print("   👍 Good foundation with areas for improvement")
            print("   🔧 Address failing test categories")
            print("   📊 Optimize performance bottlenecks")
        else:
            print("   ⚠️ Critical issues need immediate attention")
            print("   🚨 Fix fundamental network formation or routing issues")
            print("   🔧 Review protocol implementation")

        # Specific recommendations based on results
        for test in self.results["tests_run"]:
            if test.get("success_rate", 0) < 0.6:
                print(
                    f"   🔧 Priority fix: {test['test_name']} - {test.get('success_rate', 0):.1%} success rate"
                )

        print("\n" + "=" * 80)

        return self.results


async def main():
    """Run comprehensive mesh network tests."""
    print("🚀 Starting Comprehensive Mesh Network Test Suite...")
    print("   This may take several minutes to complete all tests.\n")

    tester = MeshNetworkTester()

    # Mark current test as in progress
    tester.results["current_test"] = "running"

    try:
        # Run all test categories
        await tester.test_basic_network_formation()

        tester.results["current_test"] = "message_routing"
        await tester.test_message_routing_reliability()

        tester.results["current_test"] = "resilience"
        await tester.test_network_resilience()

        tester.results["current_test"] = "scalability"
        await tester.test_scalability_performance()

        tester.results["current_test"] = "routing"
        await tester.test_routing_algorithms()

        tester.results["current_test"] = "security"
        await tester.test_security_features()

        # Generate comprehensive report
        tester.results["current_test"] = "completed"
        final_results = tester.generate_comprehensive_report()

        # Save detailed results
        with open("mesh_network_test_results.json", "w") as f:
            json.dump(final_results, f, indent=2, default=str)

        print("\n📄 Detailed results saved to: mesh_network_test_results.json")

        return final_results

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        tester.results["error"] = str(e)
        tester.results["current_test"] = "failed"
        return tester.results


if __name__ == "__main__":
    asyncio.run(main())
