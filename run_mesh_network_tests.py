#!/usr/bin/env python3
"""Comprehensive mesh network testing suite."""

import asyncio
import time
import statistics
import json
from datetime import datetime
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.append(str(Path(__file__).resolve().parent / "scripts"))

from implement_mesh_protocol import (
    MeshNetworkSimulator,
    MeshProtocol,
    MessageType,
    MeshMessage,
    MeshNode
)

async def test_network_formation():
    """Test basic mesh network formation with different sizes."""
    print("Testing network formation...")

    results = []
    test_configs = [
        {"nodes": 3, "connectivity": 0.6, "name": "Small"},
        {"nodes": 5, "connectivity": 0.5, "name": "Medium"},
        {"nodes": 10, "connectivity": 0.4, "name": "Large"},
        {"nodes": 15, "connectivity": 0.3, "name": "Very Large"}
    ]

    for config in test_configs:
        print(f"  Testing {config['name']} network ({config['nodes']} nodes)...")
        start_time = time.time()

        try:
            simulator = MeshNetworkSimulator(
                num_nodes=config["nodes"],
                connectivity=config["connectivity"]
            )
            await simulator.create_network()

            nodes_created = len(simulator.nodes)
            total_connections = sum(len(node.neighbors) for node in simulator.nodes.values())
            formation_time = time.time() - start_time

            success = nodes_created == config["nodes"] and total_connections > 0

            result = {
                "config": config,
                "nodes_created": nodes_created,
                "total_connections": total_connections,
                "formation_time": formation_time,
                "success": success
            }

            results.append(result)

            status = "PASS" if success else "FAIL"
            print(f"    {status} - {nodes_created} nodes, {total_connections} connections ({formation_time:.2f}s)")

        except Exception as e:
            results.append({
                "config": config,
                "error": str(e),
                "success": False
            })
            print(f"    FAIL - Error: {e}")

    return results

async def test_message_routing():
    """Test message routing reliability across the network."""
    print("Testing message routing...")

    # Create test network
    simulator = MeshNetworkSimulator(num_nodes=8, connectivity=0.5)
    await simulator.create_network()

    results = []
    message_types = [MessageType.DISCOVERY, MessageType.PARAMETER_UPDATE, MessageType.GRADIENT_SHARE]

    for msg_type in message_types:
        print(f"  Testing {msg_type.name} routing...")

        successful_deliveries = 0
        total_messages = 10
        delivery_times = []

        for i in range(total_messages):
            try:
                nodes = list(simulator.nodes.values())
                source_node = nodes[i % len(nodes)]
                dest_node = nodes[(i + 3) % len(nodes)]

                if source_node.node_id == dest_node.node_id:
                    continue

                # Clear previous stats
                dest_node.stats["messages_received"] = 0
                start_time = time.time()

                # Send message
                await source_node.send_message(
                    msg_type,
                    {"test_id": i, "timestamp": start_time},
                    recipient_id=dest_node.node_id,
                    priority=5
                )

                # Wait for delivery
                await asyncio.sleep(0.3)
                delivery_time = time.time() - start_time

                # Check delivery
                if dest_node.stats["messages_received"] > 0:
                    successful_deliveries += 1
                    delivery_times.append(delivery_time)

            except Exception as e:
                print(f"    Error in message {i}: {e}")

        delivery_rate = successful_deliveries / total_messages if total_messages > 0 else 0
        avg_delivery_time = statistics.mean(delivery_times) if delivery_times else 0

        result = {
            "message_type": msg_type.name,
            "delivery_rate": delivery_rate,
            "avg_delivery_time": avg_delivery_time,
            "success": delivery_rate >= 0.6
        }

        results.append(result)

        status = "PASS" if result["success"] else "FAIL"
        print(f"    {status} - {delivery_rate:.1%} delivery rate, {avg_delivery_time:.3f}s avg")

    return results

async def test_network_resilience():
    """Test network resilience under node failures."""
    print("Testing network resilience...")

    results = []
    scenarios = [
        {"name": "Single Node Failure", "nodes": 10, "failures": 1},
        {"name": "Multiple Failures", "nodes": 15, "failures": 3},
        {"name": "Major Failures", "nodes": 20, "failures": 5}
    ]

    for scenario in scenarios:
        print(f"  Testing {scenario['name']}...")

        try:
            # Create network
            simulator = MeshNetworkSimulator(
                num_nodes=scenario["nodes"],
                connectivity=0.4
            )
            await simulator.create_network()

            initial_nodes = len(simulator.nodes)
            initial_connections = sum(len(node.neighbors) for node in simulator.nodes.values())

            # Start traffic simulation
            traffic_task = asyncio.create_task(simulator.simulate_traffic(duration=6))
            await asyncio.sleep(1)

            # Introduce failures
            failed_nodes = []
            node_ids = list(simulator.nodes.keys())

            for i in range(scenario["failures"]):
                if i < len(node_ids):
                    failed_node_id = node_ids[i * 2]
                    failed_node = simulator.nodes[failed_node_id]

                    # Simulate failure
                    failed_node.neighbors.clear()
                    failed_nodes.append(failed_node_id)

                    # Remove from other nodes
                    for other_node in simulator.nodes.values():
                        if failed_node_id in other_node.neighbors:
                            del other_node.neighbors[failed_node_id]

                    await asyncio.sleep(0.5)

            # Continue traffic
            await traffic_task

            # Measure recovery
            active_nodes = [
                node for node_id, node in simulator.nodes.items()
                if node_id not in failed_nodes and len(node.neighbors) > 0
            ]

            node_survival_rate = len(active_nodes) / (initial_nodes - scenario["failures"])
            network_functional = len(active_nodes) >= 2

            result = {
                "scenario": scenario["name"],
                "initial_nodes": initial_nodes,
                "failures": scenario["failures"],
                "active_nodes": len(active_nodes),
                "survival_rate": node_survival_rate,
                "functional": network_functional,
                "success": network_functional and node_survival_rate > 0.5
            }

            results.append(result)

            status = "PASS" if result["success"] else "FAIL"
            print(f"    {status} - {len(active_nodes)} nodes active, {node_survival_rate:.1%} survival")

        except Exception as e:
            results.append({
                "scenario": scenario["name"],
                "error": str(e),
                "success": False
            })
            print(f"    FAIL - Error: {e}")

    return results

async def test_scalability():
    """Test network scalability and performance."""
    print("Testing scalability...")

    results = []
    network_sizes = [5, 10, 15, 20, 25]

    for size in network_sizes:
        print(f"  Testing {size}-node network...")

        try:
            start_time = time.time()

            # Create network
            simulator = MeshNetworkSimulator(num_nodes=size, connectivity=0.4)
            await simulator.create_network()

            formation_time = time.time() - start_time
            total_connections = sum(len(node.neighbors) for node in simulator.nodes.values())

            # Performance test
            start_time = time.time()
            message_tasks = []
            num_messages = min(30, size * 2)

            for i in range(num_messages):
                source_node = list(simulator.nodes.values())[i % size]
                dest_node = list(simulator.nodes.values())[(i + size//2) % size]

                if source_node.node_id != dest_node.node_id:
                    task = asyncio.create_task(
                        source_node.send_message(
                            MessageType.PARAMETER_UPDATE,
                            {"test": i},
                            recipient_id=dest_node.node_id
                        )
                    )
                    message_tasks.append(task)

            await asyncio.gather(*message_tasks, return_exceptions=True)

            messaging_time = time.time() - start_time
            throughput = num_messages / messaging_time if messaging_time > 0 else 0

            result = {
                "network_size": size,
                "formation_time": formation_time,
                "total_connections": total_connections,
                "throughput": throughput,
                "success": formation_time < 5.0 and throughput > 5
            }

            results.append(result)

            status = "PASS" if result["success"] else "FAIL"
            print(f"    {status} - {throughput:.1f} msg/s, {formation_time:.2f}s formation")

        except Exception as e:
            results.append({
                "network_size": size,
                "error": str(e),
                "success": False
            })
            print(f"    FAIL - Error: {e}")

    return results

async def test_routing_efficiency():
    """Test routing algorithm efficiency."""
    print("Testing routing efficiency...")

    results = []
    topologies = [
        {"name": "Dense", "nodes": 10, "connectivity": 0.7},
        {"name": "Sparse", "nodes": 12, "connectivity": 0.3},
        {"name": "Medium", "nodes": 15, "connectivity": 0.5}
    ]

    for topology in topologies:
        print(f"  Testing {topology['name']} topology...")

        try:
            simulator = MeshNetworkSimulator(
                num_nodes=topology["nodes"],
                connectivity=topology["connectivity"]
            )
            await simulator.create_network()

            nodes = list(simulator.nodes.values())
            successful_routes = 0
            total_tests = min(15, len(nodes) * 2)
            route_lengths = []

            for i in range(total_tests):
                source = nodes[i % len(nodes)]
                dest = nodes[(i + len(nodes)//2) % len(nodes)]

                if source.node_id == dest.node_id:
                    continue

                # Clear routing table
                source.routing_table.clear()

                # Send discovery message
                await source.send_message(
                    MessageType.DISCOVERY,
                    {"route_test": i},
                    recipient_id=dest.node_id
                )

                await asyncio.sleep(0.2)

                # Check if route established
                if dest.node_id in source.routing_table:
                    successful_routes += 1
                    route_length = source.routing_table[dest.node_id][1]
                    route_lengths.append(route_length)

            success_rate = successful_routes / total_tests if total_tests > 0 else 0
            avg_route_length = statistics.mean(route_lengths) if route_lengths else 0

            result = {
                "topology": topology["name"],
                "success_rate": success_rate,
                "avg_route_length": avg_route_length,
                "success": success_rate >= 0.6
            }

            results.append(result)

            status = "PASS" if result["success"] else "FAIL"
            print(f"    {status} - {success_rate:.1%} route success, {avg_route_length:.1f} hops avg")

        except Exception as e:
            results.append({
                "topology": topology["name"],
                "error": str(e),
                "success": False
            })
            print(f"    FAIL - Error: {e}")

    return results

async def main():
    """Run comprehensive mesh network tests."""
    print("="*80)
    print("DECENTRALIZED MESH NETWORK - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print()

    start_time = time.time()
    all_results = {}

    # Run all test categories
    print("[1/5] Network Formation Tests")
    all_results["formation"] = await test_network_formation()

    print("\n[2/5] Message Routing Tests")
    all_results["routing"] = await test_message_routing()

    print("\n[3/5] Network Resilience Tests")
    all_results["resilience"] = await test_network_resilience()

    print("\n[4/5] Scalability Tests")
    all_results["scalability"] = await test_scalability()

    print("\n[5/5] Routing Efficiency Tests")
    all_results["routing_efficiency"] = await test_routing_efficiency()

    total_time = time.time() - start_time

    # Generate summary report
    print("\n" + "="*80)
    print("TEST SUMMARY REPORT")
    print("="*80)

    test_categories = [
        ("Network Formation", all_results["formation"]),
        ("Message Routing", all_results["routing"]),
        ("Network Resilience", all_results["resilience"]),
        ("Scalability", all_results["scalability"]),
        ("Routing Efficiency", all_results["routing_efficiency"])
    ]

    total_tests = 0
    passed_tests = 0

    print("\nDETAILED RESULTS:")
    for category_name, results in test_categories:
        category_passed = sum(1 for r in results if r.get("success", False))
        category_total = len(results)
        category_rate = category_passed / category_total if category_total > 0 else 0

        total_tests += category_total
        passed_tests += category_passed

        status = "PASS" if category_rate >= 0.6 else "FAIL"
        print(f"  {status:4} {category_name:<20} {category_passed}/{category_total} ({category_rate:.1%})")

    overall_rate = passed_tests / total_tests if total_tests > 0 else 0

    print(f"\nOVERALL RESULTS:")
    print(f"  Tests Passed: {passed_tests}/{total_tests} ({overall_rate:.1%})")
    print(f"  Total Duration: {total_time:.1f} seconds")

    # Performance highlights
    scalability_results = all_results["scalability"]
    successful_scalability = [r for r in scalability_results if r.get("success", False)]

    if successful_scalability:
        max_size = max(r["network_size"] for r in successful_scalability)
        best_throughput = max(r["throughput"] for r in successful_scalability)

        print(f"\nPERFORMANCE HIGHLIGHTS:")
        print(f"  Max network size tested: {max_size} nodes")
        print(f"  Peak throughput: {best_throughput:.1f} messages/second")

    # Resilience highlights
    resilience_results = all_results["resilience"]
    successful_resilience = [r for r in resilience_results if r.get("success", False)]

    if successful_resilience:
        avg_survival = statistics.mean([r["survival_rate"] for r in successful_resilience])
        print(f"  Average survival rate: {avg_survival:.1%}")

    # Status assessment
    print(f"\nSTATUS ASSESSMENT:")
    if overall_rate >= 0.8:
        print("  EXCELLENT - Mesh network is production-ready")
        print("  Recommendation: Deploy with confidence")
    elif overall_rate >= 0.6:
        print("  GOOD - Mesh network is functional with minor issues")
        print("  Recommendation: Address failing tests before production")
    elif overall_rate >= 0.4:
        print("  MODERATE - Significant issues need attention")
        print("  Recommendation: Fix critical failures before deployment")
    else:
        print("  POOR - Major problems detected")
        print("  Recommendation: Substantial debugging required")

    # Save results
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "total_duration": total_time,
        "overall_success_rate": overall_rate,
        "tests_passed": passed_tests,
        "tests_total": total_tests,
        "detailed_results": all_results
    }

    with open("mesh_network_test_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\nDetailed results saved to: mesh_network_test_results.json")
    print("="*80)

    return final_results

if __name__ == "__main__":
    asyncio.run(main())
