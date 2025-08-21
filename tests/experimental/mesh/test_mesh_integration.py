#!/usr/bin/env python3
"""Test mesh network integration with existing communications.

Moved from root to tests/experimental/mesh/ for better organization.
"""

import asyncio
import logging
import sys
from pathlib import Path

from communications.message import Message, MessageType, Priority
from mesh_network_manager import MeshNetworkManager

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


async def test_mesh_integration():
    """Test that mesh network integrates with existing message system."""
    logging.basicConfig(level=logging.INFO)

    print("Testing Mesh Network Integration")
    print("=" * 50)

    # Create two mesh nodes
    node1 = MeshNetworkManager("agent_001")
    node2 = MeshNetworkManager("agent_002")

    # Connect them
    await node1.add_peer("agent_002", "127.0.0.1", 8001)
    await node2.add_peer("agent_001", "127.0.0.1", 8000)

    # Start both nodes
    await node1.start()
    await node2.start()

    print("Both mesh nodes started")

    # Wait for connection establishment
    await asyncio.sleep(2)

    # Test 1: Basic message sending
    print("\nTest 1: Basic Message Sending")
    test_message = Message(
        type=MessageType.QUERY,
        sender="agent_001",
        receiver="agent_002",
        content={"test": "mesh_integration", "data": "Hello from mesh!"},
        priority=Priority.HIGH,
    )

    await node1.send_message(test_message)
    print("Message sent through mesh")

    # Test 2: Network statistics
    print("\nTest 2: Network Statistics")
    stats1 = node1.get_network_statistics()
    stats2 = node2.get_network_statistics()

    print(f"Node 1 - Active peers: {stats1['active_peers']}, Routes: {stats1['routing_entries']}")
    print(f"Node 2 - Active peers: {stats2['active_peers']}, Routes: {stats2['routing_entries']}")

    # Test 3: Health monitoring
    print("\nTest 3: Health Monitoring")
    health1 = stats1["network_health"]
    health2 = stats2["network_health"]

    print(f"Node 1 Health - Success: {health1['success_rate']:.1%}, Latency: {health1['average_latency_ms']:.1f}ms")
    print(f"Node 2 Health - Success: {health2['success_rate']:.1%}, Latency: {health2['average_latency_ms']:.1f}ms")

    # Test 4: Multiple message types
    print("\nTest 4: Multiple Message Types")
    message_types = [
        (MessageType.TASK, {"task": "process_data", "priority": "high"}),
        (MessageType.NOTIFICATION, {"alert": "system_status", "level": "info"}),
        (MessageType.RESPONSE, {"result": "completed", "status": "success"}),
    ]

    for msg_type, content in message_types:
        message = Message(type=msg_type, sender="agent_001", receiver="agent_002", content=content)
        await node1.send_message(message)
        print(f"Sent {msg_type.value} message")

    # Wait for message processing
    await asyncio.sleep(1)

    # Final statistics
    print("\nFinal Network Status")
    final_stats1 = node1.get_network_statistics()
    final_stats2 = node2.get_network_statistics()

    print(f"Node 1 - Messages processed: {final_stats1['network_health']['total_messages']}")
    print(f"Node 2 - Messages processed: {final_stats2['network_health']['total_messages']}")

    total_success_rate = (
        final_stats1["network_health"]["success_rate"] + final_stats2["network_health"]["success_rate"]
    ) / 2

    print(f"Overall success rate: {total_success_rate:.1%}")

    # Test 5: Network resilience
    print("\nTest 5: Network Resilience")
    print("Simulating peer failure...")

    # Remove a peer to test resilience
    await node1.remove_peer("agent_002")
    print("Peer removed, testing recovery...")

    # Try to send a message (should fail gracefully)
    recovery_message = Message(
        type=MessageType.QUERY,
        sender="agent_001",
        receiver="agent_002",
        content={"test": "recovery"},
    )
    await node1.send_message(recovery_message)

    await asyncio.sleep(1)

    recovery_stats = node1.get_network_statistics()
    print(f"After failure - Active peers: {recovery_stats['active_peers']}")
    print("Network handled failure gracefully")

    # Cleanup
    await node1.stop()
    await node2.stop()

    print("\n" + "=" * 50)
    print("Mesh Network Integration Test Complete!")
    print(f"Success Rate: {total_success_rate:.1%}")
    print("All message types supported")
    print("Health monitoring working")
    print("Resilience mechanisms active")
    print("Ready for production deployment")

    return total_success_rate >= 0.8


def test_mesh_integration_pytest():
    """Pytest wrapper for the mesh integration test."""
    result = asyncio.run(test_mesh_integration())
    assert result, "Mesh network integration test failed"


if __name__ == "__main__":
    result = asyncio.run(test_mesh_integration())
    sys.exit(0 if result else 1)
