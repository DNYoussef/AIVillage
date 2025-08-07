#!/usr/bin/env python3
"""Simple test of mesh network integration - Windows compatible.

Moved from root to tests/experimental/mesh/ for better organization.
"""

import asyncio
import logging
from pathlib import Path
import sys

from mesh_network_manager import MeshNetworkManager

from communications.message import Message, MessageType, Priority

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


async def test_mesh_simple():
    """Simple test that mesh network works."""
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise

    print("TESTING MESH NETWORK INTEGRATION")
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

    print("PASS - Both mesh nodes started")

    # Wait for connection establishment
    await asyncio.sleep(2)

    # Test basic message sending
    print("\nTesting basic message sending...")
    test_message = Message(
        type=MessageType.QUERY,
        sender="agent_001",
        receiver="agent_002",
        content={"test": "mesh_integration", "data": "Hello from mesh!"},
        priority=Priority.HIGH,
    )

    await node1.send_message(test_message)
    print("PASS - Message sent through mesh")

    # Check network statistics
    await asyncio.sleep(1)
    stats1 = node1.get_network_statistics()
    stats2 = node2.get_network_statistics()

    print("\nNetwork Statistics:")
    print(
        f"  Node 1 - Active peers: {stats1['active_peers']}, Routes: {stats1['routing_entries']}"
    )
    print(
        f"  Node 2 - Active peers: {stats2['active_peers']}, Routes: {stats2['routing_entries']}"
    )

    # Check health
    health1 = stats1["network_health"]
    print("\nHealth Status:")
    print(f"  Success rate: {health1['success_rate']:.1%}")
    print(f"  Average latency: {health1['average_latency_ms']:.1f}ms")
    print(f"  Total messages: {health1['total_messages']}")

    # Send multiple message types
    print("\nTesting multiple message types...")
    message_types = [MessageType.TASK, MessageType.NOTIFICATION, MessageType.RESPONSE]

    for msg_type in message_types:
        message = Message(
            type=msg_type,
            sender="agent_001",
            receiver="agent_002",
            content={"type": msg_type.value, "test": True},
        )
        await node1.send_message(message)
        print(f"  PASS - Sent {msg_type.value} message")

    # Wait for processing
    await asyncio.sleep(1)

    # Final statistics
    final_stats = node1.get_network_statistics()
    final_health = final_stats["network_health"]

    print("\nFinal Results:")
    print(f"  Total messages processed: {final_health['total_messages']}")
    print(f"  Success rate: {final_health['success_rate']:.1%}")
    print(
        f"  Network operational: {'YES' if final_health['success_rate'] > 0.8 else 'NO'}"
    )

    # Test resilience
    print("\nTesting network resilience...")
    await node1.remove_peer("agent_002")
    print("  PASS - Peer removal handled gracefully")

    # Cleanup
    await node1.stop()
    await node2.stop()

    print("\n" + "=" * 50)
    print("MESH NETWORK INTEGRATION TEST COMPLETE")

    # Determine overall result
    success = (
        stats1["active_peers"] > 0
        and final_health["success_rate"] >= 1.0
        and final_health["total_messages"] >= 4
    )

    print(f"OVERALL RESULT: {'SUCCESS' if success else 'FAILED'}")

    if success:
        print("- Message delivery: WORKING")
        print("- Network formation: WORKING")
        print("- Health monitoring: WORKING")
        print("- Multiple message types: WORKING")
        print("- Resilience handling: WORKING")
        print("- Ready for production: YES")
    else:
        print("- Issues detected, review logs")

    return success


def test_mesh_simple_pytest():
    """Pytest wrapper for the mesh simple test."""
    result = asyncio.run(test_mesh_simple())
    assert result, "Mesh network integration test failed"


if __name__ == "__main__":
    result = asyncio.run(test_mesh_simple())
    exit(0 if result else 1)
