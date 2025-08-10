"""Validate P2P Protocol Compatibility Fix.

This script validates that the P2P integration bug has been fixed by testing:
1. Protocol mismatch is resolved (discovery + encrypted P2P compatibility)
2. 5-peer limit is removed from evolution system
3. Multi-node network can form successfully
"""

import asyncio
import json
import logging
import socket
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from src.core.p2p.p2p_node import NodeStatus, P2PNode, PeerCapabilities
    from src.core.p2p.peer_discovery import PeerDiscovery
except ImportError as e:
    print(f"X Import error: {e}")
    print("Make sure you're running from the AIVillage root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_discovery_protocol_compatibility() -> bool | None:
    """Test 1: Discovery protocol can communicate with P2P server."""
    print("[TEST 1] Discovery Protocol Compatibility")

    # Create resource monitor mock
    async def mock_resource_monitor():
        return {
            "cpu_cores": 4,
            "ram_total_mb": 8192,
            "memory_percent": 30,
            "cpu_percent": 20,
            "battery_percent": 80,
            "network_type": "wifi",
            "device_type": "laptop",
            "thermal_state": "normal",
        }

    # Create and start P2P node
    node = P2PNode(
        node_id="test_node", listen_port=9300, resource_monitor=mock_resource_monitor
    )

    try:
        await node.start()
        await asyncio.sleep(0.5)  # Give server time to start

        # Test discovery message
        discovery_msg = {
            "type": "PEER_DISCOVERY",
            "sender_id": "discovery_test_client",
            "sender_port": 9999,
            "timestamp": time.time(),
            "capabilities": {
                "cpu_cores": 2,
                "ram_mb": 4096,
                "device_type": "phone",
                "can_evolve": True,
                "evolution_capacity": 0.8,
            },
        }

        # Connect to P2P node
        reader, writer = await asyncio.open_connection("localhost", 9300)

        try:
            # Send discovery message using discovery protocol format
            msg_data = json.dumps(discovery_msg).encode("utf-8")
            length_data = len(msg_data).to_bytes(4, "big")
            writer.write(length_data + msg_data)
            await writer.drain()

            # Read response with timeout
            try:
                resp_length_data = await asyncio.wait_for(
                    reader.readexactly(4), timeout=5.0
                )
                resp_length = int.from_bytes(resp_length_data, "big")
                resp_data = await asyncio.wait_for(
                    reader.readexactly(resp_length), timeout=5.0
                )

                response = json.loads(resp_data.decode("utf-8"))

                # Validate response
                if response.get("type") == "PEER_DISCOVERY_RESPONSE":
                    print("   ✅ Discovery protocol communication successful")
                    print(f"   📡 Response from node: {response.get('node_id')}")
                    return True
                print(f"   ❌ Unexpected response: {response}")
                return False

            except asyncio.TimeoutError:
                print("   ❌ Timeout waiting for response")
                return False

        finally:
            writer.close()
            await writer.wait_closed()

    except Exception as e:
        print(f"   ❌ Test failed with error: {e}")
        return False
    finally:
        await node.stop()


async def test_evolution_peer_limit_removed() -> bool | None:
    """Test 2: Evolution peer selection no longer limited to 5."""
    print("🔍 Test 2: Evolution Peer Limit Removed")

    async def mock_resource_monitor():
        return {
            "cpu_cores": 4,
            "ram_total_mb": 8192,
            "memory_percent": 30,
            "cpu_percent": 20,
            "battery_percent": 80,
            "network_type": "wifi",
            "device_type": "laptop",
            "thermal_state": "normal",
        }

    node = P2PNode(
        node_id="evolution_test_node",
        listen_port=9301,
        resource_monitor=mock_resource_monitor,
    )

    try:
        await node.start()

        # Manually add 10 suitable evolution peers
        for i in range(10):
            peer_id = f"evolution_peer_{i}"
            capabilities = PeerCapabilities(
                device_id=peer_id,
                cpu_cores=4,
                ram_mb=8192,
                can_evolve=True,
                evolution_capacity=0.9,
                available_for_evolution=True,
                current_evolution_load=0.1,
                thermal_state="normal",
                battery_percent=90,
            )
            node.peer_registry[peer_id] = capabilities

        # Test evolution peer selection
        suitable_peers = node.get_suitable_evolution_peers(min_count=1)

        if len(suitable_peers) == 10:
            print("   ✅ Evolution peer limit removed successfully")
            print(f"   📊 Can select all {len(suitable_peers)} suitable peers")
            return True
        print(f"   ❌ Expected 10 peers, got {len(suitable_peers)}")
        return False

    except Exception as e:
        print(f"   ❌ Test failed with error: {e}")
        return False
    finally:
        await node.stop()


async def test_multi_node_network() -> bool | None:
    """Test 3: Multi-node network can form successfully."""
    print("🔍 Test 3: Multi-Node Network Formation")

    async def mock_resource_monitor():
        return {
            "cpu_cores": 4,
            "ram_total_mb": 8192,
            "memory_percent": 30,
            "cpu_percent": 20,
            "battery_percent": 80,
            "network_type": "wifi",
            "device_type": "laptop",
            "thermal_state": "normal",
        }

    # Create 5 nodes
    nodes = []
    base_port = 9310

    try:
        for i in range(5):
            node = P2PNode(
                node_id=f"network_node_{i}",
                listen_port=base_port + i,
                resource_monitor=mock_resource_monitor,
            )
            nodes.append(node)

        # Start all nodes
        for node in nodes:
            await node.start()
            await asyncio.sleep(0.1)

        # Start discovery on all nodes
        for node in nodes:
            await node.peer_discovery.start_discovery()

        # Wait for discovery
        print("   🔄 Running peer discovery for 3 seconds...")
        await asyncio.sleep(3.0)

        # Check results
        total_discoveries = 0
        successful_nodes = 0

        for i, node in enumerate(nodes):
            discovered = len(node.peer_registry)
            total_discoveries += discovered
            if discovered > 0:
                successful_nodes += 1
            print(f"   📡 Node {i}: {discovered} peers discovered")

        # Stop discovery
        for node in nodes:
            await node.peer_discovery.stop_discovery()

        if successful_nodes >= 2 and total_discoveries >= 2:
            print("   ✅ Network formation successful")
            print(f"   📊 {successful_nodes}/5 nodes discovered peers")
            print(f"   🌐 {total_discoveries} total peer connections")
            return True
        print("   ❌ Network formation failed")
        print(f"   📊 Only {successful_nodes} nodes discovered peers")
        return False

    except Exception as e:
        print(f"   ❌ Test failed with error: {e}")
        return False
    finally:
        # Cleanup
        for node in nodes:
            try:
                await node.stop()
            except Exception as e:
                logger.warning(f"Error stopping node: {e}")


async def validate_code_changes() -> bool | None:
    """Test 4: Validate that our code changes are present."""
    print("🔍 Test 4: Code Changes Validation")

    try:
        # Check if the protocol adapter methods exist
        from core.p2p.p2p_node import P2PNode

        # Create a dummy node to check methods
        node = P2PNode(node_id="validation_node")

        # Check if new methods exist
        has_read_discovery = hasattr(node, "_read_discovery_message")
        has_handle_discovery = hasattr(node, "_handle_discovery_message")

        if has_read_discovery and has_handle_discovery:
            print("   ✅ Protocol adapter methods present")
        else:
            print(
                f"   ❌ Missing methods: read_discovery={has_read_discovery}, handle_discovery={has_handle_discovery}"
            )
            return False

        # Check the evolution peer selection method
        import inspect

        source = inspect.getsource(node.get_suitable_evolution_peers)

        if "max(min_count, network_size)" in source:
            print("   ✅ Evolution peer limit fix present")
        else:
            print("   ❌ Evolution peer limit fix not found")
            return False

        print("   ✅ All code changes validated")
        return True

    except Exception as e:
        print(f"   ❌ Code validation failed: {e}")
        return False


def find_free_port(start_port=9400, max_tries=100):
    """Find a free port for testing."""
    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue
    msg = f"Could not find free port in range {start_port}-{start_port + max_tries}"
    raise RuntimeError(msg)


async def main() -> bool:
    """Run all validation tests."""
    print("🚀 P2P Protocol Fix Validation")
    print("=" * 50)

    tests = [
        ("Discovery Protocol Compatibility", test_discovery_protocol_compatibility),
        ("Evolution Peer Limit Removed", test_evolution_peer_limit_removed),
        ("Multi-Node Network Formation", test_multi_node_network),
        ("Code Changes Validation", validate_code_changes),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}:")
            result = await test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ❌ {test_name} crashed: {e}")
            failed += 1

        # Small delay between tests
        await asyncio.sleep(0.5)

    print("\n" + "=" * 50)
    print("🏁 VALIDATION SUMMARY")
    print(f"✅ Tests Passed: {passed}")
    print(f"❌ Tests Failed: {failed}")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ P2P protocol mismatch FIXED")
        print("✅ 5-peer evolution limit REMOVED")
        print("✅ Multi-node networks FUNCTIONAL")
        print("✅ Discovery and P2P protocols COMPATIBLE")
        print("\n💡 The critical P2P integration bug has been successfully resolved!")
        return True
    print(f"\n❌ {failed} test(s) failed. Please check the implementation.")
    return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
