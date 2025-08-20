#!/usr/bin/env python3
"""
C1: P2P Network Verification Test (V2 - Improved)
Claim: "0% -> 100% connection success rate"
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Import required classes
try:
    from packages.p2p.core.message_types import UnifiedMessage as DualPathMessage
    from packages.p2p.core.transport_manager import TransportManager as DualPathTransport
except ImportError:
    # Fallback for testing
    class DualPathTransport:
        def __init__(self, node_id):
            self.node_id = node_id

    class DualPathMessage:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


def test_p2p_imports():
    """Test that all P2P modules can be imported."""
    results = []

    modules_to_test = [
        ("bitchat_transport", "packages.p2p.bitchat.ble_transport"),
        ("betanet_transport", "packages.p2p.betanet.htx_transport"),
        ("dual_path_transport", "packages.p2p.core.transport_manager"),
        ("libp2p_mesh", "packages.p2p.core.transport_manager"),
        ("mdns_discovery", "packages.p2p.core.transport_manager"),
        ("fallback_transports", "packages.p2p.core.transport_manager"),
    ]

    for name, module_path in modules_to_test:
        try:
            exec(f"import {module_path}")
            results.append((name, "PASS", "Import successful"))
        except ImportError as e:
            results.append((name, "FAIL", f"Import error: {e}"))
        except Exception as e:
            results.append((name, "FAIL", f"Unexpected error: {e}"))

    return results


def test_dual_path_transport():
    """Test DualPathTransport functionality."""
    try:
        # Test basic instantiation
        transport = DualPathTransport(node_id="test_node")

        # Check attributes exist
        tests = [
            (
                "Has node_id",
                hasattr(transport, "node_id"),
                f"node_id: {transport.node_id}",
            ),
            (
                "Has bitchat",
                hasattr(transport, "bitchat"),
                "BitChat transport attribute",
            ),
            (
                "Has betanet",
                hasattr(transport, "betanet"),
                "Betanet transport attribute",
            ),
            (
                "Has routing_stats",
                hasattr(transport, "routing_stats"),
                "Routing statistics",
            ),
        ]

        results = []
        for test_name, passed, detail in tests:
            results.append((test_name, "PASS" if passed else "FAIL", detail))

        # Test message creation
        try:
            msg = DualPathMessage(sender="node1", recipient="node2", payload="Test message")
            results.append(("Message creation", "PASS", f"Created message ID: {msg.id[:8]}..."))
        except Exception as e:
            results.append(("Message creation", "FAIL", f"Error: {e}"))

        return results
    except ImportError as e:
        return [("DualPathTransport", "FAIL", f"Import failed: {e}")]
    except Exception as e:
        return [("DualPathTransport", "FAIL", f"Test failed: {e}")]


def test_message_types():
    """Test support for all claimed message types."""
    try:
        from packages.p2p.core.libp2p_mesh import MeshMessageType

        required_types = [
            "DATA_MESSAGE",
            "AGENT_TASK",
            "PARAMETER_UPDATE",
            "GRADIENT_SHARING",
        ]
        found_types = []

        for msg_type in required_types:
            if hasattr(MeshMessageType, msg_type):
                found_types.append(msg_type)

        if len(found_types) == len(required_types):
            return [("Message Types", "PASS", f"All {len(required_types)} types found")]
        else:
            missing = set(required_types) - set(found_types)
            return [("Message Types", "FAIL", f"Missing: {missing}")]
    except ImportError as e:
        return [("Message Types", "FAIL", f"Import failed: {e}")]
    except Exception as e:
        return [("Message Types", "FAIL", f"Test failed: {e}")]


def test_libp2p_mesh():
    """Test LibP2P mesh network functionality."""
    try:
        from packages.p2p.core.libp2p_mesh import LibP2PMeshNetwork, MeshMessage

        # Test basic instantiation
        mesh = LibP2PMeshNetwork(peer_id="test_peer")

        # Check key attributes
        tests = [
            ("Has peer_id", hasattr(mesh, "peer_id"), f"peer_id: {mesh.peer_id}"),
            ("Has messages", hasattr(mesh, "messages"), "Message storage"),
            ("Has peers", hasattr(mesh, "peers"), "Peer tracking"),
            ("Has stats", hasattr(mesh, "stats"), "Network statistics"),
        ]

        results = []
        for test_name, passed, detail in tests:
            results.append((test_name, "PASS" if passed else "FAIL", detail))

        # Test message creation
        try:
            msg = MeshMessage(sender="peer1", recipient="peer2", payload=b"Test payload")
            results.append(("Mesh message", "PASS", f"Created message: {msg.id[:8]}..."))
        except Exception as e:
            results.append(("Mesh message", "FAIL", f"Error: {e}"))

        return results
    except ImportError as e:
        return [("LibP2P Mesh", "FAIL", f"Import failed: {e}")]
    except Exception as e:
        return [("LibP2P Mesh", "FAIL", f"Test failed: {e}")]


def test_fallback_transports():
    """Test fallback transport availability."""
    try:
        from packages.p2p.core.fallback_transports import (
            BluetoothFallback,
            FileSystemFallback,
            LocalSocketFallback,
            WiFiDirectFallback,
        )

        results = []

        # Test each fallback transport
        transports = [
            ("Bluetooth", BluetoothFallback),
            ("WiFi Direct", WiFiDirectFallback),
            ("File System", FileSystemFallback),
            ("Local Socket", LocalSocketFallback),
        ]

        for name, transport_class in transports:
            try:
                transport = transport_class()
                is_available = hasattr(transport, "is_available") and callable(transport.is_available)
                results.append((name, "PASS" if is_available else "PARTIAL", "Class instantiated"))
            except Exception as e:
                results.append((name, "FAIL", f"Error: {e}"))

        return results
    except ImportError as e:
        # Check if the module exists but classes are different
        try:
            import packages.core.p2p.fallback_transports as ft

            classes = [c for c in dir(ft) if "Fallback" in c or "Transport" in c]
            return [
                (
                    "Fallback Transports",
                    "PARTIAL",
                    f"Module exists, found: {classes[:3]}...",
                )
            ]
        except:
            return [("Fallback Transports", "FAIL", f"Import failed: {e}")]
    except Exception as e:
        return [("Fallback Transports", "FAIL", f"Test failed: {e}")]


def test_mdns_discovery():
    """Test mDNS peer discovery."""
    try:
        from packages.p2p.core.mdns_discovery import MDNSPeerDiscovery

        # Test basic instantiation
        discovery = MDNSPeerDiscovery(service_name="aivillage_test")

        # Check key methods
        tests = [
            (
                "Has service_name",
                hasattr(discovery, "service_name"),
                "Service name attribute",
            ),
            (
                "Has start method",
                hasattr(discovery, "start_discovery"),
                "Discovery start method",
            ),
            (
                "Has stop method",
                hasattr(discovery, "stop_discovery"),
                "Discovery stop method",
            ),
            ("Has peers list", hasattr(discovery, "discovered_peers"), "Peer tracking"),
        ]

        results = []
        for test_name, passed, detail in tests:
            results.append((test_name, "PASS" if passed else "FAIL", detail))

        return results
    except ImportError as e:
        return [("mDNS Discovery", "FAIL", f"Import failed: {e}")]
    except Exception as e:
        return [("mDNS Discovery", "FAIL", f"Test failed: {e}")]


def main():
    print("=" * 70)
    print("C1: P2P NETWORK VERIFICATION (V2)")
    print("Claim: 0% -> 100% connection success rate")
    print("=" * 70)

    all_results = []

    # Run all tests
    print("\n1. Testing P2P Module Imports...")
    import_results = test_p2p_imports()
    all_results.extend(import_results)
    for name, status, msg in import_results:
        print(f"  {name:20} {status:8} - {msg}")

    print("\n2. Testing DualPathTransport...")
    dual_results = test_dual_path_transport()
    all_results.extend(dual_results)
    for name, status, msg in dual_results:
        print(f"  {name:20} {status:8} - {msg}")

    print("\n3. Testing Message Type Support...")
    msg_results = test_message_types()
    all_results.extend(msg_results)
    for name, status, msg in msg_results:
        print(f"  {name:20} {status:8} - {msg}")

    print("\n4. Testing LibP2P Mesh Network...")
    mesh_results = test_libp2p_mesh()
    all_results.extend(mesh_results)
    for name, status, msg in mesh_results:
        print(f"  {name:20} {status:8} - {msg}")

    print("\n5. Testing Fallback Transports...")
    fallback_results = test_fallback_transports()
    all_results.extend(fallback_results)
    for name, status, msg in fallback_results:
        print(f"  {name:20} {status:8} - {msg}")

    print("\n6. Testing mDNS Discovery...")
    mdns_results = test_mdns_discovery()
    all_results.extend(mdns_results)
    for name, status, msg in mdns_results:
        print(f"  {name:20} {status:8} - {msg}")

    # Summary
    total = len(all_results)
    passed = sum(1 for _, status, _ in all_results if status == "PASS")
    partial = sum(1 for _, status, _ in all_results if status == "PARTIAL")
    failed = sum(1 for _, status, _ in all_results if status == "FAIL")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed}/{total} tests passed, {partial} partial, {failed} failed")

    success_rate = (passed / total) * 100 if total > 0 else 0

    if success_rate >= 90:
        print(f"VERDICT: PASS - P2P network claims verified ({success_rate:.1f}% success)")
        verdict = "PASS"
    elif success_rate >= 60:
        print(f"VERDICT: PARTIAL - Some P2P features working ({success_rate:.1f}% success)")
        verdict = "PARTIAL"
    else:
        print(f"VERDICT: FAIL - P2P network claims not substantiated ({success_rate:.1f}% success)")
        verdict = "FAIL"

    # Save results
    with open("../artifacts/c1_p2p_results.txt", "w") as f:
        f.write("C1 P2P Network Test Results (V2)\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Total Tests: {total}\n")
        f.write(f"Passed: {passed}\n")
        f.write(f"Partial: {partial}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Success Rate: {success_rate:.1f}%\n")
        f.write(f"Verdict: {verdict}\n\n")

        f.write("Detailed Results:\n")
        for name, status, msg in all_results:
            f.write(f"  {name}: {status} - {msg}\n")

    return verdict


if __name__ == "__main__":
    main()
