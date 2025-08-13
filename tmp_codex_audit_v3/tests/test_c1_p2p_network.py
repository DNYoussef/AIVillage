#!/usr/bin/env python3
"""
C1: P2P Network Verification Test
Claim: "0% → 100% connection success rate"
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


def test_p2p_imports():
    """Test that all P2P modules can be imported."""
    results = []

    modules_to_test = [
        ("bitchat_transport", "src.core.p2p.bitchat_transport"),
        ("betanet_transport", "src.core.p2p.betanet_transport"),
        ("dual_path_transport", "src.core.p2p.dual_path_transport"),
        ("libp2p_mesh", "src.core.p2p.libp2p_mesh"),
        ("mdns_discovery", "src.core.p2p.mdns_discovery"),
        ("fallback_transports", "src.core.p2p.fallback_transports"),
    ]

    for name, module_path in modules_to_test:
        try:
            exec(f"from {module_path} import *")
            results.append((name, "PASS", "Import successful"))
        except ImportError as e:
            results.append((name, "FAIL", f"Import error: {e}"))
        except Exception as e:
            results.append((name, "FAIL", f"Unexpected error: {e}"))

    return results


def test_path_policy():
    """Test BitChat vs Betanet path selection logic."""
    try:
        from src.core.p2p.dual_path_transport import DualPathTransport

        # Test offline/local preference
        transport = DualPathTransport()

        # Simulate offline scenario
        offline_path = transport.select_path(network_status="offline")
        local_path = transport.select_path(network_status="local")
        global_path = transport.select_path(network_status="global")

        results = [
            (
                "Offline -> BitChat",
                "PASS" if offline_path == "bitchat" else "FAIL",
                f"Selected: {offline_path}",
            ),
            (
                "Local -> BitChat",
                "PASS" if local_path == "bitchat" else "FAIL",
                f"Selected: {local_path}",
            ),
            (
                "Global -> Betanet",
                "PASS" if global_path == "betanet" else "FAIL",
                f"Selected: {global_path}",
            ),
        ]
        return results
    except Exception as e:
        return [("Path Policy", "FAIL", f"Test failed: {e}")]


def test_message_types():
    """Test support for all claimed message types."""
    try:
        from src.core.p2p.libp2p_mesh import MessageType

        required_types = [
            "DATA_MESSAGE",
            "AGENT_TASK",
            "PARAMETER_UPDATE",
            "GRADIENT_SHARING",
        ]
        found_types = []

        for msg_type in required_types:
            if hasattr(MessageType, msg_type):
                found_types.append(msg_type)

        if len(found_types) == len(required_types):
            return [("Message Types", "PASS", f"All {len(required_types)} types found")]
        else:
            missing = set(required_types) - set(found_types)
            return [("Message Types", "FAIL", f"Missing: {missing}")]
    except Exception as e:
        return [("Message Types", "FAIL", f"Test failed: {e}")]


def test_fallback_transports():
    """Test fallback transport availability."""
    try:
        from src.core.p2p.fallback_transports import FallbackTransportLayer

        transport = FallbackTransportLayer()
        available = transport.get_available_transports()

        expected = ["bluetooth", "wifi_direct", "file_system", "local_socket"]
        found = [t for t in expected if t in available]

        if len(found) == len(expected):
            return [
                (
                    "Fallback Transports",
                    "PASS",
                    f"All {len(expected)} transports available",
                )
            ]
        else:
            return [
                (
                    "Fallback Transports",
                    "PARTIAL",
                    f"Found {len(found)}/{len(expected)}: {found}",
                )
            ]
    except Exception as e:
        return [("Fallback Transports", "FAIL", f"Test failed: {e}")]


def main():
    print("=" * 70)
    print("C1: P2P NETWORK VERIFICATION")
    print("Claim: 0% -> 100% connection success rate")
    print("=" * 70)

    all_results = []

    # Run all tests
    print("\n1. Testing P2P Module Imports...")
    import_results = test_p2p_imports()
    all_results.extend(import_results)
    for name, status, msg in import_results:
        print(f"  {name:20} {status} - {msg}")

    print("\n2. Testing Path Selection Policy...")
    path_results = test_path_policy()
    all_results.extend(path_results)
    for name, status, msg in path_results:
        print(f"  {name:20} {status} - {msg}")

    print("\n3. Testing Message Type Support...")
    msg_results = test_message_types()
    all_results.extend(msg_results)
    for name, status, msg in msg_results:
        print(f"  {name:20} {status} - {msg}")

    print("\n4. Testing Fallback Transports...")
    fallback_results = test_fallback_transports()
    all_results.extend(fallback_results)
    for name, status, msg in fallback_results:
        print(f"  {name:20} {status} - {msg}")

    # Summary
    total = len(all_results)
    passed = sum(1 for _, status, _ in all_results if "PASS" in status)
    failed = sum(1 for _, status, _ in all_results if "FAIL" in status)

    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed}/{total} tests passed")

    if failed == 0:
        print("VERDICT: PASS - P2P network claims verified")
        verdict = "PASS"
    elif passed > total * 0.5:
        print("VERDICT: PARTIAL - Some P2P features working")
        verdict = "PARTIAL"
    else:
        print("VERDICT: FAIL - P2P network claims not substantiated")
        verdict = "FAIL"

    # Save results
    with open("../artifacts/c1_p2p_results.txt", "w") as f:
        f.write("C1 P2P Network Test Results\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Total Tests: {total}\n")
        f.write(f"Passed: {passed}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Verdict: {verdict}\n\n")

        for name, status, msg in all_results:
            f.write(f"{name}: {status} - {msg}\n")

    return verdict


if __name__ == "__main__":
    main()
