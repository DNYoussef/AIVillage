#!/usr/bin/env python3
"""Validation script for Resilient Chat Engine implementation.

This script validates that the ChatEngine properly handles:
1. Service unavailability (the original twin:8001 connection error)
2. Graceful degradation with meaningful responses
3. Circuit breaker behavior
4. Different operation modes
"""

import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_offline_functionality():
    """Test that system works when twin:8001 is unavailable."""
    print("TEST 1: Offline Functionality")
    print("-" * 40)

    from infrastructure.twin.chat_engine import ChatEngine

    # Force local mode to simulate twin:8001 being unavailable
    os.environ["CHAT_MODE"] = "local"
    os.environ["OFFLINE_RESPONSES_ENABLED"] = "1"

    engine = ChatEngine()

    # Test messages that would previously fail with ConnectionError
    test_cases = [
        ("Hello", "Basic greeting"),
        ("What is the weather?", "Question"),
        ("/status", "System status command"),
        ("/help", "Help command"),
        ("How are you doing?", "Follow-up message"),
    ]

    success_count = 0
    for message, description in test_cases:
        try:
            response = engine.process_chat(message, "test_offline")

            # Validate response structure
            required_fields = ["response", "conversation_id", "mode", "service_status"]
            if all(field in response for field in required_fields):
                print(f"  PASS: {description} -> {response['mode']} mode")
                success_count += 1
            else:
                print(f"  FAIL: {description} -> Missing required fields")

        except Exception as e:
            print(f"  FAIL: {description} -> Exception: {e}")

    print(f"Result: {success_count}/{len(test_cases)} tests passed")
    return success_count == len(test_cases)


def test_circuit_breaker_prevents_cascade_failures():
    """Test that circuit breaker prevents cascade failures."""
    print("\nTEST 2: Circuit Breaker Protection")
    print("-" * 40)

    from infrastructure.twin.chat_engine import ChatEngine

    # Use hybrid mode with aggressive circuit breaker settings
    os.environ["CHAT_MODE"] = "hybrid"
    os.environ["CIRCUIT_BREAKER_FAILURE_THRESHOLD"] = "2"
    os.environ["CIRCUIT_BREAKER_TIMEOUT_MS"] = "10000"

    engine = ChatEngine()

    # Multiple requests should trigger circuit breaker (twin:8001 will fail)
    failure_count = 0
    fallback_count = 0

    for i in range(5):
        try:
            response = engine.process_chat(f"Test message {i}", "cb_test")

            if response["mode"] == "fallback":
                fallback_count += 1
                print(f"  Request {i+1}: Using fallback (circuit breaker working)")
            else:
                print(f"  Request {i+1}: Mode {response['mode']}")

        except Exception as e:
            failure_count += 1
            print(f"  Request {i+1}: Exception {e}")

    # Circuit breaker should have activated fallback mode
    cb_stats = engine.get_system_status()["circuit_breaker_stats"]
    print(f"Circuit breaker stats: {cb_stats['total_calls']} calls, {cb_stats['total_failures']} failures")

    # Success if we got fallback responses instead of hard failures
    success = fallback_count > 0 and failure_count == 0
    print(f"Result: {'PASS' if success else 'FAIL'} - Circuit breaker {'activated' if success else 'failed'}")
    return success


def test_configuration_modes():
    """Test different configuration modes work correctly."""
    print("\nTEST 3: Configuration Modes")
    print("-" * 40)

    from infrastructure.twin.chat_engine import ChatEngine

    modes_to_test = [
        ("local", "offline"),
        ("hybrid", "degraded"),  # Will be degraded since twin:8001 unavailable
    ]

    success_count = 0
    for mode, expected_status in modes_to_test:
        try:
            os.environ["CHAT_MODE"] = mode
            engine = ChatEngine()

            response = engine.process_chat("Test configuration", "config_test")
            response["service_status"]

            if mode == "local" and response["mode"] == "local":
                print(f"  PASS: {mode} mode working correctly")
                success_count += 1
            elif mode == "hybrid" and response["mode"] in ["fallback", "local"]:
                print(f"  PASS: {mode} mode falling back correctly")
                success_count += 1
            else:
                print(f"  FAIL: {mode} mode - unexpected response mode {response['mode']}")

        except Exception as e:
            print(f"  FAIL: {mode} mode - Exception: {e}")

    print(f"Result: {success_count}/{len(modes_to_test)} modes working")
    return success_count == len(modes_to_test)


def test_meaningful_offline_responses():
    """Test that offline responses are meaningful and helpful."""
    print("\nTEST 4: Meaningful Offline Responses")
    print("-" * 40)

    from infrastructure.twin.chat_engine import ChatEngine

    os.environ["CHAT_MODE"] = "local"
    engine = ChatEngine()

    # Test various response types
    test_messages = [
        ("Hello", "greeting"),
        ("What is AI?", "question"),
        ("I need help", "help"),
        ("/status", "status"),
        ("Random message", "default"),
    ]

    success_count = 0
    for message, expected_type in test_messages:
        try:
            response = engine.process_chat(message, "meaningful_test")

            # Check response quality indicators
            response_text = response["response"]
            quality_checks = [
                len(response_text) > 20,  # Not too short
                not response_text.startswith("Error"),  # Not an error message
                "local mode" in response_text.lower()
                or "offline" in response_text.lower()
                or message in response_text,  # Contextual
            ]

            if all(quality_checks):
                print(f"  PASS: '{message}' -> Meaningful response ({len(response_text)} chars)")
                success_count += 1
            else:
                print(f"  FAIL: '{message}' -> Poor response: {response_text[:50]}...")

        except Exception as e:
            print(f"  FAIL: '{message}' -> Exception: {e}")

    print(f"Result: {success_count}/{len(test_messages)} responses are meaningful")
    return success_count == len(test_messages)


def test_system_status_monitoring():
    """Test system status and monitoring capabilities."""
    print("\nTEST 5: System Status Monitoring")
    print("-" * 40)

    from infrastructure.twin.chat_engine import ChatEngine

    os.environ["CHAT_MODE"] = "hybrid"
    engine = ChatEngine()

    try:
        # Test health check
        health_result = engine.health_check_twin_service()
        print(f"  Health check result: {'Healthy' if health_result else 'Unavailable'} (expected for twin:8001)")

        # Test system status
        status = engine.get_system_status()
        required_status_fields = [
            "mode",
            "service_status",
            "circuit_breaker_state",
            "twin_url",
            "offline_responses_enabled",
        ]

        missing_fields = [field for field in required_status_fields if field not in status]
        if not missing_fields:
            print("  PASS: All required status fields present")
            print(f"  Mode: {status['mode']}, Status: {status['service_status']}")
            print(f"  Circuit Breaker: {status['circuit_breaker_state']}")
            return True
        else:
            print(f"  FAIL: Missing status fields: {missing_fields}")
            return False

    except Exception as e:
        print(f"  FAIL: Status monitoring exception: {e}")
        return False


def main():
    """Run all validation tests."""
    print("RESILIENT CHAT ENGINE VALIDATION")
    print("=" * 50)
    print("Testing fixes for: ConnectionError to twin:8001")
    print("=" * 50)

    # Run all tests
    tests = [
        ("Offline Functionality", test_offline_functionality),
        ("Circuit Breaker Protection", test_circuit_breaker_prevents_cascade_failures),
        ("Configuration Modes", test_configuration_modes),
        ("Meaningful Responses", test_meaningful_offline_responses),
        ("System Monitoring", test_system_status_monitoring),
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed_tests += 1
                print(f"✓ {test_name}: PASSED")
            else:
                print(f"✗ {test_name}: FAILED")
        except Exception as e:
            print(f"✗ {test_name}: EXCEPTION - {e}")

    print("\n" + "=" * 50)
    print(f"VALIDATION RESULTS: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("SUCCESS: All resilience features working correctly!")
        print("✓ System no longer fails with ConnectionError to twin:8001")
        print("✓ Users get meaningful responses when service is offline")
        print("✓ Circuit breaker prevents cascade failures")
        print("✓ Multiple operation modes supported")
        print("✓ Comprehensive system monitoring available")
        return 0
    else:
        print(f"FAILURE: {total_tests - passed_tests} tests failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
