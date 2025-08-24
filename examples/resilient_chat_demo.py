#!/usr/bin/env python3
"""Demonstration of the Resilient Chat Engine with offline fallback capability.

This script shows how the ChatEngine works in different scenarios:
1. When Twin service (twin:8001) is available
2. When Twin service is unavailable (offline fallback)
3. Circuit breaker behavior during failures
4. Different configuration modes (remote, local, hybrid)

Usage:
    python examples/resilient_chat_demo.py

Environment variables:
    CHAT_MODE=hybrid|remote|local
    TWIN_URL=https://twin:8001/v1/chat
    OFFLINE_RESPONSES_ENABLED=1|0
    CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
"""

import os
from pathlib import Path
import sys
import time

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from infrastructure.twin.chat_engine import ChatEngine


def demo_chat_modes():
    """Demonstrate different chat engine modes and their behavior."""
    print("=" * 70)
    print("RESILIENT CHAT ENGINE DEMONSTRATION")
    print("=" * 70)

    # Test different modes
    modes_to_test = ["local", "remote", "hybrid"]

    for mode in modes_to_test:
        print(f"\n🔧 Testing Chat Engine in {mode.upper()} mode")
        print("-" * 50)

        # Set environment for this test
        os.environ["CHAT_MODE"] = mode

        try:
            # Initialize chat engine
            engine = ChatEngine()

            # Show system status
            status = engine.get_system_status()
            print(f"Mode: {status['mode']}")
            print(f"Service Status: {status['service_status']}")
            print(f"Circuit Breaker: {status['circuit_breaker_state']}")
            print(f"Twin URL: {status['twin_url']}")

            # Test various message types
            test_messages = [
                ("Hello!", "test_conversation_1"),
                ("What is the weather like?", "test_conversation_1"),
                ("/status", "test_conversation_1"),
                ("/help", "test_conversation_2"),
                ("/echo This is a test message", "test_conversation_2"),
                ("How are you doing?", "test_conversation_1"),
            ]

            print(f"\n📝 Testing {len(test_messages)} messages...")

            for i, (message, conv_id) in enumerate(test_messages, 1):
                print(f"\n{i}. User: {message}")
                try:
                    response = engine.process_chat(message, conv_id)
                    print(f"   Assistant ({response['mode']}): {response['response']}")
                    print(
                        f"   Status: {response['service_status']} | "
                        f"Time: {response['processing_time_ms']}ms | "
                        f"Confidence: {response['raw_prob']:.3f}"
                    )

                    if "notice" in response:
                        print(f"   ⚠️  {response['notice']}")

                except Exception as e:
                    print(f"   ❌ Error: {e}")

                # Small delay between messages
                time.sleep(0.5)

        except Exception as e:
            print(f"❌ Failed to initialize engine in {mode} mode: {e}")

        print()


def demo_circuit_breaker_behavior():
    """Demonstrate circuit breaker behavior during service failures."""
    print("⚡ CIRCUIT BREAKER DEMONSTRATION")
    print("-" * 50)

    # Set to hybrid mode to see circuit breaker in action
    os.environ["CHAT_MODE"] = "hybrid"
    os.environ["CIRCUIT_BREAKER_FAILURE_THRESHOLD"] = "3"  # Fail fast for demo

    engine = ChatEngine()

    print("Circuit breaker starts in CLOSED state (service assumed healthy)")

    # Simulate multiple failures to trigger circuit breaker
    failure_messages = [
        "This will likely fail due to network issues",
        "Another message that will fail",
        "Third failure to open circuit breaker",
        "This should use local fallback now",
        "And this one too",
    ]

    for i, message in enumerate(failure_messages, 1):
        print(f"\n{i}. Sending: '{message[:30]}...'")
        try:
            response = engine.process_chat(message, "cb_test")
            status = engine.get_system_status()

            print(f"   Response mode: {response['mode']}")
            print(f"   Circuit breaker state: {status['circuit_breaker_state']}")
            print(f"   Service status: {status['service_status']}")

            if response["mode"] == "fallback":
                print(f"   💡 Fallback reason: {response.get('fallback_reason', 'N/A')}")

        except Exception as e:
            print(f"   ❌ Exception: {e}")

        time.sleep(1)  # Allow circuit breaker timeout logic to work


def demo_configuration_options():
    """Show different configuration options and their effects."""
    print("⚙️  CONFIGURATION OPTIONS DEMONSTRATION")
    print("-" * 50)

    config_scenarios = [
        {
            "name": "Default Hybrid Mode",
            "env": {
                "CHAT_MODE": "hybrid",
                "OFFLINE_RESPONSES_ENABLED": "1",
                "SERVICE_HEALTH_CHECK_INTERVAL": "30",
            },
        },
        {
            "name": "Local Only Mode",
            "env": {
                "CHAT_MODE": "local",
                "OFFLINE_RESPONSES_ENABLED": "1",
            },
        },
        {
            "name": "Remote Only (No Fallback)",
            "env": {
                "CHAT_MODE": "remote",
                "OFFLINE_RESPONSES_ENABLED": "0",
            },
        },
        {
            "name": "Aggressive Circuit Breaker",
            "env": {
                "CHAT_MODE": "hybrid",
                "CIRCUIT_BREAKER_FAILURE_THRESHOLD": "2",
                "CIRCUIT_BREAKER_TIMEOUT_MS": "30000",
            },
        },
    ]

    for scenario in config_scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")

        # Apply configuration
        for key, value in scenario["env"].items():
            os.environ[key] = value
            print(f"   {key}={value}")

        try:
            engine = ChatEngine()
            engine.get_system_status()

            # Test a simple message
            response = engine.process_chat("Hello, how are you?", "config_test")

            print(f"   Result: mode={response['mode']}, " f"status={response['service_status']}")

        except Exception as e:
            print(f"   ❌ Error: {e}")


def demo_health_monitoring():
    """Demonstrate health monitoring and status reporting."""
    print("🩺 HEALTH MONITORING DEMONSTRATION")
    print("-" * 50)

    os.environ["CHAT_MODE"] = "hybrid"

    engine = ChatEngine()

    print("Testing Twin service health check...")

    # Manual health check
    is_healthy = engine.health_check_twin_service()
    print(f"Direct health check result: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")

    # Get comprehensive system status
    status = engine.get_system_status()

    print("\n📊 System Status Report:")
    print(f"  Mode: {status['mode']}")
    print(f"  Service Status: {status['service_status']}")
    print(f"  Circuit Breaker State: {status['circuit_breaker_state']}")
    print(f"  Twin URL: {status['twin_url']}")
    print(f"  Offline Responses: {'Enabled' if status['offline_responses_enabled'] else 'Disabled'}")
    print(f"  Calibration: {'Enabled' if status['calibration_enabled'] else 'Disabled'}")
    print(f"  Last Health Check: {status['last_health_check']}")

    # Circuit breaker statistics
    cb_stats = status["circuit_breaker_stats"]
    print("\n📈 Circuit Breaker Statistics:")
    print(f"  Total Calls: {cb_stats['total_calls']}")
    print(f"  Total Successes: {cb_stats['total_successes']}")
    print(f"  Total Failures: {cb_stats['total_failures']}")
    print(f"  Failure Rate: {cb_stats['failure_rate']:.2%}")
    print(f"  State Changes: {cb_stats['state_changes']}")


def main():
    """Run all demonstrations."""
    print("🚀 Starting Resilient Chat Engine Demo")
    print("=" * 70)
    print("This demo shows how the ChatEngine handles:")
    print("  • Offline fallback when Twin service (twin:8001) is unavailable")
    print("  • Circuit breaker pattern to prevent cascade failures")
    print("  • Different operation modes: remote, local, hybrid")
    print("  • Health monitoring and graceful degradation")
    print("=" * 70)

    try:
        # Demo 1: Basic chat modes
        demo_chat_modes()

        # Demo 2: Circuit breaker behavior
        demo_circuit_breaker_behavior()

        # Demo 3: Configuration options
        demo_configuration_options()

        # Demo 4: Health monitoring
        demo_health_monitoring()

        print("\n" + "=" * 70)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  ✓ System works offline with meaningful local responses")
        print("  ✓ Circuit breaker prevents cascade failures")
        print("  ✓ Health checks detect service availability")
        print("  ✓ Configuration allows different resilience strategies")
        print("  ✓ Graceful degradation maintains user experience")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
