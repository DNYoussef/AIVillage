#!/usr/bin/env python3
"""
Test Script for Agent Forge Minimal Backend

Tests all endpoints to ensure they work correctly.
"""

import requests
import json
import time
import asyncio
import websockets
from datetime import datetime

BASE_URL = "http://localhost:8083"
WS_URL = "ws://localhost:8083/ws"


def test_endpoint(endpoint, method="GET", data=None, description=""):
    """Test a single endpoint."""
    url = f"{BASE_URL}{endpoint}"

    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            raise ValueError(f"Unsupported method: {method}")

        print(f"[{response.status_code}] {method} {endpoint} - {description}")

        if response.status_code == 200:
            result = response.json()
            if "message" in result:
                print(f"    Message: {result['message']}")
            if "models" in result:
                print(f"    Models: {len(result['models'])}")
            if "phases" in result:
                print(f"    Phases: {len(result['phases'])}")
            return True, result
        else:
            print(f"    Error: {response.text}")
            return False, None

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] {method} {endpoint} - Connection failed: {e}")
        return False, None
    except Exception as e:
        print(f"[ERROR] {method} {endpoint} - {e}")
        return False, None


async def test_websocket():
    """Test WebSocket connection."""
    try:
        print("\n[WebSocket] Testing WebSocket connection...")

        async with websockets.connect(WS_URL) as websocket:
            print("[OK] WebSocket connected")

            # Wait for welcome message
            welcome = await websocket.recv()
            welcome_data = json.loads(welcome)
            print(f"    Welcome: {welcome_data.get('message', 'No message')}")

            # Send ping
            await websocket.send(json.dumps({"type": "ping"}))
            pong = await websocket.recv()
            pong_data = json.loads(pong)
            print(f"    Ping/Pong: {pong_data.get('type', 'Unknown')}")

            # Request status
            await websocket.send(json.dumps({"type": "get_status"}))
            status = await websocket.recv()
            status_data = json.loads(status)
            print(f"    Status update type: {status_data.get('type', 'Unknown')}")

            print("[OK] WebSocket test completed")
            return True

    except Exception as e:
        print(f"[ERROR] WebSocket test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("AGENT FORGE BACKEND TEST SUITE")
    print("=" * 60)
    print(f"Testing backend at: {BASE_URL}")
    print(f"WebSocket at: {WS_URL}")
    print()

    results = []

    # Test basic endpoints
    print("=== Basic Endpoints ===")
    success, data = test_endpoint("/", description="Root endpoint")
    results.append(success)

    success, data = test_endpoint("/health", description="Health check")
    results.append(success)

    # Test phase management
    print("\n=== Phase Management ===")
    success, data = test_endpoint("/phases/status", description="Get phase status")
    results.append(success)

    success, data = test_endpoint("/phases/cognate/start", method="POST", description="Start Cognate phase")
    results.append(success)

    if success:
        print("    Waiting 3 seconds for phase to start...")
        time.sleep(3)

        success, data = test_endpoint("/phases/status", description="Check phase progress")
        results.append(success)

    # Test model endpoints
    print("\n=== Model Management ===")
    success, data = test_endpoint("/models", description="Get models")
    results.append(success)

    models_available = data and len(data.get("models", [])) > 0

    if models_available:
        model_id = data["models"][0]["model_id"]
        print(f"    Testing chat with model: {model_id}")

        chat_data = {"model_id": model_id, "message": "Hello, can you introduce yourself?"}
        success, data = test_endpoint("/chat", method="POST", data=chat_data, description="Test chat interface")
        results.append(success)
    else:
        print("    No models available for chat test (this is expected if Cognate phase hasn't completed)")

    # Test WebSocket (async)
    print("\n=== WebSocket Test ===")
    try:
        ws_success = asyncio.run(test_websocket())
        results.append(ws_success)
    except Exception as e:
        print(f"[ERROR] WebSocket async test failed: {e}")
        results.append(False)

    # Test web interface
    print("\n=== Web Interface ===")
    success, data = test_endpoint("/test", description="Test HTML interface")
    results.append(success)

    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")

    if passed == total:
        print("\n[SUCCESS] All tests passed! Backend is working correctly.")
    elif passed >= total * 0.8:
        print("\n[WARNING] Most tests passed. Backend is mostly functional.")
    else:
        print("\n[ERROR] Many tests failed. Backend may have issues.")

    print("\nTo manually test:")
    print(f"- Open browser: {BASE_URL}/test")
    print(f"- API Health: {BASE_URL}/health")
    print(f"- Start Cognate: curl -X POST {BASE_URL}/phases/cognate/start")
    print(f"- Check Status: curl {BASE_URL}/phases/status")


if __name__ == "__main__":
    main()
