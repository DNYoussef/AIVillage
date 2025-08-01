"""Test script for HypeRAG MCP Server

Simple test to verify the server is working correctly.
"""

import asyncio
import json
import logging
from typing import Any

import websockets

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPTestClient:
    """Simple test client for HypeRAG MCP Server"""

    def __init__(self, uri: str = "ws://localhost:8765"):
        self.uri = uri
        self.websocket = None

    async def connect(self):
        """Connect to the server"""
        self.websocket = await websockets.connect(self.uri)
        logger.info(f"Connected to {self.uri}")

    async def disconnect(self):
        """Disconnect from the server"""
        if self.websocket:
            await self.websocket.close()
            logger.info("Disconnected")

    async def send_request(
        self, method: str, params: dict[str, Any], request_id: str = None
    ) -> dict[str, Any]:
        """Send a request and get response"""
        if not self.websocket:
            raise RuntimeError("Not connected")

        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": request_id or method.replace("/", "_"),
        }

        # Send request
        await self.websocket.send(json.dumps(request))
        logger.info(f"Sent: {method}")

        # Get response
        response_str = await self.websocket.recv()
        response = json.loads(response_str)
        logger.info(f"Received: {response.get('result', {}).get('status', 'unknown')}")

        return response


async def test_server():
    """Test the HypeRAG MCP Server"""
    client = MCPTestClient()

    try:
        await client.connect()

        # Test 1: Health check (no auth required)
        logger.info("=== Testing health check ===")
        response = await client.send_request("hyperag/health", {})
        assert response.get("result", {}).get("status") == "healthy"
        print("✓ Health check passed")

        # Test 2: Query without auth (should fail)
        logger.info("=== Testing query without auth ===")
        response = await client.send_request(
            "hyperag/query", {"query": "What is machine learning?"}
        )
        assert "error" in response
        assert response["error"]["code"] == "AUTH_REQUIRED"
        print("✓ Auth required check passed")

        # Test 3: Query with API key
        logger.info("=== Testing query with API key ===")
        response = await client.send_request(
            "hyperag/query",
            {
                "api_key": "hrag_dev_test123",
                "query": "What is machine learning?",
                "mode": "NORMAL",
            },
        )

        if "error" in response:
            print(f"✗ Query with auth failed: {response['error']['message']}")
        else:
            assert response.get("result", {}).get("status") == "success"
            print("✓ Query with auth passed")

        # Test 4: Knowledge search
        logger.info("=== Testing knowledge search ===")
        response = await client.send_request(
            "hyperag/knowledge/search",
            {"api_key": "hrag_dev_test123", "query": "test search", "limit": 5},
        )

        if "error" in response:
            print(f"✗ Knowledge search failed: {response['error']['message']}")
        else:
            print("✓ Knowledge search passed")

        # Test 5: Creative query
        logger.info("=== Testing creative query ===")
        response = await client.send_request(
            "hyperag/creative",
            {
                "api_key": "hrag_dev_test123",
                "source_concept": "computer",
                "target_concept": "brain",
                "creativity_parameters": {"mode": "analogical", "max_hops": 3},
            },
        )

        if "error" in response:
            print(f"✗ Creative query failed: {response['error']['message']}")
        else:
            print("✓ Creative query passed")

        # Test 6: List adapters
        logger.info("=== Testing adapter list ===")
        response = await client.send_request(
            "hyperag/adapter/list", {"api_key": "hrag_dev_test123"}
        )

        if "error" in response:
            print(f"✗ Adapter list failed: {response['error']['message']}")
        else:
            print("✓ Adapter list passed")

        # Test 7: Metrics (requires auth)
        logger.info("=== Testing metrics ===")
        response = await client.send_request(
            "hyperag/metrics",
            {
                "api_key": "hrag_prod_king456"  # King role needed for metrics
            },
        )

        if "error" in response:
            print(f"✗ Metrics failed: {response['error']['message']}")
        else:
            print("✓ Metrics passed")

        print("\n=== All tests completed ===")

    except Exception as e:
        logger.error(f"Test failed: {e!s}")
        print(f"✗ Test error: {e!s}")

    finally:
        await client.disconnect()


async def run_interactive_test():
    """Interactive test mode"""
    client = MCPTestClient()

    try:
        await client.connect()
        print("Connected to HypeRAG MCP Server")
        print("Enter JSON-RPC requests (or 'quit' to exit):")

        while True:
            try:
                user_input = input("\n> ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    break

                # Try to parse as JSON
                request_data = json.loads(user_input)
                method = request_data.get("method")
                params = request_data.get("params", {})
                request_id = request_data.get("id")

                response = await client.send_request(method, params, request_id)
                print(json.dumps(response, indent=2))

            except json.JSONDecodeError:
                print("Invalid JSON. Please enter a valid JSON-RPC request.")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e!s}")

    except Exception as e:
        print(f"Failed to connect: {e!s}")
        print("Make sure the HypeRAG MCP Server is running on ws://localhost:8765")

    finally:
        await client.disconnect()


async def main():
    """Main test function"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        await run_interactive_test()
    else:
        await test_server()


if __name__ == "__main__":
    asyncio.run(main())
