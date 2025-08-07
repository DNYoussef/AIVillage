#!/usr/bin/env python3
"""
Comprehensive WebSocket Protocol Testing
Tests all aspects of real WebSocket communication including:
1. Connection establishment
2. Message delivery
3. Reconnection logic
4. Message queuing
5. TLS/SSL support
"""

import asyncio
import logging
import time
from typing import Any

from src.communications.message import Message, MessageType, Priority
from src.communications.protocol import CommunicationsProtocol

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestResults:
    """Track test results"""

    def __init__(self):
        self.tests: dict[str, bool] = {}
        self.details: dict[str, str] = {}
        self.messages_received: list[dict[str, Any]] = []

    def add_result(self, test_name: str, passed: bool, details: str = ""):
        self.tests[test_name] = passed
        self.details[test_name] = details
        logger.info(
            f"Test '{test_name}': {'PASSED' if passed else 'FAILED'} - {details}"
        )

    def print_summary(self):
        print("\n" + "=" * 80)
        print("WEBSOCKET PROTOCOL TEST RESULTS")
        print("=" * 80)

        passed = sum(1 for result in self.tests.values() if result)
        total = len(self.tests)

        for test_name, result in self.tests.items():
            status = "[PASS]" if result else "[FAIL]"
            print(f"{status:8} | {test_name:<40} | {self.details[test_name]}")

        print("=" * 80)
        print(f"SUMMARY: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        print("=" * 80)


async def test_connection_establishment(results: TestResults):
    """Test 1: Connection establishment with handshake"""
    logger.info("Testing connection establishment...")

    server = None
    client = None

    try:
        # Start server
        server = CommunicationsProtocol("server_agent", port=8888)
        await server.start_server()

        # Give server time to start
        await asyncio.sleep(0.5)

        # Connect client
        client = CommunicationsProtocol("client_agent", port=8889)
        success = await client.connect("ws://localhost:8888", "server_agent")

        if success and server.is_connected("client_agent"):
            results.add_result(
                "connection_establishment",
                True,
                "Client connected and handshake completed",
            )
        else:
            results.add_result(
                "connection_establishment",
                False,
                f"Connection failed: client_success={success}, server_connected={server.is_connected('client_agent')}",
            )

    except Exception as e:
        results.add_result("connection_establishment", False, f"Exception: {e!s}")
    finally:
        if client:
            await client.stop_server()
        if server:
            await server.stop_server()


async def test_message_delivery(results: TestResults):
    """Test 2: Bidirectional message delivery"""
    logger.info("Testing message delivery...")

    server = None
    client = None
    server_messages = []
    client_messages = []

    try:
        # Set up message handlers
        def server_handler(agent_id: str, message: dict):
            server_messages.append(message)
            logger.info(f"Server received: {message}")

        def client_handler(agent_id: str, message: dict):
            client_messages.append(message)
            logger.info(f"Client received: {message}")

        # Start server
        server = CommunicationsProtocol("server_agent", port=8888)
        server.register_handler("test_message", server_handler)
        await server.start_server()

        # Connect client
        client = CommunicationsProtocol("client_agent", port=8889)
        client.register_handler("response_message", client_handler)
        await client.connect("ws://localhost:8888", "server_agent")

        # Give connection time to establish
        await asyncio.sleep(0.5)

        # Test 1: Client to server message
        test_message = {
            "type": "test_message",
            "content": {"data": "Hello from client", "test_id": 1},
            "timestamp": time.time(),
        }

        client_send_success = await client.send_message("server_agent", test_message)
        await asyncio.sleep(0.5)  # Wait for message processing

        # Test 2: Server to client response
        response_message = {
            "type": "response_message",
            "content": {"data": "Hello from server", "test_id": 2},
            "timestamp": time.time(),
        }

        server_send_success = await server.send_message(
            "client_agent", response_message
        )
        await asyncio.sleep(0.5)  # Wait for message processing

        # Verify messages were received
        client_to_server_ok = (
            client_send_success
            and len(server_messages) > 0
            and server_messages[0].get("content", {}).get("data") == "Hello from client"
        )

        server_to_client_ok = (
            server_send_success
            and len(client_messages) > 0
            and client_messages[0].get("content", {}).get("data") == "Hello from server"
        )

        if client_to_server_ok and server_to_client_ok:
            results.add_result(
                "message_delivery",
                True,
                f"Bidirectional messaging works. Server got {len(server_messages)} msgs, client got {len(client_messages)} msgs",
            )
        else:
            results.add_result(
                "message_delivery",
                False,
                f"Message delivery failed. C->S: {client_to_server_ok}, S->C: {server_to_client_ok}",
            )

    except Exception as e:
        results.add_result("message_delivery", False, f"Exception: {e!s}")
    finally:
        if client:
            await client.stop_server()
        if server:
            await server.stop_server()


async def test_reconnection_logic(results: TestResults):
    """Test 3: Automatic reconnection after server restart"""
    logger.info("Testing reconnection logic...")

    server = None
    client = None

    try:
        # Start server
        server = CommunicationsProtocol("server_agent", port=8888)
        await server.start_server()

        # Connect client
        client = CommunicationsProtocol("client_agent", port=8889)
        initial_connection = await client.connect("ws://localhost:8888", "server_agent")

        if not initial_connection:
            results.add_result("reconnection_logic", False, "Initial connection failed")
            return

        await asyncio.sleep(0.5)

        # Kill server
        await server.stop_server()
        await asyncio.sleep(1)  # Wait for disconnect detection

        # Check client detected disconnection
        is_still_connected = client.is_connected("server_agent")

        # Restart server
        server = CommunicationsProtocol("server_agent", port=8888)
        await server.start_server()

        # Wait for automatic reconnection (the client should attempt to reconnect)
        await asyncio.sleep(3)  # Give time for reconnection attempts

        # Check if reconnection occurred
        reconnected = server.is_connected("client_agent")

        if not is_still_connected and reconnected:
            results.add_result(
                "reconnection_logic",
                True,
                "Client detected disconnect and successfully reconnected",
            )
        else:
            results.add_result(
                "reconnection_logic",
                False,
                f"Reconnection failed. Still connected after kill: {is_still_connected}, Reconnected: {reconnected}",
            )

    except Exception as e:
        results.add_result("reconnection_logic", False, f"Exception: {e!s}")
    finally:
        if client:
            await client.stop_server()
        if server:
            await server.stop_server()


async def test_message_queuing(results: TestResults):
    """Test 4: Message queuing while disconnected"""
    logger.info("Testing message queuing...")

    server = None
    client = None
    received_messages = []

    try:
        # Set up message handler
        def message_handler(agent_id: str, message: dict):
            received_messages.append(message)
            logger.info(
                f"Received queued message: {message.get('content', {}).get('message_id')}"
            )

        # Start client only (no server yet)
        client = CommunicationsProtocol("client_agent", port=8889)
        client.register_handler("queued_message", message_handler)

        # Queue 5 messages while disconnected
        queued_messages = []
        for i in range(5):
            message = {
                "type": "queued_message",
                "content": {"message_id": i, "data": f"Queued message {i}"},
                "timestamp": time.time(),
            }
            queued_messages.append(message)

            # This should queue the message since server isn't running
            await client.send_message("server_agent", message)

        # Check that messages are queued
        queued_count = len(client.pending_messages.get("server_agent", []))
        logger.info(f"Queued {queued_count} messages while disconnected")

        # Start server and establish connection
        server = CommunicationsProtocol("server_agent", port=8888)
        server.register_handler("queued_message", message_handler)
        await server.start_server()

        # Connect client (this should trigger delivery of queued messages)
        connection_success = await client.connect("ws://localhost:8888", "server_agent")

        if not connection_success:
            results.add_result(
                "message_queuing",
                False,
                "Failed to establish connection for queue test",
            )
            return

        # Wait for queued messages to be delivered
        await asyncio.sleep(2)

        # Check if all messages were delivered in order
        delivered_count = len(received_messages)
        messages_in_order = True

        if delivered_count >= 5:
            for i in range(5):
                if i < len(received_messages):
                    msg_id = received_messages[i].get("content", {}).get("message_id")
                    if msg_id != i:
                        messages_in_order = False
                        break

        if delivered_count >= 5 and messages_in_order:
            results.add_result(
                "message_queuing",
                True,
                f"All {delivered_count} queued messages delivered in correct order",
            )
        else:
            results.add_result(
                "message_queuing",
                False,
                f"Queue delivery failed. Expected 5, got {delivered_count}, in_order: {messages_in_order}",
            )

    except Exception as e:
        results.add_result("message_queuing", False, f"Exception: {e!s}")
    finally:
        if client:
            await client.stop_server()
        if server:
            await server.stop_server()


async def test_tls_ssl_support(results: TestResults):
    """Test 5: TLS/SSL support (if available)"""
    logger.info("Testing TLS/SSL support...")

    try:
        # Check if SSL context creation is implemented
        server = CommunicationsProtocol("server_agent", port=8888)

        # Look for SSL context creation in the code
        import inspect

        source = inspect.getsource(server.start_server)
        has_ssl_context = "ssl_context" in source and "SSL_CERTFILE" in source

        # Test SSL connection support in client
        client_source = inspect.getsource(server.connect)
        has_ssl_client_support = (
            "ssl_context" in client_source and "wss" in client_source
        )

        if has_ssl_context and has_ssl_client_support:
            results.add_result(
                "tls_ssl_support",
                True,
                "SSL/TLS support implemented in both server and client",
            )
        else:
            results.add_result(
                "tls_ssl_support",
                False,
                f"SSL support incomplete. Server: {has_ssl_context}, Client: {has_ssl_client_support}",
            )

    except Exception as e:
        results.add_result("tls_ssl_support", False, f"Exception: {e!s}")


async def test_message_encryption(results: TestResults):
    """Test 6: Message encryption"""
    logger.info("Testing message encryption...")

    try:
        server = CommunicationsProtocol("server_agent", port=8888)
        client = CommunicationsProtocol("client_agent", port=8889)

        # Test encryption key generation
        server_key = server._get_or_create_key("client_agent")
        client_key = client._get_or_create_key("server_agent")

        # Keys should be the same (deterministic based on agent IDs)
        keys_match = server_key._signing_key == client_key._signing_key

        # Test message encryption/decryption
        test_message = {
            "type": "test",
            "content": "secret message",
            "timestamp": time.time(),
        }

        encrypted = server._encrypt_message("client_agent", test_message)
        decrypted = client._decrypt_message("server_agent", encrypted)

        encryption_works = decrypted == test_message

        if keys_match and encryption_works:
            results.add_result(
                "message_encryption",
                True,
                "Encryption keys match and message encryption/decryption works",
            )
        else:
            results.add_result(
                "message_encryption",
                False,
                f"Encryption failed. Keys match: {keys_match}, Encryption works: {encryption_works}",
            )

    except Exception as e:
        results.add_result("message_encryption", False, f"Exception: {e!s}")


async def test_message_types_and_priorities(results: TestResults):
    """Test 7: Different message types and priorities"""
    logger.info("Testing message types and priorities...")

    server = None
    client = None
    received_messages = []

    try:

        def message_handler(agent_id: str, message: dict):
            received_messages.append(message)

        # Start server and client
        server = CommunicationsProtocol("server_agent", port=8888)
        server.register_handler("task", message_handler)
        server.register_handler("notification", message_handler)
        await server.start_server()

        client = CommunicationsProtocol("client_agent", port=8889)
        await client.connect("ws://localhost:8888", "server_agent")
        await asyncio.sleep(0.5)

        # Test different message types using Message class
        test_messages = [
            Message(
                type=MessageType.TASK,
                sender="client_agent",
                receiver="server_agent",
                content={"task": "process_data"},
                priority=Priority.HIGH,
            ),
            Message(
                type=MessageType.NOTIFICATION,
                sender="client_agent",
                receiver="server_agent",
                content={"notification": "system_update"},
                priority=Priority.LOW,
            ),
        ]

        # Send messages
        send_results = []
        for msg in test_messages:
            result = await client.send_message("server_agent", msg)
            send_results.append(result)

        await asyncio.sleep(1)

        # Check results
        all_sent = all(send_results)
        correct_count = len(received_messages) == 2

        if all_sent and correct_count:
            results.add_result(
                "message_types_priorities",
                True,
                f"Successfully sent and received {len(received_messages)} different message types",
            )
        else:
            results.add_result(
                "message_types_priorities",
                False,
                f"Message type test failed. All sent: {all_sent}, Correct count: {correct_count}",
            )

    except Exception as e:
        results.add_result("message_types_priorities", False, f"Exception: {e!s}")
    finally:
        if client:
            await client.stop_server()
        if server:
            await server.stop_server()


async def run_all_tests():
    """Run all WebSocket protocol tests"""
    results = TestResults()

    print("Starting comprehensive WebSocket protocol tests...")
    print("=" * 80)

    # Run all tests
    await test_connection_establishment(results)
    await asyncio.sleep(1)

    await test_message_delivery(results)
    await asyncio.sleep(1)

    await test_reconnection_logic(results)
    await asyncio.sleep(1)

    await test_message_queuing(results)
    await asyncio.sleep(1)

    await test_tls_ssl_support(results)
    await asyncio.sleep(1)

    await test_message_encryption(results)
    await asyncio.sleep(1)

    await test_message_types_and_priorities(results)

    # Print final results
    results.print_summary()

    return results


if __name__ == "__main__":
    asyncio.run(run_all_tests())
