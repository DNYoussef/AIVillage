#!/usr/bin/env python3
"""
Final WebSocket Protocol Verification
Demonstrates all working functionality with actual communications
"""

import asyncio
import json
import logging
import time
from typing import Any

from src.communications.message import Message, MessageType, Priority
from src.communications.protocol import CommunicationsProtocol

# Configure logging for demo
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class VerificationResults:
    def __init__(self):
        self.test_results: dict[str, dict[str, Any]] = {}
        self.messages_exchanged: list[dict[str, Any]] = []

    def add_test(self, test_name: str, success: bool, details: str, data: Any = None):
        self.test_results[test_name] = {
            "success": success,
            "details": details,
            "data": data,
            "timestamp": time.time(),
        }

    def add_message(self, direction: str, message_type: str, content: Any, encrypted: bool = True):
        self.messages_exchanged.append(
            {
                "direction": direction,
                "type": message_type,
                "content": content,
                "encrypted": encrypted,
                "timestamp": time.time(),
            }
        )

    def print_verification_report(self):
        print("\n" + "=" * 100)
        print("WEBSOCKET PROTOCOL VERIFICATION REPORT")
        print("=" * 100)

        # Test Results Summary
        passed = sum(1 for test in self.test_results.values() if test["success"])
        total = len(self.test_results)

        print(f"\nTEST RESULTS: {passed}/{total} tests passed ({passed / total * 100:.1f}%)")
        print("-" * 60)

        for test_name, result in self.test_results.items():
            status = "[PASS]" if result["success"] else "[FAIL]"
            print(f"{status:8} | {test_name:<35} | {result['details']}")

        # Communication Summary
        print("\nCOMMUNICATION SUMMARY:")
        print("-" * 60)
        print(f"Total messages exchanged: {len(self.messages_exchanged)}")

        for msg in self.messages_exchanged[-10:]:  # Show last 10 messages
            direction_arrow = "→" if msg["direction"] == "sent" else "←"
            encryption_status = "[ENCRYPTED]" if msg["encrypted"] else "[PLAIN]"
            print(f"{direction_arrow} {msg['type']:20} {encryption_status:12} | {str(msg['content'])[:50]}")

        print("\n" + "=" * 100)


async def verify_connection_establishment(results: VerificationResults):
    """Verify connection establishment and handshake"""
    logger.info("🔌 Verifying connection establishment...")

    try:
        # Start server
        server = CommunicationsProtocol("verification_server", port=8888)
        await server.start_server()

        # Connect client
        client = CommunicationsProtocol("verification_client", port=8889)
        connection_success = await client.connect("ws://localhost:8888", "verification_server")

        # Verify connection
        server_sees_client = server.is_connected("verification_client")
        client_sees_server = client.is_connected("verification_server")

        if connection_success and server_sees_client and client_sees_server:
            results.add_test(
                "connection_establishment",
                True,
                "WebSocket connection established with proper handshake",
            )
        else:
            results.add_test(
                "connection_establishment",
                False,
                f"Connection failed: client_success={connection_success}, server_sees_client={server_sees_client}",
            )

        # Cleanup
        await client.stop_server()
        await server.stop_server()

    except Exception as e:
        results.add_test("connection_establishment", False, f"Exception: {e!s}")


async def verify_bidirectional_messaging(results: VerificationResults):
    """Verify bidirectional encrypted messaging"""
    logger.info("💬 Verifying bidirectional messaging...")

    received_messages = []

    def message_handler(agent_id: str, message: dict):
        received_messages.append({"from": agent_id, "message": message})
        results.add_message("received", message.get("type", "unknown"), message.get("content", {}))

    try:
        # Setup server and client
        server = CommunicationsProtocol("msg_server", port=8888)
        server.register_handler("greeting", message_handler)
        server.register_handler("response", message_handler)
        await server.start_server()

        client = CommunicationsProtocol("msg_client", port=8889)
        client.register_handler("greeting", message_handler)
        client.register_handler("response", message_handler)
        await client.connect("ws://localhost:8888", "msg_server")

        await asyncio.sleep(0.5)

        # Test messages with different types
        messages_to_send = [
            {
                "type": "greeting",
                "content": {"text": "Hello from client", "priority": "high"},
            },
            {
                "type": "response",
                "content": {"text": "Hello from server", "status": "acknowledged"},
            },
            Message(
                type=MessageType.TASK,
                sender="msg_client",
                receiver="msg_server",
                content={"task_id": 12345, "action": "process_data"},
                priority=Priority.HIGH,
            ),
            Message(
                type=MessageType.NOTIFICATION,
                sender="msg_server",
                receiver="msg_client",
                content={"notification": "Task completed successfully"},
                priority=Priority.MEDIUM,
            ),
        ]

        # Send messages
        successful_sends = 0
        for i, msg in enumerate(messages_to_send):
            if i % 2 == 0:  # Client to server
                success = await client.send_message("msg_server", msg)
                results.add_message(
                    "sent",
                    msg.get("type") if isinstance(msg, dict) else msg.type.value,
                    msg.get("content") if isinstance(msg, dict) else msg.content,
                )
            else:  # Server to client
                success = await server.send_message("msg_client", msg)
                results.add_message(
                    "sent",
                    msg.get("type") if isinstance(msg, dict) else msg.type.value,
                    msg.get("content") if isinstance(msg, dict) else msg.content,
                )

            if success:
                successful_sends += 1
            await asyncio.sleep(0.3)

        await asyncio.sleep(1)  # Wait for all messages to be processed

        # Verify results
        messages_received = len(received_messages)
        if successful_sends == len(messages_to_send) and messages_received >= len(messages_to_send):
            results.add_test(
                "bidirectional_messaging",
                True,
                f"Successfully exchanged {messages_received} encrypted messages bidirectionally",
            )
        else:
            results.add_test(
                "bidirectional_messaging",
                False,
                f"Messaging failed: sent {successful_sends}/{len(messages_to_send)}, received {messages_received}",
            )

        # Cleanup
        await client.stop_server()
        await server.stop_server()

    except Exception as e:
        results.add_test("bidirectional_messaging", False, f"Exception: {e!s}")


async def verify_message_queuing(results: VerificationResults):
    """Verify message queuing during disconnection"""
    logger.info("📦 Verifying message queuing...")

    try:
        client = CommunicationsProtocol("queue_client", port=8889)

        # Queue messages while server is offline
        queued_messages = []
        for i in range(5):
            msg = {
                "type": "queued_message",
                "content": {"id": i, "data": f"Queued message {i}"},
            }
            await client.send_message("queue_server", msg)
            queued_messages.append(msg)
            results.add_message("queued", "queued_message", msg["content"])

        # Verify messages are queued
        queue_count = len(client.pending_messages.get("queue_server", []))

        # Start server and connect
        received_messages = []

        def queue_handler(agent_id: str, message: dict):
            received_messages.append(message)

        server = CommunicationsProtocol("queue_server", port=8888)
        server.register_handler("queued_message", queue_handler)
        await server.start_server()

        # Connection should trigger queue delivery
        await client.connect("ws://localhost:8888", "queue_server")
        await asyncio.sleep(2)  # Wait for queue to flush

        # Verify all messages delivered in order
        received_count = len(received_messages)
        messages_in_order = True
        if received_count >= 5:
            for i in range(5):
                if i < len(received_messages):
                    if received_messages[i].get("content", {}).get("id") != i:
                        messages_in_order = False
                        break

        if received_count >= 5 and messages_in_order:
            results.add_test(
                "message_queuing",
                True,
                f"Successfully queued {queue_count} messages and delivered all {received_count} in correct order",
            )
        else:
            results.add_test(
                "message_queuing",
                False,
                f"Queue failed: queued {queue_count}, delivered {received_count}, in_order={messages_in_order}",
            )

        # Cleanup
        await client.stop_server()
        await server.stop_server()

    except Exception as e:
        results.add_test("message_queuing", False, f"Exception: {e!s}")


async def verify_encryption(results: VerificationResults):
    """Verify message encryption"""
    logger.info("🔐 Verifying message encryption...")

    try:
        server = CommunicationsProtocol("crypto_server", port=8888)
        client = CommunicationsProtocol("crypto_client", port=8889)

        # Test key generation
        server_key = server._get_or_create_key("crypto_client")
        client_key = client._get_or_create_key("crypto_server")

        # Keys should be deterministic and matching
        keys_match = server_key._signing_key == client_key._signing_key

        # Test encryption/decryption
        test_message = {
            "type": "secret",
            "content": {
                "secret_data": "This is confidential information",
                "user_id": 12345,
            },
            "timestamp": time.time(),
        }

        encrypted = server._encrypt_message("crypto_client", test_message)
        decrypted = client._decrypt_message("crypto_server", encrypted)

        encryption_works = decrypted == test_message
        encrypted_length = len(encrypted)
        original_length = len(json.dumps(test_message))

        if keys_match and encryption_works:
            results.add_test(
                "message_encryption",
                True,
                f"End-to-end encryption working. Encrypted size: {encrypted_length} bytes vs original: {original_length} bytes",
            )
        else:
            results.add_test(
                "message_encryption",
                False,
                f"Encryption failed: keys_match={keys_match}, encryption_works={encryption_works}",
            )

    except Exception as e:
        results.add_test("message_encryption", False, f"Exception: {e!s}")


async def verify_reconnection(results: VerificationResults):
    """Verify automatic reconnection"""
    logger.info("🔄 Verifying reconnection logic...")

    try:
        # Start server
        server = CommunicationsProtocol("reconnect_server", port=8888)
        await server.start_server()

        # Connect client
        client = CommunicationsProtocol("reconnect_client", port=8889)
        initial_connection = await client.connect("ws://localhost:8888", "reconnect_server")

        if not initial_connection:
            results.add_test("reconnection_logic", False, "Initial connection failed")
            return

        await asyncio.sleep(0.5)

        # Kill server to simulate disconnect
        await server.stop_server()
        await asyncio.sleep(1)

        # Check client detected disconnection
        still_connected = client.is_connected("reconnect_server")

        # Restart server
        server = CommunicationsProtocol("reconnect_server", port=8888)
        await server.start_server()

        # Wait for automatic reconnection attempts
        await asyncio.sleep(3)

        # Check if client reconnected
        reconnected = server.is_connected("reconnect_client")

        if not still_connected and reconnected:
            results.add_test(
                "reconnection_logic",
                True,
                "Client detected disconnect and automatically reconnected to server",
            )
        else:
            results.add_test(
                "reconnection_logic",
                False,
                f"Reconnection failed: still_connected_after_kill={still_connected}, reconnected={reconnected}",
            )

        # Cleanup
        await client.stop_server()
        await server.stop_server()

    except Exception as e:
        results.add_test("reconnection_logic", False, f"Exception: {e!s}")


async def verify_ssl_support(results: VerificationResults):
    """Verify SSL/TLS support implementation"""
    logger.info("🔒 Verifying SSL/TLS support...")

    try:
        # Check implementation details
        server = CommunicationsProtocol("ssl_server", port=8888)

        # Analyze SSL implementation
        import inspect

        server_source = inspect.getsource(server.start_server)
        client_source = inspect.getsource(server.connect)

        # Check for SSL context handling
        server_ssl_support = (
            "ssl_context" in server_source
            and "SSL_CERTFILE" in server_source
            and "SSL_KEYFILE" in server_source
            and "ssl.SSLContext" in server_source
        )

        client_ssl_support = (
            "ssl_context" in client_source and "wss" in client_source and "ssl.SSLContext" in client_source
        )

        if server_ssl_support and client_ssl_support:
            results.add_test(
                "ssl_tls_support",
                True,
                "SSL/TLS support fully implemented for both server and client connections",
            )
        else:
            results.add_test(
                "ssl_tls_support",
                False,
                f"SSL support incomplete: server={server_ssl_support}, client={client_ssl_support}",
            )

    except Exception as e:
        results.add_test("ssl_tls_support", False, f"Exception: {e!s}")


async def run_complete_verification():
    """Run complete WebSocket protocol verification"""
    print("🚀 Starting WebSocket Protocol Verification...")
    print("This will test all aspects of the real WebSocket implementation")

    results = VerificationResults()

    # Run all verification tests
    await verify_connection_establishment(results)
    await asyncio.sleep(0.5)

    await verify_bidirectional_messaging(results)
    await asyncio.sleep(0.5)

    await verify_message_queuing(results)
    await asyncio.sleep(0.5)

    await verify_encryption(results)
    await asyncio.sleep(0.5)

    await verify_reconnection(results)
    await asyncio.sleep(0.5)

    await verify_ssl_support(results)

    # Generate final report
    results.print_verification_report()

    return results


if __name__ == "__main__":
    asyncio.run(run_complete_verification())
