#!/usr/bin/env python3
"""
Simple focused test for message delivery issue
"""

import asyncio
import logging

from src.communications.protocol import CommunicationsProtocol

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def test_simple_message_delivery():
    """Simple message delivery test"""
    server_messages = []
    client_messages = []

    def server_handler(agent_id: str, message: dict):
        server_messages.append(message)
        logger.info(f"SERVER received from {agent_id}: {message}")

    def client_handler(agent_id: str, message: dict):
        client_messages.append(message)
        logger.info(f"CLIENT received from {agent_id}: {message}")

    # Start server
    server = CommunicationsProtocol("server", port=8888)
    server.register_handler("test", server_handler)
    await server.start_server()
    logger.info("Server started")

    # Connect client
    client = CommunicationsProtocol("client", port=8889)
    client.register_handler("response", client_handler)
    connected = await client.connect("ws://localhost:8888", "server")
    logger.info(f"Client connected: {connected}")

    await asyncio.sleep(1)

    # Send client to server
    msg1 = {"type": "test", "content": "hello server"}
    result1 = await client.send_message("server", msg1)
    logger.info(f"Client sent message: {result1}")

    await asyncio.sleep(1)

    # Send server to client
    msg2 = {"type": "response", "content": "hello client"}
    result2 = await server.send_message("client", msg2)
    logger.info(f"Server sent message: {result2}")

    await asyncio.sleep(1)

    print(f"Server received {len(server_messages)} messages")
    print(f"Client received {len(client_messages)} messages")

    # Clean up
    await client.stop_server()
    await server.stop_server()


if __name__ == "__main__":
    asyncio.run(test_simple_message_delivery())
