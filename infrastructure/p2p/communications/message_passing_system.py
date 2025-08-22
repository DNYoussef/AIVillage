"""High-Level Message Passing System.

Provides simplified interfaces for agent-to-agent communication:
- Direct agent messaging
- Broadcast messaging
- Message routing
- Protocol abstraction
"""

import asyncio
import logging
from typing import Any

from .message import Message
from .protocol import StandardCommunicationProtocol
from .service_discovery import discover_services

logger = logging.getLogger(__name__)


class MessagePassingSystem:
    """High-level message passing system for agent communication."""

    def __init__(self, agent_id: str, port: int | None = None) -> None:
        self.agent_id = agent_id
        self.port = port or self._find_available_port()
        self.protocol = StandardCommunicationProtocol(agent_id, self.port)
        self.message_handlers: dict[str, callable] = {}
        self.running = False

    def _find_available_port(self) -> int:
        """Find an available port for this agent."""
        # Simple port allocation strategy
        base_port = 8000
        agent_hash = hash(self.agent_id) % 1000
        return base_port + agent_hash

    async def start(self) -> None:
        """Start the message passing system."""
        if self.running:
            return

        await self.protocol.start_server()
        self.running = True
        logger.info(f"Message passing system started for {self.agent_id} on port {self.port}")

    async def stop(self) -> None:
        """Stop the message passing system."""
        if not self.running:
            return

        await self.protocol.disconnect()
        self.running = False
        logger.info(f"Message passing system stopped for {self.agent_id}")

    def register_handler(self, message_type: str, handler: callable) -> None:
        """Register a handler for specific message types."""
        self.message_handlers[message_type] = handler
        self.protocol.register_handler(message_type, handler)
        logger.info(f"Registered handler for message type: {message_type}")

    async def send_message(
        self,
        target_agent_id: str,
        message_type: str,
        payload: Any,
        metadata: dict | None = None,
    ) -> bool:
        """Send a message to a specific agent."""
        try:
            # Try to find the target agent's service info
            services = await discover_services()
            target_service = None

            for service in services:
                if service.agent_id == target_agent_id:
                    target_service = service
                    break

            if not target_service:
                logger.warning(f"Could not find service info for agent: {target_agent_id}")
                # Try direct connection with default assumptions
                target_url = "ws://localhost:8000/ws"  # Default fallback
            else:
                target_url = f"ws://{target_service.host}:{target_service.port}/ws"

            # Create message
            message = Message(
                sender_id=self.agent_id,
                recipient_id=target_agent_id,
                message_type=message_type,
                payload=payload,
                metadata=metadata or {},
            )

            # Connect and send
            connected = await self.protocol.connect(target_agent_id, target_url)
            if connected:
                success = await self.protocol.send_message(target_agent_id, message)
                logger.info(f"Message sent to {target_agent_id}: {success}")
                return success
            logger.error(f"Failed to connect to {target_agent_id}")
            return False

        except Exception as e:
            logger.exception(f"Error sending message to {target_agent_id}: {e}")
            return False

    async def broadcast_message(
        self,
        message_type: str,
        payload: Any,
        service_type: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Broadcast a message to all agents or agents of a specific service type."""
        try:
            # Discover target agents
            services = await discover_services(service_type)

            # Send to each agent
            sent_count = 0
            for service in services:
                if service.agent_id != self.agent_id:  # Don't send to self
                    success = await self.send_message(service.agent_id, message_type, payload, metadata)
                    if success:
                        sent_count += 1

            logger.info(f"Broadcast message sent to {sent_count} agents")
            return sent_count

        except Exception as e:
            logger.exception(f"Error broadcasting message: {e}")
            return 0

    async def send_request_response(
        self,
        target_agent_id: str,
        request_type: str,
        payload: Any,
        timeout: float = 30.0,
        metadata: dict | None = None,
    ) -> Message | None:
        """Send a request and wait for a response."""
        try:
            # Create request message with correlation ID
            import uuid

            correlation_id = str(uuid.uuid4())

            request_metadata = metadata or {}
            request_metadata.update({"correlation_id": correlation_id, "expects_response": True})

            # Set up response handler
            response_future = asyncio.Future()

            def response_handler(message: Message) -> None:
                if message.metadata.get("correlation_id") == correlation_id and message.metadata.get(
                    "is_response", False
                ):
                    if not response_future.done():
                        response_future.set_result(message)

            # Register temporary handler
            response_type = f"{request_type}_response"
            self.register_handler(response_type, response_handler)

            try:
                # Send request
                success = await self.send_message(target_agent_id, request_type, payload, request_metadata)

                if not success:
                    return None

                # Wait for response
                response = await asyncio.wait_for(response_future, timeout=timeout)
                return response

            finally:
                # Cleanup handler
                if response_type in self.message_handlers:
                    del self.message_handlers[response_type]
                    self.protocol.unregister_handler(response_type)

        except TimeoutError:
            logger.warning(f"Request to {target_agent_id} timed out after {timeout}s")
            return None
        except Exception as e:
            logger.exception(f"Error in request-response to {target_agent_id}: {e}")
            return None

    async def send_response(
        self,
        request_message: Message,
        response_payload: Any,
        metadata: dict | None = None,
    ) -> bool:
        """Send a response to a previous request."""
        try:
            correlation_id = request_message.metadata.get("correlation_id")
            if not correlation_id:
                logger.warning("Cannot send response: no correlation ID in request")
                return False

            response_metadata = metadata or {}
            response_metadata.update({"correlation_id": correlation_id, "is_response": True})

            response_type = f"{request_message.message_type}_response"

            return await self.send_message(
                request_message.sender_id,
                response_type,
                response_payload,
                response_metadata,
            )

        except Exception as e:
            logger.exception(f"Error sending response: {e}")
            return False

    def get_connection_info(self) -> dict:
        """Get connection information for this agent."""
        return {
            "agent_id": self.agent_id,
            "host": "localhost",
            "port": self.port,
            "status": "running" if self.running else "stopped",
            "connections": len(self.protocol.connections),
            "message_handlers": list(self.message_handlers.keys()),
        }


# Convenience functions
async def create_message_system(agent_id: str, port: int | None = None) -> MessagePassingSystem:
    """Create and start a message passing system."""
    system = MessagePassingSystem(agent_id, port)
    await system.start()
    return system


# Legacy compatibility
class MessagePassing(MessagePassingSystem):
    """Legacy alias for MessagePassingSystem."""
