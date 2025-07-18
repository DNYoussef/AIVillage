"""Standardized Communication Interface

This module defines the standard interface for inter-agent communication
protocols and message handling systems.
"""

from abc import ABC, abstractmethod
import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .agent_interface import MessageInterface


class ProtocolCapability(Enum):
    """Standard communication protocol capabilities."""

    POINT_TO_POINT = "point_to_point"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    REQUEST_RESPONSE = "request_response"
    STREAMING = "streaming"
    RELIABLE_DELIVERY = "reliable_delivery"
    ORDERED_DELIVERY = "ordered_delivery"
    ENCRYPTION = "encryption"
    COMPRESSION = "compression"
    PRIORITY_QUEUING = "priority_queuing"
    LOAD_BALANCING = "load_balancing"
    FAULT_TOLERANCE = "fault_tolerance"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"


class MessagePriority(Enum):
    """Standard message priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class DeliveryGuarantee(Enum):
    """Message delivery guarantee levels."""

    AT_MOST_ONCE = "at_most_once"  # May lose messages
    AT_LEAST_ONCE = "at_least_once"  # May duplicate messages
    EXACTLY_ONCE = "exactly_once"  # Guaranteed single delivery


@dataclass
class MessageStats:
    """Statistics for message handling."""

    total_sent: int = 0
    total_received: int = 0
    total_failed: int = 0
    total_dropped: int = 0
    average_latency_ms: float = 0.0
    bytes_sent: int = 0
    bytes_received: int = 0
    last_activity: datetime | None = None

    @property
    def success_rate(self) -> float:
        """Calculate message success rate."""
        total_attempts = self.total_sent + self.total_failed
        if total_attempts == 0:
            return 0.0
        return self.total_sent / total_attempts


@dataclass
class ProtocolConfig:
    """Configuration for communication protocols."""

    protocol_name: str
    capabilities: set[ProtocolCapability]
    max_message_size: int = 1024 * 1024  # 1MB default
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE
    enable_compression: bool = False
    enable_encryption: bool = False
    max_queue_size: int = 1000
    metadata: dict[str, Any] = field(default_factory=dict)


class MessageProtocol(ABC):
    """Abstract base class for message protocols.

    Defines the standard interface for low-level message transport.
    """

    def __init__(self, config: ProtocolConfig):
        self.config = config
        self.stats = MessageStats()
        self._handlers: dict[str, Callable] = {}
        self._subscriptions: dict[str, set[str]] = {}  # topic -> subscriber_ids
        self._connection_pool: dict[str, Any] = {}

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the protocol."""

    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown the protocol."""

    @abstractmethod
    async def send_message(
        self,
        message: MessageInterface,
        delivery_guarantee: DeliveryGuarantee | None = None,
    ) -> bool:
        """Send a message using this protocol.

        Args:
            message: Message to send
            delivery_guarantee: Override default delivery guarantee

        Returns:
            bool: True if message was sent successfully
        """

    @abstractmethod
    async def receive_message(
        self, timeout_seconds: float | None = None
    ) -> MessageInterface | None:
        """Receive a message using this protocol.

        Args:
            timeout_seconds: Timeout for receive operation

        Returns:
            Received message or None if timeout
        """

    @abstractmethod
    async def broadcast_message(
        self, message: MessageInterface, recipients: list[str]
    ) -> dict[str, bool]:
        """Broadcast message to multiple recipients.

        Args:
            message: Message to broadcast
            recipients: List of recipient IDs

        Returns:
            Dictionary mapping recipient IDs to success status
        """

    # Optional methods for advanced protocols

    async def subscribe(self, topic: str, subscriber_id: str) -> bool:
        """Subscribe to a topic for publish-subscribe messaging."""
        if ProtocolCapability.PUBLISH_SUBSCRIBE not in self.config.capabilities:
            return False

        if topic not in self._subscriptions:
            self._subscriptions[topic] = set()

        self._subscriptions[topic].add(subscriber_id)
        return True

    async def unsubscribe(self, topic: str, subscriber_id: str) -> bool:
        """Unsubscribe from a topic."""
        if topic in self._subscriptions:
            self._subscriptions[topic].discard(subscriber_id)
            if not self._subscriptions[topic]:
                del self._subscriptions[topic]
        return True

    async def publish(self, topic: str, message: MessageInterface) -> int:
        """Publish message to topic subscribers.

        Returns:
            Number of subscribers that received the message
        """
        if ProtocolCapability.PUBLISH_SUBSCRIBE not in self.config.capabilities:
            return 0

        subscribers = self._subscriptions.get(topic, set())
        successful_deliveries = 0

        for subscriber_id in subscribers:
            # Create copy of message for each subscriber
            subscriber_message = MessageInterface(
                message_id=f"{message.message_id}-{subscriber_id}",
                sender=message.sender,
                receiver=subscriber_id,
                message_type=message.message_type,
                content=message.content,
                priority=message.priority,
                context={**message.context, "topic": topic},
            )

            if await self.send_message(subscriber_message):
                successful_deliveries += 1

        return successful_deliveries

    def register_handler(self, message_type: str, handler: Callable) -> None:
        """Register a handler for specific message type."""
        self._handlers[message_type] = handler

    def unregister_handler(self, message_type: str) -> None:
        """Unregister handler for message type."""
        self._handlers.pop(message_type, None)

    async def handle_message(self, message: MessageInterface) -> Any:
        """Handle incoming message using registered handlers."""
        handler = self._handlers.get(message.message_type)
        if handler:
            if asyncio.iscoroutinefunction(handler):
                return await handler(message)
            return handler(message)
        return None

    def get_stats(self) -> MessageStats:
        """Get message statistics."""
        return self.stats

    def has_capability(self, capability: ProtocolCapability) -> bool:
        """Check if protocol has specific capability."""
        return capability in self.config.capabilities


class CommunicationInterface(ABC):
    """High-level communication interface for agent messaging.

    This interface abstracts the underlying protocol details and provides
    a consistent API for agent communication.
    """

    def __init__(self):
        self.protocols: dict[str, MessageProtocol] = {}
        self.default_protocol: str | None = None
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._message_handlers: dict[str, Callable] = {}

    @abstractmethod
    async def initialize(self, protocols: list[MessageProtocol]) -> bool:
        """Initialize communication system with protocols.

        Args:
            protocols: List of message protocols to use

        Returns:
            bool: True if initialization successful
        """

    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown communication system."""

    async def send_message(
        self,
        message: MessageInterface,
        protocol: str | None = None,
        delivery_guarantee: DeliveryGuarantee | None = None,
    ) -> bool:
        """Send message using specified or default protocol.

        Args:
            message: Message to send
            protocol: Protocol name to use (default if None)
            delivery_guarantee: Delivery guarantee level

        Returns:
            bool: True if message sent successfully
        """
        protocol_name = protocol or self.default_protocol
        if not protocol_name or protocol_name not in self.protocols:
            return False

        protocol_instance = self.protocols[protocol_name]
        return await protocol_instance.send_message(message, delivery_guarantee)

    async def receive_message(
        self, timeout_seconds: float | None = None
    ) -> MessageInterface | None:
        """Receive next message from queue.

        Args:
            timeout_seconds: Timeout for receive operation

        Returns:
            Next message or None if timeout
        """
        try:
            return await asyncio.wait_for(
                self.message_queue.get(), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            return None

    async def broadcast_message(
        self,
        message: MessageInterface,
        recipients: list[str],
        protocol: str | None = None,
    ) -> dict[str, bool]:
        """Broadcast message to multiple recipients.

        Args:
            message: Message to broadcast
            recipients: List of recipient agent IDs
            protocol: Protocol to use for broadcast

        Returns:
            Dictionary mapping recipient IDs to success status
        """
        protocol_name = protocol or self.default_protocol
        if not protocol_name or protocol_name not in self.protocols:
            return dict.fromkeys(recipients, False)

        protocol_instance = self.protocols[protocol_name]
        return await protocol_instance.broadcast_message(message, recipients)

    async def subscribe(
        self, topic: str, subscriber_id: str, protocol: str | None = None
    ) -> bool:
        """Subscribe to topic for publish-subscribe messaging."""
        protocol_name = protocol or self.default_protocol
        if not protocol_name or protocol_name not in self.protocols:
            return False

        protocol_instance = self.protocols[protocol_name]
        return await protocol_instance.subscribe(topic, subscriber_id)

    async def publish(
        self, topic: str, message: MessageInterface, protocol: str | None = None
    ) -> int:
        """Publish message to topic subscribers."""
        protocol_name = protocol or self.default_protocol
        if not protocol_name or protocol_name not in self.protocols:
            return 0

        protocol_instance = self.protocols[protocol_name]
        return await protocol_instance.publish(topic, message)

    def register_message_handler(self, message_type: str, handler: Callable) -> None:
        """Register handler for specific message type."""
        self._message_handlers[message_type] = handler

    async def start_message_processing(self) -> None:
        """Start background message processing."""
        self._running = True

        # Start message receiving tasks for each protocol
        tasks = []
        for protocol_name, protocol in self.protocols.items():
            task = asyncio.create_task(
                self._message_receiver_loop(protocol_name, protocol)
            )
            tasks.append(task)

        # Start message processing task
        process_task = asyncio.create_task(self._message_processor_loop())
        tasks.append(process_task)

        # Wait for tasks
        try:
            await asyncio.gather(*tasks)
        except Exception:
            # Handle shutdown or errors
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def stop_message_processing(self) -> None:
        """Stop background message processing."""
        self._running = False

    async def _message_receiver_loop(
        self, protocol_name: str, protocol: MessageProtocol
    ) -> None:
        """Background loop for receiving messages from protocol."""
        while self._running:
            try:
                message = await protocol.receive_message(timeout_seconds=1.0)
                if message:
                    await self.message_queue.put(message)
            except Exception as e:
                # Log error but continue
                print(f"Error receiving message from {protocol_name}: {e}")
                await asyncio.sleep(1.0)

    async def _message_processor_loop(self) -> None:
        """Background loop for processing received messages."""
        while self._running:
            try:
                message = await self.receive_message(timeout_seconds=1.0)
                if message:
                    # Handle message using registered handlers
                    handler = self._message_handlers.get(message.message_type)
                    if handler:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(message)
                            else:
                                handler(message)
                        except Exception as e:
                            print(f"Error handling message {message.message_id}: {e}")
            except Exception as e:
                print(f"Error in message processor: {e}")
                await asyncio.sleep(1.0)

    def get_protocol_stats(self) -> dict[str, MessageStats]:
        """Get statistics for all protocols."""
        return {name: protocol.get_stats() for name, protocol in self.protocols.items()}

    def get_available_protocols(self) -> list[str]:
        """Get list of available protocol names."""
        return list(self.protocols.keys())

    def get_protocol_capabilities(self, protocol: str) -> set[ProtocolCapability]:
        """Get capabilities of specific protocol."""
        if protocol in self.protocols:
            return self.protocols[protocol].config.capabilities
        return set()


# Utility functions


def create_protocol_config(
    name: str, capabilities: list[ProtocolCapability], **config_options
) -> ProtocolConfig:
    """Create protocol configuration with specified capabilities.

    Args:
        name: Protocol name
        capabilities: List of protocol capabilities
        **config_options: Additional configuration options

    Returns:
        ProtocolConfig instance
    """
    return ProtocolConfig(
        protocol_name=name, capabilities=set(capabilities), **config_options
    )


def validate_communication_interface(comm: Any) -> bool:
    """Validate that an object implements CommunicationInterface.

    Args:
        comm: Object to validate

    Returns:
        bool: True if object implements interface correctly
    """
    required_methods = [
        "initialize",
        "shutdown",
        "send_message",
        "receive_message",
        "broadcast_message",
        "subscribe",
        "publish",
    ]

    for method in required_methods:
        if not hasattr(comm, method) or not callable(getattr(comm, method)):
            return False

    return True


def validate_protocol_interface(protocol: Any) -> bool:
    """Validate that an object implements MessageProtocol.

    Args:
        protocol: Object to validate

    Returns:
        bool: True if object implements interface correctly
    """
    required_methods = [
        "initialize",
        "shutdown",
        "send_message",
        "receive_message",
        "broadcast_message",
    ]

    for method in required_methods:
        if not hasattr(protocol, method) or not callable(getattr(protocol, method)):
            return False

    required_attributes = ["config", "stats"]
    for attr in required_attributes:
        if not hasattr(protocol, attr):
            return False

    return True
