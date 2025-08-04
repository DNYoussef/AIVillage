"""Evolution-Aware Message Protocol for P2P Communication."""

import asyncio
from dataclasses import dataclass
from enum import Enum
import json
import logging
import struct
import time
from typing import Any
import uuid

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Standard message types for P2P communication."""

    # Basic networking
    PING = "PING"
    PONG = "PONG"
    HEARTBEAT = "HEARTBEAT"

    # Peer management
    PEER_DISCOVERY = "PEER_DISCOVERY"
    PEER_INTRODUCTION = "PEER_INTRODUCTION"
    PEER_ANNOUNCEMENT = "PEER_ANNOUNCEMENT"
    PEER_GOODBYE = "PEER_GOODBYE"

    # Evolution coordination
    EVOLUTION_START = "EVOLUTION_START"
    EVOLUTION_PROGRESS = "EVOLUTION_PROGRESS"
    EVOLUTION_COMPLETE = "EVOLUTION_COMPLETE"
    EVOLUTION_REQUEST_HELP = "EVOLUTION_REQUEST_HELP"
    EVOLUTION_OFFER_HELP = "EVOLUTION_OFFER_HELP"
    EVOLUTION_ACCEPT_HELP = "EVOLUTION_ACCEPT_HELP"
    EVOLUTION_DECLINE_HELP = "EVOLUTION_DECLINE_HELP"

    # Distributed consensus
    EVOLUTION_PROPOSAL = "EVOLUTION_PROPOSAL"
    EVOLUTION_VOTE = "EVOLUTION_VOTE"
    EVOLUTION_CONSENSUS = "EVOLUTION_CONSENSUS"

    # Data sharing
    EVOLUTION_METRICS_SHARE = "EVOLUTION_METRICS_SHARE"
    EVOLUTION_RESULTS_SHARE = "EVOLUTION_RESULTS_SHARE"
    EVOLUTION_KNOWLEDGE_SHARE = "EVOLUTION_KNOWLEDGE_SHARE"

    # Resource management
    RESOURCE_STATUS = "RESOURCE_STATUS"
    RESOURCE_REQUEST = "RESOURCE_REQUEST"
    RESOURCE_ALLOCATION = "RESOURCE_ALLOCATION"

    # Error handling
    ERROR = "ERROR"
    ACKNOWLEDGMENT = "ACK"


class MessagePriority(Enum):
    """Message priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class EvolutionMessage:
    """Structured message for evolution coordination."""

    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: str | None = None  # None = broadcast
    timestamp: float = None
    priority: MessagePriority = MessagePriority.NORMAL

    # Evolution-specific data
    evolution_id: str | None = None
    evolution_type: str | None = None  # nightly, breakthrough, emergency
    agent_id: str | None = None

    # Message payload
    data: dict[str, Any] = None

    # Metadata
    retry_count: int = 0
    max_retries: int = 3
    expiry_time: float | None = None
    requires_ack: bool = False
    correlation_id: str | None = None  # For request-response pairs

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.data is None:
            self.data = {}
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "message_id": self.message_id,
            "type": self.message_type.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "timestamp": self.timestamp,
            "priority": self.priority.value,
            "evolution_id": self.evolution_id,
            "evolution_type": self.evolution_type,
            "agent_id": self.agent_id,
            "data": self.data,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "expiry_time": self.expiry_time,
            "requires_ack": self.requires_ack,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvolutionMessage":
        """Create message from dictionary."""
        return cls(
            message_id=data.get("message_id"),
            message_type=MessageType(data.get("type")),
            sender_id=data.get("sender_id"),
            recipient_id=data.get("recipient_id"),
            timestamp=data.get("timestamp"),
            priority=MessagePriority(
                data.get("priority", MessagePriority.NORMAL.value)
            ),
            evolution_id=data.get("evolution_id"),
            evolution_type=data.get("evolution_type"),
            agent_id=data.get("agent_id"),
            data=data.get("data", {}),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            expiry_time=data.get("expiry_time"),
            requires_ack=data.get("requires_ack", False),
            correlation_id=data.get("correlation_id"),
        )

    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expiry_time is None:
            return False
        return time.time() > self.expiry_time

    def should_retry(self) -> bool:
        """Check if message should be retried."""
        return self.retry_count < self.max_retries and not self.is_expired()


class MessageProtocol:
    """Evolution-aware message protocol handler."""

    def __init__(self, p2p_node) -> None:
        self.p2p_node = p2p_node

        # Message handling
        self.message_handlers: dict[MessageType, callable] = {}
        self.pending_responses: dict[str, asyncio.Future] = {}
        self.message_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()

        # Reliability features
        self.sent_messages: dict[str, EvolutionMessage] = {}
        self.received_messages: set[str] = set()  # For deduplication
        self.retry_queue: asyncio.Queue = asyncio.Queue()

        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_dropped": 0,
            "messages_retried": 0,
            "average_latency": 0.0,
            "by_type": {},
        }

        # Background tasks
        self.message_processor_task: asyncio.Task | None = None
        self.retry_processor_task: asyncio.Task | None = None

        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        self.message_handlers.update(
            {
                MessageType.PING: self.handle_ping,
                MessageType.PONG: self.handle_pong,
                MessageType.HEARTBEAT: self.handle_heartbeat,
                MessageType.PEER_DISCOVERY: self.handle_peer_discovery,
                MessageType.PEER_INTRODUCTION: self.handle_peer_introduction,
                MessageType.PEER_ANNOUNCEMENT: self.handle_peer_announcement,
                MessageType.PEER_GOODBYE: self.handle_peer_goodbye,
                MessageType.ACKNOWLEDGMENT: self.handle_acknowledgment,
                MessageType.ERROR: self.handle_error,
                MessageType.RESOURCE_STATUS: self.handle_resource_status,
            }
        )

    async def start_protocol(self) -> None:
        """Start message protocol processors."""
        self.message_processor_task = asyncio.create_task(self._message_processor())
        self.retry_processor_task = asyncio.create_task(self._retry_processor())
        logger.info("Message protocol started")

    async def stop_protocol(self) -> None:
        """Stop message protocol processors."""
        if self.message_processor_task:
            self.message_processor_task.cancel()
        if self.retry_processor_task:
            self.retry_processor_task.cancel()
        logger.info("Message protocol stopped")

    async def send_message(
        self, message: EvolutionMessage | dict, writer: asyncio.StreamWriter
    ) -> bool | None:
        """Send message with protocol handling."""
        if isinstance(message, dict):
            # Convert dict to EvolutionMessage
            message = EvolutionMessage.from_dict(message)

        try:
            # Serialize message
            message_data = json.dumps(message.to_dict()).encode("utf-8")

            # Create protocol frame: [length:4][data:length]
            length = len(message_data)
            frame = struct.pack(">I", length) + message_data

            # Send frame
            writer.write(frame)
            await writer.drain()

            # Track sent message
            self.sent_messages[message.message_id] = message
            self.stats["messages_sent"] += 1
            self._update_type_stats(message.message_type, "sent")

            # Set up retry if needed
            if message.requires_ack and message.should_retry():
                await self.retry_queue.put(message)

            return True

        except Exception as e:
            logger.exception(f"Failed to send message {message.message_id}: {e}")
            self.stats["messages_dropped"] += 1
            return False

    async def read_message(self, reader: asyncio.StreamReader) -> bytes | None:
        """Read message using protocol framing."""
        try:
            # Read length header
            length_data = await reader.readexactly(4)
            if not length_data:
                return None

            length = struct.unpack(">I", length_data)[0]

            # Validate length
            if length > 1024 * 1024:  # 1MB max message size
                logger.warning(f"Message too large: {length} bytes")
                return None

            # Read message data
            message_data = await reader.readexactly(length)
            return message_data

        except asyncio.IncompleteReadError:
            return None
        except Exception as e:
            logger.exception(f"Failed to read message: {e}")
            return None

    async def handle_message(self, message_data: bytes, writer: asyncio.StreamWriter) -> None:
        """Handle incoming message."""
        try:
            # Parse message
            message_dict = json.loads(message_data.decode("utf-8"))
            message = EvolutionMessage.from_dict(message_dict)

            # Check for duplicate
            if message.message_id in self.received_messages:
                logger.debug(f"Duplicate message {message.message_id}")
                return

            # Check expiry
            if message.is_expired():
                logger.debug(f"Expired message {message.message_id}")
                return

            # Record message
            self.received_messages.add(message.message_id)
            self.stats["messages_received"] += 1
            self._update_type_stats(message.message_type, "received")

            # Send acknowledgment if required
            if message.requires_ack:
                await self._send_acknowledgment(message, writer)

            # Queue for processing
            priority = -message.priority.value  # Higher priority = lower number
            await self.message_queue.put((priority, time.time(), message, writer))

        except Exception as e:
            logger.exception(f"Failed to handle message: {e}")

    async def _message_processor(self) -> None:
        """Background message processor."""
        while True:
            try:
                priority, timestamp, message, writer = await self.message_queue.get()

                # Handle message
                handler = self.message_handlers.get(message.message_type)
                if handler:
                    await handler(message, writer)
                else:
                    logger.warning(
                        f"No handler for message type: {message.message_type}"
                    )

                # Calculate latency
                latency = time.time() - message.timestamp
                self._update_latency_stats(latency)

            except Exception as e:
                logger.exception(f"Message processor error: {e}")

    async def _retry_processor(self) -> None:
        """Background retry processor."""
        while True:
            try:
                message = await self.retry_queue.get()

                # Wait before retry
                await asyncio.sleep(2**message.retry_count)  # Exponential backoff

                # Check if still needs retry
                if message.should_retry() and message.message_id in self.sent_messages:
                    message.retry_count += 1
                    self.stats["messages_retried"] += 1

                    # Re-queue for retry
                    await self.retry_queue.put(message)

            except Exception as e:
                logger.exception(f"Retry processor error: {e}")

    async def _send_acknowledgment(
        self, original_message: EvolutionMessage, writer: asyncio.StreamWriter
    ) -> None:
        """Send acknowledgment for received message."""
        ack_message = EvolutionMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.ACKNOWLEDGMENT,
            sender_id=self.p2p_node.node_id,
            recipient_id=original_message.sender_id,
            correlation_id=original_message.message_id,
            data={"acknowledged": True},
        )

        await self.send_message(ack_message, writer)

    def _update_type_stats(self, message_type: MessageType, direction: str) -> None:
        """Update statistics by message type."""
        type_key = message_type.value
        if type_key not in self.stats["by_type"]:
            self.stats["by_type"][type_key] = {"sent": 0, "received": 0}
        self.stats["by_type"][type_key][direction] += 1

    def _update_latency_stats(self, latency: float) -> None:
        """Update latency statistics."""
        # Simple moving average
        alpha = 0.1
        self.stats["average_latency"] = (
            alpha * latency + (1 - alpha) * self.stats["average_latency"]
        )

    # Default message handlers
    async def handle_ping(
        self, message: EvolutionMessage, writer: asyncio.StreamWriter
    ) -> None:
        """Handle PING message."""
        pong_message = EvolutionMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.PONG,
            sender_id=self.p2p_node.node_id,
            recipient_id=message.sender_id,
            correlation_id=message.message_id,
            data={"pong_time": time.time()},
        )

        await self.send_message(pong_message, writer)

    async def handle_pong(
        self, message: EvolutionMessage, writer: asyncio.StreamWriter
    ) -> None:
        """Handle PONG message."""
        correlation_id = message.correlation_id
        if correlation_id in self.pending_responses:
            future = self.pending_responses[correlation_id]
            if not future.done():
                future.set_result(message)
            del self.pending_responses[correlation_id]

    async def handle_heartbeat(
        self, message: EvolutionMessage, writer: asyncio.StreamWriter
    ) -> None:
        """Handle HEARTBEAT message."""
        sender_id = message.sender_id
        capabilities_data = message.data.get("capabilities", {})

        # Update peer capabilities
        if sender_id in self.p2p_node.peer_registry:
            capabilities = self.p2p_node.peer_registry[sender_id]

            # Update with heartbeat data
            capabilities.last_seen = time.time()
            capabilities.current_evolution_load = capabilities_data.get(
                "current_evolution_load", 0.0
            )
            capabilities.available_for_evolution = capabilities_data.get(
                "available_for_evolution", True
            )
            capabilities.thermal_state = capabilities_data.get(
                "thermal_state", "normal"
            )

    async def handle_peer_discovery(
        self, message: EvolutionMessage, writer: asyncio.StreamWriter
    ) -> None:
        """Handle PEER_DISCOVERY message."""
        # Respond with our information
        response = EvolutionMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.PEER_ANNOUNCEMENT,
            sender_id=self.p2p_node.node_id,
            recipient_id=message.sender_id,
            data={
                "peer_info": {
                    "node_id": self.p2p_node.node_id,
                    "listen_port": self.p2p_node.listen_port,
                },
                "capabilities": (
                    self.p2p_node.local_capabilities.__dict__
                    if self.p2p_node.local_capabilities
                    else {}
                ),
            },
        )

        await self.send_message(response, writer)

    async def handle_peer_introduction(
        self, message: EvolutionMessage, writer: asyncio.StreamWriter
    ) -> None:
        """Handle PEER_INTRODUCTION message."""
        sender_id = message.sender_id
        capabilities_data = message.data.get("capabilities", {})

        logger.info(f"Peer introduction from {sender_id}")

        # Create peer capabilities
        from .p2p_node import PeerCapabilities

        capabilities = PeerCapabilities(device_id=sender_id, **capabilities_data)

        # Register peer
        self.p2p_node.peer_registry[sender_id] = capabilities

    async def handle_peer_announcement(
        self, message: EvolutionMessage, writer: asyncio.StreamWriter
    ) -> None:
        """Handle PEER_ANNOUNCEMENT message."""
        # Process peer announcement (discovery response)
        await self.handle_peer_introduction(message, writer)

    async def handle_peer_goodbye(
        self, message: EvolutionMessage, writer: asyncio.StreamWriter
    ) -> None:
        """Handle PEER_GOODBYE message."""
        sender_id = message.sender_id

        # Remove peer from registry
        if sender_id in self.p2p_node.peer_registry:
            del self.p2p_node.peer_registry[sender_id]

        # Close connection
        if sender_id in self.p2p_node.connections:
            del self.p2p_node.connections[sender_id]

        logger.info(f"Peer {sender_id} said goodbye")

    async def handle_acknowledgment(
        self, message: EvolutionMessage, writer: asyncio.StreamWriter
    ) -> None:
        """Handle ACKNOWLEDGMENT message."""
        correlation_id = message.correlation_id

        # Remove from retry queue
        if correlation_id in self.sent_messages:
            del self.sent_messages[correlation_id]

        logger.debug(f"Received acknowledgment for {correlation_id}")

    async def handle_error(
        self, message: EvolutionMessage, writer: asyncio.StreamWriter
    ) -> None:
        """Handle ERROR message."""
        error_info = message.data.get("error", "Unknown error")
        correlation_id = message.correlation_id

        logger.error(f"Received error from {message.sender_id}: {error_info}")

        # Notify waiting coroutines
        if correlation_id in self.pending_responses:
            future = self.pending_responses[correlation_id]
            if not future.done():
                future.set_exception(Exception(error_info))
            del self.pending_responses[correlation_id]

    async def handle_resource_status(
        self, message: EvolutionMessage, writer: asyncio.StreamWriter
    ) -> None:
        """Handle RESOURCE_STATUS message."""
        sender_id = message.sender_id
        resource_data = message.data.get("resources", {})

        # Update peer capabilities with resource information
        if sender_id in self.p2p_node.peer_registry:
            capabilities = self.p2p_node.peer_registry[sender_id]
            capabilities.ram_mb = resource_data.get(
                "ram_available_mb", capabilities.ram_mb
            )
            capabilities.battery_percent = resource_data.get(
                "battery_percent", capabilities.battery_percent
            )
            capabilities.current_evolution_load = resource_data.get(
                "evolution_load", 0.0
            )
            capabilities.thermal_state = resource_data.get("thermal_state", "normal")

    def register_handler(self, message_type: MessageType, handler: callable) -> None:
        """Register custom message handler."""
        self.message_handlers[message_type] = handler

    async def send_evolution_message(
        self,
        message_type: MessageType,
        recipient_id: str | None,
        data: dict[str, Any],
        evolution_id: str | None = None,
        agent_id: str | None = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        requires_ack: bool = False,
    ) -> str:
        """Send evolution-specific message."""
        message = EvolutionMessage(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            sender_id=self.p2p_node.node_id,
            recipient_id=recipient_id,
            evolution_id=evolution_id,
            agent_id=agent_id,
            priority=priority,
            data=data,
            requires_ack=requires_ack,
        )

        if recipient_id:
            # Send to specific peer
            if recipient_id in self.p2p_node.connections:
                await self.send_message(
                    message, self.p2p_node.connections[recipient_id]
                )
        else:
            # Broadcast to all peers
            for writer in self.p2p_node.connections.values():
                await self.send_message(message, writer)

        return message.message_id

    async def wait_for_response(
        self, correlation_id: str, timeout: float = 30.0
    ) -> EvolutionMessage | None:
        """Wait for response to a message."""
        future = asyncio.Future()
        self.pending_responses[correlation_id] = future

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            if correlation_id in self.pending_responses:
                del self.pending_responses[correlation_id]
            return None

    def get_protocol_stats(self) -> dict[str, Any]:
        """Get protocol statistics."""
        return {
            **self.stats,
            "pending_responses": len(self.pending_responses),
            "sent_messages_tracked": len(self.sent_messages),
            "received_messages_tracked": len(self.received_messages),
            "message_queue_size": self.message_queue.qsize(),
            "retry_queue_size": self.retry_queue.qsize(),
        }
