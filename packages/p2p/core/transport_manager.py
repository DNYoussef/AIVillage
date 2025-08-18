"""
Unified Transport Manager for P2P Communication

Coordinates between multiple transport types (BitChat, BetaNet, QUIC) and provides
intelligent routing, failover, and optimization based on device capabilities,
network conditions, and user preferences.

Based on production-ready implementations from the betanet bounty and existing
transport systems, unified into a single coherent interface.
"""

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .message_types import MessagePriority, UnifiedMessage

logger = logging.getLogger(__name__)


class TransportType(Enum):
    """Available transport types in order of typical preference."""

    BITCHAT = "bitchat"  # Bluetooth Low Energy mesh
    BETANET = "betanet"  # Encrypted internet (HTX)
    QUIC = "quic"  # Direct QUIC connections
    FALLBACK = "fallback"  # Emergency fallback transport


class TransportPriority(Enum):
    """Transport selection strategies."""

    OFFLINE_FIRST = "offline_first"  # Prefer BitChat, then BetaNet
    PRIVACY_FIRST = "privacy_first"  # Prefer BetaNet, then BitChat
    PERFORMANCE_FIRST = "performance_first"  # Prefer QUIC, then BetaNet
    COST_AWARE = "cost_aware"  # Minimize cellular usage
    BATTERY_AWARE = "battery_aware"  # Minimize battery usage
    ADAPTIVE = "adaptive"  # Adapt based on conditions


@dataclass
class TransportCapabilities:
    """Capabilities and current status of a transport."""

    # Basic capabilities
    supports_broadcast: bool = False
    supports_multicast: bool = False
    supports_unicast: bool = True
    max_message_size: int = 65536

    # Network characteristics
    is_offline_capable: bool = False
    requires_internet: bool = True
    typical_latency_ms: int = 1000
    bandwidth_mbps: float = 1.0

    # Security features
    provides_encryption: bool = True
    supports_forward_secrecy: bool = False
    has_built_in_auth: bool = False

    # Mobile/resource considerations
    battery_impact: str = "medium"  # low, medium, high
    data_cost_impact: str = "medium"  # low, medium, high
    works_on_cellular: bool = True
    works_on_wifi: bool = True

    # Current status
    is_available: bool = False
    is_connected: bool = False
    peer_count: int = 0
    last_activity: float | None = None
    error_rate: float = 0.0


@dataclass
class DeviceContext:
    """Current device context for transport selection."""

    # Power management
    battery_level: float | None = None  # 0.0 to 1.0
    is_charging: bool = False
    power_save_mode: bool = False

    # Network context
    network_type: str = "unknown"  # wifi, cellular, ethernet
    has_internet: bool = True
    is_metered_connection: bool = False
    signal_strength: float | None = None

    # Usage context
    is_foreground: bool = True
    user_priority: MessagePriority = MessagePriority.NORMAL
    cost_budget_remaining: float | None = None

    # Device capabilities
    supports_bluetooth: bool = True
    supports_wifi_direct: bool = False
    max_concurrent_connections: int = 20


@dataclass
class RoutingDecision:
    """Result of transport selection algorithm."""

    primary_transport: TransportType
    fallback_transports: list[TransportType] = field(default_factory=list)
    chunk_size: int = 16384
    retry_strategy: str = "exponential_backoff"
    timeout_seconds: int = 30
    reasoning: str = ""


class TransportManager:
    """
    Unified P2P transport manager.

    Coordinates multiple transport types and provides intelligent routing,
    failover, and optimization based on device context and message requirements.
    """

    def __init__(
        self, device_id: str, transport_priority: TransportPriority = TransportPriority.OFFLINE_FIRST, **kwargs
    ):
        self.device_id = device_id
        self.transport_priority = transport_priority

        # Transport instances and capabilities
        self.transports: dict[TransportType, Any] = {}
        self.capabilities: dict[TransportType, TransportCapabilities] = {}

        # Device context
        self.device_context = DeviceContext()

        # Message handling
        self.message_handlers: list[Callable] = []
        self.pending_chunks: dict[str, dict[int, UnifiedMessage]] = defaultdict(dict)

        # Statistics and monitoring
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "transport_failures": defaultdict(int),
            "routing_decisions": defaultdict(int),
            "chunk_reassemblies": 0,
            "last_activity": time.time(),
        }

        # Configuration
        self.config = {
            "max_chunk_size": kwargs.get("max_chunk_size", 16384),
            "chunk_timeout_seconds": kwargs.get("chunk_timeout_seconds", 30),
            "max_retry_attempts": kwargs.get("max_retry_attempts", 3),
            "heartbeat_interval": kwargs.get("heartbeat_interval", 60),
            "peer_discovery_interval": kwargs.get("peer_discovery_interval", 300),
        }

        logger.info(f"Transport manager initialized for device {device_id}")

    def register_transport(
        self, transport_type: TransportType, transport_instance: Any, capabilities: TransportCapabilities
    ):
        """Register a transport implementation with its capabilities."""
        self.transports[transport_type] = transport_instance
        self.capabilities[transport_type] = capabilities

        logger.info(f"Registered {transport_type.value} transport")

    def update_device_context(self, **context_updates):
        """Update current device context for better routing decisions."""
        for key, value in context_updates.items():
            if hasattr(self.device_context, key):
                setattr(self.device_context, key, value)

        logger.debug(f"Updated device context: {context_updates}")

    def _calculate_transport_score(self, transport_type: TransportType, message: UnifiedMessage) -> float:
        """
        Calculate suitability score for a transport based on message and context.

        Higher scores indicate better suitability. Returns 0.0 if transport
        is unavailable or unsuitable.
        """
        capabilities = self.capabilities.get(transport_type)
        if not capabilities or not capabilities.is_available:
            return 0.0

        score = 1.0

        # Message size compatibility
        if message.size_bytes > capabilities.max_message_size:
            # Can we chunk it?
            chunks_needed = (message.size_bytes + self.config["max_chunk_size"] - 1) // self.config["max_chunk_size"]
            if chunks_needed > 10:  # Arbitrary limit for chunking
                score *= 0.1  # Heavily penalize
            else:
                score *= 0.8  # Moderate penalty for chunking

        # Priority-based scoring
        priority_multipliers = {
            MessagePriority.CRITICAL: 1.2,
            MessagePriority.HIGH: 1.1,
            MessagePriority.NORMAL: 1.0,
            MessagePriority.LOW: 0.9,
            MessagePriority.BULK: 0.8,
        }
        score *= priority_multipliers.get(message.metadata.priority, 1.0)

        # Transport priority strategy
        if self.transport_priority == TransportPriority.OFFLINE_FIRST:
            priority_scores = {
                TransportType.BITCHAT: 1.0,
                TransportType.BETANET: 0.8,
                TransportType.QUIC: 0.6,
                TransportType.FALLBACK: 0.2,
            }
        elif self.transport_priority == TransportPriority.PRIVACY_FIRST:
            priority_scores = {
                TransportType.BETANET: 1.0,
                TransportType.BITCHAT: 0.9,
                TransportType.QUIC: 0.5,
                TransportType.FALLBACK: 0.2,
            }
        elif self.transport_priority == TransportPriority.PERFORMANCE_FIRST:
            priority_scores = {
                TransportType.QUIC: 1.0,
                TransportType.BETANET: 0.8,
                TransportType.BITCHAT: 0.4,
                TransportType.FALLBACK: 0.2,
            }
        else:  # Default to balanced
            priority_scores = {
                TransportType.BITCHAT: 0.9,
                TransportType.BETANET: 0.9,
                TransportType.QUIC: 0.9,
                TransportType.FALLBACK: 0.3,
            }

        score *= priority_scores.get(transport_type, 0.5)

        # Device context adjustments
        if self.device_context.battery_level is not None:
            if self.device_context.battery_level < 0.2:  # Low battery
                battery_penalties = {
                    "low": 1.0,
                    "medium": 0.7,
                    "high": 0.3,
                }
                score *= battery_penalties.get(capabilities.battery_impact, 0.7)

        if self.device_context.is_metered_connection:
            cost_penalties = {
                "low": 1.0,
                "medium": 0.8,
                "high": 0.4,
            }
            score *= cost_penalties.get(capabilities.data_cost_impact, 0.8)

        # Network availability
        if not self.device_context.has_internet and capabilities.requires_internet:
            return 0.0

        # Error rate penalty
        score *= 1.0 - min(capabilities.error_rate, 0.9)

        return score

    def select_transport(self, message: UnifiedMessage) -> RoutingDecision:
        """Select the best transport for a message based on current conditions."""

        # Calculate scores for all transports
        transport_scores = {}
        for transport_type in TransportType:
            score = self._calculate_transport_score(transport_type, message)
            if score > 0:
                transport_scores[transport_type] = score

        if not transport_scores:
            logger.error("No suitable transports available")
            return RoutingDecision(primary_transport=TransportType.FALLBACK, reasoning="No transports available")

        # Sort by score (highest first)
        sorted_transports = sorted(transport_scores.items(), key=lambda x: x[1], reverse=True)

        primary_transport = sorted_transports[0][0]
        fallback_transports = [t[0] for t in sorted_transports[1:3]]  # Up to 2 fallbacks

        # Determine chunk size
        primary_capabilities = self.capabilities[primary_transport]
        chunk_size = min(self.config["max_chunk_size"], primary_capabilities.max_message_size)

        # Build reasoning
        reasoning = f"Selected {primary_transport.value} (score: {sorted_transports[0][1]:.2f})"
        if fallback_transports:
            reasoning += f", fallbacks: {[t.value for t in fallback_transports]}"

        decision = RoutingDecision(
            primary_transport=primary_transport,
            fallback_transports=fallback_transports,
            chunk_size=chunk_size,
            reasoning=reasoning,
        )

        self.stats["routing_decisions"][primary_transport.value] += 1
        logger.debug(f"Routing decision: {reasoning}")

        return decision

    async def send_message(self, message: UnifiedMessage) -> bool:
        """Send a message using the best available transport."""

        # Select transport
        routing_decision = self.select_transport(message)

        # Try primary transport first
        success = await self._send_via_transport(
            routing_decision.primary_transport, message, routing_decision.chunk_size
        )

        if success:
            self.stats["messages_sent"] += 1
            self.stats["bytes_sent"] += message.size_bytes
            self.stats["last_activity"] = time.time()
            return True

        # Try fallback transports
        for fallback_transport in routing_decision.fallback_transports:
            logger.warning(f"Trying fallback transport: {fallback_transport.value}")
            success = await self._send_via_transport(fallback_transport, message)
            if success:
                self.stats["messages_sent"] += 1
                self.stats["bytes_sent"] += message.size_bytes
                self.stats["last_activity"] = time.time()
                return True

        logger.error("Failed to send message via all available transports")
        return False

    async def _send_via_transport(
        self, transport_type: TransportType, message: UnifiedMessage, chunk_size: int | None = None
    ) -> bool:
        """Send message via specific transport, with chunking if needed."""

        transport = self.transports.get(transport_type)
        if not transport:
            logger.error(f"Transport {transport_type.value} not available")
            self.stats["transport_failures"][transport_type.value] += 1
            return False

        try:
            # Check if message needs chunking
            if chunk_size and message.size_bytes > chunk_size:
                return await self._send_chunked_message(transport, message, chunk_size)
            else:
                # Send as single message
                if hasattr(transport, "send_message"):
                    return await transport.send_message(message)
                elif hasattr(transport, "send"):
                    return await transport.send(message.payload)
                else:
                    logger.error(f"Transport {transport_type.value} has no send method")
                    return False

        except Exception as e:
            logger.error(f"Error sending via {transport_type.value}: {e}")
            self.stats["transport_failures"][transport_type.value] += 1
            return False

    async def _send_chunked_message(self, transport: Any, message: UnifiedMessage, chunk_size: int) -> bool:
        """Send a large message in chunks."""

        payload = message.payload
        total_chunks = (len(payload) + chunk_size - 1) // chunk_size

        # Update message for chunking
        message.total_chunks = total_chunks

        # Send each chunk
        for chunk_index in range(total_chunks):
            start_offset = chunk_index * chunk_size
            end_offset = min(start_offset + chunk_size, len(payload))
            chunk_data = payload[start_offset:end_offset]

            chunk_message = message.create_chunk(chunk_data, chunk_index)

            try:
                if hasattr(transport, "send_message"):
                    success = await transport.send_message(chunk_message)
                elif hasattr(transport, "send"):
                    success = await transport.send(chunk_message.payload)
                else:
                    return False

                if not success:
                    logger.error(f"Failed to send chunk {chunk_index}/{total_chunks}")
                    return False

            except Exception as e:
                logger.error(f"Error sending chunk {chunk_index}: {e}")
                return False

        logger.debug(f"Successfully sent chunked message ({total_chunks} chunks)")
        return True

    async def _handle_incoming_message(self, message: UnifiedMessage, transport_type: TransportType):
        """Handle incoming message, including chunk reassembly."""

        self.stats["messages_received"] += 1
        self.stats["bytes_received"] += message.size_bytes
        self.stats["last_activity"] = time.time()

        # Handle chunked messages
        if message.is_chunked:
            correlation_id = message.metadata.correlation_id
            if not correlation_id:
                logger.warning("Chunked message missing correlation_id")
                return

            # Store chunk
            self.pending_chunks[correlation_id][message.chunk_index] = message

            # Check if we have all chunks
            chunks = self.pending_chunks[correlation_id]
            if len(chunks) == message.total_chunks:
                # Reassemble message
                try:
                    reassembled_message = self._reassemble_chunks(chunks)
                    del self.pending_chunks[correlation_id]
                    self.stats["chunk_reassemblies"] += 1

                    # Process reassembled message
                    await self._process_complete_message(reassembled_message, transport_type)

                except Exception as e:
                    logger.error(f"Error reassembling chunks: {e}")
                    del self.pending_chunks[correlation_id]
        else:
            # Process single message
            await self._process_complete_message(message, transport_type)

    def _reassemble_chunks(self, chunks: dict[int, UnifiedMessage]) -> UnifiedMessage:
        """Reassemble chunks into original message."""

        # Sort chunks by index
        sorted_chunks = [chunks[i] for i in sorted(chunks.keys())]

        # Combine payloads
        combined_payload = b"".join(chunk.payload for chunk in sorted_chunks)

        # Use metadata from first chunk
        first_chunk = sorted_chunks[0]
        reassembled_message = UnifiedMessage(
            message_type=first_chunk.message_type,
            payload=combined_payload,
            metadata=first_chunk.metadata,
            chunk_index=0,
            total_chunks=1,
            max_chunk_size=first_chunk.max_chunk_size,
        )

        return reassembled_message

    async def _process_complete_message(self, message: UnifiedMessage, transport_type: TransportType):
        """Process a complete (non-chunked or reassembled) message."""

        logger.debug(f"Processing {message.message_type.value} message via {transport_type.value}")

        # Notify registered handlers
        for handler in self.message_handlers:
            try:
                await handler(message, transport_type)
            except Exception as e:
                logger.warning(f"Message handler error: {e}")

    def register_message_handler(self, handler: Callable):
        """Register a handler for incoming messages."""
        self.message_handlers.append(handler)
        logger.debug("Message handler registered")

    async def start(self) -> bool:
        """Start all registered transports."""
        logger.info("Starting transport manager...")

        start_tasks = []
        for transport_type, transport in self.transports.items():
            if hasattr(transport, "start"):
                start_tasks.append(self._start_transport(transport_type, transport))

        if start_tasks:
            results = await asyncio.gather(*start_tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            logger.info(f"Started {success_count}/{len(start_tasks)} transports")
            return success_count > 0

        logger.warning("No transports to start")
        return False

    async def _start_transport(self, transport_type: TransportType, transport: Any) -> bool:
        """Start individual transport and update capabilities."""
        try:
            result = await transport.start()
            if result and transport_type in self.capabilities:
                self.capabilities[transport_type].is_available = True
                self.capabilities[transport_type].is_connected = True
                logger.info(f"{transport_type.value} transport started successfully")
            return result
        except Exception as e:
            logger.error(f"Error starting {transport_type.value} transport: {e}")
            return False

    async def stop(self) -> bool:
        """Stop all transports."""
        logger.info("Stopping transport manager...")

        stop_tasks = []
        for transport_type, transport in self.transports.items():
            if hasattr(transport, "stop"):
                stop_tasks.append(self._stop_transport(transport_type, transport))

        if stop_tasks:
            results = await asyncio.gather(*stop_tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            logger.info(f"Stopped {success_count}/{len(stop_tasks)} transports")

        # Clear capabilities
        for capabilities in self.capabilities.values():
            capabilities.is_available = False
            capabilities.is_connected = False

        return True

    async def _stop_transport(self, transport_type: TransportType, transport: Any) -> bool:
        """Stop individual transport."""
        try:
            result = await transport.stop()
            if transport_type in self.capabilities:
                self.capabilities[transport_type].is_available = False
                self.capabilities[transport_type].is_connected = False
            logger.info(f"{transport_type.value} transport stopped")
            return result
        except Exception as e:
            logger.error(f"Error stopping {transport_type.value} transport: {e}")
            return False

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive transport manager status."""
        available_transports = [t.value for t, cap in self.capabilities.items() if cap.is_available]

        return {
            "device_id": self.device_id,
            "transport_priority": self.transport_priority.value,
            "available_transports": available_transports,
            "active_transports": available_transports,  # Alias for compatibility
            "device_context": {
                "battery_level": self.device_context.battery_level,
                "network_type": self.device_context.network_type,
                "has_internet": self.device_context.has_internet,
                "is_metered": self.device_context.is_metered_connection,
            },
            "statistics": self.stats.copy(),
            "pending_chunks": len(self.pending_chunks),
        }
