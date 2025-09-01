"""DTN Handler Service - Delay-Tolerant Networking and store-and-forward

This service manages message storage, opportunistic forwarding, and buffer
management for delay-tolerant networking scenarios in the Navigator system.
"""

import asyncio
import logging
import pickle
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..interfaces.routing_interfaces import IDTNHandlerService, RoutingEvent
from ..events.event_bus import get_event_bus
from ..path_policy import MessageContext

logger = logging.getLogger(__name__)


class MessageState(Enum):
    """States for stored messages in DTN"""

    STORED = "stored"
    QUEUED = "queued"
    FORWARDING = "forwarding"
    DELIVERED = "delivered"
    EXPIRED = "expired"
    FAILED = "failed"


class ForwardingStrategy(Enum):
    """Strategies for message forwarding in DTN"""

    EPIDEMIC = "epidemic"  # Flood to all available nodes
    SPRAY_AND_WAIT = "spray_and_wait"  # Limited flooding then wait
    PROPHET = "prophet"  # Probabilistic routing
    DIRECT = "direct"  # Only forward to destination
    OPPORTUNISTIC = "opportunistic"  # Forward when good opportunity


@dataclass
class StoredMessage:
    """Represents a stored message in the DTN system"""

    message_id: str
    destination: str
    content: bytes
    context: MessageContext
    state: MessageState = MessageState.STORED
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    attempts: int = 0
    max_attempts: int = 5
    priority_boost: int = 0
    routing_metadata: Dict[str, Any] = field(default_factory=dict)
    forwarding_history: List[str] = field(default_factory=list)
    size_bytes: int = 0

    def __post_init__(self):
        self.size_bytes = len(self.content)
        if self.expires_at == 0.0:
            # Default expiration based on priority (1-24 hours)
            hours = min(24, max(1, self.context.priority * 3))
            self.expires_at = self.created_at + (hours * 3600)

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def can_retry(self) -> bool:
        return self.attempts < self.max_attempts and not self.is_expired()


class DTNHandlerService(IDTNHandlerService):
    """Delay-Tolerant Networking and store-and-forward service

    Manages:
    - Message storage for offline scenarios
    - Opportunistic forwarding when connectivity available
    - Buffer management and cleanup
    - Routing decision history for better forwarding
    - Multiple forwarding strategies
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.event_bus = get_event_bus()

        # Storage management
        self.storage_path = Path(storage_path or "/tmp/navigator_dtn")  # nosec B108 - temp directory
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Message storage
        self.stored_messages: Dict[str, StoredMessage] = {}
        self.forwarding_queue: deque[str] = deque()  # Message IDs to forward
        self.delivery_history: deque[Dict[str, Any]] = deque(maxlen=1000)

        # Buffer management
        self.max_storage_size_mb = 1000  # 1GB max storage
        self.max_message_age_hours = 72  # 3 days max
        self.current_storage_size = 0

        # Forwarding strategies and settings
        self.forwarding_strategy = ForwardingStrategy.OPPORTUNISTIC
        self.forwarding_enabled = True
        self.forwarding_interval = 30.0  # Check for forwarding opportunities every 30s
        self.last_forwarding_attempt = 0.0

        # Peer tracking for routing decisions
        self.peer_connectivity_history: Dict[str, List[float]] = defaultdict(list)
        self.peer_delivery_success: Dict[str, float] = defaultdict(lambda: 0.5)
        self.destination_reachability: Dict[str, float] = defaultdict(lambda: 0.3)

        # Performance tracking
        self.storage_metrics: Dict[str, Any] = {
            "messages_stored": 0,
            "messages_forwarded": 0,
            "messages_delivered": 0,
            "messages_expired": 0,
            "total_bytes_stored": 0,
            "avg_storage_time": 0.0,
        }

        # Background tasks
        self.forwarding_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False

        # Load persistent data
        asyncio.create_task(self._load_persistent_data())

        logger.info(f"DTNHandlerService initialized with storage at {self.storage_path}")

    async def start_service(self) -> None:
        """Start DTN service background tasks"""
        if self.running:
            return

        self.running = True

        # Start background tasks
        self.forwarding_task = asyncio.create_task(self._forwarding_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("DTN Handler service started")

    async def stop_service(self) -> None:
        """Stop DTN service and save state"""
        self.running = False

        # Cancel background tasks
        for task in [self.forwarding_task, self.cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Save persistent data
        await self._save_persistent_data()

        logger.info("DTN Handler service stopped")

    async def store_message(self, message_id: str, destination: str, content: bytes, context: MessageContext) -> bool:
        """Store message for later forwarding"""
        try:
            # Check storage capacity
            message_size = len(content)
            if not self._can_store_message(message_size):
                # Try to free up space
                freed_space = await self._free_storage_space(message_size)
                if not freed_space:
                    logger.warning(f"Cannot store message {message_id}: insufficient storage capacity")
                    return False

            # Create stored message
            stored_msg = StoredMessage(
                message_id=message_id,
                destination=destination,
                content=content,
                context=context,
                routing_metadata={"stored_by": "DTNHandlerService"},
            )

            # Store message
            self.stored_messages[message_id] = stored_msg
            self.current_storage_size += message_size

            # Add to forwarding queue if appropriate
            if self._should_queue_for_forwarding(stored_msg):
                self.forwarding_queue.append(message_id)

            # Persist to disk for durability
            await self._persist_message(stored_msg)

            # Update metrics
            self.storage_metrics["messages_stored"] += 1
            self.storage_metrics["total_bytes_stored"] += message_size

            # Emit storage event
            self._emit_dtn_event(
                "message_stored",
                {
                    "message_id": message_id,
                    "destination": destination,
                    "size_bytes": message_size,
                    "priority": context.priority,
                    "expires_at": stored_msg.expires_at,
                },
            )

            logger.info(f"Stored message {message_id} for {destination} ({message_size} bytes)")
            return True

        except Exception as e:
            logger.error(f"Failed to store message {message_id}: {e}")
            return False

    def _can_store_message(self, message_size: int) -> bool:
        """Check if message can be stored within capacity limits"""
        size_mb = message_size / (1024 * 1024)
        current_mb = self.current_storage_size / (1024 * 1024)

        return current_mb + size_mb <= self.max_storage_size_mb

    async def _free_storage_space(self, required_bytes: int) -> bool:
        """Free up storage space by removing old/low-priority messages"""
        logger.debug(f"Attempting to free {required_bytes} bytes of storage")

        freed_bytes = 0
        messages_to_remove = []

        # Sort messages by priority for removal (lowest priority first)
        sorted_messages = sorted(self.stored_messages.values(), key=lambda m: (m.context.priority, m.created_at))

        for msg in sorted_messages:
            if freed_bytes >= required_bytes:
                break

            # Remove expired messages first
            if msg.is_expired():
                messages_to_remove.append(msg.message_id)
                freed_bytes += msg.size_bytes
                continue

            # Remove low-priority old messages
            age_hours = (time.time() - msg.created_at) / 3600
            if msg.context.priority <= 3 and age_hours > 24:  # Low priority, older than 1 day
                messages_to_remove.append(msg.message_id)
                freed_bytes += msg.size_bytes

        # Remove selected messages
        for msg_id in messages_to_remove:
            await self._remove_stored_message(msg_id, "storage_cleanup")

        logger.info(f"Freed {freed_bytes} bytes by removing {len(messages_to_remove)} messages")
        return freed_bytes >= required_bytes

    def _should_queue_for_forwarding(self, message: StoredMessage) -> bool:
        """Determine if message should be queued for forwarding"""
        # Don't queue expired messages
        if message.is_expired():
            return False

        # Always queue high-priority messages
        if message.context.priority >= 8:
            return True

        # Check if destination has reasonable reachability
        reachability = self.destination_reachability.get(message.destination, 0.3)
        if reachability > 0.1:  # Some chance of delivery
            return True

        # Queue if we have recent connectivity to peers
        recent_peer_activity = any(
            time.time() - max(times) < 300  # Activity in last 5 minutes
            for times in self.peer_connectivity_history.values()
            if times
        )

        return recent_peer_activity

    async def forward_stored_messages(self) -> Dict[str, int]:
        """Forward stored messages when connectivity available"""
        if not self.forwarding_enabled or not self.forwarding_queue:
            return {"forwarded": 0, "failed": 0, "skipped": 0}

        forwarded = 0
        failed = 0
        skipped = 0

        # Process forwarding queue
        messages_to_process = list(self.forwarding_queue)
        self.forwarding_queue.clear()

        for message_id in messages_to_process:
            if message_id not in self.stored_messages:
                continue  # Message was removed

            message = self.stored_messages[message_id]

            # Check if message is still valid
            if message.is_expired():
                await self._remove_stored_message(message_id, "expired")
                skipped += 1
                continue

            # Attempt forwarding based on strategy
            success = await self._attempt_message_forwarding(message)

            if success:
                forwarded += 1
                # Mark as delivered if forwarding succeeded
                await self._mark_message_delivered(message)
            elif message.can_retry():
                # Re-queue for retry if attempts remain
                self.forwarding_queue.append(message_id)
                message.attempts += 1
                failed += 1
            else:
                # Max attempts reached or expired
                await self._remove_stored_message(message_id, "max_attempts_exceeded")
                failed += 1

        # Update metrics
        self.storage_metrics["messages_forwarded"] += forwarded

        # Emit forwarding results
        if forwarded > 0 or failed > 0:
            self._emit_dtn_event(
                "forwarding_completed",
                {
                    "forwarded": forwarded,
                    "failed": failed,
                    "skipped": skipped,
                    "queue_size_remaining": len(self.forwarding_queue),
                },
            )

            logger.info(f"Forwarding completed: {forwarded} sent, {failed} failed, {skipped} skipped")

        return {"forwarded": forwarded, "failed": failed, "skipped": skipped}

    async def _attempt_message_forwarding(self, message: StoredMessage) -> bool:
        """Attempt to forward a specific message"""
        try:
            # Simulate forwarding based on strategy and network conditions
            success_probability = self._calculate_forwarding_probability(message)

            # Simulate network transmission (would be actual forwarding in production)
            await asyncio.sleep(0.1)  # Simulate transmission delay

            # Probabilistic success based on conditions
            import random

            success = random.random() < success_probability

            if success:
                # Record successful forwarding
                message.forwarding_history.append(f"forwarded_at_{time.time()}")
                message.state = MessageState.DELIVERED

                # Update peer success rates
                self._update_peer_success_rates(message.destination, True)

                logger.debug(f"Successfully forwarded message {message.message_id} to {message.destination}")
                return True
            else:
                # Forwarding failed
                self._update_peer_success_rates(message.destination, False)
                message.state = MessageState.FAILED

                logger.debug(f"Failed to forward message {message.message_id} to {message.destination}")
                return False

        except Exception as e:
            logger.error(f"Error forwarding message {message.message_id}: {e}")
            return False

    def _calculate_forwarding_probability(self, message: StoredMessage) -> float:
        """Calculate probability of successful forwarding"""
        base_probability = 0.7  # Base success rate

        # Adjust for destination reachability
        reachability = self.destination_reachability.get(message.destination, 0.3)
        probability = base_probability * (0.3 + 0.7 * reachability)

        # Adjust for message priority
        priority_boost = min(0.2, message.context.priority * 0.02)
        probability += priority_boost

        # Adjust for message age (older messages harder to deliver)
        age_hours = (time.time() - message.created_at) / 3600
        age_penalty = min(0.3, age_hours * 0.01)
        probability -= age_penalty

        # Adjust for previous attempts
        attempt_penalty = message.attempts * 0.1
        probability -= attempt_penalty

        return max(0.1, min(0.95, probability))  # Clamp to reasonable range

    def _update_peer_success_rates(self, peer_id: str, success: bool) -> None:
        """Update peer delivery success rates"""
        current_rate = self.peer_delivery_success[peer_id]

        # Exponential moving average
        alpha = 0.1
        new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate
        self.peer_delivery_success[peer_id] = new_rate

        # Update destination reachability
        self.destination_reachability[peer_id] = new_rate

    async def _mark_message_delivered(self, message: StoredMessage) -> None:
        """Mark message as successfully delivered"""
        message.state = MessageState.DELIVERED

        # Record delivery
        delivery_record = {
            "message_id": message.message_id,
            "destination": message.destination,
            "delivered_at": time.time(),
            "storage_time": time.time() - message.created_at,
            "attempts": message.attempts,
            "priority": message.context.priority,
        }

        self.delivery_history.append(delivery_record)

        # Update metrics
        self.storage_metrics["messages_delivered"] += 1

        # Update average storage time
        storage_time = delivery_record["storage_time"]
        current_avg = self.storage_metrics["avg_storage_time"]
        delivered_count = self.storage_metrics["messages_delivered"]

        self.storage_metrics["avg_storage_time"] = (
            current_avg * (delivered_count - 1) + storage_time
        ) / delivered_count

        # Remove from storage
        await self._remove_stored_message(message.message_id, "delivered")

        logger.info(f"Message {message.message_id} delivered after {storage_time:.1f} seconds")

    async def _remove_stored_message(self, message_id: str, reason: str) -> None:
        """Remove message from storage"""
        if message_id not in self.stored_messages:
            return

        message = self.stored_messages[message_id]

        # Update storage size
        self.current_storage_size -= message.size_bytes

        # Remove from disk storage
        await self._remove_persisted_message(message_id)

        # Remove from memory
        del self.stored_messages[message_id]

        # Update metrics based on reason
        if reason == "expired":
            self.storage_metrics["messages_expired"] += 1

        logger.debug(f"Removed message {message_id} (reason: {reason})")

    def manage_storage_buffer(self) -> Dict[str, Any]:
        """Manage message storage buffer and cleanup"""
        current_time = time.time()

        # Count messages by state
        state_counts = defaultdict(int)
        for message in self.stored_messages.values():
            state_counts[message.state.value] += 1

        # Calculate storage statistics
        size_mb = self.current_storage_size / (1024 * 1024)
        utilization = size_mb / self.max_storage_size_mb

        # Age distribution
        age_buckets = {"<1h": 0, "1-24h": 0, "1-3d": 0, ">3d": 0}
        for message in self.stored_messages.values():
            age_hours = (current_time - message.created_at) / 3600
            if age_hours < 1:
                age_buckets["<1h"] += 1
            elif age_hours < 24:
                age_buckets["1-24h"] += 1
            elif age_hours < 72:
                age_buckets["1-3d"] += 1
            else:
                age_buckets[">3d"] += 1

        buffer_status = {
            "total_messages": len(self.stored_messages),
            "storage_size_mb": size_mb,
            "max_storage_mb": self.max_storage_size_mb,
            "utilization": utilization,
            "forwarding_queue_size": len(self.forwarding_queue),
            "messages_by_state": dict(state_counts),
            "age_distribution": age_buckets,
            "cleanup_needed": utilization > 0.8 or age_buckets[">3d"] > 0,
        }

        # Emit buffer status event
        self._emit_dtn_event("buffer_status", buffer_status)

        return buffer_status

    async def _forwarding_loop(self) -> None:
        """Background loop for opportunistic forwarding"""
        logger.info("DTN forwarding loop started")

        while self.running:
            try:
                current_time = time.time()

                # Check if it's time to attempt forwarding
                if current_time - self.last_forwarding_attempt >= self.forwarding_interval:
                    await self.forward_stored_messages()
                    self.last_forwarding_attempt = current_time

                # Sleep until next check
                await asyncio.sleep(5.0)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error in forwarding loop: {e}")
                await asyncio.sleep(10.0)  # Longer sleep on error

    async def _cleanup_loop(self) -> None:
        """Background loop for buffer cleanup"""
        logger.info("DTN cleanup loop started")

        while self.running:
            try:
                # Perform cleanup every 5 minutes
                await asyncio.sleep(300)

                # Remove expired messages
                expired_messages = [msg_id for msg_id, msg in self.stored_messages.items() if msg.is_expired()]

                for msg_id in expired_messages:
                    await self._remove_stored_message(msg_id, "expired")

                if expired_messages:
                    logger.info(f"Cleaned up {len(expired_messages)} expired messages")

                # Check if additional cleanup is needed
                buffer_status = self.manage_storage_buffer()
                if buffer_status["cleanup_needed"]:
                    # Free up 20% of storage
                    target_free_bytes = int(self.max_storage_size_mb * 0.2 * 1024 * 1024)
                    await self._free_storage_space(target_free_bytes)

            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _persist_message(self, message: StoredMessage) -> None:
        """Persist message to disk for durability"""
        try:
            message_file = self.storage_path / f"{message.message_id}.pkl"

            # Serialize message data
            message_data = {
                "message_id": message.message_id,
                "destination": message.destination,
                "content": message.content,
                "context": {
                    "size_bytes": message.context.size_bytes,
                    "priority": message.context.priority,
                    "content_type": message.context.content_type,
                    "requires_realtime": message.context.requires_realtime,
                    "privacy_required": message.context.privacy_required,
                    "delivery_deadline": message.context.delivery_deadline,
                    "bandwidth_sensitive": message.context.bandwidth_sensitive,
                },
                "created_at": message.created_at,
                "expires_at": message.expires_at,
                "routing_metadata": message.routing_metadata,
            }

            with open(message_file, "wb") as f:
                pickle.dump(message_data, f)

        except Exception as e:
            logger.warning(f"Failed to persist message {message.message_id}: {e}")

    async def _remove_persisted_message(self, message_id: str) -> None:
        """Remove persisted message from disk"""
        try:
            message_file = self.storage_path / f"{message_id}.pkl"
            if message_file.exists():
                message_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove persisted message {message_id}: {e}")

    async def _load_persistent_data(self) -> None:
        """Load persistent DTN data from disk"""
        try:
            # Load stored messages
            for message_file in self.storage_path.glob("*.pkl"):
                try:
                    with open(message_file, "rb") as f:
                        message_data = pickle.load(f)

                    # Recreate MessageContext
                    context = MessageContext(
                        size_bytes=message_data["context"]["size_bytes"],
                        priority=message_data["context"]["priority"],
                        content_type=message_data["context"]["content_type"],
                        requires_realtime=message_data["context"]["requires_realtime"],
                        privacy_required=message_data["context"]["privacy_required"],
                        delivery_deadline=message_data["context"]["delivery_deadline"],
                        bandwidth_sensitive=message_data["context"]["bandwidth_sensitive"],
                    )

                    # Recreate StoredMessage
                    stored_msg = StoredMessage(
                        message_id=message_data["message_id"],
                        destination=message_data["destination"],
                        content=message_data["content"],
                        context=context,
                        created_at=message_data["created_at"],
                        expires_at=message_data["expires_at"],
                        routing_metadata=message_data["routing_metadata"],
                    )

                    # Skip expired messages
                    if not stored_msg.is_expired():
                        self.stored_messages[stored_msg.message_id] = stored_msg
                        self.current_storage_size += stored_msg.size_bytes

                        # Add to forwarding queue if appropriate
                        if self._should_queue_for_forwarding(stored_msg):
                            self.forwarding_queue.append(stored_msg.message_id)
                    else:
                        # Remove expired persistent file
                        message_file.unlink()

                except Exception as e:
                    logger.warning(f"Failed to load message from {message_file}: {e}")
                    # Remove corrupted file
                    message_file.unlink()

            logger.info(f"Loaded {len(self.stored_messages)} persistent DTN messages")

        except Exception as e:
            logger.error(f"Failed to load persistent DTN data: {e}")

    async def _save_persistent_data(self) -> None:
        """Save DTN state data"""
        # Messages are already persisted individually
        # Could save additional state like peer success rates here
        logger.debug("DTN persistent data saved")

    def _emit_dtn_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit DTN event"""
        event = RoutingEvent(
            event_type=event_type, timestamp=time.time(), source_service="DTNHandlerService", data=data
        )
        self.event_bus.publish(event)

    def get_dtn_statistics(self) -> Dict[str, Any]:
        """Get DTN service statistics"""
        return {
            **self.storage_metrics,
            "current_storage_size_mb": self.current_storage_size / (1024 * 1024),
            "storage_utilization": (self.current_storage_size / (1024 * 1024)) / self.max_storage_size_mb,
            "active_messages": len(self.stored_messages),
            "forwarding_queue_size": len(self.forwarding_queue),
            "peer_success_rates": dict(self.peer_delivery_success),
            "destination_reachability": dict(self.destination_reachability),
            "recent_deliveries": len(
                [d for d in self.delivery_history if time.time() - d["delivered_at"] < 3600]
            ),  # Last hour
        }

    def update_peer_connectivity(self, peer_id: str, connected: bool) -> None:
        """Update peer connectivity information"""
        current_time = time.time()

        # Record connectivity event
        self.peer_connectivity_history[peer_id].append(current_time if connected else -current_time)

        # Keep only recent history (last 24 hours)
        cutoff_time = current_time - 86400
        self.peer_connectivity_history[peer_id] = [
            t for t in self.peer_connectivity_history[peer_id] if abs(t) > cutoff_time
        ]

        # Update destination reachability based on connectivity patterns
        if connected:
            # Boost reachability when peer connects
            current_reach = self.destination_reachability[peer_id]
            self.destination_reachability[peer_id] = min(0.9, current_reach + 0.1)

    def set_forwarding_strategy(self, strategy: ForwardingStrategy) -> None:
        """Set message forwarding strategy"""
        self.forwarding_strategy = strategy
        logger.info(f"DTN forwarding strategy changed to: {strategy.value}")

        # Emit strategy change event
        self._emit_dtn_event("forwarding_strategy_changed", {"new_strategy": strategy.value, "timestamp": time.time()})
