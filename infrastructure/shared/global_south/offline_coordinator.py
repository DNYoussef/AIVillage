"""
Global South Offline Coordinator for AIVillage.

Provides comprehensive offline-first capabilities optimized for Global South
scenarios including intermittent connectivity, data cost awareness, and
resource-constrained environments.
"""

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from pathlib import Path
import time
from typing import Any

logger = logging.getLogger(__name__)


class ConnectivityMode(Enum):
    """Network connectivity modes."""

    FULLY_OFFLINE = "fully_offline"
    INTERMITTENT = "intermittent"
    LIMITED_BANDWIDTH = "limited_bandwidth"
    FULL_CONNECTIVITY = "full_connectivity"


class SyncPriority(Enum):
    """Data synchronization priorities."""

    CRITICAL = "critical"  # Security updates, emergency messages
    HIGH = "high"  # Agent updates, user data
    MEDIUM = "medium"  # Background sync, cache updates
    LOW = "low"  # Optional content, analytics


@dataclass
class OfflineMessage:
    """Message stored for offline delivery."""

    message_id: str
    sender: str
    recipient: str
    content: bytes
    priority: SyncPriority
    timestamp: datetime
    expiry: datetime | None = None
    delivery_attempts: int = 0
    max_attempts: int = 5

    def is_expired(self) -> bool:
        """Check if message has expired."""
        return self.expiry is not None and datetime.utcnow() > self.expiry

    def should_retry(self) -> bool:
        """Check if delivery should be retried."""
        return not self.is_expired() and self.delivery_attempts < self.max_attempts


@dataclass
class DataCache:
    """Cached data with metadata."""

    key: str
    data: bytes
    timestamp: datetime
    priority: SyncPriority
    size_bytes: int
    ttl_hours: int = 24
    access_count: int = 0
    last_accessed: datetime | None = None

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        expiry = self.timestamp + timedelta(hours=self.ttl_hours)
        return datetime.utcnow() > expiry

    def access(self) -> None:
        """Record cache access."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


@dataclass
class ConnectivityWindow:
    """Detected connectivity window for sync operations."""

    start_time: datetime
    estimated_duration: timedelta
    bandwidth_mbps: float
    data_cost_per_mb: float
    connection_type: str  # wifi, cellular, satellite

    def bytes_affordable(self, budget_usd: float) -> int:
        """Calculate how many bytes can be transferred within budget."""
        if self.data_cost_per_mb <= 0:
            return 100 * 1024 * 1024  # 100MB if free

        affordable_mb = budget_usd / self.data_cost_per_mb
        return int(affordable_mb * 1024 * 1024)

    def estimated_transfer_time(self, size_bytes: int) -> timedelta:
        """Estimate transfer time for given data size."""
        if self.bandwidth_mbps <= 0:
            return timedelta(hours=1)  # Conservative estimate

        # Convert to bytes per second, add 50% overhead
        bytes_per_second = (self.bandwidth_mbps * 1024 * 1024 / 8) * 0.5
        seconds = size_bytes / bytes_per_second
        return timedelta(seconds=seconds)


class GlobalSouthOfflineCoordinator:
    """
    Offline-first coordinator optimized for Global South scenarios.

    Features:
    - Store-and-forward messaging with priority queues
    - Intelligent sync windows for intermittent connectivity
    - Data cost awareness and budget management
    - Adaptive caching with LRU/priority eviction
    - P2P mesh networking for local distribution
    - Compression and deduplication to minimize bandwidth usage
    """

    def __init__(self, storage_path: Path = None, max_storage_mb: int = 500, daily_data_budget_usd: float = 0.50):
        """Initialize offline coordinator."""
        self.storage_path = storage_path or Path("data/offline")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Storage management
        self.max_storage_bytes = max_storage_mb * 1024 * 1024
        self.current_storage_bytes = 0

        # Data budget management
        self.daily_data_budget_usd = daily_data_budget_usd
        self.current_day_spending = 0.0
        self.last_budget_reset = datetime.utcnow().date()

        # Message queues by priority
        self.message_queues: dict[SyncPriority, deque] = {priority: deque() for priority in SyncPriority}
        self.message_index: dict[str, OfflineMessage] = {}

        # Data caching
        self.cache: dict[str, DataCache] = {}
        self.cache_lru_order: deque = deque()

        # Connectivity monitoring
        self.connectivity_mode = ConnectivityMode.FULLY_OFFLINE
        self.connectivity_windows: list[ConnectivityWindow] = []
        self.bandwidth_history: deque = deque(maxlen=100)

        # Sync state
        self.sync_in_progress = False
        self.sync_statistics = {
            "total_syncs": 0,
            "successful_syncs": 0,
            "bytes_synced": 0,
            "cost_spent_usd": 0.0,
            "messages_delivered": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # P2P mesh integration
        self.mesh_peers: set[str] = set()
        self.peer_capabilities: dict[str, dict[str, Any]] = {}

        # Compression settings
        self.compression_enabled = True
        self.compression_threshold_bytes = 1024  # Compress files > 1KB

        logger.info("Global South Offline Coordinator initialized")
        logger.info(f"Storage limit: {max_storage_mb}MB, Daily budget: ${daily_data_budget_usd}")

    async def store_message(
        self,
        sender: str,
        recipient: str,
        content: bytes,
        priority: SyncPriority = SyncPriority.MEDIUM,
        ttl_hours: int = 48,
    ) -> str:
        """Store message for offline delivery."""
        message_id = f"msg_{int(time.time() * 1000)}_{len(content)}"

        # Calculate expiry
        expiry = datetime.utcnow() + timedelta(hours=ttl_hours)

        # Compress content if beneficial
        compressed_content = await self._compress_data(content)

        # Create offline message
        message = OfflineMessage(
            message_id=message_id,
            sender=sender,
            recipient=recipient,
            content=compressed_content,
            priority=priority,
            timestamp=datetime.utcnow(),
            expiry=expiry,
        )

        # Check storage capacity
        message_size = len(compressed_content) + 200  # Metadata overhead
        if not await self._ensure_storage_capacity(message_size):
            raise ValueError(f"Insufficient storage capacity for message ({message_size} bytes)")

        # Add to appropriate queue
        self.message_queues[priority].append(message)
        self.message_index[message_id] = message
        self.current_storage_bytes += message_size

        # Persist message to disk
        await self._persist_message(message)

        logger.info(
            f"Stored offline message: {message_id} "
            f"({len(content)} -> {len(compressed_content)} bytes, "
            f"priority={priority.value})"
        )

        return message_id

    async def cache_data(
        self, key: str, data: bytes, priority: SyncPriority = SyncPriority.MEDIUM, ttl_hours: int = 24
    ) -> bool:
        """Cache data for offline access."""
        # Compress data if beneficial
        compressed_data = await self._compress_data(data)
        cache_size = len(compressed_data) + len(key.encode()) + 100  # Metadata overhead

        # Check storage capacity
        if not await self._ensure_storage_capacity(cache_size):
            logger.warning(f"Cannot cache {key}: insufficient storage")
            return False

        # Remove existing entry if present
        if key in self.cache:
            await self._remove_cache_entry(key)

        # Create cache entry
        cache_entry = DataCache(
            key=key,
            data=compressed_data,
            timestamp=datetime.utcnow(),
            priority=priority,
            size_bytes=cache_size,
            ttl_hours=ttl_hours,
        )

        # Add to cache
        self.cache[key] = cache_entry
        self.cache_lru_order.append(key)
        self.current_storage_bytes += cache_size

        # Persist to disk
        await self._persist_cache_entry(cache_entry)

        logger.debug(f"Cached data: {key} ({len(data)} -> {len(compressed_data)} bytes)")
        return True

    async def get_cached_data(self, key: str) -> bytes | None:
        """Retrieve cached data."""
        if key not in self.cache:
            self.sync_statistics["cache_misses"] += 1
            return None

        cache_entry = self.cache[key]

        # Check expiry
        if cache_entry.is_expired():
            await self._remove_cache_entry(key)
            self.sync_statistics["cache_misses"] += 1
            return None

        # Update access statistics
        cache_entry.access()

        # Update LRU order
        self.cache_lru_order.remove(key)
        self.cache_lru_order.append(key)

        # Decompress data
        data = await self._decompress_data(cache_entry.data)

        self.sync_statistics["cache_hits"] += 1
        return data

    async def detect_connectivity_window(self) -> ConnectivityWindow | None:
        """Detect available connectivity window for sync operations."""
        # Check current connectivity
        connectivity_info = await self._check_connectivity()

        if not connectivity_info["connected"]:
            self.connectivity_mode = ConnectivityMode.FULLY_OFFLINE
            return None

        # Estimate bandwidth
        bandwidth_mbps = await self._estimate_bandwidth()

        # Determine connection type and cost
        connection_type = connectivity_info.get("type", "unknown")
        data_cost_per_mb = await self._get_data_cost(connection_type)

        # Estimate window duration based on connection stability
        stability_score = connectivity_info.get("stability", 0.5)
        base_duration = timedelta(minutes=5)  # Minimum window
        estimated_duration = base_duration * (1 + stability_score)

        window = ConnectivityWindow(
            start_time=datetime.utcnow(),
            estimated_duration=estimated_duration,
            bandwidth_mbps=bandwidth_mbps,
            data_cost_per_mb=data_cost_per_mb,
            connection_type=connection_type,
        )

        # Update connectivity mode
        if bandwidth_mbps < 1.0:
            self.connectivity_mode = ConnectivityMode.LIMITED_BANDWIDTH
        elif connectivity_info.get("intermittent", False):
            self.connectivity_mode = ConnectivityMode.INTERMITTENT
        else:
            self.connectivity_mode = ConnectivityMode.FULL_CONNECTIVITY

        self.connectivity_windows.append(window)
        logger.info(
            f"Detected connectivity window: {connection_type} "
            f"({bandwidth_mbps:.1f} Mbps, ${data_cost_per_mb:.4f}/MB)"
        )

        return window

    async def sync_when_connected(self, window: ConnectivityWindow) -> dict[str, Any]:
        """Perform intelligent sync during connectivity window."""
        if self.sync_in_progress:
            logger.warning("Sync already in progress, skipping")
            return {"status": "skipped", "reason": "sync_in_progress"}

        self.sync_in_progress = True
        sync_start = datetime.utcnow()

        try:
            # Reset daily budget if needed
            await self._check_budget_reset()

            # Calculate available budget
            remaining_budget = self.daily_data_budget_usd - self.current_day_spending
            if remaining_budget <= 0:
                logger.warning("Daily data budget exceeded, skipping sync")
                return {"status": "skipped", "reason": "budget_exceeded"}

            # Determine how much data we can afford to transfer
            affordable_bytes = window.bytes_affordable(remaining_budget)

            logger.info(
                f"Starting sync: budget=${remaining_budget:.3f}, " f"affordable={affordable_bytes / 1024:.1f}KB"
            )

            sync_results = {
                "messages_sent": 0,
                "messages_received": 0,
                "bytes_transferred": 0,
                "cost_spent": 0.0,
                "cache_updates": 0,
                "peer_sync_count": 0,
            }

            # Priority-based message sync
            bytes_used = 0
            for priority in [SyncPriority.CRITICAL, SyncPriority.HIGH, SyncPriority.MEDIUM, SyncPriority.LOW]:
                if bytes_used >= affordable_bytes:
                    break

                queue = self.message_queues[priority]
                messages_sent = 0

                while queue and bytes_used < affordable_bytes:
                    message = queue.popleft()

                    # Skip expired messages
                    if message.is_expired():
                        await self._remove_message(message.message_id)
                        continue

                    # Check if we can afford this message
                    len(message.content) * window.data_cost_per_mb / (1024 * 1024)
                    if bytes_used + len(message.content) > affordable_bytes:
                        # Put message back in queue
                        queue.appendleft(message)
                        break

                    # Attempt delivery
                    try:
                        success = await self._deliver_message(message, window)
                        if success:
                            bytes_used += len(message.content)
                            messages_sent += 1
                            await self._remove_message(message.message_id)
                            logger.debug(f"Delivered message {message.message_id}")
                        else:
                            message.delivery_attempts += 1
                            if message.should_retry():
                                queue.append(message)  # Retry later
                            else:
                                await self._remove_message(message.message_id)
                                logger.warning(f"Message {message.message_id} exceeded retry limit")
                    except Exception as e:
                        logger.error(f"Failed to deliver message {message.message_id}: {e}")
                        queue.appendleft(message)  # Put back for retry
                        break

                sync_results["messages_sent"] += messages_sent
                if messages_sent > 0:
                    logger.info(f"Sent {messages_sent} {priority.value} priority messages")

            # Receive new messages
            try:
                new_messages = await self._receive_messages(window, affordable_bytes - bytes_used)
                sync_results["messages_received"] = len(new_messages)

                for message in new_messages:
                    await self.store_message(
                        message["sender"],
                        message["recipient"],
                        message["content"],
                        SyncPriority(message.get("priority", "medium")),
                    )

            except Exception as e:
                logger.error(f"Failed to receive messages: {e}")

            # Update cache from remote sources
            if bytes_used < affordable_bytes * 0.8:  # Reserve 20% for cache updates
                cache_updates = await self._sync_cache_updates(window, affordable_bytes - bytes_used)
                sync_results["cache_updates"] = cache_updates

            # P2P mesh sync with nearby peers
            peer_sync_count = await self._sync_with_mesh_peers(window, affordable_bytes - bytes_used)
            sync_results["peer_sync_count"] = peer_sync_count

            # Calculate final costs
            total_bytes = (
                sync_results["messages_sent"] + sync_results["messages_received"] + sync_results["cache_updates"]
            )
            cost_spent = total_bytes * window.data_cost_per_mb / (1024 * 1024)

            sync_results["bytes_transferred"] = total_bytes
            sync_results["cost_spent"] = cost_spent

            # Update statistics
            self.current_day_spending += cost_spent
            self.sync_statistics["total_syncs"] += 1
            self.sync_statistics["successful_syncs"] += 1
            self.sync_statistics["bytes_synced"] += total_bytes
            self.sync_statistics["cost_spent_usd"] += cost_spent
            self.sync_statistics["messages_delivered"] += sync_results["messages_sent"]

            sync_duration = datetime.utcnow() - sync_start
            logger.info(
                f"Sync completed in {sync_duration.total_seconds():.1f}s: "
                f"{sync_results['messages_sent']} sent, "
                f"{sync_results['messages_received']} received, "
                f"${cost_spent:.4f} spent"
            )

            return {"status": "success", "duration_seconds": sync_duration.total_seconds(), **sync_results}

        except Exception as e:
            logger.error(f"Sync failed: {e}")
            self.sync_statistics["total_syncs"] += 1
            return {"status": "error", "error": str(e)}

        finally:
            self.sync_in_progress = False

    async def get_pending_message_count(self) -> dict[str, int]:
        """Get count of pending messages by priority."""
        await self._cleanup_expired_messages()

        return {priority.value: len(queue) for priority, queue in self.message_queues.items()}

    async def get_storage_status(self) -> dict[str, Any]:
        """Get current storage status."""
        await self._cleanup_expired_cache()

        return {
            "storage_used_bytes": self.current_storage_bytes,
            "storage_limit_bytes": self.max_storage_bytes,
            "storage_used_percent": (self.current_storage_bytes / self.max_storage_bytes) * 100,
            "cached_items": len(self.cache),
            "pending_messages": sum(len(queue) for queue in self.message_queues.values()),
            "daily_budget_remaining": self.daily_data_budget_usd - self.current_day_spending,
            "connectivity_mode": self.connectivity_mode.value,
        }

    async def get_sync_statistics(self) -> dict[str, Any]:
        """Get sync performance statistics."""
        return {
            **self.sync_statistics,
            "cache_hit_rate": (
                self.sync_statistics["cache_hits"]
                / max(1, self.sync_statistics["cache_hits"] + self.sync_statistics["cache_misses"])
            ),
            "sync_success_rate": (
                self.sync_statistics["successful_syncs"] / max(1, self.sync_statistics["total_syncs"])
            ),
            "average_cost_per_sync": (
                self.sync_statistics["cost_spent_usd"] / max(1, self.sync_statistics["successful_syncs"])
            ),
        }

    # Internal methods

    async def _compress_data(self, data: bytes) -> bytes:
        """Compress data if beneficial."""
        if not self.compression_enabled or len(data) < self.compression_threshold_bytes:
            return data

        try:
            import gzip

            compressed = gzip.compress(data)

            # Only use compression if it saves significant space
            if len(compressed) < len(data) * 0.9:  # At least 10% saving
                return compressed
            else:
                return data
        except Exception:
            return data

    async def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data if needed."""
        if not self.compression_enabled:
            return data

        try:
            # Try to decompress - if it fails, data wasn't compressed
            import gzip

            return gzip.decompress(data)
        except Exception:
            # Data wasn't compressed or compression failed
            return data

    async def _ensure_storage_capacity(self, required_bytes: int) -> bool:
        """Ensure sufficient storage capacity."""
        while self.current_storage_bytes + required_bytes > self.max_storage_bytes:
            # Try to free space by evicting cache entries
            evicted = await self._evict_cache_entry()
            if not evicted:
                # Try to remove expired messages
                expired_removed = await self._cleanup_expired_messages()
                if expired_removed == 0:
                    # No more space can be freed
                    return False

        return True

    async def _evict_cache_entry(self) -> bool:
        """Evict least important cache entry."""
        if not self.cache:
            return False

        # Find entry to evict (LRU with priority consideration)
        candidates = []

        # Prefer low priority, old entries
        for key in list(self.cache_lru_order):
            if key not in self.cache:
                continue

            entry = self.cache[key]
            score = entry.access_count

            # Boost score for higher priority
            if entry.priority == SyncPriority.CRITICAL:
                score += 1000
            elif entry.priority == SyncPriority.HIGH:
                score += 100
            elif entry.priority == SyncPriority.MEDIUM:
                score += 10

            candidates.append((score, key))

        if not candidates:
            return False

        # Evict lowest scoring entry
        candidates.sort()
        _, key_to_evict = candidates[0]

        await self._remove_cache_entry(key_to_evict)
        logger.debug(f"Evicted cache entry: {key_to_evict}")
        return True

    async def _remove_cache_entry(self, key: str) -> None:
        """Remove cache entry."""
        if key not in self.cache:
            return

        entry = self.cache[key]
        self.current_storage_bytes -= entry.size_bytes

        del self.cache[key]
        if key in self.cache_lru_order:
            self.cache_lru_order.remove(key)

        # Remove from disk
        cache_file = self.storage_path / "cache" / f"{key}.cache"
        cache_file.unlink(missing_ok=True)

    async def _remove_message(self, message_id: str) -> None:
        """Remove message from storage."""
        if message_id not in self.message_index:
            return

        message = self.message_index[message_id]
        message_size = len(message.content) + 200  # Metadata overhead

        self.current_storage_bytes -= message_size
        del self.message_index[message_id]

        # Remove from disk
        message_file = self.storage_path / "messages" / f"{message_id}.msg"
        message_file.unlink(missing_ok=True)

    async def _cleanup_expired_messages(self) -> int:
        """Clean up expired messages."""
        removed_count = 0

        for priority, queue in self.message_queues.items():
            expired_messages = []

            # Identify expired messages
            for message in list(queue):
                if message.is_expired():
                    expired_messages.append(message)

            # Remove expired messages
            for message in expired_messages:
                queue.remove(message)
                await self._remove_message(message.message_id)
                removed_count += 1

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired messages")

        return removed_count

    async def _cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries."""
        expired_keys = []

        for key, entry in self.cache.items():
            if entry.is_expired():
                expired_keys.append(key)

        for key in expired_keys:
            await self._remove_cache_entry(key)

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)

    async def _check_connectivity(self) -> dict[str, Any]:
        """Check current network connectivity."""
        # This would integrate with actual network monitoring
        # For now, return simulated connectivity info

        import socket

        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return {
                "connected": True,
                "type": "wifi",  # or "cellular", "satellite"
                "stability": 0.8,
                "intermittent": False,
            }
        except Exception:
            return {"connected": False, "type": "none", "stability": 0.0, "intermittent": False}

    async def _estimate_bandwidth(self) -> float:
        """Estimate current bandwidth in Mbps."""
        # This would perform actual bandwidth testing
        # For now, return conservative estimate
        return 2.0  # 2 Mbps

    async def _get_data_cost(self, connection_type: str) -> float:
        """Get data cost per MB for connection type."""
        # Cost estimates for different connection types in Global South
        costs = {
            "wifi": 0.0,  # Usually free
            "cellular": 0.01,  # $0.01 per MB
            "satellite": 0.05,  # $0.05 per MB
            "unknown": 0.005,  # Conservative estimate
        }
        return costs.get(connection_type, 0.005)

    async def _check_budget_reset(self) -> None:
        """Reset daily budget if new day."""
        today = datetime.utcnow().date()
        if today > self.last_budget_reset:
            self.current_day_spending = 0.0
            self.last_budget_reset = today
            logger.info("Daily data budget reset")

    async def _deliver_message(self, message: OfflineMessage, window: ConnectivityWindow) -> bool:
        """Attempt to deliver a message."""
        # This would integrate with actual message delivery systems
        # For now, simulate delivery with some probability of success

        import random

        # Higher priority messages have better delivery success
        success_rates = {
            SyncPriority.CRITICAL: 0.95,
            SyncPriority.HIGH: 0.90,
            SyncPriority.MEDIUM: 0.85,
            SyncPriority.LOW: 0.80,
        }

        success_rate = success_rates[message.priority]

        # Adjust for bandwidth and connection quality
        if window.bandwidth_mbps > 5.0:
            success_rate += 0.05
        elif window.bandwidth_mbps < 1.0:
            success_rate -= 0.10

        return random.random() < success_rate

    async def _receive_messages(self, window: ConnectivityWindow, max_bytes: int) -> list[dict[str, Any]]:
        """Receive new messages from remote sources."""
        # This would integrate with actual message receiving systems
        # For now, return empty list
        return []

    async def _sync_cache_updates(self, window: ConnectivityWindow, max_bytes: int) -> int:
        """Sync cache updates from remote sources."""
        # This would integrate with actual cache synchronization
        # For now, return 0 updates
        return 0

    async def _sync_with_mesh_peers(self, window: ConnectivityWindow, max_bytes: int) -> int:
        """Sync with P2P mesh peers."""
        # This would integrate with P2P mesh networking
        # For now, return 0 peers synced
        return 0

    async def _persist_message(self, message: OfflineMessage) -> None:
        """Persist message to disk."""
        messages_dir = self.storage_path / "messages"
        messages_dir.mkdir(exist_ok=True)

        message_file = messages_dir / f"{message.message_id}.msg"

        # Serialize message
        message_data = {
            "message_id": message.message_id,
            "sender": message.sender,
            "recipient": message.recipient,
            "content": message.content.hex(),  # Store as hex string
            "priority": message.priority.value,
            "timestamp": message.timestamp.isoformat(),
            "expiry": message.expiry.isoformat() if message.expiry else None,
            "delivery_attempts": message.delivery_attempts,
            "max_attempts": message.max_attempts,
        }

        with open(message_file, "w") as f:
            json.dump(message_data, f)

    async def _persist_cache_entry(self, entry: DataCache) -> None:
        """Persist cache entry to disk."""
        cache_dir = self.storage_path / "cache"
        cache_dir.mkdir(exist_ok=True)

        cache_file = cache_dir / f"{entry.key}.cache"

        # Serialize cache entry
        cache_data = {
            "key": entry.key,
            "data": entry.data.hex(),  # Store as hex string
            "timestamp": entry.timestamp.isoformat(),
            "priority": entry.priority.value,
            "size_bytes": entry.size_bytes,
            "ttl_hours": entry.ttl_hours,
            "access_count": entry.access_count,
            "last_accessed": entry.last_accessed.isoformat() if entry.last_accessed else None,
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

    async def load_persisted_data(self) -> None:
        """Load persisted messages and cache from disk."""
        logger.info("Loading persisted offline data...")

        # Load messages
        messages_dir = self.storage_path / "messages"
        if messages_dir.exists():
            for message_file in messages_dir.glob("*.msg"):
                try:
                    with open(message_file) as f:
                        data = json.load(f)

                    # Reconstruct message
                    message = OfflineMessage(
                        message_id=data["message_id"],
                        sender=data["sender"],
                        recipient=data["recipient"],
                        content=bytes.fromhex(data["content"]),
                        priority=SyncPriority(data["priority"]),
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        expiry=datetime.fromisoformat(data["expiry"]) if data["expiry"] else None,
                        delivery_attempts=data["delivery_attempts"],
                        max_attempts=data["max_attempts"],
                    )

                    # Skip expired messages
                    if message.is_expired():
                        message_file.unlink()
                        continue

                    # Add to appropriate queue
                    self.message_queues[message.priority].append(message)
                    self.message_index[message.message_id] = message

                except Exception as e:
                    logger.error(f"Failed to load message from {message_file}: {e}")
                    message_file.unlink(missing_ok=True)

        # Load cache
        cache_dir = self.storage_path / "cache"
        if cache_dir.exists():
            for cache_file in cache_dir.glob("*.cache"):
                try:
                    with open(cache_file) as f:
                        data = json.load(f)

                    # Reconstruct cache entry
                    entry = DataCache(
                        key=data["key"],
                        data=bytes.fromhex(data["data"]),
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        priority=SyncPriority(data["priority"]),
                        size_bytes=data["size_bytes"],
                        ttl_hours=data["ttl_hours"],
                        access_count=data["access_count"],
                        last_accessed=datetime.fromisoformat(data["last_accessed"]) if data["last_accessed"] else None,
                    )

                    # Skip expired entries
                    if entry.is_expired():
                        cache_file.unlink()
                        continue

                    # Add to cache
                    self.cache[entry.key] = entry
                    self.cache_lru_order.append(entry.key)

                except Exception as e:
                    logger.error(f"Failed to load cache from {cache_file}: {e}")
                    cache_file.unlink(missing_ok=True)

        # Update storage usage
        self.current_storage_bytes = sum(len(msg.content) + 200 for msg in self.message_index.values()) + sum(
            entry.size_bytes for entry in self.cache.values()
        )

        logger.info(
            f"Loaded {len(self.message_index)} messages and {len(self.cache)} cache entries "
            f"({self.current_storage_bytes / 1024 / 1024:.1f}MB)"
        )


async def create_offline_coordinator(
    storage_path: Path = None, max_storage_mb: int = 500, daily_data_budget_usd: float = 0.50
) -> GlobalSouthOfflineCoordinator:
    """Create and initialize offline coordinator."""
    coordinator = GlobalSouthOfflineCoordinator(
        storage_path=storage_path, max_storage_mb=max_storage_mb, daily_data_budget_usd=daily_data_budget_usd
    )

    # Load any persisted data
    await coordinator.load_persisted_data()

    logger.info("Global South Offline Coordinator ready")
    return coordinator


if __name__ == "__main__":
    # Example usage
    async def main():
        coordinator = await create_offline_coordinator()

        # Store some test messages
        await coordinator.store_message("user1", "user2", b"Hello from offline storage!", SyncPriority.HIGH)

        # Cache some data
        await coordinator.cache_data("config_data", b'{"setting": "value"}', SyncPriority.MEDIUM)

        # Check status
        status = await coordinator.get_storage_status()
        print(f"Storage status: {status}")

        # Try to detect connectivity and sync
        window = await coordinator.detect_connectivity_window()
        if window:
            sync_result = await coordinator.sync_when_connected(window)
            print(f"Sync result: {sync_result}")

    asyncio.run(main())
