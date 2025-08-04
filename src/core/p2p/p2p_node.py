"""Enhanced P2P Node for Evolution-Aware Communication."""

import asyncio
import json
import logging
import ssl
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .encryption_layer import EncryptionLayer
from .message_protocol import MessageProtocol
from .peer_discovery import PeerDiscovery

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Enhanced node status for evolution context."""

    STARTING = "starting"
    ACTIVE = "active"
    CONNECTING = "connecting"
    EVOLVING = "evolving"  # New: Node is participating in evolution
    RESOURCE_CONSTRAINED = "resource_constrained"  # New: Low resources
    DISCONNECTED = "disconnected"
    FAILED = "failed"


@dataclass
class PeerCapabilities:
    """Enhanced peer capabilities for evolution coordination."""

    device_id: str
    cpu_cores: int
    ram_mb: int
    battery_percent: int | None = None
    network_type: str = "unknown"
    trust_score: float = 0.5
    last_seen: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    bandwidth_kbps: int | None = None

    # Evolution-specific capabilities
    can_evolve: bool = True
    evolution_capacity: float = 1.0  # 0-1 scale of evolution participation ability
    available_for_evolution: bool = True
    current_evolution_load: float = 0.0  # Current evolution workload (0-1)

    # Performance characteristics
    device_type: str = "unknown"  # phone, tablet, laptop, desktop
    performance_tier: str = "medium"  # low, medium, high, premium
    thermal_state: str = "normal"  # normal, warm, hot, throttling

    def is_suitable_for_evolution(self) -> bool:
        """Check if peer is suitable for evolution tasks."""
        return (
            self.can_evolve
            and self.available_for_evolution
            and self.current_evolution_load < 0.8
            and self.thermal_state != "throttling"
            and (self.battery_percent is None or self.battery_percent > 20)
        )

    def get_evolution_priority(self) -> float:
        """Calculate evolution priority score (higher = better candidate)."""
        base_score = self.evolution_capacity * 0.4

        # Resource availability bonus
        if self.ram_mb > 4096:
            base_score += 0.2
        if self.cpu_cores >= 4:
            base_score += 0.1

        # Battery penalty
        if self.battery_percent and self.battery_percent < 50:
            base_score -= 0.2

        # Load penalty
        base_score -= self.current_evolution_load * 0.3

        # Trust bonus
        base_score += (self.trust_score - 0.5) * 0.1

        return max(0.0, min(1.0, base_score))


class P2PNode:
    """Evolution-aware P2P node for distributed coordination."""

    def __init__(
        self,
        node_id: str | None = None,
        listen_port: int = 0,
        use_tls: bool = False,
        certfile: str | None = None,
        keyfile: str | None = None,
    ) -> None:
        self.node_id = node_id or str(uuid.uuid4())
        self.listen_port = listen_port
        self.use_tls = use_tls
        self.certfile = certfile
        self.keyfile = keyfile
        self.ssl_context: ssl.SSLContext | None = None
        self.status = NodeStatus.STARTING

        # Core networking
        self.server: asyncio.Server | None = None
        self.connections: dict[str, asyncio.StreamWriter] = {}
        self.peer_registry: dict[str, PeerCapabilities] = {}

        # Enhanced components
        self.peer_discovery = PeerDiscovery(self)
        self.message_protocol = MessageProtocol(self)
        self.encryption = EncryptionLayer(self.node_id)

        # Evolution coordination
        self.evolution_handlers: dict[str, Callable] = {}
        self.active_evolutions: dict[str, dict[str, Any]] = {}
        self.evolution_results: dict[str, Any] = {}

        # Performance tracking
        self.network_stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "evolution_messages": 0,
            "peer_discoveries": 0,
        }

        # Resource awareness
        self.local_capabilities: PeerCapabilities | None = None
        self.resource_monitor: Callable | None = None

        self._register_evolution_handlers()

    async def start(self, resource_monitor: Callable | None = None) -> None:
        """Start P2P node with resource monitoring integration."""
        try:
            self.resource_monitor = resource_monitor

            # Initialize encryption
            await self.encryption.initialize()

            # Start TCP server with optional TLS
            if self.use_tls:
                self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                if self.certfile and self.keyfile:
                    self.ssl_context.load_cert_chain(self.certfile, self.keyfile)
                self.server = await asyncio.start_server(
                    self._handle_connection,
                    "0.0.0.0",
                    self.listen_port,
                    ssl=self.ssl_context,
                )
            else:
                self.server = await asyncio.start_server(
                    self._handle_connection,
                    "0.0.0.0",
                    self.listen_port,
                )

            # Get actual port
            self.listen_port = self.server.sockets[0].getsockname()[1]

            # Update local capabilities
            await self._update_local_capabilities()

            # Start background tasks
            asyncio.create_task(self.peer_discovery.start_discovery())
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._evolution_coordination_loop())

            self.status = NodeStatus.ACTIVE

            logger.info(f"Evolution-aware P2P Node {self.node_id} started on port {self.listen_port}")

        except Exception as e:
            logger.exception(f"Failed to start P2P node: {e}")
            self.status = NodeStatus.FAILED
            raise

    async def stop(self) -> None:
        """Graceful shutdown."""
        logger.info(f"Stopping P2P Node {self.node_id}")

        # Notify peers of shutdown
        await self.broadcast_evolution_event("PEER_SHUTDOWN", {"reason": "graceful_shutdown", "timestamp": time.time()})

        # Stop discovery
        await self.peer_discovery.stop_discovery()

        # Close connections
        for writer in self.connections.values():
            writer.close()
            await writer.wait_closed()

        # Stop server
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        self.status = NodeStatus.DISCONNECTED

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle incoming peer connection with encryption."""
        peer_addr = writer.get_extra_info("peername")
        peer_id = None

        try:
            while True:
                # Read encrypted message
                encrypted_data = await self.message_protocol.read_message(reader)
                if not encrypted_data:
                    break

                # Decrypt message
                try:
                    message_data = await self.encryption.decrypt_message(encrypted_data)
                    message = json.loads(message_data)

                    self.network_stats["messages_received"] += 1
                    self.network_stats["bytes_received"] += len(message_data)

                    # Track peer
                    if "sender_id" in message:
                        peer_id = message["sender_id"]
                        self.connections[peer_id] = writer

                    # Handle message
                    await self._handle_message(message, writer)

                except Exception as e:
                    logger.exception(f"Failed to decrypt/parse message from {peer_addr}: {e}")

        except asyncio.IncompleteReadError:
            pass
        except Exception as e:
            logger.exception(f"Error handling connection from {peer_addr}: {e}")
        finally:
            if peer_id and peer_id in self.connections:
                del self.connections[peer_id]
            writer.close()
            await writer.wait_closed()

    async def _handle_message(self, message: dict, writer: asyncio.StreamWriter) -> None:
        """Enhanced message handling with evolution support."""
        msg_type = message.get("type", "UNKNOWN")

        # Track evolution messages
        if msg_type.startswith("EVOLUTION_"):
            self.network_stats["evolution_messages"] += 1

        # Route to appropriate handler
        if msg_type in self.evolution_handlers:
            await self.evolution_handlers[msg_type](message, writer)
        elif hasattr(self.message_protocol, f"handle_{msg_type.lower()}"):
            handler = getattr(self.message_protocol, f"handle_{msg_type.lower()}")
            await handler(message, writer)
        else:
            logger.warning(f"No handler for message type: {msg_type}")

    def _register_evolution_handlers(self) -> None:
        """Register evolution-specific message handlers."""
        self.evolution_handlers.update(
            {
                "EVOLUTION_START": self._handle_evolution_start,
                "EVOLUTION_PROGRESS": self._handle_evolution_progress,
                "EVOLUTION_COMPLETE": self._handle_evolution_complete,
                "EVOLUTION_REQUEST_HELP": self._handle_evolution_request_help,
                "EVOLUTION_OFFER_HELP": self._handle_evolution_offer_help,
                "EVOLUTION_CONSENSUS": self._handle_evolution_consensus,
                "EVOLUTION_METRICS_SHARE": self._handle_evolution_metrics_share,
            }
        )

    async def _handle_evolution_start(self, message: dict, writer: asyncio.StreamWriter) -> None:
        """Handle evolution start notification."""
        sender_id = message.get("sender_id")
        evolution_type = message.get("evolution_type", "unknown")

        logger.info(f"Peer {sender_id} started {evolution_type} evolution")

        # Update peer status
        if sender_id in self.peer_registry:
            self.peer_registry[sender_id].available_for_evolution = False
            self.peer_registry[sender_id].current_evolution_load = 1.0

        # Store evolution info
        self.active_evolutions[sender_id] = {
            "type": evolution_type,
            "start_time": time.time(),
            "progress": 0.0,
        }

    async def _handle_evolution_progress(self, message: dict, writer: asyncio.StreamWriter) -> None:
        """Handle evolution progress update."""
        sender_id = message.get("sender_id")
        progress = message.get("progress", 0.0)

        if sender_id in self.active_evolutions:
            self.active_evolutions[sender_id]["progress"] = progress

        if sender_id in self.peer_registry:
            self.peer_registry[sender_id].current_evolution_load = progress

    async def _handle_evolution_complete(self, message: dict, writer: asyncio.StreamWriter) -> None:
        """Handle evolution completion."""
        sender_id = message.get("sender_id")
        success = message.get("success", False)
        results = message.get("results", {})

        logger.info(f"Peer {sender_id} completed evolution: {'success' if success else 'failed'}")

        # Update peer status
        if sender_id in self.peer_registry:
            self.peer_registry[sender_id].available_for_evolution = True
            self.peer_registry[sender_id].current_evolution_load = 0.0

        # Store results
        self.evolution_results[sender_id] = {
            "success": success,
            "results": results,
            "timestamp": time.time(),
        }

        # Clean up active evolution
        if sender_id in self.active_evolutions:
            del self.active_evolutions[sender_id]

    async def _handle_evolution_request_help(self, message: dict, writer: asyncio.StreamWriter) -> None:
        """Handle request for evolution assistance."""
        sender_id = message.get("sender_id")
        help_type = message.get("help_type", "general")

        # Check if we can help
        if (
            self.local_capabilities
            and self.local_capabilities.is_suitable_for_evolution()
            and not self.active_evolutions
        ):
            # Offer help
            response = {
                "type": "EVOLUTION_OFFER_HELP",
                "sender_id": self.node_id,
                "help_type": help_type,
                "capabilities": self.local_capabilities.__dict__,
                "availability": True,
            }

            await self.send_to_peer(sender_id, response)

    async def _handle_evolution_offer_help(self, message: dict, writer: asyncio.StreamWriter) -> None:
        """Handle offer of evolution assistance."""
        sender_id = message.get("sender_id")
        message.get("capabilities", {})

        logger.info(f"Received evolution help offer from {sender_id}")

        # Store help offer for potential use
        # This would be used by evolution systems to coordinate distributed work

    async def _handle_evolution_consensus(self, message: dict, writer: asyncio.StreamWriter) -> None:
        """Handle evolution consensus voting."""
        message.get("sender_id")
        message.get("proposal", {})
        message.get("vote")  # approve, reject, abstain

        # This implements distributed consensus for evolution decisions
        # Would be used for coordinating breakthrough discoveries across peers

    async def _handle_evolution_metrics_share(self, message: dict, writer: asyncio.StreamWriter) -> None:
        """Handle shared evolution metrics."""
        message.get("sender_id")
        message.get("metrics", {})

        # Store shared metrics for analysis
        # This enables distributed performance tracking

    async def _update_local_capabilities(self) -> None:
        """Update local device capabilities."""
        if self.resource_monitor:
            try:
                # Get current resource status
                status = await self.resource_monitor()

                # Update capabilities based on current resources
                self.local_capabilities = PeerCapabilities(
                    device_id=self.node_id,
                    cpu_cores=status.get("cpu_cores", 1),
                    ram_mb=status.get("ram_total_mb", 1024),
                    battery_percent=status.get("battery_percent"),
                    network_type=status.get("network_type", "unknown"),
                    device_type=status.get("device_type", "unknown"),
                    performance_tier=self._calculate_performance_tier(status),
                    thermal_state=status.get("thermal_state", "normal"),
                    evolution_capacity=self._calculate_evolution_capacity(status),
                    available_for_evolution=self._check_evolution_availability(status),
                )

            except Exception as e:
                logger.exception(f"Failed to update local capabilities: {e}")

    def _calculate_performance_tier(self, status: dict) -> str:
        """Calculate device performance tier."""
        ram_gb = status.get("ram_total_mb", 1024) / 1024
        cpu_cores = status.get("cpu_cores", 1)

        if ram_gb >= 16 and cpu_cores >= 8:
            return "premium"
        if ram_gb >= 8 and cpu_cores >= 4:
            return "high"
        if ram_gb >= 4 and cpu_cores >= 2:
            return "medium"
        return "low"

    def _calculate_evolution_capacity(self, status: dict) -> float:
        """Calculate evolution participation capacity."""
        base_capacity = 1.0

        # Reduce based on resource usage
        memory_usage = status.get("memory_percent", 50) / 100
        cpu_usage = status.get("cpu_percent", 50) / 100

        # Resource penalty
        resource_penalty = (memory_usage + cpu_usage) / 2
        base_capacity -= resource_penalty * 0.5

        # Battery penalty
        battery = status.get("battery_percent")
        if battery and battery < 30:
            base_capacity -= 0.3

        return max(0.1, min(1.0, base_capacity))

    def _check_evolution_availability(self, status: dict) -> bool:
        """Check if device is available for evolution."""
        # Not available if resource constrained
        if status.get("memory_percent", 0) > 90 or status.get("cpu_percent", 0) > 95:
            return False

        # Not available if battery critical
        battery = status.get("battery_percent")
        if battery and battery < 10:
            return False

        # Not available if already evolving
        return not self.active_evolutions

    async def _heartbeat_loop(self) -> None:
        """Enhanced heartbeat with capability updates."""
        while self.status == NodeStatus.ACTIVE:
            try:
                # Update local capabilities
                await self._update_local_capabilities()

                # Send heartbeat with capabilities
                heartbeat_data = {
                    "capabilities": self.local_capabilities.__dict__ if self.local_capabilities else {},
                    "status": self.status.value,
                    "active_evolutions": len(self.active_evolutions),
                    "network_stats": self.network_stats.copy(),
                }

                await self.broadcast_to_peers("HEARTBEAT", heartbeat_data)
                await asyncio.sleep(30)  # 30 second heartbeat

            except Exception as e:
                logger.exception(f"Heartbeat loop error: {e}")
                await asyncio.sleep(10)

    async def _evolution_coordination_loop(self) -> None:
        """Background loop for evolution coordination."""
        while self.status == NodeStatus.ACTIVE:
            try:
                # Clean up completed evolutions
                current_time = time.time()
                expired_evolutions = [
                    peer_id
                    for peer_id, evolution in self.active_evolutions.items()
                    if current_time - evolution["start_time"] > 3600  # 1 hour timeout
                ]

                for peer_id in expired_evolutions:
                    logger.warning(f"Evolution timeout for peer {peer_id}")
                    del self.active_evolutions[peer_id]

                # Update peer trust scores based on evolution performance
                await self._update_trust_scores()

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.exception(f"Evolution coordination loop error: {e}")
                await asyncio.sleep(30)

    async def _update_trust_scores(self) -> None:
        """Update peer trust scores based on evolution performance."""
        for peer_id, capabilities in self.peer_registry.items():
            if peer_id in self.evolution_results:
                result = self.evolution_results[peer_id]

                # Increase trust for successful evolutions
                if result["success"]:
                    capabilities.trust_score = min(1.0, capabilities.trust_score + 0.05)
                else:
                    capabilities.trust_score = max(0.1, capabilities.trust_score - 0.02)

    async def broadcast_evolution_event(self, event_type: str, data: dict) -> None:
        """Broadcast evolution event to all peers."""
        {
            "type": f"EVOLUTION_{event_type}",
            "sender_id": self.node_id,
            "timestamp": time.time(),
            **data,
        }

        await self.broadcast_to_peers(f"EVOLUTION_{event_type}", data)

    async def send_to_peer(self, peer_id: str, message: dict) -> bool | None:
        """Send encrypted message to specific peer."""
        if peer_id not in self.connections:
            return False

        message["sender_id"] = self.node_id
        message["message_id"] = str(uuid.uuid4())
        message["timestamp"] = time.time()

        try:
            # Encrypt message
            message_data = json.dumps(message)
            encrypted_data = await self.encryption.encrypt_message(message_data)

            # Send via protocol
            await self.message_protocol.send_message(encrypted_data, self.connections[peer_id])

            self.network_stats["messages_sent"] += 1
            self.network_stats["bytes_sent"] += len(message_data)

            return True

        except Exception as e:
            logger.exception(f"Failed to send message to {peer_id}: {e}")
            return False

    async def broadcast_to_peers(self, msg_type: str, data: dict):
        """Broadcast message to all connected peers."""
        message = {
            "type": msg_type,
            "sender_id": self.node_id,
            "message_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            **data,
        }

        success_count = 0
        for peer_id in list(self.connections.keys()):
            if await self.send_to_peer(peer_id, message.copy()):
                success_count += 1

        return success_count

    def get_suitable_evolution_peers(self, min_count: int = 1) -> list[PeerCapabilities]:
        """Get peers suitable for evolution tasks."""
        suitable_peers = [
            capabilities for capabilities in self.peer_registry.values() if capabilities.is_suitable_for_evolution()
        ]

        # Sort by evolution priority
        suitable_peers.sort(key=lambda p: p.get_evolution_priority(), reverse=True)

        return suitable_peers[: max(min_count, 5)]  # Return top 5 or min_count

    def get_network_status(self) -> dict[str, Any]:
        """Get comprehensive network status."""
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "listen_port": self.listen_port,
            "connected_peers": len(self.connections),
            "known_peers": len(self.peer_registry),
            "active_evolutions": len(self.active_evolutions),
            "local_capabilities": self.local_capabilities.__dict__ if self.local_capabilities else None,
            "network_stats": self.network_stats.copy(),
            "suitable_evolution_peers": len(self.get_suitable_evolution_peers()),
        }
