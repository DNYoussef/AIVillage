"""Mesh Network Implementation for Agent P2P Communication.

Provides high-level mesh networking functionality built on LibP2P:
- Mesh network initialization and management
- Peer discovery and connection management
- Message routing and broadcasting
- Network topology management
"""

import asyncio
import logging
import time

from .fallback_transports import FallbackTransportManager
from .libp2p_mesh import LibP2PMeshNetwork
from .mdns_discovery import mDNSDiscovery
from .message_protocol import EvolutionMessage, MessageProtocol

logger = logging.getLogger(__name__)


class MeshNetwork:
    """High-level mesh network implementation for agent communication."""

    def __init__(
        self,
        node_id: str,
        listen_port: int = 4001,
        enable_mdns: bool = True,
        enable_fallbacks: bool = True,
    ) -> None:
        self.node_id = node_id
        self.listen_port = listen_port
        self.enable_mdns = enable_mdns
        self.enable_fallbacks = enable_fallbacks

        # Core components
        self.libp2p_mesh: LibP2PMeshNetwork | None = None
        self.mdns_discovery: mDNSDiscovery | None = None
        self.fallback_manager: FallbackTransportManager | None = None
        self.message_protocol = MessageProtocol(node_id)

        # Network state
        self.connected_peers: set[str] = set()
        self.peer_info: dict[str, dict] = {}
        self.message_handlers: dict[str, callable] = {}
        self.running = False

        # Network statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "peers_discovered": 0,
            "connection_attempts": 0,
            "successful_connections": 0,
            "start_time": None,
        }

    async def start(self) -> bool:
        """Start the mesh network."""
        if self.running:
            return True

        try:
            logger.info(f"Starting mesh network for node {self.node_id}")
            self.stats["start_time"] = time.time()

            # Initialize LibP2P mesh
            try:
                self.libp2p_mesh = LibP2PMeshNetwork(node_id=self.node_id, listen_port=self.listen_port)
                await self.libp2p_mesh.start()
                logger.info("LibP2P mesh network started successfully")
            except Exception as e:
                logger.warning(f"LibP2P mesh failed to start: {e}")
                self.libp2p_mesh = None

            # Initialize mDNS discovery
            if self.enable_mdns:
                try:
                    self.mdns_discovery = mDNSDiscovery(self.node_id, listen_port=self.listen_port)
                    await self.mdns_discovery.start()
                    logger.info("mDNS peer discovery started")
                except Exception as e:
                    logger.warning(f"mDNS discovery failed to start: {e}")
                    self.mdns_discovery = None

            # Initialize fallback transports
            if self.enable_fallbacks:
                try:
                    self.fallback_manager = FallbackTransportManager(self.node_id)
                    await self.fallback_manager.start()
                    logger.info("Fallback transports initialized")
                except Exception as e:
                    logger.warning(f"Fallback transports failed to start: {e}")
                    self.fallback_manager = None

            # Start message processing
            asyncio.create_task(self._message_processing_loop())
            asyncio.create_task(self._peer_discovery_loop())

            self.running = True
            logger.info(f"Mesh network started successfully for {self.node_id}")
            return True

        except Exception as e:
            logger.exception(f"Failed to start mesh network: {e}")
            await self.stop()
            return False

    async def stop(self) -> None:
        """Stop the mesh network."""
        if not self.running:
            return

        logger.info(f"Stopping mesh network for {self.node_id}")
        self.running = False

        # Stop components
        if self.libp2p_mesh:
            try:
                await self.libp2p_mesh.stop()
            except Exception as e:
                logger.exception(f"Error stopping LibP2P mesh: {e}")

        if self.mdns_discovery:
            try:
                await self.mdns_discovery.stop_discovery()
            except Exception as e:
                logger.exception(f"Error stopping mDNS discovery: {e}")

        if self.fallback_manager:
            try:
                await self.fallback_manager.stop()
            except Exception as e:
                logger.exception(f"Error stopping fallback manager: {e}")

        logger.info("Mesh network stopped")

    def register_message_handler(self, message_type: str, handler: callable) -> None:
        """Register a handler for specific message types."""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")

    async def send_message(
        self, target_peer: str, message_type: str, payload: dict, metadata: dict | None = None
    ) -> bool:
        """Send a message to a specific peer."""
        try:
            message = EvolutionMessage(
                sender_id=self.node_id,
                message_type=message_type,
                payload=payload,
                metadata=metadata or {},
            )

            # Try LibP2P first
            if self.libp2p_mesh and target_peer in self.connected_peers:
                try:
                    success = await self.libp2p_mesh.send_message(target_peer, message)
                    if success:
                        self.stats["messages_sent"] += 1
                        return True
                except Exception as e:
                    logger.warning(f"LibP2P send failed: {e}")

            # Try fallback transports
            if self.fallback_manager:
                try:
                    success = await self.fallback_manager.send_message(target_peer, message)
                    if success:
                        self.stats["messages_sent"] += 1
                        return True
                except Exception as e:
                    logger.warning(f"Fallback send failed: {e}")

            logger.error(f"Failed to send message to {target_peer}")
            return False

        except Exception as e:
            logger.exception(f"Error sending message: {e}")
            return False

    async def broadcast_message(self, message_type: str, payload: dict, metadata: dict | None = None) -> int:
        """Broadcast a message to all connected peers."""
        sent_count = 0

        for peer_id in self.connected_peers.copy():
            success = await self.send_message(peer_id, message_type, payload, metadata)
            if success:
                sent_count += 1

        logger.info(f"Broadcast message sent to {sent_count} peers")
        return sent_count

    async def connect_to_peer(self, peer_id: str, peer_address: str | None = None) -> bool:
        """Manually connect to a specific peer."""
        try:
            self.stats["connection_attempts"] += 1

            # Try LibP2P connection
            if self.libp2p_mesh:
                success = await self.libp2p_mesh.connect_to_peer(peer_id, peer_address)
                if success:
                    self.connected_peers.add(peer_id)
                    self.stats["successful_connections"] += 1
                    logger.info(f"Connected to peer {peer_id} via LibP2P")
                    return True

            # Try fallback connection
            if self.fallback_manager and peer_address:
                success = await self.fallback_manager.connect_to_peer(peer_id, peer_address)
                if success:
                    self.connected_peers.add(peer_id)
                    self.stats["successful_connections"] += 1
                    logger.info(f"Connected to peer {peer_id} via fallback")
                    return True

            return False

        except Exception as e:
            logger.exception(f"Error connecting to peer {peer_id}: {e}")
            return False

    async def disconnect_from_peer(self, peer_id: str) -> bool:
        """Disconnect from a specific peer."""
        try:
            if peer_id in self.connected_peers:
                self.connected_peers.remove(peer_id)

                # Disconnect from LibP2P
                if self.libp2p_mesh:
                    await self.libp2p_mesh.disconnect_from_peer(peer_id)

                # Disconnect from fallback
                if self.fallback_manager:
                    await self.fallback_manager.disconnect_from_peer(peer_id)

                logger.info(f"Disconnected from peer {peer_id}")
                return True

            return False

        except Exception as e:
            logger.exception(f"Error disconnecting from peer {peer_id}: {e}")
            return False

    def get_connected_peers(self) -> list[str]:
        """Get list of connected peer IDs."""
        return list(self.connected_peers)

    def get_peer_info(self, peer_id: str) -> dict | None:
        """Get information about a specific peer."""
        return self.peer_info.get(peer_id)

    def get_network_stats(self) -> dict:
        """Get network statistics."""
        stats = self.stats.copy()
        stats.update(
            {
                "connected_peers": len(self.connected_peers),
                "uptime_seconds": (time.time() - stats["start_time"] if stats["start_time"] else 0),
                "libp2p_available": self.libp2p_mesh is not None,
                "mdns_available": self.mdns_discovery is not None,
                "fallbacks_available": self.fallback_manager is not None,
            }
        )
        return stats

    async def _message_processing_loop(self) -> None:
        """Background loop for processing incoming messages."""
        while self.running:
            try:
                # Process LibP2P messages
                if self.libp2p_mesh:
                    messages = await self.libp2p_mesh.get_pending_messages()
                    for message in messages:
                        await self._handle_message(message)

                # Process fallback messages
                if self.fallback_manager:
                    messages = await self.fallback_manager.get_pending_messages()
                    for message in messages:
                        await self._handle_message(message)

                await asyncio.sleep(0.1)  # Prevent busy waiting

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in message processing loop: {e}")
                await asyncio.sleep(1)

    async def _peer_discovery_loop(self) -> None:
        """Background loop for peer discovery."""
        while self.running:
            try:
                if self.mdns_discovery:
                    discovered_peers = await self.mdns_discovery.get_discovered_peers()

                    for peer_id, peer_info in discovered_peers.items():
                        if peer_id not in self.connected_peers and peer_id != self.node_id:
                            # Try to connect to newly discovered peer
                            peer_address = peer_info.get("address")
                            if peer_address:
                                success = await self.connect_to_peer(peer_id, peer_address)
                                if success:
                                    self.stats["peers_discovered"] += 1
                                    self.peer_info[peer_id] = peer_info

                await asyncio.sleep(10)  # Discovery every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in peer discovery loop: {e}")
                await asyncio.sleep(5)

    async def _handle_message(self, message: EvolutionMessage) -> None:
        """Handle an incoming message."""
        try:
            self.stats["messages_received"] += 1

            # Look for registered handler
            handler = self.message_handlers.get(message.message_type)
            if handler:
                await handler(message)
            else:
                logger.debug(f"No handler for message type: {message.message_type}")

        except Exception as e:
            logger.exception(f"Error handling message: {e}")


# Legacy compatibility
class MeshNetworkNode(MeshNetwork):
    """Legacy alias for MeshNetwork."""


# Convenience function
async def create_mesh_network(node_id: str, listen_port: int = 4001, auto_start: bool = True) -> MeshNetwork:
    """Create and optionally start a mesh network."""
    mesh = MeshNetwork(node_id, listen_port)

    if auto_start:
        await mesh.start()

    return mesh
