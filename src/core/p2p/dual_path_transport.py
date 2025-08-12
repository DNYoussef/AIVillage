"""Dual-Path Transport Integration - BitChat + Betanet

Unified transport layer that seamlessly integrates BitChat (Bluetooth mesh) and 
Betanet (decentralized internet) protocols with intelligent routing via Navigator agent.

This module provides the bridge between P2P transport and agent coordination layers.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

# Import our dual-path components
try:
    from .betanet_transport import BetanetMessage, BetanetTransport
    from .bitchat_transport import BitChatMessage, BitChatTransport

    BITCHAT_AVAILABLE = True
    BETANET_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Transport import failed: {e}")
    BitChatTransport = None
    BetanetTransport = None
    BITCHAT_AVAILABLE = False
    BETANET_AVAILABLE = False

# Import Navigator for path selection
try:
    import os
    import sys

    sys.path.insert(
        0,
        os.path.join(os.path.dirname(__file__), "../../../experimental/agents/agents"),
    )
    from navigator.path_policy import (
        MessageContext,
        NavigatorAgent,
        NetworkConditions,
        PathProtocol,
    )

    NAVIGATOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Navigator import failed: {e}")
    NavigatorAgent = None
    PathProtocol = None
    NAVIGATOR_AVAILABLE = False

logger = logging.getLogger(__name__)


class DualPathMessage:
    """Unified message format for dual-path transport"""

    def __init__(
        self,
        id: str = None,
        sender: str = "",
        recipient: str = "",
        payload: Union[bytes, str, Dict] = b"",
        priority: int = 5,
        content_type: str = "application/json",
        privacy_required: bool = False,
        deadline: Optional[float] = None,
    ):
        self.id = id or str(uuid.uuid4())
        self.sender = sender
        self.recipient = recipient
        self.priority = priority
        self.content_type = content_type
        self.privacy_required = privacy_required
        self.deadline = deadline
        self.timestamp = time.time()

        # Convert payload to bytes
        if isinstance(payload, str):
            self.payload = payload.encode()
        elif isinstance(payload, dict):
            self.payload = json.dumps(payload).encode()
        else:
            self.payload = payload

        # Message context for Navigator
        self.context = MessageContext(
            size_bytes=len(self.payload),
            priority=priority,
            content_type=content_type,
            requires_realtime=priority >= 8,
            privacy_required=privacy_required,
            delivery_deadline=deadline,
        )

    def to_bitchat_message(self) -> "BitChatMessage":
        """Convert to BitChat format"""
        if not BITCHAT_AVAILABLE:
            raise ImportError("BitChat not available")

        return BitChatMessage(
            id=self.id,
            sender=self.sender,
            recipient=self.recipient,
            payload=self.payload,
            ttl=7,  # BitChat max TTL
            priority=self.priority,
        )

    def to_betanet_message(self) -> "BetanetMessage":
        """Convert to Betanet format"""
        if not BETANET_AVAILABLE:
            raise ImportError("Betanet not available")

        return BetanetMessage(
            id=self.id,
            sender=self.sender,
            recipient=self.recipient,
            payload=self.payload,
            priority=self.priority,
            content_type=self.content_type,
            encryption_layers=2 if self.privacy_required else 0,
        )


class DualPathTransport:
    """Unified dual-path transport for AIVillage

    Provides seamless integration between:
    - BitChat: Bluetooth mesh for offline/local communication
    - Betanet: Decentralized internet for global reach
    - Navigator: Intelligent path selection and routing
    """

    def __init__(
        self,
        node_id: Optional[str] = None,
        enable_bitchat: bool = True,
        enable_betanet: bool = True,
    ):
        self.node_id = node_id or f"dualpath_{uuid.uuid4().hex[:12]}"

        # Initialize transport components
        self.bitchat: Optional[BitChatTransport] = None
        self.betanet: Optional[BetanetTransport] = None
        self.navigator: Optional[NavigatorAgent] = None

        # Configuration
        self.enable_bitchat = enable_bitchat and BITCHAT_AVAILABLE
        self.enable_betanet = enable_betanet and BETANET_AVAILABLE
        self.enable_navigator = NAVIGATOR_AVAILABLE

        # Message handling
        self.message_handlers: Dict[str, Any] = {}
        self.routing_stats = {
            "bitchat_sent": 0,
            "betanet_sent": 0,
            "store_forward_queued": 0,
            "routing_decisions": 0,
            "delivery_failures": 0,
        }

        # Store-and-forward queue for offline scenarios
        self.offline_queue: List[
            Tuple[DualPathMessage, float]
        ] = []  # (message, queued_time)
        self.max_queue_size = 1000

        # Control
        self.is_running = False
        self.maintenance_task: Optional[asyncio.Task] = None

        logger.info(f"DualPathTransport initialized: {self.node_id}")
        logger.info(
            f"Available transports: BitChat={self.enable_bitchat}, "
            f"Betanet={self.enable_betanet}, Navigator={self.enable_navigator}"
        )

    async def start(self) -> bool:
        """Start dual-path transport system"""
        if self.is_running:
            logger.warning("DualPathTransport already running")
            return True

        logger.info("Starting DualPathTransport...")

        try:
            # Initialize Navigator for path selection
            if self.enable_navigator:
                from navigator.path_policy import RoutingPriority

                self.navigator = NavigatorAgent(
                    agent_id=f"nav_{self.node_id}",
                    routing_priority=RoutingPriority.OFFLINE_FIRST,
                )
                # Enable Global South optimizations
                self.navigator.enable_global_south_mode(True)

            # Initialize BitChat transport
            if self.enable_bitchat:
                self.bitchat = BitChatTransport(device_id=f"bc_{self.node_id}")
                self.bitchat.register_handler("default", self._handle_bitchat_message)

                bitchat_success = await self.bitchat.start()
                if not bitchat_success:
                    logger.warning("BitChat failed to start")
                    self.enable_bitchat = False
                else:
                    logger.info("BitChat transport started successfully")

            # Initialize Betanet transport
            if self.enable_betanet:
                self.betanet = BetanetTransport(peer_id=f"bn_{self.node_id}")
                self.betanet.register_handler("default", self._handle_betanet_message)

                betanet_success = await self.betanet.start()
                if not betanet_success:
                    logger.warning("Betanet failed to start")
                    self.enable_betanet = False
                else:
                    logger.info("Betanet transport started successfully")

            # Start maintenance loop
            self.maintenance_task = asyncio.create_task(self._maintenance_loop())

            self.is_running = True

            # Sync peer information between transports
            await self._sync_peer_information()

            logger.info("DualPathTransport started successfully")
            return True

        except Exception as e:
            logger.exception(f"Failed to start DualPathTransport: {e}")
            return False

    async def stop(self) -> None:
        """Stop dual-path transport system"""
        logger.info("Stopping DualPathTransport...")
        self.is_running = False

        # Cancel maintenance
        if self.maintenance_task:
            self.maintenance_task.cancel()
            try:
                await self.maintenance_task
            except asyncio.CancelledError:
                pass

        # Stop transports
        if self.bitchat:
            await self.bitchat.stop()

        if self.betanet:
            await self.betanet.stop()

        logger.info("DualPathTransport stopped")

    async def send_message(
        self,
        recipient: str,
        payload: Union[bytes, str, Dict],
        priority: int = 5,
        privacy_required: bool = False,
        deadline: Optional[float] = None,
        preferred_protocol: Optional[str] = None,
    ) -> bool:
        """Send message via optimal dual-path route

        Args:
            recipient: Target node/agent ID
            payload: Message content
            priority: Priority 1-10 (10=urgent)
            privacy_required: Use privacy routing (mixnodes)
            deadline: Delivery deadline (timestamp)
            preferred_protocol: Force specific protocol ('bitchat'|'betanet')

        Returns:
            True if message sent successfully
        """
        if not self.is_running:
            logger.warning("DualPathTransport not running")
            return False

        # Create unified message
        message = DualPathMessage(
            sender=self.node_id,
            recipient=recipient,
            payload=payload,
            priority=priority,
            privacy_required=privacy_required,
            deadline=deadline,
        )

        # Select optimal path via Navigator
        if preferred_protocol:
            # Use forced protocol
            if preferred_protocol == "bitchat":
                selected_protocol = PathProtocol.BITCHAT
            elif preferred_protocol == "betanet":
                selected_protocol = PathProtocol.BETANET
            else:
                selected_protocol = PathProtocol.STORE_FORWARD
            routing_metadata = {}
        else:
            # Use Navigator for intelligent path selection
            selected_protocol, routing_metadata = await self._select_optimal_path(
                recipient, message.context
            )

        self.routing_stats["routing_decisions"] += 1

        # Route message via selected protocol
        success = await self._route_message(
            message, selected_protocol, routing_metadata
        )

        if not success:
            self.routing_stats["delivery_failures"] += 1

            # Fallback to store-and-forward
            logger.info(
                f"Routing failed for {message.id[:8]} - queuing for later delivery"
            )
            await self._queue_for_store_forward(message)

        return success

    async def broadcast_message(
        self, payload: Union[bytes, str, Dict], priority: int = 5, max_hops: int = 5
    ) -> int:
        """Broadcast message to all reachable peers

        Returns:
            Number of peers message was sent to
        """
        if not self.is_running:
            return 0

        message = DualPathMessage(
            sender=self.node_id,
            recipient="",  # Broadcast
            payload=payload,
            priority=priority,
        )

        sent_count = 0

        # Broadcast via BitChat (mesh flooding)
        if self.enable_bitchat and self.bitchat:
            bitchat_msg = message.to_bitchat_message()
            bitchat_msg.ttl = min(max_hops, 7)  # BitChat limit

            if await self.bitchat.send_message(
                "", bitchat_msg.payload, priority, bitchat_msg.ttl
            ):
                sent_count += self.bitchat.get_peer_count()
                self.routing_stats["bitchat_sent"] += 1

        # Broadcast via Betanet (to connected peers)
        if self.enable_betanet and self.betanet:
            betanet_msg = message.to_betanet_message()

            # Send to all connected Betanet peers
            for peer_id in self.betanet.active_connections:
                if await self.betanet.send_message(peer_id, betanet_msg.payload):
                    sent_count += 1

            if sent_count > 0:
                self.routing_stats["betanet_sent"] += 1

        logger.info(f"Broadcast message {message.id[:8]} to {sent_count} peers")
        return sent_count

    async def _select_optimal_path(
        self, destination: str, context: MessageContext
    ) -> Tuple[PathProtocol, Dict[str, Any]]:
        """Select optimal path using Navigator agent"""
        if not self.enable_navigator or not self.navigator:
            # Fallback path selection without Navigator
            if (
                self.enable_bitchat
                and self.bitchat
                and self.bitchat.is_peer_reachable(destination)
            ):
                return PathProtocol.BITCHAT, {"fallback": True}
            elif self.enable_betanet:
                return PathProtocol.BETANET, {"fallback": True}
            else:
                return PathProtocol.STORE_FORWARD, {"fallback": True}

        # Get available protocols
        available_protocols = []
        if self.enable_bitchat and self.bitchat:
            available_protocols.append("bitchat")
        if self.enable_betanet and self.betanet:
            available_protocols.append("betanet")
        available_protocols.append("store_forward")

        # Use Navigator for intelligent selection
        return await self.navigator.select_path(
            destination, context, available_protocols
        )

    async def _route_message(
        self, message: DualPathMessage, protocol: PathProtocol, metadata: Dict[str, Any]
    ) -> bool:
        """Route message via selected protocol"""

        if protocol == PathProtocol.BITCHAT:
            return await self._send_via_bitchat(message, metadata)

        elif protocol == PathProtocol.BETANET:
            return await self._send_via_betanet(message, metadata)

        elif protocol == PathProtocol.STORE_FORWARD:
            await self._queue_for_store_forward(message)
            return True

        else:
            logger.error(f"Unknown protocol: {protocol}")
            return False

    async def _send_via_bitchat(
        self, message: DualPathMessage, metadata: Dict[str, Any]
    ) -> bool:
        """Send message via BitChat transport"""
        if not self.enable_bitchat or not self.bitchat:
            logger.warning("BitChat not available")
            return False

        try:
            bitchat_msg = message.to_bitchat_message()

            # Apply BitChat optimizations from metadata
            if metadata.get("store_forward_enabled", True):
                # BitChat handles store-and-forward automatically
                pass

            success = await self.bitchat.send_message(
                message.recipient,
                message.payload,
                message.priority,
                metadata.get("ttl_hops", 7),
            )

            if success:
                self.routing_stats["bitchat_sent"] += 1
                logger.debug(f"Sent message {message.id[:8]} via BitChat")

            return success

        except Exception as e:
            logger.exception(f"BitChat send failed: {e}")
            return False

    async def _send_via_betanet(
        self, message: DualPathMessage, metadata: Dict[str, Any]
    ) -> bool:
        """Send message via Betanet transport"""
        if not self.enable_betanet or not self.betanet:
            logger.warning("Betanet not available")
            return False

        try:
            betanet_msg = message.to_betanet_message()

            # Apply Betanet optimizations from metadata
            use_mixnodes = metadata.get("privacy_routing", message.privacy_required)

            success = await self.betanet.send_message(
                message.recipient,
                message.payload,
                betanet_msg.protocol,
                message.priority,
                use_mixnodes,
            )

            if success:
                self.routing_stats["betanet_sent"] += 1
                logger.debug(f"Sent message {message.id[:8]} via Betanet")

            return success

        except Exception as e:
            logger.exception(f"Betanet send failed: {e}")
            return False

    async def _queue_for_store_forward(self, message: DualPathMessage) -> None:
        """Queue message for store-and-forward delivery"""
        # Check queue size limit
        if len(self.offline_queue) >= self.max_queue_size:
            # Remove oldest message
            self.offline_queue.pop(0)
            logger.warning("Store-and-forward queue full - removed oldest message")

        # Add to queue
        self.offline_queue.append((message, time.time()))
        self.routing_stats["store_forward_queued"] += 1

        logger.info(f"Queued message {message.id[:8]} for store-and-forward delivery")

    async def _handle_bitchat_message(self, bitchat_msg) -> None:
        """Handle incoming BitChat message"""
        try:
            # Convert to unified format
            message = DualPathMessage(
                id=bitchat_msg.id,
                sender=bitchat_msg.sender,
                recipient=bitchat_msg.recipient or self.node_id,
                payload=bitchat_msg.payload,
                priority=bitchat_msg.priority,
            )

            # Process message
            await self._process_received_message(message, "bitchat")

        except Exception as e:
            logger.exception(f"Error handling BitChat message: {e}")

    async def _handle_betanet_message(self, betanet_msg) -> None:
        """Handle incoming Betanet message"""
        try:
            # Convert to unified format
            message = DualPathMessage(
                id=betanet_msg.id,
                sender=betanet_msg.sender,
                recipient=betanet_msg.recipient,
                payload=betanet_msg.payload,
                priority=betanet_msg.priority,
                content_type=betanet_msg.content_type,
                privacy_required=betanet_msg.encryption_layers > 0,
            )

            # Process message
            await self._process_received_message(message, "betanet")

        except Exception as e:
            logger.exception(f"Error handling Betanet message: {e}")

    async def _process_received_message(
        self, message: DualPathMessage, source_protocol: str
    ) -> None:
        """Process received message from any protocol"""
        logger.info(f"Received message {message.id[:8]} via {source_protocol}")

        # Call registered handlers
        for handler_name, handler in self.message_handlers.items():
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message, source_protocol)
                else:
                    handler(message, source_protocol)
            except Exception as e:
                logger.exception(f"Error in message handler {handler_name}: {e}")

    async def _maintenance_loop(self) -> None:
        """Background maintenance and optimization"""
        while self.is_running:
            try:
                await self._process_store_forward_queue()
                await self._sync_peer_information()
                await self._cleanup_expired_data()

                # Maintenance every 60 seconds
                await asyncio.sleep(60)

            except Exception as e:
                logger.exception(f"Maintenance loop error: {e}")
                await asyncio.sleep(30)

    async def _process_store_forward_queue(self) -> None:
        """Process queued store-and-forward messages"""
        if not self.offline_queue:
            return

        current_time = time.time()
        delivered = 0
        failed = 0

        # Try to deliver queued messages
        remaining_queue = []

        for message, queued_time in self.offline_queue:
            # Check if message expired (24 hour TTL)
            if current_time - queued_time > 86400:
                failed += 1
                continue

            # Try to deliver message
            if message.recipient:
                # Check if recipient is now reachable
                can_deliver = False

                if (
                    self.enable_bitchat
                    and self.bitchat
                    and self.bitchat.is_peer_reachable(message.recipient)
                ):
                    can_deliver = True

                elif (
                    self.enable_betanet
                    and self.betanet
                    and message.recipient in self.betanet.discovered_peers
                ):
                    can_deliver = True

                if can_deliver:
                    # Try delivery
                    success = await self.send_message(
                        message.recipient,
                        message.payload,
                        message.priority,
                        message.privacy_required,
                        message.deadline,
                    )

                    if success:
                        delivered += 1
                    else:
                        # Re-queue for later
                        remaining_queue.append((message, queued_time))
                else:
                    # Still not reachable - keep in queue
                    remaining_queue.append((message, queued_time))
            else:
                # Broadcast message - try delivery
                sent_count = await self.broadcast_message(
                    message.payload, message.priority
                )
                if sent_count > 0:
                    delivered += 1
                else:
                    remaining_queue.append((message, queued_time))

        # Update queue
        self.offline_queue = remaining_queue

        if delivered > 0 or failed > 0:
            logger.info(
                f"Store-and-forward: delivered {delivered}, expired {failed}, "
                f"queued {len(self.offline_queue)}"
            )

    async def _sync_peer_information(self) -> None:
        """Sync peer information between Navigator and transports"""
        if not self.enable_navigator or not self.navigator:
            return

        # Update Navigator with BitChat peers
        if self.enable_bitchat and self.bitchat:
            for peer_addr, peer in self.bitchat.discovered_peers.items():
                peer_info = self.navigator.PeerInfo(
                    peer_id=peer.device_id,
                    protocols={"bitchat"},
                    hop_distance=peer.hop_distance,
                    last_seen=peer.last_seen,
                    bluetooth_rssi=peer.signal_strength,
                )
                self.navigator.update_peer_info(peer.device_id, peer_info)

        # Update Navigator with Betanet peers
        if self.enable_betanet and self.betanet:
            for peer_id, peer in self.betanet.discovered_peers.items():
                peer_info = self.navigator.PeerInfo(
                    peer_id=peer_id,
                    protocols=set(peer.protocols),
                    hop_distance=1,  # Betanet peers are generally 1 hop
                    last_seen=peer.last_seen,
                    avg_latency_ms=peer.latency_ms,
                    reliability=peer.reliability_score,
                    bandwidth_mbps=peer.bandwidth_mbps,
                )
                self.navigator.update_peer_info(peer_id, peer_info)

    async def _cleanup_expired_data(self) -> None:
        """Clean up expired cached data"""
        if self.navigator:
            self.navigator.cleanup_cache()

    def register_message_handler(self, handler_name: str, handler) -> None:
        """Register handler for incoming messages"""
        self.message_handlers[handler_name] = handler
        logger.debug(f"Registered message handler: {handler_name}")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive transport status"""
        status = {
            "node_id": self.node_id,
            "is_running": self.is_running,
            "transports": {
                "bitchat_enabled": self.enable_bitchat,
                "betanet_enabled": self.enable_betanet,
                "navigator_enabled": self.enable_navigator,
            },
            "statistics": self.routing_stats.copy(),
            "store_forward_queue": len(self.offline_queue),
        }

        # Add transport-specific status
        if self.bitchat:
            status["bitchat_status"] = self.bitchat.get_status()

        if self.betanet:
            status["betanet_status"] = self.betanet.get_status()

        if self.navigator:
            status["navigator_status"] = self.navigator.get_status()

        return status

    def get_reachable_peers(self) -> Dict[str, List[str]]:
        """Get list of reachable peers by transport"""
        peers = {"bitchat": [], "betanet": [], "both": []}

        bitchat_peers = set()
        betanet_peers = set()

        if self.bitchat:
            bitchat_peers = set(
                peer.device_id for peer in self.bitchat.discovered_peers.values()
            )
            peers["bitchat"] = list(bitchat_peers)

        if self.betanet:
            betanet_peers = set(self.betanet.discovered_peers.keys())
            peers["betanet"] = list(betanet_peers)

        # Peers reachable via both transports
        peers["both"] = list(bitchat_peers & betanet_peers)

        return peers
