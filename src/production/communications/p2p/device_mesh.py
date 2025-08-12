"""Device Mesh Implementation for Mobile P2P Networks."""

import asyncio
import logging
import platform
import socket
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import psutil

from .p2p_node import MessageType, P2PMessage, P2PNode

logger = logging.getLogger(__name__)


class ConnectionType(Enum):
    """Types of network connections."""

    BLUETOOTH = "bluetooth"
    WIFI_DIRECT = "wifi_direct"
    TCP_IP = "tcp_ip"
    USB_TETHER = "usb_tether"


class MeshProtocol(Enum):
    """Mesh networking protocols."""

    FLOODING = "flooding"
    TREE_ROUTING = "tree_routing"
    OPTIMIZED_LINK_STATE = "optimized_link_state"
    GOSSIP = "gossip"


@dataclass
class DeviceCapability:
    """Device capability information."""

    device_type: str  # phone, tablet, laptop, desktop
    os_type: str  # android, ios, windows, linux, macos
    battery_level: float | None = None  # 0.0 to 1.0
    available_memory: int | None = None  # bytes
    cpu_cores: int | None = None
    network_interfaces: list[str] = field(default_factory=list)
    supports_bluetooth: bool = False
    supports_wifi_direct: bool = False
    max_concurrent_connections: int = 10


@dataclass
class MeshRoute:
    """Routing information for mesh network."""

    destination: str
    next_hop: str
    hop_count: int
    cost: float
    last_updated: float = field(default_factory=time.time)
    reliable: bool = True


@dataclass
class NetworkMetrics:
    """Network performance metrics."""

    latency_ms: float
    bandwidth_kbps: float
    packet_loss: float
    jitter_ms: float
    signal_strength: float | None = None  # For wireless connections
    last_updated: float = field(default_factory=time.time)


class DeviceMesh:
    """Advanced mesh networking for mobile devices with multiple protocols."""

    def __init__(
        self,
        node: P2PNode,
        protocol: MeshProtocol = MeshProtocol.OPTIMIZED_LINK_STATE,
        max_hops: int = 5,
        routing_update_interval: float = 30.0,
    ) -> None:
        self.node = node
        self.protocol = protocol
        self.max_hops = max_hops
        self.routing_update_interval = routing_update_interval

        # Device information
        self.device_capabilities = self._detect_device_capabilities()

        # Mesh networking
        self.routing_table: dict[str, MeshRoute] = {}
        self.network_topology: dict[
            str, set[str]
        ] = {}  # node_id -> connected neighbors
        self.connection_types: dict[str, ConnectionType] = {}
        self.network_metrics: dict[str, NetworkMetrics] = {}

        # Protocol-specific data
        self.flooding_cache: set[str] = set()  # Message IDs for flood prevention
        self.tree_parent: str | None = None
        self.tree_children: set[str] = set()
        self.link_state_db: dict[
            str, dict[str, float]
        ] = {}  # node -> {neighbor -> cost}

        # Background tasks
        self.routing_task: asyncio.Task | None = None
        self.metrics_task: asyncio.Task | None = None
        self.discovery_task: asyncio.Task | None = None

        # Store and forward for offline nodes
        self.message_store: dict[str, list[P2PMessage]] = {}  # destination -> messages
        self.offline_detection_threshold = 60.0  # seconds

        # Register mesh-specific handlers
        self._register_mesh_handlers()

    def _detect_device_capabilities(self) -> DeviceCapability:
        """Detect current device capabilities."""
        system = platform.system().lower()
        device_type = "desktop"

        # Detect device type
        if system == "android":
            device_type = "phone"  # Could be tablet
        elif system == "darwin":
            if platform.machine().startswith("iP"):
                device_type = "phone"  # iPhone/iPad
            else:
                device_type = "desktop"  # macOS
        elif system == "linux":
            # Could be phone (Android), embedded, or desktop
            device_type = "desktop"  # Default assumption
        elif system == "windows":
            device_type = "desktop"

        # Get system information
        memory = psutil.virtual_memory()

        # Detect network interfaces
        interfaces = []
        try:
            for interface, addrs in psutil.net_if_addrs().items():
                if any(addr.family == socket.AF_INET for addr in addrs):
                    interfaces.append(interface)
        except Exception as e:
            logger.warning(f"Failed to detect network interfaces: {e}")

        return DeviceCapability(
            device_type=device_type,
            os_type=system,
            available_memory=memory.available,
            cpu_cores=psutil.cpu_count(),
            network_interfaces=interfaces,
            supports_bluetooth=self._check_bluetooth_support(),
            supports_wifi_direct=self._check_wifi_direct_support(),
        )

    def _check_bluetooth_support(self) -> bool:
        """Check if device supports Bluetooth."""
        # Simplified check - in real implementation would use platform-specific APIs
        system = platform.system().lower()
        return system in ["android", "darwin", "linux"]  # Most mobile/desktop systems

    def _check_wifi_direct_support(self) -> bool:
        """Check if device supports WiFi Direct."""
        # Simplified check - would need platform-specific APIs
        system = platform.system().lower()
        return system in ["android"]  # Mainly Android devices

    def _register_mesh_handlers(self) -> None:
        """Register mesh-specific message handlers."""
        self.node.register_handler(MessageType.DATA, self._handle_mesh_data)

    async def start_mesh(self) -> None:
        """Start mesh networking."""
        logger.info(f"Starting mesh network with protocol {self.protocol.value}")

        # Initialize routing table with direct neighbors
        await self._initialize_routing_table()

        # Start background tasks
        self.routing_task = asyncio.create_task(self._routing_update_loop())
        self.metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self.discovery_task = asyncio.create_task(self._mesh_discovery_loop())

        logger.info("Mesh networking started")

    async def stop_mesh(self) -> None:
        """Stop mesh networking."""
        logger.info("Stopping mesh network")

        # Cancel background tasks
        for task in [self.routing_task, self.metrics_task, self.discovery_task]:
            if task:
                task.cancel()

        # Clear routing data
        self.routing_table.clear()
        self.network_topology.clear()

        logger.info("Mesh networking stopped")

    async def send_mesh_message(
        self,
        destination: str,
        data: dict[str, Any],
        store_and_forward: bool = True,
        max_hops: int | None = None,
    ) -> bool:
        """Send a message through the mesh network."""
        max_hops = max_hops or self.max_hops

        # Check if destination is directly connected
        if destination in self.node.peers:
            return await self.node.send_message(destination, MessageType.DATA, data)

        # Find route through mesh
        route = self._find_route(destination)

        if not route:
            # Store message for later delivery if enabled
            if store_and_forward:
                self._store_message_for_later(destination, data)
                logger.info(f"Stored message for offline destination {destination}")
                return True
            logger.warning(f"No route to destination {destination}")
            return False

        # Create mesh message with routing info
        mesh_payload = {
            "mesh_data": data,
            "destination": destination,
            "source": self.node.node_id,
            "hops_remaining": max_hops - 1,
            "route_path": [self.node.node_id],
            "message_id": str(uuid.uuid4()),
        }

        # Send to next hop
        next_hop = route.next_hop
        success = await self.node.send_message(next_hop, MessageType.DATA, mesh_payload)

        if success:
            logger.debug(f"Sent mesh message to {destination} via {next_hop}")
        # Try store and forward if direct routing failed
        elif store_and_forward:
            self._store_message_for_later(destination, data)

        return success

    async def broadcast_mesh(
        self,
        data: dict[str, Any],
        max_hops: int | None = None,
        exclude_nodes: set[str] | None = None,
    ) -> int:
        """Broadcast a message through the mesh network."""
        max_hops = max_hops or self.max_hops
        exclude_nodes = exclude_nodes or set()

        message_id = str(uuid.uuid4())

        if self.protocol == MeshProtocol.FLOODING:
            return await self._flood_broadcast(
                data, message_id, max_hops, exclude_nodes
            )
        if self.protocol == MeshProtocol.TREE_ROUTING:
            return await self._tree_broadcast(data, message_id, max_hops, exclude_nodes)
        # Default to direct broadcast to neighbors
        successful_sends = 0
        for peer_id in self.node.peers:
            if peer_id not in exclude_nodes:
                if await self.node.send_message(peer_id, MessageType.DATA, data):
                    successful_sends += 1
        return successful_sends

    def add_mesh_peer(
        self,
        peer_id: str,
        connection_type: ConnectionType,
        metrics: NetworkMetrics | None = None,
    ) -> None:
        """Add peer with mesh-specific information."""
        self.connection_types[peer_id] = connection_type

        if metrics:
            self.network_metrics[peer_id] = metrics

        # Update topology
        if self.node.node_id not in self.network_topology:
            self.network_topology[self.node.node_id] = set()
        self.network_topology[self.node.node_id].add(peer_id)

        # Update routing table
        self._update_routing_table()

    def get_mesh_status(self) -> dict[str, Any]:
        """Get comprehensive mesh network status."""
        self.node.get_connected_peers()

        return {
            "protocol": self.protocol.value,
            "device_capabilities": {
                "device_type": self.device_capabilities.device_type,
                "os_type": self.device_capabilities.os_type,
                "battery_level": self.device_capabilities.battery_level,
                "available_memory": self.device_capabilities.available_memory,
                "cpu_cores": self.device_capabilities.cpu_cores,
                "supports_bluetooth": self.device_capabilities.supports_bluetooth,
                "supports_wifi_direct": self.device_capabilities.supports_wifi_direct,
            },
            "network_topology": {
                node: list(neighbors)
                for node, neighbors in self.network_topology.items()
            },
            "routing_table_size": len(self.routing_table),
            "reachable_nodes": len(self.routing_table),
            "connection_types": {
                peer: conn_type.value
                for peer, conn_type in self.connection_types.items()
            },
            "stored_messages": sum(len(msgs) for msgs in self.message_store.values()),
            "metrics": {
                "avg_latency": self._calculate_average_latency(),
                "avg_bandwidth": self._calculate_average_bandwidth(),
                "network_diameter": self._calculate_network_diameter(),
            },
        }

    async def optimize_routing(self) -> None:
        """Optimize routing tables and connections."""
        logger.info("Optimizing mesh routing")

        # Update metrics for all connections
        await self._measure_all_connections()

        # Recalculate optimal routes
        self._update_routing_table()

        # Clean up stale routes
        current_time = time.time()
        stale_routes = [
            dest
            for dest, route in self.routing_table.items()
            if current_time - route.last_updated > self.routing_update_interval * 3
        ]

        for dest in stale_routes:
            del self.routing_table[dest]

        logger.info(
            f"Routing optimization complete, {len(stale_routes)} stale routes removed"
        )

    def _find_route(self, destination: str) -> MeshRoute | None:
        """Find best route to destination."""
        return self.routing_table.get(destination)

    def _update_routing_table(self) -> None:
        """Update routing table based on current protocol."""
        if self.protocol == MeshProtocol.OPTIMIZED_LINK_STATE:
            self._calculate_link_state_routes()
        elif self.protocol == MeshProtocol.TREE_ROUTING:
            self._calculate_tree_routes()
        else:
            # Simple direct routing
            for peer_id in self.node.peers:
                self.routing_table[peer_id] = MeshRoute(
                    destination=peer_id,
                    next_hop=peer_id,
                    hop_count=1,
                    cost=1.0,
                )

    def _calculate_link_state_routes(self) -> None:
        """Calculate routes using Dijkstra's algorithm on link state data."""
        # Simplified Dijkstra implementation
        distances = {self.node.node_id: 0.0}
        previous = {}
        unvisited = set(self.link_state_db.keys())
        unvisited.add(self.node.node_id)

        while unvisited:
            # Find node with minimum distance
            current = min(unvisited, key=lambda n: distances.get(n, float("inf")))

            if distances.get(current, float("inf")) == float("inf"):
                break  # No more reachable nodes

            unvisited.remove(current)

            # Update distances to neighbors
            neighbors = self.link_state_db.get(current, {})
            for neighbor, cost in neighbors.items():
                if neighbor in unvisited:
                    alt_distance = distances[current] + cost
                    if alt_distance < distances.get(neighbor, float("inf")):
                        distances[neighbor] = alt_distance
                        previous[neighbor] = current

        # Build routing table from shortest paths
        for destination, distance in distances.items():
            if destination != self.node.node_id and distance != float("inf"):
                # Find next hop by backtracking path
                next_hop = destination
                while previous.get(next_hop) != self.node.node_id:
                    next_hop = previous.get(next_hop)
                    if not next_hop:
                        break

                if next_hop:
                    hop_count = 0
                    temp = destination
                    while temp != self.node.node_id:
                        temp = previous.get(temp)
                        hop_count += 1
                        if hop_count > self.max_hops:
                            break

                    if hop_count <= self.max_hops:
                        self.routing_table[destination] = MeshRoute(
                            destination=destination,
                            next_hop=next_hop,
                            hop_count=hop_count,
                            cost=distance,
                        )

    def _calculate_tree_routes(self) -> None:
        """Calculate routes for tree-based routing."""
        # In tree routing, routes go either up to parent or down to children
        if self.tree_parent:
            # Route to parent and anything beyond parent
            for dest in self.link_state_db:
                if dest not in self.tree_children and dest != self.node.node_id:
                    self.routing_table[dest] = MeshRoute(
                        destination=dest,
                        next_hop=self.tree_parent,
                        hop_count=2,  # Estimated
                        cost=2.0,
                    )

        # Direct routes to children
        for child in self.tree_children:
            self.routing_table[child] = MeshRoute(
                destination=child,
                next_hop=child,
                hop_count=1,
                cost=1.0,
            )

    def _store_message_for_later(self, destination: str, data: dict[str, Any]) -> None:
        """Store message for later delivery when destination comes online."""
        if destination not in self.message_store:
            self.message_store[destination] = []

        message = P2PMessage(
            message_type=MessageType.DATA,
            sender_id=self.node.node_id,
            receiver_id=destination,
            payload=data,
        )

        self.message_store[destination].append(message)

        # Limit stored messages per destination
        max_stored = 100
        if len(self.message_store[destination]) > max_stored:
            self.message_store[destination] = self.message_store[destination][
                -max_stored:
            ]

    async def _deliver_stored_messages(self, peer_id: str) -> None:
        """Deliver stored messages when peer comes online."""
        if peer_id in self.message_store:
            messages = self.message_store.pop(peer_id)

            for message in messages:
                await self.node.send_message(
                    peer_id, message.message_type, message.payload
                )

            logger.info(f"Delivered {len(messages)} stored messages to {peer_id}")

    async def _handle_mesh_data(
        self,
        message: P2PMessage,
        writer: asyncio.StreamWriter | None = None,
    ) -> None:
        """Handle mesh-routed data messages."""
        payload = message.payload

        # Check if this is a mesh message
        if "destination" in payload and "hops_remaining" in payload:
            await self._route_mesh_message(message)
        else:
            # Regular direct message - handle normally
            pass

    async def _route_mesh_message(self, message: P2PMessage) -> None:
        """Route a mesh message to its destination."""
        payload = message.payload
        destination = payload["destination"]
        hops_remaining = payload["hops_remaining"]
        route_path = payload.get("route_path", [])
        message_id = payload.get("message_id", "")

        # Check for message loops
        if self.node.node_id in route_path:
            logger.warning(f"Message loop detected for {message_id}")
            return

        # Check hop limit
        if hops_remaining <= 0:
            logger.warning(f"Message {message_id} exceeded hop limit")
            return

        # Add ourselves to path
        route_path.append(self.node.node_id)

        # Check if we are the destination
        if destination == self.node.node_id:
            # Deliver message locally
            payload.get("mesh_data", {})
            logger.info(f"Received mesh message from {payload.get('source')}")
            return

        # Find next hop
        route = self._find_route(destination)
        if not route:
            logger.warning(f"No route to destination {destination}")
            return

        # Forward message
        forwarded_payload = {
            **payload,
            "hops_remaining": hops_remaining - 1,
            "route_path": route_path,
        }

        await self.node.send_message(
            route.next_hop, MessageType.DATA, forwarded_payload
        )

    async def _flood_broadcast(
        self,
        data: dict[str, Any],
        message_id: str,
        max_hops: int,
        exclude_nodes: set[str],
    ) -> int:
        """Broadcast using flooding protocol."""
        if message_id in self.flooding_cache:
            return 0  # Already seen this message

        self.flooding_cache.add(message_id)

        # Clean old message IDs from cache
        if len(self.flooding_cache) > 1000:
            # Remove oldest half
            old_messages = list(self.flooding_cache)[:500]
            for old_id in old_messages:
                self.flooding_cache.discard(old_id)

        # Broadcast to all neighbors except sender
        flood_payload = {
            "flood_data": data,
            "message_id": message_id,
            "hops_remaining": max_hops - 1,
            "sender": self.node.node_id,
        }

        successful_sends = 0
        for peer_id in self.node.peers:
            if peer_id not in exclude_nodes:
                if await self.node.send_message(
                    peer_id, MessageType.DATA, flood_payload
                ):
                    successful_sends += 1

        return successful_sends

    async def _tree_broadcast(
        self,
        data: dict[str, Any],
        message_id: str,
        max_hops: int,
        exclude_nodes: set[str],
    ) -> int:
        """Broadcast using tree routing."""
        tree_payload = {
            "tree_data": data,
            "message_id": message_id,
            "sender": self.node.node_id,
        }

        successful_sends = 0

        # Send to parent if not the source
        if self.tree_parent and self.tree_parent not in exclude_nodes:
            if await self.node.send_message(
                self.tree_parent, MessageType.DATA, tree_payload
            ):
                successful_sends += 1

        # Send to all children
        for child in self.tree_children:
            if child not in exclude_nodes:
                if await self.node.send_message(child, MessageType.DATA, tree_payload):
                    successful_sends += 1

        return successful_sends

    async def _initialize_routing_table(self) -> None:
        """Initialize routing table with direct neighbors."""
        for peer_id in self.node.peers:
            self.routing_table[peer_id] = MeshRoute(
                destination=peer_id,
                next_hop=peer_id,
                hop_count=1,
                cost=1.0,
            )

    async def _routing_update_loop(self) -> None:
        """Background task for routing updates."""
        while True:
            try:
                await self.optimize_routing()
                await asyncio.sleep(self.routing_update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in routing update loop: {e}")
                await asyncio.sleep(10)

    async def _metrics_collection_loop(self) -> None:
        """Background task for collecting network metrics."""
        while True:
            try:
                await self._measure_all_connections()
                await asyncio.sleep(30.0)  # Measure every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in metrics collection: {e}")
                await asyncio.sleep(10)

    async def _mesh_discovery_loop(self) -> None:
        """Background task for mesh-specific peer discovery."""
        while True:
            try:
                # Check for stored messages to deliver
                for peer_id in list(self.message_store.keys()):
                    if peer_id in self.node.peers:
                        await self._deliver_stored_messages(peer_id)

                await asyncio.sleep(60.0)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in mesh discovery: {e}")
                await asyncio.sleep(10)

    async def _measure_all_connections(self) -> None:
        """Measure performance metrics for all connections."""
        for peer_id in self.node.peers:
            try:
                # Simple latency measurement
                start_time = time.time()
                success = await self.node.send_message(
                    peer_id, MessageType.HEARTBEAT, {"ping": True}
                )

                if success:
                    latency = (time.time() - start_time) * 1000  # Convert to ms

                    # Update metrics
                    if peer_id not in self.network_metrics:
                        self.network_metrics[peer_id] = NetworkMetrics(
                            latency_ms=latency,
                            bandwidth_kbps=1000.0,  # Default estimate
                            packet_loss=0.0,
                            jitter_ms=0.0,
                        )
                    else:
                        # Update with exponential moving average
                        alpha = 0.3
                        current = self.network_metrics[peer_id]
                        current.latency_ms = (
                            alpha * latency + (1 - alpha) * current.latency_ms
                        )
                        current.last_updated = time.time()

            except Exception as e:
                logger.debug(f"Failed to measure connection to {peer_id}: {e}")

    def _calculate_average_latency(self) -> float:
        """Calculate average network latency."""
        if not self.network_metrics:
            return 0.0

        total_latency = sum(m.latency_ms for m in self.network_metrics.values())
        return total_latency / len(self.network_metrics)

    def _calculate_average_bandwidth(self) -> float:
        """Calculate average network bandwidth."""
        if not self.network_metrics:
            return 0.0

        total_bandwidth = sum(m.bandwidth_kbps for m in self.network_metrics.values())
        return total_bandwidth / len(self.network_metrics)

    def _calculate_network_diameter(self) -> int:
        """Calculate network diameter (maximum shortest path)."""
        if not self.routing_table:
            return 0

        return max(route.hop_count for route in self.routing_table.values())
