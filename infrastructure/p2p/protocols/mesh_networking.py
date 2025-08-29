"""Mesh Networking Protocols Implementation.

Advanced mesh networking protocols including:
- Gossip protocols for efficient message propagation
- Routing algorithms (distance vector, link state)
- Network topology management and optimization
- Load balancing and congestion control
- Fault tolerance and self-healing mechanisms

This module provides the protocol-level functionality that the LibP2P
mesh network uses for advanced networking features.
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import logging
import math
import time
from typing import Any

logger = logging.getLogger(__name__)


class TopologyType(Enum):
    """Network topology types."""

    MESH = "mesh"  # Full mesh - all nodes connected
    RING = "ring"  # Ring topology
    HIERARCHICAL = "hierarchical"  # Tree-like hierarchy
    SMALL_WORLD = "small_world"  # Small world network
    ADAPTIVE = "adaptive"  # Adaptive topology based on conditions


class RoutingAlgorithm(Enum):
    """Routing algorithm types."""

    FLOODING = "flooding"  # Simple flooding
    DISTANCE_VECTOR = "distance_vector"  # Distance vector routing
    LINK_STATE = "link_state"  # Link state routing
    GOSSIP = "gossip"  # Gossip-based routing
    EPIDEMIC = "epidemic"  # Epidemic routing


@dataclass
class NetworkMetrics:
    """Network performance metrics."""

    # Connectivity
    peer_count: int = 0
    connection_density: float = 0.0  # connections / max_possible
    average_path_length: float = 0.0
    clustering_coefficient: float = 0.0

    # Performance
    message_delivery_rate: float = 0.0
    average_latency_ms: float = 0.0
    bandwidth_utilization: float = 0.0

    # Resilience
    fault_tolerance: float = 0.0  # % of nodes that can fail
    network_fragmentation: float = 0.0
    recovery_time_ms: float = 0.0

    # Load distribution
    load_balance_index: float = 0.0  # Lower = better balance
    hotspot_count: int = 0
    congestion_score: float = 0.0


@dataclass
class PeerMetrics:
    """Per-peer performance metrics."""

    peer_id: str

    # Connection quality
    latency_ms: float = 0.0
    packet_loss_rate: float = 0.0
    bandwidth_mbps: float = 0.0
    connection_uptime: float = 0.0

    # Routing metrics
    hop_count: int = 1
    route_stability: float = 1.0  # How often route changes
    forwarding_success_rate: float = 1.0

    # Load metrics
    message_throughput: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0

    # Trust metrics
    trust_score: float = 1.0
    malicious_behavior_count: int = 0
    last_seen: float = field(default_factory=time.time)


@dataclass
class RoutingEntry:
    """Entry in routing table."""

    destination: str
    next_hop: str
    hop_count: int
    cost: float
    last_updated: float = field(default_factory=time.time)
    reliability: float = 1.0


class GossipProtocol:
    """Efficient gossip protocol for message propagation."""

    def __init__(
        self,
        node_id: str,
        fanout: int = 3,
        gossip_interval: float = 1.0,
        max_message_age: float = 300.0,
    ):
        self.node_id = node_id
        self.fanout = fanout
        self.gossip_interval = gossip_interval
        self.max_message_age = max_message_age

        # Message tracking
        self.seen_messages: set[str] = set()
        self.message_cache: dict[str, tuple[Any, float]] = {}
        self.peer_metrics: dict[str, PeerMetrics] = {}

        # Gossip state
        self.gossip_peers: set[str] = set()
        self.running = False
        self._gossip_task: asyncio.Task | None = None

        # Statistics
        self.stats = {
            "messages_gossiped": 0,
            "redundant_messages": 0,
            "average_propagation_time": 0.0,
        }

    async def start(self):
        """Start gossip protocol."""
        self.running = True
        self._gossip_task = asyncio.create_task(self._gossip_loop())
        logger.info(f"Gossip protocol started for {self.node_id}")

    async def stop(self):
        """Stop gossip protocol."""
        self.running = False
        if self._gossip_task:
            self._gossip_task.cancel()
            try:
                await self._gossip_task
            except asyncio.CancelledError:
                pass

    def add_peer(self, peer_id: str, metrics: PeerMetrics | None = None):
        """Add a peer to gossip with."""
        self.gossip_peers.add(peer_id)
        if metrics:
            self.peer_metrics[peer_id] = metrics
        else:
            self.peer_metrics[peer_id] = PeerMetrics(peer_id=peer_id)

    def remove_peer(self, peer_id: str):
        """Remove a peer from gossip."""
        self.gossip_peers.discard(peer_id)
        self.peer_metrics.pop(peer_id, None)

    async def gossip_message(self, message_id: str, message_data: Any, exclude_peers: set[str] = None) -> int:
        """Gossip a message to selected peers."""
        if message_id in self.seen_messages:
            self.stats["redundant_messages"] += 1
            return 0

        # Mark as seen and cache
        self.seen_messages.add(message_id)
        self.message_cache[message_id] = (message_data, time.time())

        # Select peers to gossip to
        exclude_peers = exclude_peers or set()
        available_peers = self.gossip_peers - exclude_peers - {self.node_id}

        if not available_peers:
            return 0

        # Select best peers based on metrics
        selected_peers = self._select_gossip_targets(available_peers)

        # Send to selected peers
        success_count = 0
        for peer_id in selected_peers:
            try:
                await self._send_gossip_message(peer_id, message_id, message_data)
                success_count += 1
            except Exception as e:
                logger.warning(f"Failed to gossip to {peer_id}: {e}")

        self.stats["messages_gossiped"] += 1
        return success_count

    def _select_gossip_targets(self, available_peers: set[str]) -> list[str]:
        """Select peers for gossiping based on metrics and strategy."""
        peers_list = list(available_peers)

        if len(peers_list) <= self.fanout:
            return peers_list

        # Score peers based on multiple factors
        peer_scores = []
        for peer_id in peers_list:
            metrics = self.peer_metrics.get(peer_id, PeerMetrics(peer_id=peer_id))

            # Calculate composite score
            score = (
                (1.0 - metrics.latency_ms / 1000.0) * 0.3
                + (1.0 - metrics.packet_loss_rate) * 0.2  # Lower latency better
                + metrics.trust_score * 0.2  # Lower loss better
                + metrics.forwarding_success_rate * 0.2  # Higher trust better
                + metrics.connection_uptime * 0.1  # Higher success better  # Higher uptime better
            )

            peer_scores.append((peer_id, score))

        # Sort by score and select top peers
        peer_scores.sort(key=lambda x: x[1], reverse=True)
        return [peer_id for peer_id, _ in peer_scores[: self.fanout]]

    async def _send_gossip_message(self, peer_id: str, message_id: str, message_data: Any):
        """Send gossip message to specific peer."""
        # This would be implemented by the actual network layer
        # For now, just simulate
        await asyncio.sleep(0.01)  # Simulate network delay

    async def _gossip_loop(self):
        """Main gossip loop for periodic maintenance."""
        while self.running:
            try:
                # Clean up old messages
                current_time = time.time()
                old_messages = [
                    msg_id
                    for msg_id, (_, timestamp) in self.message_cache.items()
                    if current_time - timestamp > self.max_message_age
                ]

                for msg_id in old_messages:
                    self.seen_messages.discard(msg_id)
                    del self.message_cache[msg_id]

                await asyncio.sleep(self.gossip_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Gossip loop error: {e}")
                await asyncio.sleep(1.0)


class DistanceVectorRouting:
    """Distance vector routing protocol implementation."""

    def __init__(self, node_id: str, max_cost: float = float("inf")):
        self.node_id = node_id
        self.max_cost = max_cost

        # Routing table: destination -> RoutingEntry
        self.routing_table: dict[str, RoutingEntry] = {}

        # Distance vectors from neighbors
        self.neighbor_vectors: dict[str, dict[str, float]] = {}

        # Direct neighbors with costs
        self.neighbors: dict[str, float] = {}

        # Update tracking
        self.sequence_number = 0
        self.last_update = time.time()

        # Configuration
        self.update_interval = 30.0  # seconds
        self.route_timeout = 180.0  # seconds

        self.running = False
        self._update_task: asyncio.Task | None = None

    async def start(self):
        """Start distance vector routing."""
        self.running = True
        self._update_task = asyncio.create_task(self._update_loop())

        # Initialize route to self
        self.routing_table[self.node_id] = RoutingEntry(
            destination=self.node_id,
            next_hop=self.node_id,
            hop_count=0,
            cost=0.0,
        )

        logger.info(f"Distance vector routing started for {self.node_id}")

    async def stop(self):
        """Stop distance vector routing."""
        self.running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

    def add_neighbor(self, neighbor_id: str, cost: float = 1.0):
        """Add or update a neighbor."""
        self.neighbors[neighbor_id] = cost

        # Add/update direct route
        self.routing_table[neighbor_id] = RoutingEntry(
            destination=neighbor_id,
            next_hop=neighbor_id,
            hop_count=1,
            cost=cost,
        )

        self._trigger_update()

    def remove_neighbor(self, neighbor_id: str):
        """Remove a neighbor."""
        self.neighbors.pop(neighbor_id, None)
        self.neighbor_vectors.pop(neighbor_id, None)

        # Remove routes through this neighbor
        to_remove = []
        for dest, entry in self.routing_table.items():
            if entry.next_hop == neighbor_id:
                to_remove.append(dest)

        for dest in to_remove:
            del self.routing_table[dest]

        self._trigger_update()

    def process_distance_vector(self, sender_id: str, distance_vector: dict[str, float]):
        """Process distance vector from a neighbor."""
        if sender_id not in self.neighbors:
            return  # Ignore updates from non-neighbors

        self.neighbor_vectors[sender_id] = distance_vector
        neighbor_cost = self.neighbors[sender_id]

        # Update routing table based on distance vector
        updated = False

        for destination, distance in distance_vector.items():
            if destination == self.node_id:
                continue  # Skip routes to self

            total_cost = neighbor_cost + distance

            if destination not in self.routing_table:
                # New route
                self.routing_table[destination] = RoutingEntry(
                    destination=destination,
                    next_hop=sender_id,
                    hop_count=int(distance) + 1,
                    cost=total_cost,
                )
                updated = True
            else:
                # Check if this is a better route
                current_entry = self.routing_table[destination]

                if total_cost < current_entry.cost or current_entry.next_hop == sender_id:  # Update existing route
                    current_entry.next_hop = sender_id
                    current_entry.hop_count = int(distance) + 1
                    current_entry.cost = total_cost
                    current_entry.last_updated = time.time()
                    updated = True

        if updated:
            self._trigger_update()

    def get_distance_vector(self) -> dict[str, float]:
        """Get current distance vector to share with neighbors."""
        return {dest: entry.cost for dest, entry in self.routing_table.items()}

    def get_next_hop(self, destination: str) -> str | None:
        """Get next hop for a destination."""
        entry = self.routing_table.get(destination)
        return entry.next_hop if entry else None

    def _trigger_update(self):
        """Trigger routing update."""
        self.sequence_number += 1
        self.last_update = time.time()

    async def _update_loop(self):
        """Periodic routing table maintenance."""
        while self.running:
            try:
                current_time = time.time()

                # Remove expired routes
                expired_routes = []
                for dest, entry in self.routing_table.items():
                    if dest != self.node_id and current_time - entry.last_updated > self.route_timeout:
                        expired_routes.append(dest)

                for dest in expired_routes:
                    del self.routing_table[dest]

                # Send distance vector to neighbors
                distance_vector = self.get_distance_vector()
                for neighbor_id in self.neighbors.keys():
                    await self._send_distance_vector(neighbor_id, distance_vector)

                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Routing update error: {e}")
                await asyncio.sleep(5.0)

    async def _send_distance_vector(self, neighbor_id: str, distance_vector: dict[str, float]):
        """Send distance vector to neighbor."""
        # This would be implemented by the actual network layer
        await asyncio.sleep(0.01)


class TopologyManager:
    """Manages network topology and optimization."""

    def __init__(self, node_id: str, target_topology: TopologyType = TopologyType.ADAPTIVE):
        self.node_id = node_id
        self.target_topology = target_topology

        # Network state
        self.peers: dict[str, PeerMetrics] = {}
        self.connections: dict[str, set[str]] = defaultdict(set)
        self.network_metrics = NetworkMetrics()

        # Topology optimization
        self.optimization_interval = 60.0  # seconds
        self.max_connections_per_node = 10
        self.min_connections_per_node = 3

        self.running = False
        self._optimization_task: asyncio.Task | None = None

    async def start(self):
        """Start topology management."""
        self.running = True
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        logger.info(f"Topology manager started for {self.node_id}")

    async def stop(self):
        """Stop topology management."""
        self.running = False
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass

    def add_peer(self, peer_id: str, metrics: PeerMetrics | None = None):
        """Add peer to topology."""
        if metrics:
            self.peers[peer_id] = metrics
        else:
            self.peers[peer_id] = PeerMetrics(peer_id=peer_id)

        self._update_network_metrics()

    def remove_peer(self, peer_id: str):
        """Remove peer from topology."""
        self.peers.pop(peer_id, None)

        # Remove connections
        for peer_connections in self.connections.values():
            peer_connections.discard(peer_id)
        del self.connections[peer_id]

        self._update_network_metrics()

    def add_connection(self, peer1: str, peer2: str):
        """Add connection between peers."""
        self.connections[peer1].add(peer2)
        self.connections[peer2].add(peer1)
        self._update_network_metrics()

    def remove_connection(self, peer1: str, peer2: str):
        """Remove connection between peers."""
        self.connections[peer1].discard(peer2)
        self.connections[peer2].discard(peer1)
        self._update_network_metrics()

    def _update_network_metrics(self):
        """Update network performance metrics."""
        peer_count = len(self.peers)
        if peer_count == 0:
            return

        # Calculate connection density
        total_connections = sum(len(conns) for conns in self.connections.values()) // 2
        max_possible = peer_count * (peer_count - 1) // 2
        connection_density = total_connections / max_possible if max_possible > 0 else 0.0

        # Calculate average path length (simplified)
        avg_path_length = self._calculate_average_path_length()

        # Calculate clustering coefficient
        clustering = self._calculate_clustering_coefficient()

        # Update metrics
        self.network_metrics.peer_count = peer_count
        self.network_metrics.connection_density = connection_density
        self.network_metrics.average_path_length = avg_path_length
        self.network_metrics.clustering_coefficient = clustering

    def _calculate_average_path_length(self) -> float:
        """Calculate average shortest path length."""
        if len(self.peers) < 2:
            return 0.0

        total_path_length = 0
        path_count = 0

        # Simple BFS for shortest paths
        for start_peer in self.peers.keys():
            visited = {start_peer}
            queue = deque([(start_peer, 0)])

            while queue:
                peer, distance = queue.popleft()

                for neighbor in self.connections[peer]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))
                        total_path_length += distance + 1
                        path_count += 1

        return total_path_length / path_count if path_count > 0 else 0.0

    def _calculate_clustering_coefficient(self) -> float:
        """Calculate network clustering coefficient."""
        if len(self.peers) < 3:
            return 0.0

        total_clustering = 0
        node_count = 0

        for peer in self.peers.keys():
            neighbors = self.connections[peer]
            if len(neighbors) < 2:
                continue

            # Count triangles
            triangles = 0
            for neighbor1 in neighbors:
                for neighbor2 in neighbors:
                    if neighbor1 != neighbor2 and neighbor2 in self.connections[neighbor1]:
                        triangles += 1

            # Clustering coefficient for this node
            possible_triangles = len(neighbors) * (len(neighbors) - 1)
            if possible_triangles > 0:
                clustering = triangles / possible_triangles
                total_clustering += clustering
                node_count += 1

        return total_clustering / node_count if node_count > 0 else 0.0

    async def optimize_topology(self) -> list[tuple[str, str, str]]:
        """Optimize network topology and return suggested changes."""
        changes = []  # List of (action, peer1, peer2) tuples

        if self.target_topology == TopologyType.MESH:
            changes = await self._optimize_for_mesh()
        elif self.target_topology == TopologyType.RING:
            changes = await self._optimize_for_ring()
        elif self.target_topology == TopologyType.HIERARCHICAL:
            changes = await self._optimize_for_hierarchy()
        elif self.target_topology == TopologyType.ADAPTIVE:
            changes = await self._optimize_adaptive()

        return changes

    async def _optimize_for_mesh(self) -> list[tuple[str, str, str]]:
        """Optimize for mesh topology."""
        changes = []

        # Try to connect all peers to all others (within limits)
        for peer1 in self.peers.keys():
            current_connections = len(self.connections[peer1])

            if current_connections < self.max_connections_per_node:
                # Find best peers to connect to
                candidates = []
                for peer2 in self.peers.keys():
                    if (
                        peer1 != peer2
                        and peer2 not in self.connections[peer1]
                        and len(self.connections[peer2]) < self.max_connections_per_node
                    ):
                        # Score based on peer metrics
                        metrics = self.peers[peer2]
                        score = metrics.trust_score * metrics.forwarding_success_rate
                        candidates.append((peer2, score))

                # Sort by score and connect to best candidates
                candidates.sort(key=lambda x: x[1], reverse=True)
                slots_available = self.max_connections_per_node - current_connections

                for peer2, _ in candidates[:slots_available]:
                    changes.append(("connect", peer1, peer2))

        return changes

    async def _optimize_for_ring(self) -> list[tuple[str, str, str]]:
        """Optimize for ring topology."""
        changes = []

        # For ring, each node should have exactly 2 connections
        peers_list = list(self.peers.keys())
        if len(peers_list) < 3:
            return changes

        # Remove excess connections first
        for peer in peers_list:
            connections = list(self.connections[peer])
            if len(connections) > 2:
                # Remove worst connections
                connection_scores = []
                for conn_peer in connections:
                    metrics = self.peers[conn_peer]
                    score = metrics.trust_score * metrics.connection_uptime
                    connection_scores.append((conn_peer, score))

                connection_scores.sort(key=lambda x: x[1])
                for conn_peer, _ in connection_scores[:-2]:
                    changes.append(("disconnect", peer, conn_peer))

        # Add missing connections to form ring
        for i, peer in enumerate(peers_list):
            next_peer = peers_list[(i + 1) % len(peers_list)]
            if next_peer not in self.connections[peer]:
                changes.append(("connect", peer, next_peer))

        return changes

    async def _optimize_for_hierarchy(self) -> list[tuple[str, str, str]]:
        """Optimize for hierarchical topology."""
        changes = []

        # Select root nodes based on metrics
        peer_scores = []
        for peer_id, metrics in self.peers.items():
            score = metrics.trust_score * 0.4 + metrics.forwarding_success_rate * 0.3 + metrics.connection_uptime * 0.3
            peer_scores.append((peer_id, score))

        peer_scores.sort(key=lambda x: x[1], reverse=True)

        # Create hierarchy levels
        root_count = max(1, len(self.peers) // 10)  # 10% as roots
        roots = [peer_id for peer_id, _ in peer_scores[:root_count]]

        # Connect non-root nodes to best root
        for peer_id, _ in peer_scores[root_count:]:
            best_root = None
            best_score = -1

            for root_id in roots:
                if len(self.connections[root_id]) < self.max_connections_per_node:
                    root_metrics = self.peers[root_id]
                    score = root_metrics.trust_score
                    if score > best_score:
                        best_score = score
                        best_root = root_id

            if best_root and best_root not in self.connections[peer_id]:
                changes.append(("connect", peer_id, best_root))

        return changes

    async def _optimize_adaptive(self) -> list[tuple[str, str, str]]:
        """Adaptive topology optimization based on current conditions."""
        changes = []

        # Analyze current network conditions
        avg_latency = sum(peer.latency_ms for peer in self.peers.values()) / len(self.peers) if self.peers else 0
        avg_loss = sum(peer.packet_loss_rate for peer in self.peers.values()) / len(self.peers) if self.peers else 0

        # Choose strategy based on conditions
        if avg_loss > 0.1:  # High packet loss - optimize for redundancy
            changes.extend(await self._optimize_for_mesh())
        elif avg_latency > 500:  # High latency - optimize for efficiency
            changes.extend(await self._optimize_for_hierarchy())
        else:  # Normal conditions - balanced approach
            # Ensure minimum connectivity
            for peer in self.peers.keys():
                if len(self.connections[peer]) < self.min_connections_per_node:
                    # Find best peer to connect to
                    candidates = []
                    for other_peer in self.peers.keys():
                        if peer != other_peer and other_peer not in self.connections[peer]:
                            metrics = self.peers[other_peer]
                            score = metrics.trust_score * (1.0 - metrics.latency_ms / 1000.0)
                            candidates.append((other_peer, score))

                    if candidates:
                        candidates.sort(key=lambda x: x[1], reverse=True)
                        best_peer, _ = candidates[0]
                        changes.append(("connect", peer, best_peer))

        return changes

    async def _optimization_loop(self):
        """Main optimization loop."""
        while self.running:
            try:
                changes = await self.optimize_topology()

                if changes:
                    logger.info(f"Topology optimization suggests {len(changes)} changes")
                    # The actual network layer would implement these changes

                await asyncio.sleep(self.optimization_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Topology optimization error: {e}")
                await asyncio.sleep(10.0)


# Utility functions for network analysis
def calculate_network_efficiency(peer_count: int, avg_path_length: float) -> float:
    """Calculate network efficiency metric."""
    if peer_count <= 1 or avg_path_length <= 0:
        return 0.0

    # Efficiency is inverse of average path length, normalized
    ideal_path_length = math.log(peer_count) / math.log(2)  # Logarithmic ideal
    return ideal_path_length / avg_path_length if avg_path_length > 0 else 0.0


def estimate_fault_tolerance(peer_count: int, connection_density: float) -> float:
    """Estimate fault tolerance as percentage of nodes that can fail."""
    if peer_count <= 1:
        return 0.0

    # Simple estimation based on connectivity
    # Higher density = better fault tolerance
    base_tolerance = connection_density * 0.5  # Up to 50% with full mesh
    return min(base_tolerance, 0.8)  # Cap at 80%
