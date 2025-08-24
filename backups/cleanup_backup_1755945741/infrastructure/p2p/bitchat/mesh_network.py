"""
BitChat Mesh Network Implementation

Provides mesh networking capabilities for BitChat BLE transport,
including routing, topology management, and store-and-forward messaging.
"""

from dataclasses import dataclass, field
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class MeshNode:
    """Represents a node in the BitChat mesh network."""

    node_id: str
    device_name: str = ""
    last_seen: float = field(default_factory=time.time)
    hop_count: int = 1
    signal_strength: int = -70  # dBm
    battery_level: float | None = None
    capabilities: set[str] = field(default_factory=set)

    # Routing information
    neighbors: set[str] = field(default_factory=set)
    routes: dict[str, int] = field(default_factory=dict)  # destination -> hop_count

    def is_online(self, timeout: int = 120) -> bool:
        """Check if node is considered online."""
        return (time.time() - self.last_seen) < timeout

    def update_activity(self):
        """Update node activity timestamp."""
        self.last_seen = time.time()


class MeshNetwork:
    """BitChat mesh network topology and routing manager."""

    def __init__(self, local_node_id: str):
        self.local_node_id = local_node_id
        self.nodes: dict[str, MeshNode] = {}
        self.routing_table: dict[str, str] = {}  # destination -> next_hop

        # Add local node
        self.nodes[local_node_id] = MeshNode(
            node_id=local_node_id,
            device_name="Local Node",
            hop_count=0,
        )

    def add_node(self, node: MeshNode) -> bool:
        """Add or update a node in the mesh."""
        if node.node_id == self.local_node_id:
            return False  # Don't add ourselves

        existing_node = self.nodes.get(node.node_id)

        if existing_node:
            # Update existing node
            existing_node.last_seen = node.last_seen
            existing_node.hop_count = min(existing_node.hop_count, node.hop_count)
            existing_node.signal_strength = max(existing_node.signal_strength, node.signal_strength)
            if node.battery_level is not None:
                existing_node.battery_level = node.battery_level
            existing_node.capabilities.update(node.capabilities)
        else:
            # Add new node
            self.nodes[node.node_id] = node
            logger.debug(f"Added mesh node: {node.node_id}")

        # Update routing table
        self._update_routing_table()
        return True

    def remove_node(self, node_id: str):
        """Remove a node from the mesh."""
        if node_id in self.nodes and node_id != self.local_node_id:
            del self.nodes[node_id]
            self._update_routing_table()
            logger.debug(f"Removed mesh node: {node_id}")

    def get_route_to(self, destination: str) -> str | None:
        """Get next hop for routing to destination."""
        return self.routing_table.get(destination)

    def get_neighbors(self) -> list[MeshNode]:
        """Get direct neighbors (1-hop nodes)."""
        return [node for node in self.nodes.values() if node.hop_count == 1 and node.is_online()]

    def get_all_nodes(self) -> list[MeshNode]:
        """Get all known nodes in the mesh."""
        return list(self.nodes.values())

    def get_online_nodes(self) -> list[MeshNode]:
        """Get all online nodes in the mesh."""
        return [node for node in self.nodes.values() if node.is_online()]

    def cleanup_offline_nodes(self, timeout: int = 300):
        """Remove nodes that have been offline too long."""
        offline_nodes = []

        for node_id, node in self.nodes.items():
            if node_id != self.local_node_id and not node.is_online(timeout):
                offline_nodes.append(node_id)

        for node_id in offline_nodes:
            self.remove_node(node_id)

    def _update_routing_table(self):
        """Update routing table using simplified distance-vector algorithm."""
        # Clear current routing table
        self.routing_table.clear()

        # For each destination, find the neighbor with minimum hop count
        neighbors = self.get_neighbors()

        for node_id, node in self.nodes.items():
            if node_id == self.local_node_id:
                continue

            if node.hop_count == 1:
                # Direct neighbor
                self.routing_table[node_id] = node_id
            else:
                # Find best next hop through neighbors
                best_neighbor = None
                best_distance = float("inf")

                for neighbor in neighbors:
                    if node_id in neighbor.routes:
                        total_distance = 1 + neighbor.routes[node_id]
                        if total_distance < best_distance:
                            best_distance = total_distance
                            best_neighbor = neighbor.node_id

                if best_neighbor:
                    self.routing_table[node_id] = best_neighbor

    def get_topology_info(self) -> dict:
        """Get mesh topology information."""
        neighbors = self.get_neighbors()
        online_nodes = self.get_online_nodes()

        return {
            "local_node_id": self.local_node_id,
            "total_nodes": len(self.nodes),
            "online_nodes": len(online_nodes),
            "direct_neighbors": len(neighbors),
            "routing_table_size": len(self.routing_table),
            "max_hop_count": max((node.hop_count for node in self.nodes.values()), default=0),
        }

    def find_path_to(self, destination: str, max_hops: int = 7) -> list[str] | None:
        """Find complete path to destination (for debugging)."""
        if destination not in self.nodes:
            return None

        if destination == self.local_node_id:
            return [self.local_node_id]

        path = [self.local_node_id]

        for _ in range(max_hops):
            next_hop = self.routing_table.get(destination)
            if not next_hop:
                break

            path.append(next_hop)

            if next_hop == destination:
                return path

        return None  # No path found within max_hops


def create_mesh_network(local_node_id: str) -> MeshNetwork:
    """Factory function to create mesh network."""
    return MeshNetwork(local_node_id)
