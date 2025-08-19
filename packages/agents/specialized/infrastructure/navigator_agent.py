"""Navigator Agent - Routing & Data Movement

The routing and data movement specialist of AIVillage, responsible for:
- P2P mesh networking and node discovery
- Message routing between agents
- Bandwidth optimization and load balancing
- DTN (Delay Tolerant Networking) protocols
- Data movement and synchronization
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from packages.agents.core.agent_interface import AgentInterface

logger = logging.getLogger(__name__)


class RouteType(Enum):
    DIRECT = "direct"
    MULTI_HOP = "multi_hop"
    BROADCAST = "broadcast"
    ANYCAST = "anycast"
    STORE_FORWARD = "store_forward"


class NodeStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    UNREACHABLE = "unreachable"


@dataclass
class NetworkNode:
    node_id: str
    node_type: str
    address: str
    port: int
    status: NodeStatus
    capabilities: list[str]
    last_seen: float
    bandwidth_mbps: float
    latency_ms: float
    load_factor: float


@dataclass
class Route:
    route_id: str
    source: str
    destination: str
    path: list[str]
    route_type: RouteType
    cost: float
    latency_ms: float
    bandwidth_mbps: float
    reliability: float
    created_at: float


@dataclass
class Message:
    message_id: str
    source: str
    destination: str
    payload: dict[str, Any]
    priority: int
    ttl: int
    timestamp: float
    route: Route | None = None


class NavigatorAgent(AgentInterface):
    """Navigator Agent handles all routing and data movement for AIVillage,
    including P2P mesh networking, message routing, and bandwidth optimization.
    """

    def __init__(self, agent_id: str = "navigator_agent"):
        self.agent_id = agent_id
        self.agent_type = "Navigator"
        self.capabilities = [
            "path_routing",
            "mesh_networking",
            "dtn_protocols",
            "bandwidth_optimization",
            "node_discovery",
            "load_balancing",
            "message_queuing",
            "network_monitoring",
            "data_synchronization",
            "failover_routing",
        ]

        # Network topology
        self.network_nodes: dict[str, NetworkNode] = {}
        self.routing_table: dict[str, list[Route]] = {}
        self.message_queue: list[Message] = []
        self.active_connections: dict[str, dict[str, Any]] = {}

        # Performance tracking
        self.messages_routed = 0
        self.bytes_transferred = 0
        self.routing_failures = 0
        self.discovery_events = 0

        # Network metrics
        self.network_health = 1.0
        self.average_latency_ms = 0.0
        self.bandwidth_utilization = 0.0
        self.node_count = 0

        # Configuration
        self.max_hops = 5
        self.discovery_interval = 30  # seconds
        self.route_cache_ttl = 300  # 5 minutes
        self.max_queue_size = 1000

        self.initialized = False

    async def generate(self, prompt: str) -> str:
        """Generate routing and networking responses"""
        prompt_lower = prompt.lower()

        if "route" in prompt_lower or "path" in prompt_lower:
            return "I find optimal paths and route messages between agents in the mesh network."
        if "network" in prompt_lower or "mesh" in prompt_lower:
            return "I manage P2P mesh networking with automatic node discovery and failover."
        if "bandwidth" in prompt_lower or "optimize" in prompt_lower:
            return "I optimize bandwidth usage and load balance traffic across network paths."
        if "sync" in prompt_lower or "data" in prompt_lower:
            return "I handle data synchronization and movement between distributed nodes."
        if "discovery" in prompt_lower:
            return "I discover and monitor network nodes, tracking their capabilities and status."

        return "I am Navigator Agent, managing routing and data movement across AIVillage."

    async def get_embedding(self, text: str) -> list[float]:
        """Generate network-focused embeddings"""
        hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
        # Network embeddings focus on connectivity patterns
        return [(hash_value % 1000) / 1000.0] * 256

    async def rerank(self, query: str, results: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        """Rerank based on network relevance"""
        network_keywords = [
            "route",
            "path",
            "network",
            "mesh",
            "node",
            "bandwidth",
            "latency",
            "connectivity",
            "protocol",
            "data",
            "sync",
        ]

        for result in results:
            score = 0
            content = str(result.get("content", ""))

            for keyword in network_keywords:
                score += content.lower().count(keyword) * 1.5

            # Boost networking and distributed system content
            if any(term in content.lower() for term in ["distributed", "p2p", "communication"]):
                score *= 1.4

            result["network_relevance"] = score

        return sorted(results, key=lambda x: x.get("network_relevance", 0), reverse=True)[:k]

    async def introspect(self) -> dict[str, Any]:
        """Return Navigator agent status and network metrics"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "network_nodes": len(self.network_nodes),
            "active_routes": sum(len(routes) for routes in self.routing_table.values()),
            "message_queue_size": len(self.message_queue),
            "messages_routed": self.messages_routed,
            "bytes_transferred": self.bytes_transferred,
            "routing_failures": self.routing_failures,
            "network_health": self.network_health,
            "average_latency_ms": self.average_latency_ms,
            "bandwidth_utilization": self.bandwidth_utilization,
            "discovery_events": self.discovery_events,
            "specialization": "routing_and_networking",
            "initialized": self.initialized,
        }

    async def communicate(self, message: str, recipient: "AgentInterface") -> str:
        """Route communications through optimized network paths"""
        # Add routing context to communications
        routing_context = "[NETWORK ROUTED]"
        message = f"{routing_context} {message}"

        if recipient:
            # Simulate routing through mesh network
            routing_info = await self._find_optimal_route("navigator", recipient.__class__.__name__)
            response = await recipient.generate(f"Navigator Agent routes message: {message}")
            return f"Message routed via {routing_info}: {response[:50]}..."
        return "No recipient for routing"

    async def activate_latent_space(self, query: str) -> tuple[str, str]:
        """Activate network-specific latent spaces"""
        query_lower = query.lower()

        if "route" in query_lower:
            space_type = "path_routing"
        elif "mesh" in query_lower or "network" in query_lower:
            space_type = "mesh_networking"
        elif "discovery" in query_lower:
            space_type = "node_discovery"
        elif "bandwidth" in query_lower:
            space_type = "bandwidth_optimization"
        else:
            space_type = "general_networking"

        latent_repr = f"NAVIGATOR[{space_type}:{query[:50]}]"
        return space_type, latent_repr

    async def discover_nodes(self, search_radius: int = 3) -> dict[str, Any]:
        """Discover network nodes within search radius - MVP function"""
        discovery_id = f"discovery_{int(time.time())}"

        # Simulate node discovery in mesh network
        discovered_nodes = await self._perform_node_discovery(search_radius)

        # Update network topology
        for node_data in discovered_nodes:
            node = NetworkNode(
                node_id=node_data["node_id"],
                node_type=node_data["node_type"],
                address=node_data["address"],
                port=node_data["port"],
                status=NodeStatus.ONLINE,
                capabilities=node_data["capabilities"],
                last_seen=time.time(),
                bandwidth_mbps=node_data.get("bandwidth_mbps", 100.0),
                latency_ms=node_data.get("latency_ms", 10.0),
                load_factor=node_data.get("load_factor", 0.1),
            )
            self.network_nodes[node_data["node_id"]] = node

        # Create receipt
        receipt = {
            "agent": "Navigator",
            "action": "node_discovery",
            "discovery_id": discovery_id,
            "timestamp": time.time(),
            "nodes_discovered": len(discovered_nodes),
            "search_radius": search_radius,
            "network_size": len(self.network_nodes),
            "signature": f"navigator_discovery_{discovery_id}",
        }

        self.discovery_events += 1
        self.node_count = len(self.network_nodes)

        logger.info(f"Node discovery completed: {len(discovered_nodes)} nodes found")

        return {
            "status": "success",
            "discovery_id": discovery_id,
            "nodes_discovered": discovered_nodes,
            "total_network_nodes": len(self.network_nodes),
            "receipt": receipt,
        }

    async def _perform_node_discovery(self, search_radius: int) -> list[dict[str, Any]]:
        """Perform actual node discovery simulation"""
        # Simulate discovering various agent types in the network
        base_nodes = [
            {
                "node_id": "king_agent_node",
                "node_type": "orchestrator",
                "address": "192.168.1.10",
                "port": 8001,
                "capabilities": ["task_orchestration", "agent_coordination"],
                "bandwidth_mbps": 1000.0,
                "latency_ms": 5.0,
                "load_factor": 0.3,
            },
            {
                "node_id": "magi_agent_node",
                "node_type": "engineering",
                "address": "192.168.1.11",
                "port": 8002,
                "capabilities": ["code_generation", "model_training"],
                "bandwidth_mbps": 500.0,
                "latency_ms": 8.0,
                "load_factor": 0.5,
            },
            {
                "node_id": "shield_agent_node",
                "node_type": "security",
                "address": "192.168.1.12",
                "port": 8003,
                "capabilities": ["security_scan", "threat_detection"],
                "bandwidth_mbps": 200.0,
                "latency_ms": 12.0,
                "load_factor": 0.2,
            },
            {
                "node_id": "polyglot_agent_node",
                "node_type": "translation",
                "address": "192.168.1.13",
                "port": 8004,
                "capabilities": ["translation", "localization"],
                "bandwidth_mbps": 150.0,
                "latency_ms": 15.0,
                "load_factor": 0.4,
            },
        ]

        # Return subset based on search radius
        return base_nodes[:search_radius]

    async def route_message(self, message: dict[str, Any], destination: str) -> dict[str, Any]:
        """Route message to destination through optimal path - MVP function"""
        message_id = f"msg_{int(time.time())}_{len(self.message_queue)}"

        # Create message object
        msg = Message(
            message_id=message_id,
            source=self.agent_id,
            destination=destination,
            payload=message,
            priority=message.get("priority", 5),
            ttl=message.get("ttl", 300),
            timestamp=time.time(),
        )

        # Find optimal route
        optimal_route = await self._find_optimal_route(self.agent_id, destination)
        msg.route = optimal_route

        # Route the message
        routing_result = await self._route_message_via_path(msg, optimal_route)

        # Create receipt
        receipt = {
            "agent": "Navigator",
            "action": "message_routing",
            "message_id": message_id,
            "timestamp": time.time(),
            "source": self.agent_id,
            "destination": destination,
            "route_hops": len(optimal_route["path"]) if optimal_route else 0,
            "latency_ms": routing_result.get("latency_ms", 0),
            "bytes_transferred": routing_result.get("bytes", len(json.dumps(message))),
            "routing_success": routing_result.get("success", False),
            "signature": f"navigator_route_{message_id}",
        }

        # Update metrics
        if routing_result.get("success"):
            self.messages_routed += 1
            self.bytes_transferred += receipt["bytes_transferred"]
        else:
            self.routing_failures += 1

        logger.info(
            f"Message routing: {message_id} -> {destination} ({'SUCCESS' if routing_result.get('success') else 'FAILED'})"
        )

        return {
            "status": "success" if routing_result.get("success") else "failed",
            "message_id": message_id,
            "route": optimal_route,
            "routing_result": routing_result,
            "receipt": receipt,
        }

    async def _find_optimal_route(self, source: str, destination: str) -> dict[str, Any]:
        """Find optimal route using network topology"""
        # Simple routing algorithm for MVP
        if destination in self.network_nodes:
            # Direct route available
            direct_route = Route(
                route_id=f"route_{source}_{destination}",
                source=source,
                destination=destination,
                path=[source, destination],
                route_type=RouteType.DIRECT,
                cost=1.0,
                latency_ms=self.network_nodes[destination].latency_ms,
                bandwidth_mbps=self.network_nodes[destination].bandwidth_mbps,
                reliability=0.95,
                created_at=time.time(),
            )
            return {
                "route_id": direct_route.route_id,
                "path": direct_route.path,
                "type": direct_route.route_type.value,
                "cost": direct_route.cost,
                "latency_ms": direct_route.latency_ms,
                "bandwidth_mbps": direct_route.bandwidth_mbps,
            }
        # Multi-hop route needed
        intermediate_nodes = list(self.network_nodes.keys())[:2]  # Use first 2 nodes as hops
        multi_hop_path = [source] + intermediate_nodes + [destination]

        return {
            "route_id": f"route_multihop_{source}_{destination}",
            "path": multi_hop_path,
            "type": RouteType.MULTI_HOP.value,
            "cost": len(multi_hop_path) - 1,
            "latency_ms": (len(multi_hop_path) - 1) * 20,  # 20ms per hop
            "bandwidth_mbps": 100.0,  # Conservative estimate
        }

    async def _route_message_via_path(self, message: Message, route: dict[str, Any]) -> dict[str, Any]:
        """Route message through specified path"""
        # Simulate message routing
        path = route["path"]
        total_latency = route["latency_ms"]

        # Check if all nodes in path are reachable
        reachable_nodes = 0
        for node in path[1:-1]:  # Skip source and destination
            if node in self.network_nodes and self.network_nodes[node].status == NodeStatus.ONLINE:
                reachable_nodes += 1

        # Determine success based on path reachability
        0.95 - (0.1 * len(path))  # Success decreases with path length
        routing_success = reachable_nodes >= (len(path) - 2) * 0.8  # 80% of intermediate nodes must be reachable

        return {
            "success": routing_success,
            "latency_ms": total_latency,
            "bytes": len(json.dumps(message.payload)),
            "hops": len(path) - 1,
            "path_taken": path,
            "bandwidth_used_mbps": min(100.0, route["bandwidth_mbps"]),
        }

    async def optimize_bandwidth(self, target_utilization: float = 0.8) -> dict[str, Any]:
        """Optimize network bandwidth usage - MVP function"""
        optimization_id = f"bw_opt_{int(time.time())}"

        # Analyze current bandwidth usage
        current_usage = await self._analyze_bandwidth_usage()

        # Apply optimization strategies
        optimizations_applied = []
        if current_usage["utilization"] > target_utilization:
            optimizations_applied = await self._apply_bandwidth_optimizations(current_usage)

        # Calculate new metrics
        optimized_usage = await self._analyze_bandwidth_usage()
        improvement = current_usage["utilization"] - optimized_usage["utilization"]

        # Create receipt
        receipt = {
            "agent": "Navigator",
            "action": "bandwidth_optimization",
            "optimization_id": optimization_id,
            "timestamp": time.time(),
            "before_utilization": current_usage["utilization"],
            "after_utilization": optimized_usage["utilization"],
            "improvement_percent": improvement * 100,
            "optimizations_applied": len(optimizations_applied),
            "target_utilization": target_utilization,
            "signature": f"navigator_bw_{optimization_id}",
        }

        self.bandwidth_utilization = optimized_usage["utilization"]

        logger.info(f"Bandwidth optimization completed: {improvement * 100:.1f}% improvement")

        return {
            "status": "success",
            "optimization_id": optimization_id,
            "improvement_percent": improvement * 100,
            "optimizations_applied": optimizations_applied,
            "current_utilization": optimized_usage["utilization"],
            "receipt": receipt,
        }

    async def _analyze_bandwidth_usage(self) -> dict[str, Any]:
        """Analyze current bandwidth usage patterns"""
        # Simulate bandwidth analysis
        total_bandwidth = sum(node.bandwidth_mbps for node in self.network_nodes.values())
        used_bandwidth = sum(node.bandwidth_mbps * node.load_factor for node in self.network_nodes.values())

        return {
            "total_bandwidth_mbps": total_bandwidth,
            "used_bandwidth_mbps": used_bandwidth,
            "utilization": used_bandwidth / max(1, total_bandwidth),
            "peak_usage_nodes": [node.node_id for node in self.network_nodes.values() if node.load_factor > 0.8],
            "bottleneck_links": [],  # Would identify congested network links
        }

    async def _apply_bandwidth_optimizations(self, usage_info: dict[str, Any]) -> list[str]:
        """Apply bandwidth optimization strategies"""
        optimizations = []

        # Load balancing across nodes
        if len(usage_info["peak_usage_nodes"]) > 0:
            optimizations.append("load_balancing_across_high_usage_nodes")

        # Route optimization
        if len(self.routing_table) > 10:
            optimizations.append("route_optimization_and_consolidation")

        # Message compression
        if self.bytes_transferred > 1000000:  # 1MB
            optimizations.append("message_payload_compression")

        # QoS prioritization
        optimizations.append("quality_of_service_prioritization")

        return optimizations

    async def sync_data(self, data_spec: dict[str, Any]) -> dict[str, Any]:
        """Synchronize data across network nodes - MVP function"""
        sync_id = f"sync_{int(time.time())}"

        # Determine sync strategy
        sync_nodes = data_spec.get("target_nodes", list(self.network_nodes.keys())[:3])
        data_size_mb = data_spec.get("size_mb", 1.0)
        sync_type = data_spec.get("type", "incremental")  # full or incremental

        # Execute synchronization
        sync_results = await self._execute_data_sync(sync_nodes, data_size_mb, sync_type)

        # Create receipt
        receipt = {
            "agent": "Navigator",
            "action": "data_synchronization",
            "sync_id": sync_id,
            "timestamp": time.time(),
            "nodes_synced": len(sync_results["successful_nodes"]),
            "data_size_mb": data_size_mb,
            "sync_type": sync_type,
            "success_rate": len(sync_results["successful_nodes"]) / len(sync_nodes),
            "duration_seconds": sync_results["duration_seconds"],
            "signature": f"navigator_sync_{sync_id}",
        }

        logger.info(f"Data sync completed: {sync_id} - {len(sync_results['successful_nodes'])}/{len(sync_nodes)} nodes")

        return {
            "status": "success",
            "sync_id": sync_id,
            "sync_results": sync_results,
            "receipt": receipt,
        }

    async def _execute_data_sync(self, nodes: list[str], size_mb: float, sync_type: str) -> dict[str, Any]:
        """Execute data synchronization across specified nodes"""
        successful_nodes = []
        failed_nodes = []

        start_time = time.time()

        for node_id in nodes:
            if node_id in self.network_nodes and self.network_nodes[node_id].status == NodeStatus.ONLINE:
                # Simulate sync based on node bandwidth and load
                node = self.network_nodes[node_id]
                (size_mb * 8) / node.bandwidth_mbps  # Convert MB to Mbits, then divide by bandwidth

                if node.load_factor < 0.8:  # Node not overloaded
                    successful_nodes.append(node_id)
                else:
                    failed_nodes.append(node_id)
            else:
                failed_nodes.append(node_id)

        duration = time.time() - start_time

        return {
            "successful_nodes": successful_nodes,
            "failed_nodes": failed_nodes,
            "duration_seconds": duration,
            "bytes_transferred": size_mb * 1024 * 1024 * len(successful_nodes),
        }

    async def get_network_health_report(self) -> dict[str, Any]:
        """Generate comprehensive network health report"""
        online_nodes = [n for n in self.network_nodes.values() if n.status == NodeStatus.ONLINE]
        avg_latency = sum(n.latency_ms for n in online_nodes) / max(1, len(online_nodes))

        return {
            "agent": "Navigator",
            "report_type": "network_health",
            "timestamp": time.time(),
            "network_metrics": {
                "total_nodes": len(self.network_nodes),
                "online_nodes": len(online_nodes),
                "network_health": len(online_nodes) / max(1, len(self.network_nodes)),
                "average_latency_ms": avg_latency,
                "total_bandwidth_mbps": sum(n.bandwidth_mbps for n in online_nodes),
                "messages_routed": self.messages_routed,
                "routing_success_rate": self.messages_routed / max(1, self.messages_routed + self.routing_failures),
            },
            "performance_stats": {
                "bytes_transferred": self.bytes_transferred,
                "discovery_events": self.discovery_events,
                "active_routes": sum(len(routes) for routes in self.routing_table.values()),
                "queue_size": len(self.message_queue),
            },
            "recommendations": [
                "Monitor high-latency nodes for potential issues",
                "Consider load balancing for high-utilization nodes",
                "Implement redundant paths for critical routes",
            ],
        }

    async def initialize(self):
        """Initialize the Navigator Agent"""
        try:
            logger.info("Initializing Navigator Agent - Routing & Data Movement...")

            # Initialize with self as first node
            self_node = NetworkNode(
                node_id=self.agent_id,
                node_type="routing",
                address="127.0.0.1",
                port=8000,
                status=NodeStatus.ONLINE,
                capabilities=self.capabilities,
                last_seen=time.time(),
                bandwidth_mbps=1000.0,
                latency_ms=1.0,
                load_factor=0.1,
            )
            self.network_nodes[self.agent_id] = self_node

            # Initialize routing table
            self.routing_table = {}

            self.initialized = True
            logger.info(f"Navigator Agent {self.agent_id} initialized - Network ready for routing")

        except Exception as e:
            logger.error(f"Failed to initialize Navigator Agent: {e}")
            self.initialized = False

    async def shutdown(self):
        """Shutdown Navigator Agent gracefully"""
        try:
            logger.info("Navigator Agent shutting down...")

            # Generate final network report
            final_report = await self.get_network_health_report()
            logger.info(f"Navigator Agent final report: {final_report['network_metrics']}")

            # Clear message queue
            self.message_queue.clear()

            self.initialized = False

        except Exception as e:
            logger.error(f"Error during Navigator Agent shutdown: {e}")
