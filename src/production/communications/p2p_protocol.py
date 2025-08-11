"""Enhanced Communication Protocol with P2P Integration."""

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any

from src.communications.message import Message, MessageType, Priority

# Import existing communication components
from src.communications.protocol import (
    CommunicationProtocol,
    StandardCommunicationProtocol,
)

# Import P2P components
from .p2p import DeviceMesh, P2PNode, StreamingConfig, TensorStreaming
from .p2p.device_mesh import MeshProtocol
from .p2p.tensor_streaming import TensorMetadata

logger = logging.getLogger(__name__)


class P2PMessageType(Enum):
    """Extended message types for P2P communication."""

    MESH_DATA = "mesh_data"
    TENSOR_TRANSFER = "tensor_transfer"
    PEER_DISCOVERY = "peer_discovery"
    RESOURCE_REQUEST = "resource_request"
    SYNC_REQUEST = "sync_request"
    DISTRIBUTED_TASK = "distributed_task"


@dataclass
class P2PCapabilities:
    """P2P capabilities of a node."""

    supports_mesh_routing: bool = True
    supports_tensor_streaming: bool = True
    supports_distributed_inference: bool = True
    max_concurrent_connections: int = 50
    preferred_protocols: list[str] = None
    device_resources: dict[str, Any] = None

    def __post_init__(self):
        if self.preferred_protocols is None:
            self.preferred_protocols = ["tcp", "bluetooth", "wifi_direct"]
        if self.device_resources is None:
            self.device_resources = {}


class P2PCommunicationProtocol(CommunicationProtocol):
    """Enhanced communication protocol with P2P mesh networking capabilities."""

    def __init__(
        self,
        node_id: str | None = None,
        port: int = 8000,
        capabilities: P2PCapabilities | None = None,
        standard_protocol: StandardCommunicationProtocol | None = None,
        mesh_protocol: MeshProtocol = MeshProtocol.OPTIMIZED_LINK_STATE,
        streaming_config: StreamingConfig | None = None,
    ) -> None:
        self.capabilities = capabilities or P2PCapabilities()
        self.standard_protocol = standard_protocol or StandardCommunicationProtocol()

        # Initialize P2P components
        self.p2p_node = P2PNode(node_id=node_id, port=port)
        self.device_mesh = DeviceMesh(
            node=self.p2p_node,
            protocol=mesh_protocol,
        )
        self.tensor_streaming = TensorStreaming(
            node=self.p2p_node,
            config=streaming_config or StreamingConfig(),
        )

        # P2P-specific state
        self.mesh_enabled = False
        self.distributed_agents: dict[str, str] = {}  # agent_id -> peer_id
        self.local_agents: set[str] = set()
        self.message_routing_cache: dict[str, str] = {}  # receiver -> peer_id

        # Statistics
        self.p2p_stats = {
            "messages_routed": 0,
            "tensors_streamed": 0,
            "mesh_broadcasts": 0,
            "peer_discoveries": 0,
            "distributed_tasks": 0,
        }

        # Register P2P handlers
        self._register_p2p_handlers()

    async def start_p2p(self) -> None:
        """Start P2P networking capabilities."""
        logger.info("Starting P2P communication protocol")

        try:
            # Start P2P node
            await self.p2p_node.start()

            # Start mesh networking
            await self.device_mesh.start_mesh()
            self.mesh_enabled = True

            logger.info("P2P communication protocol started successfully")

        except Exception as e:
            logger.exception(f"Failed to start P2P protocol: {e}")
            raise

    async def stop_p2p(self) -> None:
        """Stop P2P networking."""
        logger.info("Stopping P2P communication protocol")

        self.mesh_enabled = False

        try:
            await self.device_mesh.stop_mesh()
            await self.p2p_node.stop()

            logger.info("P2P communication protocol stopped")

        except Exception as e:
            logger.exception(f"Error stopping P2P protocol: {e}")

    def register_local_agent(self, agent_id: str) -> None:
        """Register an agent as running locally on this node."""
        self.local_agents.add(agent_id)
        logger.debug(f"Registered local agent: {agent_id}")

    def register_distributed_agent(self, agent_id: str, peer_id: str) -> None:
        """Register an agent as running on a remote peer."""
        self.distributed_agents[agent_id] = peer_id
        self.message_routing_cache[agent_id] = peer_id
        logger.debug(f"Registered distributed agent {agent_id} on peer {peer_id}")

    async def discover_peers(self, addresses: list[tuple[str, int]]) -> int:
        """Discover and connect to peers."""
        successful_connections = 0

        for address, port in addresses:
            self.p2p_node.add_known_address(address, port)
            success = await self.p2p_node.connect_to_peer(address, port)

            if success:
                successful_connections += 1
                self.p2p_stats["peer_discoveries"] += 1

        logger.info(f"Connected to {successful_connections}/{len(addresses)} peers")
        return successful_connections

    async def send_message(self, message: Message) -> None:
        """Send message with P2P routing capabilities."""
        receiver = message.receiver

        # Check if receiver is local
        if receiver in self.local_agents:
            await self.standard_protocol.send_message(message)
            return

        # Check if receiver is on a known peer
        if receiver in self.distributed_agents:
            peer_id = self.distributed_agents[receiver]

            # Convert to P2P message format
            p2p_payload = {
                "original_message": message.to_dict(),
                "target_agent": receiver,
                "routing_type": "direct",
            }

            success = await self.p2p_node.send_message(peer_id, MessageType.DATA, p2p_payload)

            if success:
                self.p2p_stats["messages_routed"] += 1
                return
            logger.warning(f"Failed to send message to peer {peer_id}")

        # Try mesh routing if enabled
        if self.mesh_enabled:
            success = await self._route_via_mesh(message)
            if success:
                return

        # Fallback to standard protocol
        await self.standard_protocol.send_message(message)

    async def receive_message(self, agent_id: str) -> Message:
        """Receive message for local agent."""
        if agent_id not in self.local_agents:
            msg = f"Agent {agent_id} is not registered locally"
            raise ValueError(msg)

        return await self.standard_protocol.receive_message(agent_id)

    async def query(
        self,
        sender: str,
        receiver: str,
        content: dict[str, Any],
        priority: Priority = Priority.MEDIUM,
    ) -> Any:
        """Query with P2P routing."""
        query_message = Message(
            type=MessageType.QUERY,
            sender=sender,
            receiver=receiver,
            content=content,
            priority=priority,
        )

        await self.send_message(query_message)

        # Wait for response
        if sender in self.local_agents:
            response = await self.receive_message(sender)
            return response.content
        # For distributed queries, implement response handling
        logger.warning("Distributed query response handling not yet implemented")
        return None

    async def send_and_wait(self, message: Message, timeout: float = 5.0) -> Message:
        """Send message and wait for response with P2P support."""
        if message.receiver in self.local_agents:
            return await self.standard_protocol.send_and_wait(message, timeout)
        # For P2P send and wait, need to implement response correlation
        await self.send_message(message)

        # Simple polling for now - in production would use proper response correlation
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            try:
                response = await self.receive_message(message.sender)
                if response.parent_id == message.id:
                    return response
            except:
                pass

            await asyncio.sleep(0.1)

        msg = "Response timeout in P2P send_and_wait"
        raise TimeoutError(msg)

    def subscribe(self, agent_id: str, callback: Callable[[Message], Coroutine[Any, Any, None]]) -> None:
        """Subscribe to messages for an agent."""
        self.standard_protocol.subscribe(agent_id, callback)

    async def broadcast_mesh(
        self,
        sender: str,
        message_type: MessageType,
        content: dict[str, Any],
        priority: Priority = Priority.MEDIUM,
        max_hops: int | None = None,
    ) -> int:
        """Broadcast message through mesh network."""
        if not self.mesh_enabled:
            logger.warning("Mesh networking not enabled")
            return 0

        mesh_data = {
            "message_type": message_type.value,
            "sender": sender,
            "content": content,
            "priority": priority.value,
            "timestamp": asyncio.get_event_loop().time(),
        }

        successful_sends = await self.device_mesh.broadcast_mesh(mesh_data, max_hops)

        self.p2p_stats["mesh_broadcasts"] += 1
        return successful_sends

    async def stream_tensor(
        self,
        tensor_data: Any,
        tensor_name: str,
        destination: str,
        metadata_tags: dict[str, Any] | None = None,
    ) -> str:
        """Stream tensor data to a peer."""
        # Find peer for destination
        peer_id = self.distributed_agents.get(destination)

        if not peer_id:
            # Try to find peer through mesh
            peers = self.p2p_node.get_connected_peers()
            if peers:
                peer_id = peers[0].peer_id  # Use first available peer
            else:
                msg = f"No peer found for destination {destination}"
                raise ValueError(msg)

        tensor_id = await self.tensor_streaming.send_tensor(
            tensor_data=tensor_data,
            tensor_name=tensor_name,
            destination=peer_id,
            metadata_tags=metadata_tags,
        )

        self.p2p_stats["tensors_streamed"] += 1
        return tensor_id

    async def receive_tensor(
        self,
        tensor_id: str,
        timeout: float = 300.0,
        progress_callback: Callable | None = None,
    ) -> tuple[Any, TensorMetadata] | None:
        """Receive tensor data from peer."""
        return await self.tensor_streaming.receive_tensor(tensor_id, timeout, progress_callback)

    async def distribute_task(
        self,
        task: dict[str, Any],
        target_agents: list[str],
        coordination_strategy: str = "parallel",
    ) -> list[Any]:
        """Distribute a task across multiple agents/peers."""
        results = []

        if coordination_strategy == "parallel":
            # Execute tasks in parallel
            tasks = []

            for agent_id in target_agents:
                task_message = Message(
                    type=MessageType.QUERY,
                    sender="coordinator",
                    receiver=agent_id,
                    content={"task": task, "task_type": "distributed"},
                )

                tasks.append(self.send_message(task_message))

            await asyncio.gather(*tasks)

            # Collect results (simplified)
            for agent_id in target_agents:
                try:
                    response = await self.receive_message("coordinator")
                    results.append(response.content)
                except:
                    results.append(None)

        elif coordination_strategy == "sequential":
            # Execute tasks sequentially
            for agent_id in target_agents:
                task_message = Message(
                    type=MessageType.QUERY,
                    sender="coordinator",
                    receiver=agent_id,
                    content={"task": task, "task_type": "distributed"},
                )

                await self.send_message(task_message)

                try:
                    response = await self.receive_message("coordinator")
                    results.append(response.content)
                except:
                    results.append(None)

        self.p2p_stats["distributed_tasks"] += 1
        return results

    def get_p2p_status(self) -> dict[str, Any]:
        """Get comprehensive P2P status."""
        connected_peers = self.p2p_node.get_connected_peers()

        return {
            "node_id": self.p2p_node.node_id,
            "node_status": self.p2p_node.status.value,
            "mesh_enabled": self.mesh_enabled,
            "connected_peers": len(connected_peers),
            "peer_details": [
                {
                    "peer_id": peer.peer_id,
                    "address": peer.address,
                    "port": peer.port,
                    "status": peer.status.value,
                    "latency_ms": peer.latency_ms,
                }
                for peer in connected_peers
            ],
            "local_agents": list(self.local_agents),
            "distributed_agents": dict(self.distributed_agents),
            "capabilities": {
                "supports_mesh_routing": self.capabilities.supports_mesh_routing,
                "supports_tensor_streaming": self.capabilities.supports_tensor_streaming,
                "supports_distributed_inference": self.capabilities.supports_distributed_inference,
                "max_concurrent_connections": self.capabilities.max_concurrent_connections,
            },
            "statistics": self.p2p_stats,
            "mesh_status": (self.device_mesh.get_mesh_status() if self.mesh_enabled else None),
            "streaming_stats": self.tensor_streaming.get_streaming_stats(),
        }

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get P2P performance metrics."""
        node_stats = self.p2p_node.get_stats()

        return {
            "node_metrics": node_stats,
            "mesh_metrics": {
                "avg_latency_ms": (self.device_mesh._calculate_average_latency() if self.mesh_enabled else 0),
                "avg_bandwidth_kbps": (self.device_mesh._calculate_average_bandwidth() if self.mesh_enabled else 0),
                "network_diameter": (self.device_mesh._calculate_network_diameter() if self.mesh_enabled else 0),
            },
            "streaming_metrics": self.tensor_streaming.get_streaming_stats(),
            "routing_efficiency": {
                "messages_routed": self.p2p_stats["messages_routed"],
                "routing_cache_hits": len(self.message_routing_cache),
                "peer_discovery_success": self.p2p_stats["peer_discoveries"],
            },
        }

    async def optimize_network(self) -> None:
        """Optimize P2P network performance."""
        logger.info("Optimizing P2P network")

        if self.mesh_enabled:
            await self.device_mesh.optimize_routing()

        # Clear stale routing cache
        stale_entries = []
        for agent_id, peer_id in self.message_routing_cache.items():
            if peer_id not in [p.peer_id for p in self.p2p_node.get_connected_peers()]:
                stale_entries.append(agent_id)

        for agent_id in stale_entries:
            del self.message_routing_cache[agent_id]

        logger.info(f"Network optimization complete, removed {len(stale_entries)} stale routes")

    def _register_p2p_handlers(self) -> None:
        """Register P2P-specific message handlers."""
        from .p2p.p2p_node import MessageType as P2PMessageType

        async def handle_agent_message(p2p_message, writer=None) -> None:
            """Handle messages destined for local agents."""
            payload = p2p_message.payload

            if "original_message" in payload:
                # Reconstruct the original message
                msg_data = payload["original_message"]
                message = Message(
                    type=MessageType(msg_data["type"]),
                    sender=msg_data["sender"],
                    receiver=msg_data["receiver"],
                    content=msg_data["content"],
                    priority=Priority(msg_data.get("priority", "medium")),
                    id=msg_data.get("id"),
                    parent_id=msg_data.get("parent_id"),
                    timestamp=msg_data.get("timestamp"),
                )

                # Route to local agent
                target_agent = payload.get("target_agent", message.receiver)
                if target_agent in self.local_agents:
                    await self.standard_protocol.send_message(message)

        self.p2p_node.register_handler(P2PMessageType.DATA, handle_agent_message)

    async def _route_via_mesh(self, message: Message) -> bool:
        """Route message via mesh network."""
        mesh_payload = {
            "original_message": message.to_dict(),
            "target_agent": message.receiver,
            "routing_type": "mesh",
        }

        success = await self.device_mesh.send_mesh_message(
            destination=message.receiver,
            data=mesh_payload,
            store_and_forward=True,
        )

        if success:
            self.p2p_stats["messages_routed"] += 1

        return success
