"""
Mock transport implementations for testing P2P components.

Provides mock implementations of all base P2P interfaces for
isolated testing without real network dependencies.
"""

import asyncio
from datetime import datetime
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

try:
    from infrastructure.p2p.base import (
        BaseDiscovery,
        BaseMessage,
        BaseMetrics,
        BaseNode,
        BaseProtocol,
        BaseTransport,
        MessagePriority,
        MessageType,
        TransportStatus,
    )
except ImportError:
    # Fallback for testing without base classes
    BaseTransport = object
    BaseProtocol = object
    BaseMessage = object
    BaseNode = object
    BaseDiscovery = object
    BaseMetrics = object

    # Mock the required enums/classes
    class TransportStatus:
        CONNECTED = "connected"
        DISCONNECTED = "disconnected"

    class MessageType:
        DATA = "data"

    class MessagePriority:
        MEDIUM = 2


logger = logging.getLogger(__name__)


class MockTransport(BaseTransport):
    """Mock transport implementation for testing."""

    def __init__(self, transport_id: str = "mock_transport"):
        super().__init__(transport_id)
        self.mock_peers: Dict[str, Dict[str, Any]] = {}
        self.mock_messages: List[Any] = []
        self.connection_delay = 0.0
        self.message_delay = 0.0
        self.failure_rate = 0.0
        self.should_fail_next = False

        # Mock behaviors
        self.connect_callback = None
        self.disconnect_callback = None
        self.message_callback = None

    @property
    def transport_type(self) -> str:
        return "mock"

    @property
    def capabilities(self) -> List:
        return ["mock_capability"]

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize mock transport."""
        self.status = TransportStatus.CONNECTED
        if self.connection_delay > 0:
            await asyncio.sleep(self.connection_delay)

    async def connect(self, peer_address: str, **kwargs) -> str:
        """Mock connection to peer."""
        if self.should_fail_next:
            self.should_fail_next = False
            raise ConnectionError("Mock connection failure")

        if self.connection_delay > 0:
            await asyncio.sleep(self.connection_delay)

        peer_id = f"mock_peer_{len(self.mock_peers)}"
        self.mock_peers[peer_id] = {"address": peer_address, "connected_at": datetime.now(), "status": "connected"}

        if self.connect_callback:
            await self.connect_callback(peer_id, peer_address)

        return peer_id

    async def disconnect(self, peer_id: str) -> None:
        """Mock disconnection from peer."""
        if peer_id in self.mock_peers:
            self.mock_peers[peer_id]["status"] = "disconnected"

            if self.disconnect_callback:
                await self.disconnect_callback(peer_id)

    async def send_message(self, peer_id: str, message) -> bool:
        """Mock message sending."""
        if self.should_fail_next:
            self.should_fail_next = False
            return False

        if peer_id not in self.mock_peers:
            return False

        if self.mock_peers[peer_id]["status"] != "connected":
            return False

        if self.message_delay > 0:
            await asyncio.sleep(self.message_delay)

        # Store sent message
        self.mock_messages.append(
            {"peer_id": peer_id, "message": message, "timestamp": datetime.now(), "direction": "sent"}
        )

        if self.message_callback:
            await self.message_callback(peer_id, message)

        return True

    async def receive_messages(self) -> AsyncGenerator[Any, None]:
        """Mock message receiving."""
        # Simple implementation - yield mock received messages
        for msg in self.mock_messages:
            if msg["direction"] == "received":
                yield msg["message"]

    async def get_peer_info(self, peer_id: str) -> Optional[Any]:
        """Get mock peer information."""
        if peer_id not in self.mock_peers:
            return None

        peer_data = self.mock_peers[peer_id]
        return {
            "peer_id": peer_id,
            "address": peer_data["address"],
            "status": peer_data["status"],
            "connected_at": peer_data["connected_at"],
        }

    async def list_peers(self) -> List[str]:
        """List connected peers."""
        return [pid for pid, data in self.mock_peers.items() if data["status"] == "connected"]

    async def shutdown(self) -> None:
        """Shutdown mock transport."""
        self.status = TransportStatus.DISCONNECTED
        for peer_id in list(self.mock_peers.keys()):
            await self.disconnect(peer_id)

    # Test helper methods
    def simulate_received_message(self, peer_id: str, message: Any):
        """Simulate receiving a message from peer."""
        self.mock_messages.append(
            {"peer_id": peer_id, "message": message, "timestamp": datetime.now(), "direction": "received"}
        )

    def set_failure_mode(self, should_fail: bool = True):
        """Set transport to fail next operation."""
        self.should_fail_next = should_fail

    def get_sent_messages(self, peer_id: Optional[str] = None) -> List[Dict]:
        """Get sent messages, optionally filtered by peer."""
        messages = [msg for msg in self.mock_messages if msg["direction"] == "sent"]
        if peer_id:
            messages = [msg for msg in messages if msg["peer_id"] == peer_id]
        return messages


class MockProtocol(BaseProtocol):
    """Mock protocol implementation for testing."""

    def __init__(self, protocol_name: str = "mock_protocol"):
        super().__init__(protocol_name)
        self.encoding_delay = 0.0
        self.decoding_delay = 0.0

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> List:
        return ["mock_capability"]

    async def encode_message(self, message) -> bytes:
        """Mock message encoding."""
        if self.encoding_delay > 0:
            await asyncio.sleep(self.encoding_delay)

        # Simple mock encoding
        import json

        data = {"message": str(message), "protocol": self.protocol_name}
        return json.dumps(data).encode()

    async def decode_message(self, data: bytes):
        """Mock message decoding."""
        if self.decoding_delay > 0:
            await asyncio.sleep(self.decoding_delay)

        # Simple mock decoding
        import json

        try:
            decoded = json.loads(data.decode())
            return decoded.get("message", data)
        except:
            return data

    async def handle_handshake(self, peer_id: str) -> Dict[str, Any]:
        """Mock handshake handling."""
        return {
            "protocol": self.protocol_name,
            "version": self.version,
            "peer_id": peer_id,
            "capabilities": self.capabilities,
        }

    async def validate_peer(self, peer_id: str, credentials: Dict[str, Any]) -> bool:
        """Mock peer validation."""
        # Always accept for testing
        return True


class MockNode(BaseNode):
    """Mock node implementation for testing."""

    def __init__(self, node_id: str = "mock_node"):
        super().__init__(node_id)
        self.mock_network_peers: List[str] = []
        self.is_joined = False

    @property
    def identity(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "role": self.role.value if hasattr(self.role, "value") else str(self.role),
            "capabilities": [
                cap.value if hasattr(cap, "value") else str(cap) for cap in self.capabilities.capabilities
            ],
        }

    async def join_network(self, bootstrap_nodes: List[str]) -> None:
        """Mock network joining."""
        self.mock_network_peers = bootstrap_nodes.copy()
        self.is_joined = True

    async def leave_network(self) -> None:
        """Mock network leaving."""
        self.mock_network_peers.clear()
        self.is_joined = False

    async def get_network_status(self) -> Dict[str, Any]:
        """Mock network status."""
        return {
            "joined": self.is_joined,
            "peer_count": len(self.mock_network_peers),
            "peers": self.mock_network_peers,
            "node_id": self.node_id,
        }

    async def advertise_capabilities(self) -> None:
        """Mock capability advertisement."""
        # Mock implementation


class MockDiscovery(BaseDiscovery):
    """Mock discovery implementation for testing."""

    def __init__(self, discovery_id: str = "mock_discovery"):
        super().__init__(discovery_id)
        self.mock_discovered_peers: Dict[str, Any] = {}
        self.is_discovering = False

    @property
    def supported_methods(self) -> List:
        return ["broadcast", "multicast"]

    async def start_discovery(self, method, **kwargs) -> None:
        """Mock discovery start."""
        self.is_discovering = True

        # Simulate discovering some peers
        for i in range(3):
            peer_id = f"discovered_peer_{i}"
            self.mock_discovered_peers[peer_id] = {
                "peer_id": peer_id,
                "supported_protocols": ["mock"],
                "capabilities": ["mock_capability"],
                "max_connections": 10,
                "discovered_at": datetime.now(),
            }

    async def stop_discovery(self) -> None:
        """Mock discovery stop."""
        self.is_discovering = False

    async def discover_peers(self, timeout: int = 30):
        """Mock peer discovery."""
        if not self.is_discovering:
            await self.start_discovery("broadcast")

        # Simulate discovery time
        await asyncio.sleep(0.1)

        return {
            "discovered_peers": list(self.mock_discovered_peers.keys()),
            "discovery_method": "broadcast",
            "discovery_time": 0.1,
            "network_topology": {"type": "mesh"},
            "bootstrap_nodes": [],
        }

    async def announce_presence(self) -> None:
        """Mock presence announcement."""

    async def query_peer_capabilities(self, peer_id: str) -> Optional[Any]:
        """Mock peer capability query."""
        if peer_id in self.mock_discovered_peers:
            return self.mock_discovered_peers[peer_id]
        return None


class MockMetrics(BaseMetrics):
    """Mock metrics implementation for testing."""

    def __init__(self, component_name: str = "mock_component"):
        super().__init__(component_name)
        self.recorded_metrics: List[Dict[str, Any]] = []

    async def record_metric(self, name: str, value, metric_type, labels: Optional[Dict[str, str]] = None) -> None:
        """Mock metric recording."""
        metric = {
            "name": name,
            "value": value,
            "metric_type": metric_type.value if hasattr(metric_type, "value") else str(metric_type),
            "labels": labels or {},
            "timestamp": datetime.now(),
            "component": self.component_name,
        }
        self.recorded_metrics.append(metric)

    async def get_metric(self, name: str, time_range: Optional[tuple] = None) -> List:
        """Mock metric retrieval."""
        metrics = [m for m in self.recorded_metrics if m["name"] == name]

        if time_range:
            start_time, end_time = time_range
            metrics = [m for m in metrics if start_time <= m["timestamp"] <= end_time]

        return metrics

    async def get_all_metrics(self) -> Dict[str, List]:
        """Mock all metrics retrieval."""
        result = {}
        for metric in self.recorded_metrics:
            name = metric["name"]
            if name not in result:
                result[name] = []
            result[name].append(metric)
        return result

    async def export_metrics(self, format: str = "json") -> str:
        """Mock metrics export."""
        if format == "json":
            import json

            return json.dumps(self.recorded_metrics, default=str)
        else:
            return str(self.recorded_metrics)

    async def clear_metrics(self, older_than: Optional[datetime] = None) -> None:
        """Mock metrics clearing."""
        if older_than:
            self.recorded_metrics = [m for m in self.recorded_metrics if m["timestamp"] >= older_than]
        else:
            self.recorded_metrics.clear()

    def get_recorded_metrics(self) -> List[Dict[str, Any]]:
        """Get all recorded metrics for testing."""
        return self.recorded_metrics.copy()


# Factory functions for easy test setup
def create_mock_transport(**kwargs) -> MockTransport:
    """Create mock transport with configuration."""
    transport = MockTransport()
    for key, value in kwargs.items():
        if hasattr(transport, key):
            setattr(transport, key, value)
    return transport


def create_mock_protocol(**kwargs) -> MockProtocol:
    """Create mock protocol with configuration."""
    protocol = MockProtocol()
    for key, value in kwargs.items():
        if hasattr(protocol, key):
            setattr(protocol, key, value)
    return protocol


def create_connected_mock_peers(count: int = 3) -> List[MockTransport]:
    """Create multiple connected mock transports."""
    transports = []
    for i in range(count):
        transport = MockTransport(f"mock_transport_{i}")
        transports.append(transport)

    return transports
