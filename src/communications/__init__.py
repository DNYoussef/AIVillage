from .event_dispatcher import EventDispatcher
from .message import Message, MessageType, Priority
from .message_passing_system import MessagePassing, MessagePassingSystem
from .message_queue import MessageQueue
from .message_router import MessageRouter
from .protocol_handler import ProtocolHandler
from .service_discovery import ServiceDiscovery, ServiceRegistry, discover_services
from .service_info import ServiceInfo
from .standard_protocol import StandardCommunicationProtocol
from .websocket_handler import WebSocketHandler

try:
    from .credit_manager import CreditManager
    from .federated_client import FederatedClient
    from .mesh_node import MeshNode
    from .sharder import ShardPlanner
except Exception:  # pragma: no cover - optional heavy deps may be missing
    MeshNode = None  # type: ignore
    CreditManager = None  # type: ignore
    FederatedClient = None  # type: ignore
    ShardPlanner = None  # type: ignore

__all__ = [
    "CreditManager",
    "FederatedClient",
    "MeshNode",
    "Message",
    "MessageQueue",
    "MessagePassingSystem",
    "MessagePassing",
    "MessageType",
    "Priority",
    "ServiceDiscovery",
    "ServiceInfo",
    "ServiceRegistry",
    "ShardPlanner",
    "StandardCommunicationProtocol",
    "MessageRouter",
    "ProtocolHandler",
    "EventDispatcher",
    "WebSocketHandler",
    "discover_services",
]
