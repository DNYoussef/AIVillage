"""
Unified Message Format Specification

Implements Agent 5's messaging architecture blueprint for consolidating
5 fragmented communication systems into a single unified format.

Consolidated Systems:
1. P2P Communications (infrastructure/p2p/communications/)
2. Edge Chat Engine (infrastructure/edge/communication/)
3. Gateway Server (core/gateway/server.py)
4. WebSocket Handler (infrastructure/p2p/communications/websocket_handler.py)
5. Message Passing System (infrastructure/p2p/communications/message_passing_system.py)
"""

from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime, timezone
import uuid
from dataclasses import dataclass, field


class MessageType(Enum):
    """Unified message types across all transport layers"""
    # Agent communication
    AGENT_REQUEST = "agent_request"
    AGENT_RESPONSE = "agent_response" 
    AGENT_BROADCAST = "agent_broadcast"
    
    # HTTP/REST messages
    HTTP_REQUEST = "http_request"
    HTTP_RESPONSE = "http_response"
    
    # WebSocket messages
    WS_CONNECT = "ws_connect"
    WS_MESSAGE = "ws_message"
    WS_DISCONNECT = "ws_disconnect"
    
    # P2P messages
    P2P_DISCOVERY = "p2p_discovery"
    P2P_DATA = "p2p_data" 
    P2P_HEARTBEAT = "p2p_heartbeat"
    
    # Edge communication
    EDGE_CHAT = "edge_chat"
    EDGE_STATUS = "edge_status"
    EDGE_HEALTH = "edge_health"
    
    # System messages
    SYSTEM_NOTIFICATION = "system_notification"
    ERROR_RESPONSE = "error_response"
    ACK = "ack"
    NACK = "nack"


class TransportType(Enum):
    """Transport layer types"""
    HTTP = "http"
    WEBSOCKET = "websocket"
    P2P_LIBP2P = "p2p_libp2p"
    P2P_DIRECT = "p2p_direct"
    LOCAL = "local"


@dataclass
class UnifiedMessage:
    """Unified message format for all communication systems"""
    
    message_type: MessageType
    transport: TransportType
    source_id: str
    target_id: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Auto-generated fields
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def __post_init__(self):
        """Initialize default values after object creation"""
        if self.payload is None:
            self.payload = {}
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "transport": self.transport.value,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "payload": self.payload,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedMessage':
        """Create message from dictionary"""
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            message_type=MessageType(data["message_type"]),
            transport=TransportType(data["transport"]),
            source_id=data["source_id"],
            target_id=data.get("target_id"),
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat())
        )
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        import json
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'UnifiedMessage':
        """Create message from JSON string"""
        import json
        return cls.from_dict(json.loads(json_str))
    
    def create_response(self, response_payload: Dict[str, Any], 
                      response_type: MessageType = MessageType.AGENT_RESPONSE) -> 'UnifiedMessage':
        """Create a response message to this message"""
        response_metadata = self.metadata.copy()
        response_metadata.update({
            "correlation_id": self.message_id,
            "is_response": True,
            "parent_message_id": self.message_id
        })
        
        return UnifiedMessage(
            message_type=response_type,
            transport=self.transport,
            source_id=self.target_id or "system",
            target_id=self.source_id,
            payload=response_payload,
            metadata=response_metadata
        )
    
    def create_error_response(self, error_message: str, 
                            error_code: Optional[str] = None) -> 'UnifiedMessage':
        """Create an error response to this message"""
        error_payload = {
            "error": error_message,
            "error_code": error_code,
            "original_message_id": self.message_id
        }
        
        return self.create_response(error_payload, MessageType.ERROR_RESPONSE)
    
    def is_request(self) -> bool:
        """Check if this message expects a response"""
        return self.metadata.get("expects_response", False)
    
    def is_response(self) -> bool:
        """Check if this message is a response"""
        return self.metadata.get("is_response", False)
    
    def get_correlation_id(self) -> Optional[str]:
        """Get correlation ID for request-response tracking"""
        return self.metadata.get("correlation_id")


# Legacy compatibility functions for existing systems

def create_p2p_message(sender_id: str, recipient_id: str, 
                      message_type: str, payload: Any) -> UnifiedMessage:
    """Create message compatible with P2P message passing system"""
    try:
        msg_type = MessageType(message_type.lower())
    except ValueError:
        msg_type = MessageType.P2P_DATA
    
    return UnifiedMessage(
        message_type=msg_type,
        transport=TransportType.P2P_LIBP2P,
        source_id=sender_id,
        target_id=recipient_id,
        payload={"data": payload}
    )


def create_edge_chat_message(conversation_id: str, message: str) -> UnifiedMessage:
    """Create message compatible with edge chat engine"""
    return UnifiedMessage(
        message_type=MessageType.EDGE_CHAT,
        transport=TransportType.HTTP,
        source_id="edge_client",
        target_id="chat_engine",
        payload={
            "prompt": message,
            "conversation_id": conversation_id
        }
    )


def create_websocket_message(connection_id: str, data: Any) -> UnifiedMessage:
    """Create message compatible with WebSocket handler"""
    return UnifiedMessage(
        message_type=MessageType.WS_MESSAGE,
        transport=TransportType.WEBSOCKET,
        source_id=connection_id,
        target_id="websocket_server",
        payload={"data": data}
    )


def create_gateway_message(endpoint: str, method: str, 
                         data: Any, request_id: str) -> UnifiedMessage:
    """Create message compatible with gateway server"""
    return UnifiedMessage(
        message_type=MessageType.HTTP_REQUEST,
        transport=TransportType.HTTP,
        source_id="gateway_client",
        target_id="gateway_server",
        payload={
            "endpoint": endpoint,
            "method": method,
            "data": data,
            "request_id": request_id
        }
    )
