#!/usr/bin/env python3
"""
Unified Agent Communication Protocol - MCP-Enhanced Cross-System Messaging
========================================================================

Standardizes communication across all 5 agent framework patterns:
1. DSPy coordination messages
2. Enhanced memory-based agent communication
3. Sequential thinking coordination
4. Service instrumentation events
5. P2P protocol adapter integration

This protocol consolidates:
- Memory MCP shared state management
- Sequential thinking coordination chains
- DSPy optimization data exchange
- Performance monitoring events
- Cross-session agent coordination
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Protocol
import uuid

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Unified message type enumeration - weak connascence (CoN)."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    COORDINATION = "coordination"
    MEMORY_SYNC = "memory_sync"
    REASONING_CHAIN = "reasoning_chain"
    PERFORMANCE_METRIC = "performance_metric"
    AGENT_STATUS = "agent_status"
    DSPY_OPTIMIZATION = "dspy_optimization"
    SYSTEM_EVENT = "system_event"
    ERROR = "error"


class Priority(Enum):
    """Message priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class DeliveryMode(Enum):
    """Message delivery guarantees."""
    FIRE_AND_FORGET = "fire_and_forget"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"
    ORDERED = "ordered"


@dataclass
class MessageHeader:
    """Standardized message header with routing and metadata."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.SYSTEM_EVENT
    sender_id: str = ""
    recipient_id: str = ""
    session_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    priority: Priority = Priority.MEDIUM
    delivery_mode: DeliveryMode = DeliveryMode.FIRE_AND_FORGET
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    time_to_live_ms: Optional[int] = None
    route_trace: List[str] = field(default_factory=list)


@dataclass
class MessagePayload:
    """Flexible message payload supporting all agent framework data."""
    content: Dict[str, Any] = field(default_factory=dict)
    task_data: Optional[Dict[str, Any]] = None
    coordination_data: Optional[Dict[str, Any]] = None
    memory_data: Optional[Dict[str, Any]] = None
    reasoning_data: Optional[Dict[str, Any]] = None
    metrics_data: Optional[Dict[str, Any]] = None
    dspy_data: Optional[Dict[str, Any]] = None
    system_data: Optional[Dict[str, Any]] = None
    attachments: List[Dict[str, Any]] = field(default_factory=list)


@dataclass 
class UnifiedMessage:
    """Complete unified message structure."""
    header: MessageHeader
    payload: MessagePayload
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "header": {
                **asdict(self.header),
                "timestamp": self.header.timestamp.isoformat(),
                "message_type": self.header.message_type.value,
                "priority": self.header.priority.value,
                "delivery_mode": self.header.delivery_mode.value
            },
            "payload": asdict(self.payload)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedMessage":
        """Create message from dictionary."""
        header_data = data["header"]
        header = MessageHeader(
            message_id=header_data["message_id"],
            message_type=MessageType(header_data["message_type"]),
            sender_id=header_data["sender_id"],
            recipient_id=header_data["recipient_id"],
            session_id=header_data["session_id"],
            timestamp=datetime.fromisoformat(header_data["timestamp"]),
            priority=Priority(header_data["priority"]),
            delivery_mode=DeliveryMode(header_data["delivery_mode"]),
            correlation_id=header_data.get("correlation_id"),
            reply_to=header_data.get("reply_to"),
            time_to_live_ms=header_data.get("time_to_live_ms"),
            route_trace=header_data.get("route_trace", [])
        )
        
        payload = MessagePayload(**data["payload"])
        
        return cls(header=header, payload=payload)


class MessageTransport(Protocol):
    """Protocol for message transport implementations."""
    
    async def send(self, message: UnifiedMessage) -> bool:
        """Send message through transport."""
        ...
    
    async def receive(self) -> Optional[UnifiedMessage]:
        """Receive message from transport."""
        ...
    
    async def subscribe(self, pattern: str, callback: Callable[[UnifiedMessage], None]) -> str:
        """Subscribe to message pattern."""
        ...
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from message pattern."""
        ...


class MemoryMCPTransport:
    """Memory-based MCP transport for shared memory coordination."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.memory_path = self.project_root / ".mcp" / "unified_messages.json"
        self.memory_path.parent.mkdir(exist_ok=True)
        
        self.message_queue: List[UnifiedMessage] = []
        self.subscriptions: Dict[str, Dict[str, Any]] = {}
        self.message_store: Dict[str, UnifiedMessage] = {}
        
        logger.info("MemoryMCPTransport initialized")
    
    async def send(self, message: UnifiedMessage) -> bool:
        """Send message through memory MCP."""
        try:
            # Add to message queue
            message.header.route_trace.append(f"memory_mcp_{datetime.now().isoformat()}")
            self.message_queue.append(message)
            self.message_store[message.header.message_id] = message
            
            # Persist to disk
            await self._persist_messages()
            
            # Notify subscribers
            await self._notify_subscribers(message)
            
            logger.debug(f"Message sent via Memory MCP: {message.header.message_id}")
            return True
            
        except Exception as e:
            logger.error(f"Memory MCP send failed: {e}")
            return False
    
    async def receive(self) -> Optional[UnifiedMessage]:
        """Receive message from memory MCP."""
        try:
            if self.message_queue:
                message = self.message_queue.pop(0)
                logger.debug(f"Message received via Memory MCP: {message.header.message_id}")
                return message
            return None
            
        except Exception as e:
            logger.error(f"Memory MCP receive failed: {e}")
            return None
    
    async def subscribe(self, pattern: str, callback: Callable[[UnifiedMessage], None]) -> str:
        """Subscribe to message pattern."""
        subscription_id = str(uuid.uuid4())
        self.subscriptions[subscription_id] = {
            "pattern": pattern,
            "callback": callback,
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"Subscription created: {subscription_id} for pattern: {pattern}")
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from message pattern."""
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]
            logger.info(f"Subscription removed: {subscription_id}")
            return True
        return False
    
    async def _persist_messages(self):
        """Persist messages to disk for cross-session coordination."""
        try:
            # Keep only recent messages (last 1000)
            recent_messages = self.message_queue[-1000:]
            
            messages_data = {
                "messages": [msg.to_dict() for msg in recent_messages],
                "last_updated": datetime.now().isoformat(),
                "total_messages": len(self.message_store)
            }
            
            with open(self.memory_path, 'w') as f:
                json.dump(messages_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Message persistence failed: {e}")
    
    async def _notify_subscribers(self, message: UnifiedMessage):
        """Notify relevant subscribers of new message."""
        try:
            for sub_id, sub_data in self.subscriptions.items():
                pattern = sub_data["pattern"]
                callback = sub_data["callback"]
                
                # Simple pattern matching (can be enhanced)
                if self._matches_pattern(message, pattern):
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message)
                        else:
                            callback(message)
                    except Exception as e:
                        logger.error(f"Subscriber callback failed: {e}")
                        
        except Exception as e:
            logger.error(f"Subscriber notification failed: {e}")
    
    def _matches_pattern(self, message: UnifiedMessage, pattern: str) -> bool:
        """Check if message matches subscription pattern."""
        # Basic pattern matching - can be enhanced with regex
        if pattern == "*":
            return True
        if pattern == message.header.message_type.value:
            return True
        if pattern == message.header.sender_id:
            return True
        if pattern in str(message.payload.content):
            return True
        return False


class P2PTransport:
    """P2P transport integration with existing BitChat/Betanet systems."""
    
    def __init__(self, protocol_adapter=None):
        self.protocol_adapter = protocol_adapter
        self.peer_connections: Dict[str, Any] = {}
        self.message_buffer: List[UnifiedMessage] = []
        
        logger.info("P2PTransport initialized")
    
    async def send(self, message: UnifiedMessage) -> bool:
        """Send message through P2P network."""
        try:
            # Convert unified message to P2P format
            p2p_message = self._convert_to_p2p_format(message)
            
            if self.protocol_adapter:
                # Use existing protocol adapter
                success = await self.protocol_adapter.send_unified_message(
                    recipient=message.header.recipient_id,
                    payload=p2p_message
                )
            else:
                # Buffer for later delivery
                self.message_buffer.append(message)
                success = True
            
            message.header.route_trace.append(f"p2p_{datetime.now().isoformat()}")
            logger.debug(f"Message sent via P2P: {message.header.message_id}")
            return success
            
        except Exception as e:
            logger.error(f"P2P send failed: {e}")
            return False
    
    async def receive(self) -> Optional[UnifiedMessage]:
        """Receive message from P2P network."""
        try:
            if self.protocol_adapter:
                # Receive from protocol adapter
                p2p_message = await self.protocol_adapter.receive_unified_message()
                if p2p_message:
                    return self._convert_from_p2p_format(p2p_message)
            
            # Check buffer
            if self.message_buffer:
                return self.message_buffer.pop(0)
                
            return None
            
        except Exception as e:
            logger.error(f"P2P receive failed: {e}")
            return None
    
    async def subscribe(self, pattern: str, callback: Callable[[UnifiedMessage], None]) -> str:
        """Subscribe to P2P message pattern."""
        # For now, use simple subscription tracking
        subscription_id = str(uuid.uuid4())
        logger.info(f"P2P subscription created: {subscription_id}")
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from P2P pattern."""
        logger.info(f"P2P subscription removed: {subscription_id}")
        return True
    
    def _convert_to_p2p_format(self, message: UnifiedMessage) -> Dict[str, Any]:
        """Convert unified message to P2P protocol format."""
        return {
            "id": message.header.message_id,
            "type": message.header.message_type.value,
            "from": message.header.sender_id,
            "to": message.header.recipient_id,
            "data": message.payload.content,
            "timestamp": message.header.timestamp.isoformat(),
            "priority": message.header.priority.value
        }
    
    def _convert_from_p2p_format(self, p2p_message: Dict[str, Any]) -> UnifiedMessage:
        """Convert P2P message to unified format."""
        header = MessageHeader(
            message_id=p2p_message.get("id", str(uuid.uuid4())),
            message_type=MessageType(p2p_message.get("type", MessageType.SYSTEM_EVENT.value)),
            sender_id=p2p_message.get("from", ""),
            recipient_id=p2p_message.get("to", ""),
            timestamp=datetime.fromisoformat(p2p_message.get("timestamp", datetime.now().isoformat())),
            priority=Priority(p2p_message.get("priority", Priority.MEDIUM.value))
        )
        
        payload = MessagePayload(
            content=p2p_message.get("data", {})
        )
        
        return UnifiedMessage(header=header, payload=payload)


class UnifiedCommunicationHub:
    """
    Central communication hub integrating all transport mechanisms.
    
    Provides unified interface for:
    - Memory MCP coordination
    - P2P mesh networking
    - Sequential thinking chains
    - DSPy optimization data
    - Performance monitoring events
    """
    
    def __init__(self, 
                 enable_memory_mcp: bool = True,
                 enable_p2p: bool = True,
                 project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.transports: Dict[str, MessageTransport] = {}
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        self.routing_table: Dict[str, str] = {}  # agent_id -> preferred_transport
        
        # Initialize transports
        if enable_memory_mcp:
            self.transports["memory_mcp"] = MemoryMCPTransport(project_root)
        
        if enable_p2p:
            self.transports["p2p"] = P2PTransport()
        
        # Message statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_failed": 0,
            "active_subscriptions": 0,
            "start_time": datetime.now()
        }
        
        logger.info(f"UnifiedCommunicationHub initialized with transports: {list(self.transports.keys())}")
    
    async def send_message(self,
                          message_type: MessageType,
                          sender_id: str,
                          recipient_id: str,
                          content: Dict[str, Any],
                          session_id: Optional[str] = None,
                          priority: Priority = Priority.MEDIUM,
                          delivery_mode: DeliveryMode = DeliveryMode.FIRE_AND_FORGET) -> bool:
        """Send unified message across all appropriate transports."""
        try:
            # Create unified message
            header = MessageHeader(
                message_type=message_type,
                sender_id=sender_id,
                recipient_id=recipient_id,
                session_id=session_id or "",
                priority=priority,
                delivery_mode=delivery_mode
            )
            
            payload = MessagePayload(content=content)
            message = UnifiedMessage(header=header, payload=payload)
            
            # Determine transport based on routing
            transport_name = self._select_transport(recipient_id, message_type)
            
            if transport_name in self.transports:
                transport = self.transports[transport_name]
                success = await transport.send(message)
                
                if success:
                    self.stats["messages_sent"] += 1
                    logger.debug(f"Message sent via {transport_name}: {header.message_id}")
                else:
                    self.stats["messages_failed"] += 1
                    logger.error(f"Message send failed via {transport_name}")
                
                return success
            else:
                logger.error(f"No transport available for: {transport_name}")
                self.stats["messages_failed"] += 1
                return False
                
        except Exception as e:
            logger.error(f"Message send failed: {e}")
            self.stats["messages_failed"] += 1
            return False
    
    async def receive_messages(self, timeout_ms: int = 1000) -> List[UnifiedMessage]:
        """Receive messages from all transports."""
        messages = []
        
        try:
            # Poll all transports with timeout
            tasks = []
            for name, transport in self.transports.items():
                tasks.append(self._receive_from_transport(name, transport))
            
            if tasks:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout_ms / 1000.0
                )
                
                for result in results:
                    if isinstance(result, UnifiedMessage):
                        messages.append(result)
                        self.stats["messages_received"] += 1
                    elif isinstance(result, Exception):
                        logger.error(f"Transport receive error: {result}")
            
        except asyncio.TimeoutError:
            pass  # Normal timeout
        except Exception as e:
            logger.error(f"Message receive failed: {e}")
        
        return messages
    
    async def subscribe_to_messages(self,
                                  message_type: Optional[MessageType] = None,
                                  sender_pattern: Optional[str] = None,
                                  callback: Optional[Callable[[UnifiedMessage], None]] = None) -> str:
        """Subscribe to messages matching criteria."""
        try:
            subscription_id = str(uuid.uuid4())
            
            # Create pattern based on criteria
            if message_type:
                pattern = message_type.value
            elif sender_pattern:
                pattern = sender_pattern
            else:
                pattern = "*"  # Subscribe to all
            
            # Subscribe to all transports
            for name, transport in self.transports.items():
                await transport.subscribe(pattern, callback or self._default_message_handler)
            
            self.stats["active_subscriptions"] += 1
            logger.info(f"Subscription created: {subscription_id} for pattern: {pattern}")
            return subscription_id
            
        except Exception as e:
            logger.error(f"Subscription failed: {e}")
            return ""
    
    # Specialized message sending methods for each framework pattern
    
    async def send_task_request(self,
                              sender_id: str,
                              recipient_id: str,
                              task_data: Dict[str, Any],
                              session_id: Optional[str] = None) -> bool:
        """Send task request with task-specific payload structure."""
        payload_content = {
            "task_type": "request",
            "task_data": task_data,
            "requires_response": True
        }
        
        return await self.send_message(
            message_type=MessageType.TASK_REQUEST,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=payload_content,
            session_id=session_id,
            priority=Priority.HIGH,
            delivery_mode=DeliveryMode.AT_LEAST_ONCE
        )
    
    async def send_coordination_message(self,
                                      sender_id: str,
                                      coordination_data: Dict[str, Any],
                                      session_id: str) -> bool:
        """Send coordination message for multi-agent orchestration."""
        payload_content = {
            "coordination_type": coordination_data.get("type", "general"),
            "coordination_data": coordination_data,
            "session_context": session_id
        }
        
        return await self.send_message(
            message_type=MessageType.COORDINATION,
            sender_id=sender_id,
            recipient_id="*",  # Broadcast
            content=payload_content,
            session_id=session_id,
            priority=Priority.MEDIUM
        )
    
    async def send_memory_sync(self,
                             sender_id: str,
                             memory_key: str,
                             memory_data: Dict[str, Any],
                             session_id: str) -> bool:
        """Send memory synchronization message."""
        payload_content = {
            "memory_operation": "sync",
            "memory_key": memory_key,
            "memory_data": memory_data,
            "sync_timestamp": datetime.now().isoformat()
        }
        
        return await self.send_message(
            message_type=MessageType.MEMORY_SYNC,
            sender_id=sender_id,
            recipient_id="memory_coordinator",
            content=payload_content,
            session_id=session_id,
            priority=Priority.LOW
        )
    
    async def send_reasoning_chain(self,
                                 sender_id: str,
                                 recipient_id: str,
                                 reasoning_data: Dict[str, Any],
                                 session_id: str) -> bool:
        """Send sequential thinking reasoning chain."""
        payload_content = {
            "reasoning_type": reasoning_data.get("type", "sequential"),
            "reasoning_steps": reasoning_data.get("steps", []),
            "reasoning_context": reasoning_data.get("context", {}),
            "complexity_level": reasoning_data.get("complexity", "medium")
        }
        
        return await self.send_message(
            message_type=MessageType.REASONING_CHAIN,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=payload_content,
            session_id=session_id,
            priority=Priority.HIGH
        )
    
    async def send_performance_metric(self,
                                    sender_id: str,
                                    metrics_data: Dict[str, Any],
                                    session_id: str) -> bool:
        """Send performance monitoring data."""
        payload_content = {
            "metric_type": metrics_data.get("type", "performance"),
            "metrics": metrics_data,
            "measurement_timestamp": datetime.now().isoformat()
        }
        
        return await self.send_message(
            message_type=MessageType.PERFORMANCE_METRIC,
            sender_id=sender_id,
            recipient_id="performance_monitor",
            content=payload_content,
            session_id=session_id,
            priority=Priority.BACKGROUND
        )
    
    async def send_dspy_optimization(self,
                                   sender_id: str,
                                   recipient_id: str,
                                   optimization_data: Dict[str, Any],
                                   session_id: str) -> bool:
        """Send DSPy optimization data."""
        payload_content = {
            "optimization_type": optimization_data.get("type", "prompt"),
            "optimization_data": optimization_data,
            "optimization_target": optimization_data.get("target", 0.9)
        }
        
        return await self.send_message(
            message_type=MessageType.DSPY_OPTIMIZATION,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=payload_content,
            session_id=session_id,
            priority=Priority.MEDIUM
        )
    
    # Communication Hub Management
    
    def set_routing_preference(self, agent_id: str, transport_name: str):
        """Set preferred transport for specific agent."""
        if transport_name in self.transports:
            self.routing_table[agent_id] = transport_name
            logger.info(f"Routing preference set: {agent_id} -> {transport_name}")
        else:
            logger.error(f"Invalid transport for routing: {transport_name}")
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication hub statistics."""
        runtime_seconds = (datetime.now() - self.stats["start_time"]).total_seconds()
        
        return {
            **self.stats,
            "runtime_seconds": runtime_seconds,
            "messages_per_second": self.stats["messages_sent"] / max(runtime_seconds, 1),
            "success_rate": self.stats["messages_sent"] / max(self.stats["messages_sent"] + self.stats["messages_failed"], 1),
            "active_transports": list(self.transports.keys()),
            "routing_entries": len(self.routing_table)
        }
    
    async def shutdown(self):
        """Gracefully shutdown communication hub."""
        try:
            logger.info("Shutting down communication hub")
            
            # Final stats
            final_stats = self.get_communication_stats()
            logger.info(f"Final communication stats: {final_stats}")
            
            # Clear transports
            self.transports.clear()
            self.routing_table.clear()
            
            logger.info("Communication hub shutdown complete")
            
        except Exception as e:
            logger.error(f"Communication hub shutdown failed: {e}")
    
    # Private helper methods
    
    def _select_transport(self, recipient_id: str, message_type: MessageType) -> str:
        """Select appropriate transport based on recipient and message type."""
        # Check routing table first
        if recipient_id in self.routing_table:
            return self.routing_table[recipient_id]
        
        # Default routing based on message type
        if message_type in [MessageType.MEMORY_SYNC, MessageType.COORDINATION]:
            return "memory_mcp"
        elif message_type in [MessageType.TASK_REQUEST, MessageType.TASK_RESPONSE]:
            return "p2p" if "p2p" in self.transports else "memory_mcp"
        else:
            # Use first available transport
            return next(iter(self.transports.keys())) if self.transports else ""
    
    async def _receive_from_transport(self, name: str, transport: MessageTransport) -> Optional[UnifiedMessage]:
        """Receive message from specific transport."""
        try:
            return await transport.receive()
        except Exception as e:
            logger.error(f"Transport {name} receive error: {e}")
            return None
    
    def _default_message_handler(self, message: UnifiedMessage):
        """Default message handler for unhandled messages."""
        logger.info(f"Received message: {message.header.message_type.value} from {message.header.sender_id}")


# Convenience factory functions

def create_communication_hub(project_root: Optional[Path] = None,
                           enable_all_transports: bool = True) -> UnifiedCommunicationHub:
    """Create communication hub with standard configuration."""
    return UnifiedCommunicationHub(
        enable_memory_mcp=enable_all_transports,
        enable_p2p=enable_all_transports,
        project_root=project_root
    )


def create_agent_message(message_type: MessageType,
                        sender_id: str,
                        recipient_id: str,
                        content: Dict[str, Any],
                        session_id: Optional[str] = None) -> UnifiedMessage:
    """Create standardized agent message."""
    header = MessageHeader(
        message_type=message_type,
        sender_id=sender_id,
        recipient_id=recipient_id,
        session_id=session_id or ""
    )
    
    payload = MessagePayload(content=content)
    
    return UnifiedMessage(header=header, payload=payload)


# Export unified communication protocol
__all__ = [
    "UnifiedCommunicationHub",
    "UnifiedMessage",
    "MessageType",
    "Priority",
    "DeliveryMode",
    "MessageHeader",
    "MessagePayload",
    "MemoryMCPTransport",
    "P2PTransport",
    "create_communication_hub",
    "create_agent_message"
]