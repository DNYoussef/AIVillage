"""
Message Bus Interface - Core messaging abstraction.

Defines the contract for all message bus implementations in the unified system.
This interface consolidates patterns from all 5 existing communication systems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol
from datetime import datetime

from ..core.message import Message


@dataclass
class MessageBusConfig:
    """Configuration for message bus instances."""
    
    agent_id: str
    max_queue_size: int = 10000
    message_timeout_seconds: int = 30
    retry_attempts: int = 3
    enable_persistence: bool = False
    enable_metrics: bool = True
    enable_middleware: bool = True
    buffer_size: int = 1000
    batch_size: int = 100


@dataclass 
class MessageBusMetrics:
    """Metrics tracking for message bus operations."""
    
    messages_sent: int = 0
    messages_received: int = 0
    messages_failed: int = 0
    messages_dropped: int = 0
    average_latency_ms: float = 0.0
    peak_queue_size: int = 0
    last_activity: Optional[datetime] = None
    uptime_seconds: int = 0


class MessageBus(ABC):
    """
    Abstract base class for unified message bus implementations.
    
    Consolidates patterns from:
    1. MessagePassingSystem (P2P communications)
    2. AgentCommunicationProtocol (core agents) 
    3. ChatEngine (edge communications)
    4. DigitalTwinCommunication (twin messaging)
    5. Infrastructure bridges (fog/p2p)
    """
    
    def __init__(self, config: MessageBusConfig):
        self.config = config
        self.metrics = MessageBusMetrics()
        self._handlers: Dict[str, List[Callable]] = {}
        self._middleware: List[Any] = []
        self._running = False
    
    @abstractmethod
    async def start(self) -> None:
        """Start the message bus and all transports."""
        pass
    
    @abstractmethod 
    async def stop(self) -> None:
        """Stop the message bus and cleanup resources."""
        pass
    
    @abstractmethod
    async def publish(self, message: Message) -> bool:
        """
        Publish a message to the bus for delivery.
        
        Args:
            message: The message to publish
            
        Returns:
            True if message was accepted for delivery
        """
        pass
    
    @abstractmethod
    async def send_direct(self, recipient_id: str, message: Message) -> bool:
        """
        Send message directly to specific recipient.
        
        Args:
            recipient_id: Target agent/service ID
            message: Message to send
            
        Returns:
            True if message was sent successfully
        """
        pass
    
    @abstractmethod
    async def broadcast(self, message: Message, exclude: Optional[List[str]] = None) -> int:
        """
        Broadcast message to all connected agents/services.
        
        Args:
            message: Message to broadcast  
            exclude: Agent IDs to exclude from broadcast
            
        Returns:
            Number of recipients that received the message
        """
        pass
    
    @abstractmethod
    async def send_request(
        self, 
        recipient_id: str, 
        request_message: Message,
        timeout_seconds: Optional[int] = None
    ) -> Optional[Message]:
        """
        Send request and wait for response.
        
        Args:
            recipient_id: Target for the request
            request_message: The request message  
            timeout_seconds: How long to wait for response
            
        Returns:
            Response message if received, None if timeout
        """
        pass
    
    @abstractmethod
    async def subscribe(self, message_type: str, handler: Callable[[Message], Awaitable[None]]) -> str:
        """
        Subscribe to messages of a specific type.
        
        Args:
            message_type: Type of messages to receive
            handler: Async function to handle messages
            
        Returns:
            Subscription ID for later unsubscription
        """
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from messages.
        
        Args:
            subscription_id: ID returned from subscribe()
            
        Returns:
            True if successfully unsubscribed
        """
        pass
    
    @abstractmethod
    def add_middleware(self, middleware: Any) -> None:
        """Add message processing middleware."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> MessageBusMetrics:
        """Get current metrics and statistics."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current status information.""" 
        pass
    
    # Convenience methods that can be implemented in base classes
    
    def is_running(self) -> bool:
        """Check if message bus is currently running."""
        return self._running
    
    async def send_response(self, original_request: Message, response_payload: Any) -> bool:
        """
        Send response to a previous request message.
        
        Args:
            original_request: The request message being responded to
            response_payload: The response data
            
        Returns:
            True if response was sent successfully
        """
        if not original_request.metadata.correlation_id:
            return False
            
        response_message = Message(
            type=f"{original_request.type}_response",
            sender_id=self.config.agent_id,
            recipient_id=original_request.sender_id,
            payload=response_payload,
            metadata=original_request.metadata.create_response_metadata()
        )
        
        return await self.send_direct(original_request.sender_id, response_message)