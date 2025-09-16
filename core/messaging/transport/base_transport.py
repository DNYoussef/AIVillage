"""
Base Transport Interface

Abstract base class for all transport implementations in the unified
messaging system. Defines the contract that all transports must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
import asyncio
import logging

from ..message_format import UnifiedMessage

logger = logging.getLogger(__name__)


class TransportState:
    """Transport operational states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class BaseTransport(ABC):
    """Abstract base class for all transport implementations"""
    
    def __init__(self, node_id: str, config: Dict[str, Any]):
        self.node_id = node_id
        self.config = config
        self.state = TransportState.STOPPED
        self.message_handler: Optional[Callable] = None
        self.running = False
        
        # Metrics
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "connection_errors": 0,
            "send_errors": 0,
            "connections_active": 0
        }
        
        logger.info(f"Transport initialized: {self.__class__.__name__} for node {node_id}")
    
    @abstractmethod
    async def start(self) -> None:
        """Start the transport layer"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the transport layer"""
        pass
    
    @abstractmethod
    async def send(self, message: UnifiedMessage, target: str) -> bool:
        """Send message to target"""
        pass
    
    @abstractmethod
    async def broadcast(self, message: UnifiedMessage) -> Dict[str, bool]:
        """Broadcast message to all connected peers"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check transport health status"""
        pass
    
    def set_message_handler(self, handler: Callable) -> None:
        """Set message handler for incoming messages"""
        self.message_handler = handler
        logger.debug(f"Message handler set for {self.__class__.__name__}")
    
    async def handle_incoming_message(self, message: UnifiedMessage) -> None:
        """Handle incoming message by calling registered handler"""
        if self.message_handler:
            try:
                await self.message_handler(message)
                self.metrics["messages_received"] += 1
            except Exception as e:
                logger.error(f"Error in message handler: {e}")
        else:
            logger.warning(f"No message handler set for {self.__class__.__name__}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get transport metrics"""
        return self.metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset transport metrics"""
        for key in self.metrics:
            self.metrics[key] = 0
        logger.debug(f"Metrics reset for {self.__class__.__name__}")
    
    def _update_connection_count(self, delta: int) -> None:
        """Update active connection count"""
        self.metrics["connections_active"] = max(0, self.metrics["connections_active"] + delta)
    
    def _record_send_success(self) -> None:
        """Record successful message send"""
        self.metrics["messages_sent"] += 1
    
    def _record_send_error(self) -> None:
        """Record message send error"""
        self.metrics["send_errors"] += 1
    
    def _record_connection_error(self) -> None:
        """Record connection error"""
        self.metrics["connection_errors"] += 1
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(node_id='{self.node_id}', state='{self.state}', running={self.running})"
