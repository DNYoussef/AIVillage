"""
Message Handler Interface and Registry.

Defines how components register to handle specific message types
and provides a registry for handler management.
"""

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
import logging

from ..core.message import Message, MessageType

logger = logging.getLogger(__name__)


class MessageHandler(ABC):
    """
    Abstract base class for message handlers.
    
    Handlers can process specific message types and optionally
    generate response messages.
    """
    
    @abstractmethod
    async def handle(self, message: Message) -> Optional[Message]:
        """
        Handle an incoming message.
        
        Args:
            message: The message to process
            
        Returns:
            Optional response message to send back
        """
        pass
    
    @abstractmethod 
    def get_handled_types(self) -> Set[MessageType]:
        """
        Get the message types this handler can process.
        
        Returns:
            Set of MessageType values this handler supports
        """
        pass
    
    def can_handle(self, message: Message) -> bool:
        """
        Check if this handler can process the given message.
        
        Args:
            message: Message to check
            
        Returns:
            True if handler can process this message type
        """
        return message.type in self.get_handled_types()


@dataclass
class HandlerRegistration:
    """Registration information for a message handler."""
    
    handler_id: str
    handler: MessageHandler
    message_types: Set[MessageType] 
    priority: int = 0  # Higher values processed first
    active: bool = True
    

class MessageHandlerRegistry:
    """
    Registry for managing message handlers across the system.
    
    Consolidates handler patterns from all communication systems:
    - P2P message handlers
    - Agent protocol handlers  
    - Chat engine handlers
    - Twin communication handlers
    - Infrastructure handlers
    """
    
    def __init__(self):
        self._handlers: Dict[MessageType, List[HandlerRegistration]] = {}
        self._handler_lookup: Dict[str, HandlerRegistration] = {}
        self._global_handlers: List[HandlerRegistration] = []
    
    def register_handler(
        self,
        handler_id: str,
        handler: MessageHandler,
        priority: int = 0,
        message_types: Optional[Set[MessageType]] = None
    ) -> bool:
        """
        Register a message handler.
        
        Args:
            handler_id: Unique identifier for the handler
            handler: The handler instance
            priority: Processing priority (higher = first)
            message_types: Specific types to handle, or None for handler's types
            
        Returns:
            True if registration successful
        """
        if handler_id in self._handler_lookup:
            logger.warning(f"Handler {handler_id} already registered")
            return False
        
        # Use handler's types if not specified
        if message_types is None:
            message_types = handler.get_handled_types()
        
        registration = HandlerRegistration(
            handler_id=handler_id,
            handler=handler,
            message_types=message_types,
            priority=priority,
            active=True
        )
        
        # Add to lookup
        self._handler_lookup[handler_id] = registration
        
        # Add to type-specific lists
        for msg_type in message_types:
            if msg_type not in self._handlers:
                self._handlers[msg_type] = []
            
            self._handlers[msg_type].append(registration)
            # Keep sorted by priority (descending)
            self._handlers[msg_type].sort(key=lambda x: x.priority, reverse=True)
        
        logger.info(f"Registered handler {handler_id} for types: {[t.value for t in message_types]}")
        return True
    
    def register_global_handler(
        self,
        handler_id: str, 
        handler: MessageHandler,
        priority: int = 0
    ) -> bool:
        """
        Register a global handler that processes all message types.
        
        Args:
            handler_id: Unique identifier for the handler
            handler: The handler instance
            priority: Processing priority
            
        Returns:
            True if registration successful
        """
        if handler_id in self._handler_lookup:
            logger.warning(f"Handler {handler_id} already registered")
            return False
        
        registration = HandlerRegistration(
            handler_id=handler_id,
            handler=handler,
            message_types=set(),  # Empty for global
            priority=priority,
            active=True
        )
        
        self._handler_lookup[handler_id] = registration
        self._global_handlers.append(registration)
        # Keep global handlers sorted by priority
        self._global_handlers.sort(key=lambda x: x.priority, reverse=True)
        
        logger.info(f"Registered global handler {handler_id}")
        return True
    
    def unregister_handler(self, handler_id: str) -> bool:
        """
        Unregister a message handler.
        
        Args:
            handler_id: ID of handler to remove
            
        Returns:
            True if handler was found and removed
        """
        if handler_id not in self._handler_lookup:
            return False
        
        registration = self._handler_lookup[handler_id]
        
        # Remove from type-specific lists
        for msg_type in registration.message_types:
            if msg_type in self._handlers:
                self._handlers[msg_type] = [
                    h for h in self._handlers[msg_type] 
                    if h.handler_id != handler_id
                ]
        
        # Remove from global handlers
        self._global_handlers = [
            h for h in self._global_handlers 
            if h.handler_id != handler_id
        ]
        
        # Remove from lookup
        del self._handler_lookup[handler_id]
        
        logger.info(f"Unregistered handler {handler_id}")
        return True
    
    def get_handlers(self, message: Message) -> List[HandlerRegistration]:
        """
        Get all handlers that can process the given message.
        
        Args:
            message: Message to find handlers for
            
        Returns:
            List of handler registrations sorted by priority
        """
        handlers = []
        
        # Add type-specific handlers
        if message.type in self._handlers:
            handlers.extend([
                h for h in self._handlers[message.type] 
                if h.active
            ])
        
        # Add global handlers that can handle this message
        for registration in self._global_handlers:
            if registration.active and registration.handler.can_handle(message):
                handlers.append(registration)
        
        # Sort by priority (descending)
        handlers.sort(key=lambda x: x.priority, reverse=True)
        return handlers
    
    def get_handler_count(self, message_type: Optional[MessageType] = None) -> int:
        """
        Get count of registered handlers.
        
        Args:
            message_type: Specific type to count, or None for all
            
        Returns:
            Number of handlers
        """
        if message_type is None:
            return len(self._handler_lookup)
        
        return len(self._handlers.get(message_type, []))
    
    def list_handlers(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered handlers with their information.
        
        Returns:
            Dict mapping handler IDs to handler info
        """
        handlers = {}
        
        for handler_id, registration in self._handler_lookup.items():
            handlers[handler_id] = {
                "handler_id": handler_id,
                "handler_class": type(registration.handler).__name__,
                "message_types": [t.value for t in registration.message_types],
                "priority": registration.priority,
                "active": registration.active,
                "is_global": handler_id in [h.handler_id for h in self._global_handlers]
            }
        
        return handlers
    
    def activate_handler(self, handler_id: str) -> bool:
        """Activate a registered handler."""
        if handler_id in self._handler_lookup:
            self._handler_lookup[handler_id].active = True
            return True
        return False
    
    def deactivate_handler(self, handler_id: str) -> bool:
        """Deactivate a registered handler."""
        if handler_id in self._handler_lookup:
            self._handler_lookup[handler_id].active = False
            return True 
        return False
    
    def clear_handlers(self) -> None:
        """Remove all registered handlers."""
        self._handlers.clear()
        self._handler_lookup.clear()
        self._global_handlers.clear()
        logger.info("All handlers cleared from registry")


# Utility functions for common handler patterns

class FunctionHandler(MessageHandler):
    """
    Simple wrapper to use async functions as message handlers.
    
    Provides backward compatibility for existing handler functions.
    """
    
    def __init__(
        self, 
        handler_func: Callable[[Message], Awaitable[Optional[Message]]],
        handled_types: Set[MessageType],
        handler_name: Optional[str] = None
    ):
        self._handler_func = handler_func
        self._handled_types = handled_types
        self._name = handler_name or f"FunctionHandler_{id(handler_func)}"
    
    async def handle(self, message: Message) -> Optional[Message]:
        """Execute the wrapped function."""
        return await self._handler_func(message)
    
    def get_handled_types(self) -> Set[MessageType]:
        """Return the configured message types."""
        return self._handled_types
    
    def __str__(self) -> str:
        return self._name


def create_function_handler(
    handler_func: Callable[[Message], Awaitable[Optional[Message]]],
    message_types: List[MessageType],
    handler_name: Optional[str] = None
) -> FunctionHandler:
    """
    Create a handler from an async function.
    
    Args:
        handler_func: Async function that takes Message and returns Optional[Message]
        message_types: List of MessageType values the function handles
        handler_name: Optional name for the handler
        
    Returns:
        FunctionHandler instance
    """
    return FunctionHandler(
        handler_func=handler_func,
        handled_types=set(message_types),
        handler_name=handler_name
    )