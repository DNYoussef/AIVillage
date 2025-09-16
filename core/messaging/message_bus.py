"""
Unified Message Bus Implementation

Central message bus controller that consolidates all communication systems
according to Agent 5's messaging architecture blueprint.

Key Features:
- Single point of control for all messaging
- Transport abstraction layer
- Circuit breaker protection
- Message routing and delivery
- Backward compatibility support
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List, Set
from datetime import datetime, timezone
from enum import Enum

from .message_format import UnifiedMessage, MessageType, TransportType
from .transport.base_transport import BaseTransport
from .serialization.json_serializer import JsonSerializer
from .serialization.msgpack_serializer import MessagePackSerializer
from .reliability.circuit_breaker import CircuitBreaker
from .routing.message_router import MessageRouter

logger = logging.getLogger(__name__)


class MessageBusState(Enum):
    """Message bus operational states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class MessageBus:
    """Unified message bus consolidating all communication systems"""
    
    def __init__(self, node_id: str, config: Dict[str, Any]):
        self.node_id = node_id
        self.config = config
        self.state = MessageBusState.STOPPED
        
        # Core components
        self.transports: Dict[str, BaseTransport] = {}
        self.router = MessageRouter(node_id)
        self.circuit_breaker = CircuitBreaker(config.get("circuit_breaker", {}))
        
        # Serialization management
        self.serializers = {
            "json": JsonSerializer(),
            "msgpack": MessagePackSerializer()
        }
        self.transport_serializers = {
            TransportType.HTTP: "json",        # API compatibility
            TransportType.WEBSOCKET: "msgpack", # Performance
            TransportType.P2P_LIBP2P: "msgpack", # Bandwidth
            TransportType.P2P_DIRECT: "msgpack",  # Bandwidth
            TransportType.LOCAL: "json"        # Debugging
        }
        
        # Message handling
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.response_futures: Dict[str, asyncio.Future] = {}
        
        # Metrics
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_failed": 0,
            "transport_errors": 0,
            "circuit_breaker_trips": 0
        }
        
        # Internal state
        self.running = False
        self.startup_time: Optional[datetime] = None
        self.shutdown_callbacks: List[Callable] = []
        
        logger.info(f"MessageBus initialized for node: {node_id}")
    
    # Core messaging methods
    
    async def send(self, message: UnifiedMessage, transport: str = "auto") -> bool:
        """Send message using specified or auto-selected transport"""
        if not self.running:
            logger.error("Cannot send message: MessageBus is not running")
            return False
        
        try:
            # Auto-select transport if needed
            if transport == "auto":
                transport = self._select_transport(message)
            
            # Get transport instance
            transport_instance = self.transports.get(transport)
            if not transport_instance:
                logger.error(f"Transport not available: {transport}")
                self.metrics["transport_errors"] += 1
                return False
            
            # Use circuit breaker protection
            success = await self.circuit_breaker.call(
                self._send_with_transport, transport_instance, message
            )
            
            if success:
                self.metrics["messages_sent"] += 1
                logger.debug(f"Message sent successfully: {message.message_id}")
            else:
                self.metrics["messages_failed"] += 1
                logger.warning(f"Message send failed: {message.message_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending message {message.message_id}: {e}")
            self.metrics["messages_failed"] += 1
            return False
    
    async def broadcast(self, message: UnifiedMessage, 
                      transport_filter: List[str] = None) -> Dict[str, bool]:
        """Broadcast message to all or filtered transports"""
        if not self.running:
            logger.error("Cannot broadcast: MessageBus is not running")
            return {}
        
        results = {}
        target_transports = transport_filter or list(self.transports.keys())
        
        for transport_name in target_transports:
            if transport_name in self.transports:
                try:
                    transport_instance = self.transports[transport_name]
                    success = await self.circuit_breaker.call(
                        transport_instance.broadcast, message
                    )
                    results[transport_name] = success
                    
                    if success:
                        self.metrics["messages_sent"] += 1
                    else:
                        self.metrics["messages_failed"] += 1
                        
                except Exception as e:
                    logger.error(f"Error broadcasting on {transport_name}: {e}")
                    results[transport_name] = False
                    self.metrics["transport_errors"] += 1
            else:
                results[transport_name] = False
        
        logger.info(f"Broadcast completed: {sum(results.values())}/{len(results)} successful")
        return results
    
    async def request_response(self, message: UnifiedMessage, 
                             timeout: float = 30.0) -> Optional[UnifiedMessage]:
        """Send request and wait for response"""
        if not self.running:
            logger.error("Cannot send request: MessageBus is not running")
            return None
        
        # Mark message as expecting response
        message.metadata["expects_response"] = True
        message.metadata["correlation_id"] = message.message_id
        
        # Create future for response
        response_future = asyncio.Future()
        self.response_futures[message.message_id] = response_future
        
        try:
            # Send request
            success = await self.send(message)
            if not success:
                return None
            
            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
            
        except asyncio.TimeoutError:
            logger.warning(f"Request timeout after {timeout}s: {message.message_id}")
            return None
        except Exception as e:
            logger.error(f"Error in request-response: {e}")
            return None
        finally:
            # Cleanup response future
            self.response_futures.pop(message.message_id, None)
    
    # Transport management
    
    async def register_transport(self, name: str, transport: BaseTransport) -> None:
        """Register a transport with the message bus"""
        if name in self.transports:
            logger.warning(f"Transport {name} already registered, replacing")
        
        # Set message handler for incoming messages
        transport.set_message_handler(self._handle_incoming_message)
        
        # Add to transport registry
        self.transports[name] = transport
        
        # Start transport if bus is running
        if self.running:
            await transport.start()
        
        logger.info(f"Transport registered: {name}")
    
    async def unregister_transport(self, name: str) -> bool:
        """Unregister a transport from the message bus"""
        if name not in self.transports:
            logger.warning(f"Transport not found: {name}")
            return False
        
        transport = self.transports[name]
        
        # Stop transport if running
        if self.running:
            await transport.stop()
        
        # Remove from registry
        del self.transports[name]
        
        logger.info(f"Transport unregistered: {name}")
        return True
    
    async def start_transports(self) -> None:
        """Start all registered transports"""
        for name, transport in self.transports.items():
            try:
                await transport.start()
                logger.info(f"Transport started: {name}")
            except Exception as e:
                logger.error(f"Failed to start transport {name}: {e}")
                self.state = MessageBusState.ERROR
                raise
    
    async def stop_transports(self) -> None:
        """Stop all registered transports"""
        for name, transport in self.transports.items():
            try:
                await transport.stop()
                logger.info(f"Transport stopped: {name}")
            except Exception as e:
                logger.error(f"Error stopping transport {name}: {e}")
    
    # Handler registration
    
    def register_handler(self, message_type: str, handler: Callable) -> None:
        """Register handler for specific message types"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        
        self.message_handlers[message_type].append(handler)
        logger.info(f"Handler registered for message type: {message_type}")
    
    def unregister_handler(self, message_type: str, handler: Callable) -> None:
        """Unregister handler for specific message types"""
        if message_type in self.message_handlers:
            try:
                self.message_handlers[message_type].remove(handler)
                logger.info(f"Handler unregistered for message type: {message_type}")
            except ValueError:
                logger.warning(f"Handler not found for message type: {message_type}")
    
    def clear_handlers(self, message_type: str) -> None:
        """Clear all handlers for a message type"""
        if message_type in self.message_handlers:
            del self.message_handlers[message_type]
            logger.info(f"All handlers cleared for message type: {message_type}")
    
    # Lifecycle management
    
    async def start(self) -> None:
        """Start the message bus and all transports"""
        if self.running:
            logger.warning("MessageBus is already running")
            return
        
        logger.info(f"Starting MessageBus for node: {self.node_id}")
        self.state = MessageBusState.STARTING
        
        try:
            # Start router
            await self.router.start()
            
            # Start all transports
            await self.start_transports()
            
            # Update state
            self.running = True
            self.state = MessageBusState.RUNNING
            self.startup_time = datetime.now(timezone.utc)
            
            logger.info(f"MessageBus started successfully for node: {self.node_id}")
            
        except Exception as e:
            logger.error(f"Failed to start MessageBus: {e}")
            self.state = MessageBusState.ERROR
            raise
    
    async def stop(self) -> None:
        """Stop the message bus and all transports"""
        if not self.running:
            logger.warning("MessageBus is not running")
            return
        
        logger.info(f"Stopping MessageBus for node: {self.node_id}")
        self.state = MessageBusState.STOPPING
        
        try:
            # Execute shutdown callbacks
            for callback in self.shutdown_callbacks:
                try:
                    await callback()
                except Exception as e:
                    logger.error(f"Error in shutdown callback: {e}")
            
            # Stop all transports
            await self.stop_transports()
            
            # Stop router
            await self.router.stop()
            
            # Cancel pending response futures
            for future in self.response_futures.values():
                if not future.done():
                    future.cancel()
            self.response_futures.clear()
            
            # Update state
            self.running = False
            self.state = MessageBusState.STOPPED
            
            logger.info(f"MessageBus stopped successfully for node: {self.node_id}")
            
        except Exception as e:
            logger.error(f"Error stopping MessageBus: {e}")
            self.state = MessageBusState.ERROR
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        transport_health = {}
        
        for name, transport in self.transports.items():
            try:
                transport_health[name] = await transport.health_check()
            except Exception as e:
                transport_health[name] = {"status": "error", "error": str(e)}
        
        return {
            "node_id": self.node_id,
            "state": self.state.value,
            "running": self.running,
            "uptime_seconds": (
                (datetime.now(timezone.utc) - self.startup_time).total_seconds()
                if self.startup_time else 0
            ),
            "transports": transport_health,
            "metrics": self.metrics.copy(),
            "circuit_breaker": self.circuit_breaker.get_stats(),
            "message_handlers": {
                msg_type: len(handlers) 
                for msg_type, handlers in self.message_handlers.items()
            },
            "pending_responses": len(self.response_futures)
        }
    
    def add_shutdown_callback(self, callback: Callable) -> None:
        """Add callback to execute during shutdown"""
        self.shutdown_callbacks.append(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset all metrics to zero"""
        for key in self.metrics:
            self.metrics[key] = 0
        logger.info("Metrics reset")
    
    # Internal methods
    
    def _select_transport(self, message: UnifiedMessage) -> str:
        """Auto-select optimal transport for message"""
        # Use transport specified in message if available
        transport_name = message.transport.value
        if transport_name in self.transports:
            return transport_name
        
        # Fallback selection based on message type
        if message.message_type in [MessageType.HTTP_REQUEST, MessageType.HTTP_RESPONSE]:
            return "http" if "http" in self.transports else list(self.transports.keys())[0]
        elif message.message_type in [MessageType.WS_CONNECT, MessageType.WS_MESSAGE, MessageType.WS_DISCONNECT]:
            return "websocket" if "websocket" in self.transports else list(self.transports.keys())[0]
        elif message.message_type in [MessageType.P2P_DISCOVERY, MessageType.P2P_DATA, MessageType.P2P_HEARTBEAT]:
            return "p2p" if "p2p" in self.transports else list(self.transports.keys())[0]
        else:
            # Default to first available transport
            return list(self.transports.keys())[0] if self.transports else "local"
    
    async def _send_with_transport(self, transport: BaseTransport, 
                                 message: UnifiedMessage) -> bool:
        """Send message using specific transport with serialization"""
        try:
            # Serialize message if needed
            serializer_name = self.transport_serializers.get(message.transport, "json")
            serializer = self.serializers[serializer_name]
            
            # Send via transport
            return await transport.send(message, message.target_id or "")
            
        except Exception as e:
            logger.error(f"Transport send failed: {e}")
            raise
    
    async def _handle_incoming_message(self, message: UnifiedMessage) -> None:
        """Handle incoming message from transports"""
        try:
            self.metrics["messages_received"] += 1
            
            # Check if this is a response to a pending request
            correlation_id = message.get_correlation_id()
            if correlation_id and correlation_id in self.response_futures:
                future = self.response_futures[correlation_id]
                if not future.done():
                    future.set_result(message)
                return
            
            # Route to registered handlers
            message_type = message.message_type.value
            handlers = self.message_handlers.get(message_type, [])
            
            if handlers:
                for handler in handlers:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(f"Error in message handler: {e}")
            else:
                logger.warning(f"No handlers registered for message type: {message_type}")
            
        except Exception as e:
            logger.error(f"Error handling incoming message: {e}")
    
    def __repr__(self) -> str:
        return f"MessageBus(node_id='{self.node_id}', state='{self.state.value}', transports={len(self.transports)})"
