"""
Backward Compatibility Wrappers

Provides compatibility layer for existing communication systems.
Allows seamless migration to unified messaging architecture.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timezone

from ..message_bus import MessageBus
from ..message_format import (
    UnifiedMessage, MessageType, TransportType,
    create_p2p_message, create_edge_chat_message, 
    create_websocket_message, create_gateway_message
)

logger = logging.getLogger(__name__)


class LegacyMessagePassingSystem:
    """Compatibility wrapper for existing message passing system"""
    
    def __init__(self, agent_id: str, port: int = None):
        self.agent_id = agent_id
        self.port = port or self._find_available_port()
        
        # Initialize unified message bus
        config = {
            "http_port": self.port,
            "enable_p2p": True,
            "enable_discovery": True
        }
        self.message_bus = MessageBus(agent_id, config)
        
        # Legacy handler registry
        self.message_handlers: Dict[str, Callable] = {}
        self.running = False
        
        logger.info(f"Legacy message passing system initialized for {agent_id}")
    
    def _find_available_port(self) -> int:
        """Find an available port for this agent"""
        base_port = 8000
        agent_hash = hash(self.agent_id) % 1000
        return base_port + agent_hash
    
    async def start(self) -> None:
        """Start the legacy message passing system"""
        if self.running:
            return
        
        await self.message_bus.start()
        
        # Register legacy message handler
        self.message_bus.register_handler("agent_request", self._handle_legacy_message)
        self.message_bus.register_handler("p2p_data", self._handle_legacy_message)
        
        self.running = True
        logger.info(f"Legacy message passing system started for {self.agent_id}")
    
    async def stop(self) -> None:
        """Stop the legacy message passing system"""
        if not self.running:
            return
        
        await self.message_bus.stop()
        self.running = False
        logger.info(f"Legacy message passing system stopped for {self.agent_id}")
    
    def register_handler(self, message_type: str, handler: Callable) -> None:
        """Register a handler for specific message types"""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered legacy handler for message type: {message_type}")
    
    async def send_message(
        self,
        target_agent_id: str,
        message_type: str,
        payload: Any,
        metadata: Dict = None,
    ) -> bool:
        """Send a message to a specific agent (legacy interface)"""
        try:
            # Convert to unified message format
            unified_msg = create_p2p_message(
                self.agent_id, target_agent_id, message_type, payload
            )
            
            # Add legacy metadata
            if metadata:
                unified_msg.metadata.update(metadata)
            
            # Send via message bus
            success = await self.message_bus.send(unified_msg)
            logger.debug(f"Legacy message sent to {target_agent_id}: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Error sending legacy message to {target_agent_id}: {e}")
            return False
    
    async def broadcast_message(
        self,
        message_type: str,
        payload: Any,
        service_type: str = None,
        metadata: Dict = None,
    ) -> int:
        """Broadcast a message to all agents (legacy interface)"""
        try:
            # Convert to unified message format
            unified_msg = UnifiedMessage(
                message_type=MessageType.AGENT_BROADCAST,
                transport=TransportType.P2P_LIBP2P,
                source_id=self.agent_id,
                target_id=None,  # Broadcast
                payload={"type": message_type, "data": payload, "service_type": service_type}
            )
            
            # Add legacy metadata
            if metadata:
                unified_msg.metadata.update(metadata)
            
            # Broadcast via message bus
            results = await self.message_bus.broadcast(unified_msg)
            success_count = sum(1 for success in results.values() if success)
            
            logger.info(f"Legacy broadcast sent to {success_count} agents")
            return success_count
            
        except Exception as e:
            logger.error(f"Error broadcasting legacy message: {e}")
            return 0
    
    async def send_request_response(
        self,
        target_agent_id: str,
        request_type: str,
        payload: Any,
        timeout: float = 30.0,
        metadata: Dict = None,
    ) -> Optional[Any]:
        """Send a request and wait for a response (legacy interface)"""
        try:
            # Convert to unified message format
            unified_msg = create_p2p_message(
                self.agent_id, target_agent_id, request_type, payload
            )
            
            # Mark as expecting response
            unified_msg.metadata["expects_response"] = True
            
            # Add legacy metadata
            if metadata:
                unified_msg.metadata.update(metadata)
            
            # Send request and wait for response
            response_msg = await self.message_bus.request_response(unified_msg, timeout)
            
            if response_msg:
                # Extract legacy payload format
                return response_msg.payload.get("data")
            
            return None
            
        except Exception as e:
            logger.error(f"Error in legacy request-response to {target_agent_id}: {e}")
            return None
    
    async def send_response(
        self,
        request_message: Any,  # Could be legacy Message or UnifiedMessage
        response_payload: Any,
        metadata: Dict = None,
    ) -> bool:
        """Send a response to a previous request (legacy interface)"""
        try:
            # Handle both legacy and unified message formats
            if hasattr(request_message, 'source_id'):
                # UnifiedMessage
                target_id = request_message.source_id
                correlation_id = request_message.get_correlation_id()
            else:
                # Legacy message format
                target_id = getattr(request_message, 'sender_id', None)
                correlation_id = getattr(request_message, 'metadata', {}).get('correlation_id')
            
            if not target_id or not correlation_id:
                logger.warning("Cannot send legacy response: missing target or correlation ID")
                return False
            
            # Create response message
            response_msg = UnifiedMessage(
                message_type=MessageType.AGENT_RESPONSE,
                transport=TransportType.P2P_LIBP2P,
                source_id=self.agent_id,
                target_id=target_id,
                payload={"data": response_payload}
            )
            
            # Set response metadata
            response_msg.metadata.update({
                "correlation_id": correlation_id,
                "is_response": True
            })
            
            # Add legacy metadata
            if metadata:
                response_msg.metadata.update(metadata)
            
            # Send response
            success = await self.message_bus.send(response_msg)
            logger.debug(f"Legacy response sent to {target_id}: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Error sending legacy response: {e}")
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information for this agent (legacy interface)"""
        health = asyncio.create_task(self.message_bus.health_check())
        health_data = {}
        
        try:
            # This is a sync method, so we can't await. Return basic info.
            health_data = {
                "node_id": self.message_bus.node_id,
                "running": self.message_bus.running,
                "transports": len(self.message_bus.transports)
            }
        except Exception:
            pass
        
        return {
            "agent_id": self.agent_id,
            "host": "localhost",
            "port": self.port,
            "status": "running" if self.running else "stopped",
            "connections": health_data.get("transports", 0),
            "message_handlers": list(self.message_handlers.keys()),
            **health_data
        }
    
    async def _handle_legacy_message(self, message: UnifiedMessage) -> None:
        """Handle incoming message using legacy handler interface"""
        try:
            # Extract legacy message type and payload
            if message.message_type == MessageType.P2P_DATA:
                legacy_type = message.payload.get("type", "unknown")
                legacy_payload = message.payload.get("data", message.payload)
            else:
                legacy_type = message.message_type.value
                legacy_payload = message.payload.get("data", message.payload)
            
            # Find and call legacy handler
            handler = self.message_handlers.get(legacy_type)
            if handler:
                # Create legacy message object for compatibility
                legacy_message = LegacyMessage(
                    sender_id=message.source_id,
                    recipient_id=message.target_id or self.agent_id,
                    message_type=legacy_type,
                    payload=legacy_payload,
                    metadata=message.metadata
                )
                
                await handler(legacy_message)
            else:
                logger.warning(f"No legacy handler for message type: {legacy_type}")
                
        except Exception as e:
            logger.error(f"Error handling legacy message: {e}")


class LegacyMessage:
    """Legacy message format for backward compatibility"""
    
    def __init__(self, sender_id: str, recipient_id: str, message_type: str, 
                 payload: Any, metadata: Dict = None):
        self.sender_id = sender_id
        self.recipient_id = recipient_id
        self.message_type = message_type
        self.payload = payload
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc)


class LegacyChatEngine:
    """Compatibility wrapper for edge chat engine"""
    
    def __init__(self, node_id: str = "chat_engine"):
        self.node_id = node_id
        
        # Initialize unified message bus
        config = {
            "http_port": 8001,
            "enable_websocket": False,  # Chat engine uses HTTP
            "enable_p2p": False
        }
        self.message_bus = MessageBus(node_id, config)
        
        # Chat-specific handlers
        self.chat_handlers: Dict[str, Callable] = {}
        self.running = False
        
        logger.info("Legacy chat engine initialized")
    
    async def start(self) -> None:
        """Start the legacy chat engine"""
        if self.running:
            return
        
        await self.message_bus.start()
        
        # Register chat message handler
        self.message_bus.register_handler("edge_chat", self._handle_chat_message)
        
        self.running = True
        logger.info("Legacy chat engine started")
    
    async def stop(self) -> None:
        """Stop the legacy chat engine"""
        if not self.running:
            return
        
        await self.message_bus.stop()
        self.running = False
        logger.info("Legacy chat engine stopped")
    
    async def process_chat(self, message: str, conversation_id: str) -> Dict[str, Any]:
        """Process chat message (legacy interface)"""
        try:
            # Create unified chat message
            chat_msg = create_edge_chat_message(conversation_id, message)
            
            # Process through message bus (this would route to actual chat processor)
            response_msg = await self.message_bus.request_response(chat_msg, timeout=30.0)
            
            if response_msg:
                return response_msg.payload
            else:
                # Fallback response
                return {
                    "response": "I received your message but couldn't process it fully.",
                    "conversation_id": conversation_id,
                    "mode": "fallback",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error processing legacy chat: {e}")
            return {
                "response": "Sorry, I encountered an error processing your message.",
                "conversation_id": conversation_id,
                "mode": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _handle_chat_message(self, message: UnifiedMessage) -> None:
        """Handle incoming chat message"""
        # This would be implemented by the actual chat processing logic
        logger.info(f"Received chat message: {message.message_id}")


class LegacyWebSocketHandler:
    """Compatibility wrapper for WebSocket handler"""
    
    def __init__(self, port: int = 8765):
        self.port = port
        
        # Initialize unified message bus with WebSocket transport
        config = {
            "websocket_port": port,
            "enable_http": False,
            "enable_p2p": False
        }
        self.message_bus = MessageBus("websocket_handler", config)
        
        # WebSocket-specific state
        self.connection_handlers: Dict[str, Callable] = {}
        self.running = False
        
        logger.info(f"Legacy WebSocket handler initialized on port {port}")
    
    async def start(self) -> None:
        """Start the legacy WebSocket handler"""
        if self.running:
            return
        
        await self.message_bus.start()
        
        # Register WebSocket message handlers
        self.message_bus.register_handler("ws_message", self._handle_ws_message)
        self.message_bus.register_handler("ws_connect", self._handle_ws_connect)
        self.message_bus.register_handler("ws_disconnect", self._handle_ws_disconnect)
        
        self.running = True
        logger.info(f"Legacy WebSocket handler started on port {self.port}")
    
    async def stop(self) -> None:
        """Stop the legacy WebSocket handler"""
        if not self.running:
            return
        
        await self.message_bus.stop()
        self.running = False
        logger.info("Legacy WebSocket handler stopped")
    
    def register_connection_handler(self, event: str, handler: Callable) -> None:
        """Register handler for WebSocket connection events"""
        self.connection_handlers[event] = handler
        logger.info(f"Registered WebSocket handler for event: {event}")
    
    async def send_to_connection(self, connection_id: str, data: Any) -> bool:
        """Send data to specific WebSocket connection"""
        try:
            # Create WebSocket message
            ws_msg = create_websocket_message(connection_id, data)
            
            # Send via message bus
            success = await self.message_bus.send(ws_msg, "websocket")
            logger.debug(f"WebSocket message sent to {connection_id}: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Error sending WebSocket message to {connection_id}: {e}")
            return False
    
    async def broadcast_to_all(self, data: Any) -> Dict[str, bool]:
        """Broadcast data to all WebSocket connections"""
        try:
            # Create broadcast message
            ws_msg = UnifiedMessage(
                message_type=MessageType.WS_MESSAGE,
                transport=TransportType.WEBSOCKET,
                source_id="websocket_handler",
                target_id=None,  # Broadcast
                payload={"data": data}
            )
            
            # Broadcast via message bus
            results = await self.message_bus.broadcast(ws_msg, ["websocket"])
            logger.info(f"WebSocket broadcast completed: {sum(results.values())} successful")
            return results
            
        except Exception as e:
            logger.error(f"Error broadcasting WebSocket message: {e}")
            return {}
    
    async def _handle_ws_message(self, message: UnifiedMessage) -> None:
        """Handle incoming WebSocket message"""
        handler = self.connection_handlers.get("message")
        if handler:
            await handler(message.source_id, message.payload.get("data"))
    
    async def _handle_ws_connect(self, message: UnifiedMessage) -> None:
        """Handle WebSocket connection"""
        handler = self.connection_handlers.get("connect")
        if handler:
            await handler(message.source_id)
    
    async def _handle_ws_disconnect(self, message: UnifiedMessage) -> None:
        """Handle WebSocket disconnection"""
        handler = self.connection_handlers.get("disconnect")
        if handler:
            await handler(message.source_id)


# Convenience functions for creating legacy wrappers

async def create_legacy_message_system(agent_id: str, port: int = None) -> LegacyMessagePassingSystem:
    """Create and start a legacy message passing system"""
    system = LegacyMessagePassingSystem(agent_id, port)
    await system.start()
    return system


async def create_legacy_chat_engine(node_id: str = "chat_engine") -> LegacyChatEngine:
    """Create and start a legacy chat engine"""
    engine = LegacyChatEngine(node_id)
    await engine.start()
    return engine


async def create_legacy_websocket_handler(port: int = 8765) -> LegacyWebSocketHandler:
    """Create and start a legacy WebSocket handler"""
    handler = LegacyWebSocketHandler(port)
    await handler.start()
    return handler


# Legacy aliases for backward compatibility
MessagePassingSystem = LegacyMessagePassingSystem
MessagePassing = LegacyMessagePassingSystem
