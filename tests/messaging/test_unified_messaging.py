"""
Comprehensive Tests for Unified Messaging System

Tests consolidation of 5 communication systems into unified architecture.
Validates backward compatibility and transport layer functionality.
"""

import asyncio
import pytest
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from core.messaging.message_bus import MessageBus, MessageBusState
from core.messaging.message_format import (
    UnifiedMessage, MessageType, TransportType,
    create_p2p_message, create_edge_chat_message
)
from core.messaging.transport.base_transport import BaseTransport
from core.messaging.reliability.circuit_breaker import CircuitBreaker, CircuitState
from core.messaging.compatibility.legacy_wrappers import (
    LegacyMessagePassingSystem, LegacyChatEngine, LegacyWebSocketHandler
)


class MockTransport(BaseTransport):
    """Mock transport for testing"""
    
    def __init__(self, node_id: str, config: Dict[str, Any] = None):
        super().__init__(node_id, config or {})
        self.sent_messages = []
        self.broadcast_results = {}
        self.should_fail = False
    
    async def start(self) -> None:
        self.running = True
        self.state = "running"
    
    async def stop(self) -> None:
        self.running = False
        self.state = "stopped"
    
    async def send(self, message: UnifiedMessage, target: str) -> bool:
        if self.should_fail:
            self._record_send_error()
            return False
        
        self.sent_messages.append((message, target))
        self._record_send_success()
        return True
    
    async def broadcast(self, message: UnifiedMessage) -> Dict[str, bool]:
        if self.should_fail:
            return {"peer1": False, "peer2": False}
        
        results = {"peer1": True, "peer2": True}
        self.broadcast_results[message.message_id] = results
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self.running else "stopped",
            "sent_messages": len(self.sent_messages)
        }


@pytest.fixture
async def message_bus():
    """Create message bus for testing"""
    config = {"test_mode": True}
    bus = MessageBus("test_node", config)
    
    # Add mock transport
    mock_transport = MockTransport("test_node")
    await bus.register_transport("mock", mock_transport)
    
    await bus.start()
    yield bus
    await bus.stop()


@pytest.fixture
def unified_message():
    """Create test unified message"""
    return UnifiedMessage(
        message_type=MessageType.AGENT_REQUEST,
        transport=TransportType.HTTP,
        source_id="test_sender",
        target_id="test_target",
        payload={"data": "test message"}
    )


class TestUnifiedMessageFormat:
    """Test unified message format functionality"""
    
    def test_message_creation(self):
        """Test creating unified messages"""
        message = UnifiedMessage(
            message_type=MessageType.AGENT_REQUEST,
            transport=TransportType.P2P_LIBP2P,
            source_id="sender",
            target_id="target",
            payload={"test": "data"}
        )
        
        assert message.message_type == MessageType.AGENT_REQUEST
        assert message.transport == TransportType.P2P_LIBP2P
        assert message.source_id == "sender"
        assert message.target_id == "target"
        assert message.payload["test"] == "data"
        assert message.message_id is not None
        assert message.timestamp is not None
    
    def test_message_serialization(self):
        """Test message serialization and deserialization"""
        original = UnifiedMessage(
            message_type=MessageType.P2P_DATA,
            transport=TransportType.WEBSOCKET,
            source_id="test_source",
            target_id="test_target",
            payload={"key": "value"}
        )
        
        # Test dict conversion
        message_dict = original.to_dict()
        assert message_dict["message_type"] == "p2p_data"
        assert message_dict["transport"] == "websocket"
        assert message_dict["source_id"] == "test_source"
        
        # Test reconstruction
        reconstructed = UnifiedMessage.from_dict(message_dict)
        assert reconstructed.message_type == original.message_type
        assert reconstructed.transport == original.transport
        assert reconstructed.source_id == original.source_id
        assert reconstructed.payload == original.payload
    
    def test_message_response_creation(self):
        """Test creating response messages"""
        request = UnifiedMessage(
            message_type=MessageType.AGENT_REQUEST,
            transport=TransportType.HTTP,
            source_id="client",
            target_id="server",
            payload={"request": "data"}
        )
        
        response = request.create_response({"response": "data"})
        
        assert response.message_type == MessageType.AGENT_RESPONSE
        assert response.source_id == "server"
        assert response.target_id == "client"
        assert response.payload["response"] == "data"
        assert response.metadata["correlation_id"] == request.message_id
        assert response.metadata["is_response"] is True
    
    def test_legacy_message_creation(self):
        """Test legacy message creation functions"""
        # Test P2P message creation
        p2p_msg = create_p2p_message("sender", "receiver", "test_type", {"data": "test"})
        assert p2p_msg.message_type == MessageType.P2P_DATA
        assert p2p_msg.transport == TransportType.P2P_LIBP2P
        assert p2p_msg.source_id == "sender"
        
        # Test edge chat message creation
        chat_msg = create_edge_chat_message("conv_123", "Hello world")
        assert chat_msg.message_type == MessageType.EDGE_CHAT
        assert chat_msg.transport == TransportType.HTTP
        assert chat_msg.payload["conversation_id"] == "conv_123"
        assert chat_msg.payload["prompt"] == "Hello world"


class TestMessageBus:
    """Test unified message bus functionality"""
    
    @pytest.mark.asyncio
    async def test_message_bus_lifecycle(self, message_bus):
        """Test message bus startup and shutdown"""
        assert message_bus.running is True
        assert message_bus.state == MessageBusState.RUNNING
        assert len(message_bus.transports) == 1
        
        await message_bus.stop()
        assert message_bus.running is False
        assert message_bus.state == MessageBusState.STOPPED
    
    @pytest.mark.asyncio
    async def test_message_sending(self, message_bus, unified_message):
        """Test sending messages through message bus"""
        # Send message
        success = await message_bus.send(unified_message)
        assert success is True
        
        # Check message was sent to transport
        mock_transport = message_bus.transports["mock"]
        assert len(mock_transport.sent_messages) == 1
        
        sent_message, target = mock_transport.sent_messages[0]
        assert sent_message.message_id == unified_message.message_id
        assert target == unified_message.target_id
    
    @pytest.mark.asyncio
    async def test_message_broadcasting(self, message_bus, unified_message):
        """Test broadcasting messages"""
        # Broadcast message
        results = await message_bus.broadcast(unified_message)
        assert len(results) == 1
        assert results["mock"] is True
        
        # Check broadcast was sent to transport
        mock_transport = message_bus.transports["mock"]
        assert unified_message.message_id in mock_transport.broadcast_results
    
    @pytest.mark.asyncio
    async def test_request_response(self, message_bus):
        """Test request-response messaging pattern"""
        # Create request message
        request = UnifiedMessage(
            message_type=MessageType.AGENT_REQUEST,
            transport=TransportType.HTTP,
            source_id="client",
            target_id="server",
            payload={"request": "test"}
        )
        
        # Mock response handler
        response_received = False
        
        async def response_handler(message: UnifiedMessage):
            nonlocal response_received
            if message.is_request():
                response = message.create_response({"response": "test_response"})
                await message_bus.send(response)
                response_received = True
        
        message_bus.register_handler("agent_request", response_handler)
        
        # Send request (this will timeout in test, but we can test the setup)
        try:
            response = await message_bus.request_response(request, timeout=0.1)
        except:
            pass  # Expected timeout in test
        
        # Verify request was marked correctly
        assert request.metadata["expects_response"] is True
        assert request.metadata["correlation_id"] == request.message_id
    
    @pytest.mark.asyncio
    async def test_message_handler_registration(self, message_bus):
        """Test message handler registration and execution"""
        handled_messages = []
        
        async def test_handler(message: UnifiedMessage):
            handled_messages.append(message)
        
        # Register handler
        message_bus.register_handler("agent_request", test_handler)
        
        # Create and handle message
        test_message = UnifiedMessage(
            message_type=MessageType.AGENT_REQUEST,
            transport=TransportType.HTTP,
            source_id="test",
            target_id="handler_test",
            payload={"test": True}
        )
        
        await message_bus._handle_incoming_message(test_message)
        
        # Verify handler was called
        assert len(handled_messages) == 1
        assert handled_messages[0].message_id == test_message.message_id
    
    @pytest.mark.asyncio
    async def test_health_check(self, message_bus):
        """Test message bus health check"""
        health = await message_bus.health_check()
        
        assert health["node_id"] == "test_node"
        assert health["running"] is True
        assert health["state"] == MessageBusState.RUNNING.value
        assert "transports" in health
        assert "metrics" in health
        assert "mock" in health["transports"]


class TestCircuitBreaker:
    """Test circuit breaker reliability pattern"""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_normal_operation(self):
        """Test circuit breaker in normal operation"""
        config = {"failure_threshold": 3, "timeout_seconds": 60}
        cb = CircuitBreaker(config)
        
        # Test successful operations
        async def success_func():
            return "success"
        
        result = await cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.total_successes == 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_handling(self):
        """Test circuit breaker failure handling"""
        config = {"failure_threshold": 2, "timeout_seconds": 60}
        cb = CircuitBreaker(config)
        
        # Function that always fails
        async def fail_func():
            raise Exception("Test failure")
        
        # First failure
        with pytest.raises(Exception):
            await cb.call(fail_func)
        
        assert cb.failure_count == 1
        assert cb.state == CircuitState.CLOSED
        
        # Second failure should open circuit
        with pytest.raises(Exception):
            await cb.call(fail_func)
        
        assert cb.failure_count == 2
        assert cb.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state(self):
        """Test circuit breaker in open state"""
        config = {"failure_threshold": 1, "timeout_seconds": 60}
        cb = CircuitBreaker(config)
        
        # Force circuit to open
        async def fail_func():
            raise Exception("Test failure")
        
        with pytest.raises(Exception):
            await cb.call(fail_func)
        
        assert cb.state == CircuitState.OPEN
        
        # Subsequent calls should fail fast
        from core.messaging.reliability.circuit_breaker import CircuitBreakerOpenError
        
        with pytest.raises(CircuitBreakerOpenError):
            await cb.call(lambda: "should not execute")
    
    def test_circuit_breaker_stats(self):
        """Test circuit breaker statistics"""
        config = {"failure_threshold": 5}
        cb = CircuitBreaker(config)
        
        stats = cb.get_stats()
        assert stats["state"] == "closed"
        assert stats["total_calls"] == 0
        assert stats["success_rate"] == 0.0
        assert "configuration" in stats


class TestLegacyCompatibility:
    """Test backward compatibility wrappers"""
    
    @pytest.mark.asyncio
    async def test_legacy_message_passing_system(self):
        """Test legacy message passing system wrapper"""
        legacy_system = LegacyMessagePassingSystem("test_agent")
        
        # Mock the underlying message bus
        legacy_system.message_bus = Mock()
        legacy_system.message_bus.start = AsyncMock()
        legacy_system.message_bus.stop = AsyncMock()
        legacy_system.message_bus.send = AsyncMock(return_value=True)
        legacy_system.message_bus.broadcast = AsyncMock(return_value={"peer1": True})
        legacy_system.message_bus.register_handler = Mock()
        
        # Test lifecycle
        await legacy_system.start()
        assert legacy_system.running is True
        legacy_system.message_bus.start.assert_called_once()
        
        # Test message sending
        success = await legacy_system.send_message("target", "test_type", {"data": "test"})
        assert success is True
        legacy_system.message_bus.send.assert_called_once()
        
        # Test broadcasting
        count = await legacy_system.broadcast_message("broadcast_type", {"data": "broadcast"})
        assert count == 1
        legacy_system.message_bus.broadcast.assert_called_once()
        
        await legacy_system.stop()
        assert legacy_system.running is False
    
    @pytest.mark.asyncio
    async def test_legacy_chat_engine(self):
        """Test legacy chat engine wrapper"""
        chat_engine = LegacyChatEngine()
        
        # Mock the underlying message bus
        chat_engine.message_bus = Mock()
        chat_engine.message_bus.start = AsyncMock()
        chat_engine.message_bus.stop = AsyncMock()
        chat_engine.message_bus.request_response = AsyncMock(
            return_value=UnifiedMessage(
                message_type=MessageType.EDGE_CHAT,
                transport=TransportType.HTTP,
                source_id="chat_engine",
                target_id="client",
                payload={"response": "Test response"}
            )
        )
        chat_engine.message_bus.register_handler = Mock()
        
        # Test lifecycle
        await chat_engine.start()
        assert chat_engine.running is True
        
        # Test chat processing
        result = await chat_engine.process_chat("Hello", "conv_123")
        assert "response" in result
        assert result["response"] == "Test response"
        
        await chat_engine.stop()
        assert chat_engine.running is False
    
    @pytest.mark.asyncio
    async def test_legacy_websocket_handler(self):
        """Test legacy WebSocket handler wrapper"""
        ws_handler = LegacyWebSocketHandler(8765)
        
        # Mock the underlying message bus
        ws_handler.message_bus = Mock()
        ws_handler.message_bus.start = AsyncMock()
        ws_handler.message_bus.stop = AsyncMock()
        ws_handler.message_bus.send = AsyncMock(return_value=True)
        ws_handler.message_bus.broadcast = AsyncMock(return_value={"conn1": True, "conn2": True})
        ws_handler.message_bus.register_handler = Mock()
        
        # Test lifecycle
        await ws_handler.start()
        assert ws_handler.running is True
        
        # Test connection messaging
        success = await ws_handler.send_to_connection("conn_123", {"message": "test"})
        assert success is True
        ws_handler.message_bus.send.assert_called_once()
        
        # Test broadcasting
        results = await ws_handler.broadcast_to_all({"broadcast": "message"})
        assert len(results) == 2
        ws_handler.message_bus.broadcast.assert_called_once()
        
        await ws_handler.stop()
        assert ws_handler.running is False


class TestTransportIntegration:
    """Test transport layer integration"""
    
    @pytest.mark.asyncio
    async def test_transport_registration(self):
        """Test transport registration and management"""
        config = {"test_mode": True}
        bus = MessageBus("test_node", config)
        
        # Register mock transport
        mock_transport = MockTransport("test_node")
        await bus.register_transport("test_transport", mock_transport)
        
        assert "test_transport" in bus.transports
        assert bus.transports["test_transport"] == mock_transport
        
        # Start bus and verify transport is started
        await bus.start()
        assert mock_transport.running is True
        
        # Unregister transport
        success = await bus.unregister_transport("test_transport")
        assert success is True
        assert "test_transport" not in bus.transports
        assert mock_transport.running is False
        
        await bus.stop()
    
    @pytest.mark.asyncio
    async def test_transport_failure_handling(self, message_bus, unified_message):
        """Test handling of transport failures"""
        # Configure transport to fail
        mock_transport = message_bus.transports["mock"]
        mock_transport.should_fail = True
        
        # Attempt to send message
        success = await message_bus.send(unified_message)
        assert success is False
        
        # Check failure was recorded
        assert mock_transport.metrics["send_errors"] > 0
        assert message_bus.metrics["messages_failed"] > 0
    
    @pytest.mark.asyncio
    async def test_multiple_transport_routing(self):
        """Test routing messages across multiple transports"""
        config = {"test_mode": True}
        bus = MessageBus("test_node", config)
        
        # Register multiple transports
        http_transport = MockTransport("test_node", {"type": "http"})
        ws_transport = MockTransport("test_node", {"type": "websocket"})
        
        await bus.register_transport("http", http_transport)
        await bus.register_transport("websocket", ws_transport)
        await bus.start()
        
        # Test broadcasting to multiple transports
        message = UnifiedMessage(
            message_type=MessageType.AGENT_BROADCAST,
            transport=TransportType.HTTP,
            source_id="broadcaster",
            target_id=None,
            payload={"broadcast": "data"}
        )
        
        results = await bus.broadcast(message)
        assert len(results) == 2
        assert "http" in results
        assert "websocket" in results
        
        await bus.stop()


if __name__ == "__main__":
    # Run specific test categories
    import sys
    
    if len(sys.argv) > 1:
        test_category = sys.argv[1]
        if test_category == "message_format":
            pytest.main(["-v", "test_unified_messaging.py::TestUnifiedMessageFormat"])
        elif test_category == "message_bus":
            pytest.main(["-v", "test_unified_messaging.py::TestMessageBus"])
        elif test_category == "circuit_breaker":
            pytest.main(["-v", "test_unified_messaging.py::TestCircuitBreaker"])
        elif test_category == "legacy":
            pytest.main(["-v", "test_unified_messaging.py::TestLegacyCompatibility"])
        elif test_category == "transport":
            pytest.main(["-v", "test_unified_messaging.py::TestTransportIntegration"])
        else:
            pytest.main(["-v", "test_unified_messaging.py"])
    else:
        pytest.main(["-v", "test_unified_messaging.py"])
