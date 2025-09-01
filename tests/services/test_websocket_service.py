"""
Tests for WebSocketService
"""

import asyncio
import pytest
from datetime import datetime

from infrastructure.gateway.services.websocket_service import (
    WebSocketService,
    WebSocketMessage,
    ConnectionState,
    MessageType,
    ConnectionInfo,
)


class MockWebSocket:
    """Mock WebSocket for testing."""

    def __init__(self):
        self.accepted = False
        self.closed = False
        self.sent_messages = []

    async def accept(self):
        self.accepted = True

    async def close(self):
        self.closed = True

    async def send_json(self, data):
        self.sent_messages.append(data)

    async def receive_text(self):
        return '{"type": "ping"}'


@pytest.fixture
async def websocket_service():
    """Create WebSocket service for testing."""
    service = WebSocketService(ping_interval=1, cleanup_interval=1)
    await service.start()
    yield service
    await service.stop()


@pytest.fixture
def mock_websocket():
    """Create mock WebSocket."""
    return MockWebSocket()


@pytest.mark.asyncio
async def test_websocket_connection(websocket_service, mock_websocket):
    """Test WebSocket connection management."""
    connection_id = await websocket_service.connect(mock_websocket)

    assert connection_id in websocket_service.connections
    assert mock_websocket.accepted
    assert len(mock_websocket.sent_messages) == 1

    # Check connection established message
    message = mock_websocket.sent_messages[0]
    assert message["type"] == MessageType.CONNECTION_ESTABLISHED.value
    assert message["connection_id"] == connection_id


@pytest.mark.asyncio
async def test_websocket_disconnect(websocket_service, mock_websocket):
    """Test WebSocket disconnection."""
    connection_id = await websocket_service.connect(mock_websocket)
    assert connection_id in websocket_service.connections

    await websocket_service.disconnect(connection_id)
    assert connection_id not in websocket_service.connections


@pytest.mark.asyncio
async def test_topic_subscription(websocket_service, mock_websocket):
    """Test topic subscription functionality."""
    connection_id = await websocket_service.connect(mock_websocket)

    # Subscribe to topic
    success = await websocket_service.subscribe(connection_id, "training")
    assert success
    assert "training" in websocket_service.connections[connection_id].subscription_topics
    assert connection_id in websocket_service.topic_subscriptions["training"]

    # Unsubscribe from topic
    success = websocket_service.unsubscribe(connection_id, "training")
    assert success
    assert "training" not in websocket_service.connections[connection_id].subscription_topics


@pytest.mark.asyncio
async def test_broadcast_message(websocket_service, mock_websocket):
    """Test message broadcasting."""
    connection_id = await websocket_service.connect(mock_websocket)
    await websocket_service.subscribe(connection_id, "test_topic")

    # Clear connection established message
    mock_websocket.sent_messages.clear()

    # Broadcast to topic
    message = WebSocketMessage(
        type=MessageType.TRAINING_PROGRESS, data={"progress": 0.5, "epoch": 10}, timestamp=datetime.now().isoformat()
    )

    await websocket_service.broadcast(message, topic="test_topic")

    # Check message was sent
    assert len(mock_websocket.sent_messages) == 1
    sent_message = mock_websocket.sent_messages[0]
    assert sent_message["type"] == MessageType.TRAINING_PROGRESS.value
    assert sent_message["data"]["progress"] == 0.5


@pytest.mark.asyncio
async def test_direct_message_send(websocket_service, mock_websocket):
    """Test sending message to specific connection."""
    connection_id = await websocket_service.connect(mock_websocket)

    # Clear connection established message
    mock_websocket.sent_messages.clear()

    message = WebSocketMessage(
        type=MessageType.STATUS_UPDATE, data={"status": "active"}, timestamp=datetime.now().isoformat()
    )

    success = await websocket_service.send_to_connection(connection_id, message)
    assert success
    assert len(mock_websocket.sent_messages) == 1


@pytest.mark.asyncio
async def test_handle_ping_message(websocket_service, mock_websocket):
    """Test handling ping messages."""
    connection_id = await websocket_service.connect(mock_websocket)

    # Clear connection established message
    mock_websocket.sent_messages.clear()

    await websocket_service.handle_message(connection_id, '{"type": "ping"}')

    # Should receive pong response
    assert len(mock_websocket.sent_messages) == 1
    response = mock_websocket.sent_messages[0]
    assert response["type"] == MessageType.PONG.value


@pytest.mark.asyncio
async def test_handle_subscribe_message(websocket_service, mock_websocket):
    """Test handling subscription messages."""
    connection_id = await websocket_service.connect(mock_websocket)

    # Clear connection established message
    mock_websocket.sent_messages.clear()

    await websocket_service.handle_message(connection_id, '{"type": "subscribe", "topic": "training"}')

    # Check subscription was created
    assert "training" in websocket_service.connections[connection_id].subscription_topics
    assert len(mock_websocket.sent_messages) == 1
    response = mock_websocket.sent_messages[0]
    assert response['data']['success'] is True


@pytest.mark.asyncio
async def test_connection_cleanup(websocket_service, mock_websocket):
    """Test connection cleanup functionality."""
    connection_id = await websocket_service.connect(mock_websocket)

    # Manually mark connection as error state
    websocket_service.connections[connection_id].state = ConnectionState.ERROR

    # Wait for cleanup cycle
    await asyncio.sleep(1.5)  # Cleanup interval is 1 second

    # Connection should be removed
    assert connection_id not in websocket_service.connections


def test_websocket_message_serialization():
    """Test WebSocketMessage serialization."""
    message = WebSocketMessage(
        type=MessageType.TRAINING_STARTED,
        data={"model": "test", "session_id": "123"},
        timestamp=datetime.now().isoformat(),
        connection_id="conn123",
    )

    serialized = message.to_dict()
    assert serialized["type"] == "training_started"
    assert serialized["data"]["model"] == "test"
    assert serialized["connection_id"] == "conn123"


def test_connection_info_is_alive():
    """Test ConnectionInfo alive check."""
    mock_ws = MockWebSocket()

    # Connected connection should be alive
    conn_info = ConnectionInfo(
        id="test",
        websocket=mock_ws,
        state=ConnectionState.CONNECTED,
        connected_at=datetime.now(),
        last_ping=datetime.now(),
        subscription_topics=set(),
        metadata={},
    )
    assert conn_info.is_alive()

    # Disconnected connection should not be alive
    conn_info.state = ConnectionState.DISCONNECTED
    assert not conn_info.is_alive()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__])
