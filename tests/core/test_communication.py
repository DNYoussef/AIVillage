"""Tests for core.communication module.

This module tests the communication protocol classes before cleanup to ensure
no regressions during the class renaming process.
"""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from core.communication import (
    AgentCommunicationProtocol,
    AgentMessage,
    AgentMessageType,
    Priority,
)


class TestAgentMessageType:
    """Test AgentMessageType enum."""

    def test_message_type_values(self):
        """Test AgentMessageType has correct values."""
        assert AgentMessageType.TASK.value == "TASK"
        assert AgentMessageType.RESPONSE.value == "RESPONSE"
        assert AgentMessageType.QUERY.value == "QUERY"
        assert AgentMessageType.NOTIFICATION.value == "NOTIFICATION"
        assert AgentMessageType.ERROR.value == "ERROR"
        assert AgentMessageType.SYSTEM.value == "SYSTEM"

    def test_message_type_count(self):
        """Test AgentMessageType has expected number of values."""
        assert len(AgentMessageType) == 6


class TestPriority:
    """Test Priority enum."""

    def test_priority_values(self):
        """Test Priority has correct values."""
        assert Priority.LOW.value == 1
        assert Priority.MEDIUM.value == 2
        assert Priority.HIGH.value == 3
        assert Priority.CRITICAL.value == 4

    def test_priority_ordering(self):
        """Test Priority values are properly ordered."""
        assert Priority.LOW.value < Priority.MEDIUM.value
        assert Priority.MEDIUM.value < Priority.HIGH.value
        assert Priority.HIGH.value < Priority.CRITICAL.value


class TestAgentMessage:
    """Test AgentMessage dataclass."""

    def test_message_creation(self):
        """Test basic message creation."""
        message = AgentMessage(
            type=AgentMessageType.TASK,
            sender="agent1",
            receiver="agent2",
            content="test content",
        )

        assert message.type == AgentMessageType.TASK
        assert message.sender == "agent1"
        assert message.receiver == "agent2"
        assert message.content == "test content"
        assert message.priority == Priority.MEDIUM  # default
        assert isinstance(message.id, str)
        assert isinstance(message.timestamp, datetime)

    def test_message_with_custom_priority(self):
        """Test message with custom priority."""
        message = AgentMessage(
            type=AgentMessageType.ERROR,
            sender="agent1",
            receiver="agent2",
            content="error message",
            priority=Priority.CRITICAL,
        )

        assert message.priority == Priority.CRITICAL

    def test_message_with_parent_id(self):
        """Test message with parent_id."""
        parent_message = AgentMessage(
            type=AgentMessageType.QUERY,
            sender="agent1",
            receiver="agent2",
            content="query",
        )

        response_message = AgentMessage(
            type=AgentMessageType.RESPONSE,
            sender="agent2",
            receiver="agent1",
            content="response",
            parent_id=parent_message.id,
        )

        assert response_message.parent_id == parent_message.id

    def test_message_to_dict(self):
        """Test message serialization to dict."""
        message = AgentMessage(
            type=AgentMessageType.NOTIFICATION,
            sender="system",
            receiver="agent1",
            content={"key": "value"},
            priority=Priority.HIGH,
        )

        data = message.to_dict()

        assert data["type"] == "NOTIFICATION"
        assert data["sender"] == "system"
        assert data["receiver"] == "agent1"
        assert data["content"] == {"key": "value"}
        assert data["priority"] == 3
        assert "id" in data
        assert "timestamp" in data

    def test_message_from_dict(self):
        """Test message deserialization from dict."""
        message = AgentMessage(
            type=AgentMessageType.TASK,
            sender="agent1",
            receiver="agent2",
            content="test",
        )

        data = message.to_dict()
        restored_message = AgentMessage.from_dict(data)

        assert restored_message.type == message.type
        assert restored_message.sender == message.sender
        assert restored_message.receiver == message.receiver
        assert restored_message.content == message.content
        assert restored_message.id == message.id
        assert restored_message.priority == message.priority

    def test_message_roundtrip_serialization(self):
        """Test message serialization roundtrip."""
        original = AgentMessage(
            type=AgentMessageType.SYSTEM,
            sender="system",
            receiver="all",
            content={"type": "shutdown", "reason": "maintenance"},
            priority=Priority.CRITICAL,
            metadata={"urgent": True},
        )

        data = original.to_dict()
        restored = AgentMessage.from_dict(data)

        assert original.type == restored.type
        assert original.sender == restored.sender
        assert original.receiver == restored.receiver
        assert original.content == restored.content
        assert original.priority == restored.priority
        assert original.metadata == restored.metadata


class TestAgentCommunicationProtocol:
    """Test AgentCommunicationProtocol class."""

    def test_protocol_initialization(self):
        """Test protocol initialization."""
        protocol = AgentCommunicationProtocol()

        assert protocol.subscribers == {}
        assert protocol.message_history == []

    def test_subscribe_unsubscribe(self):
        """Test agent subscription and unsubscription."""
        protocol = AgentCommunicationProtocol()
        handler = AsyncMock()

        # Test subscription
        protocol.subscribe("agent1", handler)
        assert "agent1" in protocol.subscribers
        assert protocol.subscribers["agent1"] == handler

        # Test unsubscription
        protocol.unsubscribe("agent1")
        assert "agent1" not in protocol.subscribers

    def test_unsubscribe_nonexistent_agent(self):
        """Test unsubscribing non-existent agent doesn't raise error."""
        protocol = AgentCommunicationProtocol()

        # Should not raise exception
        protocol.unsubscribe("nonexistent_agent")

    @pytest.mark.asyncio
    async def test_send_message(self):
        """Test sending message to subscribed agent."""
        protocol = AgentCommunicationProtocol()
        handler = AsyncMock()
        protocol.subscribe("agent2", handler)

        message = AgentMessage(
            type=AgentMessageType.TASK,
            sender="agent1",
            receiver="agent2",
            content="test task",
        )

        await protocol.send_message(message)

        # Check message was added to history
        assert len(protocol.message_history) == 1
        assert protocol.message_history[0] == message

        # Check handler was called
        handler.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_send_message_no_subscriber(self):
        """Test sending message when receiver is not subscribed."""
        protocol = AgentCommunicationProtocol()

        message = AgentMessage(
            type=AgentMessageType.TASK,
            sender="agent1",
            receiver="nonexistent",
            content="test task",
        )

        await protocol.send_message(message)

        # AgentMessage should still be added to history
        assert len(protocol.message_history) == 1
        assert protocol.message_history[0] == message

    @pytest.mark.asyncio
    async def test_broadcast(self):
        """Test broadcasting message to all subscribers."""
        protocol = AgentCommunicationProtocol()
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        handler3 = AsyncMock()

        protocol.subscribe("agent1", handler1)
        protocol.subscribe("agent2", handler2)
        protocol.subscribe("agent3", handler3)

        message = AgentMessage(
            type=AgentMessageType.SYSTEM,
            sender="system",
            receiver="all",
            content="broadcast message",
        )

        await protocol.broadcast(message)

        # Check all handlers were called
        assert handler1.call_count == 1
        assert handler2.call_count == 1
        assert handler3.call_count == 1

        # Check message was added to history
        assert len(protocol.message_history) == 1

    @pytest.mark.asyncio
    async def test_broadcast_with_exclusions(self):
        """Test broadcasting with excluded agents."""
        protocol = AgentCommunicationProtocol()
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        handler3 = AsyncMock()

        protocol.subscribe("agent1", handler1)
        protocol.subscribe("agent2", handler2)
        protocol.subscribe("agent3", handler3)

        message = AgentMessage(
            type=AgentMessageType.NOTIFICATION,
            sender="system",
            receiver="all",
            content="notification",
        )

        await protocol.broadcast(message, exclude=["agent2"])

        # Check only non-excluded handlers were called
        assert handler1.call_count == 1
        assert handler2.call_count == 0  # excluded
        assert handler3.call_count == 1

    @pytest.mark.asyncio
    async def test_query(self):
        """Test query functionality."""
        protocol = AgentCommunicationProtocol()
        handler = AsyncMock()
        protocol.subscribe("agent2", handler)

        response = await protocol.query("agent1", "agent2", "test query")

        # Check query message was sent
        assert len(protocol.message_history) == 1
        query_message = protocol.message_history[0]
        assert query_message.type == AgentMessageType.QUERY
        assert query_message.sender == "agent1"
        assert query_message.receiver == "agent2"
        assert query_message.content == "test query"
        assert query_message.priority == Priority.HIGH

        # Check handler was called
        handler.assert_called_once_with(query_message)

        # Check response format
        assert response["status"] == "query_sent"
        assert response["query_id"] == query_message.id

    def test_get_message_history_all(self):
        """Test getting all message history."""
        protocol = AgentCommunicationProtocol()

        # Add some messages manually
        msg1 = AgentMessage(AgentMessageType.TASK, "agent1", "agent2", "content1")
        msg2 = AgentMessage(AgentMessageType.RESPONSE, "agent2", "agent1", "content2")
        protocol.message_history = [msg1, msg2]

        history = protocol.get_message_history()

        assert len(history) == 2
        assert history[0] == msg1
        assert history[1] == msg2

    def test_get_message_history_filtered(self):
        """Test getting filtered message history."""
        protocol = AgentCommunicationProtocol()

        # Add messages
        msg1 = AgentMessage(AgentMessageType.TASK, "agent1", "agent2", "content1")
        msg2 = AgentMessage(AgentMessageType.RESPONSE, "agent2", "agent3", "content2")
        msg3 = AgentMessage(AgentMessageType.QUERY, "agent3", "agent1", "content3")
        protocol.message_history = [msg1, msg2, msg3]

        # Get history for agent1
        history = protocol.get_message_history("agent1")

        assert len(history) == 2  # msg1 (sender) and msg3 (receiver)
        assert msg1 in history
        assert msg3 in history
        assert msg2 not in history

    def test_get_message_history_with_limit(self):
        """Test getting message history with limit."""
        protocol = AgentCommunicationProtocol()

        # Add multiple messages
        for i in range(5):
            msg = AgentMessage(AgentMessageType.TASK, f"agent{i}", "target", f"content{i}")
            protocol.message_history.append(msg)

        history = protocol.get_message_history(limit=3)

        assert len(history) == 3
        # Should get the last 3 messages
        assert history[0].content == "content2"
        assert history[1].content == "content3"
        assert history[2].content == "content4"

    def test_clear_history_all(self):
        """Test clearing all message history."""
        protocol = AgentCommunicationProtocol()

        # Add messages
        msg1 = AgentMessage(AgentMessageType.TASK, "agent1", "agent2", "content1")
        msg2 = AgentMessage(AgentMessageType.RESPONSE, "agent2", "agent1", "content2")
        protocol.message_history = [msg1, msg2]

        protocol.clear_history()

        assert len(protocol.message_history) == 0

    def test_clear_history_filtered(self):
        """Test clearing message history for specific agent."""
        protocol = AgentCommunicationProtocol()

        # Add messages
        msg1 = AgentMessage(AgentMessageType.TASK, "agent1", "agent2", "content1")
        msg2 = AgentMessage(AgentMessageType.RESPONSE, "agent2", "agent3", "content2")
        msg3 = AgentMessage(AgentMessageType.QUERY, "agent3", "agent1", "content3")
        protocol.message_history = [msg1, msg2, msg3]

        protocol.clear_history("agent1")

        # Only msg2 should remain (doesn't involve agent1)
        assert len(protocol.message_history) == 1
        assert protocol.message_history[0] == msg2
