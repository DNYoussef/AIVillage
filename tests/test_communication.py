"""Tests for enhanced communication system."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
from datetime import datetime

from communications.protocol import StandardCommunicationProtocol, GroupCommunication
from communications.message import Message, MessageType, Priority
from communications.queue import MessageQueue
from agents.utils.exceptions import AIVillageException

@pytest.fixture
def protocol():
    """Create StandardCommunicationProtocol instance."""
    return StandardCommunicationProtocol()

@pytest.fixture
def message_queue():
    """Create MessageQueue instance."""
    return MessageQueue()

@pytest.fixture
def group_communication(protocol):
    """Create GroupCommunication instance."""
    return GroupCommunication(protocol)

@pytest.fixture
def test_message():
    """Create test message."""
    return Message(
        type=MessageType.TASK,
        sender="test_sender",
        receiver="test_receiver",
        content={"test": "content"},
        priority=Priority.MEDIUM
    )

@pytest.mark.asyncio
async def test_message_sending(protocol, test_message):
    """Test basic message sending."""
    # Add subscriber
    received_messages = []
    async def message_handler(message: Message):
        received_messages.append(message)
    
    protocol.subscribe(test_message.receiver, message_handler)
    
    # Send message
    await protocol.send_message(test_message)
    
    # Verify message was received
    assert len(received_messages) == 1
    assert received_messages[0].id == test_message.id
    assert received_messages[0].content == test_message.content

@pytest.mark.asyncio
async def test_message_queue_priority(message_queue):
    """Test message queue priority handling."""
    # Create messages with different priorities
    high_priority = Message(
        type=MessageType.TASK,
        sender="sender",
        receiver="receiver",
        content={"priority": "high"},
        priority=Priority.HIGH
    )
    
    medium_priority = Message(
        type=MessageType.TASK,
        sender="sender",
        receiver="receiver",
        content={"priority": "medium"},
        priority=Priority.MEDIUM
    )
    
    low_priority = Message(
        type=MessageType.TASK,
        sender="sender",
        receiver="receiver",
        content={"priority": "low"},
        priority=Priority.LOW
    )
    
    # Add messages in reverse priority order
    message_queue.enqueue(low_priority)
    message_queue.enqueue(medium_priority)
    message_queue.enqueue(high_priority)
    
    # Verify messages are dequeued in priority order
    first = message_queue.dequeue()
    second = message_queue.dequeue()
    third = message_queue.dequeue()
    
    assert first.priority == Priority.HIGH
    assert second.priority == Priority.MEDIUM
    assert third.priority == Priority.LOW

@pytest.mark.asyncio
async def test_group_communication(group_communication):
    """Test group communication functionality."""
    # Create group
    group_id = "test_group"
    members = ["agent1", "agent2", "agent3"]
    config = {"broadcast_enabled": True}
    
    await group_communication.create_group(group_id, members, config)
    
    # Verify group creation
    assert group_id in group_communication.groups
    assert all(member in group_communication.groups[group_id] for member in members)
    
    # Test member management
    await group_communication.add_to_group(group_id, "agent4")
    assert "agent4" in group_communication.groups[group_id]
    
    await group_communication.remove_from_group(group_id, "agent4")
    assert "agent4" not in group_communication.groups[group_id]
    
    # Test group queries
    agent_groups = group_communication.get_agent_groups("agent1")
    assert group_id in agent_groups
    
    group_members = group_communication.get_group_members(group_id)
    assert all(member in group_members for member in members)

@pytest.mark.asyncio
async def test_message_broadcasting(protocol):
    """Test message broadcasting functionality."""
    # Create test recipients
    recipients = ["agent1", "agent2", "agent3"]
    received_messages: Dict[str, List[Message]] = {agent: [] for agent in recipients}
    
    # Set up subscribers
    for agent in recipients:
        async def make_handler(agent_id):
            async def handler(message: Message):
                received_messages[agent_id].append(message)
            return handler
        protocol.subscribe(agent, await make_handler(agent))
    
    # Create broadcast message
    broadcast_message = Message(
        type=MessageType.NOTIFICATION,
        sender="broadcaster",
        receiver="",  # Will be set for each recipient
        content={"broadcast": "test"},
        priority=Priority.HIGH
    )
    
    # Broadcast message
    await protocol.broadcast(broadcast_message, recipients)
    
    # Verify each recipient received the message
    for agent in recipients:
        assert len(received_messages[agent]) == 1
        assert received_messages[agent][0].content == broadcast_message.content

@pytest.mark.asyncio
async def test_query_response(protocol):
    """Test query and response functionality."""
    # Set up responder
    async def responder(message: Message):
        if message.type == MessageType.QUERY:
            response = Message(
                type=MessageType.RESPONSE,
                sender="responder",
                receiver=message.sender,
                content={"answer": 42},
                parent_id=message.id,
                priority=message.priority
            )
            await protocol.send_message(response)
    
    protocol.subscribe("responder", responder)
    
    # Send query and get response
    response = await protocol.query(
        sender="querier",
        receiver="responder",
        content={"question": "meaning of life"}
    )
    
    # Verify response
    assert response["answer"] == 42

@pytest.mark.asyncio
async def test_message_history(protocol, test_message):
    """Test message history tracking."""
    # Send several messages
    for i in range(3):
        message = Message(
            type=MessageType.TASK,
            sender=f"sender{i}",
            receiver=f"receiver{i}",
            content={"index": i},
            priority=Priority.MEDIUM
        )
        await protocol.send_message(message)
    
    # Get communication stats
    stats = protocol.get_communication_stats()
    
    # Verify stats
    assert stats["messages_sent"] == 3
    assert stats["message_history_length"] == 3
    assert stats["failed_deliveries"] == 0

@pytest.mark.asyncio
async def test_queue_management(protocol):
    """Test message queue management."""
    receiver = "test_receiver"
    
    # Send messages with different priorities
    messages = [
        Message(
            type=MessageType.TASK,
            sender="sender",
            receiver=receiver,
            content={"index": i},
            priority=priority
        )
        for i, priority in enumerate([
            Priority.LOW,
            Priority.MEDIUM,
            Priority.HIGH
        ])
    ]
    
    for message in messages:
        await protocol.send_message(message)
    
    # Get queue status
    status = protocol.get_queue_status(receiver)
    
    # Verify queue status
    assert status["high_priority"] == 1
    assert status["medium_priority"] == 1
    assert status["low_priority"] == 1
    assert status["total"] == 3

@pytest.mark.asyncio
async def test_error_handling(protocol):
    """Test error handling in communication."""
    # Test invalid receiver
    with pytest.raises(AIVillageException):
        await protocol.receive_message("nonexistent_agent")
    
    # Test message to unsubscribed agent
    message = Message(
        type=MessageType.TASK,
        sender="sender",
        receiver="unsubscribed",
        content={"test": "content"},
        priority=Priority.MEDIUM
    )
    
    # Should not raise exception but update failed deliveries
    await protocol.send_message(message)
    stats = protocol.get_communication_stats()
    assert stats["failed_deliveries"] > 0

@pytest.mark.asyncio
async def test_group_config(group_communication):
    """Test group configuration handling."""
    group_id = "test_group"
    members = ["agent1", "agent2"]
    config = {
        "broadcast_enabled": True,
        "priority_level": Priority.HIGH,
        "max_members": 5
    }
    
    # Create group with config
    await group_communication.create_group(group_id, members, config)
    
    # Verify config was stored
    assert group_communication.group_configs[group_id] == config
    
    # Test config constraints
    with pytest.raises(AIVillageException):
        # Try to add more members than allowed
        for i in range(5):
            await group_communication.add_to_group(group_id, f"agent{i+3}")

if __name__ == "__main__":
    pytest.main([__file__])
