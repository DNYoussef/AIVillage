
# agents/langroid/agent/communication_protocol.py

import asyncio
import uuid
import datetime
from enum import Enum
from collections import deque
from typing import Dict, Any, Callable, List, Deque

class MessageType(Enum):
    TASK = "TASK"
    QUERY = "QUERY"
    RESPONSE = "RESPONSE"
    UPDATE = "UPDATE"
    COMMAND = "COMMAND"
    BULK_UPDATE = "BULK_UPDATE"
    PROJECT_UPDATE = "PROJECT_UPDATE"
    SYSTEM_STATUS_UPDATE = "SYSTEM_STATUS_UPDATE"
    CONFIG_UPDATE = "CONFIG_UPDATE"

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class Message:
    def __init__(
        self,
        type: MessageType,
        sender: str,
        receiver: str,
        content: Dict[str, Any],
        priority: Priority = Priority.MEDIUM,
        parent_id: str = None,
        metadata: Dict[str, Any] = None,
    ):
        self.id = str(uuid.uuid4())
        self.type = type
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.priority = priority
        self.timestamp = datetime.datetime.utcnow()
        self.parent_id = parent_id
        self.metadata = metadata or {}

    def with_updated_content(self, new_content: Dict[str, Any]) -> 'Message':
        return Message(
            type=self.type,
            sender=self.sender,
            receiver=self.receiver,
            content=new_content,
            priority=self.priority,
            parent_id=self.parent_id,
            metadata=self.metadata
        )

    def with_updated_priority(self, new_priority: Priority) -> 'Message':
        return Message(
            type=self.type,
            sender=self.sender,
            receiver=self.receiver,
            content=self.content,
            priority=new_priority,
            parent_id=self.parent_id,
            metadata=self.metadata
        )

class MessageQueue:
    def __init__(self):
        self.queues: Dict[Priority, Deque[Message]] = {
            Priority.CRITICAL: deque(),
            Priority.HIGH: deque(),
            Priority.MEDIUM: deque(),
            Priority.LOW: deque()
        }

    def enqueue(self, message: Message):
        self.queues[message.priority].append(message)

    def dequeue(self) -> Message:
        for priority in Priority:
            if self.queues[priority]:
                return self.queues[priority].popleft()
        return None

    def is_empty(self) -> bool:
        return all(not queue for queue in self.queues.values())

    def get_messages_by_priority(self, priority: Priority) -> List[Message]:
        return list(self.queues[priority])

    def get_all_messages(self) -> List[Message]:
        messages = []
        for priority in Priority:
            messages.extend(self.queues[priority])
        return messages

class StandardCommunicationProtocol:
    def __init__(self):
        self.message_queue = MessageQueue()
        self.subscribers: Dict[str, List[Callable[[Message], None]]] = {}
        self.message_history: List[Message] = []

    async def send_message(self, message: Message):
        self.message_queue.enqueue(message)
        self.message_history.append(message)
        await self._notify_subscribers(message)

    async def _notify_subscribers(self, message: Message):
        receivers = [message.receiver]
        if message.receiver in self.subscribers:
            for callback in self.subscribers[message.receiver]:
                await callback(message)
        if message.receiver == 'broadcast':
            for agent_id, callbacks in self.subscribers.items():
                if agent_id != message.sender:
                    for callback in callbacks:
                        await callback(message)

    def subscribe(self, agent_id: str, callback: Callable[[Message], None]):
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)

    def unsubscribe(self, agent_id: str, callback: Callable[[Message], None]):
        if agent_id in self.subscribers:
            self.subscribers[agent_id].remove(callback)
            if not self.subscribers[agent_id]:
                del self.subscribers[agent_id]

    async def query(self, sender: str, receiver: str, query_content: Dict[str, Any], priority: Priority = Priority.MEDIUM, timeout: int = 10) -> Any:
        query_message = Message(
            type=MessageType.QUERY,
            sender=sender,
            receiver=receiver,
            content=query_content,
            priority=priority
        )
        await self.send_message(query_message)
        response = await self._wait_for_response(query_message.id, timeout)
        return response

    async def _wait_for_response(self, query_id: str, timeout: int):
        start_time = datetime.datetime.utcnow()
        while (datetime.datetime.utcnow() - start_time).seconds < timeout:
            for message in self.message_history:
                if message.parent_id == query_id and message.type == MessageType.RESPONSE:
                    return message.content
            await asyncio.sleep(0.1)
        return None

    async def broadcast(self, sender: str, message_type: MessageType, content: Dict[str, Any], priority: Priority = Priority.MEDIUM):
        broadcast_message = Message(
            type=message_type,
            sender=sender,
            receiver='broadcast',
            content=content,
            priority=priority
        )
        await self.send_message(broadcast_message)

    def get_message_history(self, agent_id: str, message_type: MessageType = None) -> List[Message]:
        history = [
            msg for msg in self.message_history
            if msg.sender == agent_id or msg.receiver == agent_id
        ]
        if message_type:
            history = [msg for msg in history if msg.type == message_type]
        return history

