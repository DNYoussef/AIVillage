from typing import Dict, Any, Callable, Coroutine, List, Optional
from abc import ABC, abstractmethod
import asyncio
from .message import Message, MessageType, Priority
from .queue import MessageQueue
try:
    from agents.utils.exceptions import AIVillageException
except Exception:  # pragma: no cover - fallback if agents package isn't available
    class AIVillageException(Exception):
        """Custom exception class for AI Village-specific errors."""
        pass

class CommunicationProtocol(ABC):
    @abstractmethod
    async def send_message(self, message: Message) -> None:
        pass

    @abstractmethod
    async def receive_message(self, agent_id: str) -> Message:
        pass

    @abstractmethod
    async def query(
        self,
        sender: str,
        receiver: str,
        content: Dict[str, Any],
        priority: Priority = Priority.MEDIUM,
    ) -> Any:
        pass

    @abstractmethod
    async def send_and_wait(self, message: Message, timeout: float = 5.0) -> Message:
        pass

    @abstractmethod
    def subscribe(self, agent_id: str, callback: Callable[[Message], Coroutine[Any, Any, None]]) -> None:
        pass

class StandardCommunicationProtocol(CommunicationProtocol):
    def __init__(self):
        self.message_queues: Dict[str, MessageQueue] = {}
        self.subscribers: Dict[str, List[Callable[[Message], Coroutine[Any, Any, None]]]] = {}

    def enqueue(self, message: Message) -> None:
        if message.receiver not in self.message_queues:
            self.message_queues[message.receiver] = MessageQueue()
        self.message_queues[message.receiver].enqueue(message)

    def dequeue(self, agent_id: str) -> Optional[Message]:
        if agent_id in self.message_queues:
            return self.message_queues[agent_id].dequeue()
        return None

    async def _notify_subscribers(self, message: Message) -> None:
        for callback in self.subscribers.get(message.receiver, []):
            await callback(message)

    async def send_message(self, message: Message) -> None:
        self.enqueue(message)
        await self._notify_subscribers(message)

    async def receive_message(self, agent_id: str) -> Message:
        message = self.dequeue(agent_id)
        if message is None:
            raise AIVillageException(f"No messages for agent {agent_id}")
        return message

    async def query(
        self,
        sender: str,
        receiver: str,
        content: Dict[str, Any],
        priority: Priority = Priority.MEDIUM,
    ) -> Any:
        query_message = Message(
            type=MessageType.QUERY,
            sender=sender,
            receiver=receiver,
            content=content,
            priority=priority,
        )
        await self.send_message(query_message)
        response = await self.receive_message(sender)
        return response.content

    async def send_and_wait(self, message: Message, timeout: float = 5.0) -> Message:
        """Send a message and wait for a response matching the message id."""
        await self.send_message(message)
        elapsed = 0.0
        poll_interval = 0.1
        while elapsed < timeout:
            if message.sender in self.message_queues:
                queue = self.message_queues[message.sender]
                for idx, resp in enumerate(queue.get_all_messages()):
                    if resp.parent_id == message.id or resp.id == message.id:
                        # remove specific response from queue
                        self.message_queues[message.sender]._queues[resp.priority].remove(resp)
                        return resp
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        raise AIVillageException("Response timeout")

    def subscribe(self, agent_id: str, callback: Callable[[Message], Coroutine[Any, Any, None]]) -> None:
        self.subscribers.setdefault(agent_id, []).append(callback)

    def unsubscribe(self, agent_id: str, callback: Callable[[Message], Coroutine[Any, Any, None]]) -> None:
        if agent_id in self.subscribers:
            self.subscribers[agent_id] = [cb for cb in self.subscribers[agent_id] if cb != callback]
            if not self.subscribers[agent_id]:
                del self.subscribers[agent_id]

    async def broadcast(
        self,
        sender: str,
        message_type: MessageType,
        content: Dict[str, Any],
        priority: Priority = Priority.MEDIUM,
    ) -> None:
        for agent_id in list(self.subscribers.keys()):
            msg = Message(
                type=message_type,
                sender=sender,
                receiver=agent_id,
                content=content,
                priority=priority,
            )
            await self.send_message(msg)
