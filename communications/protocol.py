from typing import Dict, Any, Callable, Coroutine, List
from abc import ABC, abstractmethod
import asyncio
from .message import Message, MessageType, Priority
from agents.utils.exceptions import AIVillageException

class CommunicationProtocol(ABC):
    @abstractmethod
    async def send_message(self, message: Message) -> None:
        pass

    @abstractmethod
    async def receive_message(self, agent_id: str) -> Message:
        pass

    @abstractmethod
    async def query(self, sender: str, receiver: str, content: Dict[str, Any]) -> Any:
        pass

    @abstractmethod
    async def send_and_wait(self, message: Message, timeout: float = 5.0) -> Message:
        pass

    @abstractmethod
    def subscribe(self, agent_id: str, callback: Callable[[Message], Coroutine[Any, Any, None]]) -> None:
        pass

class StandardCommunicationProtocol(CommunicationProtocol):
    def __init__(self):
        self.message_queue: Dict[str, List[Message]] = {}
        self.subscribers: Dict[str, Callable[[Message], Coroutine[Any, Any, None]]] = {}

    async def send_message(self, message: Message) -> None:
        if message.receiver not in self.message_queue:
            self.message_queue[message.receiver] = []
        self.message_queue[message.receiver].append(message)

        if message.receiver in self.subscribers:
            await self.subscribers[message.receiver](message)

    async def receive_message(self, agent_id: str) -> Message:
        if agent_id not in self.message_queue or not self.message_queue[agent_id]:
            raise AIVillageException(f"No messages for agent {agent_id}")
        return self.message_queue[agent_id].pop(0)

    async def query(self, sender: str, receiver: str, content: Dict[str, Any]) -> Any:
        query_message = Message(
            type=MessageType.QUERY,
            sender=sender,
            receiver=receiver,
            content=content
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
            if message.sender in self.message_queue:
                queue = self.message_queue[message.sender]
                for idx, resp in enumerate(queue):
                    if resp.parent_id == message.id or resp.id == message.id:
                        return queue.pop(idx)
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        raise AIVillageException("Response timeout")

    def subscribe(self, agent_id: str, callback: Callable[[Message], Coroutine[Any, Any, None]]) -> None:
        self.subscribers[agent_id] = callback
