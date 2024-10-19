from typing import Dict, Any, Callable, Coroutine, List
from abc import ABC, abstractmethod
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

    def subscribe(self, agent_id: str, callback: Callable[[Message], Coroutine[Any, Any, None]]) -> None:
        self.subscribers[agent_id] = callback
