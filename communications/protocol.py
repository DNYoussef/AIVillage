import asyncio
from typing import Dict, List, Callable, Any, Optional
from .message import Message, MessageType, Priority
from .queue import MessageQueue
from ..utils.exceptions import AIVillageException
from ..utils.logger import logger

class StandardCommunicationProtocol:
    def __init__(self):
        self.message_queue = MessageQueue()
        self.subscribers: Dict[str, List[Callable]] = {}

    async def send_message(self, message: Message) -> None:
        self.message_queue.enqueue(message)
        await self._notify_subscribers(message)

    async def _notify_subscribers(self, message: Message) -> None:
        if message.receiver in self.subscribers:
            for callback in self.subscribers[message.receiver]:
                try:
                    await callback(message)
                except Exception as e:
                    logger.error(f"Error notifying subscriber for {message.receiver}: {str(e)}")

    async def get_next_message(self) -> Optional[Message]:
        return self.message_queue.dequeue()

    def subscribe(self, agent_id: str, callback: Callable) -> None:
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)

    def unsubscribe(self, agent_id: str, callback: Callable) -> None:
        if agent_id in self.subscribers and callback in self.subscribers[agent_id]:
            self.subscribers[agent_id].remove(callback)

    async def query(self, sender: str, receiver: str, query: str, priority: Priority = Priority.MEDIUM) -> Message:
        query_message = Message(
            type=MessageType.QUERY,
            sender=sender,
            receiver=receiver,
            content={"query": query},
            priority=priority
        )
        await self.send_message(query_message)
        return await self._wait_for_response(query_message.id)

    async def _wait_for_response(self, query_id: str, timeout: float = 30.0) -> Message:
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            message = await self.get_next_message()
            if message and message.type == MessageType.RESPONSE and message.parent_id == query_id:
                return message
            elif message:
                # Re-enqueue messages that don't match our criteria
                self.message_queue.enqueue(message)
            await asyncio.sleep(0.1)
        raise AIVillageException(f"Timeout waiting for response to query {query_id}")

    async def broadcast(self, sender: str, message_type: MessageType, content: Dict[str, Any], priority: Priority = Priority.MEDIUM) -> None:
        for receiver in self.subscribers.keys():
            message = Message(
                type=message_type,
                sender=sender,
                receiver=receiver,
                content=content,
                priority=priority
            )
            await self.send_message(message)

    def get_message_history(self, agent_id: str = None, message_type: Optional[MessageType] = None) -> List[Message]:
        all_messages = self.message_queue.get_all_messages()
        return [
            msg for msg in all_messages
            if (agent_id is None or msg.sender == agent_id or msg.receiver == agent_id)
            and (message_type is None or msg.type == message_type)
        ]

    async def process_messages(self, handler: Callable[[Message], Awaitable[None]]) -> None:
        while True:
            message = await self.get_next_message()
            if message:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
            else:
                await asyncio.sleep(0.1)