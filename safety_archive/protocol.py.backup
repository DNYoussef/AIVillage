from abc import ABC, abstractmethod
import asyncio
from collections.abc import Callable, Coroutine
from threading import Thread
from typing import Any

from .message import Message, MessageType, Priority
from .queue import MessageQueue

try:
    from .a2a_protocol import send_a2a
    from .mcp_client import MCPClient
except Exception:  # pragma: no cover - optional dependencies
    MCPClient = None  # type: ignore

    def send_a2a(*args, **kwargs):
        raise RuntimeError("a2a protocol dependencies not installed")


try:
    from agents.utils.exceptions import AIVillageException
except Exception:  # pragma: no cover - fallback if agents package isn't available

    class AIVillageException(Exception):
        """Custom exception class for AI Village-specific errors."""


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
        content: dict[str, Any],
        priority: Priority = Priority.MEDIUM,
    ) -> Any:
        pass

    @abstractmethod
    async def send_and_wait(self, message: Message, timeout: float = 5.0) -> Message:
        pass

    @abstractmethod
    def subscribe(
        self, agent_id: str, callback: Callable[[Message], Coroutine[Any, Any, None]]
    ) -> None:
        pass


class StandardCommunicationProtocol(CommunicationProtocol):
    def __init__(
        self,
        mcp_client: MCPClient | None = None,
        certs: dict[str, dict[str, str]] | None = None,
        cards: dict[str, dict[str, Any]] | None = None,
    ):
        self.message_queues: dict[str, MessageQueue] = {}
        self.subscribers: dict[
            str, list[Callable[[Message], Coroutine[Any, Any, None]]]
        ] = {}
        self.message_history: dict[str, list[Message]] = {}
        self.mcp = mcp_client
        self.certs = certs or {}
        self.cards = cards or {}
        self._running = False
        self._dispatch_thread: Thread | None = None

    def enqueue(self, message: Message) -> None:
        if message.receiver not in self.message_queues:
            self.message_queues[message.receiver] = MessageQueue()
        self.message_queues[message.receiver].enqueue(message)

    def dequeue(self, agent_id: str) -> Message | None:
        if agent_id in self.message_queues:
            return self.message_queues[agent_id].dequeue()
        return None

    async def _notify_subscribers(self, message: Message) -> None:
        for callback in self.subscribers.get(message.receiver, []):
            await callback(message)

    async def send_message(self, message: Message) -> None:
        self.enqueue(message)
        # Track history for both the sender and receiver so each agent can
        # retrieve the full conversation context.
        self.message_history.setdefault(message.receiver, []).append(message)
        self.message_history.setdefault(message.sender, []).append(message)
        await self._notify_subscribers(message)

        if self.mcp and message.type == MessageType.TOOL_CALL:
            result = self.mcp.call(
                message.content.get("tool_id"),
                message.content.get("args", {}),
            )
            if message.content.get("reply", True):
                resp = Message(
                    type=MessageType.RESPONSE,
                    sender=message.receiver,
                    receiver=message.sender,
                    content={"result": result},
                    parent_id=message.id,
                )
                await self.send_message(resp)
        elif message.receiver in self.certs:
            card = self.cards.get(message.receiver)
            if card:
                url = card.get("service_url", "") + "/inbox"
                priv = self.certs[message.sender]["key"]
                pub = self.certs[message.receiver]["crt"]
                try:
                    send_a2a(url, message.to_dict(), priv, pub)
                except Exception:
                    pass

    async def receive_message(self, agent_id: str) -> Message:
        message = self.dequeue(agent_id)
        if message is None:
            raise AIVillageException(f"No messages for agent {agent_id}")
        return message

    async def query(
        self,
        sender: str,
        receiver: str,
        content: dict[str, Any],
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
                        self.message_queues[message.sender]._queues[
                            resp.priority
                        ].remove(resp)
                        return resp
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        raise AIVillageException("Response timeout")

    def subscribe(
        self, agent_id: str, callback: Callable[[Message], Coroutine[Any, Any, None]]
    ) -> None:
        self.subscribers.setdefault(agent_id, []).append(callback)

    def unsubscribe(
        self, agent_id: str, callback: Callable[[Message], Coroutine[Any, Any, None]]
    ) -> None:
        if agent_id in self.subscribers:
            self.subscribers[agent_id] = [
                cb for cb in self.subscribers[agent_id] if cb != callback
            ]
            if not self.subscribers[agent_id]:
                del self.subscribers[agent_id]

    async def broadcast(
        self,
        sender: str,
        message_type: MessageType,
        content: dict[str, Any],
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

    def get_message_history(
        self, agent_id: str, message_type: MessageType | None = None
    ) -> list[Message]:
        history = self.message_history.get(agent_id, [])
        if message_type is None:
            return list(history)
        return [m for m in history if m.type == message_type]

    async def process_messages(
        self, handler: Callable[[Message], Coroutine[Any, Any, None]]
    ) -> None:
        self._running = True
        while self._running:
            processed = False
            for agent_id, q in list(self.message_queues.items()):
                msg = q.dequeue()
                if msg is not None:
                    await handler(msg)
                    processed = True
            if not processed:
                await asyncio.sleep(0.01)
