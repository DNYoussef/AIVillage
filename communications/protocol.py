from typing import Dict, Any, Callable, Coroutine, List, Set, Optional
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
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

@dataclass
class PriorityMessageQueue:
    """Priority-based message queue implementation."""
    def __init__(self):
        self.high_priority: List[Message] = []
        self.normal_priority: List[Message] = []
        self.low_priority: List[Message] = []
        self.last_processed: datetime = datetime.now()

    def add_message(self, message: Message):
        """Add a message to the appropriate priority queue."""
        if message.priority == Priority.HIGH:
            self.high_priority.append(message)
        elif message.priority == Priority.NORMAL:
            self.normal_priority.append(message)
        else:
            self.low_priority.append(message)

    def get_next_message(self) -> Optional[Message]:
        """Get the next message based on priority."""
        if self.high_priority:
            self.last_processed = datetime.now()
            return self.high_priority.pop(0)
        if self.normal_priority:
            self.last_processed = datetime.now()
            return self.normal_priority.pop(0)
        if self.low_priority:
            self.last_processed = datetime.now()
            return self.low_priority.pop(0)
        return None

    def has_messages(self) -> bool:
        """Check if there are any messages in the queue."""
        return bool(self.high_priority or self.normal_priority or self.low_priority)

    def get_queue_stats(self) -> Dict[str, int]:
        """Get statistics about the queue state."""
        return {
            'high_priority': len(self.high_priority),
            'normal_priority': len(self.normal_priority),
            'low_priority': len(self.low_priority),
            'total': len(self.high_priority) + len(self.normal_priority) + len(self.low_priority)
        }

class GroupCommunication:
    """Implements group communication functionality."""
    def __init__(self, communication_protocol: 'StandardCommunicationProtocol'):
        self.protocol = communication_protocol
        self.groups: Dict[str, Set[str]] = {}
        self.agent_groups: Dict[str, Set[str]] = {}  # Maps agents to their groups
        self.group_configs: Dict[str, Dict[str, Any]] = {}

    async def create_group(self, group_id: str, members: List[str], config: Optional[Dict[str, Any]] = None):
        """
        Create a new communication group.

        :param group_id: Unique identifier for the group.
        :param members: List of agent IDs to include in the group.
        :param config: Optional group configuration.
        """
        if group_id in self.groups:
            raise AIVillageException(f"Group {group_id} already exists")

        self.groups[group_id] = set(members)
        self.group_configs[group_id] = config or {}

        # Update agent_groups mapping
        for member in members:
            if member not in self.agent_groups:
                self.agent_groups[member] = set()
            self.agent_groups[member].add(group_id)

    async def broadcast_to_group(self, group_id: str, message: Message):
        """
        Broadcast message to all group members.

        :param group_id: ID of the target group.
        :param message: Message to broadcast.
        """
        if group_id not in self.groups:
            raise AIVillageException(f"Group {group_id} does not exist")

        for member in self.groups[group_id]:
            message_copy = message.copy()
            message_copy.receiver = member
            await self.protocol.send_message(message_copy)

    async def add_to_group(self, group_id: str, agent_id: str):
        """
        Add an agent to a group.

        :param group_id: ID of the target group.
        :param agent_id: ID of the agent to add.
        """
        if group_id not in self.groups:
            raise AIVillageException(f"Group {group_id} does not exist")

        self.groups[group_id].add(agent_id)
        if agent_id not in self.agent_groups:
            self.agent_groups[agent_id] = set()
        self.agent_groups[agent_id].add(group_id)

    async def remove_from_group(self, group_id: str, agent_id: str):
        """
        Remove an agent from a group.

        :param group_id: ID of the target group.
        :param agent_id: ID of the agent to remove.
        """
        if group_id not in self.groups:
            raise AIVillageException(f"Group {group_id} does not exist")

        self.groups[group_id].discard(agent_id)
        if agent_id in self.agent_groups:
            self.agent_groups[agent_id].discard(group_id)

    def get_agent_groups(self, agent_id: str) -> Set[str]:
        """
        Get all groups an agent belongs to.

        :param agent_id: ID of the agent.
        :return: Set of group IDs.
        """
        return self.agent_groups.get(agent_id, set())

    def get_group_members(self, group_id: str) -> Set[str]:
        """
        Get all members of a group.

        :param group_id: ID of the group.
        :return: Set of agent IDs.
        """
        if group_id not in self.groups:
            raise AIVillageException(f"Group {group_id} does not exist")
        return self.groups[group_id]

class StandardCommunicationProtocol(CommunicationProtocol):
    """Enhanced communication protocol with priority handling and group communication."""
    def __init__(self):
        self.message_queues: Dict[str, PriorityMessageQueue] = {}
        self.subscribers: Dict[str, Callable[[Message], Coroutine[Any, Any, None]]] = {}
        self.group_communication = GroupCommunication(self)
        self.message_history: List[Message] = []
        self.stats: Dict[str, Any] = {
            'messages_sent': 0,
            'messages_received': 0,
            'failed_deliveries': 0
        }

    async def send_message(self, message: Message) -> None:
        """
        Send a message with priority handling.

        :param message: Message to send.
        """
        try:
            if message.receiver not in self.message_queues:
                self.message_queues[message.receiver] = PriorityMessageQueue()

            self.message_queues[message.receiver].add_message(message)
            self.message_history.append(message)
            self.stats['messages_sent'] += 1

            if message.receiver in self.subscribers:
                await self.subscribers[message.receiver](message)
        except Exception as e:
            self.stats['failed_deliveries'] += 1
            raise AIVillageException(f"Failed to send message: {str(e)}")

    async def receive_message(self, agent_id: str) -> Message:
        """
        Receive message with priority handling.

        :param agent_id: ID of the receiving agent.
        :return: Next message for the agent.
        """
        if agent_id not in self.message_queues:
            raise AIVillageException(f"No message queue for agent {agent_id}")

        message = self.message_queues[agent_id].get_next_message()
        if message is None:
            raise AIVillageException(f"No messages for agent {agent_id}")

        self.stats['messages_received'] += 1
        return message

    async def query(self, sender: str, receiver: str, content: Dict[str, Any]) -> Any:
        """
        Send a query and wait for response.

        :param sender: ID of the sending agent.
        :param receiver: ID of the receiving agent.
        :param content: Query content.
        :return: Response content.
        """
        query_message = Message(
            type=MessageType.QUERY,
            sender=sender,
            receiver=receiver,
            content=content,
            priority=Priority.HIGH  # Queries are high priority by default
        )
        await self.send_message(query_message)
        response = await self.receive_message(sender)
        return response.content

    def subscribe(self, agent_id: str, callback: Callable[[Message], Coroutine[Any, Any, None]]) -> None:
        """
        Subscribe to messages.

        :param agent_id: ID of the subscribing agent.
        :param callback: Callback function for handling messages.
        """
        self.subscribers[agent_id] = callback

    async def broadcast(self, message: Message, recipients: List[str]) -> None:
        """
        Broadcast message to multiple recipients.

        :param message: Message to broadcast.
        :param recipients: List of recipient agent IDs.
        """
        for recipient in recipients:
            message_copy = message.copy()
            message_copy.receiver = recipient
            await self.send_message(message_copy)

    def get_queue_status(self, agent_id: str) -> Dict[str, int]:
        """
        Get status of an agent's message queue.

        :param agent_id: ID of the agent.
        :return: Queue statistics.
        """
        if agent_id not in self.message_queues:
            return {'high_priority': 0, 'normal_priority': 0, 'low_priority': 0, 'total': 0}
        return self.message_queues[agent_id].get_queue_stats()

    def get_communication_stats(self) -> Dict[str, Any]:
        """
        Get communication statistics.

        :return: Dictionary of communication statistics.
        """
        return {
            **self.stats,
            'active_subscribers': len(self.subscribers),
            'active_queues': len(self.message_queues),
            'message_history_length': len(self.message_history)
        }

    async def create_group(self, group_id: str, members: List[str], config: Optional[Dict[str, Any]] = None) -> None:
        """
        Create a new communication group.

        :param group_id: Unique identifier for the group.
        :param members: List of agent IDs to include in the group.
        :param config: Optional group configuration.
        """
        await self.group_communication.create_group(group_id, members, config)

    async def broadcast_to_group(self, group_id: str, message: Message) -> None:
        """
        Broadcast message to all members of a group.

        :param group_id: ID of the target group.
        :param message: Message to broadcast.
        """
        await self.group_communication.broadcast_to_group(group_id, message)

    def get_agent_groups(self, agent_id: str) -> Set[str]:
        """
        Get all groups an agent belongs to.

        :param agent_id: ID of the agent.
        :return: Set of group IDs.
        """
        return self.group_communication.get_agent_groups(agent_id)

    def get_group_members(self, group_id: str) -> Set[str]:
        """
        Get all members of a group.

        :param group_id: ID of the group.
        :return: Set of agent IDs.
        """
        return self.group_communication.get_group_members(group_id)
