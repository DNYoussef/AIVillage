"""Enhanced communication protocol implementation."""

from typing import Dict, Any, Callable, Coroutine, List, Set, Optional
from abc import ABC, abstractmethod
from datetime import datetime
from .message import Message, MessageType, Priority
from .queue import MessageQueue
from agents.utils.exceptions import AIVillageException
import logging

logger = logging.getLogger(__name__)

class CommunicationProtocol(ABC):
    """Abstract base class defining the communication protocol interface."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the communication protocol."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the communication protocol."""
        pass
    
    @abstractmethod
    async def send_message(self, message: Message) -> None:
        """Send a message to a recipient."""
        pass

    @abstractmethod
    async def receive_message(self, agent_id: str) -> Message:
        """Receive a message for a specific agent."""
        pass

    @abstractmethod
    async def query(self, sender: str, receiver: str, content: Dict[str, Any]) -> Any:
        """Send a query and wait for response."""
        pass

    @abstractmethod
    def subscribe(self, agent_id: str, callback: Callable[[Message], Coroutine[Any, Any, None]]) -> None:
        """Subscribe to messages for a specific agent."""
        pass

class GroupCommunication:
    """Implements group communication functionality."""
    
    def __init__(self, communication_protocol: 'StandardCommunicationProtocol'):
        self.protocol = communication_protocol
        self.groups: Dict[str, Set[str]] = {}
        self.agent_groups: Dict[str, Set[str]] = {}
        self.group_configs: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self):
        """Initialize group communication."""
        logger.info("Initializing GroupCommunication")
        self.groups.clear()
        self.agent_groups.clear()
        self.group_configs.clear()
    
    async def shutdown(self):
        """Shutdown group communication."""
        logger.info("Shutting down GroupCommunication")
        # Notify all groups of shutdown
        for group_id, members in self.groups.items():
            try:
                message = Message(
                    type=MessageType.SYSTEM,
                    sender="group_communication",
                    receiver="group_members",
                    content={"action": "group_shutdown", "group_id": group_id},
                    priority=Priority.HIGH
                )
                await self.protocol.broadcast_to_group(group_id, message)
            except Exception as e:
                logger.warning(f"Error notifying group {group_id} of shutdown: {str(e)}")
        
        self.groups.clear()
        self.agent_groups.clear()
        self.group_configs.clear()

    async def create_group(self, group_id: str, members: List[str], 
                         config: Optional[Dict[str, Any]] = None) -> None:
        """Create a new communication group."""
        if group_id in self.groups:
            raise AIVillageException(f"Group {group_id} already exists")

        self.groups[group_id] = set(members)
        self.group_configs[group_id] = config or {}

        for member in members:
            if member not in self.agent_groups:
                self.agent_groups[member] = set()
            self.agent_groups[member].add(group_id)

    async def broadcast_to_group(self, group_id: str, message: Message) -> None:
        """Broadcast message to all group members."""
        if group_id not in self.groups:
            raise AIVillageException(f"Group {group_id} does not exist")

        for member in self.groups[group_id]:
            message_copy = Message(
                type=message.type,
                sender=message.sender,
                receiver=member,
                content=message.content,
                priority=message.priority
            )
            await self.protocol.send_message(message_copy)

    async def add_to_group(self, group_id: str, agent_id: str) -> None:
        """Add an agent to a group."""
        if group_id not in self.groups:
            raise AIVillageException(f"Group {group_id} does not exist")

        config = self.group_configs[group_id]
        if "max_members" in config and len(self.groups[group_id]) >= config["max_members"]:
            raise AIVillageException(f"Group {group_id} has reached maximum members limit")

        self.groups[group_id].add(agent_id)
        if agent_id not in self.agent_groups:
            self.agent_groups[agent_id] = set()
        self.agent_groups[agent_id].add(group_id)

    async def remove_from_group(self, group_id: str, agent_id: str) -> None:
        """Remove an agent from a group."""
        if group_id not in self.groups:
            raise AIVillageException(f"Group {group_id} does not exist")

        self.groups[group_id].discard(agent_id)
        if agent_id in self.agent_groups:
            self.agent_groups[agent_id].discard(group_id)

    async def get_agent_groups(self, agent_id: str) -> Set[str]:
        """Get all groups an agent belongs to."""
        return self.agent_groups.get(agent_id, set())

    async def get_group_members(self, group_id: str) -> Set[str]:
        """Get all members of a group."""
        if group_id not in self.groups:
            raise AIVillageException(f"Group {group_id} does not exist")
        return self.groups[group_id].copy()

class StandardCommunicationProtocol(CommunicationProtocol):
    """Enhanced communication protocol with priority handling and group communication."""
    
    def __init__(self):
        self.message_queues: Dict[str, MessageQueue] = {}
        self.subscribers: Dict[str, Callable[[Message], Coroutine[Any, Any, None]]] = {}
        self.group_communication = GroupCommunication(self)
        self.message_history: List[Message] = []
        self.stats: Dict[str, Any] = {
            'messages_sent': 0,
            'messages_received': 0,
            'failed_deliveries': 0
        }
        logger.info("Initialized StandardCommunicationProtocol")
    
    async def initialize(self) -> None:
        """Initialize the communication protocol."""
        try:
            logger.info("Initializing StandardCommunicationProtocol...")
            
            # Clear existing state
            self.message_queues.clear()
            self.subscribers.clear()
            self.message_history.clear()
            self.stats = {
                'messages_sent': 0,
                'messages_received': 0,
                'failed_deliveries': 0
            }
            
            # Initialize group communication
            await self.group_communication.initialize()
            
            logger.info("Successfully initialized StandardCommunicationProtocol")
            
        except Exception as e:
            logger.error(f"Error initializing StandardCommunicationProtocol: {str(e)}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the communication protocol."""
        try:
            logger.info("Shutting down StandardCommunicationProtocol...")
            
            # Notify all subscribers of shutdown
            shutdown_message = Message(
                type=MessageType.SYSTEM,
                sender="system",
                receiver="all",
                content={"action": "shutdown"},
                priority=Priority.HIGH
            )
            
            for subscriber_id in self.subscribers:
                try:
                    await self.send_message(Message(
                        type=shutdown_message.type,
                        sender=shutdown_message.sender,
                        receiver=subscriber_id,
                        content=shutdown_message.content,
                        priority=shutdown_message.priority
                    ))
                except Exception as e:
                    logger.warning(f"Error notifying subscriber {subscriber_id} of shutdown: {str(e)}")
            
            # Shutdown group communication
            await self.group_communication.shutdown()
            
            # Clear all state
            self.message_queues.clear()
            self.subscribers.clear()
            self.message_history.clear()
            
            logger.info("Successfully shut down StandardCommunicationProtocol")
            
        except Exception as e:
            logger.error(f"Error shutting down StandardCommunicationProtocol: {str(e)}")
            raise

    async def send_message(self, message: Message) -> None:
        """Send a message with priority handling."""
        try:
            if message.receiver not in self.message_queues:
                self.message_queues[message.receiver] = MessageQueue()

            await self.message_queues[message.receiver].enqueue(message)
            self.message_history.append(message)
            self.stats['messages_sent'] += 1

            if message.receiver in self.subscribers:
                try:
                    await self.subscribers[message.receiver](message)
                except Exception:
                    self.stats['failed_deliveries'] += 1

        except Exception as e:
            self.stats['failed_deliveries'] += 1
            raise AIVillageException(f"Failed to send message: {str(e)}")

    async def receive_message(self, agent_id: str) -> Message:
        """Receive message with priority handling."""
        if agent_id not in self.message_queues:
            self.stats['failed_deliveries'] += 1
            raise AIVillageException(f"No message queue for agent {agent_id}")

        message = await self.message_queues[agent_id].dequeue()
        if message is None:
            self.stats['failed_deliveries'] += 1
            raise AIVillageException(f"No messages for agent {agent_id}")

        self.stats['messages_received'] += 1
        return message

    async def query(self, sender: str, receiver: str, content: Dict[str, Any]) -> Any:
        """Send a query and wait for response."""
        query_message = Message(
            type=MessageType.QUERY,
            sender=sender,
            receiver=receiver,
            content=content,
            priority=Priority.HIGH
        )
        await self.send_message(query_message)
        response = await self.receive_message(sender)
        return response.content

    def subscribe(self, agent_id: str, 
                 callback: Callable[[Message], Coroutine[Any, Any, None]]) -> None:
        """Subscribe to messages."""
        self.subscribers[agent_id] = callback

    async def broadcast(self, message: Message, recipients: List[str]) -> None:
        """Broadcast message to multiple recipients."""
        for recipient in recipients:
            message_copy = Message(
                type=message.type,
                sender=message.sender,
                receiver=recipient,
                content=message.content,
                priority=message.priority
            )
            await self.send_message(message_copy)

    async def get_queue_status(self, agent_id: str) -> Dict[str, int]:
        """Get status of an agent's message queue."""
        if agent_id not in self.message_queues:
            return {'high_priority': 0, 'medium_priority': 0, 'low_priority': 0, 'total': 0}
        return await self.message_queues[agent_id].get_queue_stats()

    async def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return {
            **self.stats,
            'active_subscribers': len(self.subscribers),
            'active_queues': len(self.message_queues),
            'message_history_length': len(self.message_history)
        }

    # Group communication methods
    async def create_group(self, group_id: str, members: List[str], 
                         config: Optional[Dict[str, Any]] = None) -> None:
        """Create a new communication group."""
        await self.group_communication.create_group(group_id, members, config)

    async def broadcast_to_group(self, group_id: str, message: Message) -> None:
        """Broadcast message to all members of a group."""
        await self.group_communication.broadcast_to_group(group_id, message)

    async def get_agent_groups(self, agent_id: str) -> Set[str]:
        """Get all groups an agent belongs to."""
        return await self.group_communication.get_agent_groups(agent_id)

    async def get_group_members(self, group_id: str) -> Set[str]:
        """Get all members of a group."""
        return await self.group_communication.get_group_members(group_id)
