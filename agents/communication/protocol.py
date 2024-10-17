from typing import Callable, Dict, Any
from langroid.agent.task import Task

class StandardCommunicationProtocol:
    """Standard communication protocol for AI Village agents."""
    
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, agent_id: str, callback: Callable):
        """Subscribe an agent to receive messages."""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)

    def unsubscribe(self, agent_id: str, callback: Callable):
        """Unsubscribe an agent from receiving messages."""
        if agent_id in self.subscribers and callback in self.subscribers[agent_id]:
            self.subscribers[agent_id].remove(callback)

    async def send_message(self, sender: str, receiver: str, content: Dict[str, Any]):
        """Send a message from one agent to another."""
        if receiver in self.subscribers:
            task = Task(content=content)
            for callback in self.subscribers[receiver]:
                await callback(task)

    async def broadcast(self, sender: str, content: Dict[str, Any]):
        """Broadcast a message to all subscribed agents."""
        for receiver in self.subscribers:
            await self.send_message(sender, receiver, content)
