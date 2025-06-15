from typing import Dict, Any, Callable, Coroutine, List, Set
from .protocol import StandardCommunicationProtocol, Message, MessageType, Priority

class MultiAgentCommunicationProtocol(StandardCommunicationProtocol):
    """Extended protocol supporting group messaging and tool requests."""

    def __init__(self):
        super().__init__()
        self.groups: Dict[str, Set[str]] = {}
        self.tool_registry: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

    def register_tool(self, name: str, handler: Callable[[Dict[str, Any]], Any]):
        self.tool_registry[name] = handler

    async def request_tool(self, sender: str, receiver: str, tool_name: str, params: Dict[str, Any]) -> Any:
        content = {"tool": tool_name, "params": params}
        msg = Message(type=MessageType.QUERY, sender=sender, receiver=receiver, content=content)
        await self.send_message(msg)
        response = await self.receive_message(sender)
        return response.content

    def create_group(self, group_id: str, members: List[str]):
        self.groups[group_id] = set(members)
        for member in members:
            self.subscribe(group_id, self.subscribers.get(member, lambda m: None))

    async def broadcast(self, sender: str, message_type: MessageType, content: Dict[str, Any], priority: Priority = Priority.MEDIUM):
        for agent_id in self.subscribers.keys():
            msg = Message(type=message_type, sender=sender, receiver=agent_id, content=content, priority=priority)
            await self.send_message(msg)
