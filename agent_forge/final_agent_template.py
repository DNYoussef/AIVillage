from typing import Dict, Any, Callable
from langroid import ChatAgent, ChatAgentConfig
from communications import MultiAgentCommunicationProtocol, Message, MessageType

class FinalAgent:
    """Template agent using Langroid and MACP for communication and tool use."""

    def __init__(self, name: str, model: str, protocol: MultiAgentCommunicationProtocol):
        config = ChatAgentConfig(name=name, llm=dict(chat_model=model))
        self.agent = ChatAgent(config)
        self.name = name
        self.protocol = protocol
        self.protocol.subscribe(self.name, self.handle_message)
        self.tools: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

    def add_tool(self, name: str, handler: Callable[[Dict[str, Any]], Any]):
        self.tools[name] = handler
        self.protocol.register_tool(name, handler)

    async def handle_message(self, message: Message):
        if message.type == MessageType.QUERY:
            tool_name = message.content.get("tool")
            params = message.content.get("params", {})
            if tool_name and tool_name in self.tools:
                result = await self.tools[tool_name](params)
                response = Message(type=MessageType.RESPONSE, sender=self.name, receiver=message.sender, content=result, parent_id=message.id)
                await self.protocol.send_message(response)

    async def query_agent(self, agent_id: str, tool: str, params: Dict[str, Any]) -> Any:
        return await self.protocol.request_tool(self.name, agent_id, tool, params)
