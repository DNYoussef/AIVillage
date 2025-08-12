from typing import Any

from agents.unified_base_agent import (
    SelfEvolvingSystem,
    UnifiedAgentConfig,
    UnifiedBaseAgent,
)
from agents.utils.task import Task as LangroidTask
from rag_system.core.config import RAGConfig
from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.retrieval.vector_store import VectorStore
from rag_system.tracking.unified_knowledge_tracker import UnifiedKnowledgeTracker

from core.error_handling import Message, MessageType, StandardCommunicationProtocol


class MagiAgentConfig(UnifiedAgentConfig):
    development_capabilities: list[str] = ["coding", "debugging", "code_review"]


class MagiAgent(UnifiedBaseAgent):
    def __init__(
        self,
        config: MagiAgentConfig,
        communication_protocol: StandardCommunicationProtocol,
        rag_config: RAGConfig,
        vector_store: VectorStore,
        knowledge_tracker: UnifiedKnowledgeTracker | None = None,
    ) -> None:
        super().__init__(config, communication_protocol, knowledge_tracker)
        self.specialized_knowledge = {}  # Initialize specialized knowledge base
        self.rag_system = EnhancedRAGPipeline(rag_config, knowledge_tracker)
        self.vector_store = vector_store
        self.self_evolving_system = SelfEvolvingSystem([self])
        self.development_capabilities = config.development_capabilities

    async def execute_task(self, task: LangroidTask) -> dict[str, Any]:
        if task.type in self.development_capabilities:
            return await getattr(self, f"handle_{task.type}")(task)
        return await super().execute_task(task)

    async def handle_coding(self, task: LangroidTask) -> dict[str, Any]:
        code_result = await self.generate(f"Write code for: {task.content}")
        return {"code_result": code_result}

    async def handle_debugging(self, task: LangroidTask) -> dict[str, Any]:
        debug_result = await self.generate(f"Debug the following code: {task.content}")
        return {"debug_result": debug_result}

    async def handle_code_review(self, task: LangroidTask) -> dict[str, Any]:
        review_result = await self.generate(
            f"Review the following code: {task.content}"
        )
        return {"review_result": review_result}

    async def handle_message(self, message: Message) -> None:
        if message.type == MessageType.TASK:
            task = LangroidTask(self, message.content["content"])
            task.type = message.content.get("task_type", "general")
            result = await self.self_evolving_system.process_task(task)
            response = Message(
                type=MessageType.RESPONSE,
                sender=self.name,
                receiver=message.sender,
                content=result,
                parent_id=message.id,
            )
            await self.communication_protocol.send_message(response)
        else:
            await super().handle_message(message)

    async def introspect(self) -> dict[str, Any]:
        base_info = await super().introspect()
        return {**base_info, "development_capabilities": self.development_capabilities}

    async def evolve(self) -> None:
        await self.self_evolving_system.evolve()

    async def query_rag(self, query: str) -> dict[str, Any]:
        return await self.rag_system.process_query(query)

    async def add_document(self, content: str, filename: str) -> None:
        await self.rag_system.add_document(content, filename)


# Example usage
if __name__ == "__main__":
    vector_store = VectorStore()  # Placeholder, implement actual VectorStore
    communication_protocol = StandardCommunicationProtocol()
    rag_config = RAGConfig()

    magi_config = MagiAgentConfig(
        name="MagiAgent",
        description="A development and coding agent",
        capabilities=["coding", "debugging", "code_review"],
        vector_store=vector_store,
        model="gpt-4",
        instructions="You are a Magi agent capable of writing, debugging, and reviewing code.",
    )

    magi_agent = MagiAgent(
        magi_config, communication_protocol, rag_config, vector_store
    )

    # Use the magi_agent to process tasks and evolve
