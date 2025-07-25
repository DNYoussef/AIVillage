import asyncio
from typing import Any

from agents.king.king_agent import KingAgent
from agents.magi.magi_agent import MagiAgent
from agents.sage.sage_agent import SageAgent
from agents.unified_base_agent import (
    SelfEvolvingSystem,
    UnifiedAgentConfig,
    UnifiedBaseAgent,
)
from agents.utils.task import Task as LangroidTask
from core.error_handling import StandardCommunicationProtocol
from rag_system.core.config import UnifiedConfig
from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.retrieval.vector_store import VectorStore


class TaskQueue:
    def __init__(self):
        self.tasks = asyncio.Queue()

    async def add_task(self, task: dict[str, Any]):
        await self.tasks.put(task)

    async def get_next_task(self) -> dict[str, Any]:
        return await self.tasks.get()

    def task_done(self):
        self.tasks.task_done()


async def get_user_input() -> dict[str, Any]:
    """Get task input from the user."""
    content = input("Enter task content: ")
    task_type = input("Enter task type: ")
    return {"content": content, "type": task_type}


async def process_result(result: dict[str, Any]):
    """Handle the result of a task execution."""
    print(f"Processing result: {result}")


def create_agents(
    config: UnifiedConfig,
    communication_protocol: StandardCommunicationProtocol,
    vector_store: VectorStore,
) -> list[UnifiedBaseAgent]:
    """Initialize and return a list of agents with their configurations."""
    agent_configs = [
        UnifiedAgentConfig(
            name="KingAgent",
            description="A decision-making and task delegation agent",
            capabilities=["decision_making", "task_delegation"],
            rag_config=config,
            vector_store=vector_store,
            model="gpt-4",
            instructions="You are a decision-making and task delegation agent.",
        ),
        UnifiedAgentConfig(
            name="SageAgent",
            description="A research and analysis agent",
            capabilities=["research", "analysis"],
            rag_config=config,
            vector_store=vector_store,
            model="gpt-4",
            instructions="You are a research and analysis agent.",
        ),
        UnifiedAgentConfig(
            name="MagiAgent",
            description="A specialized agent for complex problem-solving",
            capabilities=["problem_solving", "specialized_knowledge"],
            rag_config=config,
            vector_store=vector_store,
            model="gpt-4",
            instructions="You are a specialized agent for complex problem-solving.",
        ),
    ]

    return [
        KingAgent(agent_configs[0], communication_protocol, vector_store),
        SageAgent(agent_configs[1], communication_protocol, vector_store),
        MagiAgent(agent_configs[2], communication_protocol, config, vector_store),
    ]


async def run_task(
    self_evolving_system: SelfEvolvingSystem,
    rag_pipeline: EnhancedRAGPipeline,
    task_data: dict[str, Any],
):
    """Run a single task through the self-evolving system and RAG pipeline."""
    task_content = task_data["content"]
    task_type = task_data["type"]

    # Create a LangroidTask
    task = LangroidTask(self_evolving_system.agents[0], task_content)
    task.type = task_type

    # Process the task through the RAG pipeline
    rag_result = await rag_pipeline.process_query(task_content)

    # Use the RAG result to inform the self-evolving system's task processing
    task.content = f"{task_content}\nRAG Context: {rag_result}"
    ses_result = await self_evolving_system.process_task(task)

    combined_result = {"rag_result": rag_result, "ses_result": ses_result}

    await process_result(combined_result)

    if task_type == "evolve":
        await self_evolving_system.evolve()

    return combined_result


async def orchestrate_agents(
    agents: list[UnifiedBaseAgent], task: dict[str, Any]
) -> dict[str, Any]:
    king_agent = next(agent for agent in agents if isinstance(agent, KingAgent))
    langroid_task = LangroidTask(
        king_agent,
        task.get("content"),
        task.get("id", ""),
        task.get("priority", 1),
    )
    langroid_task.type = task.get("type", "general")
    result = await king_agent.execute_task(langroid_task)
    return result


async def main():
    """Main execution loop for the orchestration system."""
    config = UnifiedConfig()
    communication_protocol = StandardCommunicationProtocol()
    vector_store = VectorStore()
    agents = create_agents(config, communication_protocol, vector_store)
    self_evolving_system = SelfEvolvingSystem(agents)

    rag_pipeline = EnhancedRAGPipeline(config)

    task_queue = TaskQueue()

    # Add some initial tasks to the queue
    await task_queue.add_task({"content": "Analyze market trends", "type": "research"})
    await task_queue.add_task(
        {"content": "Debug login functionality", "type": "coding"}
    )
    await task_queue.add_task(
        {
            "type": "research",
            "content": "Analyze the impact of artificial intelligence on job markets in the next decade.",
        }
    )

    while True:
        # Get next task from queue or user input
        if task_queue.tasks.empty():
            task_data = await get_user_input()
        else:
            task_data = await task_queue.get_next_task()

        result = await run_task(self_evolving_system, rag_pipeline, task_data)
        orchestrated_result = await orchestrate_agents(agents, task_data)

        print(f"Task result: {result}")
        print(f"Orchestrated result: {orchestrated_result}")

        task_queue.task_done()

        # Ask if the user wants to add more tasks or exit
        user_choice = input("Enter 'a' to add more tasks, or 'q' to quit: ")
        if user_choice.lower() == "a":
            new_task = await get_user_input()
            await task_queue.add_task(new_task)
        elif user_choice.lower() == "q":
            break

    print("Orchestration complete.")


if __name__ == "__main__":
    asyncio.run(main())
