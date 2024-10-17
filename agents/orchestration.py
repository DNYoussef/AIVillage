from typing import List, Dict, Any
import asyncio
from agents.unified_base_agent import UnifiedBaseAgent, UnifiedAgentConfig, create_agent, SelfEvolvingSystem
from langroid.vector_store.base import VectorStore
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.agent.task import Task as LangroidTask
from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.core.config import RAGConfig

class TaskQueue:
    def __init__(self):
        self.tasks = asyncio.Queue()

    async def add_task(self, task: Dict[str, Any]):
        await self.tasks.put(task)

    async def get_next_task(self) -> Dict[str, Any]:
        return await self.tasks.get()

    def task_done(self):
        self.tasks.task_done()

async def get_user_input() -> Dict[str, Any]:
    """
    Get task input from the user.
    """
    content = input("Enter task content: ")
    task_type = input("Enter task type: ")
    return {"content": content, "type": task_type}

async def process_result(result: Dict[str, Any]):
    """
    Handle the result of a task execution.
    """
    print(f"Processing result: {result}")

def initialize_agents(vector_store: VectorStore) -> List[UnifiedBaseAgent]:
    """
    Initialize and return a list of agents with their configurations.
    """
    agent_configs = [
        UnifiedAgentConfig(
            name="King",
            description="Coordinator agent",
            capabilities=["coordination", "decision_making"],
            vector_store=vector_store,
            model="gpt-4",
            instructions="You are the King agent, responsible for coordination and decision making."
        ),
        UnifiedAgentConfig(
            name="Sage",
            description="Research agent",
            capabilities=["research", "analysis"],
            vector_store=vector_store,
            model="gpt-4",
            instructions="You are the Sage agent, responsible for research and analysis."
        ),
        UnifiedAgentConfig(
            name="Magi",
            description="Development agent",
            capabilities=["coding", "debugging"],
            vector_store=vector_store,
            model="gpt-4",
            instructions="You are the Magi agent, responsible for coding and debugging."
        )
    ]
    
    return [create_agent(config.name, config) for config in agent_configs]

async def run_task(self_evolving_system: SelfEvolvingSystem, rag_pipeline: EnhancedRAGPipeline, task_data: Dict[str, Any]):
    """
    Run a single task through the self-evolving system and RAG pipeline.
    """
    task_content = task_data['content']
    task_type = task_data['type']
    
    # Create a LangroidTask
    task = LangroidTask(self_evolving_system.agents[0], task_content)
    task.type = task_type
    
    # Process the task through the RAG pipeline
    rag_result = await rag_pipeline.process_query(task_content, self_evolving_system.agents[0])
    
    # Use the RAG result to inform the self-evolving system's task processing
    task.content = f"{task_content}\nRAG Context: {rag_result}"
    ses_result = await self_evolving_system.process_task(task)
    
    combined_result = {
        "rag_result": rag_result,
        "ses_result": ses_result
    }
    
    await process_result(combined_result)
    
    if task_type == "evolve":
        await self_evolving_system.evolve()

async def main():
    """
    Main execution loop for the orchestration system.
    """
    vector_store = VectorStore()  # This is a placeholder. Implement or use a concrete VectorStore.
    agents = initialize_agents(vector_store)
    self_evolving_system = SelfEvolvingSystem(agents, vector_store)
    
    rag_config = RAGConfig()
    rag_pipeline = EnhancedRAGPipeline(rag_config)
    
    task_queue = TaskQueue()

    # Add some initial tasks to the queue
    await task_queue.add_task({"content": "Analyze market trends", "type": "research"})
    await task_queue.add_task({"content": "Debug login functionality", "type": "coding"})

    while True:
        # Get next task from queue or user input
        if task_queue.tasks.empty():
            task_data = await get_user_input()
        else:
            task_data = await task_queue.get_next_task()

        await run_task(self_evolving_system, rag_pipeline, task_data)
        task_queue.task_done()

        # Ask if the user wants to add more tasks or exit
        user_choice = input("Enter 'a' to add more tasks, or 'q' to quit: ")
        if user_choice.lower() == 'a':
            new_task = await get_user_input()
            await task_queue.add_task(new_task)
        elif user_choice.lower() == 'q':
            break

if __name__ == "__main__":
    asyncio.run(main())
