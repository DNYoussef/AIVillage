import asyncio
from typing import List, Dict, Any
from agents.unified_base_agent import UnifiedBaseAgent, UnifiedAgentConfig, SelfEvolvingSystem
from agents.sage.sage_agent import SageAgent
from agents.king.king_agent import KingAgent
from agents.magi.core.magi_agent import MagiAgent
from rag_system.core.config import UnifiedConfig
from communications.protocol import StandardCommunicationProtocol
from langroid.vector_store.base import VectorStore
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.agent.task import Task as LangroidTask
from rag_system.core.pipeline import EnhancedRAGPipeline
import logging
from queue import PriorityQueue

logger = logging.getLogger(__name__)

class TaskQueue:
    def __init__(self):
        self.tasks = PriorityQueue()

    async def add_task(self, task: Dict[str, Any], priority: int):
        await self.tasks.put((priority, task))

    async def get_next_task(self) -> Dict[str, Any]:
        _, task = await self.tasks.get()
        return task

    def task_done(self):
        self.tasks.task_done()

async def get_user_input() -> Dict[str, Any]:
    """
    Get task input from the user.
    """
    content = input("Enter task content: ")
    task_type = input("Enter task type (coding/debugging/code_review): ")
    priority = int(input("Enter task priority (1-5, 1 being the highest): "))
    return {"content": content, "type": task_type, "priority": priority}

async def process_result(result: Dict[str, Any]):
    """
    Handle the result of a task execution.
    """
    print(f"Processing result: {result}")

def create_agents(config: UnifiedConfig, communication_protocol: StandardCommunicationProtocol, vector_store: VectorStore) -> List[UnifiedBaseAgent]:
    """
    Initialize and return a list of agents with their configurations.
    """
    agent_configs = [
        UnifiedAgentConfig(
            name="KingAgent",
            description="A decision-making and task delegation agent",
            capabilities=["decision_making", "task_delegation"],
            rag_config=config,
            vector_store=vector_store,
            model="gpt-4",
            instructions="You are a decision-making and task delegation agent."
        ),
        UnifiedAgentConfig(
            name="SageAgent",
            description="A research and analysis agent",
            capabilities=["research", "analysis"],
            rag_config=config,
            vector_store=vector_store,
            model="gpt-4",
            instructions="You are a research and analysis agent."
        ),
        UnifiedAgentConfig(
            name="MagiAgent",
            description="A specialized agent for complex problem-solving",
            capabilities=["problem_solving", "specialized_knowledge", "coding", "debugging", "code_review"],
            rag_config=config,
            vector_store=vector_store,
            model="gpt-4",
            instructions="You are a Magi agent capable of writing, debugging, and reviewing code."
        )
    ]
    
    return [
        KingAgent(agent_configs[0], communication_protocol),
        SageAgent(agent_configs[1], communication_protocol),
        MagiAgent(agent_configs[2], communication_protocol)
    ]

async def analyze_task(magi_agent: MagiAgent, task_data: Dict[str, Any]) -> List[str]:
    """
    Analyze the task and determine the required tools.
    """
    task_content = task_data['content']
    task_type = task_data['type']
    
    analysis_prompt = f"""
    Analyze the following task and determine the tools required to complete it:
    
    Task Type: {task_type}
    Task Content: {task_content}
    
    Provide a comma-separated list of the required tools.
    """
    
    analysis_result = await magi_agent.generate(analysis_prompt)
    required_tools = [tool.strip() for tool in analysis_result.split(',')]
    
    return required_tools

async def ensure_tools_exist(magi_agent: MagiAgent, required_tools: List[str]):
    """
    Ensure that the required tools exist, creating them if necessary.
    """
    for tool_name in required_tools:
        if tool_name not in magi_agent.tools:
            # Tool doesn't exist, create it dynamically
            tool_code = await magi_agent.generate(f"Write the code for the '{tool_name}' tool:")
            tool_description = await magi_agent.generate(f"Provide a description for the '{tool_name}' tool:")
            tool_parameters = await magi_agent.generate(f"Specify the parameters for the '{tool_name}' tool in JSON format:")
            
            create_result = await magi_agent.create_dynamic_tool(tool_name, tool_code, tool_description, tool_parameters)
            logger.info(f"Dynamic tool creation result for '{tool_name}': {create_result}")

async def run_task(magi_agent: MagiAgent, task_data: Dict[str, Any]):
    """
    Run a single task through the MagiAgent using an iterative approach.
    """
    task_content = task_data['content']
    task_type = task_data['type']
    
    # Create a LangroidTask
    task = LangroidTask(magi_agent, task_content)
    task.type = task_type
    
    # Analyze the task and determine required tools
    required_tools = await analyze_task(magi_agent, task_data)
    logger.info(f"Required tools for the task: {required_tools}")
    
    # Ensure the required tools exist
    await ensure_tools_exist(magi_agent, required_tools)
    
    # Iterative task execution
    max_iterations = 5
    for i in range(max_iterations):
        logger.info(f"Iteration {i + 1}/{max_iterations}")
        result = await magi_agent.execute_task(task)
        
        if result.get("status") == "complete":
            break
        
        # Update task content based on the result
        task.content = result.get("next_task_content", task_content)
    
    await process_result(result)
    return result

async def orchestrate_agents(agents: List[UnifiedBaseAgent], task: Dict[str, Any]) -> Dict[str, Any]:
    king_agent = next(agent for agent in agents if isinstance(agent, KingAgent))
    result = await king_agent.execute_task(task)
    return result

async def main():
    """
    Main execution loop for the orchestration system.
    """
    config = UnifiedConfig()
    communication_protocol = StandardCommunicationProtocol()
    vector_store = VectorStore()  # Implement or use a concrete VectorStore.
    agents = create_agents(config, communication_protocol, vector_store)
    magi_agent = next(agent for agent in agents if isinstance(agent, MagiAgent))
    
    task_queue = TaskQueue()

    # Add some initial tasks to the queue
    await task_queue.add_task({"content": "Analyze market trends", "type": "research"}, priority=3)
    await task_queue.add_task({"content": "Debug login functionality", "type": "debugging"}, priority=1)
    await task_queue.add_task({
        "type": "research",
        "content": "Analyze the impact of artificial intelligence on job markets in the next decade."
    }, priority=4)

    while True:
        # Get next task from queue or user input
        if task_queue.tasks.empty():
            task_data = await get_user_input()
            priority = task_data.pop("priority")
            await task_queue.add_task(task_data, priority)
        
        task_data = await task_queue.get_next_task()
        result = await run_task(magi_agent, task_data)
        orchestrated_result = await orchestrate_agents(agents, task_data)
        
        print(f"Task result: {result}")
        print(f"Orchestrated result: {orchestrated_result}")
        
        task_queue.task_done()

        # Ask if the user wants to add more tasks or exit
        user_choice = input("Enter 'a' to add more tasks, or 'q' to quit: ")
        if user_choice.lower() == 'a':
            new_task = await get_user_input()
            priority = new_task.pop("priority")
            await task_queue.add_task(new_task, priority)
        elif user_choice.lower() == 'q':
            break

    print("Orchestration complete.")

if __name__ == "__main__":
    asyncio.run(main())
