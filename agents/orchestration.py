import asyncio
from typing import List, Dict, Any
from datetime import datetime
from agents.unified_base_agent import UnifiedBaseAgent, UnifiedAgentConfig, SelfEvolvingSystem
from agents.sage.sage_agent import SageAgent
from agents.king.king_agent import KingAgent
from agents.magi.magi_agent import MagiAgent
from rag_system.core.config import UnifiedConfig
from communications.protocol import StandardCommunicationProtocol, Message, MessageType, Priority
from langroid.vector_store.base import VectorStore
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.agent.task import Task as LangroidTask
from rag_system.core.pipeline import EnhancedRAGPipeline

class TaskQueue:
    """Enhanced task queue with priority and scheduling."""
    
    def __init__(self):
        self.tasks = asyncio.PriorityQueue()
        self.completed_tasks = []
        self.failed_tasks = []

    async def add_task(self, task: Dict[str, Any], priority: int = Priority.MEDIUM):
        """Add a task with priority."""
        await self.tasks.put((priority, {
            "content": task,
            "added": datetime.now().isoformat(),
            "priority": priority
        }))

    async def get_next_task(self) -> Dict[str, Any]:
        """Get next task based on priority."""
        _, task = await self.tasks.get()
        return task

    def task_done(self, task: Dict[str, Any], success: bool = True):
        """Mark task as done and store result."""
        self.tasks.task_done()
        task["completed"] = datetime.now().isoformat()
        if success:
            self.completed_tasks.append(task)
        else:
            self.failed_tasks.append(task)

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "pending": self.tasks.qsize(),
            "completed": len(self.completed_tasks),
            "failed": len(self.failed_tasks),
            "success_rate": len(self.completed_tasks) / (len(self.completed_tasks) + len(self.failed_tasks)) if self.completed_tasks or self.failed_tasks else 0
        }

async def get_user_input() -> Dict[str, Any]:
    """Enhanced user input handling."""
    content = input("Enter task content: ")
    task_type = input("Enter task type: ")
    priority = input("Enter priority (1-5, default 3): ")
    
    try:
        priority = int(priority)
        if not 1 <= priority <= 5:
            priority = Priority.MEDIUM
    except ValueError:
        priority = Priority.MEDIUM

    return {
        "content": content,
        "type": task_type,
        "priority": priority,
        "timestamp": datetime.now().isoformat()
    }

async def process_result(result: Dict[str, Any]):
    """Enhanced result processing."""
    print(f"\nTask Result:")
    print(f"Status: {result.get('status', 'unknown')}")
    print(f"Result: {result.get('result', 'No result')}")
    if 'error' in result:
        print(f"Error: {result['error']}")
    if 'metrics' in result:
        print("\nMetrics:")
        for key, value in result['metrics'].items():
            print(f"{key}: {value}")

def create_agents(config: UnifiedConfig, communication_protocol: StandardCommunicationProtocol, vector_store: VectorStore) -> List[UnifiedBaseAgent]:
    """Create standardized agents with enhanced configuration."""
    
    base_config = {
        "max_retries": 3,
        "timeout": 30.0,
        "batch_size": 32,
        "memory_size": 1000,
        "learning_rate": 0.01,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    agent_configs = [
        UnifiedAgentConfig(
            name="KingAgent",
            description="A decision-making and task delegation agent",
            capabilities=["decision_making", "task_delegation", "resource_management"],
            rag_config=config,
            vector_store=vector_store,
            model="gpt-4",
            instructions="You are a decision-making and task delegation agent.",
            **base_config
        ),
        UnifiedAgentConfig(
            name="SageAgent",
            description="A research and analysis agent",
            capabilities=["research", "analysis", "knowledge_synthesis"],
            rag_config=config,
            vector_store=vector_store,
            model="gpt-4",
            instructions="You are a research and analysis agent.",
            **base_config
        ),
        UnifiedAgentConfig(
            name="MagiAgent",
            description="A specialized agent for complex problem-solving",
            capabilities=["problem_solving", "specialized_knowledge", "optimization"],
            rag_config=config,
            vector_store=vector_store,
            model="gpt-4",
            instructions="You are a specialized agent for complex problem-solving.",
            **base_config
        )
    ]
    
    return [
        KingAgent(agent_configs[0], communication_protocol),
        SageAgent(agent_configs[1], communication_protocol),
        MagiAgent(agent_configs[2], communication_protocol)
    ]

async def run_task(self_evolving_system: SelfEvolvingSystem, rag_pipeline: EnhancedRAGPipeline, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced task execution with monitoring."""
    try:
        task_content = task_data['content']
        task_type = task_data['type']
        
        # Create a LangroidTask with metadata
        task = LangroidTask(self_evolving_system.agents[0], task_content)
        task.type = task_type
        task.metadata = {
            "priority": task_data.get("priority", Priority.MEDIUM),
            "timestamp": task_data.get("timestamp", datetime.now().isoformat())
        }
        
        # Process through RAG pipeline
        rag_start = datetime.now()
        rag_result = await rag_pipeline.process_query(task_content)
        rag_duration = (datetime.now() - rag_start).total_seconds()
        
        # Enhance task with RAG context
        task.content = f"{task_content}\nRAG Context: {rag_result}"
        
        # Process through self-evolving system
        ses_start = datetime.now()
        ses_result = await self_evolving_system.process_task(task)
        ses_duration = (datetime.now() - ses_start).total_seconds()
        
        # Combine results with metrics
        result = {
            "rag_result": rag_result,
            "ses_result": ses_result,
            "metrics": {
                "rag_duration": rag_duration,
                "ses_duration": ses_duration,
                "total_duration": rag_duration + ses_duration
            },
            "status": "success"
        }
        
        # Trigger evolution if needed
        if task_type == "evolve" or should_evolve(result):
            await self_evolving_system.evolve()
            result["evolved"] = True

        return result

    except Exception as e:
        return {
            "error": str(e),
            "status": "failed",
            "timestamp": datetime.now().isoformat()
        }

def should_evolve(result: Dict[str, Any]) -> bool:
    """Determine if system should evolve based on results."""
    metrics = result.get("metrics", {})
    
    # Evolution triggers
    triggers = [
        metrics.get("total_duration", 0) > 60,  # Long execution time
        metrics.get("error_rate", 0) > 0.2,     # High error rate
        metrics.get("confidence", 1.0) < 0.7    # Low confidence
    ]
    
    return any(triggers)

async def orchestrate_agents(agents: List[UnifiedBaseAgent], task: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced agent orchestration with load balancing."""
    try:
        # Find the most suitable agent
        king_agent = next(agent for agent in agents if isinstance(agent, KingAgent))
        
        # Get agent loads
        agent_loads = {
            agent.name: len(agent.memory) / agent.config.memory_size
            for agent in agents
        }
        
        # Execute task with load balancing
        if min(agent_loads.values()) > 0.9:  # High load on all agents
            print("Warning: High load on all agents")
        
        result = await king_agent.execute_task(task)
        
        # Record orchestration metrics
        result["orchestration_metrics"] = {
            "agent_loads": agent_loads,
            "selected_agent": king_agent.name
        }
        
        return result

    except Exception as e:
        return {
            "error": f"Orchestration error: {str(e)}",
            "status": "failed"
        }

async def main():
    """Enhanced main execution loop."""
    config = UnifiedConfig()
    communication_protocol = StandardCommunicationProtocol()
    vector_store = VectorStore()
    
    # Initialize system components
    agents = create_agents(config, communication_protocol, vector_store)
    self_evolving_system = SelfEvolvingSystem(agents)
    rag_pipeline = EnhancedRAGPipeline(config)
    task_queue = TaskQueue()

    # Add initial tasks
    initial_tasks = [
        {"content": "Analyze market trends", "type": "research", "priority": Priority.HIGH},
        {"content": "Debug login functionality", "type": "coding", "priority": Priority.MEDIUM},
        {"content": "Analyze AI impact on job markets", "type": "research", "priority": Priority.LOW}
    ]

    for task in initial_tasks:
        await task_queue.add_task(task, task["priority"])

    print("\nAgent Village Orchestration System")
    print("=================================")

    while True:
        try:
            # Process tasks
            if not task_queue.tasks.empty():
                task_data = await task_queue.get_next_task()
                print(f"\nProcessing task: {task_data['content']}")
                
                result = await run_task(self_evolving_system, rag_pipeline, task_data)
                orchestrated_result = await orchestrate_agents(agents, task_data)
                
                # Process and display results
                await process_result(result)
                await process_result(orchestrated_result)
                
                # Update task status
                task_queue.task_done(task_data, result.get("status") == "success")
                
                # Display queue stats
                print("\nQueue Statistics:")
                stats = task_queue.get_stats()
                for key, value in stats.items():
                    print(f"{key}: {value}")

            # Get user input for next action
            print("\nOptions:")
            print("1. Add new task")
            print("2. View agent status")
            print("3. Exit")
            
            choice = input("\nEnter choice (1-3): ")
            
            if choice == "1":
                new_task = await get_user_input()
                await task_queue.add_task(new_task, new_task.get("priority", Priority.MEDIUM))
            elif choice == "2":
                print("\nAgent Status:")
                for agent in agents:
                    print(f"\n{agent.name}:")
                    info = agent.info
                    for key, value in info.items():
                        print(f"{key}: {value}")
            elif choice == "3":
                break
            else:
                print("Invalid choice. Please try again.")

        except Exception as e:
            print(f"\nError in main loop: {str(e)}")
            continue

    print("\nOrchestration complete.")

if __name__ == "__main__":
    asyncio.run(main())
