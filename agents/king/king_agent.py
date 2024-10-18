from typing import List, Dict, Any, Tuple
from agents.unified_base_agent import UnifiedBaseAgent, UnifiedAgentConfig, SelfEvolvingSystem
from agents.utils.task import Task as LangroidTask
from .coordinator import KingCoordinator
from .decision_maker import DecisionMaker
from .problem_analyzer import ProblemAnalyzer
from .unified_task_manager import UnifiedTaskManager
from agents.communication.protocol import StandardCommunicationProtocol, Message, MessageType
from rag_system.core.pipeline import EnhancedRAGPipeline as RAGSystem
from langroid.vector_store.base import VectorStore

class KingAgentConfig(UnifiedAgentConfig):
    coordinator_capabilities: List[str] = [
        "task_routing",
        "decision_making",
        "agent_management",
        "problem_analysis",
        "task_management"
    ]

class KingAgent(UnifiedBaseAgent):
    def __init__(self, config: KingAgentConfig, communication_protocol: StandardCommunicationProtocol, rag_system: RAGSystem, vector_store: VectorStore):
        super().__init__(config, communication_protocol)
        self.coordinator_capabilities = config.coordinator_capabilities
        self.rag_system = rag_system
        self.coordinator = KingCoordinator(communication_protocol, rag_system, self)
        self.decision_maker = DecisionMaker(communication_protocol, rag_system, self)
        self.problem_analyzer = ProblemAnalyzer(communication_protocol, self)
        self.task_manager = UnifiedTaskManager(communication_protocol, len(self.coordinator.agents), 10)
        self.self_evolving_system = SelfEvolvingSystem([self], vector_store)

    async def execute_task(self, task: LangroidTask) -> Dict[str, Any]:
        if task.type == "route_task":
            return await self.route_task(task)
        elif task.type == "make_decision":
            return await self.make_decision(task)
        elif task.type == "manage_agents":
            return await self.manage_agents(task)
        elif task.type == "analyze_problem":
            return await self.analyze_problem(task)
        elif task.type == "manage_task":
            return await self.manage_task(task)
        else:
            return await super().execute_task(task)

    async def route_task(self, task: LangroidTask) -> Dict[str, Any]:
        return await self.coordinator.handle_task_message(task)

    async def make_decision(self, task: LangroidTask) -> Dict[str, Any]:
        return await self.decision_maker.make_decision(task.content)

    async def manage_agents(self, task: LangroidTask) -> Dict[str, Any]:
        if task.content.get('action') == 'add':
            await self.coordinator.add_agent(task.content['agent_name'], task.content['agent_instance'])
        elif task.content.get('action') == 'remove':
            await self.coordinator.remove_agent(task.content['agent_name'])
        return {"status": "success", "message": f"Agent {task.content['action']}ed successfully"}

    async def analyze_problem(self, task: LangroidTask) -> Dict[str, Any]:
        rag_info = await self.rag_system.process_query(task.content)
        return await self.problem_analyzer.analyze(task.content, rag_info)

    async def manage_task(self, task: LangroidTask) -> Dict[str, Any]:
        if task.content.get('action') == 'create':
            new_task = await self.task_manager.create_task(task.content['description'], task.content['agent'])
            return {"task_id": new_task.id}
        elif task.content.get('action') == 'complete':
            await self.task_manager.complete_task(task.content['task_id'], task.content['result'])
            return {"status": "success"}
        else:
            return {"status": "error", "message": "Unknown task management action"}

    async def handle_message(self, message: Message):
        if message.type == MessageType.TASK:
            task = LangroidTask(self, message.content['content'])
            task.type = message.content.get('task_type', 'general')
            result = await self.self_evolving_system.process_task(task)
            response = Message(
                type=MessageType.RESPONSE,
                sender=self.name,
                receiver=message.sender,
                content=result,
                parent_id=message.id
            )
            await self.communication_protocol.send_message(response)
        else:
            await super().handle_message(message)

    async def update(self, task: Dict[str, Any], result: Any):
        await self.coordinator.process_task_completion(task, result)
        await self.decision_maker.update_model(task, result)
        await self.problem_analyzer.update_models(task, result)
        await self.task_manager.update_agent_performance(task['assigned_agent'], result)

    def save_models(self, path: str):
        self.coordinator.save_models(f"{path}/coordinator")
        self.decision_maker.save_models(f"{path}/decision_maker")
        self.problem_analyzer.save_models(f"{path}/problem_analyzer")
        self.task_manager.save_models(f"{path}/task_manager")

    def load_models(self, path: str):
        self.coordinator.load_models(f"{path}/coordinator")
        self.decision_maker.load_models(f"{path}/decision_maker")
        self.problem_analyzer.load_models(f"{path}/problem_analyzer")
        self.task_manager.load_models(f"{path}/task_manager")

    async def query_rag(self, query: str) -> Dict[str, Any]:
        return await self.rag_system.process_query(query)

    async def add_document(self, content: str, filename: str):
        await self.rag_system.add_document(content, filename)

    async def introspect(self) -> Dict[str, Any]:
        base_info = await super().introspect()
        return {
            **base_info,
            "coordinator_capabilities": self.coordinator_capabilities,
            "coordinator_info": await self.coordinator.introspect(),
            "decision_maker_info": await self.decision_maker.introspect(),
            "problem_analyzer_info": await self.problem_analyzer.introspect(),
            "task_manager_info": await self.task_manager.introspect()
        }

    async def evolve(self):
        await self.self_evolving_system.evolve()

# Example usage
if __name__ == "__main__":
    vector_store = VectorStore()  # Placeholder, implement actual VectorStore
    communication_protocol = StandardCommunicationProtocol()
    rag_system = RAGSystem()  # Placeholder, implement actual RAGSystem
    
    king_config = KingAgentConfig(
        name="KingAgent",
        description="A coordinating and decision-making agent",
        capabilities=["task_routing", "decision_making", "agent_management", "problem_analysis", "task_management"],
        vector_store=vector_store,
        model="gpt-4",
        instructions="You are a King agent capable of coordinating tasks, making decisions, and managing other agents."
    )
    
    king_agent = KingAgent(king_config, communication_protocol, rag_system, vector_store)
    
    # Use the king_agent to process tasks and evolve
