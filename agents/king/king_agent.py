from typing import List, Dict, Any
from agents.unified_base_agent import UnifiedBaseAgent, UnifiedAgentConfig
from langroid.agent.task import Task
from .coordinator import KingCoordinator
from .decision_maker import DecisionMaker
from .problem_analyzer import ProblemAnalyzer
from .unified_task_manager import UnifiedTaskManager
from ..communication.protocol import StandardCommunicationProtocol
from rag_system.core.pipeline import EnhancedRAGPipeline as RAGSystem

class KingAgentConfig(UnifiedAgentConfig):
    coordinator_capabilities: List[str] = [
        "task_routing",
        "decision_making",
        "agent_management",
        "problem_analysis",
        "task_management"
    ]

class KingAgent(UnifiedBaseAgent):
    def __init__(self, config: KingAgentConfig, communication_protocol: StandardCommunicationProtocol, rag_system: RAGSystem):
        super().__init__(config)
        self.coordinator_capabilities = config.coordinator_capabilities
        self.communication_protocol = communication_protocol
        self.rag_system = rag_system
        self.coordinator = KingCoordinator(communication_protocol, rag_system, self)
        self.decision_maker = DecisionMaker(communication_protocol, rag_system, self)
        self.problem_analyzer = ProblemAnalyzer(communication_protocol, self)
        self.task_manager = UnifiedTaskManager(communication_protocol, len(self.coordinator.agents), 10)

    async def execute_task(self, task: Task) -> Dict[str, Any]:
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

    async def route_task(self, task: Task) -> Dict[str, Any]:
        return await self.coordinator.handle_task_message(task)

    async def make_decision(self, task: Task) -> Dict[str, Any]:
        return await self.decision_maker.make_decision(task.content)

    async def manage_agents(self, task: Task) -> Dict[str, Any]:
        if task.content.get('action') == 'add':
            await self.coordinator.add_agent(task.content['agent_name'], task.content['agent_instance'])
        elif task.content.get('action') == 'remove':
            await self.coordinator.remove_agent(task.content['agent_name'])
        return {"status": "success", "message": f"Agent {task.content['action']}ed successfully"}

    async def analyze_problem(self, task: Task) -> Dict[str, Any]:
        rag_info = await self.rag_system.process_query(task.content)
        return await self.problem_analyzer.analyze(task.content, rag_info)

    async def manage_task(self, task: Task) -> Dict[str, Any]:
        if task.content.get('action') == 'create':
            new_task = await self.task_manager.create_task(task.content['description'], task.content['agent'])
            return {"task_id": new_task.id}
        elif task.content.get('action') == 'complete':
            await self.task_manager.complete_task(task.content['task_id'], task.content['result'])
            return {"status": "success"}
        else:
            return {"status": "error", "message": "Unknown task management action"}

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

    async def introspect(self) -> Dict[str, Any]:
        return {
            **await super().introspect(),
            "coordinator_capabilities": self.coordinator_capabilities,
            "coordinator_info": await self.coordinator.introspect(),
            "decision_maker_info": await self.decision_maker.introspect(),
            "problem_analyzer_info": await self.problem_analyzer.introspect(),
            "task_manager_info": await self.task_manager.introspect()
        }
