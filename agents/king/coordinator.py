from asyncio.log import logger
from typing import List, Dict, Any
from agents.unified_base_agent import UnifiedBaseAgent
from communications.protocol import StandardCommunicationProtocol, Message, MessageType
from core.config import UnifiedConfig
from ..magi.magi_agent import MagiAgent
from ..sage.sage_agent import SageAgent
from rag_system.error_handling.error_handler import error_handler, safe_execute, AIVillageException
from .analytics.unified_analytics import UnifiedAnalytics

class KingCoordinator:
    def __init__(self, config: UnifiedConfig, communication_protocol: StandardCommunicationProtocol):
        self.config = config
        self.communication_protocol = communication_protocol
        self.agents: Dict[str, UnifiedBaseAgent] = {}
        self.task_manager = None  # Initialize this in the setup method
        self.router = None  # Initialize this in the setup method
        self.decision_maker = None  # Initialize this in the setup method
        self.problem_analyzer = None  # Initialize this in the setup method
        self.king_agent = None  # Initialize this in the setup method
        self.unified_analytics = UnifiedAnalytics()

    def add_agent(self, agent_name: str, agent: UnifiedBaseAgent):
        self.agents[agent_name] = agent

    @error_handler.handle_error
    async def coordinate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        start_time = self.unified_analytics.get_current_time()
        result = await self._delegate_task(task)
        end_time = self.unified_analytics.get_current_time()
        execution_time = end_time - start_time
        
        self.unified_analytics.record_task_completion(task['id'], execution_time, result.get('success', False))
        self.unified_analytics.record_metric(f"task_type_{task['type']}_execution_time", execution_time)
        
        return result

    @error_handler.handle_error
    async def _delegate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        if task['type'] == 'research':
            sage_agent = next((agent for agent in self.agents.values() if isinstance(agent, SageAgent)), None)
            if sage_agent:
                return await sage_agent.execute_task(task)
        elif task['type'] in ['coding', 'debugging', 'code_review']:
            magi_agent = next((agent for agent in self.agents.values() if isinstance(agent, MagiAgent)), None)
            if magi_agent:
                return await magi_agent.execute_task(task)
        
        # If no specific agent is found, delegate to the first available agent
        if self.agents:
            return await next(iter(self.agents.values())).execute_task(task)
        
        raise ValueError("No suitable agent found for the task")

    async def handle_message(self, message: Message):
        if message.type == MessageType.TASK:
            result = await self.coordinate_task(message.content)
            response = Message(
                type=MessageType.RESPONSE,
                sender="KingCoordinator",
                receiver=message.sender,
                content=result,
                parent_id=message.id
            )
            await self.communication_protocol.send_message(response)
            await self.task_manager.assign_task(message.content)
        else:
            # Handle other message types if needed
            pass

    async def _implement_decision(self, decision_result: Dict[str, Any]):
        try:
            chosen_alternative = decision_result['chosen_alternative']
            plan = decision_result['plan']
            suggested_agent = decision_result['suggested_agent']

            task = await self.task_manager.create_task(
                description=chosen_alternative,
                agent=suggested_agent
            )
            await self.task_manager.assign_task(task)

            # Implement the plan
            for step in plan:
                subtask = await self.task_manager.create_task(
                    description=step['description'],
                    agent=step.get('agent', suggested_agent)
                )
                await self.task_manager.assign_task(subtask)

        except Exception as e:
            logger.error(f"Error implementing decision: {str(e)}")
            raise AIVillageException(f"Error implementing decision: {str(e)}")

    async def process_task_completion(self, task: Dict[str, Any], result: Any):
        try:
            # Update router
            await self.router.train_model([{'task': task['description'], 'assigned_agent': task['assigned_agents'][0]}])
            
            # Update task manager
            await self.task_manager.complete_task(task['id'], result)
            
            # Update decision maker
            await self.decision_maker.update_model(task, result)
            
            # Update problem analyzer (which includes SEALEnhancedPlanGenerator)
            await self.problem_analyzer.update_models(task, result)
            
            # Update MCTS in decision maker
            await self.decision_maker.update_mcts(task, result)

            # Update the King agent
            await self.king_agent.update(task, result)

            # Record analytics
            self.unified_analytics.record_metric(f"task_type_{task['type']}_success", int(result.get('success', False)))
            self.unified_analytics.record_metric(f"agent_{task['assigned_agents'][0]}_performance", result.get('performance', 0.5))

        except Exception as e:
            logger.error(f"Error processing task completion: {str(e)}")
            raise AIVillageException(f"Error processing task completion: {str(e)}")

    async def save_models(self, path: str):
        try:
            self.router.save(f"{path}/agent_router.pt")
            await self.decision_maker.save_models(f"{path}/decision_maker")
            await self.task_manager.save_models(f"{path}/task_manager")
            await self.problem_analyzer.save_models(f"{path}/problem_analyzer")
            logger.info(f"Models saved to {path}")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise AIVillageException(f"Error saving models: {str(e)}")

    async def load_models(self, path: str):
        try:
            self.router.load(f"{path}/agent_router.pt")
            await self.decision_maker.load_models(f"{path}/decision_maker")
            await self.task_manager.load_models(f"{path}/task_manager")
            await self.problem_analyzer.load_models(f"{path}/problem_analyzer")
            logger.info(f"Models loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise AIVillageException(f"Error loading models: {str(e)}")

    async def create_final_analysis(self, revised_analyses: List[Dict[str, Any]], rag_info: Dict[str, Any]) -> Dict[str, Any]:
        try:
            combined_analysis = {
                "agent_analyses": revised_analyses,
                "rag_info": rag_info
            }
            final_analysis = await self.king_agent.generate(f"Create a final analysis based on the following information: {combined_analysis}")
            return {"final_analysis": final_analysis}
        except Exception as e:
            logger.error(f"Error creating final analysis: {str(e)}")
            raise AIVillageException(f"Error creating final analysis: {str(e)}")

    def update_agent_list(self):
        agent_list = list(self.agents.keys())
        self.router.update_agent_list(agent_list)
        logger.info(f"Updated agent list: {agent_list}")

    async def add_agent(self, agent_name: str, agent_instance):
        self.agents[agent_name] = agent_instance
        self.update_agent_list()
        logger.info(f"Added new agent: {agent_name}")

    async def remove_agent(self, agent_name: str):
        if agent_name in self.agents:
            del self.agents[agent_name]
            self.update_agent_list()
            logger.info(f"Removed agent: {agent_name}")
        else:
            logger.warning(f"Attempted to remove non-existent agent: {agent_name}")

    async def introspect(self) -> Dict[str, Any]:
        return {
            "agents": list(self.agents.keys()),
            "router_info": self.router.introspect(),
            "decision_maker_info": await self.decision_maker.introspect(),
            "task_manager_info": await self.task_manager.introspect(),
            "problem_analyzer_info": await self.problem_analyzer.introspect(),
            "analytics_summary": self.unified_analytics.generate_summary_report()
        }
