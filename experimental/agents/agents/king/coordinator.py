from asyncio.log import logger
from typing import TYPE_CHECKING, Any

from rag_system.core.config import UnifiedConfig

from agents.utils.task import Task as LangroidTask
from AIVillage.experimental.agents.agents.magi.magi_agent import MagiAgent
from AIVillage.experimental.agents.agents.sage.sage_agent import SageAgent
from core.error_handling import (
    AIVillageException,
    Message,
    MessageType,
    StandardCommunicationProtocol,
    error_handler,
)

from .analytics.unified_analytics import UnifiedAnalytics

if TYPE_CHECKING:
    from agents.unified_base_agent import UnifiedBaseAgent


class KingCoordinator:
    def __init__(
        self,
        config: UnifiedConfig,
        communication_protocol: StandardCommunicationProtocol,
    ) -> None:
        self.config = config
        self.communication_protocol = communication_protocol
        self.agents: dict[str, UnifiedBaseAgent] = {}
        self.task_manager = None  # Initialize this in the setup method
        self.router = None  # Initialize this in the setup method
        self.decision_maker = None  # Initialize this in the setup method
        self.problem_analyzer = None  # Initialize this in the setup method
        self.king_agent = None  # Initialize this in the setup method
        self.unified_analytics = UnifiedAnalytics()

    @error_handler.handle_error
    async def coordinate_task(self, task: dict[str, Any]) -> dict[str, Any]:
        start_time = self.unified_analytics.get_current_time()
        langroid_task = LangroidTask(
            self.king_agent,
            task.get("content"),
            task.get("id", ""),
            task.get("priority", 1),
        )
        langroid_task.type = task.get("type", "general")
        result = await self._delegate_task(langroid_task)
        end_time = self.unified_analytics.get_current_time()
        execution_time = end_time - start_time

        self.unified_analytics.record_task_completion(
            task.get("id", "unknown"), execution_time, result.get("success", False)
        )
        self.unified_analytics.record_metric(
            f"task_type_{task.get('type', 'general')}_execution_time", execution_time
        )

        return result

    @error_handler.handle_error
    async def _delegate_task(self, task: LangroidTask) -> dict[str, Any]:
        if task.type == "research":
            sage_agent = next(
                (
                    agent
                    for agent in self.agents.values()
                    if isinstance(agent, SageAgent)
                ),
                None,
            )
            if sage_agent:
                return await sage_agent.execute_task(task)
        elif task.type in ["coding", "debugging", "code_review"]:
            magi_agent = next(
                (
                    agent
                    for agent in self.agents.values()
                    if isinstance(agent, MagiAgent)
                ),
                None,
            )
            if magi_agent:
                return await magi_agent.execute_task(task)

        # If no specific agent is found, delegate to the first available agent
        if self.agents:
            return await next(iter(self.agents.values())).execute_task(task)

        msg = "No suitable agent found for the task"
        raise ValueError(msg)

    async def handle_message(self, message: Message) -> None:
        if message.type == MessageType.TASK:
            result = await self.coordinate_task(message.content)
            response = Message(
                type=MessageType.RESPONSE,
                sender="KingCoordinator",
                receiver=message.sender,
                content=result,
                parent_id=message.id,
            )
            await self.communication_protocol.send_message(response)
            await self.task_manager.assign_task(message.content)
        elif message.type == MessageType.EVIDENCE:
            logger.debug("Evidence received %s", message.content.get("id"))
        else:
            # Handle other message types if needed
            logger.warning(f"Unhandled message type: {message.type}")
            msg = f"Message type {message.type} not supported"
            raise NotImplementedError(msg)

    async def _implement_decision(self, decision_result: dict[str, Any]) -> None:
        try:
            chosen_alternative = decision_result["chosen_alternative"]
            plan = decision_result["plan"]
            suggested_agent = decision_result["suggested_agent"]

            task = await self.task_manager.create_task(
                description=chosen_alternative, agent=suggested_agent
            )
            await self.task_manager.assign_task(task)

            # Implement the plan
            for step in plan:
                subtask = await self.task_manager.create_task(
                    description=step["description"],
                    agent=step.get("agent", suggested_agent),
                )
                await self.task_manager.assign_task(subtask)

        except Exception as e:
            logger.error(f"Error implementing decision: {e!s}")
            msg = f"Error implementing decision: {e!s}"
            raise AIVillageException(msg)

    async def process_task_completion(self, task: dict[str, Any], result: Any) -> None:
        try:
            # Update router
            await self.router.train_model(
                [
                    {
                        "task": task["description"],
                        "assigned_agent": task["assigned_agents"][0],
                    }
                ]
            )

            # Update task manager
            await self.task_manager.complete_task(task["id"], result)

            # Update decision maker
            await self.decision_maker.update_model(task, result)

            # Update problem analyzer (which includes SEALEnhancedPlanGenerator)
            await self.problem_analyzer.update_models(task, result)

            # Update MCTS in decision maker
            await self.decision_maker.update_mcts(task, result)

            # Update the King agent
            await self.king_agent.update(task, result)

            # Record analytics
            self.unified_analytics.record_metric(
                f"task_type_{task['type']}_success", int(result.get("success", False))
            )
            self.unified_analytics.record_metric(
                f"agent_{task['assigned_agents'][0]}_performance",
                result.get("performance", 0.5),
            )

        except Exception as e:
            logger.error(f"Error processing task completion: {e!s}")
            msg = f"Error processing task completion: {e!s}"
            raise AIVillageException(msg)

    async def save_models(self, path: str) -> None:
        try:
            self.router.save(f"{path}/agent_router.pt")
            await self.decision_maker.save_models(f"{path}/decision_maker")
            await self.task_manager.save_models(f"{path}/task_manager")
            await self.problem_analyzer.save_models(f"{path}/problem_analyzer")
            logger.info(f"Models saved to {path}")
        except Exception as e:
            logger.error(f"Error saving models: {e!s}")
            msg = f"Error saving models: {e!s}"
            raise AIVillageException(msg)

    async def load_models(self, path: str) -> None:
        try:
            self.router.load(f"{path}/agent_router.pt")
            await self.decision_maker.load_models(f"{path}/decision_maker")
            await self.task_manager.load_models(f"{path}/task_manager")
            await self.problem_analyzer.load_models(f"{path}/problem_analyzer")
            logger.info(f"Models loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading models: {e!s}")
            msg = f"Error loading models: {e!s}"
            raise AIVillageException(msg)

    async def create_final_analysis(
        self, revised_analyses: list[dict[str, Any]], rag_info: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            combined_analysis = {
                "agent_analyses": revised_analyses,
                "rag_info": rag_info,
            }
            final_analysis = await self.king_agent.generate(
                f"Create a final analysis based on the following information: {combined_analysis}"
            )
            return {"final_analysis": final_analysis}
        except Exception as e:
            logger.error(f"Error creating final analysis: {e!s}")
            msg = f"Error creating final analysis: {e!s}"
            raise AIVillageException(msg)

    def update_agent_list(self) -> None:
        agent_list = list(self.agents.keys())
        self.router.update_agent_list(agent_list)
        logger.info(f"Updated agent list: {agent_list}")

    async def add_agent(self, agent_name: str, agent_instance) -> None:
        self.agents[agent_name] = agent_instance
        self.update_agent_list()
        logger.info(f"Added new agent: {agent_name}")

    async def remove_agent(self, agent_name: str) -> None:
        if agent_name in self.agents:
            del self.agents[agent_name]
            self.update_agent_list()
            logger.info(f"Removed agent: {agent_name}")
        else:
            logger.warning(f"Attempted to remove non-existent agent: {agent_name}")

    async def introspect(self) -> dict[str, Any]:
        return {
            "agents": list(self.agents.keys()),
            "router_info": self.router.introspect(),
            "decision_maker_info": await self.decision_maker.introspect(),
            "task_manager_info": await self.task_manager.introspect(),
            "problem_analyzer_info": await self.problem_analyzer.introspect(),
            "analytics_summary": self.unified_analytics.generate_summary_report(),
        }
