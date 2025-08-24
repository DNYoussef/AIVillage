from asyncio.log import logger
from typing import TYPE_CHECKING, Any

from AIVillage.experimental.agents.agents.magi.magi_agent import MagiAgent
from AIVillage.experimental.agents.agents.sage.sage_agent import SageAgent
from rag_system.core.config import UnifiedConfig

from agents.utils.task import Task as LangroidTask
from core.error_handling import AIVillageException, Message, MessageType, StandardCommunicationProtocol, error_handler

from .analytics.unified_analytics import UnifiedAnalytics

# Import dual-path transport for P2P networking
try:
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../src"))
    from core.p2p.dual_path_transport import DualPathMessage, DualPathTransport

    DUAL_PATH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Dual-path transport not available: {e}")
    DualPathTransport = None
    DualPathMessage = None
    DUAL_PATH_AVAILABLE = False

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

        # Initialize dual-path P2P transport
        self.dual_path_transport: DualPathTransport = None
        self.p2p_enabled = DUAL_PATH_AVAILABLE
        if DUAL_PATH_AVAILABLE:
            self.dual_path_transport = DualPathTransport(
                node_id=f"king_{config.get('node_id', 'default')}",
                enable_bitchat=config.get("enable_bitchat", True),
                enable_betanet=config.get("enable_betanet", True),
            )
            # Register P2P message handler
            self.dual_path_transport.register_message_handler("agent_message", self._handle_p2p_message)

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
        self.unified_analytics.record_metric(f"task_type_{task.get('type', 'general')}_execution_time", execution_time)

        return result

    @error_handler.handle_error
    async def _delegate_task(self, task: LangroidTask) -> dict[str, Any]:
        if task.type == "research":
            sage_agent = next(
                (agent for agent in self.agents.values() if isinstance(agent, SageAgent)),
                None,
            )
            if sage_agent:
                return await sage_agent.execute_task(task)
        elif task.type in ["coding", "debugging", "code_review"]:
            magi_agent = next(
                (agent for agent in self.agents.values() if isinstance(agent, MagiAgent)),
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

    async def start_p2p_transport(self) -> bool:
        """Start dual-path P2P transport for mesh networking"""
        if not self.p2p_enabled or not self.dual_path_transport:
            logger.info("P2P transport not available")
            return False

        try:
            success = await self.dual_path_transport.start()
            if success:
                logger.info("Dual-path P2P transport started successfully")
                # Enable Global South optimizations for offline-first routing
                if hasattr(self.dual_path_transport.navigator, "enable_global_south_mode"):
                    self.dual_path_transport.navigator.enable_global_south_mode(True)
            return success
        except Exception as e:
            logger.exception(f"Failed to start P2P transport: {e}")
            return False

    async def stop_p2p_transport(self) -> None:
        """Stop dual-path P2P transport"""
        if self.dual_path_transport:
            await self.dual_path_transport.stop()
            logger.info("Dual-path P2P transport stopped")

    async def send_p2p_message(
        self,
        recipient: str,
        message_content: dict,
        priority: int = 5,
        privacy_required: bool = False,
    ) -> bool:
        """Send message to another agent via P2P mesh network"""
        if not self.p2p_enabled or not self.dual_path_transport:
            logger.warning("P2P transport not available for message sending")
            return False

        try:
            # Create agent message format
            agent_message = {
                "type": "agent_coordination",
                "from_agent": "king_coordinator",
                "to_agent": recipient,
                "content": message_content,
                "timestamp": self.unified_analytics.get_current_time(),
                "requires_response": message_content.get("requires_response", False),
            }

            # Send via dual-path transport
            success = await self.dual_path_transport.send_message(
                recipient=recipient,
                payload=agent_message,
                priority=priority,
                privacy_required=privacy_required,
            )

            if success:
                logger.info(f"Sent P2P message to {recipient}")
            else:
                logger.warning(f"Failed to send P2P message to {recipient}")

            return success

        except Exception as e:
            logger.exception(f"Error sending P2P message: {e}")
            return False

    async def broadcast_to_mesh(self, message_content: dict, priority: int = 5) -> int:
        """Broadcast message to all agents in mesh network"""
        if not self.p2p_enabled or not self.dual_path_transport:
            logger.warning("P2P transport not available for broadcasting")
            return 0

        try:
            # Create broadcast message
            broadcast_message = {
                "type": "agent_broadcast",
                "from_agent": "king_coordinator",
                "content": message_content,
                "timestamp": self.unified_analytics.get_current_time(),
                "broadcast_id": f"broadcast_{self.unified_analytics.get_current_time()}",
            }

            # Broadcast via dual-path transport
            peer_count = await self.dual_path_transport.broadcast_message(payload=broadcast_message, priority=priority)

            logger.info(f"Broadcast message to {peer_count} peers in mesh network")
            return peer_count

        except Exception as e:
            logger.exception(f"Error broadcasting to mesh: {e}")
            return 0

    async def _handle_p2p_message(self, dual_path_msg, source_protocol: str) -> None:
        """Handle incoming P2P message from dual-path transport"""
        try:
            # Parse message payload
            if isinstance(dual_path_msg.payload, bytes):
                import json

                message_data = json.loads(dual_path_msg.payload.decode())
            else:
                message_data = dual_path_msg.payload

            logger.info(f"Received P2P message via {source_protocol} from {dual_path_msg.sender}")

            # Handle different message types
            msg_type = message_data.get("type", "unknown")

            if msg_type == "agent_coordination":
                await self._handle_agent_coordination_message(message_data, dual_path_msg.sender)
            elif msg_type == "agent_broadcast":
                await self._handle_agent_broadcast_message(message_data, dual_path_msg.sender)
            elif msg_type == "task_delegation":
                await self._handle_task_delegation_message(message_data, dual_path_msg.sender)
            else:
                logger.warning(f"Unknown P2P message type: {msg_type}")

        except Exception as e:
            logger.exception(f"Error handling P2P message: {e}")

    async def _handle_agent_coordination_message(self, message_data: dict, sender: str) -> None:
        """Handle agent coordination message"""
        content = message_data.get("content", {})

        if content.get("type") == "task_request":
            # Another agent is requesting task delegation
            task = content.get("task", {})
            result = await self.coordinate_task(task)

            # Send response back if requested
            if message_data.get("requires_response", False):
                response = {
                    "type": "task_response",
                    "original_task": task,
                    "result": result,
                    "success": result.get("success", False),
                }

                await self.send_p2p_message(sender, response, priority=7)

        elif content.get("type") == "status_update":
            # Agent status update
            agent_status = content.get("status", {})
            self.unified_analytics.record_metric(f"agent_{sender}_status", agent_status)
            logger.info(f"Status update from agent {sender}: {agent_status}")

    async def _handle_agent_broadcast_message(self, message_data: dict, sender: str) -> None:
        """Handle agent broadcast message"""
        content = message_data.get("content", {})

        if content.get("type") == "network_discovery":
            # Agent announcing presence in network
            agent_info = content.get("agent_info", {})
            logger.info(f"Network discovery from {sender}: {agent_info}")

        elif content.get("type") == "emergency_alert":
            # Emergency alert from agent
            alert = content.get("alert", {})
            logger.warning(f"Emergency alert from {sender}: {alert}")

            # Potentially coordinate emergency response
            emergency_task = {
                "type": "emergency_response",
                "alert": alert,
                "source_agent": sender,
                "priority": 10,
            }
            await self.coordinate_task(emergency_task)

    async def _handle_task_delegation_message(self, message_data: dict, sender: str) -> None:
        """Handle task delegation from other agents"""
        task = message_data.get("task", {})

        # Delegate task through normal coordination
        result = await self.coordinate_task(task)

        # Send result back to requesting agent
        response = {
            "type": "delegation_result",
            "original_task": task,
            "result": result,
            "coordinator": "king_coordinator",
        }

        await self.send_p2p_message(sender, response, priority=6)

    def get_p2p_status(self) -> dict[str, Any]:
        """Get status of P2P mesh networking"""
        if not self.p2p_enabled or not self.dual_path_transport:
            return {"enabled": False, "reason": "not_available"}

        return {
            "enabled": True,
            "transport_status": self.dual_path_transport.get_status(),
            "reachable_peers": self.dual_path_transport.get_reachable_peers(),
        }

    async def _implement_decision(self, decision_result: dict[str, Any]) -> None:
        try:
            chosen_alternative = decision_result["chosen_alternative"]
            plan = decision_result["plan"]
            suggested_agent = decision_result["suggested_agent"]

            task = await self.task_manager.create_task(description=chosen_alternative, agent=suggested_agent)
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
            self.unified_analytics.record_metric(f"task_type_{task['type']}_success", int(result.get("success", False)))
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
