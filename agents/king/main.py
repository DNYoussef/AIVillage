from ...communications.community_hub import CommunityHub
from .rag_management import KingRAGManagement
from ...communication.protocol import StandardCommunicationProtocol
from ...utils.agent_progress_tracker import AgentProgressTracker
from ...utils.monitoring_and_adjustment import MonitoringAndAdjustment
from ...core.sage import Sage
from ...utils.exceptions import AIVillageException
import logging

logger = logging.getLogger(__name__)

class King:
    def __init__(self):
        self.communication_protocol = StandardCommunicationProtocol()
        self.community_hub = CommunityHub(self.communication_protocol)
        self.rag_management = KingRAGManagement()
        self.progress_tracker = AgentProgressTracker()
        self.monitoring = MonitoringAndAdjustment()
        self.sage = Sage()

    async def run(self, uuid: str):
        try:
            await self.manage_community()
            await self.monitor_and_adjust_projects()
            await self.handle_rag_management()
        except Exception as e:
            logger.error(f"Error in King's run method: {str(e)}")
            raise AIVillageException(f"Error in King's run method: {str(e)}") from e

    async def manage_community(self):
        try:
            projects = await self.community_hub.get_all_projects()
            for project_id, project_data in projects.items():
                await self.assign_tasks(project_id, project_data)
                await self.update_project_status(project_id)
        except Exception as e:
            logger.error(f"Error in manage_community: {str(e)}")
            raise AIVillageException(f"Error managing community: {str(e)}") from e

    async def assign_tasks(self, project_id: str, project_data: dict):
        try:
            tasks = project_data.get('tasks', [])
            for task in tasks:
                if task['status'] == 'unassigned':
                    suitable_agent = await self.community_hub.request_collaboration(
                        requester_id='King',
                        task_id=task['id'],
                        required_capabilities=task['required_capabilities']
                    )
                    if suitable_agent:
                        await self.community_hub.assign_task(task['id'], suitable_agent)
                        self.progress_tracker.start_task(suitable_agent, task['id'])
                    else:
                        new_agent = await self.create_new_agent(task['required_capabilities'])
                        await self.community_hub.assign_task(task['id'], new_agent)
                        self.progress_tracker.start_task(new_agent, task['id'])
        except Exception as e:
            logger.error(f"Error in assign_tasks: {str(e)}")
            raise AIVillageException(f"Error assigning tasks: {str(e)}") from e

    async def update_project_status(self, project_id: str):
        try:
            project_status = self.progress_tracker.get_project_status(project_id)
            await self.community_hub.update_project_status(project_id, project_status['status'], project_status['progress'])
        except Exception as e:
            logger.error(f"Error in update_project_status: {str(e)}")
            raise AIVillageException(f"Error updating project status: {str(e)}") from e

    async def monitor_and_adjust_projects(self):
        try:
            projects = await self.community_hub.get_all_projects()
            for project_id, project_data in projects.items():
                evm_metrics = self.monitoring.calculate_evm_metrics(project_data)
                if self.monitoring.needs_adjustment(evm_metrics):
                    adjustments = self.monitoring.suggest_adjustments(evm_metrics)
                    await self.implement_adjustments(project_id, adjustments)
        except Exception as e:
            logger.error(f"Error in monitor_and_adjust_projects: {str(e)}")
            raise AIVillageException(f"Error monitoring and adjusting projects: {str(e)}") from e

    async def implement_adjustments(self, project_id: str, adjustments: list):
        try:
            for adjustment in adjustments:
                if adjustment['type'] == 'reassign_task':
                    await self.community_hub.reassign_task(adjustment['task_id'], adjustment['new_agent'])
                elif adjustment['type'] == 'add_resources':
                    await self.community_hub.add_resources_to_project(project_id, adjustment['resources'])
                # Implement other types of adjustments as needed
        except Exception as e:
            logger.error(f"Error in implement_adjustments: {str(e)}")
            raise AIVillageException(f"Error implementing adjustments: {str(e)}") from e

    async def create_new_agent(self, required_capabilities: list):
        # Placeholder for agent creation logic
        pass

    async def handle_task_result(self, agent_id: str, task_id: str, result: dict):
        try:
            self.progress_tracker.complete_task(agent_id, task_id)
            await self.community_hub.post_research_results(task_id, result)
            
            if result.get('needs_analysis', False):
                analysis = await self.sage.analyze_problem(result['content'])
                await self.community_hub.update_project_data(task_id, {'analysis': analysis})
        except Exception as e:
            logger.error(f"Error in handle_task_result: {str(e)}")
            raise AIVillageException(f"Error handling task result: {str(e)}") from e

    async def handle_rag_management(self):
        try:
            health_check_result = await self.rag_management.perform_health_check()
            if health_check_result.get('issue_detected', False):
                await self.rag_management.handle_rag_health_issue(health_check_result)
        except Exception as e:
            logger.error(f"Error in handle_rag_management: {str(e)}")
            raise AIVillageException(f"Error handling RAG management: {str(e)}") from e

    # Other King methods as needed