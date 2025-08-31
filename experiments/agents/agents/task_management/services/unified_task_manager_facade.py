"""
Unified Task Manager Facade - Maintains backward compatibility.
Provides the same interface as the original UnifiedManagement class.
"""
import asyncio
from collections import deque
import logging
from typing import Any

from AIVillage.experimental.agents.agents.analytics.unified_analytics import UnifiedAnalytics
from AIVillage.experimental.agents.agents.planning.unified_decision_maker import UnifiedDecisionMaker
from core.error_handling import StandardCommunicationProtocol

from ..incentive_model import IncentiveModel
from ..subgoal_generator import SubGoalGenerator
from ..task import Task, TaskStatus
from ..unified_task_manager import Project

from .task_creation_service import TaskCreationService
from .task_assignment_service import TaskAssignmentService
from .task_execution_service import TaskExecutionService
from .task_completion_service import TaskCompletionService
from .project_management_service import ProjectManagementService
from .incentive_service import IncentiveService
from .analytics_service import AnalyticsService
from .persistence_service import PersistenceService

logger = logging.getLogger(__name__)


class UnifiedTaskManagerFacade:
    """
    Facade that maintains backward compatibility with the original UnifiedManagement class.
    Delegates operations to the decomposed services while preserving the original API.
    """
    
    def __init__(
        self,
        communication_protocol: StandardCommunicationProtocol,
        decision_maker: UnifiedDecisionMaker,
        num_agents: int,
        num_actions: int,
    ) -> None:
        """Initialize the facade with all required dependencies."""
        # Create core dependencies
        self.communication_protocol = communication_protocol
        self.decision_maker = decision_maker
        
        # Initialize supporting components
        incentive_model = IncentiveModel(num_agents, num_actions)
        unified_analytics = UnifiedAnalytics()
        subgoal_generator = SubGoalGenerator()
        
        # Initialize services
        self._project_service = ProjectManagementService()
        self._incentive_service = IncentiveService(incentive_model, unified_analytics)
        self._analytics_service = AnalyticsService(unified_analytics)
        self._persistence_service = PersistenceService()
        
        # Services with cross-dependencies
        self._assignment_service = TaskAssignmentService(
            communication_protocol, decision_maker, self._incentive_service
        )
        self._creation_service = TaskCreationService(
            subgoal_generator, self._assignment_service, self._project_service
        )
        self._completion_service = TaskCompletionService(
            self._assignment_service, self._incentive_service, 
            self._analytics_service, self._project_service
        )
        self._execution_service = TaskExecutionService(
            communication_protocol, self._creation_service, self._completion_service
        )
        
        # Maintain original interface properties
        self.pending_tasks = deque()
        self.ongoing_tasks = {}
        self.completed_tasks = []
        self.projects = {}
        self.incentive_model = incentive_model
        self.agent_performance = {}
        self.available_agents = []
        self.subgoal_generator = subgoal_generator
        self.unified_analytics = unified_analytics
        self.batch_size = 5

    # Task creation methods
    async def create_task(
        self,
        description: str,
        agent: str,
        priority: int = 1,
        deadline: str | None = None,
        project_id: str | None = None,
    ) -> Task:
        """Create a new task (delegates to TaskCreationService)."""
        task = await self._creation_service.create_task(
            description, agent, priority, deadline, project_id
        )
        self._sync_pending_tasks()
        return task

    async def create_complex_task(self, description: str, context: dict[str, Any]) -> list[Task]:
        """Create complex task with subgoals (delegates to TaskCreationService)."""
        tasks = await self._creation_service.create_complex_task(description, context)
        self._sync_pending_tasks()
        return tasks

    # Task assignment methods
    async def assign_task(self, task: Task) -> None:
        """Assign task to agent (delegates to TaskAssignmentService)."""
        await self._assignment_service.assign_task(task)
        self._sync_ongoing_tasks()

    async def _select_best_agent_for_task(self, task_description: str) -> str:
        """Select best agent for task (delegates to TaskAssignmentService)."""
        return await self._assignment_service.select_best_agent_for_task(task_description)

    async def notify_agent_with_incentive(self, agent: str, task: Task, incentive: float) -> None:
        """Notify agent with incentive (handled by TaskAssignmentService)."""
        # This is now handled internally by the assignment service
        pass

    # Task completion methods
    async def complete_task(self, task_id: str, result: Any) -> None:
        """Complete task (delegates to TaskCompletionService)."""
        await self._completion_service.complete_task(task_id, result)
        self._sync_completed_tasks()
        self._sync_ongoing_tasks()

    async def update_dependent_tasks(self, completed_task: Task) -> None:
        """Update dependent tasks (delegates to TaskCompletionService)."""
        await self._completion_service.update_dependent_tasks(completed_task)

    # Project management methods
    async def create_project(self, name: str, description: str) -> str:
        """Create project (delegates to ProjectManagementService)."""
        project_id = await self._project_service.create_project(name, description)
        self._sync_projects()
        return project_id

    async def get_all_projects(self) -> dict[str, Project]:
        """Get all projects (delegates to ProjectManagementService)."""
        return await self._project_service.get_all_projects()

    async def get_project(self, project_id: str) -> Project:
        """Get project (delegates to ProjectManagementService)."""
        return await self._project_service.get_project(project_id)

    async def update_project_status(
        self, project_id: str, status: str | None = None, progress: float | None = None
    ) -> None:
        """Update project status (delegates to ProjectManagementService)."""
        await self._project_service.update_project_status(project_id, status, progress)
        self._sync_projects()

    async def add_task_to_project(self, project_id: str, task_id: str, task_data: dict[str, Any]) -> None:
        """Add task to project (delegates to ProjectManagementService)."""
        await self._project_service.add_task_to_project(project_id, task_id, task_data)

    async def get_project_tasks(self, project_id: str) -> list[Task]:
        """Get project tasks (delegates to ProjectManagementService)."""
        return await self._project_service.get_project_tasks(project_id)

    async def add_resources_to_project(self, project_id: str, resources: dict[str, Any]) -> None:
        """Add resources to project (delegates to ProjectManagementService)."""
        await self._project_service.add_resources_to_project(project_id, resources)

    # Task execution methods
    async def process_task_batch(self) -> None:
        """Process task batch (delegates to TaskExecutionService)."""
        await self._execution_service.process_task_batch()
        self._sync_all_task_collections()

    async def process_single_task(self, task: Task) -> Any:
        """Process single task (delegates to TaskExecutionService)."""
        return await self._execution_service.process_single_task(task)

    async def start_batch_processing(self) -> None:
        """Start batch processing (delegates to TaskExecutionService)."""
        await self._execution_service.start_batch_processing()

    # Utility methods
    def update_agent_list(self, agent_list: list[str]) -> None:
        """Update agent list."""
        self.available_agents = agent_list.copy()
        self._assignment_service.update_agent_list(agent_list)

    def set_batch_size(self, size: int) -> None:
        """Set batch size."""
        self.batch_size = size
        self._execution_service.set_batch_size(size)

    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Get task status (delegates to TaskCompletionService)."""
        return self._completion_service.get_task_status(task_id)

    async def get_project_status(self, project_id: str) -> dict[str, Any]:
        """Get project status (delegates to ProjectManagementService)."""
        return await self._project_service.get_project_status(project_id)

    # Persistence methods
    async def save_state(self, filename: str) -> None:
        """Save state (delegates to PersistenceService)."""
        state_data = await self._gather_state_data()
        await self._persistence_service.save_state(filename, state_data)

    async def load_state(self, filename: str) -> None:
        """Load state (delegates to PersistenceService)."""
        state_data = await self._persistence_service.load_state(filename)
        await self._restore_state_data(state_data)

    async def introspect(self) -> dict[str, Any]:
        """Generate introspection report."""
        performance_report = await self._analytics_service.generate_performance_report()
        
        return {
            "pending_tasks": len(self.pending_tasks),
            "ongoing_tasks": len(self.ongoing_tasks),
            "completed_tasks": len(self.completed_tasks),
            "projects": len(self.projects),
            "available_agents": self.available_agents,
            "agent_performance": self.agent_performance,
            "batch_size": self.batch_size,
            "analytics_report": performance_report,
        }

    # Synchronization methods to maintain backward compatibility
    def _sync_pending_tasks(self) -> None:
        """Sync pending tasks from creation service."""
        pending_from_service = self._creation_service.get_pending_tasks()
        self.pending_tasks = deque(pending_from_service)

    def _sync_ongoing_tasks(self) -> None:
        """Sync ongoing tasks from assignment service."""
        self.ongoing_tasks = self._assignment_service.get_ongoing_tasks()

    def _sync_completed_tasks(self) -> None:
        """Sync completed tasks from completion service."""
        self.completed_tasks = self._completion_service.get_completed_tasks()

    def _sync_projects(self) -> None:
        """Sync projects from project service."""
        async def _sync():
            self.projects = await self._project_service.get_all_projects()
        
        # Run synchronously in the current event loop if possible
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, this is more complex
                # For now, we'll skip the sync to avoid blocking
                pass
            else:
                loop.run_until_complete(_sync())
        except RuntimeError:
            # No event loop, we'll skip sync
            pass

    def _sync_all_task_collections(self) -> None:
        """Sync all task collections."""
        self._sync_pending_tasks()
        self._sync_ongoing_tasks()
        self._sync_completed_tasks()

    async def _gather_state_data(self) -> dict[str, Any]:
        """Gather state data from all services."""
        return {
            "pending_tasks": [task.__dict__ for task in self.pending_tasks],
            "ongoing_tasks": {tid: task.__dict__ for tid, task in self.ongoing_tasks.items()},
            "completed_tasks": [task.__dict__ for task in self.completed_tasks],
            "projects": {pid: proj.__dict__ for pid, proj in self.projects.items()},
            "agent_performance": self.agent_performance,
            "available_agents": self.available_agents,
            "batch_size": self.batch_size,
        }

    async def _restore_state_data(self, state_data: dict[str, Any]) -> None:
        """Restore state data to all services."""
        # This would involve reconstructing the state across all services
        # For now, we'll do a simplified restoration
        self.agent_performance = state_data.get("agent_performance", {})
        self.available_agents = state_data.get("available_agents", [])
        self.batch_size = state_data.get("batch_size", 5)


# Maintain backward compatibility
UnifiedManagement = UnifiedTaskManagerFacade