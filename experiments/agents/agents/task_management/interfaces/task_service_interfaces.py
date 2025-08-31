"""
Service interfaces for the decomposed unified task management system.
Defines contracts for all services to ensure loose coupling.
"""
from abc import ABC, abstractmethod
from typing import Any, Protocol

from ..task import Task, TaskStatus
from ..unified_task_manager import Project


class ITaskCreationService(Protocol):
    """Interface for task creation operations."""
    
    @abstractmethod
    async def create_task(
        self,
        description: str,
        agent: str,
        priority: int = 1,
        deadline: str | None = None,
        project_id: str | None = None,
    ) -> Task:
        """Create a new task."""
    
    @abstractmethod
    async def create_complex_task(self, description: str, context: dict[str, Any]) -> list[Task]:
        """Create multiple tasks from complex description."""


class ITaskAssignmentService(Protocol):
    """Interface for task assignment operations."""
    
    @abstractmethod
    async def assign_task(self, task: Task) -> None:
        """Assign a task to an agent."""
    
    @abstractmethod
    async def select_best_agent_for_task(self, task_description: str) -> str:
        """Select the optimal agent for a task."""


class ITaskExecutionService(Protocol):
    """Interface for task execution operations."""
    
    @abstractmethod
    async def process_task_batch(self) -> None:
        """Process a batch of tasks concurrently."""
    
    @abstractmethod
    async def process_single_task(self, task: Task) -> Any:
        """Process a single task."""
    
    @abstractmethod
    async def start_batch_processing(self) -> None:
        """Start continuous batch processing."""


class ITaskCompletionService(Protocol):
    """Interface for task completion operations."""
    
    @abstractmethod
    async def complete_task(self, task_id: str, result: Any) -> None:
        """Complete a task and update dependencies."""
    
    @abstractmethod
    async def update_dependent_tasks(self, completed_task: Task) -> None:
        """Update tasks that depend on completed task."""


class IProjectManagementService(Protocol):
    """Interface for project management operations."""
    
    @abstractmethod
    async def create_project(self, name: str, description: str) -> str:
        """Create a new project."""
    
    @abstractmethod
    async def get_project(self, project_id: str) -> Project:
        """Get project by ID."""
    
    @abstractmethod
    async def update_project_status(
        self, project_id: str, status: str | None = None, progress: float | None = None
    ) -> None:
        """Update project status and progress."""


class IIncentiveService(Protocol):
    """Interface for incentive and performance tracking."""
    
    @abstractmethod
    async def calculate_incentive(self, agent: str, task: Task) -> float:
        """Calculate incentive for agent-task pair."""
    
    @abstractmethod
    async def update_agent_performance(self, agent: str, task_result: Any) -> None:
        """Update agent performance metrics."""


class IAnalyticsService(Protocol):
    """Interface for analytics and reporting."""
    
    @abstractmethod
    async def record_task_completion(self, task_id: str, completion_time: float, success: bool) -> None:
        """Record task completion metrics."""
    
    @abstractmethod
    async def generate_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""


class IPersistenceService(Protocol):
    """Interface for state persistence operations."""
    
    @abstractmethod
    async def save_state(self, filename: str, state_data: dict[str, Any]) -> None:
        """Save system state to file."""
    
    @abstractmethod
    async def load_state(self, filename: str) -> dict[str, Any]:
        """Load system state from file."""