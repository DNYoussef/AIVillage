"""
Task Management Services - Decomposed architecture.

This package contains the decomposed services that replace the monolithic
UnifiedManagement class, following clean architecture principles.

Services:
- TaskCreationService: Handles task creation and validation
- TaskAssignmentService: Manages task assignment and agent selection  
- TaskExecutionService: Processes task execution and batch operations
- TaskCompletionService: Handles task completion and dependency resolution
- ProjectManagementService: Manages project lifecycle and resources
- IncentiveService: Calculates incentives and tracks agent performance
- AnalyticsService: Provides performance tracking and reporting
- PersistenceService: Handles data serialization and storage

The UnifiedTaskManagerFacade provides backward compatibility with the original API.
"""

from .task_creation_service import TaskCreationService
from .task_assignment_service import TaskAssignmentService
from .task_execution_service import TaskExecutionService
from .task_completion_service import TaskCompletionService
from .project_management_service import ProjectManagementService
from .incentive_service import IncentiveService
from .analytics_service import AnalyticsService
from .persistence_service import PersistenceService
from .unified_task_manager_facade import UnifiedTaskManagerFacade, UnifiedManagement

__all__ = [
    "TaskCreationService",
    "TaskAssignmentService",
    "TaskExecutionService",
    "TaskCompletionService",
    "ProjectManagementService",
    "IncentiveService",
    "AnalyticsService",
    "PersistenceService",
    "UnifiedTaskManagerFacade",
    "UnifiedManagement",
]
