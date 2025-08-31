"""Project management constants and enums."""

from dataclasses import dataclass
from enum import Enum
from typing import Final


class ProjectStatus(Enum):
    """Project status enumeration."""
    INITIALIZED = "initialized"
    ACTIVE = "active"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    FAILED = "failed"


class ProjectPriority(Enum):
    """Project priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(frozen=True)
class ProjectDefaults:
    """Default values for project management."""
    INITIAL_STATUS: Final[ProjectStatus] = ProjectStatus.INITIALIZED
    INITIAL_PROGRESS: Final[float] = 0.0
    DEFAULT_PRIORITY: Final[ProjectPriority] = ProjectPriority.MEDIUM
    
    # Dictionary keys for project data
    PROJECT_ID_KEY: Final[str] = "project_id"
    NAME_KEY: Final[str] = "name"
    STATUS_KEY: Final[str] = "status"
    PROGRESS_KEY: Final[str] = "progress"
    TASKS_KEY: Final[str] = "tasks"
    DESCRIPTION_KEY: Final[str] = "description"
    AGENT_KEY: Final[str] = "agent"
    
    # Task status keys for project task tracking
    TASK_ID_KEY: Final[str] = "task_id"
    TASK_STATUS_KEY: Final[str] = "status"
    TASK_DESCRIPTION_KEY: Final[str] = "description"


@dataclass(frozen=True)
class ProjectConstants:
    """Core project management constants."""
    
    # Status management
    DEFAULT_STATUS: Final[str] = ProjectDefaults.INITIAL_STATUS.value
    DEFAULT_PROGRESS: Final[float] = ProjectDefaults.INITIAL_PROGRESS
    
    # Field names for consistent access
    PROJECT_ID_FIELD: Final[str] = ProjectDefaults.PROJECT_ID_KEY
    NAME_FIELD: Final[str] = ProjectDefaults.NAME_KEY
    STATUS_FIELD: Final[str] = ProjectDefaults.STATUS_KEY
    PROGRESS_FIELD: Final[str] = ProjectDefaults.PROGRESS_KEY
    TASKS_FIELD: Final[str] = ProjectDefaults.TASKS_KEY
    
    # Task field names
    TASK_ID_FIELD: Final[str] = ProjectDefaults.TASK_ID_KEY
    TASK_STATUS_FIELD: Final[str] = ProjectDefaults.TASK_STATUS_KEY
    TASK_DESCRIPTION_FIELD: Final[str] = ProjectDefaults.TASK_DESCRIPTION_KEY