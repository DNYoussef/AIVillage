"""Task management constants and type-safe enums."""

from dataclasses import dataclass
from enum import Enum
from typing import Final


class TaskType(Enum):
    """Task type enumeration for action mapping."""

    CRITICAL = "critical"
    ROUTINE = "routine"
    ANALYSIS = "analysis"
    DEFAULT = "default"


class TaskComplexityLevel(Enum):
    """Task complexity levels."""

    SIMPLE = 1
    MODERATE = 3
    COMPLEX = 5
    VERY_COMPLEX = 7
    EXTREMELY_COMPLEX = 10


class TaskPriorityLevel(Enum):
    """Task priority levels."""

    LOWEST = 1
    LOW = 2
    NORMAL = 3
    HIGH = 4
    URGENT = 5
    CRITICAL = 6


@dataclass(frozen=True)
class TaskDefaults:
    """Default values for task creation and management."""

    PRIORITY: Final[int] = 1
    BATCH_SIZE: Final[int] = 5
    MAX_RETRIES: Final[int] = 3
    DEFAULT_COMPLEXITY: Final[int] = 1
    DEFAULT_ESTIMATED_TIME: Final[int] = 1
    ROUTINE_PRIORITY_THRESHOLD: Final[int] = 5
    HIGH_COMPLEXITY_THRESHOLD: Final[int] = 7


@dataclass(frozen=True)
class TaskLimits:
    """Limits and validation constraints for tasks."""

    MAX_PRIORITY: Final[int] = 10
    MIN_PRIORITY: Final[int] = 1
    MAX_COMPLEXITY: Final[int] = 10
    MIN_COMPLEXITY: Final[int] = 1
    MAX_BATCH_SIZE: Final[int] = 100
    MIN_BATCH_SIZE: Final[int] = 1
    MAX_DESCRIPTION_LENGTH: Final[int] = 1000
    MAX_DEPENDENCIES: Final[int] = 50


@dataclass(frozen=True)
class TaskConstants:
    """Core task management constants."""

    # Default values
    DEFAULT_PRIORITY: Final[int] = TaskDefaults.PRIORITY
    DEFAULT_BATCH_SIZE: Final[int] = TaskDefaults.BATCH_SIZE
    MAX_RETRIES: Final[int] = TaskDefaults.MAX_RETRIES

    # Action mapping constants
    CRITICAL_ACTION_ID: Final[int] = 0
    ROUTINE_HIGH_PRIORITY_ACTION_ID: Final[int] = 1
    ANALYSIS_ACTION_ID: Final[int] = 2
    HIGH_COMPLEXITY_ACTION_ID: Final[int] = 3
    DEFAULT_ACTION_ID: Final[int] = 4

    # Task type keywords
    ANALYSIS_KEYWORD: Final[str] = "analysis"

    # Performance thresholds
    ROUTINE_PRIORITY_THRESHOLD: Final[int] = TaskDefaults.ROUTINE_PRIORITY_THRESHOLD
    HIGH_COMPLEXITY_THRESHOLD: Final[int] = TaskDefaults.HIGH_COMPLEXITY_THRESHOLD

    # Validation limits
    MAX_PRIORITY: Final[int] = TaskLimits.MAX_PRIORITY
    MIN_PRIORITY: Final[int] = TaskLimits.MIN_PRIORITY
    MAX_BATCH_SIZE: Final[int] = TaskLimits.MAX_BATCH_SIZE
    MIN_BATCH_SIZE: Final[int] = TaskLimits.MIN_BATCH_SIZE


class TaskActionMapping(Enum):
    """Maps task characteristics to action IDs."""

    CRITICAL = 0
    ROUTINE_HIGH_PRIORITY = 1
    ANALYSIS = 2
    HIGH_COMPLEXITY = 3
    DEFAULT = 4
