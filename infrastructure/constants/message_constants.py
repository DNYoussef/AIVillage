"""Message and communication constants."""

from dataclasses import dataclass
from enum import Enum
from typing import Final


class MessageSenders(Enum):
    """Standard message senders."""
    UNIFIED_MANAGEMENT = "UnifiedManagement"
    TASK_MANAGER = "TaskManager"
    SYSTEM = "System"
    DEFAULT_AGENT = "default_agent"


class MessageContent(Enum):
    """Standard message content fields."""
    TASK_ID = "task_id"
    DESCRIPTION = "description"
    INCENTIVE = "incentive"
    ASSIGNED_AGENT = "assigned_agent"


@dataclass(frozen=True)
class MessageDefaults:
    """Default values for message handling."""
    
    # Standard senders
    UNIFIED_MANAGEMENT_SENDER: Final[str] = MessageSenders.UNIFIED_MANAGEMENT.value
    TASK_MANAGER_SENDER: Final[str] = MessageSenders.TASK_MANAGER.value
    SYSTEM_SENDER: Final[str] = MessageSenders.SYSTEM.value
    DEFAULT_AGENT_NAME: Final[str] = MessageSenders.DEFAULT_AGENT.value
    
    # Message content field names
    TASK_ID_FIELD: Final[str] = MessageContent.TASK_ID.value
    DESCRIPTION_FIELD: Final[str] = MessageContent.DESCRIPTION.value
    INCENTIVE_FIELD: Final[str] = MessageContent.INCENTIVE.value
    ASSIGNED_AGENT_FIELD: Final[str] = MessageContent.ASSIGNED_AGENT.value
    
    # Agent selection fallback
    BEST_ALTERNATIVE_KEY: Final[str] = "best_alternative"
    
    # Decision making threshold
    DECISION_THRESHOLD: Final[float] = 0.5


@dataclass(frozen=True)
class MessageConstants:
    """Core message handling constants."""
    
    # Sender identifiers
    UNIFIED_MANAGEMENT: Final[str] = MessageDefaults.UNIFIED_MANAGEMENT_SENDER
    TASK_MANAGER: Final[str] = MessageDefaults.TASK_MANAGER_SENDER
    SYSTEM: Final[str] = MessageDefaults.SYSTEM_SENDER
    DEFAULT_AGENT: Final[str] = MessageDefaults.DEFAULT_AGENT_NAME
    
    # Content field names
    TASK_ID: Final[str] = MessageDefaults.TASK_ID_FIELD
    DESCRIPTION: Final[str] = MessageDefaults.DESCRIPTION_FIELD
    INCENTIVE: Final[str] = MessageDefaults.INCENTIVE_FIELD
    ASSIGNED_AGENT: Final[str] = MessageDefaults.ASSIGNED_AGENT_FIELD
    
    # Decision making
    BEST_ALTERNATIVE: Final[str] = MessageDefaults.BEST_ALTERNATIVE_KEY
    DECISION_THRESHOLD: Final[float] = MessageDefaults.DECISION_THRESHOLD


@dataclass(frozen=True)
class ErrorMessageConstants:
    """Error message templates for consistent error handling."""
    
    # Task error messages
    TASK_NOT_FOUND_TEMPLATE: Final[str] = "Task {task_id} not found in ongoing tasks"
    PROJECT_NOT_FOUND_TEMPLATE: Final[str] = "Project with ID {project_id} not found"
    
    # Error prefixes
    ERROR_CREATING_TASK: Final[str] = "Error creating task"
    ERROR_CREATING_COMPLEX_TASK: Final[str] = "Error creating complex task"
    ERROR_SELECTING_AGENT: Final[str] = "Error selecting best agent for task"
    ERROR_ASSIGNING_TASK: Final[str] = "Error assigning task"
    ERROR_COMPLETING_TASK: Final[str] = "Error completing task"
    ERROR_UPDATING_DEPENDENT_TASKS: Final[str] = "Error updating dependent tasks"
    ERROR_CREATING_PROJECT: Final[str] = "Error creating project"
    ERROR_UPDATING_PROJECT_STATUS: Final[str] = "Error updating project status"
    ERROR_ADDING_TASK_TO_PROJECT: Final[str] = "Error adding task to project"
    ERROR_GETTING_PROJECT_TASKS: Final[str] = "Error getting project tasks"
    ERROR_ADDING_RESOURCES: Final[str] = "Error adding resources to project"
    ERROR_UPDATING_AGENT_LIST: Final[str] = "Error updating agent list"
    ERROR_PROCESSING_TASK_BATCH: Final[str] = "Error processing task batch"
    ERROR_PROCESSING_SINGLE_TASK: Final[str] = "Error processing single task"
    ERROR_IN_BATCH_PROCESSING: Final[str] = "Error in batch processing"
    ERROR_SETTING_BATCH_SIZE: Final[str] = "Error setting batch size"
    ERROR_GETTING_TASK_STATUS: Final[str] = "Error getting task status"
    ERROR_GETTING_PROJECT_STATUS: Final[str] = "Error getting project status"
    ERROR_SAVING_STATE: Final[str] = "Error saving state"
    ERROR_LOADING_STATE: Final[str] = "Error loading state"
    ERROR_IN_INTROSPECTION: Final[str] = "Error in introspection"
    ERROR_NOTIFYING_AGENT: Final[str] = "Error notifying agent with incentive"