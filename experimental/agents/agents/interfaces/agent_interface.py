"""Standardized Agent Interface.

This module defines the core interface that all AIVillage agents must implement
to ensure consistent behavior, interoperability, and standardized capabilities.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from core import ErrorContext


class AgentStatus(Enum):
    """Standard agent status values."""

    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    OFFLINE = "offline"


class AgentCapability(Enum):
    """Standard agent capabilities."""

    # Core capabilities
    MESSAGE_PROCESSING = "message_processing"
    TASK_EXECUTION = "task_execution"
    QUERY_PROCESSING = "query_processing"

    # Communication capabilities
    INTER_AGENT_COMMUNICATION = "inter_agent_communication"
    BROADCAST_MESSAGING = "broadcast_messaging"
    POINT_TO_POINT_MESSAGING = "point_to_point_messaging"

    # Processing capabilities
    TEXT_PROCESSING = "text_processing"
    REASONING = "reasoning"
    PLANNING = "planning"
    DECISION_MAKING = "decision_making"

    # Knowledge capabilities
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    KNOWLEDGE_STORAGE = "knowledge_storage"
    LEARNING = "learning"
    MEMORY_MANAGEMENT = "memory_management"

    # Specialized capabilities
    CODE_GENERATION = "code_generation"
    IMAGE_PROCESSING = "image_processing"
    AUDIO_PROCESSING = "audio_processing"
    MULTIMODAL_PROCESSING = "multimodal_processing"

    # Meta capabilities
    SELF_REFLECTION = "self_reflection"
    PERFORMANCE_MONITORING = "performance_monitoring"
    ERROR_RECOVERY = "error_recovery"
    ADAPTATION = "adaptation"


@dataclass
class AgentMetadata:
    """Metadata about an agent."""

    agent_id: str
    agent_type: str
    name: str
    description: str
    version: str
    capabilities: set[AgentCapability]
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)
    configuration: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskInterface:
    """Standard task interface for agent operations."""

    task_id: str
    task_type: str
    content: Any
    priority: int = 0
    timeout_seconds: float | None = None
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "content": self.content,
            "priority": self.priority,
            "timeout_seconds": self.timeout_seconds,
            "context": self.context,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskInterface":
        """Create task from dictionary representation."""
        created_at = (
            datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now()
        )

        return cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            content=data["content"],
            priority=data.get("priority", 0),
            timeout_seconds=data.get("timeout_seconds"),
            context=data.get("context", {}),
            metadata=data.get("metadata", {}),
            created_at=created_at,
        )


@dataclass
class MessageInterface:
    """Standard message interface for agent communication."""

    message_id: str
    sender: str
    receiver: str
    message_type: str
    content: Any
    priority: int = 0
    expires_at: datetime | None = None
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type,
            "content": self.content,
            "priority": self.priority,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "context": self.context,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MessageInterface":
        """Create message from dictionary representation."""
        created_at = (
            datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now()
        )
        expires_at = (
            datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None
        )

        return cls(
            message_id=data["message_id"],
            sender=data["sender"],
            receiver=data["receiver"],
            message_type=data["message_type"],
            content=data["content"],
            priority=data.get("priority", 0),
            expires_at=expires_at,
            context=data.get("context", {}),
            metadata=data.get("metadata", {}),
            created_at=created_at,
        )


@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for agent monitoring."""

    total_tasks_processed: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_processing_time_ms: float = 0.0
    total_messages_sent: int = 0
    total_messages_received: int = 0
    uptime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_activity: datetime | None = None

    @property
    def success_rate(self) -> float:
        """Calculate task success rate."""
        if self.total_tasks_processed == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks_processed

    @property
    def error_rate(self) -> float:
        """Calculate task error rate."""
        if self.total_tasks_processed == 0:
            return 0.0
        return self.failed_tasks / self.total_tasks_processed


class AgentInterface(ABC):
    """Abstract base interface that all AIVillage agents must implement.

    This interface ensures consistency across all agent types and provides
    standardized methods for task processing, communication, and monitoring.
    """

    def __init__(self, metadata: AgentMetadata) -> None:
        self.metadata = metadata
        self.status = AgentStatus.INITIALIZING
        self.performance_metrics = AgentPerformanceMetrics()
        self._startup_time = datetime.now()

    # Core Agent Operations

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the agent and prepare it for operation.

        Returns:
            bool: True if initialization successful, False otherwise
        """

    @abstractmethod
    async def shutdown(self) -> bool:
        """Gracefully shutdown the agent.

        Returns:
            bool: True if shutdown successful, False otherwise
        """

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Perform health check and return status information.

        Returns:
            Dict containing health status and diagnostic information
        """

    # Task Processing Interface

    @abstractmethod
    async def process_task(self, task: TaskInterface) -> dict[str, Any]:
        """Process a task and return the result.

        Args:
            task: Task to process

        Returns:
            Dictionary containing task result and metadata
        """

    @abstractmethod
    async def can_handle_task(self, task: TaskInterface) -> bool:
        """Check if this agent can handle the given task.

        Args:
            task: Task to evaluate

        Returns:
            bool: True if agent can handle task, False otherwise
        """

    @abstractmethod
    async def estimate_task_duration(self, task: TaskInterface) -> float | None:
        """Estimate how long a task will take to process.

        Args:
            task: Task to estimate

        Returns:
            Estimated duration in seconds, or None if cannot estimate
        """

    # Communication Interface

    @abstractmethod
    async def send_message(self, message: MessageInterface) -> bool:
        """Send a message to another agent.

        Args:
            message: Message to send

        Returns:
            bool: True if message sent successfully, False otherwise
        """

    @abstractmethod
    async def receive_message(self, message: MessageInterface) -> None:
        """Receive and process an incoming message.

        Args:
            message: Incoming message to process
        """

    @abstractmethod
    async def broadcast_message(
        self, message: MessageInterface, recipients: list[str]
    ) -> dict[str, bool]:
        """Broadcast a message to multiple recipients.

        Args:
            message: Message to broadcast
            recipients: List of recipient agent IDs

        Returns:
            Dictionary mapping recipient IDs to success status
        """

    # Query Processing Interface (for agents with query capabilities)

    async def process_query(
        self, query: str, context: dict[str, Any] | None = None
    ) -> str:
        """Process a text query and return a response.

        Args:
            query: Query string to process
            context: Optional context information

        Returns:
            Response string

        Note:
            Default implementation converts query to task and processes it.
            Agents can override for direct query processing.
        """
        task = TaskInterface(
            task_id=f"query-{datetime.now().timestamp()}",
            task_type="query",
            content=query,
            context=context or {},
        )

        result = await self.process_task(task)
        return str(result.get("response", ""))

    # Capability Management

    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has specific capability."""
        return capability in self.metadata.capabilities

    def add_capability(self, capability: AgentCapability) -> None:
        """Add capability to agent."""
        self.metadata.capabilities.add(capability)
        self.metadata.last_updated = datetime.now()

    def remove_capability(self, capability: AgentCapability) -> None:
        """Remove capability from agent."""
        self.metadata.capabilities.discard(capability)
        self.metadata.last_updated = datetime.now()

    def get_capabilities(self) -> set[AgentCapability]:
        """Get all agent capabilities."""
        return self.metadata.capabilities.copy()

    # Status Management

    def get_status(self) -> AgentStatus:
        """Get current agent status."""
        return self.status

    def set_status(self, status: AgentStatus) -> None:
        """Set agent status."""
        self.status = status
        self.metadata.last_updated = datetime.now()

    # Performance Monitoring

    def update_performance_metrics(self, **metrics) -> None:
        """Update performance metrics."""
        for key, value in metrics.items():
            if hasattr(self.performance_metrics, key):
                setattr(self.performance_metrics, key, value)

        self.performance_metrics.last_activity = datetime.now()

        # Update uptime
        uptime = datetime.now() - self._startup_time
        self.performance_metrics.uptime_seconds = uptime.total_seconds()

    def get_performance_metrics(self) -> AgentPerformanceMetrics:
        """Get current performance metrics."""
        return self.performance_metrics

    # Metadata Management

    def get_metadata(self) -> AgentMetadata:
        """Get agent metadata."""
        return self.metadata

    def update_metadata(self, **updates) -> None:
        """Update agent metadata."""
        for key, value in updates.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)

        self.metadata.last_updated = datetime.now()

    # Error Context Creation

    def create_error_context(self, operation: str, **metadata) -> ErrorContext:
        """Create error context for this agent."""
        from core import create_agent_context

        return create_agent_context(
            agent_name=self.metadata.name,
            operation=operation,
            agent_id=self.metadata.agent_id,
            agent_type=self.metadata.agent_type,
            **metadata,
        )

    # Utility Methods

    def to_dict(self) -> dict[str, Any]:
        """Convert agent to dictionary representation."""
        return {
            "metadata": {
                "agent_id": self.metadata.agent_id,
                "agent_type": self.metadata.agent_type,
                "name": self.metadata.name,
                "description": self.metadata.description,
                "version": self.metadata.version,
                "capabilities": [cap.value for cap in self.metadata.capabilities],
                "created_at": self.metadata.created_at.isoformat(),
                "last_updated": self.metadata.last_updated.isoformat(),
                "tags": self.metadata.tags,
                "configuration": self.metadata.configuration,
            },
            "status": self.status.value,
            "performance_metrics": {
                "total_tasks_processed": self.performance_metrics.total_tasks_processed,
                "successful_tasks": self.performance_metrics.successful_tasks,
                "failed_tasks": self.performance_metrics.failed_tasks,
                "average_processing_time_ms": self.performance_metrics.average_processing_time_ms,
                "total_messages_sent": self.performance_metrics.total_messages_sent,
                "total_messages_received": self.performance_metrics.total_messages_received,
                "uptime_seconds": self.performance_metrics.uptime_seconds,
                "success_rate": self.performance_metrics.success_rate,
                "error_rate": self.performance_metrics.error_rate,
                "last_activity": (
                    self.performance_metrics.last_activity.isoformat()
                    if self.performance_metrics.last_activity
                    else None
                ),
            },
        }

    def __str__(self) -> str:
        """String representation of agent."""
        return f"{self.metadata.agent_type}({self.metadata.name})[{self.status.value}]"

    def __repr__(self) -> str:
        """Detailed string representation of agent."""
        return f"Agent(id={self.metadata.agent_id}, type={self.metadata.agent_type}, status={self.status.value})"


# Utility functions for interface validation


def validate_agent_interface(agent: Any) -> bool:
    """Validate that an object implements the AgentInterface.

    Args:
        agent: Object to validate

    Returns:
        bool: True if object implements interface correctly
    """
    required_methods = [
        "initialize",
        "shutdown",
        "health_check",
        "process_task",
        "can_handle_task",
        "estimate_task_duration",
        "send_message",
        "receive_message",
        "broadcast_message",
    ]

    for method in required_methods:
        if not hasattr(agent, method) or not callable(getattr(agent, method)):
            return False

    required_attributes = ["metadata", "status", "performance_metrics"]
    return all(hasattr(agent, attr) for attr in required_attributes)


def create_standard_task(
    task_type: str,
    content: Any,
    priority: int = 0,
    timeout_seconds: float | None = None,
    **context,
) -> TaskInterface:
    """Create a standard task with generated ID.

    Args:
        task_type: Type of task
        content: Task content
        priority: Task priority (higher = more important)
        timeout_seconds: Task timeout
        **context: Additional context

    Returns:
        TaskInterface instance
    """
    import uuid

    return TaskInterface(
        task_id=str(uuid.uuid4()),
        task_type=task_type,
        content=content,
        priority=priority,
        timeout_seconds=timeout_seconds,
        context=context,
    )


def create_standard_message(
    sender: str,
    receiver: str,
    message_type: str,
    content: Any,
    priority: int = 0,
    **context,
) -> MessageInterface:
    """Create a standard message with generated ID.

    Args:
        sender: Sender agent ID
        receiver: Receiver agent ID
        message_type: Type of message
        content: Message content
        priority: Message priority
        **context: Additional context

    Returns:
        MessageInterface instance
    """
    import uuid

    return MessageInterface(
        message_id=str(uuid.uuid4()),
        sender=sender,
        receiver=receiver,
        message_type=message_type,
        content=content,
        priority=priority,
        context=context,
    )
