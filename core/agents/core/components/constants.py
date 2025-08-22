"""Agent system constants for AIVillage.

This module centralizes all agent-related magic literals to eliminate
connascence of meaning and ensure consistent agent configurations.
"""

from enum import Enum
from typing import Final

# Agent lifecycle constants
AGENT_STARTUP_TIMEOUT_SECONDS: Final[int] = 30
AGENT_SHUTDOWN_TIMEOUT_SECONDS: Final[int] = 10
AGENT_HEARTBEAT_INTERVAL_SECONDS: Final[int] = 5
AGENT_HEALTH_CHECK_INTERVAL_SECONDS: Final[int] = 15
AGENT_MAX_IDLE_TIME_MINUTES: Final[int] = 30

# Memory and performance
AGENT_MAX_MEMORY_MB: Final[int] = 512
AGENT_MAX_CPU_PERCENT: Final[int] = 50
AGENT_MESSAGE_QUEUE_SIZE: Final[int] = 1000
AGENT_TASK_QUEUE_SIZE: Final[int] = 100
AGENT_RESPONSE_CACHE_SIZE: Final[int] = 200

# Communication constants
MAX_MESSAGE_SIZE_BYTES: Final[int] = 64 * 1024  # 64KB
MESSAGE_TIMEOUT_SECONDS: Final[int] = 30
BROADCAST_TIMEOUT_SECONDS: Final[int] = 10
P2P_CONNECTION_TIMEOUT_SECONDS: Final[int] = 15
AGENT_DISCOVERY_TIMEOUT_SECONDS: Final[int] = 20

# Configuration limits
MAX_AGENT_NAME_LENGTH: Final[int] = 50
MAX_AGENT_DESCRIPTION_LENGTH: Final[int] = 500
MAX_CAPABILITIES_COUNT: Final[int] = 20
MAX_CONCURRENT_TASKS: Final[int] = 10
MAX_RETRY_ATTEMPTS: Final[int] = 3

# Logging and monitoring
LOG_RETENTION_DAYS: Final[int] = 30
METRICS_COLLECTION_INTERVAL_SECONDS: Final[int] = 60
PERFORMANCE_SAMPLE_RATE: Final[float] = 0.1
ERROR_THRESHOLD_PER_HOUR: Final[int] = 10
WARNING_THRESHOLD_PER_HOUR: Final[int] = 50

# Geometric awareness constants
GEOMETRIC_PRECISION_DECIMAL_PLACES: Final[int] = 6
MAX_SPATIAL_DIMENSIONS: Final[int] = 3
DISTANCE_CALCULATION_ALGORITHM: Final[str] = "euclidean"
SPATIAL_INDEX_BUCKET_SIZE: Final[int] = 1000
GEOMETRIC_CACHE_SIZE: Final[int] = 10000

# Memory system constants
MEMORY_WINDOW_SIZE: Final[int] = 1000
MEMORY_COMPRESSION_THRESHOLD: Final[int] = 5000
MEMORY_RETRIEVAL_LIMIT: Final[int] = 100
MEMORY_RELEVANCE_THRESHOLD: Final[float] = 0.7
MEMORY_CLEANUP_INTERVAL_HOURS: Final[int] = 24

# Reflection system constants
REFLECTION_DEPTH_LIMIT: Final[int] = 5
REFLECTION_TIMEOUT_SECONDS: Final[int] = 10
REFLECTION_CACHE_TTL_SECONDS: Final[int] = 3600
QUIET_STAR_THOUGHT_LENGTH: Final[int] = 256
REFLECTION_QUALITY_THRESHOLD: Final[float] = 0.8


class AgentType(Enum):
    """Types of agents in the system."""

    CORE = "core"
    SPECIALIZED = "specialized"
    DISTRIBUTED = "distributed"
    GOVERNANCE = "governance"
    CREATIVE = "creative"
    INFRASTRUCTURE = "infrastructure"
    COORDINATOR = "coordinator"


class AgentState(Enum):
    """Agent lifecycle states."""

    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class MessagePriority(Enum):
    """Message priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class CapabilityType(Enum):
    """Agent capability categories."""

    REASONING = "reasoning"
    MEMORY = "memory"
    COMMUNICATION = "communication"
    LEARNING = "learning"
    PERCEPTION = "perception"
    ACTION = "action"
    PLANNING = "planning"
    COORDINATION = "coordination"


# Configuration defaults
DEFAULT_AGENT_CONFIG: Final[dict[str, any]] = {
    "max_memory_mb": AGENT_MAX_MEMORY_MB,
    "max_cpu_percent": AGENT_MAX_CPU_PERCENT,
    "heartbeat_interval": AGENT_HEARTBEAT_INTERVAL_SECONDS,
    "message_timeout": MESSAGE_TIMEOUT_SECONDS,
    "max_concurrent_tasks": MAX_CONCURRENT_TASKS,
    "log_level": "INFO",
    "metrics_enabled": True,
    "health_check_enabled": True,
}

# Error handling
AGENT_ERROR_RECOVERY_ATTEMPTS: Final[int] = 3
ERROR_RECOVERY_DELAY_SECONDS: Final[int] = 5
CIRCUIT_BREAKER_FAILURE_THRESHOLD: Final[int] = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS: Final[int] = 60

# Coordination constants
CONSENSUS_TIMEOUT_SECONDS: Final[int] = 30
LEADER_ELECTION_TIMEOUT_SECONDS: Final[int] = 15
FOLLOWER_HEARTBEAT_INTERVAL_SECONDS: Final[int] = 3
CLUSTER_MEMBERSHIP_TIMEOUT_SECONDS: Final[int] = 45
Byzantine_FAULT_TOLERANCE_RATIO: Final[float] = 0.33  # f = (n-1)/3

# Performance thresholds
RESPONSE_TIME_WARNING_MS: Final[int] = 1000
RESPONSE_TIME_CRITICAL_MS: Final[int] = 5000
MEMORY_USAGE_WARNING_PERCENT: Final[int] = 80
MEMORY_USAGE_CRITICAL_PERCENT: Final[int] = 95
CPU_USAGE_WARNING_PERCENT: Final[int] = 70
CPU_USAGE_CRITICAL_PERCENT: Final[int] = 90


class AgentMessages:
    """Standardized agent system messages."""

    AGENT_STARTED: Final[str] = "Agent {agent_id} started successfully"
    AGENT_STOPPED: Final[str] = "Agent {agent_id} stopped"
    AGENT_ERROR: Final[str] = "Agent {agent_id} encountered error: {error}"
    TASK_COMPLETED: Final[str] = "Task {task_id} completed by {agent_id}"
    TASK_FAILED: Final[str] = "Task {task_id} failed: {error}"
    MEMORY_LOW: Final[str] = "Agent {agent_id} low memory: {available_mb}MB"
    COMMUNICATION_ERROR: Final[str] = "Communication error with {target_agent}: {error}"
    HEARTBEAT_MISSED: Final[str] = "Missed heartbeat from {agent_id}"
    CAPABILITY_REGISTERED: Final[str] = "Capability {capability} registered for {agent_id}"
    CONSENSUS_REACHED: Final[str] = "Consensus reached for {proposal_id}"
    LEADER_ELECTED: Final[str] = "Agent {agent_id} elected as leader"


# Metric names
class MetricNames:
    """Standard metric names for monitoring."""

    AGENT_COUNT: Final[str] = "agent.count"
    AGENT_CPU_USAGE: Final[str] = "agent.cpu.usage"
    AGENT_MEMORY_USAGE: Final[str] = "agent.memory.usage"
    AGENT_TASK_COUNT: Final[str] = "agent.task.count"
    AGENT_RESPONSE_TIME: Final[str] = "agent.response.time"
    AGENT_ERROR_RATE: Final[str] = "agent.error.rate"
    AGENT_UPTIME: Final[str] = "agent.uptime"
    MESSAGE_QUEUE_SIZE: Final[str] = "agent.message.queue.size"
    CAPABILITY_USAGE: Final[str] = "agent.capability.usage"
    CONSENSUS_TIME: Final[str] = "agent.consensus.time"


# Security constants
AGENT_AUTH_TOKEN_LENGTH: Final[int] = 64
AGENT_AUTH_TOKEN_EXPIRE_HOURS: Final[int] = 24
INTER_AGENT_ENCRYPTION_ENABLED: Final[bool] = True
AGENT_AUDIT_LOG_ENABLED: Final[bool] = True
SECURE_COMMUNICATION_REQUIRED: Final[bool] = True

# Discovery and registry
AGENT_REGISTRY_CACHE_TTL_SECONDS: Final[int] = 300
AGENT_DISCOVERY_BROADCAST_INTERVAL_SECONDS: Final[int] = 30
AGENT_REGISTRATION_TIMEOUT_SECONDS: Final[int] = 10
MAX_AGENTS_PER_NODE: Final[int] = 50
AGENT_LOAD_BALANCING_ALGORITHM: Final[str] = "round_robin"

# Specialization constants
CREATIVE_AGENT_INSPIRATION_SOURCES: Final[int] = 10
GOVERNANCE_AGENT_POLICY_CACHE_SIZE: Final[int] = 100
INFRASTRUCTURE_AGENT_MONITORING_INTERVAL: Final[int] = 30
COORDINATOR_MAX_CHILD_AGENTS: Final[int] = 20
SPECIALIZED_AGENT_SKILL_LIMIT: Final[int] = 5
