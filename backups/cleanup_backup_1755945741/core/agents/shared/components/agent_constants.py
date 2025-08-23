"""Agent system constants following connascence principles.

Centralizes all magic numbers and hardcoded values from BaseAgentTemplate
to eliminate Connascence of Meaning and establish single sources of truth.
"""

from enum import Enum


class AgentConstants:
    """Core agent system constants."""

    # Default intervals (seconds)
    SELF_AWARENESS_UPDATE_INTERVAL = 30
    MEMORY_CLEANUP_INTERVAL = 3600  # 1 hour
    PERFORMANCE_METRIC_WINDOW = 300  # 5 minutes

    # Task management
    MAX_TASK_HISTORY = 1000
    MAX_JOURNAL_ENTRIES = 100
    MAX_MEMORY_ENTRIES = 500
    MAX_GEOMETRIC_STATES = 100

    # Performance thresholds
    MAX_CPU_UTILIZATION = 0.9
    MAX_MEMORY_UTILIZATION = 0.9
    MAX_RESPONSE_LATENCY_MS = 5000
    MIN_ACCURACY_SCORE = 0.7
    MIN_STABILITY_SCORE = 0.5

    # Default fog computing resources
    DEFAULT_CPU_CORES = 2.0
    DEFAULT_MEMORY_GB = 4.0
    DEFAULT_MAX_DURATION_HOURS = 1.0

    # High-compute job resources
    HIGH_COMPUTE_CPU_CORES = 4.0
    HIGH_COMPUTE_MEMORY_GB = 8.0
    HIGH_COMPUTE_MAX_DURATION_HOURS = 2.0

    # Local execution threshold
    LOCAL_EXECUTION_LOAD_THRESHOLD = 0.7

    # Agent data paths
    AGENT_DATA_DIR_PREFIX = "data/agents"
    JOURNAL_FILENAME = "journal.json"
    MEMORY_FILENAME = "memory.json"
    CONFIG_FILENAME = "config.json"


class ReflectionConstants:
    """Quiet-star reflection system constants."""

    # Unexpectedness scoring weights
    EMOTIONAL_VALENCE_WEIGHT = 0.3
    SURPRISE_WORD_WEIGHT = 0.2
    DETAILED_INSIGHT_WEIGHT = 0.1
    DETAILED_INSIGHT_MIN_LENGTH = 200

    # Memory importance thresholds
    TRANSFORMATIVE_THRESHOLD = 0.8
    CRITICAL_THRESHOLD = 0.6
    IMPORTANT_THRESHOLD = 0.4
    NOTABLE_THRESHOLD = 0.2

    # Memory storage threshold
    MEMORY_STORAGE_THRESHOLD = 0.3

    # Surprise indicator words
    SURPRISE_WORDS = [
        "unexpected",
        "surprising",
        "unusual",
        "novel",
        "strange",
        "shocking",
        "remarkable",
        "extraordinary",
    ]


class GeometricStateConstants:
    """Geometric self-awareness constants."""

    # State thresholds
    OVERLOADED_CPU_THRESHOLD = 0.8
    OVERLOADED_MEMORY_THRESHOLD = 0.8
    UNDERUTILIZED_CPU_THRESHOLD = 0.2
    UNDERUTILIZED_TASK_THRESHOLD = 2

    # Recent modification window (seconds)
    RECENT_MODIFICATION_WINDOW = 60

    # Performance calculation weights
    EFFICIENCY_UTILIZATION_MIN = 0.1

    # Stability calculation
    STABILITY_CENTER_POINT = 0.5


class MemoryConstants:
    """Langroid memory system constants."""

    # Memory limits
    MAX_MEMORY_ENTRIES = 500

    # Decay parameters
    WEEKLY_DECAY_HOURS = 24 * 7  # 1 week
    MIN_DECAY_FACTOR = 0.1
    RETRIEVAL_BOOST_FACTOR = 0.1
    MAX_RETRIEVAL_BOOST = 2.0

    # Default retrieval threshold
    DEFAULT_RETRIEVAL_THRESHOLD = 0.3

    # Similarity calculation
    MIN_SIMILARITY_THRESHOLD = 0.0


class MCPToolConstants:
    """MCP tool system constants."""

    # Fog computing tool names
    CREATE_SANDBOX_TOOL = "create_sandbox"
    RUN_JOB_TOOL = "run_job"
    STREAM_LOGS_TOOL = "stream_logs"
    FETCH_ARTIFACTS_TOOL = "fetch_artifacts"
    FOG_JOB_STATUS_TOOL = "fog_job_status"

    # Core tool names
    RAG_QUERY_TOOL = "rag_query"
    COMMUNICATE_TOOL = "communicate"

    # RAG query modes
    RAG_MODE_FAST = "fast"
    RAG_MODE_BALANCED = "balanced"
    RAG_MODE_COMPREHENSIVE = "comprehensive"

    # Default query parameters
    DEFAULT_MAX_RESULTS = 10

    # Communication channel types
    CHANNEL_DIRECT = "direct"
    CHANNEL_BROADCAST = "broadcast"
    CHANNEL_GROUP = "group"
    CHANNEL_EMERGENCY = "emergency"

    # Default message priority
    DEFAULT_MESSAGE_PRIORITY = 5

    # Fog job parameters
    DEFAULT_WASI_RUNTIME = "wasi"
    DEFAULT_FOG_TIMEOUT = 300
    DEFAULT_LOG_TAIL_LINES = 50
    DEFAULT_LOG_TIMEOUT = 30

    # Artifact types
    ARTIFACT_STDOUT = "stdout"
    ARTIFACT_STDERR = "stderr"
    ARTIFACT_METRICS = "metrics"


class ADASConstants:
    """ADAS self-modification constants."""

    # Default configuration
    DEFAULT_ADAPTATION_RATE = 0.1
    DEFAULT_STABILITY_THRESHOLD = 0.8
    DEFAULT_ARCHITECTURE = "default"

    # Optimization targets
    OPTIMIZATION_ACCURACY = "accuracy"
    OPTIMIZATION_EFFICIENCY = "efficiency"
    OPTIMIZATION_RESPONSIVENESS = "responsiveness"

    # Performance constraints
    MAX_CPU_INCREASE = 0.2
    MAX_MEMORY_INCREASE = 0.1
    MIN_ACCURACY_THRESHOLD = 0.8

    # Historical analysis window (seconds)
    PERFORMANCE_HISTORY_WINDOW = 3600  # 1 hour


# Enum classes for type safety
class ReflectionType(Enum):
    """Types of quiet-star reflections."""

    TASK_COMPLETION = "task_completion"
    PROBLEM_SOLVING = "problem_solving"
    INTERACTION = "interaction"
    LEARNING = "learning"
    ERROR_ANALYSIS = "error_analysis"
    CREATIVE_INSIGHT = "creative_insight"


class MemoryImportance(Enum):
    """Langroid-based memory importance levels."""

    ROUTINE = 1
    NOTABLE = 3
    IMPORTANT = 5
    CRITICAL = 7
    TRANSFORMATIVE = 9


class GeometricState(Enum):
    """Geometric self-awareness states."""

    BALANCED = "balanced"
    OVERLOADED = "overloaded"
    UNDERUTILIZED = "underutilized"
    ADAPTING = "adapting"
    OPTIMIZING = "optimizing"


# Export all constants
__all__ = [
    "AgentConstants",
    "ReflectionConstants",
    "GeometricStateConstants",
    "MemoryConstants",
    "MCPToolConstants",
    "ADASConstants",
    "ReflectionType",
    "MemoryImportance",
    "GeometricState",
]
