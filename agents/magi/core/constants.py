"""Constants for MAGI agent system."""

from enum import Enum, auto
from typing import Dict, Any

class TaskType(Enum):
    """Types of tasks that MAGI can handle."""
    CODING = auto()
    DEBUGGING = auto()
    CODE_REVIEW = auto()
    REVERSE_ENGINEERING = auto()
    PROBLEM_SOLVING = auto()
    RESEARCH = auto()
    ANALYSIS = auto()
    PLANNING = auto()
    OPTIMIZATION = auto()
    COLLABORATION = auto()

class TaskPriority(Enum):
    """Priority levels for tasks."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4

class TaskStatus(Enum):
    """Status of a task."""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    BLOCKED = auto()

class ToolCategory(Enum):
    """Categories of tools."""
    ANALYSIS = auto()
    GENERATION = auto()
    TRANSFORMATION = auto()
    VALIDATION = auto()
    OPTIMIZATION = auto()
    INTEGRATION = auto()
    UTILITY = auto()

class ErrorLevel(Enum):
    """Error severity levels."""
    CRITICAL = auto()
    ERROR = auto()
    WARNING = auto()
    INFO = auto()
    DEBUG = auto()

class MetricType(Enum):
    """Types of performance metrics."""
    EXECUTION_TIME = auto()
    MEMORY_USAGE = auto()
    CPU_USAGE = auto()
    SUCCESS_RATE = auto()
    QUALITY_SCORE = auto()
    EFFICIENCY_SCORE = auto()

# System-wide constants
SYSTEM_CONSTANTS: Dict[str, Any] = {
    # Version information
    "VERSION": "1.0.0",
    "API_VERSION": "v1",
    
    # Time constants (in seconds)
    "DEFAULT_TIMEOUT": 30,
    "MAX_EXECUTION_TIME": 300,
    "RETRY_INTERVAL": 5,
    "MAINTENANCE_INTERVAL": 3600,
    
    # Size limits
    "MAX_PROMPT_LENGTH": 4096,
    "MAX_RESPONSE_LENGTH": 8192,
    "MAX_HISTORY_SIZE": 1000,
    "MAX_BATCH_SIZE": 50,
    
    # Performance thresholds
    "MIN_CONFIDENCE_SCORE": 0.7,
    "MIN_QUALITY_SCORE": 0.6,
    "MAX_ERROR_RATE": 0.1,
    
    # Resource limits
    "MAX_MEMORY_PERCENT": 80,
    "MAX_CPU_PERCENT": 90,
    "MAX_CONCURRENT_TASKS": 10,
    
    # File paths
    "CONFIG_FILE": "config.yaml",
    "LOG_FILE": "magi.log",
    "CACHE_DIR": "cache",
    "DATA_DIR": "data",
    
    # API endpoints
    "API_BASE_URL": "http://localhost:8000",
    "API_TIMEOUT": 30,
    
    # Cache settings
    "CACHE_TTL": 3600,
    "MAX_CACHE_SIZE": 1024 * 1024 * 1024,  # 1GB
    
    # Security
    "MAX_LOGIN_ATTEMPTS": 3,
    "SESSION_TIMEOUT": 1800,
    "TOKEN_EXPIRY": 86400,
    
    # Development settings
    "DEBUG_MODE": False,
    "TESTING_MODE": False,
    "PROFILING_ENABLED": False
}

# HTTP status codes
HTTP_STATUS = {
    "OK": 200,
    "CREATED": 201,
    "ACCEPTED": 202,
    "NO_CONTENT": 204,
    "BAD_REQUEST": 400,
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403,
    "NOT_FOUND": 404,
    "TIMEOUT": 408,
    "CONFLICT": 409,
    "INTERNAL_ERROR": 500,
    "NOT_IMPLEMENTED": 501,
    "SERVICE_UNAVAILABLE": 503
}

# Error messages
ERROR_MESSAGES = {
    "TASK_NOT_FOUND": "Task not found",
    "INVALID_TASK_TYPE": "Invalid task type",
    "TOOL_NOT_FOUND": "Tool not found",
    "TOOL_CREATION_FAILED": "Failed to create tool",
    "EXECUTION_TIMEOUT": "Task execution timed out",
    "RESOURCE_EXHAUSTED": "System resources exhausted",
    "INVALID_CONFIG": "Invalid configuration",
    "PERMISSION_DENIED": "Permission denied",
    "VALIDATION_FAILED": "Validation failed",
    "UNKNOWN_ERROR": "An unknown error occurred"
}

# Success messages
SUCCESS_MESSAGES = {
    "TASK_CREATED": "Task created successfully",
    "TASK_COMPLETED": "Task completed successfully",
    "TOOL_CREATED": "Tool created successfully",
    "CONFIG_UPDATED": "Configuration updated successfully",
    "SYSTEM_READY": "System initialized and ready",
    "MAINTENANCE_COMPLETE": "System maintenance completed"
}

# Prompt templates
PROMPT_TEMPLATES = {
    "TASK_ANALYSIS": """
    Analyze the following task:
    {task_description}
    
    Consider:
    1. Required capabilities
    2. Potential challenges
    3. Resource requirements
    4. Success criteria
    """,
    
    "TOOL_CREATION": """
    Create a tool with the following specifications:
    Name: {tool_name}
    Purpose: {tool_purpose}
    Parameters: {tool_parameters}
    
    Provide the implementation in Python.
    """,
    
    "ERROR_ANALYSIS": """
    Analyze the following error:
    {error_message}
    
    Stack trace:
    {stack_trace}
    
    Suggest potential solutions and preventive measures.
    """,
    
    "PERFORMANCE_REVIEW": """
    Review the performance metrics:
    {performance_data}
    
    Identify:
    1. Areas of improvement
    2. Bottlenecks
    3. Optimization opportunities
    4. Success patterns
    """
}

# Default configurations
DEFAULT_CONFIG = {
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "magi.log"
    },
    "security": {
        "enable_ssl": True,
        "verify_certificates": True,
        "allowed_origins": ["localhost", "127.0.0.1"]
    },
    "performance": {
        "enable_caching": True,
        "enable_compression": True,
        "batch_size": 50
    }
}
