"""
Orchestration System Interfaces

Defines the common interfaces and types used across all orchestration components.
This eliminates the method signature conflicts identified in Agent 1's analysis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import uuid


class OrchestrationStatus(Enum):
    """Standard orchestration status enum."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class TaskType(Enum):
    """Unified task type classification."""
    ML_PIPELINE = "ml_pipeline"
    AGENT_LIFECYCLE = "agent_lifecycle"
    COGNITIVE_ANALYSIS = "cognitive_analysis"
    FOG_COORDINATION = "fog_coordination"
    SYSTEM_HEALTH = "system_health"
    BACKGROUND_PROCESS = "background_process"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class TaskContext:
    """Unified task context for all orchestration operations."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: TaskType = TaskType.SYSTEM_HEALTH
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    timeout_seconds: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class OrchestrationResult:
    """Standardized result type for all orchestrators."""
    success: bool
    task_id: str
    orchestrator_id: str
    task_type: TaskType
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    data: Optional[Any] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def status(self) -> str:
        """Get human-readable status."""
        if self.success:
            return "SUCCESS"
        elif self.errors:
            return f"FAILED ({len(self.errors)} errors)"
        else:
            return "UNKNOWN"


@dataclass
class HealthStatus:
    """Unified health status structure."""
    healthy: bool
    timestamp: datetime
    orchestrator_id: str
    components: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    uptime_seconds: float = 0.0
    
    @property
    def health_score(self) -> float:
        """Calculate overall health score (0.0 to 1.0)."""
        if not self.components:
            return 1.0 if self.healthy else 0.0
        
        healthy_count = sum(1 for status in self.components.values() if status)
        return healthy_count / len(self.components)


@dataclass
class ConfigurationSpec:
    """Unified configuration specification."""
    orchestrator_type: str
    config_version: str = "1.0"
    enabled: bool = True
    auto_start: bool = True
    health_check_interval: float = 30.0
    max_concurrent_tasks: int = 10
    background_process_config: Dict[str, Any] = field(default_factory=dict)
    custom_config: Dict[str, Any] = field(default_factory=dict)


class OrchestrationInterface(ABC):
    """
    Common interface for all orchestrators.
    
    This interface eliminates the method signature conflicts identified in 
    Agent 1's analysis by providing a standardized API across all orchestration
    components.
    """
    
    @property
    @abstractmethod
    def orchestrator_id(self) -> str:
        """Unique identifier for this orchestrator."""
        pass
    
    @property  
    @abstractmethod
    def status(self) -> OrchestrationStatus:
        """Current orchestrator status."""
        pass
    
    @abstractmethod
    async def initialize(self, config: Optional[ConfigurationSpec] = None) -> bool:
        """
        Initialize the orchestrator.
        
        Args:
            config: Optional configuration specification
            
        Returns:
            bool: True if initialization successful
        """
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """
        Start orchestrator operations.
        
        Returns:
            bool: True if startup successful
        """
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """
        Stop orchestrator operations gracefully.
        
        Returns:
            bool: True if shutdown successful
        """
        pass
    
    @abstractmethod
    async def process_task(self, context: TaskContext) -> OrchestrationResult:
        """
        Process a task using this orchestrator.
        
        Args:
            context: Task context with all required information
            
        Returns:
            OrchestrationResult: Standardized result
        """
        pass
    
    @abstractmethod
    async def get_health_status(self) -> HealthStatus:
        """
        Get current health status.
        
        Returns:
            HealthStatus: Current health information
        """
        pass
    
    @abstractmethod
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance and operational metrics.
        
        Returns:
            Dict[str, Any]: Current metrics data
        """
        pass


class BackgroundProcessManager(ABC):
    """Interface for background process management."""
    
    @abstractmethod
    async def start_background_processes(self) -> bool:
        """Start all background processes for this orchestrator."""
        pass
    
    @abstractmethod
    async def stop_background_processes(self) -> bool:
        """Stop all background processes for this orchestrator."""  
        pass
    
    @abstractmethod
    async def get_background_process_status(self) -> Dict[str, Any]:
        """Get status of all background processes."""
        pass


class ConfigurationManager(ABC):
    """Interface for configuration management."""
    
    @abstractmethod
    def load_configuration(self, config_path: Optional[Path] = None) -> ConfigurationSpec:
        """Load configuration from file or defaults."""
        pass
    
    @abstractmethod
    def validate_configuration(self, config: ConfigurationSpec) -> List[str]:
        """Validate configuration and return any errors."""
        pass
    
    @abstractmethod
    def save_configuration(self, config: ConfigurationSpec, config_path: Path) -> bool:
        """Save configuration to file."""
        pass