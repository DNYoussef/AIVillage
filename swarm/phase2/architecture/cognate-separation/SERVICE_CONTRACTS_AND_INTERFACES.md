# Service Contracts and Interfaces

## Overview

This document defines the formal contracts and interfaces that establish clear boundaries between the CognatePretrainingService and AgentForgeTrainingService, ensuring complete separation of concerns while enabling proper integration.

## Core Design Principles

### 1. Service Independence
- Each service operates completely independently
- No shared training logic or algorithms
- Separate resource pools and management
- Independent scaling and deployment

### 2. Clear Interface Contracts
- Well-defined APIs with type safety
- Explicit service boundaries
- Standardized error handling
- Consistent progress reporting

### 3. Model Lifecycle Separation
- Cognate: Creation → Pretraining → Foundation Model
- Agent Forge: Model Loading → Task Training → Specialized Agent

## Service Interface Contracts

### CognatePretrainingService Contract

```python
from typing import Protocol, AsyncIterator, Dict, List, Optional, Any
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
import uuid
from datetime import datetime

class ICognatePretrainingService(Protocol):
    """
    Formal contract for Cognate Foundation Model Pretraining.
    
    This service is exclusively responsible for creating 25M parameter 
    Cognate foundation models using specialized pretraining algorithms.
    """
    
    @abstractmethod
    async def start_cognate_pretraining(
        self,
        config: 'CognatePretrainingConfig',
        job_name: Optional[str] = None
    ) -> str:
        """
        Initiate Cognate foundation model pretraining.
        
        Args:
            config: Complete pretraining configuration including:
                   - Fixed 25M parameter architecture
                   - GrokFast optimization settings (mandatory)
                   - ACT and LTM training parameters
                   - GSM8K and HotpotQA dataset specifications
            job_name: Optional human-readable job identifier
            
        Returns:
            job_id: Unique pretraining job identifier
            
        Raises:
            ValidationError: If config doesn't meet Cognate requirements
            ResourceError: If insufficient resources for pretraining
            ServiceUnavailableError: If service is at capacity
        """
        pass
    
    @abstractmethod
    async def get_pretraining_progress(
        self,
        job_id: str
    ) -> 'CognatePretrainingProgress':
        """
        Get detailed pretraining progress with Cognate-specific metrics.
        
        Args:
            job_id: Pretraining job identifier
            
        Returns:
            Comprehensive progress including:
            - GrokFast acceleration metrics
            - ACT computation efficiency
            - LTM memory utilization
            - Mathematical reasoning convergence
            - Multi-hop QA performance
            
        Raises:
            JobNotFoundError: If job_id is invalid
            JobAccessError: If caller lacks access to job
        """
        pass
    
    @abstractmethod
    async def stream_pretraining_progress(
        self,
        job_id: str
    ) -> AsyncIterator['CognatePretrainingProgress']:
        """
        Stream real-time pretraining progress updates.
        
        Args:
            job_id: Pretraining job identifier
            
        Yields:
            Real-time progress updates with:
            - Training step progression
            - Loss convergence patterns
            - GrokFast optimization status
            - Resource utilization metrics
            
        Raises:
            JobNotFoundError: If job_id is invalid
            StreamingError: If streaming connection fails
        """
        pass
    
    @abstractmethod
    async def get_foundation_model(
        self,
        job_id: str
    ) -> 'CognateFoundationModel':
        """
        Retrieve completed Cognate foundation model.
        
        Args:
            job_id: Completed pretraining job identifier
            
        Returns:
            Complete foundation model artifact including:
            - Model weights and architecture
            - Training metadata and metrics
            - Validation results and benchmarks
            - Deployment configurations
            
        Raises:
            JobNotFoundError: If job_id is invalid
            JobNotCompletedError: If pretraining not finished
            ModelValidationError: If model failed quality checks
        """
        pass
    
    @abstractmethod
    async def validate_foundation_model(
        self,
        model_id: str,
        validation_suite: Optional[str] = None
    ) -> 'CognateValidationResult':
        """
        Run comprehensive validation on Cognate foundation model.
        
        Args:
            model_id: Foundation model identifier
            validation_suite: Specific validation tests to run
            
        Returns:
            Detailed validation results including:
            - Mathematical reasoning scores (GSM8K)
            - Multi-hop QA performance (HotpotQA)
            - ACT efficiency metrics
            - LTM memory coherence tests
            - Overall foundation model readiness
            
        Raises:
            ModelNotFoundError: If model_id is invalid
            ValidationError: If validation suite fails
        """
        pass
    
    @abstractmethod
    async def list_pretraining_jobs(
        self,
        status: Optional['CognatePretrainingStatus'] = None,
        created_after: Optional[datetime] = None
    ) -> List['CognatePretrainingJobSummary']:
        """
        List pretraining jobs with optional filtering.
        
        Args:
            status: Filter by job status
            created_after: Filter by creation date
            
        Returns:
            List of job summaries with:
            - Job identifiers and names
            - Current status and progress
            - Resource allocation
            - Estimated completion times
        """
        pass
    
    @abstractmethod
    async def cancel_pretraining(
        self,
        job_id: str,
        reason: Optional[str] = None
    ) -> bool:
        """
        Cancel active pretraining job with resource cleanup.
        
        Args:
            job_id: Job to cancel
            reason: Cancellation reason for audit
            
        Returns:
            True if successfully cancelled, False if already completed
            
        Raises:
            JobNotFoundError: If job_id is invalid
            CancellationError: If job cannot be cancelled safely
        """
        pass
```

### AgentForgeTrainingService Contract

```python
class IAgentForgeTrainingService(Protocol):
    """
    Formal contract for Agent Training and Fine-tuning.
    
    This service handles all agent training operations except 
    Cognate foundation model pretraining.
    """
    
    @abstractmethod
    async def start_agent_training(
        self,
        config: 'AgentTrainingConfig',
        priority: str = "normal"
    ) -> str:
        """
        Start agent training job for task-specific behavior.
        
        Args:
            config: Agent training configuration including:
                   - Agent architecture specification
                   - Training strategy (fine-tuning, behavioral, RL)
                   - Task-specific datasets and objectives
                   - Base model references (can include Cognate)
            priority: Job scheduling priority
            
        Returns:
            job_id: Unique training job identifier
            
        Raises:
            ConfigurationError: If config is invalid for agent training
            BaseModelError: If referenced base model is inaccessible
            ResourceError: If insufficient resources for training
        """
        pass
    
    @abstractmethod
    async def get_training_progress(
        self,
        job_id: str
    ) -> 'AgentTrainingProgress':
        """
        Get agent training progress with task-specific metrics.
        
        Args:
            job_id: Training job identifier
            
        Returns:
            Progress information including:
            - Task performance metrics
            - Behavioral adaptation progress
            - Fine-tuning convergence
            - Agent evaluation scores
            
        Raises:
            JobNotFoundError: If job_id is invalid
        """
        pass
    
    @abstractmethod
    async def get_trained_agent(
        self,
        job_id: str
    ) -> 'TrainedAgent':
        """
        Retrieve completed trained agent.
        
        Args:
            job_id: Completed training job identifier
            
        Returns:
            Complete trained agent artifact including:
            - Specialized model weights
            - Task-specific configurations
            - Performance benchmarks
            - Deployment specifications
            
        Raises:
            JobNotFoundError: If job_id is invalid
            TrainingNotCompletedError: If training not finished
        """
        pass
    
    @abstractmethod
    async def fine_tune_from_foundation(
        self,
        foundation_model_id: str,
        fine_tuning_config: 'FineTuningConfig'
    ) -> str:
        """
        Fine-tune a foundation model (e.g., Cognate) for specific tasks.
        
        Args:
            foundation_model_id: Reference to pretrained foundation model
            fine_tuning_config: Task-specific fine-tuning parameters
            
        Returns:
            job_id: Fine-tuning job identifier
            
        Note:
            This method can accept Cognate foundation models from 
            CognatePretrainingService but uses different algorithms.
        """
        pass
    
    @abstractmethod
    async def create_multi_agent_system(
        self,
        agents_config: List['AgentTrainingConfig'],
        coordination_config: 'MultiAgentConfig'
    ) -> str:
        """
        Create and train coordinated multi-agent system.
        
        Args:
            agents_config: Individual agent configurations
            coordination_config: Inter-agent coordination settings
            
        Returns:
            system_id: Multi-agent system identifier
        """
        pass
    
    @abstractmethod
    async def evaluate_agent_performance(
        self,
        agent_id: str,
        evaluation_tasks: List[str],
        benchmarks: Optional[List[str]] = None
    ) -> 'AgentEvaluationResult':
        """
        Comprehensive agent performance evaluation.
        
        Args:
            agent_id: Trained agent identifier
            evaluation_tasks: Specific tasks to evaluate
            benchmarks: Standard benchmarks to run
            
        Returns:
            Detailed evaluation results with task-specific scores
        """
        pass
```

## Data Transfer Objects (DTOs)

### Shared Base Types

```python
@dataclass
class ServiceRequest:
    """Base class for all service requests."""
    request_id: str = ""
    timestamp: datetime = None
    caller_id: str = ""
    priority: str = "normal"
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"req_{uuid.uuid4().hex[:12]}"
        if not self.timestamp:
            self.timestamp = datetime.now()

@dataclass
class ServiceResponse:
    """Base class for all service responses."""
    request_id: str
    success: bool
    timestamp: datetime
    message: Optional[str] = None
    error_code: Optional[str] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now()

class ServiceStatus(Enum):
    """Standard status codes across services."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

### Cognate-Specific DTOs

```python
@dataclass
class CognatePretrainingConfig:
    """Complete configuration for Cognate pretraining (immutable architecture)."""
    
    # Model Architecture (FIXED - Cannot be changed)
    model_parameters: int = 25_083_528  # Exact parameter count
    d_model: int = 512
    n_layers: int = 16
    n_heads: int = 8
    vocab_size: int = 32000
    max_seq_len: int = 2048
    
    # GrokFast Configuration (MANDATORY)
    grokfast_enabled: bool = True  # Cannot be disabled
    grokfast_alpha: float = 0.98
    grokfast_lambda: float = 2.0
    
    # ACT Configuration
    act_threshold: float = 0.9
    act_max_steps: int = 10
    act_penalty_weight: float = 0.01
    
    # LTM Configuration
    ltm_memory_size: int = 1024
    ltm_key_dim: int = 128
    ltm_value_dim: int = 128
    
    # Training Parameters
    learning_rate: float = 2e-4
    batch_size: int = 8
    max_steps: int = 50000
    warmup_steps: int = 5000
    
    # Datasets (FIXED - GSM8K and HotpotQA only)
    datasets: List[str] = None
    
    # Output Configuration
    output_dir: str = "./cognate_foundation_models"
    model_name: Optional[str] = None
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["GSM8K", "HotpotQA"]
        
        # Validate fixed architecture
        assert self.model_parameters == 25_083_528, "Cognate must have exactly 25M parameters"
        assert self.grokfast_enabled, "GrokFast is mandatory for Cognate pretraining"
        assert set(self.datasets) <= {"GSM8K", "HotpotQA"}, "Only GSM8K and HotpotQA allowed"

@dataclass
class CognatePretrainingProgress:
    """Detailed progress for Cognate pretraining."""
    job_id: str
    status: 'CognatePretrainingStatus'
    progress_percent: float
    current_step: int
    total_steps: int
    
    # Cognate-Specific Metrics
    grokfast_acceleration_factor: Optional[float] = None
    mathematical_reasoning_score: Optional[float] = None
    multihop_qa_score: Optional[float] = None
    act_efficiency: Optional[float] = None
    ltm_utilization: Optional[float] = None
    
    # Training Metrics
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    
    # Resource Usage
    gpu_memory_usage_gb: Optional[float] = None
    training_time_hours: Optional[float] = None
    estimated_completion_hours: Optional[float] = None
    
    # Quality Indicators
    convergence_stability: Optional[float] = None
    foundation_model_quality: Optional[float] = None

@dataclass
class CognateFoundationModel:
    """Complete Cognate foundation model artifact."""
    model_id: str
    model_name: str
    creation_timestamp: datetime
    
    # Model Specifications (Immutable)
    parameter_count: int = 25_083_528
    architecture_version: str = "cognate_v1"
    
    # Training Results
    final_training_loss: float
    mathematical_reasoning_score: float  # GSM8K performance
    multihop_qa_score: float  # HotpotQA performance
    
    # Optimization Results
    grokfast_acceleration_achieved: float
    act_average_steps: float
    ltm_memory_efficiency: float
    
    # File Locations
    model_weights_path: str
    config_path: str
    tokenizer_path: str
    training_logs_path: str
    
    # Validation Status
    foundation_model_ready: bool
    validation_scores: Dict[str, float]
    benchmark_results: Dict[str, float]
    
    # Usage Information
    recommended_fine_tuning_lr: float = 1e-5
    max_supported_sequence_length: int = 2048
    deployment_memory_requirements_gb: float = 2.0
```

### Agent Training DTOs

```python
@dataclass
class AgentTrainingConfig:
    """Flexible configuration for agent training."""
    
    # Agent Specification
    agent_name: str
    agent_architecture: 'AgentArchitecture'
    training_strategy: 'TrainingStrategy'
    task_type: str
    
    # Base Model (Optional - can reference Cognate or others)
    base_model_path: Optional[str] = None
    base_model_type: str = "none"  # "cognate", "gpt", "bert", etc.
    use_pretrained_weights: bool = True
    freeze_base_layers: bool = False
    
    # Training Parameters (Flexible)
    learning_rate: float = 1e-5
    batch_size: int = 16
    num_epochs: int = 5
    max_steps: Optional[int] = None
    
    # Optimization (NOT GrokFast - different from Cognate)
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine"
    
    # Task-Specific Data
    training_data_path: Optional[str] = None
    task_datasets: List[str] = None
    evaluation_metrics: List[str] = None
    
    # Agent-Specific Features
    behavioral_constraints: List[str] = None
    reward_function: Optional[str] = None
    multi_agent_coordination: bool = False
    
    # Output Configuration
    output_dir: str = "./trained_agents"
    save_intermediate: bool = True
    
    def __post_init__(self):
        if self.task_datasets is None:
            self.task_datasets = []
        if self.evaluation_metrics is None:
            self.evaluation_metrics = ["accuracy", "task_completion"]
        if self.behavioral_constraints is None:
            self.behavioral_constraints = []

@dataclass
class AgentTrainingProgress:
    """Progress tracking for agent training."""
    job_id: str
    agent_name: str
    status: 'AgentTrainingStatus'
    progress_percent: float
    current_step: int
    total_steps: int
    
    # Training Metrics
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    task_performance: Optional[Dict[str, float]] = None
    
    # Agent-Specific Metrics
    behavioral_adaptation_score: Optional[float] = None
    task_specialization_progress: Optional[float] = None
    multi_agent_coordination_score: Optional[float] = None
    
    # Resource Usage
    gpu_utilization: Optional[float] = None
    memory_usage_gb: Optional[float] = None
    
    # Quality Metrics
    agent_stability: Optional[float] = None
    generalization_score: Optional[float] = None

@dataclass
class TrainedAgent:
    """Complete trained agent artifact."""
    agent_id: str
    agent_name: str
    agent_type: str
    creation_timestamp: datetime
    
    # Base Model Information
    base_model_used: Optional[str] = None
    base_model_type: str = "none"
    
    # Training Information
    training_strategy: 'TrainingStrategy'
    task_specialization: str
    training_duration_hours: float
    
    # Performance Metrics
    final_training_loss: float
    task_performance_scores: Dict[str, float]
    behavioral_metrics: Dict[str, float]
    
    # Model Information
    specialized_parameter_count: int
    model_size_mb: float
    architecture_details: Dict[str, Any]
    
    # Capabilities
    supported_tasks: List[str]
    behavioral_patterns: List[str]
    deployment_requirements: Dict[str, Any]
    
    # File Locations
    model_checkpoint_path: str
    agent_config_path: str
    evaluation_results_path: str
    
    # Deployment Status
    deployment_ready: bool
    performance_benchmarks: Dict[str, float]
```

## Error Handling Contracts

### Standard Error Types

```python
class ServiceError(Exception):
    """Base class for all service errors."""
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        self.message = message
        self.error_code = error_code or "GENERIC_ERROR"
        self.details = details or {}
        super().__init__(self.message)

# Cognate Pretraining Errors
class CognatePretrainingError(ServiceError):
    """Errors specific to Cognate pretraining."""
    pass

class GrokFastOptimizationError(CognatePretrainingError):
    """Errors in GrokFast optimization."""
    pass

class CognateArchitectureError(CognatePretrainingError):
    """Errors in Cognate architecture validation."""
    pass

class FoundationModelValidationError(CognatePretrainingError):
    """Errors in foundation model validation."""
    pass

# Agent Training Errors
class AgentTrainingError(ServiceError):
    """Errors specific to agent training."""
    pass

class AgentArchitectureError(AgentTrainingError):
    """Errors in agent architecture creation."""
    pass

class TaskAdaptationError(AgentTrainingError):
    """Errors in task-specific adaptation."""
    pass

class BehavioralTrainingError(AgentTrainingError):
    """Errors in behavioral training."""
    pass

# Resource Errors
class ResourceError(ServiceError):
    """Resource allocation and management errors."""
    pass

class InsufficientResourceError(ResourceError):
    """Not enough resources available."""
    pass

class ResourceConflictError(ResourceError):
    """Resource allocation conflict."""
    pass

# Job Management Errors
class JobError(ServiceError):
    """Job lifecycle management errors."""
    pass

class JobNotFoundError(JobError):
    """Requested job does not exist."""
    pass

class JobStateError(JobError):
    """Invalid job state transition."""
    pass
```

### Error Response Format

```python
@dataclass
class ErrorResponse:
    """Standardized error response format."""
    success: bool = False
    error_code: str = ""
    error_message: str = ""
    error_details: Dict[str, Any] = None
    timestamp: datetime = None
    request_id: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now()
        if self.error_details is None:
            self.error_details = {}

class ErrorHandler:
    """Standard error handling across services."""
    
    @staticmethod
    def handle_service_error(error: Exception, request_id: str) -> ErrorResponse:
        """Convert service exceptions to standard error responses."""
        
        if isinstance(error, CognatePretrainingError):
            return ErrorResponse(
                error_code="COGNATE_PRETRAINING_ERROR",
                error_message=str(error),
                request_id=request_id,
                error_details={"service": "cognate_pretraining"}
            )
        
        elif isinstance(error, AgentTrainingError):
            return ErrorResponse(
                error_code="AGENT_TRAINING_ERROR",
                error_message=str(error),
                request_id=request_id,
                error_details={"service": "agent_training"}
            )
        
        elif isinstance(error, ResourceError):
            return ErrorResponse(
                error_code="RESOURCE_ERROR",
                error_message=str(error),
                request_id=request_id,
                error_details={"category": "resource_management"}
            )
        
        else:
            return ErrorResponse(
                error_code="UNKNOWN_ERROR",
                error_message=str(error),
                request_id=request_id,
                error_details={"type": type(error).__name__}
            )
```

## Service Communication Protocol

### Inter-Service Communication

```python
class ServiceCommunicationProtocol:
    """Defines how services communicate when needed."""
    
    @staticmethod
    async def request_foundation_model(
        model_id: str,
        requesting_service: str
    ) -> 'FoundationModelReference':
        """
        Request access to a foundation model from Cognate service.
        
        This is the ONLY communication between services - AgentForge
        can request completed Cognate models for fine-tuning.
        """
        
        # Validate request
        if requesting_service != "agent_forge_training_service":
            raise ServiceAccessError("Unauthorized service access")
        
        # Return model reference (not the model itself)
        return FoundationModelReference(
            model_id=model_id,
            model_path=f"/models/cognate/{model_id}",
            access_token="temp_access_token",
            expires_at=datetime.now() + timedelta(hours=24)
        )

@dataclass
class FoundationModelReference:
    """Reference to a foundation model (not the model itself)."""
    model_id: str
    model_path: str
    access_token: str
    expires_at: datetime
    
    # Metadata (no training logic)
    parameter_count: int = 25_083_528
    architecture: str = "cognate_v1"
    recommended_fine_tuning_lr: float = 1e-5
    
    def is_valid(self) -> bool:
        """Check if reference is still valid."""
        return datetime.now() < self.expires_at
```

## Service Registry and Discovery

### Service Registration

```python
@dataclass
class ServiceRegistration:
    """Service registration information."""
    service_id: str
    service_name: str
    service_type: str  # "cognate_pretraining" or "agent_training"
    version: str
    endpoint: str
    capabilities: List[str]
    status: str
    registered_at: datetime
    
    # Health Check
    health_check_url: str
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"

class ServiceRegistry:
    """Central registry for service discovery."""
    
    def __init__(self):
        self.services: Dict[str, ServiceRegistration] = {}
    
    async def register_service(self, registration: ServiceRegistration) -> bool:
        """Register a service."""
        self.services[registration.service_id] = registration
        return True
    
    async def discover_service(self, service_type: str) -> Optional[ServiceRegistration]:
        """Find a service by type."""
        for service in self.services.values():
            if service.service_type == service_type and service.health_status == "healthy":
                return service
        return None
    
    async def health_check_all(self) -> Dict[str, str]:
        """Perform health checks on all services."""
        health_status = {}
        
        for service_id, service in self.services.items():
            try:
                # Perform health check
                status = await self._check_service_health(service)
                service.health_status = status
                service.last_health_check = datetime.now()
                health_status[service_id] = status
            except Exception as e:
                service.health_status = "unhealthy"
                health_status[service_id] = "unhealthy"
        
        return health_status
```

## Validation and Contracts Enforcement

### Contract Validators

```python
class ContractValidator:
    """Validates service contracts and interfaces."""
    
    @staticmethod
    def validate_cognate_config(config: CognatePretrainingConfig) -> List[str]:
        """Validate Cognate pretraining configuration."""
        errors = []
        
        # Architecture validation
        if config.model_parameters != 25_083_528:
            errors.append("Cognate must have exactly 25,083,528 parameters")
        
        if not config.grokfast_enabled:
            errors.append("GrokFast optimization is mandatory for Cognate")
        
        if not set(config.datasets).issubset({"GSM8K", "HotpotQA"}):
            errors.append("Cognate only supports GSM8K and HotpotQA datasets")
        
        # Parameter validation
        if config.grokfast_alpha <= 0 or config.grokfast_alpha >= 1:
            errors.append("GrokFast alpha must be between 0 and 1")
        
        if config.learning_rate <= 0 or config.learning_rate > 1e-2:
            errors.append("Learning rate must be reasonable for pretraining")
        
        return errors
    
    @staticmethod
    def validate_agent_config(config: AgentTrainingConfig) -> List[str]:
        """Validate agent training configuration."""
        errors = []
        
        # Architecture validation
        if not config.agent_name:
            errors.append("Agent name is required")
        
        if not config.task_type:
            errors.append("Task type must be specified")
        
        # Training strategy validation
        if config.training_strategy == TrainingStrategy.FINE_TUNING:
            if not config.base_model_path:
                errors.append("Base model path required for fine-tuning")
        
        if config.training_strategy == TrainingStrategy.REINFORCEMENT_LEARNING:
            if not config.reward_function:
                errors.append("Reward function required for RL")
        
        # Parameter validation
        if config.learning_rate <= 0 or config.learning_rate > 1e-1:
            errors.append("Learning rate must be reasonable for agent training")
        
        return errors
    
    @staticmethod
    def validate_service_separation(
        cognate_config: Optional[CognatePretrainingConfig],
        agent_config: Optional[AgentTrainingConfig]
    ) -> List[str]:
        """Ensure proper separation between services."""
        errors = []
        
        # Check for inappropriate mixing
        if cognate_config and agent_config:
            errors.append("Cannot mix Cognate pretraining with agent training")
        
        # Validate Cognate exclusivity
        if cognate_config:
            if hasattr(cognate_config, 'task_type'):
                errors.append("Cognate pretraining should not have task_type")
            
            if hasattr(cognate_config, 'fine_tuning_strategy'):
                errors.append("Cognate pretraining should not have fine-tuning strategy")
        
        # Validate Agent training exclusivity
        if agent_config:
            if hasattr(agent_config, 'grokfast_alpha'):
                errors.append("Agent training should not use GrokFast parameters")
            
            if agent_config.base_model_type == "cognate":
                # This is allowed - agent can use Cognate as base
                pass
        
        return errors

class ServiceContractEnforcer:
    """Enforces service contracts at runtime."""
    
    def __init__(self):
        self.validator = ContractValidator()
    
    async def enforce_cognate_pretraining_contract(
        self,
        service_method: str,
        **kwargs
    ) -> None:
        """Enforce contract for Cognate pretraining operations."""
        
        if service_method == "start_cognate_pretraining":
            config = kwargs.get('config')
            if not isinstance(config, CognatePretrainingConfig):
                raise ContractViolationError("Invalid configuration type")
            
            errors = self.validator.validate_cognate_config(config)
            if errors:
                raise ContractViolationError(f"Configuration errors: {errors}")
        
        elif service_method == "get_foundation_model":
            # Ensure only foundation models are returned
            pass
    
    async def enforce_agent_training_contract(
        self,
        service_method: str,
        **kwargs
    ) -> None:
        """Enforce contract for agent training operations."""
        
        if service_method == "start_agent_training":
            config = kwargs.get('config')
            if not isinstance(config, AgentTrainingConfig):
                raise ContractViolationError("Invalid configuration type")
            
            errors = self.validator.validate_agent_config(config)
            if errors:
                raise ContractViolationError(f"Configuration errors: {errors}")

class ContractViolationError(Exception):
    """Raised when service contracts are violated."""
    pass
```

## Summary

These service contracts ensure:

1. **Complete Separation**: Services have entirely different interfaces and DTOs
2. **Clear Boundaries**: Each service handles distinct responsibilities
3. **Type Safety**: Strong typing prevents incorrect usage
4. **Error Handling**: Standardized error responses across services
5. **Validation**: Contract enforcement prevents service mixing
6. **Communication**: Limited, controlled communication between services
7. **Monitoring**: Health checks and service discovery

The contracts establish that:
- **CognatePretrainingService**: Creates 25M parameter foundation models using GrokFast
- **AgentForgeTrainingService**: Trains agents using various strategies, can use Cognate models as base
- **No Shared Logic**: Completely different training algorithms and approaches
- **Clear Integration**: AgentForge can reference completed Cognate models for fine-tuning