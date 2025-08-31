# Phase 4 Service Extraction Implementation Guide

## Overview

This guide provides detailed implementation instructions for extracting services from the UnifiedManagement god class and reducing SageAgent dependencies. Each service is designed with clear interfaces, single responsibility, and comprehensive testing.

## Service Architecture Design

### Core Service Interfaces

#### 1. TaskService

**Responsibility**: Complete task lifecycle management

```python
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from dataclasses import dataclass
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    id: str
    description: str
    assigned_agents: List[str]
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 1
    deadline: Optional[str] = None
    dependencies: List[str] = None
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = datetime.now()

class ITaskService(ABC):
    @abstractmethod
    async def create_task(
        self,
        description: str,
        agent: str,
        priority: int = 1,
        deadline: Optional[str] = None,
        dependencies: Optional[List[str]] = None
    ) -> Task:
        """Create a new task with specified parameters."""
        pass

    @abstractmethod
    async def assign_task(self, task_id: str, agent: str) -> None:
        """Assign task to specific agent."""
        pass

    @abstractmethod
    async def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        """Update task status."""
        pass

    @abstractmethod
    async def complete_task(self, task_id: str, result: Any) -> None:
        """Mark task as completed with result."""
        pass

    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve task by ID."""
        pass

    @abstractmethod
    async def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Get all tasks with specific status."""
        pass

    @abstractmethod
    async def get_tasks_by_agent(self, agent: str) -> List[Task]:
        """Get all tasks assigned to agent."""
        pass
```

**Implementation**:

```python
import asyncio
import uuid
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Any

class TaskService(ITaskService):
    def __init__(self, task_repository: ITaskRepository):
        self._repository = task_repository
        self._task_cache: Dict[str, Task] = {}
        
    async def create_task(
        self,
        description: str,
        agent: str,
        priority: int = 1,
        deadline: Optional[str] = None,
        dependencies: Optional[List[str]] = None
    ) -> Task:
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            description=description,
            assigned_agents=[agent],
            priority=priority,
            deadline=deadline,
            dependencies=dependencies or []
        )
        
        await self._repository.save(task)
        self._task_cache[task_id] = task
        return task

    async def assign_task(self, task_id: str, agent: str) -> None:
        task = await self.get_task(task_id)
        if not task:
            raise TaskNotFoundError(f"Task {task_id} not found")
        
        task.assigned_agents = [agent]
        task.status = TaskStatus.IN_PROGRESS
        await self._repository.save(task)
        self._task_cache[task_id] = task

    async def complete_task(self, task_id: str, result: Any) -> None:
        task = await self.get_task(task_id)
        if not task:
            raise TaskNotFoundError(f"Task {task_id} not found")
        
        task.status = TaskStatus.COMPLETED
        task.result = result
        task.completed_at = datetime.now()
        
        await self._repository.save(task)
        self._task_cache[task_id] = task
        
        # Trigger dependent tasks
        await self._process_dependent_tasks(task_id)

    async def _process_dependent_tasks(self, completed_task_id: str) -> None:
        """Process tasks that depend on the completed task."""
        dependent_tasks = await self._repository.get_tasks_depending_on(completed_task_id)
        
        for task in dependent_tasks:
            task.dependencies.remove(completed_task_id)
            if not task.dependencies:  # All dependencies satisfied
                # Task is ready for assignment
                await self._repository.save(task)

    # ... (implement remaining interface methods)
```

#### 2. ProjectService

**Responsibility**: Project management and tracking

```python
@dataclass
class Project:
    id: str
    name: str
    description: str
    status: str = "initialized"
    progress: float = 0.0
    tasks: Dict[str, str] = None  # task_id -> task_id mapping
    resources: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.tasks is None:
            self.tasks = {}
        if self.resources is None:
            self.resources = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        self.updated_at = datetime.now()

class IProjectService(ABC):
    @abstractmethod
    async def create_project(self, name: str, description: str) -> str:
        """Create a new project and return its ID."""
        pass

    @abstractmethod
    async def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID."""
        pass

    @abstractmethod
    async def update_project_status(self, project_id: str, status: str) -> None:
        """Update project status."""
        pass

    @abstractmethod
    async def add_task_to_project(self, project_id: str, task_id: str) -> None:
        """Associate task with project."""
        pass

    @abstractmethod
    async def calculate_project_progress(self, project_id: str) -> float:
        """Calculate project completion progress."""
        pass

class ProjectService(IProjectService):
    def __init__(
        self,
        project_repository: IProjectRepository,
        task_service: ITaskService
    ):
        self._repository = project_repository
        self._task_service = task_service

    async def create_project(self, name: str, description: str) -> str:
        project_id = str(uuid.uuid4())
        project = Project(
            id=project_id,
            name=name,
            description=description
        )
        
        await self._repository.save(project)
        return project_id

    async def calculate_project_progress(self, project_id: str) -> float:
        """Calculate completion percentage based on task statuses."""
        project = await self.get_project(project_id)
        if not project or not project.tasks:
            return 0.0

        total_tasks = len(project.tasks)
        completed_tasks = 0

        for task_id in project.tasks.keys():
            task = await self._task_service.get_task(task_id)
            if task and task.status == TaskStatus.COMPLETED:
                completed_tasks += 1

        progress = (completed_tasks / total_tasks) * 100
        
        # Update project progress
        project.progress = progress
        project.updated_at = datetime.now()
        await self._repository.save(project)
        
        return progress

    # ... (implement remaining methods)
```

#### 3. AgentCoordinationService

**Responsibility**: Agent assignment and communication

```python
class IAgentCoordinationService(ABC):
    @abstractmethod
    async def select_best_agent(self, task_description: str) -> str:
        """Select optimal agent for task."""
        pass

    @abstractmethod
    async def notify_agent(self, agent: str, message: Any) -> None:
        """Send notification to agent."""
        pass

    @abstractmethod
    async def get_agent_workload(self, agent: str) -> int:
        """Get current workload for agent."""
        pass

    @abstractmethod
    async def update_agent_performance(
        self, 
        agent: str, 
        task_id: str, 
        performance_score: float
    ) -> None:
        """Update agent performance metrics."""
        pass

class AgentCoordinationService(IAgentCoordinationService):
    def __init__(
        self,
        communication_protocol: StandardCommunicationProtocol,
        decision_maker: IDecisionMaker,
        agent_registry: IAgentRegistry
    ):
        self._communication = communication_protocol
        self._decision_maker = decision_maker
        self._agent_registry = agent_registry
        self._agent_workloads: Dict[str, int] = {}
        self._agent_performance: Dict[str, List[float]] = {}

    async def select_best_agent(self, task_description: str) -> str:
        """Select agent using decision maker and workload balancing."""
        available_agents = await self._agent_registry.get_available_agents()
        
        if not available_agents:
            raise NoAgentsAvailableError("No agents available for task assignment")

        # Consider workload and performance in selection
        agent_scores = {}
        for agent in available_agents:
            workload = self._agent_workloads.get(agent, 0)
            performance_history = self._agent_performance.get(agent, [1.0])
            avg_performance = sum(performance_history) / len(performance_history)
            
            # Lower workload and higher performance = better score
            agent_scores[agent] = avg_performance / max(workload, 1)

        # Use decision maker for final selection
        decision_context = {
            "task_description": task_description,
            "agent_scores": agent_scores,
            "available_agents": available_agents
        }
        
        decision = await self._decision_maker.make_decision(
            f"Select best agent for: {task_description}",
            decision_context
        )
        
        selected_agent = decision.get("best_alternative", available_agents[0])
        
        # Update workload
        self._agent_workloads[selected_agent] = self._agent_workloads.get(selected_agent, 0) + 1
        
        return selected_agent

    async def notify_agent(self, agent: str, message: Any) -> None:
        """Send structured message to agent."""
        notification = Message(
            type=MessageType.NOTIFICATION,
            sender="AgentCoordinationService",
            receiver=agent,
            content=message,
            priority=Priority.MEDIUM
        )
        
        await self._communication.send_message(notification)

    # ... (implement remaining methods)
```

### Service Locator Implementation

```python
from typing import Dict, Type, TypeVar, Any, Callable
import logging

T = TypeVar('T')

class ServiceLocator:
    """Centralized service registry and locator."""
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable[[], Any]] = {}
        self._singletons: Dict[Type, Any] = {}
        self._logger = logging.getLogger(__name__)

    def register_service(self, service_type: Type[T], service_instance: T) -> None:
        """Register a service instance."""
        self._services[service_type] = service_instance
        self._logger.debug(f"Registered service: {service_type.__name__}")

    def register_factory(self, service_type: Type[T], factory: Callable[[], T]) -> None:
        """Register a factory function for creating services."""
        self._factories[service_type] = factory
        self._logger.debug(f"Registered factory for: {service_type.__name__}")

    def register_singleton(self, service_type: Type[T], service_instance: T) -> None:
        """Register a singleton service."""
        self._singletons[service_type] = service_instance
        self._logger.debug(f"Registered singleton: {service_type.__name__}")

    def get_service(self, service_type: Type[T]) -> T:
        """Get service instance by type."""
        # Check singletons first
        if service_type in self._singletons:
            return self._singletons[service_type]
        
        # Check registered instances
        if service_type in self._services:
            return self._services[service_type]
        
        # Try factory
        if service_type in self._factories:
            instance = self._factories[service_type]()
            self._services[service_type] = instance
            return instance
        
        raise ServiceNotRegisteredError(f"Service {service_type.__name__} not registered")

    def has_service(self, service_type: Type[T]) -> bool:
        """Check if service is registered."""
        return (
            service_type in self._services or 
            service_type in self._factories or 
            service_type in self._singletons
        )

# Global service locator instance
_service_locator = ServiceLocator()

def get_service_locator() -> ServiceLocator:
    """Get global service locator instance."""
    return _service_locator
```

### ProcessingChainFactory for SageAgent

```python
from typing import List, Dict, Any, Optional

class ICognitiveLayer(ABC):
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        pass

class ProcessingChainFactory:
    """Factory for creating cognitive processing chains."""
    
    def __init__(self, service_locator: ServiceLocator):
        self._service_locator = service_locator
        self._chain_cache: Dict[str, List[ICognitiveLayer]] = {}

    def create_standard_chain(self) -> List[ICognitiveLayer]:
        """Create standard cognitive processing chain."""
        chain_key = "standard"
        
        if chain_key in self._chain_cache:
            return self._chain_cache[chain_key]
        
        chain = [
            self._service_locator.get_service(FoundationalLayer),
            self._service_locator.get_service(ContinuousLearningLayer),
            self._service_locator.get_service(CognitiveNexus),
            self._service_locator.get_service(LatentSpaceActivation)
        ]
        
        self._chain_cache[chain_key] = chain
        return chain

    def create_research_chain(self) -> List[ICognitiveLayer]:
        """Create research-optimized processing chain."""
        chain_key = "research"
        
        if chain_key in self._chain_cache:
            return self._chain_cache[chain_key]
        
        chain = [
            self._service_locator.get_service(FoundationalLayer),
            self._service_locator.get_service(ResearchCapabilities),
            self._service_locator.get_service(CognitiveNexus),
            self._service_locator.get_service(ExplorationMode)
        ]
        
        self._chain_cache[chain_key] = chain
        return chain

    def create_custom_chain(self, layer_types: List[Type[ICognitiveLayer]]) -> List[ICognitiveLayer]:
        """Create custom processing chain."""
        return [self._service_locator.get_service(layer_type) for layer_type in layer_types]
```

### Refactored SageAgent

```python
class SageAgent(UnifiedBaseAgent):
    """Refactored SageAgent with reduced coupling."""
    
    def __init__(
        self,
        config: UnifiedConfig,
        service_locator: ServiceLocator,
        communication_protocol: StandardCommunicationProtocol,
        knowledge_tracker: Optional[UnifiedKnowledgeTracker] = None,
    ):
        super().__init__(config, communication_protocol, knowledge_tracker)
        self._service_locator = service_locator
        self._processing_chain_factory = ProcessingChainFactory(service_locator)
        
        # Core required services (injected directly)
        self._config = config
        self._communication = communication_protocol
        self._knowledge_tracker = knowledge_tracker
        
        # Lazy-loaded services (accessed via service locator)
        self._rag_system = None
        self._vector_store = None
        self._processing_chain = None
        
        # Performance metrics
        self.performance_metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_execution_time": 0,
        }

    @property
    def rag_system(self) -> EnhancedRAGPipeline:
        """Lazy-loaded RAG system."""
        if self._rag_system is None:
            self._rag_system = self._service_locator.get_service(EnhancedRAGPipeline)
        return self._rag_system

    @property
    def vector_store(self) -> VectorStore:
        """Lazy-loaded vector store."""
        if self._vector_store is None:
            self._vector_store = self._service_locator.get_service(VectorStore)
        return self._vector_store

    @property
    def processing_chain(self) -> List[ICognitiveLayer]:
        """Lazy-loaded processing chain."""
        if self._processing_chain is None:
            self._processing_chain = self._processing_chain_factory.create_standard_chain()
        return self._processing_chain

    async def execute_task(self, task: LangroidTask) -> Dict[str, Any]:
        """Execute task using service-locator pattern."""
        self.performance_metrics["total_tasks"] += 1
        start_time = time.time()
        
        try:
            # Process through cognitive chain
            result = await self._process_through_chain(task)
            self.performance_metrics["successful_tasks"] += 1
            return result
            
        except Exception as e:
            self.performance_metrics["failed_tasks"] += 1
            error_controller = self._service_locator.get_service(AdaptiveErrorController)
            return await error_controller.handle_error(e, task)
            
        finally:
            execution_time = time.time() - start_time
            self._update_performance_metrics(execution_time)

    async def _process_through_chain(self, task: LangroidTask) -> Dict[str, Any]:
        """Process task through cognitive processing chain."""
        current_data = {"task": task, "content": task.content}
        
        for layer in self.processing_chain:
            current_data = await layer.process(current_data)
        
        return current_data

    # ... (simplified implementation with service locator)
```

## Constants Consolidation Strategy

### 1. Domain-Organized Constants

```python
# config/TaskConstants.py
from enum import Enum
from typing import Final

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class TaskTimeout(Enum):
    DEFAULT_TIMEOUT_SECONDS: Final = 300
    LONG_RUNNING_TIMEOUT_SECONDS: Final = 1800
    BATCH_TIMEOUT_SECONDS: Final = 600

# Task lifecycle constants
TASK_MAX_RETRIES: Final = 3
TASK_RETRY_DELAY_SECONDS: Final = 5
TASK_DEPENDENCY_CHECK_INTERVAL: Final = 10

# Task validation
MAX_TASK_DESCRIPTION_LENGTH: Final = 1000
MIN_TASK_DESCRIPTION_LENGTH: Final = 10
MAX_CONCURRENT_TASKS_PER_AGENT: Final = 5
```

```python
# config/AgentConstants.py
from typing import Final

# Agent performance thresholds
AGENT_PERFORMANCE_EXCELLENT_THRESHOLD: Final = 0.9
AGENT_PERFORMANCE_GOOD_THRESHOLD: Final = 0.7
AGENT_PERFORMANCE_POOR_THRESHOLD: Final = 0.4

# Agent workload limits
MAX_AGENT_CONCURRENT_TASKS: Final = 10
AGENT_WORKLOAD_REBALANCE_THRESHOLD: Final = 8
AGENT_IDLE_TIMEOUT_MINUTES: Final = 30

# Communication timeouts
AGENT_RESPONSE_TIMEOUT_SECONDS: Final = 60
AGENT_HEARTBEAT_INTERVAL_SECONDS: Final = 30
```

### 2. Configuration Management System

```python
from typing import Dict, Any, Optional
import os
import json
from pathlib import Path

class ConfigurationManager:
    """Centralized configuration management with environment overrides."""
    
    def __init__(self, base_config_path: Optional[Path] = None):
        self._config: Dict[str, Any] = {}
        self._load_base_configuration(base_config_path)
        self._apply_environment_overrides()

    def _load_base_configuration(self, config_path: Optional[Path]) -> None:
        """Load base configuration from file."""
        if config_path and config_path.exists():
            with open(config_path) as f:
                self._config = json.load(f)

    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides."""
        # Environment variables with AIVILLAGE_ prefix override config
        for key, value in os.environ.items():
            if key.startswith("AIVILLAGE_"):
                config_key = key[10:].lower()  # Remove prefix and lowercase
                self._config[config_key] = self._parse_env_value(value)

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # Return as string if not valid JSON
            return value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value."""
        value = self.get(key, default)
        return int(value) if value is not None else default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration value."""
        value = self.get(key, default)
        return float(value) if value is not None else default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value."""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value) if value is not None else default

# Global configuration instance
_config_manager = ConfigurationManager()

def get_config() -> ConfigurationManager:
    """Get global configuration manager."""
    return _config_manager
```

## UnifiedTaskManagerFacade (Backwards Compatibility)

```python
class UnifiedTaskManagerFacade:
    """Facade providing backwards compatibility for UnifiedManagement."""
    
    def __init__(self, service_locator: ServiceLocator):
        self._task_service = service_locator.get_service(ITaskService)
        self._project_service = service_locator.get_service(IProjectService)
        self._agent_coordination = service_locator.get_service(IAgentCoordinationService)
        self._analytics_service = service_locator.get_service(IAnalyticsService)
        self._batch_service = service_locator.get_service(IBatchProcessingService)
        self._persistence_service = service_locator.get_service(IPersistenceService)

    # Backwards compatibility methods delegating to services
    async def create_task(self, *args, **kwargs) -> Task:
        """Delegate to TaskService."""
        return await self._task_service.create_task(*args, **kwargs)

    async def assign_task(self, task: Task) -> None:
        """Delegate to TaskService and AgentCoordination."""
        agent = task.assigned_agents[0]
        await self._task_service.assign_task(task.id, agent)
        await self._agent_coordination.notify_agent(agent, {
            "task_id": task.id,
            "description": task.description
        })

    async def create_project(self, *args, **kwargs) -> str:
        """Delegate to ProjectService."""
        return await self._project_service.create_project(*args, **kwargs)

    # ... (implement all original UnifiedManagement methods as delegation)

# For backwards compatibility, alias the old name
UnifiedManagement = UnifiedTaskManagerFacade
```

## Migration Scripts

```python
# scripts/migrate_to_services.py
import asyncio
import logging
from pathlib import Path
from typing import List

class ServiceMigrationScript:
    """Script to migrate from UnifiedManagement to service architecture."""
    
    def __init__(self, service_locator: ServiceLocator):
        self._service_locator = service_locator
        self._logger = logging.getLogger(__name__)

    async def migrate_existing_data(self, data_path: Path) -> None:
        """Migrate existing task and project data to new services."""
        self._logger.info("Starting data migration...")
        
        # Load existing data
        existing_data = await self._load_existing_data(data_path)
        
        # Migrate tasks
        await self._migrate_tasks(existing_data.get("tasks", []))
        
        # Migrate projects
        await self._migrate_projects(existing_data.get("projects", []))
        
        # Migrate agent performance data
        await self._migrate_agent_performance(existing_data.get("agent_performance", {}))
        
        self._logger.info("Data migration completed successfully")

    async def _migrate_tasks(self, tasks_data: List[dict]) -> None:
        """Migrate task data to TaskService."""
        task_service = self._service_locator.get_service(ITaskService)
        
        for task_data in tasks_data:
            try:
                # Create task using new service
                task = await task_service.create_task(
                    description=task_data["description"],
                    agent=task_data["assigned_agents"][0],
                    priority=task_data.get("priority", 1),
                    deadline=task_data.get("deadline"),
                    dependencies=task_data.get("dependencies", [])
                )
                
                # Update status if not pending
                if task_data.get("status") != "pending":
                    status = TaskStatus(task_data["status"])
                    await task_service.update_task_status(task.id, status)
                
                self._logger.debug(f"Migrated task: {task.id}")
                
            except Exception as e:
                self._logger.error(f"Failed to migrate task: {e}")

    # ... (implement remaining migration methods)

    async def verify_migration(self) -> bool:
        """Verify that migration completed successfully."""
        # Implement verification logic
        pass

if __name__ == "__main__":
    # Run migration
    service_locator = setup_service_locator()
    migration_script = ServiceMigrationScript(service_locator)
    asyncio.run(migration_script.migrate_existing_data(Path("data/")))
```

## Testing Strategy

### Unit Testing Template

```python
import pytest
from unittest.mock import Mock, AsyncMock
from src.services.task_service import TaskService, ITaskRepository

class TestTaskService:
    @pytest.fixture
    def mock_repository(self):
        return Mock(spec=ITaskRepository)

    @pytest.fixture
    def task_service(self, mock_repository):
        return TaskService(mock_repository)

    @pytest.mark.asyncio
    async def test_create_task_success(self, task_service, mock_repository):
        # Arrange
        mock_repository.save = AsyncMock()
        
        # Act
        task = await task_service.create_task(
            description="Test task",
            agent="test_agent",
            priority=1
        )
        
        # Assert
        assert task.description == "Test task"
        assert task.assigned_agents == ["test_agent"]
        assert task.priority == 1
        mock_repository.save.assert_called_once_with(task)

    @pytest.mark.asyncio
    async def test_complete_task_triggers_dependents(self, task_service, mock_repository):
        # Test dependency resolution after task completion
        pass

    # ... (comprehensive test suite)
```

### Integration Testing

```python
import pytest
from src.services import ServiceLocator
from src.facades import UnifiedTaskManagerFacade

class TestServiceIntegration:
    @pytest.fixture
    def service_locator(self):
        locator = ServiceLocator()
        # Register all services
        self._register_services(locator)
        return locator

    @pytest.fixture
    def facade(self, service_locator):
        return UnifiedTaskManagerFacade(service_locator)

    @pytest.mark.asyncio
    async def test_end_to_end_task_lifecycle(self, facade):
        """Test complete task lifecycle through facade."""
        # Create task
        task = await facade.create_task("Integration test task", "test_agent")
        
        # Assign task
        await facade.assign_task(task)
        
        # Complete task
        await facade.complete_task(task.id, {"success": True})
        
        # Verify task completion
        status = await facade.get_task_status(task.id)
        assert status == TaskStatus.COMPLETED

    # ... (comprehensive integration tests)
```

This implementation guide provides the complete architecture, code examples, and migration strategy for Phase 4 refactoring. Each service is designed with clear interfaces, comprehensive error handling, and full backwards compatibility through the facade pattern.