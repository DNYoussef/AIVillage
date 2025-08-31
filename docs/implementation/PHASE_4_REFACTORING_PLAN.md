# Phase 4 Refactoring Implementation Plan

## Overview

This document provides the detailed implementation strategy for Phase 4 architectural refactoring, focusing on god class decomposition, dependency reduction, and service boundary establishment.

## Implementation Strategy

### Core Refactoring Principles

1. **Incremental Migration**: No big-bang rewrites
2. **Interface-First Design**: Define contracts before implementation  
3. **Test-Driven Refactoring**: Tests before code changes
4. **Backwards Compatibility**: Maintain existing API surface
5. **Performance Preservation**: No performance degradation

## Phase 4.1: UnifiedManagement Decomposition

### Step 1: Interface Definition

**Create service contracts first**:

```python
# experiments/agents/agents/task_management/interfaces/__init__.py

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict
from .task import Task, TaskId, TaskStatus
from .project import Project, ProjectId

class ITaskRepository(ABC):
    """Task data access interface"""
    
    @abstractmethod
    async def save(self, task: Task) -> None:
        """Save a task to persistent storage"""
    
    @abstractmethod
    async def find_by_id(self, task_id: TaskId) -> Optional[Task]:
        """Find task by ID"""
    
    @abstractmethod
    async def find_by_status(self, status: TaskStatus) -> List[Task]:
        """Find tasks by status"""
    
    @abstractmethod
    async def find_pending(self) -> List[Task]:
        """Find all pending tasks"""
        
    @abstractmethod
    async def update_status(self, task_id: TaskId, status: TaskStatus) -> None:
        """Update task status"""

class IAgentSelectionService(ABC):
    """Agent selection and assignment interface"""
    
    @abstractmethod
    async def select_best_agent(self, task: Task) -> str:
        """Select the best agent for a task"""
    
    @abstractmethod
    async def update_agent_performance(self, agent_id: str, performance: float) -> None:
        """Update agent performance metrics"""
    
    @abstractmethod
    async def get_available_agents(self) -> List[str]:
        """Get list of available agents"""

class ITaskOrchestrationService(ABC):
    """Task orchestration business logic interface"""
    
    @abstractmethod
    async def create_task(self, description: str, **kwargs) -> Task:
        """Create a new task"""
    
    @abstractmethod
    async def assign_task(self, task_id: TaskId) -> None:
        """Assign a task to an agent"""
    
    @abstractmethod
    async def complete_task(self, task_id: TaskId, result: Any) -> None:
        """Mark task as completed with result"""
```

### Step 2: Constants Extraction

**Create centralized constants**:

```python
# experiments/agents/agents/task_management/constants.py

from dataclasses import dataclass
from typing import Final

@dataclass(frozen=True)
class TaskConstants:
    """Task management constants"""
    
    # Batch processing
    DEFAULT_BATCH_SIZE: Final[int] = 5
    MAX_BATCH_SIZE: Final[int] = 20
    BATCH_PROCESSING_DELAY: Final[float] = 1.0
    
    # Task timeouts
    TASK_TIMEOUT_SECONDS: Final[int] = 300
    TASK_RETRY_DELAY: Final[float] = 2.0
    MAX_RETRY_ATTEMPTS: Final[int] = 3
    
    # Priority thresholds
    PRIORITY_LOW_THRESHOLD: Final[int] = 3
    PRIORITY_MEDIUM_THRESHOLD: Final[int] = 6
    PRIORITY_HIGH_THRESHOLD: Final[int] = 8
    PRIORITY_CRITICAL_THRESHOLD: Final[int] = 10

@dataclass(frozen=True)
class IncentiveConstants:
    """Incentive calculation constants"""
    
    # Performance multipliers
    BASE_PERFORMANCE_MULTIPLIER: Final[float] = 1.1
    MIN_PERFORMANCE_THRESHOLD: Final[float] = 0.5
    MAX_PERFORMANCE_THRESHOLD: Final[float] = 2.0
    PERFORMANCE_DECAY_FACTOR: Final[float] = 0.9
    
    # Collaboration factors
    COLLABORATION_BONUS_FACTOR: Final[float] = 0.3
    INNOVATION_BONUS_FACTOR: Final[float] = 0.1
    SPECIALIZATION_FACTOR: Final[float] = 0.1
    
    # Difficulty adjustments
    MIN_DIFFICULTY_FACTOR: Final[float] = 0.5
    MAX_DIFFICULTY_FACTOR: Final[float] = 2.0

@dataclass(frozen=True)
class ProcessingConstants:
    """Agent processing constants"""
    
    # Query processing
    CONFIDENCE_THRESHOLD: Final[float] = 0.8
    MAX_PROCESSING_TIME: Final[float] = 30.0
    QUERY_TIMEOUT: Final[float] = 60.0
    
    # Error handling
    MAX_ERROR_RETRIES: Final[int] = 3
    ERROR_BACKOFF_MULTIPLIER: Final[float] = 1.5
    ERROR_RECOVERY_DELAY: Final[float] = 1.0

# Singleton instances
TASK_CONSTANTS = TaskConstants()
INCENTIVE_CONSTANTS = IncentiveConstants()
PROCESSING_CONSTANTS = ProcessingConstants()
```

### Step 3: Repository Implementation

**Extract data access layer**:

```python
# experiments/agents/agents/task_management/repositories/task_repository.py

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from ..interfaces import ITaskRepository
from ..task import Task, TaskId, TaskStatus
from ..constants import TASK_CONSTANTS

logger = logging.getLogger(__name__)

class TaskRepository(ITaskRepository):
    """File-based task repository implementation"""
    
    def __init__(self, storage_path: str = "tasks.json"):
        self.storage_path = Path(storage_path)
        self._tasks: Dict[str, Task] = {}
        self._load_tasks()
    
    async def save(self, task: Task) -> None:
        """Save a task to storage"""
        try:
            self._tasks[task.id] = task
            await self._persist_tasks()
            logger.info(f"Saved task: {task.id}")
        except Exception as e:
            logger.exception(f"Error saving task {task.id}: {e}")
            raise
    
    async def find_by_id(self, task_id: TaskId) -> Optional[Task]:
        """Find task by ID"""
        return self._tasks.get(task_id)
    
    async def find_by_status(self, status: TaskStatus) -> List[Task]:
        """Find tasks by status"""
        return [task for task in self._tasks.values() if task.status == status]
    
    async def find_pending(self) -> List[Task]:
        """Find all pending tasks"""
        return await self.find_by_status(TaskStatus.PENDING)
    
    async def update_status(self, task_id: TaskId, status: TaskStatus) -> None:
        """Update task status"""
        if task_id in self._tasks:
            self._tasks[task_id] = self._tasks[task_id].update_status(status)
            await self._persist_tasks()
            logger.info(f"Updated task {task_id} status to {status}")
    
    def _load_tasks(self) -> None:
        """Load tasks from storage"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self._tasks = {
                        task_id: Task(**task_data) 
                        for task_id, task_data in data.items()
                    }
                logger.info(f"Loaded {len(self._tasks)} tasks from storage")
        except Exception as e:
            logger.exception(f"Error loading tasks: {e}")
            self._tasks = {}
    
    async def _persist_tasks(self) -> None:
        """Persist tasks to storage"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(
                    {task_id: task.__dict__ for task_id, task in self._tasks.items()},
                    f,
                    indent=2,
                    default=str
                )
        except Exception as e:
            logger.exception(f"Error persisting tasks: {e}")
            raise
```

### Step 4: Service Implementation

**Extract agent selection service**:

```python
# experiments/agents/agents/task_management/services/agent_selection_service.py

import logging
from typing import Dict, List
from ..interfaces import IAgentSelectionService, ITaskRepository
from ..task import Task
from ..constants import INCENTIVE_CONSTANTS

logger = logging.getLogger(__name__)

class AgentSelectionService(IAgentSelectionService):
    """Agent selection and performance tracking service"""
    
    def __init__(self, task_repository: ITaskRepository):
        self.task_repository = task_repository
        self.agent_performance: Dict[str, float] = {}
        self.available_agents: List[str] = []
    
    async def select_best_agent(self, task: Task) -> str:
        """Select the best agent for a task based on performance and availability"""
        try:
            if not self.available_agents:
                logger.warning("No available agents, returning default")
                return "default_agent"
            
            # Simple selection based on performance
            best_agent = max(
                self.available_agents,
                key=lambda agent: self.agent_performance.get(agent, 1.0)
            )
            
            logger.info(f"Selected agent {best_agent} for task {task.id}")
            return best_agent
            
        except Exception as e:
            logger.exception(f"Error selecting agent for task {task.id}: {e}")
            return self.available_agents[0] if self.available_agents else "default_agent"
    
    async def update_agent_performance(self, agent_id: str, performance: float) -> None:
        """Update agent performance metrics"""
        try:
            # Clamp performance within bounds
            performance = max(
                INCENTIVE_CONSTANTS.MIN_PERFORMANCE_THRESHOLD,
                min(performance, INCENTIVE_CONSTANTS.MAX_PERFORMANCE_THRESHOLD)
            )
            
            self.agent_performance[agent_id] = performance
            logger.info(f"Updated performance for agent {agent_id}: {performance}")
            
        except Exception as e:
            logger.exception(f"Error updating agent performance: {e}")
            raise
    
    async def get_available_agents(self) -> List[str]:
        """Get list of available agents"""
        return self.available_agents.copy()
    
    def set_available_agents(self, agents: List[str]) -> None:
        """Set the list of available agents"""
        self.available_agents = agents
        logger.info(f"Updated available agents: {agents}")
```

### Step 5: Orchestration Service

**Extract task orchestration logic**:

```python
# experiments/agents/agents/task_management/services/task_orchestration_service.py

import logging
from typing import Any, Optional
from ..interfaces import (
    ITaskOrchestrationService, 
    ITaskRepository, 
    IAgentSelectionService
)
from ..task import Task, TaskId, TaskStatus
from ..constants import TASK_CONSTANTS
from core.error_handling import (
    StandardCommunicationProtocol, 
    Message, 
    MessageType, 
    Priority
)

logger = logging.getLogger(__name__)

class TaskOrchestrationService(ITaskOrchestrationService):
    """Task orchestration business logic"""
    
    def __init__(
        self, 
        task_repository: ITaskRepository,
        agent_service: IAgentSelectionService,
        communication_protocol: StandardCommunicationProtocol
    ):
        self.task_repository = task_repository
        self.agent_service = agent_service  
        self.communication_protocol = communication_protocol
    
    async def create_task(
        self,
        description: str,
        priority: int = TASK_CONSTANTS.PRIORITY_MEDIUM_THRESHOLD,
        deadline: Optional[str] = None,
        **kwargs
    ) -> Task:
        """Create a new task with validation"""
        try:
            # Select best agent for task
            # Create preliminary task for selection
            temp_task = Task(
                description=description,
                assigned_agents=[],
                priority=priority,
                deadline=deadline
            )
            
            agent = await self.agent_service.select_best_agent(temp_task)
            
            # Create final task
            task = Task(
                description=description,
                assigned_agents=[agent],
                priority=priority,
                deadline=deadline
            )
            
            await self.task_repository.save(task)
            logger.info(f"Created task {task.id} assigned to {agent}")
            
            return task
            
        except Exception as e:
            logger.exception(f"Error creating task: {e}")
            raise
    
    async def assign_task(self, task_id: TaskId) -> None:
        """Assign a task to an agent"""
        try:
            task = await self.task_repository.find_by_id(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")
            
            if task.status != TaskStatus.PENDING:
                raise ValueError(f"Task {task_id} is not pending")
            
            # Update task status to in progress
            await self.task_repository.update_status(task_id, TaskStatus.IN_PROGRESS)
            
            # Notify agent
            await self._notify_agent(task)
            
            logger.info(f"Assigned task {task_id} to {task.assigned_agents[0]}")
            
        except Exception as e:
            logger.exception(f"Error assigning task {task_id}: {e}")
            raise
    
    async def complete_task(self, task_id: TaskId, result: Any) -> None:
        """Complete a task with result"""
        try:
            task = await self.task_repository.find_by_id(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")
            
            # Update task with result and completion status
            completed_task = task.update_result(result).update_status(TaskStatus.COMPLETED)
            await self.task_repository.save(completed_task)
            
            # Update agent performance based on result
            if completed_task.assigned_agents:
                agent = completed_task.assigned_agents[0]
                performance = self._calculate_performance_score(result)
                await self.agent_service.update_agent_performance(agent, performance)
            
            logger.info(f"Completed task {task_id}")
            
        except Exception as e:
            logger.exception(f"Error completing task {task_id}: {e}")
            raise
    
    async def _notify_agent(self, task: Task) -> None:
        """Notify agent of task assignment"""
        try:
            if not task.assigned_agents:
                return
                
            agent = task.assigned_agents[0]
            message = Message(
                type=MessageType.TASK,
                sender="TaskOrchestrationService",
                receiver=agent,
                content={
                    "task_id": task.id,
                    "description": task.description,
                    "priority": task.priority,
                    "deadline": task.deadline
                },
                priority=Priority.MEDIUM
            )
            
            await self.communication_protocol.send_message(message)
            
        except Exception as e:
            logger.exception(f"Error notifying agent: {e}")
            # Don't raise - notification failure shouldn't break assignment
    
    def _calculate_performance_score(self, result: Any) -> float:
        """Calculate performance score from task result"""
        try:
            if isinstance(result, dict):
                success = result.get("success", False)
                quality = result.get("quality", 0.5)
                if success:
                    return min(1.0 + quality * 0.1, INCENTIVE_CONSTANTS.MAX_PERFORMANCE_THRESHOLD)
                else:
                    return max(0.5 - quality * 0.1, INCENTIVE_CONSTANTS.MIN_PERFORMANCE_THRESHOLD)
            
            # Simple boolean success
            return 1.1 if result else 0.9
            
        except Exception:
            return 1.0  # Default performance
```

### Step 6: Facade Implementation

**Create backwards-compatible facade**:

```python
# experiments/agents/agents/task_management/unified_task_manager_v2.py

import logging
from typing import Any, Dict, List, Optional
from .services.task_orchestration_service import TaskOrchestrationService
from .services.agent_selection_service import AgentSelectionService
from .repositories.task_repository import TaskRepository
from .interfaces import ITaskRepository, IAgentSelectionService, ITaskOrchestrationService
from .task import Task, TaskId, TaskStatus
from .constants import TASK_CONSTANTS
from core.error_handling import StandardCommunicationProtocol

logger = logging.getLogger(__name__)

class UnifiedTaskManagerV2:
    """Refactored task manager with service-oriented architecture"""
    
    def __init__(
        self,
        communication_protocol: StandardCommunicationProtocol,
        storage_path: str = "tasks.json"
    ):
        # Initialize services
        self.task_repository: ITaskRepository = TaskRepository(storage_path)
        self.agent_service: IAgentSelectionService = AgentSelectionService(self.task_repository)
        self.orchestration_service: ITaskOrchestrationService = TaskOrchestrationService(
            self.task_repository,
            self.agent_service,
            communication_protocol
        )
        
        self.communication_protocol = communication_protocol
        
        # Backwards compatibility attributes
        self.pending_tasks = []  # Will be populated from repository
        self.ongoing_tasks = {}  # Will be populated from repository  
        self.completed_tasks = []  # Will be populated from repository
        
    async def create_task(
        self,
        description: str,
        agent: str,
        priority: int = TASK_CONSTANTS.PRIORITY_MEDIUM_THRESHOLD,
        deadline: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Task:
        """Create a task - backwards compatible interface"""
        try:
            # Set available agents to include the requested agent
            agents = await self.agent_service.get_available_agents()
            if agent not in agents:
                self.agent_service.set_available_agents(agents + [agent])
            
            task = await self.orchestration_service.create_task(
                description=description,
                priority=priority,
                deadline=deadline
            )
            
            # Update compatibility attributes
            await self._sync_compatibility_attributes()
            
            return task
            
        except Exception as e:
            logger.exception(f"Error creating task: {e}")
            raise
    
    async def assign_task(self, task: Task) -> None:
        """Assign a task - backwards compatible interface"""
        try:
            await self.orchestration_service.assign_task(task.id)
            await self._sync_compatibility_attributes()
            
        except Exception as e:
            logger.exception(f"Error assigning task: {e}")
            raise
    
    async def complete_task(self, task_id: str, result: Any) -> None:
        """Complete a task - backwards compatible interface"""
        try:
            await self.orchestration_service.complete_task(task_id, result)
            await self._sync_compatibility_attributes()
            
        except Exception as e:
            logger.exception(f"Error completing task: {e}")
            raise
    
    def update_agent_list(self, agent_list: List[str]) -> None:
        """Update available agents - backwards compatible"""
        self.agent_service.set_available_agents(agent_list)
        logger.info(f"Updated available agents: {agent_list}")
    
    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Get task status - backwards compatible"""
        task = await self.task_repository.find_by_id(task_id)
        return task.status if task else TaskStatus.PENDING
    
    async def _sync_compatibility_attributes(self) -> None:
        """Sync backwards compatibility attributes from repository"""
        try:
            pending = await self.task_repository.find_by_status(TaskStatus.PENDING)
            in_progress = await self.task_repository.find_by_status(TaskStatus.IN_PROGRESS)
            completed = await self.task_repository.find_by_status(TaskStatus.COMPLETED)
            
            self.pending_tasks = pending
            self.ongoing_tasks = {task.id: task for task in in_progress}
            self.completed_tasks = completed
            
        except Exception as e:
            logger.exception(f"Error syncing compatibility attributes: {e}")

# Backwards compatibility alias
UnifiedManagement = UnifiedTaskManagerV2
```

## Phase 4.2: SageAgent Dependency Reduction

### Step 1: Service Locator Pattern

**Create dependency management**:

```python
# experiments/agents/agents/sage/services/service_locator.py

from typing import Dict, TypeVar, Type, Optional, Any
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

class IServiceContainer(ABC):
    """Service container interface"""
    
    @abstractmethod
    def register(self, interface: Type[T], implementation: T) -> None:
        """Register a service implementation"""
    
    @abstractmethod
    def get(self, interface: Type[T]) -> T:
        """Get a service implementation"""

class ServiceContainer(IServiceContainer):
    """Simple service container implementation"""
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._singletons: Dict[Type, Any] = {}
    
    def register(self, interface: Type[T], implementation: T) -> None:
        """Register a service implementation"""
        self._services[interface] = implementation
        logger.info(f"Registered service: {interface.__name__}")
    
    def register_singleton(self, interface: Type[T], implementation: T) -> None:
        """Register a singleton service"""
        self._services[interface] = implementation
        self._singletons[interface] = implementation
        logger.info(f"Registered singleton: {interface.__name__}")
    
    def get(self, interface: Type[T]) -> T:
        """Get a service implementation"""
        if interface in self._singletons:
            return self._singletons[interface]
        
        if interface in self._services:
            service = self._services[interface]
            if callable(service):
                return service()
            return service
        
        raise ValueError(f"Service not registered: {interface.__name__}")

class SageAgentServiceLocator:
    """Service locator for SageAgent dependencies"""
    
    def __init__(self):
        self.container = ServiceContainer()
        self._setup_default_services()
    
    def _setup_default_services(self) -> None:
        """Setup default service registrations"""
        # Will be implemented with actual services
        pass
    
    def get_rag_pipeline(self):
        """Get RAG pipeline service"""
        from rag_system.core.pipeline import EnhancedRAGPipeline
        return self.container.get(EnhancedRAGPipeline)
    
    def get_cognitive_nexus(self):
        """Get cognitive nexus service"""
        from rag_system.core.cognitive_nexus import CognitiveNexus
        return self.container.get(CognitiveNexus)
    
    def get_processing_chain(self):
        """Get processing chain service"""
        # Implementation will be added
        pass
```

### Step 2: Processing Chain Factory

**Extract processing component creation**:

```python
# experiments/agents/agents/sage/factories/processing_factory.py

from typing import Any
import logging
from ..services.service_locator import SageAgentServiceLocator

logger = logging.getLogger(__name__)

class ProcessingChainFactory:
    """Factory for creating processing chains"""
    
    def __init__(self, service_locator: SageAgentServiceLocator):
        self.service_locator = service_locator
    
    def create_query_processor(self, config: dict):
        """Create query processor with dependencies"""
        try:
            from ..query_processing import QueryProcessor
            
            rag_system = self.service_locator.get_rag_pipeline()
            cognitive_nexus = self.service_locator.get_cognitive_nexus()
            
            # Additional components will be retrieved from service locator
            return QueryProcessor(rag_system, None, cognitive_nexus)  # Simplified
            
        except Exception as e:
            logger.exception(f"Error creating query processor: {e}")
            raise
    
    def create_response_generator(self, config: dict):
        """Create response generator"""
        try:
            from ..response_generator import ResponseGenerator
            return ResponseGenerator()
            
        except Exception as e:
            logger.exception(f"Error creating response generator: {e}")
            raise
    
    def create_task_executor(self, agent_reference):
        """Create task executor"""
        try:
            from ..task_execution import TaskExecutor
            return TaskExecutor(agent_reference)
            
        except Exception as e:
            logger.exception(f"Error creating task executor: {e}")
            raise

class CognitiveLayerComposite:
    """Composite for managing cognitive layers"""
    
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer: Any) -> None:
        """Add a cognitive layer"""
        self.layers.append(layer)
        logger.info(f"Added cognitive layer: {type(layer).__name__}")
    
    async def process_through_layers(self, input_data: Any) -> Any:
        """Process input through all layers"""
        result = input_data
        
        for layer in self.layers:
            try:
                if hasattr(layer, 'process'):
                    result = await layer.process(result)
                elif hasattr(layer, 'evolve'):
                    await layer.evolve()
                    
            except Exception as e:
                logger.exception(f"Error processing through layer {type(layer).__name__}: {e}")
                # Continue processing with other layers
                
        return result
    
    async def evolve_all_layers(self) -> None:
        """Evolve all cognitive layers"""
        for layer in self.layers:
            try:
                if hasattr(layer, 'evolve'):
                    await layer.evolve()
                    
            except Exception as e:
                logger.exception(f"Error evolving layer {type(layer).__name__}: {e}")
```

### Step 3: Simplified SageAgent

**Reduce dependencies through composition**:

```python
# experiments/agents/agents/sage/sage_agent_v2.py

import logging
from typing import Any, Optional
from datetime import datetime

from agents.unified_base_agent import UnifiedBaseAgent
from core.error_handling import StandardCommunicationProtocol, Message, MessageType
from rag_system.core.config import UnifiedConfig
from rag_system.tracking.unified_knowledge_tracker import UnifiedKnowledgeTracker
from agents.utils.task import Task as LangroidTask

from .services.service_locator import SageAgentServiceLocator
from .factories.processing_factory import ProcessingChainFactory, CognitiveLayerComposite
from .constants import PROCESSING_CONSTANTS

logger = logging.getLogger(__name__)

class SageAgentV2(UnifiedBaseAgent):
    """Simplified SageAgent with dependency injection"""
    
    def __init__(
        self,
        config: UnifiedConfig,
        communication_protocol: StandardCommunicationProtocol,
        knowledge_tracker: Optional[UnifiedKnowledgeTracker] = None,
        service_locator: Optional[SageAgentServiceLocator] = None
    ):
        super().__init__(config, communication_protocol, knowledge_tracker)
        
        # Core dependencies
        self.service_locator = service_locator or SageAgentServiceLocator()
        self.factory = ProcessingChainFactory(self.service_locator)
        self.cognitive_composite = CognitiveLayerComposite()
        
        # Performance tracking
        self.performance_metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_execution_time": 0,
        }
        
        # Initialize processing components
        self._setup_processing_components()
        self._setup_cognitive_layers()
    
    def _setup_processing_components(self) -> None:
        """Setup processing components via factory"""
        try:
            config = {}  # Will be extracted from agent config
            
            self.query_processor = self.factory.create_query_processor(config)
            self.response_generator = self.factory.create_response_generator(config)
            self.task_executor = self.factory.create_task_executor(self)
            
            logger.info("Processing components initialized")
            
        except Exception as e:
            logger.exception(f"Error setting up processing components: {e}")
            raise
    
    def _setup_cognitive_layers(self) -> None:
        """Setup cognitive layers via service locator"""
        try:
            # Add layers to composite
            cognitive_nexus = self.service_locator.get_cognitive_nexus()
            self.cognitive_composite.add_layer(cognitive_nexus)
            
            # Additional layers will be added through service locator
            logger.info("Cognitive layers initialized")
            
        except Exception as e:
            logger.exception(f"Error setting up cognitive layers: {e}")
            # Continue with reduced functionality
    
    async def execute_task(self, task: LangroidTask) -> Any:
        """Execute a task with performance tracking"""
        self.performance_metrics["total_tasks"] += 1
        start_time = datetime.now()
        
        try:
            if getattr(task, "is_user_query", False):
                result = await self.process_user_query(task.content)
            else:
                result = await self.task_executor.execute_task({
                    "type": getattr(task, "type", "general"),
                    "content": task.content,
                    "priority": getattr(task, "priority", 1),
                    "id": getattr(task, "task_id", ""),
                })
            
            self.performance_metrics["successful_tasks"] += 1
            return result
            
        except Exception as e:
            self.performance_metrics["failed_tasks"] += 1
            logger.exception(f"Error executing task: {e}")
            return await self._handle_error(e, task)
        
        finally:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_average_execution_time(execution_time)
    
    async def process_user_query(self, query: str) -> dict[str, Any]:
        """Process user query through simplified pipeline"""
        try:
            # Process through cognitive layers
            processed_input = await self.cognitive_composite.process_through_layers(query)
            
            # Get RAG pipeline
            rag_system = self.service_locator.get_rag_pipeline()
            rag_result = await rag_system.process_query(processed_input)
            
            # Generate response
            response = await self.response_generator.generate_response(query, rag_result, {})
            
            return {
                "original_query": query,
                "rag_result": rag_result,
                "response": response,
                "confidence": 0.8  # Simplified confidence
            }
            
        except Exception as e:
            logger.exception(f"Error processing user query: {e}")
            return {
                "original_query": query,
                "error": str(e),
                "response": "I encountered an error processing your query.",
                "confidence": 0.0
            }
    
    async def evolve(self) -> None:
        """Evolve through cognitive layers"""
        try:
            await self.cognitive_composite.evolve_all_layers()
            logger.info("SageAgent evolved")
            
        except Exception as e:
            logger.exception(f"Error evolving agent: {e}")
    
    async def _handle_error(self, error: Exception, task: LangroidTask) -> Any:
        """Handle task execution errors"""
        return {
            "error": str(error),
            "task_id": getattr(task, "task_id", "unknown"),
            "success": False
        }
    
    def _update_average_execution_time(self, execution_time: float) -> None:
        """Update average execution time"""
        total_tasks = self.performance_metrics["total_tasks"]
        current_avg = self.performance_metrics["average_execution_time"]
        
        self.performance_metrics["average_execution_time"] = (
            (current_avg * (total_tasks - 1) + execution_time) / total_tasks
        )

# Backwards compatibility
SageAgent = SageAgentV2
```

## Phase 4.3: Migration Strategy

### Migration Steps

1. **Parallel Implementation**: Build new services alongside existing code
2. **Feature Flags**: Control which implementation to use
3. **Gradual Migration**: Migrate consumers one by one
4. **Performance Monitoring**: Ensure no performance regression
5. **Rollback Plan**: Quick rollback if issues arise

### Testing Strategy

```python
# tests/test_migration_compatibility.py

import pytest
from experiments.agents.agents.task_management.unified_task_manager import UnifiedManagement as OldManager
from experiments.agents.agents.task_management.unified_task_manager_v2 import UnifiedTaskManagerV2 as NewManager

class TestMigrationCompatibility:
    """Test backwards compatibility during migration"""
    
    async def test_create_task_compatibility(self):
        """Test that create_task works the same in both versions"""
        # Setup both versions
        old_manager = OldManager(...)
        new_manager = NewManager(...)
        
        # Test same input produces same output
        task1 = await old_manager.create_task("Test task", "agent1")
        task2 = await new_manager.create_task("Test task", "agent1")
        
        assert task1.description == task2.description
        assert task1.priority == task2.priority
    
    async def test_performance_compatibility(self):
        """Test that performance is maintained"""
        import time
        
        new_manager = NewManager(...)
        
        start_time = time.time()
        for i in range(100):
            await new_manager.create_task(f"Task {i}", "agent1")
        end_time = time.time()
        
        # Performance should be maintained or improved
        assert (end_time - start_time) < 1.0  # Adjust threshold as needed
```

## Success Criteria

### Phase 4.1 Success Criteria
- [ ] UnifiedManagement reduced from 424 to <100 lines
- [ ] Coupling score reduced from 21.6 to <8.0
- [ ] All existing tests pass
- [ ] Performance maintained within 5%
- [ ] New services have >85% test coverage

### Phase 4.2 Success Criteria  
- [ ] SageAgent constructor dependencies reduced from 23 to <7
- [ ] Coupling score reduced from 47.46 to <15.0
- [ ] Service locator pattern implemented
- [ ] Processing chains extracted
- [ ] Performance maintained within 5%

### Phase 4.3 Success Criteria
- [ ] 159 magic literals replaced with constants
- [ ] Constants organized by domain
- [ ] Type-safe constant definitions
- [ ] Configuration-driven behavior

## Implementation Timeline

**Week 1**: Interface definition, constants extraction  
**Week 2**: Repository and service implementation  
**Week 3**: Orchestration service, facade creation  
**Week 4**: SageAgent refactoring  
**Week 5**: Migration testing  
**Week 6**: Performance validation  
**Week 7**: Documentation and rollout  
**Week 8**: Monitoring and optimization

## Conclusion

This implementation plan provides a systematic approach to Phase 4 refactoring while maintaining backwards compatibility and performance. The incremental approach minimizes risk while delivering significant architectural improvements.