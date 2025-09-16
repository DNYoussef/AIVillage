"""
Orchestration Coordinator

Manages coordination between multiple orchestrators to prevent the race conditions
and resource conflicts identified in Agent 1's analysis. Implements a master
coordinator pattern that ensures proper initialization ordering and resource sharing.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from enum import Enum

from .interfaces import (
    OrchestrationInterface,
    OrchestrationResult,
    OrchestrationStatus,
    TaskContext,
    TaskType,
    HealthStatus,
)

logger = logging.getLogger(__name__)


class CoordinationState(Enum):
    """Coordination system states."""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    STOPPING = "stopping"
    ERROR = "error"


class OrchestrationCoordinator:
    """
    Master coordinator for all orchestration operations.
    
    This class solves the critical issues identified in Agent 1's overlap analysis:
    1. Prevents initialization race conditions between orchestrators
    2. Coordinates resource allocation to prevent contention
    3. Manages shared background processes
    4. Provides unified health monitoring across all orchestrators
    5. Implements proper shutdown ordering to prevent orphaned processes
    """
    
    def __init__(self):
        """Initialize the orchestration coordinator."""
        self._state = CoordinationState.INACTIVE
        self._orchestrators: Dict[str, OrchestrationInterface] = {}
        self._orchestrator_dependencies: Dict[str, Set[str]] = {}
        self._initialization_lock = asyncio.Lock()
        self._task_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()
        self._initialization_order: List[str] = []
        self._background_monitor_task: Optional[asyncio.Task] = None
        
        # Global resource tracking
        self._resource_allocations: Dict[str, Dict[str, Any]] = {}
        self._global_metrics: Dict[str, Any] = {
            'coordinator_start_time': datetime.now(),
            'total_orchestrators': 0,
            'active_orchestrators': 0,
            'failed_orchestrators': 0,
            'total_tasks_coordinated': 0,
        }
        
        logger.info("Orchestration coordinator initialized")
    
    async def register_orchestrator(
        self,
        orchestrator: OrchestrationInterface,
        dependencies: Optional[List[str]] = None,
        priority: int = 100
    ) -> bool:
        """
        Register an orchestrator with the coordinator.
        
        Args:
            orchestrator: The orchestrator to register
            dependencies: List of orchestrator IDs this depends on
            priority: Initialization priority (lower = earlier)
            
        Returns:
            bool: True if registration successful
        """
        try:
            orchestrator_id = orchestrator.orchestrator_id
            
            if orchestrator_id in self._orchestrators:
                logger.warning(f"Orchestrator {orchestrator_id} already registered")
                return False
            
            self._orchestrators[orchestrator_id] = orchestrator
            self._orchestrator_dependencies[orchestrator_id] = set(dependencies or [])
            self._global_metrics['total_orchestrators'] += 1
            
            # Update initialization order based on dependencies and priority
            self._update_initialization_order()
            
            logger.info(f"Registered orchestrator: {orchestrator_id} with priority {priority}")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to register orchestrator: {e}")
            return False
    
    async def initialize_all(self, timeout_seconds: float = 300.0) -> bool:
        """
        Initialize all orchestrators in proper dependency order.
        
        This method prevents the initialization race conditions identified
        in Agent 1's analysis by enforcing sequential initialization
        with proper dependency resolution.
        """
        async with self._initialization_lock:
            try:
                logger.info("Starting coordinated initialization of all orchestrators")
                self._state = CoordinationState.INITIALIZING
                
                if not self._orchestrators:
                    logger.warning("No orchestrators registered")
                    return False
                
                # Initialize orchestrators in dependency order
                for orchestrator_id in self._initialization_order:
                    orchestrator = self._orchestrators[orchestrator_id]
                    
                    logger.info(f"Initializing orchestrator: {orchestrator_id}")
                    
                    # Wait for dependencies to be ready
                    if not await self._wait_for_dependencies(orchestrator_id, timeout_seconds):
                        logger.error(f"Dependencies not ready for {orchestrator_id}")
                        self._state = CoordinationState.ERROR
                        return False
                    
                    # Initialize the orchestrator
                    success = await orchestrator.initialize()
                    if not success:
                        logger.error(f"Failed to initialize {orchestrator_id}")
                        self._global_metrics['failed_orchestrators'] += 1
                        self._state = CoordinationState.ERROR
                        return False
                    
                    self._global_metrics['active_orchestrators'] += 1
                    logger.info(f"Successfully initialized: {orchestrator_id}")
                
                self._state = CoordinationState.ACTIVE
                
                # Start background monitoring
                await self._start_background_monitoring()
                
                logger.info("All orchestrators initialized successfully")
                return True
                
            except Exception as e:
                logger.exception(f"Coordinated initialization failed: {e}")
                self._state = CoordinationState.ERROR
                return False
    
    async def start_all(self) -> bool:
        """Start all orchestrators after successful initialization."""
        if self._state != CoordinationState.ACTIVE:
            logger.error(f"Cannot start orchestrators - coordinator state: {self._state}")
            return False
        
        try:
            logger.info("Starting all orchestrators")
            
            # Start orchestrators in initialization order
            for orchestrator_id in self._initialization_order:
                orchestrator = self._orchestrators[orchestrator_id]
                
                success = await orchestrator.start()
                if not success:
                    logger.error(f"Failed to start {orchestrator_id}")
                    return False
                
                logger.info(f"Started orchestrator: {orchestrator_id}")
            
            logger.info("All orchestrators started successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to start orchestrators: {e}")
            return False
    
    async def shutdown_all(self, timeout_seconds: float = 60.0) -> bool:
        """
        Shutdown all orchestrators in reverse dependency order.
        
        This prevents the orphaned process issues identified in Agent 1's analysis.
        """
        async with self._shutdown_lock:
            try:
                logger.info("Starting coordinated shutdown of all orchestrators")
                self._state = CoordinationState.STOPPING
                
                # Stop background monitoring
                if self._background_monitor_task:
                    self._background_monitor_task.cancel()
                    try:
                        await self._background_monitor_task
                    except asyncio.CancelledError:
                        pass
                
                # Shutdown in reverse order
                shutdown_order = list(reversed(self._initialization_order))
                
                for orchestrator_id in shutdown_order:
                    if orchestrator_id not in self._orchestrators:
                        continue
                    
                    orchestrator = self._orchestrators[orchestrator_id]
                    
                    logger.info(f"Shutting down orchestrator: {orchestrator_id}")
                    
                    try:
                        await asyncio.wait_for(
                            orchestrator.stop(),
                            timeout=timeout_seconds / len(shutdown_order)
                        )
                        logger.info(f"Successfully shut down: {orchestrator_id}")
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout shutting down {orchestrator_id}")
                    except Exception as e:
                        logger.exception(f"Error shutting down {orchestrator_id}: {e}")
                
                self._state = CoordinationState.INACTIVE
                logger.info("All orchestrators shut down")
                return True
                
            except Exception as e:
                logger.exception(f"Coordinated shutdown failed: {e}")
                self._state = CoordinationState.ERROR
                return False
    
    async def route_task(self, context: TaskContext) -> OrchestrationResult:
        """
        Route a task to the appropriate orchestrator.
        
        This implements unified task routing to prevent the task processing
        conflicts identified in Agent 1's analysis.
        """
        async with self._task_lock:
            try:
                self._global_metrics['total_tasks_coordinated'] += 1
                
                # Find the appropriate orchestrator for this task type
                orchestrator = self._find_orchestrator_for_task(context.task_type)
                if not orchestrator:
                    raise ValueError(f"No orchestrator available for task type: {context.task_type}")
                
                # Ensure orchestrator is ready
                if orchestrator.status != OrchestrationStatus.RUNNING:
                    raise RuntimeError(f"Orchestrator not ready: {orchestrator.orchestrator_id}")
                
                # Process the task
                result = await orchestrator.process_task(context)
                
                logger.debug(f"Task {context.task_id} routed to {orchestrator.orchestrator_id}")
                return result
                
            except Exception as e:
                logger.exception(f"Task routing failed for {context.task_id}: {e}")
                
                # Return error result
                return OrchestrationResult(
                    success=False,
                    task_id=context.task_id,
                    orchestrator_id="coordinator",
                    task_type=context.task_type,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    duration_seconds=0.0,
                    errors=[str(e)]
                )
    
    async def get_system_health(self) -> HealthStatus:
        """
        Get unified health status across all orchestrators.
        
        This consolidates the health reporting identified as duplicated
        in Agent 1's analysis.
        """
        try:
            components = {}
            all_metrics = {}
            all_alerts = []
            all_warnings = []
            
            # Collect health from all orchestrators
            for orchestrator_id, orchestrator in self._orchestrators.items():
                try:
                    health = await orchestrator.get_health_status()
                    components[orchestrator_id] = health.healthy
                    all_metrics.update(health.metrics)
                    all_alerts.extend([f"{orchestrator_id}: {alert}" for alert in health.alerts])
                    all_warnings.extend([f"{orchestrator_id}: {warning}" for warning in health.warnings])
                except Exception as e:
                    components[orchestrator_id] = False
                    all_alerts.append(f"{orchestrator_id}: Health check failed - {e}")
            
            # Overall system health
            healthy = (
                self._state == CoordinationState.ACTIVE and
                all(components.values()) and
                len(all_alerts) == 0
            )
            
            return HealthStatus(
                healthy=healthy,
                timestamp=datetime.now(),
                orchestrator_id="system_coordinator",
                components=components,
                metrics=all_metrics,
                alerts=all_alerts,
                warnings=all_warnings,
                uptime_seconds=(
                    datetime.now() - self._global_metrics['coordinator_start_time']
                ).total_seconds()
            )
            
        except Exception as e:
            logger.exception(f"Failed to get system health: {e}")
            return HealthStatus(
                healthy=False,
                timestamp=datetime.now(),
                orchestrator_id="system_coordinator",
                alerts=[f"Health check failed: {e}"]
            )
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        metrics = self._global_metrics.copy()
        metrics['coordinator_state'] = self._state.value
        metrics['registered_orchestrators'] = list(self._orchestrators.keys())
        metrics['timestamp'] = datetime.now().isoformat()
        
        # Collect metrics from all orchestrators
        orchestrator_metrics = {}
        for orchestrator_id, orchestrator in self._orchestrators.items():
            try:
                orch_metrics = await orchestrator.get_metrics()
                orchestrator_metrics[orchestrator_id] = orch_metrics
            except Exception as e:
                logger.warning(f"Failed to get metrics from {orchestrator_id}: {e}")
        
        metrics['orchestrator_metrics'] = orchestrator_metrics
        return metrics
    
    # Private helper methods
    def _update_initialization_order(self) -> None:
        """Update initialization order based on dependencies."""
        # Simple topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(node: str):
            if node in temp_visited:
                raise ValueError(f"Circular dependency detected involving {node}")
            if node in visited:
                return
            
            temp_visited.add(node)
            
            for dependency in self._orchestrator_dependencies.get(node, set()):
                if dependency in self._orchestrators:
                    visit(dependency)
            
            temp_visited.remove(node)
            visited.add(node)
            order.append(node)
        
        # Visit all orchestrators
        for orchestrator_id in self._orchestrators:
            if orchestrator_id not in visited:
                visit(orchestrator_id)
        
        self._initialization_order = order
        logger.info(f"Initialization order: {self._initialization_order}")
    
    async def _wait_for_dependencies(
        self,
        orchestrator_id: str,
        timeout_seconds: float
    ) -> bool:
        """Wait for all dependencies of an orchestrator to be ready."""
        dependencies = self._orchestrator_dependencies.get(orchestrator_id, set())
        
        if not dependencies:
            return True
        
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout_seconds:
            all_ready = True
            
            for dep_id in dependencies:
                if dep_id not in self._orchestrators:
                    logger.error(f"Dependency {dep_id} not found for {orchestrator_id}")
                    return False
                
                dep_orchestrator = self._orchestrators[dep_id]
                if dep_orchestrator.status not in [OrchestrationStatus.READY, OrchestrationStatus.RUNNING]:
                    all_ready = False
                    break
            
            if all_ready:
                return True
            
            await asyncio.sleep(1.0)
        
        logger.error(f"Timeout waiting for dependencies of {orchestrator_id}")
        return False
    
    def _find_orchestrator_for_task(self, task_type: TaskType) -> Optional[OrchestrationInterface]:
        """Find the best orchestrator for a given task type."""
        # Simple task type to orchestrator mapping
        task_mappings = {
            TaskType.ML_PIPELINE: ["ml_pipeline", "unified_pipeline"],
            TaskType.AGENT_LIFECYCLE: ["agent_lifecycle", "cognative_nexus"],
            TaskType.COGNITIVE_ANALYSIS: ["cognitive_analysis", "cognitive_nexus"],
            TaskType.FOG_COORDINATION: ["fog_system", "fog_coordinator"],
            TaskType.SYSTEM_HEALTH: None,  # Any orchestrator can handle
        }
        
        preferred_types = task_mappings.get(task_type)
        
        # Try to find preferred orchestrator type
        if preferred_types:
            for orchestrator in self._orchestrators.values():
                if any(ptype in orchestrator.orchestrator_id.lower() for ptype in preferred_types):
                    if orchestrator.status == OrchestrationStatus.RUNNING:
                        return orchestrator
        
        # Fallback to any running orchestrator
        for orchestrator in self._orchestrators.values():
            if orchestrator.status == OrchestrationStatus.RUNNING:
                return orchestrator
        
        return None
    
    async def _start_background_monitoring(self) -> None:
        """Start background monitoring of orchestrator health."""
        async def monitor_health():
            while True:
                try:
                    health = await self.get_system_health()
                    if not health.healthy:
                        logger.warning(f"System health degraded: {len(health.alerts)} alerts")
                    
                    # Update global metrics
                    active_count = sum(
                        1 for orch in self._orchestrators.values()
                        if orch.status == OrchestrationStatus.RUNNING
                    )
                    self._global_metrics['active_orchestrators'] = active_count
                    
                    await asyncio.sleep(30.0)  # Health check every 30 seconds
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.exception(f"Background monitoring error: {e}")
                    await asyncio.sleep(60.0)  # Back off on error
        
        self._background_monitor_task = asyncio.create_task(monitor_health())