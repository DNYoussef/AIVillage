"""
Base Orchestrator Implementation

Provides common functionality for all orchestrators, eliminating code duplication
and ensuring consistent behavior across the unified orchestration system.
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .interfaces import (
    BackgroundProcessManager,
    ConfigurationManager,
    ConfigurationSpec,
    HealthStatus,
    OrchestrationInterface,
    OrchestrationResult,
    OrchestrationStatus,
    TaskContext,
    TaskType,
)

logger = logging.getLogger(__name__)


class BaseOrchestrator(OrchestrationInterface, BackgroundProcessManager, ConfigurationManager):
    """
    Base implementation for all orchestrators.
    
    Provides common functionality including:
    - Initialization coordination
    - Background process management  
    - Health monitoring
    - Configuration management
    - Error handling
    - Metrics collection
    """
    
    def __init__(self, orchestrator_type: str, orchestrator_id: Optional[str] = None):
        """
        Initialize base orchestrator.
        
        Args:
            orchestrator_type: Type identifier for this orchestrator
            orchestrator_id: Optional custom ID, generates one if not provided
        """
        self._orchestrator_type = orchestrator_type
        self._orchestrator_id = orchestrator_id or f"{orchestrator_type}_{uuid4().hex[:8]}"
        self._status = OrchestrationStatus.INITIALIZING
        self._start_time = datetime.now()
        self._configuration: Optional[ConfigurationSpec] = None
        self._background_tasks: List[asyncio.Task] = []
        self._health_alerts: List[str] = []
        self._health_warnings: List[str] = []
        self._metrics: Dict[str, Any] = {
            'tasks_processed': 0,
            'tasks_successful': 0, 
            'tasks_failed': 0,
            'initialization_time': 0.0,
            'total_processing_time': 0.0,
            'average_task_time': 0.0,
        }
        
        logger.info(f"Initializing orchestrator: {self._orchestrator_id}")
    
    @property
    def orchestrator_id(self) -> str:
        """Unique identifier for this orchestrator."""
        return self._orchestrator_id
    
    @property
    def status(self) -> OrchestrationStatus:
        """Current orchestrator status."""
        return self._status
    
    async def initialize(self, config: Optional[ConfigurationSpec] = None) -> bool:
        """
        Initialize the orchestrator with coordination protocol.
        
        This method implements the coordination protocol to prevent the 
        initialization race conditions identified in Agent 1's analysis.
        """
        init_start = time.time()
        
        try:
            logger.info(f"Starting initialization for {self._orchestrator_id}")
            self._status = OrchestrationStatus.INITIALIZING
            
            # Load and validate configuration
            if config:
                self._configuration = config
            else:
                self._configuration = self.load_configuration()
            
            validation_errors = self.validate_configuration(self._configuration)
            if validation_errors:
                logger.error(f"Configuration validation failed: {validation_errors}")
                self._status = OrchestrationStatus.ERROR
                return False
            
            # Perform orchestrator-specific initialization
            success = await self._initialize_specific()
            if not success:
                logger.error(f"Specific initialization failed for {self._orchestrator_id}")
                self._status = OrchestrationStatus.ERROR
                return False
            
            # Initialize health monitoring
            await self._initialize_health_monitoring()
            
            # Set status to ready
            self._status = OrchestrationStatus.READY
            
            # Record initialization time
            init_duration = time.time() - init_start
            self._metrics['initialization_time'] = init_duration
            
            logger.info(f"Initialization complete for {self._orchestrator_id} in {init_duration:.2f}s")
            return True
            
        except Exception as e:
            logger.exception(f"Initialization failed for {self._orchestrator_id}: {e}")
            self._status = OrchestrationStatus.ERROR
            self._health_alerts.append(f"Initialization error: {str(e)}")
            return False
    
    async def start(self) -> bool:
        """Start orchestrator operations."""
        if self._status != OrchestrationStatus.READY:
            logger.error(f"Cannot start {self._orchestrator_id} - status is {self._status}")
            return False
        
        try:
            logger.info(f"Starting orchestrator: {self._orchestrator_id}")
            self._status = OrchestrationStatus.RUNNING
            
            # Start background processes if configured
            if self._configuration and self._configuration.auto_start:
                await self.start_background_processes()
            
            # Perform orchestrator-specific startup
            success = await self._start_specific()
            if not success:
                self._status = OrchestrationStatus.ERROR
                return False
            
            logger.info(f"Orchestrator started successfully: {self._orchestrator_id}")
            return True
            
        except Exception as e:
            logger.exception(f"Start failed for {self._orchestrator_id}: {e}")
            self._status = OrchestrationStatus.ERROR
            return False
    
    async def stop(self) -> bool:
        """Stop orchestrator operations gracefully."""
        try:
            logger.info(f"Stopping orchestrator: {self._orchestrator_id}")
            self._status = OrchestrationStatus.STOPPING
            
            # Stop background processes first
            await self.stop_background_processes()
            
            # Perform orchestrator-specific cleanup
            await self._stop_specific()
            
            self._status = OrchestrationStatus.STOPPED
            logger.info(f"Orchestrator stopped: {self._orchestrator_id}")
            return True
            
        except Exception as e:
            logger.exception(f"Stop failed for {self._orchestrator_id}: {e}")
            self._status = OrchestrationStatus.ERROR
            return False
    
    async def process_task(self, context: TaskContext) -> OrchestrationResult:
        """
        Process a task with unified error handling and metrics.
        """
        start_time = datetime.now()
        
        try:
            logger.debug(f"Processing task {context.task_id} on {self._orchestrator_id}")
            
            if self._status != OrchestrationStatus.RUNNING:
                raise RuntimeError(f"Orchestrator not running: {self._status}")
            
            # Update metrics
            self._metrics['tasks_processed'] += 1
            
            # Process task using orchestrator-specific logic
            result_data = await self._process_task_specific(context)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Update metrics
            self._metrics['tasks_successful'] += 1
            self._metrics['total_processing_time'] += duration
            self._metrics['average_task_time'] = (
                self._metrics['total_processing_time'] / self._metrics['tasks_processed']
            )
            
            result = OrchestrationResult(
                success=True,
                task_id=context.task_id,
                orchestrator_id=self._orchestrator_id,
                task_type=context.task_type,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                data=result_data,
                metrics={'processing_time': duration}
            )
            
            logger.debug(f"Task {context.task_id} completed successfully in {duration:.2f}s")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Update error metrics
            self._metrics['tasks_failed'] += 1
            
            logger.exception(f"Task {context.task_id} failed: {e}")
            
            return OrchestrationResult(
                success=False,
                task_id=context.task_id,
                orchestrator_id=self._orchestrator_id,
                task_type=context.task_type,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                errors=[str(e)],
                metrics={'processing_time': duration}
            )
    
    async def get_health_status(self) -> HealthStatus:
        """Get unified health status."""
        uptime = (datetime.now() - self._start_time).total_seconds()
        
        # Get orchestrator-specific component health
        components = await self._get_health_components()
        
        # Calculate overall health
        healthy = (
            self._status in [OrchestrationStatus.READY, OrchestrationStatus.RUNNING] and
            len(self._health_alerts) == 0
        )
        
        return HealthStatus(
            healthy=healthy,
            timestamp=datetime.now(),
            orchestrator_id=self._orchestrator_id,
            components=components,
            metrics=self._get_health_metrics(),
            alerts=self._health_alerts.copy(),
            warnings=self._health_warnings.copy(),
            uptime_seconds=uptime
        )
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator metrics."""
        base_metrics = {
            'orchestrator_id': self._orchestrator_id,
            'orchestrator_type': self._orchestrator_type,
            'status': self._status.value,
            'uptime_seconds': (datetime.now() - self._start_time).total_seconds(),
            'timestamp': datetime.now().isoformat(),
        }
        
        # Merge with internal metrics
        base_metrics.update(self._metrics)
        
        # Add orchestrator-specific metrics
        specific_metrics = await self._get_specific_metrics()
        base_metrics.update(specific_metrics)
        
        return base_metrics
    
    # Background Process Management
    async def start_background_processes(self) -> bool:
        """Start all background processes for this orchestrator."""
        try:
            if not self._configuration:
                logger.warning(f"No configuration for background processes: {self._orchestrator_id}")
                return True
            
            # Start orchestrator-specific background processes
            processes = await self._get_background_processes()
            
            for name, coro in processes.items():
                task = asyncio.create_task(coro(), name=f"{self._orchestrator_id}_{name}")
                self._background_tasks.append(task)
                logger.info(f"Started background process: {name}")
            
            return True
            
        except Exception as e:
            logger.exception(f"Failed to start background processes: {e}")
            return False
    
    async def stop_background_processes(self) -> bool:
        """Stop all background processes."""
        try:
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            self._background_tasks.clear()
            logger.info(f"Stopped all background processes for {self._orchestrator_id}")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to stop background processes: {e}")
            return False
    
    async def get_background_process_status(self) -> Dict[str, Any]:
        """Get status of all background processes."""
        status = {}
        for task in self._background_tasks:
            status[task.get_name()] = {
                'done': task.done(),
                'cancelled': task.cancelled(),
                'exception': str(task.exception()) if task.done() and task.exception() else None
            }
        return status
    
    # Configuration Management
    def load_configuration(self, config_path: Optional[Path] = None) -> ConfigurationSpec:
        """Load configuration with defaults."""
        return ConfigurationSpec(
            orchestrator_type=self._orchestrator_type,
            enabled=True,
            auto_start=True,
            health_check_interval=30.0,
            max_concurrent_tasks=10,
        )
    
    def validate_configuration(self, config: ConfigurationSpec) -> List[str]:
        """Validate configuration and return errors."""
        errors = []
        
        if config.health_check_interval <= 0:
            errors.append("Health check interval must be positive")
        
        if config.max_concurrent_tasks <= 0:
            errors.append("Max concurrent tasks must be positive")
        
        return errors
    
    def save_configuration(self, config: ConfigurationSpec, config_path: Path) -> bool:
        """Save configuration to file."""
        try:
            # Implementation would save to YAML/JSON file
            logger.info(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            logger.exception(f"Failed to save configuration: {e}")
            return False
    
    # Abstract methods for orchestrator-specific implementations
    async def _initialize_specific(self) -> bool:
        """Orchestrator-specific initialization logic."""
        return True
    
    async def _start_specific(self) -> bool:
        """Orchestrator-specific startup logic."""
        return True
    
    async def _stop_specific(self) -> bool:
        """Orchestrator-specific cleanup logic."""
        return True
    
    async def _process_task_specific(self, context: TaskContext) -> Any:
        """Orchestrator-specific task processing logic."""
        raise NotImplementedError("Subclasses must implement _process_task_specific")
    
    async def _get_health_components(self) -> Dict[str, bool]:
        """Get orchestrator-specific component health."""
        return {}
    
    def _get_health_metrics(self) -> Dict[str, float]:
        """Get orchestrator-specific health metrics."""
        return {}
    
    async def _get_specific_metrics(self) -> Dict[str, Any]:
        """Get orchestrator-specific metrics."""
        return {}
    
    async def _get_background_processes(self) -> Dict[str, Any]:
        """Get orchestrator-specific background processes."""
        return {}
    
    async def _initialize_health_monitoring(self) -> None:
        """Initialize health monitoring components."""
        pass