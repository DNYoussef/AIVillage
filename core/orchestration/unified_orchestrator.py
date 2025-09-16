"""
Unified Orchestrator

The main orchestration system that consolidates all 4 overlapping orchestration 
systems into a single, coherent interface. This is the primary entry point for 
all orchestration operations in the AIVillage system.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import BaseOrchestrator
from .coordinator import OrchestrationCoordinator
from .registry import OrchestratorRegistry
from .interfaces import (
    ConfigurationSpec,
    OrchestrationResult,
    OrchestrationStatus,
    TaskContext,
    TaskType,
    HealthStatus,
)

# Import specialized orchestrators
from .ml_orchestrator import MLPipelineOrchestrator, MLConfig
from .agent_orchestrator import AgentLifecycleOrchestrator, AgentConfig
from .cognitive_orchestrator import CognitiveAnalysisOrchestrator, CognitiveConfig
from .fog_orchestrator import FogSystemOrchestrator, FogConfig

logger = logging.getLogger(__name__)


class UnifiedOrchestrator:
    """
    Unified Orchestration System - Main Entry Point
    
    This class provides the main interface for all orchestration operations,
    consolidating the functionality from the 4 overlapping systems:
    
    1. UnifiedPipeline (ML pipeline orchestration) → MLPipelineOrchestrator
    2. CognativeNexusController (agent lifecycle) → AgentLifecycleOrchestrator  
    3. CognitiveNexus (cognitive analysis) → CognitiveAnalysisOrchestrator
    4. FogCoordinator (distributed systems) → FogSystemOrchestrator
    
    Features:
    - Unified interface eliminating method signature conflicts
    - Coordinated initialization preventing race conditions
    - Consolidated background process management
    - Standardized error handling and metrics
    - Cross-orchestrator task routing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the unified orchestration system.
        
        Args:
            config: Optional configuration dictionary with orchestrator-specific settings
        """
        self._unified_id = f"unified_orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._config = config or {}
        
        # Core coordination components
        self._coordinator = OrchestrationCoordinator()
        self._registry = OrchestratorRegistry()
        
        # Orchestrator instances
        self._orchestrators: Dict[str, BaseOrchestrator] = {}
        
        # System state
        self._is_initialized = False
        self._is_running = False
        
        logger.info(f"Unified Orchestrator initialized: {self._unified_id}")
    
    async def initialize(
        self,
        enable_ml_pipeline: bool = True,
        enable_agent_lifecycle: bool = True,
        enable_cognitive_analysis: bool = True,
        enable_fog_system: bool = True,
        custom_configs: Optional[Dict[str, ConfigurationSpec]] = None
    ) -> bool:
        """
        Initialize the unified orchestration system.
        
        This method implements the Agent 2 blueprint by:
        1. Registering all orchestrator types with the registry
        2. Creating instances with proper configurations
        3. Registering instances with the coordinator  
        4. Performing coordinated initialization to prevent race conditions
        
        Args:
            enable_ml_pipeline: Enable ML pipeline orchestration
            enable_agent_lifecycle: Enable agent lifecycle management
            enable_cognitive_analysis: Enable cognitive analysis
            enable_fog_system: Enable fog computing coordination
            custom_configs: Optional custom configurations for each orchestrator
            
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Starting unified orchestration system initialization")
            
            # Step 1: Register orchestrator types with the registry
            await self._register_orchestrator_types(
                enable_ml_pipeline, enable_agent_lifecycle, 
                enable_cognitive_analysis, enable_fog_system
            )
            
            # Step 2: Create orchestrator instances with configurations
            configs = custom_configs or {}
            orchestrator_instances = {}
            
            if enable_ml_pipeline:
                ml_config = configs.get('ml_pipeline', MLConfig())
                ml_orchestrator = MLPipelineOrchestrator()
                await ml_orchestrator.initialize(ml_config)
                orchestrator_instances['ml_pipeline'] = ml_orchestrator
            
            if enable_agent_lifecycle:
                agent_config = configs.get('agent_lifecycle', AgentConfig())
                agent_orchestrator = AgentLifecycleOrchestrator()
                await agent_orchestrator.initialize(agent_config)
                orchestrator_instances['agent_lifecycle'] = agent_orchestrator
            
            if enable_cognitive_analysis:
                cognitive_config = configs.get('cognitive_analysis', CognitiveConfig())
                cognitive_orchestrator = CognitiveAnalysisOrchestrator()
                await cognitive_orchestrator.initialize(cognitive_config)
                orchestrator_instances['cognitive_analysis'] = cognitive_orchestrator
            
            if enable_fog_system:
                fog_config = configs.get('fog_system', FogConfig())
                fog_orchestrator = FogSystemOrchestrator()
                await fog_orchestrator.initialize(fog_config)
                orchestrator_instances['fog_system'] = fog_orchestrator
            
            # Step 3: Register instances with coordinator (with dependencies)
            for orch_type, orchestrator in orchestrator_instances.items():
                dependencies = self._get_orchestrator_dependencies(orch_type)
                await self._coordinator.register_orchestrator(
                    orchestrator=orchestrator,
                    dependencies=dependencies,
                    priority=self._get_orchestrator_priority(orch_type)
                )
            
            self._orchestrators = orchestrator_instances
            
            # Step 4: Perform coordinated initialization
            success = await self._coordinator.initialize_all(timeout_seconds=300.0)
            if not success:
                logger.error("Coordinated initialization failed")
                return False
            
            self._is_initialized = True
            logger.info("Unified orchestration system initialization complete")
            return True
            
        except Exception as e:
            logger.exception(f"Unified orchestration initialization failed: {e}")
            self._is_initialized = False
            return False
    
    async def start(self) -> bool:
        """
        Start all orchestrators in the unified system.
        
        Returns:
            bool: True if startup successful
        """
        if not self._is_initialized:
            logger.error("Cannot start unified orchestrator - not initialized")
            return False
        
        try:
            logger.info("Starting unified orchestration system")
            
            # Use coordinator to start all orchestrators
            success = await self._coordinator.start_all()
            if success:
                self._is_running = True
                logger.info("Unified orchestration system started successfully")
            else:
                logger.error("Failed to start unified orchestration system")
            
            return success
            
        except Exception as e:
            logger.exception(f"Failed to start unified orchestration system: {e}")
            return False
    
    async def stop(self) -> bool:
        """
        Stop all orchestrators gracefully.
        
        Returns:
            bool: True if shutdown successful
        """
        try:
            logger.info("Stopping unified orchestration system")
            
            # Use coordinator to shutdown all orchestrators
            success = await self._coordinator.shutdown_all(timeout_seconds=60.0)
            
            self._is_running = False
            logger.info("Unified orchestration system stopped")
            return success
            
        except Exception as e:
            logger.exception(f"Failed to stop unified orchestration system: {e}")
            return False
    
    async def process_task(self, task_context: TaskContext) -> OrchestrationResult:
        """
        Process a task using the appropriate orchestrator.
        
        This method implements unified task routing, eliminating the task
        processing conflicts identified in Agent 1's analysis.
        
        Args:
            task_context: Task context with type and parameters
            
        Returns:
            OrchestrationResult: Standardized result from processing
        """
        if not self._is_running:
            return OrchestrationResult(
                success=False,
                task_id=task_context.task_id,
                orchestrator_id=self._unified_id,
                task_type=task_context.task_type,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=0.0,
                errors=["Unified orchestrator not running"]
            )
        
        try:
            # Route task through coordinator for proper orchestrator selection
            return await self._coordinator.route_task(task_context)
            
        except Exception as e:
            logger.exception(f"Task processing failed: {e}")
            return OrchestrationResult(
                success=False,
                task_id=task_context.task_id,
                orchestrator_id=self._unified_id,
                task_type=task_context.task_type,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=0.0,
                errors=[str(e)]
            )
    
    async def get_system_health(self) -> HealthStatus:
        """
        Get unified system health status.
        
        This consolidates the health reporting that was duplicated across
        all 4 original orchestration systems.
        
        Returns:
            HealthStatus: Comprehensive system health information
        """
        try:
            if not self._is_initialized:
                return HealthStatus(
                    healthy=False,
                    timestamp=datetime.now(),
                    orchestrator_id=self._unified_id,
                    alerts=["System not initialized"]
                )
            
            # Get health from coordinator (which aggregates from all orchestrators)
            return await self._coordinator.get_system_health()
            
        except Exception as e:
            logger.exception(f"Failed to get system health: {e}")
            return HealthStatus(
                healthy=False,
                timestamp=datetime.now(),
                orchestrator_id=self._unified_id,
                alerts=[f"Health check failed: {e}"]
            )
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive system metrics.
        
        Returns:
            Dict[str, Any]: Unified metrics from all orchestrators
        """
        try:
            base_metrics = {
                'unified_orchestrator_id': self._unified_id,
                'system_initialized': self._is_initialized,
                'system_running': self._is_running,
                'active_orchestrators': len(self._orchestrators),
                'timestamp': datetime.now().isoformat(),
            }
            
            if self._is_initialized:
                # Get comprehensive metrics from coordinator
                coordinator_metrics = await self._coordinator.get_system_metrics()
                base_metrics.update(coordinator_metrics)
                
                # Add registry statistics
                registry_stats = self._registry.get_registry_stats()
                base_metrics['registry_stats'] = registry_stats
            
            return base_metrics
            
        except Exception as e:
            logger.exception(f"Failed to get system metrics: {e}")
            return {
                'unified_orchestrator_id': self._unified_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
            }
    
    # Convenience methods for specific orchestration tasks
    
    async def run_ml_pipeline(self, resume_from: Optional[str] = None) -> OrchestrationResult:
        """Run ML pipeline through unified interface."""
        task_context = TaskContext(
            task_type=TaskType.ML_PIPELINE,
            metadata={
                'operation': 'run_pipeline',
                'resume_from': resume_from
            }
        )
        return await self.process_task(task_context)
    
    async def create_agent(
        self,
        agent_type: str,
        agent_id: Optional[str] = None,
        **kwargs
    ) -> OrchestrationResult:
        """Create agent through unified interface."""
        task_context = TaskContext(
            task_type=TaskType.AGENT_LIFECYCLE,
            metadata={
                'operation': 'create_agent',
                'agent_type': agent_type,
                'agent_id': agent_id,
                'kwargs': kwargs
            }
        )
        return await self.process_task(task_context)
    
    async def analyze_information(
        self,
        query: str,
        retrieved_info: List[Any],
        analysis_types: Optional[List[str]] = None
    ) -> OrchestrationResult:
        """Perform cognitive analysis through unified interface."""
        task_context = TaskContext(
            task_type=TaskType.COGNITIVE_ANALYSIS,
            metadata={
                'operation': 'analyze',
                'query': query,
                'retrieved_info': retrieved_info,
                'analysis_types': analysis_types
            }
        )
        return await self.process_task(task_context)
    
    async def process_fog_request(
        self,
        request_type: str,
        request_data: Dict[str, Any]
    ) -> OrchestrationResult:
        """Process fog computing request through unified interface."""
        task_context = TaskContext(
            task_type=TaskType.FOG_COORDINATION,
            metadata={
                'operation': 'process_fog_request',
                'request_type': request_type,
                'request_data': request_data
            }
        )
        return await self.process_task(task_context)
    
    # Private helper methods
    
    async def _register_orchestrator_types(
        self,
        enable_ml: bool,
        enable_agent: bool, 
        enable_cognitive: bool,
        enable_fog: bool
    ) -> None:
        """Register orchestrator types with the registry."""
        if enable_ml:
            self._registry.register_type(
                orchestrator_type="ml_pipeline",
                orchestrator_class=MLPipelineOrchestrator,
                description="ML pipeline orchestration with 7-phase training",
                dependencies=[],  # No dependencies
                priority=10,  # High priority
                enabled=True
            )
        
        if enable_agent:
            self._registry.register_type(
                orchestrator_type="agent_lifecycle",
                orchestrator_class=AgentLifecycleOrchestrator,
                description="Agent lifecycle and task management",
                dependencies=["cognitive_analysis"] if enable_cognitive else [],
                priority=20,
                enabled=True
            )
        
        if enable_cognitive:
            self._registry.register_type(
                orchestrator_type="cognitive_analysis",
                orchestrator_class=CognitiveAnalysisOrchestrator,
                description="Advanced cognitive reasoning and analysis",
                dependencies=["fog_system"] if enable_fog else [],
                priority=30,
                enabled=True
            )
        
        if enable_fog:
            self._registry.register_type(
                orchestrator_type="fog_system",
                orchestrator_class=FogSystemOrchestrator,
                description="Fog computing and distributed system coordination",
                dependencies=[],  # Base infrastructure
                priority=40,  # Lower priority (infrastructure)
                enabled=True
            )
    
    def _get_orchestrator_dependencies(self, orchestrator_type: str) -> List[str]:
        """Get dependencies for an orchestrator type."""
        dependency_map = {
            'ml_pipeline': [],
            'agent_lifecycle': ['cognitive_analysis'],
            'cognitive_analysis': ['fog_system'],
            'fog_system': [],
        }
        return dependency_map.get(orchestrator_type, [])
    
    def _get_orchestrator_priority(self, orchestrator_type: str) -> int:
        """Get initialization priority for orchestrator type."""
        priority_map = {
            'fog_system': 10,      # Infrastructure first
            'cognitive_analysis': 20,  # Analysis capabilities
            'agent_lifecycle': 30,     # Agent management
            'ml_pipeline': 40,         # ML training last
        }
        return priority_map.get(orchestrator_type, 50)
    
    @property
    def is_initialized(self) -> bool:
        """Check if system is initialized."""
        return self._is_initialized
    
    @property
    def is_running(self) -> bool:
        """Check if system is running."""
        return self._is_running
    
    @property
    def orchestrator_count(self) -> int:
        """Get number of active orchestrators."""
        return len(self._orchestrators)
    
    def get_orchestrator(self, orchestrator_type: str) -> Optional[BaseOrchestrator]:
        """Get a specific orchestrator by type."""
        return self._orchestrators.get(orchestrator_type)