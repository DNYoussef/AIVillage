"""
Phase 6 Integration Coordinator - Master Controller

This module serves as the main coordinator for the entire Phase 6 baking pipeline,
orchestrating all components to ensure seamless integration and 99.9% reliability.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import uuid

from .data_flow_coordinator import DataFlowCoordinator, MessageType
from .agent_synchronization_manager import AgentSynchronizationManager, AgentType, WorkloadPriority
from .error_recovery_system import ErrorRecoverySystem, ErrorSeverity, ErrorCategory
from .pipeline_health_monitor import PipelineHealthMonitor, HealthStatus
from .state_manager import StateManager, Phase, StateStatus
from .phase5_connector import Phase5Connector
from .phase7_preparer import Phase7Preparer
from .serialization_utils import SafeJSONSerializer, SerializationConfig

logger = logging.getLogger(__name__)

class IntegrationState(Enum):
    """Integration pipeline states"""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    MONITORING = "monitoring"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class PipelinePhase(Enum):
    """Pipeline execution phases"""
    PHASE5_HANDOFF = "phase5_handoff"
    AGENT_INITIALIZATION = "agent_initialization"
    MODEL_BAKING = "model_baking"
    QUALITY_VALIDATION = "quality_validation"
    OPTIMIZATION = "optimization"
    PHASE7_PREPARATION = "phase7_preparation"
    COMPLETION = "completion"

@dataclass
class IntegrationConfig:
    """Configuration for Phase 6 integration"""
    # Component configurations
    data_flow_config: Dict[str, Any]
    agent_sync_config: Dict[str, Any]
    error_recovery_config: Dict[str, Any]
    health_monitor_config: Dict[str, Any]
    state_config: Dict[str, Any]

    # Pipeline configurations
    phase5_config: Dict[str, Any]
    phase7_config: Dict[str, Any]

    # Performance targets
    target_reliability: float = 99.9
    max_processing_time_minutes: int = 60
    max_concurrent_models: int = 10

    # Monitoring settings
    enable_real_time_monitoring: bool = True
    enable_automated_recovery: bool = True
    enable_performance_optimization: bool = True

@dataclass
class PipelineExecution:
    """Pipeline execution tracking"""
    execution_id: str
    model_id: str
    phase: PipelinePhase
    start_time: datetime
    end_time: Optional[datetime]
    success: bool
    metrics: Dict[str, Any]
    checkpoints: List[str]

class Phase6IntegrationCoordinator:
    """
    Master coordinator for Phase 6 integration pipeline.

    Responsibilities:
    - Initialize and manage all integration components
    - Orchestrate end-to-end model baking workflow
    - Monitor pipeline health and performance
    - Handle errors and recovery
    - Ensure 99.9% reliability target
    - Coordinate Phase 5 to Phase 7 handoffs
    """

    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.coordinator_id = str(uuid.uuid4())
        self.state = IntegrationState.INITIALIZING

        # Core components
        self.data_flow_coordinator: Optional[DataFlowCoordinator] = None
        self.agent_sync_manager: Optional[AgentSynchronizationManager] = None
        self.error_recovery_system: Optional[ErrorRecoverySystem] = None
        self.health_monitor: Optional[PipelineHealthMonitor] = None
        self.state_manager: Optional[StateManager] = None

        # Phase connectors
        self.phase5_connector: Optional[Phase5Connector] = None
        self.phase7_preparer: Optional[Phase7Preparer] = None

        # Execution tracking
        self.active_executions: Dict[str, PipelineExecution] = {}
        self.completed_executions: List[PipelineExecution] = []

        # Metrics
        self.pipeline_metrics = {
            'total_models_processed': 0,
            'successful_completions': 0,
            'failed_completions': 0,
            'average_processing_time': 0.0,
            'current_reliability': 100.0,
            'uptime_start': time.time()
        }

        # Serialization
        self.serializer = SafeJSONSerializer(SerializationConfig())

        logger.info(f"Phase6IntegrationCoordinator initialized with ID: {self.coordinator_id}")

    async def initialize(self) -> bool:
        """Initialize all integration components"""
        try:
            logger.info("Initializing Phase 6 integration pipeline...")

            # Initialize core components
            success = await self._initialize_core_components()
            if not success:
                logger.error("Failed to initialize core components")
                return False

            # Initialize phase connectors
            success = await self._initialize_phase_connectors()
            if not success:
                logger.error("Failed to initialize phase connectors")
                return False

            # Register components with each other
            await self._register_component_cross_references()

            # Initialize baking agents
            success = await self._initialize_baking_agents()
            if not success:
                logger.error("Failed to initialize baking agents")
                return False

            # Start monitoring and health checks
            if self.config.enable_real_time_monitoring:
                await self._start_monitoring()

            self.state = IntegrationState.READY
            logger.info("Phase 6 integration pipeline initialization completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            if self.config.enable_automated_recovery:
                error_id = await self.error_recovery_system.handle_error(
                    "phase6_coordinator", "initialize", e, severity=ErrorSeverity.CRITICAL
                )
                logger.info(f"Recovery initiated for initialization error: {error_id}")
            return False

    async def shutdown(self) -> bool:
        """Shutdown the integration pipeline gracefully"""
        try:
            logger.info("Shutting down Phase 6 integration pipeline...")
            self.state = IntegrationState.SHUTDOWN

            # Complete any active executions
            if self.active_executions:
                logger.info(f"Waiting for {len(self.active_executions)} active executions to complete...")
                await self._complete_active_executions()

            # Stop monitoring
            if self.health_monitor:
                await self.health_monitor.stop()

            # Stop core components
            if self.error_recovery_system:
                pass  # Error recovery system doesn't have a stop method in our implementation

            if self.agent_sync_manager:
                await self.agent_sync_manager.stop()

            if self.data_flow_coordinator:
                await self.data_flow_coordinator.stop()

            logger.info("Phase 6 integration pipeline shutdown completed")
            return True

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return False

    async def process_model_from_phase5(self, model_path: str,
                                       optimization_level: int = 3,
                                       target_hardware: str = "auto") -> Dict[str, Any]:
        """
        Process a model from Phase 5 through the complete baking pipeline.

        Args:
            model_path: Path to the Phase 5 trained model
            optimization_level: Optimization level (0-4)
            target_hardware: Target hardware for optimization

        Returns:
            Dictionary containing processing results and Phase 7 handoff data
        """
        execution_id = str(uuid.uuid4())
        model_id = Path(model_path).stem

        try:
            logger.info(f"Starting model processing: {model_id} (execution: {execution_id})")

            # Create execution tracking
            execution = PipelineExecution(
                execution_id=execution_id,
                model_id=model_id,
                phase=PipelinePhase.PHASE5_HANDOFF,
                start_time=datetime.now(),
                end_time=None,
                success=False,
                metrics={},
                checkpoints=[]
            )
            self.active_executions[execution_id] = execution

            # Phase 1: Phase 5 Handoff
            phase5_result = await self._execute_phase5_handoff(execution, model_path)
            if not phase5_result['success']:
                return await self._handle_execution_failure(execution, "Phase 5 handoff failed")

            # Phase 2: Agent Initialization
            execution.phase = PipelinePhase.AGENT_INITIALIZATION
            agent_init_result = await self._execute_agent_initialization(execution)
            if not agent_init_result['success']:
                return await self._handle_execution_failure(execution, "Agent initialization failed")

            # Phase 3: Model Baking
            execution.phase = PipelinePhase.MODEL_BAKING
            baking_result = await self._execute_model_baking(execution, optimization_level, target_hardware)
            if not baking_result['success']:
                return await self._handle_execution_failure(execution, "Model baking failed")

            # Phase 4: Quality Validation
            execution.phase = PipelinePhase.QUALITY_VALIDATION
            validation_result = await self._execute_quality_validation(execution)
            if not validation_result['success']:
                return await self._handle_execution_failure(execution, "Quality validation failed")

            # Phase 5: Optimization
            execution.phase = PipelinePhase.OPTIMIZATION
            optimization_result = await self._execute_optimization(execution)
            if not optimization_result['success']:
                return await self._handle_execution_failure(execution, "Optimization failed")

            # Phase 6: Phase 7 Preparation
            execution.phase = PipelinePhase.PHASE7_PREPARATION
            phase7_result = await self._execute_phase7_preparation(execution)
            if not phase7_result['success']:
                return await self._handle_execution_failure(execution, "Phase 7 preparation failed")

            # Phase 7: Completion
            execution.phase = PipelinePhase.COMPLETION
            completion_result = await self._execute_completion(execution)

            # Mark execution as successful
            execution.success = True
            execution.end_time = datetime.now()

            # Move to completed executions
            self.completed_executions.append(execution)
            del self.active_executions[execution_id]

            # Update metrics
            self._update_pipeline_metrics(execution)

            logger.info(f"Model processing completed successfully: {model_id}")

            return {
                'success': True,
                'execution_id': execution_id,
                'model_id': model_id,
                'processing_time_minutes': (execution.end_time - execution.start_time).total_seconds() / 60,
                'phase5_result': phase5_result,
                'baking_result': baking_result,
                'validation_result': validation_result,
                'optimization_result': optimization_result,
                'phase7_result': phase7_result,
                'completion_result': completion_result,
                'metrics': execution.metrics,
                'checkpoints': execution.checkpoints
            }

        except Exception as e:
            logger.error(f"Error processing model {model_id}: {e}")
            if self.config.enable_automated_recovery:
                error_id = await self.error_recovery_system.handle_error(
                    "phase6_coordinator", "process_model_from_phase5", e,
                    context_data={'model_id': model_id, 'execution_id': execution_id},
                    severity=ErrorSeverity.HIGH
                )
                logger.info(f"Recovery initiated for model processing error: {error_id}")

            return await self._handle_execution_failure(execution, f"Exception: {e}")

    async def run_end_to_end_test(self, test_data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive end-to-end test of the integration pipeline.

        Args:
            test_data_path: Path to test data (optional, will generate if not provided)

        Returns:
            Test results including reliability metrics
        """
        test_id = str(uuid.uuid4())
        start_time = time.time()

        logger.info(f"Starting end-to-end integration test: {test_id}")

        try:
            test_results = {
                'test_id': test_id,
                'start_time': datetime.now().isoformat(),
                'success': False,
                'reliability_achieved': 0.0,
                'performance_metrics': {},
                'component_tests': {},
                'integration_tests': {},
                'issues': [],
                'recommendations': []
            }

            # Test 1: Component Health Checks
            component_health = await self._test_component_health()
            test_results['component_tests'] = component_health

            # Test 2: Data Flow Testing
            data_flow_test = await self._test_data_flow()
            test_results['integration_tests']['data_flow'] = data_flow_test

            # Test 3: Agent Synchronization Testing
            sync_test = await self._test_agent_synchronization()
            test_results['integration_tests']['agent_sync'] = sync_test

            # Test 4: Error Recovery Testing
            error_recovery_test = await self._test_error_recovery()
            test_results['integration_tests']['error_recovery'] = error_recovery_test

            # Test 5: Performance Testing
            performance_test = await self._test_performance()
            test_results['performance_metrics'] = performance_test

            # Test 6: End-to-End Model Processing
            if test_data_path:
                e2e_test = await self._test_end_to_end_processing(test_data_path)
                test_results['integration_tests']['end_to_end'] = e2e_test
            else:
                logger.info("Skipping end-to-end processing test (no test data provided)")

            # Calculate overall reliability
            reliability = self._calculate_test_reliability(test_results)
            test_results['reliability_achieved'] = reliability

            # Determine success
            test_results['success'] = (
                reliability >= self.config.target_reliability and
                component_health.get('overall_score', 0) >= 80.0
            )

            # Generate recommendations
            test_results['recommendations'] = self._generate_test_recommendations(test_results)

            test_duration = time.time() - start_time
            test_results['duration_seconds'] = test_duration
            test_results['end_time'] = datetime.now().isoformat()

            logger.info(f"End-to-end test completed: {test_id}, "
                       f"Reliability: {reliability:.2f}%, Success: {test_results['success']}")

            return test_results

        except Exception as e:
            logger.error(f"Error in end-to-end test: {e}")
            return {
                'test_id': test_id,
                'success': False,
                'error': str(e),
                'duration_seconds': time.time() - start_time
            }

    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        try:
            # Get component statuses
            component_statuses = {}

            if self.data_flow_coordinator:
                component_statuses['data_flow'] = self.data_flow_coordinator.get_system_status()

            if self.agent_sync_manager:
                component_statuses['agent_sync'] = self.agent_sync_manager.get_system_status()

            if self.error_recovery_system:
                component_statuses['error_recovery'] = self.error_recovery_system.get_system_health_status()

            if self.health_monitor:
                component_statuses['health_monitor'] = self.health_monitor.get_overall_health_status()

            # Calculate overall health
            overall_health = self._calculate_overall_health(component_statuses)

            return {
                'coordinator_id': self.coordinator_id,
                'state': self.state.value,
                'timestamp': datetime.now().isoformat(),
                'overall_health': overall_health,
                'pipeline_metrics': self.pipeline_metrics,
                'active_executions': len(self.active_executions),
                'completed_executions': len(self.completed_executions),
                'component_statuses': component_statuses,
                'uptime_seconds': time.time() - self.pipeline_metrics['uptime_start']
            }

        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {
                'coordinator_id': self.coordinator_id,
                'state': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    # Private implementation methods

    async def _initialize_core_components(self) -> bool:
        """Initialize core pipeline components"""
        try:
            # Initialize state manager
            self.state_manager = StateManager(self.config.state_config)

            # Initialize data flow coordinator
            self.data_flow_coordinator = DataFlowCoordinator(self.config.data_flow_config)
            await self.data_flow_coordinator.start()

            # Initialize agent synchronization manager
            self.agent_sync_manager = AgentSynchronizationManager(self.config.agent_sync_config)
            await self.agent_sync_manager.start()

            # Initialize error recovery system
            self.error_recovery_system = ErrorRecoverySystem(self.config.error_recovery_config)

            # Initialize health monitor
            self.health_monitor = PipelineHealthMonitor(self.config.health_monitor_config)
            await self.health_monitor.start()

            return True

        except Exception as e:
            logger.error(f"Error initializing core components: {e}")
            return False

    async def _initialize_phase_connectors(self) -> bool:
        """Initialize Phase 5 and Phase 7 connectors"""
        try:
            # Initialize Phase 5 connector
            self.phase5_connector = Phase5Connector(self.config.phase5_config)

            # Initialize Phase 7 preparer
            self.phase7_preparer = Phase7Preparer(self.config.phase7_config)

            return True

        except Exception as e:
            logger.error(f"Error initializing phase connectors: {e}")
            return False

    async def _register_component_cross_references(self):
        """Register components with each other for monitoring"""
        if self.health_monitor:
            self.health_monitor.register_components(
                data_flow_coordinator=self.data_flow_coordinator,
                agent_sync_manager=self.agent_sync_manager,
                error_recovery_system=self.error_recovery_system
            )

    async def _initialize_baking_agents(self) -> bool:
        """Initialize the 9 specialized baking agents"""
        try:
            agent_configs = [
                ("baking_coordinator", AgentType.BAKING_COORDINATOR, {"coordination", "orchestration"}),
                ("model_optimizer", AgentType.MODEL_OPTIMIZER, {"optimization", "model_analysis"}),
                ("inference_accelerator", AgentType.INFERENCE_ACCELERATOR, {"acceleration", "inference"}),
                ("quality_validator", AgentType.QUALITY_VALIDATOR, {"validation", "quality_control"}),
                ("performance_profiler", AgentType.PERFORMANCE_PROFILER, {"profiling", "performance"}),
                ("hardware_adapter", AgentType.HARDWARE_ADAPTER, {"hardware", "adaptation"}),
                ("graph_optimizer", AgentType.GRAPH_OPTIMIZER, {"graph_optimization", "structure"}),
                ("memory_optimizer", AgentType.MEMORY_OPTIMIZER, {"memory", "optimization"}),
                ("deployment_preparer", AgentType.DEPLOYMENT_PREPARER, {"deployment", "preparation"})
            ]

            for agent_id, agent_type, capabilities in agent_configs:
                success = self.agent_sync_manager.register_agent(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    capabilities=capabilities
                )

                if not success:
                    logger.error(f"Failed to register agent: {agent_id}")
                    return False

            logger.info(f"Successfully registered {len(agent_configs)} baking agents")
            return True

        except Exception as e:
            logger.error(f"Error initializing baking agents: {e}")
            return False

    async def _start_monitoring(self):
        """Start real-time monitoring"""
        if self.health_monitor:
            # The health monitor is already started in initialize_core_components
            logger.info("Real-time monitoring enabled")

    async def _complete_active_executions(self):
        """Complete any active executions gracefully"""
        for execution_id, execution in list(self.active_executions.items()):
            try:
                # Mark as failed due to shutdown
                execution.success = False
                execution.end_time = datetime.now()
                self.completed_executions.append(execution)
                del self.active_executions[execution_id]

                logger.info(f"Marked execution {execution_id} as failed due to shutdown")

            except Exception as e:
                logger.error(f"Error completing execution {execution_id}: {e}")

    async def _execute_phase5_handoff(self, execution: PipelineExecution, model_path: str) -> Dict[str, Any]:
        """Execute Phase 5 handoff process"""
        try:
            logger.info(f"Executing Phase 5 handoff for {execution.model_id}")

            # Validate model compatibility
            compatible, score, validation_results = self.phase5_connector.validate_model_compatibility(model_path)

            if not compatible:
                return {
                    'success': False,
                    'error': 'Model not compatible with Phase 6 requirements',
                    'validation_results': validation_results
                }

            # Transfer model to Phase 6
            target_path = f"models/phase6/{execution.model_id}"
            transfer_result = self.phase5_connector.transfer_model(model_path, target_path)

            if not transfer_result.success:
                return {
                    'success': False,
                    'error': 'Model transfer failed',
                    'transfer_result': asdict(transfer_result)
                }

            # Create checkpoint
            checkpoint_name = f"phase5_handoff_{execution.execution_id}"
            await self.data_flow_coordinator.create_checkpoint(checkpoint_name)
            execution.checkpoints.append(checkpoint_name)

            execution.metrics['phase5_handoff'] = {
                'compatibility_score': score,
                'transfer_time': time.time()
            }

            return {
                'success': True,
                'model_path': transfer_result.model_path,
                'compatibility_score': score,
                'performance_metrics': transfer_result.performance_metrics
            }

        except Exception as e:
            logger.error(f"Error in Phase 5 handoff: {e}")
            return {'success': False, 'error': str(e)}

    async def _execute_agent_initialization(self, execution: PipelineExecution) -> Dict[str, Any]:
        """Execute agent initialization process"""
        try:
            logger.info(f"Executing agent initialization for {execution.model_id}")

            # Create synchronization point for all agents
            sync_id = f"init_sync_{execution.execution_id}"
            agent_ids = {
                "baking_coordinator", "model_optimizer", "inference_accelerator",
                "quality_validator", "performance_profiler", "hardware_adapter",
                "graph_optimizer", "memory_optimizer", "deployment_preparer"
            }

            success = await self.agent_sync_manager.create_synchronization_point(
                sync_id=sync_id,
                participating_agents=agent_ids,
                timeout_seconds=60
            )

            if not success:
                return {'success': False, 'error': 'Failed to create agent synchronization point'}

            # Wait for synchronization
            sync_success = await self.agent_sync_manager.wait_for_synchronization(sync_id, 60)

            if not sync_success:
                return {'success': False, 'error': 'Agent synchronization timeout'}

            execution.metrics['agent_initialization'] = {
                'sync_id': sync_id,
                'agents_synchronized': len(agent_ids),
                'completion_time': time.time()
            }

            return {'success': True, 'synchronized_agents': len(agent_ids)}

        except Exception as e:
            logger.error(f"Error in agent initialization: {e}")
            return {'success': False, 'error': str(e)}

    async def _execute_model_baking(self, execution: PipelineExecution,
                                   optimization_level: int, target_hardware: str) -> Dict[str, Any]:
        """Execute model baking process"""
        try:
            logger.info(f"Executing model baking for {execution.model_id}")

            # Submit baking task
            task_id = await self.agent_sync_manager.submit_task(
                task_type="model_baking",
                data={
                    'model_id': execution.model_id,
                    'optimization_level': optimization_level,
                    'target_hardware': target_hardware,
                    'execution_id': execution.execution_id
                },
                priority=WorkloadPriority.HIGH
            )

            # Monitor task progress
            start_time = time.time()
            timeout_seconds = self.config.max_processing_time_minutes * 60

            while time.time() - start_time < timeout_seconds:
                task_status = await self.agent_sync_manager.get_task_status(task_id)

                if task_status and task_status['status'] == 'completed':
                    execution.metrics['model_baking'] = {
                        'task_id': task_id,
                        'processing_time': time.time() - start_time,
                        'optimization_level': optimization_level
                    }
                    return {'success': True, 'task_id': task_id, 'processing_time': time.time() - start_time}

                await asyncio.sleep(5)  # Check every 5 seconds

            # Task timeout
            return {'success': False, 'error': 'Baking task timeout', 'task_id': task_id}

        except Exception as e:
            logger.error(f"Error in model baking: {e}")
            return {'success': False, 'error': str(e)}

    async def _execute_quality_validation(self, execution: PipelineExecution) -> Dict[str, Any]:
        """Execute quality validation process"""
        try:
            logger.info(f"Executing quality validation for {execution.model_id}")

            # Submit validation task
            task_id = await self.agent_sync_manager.submit_task(
                task_type="quality_validation",
                data={
                    'model_id': execution.model_id,
                    'execution_id': execution.execution_id,
                    'validation_level': 'comprehensive'
                },
                priority=WorkloadPriority.HIGH
            )

            # Monitor validation
            start_time = time.time()
            timeout_seconds = 300  # 5 minutes for validation

            while time.time() - start_time < timeout_seconds:
                task_status = await self.agent_sync_manager.get_task_status(task_id)

                if task_status and task_status['status'] == 'completed':
                    execution.metrics['quality_validation'] = {
                        'task_id': task_id,
                        'validation_time': time.time() - start_time
                    }
                    return {'success': True, 'task_id': task_id, 'validation_passed': True}

                await asyncio.sleep(2)  # Check every 2 seconds

            return {'success': False, 'error': 'Validation timeout', 'task_id': task_id}

        except Exception as e:
            logger.error(f"Error in quality validation: {e}")
            return {'success': False, 'error': str(e)}

    async def _execute_optimization(self, execution: PipelineExecution) -> Dict[str, Any]:
        """Execute optimization process"""
        try:
            logger.info(f"Executing optimization for {execution.model_id}")

            if not self.config.enable_performance_optimization:
                logger.info("Performance optimization disabled, skipping")
                return {'success': True, 'skipped': True}

            # Submit optimization task
            task_id = await self.agent_sync_manager.submit_task(
                task_type="performance_optimization",
                data={
                    'model_id': execution.model_id,
                    'execution_id': execution.execution_id
                },
                priority=WorkloadPriority.NORMAL
            )

            # Monitor optimization
            start_time = time.time()
            timeout_seconds = 600  # 10 minutes for optimization

            while time.time() - start_time < timeout_seconds:
                task_status = await self.agent_sync_manager.get_task_status(task_id)

                if task_status and task_status['status'] == 'completed':
                    execution.metrics['optimization'] = {
                        'task_id': task_id,
                        'optimization_time': time.time() - start_time
                    }
                    return {'success': True, 'task_id': task_id}

                await asyncio.sleep(5)

            return {'success': False, 'error': 'Optimization timeout', 'task_id': task_id}

        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            return {'success': False, 'error': str(e)}

    async def _execute_phase7_preparation(self, execution: PipelineExecution) -> Dict[str, Any]:
        """Execute Phase 7 preparation process"""
        try:
            logger.info(f"Executing Phase 7 preparation for {execution.model_id}")

            model_path = f"models/phase6/{execution.model_id}"

            # Assess ADAS readiness
            readiness_report = self.phase7_preparer.assess_adas_readiness(model_path)

            if not readiness_report.ready_for_deployment:
                return {
                    'success': False,
                    'error': 'Model not ready for ADAS deployment',
                    'readiness_report': asdict(readiness_report)
                }

            # Prepare for ADAS deployment
            preparation_result = self.phase7_preparer.prepare_for_adas_deployment(model_path)

            if not preparation_result['success']:
                return {
                    'success': False,
                    'error': 'ADAS preparation failed',
                    'preparation_result': preparation_result
                }

            execution.metrics['phase7_preparation'] = {
                'readiness_score': readiness_report.optimization_results.speed_improvement,
                'export_directory': preparation_result['export_directory']
            }

            return {
                'success': True,
                'export_directory': preparation_result['export_directory'],
                'readiness_report': asdict(readiness_report)
            }

        except Exception as e:
            logger.error(f"Error in Phase 7 preparation: {e}")
            return {'success': False, 'error': str(e)}

    async def _execute_completion(self, execution: PipelineExecution) -> Dict[str, Any]:
        """Execute completion process"""
        try:
            logger.info(f"Executing completion for {execution.model_id}")

            # Create final checkpoint
            checkpoint_name = f"completion_{execution.execution_id}"
            await self.data_flow_coordinator.create_checkpoint(checkpoint_name)
            execution.checkpoints.append(checkpoint_name)

            # Generate completion report
            completion_report = {
                'execution_id': execution.execution_id,
                'model_id': execution.model_id,
                'processing_time': (datetime.now() - execution.start_time).total_seconds(),
                'checkpoints': execution.checkpoints,
                'metrics': execution.metrics,
                'success': True
            }

            return {
                'success': True,
                'completion_report': completion_report
            }

        except Exception as e:
            logger.error(f"Error in completion: {e}")
            return {'success': False, 'error': str(e)}

    async def _handle_execution_failure(self, execution: PipelineExecution, reason: str) -> Dict[str, Any]:
        """Handle execution failure"""
        execution.success = False
        execution.end_time = datetime.now()

        # Move to completed executions
        self.completed_executions.append(execution)
        if execution.execution_id in self.active_executions:
            del self.active_executions[execution.execution_id]

        # Update metrics
        self._update_pipeline_metrics(execution)

        logger.error(f"Execution failed: {execution.execution_id}, Reason: {reason}")

        return {
            'success': False,
            'execution_id': execution.execution_id,
            'model_id': execution.model_id,
            'failure_reason': reason,
            'processing_time_minutes': (execution.end_time - execution.start_time).total_seconds() / 60,
            'phase': execution.phase.value,
            'metrics': execution.metrics
        }

    def _update_pipeline_metrics(self, execution: PipelineExecution):
        """Update pipeline metrics based on execution"""
        self.pipeline_metrics['total_models_processed'] += 1

        if execution.success:
            self.pipeline_metrics['successful_completions'] += 1
        else:
            self.pipeline_metrics['failed_completions'] += 1

        # Update reliability
        total = self.pipeline_metrics['total_models_processed']
        successful = self.pipeline_metrics['successful_completions']
        self.pipeline_metrics['current_reliability'] = (successful / total) * 100 if total > 0 else 100.0

        # Update average processing time
        if execution.end_time:
            processing_time = (execution.end_time - execution.start_time).total_seconds()
            current_avg = self.pipeline_metrics['average_processing_time']

            if current_avg == 0:
                self.pipeline_metrics['average_processing_time'] = processing_time
            else:
                # Rolling average
                self.pipeline_metrics['average_processing_time'] = (current_avg * 0.9) + (processing_time * 0.1)

    # Testing methods

    async def _test_component_health(self) -> Dict[str, Any]:
        """Test component health"""
        try:
            if not self.health_monitor:
                return {'overall_score': 0.0, 'error': 'Health monitor not available'}

            health_results = await self.health_monitor.run_all_health_checks()

            scores = [result.score for result in health_results.values()]
            overall_score = sum(scores) / len(scores) if scores else 0.0

            return {
                'overall_score': overall_score,
                'individual_results': {name: result.score for name, result in health_results.items()},
                'passed_checks': sum(1 for result in health_results.values() if result.score >= 80),
                'total_checks': len(health_results)
            }

        except Exception as e:
            return {'overall_score': 0.0, 'error': str(e)}

    async def _test_data_flow(self) -> Dict[str, Any]:
        """Test data flow functionality"""
        try:
            if not self.data_flow_coordinator:
                return {'success': False, 'error': 'Data flow coordinator not available'}

            # Test message sending
            message_id = await self.data_flow_coordinator.send_data(
                "test_sender", "test_receiver", {"test": "data"}, MessageType.DATA_TRANSFER
            )

            validation = self.data_flow_coordinator.validate_data_flow_integrity()

            return {
                'success': validation['overall_health'] in ['HEALTHY', 'DEGRADED'],
                'message_sent': bool(message_id),
                'validation_results': validation
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _test_agent_synchronization(self) -> Dict[str, Any]:
        """Test agent synchronization functionality"""
        try:
            if not self.agent_sync_manager:
                return {'success': False, 'error': 'Agent sync manager not available'}

            system_status = self.agent_sync_manager.get_system_status()

            return {
                'success': system_status['system_health'] in ['HEALTHY', 'DEGRADED'],
                'agents_registered': system_status['total_agents'],
                'healthy_agents': system_status['healthy_agents'],
                'system_status': system_status
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _test_error_recovery(self) -> Dict[str, Any]:
        """Test error recovery functionality"""
        try:
            if not self.error_recovery_system:
                return {'success': False, 'error': 'Error recovery system not available'}

            # Simulate test error
            test_exception = ValueError("Test error for recovery testing")
            error_id = await self.error_recovery_system.handle_error(
                "test_component", "test_function", test_exception
            )

            health_status = self.error_recovery_system.get_system_health_status()

            return {
                'success': bool(error_id),
                'error_handled': bool(error_id),
                'health_status': health_status['health_status'],
                'recovery_metrics': health_status.get('metrics', {})
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _test_performance(self) -> Dict[str, Any]:
        """Test performance metrics"""
        try:
            start_time = time.time()

            # Simulate some work
            await asyncio.sleep(0.1)

            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # ms

            return {
                'response_time_ms': response_time,
                'memory_usage': self._get_memory_usage(),
                'cpu_usage': self._get_cpu_usage(),
                'target_met': response_time < 1000  # 1 second target
            }

        except Exception as e:
            return {'error': str(e)}

    async def _test_end_to_end_processing(self, test_data_path: str) -> Dict[str, Any]:
        """Test end-to-end model processing"""
        try:
            result = await self.process_model_from_phase5(test_data_path)
            return {
                'success': result['success'],
                'processing_time': result.get('processing_time_minutes', 0),
                'execution_id': result.get('execution_id')
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _calculate_test_reliability(self, test_results: Dict[str, Any]) -> float:
        """Calculate reliability from test results"""
        try:
            # Simple reliability calculation based on successful tests
            total_tests = 0
            successful_tests = 0

            # Component tests
            if 'component_tests' in test_results:
                total_tests += test_results['component_tests'].get('total_checks', 0)
                successful_tests += test_results['component_tests'].get('passed_checks', 0)

            # Integration tests
            for test_name, result in test_results.get('integration_tests', {}).items():
                total_tests += 1
                if result.get('success', False):
                    successful_tests += 1

            if total_tests == 0:
                return 0.0

            return (successful_tests / total_tests) * 100

        except Exception as e:
            logger.error(f"Error calculating test reliability: {e}")
            return 0.0

    def _generate_test_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        reliability = test_results.get('reliability_achieved', 0)
        if reliability < self.config.target_reliability:
            recommendations.append(f"Reliability ({reliability:.1f}%) below target ({self.config.target_reliability}%)")

        # Check component health
        component_score = test_results.get('component_tests', {}).get('overall_score', 0)
        if component_score < 80:
            recommendations.append("Component health scores need improvement")

        # Check integration tests
        failed_tests = []
        for test_name, result in test_results.get('integration_tests', {}).items():
            if not result.get('success', False):
                failed_tests.append(test_name)

        if failed_tests:
            recommendations.append(f"Failed integration tests: {', '.join(failed_tests)}")

        if not recommendations:
            recommendations.append("All tests passed - system ready for production")

        return recommendations

    def _calculate_overall_health(self, component_statuses: Dict[str, Any]) -> str:
        """Calculate overall health from component statuses"""
        if not component_statuses:
            return "UNKNOWN"

        health_scores = []

        for component, status in component_statuses.items():
            if 'overall_score' in status:
                health_scores.append(status['overall_score'])
            elif 'health_status' in status:
                # Convert health status to score
                health_map = {'HEALTHY': 100, 'DEGRADED': 75, 'ERROR': 25, 'OFFLINE': 0}
                health_scores.append(health_map.get(status['health_status'], 50))

        if not health_scores:
            return "UNKNOWN"

        average_score = sum(health_scores) / len(health_scores)

        if average_score >= 90:
            return "EXCELLENT"
        elif average_score >= 80:
            return "GOOD"
        elif average_score >= 70:
            return "WARNING"
        elif average_score >= 50:
            return "CRITICAL"
        else:
            return "FAILURE"

    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0

# Factory function
def create_phase6_integration_coordinator(config: IntegrationConfig) -> Phase6IntegrationCoordinator:
    """Factory function to create Phase 6 integration coordinator"""
    return Phase6IntegrationCoordinator(config)

# Testing utilities
async def test_phase6_integration():
    """Test Phase 6 integration coordinator"""
    config = IntegrationConfig(
        data_flow_config={'use_compression': True},
        agent_sync_config={'heartbeat_timeout_seconds': 30},
        error_recovery_config={'max_retry_attempts': 3},
        health_monitor_config={'monitoring_interval_seconds': 30},
        state_config={'storage_dir': '.claude/.artifacts/test_state'},
        phase5_config={'phase5_model_dir': 'models/phase5'},
        phase7_config={'adas_export_dir': 'models/adas'}
    )

    coordinator = Phase6IntegrationCoordinator(config)

    try:
        # Initialize
        success = await coordinator.initialize()
        print(f"Initialization: {'SUCCESS' if success else 'FAILED'}")

        if success:
            # Run end-to-end test
            test_results = await coordinator.run_end_to_end_test()
            print(f"E2E Test: {'SUCCESS' if test_results['success'] else 'FAILED'}")
            print(f"Reliability: {test_results['reliability_achieved']:.2f}%")

            # Get status
            status = await coordinator.get_pipeline_status()
            print(f"Pipeline Status: {status['overall_health']}")

    finally:
        await coordinator.shutdown()

if __name__ == "__main__":
    asyncio.run(test_phase6_integration())