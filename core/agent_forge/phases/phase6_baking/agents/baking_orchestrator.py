"""
Phase 6 Baking - Baking Orchestrator Agent
Orchestrates and coordinates all Phase 6 baking agents for tool/persona optimization
"""

import torch
import torch.nn as nn
import asyncio
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import json
import pickle
from collections import defaultdict

# Import other baking agents
from .neural_model_optimizer import NeuralModelOptimizerAgent, create_default_optimizer_config
from .inference_accelerator import InferenceAcceleratorAgent, create_default_acceleration_config
from .quality_preservation_monitor import QualityPreservationMonitorAgent, create_default_quality_config
from .performance_profiler import PerformanceProfilerAgent, create_default_profiling_config
from .state_synchronizer import StateSynchronizer, AgentState as SyncAgentState
from .deployment_validator import DeploymentValidator
from .integration_tester import IntegrationTester
from .completion_auditor import CompletionAuditor

# Import communication infrastructure
from ..emergency.core_infrastructure import MessageBus, AgentMessage, MessageType, AgentStatus as CommAgentStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrchestrationPhase(Enum):
    INITIALIZATION = "initialization"
    BASELINE_ESTABLISHMENT = "baseline_establishment"
    OPTIMIZATION = "optimization"
    ACCELERATION = "acceleration"
    QUALITY_MONITORING = "quality_monitoring"
    PERFORMANCE_PROFILING = "performance_profiling"
    VALIDATION = "validation"
    INTEGRATION_TESTING = "integration_testing"
    COMPLETION_AUDIT = "completion_audit"
    FINALIZATION = "finalization"


class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"


@dataclass
class OrchestrationConfig:
    enable_parallel_execution: bool
    quality_threshold: float
    performance_threshold: float
    max_optimization_iterations: int
    early_stopping_patience: int
    enable_checkpointing: bool
    checkpoint_frequency: int
    rollback_on_failure: bool
    detailed_logging: bool
    export_results: bool


@dataclass
class AgentResult:
    agent_id: str
    status: AgentStatus
    result: Dict[str, Any]
    execution_time: float
    memory_usage: float
    error_message: Optional[str]
    timestamp: datetime


@dataclass
class OrchestrationState:
    current_phase: OrchestrationPhase
    phase_progress: float
    overall_progress: float
    active_agents: List[str]
    completed_agents: List[str]
    failed_agents: List[str]
    agent_results: Dict[str, AgentResult]
    quality_score: float
    performance_score: float
    optimization_metrics: Dict[str, Any]
    last_update: datetime


class BakingOrchestrator:
    """Advanced orchestrator for Phase 6 baking process"""

    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.state = OrchestrationState(
            current_phase=OrchestrationPhase.INITIALIZATION,
            phase_progress=0.0,
            overall_progress=0.0,
            active_agents=[],
            completed_agents=[],
            failed_agents=[],
            agent_results={},
            quality_score=0.0,
            performance_score=0.0,
            optimization_metrics={},
            last_update=datetime.now()
        )

        # Initialize agents
        self.agents = self._initialize_agents()
        self.model = None
        self.original_model = None
        self.checkpoints = {}

    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all baking agents"""
        # Initialize communication infrastructure
        self.message_bus = MessageBus()

        agents = {
            'neural_optimizer': NeuralModelOptimizerAgent(create_default_optimizer_config()),
            'inference_accelerator': InferenceAcceleratorAgent(create_default_acceleration_config()),
            'quality_monitor': QualityPreservationMonitorAgent(create_default_quality_config()),
            'performance_profiler': PerformanceProfilerAgent(create_default_profiling_config()),
            'state_synchronizer': StateSynchronizer(),
            'deployment_validator': DeploymentValidator(),
            'integration_tester': IntegrationTester(),
            'completion_auditor': CompletionAuditor()
        }

        # Register agents with message bus
        for agent_id, agent in agents.items():
            self.message_bus.subscribe(agent_id, [MessageType.COMMAND, MessageType.STATUS])

        logger.info(f"Initialized {len(agents)} baking agents with communication protocols")
        return agents

    async def orchestrate_baking(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """Orchestrate complete baking process"""
        self.model = model
        self.original_model = self._create_model_copy(model)

        try:
            # Phase 1: Initialization
            await self._execute_phase(OrchestrationPhase.INITIALIZATION)

            # Phase 2: Baseline Establishment
            await self._execute_phase(OrchestrationPhase.BASELINE_ESTABLISHMENT)

            # Phase 3: Optimization Loop
            optimization_results = await self._execute_optimization_loop(**kwargs)

            # Phase 4: Acceleration
            await self._execute_phase(OrchestrationPhase.ACCELERATION)

            # Phase 5: Validation and Testing
            await self._execute_validation_phase()

            # Phase 6: Completion Audit
            audit_results = await self._execute_phase(OrchestrationPhase.COMPLETION_AUDIT)

            # Phase 7: Finalization
            await self._execute_phase(OrchestrationPhase.FINALIZATION)

            return {
                'orchestration_successful': True,
                'final_model': self.model,
                'optimization_results': optimization_results,
                'audit_results': audit_results,
                'orchestration_state': asdict(self.state),
                'agent_results': {k: asdict(v) for k, v in self.state.agent_results.items()}
            }

        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            if self.config.rollback_on_failure:
                await self._rollback_to_checkpoint()

            return {
                'orchestration_successful': False,
                'error': str(e),
                'orchestration_state': asdict(self.state)
            }

    async def _execute_phase(self, phase: OrchestrationPhase) -> Dict[str, Any]:
        """Execute a specific orchestration phase"""
        self.state.current_phase = phase
        self.state.phase_progress = 0.0
        phase_start_time = time.time()

        logger.info(f"Starting phase: {phase.value}")

        try:
            if phase == OrchestrationPhase.INITIALIZATION:
                result = await self._phase_initialization()
            elif phase == OrchestrationPhase.BASELINE_ESTABLISHMENT:
                result = await self._phase_baseline_establishment()
            elif phase == OrchestrationPhase.ACCELERATION:
                result = await self._phase_acceleration()
            elif phase == OrchestrationPhase.COMPLETION_AUDIT:
                result = await self._phase_completion_audit()
            elif phase == OrchestrationPhase.FINALIZATION:
                result = await self._phase_finalization()
            else:
                result = {'phase': phase.value, 'status': 'skipped'}

            execution_time = time.time() - phase_start_time
            self.state.phase_progress = 100.0

            logger.info(f"Phase {phase.value} completed in {execution_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Phase {phase.value} failed: {e}")
            raise

    async def _phase_initialization(self) -> Dict[str, Any]:
        """Initialize orchestration environment"""
        # Create checkpoint
        if self.config.enable_checkpointing:
            self._create_checkpoint('initialization')

        # Initialize state synchronizer
        await self._run_agent('state_synchronizer', self.model)

        return {'phase': 'initialization', 'status': 'completed'}

    async def _phase_baseline_establishment(self) -> Dict[str, Any]:
        """Establish baseline metrics"""
        # Run quality monitor to establish baseline
        quality_result = await self._run_agent('quality_monitor', self.model)

        # Run performance profiler for baseline
        profiler_result = await self._run_agent('performance_profiler', self.model)

        self.state.quality_score = quality_result.get('quality_score', 0.0)
        self.state.performance_score = profiler_result.get('throughput_samples_per_sec', 0.0)

        return {
            'phase': 'baseline_establishment',
            'status': 'completed',
            'quality_score': self.state.quality_score,
            'performance_score': self.state.performance_score
        }

    async def _execute_optimization_loop(self, **kwargs) -> Dict[str, Any]:
        """Execute optimization loop with quality monitoring"""
        optimization_results = []
        patience_counter = 0
        best_quality = 0.0

        for iteration in range(self.config.max_optimization_iterations):
            logger.info(f"Optimization iteration {iteration + 1}/{self.config.max_optimization_iterations}")

            # Run neural optimizer
            optimizer_result = await self._run_agent('neural_optimizer', self.model, **kwargs)

            # Monitor quality
            quality_result = await self._run_agent('quality_monitor', self.model)
            current_quality = quality_result.get('quality_score', 0.0)

            # Profile performance
            if iteration % 5 == 0:  # Profile every 5 iterations
                perf_result = await self._run_agent('performance_profiler', self.model)
                current_performance = perf_result.get('throughput_samples_per_sec', 0.0)
            else:
                current_performance = self.state.performance_score

            iteration_result = {
                'iteration': iteration,
                'quality_score': current_quality,
                'performance_score': current_performance,
                'optimizer_result': optimizer_result
            }
            optimization_results.append(iteration_result)

            # Update state
            self.state.quality_score = current_quality
            self.state.performance_score = current_performance

            # Check for improvement
            if current_quality > best_quality:
                best_quality = current_quality
                patience_counter = 0

                # Create checkpoint for best model
                if self.config.enable_checkpointing:
                    self._create_checkpoint(f'best_iteration_{iteration}')
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at iteration {iteration + 1}")
                break

            # Quality threshold check
            if current_quality < self.config.quality_threshold:
                logger.warning(f"Quality below threshold: {current_quality} < {self.config.quality_threshold}")
                if self.config.rollback_on_failure:
                    await self._rollback_to_checkpoint()
                    break

        return {
            'iterations_completed': len(optimization_results),
            'best_quality_score': best_quality,
            'final_quality_score': self.state.quality_score,
            'final_performance_score': self.state.performance_score,
            'iteration_results': optimization_results
        }

    async def _phase_acceleration(self) -> Dict[str, Any]:
        """Apply model acceleration optimizations"""
        acceleration_result = await self._run_agent('inference_accelerator', self.model)

        # Update model with accelerated version
        if 'optimized_model' in acceleration_result:
            self.model = acceleration_result['optimized_model']

        # Verify quality after acceleration
        quality_result = await self._run_agent('quality_monitor', self.model)

        return {
            'phase': 'acceleration',
            'status': 'completed',
            'acceleration_result': acceleration_result,
            'post_acceleration_quality': quality_result.get('quality_score', 0.0)
        }

    async def _execute_validation_phase(self) -> Dict[str, Any]:
        """Execute validation and testing phase"""
        # Run parallel validation
        if self.config.enable_parallel_execution:
            validation_tasks = [
                self._run_agent('deployment_validator', self.model),
                self._run_agent('integration_tester', self.model),
                self._run_agent('performance_profiler', self.model)
            ]
            validation_results = await asyncio.gather(*validation_tasks)
        else:
            validation_results = []
            validation_results.append(await self._run_agent('deployment_validator', self.model))
            validation_results.append(await self._run_agent('integration_tester', self.model))
            validation_results.append(await self._run_agent('performance_profiler', self.model))

        return {
            'phase': 'validation',
            'status': 'completed',
            'validation_results': validation_results
        }

    async def _phase_completion_audit(self) -> Dict[str, Any]:
        """Run completion audit"""
        audit_result = await self._run_agent('completion_auditor', self.model)

        return {
            'phase': 'completion_audit',
            'status': 'completed',
            'audit_result': audit_result
        }

    async def _phase_finalization(self) -> Dict[str, Any]:
        """Finalize baking process"""
        # Export results if configured
        if self.config.export_results:
            export_path = Path("baking_results.json")
            with open(export_path, 'w') as f:
                json.dump(asdict(self.state), f, indent=2, default=str)

        # Final state synchronization
        await self._run_agent('state_synchronizer', self.model)

        return {
            'phase': 'finalization',
            'status': 'completed',
            'final_state': asdict(self.state)
        }

    async def _run_agent(self, agent_id: str, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """Run a specific agent and track results"""
        if agent_id not in self.agents:
            raise ValueError(f"Unknown agent: {agent_id}")

        agent = self.agents[agent_id]
        self.state.active_agents.append(agent_id)

        # Send command message
        command_message = AgentMessage(
            message_id=f"cmd_{agent_id}_{int(time.time())}",
            agent_id="orchestrator",
            message_type=MessageType.COMMAND,
            timestamp=time.time(),
            data={'command': 'run', 'target_agent': agent_id, 'model_info': 'model_provided'}
        )
        self.message_bus.publish(command_message)

        # Update state synchronizer
        if agent_id in self.agents and hasattr(self.agents['state_synchronizer'], 'update_state'):
            self.agents['state_synchronizer'].update_state(agent_id, SyncAgentState.PROCESSING)

        start_time = time.time()
        memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        try:
            result = await agent.run(model, **kwargs)
            execution_time = time.time() - start_time
            memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            agent_result = AgentResult(
                agent_id=agent_id,
                status=AgentStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                memory_usage=(memory_after - memory_before) / 1024 / 1024,  # MB
                error_message=None,
                timestamp=datetime.now()
            )

            # Send completion status message
            status_message = AgentMessage(
                message_id=f"status_{agent_id}_{int(time.time())}",
                agent_id=agent_id,
                message_type=MessageType.STATUS,
                timestamp=time.time(),
                data={'status': 'completed', 'execution_time': execution_time, 'result_summary': str(result)[:100]}
            )
            self.message_bus.publish(status_message)

            # Update state synchronizer
            if hasattr(self.agents['state_synchronizer'], 'update_state'):
                self.agents['state_synchronizer'].update_state(agent_id, SyncAgentState.COMPLETED, progress=100.0)

            self.state.completed_agents.append(agent_id)
            logger.info(f"Agent {agent_id} completed in {execution_time:.2f}s")

        except Exception as e:
            agent_result = AgentResult(
                agent_id=agent_id,
                status=AgentStatus.FAILED,
                result={},
                execution_time=time.time() - start_time,
                memory_usage=0.0,
                error_message=str(e),
                timestamp=datetime.now()
            )

            # Send error message
            error_message = AgentMessage(
                message_id=f"error_{agent_id}_{int(time.time())}",
                agent_id=agent_id,
                message_type=MessageType.ERROR,
                timestamp=time.time(),
                data={'error': str(e), 'execution_time': time.time() - start_time}
            )
            self.message_bus.publish(error_message)

            # Update state synchronizer
            if hasattr(self.agents['state_synchronizer'], 'update_state'):
                self.agents['state_synchronizer'].update_state(agent_id, SyncAgentState.ERROR, metadata={'error': str(e)})

            self.state.failed_agents.append(agent_id)
            logger.error(f"Agent {agent_id} failed: {e}")

        finally:
            if agent_id in self.state.active_agents:
                self.state.active_agents.remove(agent_id)

        self.state.agent_results[agent_id] = agent_result
        self.state.last_update = datetime.now()

        return agent_result.result

    def _create_model_copy(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of the model"""
        try:
            model_copy = type(model)()
            model_copy.load_state_dict(model.state_dict())
            return model_copy
        except Exception:
            # Fallback: return reference (not ideal but better than failure)
            return model

    def _create_checkpoint(self, checkpoint_name: str):
        """Create a model checkpoint"""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'orchestration_state': asdict(self.state),
                'timestamp': datetime.now()
            }
            self.checkpoints[checkpoint_name] = checkpoint
            logger.info(f"Checkpoint '{checkpoint_name}' created")
        except Exception as e:
            logger.warning(f"Failed to create checkpoint: {e}")

    async def _rollback_to_checkpoint(self, checkpoint_name: str = None):
        """Rollback to a previous checkpoint"""
        try:
            if checkpoint_name is None:
                # Find the most recent checkpoint
                if not self.checkpoints:
                    logger.warning("No checkpoints available for rollback")
                    return

                checkpoint_name = max(
                    self.checkpoints.keys(),
                    key=lambda x: self.checkpoints[x]['timestamp']
                )

            checkpoint = self.checkpoints[checkpoint_name]
            self.model.load_state_dict(checkpoint['model_state_dict'])

            logger.info(f"Rolled back to checkpoint '{checkpoint_name}'")

        except Exception as e:
            logger.error(f"Rollback failed: {e}")

    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get current orchestration status"""
        return {
            'current_phase': self.state.current_phase.value,
            'phase_progress': self.state.phase_progress,
            'overall_progress': self.state.overall_progress,
            'active_agents': self.state.active_agents,
            'completed_agents': self.state.completed_agents,
            'failed_agents': self.state.failed_agents,
            'quality_score': self.state.quality_score,
            'performance_score': self.state.performance_score,
            'last_update': self.state.last_update.isoformat()
        }


def create_default_orchestration_config() -> OrchestrationConfig:
    """Create default orchestration configuration"""
    return OrchestrationConfig(
        enable_parallel_execution=True,
        quality_threshold=0.7,
        performance_threshold=10.0,
        max_optimization_iterations=50,
        early_stopping_patience=5,
        enable_checkpointing=True,
        checkpoint_frequency=10,
        rollback_on_failure=True,
        detailed_logging=True,
        export_results=True
    )


# Agent Integration Interface
class BakingOrchestratorAgent:
    """Agent wrapper for baking orchestrator"""

    def __init__(self, config: Optional[OrchestrationConfig] = None):
        self.config = config or create_default_orchestration_config()
        self.orchestrator = BakingOrchestrator(self.config)
        self.agent_id = "baking_orchestrator"
        self.status = "idle"

    async def run(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """Run orchestration agent"""
        self.status = "running"

        try:
            result = await self.orchestrator.orchestrate_baking(model, **kwargs)
            self.status = "completed" if result['orchestration_successful'] else "failed"
            return result

        except Exception as e:
            self.status = "failed"
            logger.error(f"Baking orchestrator failed: {e}")
            return {'error': str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'agent_id': self.agent_id,
            'status': self.status,
            'orchestration_status': self.orchestrator.get_orchestration_status()
        }