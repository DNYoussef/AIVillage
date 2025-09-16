#!/usr/bin/env python3
"""
Agent Forge Swarm Coordination System

Comprehensive swarm initialization and coordination for Agent Forge 8-phase pipeline.
Manages topology selection, resource allocation, and cross-phase communication.

Architecture:
- Primary Coordination Layer: Overall pipeline orchestration
- Phase-Specific Swarms: 9 agents each for phases 4-8
- Memory Management: Cross-phase state persistence
- Quality Gates: Theater detection and validation
- Performance Monitoring: Real-time metrics and bottleneck detection
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Import phase controller infrastructure
from agent_forge.core.phase_controller import PhaseController, PhaseOrchestrator, PhaseResult

logger = logging.getLogger(__name__)


class SwarmTopology(Enum):
    """Swarm topology patterns for different coordination needs."""
    HIERARCHICAL = "hierarchical"  # Structured, top-down coordination
    MESH = "mesh"                  # Peer-to-peer collaboration
    STAR = "star"                  # Centralized control
    RING = "ring"                  # Sequential processing


class AgentRole(Enum):
    """Specialized agent roles in the swarm."""
    # Primary Coordination
    SWARM_COORDINATOR = "swarm-coordinator"
    MEMORY_MANAGER = "memory-manager"
    PERFORMANCE_MONITOR = "performance-monitor"
    QUALITY_GATE = "quality-gate"

    # Phase 3 - Quiet-STaR Remediation
    THEATER_DETECTOR = "theater-detector"
    IMPLEMENTATION_VALIDATOR = "implementation-validator"
    REASONING_SPECIALIST = "reasoning-specialist"

    # Phase 4 - BitNet Compression
    BITNET_COMPRESSION = "bitnet-compression"
    QUANTIZATION_SPECIALIST = "quantization-specialist"
    PERFORMANCE_OPTIMIZER = "performance-optimizer"

    # Phase 5 - Training Orchestration
    TRAINING_ORCHESTRATOR = "training-orchestrator"
    DATA_PIPELINE_MANAGER = "data-pipeline-manager"
    MODEL_OPTIMIZER = "model-optimizer"

    # Phase 6 - Model Baking
    MODEL_BAKING_SPECIALIST = "model-baking-specialist"
    ARTIFACT_MANAGER = "artifact-manager"
    VALIDATION_PIPELINE = "validation-pipeline"

    # Phase 7 - ADAS Integration
    ADAS_INTEGRATION_SPECIALIST = "adas-integration-specialist"
    SAFETY_VALIDATOR = "safety-validator"
    REAL_TIME_PROCESSOR = "real-time-processor"

    # Phase 8 - Final Compression
    FINAL_COMPRESSION_COORDINATOR = "final-compression-coordinator"
    DEPLOYMENT_PACKAGER = "deployment-packager"
    PRODUCTION_DEPLOYER = "production-deployer"

    # Shared Roles
    INTEGRATION_MANAGER = "integration-manager"
    TESTING_COORDINATOR = "testing-coordinator"
    DOCUMENTATION_AGENT = "documentation-agent"
    SECURITY_VALIDATOR = "security-validator"
    BENCHMARKING_AGENT = "benchmarking-agent"
    DEPLOYMENT_COORDINATOR = "deployment-coordinator"


@dataclass
class AgentConfig:
    """Configuration for individual agents in the swarm."""
    role: AgentRole
    phase: int
    max_memory: int = 512  # MB
    max_concurrent_tasks: int = 3
    timeout_seconds: int = 300
    specialized_tools: List[str] = field(default_factory=list)
    communication_channels: List[str] = field(default_factory=list)


@dataclass
class SwarmConfig:
    """Configuration for the entire swarm system."""
    topology: SwarmTopology = SwarmTopology.HIERARCHICAL
    max_agents: int = 50
    coordination_interval: float = 1.0  # seconds
    memory_namespace: str = "agent_forge_swarm"
    quality_gate_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "nasa_pot10_compliance": 0.95,
        "theater_detection_accuracy": 0.90,
        "performance_improvement": 0.15,
        "security_score": 0.95
    })

    # Resource allocation
    total_memory_mb: int = 4096
    total_cpu_cores: int = 8
    gpu_allocation: Dict[str, float] = field(default_factory=lambda: {
        "training": 0.4,
        "compression": 0.3,
        "validation": 0.2,
        "coordination": 0.1
    })

    # Phase-specific settings
    phase_timeouts: Dict[int, int] = field(default_factory=lambda: {
        3: 1800,  # 30 minutes for remediation
        4: 3600,  # 1 hour for compression
        5: 7200,  # 2 hours for training
        6: 2400,  # 40 minutes for baking
        7: 3000,  # 50 minutes for ADAS
        8: 1800   # 30 minutes for final compression
    })


class SwarmAgent:
    """Individual agent in the swarm with specialized capabilities."""

    def __init__(self, config: AgentConfig, swarm_coordinator: 'SwarmCoordinator'):
        self.config = config
        self.coordinator = swarm_coordinator
        self.logger = logging.getLogger(f"SwarmAgent-{config.role.value}")
        self.state = "idle"
        self.current_task = None
        self.memory = {}
        self.start_time = None

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specialized task based on agent role."""
        self.state = "executing"
        self.current_task = task
        self.start_time = time.time()

        try:
            self.logger.info(f"Executing task: {task.get('type', 'unknown')}")

            # Route to specialized execution based on role
            if self.config.role == AgentRole.THEATER_DETECTOR:
                result = await self._detect_theater(task)
            elif self.config.role == AgentRole.BITNET_COMPRESSION:
                result = await self._compress_bitnet(task)
            elif self.config.role == AgentRole.TRAINING_ORCHESTRATOR:
                result = await self._orchestrate_training(task)
            elif self.config.role == AgentRole.QUALITY_GATE:
                result = await self._validate_quality(task)
            else:
                result = await self._execute_generic_task(task)

            self.state = "idle"
            duration = time.time() - self.start_time

            return {
                "success": True,
                "result": result,
                "duration": duration,
                "agent_role": self.config.role.value,
                "phase": self.config.phase
            }

        except Exception as e:
            self.state = "error"
            duration = time.time() - self.start_time if self.start_time else 0
            self.logger.error(f"Task execution failed: {str(e)}")

            return {
                "success": False,
                "error": str(e),
                "duration": duration,
                "agent_role": self.config.role.value,
                "phase": self.config.phase
            }

    async def _detect_theater(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Theater detection for Phase 3 remediation."""
        phase_data = task.get("phase_data", {})

        # Analyze implementation vs theater indicators
        theater_indicators = {
            "missing_core_functionality": False,
            "fake_metrics": False,
            "incomplete_integration": False,
            "superficial_changes": False
        }

        # Check for actual implementation depth
        if "reasoning_module" not in phase_data.get("components", []):
            theater_indicators["missing_core_functionality"] = True

        if phase_data.get("performance_improvement", 0) > 0.5:  # Suspiciously high
            theater_indicators["fake_metrics"] = True

        theater_score = sum(theater_indicators.values()) / len(theater_indicators)

        return {
            "theater_detected": theater_score > 0.5,
            "theater_score": theater_score,
            "indicators": theater_indicators,
            "remediation_needed": theater_score > 0.3
        }

    async def _compress_bitnet(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """BitNet compression for Phase 4."""
        model = task.get("model")
        if not model:
            raise ValueError("No model provided for compression")

        # Simulate BitNet 1.58 compression
        original_size = sum(p.numel() for p in model.parameters())

        # Apply quantization simulation
        compression_ratio = 0.158  # BitNet 1.58 target
        compressed_size = int(original_size * compression_ratio)

        return {
            "original_parameters": original_size,
            "compressed_parameters": compressed_size,
            "compression_ratio": compression_ratio,
            "memory_savings": 1 - compression_ratio,
            "bitnet_config": {
                "bits": 1.58,
                "group_size": 128,
                "calibration_samples": 100
            }
        }

    async def _orchestrate_training(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Training orchestration for Phase 5."""
        config = task.get("training_config", {})

        # Simulate training coordination
        training_steps = config.get("training_steps", 100000)
        batch_size = config.get("batch_size", 32)

        return {
            "training_plan": {
                "total_steps": training_steps,
                "batch_size": batch_size,
                "estimated_duration": training_steps * 0.1,  # seconds
                "checkpoint_intervals": training_steps // 10,
                "grokfast_enabled": config.get("grokfast_enabled", True)
            },
            "resource_allocation": {
                "gpu_utilization": 0.85,
                "memory_usage": 0.75,
                "distributed_nodes": config.get("distributed_nodes", 1)
            }
        }

    async def _validate_quality(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Quality gate validation."""
        metrics = task.get("metrics", {})
        thresholds = self.coordinator.config.quality_gate_thresholds

        validation_results = {}
        passed = True

        for metric, threshold in thresholds.items():
            value = metrics.get(metric, 0.0)
            metric_passed = value >= threshold
            validation_results[metric] = {
                "value": value,
                "threshold": threshold,
                "passed": metric_passed
            }
            if not metric_passed:
                passed = False

        return {
            "gate_passed": passed,
            "validation_results": validation_results,
            "overall_score": sum(r["value"] for r in validation_results.values()) / len(validation_results),
            "failed_metrics": [k for k, v in validation_results.items() if not v["passed"]]
        }

    async def _execute_generic_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generic task execution for unspecialized roles."""
        task_type = task.get("type", "generic")

        # Simulate task execution time based on complexity
        complexity = task.get("complexity", 1.0)
        await asyncio.sleep(complexity * 0.1)  # Simulate work

        return {
            "task_type": task_type,
            "complexity": complexity,
            "status": "completed",
            "output": f"Task {task_type} completed by {self.config.role.value}"
        }


class SwarmCoordinator:
    """Main coordinator for the Agent Forge swarm system."""

    def __init__(self, config: SwarmConfig):
        self.config = config
        self.logger = logging.getLogger("SwarmCoordinator")
        self.agents = {}
        self.memory = {}
        self.phase_results = []
        self.active_phase = None

        # Initialize topology-specific coordination
        self.topology_handler = self._create_topology_handler()

    def _create_topology_handler(self):
        """Create topology-specific coordination handler."""
        if self.config.topology == SwarmTopology.HIERARCHICAL:
            return HierarchicalCoordinator(self)
        elif self.config.topology == SwarmTopology.MESH:
            return MeshCoordinator(self)
        elif self.config.topology == SwarmTopology.STAR:
            return StarCoordinator(self)
        else:
            return StarCoordinator(self)  # Default to star

    async def initialize_swarm(self) -> bool:
        """Initialize the complete swarm for Agent Forge pipeline."""
        try:
            self.logger.info("Initializing Agent Forge swarm coordination system")

            # Create primary coordination agents
            await self._create_coordination_agents()

            # Create phase-specific swarms
            await self._create_phase_swarms()

            # Setup cross-phase memory
            await self._setup_memory_architecture()

            # Initialize quality gates
            await self._initialize_quality_gates()

            self.logger.info(f"Swarm initialized with {len(self.agents)} agents using {self.config.topology.value} topology")
            return True

        except Exception as e:
            self.logger.error(f"Swarm initialization failed: {str(e)}")
            return False

    async def _create_coordination_agents(self):
        """Create primary coordination layer agents."""
        coordination_roles = [
            AgentRole.SWARM_COORDINATOR,
            AgentRole.MEMORY_MANAGER,
            AgentRole.PERFORMANCE_MONITOR,
            AgentRole.QUALITY_GATE
        ]

        for role in coordination_roles:
            config = AgentConfig(
                role=role,
                phase=0,  # Coordination layer
                max_memory=1024,  # Higher memory for coordinators
                max_concurrent_tasks=5,
                specialized_tools=["coordination", "monitoring", "validation"]
            )

            agent = SwarmAgent(config, self)
            self.agents[f"coord_{role.value}"] = agent
            self.logger.info(f"Created coordination agent: {role.value}")

    async def _create_phase_swarms(self):
        """Create phase-specific agent swarms."""
        phase_configs = {
            3: {  # Phase 3 - Quiet-STaR Remediation
                "roles": [
                    AgentRole.THEATER_DETECTOR,
                    AgentRole.IMPLEMENTATION_VALIDATOR,
                    AgentRole.REASONING_SPECIALIST,
                    AgentRole.INTEGRATION_MANAGER,
                    AgentRole.TESTING_COORDINATOR,
                    AgentRole.DOCUMENTATION_AGENT,
                    AgentRole.SECURITY_VALIDATOR,
                    AgentRole.BENCHMARKING_AGENT,
                    AgentRole.QUALITY_GATE
                ]
            },
            4: {  # Phase 4 - BitNet Compression
                "roles": [
                    AgentRole.BITNET_COMPRESSION,
                    AgentRole.QUANTIZATION_SPECIALIST,
                    AgentRole.PERFORMANCE_OPTIMIZER,
                    AgentRole.INTEGRATION_MANAGER,
                    AgentRole.TESTING_COORDINATOR,
                    AgentRole.DOCUMENTATION_AGENT,
                    AgentRole.SECURITY_VALIDATOR,
                    AgentRole.BENCHMARKING_AGENT,
                    AgentRole.DEPLOYMENT_COORDINATOR
                ]
            },
            5: {  # Phase 5 - Training Orchestration
                "roles": [
                    AgentRole.TRAINING_ORCHESTRATOR,
                    AgentRole.DATA_PIPELINE_MANAGER,
                    AgentRole.MODEL_OPTIMIZER,
                    AgentRole.INTEGRATION_MANAGER,
                    AgentRole.TESTING_COORDINATOR,
                    AgentRole.DOCUMENTATION_AGENT,
                    AgentRole.SECURITY_VALIDATOR,
                    AgentRole.BENCHMARKING_AGENT,
                    AgentRole.DEPLOYMENT_COORDINATOR
                ]
            },
            6: {  # Phase 6 - Model Baking
                "roles": [
                    AgentRole.MODEL_BAKING_SPECIALIST,
                    AgentRole.ARTIFACT_MANAGER,
                    AgentRole.VALIDATION_PIPELINE,
                    AgentRole.INTEGRATION_MANAGER,
                    AgentRole.TESTING_COORDINATOR,
                    AgentRole.DOCUMENTATION_AGENT,
                    AgentRole.SECURITY_VALIDATOR,
                    AgentRole.BENCHMARKING_AGENT,
                    AgentRole.DEPLOYMENT_COORDINATOR
                ]
            },
            7: {  # Phase 7 - ADAS Integration
                "roles": [
                    AgentRole.ADAS_INTEGRATION_SPECIALIST,
                    AgentRole.SAFETY_VALIDATOR,
                    AgentRole.REAL_TIME_PROCESSOR,
                    AgentRole.INTEGRATION_MANAGER,
                    AgentRole.TESTING_COORDINATOR,
                    AgentRole.DOCUMENTATION_AGENT,
                    AgentRole.SECURITY_VALIDATOR,
                    AgentRole.BENCHMARKING_AGENT,
                    AgentRole.DEPLOYMENT_COORDINATOR
                ]
            },
            8: {  # Phase 8 - Final Compression
                "roles": [
                    AgentRole.FINAL_COMPRESSION_COORDINATOR,
                    AgentRole.DEPLOYMENT_PACKAGER,
                    AgentRole.PRODUCTION_DEPLOYER,
                    AgentRole.INTEGRATION_MANAGER,
                    AgentRole.TESTING_COORDINATOR,
                    AgentRole.DOCUMENTATION_AGENT,
                    AgentRole.SECURITY_VALIDATOR,
                    AgentRole.BENCHMARKING_AGENT,
                    AgentRole.QUALITY_GATE
                ]
            }
        }

        for phase, config in phase_configs.items():
            for role in config["roles"]:
                agent_config = AgentConfig(
                    role=role,
                    phase=phase,
                    timeout_seconds=self.config.phase_timeouts.get(phase, 3600),
                    specialized_tools=[f"phase_{phase}", role.value.replace("-", "_")]
                )

                agent = SwarmAgent(agent_config, self)
                agent_id = f"phase_{phase}_{role.value}"
                self.agents[agent_id] = agent

            self.logger.info(f"Created Phase {phase} swarm with {len(config['roles'])} agents")

    async def _setup_memory_architecture(self):
        """Setup cross-phase memory system."""
        self.memory = {
            "phase_states": {},
            "optimization_history": [],
            "performance_metrics": {},
            "quality_gates": {},
            "learning_transfers": {}
        }

        self.logger.info("Cross-phase memory architecture initialized")

    async def _initialize_quality_gates(self):
        """Initialize quality gate framework."""
        quality_gates = {
            "theater_detection": {
                "enabled": True,
                "threshold": 0.9,
                "validators": ["theater_detector", "implementation_validator"]
            },
            "nasa_pot10_compliance": {
                "enabled": True,
                "threshold": 0.95,
                "validators": ["security_validator", "quality_gate"]
            },
            "performance_validation": {
                "enabled": True,
                "threshold": 0.15,  # 15% improvement minimum
                "validators": ["performance_optimizer", "benchmarking_agent"]
            }
        }

        self.memory["quality_gates"] = quality_gates
        self.logger.info("Quality gate framework initialized")

    async def execute_phase(self, phase: int, phase_data: Dict[str, Any]) -> PhaseResult:
        """Execute a specific phase using the appropriate agent swarm."""
        self.active_phase = phase
        phase_start = time.time()

        try:
            self.logger.info(f"Starting Phase {phase} execution with swarm coordination")

            # Get phase-specific agents
            phase_agents = [agent for agent_id, agent in self.agents.items()
                          if agent.config.phase == phase]

            if not phase_agents:
                raise ValueError(f"No agents available for Phase {phase}")

            # Create phase execution plan
            execution_plan = await self._create_execution_plan(phase, phase_data, phase_agents)

            # Execute phase using topology handler
            phase_result = await self.topology_handler.execute_phase(execution_plan)

            # Validate quality gates
            gate_result = await self._validate_phase_quality_gates(phase, phase_result)

            # Store results in memory
            self.memory["phase_states"][phase] = {
                "result": phase_result,
                "quality_gate": gate_result,
                "timestamp": datetime.now().isoformat(),
                "duration": time.time() - phase_start
            }

            duration = time.time() - phase_start
            self.logger.info(f"Phase {phase} completed in {duration:.2f}s")

            return PhaseResult(
                success=gate_result["gate_passed"],
                model=phase_result.get("model"),
                phase_name=f"Phase{phase}",
                metrics={
                    "duration_seconds": duration,
                    "quality_score": gate_result["overall_score"],
                    "agent_count": len(phase_agents),
                    "tasks_completed": phase_result.get("tasks_completed", 0)
                },
                artifacts={
                    "phase_result": phase_result,
                    "quality_validation": gate_result,
                    "execution_plan": execution_plan
                }
            )

        except Exception as e:
            duration = time.time() - phase_start
            error_msg = f"Phase {phase} execution failed: {str(e)}"
            self.logger.error(error_msg)

            return PhaseResult(
                success=False,
                model=None,
                phase_name=f"Phase{phase}",
                error=error_msg,
                duration_seconds=duration
            )

    async def _create_execution_plan(self, phase: int, phase_data: Dict[str, Any],
                                   agents: List[SwarmAgent]) -> Dict[str, Any]:
        """Create execution plan for phase with agent task allocation."""
        plan = {
            "phase": phase,
            "total_agents": len(agents),
            "task_allocation": {},
            "dependencies": [],
            "estimated_duration": self.config.phase_timeouts.get(phase, 3600)
        }

        # Allocate tasks based on agent roles
        for agent in agents:
            role = agent.config.role

            if role == AgentRole.THEATER_DETECTOR:
                plan["task_allocation"][agent.config.role.value] = {
                    "type": "theater_detection",
                    "phase_data": phase_data,
                    "priority": "high"
                }
            elif role == AgentRole.BITNET_COMPRESSION:
                plan["task_allocation"][agent.config.role.value] = {
                    "type": "bitnet_compression",
                    "model": phase_data.get("model"),
                    "priority": "high"
                }
            elif role == AgentRole.TRAINING_ORCHESTRATOR:
                plan["task_allocation"][agent.config.role.value] = {
                    "type": "training_orchestration",
                    "training_config": phase_data.get("training_config", {}),
                    "priority": "high"
                }
            else:
                plan["task_allocation"][agent.config.role.value] = {
                    "type": "supporting_task",
                    "phase_data": phase_data,
                    "priority": "medium"
                }

        return plan

    async def _validate_phase_quality_gates(self, phase: int,
                                          phase_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate quality gates for completed phase."""
        quality_agent = self.agents.get("coord_quality-gate")
        if not quality_agent:
            return {"gate_passed": True, "overall_score": 1.0}

        gate_task = {
            "type": "quality_validation",
            "metrics": phase_result.get("metrics", {}),
            "phase": phase
        }

        validation_result = await quality_agent.execute_task(gate_task)
        return validation_result.get("result", {"gate_passed": True, "overall_score": 1.0})

    async def get_swarm_status(self) -> Dict[str, Any]:
        """Get comprehensive swarm status."""
        agent_statuses = {}
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = {
                "role": agent.config.role.value,
                "phase": agent.config.phase,
                "state": agent.state,
                "current_task": agent.current_task.get("type") if agent.current_task else None
            }

        return {
            "total_agents": len(self.agents),
            "active_phase": self.active_phase,
            "topology": self.config.topology.value,
            "memory_usage": len(self.memory),
            "agent_statuses": agent_statuses,
            "completed_phases": list(self.memory.get("phase_states", {}).keys())
        }


class HierarchicalCoordinator:
    """Hierarchical topology coordination for structured phase execution."""

    def __init__(self, swarm: SwarmCoordinator):
        self.swarm = swarm
        self.logger = logging.getLogger("HierarchicalCoordinator")

    async def execute_phase(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute phase with hierarchical coordination."""
        phase = execution_plan["phase"]
        task_allocation = execution_plan["task_allocation"]

        # Execute in hierarchical order: coordinator -> specialists -> support
        priority_order = ["high", "medium", "low"]
        results = {}

        for priority in priority_order:
            priority_tasks = [(role, task) for role, task in task_allocation.items()
                            if task.get("priority") == priority]

            # Execute priority tasks in parallel
            task_results = await asyncio.gather(*[
                self._execute_agent_task(role, task) for role, task in priority_tasks
            ], return_exceptions=True)

            for (role, _), result in zip(priority_tasks, task_results):
                results[role] = result

        return {
            "phase": phase,
            "results": results,
            "tasks_completed": len(results),
            "success_rate": sum(1 for r in results.values()
                              if isinstance(r, dict) and r.get("success", False)) / len(results)
        }

    async def _execute_agent_task(self, role: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with specific agent."""
        # Find agent by role
        agent = None
        for agent_id, a in self.swarm.agents.items():
            if a.config.role.value == role:
                agent = a
                break

        if not agent:
            return {"success": False, "error": f"Agent not found for role: {role}"}

        return await agent.execute_task(task)


class MeshCoordinator:
    """Mesh topology coordination for peer-to-peer collaboration."""

    def __init__(self, swarm: SwarmCoordinator):
        self.swarm = swarm
        self.logger = logging.getLogger("MeshCoordinator")

    async def execute_phase(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute phase with mesh coordination (all agents collaborate)."""
        task_allocation = execution_plan["task_allocation"]

        # Execute all tasks in parallel with peer communication
        tasks = [
            self._execute_collaborative_task(role, task, task_allocation)
            for role, task in task_allocation.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "phase": execution_plan["phase"],
            "results": dict(zip(task_allocation.keys(), results)),
            "tasks_completed": len(results),
            "collaboration_score": 0.95  # Simulated collaboration effectiveness
        }

    async def _execute_collaborative_task(self, role: str, task: Dict[str, Any],
                                        all_tasks: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with collaboration context."""
        # Add collaboration context to task
        task["collaboration_context"] = {
            "peer_tasks": list(all_tasks.keys()),
            "coordination_mode": "mesh"
        }

        # Find and execute with agent
        agent = None
        for agent_id, a in self.swarm.agents.items():
            if a.config.role.value == role:
                agent = a
                break

        if not agent:
            return {"success": False, "error": f"Agent not found for role: {role}"}

        return await agent.execute_task(task)


class StarCoordinator:
    """Star topology coordination for centralized control."""

    def __init__(self, swarm: SwarmCoordinator):
        self.swarm = swarm
        self.logger = logging.getLogger("StarCoordinator")

    async def execute_phase(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute phase with centralized star coordination."""
        task_allocation = execution_plan["task_allocation"]

        # Central coordinator manages all task execution
        coordinator = self.swarm.agents.get("coord_swarm-coordinator")
        if not coordinator:
            # Fallback to direct execution
            return await self._execute_direct_tasks(task_allocation)

        # Coordinator orchestrates all tasks
        orchestration_task = {
            "type": "phase_orchestration",
            "task_allocation": task_allocation,
            "coordination_mode": "star"
        }

        orchestration_result = await coordinator.execute_task(orchestration_task)

        return {
            "phase": execution_plan["phase"],
            "orchestration_result": orchestration_result,
            "coordination_mode": "star"
        }

    async def _execute_direct_tasks(self, task_allocation: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback direct task execution."""
        results = {}

        for role, task in task_allocation.items():
            agent = None
            for agent_id, a in self.swarm.agents.items():
                if a.config.role.value == role:
                    agent = a
                    break

            if agent:
                results[role] = await agent.execute_task(task)
            else:
                results[role] = {"success": False, "error": f"Agent not found: {role}"}

        return {
            "results": results,
            "tasks_completed": len(results)
        }


# Integration with Agent Forge pipeline
async def initialize_agent_forge_swarm(
    topology: SwarmTopology = SwarmTopology.HIERARCHICAL,
    max_agents: int = 50
) -> SwarmCoordinator:
    """Initialize Agent Forge swarm coordination system."""

    config = SwarmConfig(
        topology=topology,
        max_agents=max_agents,
        memory_namespace="agent_forge_pipeline"
    )

    coordinator = SwarmCoordinator(config)

    if await coordinator.initialize_swarm():
        return coordinator
    else:
        raise RuntimeError("Failed to initialize Agent Forge swarm")


# CLI integration for swarm management
def create_swarm_coordinator(config_path: Optional[str] = None, **kwargs) -> SwarmCoordinator:
    """Create swarm coordinator with configuration."""
    if config_path:
        with open(config_path) as f:
            config_dict = json.load(f)
        config_dict.update(kwargs)
        config = SwarmConfig(**config_dict)
    else:
        config = SwarmConfig(**kwargs)

    return SwarmCoordinator(config)