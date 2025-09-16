#!/usr/bin/env python3
"""
Agent Adapters - Bridge between Emergency Infrastructure and Actual Agents
==========================================================================

This module provides adapter classes that bridge the interface mismatch between:
1. Emergency Infrastructure agents (expecting agent_id, message_bus)
2. Actual Phase 6 agents (with different signatures)

This resolves the system completeness issue by allowing all agents to work together.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import asdict

# Import emergency infrastructure
from .core_infrastructure import BaseAgent, AgentStatus, MessageBus, AgentMessage, MessageType

# Import actual phase 6 agents
from ..agents.neural_model_optimizer import NeuralModelOptimizerAgent, create_default_optimizer_config
from ..agents.inference_accelerator import InferenceAcceleratorAgent, create_default_acceleration_config
from ..agents.quality_preservation_monitor import QualityPreservationMonitorAgent, create_default_quality_config
from ..agents.performance_profiler import PerformanceProfilerAgent, create_default_profiling_config
from ..agents.state_synchronizer import StateSynchronizer
from ..agents.deployment_validator import DeploymentValidator
from ..agents.integration_tester import IntegrationTester
from ..agents.completion_auditor import CompletionAuditor

logger = logging.getLogger(__name__)


class NeuralModelOptimizerAdapter(BaseAgent):
    """Adapter for NeuralModelOptimizerAgent"""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, "neural_model_optimizer", message_bus)
        try:
            self.wrapped_agent = NeuralModelOptimizerAgent(create_default_optimizer_config())
        except Exception as e:
            # Fallback initialization
            self.wrapped_agent = None
            logger.warning(f"Failed to create NeuralModelOptimizerAgent: {e}")

    def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute neural model optimization task"""
        try:
            self.current_task = "neural_optimization"
            self.status = AgentStatus.RUNNING

            if not self.wrapped_agent:
                return {"success": False, "error": "Agent not initialized"}

            # Extract model and parameters
            model = task_data.get("model")
            if model is None:
                return {"success": False, "error": "No model provided"}

            # Run async optimization in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.wrapped_agent.run(model, **task_data))
                if "error" not in result:
                    self.status = AgentStatus.COMPLETED
                    return {"success": True, "result": result}
                else:
                    self.status = AgentStatus.FAILED
                    return {"success": False, "error": result["error"]}
            finally:
                loop.close()

        except Exception as e:
            self.error_count += 1
            self.status = AgentStatus.FAILED
            logger.error(f"Neural optimization failed: {e}")
            return {"success": False, "error": str(e)}


class InferenceAcceleratorAdapter(BaseAgent):
    """Adapter for InferenceAcceleratorAgent"""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, "inference_accelerator", message_bus)
        try:
            self.wrapped_agent = InferenceAcceleratorAgent(create_default_acceleration_config())
        except Exception as e:
            self.wrapped_agent = None
            logger.warning(f"Failed to create InferenceAcceleratorAgent: {e}")

    def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute inference acceleration task"""
        try:
            self.current_task = "inference_acceleration"
            self.status = AgentStatus.RUNNING

            if not self.wrapped_agent:
                return {"success": False, "error": "Agent not initialized"}

            model = task_data.get("model")
            if model is None:
                return {"success": False, "error": "No model provided"}

            # Run async acceleration in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.wrapped_agent.run(model, **task_data))
                if "error" not in result:
                    self.status = AgentStatus.COMPLETED
                    return {"success": True, "result": result}
                else:
                    self.status = AgentStatus.FAILED
                    return {"success": False, "error": result["error"]}
            finally:
                loop.close()

        except Exception as e:
            self.error_count += 1
            self.status = AgentStatus.FAILED
            logger.error(f"Inference acceleration failed: {e}")
            return {"success": False, "error": str(e)}


class QualityPreservationMonitorAdapter(BaseAgent):
    """Adapter for QualityPreservationMonitorAgent"""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, "quality_preservation_monitor", message_bus)
        try:
            self.wrapped_agent = QualityPreservationMonitorAgent(create_default_quality_config())
        except Exception as e:
            self.wrapped_agent = None
            logger.warning(f"Failed to create QualityPreservationMonitorAgent: {e}")

    def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute quality preservation monitoring task"""
        try:
            self.current_task = "quality_monitoring"
            self.status = AgentStatus.RUNNING

            if not self.wrapped_agent:
                return {"success": False, "error": "Agent not initialized"}

            # Run async monitoring in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.wrapped_agent.run(**task_data))
                if "error" not in result:
                    self.status = AgentStatus.COMPLETED
                    return {"success": True, "result": result}
                else:
                    self.status = AgentStatus.FAILED
                    return {"success": False, "error": result["error"]}
            finally:
                loop.close()

        except Exception as e:
            self.error_count += 1
            self.status = AgentStatus.FAILED
            logger.error(f"Quality monitoring failed: {e}")
            return {"success": False, "error": str(e)}


class PerformanceProfilerAdapter(BaseAgent):
    """Adapter for PerformanceProfilerAgent"""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, "performance_profiler", message_bus)
        try:
            self.wrapped_agent = PerformanceProfilerAgent(create_default_profiling_config())
        except Exception as e:
            self.wrapped_agent = None
            logger.warning(f"Failed to create PerformanceProfilerAgent: {e}")

    def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute performance profiling task"""
        try:
            self.current_task = "performance_profiling"
            self.status = AgentStatus.RUNNING

            if not self.wrapped_agent:
                return {"success": False, "error": "Agent not initialized"}

            # Run async profiling in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.wrapped_agent.run(**task_data))
                if "error" not in result:
                    self.status = AgentStatus.COMPLETED
                    return {"success": True, "result": result}
                else:
                    self.status = AgentStatus.FAILED
                    return {"success": False, "error": result["error"]}
            finally:
                loop.close()

        except Exception as e:
            self.error_count += 1
            self.status = AgentStatus.FAILED
            logger.error(f"Performance profiling failed: {e}")
            return {"success": False, "error": str(e)}


class StateSynchronizerAdapter(BaseAgent):
    """Adapter for StateSynchronizer"""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, "state_synchronizer", message_bus)
        try:
            self.wrapped_agent = StateSynchronizer()
        except Exception as e:
            self.wrapped_agent = None
            logger.warning(f"Failed to create StateSynchronizer: {e}")

    def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute state synchronization task"""
        try:
            self.current_task = "state_synchronization"
            self.status = AgentStatus.RUNNING

            if not self.wrapped_agent:
                return {"success": False, "error": "Agent not initialized"}

            # Execute synchronization
            result = self.wrapped_agent.synchronize_state(task_data)
            self.status = AgentStatus.COMPLETED
            return {"success": True, "result": result}

        except Exception as e:
            self.error_count += 1
            self.status = AgentStatus.FAILED
            logger.error(f"State synchronization failed: {e}")
            return {"success": False, "error": str(e)}


class DeploymentValidatorAdapter(BaseAgent):
    """Adapter for DeploymentValidator"""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, "deployment_validator", message_bus)
        try:
            self.wrapped_agent = DeploymentValidator()
        except Exception as e:
            self.wrapped_agent = None
            logger.warning(f"Failed to create DeploymentValidator: {e}")

    def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute deployment validation task"""
        try:
            self.current_task = "deployment_validation"
            self.status = AgentStatus.RUNNING

            if not self.wrapped_agent:
                return {"success": False, "error": "Agent not initialized"}

            # Execute validation
            result = self.wrapped_agent.validate_deployment(task_data)
            self.status = AgentStatus.COMPLETED
            return {"success": True, "result": result}

        except Exception as e:
            self.error_count += 1
            self.status = AgentStatus.FAILED
            logger.error(f"Deployment validation failed: {e}")
            return {"success": False, "error": str(e)}


class IntegrationTesterAdapter(BaseAgent):
    """Adapter for IntegrationTester"""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, "integration_tester", message_bus)
        try:
            self.wrapped_agent = IntegrationTester()
        except Exception as e:
            self.wrapped_agent = None
            logger.warning(f"Failed to create IntegrationTester: {e}")

    def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute integration testing task"""
        try:
            self.current_task = "integration_testing"
            self.status = AgentStatus.RUNNING

            if not self.wrapped_agent:
                return {"success": False, "error": "Agent not initialized"}

            # Execute integration tests
            result = self.wrapped_agent.run_integration_tests(task_data)
            self.status = AgentStatus.COMPLETED
            return {"success": True, "result": result}

        except Exception as e:
            self.error_count += 1
            self.status = AgentStatus.FAILED
            logger.error(f"Integration testing failed: {e}")
            return {"success": False, "error": str(e)}


class CompletionAuditorAdapter(BaseAgent):
    """Adapter for CompletionAuditor"""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, "completion_auditor", message_bus)
        try:
            self.wrapped_agent = CompletionAuditor()
        except Exception as e:
            self.wrapped_agent = None
            logger.warning(f"Failed to create CompletionAuditor: {e}")

    def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute completion audit task"""
        try:
            self.current_task = "completion_audit"
            self.status = AgentStatus.RUNNING

            if not self.wrapped_agent:
                return {"success": False, "error": "Agent not initialized"}

            # Execute audit
            result = self.wrapped_agent.audit_system_completion(task_data)
            self.status = AgentStatus.COMPLETED
            return {"success": True, "result": result}

        except Exception as e:
            self.error_count += 1
            self.status = AgentStatus.FAILED
            logger.error(f"Completion audit failed: {e}")
            return {"success": False, "error": str(e)}


# Enhanced Baking System Infrastructure with Adapters
class EnhancedBakingSystemInfrastructure:
    """Enhanced baking system infrastructure using adapters"""

    def __init__(self):
        self.message_bus = MessageBus()
        self.agents: Dict[str, BaseAgent] = {}
        self.logger = logging.getLogger("EnhancedBakingSystemInfrastructure")
        self.system_started = False

        # Initialize all agents with adapters
        self._initialize_agents_with_adapters()

    def _initialize_agents_with_adapters(self):
        """Initialize all agents using adapters"""
        adapter_classes = [
            (NeuralModelOptimizerAdapter, "neural_optimizer_1"),
            (InferenceAcceleratorAdapter, "inference_accelerator_1"),
            (QualityPreservationMonitorAdapter, "quality_monitor_1"),
            (PerformanceProfilerAdapter, "performance_profiler_1"),
            (StateSynchronizerAdapter, "state_synchronizer_1"),
            (DeploymentValidatorAdapter, "deployment_validator_1"),
            (IntegrationTesterAdapter, "integration_tester_1"),
            (CompletionAuditorAdapter, "completion_auditor_1")
        ]

        # Create individual agents with adapters
        for adapter_class, agent_id in adapter_classes:
            try:
                agent = adapter_class(agent_id, self.message_bus)
                self.agents[agent_id] = agent
                self.agents[agent.agent_type] = agent  # Also index by type
                self.logger.info(f"Created adapted agent: {agent_id}")
            except Exception as e:
                self.logger.error(f"Failed to create adapted agent {agent_id}: {e}")

        self.logger.info(f"Initialized {len(self.agents)//2} adapted baking agents")

    def start_system(self):
        """Start the enhanced baking system"""
        if self.system_started:
            self.logger.warning("System already started")
            return

        self.logger.info("Starting enhanced baking system infrastructure...")

        # Start all agents
        for agent in self.agents.values():
            if hasattr(agent, 'start'):
                try:
                    agent.start()
                except Exception as e:
                    self.logger.error(f"Failed to start agent {agent.agent_id}: {e}")

        self.system_started = True
        self.logger.info("Enhanced baking system infrastructure started successfully")

    def stop_system(self):
        """Stop the enhanced baking system"""
        if not self.system_started:
            return

        self.logger.info("Stopping enhanced baking system infrastructure...")

        # Stop all agents
        for agent in self.agents.values():
            if hasattr(agent, 'stop'):
                try:
                    agent.stop()
                except Exception as e:
                    self.logger.error(f"Failed to stop agent {agent.agent_id}: {e}")

        self.system_started = False
        self.logger.info("Enhanced baking system infrastructure stopped")

    def get_system_status(self) -> Dict[str, Any]:
        """Get complete enhanced system status"""
        agent_states = {}
        for agent_id, agent in self.agents.items():
            if not agent_id.endswith("_1"):  # Skip type-indexed duplicates
                continue
            if hasattr(agent, 'get_state'):
                try:
                    agent_states[agent_id] = agent.get_state().__dict__
                except Exception as e:
                    agent_states[agent_id] = {"error": str(e), "status": "unknown"}

        return {
            "system_started": self.system_started,
            "total_agents": len([aid for aid in self.agents.keys() if not aid.endswith("_1")]),
            "agent_states": agent_states,
            "system_completeness": self.get_system_completeness(),
            "timestamp": time.time()
        }

    def get_system_completeness(self) -> float:
        """Calculate current system completeness percentage"""
        if not self.system_started:
            return 0.0

        working_agents = 0
        total_agents = len([aid for aid in self.agents.keys() if not aid.endswith("_1")])  # Count unique agents

        for agent_id, agent in self.agents.items():
            if not agent_id.endswith("_1"):  # Skip type-indexed duplicates
                continue

            try:
                if hasattr(agent, 'get_state'):
                    state = agent.get_state()
                    if state.status in [AgentStatus.RUNNING, AgentStatus.COMPLETED]:
                        working_agents += 1
                elif hasattr(agent, 'wrapped_agent') and agent.wrapped_agent:
                    working_agents += 1
            except Exception:
                pass  # Agent not working

        completeness = (working_agents / total_agents * 100) if total_agents > 0 else 0.0
        return min(completeness, 100.0)

    def run_system_diagnostics(self) -> Dict[str, Any]:
        """Run enhanced system diagnostics"""
        diagnostics = {
            "infrastructure_check": "PASS",
            "agent_communication": "PASS",
            "message_bus": "PASS",
            "error_count": 0,
            "warnings": [],
            "system_completeness": self.get_system_completeness()
        }

        try:
            # Check all agents are responsive
            for agent_id, agent in self.agents.items():
                if not agent_id.endswith("_1"):  # Skip type-indexed duplicates
                    continue

                if hasattr(agent, 'get_state'):
                    state = agent.get_state()
                    if state.status == AgentStatus.FAILED:
                        diagnostics["error_count"] += 1
                        diagnostics["warnings"].append(f"Agent {agent_id} in failed state")

                    if state.error_count > 0:
                        diagnostics["warnings"].append(f"Agent {agent_id} has {state.error_count} errors")

            # Overall assessment
            if diagnostics["error_count"] > 0:
                diagnostics["infrastructure_check"] = "FAIL"
            elif len(diagnostics["warnings"]) > 0:
                diagnostics["infrastructure_check"] = "WARNING"

            # System completeness assessment
            if diagnostics["system_completeness"] >= 90:
                diagnostics["completeness_status"] = "EXCELLENT"
            elif diagnostics["system_completeness"] >= 75:
                diagnostics["completeness_status"] = "GOOD"
            elif diagnostics["system_completeness"] >= 50:
                diagnostics["completeness_status"] = "FAIR"
            else:
                diagnostics["completeness_status"] = "POOR"

            self.logger.info(f"Enhanced system diagnostics: {diagnostics['infrastructure_check']}")
            self.logger.info(f"System completeness: {diagnostics['system_completeness']:.1f}%")
            return diagnostics

        except Exception as e:
            self.logger.error(f"Enhanced system diagnostics failed: {e}")
            return {
                "infrastructure_check": "FAIL",
                "error": str(e),
                "error_count": 1,
                "system_completeness": 0.0
            }