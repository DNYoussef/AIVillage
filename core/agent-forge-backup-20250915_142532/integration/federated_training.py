"""
Agent Forge Federated Training Integration

Integrates the unified Agent Forge pipeline with the P2P communication system
and fog compute infrastructure to enable distributed, federated training
of AI agents across multiple devices and nodes.
"""

import asyncio
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

# Import P2P transport system
try:
    from ...p2p.core.message_types import AgentForgeMessage, MessageType
    from ...p2p.core.transport_manager import TransportContext, TransportManager

    P2P_AVAILABLE = True
except ImportError as e:
    logging.warning(f"P2P system not available: {e}")
    P2P_AVAILABLE = False

# Import fog compute system
try:
    from ...edge.core.edge_manager import EdgeManager
    from ...edge.fog_compute.fog_coordinator import FogCoordinator

    FOG_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Fog compute system not available: {e}")
    FOG_AVAILABLE = False

# Import unified pipeline
from ..core.phase_controller import PhaseResult
from ..core.unified_pipeline import UnifiedConfig

# Import HRRM export adapters for fog burst capabilities
try:
    from ...models.hrrm.export_adapters import ConsistencyValidator, FogBurstAdapter

    HRRM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"HRRM export adapters not available: {e}")
    HRRM_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class FederatedTrainingConfig:
    """Configuration for federated Agent Forge training."""

    # P2P Federation settings
    enable_p2p_federation: bool = True
    peer_discovery_timeout: int = 30
    max_peers: int = 10
    communication_protocol: str = "bitchat"  # "bitchat" or "betanet"

    # Fog compute settings
    enable_fog_compute: bool = True
    fog_coordination: bool = True
    fog_task_distribution: bool = True
    min_fog_nodes: int = 2

    # Federated learning parameters
    federated_rounds: int = 10
    min_participants: int = 2
    max_participants: int = 50
    participation_threshold: float = 0.6  # Minimum participation rate

    # Model aggregation settings
    aggregation_method: str = "fedavg"  # "fedavg", "fedprox", "scaffold"
    aggregation_weights: str = "uniform"  # "uniform", "data_size", "compute_power"

    # Communication optimization
    gradient_compression: bool = True
    model_compression: bool = True
    differential_privacy: bool = True
    privacy_epsilon: float = 1.0

    # Resource management
    battery_aware: bool = True
    data_cost_aware: bool = True
    thermal_throttling: bool = True
    resource_allocation: str = "adaptive"  # "adaptive", "fixed", "priority"

    # Checkpoint and recovery
    enable_checkpointing: bool = True
    checkpoint_interval: int = 5  # rounds
    fault_tolerance: bool = True
    max_failures: int = 3

    # HRRM Integration
    hrrm_preset: str = "distributed_inference"  # "local_only", "fog_hybrid", "distributed_inference", "full_sharding"
    hrrm_sharding_strategy: str = "layer_wise"  # "layer_wise", "attention_heads", "pipeline", "hybrid"
    hrrm_fog_burst_enabled: bool = True
    hrrm_parity_validation: bool = True
    hrrm_parity_tolerance: float = 1e-4


class FederatedAgentForge:
    """
    Federated version of the Agent Forge pipeline that coordinates
    distributed training across P2P networks and fog compute nodes.
    """

    def __init__(self, base_config: UnifiedConfig, federated_config: FederatedTrainingConfig):
        self.base_config = base_config
        self.fed_config = federated_config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.transport_manager = None
        self.fog_coordinator = None
        self.edge_manager = None
        self.participants = []
        self.current_round = 0
        self.global_model = None
        self.training_history = []

        # Initialize infrastructure
        self._initialize_infrastructure()

    def _initialize_infrastructure(self):
        """Initialize P2P and fog compute infrastructure."""
        try:
            # Initialize P2P transport if available
            if P2P_AVAILABLE and self.fed_config.enable_p2p_federation:
                self.transport_manager = TransportManager()
                # Configure for federated training context
                transport_context = TransportContext(
                    device_type="agent_forge_coordinator",
                    battery_level=100,  # Assume coordinator has stable power
                    network_type="wifi",
                    data_budget_mb=1000,  # High budget for coordination
                    priority="high",
                )
                self.transport_manager.set_context(transport_context)
                self.logger.info("P2P transport manager initialized")

            # Initialize fog compute if available
            if FOG_AVAILABLE and self.fed_config.enable_fog_compute:
                self.fog_coordinator = FogCoordinator()
                self.edge_manager = EdgeManager()
                self.logger.info("Fog compute infrastructure initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize infrastructure: {e}")
            raise

    async def discover_participants(self) -> list[dict[str, Any]]:
        """Discover available participants for federated training."""
        participants = []

        try:
            # P2P peer discovery
            if self.transport_manager:
                self.logger.info("Starting P2P peer discovery...")
                peers = await self.transport_manager.discover_peers(
                    timeout=self.fed_config.peer_discovery_timeout, max_peers=self.fed_config.max_peers
                )

                for peer in peers:
                    # Query peer capabilities
                    capability_msg = AgentForgeMessage(
                        message_type=MessageType.CAPABILITY_QUERY,
                        payload={"request": "training_capability"},
                        sender_id="coordinator",
                        recipient_id=peer["peer_id"],
                    )

                    response = await self.transport_manager.send_message(capability_msg)
                    if response and response.get("capable", False):
                        participants.append(
                            {
                                "type": "p2p_peer",
                                "peer_id": peer["peer_id"],
                                "transport": peer.get("transport", "bitchat"),
                                "capabilities": response.get("capabilities", {}),
                                "device_info": peer.get("device_info", {}),
                            }
                        )

            # Fog compute node discovery
            if self.fog_coordinator:
                self.logger.info("Discovering fog compute nodes...")
                fog_nodes = await self.fog_coordinator.discover_nodes()

                for node in fog_nodes:
                    participants.append(
                        {
                            "type": "fog_node",
                            "node_id": node["node_id"],
                            "compute_power": node.get("compute_power", 1.0),
                            "memory_gb": node.get("memory_gb", 4),
                            "storage_gb": node.get("storage_gb", 32),
                            "gpu_available": node.get("gpu_available", False),
                        }
                    )

            self.participants = participants
            self.logger.info(
                f"Discovered {len(participants)} participants: "
                f"{len([p for p in participants if p['type'] == 'p2p_peer'])} P2P peers, "
                f"{len([p for p in participants if p['type'] == 'fog_node'])} fog nodes"
            )

            return participants

        except Exception as e:
            self.logger.error(f"Participant discovery failed: {e}")
            return []

    async def distribute_training_tasks(self) -> dict[str, Any]:
        """Distribute training tasks across participants."""
        if len(self.participants) < self.fed_config.min_participants:
            raise RuntimeError(
                f"Insufficient participants: {len(self.participants)} < {self.fed_config.min_participants}"
            )

        task_assignments = {}

        try:
            # Create training task specifications
            sum(
                [
                    self.base_config.enable_evomerge,
                    self.base_config.enable_quietstar,
                    self.base_config.enable_initial_compression,
                    self.base_config.enable_training,
                    self.base_config.enable_tool_baking,
                    self.base_config.enable_adas,
                    self.base_config.enable_final_compression,
                ]
            )

            # Distribute phases across participants based on capabilities
            phase_assignments = self._assign_phases_to_participants()

            for participant in self.participants:
                participant_id = participant.get("peer_id") or participant.get("node_id")
                assigned_phases = phase_assignments.get(participant_id, [])

                if assigned_phases:
                    # Create task configuration for this participant
                    task_config = self._create_participant_task_config(participant, assigned_phases)

                    # Send task assignment
                    if participant["type"] == "p2p_peer":
                        await self._send_p2p_task(participant, task_config)
                    elif participant["type"] == "fog_node":
                        await self._send_fog_task(participant, task_config)

                    task_assignments[participant_id] = {
                        "participant": participant,
                        "phases": assigned_phases,
                        "config": task_config,
                        "status": "assigned",
                        "start_time": time.time(),
                    }

            self.logger.info(f"Distributed training tasks to {len(task_assignments)} participants")
            return task_assignments

        except Exception as e:
            self.logger.error(f"Task distribution failed: {e}")
            raise

    def _assign_phases_to_participants(self) -> dict[str, list[str]]:
        """Assign Agent Forge phases to participants based on capabilities."""
        assignments = {}

        # Available phases in order
        phases = []
        if self.base_config.enable_evomerge:
            phases.append("evomerge")
        if self.base_config.enable_quietstar:
            phases.append("quietstar")
        if self.base_config.enable_initial_compression:
            phases.append("bitnet_compression")
        if self.base_config.enable_training:
            phases.append("forge_training")
        if self.base_config.enable_tool_baking:
            phases.append("tool_persona_baking")
        if self.base_config.enable_adas:
            phases.append("adas")
        if self.base_config.enable_final_compression:
            phases.append("final_compression")

        # Simple round-robin assignment for now
        # In production, this would consider compute power, memory, etc.
        for i, phase in enumerate(phases):
            participant_idx = i % len(self.participants)
            participant_id = self.participants[participant_idx].get("peer_id") or self.participants[
                participant_idx
            ].get("node_id")

            if participant_id not in assignments:
                assignments[participant_id] = []
            assignments[participant_id].append(phase)

        return assignments

    def _create_participant_task_config(self, participant: dict, assigned_phases: list[str]) -> dict[str, Any]:
        """Create training configuration for a specific participant."""
        # Create a subset of the base config for assigned phases
        config_dict = self.base_config.__dict__.copy()

        # Enable only assigned phases
        config_dict.update(
            {
                "enable_evomerge": "evomerge" in assigned_phases,
                "enable_quietstar": "quietstar" in assigned_phases,
                "enable_initial_compression": "bitnet_compression" in assigned_phases,
                "enable_training": "forge_training" in assigned_phases,
                "enable_tool_baking": "tool_persona_baking" in assigned_phases,
                "enable_adas": "adas" in assigned_phases,
                "enable_final_compression": "final_compression" in assigned_phases,
            }
        )

        # Adjust parameters based on participant capabilities
        if participant["type"] == "p2p_peer":
            # Mobile/edge device optimizations
            device_info = participant.get("device_info", {})
            if device_info.get("battery_level", 100) < 50:
                # Reduce training intensity for low battery
                config_dict["training_steps"] = min(config_dict["training_steps"], 1000)
                config_dict["batch_size"] = min(config_dict["batch_size"], 8)

        elif participant["type"] == "fog_node":
            # Fog node optimizations
            compute_power = participant.get("compute_power", 1.0)
            if compute_power > 2.0:
                # Increase training intensity for powerful nodes
                config_dict["training_steps"] = int(config_dict["training_steps"] * compute_power)
                config_dict["batch_size"] = int(config_dict["batch_size"] * min(compute_power, 4))

        return {
            "participant_id": participant.get("peer_id") or participant.get("node_id"),
            "assigned_phases": assigned_phases,
            "config": config_dict,
            "federated_round": self.current_round,
            "global_model_state": None,  # Would contain model weights in production
        }

    async def _send_p2p_task(self, participant: dict, task_config: dict):
        """Send training task to P2P peer."""
        if not self.transport_manager:
            raise RuntimeError("P2P transport not initialized")

        task_msg = AgentForgeMessage(
            message_type=MessageType.TRAINING_TASK,
            payload={"task_type": "agent_forge_training", "config": task_config, "round": self.current_round},
            sender_id="coordinator",
            recipient_id=participant["peer_id"],
        )

        await self.transport_manager.send_message(task_msg)
        self.logger.info(f"Sent P2P training task to {participant['peer_id']}")

    async def _send_fog_task(self, participant: dict, task_config: dict):
        """Send training task to fog compute node."""
        if not self.fog_coordinator:
            raise RuntimeError("Fog coordinator not initialized")

        task_spec = {
            "task_id": f"agent_forge_{self.current_round}_{participant['node_id']}",
            "task_type": "agent_forge_training",
            "config": task_config,
            "resource_requirements": {
                "cpu_cores": 2,
                "memory_gb": 4,
                "storage_gb": 10,
                "gpu_required": "adas" in task_config["assigned_phases"],
            },
        }

        await self.fog_coordinator.submit_task(participant["node_id"], task_spec)
        self.logger.info(f"Sent fog task to {participant['node_id']}")

    async def collect_results(self, task_assignments: dict[str, Any], timeout: int = 3600) -> dict[str, Any]:
        """Collect training results from all participants."""
        results = {}
        start_time = time.time()

        while len(results) < len(task_assignments) and (time.time() - start_time) < timeout:
            for participant_id, assignment in task_assignments.items():
                if participant_id in results:
                    continue

                try:
                    if assignment["participant"]["type"] == "p2p_peer":
                        result = await self._collect_p2p_result(assignment["participant"])
                    elif assignment["participant"]["type"] == "fog_node":
                        result = await self._collect_fog_result(assignment["participant"])

                    if result:
                        results[participant_id] = {
                            "participant": assignment["participant"],
                            "phases": assignment["phases"],
                            "result": result,
                            "completion_time": time.time(),
                        }
                        self.logger.info(f"Collected result from {participant_id}")

                except Exception as e:
                    self.logger.warning(f"Failed to collect result from {participant_id}: {e}")

            # Brief pause before checking again
            await asyncio.sleep(5)

        completion_rate = len(results) / len(task_assignments)
        self.logger.info(
            f"Collected {len(results)}/{len(task_assignments)} results " f"({completion_rate:.1%} completion rate)"
        )

        if completion_rate < self.fed_config.participation_threshold:
            self.logger.warning(
                f"Low participation rate: {completion_rate:.1%} < " f"{self.fed_config.participation_threshold:.1%}"
            )

        return results

    async def _collect_p2p_result(self, participant: dict) -> dict | None:
        """Collect training result from P2P peer."""
        if not self.transport_manager:
            return None

        # Query for result
        query_msg = AgentForgeMessage(
            message_type=MessageType.RESULT_QUERY,
            payload={"round": self.current_round},
            sender_id="coordinator",
            recipient_id=participant["peer_id"],
        )

        response = await self.transport_manager.send_message(query_msg)
        return response.get("result") if response else None

    async def _collect_fog_result(self, participant: dict) -> dict | None:
        """Collect training result from fog compute node."""
        if not self.fog_coordinator:
            return None

        task_id = f"agent_forge_{self.current_round}_{participant['node_id']}"
        result = await self.fog_coordinator.get_task_result(participant["node_id"], task_id)
        return result

    async def aggregate_results(self, results: dict[str, Any]) -> PhaseResult:
        """Aggregate training results from all participants."""
        self.logger.info("Starting model aggregation...")

        # Extract phase results by phase type
        phase_results = {}
        for participant_id, result_data in results.items():
            phases = result_data["phases"]
            result = result_data["result"]

            for phase in phases:
                if phase not in phase_results:
                    phase_results[phase] = []

                if result and "phase_results" in result:
                    phase_result = result["phase_results"].get(phase)
                    if phase_result:
                        phase_results[phase].append(phase_result)

        # Aggregate each phase type
        aggregated_metrics = {}
        final_model = None

        for phase, results_list in phase_results.items():
            if results_list:
                # Simple averaging for metrics (in production, would use proper aggregation)
                phase_metrics = {}

                # Collect all metric keys
                all_keys = set()
                for result in results_list:
                    if result.get("metrics"):
                        all_keys.update(result["metrics"].keys())

                # Average metrics
                for key in all_keys:
                    values = [r["metrics"].get(key, 0) for r in results_list if r.get("metrics")]
                    if values:
                        phase_metrics[key] = np.mean(values)

                aggregated_metrics[f"{phase}_metrics"] = phase_metrics

                # Use the first available model (in production, would aggregate model weights)
                if not final_model:
                    for result in results_list:
                        if result.get("model"):
                            final_model = result["model"]
                            break

        # Create aggregated result
        aggregated_result = PhaseResult(
            success=True,
            model=final_model,
            phase_name="FederatedAggregation",
            metrics={
                "federated_round": self.current_round,
                "participants": len(results),
                "completion_rate": len(results) / len(self.participants),
                "aggregation_method": self.fed_config.aggregation_method,
                **aggregated_metrics,
            },
            artifacts={"participant_results": results, "phase_distribution": phase_results},
            duration_seconds=time.time() - getattr(self, "_round_start_time", time.time()),
        )

        self.training_history.append(aggregated_result)
        self.logger.info(f"Completed federated round {self.current_round}")

        return aggregated_result

    async def run_federated_training(self) -> list[PhaseResult]:
        """Run complete federated Agent Forge training with HRRM integration."""
        self.logger.info(f"Starting federated Agent Forge training for {self.fed_config.federated_rounds} rounds")

        # Check for HRRM integration
        if HRRM_AVAILABLE and self.fed_config.hrrm_fog_burst_enabled:
            self.logger.info(f"HRRM integration enabled with preset: {self.fed_config.hrrm_preset}")

        results = []

        try:
            # Initial participant discovery
            participants = await self.discover_participants()
            if len(participants) < self.fed_config.min_participants:
                raise RuntimeError(
                    f"Insufficient participants: {len(participants)} < {self.fed_config.min_participants}"
                )

            # Run federated rounds
            for round_num in range(self.fed_config.federated_rounds):
                self.current_round = round_num + 1
                self._round_start_time = time.time()

                self.logger.info(f"Starting federated round {self.current_round}/{self.fed_config.federated_rounds}")

                # HRRM Integration: Execute HRRM distributed inference if enabled
                hrrm_result = None
                if HRRM_AVAILABLE and self.fed_config.hrrm_fog_burst_enabled:
                    try:
                        # Prepare model data for HRRM (simplified simulation)
                        model_data = b"simulated_model_data"  # In production, this would be actual model weights
                        model_config = {
                            "model_type": "hrrm",
                            "hidden_size": 768,
                            "num_layers": 12,
                            "num_attention_heads": 12,
                            "vocab_size": 50000,
                        }
                        training_data = {
                            "baseline_input": {"text": f"federated_round_{self.current_round}_input", "batch_size": 16}
                        }

                        # Execute HRRM federated training for this round
                        hrrm_result = await self.execute_hrrm_federated_training(
                            model_name=f"federated_model_round_{self.current_round}",
                            model_data=model_data,
                            model_config=model_config,
                            training_data=training_data,
                        )

                        self.logger.info(f"HRRM federated training completed for round {self.current_round}")

                    except Exception as e:
                        self.logger.error(f"HRRM federated training failed for round {self.current_round}: {e}")
                        # Continue with standard federated training

                # Standard federated training workflow
                task_assignments = await self.distribute_training_tasks()

                # Collect results
                round_results = await self.collect_results(task_assignments)

                # Aggregate results (integrate HRRM results if available)
                aggregated_result = await self.aggregate_results(round_results)

                # Enhance aggregated result with HRRM data if available
                if hrrm_result and hrrm_result.get("status") == "success":
                    aggregated_result.metadata["hrrm_integration"] = {
                        "enabled": True,
                        "preset": self.fed_config.hrrm_preset,
                        "sharding_strategy": self.fed_config.hrrm_sharding_strategy,
                        "parity_validation": hrrm_result.get("parity_validation", {}),
                        "fog_burst_enabled": self.fed_config.hrrm_fog_burst_enabled,
                        "baseline_latency_ms": hrrm_result.get("baseline_result", {})
                        .get("execution_metadata", {})
                        .get("total_time_ms", 0),
                    }
                else:
                    aggregated_result.metadata["hrrm_integration"] = {
                        "enabled": False,
                        "reason": "HRRM failed or disabled",
                    }

                results.append(aggregated_result)

                # Update global model
                self.global_model = aggregated_result.model

                # Checkpoint if configured
                if (
                    self.fed_config.enable_checkpointing
                    and self.current_round % self.fed_config.checkpoint_interval == 0
                ):
                    self._save_checkpoint(f"federated_round_{self.current_round}")

                self.logger.info(f"Completed federated round {self.current_round}")

            self.logger.info("Federated training completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Federated training failed: {e}")
            raise

    async def _setup_hrrm_fog_burst(self) -> FogBurstAdapter | None:
        """Setup HRRM fog burst adapter if HRRM integration is enabled"""

        if not HRRM_AVAILABLE or not self.fed_config.hrrm_fog_burst_enabled:
            return None

        try:
            # Initialize fog burst adapter for HRRM operations
            fog_burst_adapter = FogBurstAdapter(fog_gateway_url="http://localhost:8080")  # Default fog gateway

            self.logger.info(f"HRRM fog burst adapter initialized with preset: {self.fed_config.hrrm_preset}")
            return fog_burst_adapter

        except Exception as e:
            self.logger.error(f"Failed to setup HRRM fog burst: {e}")
            return None

    async def _execute_hrrm_distributed_inference(
        self, model_name: str, model_data: bytes, model_config: dict[str, Any], input_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute HRRM model inference using distributed fog burst"""

        if not HRRM_AVAILABLE:
            raise RuntimeError("HRRM export adapters not available")

        # Setup fog burst adapter
        fog_burst_adapter = await self._setup_hrrm_fog_burst()
        if not fog_burst_adapter:
            raise RuntimeError("Failed to setup HRRM fog burst adapter")

        try:
            # Configure model for distributed execution based on preset
            if self.fed_config.hrrm_preset == "local_only":
                # Execute locally (fallback)
                self.logger.info("HRRM preset 'local_only' - executing without fog burst")
                return {"status": "local_execution", "message": "Executed locally due to local_only preset"}

            elif self.fed_config.hrrm_preset == "fog_hybrid":
                # Use fog for heavy operations, local for light operations
                enhanced_config = model_config.copy()
                enhanced_config["fog_offload_threshold"] = 0.5
                enhanced_config["prefer_local_for_small_models"] = True

            elif self.fed_config.hrrm_preset == "distributed_inference":
                # Full distributed inference across fog nodes
                enhanced_config = model_config.copy()
                enhanced_config["force_distribution"] = True
                enhanced_config["sharding_strategy"] = self.fed_config.hrrm_sharding_strategy

            elif self.fed_config.hrrm_preset == "full_sharding":
                # Maximum distribution with aggressive sharding
                enhanced_config = model_config.copy()
                enhanced_config["aggressive_sharding"] = True
                enhanced_config["max_shards"] = 8
                enhanced_config["sharding_strategy"] = self.fed_config.hrrm_sharding_strategy

            else:
                enhanced_config = model_config.copy()

            # Execute distributed inference via fog burst
            result = await fog_burst_adapter.burst_to_fog(
                model_name=model_name, model_data=model_data, model_config=enhanced_config, input_data=input_data
            )

            self.logger.info(f"HRRM distributed inference completed for {model_name}")
            return result

        except Exception as e:
            self.logger.error(f"HRRM distributed inference failed: {e}")
            raise

    async def _validate_hrrm_parity(
        self,
        model_name: str,
        model_data: bytes,
        model_config: dict[str, Any],
        input_data: dict[str, Any],
        distributed_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate HRRM distributed execution against local execution"""

        if not self.fed_config.hrrm_parity_validation or not HRRM_AVAILABLE:
            return {"parity_validation": "skipped", "reason": "validation disabled or HRRM unavailable"}

        try:
            # Execute local version for comparison
            self.logger.info(f"Running local HRRM inference for parity validation: {model_name}")

            # Note: In production, this would execute the actual local HRRM model
            # For now, we simulate local execution results
            local_result = {
                "model_output": {"confidence": 0.85, "prediction": "simulated_local_output", "latency_ms": 150.0},
                "execution_metadata": {"strategy": "local", "memory_used_mb": 500.0},
            }

            # Validate parity using consistency validator
            consistency_validator = ConsistencyValidator(tolerance=self.fed_config.hrrm_parity_tolerance)

            parity_result = await consistency_validator.validate_distributed_parity(
                local_result=local_result,
                distributed_result=distributed_result.get("result", {}),
                model_name=model_name,
            )

            self.logger.info(
                f"HRRM parity validation for {model_name}: "
                f"{'PASSED' if parity_result.get('parity_passed', False) else 'FAILED'} "
                f"(max_diff: {parity_result.get('max_difference', 0.0):.6f})"
            )

            return parity_result

        except Exception as e:
            self.logger.error(f"HRRM parity validation failed for {model_name}: {e}")
            return {"parity_validation": "failed", "error": str(e), "parity_passed": False}

    async def execute_hrrm_federated_training(
        self, model_name: str, model_data: bytes, model_config: dict[str, Any], training_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute HRRM model training using federated learning with fog burst integration

        This method combines federated training with HRRM fog burst capabilities,
        allowing for distributed model training and inference across fog nodes.
        """

        if not HRRM_AVAILABLE:
            self.logger.warning("HRRM not available, falling back to standard federated training")
            return {"status": "fallback", "message": "HRRM not available"}

        self.logger.info(
            f"Starting HRRM federated training for {model_name} with preset: {self.fed_config.hrrm_preset}"
        )

        try:
            # Phase 1: Distributed inference for baseline evaluation
            baseline_result = await self._execute_hrrm_distributed_inference(
                model_name=f"{model_name}_baseline",
                model_data=model_data,
                model_config=model_config,
                input_data=training_data.get("baseline_input", {}),
            )

            # Phase 2: Parity validation if enabled
            parity_result = None
            if self.fed_config.hrrm_parity_validation:
                parity_result = await self._validate_hrrm_parity(
                    model_name=f"{model_name}_baseline",
                    model_data=model_data,
                    model_config=model_config,
                    input_data=training_data.get("baseline_input", {}),
                    distributed_result=baseline_result,
                )

                if not parity_result.get("parity_passed", False):
                    self.logger.warning(f"HRRM parity validation failed for {model_name}")
                    if self.fed_config.hrrm_preset == "fog_required":
                        raise RuntimeError("Parity validation failed and fog execution is required")

            # Phase 3: Federated training with HRRM integration
            training_results = []
            for round_num in range(self.fed_config.federated_rounds):
                self.logger.info(f"HRRM federated round {round_num + 1}/{self.fed_config.federated_rounds}")

                # Execute training round with fog burst if applicable
                round_result = await self._execute_hrrm_distributed_inference(
                    model_name=f"{model_name}_round_{round_num + 1}",
                    model_data=model_data,
                    model_config=model_config,
                    input_data=training_data.get(
                        f"round_{round_num + 1}_input", training_data.get("baseline_input", {})
                    ),
                )

                training_results.append(round_result)

            # Aggregate results
            final_result = {
                "status": "success",
                "model_name": model_name,
                "hrrm_preset": self.fed_config.hrrm_preset,
                "sharding_strategy": self.fed_config.hrrm_sharding_strategy,
                "baseline_result": baseline_result,
                "parity_validation": parity_result,
                "training_results": training_results,
                "federated_rounds_completed": len(training_results),
                "fog_burst_enabled": self.fed_config.hrrm_fog_burst_enabled,
            }

            self.logger.info(f"HRRM federated training completed for {model_name}")
            return final_result

        except Exception as e:
            self.logger.error(f"HRRM federated training failed for {model_name}: {e}")
            return {
                "status": "error",
                "model_name": model_name,
                "error": str(e),
                "hrrm_preset": self.fed_config.hrrm_preset,
            }

    def _save_checkpoint(self, checkpoint_name: str):
        """Save federated training checkpoint."""
        checkpoint_data = {
            "current_round": self.current_round,
            "global_model_state": self.global_model.state_dict() if self.global_model else None,
            "training_history": [
                {"round": i + 1, "metrics": result.metrics, "success": result.success}
                for i, result in enumerate(self.training_history)
            ],
            "participants": self.participants,
            "config": {"base_config": self.base_config.__dict__, "federated_config": self.fed_config.__dict__},
        }

        checkpoint_path = Path(f"./federated_checkpoints/{checkpoint_name}.pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint_data, checkpoint_path)
        self.logger.info(f"Saved federated checkpoint: {checkpoint_path}")


# Integration with unified pipeline
async def create_federated_pipeline(
    base_config: UnifiedConfig, federated_config: FederatedTrainingConfig | None = None
) -> FederatedAgentForge:
    """
    Create a federated Agent Forge pipeline.

    Args:
        base_config: Base Agent Forge configuration
        federated_config: Federated training configuration

    Returns:
        Configured FederatedAgentForge instance
    """
    if federated_config is None:
        federated_config = FederatedTrainingConfig()

    federated_pipeline = FederatedAgentForge(base_config, federated_config)
    return federated_pipeline


# CLI integration function
async def run_federated_agent_forge(
    config_path: str | None = None, federated_config_path: str | None = None, **kwargs
) -> list[PhaseResult]:
    """
    Run federated Agent Forge training from configuration files.

    Args:
        config_path: Path to base Agent Forge configuration
        federated_config_path: Path to federated training configuration
        **kwargs: Additional configuration overrides

    Returns:
        List of federated training round results
    """
    # Load base configuration
    if config_path:
        with open(config_path) as f:
            base_config_dict = json.load(f)
        base_config_dict.update(kwargs)
        base_config = UnifiedConfig(**base_config_dict)
    else:
        base_config = UnifiedConfig(**kwargs)

    # Load federated configuration
    if federated_config_path:
        with open(federated_config_path) as f:
            fed_config_dict = json.load(f)
        fed_config = FederatedTrainingConfig(**fed_config_dict)
    else:
        fed_config = FederatedTrainingConfig()

    # Create and run federated pipeline
    pipeline = await create_federated_pipeline(base_config, fed_config)
    results = await pipeline.run_federated_training()

    return results
