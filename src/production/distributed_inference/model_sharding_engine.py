"""Model Sharding Engine for Distributed Inference.

This module implements intelligent model partitioning across heterogeneous devices,
building on Sprint 6's P2P communication and resource management systems.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

# Import Sprint 6 infrastructure
from AIVillage.src.core.p2p.p2p_node import P2PNode, PeerCapabilities
from AIVillage.src.core.resources.device_profiler import DeviceProfiler
from AIVillage.src.core.resources.resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)


class ShardingStrategy(Enum):
    """Model sharding strategies."""

    MEMORY_AWARE = "memory_aware"  # Partition based on available memory
    COMPUTE_BALANCED = "compute_balanced"  # Balance compute load
    LAYER_SEQUENTIAL = "layer_sequential"  # Sequential layer distribution
    ATTENTION_SPLIT = "attention_split"  # Split attention heads
    HYBRID = "hybrid"  # Combination of strategies


@dataclass
class ModelShard:
    """Represents a model shard."""

    shard_id: str
    device_id: str
    layer_indices: list[int]
    parameters_count: int
    memory_mb: float
    compute_requirement: float
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ShardingPlan:
    """Complete model sharding plan."""

    model_name: str
    total_shards: int
    shards: list[ModelShard]
    activation_routing: dict[str, list[str]]
    memory_efficiency: float
    compute_balance_score: float
    created_at: float = field(default_factory=time.time)


@dataclass
class DeviceProfile:
    """Enhanced device profile for sharding decisions."""

    device_id: str
    capabilities: PeerCapabilities
    available_memory_mb: float
    compute_score: float
    network_latency_ms: float
    battery_level: float | None = None
    thermal_state: str = "normal"
    reliability_score: float = 0.8


class ModelShardingEngine:
    """Intelligent model sharding system for distributed inference."""

    def __init__(
        self,
        p2p_node: P2PNode,
        resource_monitor: ResourceMonitor,
        device_profiler: DeviceProfiler,
    ) -> None:
        self.p2p_node = p2p_node
        self.resource_monitor = resource_monitor
        self.device_profiler = device_profiler

        # Sharding state
        self.current_sharding_plan: ShardingPlan | None = None
        self.active_shards: dict[str, ModelShard] = {}
        self.device_assignments: dict[str, list[str]] = {}  # device_id -> shard_ids

        # Model analysis cache
        self.model_analysis_cache: dict[str, dict[str, Any]] = {}

        # Performance tracking
        self.sharding_stats = {
            "models_sharded": 0,
            "resharding_events": 0,
            "failed_assignments": 0,
            "avg_sharding_time": 0.0,
            "memory_utilization": 0.0,
            "compute_balance": 0.0,
        }

        # Configuration
        self.config = {
            "memory_buffer_percent": 20,  # Reserve 20% memory buffer
            "max_shards_per_device": 3,  # Limit shards per device
            "min_shard_size_mb": 50,  # Minimum shard size
            "max_latency_ms": 200,  # Maximum acceptable latency
            "reliability_threshold": 0.6,  # Minimum device reliability
        }

        logger.info("ModelShardingEngine initialized")

    async def shard_model(
        self,
        model_path: str,
        strategy: ShardingStrategy = ShardingStrategy.HYBRID,
        target_devices: list[str] | None = None,
    ) -> ShardingPlan:
        """Shard model across available devices.

        Args:
            model_path: Path to the model to shard
            strategy: Sharding strategy to use
            target_devices: Specific devices to use (None for auto-discovery)

        Returns:
            ShardingPlan describing the sharding configuration
        """
        start_time = time.time()

        try:
            logger.info(
                f"Starting model sharding: {model_path} with strategy {strategy.value}"
            )

            # 1. Analyze model structure
            model_analysis = await self._analyze_model(model_path)

            # 2. Get available devices
            device_profiles = await self._get_device_profiles(target_devices)

            # 3. Create sharding plan
            sharding_plan = await self._create_sharding_plan(
                model_analysis, device_profiles, strategy
            )

            # 4. Validate and optimize plan
            optimized_plan = await self._optimize_sharding_plan(
                sharding_plan, device_profiles
            )

            # 5. Store and activate plan
            self.current_sharding_plan = optimized_plan
            await self._activate_sharding_plan(optimized_plan)

            # Update stats
            duration = time.time() - start_time
            self.sharding_stats["models_sharded"] += 1
            self.sharding_stats["avg_sharding_time"] = (
                self.sharding_stats["avg_sharding_time"] + duration
            ) / 2

            logger.info(
                f"Model sharding completed in {duration:.2f}s with {len(optimized_plan.shards)} shards"
            )
            return optimized_plan

        except Exception as e:
            logger.exception(f"Model sharding failed: {e}")
            raise

    async def _analyze_model(self, model_path: str) -> dict[str, Any]:
        """Analyze model structure for sharding decisions."""
        if model_path in self.model_analysis_cache:
            return self.model_analysis_cache[model_path]

        logger.info(f"Analyzing model structure: {model_path}")

        try:
            # Load model for analysis (without full parameters)
            AutoTokenizer.from_pretrained(model_path)
            model_config = AutoModelForCausalLM.from_pretrained(
                model_path, config_only=True
            ).config

            # Analyze layer structure
            num_layers = getattr(model_config, "num_hidden_layers", 12)
            hidden_size = getattr(model_config, "hidden_size", 768)
            num_attention_heads = getattr(model_config, "num_attention_heads", 12)
            vocab_size = getattr(model_config, "vocab_size", 30000)

            # Estimate memory requirements per layer
            attention_params = hidden_size * hidden_size * 4  # Q, K, V, O projections
            mlp_params = hidden_size * (
                getattr(model_config, "intermediate_size", hidden_size * 4)
            )
            layer_params = attention_params + mlp_params
            layer_memory_mb = (layer_params * 2) / (1024 * 1024)  # FP16

            # Estimate compute requirements
            layer_compute_score = (
                attention_params + mlp_params
            ) / 1e6  # Normalized score

            analysis = {
                "model_path": model_path,
                "num_layers": num_layers,
                "hidden_size": hidden_size,
                "num_attention_heads": num_attention_heads,
                "vocab_size": vocab_size,
                "layer_memory_mb": layer_memory_mb,
                "layer_compute_score": layer_compute_score,
                "total_memory_mb": layer_memory_mb * num_layers
                + 100,  # +100MB for embeddings
                "embedding_memory_mb": (vocab_size * hidden_size * 2) / (1024 * 1024),
                "can_split_attention": num_attention_heads >= 4,
                "optimal_shard_count": max(2, min(8, num_layers // 3)),
                "analyzed_at": time.time(),
            }

            self.model_analysis_cache[model_path] = analysis
            logger.info(
                f"Model analysis complete: {num_layers} layers, {analysis['total_memory_mb']:.1f}MB"
            )

            return analysis

        except Exception as e:
            logger.exception(f"Model analysis failed: {e}")
            raise

    async def _get_device_profiles(
        self, target_devices: list[str] | None = None
    ) -> list[DeviceProfile]:
        """Get profiles of available devices for sharding."""
        device_profiles = []

        # Get suitable peers from P2P network
        suitable_peers = self.p2p_node.get_suitable_evolution_peers(min_count=2)

        # Include local device
        local_capabilities = self.p2p_node.local_capabilities
        if local_capabilities and local_capabilities.is_suitable_for_evolution():
            suitable_peers.insert(0, local_capabilities)

        for peer in suitable_peers:
            # Skip if target_devices specified and this device not in list
            if target_devices and peer.device_id not in target_devices:
                continue

            # Get current resource status
            if peer.device_id == self.p2p_node.node_id:
                # Local device - get real-time data
                snapshot = self.device_profiler.current_snapshot
                if snapshot:
                    available_memory = snapshot.ram_available_mb or peer.ram_mb
                    thermal_state = snapshot.thermal_state or "normal"
                else:
                    available_memory = peer.ram_mb * 0.7  # Estimate 70% available
                    thermal_state = "normal"
            else:
                # Remote device - use cached data with buffer
                available_memory = peer.ram_mb * 0.6  # Conservative estimate
                thermal_state = peer.thermal_state

            # Calculate compute score
            compute_score = (peer.cpu_cores * 2 + peer.ram_mb / 1024) / 10
            if peer.performance_tier == "premium":
                compute_score *= 1.5
            elif peer.performance_tier == "high":
                compute_score *= 1.2
            elif peer.performance_tier == "low":
                compute_score *= 0.7

            # Apply memory buffer
            available_memory *= (100 - self.config["memory_buffer_percent"]) / 100

            device_profile = DeviceProfile(
                device_id=peer.device_id,
                capabilities=peer,
                available_memory_mb=available_memory,
                compute_score=compute_score,
                network_latency_ms=peer.latency_ms,
                battery_level=peer.battery_percent,
                thermal_state=thermal_state,
                reliability_score=peer.trust_score,
            )

            # Filter out unreliable devices
            if device_profile.reliability_score >= self.config["reliability_threshold"]:
                device_profiles.append(device_profile)

        # Sort by suitability (memory + compute + reliability)
        device_profiles.sort(
            key=lambda d: (
                d.available_memory_mb * 0.4
                + d.compute_score * 0.3
                + d.reliability_score * 0.3
            ),
            reverse=True,
        )

        logger.info(f"Found {len(device_profiles)} suitable devices for sharding")
        return device_profiles

    async def _create_sharding_plan(
        self,
        model_analysis: dict[str, Any],
        device_profiles: list[DeviceProfile],
        strategy: ShardingStrategy,
    ) -> ShardingPlan:
        """Create initial sharding plan based on strategy."""
        if not device_profiles:
            msg = "No suitable devices available for sharding"
            raise ValueError(msg)

        if strategy == ShardingStrategy.MEMORY_AWARE:
            return await self._create_memory_aware_plan(model_analysis, device_profiles)
        if strategy == ShardingStrategy.COMPUTE_BALANCED:
            return await self._create_compute_balanced_plan(
                model_analysis, device_profiles
            )
        if strategy == ShardingStrategy.LAYER_SEQUENTIAL:
            return await self._create_sequential_plan(model_analysis, device_profiles)
        if strategy == ShardingStrategy.ATTENTION_SPLIT:
            return await self._create_attention_split_plan(
                model_analysis, device_profiles
            )
        if strategy == ShardingStrategy.HYBRID:
            return await self._create_hybrid_plan(model_analysis, device_profiles)
        msg = f"Unknown sharding strategy: {strategy}"
        raise ValueError(msg)

    async def _create_memory_aware_plan(
        self, model_analysis: dict[str, Any], device_profiles: list[DeviceProfile]
    ) -> ShardingPlan:
        """Create sharding plan based on available memory."""
        num_layers = model_analysis["num_layers"]
        layer_memory_mb = model_analysis["layer_memory_mb"]

        shards = []
        current_shard_layers = []
        current_shard_memory = 0
        device_idx = 0

        for layer_idx in range(num_layers):
            # Check if current device can accommodate another layer
            current_device = device_profiles[device_idx % len(device_profiles)]

            if (
                current_shard_memory + layer_memory_mb
                <= current_device.available_memory_mb
                and len(current_shard_layers) < self.config["max_shards_per_device"]
            ):
                current_shard_layers.append(layer_idx)
                current_shard_memory += layer_memory_mb
            else:
                # Create shard with current layers
                if current_shard_layers:
                    shard = ModelShard(
                        shard_id=str(uuid.uuid4()),
                        device_id=current_device.device_id,
                        layer_indices=current_shard_layers.copy(),
                        parameters_count=len(current_shard_layers)
                        * int(layer_memory_mb * 1024 * 1024 / 2),
                        memory_mb=current_shard_memory,
                        compute_requirement=len(current_shard_layers)
                        * model_analysis["layer_compute_score"],
                    )
                    shards.append(shard)

                # Move to next device
                device_idx += 1
                if device_idx >= len(device_profiles):
                    device_idx = 0  # Wrap around if needed

                current_device = device_profiles[device_idx]
                current_shard_layers = [layer_idx]
                current_shard_memory = layer_memory_mb

        # Add final shard
        if current_shard_layers:
            current_device = device_profiles[device_idx % len(device_profiles)]
            shard = ModelShard(
                shard_id=str(uuid.uuid4()),
                device_id=current_device.device_id,
                layer_indices=current_shard_layers,
                parameters_count=len(current_shard_layers)
                * int(layer_memory_mb * 1024 * 1024 / 2),
                memory_mb=current_shard_memory,
                compute_requirement=len(current_shard_layers)
                * model_analysis["layer_compute_score"],
            )
            shards.append(shard)

        # Create activation routing
        activation_routing = {}
        for i, shard in enumerate(shards):
            if i == 0:
                activation_routing[shard.shard_id] = []  # First shard receives input
            else:
                activation_routing[shard.shard_id] = [shards[i - 1].shard_id]

        return ShardingPlan(
            model_name=model_analysis["model_path"],
            total_shards=len(shards),
            shards=shards,
            activation_routing=activation_routing,
            memory_efficiency=self._calculate_memory_efficiency(
                shards, device_profiles
            ),
            compute_balance_score=self._calculate_compute_balance(
                shards, device_profiles
            ),
        )

    async def _create_compute_balanced_plan(
        self, model_analysis: dict[str, Any], device_profiles: list[DeviceProfile]
    ) -> ShardingPlan:
        """Create sharding plan balanced by compute capacity."""
        num_layers = model_analysis["num_layers"]
        layer_compute_score = model_analysis["layer_compute_score"]
        layer_memory_mb = model_analysis["layer_memory_mb"]

        # Calculate total compute and distribute proportionally
        total_compute = sum(d.compute_score for d in device_profiles)
        layers_per_device = []

        for device in device_profiles:
            device_proportion = device.compute_score / total_compute
            device_layers = max(1, int(num_layers * device_proportion))
            layers_per_device.append(device_layers)

        # Adjust for exact layer count
        total_assigned = sum(layers_per_device)
        diff = num_layers - total_assigned
        if diff > 0:
            # Add remaining layers to most capable devices
            for i in range(diff):
                layers_per_device[i % len(layers_per_device)] += 1
        elif diff < 0:
            # Remove excess layers from least capable devices
            for i in range(abs(diff)):
                idx = len(layers_per_device) - 1 - (i % len(layers_per_device))
                if layers_per_device[idx] > 1:
                    layers_per_device[idx] -= 1

        # Create shards
        shards = []
        layer_idx = 0

        for device_idx, device in enumerate(device_profiles):
            if layers_per_device[device_idx] == 0:
                continue

            layer_indices = list(
                range(layer_idx, layer_idx + layers_per_device[device_idx])
            )

            shard = ModelShard(
                shard_id=str(uuid.uuid4()),
                device_id=device.device_id,
                layer_indices=layer_indices,
                parameters_count=len(layer_indices)
                * int(layer_memory_mb * 1024 * 1024 / 2),
                memory_mb=len(layer_indices) * layer_memory_mb,
                compute_requirement=len(layer_indices) * layer_compute_score,
            )
            shards.append(shard)
            layer_idx += layers_per_device[device_idx]

        # Create activation routing
        activation_routing = {}
        for i, shard in enumerate(shards):
            if i == 0:
                activation_routing[shard.shard_id] = []
            else:
                activation_routing[shard.shard_id] = [shards[i - 1].shard_id]

        return ShardingPlan(
            model_name=model_analysis["model_path"],
            total_shards=len(shards),
            shards=shards,
            activation_routing=activation_routing,
            memory_efficiency=self._calculate_memory_efficiency(
                shards, device_profiles
            ),
            compute_balance_score=self._calculate_compute_balance(
                shards, device_profiles
            ),
        )

    async def _create_sequential_plan(
        self, model_analysis: dict[str, Any], device_profiles: list[DeviceProfile]
    ) -> ShardingPlan:
        """Create sequential layer distribution plan."""
        num_layers = model_analysis["num_layers"]
        num_devices = len(device_profiles)
        layers_per_shard = max(1, num_layers // num_devices)

        shards = []
        layer_idx = 0

        for device_idx, device in enumerate(device_profiles):
            # Calculate layers for this shard
            remaining_layers = num_layers - layer_idx
            remaining_devices = num_devices - device_idx

            if remaining_devices == 1:
                # Last device gets all remaining layers
                shard_layers = remaining_layers
            else:
                shard_layers = min(layers_per_shard, remaining_layers)

            if shard_layers <= 0:
                break

            layer_indices = list(range(layer_idx, layer_idx + shard_layers))

            shard = ModelShard(
                shard_id=str(uuid.uuid4()),
                device_id=device.device_id,
                layer_indices=layer_indices,
                parameters_count=len(layer_indices)
                * int(model_analysis["layer_memory_mb"] * 1024 * 1024 / 2),
                memory_mb=len(layer_indices) * model_analysis["layer_memory_mb"],
                compute_requirement=len(layer_indices)
                * model_analysis["layer_compute_score"],
            )
            shards.append(shard)
            layer_idx += shard_layers

        # Create activation routing
        activation_routing = {}
        for i, shard in enumerate(shards):
            if i == 0:
                activation_routing[shard.shard_id] = []
            else:
                activation_routing[shard.shard_id] = [shards[i - 1].shard_id]

        return ShardingPlan(
            model_name=model_analysis["model_path"],
            total_shards=len(shards),
            shards=shards,
            activation_routing=activation_routing,
            memory_efficiency=self._calculate_memory_efficiency(
                shards, device_profiles
            ),
            compute_balance_score=self._calculate_compute_balance(
                shards, device_profiles
            ),
        )

    async def _create_attention_split_plan(
        self, model_analysis: dict[str, Any], device_profiles: list[DeviceProfile]
    ) -> ShardingPlan:
        """Create plan that splits attention heads across devices."""
        if not model_analysis["can_split_attention"]:
            logger.warning(
                "Model cannot split attention heads, falling back to sequential"
            )
            return await self._create_sequential_plan(model_analysis, device_profiles)

        # For now, implement as layer-based (attention head splitting requires model architecture changes)
        # This is a placeholder for future attention head splitting implementation
        return await self._create_sequential_plan(model_analysis, device_profiles)

    async def _create_hybrid_plan(
        self, model_analysis: dict[str, Any], device_profiles: list[DeviceProfile]
    ) -> ShardingPlan:
        """Create hybrid plan combining multiple strategies."""
        # Try memory-aware first, then optimize for compute balance
        memory_plan = await self._create_memory_aware_plan(
            model_analysis, device_profiles
        )
        compute_plan = await self._create_compute_balanced_plan(
            model_analysis, device_profiles
        )

        # Choose better plan based on efficiency metrics
        if (
            memory_plan.memory_efficiency + memory_plan.compute_balance_score
            > compute_plan.memory_efficiency + compute_plan.compute_balance_score
        ):
            return memory_plan
        return compute_plan

    def _calculate_memory_efficiency(
        self, shards: list[ModelShard], device_profiles: list[DeviceProfile]
    ) -> float:
        """Calculate memory utilization efficiency."""
        device_memory_map = {
            d.device_id: d.available_memory_mb for d in device_profiles
        }

        total_used = 0
        total_available = 0

        for shard in shards:
            device_memory = device_memory_map.get(shard.device_id, 0)
            total_used += shard.memory_mb
            total_available += device_memory

        return total_used / total_available if total_available > 0 else 0.0

    def _calculate_compute_balance(
        self, shards: list[ModelShard], device_profiles: list[DeviceProfile]
    ) -> float:
        """Calculate compute load balance score."""
        device_compute_map = {d.device_id: d.compute_score for d in device_profiles}
        device_loads = {}

        # Calculate load per device
        for shard in shards:
            device_id = shard.device_id
            device_compute = device_compute_map.get(device_id, 1.0)
            load_ratio = shard.compute_requirement / device_compute

            if device_id not in device_loads:
                device_loads[device_id] = 0
            device_loads[device_id] += load_ratio

        if not device_loads:
            return 0.0

        # Calculate balance score (lower variance = better balance)
        load_values = list(device_loads.values())
        mean_load = sum(load_values) / len(load_values)
        variance = sum((x - mean_load) ** 2 for x in load_values) / len(load_values)

        # Convert to 0-1 score (lower variance = higher score)
        balance_score = 1.0 / (1.0 + variance)
        return balance_score

    async def _optimize_sharding_plan(
        self, plan: ShardingPlan, device_profiles: list[DeviceProfile]
    ) -> ShardingPlan:
        """Optimize sharding plan for better performance."""
        # Check memory constraints
        device_memory_map = {
            d.device_id: d.available_memory_mb for d in device_profiles
        }

        optimized_shards = []
        issues_found = []

        for shard in plan.shards:
            available_memory = device_memory_map.get(shard.device_id, 0)

            if shard.memory_mb > available_memory:
                issues_found.append(
                    f"Shard {shard.shard_id} requires {shard.memory_mb:.1f}MB but device {shard.device_id} only has {available_memory:.1f}MB"
                )

                # Try to split this shard
                if len(shard.layer_indices) > 1:
                    mid_point = len(shard.layer_indices) // 2

                    # First half
                    shard1 = ModelShard(
                        shard_id=str(uuid.uuid4()),
                        device_id=shard.device_id,
                        layer_indices=shard.layer_indices[:mid_point],
                        parameters_count=shard.parameters_count // 2,
                        memory_mb=shard.memory_mb / 2,
                        compute_requirement=shard.compute_requirement / 2,
                    )

                    # Second half - try to find another device
                    best_device = None
                    for device in device_profiles:
                        if (
                            device.device_id != shard.device_id
                            and device.available_memory_mb >= shard.memory_mb / 2
                        ):
                            best_device = device
                            break

                    if best_device:
                        shard2 = ModelShard(
                            shard_id=str(uuid.uuid4()),
                            device_id=best_device.device_id,
                            layer_indices=shard.layer_indices[mid_point:],
                            parameters_count=shard.parameters_count
                            - shard.parameters_count // 2,
                            memory_mb=shard.memory_mb / 2,
                            compute_requirement=shard.compute_requirement / 2,
                        )
                        optimized_shards.extend([shard1, shard2])
                    else:
                        optimized_shards.append(shard)  # Keep original if no solution
                else:
                    optimized_shards.append(shard)  # Single layer, can't split
            else:
                optimized_shards.append(shard)

        if issues_found:
            logger.warning(f"Sharding issues found and addressed: {issues_found}")

        # Update activation routing for optimized shards
        optimized_routing = {}
        for i, shard in enumerate(optimized_shards):
            if i == 0:
                optimized_routing[shard.shard_id] = []
            else:
                optimized_routing[shard.shard_id] = [optimized_shards[i - 1].shard_id]

        optimized_plan = ShardingPlan(
            model_name=plan.model_name,
            total_shards=len(optimized_shards),
            shards=optimized_shards,
            activation_routing=optimized_routing,
            memory_efficiency=self._calculate_memory_efficiency(
                optimized_shards, device_profiles
            ),
            compute_balance_score=self._calculate_compute_balance(
                optimized_shards, device_profiles
            ),
        )

        logger.info(
            f"Sharding plan optimized: {len(plan.shards)} -> {len(optimized_shards)} shards"
        )
        return optimized_plan

    async def _activate_sharding_plan(self, plan: ShardingPlan) -> None:
        """Activate sharding plan by distributing shards to devices."""
        logger.info(f"Activating sharding plan with {plan.total_shards} shards")

        # Update internal state
        self.active_shards = {shard.shard_id: shard for shard in plan.shards}
        self.device_assignments = {}

        for shard in plan.shards:
            device_id = shard.device_id
            if device_id not in self.device_assignments:
                self.device_assignments[device_id] = []
            self.device_assignments[device_id].append(shard.shard_id)

        # Notify P2P network about sharding plan
        sharding_message = {
            "type": "SHARDING_PLAN_ACTIVATED",
            "plan_id": str(uuid.uuid4()),
            "model_name": plan.model_name,
            "total_shards": plan.total_shards,
            "device_assignments": self.device_assignments,
            "timestamp": time.time(),
        }

        await self.p2p_node.broadcast_to_peers(
            "SHARDING_PLAN_ACTIVATED", sharding_message
        )

        # Update performance stats
        self.sharding_stats["memory_utilization"] = plan.memory_efficiency
        self.sharding_stats["compute_balance"] = plan.compute_balance_score

        logger.info("Sharding plan activated successfully")

    async def partition_by_memory(
        self, model_path: str, device_profiles: list[DeviceProfile]
    ) -> ShardingPlan:
        """Partition model layers based on available device memory."""
        model_analysis = await self._analyze_model(model_path)
        return await self._create_memory_aware_plan(model_analysis, device_profiles)

    async def partition_by_compute(
        self, model_path: str, device_profiles: list[DeviceProfile]
    ) -> ShardingPlan:
        """Balance compute load across heterogeneous devices."""
        model_analysis = await self._analyze_model(model_path)
        return await self._create_compute_balanced_plan(model_analysis, device_profiles)

    def get_sharding_status(self) -> dict[str, Any]:
        """Get current sharding status."""
        return {
            "active_plan": self.current_sharding_plan is not None,
            "plan_details": (
                {
                    "model_name": (
                        self.current_sharding_plan.model_name
                        if self.current_sharding_plan
                        else None
                    ),
                    "total_shards": len(self.active_shards),
                    "devices_used": len(self.device_assignments),
                    "memory_efficiency": self.sharding_stats["memory_utilization"],
                    "compute_balance": self.sharding_stats["compute_balance"],
                }
                if self.current_sharding_plan
                else None
            ),
            "statistics": self.sharding_stats.copy(),
            "cache_size": len(self.model_analysis_cache),
        }

    async def cleanup_sharding(self) -> None:
        """Clean up current sharding plan."""
        if self.current_sharding_plan:
            logger.info("Cleaning up sharding plan")

            # Notify peers
            cleanup_message = {
                "type": "SHARDING_CLEANUP",
                "model_name": self.current_sharding_plan.model_name,
                "timestamp": time.time(),
            }

            await self.p2p_node.broadcast_to_peers("SHARDING_CLEANUP", cleanup_message)

            # Reset state
            self.current_sharding_plan = None
            self.active_shards.clear()
            self.device_assignments.clear()

            logger.info("Sharding cleanup completed")
