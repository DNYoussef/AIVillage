"""
Model Sharding Engine for Distributed Inference

Provides model sharding capabilities for distributing large AI models across
multiple devices in a fog computing network. Supports various sharding strategies
and automatic placement optimization based on device capabilities.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ShardingStrategy(Enum):
    """Strategies for model sharding across devices."""

    LAYER_WISE = "layer_wise"  # Shard by transformer layers
    ATTENTION = "attention"  # Shard attention mechanisms
    FFN = "ffn"  # Shard feed-forward networks
    HYBRID = "hybrid"  # Combination of strategies
    TENSOR_PARALLEL = "tensor_parallel"  # Tensor parallelism
    PIPELINE_PARALLEL = "pipeline_parallel"  # Pipeline parallelism


@dataclass
class ModelShard:
    """Represents a shard of a model deployed on a specific device."""

    shard_id: str
    device_id: str
    layer_indices: list[int] = field(default_factory=list)
    parameters_count: int = 0
    memory_mb: float = 0.0
    compute_requirement: float = 0.0

    # Shard metadata
    shard_type: str = "layer_wise"  # layer_wise, attention, ffn
    dependency_shards: list[str] = field(default_factory=list)
    output_shape: tuple[int, ...] = field(default_factory=tuple)

    # Runtime information
    status: str = "created"  # created, deploying, active, error
    last_updated: float = 0.0


@dataclass
class ShardingPlan:
    """Complete plan for sharding a model across devices."""

    model_name: str
    total_shards: int
    shards: list[ModelShard] = field(default_factory=list)

    # Communication routing for shard coordination
    activation_routing: dict[str, list[str]] = field(default_factory=dict)

    # Performance metrics
    memory_efficiency: float = 0.0
    compute_balance_score: float = 0.0
    estimated_latency_ms: float = 0.0

    # Deployment metadata
    created_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    strategy: ShardingStrategy = ShardingStrategy.LAYER_WISE


class ModelShardingEngine:
    """Engine for creating and managing model shards across fog network."""

    def __init__(self):
        self.active_plans: dict[str, ShardingPlan] = {}
        self.device_capabilities: dict[str, dict[str, Any]] = {}

    async def shard_model(
        self, model_path: str, strategy: ShardingStrategy, target_devices: list[str], constraints: dict[str, Any] = None
    ) -> ShardingPlan:
        """Create a sharding plan for the given model and devices."""

        constraints = constraints or {}
        model_name = model_path.split("/")[-1].replace(".safetensors", "").replace(".pth", "")

        logger.info(f"Creating sharding plan for {model_name} using {strategy.value} strategy")

        # Analyze model architecture (simplified implementation)
        model_info = await self._analyze_model(model_path)

        # Create sharding plan based on strategy
        if strategy == ShardingStrategy.LAYER_WISE:
            plan = await self._create_layer_wise_plan(model_name, model_info, target_devices)
        elif strategy == ShardingStrategy.HYBRID:
            plan = await self._create_hybrid_plan(model_name, model_info, target_devices)
        elif strategy == ShardingStrategy.ATTENTION:
            plan = await self._create_attention_plan(model_name, model_info, target_devices)
        elif strategy == ShardingStrategy.FFN:
            plan = await self._create_ffn_plan(model_name, model_info, target_devices)
        else:
            # Default to layer-wise for unsupported strategies
            plan = await self._create_layer_wise_plan(model_name, model_info, target_devices)

        plan.strategy = strategy

        # Optimize placement based on device capabilities
        await self._optimize_placement(plan, target_devices, constraints)

        # Calculate performance metrics
        await self._calculate_metrics(plan)

        self.active_plans[model_name] = plan
        logger.info(f"Created sharding plan with {len(plan.shards)} shards")

        return plan

    async def _analyze_model(self, model_path: str) -> dict[str, Any]:
        """Analyze model architecture to determine sharding parameters."""

        # This would normally load and inspect the actual model
        # For now, provide reasonable defaults based on common architectures

        return {
            "total_layers": 24,  # Typical for medium-size models
            "hidden_size": 768,
            "attention_heads": 12,
            "ffn_size": 3072,
            "total_parameters": 150_000_000,  # ~150M parameters
            "estimated_memory_mb": 600,  # FP16 model
        }

    async def _create_layer_wise_plan(
        self, model_name: str, model_info: dict, target_devices: list[str]
    ) -> ShardingPlan:
        """Create layer-wise sharding plan."""

        total_layers = model_info["total_layers"]
        num_devices = len(target_devices)
        layers_per_shard = max(1, total_layers // num_devices)

        shards = []
        layer_start = 0

        for i, device_id in enumerate(target_devices):
            # Calculate layer range for this shard
            layer_end = min(layer_start + layers_per_shard, total_layers)
            if i == num_devices - 1:  # Last device gets remaining layers
                layer_end = total_layers

            layer_indices = list(range(layer_start, layer_end))

            # Estimate resource requirements
            layers_in_shard = len(layer_indices)
            params_per_layer = model_info["total_parameters"] // total_layers
            shard_params = params_per_layer * layers_in_shard
            memory_mb = (shard_params * 2) / (1024 * 1024)  # FP16 = 2 bytes per param
            compute_requirement = layers_in_shard / total_layers

            shard = ModelShard(
                shard_id=f"{model_name}_shard_{i}",
                device_id=device_id,
                layer_indices=layer_indices,
                parameters_count=shard_params,
                memory_mb=memory_mb,
                compute_requirement=compute_requirement,
                shard_type="layer_wise",
            )

            shards.append(shard)
            layer_start = layer_end

        # Create activation routing (sequential for layer-wise)
        activation_routing = {}
        for i in range(len(shards) - 1):
            activation_routing[shards[i].shard_id] = [shards[i + 1].shard_id]

        return ShardingPlan(
            model_name=model_name, total_shards=len(shards), shards=shards, activation_routing=activation_routing
        )

    async def _create_hybrid_plan(self, model_name: str, model_info: dict, target_devices: list[str]) -> ShardingPlan:
        """Create hybrid sharding plan combining multiple strategies."""

        # For hybrid, use layer-wise as base and add attention/FFN splits for larger models
        base_plan = await self._create_layer_wise_plan(model_name, model_info, target_devices)

        # If we have many devices, split some layers further
        if len(target_devices) > 4 and model_info["total_parameters"] > 100_000_000:
            # Split attention and FFN for the largest shards
            shards_to_split = sorted(base_plan.shards, key=lambda s: s.parameters_count, reverse=True)[:2]

            new_shards = []
            for shard in base_plan.shards:
                if shard in shards_to_split and len(shard.layer_indices) > 2:
                    # Split this shard into attention and FFN components
                    attention_shard = ModelShard(
                        shard_id=f"{shard.shard_id}_attention",
                        device_id=shard.device_id,
                        layer_indices=shard.layer_indices,
                        parameters_count=shard.parameters_count // 2,
                        memory_mb=shard.memory_mb * 0.6,
                        compute_requirement=shard.compute_requirement * 0.6,
                        shard_type="attention",
                    )

                    ffn_shard = ModelShard(
                        shard_id=f"{shard.shard_id}_ffn",
                        device_id=shard.device_id,
                        layer_indices=shard.layer_indices,
                        parameters_count=shard.parameters_count // 2,
                        memory_mb=shard.memory_mb * 0.4,
                        compute_requirement=shard.compute_requirement * 0.4,
                        shard_type="ffn",
                        dependency_shards=[attention_shard.shard_id],
                    )

                    new_shards.extend([attention_shard, ffn_shard])
                else:
                    new_shards.append(shard)

            base_plan.shards = new_shards
            base_plan.total_shards = len(new_shards)

        return base_plan

    async def _create_attention_plan(
        self, model_name: str, model_info: dict, target_devices: list[str]
    ) -> ShardingPlan:
        """Create attention-focused sharding plan."""

        # Split attention heads across devices
        total_heads = model_info["attention_heads"]
        num_devices = len(target_devices)
        heads_per_device = max(1, total_heads // num_devices)

        shards = []
        head_start = 0

        for i, device_id in enumerate(target_devices):
            head_end = min(head_start + heads_per_device, total_heads)
            if i == num_devices - 1:
                head_end = total_heads

            heads_in_shard = head_end - head_start
            params_ratio = heads_in_shard / total_heads
            shard_params = int(model_info["total_parameters"] * params_ratio)
            memory_mb = (shard_params * 2) / (1024 * 1024)

            shard = ModelShard(
                shard_id=f"{model_name}_attention_shard_{i}",
                device_id=device_id,
                layer_indices=list(range(model_info["total_layers"])),  # All layers, subset of heads
                parameters_count=shard_params,
                memory_mb=memory_mb,
                compute_requirement=params_ratio,
                shard_type="attention",
            )

            shards.append(shard)
            head_start = head_end

        return ShardingPlan(
            model_name=model_name,
            total_shards=len(shards),
            shards=shards,
            activation_routing={},  # Attention shards communicate via all-reduce
        )

    async def _create_ffn_plan(self, model_name: str, model_info: dict, target_devices: list[str]) -> ShardingPlan:
        """Create FFN-focused sharding plan."""

        # Similar to attention but focus on feed-forward network components
        return await self._create_layer_wise_plan(model_name, model_info, target_devices)

    async def _optimize_placement(self, plan: ShardingPlan, target_devices: list[str], constraints: dict[str, Any]):
        """Optimize shard placement based on device capabilities and constraints."""

        # Simple optimization: prefer devices with more memory for larger shards
        # In production, this would use sophisticated placement algorithms

        sorted_shards = sorted(plan.shards, key=lambda s: s.memory_mb, reverse=True)
        sorted_devices = sorted(target_devices, key=lambda d: self._get_device_memory(d), reverse=True)

        # Reassign devices to optimize placement
        for i, shard in enumerate(sorted_shards):
            device_idx = i % len(sorted_devices)
            shard.device_id = sorted_devices[device_idx]

    def _get_device_memory(self, device_id: str) -> float:
        """Get available memory for a device."""

        # Mock device capabilities - in production would query actual devices
        device_memory = {
            "fog_node_1": 8000,
            "fog_node_2": 4000,
            "fog_node_3": 6000,
        }
        return device_memory.get(device_id, 2000)  # Default 2GB

    async def _calculate_metrics(self, plan: ShardingPlan):
        """Calculate performance metrics for the sharding plan."""

        sum(shard.memory_mb for shard in plan.shards)
        max_shard_memory = max(shard.memory_mb for shard in plan.shards)
        min_shard_memory = min(shard.memory_mb for shard in plan.shards)

        # Memory efficiency: how evenly distributed the memory usage is
        plan.memory_efficiency = 1.0 - (max_shard_memory - min_shard_memory) / max_shard_memory

        # Compute balance: how evenly distributed the compute requirements are
        compute_values = [shard.compute_requirement for shard in plan.shards]
        compute_std = np.std(compute_values) if len(compute_values) > 1 else 0.0
        plan.compute_balance_score = max(0.0, 1.0 - compute_std)

        # Estimate latency based on sequential dependencies
        if plan.activation_routing:
            max_path_length = self._calculate_longest_path(plan.activation_routing)
            plan.estimated_latency_ms = max_path_length * 50  # 50ms per hop estimate
        else:
            plan.estimated_latency_ms = 100  # Single step for parallel operations

    def _calculate_longest_path(self, routing: dict[str, list[str]]) -> int:
        """Calculate the longest path through the activation routing graph."""

        # Simple DFS to find longest path
        def dfs(node: str, visited: set[str]) -> int:
            if node in visited:
                return 0

            visited.add(node)
            max_depth = 0

            for neighbor in routing.get(node, []):
                depth = dfs(neighbor, visited.copy())
                max_depth = max(max_depth, depth)

            return max_depth + 1

        max_path = 0
        for start_node in routing.keys():
            path_length = dfs(start_node, set())
            max_path = max(max_path, path_length)

        return max_path


# Import numpy for calculations
try:
    import numpy as np
except ImportError:
    # Fallback implementation without numpy
    class MockNumpy:
        @staticmethod
        def std(values):
            if not values:
                return 0.0
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return variance**0.5

    np = MockNumpy()
