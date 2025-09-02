"""
HRRM Export Adapters
Handles model sharding and fog burst operations for HRRM (Hierarchical Retrieval
and Reasoning Model) export adapters. Enables distributed inference and training
across fog computing networks while maintaining model consistency.

Key Components:
- HRRMShardingEngine: Intelligent model partitioning for distributed execution
- FogBurstAdapter: Coordinates sharded execution across fog nodes
- HRRMResultMerger: Aggregates results from distributed execution
- ConsistencyValidator: Ensures parity with local execution

This bridges HRRM models with fog computing to enable large-scale distributed
inference while preserving model performance and consistency guarantees.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import json
import logging
from typing import Any
from uuid import uuid4

import aiohttp

logger = logging.getLogger(__name__)


class ShardingStrategy(str, Enum):
    """Strategy for sharding HRRM models"""

    LAYER_WISE = "layer_wise"  # Shard by transformer layers
    ATTENTION_HEADS = "attention_heads"  # Shard by attention heads
    EMBEDDING_SPLIT = "embedding_split"  # Split embedding layers
    PIPELINE_PARALLEL = "pipeline"  # Pipeline parallelism
    TENSOR_PARALLEL = "tensor"  # Tensor parallelism
    HYBRID = "hybrid"  # Combination of strategies


class ModelPartitionType(str, Enum):
    """Type of model partition for fog execution"""

    ENCODER = "encoder"  # Encoder-only partition
    DECODER = "decoder"  # Decoder-only partition
    RETRIEVAL = "retrieval"  # Retrieval component
    REASONING = "reasoning"  # Reasoning component
    FULL_MODEL = "full_model"  # Complete model
    EMBEDDINGS = "embeddings"  # Embedding layers only


@dataclass
class ModelShard:
    """Represents a shard of an HRRM model"""

    shard_id: str
    partition_type: ModelPartitionType
    strategy: ShardingStrategy

    # Model components
    model_data: bytes | None = None
    config: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, Any] | None = None

    # Sharding metadata
    layer_range: tuple[int, int] | None = None
    attention_heads: list[int] | None = None
    embedding_dimensions: tuple[int, int] | None = None

    # Execution requirements
    min_memory_gb: float = 1.0
    min_cpu_cores: float = 1.0
    requires_gpu: bool = False
    estimated_inference_time_ms: float = 100.0

    # Dependencies
    depends_on: list[str] = field(default_factory=list)
    provides_to: list[str] = field(default_factory=list)

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class DistributedInferenceTask:
    """Task for distributed HRRM inference"""

    task_id: str
    model_name: str
    input_data: dict[str, Any]
    shards: list[ModelShard]

    # Execution planning
    fog_node_assignments: dict[str, str] = field(default_factory=dict)  # shard_id -> node_id
    execution_strategy: ShardingStrategy = ShardingStrategy.LAYER_WISE

    # Progress tracking
    status: str = "pending"
    completed_shards: list[str] = field(default_factory=list)
    failed_shards: list[str] = field(default_factory=list)

    # Results
    shard_results: dict[str, Any] = field(default_factory=dict)
    final_result: dict[str, Any] | None = None

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None


class HRRMShardingEngine:
    """
    Intelligent model partitioning engine for HRRM models

    Analyzes model architecture and determines optimal sharding strategy
    based on available fog node capabilities and target performance.
    """

    def __init__(self):
        self.supported_strategies = [
            ShardingStrategy.LAYER_WISE,
            ShardingStrategy.ATTENTION_HEADS,
            ShardingStrategy.PIPELINE_PARALLEL,
            ShardingStrategy.HYBRID,
        ]

        # Sharding history for optimization
        self.sharding_history: list[dict[str, Any]] = []

    async def analyze_model_for_sharding(
        self, model_config: dict[str, Any], target_fog_nodes: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Analyze HRRM model architecture for optimal sharding

        Args:
            model_config: Model configuration including architecture details
            target_fog_nodes: Available fog nodes with capabilities

        Returns:
            Analysis with recommended sharding strategy and configuration
        """

        analysis = {
            "model_size_mb": self._estimate_model_size(model_config),
            "total_parameters": self._estimate_parameter_count(model_config),
            "architecture_type": self._determine_architecture_type(model_config),
            "recommended_strategy": None,
            "shard_configurations": [],
            "performance_estimate": {},
        }

        # Analyze fog node capabilities
        node_analysis = self._analyze_fog_node_capabilities(target_fog_nodes)

        # Determine optimal sharding strategy
        strategy_scores = {}

        for strategy in self.supported_strategies:
            score = await self._score_sharding_strategy(strategy, model_config, node_analysis)
            strategy_scores[strategy] = score

        # Select best strategy
        best_strategy = max(strategy_scores.keys(), key=lambda s: strategy_scores[s])
        analysis["recommended_strategy"] = best_strategy
        analysis["strategy_scores"] = {s.value: score for s, score in strategy_scores.items()}

        # Generate shard configurations for best strategy
        shard_configs = await self._generate_shard_configurations(best_strategy, model_config, node_analysis)
        analysis["shard_configurations"] = shard_configs

        # Estimate performance
        analysis["performance_estimate"] = self._estimate_distributed_performance(shard_configs, node_analysis)

        return analysis

    async def create_model_shards(
        self,
        model_data: bytes,
        model_config: dict[str, Any],
        strategy: ShardingStrategy,
        shard_configurations: list[dict[str, Any]],
    ) -> list[ModelShard]:
        """
        Create model shards based on sharding strategy

        Args:
            model_data: Serialized model data
            model_config: Model configuration
            strategy: Sharding strategy to use
            shard_configurations: Shard configuration specifications

        Returns:
            List of model shards ready for distributed execution
        """

        shards = []

        try:
            # Load model for sharding (simplified - would use actual model loading)
            {"config": model_config, "data_size": len(model_data), "strategy": strategy}

            for i, shard_config in enumerate(shard_configurations):
                shard = ModelShard(
                    shard_id=f"shard_{i}_{uuid4().hex[:8]}",
                    partition_type=ModelPartitionType(shard_config["partition_type"]),
                    strategy=strategy,
                    config=shard_config,
                    min_memory_gb=shard_config.get("min_memory_gb", 1.0),
                    min_cpu_cores=shard_config.get("min_cpu_cores", 1.0),
                    requires_gpu=shard_config.get("requires_gpu", False),
                    estimated_inference_time_ms=shard_config.get("estimated_time_ms", 100.0),
                )

                # Extract shard-specific data based on strategy
                if strategy == ShardingStrategy.LAYER_WISE:
                    shard.layer_range = (shard_config.get("start_layer", 0), shard_config.get("end_layer", 1))
                    shard.model_data = await self._extract_layer_shard(model_data, shard.layer_range)

                elif strategy == ShardingStrategy.ATTENTION_HEADS:
                    shard.attention_heads = shard_config.get("head_indices", [0])
                    shard.model_data = await self._extract_attention_shard(model_data, shard.attention_heads)

                elif strategy == ShardingStrategy.PIPELINE_PARALLEL:
                    shard.layer_range = (shard_config.get("start_layer", 0), shard_config.get("end_layer", 1))
                    shard.depends_on = shard_config.get("depends_on", [])
                    shard.provides_to = shard_config.get("provides_to", [])
                    shard.model_data = await self._extract_pipeline_shard(model_data, shard.layer_range)

                else:
                    # Default: full model shard
                    shard.model_data = model_data

                shards.append(shard)

            logger.info(f"Created {len(shards)} model shards using {strategy.value} strategy")
            return shards

        except Exception as e:
            logger.error(f"Failed to create model shards: {e}")
            raise

    def _estimate_model_size(self, model_config: dict[str, Any]) -> float:
        """Estimate model size in MB"""

        # Simple estimation based on parameters
        hidden_size = model_config.get("hidden_size", 768)
        num_layers = model_config.get("num_layers", 12)
        vocab_size = model_config.get("vocab_size", 50000)

        # Rough parameter count estimation
        params = (
            vocab_size * hidden_size  # Embeddings
            + num_layers * hidden_size * hidden_size * 12  # Transformer layers
            + hidden_size * vocab_size  # Output layer
        )

        # Assume 4 bytes per parameter (float32)
        size_mb = (params * 4) / (1024 * 1024)
        return size_mb

    def _estimate_parameter_count(self, model_config: dict[str, Any]) -> int:
        """Estimate total parameter count"""

        hidden_size = model_config.get("hidden_size", 768)
        num_layers = model_config.get("num_layers", 12)
        vocab_size = model_config.get("vocab_size", 50000)

        return vocab_size * hidden_size + num_layers * hidden_size * hidden_size * 12 + hidden_size * vocab_size

    def _determine_architecture_type(self, model_config: dict[str, Any]) -> str:
        """Determine HRRM architecture type"""

        if "retrieval" in model_config and "reasoning" in model_config:
            return "hierarchical_retrieval_reasoning"
        elif "encoder_decoder" in model_config:
            return "encoder_decoder"
        elif "decoder_only" in model_config:
            return "decoder_only"
        else:
            return "transformer_based"

    def _analyze_fog_node_capabilities(self, fog_nodes: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze collective fog node capabilities"""

        total_cpu = sum(node.get("cpu_cores", 0) for node in fog_nodes)
        total_memory = sum(node.get("memory_gb", 0) for node in fog_nodes)
        gpu_nodes = sum(1 for node in fog_nodes if node.get("gpu_available", False))

        return {
            "node_count": len(fog_nodes),
            "total_cpu_cores": total_cpu,
            "total_memory_gb": total_memory,
            "gpu_nodes": gpu_nodes,
            "avg_cpu_per_node": total_cpu / max(len(fog_nodes), 1),
            "avg_memory_per_node": total_memory / max(len(fog_nodes), 1),
            "nodes": fog_nodes,
        }

    async def _score_sharding_strategy(
        self, strategy: ShardingStrategy, model_config: dict[str, Any], node_analysis: dict[str, Any]
    ) -> float:
        """Score a sharding strategy based on model and node characteristics"""

        score = 0.0

        # Base score factors
        model_size_mb = self._estimate_model_size(model_config)
        node_count = node_analysis["node_count"]
        total_memory = node_analysis["total_memory_gb"]

        if strategy == ShardingStrategy.LAYER_WISE:
            # Good for models with many layers and sufficient nodes
            num_layers = model_config.get("num_layers", 12)
            if num_layers >= node_count:
                score += 40.0
            else:
                score += 20.0

            # Memory efficiency
            if model_size_mb > total_memory * 0.8:
                score += 30.0

        elif strategy == ShardingStrategy.ATTENTION_HEADS:
            # Good for models with many attention heads
            num_heads = model_config.get("num_attention_heads", 8)
            if num_heads >= node_count and node_count <= 8:
                score += 35.0
            else:
                score += 15.0

        elif strategy == ShardingStrategy.PIPELINE_PARALLEL:
            # Good for sequential processing
            if node_count >= 2 and node_count <= 8:
                score += 45.0
            else:
                score += 20.0

            # Penalize if high communication overhead expected
            if node_analysis["avg_memory_per_node"] < 4.0:
                score -= 10.0

        elif strategy == ShardingStrategy.HYBRID:
            # Flexible but complex
            if node_count >= 4 and model_size_mb > 1000:
                score += 50.0
            else:
                score += 25.0

        # General factors
        if node_analysis["gpu_nodes"] > 0:
            score += 10.0

        if total_memory > model_size_mb * 2:
            score += 10.0

        return score

    async def _generate_shard_configurations(
        self, strategy: ShardingStrategy, model_config: dict[str, Any], node_analysis: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate shard configurations for the selected strategy"""

        configurations = []
        node_count = min(node_analysis["node_count"], 8)  # Limit to 8 shards max

        if strategy == ShardingStrategy.LAYER_WISE:
            num_layers = model_config.get("num_layers", 12)
            layers_per_shard = max(1, num_layers // node_count)

            for i in range(node_count):
                start_layer = i * layers_per_shard
                end_layer = min((i + 1) * layers_per_shard, num_layers)

                if start_layer < num_layers:
                    configurations.append(
                        {
                            "partition_type": "encoder" if i < node_count // 2 else "decoder",
                            "start_layer": start_layer,
                            "end_layer": end_layer,
                            "min_memory_gb": 2.0,
                            "min_cpu_cores": 1.0,
                            "estimated_time_ms": 150.0,
                        }
                    )

        elif strategy == ShardingStrategy.ATTENTION_HEADS:
            num_heads = model_config.get("num_attention_heads", 8)
            heads_per_shard = max(1, num_heads // node_count)

            for i in range(min(node_count, num_heads)):
                start_head = i * heads_per_shard
                end_head = min((i + 1) * heads_per_shard, num_heads)

                configurations.append(
                    {
                        "partition_type": "attention",
                        "head_indices": list(range(start_head, end_head)),
                        "min_memory_gb": 1.5,
                        "min_cpu_cores": 1.0,
                        "estimated_time_ms": 120.0,
                    }
                )

        elif strategy == ShardingStrategy.PIPELINE_PARALLEL:
            num_layers = model_config.get("num_layers", 12)
            layers_per_stage = max(1, num_layers // node_count)

            for i in range(node_count):
                start_layer = i * layers_per_stage
                end_layer = min((i + 1) * layers_per_stage, num_layers)

                if start_layer < num_layers:
                    config = {
                        "partition_type": "pipeline_stage",
                        "start_layer": start_layer,
                        "end_layer": end_layer,
                        "min_memory_gb": 3.0,
                        "min_cpu_cores": 2.0,
                        "estimated_time_ms": 200.0,
                        "depends_on": [] if i == 0 else [f"stage_{i-1}"],
                        "provides_to": [] if i == node_count - 1 else [f"stage_{i+1}"],
                    }
                    configurations.append(config)

        else:
            # Default: single shard per node
            for i in range(node_count):
                configurations.append(
                    {
                        "partition_type": "full_model",
                        "min_memory_gb": 4.0,
                        "min_cpu_cores": 2.0,
                        "estimated_time_ms": 300.0,
                    }
                )

        return configurations

    def _estimate_distributed_performance(
        self, shard_configs: list[dict[str, Any]], node_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Estimate performance of distributed execution"""

        # Simple performance estimation
        total_time_ms = max(config.get("estimated_time_ms", 100.0) for config in shard_configs)

        # Add communication overhead
        communication_overhead = len(shard_configs) * 50.0  # 50ms per shard communication
        total_time_ms += communication_overhead

        # Memory efficiency
        total_memory_needed = sum(config.get("min_memory_gb", 1.0) for config in shard_configs)

        return {
            "estimated_latency_ms": total_time_ms,
            "memory_efficiency": min(1.0, node_analysis["total_memory_gb"] / total_memory_needed),
            "parallelization_factor": len(shard_configs),
            "communication_overhead_ms": communication_overhead,
            "expected_speedup": max(1.0, len(shard_configs) * 0.7),  # 70% efficiency
        }

    async def _extract_layer_shard(self, model_data: bytes, layer_range: tuple[int, int]) -> bytes:
        """Extract layers for layer-wise sharding (simplified implementation)"""

        # In production, this would extract specific transformer layers
        # For now, return a portion of the model data
        start_ratio = layer_range[0] / 12.0  # Assume 12 layers
        end_ratio = layer_range[1] / 12.0

        start_byte = int(len(model_data) * start_ratio)
        end_byte = int(len(model_data) * end_ratio)

        return model_data[start_byte:end_byte]

    async def _extract_attention_shard(self, model_data: bytes, head_indices: list[int]) -> bytes:
        """Extract attention heads for attention-based sharding"""

        # Simplified: return proportional data based on head count
        head_ratio = len(head_indices) / 8.0  # Assume 8 heads
        shard_size = int(len(model_data) * head_ratio)

        return model_data[:shard_size]

    async def _extract_pipeline_shard(self, model_data: bytes, layer_range: tuple[int, int]) -> bytes:
        """Extract pipeline stage for pipeline parallelism"""

        # Similar to layer-wise but with pipeline metadata
        return await self._extract_layer_shard(model_data, layer_range)


class FogBurstAdapter:
    """
    Coordinates sharded HRRM model execution across fog nodes

    Manages task distribution, inter-shard communication, and result aggregation
    for distributed inference and training operations.
    """

    def __init__(self, fog_gateway_url: str = "http://localhost:8080"):
        self.fog_gateway_url = fog_gateway_url.rstrip("/")

        # Task management
        self.active_tasks: dict[str, DistributedInferenceTask] = {}
        self.sharding_engine = HRRMShardingEngine()

        # Performance tracking
        self.distributed_inference_count = 0
        self.avg_latency_ms = 0.0
        self.success_rate = 0.95

    async def burst_to_fog(
        self,
        model_name: str,
        model_data: bytes,
        model_config: dict[str, Any],
        input_data: dict[str, Any],
        target_fog_nodes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Execute HRRM model inference distributed across fog nodes

        Args:
            model_name: Name of the HRRM model
            model_data: Serialized model data
            model_config: Model configuration
            input_data: Input data for inference
            target_fog_nodes: Specific fog nodes to use (optional)

        Returns:
            Distributed inference results
        """

        task_id = str(uuid4())

        logger.info(f"Starting fog burst for model {model_name} (task: {task_id})")

        try:
            # Discover available fog nodes if not specified
            if not target_fog_nodes:
                target_fog_nodes = await self._discover_fog_nodes()

            if not target_fog_nodes:
                raise Exception("No suitable fog nodes available")

            # Analyze model for sharding
            sharding_analysis = await self.sharding_engine.analyze_model_for_sharding(model_config, target_fog_nodes)

            # Create model shards
            shards = await self.sharding_engine.create_model_shards(
                model_data,
                model_config,
                sharding_analysis["recommended_strategy"],
                sharding_analysis["shard_configurations"],
            )

            # Create distributed task
            task = DistributedInferenceTask(
                task_id=task_id,
                model_name=model_name,
                input_data=input_data,
                shards=shards,
                execution_strategy=sharding_analysis["recommended_strategy"],
            )

            self.active_tasks[task_id] = task

            # Plan execution across fog nodes
            execution_plan = await self._plan_distributed_execution(task, target_fog_nodes)

            # Execute shards across fog nodes
            task.status = "running"
            task.started_at = datetime.now(UTC)

            results = await self._execute_distributed_inference(task, execution_plan)

            # Merge results
            final_result = await self._merge_distributed_results(task, results)

            task.status = "completed"
            task.completed_at = datetime.now(UTC)
            task.final_result = final_result

            self.distributed_inference_count += 1

            logger.info(f"Fog burst completed for {model_name} (task: {task_id})")

            return {
                "status": "success",
                "task_id": task_id,
                "model_name": model_name,
                "result": final_result,
                "execution_metadata": {
                    "shards_executed": len(shards),
                    "fog_nodes_used": len(execution_plan["node_assignments"]),
                    "strategy": sharding_analysis["recommended_strategy"].value,
                    "total_time_ms": (
                        (task.completed_at - task.started_at).total_seconds() * 1000
                        if task.completed_at and task.started_at
                        else 0
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Fog burst failed for {model_name}: {e}")

            if task_id in self.active_tasks:
                self.active_tasks[task_id].status = "failed"
                del self.active_tasks[task_id]

            return {"status": "error", "message": str(e), "task_id": task_id, "model_name": model_name}

    async def _discover_fog_nodes(self) -> list[dict[str, Any]]:
        """Discover fog nodes suitable for HRRM model execution"""

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.fog_gateway_url}/v1/fog/nodes", params={"capabilities": "inference,pytorch,gpu"}
                ) as response:

                    if response.status == 200:
                        nodes_data = await response.json()
                        return nodes_data.get("nodes", [])
                    else:
                        logger.warning(f"Failed to discover fog nodes: {response.status}")
                        return []

        except Exception as e:
            logger.error(f"Error discovering fog nodes: {e}")
            return []

    async def _plan_distributed_execution(
        self, task: DistributedInferenceTask, fog_nodes: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Plan how to distribute shards across fog nodes"""

        plan = {
            "node_assignments": {},  # shard_id -> node_id
            "execution_order": [],
            "communication_plan": {},
            "resource_allocation": {},
        }

        # Simple greedy assignment for now
        available_nodes = fog_nodes.copy()

        for shard in task.shards:
            best_node = None
            best_score = -1.0

            for node in available_nodes:
                score = self._score_node_for_shard(shard, node)
                if score > best_score:
                    best_score = score
                    best_node = node

            if best_node:
                plan["node_assignments"][shard.shard_id] = best_node["node_id"]
                plan["resource_allocation"][shard.shard_id] = {
                    "node_id": best_node["node_id"],
                    "cpu_cores": shard.min_cpu_cores,
                    "memory_gb": shard.min_memory_gb,
                    "gpu_required": shard.requires_gpu,
                }

                # Remove node from available if it's fully allocated
                if (
                    len([s for s in task.shards if plan["node_assignments"].get(s.shard_id) == best_node["node_id"]])
                    >= 2
                ):
                    available_nodes.remove(best_node)

        # Determine execution order based on dependencies
        if task.execution_strategy == ShardingStrategy.PIPELINE_PARALLEL:
            plan["execution_order"] = self._determine_pipeline_order(task.shards)
        else:
            plan["execution_order"] = [shard.shard_id for shard in task.shards]

        return plan

    def _score_node_for_shard(self, shard: ModelShard, node: dict[str, Any]) -> float:
        """Score how well a fog node matches a shard's requirements"""

        score = 0.0

        # Resource matching
        node_cpu = node.get("resources", {}).get("cpu_cores", 0)
        node_memory = node.get("resources", {}).get("memory_gb", 0)
        node_gpu = node.get("resources", {}).get("gpu_available", False)

        if node_cpu >= shard.min_cpu_cores:
            score += 20.0
        else:
            return 0.0  # Hard requirement

        if node_memory >= shard.min_memory_gb:
            score += 20.0
        else:
            return 0.0  # Hard requirement

        if shard.requires_gpu and node_gpu:
            score += 30.0
        elif shard.requires_gpu and not node_gpu:
            return 0.0  # Hard requirement

        # Performance factors
        current_load = node.get("metrics", {}).get("current_load", 0.5)
        score += (1.0 - current_load) * 20.0

        # Capability matching
        capabilities = node.get("capabilities", {})
        if capabilities.get("pytorch", False):
            score += 10.0

        return score

    def _determine_pipeline_order(self, shards: list[ModelShard]) -> list[str]:
        """Determine execution order for pipeline parallelism"""

        # Build dependency graph
        dependency_map = {}
        for shard in shards:
            dependency_map[shard.shard_id] = shard.depends_on

        # Topological sort
        ordered = []
        visited = set()

        def visit(shard_id: str):
            if shard_id in visited:
                return

            visited.add(shard_id)

            # Visit dependencies first
            for dep in dependency_map.get(shard_id, []):
                if dep in dependency_map:
                    visit(dep)

            ordered.append(shard_id)

        for shard in shards:
            visit(shard.shard_id)

        return ordered

    async def _execute_distributed_inference(
        self, task: DistributedInferenceTask, execution_plan: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute inference across distributed shards"""

        results = {}

        if task.execution_strategy == ShardingStrategy.PIPELINE_PARALLEL:
            # Sequential execution for pipeline
            intermediate_results = {}

            for shard_id in execution_plan["execution_order"]:
                shard = next(s for s in task.shards if s.shard_id == shard_id)
                node_id = execution_plan["node_assignments"][shard_id]

                # Prepare input (including outputs from previous stages)
                shard_input = task.input_data.copy()
                for dep_id in shard.depends_on:
                    if dep_id in intermediate_results:
                        shard_input.update(intermediate_results[dep_id])

                # Execute shard
                result = await self._execute_shard_on_node(shard, shard_input, node_id)
                results[shard_id] = result
                intermediate_results[shard_id] = result.get("output", {})

                if result.get("status") != "success":
                    raise Exception(f"Shard {shard_id} execution failed: {result.get('message')}")

        else:
            # Parallel execution for other strategies
            async def execute_shard_task(shard: ModelShard) -> tuple[str, dict[str, Any]]:
                node_id = execution_plan["node_assignments"][shard.shard_id]
                result = await self._execute_shard_on_node(shard, task.input_data, node_id)
                return shard.shard_id, result

            # Execute all shards in parallel
            shard_tasks = [execute_shard_task(shard) for shard in task.shards]
            shard_results = await asyncio.gather(*shard_tasks, return_exceptions=True)

            for shard_result in shard_results:
                if isinstance(shard_result, Exception):
                    raise shard_result
                else:
                    shard_id, result = shard_result
                    results[shard_id] = result

        return results

    async def _execute_shard_on_node(
        self, shard: ModelShard, input_data: dict[str, Any], node_id: str
    ) -> dict[str, Any]:
        """Execute a model shard on a specific fog node"""

        try:
            # Prepare job specification for fog execution
            job_spec = {
                "namespace": "hrrm-inference",
                "runtime": "wasi",
                "image": "hrrm-shard-executor:latest",
                "args": [
                    "--shard-id",
                    shard.shard_id,
                    "--partition-type",
                    shard.partition_type.value,
                    "--input",
                    json.dumps(input_data),
                ],
                "env": {
                    "SHARD_ID": shard.shard_id,
                    "PARTITION_TYPE": shard.partition_type.value,
                    "MODEL_CONFIG": json.dumps(shard.config),
                },
                "resources": {
                    "cpu_cores": shard.min_cpu_cores,
                    "memory_gb": shard.min_memory_gb,
                    "gpu_required": shard.requires_gpu,
                    "max_duration_s": 300,
                },
                "input_data": shard.model_data,  # Model shard data
                "metadata": {"task_type": "hrrm_shard_inference", "shard_id": shard.shard_id, "target_node": node_id},
            }

            async with aiohttp.ClientSession() as session:
                # Submit job to fog gateway
                async with session.post(f"{self.fog_gateway_url}/v1/fog/jobs", json=job_spec) as response:

                    if response.status == 201:
                        job_data = await response.json()
                        job_id = job_data["job_id"]

                        # Wait for completion
                        return await self._wait_for_shard_completion(job_id)
                    else:
                        error_text = await response.text()
                        return {"status": "error", "message": f"Failed to submit shard job: {error_text}"}

        except Exception as e:
            return {"status": "error", "message": f"Shard execution error: {str(e)}"}

    async def _wait_for_shard_completion(self, job_id: str) -> dict[str, Any]:
        """Wait for shard execution to complete"""

        timeout = 300  # 5 minutes
        start_time = asyncio.get_event_loop().time()

        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.fog_gateway_url}/v1/fog/jobs/{job_id}/status") as response:

                        if response.status == 200:
                            status_data = await response.json()
                            job_status = status_data.get("status")

                            if job_status == "completed":
                                # Get results
                                async with session.get(
                                    f"{self.fog_gateway_url}/v1/fog/jobs/{job_id}/result"
                                ) as result_response:

                                    if result_response.status == 200:
                                        result_data = await result_response.json()

                                        try:
                                            output = json.loads(result_data.get("stdout", "{}"))
                                            return {
                                                "status": "success",
                                                "output": output,
                                                "job_id": job_id,
                                                "execution_time_ms": result_data.get("duration_ms", 0),
                                            }
                                        except json.JSONDecodeError:
                                            return {"status": "error", "message": "Failed to parse shard output"}

                            elif job_status in ["failed", "cancelled", "timeout"]:
                                return {"status": "error", "message": f"Shard job {job_status}"}

                            # Check timeout
                            if asyncio.get_event_loop().time() - start_time > timeout:
                                return {"status": "error", "message": f"Shard execution timed out after {timeout}s"}

                            # Wait before next check
                            await asyncio.sleep(2)
                        else:
                            return {"status": "error", "message": f"Failed to check shard status: {response.status}"}

            except Exception as e:
                return {"status": "error", "message": f"Error waiting for shard: {str(e)}"}

    async def _merge_distributed_results(
        self, task: DistributedInferenceTask, shard_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge results from distributed shard execution"""

        merger = HRRMResultMerger()
        return await merger.merge_shard_results(task.shards, shard_results, task.execution_strategy)


class HRRMResultMerger:
    """
    Aggregates results from distributed HRRM shard execution

    Handles different merging strategies based on the sharding approach
    and ensures consistent final results across distribution strategies.
    """

    async def merge_shard_results(
        self, shards: list[ModelShard], shard_results: dict[str, Any], strategy: ShardingStrategy
    ) -> dict[str, Any]:
        """
        Merge results from distributed shard execution

        Args:
            shards: List of model shards
            shard_results: Results from each shard execution
            strategy: Sharding strategy used

        Returns:
            Merged final result
        """

        if strategy == ShardingStrategy.PIPELINE_PARALLEL:
            return await self._merge_pipeline_results(shards, shard_results)
        elif strategy == ShardingStrategy.LAYER_WISE:
            return await self._merge_layer_results(shards, shard_results)
        elif strategy == ShardingStrategy.ATTENTION_HEADS:
            return await self._merge_attention_results(shards, shard_results)
        else:
            return await self._merge_parallel_results(shards, shard_results)

    async def _merge_pipeline_results(self, shards: list[ModelShard], shard_results: dict[str, Any]) -> dict[str, Any]:
        """Merge results from pipeline parallel execution"""

        # For pipeline, the final result is from the last stage
        final_shard = None
        for shard in shards:
            if not shard.provides_to:  # Last stage has no outputs
                final_shard = shard
                break

        if final_shard and final_shard.shard_id in shard_results:
            final_result = shard_results[final_shard.shard_id]

            return {
                "model_output": final_result.get("output", {}),
                "confidence": final_result.get("output", {}).get("confidence", 0.0),
                "execution_metadata": {
                    "strategy": "pipeline_parallel",
                    "stages_executed": len(shard_results),
                    "total_execution_time_ms": sum(
                        result.get("execution_time_ms", 0) for result in shard_results.values()
                    ),
                },
            }
        else:
            return {"model_output": {}, "confidence": 0.0, "error": "Failed to find final pipeline stage result"}

    async def _merge_layer_results(self, shards: list[ModelShard], shard_results: dict[str, Any]) -> dict[str, Any]:
        """Merge results from layer-wise parallel execution"""

        # Combine outputs from all layers
        combined_outputs = {}
        total_confidence = 0.0
        successful_shards = 0

        for shard in shards:
            if shard.shard_id in shard_results:
                result = shard_results[shard.shard_id]
                if result.get("status") == "success":
                    output = result.get("output", {})

                    # Merge layer outputs
                    for key, value in output.items():
                        if key not in combined_outputs:
                            combined_outputs[key] = []
                        combined_outputs[key].append(value)

                    # Accumulate confidence
                    total_confidence += output.get("confidence", 0.0)
                    successful_shards += 1

        # Average confidence across successful shards
        avg_confidence = total_confidence / max(successful_shards, 1)

        # Combine layer outputs (simplified)
        final_output = {}
        for key, values in combined_outputs.items():
            if isinstance(values[0], int | float):
                final_output[key] = sum(values) / len(values)
            elif isinstance(values[0], list):
                final_output[key] = [item for sublist in values for item in sublist]
            else:
                final_output[key] = values[-1]  # Take last value

        return {
            "model_output": final_output,
            "confidence": avg_confidence,
            "execution_metadata": {
                "strategy": "layer_wise",
                "layers_executed": successful_shards,
                "total_execution_time_ms": max(result.get("execution_time_ms", 0) for result in shard_results.values()),
            },
        }

    async def _merge_attention_results(self, shards: list[ModelShard], shard_results: dict[str, Any]) -> dict[str, Any]:
        """Merge results from attention head parallel execution"""

        # Combine attention outputs
        attention_outputs = []
        total_confidence = 0.0
        successful_shards = 0

        for shard in shards:
            if shard.shard_id in shard_results:
                result = shard_results[shard.shard_id]
                if result.get("status") == "success":
                    output = result.get("output", {})
                    attention_outputs.append(output)
                    total_confidence += output.get("confidence", 0.0)
                    successful_shards += 1

        # Merge attention outputs (simplified averaging)
        if attention_outputs:
            merged_output = attention_outputs[0].copy()
            merged_output["confidence"] = total_confidence / successful_shards

            # Average numerical outputs
            for key in merged_output:
                if isinstance(merged_output[key], int | float):
                    values = [out.get(key, 0) for out in attention_outputs if key in out]
                    merged_output[key] = sum(values) / len(values)
        else:
            merged_output = {}

        return {
            "model_output": merged_output,
            "confidence": total_confidence / max(successful_shards, 1),
            "execution_metadata": {
                "strategy": "attention_heads",
                "heads_executed": successful_shards,
                "total_execution_time_ms": max(result.get("execution_time_ms", 0) for result in shard_results.values()),
            },
        }

    async def _merge_parallel_results(self, shards: list[ModelShard], shard_results: dict[str, Any]) -> dict[str, Any]:
        """Merge results from general parallel execution"""

        # Simple ensemble approach - average results
        all_outputs = []
        total_confidence = 0.0
        successful_shards = 0

        for shard in shards:
            if shard.shard_id in shard_results:
                result = shard_results[shard.shard_id]
                if result.get("status") == "success":
                    output = result.get("output", {})
                    all_outputs.append(output)
                    total_confidence += output.get("confidence", 0.0)
                    successful_shards += 1

        if all_outputs:
            # Ensemble the outputs
            merged_output = all_outputs[0].copy()
            merged_output["confidence"] = total_confidence / successful_shards
        else:
            merged_output = {}

        return {
            "model_output": merged_output,
            "confidence": total_confidence / max(successful_shards, 1),
            "execution_metadata": {
                "strategy": "parallel",
                "shards_executed": successful_shards,
                "total_execution_time_ms": max(result.get("execution_time_ms", 0) for result in shard_results.values()),
            },
        }


class ConsistencyValidator:
    """
    Ensures parity between local and distributed HRRM execution

    Validates that distributed inference produces results within acceptable
    tolerance of local execution to maintain model consistency.
    """

    def __init__(self, tolerance: float = 1e-3):
        self.tolerance = tolerance

    async def validate_distributed_parity(
        self, local_result: dict[str, Any], distributed_result: dict[str, Any], model_name: str
    ) -> dict[str, Any]:
        """
        Validate distributed result matches local execution

        Args:
            local_result: Local inference result
            distributed_result: Distributed inference result
            model_name: Name of the model

        Returns:
            Validation result with parity assessment
        """

        validation = {
            "model_name": model_name,
            "timestamp": datetime.now(UTC).isoformat(),
            "parity_passed": False,
            "tolerance": self.tolerance,
            "differences": {},
            "assessment": "FAILED",
        }

        try:
            local_output = local_result.get("model_output", {})
            distributed_output = distributed_result.get("model_output", {})

            # Compare confidence scores
            local_confidence = local_output.get("confidence", 0.0)
            distributed_confidence = distributed_output.get("confidence", 0.0)

            confidence_diff = abs(distributed_confidence - local_confidence)
            validation["differences"]["confidence"] = {
                "local": local_confidence,
                "distributed": distributed_confidence,
                "absolute_diff": confidence_diff,
                "relative_diff": confidence_diff / max(local_confidence, 1e-6),
                "within_tolerance": confidence_diff <= self.tolerance,
            }

            # Compare other numerical outputs
            common_keys = set(local_output.keys()) & set(distributed_output.keys())
            max_difference = confidence_diff

            for key in common_keys:
                local_val = local_output[key]
                distributed_val = distributed_output[key]

                if isinstance(local_val, int | float) and isinstance(distributed_val, int | float):
                    abs_diff = abs(distributed_val - local_val)
                    rel_diff = abs_diff / max(abs(local_val), 1e-6)

                    validation["differences"][key] = {
                        "local": local_val,
                        "distributed": distributed_val,
                        "absolute_diff": abs_diff,
                        "relative_diff": rel_diff,
                        "within_tolerance": rel_diff <= self.tolerance,
                    }

                    max_difference = max(max_difference, rel_diff)

            validation["max_difference"] = max_difference
            validation["parity_passed"] = max_difference <= self.tolerance

            if validation["parity_passed"]:
                if max_difference <= self.tolerance * 0.1:
                    validation["assessment"] = "EXCELLENT"
                elif max_difference <= self.tolerance * 0.5:
                    validation["assessment"] = "GOOD"
                else:
                    validation["assessment"] = "ACCEPTABLE"

            return validation

        except Exception as e:
            validation["error"] = str(e)
            logger.error(f"Consistency validation failed for {model_name}: {e}")
            return validation


# Export main classes
__all__ = [
    "HRRMShardingEngine",
    "FogBurstAdapter",
    "HRRMResultMerger",
    "ConsistencyValidator",
    "ModelShard",
    "DistributedInferenceTask",
    "ShardingStrategy",
    "ModelPartitionType",
]
