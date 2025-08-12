"""Model Merge Operators with Evolutionary Strategies
Sprint R-4+AF1: Agent Forge Phase 1 - Task B.2.
"""

import asyncio
import copy
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import wandb

logger = logging.getLogger(__name__)


@dataclass
class MergeResult:
    """Result of model merging operation."""

    merge_id: str
    parent1_id: str
    parent2_id: str
    merge_strategy: str
    merge_config: dict[str, Any]
    success: bool
    merged_model: Any | None = None
    performance_metrics: dict[str, float] = None
    merge_time: float = 0.0
    model_size_mb: float = 0.0
    parameters_changed: int = 0
    merge_quality_score: float = 0.0
    timestamp: str = ""


@dataclass
class MergeConfig:
    """Configuration for merge operations."""

    # Linear merge parameters
    linear_alpha: float = 0.5  # Weight for first model

    # SLERP parameters
    slerp_steps: int = 10
    slerp_interpolation_factor: float = 0.5

    # DARE parameters
    dare_drop_rate: float = 0.1
    dare_rescale: bool = True
    dare_random_seed: int = 42

    # Evolutionary parameters
    mutation_strength: float = 0.05
    parameter_selection_prob: float = 0.3
    layer_wise_blending: bool = True

    # Quality control
    similarity_threshold: float = 0.95  # Max similarity to parents
    divergence_penalty: float = 0.1
    performance_weight: float = 0.8


class MergeOperator:
    """Advanced model merging with evolutionary strategies and W&B tracking."""

    def __init__(self, project_name: str = "agent-forge") -> None:
        self.project_name = project_name
        self.merge_history = []
        self.merge_strategies = {
            "linear": self.linear_merge,
            "slerp": self.slerp_merge,
            "dare": self.dare_merge,
            "evolutionary": self.evolutionary_merge,
            "layer_wise": self.layer_wise_merge,
            "attention_guided": self.attention_guided_merge,
        }

        # Performance tracking
        self.strategy_performance = {strategy: [] for strategy in self.merge_strategies}
        self.merge_analytics = {
            "total_merges": 0,
            "successful_merges": 0,
            "avg_merge_time": 0.0,
            "best_merge_quality": 0.0,
        }

        # Initialize tracking
        self.initialize_merge_tracking()

    def initialize_merge_tracking(self) -> None:
        """Initialize W&B tracking for merge operations."""
        try:
            # Use existing wandb run or create new one
            if wandb.run is None:
                wandb.init(
                    project=self.project_name,
                    job_type="model_merging",
                    config={
                        "merge_version": "1.0.0",
                        "supported_strategies": list(self.merge_strategies.keys()),
                        "quality_metrics": ["similarity", "performance", "diversity"],
                        "optimization_target": "math_tutoring_effectiveness",
                    },
                )

            logger.info("Model merge tracking initialized")

        except Exception as e:
            logger.exception(f"Failed to initialize merge tracking: {e}")

    async def merge_population(
        self,
        population: list[dict[str, Any]],
        generation: int,
        merge_config: MergeConfig = None,
    ) -> list[MergeResult]:
        """Create next generation through strategic merging."""
        logger.info(f"Merging population for generation {generation}")

        if not merge_config:
            merge_config = MergeConfig()

        offspring_results = []

        # Select parent pairs for merging
        parent_pairs = self.select_merge_pairs(population)

        # Try different merge strategies for each pair
        for parent1, parent2 in parent_pairs:
            # Choose merge strategy based on performance history
            strategy = self.select_optimal_strategy(parent1, parent2)

            logger.info(
                f"Merging {parent1['individual_id'][:8]} + {parent2['individual_id'][:8]} using {strategy}"
            )

            try:
                # Perform merge
                merge_result = await self.perform_merge(
                    parent1, parent2, strategy, merge_config, generation
                )

                if merge_result.success:
                    offspring_results.append(merge_result)

                    # Log merge success
                    wandb.log(
                        {
                            "merge/strategy": strategy,
                            "merge/parent1_fitness": parent1.get("fitness_score", 0.0),
                            "merge/parent2_fitness": parent2.get("fitness_score", 0.0),
                            "merge/quality_score": merge_result.merge_quality_score,
                            "merge/merge_time": merge_result.merge_time,
                            "merge/success": True,
                            "generation": generation,
                        }
                    )

                    logger.info(
                        f"Successful merge: quality={merge_result.merge_quality_score:.3f}"
                    )

                else:
                    # Log merge failure
                    wandb.log(
                        {
                            "merge/strategy": strategy,
                            "merge/success": False,
                            "merge/failure_reason": "merge_failed",
                            "generation": generation,
                        }
                    )

                    logger.warning(f"Merge failed for strategy {strategy}")

            except Exception as e:
                logger.exception(f"Error in merge operation: {e}")
                continue

        # Update analytics
        self.update_merge_analytics(offspring_results)

        # Log generation summary
        wandb.log(
            {
                f"generation/{generation}/merges_attempted": len(parent_pairs),
                f"generation/{generation}/merges_successful": len(offspring_results),
                f"generation/{generation}/avg_merge_quality": (
                    np.mean([r.merge_quality_score for r in offspring_results])
                    if offspring_results
                    else 0
                ),
                "generation": generation,
            }
        )

        logger.info(
            f"Generation {generation} merging complete: {len(offspring_results)} successful merges"
        )

        return offspring_results

    def select_merge_pairs(
        self, population: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Select optimal pairs for merging based on fitness and diversity."""
        pairs = []

        # Sort population by fitness
        sorted_pop = sorted(
            population, key=lambda x: x.get("fitness_score", 0), reverse=True
        )

        # Strategy 1: Best with diverse partners
        best_individual = sorted_pop[0]
        for i in range(1, min(4, len(sorted_pop))):  # Pair best with top 3 others
            pairs.append((best_individual, sorted_pop[i]))

        # Strategy 2: Complementary fitness pairs
        mid_performers = sorted_pop[len(sorted_pop) // 4 : 3 * len(sorted_pop) // 4]
        if len(mid_performers) >= 2:
            for i in range(0, len(mid_performers) - 1, 2):
                pairs.append((mid_performers[i], mid_performers[i + 1]))

        # Strategy 3: Random diversity pairs
        remaining = sorted_pop[len(pairs) * 2 :]
        while len(remaining) >= 2:
            parent1 = remaining.pop(np.random.randint(len(remaining)))
            parent2 = remaining.pop(np.random.randint(len(remaining)))
            pairs.append((parent1, parent2))

            if len(pairs) >= 6:  # Limit total pairs
                break

        return pairs

    def select_optimal_strategy(
        self, parent1: dict[str, Any], parent2: dict[str, Any]
    ) -> str:
        """Select optimal merge strategy based on parent characteristics and history."""
        # Get strategy performance history
        strategy_scores = {}
        for strategy, history in self.strategy_performance.items():
            if history:
                strategy_scores[strategy] = np.mean(history[-10:])  # Recent performance
            else:
                strategy_scores[strategy] = 0.5  # Default score

        # Consider parent characteristics
        parent1_fitness = parent1.get("fitness_score", 0.5)
        parent2_fitness = parent2.get("fitness_score", 0.5)
        fitness_gap = abs(parent1_fitness - parent2_fitness)

        # Strategy selection logic
        if fitness_gap < 0.1:
            # Similar fitness - use linear or evolutionary
            return (
                "linear"
                if strategy_scores["linear"] > strategy_scores["evolutionary"]
                else "evolutionary"
            )
        if fitness_gap > 0.3:
            # Very different fitness - use DARE or attention-guided
            return (
                "dare"
                if strategy_scores["dare"] > strategy_scores["attention_guided"]
                else "attention_guided"
            )
        # Moderate difference - use SLERP or layer-wise
        return (
            "slerp"
            if strategy_scores["slerp"] > strategy_scores["layer_wise"]
            else "layer_wise"
        )

    async def perform_merge(
        self,
        parent1: dict[str, Any],
        parent2: dict[str, Any],
        strategy: str,
        merge_config: MergeConfig,
        generation: int,
    ) -> MergeResult:
        """Perform model merge using specified strategy."""
        start_time = asyncio.get_event_loop().time()

        # Generate merge ID
        merge_id = f"merge_{parent1['individual_id'][:6]}_{parent2['individual_id'][:6]}_{strategy}_{generation}"

        # Initialize result
        result = MergeResult(
            merge_id=merge_id,
            parent1_id=parent1["individual_id"],
            parent2_id=parent2["individual_id"],
            merge_strategy=strategy,
            merge_config=asdict(merge_config),
            success=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        try:
            # Get models (assuming they're loaded or can be loaded)
            model1 = parent1.get("model")  # In practice, load from cache or disk
            model2 = parent2.get("model")

            if model1 is None or model2 is None:
                logger.error("Parent models not available for merging")
                return result

            # Perform merge using selected strategy
            merge_func = self.merge_strategies.get(strategy)
            if merge_func:
                merged_model = await merge_func(model1, model2, merge_config)

                if merged_model is not None:
                    # Calculate merge metrics
                    merge_time = asyncio.get_event_loop().time() - start_time

                    # Estimate model size
                    model_size_mb = (
                        sum(p.numel() for p in merged_model.parameters())
                        * 4
                        / (1024 * 1024)
                    )

                    # Calculate quality score
                    quality_score = await self.calculate_merge_quality(
                        merged_model, model1, model2, parent1, parent2
                    )

                    # Count changed parameters
                    parameters_changed = await self.count_parameter_changes(
                        merged_model, model1, model2
                    )

                    # Update result
                    result.success = True
                    result.merged_model = merged_model
                    result.merge_time = merge_time
                    result.model_size_mb = model_size_mb
                    result.merge_quality_score = quality_score
                    result.parameters_changed = parameters_changed
                    result.performance_metrics = {
                        "merge_efficiency": 1.0 / max(merge_time, 0.001),
                        "size_efficiency": min(1.0, 500.0 / max(model_size_mb, 1.0)),
                        "parameter_diversity": parameters_changed
                        / max(sum(p.numel() for p in merged_model.parameters()), 1),
                    }

        except Exception as e:
            logger.exception(f"Error in {strategy} merge: {e}")
            result.success = False

        # Store result
        self.merge_history.append(result)

        return result

    async def linear_merge(self, model1, model2, config: MergeConfig):
        """Linear interpolation merge: merged = α * model1 + (1-α) * model2."""
        try:
            alpha = config.linear_alpha

            # Clone first model as base
            merged_model = copy.deepcopy(model1)

            # Linear interpolation of parameters
            with torch.no_grad():
                for (name1, param1), (name2, param2) in zip(
                    model1.named_parameters(), model2.named_parameters(), strict=False
                ):
                    if name1 == name2 and param1.shape == param2.shape:
                        # Linear interpolation
                        merged_param = alpha * param1 + (1 - alpha) * param2

                        # Update merged model parameter
                        merged_model.state_dict()[name1].copy_(merged_param)

            return merged_model

        except Exception as e:
            logger.exception(f"Error in linear merge: {e}")
            return None

    async def slerp_merge(self, model1, model2, config: MergeConfig):
        """Spherical Linear Interpolation merge for smoother blending."""
        try:
            t = config.slerp_interpolation_factor

            merged_model = copy.deepcopy(model1)

            with torch.no_grad():
                for (name1, param1), (name2, param2) in zip(
                    model1.named_parameters(), model2.named_parameters(), strict=False
                ):
                    if name1 == name2 and param1.shape == param2.shape:
                        # Flatten parameters for SLERP
                        p1_flat = param1.flatten()
                        p2_flat = param2.flatten()

                        # Calculate angle between vectors
                        dot_product = torch.dot(p1_flat, p2_flat)
                        norm_product = torch.norm(p1_flat) * torch.norm(p2_flat)

                        if norm_product > 1e-8:
                            cos_angle = torch.clamp(
                                dot_product / norm_product, -1.0, 1.0
                            )
                            angle = torch.acos(cos_angle)

                            if angle > 1e-6:  # Avoid division by zero
                                # SLERP formula
                                sin_angle = torch.sin(angle)
                                weight1 = torch.sin((1 - t) * angle) / sin_angle
                                weight2 = torch.sin(t * angle) / sin_angle

                                merged_flat = weight1 * p1_flat + weight2 * p2_flat
                            else:
                                # Linear interpolation for nearly parallel vectors
                                merged_flat = (1 - t) * p1_flat + t * p2_flat
                        else:
                            # Fallback to linear interpolation
                            merged_flat = (1 - t) * p1_flat + t * p2_flat

                        # Reshape back to original shape
                        merged_param = merged_flat.reshape(param1.shape)
                        merged_model.state_dict()[name1].copy_(merged_param)

            return merged_model

        except Exception as e:
            logger.exception(f"Error in SLERP merge: {e}")
            return None

    async def dare_merge(self, model1, model2, config: MergeConfig):
        """DARE (Drop And REscale) merge with random parameter selection."""
        try:
            drop_rate = config.dare_drop_rate
            rescale = config.dare_rescale

            # Set random seed for reproducibility
            torch.manual_seed(config.dare_random_seed)
            np.random.seed(config.dare_random_seed)

            merged_model = copy.deepcopy(model1)

            with torch.no_grad():
                for (name1, param1), (name2, param2) in zip(
                    model1.named_parameters(), model2.named_parameters(), strict=False
                ):
                    if name1 == name2 and param1.shape == param2.shape:
                        # Calculate delta
                        delta = param2 - param1

                        # Create random mask for dropping parameters
                        mask = torch.rand_like(delta) > drop_rate

                        # Apply mask to delta
                        masked_delta = delta * mask.float()

                        # Rescale if enabled
                        if rescale and mask.sum() > 0:
                            scale_factor = mask.numel() / mask.sum().float()
                            masked_delta *= scale_factor

                        # Apply masked delta
                        merged_param = param1 + masked_delta
                        merged_model.state_dict()[name1].copy_(merged_param)

            return merged_model

        except Exception as e:
            logger.exception(f"Error in DARE merge: {e}")
            return None

    async def evolutionary_merge(self, model1, model2, config: MergeConfig):
        """Evolutionary merge with mutation and selection pressure."""
        try:
            mutation_strength = config.mutation_strength
            selection_prob = config.parameter_selection_prob

            merged_model = copy.deepcopy(model1)

            with torch.no_grad():
                for (name1, param1), (name2, param2) in zip(
                    model1.named_parameters(), model2.named_parameters(), strict=False
                ):
                    if name1 == name2 and param1.shape == param2.shape:
                        # Random selection between parents
                        selection_mask = torch.rand_like(param1) < selection_prob

                        # Base parameter selection
                        merged_param = torch.where(selection_mask, param2, param1)

                        # Add evolutionary mutation
                        if mutation_strength > 0:
                            mutation = (
                                torch.randn_like(merged_param) * mutation_strength
                            )
                            mutation_mask = (
                                torch.rand_like(merged_param) < 0.1
                            )  # Sparse mutation
                            merged_param += mutation * mutation_mask.float()

                        merged_model.state_dict()[name1].copy_(merged_param)

            return merged_model

        except Exception as e:
            logger.exception(f"Error in evolutionary merge: {e}")
            return None

    async def layer_wise_merge(self, model1, model2, config: MergeConfig):
        """Layer-wise merge with adaptive blending."""
        try:
            merged_model = copy.deepcopy(model1)

            # Get layer information
            layer_names = [name for name, _ in model1.named_parameters()]
            num_layers = len(layer_names)

            with torch.no_grad():
                for i, ((name1, param1), (name2, param2)) in enumerate(
                    zip(
                        model1.named_parameters(),
                        model2.named_parameters(),
                        strict=False,
                    )
                ):
                    if name1 == name2 and param1.shape == param2.shape:
                        # Calculate layer-specific blend ratio
                        # Early layers: favor model1, later layers: favor model2
                        layer_progress = i / max(num_layers - 1, 1)
                        blend_ratio = 0.3 + 0.4 * layer_progress  # 0.3 to 0.7

                        # Special handling for attention layers
                        if "attention" in name1.lower() or "attn" in name1.lower():
                            blend_ratio = 0.5  # Equal blend for attention

                        # Layer-wise blending
                        merged_param = (1 - blend_ratio) * param1 + blend_ratio * param2
                        merged_model.state_dict()[name1].copy_(merged_param)

            return merged_model

        except Exception as e:
            logger.exception(f"Error in layer-wise merge: {e}")
            return None

    async def attention_guided_merge(self, model1, model2, config: MergeConfig):
        """Attention-guided merge focusing on important parameters."""
        try:
            merged_model = copy.deepcopy(model1)

            with torch.no_grad():
                for (name1, param1), (name2, param2) in zip(
                    model1.named_parameters(), model2.named_parameters(), strict=False
                ):
                    if name1 == name2 and param1.shape == param2.shape:
                        # Calculate parameter importance (simple heuristic)
                        importance1 = torch.abs(param1)
                        importance2 = torch.abs(param2)

                        # Normalize importances
                        total_importance = importance1 + importance2 + 1e-8
                        weight1 = importance1 / total_importance
                        weight2 = importance2 / total_importance

                        # Attention-weighted merge
                        merged_param = weight1 * param1 + weight2 * param2

                        # Apply to merged model
                        merged_model.state_dict()[name1].copy_(merged_param)

            return merged_model

        except Exception as e:
            logger.exception(f"Error in attention-guided merge: {e}")
            return None

    async def calculate_merge_quality(
        self, merged_model, model1, model2, parent1_info: dict, parent2_info: dict
    ) -> float:
        """Calculate quality score for merged model."""
        try:
            quality_score = 0.0

            # 1. Parameter diversity score (how different from parents)
            diversity_score = await self.calculate_parameter_diversity(
                merged_model, model1, model2
            )
            quality_score += 0.3 * diversity_score

            # 2. Stability score (gradients and norms)
            stability_score = await self.calculate_model_stability(merged_model)
            quality_score += 0.3 * stability_score

            # 3. Parent fitness influence
            parent1_fitness = parent1_info.get("fitness_score", 0.5)
            parent2_fitness = parent2_info.get("fitness_score", 0.5)
            expected_fitness = (parent1_fitness + parent2_fitness) / 2
            quality_score += 0.4 * expected_fitness

            return min(1.0, quality_score)

        except Exception as e:
            logger.exception(f"Error calculating merge quality: {e}")
            return 0.5

    async def calculate_parameter_diversity(
        self, merged_model, model1, model2
    ) -> float:
        """Calculate how diverse the merged model is from its parents."""
        try:
            total_similarity = 0.0
            param_count = 0

            with torch.no_grad():
                for (
                    (merged_name, merged_param),
                    (name1, param1),
                    (
                        name2,
                        param2,
                    ),
                ) in zip(
                    merged_model.named_parameters(),
                    model1.named_parameters(),
                    model2.named_parameters(),
                    strict=False,
                ):
                    if (
                        merged_name == name1 == name2
                        and merged_param.shape == param1.shape == param2.shape
                    ):
                        # Calculate cosine similarity with both parents
                        merged_flat = merged_param.flatten()
                        param1_flat = param1.flatten()
                        param2_flat = param2.flatten()

                        sim1 = F.cosine_similarity(
                            merged_flat.unsqueeze(0), param1_flat.unsqueeze(0)
                        ).item()
                        sim2 = F.cosine_similarity(
                            merged_flat.unsqueeze(0), param2_flat.unsqueeze(0)
                        ).item()

                        # Diversity is inverse of maximum similarity
                        max_similarity = max(sim1, sim2)
                        diversity = 1.0 - max_similarity

                        total_similarity += diversity
                        param_count += 1

            return total_similarity / max(param_count, 1)

        except Exception as e:
            logger.exception(f"Error calculating parameter diversity: {e}")
            return 0.5

    async def calculate_model_stability(self, model) -> float:
        """Calculate stability metrics for merged model."""
        try:
            stability_metrics = []

            with torch.no_grad():
                for _name, param in model.named_parameters():
                    # Check for NaN or Inf values
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        return 0.0  # Unstable model

                    # Calculate parameter norm
                    param_norm = torch.norm(param).item()

                    # Penalize extremely large or small norms
                    if param_norm > 100.0 or param_norm < 1e-6:
                        stability_metrics.append(0.5)
                    else:
                        stability_metrics.append(1.0)

            return np.mean(stability_metrics) if stability_metrics else 0.5

        except Exception as e:
            logger.exception(f"Error calculating model stability: {e}")
            return 0.5

    async def count_parameter_changes(self, merged_model, model1, model2) -> int:
        """Count how many parameters were significantly changed in merge."""
        try:
            changed_params = 0
            threshold = 1e-6

            with torch.no_grad():
                for (
                    (merged_name, merged_param),
                    (name1, param1),
                    (
                        name2,
                        param2,
                    ),
                ) in zip(
                    merged_model.named_parameters(),
                    model1.named_parameters(),
                    model2.named_parameters(),
                    strict=False,
                ):
                    if merged_name == name1 == name2:
                        # Check if merged param is different from both parents
                        diff1 = torch.abs(merged_param - param1).max().item()
                        diff2 = torch.abs(merged_param - param2).max().item()

                        if diff1 > threshold and diff2 > threshold:
                            changed_params += merged_param.numel()

            return changed_params

        except Exception as e:
            logger.exception(f"Error counting parameter changes: {e}")
            return 0

    def update_merge_analytics(self, merge_results: list[MergeResult]) -> None:
        """Update analytics based on merge results."""
        for result in merge_results:
            # Update strategy performance
            if result.success:
                self.strategy_performance[result.merge_strategy].append(
                    result.merge_quality_score
                )

            # Update global analytics
            self.merge_analytics["total_merges"] += 1
            if result.success:
                self.merge_analytics["successful_merges"] += 1

                # Update averages
                current_avg_time = self.merge_analytics["avg_merge_time"]
                total_successful = self.merge_analytics["successful_merges"]
                new_avg_time = (
                    (current_avg_time * (total_successful - 1)) + result.merge_time
                ) / total_successful
                self.merge_analytics["avg_merge_time"] = new_avg_time

                # Update best quality
                self.merge_analytics["best_merge_quality"] = max(
                    self.merge_analytics["best_merge_quality"],
                    result.merge_quality_score,
                )

    async def merge_models(
        self, model1, model2, strategy: str, generation: int
    ) -> Any | None:
        """Simple interface for merging two models."""
        config = MergeConfig()

        # Create mock parent info for merge
        parent1_info = {
            "individual_id": "parent1",
            "fitness_score": 0.7,
            "model": model1,
        }
        parent2_info = {
            "individual_id": "parent2",
            "fitness_score": 0.8,
            "model": model2,
        }

        result = await self.perform_merge(
            parent1_info, parent2_info, strategy, config, generation
        )

        return result.merged_model if result.success else None

    def get_merge_analytics(self) -> dict[str, Any]:
        """Get comprehensive merge analytics."""
        analytics = {
            "merge_summary": self.merge_analytics,
            "strategy_performance": {
                strategy: {
                    "attempts": len(scores),
                    "avg_quality": np.mean(scores) if scores else 0.0,
                    "best_quality": max(scores) if scores else 0.0,
                    "success_rate": len(scores)
                    / max(
                        len(
                            [
                                r
                                for r in self.merge_history
                                if r.merge_strategy == strategy
                            ]
                        ),
                        1,
                    ),
                }
                for strategy, scores in self.strategy_performance.items()
            },
            "recent_merges": [
                {
                    "merge_id": result.merge_id,
                    "strategy": result.merge_strategy,
                    "quality_score": result.merge_quality_score,
                    "success": result.success,
                    "merge_time": result.merge_time,
                }
                for result in self.merge_history[-10:]  # Last 10 merges
            ],
            "best_merges": sorted(
                [r for r in self.merge_history if r.success],
                key=lambda x: x.merge_quality_score,
                reverse=True,
            )[:5],  # Top 5 merges
        }

        return analytics


# Global merge operator instance
merge_operator = MergeOperator()
