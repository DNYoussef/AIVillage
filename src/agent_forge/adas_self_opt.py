"""ADAS (Automated Design and Architecture Search) Self-Optimization.

Implements automated design and architecture search for self-improving agents:
- Neural Architecture Search (NAS) for model optimization
- Hyperparameter optimization using Bayesian optimization
- Self-modifying training strategies
- Performance-driven architecture evolution
- Integration with geometry feedback for guided optimization
"""

import asyncio
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
import wandb

from src.agent_forge.geometry_feedback import GeometryTracker

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureConfig:
    """Configuration for a model architecture candidate."""

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    dropout_rate: float
    activation_function: str
    layer_norm_eps: float
    position_embedding_type: str
    use_cache: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> bool:
        """Validate architecture configuration."""
        if self.hidden_size % self.num_attention_heads != 0:
            return False
        if self.num_layers < 1 or self.num_layers > 48:
            return False
        return not (self.hidden_size < 64 or self.hidden_size > 4096)


@dataclass
class TrainingConfig:
    """Configuration for training strategy."""

    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    weight_decay: float
    lr_scheduler_type: str
    optimizer_type: str
    max_grad_norm: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizationResult:
    """Result of an optimization trial."""

    architecture_config: ArchitectureConfig
    training_config: TrainingConfig
    performance_score: float
    geometry_metrics: dict[str, float]
    compass_direction: str
    training_time: float
    memory_usage: float
    convergence_steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture_config.to_dict(),
            "training": self.training_config.to_dict(),
            "performance_score": self.performance_score,
            "geometry_metrics": self.geometry_metrics,
            "compass_direction": self.compass_direction,
            "training_time": self.training_time,
            "memory_usage": self.memory_usage,
            "convergence_steps": self.convergence_steps,
        }


class ArchitectureGenerator:
    """Generates and evolves model architectures."""

    def __init__(
        self, base_config: ArchitectureConfig, mutation_rate: float = 0.1
    ) -> None:
        self.base_config = base_config
        self.mutation_rate = mutation_rate
        self.architecture_history = []

    def generate_random_architecture(self) -> ArchitectureConfig:
        """Generate a random architecture configuration."""
        config = ArchitectureConfig(
            num_layers=random.randint(6, 24),
            hidden_size=random.choice([256, 384, 512, 768, 1024]),
            num_attention_heads=random.choice([4, 6, 8, 12, 16]),
            intermediate_size=0,  # Will be set based on hidden_size
            dropout_rate=random.uniform(0.0, 0.3),
            activation_function=random.choice(["gelu", "relu", "swish", "mish"]),
            layer_norm_eps=random.choice([1e-5, 1e-6, 1e-12]),
            position_embedding_type=random.choice(["absolute", "relative", "rotary"]),
            use_cache=random.choice([True, False]),
        )

        # Set intermediate size based on hidden size
        config.intermediate_size = config.hidden_size * random.choice([2, 3, 4])

        # Ensure valid configuration
        if not config.validate():
            return self.generate_random_architecture()

        return config

    def mutate_architecture(self, config: ArchitectureConfig) -> ArchitectureConfig:
        """Mutate an existing architecture configuration."""
        new_config = ArchitectureConfig(**config.to_dict())

        # Randomly mutate parameters
        if random.random() < self.mutation_rate:
            new_config.num_layers = max(
                1, min(48, config.num_layers + random.randint(-2, 2))
            )

        if random.random() < self.mutation_rate:
            hidden_sizes = [256, 384, 512, 768, 1024]
            current_idx = (
                hidden_sizes.index(config.hidden_size)
                if config.hidden_size in hidden_sizes
                else 2
            )
            new_idx = max(
                0, min(len(hidden_sizes) - 1, current_idx + random.randint(-1, 1))
            )
            new_config.hidden_size = hidden_sizes[new_idx]

        if random.random() < self.mutation_rate:
            heads = [4, 6, 8, 12, 16]
            current_idx = (
                heads.index(config.num_attention_heads)
                if config.num_attention_heads in heads
                else 2
            )
            new_idx = max(0, min(len(heads) - 1, current_idx + random.randint(-1, 1)))
            new_config.num_attention_heads = heads[new_idx]

        if random.random() < self.mutation_rate:
            new_config.dropout_rate = max(
                0.0, min(0.5, config.dropout_rate + random.uniform(-0.1, 0.1))
            )

        if random.random() < self.mutation_rate:
            new_config.activation_function = random.choice(
                ["gelu", "relu", "swish", "mish"]
            )

        # Update intermediate size
        new_config.intermediate_size = new_config.hidden_size * random.choice([2, 3, 4])

        # Validate and retry if invalid
        if not new_config.validate():
            return self.mutate_architecture(config)

        return new_config

    def crossover_architectures(
        self, config1: ArchitectureConfig, config2: ArchitectureConfig
    ) -> ArchitectureConfig:
        """Create offspring architecture by crossing over two parents."""
        # Random crossover of parameters
        new_config = ArchitectureConfig(
            num_layers=random.choice([config1.num_layers, config2.num_layers]),
            hidden_size=random.choice([config1.hidden_size, config2.hidden_size]),
            num_attention_heads=random.choice(
                [config1.num_attention_heads, config2.num_attention_heads]
            ),
            intermediate_size=0,  # Will be set below
            dropout_rate=(config1.dropout_rate + config2.dropout_rate) / 2,
            activation_function=random.choice(
                [config1.activation_function, config2.activation_function]
            ),
            layer_norm_eps=random.choice(
                [config1.layer_norm_eps, config2.layer_norm_eps]
            ),
            position_embedding_type=random.choice(
                [config1.position_embedding_type, config2.position_embedding_type]
            ),
            use_cache=random.choice([config1.use_cache, config2.use_cache]),
        )

        # Set intermediate size
        new_config.intermediate_size = new_config.hidden_size * random.choice([2, 3, 4])

        # Validate and retry if invalid
        if not new_config.validate():
            return self.crossover_architectures(config1, config2)

        return new_config


class TrainingStrategyOptimizer:
    """Optimizes training strategies using Bayesian optimization."""

    def __init__(self) -> None:
        self.strategy_history = []
        self.performance_history = []
        self.gp_regressor = None

    def suggest_training_config(
        self, geometry_feedback: dict[str, float] | None = None
    ) -> TrainingConfig:
        """Suggest optimal training configuration."""
        if len(self.strategy_history) < 5:
            # Random exploration for first few trials
            return self._generate_random_training_config()
        # Use Bayesian optimization
        return self._bayesian_suggest_training_config(geometry_feedback)

    def _generate_random_training_config(self) -> TrainingConfig:
        """Generate random training configuration."""
        return TrainingConfig(
            learning_rate=random.choice([1e-5, 3e-5, 5e-5, 1e-4, 3e-4]),
            batch_size=random.choice([4, 8, 16, 32]),
            gradient_accumulation_steps=random.choice([1, 2, 4, 8]),
            warmup_steps=random.randint(100, 1000),
            weight_decay=random.uniform(0.0, 0.1),
            lr_scheduler_type=random.choice(
                ["linear", "cosine", "polynomial", "constant"]
            ),
            optimizer_type=random.choice(["adamw", "sgd", "adafactor"]),
            max_grad_norm=random.uniform(0.5, 2.0),
        )

    def _bayesian_suggest_training_config(
        self, geometry_feedback: dict[str, float] | None = None
    ) -> TrainingConfig:
        """Use Bayesian optimization to suggest training config."""
        if self.gp_regressor is None:
            self._initialize_gp_regressor()

        # Define search space bounds
        bounds = [
            (1e-6, 1e-3),  # learning_rate
            (4, 32),  # batch_size
            (1, 8),  # gradient_accumulation_steps
            (100, 2000),  # warmup_steps
            (0.0, 0.2),  # weight_decay
            (0.5, 3.0),  # max_grad_norm
        ]

        # Acquisition function
        def acquisition(x):
            x_reshaped = x.reshape(1, -1)
            mu, sigma = self.gp_regressor.predict(x_reshaped, return_std=True)
            # Upper confidence bound
            return -(mu + 2.0 * sigma)  # Negative because we minimize

        # Optimize acquisition function
        best_x = None
        best_acq = float("inf")

        for _ in range(100):  # Random restarts
            x0 = np.array([random.uniform(b[0], b[1]) for b in bounds])

            result = minimize(acquisition, x0, bounds=bounds, method="L-BFGS-B")

            if result.fun < best_acq:
                best_acq = result.fun
                best_x = result.x

        # Convert optimized parameters to TrainingConfig
        lr_scheduler_types = ["linear", "cosine", "polynomial", "constant"]
        optimizer_types = ["adamw", "sgd", "adafactor"]

        return TrainingConfig(
            learning_rate=float(best_x[0]),
            batch_size=int(best_x[1]),
            gradient_accumulation_steps=int(best_x[2]),
            warmup_steps=int(best_x[3]),
            weight_decay=float(best_x[4]),
            lr_scheduler_type=random.choice(lr_scheduler_types),
            optimizer_type=random.choice(optimizer_types),
            max_grad_norm=float(best_x[5]),
        )

    def _initialize_gp_regressor(self) -> None:
        """Initialize Gaussian Process regressor."""
        if len(self.strategy_history) < 3:
            return

        # Prepare training data
        X = []
        y = []

        for strategy, performance in zip(
            self.strategy_history, self.performance_history, strict=False
        ):
            x_vec = [
                strategy.learning_rate,
                strategy.batch_size,
                strategy.gradient_accumulation_steps,
                strategy.warmup_steps,
                strategy.weight_decay,
                strategy.max_grad_norm,
            ]
            X.append(x_vec)
            y.append(performance)

        X = np.array(X)
        y = np.array(y)

        # Initialize and fit GP
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp_regressor = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, n_restarts_optimizer=10
        )
        self.gp_regressor.fit(X, y)

    def update_performance(self, config: TrainingConfig, performance: float) -> None:
        """Update performance history."""
        self.strategy_history.append(config)
        self.performance_history.append(performance)

        # Re-initialize GP with new data
        if len(self.strategy_history) >= 3:
            self._initialize_gp_regressor()


class ModelBuilder:
    """Builds models from architecture configurations."""

    @staticmethod
    def build_model(
        arch_config: ArchitectureConfig, vocab_size: int = 32000
    ) -> nn.Module:
        """Build a model from architecture configuration."""
        # Create transformer config
        config = AutoConfig.from_pretrained("microsoft/DialoGPT-small")

        # Update with our architecture
        config.num_hidden_layers = arch_config.num_layers
        config.hidden_size = arch_config.hidden_size
        config.num_attention_heads = arch_config.num_attention_heads
        config.intermediate_size = arch_config.intermediate_size
        config.hidden_dropout_prob = arch_config.dropout_rate
        config.attention_probs_dropout_prob = arch_config.dropout_rate
        config.hidden_act = arch_config.activation_function
        config.layer_norm_eps = arch_config.layer_norm_eps
        config.use_cache = arch_config.use_cache
        config.vocab_size = vocab_size

        # Build model
        model = AutoModelForCausalLM.from_config(config)

        return model


class PerformanceEvaluator:
    """Evaluates model performance on various metrics."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device

    async def evaluate_model(
        self,
        model: nn.Module,
        arch_config: ArchitectureConfig,
        training_config: TrainingConfig,
        test_data: Any | None = None,
    ) -> OptimizationResult:
        """Comprehensive model evaluation."""
        start_time = time.time()

        # Move model to device
        model = model.to(self.device)

        # Initialize geometry tracker
        geometry_tracker = GeometryTracker(model, update_interval=10)

        # Simulate training/evaluation
        performance_score = await self._evaluate_performance(model, test_data)

        # Get geometry metrics
        dummy_input = torch.randn(4, 64, arch_config.hidden_size).to(self.device)
        geometry_metrics = geometry_tracker.update(dummy_input)

        # Calculate additional metrics
        memory_usage = self._calculate_memory_usage(model)
        convergence_steps = random.randint(100, 1000)  # Placeholder

        training_time = time.time() - start_time

        # Get compass direction
        compass_direction = "Unknown"
        if (
            geometry_metrics
            and hasattr(geometry_tracker, "compass_history")
            and geometry_tracker.compass_history
        ):
            compass_direction = geometry_tracker.compass_history[
                -1
            ].get_primary_direction()

        return OptimizationResult(
            architecture_config=arch_config,
            training_config=training_config,
            performance_score=performance_score,
            geometry_metrics=geometry_metrics.__dict__ if geometry_metrics else {},
            compass_direction=compass_direction,
            training_time=training_time,
            memory_usage=memory_usage,
            convergence_steps=convergence_steps,
        )

    async def _evaluate_performance(
        self, model: nn.Module, test_data: Any = None
    ) -> float:
        """Evaluate model performance (placeholder implementation)."""
        # Placeholder: run forward passes and calculate metrics
        model.eval()

        total_loss = 0.0
        num_batches = 5

        with torch.no_grad():
            for _ in range(num_batches):
                # Generate dummy batch
                batch_size = 4
                seq_len = 64
                vocab_size = 32000

                input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(
                    self.device
                )
                labels = input_ids.clone()

                try:
                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss
                    total_loss += loss.item()
                except Exception as e:
                    logger.warning(f"Evaluation error: {e}")
                    return 0.0

        avg_loss = total_loss / num_batches

        # Convert loss to performance score (higher is better)
        performance_score = 1.0 / (1.0 + avg_loss)

        return performance_score

    def _calculate_memory_usage(self, model: nn.Module) -> float:
        """Calculate model memory usage in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.numel() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()

        return (param_size + buffer_size) / (1024**2)  # Convert to MB


class ADASOptimizer:
    """Main ADAS optimization orchestrator."""

    def __init__(
        self,
        base_architecture: ArchitectureConfig,
        output_dir: str,
        population_size: int = 20,
        num_generations: int = 50,
        device: str = "cuda",
    ) -> None:
        self.base_architecture = base_architecture
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.population_size = population_size
        self.num_generations = num_generations
        self.device = device

        # Initialize components
        self.arch_generator = ArchitectureGenerator(base_architecture)
        self.training_optimizer = TrainingStrategyOptimizer()
        self.evaluator = PerformanceEvaluator(device)

        # Tracking
        self.optimization_history = []
        self.best_result = None

        # W&B logging
        if wandb.run is None:
            wandb.init(project="adas-optimization", job_type="architecture_search")

    async def optimize(self) -> OptimizationResult:
        """Run ADAS optimization process."""
        logger.info(
            f"Starting ADAS optimization: {self.num_generations} generations, {self.population_size} population"
        )

        # Initialize population
        population = []
        for _ in range(self.population_size):
            arch_config = self.arch_generator.generate_random_architecture()
            training_config = self.training_optimizer.suggest_training_config()
            population.append((arch_config, training_config))

        # Evolution loop
        for generation in range(self.num_generations):
            logger.info(f"Generation {generation + 1}/{self.num_generations}")

            # Evaluate population
            results = []
            for i, (arch_config, training_config) in enumerate(population):
                logger.info(f"Evaluating individual {i + 1}/{self.population_size}")

                try:
                    # Build model
                    model = ModelBuilder.build_model(arch_config)

                    # Evaluate
                    result = await self.evaluator.evaluate_model(
                        model, arch_config, training_config
                    )
                    results.append(result)

                    # Update training optimizer
                    self.training_optimizer.update_performance(
                        training_config, result.performance_score
                    )

                    # Track best result
                    if (
                        self.best_result is None
                        or result.performance_score > self.best_result.performance_score
                    ):
                        self.best_result = result

                    # Log to W&B
                    wandb.log(
                        {
                            "generation": generation,
                            "individual": i,
                            "performance_score": result.performance_score,
                            "compass_direction": result.compass_direction,
                            "memory_usage_mb": result.memory_usage,
                            "num_layers": arch_config.num_layers,
                            "hidden_size": arch_config.hidden_size,
                            "num_attention_heads": arch_config.num_attention_heads,
                            "learning_rate": training_config.learning_rate,
                        }
                    )

                except Exception as e:
                    logger.exception(f"Evaluation failed for individual {i}: {e}")
                    # Create dummy result with low score
                    result = OptimizationResult(
                        architecture_config=arch_config,
                        training_config=training_config,
                        performance_score=0.0,
                        geometry_metrics={},
                        compass_direction="Unknown",
                        training_time=0.0,
                        memory_usage=0.0,
                        convergence_steps=0,
                    )
                    results.append(result)

            # Store generation results
            self.optimization_history.append(results)

            # Selection and reproduction
            if generation < self.num_generations - 1:
                population = self._evolve_population(results)

            # Save checkpoint
            self._save_checkpoint(generation, results)

        logger.info(
            f"ADAS optimization complete. Best score: {self.best_result.performance_score:.4f}"
        )
        return self.best_result

    def _evolve_population(
        self, results: list[OptimizationResult]
    ) -> list[tuple[ArchitectureConfig, TrainingConfig]]:
        """Evolve population for next generation."""
        # Sort by performance
        sorted_results = sorted(
            results, key=lambda x: x.performance_score, reverse=True
        )

        # Select top performers (elitism)
        elite_size = self.population_size // 4
        elite = sorted_results[:elite_size]

        new_population = []

        # Keep elite
        for result in elite:
            new_population.append((result.architecture_config, result.training_config))

        # Generate offspring
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(sorted_results)
            parent2 = self._tournament_selection(sorted_results)

            # Crossover architectures
            offspring_arch = self.arch_generator.crossover_architectures(
                parent1.architecture_config, parent2.architecture_config
            )

            # Mutate architecture
            if random.random() < 0.2:
                offspring_arch = self.arch_generator.mutate_architecture(offspring_arch)

            # Generate new training config (with Bayesian optimization)
            geometry_feedback = (
                parent1.geometry_metrics if parent1.geometry_metrics else None
            )
            offspring_training = self.training_optimizer.suggest_training_config(
                geometry_feedback
            )

            new_population.append((offspring_arch, offspring_training))

        return new_population

    def _tournament_selection(
        self, results: list[OptimizationResult], tournament_size: int = 3
    ) -> OptimizationResult:
        """Tournament selection for parent selection."""
        tournament = random.sample(results, min(tournament_size, len(results)))
        return max(tournament, key=lambda x: x.performance_score)

    def _save_checkpoint(
        self, generation: int, results: list[OptimizationResult]
    ) -> None:
        """Save optimization checkpoint."""
        checkpoint = {
            "generation": generation,
            "best_result": self.best_result.to_dict() if self.best_result else None,
            "generation_results": [r.to_dict() for r in results],
            "population_size": self.population_size,
            "num_generations": self.num_generations,
        }

        checkpoint_path = self.output_dir / f"adas_checkpoint_gen_{generation}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def get_optimization_summary(self) -> dict[str, Any]:
        """Get optimization summary."""
        if not self.optimization_history:
            return {"status": "not_started"}

        all_results = [
            result for generation in self.optimization_history for result in generation
        ]

        return {
            "total_evaluations": len(all_results),
            "best_performance": max(r.performance_score for r in all_results),
            "average_performance": np.mean([r.performance_score for r in all_results]),
            "best_architecture": (
                self.best_result.architecture_config.to_dict()
                if self.best_result
                else None
            ),
            "best_training_config": (
                self.best_result.training_config.to_dict() if self.best_result else None
            ),
            "compass_directions": [r.compass_direction for r in all_results],
            "memory_usage_range": (
                min(r.memory_usage for r in all_results),
                max(r.memory_usage for r in all_results),
            ),
        }


# CLI and usage
async def main() -> None:
    """Main ADAS optimization entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="ADAS Self-Optimization")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--generations", type=int, default=20, help="Number of generations"
    )
    parser.add_argument("--population", type=int, default=10, help="Population size")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Base architecture configuration
    base_arch = ArchitectureConfig(
        num_layers=12,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        dropout_rate=0.1,
        activation_function="gelu",
        layer_norm_eps=1e-12,
        position_embedding_type="absolute",
        use_cache=True,
    )

    # Initialize ADAS optimizer
    adas = ADASOptimizer(
        base_architecture=base_arch,
        output_dir=args.output_dir,
        population_size=args.population,
        num_generations=args.generations,
        device=args.device,
    )

    # Run optimization
    best_result = await adas.optimize()

    # Print summary
    summary = adas.get_optimization_summary()
    print("\nOptimization Summary:")
    print(f"Best Performance: {summary['best_performance']:.4f}")
    print(f"Total Evaluations: {summary['total_evaluations']}")
    print(f"Best Architecture: {summary['best_architecture']}")

    # Save final results
    final_results_path = Path(args.output_dir) / "adas_final_results.json"
    with open(final_results_path, "w") as f:
        json.dump(
            {"best_result": best_result.to_dict(), "optimization_summary": summary},
            f,
            indent=2,
        )

    print(f"Final results saved: {final_results_path}")


if __name__ == "__main__":
    asyncio.run(main())
