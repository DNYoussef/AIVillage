"""Meta-Learning Engine - Learning How to Learn Better.

Implements meta-learning algorithms for optimizing agent learning strategies,
including few-shot learning, learning rate adaptation, and strategy optimization.
"""

import asyncio
import json
import logging
import pickle
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim

logger = logging.getLogger(__name__)


@dataclass
class LearningExperience:
    """Records a learning experience for meta-learning."""

    experience_id: str
    agent_id: str
    task_type: str
    learning_strategy: dict[str, Any]
    initial_performance: float
    final_performance: float
    learning_time: float
    data_efficiency: float  # Performance per data point
    convergence_rate: float
    stability_score: float
    meta_features: dict[str, Any]
    timestamp: datetime


@dataclass
class MetaLearningStrategy:
    """Defines a meta-learning strategy."""

    strategy_id: str
    name: str
    strategy_type: str  # 'optimization', 'architecture', 'regularization', etc.
    parameters: dict[str, Any]
    performance_history: list[float]
    adaptation_rules: dict[str, Any]
    success_rate: float = 0.0
    avg_improvement: float = 0.0


class LearningRateOptimizer:
    """Optimizes learning rates using meta-learning."""

    def __init__(self, memory_size: int = 1000) -> None:
        self.memory_size = memory_size
        self.experience_buffer = deque(maxlen=memory_size)
        self.lr_history = defaultdict(list)
        self.performance_history = defaultdict(list)

    def record_learning_experience(
        self,
        agent_id: str,
        learning_rate: float,
        performance_improvement: float,
        convergence_steps: int,
        task_difficulty: float,
    ) -> None:
        """Record learning experience for future optimization."""
        experience = {
            "agent_id": agent_id,
            "learning_rate": learning_rate,
            "performance_improvement": performance_improvement,
            "convergence_steps": convergence_steps,
            "task_difficulty": task_difficulty,
            "timestamp": datetime.now(),
        }

        self.experience_buffer.append(experience)
        self.lr_history[agent_id].append(learning_rate)
        self.performance_history[agent_id].append(performance_improvement)

    def optimize_learning_rate(
        self, agent_id: str, task_difficulty: float, base_lr: float = 0.001
    ) -> float:
        """Optimize learning rate based on historical performance."""
        if agent_id not in self.lr_history or len(self.lr_history[agent_id]) < 3:
            return base_lr

        # Get relevant experiences
        relevant_experiences = [
            exp
            for exp in self.experience_buffer
            if exp["agent_id"] == agent_id
            and abs(exp["task_difficulty"] - task_difficulty) < 0.3
        ]

        if not relevant_experiences:
            return base_lr

        # Find optimal learning rate based on performance
        best_lr = base_lr
        best_performance = 0.0

        for exp in relevant_experiences:
            if exp["performance_improvement"] > best_performance:
                best_performance = exp["performance_improvement"]
                best_lr = exp["learning_rate"]

        # Apply exponential moving average for stability
        if len(self.lr_history[agent_id]) > 0:
            prev_lr = self.lr_history[agent_id][-1]
            optimal_lr = 0.7 * prev_lr + 0.3 * best_lr
        else:
            optimal_lr = best_lr

        # Bound the learning rate
        optimal_lr = max(1e-6, min(0.1, optimal_lr))

        return optimal_lr

    def get_adaptive_schedule(self, agent_id: str, num_epochs: int) -> list[float]:
        """Generate adaptive learning rate schedule."""
        base_lr = self.optimize_learning_rate(agent_id, 0.5)  # Medium difficulty
        schedule = []

        # Cosine annealing with warm restarts based on historical performance
        for epoch in range(num_epochs):
            if epoch < num_epochs * 0.1:  # Warmup
                lr = base_lr * (epoch / (num_epochs * 0.1))
            else:
                # Cosine decay
                cos_inner = np.pi * (epoch - num_epochs * 0.1) / (num_epochs * 0.9)
                lr = base_lr * 0.5 * (1 + np.cos(cos_inner))

            schedule.append(lr)

        return schedule


class FewShotLearner:
    """Implements few-shot learning capabilities."""

    def __init__(self, embedding_dim: int = 128) -> None:
        self.embedding_dim = embedding_dim
        self.support_memory = {}
        self.prototypes = {}

    def create_prototype(
        self, task_id: str, support_examples: list[Any], support_labels: list[int]
    ) -> np.ndarray:
        """Create prototype representation from support examples."""
        # Simple centroid-based prototype (in practice, would use learned embeddings)
        embeddings = []

        for example in support_examples:
            # Convert example to embedding (simplified)
            if isinstance(example, list | np.ndarray):
                embedding = np.array(example)
            else:
                # Hash-based embedding for other types
                embedding = np.random.RandomState(hash(str(example)) % 2**32).normal(
                    size=self.embedding_dim
                )
            embeddings.append(embedding)

        embeddings = np.array(embeddings)

        # Create class prototypes
        unique_labels = np.unique(support_labels)
        prototypes = {}

        for label in unique_labels:
            mask = np.array(support_labels) == label
            class_embeddings = embeddings[mask]
            prototype = np.mean(class_embeddings, axis=0)
            prototypes[label] = prototype

        self.prototypes[task_id] = prototypes
        return prototypes

    def few_shot_predict(self, task_id: str, query_example: Any) -> tuple[int, float]:
        """Make prediction using few-shot learning."""
        if task_id not in self.prototypes:
            msg = f"No prototypes found for task {task_id}"
            raise ValueError(msg)

        # Convert query to embedding
        if isinstance(query_example, list | np.ndarray):
            query_embedding = np.array(query_example)
        else:
            query_embedding = np.random.RandomState(
                hash(str(query_example)) % 2**32
            ).normal(size=self.embedding_dim)

        # Find closest prototype
        min_distance = float("inf")
        predicted_label = None

        for label, prototype in self.prototypes[task_id].items():
            distance = np.linalg.norm(query_embedding - prototype)
            if distance < min_distance:
                min_distance = distance
                predicted_label = label

        # Convert distance to confidence
        confidence = 1.0 / (1.0 + min_distance)

        return predicted_label, confidence

    def update_prototypes(
        self,
        task_id: str,
        new_examples: list[Any],
        new_labels: list[int],
        alpha: float = 0.1,
    ) -> None:
        """Update prototypes with new examples using exponential moving average."""
        if task_id not in self.prototypes:
            self.create_prototype(task_id, new_examples, new_labels)
            return

        # Convert new examples to embeddings
        new_embeddings = []
        for example in new_examples:
            if isinstance(example, list | np.ndarray):
                embedding = np.array(example)
            else:
                embedding = np.random.RandomState(hash(str(example)) % 2**32).normal(
                    size=self.embedding_dim
                )
            new_embeddings.append(embedding)

        new_embeddings = np.array(new_embeddings)

        # Update prototypes
        for i, label in enumerate(new_labels):
            if label in self.prototypes[task_id]:
                # Update existing prototype
                old_prototype = self.prototypes[task_id][label]
                new_prototype = (1 - alpha) * old_prototype + alpha * new_embeddings[i]
                self.prototypes[task_id][label] = new_prototype
            else:
                # Add new prototype
                self.prototypes[task_id][label] = new_embeddings[i]


class ModelAgnosticMetaLearner:
    """Model-Agnostic Meta-Learning (MAML) implementation."""

    def __init__(
        self, model: nn.Module, inner_lr: float = 0.01, meta_lr: float = 0.001
    ) -> None:
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)

    def inner_update(
        self, support_x: torch.Tensor, support_y: torch.Tensor, num_steps: int = 1
    ) -> nn.Module:
        """Perform inner loop update on support set."""
        # Clone model for inner update
        model_copy = type(self.model)(**self.model.__dict__)
        model_copy.load_state_dict(self.model.state_dict())

        inner_optimizer = optim.SGD(model_copy.parameters(), lr=self.inner_lr)

        for _ in range(num_steps):
            predictions = model_copy(support_x)
            loss = nn.functional.cross_entropy(predictions, support_y)

            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

        return model_copy

    def meta_update(
        self,
        tasks_batch: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ],
    ):
        """Perform meta-update across batch of tasks."""
        meta_loss = 0.0

        for support_x, support_y, query_x, query_y in tasks_batch:
            # Inner update
            adapted_model = self.inner_update(support_x, support_y)

            # Query loss
            query_predictions = adapted_model(query_x)
            query_loss = nn.functional.cross_entropy(query_predictions, query_y)
            meta_loss += query_loss

        # Meta gradient step
        meta_loss /= len(tasks_batch)

        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()


class StrategyOptimizer:
    """Optimizes learning strategies based on task characteristics."""

    def __init__(self) -> None:
        self.strategy_performance = defaultdict(list)
        self.task_characteristics = {}
        self.strategy_mappings = {}

    def register_strategy(self, strategy: MetaLearningStrategy) -> None:
        """Register a learning strategy."""
        self.strategy_mappings[strategy.strategy_id] = strategy

    def record_strategy_performance(
        self, strategy_id: str, task_characteristics: dict[str, Any], performance: float
    ) -> None:
        """Record performance of a strategy on a task."""
        self.strategy_performance[strategy_id].append(
            {
                "task_characteristics": task_characteristics,
                "performance": performance,
                "timestamp": datetime.now(),
            }
        )

        # Update strategy statistics
        if strategy_id in self.strategy_mappings:
            strategy = self.strategy_mappings[strategy_id]
            strategy.performance_history.append(performance)

            # Update success rate and average improvement
            if len(strategy.performance_history) > 1:
                improvements = [
                    strategy.performance_history[i]
                    - strategy.performance_history[i - 1]
                    for i in range(1, len(strategy.performance_history))
                ]
                strategy.avg_improvement = np.mean(improvements)
                strategy.success_rate = sum(1 for imp in improvements if imp > 0) / len(
                    improvements
                )

    def recommend_strategy(
        self, task_characteristics: dict[str, Any]
    ) -> MetaLearningStrategy | None:
        """Recommend best strategy for given task characteristics."""
        best_strategy = None
        best_score = -float("inf")

        for strategy in self.strategy_mappings.values():
            score = self._calculate_strategy_score(strategy, task_characteristics)

            if score > best_score:
                best_score = score
                best_strategy = strategy

        return best_strategy

    def _calculate_strategy_score(
        self, strategy: MetaLearningStrategy, task_characteristics: dict[str, Any]
    ) -> float:
        """Calculate compatibility score between strategy and task."""
        if strategy.strategy_id not in self.strategy_performance:
            return 0.0

        # Find similar tasks
        similar_tasks = []
        for record in self.strategy_performance[strategy.strategy_id]:
            similarity = self._calculate_task_similarity(
                record["task_characteristics"], task_characteristics
            )
            if similarity > 0.7:  # Similarity threshold
                similar_tasks.append(record["performance"])

        if not similar_tasks:
            # Fall back to overall performance
            return strategy.avg_improvement * strategy.success_rate

        # Average performance on similar tasks
        avg_performance = np.mean(similar_tasks)
        confidence = min(
            1.0, len(similar_tasks) / 10.0
        )  # Confidence based on sample size

        return avg_performance * confidence

    def _calculate_task_similarity(
        self, task1: dict[str, Any], task2: dict[str, Any]
    ) -> float:
        """Calculate similarity between two task characteristic sets."""
        common_keys = set(task1.keys()) & set(task2.keys())
        if not common_keys:
            return 0.0

        similarity_sum = 0.0

        for key in common_keys:
            val1, val2 = task1[key], task2[key]

            if isinstance(val1, int | float) and isinstance(val2, int | float):
                # Numerical similarity
                max_val = max(abs(val1), abs(val2), 1.0)
                similarity = 1.0 - abs(val1 - val2) / max_val
            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity (simple)
                similarity = 1.0 if val1 == val2 else 0.0
            else:
                similarity = 1.0 if val1 == val2 else 0.0

            similarity_sum += similarity

        return similarity_sum / len(common_keys)


class MetaLearningEngine:
    """Main meta-learning engine that coordinates all meta-learning components."""

    def __init__(
        self,
        storage_path: str = "evolution_data/meta_learning",
        memory_size: int = 10000,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.lr_optimizer = LearningRateOptimizer(memory_size)
        self.few_shot_learner = FewShotLearner()
        self.strategy_optimizer = StrategyOptimizer()

        # Experience storage
        self.experiences = deque(maxlen=memory_size)
        self.agent_profiles = {}

        # Load existing data
        self.load_meta_learning_data()

        # Initialize default strategies
        self._initialize_default_strategies()

    def _initialize_default_strategies(self) -> None:
        """Initialize default meta-learning strategies."""
        strategies = [
            MetaLearningStrategy(
                strategy_id="aggressive_lr",
                name="Aggressive Learning Rate",
                strategy_type="optimization",
                parameters={"initial_lr": 0.01, "decay_rate": 0.9, "warmup": False},
                performance_history=[],
                adaptation_rules={
                    "high_loss": "increase_lr",
                    "low_loss": "decrease_lr",
                },
            ),
            MetaLearningStrategy(
                strategy_id="conservative_lr",
                name="Conservative Learning Rate",
                strategy_type="optimization",
                parameters={"initial_lr": 0.001, "decay_rate": 0.95, "warmup": True},
                performance_history=[],
                adaptation_rules={
                    "high_loss": "maintain_lr",
                    "low_loss": "slight_decrease",
                },
            ),
            MetaLearningStrategy(
                strategy_id="adaptive_dropout",
                name="Adaptive Dropout",
                strategy_type="regularization",
                parameters={"initial_dropout": 0.2, "adaptive": True},
                performance_history=[],
                adaptation_rules={
                    "overfitting": "increase_dropout",
                    "underfitting": "decrease_dropout",
                },
            ),
            MetaLearningStrategy(
                strategy_id="curriculum_learning",
                name="Curriculum Learning",
                strategy_type="training",
                parameters={"difficulty_progression": "linear", "batch_mixing": True},
                performance_history=[],
                adaptation_rules={
                    "fast_learning": "increase_difficulty",
                    "slow_learning": "decrease_difficulty",
                },
            ),
        ]

        for strategy in strategies:
            self.strategy_optimizer.register_strategy(strategy)

    async def optimize_agent_learning(
        self,
        agent_id: str,
        task_type: str,
        task_characteristics: dict[str, Any],
        current_performance: float,
    ) -> dict[str, Any]:
        """Optimize learning strategy for an agent."""
        # Get agent profile
        agent_profile = self.agent_profiles.get(
            agent_id,
            {
                "learning_history": [],
                "preferred_strategies": [],
                "performance_trends": [],
            },
        )

        # Optimize learning rate
        optimal_lr = self.lr_optimizer.optimize_learning_rate(
            agent_id, task_characteristics.get("difficulty", 0.5)
        )

        # Recommend strategy
        recommended_strategy = self.strategy_optimizer.recommend_strategy(
            task_characteristics
        )

        # Generate learning configuration
        learning_config = {
            "learning_rate": optimal_lr,
            "strategy": (
                recommended_strategy.strategy_id
                if recommended_strategy
                else "conservative_lr"
            ),
            "strategy_parameters": (
                recommended_strategy.parameters if recommended_strategy else {}
            ),
            "adaptive_schedule": self.lr_optimizer.get_adaptive_schedule(agent_id, 100),
            "few_shot_enabled": task_characteristics.get("data_limited", False),
            "meta_features": task_characteristics,
        }

        # Update agent profile
        agent_profile["learning_history"].append(
            {
                "timestamp": datetime.now(),
                "task_type": task_type,
                "performance": current_performance,
                "config_used": learning_config,
            }
        )

        self.agent_profiles[agent_id] = agent_profile

        return learning_config

    async def record_learning_outcome(
        self,
        agent_id: str,
        task_type: str,
        initial_performance: float,
        final_performance: float,
        learning_config: dict[str, Any],
        learning_time: float,
        convergence_steps: int,
    ) -> None:
        """Record the outcome of a learning session."""
        # Create learning experience
        experience = LearningExperience(
            experience_id=f"{agent_id}_{int(time.time())}",
            agent_id=agent_id,
            task_type=task_type,
            learning_strategy=learning_config,
            initial_performance=initial_performance,
            final_performance=final_performance,
            learning_time=learning_time,
            data_efficiency=(final_performance - initial_performance)
            / max(learning_time, 1.0),
            convergence_rate=1.0 / max(convergence_steps, 1),
            stability_score=min(1.0, final_performance / max(initial_performance, 0.1)),
            meta_features=learning_config.get("meta_features", {}),
            timestamp=datetime.now(),
        )

        self.experiences.append(experience)

        # Update learning rate optimizer
        performance_improvement = final_performance - initial_performance
        self.lr_optimizer.record_learning_experience(
            agent_id,
            learning_config["learning_rate"],
            performance_improvement,
            convergence_steps,
            learning_config.get("meta_features", {}).get("difficulty", 0.5),
        )

        # Update strategy optimizer
        if "strategy" in learning_config:
            self.strategy_optimizer.record_strategy_performance(
                learning_config["strategy"],
                learning_config.get("meta_features", {}),
                performance_improvement,
            )

        logger.info(
            f"Recorded learning outcome for {agent_id}: {initial_performance:.3f} -> {final_performance:.3f}"
        )

    async def few_shot_adapt(
        self,
        agent_id: str,
        task_id: str,
        support_examples: list[Any],
        support_labels: list[int],
    ) -> dict[str, Any]:
        """Perform few-shot adaptation for an agent."""
        # Create prototypes
        self.few_shot_learner.create_prototype(
            f"{agent_id}_{task_id}", support_examples, support_labels
        )

        # Generate few-shot learning configuration
        config = {
            "task_id": f"{agent_id}_{task_id}",
            "num_classes": len(np.unique(support_labels)),
            "support_size": len(support_examples),
            "adaptation_steps": min(10, len(support_examples)),
            "prototype_based": True,
        }

        return config

    def get_agent_learning_profile(self, agent_id: str) -> dict[str, Any]:
        """Get comprehensive learning profile for an agent."""
        self.agent_profiles.get(agent_id, {})

        # Calculate statistics
        agent_experiences = [
            exp for exp in self.experiences if exp.agent_id == agent_id
        ]

        if agent_experiences:
            avg_improvement = np.mean(
                [
                    exp.final_performance - exp.initial_performance
                    for exp in agent_experiences
                ]
            )
            avg_learning_time = np.mean(
                [exp.learning_time for exp in agent_experiences]
            )
            avg_data_efficiency = np.mean(
                [exp.data_efficiency for exp in agent_experiences]
            )

            # Learning trend
            recent_performances = [
                exp.final_performance for exp in agent_experiences[-10:]
            ]
            learning_trend = (
                "improving"
                if len(recent_performances) > 1
                and recent_performances[-1] > recent_performances[0]
                else "stable"
            )
        else:
            avg_improvement = 0.0
            avg_learning_time = 0.0
            avg_data_efficiency = 0.0
            learning_trend = "unknown"

        return {
            "agent_id": agent_id,
            "total_experiences": len(agent_experiences),
            "avg_improvement": avg_improvement,
            "avg_learning_time": avg_learning_time,
            "avg_data_efficiency": avg_data_efficiency,
            "learning_trend": learning_trend,
            "preferred_strategies": self._get_preferred_strategies(agent_id),
            "optimal_lr_range": self._get_optimal_lr_range(agent_id),
            "last_updated": datetime.now().isoformat(),
        }

    def _get_preferred_strategies(self, agent_id: str) -> list[str]:
        """Get preferred strategies for an agent based on historical performance."""
        agent_experiences = [
            exp for exp in self.experiences if exp.agent_id == agent_id
        ]

        strategy_performance = defaultdict(list)

        for exp in agent_experiences:
            strategy_id = exp.learning_strategy.get("strategy", "unknown")
            improvement = exp.final_performance - exp.initial_performance
            strategy_performance[strategy_id].append(improvement)

        # Rank strategies by average improvement
        strategy_rankings = []
        for strategy_id, improvements in strategy_performance.items():
            avg_improvement = np.mean(improvements)
            strategy_rankings.append((strategy_id, avg_improvement))

        strategy_rankings.sort(key=lambda x: x[1], reverse=True)

        return [strategy_id for strategy_id, _ in strategy_rankings[:3]]

    def _get_optimal_lr_range(self, agent_id: str) -> tuple[float, float]:
        """Get optimal learning rate range for an agent."""
        if agent_id not in self.lr_optimizer.lr_history:
            return (0.001, 0.01)

        lr_history = self.lr_optimizer.lr_history[agent_id]
        performance_history = self.lr_optimizer.performance_history[agent_id]

        if len(lr_history) < 3:
            return (0.001, 0.01)

        # Find LRs that led to good performance
        good_lrs = [
            lr
            for lr, perf in zip(lr_history, performance_history, strict=False)
            if perf > 0.1
        ]

        if good_lrs:
            return (min(good_lrs), max(good_lrs))
        return (min(lr_history), max(lr_history))

    def save_meta_learning_data(self) -> None:
        """Save meta-learning data to disk."""
        try:
            # Save experiences
            experiences_file = self.storage_path / "experiences.pkl"
            with open(experiences_file, "wb") as f:
                pickle.dump(list(self.experiences), f)

            # Save agent profiles
            profiles_file = self.storage_path / "agent_profiles.json"
            with open(profiles_file, "w") as f:
                json.dump(self.agent_profiles, f, indent=2, default=str)

            # Save learning rate optimizer data
            lr_data = {
                "lr_history": dict(self.lr_optimizer.lr_history),
                "performance_history": dict(self.lr_optimizer.performance_history),
            }
            lr_file = self.storage_path / "lr_optimizer.json"
            with open(lr_file, "w") as f:
                json.dump(lr_data, f, indent=2)

            # Save strategy performance
            strategy_data = {}
            for (
                strategy_id,
                strategy,
            ) in self.strategy_optimizer.strategy_mappings.items():
                strategy_data[strategy_id] = asdict(strategy)

            strategy_file = self.storage_path / "strategies.json"
            with open(strategy_file, "w") as f:
                json.dump(strategy_data, f, indent=2, default=str)

            logger.info("Meta-learning data saved successfully")

        except Exception as e:
            logger.exception(f"Failed to save meta-learning data: {e}")

    def load_meta_learning_data(self) -> None:
        """Load meta-learning data from disk."""
        try:
            # Load experiences
            experiences_file = self.storage_path / "experiences.pkl"
            if experiences_file.exists():
                with open(experiences_file, "rb") as f:
                    experiences_list = pickle.load(f)
                    self.experiences.extend(experiences_list)

            # Load agent profiles
            profiles_file = self.storage_path / "agent_profiles.json"
            if profiles_file.exists():
                with open(profiles_file) as f:
                    self.agent_profiles = json.load(f)

            # Load learning rate optimizer data
            lr_file = self.storage_path / "lr_optimizer.json"
            if lr_file.exists():
                with open(lr_file) as f:
                    lr_data = json.load(f)
                    self.lr_optimizer.lr_history = defaultdict(
                        list, lr_data.get("lr_history", {})
                    )
                    self.lr_optimizer.performance_history = defaultdict(
                        list, lr_data.get("performance_history", {})
                    )

            # Load strategies
            strategy_file = self.storage_path / "strategies.json"
            if strategy_file.exists():
                with open(strategy_file) as f:
                    strategy_data = json.load(f)
                    for data in strategy_data.values():
                        strategy = MetaLearningStrategy(**data)
                        self.strategy_optimizer.register_strategy(strategy)

            logger.info("Meta-learning data loaded successfully")

        except Exception as e:
            logger.exception(f"Failed to load meta-learning data: {e}")

    async def generate_meta_learning_report(self) -> dict[str, Any]:
        """Generate comprehensive meta-learning analysis report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_experiences": len(self.experiences),
            "active_agents": len(self.agent_profiles),
            "strategy_performance": {},
            "learning_trends": {},
            "recommendations": [],
        }

        # Analyze strategy performance
        for strategy_id, strategy in self.strategy_optimizer.strategy_mappings.items():
            if strategy.performance_history:
                report["strategy_performance"][strategy_id] = {
                    "avg_improvement": strategy.avg_improvement,
                    "success_rate": strategy.success_rate,
                    "total_uses": len(strategy.performance_history),
                    "recent_trend": (
                        "improving"
                        if len(strategy.performance_history) > 3
                        and strategy.performance_history[-1]
                        > strategy.performance_history[-3]
                        else "stable"
                    ),
                }

        # Analyze learning trends
        if self.experiences:
            recent_experiences = list(self.experiences)[-100:]  # Last 100 experiences

            avg_improvement = np.mean(
                [
                    exp.final_performance - exp.initial_performance
                    for exp in recent_experiences
                ]
            )
            avg_efficiency = np.mean(
                [exp.data_efficiency for exp in recent_experiences]
            )

            report["learning_trends"] = {
                "avg_improvement": avg_improvement,
                "avg_data_efficiency": avg_efficiency,
                "convergence_speed": np.mean(
                    [exp.convergence_rate for exp in recent_experiences]
                ),
            }

        # Generate recommendations
        recommendations = []

        # Strategy recommendations
        best_strategy = max(
            self.strategy_optimizer.strategy_mappings.values(),
            key=lambda s: s.avg_improvement if s.performance_history else 0,
        )
        if best_strategy.performance_history:
            recommendations.append(
                f"Strategy '{best_strategy.name}' shows best performance with {
                    best_strategy.avg_improvement:.3f} average improvement"
            )

        # Learning rate recommendations
        if len(self.lr_optimizer.experience_buffer) > 10:
            avg_lr = np.mean(
                [exp["learning_rate"] for exp in self.lr_optimizer.experience_buffer]
            )
            recommendations.append(
                f"Optimal learning rate range appears to be around {avg_lr:.4f}"
            )

        # General recommendations
        if report["learning_trends"].get("avg_improvement", 0) < 0.1:
            recommendations.append(
                "Overall learning improvement is low. Consider adjusting meta-learning parameters."
            )

        report["recommendations"] = recommendations

        return report


if __name__ == "__main__":

    async def example_usage() -> None:
        # Initialize meta-learning engine
        meta_engine = MetaLearningEngine()

        # Example: optimize learning for an agent
        task_characteristics = {
            "difficulty": 0.7,
            "data_size": 1000,
            "data_limited": False,
            "task_complexity": "high",
        }

        learning_config = await meta_engine.optimize_agent_learning(
            agent_id="test_agent",
            task_type="classification",
            task_characteristics=task_characteristics,
            current_performance=0.65,
        )

        print(f"Optimized learning config: {learning_config}")

        # Simulate learning outcome
        await meta_engine.record_learning_outcome(
            agent_id="test_agent",
            task_type="classification",
            initial_performance=0.65,
            final_performance=0.85,
            learning_config=learning_config,
            learning_time=120.0,
            convergence_steps=50,
        )

        # Get agent profile
        profile = meta_engine.get_agent_learning_profile("test_agent")
        print(f"Agent profile: {profile}")

        # Generate report
        report = await meta_engine.generate_meta_learning_report()
        print(f"Meta-learning report: {report}")

    asyncio.run(example_usage())
