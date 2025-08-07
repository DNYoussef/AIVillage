"""Agent Evolution Engine - Core Self-Evolution System for the 18-Agent Ecosystem.

This is the central orchestration system for the Atlantis vision's self-evolving agents.
It implements performance-based selection, genetic optimization, and autonomous improvement.
"""

import asyncio
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AgentKPIs:
    """Key Performance Indicators for agent evaluation."""

    agent_id: str
    task_success_rate: float = 0.0
    average_response_time: float = 0.0
    user_satisfaction: float = 0.0
    resource_efficiency: float = 0.0
    code_quality_score: float = 0.0
    adaptation_speed: float = 0.0
    collaboration_score: float = 0.0
    specialization_effectiveness: float = 0.0
    learning_rate: float = 0.0
    error_recovery_rate: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def fitness_score(self) -> float:
        """Calculate overall fitness score from KPIs."""
        weights = {
            "task_success_rate": 0.25,
            "user_satisfaction": 0.20,
            "resource_efficiency": 0.15,
            "code_quality_score": 0.15,
            "adaptation_speed": 0.10,
            "collaboration_score": 0.10,
            "specialization_effectiveness": 0.05,
        }

        score = 0.0
        for metric, weight in weights.items():
            score += getattr(self, metric) * weight

        return min(max(score, 0.0), 1.0)  # Clamp to [0,1]


@dataclass
class AgentGenome:
    """Genetic representation of an agent's configuration."""

    agent_id: str
    architecture_params: dict[str, Any]
    hyperparameters: dict[str, Any]
    specialization_config: dict[str, Any]
    behavior_weights: dict[str, float]
    code_templates: dict[str, str]
    learning_config: dict[str, Any]
    generation: int = 0
    parent_ids: list[str] = None

    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []


class GeneticOptimizer:
    """Implements genetic algorithms for agent evolution."""

    def __init__(
        self,
        population_size: int = 18,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elitism_rate: float = 0.2,
    ) -> None:
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate

    def selection(
        self, population: list[AgentGenome], fitness_scores: list[float]
    ) -> list[AgentGenome]:
        """Tournament selection for breeding."""
        selected = []
        tournament_size = 3

        for _ in range(len(population)):
            tournament_indices = np.random.choice(
                len(population), tournament_size, replace=False
            )
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])

        return selected

    def crossover(
        self, parent1: AgentGenome, parent2: AgentGenome
    ) -> tuple[AgentGenome, AgentGenome]:
        """Uniform crossover of agent genomes."""
        if np.random.random() > self.crossover_rate:
            return parent1, parent2

        def cross_dict(dict1: dict, dict2: dict) -> tuple[dict, dict]:
            child1_dict, child2_dict = {}, {}
            for key in dict1:
                if np.random.random() < 0.5:
                    child1_dict[key] = dict1[key]
                    child2_dict[key] = dict2.get(key, dict1[key])
                else:
                    child1_dict[key] = dict2.get(key, dict1[key])
                    child2_dict[key] = dict1[key]
            return child1_dict, child2_dict

        # Cross architecture params
        arch1, arch2 = cross_dict(
            parent1.architecture_params, parent2.architecture_params
        )
        hyper1, hyper2 = cross_dict(parent1.hyperparameters, parent2.hyperparameters)
        spec1, spec2 = cross_dict(
            parent1.specialization_config, parent2.specialization_config
        )
        behav1, behav2 = cross_dict(parent1.behavior_weights, parent2.behavior_weights)
        learn1, learn2 = cross_dict(parent1.learning_config, parent2.learning_config)

        child1 = AgentGenome(
            agent_id=f"{parent1.agent_id}_child_{int(time.time())}",
            architecture_params=arch1,
            hyperparameters=hyper1,
            specialization_config=spec1,
            behavior_weights=behav1,
            code_templates=parent1.code_templates.copy(),
            learning_config=learn1,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.agent_id, parent2.agent_id],
        )

        child2 = AgentGenome(
            agent_id=f"{parent2.agent_id}_child_{int(time.time()) + 1}",
            architecture_params=arch2,
            hyperparameters=hyper2,
            specialization_config=spec2,
            behavior_weights=behav2,
            code_templates=parent2.code_templates.copy(),
            learning_config=learn2,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.agent_id, parent2.agent_id],
        )

        return child1, child2

    def mutate(self, genome: AgentGenome) -> AgentGenome:
        """Mutate agent genome."""
        if np.random.random() > self.mutation_rate:
            return genome

        mutated = AgentGenome(**asdict(genome))

        # Mutate hyperparameters
        for key, value in mutated.hyperparameters.items():
            if isinstance(value, int | float) and np.random.random() < 0.3:
                if isinstance(value, int):
                    mutated.hyperparameters[key] = max(
                        1, int(value * np.random.normal(1.0, 0.1))
                    )
                else:
                    mutated.hyperparameters[key] = max(
                        0.001, value * np.random.normal(1.0, 0.1)
                    )

        # Mutate behavior weights
        for key, value in mutated.behavior_weights.items():
            if np.random.random() < 0.2:
                mutated.behavior_weights[key] = max(
                    0.0, min(1.0, value + np.random.normal(0, 0.05))
                )

        # Mutate learning config
        for key, value in mutated.learning_config.items():
            if isinstance(value, int | float) and np.random.random() < 0.2:
                if isinstance(value, int):
                    mutated.learning_config[key] = max(
                        1, int(value * np.random.normal(1.0, 0.05))
                    )
                else:
                    mutated.learning_config[key] = max(
                        0.0001, value * np.random.normal(1.0, 0.05)
                    )

        return mutated


class KPITracker:
    """Tracks and analyzes agent performance metrics."""

    def __init__(self, storage_path: str = "evolution_data/kpi_history.json") -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.kpi_history: dict[str, list[AgentKPIs]] = {}
        self.load_history()

    def load_history(self) -> None:
        """Load KPI history from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                    for agent_id, kpi_list in data.items():
                        self.kpi_history[agent_id] = [
                            AgentKPIs(**kpi) for kpi in kpi_list
                        ]
            except Exception as e:
                logger.exception(f"Failed to load KPI history: {e}")
                self.kpi_history = {}

    def save_history(self) -> None:
        """Save KPI history to disk."""
        try:
            data = {}
            for agent_id, kpi_list in self.kpi_history.items():
                data[agent_id] = [asdict(kpi) for kpi in kpi_list]

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.exception(f"Failed to save KPI history: {e}")

    def record_kpis(self, kpis: AgentKPIs) -> None:
        """Record KPIs for an agent."""
        if kpis.agent_id not in self.kpi_history:
            self.kpi_history[kpis.agent_id] = []

        self.kpi_history[kpis.agent_id].append(kpis)

        # Keep only last 1000 records per agent
        if len(self.kpi_history[kpis.agent_id]) > 1000:
            self.kpi_history[kpis.agent_id] = self.kpi_history[kpis.agent_id][-1000:]

        self.save_history()

    def get_fitness_scores(
        self, agent_ids: list[str], lookback_hours: int = 24
    ) -> dict[str, float]:
        """Get recent fitness scores for agents."""
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        fitness_scores = {}

        for agent_id in agent_ids:
            if agent_id not in self.kpi_history:
                fitness_scores[agent_id] = 0.0
                continue

            recent_kpis = [
                kpi
                for kpi in self.kpi_history[agent_id]
                if kpi.timestamp >= cutoff_time
            ]

            if recent_kpis:
                avg_fitness = np.mean([kpi.fitness_score() for kpi in recent_kpis])
                fitness_scores[agent_id] = avg_fitness
            else:
                fitness_scores[agent_id] = 0.0

        return fitness_scores

    def get_performance_trends(self, agent_id: str) -> dict[str, list[float]]:
        """Get performance trends for visualization."""
        if agent_id not in self.kpi_history:
            return {}

        kpis = self.kpi_history[agent_id][-100:]  # Last 100 records
        trends = {}

        for field in ["task_success_rate", "user_satisfaction", "resource_efficiency"]:
            trends[field] = [getattr(kpi, field) for kpi in kpis]

        return trends


class CodeMutator:
    """Safe code modification for agent self-improvement."""

    def __init__(self, sandbox_path: str = "evolution_data/sandbox") -> None:
        self.sandbox_path = Path(sandbox_path)
        self.sandbox_path.mkdir(parents=True, exist_ok=True)
        self.safe_mutations = [
            "adjust_hyperparameters",
            "modify_prompt_templates",
            "update_behavior_weights",
            "refine_specialization",
        ]

    async def mutate_agent_code(self, genome: AgentGenome) -> AgentGenome:
        """Safely mutate agent code based on genome."""
        mutation_type = np.random.choice(self.safe_mutations)

        try:
            if mutation_type == "adjust_hyperparameters":
                return self._mutate_hyperparameters(genome)
            if mutation_type == "modify_prompt_templates":
                return await self._mutate_prompt_templates(genome)
            if mutation_type == "update_behavior_weights":
                return self._mutate_behavior_weights(genome)
            if mutation_type == "refine_specialization":
                return self._mutate_specialization(genome)
        except Exception as e:
            logger.exception(f"Code mutation failed for {genome.agent_id}: {e}")
            return genome

        return genome

    def _mutate_hyperparameters(self, genome: AgentGenome) -> AgentGenome:
        """Mutate hyperparameters safely."""
        mutated = AgentGenome(**asdict(genome))

        for key, value in mutated.hyperparameters.items():
            if isinstance(value, float) and "rate" in key.lower():
                # Learning rates, mutation rates etc.
                mutated.hyperparameters[key] = max(
                    0.0001, min(0.1, value * np.random.normal(1.0, 0.1))
                )
            elif isinstance(value, int) and "size" in key.lower():
                # Batch sizes, hidden sizes etc.
                mutated.hyperparameters[key] = max(
                    1, int(value * np.random.normal(1.0, 0.1))
                )

        return mutated

    async def _mutate_prompt_templates(self, genome: AgentGenome) -> AgentGenome:
        """Mutate prompt templates."""
        mutated = AgentGenome(**asdict(genome))

        # Simple template variations - in production would use more sophisticated NLP
        template_variations = {
            "system_prompt": [
                "You are an expert assistant focused on",
                "As a specialized AI agent, your role is to",
                "Your expertise lies in providing",
            ],
            "task_prompt": [
                "Please analyze and provide",
                "Your task is to evaluate and",
                "Consider the following and determine",
            ],
        }

        for template_key in mutated.code_templates:
            if np.random.random() < 0.1:  # 10% chance to mutate each template
                if "system" in template_key.lower():
                    prefix = np.random.choice(template_variations["system_prompt"])
                    mutated.code_templates[template_key] = (
                        f"{prefix} {mutated.code_templates[template_key].split(' ', 5)[-1]}"
                    )

        return mutated

    def _mutate_behavior_weights(self, genome: AgentGenome) -> AgentGenome:
        """Mutate behavior weights."""
        mutated = AgentGenome(**asdict(genome))

        for key, weight in mutated.behavior_weights.items():
            if np.random.random() < 0.2:
                mutated.behavior_weights[key] = max(
                    0.0, min(1.0, weight + np.random.normal(0, 0.05))
                )

        # Normalize weights
        total_weight = sum(mutated.behavior_weights.values())
        if total_weight > 0:
            for key in mutated.behavior_weights:
                mutated.behavior_weights[key] /= total_weight

        return mutated

    def _mutate_specialization(self, genome: AgentGenome) -> AgentGenome:
        """Mutate specialization configuration."""
        mutated = AgentGenome(**asdict(genome))

        # Adjust specialization focus
        if "focus_areas" in mutated.specialization_config:
            focus_areas = mutated.specialization_config["focus_areas"]
            if isinstance(focus_areas, dict):
                for area, weight in focus_areas.items():
                    if np.random.random() < 0.15:
                        mutated.specialization_config["focus_areas"][area] = max(
                            0.0, min(1.0, weight + np.random.normal(0, 0.1))
                        )

        return mutated


class MetaLearner:
    """Learns how to learn better - optimizes learning strategies."""

    def __init__(self) -> None:
        self.learning_history = {}
        self.strategy_performance = {}

    async def optimize_learning_strategy(
        self, agent_id: str, current_performance: float, learning_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Optimize learning strategy based on performance history."""
        if agent_id not in self.learning_history:
            self.learning_history[agent_id] = []

        # Record current strategy performance
        self.learning_history[agent_id].append(
            {
                "timestamp": datetime.now(),
                "performance": current_performance,
                "config": learning_config.copy(),
            }
        )

        # Keep only recent history
        if len(self.learning_history[agent_id]) > 50:
            self.learning_history[agent_id] = self.learning_history[agent_id][-50:]

        # Analyze what configurations work best
        if len(self.learning_history[agent_id]) >= 5:
            return self._analyze_and_optimize(agent_id, learning_config)

        return learning_config

    def _analyze_and_optimize(
        self, agent_id: str, current_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze learning history and optimize configuration."""
        history = self.learning_history[agent_id]

        # Find top performing configurations
        sorted_history = sorted(history, key=lambda x: x["performance"], reverse=True)
        top_configs = sorted_history[:5]

        # Extract common patterns from top performers
        optimized_config = current_config.copy()

        # Average the best performing hyperparameters
        for key in current_config:
            if isinstance(current_config[key], int | float):
                values = [
                    config["config"].get(key, current_config[key])
                    for config in top_configs
                ]
                if isinstance(current_config[key], int):
                    optimized_config[key] = int(np.mean(values))
                else:
                    optimized_config[key] = np.mean(values)

        return optimized_config


class SpecializationManager:
    """Manages dynamic agent specialization and role assignment."""

    def __init__(self) -> None:
        self.agent_roles = {}
        self.task_performance_matrix = {}
        self.specialization_templates = self._load_specialization_templates()

    def _load_specialization_templates(self) -> dict[str, dict]:
        """Load specialization templates for the 18-agent ecosystem."""
        return {
            "general-purpose": {
                "focus_areas": {"general": 1.0},
                "behavior_weights": {"adaptability": 0.8, "specialization": 0.2},
            },
            "code-quality": {
                "focus_areas": {
                    "code_review": 0.4,
                    "static_analysis": 0.3,
                    "best_practices": 0.3,
                },
                "behavior_weights": {"precision": 0.9, "speed": 0.1},
            },
            "test-coverage": {
                "focus_areas": {
                    "unit_testing": 0.4,
                    "integration_testing": 0.3,
                    "test_analysis": 0.3,
                },
                "behavior_weights": {"thoroughness": 0.8, "efficiency": 0.2},
            },
            "security-audit": {
                "focus_areas": {
                    "vulnerability_scan": 0.4,
                    "security_review": 0.3,
                    "compliance": 0.3,
                },
                "behavior_weights": {"security": 1.0, "performance": 0.0},
            },
            "performance-monitor": {
                "focus_areas": {"metrics": 0.4, "profiling": 0.3, "optimization": 0.3},
                "behavior_weights": {"accuracy": 0.7, "real_time": 0.3},
            },
            "deployment-automation": {
                "focus_areas": {
                    "ci_cd": 0.4,
                    "infrastructure": 0.3,
                    "orchestration": 0.3,
                },
                "behavior_weights": {"reliability": 0.8, "speed": 0.2},
            },
            "dependency-manager": {
                "focus_areas": {
                    "package_analysis": 0.4,
                    "version_control": 0.3,
                    "security_scan": 0.3,
                },
                "behavior_weights": {"stability": 0.7, "innovation": 0.3},
            },
            "mobile-optimizer": {
                "focus_areas": {
                    "mobile_perf": 0.4,
                    "ui_optimization": 0.3,
                    "battery_efficiency": 0.3,
                },
                "behavior_weights": {"user_experience": 0.6, "performance": 0.4},
            },
            "mesh-network-engineer": {
                "focus_areas": {
                    "network_topology": 0.4,
                    "routing": 0.3,
                    "fault_tolerance": 0.3,
                },
                "behavior_weights": {"reliability": 0.8, "efficiency": 0.2},
            },
            "integration-testing": {
                "focus_areas": {
                    "api_testing": 0.4,
                    "system_integration": 0.3,
                    "e2e_testing": 0.3,
                },
                "behavior_weights": {"coverage": 0.7, "speed": 0.3},
            },
            "federated-learning-coordinator": {
                "focus_areas": {
                    "distributed_learning": 0.4,
                    "model_aggregation": 0.3,
                    "privacy": 0.3,
                },
                "behavior_weights": {"coordination": 0.8, "privacy": 0.2},
            },
            "experimental-validator": {
                "focus_areas": {
                    "experiment_design": 0.4,
                    "statistical_analysis": 0.3,
                    "validation": 0.3,
                },
                "behavior_weights": {"rigor": 0.9, "innovation": 0.1},
            },
            "documentation-sync": {
                "focus_areas": {
                    "doc_generation": 0.4,
                    "sync_management": 0.3,
                    "quality_check": 0.3,
                },
                "behavior_weights": {"accuracy": 0.8, "automation": 0.2},
            },
            "blockchain-architect": {
                "focus_areas": {
                    "smart_contracts": 0.4,
                    "consensus": 0.3,
                    "scalability": 0.3,
                },
                "behavior_weights": {"security": 0.7, "scalability": 0.3},
            },
            "atlantis-vision-tracker": {
                "focus_areas": {
                    "vision_alignment": 0.4,
                    "progress_tracking": 0.3,
                    "strategic_analysis": 0.3,
                },
                "behavior_weights": {"vision_consistency": 0.9, "adaptability": 0.1},
            },
            "agent-evolution-optimizer": {
                "focus_areas": {
                    "evolution_algorithms": 0.4,
                    "performance_optimization": 0.3,
                    "meta_learning": 0.3,
                },
                "behavior_weights": {"optimization": 0.8, "stability": 0.2},
            },
            "quantum-optimizer": {
                "focus_areas": {
                    "quantum_algorithms": 0.4,
                    "optimization": 0.3,
                    "simulation": 0.3,
                },
                "behavior_weights": {"innovation": 0.7, "precision": 0.3},
            },
            "edge-computing-coordinator": {
                "focus_areas": {
                    "edge_deployment": 0.4,
                    "resource_management": 0.3,
                    "latency_optimization": 0.3,
                },
                "behavior_weights": {"efficiency": 0.8, "reliability": 0.2},
            },
        }

    async def optimize_specialization(
        self,
        agent_id: str,
        current_genome: AgentGenome,
        task_performance_history: dict[str, float],
    ) -> AgentGenome:
        """Optimize agent specialization based on task performance."""
        # Analyze which tasks this agent performs best at
        best_tasks = sorted(
            task_performance_history.items(), key=lambda x: x[1], reverse=True
        )[:3]

        if not best_tasks:
            return current_genome

        # Find matching specialization template
        best_specialization = None
        best_match_score = 0.0

        for spec_name, spec_config in self.specialization_templates.items():
            match_score = self._calculate_specialization_match(best_tasks, spec_config)
            if match_score > best_match_score:
                best_match_score = match_score
                best_specialization = spec_name

        if best_specialization and best_match_score > 0.7:
            # Apply specialization
            optimized_genome = AgentGenome(**asdict(current_genome))
            optimized_genome.specialization_config.update(
                self.specialization_templates[best_specialization]
            )

            # Update behavior weights for specialization
            spec_weights = self.specialization_templates[best_specialization][
                "behavior_weights"
            ]
            for weight_key, weight_value in spec_weights.items():
                optimized_genome.behavior_weights[weight_key] = weight_value

            logger.info(f"Specialized agent {agent_id} as {best_specialization}")
            return optimized_genome

        return current_genome

    def _calculate_specialization_match(
        self, best_tasks: list[tuple[str, float]], spec_config: dict
    ) -> float:
        """Calculate how well tasks match a specialization template."""
        focus_areas = spec_config.get("focus_areas", {})
        match_score = 0.0

        for task_name, performance in best_tasks:
            for focus_area in focus_areas:
                if (
                    focus_area.lower() in task_name.lower()
                    or task_name.lower() in focus_area.lower()
                ):
                    match_score += performance * focus_areas[focus_area]

        return min(match_score, 1.0)


class AgentEvolutionEngine:
    """Main orchestration system for the self-evolving 18-agent ecosystem."""

    def __init__(
        self, evolution_data_path: str = "evolution_data", population_size: int = 18
    ) -> None:
        self.evolution_data_path = Path(evolution_data_path)
        self.evolution_data_path.mkdir(parents=True, exist_ok=True)

        self.population_size = population_size
        self.current_generation = 0

        # Initialize components
        self.genetic_optimizer = GeneticOptimizer(population_size=population_size)
        self.kpi_tracker = KPITracker(f"{evolution_data_path}/kpi_history.json")
        self.code_mutator = CodeMutator(f"{evolution_data_path}/sandbox")
        self.meta_learner = MetaLearner()
        self.specialization_manager = SpecializationManager()

        # Agent population
        self.agent_population: list[AgentGenome] = []
        self.active_agents: dict[str, Any] = {}

        # Evolution state
        self.evolution_history = []
        self.best_performers = []

        logger.info(
            f"AgentEvolutionEngine initialized with population size {population_size}"
        )

    async def initialize_population(self) -> None:
        """Initialize the 18-agent population with diverse specializations."""
        base_templates = list(
            self.specialization_manager.specialization_templates.keys()
        )

        for i in range(self.population_size):
            agent_id = f"agent_{i:02d}_{base_templates[i % len(base_templates)]}"
            specialization = base_templates[i % len(base_templates)]

            genome = self._create_initial_genome(agent_id, specialization)
            self.agent_population.append(genome)

        logger.info(f"Initialized population with {len(self.agent_population)} agents")

    def _create_initial_genome(self, agent_id: str, specialization: str) -> AgentGenome:
        """Create initial genome for an agent."""
        spec_config = self.specialization_manager.specialization_templates[
            specialization
        ]

        return AgentGenome(
            agent_id=agent_id,
            architecture_params={
                "hidden_size": np.random.choice([128, 256, 512]),
                "num_layers": np.random.choice([2, 3, 4]),
                "attention_heads": np.random.choice([4, 8, 12]),
                "dropout_rate": np.random.uniform(0.1, 0.3),
            },
            hyperparameters={
                "learning_rate": np.random.uniform(0.0001, 0.01),
                "batch_size": np.random.choice([16, 32, 64]),
                "weight_decay": np.random.uniform(0.0001, 0.001),
                "warmup_steps": np.random.choice([100, 500, 1000]),
            },
            specialization_config=spec_config.copy(),
            behavior_weights=spec_config["behavior_weights"].copy(),
            code_templates={
                "system_prompt": f"You are a specialized {specialization} agent.",
                "task_prompt": "Analyze the following task and provide expert guidance.",
                "error_prompt": "Handle this error with your specialized knowledge.",
            },
            learning_config={
                "meta_learning_rate": 0.001,
                "adaptation_steps": 5,
                "memory_size": 1000,
            },
            generation=0,
        )

    async def run_evolution_cycle(
        self, generations: int = 10, evaluation_tasks: list[Callable] | None = None
    ) -> dict[str, Any]:
        """Run complete evolution cycle."""
        if not self.agent_population:
            await self.initialize_population()

        evolution_results = {
            "initial_population": len(self.agent_population),
            "generations_run": 0,
            "best_fitness_history": [],
            "diversity_history": [],
            "specialization_distribution": [],
        }

        for generation in range(generations):
            logger.info(f"Starting evolution generation {generation + 1}/{generations}")

            # Evaluate current population
            fitness_scores = await self._evaluate_population(evaluation_tasks)

            # Record best fitness
            best_fitness = max(fitness_scores.values()) if fitness_scores else 0.0
            evolution_results["best_fitness_history"].append(best_fitness)

            # Calculate diversity
            diversity = self._calculate_population_diversity()
            evolution_results["diversity_history"].append(diversity)

            # Record specialization distribution
            spec_dist = self._get_specialization_distribution()
            evolution_results["specialization_distribution"].append(spec_dist)

            # Evolve population
            await self._evolve_population(fitness_scores)

            # Update generation counter
            self.current_generation += 1
            evolution_results["generations_run"] = generation + 1

            # Log progress
            logger.info(
                f"Generation {generation + 1} complete. Best fitness: {best_fitness:.4f}"
            )

        # Save evolution results
        await self._save_evolution_results(evolution_results)

        return evolution_results

    async def _evaluate_population(
        self, evaluation_tasks: list[Callable] | None = None
    ) -> dict[str, float]:
        """Evaluate current population fitness."""
        fitness_scores = {}

        for genome in self.agent_population:
            try:
                # Get recent KPI-based fitness
                recent_fitness = self.kpi_tracker.get_fitness_scores(
                    [genome.agent_id], lookback_hours=24
                )
                base_fitness = recent_fitness.get(genome.agent_id, 0.0)

                # Run additional evaluation tasks if provided
                task_fitness = 0.0
                if evaluation_tasks:
                    task_results = []
                    for task in evaluation_tasks[
                        :3
                    ]:  # Limit to 3 tasks for performance
                        try:
                            result = await task(genome)
                            task_results.append(result)
                        except Exception as e:
                            logger.warning(
                                f"Evaluation task failed for {genome.agent_id}: {e}"
                            )
                            task_results.append(0.0)

                    task_fitness = np.mean(task_results) if task_results else 0.0

                # Combine fitness scores
                combined_fitness = 0.7 * base_fitness + 0.3 * task_fitness
                fitness_scores[genome.agent_id] = combined_fitness

            except Exception as e:
                logger.exception(f"Failed to evaluate agent {genome.agent_id}: {e}")
                fitness_scores[genome.agent_id] = 0.0

        return fitness_scores

    async def _evolve_population(self, fitness_scores: dict[str, float]) -> None:
        """Evolve the agent population."""
        # Sort population by fitness
        sorted_agents = sorted(
            self.agent_population,
            key=lambda x: fitness_scores.get(x.agent_id, 0.0),
            reverse=True,
        )

        # Keep elite agents (top 20%)
        elite_count = max(
            1, int(self.population_size * self.genetic_optimizer.elitism_rate)
        )
        new_population = sorted_agents[:elite_count].copy()

        # Generate offspring for the rest
        fitness_values = [
            fitness_scores.get(agent.agent_id, 0.0) for agent in sorted_agents
        ]

        while len(new_population) < self.population_size:
            # Selection
            selected = self.genetic_optimizer.selection(sorted_agents, fitness_values)

            # Crossover
            if len(selected) >= 2:
                parent1, parent2 = np.random.choice(selected, 2, replace=False)
                child1, child2 = self.genetic_optimizer.crossover(parent1, parent2)

                # Mutation
                child1 = self.genetic_optimizer.mutate(child1)
                child2 = self.genetic_optimizer.mutate(child2)

                # Code mutation
                child1 = await self.code_mutator.mutate_agent_code(child1)
                child2 = await self.code_mutator.mutate_agent_code(child2)

                # Meta-learning optimization
                child1.learning_config = (
                    await self.meta_learner.optimize_learning_strategy(
                        child1.agent_id,
                        fitness_scores.get(parent1.agent_id, 0.0),
                        child1.learning_config,
                    )
                )
                child2.learning_config = (
                    await self.meta_learner.optimize_learning_strategy(
                        child2.agent_id,
                        fitness_scores.get(parent2.agent_id, 0.0),
                        child2.learning_config,
                    )
                )

                # Specialization optimization
                child1 = await self.specialization_manager.optimize_specialization(
                    child1.agent_id, child1, {}
                )
                child2 = await self.specialization_manager.optimize_specialization(
                    child2.agent_id, child2, {}
                )

                new_population.extend([child1, child2])

        # Trim to exact population size
        self.agent_population = new_population[: self.population_size]

        logger.info(
            f"Population evolved. New generation: {self.current_generation + 1}"
        )

    def _calculate_population_diversity(self) -> float:
        """Calculate genetic diversity of current population."""
        if len(self.agent_population) < 2:
            return 0.0

        diversity_sum = 0.0
        pair_count = 0

        for i in range(len(self.agent_population)):
            for j in range(i + 1, len(self.agent_population)):
                diversity = self._calculate_genome_distance(
                    self.agent_population[i], self.agent_population[j]
                )
                diversity_sum += diversity
                pair_count += 1

        return diversity_sum / pair_count if pair_count > 0 else 0.0

    def _calculate_genome_distance(
        self, genome1: AgentGenome, genome2: AgentGenome
    ) -> float:
        """Calculate distance between two genomes."""
        distance = 0.0

        # Architecture parameter distance
        for key in genome1.architecture_params:
            val1 = genome1.architecture_params.get(key, 0)
            val2 = genome2.architecture_params.get(key, 0)
            if isinstance(val1, int | float) and isinstance(val2, int | float):
                distance += abs(val1 - val2) / max(abs(val1), abs(val2), 1.0)

        # Behavior weight distance
        for key in genome1.behavior_weights:
            val1 = genome1.behavior_weights.get(key, 0)
            val2 = genome2.behavior_weights.get(key, 0)
            distance += abs(val1 - val2)

        return distance / (
            len(genome1.architecture_params) + len(genome1.behavior_weights)
        )

    def _get_specialization_distribution(self) -> dict[str, int]:
        """Get distribution of specializations in current population."""
        distribution = {}

        for genome in self.agent_population:
            # Extract specialization from agent_id or config
            spec = "unknown"
            for template_name in self.specialization_manager.specialization_templates:
                if template_name in genome.agent_id:
                    spec = template_name
                    break

            distribution[spec] = distribution.get(spec, 0) + 1

        return distribution

    async def _save_evolution_results(self, results: dict[str, Any]) -> None:
        """Save evolution results to disk."""
        results_path = (
            self.evolution_data_path
            / f"evolution_results_gen_{self.current_generation}.json"
        )

        try:
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            # Also save current population
            population_path = (
                self.evolution_data_path
                / f"population_gen_{self.current_generation}.json"
            )
            population_data = [asdict(genome) for genome in self.agent_population]

            with open(population_path, "w") as f:
                json.dump(population_data, f, indent=2, default=str)

            logger.info(f"Evolution results saved to {results_path}")

        except Exception as e:
            logger.exception(f"Failed to save evolution results: {e}")

    async def get_evolution_dashboard_data(self) -> dict[str, Any]:
        """Get data for evolution monitoring dashboard."""
        # Calculate current fitness scores
        fitness_scores = self.kpi_tracker.get_fitness_scores(
            [genome.agent_id for genome in self.agent_population]
        )

        # Get performance trends
        trends = {}
        for genome in self.agent_population[:5]:  # Top 5 agents
            trends[genome.agent_id] = self.kpi_tracker.get_performance_trends(
                genome.agent_id
            )

        # Population statistics
        population_stats = {
            "total_agents": len(self.agent_population),
            "current_generation": self.current_generation,
            "avg_fitness": (
                np.mean(list(fitness_scores.values())) if fitness_scores else 0.0
            ),
            "max_fitness": max(fitness_scores.values()) if fitness_scores else 0.0,
            "diversity": self._calculate_population_diversity(),
            "specialization_distribution": self._get_specialization_distribution(),
        }

        return {
            "population_stats": population_stats,
            "fitness_scores": fitness_scores,
            "performance_trends": trends,
            "timestamp": datetime.now().isoformat(),
        }

    async def emergency_rollback(self, generations_back: int = 1) -> bool | None:
        """Emergency rollback to previous generation."""
        try:
            target_generation = max(0, self.current_generation - generations_back)
            population_path = (
                self.evolution_data_path / f"population_gen_{target_generation}.json"
            )

            if population_path.exists():
                with open(population_path) as f:
                    population_data = json.load(f)

                self.agent_population = [
                    AgentGenome(**genome_data) for genome_data in population_data
                ]
                self.current_generation = target_generation

                logger.info(
                    f"Successfully rolled back to generation {target_generation}"
                )
                return True
            logger.error(
                f"Rollback failed: No backup found for generation {target_generation}"
            )
            return False

        except Exception as e:
            logger.exception(f"Emergency rollback failed: {e}")
            return False


# Evolution utility functions
async def create_evaluation_task(task_type: str) -> Callable:
    """Create evaluation task for agent fitness testing."""

    async def code_quality_task(genome: AgentGenome) -> float:
        """Evaluate code quality capabilities."""
        # Simulate code quality evaluation
        quality_focus = genome.specialization_config.get("focus_areas", {}).get(
            "code_review", 0.0
        )
        return min(1.0, quality_focus + np.random.normal(0, 0.1))

    async def performance_task(genome: AgentGenome) -> float:
        """Evaluate performance optimization capabilities."""
        perf_weight = genome.behavior_weights.get("performance", 0.5)
        return min(1.0, perf_weight + np.random.normal(0, 0.1))

    async def collaboration_task(genome: AgentGenome) -> float:
        """Evaluate collaboration capabilities."""
        collab_score = genome.behavior_weights.get("collaboration", 0.5)
        return min(1.0, collab_score + np.random.normal(0, 0.1))

    task_map = {
        "code_quality": code_quality_task,
        "performance": performance_task,
        "collaboration": collaboration_task,
    }

    return task_map.get(task_type, code_quality_task)


if __name__ == "__main__":

    async def main() -> None:
        # Initialize evolution engine
        engine = AgentEvolutionEngine()

        # Create evaluation tasks
        eval_tasks = [
            await create_evaluation_task("code_quality"),
            await create_evaluation_task("performance"),
            await create_evaluation_task("collaboration"),
        ]

        # Run evolution
        results = await engine.run_evolution_cycle(
            generations=5, evaluation_tasks=eval_tasks
        )

        print("Evolution completed!")
        print(f"Best fitness achieved: {max(results['best_fitness_history']):.4f}")
        print(f"Final diversity: {results['diversity_history'][-1]:.4f}")

        # Get dashboard data
        dashboard_data = await engine.get_evolution_dashboard_data()
        print(f"Current population stats: {dashboard_data['population_stats']}")

    asyncio.run(main())
