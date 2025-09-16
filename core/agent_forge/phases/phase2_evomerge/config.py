"""
EvoMerge Configuration
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch

@dataclass
class EvoMergeConfig:
    """Configuration for EvoMerge phase."""

    # Evolution parameters
    generations: int = 50
    population_size: int = 8
    elite_size: int = 2
    tournament_size: int = 3

    # Genetic operation rates
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    mutation_strength: float = 0.05

    # Merge techniques
    techniques: List[str] = field(default_factory=lambda: [
        'linear', 'slerp', 'ties', 'dare', 'frankenmerge', 'dfs'
    ])
    technique_weights: Optional[Dict[str, float]] = None

    # Fitness evaluation
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        'perplexity': 0.4,
        'accuracy': 0.3,
        'speed': 0.2,
        'memory': 0.1
    })

    # Convergence criteria
    convergence_threshold: float = 0.001
    convergence_patience: int = 5
    early_stopping: bool = True

    # Diversity management
    diversity_weight: float = 0.3
    min_diversity: float = 0.2
    diversity_penalty: float = 0.1

    # Performance optimization
    enable_parallel: bool = True
    num_workers: int = 4
    enable_caching: bool = True
    cache_size_mb: int = 512

    # GPU settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    gradient_checkpointing: bool = True

    # Checkpointing
    checkpoint_interval: int = 10
    checkpoint_dir: str = "./checkpoints/evomerge"
    keep_checkpoints: int = 3

    # Logging
    log_level: str = "INFO"
    log_interval: int = 1
    wandb_project: Optional[str] = "agent-forge-evomerge"

    # Model preferences
    prefer_seeds: bool = False  # Whether to prefer seed models over base models
    seed_models: Optional[List[str]] = None  # List of seed model paths

    def __post_init__(self):
        """Validate and adjust configuration."""
        # Ensure elite size is less than population size
        if self.elite_size >= self.population_size:
            self.elite_size = max(1, self.population_size // 4)

        # Ensure tournament size is reasonable
        if self.tournament_size > self.population_size:
            self.tournament_size = min(3, self.population_size)

        # Initialize technique weights if not provided
        if self.technique_weights is None:
            self.technique_weights = {
                tech: 1.0 / len(self.techniques)
                for tech in self.techniques
            }

        # Normalize fitness weights
        total_weight = sum(self.fitness_weights.values())
        if total_weight > 0:
            self.fitness_weights = {
                k: v / total_weight
                for k, v in self.fitness_weights.items()
            }

@dataclass
class MergeResult:
    """Result from a merge operation."""
    model: torch.nn.Module
    technique: str
    fitness: float
    metrics: Dict[str, float]
    generation: int
    parent_ids: List[int]

@dataclass
class EvolutionState:
    """Current state of evolution."""
    generation: int
    best_fitness: float
    average_fitness: float
    diversity: float
    convergence_counter: int
    population: List[torch.nn.Module]
    fitness_history: List[float]
    diversity_history: List[float]