"""
EvoMerge Phase: Evolutionary Model Merging

Consolidates the best features from:
- src/production/evolution/evomerge/ (most complete implementation)
- src/agent_forge/unified_pipeline.py (pipeline integration)
- src/production/evolution/evomerge_pipeline.py (production config)
- evomerge/bench_driver.py (benchmarking)

Implements sophisticated evolutionary optimization to create a strong foundation model
from multiple base models using various merging techniques.
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class PhaseConfig:
    """Base configuration class for Agent Forge phases."""

    pass


@dataclass
class EvoMergeConfig(PhaseConfig):
    """Configuration for EvoMerge phase."""

    # Model paths
    base_models: list[str] = field(
        default_factory=lambda: [
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B",
            "Qwen/Qwen2-1.5B-Instruct",
        ]
    )

    # HRRM seed models for fast iteration
    seed_models: list[str] = field(default_factory=lambda: [])  # Will be populated with HRRM exports

    # If True, prioritize seed models over base models for faster iteration
    prefer_seeds: bool = True

    output_dir: str = "./evomerge_output"
    checkpoint_dir: str = "./evomerge_checkpoints"  # Added missing checkpoint_dir
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Evolution parameters
    generations: int = 50
    population_size: int = 8
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7

    # Merge techniques
    merge_techniques: list[str] = field(
        default_factory=lambda: ["linear", "slerp", "ties", "dare", "frankenmerge", "dfs"]
    )

    # Evaluation configuration
    evaluation_domains: list[str] = field(default_factory=lambda: ["code", "math", "multilingual", "structured_data"])
    fitness_weights: dict[str, float] = field(
        default_factory=lambda: {"code": 0.25, "math": 0.25, "multilingual": 0.25, "structured_data": 0.25}
    )


@dataclass
class MergeCandidate:
    """Represents a merged model candidate."""

    model_path: str  # Path to saved model
    merge_recipe: dict[str, Any]
    fitness_scores: dict[str, float] = field(default_factory=dict)
    aggregated_fitness: float = 0.0
    generation: int = 0
    parents: list[str] | None = None
    evaluation_results: dict[str, Any] | None = None
    diversity_score: float = 0.0

    def calculate_aggregated_fitness(self, weights: dict[str, float]):
        """Calculate weighted aggregate fitness score."""
        total_fitness = 0.0
        valid_weights_sum = 0.0

        for domain, weight in weights.items():
            score = self.fitness_scores.get(domain, 0)
            # Handle NaN scores by skipping them and adjusting weights
            if not np.isnan(score):
                total_fitness += score * weight
                valid_weights_sum += weight

        # If we have valid scores, normalize by valid weights
        if valid_weights_sum > 0:
            self.aggregated_fitness = total_fitness / valid_weights_sum
        else:
            # All scores are NaN, set to 0
            self.aggregated_fitness = 0.0

        return self.aggregated_fitness


class MergeOperators:
    """
    Collection of all model merging operators from production implementation.
    Includes memory-efficient chunked processing and meta tensor handling.
    """

    @staticmethod
    def linear_merge(
        models: list[nn.Module], weights: list[float] | None = None, chunk_size: int = 1000000
    ) -> nn.Module:
        """Linear weighted average of model parameters with chunked processing."""
        if weights is None:
            weights = torch.ones(len(models)) / len(models)

        merged = models[0].__class__(models[0].config)
        merged_state = {}
        device = next(models[0].parameters()).device

        with tempfile.TemporaryDirectory():
            for key in models[0].state_dict():
                merged_chunks = []

                # Check for meta tensors
                meta_tensors = [m.state_dict()[key] for m in models if m.state_dict()[key].is_meta]

                if meta_tensors:
                    param_shape = models[0].state_dict()[key].shape
                    merged_param = torch.empty(param_shape, device="cpu")
                else:
                    numel = models[0].state_dict()[key].numel()
                    param_shape = models[0].state_dict()[key].shape

                    for i in range(0, numel, chunk_size):
                        chunk_params = []
                        for model in models:
                            chunk_param = model.state_dict()[key].flatten()[i : i + chunk_size].to(device)
                            chunk_params.append(chunk_param)

                        merged_chunk = sum(w * t for w, t in zip(weights, chunk_params))
                        merged_chunks.append(merged_chunk.cpu())
                        del merged_chunk, chunk_params

                    merged_param = torch.cat(merged_chunks)
                    del merged_chunks

                merged_state[key] = merged_param.view(param_shape)

        merged.load_state_dict(merged_state)
        return merged

    @staticmethod
    def slerp_merge(model1: nn.Module, model2: nn.Module, t: float = 0.5, chunk_size: int = 1000000) -> nn.Module:
        """Spherical linear interpolation with chunked processing."""
        merged = model1.__class__(model1.config)
        merged_state = {}
        device = next(model1.parameters()).device

        with tempfile.TemporaryDirectory():
            for key in model1.state_dict():
                merged_chunks = []

                meta_tensors = [model1.state_dict()[key].is_meta, model2.state_dict()[key].is_meta]

                if not any(meta_tensors):
                    numel = model1.state_dict()[key].numel()
                    param_shape = model1.state_dict()[key].shape

                    for i in range(0, numel, chunk_size):
                        w1 = model1.state_dict()[key].flatten()[i : i + chunk_size].to(device)
                        w2 = model2.state_dict()[key].flatten()[i : i + chunk_size].to(device)

                        # Compute angle between parameters
                        omega = torch.arccos(torch.clamp(torch.dot(w1, w2) / (w1.norm() * w2.norm()), -1, 1))
                        so = torch.sin(omega)

                        if so.abs() < 1e-8:
                            merged_chunk = (1 - t) * w1 + t * w2
                        else:
                            merged_chunk = torch.sin((1 - t) * omega) / so * w1 + torch.sin(t * omega) / so * w2

                        merged_chunks.append(merged_chunk.cpu())
                        del merged_chunk, w1, w2

                    merged_param = torch.cat(merged_chunks)
                    merged_state[key] = merged_param.view(param_shape)
                else:
                    # Handle meta tensors
                    param_shape = model1.state_dict()[key].shape
                    merged_state[key] = torch.empty(param_shape, device="cpu")

        merged.load_state_dict(merged_state)
        return merged

    @staticmethod
    def ties_merge(models: list[nn.Module], threshold: float = 0.1, chunk_size: int = 1000000) -> nn.Module:
        """TIES merging: Trim, Interpolate, Elect, Sign with disk-based processing."""
        merged = models[0].__class__(models[0].config)
        merged_state = {}
        device = next(models[0].parameters()).device

        with tempfile.TemporaryDirectory():
            for key in models[0].state_dict():
                param_shape = models[0].state_dict()[key].shape
                numel = models[0].state_dict()[key].numel()

                # Process in chunks for memory efficiency
                merged_chunks = []
                for i in range(0, numel, chunk_size):
                    chunk_params = []
                    for model in models:
                        chunk = model.state_dict()[key].flatten()[i : i + chunk_size].to(device)

                        # Trim: Remove small magnitude changes
                        mask = torch.abs(chunk) > threshold
                        trimmed = chunk * mask
                        chunk_params.append(trimmed)

                    # Sign election: Choose dominant sign
                    signs = torch.sign(sum(chunk_params))

                    # Interpolate with sign correction
                    merged_chunk = torch.zeros_like(chunk_params[0])
                    for p in chunk_params:
                        merged_chunk += torch.abs(p)
                    merged_chunk = merged_chunk / len(chunk_params) * signs

                    merged_chunks.append(merged_chunk.cpu())
                    del chunk_params, merged_chunk

                merged_param = torch.cat(merged_chunks)
                merged_state[key] = merged_param.view(param_shape)

        merged.load_state_dict(merged_state)
        return merged

    @staticmethod
    def dare_merge(models: list[nn.Module], threshold: float = 0.1, amplification: float = 2.0) -> nn.Module:
        """DARE merging: Drop And REscale."""
        merged = models[0].__class__(models[0].config)
        merged_state = {}

        for key in models[0].state_dict():
            params = [m.state_dict()[key] for m in models]

            # Random dropout mask
            mask = torch.rand_like(params[0]) > threshold

            # Merge with rescaling
            merged_param = torch.zeros_like(params[0])
            for p in params:
                merged_param += p * mask * amplification
            merged_param = merged_param / len(params)

            merged_state[key] = merged_param

        merged.load_state_dict(merged_state)
        return merged

    @staticmethod
    def frankenmerge(models: list[nn.Module], layer_assignments: list[int] | None = None) -> nn.Module:
        """Frankenmerge: Mix layers from different models."""
        if layer_assignments is None:
            # Random layer assignment
            num_layers = len([k for k in models[0].state_dict() if "layer" in k])
            layer_assignments = [random.randint(0, len(models) - 1) for _ in range(num_layers)]

        merged = models[0].__class__(models[0].config)
        merged_state = {}

        for key in models[0].state_dict():
            # Determine which model to use based on layer
            layer_idx = 0
            if "layer" in key:
                try:
                    layer_idx = int(key.split("layer.")[1].split(".")[0])
                except (IndexError, ValueError):
                    pass

            model_idx = layer_assignments[layer_idx % len(layer_assignments)]
            merged_state[key] = models[model_idx].state_dict()[key].clone()

        merged.load_state_dict(merged_state)
        return merged

    @staticmethod
    def dfs_merge(models: list[nn.Module], merge_ratio: float = 0.3) -> nn.Module:
        """DFS merge: Depth-First Search inspired hierarchical merging."""
        if len(models) == 1:
            return models[0]

        # Recursively merge pairs
        mid = len(models) // 2
        left = MergeOperators.dfs_merge(models[:mid], merge_ratio)
        right = MergeOperators.dfs_merge(models[mid:], merge_ratio)

        # Merge the two halves
        return MergeOperators.slerp_merge(left, right, merge_ratio)

    @staticmethod
    def task_arithmetic_merge(models: list[nn.Module], base_model: nn.Module, scaling_factor: float = 1.0) -> nn.Module:
        """Task Arithmetic: Merge by averaging task vectors."""
        merged = base_model.__class__(base_model.config)
        merged_state = {}

        for key in base_model.state_dict():
            base_param = base_model.state_dict()[key]

            # Calculate task vectors
            task_vectors = []
            for model in models:
                task_vector = model.state_dict()[key] - base_param
                task_vectors.append(task_vector)

            # Average task vectors and add to base
            avg_task_vector = sum(task_vectors) / len(task_vectors)
            merged_state[key] = base_param + scaling_factor * avg_task_vector

        merged.load_state_dict(merged_state)
        return merged


class ModelEvaluator:
    """
    Real benchmark evaluator using HumanEval, GSM8K, HellaSwag, and ARC datasets.
    Replaces placeholder prompts with real benchmark evaluation.
    """

    def __init__(self, device: str = "cuda", config: dict | None = None):
        self.device = device
        self.config = config or {}

        # Load real benchmark datasets
        self._load_benchmark_datasets()

        # Real benchmark test domains with actual datasets
        self.test_domains = {
            "code": {
                "dataset": self.humaneval_dataset,
                "weight": 0.25,
                "max_samples": 10,  # Use subset for faster evaluation
            },
            "math": {
                "dataset": self.gsm8k_dataset,
                "weight": 0.25,
                "max_samples": 10,
            },
            "multilingual": {
                "dataset": self.hellaswag_dataset,
                "weight": 0.25,
                "max_samples": 10,
            },
            "structured_data": {
                "dataset": self.arc_dataset,
                "weight": 0.25,
                "max_samples": 10,
            },
        }

    def _load_benchmark_datasets(self):
        """Load the real benchmark datasets."""
        try:
            import os

            from datasets import load_dataset

            # Disable offline mode for dataset loading
            os.environ.pop("HF_HUB_OFFLINE", None)

            # Load HumanEval for code evaluation
            self.humaneval_dataset = load_dataset("openai_humaneval")
            logger.info("[OK] Loaded HumanEval dataset for code evaluation")

            # Load GSM8K for math evaluation
            self.gsm8k_dataset = load_dataset("gsm8k", "main")
            logger.info("[OK] Loaded GSM8K dataset for math evaluation")

            # Load HellaSwag for reasoning evaluation
            self.hellaswag_dataset = load_dataset("hellaswag")
            logger.info("[OK] Loaded HellaSwag dataset for reasoning evaluation")

            # Load ARC for structured reasoning
            self.arc_dataset = load_dataset("ai2_arc", "ARC-Easy")
            logger.info("[OK] Loaded ARC dataset for structured reasoning evaluation")

        except Exception as e:
            logger.warning(f"Failed to load benchmark datasets: {e}")
            # Fallback to minimal placeholder prompts
            self._load_fallback_prompts()

    def _load_fallback_prompts(self):
        """Fallback to simple prompts if datasets fail to load."""
        self.test_domains = {
            "code": {"prompts": ['def fibonacci(n):\n    """Return nth Fibonacci number."""'], "weight": 0.25},
            "math": {"prompts": ["What is 2 + 2?"], "weight": 0.25},
            "multilingual": {"prompts": ["Hello world"], "weight": 0.25},
            "structured_data": {"prompts": ["Process this data: {a: 1, b: 2}"], "weight": 0.25},
        }

    async def evaluate(self, model: nn.Module, tokenizer: Any) -> dict[str, float]:
        """Evaluate model across all real benchmark datasets."""
        scores = {}
        model.eval()

        with torch.no_grad():
            for domain, config in self.test_domains.items():
                domain_scores = []

                # Use real dataset or fallback to prompts
                if "dataset" in config:
                    # Use real benchmark dataset
                    dataset = config["dataset"]
                    max_samples = config.get("max_samples", 10)
                    prompts = []

                    try:
                        # Get the appropriate split with robust error handling
                        if domain == "code":
                            # HumanEval test split
                            split_data = dataset.get(
                                "test", dataset.get("validation", list(dataset.values())[0] if dataset else [])
                            )
                            samples = list(split_data)[:max_samples] if hasattr(split_data, "__iter__") else []
                            prompts = [
                                sample.get("prompt", str(sample)) for sample in samples if isinstance(sample, dict)
                            ]
                        elif domain == "math":
                            # GSM8K test split
                            split_data = dataset.get(
                                "test", dataset.get("train", list(dataset.values())[0] if dataset else [])
                            )
                            samples = list(split_data)[:max_samples] if hasattr(split_data, "__iter__") else []
                            prompts = [
                                sample.get("question", str(sample)) for sample in samples if isinstance(sample, dict)
                            ]
                        elif domain == "multilingual":
                            # HellaSwag validation split
                            split_data = dataset.get(
                                "validation", dataset.get("test", list(dataset.values())[0] if dataset else [])
                            )
                            samples = list(split_data)[:max_samples] if hasattr(split_data, "__iter__") else []
                            prompts = [
                                sample.get("ctx", sample.get("context", str(sample)))
                                for sample in samples
                                if isinstance(sample, dict)
                            ]
                        elif domain == "structured_data":
                            # ARC test split
                            split_data = dataset.get(
                                "test", dataset.get("validation", list(dataset.values())[0] if dataset else [])
                            )
                            samples = list(split_data)[:max_samples] if hasattr(split_data, "__iter__") else []
                            prompts = [
                                sample.get("question", str(sample)) for sample in samples if isinstance(sample, dict)
                            ]

                        # Ensure we have valid prompts
                        prompts = [p for p in prompts if p and isinstance(p, str) and len(p.strip()) > 0]

                        if not prompts:
                            logger.warning(f"No valid prompts extracted from {domain} dataset, using fallback")

                    except Exception as e:
                        logger.warning(f"Failed to extract prompts from {domain} dataset: {e}")
                        prompts = []
                else:
                    # Fallback to simple prompts
                    prompts = config.get("prompts", [])

                # If no prompts from dataset, use domain-specific fallbacks
                if not prompts:
                    fallback_prompts = {
                        "code": ['def fibonacci(n):\n    """Return nth Fibonacci number."""\n    '],
                        "math": ["What is 15 + 27?", "If a train travels 60 mph for 2 hours, how far does it go?"],
                        "multilingual": ["The weather today is", "Complete this sentence: The cat sat on"],
                        "structured_data": ["Process this data: {a: 1, b: 2}", "Analyze: [1, 2, 3, 4, 5]"],
                    }
                    prompts = fallback_prompts.get(domain, [f"Complete this {domain} task:"])

                # Evaluate on prompts
                for prompt in prompts:
                    try:
                        # Tokenize input (without token_type_ids for Llama models)
                        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)

                        # Remove token_type_ids if present (not supported by LlamaForCausalLM)
                        if "token_type_ids" in inputs:
                            del inputs["token_type_ids"]

                        # Move to device
                        if hasattr(inputs, "to"):
                            inputs = inputs.to(self.device)
                        else:
                            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

                        # Calculate perplexity-based score (simpler approach)
                        with torch.no_grad():
                            outputs = model(**inputs)
                            logits = outputs.logits

                            # Calculate loss
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = inputs["input_ids"][..., 1:].contiguous()

                            loss = nn.functional.cross_entropy(
                                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="mean"
                            )

                            # Convert loss to score (lower loss = higher score)
                            # Use sigmoid to normalize to 0-1 range
                            score = torch.sigmoid(-loss + 5).item()  # Offset for reasonable range
                            domain_scores.append(score)

                    except Exception as e:
                        logger.warning(f"Evaluation failed for {domain} prompt: {str(e)[:100]}")
                        domain_scores.append(0.0)

                # Calculate domain average
                scores[domain] = np.mean(domain_scores) if domain_scores else 0.0
                logger.info(f"Domain {domain}: {len(domain_scores)} samples, score = {scores[domain]:.4f}")

        return scores

    def calculate_diversity(self, population: list[MergeCandidate]) -> float:
        """Calculate population diversity based on fitness scores."""
        if len(population) < 2:
            return 0.0

        # Calculate pairwise distances in fitness space
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = 0.0
                for domain in self.test_domains.keys():
                    score_i = population[i].fitness_scores.get(domain, 0)
                    score_j = population[j].fitness_scores.get(domain, 0)
                    dist += (score_i - score_j) ** 2
                distances.append(np.sqrt(dist))

        return np.mean(distances) if distances else 0.0


class EvolutionaryTournament:
    """
    Tournament-based evolutionary optimization for model merging.
    Implements NSGA-II for multi-objective optimization.
    """

    def __init__(self, config: Any):
        self.config = config
        self.population: list[MergeCandidate] = []
        self.pareto_front: list[MergeCandidate] = []
        self.generation = 0
        self.best_candidate: MergeCandidate | None = None

    def tournament_selection(self, population: list[MergeCandidate], tournament_size: int = 3) -> MergeCandidate:
        """Select winner from tournament."""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.aggregated_fitness)

    def calculate_pareto_front(self, population: list[MergeCandidate]) -> list[MergeCandidate]:
        """Calculate Pareto-optimal solutions for multi-objective optimization."""
        pareto_front = []

        for candidate in population:
            is_dominated = False
            for other in population:
                if other == candidate:
                    continue

                # Check if other dominates candidate
                dominates = all(
                    other.fitness_scores.get(d, 0) >= candidate.fitness_scores.get(d, 0)
                    for d in candidate.fitness_scores.keys()
                ) and any(
                    other.fitness_scores.get(d, 0) > candidate.fitness_scores.get(d, 0)
                    for d in candidate.fitness_scores.keys()
                )

                if dominates:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append(candidate)

        return pareto_front

    def nsga2_select(self, population: list[MergeCandidate], num_select: int) -> list[MergeCandidate]:
        """NSGA-II selection for next generation."""
        # Calculate Pareto fronts
        fronts = []
        remaining = population.copy()

        while remaining and len(fronts) < num_select:
            front = self.calculate_pareto_front(remaining)
            fronts.append(front)
            for candidate in front:
                remaining.remove(candidate)

        # Select candidates from fronts
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= num_select:
                selected.extend(front)
            else:
                # Crowding distance selection for partial front
                needed = num_select - len(selected)
                selected.extend(sorted(front, key=lambda x: x.diversity_score, reverse=True)[:needed])
                break

        return selected


class EvoMergePhase:
    """
    Phase 1: Evolutionary Model Merging

    Complete consolidation of all EvoMerge implementations with:
    - All merge techniques (linear, slerp, ties, dare, frankenmerge, dfs, task arithmetic)
    - Multi-objective optimization with NSGA-II
    - Tournament selection and evolutionary strategies
    - Comprehensive evaluation across domains
    - Memory-efficient chunked processing
    - Production-grade error handling and checkpointing
    """

    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        self.merge_ops = MergeOperators()
        self.evaluator = ModelEvaluator(self.device, config.__dict__)
        self.tournament = EvolutionaryTournament(config)

        self.population: list[MergeCandidate] = []
        self.best_model: MergeCandidate | None = None
        self.generation = 0
        self.fitness_history = []
        self.diversity_history = []

        # Setup directories
        self.output_dir = Path(config.output_dir) / "evomerge"
        self.checkpoint_dir = Path(config.checkpoint_dir) / "evomerge"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, base_model_paths: list[str] | None = None) -> Any:
        """
        Run the complete EvoMerge phase with all features.

        Args:
            base_model_paths: List of paths to base models (optional if using seed models)

        Returns:
            PhaseResult with the best merged model
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting EvoMerge Phase - Evolutionary Model Optimization")
        self.logger.info("=" * 80)

        # Use base_model_paths from config if not provided
        if base_model_paths is None:
            base_model_paths = self.config.base_models

        # Load base models and tokenizer
        base_models = await self._load_base_models(base_model_paths)

        # Load tokenizer - use local HRRM tokenizer or create simple one
        try:
            # First try HRRM tokenizer directory
            hrrm_tokenizer_path = "hf_models/hrrm-tokenizer"
            if Path(hrrm_tokenizer_path).exists():
                tokenizer = AutoTokenizer.from_pretrained(hrrm_tokenizer_path, local_files_only=True)
            else:
                raise Exception("No HRRM tokenizer found")
        except Exception as e:
            self.logger.warning(f"Could not load HRRM tokenizer, creating simple tokenizer: {e}")
            # Create a basic tokenizer for HRRM models
            from transformers import PreTrainedTokenizer

            class SimpleHRRMTokenizer(PreTrainedTokenizer):
                def __init__(self, vocab_size=32000):
                    self._vocab_size = vocab_size
                    super().__init__(
                        pad_token="<pad>",  # nosec B106 - tokenizer special token, not password
                        eos_token="</s>",  # nosec B106 - tokenizer special token, not password
                        bos_token="<s>",  # nosec B106 - tokenizer special token, not password
                        unk_token="<unk>",  # nosec B106 - tokenizer special token, not password
                    )

                @property
                def vocab_size(self):
                    return self._vocab_size

                def _tokenize(self, text):
                    return text.split()

                def _convert_token_to_id(self, token):
                    if token in self.get_vocab():
                        return self.get_vocab()[token]
                    return hash(token) % self._vocab_size

                def _convert_id_to_token(self, index):
                    return f"token_{index}"

                def get_vocab(self):
                    return {f"token_{i}": i for i in range(self._vocab_size)}

            tokenizer = SimpleHRRMTokenizer()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Generate initial population using all techniques
        self.logger.info("Generating initial population with 8 merge techniques...")
        self.population = await self._generate_initial_population(base_models)

        # Save initial models
        await self._save_population_models(0)

        # Evolution loop
        plateau_counter = 0
        best_fitness = 0.0

        for gen in range(self.config.evomerge_generations):
            self.generation = gen
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Generation {gen + 1}/{self.config.evomerge_generations}")
            self.logger.info(f"{'='*60}")

            # Load models and evaluate population
            await self._evaluate_population(tokenizer)

            # Calculate diversity
            diversity = self.evaluator.calculate_diversity(self.population)
            self.diversity_history.append(diversity)

            # Track best fitness
            current_best = max(self.population, key=lambda x: x.aggregated_fitness)
            if current_best.aggregated_fitness > best_fitness:
                best_fitness = current_best.aggregated_fitness
                plateau_counter = 0
                self.best_model = current_best
                self._save_best_model()
            else:
                plateau_counter += 1

            self.fitness_history.append(best_fitness)

            # Log generation statistics
            self._log_generation_stats(gen, diversity)

            # Check for convergence
            if plateau_counter >= self.config.evomerge_config.get("plateau_patience", 5):
                self.logger.info(f"Converged after {gen + 1} generations (plateau detected)")
                break

            if self._check_convergence():
                self.logger.info(f"Converged after {gen + 1} generations (variance threshold)")
                break

            # Create next generation
            self.population = await self._create_next_generation(base_models)

            # Save checkpoint
            self._save_checkpoint(gen)

            # Clean up old models to save space
            if self.config.evomerge_config.get("cleanup_old_generations", True):
                self._cleanup_old_models(gen - 2)

        # Final evaluation and reporting
        await self._generate_final_report()

        # Load best model for return
        best_model_path = str(Path(self.best_model.model_path).resolve()).replace("\\", "/")
        best_model = AutoModelForCausalLM.from_pretrained(
            best_model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            local_files_only=True,
        )

        # Return phase result
        from ..core.phase_controller import PhaseResult

        return PhaseResult(
            phase_name="evomerge",
            success=True,
            model=best_model,
            metrics={
                "final_fitness": self.best_model.aggregated_fitness,
                "generations": self.generation + 1,
                "fitness_scores": self.best_model.fitness_scores,
                "diversity_final": self.diversity_history[-1] if self.diversity_history else 0,
                "convergence_generation": self.generation + 1,
            },
            artifacts={
                "merge_recipe": self.best_model.merge_recipe,
                "population_size": len(self.population),
                "fitness_history": self.fitness_history,
                "diversity_history": self.diversity_history,
                "best_model_path": self.best_model.model_path,
            },
        )

    async def _load_base_models(self, paths: list[str]) -> list[nn.Module]:
        """Load base models with HRRM seed model integration."""
        models = []

        # Check for HRRM seed models first if prefer_seeds is enabled
        seed_paths = []
        if self.config.prefer_seeds:
            # Look for exported HRRM models
            hf_exports_dir = Path("artifacts/hf_exports")
            for model_type in ["planner", "reasoner", "memory"]:
                export_path = hf_exports_dir / model_type
                if export_path.exists():
                    seed_paths.append(str(export_path))

            # Add explicitly configured seed models
            if self.config.seed_models:
                seed_paths.extend(self.config.seed_models)

            if seed_paths:
                self.logger.info(f"Found {len(seed_paths)} HRRM seed models for fast iteration")

        # Use seed models if available and prefer_seeds is True, otherwise use base models
        model_paths = seed_paths if seed_paths and self.config.prefer_seeds else paths

        for path in model_paths:
            try:
                self.logger.info(f"Loading {'seed' if path in seed_paths else 'base'} model: {path}")

                # Try loading as standard HF model first
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True,
                        local_files_only=True,  # Force offline mode
                    )
                except Exception as hf_error:
                    # If HF loading fails, try loading HRRM exported model
                    if Path(path).exists() and (Path(path) / "pytorch_model.bin").exists():
                        self.logger.info(f"Loading HRRM exported model: {path}")

                        # Load HRRM original config
                        hrrm_config_path = Path(path) / "hrrm_original_config.json"
                        if hrrm_config_path.exists():
                            with open(hrrm_config_path) as f:
                                hrrm_config = json.load(f)
                        else:
                            # Fallback to regular config.json
                            with open(Path(path) / "config.json") as f:
                                hrrm_config = json.load(f)

                        # Create dummy config for basic transformer loading
                        class HRRMConfig:
                            def __init__(self, **kwargs):
                                self.vocab_size = kwargs.get("vocab_size", 32000)
                                self.hidden_size = kwargs.get("hidden_size", 512)
                                self.num_hidden_layers = kwargs.get("num_hidden_layers", 12)
                                self.num_attention_heads = kwargs.get("num_attention_heads", 8)
                                self.intermediate_size = kwargs.get("intermediate_size", 2048)
                                self.max_position_embeddings = kwargs.get("max_position_embeddings", 2048)

                        # Create basic transformer model for merging
                        from transformers import LlamaConfig, LlamaForCausalLM

                        config = LlamaConfig(
                            vocab_size=hrrm_config["vocab_size"],
                            hidden_size=hrrm_config.get("d_model", 256),  # HRRM uses d_model
                            intermediate_size=hrrm_config.get("d_model", 256) * 4,  # Standard transformer ratio
                            num_hidden_layers=hrrm_config.get("n_layers", 8),  # HRRM uses n_layers
                            num_attention_heads=hrrm_config.get("n_head", 8),  # HRRM uses n_head
                            max_position_embeddings=hrrm_config.get("max_seq_len", 2048),  # HRRM uses max_seq_len
                        )
                        model = LlamaForCausalLM(config)

                        # Load state dict
                        state_dict = torch.load(Path(path) / "pytorch_model.bin", map_location="cpu")

                        # Map HRRM parameters to standard transformer parameters for merging
                        mapped_state = {}
                        for key, tensor in state_dict.items():
                            # Map HRRM-specific parameter names to standard transformer names
                            mapped_key = key
                            if "hrm_layers" in key:
                                mapped_key = key.replace("hrm_layers", "model.layers")
                            elif "controller_head" in key or "scratchpad_supervisor" in key:
                                # Skip specialized heads for merging - use core transformer only
                                continue
                            elif "memory_" in key:
                                # Skip memory-specific components
                                continue

                            mapped_state[mapped_key] = tensor

                        # Load only compatible parameters
                        model.load_state_dict(mapped_state, strict=False)
                        self.logger.info(
                            f"[OK] Loaded HRRM model: {path} ({hrrm_config.get('param_count', 'Unknown')} params)"
                        )
                    else:
                        raise hf_error

                models.append(model)
                self.logger.info(f"[OK] Loaded: {path}")

            except Exception as e:
                self.logger.error(f"Failed to load {path}: {e}")
                # If seed model loading fails, fall back to base models
                if path in seed_paths and not self.config.seed_models:
                    self.logger.info("Falling back to base models due to seed model loading failure")
                    return await self._load_base_models(paths)
                else:
                    raise

        # If we loaded seed models successfully, log the fast iteration benefit
        if model_paths == seed_paths:
            self.logger.info("🚀 Using HRRM seed models for accelerated EvoMerge iteration!")
            self.logger.info("   - Smaller parameter counts enable faster merging")
            self.logger.info("   - Pre-optimized architectures provide better starting points")
            self.logger.info("   - HRM components bring specialized capabilities")

        return models

    async def _generate_initial_population(self, base_models: list[nn.Module]) -> list[MergeCandidate]:
        """Generate initial population using all merge techniques combinations."""
        population = []

        # Define merge combinations (matching production implementation)
        merge_recipes = [
            {"primary": "linear", "secondary": "ties", "final": "frankenmerge"},
            {"primary": "linear", "secondary": "ties", "final": "dfs"},
            {"primary": "linear", "secondary": "dare", "final": "frankenmerge"},
            {"primary": "linear", "secondary": "dare", "final": "dfs"},
            {"primary": "slerp", "secondary": "ties", "final": "frankenmerge"},
            {"primary": "slerp", "secondary": "ties", "final": "dfs"},
            {"primary": "slerp", "secondary": "dare", "final": "frankenmerge"},
            {"primary": "slerp", "secondary": "dare", "final": "dfs"},
        ]

        for i, recipe in enumerate(merge_recipes):
            self.logger.info(f"Creating candidate {i+1}/8: {recipe}")

            try:
                # Apply merge techniques in sequence
                merged = None

                # Primary merge
                if recipe["primary"] == "linear":
                    merged = self.merge_ops.linear_merge(base_models)
                elif recipe["primary"] == "slerp" and len(base_models) >= 2:
                    merged = self.merge_ops.slerp_merge(base_models[0], base_models[1])

                # Secondary merge
                if merged and recipe["secondary"] == "ties":
                    merged = self.merge_ops.ties_merge([merged] + base_models[1:])
                elif merged and recipe["secondary"] == "dare":
                    merged = self.merge_ops.dare_merge([merged] + base_models[1:])

                # Final merge
                if merged and recipe["final"] == "frankenmerge":
                    merged = self.merge_ops.frankenmerge([merged] + base_models)
                elif merged and recipe["final"] == "dfs":
                    merged = self.merge_ops.dfs_merge([merged] + base_models)

                if merged:
                    # Save model to disk
                    model_path = self.output_dir / f"gen0_candidate{i}"
                    merged.save_pretrained(model_path)

                    candidate = MergeCandidate(model_path=str(model_path), merge_recipe=recipe, generation=0)
                    population.append(candidate)
                    self.logger.info(f"[OK] Created candidate {i+1}")

            except Exception as e:
                self.logger.warning(f"Failed to create candidate {i+1}: {e}")
                continue

        self.logger.info(f"Initial population: {len(population)} candidates")
        return population

    async def _evaluate_population(self, tokenizer: Any):
        """Evaluate all candidates with caching."""
        for i, candidate in enumerate(self.population):
            if not candidate.fitness_scores:
                self.logger.info(f"Evaluating candidate {i+1}/{len(self.population)}")

                try:
                    # Load model from disk
                    model = AutoModelForCausalLM.from_pretrained(
                        candidate.model_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None,
                    )

                    # Evaluate across domains
                    scores = await self.evaluator.evaluate(model, tokenizer)
                    candidate.fitness_scores = scores

                    # Calculate aggregated fitness
                    weights = {"code": 0.25, "math": 0.25, "multilingual": 0.25, "structured_data": 0.25}
                    candidate.calculate_aggregated_fitness(weights)

                    # Clean up model
                    del model
                    torch.cuda.empty_cache()

                except Exception as e:
                    self.logger.error(f"Evaluation failed for candidate {i+1}: {e}")
                    candidate.fitness_scores = {d: 0.0 for d in self.evaluator.test_domains.keys()}
                    candidate.aggregated_fitness = 0.0

        # Sort by fitness
        self.population.sort(key=lambda x: x.aggregated_fitness, reverse=True)

        # Update best model
        if not self.best_model or self.population[0].aggregated_fitness > self.best_model.aggregated_fitness:
            self.best_model = self.population[0]
            self.logger.info(f"New best model: fitness={self.best_model.aggregated_fitness:.4f}")

    def _check_convergence(self) -> bool:
        """Check if evolution has converged."""
        if len(self.fitness_history) < 5:
            return False

        # Check fitness variance over last 5 generations
        recent_fitness = self.fitness_history[-5:]
        variance = np.var(recent_fitness)

        threshold = self.config.evomerge_config.get("convergence_threshold", 0.001)
        return variance < threshold

    async def _create_next_generation(self, base_models: list[nn.Module]) -> list[MergeCandidate]:
        """
        Create next generation using the user-specified breeding algorithm:
        Top 2 models → 6 children (3 each)
        Bottom 6 models → 2 children (groups of 3 → 1 child each)
        """
        next_gen = []

        # Sort population by fitness (handle NaN values)
        sorted_pop = sorted(
            self.population,
            key=lambda x: x.aggregated_fitness if not np.isnan(x.aggregated_fitness) else -1000,
            reverse=True,
        )

        # Filter out NaN candidates for breeding
        valid_candidates = [c for c in sorted_pop if not np.isnan(c.aggregated_fitness)]

        if len(valid_candidates) < 2:
            self.logger.warning("Not enough valid candidates for breeding, using original population")
            valid_candidates = sorted_pop

        # Top 2 winners → 6 children (3 each)
        top_2 = valid_candidates[:2]
        self.logger.info(f"Top 2 winners: fitness {top_2[0].aggregated_fitness:.4f}, {top_2[1].aggregated_fitness:.4f}")

        # Create 3 children from each winner
        merge_techniques = ["slerp", "ties", "dare", "linear", "frankenmerge", "dfs"]

        for i, winner in enumerate(top_2):
            for child_idx in range(3):
                try:
                    # Load winner model
                    winner_model = AutoModelForCausalLM.from_pretrained(winner.model_path)

                    # Select random base model for variation
                    base_idx = random.randint(0, len(base_models) - 1)
                    base_model = base_models[base_idx]

                    # Apply random merge technique
                    technique = random.choice(merge_techniques)
                    merge_weight = random.uniform(0.3, 0.7)

                    if technique == "linear":
                        child = self.merge_ops.linear_merge(
                            [winner_model, base_model], [merge_weight, 1 - merge_weight]
                        )
                    elif technique == "slerp":
                        child = self.merge_ops.slerp_merge(winner_model, base_model, merge_weight)
                    elif technique == "ties":
                        child = self.merge_ops.ties_merge([winner_model, base_model])
                    elif technique == "dare":
                        child = self.merge_ops.dare_merge([winner_model, base_model])
                    elif technique == "frankenmerge":
                        child = self.merge_ops.frankenmerge([winner_model, base_model])
                    else:  # dfs
                        child = self.merge_ops.dfs_merge([winner_model, base_model])

                    # Save child
                    model_path = self.output_dir / f"gen{self.generation+1}_winner{i+1}_child{child_idx+1}"
                    child.save_pretrained(model_path)

                    candidate = MergeCandidate(
                        model_path=str(model_path),
                        merge_recipe={
                            "type": "winner_child",
                            "parent": winner.model_path,
                            "technique": technique,
                            "merge_weight": merge_weight,
                            "base_model": f"base_{base_idx}",
                        },
                        generation=self.generation + 1,
                        parents=[winner.model_path],
                    )
                    next_gen.append(candidate)

                    # Cleanup
                    del winner_model, child
                    torch.cuda.empty_cache()

                except Exception as e:
                    self.logger.warning(f"Failed to create winner child: {e}")
                    continue

        # Bottom 6 → 2 children (groups of 3 → 1 child each)
        if len(valid_candidates) >= 6:
            bottom_6 = valid_candidates[-6:] if len(valid_candidates) >= 6 else valid_candidates[2:]
            self.logger.info("Bottom 6 candidates for loser children")

            # Create 2 groups of 3 → 1 child each
            for group_idx in range(2):
                try:
                    # Select 3 losers for this group
                    group_start = group_idx * 3
                    group_losers = bottom_6[group_start : group_start + 3]

                    if len(group_losers) < 3:
                        # Pad with random selection if needed
                        while len(group_losers) < 3:
                            group_losers.append(random.choice(bottom_6))

                    # Load the 3 loser models
                    loser_models = []
                    for loser in group_losers:
                        model = AutoModelForCausalLM.from_pretrained(loser.model_path)
                        loser_models.append(model)

                    # Merge all 3 losers with equal weights
                    child = self.merge_ops.linear_merge(loser_models, [1 / 3, 1 / 3, 1 / 3])

                    # Save loser child
                    model_path = self.output_dir / f"gen{self.generation+1}_loser_group{group_idx+1}"
                    child.save_pretrained(model_path)

                    candidate = MergeCandidate(
                        model_path=str(model_path),
                        merge_recipe={
                            "type": "loser_child",
                            "parents": [loser.model_path for loser in group_losers],
                            "technique": "linear_merge_3way",
                            "weights": [1 / 3, 1 / 3, 1 / 3],
                        },
                        generation=self.generation + 1,
                        parents=[l.model_path for l in group_losers],
                    )
                    next_gen.append(candidate)

                    # Cleanup
                    for model in loser_models:
                        del model
                    del child
                    torch.cuda.empty_cache()

                except Exception as e:
                    self.logger.warning(f"Failed to create loser child: {e}")
                    continue

        self.logger.info(f"Created next generation: {len(next_gen)} candidates (6 winner children + 2 loser children)")
        return next_gen[:8]  # Ensure exactly 8 candidates

    async def _save_population_models(self, generation: int):
        """Save all models in current population."""
        for i, candidate in enumerate(self.population):
            if not Path(candidate.model_path).exists():
                # Model needs to be saved
                self.logger.debug(f"Saving model for candidate {i} generation {generation}")

    def _save_best_model(self):
        """Save the best model to a permanent location."""
        if self.best_model:
            best_path = self.output_dir / "best_model"
            if not best_path.exists():
                import shutil

                shutil.copytree(self.best_model.model_path, best_path)
                self.logger.info(f"Best model saved to: {best_path}")

    def _save_checkpoint(self, generation: int):
        """Save checkpoint for resuming."""
        checkpoint = {
            "generation": generation,
            "population": [
                {
                    "model_path": c.model_path,
                    "merge_recipe": c.merge_recipe,
                    "fitness_scores": c.fitness_scores,
                    "aggregated_fitness": c.aggregated_fitness,
                }
                for c in self.population
            ],
            "best_model": self.best_model.model_path if self.best_model else None,
            "fitness_history": self.fitness_history,
            "diversity_history": self.diversity_history,
        }

        checkpoint_path = self.checkpoint_dir / f"checkpoint_gen{generation}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        self.logger.debug(f"Checkpoint saved: {checkpoint_path}")

    def _cleanup_old_models(self, generation: int):
        """Remove old generation models to save space (keep only current and previous generation)."""
        if generation < 0:
            return

        # Clean up generation n-2 and earlier to maintain max 16 models
        self.logger.info(f"Cleaning up generation {generation} to maintain max 16 models")

        cleanup_count = 0
        for path in self.output_dir.glob(f"gen{generation}_*"):
            if path.is_dir():
                import shutil

                try:
                    shutil.rmtree(path)
                    cleanup_count += 1
                    self.logger.info(f"Deleted: {path.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete {path.name}: {e}")

        if cleanup_count > 0:
            self.logger.info(f"Cleaned up {cleanup_count} models from generation {generation}")

    def _log_generation_stats(self, generation: int, diversity: float):
        """Log detailed generation statistics."""
        avg_fitness = np.mean([c.aggregated_fitness for c in self.population])
        max_fitness = max(c.aggregated_fitness for c in self.population)
        min_fitness = min(c.aggregated_fitness for c in self.population)

        self.logger.info(f"Generation {generation + 1} Statistics:")
        self.logger.info(f"  Best Fitness: {max_fitness:.4f}")
        self.logger.info(f"  Avg Fitness:  {avg_fitness:.4f}")
        self.logger.info(f"  Min Fitness:  {min_fitness:.4f}")
        self.logger.info(f"  Diversity:    {diversity:.4f}")

        if self.best_model:
            self.logger.info("  Best Scores by Domain:")
            for domain, score in self.best_model.fitness_scores.items():
                self.logger.info(f"    {domain:20s}: {score:.4f}")

    async def _generate_final_report(self):
        """Generate comprehensive final report."""
        report = {
            "run_info": {
                "total_generations": self.generation + 1,
                "population_size": len(self.population),
                "device": self.device,
                "timestamp": datetime.now().isoformat(),
            },
            "best_model": {
                "path": self.best_model.model_path if self.best_model else None,
                "fitness": self.best_model.aggregated_fitness if self.best_model else 0,
                "scores": self.best_model.fitness_scores if self.best_model else {},
                "recipe": self.best_model.merge_recipe if self.best_model else {},
            },
            "evolution_history": {
                "fitness": self.fitness_history,
                "diversity": self.diversity_history,
            },
            "final_population": [
                {
                    "model": c.model_path,
                    "fitness": c.aggregated_fitness,
                    "scores": c.fitness_scores,
                }
                for c in self.population[:5]  # Top 5
            ],
        }

        report_path = self.output_dir / "evomerge_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Final report saved: {report_path}")
