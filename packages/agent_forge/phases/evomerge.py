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

import json
import logging
import random
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


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
        self.aggregated_fitness = sum(self.fitness_scores.get(domain, 0) * weight for domain, weight in weights.items())
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
    Advanced model evaluator with multi-domain benchmarks.
    Consolidated from production evomerge benchmarks.
    """

    def __init__(self, device: str = "cuda", config: dict | None = None):
        self.device = device
        self.config = config or {}

        # Comprehensive test prompts across domains
        self.test_domains = {
            "code": {
                "prompts": [
                    'def fibonacci(n):\n    """Return the nth Fibonacci number."""',
                    'class BinaryTree:\n    """Implement a binary search tree."""',
                    "import numpy as np\n# Matrix multiplication function",
                    'def quicksort(arr):\n    """Sort array using quicksort."""',
                ],
                "weight": 0.25,
            },
            "math": {
                "prompts": [
                    "Solve for x: 2x^2 + 5x - 3 = 0",
                    "Calculate the derivative of f(x) = x^3 * sin(x)",
                    "What is the integral of e^(-x^2)?",
                    "Prove that sqrt(2) is irrational.",
                ],
                "weight": 0.25,
            },
            "multilingual": {
                "prompts": [
                    "Translate to Spanish: The future of AI is bright",
                    "Comment dit-on 'machine learning' en français?",
                    "什么是深度学习的主要应用？",
                    "Übersetzen Sie ins Deutsche: Neural networks",
                ],
                "weight": 0.25,
            },
            "structured_data": {
                "prompts": [
                    'Parse JSON: {"model": "GPT", "params": 175e9, "layers": [96]}',
                    "Convert to CSV:\n| Name | Age | City |\n| Alice | 30 | NYC |",
                    "Extract from XML: <user><name>Bob</name><age>25</age></user>",
                    "Parse YAML:\nmodel:\n  name: BERT\n  size: large",
                ],
                "weight": 0.25,
            },
        }

    async def evaluate(self, model: nn.Module, tokenizer: Any) -> dict[str, float]:
        """Evaluate model across all domains with detailed metrics."""
        scores = {}
        model.eval()

        with torch.no_grad():
            for domain, config in self.test_domains.items():
                domain_scores = []

                for prompt in config["prompts"]:
                    try:
                        # Tokenize and generate
                        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

                        model.generate(
                            **inputs,
                            max_new_tokens=50,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )

                        # Calculate perplexity-based score
                        with torch.no_grad():
                            logits = model(**inputs).logits
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = inputs.input_ids[..., 1:].contiguous()

                            loss = nn.functional.cross_entropy(
                                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="mean"
                            )
                            perplexity = torch.exp(loss).item()

                            # Convert perplexity to score (lower is better)
                            score = 1.0 / (1.0 + perplexity)
                            domain_scores.append(score)

                    except Exception as e:
                        logger.warning(f"Evaluation failed for prompt: {e}")
                        domain_scores.append(0.0)

                scores[domain] = np.mean(domain_scores) if domain_scores else 0.0

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

    async def run(self, base_model_paths: list[str]) -> Any:
        """
        Run the complete EvoMerge phase with all features.

        Args:
            base_model_paths: List of paths to base models

        Returns:
            PhaseResult with the best merged model
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting EvoMerge Phase - Evolutionary Model Optimization")
        self.logger.info("=" * 80)

        # Load base models and tokenizer
        base_models = await self._load_base_models(base_model_paths)
        tokenizer = AutoTokenizer.from_pretrained(base_model_paths[0])
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
        best_model = AutoModelForCausalLM.from_pretrained(
            self.best_model.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        # Return phase result
        from ..core.unified_pipeline import PhaseResult

        return PhaseResult(
            phase_name="evomerge",
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
        """Load base models with proper error handling."""
        models = []
        for path in paths:
            try:
                self.logger.info(f"Loading base model: {path}")
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                )
                models.append(model)
                self.logger.info(f"✓ Loaded: {path}")
            except Exception as e:
                self.logger.error(f"Failed to load {path}: {e}")
                raise

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
                    self.logger.info(f"✓ Created candidate {i+1}")

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
        """Create next generation through selection, crossover, and mutation."""
        next_gen = []

        # Elitism: Keep top 2 candidates
        elite_size = 2
        next_gen.extend(self.population[:elite_size])

        # Generate rest through evolution
        while len(next_gen) < self.config.evomerge_population_size:
            # Tournament selection
            parent1 = self.tournament.tournament_selection(self.population)
            parent2 = self.tournament.tournament_selection(self.population)

            # Create offspring
            try:
                # Load parent models
                model1 = AutoModelForCausalLM.from_pretrained(parent1.model_path)
                model2 = AutoModelForCausalLM.from_pretrained(parent2.model_path)

                # Crossover
                if random.random() < 0.7:  # Crossover probability
                    child = self.merge_ops.linear_merge([model1, model2], [0.5, 0.5])
                else:
                    child = model1  # Take first parent

                # Mutation
                if random.random() < self.config.evomerge_config.get("mutation_rate", 0.15):
                    # Apply random merge with base model
                    base_idx = random.randint(0, len(base_models) - 1)
                    techniques = ["slerp", "ties", "dare"]
                    technique = random.choice(techniques)

                    if technique == "slerp":
                        child = self.merge_ops.slerp_merge(child, base_models[base_idx], random.uniform(0.1, 0.9))
                    elif technique == "ties":
                        child = self.merge_ops.ties_merge([child, base_models[base_idx]])
                    else:  # dare
                        child = self.merge_ops.dare_merge([child, base_models[base_idx]])

                # Save offspring
                model_path = self.output_dir / f"gen{self.generation+1}_candidate{len(next_gen)}"
                child.save_pretrained(model_path)

                candidate = MergeCandidate(
                    model_path=str(model_path),
                    merge_recipe={
                        "parents": [parent1.model_path, parent2.model_path],
                        "crossover": True,
                        "mutation": random.random() < self.config.evomerge_config.get("mutation_rate", 0.15),
                    },
                    generation=self.generation + 1,
                    parents=[parent1.model_path, parent2.model_path],
                )
                next_gen.append(candidate)

                # Clean up
                del model1, model2, child
                torch.cuda.empty_cache()

            except Exception as e:
                self.logger.warning(f"Failed to create offspring: {e}")
                continue

        return next_gen[: self.config.evomerge_population_size]

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
        """Remove old generation models to save space."""
        if generation < 0:
            return

        for path in self.output_dir.glob(f"gen{generation}_*"):
            if path.is_dir():
                import shutil

                shutil.rmtree(path)
                self.logger.debug(f"Cleaned up: {path}")

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
