"""Mathematical Tutor Evolution System - Agent Forge Phase 1
Sprint R-4+AF1: Model Merging and Evolution - Task B.1.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import wandb

logger = logging.getLogger(__name__)


@dataclass
class ModelIndividual:
    """Individual model in the evolution population."""

    individual_id: str
    model_name: str
    model_path: str | None
    lineage: list[str]  # Parent models
    generation: int
    fitness_score: float
    performance_metrics: dict[str, float]
    model_size_mb: float
    parameters_count: int
    quantization_config: dict[str, Any]
    merge_strategy: str | None = None
    created_at: str = ""
    evaluated_at: str = ""


@dataclass
class EvolutionConfig:
    """Configuration for evolution process."""

    population_size: int = 6
    max_generations: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    selection_pressure: float = 0.8
    elitism_count: int = 1
    fitness_threshold: float = 0.85
    max_model_size_mb: float = 500.0
    target_subjects: list[str] = None
    target_grade_levels: list[int] = None


class MathTutorEvolution:
    """Evolve specialized math tutoring models through genetic algorithms."""

    def __init__(
        self,
        project_name: str = "agent-forge",
        evolution_config: EvolutionConfig = None,
    ) -> None:
        self.project_name = project_name
        self.config = evolution_config or EvolutionConfig()
        self.population = []
        self.generation_history = []
        self.fitness_history = {}

        # Model management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_models = {}  # Cache for loaded models
        self.tokenizers = {}  # Cache for tokenizers

        # Evolution tracking
        self.best_individual = None
        self.convergence_history = []

        # Initialize W&B tracking
        self.initialize_wandb_tracking()

        # Set default target subjects and grades
        if self.config.target_subjects is None:
            self.config.target_subjects = [
                "arithmetic",
                "algebra",
                "geometry",
                "statistics",
            ]
        if self.config.target_grade_levels is None:
            self.config.target_grade_levels = list(range(1, 9))  # Grades 1-8

    def initialize_wandb_tracking(self) -> None:
        """Initialize W&B tracking for evolution process."""
        try:
            wandb.init(
                project=self.project_name,
                job_type="math_tutor_evolution",
                config={
                    "evolution_version": "1.0.0",
                    "population_size": self.config.population_size,
                    "max_generations": self.config.max_generations,
                    "target_subjects": self.config.target_subjects,
                    "target_grades": self.config.target_grade_levels,
                    "fitness_threshold": self.config.fitness_threshold,
                    "max_model_size_mb": self.config.max_model_size_mb,
                },
            )

            logger.info("W&B evolution tracking initialized")

        except Exception as e:
            logger.exception(f"Failed to initialize W&B tracking: {e}")

    async def initialize_population(self) -> list[ModelIndividual]:
        """Initialize population with diverse base models optimized for math tutoring."""
        logger.info("Initializing evolution population with base models")

        # Curated base models for mathematical reasoning
        base_models = [
            {
                "name": "microsoft/phi-1_5",
                "description": "Small but capable reasoning model",
                "strengths": ["logical_reasoning", "step_by_step"],
                "size_estimate": 2700,  # MB
            },
            {
                "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "description": "Efficient conversational model",
                "strengths": ["conversation", "explanations"],
                "size_estimate": 2200,  # MB
            },
            {
                "name": "deepseek-ai/deepseek-math-1.3B",
                "description": "Specialized mathematical model",
                "strengths": ["mathematics", "problem_solving"],
                "size_estimate": 2600,  # MB
            },
            {
                "name": "microsoft/DialoGPT-small",
                "description": "Conversational and tutoring focused",
                "strengths": ["tutoring", "encouragement"],
                "size_estimate": 400,  # MB
            },
            {
                "name": "distilbert-base-uncased",
                "description": "Lightweight understanding model",
                "strengths": ["comprehension", "efficiency"],
                "size_estimate": 250,  # MB
            },
            {
                "name": "google/flan-t5-small",
                "description": "Instruction-following model",
                "strengths": ["instruction_following", "structured_responses"],
                "size_estimate": 300,  # MB
            },
        ]

        population_individuals = []

        for i, model_info in enumerate(base_models):
            if i >= self.config.population_size:
                break

            try:
                # Load and quantize model
                individual = await self.create_individual_from_base(
                    model_info, generation=0
                )

                if individual:
                    population_individuals.append(individual)

                    # Log individual creation
                    wandb.log(
                        {
                            f"population/individual_{i}/fitness": individual.fitness_score,
                            f"population/individual_{i}/size_mb": individual.model_size_mb,
                            f"population/individual_{i}/parameters": individual.parameters_count,
                            "generation": 0,
                            "population_initialized": True,
                        }
                    )

                    logger.info(
                        f"Created individual {i}: {individual.model_name} (fitness: {individual.fitness_score:.3f})"
                    )

            except Exception as e:
                logger.exception(
                    f"Failed to create individual from {model_info['name']}: {e}"
                )
                continue

        # Fill remaining spots with variations if needed
        while len(population_individuals) < self.config.population_size:
            # Create variations of existing individuals
            if population_individuals:
                base_individual = population_individuals[0]
                variation = await self.create_model_variation(base_individual)
                if variation:
                    population_individuals.append(variation)
            else:
                break

        self.population = population_individuals

        # Log population summary
        wandb.log(
            {
                "population/size": len(self.population),
                "population/avg_fitness": np.mean(
                    [ind.fitness_score for ind in self.population]
                ),
                "population/best_fitness": max(
                    [ind.fitness_score for ind in self.population]
                ),
                "population/total_size_mb": sum(
                    [ind.model_size_mb for ind in self.population]
                ),
                "generation": 0,
            }
        )

        logger.info(f"Population initialized with {len(self.population)} individuals")

        return self.population

    async def create_individual_from_base(
        self, model_info: dict[str, Any], generation: int
    ) -> ModelIndividual | None:
        """Create individual from base model with quantization and evaluation."""
        model_name = model_info["name"]

        try:
            # Create quantization config for efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            # Load model with quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Calculate model metrics
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_size_mb = param_count * 4 / (1024 * 1024)  # Approximate size in MB

            # Quick fitness evaluation
            fitness_score = await self.quick_fitness_evaluation(
                model, tokenizer, model_info
            )

            # Create individual
            individual_id = hashlib.md5(
                f"{model_name}_{generation}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]

            individual = ModelIndividual(
                individual_id=individual_id,
                model_name=model_name,
                model_path=None,  # Will be set when saved
                lineage=[model_name],
                generation=generation,
                fitness_score=fitness_score,
                performance_metrics={
                    "base_reasoning": fitness_score,
                    "model_size_penalty": max(
                        0, 1.0 - (model_size_mb / self.config.max_model_size_mb)
                    ),
                },
                model_size_mb=model_size_mb,
                parameters_count=param_count,
                quantization_config=asdict(quantization_config),
                created_at=datetime.now(timezone.utc).isoformat(),
            )

            # Cache model and tokenizer
            self.loaded_models[individual_id] = model
            self.tokenizers[individual_id] = tokenizer

            return individual

        except Exception as e:
            logger.exception(f"Failed to create individual from {model_name}: {e}")
            return None

    async def quick_fitness_evaluation(
        self, model, tokenizer, model_info: dict[str, Any]
    ) -> float:
        """Quick fitness evaluation for initial population."""
        try:
            # Simple math problems for quick evaluation
            test_problems = [
                "What is 12 + 8?",
                "If Sarah has 15 apples and gives away 7, how many does she have left?",
                "What is 6 × 4?",
                "Explain what a fraction is to a 3rd grader.",
            ]

            correct_responses = 0
            total_problems = len(test_problems)

            for problem in test_problems:
                # Create tutoring prompt
                prompt = (
                    f"You are a helpful math tutor. Explain step by step: {problem}"
                )

                # Tokenize
                inputs = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=512
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Generate response
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                # Decode response
                response = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )

                # Simple scoring heuristics
                if self.evaluate_math_response(problem, response):
                    correct_responses += 1

            # Base fitness score
            base_score = (
                correct_responses / total_problems if total_problems > 0 else 0.0
            )

            # Adjust based on model strengths
            strength_bonus = 0.0
            for strength in model_info.get("strengths", []):
                if strength in ["mathematics", "logical_reasoning", "problem_solving"]:
                    strength_bonus += 0.1
                elif strength in ["tutoring", "explanations", "step_by_step"]:
                    strength_bonus += 0.05

            final_score = min(1.0, base_score + strength_bonus)

            return final_score

        except Exception as e:
            logger.exception(f"Error in quick fitness evaluation: {e}")
            return 0.3  # Default low score

    def evaluate_math_response(self, problem: str, response: str) -> bool:
        """Simple heuristic evaluation of math response quality."""
        response_lower = response.lower()

        # Check for basic math indicators
        if any(
            indicator in problem.lower() for indicator in ["what is", "calculate", "+"]
        ):
            # Look for numerical answers
            if any(char.isdigit() for char in response):
                return True

        # Check for explanation indicators
        if "explain" in problem.lower():
            explanation_words = ["means", "is when", "example", "like", "think of"]
            if any(word in response_lower for word in explanation_words):
                return True

        # Check for step-by-step approach
        step_indicators = ["first", "then", "next", "step", "so"]
        if any(indicator in response_lower for indicator in step_indicators):
            return True

        # Check response length (not too short, not too long)
        return 10 <= len(response.split()) <= 100

    async def create_model_variation(
        self, base_individual: ModelIndividual
    ) -> ModelIndividual | None:
        """Create a variation of an existing individual through parameter perturbation."""
        try:
            # Create variation ID
            variation_id = hashlib.md5(
                f"{base_individual.individual_id}_variation_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]

            # Clone base model (simplified - in practice would modify parameters)
            if base_individual.individual_id in self.loaded_models:
                base_model = self.loaded_models[base_individual.individual_id]
                base_tokenizer = self.tokenizers[base_individual.individual_id]

                # For now, create variation with slightly different config
                variation = ModelIndividual(
                    individual_id=variation_id,
                    model_name=f"{base_individual.model_name}_var",
                    model_path=None,
                    lineage=[*base_individual.lineage, "variation"],
                    generation=base_individual.generation,
                    fitness_score=base_individual.fitness_score
                    * (0.9 + np.random.random() * 0.2),  # Add noise
                    performance_metrics=base_individual.performance_metrics.copy(),
                    model_size_mb=base_individual.model_size_mb,
                    parameters_count=base_individual.parameters_count,
                    quantization_config=base_individual.quantization_config,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )

                # Cache references (sharing same model for now)
                self.loaded_models[variation_id] = base_model
                self.tokenizers[variation_id] = base_tokenizer

                return variation

        except Exception as e:
            logger.exception(f"Failed to create model variation: {e}")

        return None

    async def evolve_generation(self, generation: int) -> list[ModelIndividual]:
        """Evolve population for one generation using genetic algorithms."""
        logger.info(f"Evolving generation {generation}")

        # Evaluate current population fitness
        await self.evaluate_population_fitness()

        # Aggregate KPI scores for this generation
        generation_kpis: dict[str, float] = {}
        if self.population:
            metric_keys = set().union(
                *(ind.performance_metrics.keys() for ind in self.population)
            )
            for key in metric_keys:
                generation_kpis[key] = float(
                    np.mean(
                        [
                            ind.performance_metrics.get(key, 0.0)
                            for ind in self.population
                        ]
                    )
                )
        self.generation_history.append(
            {"generation": generation, "kpi_scores": generation_kpis}
        )
        if generation_kpis:
            wandb.log(
                {
                    f"generation/{generation}/kpi/{k}": v
                    for k, v in generation_kpis.items()
                }
                | {"generation": generation}
            )

        # Selection
        parents = self.select_parents()

        # Create offspring through crossover and mutation
        offspring = []

        # Elitism - keep best individuals
        sorted_population = sorted(
            self.population, key=lambda x: x.fitness_score, reverse=True
        )
        for i in range(self.config.elitism_count):
            if i < len(sorted_population):
                elite = sorted_population[i]
                elite.generation = generation
                offspring.append(elite)

        # Generate new individuals through crossover
        while len(offspring) < self.config.population_size:
            if len(parents) >= 2 and np.random.random() < self.config.crossover_rate:
                # Crossover
                parent1, parent2 = np.random.choice(parents, 2, replace=False)
                child = await self.crossover(parent1, parent2, generation)
                if child:
                    offspring.append(child)
            # Mutation
            elif parents:
                parent = np.random.choice(parents)
                mutated = await self.mutate(parent, generation)
                if mutated:
                    offspring.append(mutated)

        # Trim to population size
        offspring = offspring[: self.config.population_size]

        # Update population
        self.population = offspring

        # Update best individual
        current_best = max(self.population, key=lambda x: x.fitness_score)
        if (
            self.best_individual is None
            or current_best.fitness_score > self.best_individual.fitness_score
        ):
            self.best_individual = current_best

        # Log generation results
        avg_fitness = np.mean([ind.fitness_score for ind in self.population])
        best_fitness = max([ind.fitness_score for ind in self.population])

        wandb.log(
            {
                f"generation/{generation}/avg_fitness": avg_fitness,
                f"generation/{generation}/best_fitness": best_fitness,
                f"generation/{generation}/population_size": len(self.population),
                f"generation/{generation}/fitness_improvement": best_fitness
                - (self.convergence_history[-1] if self.convergence_history else 0),
                "generation": generation,
            }
        )

        self.convergence_history.append(best_fitness)

        logger.info(
            f"Generation {generation} complete - Avg fitness: {avg_fitness:.3f}, Best: {best_fitness:.3f}"
        )

        return self.population

    async def evaluate_population_fitness(self) -> None:
        """Evaluate fitness for all individuals in population."""
        logger.info("Evaluating population fitness")

        # Import fitness evaluator
        from .math_fitness import MathFitnessEvaluator

        fitness_evaluator = MathFitnessEvaluator()

        for individual in self.population:
            try:
                if individual.individual_id in self.loaded_models:
                    model = self.loaded_models[individual.individual_id]
                    tokenizer = self.tokenizers[individual.individual_id]

                    # Comprehensive fitness evaluation
                    fitness_score = await fitness_evaluator.evaluate(
                        model=model,
                        tokenizer=tokenizer,
                        individual_id=individual.individual_id,
                        log_details=True,
                    )

                    # Update individual fitness
                    individual.fitness_score = fitness_score
                    individual.evaluated_at = datetime.now(timezone.utc).isoformat()
                    individual.performance_metrics.update(fitness_evaluator.kpi_scores)

                    # Store in fitness history
                    self.fitness_history[individual.individual_id] = fitness_score

            except Exception as e:
                logger.exception(
                    f"Error evaluating fitness for {individual.individual_id}: {e}"
                )
                individual.fitness_score = 0.1  # Low fitness for failed evaluation

    def select_parents(self) -> list[ModelIndividual]:
        """Select parents for reproduction using tournament selection."""
        # Sort population by fitness
        sorted_population = sorted(
            self.population, key=lambda x: x.fitness_score, reverse=True
        )

        # Select top performers with some randomness
        selection_size = int(len(sorted_population) * self.config.selection_pressure)
        selected_parents = sorted_population[:selection_size]

        # Add some random selection for diversity
        remaining = sorted_population[selection_size:]
        if remaining:
            random_additions = min(2, len(remaining))
            selected_parents.extend(
                np.random.choice(remaining, random_additions, replace=False)
            )

        return selected_parents

    async def crossover(
        self, parent1: ModelIndividual, parent2: ModelIndividual, generation: int
    ) -> ModelIndividual | None:
        """Create offspring through model crossover/merging."""
        try:
            # Import merge operator
            from .merge_operators import MergeOperator

            merge_operator = MergeOperator()

            # Perform model merging
            if (
                parent1.individual_id in self.loaded_models
                and parent2.individual_id in self.loaded_models
            ):
                parent1_model = self.loaded_models[parent1.individual_id]
                parent2_model = self.loaded_models[parent2.individual_id]

                # Choose merge strategy randomly
                merge_strategies = ["linear", "slerp", "dare"]
                strategy = np.random.choice(merge_strategies)

                # Perform merge
                merged_model = await merge_operator.merge_models(
                    parent1_model, parent2_model, strategy, generation
                )

                if merged_model:
                    # Create offspring individual
                    offspring_id = hashlib.md5(
                        f"{parent1.individual_id}_{parent2.individual_id}_{generation}".encode()
                    ).hexdigest()[:12]

                    offspring = ModelIndividual(
                        individual_id=offspring_id,
                        model_name=f"merged_{parent1.model_name.split('/')[-1]}_{parent2.model_name.split('/')[-1]}",
                        model_path=None,
                        lineage=parent1.lineage + parent2.lineage,
                        generation=generation,
                        fitness_score=0.0,  # Will be evaluated
                        performance_metrics={},
                        model_size_mb=(parent1.model_size_mb + parent2.model_size_mb)
                        / 2,
                        parameters_count=(
                            parent1.parameters_count + parent2.parameters_count
                        )
                        // 2,
                        quantization_config=parent1.quantization_config,
                        merge_strategy=strategy,
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )

                    # Cache merged model
                    self.loaded_models[offspring_id] = merged_model
                    # Use parent1's tokenizer for now
                    self.tokenizers[offspring_id] = self.tokenizers[
                        parent1.individual_id
                    ]

                    return offspring

        except Exception as e:
            logger.exception(f"Error in crossover: {e}")

        return None

    async def mutate(
        self, parent: ModelIndividual, generation: int
    ) -> ModelIndividual | None:
        """Create offspring through mutation."""
        if np.random.random() > self.config.mutation_rate:
            return None

        try:
            # Simple mutation - create variation with parameter noise
            mutated = await self.create_model_variation(parent)

            if mutated:
                mutated.generation = generation
                mutated.lineage = [*parent.lineage, "mutation"]

                # Add some random fitness variation (simulating mutation effect)
                mutation_strength = 0.1
                fitness_delta = np.random.normal(0, mutation_strength)
                mutated.fitness_score = max(
                    0.0, min(1.0, parent.fitness_score + fitness_delta)
                )

                return mutated

        except Exception as e:
            logger.exception(f"Error in mutation: {e}")

        return None

    async def run_evolution(self) -> ModelIndividual:
        """Run complete evolution process."""
        logger.info(
            f"Starting evolution with {self.config.max_generations} generations"
        )

        # Initialize population
        await self.initialize_population()

        if not self.population:
            msg = "Failed to initialize population"
            raise ValueError(msg)

        # Evolution loop
        for generation in range(1, self.config.max_generations + 1):
            try:
                # Evolve generation
                self.population = await self.evolve_generation(generation)

                # Check convergence
                if (
                    self.best_individual
                    and self.best_individual.fitness_score
                    >= self.config.fitness_threshold
                ):
                    logger.info(
                        f"Evolution converged at generation {generation} with fitness {self.best_individual.fitness_score:.3f}"
                    )
                    break

                # Memory cleanup
                await self.cleanup_old_models(generation)

            except Exception as e:
                logger.exception(f"Error in generation {generation}: {e}")
                continue

        # Log final results
        if self.best_individual:
            wandb.log(
                {
                    "evolution_complete": True,
                    "final_best_fitness": self.best_individual.fitness_score,
                    "final_generation": self.best_individual.generation,
                    "convergence_achieved": self.best_individual.fitness_score
                    >= self.config.fitness_threshold,
                    "total_generations": len(self.convergence_history),
                }
            )

            # Save best model
            await self.save_champion_model(self.best_individual)

        logger.info(
            f"Evolution complete. Best fitness: {self.best_individual.fitness_score:.3f}"
            if self.best_individual
            else "Evolution failed"
        )

        return self.best_individual

    async def cleanup_old_models(self, current_generation: int) -> None:
        """Clean up old models to manage memory."""
        # Keep only recent generations and best individuals
        generations_to_keep = 2
        min_generation = max(0, current_generation - generations_to_keep)

        individuals_to_remove = []

        for individual_id, individual in [
            (ind.individual_id, ind) for ind in self.population
        ]:
            if (
                individual.generation < min_generation
                and individual != self.best_individual
                and individual.fitness_score
                < np.percentile([ind.fitness_score for ind in self.population], 75)
            ):
                individuals_to_remove.append(individual_id)

        # Remove from cache
        for individual_id in individuals_to_remove:
            if individual_id in self.loaded_models:
                del self.loaded_models[individual_id]
            if individual_id in self.tokenizers:
                del self.tokenizers[individual_id]

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if individuals_to_remove:
            logger.info(
                f"Cleaned up {len(individuals_to_remove)} old models from memory"
            )

    async def save_champion_model(self, champion: ModelIndividual) -> None:
        """Save the champion model for deployment."""
        try:
            # Create save directory
            save_dir = Path("models/evolved_math_tutors")
            save_dir.mkdir(parents=True, exist_ok=True)

            champion_path = save_dir / f"champion_{champion.individual_id}"

            # Save model and metadata
            if champion.individual_id in self.loaded_models:
                model = self.loaded_models[champion.individual_id]
                tokenizer = self.tokenizers[champion.individual_id]

                # Save model
                model.save_pretrained(champion_path)
                tokenizer.save_pretrained(champion_path)

                # Save metadata
                metadata = {
                    "individual_data": asdict(champion),
                    "evolution_config": asdict(self.config),
                    "convergence_history": self.convergence_history,
                    "fitness_history": self.fitness_history.get(
                        champion.individual_id, []
                    ),
                }

                with open(champion_path / "evolution_metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                # Create W&B artifact
                artifact = wandb.Artifact(
                    f"math_tutor_champion_{champion.individual_id}",
                    type="model",
                    description=f"Champion math tutor model (fitness: {champion.fitness_score:.3f})",
                    metadata={
                        "fitness_score": champion.fitness_score,
                        "generation": champion.generation,
                        "model_size_mb": champion.model_size_mb,
                        "lineage": champion.lineage,
                        "merge_strategy": champion.merge_strategy,
                    },
                )

                artifact.add_dir(str(champion_path))
                wandb.log_artifact(artifact)

                champion.model_path = str(champion_path)

                logger.info(f"Champion model saved to {champion_path}")

        except Exception as e:
            logger.exception(f"Error saving champion model: {e}")

    def get_evolution_summary(self) -> dict[str, Any]:
        """Get comprehensive evolution summary."""
        summary = {
            "evolution_config": asdict(self.config),
            "population_size": len(self.population),
            "generations_completed": len(self.convergence_history),
            "best_individual": (
                asdict(self.best_individual) if self.best_individual else None
            ),
            "convergence_history": self.convergence_history,
            "fitness_statistics": {
                "current_avg": (
                    np.mean([ind.fitness_score for ind in self.population])
                    if self.population
                    else 0
                ),
                "current_max": (
                    max([ind.fitness_score for ind in self.population])
                    if self.population
                    else 0
                ),
                "current_min": (
                    min([ind.fitness_score for ind in self.population])
                    if self.population
                    else 0
                ),
                "improvement": (
                    (self.convergence_history[-1] - self.convergence_history[0])
                    if len(self.convergence_history) > 1
                    else 0
                ),
            },
            "population_diversity": {
                "unique_lineages": len({tuple(ind.lineage) for ind in self.population}),
                "merge_strategies_used": list(
                    {
                        ind.merge_strategy
                        for ind in self.population
                        if ind.merge_strategy
                    }
                ),
                "generation_distribution": {
                    gen: len([ind for ind in self.population if ind.generation == gen])
                    for gen in {ind.generation for ind in self.population}
                },
            },
            "model_statistics": {
                "avg_model_size_mb": (
                    np.mean([ind.model_size_mb for ind in self.population])
                    if self.population
                    else 0
                ),
                "total_parameters": sum(
                    [ind.parameters_count for ind in self.population]
                ),
                "models_cached": len(self.loaded_models),
            },
        }

        return summary


# Global evolution system instance
math_tutor_evolution = MathTutorEvolution()
