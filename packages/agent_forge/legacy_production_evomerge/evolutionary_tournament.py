import argparse
import json
import logging
import os
import random
from datetime import datetime

import torch
from agent_forge.evaluation.evaluator import evaluate_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from common.logging import setup_logging

from .config import Configuration, ModelReference
from .merging.merger import AdvancedModelMerger
from .model_tracker import model_tracker
from .multi_objective import calculate_pareto_front, nsga2_select
from .utils import EvoMergeException, clean_up_models
from .visualization import plot_fitness_over_generations, plot_pareto_front

logger = logging.getLogger(__name__)


class EvolutionaryTournament:
    def __init__(self, config: Configuration) -> None:
        self.config = config
        self.merger = AdvancedModelMerger(config)
        self.fitness_scores = []
        self.diversity_scores = []
        self.checkpoint_dir = os.path.join(self.config.merge_settings.custom_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.population_size = self.config.evolution_settings.population_size
        self.objectives = self.config.evolution_settings.objectives

        log_file = os.path.join(self.config.merge_settings.custom_dir, "evolution_log.txt")
        setup_logging(log_file=log_file, log_level=logging.DEBUG)

    def log_generation_info(self, generation: int, population: list[str], scores: list[dict[str, float]]) -> None:
        logger.info(f"Generation {generation} completed")
        logger.info(f"Population size: {len(population)}")
        logger.info(f"Best scores: {max(scores, key=lambda x: x[self.objectives[0]])}")
        logger.info("Average scores: {obj: np.mean([s[obj] for s in scores]) for obj in self.objectives}")
        logger.info(f"Diversity: {self.diversity_scores[-1]}")

    def create_initial_population(self) -> list[str]:
        logger.info("Creating initial population")
        merge_combinations = [
            ["linear", "ties", "frankenmerge"],
            ["linear", "ties", "dfs"],
            ["linear", "dare", "frankenmerge"],
            ["linear", "dare", "dfs"],
            ["slerp", "ties", "frankenmerge"],
            ["slerp", "ties", "dfs"],
            ["slerp", "dare", "frankenmerge"],
            ["slerp", "dare", "dfs"],
        ]

        population = []
        for techniques in merge_combinations:
            self.config.merge_settings.ps_techniques = techniques[:2]
            self.config.merge_settings.dfs_techniques = [techniques[2]]
            try:
                merged_model_path = self.merger.merge()
                population.append(merged_model_path)
                logger.info(f"Created merged model: {merged_model_path}")
            except Exception as e:
                logger.exception(f"Failed to create merged model with techniques {techniques}: {e!s}")

        return population

    def mutate_model(self, model_path: str, mutation_rate: float) -> str:
        logger.info(f"Mutating model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path)

        with torch.no_grad():
            for param in model.parameters():
                mutation = torch.randn_like(param) * mutation_rate
                param.add_(mutation)

        new_path = f"{model_path}_mutated_{random.randint(1000, 9999)}"
        model.save_pretrained(new_path)
        return new_path

    def merge_models(self, models: list[str]) -> str:
        logger.info(f"Merging models: {models}")
        merged_config = Configuration(
            models=[ModelReference(name=f"model_{i}", path=model) for i, model in enumerate(models)],
            merge_settings=self.config.merge_settings,
            evolution_settings=self.config.evolution_settings,
        )
        merger = AdvancedModelMerger(merged_config)
        return merger.merge()

    def calculate_diversity(self, population: list[str]) -> float:
        models = [AutoModelForCausalLM.from_pretrained(path) for path in population]
        param_vectors = [self.flatten_params(model) for model in models]

        diversity = 0
        for i in range(len(param_vectors)):
            for j in range(i + 1, len(param_vectors)):
                diversity += torch.dist(param_vectors[i], param_vectors[j]).item()

        return diversity / (len(population) * (len(population) - 1) / 2)

    def flatten_params(self, model: torch.nn.Module) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in model.parameters()])

    def create_generation_children(
        self,
        parents: list[str],
        generation: int,
        phase: str,
        target_count: int = 8,
        mutation_rate: float = 0.0,
    ) -> list[str]:
        """Create children for a generation phase using various merge techniques."""
        import os
        from pathlib import Path

        # Setup generation directory structure
        aiv_root = os.environ.get("AIV_ROOT", "D:\\AIVillage")
        gen_dir = Path(aiv_root) / "artifacts" / "gens" / f"G{generation + 1:04d}" / f"phase_{phase}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        children = []
        merge_techniques = [
            ["linear", "ties"],
            ["linear", "dare"],
            ["slerp", "ties"],
            ["slerp", "dare"],
            ["linear", "task_arithmetic"],
            ["slerp", "task_arithmetic"],
            ["linear", "dfs"],
            ["slerp", "dfs"],
        ]

        for i in range(target_count):
            child_dir = gen_dir / f"child_{i + 1:02d}"

            try:
                # Cycle through merge techniques
                techniques = merge_techniques[i % len(merge_techniques)]

                # Configure merger for this child
                child_config = Configuration(
                    models=[ModelReference(name=f"parent_{j}", path=parent) for j, parent in enumerate(parents)],
                    merge_settings=self.config.merge_settings,
                    evolution_settings=self.config.evolution_settings,
                )
                child_config.merge_settings.ps_techniques = techniques[:1]
                child_config.merge_settings.dfs_techniques = techniques[1:] if len(techniques) > 1 else []

                merger = AdvancedModelMerger(child_config)
                merged_path = merger.merge()

                # Apply mutation if specified
                if mutation_rate > 0.0:
                    mutated_path = self.mutate_model(merged_path, mutation_rate)
                    # Move to final child directory
                    import shutil

                    if child_dir.exists():
                        shutil.rmtree(child_dir)
                    shutil.move(mutated_path, child_dir)
                    children.append(str(child_dir))
                else:
                    # Move to final child directory
                    import shutil

                    if child_dir.exists():
                        shutil.rmtree(child_dir)
                    shutil.move(merged_path, child_dir)
                    children.append(str(child_dir))

                logger.info(f"Created child {i + 1}/{target_count}: {child_dir}")

            except Exception as e:
                logger.exception(f"Failed to create child {i + 1} with techniques {techniques}: {e}")
                # Create a copy of first parent as fallback
                if parents:
                    import shutil

                    if child_dir.exists():
                        shutil.rmtree(child_dir)
                    shutil.copytree(parents[0], child_dir)
                    children.append(str(child_dir))

        return children

    def should_early_stop(self, fitness_history: list[float], delta_pct: float, window_size: int) -> bool:
        """Check if early stopping criteria are met."""
        if len(fitness_history) < window_size + 1:
            return False

        # Get current best and best from window_size generations ago
        current_best = fitness_history[-1]
        window_best = max(fitness_history[-window_size - 1 : -1])

        # Calculate improvement percentage
        if window_best == 0:
            return False

        improvement_pct = ((current_best - window_best) / window_best) * 100

        logger.info(f"Improvement over {window_size} generations: {improvement_pct:.3f}%")

        return improvement_pct < delta_pct

    def save_checkpoint(
        self,
        generation: int,
        population: list[str],
        scores: list[float],
        mutation_rate: float,
    ) -> None:
        checkpoint = {
            "generation": generation,
            "population": population,
            "scores": scores,
            "mutation_rate": mutation_rate,
            "fitness_scores": self.fitness_scores,
            "diversity_scores": self.diversity_scores,
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_gen_{generation}.json")
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f)
        logger.info(f"Saved checkpoint for generation {generation}")

    def load_checkpoint(self, checkpoint_path: str) -> dict:
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint

    def evolve(
        self,
        start_from_checkpoint: str | None = None,
        max_gens: int = 50,
        early_delta_pct: float = 0.25,
        window_gens: int = 3,
        suite: str = "general",
        phase_both: bool = True,
    ) -> list[str]:
        try:
            if start_from_checkpoint:
                checkpoint = self.load_checkpoint(start_from_checkpoint)
                population = checkpoint["population"]
                start_generation = checkpoint["generation"] + 1
                base_mutation_rate = checkpoint["mutation_rate"]
                self.fitness_scores = checkpoint["fitness_scores"]
                self.diversity_scores = checkpoint["diversity_scores"]
            else:
                # Start with 3 parent models (reduced from 8)
                population = self.create_initial_population()[:3]
                start_generation = 0
                base_mutation_rate = self.config.evolution_settings.mutation_rate

            if len(population) < 3:
                msg = "Failed to create 3 initial parent models. Aborting evolution."
                raise EvoMergeException(msg)

            progress_bar = tqdm(
                total=max_gens,
                desc="Evolution Progress",
                unit="generation",
            )

            all_generation_scores = []
            aggregated_fitness_history = []  # Track for early stopping

            for generation in range(start_generation, max_gens):
                logger.info(f"Starting generation {generation + 1}")

                try:
                    # Phase 1: MERGE 3 parents → 8 children
                    phase_pre_children = self.create_generation_children(population, generation, "pre", target_count=8)

                    # BENCHMARK phase_pre children
                    orchestrator = BenchmarkOrchestrator(suite_name=suite)
                    pre_scores = []
                    for i, child_path in enumerate(phase_pre_children):
                        child_id = f"child_{i + 1:02d}"
                        scores = orchestrator.benchmark_model(child_path, generation + 1, "pre", child_id)
                        pre_scores.append(scores)

                    # SELECT top-3 via NSGA-II
                    selected_children, selected_scores = nsga2_select(phase_pre_children, pre_scores, 3)

                    # Phase 2: MERGE+MUTATE selected 3 → 8 new children
                    phase_post_children = self.create_generation_children(
                        selected_children,
                        generation,
                        "post",
                        target_count=8,
                        mutation_rate=base_mutation_rate,
                    )

                    # BENCHMARK phase_post children if enabled
                    post_scores = []
                    if phase_both:
                        for i, child_path in enumerate(phase_post_children):
                            child_id = f"child_{i + 1:02d}"
                            scores = orchestrator.benchmark_model(child_path, generation + 1, "post", child_id)
                            post_scores.append(scores)
                    else:
                        # Use pre_scores for selected children as approximation
                        post_scores = selected_scores

                    # Calculate aggregated fitness for early stopping
                    best_aggregated = max(sum(score.values()) / len(score) for score in post_scores)
                    aggregated_fitness_history.append(best_aggregated)

                    # Store the best score for this generation
                    self.fitness_scores.append(best_aggregated)

                    # Calculate diversity on final children
                    diversity = self.calculate_diversity(phase_post_children)
                    self.diversity_scores.append(diversity)
                    logger.info(f"Population diversity: {diversity}")

                    # Adjust mutation rate based on diversity
                    if len(self.diversity_scores) > 1:
                        if diversity < self.diversity_scores[-2]:
                            base_mutation_rate *= 1.1
                        else:
                            base_mutation_rate *= 0.9
                    base_mutation_rate = max(0.001, min(0.1, base_mutation_rate))
                    logger.info(f"Adjusted mutation rate: {base_mutation_rate}")

                    # Clean up intermediate models
                    clean_up_models([model for model in phase_pre_children if model not in selected_children])

                    # Select top-3 from post phase as next generation parents
                    next_parents, _ = nsga2_select(phase_post_children, post_scores, 3)
                    population = next_parents

                    # Store both phase scores
                    all_generation_scores.append(
                        {
                            "pre": pre_scores,
                            "post": post_scores,
                            "generation": generation + 1,
                        }
                    )

                    # Log detailed generation info
                    self.log_generation_info(generation + 1, population, post_scores)

                    # Save checkpoint
                    if (generation + 1) % 5 == 0:
                        self.save_checkpoint(generation + 1, population, post_scores, base_mutation_rate)

                    # Early stopping check with configurable window
                    if self.should_early_stop(aggregated_fitness_history, early_delta_pct, window_gens):
                        logger.info(
                            f"Early stopping: improvement < {early_delta_pct}% " f"over {window_gens} generations"
                        )
                        break

                    progress_bar.update(1)

                except Exception as e:
                    logger.exception(f"Error in generation {generation + 1}: {e!s}")
                    logger.exception("Attempting to continue with the next generation...")

            progress_bar.close()

            # Plot fitness and diversity over generations
            plot_fitness_over_generations(self.fitness_scores, "fitness_plot.png")
            plot_fitness_over_generations(self.diversity_scores, "diversity_plot.png")

            # Final evaluation
            final_scores = []
            for model in population:
                final_scores.append(orchestrator.benchmark_model(model, generation + 1, "final", "final"))

            # Calculate Pareto front
            pareto_front_indices = calculate_pareto_front(final_scores)
            pareto_optimal_models = [population[i] for i in pareto_front_indices]

            # Plot Pareto front
            plot_pareto_front(final_scores, pareto_front_indices, self.objectives, "pareto_front.png")

            # Generate final report
            self.generate_final_report(pareto_optimal_models, final_scores)

            return pareto_optimal_models

        except Exception as e:
            logger.exception(f"Critical error in evolution process: {e!s}")
            raise

    def generate_final_report(self, pareto_optimal_models: list[str], final_scores: list[dict[str, float]]) -> None:
        report = "EvoMerge Evolution Report\n"
        report += "========================\n\n"
        report += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += "Configuration:\n"
        report += f"- Population size: {self.population_size}\n"
        report += f"- Number of generations: {self.config.evolution_settings.num_generations}\n"
        report += f"- Objectives: {', '.join(self.objectives)}\n\n"
        report += "Results:\n"
        report += f"- Number of Pareto-optimal models: {len(pareto_optimal_models)}\n"
        report += "- Best scores:\n"
        for obj in self.objectives:
            best_score = max(score[obj] for score in final_scores)
            report += f"  - {obj}: {best_score}\n"
        report += "\nPareto-optimal models:\n"
        for i, model in enumerate(pareto_optimal_models):
            report += f"- Model {i + 1}:\n"
            report += f"  - Path: {model}\n"
            report += "  - Scores:\n"
            for obj in self.objectives:
                report += f"    - {obj}: {final_scores[i][obj]}\n"

        with open("evolution_report.txt", "w") as f:
            f.write(report)
        logger.info("Generated final evolution report: evolution_report.txt")


def run_evolutionary_tournament(
    config: Configuration,
    start_from_checkpoint: str | None = None,
    max_gens: int = 50,
    early_delta_pct: float = 0.25,
    window_gens: int = 3,
    suite: str = "general",
    phase_both: bool = True,
) -> list[str]:
    evolutionary_tournament = EvolutionaryTournament(config)
    pareto_optimal_models = evolutionary_tournament.evolve(
        start_from_checkpoint, max_gens, early_delta_pct, window_gens, suite, phase_both
    )

    if pareto_optimal_models:
        logger.info(f"Pareto-optimal models after evolution: {pareto_optimal_models}")
        for model in pareto_optimal_models:
            final_scores = evaluate_model(model)
            logger.info(f"Final scores for model {model}: {final_scores}")

            # Update final scores in model tracker
            model_id = os.path.basename(model)
            model_tracker.update_model_score(model_id, final_scores)
    else:
        logger.error("Evolutionary tournament failed to produce Pareto-optimal models.")

    return pareto_optimal_models


def main() -> None:
    parser = argparse.ArgumentParser(description="EvoMerge: Evolutionary Model Merging")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint file to resume from")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--max_gens", type=int, default=50, help="Maximum generations (default: 50)")
    parser.add_argument(
        "--early_delta_pct",
        type=float,
        default=0.25,
        help="Early stopping improvement threshold percentage (default: 0.25)",
    )
    parser.add_argument(
        "--window_gens",
        type=int,
        default=3,
        help="Early stopping window size in generations (default: 3)",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="general",
        choices=["writing", "coding", "math", "general"],
        help="Benchmark suite to use (default: general)",
    )
    parser.add_argument(
        "--phase_both",
        action="store_true",
        default=True,
        help="Benchmark both pre and post phases (default: True)",
    )
    parser.add_argument(
        "--merge_set",
        type=str,
        default="linear,slerp,ties,dare,task_arith,dfs",
        help="Comma-separated merge techniques (default: linear,slerp,ties,dare,task_arith,dfs)",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        config = Configuration(**config_dict)
    else:
        from .config import create_default_config

        config = create_default_config()

    # Override suite based on model metadata if not explicitly set
    if args.suite == "general" and config.models:
        detected_suite = determine_model_suite(config.models[0].path)
        if detected_suite != "general":
            args.suite = detected_suite
            print(f"Auto-detected benchmark suite: {args.suite}")

    pareto_optimal_models = run_evolutionary_tournament(
        config,
        args.checkpoint,
        args.max_gens,
        args.early_delta_pct,
        args.window_gens,
        args.suite,
        args.phase_both,
    )

    if pareto_optimal_models:
        print("Evolution completed successfully. Pareto-optimal models:")
        for model in pareto_optimal_models:
            print(f"- {model}")
        print("See evolution_report.txt for detailed results.")
    else:
        print("Evolution failed to produce Pareto-optimal models. Check the logs for details.")


if __name__ == "__main__":
    main()
