import argparse
from datetime import datetime
import json
import logging
import os
import random

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from .config import Configuration, ModelReference
from .evaluation import evaluate_model
from .merging.merger import AdvancedModelMerger
from .model_tracker import model_tracker
from .multi_objective import calculate_pareto_front, nsga2_select
from .utils import EvoMergeException, clean_up_models, parallel_evaluate_models
from .visualization import (
    plot_evolution_progress,
    plot_fitness_over_generations,
    plot_pareto_front,
)

logger = logging.getLogger(__name__)


class EvolutionaryTournament:
    def __init__(self, config: Configuration) -> None:
        self.config = config
        self.merger = AdvancedModelMerger(config)
        self.fitness_scores = []
        self.diversity_scores = []
        self.checkpoint_dir = os.path.join(
            self.config.merge_settings.custom_dir, "checkpoints"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.population_size = self.config.evolution_settings.population_size
        self.objectives = self.config.evolution_settings.objectives
        self.setup_logging()

    def setup_logging(self) -> None:
        self.log_file = os.path.join(
            self.config.merge_settings.custom_dir, "evolution_log.txt"
        )
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def log_generation_info(
        self, generation: int, population: list[str], scores: list[dict[str, float]]
    ) -> None:
        logger.info(f"Generation {generation} completed")
        logger.info(f"Population size: {len(population)}")
        logger.info(f"Best scores: {max(scores, key=lambda x: x[self.objectives[0]])}")
        logger.info(
            "Average scores: {obj: np.mean([s[obj] for s in scores]) for obj in self.objectives}"
        )
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
                logger.exception(
                    f"Failed to create merged model with techniques {techniques}: {e!s}"
                )

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
            models=[
                ModelReference(name=f"model_{i}", path=model)
                for i, model in enumerate(models)
            ],
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
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_gen_{generation}.json"
        )
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f)
        logger.info(f"Saved checkpoint for generation {generation}")

    def load_checkpoint(self, checkpoint_path: str) -> dict:
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint

    def evolve(self, start_from_checkpoint: str | None = None) -> list[str]:
        try:
            if start_from_checkpoint:
                checkpoint = self.load_checkpoint(start_from_checkpoint)
                population = checkpoint["population"]
                start_generation = checkpoint["generation"] + 1
                base_mutation_rate = checkpoint["mutation_rate"]
                self.fitness_scores = checkpoint["fitness_scores"]
                self.diversity_scores = checkpoint["diversity_scores"]
            else:
                population = self.create_initial_population()
                start_generation = 0
                base_mutation_rate = self.config.evolution_settings.mutation_rate

            if len(population) < self.population_size:
                msg = "Failed to create enough initial models. Aborting evolution."
                raise EvoMergeException(msg)

            progress_bar = tqdm(
                total=self.config.evolution_settings.num_generations,
                desc="Evolution Progress",
                unit="generation",
            )

            all_generation_scores = []

            for generation in range(
                start_generation, self.config.evolution_settings.num_generations
            ):
                logger.info(f"Starting generation {generation + 1}")

                try:
                    # Evaluate population in parallel
                    scores = parallel_evaluate_models(population)
                    all_generation_scores.append(scores)

                    # Calculate diversity
                    diversity = self.calculate_diversity(population)
                    self.diversity_scores.append(diversity)
                    logger.info(f"Population diversity: {diversity}")

                    # Adjust mutation rate based on diversity
                    if len(self.diversity_scores) > 1:
                        if diversity < self.diversity_scores[-2]:
                            base_mutation_rate *= (
                                1.1  # Increase mutation rate if diversity decreased
                            )
                        else:
                            base_mutation_rate *= (
                                0.9  # Decrease mutation rate if diversity increased
                            )
                    base_mutation_rate = max(
                        0.001, min(0.1, base_mutation_rate)
                    )  # Keep mutation rate within reasonable bounds
                    logger.info(f"Adjusted mutation rate: {base_mutation_rate}")

                    # Store the best score for this generation
                    self.fitness_scores.append(
                        max(score[self.objectives[0]] for score in scores)
                    )

                    # Perform NSGA-II selection
                    selected_population, selected_scores = nsga2_select(
                        population, scores, self.population_size
                    )

                    # Create new population
                    new_population = []
                    for model in selected_population:
                        new_population.append(model)  # Keep original
                        mutated_model = self.mutate_model(model, base_mutation_rate)
                        new_population.append(mutated_model)

                    # Ensure population size is maintained
                    while len(new_population) > self.population_size:
                        new_population.pop(random.randrange(len(new_population)))

                    # Clean up resources
                    clean_up_models(
                        [model for model in population if model not in new_population]
                    )

                    # Update population
                    population = new_population

                    # Log detailed generation info
                    self.log_generation_info(generation + 1, population, scores)

                    # Save checkpoint
                    if (generation + 1) % 5 == 0:  # Save checkpoint every 5 generations
                        self.save_checkpoint(
                            generation + 1, population, scores, base_mutation_rate
                        )

                    # Update real-time visualization
                    plot_evolution_progress(
                        all_generation_scores, self.objectives, "evolution_progress.png"
                    )

                    # Early stopping check
                    if (
                        generation
                        > self.config.evolution_settings.early_stopping_generations
                    ):
                        if (
                            max(
                                self.fitness_scores[
                                    -self.config.evolution_settings.early_stopping_generations :
                                ]
                            )
                            <= self.fitness_scores[
                                -self.config.evolution_settings.early_stopping_generations
                                - 1
                            ]
                        ):
                            logger.info(
                                "Early stopping criterion met. Ending evolution."
                            )
                            break

                    progress_bar.update(1)
                except Exception as e:
                    logger.exception(f"Error in generation {generation + 1}: {e!s}")
                    logger.exception(
                        "Attempting to continue with the next generation..."
                    )

            progress_bar.close()

            # Plot fitness and diversity over generations
            plot_fitness_over_generations(self.fitness_scores, "fitness_plot.png")
            plot_fitness_over_generations(self.diversity_scores, "diversity_plot.png")

            # Final evaluation
            final_scores = parallel_evaluate_models(population)

            # Calculate Pareto front
            pareto_front_indices = calculate_pareto_front(final_scores)
            pareto_optimal_models = [population[i] for i in pareto_front_indices]

            # Plot Pareto front
            plot_pareto_front(
                final_scores, pareto_front_indices, self.objectives, "pareto_front.png"
            )

            # Generate final report
            self.generate_final_report(pareto_optimal_models, final_scores)

            return pareto_optimal_models

        except Exception as e:
            logger.exception(f"Critical error in evolution process: {e!s}")
            raise

    def generate_final_report(
        self, pareto_optimal_models: list[str], final_scores: list[dict[str, float]]
    ) -> None:
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
    config: Configuration, start_from_checkpoint: str | None = None
) -> list[str]:
    evolutionary_tournament = EvolutionaryTournament(config)
    pareto_optimal_models = evolutionary_tournament.evolve(start_from_checkpoint)

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
    parser.add_argument(
        "--checkpoint", type=str, help="Path to a checkpoint file to resume from"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        config = Configuration(**config_dict)
    else:
        from .config import create_default_config

        config = create_default_config()

    pareto_optimal_models = run_evolutionary_tournament(config, args.checkpoint)

    if pareto_optimal_models:
        print("Evolution completed successfully. Pareto-optimal models:")
        for model in pareto_optimal_models:
            print(f"- {model}")
        print("See evolution_report.txt for detailed results.")
    else:
        print(
            "Evolution failed to produce Pareto-optimal models. Check the logs for details."
        )


if __name__ == "__main__":
    main()
