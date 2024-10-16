import logging
import random
from typing import List, Dict, Union, Tuple
import torch
import numpy as np
from transformers import AutoModelForCausalLM
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import psutil
import shutil
import argparse
from datetime import datetime

from .config import Configuration, ModelReference
from .merger import AdvancedModelMerger
from .evaluation import evaluate_model
from .merge_techniques import MERGE_TECHNIQUES
from .multi_objective import nsga2_select, calculate_pareto_front
from .visualization import plot_fitness_over_generations, plot_pareto_front, plot_evolution_progress
from .utils import EvoMergeException, clean_up_models
from .model_tracker import model_tracker

logger = logging.getLogger(__name__)

class EvolutionaryTournament:
    """
    Implements an evolutionary tournament for model merging and optimization.

    This class manages the entire process of evolving a population of merged models,
    including creation, mutation, evaluation, and selection of the best models.
    """

    def __init__(self, config: Configuration):
        """
        Initialize the EvolutionaryTournament.

        Args:
            config (Configuration): Configuration object containing all settings for the tournament.
        """
        self.config = config
        self.merger = AdvancedModelMerger(config)
        self.fitness_scores = []
        self.diversity_scores = []
        self.checkpoint_dir = os.path.join(self.config.merge_settings.custom_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.population_size = self.config.evolution_settings.population_size
        self.objectives = self.config.evolution_settings.objectives
        self.objective_types = self.config.evolution_settings.objective_types  # 'maximize' or 'minimize'
        self.setup_logging()

    def setup_logging(self):
        """Set up detailed logging for the evolutionary process."""
        self.log_file = os.path.join(self.config.merge_settings.custom_dir, "evolution_log.txt")
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def log_generation_info(self, generation: int, population: List[str], scores: List[Dict[str, float]]):
        """Log detailed information about the current generation."""
        logger.info(f"Generation {generation} completed")
        logger.info(f"Population size: {len(population)}")
        logger.info(f"Best scores: {max(scores, key=lambda x: x[self.objectives[0]])}")
        logger.info(f"Average scores: {{obj: np.mean([s[obj] for s in scores]) for obj in self.objectives}}")
        logger.info(f"Diversity: {self.diversity_scores[-1]}")

    def create_initial_population(self) -> List[str]:
        """
        Create the initial population of merged models.

        Returns:
            List[str]: A list of file paths to the created models.
        """
        logger.info("Creating initial population")
        merge_combinations = [
            ["linear", "ties", "frankenmerge"],
            ["linear", "ties", "dfs"],
            ["linear", "dare", "frankenmerge"],
            ["linear", "dare", "dfs"],
            ["slerp", "ties", "frankenmerge"],
            ["slerp", "ties", "dfs"],
            ["slerp", "dare", "frankenmerge"],
            ["slerp", "dare", "dfs"]
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
                logger.error(f"Failed to create merged model with techniques {techniques}: {str(e)}")

        return population

    def mutate_model(self, model_path: str, mutation_rate: float) -> str:
        """
        Mutate a given model by adding random noise to its parameters.

        Args:
            model_path (str): Path to the model to be mutated.
            mutation_rate (float): The rate at which to mutate the model parameters.

        Returns:
            str: Path to the mutated model.
        """
        logger.info(f"Mutating model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path)

        with torch.no_grad():
            for param in model.parameters():
                mutation = torch.randn_like(param) * mutation_rate
                param.add_(mutation)

        new_path = f"{model_path}_mutated_{random.randint(1000, 9999)}"
        model.save_pretrained(new_path)
        return new_path

    def merge_models(self, models: List[str]) -> str:
        """
        Merge a list of models into a single model.

        Args:
            models (List[str]): List of paths to the models to be merged.

        Returns:
            str: Path to the merged model.
        """
        logger.info(f"Merging models: {models}")
        merged_config = Configuration(
            models=[ModelReference(name=f"model_{i}", path=model) for i, model in enumerate(models)],
            merge_settings=self.config.merge_settings,
            evolution_settings=self.config.evolution_settings
        )
        merger = AdvancedModelMerger(merged_config)
        return merger.merge()

    def calculate_diversity(self, population: List[str]) -> float:
        """
        Calculate the diversity of the current population.

        Args:
            population (List[str]): List of paths to the models in the population.

        Returns:
            float: A measure of the population's diversity.
        """
        models = [AutoModelForCausalLM.from_pretrained(path) for path in population]
        param_vectors = [self.flatten_params(model) for model in models]

        diversity = 0
        for i in range(len(param_vectors)):
            for j in range(i+1, len(param_vectors)):
                diversity += torch.dist(param_vectors[i], param_vectors[j]).item()

        return diversity / (len(population) * (len(population) - 1) / 2)

    def flatten_params(self, model: torch.nn.Module) -> torch.Tensor:
        """
        Flatten all parameters of a model into a single tensor.

        Args:
            model (torch.nn.Module): The model whose parameters to flatten.

        Returns:
            torch.Tensor: A 1D tensor containing all flattened parameters.
        """
        return torch.cat([p.data.view(-1) for p in model.parameters()])

    def save_checkpoint(self, generation: int, population: List[str], scores: List[float], mutation_rate: float):
        """
        Save a checkpoint of the current state of the evolution.

        Args:
            generation (int): The current generation number.
            population (List[str]): List of paths to the current population of models.
            scores (List[float]): List of scores for the current population.
            mutation_rate (float): The current mutation rate.
        """
        checkpoint = {
            "generation": generation,
            "population": population,
            "scores": scores,
            "mutation_rate": mutation_rate,
            "fitness_scores": self.fitness_scores,
            "diversity_scores": self.diversity_scores
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_gen_{generation}.json")
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
        logger.info(f"Saved checkpoint for generation {generation}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load a checkpoint from a file.

        Args:
            checkpoint_path (str): Path to the checkpoint file.

        Returns:
            Dict: The loaded checkpoint data.
        """
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint

    def parallel_evaluate_models(self, population: List[str]) -> List[Dict[str, float]]:
        """
        Evaluate all models in the population in parallel.

        Args:
            population (List[str]): List of paths to the models to evaluate.

        Returns:
            List[Dict[str, float]]: List of dictionaries containing scores for each objective for each model in the population.
        """
        with ProcessPoolExecutor() as executor:
            future_to_model = {executor.submit(evaluate_model, model): model for model in population}
            scores = []
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    result = future.result()
                    scores.append({obj: result[obj] for obj in self.objectives})
                    logger.info(f"Evaluated model {model}: scores = {scores[-1]}")
                except Exception as exc:
                    logger.error(f"Model {model} generated an exception: {exc}")
                    scores.append({obj: float('-inf') for obj in self.objectives})
        return scores

    def tournament_selection(self, population: List[str], scores: List[Dict[str, float]], tournament_size: int = 3) -> str:
        """
        Perform tournament selection to choose a model from the population.

        Args:
            population (List[str]): List of paths to the models in the population.
            scores (List[Dict[str, float]]): List of score dictionaries corresponding to the models in the population.
            tournament_size (int): Number of models to include in each tournament.

        Returns:
            str: Path to the selected model.
        """
        tournament = random.sample(list(zip(population, scores)), tournament_size)
        return max(tournament, key=lambda x: sum(x[1].values()))[0]

    def evolve(self, start_from_checkpoint: str = None) -> List[str]:
        """
        Run the evolutionary process.

        Args:
            start_from_checkpoint (str, optional): Path to a checkpoint file to resume from.

        Returns:
            List[str]: Paths to the Pareto-optimal models produced by the evolution.
        """
        try:
            self.validate_config()

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

            if len(population) < 8:
                raise EvoMergeException("Failed to create enough initial models. Aborting evolution.")

            progress_bar = tqdm(total=self.config.evolution_settings.num_generations,
                                desc="Evolution Progress", unit="generation")

            all_generation_scores = []

            for generation in range(start_generation, self.config.evolution_settings.num_generations):
                logger.info(f"Starting generation {generation + 1}")

                try:
                    # Adjust population size based on available resources
                    self.adjust_population_size()

                    # Evaluate population in parallel
                    scores = self.parallel_evaluate_models(population)
                    all_generation_scores.append(scores)

                    # Calculate diversity
                    diversity = self.calculate_diversity(population)
                    self.diversity_scores.append(diversity)
                    logger.info(f"Population diversity: {diversity}")

                    # Adjust mutation rate based on diversity
                    if len(self.diversity_scores) > 1:
                        if diversity < self.diversity_scores[-2]:
                            base_mutation_rate *= 1.1  # Increase mutation rate if diversity decreased
                        else:
                            base_mutation_rate *= 0.9  # Decrease mutation rate if diversity increased
                    base_mutation_rate = max(0.001, min(0.1, base_mutation_rate))  # Keep mutation rate within reasonable bounds
                    logger.info(f"Adjusted mutation rate: {base_mutation_rate}")

                    # Store the best score for this generation
                    self.fitness_scores.append(max(score[self.objectives[0]] for score in scores))

                    # Perform NSGA-II selection
                    selected_population, selected_scores = nsga2_select(population, scores, self.population_size, self.objective_types)

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
                    self.clean_up_resources(new_population)

                    # Update population
                    population = new_population

                    # Log performance metrics
                    metrics = self.get_performance_metrics()
                    logger.info(f"Performance metrics: {metrics}")

                    # Log detailed generation info
                    self.log_generation_info(generation + 1, population, scores)

                    # Save checkpoint
                    if (generation + 1) % 5 == 0:  # Save checkpoint every 5 generations
                        self.save_checkpoint(generation + 1, population, scores, base_mutation_rate)

                    # Update real-time visualization
                    plot_evolution_progress(all_generation_scores, self.objectives, self.objective_types, 'evolution_progress.png')

                    # Early stopping check
                    if generation > self.config.evolution_settings.early_stopping_generations:
                        if max(self.fitness_scores[-self.config.evolution_settings.early_stopping_generations:]) <= self.fitness_scores[-self.config.evolution_settings.early_stopping_generations-1]:
                            logger.info("Early stopping criterion met. Ending evolution.")
                            break

                    progress_bar.update(1)
                except Exception as e:
                    logger.error(f"Error in generation {generation + 1}: {str(e)}")
                    logger.error("Attempting to continue with the next generation...")

            progress_bar.close()

            # Plot fitness and diversity over generations
            plot_fitness_over_generations(self.fitness_scores, 'fitness_plot.png')
            plot_fitness_over_generations(self.diversity_scores, 'diversity_plot.png')

            # Final evaluation
            final_scores = self.parallel_evaluate_models(population)

            # Calculate Pareto front
            pareto_front_indices = calculate_pareto_front(final_scores, self.objective_types)
            pareto_optimal_models = [population[i] for i in pareto_front_indices]

            # Plot Pareto front
            plot_pareto_front(final_scores, pareto_front_indices, self.objectives, 'pareto_front.png')

            # Generate final report
            self.generate_final_report(pareto_optimal_models, final_scores)

            return pareto_optimal_models

        except Exception as e:
            logger.error(f"Critical error in evolution process: {str(e)}")
            raise

    def generate_final_report(self, pareto_optimal_models: List[str], final_scores: List[Dict[str, float]]):
        """Generate a comprehensive report of the evolution process."""
        report = f"EvoMerge Evolution Report\n"
        report += f"========================\n\n"
        report += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"Configuration:\n"
        report += f"- Population size: {self.population_size}\n"
        report += f"- Number of generations: {self.config.evolution_settings.num_generations}\n"
        report += f"- Objectives: {', '.join(self.objectives)}\n\n"
        report += f"Results:\n"
        report += f"- Number of Pareto-optimal models: {len(pareto_optimal_models)}\n"
        report += f"- Best scores:\n"
        for obj in self.objectives:
            best_score = max(score[obj] for score in final_scores)
            report += f"  - {obj}: {best_score}\n"
        report += f"\nPareto-optimal models:\n"
        for i, model in enumerate(pareto_optimal_models):
            report += f"- Model {i+1}:\n"
            report += f"  - Path: {model}\n"
            report += f"  - Scores:\n"
            for obj in self.objectives:
                report += f"    - {obj}: {final_scores[i][obj]}\n"

        with open('evolution_report.txt', 'w') as f:
            f.write(report)
        logger.info("Generated final evolution report: evolution_report.txt")

def run_evolutionary_tournament(config: Configuration, start_from_checkpoint: str = None) -> List[str]:
    """
    Run the evolutionary tournament process.

    Args:
        config (Configuration): Configuration object for the tournament.
        start_from_checkpoint (str, optional): Path to a checkpoint file to resume from.

    Returns:
        List[str]: Paths to the Pareto-optimal models produced by the tournament.
    """
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

def main():
    parser = argparse.ArgumentParser(description="EvoMerge: Evolutionary Model Merging")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint file to resume from")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if args.config:
        with open(args.config, 'r') as f:
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
        print("Evolution failed to produce Pareto-optimal models. Check the logs for details.")

if __name__ == "__main__":
    main()
