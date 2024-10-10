import logging
import random
from typing import List, Dict, Union
import torch
from transformers import AutoModelForCausalLM

from .config import Configuration, ModelReference
from .merger import AdvancedModelMerger
from .utils import evaluate_model, parallel_evaluate_models, MERGE_TECHNIQUES
from .visualization import plot_fitness_over_generations

logger = logging.getLogger(__name__)

class EvolutionaryTournament:
    def __init__(self, config: Configuration):
        self.config = config
        self.merger = AdvancedModelMerger(config)
        self.fitness_scores = []

    def create_merged_models(self) -> List[str]:
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

        merged_models = []
        for techniques in merge_combinations:
            self.config.merge_settings.ps_techniques = techniques[:2]
            self.config.merge_settings.dfs_techniques = [techniques[2]]
            try:
                merged_model_path = self.merger.merge()
                merged_models.append(merged_model_path)
                logger.info(f"Successfully created merged model: {merged_model_path}")
            except Exception as e:
                logger.error(f"Failed to create merged model with techniques {techniques}: {str(e)}")
        
        return merged_models

    def mutate_model(self, model_path: str) -> str:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        with torch.no_grad():
            for param in model.parameters():
                mutation = torch.randn_like(param) * self.config.evolution_settings.mutation_rate
                param.add_(mutation)
        
        new_path = f"{model_path}_mutated_{random.randint(1000, 9999)}"
        model.save_pretrained(new_path)
        return new_path

    def tournament_selection(self, population: List[str], scores: List[float]) -> str:
        tournament = random.sample(list(zip(population, scores)), self.config.evolution_settings.tournament_size)
        return max(tournament, key=lambda x: x[1])[0]

    def evolve(self) -> str:
        population = self.create_merged_models()
        if not population:
            logger.error("Failed to create any merged models. Aborting evolution.")
            return None

        best_score = float('-inf')
        generations_without_improvement = 0

        for generation in range(self.config.evolution_settings.num_generations):
            logger.info(f"Generation {generation + 1}")

            # Evaluate population
            scores = parallel_evaluate_models([model for model in population])
            scores = [score["overall_score"] for score in scores if score is not None]

            if not scores:
                logger.error(f"All model evaluations failed in generation {generation + 1}. Skipping this generation.")
                continue

            # Store the best score for this generation
            self.fitness_scores.append(max(scores))

            # Check for improvement
            current_best_score = max(scores)
            if current_best_score > best_score:
                best_score = current_best_score
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Early stopping
            if generations_without_improvement >= self.config.evolution_settings.early_stopping_generations:
                logger.info(f"Early stopping triggered after {generation + 1} generations")
                break

            # Select top performers
            top_models = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)[:2]

            # Create new population
            new_population = [model for model, _ in top_models]

            # Mutate top performers
            for _ in range(3):
                for model, _ in top_models:
                    new_population.append(self.mutate_model(model))

            # Merge lower performers
            lower_performers = [model for model in population if model not in [m for m, _ in top_models]]
            if len(lower_performers) >= 3:
                merged_config = Configuration(
                    models=[ModelReference(name=f"model_{i}", path=model) for i, model in enumerate(lower_performers[:3])],
                    merge_settings=self.config.merge_settings,
                    evolution_settings=self.config.evolution_settings
                )
                merger = AdvancedModelMerger(merged_config)
                merged_model_1 = merger.merge()
                new_population.append(merged_model_1)

            if len(lower_performers) >= 6:
                merged_config.models = [ModelReference(name=f"model_{i}", path=model) for i, model in enumerate(lower_performers[3:6])]
                merged_model_2 = merger.merge()
                new_population.append(merged_model_2)

            # Update population
            population = new_population

            logger.info(f"Best score in generation {generation + 1}: {max(scores)}")

        # Plot fitness over generations
        plot_fitness_over_generations(self.fitness_scores, 'fitness_plot.png')

        # Final evaluation
        final_scores = parallel_evaluate_models([model for model in population])
        final_scores = [score["overall_score"] for score in final_scores if score is not None]
        
        if not final_scores:
            logger.error("All final model evaluations failed. Returning the last model in the population.")
            return population[-1]

        best_model = population[final_scores.index(max(final_scores))]

        return best_model

def run_evolutionary_tournament(config: Configuration) -> str:
    evolutionary_tournament = EvolutionaryTournament(config)
    best_model = evolutionary_tournament.evolve()
    
    if best_model:
        logger.info(f"Best model after evolution: {best_model}")
        final_score = evaluate_model(best_model)
        logger.info(f"Final scores: {final_score}")
    else:
        logger.error("Evolutionary tournament failed to produce a best model.")
    
    return best_model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from .config import create_default_config

    config = create_default_config()
    best_model = run_evolutionary_tournament(config)
