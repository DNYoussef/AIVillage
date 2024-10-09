# evolutionary_tournament.py

import logging
import random
from typing import List, Dict, Union
from pydantic import BaseModel, Field

from .config import MergeConfig, EvolutionConfig, ModelReference
from .merger import AdvancedModelMerger

logger = logging.getLogger(__name__)

class EvolutionaryMerger:
    def __init__(self, base_model: ModelReference, tasks: List[str], merge_config: MergeConfig, evolution_config: EvolutionConfig):
        self.base_model = base_model
        self.tasks = tasks
        self.merge_config = merge_config
        self.evolution_config = evolution_config
        self.merger = AdvancedModelMerger(merge_config)

    def create_expert_models(self) -> List[ModelReference]:
        expert_models = []
        for task in self.tasks:
            expert_model = self._fine_tune_model(self.base_model, task)
            expert_models.append(expert_model)
        return expert_models

    def _fine_tune_model(self, base_model: ModelReference, task: str) -> ModelReference:
        # TODO: Implement actual fine-tuning logic here
        logger.warning("Fine-tuning is not implemented. Using a placeholder.")
        fine_tuned_model = ModelReference(
            name=f"{base_model.name}_finetuned_{task}",
            path=f"{base_model.path}_finetuned_{task}"
        )
        return fine_tuned_model

    def evolve(self) -> str:
        expert_models = self.create_expert_models()
        population = self.merger.create_merged_models(expert_models)
        best_score = float('-inf')
        generations_without_improvement = 0

        for generation in range(self.evolution_config.num_generations):
            logger.info(f"Generation {generation + 1}")

            scores = self._evaluate_population(population)
            
            # Check for improvement
            current_best_score = max(scores)
            if current_best_score > best_score:
                best_score = current_best_score
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Early stopping
            if generations_without_improvement >= self.evolution_config.early_stopping_generations:
                logger.info(f"Early stopping triggered after {generation + 1} generations")
                break

            top_performers = self._select_top_performers(population, scores, 2)
            merged_models = self._merge_lower_performers(population, top_performers)
            mutated_models = self._mutate_top_performers(top_performers)
            
            new_population = mutated_models + merged_models
            population = new_population

            logger.info(f"Best score in generation {generation + 1}: {best_score}")

        final_scores = self._evaluate_population(population)
        best_model = population[final_scores.index(max(final_scores))]

        return best_model

    def _select_top_performers(self, population: List[str], scores: List[float], n: int) -> List[str]:
        return [x for _, x in sorted(zip(scores, population), reverse=True)][:n]

    def _merge_lower_performers(self, population: List[str], top_performers: List[str]) -> List[str]:
        lower_performers = [model for model in population if model not in top_performers]
        
        group1 = lower_performers[:len(lower_performers)//2]
        group2 = lower_performers[len(lower_performers)//2:]
        
        merged_model1 = self._merge_models(group1)
        merged_model2 = self._merge_models(group2)
        
        return [merged_model1, merged_model2]

    def _merge_models(self, models: List[str]) -> str:
        temp_config = MergeConfig(
            merge_method="ps_dfs",
            models=[ModelReference(name=f"model_{i}", path=model) for i, model in enumerate(models)],
            parameters=self.merge_config.parameters,
            ps_techniques=["linear", "ties"],
            dfs_techniques=["frankenmerge"]
        )
        
        temp_merger = AdvancedModelMerger(temp_config)
        merged_model = temp_merger.merge()
        return merged_model

    def _mutate_top_performers(self, top_performers: List[str]) -> List[str]:
        mutated_models = []
        for model in top_performers:
            for _ in range(3):
                mutated_model = self.merger.mutate_model(model)
                mutated_models.append(mutated_model)
        return mutated_models

    def _evaluate_population(self, population: List[str]) -> List[float]:
        return [self.merger.evaluate_model(model)["overall_score"] for model in population]

def run_evolutionary_tournament(base_model: ModelReference, tasks: List[str], merge_config: MergeConfig, evolution_config: EvolutionConfig) -> str:
    evolutionary_merger = EvolutionaryMerger(base_model, tasks, merge_config, evolution_config)
    best_model = evolutionary_merger.evolve()
    
    logger.info(f"Best model after evolution: {best_model}")
    final_score = evolutionary_merger.merger.evaluate_model(best_model)
    logger.info(f"Final scores: {final_score}")
    
    return best_model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    base_model = ModelReference(name="gpt2", path="gpt2")
    tasks = ["task1", "task2", "task3"]  # Define your specific tasks here

    merge_config = MergeConfig(
        merge_method="ps_dfs",
        models=[],  # This will be filled by create_expert_models
        parameters={
            "linear": {"weights": [1/3, 1/3, 1/3]},
            "slerp": {"t": 0.5},
            "ties": {"threshold": 0.1},
            "dare": {"threshold": 0.1, "amplification": 2.0},
        }
    )

    evolution_config = EvolutionConfig(
        population_size=8,
        num_generations=50,
        mutation_rate=0.1,
        tournament_size=3,
        early_stopping_generations=10
    )

    best_model = run_evolutionary_tournament(base_model, tasks, merge_config, evolution_config)
