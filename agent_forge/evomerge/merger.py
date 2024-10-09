import os
import torch
import numpy as np
import random
from typing import List, Dict, Union
import logging
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import cosine_similarity

from .config import MergeConfig, EvolutionConfig, ModelReference

logger = logging.getLogger(__name__)

class MergeTechnique(ABC):
    @abstractmethod
    def apply(self, weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        pass

class PSMergeTechnique(MergeTechnique):
    pass

class DFSMergeTechnique(MergeTechnique):
    pass

class LinearMerge(PSMergeTechnique):
    def apply(self, weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        merged_weights = {}
        for key in weights:
            merged_weights[key] = torch.sum(weights[key] * torch.tensor(kwargs.get("weights", [1/len(weights[key])] * len(weights[key]))).unsqueeze(-1).unsqueeze(-1), dim=0)
        return merged_weights

class SLERPMerge(PSMergeTechnique):
    def apply(self, weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        merged_weights = {}
        for key in weights:
            t = kwargs.get("t", 0.5)
            w1, w2 = weights[key][0], weights[key][1]
            omega = torch.arccos(torch.clamp(cosine_similarity(w1.flatten(), w2.flatten(), dim=0), -1, 1))
            so = torch.sin(omega)
            merged_weights[key] = (torch.sin((1.0-t)*omega) / so).unsqueeze(-1).unsqueeze(-1) * w1 + \
                                  (torch.sin(t*omega) / so).unsqueeze(-1).unsqueeze(-1) * w2
        return merged_weights

class TIESMerge(PSMergeTechnique):
    def apply(self, weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        threshold = kwargs.get("threshold", 0.1)
        merged_weights = {}
        for key in weights:
            tensor = weights[key]
            abs_tensor = torch.abs(tensor)
            mask = abs_tensor > threshold
            merged_weights[key] = torch.where(mask, tensor, torch.zeros_like(tensor))
        return merged_weights

class DAREMerge(PSMergeTechnique):
    def apply(self, weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        threshold = kwargs.get("threshold", 0.1)
        amplification = kwargs.get("amplification", 2.0)
        merged_weights = {}
        for key in weights:
            tensor = weights[key]
            abs_diff = torch.abs(tensor)
            mask = abs_diff > threshold
            merged_weights[key] = torch.where(mask, tensor * amplification, torch.zeros_like(tensor))
        return merged_weights

class TaskArithmetic(PSMergeTechnique):
    def apply(self, weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        base_weights = kwargs.get("base_weights", {})
        task_weights = kwargs.get("task_weights", [])
        merged_weights = {}
        for key in base_weights:
            task_vectors = [task_weight[key] - base_weights[key] for task_weight in task_weights]
            combined_task_vector = sum(task_vectors)
            merged_weights[key] = base_weights[key] + combined_task_vector
        return merged_weights

class Frankenmerge(DFSMergeTechnique):
    def apply(self, weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        models = kwargs.get("models", [])
        merged_weights = {}
        for i, (name, tensor) in enumerate(weights.items()):
            layer_num = int(name.split('.')[1]) if '.' in name else -1
            if layer_num == -1 or layer_num % len(models) == i:
                merged_weights[name] = tensor[i]
        return merged_weights

class DFSMerge(DFSMergeTechnique):
    def apply(self, weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        models = kwargs.get("models", [])
        I = kwargs.get("I", torch.ones(len(weights)))
        W = kwargs.get("W", torch.eye(len(weights), len(models)))
        merged_weights = {}
        layer_index = 0
        for name, tensor in weights.items():
            if any(layer_type in name for layer_type in ['layer', 'block', 'transformer']):
                if I[layer_index] > 0:
                    merged_weights[name] = torch.sum(tensor * W[layer_index].unsqueeze(1).unsqueeze(2), dim=0)
                layer_index += 1
            else:
                merged_weights[name] = tensor[0]
        return merged_weights

class AdvancedModelMerger:
    def __init__(self, config: MergeConfig):
        self.config = config
        self.ps_techniques = {
            "linear": LinearMerge(),
            "slerp": SLERPMerge(),
            "ties": TIESMerge(),
            "dare": DAREMerge(),
            "task_arithmetic": TaskArithmetic()
        }
        self.dfs_techniques = {
            "frankenmerge": Frankenmerge(),
            "dfs": DFSMerge()
        }

    def merge(self) -> str:
        logger.info("Starting advanced model merger process")
        try:
            models = self._load_models()

            if self.config.merge_method == "ps":
                merged_model = self._ps_merge(models)
            elif self.config.merge_method == "dfs":
                merged_model = self._dfs_merge(models)
            elif self.config.merge_method == "ps_dfs":
                ps_model = self._ps_merge(models)
                merged_model = self._dfs_merge([ps_model] + models)
            else:
                raise NotImplementedError(f"Merge method {self.config.merge_method} not implemented")

            merged_model_path = self._save_merged_model(merged_model)

            logger.info(f"Model merging completed successfully. Saved to: {merged_model_path}")
            return merged_model_path
        except Exception as e:
            logger.error(f"Error during merge process: {str(e)}")
            raise

    def _load_models(self) -> List[torch.nn.Module]:
        models = []
        for model_ref in self.config.models:
            logger.info(f"Loading model: {model_ref.name}")
            model = AutoModelForCausalLM.from_pretrained(model_ref.path)
            models.append(model)
        return models

    def _ps_merge(self, models: List[torch.nn.Module]) -> torch.nn.Module:
        weights = self._get_model_weights(models)
        
        for technique in self.config.ps_techniques:
            weights = self.ps_techniques[technique].apply(weights, **self.config.parameters.get(technique, {}))
        
        merged_model = AutoModelForCausalLM.from_pretrained(self.config.models[0].path)
        merged_model.load_state_dict(weights)
        return merged_model

    def _dfs_merge(self, models: List[torch.nn.Module]) -> torch.nn.Module:
        weights = self._get_model_weights(models)
        
        for technique in self.config.dfs_techniques:
            weights = self.dfs_techniques[technique].apply(weights, models=models, **self.config.parameters.get(technique, {}))
        
        merged_model = AutoModelForCausalLM.from_pretrained(self.config.models[0].path)
        merged_model.load_state_dict(weights)
        return merged_model

    def _get_model_weights(self, models: List[torch.nn.Module]) -> Dict[str, torch.Tensor]:
        weights = {}
        for key in models[0].state_dict().keys():
            weights[key] = torch.stack([model.state_dict()[key] for model in models])
        return weights

    def _save_merged_model(self, model: torch.nn.Module) -> str:
        merged_model_name = f"merged_{self.config.merge_method}_{'_'.join([m.name for m in self.config.models])}"
        merged_model_path = os.path.join(self.config.custom_dir, merged_model_name)
        os.makedirs(merged_model_path, exist_ok=True)
        model.save_pretrained(merged_model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.config.models[0].path)
        tokenizer.save_pretrained(merged_model_path)
        return merged_model_path

    def evaluate_model(self, model_path: str) -> Dict[str, Union[float, str]]:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            prompts = [
                "The capital of France is",
                "The theory of relativity was proposed by",
                "The largest planet in our solar system is"
            ]

            total_perplexity = 0
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    perplexity = torch.exp(loss)
                    total_perplexity += perplexity.item()

            avg_perplexity = total_perplexity / len(prompts)
            coherence_score = 1 / avg_perplexity  # Lower perplexity means higher coherence
            return {"overall_score": coherence_score, "perplexity": avg_perplexity}

        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            return {"overall_score": float('-inf'), "error": str(e)}

class EvolutionaryMerger:
    def __init__(self, initial_models: List[ModelReference], merge_config: MergeConfig, evolution_config: EvolutionConfig):
        self.initial_models = initial_models
        self.merge_config = merge_config
        self.evolution_config = evolution_config
        self.merger = AdvancedModelMerger(merge_config)

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
            self.merge_config.ps_techniques = techniques[:2]
            self.merge_config.dfs_techniques = [techniques[2]]
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
                mutation = torch.randn_like(param) * self.evolution_config.mutation_rate
                param.add_(mutation)
        
        new_path = f"{model_path}_mutated_{random.randint(1000, 9999)}"
        model.save_pretrained(new_path)
        return new_path

    def tournament_selection(self, population: List[str], scores: List[float]) -> str:
        tournament = random.sample(list(zip(population, scores)), self.evolution_config.tournament_size)
        return max(tournament, key=lambda x: x[1])[0]

    def evolve(self) -> str:
        population = self.create_merged_models()
        best_score = float('-inf')
        generations_without_improvement = 0

        for generation in range(self.evolution_config.num_generations):
            logger.info(f"Generation {generation + 1}")

            # Evaluate population
            scores = [self.merger.evaluate_model(model)["overall_score"] for model in population]

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
            merged_config = MergeConfig(
                merge_method="ps_dfs",
                models=[ModelReference(name=f"model_{i}", path=model) for i, model in enumerate(lower_performers[:3])],
                parameters=self.merge_config.parameters,
                ps_techniques=["linear", "ties"],
                dfs_techniques=["frankenmerge"]
            )
            merger = AdvancedModelMerger(merged_config)
            merged_model_1 = merger.merge()

            merged_config.models = [ModelReference(name=f"model_{i}", path=model) for i, model in enumerate(lower_performers[3:])]
            merged_model_2 = merger.merge()

            new_population.extend([merged_model_1, merged_model_2])

            # Update population
            population = new_population

            logger.info(f"Best score in generation {generation + 1}: {max(scores)}")

        # Final evaluation
        final_scores = [self.merger.evaluate_model(model)["overall_score"] for model in population]
        best_model = population[final_scores.index(max(final_scores))]

        return best_model

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Define initial models
    initial_models = [
        ModelReference(name="gpt2", path="gpt2"),
        ModelReference(name="gpt2-medium", path="gpt2-medium"),
        ModelReference(name="gpt2-large", path="gpt2-large")
    ]

    # Create merge configuration
    merge_config = MergeConfig(
        merge_method="ps_dfs",
        models=initial_models,
        parameters={
            "linear": {"weights": [1/3, 1/3, 1/3]},
            "slerp": {"t": 0.5},
            "ties": {"threshold": 0.1},
            "dare": {"threshold": 0.1, "amplification": 2.0},
            "frankenmerge": {},
            "dfs": {"I": torch.ones(len(initial_models)), "W": torch.eye(len(initial_models))}
        }
    )

    # Create evolution configuration
    evolution_config = EvolutionConfig(
        population_size=8,
        num_generations=50,
        mutation_rate=0.1,
        tournament_size=3,
        early_stopping_generations=10
    )

    # Create evolutionary merger
    evolutionary_merger = EvolutionaryMerger(initial_models, merge_config, evolution_config)

    # Run evolution
    best_model = evolutionary_merger.evolve()

    # Print results
    logger.info(f"Best model after evolution: {best_model}")
    final_score = evolutionary_merger.merger.evaluate_model(best_model)
    logger.info(f"Final score: {final_score}")

if __name__ == "__main__":
    main()
