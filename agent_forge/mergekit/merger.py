import os
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from scipy.stats import special_ortho_group
from torch.nn.functional import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelReference(BaseModel):
    name: str
    path: str  # Hugging Face model ID or local path

class MergeConfig(BaseModel):
    merge_method: str
    models: List[ModelReference]
    parameters: Dict[str, any] = Field(default_factory=dict)
    custom_dir: str = Field(default="./merged_models")
    evolutionary_params: Optional[Dict[str, any]] = None
    ps_techniques: List[str] = ["linear"]
    dfs_techniques: List[str] = []

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

    def evaluate_model(self, model_path: str) -> float:
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
            return coherence_score

        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            return float('-inf')  # Return worst possible score if evaluation fails

def create_merged_models(initial_models: List[ModelReference]) -> List[str]:
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
        config = MergeConfig(
            merge_method="ps_dfs",
            models=initial_models,
            parameters={
                "linear": {"weights": [1/3, 1/3, 1/3]},
                "slerp": {"t": 0.5},
                "ties": {"threshold": 0.1},
                "dare": {"threshold": 0.1, "amplification": 2.0},
                "frankenmerge": {},
                "dfs": {"I": torch.ones(len(initial_models)), "W": torch.eye(len(initial_models))}
            },
            ps_techniques=techniques[:2],
            dfs_techniques=[techniques[2]]
        )
        merger = AdvancedModelMerger(config)
        try:
            merged_model_path = merger.merge()
            merged_models.append(merged_model_path)
            logger.info(f"Successfully created merged model: {merged_model_path}")
        except Exception as e:
            logger.error(f"Failed to create merged model with techniques {techniques}: {str(e)}")
    
    return merged_models

# Usage example
if __name__ == "__main__":
    initial_models = [
        ModelReference(name="gpt2", path="gpt2"),
        ModelReference(name="gpt2-medium", path="gpt2-medium"),
        ModelReference(name="gpt2-large", path="gpt2-large")
    ]

    merged_models = create_merged_models(initial_models)
    print(f"Created {len(merged_models)} merged models")
    for model_path in merged_models:
        print(f"Merged model saved at: {model_path}")
        merger = AdvancedModelMerger(MergeConfig(merge_method="", models=[]))  # Dummy config for evaluation
        score = merger.evaluate_model(model_path)
        print(f"Model coherence score: {score}")