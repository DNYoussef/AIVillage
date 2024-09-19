import os
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import requests
import json
import logging
from pydantic import BaseModel
from abc import ABC, abstractmethod
from cmaes import CMA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelReference(BaseModel):
    name: str
    path: str  # Hugging Face model ID or local path

class MergeConfig(BaseModel):
    merge_method: str
    models: List[ModelReference]
    parameters: Optional[dict] = None
    custom_dir: Optional[str] = None
    evolutionary_params: Optional[dict] = None
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
        return weights

class SLERPMerge(PSMergeTechnique):
    def apply(self, weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        return self._slerp_interpolation(weights, **kwargs)

    def _slerp_interpolation(self, tensor1, tensor2, weight1, weight2):
        omega = torch.arccos(torch.clamp(torch.sum(tensor1*tensor2) / (torch.norm(tensor1) * torch.norm(tensor2)), -1, 1))
        so = torch.sin(omega)
        return torch.sin((1.0-weight2)*omega) / so * tensor1 + torch.sin(weight2*omega) / so * tensor2

class TIESMerge(PSMergeTechnique):
    def apply(self, weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        return AdvancedModelMerger._apply_ties(weights, **kwargs)

class DAREMerge(PSMergeTechnique):
    def apply(self, weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        return AdvancedModelMerger._apply_dare(weights, **kwargs)

class TaskArithmetic(PSMergeTechnique):
    def apply(self, weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        return AdvancedModelMerger._task_arithmetic(weights, **kwargs)

class Frankenmerge(DFSMergeTechnique):
    def apply(self, weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        return AdvancedModelMerger._frankenmerge(weights, **kwargs)

class DFSMerge(DFSMergeTechnique):
    def apply(self, weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        return AdvancedModelMerger._create_dfs_model(weights, **kwargs)

class AdvancedModelMerger:
    def __init__(self, config: MergeConfig):
        self.config = config
        self.custom_dir = config.custom_dir if config.custom_dir else "C:\\Users\\17175\\Desktop\\AI_Models"
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
            # Load models
            models = self._load_models()

            if self.config.merge_method in self.ps_techniques:
                merged_model = self._ps_merge(models)
            elif self.config.merge_method in self.dfs_techniques:
                merged_model = self._dfs_merge(models)
            elif self.config.merge_method == "ps_dfs":
                ps_model = self._ps_merge(models)
                merged_model = self._dfs_merge([ps_model] + models)
            else:
                raise NotImplementedError(f"Merge method {self.config.merge_method} not implemented")

            # Save merged model
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
            model = AutoModel.from_pretrained(model_ref.path)
            models.append(model)
        return models

    def _ps_merge(self, models: List[torch.nn.Module]) -> torch.nn.Module:
        weights = self._get_model_weights(models)
        
        for technique in self.config.ps_techniques:
            weights = self.ps_techniques[technique].apply(weights, **self.config.parameters)
        
        merged_model = AutoModel.from_pretrained(self.config.models[0].path)
        merged_model.load_state_dict(weights)
        return merged_model

    def _dfs_merge(self, models: List[torch.nn.Module]) -> torch.nn.Module:
        weights = self._get_model_weights(models)
        
        for technique in self.config.dfs_techniques:
            weights = self.dfs_techniques[technique].apply(weights, models=models, **self.config.parameters)
        
        merged_model = AutoModel.from_pretrained(self.config.models[0].path)
        merged_model.load_state_dict(weights)
        return merged_model

    def _get_model_weights(self, models: List[torch.nn.Module]) -> Dict[str, torch.Tensor]:
        weights = {}
        for key in models[0].state_dict().keys():
            weights[key] = torch.stack([model.state_dict()[key] for model in models])
        return weights

    @staticmethod
    def _apply_ties(weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        threshold = kwargs.get("ties_threshold", 0.1)
        for name, tensor in weights.items():
            if name.endswith('.weight'):
                abs_tensor = torch.abs(tensor)
                mask = abs_tensor > threshold
                weights[name] = torch.where(mask, tensor, torch.zeros_like(tensor))
        return weights

    @staticmethod
    def _apply_dare(weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        threshold = kwargs.get("dare_threshold", 0.1)
        amplification = kwargs.get("dare_amplification", 2.0)
        for name, tensor in weights.items():
            if name.endswith('.weight'):
                abs_diff = torch.abs(tensor)
                mask = abs_diff > threshold
                weights[name] = torch.where(mask, tensor * amplification, torch.zeros_like(tensor))
        return weights

    @staticmethod
    def _task_arithmetic(weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        base_weights = kwargs.get("base_weights", {})
        task_weights = kwargs.get("task_weights", [])
        task_vectors = []
        for task_weight in task_weights:
            task_vector = {name: task_weight[name] - base_weights[name] for name in base_weights}
            task_vectors.append(task_vector)
        combined_task_vector = {name: sum(task_vector[name] for task_vector in task_vectors) for name in base_weights}
        return {name: base_weights[name] + combined_task_vector[name] for name in base_weights}

    @staticmethod
    def _frankenmerge(weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        models = kwargs.get("models", [])
        merged_weights = {}
        for i, (name, tensor) in enumerate(weights.items()):
            layer_num = int(name.split('.')[1]) if '.' in name else -1
            if layer_num == -1 or layer_num % len(models) == i:
                merged_weights[name] = tensor[i]
        return merged_weights

    @staticmethod
    def _create_dfs_model(weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
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

    def _save_merged_model(self, model: torch.nn.Module) -> str:
        merged_model_name = f"merged_{self.config.merge_method}_{'_'.join([m.name for m in self.config.models])}"
        merged_model_path = os.path.join(self.custom_dir, merged_model_name)
        model.save_pretrained(merged_model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.config.models[0].path)
        tokenizer.save_pretrained(merged_model_path)
        return merged_model_path

    def evaluate_model(self, model_path: str) -> float:
        try:
            model = AutoModel.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            prompt = "Hello, are you working?"
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model(**inputs)

            # This is a basic evaluation. Implement a more sophisticated method for real use.
            coherence_score = outputs.last_hidden_state.mean().item()
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
                "slerp": {"weights": [1/3, 1/3, 1/3]},
                "ties": {"threshold": 0.1},
                "dare": {"threshold": 0.1, "amplification": 2.0}
            },
            ps_techniques=techniques[:2],
            dfs_techniques=[techniques[2]]
        )
        merger = AdvancedModelMerger(config)
        merged_model_path = merger.merge()
        merged_models.append(merged_model_path)
        
    return merged_models

# Usage example
initial_models = [
    ModelReference(name="model_a", path="bert-base-uncased"),
    ModelReference(name="model_b", path="roberta-base"),
    ModelReference(name="model_c", path="distilbert-base-uncased")
]

merged_models = create_merged_models(initial_models)
print(f"Created {len(merged_models)} merged models")
for model_path in merged_models:
    print(f"Merged model saved at: {model_path}")