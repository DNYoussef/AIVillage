"""
Implementation of various model merging techniques.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional
import math

class MergeTechniques:
    """Collection of model merging techniques."""

    def __init__(self, device="cuda"):
        self.device = device

    def merge(self, models: List[nn.Module], technique: str, **kwargs) -> nn.Module:
        """Apply specified merge technique."""
        technique_map = {
            'linear': self.linear_merge,
            'slerp': self.slerp_merge,
            'ties': self.ties_merge,
            'dare': self.dare_merge,
            'frankenmerge': self.frankenmerge,
            'dfs': self.dfs_merge
        }

        if technique not in technique_map:
            raise ValueError(f"Unknown merge technique: {technique}")

        return technique_map[technique](models, **kwargs)

    def linear_merge(self, models: List[nn.Module], weights: Optional[List[float]] = None) -> nn.Module:
        """Linear interpolation merge."""
        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        # Normalize weights
        weights = torch.tensor(weights, device=self.device)
        weights = weights / weights.sum()

        # Create merged model
        merged = models[0].__class__()
        merged.to(self.device)

        with torch.no_grad():
            # Merge parameters
            for name, param in merged.named_parameters():
                merged_param = torch.zeros_like(param, device=self.device)

                for model, weight in zip(models, weights):
                    model_param = model.state_dict()[name].to(self.device)
                    merged_param += weight * model_param

                param.data = merged_param

            # Merge buffers
            for name, buffer in merged.named_buffers():
                merged_buffer = torch.zeros_like(buffer, device=self.device)

                for model, weight in zip(models, weights):
                    model_buffer = model.state_dict()[name].to(self.device)
                    merged_buffer += weight * model_buffer

                buffer.data = merged_buffer

        return merged

    def slerp_merge(self, models: List[nn.Module], t: float = 0.5) -> nn.Module:
        """Spherical Linear Interpolation merge."""
        if len(models) != 2:
            raise ValueError("SLERP requires exactly 2 models")

        model1, model2 = models
        merged = model1.__class__()
        merged.to(self.device)

        with torch.no_grad():
            for name, param in merged.named_parameters():
                # Get parameters from both models
                p1 = model1.state_dict()[name].to(self.device).flatten()
                p2 = model2.state_dict()[name].to(self.device).flatten()

                # Compute angle between parameters
                dot_product = torch.dot(p1, p2)
                norm1 = torch.norm(p1)
                norm2 = torch.norm(p2)

                # Avoid division by zero
                if norm1 == 0 or norm2 == 0:
                    # Fall back to linear interpolation
                    merged_param = (1 - t) * p1 + t * p2
                else:
                    cos_theta = dot_product / (norm1 * norm2)
                    cos_theta = torch.clamp(cos_theta, -1, 1)
                    theta = torch.acos(cos_theta)

                    if theta < 1e-6:
                        # Parameters are nearly parallel
                        merged_param = (1 - t) * p1 + t * p2
                    else:
                        # Spherical interpolation
                        sin_theta = torch.sin(theta)
                        w1 = torch.sin((1 - t) * theta) / sin_theta
                        w2 = torch.sin(t * theta) / sin_theta
                        merged_param = w1 * p1 + w2 * p2

                param.data = merged_param.reshape(param.shape)

        return merged

    def ties_merge(self, models: List[nn.Module], threshold: float = 0.7) -> nn.Module:
        """Task Internal Expert Selection merge."""
        merged = models[0].__class__()
        merged.to(self.device)

        with torch.no_grad():
            for name, param in merged.named_parameters():
                # Calculate importance scores for each model
                importances = []
                params = []

                for model in models:
                    model_param = model.state_dict()[name].to(self.device)
                    params.append(model_param)

                    # Importance as absolute magnitude
                    importance = torch.abs(model_param).mean()
                    importances.append(importance)

                importances = torch.stack(importances)

                # Select experts based on importance threshold
                mean_importance = importances.mean()
                expert_mask = importances > (mean_importance * threshold)

                # Merge selected experts
                merged_param = torch.zeros_like(param, device=self.device)
                num_experts = expert_mask.sum()

                if num_experts > 0:
                    for i, (is_expert, model_param) in enumerate(zip(expert_mask, params)):
                        if is_expert:
                            merged_param += model_param / num_experts
                else:
                    # If no experts selected, use average
                    for model_param in params:
                        merged_param += model_param / len(models)

                param.data = merged_param

        return merged

    def dare_merge(self, models: List[nn.Module], drop_rate: float = 0.5) -> nn.Module:
        """Drop And REscale merge."""
        merged = models[0].__class__()
        merged.to(self.device)

        with torch.no_grad():
            for name, param in merged.named_parameters():
                # Create dropout mask
                mask = torch.bernoulli(
                    torch.ones_like(param, device=self.device) * (1 - drop_rate)
                )

                # Merge with dropout and rescaling
                merged_param = torch.zeros_like(param, device=self.device)

                for model in models:
                    model_param = model.state_dict()[name].to(self.device)
                    # Apply mask and rescale
                    masked_param = model_param * mask / (1 - drop_rate)
                    merged_param += masked_param / len(models)

                param.data = merged_param

        return merged

    def frankenmerge(self, models: List[nn.Module], layer_assignment: Optional[List[int]] = None) -> nn.Module:
        """Layer-wise selection from different models."""
        merged = models[0].__class__()
        merged.to(self.device)

        # Get all parameter names
        param_names = list(merged.named_parameters())

        # Generate random layer assignment if not provided
        if layer_assignment is None:
            layer_assignment = np.random.choice(
                len(models),
                size=len(param_names)
            )

        with torch.no_grad():
            for i, (name, param) in enumerate(param_names):
                # Select source model for this layer
                source_idx = layer_assignment[i]
                source_model = models[source_idx]

                # Copy parameter from source model
                source_param = source_model.state_dict()[name].to(self.device)
                merged.state_dict()[name].data = source_param.clone()

        return merged

    def dfs_merge(self, models: List[nn.Module], feature_importance: Optional[Dict[str, float]] = None) -> nn.Module:
        """Deep Feature Selection merge."""
        merged = models[0].__class__()
        merged.to(self.device)

        # Calculate feature importance if not provided
        if feature_importance is None:
            feature_importance = self._calculate_feature_importance(models)

        with torch.no_grad():
            for name, param in merged.named_parameters():
                # Get importance score for this parameter
                importance = feature_importance.get(name, 1.0)

                # Weighted merge based on importance
                merged_param = torch.zeros_like(param, device=self.device)

                for model in models:
                    model_param = model.state_dict()[name].to(self.device)

                    # Apply importance weighting
                    if 'weight' in name:
                        # For weight matrices, apply column-wise importance
                        if len(model_param.shape) >= 2:
                            importance_mask = torch.ones_like(model_param)
                            importance_mask *= importance
                            weighted_param = model_param * importance_mask
                        else:
                            weighted_param = model_param * importance
                    else:
                        # For biases and other parameters
                        weighted_param = model_param * importance

                    merged_param += weighted_param / len(models)

                param.data = merged_param

        return merged

    def _calculate_feature_importance(self, models: List[nn.Module]) -> Dict[str, float]:
        """Calculate feature importance scores for each parameter."""
        importance = {}

        for name, param in models[0].named_parameters():
            # Calculate variance across models
            params = []
            for model in models:
                model_param = model.state_dict()[name].to(self.device)
                params.append(model_param.flatten())

            params = torch.stack(params)

            # Importance as inverse of variance (stable parameters are important)
            variance = params.var(dim=0).mean()
            importance[name] = 1.0 / (1.0 + variance.item())

        # Normalize importance scores
        max_importance = max(importance.values())
        if max_importance > 0:
            importance = {k: v / max_importance for k, v in importance.items()}

        return importance

    def adaptive_merge(self, models: List[nn.Module], fitness_scores: List[float]) -> nn.Module:
        """Adaptive merging based on fitness scores."""
        # Normalize fitness scores to weights
        fitness_tensor = torch.tensor(fitness_scores, device=self.device)
        weights = torch.softmax(fitness_tensor, dim=0)

        # Use weighted linear merge
        return self.linear_merge(models, weights.tolist())