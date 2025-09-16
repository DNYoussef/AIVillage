"""
Advanced Neural Network Pruning Algorithms
Implements state-of-the-art pruning techniques for model compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PruningConfig:
    """Configuration for pruning algorithms."""
    sparsity_ratio: float = 0.5
    structured: bool = False
    global_pruning: bool = True
    importance_score: str = "magnitude"  # magnitude, gradient, fisher, snip, grasp
    recovery_iterations: int = 0

class BasePruner(ABC):
    """Base class for all pruning algorithms."""

    def __init__(self, config: PruningConfig):
        self.config = config
        self.importance_scores = {}
        self.masks = {}

    @abstractmethod
    def compute_importance_scores(self, model: nn.Module,
                                dataloader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        """Compute importance scores for parameters."""
        pass

    @abstractmethod
    def generate_masks(self, model: nn.Module,
                      importance_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate pruning masks based on importance scores."""
        pass

    def apply_masks(self, model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
        """Apply pruning masks to model parameters."""
        for name, param in model.named_parameters():
            if name in masks:
                param.data.mul_(masks[name])

    def prune_model(self, model: nn.Module,
                   dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Complete pruning pipeline."""
        logger.info(f"Starting pruning with {self.__class__.__name__}")

        # Compute importance scores
        self.importance_scores = self.compute_importance_scores(model, dataloader)

        # Generate masks
        self.masks = self.generate_masks(model, self.importance_scores)

        # Apply pruning
        self.apply_masks(model, self.masks)

        # Calculate statistics
        stats = self.calculate_pruning_stats(model)

        logger.info(f"Pruning completed. Sparsity: {stats['overall_sparsity']:.2%}")
        return stats

    def calculate_pruning_stats(self, model: nn.Module) -> Dict[str, float]:
        """Calculate pruning statistics."""
        total_params = 0
        pruned_params = 0
        layer_stats = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += param.numel()
                pruned = (param == 0).sum().item()
                pruned_params += pruned
                layer_stats[name] = pruned / param.numel()

        return {
            'overall_sparsity': pruned_params / total_params,
            'layer_sparsity': layer_stats,
            'total_parameters': total_params,
            'pruned_parameters': pruned_params
        }

class MagnitudePruner(BasePruner):
    """Magnitude-based pruning - removes weights with smallest absolute values."""

    def compute_importance_scores(self, model: nn.Module,
                                dataloader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        """Compute magnitude-based importance scores."""
        scores = {}
        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) > 1:  # Skip biases
                scores[name] = torch.abs(param.data)
        return scores

    def generate_masks(self, model: nn.Module,
                      importance_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate masks based on magnitude scores."""
        masks = {}

        if self.config.global_pruning:
            # Global pruning - consider all parameters together
            all_scores = torch.cat([scores.view(-1) for scores in importance_scores.values()])
            threshold = torch.quantile(all_scores, self.config.sparsity_ratio)

            for name, scores in importance_scores.items():
                masks[name] = (scores >= threshold).float()
        else:
            # Layer-wise pruning
            for name, scores in importance_scores.items():
                threshold = torch.quantile(scores.view(-1), self.config.sparsity_ratio)
                masks[name] = (scores >= threshold).float()

        return masks

class GradientBasedPruner(BasePruner):
    """Gradient-based pruning using parameter gradients as importance."""

    def compute_importance_scores(self, model: nn.Module,
                                dataloader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        """Compute gradient-based importance scores."""
        model.train()
        scores = {name: torch.zeros_like(param) for name, param in model.named_parameters()
                 if param.requires_grad and len(param.shape) > 1}

        sample_count = 0
        for batch in dataloader:
            if sample_count >= 1000:  # Limit samples for efficiency
                break

            inputs, targets = batch
            model.zero_grad()

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()

            # Accumulate gradient magnitudes
            for name, param in model.named_parameters():
                if name in scores and param.grad is not None:
                    scores[name] += torch.abs(param.grad.data)

            sample_count += inputs.size(0)

        # Average over samples
        for name in scores:
            scores[name] /= sample_count

        return scores

    def generate_masks(self, model: nn.Module,
                      importance_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.torch.Tensor]:
        """Generate masks based on gradient scores."""
        return MagnitudePruner.generate_masks(self, model, importance_scores)

class SNIPPruner(BasePruner):
    """SNIP: Single-shot Network Pruning based on Connection Sensitivity."""

    def compute_importance_scores(self, model: nn.Module,
                                dataloader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        """Compute SNIP sensitivity scores."""
        model.train()

        # Get a single batch for sensitivity computation
        inputs, targets = next(iter(dataloader))

        # Forward pass
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)

        # Compute gradients
        gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        # Compute connection sensitivity
        scores = {}
        for (name, param), grad in zip(model.named_parameters(), gradients):
            if param.requires_grad and len(param.shape) > 1:
                scores[name] = torch.abs(param * grad)

        return scores

    def generate_masks(self, model: nn.Module,
                      importance_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate masks based on SNIP scores."""
        return MagnitudePruner.generate_masks(self, model, importance_scores)

class GraSPPruner(BasePruner):
    """GraSP: Gradient Signal Preservation for pruning."""

    def compute_importance_scores(self, model: nn.Module,
                                dataloader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        """Compute GraSP importance scores."""
        model.train()

        # Compute gradient flow preservation scores
        scores = {name: torch.zeros_like(param) for name, param in model.named_parameters()
                 if param.requires_grad and len(param.shape) > 1}

        sample_count = 0
        for batch in dataloader:
            if sample_count >= 500:  # Limit for efficiency
                break

            inputs, targets = batch

            # First forward-backward pass
            model.zero_grad()
            outputs1 = model(inputs)
            loss1 = F.cross_entropy(outputs1, targets)
            grad1 = torch.autograd.grad(loss1, model.parameters(), retain_graph=True)

            # Second forward-backward pass with different targets
            targets_shuffled = targets[torch.randperm(targets.size(0))]
            outputs2 = model(inputs)
            loss2 = F.cross_entropy(outputs2, targets_shuffled)
            grad2 = torch.autograd.grad(loss2, model.parameters())

            # Compute gradient correlation
            for (name, param), g1, g2 in zip(model.named_parameters(), grad1, grad2):
                if name in scores:
                    scores[name] += torch.abs(g1 * g2)

            sample_count += inputs.size(0)

        # Normalize scores
        for name in scores:
            scores[name] /= sample_count

        return scores

    def generate_masks(self, model: nn.Module,
                      importance_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate masks based on GraSP scores."""
        return MagnitudePruner.generate_masks(self, model, importance_scores)

class StructuredPruner(BasePruner):
    """Structured pruning - removes entire channels, filters, or layers."""

    def __init__(self, config: PruningConfig):
        super().__init__(config)
        self.config.structured = True

    def compute_importance_scores(self, model: nn.Module,
                                dataloader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        """Compute channel/filter importance scores."""
        scores = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # For conv layers, compute filter importance
                weight = module.weight.data  # [out_channels, in_channels, H, W]
                # L2 norm of each output filter
                filter_scores = torch.norm(weight.view(weight.size(0), -1), dim=1)
                scores[name] = filter_scores
            elif isinstance(module, nn.Linear):
                # For linear layers, compute neuron importance
                weight = module.weight.data  # [out_features, in_features]
                neuron_scores = torch.norm(weight, dim=1)
                scores[name] = neuron_scores

        return scores

    def generate_masks(self, model: nn.Module,
                      importance_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate structured masks."""
        masks = {}

        for name, scores in importance_scores.items():
            # Determine number of structures to keep
            num_structures = len(scores)
            num_keep = int(num_structures * (1 - self.config.sparsity_ratio))

            # Select top-k structures
            _, top_indices = torch.topk(scores, num_keep)

            # Create structured mask
            structure_mask = torch.zeros(num_structures)
            structure_mask[top_indices] = 1.0

            # Convert to parameter mask
            module = dict(model.named_modules())[name]
            if isinstance(module, nn.Conv2d):
                param_mask = structure_mask.view(-1, 1, 1, 1).expand_as(module.weight)
            elif isinstance(module, nn.Linear):
                param_mask = structure_mask.view(-1, 1).expand_as(module.weight)

            masks[f"{name}.weight"] = param_mask

        return masks

class AdaptivePruner(BasePruner):
    """Adaptive pruning with gradual sparsity increase."""

    def __init__(self, config: PruningConfig, initial_sparsity: float = 0.0,
                 final_sparsity: float = None, pruning_steps: int = 10):
        super().__init__(config)
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity or config.sparsity_ratio
        self.pruning_steps = pruning_steps
        self.current_step = 0

        # Use magnitude pruning as base
        self.base_pruner = MagnitudePruner(config)

    def get_current_sparsity(self) -> float:
        """Calculate current sparsity based on schedule."""
        if self.current_step >= self.pruning_steps:
            return self.final_sparsity

        # Cubic sparsity schedule
        progress = self.current_step / self.pruning_steps
        sparsity = self.initial_sparsity + (self.final_sparsity - self.initial_sparsity) * (progress ** 3)
        return sparsity

    def adaptive_prune_step(self, model: nn.Module,
                          dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Perform one adaptive pruning step."""
        current_sparsity = self.get_current_sparsity()
        self.config.sparsity_ratio = current_sparsity

        stats = self.base_pruner.prune_model(model, dataloader)
        self.current_step += 1

        return stats

    def compute_importance_scores(self, model: nn.Module,
                                dataloader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        """Delegate to base pruner."""
        return self.base_pruner.compute_importance_scores(model, dataloader)

    def generate_masks(self, model: nn.Module,
                      importance_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Delegate to base pruner."""
        return self.base_pruner.generate_masks(model, importance_scores)

class PruningOrchestrator:
    """Orchestrates different pruning algorithms and recovery procedures."""

    def __init__(self):
        self.pruners = {
            'magnitude': MagnitudePruner,
            'gradient': GradientBasedPruner,
            'snip': SNIPPruner,
            'grasp': GraSPPruner,
            'structured': StructuredPruner,
            'adaptive': AdaptivePruner
        }

    def create_pruner(self, algorithm: str, config: PruningConfig) -> BasePruner:
        """Create pruner instance."""
        if algorithm not in self.pruners:
            raise ValueError(f"Unknown pruning algorithm: {algorithm}")

        return self.pruners[algorithm](config)

    def compare_pruning_methods(self, model: nn.Module,
                              dataloader: torch.utils.data.DataLoader,
                              algorithms: List[str],
                              config: PruningConfig) -> Dict[str, Dict[str, float]]:
        """Compare multiple pruning algorithms."""
        results = {}

        for algorithm in algorithms:
            logger.info(f"Testing {algorithm} pruning...")

            # Create copy of model for testing
            model_copy = torch.nn.utils.prune.identity(model)

            # Create and apply pruner
            pruner = self.create_pruner(algorithm, config)
            stats = pruner.prune_model(model_copy, dataloader)

            results[algorithm] = stats

        return results

    def progressive_pruning(self, model: nn.Module,
                          dataloader: torch.utils.data.DataLoader,
                          trainer_fn, target_sparsity: float = 0.9,
                          steps: int = 5) -> Dict[str, float]:
        """Perform progressive pruning with retraining."""
        config = PruningConfig(
            sparsity_ratio=0.0,  # Start with 0
            global_pruning=True
        )

        pruner = AdaptivePruner(
            config,
            initial_sparsity=0.0,
            final_sparsity=target_sparsity,
            pruning_steps=steps
        )

        results = []

        for step in range(steps):
            logger.info(f"Progressive pruning step {step + 1}/{steps}")

            # Prune
            stats = pruner.adaptive_prune_step(model, dataloader)

            # Retrain
            if trainer_fn:
                trainer_fn(model, epochs=2)  # Light retraining

            results.append(stats)

        return {
            'final_sparsity': results[-1]['overall_sparsity'],
            'steps': results
        }

# Factory function for easy usage
def create_pruner(algorithm: str = "magnitude",
                 sparsity_ratio: float = 0.5,
                 structured: bool = False,
                 global_pruning: bool = True) -> BasePruner:
    """Factory function to create pruners."""
    config = PruningConfig(
        sparsity_ratio=sparsity_ratio,
        structured=structured,
        global_pruning=global_pruning
    )

    orchestrator = PruningOrchestrator()
    return orchestrator.create_pruner(algorithm, config)

if __name__ == "__main__":
    # Example usage
    import torchvision.models as models

    # Create a sample model
    model = models.resnet18(pretrained=False)

    # Create sample data loader (placeholder)
    from torch.utils.data import DataLoader, TensorDataset
    dummy_data = TensorDataset(
        torch.randn(100, 3, 224, 224),
        torch.randint(0, 1000, (100,))
    )
    dataloader = DataLoader(dummy_data, batch_size=32)

    # Test different pruning methods
    orchestrator = PruningOrchestrator()
    config = PruningConfig(sparsity_ratio=0.5)

    results = orchestrator.compare_pruning_methods(
        model, dataloader,
        ['magnitude', 'gradient', 'snip'],
        config
    )

    for algorithm, stats in results.items():
        print(f"{algorithm}: {stats['overall_sparsity']:.2%} sparsity")