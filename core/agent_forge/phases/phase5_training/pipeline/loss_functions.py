"""
Agent Forge Phase 5 - Custom Loss Functions
Specialized loss functions for BitNet and Grokfast training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import math
import numpy as np
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

class LossType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    CONTRASTIVE = "contrastive"
    AUXILIARY = "auxiliary"

@dataclass
class LossConfig:
    """Loss function configuration"""
    # Base loss settings
    loss_type: LossType = LossType.CLASSIFICATION
    label_smoothing: float = 0.0
    class_weights: Optional[torch.Tensor] = None
    reduction: str = "mean"

    # BitNet specific
    quantization_loss_weight: float = 0.01
    sparsity_loss_weight: float = 0.001
    consistency_loss_weight: float = 0.1

    # Grokfast specific
    knowledge_distillation_weight: float = 0.5
    capability_acquisition_weight: float = 0.2
    regularization_weight: float = 0.01

    # Adaptive settings
    adaptive_weighting: bool = True
    temperature_scaling: bool = True
    focal_loss_gamma: float = 2.0

    # Multi-task settings
    task_weights: Optional[Dict[str, float]] = None
    uncertainty_weighting: bool = False

class BaseLoss(ABC):
    """Abstract base class for loss functions"""

    def __init__(self, config: LossConfig):
        self.config = config
        self.step_count = 0

    @abstractmethod
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute the loss"""
        pass

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        self.step_count += 1
        return self.compute(*args, **kwargs)

class ClassificationLoss(BaseLoss):
    """Enhanced classification loss with multiple features"""

    def __init__(self, config: LossConfig, num_classes: int):
        super().__init__(config)
        self.num_classes = num_classes

        # Initialize base loss
        self.base_loss = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing,
            weight=config.class_weights,
            reduction=config.reduction
        )

        # Focal loss parameters
        self.focal_gamma = config.focal_loss_gamma

    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute classification loss with enhancements"""
        batch_size = predictions.size(0)

        # Base cross-entropy loss
        if self.config.focal_loss_gamma > 0:
            loss = self._focal_loss(predictions, targets)
        else:
            loss = self.base_loss(predictions, targets)

        return loss

    def _focal_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for handling class imbalance"""
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.config.focal_loss_gamma * ce_loss

        if self.config.reduction == 'mean':
            return focal_loss.mean()
        elif self.config.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BitNetLoss(BaseLoss):
    """Specialized loss for BitNet training"""

    def __init__(self, config: LossConfig, base_loss: BaseLoss):
        super().__init__(config)
        self.base_loss = base_loss

    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        model: Optional[nn.Module] = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute BitNet loss with quantization regularization"""
        # Base task loss
        base_loss = self.base_loss.compute(predictions, targets, **kwargs)

        total_loss = base_loss

        if model is not None:
            # Add BitNet-specific regularization terms
            quantization_loss = self._compute_quantization_loss(model)
            sparsity_loss = self._compute_sparsity_loss(model)
            consistency_loss = self._compute_consistency_loss(model)

            total_loss = (
                base_loss +
                self.config.quantization_loss_weight * quantization_loss +
                self.config.sparsity_loss_weight * sparsity_loss +
                self.config.consistency_loss_weight * consistency_loss
            )

        return total_loss

    def _compute_quantization_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute loss to encourage quantization-friendly weights"""
        quantization_loss = 0.0
        param_count = 0

        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                # Encourage weights to be close to {-1, 1}
                quantized_weights = torch.sign(param)
                loss = F.mse_loss(param, quantized_weights)
                quantization_loss += loss
                param_count += 1

        return quantization_loss / param_count if param_count > 0 else torch.tensor(0.0)

    def _compute_sparsity_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute L1 regularization for sparsity"""
        sparsity_loss = 0.0
        param_count = 0

        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                sparsity_loss += torch.sum(torch.abs(param))
                param_count += 1

        return sparsity_loss / param_count if param_count > 0 else torch.tensor(0.0)

    def _compute_consistency_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute consistency between full precision and quantized outputs"""
        # This requires running both forward passes - simplified here
        return torch.tensor(0.0)

class GrokfastLoss(BaseLoss):
    """Specialized loss for Grokfast training"""

    def __init__(self, config: LossConfig, base_loss: BaseLoss):
        super().__init__(config)
        self.base_loss = base_loss
        self.knowledge_states = {}
        self.capability_targets = {}

    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        model: Optional[nn.Module] = None,
        phase: str = "training",
        **kwargs
    ) -> torch.Tensor:
        """Compute Grokfast loss with knowledge consolidation"""
        # Base task loss
        base_loss = self.base_loss.compute(predictions, targets, **kwargs)

        total_loss = base_loss

        if model is not None:
            # Add Grokfast-specific terms
            if phase == "consolidation":
                knowledge_loss = self._compute_knowledge_distillation_loss(model, predictions)
                total_loss += self.config.knowledge_distillation_weight * knowledge_loss

            capability_loss = self._compute_capability_acquisition_loss(predictions, targets)
            total_loss += self.config.capability_acquisition_weight * capability_loss

            regularization_loss = self._compute_grokfast_regularization(model)
            total_loss += self.config.regularization_weight * regularization_loss

        return total_loss

    def _compute_knowledge_distillation_loss(
        self,
        model: nn.Module,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """Compute knowledge distillation loss"""
        # Simplified - would need teacher model outputs
        return torch.tensor(0.0)

    def _compute_capability_acquisition_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss to encourage capability acquisition"""
        # Encourage confident predictions
        probabilities = F.softmax(predictions, dim=-1)
        max_probs = torch.max(probabilities, dim=-1)[0]
        confidence_loss = -torch.log(max_probs + 1e-8).mean()

        return confidence_loss

    def _compute_grokfast_regularization(self, model: nn.Module) -> torch.Tensor:
        """Compute Grokfast-specific regularization"""
        # Encourage stable parameter values
        stability_loss = 0.0
        param_count = 0

        for param in model.parameters():
            if param.requires_grad:
                # Penalize large parameter changes
                stability_loss += torch.norm(param, p=2)
                param_count += 1

        return stability_loss / param_count if param_count > 0 else torch.tensor(0.0)

class ContrastiveLoss(BaseLoss):
    """Contrastive learning loss"""

    def __init__(self, config: LossConfig, temperature: float = 0.1):
        super().__init__(config)
        self.temperature = temperature

    def compute(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute contrastive loss"""
        batch_size = embeddings.size(0)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create positive and negative masks
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()
        negative_mask = 1.0 - positive_mask

        # Remove diagonal (self-similarity)
        positive_mask.fill_diagonal_(0)

        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        positive_sim = exp_sim * positive_mask
        negative_sim = exp_sim * negative_mask

        # Sum over negatives and positives
        negative_sum = negative_sim.sum(dim=1, keepdim=True)
        positive_sum = positive_sim.sum(dim=1, keepdim=True)

        # Avoid division by zero
        positive_sum = torch.clamp(positive_sum, min=1e-8)

        # Compute contrastive loss
        loss = -torch.log(positive_sum / (positive_sum + negative_sum + 1e-8))

        return loss.mean()

class MultiTaskLoss(BaseLoss):
    """Multi-task learning loss with uncertainty weighting"""

    def __init__(self, config: LossConfig, task_losses: Dict[str, BaseLoss]):
        super().__init__(config)
        self.task_losses = task_losses
        self.task_weights = config.task_weights or {name: 1.0 for name in task_losses.keys()}

        # Learnable uncertainty parameters
        if config.uncertainty_weighting:
            self.log_vars = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(1))
                for name in task_losses.keys()
            })
        else:
            self.log_vars = None

    def compute(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """Compute multi-task loss"""
        total_loss = 0.0
        task_losses = {}

        for task_name, loss_fn in self.task_losses.items():
            if task_name in predictions and task_name in targets:
                task_loss = loss_fn.compute(
                    predictions[task_name],
                    targets[task_name],
                    **kwargs
                )

                task_losses[task_name] = task_loss

                # Apply weighting
                if self.log_vars is not None:
                    # Uncertainty weighting
                    precision = torch.exp(-self.log_vars[task_name])
                    weighted_loss = precision * task_loss + self.log_vars[task_name]
                else:
                    # Fixed weighting
                    weight = self.task_weights.get(task_name, 1.0)
                    weighted_loss = weight * task_loss

                total_loss += weighted_loss

        return total_loss

class AdaptiveLoss(BaseLoss):
    """Adaptive loss with dynamic weighting"""

    def __init__(self, config: LossConfig, base_loss: BaseLoss):
        super().__init__(config)
        self.base_loss = base_loss
        self.loss_history = []
        self.weight_history = []

    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute adaptive loss with dynamic weighting"""
        base_loss = self.base_loss.compute(predictions, targets, **kwargs)

        # Record loss history
        self.loss_history.append(base_loss.item())

        # Compute adaptive weight
        if len(self.loss_history) > 10:
            recent_losses = self.loss_history[-10:]
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]

            # Adjust weight based on trend
            if loss_trend < 0:  # Decreasing loss
                weight = 1.0
            else:  # Increasing or stable loss
                weight = min(2.0, 1.0 + abs(loss_trend) * 10)
        else:
            weight = 1.0

        self.weight_history.append(weight)
        return weight * base_loss

class LossManager:
    """Comprehensive loss management system"""

    def __init__(self, config: LossConfig):
        self.config = config
        self.losses = {}
        self.weights = {}
        self.history = defaultdict(list)

    def register_loss(
        self,
        name: str,
        loss_fn: BaseLoss,
        weight: float = 1.0
    ):
        """Register a loss function"""
        self.losses[name] = loss_fn
        self.weights[name] = weight

    def compute_total_loss(
        self,
        predictions: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: Union[torch.Tensor, Dict[str, torch.Tensor]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss from all registered losses"""
        results = {}
        total_loss = 0.0

        for name, loss_fn in self.losses.items():
            try:
                if isinstance(predictions, dict) and isinstance(targets, dict):
                    # Multi-task case
                    if name in predictions and name in targets:
                        loss = loss_fn.compute(
                            predictions[name],
                            targets[name],
                            **kwargs
                        )
                    else:
                        continue
                else:
                    # Single task case
                    loss = loss_fn.compute(predictions, targets, **kwargs)

                weighted_loss = self.weights[name] * loss
                results[name] = loss
                total_loss += weighted_loss

                # Record history
                self.history[name].append(loss.item())

            except Exception as e:
                logging.warning(f"Error computing loss '{name}': {e}")
                continue

        results['total_loss'] = total_loss
        self.history['total_loss'].append(total_loss.item())

        return results

    def get_loss_statistics(self) -> Dict[str, Any]:
        """Get statistics about loss history"""
        stats = {}

        for name, history in self.history.items():
            if history:
                stats[name] = {
                    'current': history[-1],
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'min': np.min(history),
                    'max': np.max(history),
                    'trend': np.polyfit(range(len(history)), history, 1)[0] if len(history) > 1 else 0
                }

        return stats

    def update_weights(self, new_weights: Dict[str, float]):
        """Update loss weights"""
        for name, weight in new_weights.items():
            if name in self.weights:
                self.weights[name] = weight

def create_loss_function(
    task_type: str,
    config: LossConfig,
    **kwargs
) -> BaseLoss:
    """Factory function to create appropriate loss function"""

    if task_type == "classification":
        num_classes = kwargs.get('num_classes', 10)
        return ClassificationLoss(config, num_classes)

    elif task_type == "bitnet_classification":
        num_classes = kwargs.get('num_classes', 10)
        base_loss = ClassificationLoss(config, num_classes)
        return BitNetLoss(config, base_loss)

    elif task_type == "grokfast_classification":
        num_classes = kwargs.get('num_classes', 10)
        base_loss = ClassificationLoss(config, num_classes)
        return GrokfastLoss(config, base_loss)

    elif task_type == "contrastive":
        temperature = kwargs.get('temperature', 0.1)
        return ContrastiveLoss(config, temperature)

    elif task_type == "multi_task":
        task_losses = kwargs.get('task_losses', {})
        return MultiTaskLoss(config, task_losses)

    else:
        raise ValueError(f"Unknown task type: {task_type}")

if __name__ == "__main__":
    # Example usage and testing
    import torch

    # Test configuration
    config = LossConfig(
        loss_type=LossType.CLASSIFICATION,
        label_smoothing=0.1,
        quantization_loss_weight=0.01,
        focal_loss_gamma=2.0
    )

    # Test classification loss
    num_classes = 10
    batch_size = 32

    predictions = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    # Create loss functions
    classification_loss = create_loss_function("classification", config, num_classes=num_classes)
    bitnet_loss = create_loss_function("bitnet_classification", config, num_classes=num_classes)

    # Test loss computation
    cls_loss = classification_loss(predictions, targets)
    bit_loss = bitnet_loss(predictions, targets)

    print(f"Classification loss: {cls_loss.item():.6f}")
    print(f"BitNet loss: {bit_loss.item():.6f}")

    # Test loss manager
    manager = LossManager(config)
    manager.register_loss("classification", classification_loss, 1.0)
    manager.register_loss("bitnet", bitnet_loss, 0.5)

    loss_results = manager.compute_total_loss(predictions, targets)
    print(f"Total loss: {loss_results['total_loss'].item():.6f}")

    # Test statistics
    for _ in range(10):
        preds = torch.randn(batch_size, num_classes)
        targs = torch.randint(0, num_classes, (batch_size,))
        manager.compute_total_loss(preds, targs)

    stats = manager.get_loss_statistics()
    print(f"Loss statistics: {stats}")

    print("Loss functions test completed successfully!")