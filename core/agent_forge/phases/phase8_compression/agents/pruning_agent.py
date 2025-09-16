import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import copy
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class PruningConfig:
    """Configuration for pruning strategies."""
    pruning_type: str  # 'magnitude', 'structured', 'gradual', 'lottery_ticket'
    sparsity_target: float
    structured_dim: Optional[int] = None
    gradual_steps: int = 10
    importance_metric: str = 'magnitude'  # 'magnitude', 'gradient', 'fisher'
    
@dataclass
class PruningResult:
    """Results from pruning operation."""
    original_params: int
    pruned_params: int
    compression_ratio: float
    sparsity_achieved: float
    layer_sparsities: Dict[str, float]
    performance_metrics: Dict[str, float]
    
class PruningStrategy(ABC):
    """Abstract base class for pruning strategies."""
    
    @abstractmethod
    def prune_model(self, model: nn.Module, config: PruningConfig) -> PruningResult:
        pass
        
class MagnitudePruning(PruningStrategy):
    """Magnitude-based unstructured pruning."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def prune_model(self, model: nn.Module, config: PruningConfig) -> PruningResult:
        """Apply magnitude-based pruning."""
        original_params = sum(p.numel() for p in model.parameters())
        
        # Apply magnitude pruning to all prunable layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.l1_unstructured(module, name='weight', amount=config.sparsity_target)
                if hasattr(module, 'bias') and module.bias is not None:
                    prune.l1_unstructured(module, name='bias', amount=config.sparsity_target * 0.5)
                    
        # Calculate results
        pruned_params = sum(p.numel() for p in model.parameters())
        layer_sparsities = self._calculate_layer_sparsities(model)
        
        return PruningResult(
            original_params=original_params,
            pruned_params=pruned_params,
            compression_ratio=original_params / pruned_params,
            sparsity_achieved=1 - (pruned_params / original_params),
            layer_sparsities=layer_sparsities,
            performance_metrics={}
        )
        
    def _calculate_layer_sparsities(self, model: nn.Module) -> Dict[str, float]:
        """Calculate sparsity for each layer."""
        sparsities = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight_mask'):
                total_elements = module.weight.numel()
                zero_elements = (module.weight_mask == 0).sum().item()
                sparsities[name] = zero_elements / total_elements
                
        return sparsities
        
class StructuredPruning(PruningStrategy):
    """Structured pruning (removing entire channels/filters)."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def prune_model(self, model: nn.Module, config: PruningConfig) -> PruningResult:
        """Apply structured pruning."""
        original_params = sum(p.numel() for p in model.parameters())
        
        # Apply structured pruning
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels > 1:
                # Prune output channels based on L1 norm
                self._prune_conv_channels(module, config.sparsity_target)
            elif isinstance(module, nn.Linear) and module.out_features > 1:
                # Prune output neurons
                self._prune_linear_neurons(module, config.sparsity_target)
                
        pruned_params = sum(p.numel() for p in model.parameters())
        layer_sparsities = self._calculate_structured_sparsities(model)
        
        return PruningResult(
            original_params=original_params,
            pruned_params=pruned_params,
            compression_ratio=original_params / pruned_params,
            sparsity_achieved=1 - (pruned_params / original_params),
            layer_sparsities=layer_sparsities,
            performance_metrics={}
        )
        
    def _prune_conv_channels(self, conv: nn.Conv2d, sparsity: float) -> None:
        """Prune convolutional channels based on importance."""
        # Calculate channel importance (L1 norm)
        weight = conv.weight.data
        channel_importance = weight.abs().sum(dim=(1, 2, 3))
        
        # Determine channels to prune
        num_channels_to_prune = int(conv.out_channels * sparsity)
        _, indices_to_prune = torch.topk(channel_importance, 
                                       num_channels_to_prune, 
                                       largest=False)
        
        # Create mask
        mask = torch.ones(conv.out_channels, dtype=torch.bool)
        mask[indices_to_prune] = False
        
        # Apply pruning
        prune.custom_from_mask(conv, name='weight', mask=mask.unsqueeze(1).unsqueeze(2).unsqueeze(3))
        
        if conv.bias is not None:
            prune.custom_from_mask(conv, name='bias', mask=mask)
            
    def _prune_linear_neurons(self, linear: nn.Linear, sparsity: float) -> None:
        """Prune linear layer neurons based on importance."""
        # Calculate neuron importance
        weight = linear.weight.data
        neuron_importance = weight.abs().sum(dim=1)
        
        # Determine neurons to prune
        num_neurons_to_prune = int(linear.out_features * sparsity)
        _, indices_to_prune = torch.topk(neuron_importance, 
                                       num_neurons_to_prune, 
                                       largest=False)
        
        # Create mask
        mask = torch.ones(linear.out_features, dtype=torch.bool)
        mask[indices_to_prune] = False
        
        # Apply pruning
        prune.custom_from_mask(linear, name='weight', mask=mask.unsqueeze(1))
        
        if linear.bias is not None:
            prune.custom_from_mask(linear, name='bias', mask=mask)
            
    def _calculate_structured_sparsities(self, model: nn.Module) -> Dict[str, float]:
        """Calculate structured sparsity for each layer."""
        sparsities = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight_mask'):
                if isinstance(module, nn.Conv2d):
                    # Channel-wise sparsity
                    mask = module.weight_mask
                    total_channels = mask.shape[0]
                    zero_channels = (mask.sum(dim=(1, 2, 3)) == 0).sum().item()
                    sparsities[name] = zero_channels / total_channels
                elif isinstance(module, nn.Linear):
                    # Neuron-wise sparsity
                    mask = module.weight_mask
                    total_neurons = mask.shape[0]
                    zero_neurons = (mask.sum(dim=1) == 0).sum().item()
                    sparsities[name] = zero_neurons / total_neurons
                    
        return sparsities
        
class GradualPruning(PruningStrategy):
    """Gradual pruning over multiple training steps."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.current_step = 0
        
    def prune_model(self, model: nn.Module, config: PruningConfig) -> PruningResult:
        """Apply gradual pruning."""
        original_params = sum(p.numel() for p in model.parameters())
        
        # Calculate current sparsity based on step
        current_sparsity = self._calculate_current_sparsity(config)
        
        # Apply magnitude pruning with current sparsity
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.l1_unstructured(module, name='weight', amount=current_sparsity)
                
        self.current_step += 1
        
        pruned_params = sum(p.numel() for p in model.parameters())
        layer_sparsities = self._calculate_layer_sparsities(model)
        
        return PruningResult(
            original_params=original_params,
            pruned_params=pruned_params,
            compression_ratio=original_params / pruned_params,
            sparsity_achieved=current_sparsity,
            layer_sparsities=layer_sparsities,
            performance_metrics={'current_step': self.current_step}
        )
        
    def _calculate_current_sparsity(self, config: PruningConfig) -> float:
        """Calculate sparsity for current step using polynomial schedule."""
        if self.current_step >= config.gradual_steps:
            return config.sparsity_target
            
        # Polynomial sparsity schedule
        progress = self.current_step / config.gradual_steps
        return config.sparsity_target * (progress ** 3)
        
    def _calculate_layer_sparsities(self, model: nn.Module) -> Dict[str, float]:
        """Calculate sparsity for each layer."""
        sparsities = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight_mask'):
                total_elements = module.weight.numel()
                zero_elements = (module.weight_mask == 0).sum().item()
                sparsities[name] = zero_elements / total_elements
                
        return sparsities
        
class LotteryTicketPruning(PruningStrategy):
    """Lottery Ticket Hypothesis implementation."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.initial_weights = None
        
    def prune_model(self, model: nn.Module, config: PruningConfig) -> PruningResult:
        """Apply lottery ticket pruning."""
        # Store initial weights if not done already
        if self.initial_weights is None:
            self.initial_weights = {name: param.clone() 
                                   for name, param in model.named_parameters()}
                                   
        original_params = sum(p.numel() for p in model.parameters())
        
        # Find winning ticket through iterative magnitude pruning
        winning_ticket_mask = self._find_winning_ticket(model, config)
        
        # Apply winning ticket mask
        self._apply_winning_ticket(model, winning_ticket_mask)
        
        pruned_params = sum(p.numel() for p in model.parameters())
        layer_sparsities = self._calculate_ticket_sparsities(model)
        
        return PruningResult(
            original_params=original_params,
            pruned_params=pruned_params,
            compression_ratio=original_params / pruned_params,
            sparsity_achieved=1 - (pruned_params / original_params),
            layer_sparsities=layer_sparsities,
            performance_metrics={'winning_ticket_found': True}
        )
        
    def _find_winning_ticket(self, model: nn.Module, config: PruningConfig) -> Dict[str, torch.Tensor]:
        """Find winning lottery ticket mask."""
        masks = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.data
                
                # Calculate importance scores
                importance = weight.abs()
                
                # Create mask based on top-k importance
                flat_importance = importance.flatten()
                k = int(flat_importance.numel() * (1 - config.sparsity_target))
                threshold_value, _ = torch.topk(flat_importance, k)
                threshold = threshold_value[-1]
                
                mask = (importance >= threshold).float()
                masks[name] = mask
                
        return masks
        
    def _apply_winning_ticket(self, model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
        """Apply winning ticket mask and reset to initial weights."""
        for name, module in model.named_modules():
            if name in masks and isinstance(module, (nn.Linear, nn.Conv2d)):
                # Reset to initial weights
                if f'{name}.weight' in self.initial_weights:
                    module.weight.data = self.initial_weights[f'{name}.weight'].clone()
                    
                # Apply mask
                prune.custom_from_mask(module, name='weight', mask=masks[name])
                
    def _calculate_ticket_sparsities(self, model: nn.Module) -> Dict[str, float]:
        """Calculate sparsity for lottery ticket."""
        sparsities = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight_mask'):
                total_elements = module.weight.numel()
                zero_elements = (module.weight_mask == 0).sum().item()
                sparsities[name] = zero_elements / total_elements
                
        return sparsities
        
class PruningAgent:
    """Main pruning agent coordinating different pruning strategies."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.strategies = {
            'magnitude': MagnitudePruning(logger),
            'structured': StructuredPruning(logger),
            'gradual': GradualPruning(logger),
            'lottery_ticket': LotteryTicketPruning(logger)
        }
        
    def prune_model(self, model: nn.Module, config: PruningConfig) -> PruningResult:
        """Apply pruning using specified strategy."""
        try:
            if config.pruning_type not in self.strategies:
                raise ValueError(f"Unknown pruning type: {config.pruning_type}")
                
            strategy = self.strategies[config.pruning_type]
            result = strategy.prune_model(model, config)
            
            self.logger.info(f"Pruning completed: {result.compression_ratio:.2f}x compression")
            return result
            
        except Exception as e:
            self.logger.error(f"Pruning failed: {e}")
            raise
            
    def remove_pruning(self, model: nn.Module) -> None:
        """Remove all pruning masks and make pruning permanent."""
        try:
            for module in model.modules():
                if hasattr(module, 'weight_mask'):
                    prune.remove(module, 'weight')
                if hasattr(module, 'bias_mask'):
                    prune.remove(module, 'bias')
                    
            self.logger.info("Pruning masks removed, pruning is now permanent")
            
        except Exception as e:
            self.logger.error(f"Failed to remove pruning: {e}")
            raise
            
    def evaluate_pruning_impact(self, 
                               original_model: nn.Module, 
                               pruned_model: nn.Module,
                               test_loader: torch.utils.data.DataLoader,
                               device: torch.device = torch.device('cpu')) -> Dict[str, float]:
        """Evaluate the impact of pruning on model performance."""
        try:
            metrics = {}
            
            # Move models to device
            original_model.to(device)
            pruned_model.to(device)
            
            # Evaluate original model
            original_acc = self._evaluate_accuracy(original_model, test_loader, device)
            
            # Evaluate pruned model
            pruned_acc = self._evaluate_accuracy(pruned_model, test_loader, device)
            
            # Calculate metrics
            metrics['original_accuracy'] = original_acc
            metrics['pruned_accuracy'] = pruned_acc
            metrics['accuracy_drop'] = original_acc - pruned_acc
            metrics['relative_accuracy_drop'] = (original_acc - pruned_acc) / original_acc
            
            # Model size comparison
            original_size = sum(p.numel() for p in original_model.parameters())
            pruned_size = sum(p.numel() for p in pruned_model.parameters())
            
            metrics['compression_ratio'] = original_size / pruned_size
            metrics['size_reduction'] = 1 - (pruned_size / original_size)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise
            
    def _evaluate_accuracy(self, model: nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
        """Evaluate model accuracy on test set."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        return correct / total
        
    def create_pruning_schedule(self, 
                              initial_sparsity: float = 0.1,
                              final_sparsity: float = 0.9,
                              num_steps: int = 10) -> List[PruningConfig]:
        """Create a pruning schedule for gradual pruning."""
        schedule = []
        
        for step in range(num_steps):
            progress = step / (num_steps - 1)
            current_sparsity = initial_sparsity + (final_sparsity - initial_sparsity) * progress
            
            config = PruningConfig(
                pruning_type='gradual',
                sparsity_target=current_sparsity,
                gradual_steps=num_steps,
                importance_metric='magnitude'
            )
            
            schedule.append(config)
            
        return schedule
        
    def get_pruning_statistics(self, model: nn.Module) -> Dict[str, Any]:
        """Get detailed pruning statistics."""
        stats = {
            'total_parameters': 0,
            'pruned_parameters': 0,
            'layer_statistics': {},
            'overall_sparsity': 0.0
        }
        
        total_params = 0
        pruned_params = 0
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                layer_total = module.weight.numel()
                total_params += layer_total
                
                if hasattr(module, 'weight_mask'):
                    layer_pruned = (module.weight_mask == 0).sum().item()
                    pruned_params += layer_pruned
                    
                    stats['layer_statistics'][name] = {
                        'total_params': layer_total,
                        'pruned_params': layer_pruned,
                        'sparsity': layer_pruned / layer_total,
                        'remaining_params': layer_total - layer_pruned
                    }
                else:
                    stats['layer_statistics'][name] = {
                        'total_params': layer_total,
                        'pruned_params': 0,
                        'sparsity': 0.0,
                        'remaining_params': layer_total
                    }
                    
        stats['total_parameters'] = total_params
        stats['pruned_parameters'] = pruned_params
        stats['overall_sparsity'] = pruned_params / total_params if total_params > 0 else 0.0
        
        return stats
