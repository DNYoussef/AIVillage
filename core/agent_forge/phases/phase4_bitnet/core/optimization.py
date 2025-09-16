#!/usr/bin/env python3
"""
BitNet Phase 4 - Optimization Components

Optimization algorithms and utilities specifically designed for BitNet models.
Includes custom optimizers, learning rate schedulers, and training utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from typing import Iterator, Dict, Any, Optional, Union, List
import math
import logging

class BitNetOptimizer:
    """Custom optimizer for BitNet training"""
    
    def __init__(self, learning_rate: float = 1e-4, weight_decay: float = 1e-5, 
                 eps: float = 1e-8, betas: tuple = (0.9, 0.999)):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eps = eps
        self.betas = betas
        self.logger = logging.getLogger(__name__)
        
    def create_optimizer(self, parameters: Iterator[nn.Parameter]) -> optim.Optimizer:
        """Create Adam optimizer with BitNet-specific settings"""
        return optim.AdamW(
            parameters,
            lr=self.learning_rate,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay
        )
        
    def create_scheduler(self, optimizer: optim.Optimizer, 
                        num_warmup_steps: int = 1000,
                        num_training_steps: int = 10000) -> _LRScheduler:
        """Create learning rate scheduler"""
        return BitNetLRScheduler(
            optimizer, 
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state"""
        return {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'eps': self.eps,
            'betas': self.betas
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state"""
        self.learning_rate = state_dict.get('learning_rate', self.learning_rate)
        self.weight_decay = state_dict.get('weight_decay', self.weight_decay)
        self.eps = state_dict.get('eps', self.eps)
        self.betas = state_dict.get('betas', self.betas)

class BitNetLRScheduler(_LRScheduler):
    """Custom learning rate scheduler for BitNet"""
    
    def __init__(self, optimizer: optim.Optimizer, num_warmup_steps: int = 1000, 
                 num_training_steps: int = 10000, last_epoch: int = -1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """Calculate learning rate with warmup and cosine decay"""
        if self.last_epoch < self.num_warmup_steps:
            # Linear warmup
            lr_scale = float(self.last_epoch) / float(max(1, self.num_warmup_steps))
        else:
            # Cosine decay
            progress = float(self.last_epoch - self.num_warmup_steps) / float(
                max(1, self.num_training_steps - self.num_warmup_steps)
            )
            lr_scale = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
        return [base_lr * lr_scale for base_lr in self.base_lrs]

class BitNetTrainer:
    """Training utilities for BitNet models"""
    
    def __init__(self, model: nn.Module, optimizer_config: Dict[str, Any] = None):
        self.model = model
        self.optimizer_config = optimizer_config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize optimizer
        self.bitnet_optimizer = BitNetOptimizer(**self.optimizer_config)
        self.optimizer = self.bitnet_optimizer.create_optimizer(self.model.parameters())
        self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
    def setup_scheduler(self, num_warmup_steps: int = 1000, num_training_steps: int = 10000):
        """Setup learning rate scheduler"""
        self.scheduler = self.bitnet_optimizer.create_scheduler(
            self.optimizer, num_warmup_steps, num_training_steps
        )
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        input_ids = batch['input_ids']
        labels = batch.get('labels', input_ids)
        attention_mask = batch.get('attention_mask')
        
        logits = self.model(input_ids, attention_mask)
        
        # Calculate loss
        loss = self.calculate_loss(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
            
        self.global_step += 1
        
        return {
            'loss': loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'global_step': self.global_step
        }
        
    def calculate_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculate training loss"""
        # Shift logits and labels for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        return loss
        
    def evaluate(self, eval_dataloader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch['input_ids']
                labels = batch.get('labels', input_ids)
                attention_mask = batch.get('attention_mask')
                
                logits = self.model(input_ids, attention_mask)
                loss = self.calculate_loss(logits, labels)
                
                total_loss += loss.item()
                total_steps += 1
                
        avg_loss = total_loss / total_steps if total_steps > 0 else float('inf')
        perplexity = math.exp(min(avg_loss, 20))  # Cap to prevent overflow
        
        return {
            'eval_loss': avg_loss,
            'perplexity': perplexity,
            'eval_steps': total_steps
        }
        
    def save_checkpoint(self, filepath: str, include_optimizer: bool = True):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'optimizer_config': self.optimizer_config
        }
        
        if include_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            if self.scheduler:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
                
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
        
    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if load_optimizer and self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.logger.info(f"Checkpoint loaded from {filepath}")
        
    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state"""
        return {
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'optimizer_config': self.optimizer_config
        }

class BitNetMemoryOptimizer:
    """Memory optimization utilities for BitNet"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.logger = logging.getLogger(__name__)
        
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory"""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        else:
            # Manual gradient checkpointing for transformer blocks
            for module in self.model.modules():
                if hasattr(module, 'transformer_blocks'):
                    for block in module.transformer_blocks:
                        block = torch.utils.checkpoint.checkpoint_wrapper(block)
                        
    def optimize_memory_usage(self, enable_mixed_precision: bool = True,
                            enable_gradient_checkpointing: bool = True):
        """Apply memory optimizations"""
        optimizations_applied = []
        
        if enable_gradient_checkpointing:
            self.enable_gradient_checkpointing()
            optimizations_applied.append('gradient_checkpointing')
            
        if enable_mixed_precision and torch.cuda.is_available():
            # This would typically be handled by a GradScaler in training loop
            optimizations_applied.append('mixed_precision_ready')
            
        self.logger.info(f"Applied memory optimizations: {optimizations_applied}")
        return optimizations_applied
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        stats = {}
        
        if torch.cuda.is_available():
            stats.update({
                'cuda_memory_allocated': torch.cuda.memory_allocated() / (1024**3),  # GB
                'cuda_memory_reserved': torch.cuda.memory_reserved() / (1024**3),    # GB
                'cuda_max_memory_allocated': torch.cuda.max_memory_allocated() / (1024**3),  # GB
            })
            
        # Model parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        stats.update({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024**2),  # Assuming fp32
        })
        
        return stats

def create_bitnet_trainer(model: nn.Module, learning_rate: float = 1e-4, 
                         weight_decay: float = 1e-5) -> BitNetTrainer:
    """Factory function to create BitNet trainer"""
    optimizer_config = {
        'learning_rate': learning_rate,
        'weight_decay': weight_decay
    }
    return BitNetTrainer(model, optimizer_config)

def optimize_bitnet_model(model: nn.Module, enable_memory_optimizations: bool = True) -> nn.Module:
    """Apply optimizations to BitNet model"""
    if enable_memory_optimizations:
        memory_optimizer = BitNetMemoryOptimizer(model)
        memory_optimizer.optimize_memory_usage()
        
    return model

# Training utilities
class EarlyStopping:
    """Early stopping utility for training"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
            
        return False

class TrainingMetrics:
    """Training metrics tracker"""
    
    def __init__(self):
        self.metrics = {}
        self.history = []
        
    def update(self, **kwargs):
        """Update metrics"""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
            
    def get_average(self, key: str, last_n: int = None) -> float:
        """Get average of metric"""
        if key not in self.metrics:
            return 0.0
            
        values = self.metrics[key]
        if last_n:
            values = values[-last_n:]
            
        return sum(values) / len(values) if values else 0.0
        
    def get_latest(self, key: str) -> float:
        """Get latest value of metric"""
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        return self.metrics[key][-1]
        
    def save_history(self, filepath: str):
        """Save metrics history"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def load_history(self, filepath: str):
        """Load metrics history"""
        import json
        with open(filepath, 'r') as f:
            self.metrics = json.load(f)
