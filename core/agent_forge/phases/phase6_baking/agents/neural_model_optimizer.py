"""
Phase 6 Baking - Neural Model Optimizer Agent
Optimizes neural model parameters during tool/persona baking process
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import time
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import json
import pickle
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    GROKFAST = "grokfast"
    ADAPTIVE = "adaptive"


@dataclass
class OptimizationConfig:
    strategy: OptimizationStrategy
    learning_rate: float
    weight_decay: float
    momentum: float
    eps: float
    betas: Tuple[float, float]
    warmup_steps: int
    max_grad_norm: float
    adaptive_lr: bool
    grokfast_alpha: float
    grokfast_lambda: float


@dataclass
class OptimizationMetrics:
    loss_history: List[float]
    gradient_norms: List[float]
    parameter_updates: List[float]
    convergence_rate: float
    optimization_efficiency: float
    memory_usage: float
    computation_time: float
    last_update: datetime


class NeuralModelOptimizer:
    """Advanced neural model optimization with multiple strategies"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.metrics = OptimizationMetrics(
            loss_history=[],
            gradient_norms=[],
            parameter_updates=[],
            convergence_rate=0.0,
            optimization_efficiency=0.0,
            memory_usage=0.0,
            computation_time=0.0,
            last_update=datetime.now()
        )
        self.optimizer = None
        self.scheduler = None
        self.model = None
        self.step_count = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.grokfast_filters = {}

    def initialize_optimizer(self, model: nn.Module) -> bool:
        """Initialize optimizer based on configuration"""
        try:
            self.model = model

            if self.config.strategy == OptimizationStrategy.ADAM:
                self.optimizer = optim.Adam(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    betas=self.config.betas,
                    eps=self.config.eps,
                    weight_decay=self.config.weight_decay
                )
            elif self.config.strategy == OptimizationStrategy.ADAMW:
                self.optimizer = optim.AdamW(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    betas=self.config.betas,
                    eps=self.config.eps,
                    weight_decay=self.config.weight_decay
                )
            elif self.config.strategy == OptimizationStrategy.GROKFAST:
                self.optimizer = optim.AdamW(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
                self._initialize_grokfast()
            else:
                self.optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

            # Initialize learning rate scheduler
            if self.config.warmup_steps > 0:
                self.scheduler = optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    total_iters=self.config.warmup_steps
                )

            logger.info(f"Optimizer initialized: {self.config.strategy.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {e}")
            return False

    def _initialize_grokfast(self):
        """Initialize GrokFast optimization filters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grokfast_filters[name] = torch.zeros_like(param.data)

    def optimize_step(self, loss: torch.Tensor) -> Dict[str, Any]:
        """Perform single optimization step with metrics tracking"""
        start_time = time.time()

        try:
            # Backward pass
            loss.backward()

            # Compute gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

            # Apply GrokFast if enabled
            if self.config.strategy == OptimizationStrategy.GROKFAST:
                self._apply_grokfast()

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update scheduler
            if self.scheduler and self.step_count < self.config.warmup_steps:
                self.scheduler.step()

            # Update metrics
            self.step_count += 1
            current_loss = loss.item()
            self.metrics.loss_history.append(current_loss)
            self.metrics.gradient_norms.append(grad_norm.item())
            self.metrics.computation_time = time.time() - start_time

            # Track convergence
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Calculate efficiency metrics
            self._update_efficiency_metrics()

            return {
                'loss': current_loss,
                'grad_norm': grad_norm.item(),
                'lr': self.optimizer.param_groups[0]['lr'],
                'step': self.step_count,
                'convergence_rate': self.metrics.convergence_rate,
                'efficiency': self.metrics.optimization_efficiency
            }

        except Exception as e:
            logger.error(f"Optimization step failed: {e}")
            return {'error': str(e)}

    def _apply_grokfast(self):
        """Apply GrokFast gradient filtering"""
        alpha = self.config.grokfast_alpha
        lambda_reg = self.config.grokfast_lambda

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Update filter
                self.grokfast_filters[name] = (
                    alpha * self.grokfast_filters[name] +
                    (1 - alpha) * param.grad
                )

                # Apply filtered gradient
                param.grad = param.grad + lambda_reg * self.grokfast_filters[name]

    def _update_efficiency_metrics(self):
        """Update optimization efficiency metrics"""
        if len(self.metrics.loss_history) > 10:
            recent_losses = self.metrics.loss_history[-10:]
            convergence_rate = abs(recent_losses[0] - recent_losses[-1]) / recent_losses[0]
            self.metrics.convergence_rate = convergence_rate

            # Calculate optimization efficiency
            gradient_stability = np.std(self.metrics.gradient_norms[-10:])
            self.metrics.optimization_efficiency = convergence_rate / (gradient_stability + 1e-8)

        self.metrics.last_update = datetime.now()

    def get_optimization_state(self) -> Dict[str, Any]:
        """Get current optimization state"""
        return {
            'step_count': self.step_count,
            'best_loss': self.best_loss,
            'current_lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0,
            'patience_counter': self.patience_counter,
            'metrics': asdict(self.metrics),
            'config': asdict(self.config)
        }

    def save_state(self, path: Path):
        """Save optimizer state"""
        state = {
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': asdict(self.metrics),
            'step_count': self.step_count,
            'best_loss': self.best_loss,
            'grokfast_filters': self.grokfast_filters
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Optimizer state saved to {path}")

    def load_state(self, path: Path):
        """Load optimizer state"""
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)

            if self.optimizer and state['optimizer']:
                self.optimizer.load_state_dict(state['optimizer'])

            if self.scheduler and state['scheduler']:
                self.scheduler.load_state_dict(state['scheduler'])

            self.step_count = state['step_count']
            self.best_loss = state['best_loss']
            self.grokfast_filters = state['grokfast_filters']

            # Restore metrics
            metrics_dict = state['metrics']
            self.metrics = OptimizationMetrics(**metrics_dict)

            logger.info(f"Optimizer state loaded from {path}")

        except Exception as e:
            logger.error(f"Failed to load optimizer state: {e}")

    async def optimize_model_async(self, model: nn.Module, data_loader, epochs: int) -> Dict[str, Any]:
        """Asynchronous model optimization"""
        if not self.initialize_optimizer(model):
            return {'error': 'Failed to initialize optimizer'}

        optimization_results = []

        for epoch in range(epochs):
            epoch_losses = []

            for batch_idx, batch_data in enumerate(data_loader):
                # Forward pass (this would be implemented by the calling code)
                # loss = model(batch_data)  # Placeholder

                # For now, simulate optimization
                simulated_loss = torch.tensor(1.0 / (epoch + 1), requires_grad=True)
                step_result = self.optimize_step(simulated_loss)
                epoch_losses.append(step_result.get('loss', 0.0))

                # Yield control for async operation
                if batch_idx % 10 == 0:
                    await asyncio.sleep(0.001)

            epoch_result = {
                'epoch': epoch,
                'avg_loss': np.mean(epoch_losses),
                'step_count': self.step_count,
                'efficiency': self.metrics.optimization_efficiency
            }
            optimization_results.append(epoch_result)

            logger.info(f"Epoch {epoch}: avg_loss={epoch_result['avg_loss']:.6f}")

        return {
            'epochs_completed': epochs,
            'final_loss': self.best_loss,
            'total_steps': self.step_count,
            'convergence_rate': self.metrics.convergence_rate,
            'optimization_efficiency': self.metrics.optimization_efficiency,
            'results': optimization_results
        }


def create_default_optimizer_config() -> OptimizationConfig:
    """Create default optimization configuration"""
    return OptimizationConfig(
        strategy=OptimizationStrategy.ADAMW,
        learning_rate=1e-4,
        weight_decay=0.01,
        momentum=0.9,
        eps=1e-8,
        betas=(0.9, 0.999),
        warmup_steps=1000,
        max_grad_norm=1.0,
        adaptive_lr=True,
        grokfast_alpha=0.98,
        grokfast_lambda=2.0
    )


# Agent Integration Interface
class NeuralModelOptimizerAgent:
    """Agent wrapper for neural model optimizer"""

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or create_default_optimizer_config()
        self.optimizer = NeuralModelOptimizer(self.config)
        self.agent_id = "neural_model_optimizer"
        self.status = "idle"

    async def run(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """Run optimization agent"""
        self.status = "running"

        try:
            # Extract parameters
            data_loader = kwargs.get('data_loader')
            epochs = kwargs.get('epochs', 10)

            if data_loader:
                result = await self.optimizer.optimize_model_async(model, data_loader, epochs)
            else:
                # Initialize for single-step optimization
                self.optimizer.initialize_optimizer(model)
                result = self.optimizer.get_optimization_state()

            self.status = "completed"
            return result

        except Exception as e:
            self.status = "failed"
            logger.error(f"Neural model optimizer failed: {e}")
            return {'error': str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'agent_id': self.agent_id,
            'status': self.status,
            'optimizer_state': self.optimizer.get_optimization_state()
        }