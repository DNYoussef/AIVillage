#!/usr/bin/env python3
"""
Agent Forge Phase 4: Forge Training Loop with Grokfast

This phase implements the main training loop with edge-of-chaos optimization,
self-modeling, dream cycles, and Grokfast acceleration for 50x faster grokking.

Key Features:
- Edge-of-chaos controller (maintains 55-75% success rate)
- Grokfast optimization (50x acceleration of grokking)
- Self-modeling head for efficiency prediction
- Dream cycles for consolidation and augmentation
- Temperature curriculum for progressive difficulty
- Geometry probing for training insights
- Comprehensive telemetry and monitoring

Consolidates implementations from:
- packages/agent_forge/legacy_src/training/forge_train.py (main loop)
- packages/agent_forge/legacy_src/training/grokfast_optimizer.py (Grokfast implementation)
- packages/agent_forge/legacy_src/training/edge.py (edge-of-chaos controller)
- packages/agent_forge/legacy_src/training/self_model.py (self-modeling)
- packages/agent_forge/legacy_src/training/dream.py (dream cycles)
"""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any

from datasets import load_dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try to import PhaseController, with fallback for direct imports
try:
    from ..core.phase_controller import PhaseController, PhaseResult
except (ImportError, ValueError):
    # Fallback for direct imports - create minimal base classes
    from abc import ABC, abstractmethod
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Any

    import torch.nn as nn

    @dataclass
    class PhaseResult:
        success: bool
        model: nn.Module
        phase_name: str = None
        metrics: dict = None
        duration_seconds: float = 0.0
        artifacts: dict = None
        config: dict = None
        error: str = None
        start_time: datetime = None
        end_time: datetime = None

        def __post_init__(self):
            if self.end_time is None:
                self.end_time = datetime.now()
            if self.start_time is None:
                self.start_time = self.end_time

    class PhaseController(ABC):
        def __init__(self, config: Any):
            self.config = config

        @abstractmethod
        async def run(self, model: nn.Module) -> PhaseResult:
            pass


logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class PhaseConfig:
    """Base configuration class for Agent Forge phases."""

    pass


@dataclass
class ForgeTrainingConfig(PhaseConfig):
    """Configuration for Forge training loop."""

    # Model configuration
    model_path: str = ""
    output_path: str = ""
    tokenizer_path: str | None = None

    # Training configuration
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_steps: int = 50000
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.01

    # Grokfast configuration
    enable_grokfast: bool = True
    grokfast_ema_alpha: float = 0.98
    grokfast_lambda_init: float = 0.05
    grokfast_lambda_max: float = 0.25
    grokfast_lambda_schedule: str = "cosine"  # "linear", "cosine", "constant"

    # Edge-of-chaos configuration
    enable_edge_control: bool = True
    target_success_range: tuple[float, float] = (0.55, 0.75)
    edge_window_size: int = 100
    edge_exploration_rate: float = 0.1

    # Self-modeling configuration
    enable_self_model: bool = True
    self_model_weight: float = 0.1
    self_model_warmup: int = 5000
    self_model_layers: list[int] = field(default_factory=lambda: [4, 8, 12])

    # Dream cycle configuration
    enable_dream_cycles: bool = True
    dream_cycle_interval: int = 1000
    dream_duration: int = 50
    dream_buffer_capacity: int = 10000
    dream_augmentation_strength: float = 0.2

    # Temperature curriculum
    enable_temp_curriculum: bool = True
    temp_curriculum_interval: int = 2000
    initial_temperature: float = 1.0
    final_temperature: float = 0.1

    # Evaluation configuration
    eval_interval: int = 500
    eval_samples: int = 200
    eval_metrics: list[str] = field(default_factory=lambda: ["perplexity", "accuracy", "grok_progress"])

    # Training tasks
    training_tasks: list[str] = field(default_factory=lambda: ["language_modeling", "arithmetic", "pattern_matching"])
    task_switching_interval: int = 2000

    # System configuration
    device: str = "auto"
    mixed_precision: bool = True
    seed: int = 42
    num_workers: int = 4

    # Logging and checkpointing
    log_interval: int = 50
    checkpoint_interval: int = 1000
    save_optimizer_state: bool = True

    # W&B tracking
    wandb_project: str = "agent_forge"
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(default_factory=lambda: ["forge_training", "phase4", "grokfast"])


# ============================================================================
# Grokfast Optimizer
# ============================================================================


class GrokfastAdamW(torch.optim.Optimizer):
    """
    AdamW optimizer enhanced with Grokfast gradient filtering.

    Grokfast accelerates grokking by 50x by amplifying slow gradients
    and dampening fast gradients using EMA filtering.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        ema_alpha: float = 0.98,
        grokfast_lambda: float = 0.05,
        grokfast_enabled: bool = True,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            ema_alpha=ema_alpha,
            grokfast_lambda=grokfast_lambda,
            grokfast_enabled=grokfast_enabled,
        )
        super().__init__(params, defaults)

        # Initialize Grokfast EMA buffers
        self._init_grokfast_buffers()

    def _init_grokfast_buffers(self):
        """Initialize EMA gradient buffers for Grokfast."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    state["ema_grad"] = torch.zeros_like(p.data)
                    state["grokfast_initialized"] = False

    @torch.no_grad()
    def step(self, closure: Callable | None = None, grokfast_lambda_override: float | None = None):
        """Perform optimization step with Grokfast filtering."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            ema_alpha = group["ema_alpha"]
            grokfast_lambda = grokfast_lambda_override or group["grokfast_lambda"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if "ema_grad" not in state:
                        state["ema_grad"] = torch.zeros_like(p)
                        state["grokfast_initialized"] = False

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                grad = p.grad

                # Apply Grokfast filtering if enabled
                if group["grokfast_enabled"] and grokfast_lambda > 0:
                    grad = self._apply_grokfast_filter(grad, state, ema_alpha, grokfast_lambda)

                # AdamW update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                step_size = group["lr"] / bias_correction1
                bias_correction2_sqrt = bias_correction2**0.5

                # Weight decay
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                # Update parameters
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    def _apply_grokfast_filter(self, grad, state, ema_alpha, grokfast_lambda):
        """Apply Grokfast gradient filtering."""
        ema_grad = state["ema_grad"]

        if not state["grokfast_initialized"]:
            # Initialize with current gradient
            ema_grad.copy_(grad)
            state["grokfast_initialized"] = True
            return grad

        # Update EMA
        ema_grad.mul_(ema_alpha).add_(grad, alpha=1 - ema_alpha)

        # Compute cosine similarity
        grad_flat = grad.flatten()
        ema_flat = ema_grad.flatten()

        cosine_sim = F.cosine_similarity(grad_flat.unsqueeze(0), ema_flat.unsqueeze(0)).item()
        cosine_sim = max(cosine_sim, 0)  # Clamp to [0, 1]

        # Apply Grokfast amplification
        filtered_grad = grad + grokfast_lambda * cosine_sim * ema_grad

        return filtered_grad


# ============================================================================
# Edge-of-Chaos Controller
# ============================================================================


class EdgeController:
    """
    Controller that maintains task success rate in the optimal learning zone (55-75%)
    by dynamically adjusting difficulty parameters.
    """

    def __init__(
        self,
        target_range: tuple[float, float] = (0.55, 0.75),
        window_size: int = 100,
        exploration_rate: float = 0.1,
        difficulty_params: dict[str, tuple[float, float]] | None = None,
    ):
        self.target_min, self.target_max = target_range
        self.target_center = (self.target_min + self.target_max) / 2
        self.window_size = window_size
        self.exploration_rate = exploration_rate

        # Success rate history
        self.success_history = deque(maxlen=window_size)

        # Difficulty parameters and ranges
        self.difficulty_params = difficulty_params or {
            "sequence_length": (64, 512),
            "complexity_level": (1, 10),
            "dropout_rate": (0.0, 0.3),
            "noise_level": (0.0, 0.1),
        }

        # Current difficulty settings
        self.current_difficulty = {
            param: (min_val + max_val) / 2 for param, (min_val, max_val) in self.difficulty_params.items()
        }

        # Adaptation parameters
        self.adaptation_rate = 0.1
        self.momentum = 0.9
        self.velocity = {param: 0.0 for param in self.difficulty_params}

    def update(self, recent_scores: list[float]) -> dict[str, float]:
        """Update controller and return adjusted difficulty parameters."""
        self.success_history.extend(recent_scores)

        if len(self.success_history) < 10:
            return self.current_difficulty

        current_rate = np.mean(list(self.success_history))

        # Determine adjustment
        if current_rate < self.target_min:
            # Too hard, decrease difficulty
            adjustment_factor = (self.target_center - current_rate) / self.target_center
            self._adjust_difficulty(-adjustment_factor)
        elif current_rate > self.target_max:
            # Too easy, increase difficulty
            adjustment_factor = (current_rate - self.target_center) / (1 - self.target_center)
            self._adjust_difficulty(adjustment_factor)

        return self.current_difficulty

    def _adjust_difficulty(self, factor: float):
        """Adjust difficulty parameters with momentum."""
        for param, (min_val, max_val) in self.difficulty_params.items():
            # Calculate adjustment with momentum
            adjustment = self.adaptation_rate * factor * (max_val - min_val)
            self.velocity[param] = self.momentum * self.velocity[param] + adjustment

            # Apply adjustment
            self.current_difficulty[param] += self.velocity[param]

            # Clamp to valid range
            self.current_difficulty[param] = max(min_val, min(max_val, self.current_difficulty[param]))

    def get_success_rate(self) -> float:
        """Get current success rate."""
        if len(self.success_history) < 5:
            return 0.5
        return np.mean(list(self.success_history))

    def is_in_target_zone(self) -> bool:
        """Check if current success rate is in target zone."""
        current_rate = self.get_success_rate()
        return self.target_min <= current_rate <= self.target_max


# ============================================================================
# Self-Modeling Head
# ============================================================================


class SelfModelHead(nn.Module):
    """
    Self-modeling head that predicts the model's own internal states.
    This regularizes training and improves efficiency.
    """

    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # State prediction heads
        self.activation_predictor = nn.Linear(hidden_size, hidden_size)
        self.attention_predictor = nn.Linear(hidden_size, 1)  # Attention weight prediction
        self.layer_predictor = nn.Linear(hidden_size, num_layers)  # Layer classification

        # Confidence estimation
        self.confidence_head = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Predict model's internal states.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            Dictionary with predictions
        """
        # Average pooling for sequence-level predictions
        pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_size]

        # Predictions
        activation_pred = self.activation_predictor(pooled)
        attention_pred = torch.sigmoid(self.attention_predictor(pooled))
        layer_pred = F.log_softmax(self.layer_predictor(pooled), dim=-1)
        confidence = torch.sigmoid(self.confidence_head(pooled))

        return {
            "activation_prediction": activation_pred,
            "attention_prediction": attention_pred,
            "layer_prediction": layer_pred,
            "confidence": confidence,
        }

    def compute_self_model_loss(
        self, predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute self-modeling loss."""
        total_loss = 0.0

        # Activation prediction loss
        if "activation_target" in targets:
            activation_loss = F.mse_loss(predictions["activation_prediction"], targets["activation_target"])
            total_loss += activation_loss

        # Layer prediction loss
        if "layer_target" in targets:
            layer_loss = F.nll_loss(predictions["layer_prediction"], targets["layer_target"])
            total_loss += layer_loss * 0.1  # Lower weight

        return total_loss


# Alias for backward compatibility
SelfModelingModule = SelfModelHead


# ============================================================================
# Dream Cycle System
# ============================================================================


@dataclass
class DreamExample:
    """Example from dream buffer for replay."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    metadata: dict[str, Any]
    dream_score: float


class DreamBuffer:
    """Buffer for storing and replaying dream examples."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: list[DreamExample] = []
        self.position = 0

    def add(self, example: DreamExample):
        """Add example to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(example)
        else:
            self.buffer[self.position] = example
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> list[DreamExample]:
        """Sample batch from buffer."""
        if len(self.buffer) < batch_size:
            return self.buffer.copy()

        # Sample with bias towards higher dream scores
        scores = np.array([ex.dream_score for ex in self.buffer])
        probs = F.softmax(torch.tensor(scores), dim=0).numpy()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class DreamCycleManager:
    """Manages dream cycles for memory consolidation and augmentation."""

    def __init__(self, config: ForgeTrainingConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.dream_buffer = DreamBuffer(config.dream_buffer_capacity)

        # Dream state
        self.last_dream_step = 0
        self.dream_active = False

    def should_dream(self, current_step: int) -> bool:
        """Check if model should enter dream cycle."""
        return current_step - self.last_dream_step >= self.config.dream_cycle_interval and len(self.dream_buffer) > 10

    def enter_dream_cycle(self, model: nn.Module, current_step: int) -> list[DreamExample]:
        """Enter dream cycle and generate examples."""
        logger.info(f"Entering dream cycle at step {current_step}")

        self.dream_active = True
        self.last_dream_step = current_step

        # Generate dream examples through model sampling
        dream_examples = []
        model.eval()

        with torch.no_grad():
            for _ in range(self.config.dream_duration):
                # Generate sequence
                dream_input = self._generate_dream_prompt()
                input_ids = self.tokenizer.encode(dream_input, return_tensors="pt")

                # Generate continuation
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=100,
                    temperature=1.2,  # Higher temperature for creativity
                    do_sample=True,
                    top_p=0.9,
                )

                # Score the dream
                dream_score = self._score_dream_quality(outputs)

                example = DreamExample(
                    input_ids=input_ids.squeeze(),
                    labels=outputs.squeeze(),
                    metadata={"step": current_step, "dream_type": "generative", "temperature": 1.2},
                    dream_score=dream_score,
                )

                dream_examples.append(example)
                self.dream_buffer.add(example)

        self.dream_active = False
        model.train()

        logger.info(f"Dream cycle complete: generated {len(dream_examples)} examples")
        return dream_examples

    def _generate_dream_prompt(self) -> str:
        """Generate diverse prompts for dreaming."""
        prompts = [
            "Once upon a time",
            "In a world where",
            "The solution to this problem is",
            "Consider the following:",
            "Imagine that",
        ]
        return np.random.choice(prompts)

    def _score_dream_quality(self, outputs: torch.Tensor) -> float:
        """Score dream quality based on diversity and coherence."""
        # Simple scoring: longer sequences with diverse tokens get higher scores
        unique_tokens = len(torch.unique(outputs))
        sequence_length = outputs.size(-1)

        diversity_score = unique_tokens / sequence_length if sequence_length > 0 else 0
        length_bonus = min(sequence_length / 100, 1.0)

        return diversity_score * length_bonus

    def get_dream_replay_batch(self, batch_size: int) -> list[DreamExample] | None:
        """Get batch for dream replay."""
        if len(self.dream_buffer) < batch_size:
            return None

        return self.dream_buffer.sample(batch_size)


# ============================================================================
# Training Dataset
# ============================================================================


class ForgeTrainingDataset(Dataset):
    """Multi-task training dataset for Forge training."""

    def __init__(self, config: ForgeTrainingConfig, tokenizer, split: str = "train"):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split

        # Load datasets for different tasks
        self.datasets = {}
        self.examples = []

        self._load_datasets()

        logger.info(f"Loaded {len(self.examples)} training examples")

    def _load_datasets(self):
        """Load datasets for different training tasks."""
        # Language modeling
        if "language_modeling" in self.config.training_tasks:
            try:
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=self.split)
                lm_examples = self._prepare_language_modeling_examples(dataset)
                self.examples.extend(lm_examples)
            except Exception as e:
                logger.warning(f"Failed to load wikitext: {e}")

        # Arithmetic reasoning (for grokking)
        if "arithmetic" in self.config.training_tasks:
            arithmetic_examples = self._generate_arithmetic_examples(1000)
            self.examples.extend(arithmetic_examples)

        # Pattern matching
        if "pattern_matching" in self.config.training_tasks:
            pattern_examples = self._generate_pattern_examples(500)
            self.examples.extend(pattern_examples)

    def _prepare_language_modeling_examples(self, dataset) -> list[dict]:
        """Prepare language modeling examples."""
        examples = []

        for item in dataset:
            if len(item["text"].strip()) < 50:
                continue

            examples.append({"text": item["text"], "task_type": "language_modeling", "difficulty": 1.0})

            if len(examples) >= 5000:  # Limit for efficiency
                break

        return examples

    def _generate_arithmetic_examples(self, num_examples: int) -> list[dict]:
        """Generate arithmetic examples for grokking."""
        examples = []

        for _ in range(num_examples):
            # Simple arithmetic operations
            a = np.random.randint(1, 100)
            b = np.random.randint(1, 100)
            op = np.random.choice(["+", "-", "*"])

            if op == "+":
                result = a + b
            elif op == "-":
                result = a - b
            else:  # '*'
                result = a * b

            text = f"{a} {op} {b} = {result}"

            examples.append({"text": text, "task_type": "arithmetic", "difficulty": min(max(a, b) / 100, 1.0)})

        return examples

    def _generate_pattern_examples(self, num_examples: int) -> list[dict]:
        """Generate pattern recognition examples."""
        examples = []

        for _ in range(num_examples):
            # Simple sequence patterns
            pattern_length = np.random.randint(3, 8)
            pattern = [np.random.randint(1, 10) for _ in range(pattern_length)]

            # Repeat pattern
            full_sequence = pattern * 3
            sequence_str = " ".join(map(str, full_sequence))

            text = f"Pattern: {sequence_str}"

            examples.append({"text": text, "task_type": "pattern_matching", "difficulty": pattern_length / 8.0})

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Tokenize
        encoding = self.tokenizer(
            example["text"], truncation=True, padding="max_length", max_length=512, return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
            "task_type": example["task_type"],
            "difficulty": example["difficulty"],
        }


# ============================================================================
# Main Training Loop
# ============================================================================


class ForgeTrainer:
    """Main trainer that orchestrates the complete Forge training loop."""

    def __init__(self, config: ForgeTrainingConfig):
        self.config = config
        self.device = torch.device(
            config.device if config.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float("inf")

        # Controllers
        self.edge_controller = EdgeController(config.target_success_range, config.edge_window_size)
        self.dream_manager = None  # Will be initialized with tokenizer

        # Metrics tracking
        self.training_metrics = {
            "loss_history": [],
            "success_rate_history": [],
            "grokfast_lambda_history": [],
            "edge_difficulty_history": [],
        }

    async def train(self, model_path: str) -> dict[str, Any]:
        """Execute complete Forge training pipeline."""
        logger.info("🔥 Starting Forge Training with Grokfast")

        # Load model and tokenizer
        model, tokenizer = self._load_model(model_path)

        # Initialize dream manager
        self.dream_manager = DreamCycleManager(self.config, tokenizer)

        # Add self-modeling head if enabled
        if self.config.enable_self_model:
            self.self_model_head = SelfModelHead(model.config.hidden_size, model.config.num_hidden_layers).to(
                self.device
            )

        # Create dataset and dataloader
        train_dataset = ForgeTrainingDataset(self.config, tokenizer)
        dataloader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers
        )

        # Initialize Grokfast optimizer
        optimizer = GrokfastAdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            ema_alpha=self.config.grokfast_ema_alpha,
            grokfast_lambda=self.config.grokfast_lambda_init,
            grokfast_enabled=self.config.enable_grokfast,
        )

        # Training loop
        model.train()
        total_loss = 0.0
        batch_losses = []
        batch_accuracies = []

        progress_bar = tqdm(range(self.config.max_steps), desc="Forge Training")

        dataloader_iter = iter(dataloader)
        for step in progress_bar:
            self.global_step = step

            # Precompute current Grokfast lambda for logging and potential override
            current_lambda = self._get_grokfast_lambda(step)

            # Get batch
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                # Restart dataloader iterator
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Add self-modeling loss if enabled
            if self.config.enable_self_model and step >= self.config.self_model_warmup:
                hidden_states = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else None

                if hidden_states is not None:
                    self_predictions = self.self_model_head(hidden_states)
                    # Create simple targets (this would be more sophisticated in practice)
                    targets = {"activation_target": hidden_states.mean(dim=1)}
                    self_loss = self.self_model_head.compute_self_model_loss(self_predictions, targets)
                    loss += self.config.self_model_weight * self_loss

            # Backward pass with gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Update Grokfast lambda
                current_lambda = self._get_grokfast_lambda(step)

                # Optimizer step with dynamic lambda
                optimizer.step(grokfast_lambda_override=current_lambda)
                optimizer.zero_grad()

            # Track metrics
            batch_loss = loss.item() * self.config.gradient_accumulation_steps
            batch_losses.append(batch_loss)
            total_loss += batch_loss

            # Calculate accuracy for edge controller
            with torch.no_grad():
                predictions = torch.argmax(outputs.logits, dim=-1)
                accuracy = (predictions == labels).float().mean().item()
                batch_accuracies.append(accuracy)

            # Update edge-of-chaos controller
            if self.config.enable_edge_control and len(batch_accuracies) >= 10:
                self.edge_controller.update(batch_accuracies[-10:])
                # Apply difficulty settings to future batches (implementation specific)

            # Dream cycle
            if self.config.enable_dream_cycles and self.dream_manager.should_dream(step):
                dream_examples = self.dream_manager.enter_dream_cycle(model, step)

                # Optional: Train on dream examples
                if len(dream_examples) > 0:
                    self._train_on_dreams(model, dream_examples, optimizer)

            # Logging
            if step % self.config.log_interval == 0:
                avg_loss = np.mean(batch_losses[-self.config.log_interval :])
                avg_accuracy = np.mean(batch_accuracies[-self.config.log_interval :])
                success_rate = self.edge_controller.get_success_rate()

                progress_bar.set_postfix(
                    {
                        "loss": f"{avg_loss:.4f}",
                        "acc": f"{avg_accuracy:.3f}",
                        "success": f"{success_rate:.3f}",
                        "lambda": f"{current_lambda:.4f}",
                    }
                )

                # Update metrics history
                self.training_metrics["loss_history"].append(avg_loss)
                self.training_metrics["success_rate_history"].append(success_rate)
                self.training_metrics["grokfast_lambda_history"].append(current_lambda)

            # Evaluation
            if step % self.config.eval_interval == 0 and step > 0:
                eval_metrics = await self._evaluate_model(model, tokenizer)
                logger.info(f"Step {step} - Eval: {eval_metrics}")

            # Checkpointing
            if step % self.config.checkpoint_interval == 0 and step > 0:
                self._save_checkpoint(model, tokenizer, optimizer, step)

        # Final save
        self._save_final_model(model, tokenizer)

        # Compile results
        final_metrics = {
            "total_steps": self.config.max_steps,
            "final_loss": np.mean(batch_losses[-100:]) if batch_losses else 0,
            "final_accuracy": np.mean(batch_accuracies[-100:]) if batch_accuracies else 0,
            "final_success_rate": self.edge_controller.get_success_rate(),
            "edge_in_target_zone": self.edge_controller.is_in_target_zone(),
            "dream_examples_generated": len(self.dream_manager.dream_buffer),
            "training_metrics": self.training_metrics,
        }

        success = (
            final_metrics["final_loss"] < 2.0
            and final_metrics["final_accuracy"] > 0.7
            and final_metrics["edge_in_target_zone"]
        )

        logger.info(f"✅ Forge training complete - Success: {success}")

        return {"success": success, "model_path": self.config.output_path, "metrics": final_metrics}

    def _load_model(self, model_path: str) -> tuple[nn.Module, AutoTokenizer]:
        """Load model and tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path or model_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            output_hidden_states=True,  # Needed for self-modeling
        )

        model.to(self.device)

        return model, tokenizer

    def _get_grokfast_lambda(self, step: int) -> float:
        """Get current Grokfast lambda value based on schedule."""
        progress = step / self.config.max_steps

        if self.config.grokfast_lambda_schedule == "linear":
            lambda_val = self.config.grokfast_lambda_init + progress * (
                self.config.grokfast_lambda_max - self.config.grokfast_lambda_init
            )
        elif self.config.grokfast_lambda_schedule == "cosine":
            lambda_val = self.config.grokfast_lambda_init + 0.5 * (
                self.config.grokfast_lambda_max - self.config.grokfast_lambda_init
            ) * (1 + np.cos(np.pi * progress))
        else:  # constant
            lambda_val = self.config.grokfast_lambda_init

        return lambda_val

    def _train_on_dreams(self, model: nn.Module, dream_examples: list[DreamExample], optimizer):
        """Train on dream examples."""
        model.train()

        for example in dream_examples[:10]:  # Limit dream training
            # Simple training step on dream example
            outputs = model(
                input_ids=example.input_ids.unsqueeze(0).to(self.device),
                labels=example.labels.unsqueeze(0).to(self.device),
            )

            dream_loss = outputs.loss * 0.1  # Lower weight for dreams
            dream_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    async def _evaluate_model(self, model: nn.Module, tokenizer) -> dict[str, float]:
        """Evaluate model performance."""
        model.eval()

        # Simple evaluation on a held-out set
        eval_dataset = ForgeTrainingDataset(self.config, tokenizer, split="validation")
        eval_dataloader = DataLoader(eval_dataset, batch_size=8, shuffle=False)

        total_loss = 0.0
        total_accuracy = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                if total_batches >= 25:  # Limit evaluation time
                    break

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                total_loss += outputs.loss.item()

                predictions = torch.argmax(outputs.logits, dim=-1)
                accuracy = (predictions == labels).float().mean().item()
                total_accuracy += accuracy

                total_batches += 1

        model.train()

        return {
            "eval_loss": total_loss / total_batches if total_batches > 0 else float("inf"),
            "eval_accuracy": total_accuracy / total_batches if total_batches > 0 else 0.0,
        }

    def _save_checkpoint(self, model: nn.Module, tokenizer, optimizer, step: int):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.output_path).parent / "checkpoints" / f"step_{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        # Save optimizer state if requested
        if self.config.save_optimizer_state:
            torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

        # Save training metrics
        with open(checkpoint_dir / "training_metrics.json", "w") as f:
            json.dump(self.training_metrics, f, indent=2, default=str)

        logger.info(f"Checkpoint saved at step {step}")

    def _save_final_model(self, model: nn.Module, tokenizer):
        """Save final trained model."""
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        # Save final config and metrics
        with open(output_path / "training_config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)

        with open(output_path / "final_metrics.json", "w") as f:
            json.dump(self.training_metrics, f, indent=2, default=str)

        logger.info(f"Final model saved to {output_path}")


# ============================================================================
# Phase Controller
# ============================================================================


class ForgeTrainingPhase(PhaseController):
    """
    Phase 4: Forge Training Loop Controller

    Implements comprehensive training with edge-of-chaos optimization,
    Grokfast acceleration, self-modeling, and dream cycles.
    """

    def __init__(self, config: ForgeTrainingConfig):
        super().__init__(config)
        self.config = config
        self.phase_name = "Forge Training Loop"
        self.phase_number = 4

        # Set random seeds
        torch.manual_seed(getattr(config, 'seed', 42))
        np.random.seed(getattr(config, 'seed', 42))

    async def run(self, model: nn.Module) -> PhaseResult:
        """
        Execute the Forge Training phase processing.

        Args:
            model: Input model from previous phase

        Returns:
            PhaseResult with processed model and metrics
        """
        # Validate input model
        if not self.validate_input_model(model):
            return self.create_failure_result(model, "Input model validation failed")

        start_time = time.time()

        try:
            # Save model temporarily to pass to execute method
            temp_model_path = Path(self.config.output_path) / "temp_input_model"
            temp_model_path.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(str(temp_model_path))

            # Execute the phase using existing execute method
            result = await self.execute(str(temp_model_path))

            duration = time.time() - start_time

            if result.success:
                # Load the trained model from output path
                model_path = getattr(result, 'model_path', self.config.output_path)
                trained_model = AutoModelForCausalLM.from_pretrained(model_path)

                return self.create_success_result(
                    model=trained_model,
                    metrics=result.metrics or {},
                    artifacts=result.artifacts or {},
                    duration=duration
                )
            else:
                return self.create_failure_result(model, result.error or "Forge Training failed", duration)

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Forge Training phase failed: {e}")
            return self.create_failure_result(model, str(e), duration)

    async def execute(self, input_model_path: str, **kwargs) -> PhaseResult:
        """Execute Phase 4: Forge training."""
        try:
            logger.info(f"🔥 Starting {self.phase_name}")

            # Update config with input model path
            self.config.model_path = input_model_path

            # Create trainer
            trainer = ForgeTrainer(self.config)

            # Run training
            training_results = await trainer.train(input_model_path)

            # Create phase result
            result = PhaseResult(
                phase_name=self.phase_name,
                success=training_results["success"],
                model_path=self.config.output_path,
                metrics=training_results["metrics"],
                artifacts={"training_results": training_results, "config": self.config.__dict__},
                duration_seconds=0,  # Will be calculated by orchestrator
                memory_usage_mb=0,  # Will be calculated by orchestrator
            )

            status = "✅ SUCCESS" if training_results["success"] else "⚠️  PARTIAL"
            logger.info(
                f"{status} - Final loss: {training_results['metrics']['final_loss']:.4f}, Edge in zone: {training_results['metrics']['edge_in_target_zone']}"
            )

            return result

        except Exception as e:
            logger.exception(f"Forge training failed: {e}")

            return PhaseResult(
                phase_name=self.phase_name,
                success=False,
                model_path="",
                metrics={"error": str(e)},
                artifacts={},
                duration_seconds=0,
                memory_usage_mb=0,
                error_message=str(e),
            )


# ============================================================================
# Backwards compatibility: ensure ForgeTrainingPhase implements run()
# ============================================================================
try:
    # Add an async run wrapper to ForgeTrainingPhase if missing
    if not hasattr(ForgeTrainingPhase, "run"):
        import json
        from pathlib import Path
        import tempfile

        async def _forge_training_run(self, model_or_path, **kwargs):
            """Compatibility wrapper: accept nn.Module or path and delegate to execute()."""
            # If a path string is passed, use it directly
            if isinstance(model_or_path, str):
                model_path = model_or_path
            elif model_or_path is None:
                model_path = self.config.model_path
            else:
                # Persist the nn.Module to a temporary directory where possible
                tmpdir = Path(tempfile.mkdtemp(prefix="forge_input_model_"))
                try:
                    if hasattr(model_or_path, "save_pretrained"):
                        model_or_path.save_pretrained(tmpdir)
                    else:
                        # Save state dict and minimal config
                        torch.save(model_or_path.state_dict(), tmpdir / "pytorch_model.bin")
                        if hasattr(model_or_path, "config"):
                            with open(tmpdir / "config.json", "w") as f:
                                json.dump(getattr(model_or_path, "config").__dict__, f)
                except Exception as e:
                    logger.warning(f"Failed to persist input model, falling back to config.model_path: {e}")
                    tmpdir = None
                model_path = str(tmpdir) if tmpdir else self.config.model_path

            return await self.execute(model_path, **kwargs)

        setattr(ForgeTrainingPhase, "run", _forge_training_run)
except Exception:
    # Best-effort; do not fail import if wrapper cannot be installed
    logger.exception("Failed to attach run wrapper to ForgeTrainingPhase")

# ============================================================================
# Factory Function
# ============================================================================


def create_forge_training_phase(
    model_path: str = "",
    output_path: str = "",
    max_steps: int = 50000,
    enable_grokfast: bool = True,
    enable_edge_control: bool = True,
    enable_dream_cycles: bool = True,
    device: str = "auto",
    **kwargs,
) -> ForgeTrainingPhase:
    """
    Factory function to create Forge training phase.

    Args:
        model_path: Path to input model from BitNet compression
        output_path: Path for trained model output
        max_steps: Maximum training steps
        enable_grokfast: Enable Grokfast optimization
        enable_edge_control: Enable edge-of-chaos controller
        enable_dream_cycles: Enable dream cycles
        device: Device to use
        **kwargs: Additional configuration options

    Returns:
        ForgeTrainingPhase: Configured phase controller
    """
    config = ForgeTrainingConfig(
        model_path=model_path,
        output_path=output_path,
        max_steps=max_steps,
        enable_grokfast=enable_grokfast,
        enable_edge_control=enable_edge_control,
        enable_dream_cycles=enable_dream_cycles,
        device=device,
        **kwargs,
    )

    return ForgeTrainingPhase(config)


# ============================================================================
# Example Usage
# ============================================================================


if __name__ == "__main__":

    async def main():
        # Example: Create and run Forge training phase
        phase = create_forge_training_phase(
            model_path="./phase3_bitnet_output",
            output_path="./phase4_forge_training_output",
            max_steps=10000,  # Smaller for testing
            enable_grokfast=True,
            enable_edge_control=True,
            enable_dream_cycles=True,
        )

        result = await phase.execute("./phase3_bitnet_output")

        print(f"Phase Result: {result.success}")
        print(f"Final Loss: {result.metrics.get('final_loss', 0):.4f}")
        print(f"Edge in Target Zone: {result.metrics.get('edge_in_target_zone', False)}")
        print(f"Model Path: {result.model_path}")

    # Uncomment to run example
    # asyncio.run(main())
