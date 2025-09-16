"""
Quiet-STaR Training Utilities

This module implements training components for the Quiet-STaR (Self-Taught Reasoner) model,
including specialized loss functions, training loop modifications, and evaluation metrics.

The training framework supports:
- Thought prediction loss computation
- Coherence loss for reasoning quality
- Modified training loops with gradient flow through thoughts
- Comprehensive evaluation metrics for reasoning assessment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import numpy as np
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for Quiet-STaR training"""
    thought_loss_weight: float = 1.0
    coherence_loss_weight: float = 0.5
    prediction_loss_weight: float = 1.0

    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    thought_length: int = 20
    num_thoughts_per_token: int = 4

    eval_frequency: int = 100
    log_frequency: int = 10
    save_frequency: int = 1000

    use_wandb: bool = False
    project_name: str = "quiet-star-training"


class ReasoningLoss(nn.Module):
    """
    Combined loss function for Quiet-STaR training including:
    - Thought prediction loss
    - Coherence loss
    - Standard language modeling loss
    """

    def __init__(self, config: TrainingConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        # Loss functions
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')

        # Coherence scoring network
        self.coherence_scorer = nn.Sequential(
            nn.Linear(768, 256),  # Assuming hidden_dim=768
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        thought_logits: torch.Tensor,
        thought_targets: torch.Tensor,
        thought_embeddings: torch.Tensor,
        prediction_logits: torch.Tensor,
        prediction_targets: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for Quiet-STaR training

        Args:
            thought_logits: [batch_size, seq_len, thought_len, vocab_size]
            thought_targets: [batch_size, seq_len, thought_len]
            thought_embeddings: [batch_size, seq_len, thought_len, hidden_dim]
            prediction_logits: [batch_size, seq_len, vocab_size]
            prediction_targets: [batch_size, seq_len]
            attention_weights: [batch_size, seq_len, thought_len]

        Returns:
            Dictionary containing individual and total losses
        """
        batch_size, seq_len, thought_len, vocab_size = thought_logits.shape

        # 1. Thought prediction loss
        thought_loss = self._compute_thought_loss(thought_logits, thought_targets)

        # 2. Coherence loss
        coherence_loss = self._compute_coherence_loss(
            thought_embeddings, attention_weights
        )

        # 3. Standard prediction loss
        prediction_loss = self._compute_prediction_loss(
            prediction_logits, prediction_targets
        )

        # 4. Combined loss
        total_loss = (
            self.config.thought_loss_weight * thought_loss +
            self.config.coherence_loss_weight * coherence_loss +
            self.config.prediction_loss_weight * prediction_loss
        )

        return {
            'total_loss': total_loss,
            'thought_loss': thought_loss,
            'coherence_loss': coherence_loss,
            'prediction_loss': prediction_loss
        }

    def _compute_thought_loss(
        self,
        thought_logits: torch.Tensor,
        thought_targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for thought generation"""
        batch_size, seq_len, thought_len, vocab_size = thought_logits.shape

        # Reshape for cross entropy computation
        logits_flat = thought_logits.view(-1, vocab_size)
        targets_flat = thought_targets.view(-1)

        # Compute cross entropy loss
        losses = self.cross_entropy(logits_flat, targets_flat)
        losses = losses.view(batch_size, seq_len, thought_len)

        # Average over thought length and sequence length
        return losses.mean()

    def _compute_coherence_loss(
        self,
        thought_embeddings: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute coherence loss to encourage meaningful thoughts"""
        batch_size, seq_len, thought_len, hidden_dim = thought_embeddings.shape

        # 1. Internal coherence: consistency within each thought
        thought_mean = thought_embeddings.mean(dim=2, keepdim=True)  # [B, S, 1, H]
        internal_coherence = F.cosine_similarity(
            thought_embeddings, thought_mean, dim=-1
        ).mean()

        # 2. Diversity loss: encourage different thoughts
        if thought_len > 1:
            thought_pairs = []
            for i in range(thought_len):
                for j in range(i + 1, thought_len):
                    similarity = F.cosine_similarity(
                        thought_embeddings[:, :, i, :],
                        thought_embeddings[:, :, j, :],
                        dim=-1
                    )
                    thought_pairs.append(similarity)

            diversity_penalty = torch.stack(thought_pairs).mean()
        else:
            diversity_penalty = torch.tensor(0.0, device=thought_embeddings.device)

        # 3. Attention coherence (if attention weights provided)
        attention_coherence = torch.tensor(0.0, device=thought_embeddings.device)
        if attention_weights is not None:
            # Encourage focused attention
            attention_entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-8), dim=-1
            ).mean()
            attention_coherence = attention_entropy

        # Combine coherence components
        coherence_loss = (
            -internal_coherence +  # Maximize internal coherence
            0.1 * diversity_penalty +  # Minimize similarity between thoughts
            0.1 * attention_coherence  # Control attention entropy
        )

        return coherence_loss

    def _compute_prediction_loss(
        self,
        prediction_logits: torch.Tensor,
        prediction_targets: torch.Tensor
    ) -> torch.Tensor:
        """Standard language modeling loss"""
        batch_size, seq_len, vocab_size = prediction_logits.shape

        logits_flat = prediction_logits.view(-1, vocab_size)
        targets_flat = prediction_targets.view(-1)

        losses = self.cross_entropy(logits_flat, targets_flat)
        return losses.mean()


class QuietSTaRTrainer:
    """
    Training loop implementation for Quiet-STaR with gradient flow modifications
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Initialize loss function
        vocab_size = getattr(model, 'vocab_size', 50257)  # Default to GPT-2 vocab size
        self.loss_fn = ReasoningLoss(config, vocab_size).to(device)

        # Initialize optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(self.loss_fn.parameters()),
            lr=config.learning_rate,
            weight_decay=0.01
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=config.warmup_steps
        )

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Metrics tracking
        self.metrics = EvaluationMetrics(config)

        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(project=config.project_name, config=config.__dict__)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with gradient computation through thoughts"""
        self.model.train()
        self.loss_fn.train()

        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass through model
        with torch.autograd.set_detect_anomaly(True):
            outputs = self.model(
                input_ids=batch['input_ids'],
                generate_thoughts=True,
                return_thoughts=True
            )

            # Extract outputs
            thought_logits = outputs.get('thought_logits')
            thought_embeddings = outputs.get('thought_embeddings')
            prediction_logits = outputs.get('logits')
            attention_weights = outputs.get('thought_attention')

            # Prepare targets
            thought_targets = batch.get('thought_targets', batch['input_ids'])
            prediction_targets = batch['labels']

            # Compute losses
            loss_dict = self.loss_fn(
                thought_logits=thought_logits,
                thought_targets=thought_targets,
                thought_embeddings=thought_embeddings,
                prediction_logits=prediction_logits,
                prediction_targets=prediction_targets,
                attention_weights=attention_weights
            )

            total_loss = loss_dict['total_loss']

        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.loss_fn.parameters()),
            self.config.max_grad_norm
        )

        self.optimizer.step()
        self.scheduler.step()

        # Convert losses to float for logging
        loss_dict = {k: v.item() for k, v in loss_dict.items()}

        return loss_dict

    def validate(self) -> Dict[str, float]:
        """Validation step with comprehensive metrics"""
        self.model.eval()
        self.loss_fn.eval()

        total_losses = {
            'total_loss': 0.0,
            'thought_loss': 0.0,
            'coherence_loss': 0.0,
            'prediction_loss': 0.0
        }

        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    generate_thoughts=True,
                    return_thoughts=True
                )

                # Compute losses
                loss_dict = self.loss_fn(
                    thought_logits=outputs.get('thought_logits'),
                    thought_targets=batch.get('thought_targets', batch['input_ids']),
                    thought_embeddings=outputs.get('thought_embeddings'),
                    prediction_logits=outputs.get('logits'),
                    prediction_targets=batch['labels'],
                    attention_weights=outputs.get('thought_attention')
                )

                # Accumulate losses
                for key, value in loss_dict.items():
                    total_losses[key] += value.item()

                num_batches += 1

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}

        return avg_losses

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        epoch_losses = {
            'total_loss': 0.0,
            'thought_loss': 0.0,
            'coherence_loss': 0.0,
            'prediction_loss': 0.0
        }

        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch in progress_bar:
            # Training step
            loss_dict = self.train_step(batch)

            # Accumulate losses
            for key, value in loss_dict.items():
                epoch_losses[key] += value

            num_batches += 1
            self.step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'thought': f"{loss_dict['thought_loss']:.4f}",
                'coherence': f"{loss_dict['coherence_loss']:.4f}"
            })

            # Logging
            if self.step % self.config.log_frequency == 0:
                self._log_metrics(loss_dict, 'train')

            # Validation
            if self.step % self.config.eval_frequency == 0:
                val_losses = self.validate()
                self._log_metrics(val_losses, 'val')

                # Save best model
                if val_losses['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total_loss']
                    self._save_checkpoint('best_model.pt')

            # Save checkpoint
            if self.step % self.config.save_frequency == 0:
                self._save_checkpoint(f'checkpoint_step_{self.step}.pt')

        # Average epoch losses
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}

        return avg_losses

    def _log_metrics(self, metrics: Dict[str, float], prefix: str):
        """Log metrics to console and wandb"""
        # Console logging
        metric_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Step {self.step} - {prefix} - {metric_str}")

        # Wandb logging
        if self.config.use_wandb:
            wandb_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            wandb_metrics['step'] = self.step
            wandb.log(wandb_metrics)

    def _save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'loss_fn_state_dict': self.loss_fn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }

        torch.save(checkpoint, filename)
        logger.info(f"Checkpoint saved: {filename}")


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for Quiet-STaR reasoning assessment
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.perplexity_with_thoughts = []
        self.perplexity_without_thoughts = []
        self.reasoning_improvement = []
        self.thought_coherence_scores = []
        self.attention_diversity = []
        self.thought_utilization = []

    def update(
        self,
        outputs_with_thoughts: Dict[str, torch.Tensor],
        outputs_without_thoughts: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ):
        """Update metrics with batch results"""

        # 1. Perplexity comparison
        ppl_with = self._compute_perplexity(
            outputs_with_thoughts['logits'], targets
        )
        ppl_without = self._compute_perplexity(
            outputs_without_thoughts['logits'], targets
        )

        self.perplexity_with_thoughts.append(ppl_with)
        self.perplexity_without_thoughts.append(ppl_without)

        # 2. Reasoning improvement
        improvement = (ppl_without - ppl_with) / ppl_without
        self.reasoning_improvement.append(improvement)

        # 3. Thought coherence
        if 'thought_embeddings' in outputs_with_thoughts:
            coherence = self._compute_thought_coherence(
                outputs_with_thoughts['thought_embeddings']
            )
            self.thought_coherence_scores.append(coherence)

        # 4. Attention diversity
        if 'thought_attention' in outputs_with_thoughts:
            diversity = self._compute_attention_diversity(
                outputs_with_thoughts['thought_attention']
            )
            self.attention_diversity.append(diversity)

        # 5. Thought utilization
        if 'thought_attention' in outputs_with_thoughts:
            utilization = self._compute_thought_utilization(
                outputs_with_thoughts['thought_attention']
            )
            self.thought_utilization.append(utilization)

    def _compute_perplexity(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """Compute perplexity from logits and targets"""
        log_probs = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(
            log_probs.view(-1, log_probs.size(-1)),
            targets.view(-1),
            reduction='mean'
        )
        return torch.exp(nll).item()

    def _compute_thought_coherence(
        self,
        thought_embeddings: torch.Tensor
    ) -> float:
        """Compute average coherence within thoughts"""
        batch_size, seq_len, thought_len, hidden_dim = thought_embeddings.shape

        coherence_scores = []
        for b in range(batch_size):
            for s in range(seq_len):
                thought = thought_embeddings[b, s]  # [thought_len, hidden_dim]

                # Compute pairwise cosine similarities
                similarities = []
                for i in range(thought_len):
                    for j in range(i + 1, thought_len):
                        sim = F.cosine_similarity(
                            thought[i:i+1], thought[j:j+1], dim=-1
                        )
                        similarities.append(sim.item())

                if similarities:
                    coherence_scores.append(np.mean(similarities))

        return np.mean(coherence_scores) if coherence_scores else 0.0

    def _compute_attention_diversity(
        self,
        attention_weights: torch.Tensor
    ) -> float:
        """Compute diversity of attention across thoughts"""
        # Compute entropy of attention distribution
        entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + 1e-8),
            dim=-1
        )
        return entropy.mean().item()

    def _compute_thought_utilization(
        self,
        attention_weights: torch.Tensor
    ) -> float:
        """Compute how much thoughts are actually used"""
        # Measure how concentrated attention is
        max_attention = torch.max(attention_weights, dim=-1)[0]
        return max_attention.mean().item()

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics for all metrics"""
        summary = {}

        if self.perplexity_with_thoughts:
            summary['avg_perplexity_with_thoughts'] = np.mean(self.perplexity_with_thoughts)
            summary['avg_perplexity_without_thoughts'] = np.mean(self.perplexity_without_thoughts)
            summary['avg_reasoning_improvement'] = np.mean(self.reasoning_improvement)

        if self.thought_coherence_scores:
            summary['avg_thought_coherence'] = np.mean(self.thought_coherence_scores)

        if self.attention_diversity:
            summary['avg_attention_diversity'] = np.mean(self.attention_diversity)

        if self.thought_utilization:
            summary['avg_thought_utilization'] = np.mean(self.thought_utilization)

        return summary


class GradientAnalyzer:
    """
    Utility for analyzing gradient flow through thoughts during training
    """

    def __init__(self):
        self.thought_gradients = []
        self.main_gradients = []
        self.gradient_ratios = []

    def analyze_gradients(self, model: nn.Module):
        """Analyze gradient magnitudes and flow"""
        thought_grad_norm = 0.0
        main_grad_norm = 0.0

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()

                if 'thought' in name.lower():
                    thought_grad_norm += grad_norm ** 2
                else:
                    main_grad_norm += grad_norm ** 2

        thought_grad_norm = thought_grad_norm ** 0.5
        main_grad_norm = main_grad_norm ** 0.5

        self.thought_gradients.append(thought_grad_norm)
        self.main_gradients.append(main_grad_norm)

        if main_grad_norm > 0:
            ratio = thought_grad_norm / main_grad_norm
            self.gradient_ratios.append(ratio)

        return {
            'thought_grad_norm': thought_grad_norm,
            'main_grad_norm': main_grad_norm,
            'gradient_ratio': ratio if main_grad_norm > 0 else 0.0
        }

    def get_gradient_summary(self) -> Dict[str, float]:
        """Get summary of gradient analysis"""
        return {
            'avg_thought_grad_norm': np.mean(self.thought_gradients),
            'avg_main_grad_norm': np.mean(self.main_gradients),
            'avg_gradient_ratio': np.mean(self.gradient_ratios),
            'gradient_ratio_std': np.std(self.gradient_ratios)
        }


# Training utilities
def create_thought_targets(input_ids: torch.Tensor, thought_length: int) -> torch.Tensor:
    """Create targets for thought generation"""
    batch_size, seq_len = input_ids.shape

    # Simple strategy: use next tokens as thought targets
    # In practice, this could be more sophisticated
    thought_targets = input_ids.unsqueeze(2).expand(-1, -1, thought_length)

    return thought_targets


def setup_training(
    model: nn.Module,
    train_dataset,
    val_dataset,
    config: TrainingConfig,
    device: torch.device
) -> QuietSTaRTrainer:
    """Setup complete training pipeline"""

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,  # Adjust based on memory
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Create trainer
    trainer = QuietSTaRTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    return trainer


# Example usage and training script
if __name__ == "__main__":
    # Example training configuration
    config = TrainingConfig(
        thought_loss_weight=1.0,
        coherence_loss_weight=0.5,
        prediction_loss_weight=1.0,
        learning_rate=1e-4,
        thought_length=20,
        num_thoughts_per_token=4,
        use_wandb=True
    )

    # This would be replaced with actual model and data loading
    print("Quiet-STaR training utilities initialized")
    print(f"Configuration: {config}")

    # Example of loss function usage
    vocab_size = 50257
    loss_fn = ReasoningLoss(config, vocab_size)

    # Example metrics
    metrics = EvaluationMetrics(config)

    print("Training components ready for Quiet-STaR implementation")