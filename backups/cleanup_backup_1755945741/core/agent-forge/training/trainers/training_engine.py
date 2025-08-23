"""
Training Engine for HRRM Models

Focused training logic separated from configuration and data handling.
Provides reusable training functionality with configurable behavior.
"""

import logging
from typing import Any, Protocol

import torch

from ..config.training_config import TrainingConfig

logger = logging.getLogger(__name__)


class ModelProtocol(Protocol):
    """Protocol for trainable models."""

    def train(self) -> None:
        """Set model to training mode."""
        ...

    def to(self, device: torch.device) -> Any:
        """Move model to device."""
        ...

    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through model."""
        ...


class OptimizerProtocol(Protocol):
    """Protocol for optimizers."""

    def zero_grad(self) -> None:
        """Zero gradients."""
        ...

    def step(self) -> None:
        """Optimization step."""
        ...


class TrainingEngine:
    """
    Engine for training HRRM models.

    Focused on training logic with configurable behavior.
    Uses dependency injection for configuration.
    """

    def __init__(self, config: TrainingConfig | None = None):
        self._config = config or TrainingConfig()

    async def train_model(
        self,
        model: ModelProtocol,
        optimizer: OptimizerProtocol,
        train_data: list[torch.Tensor],
        model_name: str,
        device: torch.device,
    ) -> float:
        """
        Train a model with the provided data.

        Args:
            model: Model to train
            optimizer: Optimizer for training
            train_data: Training data batches
            model_name: Name for logging purposes
            device: Device to train on

        Returns:
            Average training loss
        """
        logger.info(f"Training {model_name} for {self._config.EPOCHS} epochs on {device}")

        # Prepare model for training
        model.train()
        model.to(device)

        total_loss = 0.0
        step = 0

        for epoch in range(self._config.EPOCHS):
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(train_data):
                batch = batch.to(device)

                # Prepare input/target pairs for language modeling
                labels = batch[:, 1:].contiguous()
                inputs = batch[:, :-1].contiguous()

                optimizer.zero_grad()

                # Forward pass - handle different model types
                loss = await self._compute_loss(model, model_name, inputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Track metrics
                epoch_loss += loss.item()
                total_loss += loss.item()
                step += 1

                # Progress logging
                if step % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self._config.EPOCHS}, " f"Step {step}, Loss: {loss.item():.4f}")

            avg_epoch_loss = epoch_loss / len(train_data)
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

        avg_total_loss = total_loss / (self._config.EPOCHS * len(train_data))
        logger.info(f"{model_name} training completed. Average loss: {avg_total_loss:.4f}")

        return avg_total_loss

    async def _compute_loss(
        self, model: ModelProtocol, model_name: str, inputs: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for different model types.

        Args:
            model: The model
            model_name: Model name for type detection
            inputs: Input tensors
            labels: Target labels

        Returns:
            Computed loss tensor
        """
        try:
            if model_name in ["HRMPlanner", "HRMReasoner"]:
                # These models support labels in forward pass
                output = model(inputs, labels=labels)

                if hasattr(output, "loss"):
                    return output.loss
                else:
                    # Fallback to manual cross-entropy
                    return torch.nn.functional.cross_entropy(
                        output.logits.view(-1, output.logits.size(-1)), labels.view(-1)
                    )

            else:  # MemoryAsContextTiny and others
                # Standard language modeling approach
                output = model(inputs)

                if hasattr(output, "logits"):
                    logits = output.logits
                else:
                    logits = output  # Assume output is logits directly

                return torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        except Exception as e:
            logger.error(f"Loss computation failed for {model_name}: {e}")
            # Return a dummy loss to prevent training failure
            return torch.tensor(0.0, requires_grad=True)

    def get_training_info(self) -> dict[str, Any]:
        """Get information about training configuration."""
        return {
            "epochs": self._config.EPOCHS,
            "learning_rate": self._config.LEARNING_RATE,
            "device_preference": self._config.DEVICE_PREFERENCE,
        }
