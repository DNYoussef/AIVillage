"""
Self-modeling head for model self-awareness and compression.
Predicts internal activations to improve representation quality.
Extended with temperature awareness and grok stage prediction.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SelfModelHead(nn.Module):
    """
    Auxiliary head that predicts the model's own hidden states.

    This encourages the model to develop compressed, predictable representations
    and improves stability during training. Based on self-modeling networks research.
    """

    def __init__(
        self,
        tap_layers: list[int],
        hidden_dim: int,
        projection_dim: int = 256,
        num_prediction_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            tap_layers: Which layer indices to predict
            hidden_dim: Dimension of the model's hidden states
            projection_dim: Dimension of projection for prediction
            num_prediction_layers: Number of MLP layers for prediction
            dropout: Dropout rate
        """
        super().__init__()

        self.tap_layers = sorted(tap_layers)
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim

        # Projection layers for each tapped layer
        self.projectors = nn.ModuleDict(
            {str(layer): nn.Linear(hidden_dim, projection_dim) for layer in tap_layers}
        )

        # Prediction networks for each target layer
        self.predictors = nn.ModuleDict()
        for target_layer in tap_layers:
            layers = []
            for i in range(num_prediction_layers):
                if i == 0:
                    # Predict from previous layer
                    in_dim = projection_dim
                else:
                    in_dim = projection_dim

                layers.extend(
                    [
                        nn.Linear(in_dim, projection_dim),
                        nn.LayerNorm(projection_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )

            # Final projection back to hidden dimension
            layers.append(nn.Linear(projection_dim, hidden_dim))

            self.predictors[str(target_layer)] = nn.Sequential(*layers)

        # Loss weighting for each layer
        self.layer_weights = nn.Parameter(torch.ones(len(tap_layers)) / len(tap_layers))

        self.init_weights()

    def init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, tap_activations: dict[int, torch.Tensor]
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """
        Predict activations and compute loss.

        Args:
            tap_activations: Dict mapping layer indices to their activations
                            Shape: [batch_size, seq_len, hidden_dim]

        Returns:
            Tuple of (predictions dict, total loss)
        """
        predictions = {}
        losses = []

        # Sort layers to ensure consistent prediction order
        sorted_layers = sorted(tap_activations.keys())

        for i, target_layer in enumerate(sorted_layers):
            if target_layer not in self.tap_layers:
                continue

            # Get source layer (predict from previous tapped layer)
            source_idx = max(0, i - 1)
            source_layer = sorted_layers[source_idx]

            if source_layer == target_layer:
                # Can't predict from self, skip
                continue

            source_acts = tap_activations[source_layer]
            target_acts = tap_activations[target_layer]

            # Handle different tensor shapes
            if source_acts.dim() == 3:
                # [batch, seq, hidden]
                batch_size, seq_len, _ = source_acts.shape
                source_flat = source_acts.reshape(-1, self.hidden_dim)
                target_flat = target_acts.reshape(-1, self.hidden_dim)
            else:
                # [batch, hidden]
                source_flat = source_acts
                target_flat = target_acts

            # Project source activations
            source_proj = self.projectors[str(source_layer)](source_flat)

            # Predict target activations
            pred = self.predictors[str(target_layer)](source_proj)

            # Reshape if needed
            if source_acts.dim() == 3:
                pred = pred.reshape(batch_size, seq_len, -1)
                predictions[target_layer] = pred
            else:
                predictions[target_layer] = pred

            # Compute loss (MSE)
            layer_loss = F.mse_loss(pred, target_flat.detach())
            losses.append(layer_loss)

        if losses:
            # Weighted sum of losses
            weights = F.softmax(self.layer_weights, dim=0)
            total_loss = sum(w * l for w, l in zip(weights, losses, strict=False))
        else:
            total_loss = torch.tensor(
                0.0, device=tap_activations[sorted_layers[0]].device
            )

        return predictions, total_loss

    def loss(
        self, tap_activations: dict[int, torch.Tensor], reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute self-modeling loss.

        Args:
            tap_activations: Activations from tapped layers
            reduction: Loss reduction method

        Returns:
            Self-modeling loss
        """
        _, loss = self.forward(tap_activations)
        return loss


class TempCurriculum:
    """
    Temperature-aware curriculum for training across different generation temperatures.
    Helps model become aware of its own temperature-dependent behavior.
    """

    def __init__(
        self,
        temperature_bins: list[tuple[float, float]] = None,
        samples_per_bin: int = 100,
        overlap_ratio: float = 0.1,
    ):
        """
        Args:
            temperature_bins: List of (min_temp, max_temp) ranges
            samples_per_bin: Number of samples to generate per bin
            overlap_ratio: Overlap between adjacent bins
        """
        if temperature_bins is None:
            # Default bins from conservative to creative
            temperature_bins = [
                (0.0, 0.1),  # Deterministic
                (0.1, 0.3),  # Conservative
                (0.3, 0.5),  # Balanced
                (0.5, 0.7),  # Explorative
                (0.7, 0.9),  # Creative
                (0.9, 1.0),  # Maximum diversity
            ]

        self.bins = temperature_bins
        self.samples_per_bin = samples_per_bin
        self.overlap_ratio = overlap_ratio

        # Create overlapping bins for round 2
        self.overlapping_bins = self._create_overlapping_bins()

        # Track curriculum progress
        self.current_round = 1
        self.current_bin_idx = 0
        self.bin_accuracies = {i: [] for i in range(len(self.bins))}

    def _create_overlapping_bins(self) -> list[tuple[float, float]]:
        """Create overlapping temperature bins for round 2."""
        overlapping = []

        for i in range(len(self.bins) - 1):
            min1, max1 = self.bins[i]
            min2, max2 = self.bins[i + 1]

            # Create overlap region
            overlap_start = max1 - (max1 - min1) * self.overlap_ratio
            overlap_end = min2 + (max2 - min2) * self.overlap_ratio

            overlapping.append((overlap_start, overlap_end))

        return overlapping

    def generate_samples(
        self, model, tokenizer, prompts: list[str], device: str = "cuda"
    ) -> list[dict]:
        """
        Generate samples across temperature bins.

        Returns:
            List of dicts with 'prompt', 'temperature', 'output', 'bin_idx'
        """
        samples = []

        bins_to_use = self.bins if self.current_round == 1 else self.overlapping_bins

        for bin_idx, (min_temp, max_temp) in enumerate(bins_to_use):
            for _ in range(self.samples_per_bin):
                # Sample temperature from bin
                temperature = np.random.uniform(min_temp, max_temp)

                # Sample prompt
                prompt = np.random.choice(prompts)

                # Generate with temperature
                with torch.no_grad():
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)

                    outputs = model.generate(
                        **inputs,
                        temperature=max(temperature, 0.01),  # Avoid zero
                        max_new_tokens=100,
                        do_sample=temperature > 0.01,
                        top_p=0.95 if temperature > 0.3 else 1.0,
                    )

                    output_text = tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    )

                samples.append(
                    {
                        "prompt": prompt,
                        "temperature": temperature,
                        "output": output_text,
                        "bin_idx": bin_idx,
                        "round": self.current_round,
                    }
                )

        return samples

    def compute_consistency_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """
        Compute KL divergence loss for temperature consistency.

        Student should match teacher's distribution at given temperature.
        """
        # Apply temperature scaling
        student_probs = F.softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        # KL divergence
        kl_loss = F.kl_div(student_probs.log(), teacher_probs, reduction="batchmean")

        return kl_loss * (temperature**2)  # Scale by T^2 as in distillation

    def update_bin_accuracy(self, bin_idx: int, accuracy: float):
        """Track accuracy for each temperature bin."""
        self.bin_accuracies[bin_idx].append(accuracy)

    def should_advance_round(self) -> bool:
        """Check if we should move to round 2 (overlapping bins)."""
        if self.current_round >= 2:
            return False

        # Advance if all bins have sufficient samples and good accuracy
        for accuracies in self.bin_accuracies.values():
            if len(accuracies) < 10:  # Need at least 10 measurements
                return False
            if np.mean(accuracies[-10:]) < 0.7:  # Recent accuracy > 70%
                return False

        return True

    def advance_round(self):
        """Move to the next round of temperature curriculum."""
        if self.current_round < 2:
            self.current_round += 1
            self.current_bin_idx = 0
            logger.info(
                f"Advanced to temperature curriculum round {self.current_round}"
            )


class StageClassifier:
    """
    Classifies the current training stage based on multiple signals.
    Helps coordinate different training strategies.
    """

    def __init__(self):
        self.stage_history = []
        self.stage_transitions = []
        self.metrics_history = {
            "loss": [],
            "accuracy": [],
            "grad_norm": [],
            "id": [],
        }

    def classify(
        self,
        loss: float,
        accuracy: float,
        grad_norm: float,
        id_value: float,
        ema_cos: float,
        step: int,
    ) -> str:
        """
        Classify current training stage.

        Stages:
        - warmup: Initial training, high loss
        - pre_grok: Memorization phase
        - grok_onset: Beginning of grokking
        - grok_active: Active grokking
        - consolidation: Stabilizing after grokking
        - refined: Final refinement
        """
        # Update history
        self.metrics_history["loss"].append(loss)
        self.metrics_history["accuracy"].append(accuracy)
        self.metrics_history["grad_norm"].append(grad_norm)
        self.metrics_history["id"].append(id_value)

        # Warmup stage
        if step < 1000:
            stage = "warmup"

        # Check for grokking signals
        elif len(self.metrics_history["loss"]) > 100:
            recent_loss = self.metrics_history["loss"][-100:]
            recent_acc = self.metrics_history["accuracy"][-100:]

            loss_dropping = np.polyfit(range(100), recent_loss, 1)[0] < -0.001
            acc_rising = np.polyfit(range(100), recent_acc, 1)[0] > 0.001

            if loss > 1.0 and accuracy < 0.3:
                stage = "pre_grok"
            elif loss_dropping and acc_rising and ema_cos > 0.3:
                if accuracy < 0.5:
                    stage = "grok_onset"
                else:
                    stage = "grok_active"
            elif accuracy > 0.8 and abs(grad_norm) < 0.1:
                stage = "consolidation"
            elif accuracy > 0.9:
                stage = "refined"
            else:
                # Default to previous stage
                stage = self.stage_history[-1] if self.stage_history else "pre_grok"
        else:
            stage = "pre_grok"

        # Track transitions
        if self.stage_history and self.stage_history[-1] != stage:
            self.stage_transitions.append(
                {
                    "step": step,
                    "from": self.stage_history[-1],
                    "to": stage,
                }
            )
            logger.info(
                f"Stage transition at step {step}: {self.stage_history[-1]} -> {stage}"
            )

        self.stage_history.append(stage)
        return stage

    def get_stage_duration(self, stage: str) -> int:
        """Get how many steps have been spent in a given stage."""
        return self.stage_history.count(stage)

    def get_transitions(self) -> list[dict]:
        """Get all stage transitions."""
        return self.stage_transitions


class TempInferHead(nn.Module):
    """
    Temperature inference head for temperature bin classification.
    Learns to predict which temperature bin generated a given text.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_temp_bins: int = 6,  # Default: LOW, MID, HIGH + overlapping versions
        projection_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_temp_bins = num_temp_bins
        self.projection_dim = projection_dim

        # Temperature classification head
        self.temp_projector = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim // 2),
            nn.LayerNorm(projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, num_temp_bins),
        )

        # Temperature value regression head (optional)
        self.temp_regressor = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, 1),
            nn.Sigmoid(),  # Temperature in [0, 1] range
        )

        self.init_weights()

    def init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for temperature inference.

        Args:
            hidden_states: [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]

        Returns:
            Tuple of (temp_bin_logits, temp_value_pred)
        """
        # Handle sequence dimension
        if hidden_states.dim() == 3:
            # Use last token or mean pooling
            pooled = hidden_states.mean(dim=1)  # Mean pooling
        else:
            pooled = hidden_states

        # Temperature bin classification
        temp_bin_logits = self.temp_projector(pooled)

        # Temperature value regression
        temp_value = self.temp_regressor(pooled).squeeze(-1)

        return temp_bin_logits, temp_value

    def predict_temperature_bin(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict temperature bin indices."""
        temp_bin_logits, _ = self.forward(hidden_states)
        return torch.argmax(temp_bin_logits, dim=-1)

    def predict_temperature_value(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict continuous temperature values."""
        _, temp_value = self.forward(hidden_states)
        return temp_value * 1.5  # Scale to [0, 1.5] range


class StageHead(nn.Module):
    """
    Grok stage prediction head.
    Predicts current grokking stage: pre/onset/consolidate.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_stages: int = 3,  # pre, onset, consolidate
        projection_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_stages = num_stages
        self.projection_dim = projection_dim

        # Stage classification head
        self.stage_classifier = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim // 2),
            nn.LayerNorm(projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, num_stages),
        )

        # Grok confidence head
        self.grok_confidence = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim // 2),
            nn.LayerNorm(projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.init_weights()

    def init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for stage prediction.

        Args:
            hidden_states: [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]

        Returns:
            Tuple of (stage_logits, grok_confidence)
        """
        # Handle sequence dimension
        if hidden_states.dim() == 3:
            # Use last token for stage prediction
            pooled = hidden_states[:, -1, :]  # Last token
        else:
            pooled = hidden_states

        # Stage classification
        stage_logits = self.stage_classifier(pooled)

        # Grok confidence
        confidence = self.grok_confidence(pooled).squeeze(-1)

        return stage_logits, confidence

    def predict_stage(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict grok stage indices."""
        stage_logits, _ = self.forward(hidden_states)
        return torch.argmax(stage_logits, dim=-1)

    def get_grok_confidence(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get confidence in grok prediction."""
        _, confidence = self.forward(hidden_states)
        return confidence


class MultiHeadSelfModel(nn.Module):
    """
    Multi-head self-modeling system combining original self-modeling,
    temperature inference, and grok stage prediction.
    """

    def __init__(
        self,
        tap_layers: list[int],
        hidden_dim: int,
        projection_dim: int = 256,
        num_temp_bins: int = 6,
        num_stages: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.tap_layers = tap_layers
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim

        # Original self-modeling head
        self.self_model_head = SelfModelHead(
            tap_layers=tap_layers,
            hidden_dim=hidden_dim,
            projection_dim=projection_dim,
            dropout=dropout,
        )

        # Temperature inference head
        self.temp_infer_head = TempInferHead(
            hidden_dim=hidden_dim,
            num_temp_bins=num_temp_bins,
            projection_dim=projection_dim // 2,
            dropout=dropout,
        )

        # Grok stage head
        self.stage_head = StageHead(
            hidden_dim=hidden_dim,
            num_stages=num_stages,
            projection_dim=projection_dim // 2,
            dropout=dropout,
        )

        # Loss weights
        self.loss_weights = nn.Parameter(
            torch.tensor([1.0, 0.5, 0.5])
        )  # self_model, temp, stage

    def forward(
        self,
        tap_activations: dict[int, torch.Tensor],
        temp_bin_labels: torch.Tensor | None = None,
        stage_labels: torch.Tensor | None = None,
        temp_values: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through all heads.

        Args:
            tap_activations: Dict mapping layer indices to activations
            temp_bin_labels: Ground truth temperature bin labels
            stage_labels: Ground truth stage labels
            temp_values: Ground truth temperature values

        Returns:
            Dict with predictions and losses
        """
        results = {}

        # Original self-modeling
        self_model_preds, self_model_loss = self.self_model_head(tap_activations)
        results["self_model_preds"] = self_model_preds
        results["self_model_loss"] = self_model_loss

        # Use final layer activations for classification heads
        final_layer = max(tap_activations.keys())
        final_acts = tap_activations[final_layer]

        # Temperature inference
        temp_bin_logits, temp_value_preds = self.temp_infer_head(final_acts)
        results["temp_bin_logits"] = temp_bin_logits
        results["temp_value_preds"] = temp_value_preds

        # Stage prediction
        stage_logits, grok_confidence = self.stage_head(final_acts)
        results["stage_logits"] = stage_logits
        results["grok_confidence"] = grok_confidence

        # Compute losses if labels provided
        losses = [self_model_loss]

        if temp_bin_labels is not None:
            temp_bin_loss = F.cross_entropy(temp_bin_logits, temp_bin_labels)
            results["temp_bin_loss"] = temp_bin_loss
            losses.append(temp_bin_loss)
        else:
            losses.append(torch.tensor(0.0, device=self_model_loss.device))

        if stage_labels is not None:
            stage_loss = F.cross_entropy(stage_logits, stage_labels)
            results["stage_loss"] = stage_loss
            losses.append(stage_loss)
        else:
            losses.append(torch.tensor(0.0, device=self_model_loss.device))

        # Temperature value regression loss
        if temp_values is not None:
            temp_value_loss = F.mse_loss(temp_value_preds, temp_values)
            results["temp_value_loss"] = temp_value_loss
            # Add to existing temp loss
            if "temp_bin_loss" in results:
                results["temp_bin_loss"] = (
                    results["temp_bin_loss"] + 0.1 * temp_value_loss
                )

        # Combined loss
        weights = F.softmax(self.loss_weights, dim=0)
        total_loss = sum(w * l for w, l in zip(weights, losses, strict=False))
        results["total_loss"] = total_loss

        return results

    def predict_temperature_properties(
        self, tap_activations: dict[int, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Predict temperature-related properties."""
        final_layer = max(tap_activations.keys())
        final_acts = tap_activations[final_layer]

        temp_bin_logits, temp_values = self.temp_infer_head(final_acts)
        temp_bins = torch.argmax(temp_bin_logits, dim=-1)

        return {
            "predicted_temp_bins": temp_bins,
            "predicted_temp_values": temp_values,
            "temp_bin_probs": F.softmax(temp_bin_logits, dim=-1),
        }

    def predict_grok_stage(
        self, tap_activations: dict[int, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Predict grokking stage."""
        final_layer = max(tap_activations.keys())
        final_acts = tap_activations[final_layer]

        stage_logits, grok_confidence = self.stage_head(final_acts)
        stages = torch.argmax(stage_logits, dim=-1)

        return {
            "predicted_stages": stages,
            "stage_probs": F.softmax(stage_logits, dim=-1),
            "grok_confidence": grok_confidence,
        }
