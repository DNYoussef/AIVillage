"""
Telemetry encoding system for self-modeling and grok detection.
Encodes telemetry signals into learnable representations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .grokfast_ctrl import TelemetryState


class TelemetryEncoding(Enum):
    """Types of telemetry encoding."""

    RAW = "raw"  # Direct scalar values
    BINNED = "binned"  # Categorical binning
    EMBEDDING = "embedding"  # Learned embeddings
    POSITIONAL = "positional"  # Positional encoding style


@dataclass
class EncodedTelemetry:
    """Container for encoded telemetry features."""

    features: torch.Tensor
    raw_values: dict[str, float]
    encoding_type: TelemetryEncoding
    feature_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "features": self.features.detach().cpu().numpy().tolist(),
            "raw_values": self.raw_values,
            "encoding_type": self.encoding_type.value,
            "feature_names": self.feature_names,
        }


class TelemetryEncoder(nn.Module):
    """
    Encodes telemetry signals into feature representations.
    """

    def __init__(
        self,
        encoding_type: TelemetryEncoding = TelemetryEncoding.EMBEDDING,
        feature_dim: int = 32,
        num_bins: int = 10,
        normalize: bool = True,
    ):
        super().__init__()

        self.encoding_type = encoding_type
        self.feature_dim = feature_dim
        self.num_bins = num_bins
        self.normalize = normalize

        # Feature names in order
        self.feature_names = [
            "intrinsic_dimension",
            "slow_gradient_strength",
            "ema_cosine_similarity",
            "loss",
            "accuracy",
            "grad_norm",
            "step_normalized",
        ]

        self.num_features = len(self.feature_names)

        # Initialize encoding layers based on type
        if encoding_type == TelemetryEncoding.EMBEDDING:
            # Learned embeddings for each binned feature
            self.embeddings = nn.ModuleDict(
                {
                    name: nn.Embedding(num_bins, feature_dim // self.num_features)
                    for name in self.feature_names
                }
            )
            self.output_dim = feature_dim

        elif encoding_type == TelemetryEncoding.BINNED:
            # One-hot binned encoding
            self.output_dim = num_bins * self.num_features

        elif encoding_type == TelemetryEncoding.POSITIONAL:
            # Positional encoding style
            self.output_dim = feature_dim

        else:  # RAW
            self.output_dim = self.num_features

        # Normalization statistics (running estimates)
        self.register_buffer("feature_means", torch.zeros(self.num_features))
        self.register_buffer("feature_stds", torch.ones(self.num_features))
        self.register_buffer("num_updates", torch.zeros(1))

        # Binning boundaries (learned or fixed)
        if encoding_type in [TelemetryEncoding.EMBEDDING, TelemetryEncoding.BINNED]:
            self.register_buffer("bin_boundaries", self._initialize_bins())

    def _initialize_bins(self) -> torch.Tensor:
        """Initialize binning boundaries for each feature."""
        # Reasonable ranges for different telemetry features
        boundaries = []

        # Intrinsic dimension: [0, 1]
        boundaries.append(torch.linspace(0, 1, self.num_bins + 1))

        # Slow gradient strength: [0, 1]
        boundaries.append(torch.linspace(0, 1, self.num_bins + 1))

        # EMA cosine similarity: [-1, 1]
        boundaries.append(torch.linspace(-1, 1, self.num_bins + 1))

        # Loss: [0, 10] (log scale might be better)
        boundaries.append(torch.logspace(-3, 1, self.num_bins + 1))

        # Accuracy: [0, 1]
        boundaries.append(torch.linspace(0, 1, self.num_bins + 1))

        # Grad norm: [0, 10] (log scale)
        boundaries.append(torch.logspace(-3, 1, self.num_bins + 1))

        # Step (normalized): [0, 1]
        boundaries.append(torch.linspace(0, 1, self.num_bins + 1))

        # Stack into tensor [num_features, num_bins + 1]
        return torch.stack(boundaries)

    def _extract_features(
        self, telemetry: TelemetryState, max_step: int = 100000
    ) -> torch.Tensor:
        """Extract raw feature vector from telemetry."""
        features = torch.tensor(
            [
                telemetry.intrinsic_dimension,
                telemetry.slow_gradient_strength,
                telemetry.ema_cosine_similarity,
                telemetry.loss,
                telemetry.accuracy,
                telemetry.grad_norm,
                min(telemetry.step / max_step, 1.0),  # Normalized step
            ],
            dtype=torch.float32,
        )

        return features

    def _update_normalization_stats(self, features: torch.Tensor):
        """Update running normalization statistics."""
        if not self.normalize:
            return

        # Update running mean and std
        batch_mean = features.mean(dim=0)
        batch_var = features.var(dim=0, unbiased=False)

        # Running update
        n = self.num_updates.item()
        alpha = 1.0 / (n + 1)

        self.feature_means = (1 - alpha) * self.feature_means + alpha * batch_mean

        # Update variance with Welford's method
        if n > 0:
            delta = batch_mean - self.feature_means
            self.feature_stds = torch.sqrt(
                (1 - alpha) * self.feature_stds**2
                + alpha * batch_var
                + alpha * (1 - alpha) * delta**2
            )
        else:
            self.feature_stds = torch.sqrt(batch_var + 1e-8)

        self.num_updates += 1

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features using running statistics."""
        if not self.normalize:
            return features

        return (features - self.feature_means) / (self.feature_stds + 1e-8)

    def _bin_features(self, features: torch.Tensor) -> torch.Tensor:
        """Convert features to bin indices."""
        # features: [batch_size, num_features]
        # bin_boundaries: [num_features, num_bins + 1]

        batch_size = features.shape[0]
        bin_indices = torch.zeros(
            (batch_size, self.num_features), dtype=torch.long, device=features.device
        )

        for i, feature_vals in enumerate(features.t()):
            # Find bin for each feature value
            boundaries = self.bin_boundaries[i]  # [num_bins + 1]

            # Use searchsorted to find bin indices
            bin_idx = torch.searchsorted(boundaries[1:-1], feature_vals, right=False)
            bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)

            bin_indices[:, i] = bin_idx

        return bin_indices

    def _positional_encode(self, features: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding style transformation."""
        # features: [batch_size, num_features]
        batch_size = features.shape[0]

        # Create sinusoidal encodings
        encodings = []

        for i in range(self.num_features):
            feat_vals = features[:, i : i + 1]  # [batch_size, 1]

            # Create frequency components
            freqs = torch.exp(
                torch.arange(
                    0,
                    self.feature_dim // (2 * self.num_features),
                    dtype=torch.float32,
                    device=features.device,
                )
                * -(np.log(10000.0) / (self.feature_dim // (2 * self.num_features)))
            )

            # Apply sinusoidal encoding
            angles = feat_vals * freqs.unsqueeze(0)  # [batch_size, dim//2]
            sin_enc = torch.sin(angles)
            cos_enc = torch.cos(angles)

            encodings.append(torch.cat([sin_enc, cos_enc], dim=1))

        # Concatenate all feature encodings
        return torch.cat(encodings, dim=1)  # [batch_size, feature_dim]

    def forward(
        self, telemetry_batch: list[TelemetryState], update_stats: bool = True
    ) -> EncodedTelemetry:
        """
        Encode batch of telemetry states.

        Args:
            telemetry_batch: List of telemetry states
            update_stats: Whether to update normalization statistics

        Returns:
            Encoded telemetry features
        """
        batch_size = len(telemetry_batch)

        # Extract raw features
        raw_features = torch.stack(
            [self._extract_features(t) for t in telemetry_batch]
        )  # [batch_size, num_features]

        # Update normalization statistics if in training mode
        if self.training and update_stats:
            self._update_normalization_stats(raw_features)

        # Normalize features
        normalized_features = self._normalize_features(raw_features)

        # Encode based on type
        if self.encoding_type == TelemetryEncoding.RAW:
            encoded = normalized_features

        elif self.encoding_type == TelemetryEncoding.BINNED:
            # Bin features and create one-hot encoding
            bin_indices = self._bin_features(
                normalized_features
            )  # [batch_size, num_features]

            one_hot = F.one_hot(
                bin_indices, num_classes=self.num_bins
            )  # [batch_size, num_features, num_bins]
            encoded = one_hot.reshape(
                batch_size, -1
            ).float()  # [batch_size, num_features * num_bins]

        elif self.encoding_type == TelemetryEncoding.EMBEDDING:
            # Bin features and get embeddings
            bin_indices = self._bin_features(
                normalized_features
            )  # [batch_size, num_features]

            embeddings = []
            for i, feature_name in enumerate(self.feature_names):
                feat_bins = bin_indices[:, i]  # [batch_size]
                feat_emb = self.embeddings[feature_name](
                    feat_bins
                )  # [batch_size, emb_dim]
                embeddings.append(feat_emb)

            encoded = torch.cat(embeddings, dim=1)  # [batch_size, feature_dim]

        elif self.encoding_type == TelemetryEncoding.POSITIONAL:
            encoded = self._positional_encode(normalized_features)

        # Collect raw values for first item (for inspection)
        raw_values = telemetry_batch[0].to_dict() if telemetry_batch else {}

        return EncodedTelemetry(
            features=encoded,
            raw_values=raw_values,
            encoding_type=self.encoding_type,
            feature_names=self.feature_names,
        )

    def encode_single(self, telemetry: TelemetryState) -> EncodedTelemetry:
        """Encode single telemetry state."""
        return self.forward([telemetry], update_stats=False)


class TelemetryPredictor(nn.Module):
    """
    Predicts future telemetry states from current encodings.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_steps_ahead: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_steps_ahead = num_steps_ahead

        # Prediction network
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 7 * num_steps_ahead),  # 7 telemetry features
        )

        # Confidence estimator
        self.confidence = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_steps_ahead),
            nn.Sigmoid(),
        )

    def forward(
        self, encoded_telemetry: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict future telemetry states.

        Args:
            encoded_telemetry: [batch_size, input_dim]

        Returns:
            Tuple of (predictions, confidence)
            predictions: [batch_size, num_steps_ahead, 7]
            confidence: [batch_size, num_steps_ahead]
        """
        batch_size = encoded_telemetry.shape[0]

        # Predict future values
        pred_flat = self.predictor(
            encoded_telemetry
        )  # [batch_size, 7 * num_steps_ahead]
        predictions = pred_flat.reshape(batch_size, self.num_steps_ahead, 7)

        # Predict confidence
        confidence = self.confidence(encoded_telemetry)  # [batch_size, num_steps_ahead]

        return predictions, confidence


class TelemetryAnomalyDetector(nn.Module):
    """
    Detects anomalous telemetry patterns that might indicate interesting training dynamics.
    """

    def __init__(self, input_dim: int, threshold_percentile: float = 95.0):
        super().__init__()

        self.input_dim = input_dim
        self.threshold_percentile = threshold_percentile

        # Autoencoder for anomaly detection
        bottleneck_dim = max(input_dim // 4, 8)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
        )

        # Running statistics for anomaly threshold
        self.register_buffer("reconstruction_errors", torch.tensor([]))
        self.register_buffer("anomaly_threshold", torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.

        Returns:
            Tuple of (reconstructed, reconstruction_error)
        """
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)

        # Compute reconstruction error
        error = F.mse_loss(reconstructed, x, reduction="none").mean(dim=1)

        return reconstructed, error

    def update_threshold(self, errors: torch.Tensor):
        """Update anomaly detection threshold."""
        # Add new errors to history
        new_errors = torch.cat([self.reconstruction_errors, errors.detach().cpu()])

        # Keep only recent errors (limit memory)
        if len(new_errors) > 10000:
            new_errors = new_errors[-10000:]

        self.reconstruction_errors = new_errors

        # Update threshold to percentile
        if len(new_errors) > 10:
            self.anomaly_threshold = torch.quantile(
                new_errors, self.threshold_percentile / 100.0
            )

    def detect_anomalies(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Detect anomalies in telemetry.

        Returns:
            Tuple of (is_anomaly, anomaly_scores)
        """
        _, errors = self.forward(x)

        # Update threshold if in training mode
        if self.training:
            self.update_threshold(errors)

        # Classify anomalies
        is_anomaly = errors > self.anomaly_threshold
        anomaly_scores = errors / (self.anomaly_threshold + 1e-8)

        return is_anomaly, anomaly_scores


# Factory functions


def create_telemetry_encoder(
    encoding_type: str = "embedding", feature_dim: int = 32, num_bins: int = 10
) -> TelemetryEncoder:
    """Create telemetry encoder with specified configuration."""
    encoding_enum = TelemetryEncoding(encoding_type.lower())

    return TelemetryEncoder(
        encoding_type=encoding_enum,
        feature_dim=feature_dim,
        num_bins=num_bins,
        normalize=True,
    )


def create_telemetry_predictor(
    input_dim: int, hidden_dim: int = 128
) -> TelemetryPredictor:
    """Create telemetry predictor."""
    return TelemetryPredictor(
        input_dim=input_dim, hidden_dim=hidden_dim, num_steps_ahead=10
    )


def create_anomaly_detector(input_dim: int) -> TelemetryAnomalyDetector:
    """Create telemetry anomaly detector."""
    return TelemetryAnomalyDetector(input_dim=input_dim)


if __name__ == "__main__":
    # Demo telemetry encoding system
    print("📊 Telemetry Encoding System Demo")
    print("=" * 50)

    # Create mock telemetry states
    mock_telemetry = [
        TelemetryState(
            intrinsic_dimension=0.8 - i * 0.1,
            slow_gradient_strength=0.1 + i * 0.05,
            ema_cosine_similarity=0.2 + i * 0.1,
            loss=2.0 - i * 0.2,
            accuracy=i * 0.1,
            grad_norm=1.0 - i * 0.05,
            step=i * 100,
        )
        for i in range(10)
    ]

    # Test different encoding types
    encoding_types = ["raw", "binned", "embedding", "positional"]

    for enc_type in encoding_types:
        print(f"\n{enc_type.upper()} Encoding:")

        encoder = create_telemetry_encoder(enc_type, feature_dim=32, num_bins=5)
        encoded = encoder(mock_telemetry[:5])  # Batch of 5

        print("   Input shape: 5 telemetry states")
        print(f"   Output shape: {encoded.features.shape}")
        print(f"   Feature names: {encoded.feature_names[:3]}...")
        print(
            f"   Raw values (first): ID={encoded.raw_values['id']:.3f}, "
            f"S_slow={encoded.raw_values['s_slow']:.3f}"
        )

    print()

    # Test predictor
    print("🔮 Telemetry Prediction:")
    encoder = create_telemetry_encoder("embedding", feature_dim=64)
    predictor = create_telemetry_predictor(encoder.output_dim)

    encoded = encoder(mock_telemetry[:3])
    predictions, confidence = predictor(encoded.features)

    print(f"   Input: {encoded.features.shape}")
    print(f"   Predictions: {predictions.shape}")  # [3, 10, 7]
    print(f"   Confidence: {confidence.shape}")  # [3, 10]
    print(f"   Avg confidence: {confidence.mean().item():.3f}")

    print()

    # Test anomaly detection
    print("🚨 Anomaly Detection:")
    anomaly_detector = create_anomaly_detector(encoder.output_dim)

    # Train on normal data
    normal_encoded = encoder(mock_telemetry[:8])
    anomaly_detector.train()

    for _ in range(5):  # Quick training
        reconstructed, errors = anomaly_detector(normal_encoded.features)
        loss = errors.mean()
        print(f"   Training reconstruction loss: {loss.item():.4f}")

    # Test on anomalous data (create outlier)
    anomalous_telemetry = TelemetryState(
        intrinsic_dimension=1.5,  # Anomalous value
        slow_gradient_strength=0.9,  # Anomalous value
        ema_cosine_similarity=-0.8,
        loss=0.001,
        accuracy=0.99,
        grad_norm=0.001,
        step=5000,
    )

    anomaly_detector.eval()
    test_encoded = encoder([anomalous_telemetry])
    is_anomaly, scores = anomaly_detector.detect_anomalies(test_encoded.features)

    print(f"   Anomaly detected: {is_anomaly.item()}")
    print(f"   Anomaly score: {scores.item():.2f}")

    print()
    print("✅ Telemetry Encoding System Demo Complete")
    print()
    print("Key Features Demonstrated:")
    print("  • Multiple encoding strategies (raw, binned, embedding, positional)")
    print("  • Learned normalization and adaptive binning")
    print("  • Future telemetry state prediction")
    print("  • Anomaly detection for unusual training dynamics")
    print("  • Integration-ready feature extraction")
