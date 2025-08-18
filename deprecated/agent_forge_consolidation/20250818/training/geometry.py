"""
Geometry probe for intrinsic dimension and correlation dimension tracking.
Implements the sensing layer for grokking detection.
"""

import logging

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class GeometryProbe:
    """
    Computes intrinsic dimension (ID) and correlation dimension (d) of
    neural activations to detect phase transitions during training.

    Key metrics:
    - ID: Effective dimensionality of the representation manifold
    - d: Fractal/correlation dimension indicating complexity
    """

    def __init__(
        self,
        layer_ids: list[int] = None,
        sample_size: int = 1024,
        id_method: str = "pca",  # pca, mle, or twonn
        correlation_scales: list[float] = None,
    ):
        self.layer_ids = layer_ids or [4, 12, 24]  # Sample early, mid, late layers
        self.sample_size = sample_size
        self.id_method = id_method
        self.correlation_scales = correlation_scales or [0.01, 0.1, 1.0, 10.0]

        # Cache for historical values
        self.history = {
            "id": {layer: [] for layer in self.layer_ids},
            "d": {layer: [] for layer in self.layer_ids},
        }

    def compute(
        self, activations: dict[int, torch.Tensor]
    ) -> tuple[dict[int, float], dict[int, float]]:
        """
        Compute ID and d for specified layers.

        Args:
            activations: Dict mapping layer indices to activation tensors
                        Shape: [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]

        Returns:
            Tuple of (id_by_layer, d_by_layer)
        """
        id_by_layer = {}
        d_by_layer = {}

        for layer_id in self.layer_ids:
            if layer_id not in activations:
                logger.warning(f"Layer {layer_id} not found in activations")
                continue

            acts = activations[layer_id]

            # Flatten to 2D if needed
            if acts.dim() > 2:
                acts = acts.reshape(-1, acts.shape[-1])

            # Sample if too large
            if acts.shape[0] > self.sample_size:
                indices = torch.randperm(acts.shape[0])[: self.sample_size]
                acts = acts[indices]

            # Move to CPU and convert to numpy
            acts_np = acts.detach().cpu().numpy()

            # Compute intrinsic dimension
            id_value = self._compute_intrinsic_dimension(acts_np)
            id_by_layer[layer_id] = id_value

            # Compute correlation dimension
            d_value = self._compute_correlation_dimension(acts_np)
            d_by_layer[layer_id] = d_value

            # Update history
            self.history["id"][layer_id].append(id_value)
            self.history["d"][layer_id].append(d_value)

        return id_by_layer, d_by_layer

    def _compute_intrinsic_dimension(self, X: np.ndarray) -> float:
        """
        Compute intrinsic dimension using specified method.
        """
        if self.id_method == "pca":
            return self._id_pca(X)
        elif self.id_method == "mle":
            return self._id_mle(X)
        elif self.id_method == "twonn":
            return self._id_twonn(X)
        else:
            raise ValueError(f"Unknown ID method: {self.id_method}")

    def _id_pca(self, X: np.ndarray) -> float:
        """
        PCA-based intrinsic dimension estimation.
        ID = effective number of principal components.
        """
        if X.shape[0] < 3 or X.shape[1] < 2:
            return 1.0

        try:
            pca = PCA()
            pca.fit(X)

            # Compute participation ratio
            explained_var = pca.explained_variance_ratio_
            explained_var = explained_var[explained_var > 1e-10]  # Filter tiny values

            if len(explained_var) == 0:
                return 1.0

            # Participation ratio: (sum(p_i))^2 / sum(p_i^2)
            pr = (np.sum(explained_var) ** 2) / np.sum(explained_var**2)

            return float(pr)
        except Exception as e:
            logger.warning(f"PCA ID computation failed: {e}")
            return 1.0

    def _id_mle(self, X: np.ndarray, k: int = 10) -> float:
        """
        Maximum Likelihood Estimation of intrinsic dimension.
        Based on k-nearest neighbor distances.
        """
        n = X.shape[0]
        if n < k + 1:
            return 1.0

        try:
            # Compute pairwise distances
            distances = cdist(X, X, metric="euclidean")

            # Get k-nearest neighbor distances (excluding self)
            distances.sort(axis=1)
            knn_distances = distances[:, 1 : k + 1]  # Exclude self (distance 0)

            # MLE estimator
            log_ratios = np.log(knn_distances[:, -1] / knn_distances[:, :-1])
            id_estimates = 1.0 / np.mean(log_ratios, axis=1)

            # Remove outliers and average
            id_estimates = id_estimates[np.isfinite(id_estimates)]
            if len(id_estimates) == 0:
                return 1.0

            # Robust mean (trim outliers)
            id_estimates = np.sort(id_estimates)
            trim = int(0.1 * len(id_estimates))
            if trim > 0:
                id_estimates = id_estimates[trim:-trim]

            return float(np.mean(id_estimates))
        except Exception as e:
            logger.warning(f"MLE ID computation failed: {e}")
            return 1.0

    def _id_twonn(self, X: np.ndarray) -> float:
        """
        Two-nearest-neighbor intrinsic dimension estimator.
        More robust for high dimensions.
        """
        n = X.shape[0]
        if n < 3:
            return 1.0

        try:
            # Compute pairwise distances
            distances = cdist(X, X, metric="euclidean")

            # Get 1st and 2nd nearest neighbor distances
            distances.sort(axis=1)
            r1 = distances[:, 1]  # 1st NN (exclude self)
            r2 = distances[:, 2]  # 2nd NN

            # Compute ratios
            ratios = r2 / (r1 + 1e-10)

            # Empirical CDF
            ratios_sorted = np.sort(ratios)
            empirical_cdf = np.arange(1, n + 1) / n

            # Fit dimension via least squares
            # Theory: P(ratio < x) ≈ 1 - (1/x)^d
            valid_idx = ratios_sorted > 1.0
            if np.sum(valid_idx) < 10:
                return 1.0

            x = np.log(ratios_sorted[valid_idx])
            y = np.log(1 - empirical_cdf[valid_idx] + 1e-10)

            # Linear regression
            A = np.vstack([x, np.ones(len(x))]).T
            d_estimate, _ = np.linalg.lstsq(A, y, rcond=None)[0]

            return float(abs(d_estimate))
        except Exception as e:
            logger.warning(f"TwoNN ID computation failed: {e}")
            return 1.0

    def _compute_correlation_dimension(self, X: np.ndarray) -> float:
        """
        Compute correlation dimension using box-counting approach.
        Measures fractal dimension of the activation manifold.
        """
        n = X.shape[0]
        if n < 10:
            return 1.0

        try:
            # Normalize data
            X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

            # Compute pairwise distances
            distances = cdist(X_normalized, X_normalized, metric="euclidean")

            # Count pairs within different scales
            counts = []
            scales = []

            for scale in self.correlation_scales:
                count = np.sum(distances < scale) - n  # Exclude diagonal
                if count > 0:
                    counts.append(np.log(count))
                    scales.append(np.log(scale))

            if len(scales) < 2:
                return 1.0

            # Fit line to log-log plot
            A = np.vstack([scales, np.ones(len(scales))]).T
            slope, _ = np.linalg.lstsq(A, counts, rcond=None)[0]

            return float(abs(slope))
        except Exception as e:
            logger.warning(f"Correlation dimension computation failed: {e}")
            return 1.0

    def get_trends(self, window: int = 10) -> dict[str, dict[int, str]]:
        """
        Analyze trends in ID and d over recent history.
        Returns 'increasing', 'decreasing', or 'stable' for each layer.
        """
        trends = {"id": {}, "d": {}}

        for metric in ["id", "d"]:
            for layer_id in self.layer_ids:
                history = self.history[metric][layer_id]

                if len(history) < window:
                    trends[metric][layer_id] = "insufficient_data"
                    continue

                recent = history[-window:]
                slope = np.polyfit(range(len(recent)), recent, 1)[0]

                if abs(slope) < 0.01:
                    trends[metric][layer_id] = "stable"
                elif slope > 0:
                    trends[metric][layer_id] = "increasing"
                else:
                    trends[metric][layer_id] = "decreasing"

        return trends

    def detect_phase_transition(self) -> bool:
        """
        Detect if a phase transition (e.g., grokking onset) is occurring.
        Based on sudden changes in ID/d patterns.
        """
        if any(len(h) < 20 for h in self.history["id"].values()):
            return False

        # Check for sudden ID drop (characteristic of grokking)
        for layer_id in self.layer_ids:
            id_history = self.history["id"][layer_id][-20:]
            d_history = self.history["d"][layer_id][-20:]

            # Compute rate of change
            id_change = abs(id_history[-1] - id_history[-10]) / (
                id_history[-10] + 1e-10
            )
            d_change = abs(d_history[-1] - d_history[-10]) / (d_history[-10] + 1e-10)

            # Phase transition indicators:
            # - Rapid ID decrease (>20% drop)
            # - Rapid d increase (>30% increase)
            if id_change > 0.2 and id_history[-1] < id_history[-10]:
                return True
            if d_change > 0.3 and d_history[-1] > d_history[-10]:
                return True

        return False
