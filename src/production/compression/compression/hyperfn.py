import logging
import math

import torch

logger = logging.getLogger(__name__)


class HyperCompressionEncoder:
    """Hyper-function compression encoder using ergodic trajectory representation.

    This encoder represents weight clusters as parametric trajectories in phase space,
    achieving additional compression beyond VPTQ quantization.
    """

    def __init__(
        self, num_clusters: int = 16, trajectory_types: list[str] | None = None
    ) -> None:
        self.num_clusters = num_clusters
        self.trajectory_types = trajectory_types or ["sinusoidal", "spiral", "chaotic"]
        self.max_search_iterations = 100
        self.convergence_threshold = 1e-6

    def _cluster(self, weights: torch.Tensor) -> list[dict]:
        flat = weights.flatten()
        idx = torch.argsort(flat.abs())
        clusters = []
        size = len(flat) // self.num_clusters
        for i in range(self.num_clusters):
            s = i * size
            e = s + size if i < self.num_clusters - 1 else len(flat)
            indices = idx[s:e]
            cluster = flat[indices]
            clusters.append({"weights": cluster, "indices": indices})
        return clusters

    def _generate_sinusoidal_trajectory(
        self, length: int, params: dict
    ) -> torch.Tensor:
        """Generate sinusoidal trajectory: A*sin(2π*α*t) + B*cos(2π*α*t) + D."""
        t = torch.arange(length, dtype=torch.float32)
        alpha = params["an"] / params["ad"]
        theta = 2 * math.pi * alpha * t
        return (
            params["A"] * torch.sin(theta)
            + params["B"] * torch.cos(theta)
            + params["D"]
        )

    def _generate_spiral_trajectory(self, length: int, params: dict) -> torch.Tensor:
        """Generate spiral trajectory: A*t*sin(2π*α*t) + B*t*cos(2π*α*t) + D."""
        t = torch.arange(length, dtype=torch.float32) / length  # Normalize to [0,1]
        alpha = params["an"] / params["ad"]
        theta = 2 * math.pi * alpha * t
        return (
            params["A"] * t * torch.sin(theta)
            + params["B"] * t * torch.cos(theta)
            + params["D"]
        )

    def _generate_chaotic_trajectory(self, length: int, params: dict) -> torch.Tensor:
        """Generate chaotic trajectory using logistic map: x_{n+1} = r*x_n*(1-x_n)."""
        trajectory = torch.zeros(length)
        x = 0.5  # Initial condition
        r = 3.0 + params["A"]  # Chaotic parameter

        for i in range(length):
            trajectory[i] = params["B"] * x + params["D"]
            x = r * x * (1 - x)

        return trajectory

    def _search_params(
        self, w: torch.Tensor, trajectory_type: str = "sinusoidal"
    ) -> dict:
        """Search for optimal parameters for the specified trajectory type."""
        mean = w.mean().item()
        std = w.std().item()

        best = {
            "A": 0,
            "B": 0,
            "C": 0,
            "D": mean,
            "an": 1,
            "ad": 2,
            "err": float("inf"),
            "trajectory_type": trajectory_type,
        }

        # Adaptive parameter ranges based on data statistics
        A_range = torch.linspace(-2 * std, 2 * std, 8)
        B_range = torch.linspace(-2 * std, 2 * std, 8)

        if trajectory_type == "sinusoidal":
            alpha_n_range = [1, 2, 3, 5, 7]
            alpha_d_range = [2, 3, 5, 7, 11]
        elif trajectory_type == "spiral":
            alpha_n_range = [1, 2, 3]
            alpha_d_range = [2, 3, 5]
        elif trajectory_type == "chaotic":
            # For chaotic trajectories, we use different parameter meanings
            A_range = torch.linspace(0, 1, 5)  # Chaos parameter modifier
            B_range = torch.linspace(0.5, 2.0, 5)  # Amplitude
            alpha_n_range = [1]  # Not used for chaotic
            alpha_d_range = [1]  # Not used for chaotic
        else:
            msg = f"Unknown trajectory type: {trajectory_type}"
            raise ValueError(msg)

        # Grid search for optimal parameters
        for A in A_range:
            for B in B_range:
                for alpha_n in alpha_n_range:
                    for alpha_d in alpha_d_range:
                        params = {
                            "A": A.item(),
                            "B": B.item(),
                            "C": 0,
                            "D": mean,
                            "an": alpha_n,
                            "ad": alpha_d,
                            "trajectory_type": trajectory_type,
                        }

                        # Generate trajectory
                        if trajectory_type == "sinusoidal":
                            traj = self._generate_sinusoidal_trajectory(len(w), params)
                        elif trajectory_type == "spiral":
                            traj = self._generate_spiral_trajectory(len(w), params)
                        elif trajectory_type == "chaotic":
                            traj = self._generate_chaotic_trajectory(len(w), params)

                        # Calculate error
                        err = torch.sum((w - traj) ** 2).item()

                        if err < best["err"]:
                            best.update(params)
                            best["err"] = err

        return best

    def compress_weight_matrix(
        self, weight_matrix: torch.Tensor, trajectory_type: str = "auto"
    ) -> dict:
        """Compress weight matrix using hyper-function representation.

        Args:
            weight_matrix: Input weight matrix to compress
            trajectory_type: Type of trajectory to use ("auto", "sinusoidal", "spiral", "chaotic")

        Returns:
            Dictionary containing compression data
        """
        logger.debug(
            f"Compressing matrix of shape {weight_matrix.shape} with hyper-function"
        )

        clusters = self._cluster(weight_matrix)
        params = []

        # If auto, try all trajectory types and pick the best
        if trajectory_type == "auto":
            for i, cluster in enumerate(clusters):
                best_params = None
                best_error = float("inf")

                for traj_type in self.trajectory_types:
                    cluster_params = self._search_params(cluster["weights"], traj_type)
                    if cluster_params["err"] < best_error:
                        best_error = cluster_params["err"]
                        best_params = cluster_params

                params.append(best_params)
                logger.debug(
                    f"Cluster {i}: best trajectory type = {best_params['trajectory_type']}, error = {best_error:.6f}"
                )
        else:
            # Use specified trajectory type for all clusters
            for cluster in clusters:
                cluster_params = self._search_params(
                    cluster["weights"], trajectory_type
                )
                params.append(cluster_params)

        # Calculate compression statistics
        original_bits = weight_matrix.numel() * 32  # float32
        compressed_bits = len(params) * 8 * 4  # 8 parameters per cluster, 4 bytes each
        compression_ratio = (
            original_bits / compressed_bits if compressed_bits > 0 else 0
        )

        # Calculate total reconstruction error
        total_error = sum(p["err"] for p in params)

        return {
            "params": params,
            "original_shape": weight_matrix.shape,
            "compression_ratio": compression_ratio,
            "total_error": total_error,
            "num_clusters": self.num_clusters,
            "trajectory_types_used": list({p["trajectory_type"] for p in params}),
        }

    def _reconstruct_trajectory(self, params: dict, length: int) -> torch.Tensor:
        """Reconstruct trajectory from parameters."""
        trajectory_type = params.get("trajectory_type", "sinusoidal")

        if trajectory_type == "sinusoidal":
            return self._generate_sinusoidal_trajectory(length, params)
        if trajectory_type == "spiral":
            return self._generate_spiral_trajectory(length, params)
        if trajectory_type == "chaotic":
            return self._generate_chaotic_trajectory(length, params)
        # Fallback to sinusoidal
        return self._generate_sinusoidal_trajectory(length, params)

    def decompress_weight_matrix(self, data: dict) -> torch.Tensor:
        """Decompress weight matrix from hyper-function representation.

        Args:
            data: Dictionary containing compression data

        Returns:
            Reconstructed weight matrix
        """
        shape = data["original_shape"]
        total = (
            shape.numel()
            if isinstance(shape, torch.Size)
            else int(torch.prod(torch.tensor(shape)))
        )

        # Reconstruct clusters
        out = torch.zeros(total)

        # Re-create clusters to get indices
        temp_matrix = torch.zeros(total)
        clusters = self._cluster(temp_matrix.reshape(shape))

        for cluster, params in zip(clusters, data["params"], strict=False):
            indices = cluster["indices"]
            length = len(indices)

            # Reconstruct trajectory
            trajectory = self._reconstruct_trajectory(params, length)

            # Place reconstructed values
            out[indices] = trajectory

        return out.reshape(shape)
