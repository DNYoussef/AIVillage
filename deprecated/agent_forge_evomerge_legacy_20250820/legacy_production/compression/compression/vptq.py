import logging
import os

import torch

logger = logging.getLogger(__name__)


class VPTQQuantizer:
    """Vector Product Quantization with Hessian weighting for Stage 2 compression.

    The original research target for VPTQ claimed up to ``12×`` compression, but in
    practice this implementation reliably reaches between ``4×`` and ``8×`` depending on
    the model.  An optional activation quantization step can be enabled by setting the
    environment variable ``VPTQ_ENABLE_ACTIVATION_QUANT=1``.  Activations are stored as
    8-bit values when enabled which may introduce a small (<1%) accuracy drop.

    This quantizer operates on weights that have already been processed by Stage 1
    (BitNet + SeedLM) and performs Hessian-weighted vector quantization with optional
    weight packing for compact storage.
    """

    def __init__(self, bits_per_vector: float = 2.0, vector_length: int = 32) -> None:
        self.bits_per_vector = bits_per_vector
        self.vector_length = vector_length
        self.codebook_size = int(2**bits_per_vector)
        self.max_iterations = 50  # Increased for better convergence
        self.convergence_threshold = 1e-6

    def _reshape_vectors(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        flat = weight_matrix.flatten()
        pad = (self.vector_length - flat.numel() % self.vector_length) % self.vector_length
        if pad:
            flat = torch.cat([flat, torch.zeros(pad)])
        return flat.reshape(-1, self.vector_length)

    def _compute_hessian(self, vectors: torch.Tensor, method: str = "fisher") -> torch.Tensor:
        """Compute Hessian approximation for weighting the quantization error.

        Args:
            vectors: Input vectors of shape (n_vectors, vector_length)
            method: Method for Hessian approximation ("fisher", "diagonal", "gauss_newton")

        Returns:
            Hessian matrix of shape (vector_length, vector_length)
        """
        if method == "fisher":
            # Fisher Information Matrix approximation
            centered = vectors - vectors.mean(dim=0, keepdim=True)
            cov = torch.mm(centered.T, centered) / (vectors.size(0) - 1)
            return cov + torch.eye(vectors.size(1)) * 1e-8

        if method == "diagonal":
            # Diagonal approximation (original implementation)
            var = torch.var(vectors, dim=0)
            return torch.diag(var + 1e-8)

        if method == "gauss_newton":
            # Gauss-Newton approximation
            grad = vectors - vectors.mean(dim=0, keepdim=True)
            hessian = torch.mm(grad.T, grad) / vectors.size(0)
            return hessian + torch.eye(vectors.size(1)) * 1e-8

        msg = f"Unknown Hessian method: {method}"
        raise ValueError(msg)

    def _approx_hessian(self, vectors: torch.Tensor) -> torch.Tensor:
        """Backward compatibility wrapper."""
        return self._compute_hessian(vectors, method="diagonal")

    @staticmethod
    def _pack_bits(values: torch.Tensor, bits: int) -> torch.Tensor:
        """Pack integer ``values`` into a ``uint8`` tensor using ``bits`` per value."""
        if values.numel() == 0:
            return torch.zeros(0, dtype=torch.uint8)

        values = values.to(torch.int64).view(-1)
        mask = (1 << bits) - 1
        per_byte = 8 // bits

        pad = (-values.numel()) % per_byte
        if pad:
            values = torch.cat([values, torch.zeros(pad, dtype=torch.int64)])

        packed = torch.zeros(values.numel() // per_byte, dtype=torch.uint8)
        for i in range(per_byte):
            packed |= ((values[i::per_byte] & mask) << (i * bits)).to(torch.uint8)
        return packed

    @staticmethod
    def _unpack_bits(packed: torch.Tensor, bits: int, total_values: int) -> torch.Tensor:
        """Inverse of :meth:`_pack_bits`. Returns a ``long`` tensor."""
        if packed.numel() == 0:
            return torch.zeros(total_values, dtype=torch.long)

        mask = (1 << bits) - 1
        per_byte = 8 // bits
        values = []
        for i in range(per_byte):
            vals = ((packed >> (i * bits)) & mask).to(torch.long)
            values.append(vals)
        stacked = torch.stack(values, dim=1).view(-1)[:total_values]
        return stacked

    @staticmethod
    def _quantize_activation(x: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Simple symmetric int8 activation quantization."""
        if x.numel() == 0:
            return torch.zeros(0, dtype=torch.int8), 1.0
        scale = x.abs().max().item() / 127 if x.abs().max() != 0 else 1.0
        q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
        return q, scale

    def _weighted_distance(self, vectors: torch.Tensor, centroids: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        h_diag = torch.diag(h)
        diff = vectors.unsqueeze(1) - centroids.unsqueeze(0)
        return torch.sum(diff**2 * h_diag, dim=2)

    def _assignment(self, vectors: torch.Tensor, centroids: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        d = self._weighted_distance(vectors, centroids, h)
        return torch.argmin(d, dim=1)

    def _update(self, vectors: torch.Tensor, assignments: torch.Tensor) -> torch.Tensor:
        centroids = torch.zeros(self.codebook_size, self.vector_length)
        for k in range(self.codebook_size):
            mask = assignments == k
            if mask.any():
                centroids[k] = vectors[mask].mean(dim=0)
        return centroids

    def quantize_weight_matrix(
        self,
        weight_matrix: torch.Tensor,
        hessian: torch.Tensor | None = None,
        hessian_method: str = "fisher",
    ) -> dict[str, torch.Tensor]:
        """Quantize weight matrix using VPTQ with Hessian weighting.

        Args:
            weight_matrix: Input weight matrix to quantize
            hessian: Optional pre-computed Hessian matrix
            hessian_method: Method for Hessian approximation if not provided

        Returns:
            Dictionary containing quantization data
        """
        logger.debug(f"Quantizing matrix of shape {weight_matrix.shape}")

        vectors = self._reshape_vectors(weight_matrix)
        original_vectors = vectors.clone()

        if hessian is None:
            hessian = self._compute_hessian(vectors, method=hessian_method)

        # Initialize codebook with k-means++
        codebook = self._initialize_codebook_kmeans_plus(vectors, hessian)

        # Iterative refinement
        prev_loss = float("inf")
        for iteration in range(self.max_iterations):
            assignments = self._assignment(vectors, codebook, hessian)
            new_codebook = self._update(vectors, assignments)

            # Check convergence
            if torch.allclose(codebook, new_codebook, atol=self.convergence_threshold):
                logger.debug(f"VPTQ converged after {iteration} iterations")
                break

            # Calculate loss for monitoring
            current_loss = self._calculate_quantization_loss(vectors, new_codebook, assignments, hessian)
            if current_loss > prev_loss:
                logger.debug(f"Loss increased at iteration {iteration}, stopping")
                break

            codebook = new_codebook
            prev_loss = current_loss

        # Final assignment
        assignments = self._assignment(vectors, codebook, hessian)

        # Calculate residuals and quantize them
        residuals = vectors - codebook[assignments]
        res_codebook, res_idx = self._quantize_residuals(residuals)

        # Optional activation quantization
        activation_q = None
        activation_scale = 1.0
        if os.getenv("VPTQ_ENABLE_ACTIVATION_QUANT", "0") == "1":
            activation_q, activation_scale = self._quantize_activation(vectors)

        # Pack indices for storage
        assignments_packed = self._pack_bits(assignments, int(self.bits_per_vector))
        res_idx_packed = self._pack_bits(res_idx.flatten(), 4)

        # Calculate compression statistics
        original_bits = weight_matrix.numel() * 32  # float32
        compressed_bits = (
            codebook.numel() * 32
            + assignments_packed.numel() * 8
            + res_codebook.numel() * 32
            + res_idx_packed.numel() * 8
        )
        compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 0

        # Calculate reconstruction error
        assignments_unpacked = self._unpack_bits(assignments_packed, int(self.bits_per_vector), assignments.numel())
        res_idx_unpacked = self._unpack_bits(res_idx_packed, 4, res_idx.numel()).reshape_as(res_idx)
        reconstructed = self._reconstruct_from_quantization(
            codebook, assignments_unpacked, res_codebook, res_idx_unpacked
        )
        reconstruction_error = torch.norm(original_vectors - reconstructed).item()

        return {
            "original_shape": weight_matrix.shape,
            "vector_length": self.vector_length,
            "codebook": codebook,
            "assignments_packed": assignments_packed,
            "residual_codebook": res_codebook,
            "residual_idx_packed": res_idx_packed,
            "compression_ratio": compression_ratio,
            "reconstruction_error": reconstruction_error,
            "hessian_method": hessian_method,
            "bits_per_vector": self.bits_per_vector,
            "activation_q": activation_q,
            "activation_scale": activation_scale,
        }

    def _quantize_residuals(self, residuals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = residuals.flatten()
        if flat.numel() == 0:
            return torch.zeros(1), torch.zeros_like(flat, dtype=torch.long)
        codebook = torch.linspace(flat.min(), flat.max(), 16)
        step = (flat.max() - flat.min()) / (15)
        idx = torch.clamp(torch.round((flat - flat.min()) / step), 0, 15).long()
        return codebook, idx.reshape(residuals.shape)

    def _initialize_codebook_kmeans_plus(self, vectors: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor:
        """Initialize codebook using k-means++ algorithm with Hessian weighting."""
        n_vectors = vectors.size(0)
        codebook = torch.zeros(self.codebook_size, self.vector_length)

        # Choose first centroid randomly
        codebook[0] = vectors[torch.randint(0, n_vectors, (1,))]

        # Choose remaining centroids using weighted distance
        for i in range(1, self.codebook_size):
            distances = self._weighted_distance(vectors, codebook[:i], hessian)
            min_distances = torch.min(distances, dim=1)[0]
            probabilities = min_distances / min_distances.sum()

            # Sample next centroid
            selected_idx = torch.multinomial(probabilities, 1)
            codebook[i] = vectors[selected_idx]

        return codebook

    def _calculate_quantization_loss(
        self,
        vectors: torch.Tensor,
        codebook: torch.Tensor,
        assignments: torch.Tensor,
        hessian: torch.Tensor,
    ) -> float:
        """Calculate quantization loss with Hessian weighting."""
        reconstructed = codebook[assignments]
        diff = vectors - reconstructed

        # Apply Hessian weighting
        if hessian.dim() == 2:
            # Full Hessian matrix
            weighted_diff = torch.mm(diff, hessian)
            loss = torch.sum(diff * weighted_diff)
        else:
            # Diagonal Hessian
            weighted_diff = diff * torch.diag(hessian).unsqueeze(0)
            loss = torch.sum(weighted_diff * diff)

        return loss.item()

    def _reconstruct_from_quantization(
        self,
        codebook: torch.Tensor,
        assignments: torch.Tensor,
        res_codebook: torch.Tensor,
        res_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct vectors from quantization data."""
        vectors = codebook[assignments]
        residuals = res_codebook[res_idx]
        return vectors + residuals

    def dequantize_weight_matrix(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Reconstruct weight matrix from VPTQ quantization data.

        Args:
            data: Dictionary containing quantization data

        Returns:
            Reconstructed weight matrix
        """
        codebook = data["codebook"]
        res_codebook = data["residual_codebook"]

        original_shape = data["original_shape"]
        original_size = int(torch.prod(torch.tensor(original_shape)))
        vector_length = data.get("vector_length", self.vector_length)
        n_vectors = (original_size + vector_length - 1) // vector_length

        if "assignments" in data:
            assignments = data["assignments"]
        else:
            assignments = self._unpack_bits(
                data["assignments_packed"], int(data.get("bits_per_vector", self.bits_per_vector)), n_vectors
            )

        if "residual_idx" in data:
            res_idx = data["residual_idx"]
        else:
            res_idx = self._unpack_bits(data["residual_idx_packed"], 4, n_vectors * vector_length).reshape(
                n_vectors, vector_length
            )

        # Reconstruct vectors
        vectors = self._reconstruct_from_quantization(codebook, assignments, res_codebook, res_idx)
        # Reshape back to original shape
        flat = vectors.flatten()[:original_size]
        return flat.reshape(original_shape)
