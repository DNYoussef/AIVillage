import torch


class VPTQQuantizer:
    def __init__(self, bits_per_vector: float = 2.0, vector_length: int = 32):
        self.bits_per_vector = bits_per_vector
        self.vector_length = vector_length
        self.codebook_size = int(2**bits_per_vector)

    def _reshape_vectors(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        flat = weight_matrix.flatten()
        pad = (
            self.vector_length - flat.numel() % self.vector_length
        ) % self.vector_length
        if pad:
            flat = torch.cat([flat, torch.zeros(pad)])
        return flat.reshape(-1, self.vector_length)

    def _approx_hessian(self, vectors: torch.Tensor) -> torch.Tensor:
        var = torch.var(vectors, dim=0)
        return torch.diag(var + 1e-8)

    def _weighted_distance(
        self, vectors: torch.Tensor, centroids: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        h_diag = torch.diag(h)
        diff = vectors.unsqueeze(1) - centroids.unsqueeze(0)
        return torch.sum(diff**2 * h_diag, dim=2)

    def _assignment(
        self, vectors: torch.Tensor, centroids: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
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
        self, weight_matrix: torch.Tensor, hessian: torch.Tensor | None = None
    ) -> dict:
        vectors = self._reshape_vectors(weight_matrix)
        if hessian is None:
            hessian = self._approx_hessian(vectors)
        codebook = vectors[torch.randperm(len(vectors))[: self.codebook_size]]
        for _ in range(20):
            assignments = self._assignment(vectors, codebook, hessian)
            new_codebook = self._update(vectors, assignments)
            if torch.allclose(codebook, new_codebook, atol=1e-6):
                break
            codebook = new_codebook
        assignments = self._assignment(vectors, codebook, hessian)
        residuals = vectors - codebook[assignments]
        res_codebook, res_idx = self._quantize_residuals(residuals)
        return {
            "original_shape": weight_matrix.shape,
            "vector_length": self.vector_length,
            "codebook": codebook,
            "assignments": assignments,
            "residual_codebook": res_codebook,
            "residual_idx": res_idx,
        }

    def _quantize_residuals(
        self, residuals: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat = residuals.flatten()
        if flat.numel() == 0:
            return torch.zeros(1), torch.zeros_like(flat, dtype=torch.long)
        codebook = torch.linspace(flat.min(), flat.max(), 16)
        step = (flat.max() - flat.min()) / (15)
        idx = torch.clamp(torch.round((flat - flat.min()) / step), 0, 15).long()
        return codebook, idx.reshape(residuals.shape)

    def dequantize_weight_matrix(self, data: dict) -> torch.Tensor:
        codebook = data["codebook"]
        assign = data["assignments"]
        vectors = codebook[assign]
        res_codebook = data["residual_codebook"]
        res_idx = data["residual_idx"]
        residuals = res_codebook[res_idx]
        vecs = vectors + residuals
        flat = vecs.flatten()[: int(torch.prod(torch.tensor(data["original_shape"])))]
        return flat.reshape(data["original_shape"])
