import torch


class VPTQQuantizer:
    def __init__(self, K=4, D=32, hessian=None):
        self.K = K
        self.D = D
        self.hessian = hessian

    def train_codebook(self, X: torch.Tensor):
        centroids = X[torch.randperm(X.size(0))[:self.K]]
        for _ in range(20):
            dists = ((X.unsqueeze(1) - centroids) ** 2).sum(-1)
            if self.hessian is not None:
                dists *= self.hessian.unsqueeze(1)
            idx = dists.argmin(1)
            for k in range(self.K):
                assigned = X[idx == k]
                if assigned.numel():
                    centroids[k] = assigned.mean(0)
        self.centroids = centroids

    def encode(self, X: torch.Tensor):
        dists = ((X.unsqueeze(1) - self.centroids) ** 2).sum(-1)
        return dists.argmin(1)

    def decode(self, idx: torch.Tensor):
        return self.centroids[idx]
