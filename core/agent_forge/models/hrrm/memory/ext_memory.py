# packages/hrrm/memory/ext_memory.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralMemory(nn.Module):
    """
    Persistent key/value memory with online plasticity:
      - surprise gate: g = sigmoid(alpha * surprise)
      - momentum: m_t = beta * m_{t-1} + (1-beta) * grad_like
      - decay: W <- (1 - eta_decay) * W  (forgetting)
      - update: W <- W + eta * g * m_t
    """

    def __init__(self, d_key=256, d_val=256, n_slots=1024, alpha=4.0, beta=0.9, eta=1e-2, eta_decay=1e-4):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(n_slots, d_key) * 0.02, requires_grad=False)
        self.vals = nn.Parameter(torch.randn(n_slots, d_val) * 0.02, requires_grad=False)
        self.register_buffer("momentum_k", torch.zeros_like(self.keys))
        self.register_buffer("momentum_v", torch.zeros_like(self.vals))
        self.alpha, self.beta, self.eta, self.eta_decay = alpha, beta, eta, eta_decay

    @torch.no_grad()
    def read(self, q, topk=32):
        # cosine sim retrieval
        qn = F.normalize(q, dim=-1)
        kn = F.normalize(self.keys, dim=-1)
        scores = qn @ kn.t()  # [B, S]
        idx = scores.topk(k=min(topk, self.keys.size(0)), dim=-1).indices
        k_sel = self.keys[idx]
        v_sel = self.vals[idx]
        att = F.softmax((qn.unsqueeze(1) * F.normalize(k_sel, dim=-1)).sum(-1), dim=-1)  # [B, K]
        v = (att.unsqueeze(-1) * v_sel).sum(1)  # [B, d_val]
        return v, idx, att

    @torch.no_grad()
    def update(self, q, target, loss_like):
        # surprise -> gate
        g = torch.sigmoid(self.alpha * loss_like.detach())
        # compute grad-like deltas (pointing keys/vals toward q/target)
        grad_k = q - 0.0  # simple Hebbian-like proxy
        grad_v = target - 0.0
        # momentum
        self.momentum_k.mul_(self.beta).add_((1 - self.beta) * grad_k.mean(0))
        self.momentum_v.mul_(self.beta).add_((1 - self.beta) * grad_v.mean(0))
        # decay (forgetting)
        self.keys.mul_(1 - self.eta_decay)
        self.vals.mul_(1 - self.eta_decay)
        # apply update uniformly or using nearest slots (keep simple for bootstrap)
        self.keys.add_(self.eta * g.mean() * self.momentum_k)
        self.vals.add_(self.eta * g.mean() * self.momentum_v)
