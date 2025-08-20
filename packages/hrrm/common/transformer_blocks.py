# packages/hrrm/common/transformer_blocks.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):  # Pre-RMSNorm
        n = x.norm(2, dim=-1, keepdim=True)
        rms = n * (x.size(-1) ** -0.5)
        return (x / (rms + self.eps)) * self.weight


def swiglu(x, w1, w2, b1=None, b2=None):
    a = F.linear(x, w1, b1)
    b = F.linear(x, w2, b2)
    return F.silu(a) * b


class SwiGLU(nn.Module):
    def __init__(self, d, mult=4):
        super().__init__()
        h = mult * d
        self.w1 = nn.Parameter(torch.empty(h, d))
        self.w2 = nn.Parameter(torch.empty(h, d))
        self.bo = nn.Parameter(torch.zeros(d))
        self.wo = nn.Parameter(torch.empty(d, h))
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.wo)

    def forward(self, x):
        return F.linear(swiglu(x, self.w1, self.w2), self.wo, self.bo)


class RotaryPositionalEmbedding:
    def __init__(self, dim, base=10000):
        self.dim = dim
        self.base = base

    def get(self, seq_len, device):
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        pos = torch.arange(seq_len, device=device).float()[:, None]
        freqs = pos * theta[None, :]
        return torch.polar(torch.ones_like(freqs), freqs)  # complex cos/sin carrier


def apply_rope(x, rope_cis):  # x: [b, h, n, d]; rope dims last
    x_ = x.float().view(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_)

    # Expand rope_cis to match x_complex dimensions
    # rope_cis: [1, 1, n, d//2], x_complex: [b, h, n, d//2]
    rope = rope_cis.expand_as(x_complex)

    x_rot = x_complex * rope
    out = torch.view_as_real(x_rot).flatten(-2)
    return out.type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, rope_base=10000):
        super().__init__()
        self.h = n_head
        self.d = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryPositionalEmbedding(self.d, base=rope_base)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.h, self.d).permute(2, 0, 3, 1, 4)  # 3,B,h,N,d
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is B,h,N,d

        # Get rope embeddings and reshape for broadcasting
        rope_cis = self.rope.get(N, x.device)  # [N, d//2]
        rope_cis = rope_cis[None, None, :, :]  # [1, 1, N, d//2]

        q, k = apply_rope(q, rope_cis), apply_rope(k, rope_cis)
        att = (q @ k.transpose(-2, -1)) * (self.d**-0.5)

        # Apply causal mask if provided
        if mask is not None:
            att = att.masked_fill(mask == 0, float("-inf"))
        else:
            # Default causal mask
            causal_mask = torch.tril(torch.ones(N, N, device=x.device))
            att = att.masked_fill(causal_mask == 0, float("-inf"))

        p = att.softmax(-1)
        y = (p @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.o(y)
