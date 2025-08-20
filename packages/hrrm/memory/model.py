# packages/hrrm/memory/model.py (MAC: memory-as-context)
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from packages.hrrm.common.transformer_blocks import CausalSelfAttention, RMSNorm, SwiGLU
from packages.hrrm.memory.ext_memory import NeuralMemory


@dataclass
class MemoryOutput:
    """Output from Memory model."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None


@dataclass
class MemoryConfig:
    """Configuration for Memory model."""

    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 12  # Fewer layers than Planner/Reasoner to stay ~50M
    n_head: int = 8
    d_ff: int = 2048  # Feed-forward dimension
    max_seq_len: int = 2048
    rope_base: int = 10000

    # Memory parameters
    mem_dim: int = 256
    mem_tokens: int = 8
    mem_slots: int = 1024

    # Titans parameters
    alpha: float = 4.0  # Surprise gating
    beta: float = 0.9  # Momentum
    eta: float = 1e-2  # Learning rate
    eta_decay: float = 1e-4  # Forgetting rate

    # Training
    dropout: float = 0.0
    tie_embeddings: bool = True


class MemoryAsContextTiny(nn.Module):
    """
    Append M learned 'memory tokens' derived from NeuralMemory.read(q) to the input as extra context (MAC).
    """

    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        self.tok = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([self._block(config.d_model, config.n_head) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.head.weight = self.tok.weight
        self.mem = NeuralMemory(d_key=config.mem_dim, d_val=config.mem_dim)
        self.q_proj = nn.Linear(config.d_model, config.mem_dim)
        self.v_proj = nn.Linear(config.mem_dim, config.d_model)
        self.mem_tokens = config.mem_tokens

    def _block(self, d, h):
        return nn.Sequential(RMSNorm(d), CausalSelfAttention(d, h), RMSNorm(d), SwiGLU(d, 4))

    def forward(self, x_ids, attn_mask=None, targets=None, loss_like=None):
        x = self.tok(x_ids)  # [B,N,D]
        # query memory using last token (cheap)
        q = self.q_proj(x[:, -1, :])  # [B,mem_dim]
        v, _, _ = self.mem.read(q, topk=max(self.mem_tokens, 8))  # [B,mem_dim]
        mem_tokens = self.v_proj(v).unsqueeze(1).repeat(1, self.mem_tokens, 1)  # [B,M,D]
        x_plus = torch.cat([mem_tokens, x], dim=1)
        # causal mask must allow mem_tokens to attend anywhere, others attend left
        B, NM, D = x_plus.shape
        # Create mask: memory tokens can attend to all, input tokens follow causal pattern
        att_mask = torch.ones(NM, NM, dtype=torch.bool, device=x_plus.device)
        att_mask[self.mem_tokens :, self.mem_tokens :] = torch.tril(
            torch.ones(NM - self.mem_tokens, NM - self.mem_tokens, device=x_plus.device)
        )

        for blk in self.blocks:
            n1, attn, n2, mlp = blk
            y = x_plus + attn(n1(x_plus), att_mask)
            x_plus = y + mlp(n2(y))
        logits = self.head(self.norm(x_plus)[:, -x.size(1) :, :])  # only predict on original tokens

        # Update memory if training
        if (targets is not None) and (loss_like is not None):
            with torch.no_grad():
                self.mem.update(q, v, loss_like)  # online memory update (Titans)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

        return MemoryOutput(logits=logits, loss=loss)
