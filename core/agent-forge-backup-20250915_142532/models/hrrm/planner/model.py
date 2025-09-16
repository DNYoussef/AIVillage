# packages/hrrm/planner/model.py (Reasoner uses same skeleton)
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from packages.hrrm.common.transformer_blocks import CausalSelfAttention, RMSNorm, SwiGLU


@dataclass
class PlannerOutput:
    """Output from HRM Planner model."""

    logits: torch.Tensor
    control_logits: torch.Tensor
    loss: torch.Tensor | None = None


@dataclass
class PlannerConfig:
    """Configuration for HRM Planner model."""

    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 16
    n_head: int = 8
    d_ff: int = 2048
    max_seq_len: int = 2048
    rope_base: int = 10000

    # Control tokens
    control_tokens: list[str] = None

    # HRM parameters
    max_H: int = 4
    inner_T: int = 4

    # Training
    dropout: float = 0.0
    lambda_ctrl: float = 0.2

    def __post_init__(self):
        if self.control_tokens is None:
            self.control_tokens = ["<PLAN>", "<SUBGOAL>", "<ACTION>", "<CHECK>", "<ENDPLAN>"]


class TinyBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.att = CausalSelfAttention(d, h)
        self.n2 = RMSNorm(d)
        self.mlp = SwiGLU(d, 4)

    def forward(self, x, attn_mask=None):
        x = x + self.att(self.n1(x), attn_mask)
        x = x + self.mlp(self.n2(x))
        return x


class TinyDecoder(nn.Module):
    def __init__(self, vocab, d=512, L=16, h=8, rope_base=10000):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([TinyBlock(d, h) for _ in range(L)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # tie

    def forward(self, x_ids, attn_mask=None):
        x = self.tok(x_ids)
        for b in self.blocks:
            x = b(x, attn_mask)
        return self.head(self.norm(x))


class ControllerHead(nn.Module):  # planner aux head to detect control tokens
    def __init__(self, d, n_ctrl):
        super().__init__()
        self.proj = nn.Linear(d, n_ctrl)

    def forward(self, h_last):
        return self.proj(h_last)  # use last token hidden


class HRMPlanner(nn.Module):
    """
    H-slow/L-fast loop:
      - For each H cycle: run L-steps local refinement, compute loss (deep supervision), then detach state.
      - Approximate 1-step gradient: backprop only through final L of each H (memory-safe).
    """

    def __init__(self, config: PlannerConfig):
        super().__init__()
        self.config = config
        self.core = TinyDecoder(config.vocab_size, config.d_model, config.n_layers, config.n_head)
        self.ctrl = ControllerHead(config.d_model, len(config.control_tokens))
        self.max_H = config.max_H
        self.inner_T = config.inner_T

    def forward(self, x_ids, labels=None, attn_mask=None, control_mask=None):
        x = x_ids
        losses = []

        for h in range(self.max_H):
            # inner fast refinement (unrolled T times)
            logits = None
            hidden_states = None
            for t in range(self.inner_T):
                logits = self.core(x, attn_mask)
                # Get hidden states before the final projection
                hidden_states = self.core.norm(self.core.blocks[-1](self.core.tok(x), attn_mask))

            # Get control logits using last token hidden state
            h_last = hidden_states[:, -1, :]  # Last token hidden representation
            ctrl_logits = self.ctrl(h_last)  # [B, n_ctrl]

            # deep supervision
            if labels is not None:
                ce = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                aux = torch.tensor(0.0, device=logits.device)
                if control_mask is not None:
                    aux = F.binary_cross_entropy_with_logits(ctrl_logits, control_mask.float())
                losses.append(ce + self.config.lambda_ctrl * aux)

            # 1-step gradient approximation: stop gradient before next H
            if h < self.max_H - 1:
                x = x.detach()

        loss = sum(losses) / len(losses) if losses else None
        return PlannerOutput(logits=logits, control_logits=ctrl_logits, loss=loss)
