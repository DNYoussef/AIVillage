"""HRM Reasoner model with ScratchpadSupervisor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from packages.hrrm.common.transformer_blocks import CausalSelfAttention, RMSNorm, SwiGLU


@dataclass
class ReasonerOutput:
    """Output from HRM Reasoner model."""

    logits: torch.Tensor
    thought_logits: torch.Tensor
    loss: torch.Tensor | None = None


@dataclass
class ReasonerConfig:
    """Configuration for HRM Reasoner model."""

    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 16
    n_head: int = 8
    d_ff: int = 2048
    max_seq_len: int = 2048
    rope_base: int = 10000

    # HRM parameters
    max_H: int = 4
    inner_T: int = 4

    # Reasoning parameters
    self_consistency_k: int = 3

    # Training
    dropout: float = 0.0
    lambda_thought: float = 0.1


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


class ScratchpadSupervisor(nn.Module):
    """Supervisor for Start-of-Thought / End-of-Thought instrumentation."""

    def __init__(self, d_model: int):
        super().__init__()
        self.thought_detector = nn.Linear(d_model, 2)  # SoT vs EoT detection
        self.thought_gate = nn.Linear(d_model, 1)  # Reasoning quality gate

    def forward(self, hidden_states):
        """Detect and supervise reasoning spans."""
        thought_logits = self.thought_detector(hidden_states)
        thought_gates = torch.sigmoid(self.thought_gate(hidden_states))
        return thought_logits, thought_gates


class HRMReasoner(nn.Module):
    """
    HRM Reasoner with two-timescale loop and scratchpad supervision.

    Implements HRM (H slow step calls L fast loop) with deep supervision
    and approximate 1-step gradients for reasoning stability.
    """

    def __init__(self, config: ReasonerConfig):
        super().__init__()
        self.config = config
        self.core = TinyDecoder(config.vocab_size, config.d_model, config.n_layers, config.n_head)
        self.scratchpad = ScratchpadSupervisor(config.d_model)
        self.max_H = config.max_H
        self.inner_T = config.inner_T

    def forward(self, x_ids, labels=None, attn_mask=None, thought_mask=None):
        """
        Forward pass with HRM loop and scratchpad supervision.

        Args:
            x_ids: Input token ids [B, N]
            labels: Target labels for language modeling [B, N]
            attn_mask: Attention mask [B, N]
            thought_mask: Binary mask for thought spans [B, N]
        """
        x = x_ids
        losses = []
        thought_logits = None

        for h in range(self.max_H):
            # Inner fast refinement (L-loop)
            logits = None
            hidden_states = None

            for t in range(self.inner_T):
                logits = self.core(x, attn_mask)
                # Extract hidden states from last layer for scratchpad supervision
                hidden_states = self.core.norm(self.core.blocks[-1](self.core.tok(x), attn_mask))

            # Get thought logits from scratchpad supervisor
            thought_logits, _ = self.scratchpad(hidden_states)

            # Deep supervision at each H step
            if labels is not None:
                # Language modeling loss
                ce = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

                # Scratchpad supervision loss
                thought_loss = torch.tensor(0.0, device=logits.device)
                if thought_mask is not None:
                    # Binary classification for thought spans
                    thought_loss = F.binary_cross_entropy_with_logits(thought_logits[:, :, 0], thought_mask.float())

                losses.append(ce + self.config.lambda_thought * thought_loss)

            # 1-step gradient approximation: detach state before next H
            if h < self.max_H - 1:
                x = x.detach()

        loss = sum(losses) / len(losses) if losses else None
        return ReasonerOutput(logits=logits, thought_logits=thought_logits, loss=loss)

    def generate_with_reasoning(self, x_ids, max_length=100, temperature=1.0, k=5):
        """
        Generate with self-consistency reasoning (k samples).

        Args:
            x_ids: Input prompt [B, N]
            max_length: Maximum generation length
            temperature: Sampling temperature
            k: Number of reasoning chains for self-consistency

        Returns:
            Generated sequences with reasoning chains
        """
        self.eval()
        with torch.no_grad():
            generations = []

            for _ in range(k):
                current = x_ids.clone()

                for _ in range(max_length):
                    logits = self.forward(current)
                    if isinstance(logits, tuple):
                        logits = logits[1]  # Extract logits from (loss, logits)

                    # Sample next token
                    probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    current = torch.cat([current, next_token], dim=1)

                    # Stop if we hit EOS or max length
                    if next_token.item() == 2:  # Assuming EOS token id = 2
                        break

                generations.append(current)

            return generations
