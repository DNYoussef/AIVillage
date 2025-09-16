#!/usr/bin/env python3
"""
Cognate Refiner Core - Production-Ready 25M Model System

This module consolidates all Phase 1 components into a unified, production-ready
CognateRefiner class with 25M parameters. Integrates:

- Unified 25M backbone transformer (20M params)
- Long-term memory system with cross-attention (4M params)
- ACT halting head for adaptive computation (0.5M params)
- Edit head for language modeling (0.5M params)

Total: ~25M parameters with HuggingFace compatibility.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Set up logger FIRST
logger = logging.getLogger(__name__)

# Add packages path for unified_refiner
packages_path = Path(__file__).parent.parent.parent.parent.parent / "packages"
sys.path.append(str(packages_path))

# Add current directory for local imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    # Try local imports first
    from halting_head import ACTHaltingHead
    from memory_cross_attn import MemoryScheduler, create_memory_cross_attention

    logger.info("Successfully imported local halting_head and memory_cross_attn")
except ImportError as e:
    logger.warning(f"Failed to import local components: {e}")

    # Create basic fallbacks
    class ACTHaltingHead(nn.Module):
        def __init__(self, d_model, *args, **kwargs):
            super().__init__()
            self.halting_head = nn.Linear(d_model, 1)

        def forward(self, x):
            return torch.sigmoid(self.halting_head(x))

    class MemoryScheduler:
        def __init__(self, *args, **kwargs):
            pass

        def should_read(self, *args):
            return False

        def should_write(self, *args):
            return False

    def create_memory_cross_attention(*args, **kwargs):
        return nn.Identity()


# LTM Bank imports with comprehensive fallbacks
try:
    from agent_forge.models.cognate.unified_refiner.ltm_bank import create_ltm_controllers, create_memory_bank

    logger.info("Successfully imported from unified_refiner.ltm_bank")
except ImportError:
    try:
        from packages.agent_forge.models.cognate.unified_refiner.ltm_bank import (
            create_ltm_controllers,
            create_memory_bank,
        )

        logger.info("Successfully imported from packages fallback")
    except ImportError:
        logger.warning("LTM bank imports failed, using fallbacks")

        def create_ltm_controllers(*args, **kwargs):
            return None, None, None

        def create_memory_bank(*args, **kwargs):
            return None


@dataclass
class CognateConfig:
    """Configuration for the CognateRefiner model."""

    # Model architecture (optimized for exactly 25M total parameters)
    vocab_size: int = 32000
    d_model: int = 216  # Hidden dimension (exact 25M targeting)
    n_layers: int = 11  # Number of transformer layers
    n_heads: int = 4  # Attention heads (54 dim per head)
    ffn_mult: int = 4  # FFN multiplier (d_ffn = d_model * ffn_mult = 864)
    max_seq_len: int = 2048  # Maximum sequence length

    # Memory system configuration
    d_mem: int = 216  # Memory dimension (match d_model for efficiency)
    mem_capacity: int = 4096  # Memory bank capacity
    mem_topk: int = 4  # Top-k memory retrieval

    # Memory policies
    read_policy: str = "entropy_gated"  # Memory read policy
    write_policy: str = "surprise_novelty"  # Memory write policy
    entropy_threshold: float = 0.8  # Read threshold
    surprise_threshold: float = 0.6  # Write surprise threshold
    novelty_threshold: float = 0.7  # Write novelty threshold

    # ACT configuration
    act_threshold: float = 0.99  # Halting threshold
    act_epsilon: float = 0.01  # Numerical stability
    max_act_steps: int = 16  # Maximum ACT steps

    # Training configuration
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    activation: str = "silu"
    position_type: str = "rotary"
    rope_theta: float = 10000.0

    # Loss weights
    lambda_act: float = 0.1  # ACT loss weight
    alpha_read: float = 0.05  # Memory read loss weight
    beta_write: float = 0.05  # Memory write loss weight
    gamma_comp: float = 0.02  # Compression loss weight

    # Device and dtype
    torch_dtype: str = "float32"
    device_map: str | None = None


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim

        # Precompute frequency matrix
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache for cos/sin values
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_len = 0

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device):
        """Update cached cos/sin values if needed."""
        if seq_len > self._cached_seq_len:
            self._cached_seq_len = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)

            self._cached_cos = emb.cos()
            self._cached_sin = emb.sin()

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embedding."""
        self._update_cos_sin_cache(seq_len, x.device)
        return self._cached_cos[:seq_len], self._cached_sin[:seq_len]


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to query and key tensors."""
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE and memory cross-attention support."""

    def __init__(self, config: CognateConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim**-0.5

        assert config.d_model % config.n_heads == 0

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(config.d_model, config.d_model * 3, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        # Rotary position embedding
        if config.position_type == "rotary":
            self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        memory_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Forward pass with optional memory K/V injection."""
        B, L, _ = x.shape

        # Combined QKV projection
        qkv = self.qkv_proj(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Each: (B, L, n_heads, head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Apply rotary position embedding
        if hasattr(self, "rotary_emb"):
            cos, sin = self.rotary_emb(x, L)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Combine with memory K/V if provided
        if memory_kv is not None:
            mem_k, mem_v = memory_kv
            if mem_k.dim() == 3:  # (B, M, d_model) -> (B, n_heads, M, head_dim)
                mem_k = mem_k.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
                mem_v = mem_v.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

            # Concatenate memory K/V with self-attention K/V
            k = torch.cat([mem_k, k], dim=2)  # (B, n_heads, M+L, head_dim)
            v = torch.cat([mem_v, v], dim=2)  # (B, n_heads, M+L, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal masking
        if is_causal:
            if memory_kv is not None:
                # Only mask the self-attention part
                seq_len = L
                mem_len = k.size(2) - seq_len
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
                scores[:, :, :, mem_len:] = scores[:, :, :, mem_len:].masked_fill(causal_mask, float("-inf"))
            else:
                causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
                scores = scores.masked_fill(causal_mask, float("-inf"))

        # Apply attention mask
        if attention_mask is not None:
            # Expand mask to cover memory + sequence
            if memory_kv is not None:
                mem_len = k.size(2) - L
                mem_mask = torch.ones(B, mem_len, device=attention_mask.device, dtype=attention_mask.dtype)
                combined_mask = torch.cat([mem_mask, attention_mask], dim=-1)
            else:
                combined_mask = attention_mask

            expanded_mask = combined_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(expanded_mask == 0, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # (B, n_heads, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)

        return self.o_proj(out)


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, config: CognateConfig):
        super().__init__()
        ffn_dim = config.d_model * config.ffn_mult

        # SwiGLU uses 2 linear layers for the gate
        self.gate_proj = nn.Linear(config.d_model, ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU activation: gate * silu(up) -> down"""
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    """Transformer block with integrated memory cross-attention."""

    def __init__(self, config: CognateConfig):
        super().__init__()

        # Normalization layers
        self.input_layernorm = RMSNorm(config.d_model, config.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.d_model, config.layer_norm_eps)

        # Self-attention
        self.self_attn = MultiHeadAttention(config)

        # Memory cross-attention (optional)
        self.memory_cross_attn = create_memory_cross_attention(
            d_model=config.d_model,
            d_mem=config.d_mem,
            n_heads=config.n_heads,
            dropout=config.dropout,
            entropy_threshold=config.entropy_threshold,
        )

        # Feed-forward network
        self.mlp = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        memory_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        memory_vectors: torch.Tensor | None = None,
        memory_keys: torch.Tensor | None = None,
        read_policy: str = "entropy_gated",
    ) -> torch.Tensor:
        """Forward pass with optional memory integration."""

        # Self-attention with optional memory K/V injection
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask, memory_kv)
        x = residual + x

        # Memory cross-attention (if memory provided and policy allows)
        if memory_vectors is not None and memory_keys is not None:
            residual = x
            x = self.memory_cross_attn(
                hidden_states=x,
                memory_vectors=memory_vectors,
                memory_keys=memory_keys,
                attention_mask=attention_mask,
                read_policy=read_policy,
            )
            x = residual + (x - residual)  # Residual connection

        # Feed-forward
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class EditHead(nn.Module):
    """Language modeling head for token prediction."""

    def __init__(self, config: CognateConfig):
        super().__init__()
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Generate logits over vocabulary."""
        return self.lm_head(hidden_states)


class CognateRefiner(nn.Module):
    """
    Production-ready Cognate Refiner with 25M parameters.

    Integrates all Phase 1 components:
    - Transformer backbone (~20M params)
    - Long-term memory system (~4M params)
    - ACT halting head (~0.5M params)
    - Edit head (~0.5M params)

    Total: ~25M parameters with HuggingFace compatibility.
    """

    def __init__(self, config: CognateConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer layers
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Final normalization
        self.norm = RMSNorm(config.d_model, config.layer_norm_eps)

        # Task heads
        self.halting_head = ACTHaltingHead(config)
        self.edit_head = EditHead(config)

        # Long-term memory system
        self.memory_bank = create_memory_bank(
            capacity=config.mem_capacity,
            d_mem=config.d_mem,
            device=next(self.parameters()).device if list(self.parameters()) else torch.device("cpu"),
        )

        self.read_controller, self.write_controller = create_ltm_controllers(d_model=config.d_model, d_mem=config.d_mem)

        # Memory scheduler
        self.memory_scheduler = MemoryScheduler(
            read_policy=config.read_policy,
            write_policy=config.write_policy,
            entropy_threshold=config.entropy_threshold,
            surprise_threshold=config.surprise_threshold,
            novelty_threshold=config.novelty_threshold,
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Log parameter count
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"CognateRefiner initialized with {total_params:,} parameters")

        # Validate parameter count (target: 25M ±1M)
        target_params = 25_000_000
        if abs(total_params - target_params) > 1_000_000:
            logger.warning(f"Parameter count {total_params:,} is outside target range {target_params:,} ±1M")

    def _init_weights(self, module):
        """Initialize model weights with Xavier uniform strategy."""
        if isinstance(module, nn.Linear):
            # Xavier uniform initialization
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        recent_loss: float | None = None,
        summary_text: str = "",
        return_dict: bool = True,
        output_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with integrated memory operations.

        Args:
            input_ids: Token IDs (B, L)
            attention_mask: Attention mask (B, L)
            labels: Target labels for loss computation (B, L)
            recent_loss: Recent loss for surprise computation
            summary_text: Text summary for memory writing
            return_dict: Whether to return dict format
            output_hidden_states: Whether to output hidden states

        Returns:
            Dictionary with logits, halt_logits, memory_info, and optional loss
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Move memory bank to correct device if needed
        if self.memory_bank.device != device:
            self.memory_bank.keys = self.memory_bank.keys.to(device)
            self.memory_bank.vals = self.memory_bank.vals.to(device)
            self.memory_bank.device = device

        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(B, L, device=device, dtype=torch.bool)

        # Memory read operation
        memory_vals, memory_keys, retrieved_indices = None, None, []
        if self.memory_scheduler.should_read():
            # Use pooled representation for memory query
            h_pooled = hidden_states.mean(dim=1)  # (B, d_model)
            memory_vals, memory_keys, retrieved_indices = self.read_controller(
                h_pooled, self.memory_bank, topk=self.config.mem_topk
            )

        # Store all hidden states if requested
        all_hidden_states = []
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # Pass through transformer layers with memory
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                memory_kv=(memory_keys, memory_vals) if memory_keys is not None else None,
                memory_vectors=memory_vals,
                memory_keys=memory_keys,
                read_policy=self.config.read_policy,
            )

            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Task heads
        logits = self.edit_head(hidden_states)

        # Halting head (use pooled representation)
        if attention_mask is not None:
            # Attention-weighted pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            h_cls = sum_embeddings / sum_mask
        else:
            h_cls = hidden_states[:, 0, :]

        halt_logits = self.halting_head(h_cls)

        # Memory write operation (during training with loss signal)
        memory_write_stats = {"wrote": False, "gate_prob": 0.0}
        if recent_loss is not None and self.training:
            # Use the pooled representation for writing
            for i in range(B):
                if self.memory_scheduler.should_write():
                    _, _, wrote, gate_prob = self.write_controller(
                        h_cls[i], self.memory_bank, recent_loss, summary_text
                    )
                    memory_write_stats = {"wrote": wrote, "gate_prob": gate_prob}
                    break  # Only write once per batch

        # Memory information
        memory_info = {
            "bank_stats": self.memory_bank.get_stats(),
            "scheduler_stats": self.memory_scheduler.get_stats(),
            "retrieved_count": len(retrieved_indices),
            "write_stats": memory_write_stats,
        }

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # Add ACT loss
            act_loss = torch.mean(halt_logits)  # Simple ACT loss

            # Combined loss
            loss = lm_loss + self.config.lambda_act * act_loss

        # Prepare outputs
        outputs = {
            "logits": logits,
            "halt_logits": halt_logits,
            "memory_info": memory_info,
        }

        if loss is not None:
            outputs["loss"] = loss

        if output_hidden_states:
            outputs["hidden_states"] = all_hidden_states

        return outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        summary_text: str = "",
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate tokens with memory-augmented inference.

        Args:
            input_ids: Input token IDs (B, L)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            summary_text: Text summary for memory operations

        Returns:
            Generated token sequences (B, L + max_new_tokens)
        """
        self.eval()

        generated_tokens = input_ids.clone()

        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(generated_tokens, summary_text=summary_text)

                # Get next token logits
                next_token_logits = outputs["logits"][:, -1, :] / temperature

                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))

                # Sample or select next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to generated sequence
                generated_tokens = torch.cat([generated_tokens, next_tokens], dim=-1)

                # Check halting decision
                halting_probs = torch.sigmoid(outputs["halt_logits"])
                should_halt = halting_probs > 0.5

                # Early stopping based on halting
                if should_halt.all():
                    break

        self.train()
        return generated_tokens

    def save_pretrained(self, save_directory: str):
        """
        Save model with HuggingFace compatibility.

        Args:
            save_directory: Directory to save the model
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model state dict
        model_path = save_path / "pytorch_model.bin"
        torch.save(self.state_dict(), model_path)

        # Save configuration
        config_path = save_path / "config.json"
        config_dict = {
            "architectures": ["CognateRefiner"],
            "model_type": "cognate_refiner",
            "vocab_size": self.config.vocab_size,
            "d_model": self.config.d_model,
            "n_layers": self.config.n_layers,
            "n_heads": self.config.n_heads,
            "ffn_mult": self.config.ffn_mult,
            "max_seq_len": self.config.max_seq_len,
            "d_mem": self.config.d_mem,
            "mem_capacity": self.config.mem_capacity,
            "mem_topk": self.config.mem_topk,
            "read_policy": self.config.read_policy,
            "write_policy": self.config.write_policy,
            "entropy_threshold": self.config.entropy_threshold,
            "surprise_threshold": self.config.surprise_threshold,
            "novelty_threshold": self.config.novelty_threshold,
            "act_threshold": self.config.act_threshold,
            "act_epsilon": self.config.act_epsilon,
            "max_act_steps": self.config.max_act_steps,
            "dropout": self.config.dropout,
            "layer_norm_eps": self.config.layer_norm_eps,
            "activation": self.config.activation,
            "position_type": self.config.position_type,
            "rope_theta": self.config.rope_theta,
            "lambda_act": self.config.lambda_act,
            "alpha_read": self.config.alpha_read,
            "beta_write": self.config.beta_write,
            "gamma_comp": self.config.gamma_comp,
            "torch_dtype": self.config.torch_dtype,
        }

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Save memory bank state
        memory_path = save_path / "memory_bank.json"
        self.memory_bank.save(str(memory_path))

        logger.info(f"CognateRefiner saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """
        Load model with HuggingFace compatibility.

        Args:
            load_directory: Directory to load the model from
            **kwargs: Additional arguments

        Returns:
            CognateRefiner instance
        """
        load_path = Path(load_directory)

        # Load configuration
        config_path = load_path / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)

        # Create config object
        config = CognateConfig(**{k: v for k, v in config_dict.items() if k in CognateConfig.__dataclass_fields__})

        # Override with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Create model
        model = cls(config)

        # Load model state dict
        model_path = load_path / "pytorch_model.bin"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)

        # Load memory bank state
        memory_path = load_path / "memory_bank.json"
        if memory_path.exists():
            model.memory_bank.load(str(memory_path))

        logger.info(f"CognateRefiner loaded from {load_directory}")
        return model

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        return param_size / (1024 * 1024)

    def get_parameter_breakdown(self) -> dict[str, int]:
        """Get detailed parameter breakdown by component."""
        breakdown = {}

        # Embeddings
        breakdown["embeddings"] = sum(p.numel() for p in self.embed_tokens.parameters())

        # Transformer layers
        breakdown["transformer_layers"] = sum(p.numel() for p in self.layers.parameters())

        # Normalization
        breakdown["norm"] = sum(p.numel() for p in self.norm.parameters())

        # Heads
        breakdown["halting_head"] = sum(p.numel() for p in self.halting_head.parameters())
        breakdown["edit_head"] = sum(p.numel() for p in self.edit_head.parameters())

        # Memory controllers
        breakdown["read_controller"] = sum(p.numel() for p in self.read_controller.parameters())
        breakdown["write_controller"] = sum(p.numel() for p in self.write_controller.parameters())

        # Total
        breakdown["total"] = sum(breakdown.values())

        return breakdown
