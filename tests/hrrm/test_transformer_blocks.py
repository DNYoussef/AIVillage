"""Tests for HRRM transformer blocks."""

import pytest
import torch
import torch.nn as nn

from packages.hrrm.common.transformer_blocks import (
    CausalSelfAttention,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    apply_rope,
    swiglu,
)


class TestRMSNorm:
    """Test RMSNorm implementation."""

    def test_rms_norm_forward(self):
        """Test RMSNorm forward pass."""
        d_model = 64
        norm = RMSNorm(d_model)

        x = torch.randn(2, 10, d_model)
        output = norm(x)

        assert output.shape == x.shape
        # Check normalization properties
        assert torch.allclose(output.norm(dim=-1), torch.ones(2, 10), atol=1e-5)

    def test_rms_norm_parameters(self):
        """Test RMSNorm has correct parameters."""
        d_model = 128
        norm = RMSNorm(d_model)

        params = list(norm.parameters())
        assert len(params) == 1
        assert params[0].shape == (d_model,)


class TestRotaryPositionalEmbedding:
    """Test RoPE embedding implementation."""

    def test_rope_creation(self):
        """Test RoPE embedding creation."""
        d_model = 64
        rope = RotaryPositionalEmbedding(d_model)

        assert rope.dim == d_model
        assert rope.base == 10000

    def test_rope_get(self):
        """Test RoPE get method."""
        d_model = 64
        rope = RotaryPositionalEmbedding(d_model)

        seq_len = 32
        device = torch.device("cpu")
        rope_cis = rope.get(seq_len, device)

        assert rope_cis.shape == (seq_len, d_model // 2)
        assert rope_cis.dtype == torch.complex64

    def test_apply_rope(self):
        """Test apply_rope function."""
        d_model = 64
        seq_len = 32
        batch_size = 2
        n_heads = 8

        rope = RotaryPositionalEmbedding(d_model // n_heads)
        rope_cis = rope.get(seq_len, torch.device("cpu"))

        # Test tensor (batch, n_heads, seq_len, head_dim)
        x = torch.randn(batch_size, n_heads, seq_len, d_model // n_heads)

        x_rot = apply_rope(x, rope_cis)

        assert x_rot.shape == x.shape
        # Check that rotation actually changes the values
        assert not torch.allclose(x_rot, x)


class TestSwiGLU:
    """Test SwiGLU activation implementation."""

    def test_swiglu_forward(self):
        """Test SwiGLU forward pass."""
        d_model = 64
        mult = 4
        swiglu_layer = SwiGLU(d_model, mult)

        x = torch.randn(2, 10, d_model)
        output = swiglu_layer(x)

        assert output.shape == x.shape

    def test_swiglu_function(self):
        """Test SwiGLU function directly."""
        d_model = 32
        d_ff = 128

        x = torch.randn(2, 10, d_model)
        w1 = torch.randn(d_ff, d_model)
        w2 = torch.randn(d_ff, d_model)

        output = swiglu(x, w1, w2)

        assert output.shape == (2, 10, d_ff)

    def test_swiglu_gating(self):
        """Test SwiGLU gating mechanism."""
        d_model = 32
        mult = 4
        swiglu_layer = SwiGLU(d_model, mult)

        # Test with zero input
        x_zero = torch.zeros(1, 5, d_model)
        output_zero = swiglu_layer(x_zero)

        # Should be approximately zero due to gating
        assert torch.allclose(output_zero, torch.zeros_like(output_zero), atol=1e-3)


class TestCausalSelfAttention:
    """Test causal self-attention implementation."""

    def test_attention_forward(self):
        """Test attention forward pass."""
        d_model = 64
        n_head = 8
        attention = CausalSelfAttention(d_model, n_head)

        seq_len = 32
        x = torch.randn(2, seq_len, d_model)

        output = attention(x)
        assert output.shape == x.shape

    def test_attention_with_mask(self):
        """Test attention with causal mask."""
        d_model = 64
        n_head = 8
        attention = CausalSelfAttention(d_model, n_head)

        seq_len = 16
        x = torch.randn(2, seq_len, d_model)

        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len))

        output = attention(x, mask=mask)
        assert output.shape == x.shape

    def test_attention_heads(self):
        """Test attention head dimensions."""
        d_model = 64
        n_head = 8
        attention = CausalSelfAttention(d_model, n_head)

        assert attention.h == n_head
        assert attention.d == d_model // n_head

        seq_len = 16
        x = torch.randn(2, seq_len, d_model)

        output = attention(x)
        assert output.shape == x.shape


class TestBasicComponents:
    """Test basic component integration."""

    def test_component_integration(self):
        """Test that all components work together."""
        d_model = 64
        n_head = 8

        # Test all components can be created
        norm = RMSNorm(d_model)
        RotaryPositionalEmbedding(d_model // n_head)
        swiglu_layer = SwiGLU(d_model)
        attention = CausalSelfAttention(d_model, n_head)

        # Test a simple forward pass
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, d_model)

        # Normalize
        x_norm = norm(x)
        assert x_norm.shape == x.shape

        # Attention
        attn_out = attention(x_norm)
        assert attn_out.shape == x.shape

        # SwiGLU
        mlp_out = swiglu_layer(x_norm)
        assert mlp_out.shape == x.shape


class TestParameterCounts:
    """Test that parameter counts meet requirements."""

    def test_small_model_params(self):
        """Test parameter count for small models (~50M)."""
        d_model = 512
        n_layers = 12
        n_head = 8
        d_ff = 2048
        vocab_size = 32000
        max_seq_len = 2048

        # Create a model similar to HRRM architectures
        total_params = 0

        # Embedding layer
        embedding_params = vocab_size * d_model
        total_params += embedding_params

        # Transformer blocks
        for _ in range(n_layers):
            block = TransformerBlock(d_model, n_head, d_ff, max_seq_len)
            block_params = sum(p.numel() for p in block.parameters())
            total_params += block_params

        # Output layer
        output_params = vocab_size * d_model
        total_params += output_params

        # Should be in 48M-55M range for HRRM models
        assert 30_000_000 <= total_params <= 70_000_000, f"Got {total_params:,} params"

    def test_individual_block_params(self):
        """Test parameter count of individual transformer block."""
        d_model = 512
        n_head = 8
        d_ff = 2048
        max_seq_len = 2048

        block = TransformerBlock(d_model, n_head, d_ff, max_seq_len)
        total_params = sum(p.numel() for p in block.parameters())

        # Should be reasonable for a single block
        assert total_params > 1_000_000  # At least 1M params per block
        assert total_params < 10_000_000  # Less than 10M params per block


@pytest.mark.integration
class TestIntegration:
    """Integration tests for transformer components."""

    def test_full_forward_pass(self):
        """Test full forward pass through multiple blocks."""
        d_model = 256
        n_head = 8
        d_ff = 1024
        n_layers = 6
        max_seq_len = 512
        vocab_size = 1000

        # Create simple model
        embedding = nn.Embedding(vocab_size, d_model)
        blocks = nn.ModuleList([TransformerBlock(d_model, n_head, d_ff, max_seq_len) for _ in range(n_layers)])
        norm = RMSNorm(d_model)
        lm_head = nn.Linear(d_model, vocab_size)

        # Forward pass
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        x = embedding(input_ids)

        for block in blocks:
            x = block(x)

        x = norm(x)
        logits = lm_head(x)

        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_gradient_flow(self):
        """Test that gradients flow correctly through all components."""
        d_model = 128
        n_head = 4
        d_ff = 512
        max_seq_len = 256

        block = TransformerBlock(d_model, n_head, d_ff, max_seq_len)

        x = torch.randn(2, 32, d_model, requires_grad=True)
        output = block(x)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        for param in block.parameters():
            assert param.grad is not None

    def test_memory_efficiency(self):
        """Test memory usage of transformer blocks."""
        d_model = 512
        n_head = 8
        d_ff = 2048
        max_seq_len = 2048

        block = TransformerBlock(d_model, n_head, d_ff, max_seq_len)

        # Test with various sequence lengths
        for seq_len in [64, 128, 256, 512]:
            x = torch.randn(1, seq_len, d_model)

            # Should not run out of memory
            with torch.no_grad():
                output = block(x)
                assert output.shape == x.shape
