"""Basic tests for HRRM transformer components."""

import torch
from packages.hrrm.common.transformer_blocks import (
    CausalSelfAttention,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
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

    def test_attention_heads(self):
        """Test attention head dimensions."""
        d_model = 64
        n_head = 8
        attention = CausalSelfAttention(d_model, n_head)

        assert attention.h == n_head
        assert attention.d == d_model // n_head


class TestBasicIntegration:
    """Test basic component integration."""

    def test_all_components_work(self):
        """Test that all components can be used together."""
        d_model = 64
        n_head = 8

        # Create components
        norm = RMSNorm(d_model)
        rope = RotaryPositionalEmbedding(d_model // n_head)
        swiglu_layer = SwiGLU(d_model)
        attention = CausalSelfAttention(d_model, n_head)

        # Test data
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, d_model)

        # Forward passes
        x_norm = norm(x)
        assert x_norm.shape == x.shape

        attn_out = attention(x_norm)
        assert attn_out.shape == x.shape

        mlp_out = swiglu_layer(x_norm)
        assert mlp_out.shape == x.shape

        # Test RoPE
        rope_cis = rope.get(seq_len, x.device)
        assert rope_cis.shape == (seq_len, d_model // n_head // 2)
