"""
BitNet Core Layer Implementations for Agent Forge Phase 4
=========================================================

Core BitNet layer implementations including BitNetLinear, BitNetAttention,
and BitNetTransformer. These layers provide 1-bit quantization while maintaining
model performance and compatibility with existing architectures.

Key Features:
- 1-bit and ternary weight quantization
- Memory-efficient attention mechanisms
- Gradient computation for quantized weights
- Integration with EvoMerge and Quiet-STaR phases

Author: BitNet Core Implementation Specialist - Agent Forge Phase 4
"""

import math
import warnings
from typing import Optional, Tuple, Union, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

from ..config.bitnet_config import BitNetConfig, QuantizationMode


class BitNetQuantizer:
    """Core quantization functions for BitNet layers."""

    @staticmethod
    def ternary_quantize(x: Tensor, scale: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Ternary quantization: quantize weights to {-1, 0, 1}.

        Args:
            x: Input tensor to quantize
            scale: Optional scaling factor

        Returns:
            Tuple of (quantized_tensor, scale_factor)
        """
        if scale is None:
            # Calculate scale as absolute mean of non-zero elements
            abs_mean = x.abs().mean()
            scale = abs_mean.clamp(min=1e-8)

        # Normalize by scale
        x_normalized = x / scale

        # Quantize to {-1, 0, 1}
        quantized = torch.sign(x_normalized)

        # Apply threshold to create zeros (optional for sparse representation)
        threshold = 0.5
        mask = x_normalized.abs() > threshold
        quantized = quantized * mask.float()

        return quantized, scale

    @staticmethod
    def binary_quantize(x: Tensor, scale: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Binary quantization: quantize weights to {-1, 1}.

        Args:
            x: Input tensor to quantize
            scale: Optional scaling factor

        Returns:
            Tuple of (quantized_tensor, scale_factor)
        """
        if scale is None:
            # Calculate scale as absolute mean
            scale = x.abs().mean().clamp(min=1e-8)

        # Normalize and quantize
        x_normalized = x / scale
        quantized = torch.sign(x_normalized)

        # Handle zero case
        quantized = torch.where(quantized == 0, torch.ones_like(quantized), quantized)

        return quantized, scale

    @staticmethod
    def quantize_weights(weights: Tensor, mode: QuantizationMode) -> Tuple[Tensor, Tensor]:
        """
        Quantize weights according to specified mode.

        Args:
            weights: Weight tensor to quantize
            mode: Quantization mode (ternary or binary)

        Returns:
            Tuple of (quantized_weights, scale_factors)
        """
        if mode == QuantizationMode.TERNARY:
            return BitNetQuantizer.ternary_quantize(weights)
        elif mode == QuantizationMode.BINARY:
            return BitNetQuantizer.binary_quantize(weights)
        elif mode == QuantizationMode.ABSMEAN:
            # Absolute mean scaling without quantization (for gradual conversion)
            scale = weights.abs().mean().clamp(min=1e-8)
            return weights / scale, scale
        else:
            raise ValueError(f"Unsupported quantization mode: {mode}")


class BitNetLinear(nn.Module):
    """
    BitNet Linear Layer with 1-bit/ternary quantization.

    Replaces standard Linear layers with quantized weights while maintaining
    compatibility with existing model architectures.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quantization_mode: QuantizationMode = QuantizationMode.TERNARY,
        weight_scaling: bool = True,
        activation_scaling: bool = True,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantization_mode = quantization_mode
        self.weight_scaling = weight_scaling
        self.activation_scaling = activation_scaling

        # Initialize weights
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        # Scaling factors for quantized weights
        self.register_buffer('weight_scale', torch.ones(1, **factory_kwargs))

        # Activation scaling for better quantization
        if activation_scaling:
            self.register_buffer('activation_scale', torch.ones(1, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def quantize_weights(self) -> Tuple[Tensor, Tensor]:
        """Quantize the layer weights."""
        return BitNetQuantizer.quantize_weights(self.weight, self.quantization_mode)

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass with quantized weights."""
        # Quantize weights on-the-fly during inference
        if self.training:
            # During training, use straight-through estimator
            quantized_weight, scale = self.quantize_weights()
            # Straight-through: forward with quantized, backward with original
            quantized_weight = quantized_weight + self.weight - self.weight.detach()
        else:
            # During inference, use actual quantized weights
            quantized_weight, scale = self.quantize_weights()

        # Store scale for potential use in other layers
        self.weight_scale.data = scale

        # Apply activation scaling if enabled
        if self.activation_scaling and hasattr(self, 'activation_scale'):
            # Simple activation scaling based on input magnitude
            input_scale = input.abs().mean().clamp(min=1e-8)
            self.activation_scale.data = input_scale
            scaled_input = input / input_scale
        else:
            scaled_input = input
            input_scale = torch.ones_like(self.weight_scale)

        # Perform linear transformation
        output = F.linear(scaled_input, quantized_weight * scale, self.bias)

        # Rescale output if activation scaling was applied
        if self.activation_scaling and hasattr(self, 'activation_scale'):
            output = output * input_scale

        return output

    def extra_repr(self) -> str:
        """String representation of the layer."""
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.bias is not None}, quantization_mode={self.quantization_mode.value}'

    @classmethod
    def from_linear(cls, linear_layer: nn.Linear, quantization_mode: QuantizationMode = QuantizationMode.TERNARY) -> 'BitNetLinear':
        """
        Convert a standard Linear layer to BitNetLinear.

        Args:
            linear_layer: The standard Linear layer to convert
            quantization_mode: Quantization mode to use

        Returns:
            BitNetLinear layer with copied weights
        """
        bitnet_layer = cls(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            bias=linear_layer.bias is not None,
            quantization_mode=quantization_mode,
            device=linear_layer.weight.device,
            dtype=linear_layer.weight.dtype
        )

        # Copy weights and bias
        with torch.no_grad():
            bitnet_layer.weight.copy_(linear_layer.weight)
            if linear_layer.bias is not None:
                bitnet_layer.bias.copy_(linear_layer.bias)

        return bitnet_layer


class BitNetAttention(nn.Module):
    """
    BitNet Attention mechanism with quantized linear projections.

    Maintains attention computation while using quantized weights for
    query, key, value, and output projections.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        quantization_mode: QuantizationMode = QuantizationMode.TERNARY,
        preserve_attention_weights: bool = True,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.preserve_attention_weights = preserve_attention_weights

        assert self.head_dim * num_heads == self.embed_dim, \
            f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"

        # Query, Key, Value projections with quantization
        if preserve_attention_weights:
            # Use standard linear layers for critical attention computations
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
        else:
            # Use quantized projections for maximum compression
            self.q_proj = BitNetLinear(embed_dim, embed_dim, bias=bias,
                                     quantization_mode=quantization_mode, device=device, dtype=dtype)
            self.k_proj = BitNetLinear(embed_dim, embed_dim, bias=bias,
                                     quantization_mode=quantization_mode, device=device, dtype=dtype)
            self.v_proj = BitNetLinear(embed_dim, embed_dim, bias=bias,
                                     quantization_mode=quantization_mode, device=device, dtype=dtype)

        # Output projection (often quantized for compression)
        self.out_proj = BitNetLinear(embed_dim, embed_dim, bias=bias,
                                   quantization_mode=quantization_mode, device=device, dtype=dtype)

        self.scaling = self.head_dim ** -0.5

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        average_attn_weights: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass for BitNet attention.

        Args:
            query: Query tensor [seq_len, batch_size, embed_dim]
            key: Key tensor (defaults to query)
            value: Value tensor (defaults to query)
            attn_mask: Attention mask
            key_padding_mask: Key padding mask
            need_weights: Whether to return attention weights
            average_attn_weights: Whether to average attention weights across heads

        Returns:
            Tuple of (output, attention_weights)
        """
        if key is None:
            key = query
        if value is None:
            value = query

        seq_len, batch_size, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        # Project to Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        q = q.contiguous().view(seq_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        # Scaled dot-product attention
        attn_output, attn_weights = self._attention(q, k, v, attn_mask, key_padding_mask)

        # Reshape and project output
        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, batch_size, embed_dim)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.view(batch_size, self.num_heads, seq_len, -1)
                attn_weights = attn_weights.mean(dim=1)
            return attn_output, attn_weights
        else:
            return attn_output, None

    def _attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Core attention computation."""
        batch_size = query.size(0)
        src_len = key.size(1)
        tgt_len = query.size(1)

        # Compute attention scores
        attn_weights = torch.bmm(query, key.transpose(1, 2)) * self.scaling

        # Apply attention mask
        if attn_mask is not None:
            attn_weights += attn_mask

        # Apply key padding mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(batch_size, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
            attn_weights = attn_weights.view(batch_size * self.num_heads, tgt_len, src_len)

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention to values
        attn_output = torch.bmm(attn_weights, value)

        return attn_output, attn_weights

    @classmethod
    def from_multihead_attention(
        cls,
        mha: nn.MultiheadAttention,
        quantization_mode: QuantizationMode = QuantizationMode.TERNARY,
        preserve_attention_weights: bool = True
    ) -> 'BitNetAttention':
        """
        Convert a standard MultiheadAttention to BitNetAttention.

        Args:
            mha: The standard MultiheadAttention layer to convert
            quantization_mode: Quantization mode to use
            preserve_attention_weights: Whether to preserve attention weight precision

        Returns:
            BitNetAttention layer with copied weights
        """
        bitnet_attn = cls(
            embed_dim=mha.embed_dim,
            num_heads=mha.num_heads,
            dropout=mha.dropout,
            bias=mha.in_proj_bias is not None,
            quantization_mode=quantization_mode,
            preserve_attention_weights=preserve_attention_weights,
            device=mha.in_proj_weight.device,
            dtype=mha.in_proj_weight.dtype
        )

        # Copy weights from the original attention layer
        with torch.no_grad():
            embed_dim = mha.embed_dim

            # Split the combined in_proj_weight into Q, K, V
            q_weight = mha.in_proj_weight[:embed_dim, :]
            k_weight = mha.in_proj_weight[embed_dim:2*embed_dim, :]
            v_weight = mha.in_proj_weight[2*embed_dim:, :]

            bitnet_attn.q_proj.weight.copy_(q_weight)
            bitnet_attn.k_proj.weight.copy_(k_weight)
            bitnet_attn.v_proj.weight.copy_(v_weight)

            if mha.in_proj_bias is not None:
                q_bias = mha.in_proj_bias[:embed_dim]
                k_bias = mha.in_proj_bias[embed_dim:2*embed_dim]
                v_bias = mha.in_proj_bias[2*embed_dim:]

                bitnet_attn.q_proj.bias.copy_(q_bias)
                bitnet_attn.k_proj.bias.copy_(k_bias)
                bitnet_attn.v_proj.bias.copy_(v_bias)

            # Copy output projection
            bitnet_attn.out_proj.weight.copy_(mha.out_proj.weight)
            if mha.out_proj.bias is not None:
                bitnet_attn.out_proj.bias.copy_(mha.out_proj.bias)

        return bitnet_attn


class BitNetTransformer(nn.Module):
    """
    Complete BitNet Transformer block with quantized layers.

    Combines BitNet attention and feed-forward layers into a complete
    transformer block suitable for language model architectures.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        quantization_mode: QuantizationMode = QuantizationMode.TERNARY,
        preserve_norm_layers: bool = True,
        preserve_attention_weights: bool = True,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        # Self-attention layer
        self.self_attn = BitNetAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            quantization_mode=quantization_mode,
            preserve_attention_weights=preserve_attention_weights,
            device=device,
            dtype=dtype
        )

        # Feed-forward network
        self.linear1 = BitNetLinear(embed_dim, ffn_dim, device=device, dtype=dtype,
                                  quantization_mode=quantization_mode)
        self.linear2 = BitNetLinear(ffn_dim, embed_dim, device=device, dtype=dtype,
                                  quantization_mode=quantization_mode)

        # Layer normalization (typically preserved for stability)
        if preserve_norm_layers:
            self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps, device=device, dtype=dtype)
            self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps, device=device, dtype=dtype)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "swish":
            self.activation = F.silu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass for BitNet transformer block.

        Args:
            src: Source tensor [seq_len, batch_size, embed_dim]
            src_mask: Source attention mask
            src_key_padding_mask: Source key padding mask

        Returns:
            Output tensor with same shape as input
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False
        )
        src = src + self.dropout(attn_output)
        src = self.norm1(src)

        # Feed-forward with residual connection
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(ffn_output)
        src = self.norm2(src)

        return src

    @classmethod
    def from_transformer_layer(
        cls,
        transformer_layer: nn.TransformerEncoderLayer,
        quantization_mode: QuantizationMode = QuantizationMode.TERNARY,
        preserve_norm_layers: bool = True,
        preserve_attention_weights: bool = True
    ) -> 'BitNetTransformer':
        """
        Convert a standard TransformerEncoderLayer to BitNetTransformer.

        Args:
            transformer_layer: The standard transformer layer to convert
            quantization_mode: Quantization mode to use
            preserve_norm_layers: Whether to preserve layer norm precision
            preserve_attention_weights: Whether to preserve attention weight precision

        Returns:
            BitNetTransformer layer with copied weights
        """
        # Extract parameters from the original layer
        embed_dim = transformer_layer.self_attn.embed_dim
        num_heads = transformer_layer.self_attn.num_heads

        # Determine FFN dimension from linear1 layer
        ffn_dim = transformer_layer.linear1.out_features

        # Determine activation function (limited detection)
        activation = "relu"  # Default, could be improved with better detection

        bitnet_layer = cls(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=transformer_layer.dropout.p if hasattr(transformer_layer.dropout, 'p') else 0.1,
            activation=activation,
            quantization_mode=quantization_mode,
            preserve_norm_layers=preserve_norm_layers,
            preserve_attention_weights=preserve_attention_weights,
            device=next(transformer_layer.parameters()).device,
            dtype=next(transformer_layer.parameters()).dtype
        )

        # Copy weights
        with torch.no_grad():
            # Copy attention weights
            bitnet_layer.self_attn = BitNetAttention.from_multihead_attention(
                transformer_layer.self_attn, quantization_mode, preserve_attention_weights
            )

            # Copy FFN weights
            bitnet_layer.linear1 = BitNetLinear.from_linear(
                transformer_layer.linear1, quantization_mode
            )
            bitnet_layer.linear2 = BitNetLinear.from_linear(
                transformer_layer.linear2, quantization_mode
            )

            # Copy norm layers if preserved
            if preserve_norm_layers:
                bitnet_layer.norm1.load_state_dict(transformer_layer.norm1.state_dict())
                bitnet_layer.norm2.load_state_dict(transformer_layer.norm2.state_dict())

        return bitnet_layer


def convert_linear_to_bitnet(model: nn.Module, quantization_mode: QuantizationMode = QuantizationMode.TERNARY) -> nn.Module:
    """
    Recursively convert all Linear layers in a model to BitNetLinear layers.

    Args:
        model: The model to convert
        quantization_mode: Quantization mode to use for conversion

    Returns:
        Model with BitNetLinear layers replacing Linear layers
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace Linear with BitNetLinear
            bitnet_layer = BitNetLinear.from_linear(module, quantization_mode)
            setattr(model, name, bitnet_layer)
        else:
            # Recursively convert child modules
            convert_linear_to_bitnet(module, quantization_mode)

    return model


def convert_attention_to_bitnet(model: nn.Module, quantization_mode: QuantizationMode = QuantizationMode.TERNARY,
                              preserve_attention_weights: bool = True) -> nn.Module:
    """
    Recursively convert all MultiheadAttention layers in a model to BitNetAttention layers.

    Args:
        model: The model to convert
        quantization_mode: Quantization mode to use for conversion
        preserve_attention_weights: Whether to preserve attention weight precision

    Returns:
        Model with BitNetAttention layers replacing MultiheadAttention layers
    """
    for name, module in model.named_children():
        if isinstance(module, nn.MultiheadAttention):
            # Replace MultiheadAttention with BitNetAttention
            bitnet_attn = BitNetAttention.from_multihead_attention(
                module, quantization_mode, preserve_attention_weights
            )
            setattr(model, name, bitnet_attn)
        else:
            # Recursively convert child modules
            convert_attention_to_bitnet(module, quantization_mode, preserve_attention_weights)

    return model