"""
Transformer² (T²) Mixer - Dynamic expert dispatch and low-rank mixing.
Implements two-pass dispatch → low-rank (singular-component) mixing at inference.
"""

import logging
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import FeatureExtractor

logger = logging.getLogger(__name__)


class ExpertAdapter:
    """
    Low-rank expert adapter using singular value decomposition.
    Represents an expert as U @ S @ V^T where rank is controlled by number of singular values.
    """

    def __init__(
        self,
        expert_spec: dict[str, Any],
        target_layers: list[str],
        model_shapes: dict[str, tuple[int, ...]],
    ):
        self.spec = expert_spec
        self.target_layers = target_layers
        self.model_shapes = model_shapes

        # Parse expert configuration
        self.rank = expert_spec.get("rank", 4)
        self.svd_scope = expert_spec.get("svd_scope", "per-matrix")
        self.init_method = expert_spec.get("init", "random")
        self.activation_rule = expert_spec.get("activation_rule", "gated")

        # Low-rank decomposition storage
        self.adapters: dict[str, dict[str, torch.Tensor]] = {}

        # Initialize adapters
        self._initialize_adapters()

    def _initialize_adapters(self):
        """Initialize low-rank adapters for target layers."""
        for layer_name in self.target_layers:
            if layer_name not in self.model_shapes:
                logger.warning(f"Layer {layer_name} not found in model shapes")
                continue

            shape = self.model_shapes[layer_name]
            if len(shape) != 2:  # Expecting weight matrices
                logger.warning(f"Layer {layer_name} has unexpected shape {shape}")
                continue

            rows, cols = shape
            effective_rank = min(self.rank, min(rows, cols))

            if self.init_method == "random":
                # Random initialization with appropriate scaling
                scale = 0.01 / np.sqrt(effective_rank)
                U = torch.randn(rows, effective_rank) * scale
                S = torch.ones(effective_rank) * scale
                V = torch.randn(cols, effective_rank) * scale

            elif self.init_method == "pca_activations":
                # PCA-based initialization (simplified - would need actual activations)
                U, S, V = self._pca_init(rows, cols, effective_rank)

            elif self.init_method == "fisher":
                # Fisher information-based initialization
                U, S, V = self._fisher_init(rows, cols, effective_rank)

            else:
                logger.warning(f"Unknown init method {self.init_method}, using random")
                scale = 0.01
                U = torch.randn(rows, effective_rank) * scale
                S = torch.ones(effective_rank) * scale
                V = torch.randn(cols, effective_rank) * scale

            self.adapters[layer_name] = {
                "U": U.requires_grad_(False),  # Left singular vectors
                "S": S.requires_grad_(False),  # Singular values
                "V": V.requires_grad_(False),  # Right singular vectors
            }

            logger.debug(
                f"Initialized {layer_name} adapter: rank={effective_rank}, "
                f"shape=({rows}x{cols}), init={self.init_method}"
            )

    def _pca_init(
        self, rows: int, cols: int, rank: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PCA-based initialization (simplified version)."""
        # Generate correlated random data to simulate PCA-like structure
        base_dim = min(rank * 2, min(rows, cols))

        # Create low-rank structure
        W_base = torch.randn(rows, base_dim) @ torch.randn(base_dim, cols) * 0.01

        # SVD to get actual PCA-like decomposition
        U, S, Vt = torch.svd(W_base)

        return U[:, :rank], S[:rank] * 0.1, Vt[:rank, :].T

    def _fisher_init(
        self, rows: int, cols: int, rank: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fisher information-based initialization."""
        # Simulate Fisher-informed initialization with higher values for "important" directions
        U = torch.randn(rows, rank) * 0.01

        # Fisher-like weighting: exponential decay for singular values
        S = torch.exp(-torch.arange(rank, dtype=torch.float) * 0.5) * 0.05

        V = torch.randn(cols, rank) * 0.01

        return U, S, V

    def get_delta_weight(
        self, layer_name: str, activation_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Compute the delta weight for a layer: activation_weight * U @ diag(S) @ V^T

        Args:
            layer_name: Target layer name
            activation_weight: Scaling factor from dispatch decision

        Returns:
            Delta weight tensor to add to original layer
        """
        if layer_name not in self.adapters:
            return torch.zeros(self.model_shapes[layer_name])

        adapter = self.adapters[layer_name]
        U, S, V = adapter["U"], adapter["S"], adapter["V"]

        # Compute low-rank update: activation_weight * U @ diag(S) @ V^T
        delta_W = activation_weight * (U * S.unsqueeze(0)) @ V.T

        return delta_W


class T2Mixer:
    """
    Transformer² Mixer - Main component for expert dispatch and mixing.

    Implements two-pass dispatch:
    1. Dispatch: Compute expert weights based on input features
    2. Mix: Apply low-rank edits via context manager for inference
    """

    def __init__(
        self,
        dispatch_spec: dict[str, Any],
        expert_lib: dict[str, dict[str, Any]],
        model_shapes: dict[str, tuple[int, ...]] | None = None,
    ):
        self.dispatch_spec = dispatch_spec
        self.expert_lib = expert_lib
        self.model_shapes = model_shapes or {}

        # Parse dispatch configuration
        self.features = dispatch_spec.get("features", ["prompt_stats"])
        self.mix_fn = dispatch_spec.get("mix_fn", "softmax")
        self.granularity = dispatch_spec.get("granularity", "sequence")

        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.experts: dict[str, ExpertAdapter] = {}

        # Initialize experts
        self._initialize_experts()

        # Dispatch network (simple linear for MVP)
        self._initialize_dispatch_network()

    def _initialize_experts(self):
        """Initialize expert adapters from expert library."""
        for expert_name, expert_spec in self.expert_lib.items():
            target_layers = expert_spec.get("layers", [])

            # Map layer specifications to actual layer names
            resolved_layers = self._resolve_layer_names(target_layers)

            self.experts[expert_name] = ExpertAdapter(
                expert_spec=expert_spec,
                target_layers=resolved_layers,
                model_shapes=self.model_shapes,
            )

            logger.debug(
                f"Initialized expert '{expert_name}' for layers {resolved_layers}"
            )

    def _resolve_layer_names(self, layer_specs: list[str]) -> list[str]:
        """
        Resolve layer specifications to actual parameter names.

        Args:
            layer_specs: List of layer specifications like ['attn_qkv', 'mlp', 'block_12']

        Returns:
            List of actual parameter names in the model
        """
        resolved = []

        for spec in layer_specs:
            if spec == "attn_qkv":
                # Target attention QKV projections
                resolved.extend(
                    [
                        "transformer.h.0.attn.c_attn.weight",  # Example for GPT-2
                        "transformer.h.1.attn.c_attn.weight",
                        "transformer.h.2.attn.c_attn.weight",
                    ]
                )
            elif spec == "mlp":
                # Target MLP layers
                resolved.extend(
                    [
                        "transformer.h.0.mlp.c_fc.weight",
                        "transformer.h.1.mlp.c_fc.weight",
                        "transformer.h.2.mlp.c_fc.weight",
                    ]
                )
            elif spec.startswith("block_"):
                # Specific block targeting
                block_num = spec.split("_")[1]
                resolved.extend(
                    [
                        f"transformer.h.{block_num}.attn.c_attn.weight",
                        f"transformer.h.{block_num}.mlp.c_fc.weight",
                    ]
                )
            else:
                # Direct parameter name
                resolved.append(spec)

        # Filter to only include layers that exist in model
        if self.model_shapes:
            resolved = [layer for layer in resolved if layer in self.model_shapes]

        return list(set(resolved))  # Remove duplicates

    def _initialize_dispatch_network(self):
        """Initialize dispatch network for computing expert weights."""
        # For MVP: simple linear dispatch based on feature vectors
        # In practice, this could be a small neural network

        # Compute feature dimension
        feature_dim = self._compute_feature_dimension()
        num_experts = len(self.experts)

        if num_experts == 0:
            logger.warning("No experts initialized")
            return

        # Simple linear dispatch (learnable parameters would be optimized)
        self.dispatch_weights = torch.randn(feature_dim, num_experts) * 0.01
        self.dispatch_bias = torch.zeros(num_experts)

        logger.debug(f"Initialized dispatch network: {feature_dim} -> {num_experts}")

    def _compute_feature_dimension(self) -> int:
        """Compute total feature dimension based on feature types."""
        dim = 0

        if "prompt_stats" in self.features:
            dim += 13  # Number of prompt stat features

        if "logits_entropy" in self.features:
            dim += 4  # Number of entropy features

        if "activation_sketch" in self.features:
            dim += 16  # Estimated activation sketch features (4 stats × 4 layers)

        return max(dim, 1)  # Minimum 1 dimension

    def dispatch(
        self,
        prompt_stats: dict[str, Any],
        activation_sketch: dict[str, Any],
        logits_entropy: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Compute expert mixing weights based on input features.

        Args:
            prompt_stats: Static prompt analysis features
            activation_sketch: Dynamic activation features
            logits_entropy: Optional entropy features

        Returns:
            Dictionary mapping expert names to mixing weights
        """
        if not self.experts:
            return {}

        # Extract and combine features
        feature_vector = self._combine_features(
            prompt_stats, activation_sketch, logits_entropy
        )

        # Compute raw logits
        with torch.no_grad():
            raw_logits = feature_vector @ self.dispatch_weights + self.dispatch_bias

        # Apply mixing function
        if self.mix_fn == "softmax":
            weights = F.softmax(raw_logits, dim=0)
        elif self.mix_fn == "linear":
            weights = torch.clamp(raw_logits, 0, 1)
            weights = weights / (weights.sum() + 1e-8)
        elif self.mix_fn == "energy":
            weights = torch.exp(-torch.abs(raw_logits))
            weights = weights / (weights.sum() + 1e-8)
        else:
            logger.warning(f"Unknown mix function {self.mix_fn}, using softmax")
            weights = F.softmax(raw_logits, dim=0)

        # Convert to dictionary
        expert_names = list(self.experts.keys())
        weight_dict = {name: weights[i].item() for i, name in enumerate(expert_names)}

        # Apply sparsity (keep only top-k experts)
        budget = min(4, len(expert_names))  # Max 4 active experts
        if len(weight_dict) > budget:
            # Keep top-k experts, zero out others
            sorted_experts = sorted(
                weight_dict.items(), key=lambda x: x[1], reverse=True
            )
            active_experts = dict(sorted_experts[:budget])

            # Renormalize active weights
            total_weight = sum(active_experts.values())
            if total_weight > 0:
                active_experts = {
                    k: v / total_weight for k, v in active_experts.items()
                }

            weight_dict = active_experts

        return weight_dict

    def _combine_features(
        self,
        prompt_stats: dict[str, Any],
        activation_sketch: dict[str, Any],
        logits_entropy: dict[str, float] | None,
    ) -> torch.Tensor:
        """Combine different feature types into a single vector."""
        features = []

        if "prompt_stats" in self.features:
            # Extract numerical features from prompt stats
            prompt_features = [
                prompt_stats.get("size", 0),
                prompt_stats.get("code_ratio", 0),
                prompt_stats.get("math_ratio", 0),
                prompt_stats.get("function_density", 0),
                prompt_stats.get("api_density", 0),
                prompt_stats.get("control_complexity", 0),
                prompt_stats.get("equation_density", 0),
                prompt_stats.get("number_density", 0),
                prompt_stats.get("question_ratio", 0),
                prompt_stats.get("imperative_ratio", 0),
                prompt_stats.get("complexity_score", 0),
                len(prompt_stats.get("api_keywords", []))
                / 10,  # Normalize keyword count
                prompt_stats.get("avg_word_length", 0) / 10,  # Normalize
            ]
            features.extend(prompt_features)

        if "logits_entropy" in self.features and logits_entropy:
            entropy_features = [
                logits_entropy.get("h0", 0),
                logits_entropy.get("h_mean", 0),
                logits_entropy.get("h_std", 0),
                logits_entropy.get("h_max", 0),
            ]
            features.extend(entropy_features)
        elif "logits_entropy" in self.features:
            features.extend([0] * 4)  # Pad with zeros if entropy not available

        if "activation_sketch" in self.features:
            # Extract activation statistics (simplified)
            sketch_features = []
            for i in range(4):  # Assume 4 representative layers
                layer_key = f"layer_{i}"
                if layer_key in activation_sketch:
                    layer_stats = activation_sketch[layer_key]
                    sketch_features.extend(
                        [
                            layer_stats.get("mean_norm", 0),
                            layer_stats.get("sparsity", 0),
                            layer_stats.get("std", 0),
                            layer_stats.get("kurtosis", 0),
                        ]
                    )
                else:
                    sketch_features.extend([0] * 4)
            features.extend(sketch_features)

        # Convert to tensor
        if not features:
            features = [0.0]  # Fallback

        return torch.tensor(features, dtype=torch.float32)

    @contextmanager
    def patch(self, model: nn.Module, expert_weights: dict[str, float]):
        """
        Context manager that applies expert patches to model for inference.

        Args:
            model: PyTorch model to patch
            expert_weights: Dictionary of expert weights from dispatch()

        Yields:
            Patched model ready for inference
        """
        if not expert_weights:
            yield model
            return

        # Store original parameters
        original_params = {}
        modified_params = {}

        try:
            # Compute and apply patches
            for expert_name, weight in expert_weights.items():
                if weight <= 1e-6:  # Skip negligible weights
                    continue

                expert = self.experts[expert_name]

                for layer_name in expert.target_layers:
                    if layer_name not in dict(model.named_parameters()):
                        continue

                    # Get original parameter
                    param = dict(model.named_parameters())[layer_name]

                    # Store original if not already stored
                    if layer_name not in original_params:
                        original_params[layer_name] = param.data.clone()
                        modified_params[layer_name] = param.data.clone()

                    # Compute and apply delta
                    delta = expert.get_delta_weight(layer_name, weight).to(param.device)

                    if delta.shape == param.shape:
                        modified_params[layer_name] += delta
                    else:
                        logger.warning(
                            f"Shape mismatch for {layer_name}: "
                            f"param {param.shape} vs delta {delta.shape}"
                        )

            # Apply all modifications
            for layer_name, new_data in modified_params.items():
                param = dict(model.named_parameters())[layer_name]
                param.data.copy_(new_data)

            logger.debug(
                f"Applied {len(expert_weights)} expert patches to {len(modified_params)} layers"
            )

            yield model

        finally:
            # Restore original parameters
            for layer_name, original_data in original_params.items():
                param = dict(model.named_parameters())[layer_name]
                param.data.copy_(original_data)

    def get_statistics(self) -> dict[str, Any]:
        """Get mixer statistics and configuration info."""
        return {
            "num_experts": len(self.experts),
            "dispatch_features": self.features,
            "mix_function": self.mix_fn,
            "granularity": self.granularity,
            "feature_dimension": self._compute_feature_dimension(),
            "expert_info": {
                name: {
                    "rank": expert.rank,
                    "layers": len(expert.target_layers),
                    "init_method": expert.init_method,
                }
                for name, expert in self.experts.items()
            },
        }

    def update_model_shapes(self, model: nn.Module):
        """Update model shapes from actual model parameters."""
        self.model_shapes = {
            name: param.shape
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # Reinitialize experts with correct shapes
        self._initialize_experts()

        logger.info(f"Updated model shapes: {len(self.model_shapes)} parameters")
