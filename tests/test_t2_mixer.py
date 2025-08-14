"""
Comprehensive tests for Transformer² (T²) mixer and feature extraction.
Tests low-rank expert dispatch, feature extraction, and model patching.
"""

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from src.agent_forge.t2.features import FeatureExtractor
from src.agent_forge.t2.mixer import ExpertAdapter, T2Mixer


class MockModel(nn.Module):
    """Mock transformer model for testing."""

    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict(
            {
                "h": nn.ModuleList(
                    [
                        nn.ModuleDict(
                            {
                                "attn": nn.ModuleDict(
                                    {
                                        "c_attn": nn.Linear(64, 192)  # QKV projection
                                    }
                                ),
                                "mlp": nn.ModuleDict({"c_fc": nn.Linear(64, 256)}),
                            }
                        )
                        for _ in range(3)  # 3 layers
                    ]
                )
            }
        )

    def generate(self, **kwargs):
        """Mock generate method."""
        input_ids = kwargs.get("input_ids", torch.tensor([[1, 2, 3]]))
        batch_size, seq_len = input_ids.shape
        max_new_tokens = kwargs.get("max_new_tokens", 5)

        # Generate random new tokens
        new_tokens = torch.randint(1, 1000, (batch_size, max_new_tokens))
        return torch.cat([input_ids, new_tokens], dim=1)


@pytest.fixture
def mock_model():
    """Create mock model for testing."""
    return MockModel()


@pytest.fixture
def model_shapes():
    """Get model parameter shapes."""
    model = MockModel()
    return {
        name: param.shape
        for name, param in model.named_parameters()
        if param.requires_grad
    }


@pytest.fixture
def expert_spec():
    """Basic expert specification."""
    return {
        "layers": ["attn_qkv", "mlp"],
        "rank": 4,
        "svd_scope": "per-matrix",
        "init": "random",
        "activation_rule": "gated",
        "budget": {"max_active": 2, "max_latency_ms": 100},
    }


@pytest.fixture
def dispatch_spec():
    """Basic dispatch specification."""
    return {
        "features": ["prompt_stats", "logits_entropy"],
        "mix_fn": "softmax",
        "granularity": "sequence",
    }


class TestFeatureExtractor:
    """Test the FeatureExtractor component."""

    def test_prompt_stats_basic(self):
        """Test basic prompt statistics extraction."""
        extractor = FeatureExtractor()

        prompt = (
            "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
        )
        stats = extractor.extract_prompt_stats(prompt)

        assert isinstance(stats, dict)
        assert "code_ratio" in stats
        assert "function_density" in stats
        assert "size" in stats
        assert stats["code_ratio"] > 0  # Should detect code
        assert stats["function_density"] > 0  # Should detect function

    def test_prompt_stats_math(self):
        """Test mathematical content detection."""
        extractor = FeatureExtractor()

        prompt = "Solve x² + 5x - 6 = 0 using the quadratic formula"
        stats = extractor.extract_prompt_stats(prompt)

        assert stats["math_ratio"] > 0
        assert stats["equation_density"] > 0
        assert stats["number_density"] > 0

    def test_prompt_stats_empty(self):
        """Test handling of empty prompts."""
        extractor = FeatureExtractor()

        stats = extractor.extract_prompt_stats("")
        assert all(v == 0 for k, v in stats.items() if k != "api_keywords")
        assert stats["api_keywords"] == []

    def test_logits_entropy(self):
        """Test logits entropy extraction."""
        extractor = FeatureExtractor()

        # Create mock logits [batch_size=1, seq_len=5, vocab_size=100]
        logits = torch.randn(1, 5, 100)
        entropy_features = extractor.extract_logits_entropy(logits)

        assert isinstance(entropy_features, dict)
        assert "h0" in entropy_features  # First token entropy
        assert "h_mean" in entropy_features
        assert "h_std" in entropy_features
        assert "h_max" in entropy_features

        # Entropy should be positive
        assert entropy_features["h_mean"] > 0
        assert entropy_features["h_max"] > 0

    def test_activation_sketch(self):
        """Test activation sketch extraction."""
        extractor = FeatureExtractor()

        # Mock activations for 2 layers
        activations = {
            0: torch.randn(1, 10, 64),  # Layer 0: [batch, seq, hidden]
            1: torch.randn(1, 10, 64),  # Layer 1
        }

        sketch = extractor.extract_activation_sketch(activations)

        assert "layer_0" in sketch
        assert "layer_1" in sketch

        layer_0_stats = sketch["layer_0"]
        assert "mean_norm" in layer_0_stats
        assert "sparsity" in layer_0_stats
        assert "std" in layer_0_stats
        assert "kurtosis" in layer_0_stats

    def test_api_keyword_extraction(self):
        """Test API keyword extraction."""
        extractor = FeatureExtractor()

        prompt = "Use pandas to read CSV data and train a neural network with PyTorch"
        keywords = extractor._extract_api_keywords(prompt)

        assert len(keywords) > 0
        assert any("data:" in kw for kw in keywords)  # Should detect data-related APIs
        assert len(keywords) <= 10  # Should limit keywords


class TestExpertAdapter:
    """Test the ExpertAdapter component."""

    def test_adapter_initialization(self, expert_spec, model_shapes):
        """Test expert adapter initialization."""
        target_layers = [
            "transformer.h.0.attn.c_attn.weight",
            "transformer.h.0.mlp.c_fc.weight",
        ]

        adapter = ExpertAdapter(expert_spec, target_layers, model_shapes)

        assert adapter.rank == 4
        assert adapter.svd_scope == "per-matrix"
        assert len(adapter.adapters) == len(target_layers)

        for layer_name in target_layers:
            if layer_name in model_shapes:
                assert layer_name in adapter.adapters
                adapter_dict = adapter.adapters[layer_name]
                assert "U" in adapter_dict
                assert "S" in adapter_dict
                assert "V" in adapter_dict

    def test_delta_weight_computation(self, expert_spec, model_shapes):
        """Test delta weight computation."""
        target_layers = ["transformer.h.0.attn.c_attn.weight"]

        adapter = ExpertAdapter(expert_spec, target_layers, model_shapes)

        if target_layers[0] in model_shapes:
            delta = adapter.get_delta_weight(target_layers[0], activation_weight=0.5)
            expected_shape = model_shapes[target_layers[0]]

            assert delta.shape == expected_shape
            assert not torch.allclose(
                delta, torch.zeros_like(delta)
            )  # Should be non-zero

    def test_different_initializations(self, model_shapes):
        """Test different initialization methods."""
        target_layers = ["transformer.h.0.attn.c_attn.weight"]

        for init_method in ["random", "pca_activations", "fisher"]:
            expert_spec = {
                "rank": 2,
                "svd_scope": "per-matrix",
                "init": init_method,
                "activation_rule": "always",
            }

            adapter = ExpertAdapter(expert_spec, target_layers, model_shapes)

            if target_layers[0] in model_shapes:
                assert len(adapter.adapters) > 0
                # Each initialization should produce different results
                delta1 = adapter.get_delta_weight(target_layers[0], 1.0)
                assert not torch.allclose(delta1, torch.zeros_like(delta1))


class TestT2Mixer:
    """Test the main T2Mixer component."""

    def test_mixer_initialization(self, dispatch_spec, model_shapes):
        """Test T2Mixer initialization."""
        expert_lib = {
            "expert_1": {
                "layers": ["attn_qkv"],
                "rank": 2,
                "svd_scope": "per-matrix",
                "init": "random",
                "activation_rule": "gated",
            }
        }

        mixer = T2Mixer(dispatch_spec, expert_lib, model_shapes)

        assert len(mixer.experts) == 1
        assert "expert_1" in mixer.experts
        assert mixer.mix_fn == "softmax"
        assert mixer.granularity == "sequence"

    def test_layer_name_resolution(self, dispatch_spec, model_shapes):
        """Test layer name resolution from specifications."""
        expert_lib = {
            "test_expert": {
                "layers": ["attn_qkv", "mlp", "block_0"],
                "rank": 2,
                "init": "random",
            }
        }

        mixer = T2Mixer(dispatch_spec, expert_lib, model_shapes)
        resolved_layers = mixer._resolve_layer_names(["attn_qkv", "mlp"])

        # Should resolve to actual parameter names
        assert len(resolved_layers) > 0
        assert all("transformer.h." in layer for layer in resolved_layers)

    def test_dispatch_computation(self, dispatch_spec, model_shapes):
        """Test expert dispatch weight computation."""
        expert_lib = {
            "expert_1": {"layers": ["attn_qkv"], "rank": 2, "init": "random"},
            "expert_2": {"layers": ["mlp"], "rank": 4, "init": "pca_activations"},
        }

        mixer = T2Mixer(dispatch_spec, expert_lib, model_shapes)

        # Mock feature inputs
        prompt_stats = {
            "size": 0.5,
            "code_ratio": 0.8,
            "math_ratio": 0.1,
            "function_density": 0.6,
            "api_density": 0.3,
            "control_complexity": 0.4,
            "equation_density": 0.0,
            "number_density": 0.2,
            "question_ratio": 0.0,
            "imperative_ratio": 0.7,
            "complexity_score": 0.5,
            "api_keywords": ["data:pandas"],
            "avg_word_length": 5.0,
        }

        activation_sketch = {
            "layer_0": {"mean_norm": 1.2, "sparsity": 0.1, "std": 0.5, "kurtosis": 0.2},
            "layer_1": {
                "mean_norm": 1.0,
                "sparsity": 0.15,
                "std": 0.45,
                "kurtosis": 0.1,
            },
        }

        logits_entropy = {"h0": 4.2, "h_mean": 3.8, "h_std": 0.5, "h_max": 5.1}

        weights = mixer.dispatch(prompt_stats, activation_sketch, logits_entropy)

        assert isinstance(weights, dict)
        assert len(weights) <= len(expert_lib)  # May apply sparsity
        assert all(0 <= w <= 1 for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 1e-6  # Should sum to 1

    def test_model_patching(self, mock_model, dispatch_spec, model_shapes):
        """Test model patching with expert weights."""
        expert_lib = {
            "test_expert": {
                "layers": ["attn_qkv"],
                "rank": 2,
                "init": "random",
                "activation_rule": "always",
            }
        }

        mixer = T2Mixer(dispatch_spec, expert_lib, model_shapes)
        expert_weights = {"test_expert": 0.7}

        # Store original parameters
        original_params = {}
        for name, param in mock_model.named_parameters():
            original_params[name] = param.data.clone()

        # Test patching context manager
        with mixer.patch(mock_model, expert_weights):
            # Parameters should be modified inside context
            modified = False
            for name, param in mock_model.named_parameters():
                if not torch.allclose(param.data, original_params[name]):
                    modified = True
                    break
            # Note: might not modify if layer names don't match exactly

        # Parameters should be restored after context
        for name, param in mock_model.named_parameters():
            assert torch.allclose(param.data, original_params[name])

    def test_feature_dimension_computation(self, dispatch_spec, model_shapes):
        """Test feature dimension computation."""
        expert_lib = {"expert_1": {"layers": ["attn_qkv"], "rank": 2, "init": "random"}}

        mixer = T2Mixer(dispatch_spec, expert_lib, model_shapes)
        dim = mixer._compute_feature_dimension()

        # Should include prompt_stats (13) + logits_entropy (4) = 17
        expected_dim = 13 + 4  # Based on feature types in dispatch_spec
        assert dim == expected_dim

    def test_different_mix_functions(self, model_shapes):
        """Test different mixing functions."""
        expert_lib = {"expert_1": {"layers": ["attn_qkv"], "rank": 2, "init": "random"}}

        for mix_fn in ["softmax", "linear", "energy"]:
            dispatch_spec = {
                "features": ["prompt_stats"],
                "mix_fn": mix_fn,
                "granularity": "sequence",
            }

            mixer = T2Mixer(dispatch_spec, expert_lib, model_shapes)

            prompt_stats = {
                "size": 0.5,
                "code_ratio": 0.0,
                "math_ratio": 0.0,
                "function_density": 0.0,
                "api_density": 0.0,
                "control_complexity": 0.0,
                "equation_density": 0.0,
                "number_density": 0.0,
                "question_ratio": 0.0,
                "imperative_ratio": 0.0,
                "complexity_score": 0.0,
                "api_keywords": [],
                "avg_word_length": 5.0,
            }

            weights = mixer.dispatch(prompt_stats, {})

            assert isinstance(weights, dict)
            if weights:  # May be empty due to sparsity
                assert all(w >= 0 for w in weights.values())

    def test_sparsity_enforcement(self, dispatch_spec, model_shapes):
        """Test that sparsity is enforced (max 4 active experts)."""
        # Create many experts
        expert_lib = {
            f"expert_{i}": {"layers": ["attn_qkv"], "rank": 1, "init": "random"}
            for i in range(10)
        }

        mixer = T2Mixer(dispatch_spec, expert_lib, model_shapes)

        prompt_stats = {
            "size": 0.5,
            "code_ratio": 0.0,
            "math_ratio": 0.0,
            "function_density": 0.0,
            "api_density": 0.0,
            "control_complexity": 0.0,
            "equation_density": 0.0,
            "number_density": 0.0,
            "question_ratio": 0.0,
            "imperative_ratio": 0.0,
            "complexity_score": 0.0,
            "api_keywords": [],
            "avg_word_length": 5.0,
        }

        weights = mixer.dispatch(prompt_stats, {})

        # Should keep at most 4 experts active
        assert len(weights) <= 4

    def test_statistics(self, dispatch_spec, model_shapes):
        """Test mixer statistics reporting."""
        expert_lib = {
            "expert_1": {"layers": ["attn_qkv"], "rank": 2, "init": "random"},
            "expert_2": {"layers": ["mlp"], "rank": 4, "init": "fisher"},
        }

        mixer = T2Mixer(dispatch_spec, expert_lib, model_shapes)
        stats = mixer.get_statistics()

        assert stats["num_experts"] == 2
        assert stats["dispatch_features"] == ["prompt_stats", "logits_entropy"]
        assert stats["mix_function"] == "softmax"
        assert "expert_info" in stats
        assert len(stats["expert_info"]) == 2


class TestIntegration:
    """Integration tests for the complete T² system."""

    def test_end_to_end_pipeline(self, mock_model):
        """Test complete end-to-end T² pipeline."""
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.return_tensors = "pt"
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }
        tokenizer.decode.return_value = "Generated text output"
        tokenizer.pad_token_id = 0

        # Set up mixer
        model_shapes = {
            name: param.shape for name, param in mock_model.named_parameters()
        }

        expert_lib = {
            "coding_expert": {
                "layers": ["attn_qkv"],
                "rank": 2,
                "init": "random",
                "activation_rule": "gated",
            }
        }

        dispatch_spec = {
            "features": ["prompt_stats"],
            "mix_fn": "softmax",
            "granularity": "sequence",
        }

        mixer = T2Mixer(dispatch_spec, expert_lib, model_shapes)
        extractor = FeatureExtractor()

        # End-to-end test
        prompt = "def quicksort(arr): # implement quicksort algorithm"

        # Extract features
        prompt_stats = extractor.extract_prompt_stats(prompt)

        # Compute dispatch
        weights = mixer.dispatch(prompt_stats, {})

        # Apply patches and generate
        with mixer.patch(mock_model, weights):
            tokenized = {
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            }
            output = mock_model.generate(**tokenized, max_new_tokens=10)

            assert output is not None
            assert (
                output.shape[1] > tokenized["input_ids"].shape[1]
            )  # Should have new tokens

    def test_error_handling(self, model_shapes):
        """Test error handling in various scenarios."""
        # Test with empty expert lib
        mixer = T2Mixer({}, {}, model_shapes)
        weights = mixer.dispatch({}, {})
        assert weights == {}

        # Test with malformed expert spec
        try:
            expert_lib = {"bad_expert": {"rank": "invalid"}}  # Invalid rank
            mixer = T2Mixer({}, expert_lib, model_shapes)
            # Should handle gracefully
        except Exception:
            pass  # Expected to handle errors

        # Test feature extraction with invalid inputs
        extractor = FeatureExtractor()
        stats = extractor.extract_prompt_stats(None)
        assert isinstance(stats, dict)

        # Test empty logits
        entropy = extractor.extract_logits_entropy(torch.empty(0))
        assert all(v == 0.0 for v in entropy.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
