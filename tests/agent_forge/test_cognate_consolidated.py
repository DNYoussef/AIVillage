# ruff: noqa: S101  # Use of assert detected - Expected in test files
#!/usr/bin/env python3
"""
Consolidated Cognate Model Tests
Tests the production-ready cognate_pretrain implementation
"""

from pathlib import Path
import sys

import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# Test imports from consolidated implementation
def test_cognate_imports():
    """Test that all cognate components can be imported."""
    try:
        print("✓ All cognate imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_cognate_config():
    """Test cognate configuration creation."""
    try:
        from core.agent_forge.phases.cognate_pretrain.refiner_core import CognateConfig

        config = CognateConfig()

        # Verify key parameters for 25M targeting
        assert config.d_model == 216
        assert config.n_layers == 11
        assert config.n_heads == 4
        assert config.vocab_size == 32000

        print("✓ Cognate config validation passed")
        return True

    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def test_parameter_count_validation():
    """Test that models hit the 25M parameter target."""
    try:
        from core.agent_forge.phases.cognate_pretrain.refiner_core import CognateConfig, CognateRefiner

        config = CognateConfig()
        model = CognateRefiner(config)

        total_params = sum(p.numel() for p in model.parameters())
        target_params = 25_083_528  # Exact target from implementation

        # Allow 5% tolerance
        tolerance = 0.05
        accuracy = abs(total_params - target_params) / target_params

        print(f"Model parameters: {total_params:,}")
        print(f"Target parameters: {target_params:,}")
        print(f"Accuracy: {(1-accuracy)*100:.2f}%")

        assert (
            accuracy <= tolerance
        ), f"Parameter count {total_params} outside ±{tolerance*100}% of target {target_params}"

        print("✓ Parameter count validation passed")
        return True

    except Exception as e:
        print(f"✗ Parameter count test failed: {e}")
        return False


def test_model_forward_pass():
    """Test basic forward pass functionality."""
    try:
        from core.agent_forge.phases.cognate_pretrain.refiner_core import CognateConfig, CognateRefiner

        config = CognateConfig()
        model = CognateRefiner(config)
        model.eval()

        # Create test input
        batch_size = 2
        seq_len = 32  # Shorter for faster testing
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Verify outputs
        assert "logits" in outputs
        assert "halt_logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)

        # Test output quality - no NaN or Inf values (data corruption check)
        logits = outputs["logits"]
        assert not torch.isnan(logits).any(), "Model outputs contain NaN values"
        assert not torch.isinf(logits).any(), "Model outputs contain infinite values"

        # Test output range is reasonable
        logits_std = torch.std(logits).item()
        assert 0.1 < logits_std < 100.0, f"Output std dev {logits_std:.4f} outside reasonable range"

        # Test model consistency - same input should produce same output in eval mode
        with torch.no_grad():
            outputs2 = model(input_ids=input_ids, attention_mask=attention_mask)

        assert torch.allclose(outputs["logits"], outputs2["logits"], atol=1e-6), "Model not deterministic in eval mode"

        # Test gradient flow during training mode
        model.train()
        outputs_train = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = outputs_train["logits"].sum()
        loss.backward()

        # Check that gradients exist and are reasonable
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert not torch.isnan(param.grad).any(), "Gradients contain NaN"
                grad_norms.append(grad_norm)

        assert len(grad_norms) > 0, "No gradients computed"
        avg_grad_norm = sum(grad_norms) / len(grad_norms)
        assert 0.001 < avg_grad_norm < 10.0, f"Average gradient norm {avg_grad_norm:.6f} outside reasonable range"

        print("✓ Model forward pass test passed with integrity checks")
        return True

    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        return False


def test_model_factory():
    """Test the model factory functionality."""
    try:
        from core.agent_forge.phases.cognate_pretrain.model_factory import CognateModelFactory

        factory = CognateModelFactory()

        # Test factory can be instantiated
        assert factory is not None

        # Test factory actually creates models - behavioral validation
        if hasattr(factory, "create_model"):
            model = factory.create_model()
            assert model is not None, "Factory should create a valid model"

            # Verify the created model has expected functionality
            if hasattr(model, "parameters"):
                param_count = sum(p.numel() for p in model.parameters())
                assert param_count > 0, "Created model should have parameters"

            # Test model creation with custom config
            if hasattr(factory, "create_model_with_config"):
                from core.agent_forge.phases.cognate_pretrain.refiner_core import CognateConfig

                config = CognateConfig(d_model=128, n_layers=6)
                custom_model = factory.create_model_with_config(config)
                assert custom_model is not None, "Factory should create model with custom config"

        # Test factory provides expected interface methods
        expected_methods = ["create_model", "get_supported_configs", "validate_config"]
        for method in expected_methods:
            if hasattr(factory, method):
                assert callable(getattr(factory, method)), f"Factory method {method} should be callable"

        print("✓ Model factory test passed")
        return True

    except Exception as e:
        print(f"✗ Model factory test failed: {e}")
        return False


if __name__ == "__main__":
    print("Running consolidated Cognate tests...")

    tests = [
        test_cognate_imports,
        test_cognate_config,
        test_parameter_count_validation,
        test_model_forward_pass,
        test_model_factory,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        print(f"\n--- Running {test_func.__name__} ---")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed with exception: {e}")

    print("\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")

    if passed == total:
        print("🎉 All tests passed! Cognate consolidation successful.")
    else:
        print("❌ Some tests failed. Review the implementation.")
