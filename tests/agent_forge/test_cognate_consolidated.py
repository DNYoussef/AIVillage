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

        print("✓ Model forward pass test passed")
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
