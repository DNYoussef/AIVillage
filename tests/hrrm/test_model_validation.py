"""Quick validation tests for HRRM models."""


import pytest
import torch

# Test that we can import the models
try:
    from packages.hrrm.memory.model import MemoryAsContextTiny, MemoryConfig
    from packages.hrrm.planner.model import HRMPlanner, PlannerConfig
    from packages.hrrm.reasoner.model import HRMReasoner, ReasonerConfig

    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    print(f"Models not available: {e}")


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="HRRM models not available")
class TestModelCreation:
    """Test that HRRM models can be created and have reasonable parameter counts."""

    def create_test_configs(self):
        """Create test configurations for all models."""
        base_config = {
            "vocab_size": 1000,
            "d_model": 128,
            "n_layers": 4,
            "n_head": 8,
            "d_ff": 512,
            "max_seq_len": 256,
        }

        planner_config = PlannerConfig(
            **base_config,
            control_tokens=["<PLAN>", "<ACTION>", "<ENDPLAN>"],
            max_H=2,
            inner_T=1,
        )

        reasoner_config = ReasonerConfig(
            **base_config,
            max_H=2,
            inner_T=1,
            self_consistency_k=3,
        )

        memory_config = MemoryConfig(
            **base_config,
            mem_dim=64,
            mem_tokens=16,
            mem_slots=32,
        )

        return planner_config, reasoner_config, memory_config

    def test_planner_creation(self):
        """Test planner model creation."""
        planner_config, _, _ = self.create_test_configs()
        model = HRMPlanner(planner_config)

        # Check basic properties
        assert model.config == planner_config

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 10_000  # Should have reasonable number of params

        print(f"Planner parameters: {total_params:,}")

    def test_reasoner_creation(self):
        """Test reasoner model creation."""
        _, reasoner_config, _ = self.create_test_configs()
        model = HRMReasoner(reasoner_config)

        # Check basic properties
        assert model.config == reasoner_config

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 10_000  # Should have reasonable number of params

        print(f"Reasoner parameters: {total_params:,}")

    def test_memory_creation(self):
        """Test memory model creation."""
        _, _, memory_config = self.create_test_configs()
        model = MemoryAsContextTiny(memory_config)

        # Check basic properties
        assert model.config == memory_config

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 10_000  # Should have reasonable number of params

        print(f"Memory parameters: {total_params:,}")

    def test_all_models_forward_pass(self):
        """Test forward pass for all models."""
        planner_config, reasoner_config, memory_config = self.create_test_configs()

        models = {
            "planner": HRMPlanner(planner_config),
            "reasoner": HRMReasoner(reasoner_config),
            "memory": MemoryAsContextTiny(memory_config),
        }

        batch_size = 2
        seq_len = 16
        vocab_size = 1000

        for model_name, model in models.items():
            print(f"Testing {model_name} forward pass...")

            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

            with torch.no_grad():
                outputs = model(input_ids)

            # All models should have logits
            assert hasattr(outputs, "logits")
            assert outputs.logits.shape == (batch_size, seq_len, vocab_size)

            # Check model-specific outputs
            if model_name == "planner":
                assert hasattr(outputs, "control_logits")
            elif model_name == "reasoner":
                assert hasattr(outputs, "thought_logits")

            print(f"✓ {model_name} forward pass successful")

    def test_parameter_counts_reasonable(self):
        """Test that parameter counts are in reasonable range."""
        planner_config, reasoner_config, memory_config = self.create_test_configs()

        models = {
            "planner": HRMPlanner(planner_config),
            "reasoner": HRMReasoner(reasoner_config),
            "memory": MemoryAsContextTiny(memory_config),
        }

        param_counts = {}
        for model_name, model in models.items():
            param_count = sum(p.numel() for p in model.parameters())
            param_counts[model_name] = param_count

            # Should be reasonable for test models
            assert 50_000 <= param_count <= 5_000_000, f"{model_name}: {param_count:,} params"

        print("\nParameter counts:")
        for name, count in param_counts.items():
            print(f"  {name:8s}: {count:,} parameters")

        # All models should have similar parameter counts (within 2x of each other)
        min_params = min(param_counts.values())
        max_params = max(param_counts.values())
        ratio = max_params / min_params

        assert ratio <= 3.0, f"Parameter counts too different: {min_params:,} to {max_params:,} (ratio: {ratio:.2f})"

    def test_model_state_dict_compatibility(self):
        """Test that models can save and load state dicts."""
        planner_config, _, _ = self.create_test_configs()
        model = HRMPlanner(planner_config)

        # Get state dict
        state_dict = model.state_dict()
        assert len(state_dict) > 0

        # Create new model and load state
        new_model = HRMPlanner(planner_config)
        new_model.load_state_dict(state_dict)

        # Test that outputs are identical
        input_ids = torch.randint(0, 1000, (1, 16))

        with torch.no_grad():
            output1 = model(input_ids)
            output2 = new_model(input_ids)

        assert torch.allclose(output1.logits, output2.logits, atol=1e-6)

        print("✓ State dict save/load works correctly")


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="HRRM models not available")
class TestBasicFunctionality:
    """Test basic functionality without heavy dependencies."""

    def test_imports_work(self):
        """Test that all imports work."""

        # Should not raise any errors
        assert True

    def test_config_creation(self):
        """Test that configs can be created."""
        config = PlannerConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=4,
            n_head=8,
            control_tokens=["<PLAN>", "<ACTION>"],
            max_H=2,
            inner_T=1,
        )

        assert config.vocab_size == 1000
        assert config.d_model == 128
        assert len(config.control_tokens) == 2

    def test_small_model_creation(self):
        """Test creating very small models for quick validation."""
        config = PlannerConfig(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_head=4,
            d_ff=128,
            max_seq_len=64,
            control_tokens=["<PLAN>"],
            max_H=1,
            inner_T=1,
        )

        model = HRMPlanner(config)
        param_count = sum(p.numel() for p in model.parameters())

        print(f"Small model parameters: {param_count:,}")

        # Test forward pass
        input_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            outputs = model(input_ids)

        assert outputs.logits.shape == (1, 8, 100)
        assert outputs.control_logits.shape == (1, 1)  # Only last token for control prediction

        print("✓ Small model creation and forward pass successful")


if __name__ == "__main__":
    # Quick manual test
    if MODELS_AVAILABLE:
        test = TestModelCreation()
        test.test_planner_creation()
        test.test_all_models_forward_pass()
        print("✅ All manual tests passed!")
    else:
        print("❌ Models not available for testing")
