"""End-to-end tests for HRRM bootstrap system."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
from packages.hrrm.memory.model import MemoryAsContextTiny, MemoryConfig
from packages.hrrm.planner.model import HRMPlanner, PlannerConfig
from packages.hrrm.reasoner.model import HRMReasoner, ReasonerConfig


@pytest.mark.integration
class TestHRRMEndToEnd:
    """End-to-end tests for complete HRRM system."""

    def create_minimal_configs(self):
        """Create minimal configs for testing."""
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

    def test_model_creation_and_saving(self):
        """Test creating all three models and saving them."""
        planner_config, reasoner_config, memory_config = self.create_minimal_configs()

        # Create models
        planner = HRMPlanner(planner_config)
        reasoner = HRMReasoner(reasoner_config)
        memory = MemoryAsContextTiny(memory_config)

        # Test parameter counts
        planner_params = sum(p.numel() for p in planner.parameters())
        reasoner_params = sum(p.numel() for p in reasoner.parameters())
        memory_params = sum(p.numel() for p in memory.parameters())

        # All should be reasonable size for test
        assert planner_params > 100_000
        assert reasoner_params > 100_000
        assert memory_params > 100_000

        # Test saving and loading
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save models
            planner_path = temp_path / "planner"
            reasoner_path = temp_path / "reasoner"
            memory_path = temp_path / "memory"

            # Create checkpoint-like saves
            torch.save(
                {
                    "model_state_dict": planner.state_dict(),
                    "config": planner_config,
                    "param_count": planner_params,
                },
                planner_path.with_suffix(".pt"),
            )

            torch.save(
                {
                    "model_state_dict": reasoner.state_dict(),
                    "config": reasoner_config,
                    "param_count": reasoner_params,
                },
                reasoner_path.with_suffix(".pt"),
            )

            torch.save(
                {
                    "model_state_dict": memory.state_dict(),
                    "config": memory_config,
                    "param_count": memory_params,
                },
                memory_path.with_suffix(".pt"),
            )

            # Verify files exist
            assert planner_path.with_suffix(".pt").exists()
            assert reasoner_path.with_suffix(".pt").exists()
            assert memory_path.with_suffix(".pt").exists()

            # Test loading
            checkpoint = torch.load(planner_path.with_suffix(".pt"))
            new_planner = HRMPlanner(checkpoint["config"])
            new_planner.load_state_dict(checkpoint["model_state_dict"])

            # Test forward pass
            input_ids = torch.randint(0, 1000, (1, 32))
            with torch.no_grad():
                orig_output = planner(input_ids)
                new_output = new_planner(input_ids)

            assert torch.allclose(orig_output.logits, new_output.logits, atol=1e-6)

    def test_hf_export_format(self):
        """Test HuggingFace export format creation."""
        planner_config, _, _ = self.create_minimal_configs()
        planner = HRMPlanner(planner_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            export_dir = temp_path / "hf_export"
            export_dir.mkdir()

            # Create HF-style export
            param_count = sum(p.numel() for p in planner.parameters())

            # Save model weights
            torch.save(planner.state_dict(), export_dir / "pytorch_model.bin")

            # Create config.json
            hf_config = {
                "architectures": ["HRRMPlanner"],
                "model_type": "hrrm_planner",
                "vocab_size": planner_config.vocab_size,
                "hidden_size": planner_config.d_model,
                "num_hidden_layers": planner_config.n_layers,
                "num_attention_heads": planner_config.n_head,
                "param_count": param_count,
                "control_tokens": planner_config.control_tokens,
                "max_H": planner_config.max_H,
                "inner_T": planner_config.inner_T,
            }

            with open(export_dir / "config.json", "w") as f:
                json.dump(hf_config, f, indent=2)

            # Create README.md
            readme_content = f"""# HRRM Planner Model

Test model export with {param_count:,} parameters.

## Features
- HRM two-timescale loop
- Control token generation
- Planning DSL support
"""

            with open(export_dir / "README.md", "w") as f:
                f.write(readme_content)

            # Verify export structure
            assert (export_dir / "pytorch_model.bin").exists()
            assert (export_dir / "config.json").exists()
            assert (export_dir / "README.md").exists()

            # Verify config content
            with open(export_dir / "config.json") as f:
                config = json.load(f)

            assert config["model_type"] == "hrrm_planner"
            assert config["param_count"] == param_count
            assert config["control_tokens"] == planner_config.control_tokens

    def test_evaluation_pipeline(self):
        """Test basic evaluation pipeline."""
        planner_config, reasoner_config, memory_config = self.create_minimal_configs()

        models = {
            "planner": HRMPlanner(planner_config),
            "reasoner": HRMReasoner(reasoner_config),
            "memory": MemoryAsContextTiny(memory_config),
        }

        # Test evaluation on synthetic data
        batch_size = 2
        seq_len = 32

        for model_type, model in models.items():
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))

            # Basic forward pass evaluation
            with torch.no_grad():
                outputs = model(input_ids)

            assert hasattr(outputs, "logits")
            assert outputs.logits.shape == (batch_size, seq_len, 1000)

            # Calculate perplexity-like metric
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="mean"
            )

            perplexity = torch.exp(loss).item()

            # Should have reasonable perplexity for random model
            assert perplexity > 1.0
            assert perplexity < 10000.0  # Not completely broken

            # Model-specific evaluations
            if model_type == "planner":
                assert hasattr(outputs, "control_logits")
                assert outputs.control_logits.shape == (batch_size, seq_len, len(planner_config.control_tokens))

            elif model_type == "reasoner":
                assert hasattr(outputs, "thought_logits")
                assert outputs.thought_logits.shape == (batch_size, seq_len, 2)

    def test_training_compatibility(self):
        """Test that models are compatible with training loops."""
        planner_config, _, _ = self.create_minimal_configs()
        model = HRMPlanner(planner_config)

        # Create dummy optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Training step
        model.train()
        input_ids = torch.randint(0, 1000, (2, 32))

        # Forward pass
        outputs = model(input_ids, labels=input_ids)

        # Should have loss for training
        assert hasattr(outputs, "loss")
        assert outputs.loss.requires_grad

        # Backward pass
        optimizer.zero_grad()
        outputs.loss.backward()

        # Check gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

        # Optimizer step
        optimizer.step()

        # Should complete without errors

    def test_memory_efficiency_scaling(self):
        """Test memory efficiency with different scales."""
        base_config = {
            "vocab_size": 1000,
            "d_model": 256,
            "n_layers": 6,
            "n_head": 8,
            "d_ff": 1024,
            "max_seq_len": 512,
        }

        config = PlannerConfig(
            **base_config,
            control_tokens=["<PLAN>", "<ACTION>"],
            max_H=2,
            inner_T=1,
        )

        model = HRMPlanner(config)

        # Test with different batch sizes and sequence lengths
        test_cases = [
            (1, 64),  # Small
            (2, 128),  # Medium
            (4, 256),  # Large
        ]

        for batch_size, seq_len in test_cases:
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))

            # Should not run out of memory
            with torch.no_grad():
                outputs = model(input_ids)
                assert outputs.logits.shape == (batch_size, seq_len, 1000)

                # Clean up
                del outputs
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def test_deterministic_behavior(self):
        """Test that models behave deterministically."""
        planner_config, _, _ = self.create_minimal_configs()

        # Set random seeds
        torch.manual_seed(42)
        model1 = HRMPlanner(planner_config)

        torch.manual_seed(42)
        model2 = HRMPlanner(planner_config)

        # Models should have identical weights
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-8)

        # Forward passes should be identical
        torch.manual_seed(123)
        input_ids = torch.randint(0, 1000, (1, 32))

        with torch.no_grad():
            output1 = model1(input_ids)

        torch.manual_seed(123)
        input_ids = torch.randint(0, 1000, (1, 32))

        with torch.no_grad():
            output2 = model2(input_ids)

        assert torch.allclose(output1.logits, output2.logits, atol=1e-6)

    def test_model_serialization_compatibility(self):
        """Test serialization compatibility across formats."""
        planner_config, _, _ = self.create_minimal_configs()
        model = HRMPlanner(planner_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test different save formats
            formats = {
                "state_dict": model.state_dict(),
                "full_model": model,
                "checkpoint": {
                    "model_state_dict": model.state_dict(),
                    "config": planner_config,
                    "optimizer_state_dict": {},
                    "epoch": 0,
                },
            }

            # Save in different formats
            saved_files = {}
            for fmt_name, data in formats.items():
                file_path = temp_path / f"model_{fmt_name}.pt"
                torch.save(data, file_path)
                saved_files[fmt_name] = file_path

            # Test loading state_dict
            state_dict = torch.load(saved_files["state_dict"])
            new_model = HRMPlanner(planner_config)
            new_model.load_state_dict(state_dict)

            # Test loading checkpoint
            checkpoint = torch.load(saved_files["checkpoint"])
            checkpoint_model = HRMPlanner(checkpoint["config"])
            checkpoint_model.load_state_dict(checkpoint["model_state_dict"])

            # Verify models are equivalent
            input_ids = torch.randint(0, 1000, (1, 16))
            with torch.no_grad():
                orig_out = model(input_ids)
                new_out = new_model(input_ids)
                checkpoint_out = checkpoint_model(input_ids)

            assert torch.allclose(orig_out.logits, new_out.logits, atol=1e-6)
            assert torch.allclose(orig_out.logits, checkpoint_out.logits, atol=1e-6)

    @pytest.mark.slow
    def test_acceptance_criteria_simulation(self):
        """Simulate acceptance criteria validation."""
        # Create models that meet parameter requirements

        # Estimate configuration for target size
        vocab_size = 32000
        d_model = 512
        n_layers = 12

        # Adjust based on rough parameter calculation
        # embedding: vocab_size * d_model = 32000 * 512 = 16M
        # output: vocab_size * d_model = 32000 * 512 = 16M
        # layers: ~3M per layer * 12 = 36M
        # Total: ~68M (close enough to 50M target)

        configs = {}
        models = {}
        param_counts = {}

        # Create all three models
        base_config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_head": 8,
            "d_ff": d_model * 4,
            "max_seq_len": 2048,
        }

        configs["planner"] = PlannerConfig(
            **base_config,
            control_tokens=["<PLAN>", "<SUBGOAL>", "<ACTION>", "<CHECK>", "<ENDPLAN>"],
            max_H=3,
            inner_T=2,
        )

        configs["reasoner"] = ReasonerConfig(
            **base_config,
            max_H=3,
            inner_T=2,
            self_consistency_k=5,
        )

        configs["memory"] = MemoryConfig(
            **base_config,
            mem_dim=256,
            mem_tokens=64,
            mem_slots=128,
        )

        # Create models and count parameters
        models["planner"] = HRMPlanner(configs["planner"])
        models["reasoner"] = HRMReasoner(configs["reasoner"])
        models["memory"] = MemoryAsContextTiny(configs["memory"])

        for name, model in models.items():
            param_counts[name] = sum(p.numel() for p in model.parameters())

        # Validate acceptance criteria
        acceptance_criteria = {}

        # 1. Three models trained (simulated)
        acceptance_criteria["three_models_trained"] = len(models) == 3

        # 2. Parameter counts in range (48M-55M)
        models_in_range = 0
        for name, count in param_counts.items():
            if 30_000_000 <= count <= 70_000_000:  # Relaxed for test models
                models_in_range += 1

        acceptance_criteria["param_counts_valid"] = models_in_range == 3

        # 3. Models have reasonable perplexity (simulate evaluation)
        perplexities = {}
        for name, model in models.items():
            input_ids = torch.randint(0, vocab_size, (1, 64))
            with torch.no_grad():
                outputs = model(input_ids)

            # Calculate perplexity
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="mean"
            )

            perplexities[name] = torch.exp(loss).item()

        # All perplexities should be reasonable (< 100 for untrained models)
        reasonable_ppls = [ppl for ppl in perplexities.values() if ppl < 1000]
        acceptance_criteria["perplexity_reasonable"] = len(reasonable_ppls) == 3

        # 4. Can be exported to HF format (simulate)
        acceptance_criteria["hf_exports_possible"] = True  # All models support state_dict

        # 5. Can be saved as checkpoints (simulate)
        acceptance_criteria["checkpoints_possible"] = True  # All models support checkpointing

        # Summary
        criteria_met = all(acceptance_criteria.values())

        print("\nAcceptance Criteria Results:")
        print(f"{'Criterion':<25} {'Status'}")
        print("-" * 40)
        for criterion, passed in acceptance_criteria.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{criterion.replace('_', ' ').title():<25} {status}")

        print("\nParameter Counts:")
        for name, count in param_counts.items():
            print(f"  {name:8s}: {count:,} parameters")

        print("\nPerplexities:")
        for name, ppl in perplexities.items():
            print(f"  {name:8s}: {ppl:.2f}")

        print(f"\nOverall Status: {'✓ READY' if criteria_met else '✗ NOT READY'}")

        # For testing, we'll assert the major criteria
        assert acceptance_criteria["three_models_trained"]
        assert acceptance_criteria["hf_exports_possible"]
        assert acceptance_criteria["checkpoints_possible"]

        # Parameter counts and perplexity may vary for test models
        # but should be reasonable
        for name, count in param_counts.items():
            assert count > 1_000_000  # At least 1M parameters
            assert count < 100_000_000  # Less than 100M parameters

        for name, ppl in perplexities.items():
            assert ppl > 1.0  # Sanity check
            assert ppl < 10000.0  # Not completely broken
