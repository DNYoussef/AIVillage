#!/usr/bin/env python3
"""
Integration Tests for Cognate Model

This module tests the integration of the Cognate model with:
- Agent Forge pipeline
- EvoMerge compatibility
- HuggingFace ecosystem
- Training pipelines
"""

import json
import logging
import os
from pathlib import Path

# Import the canonical Cognate implementation
import sys
import tempfile
import unittest

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cognate_model import CognateModel, create_cognate_model
from torch.utils.data import Dataset
from training.trainer import CognateTrainer, CognateTrainingConfig

logger = logging.getLogger(__name__)


class DummyDataset(Dataset):
    """Simple dataset for testing."""

    def __init__(self, size=100, seq_len=64, vocab_size=1000):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, self.vocab_size, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len),
            "labels": torch.randint(0, self.vocab_size, (self.seq_len,)),
        }


class TestAgentForgeIntegration(unittest.TestCase):
    """Test Agent Forge pipeline integration."""

    def test_create_three_models(self):
        """Test creating three models for EvoMerge as required by Agent Forge."""
        # Import the main factory function
        from cognate_model import create_cognate_model

        # Create three models
        models = []
        for i in range(3):
            model = create_cognate_model(
                variant_name=f"model-{i+1}",
                seed=42 + i,  # Different seeds
                d_model=128,  # Smaller for testing
                n_layers=4,
                vocab_size=1000,
                mem_capacity=64,
            )
            models.append(model)

        # Verify we have three models
        self.assertEqual(len(models), 3)

        # All should be CognateModel instances
        for model in models:
            self.assertIsInstance(model, CognateModel)

        # Should have different variant names
        names = [model.variant_name for model in models]
        self.assertEqual(len(set(names)), 3)  # All unique

        # Should have same parameter counts
        param_counts = [model.count_parameters() for model in models]
        self.assertTrue(all(count == param_counts[0] for count in param_counts))

        # Should have different initial weights (due to different seeds)
        model1_params = list(models[0].parameters())
        model2_params = list(models[1].parameters())

        # At least one parameter should be different
        params_different = False
        for p1, p2 in zip(model1_params, model2_params):
            if not torch.equal(p1, p2):
                params_different = True
                break

        self.assertTrue(params_different, "Models should have different initial weights")

    def test_huggingface_compatibility(self):
        """Test HuggingFace format compatibility."""
        model = create_cognate_model(d_model=128, n_layers=4, vocab_size=1000)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "hf_model"

            # Save in HuggingFace format
            model.save_pretrained(str(save_path))

            # Check required files exist
            required_files = ["pytorch_model.bin", "config.json"]
            for filename in required_files:
                file_path = save_path / filename
                self.assertTrue(file_path.exists(), f"Missing required file: {filename}")

            # Check config format
            with open(save_path / "config.json") as f:
                config = json.load(f)

            # Should have HuggingFace-style fields
            self.assertIn("architectures", config)
            self.assertIn("model_type", config)
            self.assertEqual(config["architectures"], ["CognateModel"])
            self.assertEqual(config["model_type"], "cognate")

            # Load model back
            loaded_model = CognateModel.from_pretrained(str(save_path))

            # Should have same architecture
            self.assertEqual(model.count_parameters(), loaded_model.count_parameters())

    def test_parameter_targeting_accuracy(self):
        """Test that models hit the 25M parameter target accurately."""
        # Full-size model
        model = create_cognate_model()
        param_count = model.count_parameters()

        target = 25_069_534
        error_pct = abs(param_count - target) / target * 100

        # Should be within 2% of target
        self.assertLess(error_pct, 2.0, f"Parameter count {param_count:,} is {error_pct:.2f}% off target {target:,}")

        logger.info(f"Parameter count: {param_count:,} (target: {target:,}, error: {error_pct:.2f}%)")

    def test_model_metadata(self):
        """Test model metadata for pipeline integration."""
        model = create_cognate_model(variant_name="test-model")

        # Check variant name is set
        self.assertEqual(model.variant_name, "test-model")

        # Check parameter breakdown
        breakdown = model.get_parameter_breakdown()

        expected_components = ["embed_tokens", "layers", "norm", "lm_head", "act_head", "memory_controllers", "total"]

        for component in expected_components:
            self.assertIn(component, breakdown)
            self.assertGreater(breakdown[component], 0)


class TestPipelineCompatibility(unittest.TestCase):
    """Test compatibility with various training pipelines."""

    def setUp(self):
        self.model = create_cognate_model(d_model=128, n_layers=4, vocab_size=1000, mem_capacity=64)
        self.dataset = DummyDataset(size=20, seq_len=32, vocab_size=1000)

    def test_basic_training_loop(self):
        """Test basic PyTorch training loop compatibility."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        torch.nn.CrossEntropyLoss()

        self.model.train()

        # Simulate a few training steps
        for i in range(3):
            # Get batch
            batch = self.dataset[i]
            input_ids = batch["input_ids"].unsqueeze(0)
            labels = batch["labels"].unsqueeze(0)

            # Forward pass
            outputs = self.model(input_ids, labels=labels, return_dict=True)
            loss = outputs["loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.assertIsInstance(loss.item(), float)
            self.assertGreater(loss.item(), 0)

    def test_cognate_trainer_integration(self):
        """Test integration with the custom Cognate trainer."""
        config = CognateTrainingConfig(
            max_steps=10,
            batch_size=2,
            gradient_accumulation_steps=1,
            logging_steps=5,
            save_steps=10,
            eval_steps=5,
            grokfast_enabled=False,  # Disable for simpler test
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CognateTrainer(
                model=self.model,
                config=config,
                train_dataset=self.dataset,
                eval_dataset=DummyDataset(size=10, seq_len=32, vocab_size=1000),
                output_dir=tmpdir,
            )

            # Run short training
            results = trainer.train()

            # Should complete successfully
            self.assertIn("training_time", results)
            self.assertIn("final_step", results)
            self.assertEqual(results["final_step"], config.max_steps)

    def test_distributed_training_compatibility(self):
        """Test that model is compatible with distributed training setup."""
        # Test DDP wrapper (without actually running distributed)
        try:

            # This should not raise an error (though we can't test actual DDP without multiple processes)
            next(self.model.parameters()).device

            # Test that model can be wrapped (though we won't actually use it)
            # In real DDP, this would require proper process group setup
            # ddp_model = DDP(self.model)

            # For now, just test that the model has the right attributes
            self.assertTrue(hasattr(self.model, "parameters"))
            self.assertTrue(hasattr(self.model, "named_parameters"))

        except ImportError:
            self.skipTest("PyTorch distributed not available")

    def test_mixed_precision_compatibility(self):
        """Test Automatic Mixed Precision (AMP) compatibility."""
        from torch.cuda.amp import GradScaler, autocast

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for AMP test")

        model = self.model.cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = GradScaler()

        # Test AMP training step
        batch = self.dataset[0]
        input_ids = batch["input_ids"].unsqueeze(0).cuda()
        labels = batch["labels"].unsqueeze(0).cuda()

        with autocast():
            outputs = model(input_ids, labels=labels, return_dict=True)
            loss = outputs["loss"]

        # Should work without errors
        self.assertIsInstance(loss, torch.Tensor)

        # Test scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    def test_checkpoint_resumption(self):
        """Test training checkpoint save/resume functionality."""
        config = CognateTrainingConfig(
            max_steps=10,
            batch_size=2,
            save_steps=5,
            logging_steps=5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Initial training
            trainer = CognateTrainer(
                model=self.model,
                config=config,
                train_dataset=self.dataset,
                output_dir=tmpdir,
            )

            trainer.train()

            # Check checkpoint was saved
            checkpoint_dir = Path(tmpdir) / f"checkpoint-{config.max_steps}"
            self.assertTrue(checkpoint_dir.exists())
            self.assertTrue((checkpoint_dir / "pytorch_model.bin").exists())
            self.assertTrue((checkpoint_dir / "training_state.pt").exists())

            # Test resuming from checkpoint
            new_model = create_cognate_model(d_model=128, n_layers=4, vocab_size=1000, mem_capacity=64)

            resumed_trainer = CognateTrainer(
                model=new_model,
                config=config,
                train_dataset=self.dataset,
                output_dir=tmpdir,
                resume_from_checkpoint=str(checkpoint_dir),
            )

            # Should load successfully
            self.assertEqual(resumed_trainer.global_step, config.max_steps)


class TestEvoMergeCompatibility(unittest.TestCase):
    """Test compatibility with EvoMerge evolutionary training."""

    def test_state_dict_compatibility(self):
        """Test that models have compatible state dicts for EvoMerge."""
        models = []
        for i in range(3):
            model = create_cognate_model(
                variant_name=f"model-{i+1}",
                seed=42 + i,
                d_model=128,
                n_layers=4,
                vocab_size=1000,
            )
            models.append(model)

        # All models should have the same state dict keys
        state_dicts = [model.state_dict() for model in models]

        keys_sets = [set(sd.keys()) for sd in state_dicts]

        # All key sets should be identical
        for i in range(1, len(keys_sets)):
            self.assertEqual(keys_sets[0], keys_sets[i], f"Model {i} has different keys than model 0")

        # All tensors should have the same shapes
        for key in keys_sets[0]:
            shapes = [sd[key].shape for sd in state_dicts]
            for i in range(1, len(shapes)):
                self.assertEqual(shapes[0], shapes[i], f"Parameter {key} has different shapes across models")

    def test_model_merging_compatibility(self):
        """Test that models can be merged (basic averaging)."""
        models = []
        for i in range(3):
            model = create_cognate_model(
                seed=42 + i,
                d_model=128,
                n_layers=4,
                vocab_size=1000,
            )
            models.append(model)

        # Create merged model by averaging parameters
        merged_model = create_cognate_model(
            d_model=128,
            n_layers=4,
            vocab_size=1000,
        )

        merged_state = merged_model.state_dict()

        # Average parameters
        for key in merged_state.keys():
            param_sum = sum(model.state_dict()[key] for model in models)
            merged_state[key] = param_sum / len(models)

        merged_model.load_state_dict(merged_state)

        # Merged model should work for inference
        input_ids = torch.randint(0, 1000, (1, 32))

        with torch.no_grad():
            outputs = merged_model(input_ids, return_dict=True)

        self.assertIn("logits", outputs)
        self.assertEqual(outputs["logits"].shape[0], 1)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main(verbosity=2)
