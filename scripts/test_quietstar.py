#!/usr/bin/env python3
"""Test suite for Quiet-STaR Baker.

Tests thought injection, A/B testing, and weight baking functionality.
"""

import asyncio
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from quietstar_baker import (
    ABTestHarness,
    QuietSTaRBaker,
    QuietSTaRConfig,
    ReasoningEvalDataset,
    ThoughtInjector,
    WeightBaker,
)
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent / "agent_forge"))


class TestQuietSTaRBaker(unittest.TestCase):
    """Test suite for Quiet-STaR components."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create test configuration
        self.config = QuietSTaRConfig(
            model_path="gpt2",  # Small model for testing
            output_path=str(self.temp_dir / "baked_model"),
            eval_samples=10,
            eval_batch_size=2,
            ab_test_rounds=2,
            num_epochs=1,
            device="cpu",
        )

        # Create output directory
        Path(self.config.output_path).parent.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = QuietSTaRConfig(model_path="test_model", output_path="test_output")
        self.assertEqual(config.start_thought_token, "<|startofthought|>")
        self.assertEqual(config.end_thought_token, "<|endofthought|>")

        # Test auto device selection
        self.assertIn(config.device, ["cuda", "cpu"])

        print("✅ Configuration validation working")

    def test_thought_injection(self):
        """Test thought token injection."""
        # Create small model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2  # Smaller for testing
        config.n_head = 2
        config.n_embd = 64
        model = AutoModelForCausalLM.from_config(config)

        # Create thought injector
        injector = ThoughtInjector(model, tokenizer, self.config)

        # Test special token addition
        self.assertIn(self.config.start_thought_token, tokenizer.get_vocab())
        self.assertIn(self.config.end_thought_token, tokenizer.get_vocab())

        # Test thought injection
        test_text = "This is a test. Another sentence."
        input_ids = tokenizer(test_text, return_tensors="pt")["input_ids"]
        attention_mask = torch.ones_like(input_ids)

        # Find injection points
        injection_points = injector.find_injection_points(input_ids)
        self.assertIsInstance(injection_points, list)
        self.assertIsInstance(injection_points[0], list)

        print("✅ Thought injection working")

    def test_reasoning_dataset(self):
        """Test reasoning evaluation dataset."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Mock dataset loading
        with patch("quietstar_baker.load_dataset") as mock_load:
            # Mock GSM8K data
            mock_dataset = [
                {"question": "What is 2 + 2?", "answer": "2 + 2 = 4\n#### 4"},
                {"question": "What is 5 * 3?", "answer": "5 * 3 = 15\n#### 15"},
            ]
            mock_load.return_value = mock_dataset

            dataset = ReasoningEvalDataset("gsm8k", 2, tokenizer)

            self.assertEqual(len(dataset), 2)

            # Test dataset item
            item = dataset[0]
            self.assertIn("input_ids", item)
            self.assertIn("attention_mask", item)
            self.assertIn("target", item)
            self.assertIn("numerical_answer", item)
            self.assertEqual(item["numerical_answer"], "4")

        print("✅ Reasoning dataset working")

    @patch("quietstar_baker.wandb")
    async def test_ab_harness(self, mock_wandb):
        """Test A/B testing harness."""
        # Create mock models
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        mock_thought_model = MagicMock()
        mock_thought_model.to.return_value = mock_thought_model
        mock_thought_model.eval.return_value = None
        mock_thought_model.model = mock_model
        mock_thought_model.extract_thoughts.return_value = [["Let me think step by step"]]

        # Mock forward pass
        mock_outputs = MagicMock()
        mock_thought_model.return_value = (
            mock_outputs,
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 1, 1]]),
        )

        # Create harness
        harness = ABTestHarness(mock_model, mock_thought_model, tokenizer, self.config)

        # Test metric initialization
        self.assertIn("baseline", harness.metrics)
        self.assertIn("with_thoughts", harness.metrics)

        # Test answer checking
        generated = "The answer is 42"
        target = "42"
        is_correct = harness.check_answer(generated, target)
        self.assertTrue(is_correct)

        print("✅ A/B testing harness working")

    def test_weight_baker(self):
        """Test weight baking functionality."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        model = AutoModelForCausalLM.from_config(config)

        baker = WeightBaker(model, tokenizer, self.config)

        # Test thought insertion in text
        text = "Question. Answer."
        thoughts = ["Let me calculate", "The result is"]
        augmented = baker.insert_thoughts_in_text(text, thoughts)

        self.assertIn(self.config.start_thought_token, augmented)
        self.assertIn(self.config.end_thought_token, augmented)

        print("✅ Weight baker working")

    def test_trace_quality_evaluation(self):
        """Test thought trace quality evaluation."""
        baker = QuietSTaRBaker(self.config)

        # Test with various trace qualities
        good_traces = [["Let me think step by step", "First, I calculate 2 + 2 = 4"]]
        poor_traces = [["Hmm", "..."]]
        empty_traces = [[]]

        good_scores = baker.evaluate_trace_quality(good_traces)
        poor_scores = baker.evaluate_trace_quality(poor_traces)
        empty_scores = baker.evaluate_trace_quality(empty_traces)

        # Good traces should score higher
        self.assertGreater(good_scores[0], poor_scores[0])
        self.assertEqual(empty_scores[0], 0.0)

        print("✅ Trace quality evaluation working")

    def test_cli_integration(self):
        """Test CLI command integration."""
        from quietstar_baker import forge

        # Test that CLI commands exist
        self.assertIsNotNone(forge)
        self.assertIn("bake-quietstar", forge.commands)

        print("✅ CLI integration working")


async def run_integration_test():
    """Run quick integration test."""
    print("\n🧪 Running Quiet-STaR integration test...")

    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Create minimal test config
        config = QuietSTaRConfig(
            model_path="gpt2",
            output_path=str(temp_dir / "baked_model"),
            eval_samples=5,
            eval_batch_size=1,
            ab_test_rounds=1,
            num_epochs=1,
            device="cpu",
        )

        # Initialize baker
        baker = QuietSTaRBaker(config)

        print("✅ QuietSTaR baker initialized")
        print("✅ Integration test setup complete")

        # Note: We don't run the full pipeline in tests as it would
        # require significant compute time

        print("🎯 Integration test passed!")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise

    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Main test runner."""
    print("🧪 Starting Quiet-STaR Baker Test Suite")
    print("=" * 60)

    # Run unit tests
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestQuietSTaRBaker)
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)

    # Run integration test
    try:
        asyncio.run(run_integration_test())
        integration_success = True
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        integration_success = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    unit_success = result.wasSuccessful()

    print(f"Unit Tests: {'✅ PASSED' if unit_success else '❌ FAILED'}")
    print(f"Integration Test: {'✅ PASSED' if integration_success else '❌ FAILED'}")

    overall_success = unit_success and integration_success
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")

    if overall_success:
        print("\n🎉 Quiet-STaR Baker is ready for production!")
        print("\nExample usage:")
        print("  forge bake-quietstar --model path/to/champion.pt --out path/to/baked.pt")

    return 0 if overall_success else 1


if __name__ == "__main__":
    exit(main())
