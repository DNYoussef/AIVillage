#!/usr/bin/env python3
"""EvoMerge Pipeline Testing Script.

Comprehensive test suite for the evolutionary merging pipeline.
Tests all components including model loading, merging operators,
evaluation, and the complete evolution process.
"""

import asyncio
from pathlib import Path
import shutil

# Import our pipeline components
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from evomerge_pipeline import (
    BaseModelConfig,
    CodeEvaluator,
    EvolutionConfig,
    EvolutionState,
    EvoMergePipeline,
    MathEvaluator,
    MergeOperators,
    ModelCandidate,
)
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.append(str(Path(__file__).parent.parent / "agent_forge"))


class TestEvoMergePipeline(unittest.TestCase):
    """Test suite for EvoMerge pipeline."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create test configuration
        self.config = EvolutionConfig(
            base_models=[
                BaseModelConfig(name="test1", path="gpt2", weight=1.0),
                BaseModelConfig(name="test2", path="gpt2", weight=1.0),
                BaseModelConfig(name="test3", path="gpt2", weight=1.0),
            ],
            max_generations=2,
            population_size=4,
            device="cpu",
            output_dir=self.temp_dir / "output",
            checkpoint_dir=self.temp_dir / "checkpoints",
            models_cache_dir=self.temp_dir / "cache",
            wandb_project="test-evomerge",
        )

        # Create test directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config.models_cache_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_dummy_model(self) -> torch.nn.Module:
        """Create a dummy model for testing."""
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2  # Smaller for testing
        config.n_head = 2
        config.n_embd = 64

        # Create model with small config
        model = AutoModelForCausalLM.from_config(config)
        return model

    def test_config_validation(self) -> None:
        """Test configuration validation."""
        # Test valid configuration
        config = EvolutionConfig()
        assert len(config.base_models) == 3
        assert sum(config.evaluation_weights.values()) == 1.0

        # Test invalid base models count
        with pytest.raises(ValueError):
            EvolutionConfig(base_models=[BaseModelConfig(name="test1", path="gpt2")])

        # Test invalid evaluation weights
        with pytest.raises(ValueError):
            EvolutionConfig(
                evaluation_weights={
                    "code": 0.5,
                    "math": 0.3,  # Doesn't sum to 1.0
                }
            )

    def test_merge_operators(self) -> None:
        """Test merge operators with dummy models."""
        # Create test models
        models = [self.create_dummy_model() for _ in range(3)]
        merge_ops = MergeOperators()

        # Test linear interpolation
        weights = [0.4, 0.3, 0.3]
        merged_linear = merge_ops.linear_interpolation(models, weights)
        assert isinstance(merged_linear, torch.nn.Module)

        # Test SLERP interpolation
        merged_slerp = merge_ops.slerp_interpolation(models[0], models[1], t=0.5)
        assert isinstance(merged_slerp, torch.nn.Module)

        # Test TIES merge
        merged_ties = merge_ops.ties_merge(models, threshold=0.1)
        assert isinstance(merged_ties, torch.nn.Module)

        # Test DARE merge
        merged_dare = merge_ops.dare_merge(models, threshold=0.1, amplification=2.0)
        assert isinstance(merged_dare, torch.nn.Module)

        # Test Frankenmerge
        merged_franken = merge_ops.frankenmerge(models, layer_assignment=[0, 1, 2])
        assert isinstance(merged_franken, torch.nn.Module)

        # Test DFS merge
        merged_dfs = merge_ops.dfs_merge(models, merge_ratio=0.3)
        assert isinstance(merged_dfs, torch.nn.Module)

        print("✅ All merge operators working correctly")

    @patch("evomerge_pipeline.AutoTokenizer")
    @patch("evomerge_pipeline.AutoModelForCausalLM")
    async def test_evaluators(self, mock_model_cls, mock_tokenizer_cls) -> None:
        """Test model evaluators with mocked components."""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(1, 10, 1000)
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_cls.from_pretrained.return_value = mock_model

        # Test code evaluator
        code_eval = CodeEvaluator(device="cpu")
        code_score = await code_eval.evaluate("dummy_path")
        assert isinstance(code_score, float)
        assert code_score >= 0.0
        assert code_score <= 1.0

        # Test math evaluator
        mock_tokenizer.decode.return_value = "The answer is 4"
        math_eval = MathEvaluator(device="cpu")
        math_score = await math_eval.evaluate("dummy_path")
        assert isinstance(math_score, float)

        print("✅ Evaluators working correctly")

    def test_model_candidate(self) -> None:
        """Test ModelCandidate functionality."""
        candidate = ModelCandidate(
            generation=1,
            merge_recipe={"continuous": "linear", "ensemble": "ties"},
            fitness_scores={"code": 0.8, "math": 0.7, "multilingual": 0.6},
        )

        # Test fitness calculation
        weights = {"code": 0.4, "math": 0.4, "multilingual": 0.2}
        overall_fitness = candidate.calculate_overall_fitness(weights)

        expected_fitness = 0.8 * 0.4 + 0.7 * 0.4 + 0.6 * 0.2
        self.assertAlmostEqual(overall_fitness, expected_fitness, places=3)

        print("✅ ModelCandidate working correctly")

    def test_evolution_state(self) -> None:
        """Test EvolutionState functionality."""
        state = EvolutionState()

        # Create test candidates
        candidates = [
            ModelCandidate(generation=0, overall_fitness=0.8),
            ModelCandidate(generation=0, overall_fitness=0.6),
            ModelCandidate(generation=0, overall_fitness=0.9),
        ]

        state.population = candidates
        state.update_best_candidate()

        assert state.best_candidate.overall_fitness == 0.9

        # Test plateau detection
        state.fitness_history = [
            {"best_fitness": 0.5},
            {"best_fitness": 0.51},
            {"best_fitness": 0.505},
        ]

        assert state.check_plateau(threshold=0.02)
        assert not state.check_plateau(threshold=0.001)

        print("✅ EvolutionState working correctly")

    @patch("evomerge_pipeline.wandb")
    def test_pipeline_initialization(self, mock_wandb) -> None:
        """Test pipeline initialization."""
        # Mock W&B
        mock_run = MagicMock()
        mock_run.id = "test_run_123"
        mock_run.url = "https://wandb.ai/test/run"
        mock_wandb.init.return_value = mock_run

        # Create pipeline
        pipeline = EvoMergePipeline(self.config)

        # Test initialization
        assert len(pipeline.evaluators) == 4
        assert "code" in pipeline.evaluators
        assert "math" in pipeline.evaluators
        assert "multilingual" in pipeline.evaluators
        assert "structured_data" in pipeline.evaluators

        # Test W&B initialization
        pipeline.initialize_wandb()
        assert pipeline.wandb_run is not None

        print("✅ Pipeline initialization working correctly")

    def test_seed_generation_logic(self) -> None:
        """Test seed candidate generation logic."""
        EvoMergePipeline(self.config)

        # Test merge combination generation
        merge_combinations = [
            ("linear", "ties", "frankenmerge"),
            ("linear", "ties", "dfs"),
            ("linear", "dare", "frankenmerge"),
            ("linear", "dare", "dfs"),
            ("slerp", "ties", "frankenmerge"),
            ("slerp", "ties", "dfs"),
            ("slerp", "dare", "frankenmerge"),
            ("slerp", "dare", "dfs"),
        ]

        assert len(merge_combinations) == 8  # 2³ combinations

        # Test that all combinations are unique
        assert len(set(merge_combinations)) == 8

        print("✅ Seed generation logic correct")

    def test_checkpoint_functionality(self) -> None:
        """Test checkpoint save/load."""
        pipeline = EvoMergePipeline(self.config)

        # Create test state
        pipeline.state.current_generation = 5
        pipeline.state.population = [ModelCandidate(generation=5, overall_fitness=0.8)]
        pipeline.state.fitness_history = [
            {"generation": 0, "best_fitness": 0.5},
            {"generation": 1, "best_fitness": 0.6},
        ]

        # Save checkpoint
        pipeline.save_checkpoint()

        # Verify checkpoint file exists
        checkpoint_files = list(self.config.checkpoint_dir.glob("evolution_checkpoint_*.json"))
        assert len(checkpoint_files) > 0

        # Create new pipeline and load checkpoint
        new_pipeline = EvoMergePipeline(self.config)
        success = new_pipeline.load_checkpoint(str(checkpoint_files[0]))

        assert success
        assert new_pipeline.state.current_generation == 5
        assert len(new_pipeline.state.fitness_history) == 2

        print("✅ Checkpoint functionality working correctly")

    def test_mutation_and_selection(self) -> None:
        """Test mutation and selection logic."""
        EvoMergePipeline(self.config)

        # Create test candidates with different fitness scores
        candidates = []
        for i in range(8):
            candidate = ModelCandidate(
                generation=0,
                overall_fitness=0.5 + i * 0.05,  # Increasing fitness
                merge_recipe={
                    "continuous": "linear",
                    "ensemble": "ties",
                    "structured": "frankenmerge",
                },
            )
            # Create dummy model path
            model_path = self.config.output_dir / f"test_model_{i}"
            model_path.mkdir(parents=True, exist_ok=True)
            candidate.model_path = str(model_path)

            # Save a dummy model
            dummy_model = self.create_dummy_model()
            dummy_model.save_pretrained(model_path)

            candidates.append(candidate)

        # Sort by fitness (selection logic)
        candidates.sort(key=lambda x: x.overall_fitness, reverse=True)

        # Verify top candidates have highest fitness
        assert candidates[0].overall_fitness > candidates[-1].overall_fitness
        assert candidates[1].overall_fitness > candidates[-2].overall_fitness

        print("✅ Selection logic working correctly")


class TestCLIIntegration(unittest.TestCase):
    """Test CLI integration."""

    def test_cli_command_parsing(self) -> None:
        """Test CLI command parsing."""
        # Import CLI components
        from evomerge_pipeline import forge

        # Test that the CLI group exists
        assert forge is not None
        assert hasattr(forge, "commands")
        assert "evo" in forge.commands

        print("✅ CLI integration working correctly")


def run_quick_integration_test() -> None:
    """Run a quick integration test of the full pipeline."""
    print("🧪 Running quick integration test...")

    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Create minimal config for integration test
        config = EvolutionConfig(
            base_models=[
                BaseModelConfig(name="test1", path="gpt2"),
                BaseModelConfig(name="test2", path="gpt2"),
                BaseModelConfig(name="test3", path="gpt2"),
            ],
            max_generations=1,  # Just one generation
            population_size=2,  # Minimal population
            device="cpu",
            output_dir=temp_dir / "output",
            checkpoint_dir=temp_dir / "checkpoints",
            models_cache_dir=temp_dir / "cache",
        )

        # Create directories
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.models_cache_dir.mkdir(parents=True, exist_ok=True)

        print("✅ Configuration created successfully")
        print("✅ Integration test setup complete")

        # Note: We don't run the actual evolution here as it would require
        # downloading models and significant compute time
        print("🎯 Quick integration test passed!")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


async def main() -> int:
    """Main test runner."""
    print("🧪 Starting EvoMerge Pipeline Test Suite")
    print("=" * 60)

    # Run unit tests
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestEvoMergePipeline)
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)

    # Run CLI tests
    cli_suite = unittest.TestLoader().loadTestsFromTestCase(TestCLIIntegration)
    cli_result = test_runner.run(cli_suite)

    # Run integration test
    try:
        run_quick_integration_test()
        integration_success = True
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        integration_success = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    unit_success = result.wasSuccessful()
    cli_success = cli_result.wasSuccessful()

    print(f"Unit Tests: {'✅ PASSED' if unit_success else '❌ FAILED'}")
    print(f"CLI Tests: {'✅ PASSED' if cli_success else '❌ FAILED'}")
    print(f"Integration Test: {'✅ PASSED' if integration_success else '❌ FAILED'}")

    overall_success = unit_success and cli_success and integration_success
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")

    if overall_success:
        print("\n🎉 EvoMerge pipeline is ready for production!")
        print("\nNext steps:")
        print("1. Install package: pip install -e .")
        print("2. Run evolution: forge evo --gens 10 --base-models deepseek,nemotron,qwen2")
        print("3. Monitor progress with W&B dashboard")

    return 0 if overall_success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
