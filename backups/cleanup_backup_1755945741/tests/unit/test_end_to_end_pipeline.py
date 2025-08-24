#!/usr/bin/env python3
"""
End-to-End Pipeline Test

Tests the complete Agent Forge pipeline with mock phases to ensure
the orchestration, model passing, and error handling works correctly.
"""

import asyncio
import logging
from pathlib import Path
import sys

import torch
import torch.nn as nn

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.phase_controller import PhaseController
from core.unified_pipeline import UnifiedConfig, UnifiedPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockEvoMergePhase(PhaseController):
    """Mock EvoMerge phase for testing."""

    def __init__(self, config):
        super().__init__(config)
        self.phase_name = "MockEvoMerge"

    async def run(self, base_models):
        """Mock EvoMerge implementation."""
        await asyncio.sleep(0.1)  # Simulate processing

        # Create merged model
        model = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 50257))

        # Add config
        model.config = type(
            "Config",
            (),
            {
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "vocab_size": 50257,
                "max_position_embeddings": 2048,
            },
        )()

        return self.create_success_result(
            model=model,
            metrics={
                "merge_technique": "linear",
                "improvement_score": 0.85,
                "models_merged": len(base_models) if isinstance(base_models, list) else 2,
            },
            duration=0.1,
        )


class MockQuietSTaRPhase(PhaseController):
    """Mock Quiet-STaR phase for testing."""

    def __init__(self, config):
        super().__init__(config)
        self.phase_name = "MockQuietSTaR"

    async def run(self, model):
        """Mock Quiet-STaR implementation."""
        await asyncio.sleep(0.1)

        # Add some parameters to simulate thought baking
        for param in model.parameters():
            param.data += torch.randn_like(param.data) * 0.001

        return self.create_success_result(
            model=model, metrics={"thoughts_baked": 4, "convergence_score": 0.92, "baking_iterations": 3}, duration=0.1
        )


class MockBitNetPhase(PhaseController):
    """Mock BitNet compression phase for testing."""

    def __init__(self, config):
        super().__init__(config)
        self.phase_name = "MockBitNet"

    async def run(self, model):
        """Mock BitNet implementation."""
        await asyncio.sleep(0.1)

        # Simulate compression (just modify model slightly)
        original_params = sum(p.numel() for p in model.parameters())

        return self.create_success_result(
            model=model,
            metrics={"compression_ratio": 8.5, "original_params": original_params, "bits_per_weight": 1.58},
            duration=0.1,
        )


class MockForgeTrainingPhase(PhaseController):
    """Mock Forge Training phase for testing."""

    def __init__(self, config):
        super().__init__(config)
        self.phase_name = "MockForgeTraining"

    async def run(self, model):
        """Mock training implementation."""
        await asyncio.sleep(0.2)  # Longer for training

        # Simulate training improvements
        training_steps = getattr(self.config, "training_steps", 50)

        return self.create_success_result(
            model=model,
            metrics={
                "training_steps": training_steps,
                "final_loss": 0.023,
                "grokfast_activations": 15,
                "edge_of_chaos_score": 0.67,
            },
            duration=0.2,
        )


class MockFailingPhase(PhaseController):
    """Mock phase that fails for testing error handling."""

    def __init__(self, config):
        super().__init__(config)
        self.phase_name = "MockFailingPhase"

    async def run(self, model):
        """Mock failing implementation."""
        await asyncio.sleep(0.05)

        return self.create_failure_result(model=model, error="Simulated phase failure for testing", duration=0.05)


class TestEndToEndPipeline:
    """End-to-end pipeline test suite."""

    def __init__(self):
        self.results = []

    def log_result(self, test_name: str, success: bool, message: str = ""):
        """Log test result."""
        self.results.append((test_name, success, message))
        status = "PASS" if success else "FAIL"
        logger.info(f"{status}: {test_name} - {message}")

    async def test_successful_pipeline(self):
        """Test a complete successful pipeline run."""
        test_name = "Successful Pipeline"

        try:
            # Create mock phases
            phases = [
                ("MockEvoMerge", MockEvoMergePhase({"test": True})),
                ("MockQuietSTaR", MockQuietSTaRPhase({"test": True})),
                ("MockBitNet", MockBitNetPhase({"test": True})),
                ("MockForgeTraining", MockForgeTrainingPhase({"training_steps": 20})),
            ]

            # Create orchestrator
            from core.phase_controller import PhaseOrchestrator

            orchestrator = PhaseOrchestrator()

            # Create initial model
            initial_model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))

            # Run phase sequence
            results = await orchestrator.run_phase_sequence(phases, initial_model)

            # Validate results
            if len(results) != len(phases):
                raise ValueError(f"Expected {len(phases)} results, got {len(results)}")

            for i, result in enumerate(results):
                if not result.success:
                    raise ValueError(f"Phase {i} failed: {result.error}")

            # Check model passing
            final_model = results[-1].model
            if not isinstance(final_model, nn.Module):
                raise ValueError("Final model is not nn.Module")

            self.log_result(test_name, True, f"All {len(phases)} phases completed successfully")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    async def test_pipeline_with_failure(self):
        """Test pipeline behavior when a phase fails."""
        test_name = "Pipeline with Failure"

        try:
            # Create phases with one failing phase
            phases = [
                ("MockEvoMerge", MockEvoMergePhase({"test": True})),
                ("MockFailingPhase", MockFailingPhase({"test": True})),
                ("MockBitNet", MockBitNetPhase({"test": True})),  # Should not run
            ]

            from core.phase_controller import PhaseOrchestrator

            orchestrator = PhaseOrchestrator()

            initial_model = nn.Sequential(nn.Linear(10, 10))

            # Run phase sequence
            results = await orchestrator.run_phase_sequence(phases, initial_model)

            # Should have 2 results (first successful, second failed)
            if len(results) != 2:
                raise ValueError(f"Expected 2 results (stopped at failure), got {len(results)}")

            if results[0].success is not True:
                raise ValueError("First phase should have succeeded")

            if results[1].success is not False:
                raise ValueError("Second phase should have failed")

            self.log_result(test_name, True, "Pipeline correctly stopped at failing phase")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    async def test_model_validation(self):
        """Test model validation between phases."""
        test_name = "Model Validation"

        try:
            from core.phase_controller import ModelPassingValidator

            validator = ModelPassingValidator()

            # Test valid model transition
            model = nn.Linear(10, 10)
            is_valid, error = validator.validate_model_transition("MockEvoMerge", "MockQuietSTaR", model)

            if not is_valid:
                raise ValueError(f"Valid model transition rejected: {error}")

            # Test invalid model (None)
            is_valid, error = validator.validate_model_transition("MockEvoMerge", "MockQuietSTaR", None)

            if is_valid:
                raise ValueError("Invalid model (None) was accepted")

            self.log_result(test_name, True, "Model validation working correctly")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    async def test_unified_pipeline_config(self):
        """Test unified pipeline with configuration."""
        test_name = "Unified Pipeline Config"

        try:
            # Create configuration with all phases disabled
            config = UnifiedConfig(
                enable_evomerge=False,
                enable_quietstar=False,
                enable_initial_compression=False,
                enable_training=False,
                enable_tool_baking=False,
                enable_adas=False,
                enable_final_compression=False,
            )

            # Create pipeline
            pipeline = UnifiedPipeline(config)

            # Should have 0 phases since all disabled
            if len(pipeline.phases) != 0:
                raise ValueError(f"Expected 0 phases, got {len(pipeline.phases)}")

            self.log_result(test_name, True, "Pipeline configuration working correctly")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    async def test_phase_compatibility(self):
        """Test phase compatibility checking."""
        test_name = "Phase Compatibility"

        try:
            from core.phase_controller import PhaseOrchestrator

            orchestrator = PhaseOrchestrator()

            # Test valid phase sequence
            valid_phases = [
                ("EvoMergePhase", MockEvoMergePhase({})),
                ("QuietSTaRPhase", MockQuietSTaRPhase({})),
                ("BitNetCompressionPhase", MockBitNetPhase({})),
            ]

            is_compatible = orchestrator.validate_phase_compatibility(valid_phases)

            if not is_compatible:
                raise ValueError("Valid phase sequence rejected as incompatible")

            # Test invalid phase sequence (out of order)
            invalid_phases = [
                ("BitNetCompressionPhase", MockBitNetPhase({})),
                ("EvoMergePhase", MockEvoMergePhase({})),  # Out of order
            ]

            is_compatible = orchestrator.validate_phase_compatibility(invalid_phases)

            if is_compatible:
                raise ValueError("Invalid phase sequence accepted as compatible")

            self.log_result(test_name, True, "Phase compatibility checking working correctly")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    def print_summary(self):
        """Print test summary."""
        passed = sum(1 for _, success, _ in self.results if success)
        total = len(self.results)

        print("\n" + "=" * 70)
        print("END-TO-END PIPELINE TEST RESULTS")
        print("=" * 70)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")

        if passed < total:
            print("\nFAILED TESTS:")
            for test_name, success, message in self.results:
                if not success:
                    print(f"  - {test_name}: {message}")

        success_rate = (passed / total * 100) if total > 0 else 0
        print(f"\nSuccess Rate: {success_rate:.1f}%")
        print("=" * 70)

        return passed == total


async def run_end_to_end_tests():
    """Run all end-to-end tests."""
    tester = TestEndToEndPipeline()

    print("Starting End-to-End Pipeline Testing...")
    print("=" * 70)

    # Run all tests
    await tester.test_successful_pipeline()
    await tester.test_pipeline_with_failure()
    await tester.test_model_validation()
    await tester.test_unified_pipeline_config()
    await tester.test_phase_compatibility()

    # Print summary
    return tester.print_summary()


if __name__ == "__main__":
    success = asyncio.run(run_end_to_end_tests())

    if success:
        print("\n*** All end-to-end tests passed! Pipeline orchestration is working correctly. ***")
    else:
        print("\n*** Some tests failed. Check the details above. ***")

    sys.exit(0 if success else 1)
