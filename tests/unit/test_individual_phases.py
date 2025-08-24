#!/usr/bin/env python3
"""
Individual Phase Testing

Tests each Agent Forge phase independently to ensure they work correctly
before running the full pipeline. Each test validates the phase interface,
basic functionality, and model passing.
"""

import asyncio
import logging
from pathlib import Path
import sys

import torch.nn as nn

# Add paths for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.phase_controller import PhaseController, PhaseResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestResults:
    """Container for test results."""

    def __init__(self):
        self.passed = []
        self.failed = []
        self.skipped = []

    def add_pass(self, test_name: str, message: str = ""):
        self.passed.append((test_name, message))
        logger.info(f"✅ PASS: {test_name} - {message}")

    def add_fail(self, test_name: str, error: str):
        self.failed.append((test_name, error))
        logger.error(f"❌ FAIL: {test_name} - {error}")

    def add_skip(self, test_name: str, reason: str):
        self.skipped.append((test_name, reason))
        logger.warning(f"⚠️  SKIP: {test_name} - {reason}")

    def summary(self):
        total = len(self.passed) + len(self.failed) + len(self.skipped)
        print("\n" + "=" * 80)
        print("INDIVIDUAL PHASE TEST RESULTS")
        print("=" * 80)
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {len(self.passed)}")
        print(f"❌ Failed: {len(self.failed)}")
        print(f"⚠️  Skipped: {len(self.skipped)}")

        if self.failed:
            print("\nFAILED TESTS:")
            for test_name, error in self.failed:
                print(f"  • {test_name}: {error}")

        if self.skipped:
            print("\nSKIPPED TESTS:")
            for test_name, reason in self.skipped:
                print(f"  • {test_name}: {reason}")

        success_rate = len(self.passed) / total * 100 if total > 0 else 0
        print(f"\nSuccess Rate: {success_rate:.1f}%")
        print("=" * 80)

        return len(self.failed) == 0


def create_test_model() -> nn.Module:
    """Create a test model for phase testing."""
    model = nn.Sequential(
        nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 50257)  # vocab_size
    )

    # Add config attributes expected by phases
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

    return model


async def test_evomerge_phase(results: TestResults):
    """Test EvoMerge phase individually."""
    test_name = "EvoMerge Phase"

    try:
        from phases.evomerge import EvoMergeConfig, EvoMergePhase

        # Create minimal config for testing
        config = EvoMergeConfig(
            population_size=4,  # Small for testing
            generations=2,  # Quick test
            techniques=["linear", "slerp"],
            enable_grokfast=False,  # Disable for faster testing
        )

        phase = EvoMergePhase(config)

        # Test with list of model identifiers (EvoMerge takes model list, not single model)
        base_models = ["dummy_model_1", "dummy_model_2"]

        # Run phase (will use mock models internally)
        result = await phase.run(base_models)

        # Validate result
        if not isinstance(result, PhaseResult):
            results.add_fail(test_name, "Result is not PhaseResult instance")
            return

        if not result.success:
            results.add_fail(test_name, f"Phase failed: {result.error}")
            return

        if not isinstance(result.model, nn.Module):
            results.add_fail(test_name, "Result model is not nn.Module")
            return

        results.add_pass(test_name, f"Completed in {result.duration_seconds:.2f}s")

    except ImportError as e:
        results.add_skip(test_name, f"Import failed: {e}")
    except Exception as e:
        results.add_fail(test_name, f"Unexpected error: {e}")


async def test_quietstar_phase(results: TestResults):
    """Test Quiet-STaR phase individually."""
    test_name = "Quiet-STaR Phase"

    try:
        from phases.quietstar import QuietSTaRConfig, QuietSTaRPhase

        config = QuietSTaRConfig(
            thought_length=8,  # Small for testing
            num_thoughts=2,  # Quick test
            training_steps=10,  # Minimal training
            enable_grokfast=False,  # Disable for faster testing
        )

        phase = QuietSTaRPhase(config)
        model = create_test_model()

        result = await phase.run(model)

        if not isinstance(result, PhaseResult):
            results.add_fail(test_name, "Result is not PhaseResult instance")
            return

        if not result.success:
            results.add_fail(test_name, f"Phase failed: {result.error}")
            return

        if not isinstance(result.model, nn.Module):
            results.add_fail(test_name, "Result model is not nn.Module")
            return

        results.add_pass(test_name, f"Completed in {result.duration_seconds:.2f}s")

    except ImportError as e:
        results.add_skip(test_name, f"Import failed: {e}")
    except Exception as e:
        results.add_fail(test_name, f"Unexpected error: {e}")


async def test_bitnet_phase(results: TestResults):
    """Test BitNet compression phase individually."""
    test_name = "BitNet Compression Phase"

    try:
        from phases.bitnet_compression import BitNetCompressionPhase, BitNetConfig

        config = BitNetConfig(
            bits=1.58,
            group_size=64,  # Small for testing
            calibration_samples=10,  # Minimal calibration
            enable_fine_tuning=False,  # Skip for testing
        )

        phase = BitNetCompressionPhase(config)
        model = create_test_model()

        result = await phase.run(model)

        if not isinstance(result, PhaseResult):
            results.add_fail(test_name, "Result is not PhaseResult instance")
            return

        if not result.success:
            results.add_fail(test_name, f"Phase failed: {result.error}")
            return

        if not isinstance(result.model, nn.Module):
            results.add_fail(test_name, "Result model is not nn.Module")
            return

        results.add_pass(test_name, f"Completed in {result.duration_seconds:.2f}s")

    except ImportError as e:
        results.add_skip(test_name, f"Import failed: {e}")
    except Exception as e:
        results.add_fail(test_name, f"Unexpected error: {e}")


async def test_forge_training_phase(results: TestResults):
    """Test Forge Training phase individually."""
    test_name = "Forge Training Phase"

    try:
        from phases.forge_training import ForgeTrainingConfig, ForgeTrainingPhase

        config = ForgeTrainingConfig(
            training_steps=20,  # Minimal training
            batch_size=2,  # Small batch
            learning_rate=1e-4,
            enable_grokfast=True,  # Test Grokfast
            enable_edge_control=True,  # Test edge-of-chaos
            enable_self_model=False,  # Skip for testing
            enable_dream=False,  # Skip for testing
        )

        phase = ForgeTrainingPhase(config)
        model = create_test_model()

        result = await phase.run(model)

        if not isinstance(result, PhaseResult):
            results.add_fail(test_name, "Result is not PhaseResult instance")
            return

        if not result.success:
            results.add_fail(test_name, f"Phase failed: {result.error}")
            return

        if not isinstance(result.model, nn.Module):
            results.add_fail(test_name, "Result model is not nn.Module")
            return

        # Check if Grokfast was actually used
        if result.metrics and "grokfast_activations" in result.metrics:
            grokfast_used = result.metrics["grokfast_activations"] > 0
            results.add_pass(
                test_name, f"Completed with Grokfast: {grokfast_used}, Duration: {result.duration_seconds:.2f}s"
            )
        else:
            results.add_pass(test_name, f"Completed in {result.duration_seconds:.2f}s")

    except ImportError as e:
        results.add_skip(test_name, f"Import failed: {e}")
    except Exception as e:
        results.add_fail(test_name, f"Unexpected error: {e}")


async def test_tool_persona_baking_phase(results: TestResults):
    """Test Tool & Persona Baking phase individually."""
    test_name = "Tool & Persona Baking Phase"

    try:
        from phases.tool_persona_baking import ToolPersonaBakingConfig, ToolPersonaBakingPhase

        config = ToolPersonaBakingConfig(
            tools_to_bake=["calculator"],  # Minimal tool set
            persona_traits={"helpfulness": 0.8},  # Single trait
            enable_grokfast=True,
            baking_iterations=5,  # Quick baking
            convergence_threshold=0.1,  # Loose threshold
        )

        phase = ToolPersonaBakingPhase(config)
        model = create_test_model()

        result = await phase.run(model)

        if not isinstance(result, PhaseResult):
            results.add_fail(test_name, "Result is not PhaseResult instance")
            return

        if not result.success:
            results.add_fail(test_name, f"Phase failed: {result.error}")
            return

        if not isinstance(result.model, nn.Module):
            results.add_fail(test_name, "Result model is not nn.Module")
            return

        results.add_pass(test_name, f"Completed in {result.duration_seconds:.2f}s")

    except ImportError as e:
        results.add_skip(test_name, f"Import failed: {e}")
    except Exception as e:
        results.add_fail(test_name, f"Unexpected error: {e}")


async def test_adas_phase(results: TestResults):
    """Test ADAS phase individually."""
    test_name = "ADAS Phase"

    try:
        from phases.adas import ADASConfig, ADASPhase

        config = ADASConfig(
            population_size=4,  # Small population
            num_generations=2,  # Quick evolution
            composition_scale=0.1,
            enable_grokfast_training=False,  # Skip for testing
        )

        phase = ADASPhase(config)
        model = create_test_model()

        result = await phase.run(model)

        if not isinstance(result, PhaseResult):
            results.add_fail(test_name, "Result is not PhaseResult instance")
            return

        if not result.success:
            results.add_fail(test_name, f"Phase failed: {result.error}")
            return

        if not isinstance(result.model, nn.Module):
            results.add_fail(test_name, "Result model is not nn.Module")
            return

        # Check for ADAS-specific metrics
        if result.metrics and "best_architecture_score" in result.metrics:
            score = result.metrics["best_architecture_score"]
            results.add_pass(
                test_name, f"Best architecture score: {score:.4f}, Duration: {result.duration_seconds:.2f}s"
            )
        else:
            results.add_pass(test_name, f"Completed in {result.duration_seconds:.2f}s")

    except ImportError as e:
        results.add_skip(test_name, f"Import failed: {e}")
    except Exception as e:
        results.add_fail(test_name, f"Unexpected error: {e}")


async def test_final_compression_phase(results: TestResults):
    """Test Final Compression phase individually."""
    test_name = "Final Compression Phase"

    try:
        from phases.final_compression import FinalCompressionConfig, FinalCompressionPhase

        config = FinalCompressionConfig(
            enable_seedlm=True,
            enable_vptq=True,
            enable_hypercompression=True,
            seedlm_bits_per_weight=4,
            vptq_bits=2,
            hyper_num_clusters=4,  # Small for testing
            enable_grokfast_optimization=False,  # Skip for testing
        )

        phase = FinalCompressionPhase(config)
        model = create_test_model()

        result = await phase.run(model)

        if not isinstance(result, PhaseResult):
            results.add_fail(test_name, "Result is not PhaseResult instance")
            return

        if not result.success:
            results.add_fail(test_name, f"Phase failed: {result.error}")
            return

        if not isinstance(result.model, nn.Module):
            results.add_fail(test_name, "Result model is not nn.Module")
            return

        # Check for compression metrics
        if result.metrics and "total_compression_ratio" in result.metrics:
            ratio = result.metrics["total_compression_ratio"]
            results.add_pass(test_name, f"Compression ratio: {ratio:.2f}x, Duration: {result.duration_seconds:.2f}s")
        else:
            results.add_pass(test_name, f"Completed in {result.duration_seconds:.2f}s")

    except ImportError as e:
        results.add_skip(test_name, f"Import failed: {e}")
    except Exception as e:
        results.add_fail(test_name, f"Unexpected error: {e}")


async def test_phase_controller_interface(results: TestResults):
    """Test that all phases implement the PhaseController interface correctly."""
    test_name = "Phase Controller Interface"

    try:
        # Import all available phases
        phase_imports = [
            ("EvoMerge", "phases.evomerge", "EvoMergePhase"),
            ("QuietSTaR", "phases.quietstar", "QuietSTaRPhase"),
            ("BitNet", "phases.bitnet_compression", "BitNetCompressionPhase"),
            ("ForgeTraining", "phases.forge_training", "ForgeTrainingPhase"),
            ("ToolPersonaBaking", "phases.tool_persona_baking", "ToolPersonaBakingPhase"),
            ("ADAS", "phases.adas", "ADASPhase"),
            ("FinalCompression", "phases.final_compression", "FinalCompressionPhase"),
        ]

        interface_valid = True
        phases_checked = 0

        for phase_name, module_name, class_name in phase_imports:
            try:
                module = __import__(module_name, fromlist=[class_name])
                phase_class = getattr(module, class_name)

                # Check if it's a subclass of PhaseController
                if not issubclass(phase_class, PhaseController):
                    results.add_fail(test_name, f"{phase_name} does not inherit from PhaseController")
                    interface_valid = False
                    continue

                # Check if it has the required run method
                if not hasattr(phase_class, "run") or not asyncio.iscoroutinefunction(phase_class.run):
                    results.add_fail(test_name, f"{phase_name} does not have async run method")
                    interface_valid = False
                    continue

                phases_checked += 1

            except ImportError:
                continue  # Skip unavailable phases
            except Exception as e:
                results.add_fail(test_name, f"{phase_name} interface check failed: {e}")
                interface_valid = False

        if interface_valid and phases_checked > 0:
            results.add_pass(test_name, f"All {phases_checked} available phases implement PhaseController correctly")
        elif phases_checked == 0:
            results.add_skip(test_name, "No phases available for interface testing")

    except Exception as e:
        results.add_fail(test_name, f"Interface testing failed: {e}")


async def run_individual_phase_tests():
    """Run all individual phase tests."""
    results = TestResults()

    print("Starting Individual Phase Testing...")
    print("=" * 80)

    # Test phase controller interface first
    await test_phase_controller_interface(results)

    # Test each phase individually
    await test_evomerge_phase(results)
    await test_quietstar_phase(results)
    await test_bitnet_phase(results)
    await test_forge_training_phase(results)
    await test_tool_persona_baking_phase(results)
    await test_adas_phase(results)
    await test_final_compression_phase(results)

    # Print summary
    success = results.summary()

    return success


if __name__ == "__main__":
    success = asyncio.run(run_individual_phase_tests())
    sys.exit(0 if success else 1)
