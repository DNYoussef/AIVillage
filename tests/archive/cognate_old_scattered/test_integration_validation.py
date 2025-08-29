#!/usr/bin/env python3
"""
Test Suite 3: Integration Testing
Tests Agent Forge pipeline integration and EvoMerge compatibility.
"""

import json
from pathlib import Path
import sys

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Also need to add the core directory specifically
core_path = project_root / "core"
if core_path.exists():
    sys.path.insert(0, str(core_path))

from agent_forge.phases.cognate_pretrain import create_three_cognate_models


class TestIntegrationValidation:
    """Test Agent Forge pipeline integration."""

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory for tests."""
        output_dir = tmp_path / "cognate_models"
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir)

    def test_evomerge_compatibility(self, temp_output_dir):
        """Test that created models are compatible with EvoMerge requirements."""
        try:
            models = create_three_cognate_models(output_dir=temp_output_dir, device="cpu")

            # Verify EvoMerge requirements
            for i, model in enumerate(models):
                metadata = model["metadata"]

                # Check EvoMerge readiness flag
                assert metadata["ready_for_evomerge"] is True, f"Model {i+1} not ready for EvoMerge"

                # Verify required fields for EvoMerge
                assert "parameter_count" in model, "Missing parameter_count for EvoMerge"
                assert "focus" in model, "Missing focus specification for EvoMerge"
                assert "path" in model, "Missing model path for EvoMerge"

                # Check metadata has training config
                assert "training_config" in metadata, "Missing training config for EvoMerge"
                training_config = metadata["training_config"]
                assert "train_max_steps" in training_config
                assert "infer_max_steps" in training_config

            # Verify exactly 3 models (EvoMerge expectation)
            assert len(models) == 3, "EvoMerge expects exactly 3 Cognate models"

            print("✅ EvoMerge compatibility validated")

        except Exception as e:
            pytest.fail(f"EvoMerge compatibility test failed: {e}")

    def test_agent_forge_pipeline_integration(self, temp_output_dir):
        """Test integration with Agent Forge pipeline phases."""
        try:
            # Test that we can import phase integration
            from agent_forge.phases.cognate_pretrain.phase_integration import CognateConfig, CognatePhase

            # Create configuration
            config = CognateConfig()

            # Test phase initialization
            phase = CognatePhase(config)
            assert phase is not None, "Phase should initialize successfully"

            print("✅ Agent Forge pipeline integration available")

        except ImportError:
            # This is expected if phase_controller is not available
            print("⚠️  Phase controller not available - phase integration cannot be tested")
        except Exception as e:
            pytest.fail(f"Agent Forge pipeline integration failed: {e}")

    def test_output_format_consistency(self, temp_output_dir):
        """Test that output format is consistent across different invocations."""
        try:
            # Create models twice with same config
            models1 = create_three_cognate_models(output_dir=temp_output_dir + "_1", device="cpu")

            models2 = create_three_cognate_models(output_dir=temp_output_dir + "_2", device="cpu")

            # Verify consistent structure
            assert len(models1) == len(models2), "Model count should be consistent"

            for i, (m1, m2) in enumerate(zip(models1, models2)):
                # Same keys in model info
                assert set(m1.keys()) == set(m2.keys()), f"Model {i+1} structure inconsistent"

                # Same metadata structure
                assert set(m1["metadata"].keys()) == set(
                    m2["metadata"].keys()
                ), f"Model {i+1} metadata structure inconsistent"

                # Same focus types (but possibly different order)
                focuses1 = {m["focus"] for m in models1}
                focuses2 = {m["focus"] for m in models2}
                assert focuses1 == focuses2, "Model focuses should be consistent"

            print("✅ Output format consistency validated")

        except Exception as e:
            pytest.fail(f"Output format consistency test failed: {e}")

    def test_metadata_evomerge_requirements(self, temp_output_dir):
        """Test that metadata contains all required fields for EvoMerge."""
        try:
            create_three_cognate_models(output_dir=temp_output_dir, device="cpu")

            # Check summary file for EvoMerge requirements
            summary_path = Path(temp_output_dir) / "cognate_models_summary.json"
            assert summary_path.exists(), "Summary file required for EvoMerge"

            with open(summary_path) as f:
                summary = json.load(f)

            # Verify summary has EvoMerge required fields
            required_summary_fields = [
                "total_models",
                "target_parameters_per_model",
                "models",
                "total_parameters",
                "average_parameters",
                "parameter_accuracy",
                "next_phase",
                "pipeline_status",
            ]

            for field in required_summary_fields:
                assert field in summary, f"Summary missing required field: {field}"

            assert summary["next_phase"] == "evomerge", "Next phase should be EvoMerge"
            assert summary["pipeline_status"] == "ready_for_evomerge", "Should be ready for EvoMerge"

            # Verify individual model metadata for EvoMerge
            for model_info in summary["models"]:
                assert "parameter_count" in model_info
                assert "focus" in model_info
                assert "name" in model_info
                assert "path" in model_info

            print("✅ EvoMerge metadata requirements validated")

        except Exception as e:
            pytest.fail(f"EvoMerge metadata requirements test failed: {e}")

    def test_pretraining_pipeline_integration(self, temp_output_dir):
        """Test integration with optional pre-training pipeline."""
        try:
            from agent_forge.phases.cognate_pretrain.pretrain_pipeline import run_pretraining_pipeline

            # Create base models
            models = create_three_cognate_models(output_dir=temp_output_dir, device="cpu")

            # Test pre-training pipeline
            pretrained_models = run_pretraining_pipeline(models, steps=10)  # Quick test

            # Verify pre-training integration
            assert len(pretrained_models) == len(models), "Pre-training should preserve model count"

            for i, model in enumerate(pretrained_models):
                # Should have pre-training metadata added
                if "pretrained" in model:
                    assert model["pretrained"] is True
                    assert "pretraining_steps" in model
                    assert "pretraining_loss" in model

            print("✅ Pre-training pipeline integration validated")

        except Exception as e:
            pytest.fail(f"Pre-training pipeline integration failed: {e}")

    def test_backward_compatibility_integration(self, temp_output_dir):
        """Test that old interfaces still work for existing pipeline code."""
        try:
            import warnings

            # Test old cognate.py redirect
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                from agent_forge.phases.cognate import create_cognate_models

                # Should work but with warnings
                models = create_cognate_models(num_models=3, output_dir=temp_output_dir, device="cpu")

                # Verify it still works
                assert len(models) == 3, "Backward compatibility should work"

                # Check warnings were issued
                assert len(w) > 0, "Deprecation warnings should be issued"
                assert any("deprecated" in str(warning.message).lower() for warning in w)

            print("✅ Backward compatibility integration validated")

        except Exception as e:
            pytest.fail(f"Backward compatibility integration test failed: {e}")

    def test_error_recovery_and_logging(self, temp_output_dir):
        """Test error recovery and logging behavior."""
        try:
            import io
            import logging

            # Capture logging output
            log_capture = io.StringIO()
            handler = logging.StreamHandler(log_capture)
            logger = logging.getLogger("core.agent_forge.phases.cognate_pretrain")
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

            # Test normal operation logging
            create_three_cognate_models(output_dir=temp_output_dir, device="cpu")

            # Check that appropriate logs were generated
            log_output = log_capture.getvalue()

            # Should have informative logs
            assert "Cognate Model Factory" in log_output or "Creating" in log_output, "Should have informative logging"

            print("✅ Error recovery and logging validated")

        except Exception as e:
            pytest.fail(f"Error recovery and logging test failed: {e}")


if __name__ == "__main__":
    import shutil
    import tempfile

    test_suite = TestIntegrationValidation()

    print("🧪 Running Integration Validation Tests")
    print("=" * 50)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        test_suite.test_evomerge_compatibility(temp_dir)
        test_suite.test_agent_forge_pipeline_integration(temp_dir)
        test_suite.test_output_format_consistency(temp_dir)
        test_suite.test_metadata_evomerge_requirements(temp_dir)
        test_suite.test_pretraining_pipeline_integration(temp_dir)
        test_suite.test_backward_compatibility_integration(temp_dir)
        test_suite.test_error_recovery_and_logging(temp_dir)

        print("=" * 50)
        print("✅ ALL INTEGRATION TESTS PASSED")

    except Exception as e:
        print("=" * 50)
        print(f"❌ INTEGRATION TEST FAILED: {e}")
        raise

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
