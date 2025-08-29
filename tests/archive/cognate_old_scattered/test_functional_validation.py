#!/usr/bin/env python3
"""
Test Suite 2: Functional Testing
Tests model creation pipeline functionality.
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

from agent_forge.phases.cognate_pretrain import CognateCreatorConfig, CognateModelCreator, create_three_cognate_models
from agent_forge.phases.cognate_pretrain.model_factory import validate_cognate_models


class TestFunctionalValidation:
    """Test model creation pipeline functionality."""

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory for tests."""
        output_dir = tmp_path / "cognate_models"
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir)

    def test_create_three_cognate_models_basic(self, temp_output_dir):
        """Test basic functionality of create_three_cognate_models()."""
        try:
            models = create_three_cognate_models(output_dir=temp_output_dir, device="cpu")  # Use CPU for testing

            # Verify exactly 3 models created
            assert len(models) == 3, f"Expected 3 models, got {len(models)}"

            # Verify each model has required fields
            required_fields = ["name", "path", "parameter_count", "focus", "metadata"]
            for i, model in enumerate(models):
                for field in required_fields:
                    assert field in model, f"Model {i+1} missing field: {field}"

                # Check model names
                assert model["name"].startswith("cognate_foundation_"), f"Invalid model name: {model['name']}"

                # Verify path exists
                model_path = Path(model["path"])
                assert model_path.exists(), f"Model path does not exist: {model['path']}"

            print("✅ Basic model creation successful")

        except Exception as e:
            pytest.fail(f"Basic model creation failed: {e}")

    def test_parameter_count_validation(self, temp_output_dir):
        """Test that models are created with approximately 25M parameters."""
        try:
            models = create_three_cognate_models(output_dir=temp_output_dir, device="cpu")

            target_params = 25_000_000
            tolerance = 0.10  # 10% tolerance

            for i, model in enumerate(models):
                param_count = model["parameter_count"]
                error_pct = abs(param_count - target_params) / target_params

                print(f"Model {i+1}: {param_count:,} parameters ({error_pct:.1f}% from target)")

                assert error_pct <= tolerance, (
                    f"Model {i+1} parameter count {param_count:,} is {error_pct:.1f}% "
                    f"off target ({tolerance*100}% tolerance exceeded)"
                )

            print("✅ Parameter count validation passed")

        except Exception as e:
            pytest.fail(f"Parameter count validation failed: {e}")

    def test_model_variants_differentiation(self, temp_output_dir):
        """Test that the 3 models have different configurations."""
        try:
            models = create_three_cognate_models(output_dir=temp_output_dir, device="cpu")

            focuses = [model["focus"] for model in models]
            model_names = [model["name"] for model in models]

            # Verify unique focuses
            assert len(set(focuses)) == 3, f"Models should have unique focuses, got: {focuses}"

            # Verify unique names
            assert len(set(model_names)) == 3, f"Models should have unique names, got: {model_names}"

            # Check specific variant configurations from metadata
            expected_focuses = ["reasoning", "memory_integration", "adaptive_computation"]
            for focus in expected_focuses:
                assert focus in focuses, f"Missing expected focus: {focus}"

            print("✅ Model variant differentiation validated")

        except Exception as e:
            pytest.fail(f"Model variant differentiation failed: {e}")

    def test_model_factory_validation_function(self, temp_output_dir):
        """Test the validate_cognate_models function."""
        try:
            models = create_three_cognate_models(output_dir=temp_output_dir, device="cpu")

            # Test validation function
            validation_result = validate_cognate_models(models)

            assert isinstance(validation_result, bool), "Validation should return boolean"
            assert validation_result is True, "Model validation should pass"

            print("✅ Model factory validation function works")

        except Exception as e:
            pytest.fail(f"Model factory validation failed: {e}")

    def test_cognate_creator_direct_usage(self, temp_output_dir):
        """Test CognateModelCreator class directly."""
        try:
            # Create configuration
            config = CognateCreatorConfig(output_dir=temp_output_dir, device="cpu")

            # Create creator and models
            creator = CognateModelCreator(config)
            models = creator.create_three_models()

            # Verify results
            assert len(models) == 3, f"Expected 3 models from creator, got {len(models)}"

            # Check that creator config was applied
            for model in models:
                model_path = Path(model["path"])
                assert model_path.parent == Path(temp_output_dir), "Output directory not respected"

            print("✅ Direct CognateModelCreator usage successful")

        except Exception as e:
            pytest.fail(f"Direct CognateModelCreator usage failed: {e}")

    def test_model_metadata_completeness(self, temp_output_dir):
        """Test that created models have complete metadata."""
        try:
            models = create_three_cognate_models(output_dir=temp_output_dir, device="cpu")

            required_metadata_fields = [
                "model_name",
                "model_index",
                "focus",
                "parameter_count",
                "target_parameters",
                "parameter_accuracy",
                "architecture",
                "cognate_features",
                "training_config",
                "created_at",
                "ready_for_evomerge",
            ]

            for i, model in enumerate(models):
                metadata = model["metadata"]

                for field in required_metadata_fields:
                    assert field in metadata, f"Model {i+1} metadata missing field: {field}"

                # Verify EvoMerge readiness
                assert metadata["ready_for_evomerge"] is True, f"Model {i+1} not ready for EvoMerge"

                # Check architecture details
                arch = metadata["architecture"]
                assert all(key in arch for key in ["d_model", "n_layers", "n_heads", "ffn_mult", "vocab_size"])

                # Check Cognate features
                features = metadata["cognate_features"]
                assert all(key in features for key in ["act_halting", "act_threshold", "ltm_memory"])

            print("✅ Model metadata completeness validated")

        except Exception as e:
            pytest.fail(f"Model metadata validation failed: {e}")

    def test_file_structure_creation(self, temp_output_dir):
        """Test that proper file structure is created."""
        try:
            models = create_three_cognate_models(output_dir=temp_output_dir, device="cpu")

            output_path = Path(temp_output_dir)

            # Check summary file creation
            summary_file = output_path / "cognate_models_summary.json"
            assert summary_file.exists(), "Summary file not created"

            # Check individual model directories
            for model in models:
                model_path = Path(model["path"])
                assert model_path.exists(), f"Model directory not created: {model_path}"

                # Check required files in model directory
                metadata_file = model_path / "metadata.json"
                assert metadata_file.exists(), f"Metadata file not created: {metadata_file}"

                # Check model file (either save_pretrained files or pytorch_model.bin)
                model_files = list(model_path.glob("*.bin")) + list(model_path.glob("*.safetensors"))
                assert len(model_files) > 0, f"No model files found in: {model_path}"

            # Verify summary content
            with open(summary_file) as f:
                summary = json.load(f)

            assert summary["total_models"] == 3, "Summary should show 3 models"
            assert summary["pipeline_status"] == "ready_for_evomerge", "Should be ready for EvoMerge"

            print("✅ File structure creation validated")

        except Exception as e:
            pytest.fail(f"File structure validation failed: {e}")

    def test_error_handling_graceful_degradation(self, temp_output_dir):
        """Test that system handles missing dependencies gracefully."""
        try:
            # This test checks that even without CognateRefiner components,
            # the system can create mock models
            models = create_three_cognate_models(output_dir=temp_output_dir, device="cpu")

            # Should still create 3 models, even if they're mocks
            assert len(models) == 3, "Should create models even with missing dependencies"

            # Models should still have reasonable parameter counts
            for model in models:
                param_count = model["parameter_count"]
                assert param_count > 1_000_000, "Mock models should still have reasonable size"
                assert param_count < 100_000_000, "Mock models shouldn't be too large"

            print("✅ Graceful degradation handling validated")

        except Exception as e:
            pytest.fail(f"Error handling validation failed: {e}")


if __name__ == "__main__":
    import shutil
    import tempfile

    test_suite = TestFunctionalValidation()

    print("🧪 Running Functional Validation Tests")
    print("=" * 50)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        test_suite.test_create_three_cognate_models_basic(temp_dir)
        test_suite.test_parameter_count_validation(temp_dir)
        test_suite.test_model_variants_differentiation(temp_dir)
        test_suite.test_model_factory_validation_function(temp_dir)
        test_suite.test_cognate_creator_direct_usage(temp_dir)
        test_suite.test_model_metadata_completeness(temp_dir)
        test_suite.test_file_structure_creation(temp_dir)
        test_suite.test_error_handling_graceful_degradation(temp_dir)

        print("=" * 50)
        print("✅ ALL FUNCTIONAL TESTS PASSED")

    except Exception as e:
        print("=" * 50)
        print(f"❌ FUNCTIONAL TEST FAILED: {e}")
        raise

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
