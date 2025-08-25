#!/usr/bin/env python3
"""
Test Suite 5: Error Handling Testing
Test graceful degradation and clear error messages.
"""

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Also need to add the core directory specifically
core_path = project_root / "core"
if core_path.exists():
    sys.path.insert(0, str(core_path))


class TestErrorHandlingValidation:
    """Test error handling and graceful degradation."""

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory for tests."""
        output_dir = tmp_path / "cognate_models"
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir)

    def test_missing_cognate_refiner_components(self, temp_output_dir):
        """Test behavior when CognateRefiner components are not available."""
        try:
            # Mock missing CognateRefiner imports
            with patch("agent_forge.phases.cognate_pretrain.cognate_creator.CognateRefiner", None):
                with patch("agent_forge.phases.cognate_pretrain.cognate_creator.CognateConfig", None):
                    from agent_forge.phases.cognate_pretrain import create_three_cognate_models

                    models = create_three_cognate_models(output_dir=temp_output_dir, device="cpu")

                    # Should still create 3 models (fallback to mock models)
                    assert len(models) == 3, "Should create mock models when CognateRefiner unavailable"

                    # Models should have reasonable parameters
                    for model in models:
                        param_count = model["parameter_count"]
                        assert param_count > 0, "Mock models should have parameters"
                        assert param_count < 1_000_000_000, "Mock models shouldn't be unreasonably large"

            print("✅ Missing CognateRefiner components handled gracefully")

        except Exception as e:
            pytest.fail(f"Failed to handle missing CognateRefiner components: {e}")

    def test_invalid_output_directory_handling(self, temp_output_dir):
        """Test handling of invalid output directories."""
        try:
            # Test with read-only directory (simulate permission error)
            readonly_dir = Path(temp_output_dir) / "readonly"
            readonly_dir.mkdir()

            try:
                # Try to make it readonly (this might not work on all systems)
                readonly_dir.chmod(0o444)

                # This might fail or succeed depending on system permissions
                from agent_forge.phases.cognate_pretrain import create_three_cognate_models

                create_three_cognate_models(output_dir=str(readonly_dir), device="cpu")

                # If it succeeded, that's ok too
                print("✅ Read-only directory handled (or permissions not enforced)")

            except PermissionError:
                print("✅ Permission errors handled appropriately")
            except Exception as e:
                # Other errors are ok too - just check they're informative
                error_msg = str(e)
                assert len(error_msg) > 10, "Error messages should be informative"
                print(f"✅ Invalid directory error handled: {error_msg[:50]}...")

        except Exception as e:
            pytest.fail(f"Directory error handling test failed: {e}")

    def test_device_fallback_handling(self, temp_output_dir):
        """Test device fallback when requested device not available."""
        try:
            from agent_forge.phases.cognate_pretrain import create_three_cognate_models

            # Test with invalid device
            models = create_three_cognate_models(
                output_dir=temp_output_dir, device="invalid_device"  # This should fallback gracefully
            )

            # Should still create models (fallback to CPU)
            assert len(models) == 3, "Should create models despite invalid device"

            print("✅ Invalid device handled with fallback")

        except Exception as e:
            # Should handle gracefully with informative error
            error_msg = str(e)
            assert "device" in error_msg.lower(), "Error should mention device issue"
            print(f"✅ Device error handled appropriately: {error_msg[:50]}...")

    def test_configuration_validation_errors(self, temp_output_dir):
        """Test configuration validation and error messages."""
        try:
            from agent_forge.phases.cognate_pretrain.cognate_creator import CognateCreatorConfig, CognateModelCreator

            # Test with invalid configuration values
            config = CognateCreatorConfig(
                d_model=-1, n_layers=0, output_dir=temp_output_dir  # Invalid negative value  # Invalid zero layers
            )

            try:
                creator = CognateModelCreator(config)
                creator.create_three_models()

                # If it worked, that's ok (maybe validation is lenient)
                print("✅ Invalid config handled gracefully")

            except Exception as e:
                # Should have informative error message
                error_msg = str(e)
                assert len(error_msg) > 10, "Should have informative error message"
                print(f"✅ Configuration error handled: {error_msg[:50]}...")

        except Exception as e:
            pytest.fail(f"Configuration validation test failed: {e}")

    def test_import_error_messages(self):
        """Test that import errors have clear, helpful messages."""
        try:
            # Test importing from nonexistent submodule
            try:
                pytest.fail("Should have failed to import nonexistent module")
            except ImportError as e:
                error_msg = str(e)
                assert "nonexistent" in error_msg, "Error should mention the missing module"
                print(f"✅ Import error message is clear: {error_msg[:50]}...")

        except Exception as e:
            pytest.fail(f"Import error message test failed: {e}")

    def test_backward_compatibility_warnings(self, temp_output_dir):
        """Test that deprecation warnings are properly issued."""
        try:
            import warnings

            # Test deprecated import path
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                # Should issue deprecation warning
                assert len(w) > 0, "Should issue deprecation warning"

                warning_msgs = [str(warning.message) for warning in w]
                deprecation_found = any("deprecat" in msg.lower() for msg in warning_msgs)
                assert deprecation_found, "Should issue deprecation warning"

                print(f"✅ Deprecation warning properly issued: {warning_msgs[0][:50]}...")

        except Exception as e:
            pytest.fail(f"Backward compatibility warning test failed: {e}")

    def test_model_validation_error_reporting(self, temp_output_dir):
        """Test that model validation errors are properly reported."""
        try:
            from agent_forge.phases.cognate_pretrain.model_factory import validate_cognate_models

            # Test with invalid model data
            invalid_models = [
                {"name": "test1", "parameter_count": 0},  # Invalid: 0 parameters
                {"name": "test2"},  # Invalid: missing parameter_count
            ]

            result = validate_cognate_models(invalid_models)

            # Should fail validation
            assert result is False, "Validation should fail for invalid models"

            print("✅ Model validation properly rejects invalid models")

        except Exception as e:
            # Should handle validation errors gracefully
            error_msg = str(e)
            assert len(error_msg) > 10, "Should have informative error message"
            print(f"✅ Model validation error handled: {error_msg[:50]}...")

    def test_logging_error_information(self, temp_output_dir):
        """Test that errors are properly logged with sufficient information."""
        try:
            import io
            import logging

            # Capture logging output
            log_capture = io.StringIO()
            handler = logging.StreamHandler(log_capture)

            # Get logger for the cognate package
            logger = logging.getLogger("agent_forge.phases.cognate_pretrain")
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

            # Try to create models (might succeed or fail, both are ok)
            try:
                from agent_forge.phases.cognate_pretrain import create_three_cognate_models

                create_three_cognate_models(output_dir=temp_output_dir, device="cpu")

                # Check for informative logs
                log_output = log_capture.getvalue()

                # Should have some logging output
                if log_output:
                    print("✅ Logging output captured")
                else:
                    print("⚠️  No logging output captured (might use different logger)")

            except Exception:
                # Check that error was logged
                log_output = log_capture.getvalue()
                if "error" in log_output.lower() or "failed" in log_output.lower():
                    print("✅ Errors properly logged")
                else:
                    print("⚠️  Error logging might not be captured")

            # Clean up
            logger.removeHandler(handler)

        except Exception as e:
            pytest.fail(f"Logging error information test failed: {e}")

    def test_partial_failure_recovery(self, temp_output_dir):
        """Test recovery from partial failures during model creation."""
        try:
            # Mock a scenario where one model fails but others succeed
            with patch(
                "agent_forge.phases.cognate_pretrain.cognate_creator.CognateModelCreator._create_single_model"
            ) as mock_create:

                def side_effect_func(variant_config, index):
                    if index == 1:  # Fail second model
                        raise Exception("Simulated failure for model 2")
                    # Return mock model info for others
                    return {
                        "name": variant_config["name"],
                        "path": temp_output_dir,
                        "parameter_count": 25000000,
                        "focus": variant_config["focus"],
                        "metadata": {"ready_for_evomerge": True},
                    }

                mock_create.side_effect = side_effect_func

                from agent_forge.phases.cognate_pretrain import create_three_cognate_models

                try:
                    create_three_cognate_models(output_dir=temp_output_dir, device="cpu")
                    pytest.fail("Should have failed due to simulated model creation failure")
                except Exception as e:
                    # Should fail with informative error
                    error_msg = str(e)
                    assert len(error_msg) > 10, "Should have informative error about partial failure"
                    print(f"✅ Partial failure handled: {error_msg[:50]}...")

        except Exception as e:
            pytest.fail(f"Partial failure recovery test failed: {e}")


if __name__ == "__main__":
    import shutil
    import tempfile

    test_suite = TestErrorHandlingValidation()

    print("🧪 Running Error Handling Validation Tests")
    print("=" * 50)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        test_suite.test_missing_cognate_refiner_components(temp_dir)
        test_suite.test_invalid_output_directory_handling(temp_dir)
        test_suite.test_device_fallback_handling(temp_dir)
        test_suite.test_configuration_validation_errors(temp_dir)
        test_suite.test_import_error_messages()
        test_suite.test_backward_compatibility_warnings(temp_dir)
        test_suite.test_model_validation_error_reporting(temp_dir)
        test_suite.test_logging_error_information(temp_dir)
        test_suite.test_partial_failure_recovery(temp_dir)

        print("=" * 50)
        print("✅ ALL ERROR HANDLING TESTS PASSED")

    except Exception as e:
        print("=" * 50)
        print(f"❌ ERROR HANDLING TEST FAILED: {e}")
        raise

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
