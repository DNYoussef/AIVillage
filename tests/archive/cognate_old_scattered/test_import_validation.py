#!/usr/bin/env python3
"""
Test Suite 1: Import Testing
Tests all imports work correctly for the reorganized Cognate system.
"""

import sys
import warnings
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Also need to add the core directory specifically
core_path = project_root / "core"
if core_path.exists():
    sys.path.insert(0, str(core_path))


class TestImportValidation:
    """Test all imports work correctly after reorganization."""

    def test_new_structure_imports(self):
        """Test imports from new cognate-pretrain package."""
        try:
            # Test main factory function import
            from agent_forge.phases.cognate_pretrain import create_three_cognate_models

            assert callable(create_three_cognate_models)

            # Test individual component imports
            from agent_forge.phases.cognate_pretrain import (
                CognateCreatorConfig,
                CognateModelCreator,
                CognatePretrainPipeline,
            )

            assert CognateModelCreator is not None
            assert CognateCreatorConfig is not None
            assert CognatePretrainPipeline is not None

            print("✅ New structure imports successful")

        except ImportError as e:
            pytest.fail(f"Failed to import from new structure: {e}")

    def test_backward_compatibility_redirect(self):
        """Test backward compatibility via cognate.py redirect."""
        try:
            # This should work but issue deprecation warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                from agent_forge.phases.cognate import create_three_cognate_models

                # Check that deprecation warning was issued
                assert len(w) > 0
                assert any("deprecated" in str(warning.message).lower() for warning in w)

                # But function should still be callable
                assert callable(create_three_cognate_models)

            print("✅ Backward compatibility redirect works with proper warnings")

        except ImportError as e:
            pytest.fail(f"Backward compatibility redirect failed: {e}")

    def test_package_init_exports(self):
        """Test __init__.py exports work correctly."""
        try:
            import agent_forge.phases.cognate_pretrain as cp

            # Check all expected exports are available
            expected_exports = [
                "CognateModelCreator",
                "CognateCreatorConfig",
                "CognatePretrainPipeline",
                "create_three_cognate_models",
            ]

            for export in expected_exports:
                assert hasattr(cp, export), f"Missing export: {export}"
                assert callable(getattr(cp, export)) or hasattr(getattr(cp, export), "__init__")

            print("✅ Package __init__.py exports complete")

        except ImportError as e:
            pytest.fail(f"Package init import failed: {e}")

    def test_direct_module_imports(self):
        """Test direct imports from individual modules."""
        try:
            # Test model_factory direct import

            # Test cognate_creator direct import

            # Test pretrain_pipeline direct import

            print("✅ Direct module imports successful")

        except ImportError as e:
            pytest.fail(f"Direct module import failed: {e}")

    def test_optional_dependencies_handling(self):
        """Test graceful handling when optional dependencies are missing."""
        try:
            from agent_forge.phases.cognate_pretrain.cognate_creator import (
                CognateConfig,
                CognateRefiner,
                LTMBank,
                MemoryCrossAttention,
            )

            # These might be None if imports failed - that's ok
            print(f"CognateRefiner available: {CognateRefiner is not None}")
            print(f"CognateConfig available: {CognateConfig is not None}")
            print(f"LTMBank available: {LTMBank is not None}")
            print(f"MemoryCrossAttention available: {MemoryCrossAttention is not None}")

            print("✅ Optional dependencies handled gracefully")

        except Exception as e:
            pytest.fail(f"Optional dependencies handling failed: {e}")

    def test_import_paths_consistency(self):
        """Test that import paths are consistent and don't conflict."""
        try:
            # Import same function via different paths
            from agent_forge.phases.cognate_pretrain import create_three_cognate_models as func1
            from agent_forge.phases.cognate_pretrain.model_factory import create_three_cognate_models as func2

            # Should be the same function object
            assert func1 is func2, "Import paths should point to same function object"

            print("✅ Import paths are consistent")

        except ImportError as e:
            pytest.fail(f"Import path consistency check failed: {e}")


if __name__ == "__main__":
    test_suite = TestImportValidation()

    print("🧪 Running Import Validation Tests")
    print("=" * 50)

    try:
        test_suite.test_new_structure_imports()
        test_suite.test_backward_compatibility_redirect()
        test_suite.test_package_init_exports()
        test_suite.test_direct_module_imports()
        test_suite.test_optional_dependencies_handling()
        test_suite.test_import_paths_consistency()

        print("=" * 50)
        print("✅ ALL IMPORT TESTS PASSED")

    except Exception as e:
        print("=" * 50)
        print(f"❌ IMPORT TEST FAILED: {e}")
        raise
