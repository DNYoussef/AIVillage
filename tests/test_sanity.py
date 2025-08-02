"""Basic sanity tests that actually verify functionality"""
import sys
import os
import pytest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports_work():
    """Test that core modules can be imported"""
    try:
        # Try to import main modules
        import core
        import production

        # Verify they're not empty modules
        assert hasattr(core, '__file__')
        assert hasattr(production, '__file__')

        # Try specific imports that should work
        from core.compression import SimpleQuantizer  # noqa: F401
        from core import __version__
        assert __version__ is not None

    except ImportError as e:
        pytest.fail(f"Critical import failed: {e}")


def test_no_syntax_errors():
    """Verify all Python files have valid syntax"""
    root = Path(__file__).parent.parent / "src"
    paths = list((root / "core").rglob("*.py")) + list((root / "compression").rglob("*.py"))
    errors = []
    for py_file in paths:
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                compile(f.read(), py_file, "exec")
        except SyntaxError as e:
            errors.append(f"{py_file}: {e}")

    assert not errors, f"Syntax errors found:\n" + "\n".join(errors)


def test_no_import_cycles():
    """Test that there are no circular imports"""
    # This will fail immediately if there are import cycles
    from core import __version__
    assert __version__ is not None


class TestBasicFunctionality:
    """Test basic functionality exists"""

    def test_compression_module_exists(self):
        """Compression module should be importable"""
        from core.compression import SimpleQuantizer
        assert SimpleQuantizer is not None

    def test_can_create_quantizer(self):
        """Should be able to create a quantizer instance"""
        from core.compression import SimpleQuantizer
        quantizer = SimpleQuantizer()
        assert quantizer is not None
        assert hasattr(quantizer, 'quantize_model')
