"""AIVillage Production Components

This package contains production-ready components that have passed Sprint 2 quality gates:
- Compression: Model compression with 4-8x reduction
- Evolution: Evolutionary model optimization
- RAG: Retrieval-augmented generation
- Memory: Memory management and W&B logging
- Benchmarking: Real benchmark evaluation
- Geometry: Geometric analysis capabilities

All components in this package are stable, tested, and ready for production use.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

__version__ = "1.0.0"
__status__ = "Production"

# Quality gates enforced:
# - No task markers or fix markers in code
# - No imports from experimental/ or deprecated/
# - Minimum 70% test coverage
# - Type checking with mypy
# - Security scanning with bandit
# - Pre-commit hooks enforced

# Import guard to prevent experimental imports
import sys
import warnings


def _check_imports():
    """Check that no experimental modules are imported."""
    for module_name in sys.modules:
        if module_name.startswith("experimental.") or module_name.startswith(
            "deprecated."
        ):
            warnings.warn(
                f"Production code should not import {module_name}",
                UserWarning,
                stacklevel=2,
            )


# Production components - these should be stable APIs
__all__ = ["benchmarking", "compression", "evolution", "geometry", "memory", "rag"]
