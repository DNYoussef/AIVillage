"""Production benchmarking components organized by Sprint 2."""

# Export main benchmarking classes
try:
    from .real_benchmark import RealBenchmark

    __all__ = ['RealBenchmark']
except ImportError:
    # Handle missing dependencies gracefully
    RealBenchmark = None
    __all__ = []
