"""P2P Tools Package

Utilities for testing, benchmarking, and debugging P2P networks.

Archaeological Enhancement: Comprehensive tooling suite for P2P development.

Tools:
- test_runner: Automated test execution for P2P components
- benchmark: Performance benchmarking and analysis
- network_analyzer: Network topology and performance analysis
- debug_tools: Debugging and diagnostic utilities

Version: 2.0.0
"""

__version__ = "2.0.0"
__all__ = [
    "TestRunner",
    "BenchmarkSuite", 
    "NetworkAnalyzer",
    "DebugTools",
]

# Import tools with graceful fallback
try:
    from .test_runner import TestRunner
except ImportError:
    TestRunner = None

try:
    from .benchmark import BenchmarkSuite
except ImportError:
    BenchmarkSuite = None

try:
    from .network_analyzer import NetworkAnalyzer
except ImportError:
    NetworkAnalyzer = None

try:
    from .debug_tools import DebugTools
except ImportError:
    DebugTools = None