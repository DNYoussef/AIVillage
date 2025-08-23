"""
AIVillage Load Testing Infrastructure
====================================

Comprehensive load testing, soak testing, and performance regression detection
for production validation of AIVillage systems.

Main Components:
- ProductionLoadTestSuite: High-throughput load testing with multiple profiles
- SoakTestOrchestrator: Long-running stability testing with chaos injection
- PerformanceRegressionDetector: Statistical performance comparison and baseline management
- IntegratedLoadTestRunner: Master orchestrator for complete test suites

Usage Examples:
    # Quick validation
    python -m tests.load_testing.integrated_load_test_runner --quick-validation

    # Full production readiness test
    python -m tests.load_testing.integrated_load_test_runner --production-readiness

    # 24-hour soak test
    python -m tests.load_testing.soak_test_orchestrator --duration 24 --enable-chaos

    # Performance regression detection
    python -m tests.load_testing.performance_regression_detector --baseline
    python -m tests.load_testing.performance_regression_detector --compare
"""

from .integrated_load_test_runner import (
    AlertManager,
    IntegratedLoadTestRunner,
    IntegratedTestReport,
    ReportGenerator,
    SystemHealthChecker,
    TestResult,
    TestSuiteConfig,
)
from .performance_regression_detector import (
    PerformanceBenchmark,
    PerformanceBenchmarkRunner,
    PerformanceMetric,
    PerformanceRegressionDetector,
    RegressionReport,
    RegressionResult,
    StatisticalAnalysis,
)
from .production_load_test_suite import LoadTestConfig, ProductionLoadTestSuite, TestMetrics, create_test_profiles
from .soak_test_orchestrator import (
    ChaosTestingEngine,
    MemoryLeakDetector,
    PerformanceDegradationDetector,
    SoakTestConfig,
    SoakTestMetrics,
    SoakTestOrchestrator,
)

__version__ = "1.0.0"
__author__ = "AIVillage Team"

__all__ = [
    # Main test runners
    "ProductionLoadTestSuite",
    "SoakTestOrchestrator",
    "PerformanceRegressionDetector",
    "IntegratedLoadTestRunner",
    # Configuration classes
    "LoadTestConfig",
    "SoakTestConfig",
    "TestSuiteConfig",
    # Result classes
    "TestMetrics",
    "SoakTestMetrics",
    "PerformanceMetric",
    "PerformanceBenchmark",
    "RegressionResult",
    "RegressionReport",
    "TestResult",
    "IntegratedTestReport",
    # Utility classes
    "MemoryLeakDetector",
    "PerformanceDegradationDetector",
    "ChaosTestingEngine",
    "StatisticalAnalysis",
    "SystemHealthChecker",
    "AlertManager",
    "ReportGenerator",
    "PerformanceBenchmarkRunner",
    # Helper functions
    "create_test_profiles",
]

# Package metadata
PACKAGE_INFO = {
    "name": "aivillage-load-testing",
    "version": __version__,
    "description": "Comprehensive load testing infrastructure for AIVillage production validation",
    "features": [
        "Production load testing with multiple profiles",
        "Long-running soak testing with stability monitoring",
        "Performance regression detection with statistical analysis",
        "Chaos testing for resilience validation",
        "Memory leak detection and performance degradation monitoring",
        "Automated reporting with HTML, JSON, and JUnit XML formats",
        "CI/CD integration with webhooks and Slack notifications",
        "Resource monitoring and threshold validation",
    ],
    "requirements": [
        "asyncio (built-in)",
        "psutil (optional, for resource monitoring)",
        "aiohttp (optional, for async HTTP)",
        "matplotlib (optional, for plotting)",
        "numpy (optional, for statistical analysis)",
    ],
}
