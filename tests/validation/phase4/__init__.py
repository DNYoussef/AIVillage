"""
Phase 4 Architectural Validation Framework

Comprehensive validation suite for Phase 4 architectural improvements ensuring:
- Coupling score reductions meet targets
- Performance benchmarks are maintained
- Backwards compatibility is preserved
- Code quality metrics improve
"""

from .core.phase4_validator import Phase4ValidationSuite
from .core.coupling_analyzer import CouplingAnalyzer
from .core.performance_monitor import PerformanceMonitor
from .core.quality_analyzer import QualityAnalyzer
from .compatibility.backwards_compatibility_tester import BackwardsCompatibilityTester
from .performance.regression_tester import RegressionTester
from .integration.service_integration_tester import ServiceIntegrationTester
from .reports.validation_reporter import ValidationReporter

__all__ = [
    'Phase4ValidationSuite',
    'CouplingAnalyzer',
    'PerformanceMonitor',
    'QualityAnalyzer',
    'BackwardsCompatibilityTester',
    'RegressionTester',
    'ServiceIntegrationTester',
    'ValidationReporter'
]