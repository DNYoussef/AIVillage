"""
Phase 4 Validation Suite - Main Orchestrator

Coordinates all validation activities for Phase 4 architectural improvements.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import json

from .coupling_analyzer import CouplingAnalyzer
from .performance_monitor import PerformanceMonitor
from .quality_analyzer import QualityAnalyzer
from ..compatibility.backwards_compatibility_tester import BackwardsCompatibilityTester
from ..performance.regression_tester import RegressionTester
from ..integration.service_integration_tester import ServiceIntegrationTester
from ..reports.validation_reporter import ValidationReporter


@dataclass
class ValidationTargets:
    """Phase 4 validation targets"""

    # Coupling score targets
    unified_management_coupling: float = 8.0  # From 21.6
    sage_agent_coupling: float = 25.0  # From 47.46
    task_management_avg_coupling: float = 6.0  # From 9.56

    # Code quality targets
    max_lines_per_class: int = 150
    magic_literals_target: int = 0  # From 159
    min_test_coverage: float = 90.0
    max_cyclomatic_complexity: int = 10

    # Performance targets
    max_memory_increase: float = 10.0  # Percentage
    min_throughput_ratio: float = 1.0  # Maintain or improve
    max_init_time_ms: int = 100
    max_performance_degradation: float = 5.0  # Percentage


@dataclass
class ValidationResult:
    """Results from Phase 4 validation"""

    passed: bool
    coupling_results: Dict[str, Any]
    performance_results: Dict[str, Any]
    quality_results: Dict[str, Any]
    compatibility_results: Dict[str, Any]
    integration_results: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    execution_time_ms: int
    timestamp: str


class Phase4ValidationSuite:
    """
    Main validation orchestrator for Phase 4 architectural improvements
    """

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.targets = ValidationTargets()
        self.logger = logging.getLogger(__name__)

        # Initialize component validators
        self.coupling_analyzer = CouplingAnalyzer(self.project_root)
        self.performance_monitor = PerformanceMonitor()
        self.quality_analyzer = QualityAnalyzer(self.project_root)
        self.compatibility_tester = BackwardsCompatibilityTester(self.project_root)
        self.regression_tester = RegressionTester()
        self.integration_tester = ServiceIntegrationTester()
        self.reporter = ValidationReporter()

        # Validation state
        self.baseline_metrics = {}
        self.current_metrics = {}

    async def initialize(self) -> None:
        """Initialize validation suite and load baseline metrics"""
        self.logger.info("Initializing Phase 4 validation suite...")

        # Load baseline metrics if available
        baseline_path = self.project_root / "tests/validation/phase4/baseline_metrics.json"
        if baseline_path.exists():
            with open(baseline_path, "r") as f:
                self.baseline_metrics = json.load(f)
                self.logger.info("Loaded baseline metrics")
        else:
            self.logger.warning("No baseline metrics found - will create during first run")

    async def run_full_validation(self) -> ValidationResult:
        """
        Run complete Phase 4 validation suite

        Returns:
            ValidationResult containing all validation outcomes
        """
        start_time = time.time()
        errors = []
        warnings = []

        self.logger.info("Starting Phase 4 full validation suite...")

        try:
            # Run all validation phases concurrently where possible
            results = await asyncio.gather(
                self._validate_coupling_improvements(),
                self._validate_performance_benchmarks(),
                self._validate_code_quality(),
                self._validate_backwards_compatibility(),
                self._validate_integration(),
                return_exceptions=True,
            )

            coupling_results, performance_results, quality_results, compatibility_results, integration_results = results

            # Check for exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    errors.append(f"Validation phase {i} failed: {str(result)}")

            # Determine overall pass/fail status
            passed = (
                self._check_coupling_targets(coupling_results)
                and self._check_performance_targets(performance_results)
                and self._check_quality_targets(quality_results)
                and self._check_compatibility_targets(compatibility_results)
                and self._check_integration_targets(integration_results)
                and len(errors) == 0
            )

            execution_time = int((time.time() - start_time) * 1000)

            result = ValidationResult(
                passed=passed,
                coupling_results=coupling_results or {},
                performance_results=performance_results or {},
                quality_results=quality_results or {},
                compatibility_results=compatibility_results or {},
                integration_results=integration_results or {},
                errors=errors,
                warnings=warnings,
                execution_time_ms=execution_time,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

            # Generate validation report
            await self.reporter.generate_report(result, self.targets)

            return result

        except Exception as e:
            self.logger.error(f"Validation suite failed: {e}")
            return ValidationResult(
                passed=False,
                coupling_results={},
                performance_results={},
                quality_results={},
                compatibility_results={},
                integration_results={},
                errors=[str(e)],
                warnings=[],
                execution_time_ms=int((time.time() - start_time) * 1000),
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

    async def _validate_coupling_improvements(self) -> Dict[str, Any]:
        """Validate coupling score improvements meet targets"""
        self.logger.info("Validating coupling improvements...")

        # Analyze current coupling scores
        coupling_results = await self.coupling_analyzer.analyze_all_components()

        # Compare with baseline and targets
        improvements = {
            "unified_management": {
                "current": coupling_results.get("UnifiedManagement", {}).get("coupling_score", 0),
                "target": self.targets.unified_management_coupling,
                "baseline": self.baseline_metrics.get("coupling", {}).get("UnifiedManagement", 21.6),
            },
            "sage_agent": {
                "current": coupling_results.get("SageAgent", {}).get("coupling_score", 0),
                "target": self.targets.sage_agent_coupling,
                "baseline": self.baseline_metrics.get("coupling", {}).get("SageAgent", 47.46),
            },
            "task_management_avg": {
                "current": coupling_results.get("task_management_average", 0),
                "target": self.targets.task_management_avg_coupling,
                "baseline": self.baseline_metrics.get("coupling", {}).get("task_management_average", 9.56),
            },
        }

        return {
            "coupling_scores": coupling_results,
            "improvements": improvements,
            "meets_targets": self._check_coupling_targets(coupling_results),
        }

    async def _validate_performance_benchmarks(self) -> Dict[str, Any]:
        """Validate performance benchmarks are maintained"""
        self.logger.info("Validating performance benchmarks...")

        return await self.regression_tester.run_performance_tests()

    async def _validate_code_quality(self) -> Dict[str, Any]:
        """Validate code quality improvements"""
        self.logger.info("Validating code quality metrics...")

        return await self.quality_analyzer.analyze_quality_metrics()

    async def _validate_backwards_compatibility(self) -> Dict[str, Any]:
        """Validate backwards compatibility is maintained"""
        self.logger.info("Validating backwards compatibility...")

        return await self.compatibility_tester.run_compatibility_tests()

    async def _validate_integration(self) -> Dict[str, Any]:
        """Validate service integration functionality"""
        self.logger.info("Validating service integration...")

        return await self.integration_tester.run_integration_tests()

    def _check_coupling_targets(self, results: Dict[str, Any]) -> bool:
        """Check if coupling improvement targets are met"""
        if not results:
            return False

        coupling_scores = results.get("coupling_scores", {})

        unified_ok = (
            coupling_scores.get("UnifiedManagement", {}).get("coupling_score", float("inf"))
            <= self.targets.unified_management_coupling
        )
        sage_ok = (
            coupling_scores.get("SageAgent", {}).get("coupling_score", float("inf")) <= self.targets.sage_agent_coupling
        )
        task_avg_ok = (
            coupling_scores.get("task_management_average", float("inf")) <= self.targets.task_management_avg_coupling
        )

        return unified_ok and sage_ok and task_avg_ok

    def _check_performance_targets(self, results: Dict[str, Any]) -> bool:
        """Check if performance targets are met"""
        if not results:
            return False

        memory_ok = results.get("memory_increase_percent", float("inf")) <= self.targets.max_memory_increase
        throughput_ok = results.get("throughput_ratio", 0) >= self.targets.min_throughput_ratio
        init_time_ok = results.get("init_time_ms", float("inf")) <= self.targets.max_init_time_ms
        degradation_ok = (
            results.get("performance_degradation_percent", float("inf")) <= self.targets.max_performance_degradation
        )

        return memory_ok and throughput_ok and init_time_ok and degradation_ok

    def _check_quality_targets(self, results: Dict[str, Any]) -> bool:
        """Check if code quality targets are met"""
        if not results:
            return False

        lines_ok = results.get("max_lines_per_class", float("inf")) <= self.targets.max_lines_per_class
        magic_ok = results.get("magic_literals_count", float("inf")) <= self.targets.magic_literals_target
        coverage_ok = results.get("test_coverage_percent", 0) >= self.targets.min_test_coverage
        complexity_ok = results.get("max_cyclomatic_complexity", float("inf")) <= self.targets.max_cyclomatic_complexity

        return lines_ok and magic_ok and coverage_ok and complexity_ok

    def _check_compatibility_targets(self, results: Dict[str, Any]) -> bool:
        """Check if compatibility targets are met"""
        return results.get("all_tests_passed", False) if results else False

    def _check_integration_targets(self, results: Dict[str, Any]) -> bool:
        """Check if integration targets are met"""
        return results.get("all_services_integrated", False) if results else False

    async def save_baseline_metrics(self) -> None:
        """Save current metrics as baseline for future comparisons"""
        self.logger.info("Saving baseline metrics...")

        baseline = {
            "coupling": await self.coupling_analyzer.analyze_all_components(),
            "performance": await self.regression_tester.get_baseline_metrics(),
            "quality": await self.quality_analyzer.analyze_quality_metrics(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        baseline_path = self.project_root / "tests/validation/phase4/baseline_metrics.json"
        baseline_path.parent.mkdir(parents=True, exist_ok=True)

        with open(baseline_path, "w") as f:
            json.dump(baseline, f, indent=2)

        self.logger.info(f"Baseline metrics saved to {baseline_path}")

    async def generate_improvement_report(self) -> str:
        """Generate comprehensive improvement report"""
        validation_result = await self.run_full_validation()
        return await self.reporter.generate_improvement_report(validation_result, self.baseline_metrics)
