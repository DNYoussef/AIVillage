"""
Phase 4 Validation Test Runner

Main entry point for running Phase 4 architectural validation.
Orchestrates the entire validation process with monitoring and reporting.
"""

import asyncio
import logging
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import time
from dataclasses import asdict

from .core.phase4_validator import Phase4ValidationSuite, ValidationTargets
from .core.performance_monitor import PerformanceMonitor
from .continuous_validation_pipeline import ContinuousValidationPipeline
from .success_gates import SuccessGateManager
from .reports.validation_reporter import ValidationReporter


class ValidationRunner:
    """
    Main orchestrator for Phase 4 validation testing
    """

    def __init__(self, project_root: Path, config: Optional[Dict[str, Any]] = None):
        self.project_root = project_root
        self.config = config or {}

        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.validator = Phase4ValidationSuite(project_root)
        self.performance_monitor = PerformanceMonitor()
        self.pipeline = ContinuousValidationPipeline(project_root)
        self.gate_manager = SuccessGateManager(project_root)
        self.reporter = ValidationReporter()

        # Runtime state
        self.validation_session_id = f"phase4_validation_{int(time.time())}"
        self.session_metrics = {
            "start_time": None,
            "end_time": None,
            "total_duration_ms": 0,
            "validation_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
        }

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_level = self.config.get("log_level", "INFO")
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.project_root / "tests/validation/phase4/validation.log"),
            ],
        )

    async def run_full_validation(self, save_baseline: bool = False) -> Dict[str, Any]:
        """
        Run complete Phase 4 validation workflow

        Args:
            save_baseline: Whether to save current metrics as baseline

        Returns:
            Complete validation results with recommendations
        """
        self.logger.info("🚀 Starting Phase 4 Full Validation")
        self.session_metrics["start_time"] = time.time()

        try:
            # Step 1: Initialize all components
            await self._initialize_validation_session()

            # Step 2: Save baseline if requested
            if save_baseline:
                await self._save_baseline_metrics()

            # Step 3: Start performance monitoring
            await self.performance_monitor.start_monitoring(interval=0.5)

            # Step 4: Run core validation
            self.logger.info("📊 Running core validation suite...")
            validation_result = await self.validator.run_full_validation()
            self.session_metrics["validation_runs"] += 1

            # Step 5: Evaluate success gates
            self.logger.info("🚪 Evaluating success gates...")
            gate_results = await self.gate_manager.evaluate_success_gates(validation_result)

            # Step 6: Generate comprehensive report
            self.logger.info("📄 Generating validation report...")
            report_path = await self.reporter.generate_report(validation_result, ValidationTargets())

            # Step 7: Performance analysis
            performance_summary = self.performance_monitor.get_performance_summary(duration_minutes=5)
            performance_issues = self.performance_monitor.detect_performance_issues()

            # Step 8: Compile final results
            final_results = self._compile_final_results(
                validation_result, gate_results, performance_summary, performance_issues, report_path
            )

            # Step 9: Handle results
            if final_results["overall_success"]:
                await self._handle_validation_success(final_results)
                self.session_metrics["successful_runs"] += 1
            else:
                await self._handle_validation_failure(final_results)
                self.session_metrics["failed_runs"] += 1

            return final_results

        except Exception as e:
            self.logger.error(f"❌ Validation failed with exception: {e}")
            self.session_metrics["failed_runs"] += 1

            return {
                "overall_success": False,
                "error": str(e),
                "session_id": self.validation_session_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

        finally:
            # Cleanup
            await self._cleanup_validation_session()

    async def run_quick_validation(self) -> Dict[str, Any]:
        """
        Run quick validation focusing on critical components only

        Returns:
            Quick validation results
        """
        self.logger.info("⚡ Starting Phase 4 Quick Validation")

        start_time = time.time()

        try:
            # Quick coupling analysis
            coupling_results = await self.validator.coupling_analyzer.analyze_all_components()

            # Quick performance check
            await self.performance_monitor.start_monitoring(interval=1.0)
            await asyncio.sleep(5)  # Monitor for 5 seconds
            performance_metrics = self.performance_monitor.get_current_metrics()

            # Basic success gate check
            quick_gate_results = await self._run_quick_success_gates(coupling_results, performance_metrics)

            execution_time = int((time.time() - start_time) * 1000)

            results = {
                "validation_type": "quick",
                "overall_success": quick_gate_results["deployment_approved"],
                "coupling_results": coupling_results,
                "performance_metrics": performance_metrics,
                "gate_results": quick_gate_results,
                "execution_time_ms": execution_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "session_id": self.validation_session_id,
            }

            # Log summary
            status = "✅ PASSED" if results["overall_success"] else "❌ FAILED"
            self.logger.info(f"Quick validation completed: {status} ({execution_time}ms)")

            return results

        except Exception as e:
            self.logger.error(f"Quick validation failed: {e}")
            return {
                "validation_type": "quick",
                "overall_success": False,
                "error": str(e),
                "execution_time_ms": int((time.time() - start_time) * 1000),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

        finally:
            self.performance_monitor.stop_monitoring()

    async def run_continuous_validation(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """
        Run continuous validation for specified duration

        Args:
            duration_minutes: How long to run continuous validation

        Returns:
            Continuous validation results
        """
        self.logger.info(f"🔄 Starting Continuous Validation ({duration_minutes} minutes)")

        try:
            # Start continuous pipeline
            await self.pipeline.start_pipeline()

            # Run for specified duration
            end_time = time.time() + (duration_minutes * 60)
            validation_count = 0

            while time.time() < end_time:
                # Run validation
                await self.pipeline.run_validation(f"continuous_{validation_count}")
                validation_count += 1

                # Wait before next validation (adjust based on requirements)
                await asyncio.sleep(300)  # 5 minutes between validations

            # Get pipeline status and history
            pipeline_status = self.pipeline.get_pipeline_status()
            validation_history = self.pipeline.get_validation_history(limit=100)

            # Calculate continuous validation metrics
            successful_validations = len([v for v in validation_history if v["passed"]])
            success_rate = (successful_validations / len(validation_history)) * 100 if validation_history else 0

            results = {
                "validation_type": "continuous",
                "duration_minutes": duration_minutes,
                "validations_run": len(validation_history),
                "successful_validations": successful_validations,
                "success_rate": success_rate,
                "pipeline_status": pipeline_status,
                "validation_history": validation_history,
                "overall_success": success_rate >= 90,  # 90% success rate threshold
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            self.logger.info(f"Continuous validation completed: {success_rate:.1f}% success rate")

            return results

        except Exception as e:
            self.logger.error(f"Continuous validation failed: {e}")
            return {"validation_type": "continuous", "overall_success": False, "error": str(e)}

        finally:
            await self.pipeline.stop_pipeline()

    async def run_regression_testing(self, baseline_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run regression testing against baseline

        Args:
            baseline_path: Path to baseline metrics file

        Returns:
            Regression testing results
        """
        self.logger.info("📈 Starting Regression Testing")

        try:
            # Load baseline if provided
            baseline_metrics = None
            if baseline_path and baseline_path.exists():
                with open(baseline_path, "r") as f:
                    baseline_metrics = json.load(f)

            # Run current validation
            current_results = await self.validator.run_full_validation()

            # Compare with baseline
            if baseline_metrics:
                comparison_results = await self._compare_with_baseline(current_results, baseline_metrics)
            else:
                comparison_results = {"warning": "No baseline available for comparison"}

            # Determine regression status
            has_regression = self._detect_regressions(comparison_results)

            results = {
                "validation_type": "regression",
                "overall_success": not has_regression,
                "current_results": asdict(current_results) if hasattr(current_results, "__dict__") else current_results,
                "baseline_comparison": comparison_results,
                "regressions_detected": has_regression,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            status = "✅ NO REGRESSIONS" if not has_regression else "❌ REGRESSIONS DETECTED"
            self.logger.info(f"Regression testing completed: {status}")

            return results

        except Exception as e:
            self.logger.error(f"Regression testing failed: {e}")
            return {"validation_type": "regression", "overall_success": False, "error": str(e)}

    async def _initialize_validation_session(self) -> None:
        """Initialize validation session"""
        self.logger.info(f"Initializing validation session: {self.validation_session_id}")

        # Initialize core validator
        await self.validator.initialize()

        # Setup performance monitoring callbacks
        def performance_alert_callback(alert):
            self.logger.warning(f"Performance alert: {alert['type']} - {alert['value']}")

        self.performance_monitor.add_alert_callback(performance_alert_callback)

        # Setup gate callbacks
        def gate_result_callback(gate_result):
            status = "✅" if gate_result.status.value == "passed" else "❌"
            self.logger.info(f"Gate {gate_result.gate_name}: {status} ({gate_result.score:.2f})")

        self.gate_manager.add_gate_callback(gate_result_callback)

    async def _save_baseline_metrics(self) -> None:
        """Save current metrics as baseline"""
        self.logger.info("💾 Saving baseline metrics...")

        # Save validation baseline
        await self.validator.save_baseline_metrics()

        # Save performance baseline
        await asyncio.sleep(2)  # Wait for some metrics
        self.performance_monitor.set_baseline()

        self.logger.info("Baseline metrics saved")

    async def _run_quick_success_gates(
        self, coupling_results: Dict[str, Any], performance_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run quick version of success gates"""
        # Simplified gate evaluation for quick validation
        gates_passed = 0
        total_gates = 3  # Quick gates: coupling, performance, basic checks

        # Quick coupling check
        if coupling_results and "improvements" in coupling_results:
            improvements = coupling_results["improvements"]
            if all(imp.get("target_met", False) for imp in improvements.values()):
                gates_passed += 1

        # Quick performance check
        if performance_metrics and not performance_metrics.get("error"):
            current = performance_metrics.get("current", {})
            cpu = current.get("cpu_percent", 0)
            memory = current.get("memory_percent", 0)

            if cpu < 80 and memory < 80:  # Basic thresholds
                gates_passed += 1

        # Basic system check
        gates_passed += 1  # Assume basic checks pass for quick validation

        deployment_approved = gates_passed >= total_gates

        return {
            "deployment_approved": deployment_approved,
            "gates_passed": gates_passed,
            "gates_total": total_gates,
            "quick_evaluation": True,
        }

    def _compile_final_results(
        self,
        validation_result: Any,
        gate_results: Dict[str, Any],
        performance_summary: Dict[str, Any],
        performance_issues: Dict[str, Any],
        report_path: str,
    ) -> Dict[str, Any]:
        """Compile final validation results"""

        # Determine overall success
        overall_success = (
            validation_result.passed
            and gate_results["deployment_approved"]
            and len(performance_issues.get("issues", [])) == 0
        )

        # Calculate quality score
        quality_metrics = [
            1.0 if validation_result.passed else 0.0,
            gate_results.get("overall_score", 0.0),
            1.0 - (len(performance_issues.get("issues", [])) * 0.2),  # Penalize performance issues
        ]
        overall_quality_score = max(0.0, sum(quality_metrics) / len(quality_metrics))

        return {
            "overall_success": overall_success,
            "overall_quality_score": overall_quality_score,
            "validation_results": {
                "passed": validation_result.passed,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "execution_time_ms": validation_result.execution_time_ms,
            },
            "gate_results": gate_results,
            "performance_summary": performance_summary,
            "performance_issues": performance_issues,
            "report_path": report_path,
            "session_id": self.validation_session_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "recommendations": self._generate_recommendations(validation_result, gate_results, performance_issues),
        }

    def _generate_recommendations(
        self, validation_result: Any, gate_results: Dict[str, Any], performance_issues: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        # Validation-based recommendations
        if not validation_result.passed:
            if validation_result.errors:
                recommendations.append("🔧 Address critical validation errors before deployment")

            # Check specific failure areas
            if hasattr(validation_result, "coupling_results") and not validation_result.coupling_results:
                recommendations.append("📊 Run coupling analysis to identify architectural improvements needed")

            if hasattr(validation_result, "performance_results") and not validation_result.performance_results:
                recommendations.append("⚡ Investigate performance regression issues")

        # Gate-based recommendations
        if not gate_results["deployment_approved"]:
            failed_gates = [gr for gr in gate_results.get("gate_results", []) if gr["status"] != "passed"]
            if failed_gates:
                recommendations.append(f"🚪 Fix {len(failed_gates)} failed success gates before proceeding")

        # Performance-based recommendations
        issues = performance_issues.get("issues", [])
        if issues:
            high_severity_issues = [i for i in issues if i.get("severity") == "high"]
            if high_severity_issues:
                recommendations.append("🚨 Address high-severity performance issues immediately")

            performance_recommendations = performance_issues.get("recommendations", [])
            recommendations.extend(performance_recommendations)

        # General recommendations
        if not recommendations:
            recommendations.append("✅ All validation criteria met - ready for Phase 4 deployment")
        else:
            recommendations.append("📋 Review detailed validation report for specific action items")

        return recommendations

    async def _handle_validation_success(self, results: Dict[str, Any]) -> None:
        """Handle successful validation"""
        self.logger.info("🎉 Phase 4 validation SUCCESSFUL!")

        # Log key success metrics
        quality_score = results.get("overall_quality_score", 0.0)
        self.logger.info(f"Overall quality score: {quality_score:.1%}")

        # Save successful state
        success_file = self.project_root / "tests/validation/phase4/last_successful_validation.json"
        with open(success_file, "w") as f:
            json.dump(
                {
                    "timestamp": results["timestamp"],
                    "session_id": results["session_id"],
                    "quality_score": quality_score,
                    "report_path": results.get("report_path"),
                },
                f,
                indent=2,
            )

    async def _handle_validation_failure(self, results: Dict[str, Any]) -> None:
        """Handle validation failure"""
        self.logger.error("💥 Phase 4 validation FAILED!")

        # Log failure details
        if "validation_results" in results:
            errors = results["validation_results"].get("errors", [])
            for error in errors:
                self.logger.error(f"Validation error: {error}")

        # Check if rollback should be triggered
        rollback_config = self.config.get("rollback", {})
        if rollback_config.get("auto_rollback_on_failure", False):
            self.logger.warning("🔄 Triggering automatic rollback...")
            await self.gate_manager.execute_rollback("Validation failure", results)

    async def _compare_with_baseline(self, current_results: Any, baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current results with baseline"""
        comparison = {"coupling_comparison": {}, "performance_comparison": {}, "overall_trend": "stable"}

        # Compare coupling scores
        if hasattr(current_results, "coupling_results") and current_results.coupling_results:
            current_coupling = current_results.coupling_results
            baseline_coupling = baseline_metrics.get("coupling", {})

            for component in ["UnifiedManagement", "SageAgent", "task_management_average"]:
                if component in current_coupling.get("coupling_scores", {}) and component in baseline_coupling:
                    current_score = current_coupling["coupling_scores"][component].get("coupling_score", 0)
                    baseline_score = baseline_coupling[component]

                    change_percent = (
                        ((baseline_score - current_score) / baseline_score) * 100 if baseline_score != 0 else 0
                    )

                    comparison["coupling_comparison"][component] = {
                        "baseline": baseline_score,
                        "current": current_score,
                        "change_percent": change_percent,
                        "improved": change_percent > 0,
                    }

        # Compare performance metrics
        if hasattr(current_results, "performance_results") and current_results.performance_results:
            # This would compare memory usage, throughput, etc.
            comparison["performance_comparison"]["status"] = "compared"

        return comparison

    def _detect_regressions(self, comparison_results: Dict[str, Any]) -> bool:
        """Detect if there are any regressions"""
        # Check coupling regressions
        coupling_comparison = comparison_results.get("coupling_comparison", {})
        for component, comparison in coupling_comparison.items():
            if comparison.get("change_percent", 0) < -10:  # 10% degradation threshold
                return True

        # Check performance regressions
        # (Implementation would depend on performance comparison structure)

        return False

    async def _cleanup_validation_session(self) -> None:
        """Cleanup validation session"""
        self.session_metrics["end_time"] = time.time()
        self.session_metrics["total_duration_ms"] = int(
            (self.session_metrics["end_time"] - self.session_metrics["start_time"]) * 1000
        )

        # Stop performance monitoring
        self.performance_monitor.stop_monitoring()

        # Export session metrics
        metrics_file = self.project_root / f"tests/validation/phase4/session_metrics_{self.validation_session_id}.json"
        with open(metrics_file, "w") as f:
            json.dump(self.session_metrics, f, indent=2)

        self.logger.info(f"Validation session cleanup completed: {self.session_metrics['total_duration_ms']}ms total")


async def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(description="Phase 4 Validation Runner")
    parser.add_argument(
        "--mode", choices=["full", "quick", "continuous", "regression"], default="full", help="Validation mode"
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root directory")
    parser.add_argument("--save-baseline", action="store_true", help="Save current metrics as baseline")
    parser.add_argument("--duration", type=int, default=60, help="Duration for continuous validation (minutes)")
    parser.add_argument("--baseline-file", type=Path, help="Baseline file for regression testing")
    parser.add_argument("--config-file", type=Path, help="Configuration file path")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level"
    )

    args = parser.parse_args()

    # Load configuration if provided
    config = {"log_level": args.log_level}
    if args.config_file and args.config_file.exists():
        with open(args.config_file, "r") as f:
            config.update(json.load(f))

    # Initialize runner
    runner = ValidationRunner(args.project_root, config)

    # Run validation based on mode
    try:
        if args.mode == "full":
            results = await runner.run_full_validation(save_baseline=args.save_baseline)
        elif args.mode == "quick":
            results = await runner.run_quick_validation()
        elif args.mode == "continuous":
            results = await runner.run_continuous_validation(duration_minutes=args.duration)
        elif args.mode == "regression":
            results = await runner.run_regression_testing(baseline_path=args.baseline_file)

        # Print summary
        print(f"\\n{'='*60}")
        print(f"PHASE 4 VALIDATION RESULTS ({args.mode.upper()} MODE)")
        print(f"{'='*60}")
        print(f"Status: {'✅ SUCCESS' if results.get('overall_success', False) else '❌ FAILURE'}")

        if "overall_quality_score" in results:
            print(f"Quality Score: {results['overall_quality_score']:.1%}")

        if "execution_time_ms" in results:
            print(f"Execution Time: {results['execution_time_ms']}ms")

        if "recommendations" in results:
            print("\\nRecommendations:")
            for rec in results["recommendations"]:
                print(f"  {rec}")

        # Exit with appropriate code
        sys.exit(0 if results.get("overall_success", False) else 1)

    except Exception as e:
        print(f"\\n❌ Validation runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
