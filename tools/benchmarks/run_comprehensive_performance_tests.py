#!/usr/bin/env python3
"""
Comprehensive Performance Test Runner

Orchestrates all performance benchmarks, load tests, and baseline measurements
to provide a complete assessment of AIVillage system performance with real data.

This replaces the mock-based performance validation with actual system testing.
"""

import asyncio
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.benchmarks.performance_benchmarker import PerformanceBenchmarker
from tools.benchmarks.load_test_orchestrator import LoadTestOrchestrator
from tools.benchmarks.baseline_performance_suite import BaselinePerformanceSuite

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensivePerformanceTestRunner:
    """Orchestrates all performance testing activities."""
    
    def __init__(self, output_dir: str = "tools/benchmarks/results"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Test results storage
        self.benchmark_results = None
        self.load_test_results = None
        self.baseline_results = None
        
        # Overall test status
        self.test_start_time = None
        self.test_end_time = None
        self.tests_completed = []
        self.tests_failed = []
        
    async def run_performance_benchmarks(self) -> bool:
        """Run comprehensive performance benchmarks."""
        logger.info("=== Starting Performance Benchmarking ===")
        
        try:
            benchmarker = PerformanceBenchmarker()
            
            # Run comprehensive benchmarks
            benchmark_metrics = await benchmarker.run_comprehensive_benchmark()
            
            if not benchmark_metrics:
                logger.error("No benchmark metrics collected")
                self.tests_failed.append("performance_benchmarks")
                return False
            
            # Generate and save report
            self.benchmark_results = benchmarker.generate_performance_report()
            
            benchmark_report_path = os.path.join(
                self.output_dir, 
                f"performance_benchmark_{self.timestamp}.json"
            )
            benchmarker.save_report(benchmark_report_path)
            
            logger.info(f"Performance benchmarks completed: {len(benchmark_metrics)} tests")
            self.tests_completed.append("performance_benchmarks")
            return True
            
        except Exception as e:
            logger.error(f"Performance benchmarking failed: {e}")
            logger.error(traceback.format_exc())
            self.tests_failed.append("performance_benchmarks")
            return False
            
    async def run_load_tests(self) -> bool:
        """Run comprehensive load testing."""
        logger.info("=== Starting Load Testing ===")
        
        try:
            orchestrator = LoadTestOrchestrator()
            
            # Run comprehensive load tests
            load_results = await orchestrator.run_comprehensive_load_tests()
            
            if not load_results:
                logger.error("No load test results collected")
                self.tests_failed.append("load_tests")
                return False
                
            # Generate and save report
            self.load_test_results = orchestrator.generate_load_test_report()
            
            load_report_path = os.path.join(
                self.output_dir,
                f"load_test_report_{self.timestamp}.json"
            )
            orchestrator.save_load_test_report(load_report_path)
            
            logger.info(f"Load testing completed: {len(load_results)} scenarios")
            self.tests_completed.append("load_tests")
            return True
            
        except Exception as e:
            logger.error(f"Load testing failed: {e}")
            logger.error(traceback.format_exc())
            self.tests_failed.append("load_tests")
            return False
            
    async def run_baseline_establishment(self) -> bool:
        """Establish performance baselines."""
        logger.info("=== Establishing Performance Baselines ===")
        
        try:
            baseline_suite = BaselinePerformanceSuite()
            
            # Establish baselines for all components
            baselines = await baseline_suite.establish_all_baselines()
            
            if not baselines:
                logger.error("No baselines established")
                self.tests_failed.append("baseline_establishment")
                return False
                
            # Generate report
            self.baseline_results = baseline_suite.generate_baseline_report()
            
            baseline_report_path = os.path.join(
                self.output_dir,
                f"baseline_report_{self.timestamp}.json"
            )
            
            with open(baseline_report_path, 'w') as f:
                json.dump(self.baseline_results, f, indent=2)
                
            logger.info(f"Baseline establishment completed: {len(baselines)} components")
            self.tests_completed.append("baseline_establishment")
            return True
            
        except Exception as e:
            logger.error(f"Baseline establishment failed: {e}")
            logger.error(traceback.format_exc())
            self.tests_failed.append("baseline_establishment")
            return False
            
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance assessment report."""
        
        # Overall test execution summary
        total_tests = len(self.tests_completed) + len(self.tests_failed)
        success_rate = len(self.tests_completed) / total_tests if total_tests > 0 else 0
        
        test_duration = (self.test_end_time - self.test_start_time) if self.test_start_time and self.test_end_time else 0
        
        # Aggregate performance metrics
        performance_summary = self._aggregate_performance_metrics()
        
        # System health assessment
        health_assessment = self._assess_system_health()
        
        # Production readiness evaluation
        production_readiness = self._evaluate_production_readiness()
        
        return {
            "comprehensive_performance_report": {
                "timestamp": datetime.now().isoformat(),
                "test_execution": {
                    "duration_seconds": test_duration,
                    "tests_completed": self.tests_completed,
                    "tests_failed": self.tests_failed,
                    "success_rate": success_rate,
                    "overall_status": "SUCCESS" if len(self.tests_failed) == 0 else "PARTIAL" if len(self.tests_completed) > 0 else "FAILED"
                },
                "performance_summary": performance_summary,
                "system_health_assessment": health_assessment,
                "production_readiness": production_readiness,
                "detailed_results": {
                    "benchmark_results": self.benchmark_results,
                    "load_test_results": self.load_test_results,
                    "baseline_results": self.baseline_results
                }
            }
        }
        
    def _aggregate_performance_metrics(self) -> Dict[str, Any]:
        """Aggregate performance metrics across all tests."""
        metrics = {
            "components_tested": 0,
            "total_operations": 0,
            "overall_throughput": 0.0,
            "average_latency_ms": 0.0,
            "average_success_rate": 0.0,
            "performance_issues": []
        }
        
        # Aggregate benchmark results
        if self.benchmark_results:
            benchmark_summary = self.benchmark_results.get("benchmark_summary", {})
            metrics["components_tested"] += len(self.benchmark_results.get("component_performance", {}))
            metrics["total_operations"] += benchmark_summary.get("total_items_processed", 0)
            
            # Get throughput from benchmark summary
            if benchmark_summary.get("overall_throughput", 0) > 0:
                metrics["overall_throughput"] = benchmark_summary["overall_throughput"]
                
            # Aggregate latency and success rates from detailed results
            detailed_results = self.benchmark_results.get("detailed_results", [])
            if detailed_results:
                latencies = [r["latency_avg_ms"] for r in detailed_results]
                success_rates = [r["success_rate"] for r in detailed_results]
                
                metrics["average_latency_ms"] = sum(latencies) / len(latencies)
                metrics["average_success_rate"] = sum(success_rates) / len(success_rates)
                
            # Check for performance issues
            bottlenecks = self.benchmark_results.get("bottlenecks_identified", [])
            metrics["performance_issues"].extend([
                f"Benchmark: {b['type']} in {b['component']}" for b in bottlenecks
            ])
        
        # Aggregate load test results
        if self.load_test_results:
            load_summary = self.load_test_results.get("load_test_summary", {})
            
            # Add load test issues
            detailed_load_results = self.load_test_results.get("detailed_results", [])
            for result in detailed_load_results:
                if not result.get("passed", True):
                    metrics["performance_issues"].append(
                        f"Load Test: {result['scenario']} failed - {result.get('breaking_point_reason', 'Unknown')}"
                    )
        
        return metrics
        
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health based on test results."""
        
        health_indicators = {
            "functional_components": 0,
            "total_components": 4,  # P2P, Agent, Gateway, Twin
            "critical_issues": [],
            "performance_concerns": [],
            "resource_concerns": []
        }
        
        # Analyze benchmark results for health indicators
        if self.benchmark_results:
            component_performance = self.benchmark_results.get("component_performance", {})
            health_indicators["functional_components"] = len(component_performance)
            
            # Check for critical performance issues
            for component, stats in component_performance.items():
                if stats.get("avg_success_rate", 0) < 0.8:
                    health_indicators["critical_issues"].append(
                        f"{component}: Low success rate ({stats['avg_success_rate']:.1%})"
                    )
                    
                if stats.get("avg_latency_ms", 0) > 2000:
                    health_indicators["performance_concerns"].append(
                        f"{component}: High latency ({stats['avg_latency_ms']:.0f}ms)"
                    )
        
        # Analyze load test results for scalability concerns
        if self.load_test_results:
            detailed_results = self.load_test_results.get("detailed_results", [])
            for result in detailed_results:
                if result.get("breaking_point_load") and result["breaking_point_load"] < 100:
                    health_indicators["critical_issues"].append(
                        f"{result['component']}: Low breaking point ({result['breaking_point_load']})"
                    )
        
        # Overall health assessment
        functional_ratio = health_indicators["functional_components"] / health_indicators["total_components"]
        critical_count = len(health_indicators["critical_issues"])
        
        if functional_ratio >= 0.75 and critical_count == 0:
            overall_health = "HEALTHY"
        elif functional_ratio >= 0.5 and critical_count <= 2:
            overall_health = "DEGRADED"
        else:
            overall_health = "CRITICAL"
            
        health_indicators["overall_health"] = overall_health
        
        return health_indicators
        
    def _evaluate_production_readiness(self) -> Dict[str, Any]:
        """Evaluate production readiness based on all test results."""
        
        readiness_criteria = {
            "performance_benchmarks_passed": "performance_benchmarks" in self.tests_completed,
            "load_tests_passed": "load_tests" in self.tests_completed, 
            "baselines_established": "baseline_establishment" in self.tests_completed,
            "system_health_acceptable": False,
            "scalability_validated": False,
            "no_critical_issues": False
        }
        
        # Check system health
        health_assessment = self._assess_system_health()
        readiness_criteria["system_health_acceptable"] = health_assessment["overall_health"] in ["HEALTHY", "DEGRADED"]
        readiness_criteria["no_critical_issues"] = len(health_assessment["critical_issues"]) == 0
        
        # Check scalability from load tests
        if self.load_test_results:
            load_summary = self.load_test_results.get("load_test_summary", {})
            readiness_criteria["scalability_validated"] = load_summary.get("system_resilience") in ["HIGH", "MEDIUM"]
        
        # Calculate readiness score
        passed_criteria = sum(1 for v in readiness_criteria.values() if v)
        total_criteria = len(readiness_criteria)
        readiness_score = passed_criteria / total_criteria
        
        # Determine overall readiness
        if readiness_score >= 0.85:
            overall_readiness = "PRODUCTION_READY"
            recommendation = "System meets production readiness criteria"
        elif readiness_score >= 0.70:
            overall_readiness = "CONDITIONALLY_READY"  
            recommendation = "Address remaining issues before full production deployment"
        elif readiness_score >= 0.50:
            overall_readiness = "NEEDS_IMPROVEMENT"
            recommendation = "Significant improvements needed before production"
        else:
            overall_readiness = "NOT_READY"
            recommendation = "System requires major fixes before production deployment"
        
        return {
            "overall_readiness": overall_readiness,
            "readiness_score": readiness_score,
            "criteria_met": readiness_criteria,
            "recommendation": recommendation,
            "blocking_issues": [
                criterion for criterion, passed in readiness_criteria.items() 
                if not passed
            ]
        }
        
    def save_comprehensive_report(self, filepath: str = None):
        """Save comprehensive performance report."""
        if filepath is None:
            filepath = os.path.join(
                self.output_dir,
                f"comprehensive_performance_report_{self.timestamp}.json"
            )
            
        report = self.generate_comprehensive_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Also save human-readable summary
        summary_path = filepath.replace('.json', '_executive_summary.txt')
        self._save_executive_summary(summary_path, report)
        
        logger.info(f"Comprehensive report saved to {filepath}")
        logger.info(f"Executive summary saved to {summary_path}")
        
        return filepath
        
    def _save_executive_summary(self, filepath: str, report: Dict[str, Any]):
        """Save executive summary in human-readable format."""
        
        main_report = report["comprehensive_performance_report"]
        
        with open(filepath, 'w') as f:
            f.write("AIVillage Performance Assessment - Executive Summary\n")
            f.write("=" * 60 + "\n\n")
            
            # Test execution summary
            execution = main_report["test_execution"]
            f.write(f"Test Execution Summary:\n")
            f.write(f"Duration: {execution['duration_seconds']:.1f} seconds\n")
            f.write(f"Tests Completed: {len(execution['tests_completed'])}\n")
            f.write(f"Tests Failed: {len(execution['tests_failed'])}\n")
            f.write(f"Overall Status: {execution['overall_status']}\n\n")
            
            # Performance summary
            performance = main_report["performance_summary"]
            f.write(f"Performance Summary:\n")
            f.write(f"Components Tested: {performance['components_tested']}\n")
            f.write(f"Total Operations: {performance['total_operations']}\n")
            f.write(f"Overall Throughput: {performance['overall_throughput']:.2f} ops/sec\n")
            f.write(f"Average Latency: {performance['average_latency_ms']:.0f}ms\n")
            f.write(f"Average Success Rate: {performance['average_success_rate']:.1%}\n\n")
            
            # System health
            health = main_report["system_health_assessment"]
            f.write(f"System Health: {health['overall_health']}\n")
            f.write(f"Functional Components: {health['functional_components']}/{health['total_components']}\n")
            
            if health["critical_issues"]:
                f.write(f"Critical Issues ({len(health['critical_issues'])}):\n")
                for issue in health["critical_issues"]:
                    f.write(f"  - {issue}\n")
                f.write("\n")
            
            # Production readiness
            readiness = main_report["production_readiness"]
            f.write(f"Production Readiness: {readiness['overall_readiness']}\n")
            f.write(f"Readiness Score: {readiness['readiness_score']:.1%}\n")
            f.write(f"Recommendation: {readiness['recommendation']}\n")
            
            if readiness["blocking_issues"]:
                f.write(f"\nBlocking Issues:\n")
                for issue in readiness["blocking_issues"]:
                    f.write(f"  - {issue.replace('_', ' ').title()}\n")
                    
            # Performance issues
            if performance["performance_issues"]:
                f.write(f"\nPerformance Issues ({len(performance['performance_issues'])}):\n")
                for issue in performance["performance_issues"]:
                    f.write(f"  - {issue}\n")


async def main():
    """Main entry point for comprehensive performance testing."""
    print("=" * 60)
    print("AIVillage Comprehensive Performance Testing")
    print("=" * 60)
    print("Replacing mock performance data with real system measurements")
    print()
    
    runner = ComprehensivePerformanceTestRunner()
    runner.test_start_time = datetime.now().timestamp()
    
    try:
        # Run all performance tests
        print("Phase 1: Performance Benchmarking")
        benchmark_success = await runner.run_performance_benchmarks()
        
        print("\nPhase 2: Load Testing")
        load_test_success = await runner.run_load_tests()
        
        print("\nPhase 3: Baseline Establishment")
        baseline_success = await runner.run_baseline_establishment()
        
        runner.test_end_time = datetime.now().timestamp()
        
        # Generate and save comprehensive report
        print("\nGenerating comprehensive performance report...")
        report_path = runner.save_comprehensive_report()
        
        # Print final summary
        report = runner.generate_comprehensive_report()
        main_report = report["comprehensive_performance_report"]
        
        print("\n" + "=" * 60)
        print("COMPREHENSIVE PERFORMANCE TEST RESULTS")
        print("=" * 60)
        
        execution = main_report["test_execution"]
        performance = main_report["performance_summary"]
        health = main_report["system_health_assessment"]
        readiness = main_report["production_readiness"]
        
        print(f"Test Execution: {execution['overall_status']}")
        print(f"Tests Completed: {len(execution['tests_completed'])}/{len(execution['tests_completed']) + len(execution['tests_failed'])}")
        print(f"Duration: {execution['duration_seconds']:.1f} seconds")
        print()
        
        print(f"System Health: {health['overall_health']}")
        print(f"Functional Components: {health['functional_components']}/{health['total_components']}")
        print(f"Average Success Rate: {performance['average_success_rate']:.1%}")
        print(f"Overall Throughput: {performance['overall_throughput']:.2f} ops/sec")
        print()
        
        print(f"Production Readiness: {readiness['overall_readiness']}")
        print(f"Readiness Score: {readiness['readiness_score']:.1%}")
        print(f"Recommendation: {readiness['recommendation']}")
        print()
        
        if health["critical_issues"]:
            print(f"Critical Issues ({len(health['critical_issues'])}):")
            for issue in health["critical_issues"][:3]:  # Show first 3
                print(f"  - {issue}")
            if len(health["critical_issues"]) > 3:
                print(f"  ... and {len(health['critical_issues']) - 3} more")
            print()
        
        print(f"Detailed reports saved to: {runner.output_dir}")
        print(f"Main report: {report_path}")
        
        # Return appropriate exit code
        if readiness["overall_readiness"] in ["PRODUCTION_READY", "CONDITIONALLY_READY"]:
            return 0
        else:
            return 1
            
    except Exception as e:
        logger.error(f"Comprehensive performance testing failed: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))