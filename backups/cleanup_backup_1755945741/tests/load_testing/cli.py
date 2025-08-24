#!/usr/bin/env python3
"""
AIVillage Load Testing CLI
=========================

Command-line interface for AIVillage load testing infrastructure.
Provides easy access to all testing capabilities with sensible defaults.

Usage:
    # Quick system validation (5 minutes)
    python cli.py quick

    # Basic load test (15 minutes)
    python cli.py load --profile basic

    # Full production readiness suite (30-60 minutes)
    python cli.py production

    # 24-hour soak test
    python cli.py soak --duration 24

    # Performance regression testing
    python cli.py regression --baseline
    python cli.py regression --compare

    # Help for specific commands
    python cli.py <command> --help
"""

import argparse
import asyncio
from pathlib import Path
import sys

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import our load testing infrastructure
try:
    from .integrated_load_test_runner import IntegratedLoadTestRunner, TestSuiteConfig
    from .performance_regression_detector import PerformanceBenchmarkRunner, PerformanceRegressionDetector
    from .production_load_test_suite import ProductionLoadTestSuite, create_test_profiles
    from .soak_test_orchestrator import SoakTestConfig, SoakTestOrchestrator
except ImportError:
    # Fallback for direct execution
    from integrated_load_test_runner import IntegratedLoadTestRunner, TestSuiteConfig
    from performance_regression_detector import PerformanceBenchmarkRunner, PerformanceRegressionDetector
    from production_load_test_suite import ProductionLoadTestSuite, create_test_profiles
    from soak_test_orchestrator import SoakTestConfig, SoakTestOrchestrator


def create_common_parser() -> argparse.ArgumentParser:
    """Create parser with common arguments"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of AIVillage system (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="load_test_results",
        help="Output directory for results (default: load_test_results)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    return parser


async def cmd_quick(args):
    """Quick validation test (5 minutes)"""
    print("Running quick validation test...")

    config = TestSuiteConfig(
        suite_name="quick_validation",
        base_url=args.base_url,
        output_dir=args.output_dir,
        run_quick_load_test=True,
        run_full_load_test=False,
        run_soak_test=False,
        load_test_profiles=["quick"],
    )

    runner = IntegratedLoadTestRunner(config)
    report = await runner.run_full_suite()

    print(f"\nQuick validation completed: {report.overall_status}")
    return 0 if report.overall_status == "PASS" else 1


async def cmd_load(args):
    """Load testing with specified profile"""
    print(f"Running load test with {args.profile} profile...")

    profiles = create_test_profiles()
    if args.profile not in profiles:
        print(f"[ERROR] Unknown profile: {args.profile}")
        print(f"Available profiles: {', '.join(profiles.keys())}")
        return 1

    config = profiles[args.profile]
    config.base_url = args.base_url

    test_suite = ProductionLoadTestSuite(config)
    metrics = await test_suite.run_load_test()

    print(f"\nLoad test completed: {'PASS' if metrics.passed else 'FAIL'}")
    print(
        f"Stats: {metrics.total_requests} requests, {metrics.error_rate:.3f} error rate, {metrics.requests_per_second:.1f} RPS"
    )

    return 0 if metrics.passed else 1


async def cmd_production(args):
    """Full production readiness test suite"""
    print("Running production readiness test suite...")

    config = TestSuiteConfig(
        suite_name="production_readiness",
        base_url=args.base_url,
        output_dir=args.output_dir,
        run_quick_load_test=True,
        run_full_load_test=True,
        run_soak_test=args.include_soak,
        run_regression_test=True,
        load_test_profiles=["quick", "basic", "stress"],
        soak_test_duration_hours=args.soak_duration,
        enable_alerts=True,
        generate_html_report=True,
    )

    runner = IntegratedLoadTestRunner(config)
    report = await runner.run_full_suite()

    print(f"\nProduction readiness test completed: {report.overall_status}")
    print(f"Summary: {report.summary['total_tests']} tests, {report.summary['success_rate']:.1f}% success rate")

    return 0 if report.overall_status == "PASS" else 1


async def cmd_soak(args):
    """Long-running soak test"""
    print(f"Starting soak test for {args.duration} hours...")

    config = SoakTestConfig(
        test_name=f"soak_test_{args.duration}h",
        duration_hours=args.duration,
        base_url=args.base_url,
        output_dir=args.output_dir / "soak_test",
        concurrent_users=args.users,
        request_rate_per_second=args.rps,
        enable_chaos_testing=args.chaos,
        generate_plots=True,
    )

    orchestrator = SoakTestOrchestrator(config)
    metrics = await orchestrator.run_soak_test()

    test_passed = orchestrator._evaluate_test_success()
    print(f"\nSoak test completed: {'PASS' if test_passed else 'FAIL'}")
    print(f"Stats: {metrics.total_requests} requests over {metrics.elapsed_hours:.1f} hours")

    return 0 if test_passed else 1


async def cmd_regression(args):
    """Performance regression testing"""
    detector = PerformanceRegressionDetector(args.output_dir / "performance_data")
    benchmark_runner = PerformanceBenchmarkRunner({})

    if args.baseline:
        print("Establishing performance baseline...")
        benchmark = await benchmark_runner.run_full_benchmark()
        detector.save_baseline(benchmark, args.baseline_name)
        print(f"Baseline established: {args.baseline_name}")
        return 0

    elif args.compare:
        print("Running performance comparison...")

        # Load baseline
        baseline = detector.load_baseline(args.baseline_name)
        if baseline is None:
            print(f"[ERROR] Baseline not found: {args.baseline_name}")
            return 1

        # Run current benchmark
        current = await benchmark_runner.run_full_benchmark()

        # Compare
        report = detector.detect_regressions(baseline, current)
        detector.save_report(report)
        detector.print_report(report)

        if report.overall_verdict == "PASS":
            print("\n[PASS] No performance regressions detected")
            return 0
        elif report.overall_verdict == "WARNING":
            print("\n[WARN] Performance warnings detected")
            return 1
        else:
            print("\n[FAIL] Performance regressions detected")
            return 2
    else:
        print("[ERROR] Must specify --baseline or --compare")
        return 1


def cmd_status(args):
    """Show system status and test history"""
    print("AIVillage Load Testing Status")
    print("=" * 50)

    # Check if system is reachable
    try:
        import urllib.request

        with urllib.request.urlopen(f"{args.base_url}/health", timeout=5) as response:
            if response.getcode() == 200:
                print("[PASS] System reachable")
            else:
                print(f"[WARN] System returned HTTP {response.getcode()}")
    except Exception as e:
        print(f"[FAIL] System unreachable: {e}")

    # Check for recent test results
    results_dir = args.output_dir
    if results_dir.exists():
        json_files = list(results_dir.glob("*.json"))
        if json_files:
            latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
            print(f"Latest test result: {latest_file.name}")
        else:
            print("No test results found")
    else:
        print("No test results directory")

    return 0


def main():
    """Main CLI entry point"""
    # Create main parser
    parser = argparse.ArgumentParser(
        description="AIVillage Load Testing CLI", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Quick command
    quick_parser = subparsers.add_parser(
        "quick", parents=[create_common_parser()], help="Quick validation test (5 minutes)"
    )
    quick_parser.set_defaults(func=cmd_quick)

    # Load command
    load_parser = subparsers.add_parser(
        "load", parents=[create_common_parser()], help="Load testing with specified profile"
    )
    load_parser.add_argument(
        "--profile",
        default="basic",
        choices=["quick", "basic", "stress", "scale", "soak"],
        help="Load test profile (default: basic)",
    )
    load_parser.set_defaults(func=cmd_load)

    # Production command
    production_parser = subparsers.add_parser(
        "production", parents=[create_common_parser()], help="Full production readiness test suite"
    )
    production_parser.add_argument("--include-soak", action="store_true", help="Include soak test in production suite")
    production_parser.add_argument(
        "--soak-duration", type=float, default=1.0, help="Soak test duration in hours (default: 1.0)"
    )
    production_parser.set_defaults(func=cmd_production)

    # Soak command
    soak_parser = subparsers.add_parser("soak", parents=[create_common_parser()], help="Long-running soak test")
    soak_parser.add_argument("--duration", type=float, default=24.0, help="Test duration in hours (default: 24.0)")
    soak_parser.add_argument("--users", type=int, default=50, help="Concurrent users (default: 50)")
    soak_parser.add_argument("--rps", type=float, default=10.0, help="Requests per second (default: 10.0)")
    soak_parser.add_argument("--chaos", action="store_true", help="Enable chaos testing")
    soak_parser.set_defaults(func=cmd_soak)

    # Regression command
    regression_parser = subparsers.add_parser(
        "regression", parents=[create_common_parser()], help="Performance regression testing"
    )
    regression_group = regression_parser.add_mutually_exclusive_group(required=True)
    regression_group.add_argument("--baseline", action="store_true", help="Establish new performance baseline")
    regression_group.add_argument("--compare", action="store_true", help="Compare against existing baseline")
    regression_parser.add_argument("--baseline-name", default="baseline", help="Baseline name (default: baseline)")
    regression_parser.set_defaults(func=cmd_regression)

    # Status command
    status_parser = subparsers.add_parser(
        "status", parents=[create_common_parser()], help="Show system status and test history"
    )
    status_parser.set_defaults(func=cmd_status)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Set up logging
    if args.verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)

    # Run the selected command
    if asyncio.iscoroutinefunction(args.func):
        return asyncio.run(args.func(args))
    else:
        return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
