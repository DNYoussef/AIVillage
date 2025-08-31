#!/usr/bin/env python3
"""
Constitutional Testing Suite Runner

Comprehensive test runner for constitutional fog compute safety validation
with support for different test categories, performance benchmarking,
and CI/CD integration.
"""

import sys
import os
import subprocess
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class ConstitutionalTestRunner:
    """Comprehensive test runner for constitutional system validation"""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent.parent
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    def run_all_tests(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Run complete constitutional test suite"""
        print("🧪 Starting Constitutional Safety Validation Test Suite")
        print("=" * 60)
        
        self.start_time = datetime.now()
        test_results = {}
        
        # Test categories to run
        test_categories = self._get_test_categories(args)
        
        for category, config in test_categories.items():
            if getattr(args, category, True):  # Run unless explicitly disabled
                print(f"\n📋 Running {category.replace('_', ' ').title()} Tests")
                print("-" * 40)
                
                result = self._run_test_category(category, config, args)
                test_results[category] = result
                
                if result['status'] == 'failed' and args.fail_fast:
                    print(f"❌ {category} tests failed - stopping due to fail-fast mode")
                    break
        
        self.end_time = datetime.now()
        
        # Generate comprehensive report
        final_report = self._generate_final_report(test_results, args)
        
        # Save results
        if args.output_file:
            self._save_results(final_report, args.output_file)
        
        # Print summary
        self._print_summary(final_report)
        
        return final_report
    
    def _get_test_categories(self, args: argparse.Namespace) -> Dict[str, Dict]:
        """Get test categories and their configurations"""
        return {
            'safety_validation': {
                'files': ['test_safety_validation.py'],
                'markers': ['unit', 'constitutional'],
                'description': 'Core constitutional safety validation tests',
                'timeout': 300,
                'parallel': True
            },
            'harm_classification': {
                'files': ['test_harm_classification.py'],
                'markers': ['unit', 'harm_classification', 'bias'],
                'description': 'ML harm classification accuracy and bias testing',
                'timeout': 600,
                'parallel': True
            },
            'constitutional_compliance': {
                'files': ['test_constitutional_compliance.py'],
                'markers': ['constitutional', 'integration'],
                'description': 'Constitutional principle adherence testing',
                'timeout': 400,
                'parallel': True
            },
            'tier_enforcement': {
                'files': ['test_tier_enforcement.py'],
                'markers': ['tier_enforcement', 'integration'],
                'description': 'Tier-based constitutional protection testing',
                'timeout': 300,
                'parallel': True
            },
            'e2e_integration': {
                'files': ['test_integration_e2e.py'],
                'markers': ['e2e', 'integration'],
                'description': 'End-to-end system integration testing',
                'timeout': 900,
                'parallel': False  # E2E tests often need sequential execution
            },
            'performance_benchmarks': {
                'files': ['test_performance_benchmarks.py'],
                'markers': ['performance', 'benchmark'],
                'description': 'Performance and latency benchmark testing',
                'timeout': 1200,
                'parallel': args.parallel_performance if hasattr(args, 'parallel_performance') else False
            },
            'adversarial_testing': {
                'files': ['adversarial/test_adversarial_attacks.py'],
                'markers': ['adversarial', 'edge_case', 'robustness'],
                'description': 'Adversarial attack and edge case testing',
                'timeout': 800,
                'parallel': True
            }
        }
    
    def _run_test_category(self, category: str, config: Dict, args: argparse.Namespace) -> Dict[str, Any]:
        """Run a specific test category"""
        start_time = time.time()
        
        # Build pytest command
        cmd = ['python', '-m', 'pytest']
        
        # Add test files
        for test_file in config['files']:
            cmd.append(str(self.test_dir / test_file))
        
        # Add markers
        if config['markers'] and not args.ignore_markers:
            marker_expr = ' or '.join(config['markers'])
            cmd.extend(['-m', marker_expr])
        
        # Add pytest options
        cmd.extend([
            '--verbose',
            '--tb=short',
            f'--timeout={config["timeout"]}',
            '--color=yes',
            '--disable-warnings'
        ])
        
        # Parallel execution
        if config['parallel'] and args.parallel and not args.no_parallel:
            import multiprocessing
            max_workers = min(args.max_workers, multiprocessing.cpu_count())
            cmd.extend(['-n', str(max_workers)])
        
        # Coverage options
        if args.coverage:
            cmd.extend([
                '--cov=core/constitutional',
                '--cov-report=term-missing',
                f'--cov-fail-under={args.coverage_threshold}'
            ])
        
        # Output options
        if args.junit_xml:
            junit_file = self.test_dir / f'results/junit_{category}.xml'
            junit_file.parent.mkdir(exist_ok=True)
            cmd.extend(['--junit-xml', str(junit_file)])
        
        # Performance profiling
        if category == 'performance_benchmarks' and args.profile:
            cmd.extend(['--profile'])
        
        # Execute tests
        print(f"🔧 Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=config['timeout']
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            test_result = {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'returncode': result.returncode,
                'execution_time_seconds': execution_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'config': config,
                'command': ' '.join(cmd)
            }
            
            # Extract test metrics from output
            test_result.update(self._parse_pytest_output(result.stdout))
            
            # Print results
            if test_result['status'] == 'passed':
                print(f"✅ {category} tests passed ({execution_time:.1f}s)")
                if 'tests_passed' in test_result:
                    print(f"   📊 {test_result['tests_passed']} tests passed")
            else:
                print(f"❌ {category} tests failed ({execution_time:.1f}s)")
                if 'tests_failed' in test_result:
                    print(f"   📊 {test_result['tests_failed']} tests failed, {test_result.get('tests_passed', 0)} passed")
                
                # Print error details if verbose
                if args.verbose and result.stderr:
                    print(f"   🔍 Error details: {result.stderr[:500]}...")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            print(f"⏰ {category} tests timed out after {execution_time:.1f}s")
            
            return {
                'status': 'timeout',
                'returncode': -1,
                'execution_time_seconds': execution_time,
                'error': f'Tests timed out after {config["timeout"]}s',
                'config': config
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"💥 {category} tests encountered error: {str(e)}")
            
            return {
                'status': 'error',
                'returncode': -1,
                'execution_time_seconds': execution_time,
                'error': str(e),
                'config': config
            }
    
    def _parse_pytest_output(self, stdout: str) -> Dict[str, Any]:
        """Parse pytest output to extract test metrics"""
        metrics = {}
        
        lines = stdout.split('\n')
        for line in lines:
            # Parse test results summary
            if 'passed' in line and 'failed' in line:
                # Example: "5 failed, 23 passed, 2 skipped in 45.67s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'failed,' and i > 0:
                        metrics['tests_failed'] = int(parts[i-1])
                    elif part == 'passed,' and i > 0:
                        metrics['tests_passed'] = int(parts[i-1])
                    elif part == 'skipped' and i > 0:
                        metrics['tests_skipped'] = int(parts[i-1])
                    elif 'passed' in part and i > 0:  # Just "passed"
                        try:
                            metrics['tests_passed'] = int(parts[i-1])
                        except (ValueError, IndexError):
                            pass
            
            # Parse performance metrics
            if 'slowest durations' in line.lower():
                metrics['performance_data_available'] = True
            
            # Parse coverage information
            if 'TOTAL' in line and '%' in line:
                parts = line.split()
                for part in parts:
                    if '%' in part:
                        try:
                            metrics['coverage_percent'] = float(part.strip('%'))
                        except ValueError:
                            pass
        
        return metrics
    
    def _generate_final_report(self, test_results: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
        """Generate comprehensive final test report"""
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        # Calculate aggregate statistics
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        total_errors = 0
        total_timeouts = 0
        
        categories_passed = 0
        categories_failed = 0
        
        for category, result in test_results.items():
            total_tests += result.get('tests_passed', 0) + result.get('tests_failed', 0) + result.get('tests_skipped', 0)
            total_passed += result.get('tests_passed', 0)
            total_failed += result.get('tests_failed', 0)
            total_skipped += result.get('tests_skipped', 0)
            
            if result['status'] == 'passed':
                categories_passed += 1
            elif result['status'] == 'failed':
                categories_failed += 1
            elif result['status'] == 'error':
                total_errors += 1
            elif result['status'] == 'timeout':
                total_timeouts += 1
        
        # Calculate success rates
        test_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        category_success_rate = (categories_passed / len(test_results) * 100) if test_results else 0
        
        # Generate report
        report = {
            'summary': {
                'execution_time': {
                    'start': self.start_time.isoformat(),
                    'end': self.end_time.isoformat(),
                    'duration_seconds': total_duration,
                    'duration_formatted': f"{total_duration:.1f}s"
                },
                'test_statistics': {
                    'total_tests': total_tests,
                    'tests_passed': total_passed,
                    'tests_failed': total_failed,
                    'tests_skipped': total_skipped,
                    'test_success_rate_percent': round(test_success_rate, 2)
                },
                'category_statistics': {
                    'total_categories': len(test_results),
                    'categories_passed': categories_passed,
                    'categories_failed': categories_failed,
                    'categories_errors': total_errors,
                    'categories_timeouts': total_timeouts,
                    'category_success_rate_percent': round(category_success_rate, 2)
                },
                'overall_status': 'PASSED' if categories_failed == 0 and total_errors == 0 and total_timeouts == 0 else 'FAILED',
                'constitutional_safety_validated': categories_failed == 0 and total_failed == 0,
            },
            'detailed_results': test_results,
            'configuration': {
                'parallel_execution': args.parallel,
                'max_workers': getattr(args, 'max_workers', 1),
                'coverage_enabled': args.coverage,
                'fail_fast_mode': args.fail_fast,
                'verbose_output': args.verbose
            },
            'environment': {
                'python_version': sys.version,
                'platform': sys.platform,
                'test_runner_version': '1.0.0',
                'project_root': str(self.project_root)
            }
        }
        
        return report
    
    def _save_results(self, report: Dict[str, Any], output_file: str):
        """Save test results to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Test results saved to: {output_path}")
    
    def _print_summary(self, report: Dict[str, Any]):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 CONSTITUTIONAL SAFETY VALIDATION SUMMARY")
        print("=" * 60)
        
        summary = report['summary']
        
        # Overall status
        status = summary['overall_status']
        status_emoji = "✅" if status == 'PASSED' else "❌"
        print(f"{status_emoji} Overall Status: {status}")
        
        # Constitutional safety validation
        safety_validated = summary['constitutional_safety_validated']
        safety_emoji = "🛡️" if safety_validated else "⚠️"
        safety_status = "VALIDATED" if safety_validated else "VALIDATION FAILED"
        print(f"{safety_emoji} Constitutional Safety: {safety_status}")
        
        # Test statistics
        test_stats = summary['test_statistics']
        print(f"\n📈 Test Statistics:")
        print(f"   Total Tests: {test_stats['total_tests']}")
        print(f"   Passed: {test_stats['tests_passed']} ({test_stats['test_success_rate_percent']:.1f}%)")
        print(f"   Failed: {test_stats['tests_failed']}")
        print(f"   Skipped: {test_stats['tests_skipped']}")
        
        # Category statistics
        cat_stats = summary['category_statistics']
        print(f"\n📋 Category Statistics:")
        print(f"   Total Categories: {cat_stats['total_categories']}")
        print(f"   Passed: {cat_stats['categories_passed']} ({cat_stats['category_success_rate_percent']:.1f}%)")
        print(f"   Failed: {cat_stats['categories_failed']}")
        if cat_stats['categories_errors'] > 0:
            print(f"   Errors: {cat_stats['categories_errors']}")
        if cat_stats['categories_timeouts'] > 0:
            print(f"   Timeouts: {cat_stats['categories_timeouts']}")
        
        # Execution time
        duration = summary['execution_time']['duration_formatted']
        print(f"\n⏱️  Total Execution Time: {duration}")
        
        # Category details
        print(f"\n📂 Category Results:")
        for category, result in report['detailed_results'].items():
            status_emoji = "✅" if result['status'] == 'passed' else "❌" if result['status'] == 'failed' else "⚠️"
            duration = f"{result.get('execution_time_seconds', 0):.1f}s"
            print(f"   {status_emoji} {category.replace('_', ' ').title()}: {result['status'].upper()} ({duration})")
        
        # Final verdict
        print("\n" + "=" * 60)
        if status == 'PASSED' and safety_validated:
            print("🎉 CONSTITUTIONAL SAFETY VALIDATION SUCCESSFUL")
            print("   All constitutional safety requirements validated ✓")
            print("   System ready for democratic fog compute deployment ✓")
        else:
            print("🚨 CONSTITUTIONAL SAFETY VALIDATION FAILED")
            print("   System requires fixes before deployment ✗")
            print("   Review failed tests and constitutional compliance ✗")
        print("=" * 60)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Constitutional Safety Validation Test Suite Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python run_tests.py

  # Run only safety validation tests
  python run_tests.py --safety-validation-only

  # Run with coverage and parallel execution
  python run_tests.py --coverage --parallel --max-workers 4

  # Run performance benchmarks only
  python run_tests.py --performance-benchmarks-only --profile

  # Run adversarial tests only
  python run_tests.py --adversarial-testing-only --verbose

  # Run with custom output
  python run_tests.py --output-file results/test_report.json --junit-xml
        """
    )
    
    # Test selection options
    parser.add_argument('--safety-validation-only', action='store_true',
                       help='Run only safety validation tests')
    parser.add_argument('--harm-classification-only', action='store_true', 
                       help='Run only harm classification tests')
    parser.add_argument('--constitutional-compliance-only', action='store_true',
                       help='Run only constitutional compliance tests')
    parser.add_argument('--tier-enforcement-only', action='store_true',
                       help='Run only tier enforcement tests')
    parser.add_argument('--e2e-integration-only', action='store_true',
                       help='Run only end-to-end integration tests')
    parser.add_argument('--performance-benchmarks-only', action='store_true',
                       help='Run only performance benchmark tests')
    parser.add_argument('--adversarial-testing-only', action='store_true',
                       help='Run only adversarial testing tests')
    
    # Execution options
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Enable parallel test execution (default: True)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel test execution')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of parallel workers (default: 4)')
    parser.add_argument('--fail-fast', action='store_true',
                       help='Stop on first test category failure')
    
    # Coverage options
    parser.add_argument('--coverage', action='store_true',
                       help='Enable code coverage reporting')
    parser.add_argument('--coverage-threshold', type=int, default=80,
                       help='Coverage threshold percentage (default: 80)')
    
    # Output options
    parser.add_argument('--output-file', type=str,
                       help='Save results to JSON file')
    parser.add_argument('--junit-xml', action='store_true',
                       help='Generate JUnit XML reports')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output with detailed error information')
    
    # Performance options
    parser.add_argument('--profile', action='store_true',
                       help='Enable performance profiling')
    parser.add_argument('--parallel-performance', action='store_true',
                       help='Enable parallel execution for performance tests')
    
    # Advanced options
    parser.add_argument('--ignore-markers', action='store_true',
                       help='Ignore pytest markers and run all tests in files')
    parser.add_argument('--timeout-multiplier', type=float, default=1.0,
                       help='Multiply all timeouts by this factor (default: 1.0)')
    
    return parser


def main():
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Handle mutually exclusive test selection
    test_only_flags = [
        args.safety_validation_only,
        args.harm_classification_only,
        args.constitutional_compliance_only,
        args.tier_enforcement_only,
        args.e2e_integration_only,
        args.performance_benchmarks_only,
        args.adversarial_testing_only
    ]
    
    if sum(test_only_flags) > 1:
        parser.error("Only one --*-only flag can be specified")
    
    # If any --*-only flag is specified, disable others
    if any(test_only_flags):
        args.safety_validation = args.safety_validation_only
        args.harm_classification = args.harm_classification_only
        args.constitutional_compliance = args.constitutional_compliance_only
        args.tier_enforcement = args.tier_enforcement_only
        args.e2e_integration = args.e2e_integration_only
        args.performance_benchmarks = args.performance_benchmarks_only
        args.adversarial_testing = args.adversarial_testing_only
    else:
        # Run all tests by default
        args.safety_validation = True
        args.harm_classification = True
        args.constitutional_compliance = True
        args.tier_enforcement = True
        args.e2e_integration = True
        args.performance_benchmarks = True
        args.adversarial_testing = True
    
    # Create and run test suite
    runner = ConstitutionalTestRunner()
    
    try:
        report = runner.run_all_tests(args)
        
        # Exit with appropriate code
        if report['summary']['overall_status'] == 'PASSED':
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test runner encountered fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()