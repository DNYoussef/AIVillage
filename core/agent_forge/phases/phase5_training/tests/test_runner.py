"""
Automated test runner for Phase 5 Training tests
Comprehensive test execution with coverage analysis and reporting
"""

import pytest
import subprocess
import sys
import os
import json
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import argparse

@dataclass
class TestResult:
    """Test result data structure"""
    name: str
    status: str  # 'passed', 'failed', 'skipped', 'error'
    duration: float
    output: str
    error: Optional[str] = None
    coverage: Optional[float] = None

@dataclass
class TestSuite:
    """Test suite configuration"""
    name: str
    path: str
    description: str
    dependencies: List[str]
    timeout: int = 300  # 5 minutes default
    parallel: bool = True
    markers: List[str] = None
    
    def __post_init__(self):
        if self.markers is None:
            self.markers = []

class Phase5TestRunner:
    """Comprehensive test runner for Phase 5 training tests"""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path(__file__).parent
        self.results: Dict[str, TestResult] = {}
        self.coverage_data = {}
        self.start_time = None
        self.end_time = None
        
        # Define test suites
        self.test_suites = self._define_test_suites()
    
    def _define_test_suites(self) -> Dict[str, TestSuite]:
        """Define all test suites for Phase 5"""
        return {
            'unit_data_loader': TestSuite(
                name='Unit Tests - Data Loader',
                path='unit/test_data_loader.py',
                description='Unit tests for data loading components',
                dependencies=[],
                timeout=120,
                markers=['unit', 'data']
            ),
            'unit_training_loop': TestSuite(
                name='Unit Tests - Training Loop',
                path='unit/test_training_loop.py',
                description='Unit tests for training loop components',
                dependencies=['unit_data_loader'],
                timeout=180,
                markers=['unit', 'training']
            ),
            'unit_bitnet_optimizer': TestSuite(
                name='Unit Tests - BitNet Optimizer',
                path='unit/test_bitnet_optimizer.py',
                description='Unit tests for BitNet optimization',
                dependencies=[],
                timeout=150,
                markers=['unit', 'bitnet']
            ),
            'integration_phase4': TestSuite(
                name='Integration Tests - Phase 4',
                path='integration/test_phase4_integration.py',
                description='Integration tests with Phase 4 BitNet models',
                dependencies=['unit_bitnet_optimizer'],
                timeout=300,
                markers=['integration', 'phase4']
            ),
            'integration_phase6': TestSuite(
                name='Integration Tests - Phase 6',
                path='integration/test_phase6_preparation.py',
                description='Integration tests for Phase 6 preparation',
                dependencies=['unit_training_loop'],
                timeout=240,
                markers=['integration', 'phase6']
            ),
            'performance_training': TestSuite(
                name='Performance Tests - Training',
                path='performance/test_training_performance.py',
                description='Performance benchmarks for training',
                dependencies=['unit_training_loop'],
                timeout=600,
                parallel=False,  # Performance tests should run sequentially
                markers=['performance', 'benchmark']
            ),
            'quality_model': TestSuite(
                name='Quality Tests - Model Quality',
                path='quality/test_model_quality.py',
                description='Model quality and stability validation',
                dependencies=['unit_training_loop', 'integration_phase4'],
                timeout=400,
                markers=['quality', 'validation']
            ),
            'distributed_training': TestSuite(
                name='Distributed Tests - Training',
                path='distributed/test_distributed_training.py',
                description='Distributed training coordination tests',
                dependencies=['unit_training_loop'],
                timeout=500,
                parallel=False,  # Distributed tests need controlled environment
                markers=['distributed', 'gpu']
            )
        }
    
    def run_all_tests(self, 
                      parallel: bool = True,
                      coverage: bool = True,
                      markers: List[str] = None,
                      exclude_markers: List[str] = None) -> Dict[str, TestResult]:
        """Run all test suites"""
        self.start_time = time.time()
        
        print("🚀 Starting Phase 5 Training Test Suite")
        print("=" * 60)
        
        # Filter test suites based on markers
        suites_to_run = self._filter_test_suites(markers, exclude_markers)
        
        # Sort suites by dependencies
        execution_order = self._resolve_dependencies(suites_to_run)
        
        print(f"📋 Test execution plan ({len(execution_order)} suites):")
        for i, suite_name in enumerate(execution_order, 1):
            suite = self.test_suites[suite_name]
            print(f"  {i}. {suite.name}")
        print()
        
        # Execute test suites
        if parallel:
            self._run_suites_parallel(execution_order, coverage)
        else:
            self._run_suites_sequential(execution_order, coverage)
        
        self.end_time = time.time()
        
        # Generate summary report
        self._print_summary()
        
        return self.results
    
    def _filter_test_suites(self, 
                           include_markers: List[str] = None,
                           exclude_markers: List[str] = None) -> List[str]:
        """Filter test suites based on markers"""
        suites_to_run = []
        
        for suite_name, suite in self.test_suites.items():
            # Check include markers
            if include_markers:
                if not any(marker in suite.markers for marker in include_markers):
                    continue
            
            # Check exclude markers
            if exclude_markers:
                if any(marker in suite.markers for marker in exclude_markers):
                    continue
            
            suites_to_run.append(suite_name)
        
        return suites_to_run
    
    def _resolve_dependencies(self, suite_names: List[str]) -> List[str]:
        """Resolve dependencies and return execution order"""
        resolved = []
        remaining = suite_names.copy()
        
        while remaining:
            # Find suites with no unresolved dependencies
            ready = []
            for suite_name in remaining:
                suite = self.test_suites[suite_name]
                deps_resolved = all(dep in resolved or dep not in suite_names 
                                  for dep in suite.dependencies)
                if deps_resolved:
                    ready.append(suite_name)
            
            if not ready:
                # Circular dependency or missing dependency
                print(f"⚠️  Warning: Circular or missing dependencies for {remaining}")
                ready = remaining  # Run anyway
            
            # Add ready suites to resolved list
            for suite_name in ready:
                resolved.append(suite_name)
                remaining.remove(suite_name)
        
        return resolved
    
    def _run_suites_sequential(self, suite_names: List[str], coverage: bool):
        """Run test suites sequentially"""
        for suite_name in suite_names:
            print(f"🧪 Running {self.test_suites[suite_name].name}...")
            result = self._run_single_suite(suite_name, coverage)
            self.results[suite_name] = result
            
            # Print immediate result
            status_emoji = "✅" if result.status == "passed" else "❌"
            print(f"   {status_emoji} {result.status.upper()} in {result.duration:.2f}s")
            
            if result.error:
                print(f"   Error: {result.error}")
            print()
    
    def _run_suites_parallel(self, suite_names: List[str], coverage: bool):
        """Run test suites in parallel where possible"""
        # Group suites by dependency level
        levels = self._group_by_dependency_level(suite_names)
        
        for level, suites_in_level in levels.items():
            print(f"📊 Running level {level} tests ({len(suites_in_level)} suites)...")
            
            # Separate parallel and sequential suites
            parallel_suites = [s for s in suites_in_level 
                             if self.test_suites[s].parallel]
            sequential_suites = [s for s in suites_in_level 
                               if not self.test_suites[s].parallel]
            
            # Run parallel suites
            if parallel_suites:
                with ThreadPoolExecutor(max_workers=min(4, len(parallel_suites))) as executor:
                    futures = {
                        executor.submit(self._run_single_suite, suite_name, coverage): suite_name
                        for suite_name in parallel_suites
                    }
                    
                    for future in futures:
                        suite_name = futures[future]
                        try:
                            result = future.result()
                            self.results[suite_name] = result
                        except Exception as e:
                            self.results[suite_name] = TestResult(
                                name=suite_name,
                                status='error',
                                duration=0,
                                output='',
                                error=str(e)
                            )
            
            # Run sequential suites
            for suite_name in sequential_suites:
                result = self._run_single_suite(suite_name, coverage)
                self.results[suite_name] = result
            
            # Print level results
            for suite_name in suites_in_level:
                result = self.results[suite_name]
                status_emoji = "✅" if result.status == "passed" else "❌"
                print(f"   {status_emoji} {self.test_suites[suite_name].name}: "
                      f"{result.status.upper()} in {result.duration:.2f}s")
            print()
    
    def _group_by_dependency_level(self, suite_names: List[str]) -> Dict[int, List[str]]:
        """Group suites by dependency level for parallel execution"""
        levels = {}
        suite_levels = {}
        
        def get_level(suite_name):
            if suite_name in suite_levels:
                return suite_levels[suite_name]
            
            suite = self.test_suites[suite_name]
            if not suite.dependencies:
                level = 0
            else:
                level = max(get_level(dep) for dep in suite.dependencies 
                           if dep in suite_names) + 1
            
            suite_levels[suite_name] = level
            return level
        
        for suite_name in suite_names:
            level = get_level(suite_name)
            if level not in levels:
                levels[level] = []
            levels[level].append(suite_name)
        
        return levels
    
    def _run_single_suite(self, suite_name: str, coverage: bool) -> TestResult:
        """Run a single test suite"""
        suite = self.test_suites[suite_name]
        test_path = self.base_path / suite.path
        
        if not test_path.exists():
            return TestResult(
                name=suite_name,
                status='error',
                duration=0,
                output='',
                error=f'Test file not found: {test_path}'
            )
        
        # Build pytest command
        cmd = [sys.executable, '-m', 'pytest', str(test_path), '-v']
        
        # Add coverage if requested
        if coverage:
            cmd.extend(['--cov', '--cov-report=json'])
        
        # Add markers
        if suite.markers:
            cmd.extend(['-m', ' or '.join(suite.markers)])
        
        # Add timeout
        cmd.extend(['--timeout', str(suite.timeout)])
        
        # Execute test
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=suite.timeout + 30,  # Extra buffer
                cwd=self.base_path
            )
            
            duration = time.time() - start_time
            
            # Determine status
            if result.returncode == 0:
                status = 'passed'
            elif result.returncode == 5:  # No tests collected
                status = 'skipped'
            else:
                status = 'failed'
            
            # Extract coverage if available
            coverage_value = None
            if coverage:
                coverage_value = self._extract_coverage(result.stdout)
            
            return TestResult(
                name=suite_name,
                status=status,
                duration=duration,
                output=result.stdout,
                error=result.stderr if result.stderr else None,
                coverage=coverage_value
            )
            
        except subprocess.TimeoutExpired:
            return TestResult(
                name=suite_name,
                status='timeout',
                duration=suite.timeout,
                output='',
                error=f'Test timed out after {suite.timeout} seconds'
            )
        except Exception as e:
            return TestResult(
                name=suite_name,
                status='error',
                duration=time.time() - start_time,
                output='',
                error=str(e)
            )
    
    def _extract_coverage(self, output: str) -> Optional[float]:
        """Extract coverage percentage from pytest output"""
        lines = output.split('\n')
        for line in lines:
            if 'TOTAL' in line and '%' in line:
                # Try to extract percentage
                parts = line.split()
                for part in parts:
                    if part.endswith('%'):
                        try:
                            return float(part[:-1])
                        except ValueError:
                            continue
        return None
    
    def _print_summary(self):
        """Print test execution summary"""
        total_duration = self.end_time - self.start_time
        
        print("📊 TEST EXECUTION SUMMARY")
        print("=" * 60)
        print(f"⏱️  Total execution time: {total_duration:.2f} seconds")
        print()
        
        # Count results by status
        status_counts = {}
        total_tests = len(self.results)
        
        for result in self.results.values():
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
        
        # Print status summary
        print("📈 Test Results:")
        for status, count in status_counts.items():
            percentage = (count / total_tests) * 100
            emoji = {
                'passed': '✅',
                'failed': '❌',
                'skipped': '⏭️',
                'error': '💥',
                'timeout': '⏰'
            }.get(status, '❓')
            print(f"   {emoji} {status.upper()}: {count} ({percentage:.1f}%)")
        
        print()
        
        # Print detailed results
        print("📋 Detailed Results:")
        for suite_name, result in self.results.items():
            suite = self.test_suites[suite_name]
            status_emoji = {
                'passed': '✅',
                'failed': '❌',
                'skipped': '⏭️',
                'error': '💥',
                'timeout': '⏰'
            }.get(result.status, '❓')
            
            coverage_str = ""
            if result.coverage is not None:
                coverage_str = f" (Coverage: {result.coverage:.1f}%)"
            
            print(f"   {status_emoji} {suite.name}: {result.status.upper()} "
                  f"in {result.duration:.2f}s{coverage_str}")
            
            if result.error:
                print(f"      Error: {result.error[:100]}...")
        
        print()
        
        # Overall success rate
        passed = status_counts.get('passed', 0)
        success_rate = (passed / total_tests) * 100 if total_tests > 0 else 0
        
        if success_rate >= 95:
            print(f"🎉 EXCELLENT: {success_rate:.1f}% success rate!")
        elif success_rate >= 80:
            print(f"👍 GOOD: {success_rate:.1f}% success rate")
        elif success_rate >= 60:
            print(f"⚠️  FAIR: {success_rate:.1f}% success rate - needs improvement")
        else:
            print(f"🚨 POOR: {success_rate:.1f}% success rate - significant issues")
    
    def save_results(self, output_file: str):
        """Save test results to JSON file"""
        report_data = {
            'execution_time': {
                'start': self.start_time,
                'end': self.end_time,
                'duration': self.end_time - self.start_time if self.end_time else 0
            },
            'test_suites': {
                name: {
                    'name': suite.name,
                    'description': suite.description,
                    'path': suite.path,
                    'markers': suite.markers
                }
                for name, suite in self.test_suites.items()
            },
            'results': {
                name: {
                    'name': result.name,
                    'status': result.status,
                    'duration': result.duration,
                    'coverage': result.coverage,
                    'error': result.error
                }
                for name, result in self.results.items()
            },
            'summary': {
                'total_tests': len(self.results),
                'passed': sum(1 for r in self.results.values() if r.status == 'passed'),
                'failed': sum(1 for r in self.results.values() if r.status == 'failed'),
                'errors': sum(1 for r in self.results.values() if r.status == 'error'),
                'skipped': sum(1 for r in self.results.values() if r.status == 'skipped'),
                'timeouts': sum(1 for r in self.results.values() if r.status == 'timeout')
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"💾 Test results saved to: {output_file}")
    
    def run_specific_suite(self, suite_name: str, coverage: bool = True) -> TestResult:
        """Run a specific test suite"""
        if suite_name not in self.test_suites:
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        print(f"🧪 Running {self.test_suites[suite_name].name}...")
        
        self.start_time = time.time()
        result = self._run_single_suite(suite_name, coverage)
        self.end_time = time.time()
        
        self.results[suite_name] = result
        
        # Print result
        status_emoji = "✅" if result.status == "passed" else "❌"
        print(f"{status_emoji} {result.status.upper()} in {result.duration:.2f}s")
        
        if result.error:
            print(f"Error: {result.error}")
        
        return result

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='Phase 5 Training Test Runner')
    
    parser.add_argument('--suite', type=str, help='Run specific test suite')
    parser.add_argument('--markers', nargs='+', help='Include tests with these markers')
    parser.add_argument('--exclude-markers', nargs='+', help='Exclude tests with these markers')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel execution')
    parser.add_argument('--no-coverage', action='store_true', help='Disable coverage reporting')
    parser.add_argument('--output', type=str, help='Save results to JSON file')
    parser.add_argument('--list-suites', action='store_true', help='List available test suites')
    
    args = parser.parse_args()
    
    runner = Phase5TestRunner()
    
    if args.list_suites:
        print("📋 Available Test Suites:")
        print("=" * 40)
        for name, suite in runner.test_suites.items():
            print(f"🧪 {name}")
            print(f"   Name: {suite.name}")
            print(f"   Description: {suite.description}")
            print(f"   Path: {suite.path}")
            print(f"   Markers: {', '.join(suite.markers)}")
            print(f"   Dependencies: {', '.join(suite.dependencies) if suite.dependencies else 'None'}")
            print()
        return
    
    try:
        if args.suite:
            # Run specific suite
            result = runner.run_specific_suite(args.suite, coverage=not args.no_coverage)
        else:
            # Run all tests
            results = runner.run_all_tests(
                parallel=not args.no_parallel,
                coverage=not args.no_coverage,
                markers=args.markers,
                exclude_markers=args.exclude_markers
            )
        
        # Save results if requested
        if args.output:
            runner.save_results(args.output)
        
        # Exit with appropriate code
        failed_tests = sum(1 for r in runner.results.values() 
                          if r.status in ['failed', 'error', 'timeout'])
        sys.exit(1 if failed_tests > 0 else 0)
        
    except KeyboardInterrupt:
        print("\n❌ Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Test runner error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()