#!/usr/bin/env python3
"""
Automated Test Execution Pipeline with MCP Coordination
======================================================

Script for running tests with TDD London School patterns, MCP coordination,
and comprehensive reporting. Integrates with the unified testing infrastructure.
"""

import os
import sys
import subprocess
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import time
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class TestExecutionResult:
    """Results from test execution."""
    suite_name: str
    status: str  # 'passed', 'failed', 'error'
    duration: float
    tests_run: int
    tests_passed: int
    tests_failed: int
    tests_skipped: int
    coverage_percentage: float
    exit_code: int
    output: str
    error_output: str


@dataclass
class PipelineReport:
    """Comprehensive pipeline execution report."""
    total_duration: float
    total_tests_run: int
    total_tests_passed: int
    total_tests_failed: int
    total_tests_skipped: int
    overall_coverage: float
    suite_results: List[TestExecutionResult]
    mcp_coordination_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]


class MCPCoordinator:
    """MCP server coordination for testing pipeline."""
    
    def __init__(self):
        self.enabled = True
        self.hooks_available = self._check_hooks_availability()
        self.logger = logging.getLogger('MCPCoordinator')
    
    def _check_hooks_availability(self) -> bool:
        """Check if claude-flow hooks are available."""
        try:
            result = subprocess.run(
                ['npx', 'claude-flow@alpha', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def initialize_session(self, session_id: str) -> bool:
        """Initialize MCP testing session."""
        if not self.hooks_available:
            self.logger.warning("MCP hooks not available - running without coordination")
            return False
        
        try:
            # Pre-task hook for testing session
            subprocess.run([
                'npx', 'claude-flow@alpha', 'hooks', 'pre-task',
                '--description', f'Testing pipeline session {session_id}'
            ], check=True, capture_output=True, timeout=30)
            
            # Session restore if available
            subprocess.run([
                'npx', 'claude-flow@alpha', 'hooks', 'session-restore',
                '--session-id', f'testing-{session_id}'
            ], capture_output=True, timeout=30)
            
            return True
        except Exception as e:
            self.logger.error(f"MCP session initialization failed: {e}")
            return False
    
    async def notify_test_start(self, suite_name: str) -> None:
        """Notify MCP about test suite start."""
        if not self.hooks_available:
            return
        
        try:
            subprocess.run([
                'npx', 'claude-flow@alpha', 'hooks', 'notify',
                '--message', f'Starting test suite: {suite_name}'
            ], capture_output=True, timeout=10)
        except Exception as e:
            self.logger.debug(f"MCP notify failed: {e}")
    
    async def store_test_results(self, results: TestExecutionResult) -> None:
        """Store test results in MCP memory."""
        if not self.hooks_available:
            return
        
        try:
            memory_key = f"testing/results/{results.suite_name}"
            result_data = json.dumps(asdict(results))
            
            subprocess.run([
                'npx', 'claude-flow@alpha', 'hooks', 'post-edit',
                '--file', f'{results.suite_name}_results.json',
                '--memory-key', memory_key
            ], input=result_data, text=True, capture_output=True, timeout=15)
        except Exception as e:
            self.logger.debug(f"MCP result storage failed: {e}")
    
    async def finalize_session(self, session_id: str) -> Dict[str, Any]:
        """Finalize MCP session and get metrics."""
        if not self.hooks_available:
            return {}
        
        try:
            # Post-task hook
            subprocess.run([
                'npx', 'claude-flow@alpha', 'hooks', 'post-task',
                '--task-id', f'testing-{session_id}'
            ], capture_output=True, timeout=30)
            
            # Session end with metrics export
            result = subprocess.run([
                'npx', 'claude-flow@alpha', 'hooks', 'session-end',
                '--export-metrics', 'true'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout:
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    pass
            
            return {'session_finalized': True}
        except Exception as e:
            self.logger.error(f"MCP session finalization failed: {e}")
            return {}


class TestSuiteRunner:
    """Individual test suite execution with behavior verification."""
    
    def __init__(self, mcp_coordinator: MCPCoordinator):
        self.mcp = mcp_coordinator
        self.logger = logging.getLogger('TestSuiteRunner')
    
    async def run_suite(self, suite_config: Dict[str, Any]) -> TestExecutionResult:
        """Execute a test suite with full coordination."""
        suite_name = suite_config['name']
        test_path = suite_config['path']
        markers = suite_config.get('markers', [])
        
        self.logger.info(f"Starting test suite: {suite_name}")
        await self.mcp.notify_test_start(suite_name)
        
        # Build pytest command
        cmd = self._build_pytest_command(test_path, markers, suite_config)
        
        # Execute tests
        start_time = time.time()
        try:
            result = await self._execute_pytest(cmd)
            duration = time.time() - start_time
            
            # Parse results
            test_result = self._parse_pytest_output(
                suite_name, result, duration
            )
            
            # Store results in MCP
            await self.mcp.store_test_results(test_result)
            
            return test_result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test suite {suite_name} failed: {e}")
            
            return TestExecutionResult(
                suite_name=suite_name,
                status='error',
                duration=duration,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                coverage_percentage=0.0,
                exit_code=-1,
                output='',
                error_output=str(e)
            )
    
    def _build_pytest_command(self, test_path: str, markers: List[str], config: Dict[str, Any]) -> List[str]:
        """Build pytest command with appropriate options."""
        cmd = ['python', '-m', 'pytest']
        
        # Add test path
        cmd.append(test_path)
        
        # Add configuration file
        cmd.extend(['--config-file', 'tests/pytest.ini'])
        
        # Add markers
        if markers:
            for marker in markers:
                cmd.extend(['-m', marker])
        
        # Add output format
        cmd.extend([
            '--tb=short',
            '--verbose',
            '--junit-xml=tests/reports/junit.xml',
            '--cov-report=json:tests/reports/coverage.json'
        ])
        
        # Add parallel execution if enabled
        if config.get('parallel', False):
            cmd.extend(['-n', 'auto'])
        
        # Add timeout
        timeout = config.get('timeout', 300)
        cmd.extend(['--timeout', str(timeout)])
        
        return cmd
    
    async def _execute_pytest(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Execute pytest command asynchronously."""
        loop = asyncio.get_event_loop()
        
        def run_command():
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
        
        with ThreadPoolExecutor() as executor:
            future = executor.submit(run_command)
            return await loop.run_in_executor(None, lambda: future.result())
    
    def _parse_pytest_output(self, suite_name: str, result: subprocess.CompletedProcess, duration: float) -> TestExecutionResult:
        """Parse pytest output to extract test results."""
        output = result.stdout
        error_output = result.stderr
        
        # Default values
        tests_run = tests_passed = tests_failed = tests_skipped = 0
        coverage_percentage = 0.0
        status = 'passed' if result.returncode == 0 else 'failed'
        
        # Parse test counts from output
        import re
        
        # Look for test results pattern
        test_pattern = r'(\d+) failed.*?(\d+) passed.*?(\d+) skipped'
        match = re.search(test_pattern, output)
        if match:
            tests_failed = int(match.group(1))
            tests_passed = int(match.group(2))
            tests_skipped = int(match.group(3))
            tests_run = tests_failed + tests_passed + tests_skipped
        else:
            # Alternative pattern
            passed_pattern = r'(\d+) passed'
            failed_pattern = r'(\d+) failed'
            skipped_pattern = r'(\d+) skipped'
            
            passed_match = re.search(passed_pattern, output)
            failed_match = re.search(failed_pattern, output)
            skipped_match = re.search(skipped_pattern, output)
            
            if passed_match:
                tests_passed = int(passed_match.group(1))
            if failed_match:
                tests_failed = int(failed_match.group(1))
            if skipped_match:
                tests_skipped = int(skipped_match.group(1))
            
            tests_run = tests_passed + tests_failed + tests_skipped
        
        # Parse coverage from JSON report if available
        coverage_file = Path('tests/reports/coverage.json')
        if coverage_file.exists():
            try:
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    coverage_percentage = coverage_data.get('totals', {}).get('percent_covered', 0.0)
            except Exception as e:
                self.logger.debug(f"Failed to parse coverage: {e}")
        
        return TestExecutionResult(
            suite_name=suite_name,
            status=status,
            duration=duration,
            tests_run=tests_run,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_skipped=tests_skipped,
            coverage_percentage=coverage_percentage,
            exit_code=result.returncode,
            output=output,
            error_output=error_output
        )


class TestPipeline:
    """Main testing pipeline coordinator."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.mcp = MCPCoordinator()
        self.runner = TestSuiteRunner(self.mcp)
        self.logger = logging.getLogger('TestPipeline')
        self.session_id = f"pipeline_{int(time.time())}"
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load pipeline configuration."""
        default_config = {
            'test_suites': [
                {
                    'name': 'unit_tests',
                    'path': 'tests/',
                    'markers': ['unit'],
                    'parallel': False,
                    'timeout': 300
                },
                {
                    'name': 'integration_tests',
                    'path': 'tests/',
                    'markers': ['integration'],
                    'parallel': False,
                    'timeout': 600
                },
                {
                    'name': 'behavior_verification_tests',
                    'path': 'tests/',
                    'markers': ['behavior_verification', 'mockist'],
                    'parallel': False,
                    'timeout': 300
                },
                {
                    'name': 'security_tests',
                    'path': 'tests/',
                    'markers': ['security'],
                    'parallel': False,
                    'timeout': 600
                }
            ],
            'coverage_threshold': 90.0,
            'fail_fast': False,
            'parallel_suites': True,
            'max_concurrent_suites': 4
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file) as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"Failed to load config {config_file}: {e}")
        
        return default_config
    
    async def run_pipeline(self) -> PipelineReport:
        """Execute the complete testing pipeline."""
        self.logger.info(f"Starting testing pipeline session: {session_id}")
        
        # Initialize MCP session
        await self.mcp.initialize_session(self.session_id)
        
        start_time = time.time()
        suite_results = []
        
        try:
            if self.config.get('parallel_suites', True):
                suite_results = await self._run_suites_parallel()
            else:
                suite_results = await self._run_suites_sequential()
            
            # Calculate totals
            total_duration = time.time() - start_time
            total_tests_run = sum(r.tests_run for r in suite_results)
            total_tests_passed = sum(r.tests_passed for r in suite_results)
            total_tests_failed = sum(r.tests_failed for r in suite_results)
            total_tests_skipped = sum(r.tests_skipped for r in suite_results)
            
            # Calculate overall coverage (weighted by test count)
            if total_tests_run > 0:
                overall_coverage = sum(
                    r.coverage_percentage * r.tests_run for r in suite_results
                ) / total_tests_run
            else:
                overall_coverage = 0.0
            
            # Get MCP metrics
            mcp_metrics = await self.mcp.finalize_session(self.session_id)
            
            # Performance metrics
            performance_metrics = {
                'average_suite_duration': total_duration / len(suite_results) if suite_results else 0,
                'tests_per_second': total_tests_run / total_duration if total_duration > 0 else 0,
                'coverage_efficiency': overall_coverage / (total_duration / 60) if total_duration > 0 else 0
            }
            
            return PipelineReport(
                total_duration=total_duration,
                total_tests_run=total_tests_run,
                total_tests_passed=total_tests_passed,
                total_tests_failed=total_tests_failed,
                total_tests_skipped=total_tests_skipped,
                overall_coverage=overall_coverage,
                suite_results=suite_results,
                mcp_coordination_metrics=mcp_metrics,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            await self.mcp.finalize_session(self.session_id)
            raise
    
    async def _run_suites_parallel(self) -> List[TestExecutionResult]:
        """Run test suites in parallel."""
        max_concurrent = self.config.get('max_concurrent_suites', 4)
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(suite_config):
            async with semaphore:
                return await self.runner.run_suite(suite_config)
        
        tasks = [
            run_with_semaphore(suite_config)
            for suite_config in self.config['test_suites']
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        suite_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                suite_name = self.config['test_suites'][i]['name']
                self.logger.error(f"Suite {suite_name} failed with exception: {result}")
                suite_results.append(TestExecutionResult(
                    suite_name=suite_name,
                    status='error',
                    duration=0.0,
                    tests_run=0,
                    tests_passed=0,
                    tests_failed=0,
                    tests_skipped=0,
                    coverage_percentage=0.0,
                    exit_code=-1,
                    output='',
                    error_output=str(result)
                ))
            else:
                suite_results.append(result)
        
        return suite_results
    
    async def _run_suites_sequential(self) -> List[TestExecutionResult]:
        """Run test suites sequentially."""
        results = []
        
        for suite_config in self.config['test_suites']:
            result = await self.runner.run_suite(suite_config)
            results.append(result)
            
            # Check fail-fast
            if self.config.get('fail_fast', False) and result.status == 'failed':
                self.logger.info("Stopping pipeline due to fail-fast configuration")
                break
        
        return results
    
    def generate_report(self, report: PipelineReport) -> str:
        """Generate comprehensive test pipeline report."""
        lines = [
            "TDD London School Test Pipeline Report",
            "=" * 50,
            "",
            f"Execution Time: {report.total_duration:.2f} seconds",
            f"Total Tests: {report.total_tests_run}",
            f"Passed: {report.total_tests_passed}",
            f"Failed: {report.total_tests_failed}",
            f"Skipped: {report.total_tests_skipped}",
            f"Overall Coverage: {report.overall_coverage:.1f}%",
            "",
            "Suite Results:",
            "-" * 30
        ]
        
        for result in report.suite_results:
            lines.extend([
                f"Suite: {result.suite_name}",
                f"  Status: {result.status}",
                f"  Duration: {result.duration:.2f}s",
                f"  Tests: {result.tests_run} (P:{result.tests_passed}, F:{result.tests_failed}, S:{result.tests_skipped})",
                f"  Coverage: {result.coverage_percentage:.1f}%",
                ""
            ])
        
        # Performance metrics
        lines.extend([
            "Performance Metrics:",
            "-" * 20,
            f"Average suite duration: {report.performance_metrics.get('average_suite_duration', 0):.2f}s",
            f"Tests per second: {report.performance_metrics.get('tests_per_second', 0):.2f}",
            f"Coverage efficiency: {report.performance_metrics.get('coverage_efficiency', 0):.2f}",
            ""
        ])
        
        # MCP coordination metrics
        if report.mcp_coordination_metrics:
            lines.extend([
                "MCP Coordination Metrics:",
                "-" * 25,
            ])
            for key, value in report.mcp_coordination_metrics.items():
                lines.append(f"{key}: {value}")
            lines.append("")
        
        # Coverage threshold check
        threshold = self.config.get('coverage_threshold', 90.0)
        if report.overall_coverage < threshold:
            lines.extend([
                f"WARNING: Coverage {report.overall_coverage:.1f}% is below threshold {threshold}%",
                ""
            ])
        
        return "\\n".join(lines)


async def main():
    """Main entry point for test pipeline."""
    parser = argparse.ArgumentParser(
        description="TDD London School Test Pipeline"
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to pipeline configuration file'
    )
    parser.add_argument(
        '--output-report',
        type=str,
        default='tests/reports/pipeline_report.txt',
        help='Path to save the execution report'
    )
    parser.add_argument(
        '--json-output',
        type=str,
        help='Path to save JSON results'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run pipeline
    pipeline = TestPipeline(args.config)
    
    try:
        report = await pipeline.run_pipeline()
        
        # Generate text report
        text_report = pipeline.generate_report(report)
        print(text_report)
        
        # Save text report
        os.makedirs(Path(args.output_report).parent, exist_ok=True)
        with open(args.output_report, 'w') as f:
            f.write(text_report)
        
        # Save JSON report if requested
        if args.json_output:
            os.makedirs(Path(args.json_output).parent, exist_ok=True)
            with open(args.json_output, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
        
        # Exit with error code if tests failed
        if report.total_tests_failed > 0:
            sys.exit(1)
        
        # Check coverage threshold
        threshold = pipeline.config.get('coverage_threshold', 90.0)
        if report.overall_coverage < threshold:
            print(f"Coverage {report.overall_coverage:.1f}% below threshold {threshold}%")
            sys.exit(1)
        
        print("All tests passed and coverage threshold met!")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())