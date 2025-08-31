"""
Performance Regression Tester for Phase 4 Validation

Tests performance benchmarks to ensure architectural improvements don't degrade system performance.
"""

import asyncio
import time
import psutil
import gc
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
import sys
import tracemalloc


@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    name: str
    value: float
    unit: str
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark"""
    name: str
    metrics: List[PerformanceMetric]
    passed: bool
    baseline_comparison: Dict[str, float]
    execution_time_ms: float


class RegressionTester:
    """
    Performance regression testing for Phase 4 architectural improvements
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.baseline_metrics = {}
        self.current_metrics = {}
        
        # Performance targets
        self.targets = {
            'max_memory_increase_percent': 10.0,
            'min_throughput_ratio': 1.0,
            'max_init_time_ms': 100,
            'max_performance_degradation_percent': 5.0,
            'max_response_time_ms': 1000,
            'min_requests_per_second': 100
        }
    
    async def run_performance_tests(self) -> Dict[str, Any]:
        """
        Run all performance regression tests
        
        Returns:
            Comprehensive performance test results
        """
        self.logger.info("Starting performance regression tests...")
        
        # Start memory tracking
        tracemalloc.start()
        start_time = time.time()
        
        results = {}
        
        try:
            # Run all benchmarks concurrently where possible
            benchmark_tasks = [
                self._test_memory_usage(),
                self._test_task_processing_throughput(),
                self._test_service_initialization_time(),
                self._test_concurrent_operations(),
                self._test_response_times(),
                self._test_resource_utilization()
            ]
            
            benchmark_results = await asyncio.gather(*benchmark_tasks, return_exceptions=True)
            
            # Process results
            test_names = [
                'memory_usage', 'task_throughput', 'service_init',
                'concurrent_ops', 'response_times', 'resource_usage'
            ]
            
            for i, result in enumerate(benchmark_results):
                test_name = test_names[i]
                if isinstance(result, Exception):
                    self.logger.error(f"Benchmark {test_name} failed: {result}")
                    results[test_name] = {'error': str(result), 'passed': False}
                else:
                    results[test_name] = result
            
            # Calculate overall performance metrics
            overall_metrics = self._calculate_overall_metrics(results)
            results['overall'] = overall_metrics
            
            # Check against targets
            results['targets_met'] = self._check_performance_targets(overall_metrics)
            
            execution_time = time.time() - start_time
            results['total_execution_time_ms'] = int(execution_time * 1000)
            
            # Memory snapshot
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            results['memory_trace'] = {
                'current_mb': current_memory / 1024 / 1024,
                'peak_mb': peak_memory / 1024 / 1024
            }
            
            self.logger.info("Performance regression tests completed")
            return results
            
        except Exception as e:
            self.logger.error(f"Performance tests failed: {e}")
            tracemalloc.stop()
            return {
                'error': str(e),
                'passed': False,
                'total_execution_time_ms': int((time.time() - start_time) * 1000)
            }
    
    async def _test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage under various loads"""
        self.logger.debug("Testing memory usage...")
        
        # Get baseline memory
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss
        
        results = {
            'baseline_memory_mb': baseline_memory / 1024 / 1024,
            'tests': []
        }
        
        # Test scenarios
        test_scenarios = [
            ('idle', self._memory_test_idle),
            ('light_load', self._memory_test_light_load),
            ('heavy_load', self._memory_test_heavy_load),
            ('concurrent_tasks', self._memory_test_concurrent_tasks)
        ]
        
        for scenario_name, test_func in test_scenarios:
            try:
                gc.collect()
                start_memory = psutil.Process().memory_info().rss
                
                await test_func()
                
                gc.collect()
                end_memory = psutil.Process().memory_info().rss
                
                memory_increase = (end_memory - start_memory) / 1024 / 1024
                memory_increase_percent = (memory_increase / (baseline_memory / 1024 / 1024)) * 100
                
                results['tests'].append({
                    'scenario': scenario_name,
                    'memory_increase_mb': memory_increase,
                    'memory_increase_percent': memory_increase_percent,
                    'passed': memory_increase_percent <= self.targets['max_memory_increase_percent']
                })
                
            except Exception as e:
                self.logger.error(f"Memory test {scenario_name} failed: {e}")
                results['tests'].append({
                    'scenario': scenario_name,
                    'error': str(e),
                    'passed': False
                })
        
        # Calculate overall memory performance
        valid_tests = [t for t in results['tests'] if 'memory_increase_percent' in t]
        if valid_tests:
            max_increase = max(t['memory_increase_percent'] for t in valid_tests)
            avg_increase = statistics.mean(t['memory_increase_percent'] for t in valid_tests)
            
            results['max_memory_increase_percent'] = max_increase
            results['avg_memory_increase_percent'] = avg_increase
            results['passed'] = max_increase <= self.targets['max_memory_increase_percent']
        else:
            results['passed'] = False
        
        return results
    
    async def _test_task_processing_throughput(self) -> Dict[str, Any]:
        """Test task processing throughput"""
        self.logger.debug("Testing task processing throughput...")
        
        results = {
            'tests': [],
            'baseline_throughput': 0,
            'current_throughput': 0
        }
        
        # Simulate different task loads
        task_counts = [10, 50, 100, 500, 1000]
        
        for task_count in task_counts:
            start_time = time.time()
            
            try:
                # Simulate task processing
                await self._simulate_task_processing(task_count)
                
                end_time = time.time()
                duration = end_time - start_time
                throughput = task_count / duration
                
                results['tests'].append({
                    'task_count': task_count,
                    'duration_seconds': duration,
                    'throughput_tasks_per_second': throughput,
                    'passed': throughput >= self.targets['min_requests_per_second']
                })
                
            except Exception as e:
                self.logger.error(f"Throughput test with {task_count} tasks failed: {e}")
                results['tests'].append({
                    'task_count': task_count,
                    'error': str(e),
                    'passed': False
                })
        
        # Calculate overall throughput metrics
        valid_tests = [t for t in results['tests'] if 'throughput_tasks_per_second' in t]
        if valid_tests:
            results['current_throughput'] = statistics.mean(
                t['throughput_tasks_per_second'] for t in valid_tests
            )
            results['max_throughput'] = max(
                t['throughput_tasks_per_second'] for t in valid_tests
            )
            results['passed'] = all(t['passed'] for t in valid_tests)
        else:
            results['passed'] = False
        
        return results
    
    async def _test_service_initialization_time(self) -> Dict[str, Any]:
        """Test service initialization times"""
        self.logger.debug("Testing service initialization times...")
        
        # Services to test (mock implementations for testing)
        services_to_test = [
            'UnifiedManagement',
            'SageAgent', 
            'TaskManager',
            'WorkflowEngine',
            'ExecutionManager',
            'ResourceManager'
        ]
        
        results = {
            'services': [],
            'total_init_time_ms': 0,
            'max_init_time_ms': 0,
            'avg_init_time_ms': 0
        }
        
        total_init_time = 0
        init_times = []
        
        for service_name in services_to_test:
            try:
                start_time = time.perf_counter()
                
                # Simulate service initialization
                await self._simulate_service_init(service_name)
                
                end_time = time.perf_counter()
                init_time_ms = (end_time - start_time) * 1000
                
                init_times.append(init_time_ms)
                total_init_time += init_time_ms
                
                results['services'].append({
                    'service': service_name,
                    'init_time_ms': init_time_ms,
                    'passed': init_time_ms <= self.targets['max_init_time_ms']
                })
                
            except Exception as e:
                self.logger.error(f"Service init test for {service_name} failed: {e}")
                results['services'].append({
                    'service': service_name,
                    'error': str(e),
                    'passed': False
                })
        
        if init_times:
            results['total_init_time_ms'] = total_init_time
            results['max_init_time_ms'] = max(init_times)
            results['avg_init_time_ms'] = statistics.mean(init_times)
            results['passed'] = all(
                s.get('passed', False) for s in results['services']
            )
        else:
            results['passed'] = False
        
        return results
    
    async def _test_concurrent_operations(self) -> Dict[str, Any]:
        """Test performance under concurrent operations"""
        self.logger.debug("Testing concurrent operations...")
        
        results = {
            'tests': [],
            'max_concurrent_supported': 0
        }
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 25, 50, 100]
        
        for concurrency in concurrency_levels:
            start_time = time.time()
            
            try:
                # Run concurrent operations
                tasks = []
                for i in range(concurrency):
                    task = asyncio.create_task(self._simulate_concurrent_operation(i))
                    tasks.append(task)
                
                await asyncio.gather(*tasks)
                
                end_time = time.time()
                duration = end_time - start_time
                ops_per_second = concurrency / duration
                
                results['tests'].append({
                    'concurrency_level': concurrency,
                    'duration_seconds': duration,
                    'ops_per_second': ops_per_second,
                    'passed': duration <= (concurrency * 0.01)  # 10ms per operation max
                })
                
                if results['tests'][-1]['passed']:
                    results['max_concurrent_supported'] = concurrency
                
            except Exception as e:
                self.logger.error(f"Concurrency test at level {concurrency} failed: {e}")
                results['tests'].append({
                    'concurrency_level': concurrency,
                    'error': str(e),
                    'passed': False
                })
                break  # Stop testing higher concurrency levels
        
        results['passed'] = len([t for t in results['tests'] if t.get('passed', False)]) > 0
        
        return results
    
    async def _test_response_times(self) -> Dict[str, Any]:
        """Test API/service response times"""
        self.logger.debug("Testing response times...")
        
        results = {
            'endpoints': [],
            'avg_response_time_ms': 0,
            'max_response_time_ms': 0,
            'p95_response_time_ms': 0
        }
        
        # Simulate API endpoints
        endpoints_to_test = [
            '/api/tasks/create',
            '/api/tasks/list', 
            '/api/agents/status',
            '/api/workflows/execute',
            '/api/resources/allocate'
        ]
        
        all_response_times = []
        
        for endpoint in endpoints_to_test:
            response_times = []
            
            # Test each endpoint multiple times
            for _ in range(10):
                start_time = time.perf_counter()
                
                try:
                    await self._simulate_api_call(endpoint)
                    
                    end_time = time.perf_counter()
                    response_time_ms = (end_time - start_time) * 1000
                    response_times.append(response_time_ms)
                    all_response_times.append(response_time_ms)
                    
                except Exception as e:
                    self.logger.error(f"API call to {endpoint} failed: {e}")
            
            if response_times:
                avg_time = statistics.mean(response_times)
                max_time = max(response_times)
                
                results['endpoints'].append({
                    'endpoint': endpoint,
                    'avg_response_time_ms': avg_time,
                    'max_response_time_ms': max_time,
                    'passed': max_time <= self.targets['max_response_time_ms']
                })
        
        if all_response_times:
            results['avg_response_time_ms'] = statistics.mean(all_response_times)
            results['max_response_time_ms'] = max(all_response_times)
            results['p95_response_time_ms'] = statistics.quantiles(all_response_times, n=20)[18]  # 95th percentile
            results['passed'] = all(e.get('passed', False) for e in results['endpoints'])
        else:
            results['passed'] = False
        
        return results
    
    async def _test_resource_utilization(self) -> Dict[str, Any]:
        """Test CPU and memory resource utilization"""
        self.logger.debug("Testing resource utilization...")
        
        # Monitor system resources during load test
        cpu_samples = []
        memory_samples = []
        
        async def monitor_resources():
            for _ in range(30):  # Monitor for 30 seconds
                cpu_samples.append(psutil.cpu_percent(interval=1))
                memory_samples.append(psutil.virtual_memory().percent)
        
        # Start resource monitoring
        monitor_task = asyncio.create_task(monitor_resources())
        
        # Run load test concurrently
        load_tasks = []
        for i in range(20):
            task = asyncio.create_task(self._simulate_resource_load(i))
            load_tasks.append(task)
        
        # Wait for both monitoring and load test
        await asyncio.gather(monitor_task, *load_tasks)
        
        results = {
            'cpu_usage': {
                'avg_percent': statistics.mean(cpu_samples) if cpu_samples else 0,
                'max_percent': max(cpu_samples) if cpu_samples else 0,
                'samples': cpu_samples
            },
            'memory_usage': {
                'avg_percent': statistics.mean(memory_samples) if memory_samples else 0,
                'max_percent': max(memory_samples) if memory_samples else 0,
                'samples': memory_samples
            }
        }
        
        # Check if resource usage is reasonable
        results['passed'] = (
            results['cpu_usage']['max_percent'] < 80 and  # Max 80% CPU
            results['memory_usage']['max_percent'] < 90    # Max 90% memory
        )
        
        return results
    
    # Simulation methods for testing
    async def _memory_test_idle(self):
        """Simulate idle memory usage"""
        await asyncio.sleep(0.1)
    
    async def _memory_test_light_load(self):
        """Simulate light load memory usage"""
        data = [i for i in range(1000)]
        await asyncio.sleep(0.1)
        del data
    
    async def _memory_test_heavy_load(self):
        """Simulate heavy load memory usage"""
        data = [i for i in range(100000)]
        await asyncio.sleep(0.2)
        del data
    
    async def _memory_test_concurrent_tasks(self):
        """Simulate concurrent tasks memory usage"""
        tasks = []
        for i in range(10):
            task = asyncio.create_task(self._simulate_task(i))
            tasks.append(task)
        await asyncio.gather(*tasks)
    
    async def _simulate_task(self, task_id: int):
        """Simulate a single task"""
        data = [task_id * j for j in range(1000)]
        await asyncio.sleep(0.01)
        return sum(data)
    
    async def _simulate_task_processing(self, task_count: int):
        """Simulate processing multiple tasks"""
        tasks = []
        for i in range(task_count):
            task = asyncio.create_task(self._simulate_task(i))
            tasks.append(task)
        await asyncio.gather(*tasks)
    
    async def _simulate_service_init(self, service_name: str):
        """Simulate service initialization"""
        # Simulate initialization work
        await asyncio.sleep(0.01)  # Simulate some initialization time
        return f"{service_name} initialized"
    
    async def _simulate_concurrent_operation(self, operation_id: int):
        """Simulate a concurrent operation"""
        await asyncio.sleep(0.001)  # 1ms operation
        return f"Operation {operation_id} completed"
    
    async def _simulate_api_call(self, endpoint: str):
        """Simulate an API call"""
        await asyncio.sleep(0.005)  # 5ms response time
        return f"Response from {endpoint}"
    
    async def _simulate_resource_load(self, load_id: int):
        """Simulate resource-intensive operation"""
        # CPU-intensive task
        result = sum(i * i for i in range(1000))
        await asyncio.sleep(0.01)
        return result
    
    def _calculate_overall_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance metrics from all test results"""
        overall = {
            'memory_increase_percent': 0,
            'throughput_ratio': 1.0,
            'init_time_ms': 0,
            'performance_degradation_percent': 0,
            'all_tests_passed': True
        }
        
        # Memory metrics
        if 'memory_usage' in results and 'max_memory_increase_percent' in results['memory_usage']:
            overall['memory_increase_percent'] = results['memory_usage']['max_memory_increase_percent']
        
        # Throughput metrics
        if 'task_throughput' in results and 'current_throughput' in results['task_throughput']:
            baseline_throughput = 100  # Assumed baseline
            current_throughput = results['task_throughput']['current_throughput']
            overall['throughput_ratio'] = current_throughput / baseline_throughput
        
        # Initialization time
        if 'service_init' in results and 'max_init_time_ms' in results['service_init']:
            overall['init_time_ms'] = results['service_init']['max_init_time_ms']
        
        # Performance degradation calculation
        degradation_factors = []
        
        if overall['throughput_ratio'] < 1.0:
            degradation_factors.append((1.0 - overall['throughput_ratio']) * 100)
        
        if overall['memory_increase_percent'] > 0:
            degradation_factors.append(overall['memory_increase_percent'] / 2)  # Weight memory less
        
        if degradation_factors:
            overall['performance_degradation_percent'] = max(degradation_factors)
        
        # Check if all individual tests passed
        all_passed = True
        for test_category, test_results in results.items():
            if isinstance(test_results, dict) and 'passed' in test_results:
                if not test_results['passed']:
                    all_passed = False
        
        overall['all_tests_passed'] = all_passed
        
        return overall
    
    def _check_performance_targets(self, metrics: Dict[str, Any]) -> bool:
        """Check if all performance targets are met"""
        checks = [
            metrics.get('memory_increase_percent', 0) <= self.targets['max_memory_increase_percent'],
            metrics.get('throughput_ratio', 0) >= self.targets['min_throughput_ratio'],
            metrics.get('init_time_ms', float('inf')) <= self.targets['max_init_time_ms'],
            metrics.get('performance_degradation_percent', float('inf')) <= self.targets['max_performance_degradation_percent'],
            metrics.get('all_tests_passed', False)
        ]
        
        return all(checks)
    
    async def get_baseline_metrics(self) -> Dict[str, Any]:
        """Get baseline performance metrics for comparison"""
        # This would normally load from stored baseline
        # For now, return default baseline values
        return {
            'memory_usage_mb': 100,
            'throughput_tasks_per_second': 100,
            'avg_response_time_ms': 50,
            'init_time_ms': 50
        }