"""
Performance Benchmarker - Core Infrastructure

Implements comprehensive performance benchmarking and optimization analysis
for distributed systems and microservices architectures.
"""

import asyncio
import time
import json
import statistics
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
import websockets
import logging
import resource
import gc
import tracemalloc
import sys
import os

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    name: str
    duration: float
    throughput: float
    latency_stats: Dict[str, float]
    resource_usage: Dict[str, Any]
    success_rate: float
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    max_latency: float
    min_latency: float
    throughput: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    network_io: Dict[str, float]
    disk_io: Dict[str, float]

class ResourceMonitor:
    """Monitors system resource usage during benchmarks"""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.measurements = []
        self.process = psutil.Process()
        
    async def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.measurements = []
        
        while self.monitoring:
            measurement = {
                'timestamp': time.time(),
                'cpu_percent': self.process.cpu_percent(),
                'memory_info': self.process.memory_info(),
                'memory_percent': self.process.memory_percent(),
                'num_threads': self.process.num_threads(),
                'connections': len(self.process.connections()),
                'open_files': len(self.process.open_files()),
                'system_cpu': psutil.cpu_percent(interval=None),
                'system_memory': psutil.virtual_memory(),
                'disk_io': psutil.disk_io_counters(),
                'network_io': psutil.net_io_counters()
            }
            self.measurements.append(measurement)
            await asyncio.sleep(self.sampling_interval)
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Calculate resource usage statistics"""
        if not self.measurements:
            return {}
            
        cpu_values = [m['cpu_percent'] for m in self.measurements]
        memory_values = [m['memory_info'].rss for m in self.measurements]
        
        return {
            'cpu': {
                'avg': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values),
                'std': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
            },
            'memory': {
                'avg': statistics.mean(memory_values),
                'max': max(memory_values),
                'min': min(memory_values),
                'peak_mb': max(memory_values) / 1024 / 1024
            },
            'threads': {
                'avg': statistics.mean([m['num_threads'] for m in self.measurements]),
                'max': max([m['num_threads'] for m in self.measurements])
            },
            'connections': {
                'avg': statistics.mean([m['connections'] for m in self.measurements]),
                'max': max([m['connections'] for m in self.measurements])
            }
        }

class LatencyHistogram:
    """Tracks latency distribution"""
    
    def __init__(self):
        self.measurements = []
        self.bins = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]  # milliseconds
    
    def record(self, latency_ms: float):
        """Record a latency measurement"""
        self.measurements.append(latency_ms)
    
    def get_percentiles(self, percentiles: List[float]) -> Dict[float, float]:
        """Calculate latency percentiles"""
        if not self.measurements:
            return {p: 0 for p in percentiles}
        
        sorted_measurements = sorted(self.measurements)
        return {
            p: np.percentile(sorted_measurements, p) for p in percentiles
        }
    
    def get_histogram(self) -> Dict[str, int]:
        """Get latency histogram"""
        if not self.measurements:
            return {}
        
        histogram = {}
        for i, upper_bound in enumerate(self.bins):
            lower_bound = self.bins[i-1] if i > 0 else 0
            count = sum(1 for m in self.measurements if lower_bound < m <= upper_bound)
            histogram[f"{lower_bound}-{upper_bound}ms"] = count
        
        # Count measurements above highest bin
        above_max = sum(1 for m in self.measurements if m > self.bins[-1])
        if above_max > 0:
            histogram[f">{self.bins[-1]}ms"] = above_max
        
        return histogram

class PerformanceBenchmarker:
    """
    Core performance benchmarking framework for distributed consensus protocols
    and microservices architectures.
    """
    
    def __init__(self):
        self.benchmarks = {}
        self.resource_monitor = ResourceMonitor()
        self.latency_histogram = LatencyHistogram()
        self.results_history = []
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup benchmark logging"""
        logger = logging.getLogger('performance_benchmarker')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def register_benchmark(self, name: str, benchmark_func: Callable) -> None:
        """Register a benchmark function"""
        self.benchmarks[name] = benchmark_func
        self.logger.info(f"Registered benchmark: {name}")
    
    async def run_benchmark(self, name: str, config: Dict[str, Any]) -> BenchmarkResult:
        """Run a specific benchmark"""
        if name not in self.benchmarks:
            raise ValueError(f"Benchmark '{name}' not found")
        
        self.logger.info(f"Starting benchmark: {name}")
        
        # Start resource monitoring
        monitor_task = asyncio.create_task(self.resource_monitor.start_monitoring())
        
        # Start memory tracing
        tracemalloc.start()
        gc.collect()  # Clean up before benchmark
        
        start_time = time.time()
        
        try:
            # Run the benchmark
            result = await self.benchmarks[name](config)
            
            # Stop monitoring
            self.resource_monitor.stop_monitoring()
            monitor_task.cancel()
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Get memory stats
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Get resource stats
            resource_stats = self.resource_monitor.get_stats()
            resource_stats.update({
                'memory_traced': {
                    'current_mb': current / 1024 / 1024,
                    'peak_mb': peak / 1024 / 1024
                }
            })
            
            # Create benchmark result
            benchmark_result = BenchmarkResult(
                name=name,
                duration=duration,
                throughput=result.get('throughput', 0),
                latency_stats=result.get('latency_stats', {}),
                resource_usage=resource_stats,
                success_rate=result.get('success_rate', 0),
                timestamp=datetime.now().isoformat(),
                metadata=result.get('metadata', {})
            )
            
            self.results_history.append(benchmark_result)
            self.logger.info(f"Completed benchmark: {name} in {duration:.2f}s")
            
            return benchmark_result
            
        except Exception as e:
            self.resource_monitor.stop_monitoring()
            monitor_task.cancel()
            tracemalloc.stop()
            self.logger.error(f"Benchmark failed: {name} - {e}")
            raise
    
    async def run_comparative_benchmark(self, 
                                       monolithic_config: Dict[str, Any],
                                       microservices_config: Dict[str, Any],
                                       benchmark_names: List[str]) -> Dict[str, Any]:
        """Run comparative benchmarks between architectures"""
        results = {
            'monolithic': {},
            'microservices': {},
            'comparison': {}
        }
        
        self.logger.info("Starting comparative benchmark analysis")
        
        # Run monolithic benchmarks
        self.logger.info("Benchmarking monolithic architecture")
        for name in benchmark_names:
            result = await self.run_benchmark(name, monolithic_config)
            results['monolithic'][name] = result
        
        # Clean up between runs
        gc.collect()
        await asyncio.sleep(2)
        
        # Run microservices benchmarks
        self.logger.info("Benchmarking microservices architecture")
        for name in benchmark_names:
            result = await self.run_benchmark(name, microservices_config)
            results['microservices'][name] = result
        
        # Generate comparison analysis
        results['comparison'] = self._generate_comparison_analysis(
            results['monolithic'], results['microservices']
        )
        
        return results
    
    def _generate_comparison_analysis(self, 
                                    mono_results: Dict[str, BenchmarkResult],
                                    micro_results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        """Generate comparison analysis between architectures"""
        analysis = {}
        
        for name in mono_results.keys():
            if name in micro_results:
                mono = mono_results[name]
                micro = micro_results[name]
                
                # Calculate performance deltas
                throughput_delta = ((micro.throughput - mono.throughput) / mono.throughput * 100) if mono.throughput > 0 else 0
                
                # Memory usage comparison
                mono_memory = mono.resource_usage.get('memory', {}).get('peak_mb', 0)
                micro_memory = micro.resource_usage.get('memory', {}).get('peak_mb', 0)
                memory_delta = ((micro_memory - mono_memory) / mono_memory * 100) if mono_memory > 0 else 0
                
                # Latency comparison
                mono_p95 = mono.latency_stats.get('p95', 0)
                micro_p95 = micro.latency_stats.get('p95', 0)
                latency_delta = ((micro_p95 - mono_p95) / mono_p95 * 100) if mono_p95 > 0 else 0
                
                analysis[name] = {
                    'throughput_change_percent': round(throughput_delta, 2),
                    'memory_change_percent': round(memory_delta, 2),
                    'latency_change_percent': round(latency_delta, 2),
                    'performance_regression': throughput_delta < -5.0,  # >5% degradation
                    'memory_improvement': memory_delta < -10.0,  # >10% reduction
                    'latency_improvement': latency_delta < -5.0,  # >5% improvement
                    'overall_score': self._calculate_overall_score(
                        throughput_delta, memory_delta, latency_delta
                    )
                }
        
        return analysis
    
    def _calculate_overall_score(self, throughput_delta: float, 
                               memory_delta: float, latency_delta: float) -> str:
        """Calculate overall performance score"""
        score = 0
        
        # Throughput weight: 40%
        if throughput_delta > 10:
            score += 4
        elif throughput_delta > 0:
            score += 2
        elif throughput_delta > -5:
            score += 1
        
        # Memory weight: 30%
        if memory_delta < -20:
            score += 3
        elif memory_delta < -10:
            score += 2
        elif memory_delta < 0:
            score += 1
        
        # Latency weight: 30%
        if latency_delta < -10:
            score += 3
        elif latency_delta < -5:
            score += 2
        elif latency_delta < 0:
            score += 1
        
        if score >= 8:
            return "EXCELLENT"
        elif score >= 6:
            return "GOOD"
        elif score >= 4:
            return "FAIR"
        else:
            return "POOR"
    
    def save_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Save benchmark results to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert dataclasses to dict for JSON serialization
        serializable_results = {}
        for arch, arch_results in results.items():
            if arch == 'comparison':
                serializable_results[arch] = arch_results
            else:
                serializable_results[arch] = {
                    name: asdict(result) for name, result in arch_results.items()
                }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load benchmark results from file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def get_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate a performance report"""
        report = []
        report.append("=" * 80)
        report.append("PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        if 'comparison' in results:
            report.append("ARCHITECTURE COMPARISON SUMMARY")
            report.append("-" * 40)
            
            for benchmark, analysis in results['comparison'].items():
                report.append(f"\n{benchmark.upper()}:")
                report.append(f"  Overall Score: {analysis['overall_score']}")
                report.append(f"  Throughput Change: {analysis['throughput_change_percent']:+.2f}%")
                report.append(f"  Memory Change: {analysis['memory_change_percent']:+.2f}%")
                report.append(f"  Latency Change: {analysis['latency_change_percent']:+.2f}%")
                
                if analysis['performance_regression']:
                    report.append("  ⚠️  Performance regression detected!")
                if analysis['memory_improvement']:
                    report.append("  ✅ Significant memory improvement")
                if analysis['latency_improvement']:
                    report.append("  ✅ Latency improvement achieved")
        
        return "\n".join(report)

class ThroughputBenchmark:
    """Specialized throughput benchmark implementation"""
    
    def __init__(self, benchmarker: PerformanceBenchmarker):
        self.benchmarker = benchmarker
    
    async def training_throughput_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark training throughput (models/second)"""
        duration = config.get('duration', 60)  # seconds
        concurrent_models = config.get('concurrent_models', 10)
        model_size = config.get('model_size', 'small')
        
        models_processed = 0
        start_time = time.time()
        latencies = []
        
        async def process_model(model_id: int) -> float:
            """Simulate model training"""
            model_start = time.time()
            
            # Simulate different model sizes
            if model_size == 'small':
                await asyncio.sleep(0.1 + np.random.exponential(0.05))
            elif model_size == 'medium':
                await asyncio.sleep(0.5 + np.random.exponential(0.2))
            else:  # large
                await asyncio.sleep(2.0 + np.random.exponential(0.5))
            
            return time.time() - model_start
        
        # Run concurrent model training
        tasks = []
        model_id = 0
        
        while time.time() - start_time < duration:
            # Maintain concurrent_models tasks
            while len(tasks) < concurrent_models and time.time() - start_time < duration:
                task = asyncio.create_task(process_model(model_id))
                tasks.append(task)
                model_id += 1
            
            # Wait for at least one task to complete
            if tasks:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                
                for task in done:
                    latency = await task
                    latencies.append(latency * 1000)  # Convert to ms
                    models_processed += 1
                    self.benchmarker.latency_histogram.record(latency * 1000)
                
                tasks = list(pending)
        
        # Wait for remaining tasks
        if tasks:
            for task in tasks:
                latency = await task
                latencies.append(latency * 1000)
                models_processed += 1
        
        actual_duration = time.time() - start_time
        throughput = models_processed / actual_duration
        
        # Calculate latency statistics
        latency_stats = {
            'avg': statistics.mean(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'max': max(latencies),
            'min': min(latencies)
        } if latencies else {}
        
        return {
            'throughput': throughput,
            'latency_stats': latency_stats,
            'success_rate': 1.0,  # Assuming all models processed successfully
            'metadata': {
                'models_processed': models_processed,
                'actual_duration': actual_duration,
                'concurrent_models': concurrent_models,
                'model_size': model_size
            }
        }

class NetworkBenchmark:
    """Network-related benchmarks (WebSocket, API)"""
    
    def __init__(self, benchmarker: PerformanceBenchmarker):
        self.benchmarker = benchmarker
    
    async def websocket_latency_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark WebSocket round-trip latency"""
        ws_url = config.get('websocket_url', 'ws://localhost:8080')
        message_count = config.get('message_count', 1000)
        concurrent_connections = config.get('concurrent_connections', 10)
        
        latencies = []
        successful_messages = 0
        failed_messages = 0
        
        async def test_connection(conn_id: int, messages_per_conn: int):
            """Test a single WebSocket connection"""
            nonlocal successful_messages, failed_messages
            
            try:
                async with websockets.connect(ws_url) as websocket:
                    for i in range(messages_per_conn):
                        message = f"test_message_{conn_id}_{i}_{time.time()}"
                        
                        start_time = time.time()
                        await websocket.send(message)
                        response = await websocket.recv()
                        end_time = time.time()
                        
                        latency = (end_time - start_time) * 1000  # Convert to ms
                        latencies.append(latency)
                        self.benchmarker.latency_histogram.record(latency)
                        successful_messages += 1
                        
            except Exception as e:
                failed_messages += messages_per_conn
                self.benchmarker.logger.error(f"WebSocket connection {conn_id} failed: {e}")
        
        # Calculate messages per connection
        messages_per_conn = message_count // concurrent_connections
        
        # Run concurrent connections
        tasks = [
            test_connection(i, messages_per_conn) 
            for i in range(concurrent_connections)
        ]
        
        start_time = time.time()
        await asyncio.gather(*tasks, return_exceptions=True)
        total_duration = time.time() - start_time
        
        # Calculate statistics
        success_rate = successful_messages / (successful_messages + failed_messages) if (successful_messages + failed_messages) > 0 else 0
        throughput = successful_messages / total_duration if total_duration > 0 else 0
        
        latency_stats = {
            'avg': statistics.mean(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'max': max(latencies),
            'min': min(latencies)
        } if latencies else {}
        
        return {
            'throughput': throughput,
            'latency_stats': latency_stats,
            'success_rate': success_rate,
            'metadata': {
                'successful_messages': successful_messages,
                'failed_messages': failed_messages,
                'total_duration': total_duration,
                'concurrent_connections': concurrent_connections
            }
        }
    
    async def api_response_time_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark API response times"""
        base_url = config.get('api_url', 'http://localhost:8080')
        endpoints = config.get('endpoints', ['/health', '/api/status'])
        requests_per_endpoint = config.get('requests_per_endpoint', 100)
        concurrent_requests = config.get('concurrent_requests', 10)
        
        latencies = []
        successful_requests = 0
        failed_requests = 0
        
        async def make_request(session: aiohttp.ClientSession, endpoint: str):
            """Make a single API request"""
            nonlocal successful_requests, failed_requests
            
            try:
                start_time = time.time()
                async with session.get(f"{base_url}{endpoint}") as response:
                    await response.read()  # Ensure full response is read
                    end_time = time.time()
                    
                    latency = (end_time - start_time) * 1000  # Convert to ms
                    latencies.append(latency)
                    self.benchmarker.latency_histogram.record(latency)
                    
                    if response.status == 200:
                        successful_requests += 1
                    else:
                        failed_requests += 1
                        
            except Exception as e:
                failed_requests += 1
                self.benchmarker.logger.error(f"API request failed: {e}")
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=concurrent_requests)
        
        start_time = time.time()
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            # Create all requests
            tasks = []
            for endpoint in endpoints:
                for _ in range(requests_per_endpoint):
                    task = make_request(session, endpoint)
                    tasks.append(task)
            
            # Execute requests with concurrency control
            semaphore = asyncio.Semaphore(concurrent_requests)
            
            async def limited_request(task):
                async with semaphore:
                    return await task
            
            limited_tasks = [limited_request(task) for task in tasks]
            await asyncio.gather(*limited_tasks, return_exceptions=True)
        
        total_duration = time.time() - start_time
        
        # Calculate statistics
        total_requests = successful_requests + failed_requests
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        throughput = successful_requests / total_duration if total_duration > 0 else 0
        
        latency_stats = {
            'avg': statistics.mean(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'max': max(latencies),
            'min': min(latencies)
        } if latencies else {}
        
        return {
            'throughput': throughput,
            'latency_stats': latency_stats,
            'success_rate': success_rate,
            'metadata': {
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'total_duration': total_duration,
                'endpoints_tested': endpoints,
                'requests_per_endpoint': requests_per_endpoint
            }
        }

class ConcurrentRequestBenchmark:
    """Benchmark concurrent request handling capabilities"""
    
    def __init__(self, benchmarker: PerformanceBenchmarker):
        self.benchmarker = benchmarker
    
    async def concurrent_load_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark system under concurrent load"""
        base_url = config.get('api_url', 'http://localhost:8080')
        max_concurrent = config.get('max_concurrent', 100)
        ramp_up_duration = config.get('ramp_up_duration', 30)  # seconds
        steady_duration = config.get('steady_duration', 60)  # seconds
        endpoint = config.get('endpoint', '/api/health')
        
        latencies = []
        successful_requests = 0
        failed_requests = 0
        concurrent_requests = []  # Track concurrent request count over time
        
        async def make_request(session: aiohttp.ClientSession, request_id: int):
            """Make a single concurrent request"""
            nonlocal successful_requests, failed_requests
            
            try:
                start_time = time.time()
                async with session.get(f"{base_url}{endpoint}") as response:
                    await response.read()
                    end_time = time.time()
                    
                    latency = (end_time - start_time) * 1000
                    latencies.append(latency)
                    self.benchmarker.latency_histogram.record(latency)
                    
                    if response.status == 200:
                        successful_requests += 1
                    else:
                        failed_requests += 1
                        
            except Exception as e:
                failed_requests += 1
                self.benchmarker.logger.error(f"Concurrent request {request_id} failed: {e}")
        
        # Phase 1: Ramp up
        self.benchmarker.logger.info("Starting ramp-up phase")
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=max_concurrent * 2)
        
        start_time = time.time()
        active_tasks = set()
        request_id = 0
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            # Ramp-up phase
            ramp_start = time.time()
            while time.time() - ramp_start < ramp_up_duration:
                current_concurrent = int(max_concurrent * (time.time() - ramp_start) / ramp_up_duration)
                concurrent_requests.append((time.time() - start_time, current_concurrent))
                
                # Add tasks to reach target concurrency
                while len(active_tasks) < current_concurrent:
                    task = asyncio.create_task(make_request(session, request_id))
                    active_tasks.add(task)
                    request_id += 1
                
                # Remove completed tasks
                done_tasks = [task for task in active_tasks if task.done()]
                for task in done_tasks:
                    active_tasks.remove(task)
                    await task  # Handle any exceptions
                
                await asyncio.sleep(0.1)
            
            # Phase 2: Steady state
            self.benchmarker.logger.info(f"Steady state phase with {max_concurrent} concurrent requests")
            steady_start = time.time()
            while time.time() - steady_start < steady_duration:
                concurrent_requests.append((time.time() - start_time, len(active_tasks)))
                
                # Maintain target concurrency
                while len(active_tasks) < max_concurrent:
                    task = asyncio.create_task(make_request(session, request_id))
                    active_tasks.add(task)
                    request_id += 1
                
                # Remove completed tasks
                done_tasks = [task for task in active_tasks if task.done()]
                for task in done_tasks:
                    active_tasks.remove(task)
                    await task
                
                await asyncio.sleep(0.1)
            
            # Phase 3: Wait for remaining tasks
            self.benchmarker.logger.info("Waiting for remaining tasks to complete")
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
        
        total_duration = time.time() - start_time
        
        # Calculate statistics
        total_requests = successful_requests + failed_requests
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        throughput = successful_requests / total_duration if total_duration > 0 else 0
        
        latency_stats = {
            'avg': statistics.mean(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'max': max(latencies),
            'min': min(latencies)
        } if latencies else {}
        
        return {
            'throughput': throughput,
            'latency_stats': latency_stats,
            'success_rate': success_rate,
            'metadata': {
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'total_duration': total_duration,
                'max_concurrent': max_concurrent,
                'ramp_up_duration': ramp_up_duration,
                'steady_duration': steady_duration,
                'concurrent_timeline': concurrent_requests
            }
        }

# Export main classes
__all__ = [
    'PerformanceBenchmarker',
    'BenchmarkResult',
    'PerformanceMetrics',
    'ResourceMonitor',
    'ThroughputBenchmark',
    'NetworkBenchmark',
    'ConcurrentRequestBenchmark'
]