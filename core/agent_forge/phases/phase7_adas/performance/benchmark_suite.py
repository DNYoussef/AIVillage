"""
ADAS Benchmark Suite - Comprehensive Performance Testing
Automotive-grade benchmarking with regression detection and optimization
"""

import time
import statistics
import json
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import warnings
import traceback


class BenchmarkType(Enum):
    """Types of benchmarks for ADAS systems"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    MEMORY = "memory"
    POWER = "power"
    THERMAL = "thermal"
    STRESS = "stress"
    REGRESSION = "regression"


class Severity(Enum):
    """Issue severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BenchmarkResult:
    """Individual benchmark result"""
    name: str
    benchmark_type: BenchmarkType
    value: float
    unit: str
    target: float
    passed: bool
    timestamp: float
    duration_ms: float
    metadata: Dict[str, Any]


@dataclass
class PerformanceProfile:
    """Performance profile for specific hardware/scenario"""
    platform: str
    scenario: str
    baseline_latency_ms: float
    baseline_throughput_fps: float
    baseline_accuracy_percent: float
    baseline_memory_mb: float
    baseline_power_watts: float
    confidence_interval: float = 0.95


@dataclass
class Bottleneck:
    """Performance bottleneck identification"""
    component: str
    severity: Severity
    impact_percent: float
    description: str
    recommendations: List[str]
    metrics: Dict[str, float]


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    category: str
    priority: int
    description: str
    expected_improvement: str
    implementation_effort: str
    risk_level: str
    code_changes: List[str]


class BenchmarkSuite:
    """
    Comprehensive ADAS performance benchmark suite
    Supports automotive ECUs, NVIDIA Drive, Qualcomm Snapdragon Ride
    """
    
    def __init__(self, platform: str = "generic"):
        self.platform = platform
        self.results: List[BenchmarkResult] = []
        self.baselines: Dict[str, PerformanceProfile] = {}
        self.bottlenecks: List[Bottleneck] = []
        self.recommendations: List[OptimizationRecommendation] = []
        
        # Benchmark configuration
        self.max_workers = min(4, psutil.cpu_count())
        self.timeout_seconds = 300
        self.warmup_iterations = 3
        self.measurement_iterations = 10
        
        # Performance thresholds
        self.thresholds = self._load_platform_thresholds()
        
        # Regression detection
        self.regression_threshold_percent = 5.0
        self.regression_history: List[BenchmarkResult] = []
        
        print(f"BenchmarkSuite initialized for {platform}")
        print(f"Thresholds: {self.thresholds}")
    
    def _load_platform_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load performance thresholds for platform"""
        base_thresholds = {
            "latency": {
                "inference_ms": 10.0,
                "preprocessing_ms": 2.0,
                "postprocessing_ms": 1.0,
                "end_to_end_ms": 15.0
            },
            "throughput": {
                "frames_per_second": 30.0,
                "detections_per_second": 100.0,
                "decisions_per_second": 50.0
            },
            "accuracy": {
                "object_detection_percent": 95.0,
                "lane_detection_percent": 98.0,
                "classification_percent": 92.0
            },
            "memory": {
                "peak_usage_mb": 2048.0,
                "steady_state_mb": 1024.0,
                "memory_growth_mb_per_hour": 10.0
            },
            "power": {
                "average_watts": 50.0,
                "peak_watts": 80.0,
                "idle_watts": 15.0
            }
        }
        
        # Platform-specific adjustments
        if self.platform == "nvidia_drive":
            base_thresholds["latency"]["inference_ms"] = 5.0
            base_thresholds["memory"]["peak_usage_mb"] = 4096.0
            base_thresholds["power"]["average_watts"] = 100.0
        elif self.platform == "snapdragon_ride":
            base_thresholds["latency"]["inference_ms"] = 8.0
            base_thresholds["memory"]["peak_usage_mb"] = 1536.0
            base_thresholds["power"]["average_watts"] = 30.0
        elif self.platform == "automotive_ecu":
            base_thresholds["latency"]["inference_ms"] = 20.0
            base_thresholds["memory"]["peak_usage_mb"] = 512.0
            base_thresholds["power"]["average_watts"] = 20.0
        
        return base_thresholds
    
    async def run_comprehensive_benchmark(self, 
                                        test_functions: Dict[str, Callable],
                                        scenarios: List[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive benchmark suite
        
        Args:
            test_functions: Dictionary of test name -> test function
            scenarios: List of scenarios to test (optional)
            
        Returns:
            Comprehensive benchmark results
        """
        print("Starting comprehensive ADAS benchmark suite...")
        
        # Default scenarios if not provided
        if scenarios is None:
            scenarios = ["urban", "highway", "parking", "night", "rain"]
        
        all_results = {}
        
        for scenario in scenarios:
            print(f"\n=== Benchmarking scenario: {scenario} ===")
            scenario_results = await self._run_scenario_benchmark(test_functions, scenario)
            all_results[scenario] = scenario_results
        
        # Analyze results across scenarios
        analysis = self._analyze_comprehensive_results(all_results)
        
        # Generate recommendations
        self._generate_optimization_recommendations(all_results)
        
        return {
            "results": all_results,
            "analysis": analysis,
            "bottlenecks": [asdict(b) for b in self.bottlenecks],
            "recommendations": [asdict(r) for r in self.recommendations],
            "platform": self.platform,
            "timestamp": time.time()
        }
    
    async def _run_scenario_benchmark(self, 
                                    test_functions: Dict[str, Callable],
                                    scenario: str) -> Dict[str, Any]:
        """Run benchmark for specific scenario"""
        scenario_results = {
            "latency": {},
            "throughput": {},
            "accuracy": {},
            "memory": {},
            "power": {},
            "regression": {}
        }
        
        # Run different types of benchmarks
        benchmark_types = [
            (BenchmarkType.LATENCY, self._benchmark_latency),
            (BenchmarkType.THROUGHPUT, self._benchmark_throughput),
            (BenchmarkType.ACCURACY, self._benchmark_accuracy),
            (BenchmarkType.MEMORY, self._benchmark_memory),
            (BenchmarkType.POWER, self._benchmark_power),
            (BenchmarkType.REGRESSION, self._benchmark_regression)
        ]
        
        for benchmark_type, benchmark_func in benchmark_types:
            try:
                print(f"Running {benchmark_type.value} benchmark...")
                results = await benchmark_func(test_functions, scenario)
                scenario_results[benchmark_type.value] = results
            except Exception as e:
                print(f"Benchmark {benchmark_type.value} failed: {e}")
                scenario_results[benchmark_type.value] = {"error": str(e)}
        
        return scenario_results
    
    async def _benchmark_latency(self, 
                                test_functions: Dict[str, Callable],
                                scenario: str) -> Dict[str, Any]:
        """Benchmark latency performance"""
        latency_results = {}
        
        for test_name, test_func in test_functions.items():
            print(f"  Latency benchmark: {test_name}")
            
            # Warmup
            for _ in range(self.warmup_iterations):
                try:
                    await self._run_test_function(test_func, scenario)
                except:
                    pass
            
            # Measurement
            measurements = []
            for i in range(self.measurement_iterations):
                start_time = time.perf_counter()
                try:
                    result = await self._run_test_function(test_func, scenario)
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    measurements.append(latency_ms)
                except Exception as e:
                    measurements.append(float('inf'))
                    print(f"    Iteration {i} failed: {e}")
            
            # Analyze measurements
            valid_measurements = [m for m in measurements if m != float('inf')]
            if valid_measurements:
                latency_stats = self._calculate_latency_statistics(valid_measurements)
                
                # Check against thresholds
                target = self.thresholds["latency"].get(f"{test_name}_ms", 
                                                      self.thresholds["latency"]["inference_ms"])
                
                # Create benchmark result
                benchmark_result = BenchmarkResult(
                    name=f"{test_name}_{scenario}",
                    benchmark_type=BenchmarkType.LATENCY,
                    value=latency_stats["p95"],
                    unit="ms",
                    target=target,
                    passed=latency_stats["p95"] <= target,
                    timestamp=time.time(),
                    duration_ms=sum(valid_measurements),
                    metadata={
                        "scenario": scenario,
                        "statistics": latency_stats,
                        "success_rate": len(valid_measurements) / len(measurements)
                    }
                )
                
                self.results.append(benchmark_result)
                latency_results[test_name] = asdict(benchmark_result)
            else:
                latency_results[test_name] = {"error": "All measurements failed"}
        
        return latency_results
    
    async def _benchmark_throughput(self,
                                  test_functions: Dict[str, Callable],
                                  scenario: str) -> Dict[str, Any]:
        """Benchmark throughput performance"""
        throughput_results = {}
        
        for test_name, test_func in test_functions.items():
            print(f"  Throughput benchmark: {test_name}")
            
            # Run for fixed duration and count completions
            duration_seconds = 10.0
            start_time = time.perf_counter()
            completions = 0
            errors = 0
            
            while (time.perf_counter() - start_time) < duration_seconds:
                try:
                    await self._run_test_function(test_func, scenario)
                    completions += 1
                except:
                    errors += 1
            
            actual_duration = time.perf_counter() - start_time
            throughput_per_second = completions / actual_duration
            
            # Check against thresholds
            target = self.thresholds["throughput"].get(f"{test_name}_per_second", 
                                                     self.thresholds["throughput"]["frames_per_second"])
            
            benchmark_result = BenchmarkResult(
                name=f"{test_name}_{scenario}_throughput",
                benchmark_type=BenchmarkType.THROUGHPUT,
                value=throughput_per_second,
                unit="ops/sec",
                target=target,
                passed=throughput_per_second >= target,
                timestamp=time.time(),
                duration_ms=actual_duration * 1000,
                metadata={
                    "scenario": scenario,
                    "completions": completions,
                    "errors": errors,
                    "success_rate": completions / (completions + errors) if (completions + errors) > 0 else 0
                }
            )
            
            self.results.append(benchmark_result)
            throughput_results[test_name] = asdict(benchmark_result)
        
        return throughput_results
    
    async def _benchmark_accuracy(self,
                                test_functions: Dict[str, Callable],
                                scenario: str) -> Dict[str, Any]:
        """Benchmark accuracy performance"""
        accuracy_results = {}
        
        # This would need actual test data and ground truth
        # For demo purposes, simulate accuracy measurements
        
        for test_name, test_func in test_functions.items():
            print(f"  Accuracy benchmark: {test_name}")
            
            # Simulate accuracy measurement
            try:
                # Run test with known inputs and compare outputs
                accuracy_percent = np.random.normal(95.0, 2.0)  # Simulated
                accuracy_percent = max(0, min(100, accuracy_percent))
                
                target = self.thresholds["accuracy"].get(f"{test_name}_percent",
                                                       self.thresholds["accuracy"]["object_detection_percent"])
                
                benchmark_result = BenchmarkResult(
                    name=f"{test_name}_{scenario}_accuracy",
                    benchmark_type=BenchmarkType.ACCURACY,
                    value=accuracy_percent,
                    unit="percent",
                    target=target,
                    passed=accuracy_percent >= target,
                    timestamp=time.time(),
                    duration_ms=0,
                    metadata={
                        "scenario": scenario,
                        "test_samples": 100,  # Simulated
                        "method": "simulated"
                    }
                )
                
                self.results.append(benchmark_result)
                accuracy_results[test_name] = asdict(benchmark_result)
                
            except Exception as e:
                accuracy_results[test_name] = {"error": str(e)}
        
        return accuracy_results
    
    async def _benchmark_memory(self,
                              test_functions: Dict[str, Callable],
                              scenario: str) -> Dict[str, Any]:
        """Benchmark memory usage"""
        memory_results = {}
        
        for test_name, test_func in test_functions.items():
            print(f"  Memory benchmark: {test_name}")
            
            # Monitor memory during test execution
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            peak_memory = initial_memory
            memory_samples = []
            
            try:
                # Run test while monitoring memory
                for _ in range(5):  # Multiple iterations
                    await self._run_test_function(test_func, scenario)
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(current_memory)
                    peak_memory = max(peak_memory, current_memory)
                
                avg_memory = statistics.mean(memory_samples)
                memory_growth = peak_memory - initial_memory
                
                target = self.thresholds["memory"]["peak_usage_mb"]
                
                benchmark_result = BenchmarkResult(
                    name=f"{test_name}_{scenario}_memory",
                    benchmark_type=BenchmarkType.MEMORY,
                    value=peak_memory,
                    unit="MB",
                    target=target,
                    passed=peak_memory <= target,
                    timestamp=time.time(),
                    duration_ms=0,
                    metadata={
                        "scenario": scenario,
                        "initial_memory_mb": initial_memory,
                        "peak_memory_mb": peak_memory,
                        "average_memory_mb": avg_memory,
                        "memory_growth_mb": memory_growth
                    }
                )
                
                self.results.append(benchmark_result)
                memory_results[test_name] = asdict(benchmark_result)
                
            except Exception as e:
                memory_results[test_name] = {"error": str(e)}
        
        return memory_results
    
    async def _benchmark_power(self,
                             test_functions: Dict[str, Callable],
                             scenario: str) -> Dict[str, Any]:
        """Benchmark power consumption"""
        power_results = {}
        
        for test_name, test_func in test_functions.items():
            print(f"  Power benchmark: {test_name}")
            
            # Simulate power measurement (would use actual power APIs)
            try:
                cpu_usage_before = psutil.cpu_percent(interval=1.0)
                
                start_time = time.perf_counter()
                await self._run_test_function(test_func, scenario)
                duration = time.perf_counter() - start_time
                
                cpu_usage_after = psutil.cpu_percent(interval=1.0)
                avg_cpu_usage = (cpu_usage_before + cpu_usage_after) / 2
                
                # Estimate power consumption based on CPU usage
                base_power = 15.0  # Watts
                dynamic_power = (avg_cpu_usage / 100.0) * 35.0  # Additional power
                estimated_power = base_power + dynamic_power
                
                target = self.thresholds["power"]["average_watts"]
                
                benchmark_result = BenchmarkResult(
                    name=f"{test_name}_{scenario}_power",
                    benchmark_type=BenchmarkType.POWER,
                    value=estimated_power,
                    unit="watts",
                    target=target,
                    passed=estimated_power <= target,
                    timestamp=time.time(),
                    duration_ms=duration * 1000,
                    metadata={
                        "scenario": scenario,
                        "cpu_usage_percent": avg_cpu_usage,
                        "estimation_method": "cpu_based",
                        "duration_seconds": duration
                    }
                )
                
                self.results.append(benchmark_result)
                power_results[test_name] = asdict(benchmark_result)
                
            except Exception as e:
                power_results[test_name] = {"error": str(e)}
        
        return power_results
    
    async def _benchmark_regression(self,
                                  test_functions: Dict[str, Callable],
                                  scenario: str) -> Dict[str, Any]:
        """Benchmark for performance regression detection"""
        regression_results = {}
        
        # Compare current results with historical baselines
        for test_name, test_func in test_functions.items():
            print(f"  Regression benchmark: {test_name}")
            
            baseline_key = f"{test_name}_{scenario}"
            
            if baseline_key in self.baselines:
                baseline = self.baselines[baseline_key]
                
                # Run current test
                try:
                    start_time = time.perf_counter()
                    await self._run_test_function(test_func, scenario)
                    current_latency = (time.perf_counter() - start_time) * 1000
                    
                    # Calculate regression
                    regression_percent = ((current_latency - baseline.baseline_latency_ms) / 
                                        baseline.baseline_latency_ms) * 100
                    
                    is_regression = regression_percent > self.regression_threshold_percent
                    
                    benchmark_result = BenchmarkResult(
                        name=f"{test_name}_{scenario}_regression",
                        benchmark_type=BenchmarkType.REGRESSION,
                        value=regression_percent,
                        unit="percent",
                        target=self.regression_threshold_percent,
                        passed=not is_regression,
                        timestamp=time.time(),
                        duration_ms=current_latency,
                        metadata={
                            "scenario": scenario,
                            "baseline_latency_ms": baseline.baseline_latency_ms,
                            "current_latency_ms": current_latency,
                            "regression_threshold_percent": self.regression_threshold_percent
                        }
                    )
                    
                    self.results.append(benchmark_result)
                    regression_results[test_name] = asdict(benchmark_result)
                    
                except Exception as e:
                    regression_results[test_name] = {"error": str(e)}
            else:
                regression_results[test_name] = {"error": "No baseline available"}
        
        return regression_results
    
    async def _run_test_function(self, test_func: Callable, scenario: str) -> Any:
        """Run test function with timeout and error handling"""
        try:
            # Create test context with scenario
            test_context = {"scenario": scenario, "platform": self.platform}
            
            # Run with timeout
            return await asyncio.wait_for(
                asyncio.coroutine(test_func)(test_context) if asyncio.iscoroutinefunction(test_func) 
                else asyncio.get_event_loop().run_in_executor(None, test_func, test_context),
                timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            raise Exception(f"Test function timed out after {self.timeout_seconds} seconds")
        except Exception as e:
            raise Exception(f"Test function failed: {e}")
    
    def _calculate_latency_statistics(self, measurements: List[float]) -> Dict[str, float]:
        """Calculate comprehensive latency statistics"""
        return {
            "mean": statistics.mean(measurements),
            "median": statistics.median(measurements),
            "min": min(measurements),
            "max": max(measurements),
            "std_dev": statistics.stdev(measurements) if len(measurements) > 1 else 0,
            "p95": np.percentile(measurements, 95),
            "p99": np.percentile(measurements, 99),
            "p999": np.percentile(measurements, 99.9) if len(measurements) >= 10 else max(measurements)
        }
    
    def _analyze_comprehensive_results(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results across all scenarios"""
        analysis = {
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "scenarios_tested": len(all_results)
            },
            "performance_trends": {},
            "bottlenecks": [],
            "regressions": []
        }
        
        # Aggregate results
        all_benchmark_results = []
        for scenario, scenario_results in all_results.items():
            for benchmark_type, benchmarks in scenario_results.items():
                if isinstance(benchmarks, dict) and "error" not in benchmarks:
                    for test_name, result in benchmarks.items():
                        if isinstance(result, dict) and "passed" in result:
                            all_benchmark_results.append(result)
                            analysis["summary"]["total_tests"] += 1
                            if result["passed"]:
                                analysis["summary"]["passed_tests"] += 1
                            else:
                                analysis["summary"]["failed_tests"] += 1
        
        # Calculate success rate
        total_tests = analysis["summary"]["total_tests"]
        if total_tests > 0:
            analysis["summary"]["success_rate"] = analysis["summary"]["passed_tests"] / total_tests
        else:
            analysis["summary"]["success_rate"] = 0
        
        # Identify performance bottlenecks
        self._identify_bottlenecks(all_benchmark_results)
        analysis["bottlenecks"] = [asdict(b) for b in self.bottlenecks]
        
        return analysis
    
    def _identify_bottlenecks(self, benchmark_results: List[Dict[str, Any]]):
        """Identify performance bottlenecks"""
        
        # Group results by test type
        latency_results = [r for r in benchmark_results if r.get("benchmark_type") == "latency"]
        memory_results = [r for r in benchmark_results if r.get("benchmark_type") == "memory"]
        
        # Check for latency bottlenecks
        for result in latency_results:
            if not result["passed"] and result["value"] > result["target"]:
                impact = ((result["value"] - result["target"]) / result["target"]) * 100
                severity = Severity.CRITICAL if impact > 50 else Severity.HIGH if impact > 20 else Severity.MEDIUM
                
                bottleneck = Bottleneck(
                    component=result["name"],
                    severity=severity,
                    impact_percent=impact,
                    description=f"Latency {result['value']:.2f}{result['unit']} exceeds target {result['target']:.2f}{result['unit']}",
                    recommendations=[
                        "Optimize algorithm complexity",
                        "Enable hardware acceleration",
                        "Improve memory access patterns",
                        "Consider parallel processing"
                    ],
                    metrics={"latency_ms": result["value"], "target_ms": result["target"]}
                )
                self.bottlenecks.append(bottleneck)
        
        # Check for memory bottlenecks
        for result in memory_results:
            if not result["passed"]:
                impact = ((result["value"] - result["target"]) / result["target"]) * 100
                severity = Severity.HIGH if impact > 30 else Severity.MEDIUM
                
                bottleneck = Bottleneck(
                    component=result["name"],
                    severity=severity,
                    impact_percent=impact,
                    description=f"Memory usage {result['value']:.2f}{result['unit']} exceeds target {result['target']:.2f}{result['unit']}",
                    recommendations=[
                        "Implement memory pooling",
                        "Optimize data structures",
                        "Enable garbage collection tuning",
                        "Consider streaming processing"
                    ],
                    metrics={"memory_mb": result["value"], "target_mb": result["target"]}
                )
                self.bottlenecks.append(bottleneck)
    
    def _generate_optimization_recommendations(self, all_results: Dict[str, Any]):
        """Generate optimization recommendations based on results"""
        
        # Analyze patterns across results
        failed_latency_tests = []
        failed_memory_tests = []
        
        for scenario, scenario_results in all_results.items():
            latency_results = scenario_results.get("latency", {})
            memory_results = scenario_results.get("memory", {})
            
            for test_name, result in latency_results.items():
                if isinstance(result, dict) and not result.get("passed", True):
                    failed_latency_tests.append(result)
            
            for test_name, result in memory_results.items():
                if isinstance(result, dict) and not result.get("passed", True):
                    failed_memory_tests.append(result)
        
        # Generate latency optimization recommendations
        if failed_latency_tests:
            self.recommendations.append(OptimizationRecommendation(
                category="latency",
                priority=1,
                description="Implement aggressive latency optimizations",
                expected_improvement="20-40% latency reduction",
                implementation_effort="medium",
                risk_level="low",
                code_changes=[
                    "Enable compiler optimizations (-O3, -march=native)",
                    "Implement SIMD vectorization",
                    "Add memory prefetching hints",
                    "Optimize hot paths with profiling"
                ]
            ))
        
        # Generate memory optimization recommendations
        if failed_memory_tests:
            self.recommendations.append(OptimizationRecommendation(
                category="memory",
                priority=2,
                description="Implement memory usage optimizations",
                expected_improvement="30-50% memory reduction",
                implementation_effort="high",
                risk_level="medium",
                code_changes=[
                    "Implement object pooling",
                    "Use memory-mapped files for large data",
                    "Optimize data structure layout",
                    "Enable memory compression"
                ]
            ))
        
        # Platform-specific recommendations
        if self.platform == "automotive_ecu":
            self.recommendations.append(OptimizationRecommendation(
                category="ecu_specific",
                priority=3,
                description="Automotive ECU optimizations",
                expected_improvement="15-25% overall performance",
                implementation_effort="low",
                risk_level="low",
                code_changes=[
                    "Use fixed-point arithmetic instead of float",
                    "Implement deterministic execution paths",
                    "Minimize dynamic memory allocation",
                    "Use real-time scheduling policies"
                ]
            ))
    
    def generate_benchmark_report(self) -> str:
        """Generate comprehensive benchmark report"""
        
        if not self.results:
            return "No benchmark results available"
        
        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Group by benchmark type
        by_type = {}
        for result in self.results:
            type_name = result.benchmark_type.value
            if type_name not in by_type:
                by_type[type_name] = []
            by_type[type_name].append(result)
        
        report = f"""
ADAS Comprehensive Benchmark Report
==================================

Platform: {self.platform}
Test Summary: {passed_tests}/{total_tests} passed ({success_rate:.1f}%)
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}

Benchmark Results by Type:
"""
        
        for benchmark_type, results in by_type.items():
            type_passed = sum(1 for r in results if r.passed)
            type_total = len(results)
            type_success = (type_passed / type_total) * 100 if type_total > 0 else 0
            
            report += f"""
{benchmark_type.upper()}:
- Tests: {type_passed}/{type_total} passed ({type_success:.1f}%)
"""
            
            for result in results:
                status = "PASS" if result.passed else "FAIL"
                report += f"  - {result.name}: {result.value:.2f}{result.unit} (target: {result.target:.2f}{result.unit}) [{status}]\n"
        
        # Add bottlenecks
        if self.bottlenecks:
            report += f"\nPerformance Bottlenecks ({len(self.bottlenecks)}):\n"
            for bottleneck in self.bottlenecks:
                report += f"- {bottleneck.component} ({bottleneck.severity.value}): {bottleneck.description}\n"
        
        # Add recommendations
        if self.recommendations:
            report += f"\nOptimization Recommendations ({len(self.recommendations)}):\n"
            for rec in sorted(self.recommendations, key=lambda x: x.priority):
                report += f"{rec.priority}. {rec.description} (Expected: {rec.expected_improvement})\n"
        
        return report
    
    def export_results(self, filename: str = "benchmark_results.json"):
        """Export benchmark results to file"""
        export_data = {
            "platform": self.platform,
            "timestamp": time.time(),
            "results": [asdict(r) for r in self.results],
            "bottlenecks": [asdict(b) for b in self.bottlenecks],
            "recommendations": [asdict(r) for r in self.recommendations],
            "thresholds": self.thresholds
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Benchmark results exported to {filename}")


# Example test functions for automotive scenarios
async def example_object_detection(context: Dict[str, Any]) -> str:
    """Example object detection test"""
    scenario = context.get("scenario", "urban")
    
    # Simulate processing time based on scenario complexity
    processing_times = {
        "urban": 0.008,      # 8ms
        "highway": 0.005,    # 5ms
        "parking": 0.012,    # 12ms
        "night": 0.015,      # 15ms
        "rain": 0.010        # 10ms
    }
    
    await asyncio.sleep(processing_times.get(scenario, 0.008))
    return f"detected_objects_{scenario}"


async def example_lane_detection(context: Dict[str, Any]) -> str:
    """Example lane detection test"""
    scenario = context.get("scenario", "urban")
    
    processing_times = {
        "urban": 0.003,      # 3ms
        "highway": 0.002,    # 2ms
        "parking": 0.001,    # 1ms
        "night": 0.006,      # 6ms
        "rain": 0.005        # 5ms
    }
    
    await asyncio.sleep(processing_times.get(scenario, 0.003))
    return f"detected_lanes_{scenario}"


async def example_path_planning(context: Dict[str, Any]) -> str:
    """Example path planning test"""
    scenario = context.get("scenario", "urban")
    
    processing_times = {
        "urban": 0.020,      # 20ms
        "highway": 0.010,    # 10ms
        "parking": 0.050,    # 50ms
        "night": 0.025,      # 25ms
        "rain": 0.030        # 30ms
    }
    
    await asyncio.sleep(processing_times.get(scenario, 0.020))
    return f"planned_path_{scenario}"


async def demo_comprehensive_benchmark():
    """Demonstrate comprehensive ADAS benchmarking"""
    
    print("=== ADAS Comprehensive Benchmark Demo ===")
    
    # Initialize benchmark suite
    suite = BenchmarkSuite("automotive_ecu")
    
    # Define test functions
    test_functions = {
        "object_detection": example_object_detection,
        "lane_detection": example_lane_detection,
        "path_planning": example_path_planning
    }
    
    # Run comprehensive benchmark
    results = await suite.run_comprehensive_benchmark(
        test_functions, 
        scenarios=["urban", "highway", "night"]
    )
    
    # Generate and display report
    print("\n" + suite.generate_benchmark_report())
    
    # Export results
    suite.export_results("adas_benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(demo_comprehensive_benchmark())