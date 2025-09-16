"""
ADAS Performance Integration Module
==================================

Unified interface for comprehensive ADAS performance optimization.
Integrates latency optimization, resource management, benchmarking, and profiling.
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum

from .latency_optimizer import LatencyOptimizer, OptimizationLevel, PipelineStage
from .resource_manager import ResourceManager, PowerMode, ResourceLimits
from .benchmark_suite import BenchmarkSuite, BenchmarkType
from .edge_profiler import EdgeProfiler, HardwarePlatform, OptimizationTarget


class PerformanceMode(Enum):
    """Integrated performance modes"""
    AUTOMOTIVE_CRITICAL = "automotive_critical"    # <5ms, safety-critical
    AUTOMOTIVE_STANDARD = "automotive_standard"    # <10ms, standard ADAS
    DEVELOPMENT = "development"                     # <20ms, development/testing
    VALIDATION = "validation"                       # Comprehensive validation


@dataclass
class IntegratedPerformanceConfig:
    """Configuration for integrated performance optimization"""
    mode: PerformanceMode
    platform: HardwarePlatform
    optimization_targets: List[OptimizationTarget]
    resource_limits: ResourceLimits
    benchmark_scenarios: List[str]
    enable_real_time_monitoring: bool = True
    enable_adaptive_optimization: bool = True
    enable_thermal_management: bool = True
    export_results: bool = True


@dataclass
class IntegratedPerformanceResult:
    """Comprehensive performance optimization result"""
    config: IntegratedPerformanceConfig
    latency_results: Dict[str, Any]
    resource_results: Dict[str, Any]
    benchmark_results: Dict[str, Any]
    profiling_results: Dict[str, Any]
    overall_success: bool
    performance_score: float
    recommendations: List[str]
    issues: List[str]
    metadata: Dict[str, Any]


class ADASPerformanceIntegrator:
    """
    Unified ADAS performance optimization integrator
    
    Combines all performance optimization components into a single,
    easy-to-use interface for automotive applications.
    """
    
    def __init__(self, config: IntegratedPerformanceConfig):
        self.config = config
        
        # Initialize components
        self.latency_optimizer = self._initialize_latency_optimizer()
        self.resource_manager = self._initialize_resource_manager()
        self.benchmark_suite = self._initialize_benchmark_suite()
        self.edge_profiler = self._initialize_edge_profiler()
        
        # Integration state
        self.optimization_active = False
        self.monitoring_active = False
        self.results_history: List[IntegratedPerformanceResult] = []
        
        print(f"ADASPerformanceIntegrator initialized for {config.mode.value}")
        print(f"Platform: {config.platform.value}")
        print(f"Optimization targets: {[t.value for t in config.optimization_targets]}")
    
    def _initialize_latency_optimizer(self) -> LatencyOptimizer:
        """Initialize latency optimizer based on config"""
        
        # Map performance mode to optimization level
        mode_mapping = {
            PerformanceMode.AUTOMOTIVE_CRITICAL: OptimizationLevel.CRITICAL,
            PerformanceMode.AUTOMOTIVE_STANDARD: OptimizationLevel.AUTOMOTIVE,
            PerformanceMode.DEVELOPMENT: OptimizationLevel.STANDARD,
            PerformanceMode.VALIDATION: OptimizationLevel.BASIC
        }
        
        optimization_level = mode_mapping[self.config.mode]
        optimizer = LatencyOptimizer(optimization_level)
        
        if self.config.enable_real_time_monitoring:
            optimizer.start_real_time_monitoring()
        
        return optimizer
    
    def _initialize_resource_manager(self) -> ResourceManager:
        """Initialize resource manager based on config"""
        
        manager = ResourceManager(
            limits=self.config.resource_limits,
            monitoring_interval=0.1
        )
        
        if self.config.enable_real_time_monitoring:
            manager.start_monitoring()
        
        return manager
    
    def _initialize_benchmark_suite(self) -> BenchmarkSuite:
        """Initialize benchmark suite based on config"""
        
        return BenchmarkSuite(platform=self.config.platform.value)
    
    def _initialize_edge_profiler(self) -> EdgeProfiler:
        """Initialize edge profiler based on config"""
        
        return EdgeProfiler(target_platform=self.config.platform)
    
    async def run_comprehensive_optimization(self,
                                           test_functions: Dict[str, Callable],
                                           pipeline_stages: Optional[List[PipelineStage]] = None
                                           ) -> IntegratedPerformanceResult:
        """
        Run comprehensive performance optimization across all components
        
        Args:
            test_functions: Dictionary of test functions to optimize
            pipeline_stages: Optional pipeline stages for latency optimization
            
        Returns:
            Comprehensive performance results
        """
        print("Starting comprehensive ADAS performance optimization...")
        
        start_time = time.time()
        self.optimization_active = True
        
        try:
            # Phase 1: Resource allocation and setup
            print("\n=== Phase 1: Resource Allocation ===")
            resource_allocations = await self._allocate_resources_for_tests(test_functions)
            
            # Phase 2: Latency optimization
            print("\n=== Phase 2: Latency Optimization ===")
            latency_results = await self._run_latency_optimization(
                test_functions, pipeline_stages
            )
            
            # Phase 3: Comprehensive benchmarking
            print("\n=== Phase 3: Comprehensive Benchmarking ===")
            benchmark_results = await self._run_comprehensive_benchmarking(test_functions)
            
            # Phase 4: Hardware profiling
            print("\n=== Phase 4: Hardware Profiling ===")
            profiling_results = await self._run_hardware_profiling(test_functions)
            
            # Phase 5: Integration analysis
            print("\n=== Phase 5: Integration Analysis ===")
            integration_analysis = self._analyze_integrated_results(
                latency_results, resource_allocations, benchmark_results, profiling_results
            )
            
            # Phase 6: Generate recommendations
            print("\n=== Phase 6: Generating Recommendations ===")
            recommendations = self._generate_integrated_recommendations(integration_analysis)
            
            # Create comprehensive result
            result = IntegratedPerformanceResult(
                config=self.config,
                latency_results=latency_results,
                resource_results={
                    "allocations": resource_allocations,
                    "summary": self.resource_manager.get_resource_summary()
                },
                benchmark_results=benchmark_results,
                profiling_results=profiling_results,
                overall_success=integration_analysis["overall_success"],
                performance_score=integration_analysis["performance_score"],
                recommendations=recommendations,
                issues=integration_analysis["issues"],
                metadata={
                    "optimization_duration_seconds": time.time() - start_time,
                    "timestamp": time.time(),
                    "version": "1.0.0"
                }
            )
            
            self.results_history.append(result)
            
            # Export results if enabled
            if self.config.export_results:
                self._export_integrated_results(result)
            
            return result
            
        finally:
            self.optimization_active = False
            await self._cleanup_resources()
    
    async def _allocate_resources_for_tests(self, 
                                          test_functions: Dict[str, Callable]
                                          ) -> Dict[str, Any]:
        """Allocate resources for test functions"""
        
        allocations = {}
        
        for test_name, test_func in test_functions.items():
            # Determine resource requirements based on test type
            if "detection" in test_name.lower():
                # Object detection requires more resources
                power_mode = PowerMode.PERFORMANCE
                memory_mb = 1024
                priority = 8
            elif "tracking" in test_name.lower():
                # Tracking requires moderate resources
                power_mode = PowerMode.BALANCED
                memory_mb = 512
                priority = 6
            else:
                # Other functions use minimal resources
                power_mode = PowerMode.ECO
                memory_mb = 256
                priority = 4
            
            try:
                allocation = self.resource_manager.allocate_resources(
                    task_id=test_name,
                    memory_mb=memory_mb,
                    power_mode=power_mode,
                    priority=priority
                )
                allocations[test_name] = asdict(allocation)
                
            except Exception as e:
                print(f"Failed to allocate resources for {test_name}: {e}")
                allocations[test_name] = {"error": str(e)}
        
        return allocations
    
    async def _run_latency_optimization(self,
                                      test_functions: Dict[str, Callable],
                                      pipeline_stages: Optional[List[PipelineStage]]
                                      ) -> Dict[str, Any]:
        """Run latency optimization across test functions"""
        
        latency_results = {}
        
        if pipeline_stages:
            # Optimize complete pipeline
            for i in range(5):  # Multiple optimization runs
                dummy_input = f"test_input_{i}"
                result, benchmark = await self.latency_optimizer.optimize_inference_pipeline(
                    pipeline_stages, dummy_input
                )
                
                latency_results[f"pipeline_run_{i}"] = asdict(benchmark)
        
        # Individual function optimization
        for test_name, test_func in test_functions.items():
            function_results = []
            
            for i in range(3):  # Multiple runs per function
                start_time = time.perf_counter()
                try:
                    await test_func({"scenario": "optimization", "run": i})
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    
                    function_results.append({
                        "run": i,
                        "latency_ms": latency_ms,
                        "success": True
                    })
                    
                except Exception as e:
                    function_results.append({
                        "run": i,
                        "latency_ms": float('inf'),
                        "success": False,
                        "error": str(e)
                    })
            
            latency_results[test_name] = function_results
        
        # Add optimizer metrics
        latency_results["optimizer_metrics"] = self.latency_optimizer.get_performance_metrics()
        
        return latency_results
    
    async def _run_comprehensive_benchmarking(self,
                                            test_functions: Dict[str, Callable]
                                            ) -> Dict[str, Any]:
        """Run comprehensive benchmarking"""
        
        return await self.benchmark_suite.run_comprehensive_benchmark(
            test_functions,
            scenarios=self.config.benchmark_scenarios
        )
    
    async def _run_hardware_profiling(self,
                                     test_functions: Dict[str, Callable]
                                     ) -> Dict[str, Any]:
        """Run hardware-specific profiling"""
        
        profiling_results = {}
        
        # Profile each optimization target
        for target in self.config.optimization_targets:
            target_results = {}
            
            for test_name, test_func in test_functions.items():
                result = await self.edge_profiler.profile_hardware_configuration(
                    test_func, target, f"{test_name}_{target.value}"
                )
                target_results[test_name] = asdict(result)
            
            profiling_results[target.value] = target_results
        
        # Generate optimization recommendations
        recommendations = self.edge_profiler.generate_optimization_recommendations()
        profiling_results["recommendations"] = [asdict(r) for r in recommendations]
        
        return profiling_results
    
    def _analyze_integrated_results(self,
                                  latency_results: Dict[str, Any],
                                  resource_results: Dict[str, Any],
                                  benchmark_results: Dict[str, Any],
                                  profiling_results: Dict[str, Any]
                                  ) -> Dict[str, Any]:
        """Analyze integrated results across all components"""
        
        analysis = {
            "overall_success": True,
            "performance_score": 0.0,
            "issues": [],
            "component_scores": {}
        }
        
        # Analyze latency performance
        latency_score = self._analyze_latency_performance(latency_results)
        analysis["component_scores"]["latency"] = latency_score
        
        if latency_score < 0.7:
            analysis["issues"].append("Latency performance below acceptable threshold")
            analysis["overall_success"] = False
        
        # Analyze resource efficiency
        resource_score = self._analyze_resource_efficiency(resource_results)
        analysis["component_scores"]["resource"] = resource_score
        
        if resource_score < 0.7:
            analysis["issues"].append("Resource utilization inefficient")
        
        # Analyze benchmark results
        benchmark_score = self._analyze_benchmark_performance(benchmark_results)
        analysis["component_scores"]["benchmark"] = benchmark_score
        
        if benchmark_score < 0.7:
            analysis["issues"].append("Benchmark performance below targets")
            analysis["overall_success"] = False
        
        # Analyze profiling results
        profiling_score = self._analyze_profiling_performance(profiling_results)
        analysis["component_scores"]["profiling"] = profiling_score
        
        # Calculate overall performance score
        scores = list(analysis["component_scores"].values())
        analysis["performance_score"] = sum(scores) / len(scores) if scores else 0.0
        
        # Overall success criteria
        if analysis["performance_score"] < 0.75:
            analysis["overall_success"] = False
        
        return analysis
    
    def _analyze_latency_performance(self, latency_results: Dict[str, Any]) -> float:
        """Analyze latency performance and return score (0-1)"""
        
        target_latency_ms = {
            PerformanceMode.AUTOMOTIVE_CRITICAL: 5.0,
            PerformanceMode.AUTOMOTIVE_STANDARD: 10.0,
            PerformanceMode.DEVELOPMENT: 20.0,
            PerformanceMode.VALIDATION: 50.0
        }
        
        target = target_latency_ms[self.config.mode]
        
        # Analyze optimizer metrics
        metrics = latency_results.get("optimizer_metrics", {})
        if "avg_latency_ms" in metrics:
            avg_latency = metrics["avg_latency_ms"]
            if avg_latency <= target:
                return 1.0
            elif avg_latency <= target * 2:
                return 0.5
            else:
                return 0.0
        
        return 0.5  # Default score if no metrics available
    
    def _analyze_resource_efficiency(self, resource_results: Dict[str, Any]) -> float:
        """Analyze resource efficiency and return score (0-1)"""
        
        summary = resource_results.get("summary", {})
        
        if "current_usage" in summary:
            usage = summary["current_usage"]
            cpu_usage = usage.get("cpu_percent", 0)
            memory_usage = usage.get("memory_percent", 0)
            
            # Good efficiency: CPU < 80%, Memory < 75%
            cpu_score = max(0, (100 - cpu_usage) / 100)
            memory_score = max(0, (100 - memory_usage) / 100)
            
            return (cpu_score + memory_score) / 2
        
        return 0.5  # Default score
    
    def _analyze_benchmark_performance(self, benchmark_results: Dict[str, Any]) -> float:
        """Analyze benchmark performance and return score (0-1)"""
        
        analysis = benchmark_results.get("analysis", {})
        summary = analysis.get("summary", {})
        
        success_rate = summary.get("success_rate", 0)
        return success_rate
    
    def _analyze_profiling_performance(self, profiling_results: Dict[str, Any]) -> float:
        """Analyze profiling performance and return score (0-1)"""
        
        successful_profiles = 0
        total_profiles = 0
        
        for target_name, target_results in profiling_results.items():
            if target_name == "recommendations":
                continue
                
            for test_name, result in target_results.items():
                total_profiles += 1
                if result.get("success", False):
                    successful_profiles += 1
        
        return successful_profiles / total_profiles if total_profiles > 0 else 0.0
    
    def _generate_integrated_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate integrated optimization recommendations"""
        
        recommendations = []
        
        # Performance-specific recommendations
        if analysis["performance_score"] < 0.8:
            recommendations.append(
                "Consider upgrading to higher performance hardware platform"
            )
        
        # Component-specific recommendations
        if analysis["component_scores"].get("latency", 0) < 0.7:
            recommendations.append(
                "Implement aggressive latency optimizations: SIMD, hardware acceleration"
            )
        
        if analysis["component_scores"].get("resource", 0) < 0.7:
            recommendations.append(
                "Optimize resource allocation: reduce memory usage, improve CPU efficiency"
            )
        
        if analysis["component_scores"].get("benchmark", 0) < 0.7:
            recommendations.append(
                "Address benchmark failures: review failed tests and optimize accordingly"
            )
        
        # Mode-specific recommendations
        if self.config.mode == PerformanceMode.AUTOMOTIVE_CRITICAL:
            recommendations.append(
                "Enable all safety-critical optimizations and real-time guarantees"
            )
        
        # Platform-specific recommendations
        if self.config.platform == HardwarePlatform.NVIDIA_DRIVE_AGX:
            recommendations.append(
                "Leverage NVIDIA Drive SDK: TensorRT, CUDA optimizations, VPI acceleration"
            )
        elif self.config.platform == HardwarePlatform.QUALCOMM_SNAPDRAGON_RIDE:
            recommendations.append(
                "Utilize Snapdragon features: Hexagon DSP, Adreno GPU, power efficiency"
            )
        
        return recommendations
    
    def _export_integrated_results(self, result: IntegratedPerformanceResult):
        """Export integrated results to file"""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"adas_integrated_performance_{timestamp}.json"
        
        # Convert result to serializable format
        export_data = asdict(result)
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Integrated performance results exported to {filename}")
    
    async def _cleanup_resources(self):
        """Cleanup allocated resources"""
        
        # Deallocate all resources
        allocations = self.resource_manager.allocations
        for task_id in list(allocations.keys()):
            self.resource_manager.deallocate_resources(task_id)
        
        # Stop monitoring
        if self.config.enable_real_time_monitoring:
            self.latency_optimizer.stop_real_time_monitoring()
            self.resource_manager.stop_monitoring()
        
        # Cleanup optimizers
        self.latency_optimizer.cleanup()
        self.resource_manager.cleanup()
    
    def generate_integrated_report(self) -> str:
        """Generate comprehensive integrated performance report"""
        
        if not self.results_history:
            return "No optimization results available"
        
        latest_result = self.results_history[-1]
        
        report = f"""
ADAS Integrated Performance Optimization Report
==============================================

Configuration:
- Mode: {latest_result.config.mode.value}
- Platform: {latest_result.config.platform.value}
- Optimization Targets: {[t.value for t in latest_result.config.optimization_targets]}

Overall Performance:
- Success: {'YES' if latest_result.overall_success else 'NO'}
- Performance Score: {latest_result.performance_score:.2f}/1.00
- Duration: {latest_result.metadata['optimization_duration_seconds']:.1f}s

Component Scores:
"""
        
        # Add component scores from analysis
        for component, score in latest_result.metadata.get("component_scores", {}).items():
            report += f"- {component.title()}: {score:.2f}/1.00\n"
        
        # Add issues
        if latest_result.issues:
            report += f"\nIssues Identified ({len(latest_result.issues)}):\n"
            for issue in latest_result.issues:
                report += f"- {issue}\n"
        
        # Add recommendations
        if latest_result.recommendations:
            report += f"\nRecommendations ({len(latest_result.recommendations)}):\n"
            for i, rec in enumerate(latest_result.recommendations, 1):
                report += f"{i}. {rec}\n"
        
        return report


# Convenience functions for common automotive scenarios
def create_automotive_critical_config(platform: HardwarePlatform) -> IntegratedPerformanceConfig:
    """Create configuration for safety-critical automotive applications"""
    
    return IntegratedPerformanceConfig(
        mode=PerformanceMode.AUTOMOTIVE_CRITICAL,
        platform=platform,
        optimization_targets=[
            OptimizationTarget.LATENCY,
            OptimizationTarget.POWER_EFFICIENCY,
            OptimizationTarget.THERMAL_EFFICIENCY
        ],
        resource_limits=ResourceLimits(
            max_cpu_percent=70.0,
            max_memory_percent=60.0,
            max_temperature_c=75.0,
            max_power_watts=30.0
        ),
        benchmark_scenarios=["urban", "highway", "emergency"],
        enable_real_time_monitoring=True,
        enable_adaptive_optimization=True,
        enable_thermal_management=True
    )


def create_automotive_standard_config(platform: HardwarePlatform) -> IntegratedPerformanceConfig:
    """Create configuration for standard automotive applications"""
    
    return IntegratedPerformanceConfig(
        mode=PerformanceMode.AUTOMOTIVE_STANDARD,
        platform=platform,
        optimization_targets=[
            OptimizationTarget.BALANCED,
            OptimizationTarget.POWER_EFFICIENCY
        ],
        resource_limits=ResourceLimits(
            max_cpu_percent=80.0,
            max_memory_percent=75.0,
            max_temperature_c=85.0,
            max_power_watts=50.0
        ),
        benchmark_scenarios=["urban", "highway", "parking", "night"],
        enable_real_time_monitoring=True,
        enable_adaptive_optimization=True,
        enable_thermal_management=True
    )


async def demo_integrated_performance():
    """Demonstrate integrated ADAS performance optimization"""
    
    print("=== ADAS Integrated Performance Optimization Demo ===")
    
    # Create configuration for automotive-grade optimization
    config = create_automotive_standard_config(HardwarePlatform.AUTOMOTIVE_ECU_ARM)
    
    # Initialize integrator
    integrator = ADASPerformanceIntegrator(config)
    
    # Define test functions (simulated ADAS functions)
    async def object_detection_test(context):
        await asyncio.sleep(0.008)  # 8ms processing
        return "objects_detected"
    
    async def lane_detection_test(context):
        await asyncio.sleep(0.003)  # 3ms processing
        return "lanes_detected"
    
    async def path_planning_test(context):
        await asyncio.sleep(0.015)  # 15ms processing
        return "path_planned"
    
    test_functions = {
        "object_detection": object_detection_test,
        "lane_detection": lane_detection_test,
        "path_planning": path_planning_test
    }
    
    # Run comprehensive optimization
    result = await integrator.run_comprehensive_optimization(test_functions)
    
    # Display results
    print("\n" + integrator.generate_integrated_report())
    
    print(f"\nOptimization completed!")
    print(f"Overall success: {'YES' if result.overall_success else 'NO'}")
    print(f"Performance score: {result.performance_score:.2f}/1.00")


if __name__ == "__main__":
    asyncio.run(demo_integrated_performance())