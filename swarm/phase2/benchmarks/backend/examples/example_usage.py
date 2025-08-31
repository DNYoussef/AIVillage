#!/usr/bin/env python3
"""
Example Usage of Backend Performance Benchmark Suite

Demonstrates various ways to use the benchmark suite for performance validation
and regression testing between monolithic and microservices architectures.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from suite.benchmark_suite import BackendBenchmarkSuite
from tools.config_manager import ConfigManager
from tools.memory_profiler import MemoryProfiler, MemoryBenchmarkIntegration
from tools.regression_detector import RegressionDetector
from tools.visualization import PerformanceVisualizer

async def example_basic_comparison():
    """
    Example 1: Basic performance comparison between architectures
    """
    print("=" * 60)
    print("Example 1: Basic Performance Comparison")
    print("=" * 60)
    
    # Initialize benchmark suite
    suite = BackendBenchmarkSuite("examples/results")
    
    # Create simple configurations
    monolithic_config = {
        'training': {'duration': 30, 'concurrent_models': 5},
        'websocket': {'websocket_url': 'ws://localhost:8080/ws', 'message_count': 100},
        'api': {'api_url': 'http://localhost:8080', 'endpoints': ['/health'], 'requests_per_endpoint': 50},
        'concurrent': {'api_url': 'http://localhost:8080', 'max_concurrent': 20}
    }
    
    microservices_config = {
        'training': {'duration': 30, 'concurrent_models': 5},
        'websocket': {'websocket_url': 'ws://localhost:8081/ws', 'message_count': 100},
        'api': {'api_url': 'http://localhost:8081', 'endpoints': ['/health'], 'requests_per_endpoint': 50},
        'concurrent': {'api_url': 'http://localhost:8081', 'max_concurrent': 20}
    }
    
    # Run comparison (in real usage, this would connect to actual services)
    try:
        print("🚀 Running basic performance comparison...")
        # results = await suite.run_full_comparison(monolithic_config, microservices_config)
        
        # For demonstration, we'll create mock results
        results = create_mock_results()
        
        # Validate performance
        validation = suite.validate_performance_requirements(results)
        
        print(f"✅ Validation Results: {sum(validation.values())}/{len(validation)} checks passed")
        
        # Generate report
        report = suite.generate_validation_report(results, validation)
        print("\n" + report[:500] + "..." if len(report) > 500 else report)
        
    except Exception as e:
        print(f"❌ Example failed (expected without running services): {e}")

async def example_memory_profiling():
    """
    Example 2: Memory profiling integration
    """
    print("\n" + "=" * 60)
    print("Example 2: Memory Profiling Integration") 
    print("=" * 60)
    
    # Initialize memory profiler
    profiler = MemoryProfiler()
    integration = MemoryBenchmarkIntegration(profiler)
    
    # Mock benchmark function
    async def mock_benchmark(config):
        await asyncio.sleep(0.5)  # Simulate work
        
        # Simulate some memory allocation
        data = [i * i for i in range(10000)]
        
        return {
            'throughput': 15.2,
            'latency_stats': {'avg': 125, 'p95': 200, 'p99': 300},
            'success_rate': 0.95
        }
    
    # Profile the benchmark
    print("🧠 Running benchmark with memory profiling...")
    
    try:
        result, memory_profile = await profiler.profile_function(
            mock_benchmark, {'duration': 30}
        )
        
        print(f"✅ Benchmark completed with memory profiling")
        print(f"   Memory Stats: {memory_profile.get('memory_stats', {}).get('rss_memory', {})}")
        print(f"   Recommendations: {len(memory_profile.get('recommendations', []))} generated")
        
        for rec in memory_profile.get('recommendations', [])[:3]:
            print(f"   💡 {rec}")
            
    except Exception as e:
        print(f"✅ Memory profiling example completed (simulated): {e}")

async def example_regression_detection():
    """
    Example 3: Performance regression detection
    """
    print("\n" + "=" * 60)
    print("Example 3: Performance Regression Detection")
    print("=" * 60)
    
    # Initialize regression detector
    detector = RegressionDetector()
    
    # Create mock baseline and current results
    baseline_results = {
        'training_throughput': create_mock_benchmark_result(throughput=20.0, avg_latency=100),
        'api_response_time': create_mock_benchmark_result(throughput=150.0, avg_latency=50)
    }
    
    current_results = {
        'training_throughput': create_mock_benchmark_result(throughput=18.0, avg_latency=120),  # Regression
        'api_response_time': create_mock_benchmark_result(throughput=155.0, avg_latency=45)    # Improvement
    }
    
    print("🔍 Analyzing performance regression...")
    
    # Detect regressions
    alerts = detector.analyze_regression(current_results, baseline_results)
    
    print(f"📊 Found {len(alerts)} performance alerts")
    
    for alert in alerts:
        severity_icon = "🔴" if alert.severity.value == "critical" else "🟡"
        print(f"   {severity_icon} {alert.benchmark_name}: {alert.description}")
        print(f"      Action: {alert.recommendation}")
    
    # Generate regression report
    if alerts:
        report = detector.generate_regression_report(alerts)
        print(f"\n📋 Regression report generated ({len(report.split(chr(10)))} lines)")

async def example_custom_configuration():
    """
    Example 4: Custom configuration management
    """
    print("\n" + "=" * 60)
    print("Example 4: Custom Configuration Management")
    print("=" * 60)
    
    # Initialize config manager
    config_manager = ConfigManager("examples/config")
    
    print("🔧 Creating custom benchmark configurations...")
    
    # Get optimized configs for different workloads
    workloads = ['high_throughput', 'low_latency', 'memory_constrained']
    
    for workload in workloads:
        mono_config = config_manager.get_optimized_config('monolithic', workload)
        micro_config = config_manager.get_optimized_config('microservices', workload)
        
        print(f"   📊 {workload.title().replace('_', ' ')} Profile:")
        print(f"      Monolithic - Training Models: {mono_config['training']['concurrent_models']}")
        print(f"      Microservices - Training Models: {micro_config['training']['concurrent_models']}")
        
        # Validate configurations
        mono_issues = config_manager.validate_config(mono_config)
        micro_issues = config_manager.validate_config(micro_config)
        
        if not mono_issues and not micro_issues:
            print(f"      ✅ Configurations validated")
        else:
            print(f"      ⚠️ {len(mono_issues + micro_issues)} validation issues")
    
    # Save custom configuration
    custom_scenario = config_manager.create_benchmark_scenario(
        'custom_test',
        ['training_throughput', 'api_response_time'],
        'microservices',
        'high_performance'
    )
    
    config_manager.save_config('custom_scenario', custom_scenario)
    print("💾 Custom scenario configuration saved")

async def example_visualization():
    """
    Example 5: Performance visualization
    """
    print("\n" + "=" * 60)
    print("Example 5: Performance Visualization")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = PerformanceVisualizer("examples/reports")
    
    # Create mock comparison results
    results = create_mock_results()
    validation = {
        'no_performance_regression': True,
        'memory_efficiency': True,
        'latency_acceptable': True,
        'overall_quality': True,
        'scalability_improvement': False
    }
    
    print("📈 Generating performance visualizations...")
    
    try:
        # Create all visualizations
        files = visualizer.create_all_visualizations(results, validation)
        
        print(f"✅ Generated {len(files)} visualization files:")
        for file_type, file_path in files.items():
            print(f"   📊 {file_type.title()}: {Path(file_path).name}")
    
    except Exception as e:
        print(f"📊 Visualization example completed (dependencies may be missing): {e}")

def create_mock_benchmark_result(throughput=10.0, avg_latency=100, success_rate=0.95):
    """Create a mock benchmark result for examples"""
    
    class MockResult:
        def __init__(self, throughput, avg_latency, success_rate):
            self.throughput = throughput
            self.latency_stats = {
                'avg': avg_latency,
                'p50': avg_latency * 0.8,
                'p95': avg_latency * 1.5,
                'p99': avg_latency * 2.0,
                'max': avg_latency * 3.0,
                'min': avg_latency * 0.5
            }
            self.success_rate = success_rate
            self.resource_usage = {
                'memory': {'peak_mb': 512, 'avg': 256},
                'cpu': {'avg': 45, 'max': 80}
            }
            self.duration = 60
            self.timestamp = "2024-01-01T10:00:00"
            self.metadata = {'example': True}
    
    return MockResult(throughput, avg_latency, success_rate)

def create_mock_results():
    """Create mock benchmark comparison results"""
    
    monolithic_results = {
        'training_throughput': create_mock_benchmark_result(20.0, 100, 0.98),
        'websocket_latency': create_mock_benchmark_result(100.0, 25, 0.99),
        'api_response_time': create_mock_benchmark_result(150.0, 50, 0.97),
        'concurrent_load': create_mock_benchmark_result(200.0, 75, 0.95)
    }
    
    microservices_results = {
        'training_throughput': create_mock_benchmark_result(22.0, 95, 0.98),    # Better throughput, better latency
        'websocket_latency': create_mock_benchmark_result(95.0, 28, 0.98),     # Slightly lower throughput, higher latency
        'api_response_time': create_mock_benchmark_result(160.0, 45, 0.98),    # Better throughput and latency
        'concurrent_load': create_mock_benchmark_result(250.0, 70, 0.96)       # Much better throughput, better latency
    }
    
    # Mock comparison analysis
    comparison = {}
    for benchmark in monolithic_results.keys():
        mono = monolithic_results[benchmark]
        micro = microservices_results[benchmark]
        
        throughput_change = ((micro.throughput - mono.throughput) / mono.throughput) * 100
        latency_change = ((micro.latency_stats['avg'] - mono.latency_stats['avg']) / mono.latency_stats['avg']) * 100
        memory_change = -10.0  # Assume 10% memory reduction
        
        # Determine overall score
        if throughput_change > 5 and latency_change < 10 and memory_change < 0:
            score = "EXCELLENT"
        elif throughput_change > 0 or latency_change < 5:
            score = "GOOD"
        elif throughput_change > -5:
            score = "FAIR"
        else:
            score = "POOR"
        
        comparison[benchmark] = {
            'throughput_change_percent': throughput_change,
            'latency_change_percent': latency_change,
            'memory_change_percent': memory_change,
            'performance_regression': throughput_change < -5.0,
            'memory_improvement': memory_change < -10.0,
            'latency_improvement': latency_change < -5.0,
            'overall_score': score
        }
    
    return {
        'monolithic': monolithic_results,
        'microservices': microservices_results,
        'comparison': comparison
    }

async def run_all_examples():
    """Run all example demonstrations"""
    
    print("🚀 Backend Performance Benchmark Suite - Example Usage")
    print("=" * 80)
    print("This demonstrates the benchmark suite capabilities without requiring")
    print("actual running services. In production, connect to real endpoints.")
    print("=" * 80)
    
    # Run examples
    await example_basic_comparison()
    await example_memory_profiling()
    await example_regression_detection()
    await example_custom_configuration()
    await example_visualization()
    
    print("\n" + "=" * 80)
    print("🎉 All examples completed successfully!")
    print("=" * 80)
    print("Next steps:")
    print("1. Set up your monolithic and microservices backends")
    print("2. Update configurations with actual URLs")
    print("3. Run: python run_benchmarks.py")
    print("4. Review generated reports and visualizations")
    print("=" * 80)

if __name__ == "__main__":
    # Set up proper event loop for Windows
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(run_all_examples())