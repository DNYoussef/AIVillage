"""
Comprehensive Benchmark Suite

Orchestrates all performance benchmarks for comparing monolithic vs microservices
architectures with focus on training throughput, WebSocket latency, API response times,
memory usage, and concurrent request handling.
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Any
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.performance_benchmarker import (
    PerformanceBenchmarker, 
    ThroughputBenchmark,
    NetworkBenchmark,
    ConcurrentRequestBenchmark
)

class BackendBenchmarkSuite:
    """
    Complete benchmark suite for backend performance validation
    """
    
    def __init__(self, results_dir: str = "swarm/phase2/benchmarks/backend/results"):
        self.benchmarker = PerformanceBenchmarker()
        self.results_dir = results_dir
        self.throughput_bench = ThroughputBenchmark(self.benchmarker)
        self.network_bench = NetworkBenchmark(self.benchmarker)
        self.concurrent_bench = ConcurrentRequestBenchmark(self.benchmarker)
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        # Register all benchmarks
        self._register_benchmarks()
    
    def _register_benchmarks(self):
        """Register all benchmark functions"""
        self.benchmarker.register_benchmark(
            'training_throughput', 
            self.throughput_bench.training_throughput_benchmark
        )
        self.benchmarker.register_benchmark(
            'websocket_latency', 
            self.network_bench.websocket_latency_benchmark
        )
        self.benchmarker.register_benchmark(
            'api_response_time', 
            self.network_bench.api_response_time_benchmark
        )
        self.benchmarker.register_benchmark(
            'concurrent_load', 
            self.concurrent_bench.concurrent_load_benchmark
        )
    
    async def run_full_comparison(self, 
                                 monolithic_config: Dict[str, Any],
                                 microservices_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run complete benchmark comparison between architectures
        """
        print("🚀 Starting comprehensive backend performance benchmark comparison")
        print("=" * 80)
        
        benchmark_names = [
            'training_throughput',
            'websocket_latency', 
            'api_response_time',
            'concurrent_load'
        ]
        
        # Run comparative benchmarks
        results = await self.benchmarker.run_comparative_benchmark(
            monolithic_config, microservices_config, benchmark_names
        )
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f"comparison_{timestamp}.json")
        self.benchmarker.save_results(results, results_file)
        
        # Generate and save report
        report = self.benchmarker.get_performance_report(results)
        report_file = os.path.join(self.results_dir, f"report_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📊 Results saved to: {results_file}")
        print(f"📋 Report saved to: {report_file}")
        print("\n" + report)
        
        return results
    
    async def run_single_architecture(self, 
                                     config: Dict[str, Any], 
                                     architecture_name: str) -> Dict[str, Any]:
        """
        Run benchmarks for a single architecture
        """
        print(f"🔧 Benchmarking {architecture_name} architecture")
        print("-" * 50)
        
        results = {}
        benchmark_configs = {
            'training_throughput': config.get('training', {}),
            'websocket_latency': config.get('websocket', {}),
            'api_response_time': config.get('api', {}),
            'concurrent_load': config.get('concurrent', {})
        }
        
        for benchmark_name, bench_config in benchmark_configs.items():
            print(f"Running {benchmark_name} benchmark...")
            result = await self.benchmarker.run_benchmark(benchmark_name, bench_config)
            results[benchmark_name] = result
            
            # Print quick summary
            print(f"  ✅ Completed - Throughput: {result.throughput:.2f}, "
                  f"Avg Latency: {result.latency_stats.get('avg', 0):.2f}ms")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f"{architecture_name}_{timestamp}.json")
        
        # Convert results for JSON serialization
        serializable_results = {
            name: {
                'name': result.name,
                'duration': result.duration,
                'throughput': result.throughput,
                'latency_stats': result.latency_stats,
                'resource_usage': result.resource_usage,
                'success_rate': result.success_rate,
                'timestamp': result.timestamp,
                'metadata': result.metadata
            } for name, result in results.items()
        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n📊 {architecture_name} results saved to: {results_file}")
        
        return results
    
    def validate_performance_requirements(self, results: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate that performance requirements are met
        """
        validation_results = {}
        
        if 'comparison' in results:
            comparison = results['comparison']
            
            # Performance regression check (no more than 5% degradation)
            validation_results['no_performance_regression'] = all(
                not analysis.get('performance_regression', True) 
                for analysis in comparison.values()
            )
            
            # Memory efficiency check (at least neutral, prefer improvement)
            validation_results['memory_efficiency'] = all(
                analysis.get('memory_change_percent', 0) <= 5.0
                for analysis in comparison.values()
            )
            
            # Latency requirement (no significant degradation)
            validation_results['latency_acceptable'] = all(
                analysis.get('latency_change_percent', 0) <= 10.0
                for analysis in comparison.values()
            )
            
            # Overall score requirement
            validation_results['overall_quality'] = all(
                analysis.get('overall_score', 'POOR') in ['GOOD', 'EXCELLENT']
                for analysis in comparison.values()
            )
            
            # Scalability improvement (concurrent handling should be better)
            concurrent_analysis = comparison.get('concurrent_load', {})
            validation_results['scalability_improvement'] = (
                concurrent_analysis.get('throughput_change_percent', -100) >= -5.0
            )
        
        return validation_results
    
    def generate_validation_report(self, 
                                  results: Dict[str, Any], 
                                  validation: Dict[str, bool]) -> str:
        """
        Generate a validation report
        """
        report = []
        report.append("🔍 PERFORMANCE VALIDATION REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Validation summary
        passed_checks = sum(validation.values())
        total_checks = len(validation)
        
        report.append(f"VALIDATION SUMMARY: {passed_checks}/{total_checks} checks passed")
        report.append("")
        
        # Detailed validation results
        status_map = {True: "✅ PASS", False: "❌ FAIL"}
        
        report.append("DETAILED VALIDATION RESULTS:")
        report.append("-" * 30)
        report.append(f"{status_map[validation.get('no_performance_regression', False)]:<10} No Performance Regression")
        report.append(f"{status_map[validation.get('memory_efficiency', False)]:<10} Memory Efficiency")
        report.append(f"{status_map[validation.get('latency_acceptable', False)]:<10} Latency Acceptable")
        report.append(f"{status_map[validation.get('overall_quality', False)]:<10} Overall Quality")
        report.append(f"{status_map[validation.get('scalability_improvement', False)]:<10} Scalability Improvement")
        report.append("")
        
        # Performance highlights
        if 'comparison' in results:
            report.append("PERFORMANCE HIGHLIGHTS:")
            report.append("-" * 25)
            
            for benchmark, analysis in results['comparison'].items():
                report.append(f"\n{benchmark.upper()}:")
                report.append(f"  Score: {analysis['overall_score']}")
                
                if analysis['throughput_change_percent'] > 0:
                    report.append(f"  🚀 Throughput improved by {analysis['throughput_change_percent']:.1f}%")
                elif analysis['throughput_change_percent'] < -5:
                    report.append(f"  ⚠️  Throughput decreased by {abs(analysis['throughput_change_percent']):.1f}%")
                
                if analysis['memory_change_percent'] < -10:
                    report.append(f"  💾 Memory usage reduced by {abs(analysis['memory_change_percent']):.1f}%")
                elif analysis['memory_change_percent'] > 20:
                    report.append(f"  ⚠️  Memory usage increased by {analysis['memory_change_percent']:.1f}%")
                
                if analysis['latency_change_percent'] < -5:
                    report.append(f"  ⚡ Latency improved by {abs(analysis['latency_change_percent']):.1f}%")
                elif analysis['latency_change_percent'] > 10:
                    report.append(f"  🐌 Latency increased by {analysis['latency_change_percent']:.1f}%")
        
        # Recommendations
        report.append("\n")
        report.append("RECOMMENDATIONS:")
        report.append("-" * 20)
        
        if validation.get('no_performance_regression', True):
            report.append("✅ Microservices architecture maintains performance characteristics")
        else:
            report.append("⚠️  Consider optimizing microservices for better throughput")
        
        if validation.get('memory_efficiency', True):
            report.append("✅ Memory usage is efficient in microservices architecture")
        else:
            report.append("💡 Investigate memory optimization opportunities")
        
        if validation.get('scalability_improvement', True):
            report.append("✅ Scalability characteristics improved")
        else:
            report.append("📈 Focus on improving concurrent request handling")
        
        return "\n".join(report)

def create_default_configs() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Create default configuration for benchmarking"""
    
    # Monolithic configuration
    monolithic_config = {
        'training': {
            'duration': 30,
            'concurrent_models': 5,
            'model_size': 'small'
        },
        'websocket': {
            'websocket_url': 'ws://localhost:8080/ws',
            'message_count': 500,
            'concurrent_connections': 5
        },
        'api': {
            'api_url': 'http://localhost:8080',
            'endpoints': ['/health', '/api/status', '/api/models'],
            'requests_per_endpoint': 50,
            'concurrent_requests': 10
        },
        'concurrent': {
            'api_url': 'http://localhost:8080',
            'max_concurrent': 50,
            'ramp_up_duration': 15,
            'steady_duration': 30,
            'endpoint': '/api/health'
        }
    }
    
    # Microservices configuration
    microservices_config = {
        'training': {
            'duration': 30,
            'concurrent_models': 5,
            'model_size': 'small'
        },
        'websocket': {
            'websocket_url': 'ws://localhost:8081/ws',  # Different port for microservice
            'message_count': 500,
            'concurrent_connections': 5
        },
        'api': {
            'api_url': 'http://localhost:8081',  # API gateway
            'endpoints': ['/health', '/api/status', '/api/models'],
            'requests_per_endpoint': 50,
            'concurrent_requests': 10
        },
        'concurrent': {
            'api_url': 'http://localhost:8081',
            'max_concurrent': 50,
            'ramp_up_duration': 15,
            'steady_duration': 30,
            'endpoint': '/api/health'
        }
    }
    
    return monolithic_config, microservices_config

async def main():
    """Main benchmark execution function"""
    parser = argparse.ArgumentParser(description='Backend Performance Benchmark Suite')
    parser.add_argument('--mode', choices=['comparison', 'monolithic', 'microservices'], 
                       default='comparison', help='Benchmark mode')
    parser.add_argument('--config', type=str, help='Config file path (JSON)')
    parser.add_argument('--results-dir', type=str, 
                       default='swarm/phase2/benchmarks/backend/results',
                       help='Results directory')
    
    args = parser.parse_args()
    
    # Initialize benchmark suite
    suite = BackendBenchmarkSuite(results_dir=args.results_dir)
    
    # Load or create configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        monolithic_config = config.get('monolithic', {})
        microservices_config = config.get('microservices', {})
    else:
        monolithic_config, microservices_config = create_default_configs()
    
    try:
        if args.mode == 'comparison':
            # Run full comparison
            results = await suite.run_full_comparison(monolithic_config, microservices_config)
            
            # Validate performance
            validation = suite.validate_performance_requirements(results)
            
            # Generate validation report
            validation_report = suite.generate_validation_report(results, validation)
            
            # Save validation report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            validation_file = os.path.join(args.results_dir, f"validation_{timestamp}.txt")
            with open(validation_file, 'w') as f:
                f.write(validation_report)
            
            print("\n" + validation_report)
            print(f"\n📋 Validation report saved to: {validation_file}")
            
        elif args.mode == 'monolithic':
            await suite.run_single_architecture(monolithic_config, 'monolithic')
            
        elif args.mode == 'microservices':
            await suite.run_single_architecture(microservices_config, 'microservices')
    
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    # Set up proper event loop for Windows
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())