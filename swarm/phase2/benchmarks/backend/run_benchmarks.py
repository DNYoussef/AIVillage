#!/usr/bin/env python3
"""
Unified Benchmark Runner

Main entry point for running comprehensive performance benchmarks
comparing monolithic vs microservices backend architectures.
"""

import asyncio
import sys
import argparse
import json
import os
import logging
from datetime import datetime
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from suite.benchmark_suite import BackendBenchmarkSuite
from tools.config_manager import ConfigManager
from tools.memory_profiler import MemoryProfiler, MemoryBenchmarkIntegration
from tools.regression_detector import RegressionDetector
from tools.visualization import PerformanceVisualizer

class BenchmarkRunner:
    """
    Main benchmark execution and coordination class
    """
    
    def __init__(self, results_dir: str = None, config_dir: str = None):
        self.results_dir = results_dir or "swarm/phase2/benchmarks/backend/results"
        self.config_dir = config_dir or "swarm/phase2/benchmarks/backend/config"
        
        # Ensure directories exist
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.config_manager = ConfigManager(self.config_dir)
        self.benchmark_suite = BackendBenchmarkSuite(self.results_dir)
        self.memory_profiler = MemoryProfiler()
        self.memory_integration = MemoryBenchmarkIntegration(self.memory_profiler)
        self.regression_detector = RegressionDetector()
        self.visualizer = PerformanceVisualizer(
            os.path.join(self.results_dir, "../reports")
        )
        
        self.logger = self._setup_logging()
        
        # Export default configurations
        self.config_manager.export_configs()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup benchmark runner logging"""
        logger = logging.getLogger('benchmark_runner')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_file = os.path.join(self.results_dir, 'benchmark.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    async def run_full_benchmark_suite(self, 
                                      workload_profile: str = 'default',
                                      include_memory_profiling: bool = True,
                                      include_regression_analysis: bool = True,
                                      create_visualizations: bool = True) -> Dict[str, Any]:
        """
        Run the complete benchmark suite with all analysis
        """
        
        self.logger.info("🚀 Starting comprehensive backend performance benchmark suite")
        self.logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Generate optimized configurations
            self.logger.info("📋 Generating optimized benchmark configurations...")
            
            monolithic_config = self.config_manager.get_optimized_config('monolithic', workload_profile)
            microservices_config = self.config_manager.get_optimized_config('microservices', workload_profile)
            
            # Validate configurations
            mono_issues = self.config_manager.validate_config(monolithic_config)
            micro_issues = self.config_manager.validate_config(microservices_config)
            
            if mono_issues or micro_issues:
                self.logger.warning("Configuration validation issues detected:")
                for issue in mono_issues + micro_issues:
                    self.logger.warning(f"  - {issue}")
            
            # Step 2: Run benchmarks with optional memory profiling
            self.logger.info("🔧 Executing performance benchmarks...")
            
            if include_memory_profiling:
                results = await self._run_benchmarks_with_memory_profiling(
                    monolithic_config, microservices_config
                )
            else:
                results = await self.benchmark_suite.run_full_comparison(
                    monolithic_config, microservices_config
                )
            
            # Step 3: Performance validation
            self.logger.info("✅ Validating performance requirements...")
            validation_results = self.benchmark_suite.validate_performance_requirements(results)
            
            # Step 4: Regression analysis
            regression_alerts = []
            if include_regression_analysis:
                self.logger.info("🔍 Performing regression analysis...")
                regression_alerts = self._perform_regression_analysis(results)
            
            # Step 5: Generate comprehensive reports
            self.logger.info("📊 Generating performance reports...")
            
            # Performance validation report
            validation_report = self.benchmark_suite.generate_validation_report(
                results, validation_results
            )
            
            # Regression report
            regression_report = ""
            if regression_alerts:
                regression_report = self.regression_detector.generate_regression_report(
                    regression_alerts
                )
            
            # Step 6: Create visualizations
            visualization_files = {}
            if create_visualizations:
                self.logger.info("📈 Creating performance visualizations...")
                visualization_files = self.visualizer.create_all_visualizations(
                    results, validation_results
                )
            
            # Step 7: Save all results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save benchmark results
            results_file = os.path.join(self.results_dir, f"comprehensive_results_{timestamp}.json")
            self.benchmark_suite.benchmarker.save_results(results, results_file)
            
            # Save validation report
            validation_file = os.path.join(self.results_dir, f"validation_report_{timestamp}.txt")
            with open(validation_file, 'w') as f:
                f.write(validation_report)
            
            # Save regression report
            if regression_report:
                regression_file = os.path.join(self.results_dir, f"regression_report_{timestamp}.txt")
                with open(regression_file, 'w') as f:
                    f.write(regression_report)
            
            # Create summary
            execution_time = datetime.now() - start_time
            summary = self._create_execution_summary(
                results, validation_results, regression_alerts,
                execution_time, visualization_files
            )
            
            # Save execution summary
            summary_file = os.path.join(self.results_dir, f"execution_summary_{timestamp}.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Display final summary
            self._display_final_summary(summary)
            
            return {
                'results': results,
                'validation': validation_results,
                'regression_alerts': regression_alerts,
                'summary': summary,
                'files': {
                    'results': results_file,
                    'validation': validation_file,
                    'regression': regression_file if regression_report else None,
                    'summary': summary_file,
                    'visualizations': visualization_files
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Benchmark suite execution failed: {e}")
            raise
    
    async def _run_benchmarks_with_memory_profiling(self,
                                                   monolithic_config: Dict[str, Any],
                                                   microservices_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run benchmarks with integrated memory profiling"""
        
        self.logger.info("🧠 Running benchmarks with memory profiling enabled")
        
        benchmark_names = ['training_throughput', 'websocket_latency', 'api_response_time', 'concurrent_load']
        
        results = {
            'monolithic': {},
            'microservices': {},
            'comparison': {}
        }
        
        # Profile monolithic architecture
        self.logger.info("📊 Profiling monolithic architecture...")
        for benchmark_name in benchmark_names:
            config = monolithic_config.get(benchmark_name.split('_')[0], {})
            
            result = await self.memory_integration.benchmark_with_memory_profiling(
                self.benchmark_suite.benchmarker.benchmarks[benchmark_name], config
            )
            
            results['monolithic'][benchmark_name] = result
            self.logger.info(f"  ✅ Completed {benchmark_name}")
        
        # Short break between architectures
        await asyncio.sleep(5)
        
        # Profile microservices architecture
        self.logger.info("📊 Profiling microservices architecture...")
        for benchmark_name in benchmark_names:
            config = microservices_config.get(benchmark_name.split('_')[0], {})
            
            result = await self.memory_integration.benchmark_with_memory_profiling(
                self.benchmark_suite.benchmarker.benchmarks[benchmark_name], config
            )
            
            results['microservices'][benchmark_name] = result
            self.logger.info(f"  ✅ Completed {benchmark_name}")
        
        # Generate comparison analysis
        results['comparison'] = self.benchmark_suite.benchmarker._generate_comparison_analysis(
            results['monolithic'], results['microservices']
        )
        
        # Add memory profile comparison
        memory_comparison = {}
        for benchmark_name in benchmark_names:
            mono_profile = results['monolithic'][benchmark_name].get('memory_profile', {})
            micro_profile = results['microservices'][benchmark_name].get('memory_profile', {})
            
            if mono_profile and micro_profile:
                memory_comparison[benchmark_name] = self.memory_integration.compare_memory_profiles(
                    mono_profile, micro_profile
                )
        
        results['memory_comparison'] = memory_comparison
        
        return results
    
    def _perform_regression_analysis(self, results: Dict[str, Any]) -> List:
        """Perform comprehensive regression analysis"""
        
        # For now, we'll compare against baseline (monolithic as baseline)
        baseline_results = results.get('monolithic', {})
        current_results = results.get('microservices', {})
        
        if not baseline_results or not current_results:
            self.logger.warning("Insufficient data for regression analysis")
            return []
        
        # Perform regression detection
        regression_alerts = self.regression_detector.analyze_regression(
            current_results, baseline_results
        )
        
        # Also check for anomalies if we have historical data
        # (In a real implementation, this would load historical benchmark results)
        
        return regression_alerts
    
    def _create_execution_summary(self, results: Dict[str, Any],
                                validation: Dict[str, bool],
                                regression_alerts: List,
                                execution_time,
                                visualization_files: Dict[str, str]) -> Dict[str, Any]:
        """Create comprehensive execution summary"""
        
        # Count successful benchmarks
        mono_benchmarks = len(results.get('monolithic', {}))
        micro_benchmarks = len(results.get('microservices', {}))
        
        # Analyze comparison results
        comparison = results.get('comparison', {})
        scores = [analysis.get('overall_score', 'POOR') for analysis in comparison.values()]
        score_counts = {score: scores.count(score) for score in ['EXCELLENT', 'GOOD', 'FAIR', 'POOR']}
        
        # Performance highlights
        avg_throughput_change = sum(
            analysis.get('throughput_change_percent', 0)
            for analysis in comparison.values()
        ) / len(comparison) if comparison else 0
        
        avg_memory_change = sum(
            analysis.get('memory_change_percent', 0)
            for analysis in comparison.values()
        ) / len(comparison) if comparison else 0
        
        avg_latency_change = sum(
            analysis.get('latency_change_percent', 0)
            for analysis in comparison.values()
        ) / len(comparison) if comparison else 0
        
        # Regression summary
        regression_summary = {
            'total_alerts': len(regression_alerts),
            'critical_alerts': len([a for a in regression_alerts if a.severity.value == 'critical']),
            'high_alerts': len([a for a in regression_alerts if a.severity.value == 'high']),
            'medium_alerts': len([a for a in regression_alerts if a.severity.value == 'medium']),
            'low_alerts': len([a for a in regression_alerts if a.severity.value == 'low'])
        }
        
        return {
            'execution_info': {
                'start_time': datetime.now().isoformat(),
                'duration_seconds': execution_time.total_seconds(),
                'benchmarks_executed': mono_benchmarks + micro_benchmarks,
                'architectures_tested': ['monolithic', 'microservices']
            },
            'performance_summary': {
                'score_distribution': score_counts,
                'avg_throughput_change_percent': round(avg_throughput_change, 2),
                'avg_memory_change_percent': round(avg_memory_change, 2),
                'avg_latency_change_percent': round(avg_latency_change, 2)
            },
            'validation_summary': {
                'total_checks': len(validation),
                'passed_checks': sum(validation.values()),
                'success_rate': sum(validation.values()) / len(validation) * 100 if validation else 0,
                'details': validation
            },
            'regression_summary': regression_summary,
            'files_generated': len(visualization_files) + 3,  # +3 for reports
            'recommendations': self._generate_recommendations(
                results, validation, regression_alerts
            )
        }
    
    def _generate_recommendations(self, results: Dict[str, Any],
                                validation: Dict[str, bool],
                                regression_alerts: List) -> List[str]:
        """Generate actionable recommendations based on results"""
        
        recommendations = []
        
        # Overall assessment
        passed_validation = sum(validation.values()) if validation else 0
        total_validation = len(validation) if validation else 1
        
        if passed_validation / total_validation >= 0.8:
            recommendations.append("✅ Microservices refactoring is successful - proceed with deployment")
        elif passed_validation / total_validation >= 0.6:
            recommendations.append("⚠️ Microservices shows promise but needs optimization before deployment")
        else:
            recommendations.append("❌ Microservices architecture needs significant improvements")
        
        # Specific recommendations based on comparison
        comparison = results.get('comparison', {})
        
        for benchmark, analysis in comparison.items():
            score = analysis.get('overall_score', 'POOR')
            
            if score == 'POOR':
                recommendations.append(f"🔧 Priority: Optimize {benchmark.replace('_', ' ')} performance")
            elif score == 'FAIR':
                recommendations.append(f"📈 Consider: Improve {benchmark.replace('_', ' ')} efficiency")
        
        # Memory recommendations
        memory_comparison = results.get('memory_comparison', {})
        for benchmark, mem_analysis in memory_comparison.items():
            efficiency_rating = mem_analysis.get('efficiency_rating', 'POOR')
            
            if efficiency_rating == 'POOR':
                recommendations.append(f"💾 Critical: Address memory issues in {benchmark}")
            elif efficiency_rating == 'FAIR':
                recommendations.append(f"💡 Optimize: Memory usage in {benchmark}")
        
        # Regression-based recommendations
        critical_alerts = [a for a in regression_alerts if a.severity.value == 'critical']
        if critical_alerts:
            recommendations.append("🚨 Urgent: Address critical performance regressions before deployment")
        
        high_alerts = [a for a in regression_alerts if a.severity.value == 'high']
        if high_alerts:
            recommendations.append("⚠️ Important: Resolve high-priority performance issues")
        
        # Infrastructure recommendations
        if not validation.get('scalability_improvement', True):
            recommendations.append("📊 Focus: Improve concurrent request handling capabilities")
        
        if not validation.get('memory_efficiency', True):
            recommendations.append("🔧 Optimize: Memory allocation and garbage collection settings")
        
        return recommendations
    
    def _display_final_summary(self, summary: Dict[str, Any]):
        """Display final benchmark summary"""
        
        print("\n" + "=" * 80)
        print("🎯 BENCHMARK EXECUTION SUMMARY")
        print("=" * 80)
        
        exec_info = summary['execution_info']
        perf_summary = summary['performance_summary']
        val_summary = summary['validation_summary']
        
        print(f"⏱️  Execution Time: {exec_info['duration_seconds']:.1f} seconds")
        print(f"🔧 Benchmarks Executed: {exec_info['benchmarks_executed']}")
        print(f"🏗️  Architectures Tested: {', '.join(exec_info['architectures_tested'])}")
        
        print(f"\n📊 PERFORMANCE OVERVIEW:")
        print(f"   Throughput Change: {perf_summary['avg_throughput_change_percent']:+.1f}%")
        print(f"   Memory Change: {perf_summary['avg_memory_change_percent']:+.1f}%")
        print(f"   Latency Change: {perf_summary['avg_latency_change_percent']:+.1f}%")
        
        print(f"\n✅ VALIDATION RESULTS:")
        print(f"   Success Rate: {val_summary['success_rate']:.1f}% ({val_summary['passed_checks']}/{val_summary['total_checks']})")
        
        print(f"\n📋 KEY RECOMMENDATIONS:")
        for rec in summary['recommendations'][:5]:  # Show top 5
            print(f"   {rec}")
        
        print(f"\n📁 Generated {summary['files_generated']} output files")
        print("=" * 80)

async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Comprehensive Backend Performance Benchmark Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmarks.py                           # Full benchmark suite
  python run_benchmarks.py --workload high_throughput   # Optimized for throughput
  python run_benchmarks.py --workload low_latency       # Optimized for latency
  python run_benchmarks.py --no-memory                  # Skip memory profiling
  python run_benchmarks.py --no-visualizations         # Skip chart generation
        """
    )
    
    parser.add_argument('--workload', 
                       choices=['default', 'high_throughput', 'low_latency', 'memory_constrained', 'stress_test'],
                       default='default',
                       help='Workload profile for optimization')
    
    parser.add_argument('--results-dir', type=str,
                       default='swarm/phase2/benchmarks/backend/results',
                       help='Results output directory')
    
    parser.add_argument('--config-dir', type=str,
                       default='swarm/phase2/benchmarks/backend/config', 
                       help='Configuration directory')
    
    parser.add_argument('--no-memory', action='store_true',
                       help='Skip memory profiling (faster execution)')
    
    parser.add_argument('--no-regression', action='store_true',
                       help='Skip regression analysis')
    
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip visualization generation')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize benchmark runner
        runner = BenchmarkRunner(args.results_dir, args.config_dir)
        
        # Run comprehensive benchmark suite
        results = await runner.run_full_benchmark_suite(
            workload_profile=args.workload,
            include_memory_profiling=not args.no_memory,
            include_regression_analysis=not args.no_regression,
            create_visualizations=not args.no_visualizations
        )
        
        print(f"\n🎉 Benchmark suite completed successfully!")
        print(f"📊 Results saved to: {args.results_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        logging.exception("Benchmark execution failed")
        return 1

if __name__ == "__main__":
    # Set up proper event loop for Windows
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    sys.exit(asyncio.run(main()))