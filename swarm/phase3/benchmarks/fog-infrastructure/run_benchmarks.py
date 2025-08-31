"""
Main benchmark runner for Phase 3 Fog Infrastructure Performance Validation
Complete orchestration of system, privacy, graph, and integration benchmarks.
"""

import asyncio
import time
import logging
import argparse
from pathlib import Path
import sys
import os

# Add project paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))

# Import our benchmark components
from benchmark_suite import PerformanceBenchmarkSuite
from framework.validation_framework import ValidationFramework

class BenchmarkOrchestrator:
    """Main orchestrator for Phase 3 performance benchmarks"""
    
    def __init__(self, output_dir: str = None, verbose: bool = False):
        self.output_dir = Path(output_dir or "swarm/phase3/benchmarks/fog-infrastructure/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'benchmark_execution.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.benchmark_suite = PerformanceBenchmarkSuite(str(self.output_dir))
        self.validation_framework = ValidationFramework(str(self.output_dir))

    async def run_complete_benchmark_validation(self, create_baseline: bool = False, 
                                               skip_validation: bool = False) -> Dict[str, Any]:
        """Run complete benchmark validation process"""
        
        self.logger.info("=" * 80)
        self.logger.info("PHASE 3 FOG INFRASTRUCTURE PERFORMANCE BENCHMARK SUITE")
        self.logger.info("=" * 80)
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Create baseline: {create_baseline}")
        self.logger.info(f"Skip validation: {skip_validation}")
        self.logger.info("=" * 80)
        
        overall_start_time = time.time()
        
        try:
            # Step 1: Create baseline if requested
            if create_baseline:
                self.logger.info("Creating performance baseline...")
                baseline_result = await self.validation_framework.create_performance_baseline()
                self.logger.info("Baseline creation completed")
                
                if skip_validation:
                    return {
                        'operation': 'baseline_creation',
                        'success': True,
                        'baseline_data': baseline_result,
                        'execution_time': time.time() - overall_start_time
                    }
            
            # Step 2: Run comprehensive benchmarks
            self.logger.info("Running comprehensive performance benchmarks...")
            benchmark_results = await self.benchmark_suite.run_complete_benchmark_suite()
            self.logger.info("Benchmark execution completed")
            
            # Step 3: Run validation if not skipped
            if not skip_validation:
                self.logger.info("Running performance validation...")
                validation_results = await self.validation_framework.run_comprehensive_validation(
                    compare_with_baseline=not create_baseline
                )
                self.logger.info("Validation completed")
                
                # Step 4: Generate comprehensive report
                final_report = await self._generate_final_report(benchmark_results, validation_results)
                
                total_execution_time = time.time() - overall_start_time
                
                # Step 5: Print executive summary
                self._print_executive_summary(validation_results, total_execution_time)
                
                return {
                    'operation': 'complete_validation',
                    'success': True,
                    'benchmark_results': benchmark_results,
                    'validation_results': validation_results,
                    'final_report': final_report,
                    'execution_time': total_execution_time
                }
            else:
                total_execution_time = time.time() - overall_start_time
                
                return {
                    'operation': 'benchmark_only',
                    'success': True,
                    'benchmark_results': benchmark_results,
                    'execution_time': total_execution_time
                }
                
        except Exception as e:
            self.logger.error(f"Benchmark validation failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            return {
                'operation': 'failed',
                'success': False,
                'error': str(e),
                'execution_time': time.time() - overall_start_time
            }

    async def _generate_final_report(self, benchmark_results: Dict[str, Any], 
                                   validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        
        self.logger.info("Generating final comprehensive report...")
        
        # Extract key metrics
        key_metrics = self._extract_key_metrics(benchmark_results, validation_results)
        
        # Performance achievements
        achievements = self._identify_achievements(validation_results)
        
        # Critical issues
        critical_issues = validation_results['validation_summary']['critical_issues']
        
        # Deployment readiness assessment
        deployment_readiness = self._assess_deployment_readiness(validation_results)
        
        # Generate recommendations
        recommendations = self._generate_strategic_recommendations(validation_results, deployment_readiness)
        
        final_report = {
            'executive_summary': {
                'overall_grade': validation_results['validation_summary']['overall_grade'],
                'pass_rate': f"{validation_results['validation_summary']['passed_benchmarks']}/{validation_results['validation_summary']['total_benchmarks']}",
                'average_improvement': validation_results['validation_summary']['average_improvement'],
                'regressions_detected': validation_results['validation_summary']['regressions_detected'],
                'deployment_ready': deployment_readiness['ready_for_deployment']
            },
            'key_metrics': key_metrics,
            'achievements': achievements,
            'critical_issues': critical_issues,
            'deployment_readiness': deployment_readiness,
            'strategic_recommendations': recommendations,
            'detailed_analysis': {
                'benchmark_results': benchmark_results,
                'validation_results': validation_results
            }
        }
        
        # Save final report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"final_report_{timestamp}.json"
        
        import json
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        self.logger.info(f"Final report saved to: {report_file}")
        
        return final_report

    def _extract_key_metrics(self, benchmark_results: Dict[str, Any], validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key performance metrics"""
        
        key_metrics = {}
        
        # Find key benchmark results
        for result in validation_results['detailed_results']:
            metric_name = result['benchmark_name']
            
            # Map to user-friendly names
            friendly_names = {
                'fog_coordinator_optimization': 'Fog Coordinator Improvement',
                'onion_coordinator_optimization': 'Privacy Coordinator Improvement',
                'graph_gap_detection_optimization': 'Graph Processing Improvement',
                'system_startup_time': 'System Startup Time',
                'device_registration_time': 'Device Registration Time',
                'cross_service_communication': 'Cross-Service Latency',
                'end_to_end_workflows': 'End-to-End Latency'
            }
            
            if metric_name in friendly_names:
                key_metrics[friendly_names[metric_name]] = {
                    'current_value': result['current_value'],
                    'target': result['target_improvement'],
                    'improvement_percent': result['improvement_percent'],
                    'target_met': result['target_met'],
                    'grade': result['performance_grade']
                }
        
        return key_metrics

    def _identify_achievements(self, validation_results: Dict[str, Any]) -> List[str]:
        """Identify notable achievements"""
        
        achievements = []
        
        # High-performing benchmarks
        for result in validation_results['detailed_results']:
            if result['target_met'] and result['performance_grade'] == 'A':
                if result['improvement_percent'] and result['improvement_percent'] > result['target_improvement']:
                    achievements.append(
                        f"{result['benchmark_name']}: {result['improvement_percent']:.1f}% improvement "
                        f"(exceeded {result['target_improvement']}% target)"
                    )
        
        # Category achievements
        categories = {}
        for result in validation_results['detailed_results']:
            category = result['category']
            if category not in categories:
                categories[category] = {'total': 0, 'passed': 0}
            categories[category]['total'] += 1
            if result['target_met']:
                categories[category]['passed'] += 1
        
        for category, data in categories.items():
            if data['passed'] == data['total'] and data['total'] >= 3:
                achievements.append(f"All {category} benchmarks passed ({data['total']}/{data['total']})")
        
        # Overall achievements
        overall_grade = validation_results['validation_summary']['overall_grade']
        if overall_grade in ['A', 'B']:
            achievements.append(f"Excellent overall performance grade: {overall_grade}")
        
        return achievements[:8]  # Top 8 achievements

    def _assess_deployment_readiness(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess deployment readiness"""
        
        validation_summary = validation_results['validation_summary']
        
        # Deployment criteria
        criteria = {
            'overall_grade_acceptable': validation_summary['overall_grade'] in ['A', 'B', 'C'],
            'no_critical_regressions': validation_results['regression_analysis']['critical_regressions'] == 0,
            'core_benchmarks_passed': self._check_core_benchmarks_passed(validation_results),
            'acceptable_pass_rate': validation_summary['passed_benchmarks'] / validation_summary['total_benchmarks'] >= 0.75
        }
        
        ready_for_deployment = all(criteria.values())
        
        # Determine deployment recommendation
        if ready_for_deployment:
            if validation_summary['overall_grade'] == 'A':
                recommendation = 'Ready for production deployment'
            elif validation_summary['overall_grade'] == 'B':
                recommendation = 'Ready for production with monitoring'
            else:
                recommendation = 'Ready for staged deployment'
        else:
            if not criteria['no_critical_regressions']:
                recommendation = 'Critical regressions must be resolved before deployment'
            elif not criteria['core_benchmarks_passed']:
                recommendation = 'Core functionality benchmarks must pass before deployment'
            else:
                recommendation = 'Performance improvements needed before deployment'
        
        return {
            'ready_for_deployment': ready_for_deployment,
            'recommendation': recommendation,
            'criteria': criteria,
            'risk_level': self._calculate_deployment_risk(criteria, validation_summary)
        }

    def _check_core_benchmarks_passed(self, validation_results: Dict[str, Any]) -> bool:
        """Check if core benchmarks have passed"""
        
        core_benchmarks = [
            'fog_coordinator_optimization',
            'onion_coordinator_optimization', 
            'graph_gap_detection_optimization'
        ]
        
        core_results = [
            result for result in validation_results['detailed_results']
            if result['benchmark_name'] in core_benchmarks
        ]
        
        return all(result['target_met'] for result in core_results)

    def _calculate_deployment_risk(self, criteria: Dict[str, bool], validation_summary: Dict[str, Any]) -> str:
        """Calculate deployment risk level"""
        
        if all(criteria.values()) and validation_summary['overall_grade'] == 'A':
            return 'Low'
        elif all(criteria.values()):
            return 'Medium'
        elif criteria['no_critical_regressions'] and criteria['core_benchmarks_passed']:
            return 'Medium-High'
        else:
            return 'High'

    def _generate_strategic_recommendations(self, validation_results: Dict[str, Any], 
                                          deployment_readiness: Dict[str, Any]) -> List[str]:
        """Generate strategic recommendations"""
        
        recommendations = []
        
        # Deployment-specific recommendations
        if not deployment_readiness['ready_for_deployment']:
            recommendations.append("Address deployment blockers before proceeding to production")
        
        # Performance-specific recommendations
        validation_summary = validation_results['validation_summary']
        
        if validation_summary['regressions_detected'] > 0:
            recommendations.append(f"Investigate and resolve {validation_summary['regressions_detected']} performance regressions")
        
        if validation_summary['average_improvement'] < 50.0:
            recommendations.append("Consider additional optimization efforts to achieve target improvements")
        
        # Category-specific recommendations based on failures
        failed_categories = {}
        for result in validation_results['detailed_results']:
            if not result['target_met']:
                category = result['category']
                if category not in failed_categories:
                    failed_categories[category] = 0
                failed_categories[category] += 1
        
        for category, count in failed_categories.items():
            if count >= 2:
                recommendations.append(f"Focus optimization efforts on {category} performance")
        
        # Strategic recommendations
        recommendations.extend([
            "Establish continuous performance monitoring in production",
            "Set up automated performance regression detection",
            "Create performance budgets for future feature development",
            "Schedule regular performance review cycles"
        ])
        
        return recommendations[:10]  # Top 10 recommendations

    def _print_executive_summary(self, validation_results: Dict[str, Any], execution_time: float):
        """Print executive summary to console"""
        
        validation_summary = validation_results['validation_summary']
        
        print("\n" + "=" * 80)
        print("PHASE 3 PERFORMANCE VALIDATION - EXECUTIVE SUMMARY")
        print("=" * 80)
        
        print(f"Overall Grade: {validation_summary['overall_grade']}")
        print(f"Benchmarks Passed: {validation_summary['passed_benchmarks']}/{validation_summary['total_benchmarks']} ({validation_summary['passed_benchmarks']/validation_summary['total_benchmarks']*100:.1f}%)")
        print(f"Average Improvement: {validation_summary['average_improvement']:.1f}%")
        print(f"Regressions Detected: {validation_summary['regressions_detected']}")
        print(f"Total Execution Time: {execution_time:.1f} seconds")
        
        if validation_summary['critical_issues']:
            print(f"\nCritical Issues ({len(validation_summary['critical_issues'])}):")
            for issue in validation_summary['critical_issues'][:3]:
                print(f"  • {issue}")
            if len(validation_summary['critical_issues']) > 3:
                print(f"  ... and {len(validation_summary['critical_issues']) - 3} more")
        
        print(f"\nRecommendations:")
        for recommendation in validation_summary['recommendations'][:3]:
            print(f"  • {recommendation}")
        if len(validation_summary['recommendations']) > 3:
            print(f"  ... and {len(validation_summary['recommendations']) - 3} more")
        
        print("\n" + "=" * 80)

    async def run_quick_validation(self) -> Dict[str, Any]:
        """Run quick validation with subset of benchmarks"""
        
        self.logger.info("Running quick validation...")
        
        # Run only critical benchmarks
        quick_results = await self.benchmark_suite.run_complete_benchmark_suite()
        
        # Quick validation
        validation_results = await self.validation_framework.run_comprehensive_validation()
        
        return {
            'quick_validation': True,
            'validation_results': validation_results,
            'benchmark_results': quick_results
        }

async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Phase 3 Fog Infrastructure Performance Benchmarks')
    parser.add_argument('--output-dir', '-o', type=str, 
                       help='Output directory for benchmark results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--create-baseline', '-b', action='store_true',
                       help='Create performance baseline')
    parser.add_argument('--skip-validation', '-s', action='store_true',
                       help='Skip validation, only run benchmarks')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Run quick validation with subset of benchmarks')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = BenchmarkOrchestrator(
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    try:
        if args.quick:
            results = await orchestrator.run_quick_validation()
        else:
            results = await orchestrator.run_complete_benchmark_validation(
                create_baseline=args.create_baseline,
                skip_validation=args.skip_validation
            )
        
        if results['success']:
            print(f"\n✅ Benchmark validation completed successfully!")
            if 'validation_results' in results:
                grade = results['validation_results']['validation_summary']['overall_grade']
                print(f"📊 Overall Grade: {grade}")
        else:
            print(f"\n❌ Benchmark validation failed: {results['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())