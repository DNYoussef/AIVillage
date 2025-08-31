"""
Automated Validation Framework for Phase 3 Performance Benchmarks
Before/after comparison, regression detection, and automated validation.
"""

import asyncio
import time
import json
import statistics
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import os

# Add project paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../..'))

# Import our benchmark modules
from ..system.fog_system_benchmarks import FogSystemBenchmarks
from ..privacy.privacy_performance_benchmarks import PrivacyPerformanceBenchmarks
from ..graph.graph_performance_benchmarks import GraphPerformanceBenchmarks
from ..integration.integration_benchmarks import IntegrationBenchmarks

@dataclass
class ValidationResult:
    """Validation result for a specific benchmark"""
    benchmark_name: str
    category: str
    baseline_value: Optional[float]
    current_value: float
    improvement_percent: float
    target_improvement: float
    target_met: bool
    regression_detected: bool
    performance_grade: str
    timestamp: float

@dataclass
class ValidationSummary:
    """Overall validation summary"""
    total_benchmarks: int
    passed_benchmarks: int
    failed_benchmarks: int
    regressions_detected: int
    average_improvement: float
    overall_grade: str
    critical_issues: List[str]
    recommendations: List[str]

class ValidationFramework:
    """Comprehensive validation framework for performance benchmarks"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or "swarm/phase3/benchmarks/fog-infrastructure/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Validation targets from Phase 3 requirements
        self.validation_targets = {
            'fog_coordinator_improvement': 70.0,  # 60-80% target
            'onion_coordinator_improvement': 40.0,  # 30-50% target
            'graph_fixer_improvement': 50.0,       # 40-60% target
            'system_startup_time': 30.0,           # seconds
            'device_registration_time': 2.0,       # seconds
            'privacy_task_routing_time': 3.0,      # seconds
            'graph_gap_detection_time': 30.0,      # seconds for 1000 nodes
            'memory_reduction_percent': 30.0,      # 20-40% target
            'coupling_reduction_percent': 70.0,    # minimum target
            'cross_service_latency_ms': 50.0,      # Max 50ms
            'coordination_overhead_percent': 10.0,  # Max 10% overhead
            'end_to_end_latency_ms': 500.0         # Max 500ms
        }
        
        # Regression detection thresholds
        self.regression_thresholds = {
            'performance_degradation': 5.0,    # 5% degradation triggers regression
            'memory_increase': 10.0,           # 10% memory increase triggers regression
            'latency_increase': 15.0,          # 15% latency increase triggers regression
            'success_rate_decrease': 2.0       # 2% success rate decrease triggers regression
        }
        
        self.validation_results: List[ValidationResult] = []
        self.baseline_data: Dict[str, Any] = {}

    async def run_comprehensive_validation(self, compare_with_baseline: bool = True) -> Dict[str, Any]:
        """Run comprehensive validation of all benchmark categories"""
        self.logger.info("Starting comprehensive benchmark validation")
        
        start_time = time.time()
        
        try:
            # Load baseline data if available
            if compare_with_baseline:
                await self._load_baseline_data()
            
            # Run all benchmark categories
            benchmark_results = await self._run_all_benchmarks()
            
            # Validate results against targets and baselines
            validation_results = await self._validate_benchmark_results(benchmark_results)
            
            # Detect regressions
            regression_results = await self._detect_regressions(validation_results)
            
            # Generate comprehensive summary
            validation_summary = self._generate_validation_summary(validation_results, regression_results)
            
            # Save validation results
            await self._save_validation_results(validation_results, validation_summary)
            
            total_duration = time.time() - start_time
            
            return {
                'validation_summary': asdict(validation_summary),
                'detailed_results': [asdict(r) for r in validation_results],
                'regression_analysis': regression_results,
                'benchmark_results': benchmark_results,
                'total_validation_time': total_duration,
                'validation_success': validation_summary.overall_grade in ['A', 'B', 'C'],
                'critical_failures': validation_summary.critical_issues
            }
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            raise

    async def _load_baseline_data(self):
        """Load baseline performance data for comparison"""
        baseline_file = self.output_dir / "baseline_performance.json"
        
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                self.baseline_data = json.load(f)
            self.logger.info(f"Loaded baseline data from {baseline_file}")
        else:
            self.logger.warning("No baseline data found - will create new baseline")

    async def _run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark categories"""
        self.logger.info("Running all benchmark categories")
        
        # Initialize benchmark classes
        system_benchmarks = FogSystemBenchmarks()
        privacy_benchmarks = PrivacyPerformanceBenchmarks()
        graph_benchmarks = GraphPerformanceBenchmarks()
        integration_benchmarks = IntegrationBenchmarks()
        
        # Run benchmarks concurrently where possible
        benchmark_tasks = {
            'system': system_benchmarks.run_fog_system_benchmarks(),
            'privacy': privacy_benchmarks.run_privacy_performance_benchmarks(),
            'graph': graph_benchmarks.run_graph_performance_benchmarks(),
            'integration': integration_benchmarks.run_integration_benchmarks()
        }
        
        # Execute all benchmarks
        benchmark_results = {}
        for category, task in benchmark_tasks.items():
            try:
                self.logger.info(f"Running {category} benchmarks...")
                benchmark_results[category] = await task
                self.logger.info(f"Completed {category} benchmarks")
            except Exception as e:
                self.logger.error(f"Failed to run {category} benchmarks: {e}")
                benchmark_results[category] = {'error': str(e)}
        
        return benchmark_results

    async def _validate_benchmark_results(self, benchmark_results: Dict[str, Any]) -> List[ValidationResult]:
        """Validate benchmark results against targets and baselines"""
        self.logger.info("Validating benchmark results")
        
        validation_results = []
        
        # System benchmarks validation
        if 'system' in benchmark_results and 'error' not in benchmark_results['system']:
            system_validations = await self._validate_system_benchmarks(benchmark_results['system'])
            validation_results.extend(system_validations)
        
        # Privacy benchmarks validation
        if 'privacy' in benchmark_results and 'error' not in benchmark_results['privacy']:
            privacy_validations = await self._validate_privacy_benchmarks(benchmark_results['privacy'])
            validation_results.extend(privacy_validations)
        
        # Graph benchmarks validation
        if 'graph' in benchmark_results and 'error' not in benchmark_results['graph']:
            graph_validations = await self._validate_graph_benchmarks(benchmark_results['graph'])
            validation_results.extend(graph_validations)
        
        # Integration benchmarks validation
        if 'integration' in benchmark_results and 'error' not in benchmark_results['integration']:
            integration_validations = await self._validate_integration_benchmarks(benchmark_results['integration'])
            validation_results.extend(integration_validations)
        
        return validation_results

    async def _validate_system_benchmarks(self, system_results: Dict[str, Any]) -> List[ValidationResult]:
        """Validate system benchmark results"""
        validations = []
        
        # Validate fog coordinator improvement
        if 'monolithic_vs_microservices' in system_results:
            architecture_results = system_results['monolithic_vs_microservices']
            if 'overall_improvement' in architecture_results:
                improvement = architecture_results['overall_improvement']
                
                validations.append(ValidationResult(
                    benchmark_name='fog_coordinator_optimization',
                    category='system',
                    baseline_value=self.baseline_data.get('system', {}).get('fog_coordinator_optimization'),
                    current_value=improvement,
                    improvement_percent=improvement,
                    target_improvement=self.validation_targets['fog_coordinator_improvement'],
                    target_met=improvement >= self.validation_targets['fog_coordinator_improvement'],
                    regression_detected=False,  # Will be calculated later
                    performance_grade=self._calculate_grade(improvement, self.validation_targets['fog_coordinator_improvement']),
                    timestamp=time.time()
                ))
        
        # Validate startup performance
        if 'service_startup_performance' in system_results:
            startup_results = system_results['service_startup_performance']
            if 'parallel_startup' in startup_results:
                startup_time = startup_results['parallel_startup']['total_parallel_startup_seconds']
                
                validations.append(ValidationResult(
                    benchmark_name='system_startup_time',
                    category='system',
                    baseline_value=self.baseline_data.get('system', {}).get('system_startup_time'),
                    current_value=startup_time,
                    improvement_percent=self._calculate_time_improvement(startup_time, self.validation_targets['system_startup_time']),
                    target_improvement=self.validation_targets['system_startup_time'],
                    target_met=startup_time <= self.validation_targets['system_startup_time'],
                    regression_detected=False,
                    performance_grade=self._calculate_time_grade(startup_time, self.validation_targets['system_startup_time']),
                    timestamp=time.time()
                ))
        
        # Validate device registration performance
        if 'device_registration_flow' in system_results:
            registration_results = system_results['device_registration_flow']
            if 'average_registration_ms' in registration_results:
                registration_time = registration_results['average_registration_ms'] / 1000  # Convert to seconds
                
                validations.append(ValidationResult(
                    benchmark_name='device_registration_time',
                    category='system',
                    baseline_value=self.baseline_data.get('system', {}).get('device_registration_time'),
                    current_value=registration_time,
                    improvement_percent=self._calculate_time_improvement(registration_time, self.validation_targets['device_registration_time']),
                    target_improvement=self.validation_targets['device_registration_time'],
                    target_met=registration_time <= self.validation_targets['device_registration_time'],
                    regression_detected=False,
                    performance_grade=self._calculate_time_grade(registration_time, self.validation_targets['device_registration_time']),
                    timestamp=time.time()
                ))
        
        # Validate memory optimization
        if 'memory_optimization_validation' in system_results:
            memory_results = system_results['memory_optimization_validation']
            if 'memory_efficiency' in memory_results and 'memory_reduction_mb' in memory_results['memory_efficiency']:
                memory_reduction = memory_results['memory_efficiency']['memory_reduction_mb']
                baseline_memory = memory_results['baseline_memory']['rss_mb']
                memory_improvement = (memory_reduction / baseline_memory) * 100 if baseline_memory > 0 else 0
                
                validations.append(ValidationResult(
                    benchmark_name='memory_optimization',
                    category='system',
                    baseline_value=self.baseline_data.get('system', {}).get('memory_optimization'),
                    current_value=memory_improvement,
                    improvement_percent=memory_improvement,
                    target_improvement=self.validation_targets['memory_reduction_percent'],
                    target_met=memory_improvement >= self.validation_targets['memory_reduction_percent'],
                    regression_detected=False,
                    performance_grade=self._calculate_grade(memory_improvement, self.validation_targets['memory_reduction_percent']),
                    timestamp=time.time()
                ))
        
        return validations

    async def _validate_privacy_benchmarks(self, privacy_results: Dict[str, Any]) -> List[ValidationResult]:
        """Validate privacy benchmark results"""
        validations = []
        
        # Validate onion coordinator optimization
        if 'fog_onion_coordinator_optimization' in privacy_results:
            coordinator_results = privacy_results['fog_onion_coordinator_optimization']
            if 'average_improvement_percent' in coordinator_results:
                improvement = coordinator_results['average_improvement_percent']
                
                validations.append(ValidationResult(
                    benchmark_name='onion_coordinator_optimization',
                    category='privacy',
                    baseline_value=self.baseline_data.get('privacy', {}).get('onion_coordinator_optimization'),
                    current_value=improvement,
                    improvement_percent=improvement,
                    target_improvement=self.validation_targets['onion_coordinator_improvement'],
                    target_met=improvement >= self.validation_targets['onion_coordinator_improvement'],
                    regression_detected=False,
                    performance_grade=self._calculate_grade(improvement, self.validation_targets['onion_coordinator_improvement']),
                    timestamp=time.time()
                ))
        
        # Validate privacy task routing
        if 'privacy_task_routing' in privacy_results:
            routing_results = privacy_results['privacy_task_routing']
            for scenario_name, scenario_data in routing_results.get('task_routing_results', {}).items():
                if 'avg_routing_ms' in scenario_data:
                    routing_time = scenario_data['avg_routing_ms'] / 1000  # Convert to seconds
                    
                    validations.append(ValidationResult(
                        benchmark_name=f'privacy_task_routing_{scenario_name}',
                        category='privacy',
                        baseline_value=self.baseline_data.get('privacy', {}).get(f'privacy_task_routing_{scenario_name}'),
                        current_value=routing_time,
                        improvement_percent=self._calculate_time_improvement(routing_time, self.validation_targets['privacy_task_routing_time']),
                        target_improvement=self.validation_targets['privacy_task_routing_time'],
                        target_met=routing_time <= self.validation_targets['privacy_task_routing_time'],
                        regression_detected=False,
                        performance_grade=self._calculate_time_grade(routing_time, self.validation_targets['privacy_task_routing_time']),
                        timestamp=time.time()
                    ))
        
        # Validate circuit creation performance
        if 'circuit_creation_optimization' in privacy_results:
            circuit_results = privacy_results['circuit_creation_optimization']
            if 'optimization_impact' in circuit_results and 'improvement_percent' in circuit_results['optimization_impact']:
                improvement = circuit_results['optimization_impact']['improvement_percent']
                
                validations.append(ValidationResult(
                    benchmark_name='circuit_creation_optimization',
                    category='privacy',
                    baseline_value=self.baseline_data.get('privacy', {}).get('circuit_creation_optimization'),
                    current_value=improvement,
                    improvement_percent=improvement,
                    target_improvement=30.0,  # 30% improvement target for circuit creation
                    target_met=improvement >= 30.0,
                    regression_detected=False,
                    performance_grade=self._calculate_grade(improvement, 30.0),
                    timestamp=time.time()
                ))
        
        return validations

    async def _validate_graph_benchmarks(self, graph_results: Dict[str, Any]) -> List[ValidationResult]:
        """Validate graph benchmark results"""
        validations = []
        
        # Validate gap detection optimization
        if 'gap_detection_optimization' in graph_results:
            gap_results = graph_results['gap_detection_optimization']
            if 'optimization_impact' in gap_results and 'average_improvement_percent' in gap_results['optimization_impact']:
                improvement = gap_results['optimization_impact']['average_improvement_percent']
                
                validations.append(ValidationResult(
                    benchmark_name='graph_gap_detection_optimization',
                    category='graph',
                    baseline_value=self.baseline_data.get('graph', {}).get('graph_gap_detection_optimization'),
                    current_value=improvement,
                    improvement_percent=improvement,
                    target_improvement=self.validation_targets['graph_fixer_improvement'],
                    target_met=improvement >= self.validation_targets['graph_fixer_improvement'],
                    regression_detected=False,
                    performance_grade=self._calculate_grade(improvement, self.validation_targets['graph_fixer_improvement']),
                    timestamp=time.time()
                ))
        
        # Validate semantic similarity optimization
        if 'semantic_similarity_optimization' in graph_results:
            similarity_results = graph_results['semantic_similarity_optimization']
            if 'performance_characteristics' in similarity_results and 'average_improvement_percent' in similarity_results['performance_characteristics']:
                improvement = similarity_results['performance_characteristics']['average_improvement_percent']
                
                validations.append(ValidationResult(
                    benchmark_name='semantic_similarity_optimization',
                    category='graph',
                    baseline_value=self.baseline_data.get('graph', {}).get('semantic_similarity_optimization'),
                    current_value=improvement,
                    improvement_percent=improvement,
                    target_improvement=60.0,  # 60% improvement target for semantic similarity
                    target_met=improvement >= 60.0,
                    regression_detected=False,
                    performance_grade=self._calculate_grade(improvement, 60.0),
                    timestamp=time.time()
                ))
        
        # Validate algorithm complexity improvements
        if 'algorithm_complexity_analysis' in graph_results:
            complexity_results = graph_results['algorithm_complexity_analysis']
            if 'practical_impact' in complexity_results and 'improvement_percent_at_largest_test' in complexity_results['practical_impact']:
                improvement = complexity_results['practical_impact']['improvement_percent_at_largest_test']
                
                validations.append(ValidationResult(
                    benchmark_name='algorithm_complexity_improvement',
                    category='graph',
                    baseline_value=self.baseline_data.get('graph', {}).get('algorithm_complexity_improvement'),
                    current_value=improvement,
                    improvement_percent=improvement,
                    target_improvement=75.0,  # 75% improvement target for O(n²) → O(n log n)
                    target_met=improvement >= 75.0,
                    regression_detected=False,
                    performance_grade=self._calculate_grade(improvement, 75.0),
                    timestamp=time.time()
                ))
        
        return validations

    async def _validate_integration_benchmarks(self, integration_results: Dict[str, Any]) -> List[ValidationResult]:
        """Validate integration benchmark results"""
        validations = []
        
        # Validate cross-service communication
        if 'cross_service_communication' in integration_results:
            comm_results = integration_results['cross_service_communication']
            if 'overall_performance' in comm_results and 'average_latency_ms' in comm_results['overall_performance']:
                avg_latency = comm_results['overall_performance']['average_latency_ms']
                
                validations.append(ValidationResult(
                    benchmark_name='cross_service_communication',
                    category='integration',
                    baseline_value=self.baseline_data.get('integration', {}).get('cross_service_communication'),
                    current_value=avg_latency,
                    improvement_percent=self._calculate_latency_improvement(avg_latency, self.validation_targets['cross_service_latency_ms']),
                    target_improvement=self.validation_targets['cross_service_latency_ms'],
                    target_met=avg_latency <= self.validation_targets['cross_service_latency_ms'],
                    regression_detected=False,
                    performance_grade=self._calculate_latency_grade(avg_latency, self.validation_targets['cross_service_latency_ms']),
                    timestamp=time.time()
                ))
        
        # Validate service coordination overhead
        if 'service_coordination' in integration_results:
            coord_results = integration_results['service_coordination']
            if 'overhead_analysis' in coord_results and 'average_overhead_percent' in coord_results['overhead_analysis']:
                avg_overhead = coord_results['overhead_analysis']['average_overhead_percent']
                
                validations.append(ValidationResult(
                    benchmark_name='service_coordination_overhead',
                    category='integration',
                    baseline_value=self.baseline_data.get('integration', {}).get('service_coordination_overhead'),
                    current_value=avg_overhead,
                    improvement_percent=self._calculate_overhead_improvement(avg_overhead, self.validation_targets['coordination_overhead_percent']),
                    target_improvement=self.validation_targets['coordination_overhead_percent'],
                    target_met=avg_overhead <= self.validation_targets['coordination_overhead_percent'],
                    regression_detected=False,
                    performance_grade=self._calculate_overhead_grade(avg_overhead, self.validation_targets['coordination_overhead_percent']),
                    timestamp=time.time()
                ))
        
        # Validate end-to-end workflows
        if 'end_to_end_workflows' in integration_results:
            workflow_results = integration_results['end_to_end_workflows']
            if 'workflow_analysis' in workflow_results and 'average_execution_time_ms' in workflow_results['workflow_analysis']:
                avg_exec_time = workflow_results['workflow_analysis']['average_execution_time_ms']
                
                validations.append(ValidationResult(
                    benchmark_name='end_to_end_workflows',
                    category='integration',
                    baseline_value=self.baseline_data.get('integration', {}).get('end_to_end_workflows'),
                    current_value=avg_exec_time,
                    improvement_percent=self._calculate_latency_improvement(avg_exec_time, self.validation_targets['end_to_end_latency_ms']),
                    target_improvement=self.validation_targets['end_to_end_latency_ms'],
                    target_met=avg_exec_time <= self.validation_targets['end_to_end_latency_ms'],
                    regression_detected=False,
                    performance_grade=self._calculate_latency_grade(avg_exec_time, self.validation_targets['end_to_end_latency_ms']),
                    timestamp=time.time()
                ))
        
        return validations

    async def _detect_regressions(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Detect performance regressions compared to baseline"""
        self.logger.info("Detecting performance regressions")
        
        regressions = []
        
        for result in validation_results:
            if result.baseline_value is not None:
                # Calculate regression metrics
                regression_info = self._calculate_regression_metrics(result)
                
                if regression_info['regression_detected']:
                    result.regression_detected = True
                    regressions.append({
                        'benchmark': result.benchmark_name,
                        'category': result.category,
                        'regression_type': regression_info['regression_type'],
                        'regression_severity': regression_info['severity'],
                        'baseline_value': result.baseline_value,
                        'current_value': result.current_value,
                        'degradation_percent': regression_info['degradation_percent']
                    })
        
        regression_summary = {
            'total_regressions': len(regressions),
            'critical_regressions': len([r for r in regressions if r['regression_severity'] == 'critical']),
            'major_regressions': len([r for r in regressions if r['regression_severity'] == 'major']),
            'minor_regressions': len([r for r in regressions if r['regression_severity'] == 'minor']),
            'regressions_by_category': self._group_regressions_by_category(regressions),
            'detailed_regressions': regressions
        }
        
        return regression_summary

    def _calculate_regression_metrics(self, result: ValidationResult) -> Dict[str, Any]:
        """Calculate regression metrics for a validation result"""
        
        if result.baseline_value is None:
            return {'regression_detected': False}
        
        baseline = result.baseline_value
        current = result.current_value
        
        # Determine if this is a "lower is better" metric
        lower_is_better = result.benchmark_name in [
            'system_startup_time', 'device_registration_time', 'privacy_task_routing_time',
            'cross_service_communication', 'end_to_end_workflows', 'service_coordination_overhead'
        ]
        
        if lower_is_better:
            # For time/latency metrics, increase is bad
            if current > baseline:
                degradation = ((current - baseline) / baseline) * 100
                regression_type = 'performance_degradation'
                threshold = self.regression_thresholds['performance_degradation']
            else:
                degradation = 0
                regression_type = None
                threshold = 0
        else:
            # For improvement metrics, decrease is bad
            if current < baseline:
                degradation = ((baseline - current) / baseline) * 100
                regression_type = 'performance_degradation'
                threshold = self.regression_thresholds['performance_degradation']
            else:
                degradation = 0
                regression_type = None
                threshold = 0
        
        regression_detected = degradation > threshold
        
        # Determine severity
        if regression_detected:
            if degradation > threshold * 3:
                severity = 'critical'
            elif degradation > threshold * 2:
                severity = 'major'
            else:
                severity = 'minor'
        else:
            severity = None
        
        return {
            'regression_detected': regression_detected,
            'regression_type': regression_type,
            'degradation_percent': degradation,
            'severity': severity,
            'threshold': threshold
        }

    def _group_regressions_by_category(self, regressions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group regressions by category"""
        
        by_category = {}
        for regression in regressions:
            category = regression['category']
            if category not in by_category:
                by_category[category] = 0
            by_category[category] += 1
        
        return by_category

    def _generate_validation_summary(self, validation_results: List[ValidationResult], regression_results: Dict[str, Any]) -> ValidationSummary:
        """Generate comprehensive validation summary"""
        
        total_benchmarks = len(validation_results)
        passed_benchmarks = sum(1 for r in validation_results if r.target_met)
        failed_benchmarks = total_benchmarks - passed_benchmarks
        regressions_detected = regression_results['total_regressions']
        
        # Calculate average improvement
        improvements = [r.improvement_percent for r in validation_results if r.improvement_percent is not None]
        average_improvement = statistics.mean(improvements) if improvements else 0.0
        
        # Calculate overall grade
        overall_grade = self._calculate_overall_grade(validation_results, regression_results)
        
        # Identify critical issues
        critical_issues = self._identify_critical_issues(validation_results, regression_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validation_results, regression_results)
        
        return ValidationSummary(
            total_benchmarks=total_benchmarks,
            passed_benchmarks=passed_benchmarks,
            failed_benchmarks=failed_benchmarks,
            regressions_detected=regressions_detected,
            average_improvement=average_improvement,
            overall_grade=overall_grade,
            critical_issues=critical_issues,
            recommendations=recommendations
        )

    def _calculate_overall_grade(self, validation_results: List[ValidationResult], regression_results: Dict[str, Any]) -> str:
        """Calculate overall validation grade"""
        
        if not validation_results:
            return "F"
        
        pass_rate = sum(1 for r in validation_results if r.target_met) / len(validation_results) * 100
        critical_regressions = regression_results['critical_regressions']
        major_regressions = regression_results['major_regressions']
        
        # Penalize for critical issues
        if critical_regressions > 0:
            grade_score = min(pass_rate, 60)  # Cap at D
        elif major_regressions > 2:
            grade_score = min(pass_rate, 75)  # Cap at C
        else:
            grade_score = pass_rate
        
        # Convert to letter grade
        if grade_score >= 90:
            return "A"
        elif grade_score >= 80:
            return "B"
        elif grade_score >= 70:
            return "C"
        elif grade_score >= 60:
            return "D"
        else:
            return "F"

    def _identify_critical_issues(self, validation_results: List[ValidationResult], regression_results: Dict[str, Any]) -> List[str]:
        """Identify critical issues that need immediate attention"""
        
        critical_issues = []
        
        # Failed critical benchmarks
        critical_benchmarks = [
            'fog_coordinator_optimization', 'onion_coordinator_optimization',
            'graph_gap_detection_optimization', 'coupling_reduction_validation'
        ]
        
        for result in validation_results:
            if result.benchmark_name in critical_benchmarks and not result.target_met:
                critical_issues.append(
                    f"Critical benchmark failed: {result.benchmark_name} "
                    f"({result.improvement_percent:.1f}% vs {result.target_improvement}% target)"
                )
        
        # Critical regressions
        for regression in regression_results.get('detailed_regressions', []):
            if regression['regression_severity'] == 'critical':
                critical_issues.append(
                    f"Critical regression: {regression['benchmark']} degraded by "
                    f"{regression['degradation_percent']:.1f}%"
                )
        
        # System-wide issues
        failed_categories = {}
        for result in validation_results:
            if not result.target_met:
                category = result.category
                if category not in failed_categories:
                    failed_categories[category] = 0
                failed_categories[category] += 1
        
        for category, count in failed_categories.items():
            if count >= 3:  # 3 or more failures in a category
                critical_issues.append(
                    f"Systematic issues in {category} category: {count} benchmark failures"
                )
        
        return critical_issues[:10]  # Top 10 critical issues

    def _generate_recommendations(self, validation_results: List[ValidationResult], regression_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Performance improvement recommendations
        low_performers = [r for r in validation_results if not r.target_met and r.performance_grade in ['D', 'F']]
        if low_performers:
            recommendations.append(
                f"Prioritize optimization of {len(low_performers)} underperforming benchmarks"
            )
        
        # Regression recommendations
        if regression_results['total_regressions'] > 0:
            recommendations.append(
                f"Address {regression_results['total_regressions']} performance regressions before deployment"
            )
        
        # Category-specific recommendations
        category_issues = {}
        for result in validation_results:
            if not result.target_met:
                category = result.category
                if category not in category_issues:
                    category_issues[category] = []
                category_issues[category].append(result.benchmark_name)
        
        for category, issues in category_issues.items():
            if len(issues) >= 2:
                recommendations.append(
                    f"Focus on {category} optimization: {len(issues)} benchmarks need improvement"
                )
        
        # General recommendations
        if any(r.benchmark_name.endswith('_time') and not r.target_met for r in validation_results):
            recommendations.append("Implement performance profiling to identify latency bottlenecks")
        
        if any('memory' in r.benchmark_name and not r.target_met for r in validation_results):
            recommendations.append("Conduct memory usage analysis and optimization")
        
        recommendations.extend([
            "Establish continuous performance monitoring",
            "Set up automated performance regression detection",
            "Create performance budgets for new features"
        ])
        
        return recommendations[:8]  # Top 8 recommendations

    async def _save_validation_results(self, validation_results: List[ValidationResult], validation_summary: ValidationSummary):
        """Save validation results to files"""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detailed_results_file = self.output_dir / f"validation_results_{timestamp}.json"
        detailed_results = {
            'validation_summary': asdict(validation_summary),
            'detailed_results': [asdict(r) for r in validation_results],
            'validation_metadata': {
                'timestamp': timestamp,
                'targets': self.validation_targets,
                'regression_thresholds': self.regression_thresholds,
                'total_benchmarks': len(validation_results)
            }
        }
        
        with open(detailed_results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # Save summary report
        summary_file = self.output_dir / f"validation_summary_{timestamp}.md"
        await self._generate_summary_report(validation_summary, validation_results, summary_file)
        
        # Update baseline if this is a good run
        if validation_summary.overall_grade in ['A', 'B'] and validation_summary.regressions_detected == 0:
            await self._update_baseline(validation_results)
        
        self.logger.info(f"Validation results saved to {detailed_results_file}")
        self.logger.info(f"Summary report saved to {summary_file}")

    async def _generate_summary_report(self, summary: ValidationSummary, results: List[ValidationResult], output_file: Path):
        """Generate human-readable summary report"""
        
        with open(output_file, 'w') as f:
            f.write("# Phase 3 Performance Validation Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Overall Grade:** {summary.overall_grade}\n")
            f.write(f"- **Benchmarks Passed:** {summary.passed_benchmarks}/{summary.total_benchmarks} ({summary.passed_benchmarks/summary.total_benchmarks*100:.1f}%)\n")
            f.write(f"- **Average Improvement:** {summary.average_improvement:.1f}%\n")
            f.write(f"- **Regressions Detected:** {summary.regressions_detected}\n\n")
            
            # Critical Issues
            if summary.critical_issues:
                f.write("## Critical Issues\n\n")
                for issue in summary.critical_issues:
                    f.write(f"- {issue}\n")
                f.write("\n")
            
            # Results by Category
            f.write("## Results by Category\n\n")
            categories = {}
            for result in results:
                if result.category not in categories:
                    categories[result.category] = {'passed': 0, 'total': 0, 'results': []}
                categories[result.category]['total'] += 1
                if result.target_met:
                    categories[result.category]['passed'] += 1
                categories[result.category]['results'].append(result)
            
            for category, data in categories.items():
                f.write(f"### {category.title()}\n")
                f.write(f"- Passed: {data['passed']}/{data['total']} ({data['passed']/data['total']*100:.1f}%)\n\n")
                
                for result in data['results']:
                    status = "✅ PASS" if result.target_met else "❌ FAIL"
                    regression = " 🔴 REGRESSION" if result.regression_detected else ""
                    f.write(f"#### {result.benchmark_name} {status}{regression}\n")
                    f.write(f"- **Current Value:** {result.current_value:.3f}\n")
                    f.write(f"- **Target:** {result.target_improvement}\n")
                    if result.improvement_percent is not None:
                        f.write(f"- **Improvement:** {result.improvement_percent:.1f}%\n")
                    f.write(f"- **Grade:** {result.performance_grade}\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            for i, recommendation in enumerate(summary.recommendations, 1):
                f.write(f"{i}. {recommendation}\n")
            f.write("\n")

    async def _update_baseline(self, validation_results: List[ValidationResult]):
        """Update baseline performance data"""
        
        baseline_data = {}
        
        for result in validation_results:
            category = result.category
            if category not in baseline_data:
                baseline_data[category] = {}
            
            baseline_data[category][result.benchmark_name] = result.current_value
        
        baseline_file = self.output_dir / "baseline_performance.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        self.logger.info(f"Updated baseline data: {baseline_file}")

    # Helper methods for grade calculations
    def _calculate_grade(self, value: float, target: float) -> str:
        """Calculate grade based on improvement vs target"""
        if value >= target:
            return "A"
        elif value >= target * 0.8:
            return "B"
        elif value >= target * 0.6:
            return "C"
        elif value >= target * 0.4:
            return "D"
        else:
            return "F"

    def _calculate_time_grade(self, time_value: float, target_time: float) -> str:
        """Calculate grade for time-based metrics (lower is better)"""
        if time_value <= target_time:
            return "A"
        elif time_value <= target_time * 1.25:
            return "B"
        elif time_value <= target_time * 1.5:
            return "C"
        elif time_value <= target_time * 2.0:
            return "D"
        else:
            return "F"

    def _calculate_latency_grade(self, latency: float, target_latency: float) -> str:
        """Calculate grade for latency metrics (lower is better)"""
        return self._calculate_time_grade(latency, target_latency)

    def _calculate_overhead_grade(self, overhead: float, target_overhead: float) -> str:
        """Calculate grade for overhead metrics (lower is better)"""
        return self._calculate_time_grade(overhead, target_overhead)

    def _calculate_time_improvement(self, current_time: float, target_time: float) -> float:
        """Calculate improvement percentage for time-based metrics"""
        if current_time <= target_time:
            return ((target_time - current_time) / target_time) * 100
        else:
            return 0.0

    def _calculate_latency_improvement(self, current_latency: float, target_latency: float) -> float:
        """Calculate improvement percentage for latency metrics"""
        return self._calculate_time_improvement(current_latency, target_latency)

    def _calculate_overhead_improvement(self, current_overhead: float, target_overhead: float) -> float:
        """Calculate improvement percentage for overhead metrics"""
        return self._calculate_time_improvement(current_overhead, target_overhead)

    async def create_performance_baseline(self) -> Dict[str, Any]:
        """Create initial performance baseline"""
        self.logger.info("Creating performance baseline")
        
        # Run benchmarks to establish baseline
        benchmark_results = await self._run_all_benchmarks()
        
        # Extract key metrics for baseline
        baseline_data = self._extract_baseline_metrics(benchmark_results)
        
        # Save baseline
        baseline_file = self.output_dir / "baseline_performance.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        self.logger.info(f"Performance baseline created: {baseline_file}")
        
        return baseline_data

    def _extract_baseline_metrics(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for baseline comparison"""
        
        baseline = {}
        
        # System metrics
        if 'system' in benchmark_results:
            system_results = benchmark_results['system']
            baseline['system'] = {}
            
            if 'monolithic_vs_microservices' in system_results:
                baseline['system']['fog_coordinator_optimization'] = system_results['monolithic_vs_microservices'].get('overall_improvement', 0)
            
            if 'service_startup_performance' in system_results:
                startup_results = system_results['service_startup_performance']
                if 'parallel_startup' in startup_results:
                    baseline['system']['system_startup_time'] = startup_results['parallel_startup']['total_parallel_startup_seconds']
        
        # Privacy metrics
        if 'privacy' in benchmark_results:
            privacy_results = benchmark_results['privacy']
            baseline['privacy'] = {}
            
            if 'fog_onion_coordinator_optimization' in privacy_results:
                baseline['privacy']['onion_coordinator_optimization'] = privacy_results['fog_onion_coordinator_optimization'].get('average_improvement_percent', 0)
        
        # Graph metrics
        if 'graph' in benchmark_results:
            graph_results = benchmark_results['graph']
            baseline['graph'] = {}
            
            if 'gap_detection_optimization' in graph_results:
                gap_results = graph_results['gap_detection_optimization']
                if 'optimization_impact' in gap_results:
                    baseline['graph']['graph_gap_detection_optimization'] = gap_results['optimization_impact'].get('average_improvement_percent', 0)
        
        # Integration metrics
        if 'integration' in benchmark_results:
            integration_results = benchmark_results['integration']
            baseline['integration'] = {}
            
            if 'cross_service_communication' in integration_results:
                comm_results = integration_results['cross_service_communication']
                if 'overall_performance' in comm_results:
                    baseline['integration']['cross_service_communication'] = comm_results['overall_performance'].get('average_latency_ms', 0)
        
        return baseline