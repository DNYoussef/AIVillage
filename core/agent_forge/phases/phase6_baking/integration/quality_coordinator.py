"""
Quality Gate Coordinator for Phase 6 Integration

This module coordinates all quality gates across Phase 6 baking pipeline,
ensuring unified validation, consistent quality standards, and comprehensive
quality assurance throughout the integration process.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

logger = logging.getLogger(__name__)

class QualityLevel(Enum):
    """Quality level enumeration"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class GateStatus(Enum):
    """Quality gate status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"

@dataclass
class QualityMetric:
    """Quality metric definition"""
    name: str
    description: str
    threshold: float
    operator: str  # ">", ">=", "<", "<=", "==", "!="
    weight: float
    level: QualityLevel
    unit: str

@dataclass
class QualityGateResult:
    """Result of a quality gate execution"""
    gate_id: str
    gate_name: str
    status: GateStatus
    score: float
    threshold: float
    passed: bool
    metrics: Dict[str, float]
    issues: List[str]
    warnings: List[str]
    execution_time_ms: float
    timestamp: datetime

@dataclass
class QualityAssessment:
    """Overall quality assessment"""
    overall_status: GateStatus
    overall_score: float
    total_gates: int
    passed_gates: int
    failed_gates: int
    warning_gates: int
    critical_issues: List[str]
    gate_results: List[QualityGateResult]
    quality_trends: Dict[str, List[float]]
    recommendations: List[str]

class QualityCoordinator:
    """Coordinator for quality gates across Phase 6 integration"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quality_gates = {}
        self.quality_metrics = {}
        self.execution_history = []

        # Initialize quality standards
        self._initialize_quality_standards()
        self._load_quality_configuration()

    def _initialize_quality_standards(self):
        """Initialize default quality standards and metrics"""
        # Core performance metrics
        self.quality_metrics.update({
            'model_accuracy': QualityMetric(
                name='model_accuracy',
                description='Model prediction accuracy',
                threshold=0.95,
                operator='>=',
                weight=30.0,
                level=QualityLevel.CRITICAL,
                unit='percentage'
            ),
            'inference_latency': QualityMetric(
                name='inference_latency',
                description='Model inference latency',
                threshold=50.0,
                operator='<=',
                weight=25.0,
                level=QualityLevel.CRITICAL,
                unit='milliseconds'
            ),
            'model_size': QualityMetric(
                name='model_size',
                description='Optimized model size',
                threshold=100.0,
                operator='<=',
                weight=15.0,
                level=QualityLevel.HIGH,
                unit='megabytes'
            ),
            'memory_usage': QualityMetric(
                name='memory_usage',
                description='Runtime memory usage',
                threshold=1000.0,
                operator='<=',
                weight=15.0,
                level=QualityLevel.HIGH,
                unit='megabytes'
            ),
            'throughput': QualityMetric(
                name='throughput',
                description='Processing throughput',
                threshold=30.0,
                operator='>=',
                weight=10.0,
                level=QualityLevel.MEDIUM,
                unit='fps'
            ),
            'optimization_ratio': QualityMetric(
                name='optimization_ratio',
                description='Size reduction ratio',
                threshold=2.0,
                operator='>=',
                weight=5.0,
                level=QualityLevel.LOW,
                unit='ratio'
            )
        })

        # Safety and compliance metrics
        self.quality_metrics.update({
            'deterministic_score': QualityMetric(
                name='deterministic_score',
                description='Deterministic behavior score',
                threshold=0.98,
                operator='>=',
                weight=20.0,
                level=QualityLevel.CRITICAL,
                unit='score'
            ),
            'safety_compliance': QualityMetric(
                name='safety_compliance',
                description='Safety compliance score',
                threshold=0.95,
                operator='>=',
                weight=25.0,
                level=QualityLevel.CRITICAL,
                unit='score'
            ),
            'robustness_score': QualityMetric(
                name='robustness_score',
                description='Model robustness under adversarial inputs',
                threshold=0.90,
                operator='>=',
                weight=15.0,
                level=QualityLevel.HIGH,
                unit='score'
            )
        })

    def _load_quality_configuration(self):
        """Load quality configuration from files"""
        try:
            # Load thresholds
            thresholds_file = Path('config/quality_thresholds.json')
            if thresholds_file.exists():
                with open(thresholds_file, 'r') as f:
                    thresholds = json.load(f)

                # Update metric thresholds
                for metric_name, threshold_value in thresholds.items():
                    if metric_name in self.quality_metrics:
                        self.quality_metrics[metric_name].threshold = threshold_value

            # Load gate definitions
            gates_file = Path('config/quality_gates.json')
            if gates_file.exists():
                with open(gates_file, 'r') as f:
                    gate_definitions = json.load(f)

                self.quality_gates.update(gate_definitions)

        except Exception as e:
            logger.warning(f"Could not load quality configuration: {e}")

    def register_quality_gate(self, gate_id: str, gate_config: Dict[str, Any]) -> bool:
        """Register a new quality gate"""
        try:
            required_fields = ['name', 'description', 'metrics', 'weight']
            for field in required_fields:
                if field not in gate_config:
                    logger.error(f"Missing required field '{field}' in gate configuration")
                    return False

            self.quality_gates[gate_id] = gate_config
            logger.info(f"Registered quality gate: {gate_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to register quality gate {gate_id}: {e}")
            return False

    def execute_quality_gate(self, gate_id: str, model_data: Dict[str, Any]) -> QualityGateResult:
        """Execute a single quality gate"""
        start_time = datetime.now()
        gate_config = self.quality_gates.get(gate_id, {})
        gate_name = gate_config.get('name', gate_id)

        try:
            # Extract required metrics
            required_metrics = gate_config.get('metrics', [])
            measured_metrics = {}
            issues = []
            warnings = []

            # Measure each required metric
            for metric_name in required_metrics:
                if metric_name not in self.quality_metrics:
                    issues.append(f"Unknown metric: {metric_name}")
                    continue

                metric_definition = self.quality_metrics[metric_name]
                measured_value = self._measure_metric(metric_name, model_data)

                if measured_value is None:
                    issues.append(f"Could not measure metric: {metric_name}")
                    continue

                measured_metrics[metric_name] = measured_value

                # Check threshold
                threshold_passed = self._check_threshold(
                    measured_value,
                    metric_definition.threshold,
                    metric_definition.operator
                )

                if not threshold_passed:
                    if metric_definition.level == QualityLevel.CRITICAL:
                        issues.append(f"{metric_name}: {measured_value} {metric_definition.operator} {metric_definition.threshold} (CRITICAL)")
                    elif metric_definition.level == QualityLevel.HIGH:
                        issues.append(f"{metric_name}: {measured_value} {metric_definition.operator} {metric_definition.threshold} (HIGH)")
                    else:
                        warnings.append(f"{metric_name}: {measured_value} {metric_definition.operator} {metric_definition.threshold}")

            # Calculate gate score
            gate_score = self._calculate_gate_score(measured_metrics, required_metrics)

            # Determine gate threshold
            gate_threshold = gate_config.get('threshold', 70.0)

            # Determine status
            if issues:
                status = GateStatus.FAILED
                passed = False
            elif warnings and gate_score < gate_threshold:
                status = GateStatus.WARNING
                passed = False
            elif gate_score >= gate_threshold:
                status = GateStatus.PASSED
                passed = True
            else:
                status = GateStatus.FAILED
                passed = False

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            result = QualityGateResult(
                gate_id=gate_id,
                gate_name=gate_name,
                status=status,
                score=gate_score,
                threshold=gate_threshold,
                passed=passed,
                metrics=measured_metrics,
                issues=issues,
                warnings=warnings,
                execution_time_ms=execution_time,
                timestamp=datetime.now()
            )

            self.execution_history.append(result)
            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Quality gate {gate_id} execution failed: {e}")

            return QualityGateResult(
                gate_id=gate_id,
                gate_name=gate_name,
                status=GateStatus.FAILED,
                score=0.0,
                threshold=gate_config.get('threshold', 70.0),
                passed=False,
                metrics={},
                issues=[f"Execution failed: {e}"],
                warnings=[],
                execution_time_ms=execution_time,
                timestamp=datetime.now()
            )

    def execute_all_quality_gates(self, model_data: Dict[str, Any],
                                parallel: bool = True) -> QualityAssessment:
        """Execute all registered quality gates"""
        start_time = datetime.now()
        gate_results = []

        try:
            if parallel and len(self.quality_gates) > 1:
                # Execute gates in parallel
                with ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_gate = {
                        executor.submit(self.execute_quality_gate, gate_id, model_data): gate_id
                        for gate_id in self.quality_gates.keys()
                    }

                    for future in as_completed(future_to_gate):
                        try:
                            result = future.result()
                            gate_results.append(result)
                        except Exception as e:
                            gate_id = future_to_gate[future]
                            logger.error(f"Parallel execution failed for gate {gate_id}: {e}")
            else:
                # Execute gates sequentially
                for gate_id in self.quality_gates.keys():
                    result = self.execute_quality_gate(gate_id, model_data)
                    gate_results.append(result)

            # Analyze results
            assessment = self._analyze_quality_results(gate_results)

            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Quality assessment completed in {execution_time:.0f}ms - Status: {assessment.overall_status.value}")

            return assessment

        except Exception as e:
            logger.error(f"Quality gate execution failed: {e}")
            return QualityAssessment(
                overall_status=GateStatus.FAILED,
                overall_score=0.0,
                total_gates=len(self.quality_gates),
                passed_gates=0,
                failed_gates=len(self.quality_gates),
                warning_gates=0,
                critical_issues=[f"Assessment failed: {e}"],
                gate_results=[],
                quality_trends={},
                recommendations=["Fix quality assessment system"]
            )

    def _measure_metric(self, metric_name: str, model_data: Dict[str, Any]) -> Optional[float]:
        """Measure a specific quality metric"""
        try:
            # Extract metric from model data based on metric name
            if metric_name == 'model_accuracy':
                return model_data.get('accuracy', model_data.get('test_accuracy', None))

            elif metric_name == 'inference_latency':
                return model_data.get('inference_time_ms', model_data.get('latency_ms', None))

            elif metric_name == 'model_size':
                return model_data.get('model_size_mb', model_data.get('size_mb', None))

            elif metric_name == 'memory_usage':
                return model_data.get('memory_usage_mb', model_data.get('peak_memory_mb', None))

            elif metric_name == 'throughput':
                return model_data.get('throughput_fps', model_data.get('fps', None))

            elif metric_name == 'optimization_ratio':
                original_size = model_data.get('original_size_mb', 1.0)
                optimized_size = model_data.get('model_size_mb', original_size)
                return original_size / optimized_size if optimized_size > 0 else 1.0

            elif metric_name == 'deterministic_score':
                return model_data.get('deterministic_score', model_data.get('consistency_score', None))

            elif metric_name == 'safety_compliance':
                return model_data.get('safety_score', model_data.get('compliance_score', None))

            elif metric_name == 'robustness_score':
                return model_data.get('robustness_score', model_data.get('adversarial_score', None))

            else:
                # Try direct lookup
                return model_data.get(metric_name, None)

        except Exception as e:
            logger.error(f"Failed to measure metric {metric_name}: {e}")
            return None

    def _check_threshold(self, value: float, threshold: float, operator: str) -> bool:
        """Check if value meets threshold criteria"""
        try:
            if operator == '>':
                return value > threshold
            elif operator == '>=':
                return value >= threshold
            elif operator == '<':
                return value < threshold
            elif operator == '<=':
                return value <= threshold
            elif operator == '==':
                return abs(value - threshold) < 1e-6
            elif operator == '!=':
                return abs(value - threshold) >= 1e-6
            else:
                logger.warning(f"Unknown operator: {operator}")
                return False

        except Exception as e:
            logger.error(f"Threshold check failed: {e}")
            return False

    def _calculate_gate_score(self, measured_metrics: Dict[str, float],
                            required_metrics: List[str]) -> float:
        """Calculate overall score for a quality gate"""
        try:
            if not measured_metrics or not required_metrics:
                return 0.0

            total_weighted_score = 0.0
            total_weight = 0.0

            for metric_name in required_metrics:
                if metric_name not in measured_metrics:
                    continue

                metric_definition = self.quality_metrics.get(metric_name)
                if not metric_definition:
                    continue

                measured_value = measured_metrics[metric_name]
                threshold = metric_definition.threshold
                operator = metric_definition.operator
                weight = metric_definition.weight

                # Calculate normalized score (0-100)
                if self._check_threshold(measured_value, threshold, operator):
                    # Passed threshold - calculate bonus score
                    if operator in ['>=', '>']:
                        bonus = min((measured_value / threshold - 1) * 10, 20)
                        metric_score = 100 + bonus
                    elif operator in ['<=', '<']:
                        bonus = min((threshold / measured_value - 1) * 10, 20)
                        metric_score = 100 + bonus
                    else:
                        metric_score = 100
                else:
                    # Failed threshold - calculate penalty
                    if operator in ['>=', '>']:
                        penalty = (1 - measured_value / threshold) * 100
                        metric_score = max(0, 100 - penalty)
                    elif operator in ['<=', '<']:
                        penalty = (measured_value / threshold - 1) * 100
                        metric_score = max(0, 100 - penalty)
                    else:
                        metric_score = 0

                total_weighted_score += metric_score * weight
                total_weight += weight

            return total_weighted_score / total_weight if total_weight > 0 else 0.0

        except Exception as e:
            logger.error(f"Gate score calculation failed: {e}")
            return 0.0

    def _analyze_quality_results(self, gate_results: List[QualityGateResult]) -> QualityAssessment:
        """Analyze quality gate results and generate assessment"""
        try:
            total_gates = len(gate_results)
            passed_gates = sum(1 for r in gate_results if r.status == GateStatus.PASSED)
            failed_gates = sum(1 for r in gate_results if r.status == GateStatus.FAILED)
            warning_gates = sum(1 for r in gate_results if r.status == GateStatus.WARNING)

            # Collect critical issues
            critical_issues = []
            for result in gate_results:
                if result.status == GateStatus.FAILED:
                    for issue in result.issues:
                        if 'CRITICAL' in issue:
                            critical_issues.append(f"{result.gate_name}: {issue}")

            # Calculate overall score
            if gate_results:
                overall_score = sum(r.score for r in gate_results) / len(gate_results)
            else:
                overall_score = 0.0

            # Determine overall status
            if failed_gates > 0:
                overall_status = GateStatus.FAILED
            elif warning_gates > 0:
                overall_status = GateStatus.WARNING
            elif passed_gates == total_gates:
                overall_status = GateStatus.PASSED
            else:
                overall_status = GateStatus.FAILED

            # Generate quality trends
            quality_trends = self._calculate_quality_trends()

            # Generate recommendations
            recommendations = self._generate_quality_recommendations(gate_results, critical_issues)

            return QualityAssessment(
                overall_status=overall_status,
                overall_score=overall_score,
                total_gates=total_gates,
                passed_gates=passed_gates,
                failed_gates=failed_gates,
                warning_gates=warning_gates,
                critical_issues=critical_issues,
                gate_results=gate_results,
                quality_trends=quality_trends,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return QualityAssessment(
                overall_status=GateStatus.FAILED,
                overall_score=0.0,
                total_gates=len(gate_results),
                passed_gates=0,
                failed_gates=len(gate_results),
                warning_gates=0,
                critical_issues=[f"Analysis failed: {e}"],
                gate_results=gate_results,
                quality_trends={},
                recommendations=["Fix quality analysis system"]
            )

    def _calculate_quality_trends(self) -> Dict[str, List[float]]:
        """Calculate quality trends from execution history"""
        trends = {}

        try:
            # Group results by gate
            gate_histories = {}
            for result in self.execution_history[-50:]:  # Last 50 executions
                if result.gate_id not in gate_histories:
                    gate_histories[result.gate_id] = []
                gate_histories[result.gate_id].append(result.score)

            # Calculate trends
            for gate_id, scores in gate_histories.items():
                if len(scores) >= 3:
                    # Simple linear trend
                    x = np.arange(len(scores))
                    y = np.array(scores)
                    trend = np.polyfit(x, y, 1)[0]  # Slope
                    trends[gate_id] = scores[-10:]  # Last 10 scores

        except Exception as e:
            logger.error(f"Trend calculation failed: {e}")

        return trends

    def _generate_quality_recommendations(self, gate_results: List[QualityGateResult],
                                        critical_issues: List[str]) -> List[str]:
        """Generate recommendations based on quality results"""
        recommendations = []

        try:
            # Critical issue recommendations
            if critical_issues:
                recommendations.append("Address critical quality issues immediately before deployment")

            # Failed gate recommendations
            failed_gates = [r for r in gate_results if r.status == GateStatus.FAILED]
            if failed_gates:
                if len(failed_gates) > len(gate_results) // 2:
                    recommendations.append("Multiple quality gates failing - review overall model quality")
                else:
                    for gate in failed_gates:
                        recommendations.append(f"Fix issues in {gate.gate_name} gate")

            # Performance recommendations
            performance_issues = []
            for result in gate_results:
                if 'latency' in result.gate_name.lower() and not result.passed:
                    performance_issues.append("latency")
                if 'memory' in result.gate_name.lower() and not result.passed:
                    performance_issues.append("memory")
                if 'size' in result.gate_name.lower() and not result.passed:
                    performance_issues.append("size")

            if performance_issues:
                recommendations.append(f"Optimize model for: {', '.join(set(performance_issues))}")

            # Accuracy recommendations
            accuracy_issues = [r for r in gate_results if 'accuracy' in r.gate_name.lower() and not r.passed]
            if accuracy_issues:
                recommendations.append("Improve model accuracy through additional training or better data")

            # Safety recommendations
            safety_issues = [r for r in gate_results if 'safety' in r.gate_name.lower() and not r.passed]
            if safety_issues:
                recommendations.append("Address safety compliance requirements for ADAS deployment")

            # General recommendations
            if not recommendations:
                if all(r.passed for r in gate_results):
                    recommendations.append("All quality gates passed - model ready for deployment")
                else:
                    recommendations.append("Review and address remaining quality concerns")

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            recommendations.append("Review quality assessment system")

        return recommendations

    def get_quality_dashboard_data(self) -> Dict[str, Any]:
        """Get data for quality dashboard"""
        try:
            recent_results = self.execution_history[-20:] if self.execution_history else []

            # Calculate pass rates
            gate_pass_rates = {}
            for gate_id in self.quality_gates.keys():
                gate_results = [r for r in recent_results if r.gate_id == gate_id]
                if gate_results:
                    passed = sum(1 for r in gate_results if r.passed)
                    gate_pass_rates[gate_id] = passed / len(gate_results)
                else:
                    gate_pass_rates[gate_id] = 0.0

            # Calculate metric trends
            metric_trends = {}
            for metric_name in self.quality_metrics.keys():
                values = []
                for result in recent_results:
                    if metric_name in result.metrics:
                        values.append(result.metrics[metric_name])
                metric_trends[metric_name] = values[-10:] if values else []

            return {
                'total_gates': len(self.quality_gates),
                'total_metrics': len(self.quality_metrics),
                'recent_executions': len(recent_results),
                'gate_pass_rates': gate_pass_rates,
                'metric_trends': metric_trends,
                'average_execution_time': np.mean([r.execution_time_ms for r in recent_results]) if recent_results else 0,
                'quality_score_trend': [r.score for r in recent_results[-10:]],
                'last_execution': recent_results[-1].timestamp.isoformat() if recent_results else None
            }

        except Exception as e:
            logger.error(f"Dashboard data generation failed: {e}")
            return {}

    def export_quality_report(self, assessment: QualityAssessment,
                            output_path: str) -> bool:
        """Export detailed quality report"""
        try:
            report_data = {
                'assessment_summary': {
                    'overall_status': assessment.overall_status.value,
                    'overall_score': assessment.overall_score,
                    'total_gates': assessment.total_gates,
                    'passed_gates': assessment.passed_gates,
                    'failed_gates': assessment.failed_gates,
                    'warning_gates': assessment.warning_gates
                },
                'critical_issues': assessment.critical_issues,
                'recommendations': assessment.recommendations,
                'gate_results': [
                    {
                        'gate_id': r.gate_id,
                        'gate_name': r.gate_name,
                        'status': r.status.value,
                        'score': r.score,
                        'threshold': r.threshold,
                        'passed': r.passed,
                        'metrics': r.metrics,
                        'issues': r.issues,
                        'warnings': r.warnings,
                        'execution_time_ms': r.execution_time_ms,
                        'timestamp': r.timestamp.isoformat()
                    }
                    for r in assessment.gate_results
                ],
                'quality_trends': assessment.quality_trends,
                'export_timestamp': datetime.now().isoformat()
            }

            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)

            logger.info(f"Quality report exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export quality report: {e}")
            return False

def create_quality_coordinator(config: Dict[str, Any]) -> QualityCoordinator:
    """Factory function to create quality coordinator"""
    return QualityCoordinator(config)

# Testing utilities
def test_quality_coordination():
    """Test quality coordination functionality"""
    config = {
        'parallel_execution': True,
        'default_threshold': 75.0
    }

    coordinator = QualityCoordinator(config)

    # Register test gates
    coordinator.register_quality_gate('performance_gate', {
        'name': 'Performance Gate',
        'description': 'Model performance validation',
        'metrics': ['model_accuracy', 'inference_latency'],
        'weight': 1.0,
        'threshold': 80.0
    })

    coordinator.register_quality_gate('optimization_gate', {
        'name': 'Optimization Gate',
        'description': 'Model optimization validation',
        'metrics': ['model_size', 'optimization_ratio'],
        'weight': 0.8,
        'threshold': 70.0
    })

    # Test model data
    test_model_data = {
        'accuracy': 0.96,
        'inference_time_ms': 45.0,
        'model_size_mb': 85.0,
        'original_size_mb': 200.0,
        'memory_usage_mb': 800.0,
        'throughput_fps': 35.0
    }

    # Execute quality assessment
    assessment = coordinator.execute_all_quality_gates(test_model_data)
    print(f"Quality assessment: {assessment.overall_status.value} - Score: {assessment.overall_score:.1f}")

    # Get dashboard data
    dashboard_data = coordinator.get_quality_dashboard_data()
    print(f"Dashboard data: {dashboard_data}")

    return assessment

if __name__ == "__main__":
    test_quality_coordination()