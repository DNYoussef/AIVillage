import logging
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import traceback
from pathlib import Path
import numpy as np
from rag_system.error_handling.error_handler import error_handler
from rag_system.error_handling.performance_monitor import performance_monitor

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Quality metrics for system components."""
    accuracy: float = 0.0
    reliability: float = 0.0
    performance: float = 0.0
    resource_usage: float = 0.0
    error_rate: float = 0.0
    response_time: float = 0.0
    test_coverage: float = 0.0

@dataclass
class ComponentQuality:
    """Quality tracking for a system component."""
    name: str
    metrics: QualityMetrics = field(default_factory=QualityMetrics)
    history: List[Dict[str, Any]] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

class QualityAssuranceLayer:
    """
    Quality assurance system for monitoring and maintaining system quality.
    
    Features:
    - Component quality tracking
    - Performance monitoring
    - Error detection and handling
    - Test coverage tracking
    - Quality metrics analysis
    """
    
    def __init__(self):
        self.components: Dict[str, ComponentQuality] = {}
        self.global_thresholds = {
            'accuracy': 0.95,
            'reliability': 0.99,
            'performance': 0.90,
            'resource_usage': 0.80,
            'error_rate': 0.01,
            'response_time': 1.0,  # seconds
            'test_coverage': 0.90
        }
        self.quality_history: List[Dict[str, Any]] = []
        self.test_results: Dict[str, List[Dict[str, Any]]] = {}
        self.setup_logging()

    def setup_logging(self):
        """Set up quality assurance logging."""
        qa_handler = logging.FileHandler('quality_assurance.log')
        qa_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        qa_logger = logging.getLogger('quality_assurance')
        qa_logger.addHandler(qa_handler)
        qa_logger.setLevel(logging.INFO)

    async def monitor_component(
        self,
        component_name: str,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Monitor quality metrics for a component."""
        try:
            if component_name not in self.components:
                self.components[component_name] = ComponentQuality(name=component_name)
            
            component = self.components[component_name]
            
            # Update metrics if provided
            if metrics:
                self._update_component_metrics(component, metrics)
            
            # Check thresholds
            violations = self._check_thresholds(component)
            if violations:
                await self._handle_threshold_violations(component, violations)
            
            # Record history
            self._record_quality_history(component)
            
            logger.info(f"Monitored quality metrics for {component_name}")
        except Exception as e:
            logger.error(f"Error monitoring component {component_name}: {str(e)}")
            error_handler.log_error(e, {'component': component_name})

    def _update_component_metrics(
        self,
        component: ComponentQuality,
        metrics: Dict[str, float]
    ):
        """Update component metrics."""
        for metric, value in metrics.items():
            if hasattr(component.metrics, metric):
                setattr(component.metrics, metric, value)
        component.last_updated = datetime.now()

    def _check_thresholds(
        self,
        component: ComponentQuality
    ) -> List[Dict[str, Any]]:
        """Check if metrics exceed thresholds."""
        violations = []
        
        # Check component-specific thresholds
        for metric, threshold in component.thresholds.items():
            if hasattr(component.metrics, metric):
                value = getattr(component.metrics, metric)
                if self._is_threshold_violated(metric, value, threshold):
                    violations.append({
                        'metric': metric,
                        'value': value,
                        'threshold': threshold,
                        'type': 'component'
                    })
        
        # Check global thresholds
        for metric, threshold in self.global_thresholds.items():
            if metric not in component.thresholds and hasattr(component.metrics, metric):
                value = getattr(component.metrics, metric)
                if self._is_threshold_violated(metric, value, threshold):
                    violations.append({
                        'metric': metric,
                        'value': value,
                        'threshold': threshold,
                        'type': 'global'
                    })
        
        return violations

    def _is_threshold_violated(
        self,
        metric: str,
        value: float,
        threshold: float
    ) -> bool:
        """Check if a metric violates its threshold."""
        if metric == 'error_rate':
            return value > threshold
        return value < threshold

    async def _handle_threshold_violations(
        self,
        component: ComponentQuality,
        violations: List[Dict[str, Any]]
    ):
        """Handle threshold violations."""
        for violation in violations:
            logger.warning(
                f"Quality threshold violation in {component.name}: "
                f"{violation['metric']} = {violation['value']} "
                f"(threshold: {violation['threshold']})"
            )
            
            # Record violation
            self.quality_history.append({
                'timestamp': datetime.now().isoformat(),
                'component': component.name,
                'violation': violation,
                'context': {
                    'metrics': component.metrics.__dict__,
                    'history': component.history[-10:] if component.history else []
                }
            })
            
            # Trigger alerts
            await self._trigger_quality_alert(component, violation)

    async def _trigger_quality_alert(
        self,
        component: ComponentQuality,
        violation: Dict[str, Any]
    ):
        """Trigger quality alert for violation."""
        alert = {
            'component': component.name,
            'metric': violation['metric'],
            'value': violation['value'],
            'threshold': violation['threshold'],
            'timestamp': datetime.now().isoformat(),
            'severity': 'high' if violation['type'] == 'global' else 'medium'
        }
        
        # Log alert
        logger.error(f"Quality alert: {json.dumps(alert, indent=2)}")
        
        # Store alert
        self._record_quality_history(component, alert)
        
        # Notify error handling system
        error_handler.log_error(
            Exception(f"Quality threshold violation in {component.name}"),
            {'alert': alert}
        )

    def _record_quality_history(
        self,
        component: ComponentQuality,
        event: Optional[Dict[str, Any]] = None
    ):
        """Record quality history for a component."""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'metrics': component.metrics.__dict__.copy(),
            'event': event
        }
        component.history.append(history_entry)
        
        # Limit history size
        if len(component.history) > 1000:
            component.history = component.history[-1000:]

    async def run_component_tests(
        self,
        component_name: str,
        tests: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run quality assurance tests for a component."""
        try:
            results = []
            for test in tests:
                start_time = datetime.now()
                try:
                    # Run test
                    result = await self._execute_test(test)
                    success = result.get('success', False)
                    
                    # Record metrics
                    execution_time = (datetime.now() - start_time).total_seconds()
                    performance_monitor.record_operation(
                        component_name,
                        f"test_{test['name']}",
                        execution_time
                    )
                    
                    results.append({
                        'test': test['name'],
                        'success': success,
                        'execution_time': execution_time,
                        'result': result
                    })
                    
                except Exception as e:
                    results.append({
                        'test': test['name'],
                        'success': False,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    })
            
            # Update test results
            self.test_results[component_name] = results
            
            # Update component metrics
            success_rate = sum(1 for r in results if r['success']) / len(results)
            await self.monitor_component(component_name, {
                'accuracy': success_rate,
                'test_coverage': len(results) / len(tests)
            })
            
            return {
                'component': component_name,
                'total_tests': len(tests),
                'passed_tests': sum(1 for r in results if r['success']),
                'execution_time': sum(r['execution_time'] for r in results if 'execution_time' in r),
                'results': results
            }
        except Exception as e:
            logger.error(f"Error running tests for {component_name}: {str(e)}")
            error_handler.log_error(e, {
                'component': component_name,
                'test_count': len(tests)
            })
            return {
                'component': component_name,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    async def _execute_test(self, test: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single test."""
        try:
            # Get test function
            test_func = test.get('function')
            if not test_func:
                raise ValueError(f"No test function provided for test {test['name']}")
            
            # Execute test
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            return {
                'success': True,
                'result': result
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def get_component_quality(self, component_name: str) -> Optional[ComponentQuality]:
        """Get quality metrics for a component."""
        return self.components.get(component_name)

    def get_quality_report(self) -> Dict[str, Any]:
        """Generate quality report for all components."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'global_metrics': {
                'average_accuracy': 0.0,
                'average_reliability': 0.0,
                'total_errors': 0,
                'test_coverage': 0.0
            },
            'recent_violations': self.quality_history[-10:],
            'test_summary': {}
        }
        
        # Compile component metrics
        accuracies = []
        reliabilities = []
        total_errors = 0
        test_coverage = []
        
        for name, component in self.components.items():
            metrics = component.metrics
            report['components'][name] = {
                'metrics': metrics.__dict__,
                'last_updated': component.last_updated.isoformat(),
                'recent_history': component.history[-5:]
            }
            
            accuracies.append(metrics.accuracy)
            reliabilities.append(metrics.reliability)
            total_errors += len([
                h for h in component.history
                if h.get('event', {}).get('severity') == 'high'
            ])
            test_coverage.append(metrics.test_coverage)
        
        # Calculate global metrics
        if accuracies:
            report['global_metrics']['average_accuracy'] = np.mean(accuracies)
        if reliabilities:
            report['global_metrics']['average_reliability'] = np.mean(reliabilities)
        if test_coverage:
            report['global_metrics']['test_coverage'] = np.mean(test_coverage)
        report['global_metrics']['total_errors'] = total_errors
        
        # Add test summary
        for component, results in self.test_results.items():
            passed = sum(1 for r in results if r['success'])
            report['test_summary'][component] = {
                'total_tests': len(results),
                'passed_tests': passed,
                'success_rate': passed / len(results) if results else 0
            }
        
        return report

    def set_component_threshold(
        self,
        component_name: str,
        metric: str,
        threshold: float
    ):
        """Set quality threshold for a component."""
        if component_name not in self.components:
            self.components[component_name] = ComponentQuality(name=component_name)
        self.components[component_name].thresholds[metric] = threshold

    def set_global_threshold(self, metric: str, threshold: float):
        """Set global quality threshold."""
        self.global_thresholds[metric] = threshold

    async def start_monitoring(self, interval: int = 60):
        """Start periodic quality monitoring."""
        while True:
            try:
                for component in self.components.values():
                    await self.monitor_component(component.name)
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in quality monitoring: {str(e)}")
                await asyncio.sleep(interval)

# Create singleton instance
quality_assurance = QualityAssuranceLayer()

# Start monitoring
asyncio.create_task(quality_assurance.start_monitoring())
