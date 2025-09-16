"""
Comprehensive Pipeline Health Monitor for Phase 6 Integration

This module provides real-time monitoring, health checks, and performance
tracking for the entire Phase 6 baking pipeline, ensuring 99.9% reliability.
"""

import asyncio
import logging
import time
import psutil
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import uuid
import statistics

from .data_flow_coordinator import DataFlowCoordinator
from .agent_synchronization_manager import AgentSynchronizationManager
from .error_recovery_system import ErrorRecoverySystem
from .state_manager import StateManager
from .serialization_utils import SafeJSONSerializer, SerializationConfig

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    EXCELLENT = "excellent"      # 99.9%+ uptime, no issues
    GOOD = "good"               # 99.0%+ uptime, minor issues
    WARNING = "warning"         # 95.0%+ uptime, some issues
    CRITICAL = "critical"       # <95.0% uptime, major issues
    FAILURE = "failure"         # System not operational

class MetricType(Enum):
    """Types of metrics collected"""
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    RESOURCE = "resource"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"

@dataclass
class HealthCheckResult:
    """Result of a health check"""
    check_id: str
    check_name: str
    status: HealthStatus
    score: float  # 0-100
    details: Dict[str, Any]
    timestamp: datetime
    duration_ms: float
    issues: List[str]
    recommendations: List[str]

@dataclass
class MetricPoint:
    """Single metric data point"""
    metric_name: str
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime
    component: str
    tags: Dict[str, str]

@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison"""
    metric_name: str
    baseline_value: float
    tolerance_percentage: float
    warning_threshold: float
    critical_threshold: float

class PipelineHealthMonitor:
    """
    Comprehensive health monitor for Phase 6 integration pipeline.

    Features:
    - Real-time health monitoring
    - Performance metrics collection
    - Automated health checks
    - SLA monitoring (99.9% reliability)
    - Alerting and notifications
    - Trend analysis and predictions
    - Resource usage monitoring
    - Component dependency tracking
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitor_id = str(uuid.uuid4())

        # Core components
        self.data_flow_coordinator: Optional[DataFlowCoordinator] = None
        self.agent_sync_manager: Optional[AgentSynchronizationManager] = None
        self.error_recovery_system: Optional[ErrorRecoverySystem] = None
        self.state_manager = StateManager(config.get('state_config', {}))

        # Health check registry
        self.health_checks: Dict[str, Callable] = {}
        self.health_check_results: Dict[str, HealthCheckResult] = {}
        self.health_check_history: deque = deque(maxlen=config.get('health_history_size', 1000))

        # Metrics collection
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=config.get('metric_history_size', 10000)))
        self.metric_baselines: Dict[str, PerformanceBaseline] = {}

        # System monitoring
        self.system_metrics = {
            'uptime_start': time.time(),
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'current_availability': 100.0,
            'sla_compliance': 100.0
        }

        # Configuration
        self.monitoring_interval = config.get('monitoring_interval_seconds', 30)
        self.health_check_interval = config.get('health_check_interval_seconds', 60)
        self.metric_collection_interval = config.get('metric_collection_interval_seconds', 10)
        self.sla_target = config.get('sla_target_percentage', 99.9)
        self.alert_thresholds = config.get('alert_thresholds', {
            'availability': 99.0,
            'error_rate': 1.0,
            'response_time': 1000.0
        })

        # Background tasks
        self.running = False
        self.background_tasks = []

        # Serialization
        self.serializer = SafeJSONSerializer(SerializationConfig())

        # Lock for thread safety
        self.metrics_lock = threading.RLock()

        logger.info(f"PipelineHealthMonitor initialized with ID: {self.monitor_id}")

    async def start(self):
        """Start the pipeline health monitor"""
        if self.running:
            logger.warning("PipelineHealthMonitor already running")
            return

        self.running = True
        logger.info("Starting PipelineHealthMonitor...")

        # Initialize default health checks
        self._register_default_health_checks()

        # Initialize performance baselines
        self._initialize_performance_baselines()

        # Start background monitoring tasks
        self.background_tasks = [
            asyncio.create_task(self._continuous_health_monitoring()),
            asyncio.create_task(self._metrics_collector()),
            asyncio.create_task(self._sla_monitor()),
            asyncio.create_task(self._trend_analyzer()),
            asyncio.create_task(self._alert_manager())
        ]

        logger.info("PipelineHealthMonitor started successfully")

    async def stop(self):
        """Stop the pipeline health monitor"""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping PipelineHealthMonitor...")

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()

        logger.info("PipelineHealthMonitor stopped")

    def register_components(self,
                          data_flow_coordinator: Optional[DataFlowCoordinator] = None,
                          agent_sync_manager: Optional[AgentSynchronizationManager] = None,
                          error_recovery_system: Optional[ErrorRecoverySystem] = None):
        """Register core components for monitoring"""
        if data_flow_coordinator:
            self.data_flow_coordinator = data_flow_coordinator
            logger.info("Registered DataFlowCoordinator for monitoring")

        if agent_sync_manager:
            self.agent_sync_manager = agent_sync_manager
            logger.info("Registered AgentSynchronizationManager for monitoring")

        if error_recovery_system:
            self.error_recovery_system = error_recovery_system
            logger.info("Registered ErrorRecoverySystem for monitoring")

    def register_health_check(self, check_name: str, check_function: Callable):
        """Register a custom health check"""
        self.health_checks[check_name] = check_function
        logger.info(f"Registered health check: {check_name}")

    async def run_health_check(self, check_name: str) -> HealthCheckResult:
        """Run a specific health check"""
        if check_name not in self.health_checks:
            raise ValueError(f"Health check '{check_name}' not found")

        start_time = time.time()
        check_id = str(uuid.uuid4())

        try:
            # Run the health check
            result = await self.health_checks[check_name]()

            duration_ms = (time.time() - start_time) * 1000

            # Create health check result
            health_result = HealthCheckResult(
                check_id=check_id,
                check_name=check_name,
                status=result.get('status', HealthStatus.WARNING),
                score=result.get('score', 0.0),
                details=result.get('details', {}),
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                issues=result.get('issues', []),
                recommendations=result.get('recommendations', [])
            )

            # Store result
            self.health_check_results[check_name] = health_result
            self.health_check_history.append(health_result)

            return health_result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            # Create error result
            error_result = HealthCheckResult(
                check_id=check_id,
                check_name=check_name,
                status=HealthStatus.FAILURE,
                score=0.0,
                details={'error': str(e)},
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                issues=[f"Health check failed: {e}"],
                recommendations=["Investigate health check implementation"]
            )

            self.health_check_results[check_name] = error_result
            self.health_check_history.append(error_result)

            logger.error(f"Health check '{check_name}' failed: {e}")
            return error_result

    async def run_all_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}

        # Run health checks in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(asyncio.run, self.run_health_check(check_name)): check_name
                for check_name in self.health_checks
            }

            for future in futures:
                check_name = futures[future]
                try:
                    results[check_name] = future.result()
                except Exception as e:
                    logger.error(f"Failed to run health check '{check_name}': {e}")

        return results

    def record_metric(self, metric_name: str, value: float,
                     metric_type: MetricType = MetricType.PERFORMANCE,
                     unit: str = "", component: str = "system",
                     tags: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        with self.metrics_lock:
            metric_point = MetricPoint(
                metric_name=metric_name,
                metric_type=metric_type,
                value=value,
                unit=unit,
                timestamp=datetime.now(),
                component=component,
                tags=tags or {}
            )

            self.metrics[metric_name].append(metric_point)

    def get_metric_statistics(self, metric_name: str,
                            time_window_minutes: int = 60) -> Dict[str, float]:
        """Get statistics for a metric within a time window"""
        with self.metrics_lock:
            if metric_name not in self.metrics:
                return {}

            # Filter metrics within time window
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            recent_metrics = [
                m for m in self.metrics[metric_name]
                if m.timestamp >= cutoff_time
            ]

            if not recent_metrics:
                return {}

            values = [m.value for m in recent_metrics]

            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
                'percentile_95': self._percentile(values, 95),
                'percentile_99': self._percentile(values, 99)
            }

    def get_overall_health_status(self) -> Dict[str, Any]:
        """Get overall health status of the pipeline"""
        current_time = datetime.now()

        # Collect component health
        component_health = {}
        overall_scores = []

        for check_name, result in self.health_check_results.items():
            component_health[check_name] = {
                'status': result.status.value,
                'score': result.score,
                'last_check': result.timestamp.isoformat(),
                'issues': len(result.issues)
            }
            overall_scores.append(result.score)

        # Calculate overall health score
        overall_score = statistics.mean(overall_scores) if overall_scores else 0.0

        # Determine overall status
        if overall_score >= 95.0:
            overall_status = HealthStatus.EXCELLENT
        elif overall_score >= 90.0:
            overall_status = HealthStatus.GOOD
        elif overall_score >= 80.0:
            overall_status = HealthStatus.WARNING
        elif overall_score >= 60.0:
            overall_status = HealthStatus.CRITICAL
        else:
            overall_status = HealthStatus.FAILURE

        # Calculate uptime
        uptime_seconds = time.time() - self.system_metrics['uptime_start']

        return {
            'monitor_id': self.monitor_id,
            'timestamp': current_time.isoformat(),
            'overall_status': overall_status.value,
            'overall_score': overall_score,
            'uptime_seconds': uptime_seconds,
            'sla_compliance': self.system_metrics['sla_compliance'],
            'current_availability': self.system_metrics['current_availability'],
            'total_requests': self.system_metrics['total_requests'],
            'success_rate': (self.system_metrics['successful_requests'] /
                           max(1, self.system_metrics['total_requests'])) * 100,
            'average_response_time': self.system_metrics['average_response_time'],
            'component_health': component_health,
            'active_health_checks': len(self.health_checks),
            'metrics_collected': sum(len(metrics) for metrics in self.metrics.values())
        }

    def generate_health_report(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        current_time = datetime.now()
        window_start = current_time - timedelta(hours=time_window_hours)

        # Filter health checks in time window
        window_checks = [
            check for check in self.health_check_history
            if check.timestamp >= window_start
        ]

        # Analyze health trends
        status_counts = defaultdict(int)
        component_performance = defaultdict(list)

        for check in window_checks:
            status_counts[check.status.value] += 1
            component_performance[check.check_name].append(check.score)

        # Calculate component averages
        component_averages = {
            component: statistics.mean(scores)
            for component, scores in component_performance.items()
        }

        # System performance metrics
        performance_metrics = {}
        for metric_name in ['response_time', 'throughput', 'error_rate', 'resource_usage']:
            stats = self.get_metric_statistics(metric_name, time_window_hours * 60)
            if stats:
                performance_metrics[metric_name] = stats

        return {
            'report_timestamp': current_time.isoformat(),
            'time_window_hours': time_window_hours,
            'total_health_checks': len(window_checks),
            'status_distribution': dict(status_counts),
            'component_performance': component_averages,
            'performance_metrics': performance_metrics,
            'sla_compliance_summary': self._calculate_sla_compliance(window_start),
            'availability_summary': self._calculate_availability_summary(window_start),
            'recommendations': self._generate_health_recommendations()
        }

    # Private methods

    def _register_default_health_checks(self):
        """Register default health checks"""
        self.register_health_check('system_resources', self._check_system_resources)
        self.register_health_check('data_flow_health', self._check_data_flow_health)
        self.register_health_check('agent_synchronization', self._check_agent_synchronization)
        self.register_health_check('error_recovery', self._check_error_recovery)
        self.register_health_check('state_management', self._check_state_management)
        self.register_health_check('pipeline_throughput', self._check_pipeline_throughput)
        self.register_health_check('serialization_health', self._check_serialization_health)

    def _initialize_performance_baselines(self):
        """Initialize performance baselines for monitoring"""
        self.metric_baselines = {
            'response_time': PerformanceBaseline(
                metric_name='response_time',
                baseline_value=100.0,  # 100ms baseline
                tolerance_percentage=20.0,
                warning_threshold=150.0,  # 150ms warning
                critical_threshold=500.0   # 500ms critical
            ),
            'throughput': PerformanceBaseline(
                metric_name='throughput',
                baseline_value=100.0,  # 100 requests/sec baseline
                tolerance_percentage=30.0,
                warning_threshold=70.0,   # 70 req/sec warning
                critical_threshold=50.0   # 50 req/sec critical
            ),
            'error_rate': PerformanceBaseline(
                metric_name='error_rate',
                baseline_value=0.1,    # 0.1% baseline
                tolerance_percentage=50.0,
                warning_threshold=1.0,   # 1% warning
                critical_threshold=5.0   # 5% critical
            )
        }

    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource health"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage('.')
            disk_percent = (disk.used / disk.total) * 100

            # Calculate score based on resource usage
            score = 100.0
            issues = []

            if cpu_percent > 90:
                score -= 30
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > 80:
                score -= 15
                issues.append(f"Elevated CPU usage: {cpu_percent:.1f}%")

            if memory_percent > 90:
                score -= 30
                issues.append(f"High memory usage: {memory_percent:.1f}%")
            elif memory_percent > 80:
                score -= 15
                issues.append(f"Elevated memory usage: {memory_percent:.1f}%")

            if disk_percent > 95:
                score -= 20
                issues.append(f"High disk usage: {disk_percent:.1f}%")
            elif disk_percent > 85:
                score -= 10
                issues.append(f"Elevated disk usage: {disk_percent:.1f}%")

            # Determine status
            if score >= 90:
                status = HealthStatus.EXCELLENT
            elif score >= 80:
                status = HealthStatus.GOOD
            elif score >= 70:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.CRITICAL

            # Record metrics
            self.record_metric('cpu_usage', cpu_percent, MetricType.RESOURCE, '%')
            self.record_metric('memory_usage', memory_percent, MetricType.RESOURCE, '%')
            self.record_metric('disk_usage', disk_percent, MetricType.RESOURCE, '%')

            return {
                'status': status,
                'score': max(0.0, score),
                'details': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'disk_percent': disk_percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_free_gb': disk.free / (1024**3)
                },
                'issues': issues,
                'recommendations': self._get_resource_recommendations(cpu_percent, memory_percent, disk_percent)
            }

        except Exception as e:
            return {
                'status': HealthStatus.FAILURE,
                'score': 0.0,
                'details': {'error': str(e)},
                'issues': [f"Resource check failed: {e}"],
                'recommendations': ["Check system monitoring tools"]
            }

    async def _check_data_flow_health(self) -> Dict[str, Any]:
        """Check data flow coordinator health"""
        try:
            if not self.data_flow_coordinator:
                return {
                    'status': HealthStatus.WARNING,
                    'score': 50.0,
                    'details': {'message': 'Data flow coordinator not registered'},
                    'issues': ['Data flow coordinator not available'],
                    'recommendations': ['Register data flow coordinator']
                }

            # Get data flow status
            status = self.data_flow_coordinator.get_system_status()

            score = 100.0
            issues = []

            # Check component health
            healthy_components = status.get('healthy_components', 0)
            total_components = status.get('total_components', 1)
            health_ratio = healthy_components / total_components

            if health_ratio < 0.8:
                score -= 40
                issues.append(f"Low component health ratio: {health_ratio:.2%}")
            elif health_ratio < 0.9:
                score -= 20
                issues.append(f"Some components unhealthy: {health_ratio:.2%}")

            # Check message queue
            queue_size = status.get('queue_size', 0)
            if queue_size > 1000:
                score -= 30
                issues.append(f"Large message queue: {queue_size}")
            elif queue_size > 500:
                score -= 15
                issues.append(f"Growing message queue: {queue_size}")

            # Determine status
            if score >= 90:
                health_status = HealthStatus.EXCELLENT
            elif score >= 80:
                health_status = HealthStatus.GOOD
            elif score >= 70:
                health_status = HealthStatus.WARNING
            else:
                health_status = HealthStatus.CRITICAL

            return {
                'status': health_status,
                'score': max(0.0, score),
                'details': status,
                'issues': issues,
                'recommendations': ['Monitor component registration', 'Check message processing rates']
            }

        except Exception as e:
            return {
                'status': HealthStatus.FAILURE,
                'score': 0.0,
                'details': {'error': str(e)},
                'issues': [f"Data flow health check failed: {e}"],
                'recommendations': ["Check data flow coordinator status"]
            }

    async def _check_agent_synchronization(self) -> Dict[str, Any]:
        """Check agent synchronization health"""
        try:
            if not self.agent_sync_manager:
                return {
                    'status': HealthStatus.WARNING,
                    'score': 50.0,
                    'details': {'message': 'Agent sync manager not registered'},
                    'issues': ['Agent synchronization manager not available'],
                    'recommendations': ['Register agent synchronization manager']
                }

            # Get agent sync status
            status = self.agent_sync_manager.get_system_status()

            score = 100.0
            issues = []

            # Check agent health
            healthy_agents = status.get('healthy_agents', 0)
            total_agents = status.get('total_agents', 1)

            if total_agents == 0:
                score -= 50
                issues.append("No agents registered")
            else:
                health_ratio = healthy_agents / total_agents
                if health_ratio < 0.8:
                    score -= 40
                    issues.append(f"Low agent health ratio: {health_ratio:.2%}")
                elif health_ratio < 0.9:
                    score -= 20
                    issues.append(f"Some agents unhealthy: {health_ratio:.2%}")

            # Check task processing
            pending_tasks = status.get('pending_tasks', 0)
            if pending_tasks > 100:
                score -= 30
                issues.append(f"Large task backlog: {pending_tasks}")
            elif pending_tasks > 50:
                score -= 15
                issues.append(f"Growing task backlog: {pending_tasks}")

            # Determine status
            if score >= 90:
                health_status = HealthStatus.EXCELLENT
            elif score >= 80:
                health_status = HealthStatus.GOOD
            elif score >= 70:
                health_status = HealthStatus.WARNING
            else:
                health_status = HealthStatus.CRITICAL

            return {
                'status': health_status,
                'score': max(0.0, score),
                'details': status,
                'issues': issues,
                'recommendations': ['Monitor agent registration', 'Check task processing rates']
            }

        except Exception as e:
            return {
                'status': HealthStatus.FAILURE,
                'score': 0.0,
                'details': {'error': str(e)},
                'issues': [f"Agent sync health check failed: {e}"],
                'recommendations': ["Check agent synchronization manager status"]
            }

    async def _check_error_recovery(self) -> Dict[str, Any]:
        """Check error recovery system health"""
        try:
            if not self.error_recovery_system:
                return {
                    'status': HealthStatus.WARNING,
                    'score': 50.0,
                    'details': {'message': 'Error recovery system not registered'},
                    'issues': ['Error recovery system not available'],
                    'recommendations': ['Register error recovery system']
                }

            # Get error recovery status
            status = self.error_recovery_system.get_system_health_status()

            score = 100.0
            issues = []

            # Check active errors
            active_errors = status.get('active_errors', 0)
            if active_errors > 10:
                score -= 40
                issues.append(f"High number of active errors: {active_errors}")
            elif active_errors > 5:
                score -= 20
                issues.append(f"Multiple active errors: {active_errors}")

            # Check critical errors
            critical_errors_24h = status.get('critical_errors_24h', 0)
            if critical_errors_24h > 5:
                score -= 50
                issues.append(f"Multiple critical errors in 24h: {critical_errors_24h}")
            elif critical_errors_24h > 0:
                score -= 25
                issues.append(f"Critical errors in 24h: {critical_errors_24h}")

            # Check recovery success rate
            metrics = status.get('metrics', {})
            recovery_rate = metrics.get('recovery_success_rate', 0.0)
            if recovery_rate < 0.8:
                score -= 30
                issues.append(f"Low recovery success rate: {recovery_rate:.2%}")
            elif recovery_rate < 0.9:
                score -= 15
                issues.append(f"Recovery success rate below target: {recovery_rate:.2%}")

            # Determine status
            if score >= 90:
                health_status = HealthStatus.EXCELLENT
            elif score >= 80:
                health_status = HealthStatus.GOOD
            elif score >= 70:
                health_status = HealthStatus.WARNING
            else:
                health_status = HealthStatus.CRITICAL

            return {
                'status': health_status,
                'score': max(0.0, score),
                'details': status,
                'issues': issues,
                'recommendations': ['Monitor error patterns', 'Review recovery strategies']
            }

        except Exception as e:
            return {
                'status': HealthStatus.FAILURE,
                'score': 0.0,
                'details': {'error': str(e)},
                'issues': [f"Error recovery health check failed: {e}"],
                'recommendations': ["Check error recovery system status"]
            }

    async def _check_state_management(self) -> Dict[str, Any]:
        """Check state management health"""
        try:
            # Validate state consistency
            validation_results = self.state_manager.validate_state_consistency()

            score = 100.0
            issues = []

            if not validation_results['consistent']:
                score -= 50
                issues.extend(validation_results['issues'])

            # Check warnings
            if validation_results['warnings']:
                score -= len(validation_results['warnings']) * 5
                issues.extend(validation_results['warnings'])

            # Check orphaned states
            if validation_results['orphaned_states']:
                score -= len(validation_results['orphaned_states']) * 10
                issues.append(f"Orphaned states: {len(validation_results['orphaned_states'])}")

            # Determine status
            if score >= 90:
                status = HealthStatus.EXCELLENT
            elif score >= 80:
                status = HealthStatus.GOOD
            elif score >= 70:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.CRITICAL

            return {
                'status': status,
                'score': max(0.0, score),
                'details': validation_results,
                'issues': issues,
                'recommendations': ['Clean up orphaned states', 'Validate state dependencies']
            }

        except Exception as e:
            return {
                'status': HealthStatus.FAILURE,
                'score': 0.0,
                'details': {'error': str(e)},
                'issues': [f"State management check failed: {e}"],
                'recommendations': ["Check state manager configuration"]
            }

    async def _check_pipeline_throughput(self) -> Dict[str, Any]:
        """Check pipeline throughput health"""
        try:
            # Get throughput metrics
            throughput_stats = self.get_metric_statistics('throughput', 60)

            score = 100.0
            issues = []

            if throughput_stats:
                current_throughput = throughput_stats.get('mean', 0)
                baseline = self.metric_baselines.get('throughput')

                if baseline and current_throughput < baseline.critical_threshold:
                    score -= 50
                    issues.append(f"Throughput below critical threshold: {current_throughput}")
                elif baseline and current_throughput < baseline.warning_threshold:
                    score -= 25
                    issues.append(f"Throughput below warning threshold: {current_throughput}")
            else:
                score -= 30
                issues.append("No throughput metrics available")

            # Determine status
            if score >= 90:
                status = HealthStatus.EXCELLENT
            elif score >= 80:
                status = HealthStatus.GOOD
            elif score >= 70:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.CRITICAL

            return {
                'status': status,
                'score': max(0.0, score),
                'details': throughput_stats,
                'issues': issues,
                'recommendations': ['Monitor throughput trends', 'Check for bottlenecks']
            }

        except Exception as e:
            return {
                'status': HealthStatus.FAILURE,
                'score': 0.0,
                'details': {'error': str(e)},
                'issues': [f"Throughput check failed: {e}"],
                'recommendations': ["Check throughput monitoring"]
            }

    async def _check_serialization_health(self) -> Dict[str, Any]:
        """Check serialization system health"""
        try:
            # Test serialization with various data types
            test_data = {
                'datetime': datetime.now(),
                'integer': 42,
                'float': 3.14159,
                'string': 'test_string',
                'list': [1, 2, 3],
                'dict': {'key': 'value'}
            }

            score = 100.0
            issues = []

            try:
                # Test JSON serialization
                json_str = self.serializer.serialize(test_data)
                deserialized = self.serializer.deserialize(json_str)

                if len(json_str) == 0:
                    score -= 50
                    issues.append("Serialization produced empty result")

            except Exception as e:
                score -= 70
                issues.append(f"Serialization test failed: {e}")

            # Determine status
            if score >= 90:
                status = HealthStatus.EXCELLENT
            elif score >= 80:
                status = HealthStatus.GOOD
            elif score >= 70:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.CRITICAL

            return {
                'status': status,
                'score': max(0.0, score),
                'details': {'test_data_size': len(str(test_data))},
                'issues': issues,
                'recommendations': ['Monitor serialization performance', 'Check data types']
            }

        except Exception as e:
            return {
                'status': HealthStatus.FAILURE,
                'score': 0.0,
                'details': {'error': str(e)},
                'issues': [f"Serialization health check failed: {e}"],
                'recommendations': ["Check serialization configuration"]
            }

    def _get_resource_recommendations(self, cpu: float, memory: float, disk: float) -> List[str]:
        """Get recommendations based on resource usage"""
        recommendations = []

        if cpu > 80:
            recommendations.append("Consider CPU optimization or scaling")
        if memory > 80:
            recommendations.append("Monitor memory usage and consider garbage collection")
        if disk > 85:
            recommendations.append("Clean up temporary files and consider disk expansion")

        return recommendations

    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)

        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

    def _calculate_sla_compliance(self, start_time: datetime) -> Dict[str, Any]:
        """Calculate SLA compliance within time window"""
        # This would calculate actual SLA compliance
        # For now, return placeholder data
        return {
            'target_percentage': self.sla_target,
            'actual_percentage': self.system_metrics['sla_compliance'],
            'compliant': self.system_metrics['sla_compliance'] >= self.sla_target
        }

    def _calculate_availability_summary(self, start_time: datetime) -> Dict[str, Any]:
        """Calculate availability summary within time window"""
        uptime_seconds = time.time() - self.system_metrics['uptime_start']
        total_seconds = (datetime.now() - start_time).total_seconds()

        return {
            'uptime_seconds': uptime_seconds,
            'total_seconds': total_seconds,
            'availability_percentage': (uptime_seconds / total_seconds) * 100 if total_seconds > 0 else 100.0
        }

    def _generate_health_recommendations(self) -> List[str]:
        """Generate health recommendations based on current status"""
        recommendations = []

        # Check recent health check results
        recent_failures = [
            check for check in self.health_check_results.values()
            if check.status in [HealthStatus.CRITICAL, HealthStatus.FAILURE]
        ]

        if recent_failures:
            recommendations.append("Address critical health check failures")

        # Check metrics trends
        for metric_name, baseline in self.metric_baselines.items():
            stats = self.get_metric_statistics(metric_name, 60)
            if stats and stats.get('mean', 0) > baseline.critical_threshold:
                recommendations.append(f"Investigate {metric_name} performance issues")

        if not recommendations:
            recommendations.append("System is operating within normal parameters")

        return recommendations

    # Background monitoring tasks

    async def _continuous_health_monitoring(self):
        """Continuous health monitoring task"""
        while self.running:
            try:
                await self.run_all_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Error in continuous health monitoring: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _metrics_collector(self):
        """Continuous metrics collection task"""
        while self.running:
            try:
                # Collect system metrics
                current_time = time.time()

                # Update request metrics (placeholder - would be updated by actual requests)
                self.record_metric('response_time', self.system_metrics['average_response_time'],
                                 MetricType.LATENCY, 'ms')

                # Record throughput
                uptime = current_time - self.system_metrics['uptime_start']
                if uptime > 0:
                    throughput = self.system_metrics['total_requests'] / uptime
                    self.record_metric('throughput', throughput, MetricType.THROUGHPUT, 'req/sec')

                # Record error rate
                if self.system_metrics['total_requests'] > 0:
                    error_rate = (self.system_metrics['failed_requests'] /
                                self.system_metrics['total_requests']) * 100
                    self.record_metric('error_rate', error_rate, MetricType.ERROR_RATE, '%')

                await asyncio.sleep(self.metric_collection_interval)

            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(self.metric_collection_interval)

    async def _sla_monitor(self):
        """SLA monitoring task"""
        while self.running:
            try:
                # Calculate current availability
                uptime = time.time() - self.system_metrics['uptime_start']

                # Simple availability calculation (would be more complex in practice)
                if self.system_metrics['total_requests'] > 0:
                    success_rate = (self.system_metrics['successful_requests'] /
                                  self.system_metrics['total_requests'])
                    self.system_metrics['current_availability'] = success_rate * 100

                    # Update SLA compliance
                    if success_rate >= (self.sla_target / 100):
                        self.system_metrics['sla_compliance'] = 100.0
                    else:
                        self.system_metrics['sla_compliance'] = success_rate * 100

                await asyncio.sleep(60)  # Check SLA every minute

            except Exception as e:
                logger.error(f"Error in SLA monitor: {e}")
                await asyncio.sleep(60)

    async def _trend_analyzer(self):
        """Trend analysis task"""
        while self.running:
            try:
                # Analyze metric trends
                for metric_name in self.metrics:
                    stats = self.get_metric_statistics(metric_name, 60)
                    if stats and stats['count'] > 10:
                        # Simple trend detection (could be more sophisticated)
                        recent_mean = stats['mean']
                        baseline = self.metric_baselines.get(metric_name)

                        if baseline:
                            deviation = abs(recent_mean - baseline.baseline_value) / baseline.baseline_value
                            if deviation > baseline.tolerance_percentage / 100:
                                logger.warning(f"Metric {metric_name} deviating from baseline: {deviation:.2%}")

                await asyncio.sleep(300)  # Analyze trends every 5 minutes

            except Exception as e:
                logger.error(f"Error in trend analyzer: {e}")
                await asyncio.sleep(300)

    async def _alert_manager(self):
        """Alert management task"""
        while self.running:
            try:
                # Check for alert conditions
                overall_status = self.get_overall_health_status()

                # Check availability threshold
                if overall_status['current_availability'] < self.alert_thresholds['availability']:
                    logger.warning(f"ALERT: Availability below threshold: {overall_status['current_availability']:.2f}%")

                # Check error rate threshold
                error_rate_stats = self.get_metric_statistics('error_rate', 60)
                if error_rate_stats and error_rate_stats.get('mean', 0) > self.alert_thresholds['error_rate']:
                    logger.warning(f"ALERT: Error rate above threshold: {error_rate_stats['mean']:.2f}%")

                # Check response time threshold
                response_time_stats = self.get_metric_statistics('response_time', 60)
                if (response_time_stats and
                    response_time_stats.get('mean', 0) > self.alert_thresholds['response_time']):
                    logger.warning(f"ALERT: Response time above threshold: {response_time_stats['mean']:.2f}ms")

                await asyncio.sleep(60)  # Check alerts every minute

            except Exception as e:
                logger.error(f"Error in alert manager: {e}")
                await asyncio.sleep(60)

# Factory function
def create_pipeline_health_monitor(config: Dict[str, Any]) -> PipelineHealthMonitor:
    """Factory function to create pipeline health monitor"""
    return PipelineHealthMonitor(config)

# Testing utilities
async def test_pipeline_health_monitoring():
    """Test pipeline health monitoring functionality"""
    config = {
        'monitoring_interval_seconds': 30,
        'health_check_interval_seconds': 60,
        'sla_target_percentage': 99.9
    }

    monitor = PipelineHealthMonitor(config)

    try:
        await monitor.start()

        # Run health checks
        health_results = await monitor.run_all_health_checks()
        print(f"Health check results: {len(health_results)} checks completed")

        # Record some test metrics
        monitor.record_metric('test_metric', 42.0, MetricType.PERFORMANCE, 'units')

        # Get overall status
        status = monitor.get_overall_health_status()
        print(f"Overall health status: {status}")

        # Generate health report
        report = monitor.generate_health_report()
        print(f"Health report generated with {len(report)} sections")

        await asyncio.sleep(2)  # Let monitoring run

    finally:
        await monitor.stop()

if __name__ == "__main__":
    asyncio.run(test_pipeline_health_monitoring())