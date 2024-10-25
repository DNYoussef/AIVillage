import logging
import time
import psutil
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import threading
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class ComponentMetrics:
    name: str
    operation_times: Dict[str, List[float]] = field(default_factory=dict)
    error_count: int = 0
    success_count: int = 0
    last_operation_time: Optional[float] = None
    memory_usage: deque = field(default_factory=lambda: deque(maxlen=100))
    cpu_usage: deque = field(default_factory=lambda: deque(maxlen=100))

class PerformanceMonitor:
    def __init__(self):
        self.component_metrics: Dict[str, ComponentMetrics] = {}
        self.start_time = time.time()
        self._monitoring = False
        self._monitor_thread = None
        self.setup_logging()
        self.metrics_history: deque = deque(maxlen=1000)
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 80.0,
            'operation_time': 5.0  # seconds
        }

    def setup_logging(self):
        """Set up performance logging configuration."""
        performance_handler = logging.FileHandler('performance.log')
        performance_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        performance_logger = logging.getLogger('performance')
        performance_logger.addHandler(performance_handler)
        performance_logger.setLevel(logging.INFO)

    def start_monitoring(self):
        """Start performance monitoring."""
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()

    def _monitor_resources(self):
        """Monitor system resources."""
        while self._monitoring:
            try:
                current_time = time.time() - self.start_time
                process = psutil.Process()
                
                # Get CPU and memory usage
                cpu_percent = process.cpu_percent()
                memory_percent = process.memory_percent()
                
                # Check thresholds and log alerts
                if cpu_percent > self.alert_thresholds['cpu_percent']:
                    logger.warning(f"High CPU usage detected: {cpu_percent}%")
                if memory_percent > self.alert_thresholds['memory_percent']:
                    logger.warning(f"High memory usage detected: {memory_percent}%")
                
                # Store metrics for each component
                for metrics in self.component_metrics.values():
                    metrics.cpu_usage.append((current_time, cpu_percent))
                    metrics.memory_usage.append((current_time, memory_percent))
                
                # Store overall system metrics
                self.metrics_history.append({
                    'timestamp': current_time,
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'thread_count': threading.active_count(),
                    'open_files': len(process.open_files()),
                    'connections': len(process.connections())
                })
                
                time.sleep(1)  # Sample every second
            except Exception as e:
                logger.error(f"Error in resource monitoring: {str(e)}")

    def record_operation(self, component: str, operation: str, duration: float, success: bool = True):
        """Record an operation's performance metrics."""
        if component not in self.component_metrics:
            self.component_metrics[component] = ComponentMetrics(name=component)
        
        metrics = self.component_metrics[component]
        
        if operation not in metrics.operation_times:
            metrics.operation_times[operation] = []
        
        metrics.operation_times[operation].append(duration)
        metrics.last_operation_time = time.time()
        
        if success:
            metrics.success_count += 1
        else:
            metrics.error_count += 1
        
        # Log slow operations
        if duration > self.alert_thresholds['operation_time']:
            logger.warning(f"Slow operation detected - Component: {component}, Operation: {operation}, Duration: {duration}s")
        
        # Log to performance log
        logger.info(json.dumps({
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'operation': operation,
            'duration': duration,
            'success': success
        }))

    def get_component_metrics(self, component: str) -> Dict[str, Any]:
        """Get metrics for a specific component."""
        if component not in self.component_metrics:
            return {}
        
        metrics = self.component_metrics[component]
        operation_stats = {}
        
        for operation, times in metrics.operation_times.items():
            if times:
                operation_stats[operation] = {
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times),
                    'last_10_avg': sum(times[-10:]) / len(times[-10:]) if len(times) >= 10 else sum(times) / len(times)
                }
        
        return {
            'operation_stats': operation_stats,
            'success_rate': metrics.success_count / (metrics.success_count + metrics.error_count) if (metrics.success_count + metrics.error_count) > 0 else 0,
            'error_count': metrics.error_count,
            'success_count': metrics.success_count,
            'last_operation_time': metrics.last_operation_time,
            'memory_usage': {
                'current': metrics.memory_usage[-1][1] if metrics.memory_usage else 0,
                'average': sum(m[1] for m in metrics.memory_usage) / len(metrics.memory_usage) if metrics.memory_usage else 0
            },
            'cpu_usage': {
                'current': metrics.cpu_usage[-1][1] if metrics.cpu_usage else 0,
                'average': sum(c[1] for c in metrics.cpu_usage) / len(metrics.cpu_usage) if metrics.cpu_usage else 0
            }
        }

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system metrics."""
        if not self.metrics_history:
            return {}
        
        latest_metrics = self.metrics_history[-1]
        avg_metrics = {
            'cpu_percent': sum(m['cpu_percent'] for m in self.metrics_history) / len(self.metrics_history),
            'memory_percent': sum(m['memory_percent'] for m in self.metrics_history) / len(self.metrics_history),
            'thread_count': sum(m['thread_count'] for m in self.metrics_history) / len(self.metrics_history)
        }
        
        return {
            'current': latest_metrics,
            'averages': avg_metrics,
            'uptime': time.time() - self.start_time,
            'components': len(self.component_metrics),
            'total_operations': sum(
                len(times)
                for metrics in self.component_metrics.values()
                for times in metrics.operation_times.values()
            )
        }

    def set_alert_threshold(self, metric: str, value: float):
        """Set an alert threshold for a specific metric."""
        if metric in self.alert_thresholds:
            self.alert_thresholds[metric] = value
            logger.info(f"Updated alert threshold for {metric}: {value}")

    async def monitor_performance(self, interval: int = 60):
        """Periodically monitor and log performance metrics."""
        while True:
            try:
                system_metrics = self.get_system_metrics()
                component_metrics = {
                    component: self.get_component_metrics(component)
                    for component in self.component_metrics
                }
                
                # Log detailed performance metrics
                logger.info(json.dumps({
                    'timestamp': datetime.now().isoformat(),
                    'system_metrics': system_metrics,
                    'component_metrics': component_metrics
                }, indent=2))
                
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in performance monitoring: {str(e)}")
                await asyncio.sleep(interval)

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        return {
            'system_metrics': self.get_system_metrics(),
            'component_metrics': {
                component: self.get_component_metrics(component)
                for component in self.component_metrics
            },
            'alert_thresholds': dict(self.alert_thresholds),
            'monitoring_status': self._monitoring
        }

# Create singleton instance
performance_monitor = PerformanceMonitor()

# Start monitoring
performance_monitor.start_monitoring()

# Start periodic performance logging
asyncio.create_task(performance_monitor.monitor_performance())
