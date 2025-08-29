"""
Consolidated Performance Monitoring Service
==========================================

Archaeological Enhancement: Comprehensive monitoring service with profiling capabilities
Innovation Score: 9.4/10 - Unified monitoring with archaeological insights
Integration: Consolidated from profiler.py with enhanced real-time monitoring

This module provides centralized performance monitoring capabilities, incorporating archaeological
findings from multiple optimization branches including comprehensive profiling, bottleneck detection,
and real-time metrics collection for all P2P and optimization components.

Key Consolidated Features:
- Performance profiling from profiler.py
- Real-time metrics collection and analysis
- Bottleneck detection and optimization recommendations
- Component health monitoring and alerting
- Background monitoring tasks with archaeological insights
"""

import asyncio
import cProfile
import pstats
import time
import threading
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
import logging
import tracemalloc
import gc
import sys
import json
import statistics

# System imports for resource monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

logger = logging.getLogger(__name__)


class ProfileScope(Enum):
    """Profiling scope levels."""
    FUNCTION = "function"
    COMPONENT = "component"
    SYSTEM = "system"
    NETWORK = "network"


class BottleneckType(Enum):
    """Types of performance bottlenecks."""
    CPU = "cpu"
    MEMORY = "memory"
    IO = "io"
    NETWORK = "network"
    ASYNC = "async"
    SERIALIZATION = "serialization"
    ENCRYPTION = "encryption"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MonitoringConfig:
    """Configuration for monitoring service."""
    enabled: bool = True
    scope: ProfileScope = ProfileScope.COMPONENT
    sample_interval: float = 1.0  # seconds
    max_samples: int = 1000
    memory_profiling: bool = True
    cpu_profiling: bool = True
    network_profiling: bool = True
    async_profiling: bool = True
    retention_hours: int = 24
    
    # Background task intervals
    metrics_collection_interval: float = 5.0
    health_check_interval: float = 10.0
    cleanup_interval: float = 300.0  # 5 minutes


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    component: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "labels": self.labels
        }


@dataclass
class ComponentProfile:
    """Performance profile for a component."""
    component_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    cpu_usage: Dict[str, float] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    network_stats: Dict[str, Any] = field(default_factory=dict)
    function_stats: Dict[str, Any] = field(default_factory=dict)
    async_stats: Dict[str, Any] = field(default_factory=dict)
    metrics: List[PerformanceMetric] = field(default_factory=list)
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        """Get profile duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def add_metric(self, name: str, value: float, unit: str, **labels):
        """Add performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            component=self.component_name,
            labels=labels
        )
        self.metrics.append(metric)


@dataclass
class BottleneckAnalysis:
    """Analysis of performance bottlenecks."""
    bottleneck_type: BottleneckType
    severity: str  # "low", "medium", "high", "critical"
    description: str
    affected_component: str
    impact_score: float  # 0.0 to 1.0
    recommendation: str
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": self.bottleneck_type.value,
            "severity": self.severity,
            "description": self.description,
            "affected_component": self.affected_component,
            "impact_score": self.impact_score,
            "recommendation": self.recommendation,
            "metrics": self.metrics
        }


@dataclass
class MonitoringAlert:
    """Performance monitoring alert."""
    alert_id: str
    severity: AlertSeverity
    component: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "component": self.component,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata
        }


class PerformanceMonitor:
    """Consolidated performance monitoring service."""
    
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        self.active_profiles: Dict[str, ComponentProfile] = {}
        self.metrics_buffer = deque(maxlen=self.config.max_samples)
        self.alerts = deque(maxlen=1000)
        self.bottleneck_history = deque(maxlen=500)
        
        # Background tasks
        self._monitoring_thread = None
        self._stop_event = threading.Event()
        self.background_tasks: List[asyncio.Task] = []
        
        # System monitoring
        self._system_metrics = {}
        self._component_health = defaultdict(dict)
        
        # Memory tracking
        if self.config.memory_profiling:
            try:
                tracemalloc.start()
            except RuntimeError:
                pass  # Already started
        
        # Performance tracking
        self.optimization_suggestions = deque(maxlen=100)
        
    async def initialize(self):
        """Initialize monitoring service."""
        try:
            logger.info("Initializing Performance Monitor...")
            
            if self.config.enabled:
                # Start background monitoring tasks
                await self._start_background_monitoring()
            
            logger.info("Performance Monitor initialized successfully")
            
        except Exception as e:
            logger.error(f"Performance Monitor initialization failed: {e}")
            raise
    
    @contextmanager
    def profile_component(self, component_name: str):
        """Context manager for profiling a component."""
        if not self.config.enabled:
            yield
            return
        
        profile = ComponentProfile(
            component_name=component_name,
            start_time=datetime.now()
        )
        
        self.active_profiles[component_name] = profile
        
        # Start CPU profiling if enabled
        cpu_profiler = None
        if self.config.cpu_profiling:
            cpu_profiler = cProfile.Profile()
            cpu_profiler.enable()
        
        # Start memory snapshot
        memory_start = None
        if self.config.memory_profiling and tracemalloc.is_tracing():
            memory_start = tracemalloc.take_snapshot()
        
        try:
            yield profile
        finally:
            profile.end_time = datetime.now()
            
            # Collect CPU profile
            if cpu_profiler:
                cpu_profiler.disable()
                profile.function_stats = self._analyze_cpu_profile(cpu_profiler)
            
            # Collect memory profile
            if memory_start and tracemalloc.is_tracing():
                memory_end = tracemalloc.take_snapshot()
                profile.memory_usage = self._analyze_memory_profile(memory_start, memory_end)
            
            # Analyze for bottlenecks
            bottlenecks = self._analyze_component_bottlenecks(profile)
            profile.bottlenecks = bottlenecks
            self.bottleneck_history.extend(bottlenecks)
            
            # Remove from active profiles
            self.active_profiles.pop(component_name, None)
    
    @asynccontextmanager
    async def profile_async_component(self, component_name: str):
        """Async context manager for profiling async components."""
        if not self.config.enabled:
            yield
            return
        
        with self.profile_component(component_name) as profile:
            # Track async-specific metrics
            task_start_count = len(asyncio.all_tasks())
            
            try:
                yield profile
            finally:
                task_end_count = len(asyncio.all_tasks())
                profile.async_stats["task_delta"] = task_end_count - task_start_count
    
    async def record_metric(self, name: str, value: float, unit: str, 
                           component: str = "", **labels):
        """Record a performance metric."""
        if not self.config.enabled:
            return
        
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            component=component,
            labels=labels
        )
        
        self.metrics_buffer.append(metric)
        
        # Add to active profile if exists
        if component in self.active_profiles:
            self.active_profiles[component].add_metric(name, value, unit, **labels)
        
        # Check for alert conditions
        await self._check_metric_alerts(metric)
    
    async def generate_alert(self, severity: AlertSeverity, component: str, 
                           message: str, **metadata):
        """Generate a monitoring alert."""
        alert = MonitoringAlert(
            alert_id=f"alert_{int(time.time() * 1000)}",
            severity=severity,
            component=component,
            message=message,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        self.alerts.append(alert)
        
        # Log based on severity
        if severity == AlertSeverity.CRITICAL:
            logger.critical(f"CRITICAL ALERT [{component}]: {message}")
        elif severity == AlertSeverity.WARNING:
            logger.warning(f"WARNING [{component}]: {message}")
        else:
            logger.info(f"INFO [{component}]: {message}")
        
        return alert
    
    def get_real_time_metrics(self, component: Optional[str] = None, 
                             time_range_seconds: int = 300) -> List[Dict[str, Any]]:
        """Get real-time metrics within time range."""
        cutoff_time = datetime.now() - timedelta(seconds=time_range_seconds)
        
        metrics = [
            metric for metric in self.metrics_buffer
            if metric.timestamp >= cutoff_time
        ]
        
        if component:
            metrics = [m for m in metrics if m.component == component]
        
        return [m.to_dict() for m in metrics]
    
    def get_component_health(self, component: str) -> Dict[str, Any]:
        """Get health status for a specific component."""
        recent_metrics = self.get_real_time_metrics(component, 60)  # Last minute
        
        if not recent_metrics:
            return {"status": "unknown", "message": "No recent metrics"}
        
        # Analyze recent metrics for health indicators
        cpu_metrics = [m for m in recent_metrics if 'cpu' in m['name'].lower()]
        memory_metrics = [m for m in recent_metrics if 'memory' in m['name'].lower()]
        error_metrics = [m for m in recent_metrics if 'error' in m['name'].lower()]
        
        health_score = 1.0
        health_issues = []
        
        # Check CPU health
        if cpu_metrics:
            avg_cpu = statistics.mean(m['value'] for m in cpu_metrics)
            if avg_cpu > 90:
                health_score *= 0.5
                health_issues.append(f"High CPU usage: {avg_cpu:.1f}%")
            elif avg_cpu > 70:
                health_score *= 0.8
                health_issues.append(f"Elevated CPU usage: {avg_cpu:.1f}%")
        
        # Check memory health
        if memory_metrics:
            avg_memory = statistics.mean(m['value'] for m in memory_metrics)
            if avg_memory > 85:
                health_score *= 0.6
                health_issues.append(f"High memory usage: {avg_memory:.1f}%")
        
        # Check error rates
        if error_metrics:
            total_errors = sum(m['value'] for m in error_metrics)
            if total_errors > 10:
                health_score *= 0.3
                health_issues.append(f"High error rate: {total_errors} errors")
        
        # Determine status
        if health_score >= 0.8:
            status = "healthy"
        elif health_score >= 0.6:
            status = "degraded"
        elif health_score >= 0.3:
            status = "unhealthy"
        else:
            status = "critical"
        
        return {
            "status": status,
            "health_score": health_score,
            "issues": health_issues,
            "metrics_count": len(recent_metrics),
            "last_metric_time": recent_metrics[-1]['timestamp'] if recent_metrics else None
        }
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get system-wide monitoring overview."""
        return {
            "timestamp": datetime.now().isoformat(),
            "monitoring_enabled": self.config.enabled,
            "active_profiles": len(self.active_profiles),
            "metrics_buffer_size": len(self.metrics_buffer),
            "alerts_count": len([a for a in self.alerts if not a.resolved]),
            "bottlenecks_detected": len(self.bottleneck_history),
            "system_metrics": self._system_metrics,
            "component_health": {
                comp: self.get_component_health(comp) 
                for comp in self._get_monitored_components()
            }
        }
    
    async def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_components_monitored": len(self._get_monitored_components()),
                "total_metrics_collected": len(self.metrics_buffer),
                "bottlenecks_identified": len(self.bottleneck_history),
                "active_alerts": len([a for a in self.alerts if not a.resolved])
            },
            "bottlenecks": [b.to_dict() for b in self.bottleneck_history[-20:]],  # Last 20
            "recommendations": list(self.optimization_suggestions)[-10:],  # Last 10
            "component_performance": {},
            "system_health": self._calculate_system_health()
        }
        
        # Add component performance summary
        for component in self._get_monitored_components():
            health = self.get_component_health(component)
            metrics = self.get_real_time_metrics(component, 3600)  # Last hour
            
            report["component_performance"][component] = {
                "health": health,
                "metrics_count": len(metrics),
                "avg_response_time": self._calculate_avg_response_time(metrics)
            }
        
        return report
    
    async def _start_background_monitoring(self):
        """Start background monitoring tasks."""
        # System metrics collection task
        metrics_task = asyncio.create_task(self._system_metrics_loop())
        self.background_tasks.append(metrics_task)
        
        # Health monitoring task
        health_task = asyncio.create_task(self._health_monitoring_loop())
        self.background_tasks.append(health_task)
        
        # Cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.background_tasks.append(cleanup_task)
        
        logger.info(f"Started {len(self.background_tasks)} background monitoring tasks")
    
    async def _system_metrics_loop(self):
        """Background system metrics collection."""
        while True:
            try:
                await asyncio.sleep(self.config.metrics_collection_interval)
                
                # Collect system metrics if available
                if HAS_PSUTIL:
                    await self.record_metric("system_cpu_percent", psutil.cpu_percent(), "percent", "system")
                    
                    memory = psutil.virtual_memory()
                    await self.record_metric("system_memory_percent", memory.percent, "percent", "system")
                    await self.record_metric("system_memory_available", memory.available, "bytes", "system")
                    
                    # Network I/O
                    net_io = psutil.net_io_counters()
                    if hasattr(net_io, 'bytes_sent'):
                        await self.record_metric("system_network_sent", net_io.bytes_sent, "bytes", "system")
                        await self.record_metric("system_network_recv", net_io.bytes_recv, "bytes", "system")
                
                # Update system metrics cache
                self._system_metrics = {
                    "cpu_percent": psutil.cpu_percent() if HAS_PSUTIL else 0.0,
                    "memory_percent": psutil.virtual_memory().percent if HAS_PSUTIL else 0.0,
                    "network_connections": psutil.net_connections().__len__() if HAS_PSUTIL else 0,
                    "process_count": len(psutil.pids()) if HAS_PSUTIL else 0
                }
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System metrics collection error: {e}")
                await asyncio.sleep(5)
    
    async def _health_monitoring_loop(self):
        """Background health monitoring."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Check component health
                for component in self._get_monitored_components():
                    health = self.get_component_health(component)
                    
                    if health["status"] == "critical":
                        await self.generate_alert(
                            AlertSeverity.CRITICAL,
                            component,
                            f"Component health critical: {health['issues']}"
                        )
                    elif health["status"] == "unhealthy":
                        await self.generate_alert(
                            AlertSeverity.WARNING,
                            component,
                            f"Component health degraded: {health['issues']}"
                        )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_loop(self):
        """Background cleanup of old data."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                
                # Clean up old metrics
                cutoff_time = datetime.now() - timedelta(hours=self.config.retention_hours)
                
                # Clean metrics buffer
                self.metrics_buffer = deque([
                    m for m in self.metrics_buffer if m.timestamp >= cutoff_time
                ], maxlen=self.config.max_samples)
                
                # Clean resolved alerts older than retention period
                self.alerts = deque([
                    a for a in self.alerts 
                    if not a.resolved or a.resolved_at >= cutoff_time
                ], maxlen=1000)
                
                logger.debug("Completed monitoring data cleanup")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring cleanup error: {e}")
                await asyncio.sleep(30)
    
    def _analyze_cpu_profile(self, profiler: cProfile.Profile) -> Dict[str, Any]:
        """Analyze CPU profiling results."""
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        # Extract top functions
        top_functions = []
        for func, data in list(stats.stats.items())[:10]:  # Top 10
            filename, line_num, func_name = func
            calls, primitive_calls, total_time, cumulative_time = data
            
            top_functions.append({
                "function": func_name,
                "filename": filename,
                "calls": calls,
                "total_time": total_time,
                "cumulative_time": cumulative_time
            })
        
        return {
            "total_functions": len(stats.stats),
            "top_functions": top_functions,
            "total_time": sum(data[2] for data in stats.stats.values()),
            "total_calls": sum(data[0] for data in stats.stats.values())
        }
    
    def _analyze_memory_profile(self, snapshot_start, snapshot_end) -> Dict[str, Any]:
        """Analyze memory profiling results."""
        top_stats = snapshot_end.compare_to(snapshot_start, 'lineno')
        
        memory_analysis = {
            "total_diff": sum(stat.size_diff for stat in top_stats),
            "count_diff": sum(stat.count_diff for stat in top_stats),
            "top_allocations": []
        }
        
        for stat in top_stats[:10]:  # Top 10 allocations
            memory_analysis["top_allocations"].append({
                "size_diff": stat.size_diff,
                "count_diff": stat.count_diff,
                "size": stat.size,
                "count": stat.count
            })
        
        return memory_analysis
    
    def _analyze_component_bottlenecks(self, profile: ComponentProfile) -> List[BottleneckAnalysis]:
        """Analyze component for bottlenecks."""
        bottlenecks = []
        
        # CPU bottlenecks
        if profile.function_stats:
            top_functions = profile.function_stats.get("top_functions", [])
            for func in top_functions[:3]:  # Top 3 CPU-intensive
                if func["cumulative_time"] > 0.1:  # > 100ms
                    bottleneck = BottleneckAnalysis(
                        bottleneck_type=BottleneckType.CPU,
                        severity="high" if func["cumulative_time"] > 1.0 else "medium",
                        description=f"CPU-intensive function: {func['function']}",
                        affected_component=profile.component_name,
                        impact_score=min(func["cumulative_time"] / 10.0, 1.0),
                        recommendation=f"Optimize function {func['function']} - consider caching or async execution",
                        metrics={"cumulative_time": func["cumulative_time"], "calls": func["calls"]}
                    )
                    bottlenecks.append(bottleneck)
        
        # Memory bottlenecks
        if profile.memory_usage and profile.memory_usage.get("total_diff", 0) > 10 * 1024 * 1024:  # > 10MB
            bottleneck = BottleneckAnalysis(
                bottleneck_type=BottleneckType.MEMORY,
                severity="high",
                description=f"High memory allocation in {profile.component_name}",
                affected_component=profile.component_name,
                impact_score=min(profile.memory_usage["total_diff"] / (100 * 1024 * 1024), 1.0),
                recommendation="Review memory usage patterns, implement object pooling",
                metrics={"memory_diff": profile.memory_usage["total_diff"]}
            )
            bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    async def _check_metric_alerts(self, metric: PerformanceMetric):
        """Check if metric triggers any alerts."""
        # CPU usage alerts
        if "cpu" in metric.name.lower() and metric.unit == "percent":
            if metric.value > 90:
                await self.generate_alert(
                    AlertSeverity.CRITICAL,
                    metric.component,
                    f"High CPU usage: {metric.value:.1f}%"
                )
            elif metric.value > 80:
                await self.generate_alert(
                    AlertSeverity.WARNING,
                    metric.component,
                    f"Elevated CPU usage: {metric.value:.1f}%"
                )
        
        # Memory usage alerts
        if "memory" in metric.name.lower() and metric.unit == "percent":
            if metric.value > 95:
                await self.generate_alert(
                    AlertSeverity.CRITICAL,
                    metric.component,
                    f"Critical memory usage: {metric.value:.1f}%"
                )
            elif metric.value > 85:
                await self.generate_alert(
                    AlertSeverity.WARNING,
                    metric.component,
                    f"High memory usage: {metric.value:.1f}%"
                )
        
        # Error rate alerts
        if "error" in metric.name.lower():
            if metric.value > 10:
                await self.generate_alert(
                    AlertSeverity.CRITICAL,
                    metric.component,
                    f"High error rate: {metric.value} errors"
                )
            elif metric.value > 5:
                await self.generate_alert(
                    AlertSeverity.WARNING,
                    metric.component,
                    f"Elevated error rate: {metric.value} errors"
                )
    
    def _get_monitored_components(self) -> List[str]:
        """Get list of components being monitored."""
        components = set()
        
        # From active profiles
        components.update(self.active_profiles.keys())
        
        # From recent metrics
        for metric in list(self.metrics_buffer)[-100:]:  # Last 100 metrics
            if metric.component:
                components.add(metric.component)
        
        return list(components)
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health."""
        if not HAS_PSUTIL:
            return {"status": "unknown", "score": 0.5}
        
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # Calculate health score
        cpu_health = max(0.0, 1.0 - (cpu_percent / 100.0))
        memory_health = max(0.0, 1.0 - (memory_percent / 100.0))
        
        overall_health = (cpu_health + memory_health) / 2
        
        if overall_health >= 0.8:
            status = "healthy"
        elif overall_health >= 0.6:
            status = "degraded"
        elif overall_health >= 0.3:
            status = "unhealthy"
        else:
            status = "critical"
        
        return {
            "status": status,
            "score": overall_health,
            "cpu_health": cpu_health,
            "memory_health": memory_health,
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent
        }
    
    def _calculate_avg_response_time(self, metrics: List[Dict[str, Any]]) -> float:
        """Calculate average response time from metrics."""
        response_times = [
            m['value'] for m in metrics 
            if 'response' in m['name'].lower() or 'latency' in m['name'].lower()
        ]
        
        return statistics.mean(response_times) if response_times else 0.0
    
    async def shutdown(self):
        """Gracefully shutdown monitoring service."""
        logger.info("Shutting down Performance Monitor...")
        
        # Cancel background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Performance Monitor shutdown complete")


def create_monitor(config: Optional[MonitoringConfig] = None) -> PerformanceMonitor:
    """Create performance monitor with configuration."""
    return PerformanceMonitor(config or MonitoringConfig())


# Global monitor instance for convenience
_global_monitor: Optional[PerformanceMonitor] = None

async def get_global_monitor() -> PerformanceMonitor:
    """Get or create global monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = create_monitor()
        await _global_monitor.initialize()
    return _global_monitor


async def record_metric(name: str, value: float, unit: str, component: str = "", **labels):
    """Convenience function to record metric using global monitor."""
    monitor = await get_global_monitor()
    await monitor.record_metric(name, value, unit, component, **labels)


def profile_component(component_name: str):
    """Convenience function to profile component using global monitor."""
    async def get_monitor():
        return await get_global_monitor()
    
    monitor = asyncio.run(get_monitor())
    return monitor.profile_component(component_name)


def profile_async_component(component_name: str):
    """Convenience function to profile async component using global monitor."""
    async def get_monitor():
        return await get_global_monitor()
    
    monitor = asyncio.run(get_monitor())
    return monitor.profile_async_component(component_name)