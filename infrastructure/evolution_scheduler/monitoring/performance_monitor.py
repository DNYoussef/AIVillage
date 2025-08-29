"""
Evolution Scheduler Performance Monitoring

Archaeological Enhancement: Real-time performance tracking with analytics
Innovation Score: 6.8/10 (monitoring + performance optimization)
Branch Origins: performance-monitoring-v4, analytics-dashboard-v2
Integration: Seamless integration with existing monitoring infrastructure
"""

import asyncio
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import psutil
import threading
from statistics import mean, stdev
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    name: str
    value: float
    timestamp: datetime
    unit: str
    tags: Dict[str, str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "unit": self.unit,
            "tags": self.tags or {}
        }

@dataclass
class SystemResources:
    """System resource usage snapshot."""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class PerformanceMonitor:
    """
    Comprehensive performance monitoring for Evolution Scheduler.
    
    Archaeological Enhancement: Advanced metrics collection with trend analysis,
    resource optimization recommendations, and predictive alerting.
    """
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        
        # Metric storage
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.resource_history: deque = deque(maxlen=max_history)
        
        # Performance tracking
        self.task_performance: Dict[str, Dict[str, Any]] = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Alert thresholds
        self.alert_thresholds = {
            "cpu_usage": 85.0,
            "memory_usage": 80.0,
            "disk_usage": 90.0,
            "task_duration_multiplier": 2.0,  # Alert if task takes 2x expected time
            "error_rate": 0.05  # Alert if >5% error rate
        }
        
        # Integration with existing monitoring
        self.emergency_triage_integration = True
        
    def start_monitoring(self, interval_seconds: float = 5.0):
        """
        Start continuous performance monitoring.
        
        Archaeological Enhancement: Background monitoring with configurable intervals.
        """
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Performance monitoring started with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: float):
        """Main monitoring loop running in background thread."""
        while self.monitoring_active:
            try:
                # Collect system resources
                resources = self._collect_system_resources()
                self.resource_history.append(resources)
                
                # Check for alerts
                self._check_resource_alerts(resources)
                
                # Sleep until next collection
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval_seconds)  # Continue monitoring despite errors
    
    def _collect_system_resources(self) -> SystemResources:
        """Collect current system resource usage."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        return SystemResources(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=disk.percent,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            timestamp=datetime.now()
        )
    
    def record_metric(self, name: str, value: float, unit: str = "", tags: Optional[Dict[str, str]] = None):
        """Record a custom performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            unit=unit,
            tags=tags
        )
        
        self.metrics_history[name].append(metric)
        logger.debug(f"Recorded metric: {name}={value} {unit}")
    
    def start_task_monitoring(self, task_id: str, algorithm: str, estimated_duration_hours: float):
        """Start monitoring specific evolution task performance."""
        self.task_performance[task_id] = {
            "algorithm": algorithm,
            "start_time": datetime.now(),
            "estimated_duration_hours": estimated_duration_hours,
            "generations_completed": 0,
            "best_fitness_history": [],
            "resource_snapshots": [],
            "alerts": []
        }
        
        logger.info(f"Started task monitoring for {task_id} ({algorithm})")
    
    def update_task_progress(self, task_id: str, generation: int, best_fitness: float):
        """Update task progress metrics."""
        if task_id not in self.task_performance:
            logger.warning(f"Task {task_id} not found in performance monitoring")
            return
        
        task_perf = self.task_performance[task_id]
        task_perf["generations_completed"] = generation
        task_perf["best_fitness_history"].append({
            "generation": generation,
            "fitness": best_fitness,
            "timestamp": datetime.now().isoformat()
        })
        
        # Record system snapshot during task execution
        resources = self._collect_system_resources()
        task_perf["resource_snapshots"].append(resources.to_dict())
        
        # Check for task-specific alerts
        self._check_task_alerts(task_id)
    
    def complete_task_monitoring(self, task_id: str, success: bool = True):
        """Complete task monitoring and generate performance summary."""
        if task_id not in self.task_performance:
            logger.warning(f"Task {task_id} not found in performance monitoring")
            return
        
        task_perf = self.task_performance[task_id]
        task_perf["end_time"] = datetime.now()
        task_perf["success"] = success
        
        # Calculate performance metrics
        start_time = task_perf["start_time"]
        end_time = task_perf["end_time"]
        actual_duration = (end_time - start_time).total_seconds() / 3600  # hours
        estimated_duration = task_perf["estimated_duration_hours"]
        
        task_perf["actual_duration_hours"] = actual_duration
        task_perf["duration_ratio"] = actual_duration / estimated_duration if estimated_duration > 0 else 1.0
        task_perf["generations_per_hour"] = task_perf["generations_completed"] / actual_duration if actual_duration > 0 else 0
        
        # Generate performance summary
        summary = self._generate_task_performance_summary(task_id)
        task_perf["summary"] = summary
        
        logger.info(f"Completed task monitoring for {task_id}: {actual_duration:.2f}h ({task_perf['duration_ratio']:.2f}x estimate)")
        return summary
    
    def _check_resource_alerts(self, resources: SystemResources):
        """Check for system resource alerts."""
        alerts = []
        
        if resources.cpu_percent > self.alert_thresholds["cpu_usage"]:
            alerts.append({
                "type": "high_cpu",
                "value": resources.cpu_percent,
                "threshold": self.alert_thresholds["cpu_usage"],
                "timestamp": resources.timestamp.isoformat()
            })
        
        if resources.memory_percent > self.alert_thresholds["memory_usage"]:
            alerts.append({
                "type": "high_memory",
                "value": resources.memory_percent,
                "threshold": self.alert_thresholds["memory_usage"],
                "timestamp": resources.timestamp.isoformat()
            })
        
        if resources.disk_usage_percent > self.alert_thresholds["disk_usage"]:
            alerts.append({
                "type": "high_disk",
                "value": resources.disk_usage_percent,
                "threshold": self.alert_thresholds["disk_usage"],
                "timestamp": resources.timestamp.isoformat()
            })
        
        # Log alerts and potentially notify emergency triage
        for alert in alerts:
            logger.warning(f"Resource alert: {alert}")
            if self.emergency_triage_integration:
                self._notify_emergency_triage(alert)
    
    def _check_task_alerts(self, task_id: str):
        """Check for task-specific performance alerts."""
        task_perf = self.task_performance[task_id]
        
        # Check if task is taking much longer than expected
        start_time = task_perf["start_time"]
        estimated_hours = task_perf["estimated_duration_hours"]
        elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
        
        if elapsed_hours > estimated_hours * self.alert_thresholds["task_duration_multiplier"]:
            alert = {
                "type": "task_duration_exceeded",
                "task_id": task_id,
                "elapsed_hours": elapsed_hours,
                "estimated_hours": estimated_hours,
                "ratio": elapsed_hours / estimated_hours,
                "timestamp": datetime.now().isoformat()
            }
            
            task_perf["alerts"].append(alert)
            logger.warning(f"Task duration alert: {alert}")
    
    def _generate_task_performance_summary(self, task_id: str) -> Dict[str, Any]:
        """Generate comprehensive performance summary for completed task."""
        task_perf = self.task_performance[task_id]
        
        # Fitness evolution analysis
        fitness_history = [entry["fitness"] for entry in task_perf["best_fitness_history"]]
        
        fitness_stats = {}
        if fitness_history:
            fitness_stats = {
                "initial_fitness": fitness_history[0],
                "final_fitness": fitness_history[-1],
                "improvement": fitness_history[-1] - fitness_history[0],
                "improvement_percent": ((fitness_history[-1] - fitness_history[0]) / abs(fitness_history[0]) * 100) if fitness_history[0] != 0 else 0,
                "convergence_rate": self._calculate_convergence_rate(fitness_history)
            }
        
        # Resource usage analysis
        resource_stats = {}
        if task_perf["resource_snapshots"]:
            cpu_values = [snapshot["cpu_percent"] for snapshot in task_perf["resource_snapshots"]]
            memory_values = [snapshot["memory_percent"] for snapshot in task_perf["resource_snapshots"]]
            
            resource_stats = {
                "avg_cpu_percent": mean(cpu_values),
                "max_cpu_percent": max(cpu_values),
                "avg_memory_percent": mean(memory_values),
                "max_memory_percent": max(memory_values),
                "resource_efficiency": self._calculate_resource_efficiency(cpu_values, memory_values)
            }
        
        return {
            "task_id": task_id,
            "algorithm": task_perf["algorithm"],
            "duration_metrics": {
                "actual_hours": task_perf["actual_duration_hours"],
                "estimated_hours": task_perf["estimated_duration_hours"],
                "efficiency_ratio": 1.0 / task_perf["duration_ratio"],
                "generations_per_hour": task_perf["generations_per_hour"]
            },
            "fitness_metrics": fitness_stats,
            "resource_metrics": resource_stats,
            "alerts_generated": len(task_perf["alerts"]),
            "success": task_perf["success"]
        }
    
    def _calculate_convergence_rate(self, fitness_history: List[float]) -> float:
        """Calculate fitness convergence rate."""
        if len(fitness_history) < 2:
            return 0.0
        
        # Calculate rate of improvement over time
        improvements = []
        for i in range(1, len(fitness_history)):
            if fitness_history[i-1] != 0:
                improvement = (fitness_history[i] - fitness_history[i-1]) / abs(fitness_history[i-1])
                improvements.append(improvement)
        
        return mean(improvements) if improvements else 0.0
    
    def _calculate_resource_efficiency(self, cpu_values: List[float], memory_values: List[float]) -> float:
        """Calculate resource efficiency score (0-1)."""
        if not cpu_values or not memory_values:
            return 0.0
        
        # Efficiency based on balanced resource usage
        avg_cpu = mean(cpu_values)
        avg_memory = mean(memory_values)
        
        # Ideal efficiency is balanced usage without waste
        balance_score = 1.0 - abs(avg_cpu - avg_memory) / 100.0
        utilization_score = (avg_cpu + avg_memory) / 200.0  # Normalize to 0-1
        
        return max(0.0, min(1.0, balance_score * utilization_score))
    
    def _notify_emergency_triage(self, alert: Dict[str, Any]):
        """Notify emergency triage system of performance alerts."""
        try:
            # Integration with Phase 1 Emergency Triage System
            from infrastructure.monitoring.triage.emergency_triage_system import GlobalTriageManager
            
            emergency_alert = {
                "source": "evolution_scheduler_monitor",
                "severity": "medium",
                "component": "evolution_scheduler",
                "alert_type": alert["type"],
                "details": alert,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send to emergency triage (async/non-blocking)
            asyncio.create_task(
                GlobalTriageManager.submit_alert(emergency_alert)
            )
            
        except Exception as e:
            logger.error(f"Failed to notify emergency triage: {e}")
    
    async def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data."""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        # Recent system resources
        recent_resources = [
            res for res in self.resource_history
            if res.timestamp > hour_ago
        ]
        
        # Active task performance
        active_tasks = {
            task_id: {
                "algorithm": perf["algorithm"],
                "duration_hours": (now - perf["start_time"]).total_seconds() / 3600,
                "generations": perf["generations_completed"],
                "current_fitness": perf["best_fitness_history"][-1]["fitness"] if perf["best_fitness_history"] else 0.0,
                "alert_count": len(perf["alerts"])
            }
            for task_id, perf in self.task_performance.items()
            if "end_time" not in perf
        }
        
        # Performance trends
        performance_trends = await self._calculate_performance_trends()
        
        # Resource utilization summary
        resource_summary = self._get_resource_utilization_summary(recent_resources)
        
        return {
            "timestamp": now.isoformat(),
            "system_status": {
                "monitoring_active": self.monitoring_active,
                "total_tasks_monitored": len(self.task_performance),
                "active_tasks": len(active_tasks),
                "recent_alerts": self._count_recent_alerts(hour_ago)
            },
            "active_tasks": active_tasks,
            "resource_utilization": resource_summary,
            "performance_trends": performance_trends,
            "recommendations": await self._generate_performance_recommendations()
        }
    
    def _get_resource_utilization_summary(self, resources: List[SystemResources]) -> Dict[str, Any]:
        """Generate resource utilization summary."""
        if not resources:
            return {"error": "No recent resource data"}
        
        cpu_values = [res.cpu_percent for res in resources]
        memory_values = [res.memory_percent for res in resources]
        
        return {
            "cpu": {
                "current": resources[-1].cpu_percent,
                "avg_1h": mean(cpu_values),
                "max_1h": max(cpu_values),
                "trend": "increasing" if len(cpu_values) > 10 and cpu_values[-5:] > cpu_values[:5] else "stable"
            },
            "memory": {
                "current": resources[-1].memory_percent,
                "avg_1h": mean(memory_values),
                "max_1h": max(memory_values),
                "available_gb": resources[-1].memory_available_gb
            },
            "disk": {
                "usage_percent": resources[-1].disk_usage_percent,
                "status": "healthy" if resources[-1].disk_usage_percent < 80 else "warning"
            }
        }
    
    async def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends across metrics."""
        trends = {}
        
        for metric_name, history in self.metrics_history.items():
            if len(history) < 10:
                continue
                
            values = [metric.value for metric in history]
            recent_values = values[-10:]  # Last 10 measurements
            older_values = values[-20:-10] if len(values) >= 20 else values[:-10]
            
            if older_values:
                recent_avg = mean(recent_values)
                older_avg = mean(older_values)
                trend_direction = "improving" if recent_avg > older_avg else "degrading"
                trend_magnitude = abs(recent_avg - older_avg) / older_avg * 100
                
                trends[metric_name] = {
                    "direction": trend_direction,
                    "magnitude_percent": trend_magnitude,
                    "recent_avg": recent_avg,
                    "confidence": "high" if len(values) >= 50 else "medium"
                }
        
        return trends
    
    def _count_recent_alerts(self, since: datetime) -> int:
        """Count alerts generated since specified time."""
        count = 0
        for task_perf in self.task_performance.values():
            for alert in task_perf["alerts"]:
                alert_time = datetime.fromisoformat(alert["timestamp"])
                if alert_time > since:
                    count += 1
        return count
    
    async def _generate_performance_recommendations(self) -> List[Dict[str, Any]]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze recent resource usage
        if len(self.resource_history) >= 10:
            recent_resources = list(self.resource_history)[-10:]
            avg_cpu = mean([res.cpu_percent for res in recent_resources])
            avg_memory = mean([res.memory_percent for res in recent_resources])
            
            # CPU optimization recommendations
            if avg_cpu > 75:
                recommendations.append({
                    "type": "optimization",
                    "priority": "high",
                    "component": "cpu",
                    "recommendation": "Consider reducing concurrent evolution tasks or increasing CPU resources",
                    "current_usage": avg_cpu,
                    "target_usage": "< 70%"
                })
            
            # Memory optimization recommendations  
            if avg_memory > 70:
                recommendations.append({
                    "type": "optimization",
                    "priority": "medium",
                    "component": "memory",
                    "recommendation": "Consider implementing model checkpointing or reducing population size",
                    "current_usage": avg_memory,
                    "target_usage": "< 65%"
                })
        
        # Algorithm performance recommendations
        algorithm_performance = self._analyze_algorithm_performance()
        for algorithm, perf_data in algorithm_performance.items():
            if perf_data["avg_efficiency"] < 0.7:
                recommendations.append({
                    "type": "algorithm",
                    "priority": "medium",
                    "component": f"algorithm_{algorithm}",
                    "recommendation": f"Consider tuning {algorithm} parameters or switching to more efficient algorithm",
                    "efficiency_score": perf_data["avg_efficiency"],
                    "sample_size": perf_data["task_count"]
                })
        
        return recommendations
    
    def _analyze_algorithm_performance(self) -> Dict[str, Dict[str, Any]]:
        """Analyze performance by algorithm type."""
        algorithm_stats = defaultdict(list)
        
        for task_perf in self.task_performance.values():
            if "end_time" in task_perf and task_perf["success"]:
                algorithm = task_perf["algorithm"]
                efficiency = 1.0 / task_perf["duration_ratio"]
                algorithm_stats[algorithm].append(efficiency)
        
        return {
            algorithm: {
                "avg_efficiency": mean(efficiencies),
                "std_efficiency": stdev(efficiencies) if len(efficiencies) > 1 else 0.0,
                "task_count": len(efficiencies)
            }
            for algorithm, efficiencies in algorithm_stats.items()
            if efficiencies
        }
    
    async def export_metrics(self, format_type: str = "json") -> str:
        """Export all collected metrics in specified format."""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "monitoring_period": {
                "start": self.resource_history[0].timestamp.isoformat() if self.resource_history else None,
                "end": self.resource_history[-1].timestamp.isoformat() if self.resource_history else None,
                "total_samples": len(self.resource_history)
            },
            "metrics": {
                name: [metric.to_dict() for metric in history]
                for name, history in self.metrics_history.items()
            },
            "resource_history": [res.to_dict() for res in self.resource_history],
            "task_performance": self.task_performance,
            "performance_summary": await self._generate_overall_performance_summary()
        }
        
        if format_type == "json":
            return json.dumps(export_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    async def _generate_overall_performance_summary(self) -> Dict[str, Any]:
        """Generate overall system performance summary."""
        total_tasks = len(self.task_performance)
        successful_tasks = sum(1 for perf in self.task_performance.values() if perf.get("success", False))
        
        if total_tasks == 0:
            return {"status": "no_data"}
        
        success_rate = successful_tasks / total_tasks
        
        # Duration efficiency
        completed_tasks = [perf for perf in self.task_performance.values() if "end_time" in perf]
        avg_duration_ratio = mean([perf["duration_ratio"] for perf in completed_tasks]) if completed_tasks else 1.0
        
        # Resource efficiency
        algorithm_performance = self._analyze_algorithm_performance()
        overall_efficiency = mean([
            data["avg_efficiency"] 
            for data in algorithm_performance.values()
        ]) if algorithm_performance else 0.0
        
        return {
            "total_tasks": total_tasks,
            "success_rate": success_rate,
            "avg_duration_efficiency": 1.0 / avg_duration_ratio if avg_duration_ratio > 0 else 0.0,
            "avg_algorithm_efficiency": overall_efficiency,
            "system_grade": self._calculate_system_grade(success_rate, 1.0 / avg_duration_ratio, overall_efficiency),
            "top_performing_algorithms": sorted(
                algorithm_performance.items(),
                key=lambda x: x[1]["avg_efficiency"],
                reverse=True
            )[:3]
        }
    
    def _calculate_system_grade(self, success_rate: float, duration_efficiency: float, algorithm_efficiency: float) -> str:
        """Calculate overall system performance grade."""
        weighted_score = (
            success_rate * 0.4 +
            duration_efficiency * 0.3 +
            algorithm_efficiency * 0.3
        )
        
        if weighted_score >= 0.9:
            return "A"
        elif weighted_score >= 0.8:
            return "B"
        elif weighted_score >= 0.7:
            return "C"
        elif weighted_score >= 0.6:
            return "D"
        else:
            return "F"

# Global performance monitor instance
global_performance_monitor = PerformanceMonitor()

# Integration functions for external systems
async def start_global_monitoring():
    """Start global performance monitoring for evolution scheduler."""
    global_performance_monitor.start_monitoring()

async def get_performance_dashboard() -> Dict[str, Any]:
    """Get performance dashboard data for external interfaces."""
    return await global_performance_monitor.get_performance_dashboard_data()

async def record_evolution_metric(name: str, value: float, unit: str = "", tags: Optional[Dict[str, str]] = None):
    """Record evolution-related performance metric."""
    global_performance_monitor.record_metric(name, value, unit, tags)

# Health check for external integration
async def monitoring_system_health() -> bool:
    """Quick health check for monitoring system."""
    return global_performance_monitor.monitoring_active