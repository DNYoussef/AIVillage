"""Agent-specific metrics collection and monitoring.

This module provides metrics collection and monitoring specifically designed 
for AIVillage agents, including task processing metrics, performance tracking,
and agent health monitoring.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Union
from pathlib import Path

from core.error_handling import AIVillageException, ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent operational states."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    OFFLINE = "offline"


class TaskStatus(Enum):
    """Task processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class TaskMetrics:
    """Metrics for individual task processing."""
    task_id: str
    agent_id: str
    task_type: str
    status: TaskStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    processing_duration: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    input_size_bytes: Optional[int] = None
    output_size_bytes: Optional[int] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def mark_completed(self, end_time: Optional[datetime] = None) -> None:
        """Mark task as completed."""
        if end_time is None:
            end_time = datetime.now()
        self.end_time = end_time
        self.processing_duration = (end_time - self.start_time).total_seconds()
        self.status = TaskStatus.COMPLETED
    
    def mark_failed(self, error: str, end_time: Optional[datetime] = None) -> None:
        """Mark task as failed."""
        if end_time is None:
            end_time = datetime.now()
        self.end_time = end_time
        self.processing_duration = (end_time - self.start_time).total_seconds()
        self.status = TaskStatus.FAILED
        self.error_message = error


@dataclass
class AgentHealthMetrics:
    """Comprehensive agent health metrics."""
    agent_id: str
    state: AgentState
    last_activity: datetime
    uptime_seconds: float
    total_tasks_processed: int
    successful_tasks: int
    failed_tasks: int
    average_processing_time: float
    current_memory_usage_mb: float
    peak_memory_usage_mb: float
    error_rate: float
    throughput_tasks_per_minute: float
    queue_size: int
    active_connections: int
    custom_health_indicators: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate task success rate."""
        if self.total_tasks_processed == 0:
            return 1.0
        return self.successful_tasks / self.total_tasks_processed
    
    @property
    def health_score(self) -> float:
        """Calculate overall health score (0-100)."""
        # Base score starts at 100
        score = 100.0
        
        # Penalize for high error rate
        score -= min(50, self.error_rate * 100)
        
        # Penalize for high memory usage (assuming 1GB is high)
        memory_penalty = min(20, (self.current_memory_usage_mb / 1024) * 10)
        score -= memory_penalty
        
        # Penalize for large queue size
        queue_penalty = min(15, (self.queue_size / 100) * 10)
        score -= queue_penalty
        
        # Bonus for high success rate
        score += (self.success_rate - 0.9) * 20 if self.success_rate > 0.9 else 0
        
        return max(0.0, min(100.0, score))


class AgentMetricsCollector:
    """Collects and manages metrics for a single agent."""
    
    def __init__(self, agent_id: str, max_history_size: int = 10000):
        """Initialize agent metrics collector.
        
        Args:
            agent_id: Unique identifier for the agent
            max_history_size: Maximum number of historical records to keep
        """
        self.agent_id = agent_id
        self.max_history_size = max_history_size
        
        # State tracking
        self.state = AgentState.OFFLINE
        self.start_time = datetime.now()
        self.last_activity = datetime.now()
        
        # Task metrics
        self.active_tasks: Dict[str, TaskMetrics] = {}
        self.completed_tasks: deque = deque(maxlen=max_history_size)
        self.task_history_by_type: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Performance metrics
        self.total_tasks_processed = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.processing_times: deque = deque(maxlen=1000)
        self.memory_usage_history: deque = deque(maxlen=1000)
        self.peak_memory_usage = 0.0
        
        # Error tracking
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.recent_errors: deque = deque(maxlen=100)
        
        # Queue and connection metrics
        self.queue_size = 0
        self.active_connections = 0
        
        # Custom metrics
        self.custom_metrics: Dict[str, Any] = {}
        
        logger.info(f"AgentMetricsCollector initialized for agent {agent_id}")
    
    def set_state(self, state: AgentState) -> None:
        """Update agent state."""
        old_state = self.state
        self.state = state
        self.last_activity = datetime.now()
        
        logger.debug(f"Agent {self.agent_id} state changed: {old_state.value} -> {state.value}")
    
    def start_task(self, task_id: str, task_type: str, input_size_bytes: Optional[int] = None) -> TaskMetrics:
        """Start tracking a new task."""
        task_metrics = TaskMetrics(
            task_id=task_id,
            agent_id=self.agent_id,
            task_type=task_type,
            status=TaskStatus.PROCESSING,
            start_time=datetime.now(),
            input_size_bytes=input_size_bytes
        )
        
        self.active_tasks[task_id] = task_metrics
        self.last_activity = datetime.now()
        self.set_state(AgentState.PROCESSING)
        
        logger.debug(f"Started tracking task {task_id} of type {task_type}")
        return task_metrics
    
    def complete_task(self, task_id: str, output_size_bytes: Optional[int] = None, 
                     custom_metrics: Optional[Dict[str, Any]] = None) -> Optional[TaskMetrics]:
        """Mark a task as completed."""
        if task_id not in self.active_tasks:
            logger.warning(f"Attempted to complete unknown task {task_id}")
            return None
        
        task_metrics = self.active_tasks[task_id]
        task_metrics.mark_completed()
        
        if output_size_bytes is not None:
            task_metrics.output_size_bytes = output_size_bytes
        
        if custom_metrics:
            task_metrics.custom_metrics.update(custom_metrics)
        
        # Update statistics
        self.total_tasks_processed += 1
        self.successful_tasks += 1
        self.processing_times.append(task_metrics.processing_duration)
        
        # Move to completed tasks
        self.completed_tasks.append(task_metrics)
        self.task_history_by_type[task_metrics.task_type].append(task_metrics)
        del self.active_tasks[task_id]
        
        # Update state
        if not self.active_tasks:
            self.set_state(AgentState.IDLE)
        
        self.last_activity = datetime.now()
        
        logger.debug(f"Task {task_id} completed in {task_metrics.processing_duration:.3f}s")
        return task_metrics
    
    def fail_task(self, task_id: str, error_message: str, error_category: Optional[str] = None) -> Optional[TaskMetrics]:
        """Mark a task as failed."""
        if task_id not in self.active_tasks:
            logger.warning(f"Attempted to fail unknown task {task_id}")
            return None
        
        task_metrics = self.active_tasks[task_id]
        task_metrics.mark_failed(error_message)
        
        # Update statistics
        self.total_tasks_processed += 1
        self.failed_tasks += 1
        self.processing_times.append(task_metrics.processing_duration)
        
        # Track error
        error_key = error_category or "unknown"
        self.error_counts[error_key] += 1
        self.recent_errors.append({
            "timestamp": datetime.now(),
            "task_id": task_id,
            "error": error_message,
            "category": error_category
        })
        
        # Move to completed tasks
        self.completed_tasks.append(task_metrics)
        self.task_history_by_type[task_metrics.task_type].append(task_metrics)
        del self.active_tasks[task_id]
        
        # Update state
        if not self.active_tasks:
            self.set_state(AgentState.ERROR if self.failed_tasks > self.successful_tasks else AgentState.IDLE)
        
        self.last_activity = datetime.now()
        
        logger.warning(f"Task {task_id} failed: {error_message}")
        return task_metrics
    
    def update_memory_usage(self, memory_mb: float) -> None:
        """Update current memory usage."""
        self.memory_usage_history.append((datetime.now(), memory_mb))
        self.peak_memory_usage = max(self.peak_memory_usage, memory_mb)
    
    def update_queue_size(self, size: int) -> None:
        """Update queue size."""
        self.queue_size = size
    
    def update_connections(self, count: int) -> None:
        """Update active connections count."""
        self.active_connections = count
    
    def set_custom_metric(self, name: str, value: Any) -> None:
        """Set a custom metric value."""
        self.custom_metrics[name] = value
    
    def get_health_metrics(self) -> AgentHealthMetrics:
        """Get comprehensive health metrics."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        # Calculate averages
        avg_processing_time = 0.0
        if self.processing_times:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        
        # Calculate error rate
        error_rate = 0.0
        if self.total_tasks_processed > 0:
            error_rate = self.failed_tasks / self.total_tasks_processed
        
        # Calculate throughput (tasks per minute)
        throughput = 0.0
        if uptime > 0:
            throughput = (self.total_tasks_processed / uptime) * 60
        
        # Current memory usage
        current_memory = 0.0
        if self.memory_usage_history:
            current_memory = self.memory_usage_history[-1][1]
        
        return AgentHealthMetrics(
            agent_id=self.agent_id,
            state=self.state,
            last_activity=self.last_activity,
            uptime_seconds=uptime,
            total_tasks_processed=self.total_tasks_processed,
            successful_tasks=self.successful_tasks,
            failed_tasks=self.failed_tasks,
            average_processing_time=avg_processing_time,
            current_memory_usage_mb=current_memory,
            peak_memory_usage_mb=self.peak_memory_usage,
            error_rate=error_rate,
            throughput_tasks_per_minute=throughput,
            queue_size=self.queue_size,
            active_connections=self.active_connections,
            custom_health_indicators=self.custom_metrics.copy()
        )
    
    def get_task_statistics(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get task processing statistics."""
        if time_window:
            cutoff_time = datetime.now() - time_window
            relevant_tasks = [
                task for task in self.completed_tasks
                if task.start_time >= cutoff_time
            ]
        else:
            relevant_tasks = list(self.completed_tasks)
        
        if not relevant_tasks:
            return {
                "total_tasks": 0,
                "successful_tasks": 0,
                "failed_tasks": 0,
                "success_rate": 1.0,
                "average_duration": 0.0,
                "task_types": {}
            }
        
        # Calculate statistics
        total = len(relevant_tasks)
        successful = len([t for t in relevant_tasks if t.status == TaskStatus.COMPLETED])
        failed = len([t for t in relevant_tasks if t.status == TaskStatus.FAILED])
        
        durations = [t.processing_duration for t in relevant_tasks if t.processing_duration is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        
        # Task type breakdown
        task_types = defaultdict(int)
        for task in relevant_tasks:
            task_types[task.task_type] += 1
        
        return {
            "total_tasks": total,
            "successful_tasks": successful,
            "failed_tasks": failed,
            "success_rate": successful / total if total > 0 else 1.0,
            "average_duration": avg_duration,
            "task_types": dict(task_types),
            "time_window": str(time_window) if time_window else "all"
        }
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary and recent errors."""
        return {
            "error_counts": dict(self.error_counts),
            "total_errors": sum(self.error_counts.values()),
            "recent_errors": list(self.recent_errors)[-10:],  # Last 10 errors
            "error_rate": self.failed_tasks / max(self.total_tasks_processed, 1)
        }
    
    def export_metrics(self, include_task_history: bool = False) -> Dict[str, Any]:
        """Export all metrics data."""
        health = self.get_health_metrics()
        stats = self.get_task_statistics()
        errors = self.get_error_summary()
        
        export_data = {
            "agent_id": self.agent_id,
            "export_timestamp": datetime.now().isoformat(),
            "health_metrics": asdict(health),
            "task_statistics": stats,
            "error_summary": errors,
            "memory_usage_history": [
                {"timestamp": ts.isoformat(), "memory_mb": mem}
                for ts, mem in list(self.memory_usage_history)
            ],
            "custom_metrics": self.custom_metrics
        }
        
        if include_task_history:
            export_data["task_history"] = [
                asdict(task) for task in list(self.completed_tasks)
            ]
        
        return export_data


class MultiAgentMetricsManager:
    """Manages metrics for multiple agents in the system."""
    
    def __init__(self, export_interval: float = 300.0):  # Export every 5 minutes
        """Initialize multi-agent metrics manager."""
        self.export_interval = export_interval
        self.agent_collectors: Dict[str, AgentMetricsCollector] = {}
        self.system_start_time = datetime.now()
        
        # Aggregated metrics
        self.system_metrics = {
            "total_agents": 0,
            "active_agents": 0,
            "total_tasks_processed": 0,
            "system_error_rate": 0.0,
            "average_agent_health": 100.0
        }
        
        # Background tasks
        self.export_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info("MultiAgentMetricsManager initialized")
    
    def register_agent(self, agent_id: str) -> AgentMetricsCollector:
        """Register a new agent for metrics collection."""
        if agent_id in self.agent_collectors:
            logger.warning(f"Agent {agent_id} is already registered")
            return self.agent_collectors[agent_id]
        
        collector = AgentMetricsCollector(agent_id)
        self.agent_collectors[agent_id] = collector
        
        logger.info(f"Registered agent {agent_id} for metrics collection")
        return collector
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from metrics collection."""
        if agent_id not in self.agent_collectors:
            return False
        
        del self.agent_collectors[agent_id]
        logger.info(f"Unregistered agent {agent_id}")
        return True
    
    def get_agent_collector(self, agent_id: str) -> Optional[AgentMetricsCollector]:
        """Get metrics collector for an agent."""
        return self.agent_collectors.get(agent_id)
    
    def update_system_metrics(self) -> None:
        """Update aggregated system metrics."""
        if not self.agent_collectors:
            return
        
        active_agents = 0
        total_tasks = 0
        total_errors = 0
        health_scores = []
        
        for collector in self.agent_collectors.values():
            health = collector.get_health_metrics()
            
            if health.state not in [AgentState.OFFLINE, AgentState.SHUTTING_DOWN]:
                active_agents += 1
            
            total_tasks += health.total_tasks_processed
            total_errors += health.failed_tasks
            health_scores.append(health.health_score)
        
        self.system_metrics.update({
            "total_agents": len(self.agent_collectors),
            "active_agents": active_agents,
            "total_tasks_processed": total_tasks,
            "system_error_rate": total_errors / max(total_tasks, 1),
            "average_agent_health": sum(health_scores) / len(health_scores) if health_scores else 100.0
        })
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get system-wide metrics overview."""
        self.update_system_metrics()
        
        uptime = (datetime.now() - self.system_start_time).total_seconds()
        
        # Get per-agent summaries
        agent_summaries = {}
        for agent_id, collector in self.agent_collectors.items():
            health = collector.get_health_metrics()
            agent_summaries[agent_id] = {
                "state": health.state.value,
                "health_score": health.health_score,
                "total_tasks": health.total_tasks_processed,
                "success_rate": health.success_rate,
                "last_activity": health.last_activity.isoformat()
            }
        
        return {
            "system_uptime_seconds": uptime,
            "system_metrics": self.system_metrics,
            "agents": agent_summaries,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Generate performance insights across all agents."""
        insights = {
            "top_performers": [],
            "problematic_agents": [],
            "trending_task_types": {},
            "system_bottlenecks": [],
            "recommendations": []
        }
        
        # Analyze each agent
        agent_performances = []
        for agent_id, collector in self.agent_collectors.items():
            health = collector.get_health_metrics()
            stats = collector.get_task_statistics(timedelta(hours=1))  # Last hour
            
            agent_performances.append({
                "agent_id": agent_id,
                "health_score": health.health_score,
                "throughput": health.throughput_tasks_per_minute,
                "success_rate": health.success_rate,
                "avg_processing_time": health.average_processing_time
            })
        
        # Sort by health score
        agent_performances.sort(key=lambda x: x["health_score"], reverse=True)
        
        # Top performers (top 3 or top 25%)
        top_count = max(3, len(agent_performances) // 4)
        insights["top_performers"] = agent_performances[:top_count]
        
        # Problematic agents (bottom 25% or health < 60)
        problematic = [
            agent for agent in agent_performances
            if agent["health_score"] < 60 or agent["success_rate"] < 0.8
        ]
        insights["problematic_agents"] = problematic
        
        # Generate recommendations
        if problematic:
            insights["recommendations"].append(
                f"Monitor {len(problematic)} agents with low health scores or success rates"
            )
        
        if self.system_metrics["system_error_rate"] > 0.1:
            insights["recommendations"].append(
                "System error rate is above 10% - investigate common failure patterns"
            )
        
        return insights
    
    async def start_background_export(self, export_path: str) -> None:
        """Start background metrics export."""
        if self.is_running:
            logger.warning("Background export is already running")
            return
        
        self.is_running = True
        self.export_task = asyncio.create_task(self._export_loop(export_path))
        logger.info(f"Started background metrics export to {export_path}")
    
    async def stop_background_export(self) -> None:
        """Stop background metrics export."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.export_task:
            self.export_task.cancel()
            try:
                await self.export_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped background metrics export")
    
    async def _export_loop(self, export_path: str) -> None:
        """Background export loop."""
        while self.is_running:
            try:
                # Export system overview
                overview = self.get_system_overview()
                insights = self.get_performance_insights()
                
                export_data = {
                    "system_overview": overview,
                    "performance_insights": insights,
                    "detailed_metrics": {
                        agent_id: collector.export_metrics()
                        for agent_id, collector in self.agent_collectors.items()
                    }
                }
                
                # Write to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = Path(export_path) / f"agent_metrics_{timestamp}.json"
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                logger.debug(f"Exported metrics to {file_path}")
                
                # Clean up old exports (keep last 10)
                export_files = sorted(Path(export_path).glob("agent_metrics_*.json"))
                if len(export_files) > 10:
                    for old_file in export_files[:-10]:
                        old_file.unlink()
                
                await asyncio.sleep(self.export_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics export loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying


# Global metrics manager instance
_global_metrics_manager: Optional[MultiAgentMetricsManager] = None


def get_global_metrics_manager() -> MultiAgentMetricsManager:
    """Get or create global metrics manager."""
    global _global_metrics_manager
    if _global_metrics_manager is None:
        _global_metrics_manager = MultiAgentMetricsManager()
    return _global_metrics_manager


def get_agent_metrics(agent_id: str) -> Optional[AgentMetricsCollector]:
    """Get metrics collector for an agent."""
    manager = get_global_metrics_manager()
    return manager.get_agent_collector(agent_id)


def register_agent_metrics(agent_id: str) -> AgentMetricsCollector:
    """Register an agent for metrics collection."""
    manager = get_global_metrics_manager()
    return manager.register_agent(agent_id)


if __name__ == "__main__":
    async def demo():
        """Demonstrate agent metrics collection."""
        manager = MultiAgentMetricsManager()
        
        # Register some agents
        agent1 = manager.register_agent("test_agent_1")
        agent2 = manager.register_agent("test_agent_2")
        
        # Simulate some task processing
        print("Simulating task processing...")
        
        # Agent 1 processes some tasks successfully
        for i in range(10):
            task_id = f"task_1_{i}"
            agent1.start_task(task_id, "text_processing")
            await asyncio.sleep(0.1)  # Simulate processing
            agent1.complete_task(task_id)
        
        # Agent 2 has some failures
        for i in range(8):
            task_id = f"task_2_{i}"
            agent2.start_task(task_id, "data_analysis")
            await asyncio.sleep(0.1)
            if i < 6:
                agent2.complete_task(task_id)
            else:
                agent2.fail_task(task_id, "Processing error", "timeout")
        
        # Get system overview
        print("\n=== System Overview ===")
        overview = manager.get_system_overview()
        print(f"Total agents: {overview['system_metrics']['total_agents']}")
        print(f"Tasks processed: {overview['system_metrics']['total_tasks_processed']}")
        print(f"System error rate: {overview['system_metrics']['system_error_rate']:.2%}")
        
        # Get performance insights
        print("\n=== Performance Insights ===")
        insights = manager.get_performance_insights()
        
        if insights['top_performers']:
            print("Top performing agents:")
            for agent in insights['top_performers']:
                print(f"  - {agent['agent_id']}: Health {agent['health_score']:.1f}")
        
        if insights['problematic_agents']:
            print("Problematic agents:")
            for agent in insights['problematic_agents']:
                print(f"  - {agent['agent_id']}: Health {agent['health_score']:.1f}, Success {agent['success_rate']:.1%}")
        
        if insights['recommendations']:
            print("Recommendations:")
            for rec in insights['recommendations']:
                print(f"  - {rec}")
    
    # Run the demo
    asyncio.run(demo())