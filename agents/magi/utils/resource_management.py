"""Resource management utilities for MAGI agent system."""

import os
import psutil
import asyncio
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import time
from ..core.exceptions import ResourceError
from ..core.constants import SYSTEM_CONSTANTS
from .logging import get_logger

logger = get_logger(__name__)

@dataclass
class ResourceLimits:
    """Resource limits configuration."""
    max_memory_percent: float = SYSTEM_CONSTANTS["MAX_MEMORY_PERCENT"]
    max_cpu_percent: float = SYSTEM_CONSTANTS["MAX_CPU_PERCENT"]
    max_concurrent_tasks: int = SYSTEM_CONSTANTS["MAX_CONCURRENT_TASKS"]
    max_execution_time: int = SYSTEM_CONSTANTS["MAX_EXECUTION_TIME"]

class ResourceMonitor:
    """Monitor system resources."""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.process = psutil.Process()
        self._lock = threading.Lock()
        self._active_tasks = 0
        self._monitoring = False
        self._history: List[Dict[str, Any]] = []
    
    async def start_monitoring(self, interval: float = 1.0):
        """Start resource monitoring."""
        self._monitoring = True
        while self._monitoring:
            self._record_metrics()
            await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
    
    def _record_metrics(self):
        """Record current resource metrics."""
        with self._lock:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': self.process.cpu_percent(),
                'memory_percent': self.process.memory_percent(),
                'active_tasks': self._active_tasks
            }
            self._history.append(metrics)
            
            # Keep only recent history
            if len(self._history) > 1000:
                self._history = self._history[-1000:]
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        return {
            'cpu_percent': self.process.cpu_percent(),
            'memory_percent': self.process.memory_percent(),
            'active_tasks': self._active_tasks
        }
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get resource usage history."""
        return self._history.copy()
    
    def check_resources(self) -> Tuple[bool, str]:
        """Check if resources are available."""
        usage = self.get_current_usage()
        
        if usage['memory_percent'] > self.limits.max_memory_percent:
            return False, f"Memory usage too high: {usage['memory_percent']}%"
        
        if usage['cpu_percent'] > self.limits.max_cpu_percent:
            return False, f"CPU usage too high: {usage['cpu_percent']}%"
        
        if usage['active_tasks'] >= self.limits.max_concurrent_tasks:
            return False, f"Too many active tasks: {usage['active_tasks']}"
        
        return True, "Resources available"

class ResourceManager:
    """Manager for system resources and monitoring."""
    
    def __init__(
        self,
        monitoring_interval: float = 1.0,
        history_size: int = 3600,  # 1 hour at 1 second intervals
        cpu_threshold: float = 90.0,
        memory_threshold: float = 90.0,
        disk_threshold: float = 90.0
    ):
        """
        Initialize resource manager.
        
        Args:
            monitoring_interval: Interval between resource checks in seconds
            history_size: Number of historical metrics to keep
            cpu_threshold: CPU usage threshold percentage
            memory_threshold: Memory usage threshold percentage
            disk_threshold: Disk usage threshold percentage
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
        
        self.metrics_history: List[ResourceMetrics] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def start_monitoring(self) -> None:
        """Start resource monitoring in a background thread."""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_resources)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
            logger.info("Resource monitoring stopped")
    
    def _monitor_resources(self) -> None:
        """Monitor system resources periodically."""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Trim history if needed
                if len(self.metrics_history) > self.history_size:
                    self.metrics_history = self.metrics_history[-self.history_size:]
                
                # Check thresholds
                self._check_thresholds(metrics)
                
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error monitoring resources: {str(e)}")
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system resource metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            return ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage_percent=disk.percent,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                timestamp=datetime.now()
            )
        except Exception as e:
            raise ResourceError(f"Error collecting metrics: {str(e)}")
    
    def _check_thresholds(self, metrics: ResourceMetrics) -> None:
        """Check if resource usage exceeds thresholds."""
        if metrics.cpu_percent > self.cpu_threshold:
            logger.warning(f"CPU usage ({metrics.cpu_percent}%) exceeds threshold ({self.cpu_threshold}%)")
        
        if metrics.memory_percent > self.memory_threshold:
            logger.warning(f"Memory usage ({metrics.memory_percent}%) exceeds threshold ({self.memory_threshold}%)")
        
        if metrics.disk_usage_percent > self.disk_threshold:
            logger.warning(f"Disk usage ({metrics.disk_usage_percent}%) exceeds threshold ({self.disk_threshold}%)")
    
    async def get_resource_availability(self) -> Dict[str, float]:
        """
        Get current resource availability percentages.
        
        Returns:
            Dictionary with resource availability percentages
        """
        try:
            metrics = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._collect_metrics
            )
            
            return {
                'cpu_available': 100 - metrics.cpu_percent,
                'memory_available': 100 - metrics.memory_percent,
                'disk_available': 100 - metrics.disk_usage_percent
            }
        except Exception as e:
            raise ResourceError(f"Error getting resource availability: {str(e)}")
    
    def get_metrics_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[ResourceMetrics]:
        """
        Get historical metrics within the specified time range.
        
        Args:
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            
        Returns:
            List of ResourceMetrics objects
        """
        if not start_time and not end_time:
            return self.metrics_history
        
        filtered_metrics = [
            metric for metric in self.metrics_history
            if (not start_time or metric.timestamp >= start_time) and
               (not end_time or metric.timestamp <= end_time)
        ]
        
        return filtered_metrics
    
    async def check_resource_requirements(
        self,
        required_resources: Dict[str, float]
    ) -> Tuple[bool, str]:
        """
        Check if required resources are available.
        
        Args:
            required_resources: Dictionary of required resource percentages
            
        Returns:
            Tuple of (bool: requirements met, str: reason if not met)
        """
        try:
            availability = await self.get_resource_availability()
            
            for resource, required in required_resources.items():
                if resource not in availability:
                    return False, f"Unknown resource type: {resource}"
                
                if availability[resource] < required:
                    return False, f"Insufficient {resource}: {availability[resource]}% available, {required}% required"
            
            return True, "Resource requirements met"
        except Exception as e:
            raise ResourceError(f"Error checking resource requirements: {str(e)}")
    
    def cleanup(self) -> None:
        """Clean up resources and stop monitoring."""
        self.stop_monitoring()
        self._executor.shutdown(wait=True)
        logger.info("Resource manager cleanup completed")

# Example usage
if __name__ == "__main__":
    resource_manager = ResourceManager()
    resource_manager.start_monitoring()
    
    try:
        # Monitor for 10 seconds
        time.sleep(10)
        
        # Get current metrics
        metrics = resource_manager._collect_metrics()
        print(f"Current CPU usage: {metrics.cpu_percent}%")
        print(f"Current memory usage: {metrics.memory_percent}%")
        print(f"Current disk usage: {metrics.disk_usage_percent}%")
        
    finally:
        resource_manager.cleanup()

