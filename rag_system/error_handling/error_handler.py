import logging
import sys
from typing import Any, Dict, Optional, List, Tuple
from functools import wraps
import traceback
import time
import json
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import psutil

@dataclass
class ErrorMetrics:
    error_count: int = 0
    error_types: Dict[str, int] = field(default_factory=dict)
    component_errors: Dict[str, int] = field(default_factory=dict)
    last_error_time: Optional[float] = None
    error_rate: float = 0.0
    total_handled: int = 0
    total_unhandled: int = 0

class AIVillageException(Exception):
    """Custom exception class for AI Village errors."""
    def __init__(self, message: str, component: str = None, severity: str = "ERROR"):
        self.message = message
        self.component = component
        self.severity = severity
        self.timestamp = datetime.now()
        super().__init__(self.message)

class PerformanceMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.operation_times: Dict[str, List[float]] = {}
        self.memory_usage: List[Tuple[float, float]] = []
        self.cpu_usage: List[Tuple[float, float]] = []
        self._monitoring = False
        self._monitor_thread = None

    def start_monitoring(self):
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def stop_monitoring(self):
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()

    def _monitor_resources(self):
        while self._monitoring:
            current_time = time.time() - self.start_time
            self.memory_usage.append((current_time, psutil.Process().memory_percent()))
            self.cpu_usage.append((current_time, psutil.Process().cpu_percent()))
            time.sleep(1)

    def record_operation_time(self, operation: str, duration: float):
        if operation not in self.operation_times:
            self.operation_times[operation] = []
        self.operation_times[operation].append(duration)

    def get_metrics(self) -> Dict[str, Any]:
        metrics = {
            "operation_stats": {},
            "memory_usage": {
                "current": self.memory_usage[-1][1] if self.memory_usage else 0,
                "average": sum(m[1] for m in self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
            },
            "cpu_usage": {
                "current": self.cpu_usage[-1][1] if self.cpu_usage else 0,
                "average": sum(c[1] for c in self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
            }
        }
        
        for operation, times in self.operation_times.items():
            metrics["operation_stats"][operation] = {
                "average": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "count": len(times)
            }
        
        return metrics

class ErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        self.error_metrics = ErrorMetrics()
        self.performance_metrics = PerformanceMetrics()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.error_callbacks: List[callable] = []
        self.performance_metrics.start_monitoring()

    def setup_logging(self):
        """Set up comprehensive logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ai_village.log'),
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('error.log', level=logging.ERROR),
                logging.FileHandler('performance.log', level=logging.INFO)
            ]
        )

        # Create separate loggers for different components
        component_loggers = ['task_manager', 'planning', 'knowledge', 'rag']
        for component in component_loggers:
            logger = logging.getLogger(f'ai_village.{component}')
            handler = logging.FileHandler(f'{component}.log')
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)

    def register_error_callback(self, callback: callable):
        """Register a callback to be called when errors occur."""
        self.error_callbacks.append(callback)

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log an error with optional context and update metrics."""
        try:
            error_type = type(error).__name__
            self.error_metrics.error_count += 1
            self.error_metrics.error_types[error_type] = self.error_metrics.error_types.get(error_type, 0) + 1
            
            if isinstance(error, AIVillageException) and error.component:
                self.error_metrics.component_errors[error.component] = self.error_metrics.component_errors.get(error.component, 0) + 1
            
            current_time = time.time()
            if self.error_metrics.last_error_time:
                time_diff = current_time - self.error_metrics.last_error_time
                self.error_metrics.error_rate = 1 / time_diff if time_diff > 0 else float('inf')
            
            self.error_metrics.last_error_time = current_time

            error_message = f"Error: {str(error)}"
            if context:
                error_message += f" Context: {json.dumps(context)}"

            self.logger.error(error_message, exc_info=True)

            # Asynchronously notify error callbacks
            for callback in self.error_callbacks:
                self.executor.submit(callback, error, context)

        except Exception as e:
            self.logger.critical(f"Error in error logging system: {str(e)}", exc_info=True)

    def handle_error(self, func):
        """Enhanced decorator to handle errors in functions with performance monitoring."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                self.performance_metrics.record_operation_time(func.__name__, duration)
                return result
            except Exception as e:
                self.error_metrics.total_unhandled += 1
                self.log_error(e, {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs),
                    'duration': time.time() - start_time
                })
                if isinstance(e, AIVillageException):
                    raise e
                raise AIVillageException(f"Error in {func.__name__}: {str(e)}", 
                                       component=func.__module__)
        return wrapper

    def safe_execute(self, func):
        """Enhanced decorator to safely execute a function with performance monitoring."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                self.performance_metrics.record_operation_time(func.__name__, duration)
                self.error_metrics.total_handled += 1
                return result
            except Exception as e:
                self.log_error(e, {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs),
                    'duration': time.time() - start_time
                })
                return {
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'context': {
                        'function': func.__name__,
                        'duration': time.time() - start_time
                    }
                }
        return wrapper

    def get_error_metrics(self) -> Dict[str, Any]:
        """Get current error metrics."""
        return {
            'error_count': self.error_metrics.error_count,
            'error_types': dict(self.error_metrics.error_types),
            'component_errors': dict(self.error_metrics.component_errors),
            'error_rate': self.error_metrics.error_rate,
            'total_handled': self.error_metrics.total_handled,
            'total_unhandled': self.error_metrics.total_unhandled,
            'performance_metrics': self.performance_metrics.get_metrics()
        }

    async def monitor_performance(self, interval: int = 60):
        """Periodically monitor and log performance metrics."""
        while True:
            try:
                metrics = self.get_error_metrics()
                self.logger.info(f"Performance metrics: {json.dumps(metrics, indent=2)}")
                await asyncio.sleep(interval)
            except Exception as e:
                self.log_error(e, {'component': 'performance_monitoring'})
                await asyncio.sleep(interval)

    def __del__(self):
        """Cleanup resources."""
        self.performance_metrics.stop_monitoring()
        self.executor.shutdown(wait=False)

# Create a singleton instance
error_handler = ErrorHandler()

# Start performance monitoring
asyncio.create_task(error_handler.monitor_performance())
