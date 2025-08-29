"""
Base test classes for P2P infrastructure testing.

Provides standardized base classes for all P2P component tests
with proper setup, teardown, and common utilities.
"""

import unittest
import asyncio
import tempfile
import shutil
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import logging
import time
from contextlib import asynccontextmanager, contextmanager

# Import common utilities
try:
    from ..common.logging import get_logger, P2PLogger
    from ..common.monitoring import StandardMetrics, MetricsCollector
    from ..common.configuration import ConfigManager
except ImportError:
    # Fallback if common utilities not available
    get_logger = None
    P2PLogger = None
    StandardMetrics = None
    MetricsCollector = None
    ConfigManager = None

logger = logging.getLogger(__name__)


class P2PTestCase(unittest.TestCase):
    """Base test case for P2P components."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_id = f"test_{int(time.time() * 1000)}"
        self.temp_dirs: List[Path] = []
        self.cleanup_callbacks: List[Callable] = []
        
        # Set up logging
        if get_logger:
            self.logger = get_logger(f"test_{self.__class__.__name__}")
        else:
            self.logger = logging.getLogger(f"test_{self.__class__.__name__}")
        
        # Set up metrics
        if StandardMetrics:
            self.metrics = StandardMetrics(f"test_{self.__class__.__name__}")
        else:
            self.metrics = None
        
        # Set up configuration
        if ConfigManager:
            self.config_manager = ConfigManager(f"test_{self.__class__.__name__}")
        else:
            self.config_manager = None
    
    def tearDown(self):
        """Clean up test environment."""
        # Run cleanup callbacks
        for callback in reversed(self.cleanup_callbacks):
            try:
                callback()
            except Exception as e:
                self.logger.warning(f"Cleanup callback failed: {e}")
        
        # Clean up temporary directories
        for temp_dir in self.temp_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception as e:
                self.logger.warning(f"Failed to clean up temp dir {temp_dir}: {e}")
    
    def create_temp_directory(self) -> Path:
        """Create temporary directory for test."""
        temp_dir = Path(tempfile.mkdtemp(prefix=f"p2p_test_{self.test_id}_"))
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def add_cleanup_callback(self, callback: Callable):
        """Add cleanup callback to be run during teardown."""
        self.cleanup_callbacks.append(callback)
    
    def assert_eventually(self, condition: Callable[[], bool], timeout: float = 5.0, 
                         interval: float = 0.1, message: str = None):
        """Assert that condition becomes true within timeout."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if condition():
                return
            time.sleep(interval)
        
        if message:
            self.fail(message)
        else:
            self.fail(f"Condition not met within {timeout}s")
    
    def assert_metrics_recorded(self, metric_name: str):
        """Assert that a metric was recorded."""
        if not self.metrics:
            self.skipTest("Metrics not available")
        
        metrics = self.metrics.get_metrics(metric_name)
        self.assertGreater(len(metrics.get(metric_name, [])), 0, 
                          f"No samples recorded for metric {metric_name}")
    
    def get_test_config(self, **overrides) -> Dict[str, Any]:
        """Get test configuration with optional overrides."""
        base_config = {
            "test_mode": True,
            "log_level": "DEBUG",
            "timeout": 10.0,
            "max_connections": 5,
            "discovery_enabled": False
        }
        base_config.update(overrides)
        return base_config


class AsyncP2PTestCase(P2PTestCase):
    """Base async test case for P2P components."""
    
    def setUp(self):
        """Set up async test environment."""
        super().setUp()
        self.loop = None
        self.async_cleanup_callbacks: List[Callable] = []
    
    def tearDown(self):
        """Clean up async test environment."""
        # Run async cleanup callbacks
        if self.async_cleanup_callbacks:
            if self.loop and not self.loop.is_closed():
                for callback in reversed(self.async_cleanup_callbacks):
                    try:
                        self.loop.run_until_complete(callback())
                    except Exception as e:
                        self.logger.warning(f"Async cleanup callback failed: {e}")
        
        # Close event loop if we created one
        if self.loop:
            try:
                # Cancel all pending tasks
                pending = asyncio.all_tasks(self.loop)
                for task in pending:
                    task.cancel()
                
                if pending:
                    self.loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                
                self.loop.close()
            except Exception as e:
                self.logger.warning(f"Error closing event loop: {e}")
        
        super().tearDown()
    
    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop for testing."""
        if self.loop is None:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        return self.loop
    
    def run_async(self, coro):
        """Run async coroutine in test loop."""
        loop = self.get_event_loop()
        return loop.run_until_complete(coro)
    
    def add_async_cleanup_callback(self, callback: Callable):
        """Add async cleanup callback."""
        self.async_cleanup_callbacks.append(callback)
    
    async def assert_eventually_async(self, condition: Callable[[], Any], timeout: float = 5.0,
                                    interval: float = 0.1, message: str = None):
        """Async version of assert_eventually."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if asyncio.iscoroutinefunction(condition):
                    result = await condition()
                else:
                    result = condition()
                
                if result:
                    return
            except Exception:
                pass
            
            await asyncio.sleep(interval)
        
        if message:
            self.fail(message)
        else:
            self.fail(f"Async condition not met within {timeout}s")
    
    @asynccontextmanager
    async def async_test_context(self):
        """Context manager for async test setup/cleanup."""
        try:
            yield
        finally:
            # Cleanup will be handled in tearDown
            pass


class IntegrationTestCase(AsyncP2PTestCase):
    """Base class for integration tests across P2P components."""
    
    def setUp(self):
        """Set up integration test environment."""
        super().setUp()
        self.components: Dict[str, Any] = {}
        self.connections: List[Any] = []
    
    async def start_component(self, name: str, component_class, config: Dict[str, Any]):
        """Start a test component."""
        component = component_class()
        
        if hasattr(component, 'initialize'):
            await component.initialize(config)
        elif hasattr(component, 'start'):
            await component.start(config)
        
        self.components[name] = component
        
        # Add cleanup callback
        async def cleanup():
            if hasattr(component, 'shutdown'):
                await component.shutdown()
            elif hasattr(component, 'stop'):
                await component.stop()
        
        self.add_async_cleanup_callback(cleanup)
        return component
    
    async def connect_components(self, component1_name: str, component2_name: str):
        """Connect two test components."""
        comp1 = self.components[component1_name]
        comp2 = self.components[component2_name]
        
        # This would depend on the specific interface of components
        if hasattr(comp1, 'connect') and hasattr(comp2, 'get_address'):
            connection = await comp1.connect(comp2.get_address())
            self.connections.append(connection)
            return connection
    
    def assert_all_connected(self):
        """Assert that all components are connected."""
        for name, component in self.components.items():
            if hasattr(component, 'is_connected'):
                self.assertTrue(component.is_connected(), 
                              f"Component {name} not connected")
    
    async def wait_for_propagation(self, timeout: float = 5.0):
        """Wait for network changes to propagate."""
        await asyncio.sleep(0.1)  # Basic propagation delay


class PerformanceTestCase(AsyncP2PTestCase):
    """Base class for performance testing of P2P components."""
    
    def setUp(self):
        """Set up performance test environment."""
        super().setUp()
        self.benchmark_results: Dict[str, Dict[str, float]] = {}
        self.performance_thresholds = self.get_performance_thresholds()
    
    def get_performance_thresholds(self) -> Dict[str, float]:
        """Get performance thresholds for assertions."""
        return {
            "max_latency_ms": 100.0,
            "min_throughput_mbps": 1.0,
            "max_connection_time_s": 5.0,
            "min_messages_per_second": 100.0
        }
    
    @contextmanager
    def benchmark(self, name: str):
        """Context manager for benchmarking operations."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self.benchmark_results[name] = {
                "duration_s": duration,
                "duration_ms": duration * 1000,
                "memory_delta_mb": memory_delta / (1024 * 1024)
            }
            
            self.logger.info(f"Benchmark {name}: {duration:.3f}s, {memory_delta/1024/1024:.2f}MB")
    
    def assert_performance(self, benchmark_name: str, threshold_key: str):
        """Assert performance meets threshold."""
        if benchmark_name not in self.benchmark_results:
            self.fail(f"No benchmark results for {benchmark_name}")
        
        if threshold_key not in self.performance_thresholds:
            self.fail(f"No threshold defined for {threshold_key}")
        
        result = self.benchmark_results[benchmark_name]
        threshold = self.performance_thresholds[threshold_key]
        
        if threshold_key.startswith("max_"):
            metric_name = threshold_key[4:]  # Remove "max_" prefix
            if metric_name in result:
                self.assertLessEqual(result[metric_name], threshold,
                                   f"Performance threshold exceeded: {metric_name}")
        elif threshold_key.startswith("min_"):
            metric_name = threshold_key[4:]  # Remove "min_" prefix  
            if metric_name in result:
                self.assertGreaterEqual(result[metric_name], threshold,
                                      f"Performance threshold not met: {metric_name}")
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0  # Skip memory tracking if psutil not available
    
    def print_benchmark_summary(self):
        """Print summary of all benchmark results."""
        if not self.benchmark_results:
            self.logger.info("No benchmark results to summarize")
            return
        
        self.logger.info("\n=== Benchmark Summary ===")
        for name, results in self.benchmark_results.items():
            self.logger.info(f"{name}:")
            for metric, value in results.items():
                self.logger.info(f"  {metric}: {value:.3f}")


# Utility functions for test cases
def create_test_case_suite(*test_classes) -> unittest.TestSuite:
    """Create test suite from multiple test case classes."""
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    return suite


def run_test_suite(suite: unittest.TestSuite, verbosity: int = 2) -> unittest.TestResult:
    """Run test suite and return results."""
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)
