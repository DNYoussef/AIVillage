"""
Performance benchmarks and stress tests for BaseAnalytics.
Tests scalability, memory efficiency, and performance under load.
"""

import unittest
import time
import tempfile
import psutil
import gc
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Import the implementation
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../experiments/agents/agents/king/analytics'))

from base_analytics import BaseAnalytics


class ConcreteAnalytics(BaseAnalytics):
    """Concrete implementation for testing."""
    pass


class PerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests."""
    
    def setUp(self):
        """Set up test environment."""
        self.analytics = ConcreteAnalytics()
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def measure_performance(self, func, *args, **kwargs):
        """Measure execution time and memory usage."""
        # Measure initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Measure execution time
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Measure final memory
        final_memory = process.memory_info().rss
        memory_delta = final_memory - initial_memory
        
        return {
            'result': result,
            'execution_time': end_time - start_time,
            'memory_delta': memory_delta,
            'initial_memory': initial_memory,
            'final_memory': final_memory
        }
    
    def test_large_dataset_ingestion(self):
        """Test ingesting large amounts of data."""
        print(f"\\n{'='*50}")
        print("Testing large dataset ingestion performance")
        print(f"{'='*50}")
        
        # Test with different dataset sizes
        dataset_sizes = [1000, 10000, 100000]
        
        for size in dataset_sizes:
            print(f"\\nTesting with {size:,} data points...")
            
            def ingest_data():
                for i in range(size):
                    self.analytics.record_metric("performance_test", float(i))
                return len(self.analytics.metrics["performance_test"])
            
            perf = self.measure_performance(ingest_data)
            
            print(f"  Execution time: {perf['execution_time']:.4f}s")
            print(f"  Memory delta: {perf['memory_delta'] / 1024 / 1024:.2f} MB")
            print(f"  Throughput: {size / perf['execution_time']:.0f} records/second")
            
            # Performance assertions
            self.assertLess(perf['execution_time'], size * 0.001)  # Should be faster than 1ms per record
            self.assertEqual(perf['result'], size)
            
            # Clean up for next iteration
            self.analytics = ConcreteAnalytics()
            gc.collect()
    
    def test_report_generation_performance(self):
        """Test report generation performance with varying data sizes."""
        print(f"\\n{'='*50}")
        print("Testing report generation performance")
        print(f"{'='*50}")
        
        # Prepare datasets
        data_sizes = [1000, 10000, 50000]
        metric_counts = [1, 10, 50]
        
        for data_size in data_sizes:
            for metric_count in metric_counts:
                print(f"\\nTesting {metric_count} metrics with {data_size:,} points each...")
                
                # Setup data
                analytics = ConcreteAnalytics()
                for metric_idx in range(metric_count):
                    metric_name = f"metric_{metric_idx}"
                    for i in range(data_size):
                        analytics.record_metric(metric_name, np.random.normal(100, 15))
                
                # Test different report formats
                formats = ["summary", "json", "detailed"]
                for report_format in formats:
                    def generate_report():
                        return analytics.generate_analytics_report(
                            report_format=report_format,
                            include_trends=True
                        )
                    
                    perf = self.measure_performance(generate_report)
                    report = perf['result']
                    
                    print(f"  {report_format:8s}: {perf['execution_time']:.4f}s, "
                          f"Memory: {perf['memory_delta'] / 1024:.0f} KB")
                    
                    # Verify report correctness
                    self.assertEqual(report['metadata']['total_metrics'], metric_count)
                    self.assertEqual(report['metadata']['data_points'], data_size * metric_count)
                    
                    # Performance assertion - should generate report within reasonable time
                    max_time = (data_size * metric_count) / 10000  # 10k points per second baseline
                    self.assertLess(perf['execution_time'], max_time)
    
    def test_save_load_performance(self):
        """Test save/load performance with different formats."""
        print(f"\\n{'='*50}")
        print("Testing save/load performance")
        print(f"{'='*50}")
        
        # Prepare substantial dataset
        data_points = 50000
        metric_count = 10
        
        analytics = ConcreteAnalytics()
        for metric_idx in range(metric_count):
            metric_name = f"test_metric_{metric_idx}"
            for i in range(data_points // metric_count):
                analytics.record_metric(metric_name, np.random.exponential(10.0))
        
        formats = [
            ("json", "json"),
            ("json_compressed", "json"),
            ("pickle", "pkl"),
            ("pickle_compressed", "pkl"),
            ("sqlite", "db")
        ]
        
        for format_name, extension in formats:
            print(f"\\nTesting {format_name} format...")
            
            file_path = Path(self.test_dir) / f"benchmark_{format_name}.{extension}"
            compress = "compressed" in format_name
            format_type = format_name.split("_")[0]
            
            # Test save performance
            def save_data():
                return analytics.save(str(file_path), format_type=format_type, compress=compress)
            
            save_perf = self.measure_performance(save_data)
            file_size = file_path.stat().st_size if save_perf['result'] else 0
            if compress and format_type == "json":
                file_path = file_path.with_suffix(f".{extension}.gz")
                file_size = file_path.stat().st_size if file_path.exists() else 0
            
            print(f"  Save time: {save_perf['execution_time']:.4f}s")
            print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
            print(f"  Save throughput: {data_points / save_perf['execution_time']:.0f} records/second")
            
            self.assertTrue(save_perf['result'])
            
            # Test load performance
            new_analytics = ConcreteAnalytics()
            
            def load_data():
                return new_analytics.load(str(file_path))
            
            load_perf = self.measure_performance(load_data)
            
            print(f"  Load time: {load_perf['execution_time']:.4f}s")
            print(f"  Load throughput: {data_points / load_perf['execution_time']:.0f} records/second")
            
            self.assertTrue(load_perf['result'])
            self.assertEqual(len(new_analytics.metrics), len(analytics.metrics))
    
    def test_concurrent_access_performance(self):
        """Test performance under concurrent access."""
        print(f"\\n{'='*50}")
        print("Testing concurrent access performance")
        print(f"{'='*50}")
        
        # Shared analytics instance
        analytics = ConcreteAnalytics()
        
        def record_metrics_worker(worker_id, num_records):
            """Worker function for recording metrics."""
            for i in range(num_records):
                metric_name = f"worker_{worker_id}_metric"
                value = np.random.normal(worker_id * 10, 5)
                analytics.record_metric(metric_name, value)
            return num_records
        
        # Test with different numbers of concurrent workers
        worker_configs = [
            (1, 10000),   # 1 worker, 10k records
            (4, 2500),    # 4 workers, 2.5k records each
            (8, 1250),    # 8 workers, 1.25k records each
            (16, 625),    # 16 workers, 625 records each
        ]
        
        for num_workers, records_per_worker in worker_configs:
            print(f"\\nTesting {num_workers} concurrent workers, {records_per_worker} records each...")
            
            analytics = ConcreteAnalytics()  # Fresh instance
            
            def run_concurrent_test():
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(record_metrics_worker, i, records_per_worker)
                        for i in range(num_workers)
                    ]
                    
                    results = [future.result() for future in futures]
                    return sum(results)
            
            perf = self.measure_performance(run_concurrent_test)
            total_records = perf['result']
            
            print(f"  Total records: {total_records:,}")
            print(f"  Execution time: {perf['execution_time']:.4f}s")
            print(f"  Throughput: {total_records / perf['execution_time']:.0f} records/second")
            print(f"  Memory usage: {perf['memory_delta'] / 1024 / 1024:.2f} MB")
            
            self.assertEqual(total_records, num_workers * records_per_worker)
            self.assertEqual(len(analytics.metrics), num_workers)
    
    def test_memory_efficiency_with_retention(self):
        """Test memory efficiency with retention policies."""
        print(f"\\n{'='*50}")
        print("Testing memory efficiency with retention policies")
        print(f"{'='*50}")
        
        # Test without retention policy
        analytics_no_retention = ConcreteAnalytics()
        
        print("\\nTesting without retention policy...")
        
        def record_large_dataset_no_retention():
            for i in range(100000):
                analytics_no_retention.record_metric("test_metric", float(i))
            return len(analytics_no_retention.metrics["test_metric"])
        
        perf_no_retention = self.measure_performance(record_large_dataset_no_retention)
        
        print(f"  Final dataset size: {perf_no_retention['result']:,}")
        print(f"  Memory usage: {perf_no_retention['memory_delta'] / 1024 / 1024:.2f} MB")
        
        # Test with max points retention policy
        analytics_with_retention = ConcreteAnalytics()
        analytics_with_retention.set_retention_policy(max_data_points=10000)
        
        print("\\nTesting with max_data_points=10,000...")
        
        def record_large_dataset_with_retention():
            for i in range(100000):
                analytics_with_retention.record_metric("test_metric", float(i))
            return len(analytics_with_retention.metrics["test_metric"])
        
        perf_with_retention = self.measure_performance(record_large_dataset_with_retention)
        
        print(f"  Final dataset size: {perf_with_retention['result']:,}")
        print(f"  Memory usage: {perf_with_retention['memory_delta'] / 1024 / 1024:.2f} MB")
        print(f"  Memory reduction: {((perf_no_retention['memory_delta'] - perf_with_retention['memory_delta']) / perf_no_retention['memory_delta'] * 100):.1f}%")
        
        # Verify retention policy effectiveness
        self.assertEqual(perf_with_retention['result'], 10000)
        self.assertLess(perf_with_retention['memory_delta'], perf_no_retention['memory_delta'])
    
    def test_time_window_analysis_performance(self):
        """Test performance of time-window based analysis."""
        print(f"\\n{'='*50}")
        print("Testing time-window analysis performance")
        print(f"{'='*50}")
        
        # Create analytics with time-series data
        analytics = ConcreteAnalytics()
        
        # Generate data spanning 30 days
        start_time = datetime.now() - timedelta(days=30)
        total_points = 43200  # 30 days * 24 hours * 60 minutes = one point per minute
        
        print(f"Generating {total_points:,} time-series data points...")
        
        for i in range(total_points):
            timestamp = start_time + timedelta(minutes=i)
            value = 100 + 50 * np.sin(2 * np.pi * i / 1440) + np.random.normal(0, 10)  # Daily pattern
            analytics.record_metric("time_series_metric", value, timestamp)
        
        # Test analysis with different time windows
        time_windows = [
            ("1 hour", timedelta(hours=1)),
            ("1 day", timedelta(days=1)),
            ("1 week", timedelta(weeks=1)),
            ("All data", None)
        ]
        
        for window_name, window in time_windows:
            print(f"\\nAnalyzing {window_name}...")
            
            def analyze_time_window():
                return analytics.generate_analytics_report(
                    time_window=window,
                    include_trends=True
                )
            
            perf = self.measure_performance(analyze_time_window)
            report = perf['result']
            
            data_points = report['metadata']['data_points']
            print(f"  Data points analyzed: {data_points:,}")
            print(f"  Analysis time: {perf['execution_time']:.4f}s")
            print(f"  Analysis rate: {data_points / perf['execution_time']:.0f} points/second")
            
            # Performance should be reasonable even for large datasets
            self.assertLess(perf['execution_time'], 5.0)  # Should complete within 5 seconds
    
    def test_stress_test_extreme_scenarios(self):
        """Stress test with extreme scenarios."""
        print(f"\\n{'='*50}")
        print("Running stress tests with extreme scenarios")
        print(f"{'='*50}")
        
        # Test 1: Very large metric names
        print("\\nTest 1: Very large metric names...")
        analytics = ConcreteAnalytics()
        
        large_metric_name = "x" * 10000  # 10KB metric name
        
        def test_large_metric_names():
            for i in range(1000):
                analytics.record_metric(f"{large_metric_name}_{i}", float(i))
            return len(analytics.metrics)
        
        perf = self.measure_performance(test_large_metric_names)
        print(f"  Created {perf['result']:,} metrics with large names")
        print(f"  Time: {perf['execution_time']:.4f}s")
        print(f"  Memory: {perf['memory_delta'] / 1024 / 1024:.2f} MB")
        
        # Test 2: Extreme values
        print("\\nTest 2: Extreme numeric values...")
        analytics = ConcreteAnalytics()
        
        extreme_values = [
            float('inf'), float('-inf'), 1e308, -1e308,
            1e-308, -1e-308, 0.0, -0.0
        ]
        
        def test_extreme_values():
            for i, value in enumerate(extreme_values * 1000):
                try:
                    analytics.record_metric("extreme_metric", value)
                except:
                    pass  # Some extreme values might not be recordable
            return len(analytics.metrics.get("extreme_metric", []))
        
        perf = self.measure_performance(test_extreme_values)
        print(f"  Recorded {perf['result']:,} extreme values")
        print(f"  Time: {perf['execution_time']:.4f}s")
        
        # Test 3: Many metrics with few points each
        print("\\nTest 3: Many metrics with few data points each...")
        analytics = ConcreteAnalytics()
        
        def test_many_sparse_metrics():
            for i in range(10000):
                analytics.record_metric(f"sparse_metric_{i}", float(i % 100))
            return len(analytics.metrics)
        
        perf = self.measure_performance(test_many_sparse_metrics)
        print(f"  Created {perf['result']:,} sparse metrics")
        print(f"  Time: {perf['execution_time']:.4f}s")
        print(f"  Memory: {perf['memory_delta'] / 1024 / 1024:.2f} MB")
        
        # Generate report for sparse metrics
        def generate_sparse_report():
            return analytics.generate_analytics_report()
        
        report_perf = self.measure_performance(generate_sparse_report)
        print(f"  Report generation time: {report_perf['execution_time']:.4f}s")


class StressTests(unittest.TestCase):
    """Stress tests for reliability under extreme conditions."""
    
    def setUp(self):
        """Set up stress test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up stress test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_memory_pressure_resilience(self):
        """Test resilience under memory pressure."""
        print(f"\\n{'='*50}")
        print("Testing resilience under memory pressure")
        print(f"{'='*50}")
        
        analytics = ConcreteAnalytics()
        
        # Set aggressive retention policy
        analytics.set_retention_policy(max_data_points=1000)
        
        # Keep adding data and verify system remains stable
        for batch in range(100):  # 100 batches
            print(f"\\rProcessing batch {batch + 1}/100...", end="", flush=True)
            
            # Add a batch of data
            for i in range(10000):  # 10k points per batch = 1M total points
                analytics.record_metric("stress_metric", np.random.normal(0, 1))
            
            # Verify retention policy is working
            self.assertLessEqual(len(analytics.metrics["stress_metric"]), 1000)
            
            # Generate report to test analysis under pressure
            report = analytics.generate_analytics_report()
            self.assertIsInstance(report, dict)
            self.assertIn("metadata", report)
        
        print("\\nCompleted memory pressure test successfully!")
    
    def test_file_system_stress(self):
        """Test file system operations under stress."""
        print(f"\\n{'='*50}")
        print("Testing file system operations under stress")
        print(f"{'='*50}")
        
        analytics = ConcreteAnalytics()
        
        # Add substantial data
        for i in range(10000):
            analytics.record_metric("fs_stress_metric", np.random.exponential(5.0))
        
        formats = ["json", "pickle", "sqlite"]
        
        for format_type in formats:
            print(f"\\nStress testing {format_type} format...")
            
            # Rapid save/load cycles
            for cycle in range(20):
                file_path = Path(self.test_dir) / f"stress_{format_type}_{cycle}.{format_type if format_type != 'sqlite' else 'db'}"
                
                # Save
                success = analytics.save(str(file_path), format_type=format_type)
                self.assertTrue(success, f"Save failed on cycle {cycle}")
                
                # Load into new instance
                new_analytics = ConcreteAnalytics()
                success = new_analytics.load(str(file_path))
                self.assertTrue(success, f"Load failed on cycle {cycle}")
                
                # Verify data integrity
                self.assertEqual(
                    len(new_analytics.metrics["fs_stress_metric"]),
                    len(analytics.metrics["fs_stress_metric"]),
                    f"Data integrity check failed on cycle {cycle}"
                )
            
            print(f"  Completed {20} save/load cycles successfully")
    
    def test_error_recovery(self):
        """Test error recovery and graceful degradation."""
        print(f"\\n{'='*50}")
        print("Testing error recovery and graceful degradation")
        print(f"{'='*50}")
        
        analytics = ConcreteAnalytics()
        
        # Test recovery from various error conditions
        error_scenarios = [
            ("Invalid file path", "/invalid/path/test.json"),
            ("Permission denied", "/root/test.json"),  # Assuming no root access
            ("Disk full simulation", ""),  # We'll mock this
        ]
        
        for scenario_name, path in error_scenarios[:2]:  # Skip disk full for now
            print(f"\\nTesting {scenario_name}...")
            
            # Add some data
            analytics.record_metric("error_test", 42.0)
            
            # Attempt save (should fail gracefully)
            success = analytics.save(path)
            self.assertFalse(success, f"Save should have failed for {scenario_name}")
            
            # System should remain functional
            analytics.record_metric("error_test", 43.0)
            report = analytics.generate_analytics_report()
            self.assertIsInstance(report, dict)
            self.assertIn("metadata", report)
            
            print(f"  System remained functional after {scenario_name}")


if __name__ == '__main__':
    print("="*70)
    print("BASEANALYTICS PERFORMANCE BENCHMARK SUITE")
    print("="*70)
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    
    try:
        import psutil
        print(f"System memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        print(f"Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")
        print(f"CPU count: {psutil.cpu_count()}")
    except ImportError:
        print("psutil not available - memory monitoring disabled")
    
    print("="*70)
    
    # Run performance benchmarks
    unittest.main(verbosity=2)