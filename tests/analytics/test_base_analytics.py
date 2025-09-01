"""
Comprehensive unit tests for BaseAnalytics implementation.
Tests all methods with various scenarios including edge cases and error conditions.
"""

import unittest
import tempfile
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
import numpy as np

# Import the implementation (adjust path as needed)
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../experiments/agents/agents/king/analytics"))

from base_analytics import BaseAnalytics, AnalyticsMetadata


class ConcreteAnalytics(BaseAnalytics):
    """Concrete implementation for testing purposes."""

    pass


class TestAnalyticsMetadata(unittest.TestCase):
    """Test AnalyticsMetadata dataclass."""

    def test_default_initialization(self):
        """Test default metadata initialization."""
        metadata = AnalyticsMetadata()

        self.assertEqual(metadata.version, "1.0.0")
        self.assertEqual(metadata.data_schema, "base_analytics_v1")
        self.assertIsInstance(metadata.created_at, datetime)
        self.assertIsInstance(metadata.updated_at, datetime)
        self.assertFalse(metadata.compression)
        self.assertEqual(metadata.format_type, "json")

    def test_custom_initialization(self):
        """Test custom metadata initialization."""
        created_time = datetime(2023, 1, 1, 12, 0, 0)
        metadata = AnalyticsMetadata(version="2.0.0", created_at=created_time, compression=True, format_type="pickle")

        self.assertEqual(metadata.version, "2.0.0")
        self.assertEqual(metadata.created_at, created_time)
        self.assertTrue(metadata.compression)
        self.assertEqual(metadata.format_type, "pickle")


class TestBaseAnalytics(unittest.TestCase):
    """Test BaseAnalytics implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.analytics = ConcreteAnalytics()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(self.analytics.metrics, {})
        self.assertEqual(self.analytics.timestamps, {})
        self.assertIsInstance(self.analytics.metadata, AnalyticsMetadata)
        self.assertIsNone(self.analytics._retention_policy)
        self.assertIsNone(self.analytics._max_data_points)

    def test_record_metric_basic(self):
        """Test basic metric recording."""
        self.analytics.record_metric("test_metric", 10.5)

        self.assertEqual(len(self.analytics.metrics["test_metric"]), 1)
        self.assertEqual(self.analytics.metrics["test_metric"][0], 10.5)
        self.assertEqual(len(self.analytics.timestamps["test_metric"]), 1)
        self.assertIsInstance(self.analytics.timestamps["test_metric"][0], datetime)

    def test_record_metric_multiple_values(self):
        """Test recording multiple values for same metric."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            self.analytics.record_metric("test_metric", value)

        self.assertEqual(self.analytics.metrics["test_metric"], values)
        self.assertEqual(len(self.analytics.timestamps["test_metric"]), len(values))

    def test_record_metric_with_timestamp(self):
        """Test recording metric with custom timestamp."""
        custom_time = datetime(2023, 1, 1, 12, 0, 0)
        self.analytics.record_metric("test_metric", 42.0, custom_time)

        self.assertEqual(self.analytics.timestamps["test_metric"][0], custom_time)

    def test_retention_policy_time_based(self):
        """Test time-based retention policy."""
        # Set up retention policy for 1 hour
        self.analytics.set_retention_policy(retention_period=timedelta(hours=1))

        # Add old data (should be removed)
        old_time = datetime.now() - timedelta(hours=2)
        self.analytics.record_metric("test_metric", 1.0, old_time)

        # Add recent data (should be kept)
        recent_time = datetime.now() - timedelta(minutes=30)
        self.analytics.record_metric("test_metric", 2.0, recent_time)

        # The old data should have been filtered out
        self.assertEqual(len(self.analytics.metrics["test_metric"]), 1)
        self.assertEqual(self.analytics.metrics["test_metric"][0], 2.0)

    def test_retention_policy_max_points(self):
        """Test max data points retention policy."""
        self.analytics.set_retention_policy(max_data_points=3)

        # Add more data than the limit
        for i in range(5):
            self.analytics.record_metric("test_metric", float(i))

        # Should only keep the last 3 points
        self.assertEqual(len(self.analytics.metrics["test_metric"]), 3)
        self.assertEqual(self.analytics.metrics["test_metric"], [2.0, 3.0, 4.0])

    def test_generate_analytics_report_empty_data(self):
        """Test report generation with empty data."""
        report = self.analytics.generate_analytics_report()

        self.assertIn("metadata", report)
        self.assertIn("metrics", report)
        self.assertIn("summary", report)
        self.assertEqual(report["metadata"]["total_metrics"], 0)
        self.assertEqual(report["metadata"]["data_points"], 0)

    def test_generate_analytics_report_basic(self):
        """Test basic report generation."""
        # Add some test data
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            self.analytics.record_metric("test_metric", value)

        report = self.analytics.generate_analytics_report()

        self.assertEqual(report["metadata"]["total_metrics"], 1)
        self.assertEqual(report["metadata"]["data_points"], 5)
        self.assertIn("test_metric", report["metrics"])

        metric_data = report["metrics"]["test_metric"]
        self.assertIn("statistics", metric_data)
        self.assertIn("trends", metric_data)
        self.assertIn("quality", metric_data)

    def test_generate_analytics_report_formats(self):
        """Test different report formats."""
        # Add test data
        for i in range(10):
            self.analytics.record_metric("test_metric", float(i))

        # Test summary format
        summary_report = self.analytics.generate_analytics_report(report_format="summary")
        self.assertIn("count", summary_report["metrics"]["test_metric"])
        self.assertIn("mean", summary_report["metrics"]["test_metric"])
        self.assertNotIn("statistics", summary_report["metrics"]["test_metric"])

        # Test detailed format
        detailed_report = self.analytics.generate_analytics_report(report_format="detailed")
        metric_data = detailed_report["metrics"]["test_metric"]
        self.assertIn("statistics", metric_data)
        self.assertIn("trends", metric_data)
        self.assertIn("quality", metric_data)

    def test_generate_analytics_report_time_window(self):
        """Test report generation with time window."""
        # Add old data
        old_time = datetime.now() - timedelta(hours=2)
        self.analytics.record_metric("test_metric", 1.0, old_time)

        # Add recent data
        recent_time = datetime.now() - timedelta(minutes=30)
        self.analytics.record_metric("test_metric", 2.0, recent_time)

        # Generate report with 1-hour window
        report = self.analytics.generate_analytics_report(time_window=timedelta(hours=1))

        # Should only include recent data
        self.assertEqual(report["metadata"]["data_points"], 1)

    def test_statistical_analysis(self):
        """Test statistical analysis accuracy."""
        # Use known data for verification
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for value in values:
            self.analytics.record_metric("test_metric", float(value))

        analysis = self.analytics._analyze_metric("test_metric", values, [])
        stats = analysis["statistics"]

        self.assertEqual(stats["count"], 10)
        self.assertEqual(stats["mean"], 5.5)
        self.assertEqual(stats["median"], 5.5)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 10.0)
        self.assertEqual(stats["range"], 9.0)
        self.assertAlmostEqual(stats["std_dev"], np.std(values, ddof=1), places=5)

    def test_trend_analysis(self):
        """Test trend analysis."""
        # Increasing trend
        increasing_values = [1, 2, 3, 4, 5]
        for value in increasing_values:
            self.analytics.record_metric("increasing_metric", float(value))

        analysis = self.analytics._analyze_metric("increasing_metric", increasing_values, [])
        self.assertEqual(analysis["trends"]["linear_trend"], "increasing")
        self.assertGreater(analysis["trends"]["slope"], 0)

        # Decreasing trend
        decreasing_values = [5, 4, 3, 2, 1]
        analysis = self.analytics._analyze_metric("decreasing_metric", decreasing_values, [])
        self.assertEqual(analysis["trends"]["linear_trend"], "decreasing")
        self.assertLess(analysis["trends"]["slope"], 0)

    def test_outlier_detection(self):
        """Test outlier detection."""
        # Normal data with outliers
        values = [1, 2, 3, 4, 5, 100, 6, 7, 8, 9]  # 100 is an outlier
        outliers = self.analytics._detect_outliers(values)

        self.assertGreater(outliers["count"], 0)
        self.assertIn(5, outliers["indices"])  # Index of outlier value 100

    def test_save_json_format(self):
        """Test saving in JSON format."""
        # Add test data
        self.analytics.record_metric("test_metric", 42.0)

        file_path = Path(self.test_dir) / "test_analytics.json"
        success = self.analytics.save(str(file_path), format_type="json")

        self.assertTrue(success)
        self.assertTrue(file_path.exists())

        # Verify content
        with open(file_path, "r") as f:
            data = json.load(f)

        self.assertIn("metadata", data)
        self.assertIn("metrics", data)
        self.assertEqual(data["metrics"]["test_metric"], [42.0])

    def test_save_json_compressed(self):
        """Test saving compressed JSON format."""
        self.analytics.record_metric("test_metric", 42.0)

        file_path = Path(self.test_dir) / "test_analytics.json"
        success = self.analytics.save(str(file_path), format_type="json", compress=True)

        self.assertTrue(success)
        # Compressed file should have .gz extension
        compressed_path = file_path.with_suffix(".json.gz")
        self.assertTrue(compressed_path.exists())

    def test_save_pickle_format(self):
        """Test saving in pickle format."""
        self.analytics.record_metric("test_metric", 42.0)

        file_path = Path(self.test_dir) / "test_analytics.pkl"
        success = self.analytics.save(str(file_path), format_type="pickle")

        self.assertTrue(success)
        self.assertTrue(file_path.exists())

    def test_save_sqlite_format(self):
        """Test saving in SQLite format."""
        self.analytics.record_metric("test_metric", 42.0)

        file_path = Path(self.test_dir) / "test_analytics.db"
        success = self.analytics.save(str(file_path), format_type="sqlite")

        self.assertTrue(success)
        self.assertTrue(file_path.exists())

        # Verify SQLite content
        with sqlite3.connect(file_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM metrics")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)

    def test_save_auto_format_detection(self):
        """Test automatic format detection from file extension."""
        self.analytics.record_metric("test_metric", 42.0)

        # Test JSON auto-detection
        json_path = Path(self.test_dir) / "test.json"
        success = self.analytics.save(str(json_path), format_type="auto")
        self.assertTrue(success)

        # Test pickle auto-detection
        pkl_path = Path(self.test_dir) / "test.pkl"
        success = self.analytics.save(str(pkl_path), format_type="auto")
        self.assertTrue(success)

        # Test SQLite auto-detection
        db_path = Path(self.test_dir) / "test.db"
        success = self.analytics.save(str(db_path), format_type="auto")
        self.assertTrue(success)

    def test_save_with_backup(self):
        """Test backup creation during save."""
        # Create initial file
        file_path = Path(self.test_dir) / "test_analytics.json"
        with open(file_path, "w") as f:
            json.dump({"old": "data"}, f)

        self.analytics.record_metric("test_metric", 42.0)
        success = self.analytics.save(str(file_path), create_backup=True)

        self.assertTrue(success)

        # Check that backup was created
        backup_files = list(Path(self.test_dir).glob("*.backup.*"))
        self.assertGreater(len(backup_files), 0)

    def test_load_json_format(self):
        """Test loading JSON format."""
        # First save some data
        self.analytics.record_metric("test_metric", 42.0)
        file_path = Path(self.test_dir) / "test_analytics.json"
        self.analytics.save(str(file_path))

        # Create new instance and load
        new_analytics = ConcreteAnalytics()
        success = new_analytics.load(str(file_path))

        self.assertTrue(success)
        self.assertEqual(new_analytics.metrics["test_metric"], [42.0])
        self.assertEqual(len(new_analytics.timestamps["test_metric"]), 1)

    def test_load_pickle_format(self):
        """Test loading pickle format."""
        # Save in pickle format
        self.analytics.record_metric("test_metric", 42.0)
        file_path = Path(self.test_dir) / "test_analytics.pkl"
        self.analytics.save(str(file_path), format_type="pickle")

        # Load in new instance
        new_analytics = ConcreteAnalytics()
        success = new_analytics.load(str(file_path))

        self.assertTrue(success)
        self.assertEqual(new_analytics.metrics["test_metric"], [42.0])

    def test_load_sqlite_format(self):
        """Test loading SQLite format."""
        # Save in SQLite format
        self.analytics.record_metric("test_metric", 42.0)
        file_path = Path(self.test_dir) / "test_analytics.db"
        self.analytics.save(str(file_path), format_type="sqlite")

        # Load in new instance
        new_analytics = ConcreteAnalytics()
        success = new_analytics.load(str(file_path))

        self.assertTrue(success)
        self.assertEqual(new_analytics.metrics["test_metric"], [42.0])

    def test_load_compressed_files(self):
        """Test loading compressed files."""
        # Save compressed
        self.analytics.record_metric("test_metric", 42.0)
        file_path = Path(self.test_dir) / "test_analytics.json"
        self.analytics.save(str(file_path), compress=True)

        # Load compressed file
        new_analytics = ConcreteAnalytics()
        compressed_path = file_path.with_suffix(".json.gz")
        success = new_analytics.load(str(compressed_path))

        self.assertTrue(success)
        self.assertEqual(new_analytics.metrics["test_metric"], [42.0])

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file."""
        success = self.analytics.load("/nonexistent/path/file.json")
        self.assertFalse(success)

    def test_load_corrupted_file(self):
        """Test loading corrupted file."""
        # Create corrupted JSON file
        file_path = Path(self.test_dir) / "corrupted.json"
        with open(file_path, "w") as f:
            f.write("{ invalid json")

        success = self.analytics.load(str(file_path))
        self.assertFalse(success)

    def test_load_with_schema_validation(self):
        """Test loading with schema validation."""
        # Create file with invalid schema
        file_path = Path(self.test_dir) / "invalid_schema.json"
        with open(file_path, "w") as f:
            json.dump({"invalid": "schema"}, f)

        success = self.analytics.load(str(file_path), validate_schema=True)
        self.assertFalse(success)

    def test_load_fallback_detection(self):
        """Test fallback format detection."""
        # Create file with unknown extension but JSON content
        self.analytics.record_metric("test_metric", 42.0)
        file_path = Path(self.test_dir) / "test_analytics.unknown"

        # Manually create JSON content with unknown extension
        data = {
            "metadata": {},
            "metrics": {"test_metric": [42.0]},
            "timestamps": {"test_metric": [datetime.now().isoformat()]},
        }
        with open(file_path, "w") as f:
            json.dump(data, f)

        # Should still load via fallback
        new_analytics = ConcreteAnalytics()
        success = new_analytics.load(str(file_path))
        self.assertTrue(success)

    @patch("experiments.agents.agents.king.analytics.base_analytics.logger")
    def test_error_handling(self, mock_logger):
        """Test error handling and logging."""
        # Test save error handling
        success = self.analytics.save("/invalid/path/file.json")
        self.assertFalse(success)
        mock_logger.error.assert_called()

        # Test load error handling
        success = self.analytics.load("/invalid/path/file.json")
        self.assertFalse(success)

    def test_consistency_assessment(self):
        """Test data consistency assessment."""
        # Test consistent data
        consistent_values = [5.0] * 10
        consistency = self.analytics._assess_consistency(consistent_values)
        self.assertAlmostEqual(consistency, 1.0, places=2)

        # Test inconsistent data
        inconsistent_values = [1, 100, 2, 200, 3]
        consistency = self.analytics._assess_consistency(inconsistent_values)
        self.assertLess(consistency, 0.5)

    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with large datasets."""
        # Add large amount of data
        for i in range(10000):
            self.analytics.record_metric("large_metric", float(i))

        # Should handle large datasets without issues
        report = self.analytics.generate_analytics_report()
        self.assertEqual(report["metadata"]["data_points"], 10000)

        # Test with retention policy to manage memory
        self.analytics.set_retention_policy(max_data_points=1000)
        self.analytics.record_metric("large_metric", 10000.0)  # Trigger cleanup

        self.assertLessEqual(len(self.analytics.metrics["large_metric"]), 1000)


class TestIntegrationScenarios(unittest.TestCase):
    """Test realistic integration scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.analytics = ConcreteAnalytics()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_ml_training_scenario(self):
        """Test ML model training analytics scenario."""
        # Simulate training metrics
        epochs = 100
        for epoch in range(epochs):
            # Simulate improving metrics
            loss = 1.0 / (epoch + 1) + np.random.normal(0, 0.01)
            accuracy = 1 - loss + np.random.normal(0, 0.005)

            self.analytics.record_metric("training_loss", loss)
            self.analytics.record_metric("training_accuracy", accuracy)

        # Generate comprehensive report
        report = self.analytics.generate_analytics_report(report_format="detailed")

        # Verify trend analysis captures improvement
        loss_trends = report["metrics"]["training_loss"]["trends"]
        accuracy_trends = report["metrics"]["training_accuracy"]["trends"]

        self.assertEqual(loss_trends["linear_trend"], "decreasing")
        self.assertEqual(accuracy_trends["linear_trend"], "increasing")

    def test_system_monitoring_scenario(self):
        """Test system performance monitoring scenario."""
        # Simulate system metrics over time

        base_time = datetime.now() - timedelta(hours=24)

        for hour in range(24):
            timestamp = base_time + timedelta(hours=hour)

            # Simulate daily patterns
            cpu_usage = 50 + 20 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 5)
            memory_usage = 60 + 15 * np.cos(2 * np.pi * hour / 24) + np.random.normal(0, 3)

            self.analytics.record_metric("cpu_usage", max(0, min(100, cpu_usage)), timestamp)
            self.analytics.record_metric("memory_usage", max(0, min(100, memory_usage)), timestamp)

        # Test time-windowed analysis
        recent_report = self.analytics.generate_analytics_report(time_window=timedelta(hours=6), include_trends=True)

        self.assertGreater(recent_report["metadata"]["data_points"], 0)

        # Test data persistence and recovery
        file_path = Path(self.test_dir) / "system_metrics.json"
        self.assertTrue(self.analytics.save(str(file_path)))

        # Simulate system restart - load previous data
        new_analytics = ConcreteAnalytics()
        self.assertTrue(new_analytics.load(str(file_path)))

        # Verify data integrity
        self.assertEqual(len(new_analytics.metrics), len(self.analytics.metrics))

    def test_performance_benchmarking_scenario(self):
        """Test performance benchmarking with different formats."""
        # Generate substantial dataset
        metrics = ["latency", "throughput", "error_rate", "memory_usage"]

        for _ in range(1000):
            for metric in metrics:
                value = np.random.exponential(scale=10.0)  # Realistic performance distribution
                self.analytics.record_metric(metric, value)

        # Test performance with different save/load formats
        formats = ["json", "pickle", "sqlite"]

        for fmt in formats:
            file_path = Path(self.test_dir) / f"benchmark_{fmt}.{fmt if fmt != 'sqlite' else 'db'}"

            # Measure save performance
            start_time = datetime.now()
            success = self.analytics.save(str(file_path), format_type=fmt)
            save_duration = (datetime.now() - start_time).total_seconds()

            self.assertTrue(success)
            self.assertLess(save_duration, 5.0)  # Should save within 5 seconds

            # Measure load performance
            new_analytics = ConcreteAnalytics()
            start_time = datetime.now()
            success = new_analytics.load(str(file_path))
            load_duration = (datetime.now() - start_time).total_seconds()

            self.assertTrue(success)
            self.assertLess(load_duration, 5.0)  # Should load within 5 seconds

            # Verify data integrity
            self.assertEqual(len(new_analytics.metrics), len(self.analytics.metrics))


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add all test cases
    for test_class in [TestAnalyticsMetadata, TestBaseAnalytics, TestIntegrationScenarios]:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
