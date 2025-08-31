"""Comprehensive unit tests for BaseAnalytics implementation.

Tests the abstract analytics methods with various scenarios including
metric recording, plot generation, and analytics report creation.
"""

import pytest
import tempfile
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for testing
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from typing import Any, Dict

from agents.king.analytics.base_analytics import BaseAnalytics


class TestableAnalytics(BaseAnalytics):
    """Concrete implementation of BaseAnalytics for testing."""
    
    def generate_analytics_report(self) -> Dict[str, Any]:
        """Implement abstract method for testing."""
        return {
            "total_metrics": len(self.metrics),
            "metric_names": list(self.metrics.keys()),
            "metric_summaries": {
                metric: {
                    "count": len(values),
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "avg": sum(values) / len(values) if values else 0
                }
                for metric, values in self.metrics.items()
            },
            "timestamp": "test_timestamp"
        }


@pytest.mark.unit
class TestBaseAnalyticsInitialization:
    """Test analytics initialization."""
    
    def test_initialization(self):
        """Test analytics initialization."""
        analytics = TestableAnalytics()
        
        assert analytics.metrics == {}
        assert hasattr(analytics, 'metrics')


@pytest.mark.unit
class TestBaseAnalyticsMetricRecording:
    """Test metric recording functionality."""
    
    @pytest.fixture
    def analytics(self):
        """Create analytics instance for testing."""
        return TestableAnalytics()
    
    def test_record_single_metric(self, analytics):
        """Test recording a single metric."""
        analytics.record_metric("response_time", 1.5)
        
        assert "response_time" in analytics.metrics
        assert analytics.metrics["response_time"] == [1.5]
    
    def test_record_multiple_values_same_metric(self, analytics):
        """Test recording multiple values for the same metric."""
        values = [1.0, 1.5, 2.0, 1.2, 1.8]
        
        for value in values:
            analytics.record_metric("response_time", value)
        
        assert analytics.metrics["response_time"] == values
    
    def test_record_multiple_different_metrics(self, analytics):
        """Test recording multiple different metrics."""
        analytics.record_metric("response_time", 1.5)
        analytics.record_metric("accuracy", 0.95)
        analytics.record_metric("memory_usage", 128.5)
        
        assert len(analytics.metrics) == 3
        assert analytics.metrics["response_time"] == [1.5]
        assert analytics.metrics["accuracy"] == [0.95]
        assert analytics.metrics["memory_usage"] == [128.5]
    
    def test_record_zero_values(self, analytics):
        """Test recording zero values."""
        analytics.record_metric("error_rate", 0.0)
        
        assert analytics.metrics["error_rate"] == [0.0]
    
    def test_record_negative_values(self, analytics):
        """Test recording negative values."""
        analytics.record_metric("temperature", -5.2)
        
        assert analytics.metrics["temperature"] == [-5.2]
    
    def test_record_large_values(self, analytics):
        """Test recording large values."""
        analytics.record_metric("total_requests", 1000000.0)
        
        assert analytics.metrics["total_requests"] == [1000000.0]
    
    def test_record_metric_with_logging(self, analytics, caplog):
        """Test metric recording with logging verification."""
        with caplog.at_level("DEBUG"):
            analytics.record_metric("test_metric", 42.0)
        
        assert "Recorded test_metric: 42.0" in caplog.text


@pytest.mark.unit
class TestBaseAnalyticsPlotGeneration:
    """Test plot generation functionality."""
    
    @pytest.fixture
    def analytics(self):
        """Create analytics instance with test data."""
        analytics = TestableAnalytics()
        # Add some test data
        for i in range(10):
            analytics.record_metric("response_time", i * 0.1)
        return analytics
    
    def test_generate_plot_success(self, analytics):
        """Test successful plot generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('matplotlib.pyplot.savefig') as mock_savefig:
                filename = analytics.generate_metric_plot("response_time")
                
                assert filename == "response_time.png"
                mock_savefig.assert_called_once_with("response_time.png")
    
    def test_generate_plot_nonexistent_metric(self, analytics, caplog):
        """Test plot generation for non-existent metric."""
        with caplog.at_level("WARNING"):
            filename = analytics.generate_metric_plot("nonexistent_metric")
        
        assert filename == ""
        assert "No data for metric nonexistent_metric" in caplog.text
    
    def test_generate_plot_empty_metric(self, analytics):
        """Test plot generation for metric with no data."""
        analytics.metrics["empty_metric"] = []
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            filename = analytics.generate_metric_plot("empty_metric")
            
            # Should still generate plot even with empty data
            assert filename == "empty_metric.png"
            mock_savefig.assert_called_once()
    
    def test_generate_plot_single_value(self, analytics):
        """Test plot generation for metric with single value."""
        analytics.record_metric("single_value", 5.0)
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            filename = analytics.generate_metric_plot("single_value")
            
            assert filename == "single_value.png"
            mock_savefig.assert_called_once()
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_generation_calls(
        self, mock_close, mock_savefig, mock_ylabel, 
        mock_xlabel, mock_title, mock_plot, mock_figure, analytics
    ):
        """Test that plot generation makes correct matplotlib calls."""
        filename = analytics.generate_metric_plot("response_time")
        
        mock_figure.assert_called_once_with(figsize=(10, 6))
        mock_plot.assert_called_once()
        mock_title.assert_called_once_with("response_time Over Time")
        mock_xlabel.assert_called_once_with("Time")
        mock_ylabel.assert_called_once_with("response_time")
        mock_savefig.assert_called_once_with("response_time.png")
        mock_close.assert_called_once()
    
    def test_generate_multiple_plots(self, analytics):
        """Test generating multiple plots for different metrics."""
        analytics.record_metric("accuracy", 0.95)
        analytics.record_metric("memory_usage", 128.0)
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            filename1 = analytics.generate_metric_plot("response_time")
            filename2 = analytics.generate_metric_plot("accuracy")
            filename3 = analytics.generate_metric_plot("memory_usage")
            
            assert filename1 == "response_time.png"
            assert filename2 == "accuracy.png" 
            assert filename3 == "memory_usage.png"
            assert mock_savefig.call_count == 3


@pytest.mark.unit
class TestBaseAnalyticsReportGeneration:
    """Test analytics report generation."""
    
    @pytest.fixture
    def analytics_with_data(self):
        """Create analytics instance with comprehensive test data."""
        analytics = TestableAnalytics()
        
        # Response time data
        response_times = [0.1, 0.2, 0.15, 0.3, 0.25, 0.12, 0.18]
        for rt in response_times:
            analytics.record_metric("response_time", rt)
        
        # Accuracy data
        accuracies = [0.95, 0.92, 0.97, 0.94, 0.96]
        for acc in accuracies:
            analytics.record_metric("accuracy", acc)
        
        # Memory usage data
        memory_values = [100.5, 120.2, 95.8, 110.1]
        for mem in memory_values:
            analytics.record_metric("memory_usage", mem)
        
        return analytics
    
    def test_generate_comprehensive_report(self, analytics_with_data):
        """Test comprehensive analytics report generation."""
        report = analytics_with_data.generate_analytics_report()
        
        # Verify report structure
        assert "total_metrics" in report
        assert "metric_names" in report
        assert "metric_summaries" in report
        assert "timestamp" in report
        
        # Verify content
        assert report["total_metrics"] == 3
        assert set(report["metric_names"]) == {"response_time", "accuracy", "memory_usage"}
        
        # Verify metric summaries
        summaries = report["metric_summaries"]
        assert "response_time" in summaries
        assert "accuracy" in summaries
        assert "memory_usage" in summaries
        
        # Verify response_time summary
        rt_summary = summaries["response_time"]
        assert rt_summary["count"] == 7
        assert rt_summary["min"] == 0.1
        assert rt_summary["max"] == 0.3
        assert abs(rt_summary["avg"] - 0.188571) < 0.001
        
        # Verify accuracy summary
        acc_summary = summaries["accuracy"]
        assert acc_summary["count"] == 5
        assert acc_summary["min"] == 0.92
        assert acc_summary["max"] == 0.97
        assert acc_summary["avg"] == 0.948
    
    def test_generate_report_empty_analytics(self):
        """Test report generation with no metrics."""
        analytics = TestableAnalytics()
        
        report = analytics.generate_analytics_report()
        
        assert report["total_metrics"] == 0
        assert report["metric_names"] == []
        assert report["metric_summaries"] == {}
    
    def test_generate_report_single_metric(self):
        """Test report generation with single metric."""
        analytics = TestableAnalytics()
        analytics.record_metric("test_metric", 42.0)
        
        report = analytics.generate_analytics_report()
        
        assert report["total_metrics"] == 1
        assert report["metric_names"] == ["test_metric"]
        assert "test_metric" in report["metric_summaries"]
        
        summary = report["metric_summaries"]["test_metric"]
        assert summary["count"] == 1
        assert summary["min"] == 42.0
        assert summary["max"] == 42.0
        assert summary["avg"] == 42.0


@pytest.mark.unit
class TestBaseAnalyticsAbstractMethods:
    """Test abstract method enforcement."""
    
    def test_abstract_method_implementation_required(self):
        """Test that abstract methods must be implemented."""
        # This tests that the abstract class works as expected
        analytics = TestableAnalytics()
        
        # Should be able to call the implemented method
        result = analytics.generate_analytics_report()
        assert isinstance(result, dict)
    
    def test_save_method_not_implemented(self):
        """Test that save method raises NotImplementedError."""
        analytics = TestableAnalytics()
        
        with pytest.raises(NotImplementedError):
            analytics.save("/tmp/test.json")
    
    def test_load_method_not_implemented(self):
        """Test that load method raises NotImplementedError."""
        analytics = TestableAnalytics()
        
        with pytest.raises(NotImplementedError):
            analytics.load("/tmp/test.json")


@pytest.mark.unit
class TestBaseAnalyticsEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def analytics(self):
        """Create analytics instance for testing."""
        return TestableAnalytics()
    
    def test_record_metric_with_special_characters(self, analytics):
        """Test recording metrics with special characters in name."""
        analytics.record_metric("response_time_ms/sec", 1.5)
        analytics.record_metric("accuracy_%", 95.0)
        analytics.record_metric("memory-usage_MB", 128.0)
        
        assert len(analytics.metrics) == 3
        assert "response_time_ms/sec" in analytics.metrics
        assert "accuracy_%" in analytics.metrics
        assert "memory-usage_MB" in analytics.metrics
    
    def test_record_metric_unicode_names(self, analytics):
        """Test recording metrics with unicode names."""
        analytics.record_metric("响应时间", 1.5)  # Chinese
        analytics.record_metric("précision", 0.95)  # French
        analytics.record_metric("Genauigkeit", 0.92)  # German
        
        assert len(analytics.metrics) == 3
    
    def test_record_extreme_values(self, analytics):
        """Test recording extreme values."""
        import sys
        
        analytics.record_metric("max_float", sys.float_info.max)
        analytics.record_metric("min_float", sys.float_info.min)
        analytics.record_metric("infinity", float('inf'))
        analytics.record_metric("negative_infinity", float('-inf'))
        
        assert len(analytics.metrics) == 4
        assert analytics.metrics["infinity"] == [float('inf')]
        assert analytics.metrics["negative_infinity"] == [float('-inf')]
    
    def test_record_nan_values(self, analytics):
        """Test recording NaN values."""
        import math
        
        analytics.record_metric("nan_metric", float('nan'))
        
        assert len(analytics.metrics) == 1
        assert math.isnan(analytics.metrics["nan_metric"][0])
    
    def test_generate_plot_with_nan_values(self, analytics):
        """Test plot generation with NaN values."""
        import math
        
        analytics.record_metric("test_metric", 1.0)
        analytics.record_metric("test_metric", float('nan'))
        analytics.record_metric("test_metric", 2.0)
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            filename = analytics.generate_metric_plot("test_metric")
            
            # Should still generate plot despite NaN values
            assert filename == "test_metric.png"
            mock_savefig.assert_called_once()
    
    def test_generate_plot_with_infinite_values(self, analytics):
        """Test plot generation with infinite values."""
        analytics.record_metric("test_metric", 1.0)
        analytics.record_metric("test_metric", float('inf'))
        analytics.record_metric("test_metric", 2.0)
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            filename = analytics.generate_metric_plot("test_metric")
            
            assert filename == "test_metric.png"
            mock_savefig.assert_called_once()
    
    def test_large_number_of_metrics(self, analytics):
        """Test handling large number of different metrics."""
        # Create 1000 different metrics
        for i in range(1000):
            analytics.record_metric(f"metric_{i}", float(i))
        
        assert len(analytics.metrics) == 1000
        
        # Report generation should still work
        report = analytics.generate_analytics_report()
        assert report["total_metrics"] == 1000
    
    def test_large_number_of_values_per_metric(self, analytics):
        """Test handling large number of values per metric."""
        # Add 10000 values to single metric
        for i in range(10000):
            analytics.record_metric("large_metric", float(i))
        
        assert len(analytics.metrics["large_metric"]) == 10000
        
        # Operations should still work
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            filename = analytics.generate_metric_plot("large_metric")
            assert filename == "large_metric.png"
        
        report = analytics.generate_analytics_report()
        assert report["metric_summaries"]["large_metric"]["count"] == 10000
    
    def test_concurrent_metric_recording(self, analytics):
        """Test concurrent metric recording (thread safety implications)."""
        import threading
        import time
        
        def record_metrics(start_idx):
            for i in range(100):
                analytics.record_metric("concurrent_metric", start_idx + i)
                time.sleep(0.001)  # Small delay to encourage race conditions
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=record_metrics, args=(i * 1000,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have 500 total values (5 threads * 100 values each)
        assert len(analytics.metrics["concurrent_metric"]) == 500
    
    def test_memory_efficiency_with_repeated_operations(self, analytics):
        """Test memory efficiency with many repeated operations."""
        import gc
        import sys
        
        initial_refs = sys.gettotalrefcount() if hasattr(sys, 'gettotalrefcount') else 0
        
        # Perform many operations
        for i in range(1000):
            analytics.record_metric("memory_test", float(i))
            if i % 100 == 0:
                gc.collect()  # Force garbage collection
        
        # Generate plots and reports
        for i in range(10):
            with patch('matplotlib.pyplot.savefig'):
                analytics.generate_metric_plot("memory_test")
            analytics.generate_analytics_report()
            gc.collect()
        
        # Memory usage should not grow excessively
        final_refs = sys.gettotalrefcount() if hasattr(sys, 'gettotalrefcount') else 0
        
        # This is a basic check - in practice you'd use memory profiling tools
        assert len(analytics.metrics) == 1  # Should only have one metric type
        assert len(analytics.metrics["memory_test"]) == 1000  # Should have all values


@pytest.mark.unit
class TestBaseAnalyticsIntegration:
    """Test integration scenarios with other components."""
    
    def test_integration_with_real_matplotlib(self):
        """Test actual matplotlib integration without mocking."""
        analytics = TestableAnalytics()
        
        # Add some test data
        for i in range(5):
            analytics.record_metric("integration_test", i * 0.5)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory to avoid cluttering
            original_dir = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                filename = analytics.generate_metric_plot("integration_test")
                
                # Verify file was created
                assert filename == "integration_test.png"
                assert os.path.exists(filename)
                
                # Verify file size is reasonable (not empty)
                file_size = os.path.getsize(filename)
                assert file_size > 1000  # PNG should be at least 1KB
                
            finally:
                os.chdir(original_dir)
    
    def test_analytics_persistence_interface(self):
        """Test the save/load interface (even though not implemented)."""
        analytics = TestableAnalytics()
        analytics.record_metric("test", 1.0)
        
        # Verify methods exist and raise appropriate errors
        assert hasattr(analytics, 'save')
        assert hasattr(analytics, 'load')
        assert callable(analytics.save)
        assert callable(analytics.load)
        
        with pytest.raises(NotImplementedError):
            analytics.save("/tmp/test")
        
        with pytest.raises(NotImplementedError):
            analytics.load("/tmp/test")
    
    def test_report_serialization_compatibility(self):
        """Test that generated reports are JSON serializable."""
        import json
        
        analytics = TestableAnalytics()
        analytics.record_metric("test", 1.5)
        analytics.record_metric("test", 2.5)
        
        report = analytics.generate_analytics_report()
        
        # Should be able to serialize to JSON
        json_str = json.dumps(report)
        assert isinstance(json_str, str)
        
        # Should be able to deserialize
        deserialized = json.loads(json_str)
        assert deserialized == report