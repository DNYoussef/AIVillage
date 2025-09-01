"""Base Analytics Implementation for Infrastructure Shared Components.

This module provides the base analytics implementation for shared experimental components,
ensuring consistency with the main analytics system while supporting infrastructure needs.
"""

from abc import ABC, abstractmethod
import json
import logging
import pickle
import gzip
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import sqlite3
from dataclasses import dataclass, asdict
import numpy as np
from statistics import mean, median, stdev
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsMetadata:
    """Metadata for analytics data validation and versioning."""
    version: str = "1.0.0"
    created_at: datetime = None
    updated_at: datetime = None
    data_schema: str = "infrastructure_analytics_v1"
    compression: bool = False
    format_type: str = "json"
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


class BaseAnalytics(ABC):
    """Base analytics class for infrastructure components."""
    
    def __init__(self) -> None:
        self.metrics: Dict[str, List[float]] = {}
        self.timestamps: Dict[str, List[datetime]] = {}
        self.metadata: AnalyticsMetadata = AnalyticsMetadata()
        self._retention_policy: Optional[timedelta] = None
        self._max_data_points: Optional[int] = None

    def record_metric(self, metric: str, value: float, timestamp: Optional[datetime] = None) -> None:
        """Record a metric value with timestamp and apply retention policies."""
        if timestamp is None:
            timestamp = datetime.now()
            
        if metric not in self.metrics:
            self.metrics[metric] = []
            self.timestamps[metric] = []
            
        self.metrics[metric].append(value)
        self.timestamps[metric].append(timestamp)
        self.metadata.updated_at = datetime.now()
        
        # Apply retention policies
        self._apply_retention_policy(metric)
        
        logger.debug(f"Recorded {metric}: {value} at {timestamp}")
    
    def _apply_retention_policy(self, metric: str) -> None:
        """Apply configured retention policies to the metric data."""
        if self._retention_policy:
            cutoff_time = datetime.now() - self._retention_policy
            indices_to_keep = [
                i for i, ts in enumerate(self.timestamps[metric])
                if ts > cutoff_time
            ]
            self.metrics[metric] = [self.metrics[metric][i] for i in indices_to_keep]
            self.timestamps[metric] = [self.timestamps[metric][i] for i in indices_to_keep]
            
        if self._max_data_points and len(self.metrics[metric]) > self._max_data_points:
            excess = len(self.metrics[metric]) - self._max_data_points
            self.metrics[metric] = self.metrics[metric][excess:]
            self.timestamps[metric] = self.timestamps[metric][excess:]
    
    def set_retention_policy(self, retention_period: Optional[timedelta] = None, 
                           max_data_points: Optional[int] = None) -> None:
        """Configure data retention policies."""
        self._retention_policy = retention_period
        self._max_data_points = max_data_points
        
        # Apply to existing data
        for metric in self.metrics.keys():
            self._apply_retention_policy(metric)

    def generate_metric_plot(self, metric: str) -> str:
        """Generate plot for a metric."""
        if metric not in self.metrics:
            logger.warning(f"No data for metric {metric}")
            return ""
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics[metric])
        plt.title(f"{metric} Over Time")
        plt.xlabel("Time")
        plt.ylabel(metric)
        filename = f"{metric}.png"
        plt.savefig(filename)
        plt.close()
        return filename

    def generate_analytics_report(self, report_format: str = "json", 
                                include_trends: bool = True,
                                time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Generate comprehensive analytics report with multiple format support.
        
        Args:
            report_format: Output format ('json', 'summary', 'detailed')
            include_trends: Whether to include trend analysis
            time_window: Time window for analysis (None for all data)
            
        Returns:
            Dictionary containing structured analytics data
        """
        try:
            # Filter data by time window if specified
            filtered_metrics = self._filter_by_time_window(time_window) if time_window else self.metrics
            filtered_timestamps = self._filter_timestamps_by_window(time_window) if time_window else self.timestamps
            
            # Generate base report structure
            report = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "report_format": report_format,
                    "time_window": str(time_window) if time_window else "all",
                    "total_metrics": len(filtered_metrics),
                    "data_points": sum(len(values) for values in filtered_metrics.values())
                },
                "metrics": {},
                "summary": {}
            }
            
            # Process each metric
            for metric_name, values in filtered_metrics.items():
                if not values:
                    continue
                    
                metric_analysis = self._analyze_metric(metric_name, values, filtered_timestamps.get(metric_name, []))
                
                if report_format == "summary":
                    stats = metric_analysis.get("statistics", {})
                    report["metrics"][metric_name] = {
                        "count": stats.get("count", 0),
                        "mean": stats.get("mean", 0.0),
                        "latest": metric_analysis.get("latest_value", 0.0)
                    }
                elif report_format == "detailed":
                    report["metrics"][metric_name] = metric_analysis
                else:  # json format
                    report["metrics"][metric_name] = {
                        "statistics": metric_analysis["statistics"],
                        "trends": metric_analysis["trends"] if include_trends else None,
                        "quality": metric_analysis["quality"]
                    }
            
            # Generate summary statistics
            report["summary"] = self._generate_summary_statistics(filtered_metrics)
            
            if include_trends and report_format != "summary":
                report["global_trends"] = self._analyze_global_trends(filtered_metrics, filtered_timestamps)
                
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate analytics report: {e}")
            return {
                "error": str(e),
                "generated_at": datetime.now().isoformat(),
                "status": "failed"
            }
    
    def _filter_by_time_window(self, time_window: timedelta) -> Dict[str, List[float]]:
        """Filter metrics by time window."""
        cutoff_time = datetime.now() - time_window
        filtered = {}
        
        for metric, timestamps in self.timestamps.items():
            indices = [i for i, ts in enumerate(timestamps) if ts > cutoff_time]
            filtered[metric] = [self.metrics[metric][i] for i in indices]
            
        return filtered
    
    def _filter_timestamps_by_window(self, time_window: timedelta) -> Dict[str, List[datetime]]:
        """Filter timestamps by time window."""
        cutoff_time = datetime.now() - time_window
        filtered = {}
        
        for metric, timestamps in self.timestamps.items():
            filtered[metric] = [ts for ts in timestamps if ts > cutoff_time]
            
        return filtered
    
    def _analyze_metric(self, name: str, values: List[float], timestamps: List[datetime]) -> Dict[str, Any]:
        """Perform comprehensive analysis of a single metric."""
        if not values:
            return {"error": "No data available"}
            
        try:
            # Basic statistics
            statistics = {
                "count": len(values),
                "mean": mean(values),
                "median": median(values),
                "min": min(values),
                "max": max(values),
                "range": max(values) - min(values),
                "std_dev": stdev(values) if len(values) > 1 else 0.0
            }
            
            # Trend analysis
            trends = {}
            if len(values) > 1:
                # Simple linear trend
                x = list(range(len(values)))
                slope = np.polyfit(x, values, 1)[0] if len(values) > 1 else 0
                trends["linear_trend"] = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                trends["slope"] = float(slope)
                
                # Recent vs historical comparison
                if len(values) >= 10:
                    recent_mean = mean(values[-5:])
                    historical_mean = mean(values[:-5])
                    trends["recent_vs_historical"] = {
                        "recent_mean": recent_mean,
                        "historical_mean": historical_mean,
                        "change_percent": ((recent_mean - historical_mean) / historical_mean * 100) if historical_mean != 0 else 0
                    }
            
            # Data quality assessment
            quality = {
                "completeness": 1.0,  # Assuming no missing values in recorded data
                "consistency": self._assess_consistency(values),
                "outliers": self._detect_outliers(values)
            }
            
            return {
                "statistics": statistics,
                "trends": trends,
                "quality": quality,
                "latest_value": values[-1],
                "timestamps_analyzed": len(timestamps)
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _assess_consistency(self, values: List[float]) -> float:
        """Assess data consistency (lower coefficient of variation = more consistent)."""
        if len(values) < 2:
            return 1.0
        try:
            cv = stdev(values) / mean(values) if mean(values) != 0 else float('inf')
            return max(0.0, 1.0 - min(cv, 1.0))  # Normalize to 0-1 scale
        except:
            return 0.0
    
    def _detect_outliers(self, values: List[float]) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        if len(values) < 4:
            return {"count": 0, "indices": []}
            
        try:
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_indices = [i for i, v in enumerate(values) if v < lower_bound or v > upper_bound]
            
            return {
                "count": len(outlier_indices),
                "indices": outlier_indices,
                "bounds": {"lower": lower_bound, "upper": upper_bound}
            }
        except:
            return {"count": 0, "indices": []}
    
    def _generate_summary_statistics(self, metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        """Generate overall summary statistics across all metrics."""
        total_data_points = sum(len(values) for values in metrics.values())
        active_metrics = len([m for m, v in metrics.items() if v])
        
        return {
            "total_metrics": len(metrics),
            "active_metrics": active_metrics,
            "total_data_points": total_data_points,
            "average_points_per_metric": total_data_points / max(active_metrics, 1),
            "data_density": active_metrics / max(len(metrics), 1)
        }
    
    def _analyze_global_trends(self, metrics: Dict[str, List[float]], 
                             timestamps: Dict[str, List[datetime]]) -> Dict[str, Any]:
        """Analyze trends across all metrics."""
        trend_summary = {"increasing": 0, "decreasing": 0, "stable": 0}
        
        for metric_name, values in metrics.items():
            if len(values) > 1:
                x = list(range(len(values)))
                slope = np.polyfit(x, values, 1)[0]
                if abs(slope) < 0.01:  # Threshold for "stable"
                    trend_summary["stable"] += 1
                elif slope > 0:
                    trend_summary["increasing"] += 1
                else:
                    trend_summary["decreasing"] += 1
        
        return trend_summary

    def save(self, path: str, format_type: str = "auto", compress: bool = False, 
             create_backup: bool = True) -> bool:
        """Save analytics data with multiple format support and atomic operations.
        
        Args:
            path: File path for saving
            format_type: Format ('auto', 'json', 'pickle', 'sqlite')
            compress: Whether to compress the data
            create_backup: Whether to create backup of existing file
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            path_obj = Path(path)
            
            # Auto-detect format from extension
            if format_type == "auto":
                extension = path_obj.suffix.lower()
                if extension == ".json":
                    format_type = "json"
                elif extension == ".pkl":
                    format_type = "pickle"
                elif extension == ".db":
                    format_type = "sqlite"
                else:
                    format_type = "json"  # Default
                    
            # Create directory if it doesn't exist
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup if requested and file exists
            if create_backup and path_obj.exists():
                backup_path = path_obj.with_suffix(f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}{path_obj.suffix}")
                path_obj.rename(backup_path)
                logger.info(f"Created backup: {backup_path}")
            
            # Prepare data for serialization
            data_to_save = {
                "metadata": asdict(self.metadata),
                "metrics": self.metrics,
                "timestamps": {k: [ts.isoformat() for ts in v] for k, v in self.timestamps.items()},
                "retention_policy": str(self._retention_policy) if self._retention_policy else None,
                "max_data_points": self._max_data_points
            }
            
            self.metadata.format_type = format_type
            self.metadata.compression = compress
            self.metadata.updated_at = datetime.now()
            
            # Use atomic write with temporary file
            with tempfile.NamedTemporaryFile(mode='wb' if format_type != 'json' else 'w', 
                                           delete=False, suffix=path_obj.suffix) as temp_file:
                temp_path = Path(temp_file.name)
                
                if format_type == "json":
                    content = json.dumps(data_to_save, indent=2, default=str)
                    if compress:
                        with gzip.open(temp_file.name + ".gz", 'wt') as gz_file:
                            gz_file.write(content)
                        temp_path = Path(temp_file.name + ".gz")
                        path_obj = path_obj.with_suffix(path_obj.suffix + ".gz")
                    else:
                        temp_file.write(content)
                        
                elif format_type == "pickle":
                    if compress:
                        with gzip.open(temp_file.name, 'wb') as gz_file:
                            pickle.dump(data_to_save, gz_file)
                    else:
                        pickle.dump(data_to_save, temp_file)
                        
                elif format_type == "sqlite":
                    self._save_to_sqlite(temp_path, data_to_save)
                    
            # Atomic move
            temp_path.replace(path_obj)
            logger.info(f"Successfully saved analytics data to {path_obj}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save analytics data: {e}")
            return False
    
    def _save_to_sqlite(self, path: Path, data: Dict[str, Any]) -> None:
        """Save data to SQLite database format."""
        with sqlite3.connect(path) as conn:
            # Create tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    metric_name TEXT,
                    value REAL,
                    timestamp TEXT,
                    PRIMARY KEY (metric_name, timestamp)
                )
            """)
            
            # Insert metadata
            for key, value in data["metadata"].items():
                conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", 
                           (key, json.dumps(value)))
            
            # Insert metrics
            for metric_name, values in data["metrics"].items():
                timestamps = data["timestamps"].get(metric_name, [])
                for i, value in enumerate(values):
                    timestamp = timestamps[i] if i < len(timestamps) else datetime.now().isoformat()
                    conn.execute("INSERT OR REPLACE INTO metrics (metric_name, value, timestamp) VALUES (?, ?, ?)",
                               (metric_name, value, timestamp))

    def load(self, path: str, validate_schema: bool = True) -> bool:
        """Load analytics data with automatic format detection and validation.
        
        Args:
            path: File path to load from
            validate_schema: Whether to validate data schema
            
        Returns:
            True if load successful, False otherwise
        """
        try:
            path_obj = Path(path)
            
            if not path_obj.exists():
                logger.warning(f"File does not exist: {path}")
                return False
            
            # Detect format and compression
            is_compressed = path_obj.suffix.lower() == ".gz"
            if is_compressed:
                actual_path = path_obj.with_suffix('')
            else:
                actual_path = path_obj
                
            file_extension = actual_path.suffix.lower()
            
            # Load data based on format
            if file_extension == ".json":
                data = self._load_json(path_obj, is_compressed)
            elif file_extension == ".pkl":
                data = self._load_pickle(path_obj, is_compressed)
            elif file_extension == ".db":
                data = self._load_sqlite(path_obj)
            else:
                # Try to detect format by content
                data = self._load_with_fallback(path_obj, is_compressed)
                
            if data is None:
                return False
            
            # Validate schema if requested
            if validate_schema and not self._validate_data_schema(data):
                logger.error("Data schema validation failed")
                return False
            
            # Load data into instance
            self._load_data_into_instance(data)
            logger.info(f"Successfully loaded analytics data from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load analytics data: {e}")
            return False
    
    def _load_json(self, path: Path, is_compressed: bool) -> Optional[Dict[str, Any]]:
        """Load JSON format data."""
        try:
            if is_compressed:
                with gzip.open(path, 'rt') as file:
                    return json.load(file)
            else:
                with open(path, 'r') as file:
                    return json.load(file)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None
    
    def _load_pickle(self, path: Path, is_compressed: bool) -> Optional[Dict[str, Any]]:
        """Load pickle format data."""
        try:
            if is_compressed:
                with gzip.open(path, 'rb') as file:
                    return pickle.load(file)
            else:
                with open(path, 'rb') as file:
                    return pickle.load(file)
        except (pickle.PickleError, EOFError) as e:
            logger.error(f"Pickle load error: {e}")
            return None
    
    def _load_sqlite(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load SQLite format data."""
        try:
            data = {"metadata": {}, "metrics": {}, "timestamps": {}}
            
            with sqlite3.connect(path) as conn:
                # Load metadata
                cursor = conn.execute("SELECT key, value FROM metadata")
                for key, value in cursor.fetchall():
                    data["metadata"][key] = json.loads(value)
                
                # Load metrics
                cursor = conn.execute("SELECT metric_name, value, timestamp FROM metrics ORDER BY metric_name, timestamp")
                for metric_name, value, timestamp in cursor.fetchall():
                    if metric_name not in data["metrics"]:
                        data["metrics"][metric_name] = []
                        data["timestamps"][metric_name] = []
                    data["metrics"][metric_name].append(value)
                    data["timestamps"][metric_name].append(timestamp)
            
            return data
        except sqlite3.Error as e:
            logger.error(f"SQLite load error: {e}")
            return None
    
    def _load_with_fallback(self, path: Path, is_compressed: bool) -> Optional[Dict[str, Any]]:
        """Attempt to load with multiple formats as fallback."""
        # Try JSON first
        data = self._load_json(path, is_compressed)
        if data:
            return data
            
        # Try pickle
        data = self._load_pickle(path, is_compressed)
        if data:
            return data
            
        logger.error("Could not determine file format or file is corrupted")
        return None
    
    def _validate_data_schema(self, data: Dict[str, Any]) -> bool:
        """Validate loaded data schema."""
        required_keys = ["metadata", "metrics", "timestamps"]
        
        if not all(key in data for key in required_keys):
            return False
            
        # Validate metadata structure
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            return False
            
        # Validate metrics structure
        metrics = data.get("metrics", {})
        if not isinstance(metrics, dict):
            return False
            
        # Validate that all metrics have corresponding timestamps
        timestamps = data.get("timestamps", {})
        for metric_name, values in metrics.items():
            if metric_name not in timestamps:
                logger.warning(f"Missing timestamps for metric: {metric_name}")
                # Create default timestamps
                timestamps[metric_name] = [datetime.now().isoformat()] * len(values)
        
        return True
    
    def _load_data_into_instance(self, data: Dict[str, Any]) -> None:
        """Load validated data into the current instance."""
        # Load metadata
        metadata_dict = data.get("metadata", {})
        if "created_at" in metadata_dict:
            metadata_dict["created_at"] = datetime.fromisoformat(metadata_dict["created_at"])
        if "updated_at" in metadata_dict:
            metadata_dict["updated_at"] = datetime.fromisoformat(metadata_dict["updated_at"])
        self.metadata = AnalyticsMetadata(**metadata_dict)
        
        # Load metrics
        self.metrics = data.get("metrics", {})
        
        # Load timestamps
        timestamps_data = data.get("timestamps", {})
        self.timestamps = {}
        for metric_name, timestamp_strs in timestamps_data.items():
            self.timestamps[metric_name] = []
            for ts in timestamp_strs:
                if isinstance(ts, str):
                    try:
                        self.timestamps[metric_name].append(datetime.fromisoformat(ts))
                    except ValueError:
                        # Fallback for invalid timestamp strings
                        self.timestamps[metric_name].append(datetime.now())
                elif isinstance(ts, datetime):
                    self.timestamps[metric_name].append(ts)
                else:
                    # Fallback for other types
                    self.timestamps[metric_name].append(datetime.now())
        
        # Load retention policy
        retention_str = data.get("retention_policy")
        if retention_str and retention_str != "None":
            # Parse timedelta from string representation
            self._retention_policy = eval(retention_str) if retention_str.startswith("datetime.") else None
            
        self._max_data_points = data.get("max_data_points")


# Concrete implementation for infrastructure components
class InfrastructureAnalytics(BaseAnalytics):
    """Infrastructure-specific analytics implementation."""
    
    def __init__(self, component_name: str = "infrastructure"):
        super().__init__()
        self.component_name = component_name
        self.metadata.data_schema = f"{component_name}_infrastructure_analytics_v1"
    
    def record_performance_metric(self, operation: str, duration_ms: float, success: bool = True) -> None:
        """Record performance metrics for infrastructure operations."""
        self.record_metric(f"{operation}_duration_ms", duration_ms)
        self.record_metric(f"{operation}_success_rate", 1.0 if success else 0.0)
    
    def record_resource_usage(self, resource_type: str, usage_percent: float) -> None:
        """Record resource usage metrics."""
        self.record_metric(f"{resource_type}_usage_percent", usage_percent)
    
    def record_error(self, error_type: str, error_count: int = 1) -> None:
        """Record error occurrences."""
        self.record_metric(f"error_{error_type}_count", float(error_count))
    
    def get_component_health_score(self) -> float:
        """Calculate component health score based on metrics."""
        if not self.metrics:
            return 100.0
        
        # Get recent performance metrics
        success_rates = []
        error_counts = []
        
        for metric_name, values in self.metrics.items():
            if "success_rate" in metric_name and values:
                success_rates.extend(values[-10:])  # Last 10 values
            elif "error_" in metric_name and "count" in metric_name and values:
                error_counts.extend(values[-10:])  # Last 10 values
        
        # Calculate health score
        health_score = 100.0
        
        if success_rates:
            avg_success_rate = sum(success_rates) / len(success_rates)
            health_score *= avg_success_rate
        
        if error_counts:
            total_errors = sum(error_counts)
            error_penalty = min(50, total_errors * 5)  # Max 50 point penalty
            health_score -= error_penalty
        
        return max(0.0, min(100.0, health_score))