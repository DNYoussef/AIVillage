# BaseAnalytics API Documentation

## Overview

The `BaseAnalytics` class provides a comprehensive framework for collecting, analyzing, and persisting analytics data with advanced features like time-series analysis, statistical computation, and multi-format persistence.

## Key Features

- **Multi-format Data Persistence**: JSON, Pickle, SQLite with compression support
- **Time-series Analytics**: Trend analysis, statistical metrics, and outlier detection
- **Memory Management**: Configurable retention policies for large datasets
- **Atomic Operations**: Safe save/load with backup and recovery
- **Performance Optimized**: Efficient handling of large datasets with concurrent access
- **Schema Validation**: Data integrity checks and migration support

## Quick Start

```python
from base_analytics import BaseAnalytics

class MyAnalytics(BaseAnalytics):
    pass

# Initialize analytics
analytics = MyAnalytics()

# Record metrics
analytics.record_metric("response_time", 0.150)
analytics.record_metric("throughput", 1200.0)
analytics.record_metric("error_rate", 0.02)

# Generate report
report = analytics.generate_analytics_report()
print(f"Total metrics: {report['metadata']['total_metrics']}")

# Save data
analytics.save("analytics_data.json")

# Load in new instance
new_analytics = MyAnalytics()
new_analytics.load("analytics_data.json")
```

## Core Methods

### Recording Metrics

#### `record_metric(metric: str, value: float, timestamp: datetime = None)`

Records a single metric value with optional timestamp.

```python
from datetime import datetime

# Record with current timestamp
analytics.record_metric("cpu_usage", 75.5)

# Record with custom timestamp
custom_time = datetime(2024, 1, 15, 10, 30, 0)
analytics.record_metric("cpu_usage", 80.2, custom_time)
```

**Parameters:**
- `metric`: Metric name (string identifier)
- `value`: Numeric value to record
- `timestamp`: Optional datetime (defaults to current time)

### Report Generation

#### `generate_analytics_report(report_format="json", include_trends=True, time_window=None)`

Generates comprehensive analytics reports with multiple format options.

```python
from datetime import timedelta

# Basic JSON report
report = analytics.generate_analytics_report()

# Summary format (minimal)
summary = analytics.generate_analytics_report(report_format="summary")

# Detailed format (comprehensive)
detailed = analytics.generate_analytics_report(
    report_format="detailed",
    include_trends=True
)

# Time-windowed analysis (last 24 hours)
recent = analytics.generate_analytics_report(
    time_window=timedelta(hours=24)
)
```

**Parameters:**
- `report_format`: "json", "summary", or "detailed"
- `include_trends`: Whether to include trend analysis
- `time_window`: Optional timedelta for filtering data

**Returns:** Dictionary containing structured analytics data

### Data Persistence

#### `save(path: str, format_type="auto", compress=False, create_backup=True)`

Saves analytics data with multiple format support and atomic operations.

```python
# Auto-detect format from extension
analytics.save("data.json")            # JSON format
analytics.save("data.pkl")             # Pickle format
analytics.save("data.db")              # SQLite format

# Explicit format specification
analytics.save("data.analytics", format_type="json")

# Compressed storage
analytics.save("data.json", compress=True)  # Creates data.json.gz

# Disable backup creation
analytics.save("data.json", create_backup=False)
```

**Parameters:**
- `path`: File path for saving
- `format_type`: "auto", "json", "pickle", or "sqlite"
- `compress`: Enable compression (JSON and Pickle only)
- `create_backup`: Create backup of existing file

**Returns:** Boolean success indicator

#### `load(path: str, validate_schema=True)`

Loads analytics data with automatic format detection and validation.

```python
# Basic loading
success = analytics.load("data.json")

# Disable schema validation
success = analytics.load("data.json", validate_schema=False)

# Load compressed files
success = analytics.load("data.json.gz")
```

**Parameters:**
- `path`: File path to load from
- `validate_schema`: Enable data schema validation

**Returns:** Boolean success indicator

### Memory Management

#### `set_retention_policy(retention_period=None, max_data_points=None)`

Configures data retention policies for memory efficiency.

```python
from datetime import timedelta

# Time-based retention (keep last 7 days)
analytics.set_retention_policy(retention_period=timedelta(days=7))

# Size-based retention (keep last 10,000 points per metric)
analytics.set_retention_policy(max_data_points=10000)

# Combined retention policy
analytics.set_retention_policy(
    retention_period=timedelta(hours=24),
    max_data_points=5000
)
```

## Report Structure

### JSON Format

```json
{
  "metadata": {
    "generated_at": "2024-01-15T10:30:00",
    "report_format": "json",
    "time_window": "all",
    "total_metrics": 3,
    "data_points": 150
  },
  "metrics": {
    "response_time": {
      "statistics": {
        "count": 50,
        "mean": 0.142,
        "median": 0.138,
        "min": 0.089,
        "max": 0.245,
        "std_dev": 0.023
      },
      "trends": {
        "linear_trend": "stable",
        "slope": -0.0001,
        "recent_vs_historical": {
          "recent_mean": 0.140,
          "historical_mean": 0.144,
          "change_percent": -2.8
        }
      },
      "quality": {
        "completeness": 1.0,
        "consistency": 0.85,
        "outliers": {
          "count": 2,
          "indices": [15, 32]
        }
      }
    }
  },
  "summary": {
    "total_metrics": 3,
    "active_metrics": 3,
    "total_data_points": 150,
    "average_points_per_metric": 50.0,
    "data_density": 1.0
  },
  "global_trends": {
    "increasing": 1,
    "decreasing": 0,
    "stable": 2
  }
}
```

### Summary Format

```json
{
  "metadata": {...},
  "metrics": {
    "response_time": {
      "count": 50,
      "mean": 0.142,
      "latest": 0.135
    }
  },
  "summary": {...}
}
```

## Usage Examples

### ML Model Training Analytics

```python
class TrainingAnalytics(BaseAnalytics):
    def __init__(self):
        super().__init__()
        # Set retention for training sessions
        self.set_retention_policy(max_data_points=10000)
    
    def log_epoch(self, epoch, train_loss, val_loss, accuracy):
        """Log training epoch metrics."""
        from datetime import datetime
        
        timestamp = datetime.now()
        self.record_metric("train_loss", train_loss, timestamp)
        self.record_metric("val_loss", val_loss, timestamp)
        self.record_metric("accuracy", accuracy, timestamp)
        
        # Log epoch number for correlation
        self.record_metric("epoch", float(epoch), timestamp)

# Usage
training = TrainingAnalytics()

for epoch in range(100):
    # Simulate training
    train_loss = 1.0 / (epoch + 1) + 0.1 * np.random.random()
    val_loss = train_loss + 0.05 * np.random.random()
    accuracy = 1.0 - val_loss + 0.02 * np.random.random()
    
    training.log_epoch(epoch, train_loss, val_loss, accuracy)
    
    # Generate progress report every 10 epochs
    if (epoch + 1) % 10 == 0:
        report = training.generate_analytics_report(
            time_window=timedelta(hours=1)  # Recent progress
        )
        print(f"Epoch {epoch + 1}: Loss trend = {report['metrics']['train_loss']['trends']['linear_trend']}")

# Save training session
training.save("training_session.json", compress=True)
```

### System Performance Monitoring

```python
import psutil
import time
from datetime import datetime, timedelta

class SystemMonitor(BaseAnalytics):
    def __init__(self):
        super().__init__()
        # Retain 24 hours of data
        self.set_retention_policy(retention_period=timedelta(hours=24))
    
    def collect_system_metrics(self):
        """Collect current system metrics."""
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self.record_metric("cpu_usage", cpu_percent, timestamp)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.record_metric("memory_usage", memory.percent, timestamp)
        self.record_metric("memory_available", memory.available / 1024**3, timestamp)  # GB
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.record_metric("disk_usage", disk.percent, timestamp)
        
        # Network metrics (simplified)
        net_io = psutil.net_io_counters()
        self.record_metric("network_bytes_sent", float(net_io.bytes_sent), timestamp)
        self.record_metric("network_bytes_recv", float(net_io.bytes_recv), timestamp)
    
    def generate_health_report(self):
        """Generate system health assessment."""
        report = self.generate_analytics_report(
            report_format="detailed",
            time_window=timedelta(hours=1)  # Last hour
        )
        
        # Custom health scoring
        health_score = 100
        
        for metric_name, metric_data in report['metrics'].items():
            stats = metric_data.get('statistics', {})
            
            if 'cpu_usage' in metric_name and stats.get('mean', 0) > 80:
                health_score -= 20
            elif 'memory_usage' in metric_name and stats.get('mean', 0) > 90:
                health_score -= 15
            elif 'disk_usage' in metric_name and stats.get('mean', 0) > 95:
                health_score -= 25
        
        return {
            'health_score': max(0, health_score),
            'detailed_report': report,
            'recommendations': self._generate_recommendations(report)
        }
    
    def _generate_recommendations(self, report):
        """Generate system optimization recommendations."""
        recommendations = []
        
        for metric_name, metric_data in report['metrics'].items():
            stats = metric_data.get('statistics', {})
            trends = metric_data.get('trends', {})
            
            if 'cpu_usage' in metric_name:
                if stats.get('mean', 0) > 75:
                    recommendations.append("Consider CPU optimization or scaling")
                if trends.get('linear_trend') == 'increasing':
                    recommendations.append("CPU usage is trending upward")
            
            elif 'memory_usage' in metric_name:
                if stats.get('mean', 0) > 85:
                    recommendations.append("Memory usage is high - consider optimization")
        
        return recommendations

# Usage
monitor = SystemMonitor()

# Continuous monitoring loop
for i in range(60):  # Monitor for 1 hour
    monitor.collect_system_metrics()
    
    if i % 10 == 0:  # Health check every 10 minutes
        health = monitor.generate_health_report()
        print(f"System Health Score: {health['health_score']}/100")
        
        if health['recommendations']:
            print("Recommendations:")
            for rec in health['recommendations']:
                print(f"  - {rec}")
    
    time.sleep(60)  # 1-minute intervals

# Save monitoring session
monitor.save("system_monitoring.db", format_type="sqlite")
```

### API Performance Analytics

```python
import time
from datetime import datetime, timedelta
from functools import wraps

class APIAnalytics(BaseAnalytics):
    def __init__(self):
        super().__init__()
        # Keep 7 days of API metrics
        self.set_retention_policy(retention_period=timedelta(days=7))
    
    def performance_decorator(self, endpoint_name):
        """Decorator for automatic performance tracking."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                timestamp = datetime.now()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Record success metrics
                    duration = time.perf_counter() - start_time
                    self.record_metric(f"{endpoint_name}_response_time", duration, timestamp)
                    self.record_metric(f"{endpoint_name}_success_count", 1.0, timestamp)
                    
                    return result
                    
                except Exception as e:
                    # Record error metrics
                    duration = time.perf_counter() - start_time
                    self.record_metric(f"{endpoint_name}_response_time", duration, timestamp)
                    self.record_metric(f"{endpoint_name}_error_count", 1.0, timestamp)
                    raise
            
            return wrapper
        return decorator
    
    def generate_sla_report(self):
        """Generate SLA compliance report."""
        report = self.generate_analytics_report(
            time_window=timedelta(hours=24),
            include_trends=True
        )
        
        sla_report = {
            'period': '24 hours',
            'endpoints': {},
            'overall_health': 'healthy'
        }
        
        for metric_name, metric_data in report['metrics'].items():
            if '_response_time' in metric_name:
                endpoint = metric_name.replace('_response_time', '')
                stats = metric_data['statistics']
                
                # Calculate SLA metrics
                p95_response_time = np.percentile(
                    self.metrics.get(metric_name, []), 95
                ) if metric_name in self.metrics else 0
                
                sla_report['endpoints'][endpoint] = {
                    'avg_response_time': stats.get('mean', 0),
                    'p95_response_time': p95_response_time,
                    'max_response_time': stats.get('max', 0),
                    'sla_violation': p95_response_time > 0.5,  # 500ms SLA
                    'trend': metric_data['trends']['linear_trend']
                }
        
        return sla_report

# Usage with Flask/FastAPI
api_analytics = APIAnalytics()

# Manual tracking
@api_analytics.performance_decorator("get_users")
def get_users():
    time.sleep(0.1)  # Simulate work
    return {"users": []}

@api_analytics.performance_decorator("create_user")
def create_user(user_data):
    time.sleep(0.2)  # Simulate work
    return {"user_id": 123}

# Simulate API calls
for i in range(1000):
    if i % 3 == 0:
        get_users()
    else:
        create_user({"name": f"user_{i}"})

# Generate SLA report
sla_report = api_analytics.generate_sla_report()
print("SLA Report:")
for endpoint, metrics in sla_report['endpoints'].items():
    status = "VIOLATION" if metrics['sla_violation'] else "OK"
    print(f"  {endpoint}: {metrics['avg_response_time']:.3f}s avg, {status}")
```

## Performance Considerations

### Memory Management

- Use retention policies for long-running applications
- Consider compression for storage-heavy scenarios
- SQLite format is most memory-efficient for very large datasets

### Storage Formats

| Format | Speed | Size | Compression | Human Readable |
|--------|-------|------|-------------|----------------|
| JSON   | Medium| Large| Yes         | Yes            |
| Pickle | Fast  | Medium| Yes         | No             |
| SQLite | Slow  | Small | No          | Partially      |

### Best Practices

1. **Set appropriate retention policies** for production systems
2. **Use time windows** for analysis of large historical datasets
3. **Choose storage format** based on your use case:
   - JSON: Human-readable, web-friendly
   - Pickle: Fastest Python serialization
   - SQLite: Query-friendly, most compact
4. **Monitor memory usage** in long-running applications
5. **Use compression** for archival storage
6. **Validate data** after loading in production systems

## Error Handling

The analytics system provides graceful error handling:

```python
# Save operations return boolean status
success = analytics.save("invalid/path/file.json")
if not success:
    print("Save failed - check logs for details")

# Load operations handle missing/corrupted files
success = analytics.load("missing_file.json")
if not success:
    print("Load failed - using empty analytics instance")

# Report generation handles edge cases
report = analytics.generate_analytics_report()
if 'error' in report:
    print(f"Report generation failed: {report['error']}")
```

## Advanced Features

### Custom Metric Analysis

```python
class CustomAnalytics(BaseAnalytics):
    def calculate_custom_metric(self, base_metric, window_minutes=60):
        """Calculate custom derived metrics."""
        if base_metric not in self.metrics:
            return None
        
        # Get recent data
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_indices = [
            i for i, ts in enumerate(self.timestamps[base_metric])
            if ts > cutoff_time
        ]
        
        if not recent_indices:
            return None
        
        recent_values = [self.metrics[base_metric][i] for i in recent_indices]
        
        # Custom calculations
        return {
            'moving_average': np.mean(recent_values),
            'volatility': np.std(recent_values),
            'momentum': recent_values[-1] - recent_values[0] if len(recent_values) > 1 else 0
        }
```

### Integration with Monitoring Systems

```python
def export_to_prometheus(analytics, output_file):
    """Export metrics in Prometheus format."""
    report = analytics.generate_analytics_report()
    
    with open(output_file, 'w') as f:
        for metric_name, metric_data in report['metrics'].items():
            stats = metric_data['statistics']
            
            # Write Prometheus metrics
            f.write(f"# HELP {metric_name}_mean Average value\\n")
            f.write(f"# TYPE {metric_name}_mean gauge\\n")
            f.write(f"{metric_name}_mean {stats['mean']}\\n\\n")
            
            f.write(f"# HELP {metric_name}_count Total count\\n")
            f.write(f"# TYPE {metric_name}_count counter\\n")
            f.write(f"{metric_name}_count {stats['count']}\\n\\n")
```

## Troubleshooting

### Common Issues

1. **Memory growth**: Set retention policies
2. **Slow report generation**: Use time windows for large datasets
3. **Save/load failures**: Check file permissions and disk space
4. **Missing timestamps**: Verify timestamp data integrity
5. **Schema validation errors**: Check data format compatibility

### Debugging

Enable debug logging to troubleshoot issues:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('base_analytics')

# Analytics operations will now log detailed information
analytics = MyAnalytics()
analytics.record_metric("debug_test", 42.0)  # Logs debug info
```