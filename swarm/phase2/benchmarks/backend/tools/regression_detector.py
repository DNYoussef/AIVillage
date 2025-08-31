"""
Performance Regression Detection

Advanced regression detection system that analyzes performance trends,
identifies degradations, and provides automated alerts and recommendations.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics
from enum import Enum
import logging

class RegressionSeverity(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RegressionAlert:
    """Performance regression alert"""
    metric_name: str
    benchmark_name: str
    severity: RegressionSeverity
    current_value: float
    baseline_value: float
    change_percent: float
    threshold_exceeded: str
    description: str
    recommendation: str
    timestamp: str

@dataclass
class PerformanceTrend:
    """Performance trend analysis"""
    metric_name: str
    values: List[float]
    timestamps: List[str]
    trend_direction: str  # 'improving', 'degrading', 'stable'
    trend_strength: float  # 0-1, how strong the trend is
    volatility: float  # standard deviation of values
    change_rate: float  # rate of change per time unit

class RegressionDetector:
    """
    Advanced performance regression detection system
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.logger = self._setup_logging()
        self.historical_data = []
        
    def _default_config(self) -> Dict[str, Any]:
        """Default regression detection configuration"""
        return {
            'thresholds': {
                'throughput': {
                    'warning': -5.0,    # 5% decrease is warning
                    'critical': -15.0   # 15% decrease is critical
                },
                'latency': {
                    'warning': 10.0,    # 10% increase is warning 
                    'critical': 25.0    # 25% increase is critical
                },
                'memory': {
                    'warning': 20.0,    # 20% increase is warning
                    'critical': 50.0    # 50% increase is critical
                },
                'success_rate': {
                    'warning': -2.0,    # 2% decrease is warning
                    'critical': -5.0    # 5% decrease is critical
                }
            },
            'trend_analysis': {
                'min_samples': 3,
                'stability_threshold': 0.05,  # 5% variation for stability
                'trend_strength_threshold': 0.7
            },
            'anomaly_detection': {
                'enabled': True,
                'std_dev_threshold': 2.0,  # 2 standard deviations
                'min_samples_for_anomaly': 5
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup regression detection logging"""
        logger = logging.getLogger('regression_detector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze_regression(self, current_results: Dict[str, Any], 
                          baseline_results: Dict[str, Any]) -> List[RegressionAlert]:
        """
        Analyze performance regression between current and baseline results
        """
        alerts = []
        
        self.logger.info("Starting regression analysis")
        
        # Compare each benchmark
        for benchmark_name in current_results.keys():
            if benchmark_name not in baseline_results:
                self.logger.warning(f"No baseline data for benchmark: {benchmark_name}")
                continue
            
            current = current_results[benchmark_name]
            baseline = baseline_results[benchmark_name]
            
            # Analyze different metrics
            benchmark_alerts = self._analyze_benchmark_regression(
                benchmark_name, current, baseline
            )
            alerts.extend(benchmark_alerts)
        
        # Sort alerts by severity
        severity_order = {
            RegressionSeverity.CRITICAL: 0,
            RegressionSeverity.HIGH: 1,
            RegressionSeverity.MEDIUM: 2,
            RegressionSeverity.LOW: 3,
            RegressionSeverity.NONE: 4
        }
        
        alerts.sort(key=lambda x: severity_order[x.severity])
        
        self.logger.info(f"Regression analysis complete. Found {len(alerts)} alerts")
        
        return alerts
    
    def _analyze_benchmark_regression(self, benchmark_name: str,
                                    current: Any, baseline: Any) -> List[RegressionAlert]:
        """Analyze regression for a specific benchmark"""
        alerts = []
        
        # Extract metrics from benchmark results
        current_metrics = self._extract_metrics(current)
        baseline_metrics = self._extract_metrics(baseline)
        
        # Compare each metric
        for metric_name, current_value in current_metrics.items():
            if metric_name not in baseline_metrics:
                continue
                
            baseline_value = baseline_metrics[metric_name]
            
            # Calculate change percentage
            if baseline_value != 0:
                change_percent = ((current_value - baseline_value) / baseline_value) * 100
            else:
                change_percent = 0 if current_value == 0 else float('inf')
            
            # Check for regression
            alert = self._check_metric_regression(
                metric_name, benchmark_name, current_value, 
                baseline_value, change_percent
            )
            
            if alert:
                alerts.append(alert)
        
        return alerts
    
    def _extract_metrics(self, benchmark_result: Any) -> Dict[str, float]:
        """Extract key metrics from benchmark result"""
        metrics = {}
        
        # Handle different result formats
        if hasattr(benchmark_result, '__dict__'):
            result_dict = benchmark_result.__dict__
        elif isinstance(benchmark_result, dict):
            result_dict = benchmark_result
        else:
            return metrics
        
        # Extract throughput
        metrics['throughput'] = result_dict.get('throughput', 0)
        
        # Extract latency metrics
        latency_stats = result_dict.get('latency_stats', {})
        if isinstance(latency_stats, dict):
            metrics['latency_avg'] = latency_stats.get('avg', 0)
            metrics['latency_p95'] = latency_stats.get('p95', 0)
            metrics['latency_p99'] = latency_stats.get('p99', 0)
        
        # Extract success rate
        metrics['success_rate'] = result_dict.get('success_rate', 1.0) * 100
        
        # Extract resource usage
        resource_usage = result_dict.get('resource_usage', {})
        if isinstance(resource_usage, dict):
            memory_stats = resource_usage.get('memory', {})
            cpu_stats = resource_usage.get('cpu', {})
            
            if isinstance(memory_stats, dict):
                metrics['memory_peak'] = memory_stats.get('peak_mb', 0)
                metrics['memory_avg'] = memory_stats.get('avg', 0)
            
            if isinstance(cpu_stats, dict):
                metrics['cpu_avg'] = cpu_stats.get('avg', 0)
                metrics['cpu_max'] = cpu_stats.get('max', 0)
        
        return metrics
    
    def _check_metric_regression(self, metric_name: str, benchmark_name: str,
                                current_value: float, baseline_value: float,
                                change_percent: float) -> Optional[RegressionAlert]:
        """Check if a metric shows regression"""
        
        # Determine metric category for thresholds
        metric_category = self._categorize_metric(metric_name)
        
        if metric_category not in self.config['thresholds']:
            return None
        
        thresholds = self.config['thresholds'][metric_category]
        
        # Determine if this is a "lower is better" or "higher is better" metric
        lower_is_better = metric_category in ['latency', 'memory']
        
        if lower_is_better:
            # For latency/memory, increases are bad
            warning_threshold = thresholds['warning']
            critical_threshold = thresholds['critical']
        else:
            # For throughput/success_rate, decreases are bad
            warning_threshold = thresholds['warning'] 
            critical_threshold = thresholds['critical']
        
        # Determine severity
        severity = RegressionSeverity.NONE
        threshold_exceeded = None
        
        if lower_is_better:
            if change_percent >= critical_threshold:
                severity = RegressionSeverity.CRITICAL
                threshold_exceeded = f">{critical_threshold}%"
            elif change_percent >= warning_threshold:
                severity = RegressionSeverity.HIGH if change_percent >= warning_threshold * 2 else RegressionSeverity.MEDIUM
                threshold_exceeded = f">{warning_threshold}%"
        else:
            if change_percent <= critical_threshold:
                severity = RegressionSeverity.CRITICAL
                threshold_exceeded = f"<{critical_threshold}%"
            elif change_percent <= warning_threshold:
                severity = RegressionSeverity.HIGH if change_percent <= warning_threshold * 2 else RegressionSeverity.MEDIUM
                threshold_exceeded = f"<{warning_threshold}%"
        
        if severity == RegressionSeverity.NONE:
            return None
        
        # Generate description and recommendation
        description = self._generate_description(
            metric_name, benchmark_name, change_percent, lower_is_better
        )
        
        recommendation = self._generate_recommendation(
            metric_name, benchmark_name, change_percent, severity, lower_is_better
        )
        
        return RegressionAlert(
            metric_name=metric_name,
            benchmark_name=benchmark_name,
            severity=severity,
            current_value=current_value,
            baseline_value=baseline_value,
            change_percent=change_percent,
            threshold_exceeded=threshold_exceeded,
            description=description,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )
    
    def _categorize_metric(self, metric_name: str) -> str:
        """Categorize metric for threshold selection"""
        if 'throughput' in metric_name.lower():
            return 'throughput'
        elif 'latency' in metric_name.lower():
            return 'latency'
        elif 'memory' in metric_name.lower():
            return 'memory'
        elif 'success' in metric_name.lower() or 'rate' in metric_name.lower():
            return 'success_rate'
        elif 'cpu' in metric_name.lower():
            return 'memory'  # Use memory thresholds for CPU
        else:
            return 'throughput'  # Default
    
    def _generate_description(self, metric_name: str, benchmark_name: str,
                            change_percent: float, lower_is_better: bool) -> str:
        """Generate human-readable description of regression"""
        direction = "increased" if change_percent > 0 else "decreased"
        abs_change = abs(change_percent)
        
        if lower_is_better and change_percent > 0:
            impact = "degradation"
        elif not lower_is_better and change_percent < 0:
            impact = "degradation"
        else:
            impact = "improvement"
        
        return (f"{metric_name} in {benchmark_name} {direction} by {abs_change:.1f}%, "
                f"indicating a performance {impact}")
    
    def _generate_recommendation(self, metric_name: str, benchmark_name: str,
                               change_percent: float, severity: RegressionSeverity,
                               lower_is_better: bool) -> str:
        """Generate actionable recommendations"""
        
        base_recommendations = {
            'throughput': [
                "Review algorithm efficiency and bottlenecks",
                "Check for resource contention (CPU, memory, I/O)",
                "Analyze concurrent processing capabilities",
                "Consider connection pooling and caching strategies"
            ],
            'latency': [
                "Optimize database queries and indexes",
                "Review network communication patterns",
                "Check for blocking operations in critical paths",
                "Consider async processing for heavy operations"
            ],
            'memory': [
                "Review memory allocation patterns",
                "Check for memory leaks",
                "Optimize data structures and caching",
                "Consider garbage collection tuning"
            ],
            'success_rate': [
                "Review error handling and retry logic",
                "Check for timeout configurations",
                "Analyze failure patterns in logs",
                "Validate resource availability and limits"
            ]
        }
        
        category = self._categorize_metric(metric_name)
        recommendations = base_recommendations.get(category, ["Review system performance"])
        
        if severity == RegressionSeverity.CRITICAL:
            prefix = "URGENT: "
            action = "Immediate investigation required. "
        elif severity == RegressionSeverity.HIGH:
            prefix = "HIGH PRIORITY: "
            action = "Priority investigation needed. "
        else:
            prefix = ""
            action = "Consider investigating. "
        
        recommendation = recommendations[0]  # Take first recommendation
        
        return f"{prefix}{action}{recommendation}"
    
    def analyze_trends(self, historical_results: List[Dict[str, Any]]) -> Dict[str, List[PerformanceTrend]]:
        """Analyze performance trends over time"""
        if len(historical_results) < self.config['trend_analysis']['min_samples']:
            return {}
        
        trends = {}
        
        # Group results by benchmark
        benchmark_groups = {}
        for result in historical_results:
            for benchmark_name, benchmark_result in result.items():
                if benchmark_name not in benchmark_groups:
                    benchmark_groups[benchmark_name] = []
                benchmark_groups[benchmark_name].append(benchmark_result)
        
        # Analyze trends for each benchmark
        for benchmark_name, results in benchmark_groups.items():
            benchmark_trends = self._analyze_benchmark_trends(benchmark_name, results)
            if benchmark_trends:
                trends[benchmark_name] = benchmark_trends
        
        return trends
    
    def _analyze_benchmark_trends(self, benchmark_name: str, results: List[Any]) -> List[PerformanceTrend]:
        """Analyze trends for a specific benchmark"""
        trends = []
        
        # Extract metrics time series
        metrics_series = {}
        timestamps = []
        
        for result in results:
            metrics = self._extract_metrics(result)
            timestamp = getattr(result, 'timestamp', datetime.now().isoformat())
            timestamps.append(timestamp)
            
            for metric_name, value in metrics.items():
                if metric_name not in metrics_series:
                    metrics_series[metric_name] = []
                metrics_series[metric_name].append(value)
        
        # Analyze trend for each metric
        for metric_name, values in metrics_series.items():
            if len(values) >= self.config['trend_analysis']['min_samples']:
                trend = self._calculate_trend(metric_name, values, timestamps)
                if trend:
                    trends.append(trend)
        
        return trends
    
    def _calculate_trend(self, metric_name: str, values: List[float], 
                        timestamps: List[str]) -> Optional[PerformanceTrend]:
        """Calculate trend for a metric"""
        if len(values) < 2:
            return None
        
        # Calculate basic statistics
        mean_value = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        volatility = (std_dev / mean_value) if mean_value != 0 else 0
        
        # Calculate trend direction and strength using linear regression
        x_values = list(range(len(values)))
        
        try:
            # Simple linear regression
            n = len(values)
            sum_x = sum(x_values)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(x_values, values))
            sum_x2 = sum(x * x for x in x_values)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # Calculate R-squared (trend strength)
            y_mean = sum_y / n
            ss_tot = sum((y - y_mean) ** 2 for y in values)
            ss_res = sum((y - (slope * x + (sum_y - slope * sum_x) / n)) ** 2 
                        for x, y in zip(x_values, values))
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
        except (ZeroDivisionError, ValueError):
            slope = 0
            r_squared = 0
        
        # Determine trend direction
        stability_threshold = self.config['trend_analysis']['stability_threshold']
        
        if abs(slope) < stability_threshold * mean_value:
            trend_direction = "stable"
        elif slope > 0:
            # Check if this is a "lower is better" metric
            lower_is_better = self._categorize_metric(metric_name) in ['latency', 'memory']
            trend_direction = "degrading" if lower_is_better else "improving"
        else:
            lower_is_better = self._categorize_metric(metric_name) in ['latency', 'memory']
            trend_direction = "improving" if lower_is_better else "degrading"
        
        return PerformanceTrend(
            metric_name=metric_name,
            values=values,
            timestamps=timestamps,
            trend_direction=trend_direction,
            trend_strength=r_squared,
            volatility=volatility,
            change_rate=slope
        )
    
    def detect_anomalies(self, current_results: Dict[str, Any],
                        historical_results: List[Dict[str, Any]]) -> List[RegressionAlert]:
        """Detect performance anomalies using statistical analysis"""
        
        if not self.config['anomaly_detection']['enabled']:
            return []
        
        if len(historical_results) < self.config['anomaly_detection']['min_samples_for_anomaly']:
            return []
        
        alerts = []
        
        # Analyze each benchmark
        for benchmark_name in current_results.keys():
            # Collect historical values for this benchmark
            historical_metrics = {}
            
            for historical_result in historical_results:
                if benchmark_name in historical_result:
                    metrics = self._extract_metrics(historical_result[benchmark_name])
                    for metric_name, value in metrics.items():
                        if metric_name not in historical_metrics:
                            historical_metrics[metric_name] = []
                        historical_metrics[metric_name].append(value)
            
            # Check current values against historical distribution
            current_metrics = self._extract_metrics(current_results[benchmark_name])
            
            for metric_name, current_value in current_metrics.items():
                if metric_name in historical_metrics and len(historical_metrics[metric_name]) >= 3:
                    anomaly_alert = self._check_anomaly(
                        metric_name, benchmark_name, current_value, 
                        historical_metrics[metric_name]
                    )
                    
                    if anomaly_alert:
                        alerts.append(anomaly_alert)
        
        return alerts
    
    def _check_anomaly(self, metric_name: str, benchmark_name: str,
                      current_value: float, historical_values: List[float]) -> Optional[RegressionAlert]:
        """Check if current value is an anomaly compared to historical data"""
        
        if len(historical_values) < 3:
            return None
        
        mean_val = statistics.mean(historical_values)
        std_val = statistics.stdev(historical_values)
        
        if std_val == 0:  # No variation in historical data
            return None
        
        # Calculate z-score
        z_score = abs(current_value - mean_val) / std_val
        
        threshold = self.config['anomaly_detection']['std_dev_threshold']
        
        if z_score > threshold:
            # Determine severity based on z-score
            if z_score > threshold * 2:
                severity = RegressionSeverity.CRITICAL
            elif z_score > threshold * 1.5:
                severity = RegressionSeverity.HIGH
            else:
                severity = RegressionSeverity.MEDIUM
            
            # Calculate change from mean
            change_percent = ((current_value - mean_val) / mean_val) * 100 if mean_val != 0 else 0
            
            description = (f"Anomaly detected in {metric_name} for {benchmark_name}. "
                          f"Current value ({current_value:.2f}) is {z_score:.1f} standard deviations "
                          f"from historical mean ({mean_val:.2f})")
            
            recommendation = (f"Investigate unusual {metric_name} behavior. "
                             f"Check for environmental changes, data quality issues, or system anomalies.")
            
            return RegressionAlert(
                metric_name=metric_name,
                benchmark_name=benchmark_name,
                severity=severity,
                current_value=current_value,
                baseline_value=mean_val,
                change_percent=change_percent,
                threshold_exceeded=f">{threshold} std dev",
                description=description,
                recommendation=recommendation,
                timestamp=datetime.now().isoformat()
            )
        
        return None
    
    def generate_regression_report(self, alerts: List[RegressionAlert],
                                 trends: Dict[str, List[PerformanceTrend]] = None) -> str:
        """Generate comprehensive regression analysis report"""
        
        report = []
        report.append("🚨 PERFORMANCE REGRESSION ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Total Alerts: {len(alerts)}")
        report.append("")
        
        if not alerts:
            report.append("✅ No performance regressions detected!")
            return "\n".join(report)
        
        # Group alerts by severity
        severity_groups = {}
        for alert in alerts:
            if alert.severity not in severity_groups:
                severity_groups[alert.severity] = []
            severity_groups[alert.severity].append(alert)
        
        # Report by severity
        severity_icons = {
            RegressionSeverity.CRITICAL: "🔴",
            RegressionSeverity.HIGH: "🟠", 
            RegressionSeverity.MEDIUM: "🟡",
            RegressionSeverity.LOW: "🔵"
        }
        
        for severity in [RegressionSeverity.CRITICAL, RegressionSeverity.HIGH, 
                        RegressionSeverity.MEDIUM, RegressionSeverity.LOW]:
            if severity in severity_groups:
                report.append(f"{severity_icons[severity]} {severity.value.upper()} SEVERITY ({len(severity_groups[severity])} alerts)")
                report.append("-" * 50)
                
                for alert in severity_groups[severity]:
                    report.append(f"📊 {alert.benchmark_name} - {alert.metric_name}")
                    report.append(f"   Change: {alert.change_percent:+.1f}% "
                                 f"({alert.baseline_value:.2f} → {alert.current_value:.2f})")
                    report.append(f"   Issue: {alert.description}")
                    report.append(f"   Action: {alert.recommendation}")
                    report.append("")
        
        # Trend analysis summary
        if trends:
            report.append("📈 TREND ANALYSIS SUMMARY")
            report.append("-" * 30)
            
            for benchmark_name, benchmark_trends in trends.items():
                report.append(f"\n{benchmark_name}:")
                
                for trend in benchmark_trends:
                    trend_icon = {
                        'improving': '📈',
                        'degrading': '📉',
                        'stable': '➡️'
                    }.get(trend.trend_direction, '❓')
                    
                    report.append(f"  {trend_icon} {trend.metric_name}: {trend.trend_direction} "
                                 f"(strength: {trend.trend_strength:.2f}, volatility: {trend.volatility:.2f})")
        
        return "\n".join(report)

# Export main classes
__all__ = ['RegressionDetector', 'RegressionAlert', 'RegressionSeverity', 'PerformanceTrend']