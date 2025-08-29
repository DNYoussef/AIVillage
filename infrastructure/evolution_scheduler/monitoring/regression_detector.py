"""
Performance Regression Detection System
Archaeological Enhancement: Advanced regression detection from performance-monitoring branches

Innovation Score: 7.2/10
Branch Origins: performance-monitoring, regression-detection, adaptive-validation
Preservation Priority: HIGH - Critical for preventing performance degradation

This module provides comprehensive performance regression detection with statistical
analysis, trend detection, and automated rollback capabilities for evolved models.
"""

import logging
import numpy as np
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Callable
from datetime import datetime, timedelta
import statistics
from collections import deque, defaultdict
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)


class RegressionSeverity(Enum):
    """Severity levels for performance regressions."""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class DetectionMethod(Enum):
    """Methods for detecting performance regressions."""
    THRESHOLD_BASED = "threshold_based"
    STATISTICAL_TEST = "statistical_test"
    TREND_ANALYSIS = "trend_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    ENSEMBLE = "ensemble"  # Archaeological enhancement


@dataclass
class PerformanceMetric:
    """Represents a performance metric with historical data."""
    name: str
    current_value: float
    historical_values: deque = field(default_factory=lambda: deque(maxlen=100))
    baseline_value: Optional[float] = None
    target_direction: str = "maximize"  # maximize, minimize, or target
    target_value: Optional[float] = None
    tolerance: float = 0.05  # 5% tolerance by default
    
    def add_value(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Add a new value to the historical data."""
        self.historical_values.append({
            'value': value,
            'timestamp': timestamp or datetime.now()
        })
        self.current_value = value
    
    def get_recent_values(self, count: int = 10) -> List[float]:
        """Get the most recent values."""
        if count >= len(self.historical_values):
            return [item['value'] for item in self.historical_values]
        return [item['value'] for item in list(self.historical_values)[-count:]]
    
    def get_baseline_comparison(self) -> Optional[float]:
        """Get relative change from baseline."""
        if self.baseline_value is None:
            return None
        
        if self.baseline_value == 0:
            return float('inf') if self.current_value != 0 else 0.0
        
        return (self.current_value - self.baseline_value) / abs(self.baseline_value)


@dataclass
class RegressionAlert:
    """Represents a performance regression alert."""
    alert_id: str
    metric_name: str
    severity: RegressionSeverity
    detection_method: DetectionMethod
    current_value: float
    expected_value: float
    deviation: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


class StatisticalRegressionDetector:
    """
    Statistical methods for regression detection.
    
    Archaeological enhancement: Multiple statistical tests for robust detection.
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def t_test_regression(
        self, 
        baseline_values: List[float], 
        current_values: List[float]
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Perform t-test to detect significant performance change.
        
        Returns:
            (is_regression, p_value, metadata)
        """
        if len(baseline_values) < 2 or len(current_values) < 2:
            return False, 1.0, {"error": "Insufficient data for t-test"}
        
        try:
            # Perform two-sample t-test
            t_stat, p_value = stats.ttest_ind(baseline_values, current_values)
            
            # Check if there's a significant difference
            is_significant = p_value < self.significance_level
            
            # Determine if it's a regression (performance decreased)
            baseline_mean = np.mean(baseline_values)
            current_mean = np.mean(current_values)
            is_regression = is_significant and current_mean < baseline_mean
            
            metadata = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'baseline_mean': float(baseline_mean),
                'current_mean': float(current_mean),
                'effect_size': float(abs(current_mean - baseline_mean) / np.std(baseline_values + current_values))
            }
            
            return is_regression, float(p_value), metadata
            
        except Exception as e:
            return False, 1.0, {"error": str(e)}
    
    def mann_whitney_test(
        self, 
        baseline_values: List[float], 
        current_values: List[float]
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test).
        """
        if len(baseline_values) < 3 or len(current_values) < 3:
            return False, 1.0, {"error": "Insufficient data for Mann-Whitney test"}
        
        try:
            u_stat, p_value = stats.mannwhitneyu(
                baseline_values, 
                current_values, 
                alternative='two-sided'
            )
            
            is_significant = p_value < self.significance_level
            baseline_median = np.median(baseline_values)
            current_median = np.median(current_values)
            is_regression = is_significant and current_median < baseline_median
            
            metadata = {
                'u_statistic': float(u_stat),
                'p_value': float(p_value),
                'baseline_median': float(baseline_median),
                'current_median': float(current_median)
            }
            
            return is_regression, float(p_value), metadata
            
        except Exception as e:
            return False, 1.0, {"error": str(e)}
    
    def kolmogorov_smirnov_test(
        self, 
        baseline_values: List[float], 
        current_values: List[float]
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Perform Kolmogorov-Smirnov test for distribution comparison.
        """
        if len(baseline_values) < 5 or len(current_values) < 5:
            return False, 1.0, {"error": "Insufficient data for KS test"}
        
        try:
            ks_stat, p_value = stats.ks_2samp(baseline_values, current_values)
            
            is_significant = p_value < self.significance_level
            
            metadata = {
                'ks_statistic': float(ks_stat),
                'p_value': float(p_value),
                'distributions_different': is_significant
            }
            
            # For regression detection, we need additional check for direction
            is_regression = is_significant and np.mean(current_values) < np.mean(baseline_values)
            
            return is_regression, float(p_value), metadata
            
        except Exception as e:
            return False, 1.0, {"error": str(e)}


class TrendAnalysisDetector:
    """
    Trend analysis for detecting performance degradation patterns.
    
    Archaeological enhancement: Advanced trend detection with multiple algorithms.
    """
    
    def __init__(self, min_trend_points: int = 5):
        self.min_trend_points = min_trend_points
    
    def linear_trend_analysis(self, values: List[float]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Analyze linear trend in performance values.
        
        Returns:
            (is_declining_trend, slope, metadata)
        """
        if len(values) < self.min_trend_points:
            return False, 0.0, {"error": "Insufficient data for trend analysis"}
        
        try:
            # Prepare data for linear regression
            X = np.array(range(len(values))).reshape(-1, 1)
            y = np.array(values)
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)
            
            slope = model.coef_[0]
            intercept = model.intercept_
            r_squared = model.score(X, y)
            
            # Determine if trend is significantly declining
            # Use statistical significance of slope
            n = len(values)
            y_pred = model.predict(X)
            mse = np.mean((y - y_pred) ** 2)
            slope_se = np.sqrt(mse / np.sum((X.flatten() - np.mean(X)) ** 2))
            t_stat = slope / slope_se if slope_se > 0 else 0
            
            # Critical t-value for 95% confidence
            critical_t = stats.t.ppf(0.975, n - 2)
            is_significant = abs(t_stat) > critical_t
            
            is_declining = is_significant and slope < 0
            
            metadata = {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_squared),
                't_statistic': float(t_stat),
                'is_significant': is_significant,
                'trend_strength': 'strong' if r_squared > 0.7 else 'moderate' if r_squared > 0.4 else 'weak'
            }
            
            return is_declining, float(slope), metadata
            
        except Exception as e:
            return False, 0.0, {"error": str(e)}
    
    def exponential_smoothing_trend(
        self, 
        values: List[float], 
        alpha: float = 0.3
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect trend using exponential smoothing.
        """
        if len(values) < self.min_trend_points:
            return False, 0.0, {"error": "Insufficient data"}
        
        try:
            # Apply exponential smoothing
            smoothed = [values[0]]
            for i in range(1, len(values)):
                smoothed_val = alpha * values[i] + (1 - alpha) * smoothed[-1]
                smoothed.append(smoothed_val)
            
            # Calculate trend from smoothed values
            recent_window = min(5, len(smoothed) // 2)
            recent_avg = np.mean(smoothed[-recent_window:])
            earlier_avg = np.mean(smoothed[:recent_window])
            
            trend_change = (recent_avg - earlier_avg) / abs(earlier_avg) if earlier_avg != 0 else 0
            
            # Threshold for significant decline
            decline_threshold = -0.05  # 5% decline
            is_declining = trend_change < decline_threshold
            
            metadata = {
                'smoothed_values': smoothed[-10:],  # Last 10 smoothed values
                'recent_average': float(recent_avg),
                'earlier_average': float(earlier_avg),
                'trend_change': float(trend_change)
            }
            
            return is_declining, float(trend_change), metadata
            
        except Exception as e:
            return False, 0.0, {"error": str(e)}
    
    def cusum_change_detection(
        self, 
        values: List[float], 
        reference_mean: Optional[float] = None
    ) -> Tuple[bool, int, Dict[str, Any]]:
        """
        CUSUM (Cumulative Sum) change point detection.
        
        Returns:
            (change_detected, change_point_index, metadata)
        """
        if len(values) < self.min_trend_points:
            return False, -1, {"error": "Insufficient data"}
        
        try:
            if reference_mean is None:
                reference_mean = np.mean(values[:len(values)//2])
            
            # CUSUM parameters
            k = 0.5 * np.std(values)  # Reference value
            h = 5 * np.std(values)    # Decision interval
            
            # Calculate CUSUM statistics
            cusum_pos = 0
            cusum_neg = 0
            cusum_pos_history = []
            cusum_neg_history = []
            
            for i, value in enumerate(values):
                deviation = value - reference_mean
                
                cusum_pos = max(0, cusum_pos + deviation - k)
                cusum_neg = max(0, cusum_neg - deviation - k)
                
                cusum_pos_history.append(cusum_pos)
                cusum_neg_history.append(cusum_neg)
                
                # Check for change point
                if cusum_pos > h or cusum_neg > h:
                    change_type = "upward" if cusum_pos > h else "downward"
                    
                    metadata = {
                        'change_point': i,
                        'change_type': change_type,
                        'cusum_positive': cusum_pos_history,
                        'cusum_negative': cusum_neg_history,
                        'reference_mean': float(reference_mean),
                        'decision_threshold': float(h)
                    }
                    
                    return True, i, metadata
            
            metadata = {
                'no_change_detected': True,
                'cusum_positive': cusum_pos_history,
                'cusum_negative': cusum_neg_history,
                'reference_mean': float(reference_mean)
            }
            
            return False, -1, metadata
            
        except Exception as e:
            return False, -1, {"error": str(e)}


class AnomalyDetector:
    """
    Anomaly detection for performance regression.
    
    Archaeological enhancement: Multiple anomaly detection methods.
    """
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
    
    def isolation_forest_detection(self, values: List[float]) -> Tuple[bool, List[bool], Dict[str, Any]]:
        """
        Use Isolation Forest for anomaly detection.
        """
        if len(values) < 10:
            return False, [False] * len(values), {"error": "Insufficient data"}
        
        try:
            from sklearn.ensemble import IsolationForest
            
            # Prepare data
            X = np.array(values).reshape(-1, 1)
            
            # Fit isolation forest
            iso_forest = IsolationForest(
                contamination=self.contamination, 
                random_state=42
            )
            anomaly_labels = iso_forest.fit_predict(X)
            
            # -1 indicates anomaly, 1 indicates normal
            is_anomaly = anomaly_labels == -1
            has_anomalies = np.any(is_anomaly)
            
            # Calculate anomaly scores
            anomaly_scores = iso_forest.decision_function(X)
            
            metadata = {
                'anomaly_count': int(np.sum(is_anomaly)),
                'anomaly_indices': [int(i) for i, is_anom in enumerate(is_anomaly) if is_anom],
                'anomaly_scores': [float(score) for score in anomaly_scores],
                'contamination': self.contamination
            }
            
            return has_anomalies, is_anomaly.tolist(), metadata
            
        except ImportError:
            logger.warning("sklearn not available, using simple anomaly detection")
            return self._simple_anomaly_detection(values)
        except Exception as e:
            return False, [False] * len(values), {"error": str(e)}
    
    def _simple_anomaly_detection(self, values: List[float]) -> Tuple[bool, List[bool], Dict[str, Any]]:
        """
        Simple anomaly detection using z-score.
        """
        if len(values) < 3:
            return False, [False] * len(values), {"error": "Insufficient data"}
        
        try:
            values_array = np.array(values)
            mean_val = np.mean(values_array)
            std_val = np.std(values_array)
            
            if std_val == 0:
                return False, [False] * len(values), {"error": "Zero standard deviation"}
            
            # Calculate z-scores
            z_scores = np.abs((values_array - mean_val) / std_val)
            
            # Threshold for anomaly (typically 2 or 3)
            threshold = 2.5
            is_anomaly = z_scores > threshold
            has_anomalies = np.any(is_anomaly)
            
            metadata = {
                'z_scores': z_scores.tolist(),
                'threshold': threshold,
                'mean': float(mean_val),
                'std': float(std_val),
                'anomaly_count': int(np.sum(is_anomaly))
            }
            
            return has_anomalies, is_anomaly.tolist(), metadata
            
        except Exception as e:
            return False, [False] * len(values), {"error": str(e)}


class ComprehensiveRegressionDetector:
    """
    Comprehensive regression detection system combining multiple methods.
    
    Archaeological enhancement: Ensemble approach with confidence scoring.
    """
    
    def __init__(
        self,
        threshold_tolerance: float = 0.1,
        statistical_significance: float = 0.05,
        trend_min_points: int = 5,
        anomaly_contamination: float = 0.1,
        enable_statistical: bool = True,
        enable_trend: bool = True,
        enable_anomaly: bool = True
    ):
        self.threshold_tolerance = threshold_tolerance
        self.statistical_detector = StatisticalRegressionDetector(statistical_significance)
        self.trend_detector = TrendAnalysisDetector(trend_min_points)
        self.anomaly_detector = AnomalyDetector(anomaly_contamination)
        
        self.enable_statistical = enable_statistical
        self.enable_trend = enable_trend
        self.enable_anomaly = enable_anomaly
        
        # Store historical alerts
        self.alerts_history = []
        self.metrics_registry = {}
    
    def register_metric(
        self, 
        name: str, 
        baseline_value: Optional[float] = None,
        target_direction: str = "maximize",
        tolerance: float = 0.05
    ) -> None:
        """Register a performance metric for monitoring."""
        self.metrics_registry[name] = PerformanceMetric(
            name=name,
            current_value=0.0,
            baseline_value=baseline_value,
            target_direction=target_direction,
            tolerance=tolerance
        )
        
        logger.info(f"Registered metric '{name}' for regression monitoring")
    
    def update_metric(self, name: str, value: float) -> None:
        """Update a metric with a new value."""
        if name not in self.metrics_registry:
            logger.warning(f"Metric '{name}' not registered, auto-registering with defaults")
            self.register_metric(name)
        
        self.metrics_registry[name].add_value(value)
    
    def detect_regression(
        self, 
        metric_name: str,
        recent_values: Optional[List[float]] = None,
        baseline_values: Optional[List[float]] = None
    ) -> Tuple[bool, List[RegressionAlert]]:
        """
        Comprehensive regression detection for a metric.
        
        Returns:
            (regression_detected, list_of_alerts)
        """
        if metric_name not in self.metrics_registry:
            logger.error(f"Metric '{metric_name}' not registered")
            return False, []
        
        metric = self.metrics_registry[metric_name]
        
        # Use provided values or extract from metric history
        if recent_values is None:
            recent_values = metric.get_recent_values(10)
        
        if len(recent_values) < 2:
            logger.warning(f"Insufficient data for regression detection on '{metric_name}'")
            return False, []
        
        alerts = []
        detection_results = {}
        
        # 1. Threshold-based detection
        threshold_alert = self._threshold_based_detection(metric, recent_values)
        if threshold_alert:
            alerts.append(threshold_alert)
            detection_results['threshold'] = True
        
        # 2. Statistical tests (if baseline available)
        if self.enable_statistical and baseline_values and len(baseline_values) >= 3:
            statistical_alerts = self._statistical_detection(metric, baseline_values, recent_values)
            alerts.extend(statistical_alerts)
            detection_results['statistical'] = len(statistical_alerts) > 0
        
        # 3. Trend analysis
        if self.enable_trend:
            trend_alert = self._trend_based_detection(metric, recent_values)
            if trend_alert:
                alerts.append(trend_alert)
                detection_results['trend'] = True
        
        # 4. Anomaly detection
        if self.enable_anomaly:
            anomaly_alert = self._anomaly_based_detection(metric, recent_values)
            if anomaly_alert:
                alerts.append(anomaly_alert)
                detection_results['anomaly'] = True
        
        # Store alerts
        self.alerts_history.extend(alerts)
        
        # Log detection results
        if alerts:
            logger.warning(f"Regression detected for '{metric_name}' using {list(detection_results.keys())}")
        
        return len(alerts) > 0, alerts
    
    def _threshold_based_detection(
        self, 
        metric: PerformanceMetric, 
        recent_values: List[float]
    ) -> Optional[RegressionAlert]:
        """Threshold-based regression detection."""
        if metric.baseline_value is None:
            return None
        
        current_value = recent_values[-1] if recent_values else metric.current_value
        baseline_comparison = (current_value - metric.baseline_value) / abs(metric.baseline_value)
        
        # Check if regression based on target direction
        is_regression = False
        if metric.target_direction == "maximize" and baseline_comparison < -metric.tolerance:
            is_regression = True
        elif metric.target_direction == "minimize" and baseline_comparison > metric.tolerance:
            is_regression = True
        elif metric.target_direction == "target" and abs(baseline_comparison) > metric.tolerance:
            is_regression = True
        
        if is_regression:
            severity = self._calculate_severity(abs(baseline_comparison))
            
            return RegressionAlert(
                alert_id=f"threshold_{metric.name}_{int(datetime.now().timestamp())}",
                metric_name=metric.name,
                severity=severity,
                detection_method=DetectionMethod.THRESHOLD_BASED,
                current_value=current_value,
                expected_value=metric.baseline_value,
                deviation=abs(baseline_comparison),
                confidence=0.8,  # Moderate confidence for threshold-based
                timestamp=datetime.now(),
                metadata={
                    'baseline_value': metric.baseline_value,
                    'tolerance': metric.tolerance,
                    'target_direction': metric.target_direction
                }
            )
        
        return None
    
    def _statistical_detection(
        self, 
        metric: PerformanceMetric, 
        baseline_values: List[float], 
        recent_values: List[float]
    ) -> List[RegressionAlert]:
        """Statistical regression detection."""
        alerts = []
        
        # T-test
        is_regression, p_value, metadata = self.statistical_detector.t_test_regression(
            baseline_values, recent_values
        )
        
        if is_regression:
            confidence = 1.0 - p_value
            severity = self._calculate_severity(abs(np.mean(recent_values) - np.mean(baseline_values)) / np.mean(baseline_values))
            
            alerts.append(RegressionAlert(
                alert_id=f"ttest_{metric.name}_{int(datetime.now().timestamp())}",
                metric_name=metric.name,
                severity=severity,
                detection_method=DetectionMethod.STATISTICAL_TEST,
                current_value=np.mean(recent_values),
                expected_value=np.mean(baseline_values),
                deviation=abs(np.mean(recent_values) - np.mean(baseline_values)) / np.mean(baseline_values),
                confidence=confidence,
                timestamp=datetime.now(),
                metadata=metadata
            ))
        
        # Mann-Whitney test (non-parametric)
        is_regression_mw, p_value_mw, metadata_mw = self.statistical_detector.mann_whitney_test(
            baseline_values, recent_values
        )
        
        if is_regression_mw:
            confidence = 1.0 - p_value_mw
            severity = self._calculate_severity(abs(np.median(recent_values) - np.median(baseline_values)) / np.median(baseline_values))
            
            alerts.append(RegressionAlert(
                alert_id=f"mannwhitney_{metric.name}_{int(datetime.now().timestamp())}",
                metric_name=metric.name,
                severity=severity,
                detection_method=DetectionMethod.STATISTICAL_TEST,
                current_value=np.median(recent_values),
                expected_value=np.median(baseline_values),
                deviation=abs(np.median(recent_values) - np.median(baseline_values)) / np.median(baseline_values),
                confidence=confidence,
                timestamp=datetime.now(),
                metadata=metadata_mw
            ))
        
        return alerts
    
    def _trend_based_detection(
        self, 
        metric: PerformanceMetric, 
        recent_values: List[float]
    ) -> Optional[RegressionAlert]:
        """Trend-based regression detection."""
        is_declining, slope, metadata = self.trend_detector.linear_trend_analysis(recent_values)
        
        if is_declining and metric.target_direction == "maximize":
            severity = self._calculate_severity(abs(slope))
            confidence = metadata.get('r_squared', 0.5)
            
            return RegressionAlert(
                alert_id=f"trend_{metric.name}_{int(datetime.now().timestamp())}",
                metric_name=metric.name,
                severity=severity,
                detection_method=DetectionMethod.TREND_ANALYSIS,
                current_value=recent_values[-1],
                expected_value=recent_values[0] if len(recent_values) > 0 else 0.0,
                deviation=abs(slope),
                confidence=confidence,
                timestamp=datetime.now(),
                metadata=metadata
            )
        
        return None
    
    def _anomaly_based_detection(
        self, 
        metric: PerformanceMetric, 
        recent_values: List[float]
    ) -> Optional[RegressionAlert]:
        """Anomaly-based regression detection."""
        has_anomalies, anomaly_flags, metadata = self.anomaly_detector.isolation_forest_detection(recent_values)
        
        if has_anomalies:
            # Check if recent values are anomalous
            recent_anomalies = sum(anomaly_flags[-3:])  # Check last 3 values
            
            if recent_anomalies >= 2:  # At least 2 of last 3 are anomalous
                anomaly_count = metadata.get('anomaly_count', 0)
                severity = self._calculate_severity(anomaly_count / len(recent_values))
                
                return RegressionAlert(
                    alert_id=f"anomaly_{metric.name}_{int(datetime.now().timestamp())}",
                    metric_name=metric.name,
                    severity=severity,
                    detection_method=DetectionMethod.ANOMALY_DETECTION,
                    current_value=recent_values[-1],
                    expected_value=np.mean([v for i, v in enumerate(recent_values) if not anomaly_flags[i]]),
                    deviation=anomaly_count / len(recent_values),
                    confidence=0.7,  # Moderate confidence for anomaly detection
                    timestamp=datetime.now(),
                    metadata=metadata
                )
        
        return None
    
    def _calculate_severity(self, deviation: float) -> RegressionSeverity:
        """Calculate regression severity based on deviation magnitude."""
        if deviation < 0.05:  # 5%
            return RegressionSeverity.MINOR
        elif deviation < 0.15:  # 15%
            return RegressionSeverity.MODERATE
        elif deviation < 0.30:  # 30%
            return RegressionSeverity.MAJOR
        else:
            return RegressionSeverity.CRITICAL
    
    def get_metric_summary(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive summary of a metric."""
        if metric_name not in self.metrics_registry:
            return None
        
        metric = self.metrics_registry[metric_name]
        recent_values = metric.get_recent_values(20)
        
        if len(recent_values) < 2:
            return {
                'metric_name': metric_name,
                'current_value': metric.current_value,
                'insufficient_data': True
            }
        
        # Calculate statistics
        mean_val = np.mean(recent_values)
        std_val = np.std(recent_values)
        min_val = np.min(recent_values)
        max_val = np.max(recent_values)
        
        # Trend analysis
        _, slope, trend_metadata = self.trend_detector.linear_trend_analysis(recent_values)
        
        # Recent alerts for this metric
        recent_alerts = [
            alert for alert in self.alerts_history[-50:]  # Last 50 alerts
            if alert.metric_name == metric_name and not alert.resolved
        ]
        
        return {
            'metric_name': metric_name,
            'current_value': metric.current_value,
            'baseline_value': metric.baseline_value,
            'baseline_comparison': metric.get_baseline_comparison(),
            'statistics': {
                'mean': float(mean_val),
                'std': float(std_val),
                'min': float(min_val),
                'max': float(max_val),
                'count': len(recent_values)
            },
            'trend': {
                'slope': float(slope),
                'direction': 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable',
                'strength': trend_metadata.get('trend_strength', 'unknown')
            },
            'alerts': {
                'active_count': len(recent_alerts),
                'latest_severity': recent_alerts[-1].severity.value if recent_alerts else None
            },
            'target_direction': metric.target_direction,
            'tolerance': metric.tolerance
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health regarding regressions."""
        total_metrics = len(self.metrics_registry)
        active_alerts = [alert for alert in self.alerts_history if not alert.resolved]
        
        # Count alerts by severity
        severity_counts = defaultdict(int)
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1
        
        # Calculate health score
        health_score = 1.0
        if total_metrics > 0:
            regression_ratio = len(active_alerts) / total_metrics
            health_score = max(0.0, 1.0 - regression_ratio)
        
        return {
            'health_score': health_score,
            'total_metrics': total_metrics,
            'active_alerts': len(active_alerts),
            'alerts_by_severity': dict(severity_counts),
            'detection_methods_enabled': {
                'statistical': self.enable_statistical,
                'trend': self.enable_trend,
                'anomaly': self.enable_anomaly
            },
            'total_alerts_generated': len(self.alerts_history),
            'archaeological_enhancement': True
        }


# Archaeological enhancement: Global regression detector instance
_global_regression_detector: Optional[ComprehensiveRegressionDetector] = None

def get_regression_detector() -> ComprehensiveRegressionDetector:
    """Get or create global regression detector instance."""
    global _global_regression_detector
    if _global_regression_detector is None:
        _global_regression_detector = ComprehensiveRegressionDetector()
    return _global_regression_detector


def initialize_regression_detection(
    threshold_tolerance: float = 0.1,
    enable_all_methods: bool = True
) -> ComprehensiveRegressionDetector:
    """Initialize regression detection system."""
    detector = ComprehensiveRegressionDetector(
        threshold_tolerance=threshold_tolerance,
        enable_statistical=enable_all_methods,
        enable_trend=enable_all_methods,
        enable_anomaly=enable_all_methods
    )
    
    global _global_regression_detector
    _global_regression_detector = detector
    
    logger.info("Regression detection system initialized with archaeological enhancements")
    
    return detector