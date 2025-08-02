"""Real-time Resource Monitor for Evolution-Aware Systems"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

from .device_profiler import DeviceProfiler, ResourceSnapshot

logger = logging.getLogger(__name__)


class MonitoringMode(Enum):
    """Resource monitoring modes"""
    PASSIVE = "passive"      # Basic monitoring
    ACTIVE = "active"        # Active monitoring with predictions
    EVOLUTION = "evolution"  # Evolution-optimized monitoring
    EMERGENCY = "emergency"  # High-frequency emergency monitoring

@dataclass
class ResourceTrend:
    """Resource usage trend analysis"""
    metric_name: str
    current_value: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0-1, how strong the trend is
    predicted_value_5min: Optional[float] = None
    predicted_value_15min: Optional[float] = None
    confidence: float = 0.5  # Prediction confidence


class ResourceMonitor:
    """Advanced resource monitor with trend analysis and predictions"""
    
    def __init__(self, device_profiler: DeviceProfiler):
        self.device_profiler = device_profiler
        self.monitoring_mode = MonitoringMode.PASSIVE
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Trend analysis
        self.trend_window_size = 20  # Number of snapshots for trend analysis
        self.trends: Dict[str, ResourceTrend] = {}
        
        # Adaptive monitoring intervals
        self.base_interval = 5.0  # Base monitoring interval
        self.current_interval = self.base_interval
        self.interval_adjustments = {
            MonitoringMode.PASSIVE: 1.0,     # 5s
            MonitoringMode.ACTIVE: 0.6,      # 3s
            MonitoringMode.EVOLUTION: 0.4,   # 2s
            MonitoringMode.EMERGENCY: 0.2    # 1s
        }
        
        # Event detection
        self.event_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        self.event_thresholds = {
            'memory_spike': 20.0,      # % increase in 1 minute
            'cpu_spike': 30.0,         # % increase in 1 minute
            'battery_drop': 5.0,       # % drop in 5 minutes
            'thermal_rise': 10.0,      # °C increase in 2 minutes
        }
        
        # Prediction models (simple linear regression)
        self.prediction_weights: Dict[str, List[float]] = {}
        
        # Statistics
        self.stats = {
            'monitoring_cycles': 0,
            'trends_detected': 0,
            'events_detected': 0,
            'predictions_made': 0,
            'mode_changes': 0
        }
        
    async def start_monitoring(self, mode: MonitoringMode = MonitoringMode.PASSIVE):
        """Start resource monitoring"""
        if self.monitoring_active:
            logger.warning("Resource monitoring already active")
            return
            
        self.monitoring_mode = mode
        self.monitoring_active = True
        self._update_monitoring_interval()
        
        # Start device profiler if not already running
        if not self.device_profiler.monitoring_active:
            self.device_profiler.start_monitoring()
            
        # Start monitoring loop
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info(f"Resource monitoring started in {mode.value} mode")
        
    async def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Resource monitoring stopped")
        
    def set_monitoring_mode(self, mode: MonitoringMode):
        """Change monitoring mode"""
        if mode != self.monitoring_mode:
            self.monitoring_mode = mode
            self._update_monitoring_interval()
            self.stats['mode_changes'] += 1
            logger.info(f"Monitoring mode changed to {mode.value}")
            
    def _update_monitoring_interval(self):
        """Update monitoring interval based on mode"""
        multiplier = self.interval_adjustments[self.monitoring_mode]
        self.current_interval = self.base_interval * multiplier
        
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Get latest snapshot
                snapshot = self.device_profiler.current_snapshot
                if snapshot:
                    await self._analyze_snapshot(snapshot)
                    
                self.stats['monitoring_cycles'] += 1
                await asyncio.sleep(self.current_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.current_interval)
                
    async def _analyze_snapshot(self, snapshot: ResourceSnapshot):
        """Analyze resource snapshot for trends and events"""
        # Update trend analysis
        await self._update_trends(snapshot)
        
        # Detect events
        await self._detect_events(snapshot)
        
        # Make predictions if in active modes
        if self.monitoring_mode in [MonitoringMode.ACTIVE, MonitoringMode.EVOLUTION]:
            await self._update_predictions()
            
    async def _update_trends(self, snapshot: ResourceSnapshot):
        """Update trend analysis for key metrics"""
        metrics = {
            'memory_percent': snapshot.memory_percent,
            'cpu_percent': snapshot.cpu_percent,
            'battery_percent': snapshot.battery_percent,
            'cpu_temp': snapshot.cpu_temp,
        }
        
        # Get recent snapshots for trend analysis
        recent_snapshots = self.device_profiler.snapshots[-self.trend_window_size:]
        
        if len(recent_snapshots) < 3:
            return  # Need at least 3 points for trend
            
        for metric_name, current_value in metrics.items():
            if current_value is None:
                continue
                
            # Extract metric values from recent snapshots
            values = []
            timestamps = []
            
            for snap in recent_snapshots:
                metric_value = getattr(snap, metric_name, None)
                if metric_value is not None:
                    values.append(metric_value)
                    timestamps.append(snap.timestamp)
                    
            if len(values) < 3:
                continue
                
            # Calculate trend
            trend = self._calculate_trend(values, timestamps)
            self.trends[metric_name] = trend
            
            if trend.trend_direction != 'stable':
                self.stats['trends_detected'] += 1
                
    def _calculate_trend(self, values: List[float], timestamps: List[float]) -> ResourceTrend:
        """Calculate trend for a metric"""
        if len(values) < 2:
            return ResourceTrend("unknown", 0.0, "stable", 0.0)
            
        # Simple linear regression
        n = len(values)
        x = list(range(n))  # Time indices
        y = values
        
        # Calculate slope
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
            
        # Determine trend direction and strength
        if abs(slope) < 0.1:
            direction = "stable"
            strength = 0.0
        elif slope > 0:
            direction = "increasing"
            strength = min(abs(slope) / 5.0, 1.0)  # Normalize to 0-1
        else:
            direction = "decreasing"
            strength = min(abs(slope) / 5.0, 1.0)
            
        # Make predictions
        current_value = values[-1]
        predicted_5min = None
        predicted_15min = None
        confidence = 0.5
        
        if strength > 0.3:  # Only predict if trend is strong enough
            # Predict future values (simple linear extrapolation)
            time_points_5min = 5 * 60 / self.current_interval  # 5 minutes in intervals
            time_points_15min = 15 * 60 / self.current_interval  # 15 minutes in intervals
            
            predicted_5min = current_value + slope * time_points_5min
            predicted_15min = current_value + slope * time_points_15min
            
            # Clip predictions to reasonable ranges
            if 'percent' in timestamps[0] if timestamps else "":
                predicted_5min = max(0, min(100, predicted_5min))
                predicted_15min = max(0, min(100, predicted_15min))
                
            confidence = min(strength * 2, 1.0)  # Higher strength = higher confidence
            
        return ResourceTrend(
            metric_name=timestamps[0] if timestamps else "unknown",
            current_value=current_value,
            trend_direction=direction,
            trend_strength=strength,
            predicted_value_5min=predicted_5min,
            predicted_value_15min=predicted_15min,
            confidence=confidence
        )
        
    async def _detect_events(self, snapshot: ResourceSnapshot):
        """Detect resource events"""
        events_detected = []
        
        # Get previous snapshots for comparison
        recent_snapshots = self.device_profiler.snapshots[-10:]  # Last 10 snapshots
        
        if len(recent_snapshots) < 2:
            return
            
        # Memory spike detection
        memory_spike = self._detect_spike(
            recent_snapshots,
            'memory_percent',
            self.event_thresholds['memory_spike'],
            time_window=60  # 1 minute
        )
        if memory_spike:
            events_detected.append(('memory_spike', memory_spike))
            
        # CPU spike detection
        cpu_spike = self._detect_spike(
            recent_snapshots,
            'cpu_percent',
            self.event_thresholds['cpu_spike'],
            time_window=60
        )
        if cpu_spike:
            events_detected.append(('cpu_spike', cpu_spike))
            
        # Battery drop detection
        if snapshot.battery_percent is not None:
            battery_drop = self._detect_drop(
                recent_snapshots,
                'battery_percent',
                self.event_thresholds['battery_drop'],
                time_window=300  # 5 minutes
            )
            if battery_drop:
                events_detected.append(('battery_drop', battery_drop))
                
        # Thermal rise detection
        if snapshot.cpu_temp is not None:
            thermal_rise = self._detect_spike(
                recent_snapshots,
                'cpu_temp',
                self.event_thresholds['thermal_rise'],
                time_window=120  # 2 minutes
            )
            if thermal_rise:
                events_detected.append(('thermal_rise', thermal_rise))
                
        # Trigger event callbacks
        for event_type, event_data in events_detected:
            self.stats['events_detected'] += 1
            await self._trigger_event(event_type, event_data)
            
    def _detect_spike(self, snapshots: List[ResourceSnapshot], metric: str,
                     threshold: float, time_window: float) -> Optional[Dict[str, Any]]:
        """Detect spikes in a metric"""
        current_time = time.time()
        
        # Filter snapshots within time window
        recent_snapshots = [
            s for s in snapshots
            if current_time - s.timestamp <= time_window
        ]
        
        if len(recent_snapshots) < 2:
            return None
            
        # Get metric values
        values = []
        for snap in recent_snapshots:
            value = getattr(snap, metric, None)
            if value is not None:
                values.append(value)
                
        if len(values) < 2:
            return None
            
        # Check for spike
        min_value = min(values)
        max_value = max(values)
        spike_magnitude = max_value - min_value
        
        if spike_magnitude >= threshold:
            return {
                'metric': metric,
                'spike_magnitude': spike_magnitude,
                'min_value': min_value,
                'max_value': max_value,
                'time_window': time_window,
                'snapshots_analyzed': len(recent_snapshots)
            }
            
        return None
        
    def _detect_drop(self, snapshots: List[ResourceSnapshot], metric: str,
                    threshold: float, time_window: float) -> Optional[Dict[str, Any]]:
        """Detect drops in a metric"""
        current_time = time.time()
        
        # Filter snapshots within time window
        recent_snapshots = [
            s for s in snapshots
            if current_time - s.timestamp <= time_window
        ]
        
        if len(recent_snapshots) < 2:
            return None
            
        # Get metric values
        values = []
        for snap in recent_snapshots:
            value = getattr(snap, metric, None)
            if value is not None:
                values.append(value)
                
        if len(values) < 2:
            return None
            
        # Check for drop
        max_value = max(values)
        min_value = min(values)
        drop_magnitude = max_value - min_value
        
        if drop_magnitude >= threshold:
            return {
                'metric': metric,
                'drop_magnitude': drop_magnitude,
                'max_value': max_value,
                'min_value': min_value,
                'time_window': time_window,
                'snapshots_analyzed': len(recent_snapshots)
            }
            
        return None
        
    async def _update_predictions(self):
        """Update resource predictions"""
        for metric_name, trend in self.trends.items():
            if trend.confidence > 0.5:  # Only use confident predictions
                self.stats['predictions_made'] += 1
                
    async def _trigger_event(self, event_type: str, event_data: Dict[str, Any]):
        """Trigger event callbacks"""
        logger.info(f"Resource event detected: {event_type} - {event_data}")
        
        for callback in self.event_callbacks:
            try:
                await callback(event_type, event_data)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
                
    def register_event_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Register event callback"""
        self.event_callbacks.append(callback)
        
    def get_resource_prediction(self, metric: str, time_horizon_minutes: int = 5) -> Optional[float]:
        """Get resource prediction for a metric"""
        if metric not in self.trends:
            return None
            
        trend = self.trends[metric]
        
        if time_horizon_minutes <= 5 and trend.predicted_value_5min is not None:
            return trend.predicted_value_5min
        elif time_horizon_minutes <= 15 and trend.predicted_value_15min is not None:
            return trend.predicted_value_15min
        else:
            # Extrapolate based on trend
            if trend.trend_direction == 'stable':
                return trend.current_value
            elif trend.confidence > 0.3:
                # Simple linear extrapolation
                intervals_ahead = (time_horizon_minutes * 60) / self.current_interval
                slope = trend.trend_strength * (1 if trend.trend_direction == 'increasing' else -1)
                return trend.current_value + slope * intervals_ahead
                
        return None
        
    def get_evolution_readiness_forecast(self, look_ahead_minutes: int = 15) -> Dict[str, Any]:
        """Forecast evolution readiness"""
        current_snapshot = self.device_profiler.current_snapshot
        if not current_snapshot:
            return {'status': 'no_data'}
            
        # Current suitability
        current_suitability = current_snapshot.evolution_suitability_score
        
        # Predict future suitability
        predicted_memory = self.get_resource_prediction('memory_percent', look_ahead_minutes)
        predicted_cpu = self.get_resource_prediction('cpu_percent', look_ahead_minutes)
        predicted_battery = self.get_resource_prediction('battery_percent', look_ahead_minutes)
        predicted_temp = self.get_resource_prediction('cpu_temp', look_ahead_minutes)
        
        # Calculate predicted suitability
        predicted_suitability = current_suitability
        confidence = 0.5
        
        if predicted_memory is not None and predicted_cpu is not None:
            # Recalculate suitability score with predictions
            score = 1.0
            
            if predicted_memory > 80:
                score -= (predicted_memory - 80) / 20 * 0.3
            if predicted_cpu > 70:
                score -= (predicted_cpu - 70) / 30 * 0.2
            if predicted_battery is not None and predicted_battery < 30:
                score -= (30 - predicted_battery) / 30 * 0.3
            if predicted_temp is not None and predicted_temp > 75:
                score -= (predicted_temp - 75) / 25 * 0.2
                
            predicted_suitability = max(0.0, min(1.0, score))
            
            # Calculate confidence based on trend strengths
            memory_trend = self.trends.get('memory_percent')
            cpu_trend = self.trends.get('cpu_percent')
            
            if memory_trend and cpu_trend:
                confidence = (memory_trend.confidence + cpu_trend.confidence) / 2
                
        return {
            'current_suitability': current_suitability,
            'predicted_suitability': predicted_suitability,
            'confidence': confidence,
            'look_ahead_minutes': look_ahead_minutes,
            'predictions': {
                'memory_percent': predicted_memory,
                'cpu_percent': predicted_cpu,
                'battery_percent': predicted_battery,
                'cpu_temp': predicted_temp
            },
            'recommendation': 'suitable' if predicted_suitability > 0.6 else 'not_suitable'
        }
        
    def get_resource_trends(self) -> Dict[str, ResourceTrend]:
        """Get current resource trends"""
        return self.trends.copy()
        
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            **self.stats,
            'monitoring_active': self.monitoring_active,
            'monitoring_mode': self.monitoring_mode.value,
            'current_interval': self.current_interval,
            'trends_tracked': len(self.trends),
            'event_callbacks': len(self.event_callbacks)
        }

