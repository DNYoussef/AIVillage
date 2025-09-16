"""
SafetyMonitor - Real-time safety monitoring agent for ADAS systems

Specialized agent providing continuous safety monitoring with real-time
violation detection and automated safety responses for ISO 26262 compliance.
"""

import asyncio
import logging
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

from ..config.adas_config import ADASConfig, ASILLevel
from ..safety.safety_manager import SafetyManager, SafetyViolation, SafetyViolationType

class MonitoringMode(Enum):
    """Safety monitoring modes"""
    NORMAL = "normal"
    ENHANCED = "enhanced"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"

class SafetyChannel(Enum):
    """Safety monitoring channels"""
    PERCEPTION = "perception"
    PREDICTION = "prediction"
    PLANNING = "planning"
    SENSOR_FUSION = "sensor_fusion"
    SYSTEM_HEALTH = "system_health"

@dataclass
class MonitoringAlert:
    """Real-time monitoring alert"""
    alert_id: str
    channel: SafetyChannel
    severity: ASILLevel
    timestamp: float
    message: str
    metric_values: Dict[str, float]
    threshold_violated: str
    recovery_action: str

@dataclass
class SafetyMetrics:
    """Real-time safety metrics"""
    timestamp: float
    channel: SafetyChannel
    latency_ms: float
    confidence_score: float
    error_rate: float
    throughput_hz: float
    resource_usage: float
    safety_margin: float

class SafetyMonitor:
    """
    Real-time safety monitoring agent for ADAS systems

    Provides continuous monitoring of all ADAS components with real-time
    violation detection, alerting, and automated safety responses.
    """

    def __init__(self, config: ADASConfig, safety_manager: SafetyManager):
        self.config = config
        self.safety_manager = safety_manager
        self.logger = logging.getLogger(__name__)

        # Monitoring state
        self.monitoring_mode = MonitoringMode.NORMAL
        self.monitoring_enabled = False
        self.alert_queue = queue.Queue(maxsize=1000)

        # Monitoring thresholds
        self.thresholds = {
            SafetyChannel.PERCEPTION: {
                'latency_ms': config.latency.perception_max_ms,
                'min_confidence': config.safety.min_detection_confidence,
                'max_error_rate': 0.05,
                'min_throughput_hz': 20.0
            },
            SafetyChannel.PREDICTION: {
                'latency_ms': config.latency.prediction_max_ms,
                'min_confidence': 0.8,
                'max_error_rate': 0.10,
                'min_throughput_hz': 10.0
            },
            SafetyChannel.PLANNING: {
                'latency_ms': config.latency.planning_max_ms,
                'min_confidence': 0.85,
                'max_error_rate': 0.02,
                'min_throughput_hz': 5.0
            },
            SafetyChannel.SENSOR_FUSION: {
                'latency_ms': config.latency.sensor_fusion_max_ms,
                'min_confidence': 0.90,
                'max_error_rate': 0.01,
                'min_throughput_hz': 30.0
            }
        }

        # Real-time metrics tracking
        self.metrics_history: Dict[SafetyChannel, List[SafetyMetrics]] = {
            channel: [] for channel in SafetyChannel
        }

        # Performance tracking
        self.performance_counters = {
            'alerts_generated': 0,
            'violations_detected': 0,
            'emergency_responses': 0,
            'false_alarms': 0
        }

        # Alert handlers
        self.alert_handlers: Dict[str, Callable] = {
            'latency_exceeded': self._handle_latency_violation,
            'confidence_low': self._handle_confidence_violation,
            'error_rate_high': self._handle_error_rate_violation,
            'throughput_low': self._handle_throughput_violation,
            'resource_exhaustion': self._handle_resource_violation
        }

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.running = False
        self.monitoring_thread = None
        self.alert_processing_thread = None

        # Watchdog timers
        self.watchdog_timers = {}
        self.last_heartbeats = {}

    async def start(self) -> bool:
        """Start the safety monitoring agent"""
        try:
            self.logger.info("Starting SafetyMonitor...")

            # Initialize watchdog timers
            self._initialize_watchdogs()

            # Start monitoring threads
            self.running = True
            self.monitoring_enabled = True

            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()

            self.alert_processing_thread = threading.Thread(target=self._alert_processing_loop, daemon=True)
            self.alert_processing_thread.start()

            self.monitoring_mode = MonitoringMode.NORMAL
            self.logger.info("SafetyMonitor started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start SafetyMonitor: {e}")
            return False

    def _initialize_watchdogs(self) -> None:
        """Initialize watchdog timers for all monitored channels"""
        current_time = time.time()
        for channel in SafetyChannel:
            self.watchdog_timers[channel] = current_time
            self.last_heartbeats[channel] = current_time

    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.running and self.monitoring_enabled:
            try:
                # Check watchdog timers
                self._check_watchdog_timers()

                # Analyze metric trends
                self._analyze_metric_trends()

                # Check system health indicators
                self._check_system_health()

                # Monitor at 50 Hz for real-time response
                time.sleep(0.02)

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                self._handle_monitoring_error(e)

    def _alert_processing_loop(self) -> None:
        """Alert processing loop"""
        while self.running:
            try:
                # Process alerts from queue
                try:
                    alert = self.alert_queue.get(timeout=0.1)
                    await self._process_alert(alert)
                    self.alert_queue.task_done()
                except queue.Empty:
                    continue

            except Exception as e:
                self.logger.error(f"Alert processing error: {e}")

    async def monitor_perception(self, latency_ms: float, confidence: float,
                               detection_count: int, timestamp: float) -> None:
        """Monitor perception system metrics"""
        channel = SafetyChannel.PERCEPTION

        # Update heartbeat
        self.last_heartbeats[channel] = timestamp

        # Create metrics record
        metrics = SafetyMetrics(
            timestamp=timestamp,
            channel=channel,
            latency_ms=latency_ms,
            confidence_score=confidence,
            error_rate=self._compute_error_rate(channel),
            throughput_hz=self._compute_throughput(channel, timestamp),
            resource_usage=self._get_resource_usage(),
            safety_margin=self._compute_safety_margin(channel, latency_ms, confidence)
        )

        # Store metrics
        self._store_metrics(metrics)

        # Check thresholds
        await self._check_thresholds(metrics)

    async def monitor_prediction(self, latency_ms: float, confidence: float,
                               trajectory_count: int, timestamp: float) -> None:
        """Monitor prediction system metrics"""
        channel = SafetyChannel.PREDICTION

        # Update heartbeat
        self.last_heartbeats[channel] = timestamp

        # Create metrics record
        metrics = SafetyMetrics(
            timestamp=timestamp,
            channel=channel,
            latency_ms=latency_ms,
            confidence_score=confidence,
            error_rate=self._compute_error_rate(channel),
            throughput_hz=self._compute_throughput(channel, timestamp),
            resource_usage=self._get_resource_usage(),
            safety_margin=self._compute_safety_margin(channel, latency_ms, confidence)
        )

        # Store metrics
        self._store_metrics(metrics)

        # Check thresholds
        await self._check_thresholds(metrics)

    async def monitor_planning(self, latency_ms: float, path_quality: float,
                             constraint_violations: int, timestamp: float) -> None:
        """Monitor planning system metrics"""
        channel = SafetyChannel.PLANNING

        # Update heartbeat
        self.last_heartbeats[channel] = timestamp

        # Create metrics record
        metrics = SafetyMetrics(
            timestamp=timestamp,
            channel=channel,
            latency_ms=latency_ms,
            confidence_score=path_quality,
            error_rate=constraint_violations / 10.0,  # Normalize constraint violations
            throughput_hz=self._compute_throughput(channel, timestamp),
            resource_usage=self._get_resource_usage(),
            safety_margin=self._compute_safety_margin(channel, latency_ms, path_quality)
        )

        # Store metrics
        self._store_metrics(metrics)

        # Check thresholds
        await self._check_thresholds(metrics)

    async def monitor_sensor_fusion(self, latency_ms: float, fusion_quality: float,
                                  sensor_count: int, timestamp: float) -> None:
        """Monitor sensor fusion metrics"""
        channel = SafetyChannel.SENSOR_FUSION

        # Update heartbeat
        self.last_heartbeats[channel] = timestamp

        # Create metrics record
        metrics = SafetyMetrics(
            timestamp=timestamp,
            channel=channel,
            latency_ms=latency_ms,
            confidence_score=fusion_quality,
            error_rate=self._compute_error_rate(channel),
            throughput_hz=self._compute_throughput(channel, timestamp),
            resource_usage=self._get_resource_usage(),
            safety_margin=self._compute_safety_margin(channel, latency_ms, fusion_quality)
        )

        # Store metrics
        self._store_metrics(metrics)

        # Check thresholds
        await self._check_thresholds(metrics)

    def _store_metrics(self, metrics: SafetyMetrics) -> None:
        """Store metrics in history buffer"""
        history = self.metrics_history[metrics.channel]
        history.append(metrics)

        # Keep only recent metrics (last 1000 samples)
        if len(history) > 1000:
            history.pop(0)

    async def _check_thresholds(self, metrics: SafetyMetrics) -> None:
        """Check if metrics violate safety thresholds"""
        channel = metrics.channel
        thresholds = self.thresholds.get(channel, {})

        alerts = []

        # Check latency threshold
        if metrics.latency_ms > thresholds.get('latency_ms', float('inf')):
            alert = MonitoringAlert(
                alert_id=f"LATENCY_{channel.value}_{int(time.time())}",
                channel=channel,
                severity=ASILLevel.C,
                timestamp=metrics.timestamp,
                message=f"{channel.value} latency {metrics.latency_ms:.2f}ms exceeds threshold {thresholds['latency_ms']}ms",
                metric_values={'latency_ms': metrics.latency_ms},
                threshold_violated='latency_ms',
                recovery_action='reduce_computation_load'
            )
            alerts.append(alert)

        # Check confidence threshold
        min_confidence = thresholds.get('min_confidence', 0.0)
        if metrics.confidence_score < min_confidence:
            alert = MonitoringAlert(
                alert_id=f"CONFIDENCE_{channel.value}_{int(time.time())}",
                channel=channel,
                severity=ASILLevel.B,
                timestamp=metrics.timestamp,
                message=f"{channel.value} confidence {metrics.confidence_score:.3f} below threshold {min_confidence}",
                metric_values={'confidence': metrics.confidence_score},
                threshold_violated='min_confidence',
                recovery_action='increase_redundancy'
            )
            alerts.append(alert)

        # Check error rate threshold
        max_error_rate = thresholds.get('max_error_rate', 1.0)
        if metrics.error_rate > max_error_rate:
            alert = MonitoringAlert(
                alert_id=f"ERROR_RATE_{channel.value}_{int(time.time())}",
                channel=channel,
                severity=ASILLevel.C,
                timestamp=metrics.timestamp,
                message=f"{channel.value} error rate {metrics.error_rate:.3f} exceeds threshold {max_error_rate}",
                metric_values={'error_rate': metrics.error_rate},
                threshold_violated='max_error_rate',
                recovery_action='restart_component'
            )
            alerts.append(alert)

        # Check throughput threshold
        min_throughput = thresholds.get('min_throughput_hz', 0.0)
        if metrics.throughput_hz < min_throughput:
            alert = MonitoringAlert(
                alert_id=f"THROUGHPUT_{channel.value}_{int(time.time())}",
                channel=channel,
                severity=ASILLevel.B,
                timestamp=metrics.timestamp,
                message=f"{channel.value} throughput {metrics.throughput_hz:.1f}Hz below threshold {min_throughput}Hz",
                metric_values={'throughput_hz': metrics.throughput_hz},
                threshold_violated='min_throughput_hz',
                recovery_action='optimize_processing'
            )
            alerts.append(alert)

        # Queue alerts for processing
        for alert in alerts:
            try:
                self.alert_queue.put_nowait(alert)
                self.performance_counters['alerts_generated'] += 1
            except queue.Full:
                self.logger.warning("Alert queue full, dropping alert")

    async def _process_alert(self, alert: MonitoringAlert) -> None:
        """Process monitoring alert"""
        try:
            self.logger.warning(f"Safety alert: {alert.message}")

            # Determine if this is a safety violation
            if alert.severity in [ASILLevel.C, ASILLevel.D]:
                # Convert to safety violation
                violation = self._alert_to_violation(alert)
                await self.safety_manager._process_violations([violation])
                self.performance_counters['violations_detected'] += 1

            # Execute recovery action
            handler = self.alert_handlers.get(alert.recovery_action)
            if handler:
                await handler(alert)

        except Exception as e:
            self.logger.error(f"Alert processing failed: {e}")

    def _alert_to_violation(self, alert: MonitoringAlert) -> SafetyViolation:
        """Convert monitoring alert to safety violation"""
        violation_type_map = {
            'latency_ms': SafetyViolationType.LATENCY_EXCEEDED,
            'min_confidence': SafetyViolationType.CONFIDENCE_LOW,
            'max_error_rate': SafetyViolationType.COMPUTATION_ERROR,
            'min_throughput_hz': SafetyViolationType.CONSTRAINT_VIOLATED
        }

        violation_type = violation_type_map.get(
            alert.threshold_violated,
            SafetyViolationType.CONSTRAINT_VIOLATED
        )

        return SafetyViolation(
            violation_type=violation_type,
            severity=alert.severity,
            timestamp=alert.timestamp,
            description=alert.message,
            affected_components=[alert.channel.value],
            recommended_action=alert.recovery_action,
            auto_recovery_possible=True,
            violation_id=alert.alert_id
        )

    async def _handle_latency_violation(self, alert: MonitoringAlert) -> None:
        """Handle latency violation"""
        self.logger.warning(f"Handling latency violation for {alert.channel.value}")
        # In real implementation, would reduce computation complexity

    async def _handle_confidence_violation(self, alert: MonitoringAlert) -> None:
        """Handle confidence violation"""
        self.logger.warning(f"Handling confidence violation for {alert.channel.value}")
        # In real implementation, would increase sensor redundancy

    async def _handle_error_rate_violation(self, alert: MonitoringAlert) -> None:
        """Handle error rate violation"""
        self.logger.warning(f"Handling error rate violation for {alert.channel.value}")
        # In real implementation, would restart component

    async def _handle_throughput_violation(self, alert: MonitoringAlert) -> None:
        """Handle throughput violation"""
        self.logger.warning(f"Handling throughput violation for {alert.channel.value}")
        # In real implementation, would optimize processing

    async def _handle_resource_violation(self, alert: MonitoringAlert) -> None:
        """Handle resource exhaustion"""
        self.logger.critical(f"Handling resource exhaustion for {alert.channel.value}")
        # In real implementation, would free resources or scale down

    def _check_watchdog_timers(self) -> None:
        """Check watchdog timers for component health"""
        current_time = time.time()

        for channel, last_heartbeat in self.last_heartbeats.items():
            time_since_heartbeat = current_time - last_heartbeat

            # Component-specific timeout thresholds
            timeout_thresholds = {
                SafetyChannel.PERCEPTION: 0.5,      # 500ms
                SafetyChannel.PREDICTION: 1.0,      # 1s
                SafetyChannel.PLANNING: 2.0,        # 2s
                SafetyChannel.SENSOR_FUSION: 0.2,   # 200ms
                SafetyChannel.SYSTEM_HEALTH: 5.0    # 5s
            }

            threshold = timeout_thresholds.get(channel, 1.0)

            if time_since_heartbeat > threshold:
                self.logger.error(f"Watchdog timeout for {channel.value}: {time_since_heartbeat:.2f}s")
                self._handle_watchdog_timeout(channel, time_since_heartbeat)

    def _handle_watchdog_timeout(self, channel: SafetyChannel, timeout_duration: float) -> None:
        """Handle watchdog timeout"""
        # Create timeout alert
        alert = MonitoringAlert(
            alert_id=f"TIMEOUT_{channel.value}_{int(time.time())}",
            channel=channel,
            severity=ASILLevel.D,
            timestamp=time.time(),
            message=f"{channel.value} watchdog timeout: {timeout_duration:.2f}s",
            metric_values={'timeout_duration': timeout_duration},
            threshold_violated='watchdog_timeout',
            recovery_action='restart_component'
        )

        try:
            self.alert_queue.put_nowait(alert)
            self.performance_counters['violations_detected'] += 1
        except queue.Full:
            self.logger.critical("Cannot queue watchdog timeout alert - queue full")

    def _analyze_metric_trends(self) -> None:
        """Analyze metric trends for predictive monitoring"""
        for channel, history in self.metrics_history.items():
            if len(history) < 10:  # Need sufficient history
                continue

            recent_metrics = history[-10:]

            # Analyze latency trend
            latencies = [m.latency_ms for m in recent_metrics]
            if len(latencies) > 1:
                latency_trend = np.polyfit(range(len(latencies)), latencies, 1)[0]

                # If latency is increasing rapidly, generate predictive alert
                if latency_trend > 1.0:  # 1ms/sample increase
                    self.logger.warning(f"Increasing latency trend detected for {channel.value}: {latency_trend:.2f}ms/sample")

            # Analyze confidence trend
            confidences = [m.confidence_score for m in recent_metrics]
            if len(confidences) > 1:
                confidence_trend = np.polyfit(range(len(confidences)), confidences, 1)[0]

                # If confidence is decreasing rapidly, generate predictive alert
                if confidence_trend < -0.05:  # 5% decrease per sample
                    self.logger.warning(f"Decreasing confidence trend detected for {channel.value}: {confidence_trend:.3f}/sample")

    def _check_system_health(self) -> None:
        """Check overall system health indicators"""
        try:
            # Check alert queue health
            queue_usage = self.alert_queue.qsize() / self.alert_queue.maxsize
            if queue_usage > 0.8:
                self.logger.warning(f"Alert queue nearly full: {queue_usage:.1%}")

            # Check if monitoring is keeping up
            if self.monitoring_enabled and time.time() - max(self.last_heartbeats.values()) > 5.0:
                self.logger.error("Monitoring system falling behind")

        except Exception as e:
            self.logger.error(f"System health check failed: {e}")

    def _compute_error_rate(self, channel: SafetyChannel) -> float:
        """Compute error rate for channel"""
        # Simplified error rate computation
        # In real implementation, would track actual errors
        return 0.01  # 1% baseline error rate

    def _compute_throughput(self, channel: SafetyChannel, timestamp: float) -> float:
        """Compute throughput for channel"""
        history = self.metrics_history[channel]
        if len(history) < 2:
            return 0.0

        # Count samples in last second
        one_second_ago = timestamp - 1.0
        recent_samples = [m for m in history if m.timestamp > one_second_ago]
        return len(recent_samples)

    def _get_resource_usage(self) -> float:
        """Get current resource usage"""
        # Simplified resource usage
        # In real implementation, would monitor CPU, memory, GPU usage
        return 0.5  # 50% resource usage

    def _compute_safety_margin(self, channel: SafetyChannel, latency_ms: float, confidence: float) -> float:
        """Compute safety margin for current metrics"""
        thresholds = self.thresholds.get(channel, {})

        # Latency margin (0.0 = at threshold, 1.0 = well below threshold)
        latency_threshold = thresholds.get('latency_ms', float('inf'))
        latency_margin = max(0.0, 1.0 - latency_ms / latency_threshold) if latency_threshold < float('inf') else 1.0

        # Confidence margin
        min_confidence = thresholds.get('min_confidence', 0.0)
        confidence_margin = (confidence - min_confidence) / (1.0 - min_confidence) if min_confidence < 1.0 else 1.0
        confidence_margin = max(0.0, min(1.0, confidence_margin))

        # Combined safety margin
        return (latency_margin + confidence_margin) / 2.0

    def set_monitoring_mode(self, mode: MonitoringMode) -> None:
        """Set monitoring mode"""
        self.monitoring_mode = mode
        self.logger.info(f"Monitoring mode set to {mode.value}")

        # Adjust thresholds based on mode
        if mode == MonitoringMode.ENHANCED:
            # Stricter thresholds for enhanced monitoring
            for channel_thresholds in self.thresholds.values():
                channel_thresholds['latency_ms'] *= 0.8
                channel_thresholds['min_confidence'] *= 1.1

        elif mode == MonitoringMode.DEGRADED:
            # Relaxed thresholds for degraded mode
            for channel_thresholds in self.thresholds.values():
                channel_thresholds['latency_ms'] *= 1.5
                channel_thresholds['min_confidence'] *= 0.9

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'monitoring_enabled': self.monitoring_enabled,
            'monitoring_mode': self.monitoring_mode.value,
            'alert_queue_size': self.alert_queue.qsize(),
            'performance_counters': self.performance_counters.copy(),
            'last_heartbeats': {
                channel.value: heartbeat
                for channel, heartbeat in self.last_heartbeats.items()
            },
            'active_channels': len([
                channel for channel, heartbeat in self.last_heartbeats.items()
                if time.time() - heartbeat < 5.0
            ])
        }

    async def stop(self) -> None:
        """Stop the safety monitoring agent"""
        self.logger.info("Stopping SafetyMonitor...")
        self.running = False
        self.monitoring_enabled = False

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)

        if self.alert_processing_thread:
            self.alert_processing_thread.join(timeout=1.0)

        self.executor.shutdown(wait=True)
        self.logger.info("SafetyMonitor stopped")