"""Network Monitoring Service - Network condition monitoring and link detection

This service monitors network links, detects changes, and assesses link quality
for routing decisions in the Navigator system.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from statistics import mean

from ..interfaces.routing_interfaces import INetworkMonitoringService, RoutingEvent
from ..events.event_bus import get_event_bus
from ..path_policy import NetworkConditions, PathProtocol

logger = logging.getLogger(__name__)

# Bluetooth imports with fallback
try:
    import bluetooth

    BLUETOOTH_AVAILABLE = True
except ImportError:
    BLUETOOTH_AVAILABLE = False


@dataclass
class LinkQualityMetrics:
    """Link quality metrics for specific protocol"""

    latency_ms: float = 100.0
    bandwidth_mbps: float = 1.0
    packet_loss_rate: float = 0.0
    jitter_ms: float = 5.0
    availability: float = 1.0
    signal_strength: Optional[int] = None  # For wireless links
    last_updated: float = field(default_factory=time.time)


@dataclass
class LinkChangeEvent:
    """Network link change event"""

    timestamp: float
    link_type: str
    change_type: str  # "connected", "disconnected", "degraded", "improved"
    old_metrics: Optional[LinkQualityMetrics]
    new_metrics: LinkQualityMetrics
    significance_score: float  # 0.0 to 1.0, how significant the change is


class NetworkMonitoringService(INetworkMonitoringService):
    """Network condition monitoring and link detection service

    Monitors:
    - Internet connectivity (WiFi, cellular)
    - Bluetooth mesh connectivity
    - Link quality metrics (latency, bandwidth, loss)
    - Network topology changes
    - Rapid change detection for fast switching
    """

    def __init__(self):
        self.event_bus = get_event_bus()

        # Network state tracking
        self.current_conditions = NetworkConditions()
        self.link_quality_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.quality_metrics: Dict[PathProtocol, LinkQualityMetrics] = {}

        # Change detection
        self.change_events: deque[LinkChangeEvent] = deque(maxlen=1000)
        self.change_detection_threshold = 0.3  # Significance threshold for reporting changes
        self.last_significant_change = 0.0

        # Monitoring configuration
        self.monitoring_interval = 5.0  # Monitor every 5 seconds
        self.fast_monitoring_enabled = True
        self.fast_monitoring_interval = 0.5  # Fast checks every 500ms
        self.last_monitoring_time = 0.0

        # Quality assessment weights
        self.quality_weights = {
            "latency": 0.3,
            "bandwidth": 0.25,
            "packet_loss": 0.25,
            "availability": 0.15,
            "jitter": 0.05,
        }

        # Link-specific configurations
        self.link_configs = {
            "wifi": {"expected_bandwidth": 50.0, "expected_latency": 20.0},
            "cellular": {"expected_bandwidth": 5.0, "expected_latency": 50.0},
            "bluetooth": {"expected_bandwidth": 0.1, "expected_latency": 200.0},
            "ethernet": {"expected_bandwidth": 100.0, "expected_latency": 5.0},
        }

        # Monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False

        logger.info("NetworkMonitoringService initialized")

    async def start_monitoring(self) -> None:
        """Start continuous network monitoring"""
        if self.running:
            return

        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Network monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop network monitoring"""
        self.running = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Network monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.running:
            try:
                # Perform monitoring checks
                await self.monitor_network_links()

                # Wait for next interval
                if self.fast_monitoring_enabled:
                    await asyncio.sleep(self.fast_monitoring_interval)
                else:
                    await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1.0)  # Brief pause before retry

    async def monitor_network_links(self) -> NetworkConditions:
        """Monitor current network link conditions"""
        start_time = time.time()

        # Store previous conditions for change detection
        previous_conditions = NetworkConditions(
            bluetooth_available=self.current_conditions.bluetooth_available,
            internet_available=self.current_conditions.internet_available,
            cellular_available=self.current_conditions.cellular_available,
            wifi_connected=self.current_conditions.wifi_connected,
            latency_ms=self.current_conditions.latency_ms,
            bandwidth_mbps=self.current_conditions.bandwidth_mbps,
            reliability_score=self.current_conditions.reliability_score,
        )

        # Monitor each link type
        wifi_quality = await self._monitor_wifi_link()
        cellular_quality = await self._monitor_cellular_link()
        bluetooth_quality = await self._monitor_bluetooth_link()

        # Update current conditions
        self.current_conditions.wifi_connected = wifi_quality.availability > 0.8
        self.current_conditions.cellular_available = cellular_quality.availability > 0.8
        self.current_conditions.bluetooth_available = bluetooth_quality.availability > 0.8
        self.current_conditions.internet_available = (
            self.current_conditions.wifi_connected or self.current_conditions.cellular_available
        )

        # Update performance metrics
        if self.current_conditions.internet_available:
            if self.current_conditions.wifi_connected:
                self.current_conditions.latency_ms = wifi_quality.latency_ms
                self.current_conditions.bandwidth_mbps = wifi_quality.bandwidth_mbps
            else:
                self.current_conditions.latency_ms = cellular_quality.latency_ms
                self.current_conditions.bandwidth_mbps = cellular_quality.bandwidth_mbps
        else:
            self.current_conditions.latency_ms = 999.0  # High latency when offline
            self.current_conditions.bandwidth_mbps = 0.0

        # Calculate overall reliability
        self.current_conditions.reliability_score = self._calculate_overall_reliability(
            wifi_quality, cellular_quality, bluetooth_quality
        )

        # Store quality metrics
        self.quality_metrics[PathProtocol.BETANET] = (
            wifi_quality if self.current_conditions.wifi_connected else cellular_quality
        )
        self.quality_metrics[PathProtocol.BITCHAT] = bluetooth_quality

        # Detect and process changes
        await self._detect_and_process_changes(previous_conditions, self.current_conditions)

        # Update monitoring history
        monitoring_time = (time.time() - start_time) * 1000
        self.link_quality_history["monitoring_time"].append(monitoring_time)

        self.last_monitoring_time = time.time()

        logger.debug(
            f"Network monitoring completed in {monitoring_time:.1f}ms: "
            f"Internet={self.current_conditions.internet_available}, "
            f"BT={self.current_conditions.bluetooth_available}, "
            f"BW={self.current_conditions.bandwidth_mbps:.1f}Mbps"
        )

        return self.current_conditions

    async def _monitor_wifi_link(self) -> LinkQualityMetrics:
        """Monitor WiFi link quality"""
        quality = LinkQualityMetrics()

        try:
            # Check WiFi connectivity via internet connectivity test
            internet_available = await self._test_internet_connectivity()

            if internet_available:
                # Measure WiFi-specific metrics
                quality.availability = 1.0
                quality.latency_ms = await self._measure_latency("8.8.8.8")
                quality.bandwidth_mbps = await self._estimate_bandwidth("wifi")
                quality.packet_loss_rate = await self._measure_packet_loss("8.8.8.8")
                quality.jitter_ms = await self._measure_jitter("8.8.8.8")
            else:
                quality.availability = 0.0
                quality.latency_ms = 999.0
                quality.bandwidth_mbps = 0.0
                quality.packet_loss_rate = 1.0

        except Exception as e:
            logger.warning(f"WiFi monitoring error: {e}")
            quality.availability = 0.0

        return quality

    async def _monitor_cellular_link(self) -> LinkQualityMetrics:
        """Monitor cellular link quality"""
        quality = LinkQualityMetrics()

        try:
            # For now, assume cellular follows WiFi status with different characteristics
            internet_available = await self._test_internet_connectivity()

            if internet_available:
                # Cellular typically has different characteristics than WiFi
                quality.availability = 0.9  # Slightly less reliable than WiFi
                quality.latency_ms = await self._measure_latency("8.8.8.8") * 1.5  # Higher latency
                quality.bandwidth_mbps = await self._estimate_bandwidth("cellular")
                quality.packet_loss_rate = await self._measure_packet_loss("8.8.8.8") * 1.2
                quality.jitter_ms = await self._measure_jitter("8.8.8.8") * 2.0
            else:
                quality.availability = 0.0

        except Exception as e:
            logger.warning(f"Cellular monitoring error: {e}")
            quality.availability = 0.0

        return quality

    async def _monitor_bluetooth_link(self) -> LinkQualityMetrics:
        """Monitor Bluetooth link quality"""
        quality = LinkQualityMetrics()

        try:
            if not BLUETOOTH_AVAILABLE:
                quality.availability = 0.0
                return quality

            # Test Bluetooth availability
            bt_available = await self._test_bluetooth_availability()

            if bt_available:
                quality.availability = 1.0
                quality.latency_ms = 200.0  # Typical Bluetooth latency
                quality.bandwidth_mbps = 0.1  # Low bandwidth for BitChat
                quality.packet_loss_rate = 0.15  # Bluetooth can be lossy
                quality.jitter_ms = 20.0  # Variable Bluetooth timing
                quality.signal_strength = -50  # Simulated RSSI
            else:
                quality.availability = 0.0

        except Exception as e:
            logger.warning(f"Bluetooth monitoring error: {e}")
            quality.availability = 0.0

        return quality

    async def _test_internet_connectivity(self) -> bool:
        """Test internet connectivity"""
        try:
            # Test connection to multiple reliable servers
            for server in ["8.8.8.8", "1.1.1.1"]:
                try:
                    _, writer = await asyncio.wait_for(asyncio.open_connection(server, 53), timeout=2.0)
                    writer.close()
                    await writer.wait_closed()
                    return True
                except (asyncio.TimeoutError, OSError):
                    continue
            return False
        except Exception:
            return False

    async def _test_bluetooth_availability(self) -> bool:
        """Test Bluetooth availability"""
        if not BLUETOOTH_AVAILABLE:
            return False

        try:
            # Quick Bluetooth discovery test
            bluetooth.discover_devices(duration=1, lookup_names=False)
            return True  # If discovery works, Bluetooth is available
        except Exception:
            return False

    async def _measure_latency(self, target: str) -> float:
        """Measure network latency to target"""
        try:
            start_time = time.time()
            _, writer = await asyncio.wait_for(asyncio.open_connection(target, 53), timeout=2.0)
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            writer.close()
            await writer.wait_closed()
            return latency
        except Exception:
            return 999.0  # High latency on failure

    async def _estimate_bandwidth(self, link_type: str) -> float:
        """Estimate available bandwidth for link type"""
        config = self.link_configs.get(link_type, {"expected_bandwidth": 1.0})

        # For now, return expected bandwidth with some variation
        # In production, this would involve actual bandwidth testing
        base_bandwidth = config["expected_bandwidth"]

        # Simulate some variation based on time and conditions
        variation_factor = 0.8 + (time.time() % 10) * 0.04  # 0.8 to 1.2 factor
        return base_bandwidth * variation_factor

    async def _measure_packet_loss(self, target: str) -> float:
        """Measure packet loss rate"""
        # Simplified packet loss estimation
        # In production, this would involve sending multiple packets and measuring loss
        try:
            success_count = 0
            total_attempts = 3

            for _ in range(total_attempts):
                try:
                    _, writer = await asyncio.wait_for(asyncio.open_connection(target, 53), timeout=1.0)
                    writer.close()
                    await writer.wait_closed()
                    success_count += 1
                except Exception as e:
                    import logging

                    logging.exception("Exception in network connection test: %s", str(e))

            loss_rate = 1.0 - (success_count / total_attempts)
            return max(0.0, min(1.0, loss_rate))
        except:
            return 0.1  # Assume 10% loss on error

    async def _measure_jitter(self, target: str) -> float:
        """Measure network jitter"""
        latencies = []

        try:
            for _ in range(3):
                latency = await self._measure_latency(target)
                if latency < 900:  # Valid latency measurement
                    latencies.append(latency)
                await asyncio.sleep(0.1)

            if len(latencies) > 1:
                avg_latency = mean(latencies)
                jitter = sum(abs(l - avg_latency) for l in latencies) / len(latencies)
                return jitter
            else:
                return 5.0  # Default jitter
        except:
            return 10.0  # Higher jitter on error

    def _calculate_overall_reliability(
        self,
        wifi_quality: LinkQualityMetrics,
        cellular_quality: LinkQualityMetrics,
        bluetooth_quality: LinkQualityMetrics,
    ) -> float:
        """Calculate overall network reliability score"""
        # Weight different links based on their importance and availability
        weights = {"wifi": 0.5, "cellular": 0.3, "bluetooth": 0.2}

        reliability = 0.0

        # WiFi contribution
        if wifi_quality.availability > 0:
            wifi_score = (
                wifi_quality.availability * 0.4
                + (1.0 - min(1.0, wifi_quality.packet_loss_rate)) * 0.3
                + (1.0 - min(1.0, wifi_quality.latency_ms / 200.0)) * 0.3
            )
            reliability += weights["wifi"] * wifi_score

        # Cellular contribution
        if cellular_quality.availability > 0:
            cellular_score = (
                cellular_quality.availability * 0.4
                + (1.0 - min(1.0, cellular_quality.packet_loss_rate)) * 0.3
                + (1.0 - min(1.0, cellular_quality.latency_ms / 300.0)) * 0.3
            )
            reliability += weights["cellular"] * cellular_score

        # Bluetooth contribution
        if bluetooth_quality.availability > 0:
            bluetooth_score = (
                bluetooth_quality.availability * 0.6 + (1.0 - min(1.0, bluetooth_quality.packet_loss_rate)) * 0.4
            )
            reliability += weights["bluetooth"] * bluetooth_score

        return max(0.0, min(1.0, reliability))

    async def _detect_and_process_changes(self, previous: NetworkConditions, current: NetworkConditions) -> None:
        """Detect and process network changes"""
        changes_detected = []

        # Check connectivity changes
        if previous.internet_available != current.internet_available:
            change_event = LinkChangeEvent(
                timestamp=time.time(),
                link_type="internet",
                change_type="connected" if current.internet_available else "disconnected",
                old_metrics=None,
                new_metrics=self.quality_metrics.get(PathProtocol.BETANET, LinkQualityMetrics()),
                significance_score=1.0,  # Connectivity changes are always significant
            )
            changes_detected.append(change_event)

        if previous.bluetooth_available != current.bluetooth_available:
            change_event = LinkChangeEvent(
                timestamp=time.time(),
                link_type="bluetooth",
                change_type="connected" if current.bluetooth_available else "disconnected",
                old_metrics=None,
                new_metrics=self.quality_metrics.get(PathProtocol.BITCHAT, LinkQualityMetrics()),
                significance_score=0.8,  # Bluetooth changes are significant but less critical
            )
            changes_detected.append(change_event)

        # Check performance changes
        bandwidth_change = abs(current.bandwidth_mbps - previous.bandwidth_mbps)
        if bandwidth_change > 2.0:  # Significant bandwidth change
            significance = min(1.0, bandwidth_change / 10.0)
            change_type = "improved" if current.bandwidth_mbps > previous.bandwidth_mbps else "degraded"

            change_event = LinkChangeEvent(
                timestamp=time.time(),
                link_type="bandwidth",
                change_type=change_type,
                old_metrics=None,
                new_metrics=self.quality_metrics.get(PathProtocol.BETANET, LinkQualityMetrics()),
                significance_score=significance,
            )
            changes_detected.append(change_event)

        latency_change = abs(current.latency_ms - previous.latency_ms)
        if latency_change > 50.0:  # Significant latency change
            significance = min(1.0, latency_change / 200.0)
            change_type = "degraded" if current.latency_ms > previous.latency_ms else "improved"

            change_event = LinkChangeEvent(
                timestamp=time.time(),
                link_type="latency",
                change_type=change_type,
                old_metrics=None,
                new_metrics=self.quality_metrics.get(PathProtocol.BETANET, LinkQualityMetrics()),
                significance_score=significance,
            )
            changes_detected.append(change_event)

        # Process significant changes
        for change in changes_detected:
            if change.significance_score >= self.change_detection_threshold:
                await self._process_significant_change(change)
                self.change_events.append(change)
                self.last_significant_change = time.time()

    async def _process_significant_change(self, change: LinkChangeEvent) -> None:
        """Process significant network change"""
        logger.info(
            f"Significant network change detected: {change.link_type} {change.change_type} "
            f"(significance: {change.significance_score:.2f})"
        )

        # Emit change event
        self._emit_monitoring_event(
            "significant_change_detected",
            {
                "link_type": change.link_type,
                "change_type": change.change_type,
                "significance_score": change.significance_score,
                "timestamp": change.timestamp,
                "current_conditions": {
                    "internet_available": self.current_conditions.internet_available,
                    "bluetooth_available": self.current_conditions.bluetooth_available,
                    "bandwidth_mbps": self.current_conditions.bandwidth_mbps,
                    "latency_ms": self.current_conditions.latency_ms,
                },
            },
        )

    async def detect_link_changes(self) -> Tuple[bool, Dict[str, Any]]:
        """Detect significant network link changes"""
        current_time = time.time()

        # Check if we have recent significant changes
        recent_changes = [
            change for change in self.change_events if current_time - change.timestamp < 10.0  # Last 10 seconds
        ]

        if recent_changes:
            # Calculate overall change significance
            max_significance = max(change.significance_score for change in recent_changes)

            change_summary = {
                "has_changes": True,
                "max_significance": max_significance,
                "change_count": len(recent_changes),
                "change_types": [change.change_type for change in recent_changes],
                "affected_links": list(set(change.link_type for change in recent_changes)),
                "time_since_last_change": current_time - self.last_significant_change,
            }

            return True, change_summary

        return False, {"has_changes": False, "time_since_last_change": current_time - self.last_significant_change}

    def assess_link_quality(self, protocol: PathProtocol) -> float:
        """Assess quality score for specific protocol link"""
        if protocol not in self.quality_metrics:
            return 0.5  # Default neutral score

        metrics = self.quality_metrics[protocol]

        # Calculate weighted quality score
        scores = {
            "latency": max(0.0, 1.0 - min(1.0, metrics.latency_ms / 200.0)),
            "bandwidth": min(1.0, metrics.bandwidth_mbps / 10.0),  # Normalize to 10 Mbps
            "packet_loss": 1.0 - metrics.packet_loss_rate,
            "availability": metrics.availability,
            "jitter": max(0.0, 1.0 - min(1.0, metrics.jitter_ms / 50.0)),
        }

        # Calculate weighted score
        quality_score = sum(
            self.quality_weights[metric] * score for metric, score in scores.items() if metric in self.quality_weights
        )

        return max(0.0, min(1.0, quality_score))

    def _emit_monitoring_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit network monitoring event"""
        event = RoutingEvent(
            event_type=event_type, timestamp=time.time(), source_service="NetworkMonitoringService", data=data
        )
        self.event_bus.publish(event)

    def get_monitoring_metrics(self) -> Dict[str, Any]:
        """Get network monitoring performance metrics"""
        monitoring_times = list(self.link_quality_history["monitoring_time"])

        return {
            "monitoring_interval": self.monitoring_interval,
            "fast_monitoring_enabled": self.fast_monitoring_enabled,
            "avg_monitoring_time_ms": mean(monitoring_times) if monitoring_times else 0,
            "max_monitoring_time_ms": max(monitoring_times) if monitoring_times else 0,
            "change_events_count": len(self.change_events),
            "last_significant_change": self.last_significant_change,
            "quality_metrics": {
                protocol.value: {
                    "latency_ms": metrics.latency_ms,
                    "bandwidth_mbps": metrics.bandwidth_mbps,
                    "availability": metrics.availability,
                    "quality_score": self.assess_link_quality(protocol),
                }
                for protocol, metrics in self.quality_metrics.items()
            },
        }

    def get_current_network_conditions(self) -> NetworkConditions:
        """Get current network conditions"""
        return self.current_conditions

    def get_link_history(self, link_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get historical link quality data"""
        history = self.link_quality_history.get(link_type, deque())
        return list(history)[-limit:] if history else []
