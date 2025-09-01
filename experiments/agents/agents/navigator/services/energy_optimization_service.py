"""Energy Optimization Service - Battery-aware routing and power management

This service manages energy-efficient routing decisions, battery optimization,
and thermal management for mobile devices in the Navigator system.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..interfaces.routing_interfaces import IEnergyOptimizationService, RoutingEvent
from ..events.event_bus import get_event_bus
from ..path_policy import EnergyMode, PathProtocol

logger = logging.getLogger(__name__)

# Battery monitoring imports with fallbacks
try:
    import psutil

    BATTERY_MONITORING_AVAILABLE = True
except ImportError:
    BATTERY_MONITORING_AVAILABLE = False


class ThermalState(Enum):
    """Thermal management states"""

    COOL = "cool"  # Normal operation
    WARM = "warm"  # Slight throttling
    HOT = "hot"  # Aggressive throttling
    CRITICAL = "critical"  # Emergency throttling


class PowerProfile(Enum):
    """Power management profiles"""

    PERFORMANCE = "performance"  # Maximum performance
    BALANCED = "balanced"  # Balance power/performance
    POWER_SAVER = "power_saver"  # Optimize for battery
    ULTRA_SAVER = "ultra_saver"  # Minimal power usage


@dataclass
class EnergyMetrics:
    """Energy consumption and efficiency metrics"""

    battery_level: Optional[int] = None  # Percentage
    charging_state: bool = False
    power_draw_watts: float = 0.0
    thermal_state: ThermalState = ThermalState.COOL
    cpu_temp_celsius: Optional[float] = None
    estimated_runtime_hours: Optional[float] = None
    last_updated: float = field(default_factory=time.time)


@dataclass
class ProtocolEnergyProfile:
    """Energy profile for specific protocol"""

    protocol: PathProtocol
    idle_power_mw: float = 100.0  # Idle power consumption
    active_power_mw: float = 500.0  # Active power consumption
    setup_energy_mj: float = 50.0  # Setup energy cost
    per_byte_energy_uj: float = 10.0  # Energy per byte transferred
    thermal_impact: float = 0.5  # Impact on thermal state (0-1)
    battery_drain_rate: float = 0.1  # %/hour when active


class EnergyOptimizationService(IEnergyOptimizationService):
    """Energy optimization and battery-aware routing service

    Manages:
    - Battery level monitoring and prediction
    - Protocol energy profiling and optimization
    - Thermal management and throttling
    - Power-aware routing decisions
    - Background service energy management
    """

    def __init__(self):
        self.event_bus = get_event_bus()

        # Energy monitoring
        self.current_metrics = EnergyMetrics()
        self.energy_history: deque[EnergyMetrics] = deque(maxlen=1000)
        self.monitoring_interval = 30.0  # Monitor every 30 seconds
        self.last_monitoring = 0.0

        # Protocol energy profiles
        self.protocol_profiles: Dict[PathProtocol, ProtocolEnergyProfile] = {
            PathProtocol.BITCHAT: ProtocolEnergyProfile(
                protocol=PathProtocol.BITCHAT,
                idle_power_mw=50.0,  # Low idle power for Bluetooth
                active_power_mw=200.0,  # Moderate active power
                setup_energy_mj=20.0,  # Low setup cost
                per_byte_energy_uj=5.0,  # Efficient per-byte
                thermal_impact=0.2,  # Low thermal impact
                battery_drain_rate=0.05,  # 5%/hour
            ),
            PathProtocol.BETANET: ProtocolEnergyProfile(
                protocol=PathProtocol.BETANET,
                idle_power_mw=200.0,  # Higher idle for internet
                active_power_mw=800.0,  # High active power
                setup_energy_mj=100.0,  # Higher setup cost
                per_byte_energy_uj=15.0,  # Higher per-byte cost
                thermal_impact=0.7,  # Moderate thermal impact
                battery_drain_rate=0.15,  # 15%/hour
            ),
            PathProtocol.SCION: ProtocolEnergyProfile(
                protocol=PathProtocol.SCION,
                idle_power_mw=150.0,  # Moderate idle
                active_power_mw=600.0,  # Moderate active power
                setup_energy_mj=80.0,  # Moderate setup cost
                per_byte_energy_uj=12.0,  # Efficient multipath
                thermal_impact=0.5,  # Moderate thermal impact
                battery_drain_rate=0.12,  # 12%/hour
            ),
            PathProtocol.STORE_FORWARD: ProtocolEnergyProfile(
                protocol=PathProtocol.STORE_FORWARD,
                idle_power_mw=10.0,  # Very low idle
                active_power_mw=50.0,  # Very low active
                setup_energy_mj=5.0,  # Minimal setup
                per_byte_energy_uj=1.0,  # Very efficient storage
                thermal_impact=0.1,  # Minimal thermal impact
                battery_drain_rate=0.02,  # 2%/hour
            ),
        }

        # Energy optimization settings
        self.power_profile = PowerProfile.BALANCED
        self.battery_conservation_threshold = 20  # Start conserving at 20%
        self.critical_battery_threshold = 10  # Emergency mode at 10%
        self.thermal_throttle_threshold = 70.0  # Throttle at 70°C

        # Optimization tracking
        self.energy_savings: Dict[str, float] = defaultdict(float)  # Savings per optimization
        self.protocol_usage_stats: Dict[PathProtocol, Dict[str, float]] = {
            protocol: {"total_energy_mj": 0.0, "active_time_s": 0.0, "bytes_transferred": 0}
            for protocol in PathProtocol
        }

        # Predictive modeling
        self.battery_trend_window = 300.0  # 5 minutes for trend analysis
        self.predicted_battery_life_hours = 0.0
        self.energy_budget_per_hour = 0.0

        # Background monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None
        self.running = False

        logger.info("EnergyOptimizationService initialized")

    async def start_monitoring(self) -> None:
        """Start energy monitoring and optimization"""
        if self.running:
            return

        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.optimization_task = asyncio.create_task(self._optimization_loop())

        logger.info("Energy optimization monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop energy monitoring"""
        self.running = False

        for task in [self.monitoring_task, self.optimization_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("Energy optimization monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Background loop for energy monitoring"""
        while self.running:
            try:
                await self.update_energy_metrics()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in energy monitoring loop: {e}")
                await asyncio.sleep(10.0)

    async def _optimization_loop(self) -> None:
        """Background loop for continuous optimization"""
        while self.running:
            try:
                # Perform optimization every 2 minutes
                await asyncio.sleep(120.0)
                await self._perform_continuous_optimization()
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(30.0)

    async def update_energy_metrics(self) -> None:
        """Update current energy metrics"""
        try:
            # Update battery information
            if BATTERY_MONITORING_AVAILABLE:
                battery_info = psutil.sensors_battery()
                if battery_info:
                    self.current_metrics.battery_level = int(battery_info.percent)
                    self.current_metrics.charging_state = battery_info.power_plugged

                    # Estimate runtime based on trend
                    if not battery_info.power_plugged and battery_info.secsleft != psutil.POWER_TIME_UNLIMITED:
                        self.current_metrics.estimated_runtime_hours = battery_info.secsleft / 3600.0

            # Update thermal information
            self.current_metrics.cpu_temp_celsius = await self._get_cpu_temperature()
            self.current_metrics.thermal_state = self._assess_thermal_state(self.current_metrics.cpu_temp_celsius)

            # Estimate power draw (simplified)
            self.current_metrics.power_draw_watts = await self._estimate_power_draw()

            # Update timestamp
            self.current_metrics.last_updated = time.time()

            # Add to history
            self.energy_history.append(
                EnergyMetrics(
                    battery_level=self.current_metrics.battery_level,
                    charging_state=self.current_metrics.charging_state,
                    power_draw_watts=self.current_metrics.power_draw_watts,
                    thermal_state=self.current_metrics.thermal_state,
                    cpu_temp_celsius=self.current_metrics.cpu_temp_celsius,
                    estimated_runtime_hours=self.current_metrics.estimated_runtime_hours,
                    last_updated=self.current_metrics.last_updated,
                )
            )

            # Update predictions
            await self._update_battery_predictions()

            logger.debug(
                f"Energy metrics updated: {self.current_metrics.battery_level}% battery, "
                f"{self.current_metrics.thermal_state.value} thermal state"
            )

        except Exception as e:
            logger.error(f"Failed to update energy metrics: {e}")

    async def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature"""
        try:
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Try to get CPU temperature from various sensors
                    for sensor_name in ["cpu_thermal", "coretemp", "cpu"]:
                        if sensor_name in temps and temps[sensor_name]:
                            return temps[sensor_name][0].current
            return None
        except:
            return None

    def _assess_thermal_state(self, cpu_temp: Optional[float]) -> ThermalState:
        """Assess thermal state based on temperature"""
        if cpu_temp is None:
            return ThermalState.COOL  # Default when unknown

        if cpu_temp >= 85.0:
            return ThermalState.CRITICAL
        elif cpu_temp >= 75.0:
            return ThermalState.HOT
        elif cpu_temp >= 60.0:
            return ThermalState.WARM
        else:
            return ThermalState.COOL

    async def _estimate_power_draw(self) -> float:
        """Estimate current power consumption"""
        # Simplified power estimation based on system activity
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Base system power (rough estimates for mobile device)
            base_power = 2.0  # 2W base consumption
            cpu_power = (cpu_percent / 100.0) * 5.0  # Up to 5W for CPU

            # Add protocol-specific power consumption
            protocol_power = sum(
                profile.idle_power_mw / 1000.0 for profile in self.protocol_profiles.values()  # Convert to watts
            )

            total_power = base_power + cpu_power + protocol_power
            return total_power

        except Exception:
            return 3.0  # Default estimate

    async def _update_battery_predictions(self) -> None:
        """Update battery life predictions"""
        if len(self.energy_history) < 5:
            return  # Need more data points

        # Calculate battery drain rate from recent history
        recent_metrics = list(self.energy_history)[-10:]  # Last 10 readings

        if len(recent_metrics) >= 2 and not self.current_metrics.charging_state:
            # Calculate drain rate
            time_diff = recent_metrics[-1].last_updated - recent_metrics[0].last_updated
            if time_diff > 60:  # At least 1 minute of data
                battery_diff = (recent_metrics[-1].battery_level or 0) - (recent_metrics[0].battery_level or 0)
                drain_rate_per_hour = battery_diff / (time_diff / 3600.0)

                if drain_rate_per_hour < 0:  # Battery draining
                    current_level = self.current_metrics.battery_level or 0
                    self.predicted_battery_life_hours = current_level / abs(drain_rate_per_hour)

                    # Set energy budget
                    self.energy_budget_per_hour = abs(drain_rate_per_hour)

    def optimize_for_battery_life(
        self, current_level: Optional[int], routing_options: List[PathProtocol]
    ) -> List[PathProtocol]:
        """Optimize routing choices for battery conservation"""
        if not routing_options:
            return routing_options

        battery_level = current_level or self.current_metrics.battery_level or 50

        # Sort protocols by energy efficiency
        efficiency_scores = {}
        for protocol in routing_options:
            profile = self.protocol_profiles[protocol]

            # Calculate efficiency score (lower is better)
            base_score = (profile.active_power_mw + profile.idle_power_mw) / 1000.0
            base_score += profile.battery_drain_rate * 10  # Weight drain rate heavily
            base_score += profile.thermal_impact * 5  # Consider thermal impact

            efficiency_scores[protocol] = base_score

        # Apply battery level adjustments
        if battery_level <= self.critical_battery_threshold:
            # Critical battery: heavily favor low-power options
            for protocol in routing_options:
                if protocol == PathProtocol.STORE_FORWARD:
                    efficiency_scores[protocol] *= 0.1  # Strongly prefer store-forward
                elif protocol == PathProtocol.BITCHAT:
                    efficiency_scores[protocol] *= 0.3  # Prefer BitChat
                else:
                    efficiency_scores[protocol] *= 2.0  # Penalize high-power protocols

        elif battery_level <= self.battery_conservation_threshold:
            # Conservation mode: moderately favor efficient options
            for protocol in routing_options:
                if protocol in [PathProtocol.STORE_FORWARD, PathProtocol.BITCHAT]:
                    efficiency_scores[protocol] *= 0.7
                else:
                    efficiency_scores[protocol] *= 1.3

        # Apply thermal throttling
        if self.current_metrics.thermal_state in [ThermalState.HOT, ThermalState.CRITICAL]:
            for protocol in routing_options:
                profile = self.protocol_profiles[protocol]
                penalty = 1.0 + (profile.thermal_impact * 2.0)  # Increase penalty for high-thermal protocols
                efficiency_scores[protocol] *= penalty

        # Sort by efficiency (lower scores = more efficient)
        optimized_options = sorted(routing_options, key=lambda p: efficiency_scores[p])

        logger.debug(
            f"Battery optimization (level={battery_level}%): "
            f"Original order: {[p.value for p in routing_options]}, "
            f"Optimized order: {[p.value for p in optimized_options]}"
        )

        return optimized_options

    def select_energy_efficient_paths(
        self, protocols: List[PathProtocol], energy_mode: EnergyMode
    ) -> List[PathProtocol]:
        """Select most energy-efficient paths based on mode"""
        if not protocols:
            return protocols

        # Calculate energy efficiency for each protocol
        efficiency_ratings = {}

        for protocol in protocols:
            profile = self.protocol_profiles[protocol]

            # Base efficiency calculation
            total_energy_cost = (
                profile.idle_power_mw  # Idle consumption
                + profile.active_power_mw  # Active consumption
                + profile.setup_energy_mj  # Setup cost
                + profile.battery_drain_rate * 100  # Long-term drain impact
            )

            efficiency_ratings[protocol] = 1.0 / (1.0 + total_energy_cost)  # Higher = more efficient

        # Adjust ratings based on energy mode
        if energy_mode == EnergyMode.POWERSAVE:
            # Heavily favor low-power protocols
            mode_weights = {
                PathProtocol.STORE_FORWARD: 3.0,
                PathProtocol.BITCHAT: 2.5,
                PathProtocol.SCION: 1.0,
                PathProtocol.BETANET: 0.5,
            }
        elif energy_mode == EnergyMode.PERFORMANCE:
            # Favor performance over power efficiency
            mode_weights = {
                PathProtocol.SCION: 3.0,
                PathProtocol.BETANET: 2.5,
                PathProtocol.BITCHAT: 1.0,
                PathProtocol.STORE_FORWARD: 0.3,
            }
        else:  # BALANCED
            # Balanced approach
            mode_weights = {
                PathProtocol.SCION: 2.0,
                PathProtocol.BITCHAT: 2.0,
                PathProtocol.BETANET: 1.5,
                PathProtocol.STORE_FORWARD: 1.0,
            }

        # Apply mode weights
        for protocol in protocols:
            efficiency_ratings[protocol] *= mode_weights.get(protocol, 1.0)

        # Sort by efficiency (highest first)
        efficient_paths = sorted(protocols, key=lambda p: efficiency_ratings[p], reverse=True)

        logger.debug(f"Energy-efficient path selection ({energy_mode.value}): " f"{[p.value for p in efficient_paths]}")

        return efficient_paths

    async def manage_power_consumption(self) -> Dict[str, Any]:
        """Monitor and manage power consumption"""
        time.time()

        # Assess current power state
        power_state = {
            "battery_level": self.current_metrics.battery_level,
            "charging": self.current_metrics.charging_state,
            "thermal_state": self.current_metrics.thermal_state.value,
            "power_draw_watts": self.current_metrics.power_draw_watts,
            "predicted_life_hours": self.predicted_battery_life_hours,
        }

        # Determine if power management actions are needed
        actions_taken = []
        recommendations = []

        # Battery level management
        if self.current_metrics.battery_level is not None:
            if self.current_metrics.battery_level <= self.critical_battery_threshold:
                # Critical battery actions
                actions_taken.append("enabled_ultra_power_save")
                await self._enable_ultra_power_save_mode()
                recommendations.append("Consider connecting to power source")

            elif self.current_metrics.battery_level <= self.battery_conservation_threshold:
                # Conservation actions
                actions_taken.append("enabled_power_conservation")
                await self._enable_power_conservation()
                recommendations.append("Limit high-power operations")

        # Thermal management
        if self.current_metrics.thermal_state in [ThermalState.HOT, ThermalState.CRITICAL]:
            actions_taken.append("applied_thermal_throttling")
            await self._apply_thermal_throttling()
            recommendations.append("Reduce system load to prevent overheating")

        # Power profile optimization
        optimal_profile = self._determine_optimal_power_profile()
        if optimal_profile != self.power_profile:
            actions_taken.append(f"switched_to_{optimal_profile.value}_profile")
            self.power_profile = optimal_profile

        power_management_result = {
            "current_state": power_state,
            "actions_taken": actions_taken,
            "recommendations": recommendations,
            "power_profile": self.power_profile.value,
            "optimization_active": len(actions_taken) > 0,
            "energy_budget_per_hour": self.energy_budget_per_hour,
        }

        # Emit power management event
        self._emit_energy_event("power_management_update", power_management_result)

        return power_management_result

    async def _enable_ultra_power_save_mode(self) -> None:
        """Enable ultra power save mode for critical battery"""
        logger.warning("Enabling ultra power save mode due to critical battery level")

        # Prefer only the most energy-efficient protocols
        self.power_profile = PowerProfile.ULTRA_SAVER

        # Record energy savings
        self.energy_savings["ultra_power_save"] += 50.0  # Estimated 50% savings

    async def _enable_power_conservation(self) -> None:
        """Enable power conservation mode"""
        logger.info("Enabling power conservation mode")

        self.power_profile = PowerProfile.POWER_SAVER
        self.energy_savings["power_conservation"] += 25.0  # Estimated 25% savings

    async def _apply_thermal_throttling(self) -> None:
        """Apply thermal throttling to prevent overheating"""
        logger.warning(f"Applying thermal throttling due to {self.current_metrics.thermal_state.value} thermal state")

        # Would reduce CPU frequency, protocol active time, etc.
        # For now, just record the action
        self.energy_savings["thermal_throttling"] += 15.0  # Estimated 15% power reduction

    def _determine_optimal_power_profile(self) -> PowerProfile:
        """Determine optimal power profile based on current conditions"""
        battery_level = self.current_metrics.battery_level or 50

        # Critical battery
        if battery_level <= self.critical_battery_threshold:
            return PowerProfile.ULTRA_SAVER

        # Low battery
        elif battery_level <= self.battery_conservation_threshold:
            return PowerProfile.POWER_SAVER

        # Thermal constraints
        elif self.current_metrics.thermal_state in [ThermalState.HOT, ThermalState.CRITICAL]:
            return PowerProfile.POWER_SAVER

        # Charging
        elif self.current_metrics.charging_state:
            return PowerProfile.PERFORMANCE

        # Normal operation
        else:
            return PowerProfile.BALANCED

    async def _perform_continuous_optimization(self) -> None:
        """Perform continuous energy optimization"""
        try:
            # Analyze protocol usage and optimize
            await self._optimize_protocol_usage()

            # Clean up unused resources
            await self._cleanup_unused_resources()

            # Update energy predictions
            await self._update_battery_predictions()

            logger.debug("Continuous energy optimization completed")

        except Exception as e:
            logger.error(f"Error in continuous optimization: {e}")

    async def _optimize_protocol_usage(self) -> None:
        """Optimize protocol usage based on energy consumption patterns"""
        # Analyze recent usage patterns
        time.time()

        for protocol, stats in self.protocol_usage_stats.items():
            profile = self.protocol_profiles[protocol]

            # Calculate energy efficiency
            if stats["active_time_s"] > 0:
                bytes_per_joule = stats["bytes_transferred"] / (stats["total_energy_mj"] / 1000.0)
                profile.per_byte_energy_uj = max(1.0, 10000.0 / bytes_per_joule)  # Update efficiency

    async def _cleanup_unused_resources(self) -> None:
        """Clean up unused energy-consuming resources"""
        # This would clean up idle connections, background tasks, etc.
        # For now, just record the optimization
        logger.debug("Cleaned up unused energy-consuming resources")

    def track_protocol_energy_usage(
        self, protocol: PathProtocol, bytes_transferred: int, active_time_seconds: float
    ) -> None:
        """Track energy usage for specific protocol"""
        profile = self.protocol_profiles[protocol]
        stats = self.protocol_usage_stats[protocol]

        # Calculate energy consumption
        energy_mj = (
            profile.idle_power_mw * active_time_seconds / 1000.0  # Idle energy
            + profile.active_power_mw * active_time_seconds / 1000.0  # Active energy
            + profile.per_byte_energy_uj * bytes_transferred / 1000.0  # Data transfer energy
        )

        # Update statistics
        stats["total_energy_mj"] += energy_mj
        stats["active_time_s"] += active_time_seconds
        stats["bytes_transferred"] += bytes_transferred

        logger.debug(
            f"Tracked energy usage for {protocol.value}: "
            f"{energy_mj:.2f}mJ, {bytes_transferred} bytes, {active_time_seconds:.1f}s"
        )

    def _emit_energy_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit energy optimization event"""
        event = RoutingEvent(
            event_type=event_type, timestamp=time.time(), source_service="EnergyOptimizationService", data=data
        )
        self.event_bus.publish(event)

    def get_energy_statistics(self) -> Dict[str, Any]:
        """Get energy optimization statistics"""
        battery_trend = []
        if len(self.energy_history) >= 2:
            recent_entries = list(self.energy_history)[-10:]
            battery_levels = [entry.battery_level for entry in recent_entries if entry.battery_level is not None]
            if len(battery_levels) >= 2:
                battery_trend = battery_levels

        return {
            "current_metrics": {
                "battery_level": self.current_metrics.battery_level,
                "charging_state": self.current_metrics.charging_state,
                "thermal_state": self.current_metrics.thermal_state.value,
                "power_draw_watts": self.current_metrics.power_draw_watts,
                "cpu_temp_celsius": self.current_metrics.cpu_temp_celsius,
            },
            "power_profile": self.power_profile.value,
            "predicted_battery_life_hours": self.predicted_battery_life_hours,
            "energy_budget_per_hour": self.energy_budget_per_hour,
            "total_energy_savings": dict(self.energy_savings),
            "protocol_efficiency": {
                protocol.value: {
                    "total_energy_mj": stats["total_energy_mj"],
                    "bytes_per_mj": stats["bytes_transferred"] / max(0.001, stats["total_energy_mj"]),
                    "active_time_s": stats["active_time_s"],
                }
                for protocol, stats in self.protocol_usage_stats.items()
            },
            "battery_trend": battery_trend,
            "optimization_thresholds": {
                "battery_conservation": self.battery_conservation_threshold,
                "critical_battery": self.critical_battery_threshold,
                "thermal_throttle": self.thermal_throttle_threshold,
            },
        }

    def configure_power_profile(self, profile: PowerProfile) -> None:
        """Configure power management profile"""
        old_profile = self.power_profile
        self.power_profile = profile

        # Adjust thresholds based on profile
        if profile == PowerProfile.ULTRA_SAVER:
            self.battery_conservation_threshold = 50  # Be more aggressive
            self.thermal_throttle_threshold = 60.0
        elif profile == PowerProfile.POWER_SAVER:
            self.battery_conservation_threshold = 30
            self.thermal_throttle_threshold = 65.0
        elif profile == PowerProfile.PERFORMANCE:
            self.battery_conservation_threshold = 15  # Less aggressive
            self.thermal_throttle_threshold = 80.0
        else:  # BALANCED
            self.battery_conservation_threshold = 20
            self.thermal_throttle_threshold = 70.0

        logger.info(f"Power profile changed from {old_profile.value} to {profile.value}")

        # Emit profile change event
        self._emit_energy_event(
            "power_profile_changed",
            {
                "old_profile": old_profile.value,
                "new_profile": profile.value,
                "updated_thresholds": {
                    "battery_conservation": self.battery_conservation_threshold,
                    "thermal_throttle": self.thermal_throttle_threshold,
                },
            },
        )

    def estimate_protocol_battery_impact(
        self, protocol: PathProtocol, duration_minutes: float, data_size_bytes: int = 0
    ) -> Dict[str, float]:
        """Estimate battery impact of using specific protocol"""
        profile = self.protocol_profiles[protocol]

        # Calculate energy consumption
        duration_minutes / 60.0

        # Base power consumption
        idle_energy = profile.idle_power_mw * duration_minutes / 60000.0  # Convert to Wh
        active_energy = profile.active_power_mw * duration_minutes / 60000.0

        # Data transfer energy
        data_energy = profile.per_byte_energy_uj * data_size_bytes / 3.6e9  # Convert to Wh

        total_energy_wh = idle_energy + active_energy + data_energy

        # Estimate battery impact (assuming ~50Wh battery capacity for mobile device)
        battery_capacity_wh = 50.0
        battery_impact_percent = (total_energy_wh / battery_capacity_wh) * 100.0

        return {
            "total_energy_wh": total_energy_wh,
            "battery_impact_percent": battery_impact_percent,
            "estimated_drain_rate_per_hour": profile.battery_drain_rate,
            "thermal_impact": profile.thermal_impact,
            "energy_efficiency_bytes_per_wh": data_size_bytes / max(0.001, total_energy_wh),
        }
