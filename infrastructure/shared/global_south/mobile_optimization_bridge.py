"""
Mobile Optimization Bridge for Global South Offline Coordinator.

This module provides mobile-specific optimizations including battery management,
thermal throttling, data compression, and adaptive UI for offline-first operation.
Designed for resource-constrained mobile devices in Global South scenarios.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import platform
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class BatteryStatus(Enum):
    """Battery status levels for mobile devices."""

    CRITICAL = "critical"  # < 10%
    LOW = "low"  # 10-25%
    MODERATE = "moderate"  # 25-50%
    GOOD = "good"  # 50-80%
    EXCELLENT = "excellent"  # > 80%


class ThermalState(Enum):
    """Thermal management states."""

    NORMAL = "normal"  # < 40°C
    WARM = "warm"  # 40-50°C
    HOT = "hot"  # 50-60°C
    CRITICAL = "critical"  # > 60°C


class NetworkType(Enum):
    """Network connection types with cost implications."""

    WIFI = "wifi"  # Usually free
    CELLULAR_2G = "cellular_2g"  # Very slow, cheap
    CELLULAR_3G = "cellular_3g"  # Moderate speed, moderate cost
    CELLULAR_4G = "cellular_4g"  # Fast, expensive
    CELLULAR_5G = "cellular_5g"  # Very fast, very expensive
    SATELLITE = "satellite"  # Variable speed, expensive
    OFFLINE = "offline"  # No connection


@dataclass
class MobileDeviceState:
    """Current state of mobile device for optimization decisions."""

    battery_percent: float
    battery_status: BatteryStatus
    is_charging: bool
    cpu_temp_celsius: float
    thermal_state: ThermalState
    available_memory_mb: float
    network_type: NetworkType
    data_cost_per_mb: float
    screen_on: bool
    app_in_foreground: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def detect_current_state(cls) -> "MobileDeviceState":
        """Detect current mobile device state."""
        try:
            # Battery detection
            if hasattr(psutil, "sensors_battery"):
                battery = psutil.sensors_battery()
                battery_percent = battery.percent if battery else 50.0
                is_charging = battery.power_plugged if battery else False
            else:
                battery_percent = 50.0  # Default assumption
                is_charging = False

            # Determine battery status
            if battery_percent < 10:
                battery_status = BatteryStatus.CRITICAL
            elif battery_percent < 25:
                battery_status = BatteryStatus.LOW
            elif battery_percent < 50:
                battery_status = BatteryStatus.MODERATE
            elif battery_percent < 80:
                battery_status = BatteryStatus.GOOD
            else:
                battery_status = BatteryStatus.EXCELLENT

            # CPU temperature (if available)
            cpu_temp = 35.0  # Default safe temperature
            try:
                if hasattr(psutil, "sensors_temperatures"):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        # Get average CPU temperature
                        cpu_temps = []
                        for name, entries in temps.items():
                            if "cpu" in name.lower() or "core" in name.lower():
                                cpu_temps.extend([entry.current for entry in entries])
                        if cpu_temps:
                            cpu_temp = sum(cpu_temps) / len(cpu_temps)
            except Exception:
                pass

            # Determine thermal state
            if cpu_temp < 40:
                thermal_state = ThermalState.NORMAL
            elif cpu_temp < 50:
                thermal_state = ThermalState.WARM
            elif cpu_temp < 60:
                thermal_state = ThermalState.HOT
            else:
                thermal_state = ThermalState.CRITICAL

            # Memory detection
            memory = psutil.virtual_memory()
            available_memory_mb = memory.available / (1024 * 1024)

            # Network detection (simplified)
            network_type = NetworkType.WIFI  # Default assumption
            data_cost_per_mb = 0.0  # Free WiFi assumption

            # Check if we're on mobile/cellular
            if platform.system() == "Android":
                # On Android, assume cellular if not on WiFi
                network_type = NetworkType.CELLULAR_4G
                data_cost_per_mb = 0.01  # $0.01 per MB

            # Screen and app state (simplified detection)
            screen_on = True  # Default assumption
            app_in_foreground = True  # Default assumption

            return cls(
                battery_percent=battery_percent,
                battery_status=battery_status,
                is_charging=is_charging,
                cpu_temp_celsius=cpu_temp,
                thermal_state=thermal_state,
                available_memory_mb=available_memory_mb,
                network_type=network_type,
                data_cost_per_mb=data_cost_per_mb,
                screen_on=screen_on,
                app_in_foreground=app_in_foreground,
            )

        except Exception as e:
            logger.warning(f"Failed to detect device state: {e}")
            # Return conservative defaults
            return cls(
                battery_percent=50.0,
                battery_status=BatteryStatus.MODERATE,
                is_charging=False,
                cpu_temp_celsius=35.0,
                thermal_state=ThermalState.NORMAL,
                available_memory_mb=1024.0,
                network_type=NetworkType.WIFI,
                data_cost_per_mb=0.0,
                screen_on=True,
                app_in_foreground=True,
            )


@dataclass
class OptimizationPolicy:
    """Optimization policy based on device state."""

    max_cpu_usage_percent: float = 70.0
    max_memory_usage_mb: float = 512.0
    max_sync_data_mb: float = 10.0
    sync_frequency_minutes: float = 15.0
    enable_compression: bool = True
    compression_level: int = 6  # 1-9, higher = more compression
    background_sync_enabled: bool = True
    preemptive_sync_enabled: bool = True
    cache_aggressive_eviction: bool = False

    @classmethod
    def for_device_state(cls, state: MobileDeviceState) -> "OptimizationPolicy":
        """Create optimization policy based on device state."""
        policy = cls()

        # Battery-based optimizations
        if state.battery_status == BatteryStatus.CRITICAL:
            # Extreme battery saving
            policy.max_cpu_usage_percent = 30.0
            policy.max_memory_usage_mb = 256.0
            policy.max_sync_data_mb = 1.0
            policy.sync_frequency_minutes = 60.0
            policy.compression_level = 9
            policy.background_sync_enabled = False
            policy.preemptive_sync_enabled = False
            policy.cache_aggressive_eviction = True

        elif state.battery_status == BatteryStatus.LOW:
            # Aggressive battery saving
            policy.max_cpu_usage_percent = 50.0
            policy.max_memory_usage_mb = 384.0
            policy.max_sync_data_mb = 5.0
            policy.sync_frequency_minutes = 30.0
            policy.compression_level = 8
            policy.background_sync_enabled = True
            policy.preemptive_sync_enabled = False
            policy.cache_aggressive_eviction = True

        elif state.battery_status == BatteryStatus.MODERATE:
            # Moderate battery saving
            policy.max_cpu_usage_percent = 60.0
            policy.max_memory_usage_mb = 450.0
            policy.max_sync_data_mb = 7.0
            policy.sync_frequency_minutes = 20.0
            policy.compression_level = 7
            policy.cache_aggressive_eviction = False

        # Thermal throttling
        if state.thermal_state == ThermalState.CRITICAL:
            # Extreme thermal throttling
            policy.max_cpu_usage_percent = min(policy.max_cpu_usage_percent, 25.0)
            policy.background_sync_enabled = False
            policy.preemptive_sync_enabled = False

        elif state.thermal_state == ThermalState.HOT:
            # Thermal throttling
            policy.max_cpu_usage_percent = min(policy.max_cpu_usage_percent, 40.0)
            policy.background_sync_enabled = False

        # Network cost optimization
        if state.data_cost_per_mb > 0.005:  # Expensive data
            policy.max_sync_data_mb = min(policy.max_sync_data_mb, 3.0)
            policy.compression_level = 9
            policy.sync_frequency_minutes = max(policy.sync_frequency_minutes, 30.0)

        # Charging state optimizations
        if state.is_charging:
            # More aggressive when charging
            policy.max_cpu_usage_percent = min(policy.max_cpu_usage_percent * 1.2, 85.0)
            policy.max_memory_usage_mb = min(policy.max_memory_usage_mb * 1.3, 768.0)
            policy.sync_frequency_minutes = max(policy.sync_frequency_minutes * 0.7, 5.0)
            policy.preemptive_sync_enabled = True

        # Background operation optimization
        if not state.app_in_foreground:
            # Reduce activity when in background
            policy.max_cpu_usage_percent *= 0.6
            policy.max_memory_usage_mb *= 0.8
            policy.sync_frequency_minutes *= 1.5

        return policy


class MobileOptimizationBridge:
    """
    Mobile optimization bridge for Global South offline coordinator.

    Provides device-aware optimizations for battery, thermal, memory, and
    network usage in resource-constrained mobile environments.
    """

    def __init__(self, offline_coordinator=None, monitoring_interval: int = 30):
        """Initialize mobile optimization bridge."""
        self.offline_coordinator = offline_coordinator
        self.monitoring_interval = monitoring_interval  # seconds

        # Device state tracking
        self.current_state: MobileDeviceState | None = None
        self.current_policy: OptimizationPolicy | None = None
        self.state_history: list[MobileDeviceState] = []
        self.max_history_size = 100

        # Optimization metrics
        self.optimization_stats = {
            "state_detections": 0,
            "policy_changes": 0,
            "battery_saves_triggered": 0,
            "thermal_throttles_triggered": 0,
            "data_optimizations": 0,
            "background_optimizations": 0,
            "total_energy_saved_estimated": 0.0,  # In percentage points
            "total_data_saved_mb": 0.0,
        }

        # Monitoring task
        self.monitoring_task: asyncio.Task | None = None
        self.is_monitoring = False

        # Adaptive thresholds
        self.adaptive_thresholds = {
            "low_battery_threshold": 25.0,
            "critical_battery_threshold": 10.0,
            "high_temp_threshold": 50.0,
            "critical_temp_threshold": 60.0,
            "expensive_data_threshold": 0.005,  # $/MB
        }

        logger.info("Mobile optimization bridge initialized")

    async def start_monitoring(self) -> None:
        """Start continuous device state monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Started mobile device monitoring (interval: {self.monitoring_interval}s)")

    async def stop_monitoring(self) -> None:
        """Stop device state monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped mobile device monitoring")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Detect current device state
                state = MobileDeviceState.detect_current_state()
                await self._update_device_state(state)

                # Sleep until next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)

    async def _update_device_state(self, state: MobileDeviceState) -> None:
        """Update device state and apply optimizations."""
        previous_state = self.current_state
        self.current_state = state

        # Add to history
        self.state_history.append(state)
        if len(self.state_history) > self.max_history_size:
            self.state_history.pop(0)

        # Update statistics
        self.optimization_stats["state_detections"] += 1

        # Generate optimization policy
        new_policy = OptimizationPolicy.for_device_state(state)
        policy_changed = self.current_policy != new_policy

        if policy_changed:
            self.current_policy = new_policy
            self.optimization_stats["policy_changes"] += 1

            logger.info(
                f"Device state changed: battery={state.battery_percent:.1f}%, "
                f"temp={state.cpu_temp_celsius:.1f}°C, "
                f"memory={state.available_memory_mb:.0f}MB"
            )

            # Apply optimizations
            await self._apply_optimizations(state, new_policy, previous_state)

    async def _apply_optimizations(
        self, state: MobileDeviceState, policy: OptimizationPolicy, previous_state: MobileDeviceState | None
    ) -> None:
        """Apply optimization policy based on device state."""

        # Battery optimizations
        if state.battery_status in [BatteryStatus.CRITICAL, BatteryStatus.LOW]:
            if not previous_state or previous_state.battery_status not in [BatteryStatus.CRITICAL, BatteryStatus.LOW]:
                logger.warning(f"Battery {state.battery_status.value} - applying aggressive power saving")
                self.optimization_stats["battery_saves_triggered"] += 1

                # Integrate with offline coordinator if available
                if self.offline_coordinator:
                    await self._optimize_offline_coordinator_for_battery(state, policy)

        # Thermal optimizations
        if state.thermal_state in [ThermalState.HOT, ThermalState.CRITICAL]:
            if not previous_state or previous_state.thermal_state not in [ThermalState.HOT, ThermalState.CRITICAL]:
                logger.warning(f"Thermal {state.thermal_state.value} - applying thermal throttling")
                self.optimization_stats["thermal_throttles_triggered"] += 1

                if self.offline_coordinator:
                    await self._optimize_offline_coordinator_for_thermal(state, policy)

        # Data cost optimizations
        if state.data_cost_per_mb > self.adaptive_thresholds["expensive_data_threshold"]:
            self.optimization_stats["data_optimizations"] += 1

            if self.offline_coordinator:
                await self._optimize_offline_coordinator_for_data_cost(state, policy)

        # Background optimizations
        if not state.app_in_foreground:
            self.optimization_stats["background_optimizations"] += 1

            if self.offline_coordinator:
                await self._optimize_offline_coordinator_for_background(state, policy)

    async def _optimize_offline_coordinator_for_battery(
        self, state: MobileDeviceState, policy: OptimizationPolicy
    ) -> None:
        """Optimize offline coordinator for battery conservation."""
        try:
            # Reduce sync frequency
            if hasattr(self.offline_coordinator, "sync_frequency_minutes"):
                self.offline_coordinator.sync_frequency_minutes = policy.sync_frequency_minutes

            # Enable aggressive compression
            if hasattr(self.offline_coordinator, "compression_enabled"):
                self.offline_coordinator.compression_enabled = policy.enable_compression

            if hasattr(self.offline_coordinator, "compression_threshold_bytes"):
                # Compress smaller files to save more battery
                self.offline_coordinator.compression_threshold_bytes = 512

            # Reduce data budget for low battery
            if hasattr(self.offline_coordinator, "daily_data_budget_usd"):
                if state.battery_status == BatteryStatus.CRITICAL:
                    # Reduce budget by 80%
                    original_budget = self.offline_coordinator.daily_data_budget_usd
                    self.offline_coordinator.daily_data_budget_usd = original_budget * 0.2
                elif state.battery_status == BatteryStatus.LOW:
                    # Reduce budget by 50%
                    original_budget = self.offline_coordinator.daily_data_budget_usd
                    self.offline_coordinator.daily_data_budget_usd = original_budget * 0.5

            logger.debug("Applied battery optimization to offline coordinator")

        except Exception as e:
            logger.error(f"Failed to optimize offline coordinator for battery: {e}")

    async def _optimize_offline_coordinator_for_thermal(
        self, state: MobileDeviceState, policy: OptimizationPolicy
    ) -> None:
        """Optimize offline coordinator for thermal management."""
        try:
            # Disable background sync during thermal stress
            if hasattr(self.offline_coordinator, "background_sync_enabled"):
                self.offline_coordinator.background_sync_enabled = policy.background_sync_enabled

            # Reduce compression level to save CPU (trade bandwidth for CPU)
            if hasattr(self.offline_coordinator, "compression_level"):
                if state.thermal_state == ThermalState.CRITICAL:
                    # Minimal compression to save CPU
                    self.offline_coordinator.compression_level = 1
                elif state.thermal_state == ThermalState.HOT:
                    # Light compression
                    self.offline_coordinator.compression_level = 3

            logger.debug("Applied thermal optimization to offline coordinator")

        except Exception as e:
            logger.error(f"Failed to optimize offline coordinator for thermal: {e}")

    async def _optimize_offline_coordinator_for_data_cost(
        self, state: MobileDeviceState, policy: OptimizationPolicy
    ) -> None:
        """Optimize offline coordinator for data cost management."""
        try:
            # Reduce max sync data
            if hasattr(self.offline_coordinator, "max_sync_data_mb"):
                self.offline_coordinator.max_sync_data_mb = policy.max_sync_data_mb

            # Enable maximum compression
            if hasattr(self.offline_coordinator, "compression_level"):
                self.offline_coordinator.compression_level = policy.compression_level

            # Increase sync frequency to use smaller chunks
            if hasattr(self.offline_coordinator, "sync_frequency_minutes"):
                self.offline_coordinator.sync_frequency_minutes = max(
                    policy.sync_frequency_minutes, 30.0  # At least 30 minutes for expensive data
                )

            self.optimization_stats["total_data_saved_mb"] += 1.0  # Estimate
            logger.debug("Applied data cost optimization to offline coordinator")

        except Exception as e:
            logger.error(f"Failed to optimize offline coordinator for data cost: {e}")

    async def _optimize_offline_coordinator_for_background(
        self, state: MobileDeviceState, policy: OptimizationPolicy
    ) -> None:
        """Optimize offline coordinator for background operation."""
        try:
            # Reduce background activity
            if hasattr(self.offline_coordinator, "background_sync_enabled"):
                self.offline_coordinator.background_sync_enabled = policy.background_sync_enabled

            # Increase sync intervals when in background
            if hasattr(self.offline_coordinator, "sync_frequency_minutes"):
                self.offline_coordinator.sync_frequency_minutes = policy.sync_frequency_minutes

            logger.debug("Applied background optimization to offline coordinator")

        except Exception as e:
            logger.error(f"Failed to optimize offline coordinator for background: {e}")

    def get_current_state(self) -> MobileDeviceState | None:
        """Get current device state."""
        return self.current_state

    def get_current_policy(self) -> OptimizationPolicy | None:
        """Get current optimization policy."""
        return self.current_policy

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get optimization statistics."""
        return {
            **self.optimization_stats,
            "monitoring_active": self.is_monitoring,
            "state_history_size": len(self.state_history),
            "current_battery_percent": self.current_state.battery_percent if self.current_state else None,
            "current_thermal_state": self.current_state.thermal_state.value if self.current_state else None,
        }

    def get_energy_saving_report(self) -> dict[str, Any]:
        """Generate energy saving report."""
        if not self.state_history:
            return {"status": "no_data"}

        # Calculate battery trend
        battery_levels = [state.battery_percent for state in self.state_history]
        if len(battery_levels) >= 2:
            battery_trend = battery_levels[-1] - battery_levels[0]
        else:
            battery_trend = 0.0

        # Count optimization events
        critical_battery_events = sum(
            1 for state in self.state_history if state.battery_status == BatteryStatus.CRITICAL
        )

        thermal_events = sum(
            1 for state in self.state_history if state.thermal_state in [ThermalState.HOT, ThermalState.CRITICAL]
        )

        return {
            "monitoring_duration_hours": len(self.state_history) * self.monitoring_interval / 3600,
            "battery_trend_percent": battery_trend,
            "critical_battery_events": critical_battery_events,
            "thermal_events": thermal_events,
            "estimated_energy_saved_percent": self.optimization_stats.get("total_energy_saved_estimated", 0.0),
            "data_saved_mb": self.optimization_stats.get("total_data_saved_mb", 0.0),
            "optimization_events": {
                "battery_saves": self.optimization_stats["battery_saves_triggered"],
                "thermal_throttles": self.optimization_stats["thermal_throttles_triggered"],
                "data_optimizations": self.optimization_stats["data_optimizations"],
                "background_optimizations": self.optimization_stats["background_optimizations"],
            },
        }

    async def force_optimization_check(self) -> dict[str, Any]:
        """Force an immediate optimization check and return results."""
        state = MobileDeviceState.detect_current_state()
        await self._update_device_state(state)

        return {
            "device_state": {
                "battery_percent": state.battery_percent,
                "battery_status": state.battery_status.value,
                "is_charging": state.is_charging,
                "cpu_temp_celsius": state.cpu_temp_celsius,
                "thermal_state": state.thermal_state.value,
                "available_memory_mb": state.available_memory_mb,
                "network_type": state.network_type.value,
                "data_cost_per_mb": state.data_cost_per_mb,
            },
            "optimization_policy": {
                "max_cpu_usage_percent": self.current_policy.max_cpu_usage_percent,
                "max_memory_usage_mb": self.current_policy.max_memory_usage_mb,
                "max_sync_data_mb": self.current_policy.max_sync_data_mb,
                "sync_frequency_minutes": self.current_policy.sync_frequency_minutes,
                "compression_level": self.current_policy.compression_level,
            }
            if self.current_policy
            else None,
            "timestamp": datetime.utcnow().isoformat(),
        }


async def create_mobile_optimization_bridge(
    offline_coordinator=None, monitoring_interval: int = 30, start_monitoring: bool = True
) -> MobileOptimizationBridge:
    """Create and optionally start mobile optimization bridge."""
    bridge = MobileOptimizationBridge(offline_coordinator=offline_coordinator, monitoring_interval=monitoring_interval)

    if start_monitoring:
        await bridge.start_monitoring()

    logger.info("Mobile optimization bridge created and started")
    return bridge


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create bridge
        bridge = await create_mobile_optimization_bridge(monitoring_interval=10)  # Check every 10 seconds for demo

        try:
            # Run for 1 minute
            print("Running mobile optimization monitoring for 60 seconds...")
            await asyncio.sleep(60)

            # Get current state
            state = bridge.get_current_state()
            if state:
                print(f"Current battery: {state.battery_percent:.1f}% ({state.battery_status.value})")
                print(f"Current temperature: {state.cpu_temp_celsius:.1f}°C ({state.thermal_state.value})")

            # Get optimization stats
            stats = bridge.get_optimization_stats()
            print(f"Optimization events: {stats['policy_changes']}")
            print(f"Battery saves: {stats['battery_saves_triggered']}")

            # Get energy report
            report = bridge.get_energy_saving_report()
            print(f"Energy report: {report}")

        finally:
            await bridge.stop_monitoring()

    asyncio.run(main())
