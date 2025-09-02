"""
Resource Monitor for Edge Devices

Monitors device health, resource utilization, and performance metrics:
- CPU, memory, disk, and network monitoring
- Battery and thermal state tracking
- Performance profiling and bottleneck detection
- Integration with capability beacon for real-time updates

Acts as the vital signs monitoring system for the edge device,
feeding data to both the beacon and the execution fabric.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
import logging
import time
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Overall device health status"""

    HEALTHY = "healthy"  # All systems normal
    DEGRADED = "degraded"  # Some performance issues
    CRITICAL = "critical"  # Severe performance issues
    OFFLINE = "offline"  # Device not responsive


class ThermalState(str, Enum):
    """Device thermal state"""

    NORMAL = "normal"  # Normal operating temperature
    WARM = "warm"  # Elevated but acceptable
    HOT = "hot"  # High temperature, throttling may occur
    CRITICAL = "critical"  # Dangerous temperature, emergency throttling


class BatteryState(str, Enum):
    """Battery power state"""

    FULL = "full"  # >90%
    HIGH = "high"  # 70-90%
    MEDIUM = "medium"  # 30-70%
    LOW = "low"  # 10-30%
    CRITICAL = "critical"  # <10%
    CHARGING = "charging"  # Plugged in and charging


@dataclass
class ResourceSnapshot:
    """Point-in-time resource utilization snapshot"""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # CPU metrics
    cpu_percent: float = 0.0
    cpu_cores_logical: int = 0
    cpu_cores_physical: int = 0
    cpu_freq_mhz: float = 0.0
    cpu_temp_celsius: float | None = None

    # Memory metrics
    memory_total_mb: int = 0
    memory_used_mb: int = 0
    memory_percent: float = 0.0
    memory_available_mb: int = 0
    swap_total_mb: int = 0
    swap_used_mb: int = 0

    # Disk metrics
    disk_total_mb: int = 0
    disk_used_mb: int = 0
    disk_free_mb: int = 0
    disk_percent: float = 0.0
    disk_read_mb_s: float = 0.0
    disk_write_mb_s: float = 0.0

    # Network metrics
    network_sent_mb_s: float = 0.0
    network_recv_mb_s: float = 0.0
    network_connections: int = 0

    # Power metrics
    battery_percent: float | None = None
    battery_time_left_s: int | None = None
    power_plugged: bool | None = None

    # Process metrics
    process_count: int = 0
    active_fog_jobs: int = 0

    # Performance metrics
    load_avg_1m: float | None = None
    load_avg_5m: float | None = None
    load_avg_15m: float | None = None


@dataclass
class PerformanceProfile:
    """Device performance characteristics"""

    device_id: str
    profiling_start: datetime
    profiling_duration_s: float

    # CPU performance
    cpu_single_core_score: float = 0.0  # Relative performance score
    cpu_multi_core_score: float = 0.0  # Multi-core performance
    cpu_sustained_percent: float = 100.0  # Sustained performance under load

    # Memory performance
    memory_bandwidth_mb_s: float = 0.0  # Memory bandwidth
    memory_latency_ns: float = 0.0  # Memory access latency

    # Disk performance
    disk_seq_read_mb_s: float = 0.0  # Sequential read speed
    disk_seq_write_mb_s: float = 0.0  # Sequential write speed
    disk_random_iops: float = 0.0  # Random I/O operations per second

    # Network performance
    network_bandwidth_mb_s: float = 0.0  # Network bandwidth
    network_latency_ms: float = 0.0  # Network latency

    # Thermal characteristics
    thermal_throttling_temp: float | None = None
    thermal_max_sustained_load: float = 100.0

    # Power characteristics
    power_efficiency_score: float = 1.0  # Work per watt
    battery_drain_rate_w: float | None = None


class ResourceMonitor:
    """
    Comprehensive resource monitoring system

    Continuously monitors device resources and provides:
    - Real-time utilization metrics
    - Performance profiling
    - Health assessment
    - Predictive analytics for resource planning
    """

    def __init__(
        self,
        device_id: str,
        monitoring_interval: float = 5.0,
        history_retention_hours: int = 24,
        enable_profiling: bool = True,
    ):
        """
        Initialize resource monitor

        Args:
            device_id: Unique device identifier
            monitoring_interval: Monitoring frequency in seconds
            history_retention_hours: How long to keep historical data
            enable_profiling: Enable performance profiling
        """

        self.device_id = device_id
        self.monitoring_interval = monitoring_interval
        self.history_retention = timedelta(hours=history_retention_hours)
        self.enable_profiling = enable_profiling

        # Historical data
        self.snapshots: list[ResourceSnapshot] = []
        self.max_snapshots = int(history_retention_hours * 3600 / monitoring_interval)

        # Performance profile
        self.performance_profile: PerformanceProfile | None = None

        # Current state
        self.current_snapshot: ResourceSnapshot | None = None
        self.health_status = HealthStatus.HEALTHY
        self.thermal_state = ThermalState.NORMAL
        self.battery_state = BatteryState.HIGH

        # Monitoring state
        self._running = False
        self._monitor_task: asyncio.Task | None = None
        self._baseline_metrics: dict[str, float] | None = None

        # Callbacks
        self.on_health_change = None
        self.on_thermal_change = None
        self.on_critical_resource = None

        # Previous values for rate calculations
        self._prev_disk_io: dict[str, int] | None = None
        self._prev_network_io: dict[str, int] | None = None
        self._prev_timestamp: float | None = None

    async def start_monitoring(self):
        """Start continuous resource monitoring"""

        if self._running:
            return

        self._running = True
        logger.info(f"Starting resource monitoring for device {self.device_id}")

        # Take initial baseline measurement
        await self._collect_baseline()

        # Start monitoring loop
        self._monitor_task = asyncio.create_task(self._monitoring_loop())

        # Start performance profiling if enabled
        if self.enable_profiling and not self.performance_profile:
            asyncio.create_task(self._profile_performance())

    async def stop_monitoring(self):
        """Stop resource monitoring"""

        if not self._running:
            return

        self._running = False
        logger.info("Stopping resource monitoring")

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitoring_loop(self):
        """Main monitoring loop"""

        while self._running:
            try:
                # Collect current metrics
                snapshot = await self._collect_snapshot()

                # Store snapshot
                self._store_snapshot(snapshot)

                # Update health assessments
                await self._assess_health(snapshot)

                # Clean up old data
                self._cleanup_old_snapshots()

                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5.0)

    async def _collect_snapshot(self) -> ResourceSnapshot:
        """Collect current resource utilization snapshot"""

        snapshot = ResourceSnapshot()
        current_time = time.time()

        # CPU metrics
        psutil.cpu_times()
        snapshot.cpu_percent = psutil.cpu_percent(interval=0.1)
        snapshot.cpu_cores_logical = psutil.cpu_count(logical=True)
        snapshot.cpu_cores_physical = psutil.cpu_count(logical=False)

        # CPU frequency
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                snapshot.cpu_freq_mhz = cpu_freq.current
        except Exception as e:
            import logging

            logging.exception("Exception in CPU frequency collection: %s", str(e))

        # CPU temperature
        try:
            temps = psutil.sensors_temperatures()
            if "coretemp" in temps:
                snapshot.cpu_temp_celsius = temps["coretemp"][0].current
            elif "cpu_thermal" in temps:
                snapshot.cpu_temp_celsius = temps["cpu_thermal"][0].current
        except Exception as e:
            import logging

            logging.exception("Exception in CPU temperature collection: %s", str(e))

        # Memory metrics
        memory = psutil.virtual_memory()
        snapshot.memory_total_mb = int(memory.total / (1024 * 1024))
        snapshot.memory_used_mb = int(memory.used / (1024 * 1024))
        snapshot.memory_percent = memory.percent
        snapshot.memory_available_mb = int(memory.available / (1024 * 1024))

        # Swap metrics
        swap = psutil.swap_memory()
        snapshot.swap_total_mb = int(swap.total / (1024 * 1024))
        snapshot.swap_used_mb = int(swap.used / (1024 * 1024))

        # Disk metrics
        disk = psutil.disk_usage("/")
        snapshot.disk_total_mb = int(disk.total / (1024 * 1024))
        snapshot.disk_used_mb = int(disk.used / (1024 * 1024))
        snapshot.disk_free_mb = int(disk.free / (1024 * 1024))
        snapshot.disk_percent = (disk.used / disk.total) * 100

        # Disk I/O rates
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io and self._prev_disk_io and self._prev_timestamp:
                time_delta = current_time - self._prev_timestamp
                read_delta = disk_io.read_bytes - self._prev_disk_io["read_bytes"]
                write_delta = disk_io.write_bytes - self._prev_disk_io["write_bytes"]

                snapshot.disk_read_mb_s = (read_delta / time_delta) / (1024 * 1024)
                snapshot.disk_write_mb_s = (write_delta / time_delta) / (1024 * 1024)

            if disk_io:
                self._prev_disk_io = {"read_bytes": disk_io.read_bytes, "write_bytes": disk_io.write_bytes}
        except Exception as e:
            import logging

            logging.exception("Exception in disk I/O metrics collection: %s", str(e))

        # Network metrics
        try:
            network_io = psutil.net_io_counters()
            if network_io and self._prev_network_io and self._prev_timestamp:
                time_delta = current_time - self._prev_timestamp
                sent_delta = network_io.bytes_sent - self._prev_network_io["bytes_sent"]
                recv_delta = network_io.bytes_recv - self._prev_network_io["bytes_recv"]

                snapshot.network_sent_mb_s = (sent_delta / time_delta) / (1024 * 1024)
                snapshot.network_recv_mb_s = (recv_delta / time_delta) / (1024 * 1024)

            if network_io:
                self._prev_network_io = {"bytes_sent": network_io.bytes_sent, "bytes_recv": network_io.bytes_recv}

            # Network connections
            snapshot.network_connections = len(psutil.net_connections())
        except Exception as e:
            import logging

            logging.exception("Exception in network metrics collection: %s", str(e))

        # Battery metrics
        try:
            battery = psutil.sensors_battery()
            if battery:
                snapshot.battery_percent = battery.percent
                snapshot.battery_time_left_s = (
                    battery.secsleft if battery.secsleft != psutil.POWER_TIME_UNLIMITED else None
                )
                snapshot.power_plugged = battery.power_plugged
        except Exception as e:
            import logging

            logging.exception("Exception in battery metrics collection: %s", str(e))

        # Process metrics
        try:
            snapshot.process_count = len(psutil.pids())
        except Exception as e:
            import logging

            logging.exception("Exception in process count collection: %s", str(e))

        # Load average (Unix/Linux only)
        try:
            load_avg = psutil.getloadavg()
            snapshot.load_avg_1m = load_avg[0]
            snapshot.load_avg_5m = load_avg[1]
            snapshot.load_avg_15m = load_avg[2]
        except (AttributeError, OSError):
            pass

        self._prev_timestamp = current_time
        return snapshot

    def _store_snapshot(self, snapshot: ResourceSnapshot):
        """Store snapshot in historical data"""

        self.snapshots.append(snapshot)
        self.current_snapshot = snapshot

        # Limit history size
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots :]

    async def _assess_health(self, snapshot: ResourceSnapshot):
        """Assess overall device health based on metrics"""

        # Calculate health score (0.0 = critical, 1.0 = perfect)
        health_score = 1.0

        # CPU health (high sustained usage is concerning)
        if snapshot.cpu_percent > 90:
            health_score -= 0.3
        elif snapshot.cpu_percent > 70:
            health_score -= 0.1

        # Memory health
        if snapshot.memory_percent > 90:
            health_score -= 0.4
        elif snapshot.memory_percent > 80:
            health_score -= 0.2

        # Disk health
        if snapshot.disk_percent > 95:
            health_score -= 0.3
        elif snapshot.disk_percent > 85:
            health_score -= 0.1

        # Thermal health
        thermal_state = ThermalState.NORMAL
        if snapshot.cpu_temp_celsius:
            if snapshot.cpu_temp_celsius > 85:
                thermal_state = ThermalState.CRITICAL
                health_score -= 0.5
            elif snapshot.cpu_temp_celsius > 75:
                thermal_state = ThermalState.HOT
                health_score -= 0.3
            elif snapshot.cpu_temp_celsius > 65:
                thermal_state = ThermalState.WARM
                health_score -= 0.1

        # Update thermal state
        if thermal_state != self.thermal_state:
            old_state = self.thermal_state
            self.thermal_state = thermal_state
            if self.on_thermal_change:
                await self.on_thermal_change(old_state, thermal_state)

        # Battery health
        battery_state = BatteryState.HIGH
        if snapshot.battery_percent is not None:
            if snapshot.power_plugged:
                battery_state = BatteryState.CHARGING
            elif snapshot.battery_percent < 10:
                battery_state = BatteryState.CRITICAL
                health_score -= 0.4
            elif snapshot.battery_percent < 30:
                battery_state = BatteryState.LOW
                health_score -= 0.2
            elif snapshot.battery_percent < 70:
                battery_state = BatteryState.MEDIUM
            elif snapshot.battery_percent > 90:
                battery_state = BatteryState.FULL

        self.battery_state = battery_state

        # Determine overall health status
        old_health = self.health_status

        if health_score >= 0.8:
            self.health_status = HealthStatus.HEALTHY
        elif health_score >= 0.5:
            self.health_status = HealthStatus.DEGRADED
        else:
            self.health_status = HealthStatus.CRITICAL

        # Notify of health changes
        if old_health != self.health_status:
            if self.on_health_change:
                await self.on_health_change(old_health, self.health_status)

        # Check for critical resource conditions
        critical_conditions = []

        if snapshot.memory_percent > 95:
            critical_conditions.append("memory_critical")
        if snapshot.disk_percent > 98:
            critical_conditions.append("disk_critical")
        if snapshot.cpu_temp_celsius and snapshot.cpu_temp_celsius > 90:
            critical_conditions.append("thermal_critical")
        if snapshot.battery_percent and snapshot.battery_percent < 5:
            critical_conditions.append("battery_critical")

        if critical_conditions and self.on_critical_resource:
            await self.on_critical_resource(critical_conditions, snapshot)

    async def _collect_baseline(self):
        """Collect baseline performance metrics"""

        logger.info("Collecting baseline performance metrics...")

        # Take several samples to establish baseline
        samples = []
        for _ in range(10):
            snapshot = await self._collect_snapshot()
            samples.append(snapshot)
            await asyncio.sleep(1.0)

        # Calculate baseline values
        self._baseline_metrics = {
            "cpu_percent_baseline": sum(s.cpu_percent for s in samples) / len(samples),
            "memory_percent_baseline": sum(s.memory_percent for s in samples) / len(samples),
            "disk_percent_baseline": samples[-1].disk_percent,  # Use latest disk usage
        }

        logger.info(f"Baseline metrics established: {self._baseline_metrics}")

    async def _profile_performance(self):
        """Profile device performance characteristics"""

        logger.info("Starting performance profiling...")

        profile_start = datetime.now(UTC)

        # Reference implementation: comprehensive performance profiling
        # This would include:
        # - CPU benchmarking (single/multi-core)
        # - Memory bandwidth testing
        # - Disk I/O benchmarking
        # - Network performance testing
        # - Thermal characterization under load

        # For now, create a basic profile based on hardware specs
        snapshot = self.current_snapshot or await self._collect_snapshot()

        self.performance_profile = PerformanceProfile(
            device_id=self.device_id,
            profiling_start=profile_start,
            profiling_duration_s=60.0,  # Mock profiling duration
            # Estimate performance based on hardware
            cpu_single_core_score=snapshot.cpu_freq_mhz / 10.0 if snapshot.cpu_freq_mhz else 300.0,
            cpu_multi_core_score=(
                (snapshot.cpu_freq_mhz * snapshot.cpu_cores_logical / 10.0) if snapshot.cpu_freq_mhz else 800.0
            ),
            memory_bandwidth_mb_s=15000.0,  # Typical DDR4 bandwidth
            disk_seq_read_mb_s=500.0,  # Typical SSD performance
            disk_seq_write_mb_s=400.0,
            network_bandwidth_mb_s=100.0,  # Typical WiFi bandwidth
            network_latency_ms=20.0,
            power_efficiency_score=1.0,
        )

        logger.info("Performance profiling completed")

    def _cleanup_old_snapshots(self):
        """Remove snapshots older than retention period"""

        cutoff_time = datetime.now(UTC) - self.history_retention

        # Remove old snapshots
        self.snapshots = [s for s in self.snapshots if s.timestamp > cutoff_time]

    def get_current_status(self) -> dict[str, Any]:
        """Get current device status summary"""

        snapshot = self.current_snapshot
        if not snapshot:
            return {"status": "no_data"}

        return {
            "device_id": self.device_id,
            "timestamp": snapshot.timestamp.isoformat(),
            "health_status": self.health_status.value,
            "thermal_state": self.thermal_state.value,
            "battery_state": self.battery_state.value,
            "cpu_percent": snapshot.cpu_percent,
            "memory_percent": snapshot.memory_percent,
            "disk_percent": snapshot.disk_percent,
            "cpu_temp_celsius": snapshot.cpu_temp_celsius,
            "battery_percent": snapshot.battery_percent,
            "available_resources": {
                "cpu_cores": snapshot.cpu_cores_logical,
                "memory_mb": snapshot.memory_total_mb - snapshot.memory_used_mb,
                "disk_mb": snapshot.disk_free_mb,
            },
        }

    def get_historical_data(self, hours: int = 1, metric: str = "cpu_percent") -> list[tuple[datetime, float]]:
        """Get historical data for a specific metric"""

        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)

        data_points = []
        for snapshot in self.snapshots:
            if snapshot.timestamp > cutoff_time:
                value = getattr(snapshot, metric, None)
                if value is not None:
                    data_points.append((snapshot.timestamp, value))

        return data_points

    def get_performance_profile(self) -> PerformanceProfile | None:
        """Get device performance profile"""
        return self.performance_profile

    def is_suitable_for_workload(
        self, cpu_requirement: float, memory_mb: int, duration_s: int
    ) -> tuple[bool, list[str]]:
        """
        Check if device is suitable for a workload

        Args:
            cpu_requirement: Required CPU cores
            memory_mb: Required memory in MB
            duration_s: Expected duration in seconds

        Returns:
            Tuple of (suitable, list_of_issues)
        """

        if not self.current_snapshot:
            return False, ["no_monitoring_data"]

        issues = []
        snapshot = self.current_snapshot

        # Check health status
        if self.health_status == HealthStatus.CRITICAL:
            issues.append("device_critical_health")
        elif self.health_status == HealthStatus.DEGRADED:
            issues.append("device_degraded_health")

        # Check thermal state
        if self.thermal_state == ThermalState.CRITICAL:
            issues.append("thermal_critical")
        elif self.thermal_state == ThermalState.HOT:
            issues.append("thermal_hot")

        # Check battery
        if self.battery_state == BatteryState.CRITICAL:
            issues.append("battery_critical")
        elif self.battery_state == BatteryState.LOW and duration_s > 600:
            issues.append("battery_low_for_duration")

        # Check CPU availability
        available_cpu = snapshot.cpu_cores_logical * (100 - snapshot.cpu_percent) / 100
        if available_cpu < cpu_requirement:
            issues.append("insufficient_cpu")

        # Check memory availability
        if snapshot.memory_available_mb < memory_mb:
            issues.append("insufficient_memory")

        # Check disk space (assume 10% of requirement for temp files)
        temp_disk_needed = memory_mb * 0.1
        if snapshot.disk_free_mb < temp_disk_needed:
            issues.append("insufficient_disk")

        suitable = len(issues) == 0
        return suitable, issues
