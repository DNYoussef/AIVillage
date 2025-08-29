"""
Fog Compute Harvest Manager - Idle Resource Collection System

Manages the collection of idle compute resources from mobile devices during charging.
Implements battery-aware, thermal-safe compute harvesting with tokenomics tracking.

Key Features:
- Detects optimal harvesting conditions (charging + battery > 20% + thermal safe)
- Tracks compute contributions for token rewards
- Integrates with P2P networks (BitChat/Betanet) for decentralized coordination
- Provides AWS-alternative fog compute marketplace backend
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import logging
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class HarvestState(Enum):
    """Device harvest state for compute collection"""

    IDLE = "idle"  # Device available but not harvesting
    HARVESTING = "harvesting"  # Actively contributing compute
    CHARGING_LOW = "charging_low"  # Charging but battery < 20%
    THERMAL_THROTTLE = "thermal_throttle"  # Too hot to harvest
    USER_ACTIVE = "user_active"  # User using device
    NETWORK_METERED = "network_metered"  # On cellular/metered connection
    DISABLED = "disabled"  # User disabled harvesting


class ComputeType(Enum):
    """Types of compute resources that can be harvested"""

    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    BANDWIDTH = "bandwidth"
    ML_INFERENCE = "ml_inference"


@dataclass
class HarvestPolicy:
    """Policy configuration for compute harvesting"""

    # Battery thresholds
    min_battery_percent: int = 20
    optimal_battery_percent: int = 50
    require_charging: bool = True

    # Thermal thresholds (Celsius)
    max_cpu_temp: float = 45.0
    throttle_temp: float = 55.0
    critical_temp: float = 65.0

    # Resource limits
    max_cpu_percent: float = 50.0  # Max CPU usage when harvesting
    max_memory_percent: float = 30.0  # Max memory usage
    max_bandwidth_mbps: float = 10.0  # Max network bandwidth

    # Network requirements
    require_wifi: bool = True
    allow_metered: bool = False
    min_bandwidth_mbps: float = 5.0

    # User activity
    require_screen_off: bool = True
    idle_timeout_minutes: int = 5

    # Scheduling
    harvest_hours: list[tuple[int, int]] = field(default_factory=lambda: [(22, 7)])  # 10pm-7am default
    max_harvest_duration_hours: int = 8
    cooldown_minutes: int = 30


@dataclass
class DeviceCapabilities:
    """Hardware capabilities of a fog computing device"""

    device_id: str
    device_type: str  # smartphone, tablet, laptop, desktop

    # CPU specs
    cpu_cores: int
    cpu_freq_mhz: int
    cpu_architecture: str  # arm64, x86_64, etc

    # Memory
    ram_total_mb: int
    ram_available_mb: int

    # Storage
    storage_total_gb: int
    storage_available_gb: int

    # GPU (if available)
    has_gpu: bool = False
    gpu_model: str | None = None
    gpu_memory_mb: int | None = None
    gpu_compute_capability: float | None = None

    # Network
    network_type: str = "wifi"  # wifi, 4g, 5g, ethernet
    network_speed_mbps: float = 10.0

    # Special capabilities
    supports_ml_inference: bool = False
    supports_webassembly: bool = True
    supports_docker: bool = False

    def compute_score(self) -> float:
        """Calculate device compute capability score (0.0-1.0)"""
        cpu_score = min(1.0, self.cpu_cores / 8) * 0.3
        memory_score = min(1.0, self.ram_total_mb / 8192) * 0.3
        network_score = min(1.0, self.network_speed_mbps / 100) * 0.2
        gpu_score = 0.2 if self.has_gpu else 0.0

        return cpu_score + memory_score + network_score + gpu_score


@dataclass
class HarvestSession:
    """Active compute harvesting session"""

    session_id: str
    device_id: str
    start_time: datetime
    end_time: datetime | None = None

    # Resources harvested
    cpu_cycles: int = 0
    memory_mb_hours: float = 0.0
    bandwidth_gb: float = 0.0
    storage_gb_hours: float = 0.0

    # Tasks completed
    tasks_completed: int = 0
    tasks_failed: int = 0

    # Quality metrics
    uptime_percent: float = 100.0
    average_latency_ms: float = 0.0

    # Token rewards
    tokens_earned: int = 0
    bonus_multiplier: float = 1.0

    def calculate_contribution_score(self) -> float:
        """Calculate overall contribution score for this session"""
        duration_hours = ((self.end_time or datetime.now(UTC)) - self.start_time).total_seconds() / 3600

        if duration_hours <= 0:
            return 0.0

        # Weighted scoring
        cpu_score = min(100, self.cpu_cycles / 1e9)  # Normalize to GHz-hours
        memory_score = self.memory_mb_hours / 1024  # Convert to GB-hours
        bandwidth_score = self.bandwidth_gb * 10  # Higher weight for bandwidth
        reliability_score = (self.uptime_percent / 100) * 50

        return cpu_score + memory_score + bandwidth_score + reliability_score


@dataclass
class ContributionLedger:
    """Tracks device contributions for tokenomics rewards"""

    device_id: str
    total_sessions: int = 0
    total_hours: float = 0.0

    # Cumulative resources contributed
    total_cpu_cycles: int = 0
    total_memory_gb_hours: float = 0.0
    total_bandwidth_gb: float = 0.0
    total_storage_gb_hours: float = 0.0

    # Task metrics
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    success_rate: float = 100.0

    # Quality metrics
    average_uptime: float = 100.0
    average_latency_ms: float = 0.0
    reliability_score: float = 1.0

    # Token economics
    total_tokens_earned: int = 0
    pending_tokens: int = 0
    last_payout: datetime | None = None

    # Reputation
    trust_score: float = 0.5  # 0.0-1.0
    verified_device: bool = False
    stake_amount: int = 0

    def update_from_session(self, session: HarvestSession):
        """Update ledger with completed session data"""
        self.total_sessions += 1
        duration_hours = ((session.end_time or datetime.now(UTC)) - session.start_time).total_seconds() / 3600
        self.total_hours += duration_hours

        # Update resource contributions
        self.total_cpu_cycles += session.cpu_cycles
        self.total_memory_gb_hours += session.memory_mb_hours / 1024
        self.total_bandwidth_gb += session.bandwidth_gb

        # Update task metrics
        self.total_tasks_completed += session.tasks_completed
        self.total_tasks_failed += session.tasks_failed

        # Update success rate (weighted average)
        total_tasks = self.total_tasks_completed + self.total_tasks_failed
        if total_tasks > 0:
            self.success_rate = (self.total_tasks_completed / total_tasks) * 100

        # Update tokens
        self.pending_tokens += session.tokens_earned


class FogHarvestManager:
    """
    Manages idle compute harvesting across fog network devices.
    Coordinates with P2P networks and handles tokenomics tracking.
    """

    def __init__(
        self,
        node_id: str,
        policy: HarvestPolicy | None = None,
        token_rate_per_hour: int = 100,
        enable_mixnet: bool = True,
    ):
        self.node_id = node_id
        self.policy = policy or HarvestPolicy()
        self.token_rate_per_hour = token_rate_per_hour
        self.enable_mixnet = enable_mixnet

        # Active devices and sessions
        self.registered_devices: dict[str, DeviceCapabilities] = {}
        self.active_sessions: dict[str, HarvestSession] = {}
        self.device_states: dict[str, HarvestState] = {}

        # Contribution tracking
        self.contribution_ledgers: dict[str, ContributionLedger] = {}

        # Task queues
        self.pending_tasks: list[dict[str, Any]] = []
        self.assigned_tasks: dict[str, list[str]] = {}  # device_id -> task_ids

        # Network coordination
        self.peer_nodes: set[str] = set()
        self.is_coordinator = False

        logger.info(f"FogHarvestManager initialized: {node_id}")

    async def register_device(
        self, device_id: str, capabilities: DeviceCapabilities, initial_state: dict[str, Any] | None = None
    ) -> bool:
        """Register a device for compute harvesting"""
        try:
            # Store device capabilities
            self.registered_devices[device_id] = capabilities

            # Initialize contribution ledger if new device
            if device_id not in self.contribution_ledgers:
                self.contribution_ledgers[device_id] = ContributionLedger(device_id)

            # Set initial state
            if initial_state:
                harvest_state = await self._evaluate_harvest_eligibility(device_id, initial_state)
                self.device_states[device_id] = harvest_state
            else:
                self.device_states[device_id] = HarvestState.IDLE

            logger.info(
                f"Registered device {device_id}: {capabilities.device_type}, "
                f"score: {capabilities.compute_score():.2f}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to register device {device_id}: {e}")
            return False

    async def _evaluate_harvest_eligibility(self, device_id: str, state: dict[str, Any]) -> HarvestState:
        """Evaluate if device is eligible for compute harvesting"""

        # Check battery level and charging status
        battery_percent = state.get("battery_percent", 0)
        is_charging = state.get("is_charging", False)

        if self.policy.require_charging and not is_charging:
            return HarvestState.IDLE

        if battery_percent < self.policy.min_battery_percent:
            return HarvestState.CHARGING_LOW

        # Check thermal state
        cpu_temp = state.get("cpu_temp_celsius", 0)
        if cpu_temp > self.policy.max_cpu_temp:
            return HarvestState.THERMAL_THROTTLE

        # Check network
        network_type = state.get("network_type", "unknown")
        if self.policy.require_wifi and network_type != "wifi":
            return HarvestState.NETWORK_METERED

        # Check user activity
        is_screen_on = state.get("screen_on", False)
        if self.policy.require_screen_off and is_screen_on:
            return HarvestState.USER_ACTIVE

        # Check time window
        current_hour = datetime.now().hour
        in_harvest_window = any(
            start <= current_hour < end or (start > end and (current_hour >= start or current_hour < end))
            for start, end in self.policy.harvest_hours
        )

        if not in_harvest_window:
            return HarvestState.IDLE

        return HarvestState.HARVESTING

    async def start_harvesting(self, device_id: str, state: dict[str, Any]) -> str | None:
        """Start a compute harvesting session for a device"""

        # Verify device is registered
        if device_id not in self.registered_devices:
            logger.warning(f"Unknown device attempted harvesting: {device_id}")
            return None

        # Check eligibility
        harvest_state = await self._evaluate_harvest_eligibility(device_id, state)
        self.device_states[device_id] = harvest_state

        if harvest_state != HarvestState.HARVESTING:
            logger.info(f"Device {device_id} not eligible for harvesting: {harvest_state.value}")
            return None

        # Check for existing session
        if device_id in self.active_sessions:
            logger.warning(f"Device {device_id} already has active session")
            return self.active_sessions[device_id].session_id

        # Create new harvest session
        session = HarvestSession(session_id=str(uuid4()), device_id=device_id, start_time=datetime.now(UTC))

        self.active_sessions[device_id] = session

        logger.info(f"Started harvesting session {session.session_id} for device {device_id}")
        return session.session_id

    async def update_session_metrics(self, device_id: str, metrics: dict[str, Any]) -> bool:
        """Update metrics for an active harvesting session"""

        if device_id not in self.active_sessions:
            logger.warning(f"No active session for device {device_id}")
            return False

        session = self.active_sessions[device_id]

        # Update resource contributions
        session.cpu_cycles += metrics.get("cpu_cycles", 0)
        session.memory_mb_hours += metrics.get("memory_mb_hours", 0)
        session.bandwidth_gb += metrics.get("bandwidth_gb", 0)
        session.storage_gb_hours += metrics.get("storage_gb_hours", 0)

        # Update task metrics
        session.tasks_completed += metrics.get("tasks_completed", 0)
        session.tasks_failed += metrics.get("tasks_failed", 0)

        # Update quality metrics
        if "latency_ms" in metrics:
            # Running average
            alpha = 0.1
            session.average_latency_ms = alpha * metrics["latency_ms"] + (1 - alpha) * session.average_latency_ms

        # Calculate earned tokens
        duration_hours = (datetime.now(UTC) - session.start_time).total_seconds() / 3600
        base_tokens = int(duration_hours * self.token_rate_per_hour)

        # Apply quality multiplier
        quality_multiplier = min(2.0, session.uptime_percent / 50)  # 2x max for perfect uptime
        session.tokens_earned = int(base_tokens * quality_multiplier * session.bonus_multiplier)

        return True

    async def stop_harvesting(self, device_id: str, reason: str = "user_requested") -> HarvestSession | None:
        """Stop a harvesting session and finalize contributions"""

        if device_id not in self.active_sessions:
            logger.warning(f"No active session to stop for device {device_id}")
            return None

        session = self.active_sessions[device_id]
        session.end_time = datetime.now(UTC)

        # Update contribution ledger
        ledger = self.contribution_ledgers[device_id]
        ledger.update_from_session(session)

        # Clean up
        del self.active_sessions[device_id]
        self.device_states[device_id] = HarvestState.IDLE

        logger.info(
            f"Stopped harvesting session {session.session_id}: "
            f"{session.tasks_completed} tasks, {session.tokens_earned} tokens earned"
        )

        return session

    async def assign_task(self, task: dict[str, Any], preferred_device: str | None = None) -> str | None:
        """Assign a compute task to an available device"""

        # Find eligible devices
        eligible_devices = [
            device_id
            for device_id, state in self.device_states.items()
            if state == HarvestState.HARVESTING and device_id in self.active_sessions
        ]

        if not eligible_devices:
            logger.warning("No eligible devices for task assignment")
            self.pending_tasks.append(task)
            return None

        # Select device based on capabilities and current load
        if preferred_device and preferred_device in eligible_devices:
            selected_device = preferred_device
        else:
            # Score devices based on capabilities and current load
            device_scores = {}
            for device_id in eligible_devices:
                capabilities = self.registered_devices[device_id]
                current_tasks = len(self.assigned_tasks.get(device_id, []))

                # Lower score for devices with more tasks
                load_penalty = current_tasks * 0.1
                device_scores[device_id] = capabilities.compute_score() - load_penalty

            # Select highest scoring device
            selected_device = max(device_scores, key=device_scores.get)

        # Assign task
        task_id = task.get("task_id", str(uuid4()))
        task["task_id"] = task_id
        task["assigned_device"] = selected_device
        task["assigned_time"] = datetime.now(UTC).isoformat()

        if selected_device not in self.assigned_tasks:
            self.assigned_tasks[selected_device] = []
        self.assigned_tasks[selected_device].append(task_id)

        logger.info(f"Assigned task {task_id} to device {selected_device}")
        return selected_device

    async def get_network_stats(self) -> dict[str, Any]:
        """Get current fog network statistics"""

        total_devices = len(self.registered_devices)
        active_devices = len(self.active_sessions)
        total_compute_score = sum(cap.compute_score() for cap in self.registered_devices.values())

        # Calculate total contributions
        total_cpu_cycles = sum(ledger.total_cpu_cycles for ledger in self.contribution_ledgers.values())
        total_tokens_earned = sum(ledger.total_tokens_earned for ledger in self.contribution_ledgers.values())

        # Device state distribution
        state_distribution = {}
        for state in HarvestState:
            count = sum(1 for s in self.device_states.values() if s == state)
            state_distribution[state.value] = count

        return {
            "network_id": self.node_id,
            "total_devices": total_devices,
            "active_devices": active_devices,
            "total_compute_score": total_compute_score,
            "total_cpu_cycles": total_cpu_cycles,
            "total_tokens_earned": total_tokens_earned,
            "pending_tasks": len(self.pending_tasks),
            "state_distribution": state_distribution,
            "coordinator": self.is_coordinator,
            "peer_nodes": len(self.peer_nodes),
        }

    async def export_contribution_data(self) -> dict[str, Any]:
        """Export contribution data for blockchain/DAO integration"""

        contributions = []
        for device_id, ledger in self.contribution_ledgers.items():
            contributions.append(
                {
                    "device_id": device_id,
                    "total_hours": ledger.total_hours,
                    "total_tasks": ledger.total_tasks_completed,
                    "success_rate": ledger.success_rate,
                    "reliability_score": ledger.reliability_score,
                    "tokens_earned": ledger.total_tokens_earned,
                    "pending_tokens": ledger.pending_tokens,
                    "trust_score": ledger.trust_score,
                }
            )

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "network_id": self.node_id,
            "total_contributors": len(contributions),
            "contributions": contributions,
        }
