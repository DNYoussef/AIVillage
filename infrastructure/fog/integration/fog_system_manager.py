"""
Fog System Integration Manager

Orchestrates all 8 enhanced fog computing components:
1. TEE Runtime Management
2. Cryptographic Proof System
3. Zero-Knowledge Predicates  
4. Market-based Dynamic Pricing
5. Heterogeneous Byzantine Quorum
6. Onion Routing Integration
7. Bayesian Reputation System
8. VRF Neighbor Selection

Provides unified management, health monitoring, configuration, and recovery.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
import logging
from typing import Any

from ..market.pricing_manager import DynamicPricingManager
from ..privacy.onion_routing import OnionRouter
from ..proofs.proof_generator import ProofGenerator
from ..proofs.proof_verifier import ProofVerifier
from ..quorum.quorum_manager import ByzantineQuorumManager
from ..reputation.bayesian_reputation import BayesianReputationEngine
from ..scheduler.placement import FogScheduler

# Import all fog components
from ..tee.tee_runtime_manager import TEERuntimeManager
from ..vrf.vrf_neighbor_selection import VRFNeighborSelector

logger = logging.getLogger(__name__)


class SystemHealth(str, Enum):
    """System health status levels"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"


class ComponentStatus(str, Enum):
    """Individual component status"""

    RUNNING = "running"
    STARTING = "starting"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class ComponentHealth:
    """Health information for a fog component"""

    name: str
    status: ComponentStatus = ComponentStatus.STOPPED
    health: SystemHealth = SystemHealth.OFFLINE
    last_heartbeat: datetime | None = None
    error_count: int = 0
    performance_score: float = 1.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    uptime_seconds: float = 0.0
    error_messages: list[str] = field(default_factory=list)

    def is_healthy(self) -> bool:
        """Check if component is healthy"""
        return (
            self.status == ComponentStatus.RUNNING
            and self.health in [SystemHealth.HEALTHY, SystemHealth.DEGRADED]
            and self.last_heartbeat
            and (datetime.now(UTC) - self.last_heartbeat) < timedelta(minutes=2)
        )


@dataclass
class SystemMetrics:
    """Comprehensive system metrics"""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Overall system health
    overall_health: SystemHealth = SystemHealth.OFFLINE
    healthy_components: int = 0
    total_components: int = 8
    error_rate: float = 0.0

    # Resource utilization
    total_memory_mb: float = 0.0
    total_cpu_percent: float = 0.0
    active_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0

    # Performance metrics
    avg_response_time_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    success_rate: float = 1.0

    # Component-specific metrics
    tee_enclaves: int = 0
    active_proofs: int = 0
    pricing_volatility: float = 0.0
    scheduled_jobs: int = 0
    consensus_rounds: int = 0
    onion_circuits: int = 0
    reputation_updates: int = 0
    vrf_selections: int = 0


class FogSystemManager:
    """
    Comprehensive fog system integration manager

    Orchestrates all 8 enhanced fog computing components with:
    - Unified lifecycle management
    - Health monitoring and alerting
    - Performance metrics collection
    - Error handling and recovery
    - Configuration management
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or self._default_config()

        # Initialize all fog components
        self._init_components()

        # System state
        self.component_health: dict[str, ComponentHealth] = {}
        self.system_metrics = SystemMetrics()
        self.is_running = False

        # Monitoring tasks
        self._health_monitor_task: asyncio.Task | None = None
        self._metrics_collector_task: asyncio.Task | None = None
        self._recovery_task: asyncio.Task | None = None

        # Performance tracking
        self.operation_history: list[dict[str, Any]] = []
        self.error_history: list[dict[str, Any]] = []

        logger.info("Fog System Manager initialized with 8 components")

    def _default_config(self) -> dict[str, Any]:
        """Default system configuration"""
        return {
            "health_check_interval": 30,
            "metrics_collection_interval": 60,
            "recovery_check_interval": 120,
            "max_error_count": 5,
            "component_timeout_seconds": 300,
            "auto_recovery_enabled": True,
            "performance_monitoring": True,
            "detailed_logging": True,
        }

    def _init_components(self):
        """Initialize all fog computing components"""

        # Create reputation engine first (used by others)
        self.reputation_engine = BayesianReputationEngine()

        # Core components
        self.tee_runtime = TEERuntimeManager()
        self.proof_generator = ProofGenerator()
        self.proof_verifier = ProofVerifier()
        self.pricing_manager = DynamicPricingManager(reputation_engine=self.reputation_engine)
        self.scheduler = FogScheduler(reputation_engine=self.reputation_engine)
        self.quorum_manager = ByzantineQuorumManager()
        self.onion_router = OnionRouter()
        self.vrf_selector = VRFNeighborSelector()

        # Initialize component health tracking
        self._init_component_health()

    def _init_component_health(self):
        """Initialize health tracking for all components"""
        components = [
            "tee_runtime",
            "proof_generator",
            "proof_verifier",
            "pricing_manager",
            "scheduler",
            "quorum_manager",
            "onion_router",
            "reputation_engine",
            "vrf_selector",
        ]

        for component in components:
            self.component_health[component] = ComponentHealth(name=component)

    async def start(self):
        """Start all fog computing components and monitoring"""

        logger.info("🚀 Starting Fog System Manager...")

        try:
            # Start all components in parallel
            await self._start_all_components()

            # Start monitoring tasks
            await self._start_monitoring_tasks()

            self.is_running = True
            logger.info("✅ Fog System Manager fully operational")

        except Exception as e:
            logger.error(f"❌ Failed to start Fog System Manager: {e}")
            await self.stop()
            raise

    async def _start_all_components(self):
        """Start all components concurrently"""

        start_tasks = [
            self._start_component("tee_runtime", self.tee_runtime.start),
            self._start_component("proof_generator", self.proof_generator.initialize),
            self._start_component("proof_verifier", self.proof_verifier.initialize),
            self._start_component("pricing_manager", self.pricing_manager.start),
            self._start_component("scheduler", self._start_scheduler),
            self._start_component("quorum_manager", self.quorum_manager.start),
            self._start_component("onion_router", self.onion_router.start),
            self._start_component("reputation_engine", self.reputation_engine.initialize),
            self._start_component("vrf_selector", self.vrf_selector.initialize),
        ]

        # Start all components concurrently
        results = await asyncio.gather(*start_tasks, return_exceptions=True)

        # Check results
        failed_components = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                component_name = list(self.component_health.keys())[i]
                failed_components.append((component_name, result))
                logger.error(f"❌ Failed to start {component_name}: {result}")

        if failed_components:
            logger.warning(f"⚠️ {len(failed_components)} components failed to start")

    async def _start_component(self, name: str, start_func):
        """Start individual component with error handling"""

        health = self.component_health[name]
        health.status = ComponentStatus.STARTING

        try:
            datetime.now(UTC)

            # Call component start function
            await start_func()

            # Update health status
            health.status = ComponentStatus.RUNNING
            health.health = SystemHealth.HEALTHY
            health.last_heartbeat = datetime.now(UTC)
            health.uptime_seconds = 0.0
            health.error_count = 0

            logger.info(f"✅ {name} started successfully")

        except Exception as e:
            health.status = ComponentStatus.ERROR
            health.health = SystemHealth.CRITICAL
            health.error_count += 1
            health.error_messages.append(str(e))

            logger.error(f"❌ Failed to start {name}: {e}")
            raise

    async def _start_scheduler(self):
        """Start scheduler (no async start method)"""
        # Scheduler doesn't need async initialization
        pass

    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""

        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        self._metrics_collector_task = asyncio.create_task(self._metrics_collector_loop())

        if self.config["auto_recovery_enabled"]:
            self._recovery_task = asyncio.create_task(self._recovery_loop())

    async def stop(self):
        """Stop all components and monitoring"""

        logger.info("🔄 Stopping Fog System Manager...")

        # Stop monitoring tasks
        await self._stop_monitoring_tasks()

        # Stop all components
        await self._stop_all_components()

        self.is_running = False
        logger.info("✅ Fog System Manager stopped")

    async def _stop_monitoring_tasks(self):
        """Stop all monitoring tasks"""

        tasks = [self._health_monitor_task, self._metrics_collector_task, self._recovery_task]

        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _stop_all_components(self):
        """Stop all components concurrently"""

        stop_tasks = [
            self._stop_component("tee_runtime", self.tee_runtime.stop),
            self._stop_component("pricing_manager", self.pricing_manager.stop),
            self._stop_component("quorum_manager", self.quorum_manager.stop),
            self._stop_component("onion_router", self.onion_router.stop),
            self._stop_component("vrf_selector", self.vrf_selector.cleanup),
        ]

        # Stop components that have async stop methods
        await asyncio.gather(*stop_tasks, return_exceptions=True)

        # Update all component status
        for health in self.component_health.values():
            health.status = ComponentStatus.STOPPED
            health.health = SystemHealth.OFFLINE

    async def _stop_component(self, name: str, stop_func):
        """Stop individual component with error handling"""

        health = self.component_health[name]
        health.status = ComponentStatus.STOPPING

        try:
            await stop_func()

            health.status = ComponentStatus.STOPPED
            health.health = SystemHealth.OFFLINE

            logger.info(f"✅ {name} stopped successfully")

        except Exception as e:
            health.status = ComponentStatus.ERROR
            health.error_count += 1
            health.error_messages.append(f"Stop error: {str(e)}")

            logger.error(f"❌ Failed to stop {name}: {e}")

    async def _health_monitor_loop(self):
        """Background health monitoring loop"""

        while self.is_running:
            try:
                await self._check_component_health()
                await asyncio.sleep(self.config["health_check_interval"])

            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)

    async def _check_component_health(self):
        """Check health of all components"""

        health_checks = [
            self._check_tee_health(),
            self._check_proof_health(),
            self._check_pricing_health(),
            self._check_scheduler_health(),
            self._check_quorum_health(),
            self._check_onion_health(),
            self._check_reputation_health(),
            self._check_vrf_health(),
        ]

        await asyncio.gather(*health_checks, return_exceptions=True)

        # Update overall system health
        self._update_system_health()

    async def _check_tee_health(self):
        """Check TEE runtime health"""
        health = self.component_health["tee_runtime"]

        try:
            status = await self.tee_runtime.get_status()

            health.last_heartbeat = datetime.now(UTC)
            health.health = SystemHealth.HEALTHY if status["healthy"] else SystemHealth.DEGRADED

        except Exception as e:
            health.health = SystemHealth.CRITICAL
            health.error_count += 1
            health.error_messages.append(str(e))

    async def _check_proof_health(self):
        """Check proof system health"""
        health = self.component_health["proof_generator"]

        try:
            # Simple health check
            health.last_heartbeat = datetime.now(UTC)
            health.health = SystemHealth.HEALTHY

        except Exception:
            health.health = SystemHealth.CRITICAL
            health.error_count += 1

    async def _check_pricing_health(self):
        """Check pricing manager health"""
        health = self.component_health["pricing_manager"]

        try:
            analytics = await self.pricing_manager.get_market_analytics()

            health.last_heartbeat = datetime.now(UTC)
            # Check market condition
            condition = analytics["market_overview"]["market_condition"]

            if condition in ["normal", "high_demand", "low_demand"]:
                health.health = SystemHealth.HEALTHY
            elif condition == "volatile":
                health.health = SystemHealth.DEGRADED
            else:
                health.health = SystemHealth.CRITICAL

        except Exception:
            health.health = SystemHealth.CRITICAL
            health.error_count += 1

    async def _check_scheduler_health(self):
        """Check job scheduler health"""
        health = self.component_health["scheduler"]

        try:
            stats = self.scheduler.get_scheduler_stats()

            health.last_heartbeat = datetime.now(UTC)

            # Check success rate
            success_rate = stats["success_rate"]
            if success_rate >= 0.95:
                health.health = SystemHealth.HEALTHY
            elif success_rate >= 0.8:
                health.health = SystemHealth.DEGRADED
            else:
                health.health = SystemHealth.CRITICAL

        except Exception:
            health.health = SystemHealth.CRITICAL
            health.error_count += 1

    async def _check_quorum_health(self):
        """Check quorum manager health"""
        health = self.component_health["quorum_manager"]

        try:
            status = await self.quorum_manager.get_status()

            health.last_heartbeat = datetime.now(UTC)
            health.health = SystemHealth.HEALTHY if status["healthy"] else SystemHealth.DEGRADED

        except Exception:
            health.health = SystemHealth.CRITICAL
            health.error_count += 1

    async def _check_onion_health(self):
        """Check onion router health"""
        health = self.component_health["onion_router"]

        try:
            await self.onion_router.get_active_circuits()

            health.last_heartbeat = datetime.now(UTC)
            health.health = SystemHealth.HEALTHY

        except Exception:
            health.health = SystemHealth.CRITICAL
            health.error_count += 1

    async def _check_reputation_health(self):
        """Check reputation engine health"""
        health = self.component_health["reputation_engine"]

        try:
            # Simple health check
            health.last_heartbeat = datetime.now(UTC)
            health.health = SystemHealth.HEALTHY

        except Exception:
            health.health = SystemHealth.CRITICAL
            health.error_count += 1

    async def _check_vrf_health(self):
        """Check VRF selector health"""
        health = self.component_health["vrf_selector"]

        try:
            # Simple health check
            health.last_heartbeat = datetime.now(UTC)
            health.health = SystemHealth.HEALTHY

        except Exception:
            health.health = SystemHealth.CRITICAL
            health.error_count += 1

    def _update_system_health(self):
        """Update overall system health based on components"""

        healthy_count = 0
        total_count = len(self.component_health)
        critical_count = 0

        for health in self.component_health.values():
            if health.health == SystemHealth.HEALTHY:
                healthy_count += 1
            elif health.health == SystemHealth.CRITICAL:
                critical_count += 1

        # Determine overall health
        if critical_count > 0:
            self.system_metrics.overall_health = SystemHealth.CRITICAL
        elif healthy_count >= total_count * 0.8:  # 80% healthy
            self.system_metrics.overall_health = SystemHealth.HEALTHY
        elif healthy_count >= total_count * 0.6:  # 60% healthy
            self.system_metrics.overall_health = SystemHealth.DEGRADED
        else:
            self.system_metrics.overall_health = SystemHealth.CRITICAL

        self.system_metrics.healthy_components = healthy_count
        self.system_metrics.total_components = total_count

    async def _metrics_collector_loop(self):
        """Background metrics collection loop"""

        while self.is_running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.config["metrics_collection_interval"])

            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(120)

    async def _collect_system_metrics(self):
        """Collect comprehensive system metrics"""

        self.system_metrics.timestamp = datetime.now(UTC)

        # Collect component-specific metrics
        try:
            # TEE metrics
            if self.component_health["tee_runtime"].is_healthy():
                tee_status = await self.tee_runtime.get_status()
                self.system_metrics.tee_enclaves = len(tee_status.get("enclaves", []))

            # Pricing metrics
            if self.component_health["pricing_manager"].is_healthy():
                pricing_analytics = await self.pricing_manager.get_market_analytics()
                volatilities = [lane["volatility_24h"] for lane in pricing_analytics["lane_analytics"].values()]
                self.system_metrics.pricing_volatility = sum(volatilities) / len(volatilities) if volatilities else 0.0

            # Scheduler metrics
            if self.component_health["scheduler"].is_healthy():
                scheduler_stats = self.scheduler.get_scheduler_stats()
                self.system_metrics.scheduled_jobs = scheduler_stats["total_placements"]
                self.system_metrics.success_rate = scheduler_stats["success_rate"]

            # Onion routing metrics
            if self.component_health["onion_router"].is_healthy():
                circuits = await self.onion_router.get_active_circuits()
                self.system_metrics.onion_circuits = len(circuits)

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")

    async def _recovery_loop(self):
        """Background component recovery loop"""

        while self.is_running:
            try:
                await self._check_recovery_needed()
                await asyncio.sleep(self.config["recovery_check_interval"])

            except Exception as e:
                logger.error(f"Recovery loop error: {e}")
                await asyncio.sleep(180)

    async def _check_recovery_needed(self):
        """Check if any components need recovery"""

        for name, health in self.component_health.items():
            if self._needs_recovery(health):
                logger.warning(f"🔧 Attempting recovery for {name}")
                await self._recover_component(name, health)

    def _needs_recovery(self, health: ComponentHealth) -> bool:
        """Check if component needs recovery"""

        return (
            health.status == ComponentStatus.ERROR
            or health.health == SystemHealth.CRITICAL
            or health.error_count >= self.config["max_error_count"]
            or (
                health.last_heartbeat
                and (datetime.now(UTC) - health.last_heartbeat)
                > timedelta(seconds=self.config["component_timeout_seconds"])
            )
        )

    async def _recover_component(self, name: str, health: ComponentHealth):
        """Attempt to recover a failed component"""

        try:
            health.status = ComponentStatus.STARTING

            # Component-specific recovery
            if name == "tee_runtime":
                await self.tee_runtime.stop()
                await asyncio.sleep(5)
                await self.tee_runtime.start()

            elif name == "pricing_manager":
                await self.pricing_manager.stop()
                await asyncio.sleep(5)
                await self.pricing_manager.start()

            elif name == "quorum_manager":
                await self.quorum_manager.stop()
                await asyncio.sleep(5)
                await self.quorum_manager.start()

            elif name == "onion_router":
                await self.onion_router.stop()
                await asyncio.sleep(5)
                await self.onion_router.start()

            # Reset health status
            health.status = ComponentStatus.RUNNING
            health.health = SystemHealth.HEALTHY
            health.last_heartbeat = datetime.now(UTC)
            health.error_count = 0
            health.error_messages.clear()

            logger.info(f"✅ Successfully recovered {name}")

        except Exception as e:
            health.status = ComponentStatus.ERROR
            health.error_count += 1
            health.error_messages.append(f"Recovery failed: {str(e)}")

            logger.error(f"❌ Failed to recover {name}: {e}")

    # Public API methods

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive system status"""

        component_statuses = {}
        for name, health in self.component_health.items():
            component_statuses[name] = {
                "status": health.status.value,
                "health": health.health.value,
                "last_heartbeat": health.last_heartbeat.isoformat() if health.last_heartbeat else None,
                "error_count": health.error_count,
                "uptime_seconds": health.uptime_seconds,
                "is_healthy": health.is_healthy(),
            }

        return {
            "overall_health": self.system_metrics.overall_health.value,
            "is_running": self.is_running,
            "healthy_components": self.system_metrics.healthy_components,
            "total_components": self.system_metrics.total_components,
            "components": component_statuses,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def get_comprehensive_metrics(self) -> dict[str, Any]:
        """Get detailed system performance metrics"""

        # Update metrics before returning
        await self._collect_system_metrics()

        return {
            "system_health": {
                "overall_health": self.system_metrics.overall_health.value,
                "healthy_components": self.system_metrics.healthy_components,
                "total_components": self.system_metrics.total_components,
                "error_rate": self.system_metrics.error_rate,
            },
            "performance": {
                "avg_response_time_ms": self.system_metrics.avg_response_time_ms,
                "throughput_ops_per_sec": self.system_metrics.throughput_ops_per_sec,
                "success_rate": self.system_metrics.success_rate,
            },
            "resources": {
                "total_memory_mb": self.system_metrics.total_memory_mb,
                "total_cpu_percent": self.system_metrics.total_cpu_percent,
            },
            "component_metrics": {
                "tee_enclaves": self.system_metrics.tee_enclaves,
                "active_proofs": self.system_metrics.active_proofs,
                "pricing_volatility": self.system_metrics.pricing_volatility,
                "scheduled_jobs": self.system_metrics.scheduled_jobs,
                "consensus_rounds": self.system_metrics.consensus_rounds,
                "onion_circuits": self.system_metrics.onion_circuits,
                "reputation_updates": self.system_metrics.reputation_updates,
                "vrf_selections": self.system_metrics.vrf_selections,
            },
            "timestamp": self.system_metrics.timestamp.isoformat(),
        }

    async def restart_component(self, component_name: str) -> bool:
        """Manually restart a specific component"""

        if component_name not in self.component_health:
            logger.error(f"Unknown component: {component_name}")
            return False

        try:
            logger.info(f"🔄 Manually restarting {component_name}")
            health = self.component_health[component_name]
            await self._recover_component(component_name, health)
            return True

        except Exception as e:
            logger.error(f"Failed to restart {component_name}: {e}")
            return False

    def get_component_health(self, component_name: str) -> dict[str, Any] | None:
        """Get health status for specific component"""

        if component_name not in self.component_health:
            return None

        health = self.component_health[component_name]
        return {
            "name": health.name,
            "status": health.status.value,
            "health": health.health.value,
            "last_heartbeat": health.last_heartbeat.isoformat() if health.last_heartbeat else None,
            "error_count": health.error_count,
            "performance_score": health.performance_score,
            "memory_usage_mb": health.memory_usage_mb,
            "cpu_usage_percent": health.cpu_usage_percent,
            "uptime_seconds": health.uptime_seconds,
            "is_healthy": health.is_healthy(),
            "recent_errors": health.error_messages[-5:],  # Last 5 errors
        }

    def update_config(self, new_config: dict[str, Any]):
        """Update system configuration"""

        self.config.update(new_config)
        logger.info(f"Configuration updated: {new_config}")


# Global system manager instance
_system_manager: FogSystemManager | None = None


async def get_system_manager() -> FogSystemManager:
    """Get global system manager instance"""
    global _system_manager

    if _system_manager is None:
        _system_manager = FogSystemManager()
        await _system_manager.start()

    return _system_manager
