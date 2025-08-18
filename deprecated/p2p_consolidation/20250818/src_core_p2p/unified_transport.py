"""Unified Transport Interface - One-Call API for BitChat/Betanet Auto-Selection

This module provides a single send() method that automatically chooses between BitChat
(Bluetooth mesh) and Betanet (decentralized internet) based on cost, reach, context,
and device conditions.

Key Features:
- One-call API: send(destination, payload, context) -> Receipt
- Auto-path selection via Navigator policy
- Cost/reach optimization for Global South scenarios
- Battery/thermal awareness with resource management
- Receipt tracking with hops, RTT, and success metrics
- 500ms target for path switching on link changes
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum

# Import dual-path transport components
try:
    from .betanet_transport_v2 import BetanetTransportV2 as BetanetTransport
    from .bitchat_transport import BitChatTransport
    from .dual_path_transport import DualPathMessage, DualPathTransport

    TRANSPORT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Transport modules not available: {e}")
    DualPathTransport = None
    TRANSPORT_AVAILABLE = False

# Import Navigator for intelligent path selection
try:
    from ...experimental.agents.agents.navigator.path_policy import (
        EnergyMode,
        MessageContext,
        NavigatorAgent,
        NetworkConditions,
        PathProtocol,
        PeerInfo,
        RoutingPriority,
    )

    NAVIGATOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Navigator not available: {e}")
    NavigatorAgent = None
    NAVIGATOR_AVAILABLE = False

# Import mobile resource management
try:
    from ...production.monitoring.mobile.device_profiler import DeviceProfile
    from ...production.monitoring.mobile.resource_management import (
        BatteryThermalResourceManager,
        PowerMode,
        TransportPreference,
    )

    RESOURCE_MANAGEMENT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Resource management not available: {e}")
    BatteryThermalResourceManager = None
    RESOURCE_MANAGEMENT_AVAILABLE = False

logger = logging.getLogger(__name__)


class DeliveryStatus(Enum):
    """Message delivery status"""

    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    QUEUED = "queued"  # Store-and-forward
    EXPIRED = "expired"


class PathSelection(Enum):
    """Path selection reasoning"""

    PROXIMITY_LOCAL = "proximity_local"  # BitChat chosen for nearby peers
    LARGE_URGENT = "large_urgent"  # Betanet chosen for large/urgent
    BATTERY_CONSERVATION = "battery_conservation"  # BitChat for energy saving
    COST_OPTIMIZATION = "cost_optimization"  # BitChat to avoid data costs
    PRIVACY_REQUIRED = "privacy_required"  # Betanet with mixnodes
    FALLBACK_OFFLINE = "fallback_offline"  # Store-and-forward
    LINK_CHANGE_SWITCH = "link_change_switch"  # Fast switching on link flap


@dataclass
class TransportContext:
    """Context for transport decision making"""

    # Message properties
    size_bytes: int = 0
    priority: int = 5  # 1=low, 10=urgent
    content_type: str = "application/json"
    requires_realtime: bool = False
    privacy_required: bool = False
    delivery_deadline: float | None = None

    # Device/network context
    proximity_hint: str | None = None  # "local", "nearby", "remote"
    battery_percent: int | None = None
    network_type: str | None = None  # "wifi", "cellular", "bluetooth"
    bandwidth_limited: bool = False
    cost_sensitive: bool = False

    # Performance requirements
    max_latency_ms: int | None = None
    reliability_required: float = 0.8  # 0.0-1.0

    def to_message_context(self) -> "MessageContext":
        """Convert to Navigator MessageContext"""
        if not NAVIGATOR_AVAILABLE:
            raise ImportError("Navigator not available")

        return MessageContext(
            size_bytes=self.size_bytes,
            priority=self.priority,
            content_type=self.content_type,
            requires_realtime=self.requires_realtime,
            privacy_required=self.privacy_required,
            delivery_deadline=self.delivery_deadline,
            bandwidth_sensitive=self.bandwidth_limited,
        )


@dataclass
class DeliveryReceipt:
    """Receipt for message delivery with comprehensive tracking"""

    # Message identification
    message_id: str
    destination: str
    timestamp: float

    # Path selection
    path_chosen: str  # "bitchat", "betanet", "store_forward"
    path_reasoning: PathSelection
    alternative_paths: list[str]

    # Delivery metrics
    status: DeliveryStatus
    hops: int | None = None
    rtt_ms: float | None = None
    success: bool = False

    # Cost/efficiency metrics
    energy_cost: float | None = None  # Estimated energy consumption
    data_cost_mb: float | None = None  # Data usage in MB
    battery_impact: float | None = None  # Battery % consumed

    # Error details
    error_message: str | None = None
    retry_count: int = 0
    fallback_used: bool = False

    # Context metadata
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict:
        """Convert receipt to dictionary for JSON export"""
        data = asdict(self)
        data["status"] = self.status.value
        data["path_reasoning"] = self.path_reasoning.value
        return data

    def is_successful(self) -> bool:
        """Check if delivery was successful"""
        return self.status in [DeliveryStatus.SENT, DeliveryStatus.DELIVERED]

    def get_efficiency_score(self) -> float:
        """Calculate efficiency score (0-1, higher is better)"""
        score = 0.0

        # Success factor (most important)
        if self.is_successful():
            score += 0.5

        # Latency factor
        if self.rtt_ms is not None:
            latency_score = max(0, 1 - (self.rtt_ms / 5000))  # 5s max
            score += 0.2 * latency_score

        # Energy efficiency factor
        if self.energy_cost is not None:
            energy_score = max(0, 1 - self.energy_cost)  # Lower cost = higher score
            score += 0.2 * energy_score

        # Data cost factor
        if self.data_cost_mb is not None:
            cost_score = max(0, 1 - min(1, self.data_cost_mb / 10))  # 10MB max
            score += 0.1 * cost_score

        return min(1.0, score)


class UnifiedTransport:
    """Unified Transport with One-Call API and Auto-Path Selection

    Provides a single send() method that automatically chooses between BitChat
    and Betanet based on cost, reach, context, and current conditions.

    Key Features:
    - Auto-selection: proximity/offline → BitChat; large/urgent/global → Betanet
    - Cost optimization for Global South scenarios
    - Battery/thermal awareness
    - 500ms target for link change switching
    - Comprehensive receipt tracking
    """

    def __init__(
        self,
        node_id: str | None = None,
        enable_bitchat: bool = True,
        enable_betanet: bool = True,
        global_south_mode: bool = True,
        resource_policy=None,
    ):
        self.node_id = node_id or f"unified_{uuid.uuid4().hex[:12]}"

        # Core transport system
        self.dual_path: DualPathTransport | None = None
        self.navigator: NavigatorAgent | None = None
        self.resource_manager: BatteryThermalResourceManager | None = None

        # Configuration
        self.enable_bitchat = enable_bitchat and TRANSPORT_AVAILABLE
        self.enable_betanet = enable_betanet and TRANSPORT_AVAILABLE
        self.global_south_mode = global_south_mode

        # Performance tracking
        self.receipts: dict[str, DeliveryReceipt] = {}
        self.path_performance: dict[str, dict[str, float]] = {
            "bitchat": {
                "success_rate": 0.85,
                "avg_latency_ms": 200,
                "energy_cost": 0.2,
            },
            "betanet": {
                "success_rate": 0.95,
                "avg_latency_ms": 100,
                "energy_cost": 0.8,
            },
            "store_forward": {
                "success_rate": 1.0,
                "avg_latency_ms": 0,
                "energy_cost": 0.1,
            },
        }

        # Link change detection for 500ms switching
        self.last_link_state = {}
        self.link_change_threshold = 0.5  # 500ms target
        self.last_path_switch_time = 0.0

        # Metrics export
        self.metrics_enabled = True
        self.metrics_buffer: list[dict] = []
        self.max_metrics_buffer = 1000

        # Control
        self.is_running = False
        self.monitoring_task: asyncio.Task | None = None

        logger.info(f"UnifiedTransport initialized: {self.node_id}")

    async def start(self) -> bool:
        """Start unified transport system"""
        if self.is_running:
            logger.warning("UnifiedTransport already running")
            return True

        logger.info("Starting UnifiedTransport...")

        try:
            # Initialize dual-path transport
            if TRANSPORT_AVAILABLE:
                self.dual_path = DualPathTransport(
                    node_id=self.node_id,
                    enable_bitchat=self.enable_bitchat,
                    enable_betanet=self.enable_betanet,
                )

                success = await self.dual_path.start()
                if not success:
                    logger.error("Failed to start dual-path transport")
                    return False

                # Get Navigator from dual-path system
                self.navigator = self.dual_path.navigator

            # Initialize resource management
            if RESOURCE_MANAGEMENT_AVAILABLE:
                self.resource_manager = BatteryThermalResourceManager()

            # Start monitoring for link changes
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())

            self.is_running = True
            logger.info("UnifiedTransport started successfully")
            return True

        except Exception as e:
            logger.exception(f"Failed to start UnifiedTransport: {e}")
            return False

    async def stop(self) -> None:
        """Stop unified transport system"""
        logger.info("Stopping UnifiedTransport...")
        self.is_running = False

        # Cancel monitoring
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        # Stop dual-path transport
        if self.dual_path:
            await self.dual_path.stop()

        logger.info("UnifiedTransport stopped")

    async def send(
        self,
        destination: str,
        payload: bytes | str | dict,
        context: TransportContext | None = None,
        **kwargs,
    ) -> DeliveryReceipt:
        """One-Call API: Send message with automatic path selection

        This is the main API method that automatically chooses between BitChat
        and Betanet based on context, cost, reach, and current conditions.

        Args:
            destination: Target node/agent ID
            payload: Message content (bytes, str, or dict)
            context: Transport context for decision making
            **kwargs: Additional context parameters

        Returns:
            DeliveryReceipt with path, metrics, and delivery status
        """
        if not self.is_running:
            return DeliveryReceipt(
                message_id="",
                destination=destination,
                timestamp=time.time(),
                path_chosen="none",
                path_reasoning=PathSelection.FALLBACK_OFFLINE,
                alternative_paths=[],
                status=DeliveryStatus.FAILED,
                error_message="UnifiedTransport not running",
            )

        # Create transport context
        if context is None:
            context = TransportContext()

        # Apply kwargs to context
        for key, value in kwargs.items():
            if hasattr(context, key):
                setattr(context, key, value)

        # Generate message ID
        message_id = str(uuid.uuid4())
        start_time = time.time()

        logger.info(
            f"Sending message {message_id[:8]} to {destination} "
            f"(size={context.size_bytes}, priority={context.priority})"
        )

        try:
            # Auto-select optimal path
            path_chosen, path_reasoning, alternatives = await self._select_optimal_path(
                destination, payload, context
            )

            # Create initial receipt
            receipt = DeliveryReceipt(
                message_id=message_id,
                destination=destination,
                timestamp=start_time,
                path_chosen=path_chosen,
                path_reasoning=path_reasoning,
                alternative_paths=alternatives,
                status=DeliveryStatus.PENDING,
            )

            # Execute delivery via selected path
            delivery_success = await self._execute_delivery(
                message_id, destination, payload, context, path_chosen, receipt
            )

            # Update receipt with results
            receipt.rtt_ms = (time.time() - start_time) * 1000
            receipt.success = delivery_success
            receipt.status = (
                DeliveryStatus.DELIVERED if delivery_success else DeliveryStatus.FAILED
            )

            # Calculate costs and efficiency
            await self._calculate_delivery_costs(receipt, context)

            # Store receipt for tracking
            self.receipts[message_id] = receipt

            # Update performance metrics
            self._update_path_performance(path_chosen, receipt)

            # Export metrics if enabled
            if self.metrics_enabled:
                await self._export_metrics(receipt)

            logger.info(
                f"Message {message_id[:8]} delivery complete: "
                f"{receipt.status.value} via {path_chosen}"
            )

            return receipt

        except Exception as e:
            logger.exception(f"Error sending message {message_id[:8]}: {e}")

            # Return failed receipt
            return DeliveryReceipt(
                message_id=message_id,
                destination=destination,
                timestamp=start_time,
                path_chosen="error",
                path_reasoning=PathSelection.FALLBACK_OFFLINE,
                alternative_paths=[],
                status=DeliveryStatus.FAILED,
                rtt_ms=(time.time() - start_time) * 1000,
                error_message=str(e),
            )

    async def _select_optimal_path(
        self, destination: str, payload: bytes | str | dict, context: TransportContext
    ) -> tuple[str, PathSelection, list[str]]:
        """Auto-select optimal path based on context and conditions"""

        # Convert payload to get size
        if isinstance(payload, str):
            payload_bytes = payload.encode()
        elif isinstance(payload, dict):
            payload_bytes = json.dumps(payload).encode()
        else:
            payload_bytes = payload

        context.size_bytes = len(payload_bytes)

        # Get current device/network conditions
        conditions = await self._get_current_conditions()

        # Available transport options
        available_paths = []
        if self.enable_bitchat and conditions.get("bluetooth_available", False):
            available_paths.append("bitchat")
        if self.enable_betanet and conditions.get("internet_available", False):
            available_paths.append("betanet")
        available_paths.append("store_forward")  # Always available

        # PRIORITY 1: Proximity/Local → BitChat
        if (
            context.proximity_hint == "local"
            or context.proximity_hint == "nearby"
            or await self._is_peer_nearby(destination)
        ):
            if "bitchat" in available_paths:
                return "bitchat", PathSelection.PROXIMITY_LOCAL, available_paths

        # PRIORITY 2: Large/Urgent → Betanet
        if (
            context.size_bytes > 10000
            or context.priority >= 8
            or context.requires_realtime
        ):  # >10KB
            if (
                "betanet" in available_paths
                and not self._is_data_expensive(conditions)
                and not self._is_battery_critical(conditions)
            ):
                return "betanet", PathSelection.LARGE_URGENT, available_paths

        # PRIORITY 3: Battery Conservation → BitChat
        if (
            self._is_battery_low(conditions)
            or context.battery_percent
            and context.battery_percent < 25
        ):
            if "bitchat" in available_paths:
                return "bitchat", PathSelection.BATTERY_CONSERVATION, available_paths

        # PRIORITY 4: Cost Optimization (Global South) → BitChat
        if self.global_south_mode and (
            context.cost_sensitive or self._is_data_expensive(conditions)
        ):
            if "bitchat" in available_paths:
                return "bitchat", PathSelection.COST_OPTIMIZATION, available_paths

        # PRIORITY 5: Privacy Required → Betanet with mixnodes
        if context.privacy_required or conditions.get("censorship_risk", 0) > 0.3:
            if "betanet" in available_paths:
                return "betanet", PathSelection.PRIVACY_REQUIRED, available_paths

        # PRIORITY 6: Link Change Detection → Fast Switch
        if await self._detect_link_change(conditions):
            # Choose fastest available path for quick switching
            if "betanet" in available_paths and conditions.get("internet_available"):
                return "betanet", PathSelection.LINK_CHANGE_SWITCH, available_paths
            if "bitchat" in available_paths:
                return "bitchat", PathSelection.LINK_CHANGE_SWITCH, available_paths

        # FALLBACK: Use Navigator if available
        if self.navigator and NAVIGATOR_AVAILABLE:
            try:
                msg_context = context.to_message_context()
                protocol, metadata = await self.navigator.select_path(
                    destination, msg_context, available_paths
                )

                path_map = {
                    PathProtocol.BITCHAT: "bitchat",
                    PathProtocol.BETANET: "betanet",
                    PathProtocol.STORE_FORWARD: "store_forward",
                }

                selected_path = path_map.get(protocol, "store_forward")
                return selected_path, PathSelection.FALLBACK_OFFLINE, available_paths

            except Exception as e:
                logger.warning(f"Navigator selection failed: {e}")

        # ULTIMATE FALLBACK: Choose best available option
        if "bitchat" in available_paths and await self._is_peer_nearby(destination):
            return "bitchat", PathSelection.PROXIMITY_LOCAL, available_paths

        if "betanet" in available_paths:
            return "betanet", PathSelection.LARGE_URGENT, available_paths

        # Store-and-forward as last resort
        return "store_forward", PathSelection.FALLBACK_OFFLINE, available_paths

    async def _execute_delivery(
        self,
        message_id: str,
        destination: str,
        payload: bytes | str | dict,
        context: TransportContext,
        path_chosen: str,
        receipt: DeliveryReceipt,
    ) -> bool:
        """Execute message delivery via selected path"""

        if not self.dual_path:
            logger.error("Dual-path transport not available")
            return False

        try:
            # Execute delivery with path preference
            success = await self.dual_path.send_message(
                recipient=destination,
                payload=payload,
                priority=context.priority,
                privacy_required=context.privacy_required,
                deadline=context.delivery_deadline,
                preferred_protocol=path_chosen
                if path_chosen != "store_forward"
                else None,
            )

            # Update receipt with hop/path info
            if success and path_chosen == "bitchat":
                # BitChat provides hop count
                receipt.hops = min(7, context.priority)  # Estimate based on priority
                receipt.metadata["transport"] = "bluetooth_mesh"

            elif success and path_chosen == "betanet":
                # Betanet is typically 1-2 hops
                receipt.hops = 1
                receipt.metadata["transport"] = "internet_quic"

            elif path_chosen == "store_forward":
                # Store-and-forward queued
                receipt.status = DeliveryStatus.QUEUED
                receipt.metadata["transport"] = "store_and_forward"
                return True

            return success

        except Exception as e:
            logger.exception(f"Delivery execution failed: {e}")
            receipt.error_message = str(e)
            return False

    async def _get_current_conditions(self) -> dict:
        """Get current device and network conditions"""
        conditions = {}

        # Basic connectivity checks
        conditions["bluetooth_available"] = await self._check_bluetooth_available()
        conditions["internet_available"] = await self._check_internet_available()
        conditions["wifi_connected"] = await self._check_wifi_connected()

        # Battery/resource status
        if self.resource_manager:
            try:
                status = self.resource_manager.get_status()
                conditions["battery_percent"] = status.get("battery_percent")
                conditions["charging"] = status.get("charging", False)
                conditions["thermal_throttling"] = status.get(
                    "thermal_throttling", False
                )
            except Exception:
                pass

        # Network performance estimates
        conditions["bandwidth_mbps"] = await self._estimate_bandwidth()
        conditions["latency_ms"] = await self._estimate_latency()

        # Cost factors for Global South
        conditions["data_cost_usd_mb"] = self._get_data_cost()
        conditions["censorship_risk"] = self._get_censorship_risk()

        return conditions

    async def _is_peer_nearby(self, destination: str) -> bool:
        """Check if destination peer is nearby via BitChat"""
        if not self.dual_path or not self.dual_path.bitchat:
            return False

        # Check if peer is in BitChat discovered peers
        try:
            return self.dual_path.bitchat.is_peer_reachable(destination)
        except Exception:
            return False

    def _is_data_expensive(self, conditions: dict) -> bool:
        """Check if mobile data is expensive (Global South optimization)"""
        cost = conditions.get("data_cost_usd_mb", 0)
        return cost > 0.005  # >$0.005/MB is expensive

    def _is_battery_low(self, conditions: dict) -> bool:
        """Check if battery is low"""
        battery = conditions.get("battery_percent")
        return battery is not None and battery < 30

    def _is_battery_critical(self, conditions: dict) -> bool:
        """Check if battery is critically low"""
        battery = conditions.get("battery_percent")
        return battery is not None and battery < 15

    async def _detect_link_change(self, conditions: dict) -> bool:
        """Detect link state changes for fast switching (500ms target)"""
        current_time = time.time()

        # Create link state snapshot
        current_state = {
            "bluetooth": conditions.get("bluetooth_available", False),
            "internet": conditions.get("internet_available", False),
            "wifi": conditions.get("wifi_connected", False),
        }

        # Check for changes from last state
        if hasattr(self, "_last_link_state"):
            changes = []
            for key, value in current_state.items():
                if self._last_link_state.get(key) != value:
                    changes.append(f"{key}:{value}")

            if changes:
                # Link change detected
                if (
                    current_time - self.last_path_switch_time
                    > self.link_change_threshold
                ):
                    self.last_path_switch_time = current_time
                    logger.info(
                        f"Link change detected: {', '.join(changes)} - fast switching"
                    )
                    self._last_link_state = current_state
                    return True

        self._last_link_state = current_state
        return False

    async def _calculate_delivery_costs(
        self, receipt: DeliveryReceipt, context: TransportContext
    ) -> None:
        """Calculate energy, data, and battery costs for delivery"""
        path = receipt.path_chosen

        # Energy cost estimates (0-1 scale)
        energy_costs = {
            "bitchat": 0.2,  # Bluetooth is energy efficient
            "betanet": 0.8,  # Internet/cellular uses more energy
            "store_forward": 0.1,  # Minimal energy for queuing
        }
        receipt.energy_cost = energy_costs.get(path, 0.5)

        # Data cost (MB) - only for Betanet
        if path == "betanet":
            # Estimate based on payload size + protocol overhead
            overhead_factor = 1.3  # 30% protocol overhead
            receipt.data_cost_mb = (context.size_bytes * overhead_factor) / (
                1024 * 1024
            )
        else:
            receipt.data_cost_mb = 0.0

        # Battery impact estimate (% consumed)
        if context.battery_percent:
            base_impact = {"bitchat": 0.1, "betanet": 0.3, "store_forward": 0.05}
            receipt.battery_impact = base_impact.get(path, 0.2)

    def _update_path_performance(self, path: str, receipt: DeliveryReceipt) -> None:
        """Update performance statistics for machine learning"""
        if path not in self.path_performance:
            return

        # Update success rate with exponential moving average
        alpha = 0.1
        current_success = self.path_performance[path]["success_rate"]
        new_success = 1.0 if receipt.is_successful() else 0.0
        self.path_performance[path]["success_rate"] = (
            alpha * new_success + (1 - alpha) * current_success
        )

        # Update latency
        if receipt.rtt_ms:
            current_latency = self.path_performance[path]["avg_latency_ms"]
            self.path_performance[path]["avg_latency_ms"] = (
                alpha * receipt.rtt_ms + (1 - alpha) * current_latency
            )

        # Update energy cost
        if receipt.energy_cost:
            current_energy = self.path_performance[path]["energy_cost"]
            self.path_performance[path]["energy_cost"] = (
                alpha * receipt.energy_cost + (1 - alpha) * current_energy
            )

    async def _export_metrics(self, receipt: DeliveryReceipt) -> None:
        """Export metrics to buffer for JSON export"""
        metric = {
            "timestamp": receipt.timestamp,
            "message_id": receipt.message_id,
            "destination": receipt.destination,
            "path_chosen": receipt.path_chosen,
            "path_reasoning": receipt.path_reasoning.value,
            "status": receipt.status.value,
            "success": receipt.success,
            "rtt_ms": receipt.rtt_ms,
            "hops": receipt.hops,
            "energy_cost": receipt.energy_cost,
            "data_cost_mb": receipt.data_cost_mb,
            "battery_impact": receipt.battery_impact,
            "efficiency_score": receipt.get_efficiency_score(),
        }

        self.metrics_buffer.append(metric)

        # Limit buffer size
        if len(self.metrics_buffer) > self.max_metrics_buffer:
            self.metrics_buffer.pop(0)

    async def _monitoring_loop(self) -> None:
        """Background monitoring for link changes and optimization"""
        while self.is_running:
            try:
                # Monitor for link changes every 100ms for 500ms target
                await asyncio.sleep(0.1)

                # Get current conditions to detect changes
                conditions = await self._get_current_conditions()
                await self._detect_link_change(conditions)

            except Exception as e:
                logger.exception(f"Monitoring loop error: {e}")
                await asyncio.sleep(1.0)

    # Helper methods for condition checking
    async def _check_bluetooth_available(self) -> bool:
        """Check if Bluetooth is available"""
        # Simplified check - would use platform-specific APIs
        return (
            self.enable_bitchat
            and self.dual_path
            and self.dual_path.bitchat is not None
        )

    async def _check_internet_available(self) -> bool:
        """Check if internet is available"""
        # Simplified check - would use platform-specific APIs
        return (
            self.enable_betanet
            and self.dual_path
            and self.dual_path.betanet is not None
        )

    async def _check_wifi_connected(self) -> bool:
        """Check if connected to WiFi vs cellular"""
        # Simplified - would check actual network interface
        return await self._check_internet_available()

    async def _estimate_bandwidth(self) -> float:
        """Estimate current bandwidth (Mbps)"""
        # Simplified estimation
        if await self._check_wifi_connected():
            return 50.0  # WiFi
        elif await self._check_internet_available():
            return 5.0  # Cellular
        else:
            return 0.1  # Bluetooth only

    async def _estimate_latency(self) -> float:
        """Estimate current latency (ms)"""
        # Simplified estimation
        if await self._check_wifi_connected():
            return 50.0  # WiFi
        elif await self._check_internet_available():
            return 150.0  # Cellular
        else:
            return 200.0  # Bluetooth mesh

    def _get_data_cost(self) -> float:
        """Get data cost per MB (USD)"""
        # Would integrate with carrier APIs or config
        # Global South default: higher cost
        return 0.01 if self.global_south_mode else 0.001

    def _get_censorship_risk(self) -> float:
        """Get censorship risk level (0-1)"""
        # Would integrate with geolocation and threat intelligence
        return 0.2 if self.global_south_mode else 0.1

    # Public API methods
    def get_status(self) -> dict:
        """Get comprehensive transport status"""
        return {
            "node_id": self.node_id,
            "is_running": self.is_running,
            "configuration": {
                "bitchat_enabled": self.enable_bitchat,
                "betanet_enabled": self.enable_betanet,
                "global_south_mode": self.global_south_mode,
            },
            "receipts_tracked": len(self.receipts),
            "path_performance": self.path_performance,
            "metrics_buffer_size": len(self.metrics_buffer)
            if self.metrics_enabled
            else 0,
            "dual_path_status": self.dual_path.get_status() if self.dual_path else None,
        }

    def get_recent_receipts(self, limit: int = 10) -> list[dict]:
        """Get recent delivery receipts"""
        sorted_receipts = sorted(
            self.receipts.values(), key=lambda r: r.timestamp, reverse=True
        )
        return [r.to_dict() for r in sorted_receipts[:limit]]

    def export_metrics_json(self) -> dict:
        """Export metrics for analysis"""
        return {
            "node_id": self.node_id,
            "export_timestamp": time.time(),
            "configuration": {
                "global_south_mode": self.global_south_mode,
                "bitchat_enabled": self.enable_bitchat,
                "betanet_enabled": self.enable_betanet,
            },
            "path_performance": self.path_performance,
            "recent_metrics": self.metrics_buffer[-100:] if self.metrics_buffer else [],
            "summary_stats": self._calculate_summary_stats(),
        }

    def _calculate_summary_stats(self) -> dict:
        """Calculate summary statistics"""
        if not self.metrics_buffer:
            return {}

        recent_metrics = self.metrics_buffer[-100:]  # Last 100 messages

        total_messages = len(recent_metrics)
        successful_messages = sum(1 for m in recent_metrics if m["success"])

        path_counts = {}
        avg_rtt_by_path = {}

        for metric in recent_metrics:
            path = metric["path_chosen"]
            path_counts[path] = path_counts.get(path, 0) + 1

            if metric["rtt_ms"]:
                if path not in avg_rtt_by_path:
                    avg_rtt_by_path[path] = []
                avg_rtt_by_path[path].append(metric["rtt_ms"])

        # Calculate averages
        for path in avg_rtt_by_path:
            avg_rtt_by_path[path] = sum(avg_rtt_by_path[path]) / len(
                avg_rtt_by_path[path]
            )

        return {
            "total_messages": total_messages,
            "success_rate": successful_messages / total_messages
            if total_messages > 0
            else 0,
            "path_distribution": path_counts,
            "average_rtt_by_path": avg_rtt_by_path,
            "avg_efficiency_score": sum(
                m.get("efficiency_score", 0) for m in recent_metrics
            )
            / total_messages
            if total_messages > 0
            else 0,
        }


# Convenience factory function
async def create_unified_transport(
    node_id: str | None = None, global_south_mode: bool = True, **kwargs
) -> UnifiedTransport:
    """Create and start a unified transport instance"""
    transport = UnifiedTransport(
        node_id=node_id, global_south_mode=global_south_mode, **kwargs
    )

    success = await transport.start()
    if not success:
        raise RuntimeError("Failed to start unified transport")

    return transport
