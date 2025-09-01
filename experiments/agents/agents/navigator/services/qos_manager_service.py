"""QoS Manager Service - Quality of Service management and adaptation

This service manages QoS parameters, adapts bandwidth usage, and prioritizes
traffic for optimal routing performance in the Navigator system.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

from ..interfaces.routing_interfaces import IQoSManagerService, RoutingEvent
from ..events.event_bus import get_event_bus
from ..path_policy import PathProtocol, MessageContext

logger = logging.getLogger(__name__)


class QoSLevel(Enum):
    """QoS service levels"""

    BEST_EFFORT = "best_effort"
    ASSURED = "assured"
    PREMIUM = "premium"
    CRITICAL = "critical"


class TrafficClass(Enum):
    """Traffic classification for QoS"""

    BULK = "bulk"  # Low priority bulk data
    STANDARD = "standard"  # Normal application traffic
    INTERACTIVE = "interactive"  # Interactive applications
    REALTIME = "realtime"  # Real-time traffic (voice, video)
    CONTROL = "control"  # Network control traffic


@dataclass
class QoSParameters:
    """QoS parameters for a connection or flow"""

    max_bandwidth_mbps: float = 10.0
    guaranteed_bandwidth_mbps: float = 1.0
    max_latency_ms: float = 200.0
    max_jitter_ms: float = 20.0
    max_packet_loss_rate: float = 0.05
    priority: int = 5  # 1-10, higher = more priority
    traffic_class: TrafficClass = TrafficClass.STANDARD
    qos_level: QoSLevel = QoSLevel.BEST_EFFORT


@dataclass
class TrafficFlow:
    """Represents a traffic flow with QoS requirements"""

    flow_id: str
    destination: str
    protocol: PathProtocol
    qos_params: QoSParameters
    current_bandwidth: float = 0.0
    current_latency: float = 0.0
    packet_count: int = 0
    bytes_transferred: int = 0
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    violations: List[str] = field(default_factory=list)


class QoSManagerService(IQoSManagerService):
    """Quality of Service management and adaptation service

    Manages:
    - QoS parameter enforcement and monitoring
    - Adaptive bandwidth allocation and traffic shaping
    - Traffic prioritization and queuing
    - SLA violation detection and response
    - Protocol-specific QoS optimization
    """

    def __init__(self):
        self.event_bus = get_event_bus()

        # Traffic flow management
        self.active_flows: Dict[str, TrafficFlow] = {}
        self.flow_history: deque[TrafficFlow] = deque(maxlen=1000)

        # QoS enforcement
        self.total_allocated_bandwidth = 0.0
        self.available_bandwidth = 100.0  # Default available bandwidth
        self.bandwidth_reservation_ratio = 0.8  # Reserve 80% max

        # Traffic classification and prioritization
        self.traffic_queues: Dict[TrafficClass, List[str]] = {tc: [] for tc in TrafficClass}
        self.queue_weights: Dict[TrafficClass, float] = {
            TrafficClass.CONTROL: 0.3,
            TrafficClass.REALTIME: 0.25,
            TrafficClass.INTERACTIVE: 0.25,
            TrafficClass.STANDARD: 0.15,
            TrafficClass.BULK: 0.05,
        }

        # Adaptive QoS
        self.adaptation_enabled = True
        self.adaptation_interval = 10.0  # Adapt every 10 seconds
        self.last_adaptation = 0.0
        self.congestion_threshold = 0.85  # 85% utilization triggers adaptation

        # Performance tracking
        self.qos_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.violation_counts: Dict[str, int] = defaultdict(int)
        self.adaptation_history: List[Dict[str, Any]] = []

        # Protocol-specific QoS profiles
        self.protocol_qos_profiles: Dict[PathProtocol, Dict[str, Any]] = {
            PathProtocol.SCION: {
                "supports_guaranteed_bandwidth": True,
                "supports_low_latency": True,
                "supports_traffic_classes": True,
                "max_concurrent_flows": 100,
                "default_qos_level": QoSLevel.PREMIUM,
            },
            PathProtocol.BETANET: {
                "supports_guaranteed_bandwidth": True,
                "supports_low_latency": True,
                "supports_traffic_classes": True,
                "max_concurrent_flows": 50,
                "default_qos_level": QoSLevel.ASSURED,
            },
            PathProtocol.BITCHAT: {
                "supports_guaranteed_bandwidth": False,
                "supports_low_latency": False,
                "supports_traffic_classes": False,
                "max_concurrent_flows": 10,
                "default_qos_level": QoSLevel.BEST_EFFORT,
            },
            PathProtocol.STORE_FORWARD: {
                "supports_guaranteed_bandwidth": False,
                "supports_low_latency": False,
                "supports_traffic_classes": False,
                "max_concurrent_flows": 1000,
                "default_qos_level": QoSLevel.BEST_EFFORT,
            },
        }

        logger.info("QoSManagerService initialized")

    async def manage_qos_parameters(self, protocol: PathProtocol, context: MessageContext) -> Dict[str, Any]:
        """Manage QoS parameters for protocol and message context"""
        # Determine appropriate QoS parameters based on context
        qos_params = self._determine_qos_parameters(protocol, context)

        # Create or update traffic flow
        flow_id = f"{protocol.value}_{context.content_type}_{hash(str(context)) % 10000:04d}"
        flow = self._create_or_update_flow(flow_id, protocol, context, qos_params)

        # Allocate bandwidth if needed
        bandwidth_allocated = await self._allocate_bandwidth(flow)

        # Apply traffic shaping and prioritization
        shaping_config = await self._apply_traffic_shaping(flow)

        # Monitor QoS compliance
        compliance_status = self._check_qos_compliance(flow)

        qos_config = {
            "flow_id": flow_id,
            "qos_parameters": {
                "max_bandwidth_mbps": qos_params.max_bandwidth_mbps,
                "guaranteed_bandwidth_mbps": qos_params.guaranteed_bandwidth_mbps,
                "max_latency_ms": qos_params.max_latency_ms,
                "priority": qos_params.priority,
                "traffic_class": qos_params.traffic_class.value,
                "qos_level": qos_params.qos_level.value,
            },
            "bandwidth_allocated": bandwidth_allocated,
            "shaping_config": shaping_config,
            "compliance_status": compliance_status,
            "protocol_capabilities": self.protocol_qos_profiles[protocol],
        }

        # Emit QoS management event
        self._emit_qos_event(
            "qos_parameters_managed", {"protocol": protocol.value, "flow_id": flow_id, "qos_config": qos_config}
        )

        logger.debug(f"QoS parameters managed for {protocol.value}: {qos_params.qos_level.value}")

        return qos_config

    def _determine_qos_parameters(self, protocol: PathProtocol, context: MessageContext) -> QoSParameters:
        """Determine appropriate QoS parameters based on protocol and context"""
        profile = self.protocol_qos_profiles[protocol]

        # Start with protocol defaults
        qos_params = QoSParameters(qos_level=profile["default_qos_level"])

        # Adjust based on message context
        if context.requires_realtime:
            qos_params.qos_level = QoSLevel.CRITICAL
            qos_params.traffic_class = TrafficClass.REALTIME
            qos_params.max_latency_ms = 50.0
            qos_params.max_jitter_ms = 5.0
            qos_params.priority = 9
            qos_params.guaranteed_bandwidth_mbps = min(5.0, context.size_bytes / 1000.0)

        elif context.priority >= 8:
            qos_params.qos_level = QoSLevel.PREMIUM
            qos_params.traffic_class = TrafficClass.INTERACTIVE
            qos_params.max_latency_ms = 100.0
            qos_params.priority = context.priority
            qos_params.guaranteed_bandwidth_mbps = min(2.0, context.size_bytes / 2000.0)

        elif context.priority >= 6:
            qos_params.qos_level = QoSLevel.ASSURED
            qos_params.traffic_class = TrafficClass.STANDARD
            qos_params.max_latency_ms = 200.0
            qos_params.priority = context.priority

        else:
            qos_params.qos_level = QoSLevel.BEST_EFFORT
            qos_params.traffic_class = TrafficClass.BULK
            qos_params.max_latency_ms = 500.0
            qos_params.priority = max(1, context.priority)

        # Adjust for large messages
        if context.is_large_message():
            qos_params.max_bandwidth_mbps = min(20.0, context.size_bytes / 5000.0)
            if qos_params.traffic_class == TrafficClass.BULK:
                qos_params.guaranteed_bandwidth_mbps = 0.5  # Minimum for bulk transfers

        # Protocol-specific adjustments
        if protocol == PathProtocol.BITCHAT:
            # BitChat has limited QoS capabilities
            qos_params.max_bandwidth_mbps = 0.1  # Very limited bandwidth
            qos_params.max_latency_ms = max(200.0, qos_params.max_latency_ms)
            qos_params.guaranteed_bandwidth_mbps = 0.01  # Minimal guarantee

        elif protocol == PathProtocol.STORE_FORWARD:
            # Store-and-forward doesn't have real-time guarantees
            qos_params.max_latency_ms = 0.0  # Delivered when possible
            qos_params.guaranteed_bandwidth_mbps = 0.0  # No bandwidth guarantees
            qos_params.qos_level = QoSLevel.BEST_EFFORT

        return qos_params

    def _create_or_update_flow(
        self, flow_id: str, protocol: PathProtocol, context: MessageContext, qos_params: QoSParameters
    ) -> TrafficFlow:
        """Create or update traffic flow"""
        if flow_id in self.active_flows:
            # Update existing flow
            flow = self.active_flows[flow_id]
            flow.qos_params = qos_params
            flow.last_activity = time.time()
        else:
            # Create new flow
            flow = TrafficFlow(
                flow_id=flow_id,
                destination=getattr(context, "destination", "unknown"),
                protocol=protocol,
                qos_params=qos_params,
            )
            self.active_flows[flow_id] = flow

        # Add to appropriate traffic queue
        traffic_class = qos_params.traffic_class
        if flow_id not in self.traffic_queues[traffic_class]:
            self.traffic_queues[traffic_class].append(flow_id)

        return flow

    async def _allocate_bandwidth(self, flow: TrafficFlow) -> bool:
        """Allocate bandwidth for traffic flow"""
        required_bandwidth = flow.qos_params.guaranteed_bandwidth_mbps

        if required_bandwidth <= 0:
            return True  # No bandwidth guarantee required

        # Check if bandwidth can be allocated
        max_allocatable = self.available_bandwidth * self.bandwidth_reservation_ratio

        if self.total_allocated_bandwidth + required_bandwidth <= max_allocatable:
            # Allocate bandwidth
            self.total_allocated_bandwidth += required_bandwidth
            flow.current_bandwidth = required_bandwidth

            logger.debug(f"Allocated {required_bandwidth}Mbps to flow {flow.flow_id}")
            return True
        else:
            # Try to free up bandwidth through adaptation
            freed_bandwidth = await self._attempt_bandwidth_reallocation(required_bandwidth)

            if freed_bandwidth >= required_bandwidth:
                self.total_allocated_bandwidth += required_bandwidth
                flow.current_bandwidth = required_bandwidth
                return True
            else:
                # Partial allocation or denial
                available = max(0, max_allocatable - self.total_allocated_bandwidth)
                flow.current_bandwidth = available
                self.total_allocated_bandwidth += available

                if available < required_bandwidth:
                    flow.violations.append(
                        f"Bandwidth allocation shortfall: got {available}, needed {required_bandwidth}"
                    )
                    self.violation_counts[flow.flow_id] += 1

                return available > 0

    async def _attempt_bandwidth_reallocation(self, required_bandwidth: float) -> float:
        """Attempt to reallocate bandwidth from lower priority flows"""
        freed_bandwidth = 0.0

        # Sort active flows by priority (lower priority first)
        flows_by_priority = sorted(self.active_flows.values(), key=lambda f: f.qos_params.priority)

        for flow in flows_by_priority:
            if freed_bandwidth >= required_bandwidth:
                break

            # Can reduce bandwidth from best-effort and bulk flows
            if flow.qos_params.qos_level in [QoSLevel.BEST_EFFORT]:
                if flow.current_bandwidth > 0.1:  # Keep minimal bandwidth
                    reduction = min(flow.current_bandwidth - 0.1, required_bandwidth - freed_bandwidth)
                    flow.current_bandwidth -= reduction
                    self.total_allocated_bandwidth -= reduction
                    freed_bandwidth += reduction

                    logger.debug(f"Reduced bandwidth for flow {flow.flow_id} by {reduction}Mbps")

        return freed_bandwidth

    async def apply_traffic_shaping(self, flow: TrafficFlow) -> Dict[str, Any]:
        """Apply traffic shaping to flow"""
        shaping_config = {
            "enabled": True,
            "rate_limit_mbps": flow.qos_params.max_bandwidth_mbps,
            "burst_size_kb": flow.qos_params.max_bandwidth_mbps * 100,  # 100KB per Mbps
            "queue_discipline": "fq_codel",  # Fair queuing with CoDel
            "priority_class": flow.qos_params.priority,
        }

        # Protocol-specific shaping
        if flow.protocol == PathProtocol.BITCHAT:
            # BitChat needs different shaping for mesh networks
            shaping_config.update(
                {
                    "rate_limit_mbps": min(0.1, flow.qos_params.max_bandwidth_mbps),
                    "burst_size_kb": 10,  # Small bursts for mesh
                    "queue_discipline": "pfifo",  # Simple FIFO for low bandwidth
                    "mesh_aware": True,
                }
            )

        elif flow.protocol == PathProtocol.SCION:
            # SCION supports advanced traffic engineering
            shaping_config.update(
                {
                    "multipath_aware": True,
                    "path_selection_policy": (
                        "latency_optimized" if flow.qos_params.qos_level == QoSLevel.CRITICAL else "balanced"
                    ),
                    "load_balancing": True,
                }
            )

        return shaping_config

    async def _apply_traffic_shaping(self, flow: TrafficFlow) -> Dict[str, Any]:
        """Apply traffic shaping configuration"""
        return await self.apply_traffic_shaping(flow)

    def _check_qos_compliance(self, flow: TrafficFlow) -> Dict[str, Any]:
        """Check QoS compliance for flow"""
        compliance = {
            "compliant": True,
            "violations": [],
            "metrics": {
                "current_bandwidth_mbps": flow.current_bandwidth,
                "current_latency_ms": flow.current_latency,
                "packet_loss_rate": 0.0,  # Would be measured in real implementation
            },
        }

        # Check bandwidth compliance
        if flow.current_bandwidth < flow.qos_params.guaranteed_bandwidth_mbps:
            compliance["compliant"] = False
            compliance["violations"].append("bandwidth_guarantee_violation")

        # Check latency compliance (if measurable)
        if flow.current_latency > flow.qos_params.max_latency_ms:
            compliance["compliant"] = False
            compliance["violations"].append("latency_sla_violation")

        return compliance

    async def adapt_bandwidth_usage(self, available_bandwidth: float, required_bandwidth: float) -> Dict[str, Any]:
        """Adapt bandwidth usage based on availability"""
        if not self.adaptation_enabled:
            return {"adapted": False, "reason": "adaptation_disabled"}

        current_time = time.time()
        if current_time - self.last_adaptation < self.adaptation_interval:
            return {"adapted": False, "reason": "too_soon"}

        self.last_adaptation = current_time
        self.available_bandwidth = available_bandwidth

        utilization_ratio = self.total_allocated_bandwidth / available_bandwidth if available_bandwidth > 0 else 1.0

        adaptation_result = {
            "adapted": True,
            "previous_utilization": utilization_ratio,
            "actions_taken": [],
            "flows_affected": [],
        }

        if utilization_ratio > self.congestion_threshold:
            # Network is congested, need to adapt
            logger.info(f"Network congestion detected (utilization: {utilization_ratio:.2f}), adapting...")

            # Reduce bandwidth for lower priority flows
            reductions = await self._reduce_low_priority_bandwidth()
            adaptation_result["actions_taken"].append("reduced_low_priority_bandwidth")
            adaptation_result["flows_affected"].extend(reductions)

            # Enable more aggressive traffic shaping
            await self._enable_aggressive_shaping()
            adaptation_result["actions_taken"].append("enabled_aggressive_shaping")

        elif utilization_ratio < 0.5:
            # Network is under-utilized, can relax constraints
            logger.debug(f"Network under-utilized (utilization: {utilization_ratio:.2f}), relaxing constraints...")

            # Increase bandwidth for flows that were previously constrained
            increases = await self._increase_constrained_bandwidth()
            adaptation_result["actions_taken"].append("increased_constrained_bandwidth")
            adaptation_result["flows_affected"].extend(increases)

        # Record adaptation
        self.adaptation_history.append(
            {
                "timestamp": current_time,
                "utilization_before": utilization_ratio,
                "available_bandwidth": available_bandwidth,
                "required_bandwidth": required_bandwidth,
                "actions_taken": adaptation_result["actions_taken"],
            }
        )

        # Emit adaptation event
        self._emit_qos_event("bandwidth_adapted", adaptation_result)

        return adaptation_result

    async def _reduce_low_priority_bandwidth(self) -> List[str]:
        """Reduce bandwidth for lower priority flows"""
        affected_flows = []

        # Target flows with priority <= 3 and best effort QoS
        for flow in self.active_flows.values():
            if (
                flow.qos_params.priority <= 3
                and flow.qos_params.qos_level == QoSLevel.BEST_EFFORT
                and flow.current_bandwidth > 0.1
            ):

                reduction = min(flow.current_bandwidth * 0.3, flow.current_bandwidth - 0.1)
                flow.current_bandwidth -= reduction
                self.total_allocated_bandwidth -= reduction
                affected_flows.append(flow.flow_id)

                logger.debug(f"Reduced bandwidth for flow {flow.flow_id} by {reduction}Mbps")

        return affected_flows

    async def _enable_aggressive_shaping(self) -> None:
        """Enable more aggressive traffic shaping during congestion"""
        # This would configure the underlying traffic shaping mechanisms
        # For now, just log the action
        logger.debug("Enabled aggressive traffic shaping")

    async def _increase_constrained_bandwidth(self) -> List[str]:
        """Increase bandwidth for previously constrained flows"""
        affected_flows = []
        available_increase = self.available_bandwidth * 0.8 - self.total_allocated_bandwidth

        if available_increase <= 0:
            return affected_flows

        # Prioritize flows that have violations or are under their desired bandwidth
        for flow in sorted(self.active_flows.values(), key=lambda f: -f.qos_params.priority):
            if available_increase <= 0:
                break

            desired_bandwidth = flow.qos_params.guaranteed_bandwidth_mbps
            if flow.current_bandwidth < desired_bandwidth:
                increase = min(desired_bandwidth - flow.current_bandwidth, available_increase)
                flow.current_bandwidth += increase
                self.total_allocated_bandwidth += increase
                available_increase -= increase
                affected_flows.append(flow.flow_id)

                logger.debug(f"Increased bandwidth for flow {flow.flow_id} by {increase}Mbps")

        return affected_flows

    def prioritize_traffic(self, messages: List[MessageContext]) -> List[MessageContext]:
        """Prioritize message traffic based on QoS policies"""

        # Sort messages by priority and QoS requirements
        def priority_key(msg: MessageContext) -> Tuple[int, bool, int]:
            # Sort by: (priority desc, realtime first, size asc for same priority)
            return (-msg.priority, not msg.requires_realtime, msg.size_bytes)

        prioritized = sorted(messages, key=priority_key)

        # Group by traffic class for queue assignment
        traffic_groups = {tc: [] for tc in TrafficClass}

        for msg in prioritized:
            if msg.requires_realtime:
                traffic_groups[TrafficClass.REALTIME].append(msg)
            elif msg.priority >= 8:
                traffic_groups[TrafficClass.INTERACTIVE].append(msg)
            elif msg.priority >= 5:
                traffic_groups[TrafficClass.STANDARD].append(msg)
            else:
                traffic_groups[TrafficClass.BULK].append(msg)

        # Rebuild list based on queue weights and priorities
        final_order = []

        # High priority queues first
        for tc in [TrafficClass.CONTROL, TrafficClass.REALTIME, TrafficClass.INTERACTIVE]:
            final_order.extend(traffic_groups[tc])

        # Then standard and bulk
        final_order.extend(traffic_groups[TrafficClass.STANDARD])
        final_order.extend(traffic_groups[TrafficClass.BULK])

        logger.debug(f"Prioritized {len(messages)} messages into {len(final_order)} ordered list")

        return final_order

    def _emit_qos_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit QoS management event"""
        event = RoutingEvent(
            event_type=event_type, timestamp=time.time(), source_service="QoSManagerService", data=data
        )
        self.event_bus.publish(event)

    def get_qos_statistics(self) -> Dict[str, Any]:
        """Get QoS management statistics"""
        active_flow_count = len(self.active_flows)
        total_violations = sum(self.violation_counts.values())

        # Calculate utilization by traffic class
        utilization_by_class = {}
        for traffic_class, flow_ids in self.traffic_queues.items():
            class_bandwidth = sum(
                self.active_flows[fid].current_bandwidth for fid in flow_ids if fid in self.active_flows
            )
            utilization_by_class[traffic_class.value] = class_bandwidth

        # Recent adaptation history
        recent_adaptations = [
            adapt for adapt in self.adaptation_history if time.time() - adapt["timestamp"] < 300  # Last 5 minutes
        ]

        return {
            "active_flows": active_flow_count,
            "total_allocated_bandwidth_mbps": self.total_allocated_bandwidth,
            "available_bandwidth_mbps": self.available_bandwidth,
            "bandwidth_utilization": (
                self.total_allocated_bandwidth / self.available_bandwidth if self.available_bandwidth > 0 else 0
            ),
            "total_qos_violations": total_violations,
            "utilization_by_traffic_class": utilization_by_class,
            "recent_adaptations": len(recent_adaptations),
            "adaptation_enabled": self.adaptation_enabled,
            "flows_by_qos_level": self._count_flows_by_qos_level(),
        }

    def _count_flows_by_qos_level(self) -> Dict[str, int]:
        """Count active flows by QoS level"""
        counts = defaultdict(int)
        for flow in self.active_flows.values():
            counts[flow.qos_params.qos_level.value] += 1
        return dict(counts)

    def cleanup_expired_flows(self) -> int:
        """Clean up expired flows"""
        current_time = time.time()
        expired_flows = []

        for flow_id, flow in self.active_flows.items():
            # Flows inactive for 5 minutes are considered expired
            if current_time - flow.last_activity > 300:
                expired_flows.append(flow_id)

        # Clean up expired flows
        for flow_id in expired_flows:
            flow = self.active_flows[flow_id]

            # Release allocated bandwidth
            self.total_allocated_bandwidth -= flow.current_bandwidth

            # Remove from traffic queues
            for traffic_class, flow_ids in self.traffic_queues.items():
                if flow_id in flow_ids:
                    flow_ids.remove(flow_id)

            # Move to history and remove from active
            self.flow_history.append(flow)
            del self.active_flows[flow_id]

        if expired_flows:
            logger.info(f"Cleaned up {len(expired_flows)} expired QoS flows")

        return len(expired_flows)
