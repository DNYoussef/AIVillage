"""
SLA Classes and Management for Fog Computing

Defines Service Level Agreement classes and enforcement for fog jobs.
"""

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SLAClass(str, Enum):
    """Service Level Agreement classes"""

    SPOT = "spot"  # Best effort, preemptible
    STANDARD = "standard"  # Normal priority, guaranteed resources
    PREMIUM = "premium"  # High priority, dedicated resources
    REALTIME = "realtime"  # Ultra-low latency, highest priority


@dataclass
class SLARequirements:
    """Requirements for each SLA class"""

    max_latency_ms: int | None = None
    min_availability: float | None = None
    guaranteed_bandwidth_mbps: int | None = None
    max_preemptions: int | None = None
    priority: int = 0


@dataclass
class SLAManager:
    """Manages SLA enforcement and monitoring"""

    sla_configs: dict[SLAClass, SLARequirements] = None

    def __post_init__(self):
        if self.sla_configs is None:
            self.sla_configs = {
                SLAClass.SPOT: SLARequirements(max_preemptions=None, priority=0),  # Unlimited preemptions
                SLAClass.STANDARD: SLARequirements(
                    max_latency_ms=1000, min_availability=0.95, max_preemptions=3, priority=1
                ),
                SLAClass.PREMIUM: SLARequirements(
                    max_latency_ms=100,
                    min_availability=0.99,
                    guaranteed_bandwidth_mbps=100,
                    max_preemptions=0,
                    priority=2,
                ),
                SLAClass.REALTIME: SLARequirements(
                    max_latency_ms=10,
                    min_availability=0.999,
                    guaranteed_bandwidth_mbps=1000,
                    max_preemptions=0,
                    priority=3,
                ),
            }

    def get_requirements(self, sla_class: SLAClass) -> SLARequirements:
        """Get requirements for an SLA class"""
        return self.sla_configs.get(sla_class, self.sla_configs[SLAClass.STANDARD])

    def validate_sla_compliance(self, sla_class: SLAClass, metrics: dict[str, Any]) -> bool:
        """Check if current metrics meet SLA requirements"""
        requirements = self.get_requirements(sla_class)

        # Check latency
        if requirements.max_latency_ms and metrics.get("latency_ms"):
            if metrics["latency_ms"] > requirements.max_latency_ms:
                logger.warning(f"SLA violation: latency {metrics['latency_ms']}ms > {requirements.max_latency_ms}ms")
                return False

        # Check availability
        if requirements.min_availability and metrics.get("availability"):
            if metrics["availability"] < requirements.min_availability:
                logger.warning(
                    f"SLA violation: availability {metrics['availability']} < {requirements.min_availability}"
                )
                return False

        return True

    def get_priority(self, sla_class: SLAClass) -> int:
        """Get scheduling priority for an SLA class"""
        return self.get_requirements(sla_class).priority
