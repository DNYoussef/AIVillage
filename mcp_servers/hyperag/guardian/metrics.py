"""Guardian Gate Prometheus Metrics

Provides Prometheus metrics for monitoring Guardian Gate decisions
and system behavior in production environments.
"""

import logging
from typing import Any

try:
    from prometheus_client import Counter, Gauge, Histogram, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    # Fallback implementations for environments without prometheus_client
    PROMETHEUS_AVAILABLE = False

    class MockMetric:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get("name", "mock_metric")

        def inc(self, amount=1, **labels):
            pass

        def observe(self, amount, **labels):
            pass

        def set(self, value, **labels):
            pass

        def info(self, info_dict):
            pass

    Counter = Histogram = Gauge = Info = MockMetric

logger = logging.getLogger(__name__)


class GuardianMetrics:
    """Prometheus metrics collector for Guardian Gate operations"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and PROMETHEUS_AVAILABLE

        if not self.enabled:
            logger.warning("Prometheus metrics disabled (prometheus_client not available)")
            return

        # Decision counters
        self.guardian_blocks_total = Counter(
            "hyperag_guardian_blocks_total",
            "Total number of Guardian Gate blocks",
            ["decision_type", "domain", "component"]
        )

        self.guardian_quarantine_total = Counter(
            "hyperag_guardian_quarantine_total",
            "Total number of Guardian Gate quarantines",
            ["domain", "component", "reason"]
        )

        self.guardian_autoapply_total = Counter(
            "hyperag_guardian_autoapply_total",
            "Total number of Guardian Gate auto-approvals",
            ["domain", "component"]
        )

        # Performance metrics
        self.guardian_decision_duration = Histogram(
            "hyperag_guardian_decision_duration_seconds",
            "Time taken for Guardian Gate decisions",
            ["decision_type", "domain"],
            buckets=[0.001, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.500, 1.0]
        )

        self.guardian_confidence_score = Histogram(
            "hyperag_guardian_confidence_score",
            "Confidence scores from Guardian evaluations",
            ["component", "domain"],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        # System health gauges
        self.guardian_active_quarantines = Gauge(
            "hyperag_guardian_active_quarantines",
            "Number of items currently in quarantine",
            ["domain"]
        )

        self.guardian_policy_version = Info(
            "hyperag_guardian_policy_version",
            "Current Guardian policy version and configuration"
        )

        # Error tracking
        self.guardian_errors_total = Counter(
            "hyperag_guardian_errors_total",
            "Total number of Guardian Gate errors",
            ["error_type", "component"]
        )

        # Component-specific metrics
        self.query_guardian_validations = Counter(
            "hyperag_query_guardian_validations_total",
            "Query pipeline Guardian validations",
            ["domain", "decision", "confidence_tier"]
        )

        self.repair_guardian_validations = Counter(
            "hyperag_repair_guardian_validations_total",
            "Repair pipeline Guardian validations",
            ["domain", "decision", "operation_count"]
        )

        self.consolidation_guardian_validations = Counter(
            "hyperag_consolidation_guardian_validations_total",
            "Consolidation pipeline Guardian validations",
            ["domain", "decision", "item_type"]
        )

        self.adapter_load_validations = Counter(
            "hyperag_adapter_load_validations_total",
            "LoRA adapter load validations",
            ["domain", "verification_result"]
        )

        logger.info("Guardian metrics initialized")

    def record_decision(self,
                       decision: str,
                       domain: str = "general",
                       component: str = "unknown",
                       duration_seconds: float = 0.0,
                       confidence: float = 0.0):
        """Record a Guardian Gate decision

        Args:
            decision: APPLY, QUARANTINE, or REJECT
            domain: Domain of the decision
            component: Component that triggered validation
            duration_seconds: Time taken for decision
            confidence: Confidence score
        """
        if not self.enabled:
            return

        try:
            # Record decision counter
            if decision == "APPLY":
                self.guardian_autoapply_total.inc(domain=domain, component=component)
            elif decision == "QUARANTINE":
                self.guardian_quarantine_total.inc(
                    domain=domain,
                    component=component,
                    reason="low_confidence"
                )
            elif decision == "REJECT":
                self.guardian_blocks_total.inc(
                    decision_type="reject",
                    domain=domain,
                    component=component
                )

            # Record timing
            if duration_seconds > 0:
                self.guardian_decision_duration.observe(
                    duration_seconds,
                    decision_type=decision.lower(),
                    domain=domain
                )

            # Record confidence
            if confidence > 0:
                self.guardian_confidence_score.observe(
                    confidence,
                    component=component,
                    domain=domain
                )

        except Exception as e:
            logger.error(f"Failed to record Guardian decision metrics: {e}")

    def record_query_validation(self,
                               domain: str,
                               decision: str,
                               confidence: float):
        """Record query pipeline validation"""
        if not self.enabled:
            return

        try:
            confidence_tier = self._get_confidence_tier(confidence)
            self.query_guardian_validations.inc(
                domain=domain,
                decision=decision.lower(),
                confidence_tier=confidence_tier
            )
        except Exception as e:
            logger.error(f"Failed to record query validation metrics: {e}")

    def record_repair_validation(self,
                                domain: str,
                                decision: str,
                                operation_count: int):
        """Record repair pipeline validation"""
        if not self.enabled:
            return

        try:
            op_count_bucket = self._get_operation_count_bucket(operation_count)
            self.repair_guardian_validations.inc(
                domain=domain,
                decision=decision.lower(),
                operation_count=op_count_bucket
            )
        except Exception as e:
            logger.error(f"Failed to record repair validation metrics: {e}")

    def record_consolidation_validation(self,
                                      domain: str,
                                      decision: str,
                                      item_type: str):
        """Record consolidation pipeline validation"""
        if not self.enabled:
            return

        try:
            self.consolidation_guardian_validations.inc(
                domain=domain,
                decision=decision.lower(),
                item_type=item_type
            )
        except Exception as e:
            logger.error(f"Failed to record consolidation validation metrics: {e}")

    def record_adapter_validation(self,
                                 domain: str,
                                 verification_result: str):
        """Record adapter load validation"""
        if not self.enabled:
            return

        try:
            self.adapter_load_validations.inc(
                domain=domain,
                verification_result=verification_result
            )
        except Exception as e:
            logger.error(f"Failed to record adapter validation metrics: {e}")

    def record_validation(self, domain: str, decision: str, validation_type: str):
        """Generic validation recording for different types"""
        if not self.enabled:
            return

        try:
            # Update the generic counters based on decision
            if decision == "REJECT":
                self.guardian_blocks_total.inc(
                    decision_type="reject",
                    domain=domain,
                    component=validation_type
                )
            elif decision == "QUARANTINE":
                self.guardian_quarantine_total.inc(
                    domain=domain,
                    component=validation_type,
                    reason="validation_failed"
                )
            elif decision == "APPLY":
                self.guardian_autoapply_total.inc(
                    domain=domain,
                    component=validation_type
                )
        except Exception as e:
            logger.error(f"Failed to record validation metrics: {e}")

    def record_error(self,
                    error_type: str,
                    component: str):
        """Record Guardian error"""
        if not self.enabled:
            return

        try:
            self.guardian_errors_total.inc(
                error_type=error_type,
                component=component
            )
        except Exception as e:
            logger.error(f"Failed to record Guardian error metrics: {e}")

    def update_quarantine_count(self, domain: str, count: int):
        """Update active quarantine count"""
        if not self.enabled:
            return

        try:
            self.guardian_active_quarantines.set(count, domain=domain)
        except Exception as e:
            logger.error(f"Failed to update quarantine count metrics: {e}")

    def set_policy_info(self, policy_info: dict[str, Any]):
        """Set Guardian policy information"""
        if not self.enabled:
            return

        try:
            # Convert values to strings for Info metric
            string_info = {
                key: str(value) for key, value in policy_info.items()
                if isinstance(value, (str, int, float, bool))
            }
            self.guardian_policy_version.info(string_info)
        except Exception as e:
            logger.error(f"Failed to set policy info metrics: {e}")

    def _get_confidence_tier(self, confidence: float) -> str:
        """Convert confidence to tier label"""
        if confidence >= 0.8:
            return "high"
        if confidence >= 0.6:
            return "medium"
        if confidence >= 0.4:
            return "low"
        return "very_low"

    def _get_operation_count_bucket(self, count: int) -> str:
        """Convert operation count to bucket label"""
        if count <= 1:
            return "single"
        if count <= 3:
            return "few"
        if count <= 10:
            return "many"
        return "bulk"

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of current metrics (for debugging)"""
        if not self.enabled:
            return {"status": "disabled"}

        # This would need to be implemented with actual metric collection
        # For now, return basic status
        return {
            "status": "enabled",
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "metrics_initialized": True
        }


# Global metrics instance
_metrics_instance: GuardianMetrics | None = None

def get_guardian_metrics() -> GuardianMetrics:
    """Get global Guardian metrics instance"""
    global _metrics_instance

    if _metrics_instance is None:
        _metrics_instance = GuardianMetrics()

    return _metrics_instance

def init_guardian_metrics(enabled: bool = True) -> GuardianMetrics:
    """Initialize Guardian metrics with specific configuration"""
    global _metrics_instance
    _metrics_instance = GuardianMetrics(enabled=enabled)
    return _metrics_instance
