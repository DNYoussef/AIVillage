"""
Real-Time Constitutional Compliance Monitoring System
Advanced monitoring and alerting for constitutional violations and system health
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import threading
import statistics

# Import transparency system components
from .merkle_audit import ConstitutionalMerkleAudit
from .constitutional_logging import ConstitutionalDecisionLogger
from .privacy_preserving_audit import PrivacyPreservingAuditSystem
from .governance_audit import GovernanceAuditTrail
from .cryptographic_verification import ConstitutionalCryptographicVerifier


class ComplianceStatus(Enum):
    """Constitutional compliance status levels"""

    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MonitoringMetric(Enum):
    """Types of monitoring metrics"""

    COMPLIANCE_RATE = "compliance_rate"
    VIOLATION_FREQUENCY = "violation_frequency"
    DEMOCRATIC_PARTICIPATION = "democratic_participation"
    SYSTEM_INTEGRITY = "system_integrity"
    PRIVACY_PRESERVATION = "privacy_preservation"
    AUDIT_TRAIL_COMPLETENESS = "audit_trail_completeness"
    CRYPTOGRAPHIC_VERIFICATION = "cryptographic_verification"
    RESPONSE_TIME = "response_time"


@dataclass
class ComplianceThreshold:
    """Threshold configuration for compliance monitoring"""

    metric: MonitoringMetric
    warning_threshold: float
    critical_threshold: float
    measurement_window_minutes: int
    evaluation_frequency_seconds: int


@dataclass
class ComplianceAlert:
    """Constitutional compliance alert"""

    alert_id: str
    timestamp: float
    severity: AlertSeverity
    metric: MonitoringMetric
    current_value: float
    threshold_value: float
    message: str
    affected_components: List[str]
    recommended_actions: List[str]
    alert_context: Dict[str, Any]


@dataclass
class SystemHealthSnapshot:
    """Snapshot of constitutional system health"""

    timestamp: float
    overall_compliance_rate: float
    active_violations: int
    system_integrity_score: float
    democratic_participation_rate: float
    privacy_preservation_rate: float
    audit_completeness: float
    cryptographic_health: float
    component_status: Dict[str, str]
    performance_metrics: Dict[str, float]


class RealTimeComplianceMonitor:
    """
    Advanced real-time constitutional compliance monitoring system
    Provides continuous monitoring, alerting, and health assessment
    """

    def __init__(
        self,
        merkle_audit: ConstitutionalMerkleAudit,
        decision_logger: ConstitutionalDecisionLogger,
        privacy_system: PrivacyPreservingAuditSystem,
        governance_audit: GovernanceAuditTrail,
        crypto_verifier: ConstitutionalCryptographicVerifier,
        monitoring_interval: int = 30,
    ):

        # Core transparency components
        self.merkle_audit = merkle_audit
        self.decision_logger = decision_logger
        self.privacy_system = privacy_system
        self.governance_audit = governance_audit
        self.crypto_verifier = crypto_verifier

        # Monitoring configuration
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitoring_thread = None

        # Compliance thresholds
        self.compliance_thresholds: Dict[MonitoringMetric, ComplianceThreshold] = {}
        self._initialize_default_thresholds()

        # Alert management
        self.active_alerts: Dict[str, ComplianceAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable[[ComplianceAlert], None]] = []

        # Health monitoring
        self.health_snapshots: deque = deque(maxlen=288)  # 24 hours at 5-minute intervals
        self.current_health_status: Optional[SystemHealthSnapshot] = None

        # Performance tracking
        self.performance_metrics = {
            "monitoring_cycles_completed": 0,
            "alerts_generated": 0,
            "violations_detected": 0,
            "false_positives": 0,
            "system_uptime": time.time(),
            "average_response_time": 0.0,
        }

        # Metric calculations
        self.metric_calculators: Dict[MonitoringMetric, Callable[[], float]] = {
            MonitoringMetric.COMPLIANCE_RATE: self._calculate_compliance_rate,
            MonitoringMetric.VIOLATION_FREQUENCY: self._calculate_violation_frequency,
            MonitoringMetric.DEMOCRATIC_PARTICIPATION: self._calculate_democratic_participation,
            MonitoringMetric.SYSTEM_INTEGRITY: self._calculate_system_integrity,
            MonitoringMetric.PRIVACY_PRESERVATION: self._calculate_privacy_preservation,
            MonitoringMetric.AUDIT_TRAIL_COMPLETENESS: self._calculate_audit_completeness,
            MonitoringMetric.CRYPTOGRAPHIC_VERIFICATION: self._calculate_crypto_health,
            MonitoringMetric.RESPONSE_TIME: self._calculate_response_time,
        }

        self.logger = logging.getLogger(__name__)

        # Lock for thread safety
        self._lock = threading.Lock()

    def _initialize_default_thresholds(self):
        """Initialize default compliance thresholds"""
        default_thresholds = [
            ComplianceThreshold(
                MonitoringMetric.COMPLIANCE_RATE,
                warning_threshold=90.0,
                critical_threshold=85.0,
                measurement_window_minutes=60,
                evaluation_frequency_seconds=300,  # 5 minutes
            ),
            ComplianceThreshold(
                MonitoringMetric.VIOLATION_FREQUENCY,
                warning_threshold=5.0,  # violations per hour
                critical_threshold=10.0,
                measurement_window_minutes=60,
                evaluation_frequency_seconds=300,
            ),
            ComplianceThreshold(
                MonitoringMetric.DEMOCRATIC_PARTICIPATION,
                warning_threshold=70.0,
                critical_threshold=50.0,
                measurement_window_minutes=1440,  # 24 hours
                evaluation_frequency_seconds=3600,  # 1 hour
            ),
            ComplianceThreshold(
                MonitoringMetric.SYSTEM_INTEGRITY,
                warning_threshold=95.0,
                critical_threshold=90.0,
                measurement_window_minutes=30,
                evaluation_frequency_seconds=300,
            ),
            ComplianceThreshold(
                MonitoringMetric.PRIVACY_PRESERVATION,
                warning_threshold=85.0,
                critical_threshold=75.0,
                measurement_window_minutes=60,
                evaluation_frequency_seconds=600,
            ),
            ComplianceThreshold(
                MonitoringMetric.AUDIT_TRAIL_COMPLETENESS,
                warning_threshold=99.0,
                critical_threshold=95.0,
                measurement_window_minutes=30,
                evaluation_frequency_seconds=300,
            ),
            ComplianceThreshold(
                MonitoringMetric.CRYPTOGRAPHIC_VERIFICATION,
                warning_threshold=98.0,
                critical_threshold=95.0,
                measurement_window_minutes=60,
                evaluation_frequency_seconds=300,
            ),
            ComplianceThreshold(
                MonitoringMetric.RESPONSE_TIME,
                warning_threshold=1000.0,  # milliseconds
                critical_threshold=2000.0,
                measurement_window_minutes=30,
                evaluation_frequency_seconds=300,
            ),
        ]

        for threshold in default_thresholds:
            self.compliance_thresholds[threshold.metric] = threshold

    async def start_monitoring(self):
        """Start real-time constitutional compliance monitoring"""
        if self.is_monitoring:
            self.logger.warning("Compliance monitoring already running")
            return

        self.logger.info("Starting Real-Time Constitutional Compliance Monitoring")

        self.is_monitoring = True
        self.performance_metrics["system_uptime"] = time.time()

        # Start monitoring in background
        self.monitoring_thread = threading.Thread(target=self._run_monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        self.logger.info("Real-time compliance monitoring started")

    def stop_monitoring(self):
        """Stop real-time compliance monitoring"""
        if not self.is_monitoring:
            return

        self.logger.info("Stopping Real-Time Constitutional Compliance Monitoring")

        self.is_monitoring = False

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)

        self.logger.info("Real-time compliance monitoring stopped")

    def _run_monitoring_loop(self):
        """Main monitoring loop (runs in separate thread)"""
        while self.is_monitoring:
            try:
                cycle_start = time.time()

                # Perform monitoring cycle
                asyncio.run(self._perform_monitoring_cycle())

                # Update performance metrics
                cycle_duration = time.time() - cycle_start
                self.performance_metrics["monitoring_cycles_completed"] += 1

                # Calculate average response time
                if self.performance_metrics["average_response_time"] == 0:
                    self.performance_metrics["average_response_time"] = cycle_duration * 1000  # ms
                else:
                    # Exponential moving average
                    alpha = 0.1
                    self.performance_metrics["average_response_time"] = (
                        alpha * (cycle_duration * 1000)
                        + (1 - alpha) * self.performance_metrics["average_response_time"]
                    )

                # Wait for next monitoring cycle
                sleep_time = max(0, self.monitoring_interval - cycle_duration)
                time.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error

    async def _perform_monitoring_cycle(self):
        """Perform single monitoring cycle"""
        try:
            # Calculate current system health snapshot
            health_snapshot = await self._create_health_snapshot()

            with self._lock:
                self.current_health_status = health_snapshot
                self.health_snapshots.append(health_snapshot)

            # Evaluate each compliance threshold
            for metric, threshold in self.compliance_thresholds.items():
                await self._evaluate_compliance_threshold(metric, threshold)

            # Check for system-wide issues
            await self._check_system_wide_issues(health_snapshot)

            # Clean up resolved alerts
            self._cleanup_resolved_alerts()

        except Exception as e:
            self.logger.error(f"Error in monitoring cycle: {e}")

    async def _create_health_snapshot(self) -> SystemHealthSnapshot:
        """Create comprehensive system health snapshot"""
        timestamp = time.time()

        # Calculate key metrics
        compliance_rate = await asyncio.get_event_loop().run_in_executor(None, self._calculate_compliance_rate)

        violation_count = await asyncio.get_event_loop().run_in_executor(None, self._calculate_current_violations)

        system_integrity = await asyncio.get_event_loop().run_in_executor(None, self._calculate_system_integrity)

        democratic_participation = await asyncio.get_event_loop().run_in_executor(
            None, self._calculate_democratic_participation
        )

        privacy_preservation = await asyncio.get_event_loop().run_in_executor(
            None, self._calculate_privacy_preservation
        )

        audit_completeness = await asyncio.get_event_loop().run_in_executor(None, self._calculate_audit_completeness)

        crypto_health = await asyncio.get_event_loop().run_in_executor(None, self._calculate_crypto_health)

        # Component status
        component_status = {
            "merkle_audit": "operational" if len(self.merkle_audit.audit_entries) > 0 else "initializing",
            "decision_logger": "operational" if len(self.decision_logger.decision_logs) > 0 else "initializing",
            "privacy_system": "operational",  # Always operational
            "governance_audit": "operational",  # Always operational
            "crypto_verifier": "operational" if self.crypto_verifier.system_integrity_hash else "initializing",
        }

        # Performance metrics
        performance_metrics = {
            "average_response_time": self.performance_metrics["average_response_time"],
            "monitoring_cycles_per_hour": self.performance_metrics["monitoring_cycles_completed"]
            / max(1, (time.time() - self.performance_metrics["system_uptime"]) / 3600),
            "alerts_per_hour": len(self.alert_history)
            / max(1, (time.time() - self.performance_metrics["system_uptime"]) / 3600),
            "uptime_hours": (time.time() - self.performance_metrics["system_uptime"]) / 3600,
        }

        return SystemHealthSnapshot(
            timestamp=timestamp,
            overall_compliance_rate=compliance_rate,
            active_violations=violation_count,
            system_integrity_score=system_integrity,
            democratic_participation_rate=democratic_participation,
            privacy_preservation_rate=privacy_preservation,
            audit_completeness=audit_completeness,
            cryptographic_health=crypto_health,
            component_status=component_status,
            performance_metrics=performance_metrics,
        )

    async def _evaluate_compliance_threshold(self, metric: MonitoringMetric, threshold: ComplianceThreshold):
        """Evaluate specific compliance threshold"""
        try:
            # Calculate current metric value
            current_value = self.metric_calculators[metric]()

            # Determine compliance status
            if metric in [MonitoringMetric.RESPONSE_TIME, MonitoringMetric.VIOLATION_FREQUENCY]:
                # Lower is better for these metrics
                if current_value >= threshold.critical_threshold:
                    status = ComplianceStatus.CRITICAL
                elif current_value >= threshold.warning_threshold:
                    status = ComplianceStatus.WARNING
                else:
                    status = ComplianceStatus.COMPLIANT
            else:
                # Higher is better for other metrics
                if current_value <= threshold.critical_threshold:
                    status = ComplianceStatus.CRITICAL
                elif current_value <= threshold.warning_threshold:
                    status = ComplianceStatus.WARNING
                else:
                    status = ComplianceStatus.COMPLIANT

            # Generate alert if threshold exceeded
            if status != ComplianceStatus.COMPLIANT:
                await self._generate_compliance_alert(metric, status, current_value, threshold)

        except Exception as e:
            self.logger.error(f"Error evaluating compliance threshold for {metric.value}: {e}")

    async def _generate_compliance_alert(
        self, metric: MonitoringMetric, status: ComplianceStatus, current_value: float, threshold: ComplianceThreshold
    ):
        """Generate compliance alert"""
        alert_id = f"alert_{metric.value}_{int(time.time() * 1000000)}"

        # Map compliance status to alert severity
        severity_mapping = {
            ComplianceStatus.WARNING: AlertSeverity.WARNING,
            ComplianceStatus.VIOLATION: AlertSeverity.CRITICAL,
            ComplianceStatus.CRITICAL: AlertSeverity.CRITICAL,
        }

        severity = severity_mapping.get(status, AlertSeverity.WARNING)

        # Generate contextual message
        message = self._generate_alert_message(metric, status, current_value, threshold)

        # Identify affected components
        affected_components = self._identify_affected_components(metric)

        # Generate recommended actions
        recommended_actions = self._generate_recommended_actions(metric, status)

        # Create alert context
        alert_context = {
            "measurement_window_minutes": threshold.measurement_window_minutes,
            "threshold_type": "critical" if current_value <= threshold.critical_threshold else "warning",
            "historical_trend": self._calculate_metric_trend(metric),
            "system_load": self._calculate_system_load(),
            "related_alerts": self._find_related_alerts(metric),
        }

        alert = ComplianceAlert(
            alert_id=alert_id,
            timestamp=time.time(),
            severity=severity,
            metric=metric,
            current_value=current_value,
            threshold_value=(
                threshold.critical_threshold if status == ComplianceStatus.CRITICAL else threshold.warning_threshold
            ),
            message=message,
            affected_components=affected_components,
            recommended_actions=recommended_actions,
            alert_context=alert_context,
        )

        # Store alert
        with self._lock:
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            self.performance_metrics["alerts_generated"] += 1

            if status == ComplianceStatus.CRITICAL:
                self.performance_metrics["violations_detected"] += 1

        # Trigger alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")

        self.logger.warning(f"Generated compliance alert {alert_id}: {message}")

    def _generate_alert_message(
        self, metric: MonitoringMetric, status: ComplianceStatus, current_value: float, threshold: ComplianceThreshold
    ) -> str:
        """Generate contextual alert message"""
        metric_names = {
            MonitoringMetric.COMPLIANCE_RATE: "Constitutional Compliance Rate",
            MonitoringMetric.VIOLATION_FREQUENCY: "Constitutional Violation Frequency",
            MonitoringMetric.DEMOCRATIC_PARTICIPATION: "Democratic Participation Rate",
            MonitoringMetric.SYSTEM_INTEGRITY: "System Integrity Score",
            MonitoringMetric.PRIVACY_PRESERVATION: "Privacy Preservation Rate",
            MonitoringMetric.AUDIT_TRAIL_COMPLETENESS: "Audit Trail Completeness",
            MonitoringMetric.CRYPTOGRAPHIC_VERIFICATION: "Cryptographic Verification Health",
            MonitoringMetric.RESPONSE_TIME: "System Response Time",
        }

        metric_name = metric_names.get(metric, metric.value)

        if status == ComplianceStatus.CRITICAL:
            return f"CRITICAL: {metric_name} at {current_value:.2f}, below critical threshold of {threshold.critical_threshold}"
        else:
            return f"WARNING: {metric_name} at {current_value:.2f}, below warning threshold of {threshold.warning_threshold}"

    def _identify_affected_components(self, metric: MonitoringMetric) -> List[str]:
        """Identify system components affected by metric threshold breach"""
        component_mappings = {
            MonitoringMetric.COMPLIANCE_RATE: ["merkle_audit", "decision_logger"],
            MonitoringMetric.VIOLATION_FREQUENCY: ["merkle_audit", "decision_logger"],
            MonitoringMetric.DEMOCRATIC_PARTICIPATION: ["governance_audit"],
            MonitoringMetric.SYSTEM_INTEGRITY: ["merkle_audit", "crypto_verifier"],
            MonitoringMetric.PRIVACY_PRESERVATION: ["privacy_system"],
            MonitoringMetric.AUDIT_TRAIL_COMPLETENESS: ["merkle_audit", "decision_logger"],
            MonitoringMetric.CRYPTOGRAPHIC_VERIFICATION: ["crypto_verifier"],
            MonitoringMetric.RESPONSE_TIME: ["all_components"],
        }

        return component_mappings.get(metric, ["unknown"])

    def _generate_recommended_actions(self, metric: MonitoringMetric, status: ComplianceStatus) -> List[str]:
        """Generate recommended actions based on metric and status"""
        actions = {
            MonitoringMetric.COMPLIANCE_RATE: [
                "Review recent constitutional decisions for pattern analysis",
                "Check constitutional harm classification accuracy",
                "Verify tier-based decision appropriateness",
                "Escalate to constitutional review committee if critical",
            ],
            MonitoringMetric.VIOLATION_FREQUENCY: [
                "Investigate root causes of constitutional violations",
                "Review moderation pipeline effectiveness",
                "Check for systematic bias in decision making",
                "Implement additional safeguards if critical",
            ],
            MonitoringMetric.DEMOCRATIC_PARTICIPATION: [
                "Encourage community engagement initiatives",
                "Review governance proposal accessibility",
                "Check voting system functionality",
                "Consider adjusting participation incentives",
            ],
            MonitoringMetric.SYSTEM_INTEGRITY: [
                "Verify Merkle tree integrity",
                "Check cryptographic verification systems",
                "Review system component health",
                "Initiate integrity restoration procedures if critical",
            ],
            MonitoringMetric.PRIVACY_PRESERVATION: [
                "Review privacy-preserving mechanisms",
                "Check zero-knowledge proof generation",
                "Verify tier-based privacy enforcement",
                "Audit privacy compliance procedures",
            ],
            MonitoringMetric.AUDIT_TRAIL_COMPLETENESS: [
                "Check audit log persistence mechanisms",
                "Verify decision logging completeness",
                "Review audit trail integrity",
                "Implement backup audit procedures if critical",
            ],
            MonitoringMetric.CRYPTOGRAPHIC_VERIFICATION: [
                "Check cryptographic key health",
                "Verify signature generation/verification",
                "Review integrity proof systems",
                "Regenerate compromised keys if critical",
            ],
            MonitoringMetric.RESPONSE_TIME: [
                "Check system resource utilization",
                "Review performance bottlenecks",
                "Consider scaling system components",
                "Implement performance optimizations",
            ],
        }

        base_actions = actions.get(metric, ["Review system component health"])

        if status == ComplianceStatus.CRITICAL:
            base_actions.append("IMMEDIATE ACTION REQUIRED: Escalate to system administrators")
            base_actions.append("Consider temporary system restrictions to prevent constitutional harm")

        return base_actions

    # Metric calculation methods
    def _calculate_compliance_rate(self) -> float:
        """Calculate overall constitutional compliance rate"""
        try:
            recent_cutoff = time.time() - (24 * 3600)  # Last 24 hours
            recent_entries = [e for e in self.merkle_audit.audit_entries if e.timestamp >= recent_cutoff]

            if not recent_entries:
                return 100.0  # No recent entries = 100% compliant

            violations = sum(1 for e in recent_entries if e.violation_type)
            compliance_rate = (1 - violations / len(recent_entries)) * 100

            return max(0, min(100, compliance_rate))

        except Exception as e:
            self.logger.error(f"Error calculating compliance rate: {e}")
            return 0.0

    def _calculate_violation_frequency(self) -> float:
        """Calculate constitutional violation frequency (per hour)"""
        try:
            recent_cutoff = time.time() - (3600)  # Last hour
            recent_violations = [
                e for e in self.merkle_audit.audit_entries if e.timestamp >= recent_cutoff and e.violation_type
            ]

            return len(recent_violations)  # Violations per hour

        except Exception as e:
            self.logger.error(f"Error calculating violation frequency: {e}")
            return 0.0

    def _calculate_current_violations(self) -> int:
        """Calculate current active violations"""
        try:
            recent_cutoff = time.time() - (24 * 3600)  # Last 24 hours
            recent_violations = [
                e for e in self.merkle_audit.audit_entries if e.timestamp >= recent_cutoff and e.violation_type
            ]

            return len(recent_violations)

        except Exception as e:
            self.logger.error(f"Error calculating current violations: {e}")
            return 0

    def _calculate_democratic_participation(self) -> float:
        """Calculate democratic participation rate"""
        try:
            if not self.governance_audit.participants:
                return 100.0  # No participants = not applicable

            recent_cutoff = time.time() - (7 * 24 * 3600)  # Last week
            active_participants = set()

            for event in self.governance_audit.democratic_events:
                if event.timestamp >= recent_cutoff:
                    active_participants.add(event.participant_id_hash)

            participation_rate = len(active_participants) / len(self.governance_audit.participants) * 100
            return max(0, min(100, participation_rate))

        except Exception as e:
            self.logger.error(f"Error calculating democratic participation: {e}")
            return 0.0

    def _calculate_system_integrity(self) -> float:
        """Calculate overall system integrity score"""
        try:
            integrity_factors = []

            # Merkle tree integrity
            if self.merkle_audit.merkle_trees:
                integrity_factors.append(90.0)  # Trees exist and functional
            else:
                integrity_factors.append(70.0)  # No trees yet

            # Decision logging integrity
            if self.decision_logger.decision_logs:
                integrity_factors.append(95.0)  # Decision logs active
            else:
                integrity_factors.append(80.0)  # No decisions yet

            # Cryptographic verification
            if self.crypto_verifier.system_integrity_hash:
                integrity_factors.append(100.0)  # Crypto system operational
            else:
                integrity_factors.append(60.0)  # Crypto system issues

            # Privacy system integrity
            integrity_factors.append(95.0)  # Privacy system always operational

            return statistics.mean(integrity_factors)

        except Exception as e:
            self.logger.error(f"Error calculating system integrity: {e}")
            return 0.0

    def _calculate_privacy_preservation(self) -> float:
        """Calculate privacy preservation rate"""
        try:
            recent_cutoff = time.time() - (24 * 3600)  # Last 24 hours
            recent_entries = [e for e in self.merkle_audit.audit_entries if e.timestamp >= recent_cutoff]

            if not recent_entries:
                return 100.0  # No entries = 100% privacy preserved

            privacy_preserving = sum(1 for e in recent_entries if e.audit_level.value in ["gold", "platinum"])

            preservation_rate = privacy_preserving / len(recent_entries) * 100
            return max(0, min(100, preservation_rate))

        except Exception as e:
            self.logger.error(f"Error calculating privacy preservation: {e}")
            return 0.0

    def _calculate_audit_completeness(self) -> float:
        """Calculate audit trail completeness"""
        try:
            # Check if all expected audit components are operational
            components_operational = 0
            total_components = 5

            if len(self.merkle_audit.audit_entries) > 0:
                components_operational += 1
            if len(self.decision_logger.decision_logs) > 0:
                components_operational += 1
            if len(self.privacy_system.zk_proofs) >= 0:  # Can be 0 initially
                components_operational += 1
            if len(self.governance_audit.participants) >= 0:  # Can be 0 initially
                components_operational += 1
            if self.crypto_verifier.system_integrity_hash:
                components_operational += 1

            completeness = components_operational / total_components * 100
            return max(0, min(100, completeness))

        except Exception as e:
            self.logger.error(f"Error calculating audit completeness: {e}")
            return 0.0

    def _calculate_crypto_health(self) -> float:
        """Calculate cryptographic verification health"""
        try:
            crypto_metrics = self.crypto_verifier.get_cryptographic_verification_metrics()
            success_rate = crypto_metrics["verification_rates"]["success_rate"]

            # Adjust based on number of verifications performed
            total_verifications = crypto_metrics["verification_metrics"]["total_verifications"]

            if total_verifications == 0:
                return 100.0  # No verifications yet = healthy system

            # Health score based on success rate and system operational status
            base_health = success_rate

            if crypto_metrics["security_status"]["master_keys_operational"]:
                base_health = min(100, base_health + 5)

            return max(0, min(100, base_health))

        except Exception as e:
            self.logger.error(f"Error calculating crypto health: {e}")
            return 0.0

    def _calculate_response_time(self) -> float:
        """Calculate average system response time"""
        return self.performance_metrics["average_response_time"]

    def _calculate_metric_trend(self, metric: MonitoringMetric) -> str:
        """Calculate trend for specific metric"""
        try:
            if len(self.health_snapshots) < 2:
                return "insufficient_data"

            recent_snapshots = list(self.health_snapshots)[-10:]  # Last 10 snapshots

            if metric == MonitoringMetric.COMPLIANCE_RATE:
                values = [s.overall_compliance_rate for s in recent_snapshots]
            elif metric == MonitoringMetric.DEMOCRATIC_PARTICIPATION:
                values = [s.democratic_participation_rate for s in recent_snapshots]
            elif metric == MonitoringMetric.SYSTEM_INTEGRITY:
                values = [s.system_integrity_score for s in recent_snapshots]
            elif metric == MonitoringMetric.PRIVACY_PRESERVATION:
                values = [s.privacy_preservation_rate for s in recent_snapshots]
            else:
                return "unknown"

            if len(values) < 2:
                return "insufficient_data"

            # Simple trend calculation
            if values[-1] > values[-2] * 1.05:
                return "improving"
            elif values[-1] < values[-2] * 0.95:
                return "declining"
            else:
                return "stable"

        except Exception as e:
            self.logger.error(f"Error calculating metric trend: {e}")
            return "unknown"

    def _calculate_system_load(self) -> float:
        """Calculate current system load"""
        try:
            # Simple load calculation based on recent activity
            recent_cutoff = time.time() - (3600)  # Last hour

            recent_audit_entries = len([e for e in self.merkle_audit.audit_entries if e.timestamp >= recent_cutoff])
            recent_decisions = len([d for d in self.decision_logger.decision_logs if d.timestamp >= recent_cutoff])
            recent_governance_events = len(
                [e for e in self.governance_audit.democratic_events if e.timestamp >= recent_cutoff]
            )

            total_activity = recent_audit_entries + recent_decisions + recent_governance_events

            # Normalize to 0-100 scale (assuming max 100 events per hour is high load)
            load_percentage = min(100, (total_activity / 100) * 100)

            return load_percentage

        except Exception as e:
            self.logger.error(f"Error calculating system load: {e}")
            return 0.0

    def _find_related_alerts(self, metric: MonitoringMetric) -> List[str]:
        """Find alerts related to specific metric"""
        try:
            related_metrics = {
                MonitoringMetric.COMPLIANCE_RATE: [
                    MonitoringMetric.VIOLATION_FREQUENCY,
                    MonitoringMetric.SYSTEM_INTEGRITY,
                ],
                MonitoringMetric.VIOLATION_FREQUENCY: [MonitoringMetric.COMPLIANCE_RATE],
                MonitoringMetric.DEMOCRATIC_PARTICIPATION: [MonitoringMetric.SYSTEM_INTEGRITY],
                MonitoringMetric.SYSTEM_INTEGRITY: [
                    MonitoringMetric.CRYPTOGRAPHIC_VERIFICATION,
                    MonitoringMetric.AUDIT_TRAIL_COMPLETENESS,
                ],
                MonitoringMetric.PRIVACY_PRESERVATION: [MonitoringMetric.AUDIT_TRAIL_COMPLETENESS],
                MonitoringMetric.AUDIT_TRAIL_COMPLETENESS: [MonitoringMetric.SYSTEM_INTEGRITY],
                MonitoringMetric.CRYPTOGRAPHIC_VERIFICATION: [MonitoringMetric.SYSTEM_INTEGRITY],
                MonitoringMetric.RESPONSE_TIME: [],  # Performance related to all
            }

            related_metric_types = related_metrics.get(metric, [])

            with self._lock:
                related_alerts = []
                for alert in self.active_alerts.values():
                    if alert.metric in related_metric_types:
                        related_alerts.append(alert.alert_id)

                return related_alerts

        except Exception as e:
            self.logger.error(f"Error finding related alerts: {e}")
            return []

    async def _check_system_wide_issues(self, health_snapshot: SystemHealthSnapshot):
        """Check for system-wide constitutional issues"""
        try:
            # Check for cascading failures
            critical_components_down = sum(
                1 for status in health_snapshot.component_status.values() if status != "operational"
            )

            if critical_components_down >= 3:
                await self._generate_system_wide_alert(
                    "Multiple critical components experiencing issues", AlertSeverity.EMERGENCY, health_snapshot
                )

            # Check for constitutional crisis indicators
            if (
                health_snapshot.overall_compliance_rate < 80
                and health_snapshot.active_violations > 10
                and health_snapshot.system_integrity_score < 85
            ):

                await self._generate_system_wide_alert(
                    "Constitutional crisis indicators detected", AlertSeverity.CRITICAL, health_snapshot
                )

        except Exception as e:
            self.logger.error(f"Error checking system-wide issues: {e}")

    async def _generate_system_wide_alert(
        self, message: str, severity: AlertSeverity, health_snapshot: SystemHealthSnapshot
    ):
        """Generate system-wide alert"""
        alert_id = f"system_wide_{int(time.time() * 1000000)}"

        alert = ComplianceAlert(
            alert_id=alert_id,
            timestamp=time.time(),
            severity=severity,
            metric=MonitoringMetric.SYSTEM_INTEGRITY,
            current_value=health_snapshot.system_integrity_score,
            threshold_value=90.0,
            message=message,
            affected_components=["all_components"],
            recommended_actions=[
                "IMMEDIATE: Contact system administrators",
                "IMMEDIATE: Implement emergency constitutional procedures",
                "Review all system components for failures",
                "Consider temporary service restrictions",
                "Initiate constitutional crisis response protocol",
            ],
            alert_context={
                "system_wide_alert": True,
                "health_snapshot": asdict(health_snapshot),
                "alert_type": "system_crisis",
            },
        )

        with self._lock:
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            self.performance_metrics["violations_detected"] += 1

        # Trigger all callbacks for system-wide alerts
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in system-wide alert callback: {e}")

        self.logger.critical(f"SYSTEM-WIDE ALERT {alert_id}: {message}")

    def _cleanup_resolved_alerts(self):
        """Clean up resolved alerts"""
        try:
            with self._lock:
                # Remove alerts older than 24 hours
                cutoff_time = time.time() - (24 * 3600)
                expired_alerts = [
                    alert_id for alert_id, alert in self.active_alerts.items() if alert.timestamp < cutoff_time
                ]

                for alert_id in expired_alerts:
                    del self.active_alerts[alert_id]

                if expired_alerts:
                    self.logger.info(f"Cleaned up {len(expired_alerts)} expired alerts")

        except Exception as e:
            self.logger.error(f"Error cleaning up resolved alerts: {e}")

    # PUBLIC API METHODS

    def add_alert_callback(self, callback: Callable[[ComplianceAlert], None]):
        """Add callback function for alert notifications"""
        self.alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable[[ComplianceAlert], None]):
        """Remove alert callback function"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)

    def get_current_system_health(self) -> Optional[SystemHealthSnapshot]:
        """Get current system health snapshot"""
        with self._lock:
            return self.current_health_status

    def get_active_alerts(self) -> List[ComplianceAlert]:
        """Get all active compliance alerts"""
        with self._lock:
            return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[ComplianceAlert]:
        """Get recent alert history"""
        with self._lock:
            return list(self.alert_history)[-limit:]

    def get_monitoring_metrics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring metrics"""
        with self._lock:
            current_health = self.current_health_status

            return {
                "monitoring_status": {
                    "is_monitoring": self.is_monitoring,
                    "monitoring_interval_seconds": self.monitoring_interval,
                    "system_uptime_hours": (time.time() - self.performance_metrics["system_uptime"]) / 3600,
                },
                "performance_metrics": self.performance_metrics.copy(),
                "current_system_health": asdict(current_health) if current_health else None,
                "alert_summary": {
                    "total_active_alerts": len(self.active_alerts),
                    "critical_alerts": len(
                        [a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL]
                    ),
                    "warning_alerts": len(
                        [a for a in self.active_alerts.values() if a.severity == AlertSeverity.WARNING]
                    ),
                    "total_alerts_generated": self.performance_metrics["alerts_generated"],
                },
                "compliance_thresholds": {
                    metric.value: asdict(threshold) for metric, threshold in self.compliance_thresholds.items()
                },
            }

    def update_compliance_threshold(self, metric: MonitoringMetric, threshold: ComplianceThreshold):
        """Update compliance threshold for specific metric"""
        self.compliance_thresholds[metric] = threshold
        self.logger.info(f"Updated compliance threshold for {metric.value}")

    def acknowledge_alert(self, alert_id: str, acknowledger: str, notes: str = ""):
        """Acknowledge active alert"""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.alert_context["acknowledged"] = {
                    "acknowledger": acknowledger,
                    "timestamp": time.time(),
                    "notes": notes,
                }
                self.logger.info(f"Alert {alert_id} acknowledged by {acknowledger}")
                return True
            return False

    def resolve_alert(self, alert_id: str, resolver: str, resolution_notes: str = ""):
        """Resolve active alert"""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.alert_context["resolved"] = {
                    "resolver": resolver,
                    "timestamp": time.time(),
                    "resolution_notes": resolution_notes,
                }
                # Move to history but keep in active for cleanup cycle
                self.logger.info(f"Alert {alert_id} resolved by {resolver}")
                return True
            return False


# Example alert callback function
def example_alert_callback(alert: ComplianceAlert):
    """Example alert callback for logging/notification"""
    print(f"CONSTITUTIONAL ALERT: {alert.severity.value.upper()} - {alert.message}")
    if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
        print("IMMEDIATE ACTION REQUIRED!")
        for action in alert.recommended_actions[:3]:  # Show first 3 actions
            print(f"  - {action}")


# Example usage
if __name__ == "__main__":
    import asyncio

    async def test_compliance_monitoring():
        # Import required components (would be initialized elsewhere)
        from .merkle_audit import ConstitutionalMerkleAudit
        from .constitutional_logging import ConstitutionalDecisionLogger
        from .privacy_preserving_audit import PrivacyPreservingAuditSystem
        from .governance_audit import GovernanceAuditTrail
        from .cryptographic_verification import ConstitutionalCryptographicVerifier

        # Initialize components
        merkle_audit = ConstitutionalMerkleAudit()
        decision_logger = ConstitutionalDecisionLogger()
        privacy_system = PrivacyPreservingAuditSystem()
        governance_audit = GovernanceAuditTrail()
        crypto_verifier = ConstitutionalCryptographicVerifier()

        # Create monitor
        monitor = RealTimeComplianceMonitor(
            merkle_audit,
            decision_logger,
            privacy_system,
            governance_audit,
            crypto_verifier,
            monitoring_interval=10,  # 10 second intervals for testing
        )

        # Add alert callback
        monitor.add_alert_callback(example_alert_callback)

        # Start monitoring
        await monitor.start_monitoring()

        print("Real-time constitutional compliance monitoring started...")
        print("Monitor will run for 60 seconds...")

        # Let it run for demonstration
        await asyncio.sleep(60)

        # Get current system health
        health = monitor.get_current_system_health()
        if health:
            print(f"Current compliance rate: {health.overall_compliance_rate:.2f}%")
            print(f"Active violations: {health.active_violations}")
            print(f"System integrity: {health.system_integrity_score:.2f}%")

        # Get monitoring metrics
        metrics = monitor.get_monitoring_metrics()
        print(f"Monitoring cycles completed: {metrics['performance_metrics']['monitoring_cycles_completed']}")
        print(f"Total alerts generated: {metrics['alert_summary']['total_alerts_generated']}")

        # Stop monitoring
        monitor.stop_monitoring()

        print("Real-time constitutional compliance monitoring stopped.")

    # Run test
    # asyncio.run(test_compliance_monitoring())
