"""
Comprehensive Audit Trail System for Federated Security

This module provides immutable audit trails and forensic capabilities for federated
learning systems, integrating with BetaNet anchoring for blockchain-based verification.
Includes compliance reporting, anomaly detection, and forensic analysis tools.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Dict, List, Optional
import secrets
from collections import defaultdict, deque

from ..proof.betanet_anchor import BetanetAnchorService

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of auditable events"""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    MODEL_UPDATE = "model_update"
    GRADIENT_SUBMISSION = "gradient_submission"
    AGGREGATION = "aggregation"
    KEY_ROTATION = "key_rotation"
    SECURITY_ALERT = "security_alert"
    CONFIGURATION_CHANGE = "configuration_change"
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    ERROR_OCCURRED = "error_occurred"
    COMPLIANCE_CHECK = "compliance_check"


class EventSeverity(Enum):
    """Severity levels for audit events"""

    INFO = "info"
    NOTICE = "notice"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""

    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    NIST_CSF = "nist_csf"
    PCI_DSS = "pci_dss"


@dataclass
class AuditEvent:
    """Immutable audit event record"""

    event_id: str
    timestamp: float
    event_type: EventType
    severity: EventSeverity
    source_node: str
    target_node: Optional[str]
    actor: str  # Who performed the action
    action: str  # What action was performed
    resource: str  # What resource was affected
    outcome: str  # Success, failure, partial
    details: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None
    parent_event_id: Optional[str] = None  # For event chains
    correlation_id: Optional[str] = None  # For related events

    def __post_init__(self):
        """Calculate checksum after initialization"""
        if self.checksum is None:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate event integrity checksum"""
        event_data = {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "outcome": self.outcome,
            "details": self.details,
            "metadata": self.metadata,
        }

        json_data = json.dumps(event_data, sort_keys=True, default=str)
        return hashlib.sha256(json_data.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify event integrity"""
        expected_checksum = self._calculate_checksum()
        return expected_checksum == self.checksum


@dataclass
class ComplianceRule:
    """Compliance rule definition"""

    rule_id: str
    framework: ComplianceFramework
    rule_name: str
    description: str
    event_patterns: List[Dict[str, Any]]  # Patterns that trigger this rule
    required_fields: List[str]
    retention_period_days: int
    notification_required: bool
    automated_response: Optional[str] = None


@dataclass
class ForensicQuery:
    """Forensic analysis query"""

    query_id: str
    analyst: str
    query_type: str
    time_range: Tuple[float, float]
    filters: Dict[str, Any]
    nodes_of_interest: List[str]
    event_types: List[EventType]
    correlation_analysis: bool = False
    pattern_detection: bool = False
    created_at: float = field(default_factory=time.time)


@dataclass
class AnomalyPattern:
    """Detected anomaly pattern in audit logs"""

    pattern_id: str
    pattern_type: str
    description: str
    affected_nodes: List[str]
    time_window: Tuple[float, float]
    severity: EventSeverity
    confidence: float  # 0.0 to 1.0
    evidence_events: List[str]  # Event IDs
    statistical_metrics: Dict[str, float]
    detected_at: float = field(default_factory=time.time)


class EventBuffer:
    """High-performance circular buffer for audit events"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.event_index: Dict[str, int] = {}  # event_id -> buffer position
        self.lock = asyncio.Lock()

    async def add_event(self, event: AuditEvent):
        """Add event to buffer"""
        async with self.lock:
            # Remove old event from index if buffer is full
            if len(self.buffer) >= self.max_size:
                old_event = self.buffer[0]
                if old_event.event_id in self.event_index:
                    del self.event_index[old_event.event_id]

            # Add new event
            self.buffer.append(event)
            self.event_index[event.event_id] = len(self.buffer) - 1

    async def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Retrieve event by ID"""
        async with self.lock:
            if event_id in self.event_index:
                position = self.event_index[event_id]
                if position < len(self.buffer):
                    return self.buffer[position]
        return None

    async def get_recent_events(self, count: int = 100) -> List[AuditEvent]:
        """Get most recent events"""
        async with self.lock:
            return list(self.buffer)[-count:]

    async def search_events(
        self,
        time_range: Optional[Tuple[float, float]] = None,
        event_types: Optional[List[EventType]] = None,
        severity: Optional[EventSeverity] = None,
        nodes: Optional[List[str]] = None,
    ) -> List[AuditEvent]:
        """Search events with filters"""
        async with self.lock:
            results = []

            for event in self.buffer:
                # Time range filter
                if time_range and not (time_range[0] <= event.timestamp <= time_range[1]):
                    continue

                # Event type filter
                if event_types and event.event_type not in event_types:
                    continue

                # Severity filter
                if severity and event.severity != severity:
                    continue

                # Node filter
                if nodes and not (event.source_node in nodes or (event.target_node and event.target_node in nodes)):
                    continue

                results.append(event)

            return sorted(results, key=lambda x: x.timestamp, reverse=True)


class ComplianceEngine:
    """Engine for compliance monitoring and reporting"""

    def __init__(self):
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.rule_violations: List[Dict[str, Any]] = []
        self.compliance_reports: Dict[str, Dict[str, Any]] = {}
        self._initialize_standard_rules()

    def _initialize_standard_rules(self):
        """Initialize standard compliance rules"""
        # GDPR Data Access Logging
        self.add_compliance_rule(
            ComplianceRule(
                rule_id="GDPR_001",
                framework=ComplianceFramework.GDPR,
                rule_name="Data Access Logging",
                description="Log all personal data access events",
                event_patterns=[
                    {"event_type": "data_access", "details.data_type": "personal"},
                    {"event_type": "model_update", "details.contains_personal_data": True},
                ],
                required_fields=["actor", "resource", "timestamp", "outcome"],
                retention_period_days=2555,  # 7 years
                notification_required=False,
            )
        )

        # HIPAA Audit Controls
        self.add_compliance_rule(
            ComplianceRule(
                rule_id="HIPAA_001",
                framework=ComplianceFramework.HIPAA,
                rule_name="Access Control Audit",
                description="Audit access to protected health information",
                event_patterns=[
                    {"event_type": "data_access", "details.data_classification": "phi"},
                    {"event_type": "authentication", "resource": "phi_system"},
                ],
                required_fields=["actor", "resource", "timestamp", "outcome", "source_node"],
                retention_period_days=2190,  # 6 years
                notification_required=True,
            )
        )

        # SOC2 Security Monitoring
        self.add_compliance_rule(
            ComplianceRule(
                rule_id="SOC2_001",
                framework=ComplianceFramework.SOC2,
                rule_name="Security Event Monitoring",
                description="Monitor security-related events",
                event_patterns=[
                    {"event_type": "security_alert"},
                    {"event_type": "authentication", "outcome": "failure"},
                    {"event_type": "configuration_change", "severity": "critical"},
                ],
                required_fields=["event_id", "timestamp", "severity", "details"],
                retention_period_days=365,
                notification_required=True,
                automated_response="escalate_security_team",
            )
        )

    def add_compliance_rule(self, rule: ComplianceRule):
        """Add compliance rule"""
        self.compliance_rules[rule.rule_id] = rule
        logger.info(f"Added compliance rule: {rule.rule_name} ({rule.framework.value})")

    async def check_compliance(self, event: AuditEvent) -> List[Dict[str, Any]]:
        """Check event against compliance rules"""
        violations = []

        for rule in self.compliance_rules.values():
            if await self._event_matches_rule(event, rule):
                violation = await self._validate_compliance(event, rule)
                if violation:
                    violations.append(violation)
                    self.rule_violations.append(violation)

        return violations

    async def _event_matches_rule(self, event: AuditEvent, rule: ComplianceRule) -> bool:
        """Check if event matches rule pattern"""
        for pattern in rule.event_patterns:
            if await self._matches_pattern(event, pattern):
                return True
        return False

    async def _matches_pattern(self, event: AuditEvent, pattern: Dict[str, Any]) -> bool:
        """Check if event matches specific pattern"""
        for key, expected_value in pattern.items():
            if key == "event_type":
                if event.event_type.value != expected_value:
                    return False
            elif key.startswith("details."):
                detail_key = key[8:]  # Remove "details." prefix
                if detail_key not in event.details:
                    return False
                if event.details[detail_key] != expected_value:
                    return False
            elif key == "severity":
                if event.severity.value != expected_value:
                    return False
            elif key == "outcome":
                if event.outcome != expected_value:
                    return False
            # Add more pattern matching as needed

        return True

    async def _validate_compliance(self, event: AuditEvent, rule: ComplianceRule) -> Optional[Dict[str, Any]]:
        """Validate event compliance with rule"""
        violations = []

        # Check required fields
        for field in rule.required_fields:
            if field == "actor" and not event.actor:
                violations.append(f"Missing required field: {field}")
            elif field == "resource" and not event.resource:
                violations.append(f"Missing required field: {field}")
            elif field == "timestamp" and not event.timestamp:
                violations.append(f"Missing required field: {field}")
            elif field == "outcome" and not event.outcome:
                violations.append(f"Missing required field: {field}")
            elif field == "source_node" and not event.source_node:
                violations.append(f"Missing required field: {field}")
            elif field == "event_id" and not event.event_id:
                violations.append(f"Missing required field: {field}")
            elif field == "details" and not event.details:
                violations.append(f"Missing required field: {field}")

        if violations:
            return {
                "rule_id": rule.rule_id,
                "framework": rule.framework.value,
                "rule_name": rule.rule_name,
                "event_id": event.event_id,
                "violations": violations,
                "timestamp": time.time(),
                "requires_notification": rule.notification_required,
                "automated_response": rule.automated_response,
            }

        return None

    async def generate_compliance_report(
        self, framework: ComplianceFramework, time_range: Tuple[float, float], events: List[AuditEvent]
    ) -> Dict[str, Any]:
        """Generate compliance report"""
        framework_rules = [rule for rule in self.compliance_rules.values() if rule.framework == framework]

        # Filter events in time range
        filtered_events = [event for event in events if time_range[0] <= event.timestamp <= time_range[1]]

        # Analyze compliance
        total_events = len(filtered_events)
        compliant_events = 0
        violations_by_rule = defaultdict(int)

        for event in filtered_events:
            event_violations = await self.check_compliance(event)
            if not event_violations:
                compliant_events += 1
            else:
                for violation in event_violations:
                    violations_by_rule[violation["rule_id"]] += 1

        compliance_rate = compliant_events / total_events if total_events > 0 else 1.0

        report = {
            "framework": framework.value,
            "report_period": {
                "start": datetime.fromtimestamp(time_range[0], UTC).isoformat(),
                "end": datetime.fromtimestamp(time_range[1], UTC).isoformat(),
            },
            "summary": {
                "total_events": total_events,
                "compliant_events": compliant_events,
                "violations": len(self.rule_violations),
                "compliance_rate": compliance_rate,
                "overall_status": "COMPLIANT" if compliance_rate >= 0.95 else "NON_COMPLIANT",
            },
            "rule_analysis": [
                {
                    "rule_id": rule.rule_id,
                    "rule_name": rule.rule_name,
                    "violations": violations_by_rule.get(rule.rule_id, 0),
                    "compliance_rate": (
                        1.0 - (violations_by_rule.get(rule.rule_id, 0) / total_events) if total_events > 0 else 1.0
                    ),
                }
                for rule in framework_rules
            ],
            "violations_detail": [
                v
                for v in self.rule_violations
                if v.get("framework") == framework.value and time_range[0] <= v.get("timestamp", 0) <= time_range[1]
            ],
            "generated_at": datetime.now(UTC).isoformat(),
        }

        # Store report
        report_id = f"{framework.value}_{int(time.time())}"
        self.compliance_reports[report_id] = report

        logger.info(f"Generated compliance report for {framework.value}: {compliance_rate:.2%} compliance rate")

        return report


class AnomalyDetectionEngine:
    """Engine for detecting anomalous patterns in audit logs"""

    def __init__(self):
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        self.detected_anomalies: Dict[str, AnomalyPattern] = {}
        self.detection_rules: Dict[str, Dict[str, Any]] = {}
        self._initialize_detection_rules()

    def _initialize_detection_rules(self):
        """Initialize anomaly detection rules"""
        self.detection_rules = {
            "unusual_authentication_volume": {
                "description": "Detect unusual authentication attempt volumes",
                "event_type": EventType.AUTHENTICATION,
                "window_minutes": 60,
                "threshold_multiplier": 3.0,
                "baseline_period_hours": 24,
            },
            "failed_authentication_spike": {
                "description": "Detect spikes in authentication failures",
                "event_type": EventType.AUTHENTICATION,
                "filter": {"outcome": "failure"},
                "window_minutes": 15,
                "threshold_count": 10,
                "severity": EventSeverity.WARNING,
            },
            "configuration_change_frequency": {
                "description": "Detect unusual configuration change patterns",
                "event_type": EventType.CONFIGURATION_CHANGE,
                "window_minutes": 30,
                "threshold_multiplier": 2.5,
                "baseline_period_hours": 168,  # 1 week
            },
            "data_access_anomaly": {
                "description": "Detect unusual data access patterns",
                "event_type": EventType.DATA_ACCESS,
                "window_minutes": 120,
                "threshold_multiplier": 4.0,
                "baseline_period_hours": 72,  # 3 days
            },
        }

    async def update_baseline_metrics(self, events: List[AuditEvent]):
        """Update baseline metrics from historical events"""
        logger.info("Updating baseline metrics for anomaly detection")

        # Group events by type and time windows
        event_counts = defaultdict(lambda: defaultdict(int))

        for event in events:
            event_type = event.event_type.value
            hour = int(event.timestamp // 3600)  # Hour bucket
            event_counts[event_type][hour] += 1

        # Calculate baseline statistics
        for event_type, hourly_counts in event_counts.items():
            if len(hourly_counts) < 24:  # Need at least 24 hours of data
                continue

            counts = list(hourly_counts.values())
            self.baseline_metrics[event_type] = {
                "mean": sum(counts) / len(counts),
                "std": (sum((x - sum(counts) / len(counts)) ** 2 for x in counts) / len(counts)) ** 0.5,
                "min": min(counts),
                "max": max(counts),
                "sample_size": len(counts),
            }

        logger.info(f"Updated baselines for {len(self.baseline_metrics)} event types")

    async def detect_anomalies(
        self, recent_events: List[AuditEvent], time_window_minutes: int = 60
    ) -> List[AnomalyPattern]:
        """Detect anomalies in recent events"""
        current_time = time.time()
        window_start = current_time - (time_window_minutes * 60)

        # Filter events to analysis window
        window_events = [event for event in recent_events if event.timestamp >= window_start]

        detected_anomalies = []

        # Apply each detection rule
        for rule_id, rule in self.detection_rules.items():
            anomaly = await self._apply_detection_rule(rule_id, rule, window_events)
            if anomaly:
                detected_anomalies.append(anomaly)
                self.detected_anomalies[anomaly.pattern_id] = anomaly

        return detected_anomalies

    async def _apply_detection_rule(
        self, rule_id: str, rule: Dict[str, Any], events: List[AuditEvent]
    ) -> Optional[AnomalyPattern]:
        """Apply specific detection rule"""
        # Filter events by rule criteria
        filtered_events = []
        target_event_type = rule.get("event_type")
        event_filter = rule.get("filter", {})

        for event in events:
            if target_event_type and event.event_type != target_event_type:
                continue

            # Apply additional filters
            passes_filter = True
            for filter_key, filter_value in event_filter.items():
                if filter_key == "outcome" and event.outcome != filter_value:
                    passes_filter = False
                    break
                # Add more filter conditions as needed

            if passes_filter:
                filtered_events.append(event)

        if not filtered_events:
            return None

        # Analyze for anomalies
        event_count = len(filtered_events)

        # Threshold-based detection
        if "threshold_count" in rule:
            if event_count >= rule["threshold_count"]:
                return await self._create_anomaly_pattern(rule_id, rule, filtered_events, "threshold_exceeded")

        # Statistical anomaly detection
        if "threshold_multiplier" in rule and target_event_type:
            baseline = self.baseline_metrics.get(target_event_type.value)
            if baseline:
                expected_count = baseline["mean"]
                threshold = expected_count + (rule["threshold_multiplier"] * baseline["std"])

                if event_count > threshold:
                    return await self._create_anomaly_pattern(rule_id, rule, filtered_events, "statistical_anomaly")

        return None

    async def _create_anomaly_pattern(
        self, rule_id: str, rule: Dict[str, Any], events: List[AuditEvent], anomaly_type: str
    ) -> AnomalyPattern:
        """Create anomaly pattern from detected anomaly"""
        pattern_id = f"{rule_id}_{anomaly_type}_{secrets.token_hex(4)}"

        # Extract affected nodes
        affected_nodes = set()
        for event in events:
            affected_nodes.add(event.source_node)
            if event.target_node:
                affected_nodes.add(event.target_node)

        # Calculate time window
        timestamps = [event.timestamp for event in events]
        time_window = (min(timestamps), max(timestamps))

        # Calculate statistical metrics
        event_count = len(events)
        window_duration = max(1, time_window[1] - time_window[0])
        event_rate = event_count / (window_duration / 60)  # events per minute

        statistical_metrics = {
            "event_count": event_count,
            "event_rate_per_minute": event_rate,
            "unique_nodes": len(affected_nodes),
            "time_span_minutes": window_duration / 60,
        }

        # Determine confidence and severity
        confidence = 0.8 if anomaly_type == "statistical_anomaly" else 0.9
        severity = rule.get("severity", EventSeverity.WARNING)

        pattern = AnomalyPattern(
            pattern_id=pattern_id,
            pattern_type=anomaly_type,
            description=rule.get("description", f"Anomaly detected by rule {rule_id}"),
            affected_nodes=list(affected_nodes),
            time_window=time_window,
            severity=severity,
            confidence=confidence,
            evidence_events=[event.event_id for event in events],
            statistical_metrics=statistical_metrics,
        )

        logger.warning(f"Detected anomaly: {pattern.description} (confidence: {confidence:.2f})")

        return pattern


class ForensicAnalysisEngine:
    """Engine for forensic analysis of audit logs"""

    def __init__(self):
        self.queries: Dict[str, ForensicQuery] = {}
        self.analysis_results: Dict[str, Dict[str, Any]] = {}

    async def create_forensic_query(self, query: ForensicQuery) -> str:
        """Create forensic analysis query"""
        self.queries[query.query_id] = query
        logger.info(f"Created forensic query {query.query_id} by analyst {query.analyst}")
        return query.query_id

    async def execute_forensic_analysis(self, query_id: str, events: List[AuditEvent]) -> Dict[str, Any]:
        """Execute forensic analysis query"""
        if query_id not in self.queries:
            raise ValueError(f"Forensic query {query_id} not found")

        query = self.queries[query_id]

        # Filter events by query criteria
        filtered_events = await self._filter_events_for_query(events, query)

        # Perform analysis
        analysis_result = {
            "query_id": query_id,
            "analyst": query.analyst,
            "executed_at": time.time(),
            "total_events_analyzed": len(events),
            "filtered_events_count": len(filtered_events),
            "time_range": {
                "start": datetime.fromtimestamp(query.time_range[0], UTC).isoformat(),
                "end": datetime.fromtimestamp(query.time_range[1], UTC).isoformat(),
            },
            "summary": await self._generate_event_summary(filtered_events),
            "timeline": await self._generate_timeline(filtered_events),
            "node_analysis": await self._analyze_nodes(filtered_events, query.nodes_of_interest),
            "pattern_analysis": None,
            "correlation_analysis": None,
            "recommendations": [],
        }

        # Optional pattern detection
        if query.pattern_detection:
            analysis_result["pattern_analysis"] = await self._detect_forensic_patterns(filtered_events)

        # Optional correlation analysis
        if query.correlation_analysis:
            analysis_result["correlation_analysis"] = await self._perform_correlation_analysis(filtered_events)

        # Generate recommendations
        analysis_result["recommendations"] = await self._generate_forensic_recommendations(
            filtered_events, analysis_result
        )

        # Store results
        self.analysis_results[query_id] = analysis_result

        logger.info(f"Completed forensic analysis {query_id}: analyzed {len(filtered_events)} events")

        return analysis_result

    async def _filter_events_for_query(self, events: List[AuditEvent], query: ForensicQuery) -> List[AuditEvent]:
        """Filter events based on query criteria"""
        filtered = []

        for event in events:
            # Time range filter
            if not (query.time_range[0] <= event.timestamp <= query.time_range[1]):
                continue

            # Event type filter
            if query.event_types and event.event_type not in query.event_types:
                continue

            # Node filter
            if query.nodes_of_interest:
                if not (
                    event.source_node in query.nodes_of_interest
                    or (event.target_node and event.target_node in query.nodes_of_interest)
                ):
                    continue

            # Additional filters
            passes_filters = True
            for filter_key, filter_value in query.filters.items():
                if filter_key == "severity":
                    if event.severity.value != filter_value:
                        passes_filters = False
                        break
                elif filter_key == "outcome":
                    if event.outcome != filter_value:
                        passes_filters = False
                        break
                elif filter_key == "actor":
                    if event.actor != filter_value:
                        passes_filters = False
                        break
                # Add more filter conditions as needed

            if passes_filters:
                filtered.append(event)

        return filtered

    async def _generate_event_summary(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Generate summary statistics for events"""
        if not events:
            return {"total": 0}

        # Event type distribution
        event_type_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        outcome_counts = defaultdict(int)
        actor_counts = defaultdict(int)

        for event in events:
            event_type_counts[event.event_type.value] += 1
            severity_counts[event.severity.value] += 1
            outcome_counts[event.outcome] += 1
            actor_counts[event.actor] += 1

        return {
            "total": len(events),
            "time_span": {
                "start": min(event.timestamp for event in events),
                "end": max(event.timestamp for event in events),
            },
            "event_types": dict(event_type_counts),
            "severities": dict(severity_counts),
            "outcomes": dict(outcome_counts),
            "top_actors": dict(sorted(actor_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "unique_nodes": len(set(event.source_node for event in events)),
            "integrity_status": "verified" if all(event.verify_integrity() for event in events) else "compromised",
        }

    async def _generate_timeline(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Generate chronological timeline of significant events"""
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda x: x.timestamp)

        timeline = []
        for event in sorted_events:
            timeline_entry = {
                "timestamp": datetime.fromtimestamp(event.timestamp, UTC).isoformat(),
                "event_id": event.event_id,
                "type": event.event_type.value,
                "severity": event.severity.value,
                "source": event.source_node,
                "target": event.target_node,
                "actor": event.actor,
                "action": event.action,
                "outcome": event.outcome,
                "summary": f"{event.actor} performed {event.action} on {event.resource} with outcome {event.outcome}",
            }
            timeline.append(timeline_entry)

        return timeline

    async def _analyze_nodes(self, events: List[AuditEvent], nodes_of_interest: List[str]) -> Dict[str, Any]:
        """Analyze node behavior patterns"""
        node_stats = defaultdict(
            lambda: {
                "events_as_source": 0,
                "events_as_target": 0,
                "event_types": defaultdict(int),
                "success_rate": 0.0,
                "first_activity": None,
                "last_activity": None,
            }
        )

        for event in events:
            # Source node stats
            source_stats = node_stats[event.source_node]
            source_stats["events_as_source"] += 1
            source_stats["event_types"][event.event_type.value] += 1

            if source_stats["first_activity"] is None or event.timestamp < source_stats["first_activity"]:
                source_stats["first_activity"] = event.timestamp
            if source_stats["last_activity"] is None or event.timestamp > source_stats["last_activity"]:
                source_stats["last_activity"] = event.timestamp

            # Target node stats (if applicable)
            if event.target_node:
                target_stats = node_stats[event.target_node]
                target_stats["events_as_target"] += 1

        # Calculate success rates
        for node_id, stats in node_stats.items():
            node_events = [e for e in events if e.source_node == node_id]
            if node_events:
                successful_events = len([e for e in node_events if e.outcome == "success"])
                stats["success_rate"] = successful_events / len(node_events)

        # Focus on nodes of interest
        focused_analysis = {}
        for node in nodes_of_interest:
            if node in node_stats:
                focused_analysis[node] = dict(node_stats[node])
                # Convert defaultdict to dict for JSON serialization
                focused_analysis[node]["event_types"] = dict(focused_analysis[node]["event_types"])

        return {
            "nodes_analyzed": len(node_stats),
            "nodes_of_interest": focused_analysis,
            "most_active_nodes": dict(
                sorted(
                    {k: v["events_as_source"] + v["events_as_target"] for k, v in node_stats.items()}.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            ),
        }

    async def _detect_forensic_patterns(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Detect forensic patterns in filtered events"""
        patterns = {
            "privilege_escalation": await self._detect_privilege_escalation(events),
            "data_exfiltration": await self._detect_data_exfiltration(events),
            "lateral_movement": await self._detect_lateral_movement(events),
            "persistence_mechanisms": await self._detect_persistence_mechanisms(events),
        }

        return patterns

    async def _detect_privilege_escalation(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Detect privilege escalation patterns"""
        escalation_patterns = []

        # Look for sequences of authorization events with increasing privileges
        auth_events = [e for e in events if e.event_type == EventType.AUTHORIZATION]

        # Group by actor
        actor_events = defaultdict(list)
        for event in auth_events:
            actor_events[event.actor].append(event)

        for actor, actor_event_list in actor_events.items():
            # Sort by timestamp
            actor_event_list.sort(key=lambda x: x.timestamp)

            # Look for escalating access patterns
            for i in range(len(actor_event_list) - 1):
                current = actor_event_list[i]
                next_event = actor_event_list[i + 1]

                # Time proximity check (within 1 hour)
                if next_event.timestamp - current.timestamp <= 3600:
                    # Check for increasing privilege levels (simplified)
                    if "admin" in next_event.resource.lower() and "user" in current.resource.lower():
                        escalation_patterns.append(
                            {
                                "pattern": "privilege_escalation",
                                "actor": actor,
                                "events": [current.event_id, next_event.event_id],
                                "time_window": next_event.timestamp - current.timestamp,
                                "confidence": 0.7,
                            }
                        )

        return escalation_patterns

    async def _detect_data_exfiltration(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Detect data exfiltration patterns"""
        exfiltration_patterns = []

        # Look for unusual data access patterns
        data_events = [e for e in events if e.event_type == EventType.DATA_ACCESS]

        # Group by actor and analyze volume
        actor_data_access = defaultdict(list)
        for event in data_events:
            actor_data_access[event.actor].append(event)

        for actor, access_events in actor_data_access.items():
            # Check for high-volume access in short time
            if len(access_events) > 50:  # Threshold for suspicious activity
                time_span = max(e.timestamp for e in access_events) - min(e.timestamp for e in access_events)
                access_rate = len(access_events) / max(1, time_span / 60)  # events per minute

                if access_rate > 5:  # More than 5 accesses per minute
                    exfiltration_patterns.append(
                        {
                            "pattern": "high_volume_data_access",
                            "actor": actor,
                            "access_count": len(access_events),
                            "time_span_minutes": time_span / 60,
                            "access_rate": access_rate,
                            "confidence": 0.8,
                        }
                    )

        return exfiltration_patterns

    async def _detect_lateral_movement(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Detect lateral movement patterns"""
        lateral_patterns = []

        # Look for authentication events across multiple nodes
        auth_events = [e for e in events if e.event_type == EventType.AUTHENTICATION]

        # Group by actor
        actor_auths = defaultdict(list)
        for event in auth_events:
            actor_auths[event.actor].append(event)

        for actor, auth_list in actor_auths.items():
            # Check for authentication to multiple distinct nodes
            unique_nodes = set(e.source_node for e in auth_list)
            if len(unique_nodes) > 3:  # Authenticated to more than 3 nodes
                # Check time pattern
                auth_list.sort(key=lambda x: x.timestamp)
                time_span = auth_list[-1].timestamp - auth_list[0].timestamp

                if time_span <= 7200:  # Within 2 hours
                    lateral_patterns.append(
                        {
                            "pattern": "multi_node_authentication",
                            "actor": actor,
                            "node_count": len(unique_nodes),
                            "nodes": list(unique_nodes),
                            "time_span_minutes": time_span / 60,
                            "confidence": 0.7,
                        }
                    )

        return lateral_patterns

    async def _detect_persistence_mechanisms(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Detect persistence mechanism patterns"""
        persistence_patterns = []

        # Look for configuration changes that could establish persistence
        config_events = [e for e in events if e.event_type == EventType.CONFIGURATION_CHANGE]

        for event in config_events:
            # Check for suspicious configuration changes
            if any(keyword in event.action.lower() for keyword in ["schedule", "startup", "service", "daemon", "cron"]):
                persistence_patterns.append(
                    {
                        "pattern": "configuration_persistence",
                        "event_id": event.event_id,
                        "actor": event.actor,
                        "action": event.action,
                        "resource": event.resource,
                        "confidence": 0.6,
                    }
                )

        return persistence_patterns

    async def _perform_correlation_analysis(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Perform correlation analysis on events"""
        correlation_results = {
            "temporal_correlations": await self._find_temporal_correlations(events),
            "actor_correlations": await self._find_actor_correlations(events),
            "resource_correlations": await self._find_resource_correlations(events),
        }

        return correlation_results

    async def _find_temporal_correlations(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Find temporally correlated events"""
        correlations = []

        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda x: x.timestamp)

        # Look for events that occur within small time windows
        time_window = 300  # 5 minutes

        for i in range(len(sorted_events)):
            base_event = sorted_events[i]
            correlated_events = []

            # Find events within time window
            for j in range(i + 1, len(sorted_events)):
                if sorted_events[j].timestamp - base_event.timestamp > time_window:
                    break
                correlated_events.append(sorted_events[j])

            # If we found multiple related events
            if len(correlated_events) >= 2:
                correlations.append(
                    {
                        "base_event": base_event.event_id,
                        "correlated_events": [e.event_id for e in correlated_events],
                        "time_window_seconds": time_window,
                        "event_count": len(correlated_events) + 1,
                        "actors_involved": list(set([base_event.actor] + [e.actor for e in correlated_events])),
                        "correlation_strength": min(1.0, len(correlated_events) / 10.0),
                    }
                )

        return correlations

    async def _find_actor_correlations(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Find correlations between actors"""
        actor_interactions = defaultdict(lambda: defaultdict(int))

        # Track interactions between actors
        for event in events:
            if event.target_node:
                # Find events from target node around the same time
                time_window = 600  # 10 minutes
                related_events = [
                    e
                    for e in events
                    if (
                        e.source_node == event.target_node
                        and abs(e.timestamp - event.timestamp) <= time_window
                        and e.event_id != event.event_id
                    )
                ]

                for related in related_events:
                    actor_interactions[event.actor][related.actor] += 1

        # Find strong correlations
        strong_correlations = {}
        for actor1, interactions in actor_interactions.items():
            for actor2, count in interactions.items():
                if count >= 5:  # Threshold for significant interaction
                    correlation_key = tuple(sorted([actor1, actor2]))
                    if correlation_key not in strong_correlations:
                        strong_correlations[correlation_key] = count

        return {
            "total_actor_pairs": len(actor_interactions),
            "strong_correlations": [
                {"actors": list(pair), "interaction_count": count} for pair, count in strong_correlations.items()
            ],
        }

    async def _find_resource_correlations(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Find correlations between accessed resources"""
        resource_access_patterns = defaultdict(lambda: defaultdict(int))

        # Track which resources are accessed together
        actor_sessions = defaultdict(list)

        # Group events by actor and time proximity
        for event in events:
            actor_sessions[event.actor].append(event)

        # Analyze resource access patterns within sessions
        for actor, actor_events in actor_sessions.items():
            actor_events.sort(key=lambda x: x.timestamp)

            session_window = 1800  # 30 minutes
            current_session = []

            for event in actor_events:
                # Check if this event starts a new session
                if not current_session or event.timestamp - current_session[-1].timestamp > session_window:
                    # Process previous session
                    if len(current_session) > 1:
                        resources = [e.resource for e in current_session]
                        for i, res1 in enumerate(resources):
                            for res2 in resources[i + 1 :]:
                                if res1 != res2:
                                    key = tuple(sorted([res1, res2]))
                                    resource_access_patterns[actor][key] += 1

                    current_session = [event]
                else:
                    current_session.append(event)

            # Process final session
            if len(current_session) > 1:
                resources = [e.resource for e in current_session]
                for i, res1 in enumerate(resources):
                    for res2 in resources[i + 1 :]:
                        if res1 != res2:
                            key = tuple(sorted([res1, res2]))
                            resource_access_patterns[actor][key] += 1

        # Find strong resource correlations
        strong_patterns = []
        for actor, patterns in resource_access_patterns.items():
            for resource_pair, count in patterns.items():
                if count >= 3:  # Threshold for significant pattern
                    strong_patterns.append({"actor": actor, "resources": list(resource_pair), "co_access_count": count})

        return {
            "total_patterns": sum(len(patterns) for patterns in resource_access_patterns.values()),
            "strong_patterns": strong_patterns,
        }

    async def _generate_forensic_recommendations(
        self, events: List[AuditEvent], analysis_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate forensic investigation recommendations"""
        recommendations = []

        # Check for high-risk patterns
        if analysis_result.get("pattern_analysis"):
            patterns = analysis_result["pattern_analysis"]

            if patterns.get("privilege_escalation"):
                recommendations.append(
                    {
                        "priority": "HIGH",
                        "category": "Access Control",
                        "recommendation": "Investigate privilege escalation patterns",
                        "details": f"Found {len(patterns['privilege_escalation'])} potential privilege escalation sequences",
                        "next_steps": [
                            "Review access control policies",
                            "Validate user permissions",
                            "Implement additional authorization checks",
                        ],
                    }
                )

            if patterns.get("data_exfiltration"):
                recommendations.append(
                    {
                        "priority": "CRITICAL",
                        "category": "Data Protection",
                        "recommendation": "Investigate potential data exfiltration",
                        "details": f"Detected {len(patterns['data_exfiltration'])} high-volume data access patterns",
                        "next_steps": [
                            "Review data access logs",
                            "Check for unauthorized data transfers",
                            "Implement data loss prevention measures",
                        ],
                    }
                )

        # Check for integrity issues
        summary = analysis_result.get("summary", {})
        if summary.get("integrity_status") == "compromised":
            recommendations.append(
                {
                    "priority": "CRITICAL",
                    "category": "Data Integrity",
                    "recommendation": "Address log integrity issues immediately",
                    "details": "Some audit events failed integrity verification",
                    "next_steps": [
                        "Identify compromised log entries",
                        "Investigate potential tampering",
                        "Strengthen log protection mechanisms",
                    ],
                }
            )

        # Check for anomalous activity levels
        if summary.get("total", 0) > 1000:  # High activity threshold
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "Monitoring",
                    "recommendation": "Review high activity period",
                    "details": f"Analyzed {summary['total']} events in the specified time range",
                    "next_steps": [
                        "Determine if activity level is expected",
                        "Investigate any unusual spikes",
                        "Adjust monitoring thresholds if necessary",
                    ],
                }
            )

        return recommendations


class AuditTrailSystem:
    """
    Main audit trail system coordinating all components
    """

    def __init__(self, node_id: str, betanet_anchor: Optional[BetanetAnchorService] = None):
        self.node_id = node_id
        self.betanet_anchor = betanet_anchor

        # Core components
        self.event_buffer = EventBuffer(max_size=50000)
        self.compliance_engine = ComplianceEngine()
        self.anomaly_detector = AnomalyDetectionEngine()
        self.forensic_engine = ForensicAnalysisEngine()

        # Storage and archival
        self.archived_events: List[AuditEvent] = []
        self.audit_statistics = {
            "events_logged": 0,
            "events_archived": 0,
            "compliance_violations": 0,
            "anomalies_detected": 0,
            "forensic_queries": 0,
        }

        # Configuration
        self.config = {
            "auto_archive_hours": 168,  # 1 week
            "anomaly_detection_interval": 300,  # 5 minutes
            "blockchain_anchoring": betanet_anchor is not None,
            "compliance_frameworks": [ComplianceFramework.SOC2, ComplianceFramework.ISO27001],
        }

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []

    async def initialize(self):
        """Initialize audit trail system"""
        logger.info(f"Initializing audit trail system for node {self.node_id}")

        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._anomaly_detection_loop()),
            asyncio.create_task(self._archival_loop()),
            asyncio.create_task(self._compliance_monitoring_loop()),
        ]

        # Initialize baseline metrics
        if self.archived_events:
            await self.anomaly_detector.update_baseline_metrics(self.archived_events)

        logger.info("Audit trail system initialized successfully")

    async def log_event(
        self,
        event_type: EventType,
        severity: EventSeverity,
        actor: str,
        action: str,
        resource: str,
        outcome: str,
        details: Optional[Dict[str, Any]] = None,
        target_node: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> str:
        """Log audit event"""
        event_id = f"{self.node_id}_{event_type.value}_{int(time.time() * 1000)}_{secrets.token_hex(4)}"

        event = AuditEvent(
            event_id=event_id,
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            source_node=self.node_id,
            target_node=target_node,
            actor=actor,
            action=action,
            resource=resource,
            outcome=outcome,
            details=details or {},
            correlation_id=correlation_id,
        )

        # Add to buffer
        await self.event_buffer.add_event(event)

        # Update statistics
        self.audit_statistics["events_logged"] += 1

        # Check compliance
        violations = await self.compliance_engine.check_compliance(event)
        if violations:
            self.audit_statistics["compliance_violations"] += len(violations)
            logger.warning(f"Compliance violations detected for event {event_id}: {len(violations)}")

        # Blockchain anchoring for critical events
        if self.config["blockchain_anchoring"] and event.severity in [EventSeverity.ERROR, EventSeverity.CRITICAL]:
            await self._anchor_event_to_blockchain(event)

        logger.debug(f"Logged audit event {event_id}: {actor} {action} {resource}")

        return event_id

    async def _anchor_event_to_blockchain(self, event: AuditEvent):
        """Anchor critical event to blockchain"""
        if not self.betanet_anchor:
            return

        try:
            # Create proof hash from event
            event_hash = event.checksum
            merkle_root = hashlib.sha256(f"{event_hash}_{event.timestamp}".encode()).hexdigest()

            # Anchor to blockchain
            anchor_id = await self.betanet_anchor.anchor_proof(event_hash, merkle_root, priority="high")

            # Update event metadata with anchor information
            event.metadata["blockchain_anchor"] = {
                "anchor_id": anchor_id,
                "anchored_at": time.time(),
                "proof_hash": event_hash,
            }

            logger.info(f"Anchored critical event {event.event_id} to blockchain: {anchor_id}")

        except Exception as e:
            logger.error(f"Failed to anchor event to blockchain: {e}")

    async def query_events(
        self,
        time_range: Optional[Tuple[float, float]] = None,
        event_types: Optional[List[EventType]] = None,
        severity: Optional[EventSeverity] = None,
        nodes: Optional[List[str]] = None,
        limit: int = 1000,
    ) -> List[AuditEvent]:
        """Query audit events with filters"""
        # Search in buffer
        buffer_events = await self.event_buffer.search_events(time_range, event_types, severity, nodes)

        # Search in archived events if needed
        archived_results = []
        if time_range and len(buffer_events) < limit:
            for event in self.archived_events:
                if time_range and not (time_range[0] <= event.timestamp <= time_range[1]):
                    continue
                if event_types and event.event_type not in event_types:
                    continue
                if severity and event.severity != severity:
                    continue
                if nodes and not (event.source_node in nodes or (event.target_node and event.target_node in nodes)):
                    continue

                archived_results.append(event)

        # Combine and sort results
        all_results = buffer_events + archived_results
        all_results.sort(key=lambda x: x.timestamp, reverse=True)

        return all_results[:limit]

    async def generate_compliance_report(
        self, framework: ComplianceFramework, time_range: Tuple[float, float]
    ) -> Dict[str, Any]:
        """Generate compliance report"""
        # Get all events in time range
        all_events = await self.query_events(time_range=time_range, limit=10000)

        # Generate report
        report = await self.compliance_engine.generate_compliance_report(framework, time_range, all_events)

        return report

    async def detect_anomalies(self, time_window_minutes: int = 60) -> List[AnomalyPattern]:
        """Detect anomalies in recent events"""
        recent_events = await self.event_buffer.get_recent_events(1000)
        anomalies = await self.anomaly_detector.detect_anomalies(recent_events, time_window_minutes)

        if anomalies:
            self.audit_statistics["anomalies_detected"] += len(anomalies)
            logger.info(f"Detected {len(anomalies)} anomalies in recent activity")

        return anomalies

    async def create_forensic_query(
        self,
        analyst: str,
        query_type: str,
        time_range: Tuple[float, float],
        filters: Optional[Dict[str, Any]] = None,
        nodes_of_interest: Optional[List[str]] = None,
        event_types: Optional[List[EventType]] = None,
    ) -> str:
        """Create forensic analysis query"""
        query_id = f"forensic_{analyst}_{int(time.time())}_{secrets.token_hex(4)}"

        query = ForensicQuery(
            query_id=query_id,
            analyst=analyst,
            query_type=query_type,
            time_range=time_range,
            filters=filters or {},
            nodes_of_interest=nodes_of_interest or [],
            event_types=event_types or [],
            correlation_analysis=True,
            pattern_detection=True,
        )

        await self.forensic_engine.create_forensic_query(query)
        self.audit_statistics["forensic_queries"] += 1

        return query_id

    async def execute_forensic_analysis(self, query_id: str) -> Dict[str, Any]:
        """Execute forensic analysis"""
        # Get all events for analysis
        all_events = await self.query_events(limit=50000)  # Large limit for comprehensive analysis

        # Execute analysis
        result = await self.forensic_engine.execute_forensic_analysis(query_id, all_events)

        logger.info(f"Executed forensic analysis {query_id}")

        return result

    async def _anomaly_detection_loop(self):
        """Background task for continuous anomaly detection"""
        while True:
            try:
                await asyncio.sleep(self.config["anomaly_detection_interval"])

                # Detect anomalies
                anomalies = await self.detect_anomalies()

                # Log detected anomalies as security alerts
                for anomaly in anomalies:
                    await self.log_event(
                        event_type=EventType.SECURITY_ALERT,
                        severity=anomaly.severity,
                        actor="anomaly_detector",
                        action="detected_anomaly",
                        resource="audit_trail",
                        outcome="alert_generated",
                        details={
                            "anomaly_type": anomaly.pattern_type,
                            "confidence": anomaly.confidence,
                            "affected_nodes": anomaly.affected_nodes,
                            "evidence_count": len(anomaly.evidence_events),
                        },
                    )

            except Exception as e:
                logger.error(f"Error in anomaly detection loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _archival_loop(self):
        """Background task for archiving old events"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check hourly

                cutoff_time = time.time() - (self.config["auto_archive_hours"] * 3600)

                # Get events to archive
                events_to_archive = await self.event_buffer.search_events(time_range=(0, cutoff_time))

                if events_to_archive:
                    # Move to archive
                    self.archived_events.extend(events_to_archive)
                    self.audit_statistics["events_archived"] += len(events_to_archive)

                    logger.info(f"Archived {len(events_to_archive)} old events")

            except Exception as e:
                logger.error(f"Error in archival loop: {e}")

    async def _compliance_monitoring_loop(self):
        """Background task for compliance monitoring"""
        while True:
            try:
                await asyncio.sleep(1800)  # Check every 30 minutes

                # Update baseline metrics for anomaly detection
                if len(self.archived_events) > 100:
                    await self.anomaly_detector.update_baseline_metrics(
                        self.archived_events[-1000:]  # Use recent archived events
                    )

            except Exception as e:
                logger.error(f"Error in compliance monitoring loop: {e}")

    async def shutdown(self):
        """Shutdown audit trail system"""
        logger.info("Shutting down audit trail system")

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)

        logger.info("Audit trail system shutdown complete")

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get audit system statistics"""
        return {
            "node_id": self.node_id,
            "statistics": self.audit_statistics,
            "buffer_status": {
                "current_size": len(self.event_buffer.buffer),
                "max_size": self.event_buffer.max_size,
                "indexed_events": len(self.event_buffer.event_index),
            },
            "archive_status": {
                "archived_events": len(self.archived_events),
                "auto_archive_hours": self.config["auto_archive_hours"],
            },
            "compliance_status": {
                "active_rules": len(self.compliance_engine.compliance_rules),
                "total_violations": len(self.compliance_engine.rule_violations),
                "frameworks": [f.value for f in self.config["compliance_frameworks"]],
            },
            "anomaly_detection": {
                "detected_anomalies": len(self.anomaly_detector.detected_anomalies),
                "baseline_metrics": len(self.anomaly_detector.baseline_metrics),
                "detection_rules": len(self.anomaly_detector.detection_rules),
            },
            "blockchain_anchoring": {
                "enabled": self.config["blockchain_anchoring"],
                "anchor_service": self.betanet_anchor is not None,
            },
        }


# Factory function for system creation
def create_audit_trail_system(
    node_id: str, betanet_anchor: Optional[BetanetAnchorService] = None, config: Optional[Dict[str, Any]] = None
) -> AuditTrailSystem:
    """
    Factory function to create audit trail system

    Args:
        node_id: Unique identifier for this audit node
        betanet_anchor: Optional BetaNet anchoring service
        config: Optional configuration overrides

    Returns:
        Configured audit trail system
    """
    system = AuditTrailSystem(node_id, betanet_anchor)

    if config:
        system.config.update(config)

    logger.info(f"Created audit trail system for node {node_id}")

    return system
