"""
Advanced Threat Detection System for AIVillage
==============================================

Comprehensive threat detection and mitigation system with machine learning-based pattern recognition.
Integrates with all security components for real-time threat monitoring and response.
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatCategory(Enum):
    """Threat categories."""

    NETWORK_ATTACK = "network_attack"
    AUTHENTICATION_ATTACK = "authentication_attack"
    CONSENSUS_ATTACK = "consensus_attack"
    DATA_PRIVACY_VIOLATION = "data_privacy_violation"
    RESOURCE_ABUSE = "resource_abuse"
    MALICIOUS_BEHAVIOR = "malicious_behavior"
    INSIDER_THREAT = "insider_threat"


class AttackVector(Enum):
    """Known attack vectors."""

    BRUTE_FORCE = "brute_force"
    CREDENTIAL_STUFFING = "credential_stuffing"
    MAN_IN_THE_MIDDLE = "man_in_the_middle"
    REPLAY_ATTACK = "replay_attack"
    INJECTION_ATTACK = "injection_attack"
    DENIAL_OF_SERVICE = "denial_of_service"
    SOCIAL_ENGINEERING = "social_engineering"
    ZERO_DAY_EXPLOIT = "zero_day_exploit"


@dataclass
class ThreatIndicator:
    """Indicator of potential threat."""

    indicator_id: str
    indicator_type: str
    source_node: str
    target_node: str
    metric_name: str
    metric_value: float
    baseline_value: float
    deviation_score: float
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatEvent:
    """Detected threat event."""

    event_id: str
    threat_category: ThreatCategory
    threat_level: ThreatLevel
    attack_vector: Optional[AttackVector]
    source_nodes: Set[str]
    target_nodes: Set[str]
    indicators: List[ThreatIndicator] = field(default_factory=list)
    confidence_score: float = 0.0
    risk_score: float = 0.0
    detected_at: float = field(default_factory=time.time)
    mitigated: bool = False
    mitigation_actions: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    false_positive: bool = False


@dataclass
class BehaviorProfile:
    """Node behavior profile for anomaly detection."""

    node_id: str
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    recent_metrics: deque = field(default_factory=lambda: deque(maxlen=100))
    anomaly_scores: deque = field(default_factory=lambda: deque(maxlen=50))
    last_updated: float = field(default_factory=time.time)
    profile_confidence: float = 0.0
    behavioral_patterns: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MitigationStrategy:
    """Threat mitigation strategy."""

    strategy_id: str
    threat_categories: Set[ThreatCategory]
    attack_vectors: Set[AttackVector]
    actions: List[Dict[str, Any]]
    effectiveness_score: float
    resource_cost: float
    execution_time: float
    dependencies: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)


class ThreatDetectionSystem:
    """
    Advanced threat detection system with machine learning capabilities.

    Features:
    - Real-time behavior analysis and anomaly detection
    - Pattern recognition for known attack vectors
    - Multi-layer threat correlation and scoring
    - Adaptive learning from security events
    - Automated mitigation strategy selection
    - Integration with all AIVillage security components
    """

    def __init__(self, node_id: str):
        """Initialize threat detection system."""
        self.node_id = node_id

        # Threat detection state
        self.behavior_profiles: Dict[str, BehaviorProfile] = {}
        self.detected_threats: List[ThreatEvent] = []
        self.threat_patterns: Dict[str, Dict[str, Any]] = {}
        self.mitigation_strategies: Dict[str, MitigationStrategy] = {}

        # Real-time monitoring
        self.metric_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.event_correlation_window = 300  # 5 minutes
        self.anomaly_thresholds: Dict[str, float] = {}

        # Machine learning models (simplified)
        self.anomaly_model_weights: Dict[str, np.ndarray] = {}
        self.pattern_recognition_models: Dict[str, Dict[str, Any]] = {}

        # Configuration
        self.detection_config = {
            "anomaly_detection_enabled": True,
            "pattern_recognition_enabled": True,
            "behavioral_analysis_enabled": True,
            "real_time_correlation_enabled": True,
            "adaptive_learning_enabled": True,
            "auto_mitigation_enabled": True,
            "sensitivity_level": 0.7,  # 0.0 (low) to 1.0 (high)
            "correlation_threshold": 0.8,
            "learning_rate": 0.01,
            "profile_update_interval": 60,  # seconds
            "threat_retention_days": 30,
        }

        # Statistics
        self.detection_stats = {
            "total_threats_detected": 0,
            "threats_by_category": defaultdict(int),
            "threats_by_level": defaultdict(int),
            "false_positives": 0,
            "true_positives": 0,
            "threats_mitigated": 0,
            "anomalies_detected": 0,
            "patterns_learned": 0,
            "behavior_profiles_created": 0,
            "mitigation_strategies_executed": 0,
        }

        # Initialize threat patterns and mitigation strategies
        self._initialize_threat_patterns()
        self._initialize_mitigation_strategies()

        logger.info(f"Threat Detection System initialized for node {node_id}")

    async def start_monitoring(self) -> None:
        """Start continuous threat monitoring."""
        logger.info("Starting threat detection monitoring")

        # Start background monitoring tasks
        monitoring_tasks = [
            self._anomaly_detection_loop(),
            self._pattern_recognition_loop(),
            self._behavioral_analysis_loop(),
            self._threat_correlation_loop(),
            self._adaptive_learning_loop(),
            self._profile_maintenance_loop(),
        ]

        await asyncio.gather(*monitoring_tasks)

    async def ingest_security_event(
        self,
        event_type: str,
        source_node: str,
        target_node: str,
        event_data: Dict[str, Any],
        timestamp: Optional[float] = None,
    ) -> None:
        """Ingest a security event for analysis."""

        timestamp = timestamp or time.time()

        try:
            # Store event in appropriate metric buffer
            metric_key = f"{event_type}:{source_node}:{target_node}"
            event_record = {
                "timestamp": timestamp,
                "source": source_node,
                "target": target_node,
                "data": event_data,
                "processed": False,
            }

            self.metric_buffers[metric_key].append(event_record)

            # Update behavior profiles
            await self._update_behavior_profile(source_node, event_type, event_data, timestamp)

            # Immediate threat detection for critical events
            if self._is_critical_event(event_type, event_data):
                await self._immediate_threat_analysis(event_record)

        except Exception as e:
            logger.error(f"Failed to ingest security event: {e}")

    async def detect_threats(self, analysis_window: Optional[float] = None) -> List[ThreatEvent]:
        """Run comprehensive threat detection analysis."""

        analysis_window = analysis_window or self.event_correlation_window
        current_time = time.time()
        cutoff_time = current_time - analysis_window

        detected_threats = []

        try:
            # Collect recent events for analysis
            recent_events = self._collect_recent_events(cutoff_time)

            if self.detection_config["anomaly_detection_enabled"]:
                anomaly_threats = await self._detect_anomalies(recent_events)
                detected_threats.extend(anomaly_threats)

            if self.detection_config["pattern_recognition_enabled"]:
                pattern_threats = await self._detect_known_patterns(recent_events)
                detected_threats.extend(pattern_threats)

            if self.detection_config["behavioral_analysis_enabled"]:
                behavioral_threats = await self._detect_behavioral_anomalies(recent_events)
                detected_threats.extend(behavioral_threats)

            # Correlate and deduplicate threats
            if self.detection_config["real_time_correlation_enabled"]:
                correlated_threats = await self._correlate_threats(detected_threats)
                detected_threats = correlated_threats

            # Calculate risk scores
            for threat in detected_threats:
                threat.risk_score = await self._calculate_risk_score(threat)

            # Store detected threats
            self.detected_threats.extend(detected_threats)
            self.detection_stats["total_threats_detected"] += len(detected_threats)

            # Update statistics
            for threat in detected_threats:
                self.detection_stats["threats_by_category"][threat.threat_category.value] += 1
                self.detection_stats["threats_by_level"][threat.threat_level.value] += 1

            if detected_threats:
                logger.warning(f"Detected {len(detected_threats)} potential threats")

            return detected_threats

        except Exception as e:
            logger.error(f"Threat detection failed: {e}")
            return []

    async def mitigate_threat(
        self, threat: ThreatEvent, strategy_preference: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """Mitigate a detected threat using appropriate strategies."""

        if threat.mitigated:
            return True, ["Threat already mitigated"]

        try:
            # Select appropriate mitigation strategy
            strategy = await self._select_mitigation_strategy(threat, strategy_preference)

            if not strategy:
                logger.error(f"No suitable mitigation strategy found for threat {threat.event_id}")
                return False, ["No suitable mitigation strategy available"]

            executed_actions = []
            success = True

            # Execute mitigation actions
            for action in strategy.actions:
                action_success = await self._execute_mitigation_action(threat, action)

                if action_success:
                    executed_actions.append(action["description"])
                else:
                    success = False
                    logger.error(f"Mitigation action failed: {action['description']}")

            # Update threat status
            if success:
                threat.mitigated = True
                threat.mitigation_actions = executed_actions
                self.detection_stats["threats_mitigated"] += 1
                self.detection_stats["mitigation_strategies_executed"] += 1

                logger.info(f"Successfully mitigated threat {threat.event_id}")

            return success, executed_actions

        except Exception as e:
            logger.error(f"Threat mitigation failed for {threat.event_id}: {e}")
            return False, [f"Mitigation error: {str(e)}"]

    async def learn_from_incident(self, threat: ThreatEvent, outcome: str, feedback: Dict[str, Any]) -> None:
        """Learn from incident outcomes to improve detection."""

        if not self.detection_config["adaptive_learning_enabled"]:
            return

        try:
            # Update false positive/true positive statistics
            if outcome == "false_positive":
                threat.false_positive = True
                self.detection_stats["false_positives"] += 1

                # Reduce sensitivity for similar patterns
                await self._adjust_detection_sensitivity(threat, -0.1)

            elif outcome == "true_positive":
                self.detection_stats["true_positives"] += 1

                # Increase sensitivity for similar patterns
                await self._adjust_detection_sensitivity(threat, 0.1)

                # Learn new threat patterns
                await self._learn_threat_pattern(threat, feedback)

            # Update behavior baselines if needed
            if "behavioral_adjustment" in feedback:
                await self._update_behavioral_baselines(threat, feedback["behavioral_adjustment"])

            self.detection_stats["patterns_learned"] += 1

            logger.info(f"Learned from incident {threat.event_id} with outcome: {outcome}")

        except Exception as e:
            logger.error(f"Incident learning failed: {e}")

    # Private detection methods

    async def _anomaly_detection_loop(self) -> None:
        """Continuous anomaly detection loop."""
        while True:
            try:
                if self.detection_config["anomaly_detection_enabled"]:
                    # Detect statistical anomalies in metrics
                    anomalies = await self._detect_statistical_anomalies()

                    # Convert anomalies to threat indicators
                    for anomaly in anomalies:
                        indicator = ThreatIndicator(
                            indicator_id=str(uuid.uuid4()),
                            indicator_type="statistical_anomaly",
                            source_node=anomaly["source_node"],
                            target_node=anomaly["target_node"],
                            metric_name=anomaly["metric"],
                            metric_value=anomaly["value"],
                            baseline_value=anomaly["baseline"],
                            deviation_score=anomaly["deviation"],
                            context=anomaly["context"],
                        )

                        # Check if anomaly indicates a threat
                        if anomaly["deviation"] > self.detection_config["sensitivity_level"]:
                            await self._process_threat_indicator(indicator)

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Anomaly detection loop error: {e}")
                await asyncio.sleep(30)

    async def _pattern_recognition_loop(self) -> None:
        """Continuous pattern recognition loop."""
        while True:
            try:
                if self.detection_config["pattern_recognition_enabled"]:
                    # Analyze recent events for known attack patterns
                    current_time = time.time()
                    recent_events = self._collect_recent_events(current_time - 60)  # Last minute

                    pattern_matches = await self._match_attack_patterns(recent_events)

                    for match in pattern_matches:
                        threat = await self._create_threat_from_pattern(match)
                        if threat:
                            await self._process_detected_threat(threat)

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Pattern recognition loop error: {e}")
                await asyncio.sleep(60)

    async def _behavioral_analysis_loop(self) -> None:
        """Continuous behavioral analysis loop."""
        while True:
            try:
                if self.detection_config["behavioral_analysis_enabled"]:
                    # Analyze behavior profiles for anomalies
                    for node_id, profile in self.behavior_profiles.items():
                        behavioral_anomalies = await self._analyze_behavior_profile(profile)

                        for anomaly in behavioral_anomalies:
                            threat = await self._create_threat_from_behavioral_anomaly(node_id, anomaly)
                            if threat:
                                await self._process_detected_threat(threat)

                await asyncio.sleep(self.detection_config["profile_update_interval"])

            except Exception as e:
                logger.error(f"Behavioral analysis loop error: {e}")
                await asyncio.sleep(120)

    async def _threat_correlation_loop(self) -> None:
        """Continuous threat correlation loop."""
        while True:
            try:
                if self.detection_config["real_time_correlation_enabled"]:
                    # Correlate recent threats
                    recent_threats = [
                        threat
                        for threat in self.detected_threats
                        if time.time() - threat.detected_at < self.event_correlation_window and not threat.mitigated
                    ]

                    correlations = await self._find_threat_correlations(recent_threats)

                    for correlation in correlations:
                        # Merge correlated threats or escalate severity
                        await self._process_threat_correlation(correlation)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Threat correlation loop error: {e}")
                await asyncio.sleep(120)

    async def _adaptive_learning_loop(self) -> None:
        """Continuous adaptive learning loop."""
        while True:
            try:
                if self.detection_config["adaptive_learning_enabled"]:
                    # Update detection models based on recent events
                    await self._update_anomaly_models()
                    await self._update_pattern_models()
                    await self._optimize_thresholds()

                await asyncio.sleep(300)  # Update every 5 minutes

            except Exception as e:
                logger.error(f"Adaptive learning loop error: {e}")
                await asyncio.sleep(600)

    async def _profile_maintenance_loop(self) -> None:
        """Continuous profile maintenance loop."""
        while True:
            try:
                current_time = time.time()

                # Clean up old behavior profiles
                for node_id, profile in list(self.behavior_profiles.items()):
                    if current_time - profile.last_updated > 86400:  # 24 hours
                        del self.behavior_profiles[node_id]

                # Clean up old threats
                retention_cutoff = current_time - (self.detection_config["threat_retention_days"] * 86400)
                self.detected_threats = [
                    threat for threat in self.detected_threats if threat.detected_at > retention_cutoff
                ]

                # Clean up old metric buffers
                for metric_key, buffer in self.metric_buffers.items():
                    # Remove events older than 1 hour
                    cutoff_time = current_time - 3600
                    while buffer and buffer[0]["timestamp"] < cutoff_time:
                        buffer.popleft()

                await asyncio.sleep(3600)  # Maintenance every hour

            except Exception as e:
                logger.error(f"Profile maintenance loop error: {e}")
                await asyncio.sleep(1800)

    def _collect_recent_events(self, cutoff_time: float) -> List[Dict[str, Any]]:
        """Collect recent events from metric buffers."""
        recent_events = []

        for metric_key, buffer in self.metric_buffers.items():
            for event in buffer:
                if event["timestamp"] >= cutoff_time:
                    recent_events.append({"metric_key": metric_key, **event})

        return recent_events

    async def _detect_statistical_anomalies(self) -> List[Dict[str, Any]]:
        """Detect statistical anomalies in metrics."""
        anomalies = []

        for metric_key, buffer in self.metric_buffers.items():
            if len(buffer) < 10:  # Need minimum data points
                continue

            try:
                # Extract numeric values from recent events
                values = []
                for event in buffer:
                    if isinstance(event["data"], dict):
                        # Extract numeric metrics from event data
                        for key, value in event["data"].items():
                            if isinstance(value, (int, float)):
                                values.append(value)
                    elif isinstance(event["data"], (int, float)):
                        values.append(event["data"])

                if len(values) < 5:
                    continue

                # Calculate statistical measures
                mean = np.mean(values)
                std = np.std(values)
                recent_values = values[-5:]  # Last 5 values

                for i, value in enumerate(recent_values):
                    if std > 0:
                        z_score = abs((value - mean) / std)

                        if z_score > 2.5:  # Significant deviation
                            parts = metric_key.split(":")
                            anomalies.append(
                                {
                                    "source_node": parts[1] if len(parts) > 1 else "unknown",
                                    "target_node": parts[2] if len(parts) > 2 else "unknown",
                                    "metric": parts[0] if len(parts) > 0 else metric_key,
                                    "value": value,
                                    "baseline": mean,
                                    "deviation": z_score,
                                    "context": {
                                        "std_dev": std,
                                        "sample_size": len(values),
                                        "recent_values": recent_values,
                                    },
                                }
                            )

            except Exception as e:
                logger.error(f"Statistical anomaly detection failed for {metric_key}: {e}")

        return anomalies

    async def _match_attack_patterns(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Match events against known attack patterns."""
        pattern_matches = []

        for pattern_name, pattern_config in self.threat_patterns.items():
            try:
                match_score = await self._calculate_pattern_match_score(events, pattern_config)

                if match_score > self.detection_config["correlation_threshold"]:
                    pattern_matches.append(
                        {
                            "pattern_name": pattern_name,
                            "pattern_config": pattern_config,
                            "match_score": match_score,
                            "matching_events": events,
                            "threat_category": pattern_config.get("category", ThreatCategory.MALICIOUS_BEHAVIOR),
                            "attack_vector": pattern_config.get("attack_vector", AttackVector.ZERO_DAY_EXPLOIT),
                        }
                    )

            except Exception as e:
                logger.error(f"Pattern matching failed for {pattern_name}: {e}")

        return pattern_matches

    async def _calculate_pattern_match_score(
        self, events: List[Dict[str, Any]], pattern_config: Dict[str, Any]
    ) -> float:
        """Calculate how well events match a threat pattern."""

        if not events or not pattern_config.get("rules"):
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        for rule in pattern_config["rules"]:
            rule_score = 0.0
            rule_weight = rule.get("weight", 1.0)

            matching_events = [event for event in events if self._event_matches_rule(event, rule)]

            if matching_events:
                # Calculate rule-specific score
                if rule["type"] == "frequency":
                    expected_count = rule["threshold"]
                    actual_count = len(matching_events)
                    rule_score = min(1.0, actual_count / expected_count)

                elif rule["type"] == "sequence":
                    rule_score = self._calculate_sequence_score(matching_events, rule)

                elif rule["type"] == "timing":
                    rule_score = self._calculate_timing_score(matching_events, rule)

                elif rule["type"] == "correlation":
                    rule_score = self._calculate_correlation_score(matching_events, rule)

            total_score += rule_score * rule_weight
            total_weight += rule_weight

        return total_score / max(1.0, total_weight)

    def _event_matches_rule(self, event: Dict[str, Any], rule: Dict[str, Any]) -> bool:
        """Check if an event matches a rule."""
        conditions = rule.get("conditions", {})

        for field, expected_value in conditions.items():
            actual_value = event.get(field) or event.get("data", {}).get(field)

            if isinstance(expected_value, dict):
                # Complex condition (e.g., range, regex)
                if "min" in expected_value and "max" in expected_value:
                    if not (expected_value["min"] <= actual_value <= expected_value["max"]):
                        return False
                elif "contains" in expected_value:
                    if expected_value["contains"] not in str(actual_value):
                        return False
            else:
                # Simple equality check
                if actual_value != expected_value:
                    return False

        return True

    def _calculate_sequence_score(self, events: List[Dict[str, Any]], rule: Dict[str, Any]) -> float:
        """Calculate score for sequential pattern matching."""
        expected_sequence = rule.get("sequence", [])

        if len(events) < len(expected_sequence):
            return 0.0

        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda x: x["timestamp"])

        # Check for expected sequence
        sequence_matches = 0
        for i in range(len(sorted_events) - len(expected_sequence) + 1):
            window = sorted_events[i : i + len(expected_sequence)]
            if self._events_match_sequence(window, expected_sequence):
                sequence_matches += 1

        return min(1.0, sequence_matches / max(1, len(expected_sequence)))

    def _calculate_timing_score(self, events: List[Dict[str, Any]], rule: Dict[str, Any]) -> float:
        """Calculate score for timing pattern matching."""
        if len(events) < 2:
            return 0.0

        timestamps = [event["timestamp"] for event in events]
        timestamps.sort()

        intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]

        expected_interval = rule.get("interval", 1.0)
        tolerance = rule.get("tolerance", 0.5)

        matching_intervals = [interval for interval in intervals if abs(interval - expected_interval) <= tolerance]

        return len(matching_intervals) / len(intervals)

    def _calculate_correlation_score(self, events: List[Dict[str, Any]], rule: Dict[str, Any]) -> float:
        """Calculate score for event correlation."""
        correlation_fields = rule.get("fields", [])

        if len(correlation_fields) < 2 or len(events) < 2:
            return 0.0

        # Extract values for correlation analysis
        field_values = {field: [] for field in correlation_fields}

        for event in events:
            for field in correlation_fields:
                value = event.get(field) or event.get("data", {}).get(field)
                if value is not None and isinstance(value, (int, float)):
                    field_values[field].append(value)

        # Calculate correlation between fields
        correlations = []
        field_names = list(field_values.keys())

        for i in range(len(field_names)):
            for j in range(i + 1, len(field_names)):
                field1_values = field_values[field_names[i]]
                field2_values = field_values[field_names[j]]

                if len(field1_values) > 1 and len(field2_values) > 1:
                    # Calculate Pearson correlation coefficient
                    min_len = min(len(field1_values), len(field2_values))
                    field1_sample = field1_values[:min_len]
                    field2_sample = field2_values[:min_len]

                    correlation = np.corrcoef(field1_sample, field2_sample)[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(abs(correlation))

        return np.mean(correlations) if correlations else 0.0

    def _events_match_sequence(self, events: List[Dict[str, Any]], sequence: List[str]) -> bool:
        """Check if events match expected sequence."""
        if len(events) != len(sequence):
            return False

        for event, expected_type in zip(events, sequence):
            event_type = event.get("metric_key", "").split(":")[0]
            if event_type != expected_type:
                return False

        return True

    async def _update_behavior_profile(
        self, node_id: str, event_type: str, event_data: Dict[str, Any], timestamp: float
    ) -> None:
        """Update behavior profile for a node."""

        if node_id not in self.behavior_profiles:
            self.behavior_profiles[node_id] = BehaviorProfile(node_id=node_id)
            self.detection_stats["behavior_profiles_created"] += 1

        profile = self.behavior_profiles[node_id]

        # Extract metrics from event data
        metrics = {}
        if isinstance(event_data, dict):
            for key, value in event_data.items():
                if isinstance(value, (int, float)):
                    metrics[key] = value

        # Update recent metrics
        profile.recent_metrics.append({"timestamp": timestamp, "event_type": event_type, "metrics": metrics})

        # Update baseline metrics (exponential moving average)
        alpha = 0.1  # Learning rate
        for metric_name, value in metrics.items():
            if metric_name in profile.baseline_metrics:
                profile.baseline_metrics[metric_name] = (1 - alpha) * profile.baseline_metrics[
                    metric_name
                ] + alpha * value
            else:
                profile.baseline_metrics[metric_name] = value

        profile.last_updated = timestamp

        # Update profile confidence
        profile.profile_confidence = min(1.0, len(profile.recent_metrics) / 50.0)

    def _is_critical_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """Check if an event is critical and requires immediate analysis."""
        critical_event_types = {
            "authentication_failure",
            "authorization_violation",
            "consensus_attack",
            "byzantine_behavior",
            "security_violation",
        }

        return event_type in critical_event_types

    async def _immediate_threat_analysis(self, event: Dict[str, Any]) -> None:
        """Perform immediate threat analysis for critical events."""
        # Create threat indicator
        indicator = ThreatIndicator(
            indicator_id=str(uuid.uuid4()),
            indicator_type="critical_event",
            source_node=event["source"],
            target_node=event["target"],
            metric_name="critical_event_detected",
            metric_value=1.0,
            baseline_value=0.0,
            deviation_score=1.0,
            context=event["data"],
        )

        # Process immediately
        await self._process_threat_indicator(indicator)

    async def _process_threat_indicator(self, indicator: ThreatIndicator) -> None:
        """Process a threat indicator and potentially create a threat event."""
        # Determine threat category and level based on indicator
        threat_category, threat_level = self._classify_threat(indicator)

        if threat_level != ThreatLevel.LOW or indicator.deviation_score > 0.8:
            # Create threat event
            threat = ThreatEvent(
                event_id=str(uuid.uuid4()),
                threat_category=threat_category,
                threat_level=threat_level,
                source_nodes={indicator.source_node},
                target_nodes={indicator.target_node},
                indicators=[indicator],
                confidence_score=min(1.0, indicator.deviation_score),
                evidence={"indicators": [indicator.__dict__]},
            )

            await self._process_detected_threat(threat)

    def _classify_threat(self, indicator: ThreatIndicator) -> Tuple[ThreatCategory, ThreatLevel]:
        """Classify threat based on indicator characteristics."""

        # Simple classification logic - in production would use ML models
        if "authentication" in indicator.metric_name:
            category = ThreatCategory.AUTHENTICATION_ATTACK
        elif "consensus" in indicator.metric_name:
            category = ThreatCategory.CONSENSUS_ATTACK
        elif "privacy" in indicator.metric_name:
            category = ThreatCategory.DATA_PRIVACY_VIOLATION
        elif "resource" in indicator.metric_name:
            category = ThreatCategory.RESOURCE_ABUSE
        else:
            category = ThreatCategory.MALICIOUS_BEHAVIOR

        # Determine threat level based on deviation score
        if indicator.deviation_score >= 0.9:
            level = ThreatLevel.CRITICAL
        elif indicator.deviation_score >= 0.7:
            level = ThreatLevel.HIGH
        elif indicator.deviation_score >= 0.4:
            level = ThreatLevel.MEDIUM
        else:
            level = ThreatLevel.LOW

        return category, level

    async def _process_detected_threat(self, threat: ThreatEvent) -> None:
        """Process a detected threat event."""
        self.detected_threats.append(threat)

        logger.warning(
            f"Threat detected: {threat.threat_category.value} "
            f"(Level: {threat.threat_level.value}, "
            f"Confidence: {threat.confidence_score:.2f})"
        )

        # Auto-mitigate if enabled and threat is severe
        if self.detection_config["auto_mitigation_enabled"] and threat.threat_level in [
            ThreatLevel.HIGH,
            ThreatLevel.CRITICAL,
        ]:

            await self.mitigate_threat(threat)

    # Initialize threat patterns and strategies

    def _initialize_threat_patterns(self) -> None:
        """Initialize known threat patterns."""

        self.threat_patterns = {
            "brute_force_attack": {
                "category": ThreatCategory.AUTHENTICATION_ATTACK,
                "attack_vector": AttackVector.BRUTE_FORCE,
                "description": "Multiple failed authentication attempts",
                "rules": [
                    {
                        "type": "frequency",
                        "conditions": {"event_type": "authentication_failure"},
                        "threshold": 10,
                        "time_window": 300,  # 5 minutes
                        "weight": 1.0,
                    }
                ],
            },
            "byzantine_coordination": {
                "category": ThreatCategory.CONSENSUS_ATTACK,
                "attack_vector": AttackVector.DENIAL_OF_SERVICE,
                "description": "Coordinated Byzantine behavior",
                "rules": [
                    {
                        "type": "correlation",
                        "fields": ["vote_timing", "message_content"],
                        "threshold": 0.9,
                        "weight": 1.0,
                    },
                    {
                        "type": "frequency",
                        "conditions": {"event_type": "contradictory_message"},
                        "threshold": 3,
                        "time_window": 60,
                        "weight": 0.8,
                    },
                ],
            },
            "data_exfiltration": {
                "category": ThreatCategory.DATA_PRIVACY_VIOLATION,
                "description": "Unusual data access patterns",
                "rules": [
                    {
                        "type": "frequency",
                        "conditions": {"event_type": "data_access"},
                        "threshold": 50,
                        "time_window": 3600,  # 1 hour
                        "weight": 0.7,
                    },
                    {"type": "timing", "interval": 60, "tolerance": 10, "weight": 0.5},  # Every minute
                ],
            },
            "resource_exhaustion": {
                "category": ThreatCategory.RESOURCE_ABUSE,
                "attack_vector": AttackVector.DENIAL_OF_SERVICE,
                "description": "Resource exhaustion attack",
                "rules": [
                    {
                        "type": "frequency",
                        "conditions": {"event_type": "resource_request"},
                        "threshold": 100,
                        "time_window": 300,
                        "weight": 1.0,
                    }
                ],
            },
        }

    def _initialize_mitigation_strategies(self) -> None:
        """Initialize threat mitigation strategies."""

        self.mitigation_strategies = {
            "block_malicious_node": MitigationStrategy(
                strategy_id="block_malicious_node",
                threat_categories={ThreatCategory.AUTHENTICATION_ATTACK, ThreatCategory.MALICIOUS_BEHAVIOR},
                attack_vectors={AttackVector.BRUTE_FORCE, AttackVector.CREDENTIAL_STUFFING},
                actions=[
                    {
                        "type": "block_node",
                        "description": "Block malicious node from network",
                        "parameters": {"block_duration": 3600},  # 1 hour
                    }
                ],
                effectiveness_score=0.9,
                resource_cost=0.1,
                execution_time=1.0,
            ),
            "rate_limit_requests": MitigationStrategy(
                strategy_id="rate_limit_requests",
                threat_categories={ThreatCategory.RESOURCE_ABUSE, ThreatCategory.NETWORK_ATTACK},
                attack_vectors={AttackVector.DENIAL_OF_SERVICE},
                actions=[
                    {
                        "type": "apply_rate_limit",
                        "description": "Apply rate limiting to source nodes",
                        "parameters": {"max_requests_per_minute": 10},
                    }
                ],
                effectiveness_score=0.7,
                resource_cost=0.2,
                execution_time=0.5,
            ),
            "isolate_byzantine_nodes": MitigationStrategy(
                strategy_id="isolate_byzantine_nodes",
                threat_categories={ThreatCategory.CONSENSUS_ATTACK},
                attack_vectors={AttackVector.DENIAL_OF_SERVICE},
                actions=[
                    {
                        "type": "isolate_nodes",
                        "description": "Isolate Byzantine nodes from consensus",
                        "parameters": {"isolation_duration": 7200},  # 2 hours
                    }
                ],
                effectiveness_score=0.95,
                resource_cost=0.3,
                execution_time=2.0,
            ),
            "enhance_monitoring": MitigationStrategy(
                strategy_id="enhance_monitoring",
                threat_categories={ThreatCategory.DATA_PRIVACY_VIOLATION, ThreatCategory.INSIDER_THREAT},
                attack_vectors={AttackVector.ZERO_DAY_EXPLOIT},
                actions=[
                    {
                        "type": "increase_monitoring",
                        "description": "Enhance monitoring for suspicious activities",
                        "parameters": {"monitoring_level": "high"},
                    }
                ],
                effectiveness_score=0.6,
                resource_cost=0.4,
                execution_time=0.1,
            ),
        }

    async def _select_mitigation_strategy(
        self, threat: ThreatEvent, preference: Optional[str] = None
    ) -> Optional[MitigationStrategy]:
        """Select appropriate mitigation strategy for a threat."""

        if preference and preference in self.mitigation_strategies:
            strategy = self.mitigation_strategies[preference]
            if threat.threat_category in strategy.threat_categories:
                return strategy

        # Find matching strategies
        suitable_strategies = []

        for strategy_id, strategy in self.mitigation_strategies.items():
            if threat.threat_category in strategy.threat_categories:
                # Check attack vector match if available
                if threat.attack_vector and threat.attack_vector in strategy.attack_vectors:
                    suitable_strategies.append((strategy, strategy.effectiveness_score + 0.1))
                else:
                    suitable_strategies.append((strategy, strategy.effectiveness_score))

        if not suitable_strategies:
            return None

        # Select strategy with highest effectiveness score
        suitable_strategies.sort(key=lambda x: x[1], reverse=True)
        return suitable_strategies[0][0]

    async def _execute_mitigation_action(self, threat: ThreatEvent, action: Dict[str, Any]) -> bool:
        """Execute a specific mitigation action."""

        action_type = action["type"]

        try:
            if action_type == "block_node":
                # Block malicious nodes
                for node_id in threat.source_nodes:
                    logger.info(f"Blocking node {node_id} for {action['parameters']['block_duration']} seconds")
                    # Implementation would integrate with network layer
                return True

            elif action_type == "apply_rate_limit":
                # Apply rate limiting
                max_requests = action["parameters"]["max_requests_per_minute"]
                for node_id in threat.source_nodes:
                    logger.info(f"Applying rate limit of {max_requests}/min to node {node_id}")
                    # Implementation would integrate with request handling
                return True

            elif action_type == "isolate_nodes":
                # Isolate nodes from consensus
                for node_id in threat.source_nodes:
                    logger.info(f"Isolating node {node_id} from consensus")
                    # Implementation would integrate with consensus layer
                return True

            elif action_type == "increase_monitoring":
                # Enhance monitoring
                monitoring_level = action["parameters"]["monitoring_level"]
                logger.info(f"Enhancing monitoring level to {monitoring_level}")
                # Implementation would adjust monitoring parameters
                return True

            else:
                logger.warning(f"Unknown mitigation action type: {action_type}")
                return False

        except Exception as e:
            logger.error(f"Mitigation action execution failed: {e}")
            return False

    # Statistics and monitoring methods

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get threat detection statistics."""

        total_threats = self.detection_stats["total_threats_detected"]
        accuracy = 0.0

        if total_threats > 0:
            true_positives = self.detection_stats["true_positives"]
            false_positives = self.detection_stats["false_positives"]
            accuracy = (
                true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            )

        return {
            **dict(self.detection_stats),
            "detection_accuracy": accuracy,
            "active_behavior_profiles": len(self.behavior_profiles),
            "threat_patterns_loaded": len(self.threat_patterns),
            "mitigation_strategies_available": len(self.mitigation_strategies),
            "recent_threats": len(
                [threat for threat in self.detected_threats if time.time() - threat.detected_at < 3600]  # Last hour
            ),
        }

    def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary of detected threats."""

        current_time = time.time()

        # Recent threats (last 24 hours)
        recent_threats = [threat for threat in self.detected_threats if current_time - threat.detected_at < 86400]

        # Active threats (not mitigated)
        active_threats = [threat for threat in recent_threats if not threat.mitigated]

        # High severity threats
        critical_threats = [threat for threat in active_threats if threat.threat_level == ThreatLevel.CRITICAL]

        high_threats = [threat for threat in active_threats if threat.threat_level == ThreatLevel.HIGH]

        return {
            "total_recent_threats": len(recent_threats),
            "active_threats": len(active_threats),
            "critical_threats": len(critical_threats),
            "high_severity_threats": len(high_threats),
            "mitigated_threats": len([t for t in recent_threats if t.mitigated]),
            "false_positives": len([t for t in recent_threats if t.false_positive]),
            "threat_categories": dict(self.detection_stats["threats_by_category"]),
            "threat_levels": dict(self.detection_stats["threats_by_level"]),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""

        issues = []
        warnings = []

        # Check detection system status
        if not self.detection_config["anomaly_detection_enabled"]:
            warnings.append("Anomaly detection is disabled")

        if not self.detection_config["pattern_recognition_enabled"]:
            warnings.append("Pattern recognition is disabled")

        # Check for recent critical threats
        current_time = time.time()
        unmitigated_critical = [
            threat
            for threat in self.detected_threats
            if (
                threat.threat_level == ThreatLevel.CRITICAL
                and not threat.mitigated
                and current_time - threat.detected_at < 3600
            )
        ]

        if unmitigated_critical:
            issues.append(f"{len(unmitigated_critical)} unmitigated critical threats")

        # Check behavior profile health
        stale_profiles = [
            profile
            for profile in self.behavior_profiles.values()
            if current_time - profile.last_updated > 7200  # 2 hours
        ]

        if len(stale_profiles) > len(self.behavior_profiles) * 0.5:
            warnings.append("Many behavior profiles are stale")

        # Check detection accuracy
        stats = self.get_detection_stats()
        if stats["detection_accuracy"] < 0.7:
            warnings.append("Low detection accuracy")

        return {
            "healthy": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "detection_stats": stats,
            "threat_summary": self.get_threat_summary(),
        }
