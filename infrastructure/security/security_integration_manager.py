"""
Security Integration Manager for AIVillage
==========================================

Integrates all security components with existing federated learning infrastructure.
Provides unified security layer for the entire AIVillage ecosystem.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .federated_auth_system import FederatedAuthenticationSystem, NodeRole
from .secure_aggregation import SecureAggregationProtocol, AggregationMethod, PrivacyLevel
from .betanet_security_manager import BetaNetSecurityManager, SecurityLevel, ChannelType
from .consensus_security_manager import ConsensusSecurityManager, ConsensusProtocol
from .threat_detection_system import ThreatDetectionSystem
from .reputation_trust_system import ReputationTrustSystem, TrustLevel

logger = logging.getLogger(__name__)


class SecurityEvent(Enum):
    """Security event types."""

    AUTHENTICATION_SUCCESS = "authentication_success"
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_GRANTED = "authorization_granted"
    AUTHORIZATION_DENIED = "authorization_denied"
    SECURE_CHANNEL_ESTABLISHED = "secure_channel_established"
    SECURE_AGGREGATION_COMPLETED = "secure_aggregation_completed"
    CONSENSUS_ROUND_COMPLETED = "consensus_round_completed"
    THREAT_DETECTED = "threat_detected"
    THREAT_MITIGATED = "threat_mitigated"
    TRUST_UPDATED = "trust_updated"
    BYZANTINE_BEHAVIOR_DETECTED = "byzantine_behavior_detected"


@dataclass
class SecurityConfiguration:
    """Unified security configuration."""

    # Authentication settings
    enable_mfa: bool = True
    password_policy_enabled: bool = True
    session_timeout: int = 3600  # 1 hour

    # Aggregation security
    default_aggregation_method: AggregationMethod = AggregationMethod.HOMOMORPHIC
    privacy_level: PrivacyLevel = PrivacyLevel.HIGH
    differential_privacy_epsilon: float = 1.0

    # Transport security
    transport_security_level: SecurityLevel = SecurityLevel.HIGH
    enable_forward_secrecy: bool = True
    channel_rotation_interval: int = 3600  # 1 hour

    # Consensus security
    consensus_protocol: ConsensusProtocol = ConsensusProtocol.BYZANTINE
    byzantine_threshold: float = 0.33
    enable_threshold_signatures: bool = True

    # Threat detection
    threat_detection_enabled: bool = True
    auto_mitigation_enabled: bool = True
    detection_sensitivity: float = 0.7

    # Trust management
    initial_trust_score: float = 0.5
    trust_decay_enabled: bool = True
    min_trust_for_participation: float = 0.4


@dataclass
class SecurityMetrics:
    """Security system metrics."""

    # Authentication metrics
    total_authentications: int = 0
    failed_authentications: int = 0
    active_sessions: int = 0

    # Aggregation metrics
    secure_aggregations: int = 0
    privacy_violations_prevented: int = 0

    # Transport metrics
    secure_channels_established: int = 0
    threats_detected: int = 0

    # Consensus metrics
    consensus_rounds_secured: int = 0
    byzantine_attacks_prevented: int = 0

    # Trust metrics
    trust_updates: int = 0
    nodes_blacklisted: int = 0

    # Overall metrics
    security_incidents: int = 0
    incidents_resolved: int = 0
    uptime: float = 0.0


class SecurityIntegrationManager:
    """
    Unified security integration manager for AIVillage.

    Integrates and coordinates all security components:
    - Federated Authentication System
    - Secure Aggregation Protocol
    - BetaNet Security Manager
    - Consensus Security Manager
    - Threat Detection System
    - Reputation Trust System
    """

    def __init__(self, node_id: str, config: Optional[SecurityConfiguration] = None):
        """Initialize security integration manager."""
        self.node_id = node_id
        self.config = config or SecurityConfiguration()

        # Initialize security components
        self.auth_system = FederatedAuthenticationSystem(enable_mfa=self.config.enable_mfa)

        self.secure_aggregation = SecureAggregationProtocol(
            default_method=self.config.default_aggregation_method, privacy_level=self.config.privacy_level
        )

        self.transport_security = BetaNetSecurityManager(node_id)

        self.consensus_security = ConsensusSecurityManager(
            node_id, self.config.consensus_protocol, self.config.byzantine_threshold
        )

        self.threat_detection = ThreatDetectionSystem(node_id)

        self.trust_system = ReputationTrustSystem(node_id)

        # Integration state
        self.security_metrics = SecurityMetrics()
        self.active_integrations: Set[str] = set()
        self.security_event_handlers: Dict[SecurityEvent, List[callable]] = {}
        self.component_health: Dict[str, bool] = {}

        # Background tasks
        self.monitoring_tasks: List[asyncio.Task] = []
        self.started = False

        logger.info(f"Security Integration Manager initialized for node {node_id}")

    async def initialize(self) -> bool:
        """Initialize all security components."""

        try:
            logger.info("Initializing security components...")

            # Initialize authentication system
            auth_success = await self.auth_system.initialize()  # We need to add this method
            self.component_health["authentication"] = auth_success

            # Initialize transport security
            transport_success = await self.transport_security.initialize()
            self.component_health["transport"] = transport_success

            # Initialize consensus security
            consensus_success = True  # Already initialized in constructor
            self.component_health["consensus"] = consensus_success

            # Initialize threat detection
            # threat_detection already initialized
            self.component_health["threat_detection"] = True

            # Initialize trust system
            # trust_system already initialized
            self.component_health["trust"] = True

            # Initialize secure aggregation
            # secure_aggregation already initialized
            self.component_health["aggregation"] = True

            # Set up event handlers
            await self._setup_event_handlers()

            # Start monitoring tasks
            await self._start_monitoring()

            self.started = True

            all_healthy = all(self.component_health.values())
            if all_healthy:
                logger.info("All security components initialized successfully")
            else:
                failed_components = [k for k, v in self.component_health.items() if not v]
                logger.warning(f"Some components failed to initialize: {failed_components}")

            return all_healthy

        except Exception as e:
            logger.error(f"Security initialization failed: {e}")
            return False

    async def register_federated_node(
        self,
        node_id: str,
        role: NodeRole,
        password: str,
        capabilities: Optional[Dict[str, Any]] = None,
        initial_trust: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Register a new federated learning node with comprehensive security setup."""

        try:
            # Register with authentication system
            identity = await self.auth_system.register_node(node_id, role, password, capabilities)

            # Initialize trust profile
            await self.trust_system.initialize_node_trust(node_id, bootstrap_method="neutral")

            # Set up secure aggregation for the node
            await self.secure_aggregation.setup_participant(node_id, capabilities or {}, {"epsilon_limit": 10.0})

            # Emit security event
            await self._emit_security_event(
                SecurityEvent.AUTHENTICATION_SUCCESS, {"node_id": node_id, "role": role.value}
            )

            self.security_metrics.total_authentications += 1

            logger.info(f"Successfully registered federated node {node_id}")
            return True, identity.metadata.get("totp_secret") if identity else None

        except Exception as e:
            logger.error(f"Failed to register federated node {node_id}: {e}")

            await self._emit_security_event(SecurityEvent.AUTHENTICATION_FAILURE, {"node_id": node_id, "error": str(e)})

            self.security_metrics.failed_authentications += 1
            return False, None

    async def authenticate_federated_participant(
        self, node_id: str, password: str, mfa_token: Optional[str] = None, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Authenticate a federated learning participant."""

        try:
            # Check trust level first
            trust_decision, trust_score, trust_reason = await self.trust_system.get_trust_decision(
                node_id, TrustLevel.LOW, context
            )

            if not trust_decision:
                logger.warning(f"Authentication rejected for {node_id}: {trust_reason}")

                await self._emit_security_event(
                    SecurityEvent.AUTHORIZATION_DENIED, {"node_id": node_id, "reason": trust_reason}
                )

                return False, None, {"reason": trust_reason, "trust_score": trust_score}

            # Perform authentication
            auth_success, session = await self.auth_system.authenticate_node(node_id, password, mfa_token)

            if auth_success and session:
                # Record successful interaction
                await self.trust_system.record_interaction(node_id, "authentication", "success")

                # Emit security event
                await self._emit_security_event(
                    SecurityEvent.AUTHENTICATION_SUCCESS, {"node_id": node_id, "session_id": session.session_id}
                )

                self.security_metrics.total_authentications += 1
                self.security_metrics.active_sessions += 1

                return True, session.session_id, {"trust_score": trust_score}

            else:
                # Record failed interaction
                await self.trust_system.record_interaction(node_id, "authentication", "failure")

                await self._emit_security_event(
                    SecurityEvent.AUTHENTICATION_FAILURE, {"node_id": node_id, "reason": "invalid_credentials"}
                )

                self.security_metrics.failed_authentications += 1

                return False, None, {"reason": "authentication_failed"}

        except Exception as e:
            logger.error(f"Authentication failed for {node_id}: {e}")

            await self._emit_security_event(SecurityEvent.AUTHENTICATION_FAILURE, {"node_id": node_id, "error": str(e)})

            return False, None, {"reason": "system_error", "error": str(e)}

    async def secure_federated_aggregation(
        self,
        aggregation_id: str,
        participant_gradients: List[Dict[str, Any]],
        aggregation_method: Optional[AggregationMethod] = None,
        privacy_level: Optional[PrivacyLevel] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]], Dict[str, Any]]:
        """Perform secure federated learning aggregation."""

        try:
            # Validate participants
            validated_participants = []

            for gradient_data in participant_gradients:
                participant_id = gradient_data["participant_id"]

                # Check authentication
                session_valid, session = await self.auth_system.validate_session(gradient_data.get("session_id", ""))

                if not session_valid:
                    logger.warning(f"Invalid session for participant {participant_id}")
                    continue

                # Check trust level
                trust_decision, _, _ = await self.trust_system.get_trust_decision(participant_id, TrustLevel.MEDIUM)

                if not trust_decision:
                    logger.warning(f"Insufficient trust for participant {participant_id}")
                    continue

                validated_participants.append(gradient_data)

            if len(validated_participants) < 2:
                return False, None, {"error": "Insufficient validated participants"}

            # Convert to secure gradients
            secure_gradients = []
            for gradient_data in validated_participants:
                secure_gradient = await self.secure_aggregation.create_secure_gradient(
                    gradient_data["participant_id"], gradient_data["gradients"], gradient_data.get("privacy_params")
                )
                secure_gradients.append(secure_gradient)

            # Perform secure aggregation
            success, aggregated_result, metadata = await self.secure_aggregation.secure_aggregate(
                aggregation_id, secure_gradients, aggregation_method, privacy_level
            )

            if success:
                # Update trust for successful participants
                for gradient_data in validated_participants:
                    await self.trust_system.record_interaction(
                        gradient_data["participant_id"],
                        "federated_aggregation",
                        "success",
                        {"aggregation_id": aggregation_id},
                    )

                await self._emit_security_event(
                    SecurityEvent.SECURE_AGGREGATION_COMPLETED,
                    {"aggregation_id": aggregation_id, "participants": len(validated_participants)},
                )

                self.security_metrics.secure_aggregations += 1

            return success, aggregated_result, metadata

        except Exception as e:
            logger.error(f"Secure aggregation failed: {e}")
            return False, None, {"error": str(e)}

    async def establish_secure_communication(
        self,
        remote_node_id: str,
        channel_type: ChannelType = ChannelType.HTTP3_COVERT,
        security_level: SecurityLevel = SecurityLevel.HIGH,
    ) -> Tuple[bool, Optional[str]]:
        """Establish secure communication channel with remote node."""

        try:
            # Check if remote node is trusted
            trust_decision, trust_score, reason = await self.trust_system.get_trust_decision(
                remote_node_id, TrustLevel.MEDIUM
            )

            if not trust_decision:
                logger.warning(f"Cannot establish channel with untrusted node {remote_node_id}: {reason}")
                return False, None

            # Establish secure channel
            success, channel_id = await self.transport_security.establish_secure_channel(
                remote_node_id, channel_type, security_level
            )

            if success:
                await self._emit_security_event(
                    SecurityEvent.SECURE_CHANNEL_ESTABLISHED,
                    {"remote_node": remote_node_id, "channel_id": channel_id, "security_level": security_level.value},
                )

                self.security_metrics.secure_channels_established += 1

            return success, channel_id

        except Exception as e:
            logger.error(f"Failed to establish secure channel with {remote_node_id}: {e}")
            return False, None

    async def secure_consensus_round(
        self, round_id: str, participants: List[str], proposal: Dict[str, Any]
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Execute a secure consensus round with Byzantine fault tolerance."""

        try:
            # Filter participants by trust level
            trusted_participants = []

            for participant_id in participants:
                trust_decision, _, _ = await self.trust_system.get_trust_decision(participant_id, TrustLevel.MEDIUM)

                if trust_decision:
                    trusted_participants.append(participant_id)
                else:
                    logger.warning(f"Excluding untrusted participant {participant_id} from consensus")

            if len(trusted_participants) < 3:
                logger.error("Insufficient trusted participants for consensus")
                return False, None

            # Initialize distributed keys if not done
            if not self.consensus_security.threshold_keys:
                success = await self.consensus_security.initialize_distributed_keys(trusted_participants)
                if not success:
                    return False, None

            # Create consensus round (simplified)
            from .consensus_security_manager import ConsensusRound

            consensus_round = ConsensusRound(
                round_id=round_id,
                round_number=1,
                view_number=0,
                leader_id=trusted_participants[0],
                participants=set(trusted_participants),
                proposal=proposal,
            )

            # Detect Byzantine behavior
            detected_attacks = await self.consensus_security.detect_byzantine_behavior(consensus_round)

            if detected_attacks:
                # Mitigate attacks
                await self.consensus_security.mitigate_attacks(detected_attacks)

                # Update trust for Byzantine nodes
                for attack in detected_attacks:
                    for node_id in attack.suspected_nodes:
                        await self.trust_system.report_byzantine_behavior(
                            node_id, attack.attack_type.value, attack.evidence_data, attack.confidence_score
                        )

                await self._emit_security_event(
                    SecurityEvent.BYZANTINE_BEHAVIOR_DETECTED, {"round_id": round_id, "attacks": len(detected_attacks)}
                )

                self.security_metrics.byzantine_attacks_prevented += len(detected_attacks)

            # Mark round as completed (simplified)
            consensus_round.status = "completed"
            consensus_round.end_time = time.time()

            await self._emit_security_event(
                SecurityEvent.CONSENSUS_ROUND_COMPLETED,
                {"round_id": round_id, "participants": len(trusted_participants)},
            )

            self.security_metrics.consensus_rounds_secured += 1

            return True, {"result": "consensus_achieved", "participants": len(trusted_participants)}

        except Exception as e:
            logger.error(f"Secure consensus round failed: {e}")
            return False, None

    async def handle_security_incident(
        self, incident_type: str, source_node: str, incident_data: Dict[str, Any]
    ) -> bool:
        """Handle a security incident with integrated response."""

        try:
            # Ingest event into threat detection
            await self.threat_detection.ingest_security_event(incident_type, source_node, self.node_id, incident_data)

            # Detect threats
            detected_threats = await self.threat_detection.detect_threats()

            if detected_threats:
                for threat in detected_threats:
                    # Mitigate threat
                    success, actions = await self.threat_detection.mitigate_threat(threat)

                    if success:
                        await self._emit_security_event(
                            SecurityEvent.THREAT_MITIGATED,
                            {
                                "threat_id": threat.event_id,
                                "threat_type": threat.threat_category.value,
                                "actions": actions,
                            },
                        )

                        self.security_metrics.incidents_resolved += 1

                    # Update trust based on threat
                    if threat.source_nodes:
                        for node_id in threat.source_nodes:
                            await self.trust_system.record_interaction(node_id, "security_incident", "threat_detected")

                await self._emit_security_event(SecurityEvent.THREAT_DETECTED, {"threats_count": len(detected_threats)})

                self.security_metrics.threats_detected += len(detected_threats)
                self.security_metrics.security_incidents += 1

            return True

        except Exception as e:
            logger.error(f"Security incident handling failed: {e}")
            return False

    # Private methods

    async def _setup_event_handlers(self) -> None:
        """Set up cross-component event handlers."""

        # Trust update handler
        async def handle_trust_update(event_data: Dict[str, Any]) -> None:
            node_id = event_data.get("node_id")
            if node_id:
                self.security_metrics.trust_updates += 1

        self.security_event_handlers[SecurityEvent.TRUST_UPDATED] = [handle_trust_update]

        # Threat detection handler
        async def handle_threat_detection(event_data: Dict[str, Any]) -> None:
            # Could trigger additional security measures
            pass

        self.security_event_handlers[SecurityEvent.THREAT_DETECTED] = [handle_threat_detection]

    async def _start_monitoring(self) -> None:
        """Start background monitoring tasks."""

        monitoring_tasks = [
            self._security_metrics_monitor(),
            self._component_health_monitor(),
            self._trust_maintenance_monitor(),
        ]

        for task_coro in monitoring_tasks:
            task = asyncio.create_task(task_coro)
            self.monitoring_tasks.append(task)

    async def _security_metrics_monitor(self) -> None:
        """Monitor security metrics and performance."""

        while self.started:
            try:
                # Update uptime
                self.security_metrics.uptime += 60.0  # 1 minute

                # Collect component metrics
                auth_stats = self.auth_system.get_auth_stats()
                self.security_metrics.active_sessions = auth_stats["active_sessions"]

                # Log periodic metrics
                if int(self.security_metrics.uptime) % 3600 == 0:  # Every hour
                    logger.info(f"Security metrics: {self.get_security_summary()}")

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Security metrics monitoring error: {e}")
                await asyncio.sleep(60)

    async def _component_health_monitor(self) -> None:
        """Monitor health of security components."""

        while self.started:
            try:
                # Check authentication system
                auth_health = await self.auth_system.health_check()
                self.component_health["authentication"] = auth_health["healthy"]

                # Check transport security
                transport_health = await self.transport_security.health_check()
                self.component_health["transport"] = transport_health["healthy"]

                # Check threat detection
                threat_health = await self.threat_detection.health_check()
                self.component_health["threat_detection"] = threat_health["healthy"]

                # Check trust system
                trust_health = await self.trust_system.health_check()
                self.component_health["trust"] = trust_health["healthy"]

                # Log unhealthy components
                unhealthy = [k for k, v in self.component_health.items() if not v]
                if unhealthy:
                    logger.warning(f"Unhealthy security components: {unhealthy}")

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Component health monitoring error: {e}")
                await asyncio.sleep(300)

    async def _trust_maintenance_monitor(self) -> None:
        """Monitor and maintain trust system."""

        while self.started:
            try:
                # Get trust statistics
                self.trust_system.get_trust_statistics()

                # Check for nodes with critically low trust
                self.trust_system.get_trusted_nodes(TrustLevel.UNTRUSTED)

                # Could implement additional trust maintenance logic here

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Trust maintenance monitoring error: {e}")
                await asyncio.sleep(3600)

    async def _emit_security_event(self, event: SecurityEvent, data: Dict[str, Any]) -> None:
        """Emit a security event to registered handlers."""

        try:
            handlers = self.security_event_handlers.get(event, [])

            for handler in handlers:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error(f"Security event handler error: {e}")

            # Log important events
            if event in [SecurityEvent.THREAT_DETECTED, SecurityEvent.BYZANTINE_BEHAVIOR_DETECTED]:
                logger.warning(f"Security event: {event.value} - {data}")
            else:
                logger.debug(f"Security event: {event.value} - {data}")

        except Exception as e:
            logger.error(f"Failed to emit security event: {e}")

    # Public API methods

    def get_security_summary(self) -> Dict[str, Any]:
        """Get comprehensive security summary."""

        return {
            "node_id": self.node_id,
            "security_metrics": {
                "total_authentications": self.security_metrics.total_authentications,
                "failed_authentications": self.security_metrics.failed_authentications,
                "active_sessions": self.security_metrics.active_sessions,
                "secure_aggregations": self.security_metrics.secure_aggregations,
                "secure_channels_established": self.security_metrics.secure_channels_established,
                "threats_detected": self.security_metrics.threats_detected,
                "consensus_rounds_secured": self.security_metrics.consensus_rounds_secured,
                "byzantine_attacks_prevented": self.security_metrics.byzantine_attacks_prevented,
                "security_incidents": self.security_metrics.security_incidents,
                "incidents_resolved": self.security_metrics.incidents_resolved,
                "uptime_hours": self.security_metrics.uptime / 3600.0,
            },
            "component_health": self.component_health,
            "authentication_stats": (
                self.auth_system.get_auth_stats() if hasattr(self.auth_system, "get_auth_stats") else {}
            ),
            "trust_stats": self.trust_system.get_trust_statistics(),
            "threat_detection_stats": self.threat_detection.get_detection_stats(),
            "aggregation_stats": self.secure_aggregation.get_aggregation_stats(),
        }

    def get_node_security_status(self, node_id: str) -> Dict[str, Any]:
        """Get security status for a specific node."""

        return {
            "node_id": node_id,
            "authentication": {
                "registered": node_id in self.auth_system.node_identities,
                "active_sessions": len(
                    [s for s in self.auth_system.active_sessions.values() if s.node_id == node_id and s.is_active]
                ),
            },
            "trust": self.trust_system.get_node_trust_details(node_id),
            "secure_channels": (
                len(
                    [
                        ch
                        for ch in self.transport_security.active_channels.values()
                        if ch.remote_node_id == node_id and ch.is_active
                    ]
                )
                if hasattr(self.transport_security, "active_channels")
                else 0
            ),
        }

    async def shutdown(self) -> None:
        """Shutdown security integration manager."""

        logger.info("Shutting down security integration manager...")

        self.started = False

        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)

        logger.info("Security integration manager shutdown complete")

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all security components."""

        health_results = {}

        # Check each component
        if "authentication" in self.component_health:
            health_results["authentication"] = await self.auth_system.health_check()

        if "transport" in self.component_health:
            health_results["transport"] = await self.transport_security.health_check()

        if "consensus" in self.component_health:
            health_results["consensus"] = await self.consensus_security.health_check()

        if "threat_detection" in self.component_health:
            health_results["threat_detection"] = await self.threat_detection.health_check()

        if "trust" in self.component_health:
            health_results["trust"] = await self.trust_system.health_check()

        # Overall health assessment
        all_healthy = all(result.get("healthy", False) for result in health_results.values())

        overall_issues = []
        overall_warnings = []

        for component, result in health_results.items():
            overall_issues.extend(result.get("issues", []))
            overall_warnings.extend(result.get("warnings", []))

        return {
            "healthy": all_healthy,
            "issues": overall_issues,
            "warnings": overall_warnings,
            "components": health_results,
            "security_summary": self.get_security_summary(),
        }
