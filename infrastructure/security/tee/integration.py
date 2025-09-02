"""
TEE Security Integration Manager

Integrates TEE attestation, enclave management, and constitutional policy enforcement
with the existing fog computing infrastructure. Provides unified API for:
- Secure node registration and attestation
- Constitutional workload deployment
- Real-time security monitoring
- Policy enforcement and compliance
- Audit trail and reporting

This is the primary integration point for fog computing constitutional security.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
import json
import logging
from typing import Any
import uuid

from ..constitutional.security_policy import ConstitutionalPolicyEngine, HarmCategory, get_policy_engine
from .attestation import ConstitutionalTier, TEEAttestationManager, TEEType, get_attestation_manager
from .enclave_manager import TEEEnclaveManager, WorkloadType, get_enclave_manager

# Import fog computing components
try:

    FOG_INTEGRATION_AVAILABLE = True
except ImportError:
    FOG_INTEGRATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class NodeSecurityStatus(Enum):
    """Security status of fog computing nodes."""

    UNATTESTED = "unattested"
    ATTESTATION_PENDING = "attestation_pending"
    ATTESTED = "attested"
    CONSTITUTIONAL_COMPLIANT = "constitutional_compliant"
    SECURITY_VIOLATION = "security_violation"
    QUARANTINED = "quarantined"


@dataclass
class SecureFogNode:
    """Secure fog node with TEE capabilities and constitutional compliance."""

    node_id: str

    # Node information
    node_type: str = "fog_node"  # fog_node, edge_device, cloud_instance
    hardware_capabilities: list[str] = field(default_factory=list)

    # Security status
    security_status: NodeSecurityStatus = NodeSecurityStatus.UNATTESTED
    tee_type: TEEType | None = None
    constitutional_tier: ConstitutionalTier = ConstitutionalTier.BRONZE

    # Attestation information
    attestation_result: dict[str, Any] | None = None
    last_attestation: datetime | None = None
    attestation_expires: datetime | None = None

    # Constitutional compliance
    policy_compliance_score: float = 0.0
    constitutional_violations: int = 0
    last_violation: datetime | None = None

    # Workload tracking
    active_enclaves: list[str] = field(default_factory=list)
    workload_history: list[dict[str, Any]] = field(default_factory=list)

    # Performance metrics
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    network_latency_ms: float = 0.0
    trust_score: float = 0.5

    # Registration details
    registered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(UTC))

    def is_attestation_valid(self) -> bool:
        """Check if attestation is still valid."""
        if not self.attestation_expires:
            return False
        return datetime.now(UTC) < self.attestation_expires

    def update_heartbeat(self):
        """Update node heartbeat timestamp."""
        self.last_heartbeat = datetime.now(UTC)


@dataclass
class ConstitutionalWorkloadRequest:
    """Request for constitutional workload execution."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Workload specification
    workload_type: WorkloadType = WorkloadType.INFERENCE
    workload_name: str = ""
    workload_description: str = ""

    # Constitutional requirements
    required_tier: ConstitutionalTier = ConstitutionalTier.SILVER
    harm_categories_monitored: list[HarmCategory] = field(default_factory=list)
    max_risk_tolerance: float = 0.3

    # Resource requirements
    min_memory_mb: int = 512
    min_cpu_cores: int = 1
    estimated_duration_seconds: int = 300
    requires_network_access: bool = False

    # Input data
    input_data: dict[str, Any] = field(default_factory=dict)
    model_requirements: dict[str, Any] = field(default_factory=dict)

    # Requester information
    requester_id: str = ""
    priority: int = 1  # 1-5, higher is more priority

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    deadline: datetime | None = None


class TEESecurityIntegrationManager:
    """
    Main TEE Security Integration Manager

    Coordinates TEE attestation, enclave management, and constitutional policy
    enforcement across the fog computing infrastructure.
    """

    def __init__(self):
        # Component managers
        self.attestation_manager: TEEAttestationManager | None = None
        self.enclave_manager: TEEEnclaveManager | None = None
        self.policy_engine: ConstitutionalPolicyEngine | None = None

        # State management
        self.secure_nodes: dict[str, SecureFogNode] = {}
        self.workload_requests: dict[str, ConstitutionalWorkloadRequest] = {}
        self.security_events: list[dict[str, Any]] = []

        # Configuration
        self.attestation_refresh_hours = 24
        self.max_constitutional_violations = 5
        self.quarantine_duration_hours = 24

        # Background tasks
        self.monitoring_task: asyncio.Task | None = None
        self.running = False

        logger.info("TEE Security Integration Manager initialized")

    async def start(self):
        """Start the security integration manager."""
        # Initialize component managers
        self.attestation_manager = await get_attestation_manager()
        self.enclave_manager = await get_enclave_manager()
        self.policy_engine = await get_policy_engine()

        # Start monitoring
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("TEE Security Integration Manager started")

    async def stop(self):
        """Stop the security integration manager."""
        self.running = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("TEE Security Integration Manager stopped")

    async def register_fog_node(self, node_id: str, node_info: dict[str, Any]) -> SecureFogNode:
        """Register a new fog computing node with security validation."""

        if node_id in self.secure_nodes:
            logger.info(f"Node {node_id} already registered, updating info")
            node = self.secure_nodes[node_id]
        else:
            node = SecureFogNode(node_id=node_id)

        # Update node information
        node.node_type = node_info.get("node_type", "fog_node")
        node.hardware_capabilities = node_info.get("hardware_capabilities", [])
        node.cpu_utilization = node_info.get("cpu_utilization", 0.0)
        node.memory_utilization = node_info.get("memory_utilization", 0.0)
        node.network_latency_ms = node_info.get("network_latency_ms", 0.0)

        # Start attestation process
        node.security_status = NodeSecurityStatus.ATTESTATION_PENDING

        try:
            # Detect TEE capabilities
            capabilities = await self.attestation_manager.detect_hardware_capabilities(node_id)

            if capabilities:
                # Use best available TEE
                if TEEType.INTEL_SGX in capabilities:
                    node.tee_type = TEEType.INTEL_SGX
                    node.constitutional_tier = ConstitutionalTier.GOLD
                elif TEEType.AMD_SEV_SNP in capabilities:
                    node.tee_type = TEEType.AMD_SEV_SNP
                    node.constitutional_tier = ConstitutionalTier.GOLD
                else:
                    node.tee_type = TEEType.SOFTWARE_TEE
                    node.constitutional_tier = ConstitutionalTier.BRONZE
            else:
                node.tee_type = TEEType.SOFTWARE_TEE
                node.constitutional_tier = ConstitutionalTier.BRONZE

            # Perform attestation
            attestation_result = await self._perform_node_attestation(node)

            if attestation_result.status.value == "verified":
                node.security_status = NodeSecurityStatus.ATTESTED
                node.attestation_result = {
                    "status": attestation_result.status.value,
                    "trust_score": attestation_result.trust_score,
                    "capabilities": [cap.value for cap in attestation_result.capabilities],
                    "constitutional_tier": attestation_result.constitutional_tier.value,
                }
                node.trust_score = attestation_result.trust_score
                node.last_attestation = datetime.now(UTC)
                node.attestation_expires = datetime.now(UTC) + timedelta(hours=self.attestation_refresh_hours)

                # Check constitutional compliance
                if await self._validate_constitutional_compliance(node):
                    node.security_status = NodeSecurityStatus.CONSTITUTIONAL_COMPLIANT
                    node.policy_compliance_score = 0.95  # High initial score

                logger.info(f"Node {node_id} successfully attested and registered")
            else:
                node.security_status = NodeSecurityStatus.SECURITY_VIOLATION
                logger.warning(f"Node {node_id} attestation failed")

        except Exception as e:
            logger.error(f"Error registering node {node_id}: {e}")
            node.security_status = NodeSecurityStatus.UNATTESTED

        # Store node
        self.secure_nodes[node_id] = node

        # Log security event
        await self._log_security_event(
            "node_registration", {"node_id": node_id, "security_status": node.security_status.value}, "info"
        )

        return node

    async def deploy_constitutional_workload(self, request: ConstitutionalWorkloadRequest) -> dict[str, Any]:
        """Deploy constitutional workload to appropriate secure node."""

        try:
            # Find suitable node
            suitable_nodes = await self._find_suitable_nodes(request)

            if not suitable_nodes:
                return {
                    "success": False,
                    "error": "No suitable nodes available for constitutional workload",
                    "request_id": request.request_id,
                }

            # Select best node (highest trust score)
            selected_node = max(suitable_nodes, key=lambda n: n.trust_score)

            # Validate workload against constitutional policy
            workload_manifest = {
                "type": request.workload_type.value,
                "constitutional_tier": request.required_tier.value,
                "harm_categories": [cat.value for cat in request.harm_categories_monitored],
                "privacy_requirements": "high",
            }

            node_attestation = selected_node.attestation_result

            if not await self.policy_engine.validate_workload_deployment(workload_manifest, node_attestation):
                return {
                    "success": False,
                    "error": "Workload failed constitutional policy validation",
                    "request_id": request.request_id,
                }

            # Deploy workload to enclave
            result = await self.enclave_manager.execute_constitutional_workload(
                selected_node.node_id,
                self._create_workload_manifest(request),
                request.input_data,
                request.required_tier,
            )

            # Update node tracking
            if result.get("success"):
                selected_node.workload_history.append(
                    {
                        "request_id": request.request_id,
                        "workload_type": request.workload_type.value,
                        "executed_at": datetime.now(UTC).isoformat(),
                        "constitutional_tier": request.required_tier.value,
                    }
                )

            # Store request
            self.workload_requests[request.request_id] = request

            # Log deployment
            await self._log_security_event(
                "workload_deployment",
                {
                    "request_id": request.request_id,
                    "node_id": selected_node.node_id,
                    "success": result.get("success", False),
                },
                "info",
            )

            return {
                "success": result.get("success", False),
                "result": result.get("result", {}),
                "node_id": selected_node.node_id,
                "execution_details": result,
                "request_id": request.request_id,
            }

        except Exception as e:
            logger.error(f"Error deploying constitutional workload: {e}")
            return {"success": False, "error": str(e), "request_id": request.request_id}

    async def monitor_constitutional_compliance(
        self, node_id: str, content: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Monitor content for constitutional compliance."""

        if node_id not in self.secure_nodes:
            return {"error": "Node not registered", "compliant": False}

        node = self.secure_nodes[node_id]
        context = context or {}
        context["node_id"] = node_id

        # Evaluate content against constitutional policy
        is_safe, evaluation = await self.policy_engine.evaluate_content(content, context=context)

        # Update node compliance tracking
        if not is_safe:
            node.constitutional_violations += 1
            node.last_violation = datetime.now(UTC)
            node.policy_compliance_score = max(0.0, node.policy_compliance_score - 0.1)

            # Check for quarantine threshold
            if node.constitutional_violations >= self.max_constitutional_violations:
                await self._quarantine_node(node_id, "Excessive constitutional violations")
        else:
            # Improve compliance score for good content
            node.policy_compliance_score = min(1.0, node.policy_compliance_score + 0.01)

        return {
            "compliant": is_safe,
            "evaluation": evaluation,
            "node_compliance_score": node.policy_compliance_score,
            "node_violations": node.constitutional_violations,
        }

    def get_secure_nodes_summary(self) -> dict[str, Any]:
        """Get summary of secure fog nodes."""

        summary = {
            "total_nodes": len(self.secure_nodes),
            "by_status": {},
            "by_constitutional_tier": {},
            "by_tee_type": {},
            "compliance_metrics": {
                "average_trust_score": 0.0,
                "average_compliance_score": 0.0,
                "total_violations": 0,
                "quarantined_nodes": 0,
            },
        }

        trust_scores = []
        compliance_scores = []
        total_violations = 0

        for node in self.secure_nodes.values():
            # Count by status
            status = node.security_status.value
            summary["by_status"][status] = summary["by_status"].get(status, 0) + 1

            # Count by constitutional tier
            tier = node.constitutional_tier.value
            summary["by_constitutional_tier"][tier] = summary["by_constitutional_tier"].get(tier, 0) + 1

            # Count by TEE type
            if node.tee_type:
                tee_type = node.tee_type.value
                summary["by_tee_type"][tee_type] = summary["by_tee_type"].get(tee_type, 0) + 1

            # Collect metrics
            trust_scores.append(node.trust_score)
            compliance_scores.append(node.policy_compliance_score)
            total_violations += node.constitutional_violations

            if node.security_status == NodeSecurityStatus.QUARANTINED:
                summary["compliance_metrics"]["quarantined_nodes"] += 1

        # Calculate averages
        if trust_scores:
            summary["compliance_metrics"]["average_trust_score"] = sum(trust_scores) / len(trust_scores)
            summary["compliance_metrics"]["average_compliance_score"] = sum(compliance_scores) / len(compliance_scores)

        summary["compliance_metrics"]["total_violations"] = total_violations

        return summary

    async def generate_security_report(self, report_period_days: int = 7) -> dict[str, Any]:
        """Generate comprehensive security report."""

        cutoff_time = datetime.now(UTC) - timedelta(days=report_period_days)

        # Get nodes summary
        nodes_summary = self.get_secure_nodes_summary()

        # Get recent security events
        recent_events = [
            event for event in self.security_events if datetime.fromisoformat(event["timestamp"]) > cutoff_time
        ]

        # Get policy violations summary
        violations_summary = self.policy_engine.get_policy_violations_summary(time_window_hours=report_period_days * 24)

        # Calculate security metrics
        attestation_success_rate = 0.0
        if nodes_summary["total_nodes"] > 0:
            attested_nodes = nodes_summary["by_status"].get("attested", 0) + nodes_summary["by_status"].get(
                "constitutional_compliant", 0
            )
            attestation_success_rate = attested_nodes / nodes_summary["total_nodes"]

        report = {
            "report_id": str(uuid.uuid4()),
            "generated_at": datetime.now(UTC).isoformat(),
            "report_period_days": report_period_days,
            "summary": {
                "total_nodes": nodes_summary["total_nodes"],
                "attestation_success_rate": attestation_success_rate,
                "average_trust_score": nodes_summary["compliance_metrics"]["average_trust_score"],
                "average_compliance_score": nodes_summary["compliance_metrics"]["average_compliance_score"],
                "total_violations": violations_summary["total_violations"],
                "quarantined_nodes": nodes_summary["compliance_metrics"]["quarantined_nodes"],
            },
            "detailed_metrics": {
                "nodes_by_status": nodes_summary["by_status"],
                "nodes_by_tier": nodes_summary["by_constitutional_tier"],
                "nodes_by_tee_type": nodes_summary["by_tee_type"],
                "violations_by_category": violations_summary["violations_by_category"],
                "security_events": len(recent_events),
            },
            "recommendations": await self._generate_security_recommendations(nodes_summary, violations_summary),
        }

        return report

    # Private helper methods

    async def _perform_node_attestation(self, node: SecureFogNode):
        """Perform TEE attestation for node."""

        workload_hash = b"constitutional_fog_workload"

        quote = await self.attestation_manager.generate_attestation_quote(
            node.node_id, node.tee_type, workload_hash, node.constitutional_tier
        )

        return await self.attestation_manager.verify_attestation(node.node_id, quote)

    async def _validate_constitutional_compliance(self, node: SecureFogNode) -> bool:
        """Validate node meets constitutional compliance requirements."""

        if not node.attestation_result:
            return False

        # Check minimum trust score
        if node.trust_score < 0.7:
            return False

        # Check attestation status
        if node.attestation_result.get("status") != "verified":
            return False

        # Check constitutional tier is appropriate
        required_capabilities = ["memory_encryption", "remote_attestation"]
        node_capabilities = node.attestation_result.get("capabilities", [])

        for capability in required_capabilities:
            if capability not in node_capabilities:
                return False

        return True

    async def _find_suitable_nodes(self, request: ConstitutionalWorkloadRequest) -> list[SecureFogNode]:
        """Find nodes suitable for constitutional workload."""

        suitable_nodes = []

        for node in self.secure_nodes.values():
            # Check security status
            if node.security_status != NodeSecurityStatus.CONSTITUTIONAL_COMPLIANT:
                continue

            # Check attestation is valid
            if not node.is_attestation_valid():
                continue

            # Check constitutional tier compatibility
            tier_hierarchy = {ConstitutionalTier.BRONZE: 1, ConstitutionalTier.SILVER: 2, ConstitutionalTier.GOLD: 3}

            node_level = tier_hierarchy.get(node.constitutional_tier, 1)
            required_level = tier_hierarchy.get(request.required_tier, 2)

            if node_level < required_level:
                continue

            # Check resource availability (simplified)
            if node.memory_utilization > 0.8:  # 80% memory threshold
                continue

            if node.cpu_utilization > 0.9:  # 90% CPU threshold
                continue

            # Check compliance score
            if node.policy_compliance_score < 0.7:
                continue

            suitable_nodes.append(node)

        return suitable_nodes

    def _create_workload_manifest(self, request: ConstitutionalWorkloadRequest):
        """Create workload manifest from request."""
        from .enclave_manager import WorkloadManifest

        return WorkloadManifest(
            name=request.workload_name,
            workload_type=request.workload_type,
            constitutional_tier=request.required_tier,
            harm_categories=[cat.value for cat in request.harm_categories_monitored],
            min_memory_mb=request.min_memory_mb,
            estimated_runtime_seconds=request.estimated_duration_seconds,
            requires_network_access=request.requires_network_access,
            created_by=request.requester_id,
        )

    async def _quarantine_node(self, node_id: str, reason: str):
        """Quarantine node for security violations."""

        if node_id not in self.secure_nodes:
            return

        node = self.secure_nodes[node_id]
        node.security_status = NodeSecurityStatus.QUARANTINED

        # Terminate any active enclaves
        for enclave_id in node.active_enclaves:
            try:
                await self.enclave_manager.terminate_enclave(enclave_id)
            except Exception as e:
                logger.error(f"Error terminating enclave {enclave_id}: {e}")

        node.active_enclaves.clear()

        # Log quarantine event
        await self._log_security_event("node_quarantine", {"node_id": node_id, "reason": reason}, "warning")

        logger.warning(f"Node {node_id} quarantined: {reason}")

    async def _log_security_event(self, event_type: str, details: dict[str, Any], severity: str = "info"):
        """Log security event."""

        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "severity": severity,
            "details": details,
        }

        self.security_events.append(event)

        # Keep only recent events (last 1000)
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]

        logger.info(f"Security event logged: {event_type} ({severity})")

    async def _generate_security_recommendations(
        self, nodes_summary: dict[str, Any], violations_summary: dict[str, Any]
    ) -> list[str]:
        """Generate security recommendations based on metrics."""

        recommendations = []

        # Check attestation success rate
        total_nodes = nodes_summary["total_nodes"]
        if total_nodes > 0:
            attested_nodes = nodes_summary["by_status"].get("attested", 0) + nodes_summary["by_status"].get(
                "constitutional_compliant", 0
            )
            success_rate = attested_nodes / total_nodes

            if success_rate < 0.8:
                recommendations.append("Improve node attestation process - low success rate detected")

        # Check violations
        total_violations = violations_summary["total_violations"]
        if total_violations > 10:
            recommendations.append("Review constitutional policies - high violation count")

        # Check quarantined nodes
        quarantined = nodes_summary["compliance_metrics"]["quarantined_nodes"]
        if quarantined > 0:
            recommendations.append(f"Investigate {quarantined} quarantined nodes")

        # Check trust scores
        avg_trust = nodes_summary["compliance_metrics"]["average_trust_score"]
        if avg_trust < 0.7:
            recommendations.append("Improve node trust scores through better attestation")

        # Check TEE distribution
        tee_types = nodes_summary["by_tee_type"]
        software_tee_ratio = tee_types.get("software_tee", 0) / max(1, total_nodes)
        if software_tee_ratio > 0.5:
            recommendations.append("Consider upgrading to hardware TEE for better security")

        if not recommendations:
            recommendations.append("Security posture is healthy - maintain current practices")

        return recommendations

    async def _monitoring_loop(self):
        """Background monitoring for security events."""

        while self.running:
            try:
                current_time = datetime.now(UTC)

                # Check node attestation expiration
                for node_id, node in list(self.secure_nodes.items()):
                    if not node.is_attestation_valid():
                        if node.security_status in [
                            NodeSecurityStatus.ATTESTED,
                            NodeSecurityStatus.CONSTITUTIONAL_COMPLIANT,
                        ]:
                            logger.warning(f"Node {node_id} attestation expired")
                            node.security_status = NodeSecurityStatus.ATTESTATION_PENDING

                            # Try to refresh attestation
                            try:
                                await self._perform_node_attestation(node)
                            except Exception as e:
                                logger.error(f"Failed to refresh attestation for {node_id}: {e}")

                # Check for stale heartbeats
                stale_threshold = timedelta(minutes=10)
                for node_id, node in list(self.secure_nodes.items()):
                    if current_time - node.last_heartbeat > stale_threshold:
                        logger.warning(f"Node {node_id} has stale heartbeat")
                        if node.security_status == NodeSecurityStatus.CONSTITUTIONAL_COMPLIANT:
                            node.security_status = NodeSecurityStatus.SECURITY_VIOLATION

                await asyncio.sleep(300)  # Check every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)


# Global integration manager instance
_integration_manager: TEESecurityIntegrationManager | None = None


async def get_integration_manager() -> TEESecurityIntegrationManager:
    """Get global TEE security integration manager instance."""
    global _integration_manager

    if _integration_manager is None:
        _integration_manager = TEESecurityIntegrationManager()
        await _integration_manager.start()

    return _integration_manager


# High-level convenience functions


async def register_constitutional_fog_node(node_id: str, node_capabilities: dict[str, Any]) -> dict[str, Any]:
    """Register a fog node with constitutional security capabilities."""

    manager = await get_integration_manager()
    node = await manager.register_fog_node(node_id, node_capabilities)

    return {
        "node_id": node.node_id,
        "security_status": node.security_status.value,
        "constitutional_tier": node.constitutional_tier.value,
        "trust_score": node.trust_score,
        "tee_type": node.tee_type.value if node.tee_type else None,
    }


async def execute_constitutional_workload(
    workload_type: str, workload_name: str, input_data: dict[str, Any], constitutional_requirements: dict[str, Any]
) -> dict[str, Any]:
    """Execute constitutional workload on secure fog infrastructure."""

    # Convert workload type
    workload_type_enum = WorkloadType(workload_type)

    # Create request
    request = ConstitutionalWorkloadRequest(
        workload_type=workload_type_enum,
        workload_name=workload_name,
        input_data=input_data,
        required_tier=ConstitutionalTier(constitutional_requirements.get("tier", "silver")),
        harm_categories_monitored=[HarmCategory(cat) for cat in constitutional_requirements.get("harm_categories", [])],
        requester_id=constitutional_requirements.get("requester_id", "anonymous"),
    )

    manager = await get_integration_manager()
    return await manager.deploy_constitutional_workload(request)


if __name__ == "__main__":

    async def test_integration():
        """Test TEE security integration."""

        # Register a fog node
        node_result = await register_constitutional_fog_node(
            "test_fog_node_001",
            {
                "node_type": "fog_node",
                "hardware_capabilities": ["intel_sgx", "memory_encryption"],
                "cpu_utilization": 0.3,
                "memory_utilization": 0.2,
            },
        )
        print(f"Node registration: {json.dumps(node_result, indent=2)}")

        # Execute constitutional workload
        workload_result = await execute_constitutional_workload(
            "inference",
            "constitutional-qa",
            {"question": "What is the capital of France?"},
            {"tier": "silver", "harm_categories": ["misinformation", "hate_speech"], "requester_id": "test_user"},
        )
        print(f"Workload execution: {json.dumps(workload_result, indent=2)}")

        # Generate security report
        manager = await get_integration_manager()
        report = await manager.generate_security_report()
        print(f"Security report: {json.dumps(report, indent=2)}")

        await manager.stop()

    asyncio.run(test_integration())
