"""
Unit tests for TEE Security Integration Manager using TDD London School methodology.

Tests focus on TEE attestation, enclave management, and constitutional policy
coordination through behavior verification and mock-driven development.
"""

import asyncio
import pytest
from datetime import datetime, UTC, timedelta
from unittest.mock import AsyncMock, Mock

# Import the module under test
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from infrastructure.security.tee.integration import (
    TEESecurityIntegrationManager,
    SecureFogNode,
    NodeSecurityStatus,
    ConstitutionalWorkloadRequest,
)

# Mock the imports that may not be available
try:
    from infrastructure.security.tee.attestation import TEEType, ConstitutionalTier
    from infrastructure.security.tee.enclave_manager import WorkloadType
    from infrastructure.security.constitutional.security_policy import HarmCategory
except ImportError:
    # Create mock enums for testing
    from enum import Enum

    class TEEType(Enum):
        INTEL_SGX = "intel_sgx"
        AMD_SEV_SNP = "amd_sev_snp"
        SOFTWARE_TEE = "software_tee"

    class ConstitutionalTier(Enum):
        BRONZE = "bronze"
        SILVER = "silver"
        GOLD = "gold"

    class WorkloadType(Enum):
        INFERENCE = "inference"
        TRAINING = "training"
        VALIDATION = "validation"

    class HarmCategory(Enum):
        MISINFORMATION = "misinformation"
        HATE_SPEECH = "hate_speech"
        PRIVACY_VIOLATION = "privacy_violation"


class TestTEESecurityIntegrationManager:
    """Test suite for TEE Security Integration Manager using London School TDD."""

    @pytest.fixture
    def mock_attestation_manager(self):
        """Mock TEE attestation manager."""
        mock = AsyncMock()

        # Mock attestation result
        mock_result = Mock()
        mock_result.status.value = "verified"
        mock_result.trust_score = 0.9
        mock_result.capabilities = ["memory_encryption", "remote_attestation"]
        mock_result.constitutional_tier = ConstitutionalTier.GOLD

        mock.detect_hardware_capabilities = AsyncMock(return_value=[TEEType.INTEL_SGX])
        mock.generate_attestation_quote = AsyncMock(return_value="quote_data")
        mock.verify_attestation = AsyncMock(return_value=mock_result)

        return mock

    @pytest.fixture
    def mock_enclave_manager(self):
        """Mock TEE enclave manager."""
        mock = AsyncMock()
        mock.execute_constitutional_workload = AsyncMock(
            return_value={"success": True, "result": {"output": "constitutional_response"}, "execution_time": 150.0}
        )
        mock.terminate_enclave = AsyncMock()
        return mock

    @pytest.fixture
    def mock_policy_engine(self):
        """Mock constitutional policy engine."""
        mock = AsyncMock()
        mock.validate_workload_deployment = AsyncMock(return_value=True)
        mock.evaluate_content = AsyncMock(return_value=(True, {"risk_score": 0.1}))
        mock.get_policy_violations_summary = Mock(
            return_value={"total_violations": 5, "violations_by_category": {"hate_speech": 2, "misinformation": 3}}
        )
        return mock

    @pytest.fixture
    def tee_manager(self, mock_attestation_manager, mock_enclave_manager, mock_policy_engine):
        """Create TEE Security Integration Manager with mocked dependencies."""
        manager = TEESecurityIntegrationManager()

        # Inject mock dependencies
        manager.attestation_manager = mock_attestation_manager
        manager.enclave_manager = mock_enclave_manager
        manager.policy_engine = mock_policy_engine

        return manager

    @pytest.mark.asyncio
    async def test_register_fog_node_coordinates_attestation_flow(self, tee_manager):
        """Test fog node registration coordinates complete attestation workflow."""
        # Given node information with hardware capabilities
        node_info = {
            "node_type": "fog_node",
            "hardware_capabilities": ["intel_sgx", "memory_encryption"],
            "cpu_utilization": 0.3,
            "memory_utilization": 0.2,
            "network_latency_ms": 15.0,
        }

        # When registering fog node
        node = await tee_manager.register_fog_node("fog_node_001", node_info)

        # Then it should detect hardware capabilities
        tee_manager.attestation_manager.detect_hardware_capabilities.assert_called_once_with("fog_node_001")

        # And perform attestation
        tee_manager.attestation_manager.generate_attestation_quote.assert_called_once()
        tee_manager.attestation_manager.verify_attestation.assert_called_once()

        # And configure node based on attestation results
        assert node.node_id == "fog_node_001"
        assert node.tee_type == TEEType.INTEL_SGX
        assert node.constitutional_tier == ConstitutionalTier.GOLD
        assert node.security_status == NodeSecurityStatus.CONSTITUTIONAL_COMPLIANT
        assert node.trust_score == 0.9

        # And store node information
        assert tee_manager.secure_nodes["fog_node_001"] == node

    @pytest.mark.asyncio
    async def test_deploy_constitutional_workload_coordinates_secure_execution(self, tee_manager):
        """Test workload deployment coordinates secure execution workflow."""
        # Given a registered and attested node
        await tee_manager.register_fog_node(
            "fog_node_001", {"hardware_capabilities": ["intel_sgx"], "cpu_utilization": 0.2, "memory_utilization": 0.1}
        )

        # And a constitutional workload request
        request = ConstitutionalWorkloadRequest(
            workload_type=WorkloadType.INFERENCE,
            workload_name="constitutional_qa",
            required_tier=ConstitutionalTier.SILVER,
            harm_categories_monitored=[HarmCategory.MISINFORMATION, HarmCategory.HATE_SPEECH],
            input_data={"question": "What is the capital of France?"},
            requester_id="test_user",
        )

        # When deploying workload
        result = await tee_manager.deploy_constitutional_workload(request)

        # Then it should validate workload against constitutional policy
        tee_manager.policy_engine.validate_workload_deployment.assert_called_once()

        # And execute workload in secure enclave
        tee_manager.enclave_manager.execute_constitutional_workload.assert_called_once()

        # And return successful deployment
        assert result["success"] is True
        assert result["node_id"] == "fog_node_001"
        assert result["result"]["output"] == "constitutional_response"
        assert "request_id" in result

    @pytest.mark.asyncio
    async def test_deploy_workload_rejects_unsuitable_nodes(self, tee_manager):
        """Test workload deployment rejects nodes that don't meet requirements."""
        # Given a node with insufficient constitutional tier
        await tee_manager.register_fog_node(
            "weak_node_001",
            {
                "hardware_capabilities": [],  # No TEE capabilities
                "cpu_utilization": 0.9,  # High CPU usage
                "memory_utilization": 0.85,  # High memory usage
            },
        )

        # Force the node to have lower capabilities
        node = tee_manager.secure_nodes["weak_node_001"]
        node.constitutional_tier = ConstitutionalTier.BRONZE
        node.security_status = NodeSecurityStatus.ATTESTED  # Not constitutional compliant

        # And a workload requiring high security
        request = ConstitutionalWorkloadRequest(
            workload_type=WorkloadType.TRAINING,
            required_tier=ConstitutionalTier.GOLD,
            harm_categories_monitored=[HarmCategory.PRIVACY_VIOLATION],
        )

        # When attempting to deploy workload
        result = await tee_manager.deploy_constitutional_workload(request)

        # Then it should reject the deployment
        assert result["success"] is False
        assert "No suitable nodes available" in result["error"]

        # And not execute workload
        tee_manager.enclave_manager.execute_constitutional_workload.assert_not_called()

    @pytest.mark.asyncio
    async def test_monitor_constitutional_compliance_coordinates_policy_evaluation(self, tee_manager):
        """Test compliance monitoring coordinates policy evaluation and enforcement."""
        # Given a registered node
        await tee_manager.register_fog_node("fog_node_001", {"hardware_capabilities": ["intel_sgx"]})

        # When monitoring content for compliance
        result = await tee_manager.monitor_constitutional_compliance(
            "fog_node_001", "This is safe educational content about history.", {"context": "educational"}
        )

        # Then it should evaluate content with policy engine
        tee_manager.policy_engine.evaluate_content.assert_called_once_with(
            "This is safe educational content about history.",
            context={"context": "educational", "node_id": "fog_node_001"},
        )

        # And return compliance result
        assert result["compliant"] is True
        assert result["evaluation"]["risk_score"] == 0.1
        assert "node_compliance_score" in result

    @pytest.mark.asyncio
    async def test_monitor_compliance_tracks_violations_and_quarantine(self, tee_manager):
        """Test compliance monitoring tracks violations and triggers quarantine."""
        # Given a registered node
        await tee_manager.register_fog_node("fog_node_001", {"hardware_capabilities": ["intel_sgx"]})

        # And policy engine detects violation
        tee_manager.policy_engine.evaluate_content.return_value = (False, {"risk_score": 0.9})

        # When monitoring multiple violations
        for i in range(6):  # Exceed max_constitutional_violations (5)
            await tee_manager.monitor_constitutional_compliance("fog_node_001", f"Harmful content {i}", {})

        # Then it should track violations
        node = tee_manager.secure_nodes["fog_node_001"]
        assert node.constitutional_violations >= 5

        # And quarantine node after threshold
        assert node.security_status == NodeSecurityStatus.QUARANTINED

        # And terminate active enclaves
        if node.active_enclaves:
            tee_manager.enclave_manager.terminate_enclave.assert_called()

    @pytest.mark.asyncio
    async def test_generate_security_report_aggregates_comprehensive_metrics(self, tee_manager):
        """Test security report generation aggregates metrics from all components."""
        # Given multiple registered nodes with different statuses
        await tee_manager.register_fog_node("compliant_node", {"hardware_capabilities": ["intel_sgx"]})
        await tee_manager.register_fog_node("attested_node", {"hardware_capabilities": []})

        # Force different statuses for testing
        tee_manager.secure_nodes["attested_node"].security_status = NodeSecurityStatus.ATTESTED
        tee_manager.secure_nodes["attested_node"].constitutional_tier = ConstitutionalTier.BRONZE

        # When generating security report
        report = await tee_manager.generate_security_report(report_period_days=7)

        # Then it should aggregate node metrics
        assert report["summary"]["total_nodes"] == 2
        assert report["detailed_metrics"]["nodes_by_status"]["constitutional_compliant"] == 1
        assert report["detailed_metrics"]["nodes_by_status"]["attested"] == 1

        # And get policy violations summary
        tee_manager.policy_engine.get_policy_violations_summary.assert_called_once_with(
            time_window_hours=168  # 7 days * 24 hours
        )

        # And include policy violation metrics
        assert report["summary"]["total_violations"] == 5
        assert report["detailed_metrics"]["violations_by_category"]["hate_speech"] == 2

        # And provide recommendations
        assert "recommendations" in report
        assert len(report["recommendations"]) > 0

    def test_get_secure_nodes_summary_aggregates_node_metrics(self, tee_manager):
        """Test node summary aggregation coordinates metric collection."""
        # Given nodes with different configurations
        node1 = SecureFogNode(
            node_id="node1",
            security_status=NodeSecurityStatus.CONSTITUTIONAL_COMPLIANT,
            constitutional_tier=ConstitutionalTier.GOLD,
            tee_type=TEEType.INTEL_SGX,
            trust_score=0.9,
            policy_compliance_score=0.95,
            constitutional_violations=0,
        )

        node2 = SecureFogNode(
            node_id="node2",
            security_status=NodeSecurityStatus.QUARANTINED,
            constitutional_tier=ConstitutionalTier.BRONZE,
            tee_type=TEEType.SOFTWARE_TEE,
            trust_score=0.3,
            policy_compliance_score=0.2,
            constitutional_violations=8,
        )

        tee_manager.secure_nodes = {"node1": node1, "node2": node2}

        # When getting summary
        summary = tee_manager.get_secure_nodes_summary()

        # Then it should aggregate all metrics
        assert summary["total_nodes"] == 2
        assert summary["by_status"]["constitutional_compliant"] == 1
        assert summary["by_status"]["quarantined"] == 1
        assert summary["by_constitutional_tier"]["gold"] == 1
        assert summary["by_constitutional_tier"]["bronze"] == 1
        assert summary["by_tee_type"]["intel_sgx"] == 1
        assert summary["by_tee_type"]["software_tee"] == 1

        # And calculate compliance metrics
        assert summary["compliance_metrics"]["average_trust_score"] == 0.6  # (0.9 + 0.3) / 2
        assert summary["compliance_metrics"]["average_compliance_score"] == 0.575  # (0.95 + 0.2) / 2
        assert summary["compliance_metrics"]["total_violations"] == 8
        assert summary["compliance_metrics"]["quarantined_nodes"] == 1

    @pytest.mark.asyncio
    async def test_monitoring_loop_coordinates_background_maintenance(self, tee_manager):
        """Test monitoring loop coordinates background maintenance tasks."""
        # Given a node with expiring attestation
        await tee_manager.register_fog_node("fog_node_001", {"hardware_capabilities": ["intel_sgx"]})

        node = tee_manager.secure_nodes["fog_node_001"]
        node.attestation_expires = datetime.now(UTC) - timedelta(minutes=1)  # Expired

        # And start monitoring
        tee_manager.running = True
        monitoring_task = asyncio.create_task(tee_manager._monitoring_loop())

        # Let it run briefly
        await asyncio.sleep(0.1)
        tee_manager.running = False

        try:
            await asyncio.wait_for(monitoring_task, timeout=1.0)
        except asyncio.TimeoutError:
            monitoring_task.cancel()

        # Then it should detect expired attestation
        assert node.security_status == NodeSecurityStatus.ATTESTATION_PENDING

        # And attempt to refresh attestation
        # (In actual implementation, would verify attestation refresh was attempted)

    def test_find_suitable_nodes_filters_by_security_criteria(self, tee_manager):
        """Test suitable node filtering coordinates security requirement validation."""
        # Given nodes with different security profiles
        compliant_node = SecureFogNode(
            node_id="compliant",
            security_status=NodeSecurityStatus.CONSTITUTIONAL_COMPLIANT,
            constitutional_tier=ConstitutionalTier.GOLD,
            policy_compliance_score=0.8,
            cpu_utilization=0.3,
            memory_utilization=0.4,
            attestation_expires=datetime.now(UTC) + timedelta(hours=1),
        )

        insufficient_node = SecureFogNode(
            node_id="insufficient",
            security_status=NodeSecurityStatus.ATTESTED,  # Not constitutional compliant
            constitutional_tier=ConstitutionalTier.SILVER,
            policy_compliance_score=0.6,
            cpu_utilization=0.9,  # High resource usage
            memory_utilization=0.9,
        )

        tee_manager.secure_nodes = {"compliant": compliant_node, "insufficient": insufficient_node}

        # When finding suitable nodes for high-tier workload
        request = ConstitutionalWorkloadRequest(required_tier=ConstitutionalTier.GOLD)

        suitable_nodes = asyncio.run(tee_manager._find_suitable_nodes(request))

        # Then it should filter based on security criteria
        assert len(suitable_nodes) == 1
        assert suitable_nodes[0].node_id == "compliant"

    def test_workload_manifest_creation_coordinates_configuration(self, tee_manager):
        """Test workload manifest creation coordinates proper configuration."""
        # Given a constitutional workload request
        request = ConstitutionalWorkloadRequest(
            workload_name="test_workload",
            workload_type=WorkloadType.INFERENCE,
            required_tier=ConstitutionalTier.SILVER,
            harm_categories_monitored=[HarmCategory.MISINFORMATION],
            min_memory_mb=1024,
            estimated_duration_seconds=300,
            requires_network_access=True,
            requester_id="test_user",
        )

        # When creating workload manifest
        manifest = tee_manager._create_workload_manifest(request)

        # Then it should coordinate proper configuration
        assert manifest.name == "test_workload"
        assert manifest.workload_type == WorkloadType.INFERENCE
        assert manifest.constitutional_tier == ConstitutionalTier.SILVER
        assert manifest.harm_categories == ["misinformation"]
        assert manifest.min_memory_mb == 1024
        assert manifest.estimated_runtime_seconds == 300
        assert manifest.requires_network_access is True
        assert manifest.created_by == "test_user"

    def test_security_recommendations_coordinate_policy_guidance(self, tee_manager):
        """Test security recommendations coordinate appropriate policy guidance."""
        # Given various security metrics
        nodes_summary = {
            "total_nodes": 10,
            "by_status": {"attested": 6, "constitutional_compliant": 3, "quarantined": 1},
            "by_tee_type": {"software_tee": 8, "intel_sgx": 2},
            "compliance_metrics": {"average_trust_score": 0.6, "quarantined_nodes": 1},
        }

        violations_summary = {"total_violations": 25}

        # When generating recommendations
        recommendations = asyncio.run(tee_manager._generate_security_recommendations(nodes_summary, violations_summary))

        # Then it should coordinate appropriate guidance
        assert len(recommendations) > 0

        # Check for specific recommendation types based on metrics
        rec_text = " ".join(recommendations)
        assert "attestation" in rec_text.lower()  # Low attestation success rate
        assert "violation" in rec_text.lower()  # High violation count
        assert "quarantine" in rec_text.lower()  # Quarantined nodes present
        assert "trust" in rec_text.lower()  # Low trust scores
        assert "hardware" in rec_text.lower()  # Too many software TEEs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
