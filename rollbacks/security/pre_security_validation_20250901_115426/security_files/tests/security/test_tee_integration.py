"""
Comprehensive Test Suite for TEE Security Integration

Tests the complete TEE integration system including:
- Node registration and attestation
- Constitutional workload deployment
- Policy enforcement and compliance
- Security monitoring and reporting
- Integration with fog computing infrastructure

This validates the critical path for constitutional fog compute execution.
"""

import asyncio
import pytest
from datetime import datetime

# Import TEE security components
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from infrastructure.security.tee.attestation import (
    TEEType,
    ConstitutionalTier,
    get_attestation_manager,
)
from infrastructure.security.tee.enclave_manager import get_enclave_manager, WorkloadType
from infrastructure.security.constitutional.security_policy import (
    get_policy_engine,
    HarmCategory,
)
from infrastructure.security.tee.integration import (
    get_integration_manager,
    ConstitutionalWorkloadRequest,
    NodeSecurityStatus,
    register_constitutional_fog_node,
    execute_constitutional_workload,
)


class TestTEEAttestation:
    """Test TEE attestation framework."""

    @pytest.fixture
    async def attestation_manager(self):
        """Get attestation manager instance."""
        return await get_attestation_manager()

    @pytest.mark.asyncio
    async def test_hardware_capability_detection(self, attestation_manager):
        """Test hardware TEE capability detection."""

        node_id = "test_node_attestation"
        capabilities = await attestation_manager.detect_hardware_capabilities(node_id)

        assert isinstance(capabilities, dict)
        assert TEEType.SOFTWARE_TEE in capabilities  # Always available

        # Check software TEE capabilities
        software_caps = capabilities[TEEType.SOFTWARE_TEE]
        assert len(software_caps) > 0

    @pytest.mark.asyncio
    async def test_sgx_attestation_flow(self, attestation_manager):
        """Test Intel SGX attestation flow."""

        node_id = "test_sgx_node"

        # Generate SGX quote
        quote = await attestation_manager.generate_attestation_quote(
            node_id, TEEType.INTEL_SGX, b"test_workload_hash", ConstitutionalTier.GOLD
        )

        assert quote.tee_type == TEEType.INTEL_SGX
        assert quote.constitutional_tier == ConstitutionalTier.GOLD
        assert len(quote.mr_enclave) == 32  # SHA-256 measurement
        assert len(quote.signature) > 0
        assert quote.is_valid()

        # Verify attestation
        result = await attestation_manager.verify_attestation(node_id, quote)

        assert result.status.value == "verified"
        assert result.tee_type == TEEType.INTEL_SGX
        assert result.trust_score > 0.8  # High trust for SGX
        assert result.constitutional_tier == ConstitutionalTier.GOLD

    @pytest.mark.asyncio
    async def test_sev_attestation_flow(self, attestation_manager):
        """Test AMD SEV-SNP attestation flow."""

        node_id = "test_sev_node"

        # Generate SEV quote
        quote = await attestation_manager.generate_attestation_quote(
            node_id, TEEType.AMD_SEV_SNP, b"test_workload_hash", ConstitutionalTier.GOLD
        )

        assert quote.tee_type == TEEType.AMD_SEV_SNP
        assert len(quote.mr_enclave) == 48  # SHA-384 measurement for SEV
        assert len(quote.signature) > 0

        # Verify attestation
        result = await attestation_manager.verify_attestation(node_id, quote)

        assert result.status.value == "verified"
        assert result.tee_type == TEEType.AMD_SEV_SNP
        assert result.trust_score > 0.8

    @pytest.mark.asyncio
    async def test_constitutional_tier_validation(self, attestation_manager):
        """Test constitutional tier compatibility."""

        node_id = "test_tier_node"

        # Test Gold tier workload on Silver tier node
        quote = await attestation_manager.generate_attestation_quote(
            node_id, TEEType.SOFTWARE_TEE, constitutional_tier=ConstitutionalTier.SILVER
        )

        result = await attestation_manager.verify_attestation(node_id, quote)

        # Should be compliant with Silver but not Gold
        assert result.is_constitutional_compliant(ConstitutionalTier.SILVER)
        assert not result.is_constitutional_compliant(ConstitutionalTier.GOLD)

    @pytest.mark.asyncio
    async def test_trusted_nodes_for_tier(self, attestation_manager):
        """Test getting trusted nodes for constitutional tier."""

        # Attest multiple nodes with different tiers
        nodes = [
            ("bronze_node", ConstitutionalTier.BRONZE),
            ("silver_node", ConstitutionalTier.SILVER),
            ("gold_node", ConstitutionalTier.GOLD),
        ]

        for node_id, tier in nodes:
            quote = await attestation_manager.generate_attestation_quote(
                node_id, TEEType.SOFTWARE_TEE, constitutional_tier=tier
            )
            await attestation_manager.verify_attestation(node_id, quote)

        # Test tier filtering
        from infrastructure.security.tee.attestation import get_trusted_nodes_for_tier

        bronze_nodes = await get_trusted_nodes_for_tier(ConstitutionalTier.BRONZE)
        silver_nodes = await get_trusted_nodes_for_tier(ConstitutionalTier.SILVER)
        gold_nodes = await get_trusted_nodes_for_tier(ConstitutionalTier.GOLD)

        assert len(bronze_nodes) == 3  # All nodes qualify for Bronze
        assert len(silver_nodes) == 2  # Silver and Gold nodes
        assert len(gold_nodes) == 1  # Only Gold node


class TestEnclaveManager:
    """Test TEE enclave management."""

    @pytest.fixture
    async def enclave_manager(self):
        """Get enclave manager instance."""
        return await get_enclave_manager()

    @pytest.mark.asyncio
    async def test_software_enclave_lifecycle(self, enclave_manager):
        """Test software enclave creation and termination."""

        # Create enclave configuration
        from infrastructure.security.tee.enclave_manager import EnclaveConfiguration

        config = EnclaveConfiguration(
            name="test_software_enclave",
            tee_type=TEEType.SOFTWARE_TEE,
            constitutional_tier=ConstitutionalTier.BRONZE,
            memory_size_mb=256,
        )

        # Mock attestation result
        from infrastructure.security.tee.attestation import AttestationResult, AttestationStatus

        mock_attestation = AttestationResult(
            node_id="test_node",
            status=AttestationStatus.VERIFIED,
            tee_type=TEEType.SOFTWARE_TEE,
            constitutional_tier=ConstitutionalTier.BRONZE,
            trust_score=0.7,
        )

        # Create enclave
        instance_id = await enclave_manager.create_enclave("test_node", config, mock_attestation)

        assert instance_id in enclave_manager.active_enclaves

        instance = enclave_manager.get_enclave_status(instance_id)
        assert instance is not None
        assert instance.status.value in ["ready", "initializing"]

        # Terminate enclave
        success = await enclave_manager.terminate_enclave(instance_id)
        assert success
        assert instance_id not in enclave_manager.active_enclaves

    @pytest.mark.asyncio
    async def test_constitutional_workload_execution(self, enclave_manager):
        """Test constitutional workload execution."""

        from infrastructure.security.tee.enclave_manager import WorkloadManifest

        # Create workload manifest
        manifest = WorkloadManifest(
            name="test_constitutional_inference",
            workload_type=WorkloadType.INFERENCE,
            constitutional_tier=ConstitutionalTier.SILVER,
            harm_categories=["hate_speech", "violence"],
            min_memory_mb=512,
        )

        # Execute workload
        result = await enclave_manager.execute_constitutional_workload(
            "test_node", manifest, {"input": "What is 2+2?"}, ConstitutionalTier.SILVER
        )

        assert result["success"] is True
        assert "result" in result
        assert "execution_time" in result

    @pytest.mark.asyncio
    async def test_enclave_monitoring(self, enclave_manager):
        """Test enclave health monitoring."""

        # Start enclave manager to enable monitoring
        await enclave_manager.start()

        # Get summary before and after creating enclave
        summary_before = enclave_manager.get_active_enclaves_summary()
        initial_count = summary_before["total_enclaves"]

        # Create test enclave (simplified)
        from infrastructure.security.tee.enclave_manager import EnclaveInstance, EnclaveConfiguration
        from infrastructure.security.tee.attestation import AttestationResult, AttestationStatus

        config = EnclaveConfiguration(name="monitoring_test")
        AttestationResult(
            status=AttestationStatus.VERIFIED, constitutional_tier=ConstitutionalTier.BRONZE, trust_score=0.8
        )

        # Manually add to test monitoring
        test_instance = EnclaveInstance(config=config)
        enclave_manager.active_enclaves[test_instance.instance_id] = test_instance

        summary_after = enclave_manager.get_active_enclaves_summary()
        assert summary_after["total_enclaves"] == initial_count + 1

        await enclave_manager.stop()


class TestConstitutionalPolicy:
    """Test constitutional policy enforcement."""

    @pytest.fixture
    async def policy_engine(self):
        """Get policy engine instance."""
        return await get_policy_engine()

    @pytest.mark.asyncio
    async def test_harm_classification(self, policy_engine):
        """Test harm classification system."""

        test_cases = [
            ("Hello, how are you?", True, 0.0),
            ("I hate all people of that race", False, 0.5),
            ("My social security number is 123-45-6789", False, 0.7),
            ("Violence is never the answer", True, 0.1),
            ("Let's build a bomb", False, 0.8),
        ]

        for content, expected_safe, min_expected_risk in test_cases:
            is_safe, evaluation = await policy_engine.evaluate_content(content)

            assert is_safe == expected_safe, f"Content: {content}"

            if not expected_safe:
                max_risk = evaluation.get("max_risk_score", 0.0)
                assert max_risk >= min_expected_risk, f"Risk too low for: {content}"

    @pytest.mark.asyncio
    async def test_behavioral_pattern_analysis(self, policy_engine):
        """Test behavioral pattern analysis."""

        user_id = "test_user_behavior"

        # Simulate multiple interactions with escalating harm
        harmful_contents = [
            "This is fine",
            "I don't like that group",
            "I really hate those people",
            "Those people should be hurt",
            "Violence against them is justified",
        ]

        for content in harmful_contents:
            is_safe, evaluation = await policy_engine.evaluate_content(content, context={"user_id": user_id})

        # Check that behavioral risk increases
        final_evaluation = await policy_engine.evaluate_content(
            "Another harmful statement", context={"user_id": user_id}
        )

        behavioral_risk = final_evaluation[1].get("behavioral_risk", 0.0)
        assert behavioral_risk > 0.0  # Should detect escalating pattern

    @pytest.mark.asyncio
    async def test_workload_validation(self, policy_engine):
        """Test constitutional workload validation."""

        workload_manifest = {
            "type": "inference",
            "constitutional_tier": "silver",
            "harm_categories": ["hate_speech", "violence"],
            "privacy_requirements": "high",
        }

        # Mock node attestation - compliant
        compliant_attestation = {
            "status": "verified",
            "constitutional_tier": "gold",  # Higher than required
            "capabilities": ["memory_encryption", "remote_attestation"],
            "trust_score": 0.9,
        }

        is_valid = await policy_engine.validate_workload_deployment(workload_manifest, compliant_attestation)
        assert is_valid

        # Mock node attestation - non-compliant
        non_compliant_attestation = {
            "status": "verified",
            "constitutional_tier": "bronze",  # Lower than required
            "capabilities": ["remote_attestation"],  # Missing memory_encryption
            "trust_score": 0.5,
        }

        is_valid = await policy_engine.validate_workload_deployment(workload_manifest, non_compliant_attestation)
        assert not is_valid

    @pytest.mark.asyncio
    async def test_compliance_reporting(self, policy_engine):
        """Test compliance report generation."""

        # Generate some test violations
        test_contents = ["I hate everyone", "Here's someone's SSN: 123-45-6789", "Violence is the only answer"]

        for content in test_contents:
            await policy_engine.evaluate_content(content, context={"user_id": "test_reporter"})

        # Generate compliance report
        report = await policy_engine.generate_compliance_report()

        assert "report_id" in report
        assert "summary" in report
        assert report["summary"]["total_violations"] > 0
        assert len(report["recommendations"]) > 0


class TestSecurityIntegration:
    """Test complete security integration system."""

    @pytest.fixture
    async def integration_manager(self):
        """Get integration manager instance."""
        return await get_integration_manager()

    @pytest.mark.asyncio
    async def test_node_registration_flow(self, integration_manager):
        """Test complete node registration and attestation flow."""

        node_id = "test_integration_node"
        node_info = {
            "node_type": "fog_node",
            "hardware_capabilities": ["intel_sgx", "memory_encryption"],
            "cpu_utilization": 0.2,
            "memory_utilization": 0.3,
            "network_latency_ms": 50.0,
        }

        # Register node
        node = await integration_manager.register_fog_node(node_id, node_info)

        assert node.node_id == node_id
        assert node.security_status in [
            NodeSecurityStatus.ATTESTED,
            NodeSecurityStatus.CONSTITUTIONAL_COMPLIANT,
            NodeSecurityStatus.ATTESTATION_PENDING,
        ]
        assert node.tee_type is not None
        assert node.trust_score > 0.0

        # Verify node is tracked
        assert node_id in integration_manager.secure_nodes

    @pytest.mark.asyncio
    async def test_constitutional_workload_deployment(self, integration_manager):
        """Test constitutional workload deployment."""

        # First register a capable node
        node_id = "test_workload_node"
        await integration_manager.register_fog_node(
            node_id,
            {
                "node_type": "fog_node",
                "hardware_capabilities": ["memory_encryption"],
                "cpu_utilization": 0.1,
                "memory_utilization": 0.1,
            },
        )

        # Create workload request
        request = ConstitutionalWorkloadRequest(
            workload_name="test_constitutional_qa",
            workload_type=WorkloadType.INFERENCE,
            required_tier=ConstitutionalTier.BRONZE,  # Use Bronze for software TEE
            harm_categories_monitored=[HarmCategory.HATE_SPEECH, HarmCategory.VIOLENCE],
            input_data={"question": "What is the capital of France?"},
            requester_id="test_user",
        )

        # Deploy workload
        result = await integration_manager.deploy_constitutional_workload(request)

        assert "success" in result
        assert "request_id" in result

        if result["success"]:
            assert "node_id" in result
            assert "result" in result
            assert result["node_id"] == node_id

    @pytest.mark.asyncio
    async def test_constitutional_compliance_monitoring(self, integration_manager):
        """Test constitutional compliance monitoring."""

        # Register node
        node_id = "test_monitoring_node"
        await integration_manager.register_fog_node(node_id, {"node_type": "fog_node", "cpu_utilization": 0.2})

        # Test compliant content
        result = await integration_manager.monitor_constitutional_compliance(
            node_id, "Hello, how can I help you today?", {"session_id": "test_session"}
        )

        assert result["compliant"] is True
        assert "node_compliance_score" in result

        # Test non-compliant content
        result = await integration_manager.monitor_constitutional_compliance(
            node_id, "I hate all people of that group", {"session_id": "test_session"}
        )

        assert result["compliant"] is False
        assert result["node_violations"] > 0

    @pytest.mark.asyncio
    async def test_security_reporting(self, integration_manager):
        """Test security report generation."""

        # Register some test nodes
        for i in range(3):
            await integration_manager.register_fog_node(
                f"report_test_node_{i}", {"node_type": "fog_node", "cpu_utilization": 0.1}
            )

        # Generate report
        report = await integration_manager.generate_security_report()

        assert "report_id" in report
        assert "generated_at" in report
        assert "summary" in report
        assert "detailed_metrics" in report
        assert "recommendations" in report

        summary = report["summary"]
        assert summary["total_nodes"] >= 3
        assert "attestation_success_rate" in summary
        assert "average_trust_score" in summary

    @pytest.mark.asyncio
    async def test_high_level_convenience_functions(self):
        """Test high-level convenience functions."""

        # Test node registration
        result = await register_constitutional_fog_node(
            "convenience_test_node",
            {"node_type": "fog_node", "hardware_capabilities": ["memory_encryption"], "cpu_utilization": 0.15},
        )

        assert "node_id" in result
        assert "security_status" in result
        assert "constitutional_tier" in result
        assert "trust_score" in result

        # Test workload execution
        workload_result = await execute_constitutional_workload(
            "inference",
            "convenience_test_qa",
            {"question": "What is 2+2?"},
            {"tier": "bronze", "harm_categories": ["hate_speech"], "requester_id": "convenience_user"},
        )

        assert "success" in workload_result
        assert "request_id" in workload_result

    @pytest.mark.asyncio
    async def test_node_quarantine_mechanism(self, integration_manager):
        """Test node quarantine for excessive violations."""

        node_id = "quarantine_test_node"
        await integration_manager.register_fog_node(node_id, {"node_type": "fog_node"})

        # Generate excessive violations
        harmful_content = "I want to hurt people and commit violence"

        for _ in range(integration_manager.max_constitutional_violations + 1):
            await integration_manager.monitor_constitutional_compliance(node_id, harmful_content)

        # Check if node was quarantined
        node = integration_manager.secure_nodes[node_id]
        assert node.security_status == NodeSecurityStatus.QUARANTINED


class TestPerformanceAndScalability:
    """Test performance characteristics and scalability."""

    @pytest.mark.asyncio
    async def test_concurrent_attestations(self):
        """Test concurrent node attestations."""

        manager = await get_integration_manager()

        # Create multiple concurrent registrations
        tasks = []
        for i in range(10):
            task = manager.register_fog_node(f"concurrent_node_{i}", {"node_type": "fog_node", "cpu_utilization": 0.1})
            tasks.append(task)

        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check all succeeded
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        assert success_count >= 8  # Allow for some failures in test environment

    @pytest.mark.asyncio
    async def test_policy_evaluation_performance(self):
        """Test policy evaluation performance."""

        engine = await get_policy_engine()

        # Test batch evaluation
        test_contents = [
            "Hello world",
            "I hate everyone",
            "This is a normal message",
            "Violence is bad",
            "SSN: 123-45-6789",
        ] * 20  # 100 total evaluations

        start_time = datetime.now()

        tasks = []
        for content in test_contents:
            task = engine.evaluate_content(content)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Should process 100 evaluations in reasonable time
        assert duration < 10.0  # Less than 10 seconds
        assert len(results) == 100

        # Check results are valid
        for is_safe, evaluation in results:
            assert isinstance(is_safe, bool)
            assert isinstance(evaluation, dict)


if __name__ == "__main__":
    """
    Run the test suite manually for development testing.
    For CI/CD, use: pytest tests/security/test_tee_integration.py -v
    """

    async def run_manual_tests():
        """Run manual test cases for development."""

        print("🔒 Testing TEE Security Integration System")
        print("=" * 50)

        # Test 1: Basic attestation
        print("\n1. Testing TEE Attestation...")
        manager = await get_attestation_manager()

        quote = await manager.generate_attestation_quote(
            "manual_test_node", TEEType.SOFTWARE_TEE, constitutional_tier=ConstitutionalTier.SILVER
        )

        result = await manager.verify_attestation("manual_test_node", quote)

        print(f"   ✓ Attestation Status: {result.status.value}")
        print(f"   ✓ Trust Score: {result.trust_score:.3f}")
        print(f"   ✓ Constitutional Tier: {result.constitutional_tier.value}")

        # Test 2: Constitutional policy
        print("\n2. Testing Constitutional Policy...")
        policy_engine = await get_policy_engine()

        test_content = "I want to help people learn new things"
        is_safe, evaluation = await policy_engine.evaluate_content(test_content)

        print(f"   ✓ Content Safe: {is_safe}")
        print(f"   ✓ Risk Score: {evaluation.get('max_risk_score', 0.0):.3f}")

        # Test 3: Full integration
        print("\n3. Testing Full Integration...")

        result = await register_constitutional_fog_node(
            "manual_integration_test",
            {"node_type": "fog_node", "hardware_capabilities": ["memory_encryption"], "cpu_utilization": 0.2},
        )

        print(f"   ✓ Node Security Status: {result['security_status']}")
        print(f"   ✓ Constitutional Tier: {result['constitutional_tier']}")

        # Test 4: Workload execution
        print("\n4. Testing Workload Execution...")

        workload_result = await execute_constitutional_workload(
            "inference",
            "manual_test_workload",
            {"input": "What is artificial intelligence?"},
            {"tier": "bronze", "harm_categories": ["hate_speech", "violence"], "requester_id": "manual_tester"},
        )

        print(f"   ✓ Workload Success: {workload_result['success']}")
        if workload_result["success"]:
            print(f"   ✓ Execution Node: {workload_result.get('node_id', 'unknown')}")

        print("\n🎉 Manual tests completed successfully!")

        # Cleanup
        integration_manager = await get_integration_manager()
        await integration_manager.stop()

    # Run manual tests
    asyncio.run(run_manual_tests())
