"""
Integration Tests for Zero-Knowledge Predicates

Comprehensive test suite for ZK predicate engine covering:
- Individual predicate functionality
- Audit system integration
- Workflow orchestration
- Performance benchmarks
- Privacy guarantee validation

Tests are designed to validate both correctness and privacy properties
of the ZK predicate system in fog computing scenarios.
"""

from datetime import datetime, timedelta, timezone
import json
import os

# Import ZK predicate system components
import sys
import tempfile
import time

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from infrastructure.fog.zk.zk_audit_integration import ZKAuditIntegration, ZKPredicateWorkflow
from infrastructure.fog.zk.zk_expansion_roadmap import ZKPredicateExpansionRoadmap
from infrastructure.fog.zk.zk_predicates import PredicateContext, ProofResult, ZKPredicateEngine


class TestZKPredicateEngine:
    """Test core ZK predicate engine functionality."""

    @pytest.fixture
    def zk_engine(self):
        """Create ZK predicate engine for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = ZKPredicateEngine(node_id="test_node", data_dir=tmpdir)
            yield engine

    @pytest.fixture
    def test_context(self):
        """Create test predicate context."""
        return PredicateContext(
            network_policies={"strict_mode": False},
            allowed_mime_types={"text/plain", "application/json", "image/jpeg"},
            trusted_model_hashes={"abc123", "def456", "ghi789"},
            compliance_rules={"min_consent_percentage": 90.0},
        )

    @pytest.mark.asyncio
    async def test_network_policy_predicate_full_workflow(self, zk_engine, test_context):
        """Test complete network policy predicate workflow."""

        # Test network configuration
        network_config = {
            "services": [
                {"name": "api", "port": 8080, "protocol": "tcp"},
                {"name": "websocket", "port": 8081, "protocol": "tcp"},
                {"name": "metrics", "port": 9090, "protocol": "tcp"},
            ]
        }

        # Public parameters for policy compliance
        policy_params = {
            "allowed_protocols": ["tcp", "udp"],
            "allowed_port_ranges": ["registered", "dynamic"],
            "max_services": 5,
        }

        # Generate commitment
        commitment_id = await zk_engine.generate_commitment(
            predicate_id="network_policy", secret_data=network_config, context=test_context
        )

        assert commitment_id is not None
        assert commitment_id in zk_engine.commitments

        # Generate proof
        proof_id = await zk_engine.generate_proof(
            commitment_id=commitment_id,
            predicate_id="network_policy",
            secret_data=network_config,
            public_parameters=policy_params,
        )

        assert proof_id is not None
        assert proof_id in zk_engine.proofs

        # Verify proof
        verification_result = await zk_engine.verify_proof(
            proof_id=proof_id, public_parameters=policy_params, context=test_context
        )

        assert verification_result == ProofResult.VALID

        # Check proof properties
        proof = zk_engine.proofs[proof_id]
        assert proof.is_valid()
        assert proof.verification_result == ProofResult.VALID
        assert proof.verifier_id == "test_node"
        assert "total_compliance" in proof.proof_data

    @pytest.mark.asyncio
    async def test_mime_type_predicate_validation(self, zk_engine, test_context):
        """Test MIME type predicate with various content types."""

        # Test valid MIME type
        file_metadata = {"mime_type": "application/json", "size": 1024, "extension": ".json"}

        content_policy = {
            "allowed_mime_types": ["application/json", "text/plain"],
            "max_file_size": 10 * 1024 * 1024,  # 10MB
        }

        commitment_id = await zk_engine.generate_commitment(
            predicate_id="mime_type", secret_data=file_metadata, context=test_context
        )

        proof_id = await zk_engine.generate_proof(
            commitment_id=commitment_id,
            predicate_id="mime_type",
            secret_data=file_metadata,
            public_parameters=content_policy,
        )

        result = await zk_engine.verify_proof(proof_id=proof_id, public_parameters=content_policy, context=test_context)

        assert result == ProofResult.VALID

        # Test invalid MIME type
        invalid_metadata = {
            "mime_type": "application/executable",  # Not in allowed list
            "size": 1024,
            "extension": ".exe",
        }

        invalid_commitment_id = await zk_engine.generate_commitment(
            predicate_id="mime_type", secret_data=invalid_metadata, context=test_context
        )

        invalid_proof_id = await zk_engine.generate_proof(
            commitment_id=invalid_commitment_id,
            predicate_id="mime_type",
            secret_data=invalid_metadata,
            public_parameters=content_policy,
        )

        invalid_result = await zk_engine.verify_proof(
            proof_id=invalid_proof_id, public_parameters=content_policy, context=test_context
        )

        assert invalid_result == ProofResult.VALID  # Proof structure is valid

        # But compliance should be false
        invalid_proof = zk_engine.proofs[invalid_proof_id]
        assert invalid_proof.proof_data["total_compliance"] is False

    @pytest.mark.asyncio
    async def test_model_hash_predicate_integrity(self, zk_engine, test_context):
        """Test model hash predicate for ML model integrity."""

        # Test trusted model
        model_metadata = {
            "model_hash": "abc123",  # In trusted set
            "model_type": "classification",
            "size_bytes": 50 * 1024 * 1024,  # 50MB
        }

        trusted_models = {
            "trusted_model_hashes": ["abc123", "def456"],
            "allowed_model_types": ["classification", "regression"],
            "max_model_size": 100 * 1024 * 1024,  # 100MB
        }

        commitment_id = await zk_engine.generate_commitment(
            predicate_id="model_hash", secret_data=model_metadata, context=test_context
        )

        proof_id = await zk_engine.generate_proof(
            commitment_id=commitment_id,
            predicate_id="model_hash",
            secret_data=model_metadata,
            public_parameters=trusted_models,
        )

        result = await zk_engine.verify_proof(proof_id=proof_id, public_parameters=trusted_models, context=test_context)

        assert result == ProofResult.VALID

        # Verify proof contains expected claims
        proof = zk_engine.proofs[proof_id]
        assert proof.proof_data["hash_compliance"] is True
        assert proof.proof_data["type_compliance"] is True
        assert proof.proof_data["size_compliance"] is True
        assert proof.proof_data["total_compliance"] is True

    @pytest.mark.asyncio
    async def test_compliance_predicate_privacy_preservation(self, zk_engine, test_context):
        """Test compliance predicate privacy preservation."""

        # Sensitive compliance data
        compliance_data = {
            "data_retention_days": 730,  # 2 years
            "user_consent_percentage": 95.5,
            "security_score": 0.92,
            "audit_findings_count": 2,
        }

        compliance_requirements = {
            "min_consent_percentage": 90.0,
            "max_retention_days": 1095,  # 3 years
            "min_security_score": 0.8,
            "max_audit_findings": 5,
        }

        commitment_id = await zk_engine.generate_commitment(
            predicate_id="compliance_check", secret_data=compliance_data, context=test_context
        )

        proof_id = await zk_engine.generate_proof(
            commitment_id=commitment_id,
            predicate_id="compliance_check",
            secret_data=compliance_data,
            public_parameters=compliance_requirements,
        )

        result = await zk_engine.verify_proof(
            proof_id=proof_id, public_parameters=compliance_requirements, context=test_context
        )

        assert result == ProofResult.VALID

        # Verify privacy preservation - exact values should not be in proof
        proof = zk_engine.proofs[proof_id]
        proof_data_str = json.dumps(proof.proof_data)

        # Exact sensitive values should not appear in proof
        assert "95.5" not in proof_data_str  # Exact consent percentage
        assert "0.92" not in proof_data_str  # Exact security score
        assert "730" not in proof_data_str  # Exact retention days

        # But categories should be present
        assert "retention_category" in proof.proof_data.get("compliance_categories", {})
        assert "consent_category" in proof.proof_data.get("compliance_categories", {})

    @pytest.mark.asyncio
    async def test_commitment_expiration(self, zk_engine, test_context):
        """Test commitment expiration handling."""

        network_config = {"services": [{"name": "test", "port": 8080, "protocol": "tcp"}]}

        # Generate commitment with short TTL
        commitment_id = await zk_engine.generate_commitment(
            predicate_id="network_policy",
            secret_data=network_config,
            context=test_context,
            ttl_hours=0,  # Immediate expiration
        )

        # Manually expire the commitment
        zk_engine.commitments[commitment_id].expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

        # Attempt to generate proof with expired commitment should fail
        with pytest.raises(ValueError, match="expired"):
            await zk_engine.generate_proof(
                commitment_id=commitment_id,
                predicate_id="network_policy",
                secret_data=network_config,
                public_parameters={"allowed_protocols": ["tcp"]},
            )

    @pytest.mark.asyncio
    async def test_zk_engine_statistics(self, zk_engine, test_context):
        """Test ZK engine statistics and metrics."""

        # Generate several proofs
        for i in range(3):
            network_config = {"services": [{"name": f"service_{i}", "port": 8080 + i, "protocol": "tcp"}]}
            policy_params = {"allowed_protocols": ["tcp"], "allowed_port_ranges": ["registered"], "max_services": 10}

            commitment_id = await zk_engine.generate_commitment(
                predicate_id="network_policy", secret_data=network_config, context=test_context
            )

            proof_id = await zk_engine.generate_proof(
                commitment_id=commitment_id,
                predicate_id="network_policy",
                secret_data=network_config,
                public_parameters=policy_params,
            )

            await zk_engine.verify_proof(proof_id=proof_id, public_parameters=policy_params, context=test_context)

        # Check statistics
        stats = await zk_engine.get_proof_stats()

        assert stats["total_commitments"] == 3
        assert stats["total_proofs"] == 3
        assert stats["verified_proofs"] == 3
        assert stats["valid_proofs"] == 3
        assert stats["verification_rate"] == 1.0
        assert stats["validity_rate"] == 1.0
        assert "network_policy" in stats["proofs_by_predicate"]
        assert stats["proofs_by_predicate"]["network_policy"] == 3

    @pytest.mark.asyncio
    async def test_cleanup_operations(self, zk_engine, test_context):
        """Test cleanup of expired commitments and old proofs."""

        # Generate commitments and proofs
        for i in range(5):
            network_config = {"services": [{"name": f"service_{i}", "port": 8080 + i, "protocol": "tcp"}]}

            await zk_engine.generate_commitment(
                predicate_id="network_policy", secret_data=network_config, context=test_context, ttl_hours=1
            )

        # Manually expire some commitments
        commitment_ids = list(zk_engine.commitments.keys())
        for i in range(2):
            zk_engine.commitments[commitment_ids[i]].expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

        # Run cleanup
        cleaned_count = await zk_engine.cleanup_expired()

        assert cleaned_count == 2  # Should clean up 2 expired commitments
        assert len(zk_engine.commitments) == 3  # 5 - 2 = 3


class TestZKAuditIntegration:
    """Test ZK predicate audit system integration."""

    @pytest.fixture
    def audit_integration(self):
        """Create audit integration for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zk_engine = ZKPredicateEngine(node_id="audit_test_node", data_dir=tmpdir)
            integration = ZKAuditIntegration(zk_engine=zk_engine)
            yield integration

    @pytest.mark.asyncio
    async def test_network_policy_audit_workflow(self, audit_integration):
        """Test complete network policy audit workflow."""

        network_config = {
            "services": [
                {"name": "api", "port": 8080, "protocol": "tcp"},
                {"name": "db", "port": 5432, "protocol": "tcp"},
            ]
        }

        policy_params = {"allowed_protocols": ["tcp"], "allowed_port_ranges": ["registered"], "max_services": 10}

        # Execute compliance verification
        is_compliant, proof_id = await audit_integration.verify_network_policy_compliance(
            network_config=network_config, policy_parameters=policy_params, entity_id="test_entity_123"
        )

        assert is_compliant is True
        assert proof_id is not None

        # Check audit events were recorded
        assert len(audit_integration.audit_events) >= 2  # commitment + verification events

        # Verify audit event structure
        for event in audit_integration.audit_events:
            assert "event_id" in event
            assert "event_type" in event
            assert "predicate_type" in event
            assert "entity_id_hash" in event
            assert "timestamp" in event
            assert event["node_id"] == "audit_test_node"

    @pytest.mark.asyncio
    async def test_batch_compliance_verification(self, audit_integration):
        """Test batch compliance verification."""

        verification_requests = [
            {
                "type": "network_policy",
                "secret_data": {"services": [{"name": "api", "port": 8080, "protocol": "tcp"}]},
                "public_parameters": {
                    "allowed_protocols": ["tcp"],
                    "allowed_port_ranges": ["registered"],
                    "max_services": 5,
                },
                "entity_id": "entity_1",
            },
            {
                "type": "mime_type",
                "secret_data": {"mime_type": "application/json", "size": 1024, "extension": ".json"},
                "public_parameters": {"allowed_mime_types": ["application/json"], "max_file_size": 10240},
                "entity_id": "entity_2",
            },
            {
                "type": "compliance_check",
                "secret_data": {
                    "data_retention_days": 365,
                    "user_consent_percentage": 95.0,
                    "security_score": 0.9,
                    "audit_findings_count": 1,
                },
                "public_parameters": {
                    "min_consent_percentage": 90.0,
                    "max_retention_days": 1095,
                    "min_security_score": 0.8,
                    "max_audit_findings": 5,
                },
                "entity_id": "entity_3",
            },
        ]

        results = await audit_integration.batch_verify_compliance(verification_requests)

        assert len(results) == 3
        assert "entity_1" in results
        assert "entity_2" in results
        assert "entity_3" in results

        # All should be compliant
        for entity_id, (is_compliant, proof_id) in results.items():
            assert is_compliant is True
            assert proof_id is not None

    @pytest.mark.asyncio
    async def test_compliance_report_generation(self, audit_integration):
        """Test compliance report generation."""

        # Generate some audit events
        await audit_integration.verify_network_policy_compliance(
            network_config={"services": [{"name": "test", "port": 8080, "protocol": "tcp"}]},
            policy_parameters={"allowed_protocols": ["tcp"], "allowed_port_ranges": ["registered"], "max_services": 5},
            entity_id="report_test_entity",
        )

        # Generate compliance report
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc) + timedelta(hours=1)

        report = await audit_integration.generate_compliance_report(start_time=start_time, end_time=end_time)

        assert "reporting_period" in report
        assert "total_zk_operations" in report
        assert report["total_zk_operations"] > 0
        assert "operations_by_type" in report
        assert "verification_results" in report
        assert "zk_engine_stats" in report

    @pytest.mark.asyncio
    async def test_audit_data_privacy_preservation(self, audit_integration):
        """Test that audit data preserves privacy."""

        sensitive_entity_id = "very_sensitive_entity_12345"

        await audit_integration.record_zk_audit_event(
            event_type="test_event",
            predicate_type=audit_integration.zk_engine.predicates["network_policy"].predicate_type,
            entity_id=sensitive_entity_id,
            privacy_level="high",
            additional_context={"sensitive_data": "top_secret_info", "user_id": "user_12345", "password": "secret123"},
        )

        # Check that sensitive data is not in audit events
        audit_events_str = json.dumps(audit_integration.audit_events)
        assert sensitive_entity_id not in audit_events_str  # Original entity ID should be hashed
        assert "top_secret_info" not in audit_events_str
        assert "user_12345" not in audit_events_str
        assert "secret123" not in audit_events_str

        # But audit event should exist
        assert len(audit_integration.audit_events) > 0
        event = audit_integration.audit_events[-1]  # Latest event
        assert event["event_type"] == "test_event"
        assert event["privacy_level"] == "high"
        assert len(event["entity_id_hash"]) == 16  # Hash should be 16 chars


class TestZKPredicateWorkflow:
    """Test ZK predicate workflow orchestration."""

    @pytest.fixture
    def workflow_orchestrator(self):
        """Create workflow orchestrator for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zk_engine = ZKPredicateEngine(node_id="workflow_test_node", data_dir=tmpdir)
            audit_integration = ZKAuditIntegration(zk_engine=zk_engine)
            orchestrator = ZKPredicateWorkflow(audit_integration=audit_integration)
            yield orchestrator

    @pytest.mark.asyncio
    async def test_fog_node_onboarding_workflow(self, workflow_orchestrator):
        """Test complete fog node onboarding workflow."""

        verification_data = {
            "network_policy": {
                "secret_data": {
                    "services": [
                        {"name": "fog_api", "port": 8080, "protocol": "tcp"},
                        {"name": "fog_sync", "port": 8081, "protocol": "tcp"},
                    ]
                },
                "public_parameters": {
                    "allowed_protocols": ["tcp", "udp"],
                    "allowed_port_ranges": ["registered", "dynamic"],
                    "max_services": 10,
                },
            },
            "compliance_check": {
                "secret_data": {
                    "data_retention_days": 365,
                    "user_consent_percentage": 95.0,
                    "security_score": 0.9,
                    "audit_findings_count": 0,
                },
                "public_parameters": {
                    "min_consent_percentage": 90.0,
                    "max_retention_days": 1095,
                    "min_security_score": 0.8,
                    "max_audit_findings": 3,
                },
            },
        }

        # Execute onboarding workflow
        results = await workflow_orchestrator.execute_workflow(
            workflow_name="fog_node_onboarding", entity_id="new_fog_node_001", verification_data=verification_data
        )

        assert results["workflow_name"] == "fog_node_onboarding"
        assert results["entity_id"] == "new_fog_node_001"
        assert results["overall_success"] is True
        assert len(results["steps"]) >= 2  # network_policy and compliance_check
        assert len(results["proof_ids"]) >= 2

        # Check individual steps
        network_step = next(step for step in results["steps"] if step["type"] == "network_policy")
        compliance_step = next(step for step in results["steps"] if step["type"] == "compliance_check")

        assert network_step["success"] is True
        assert compliance_step["success"] is True
        assert network_step["proof_id"] is not None
        assert compliance_step["proof_id"] is not None

    @pytest.mark.asyncio
    async def test_content_processing_workflow(self, workflow_orchestrator):
        """Test content processing verification workflow."""

        verification_data = {
            "mime_type": {
                "secret_data": {"mime_type": "application/json", "size": 2048, "extension": ".json"},
                "public_parameters": {
                    "allowed_mime_types": ["application/json", "text/plain"],
                    "max_file_size": 10 * 1024 * 1024,
                },
            },
            "compliance_check": {
                "secret_data": {
                    "data_retention_days": 30,
                    "user_consent_percentage": 100.0,
                    "security_score": 0.95,
                    "audit_findings_count": 0,
                },
                "public_parameters": {
                    "min_consent_percentage": 95.0,
                    "max_retention_days": 90,
                    "min_security_score": 0.9,
                    "max_audit_findings": 0,
                },
            },
        }

        results = await workflow_orchestrator.execute_workflow(
            workflow_name="content_processing", entity_id="content_batch_001", verification_data=verification_data
        )

        assert results["overall_success"] is True
        assert len(results["proof_ids"]) == 2  # Both required steps completed

    @pytest.mark.asyncio
    async def test_custom_workflow_registration(self, workflow_orchestrator):
        """Test custom workflow registration and execution."""

        # Register custom workflow
        custom_workflow = {
            "name": "Custom Security Workflow",
            "description": "Custom security verification workflow",
            "steps": [{"type": "network_policy", "required": True}, {"type": "model_hash", "required": True}],
        }

        workflow_orchestrator.register_workflow("custom_security", custom_workflow)

        # Verify workflow was registered
        available_workflows = workflow_orchestrator.get_available_workflows()
        assert "custom_security" in available_workflows

        # Execute custom workflow
        verification_data = {
            "network_policy": {
                "secret_data": {"services": [{"name": "secure_api", "port": 443, "protocol": "tcp"}]},
                "public_parameters": {
                    "allowed_protocols": ["tcp"],
                    "allowed_port_ranges": ["system"],
                    "max_services": 1,
                },
            },
            "model_hash": {
                "secret_data": {"model_hash": "secure_model_hash", "model_type": "security", "size_bytes": 1024},
                "public_parameters": {
                    "trusted_model_hashes": ["secure_model_hash"],
                    "allowed_model_types": ["security"],
                    "max_model_size": 2048,
                },
            },
        }

        results = await workflow_orchestrator.execute_workflow(
            workflow_name="custom_security", entity_id="custom_test_entity", verification_data=verification_data
        )

        assert results["overall_success"] is True
        assert len(results["steps"]) == 2


class TestZKExpansionRoadmap:
    """Test ZK predicate expansion roadmap functionality."""

    @pytest.fixture
    def roadmap(self):
        """Create expansion roadmap for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            roadmap = ZKPredicateExpansionRoadmap(config_dir=tmpdir)
            yield roadmap

    def test_roadmap_initialization(self, roadmap):
        """Test roadmap initialization with default predicates."""

        assert len(roadmap.predicate_specs) > 0
        assert len(roadmap.milestones) > 0

        # Check for expected default predicates
        expected_predicates = ["range_proofs", "threshold_signatures", "private_set_intersection", "zk_ml_inference"]
        for pred_name in expected_predicates:
            assert pred_name in roadmap.predicate_specs

    def test_roadmap_status_summary(self, roadmap):
        """Test roadmap status summary generation."""

        status = roadmap.get_roadmap_status()

        assert "total_predicates" in status
        assert "predicates_by_phase" in status
        assert "total_milestones" in status
        assert "milestones_by_status" in status
        assert "overdue_milestones" in status
        assert status["total_predicates"] > 0
        assert status["total_milestones"] > 0

    def test_predicate_dependency_validation(self, roadmap):
        """Test predicate dependency validation."""

        issues = roadmap.validate_dependencies()

        # Should have minimal issues with default roadmap
        # (Dependencies should be properly configured)
        for predicate, predicate_issues in issues.items():
            print(f"Predicate {predicate} has issues: {predicate_issues}")

        # Most predicates should not have missing dependencies
        missing_dep_issues = [
            issue for issues_list in issues.values() for issue in issues_list if "Missing dependency" in issue
        ]
        assert len(missing_dep_issues) == 0, f"Missing dependencies: {missing_dep_issues}"

    def test_roadmap_report_generation(self, roadmap):
        """Test roadmap report generation."""

        report = roadmap.generate_roadmap_report()

        assert "# ZK Predicate Expansion Roadmap Report" in report
        assert "## Current Status" in report
        assert "## Predicates by Phase" in report
        assert "Total Predicates:" in report
        assert "Total Milestones:" in report


class TestZKPerformanceBenchmarks:
    """Performance benchmarks for ZK predicate system."""

    @pytest.fixture
    def performance_engine(self):
        """Create ZK engine for performance testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = ZKPredicateEngine(node_id="perf_test_node", data_dir=tmpdir)
            yield engine

    @pytest.mark.asyncio
    async def test_commitment_generation_performance(self, performance_engine):
        """Benchmark commitment generation performance."""

        network_config = {
            "services": [{"name": f"service_{i}", "port": 8000 + i, "protocol": "tcp"} for i in range(10)]
        }

        context = PredicateContext()

        # Benchmark commitment generation
        start_time = time.time()
        iterations = 100

        for i in range(iterations):
            await performance_engine.generate_commitment(
                predicate_id="network_policy", secret_data=network_config, context=context
            )

        end_time = time.time()
        avg_time_ms = ((end_time - start_time) / iterations) * 1000

        print(f"Average commitment generation time: {avg_time_ms:.2f}ms")

        # Performance target: under 50ms per commitment
        assert avg_time_ms < 50, f"Commitment generation too slow: {avg_time_ms:.2f}ms"

    @pytest.mark.asyncio
    async def test_proof_verification_performance(self, performance_engine):
        """Benchmark proof verification performance."""

        # Generate some proofs first
        proofs = []
        for i in range(10):
            network_config = {"services": [{"name": f"service_{i}", "port": 8000 + i, "protocol": "tcp"}]}
            policy_params = {"allowed_protocols": ["tcp"], "allowed_port_ranges": ["registered"], "max_services": 20}

            commitment_id = await performance_engine.generate_commitment(
                predicate_id="network_policy", secret_data=network_config, context=PredicateContext()
            )

            proof_id = await performance_engine.generate_proof(
                commitment_id=commitment_id,
                predicate_id="network_policy",
                secret_data=network_config,
                public_parameters=policy_params,
            )
            proofs.append((proof_id, policy_params))

        # Benchmark verification
        start_time = time.time()

        for proof_id, policy_params in proofs:
            result = await performance_engine.verify_proof(
                proof_id=proof_id, public_parameters=policy_params, context=PredicateContext()
            )
            assert result == ProofResult.VALID

        end_time = time.time()
        avg_verification_time_ms = ((end_time - start_time) / len(proofs)) * 1000

        print(f"Average proof verification time: {avg_verification_time_ms:.2f}ms")

        # Performance target: under 30ms per verification
        assert avg_verification_time_ms < 30, f"Proof verification too slow: {avg_verification_time_ms:.2f}ms"

    @pytest.mark.asyncio
    async def test_batch_operation_scaling(self, performance_engine):
        """Test performance scaling of batch operations."""

        batch_sizes = [1, 5, 10, 25, 50]
        results = {}

        for batch_size in batch_sizes:
            # Generate batch of network configs
            network_configs = [
                {"services": [{"name": f"service_{i}_{j}", "port": 8000 + i * 100 + j, "protocol": "tcp"}]}
                for j in range(batch_size)
            ]

            # Measure batch processing time
            start_time = time.time()

            commitment_ids = []
            for config in network_configs:
                commitment_id = await performance_engine.generate_commitment(
                    predicate_id="network_policy", secret_data=config, context=PredicateContext()
                )
                commitment_ids.append(commitment_id)

            end_time = time.time()
            batch_time = (end_time - start_time) * 1000  # Convert to ms
            avg_time_per_item = batch_time / batch_size

            results[batch_size] = {"total_time_ms": batch_time, "avg_time_per_item_ms": avg_time_per_item}

            print(f"Batch size {batch_size}: {batch_time:.2f}ms total, {avg_time_per_item:.2f}ms per item")

        # Verify that scaling is reasonable (not exponential)
        # Performance should not degrade significantly with larger batches
        small_batch_avg = results[1]["avg_time_per_item_ms"]
        large_batch_avg = results[50]["avg_time_per_item_ms"]

        # Performance degradation should be less than 50%
        performance_degradation = (large_batch_avg - small_batch_avg) / small_batch_avg
        assert performance_degradation < 0.5, f"Performance degrades too much with scale: {performance_degradation:.2%}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
