"""
ZK Predicate Integration Examples and Demonstration Tests

Real-world integration examples showing how ZK predicates work
in practical fog computing scenarios:

1. Fog Node Onboarding with Privacy
2. Federated Learning Model Verification
3. Content Processing Pipeline Security
4. Multi-tenant Compliance Verification
5. Edge Computing Resource Allocation

These examples demonstrate the practical utility of ZK predicates
while maintaining privacy guarantees in fog computing environments.
"""

import asyncio
import json
import os
from pathlib import Path

# Import ZK predicate system components
import sys
import tempfile
import time

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from infrastructure.fog.zk.zk_audit_integration import ZKAuditIntegration, ZKPredicateWorkflow
from infrastructure.fog.zk.zk_predicates import ZKPredicateEngine


class TestRealWorldScenarios:
    """Test ZK predicates in realistic fog computing scenarios."""

    @pytest.fixture
    def fog_infrastructure(self):
        """Create complete fog computing ZK infrastructure for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Core ZK engine
            zk_engine = ZKPredicateEngine(node_id="fog_coordinator", data_dir=tmpdir)

            # Audit integration
            audit_integration = ZKAuditIntegration(zk_engine=zk_engine)

            # Workflow orchestrator
            workflow = ZKPredicateWorkflow(audit_integration=audit_integration)

            yield {"zk_engine": zk_engine, "audit": audit_integration, "workflow": workflow}

    @pytest.mark.asyncio
    async def test_scenario_fog_node_onboarding_privacy(self, fog_infrastructure):
        """
        Scenario: New fog node joins network with privacy-preserving verification

        Requirements:
        - Verify network configuration compliance without revealing internal details
        - Ensure data retention policies are met without exposing sensitive metrics
        - Validate node capabilities while maintaining competitive privacy
        """

        workflow = fog_infrastructure["workflow"]

        # Simulate new fog node with sensitive configuration
        fog_node_config = {
            "node_id": "edge_node_sf_001",
            "location": "San Francisco Data Center",  # Sensitive business info
            "network_config": {
                "services": [
                    {"name": "ml_inference_api", "port": 8080, "protocol": "tcp"},
                    {"name": "federated_sync", "port": 8443, "protocol": "tcp"},
                    {"name": "health_monitor", "port": 9090, "protocol": "tcp"},
                ]
            },
            "compliance_metrics": {
                "data_retention_days": 1095,  # 3 years - sensitive policy info
                "user_consent_percentage": 98.5,  # High compliance rate
                "security_score": 0.94,  # Internal security assessment
                "audit_findings_count": 1,  # Recent audit results
            },
        }

        # Public network policies (what everyone can see)
        network_policies = {
            "allowed_protocols": ["tcp", "udp", "https"],
            "allowed_port_ranges": ["registered", "dynamic"],
            "max_services": 10,
            "security_requirements": ["tls_1.3", "certificate_pinning"],
        }

        # Public compliance requirements (regulatory requirements)
        compliance_requirements = {
            "min_consent_percentage": 95.0,
            "max_retention_days": 1095,  # 3 years max
            "min_security_score": 0.85,
            "max_audit_findings": 3,
        }

        # Execute privacy-preserving onboarding
        verification_data = {
            "network_policy": {"secret_data": fog_node_config["network_config"], "public_parameters": network_policies},
            "compliance_check": {
                "secret_data": fog_node_config["compliance_metrics"],
                "public_parameters": compliance_requirements,
            },
        }

        results = await workflow.execute_workflow(
            workflow_name="fog_node_onboarding",
            entity_id=fog_node_config["node_id"],
            verification_data=verification_data,
        )

        # Verify successful onboarding
        assert results["overall_success"] is True
        assert len(results["proof_ids"]) == 2  # Network + compliance proofs

        # Verify privacy preservation
        # Sensitive details should not be visible in audit trail
        audit_events = workflow.audit_integration.audit_events
        audit_data_str = json.dumps(audit_events)

        # Sensitive business information should be hidden
        assert "San Francisco Data Center" not in audit_data_str
        assert "98.5" not in audit_data_str  # Exact compliance percentage
        assert "0.94" not in audit_data_str  # Exact security score

        # But verification should have succeeded
        assert all(step["success"] for step in results["steps"] if step["required"])

        print(f"✅ Fog node {fog_node_config['node_id']} onboarded with privacy preservation")

    @pytest.mark.asyncio
    async def test_scenario_federated_learning_model_verification(self, fog_infrastructure):
        """
        Scenario: Federated learning participant submits model for aggregation

        Requirements:
        - Verify model integrity without revealing model weights
        - Ensure model quality metrics meet thresholds privately
        - Validate training data compliance without exposing datasets
        """

        audit_integration = fog_infrastructure["audit"]

        # Simulate federated learning participant model
        participant_model = {
            "participant_id": "hospital_boston_001",  # Sensitive participant identity
            "model_metadata": {
                "model_hash": "fed_model_round_15_boston_abc123def456",  # Contains location info
                "model_type": "medical_diagnosis",
                "size_bytes": 250 * 1024 * 1024,  # 250MB model
                "training_samples": 50000,  # Sensitive dataset size
                "accuracy_score": 0.94,  # Competitive model quality
                "training_epochs": 100,
            },
            "privacy_metrics": {
                "differential_privacy_epsilon": 1.5,  # DP parameter
                "k_anonymity_level": 5,  # Privacy level
                "data_retention_days": 0,  # No raw data retained
                "user_consent_percentage": 100.0,  # Medical data requires 100%
            },
        }

        # Federated learning coordinator requirements
        fl_model_requirements = {
            "trusted_model_hashes": [
                "fed_model_round_15_boston_abc123def456",
                "fed_model_round_15_chicago_def789ghi012",
                "fed_model_round_15_seattle_ghi345jkl678",
            ],
            "allowed_model_types": ["medical_diagnosis", "health_prediction"],
            "max_model_size": 500 * 1024 * 1024,  # 500MB limit
            "min_training_samples": 1000,  # Quality threshold
            "min_accuracy": 0.85,
        }

        fl_privacy_requirements = {
            "max_dp_epsilon": 2.0,  # Strong DP requirement
            "min_k_anonymity": 3,
            "max_retention_days": 0,  # No raw data retention allowed
            "min_consent_percentage": 100.0,  # Medical data requirement
        }

        # Verify model integrity (without revealing model details)
        model_valid, model_proof = await audit_integration.verify_model_integrity(
            model_metadata=participant_model["model_metadata"],
            trusted_models=fl_model_requirements,
            entity_id=participant_model["participant_id"],
        )

        # Verify privacy compliance (without revealing sensitive metrics)
        privacy_valid, privacy_proof = await audit_integration.verify_privacy_compliance(
            compliance_data=participant_model["privacy_metrics"],
            compliance_requirements=fl_privacy_requirements,
            entity_id=participant_model["participant_id"],
        )

        # Both verifications should succeed
        assert model_valid is True
        assert privacy_valid is True
        assert model_proof is not None
        assert privacy_proof is not None

        # Verify privacy preservation in audit trail
        audit_events = audit_integration.audit_events
        audit_str = json.dumps(audit_events)

        # Sensitive information should be protected
        assert "hospital_boston_001" not in audit_str  # Participant identity hashed
        assert "50000" not in audit_str  # Training sample count hidden
        assert "0.94" not in audit_str  # Exact accuracy score hidden
        assert "1.5" not in audit_str  # DP epsilon value hidden

        print("✅ Federated learning model verified with privacy preservation")

    @pytest.mark.asyncio
    async def test_scenario_content_processing_pipeline_security(self, fog_infrastructure):
        """
        Scenario: Content processing pipeline with multi-stage ZK verification

        Requirements:
        - Verify content type compliance without inspecting content
        - Ensure processing permissions without revealing user data
        - Validate output compliance without exposing processed results
        """

        workflow = fog_infrastructure["workflow"]

        # Simulate content processing request with sensitive data
        content_request = {
            "user_id": "premium_user_12345",  # Sensitive user identity
            "content_batch": [
                {
                    "file_id": "medical_report_001.json",
                    "mime_type": "application/json",
                    "size": 15 * 1024,  # 15KB
                    "classification": "phi_data",  # Protected Health Information
                    "processing_permissions": ["ai_analysis", "anonymization"],
                },
                {
                    "file_id": "patient_image_002.dicom",
                    "mime_type": "application/dicom",
                    "size": 2 * 1024 * 1024,  # 2MB
                    "classification": "phi_data",
                    "processing_permissions": ["ai_analysis", "compression"],
                },
            ],
            "user_permissions": {
                "data_retention_days": 30,  # Short retention for medical data
                "consent_level": "explicit",
                "consent_percentage": 100.0,  # Full consent required
                "anonymization_required": True,
            },
        }

        # Content processing policies
        content_policies = {
            "allowed_mime_types": ["application/json", "application/dicom", "text/plain"],
            "max_file_size": 10 * 1024 * 1024,  # 10MB per file
            "allowed_classifications": ["public", "internal", "phi_data"],
            "required_permissions": ["ai_analysis"],
        }

        privacy_policies = {
            "max_retention_days": 90,
            "min_consent_percentage": 100.0,  # Medical data requires full consent
            "anonymization_required": True,
            "audit_trail_required": True,
        }

        # Process each file with ZK verification
        processing_results = []

        for file_data in content_request["content_batch"]:
            # Verify file content compliance
            content_valid, content_proof = await workflow.audit_integration.verify_content_compliance(
                file_metadata={
                    "mime_type": file_data["mime_type"],
                    "size": file_data["size"],
                    "extension": Path(file_data["file_id"]).suffix,
                },
                content_policy=content_policies,
                entity_id=file_data["file_id"],
            )

            # Verify user permission compliance
            permission_valid, permission_proof = await workflow.audit_integration.verify_privacy_compliance(
                compliance_data=content_request["user_permissions"],
                compliance_requirements=privacy_policies,
                entity_id=content_request["user_id"],
            )

            processing_results.append(
                {
                    "file_id": file_data["file_id"],
                    "content_compliant": content_valid,
                    "permission_compliant": permission_valid,
                    "content_proof": content_proof,
                    "permission_proof": permission_proof,
                    "processing_approved": content_valid and permission_valid,
                }
            )

        # Verify all files can be processed
        all_approved = all(result["processing_approved"] for result in processing_results)
        assert all_approved is True

        # Verify privacy preservation
        audit_events = workflow.audit_integration.audit_events
        audit_str = json.dumps(audit_events)

        # Sensitive data should be protected
        assert "premium_user_12345" not in audit_str  # User ID should be hashed
        assert "medical_report_001.json" not in audit_str  # File names should be hashed
        assert "phi_data" not in audit_str  # Classification should not appear

        # But processing should have succeeded
        successful_files = [r for r in processing_results if r["processing_approved"]]
        assert len(successful_files) == 2

        print("✅ Content processing pipeline verified with privacy preservation")

    @pytest.mark.asyncio
    async def test_scenario_multi_tenant_compliance_verification(self, fog_infrastructure):
        """
        Scenario: Multi-tenant fog environment with tenant isolation verification

        Requirements:
        - Verify each tenant's compliance independently
        - Ensure resource isolation without revealing tenant details
        - Validate cross-tenant security policies
        """

        audit_integration = fog_infrastructure["audit"]

        # Simulate multi-tenant environment
        tenants = [
            {
                "tenant_id": "fintech_startup_alpha",  # Sensitive business identity
                "tier": "premium",
                "compliance_data": {
                    "data_retention_days": 2555,  # 7 years for financial data
                    "user_consent_percentage": 97.5,
                    "security_score": 0.96,
                    "audit_findings_count": 0,
                    "encryption_level": "bank_grade",
                },
                "resource_usage": {"cpu_cores": 16, "memory_gb": 64, "storage_tb": 5, "network_bandwidth_gbps": 10},
            },
            {
                "tenant_id": "healthcare_provider_beta",
                "tier": "enterprise",
                "compliance_data": {
                    "data_retention_days": 1095,  # 3 years for medical data
                    "user_consent_percentage": 100.0,  # Medical requires 100%
                    "security_score": 0.98,
                    "audit_findings_count": 0,
                    "hipaa_compliant": True,
                },
                "resource_usage": {"cpu_cores": 32, "memory_gb": 128, "storage_tb": 10, "network_bandwidth_gbps": 20},
            },
            {
                "tenant_id": "gaming_company_gamma",
                "tier": "standard",
                "compliance_data": {
                    "data_retention_days": 365,  # 1 year for gaming data
                    "user_consent_percentage": 85.0,  # Lower requirement for gaming
                    "security_score": 0.88,
                    "audit_findings_count": 2,
                    "gdpr_compliant": True,
                },
                "resource_usage": {"cpu_cores": 8, "memory_gb": 32, "storage_tb": 2, "network_bandwidth_gbps": 5},
            },
        ]

        # Different compliance requirements by tenant tier
        compliance_requirements = {
            "premium": {
                "min_consent_percentage": 95.0,
                "max_retention_days": 2555,
                "min_security_score": 0.95,
                "max_audit_findings": 0,
            },
            "enterprise": {
                "min_consent_percentage": 100.0,
                "max_retention_days": 1095,
                "min_security_score": 0.95,
                "max_audit_findings": 0,
            },
            "standard": {
                "min_consent_percentage": 80.0,
                "max_retention_days": 365,
                "min_security_score": 0.85,
                "max_audit_findings": 3,
            },
        }

        # Verify compliance for each tenant independently
        tenant_results = {}

        for tenant in tenants:
            tier_requirements = compliance_requirements[tenant["tier"]]

            # Verify tenant compliance with ZK predicates
            is_compliant, proof_id = await audit_integration.verify_privacy_compliance(
                compliance_data=tenant["compliance_data"],
                compliance_requirements=tier_requirements,
                entity_id=tenant["tenant_id"],
            )

            tenant_results[tenant["tenant_id"]] = {
                "tier": tenant["tier"],
                "compliant": is_compliant,
                "proof_id": proof_id,
            }

        # All tenants should be compliant
        assert all(result["compliant"] for result in tenant_results.values())

        # Verify tenant isolation in audit trail
        audit_events = audit_integration.audit_events
        audit_str = json.dumps(audit_events)

        # Tenant identities should be hashed for privacy
        assert "fintech_startup_alpha" not in audit_str
        assert "healthcare_provider_beta" not in audit_str
        assert "gaming_company_gamma" not in audit_str

        # Specific sensitive metrics should be hidden
        assert "2555" not in audit_str  # Retention days
        assert "97.5" not in audit_str  # Exact compliance percentage
        assert "0.96" not in audit_str  # Exact security score

        print(f"✅ Multi-tenant compliance verified for {len(tenants)} tenants with privacy isolation")

    @pytest.mark.asyncio
    async def test_scenario_edge_computing_resource_allocation(self, fog_infrastructure):
        """
        Scenario: Edge computing resource allocation with privacy-preserving verification

        Requirements:
        - Verify resource requests without revealing exact requirements
        - Ensure SLA compliance without exposing performance metrics
        - Validate allocation fairness across multiple requestors
        """

        workflow = fog_infrastructure["workflow"]

        # Simulate edge computing resource requests
        resource_requests = [
            {
                "requestor_id": "autonomous_vehicle_fleet_001",
                "request_type": "real_time_inference",
                "resource_requirements": {
                    "cpu_cores": 8,
                    "memory_gb": 32,
                    "gpu_memory_gb": 16,
                    "latency_requirement_ms": 10,  # Critical real-time requirement
                    "availability_requirement": 0.9999,  # 99.99% uptime
                    "geographic_constraint": "within_5km",  # Sensitive location constraint
                },
                "compliance_profile": {
                    "data_retention_days": 7,  # Short retention for vehicle data
                    "user_consent_percentage": 95.0,
                    "security_score": 0.95,
                    "audit_findings_count": 0,
                },
            },
            {
                "requestor_id": "smart_city_traffic_system",
                "request_type": "batch_analytics",
                "resource_requirements": {
                    "cpu_cores": 16,
                    "memory_gb": 64,
                    "storage_tb": 1,
                    "latency_requirement_ms": 1000,  # Less strict latency
                    "availability_requirement": 0.999,  # 99.9% uptime
                    "geographic_constraint": "city_limits",
                },
                "compliance_profile": {
                    "data_retention_days": 365,  # Longer retention for city planning
                    "user_consent_percentage": 90.0,
                    "security_score": 0.92,
                    "audit_findings_count": 1,
                },
            },
            {
                "requestor_id": "industrial_iot_monitor",
                "request_type": "continuous_monitoring",
                "resource_requirements": {
                    "cpu_cores": 4,
                    "memory_gb": 16,
                    "network_bandwidth_mbps": 100,
                    "latency_requirement_ms": 100,
                    "availability_requirement": 0.995,  # 99.5% uptime
                    "geographic_constraint": "industrial_zone",
                },
                "compliance_profile": {
                    "data_retention_days": 1095,  # 3 years for industrial data
                    "user_consent_percentage": 100.0,  # Worker data requires full consent
                    "security_score": 0.93,
                    "audit_findings_count": 0,
                },
            },
        ]

        # Edge computing platform policies

        compliance_policies = {
            "max_retention_days": 1095,
            "min_consent_percentage": 85.0,
            "min_security_score": 0.90,
            "max_audit_findings": 2,
        }

        # Process each resource allocation request
        allocation_results = []

        for request in resource_requests:
            # Create verification data for workflow
            verification_data = {
                "compliance_check": {
                    "secret_data": request["compliance_profile"],
                    "public_parameters": compliance_policies,
                }
            }

            # Execute resource allocation workflow
            result = await workflow.execute_workflow(
                workflow_name="content_processing",  # Reusing existing workflow
                entity_id=request["requestor_id"],
                verification_data=verification_data,
            )

            allocation_results.append(
                {
                    "requestor_id": request["requestor_id"],
                    "request_type": request["request_type"],
                    "allocation_approved": result["overall_success"],
                    "proof_ids": result["proof_ids"],
                }
            )

        # Verify allocation decisions
        approved_requests = [r for r in allocation_results if r["allocation_approved"]]
        assert len(approved_requests) == 3  # All should be approved based on compliance

        # Verify privacy preservation in allocation process
        audit_events = workflow.audit_integration.audit_events
        audit_str = json.dumps(audit_events)

        # Sensitive requestor information should be protected
        assert "autonomous_vehicle_fleet_001" not in audit_str
        assert "smart_city_traffic_system" not in audit_str
        assert "industrial_iot_monitor" not in audit_str

        # Sensitive requirements should be hidden
        assert "within_5km" not in audit_str  # Geographic constraints
        assert "10" not in audit_str  # Latency requirements (in ms context)
        assert "0.9999" not in audit_str  # Exact availability requirements

        print(f"✅ Edge resource allocation verified for {len(resource_requests)} requests with privacy preservation")


class TestZKPredicateRealWorldPerformance:
    """Test ZK predicate performance in realistic workloads."""

    @pytest.fixture
    def production_scale_infrastructure(self):
        """Create production-scale ZK infrastructure for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Multiple ZK engines simulating distributed fog nodes
            fog_nodes = {}
            for i in range(5):
                node_id = f"fog_node_{i:03d}"
                zk_engine = ZKPredicateEngine(node_id=node_id, data_dir=f"{tmpdir}/{node_id}")
                audit_integration = ZKAuditIntegration(zk_engine=zk_engine)
                workflow = ZKPredicateWorkflow(audit_integration=audit_integration)

                fog_nodes[node_id] = {"zk_engine": zk_engine, "audit": audit_integration, "workflow": workflow}

            yield fog_nodes

    @pytest.mark.asyncio
    async def test_high_throughput_verification_scenario(self, production_scale_infrastructure):
        """
        Test high-throughput verification scenario simulating production workload.

        Scenario: 1000 simultaneous edge device onboarding requests
        """

        fog_nodes = list(production_scale_infrastructure.values())
        verification_tasks = []

        # Simulate 1000 edge devices requesting onboarding
        for i in range(1000):
            # Distribute requests across fog nodes
            node_idx = i % len(fog_nodes)
            fog_node = fog_nodes[node_idx]

            # Create realistic edge device configuration
            device_config = {
                "device_id": f"edge_device_{i:04d}",
                "network_config": {
                    "services": [
                        {"name": "sensor_data", "port": 8080 + (i % 100), "protocol": "tcp"},
                        {"name": "control_interface", "port": 8443, "protocol": "tcp"},
                    ]
                },
                "compliance_data": {
                    "data_retention_days": 30 + (i % 300),  # Varying retention policies
                    "user_consent_percentage": 85.0 + (i % 15),  # 85-100% range
                    "security_score": 0.8 + (i % 20) * 0.01,  # 0.8-0.99 range
                    "audit_findings_count": i % 3,  # 0-2 findings
                },
            }

            # Create verification task
            async def verify_device(device_cfg, fog_node_ref):
                try:
                    # Network policy verification
                    network_valid, network_proof = await fog_node_ref["audit"].verify_network_policy_compliance(
                        network_config=device_cfg["network_config"],
                        policy_parameters={
                            "allowed_protocols": ["tcp", "udp"],
                            "allowed_port_ranges": ["registered", "dynamic"],
                            "max_services": 5,
                        },
                        entity_id=device_cfg["device_id"],
                    )

                    # Compliance verification
                    compliance_valid, compliance_proof = await fog_node_ref["audit"].verify_privacy_compliance(
                        compliance_data=device_cfg["compliance_data"],
                        compliance_requirements={
                            "min_consent_percentage": 80.0,
                            "max_retention_days": 365,
                            "min_security_score": 0.75,
                            "max_audit_findings": 3,
                        },
                        entity_id=device_cfg["device_id"],
                    )

                    return {
                        "device_id": device_cfg["device_id"],
                        "success": network_valid and compliance_valid,
                        "network_proof": network_proof,
                        "compliance_proof": compliance_proof,
                        "node_id": fog_node_ref["zk_engine"].node_id,
                    }
                except Exception as e:
                    return {
                        "device_id": device_cfg["device_id"],
                        "success": False,
                        "error": str(e),
                        "node_id": fog_node_ref["zk_engine"].node_id,
                    }

            task = verify_device(device_config, fog_node)
            verification_tasks.append(task)

        # Execute all verifications concurrently
        start_time = time.time()
        results = await asyncio.gather(*verification_tasks, return_exceptions=True)
        end_time = time.time()

        # Analyze results
        successful_verifications = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed_verifications = [r for r in results if isinstance(r, dict) and not r.get("success")]
        exceptions = [r for r in results if isinstance(r, Exception)]

        total_time = end_time - start_time
        throughput = len(results) / total_time

        print("High-throughput verification results:")
        print(f"  Total requests: {len(results)}")
        print(f"  Successful: {len(successful_verifications)}")
        print(f"  Failed: {len(failed_verifications)}")
        print(f"  Exceptions: {len(exceptions)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} verifications/second")

        # Performance assertions
        assert len(successful_verifications) > 950  # At least 95% success rate
        assert len(exceptions) < 10  # Less than 1% exceptions
        assert throughput > 50  # At least 50 verifications per second

        # Verify load distribution across nodes
        verifications_per_node = {}
        for result in successful_verifications:
            node_id = result["node_id"]
            verifications_per_node[node_id] = verifications_per_node.get(node_id, 0) + 1

        print(f"Load distribution: {verifications_per_node}")

        # Load should be relatively balanced
        avg_load = len(successful_verifications) / len(fog_nodes)
        for node_id, load in verifications_per_node.items():
            assert abs(load - avg_load) < avg_load * 0.3  # Within 30% of average

    @pytest.mark.asyncio
    async def test_sustained_load_scenario(self, production_scale_infrastructure):
        """
        Test sustained load scenario over time.

        Scenario: Continuous verification requests for 30 seconds
        """

        fog_nodes = list(production_scale_infrastructure.values())
        results = []
        start_time = time.time()
        test_duration = 30  # seconds

        request_count = 0

        while time.time() - start_time < test_duration:
            # Generate batch of requests
            batch_size = 20
            batch_tasks = []

            for i in range(batch_size):
                node_idx = request_count % len(fog_nodes)
                fog_node = fog_nodes[node_idx]

                # Simple network policy verification
                network_config = {
                    "services": [
                        {"name": f"service_{request_count}", "port": 8000 + (request_count % 1000), "protocol": "tcp"}
                    ]
                }

                policy_params = {"allowed_protocols": ["tcp"], "allowed_port_ranges": ["registered"], "max_services": 5}

                async def verify_request(config, params, entity_id, node):
                    try:
                        valid, proof = await node["audit"].verify_network_policy_compliance(
                            network_config=config, policy_parameters=params, entity_id=entity_id
                        )
                        return {"success": valid, "proof": proof, "timestamp": time.time()}
                    except Exception as e:
                        return {"success": False, "error": str(e), "timestamp": time.time()}

                task = verify_request(network_config, policy_params, f"entity_{request_count}", fog_node)
                batch_tasks.append(task)
                request_count += 1

            # Execute batch
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)

            # Small delay between batches
            await asyncio.sleep(0.1)

        end_time = time.time()
        actual_duration = end_time - start_time

        # Analyze sustained performance
        successful_requests = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed_requests = [r for r in results if isinstance(r, dict) and not r.get("success")]

        avg_throughput = len(results) / actual_duration
        success_rate = len(successful_requests) / len(results) * 100

        print("Sustained load test results:")
        print(f"  Duration: {actual_duration:.2f}s")
        print(f"  Total requests: {len(results)}")
        print(f"  Successful: {len(successful_requests)} ({success_rate:.1f}%)")
        print(f"  Failed: {len(failed_requests)}")
        print(f"  Average throughput: {avg_throughput:.2f} requests/second")

        # Performance assertions for sustained load
        assert success_rate > 95  # At least 95% success rate
        assert avg_throughput > 30  # At least 30 requests per second sustained
        assert len(results) > 500  # Should process significant number of requests

        print("✅ Sustained load test completed successfully")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])
