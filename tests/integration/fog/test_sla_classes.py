"""
SLA Classes Integration Tests

Comprehensive integration tests for the three-tier SLA system:
- S (replicated+attested): Mission-critical jobs with cryptographic attestation
- A (replicated): High-availability jobs with multi-node replication
- B (best-effort): Standard jobs with single-node placement

Tests SLA enforcement across the entire fog computing stack including:
- Job replication according to SLA requirements
- Cryptographic attestation for S-class jobs
- Placement latency compliance
- Resource allocation policies
- Failure handling and consensus mechanisms
"""

import time

import pytest

from packages.fog.gateway.monitoring.metrics import FogMetricsCollector
from packages.fog.gateway.scheduler.placement import FogNode
from packages.fog.gateway.scheduler.sla_classes import SLAClass, SLAManager, validate_sla_class


@pytest.fixture
def metrics_collector():
    """Create metrics collector for testing"""
    return FogMetricsCollector()


@pytest.fixture
def sla_manager(metrics_collector):
    """Create SLA manager for testing"""
    return SLAManager(metrics_collector)


@pytest.fixture
def mock_fog_nodes():
    """Create mock fog nodes for testing"""
    nodes = []
    for i in range(6):
        node = FogNode(
            node_id=f"test-node-{i:02d}",
            cpu_cores=4.0,
            memory_gb=8.0,
            available_cpu=0.8,
            available_memory=0.7,
            trust_score=0.9,
            region="test-region",
            latency_ms=50,
        )
        nodes.append(node)
    return nodes


class TestSLAClassRequirements:
    """Test SLA class requirement definitions and validation"""

    def test_sla_class_definitions(self, sla_manager):
        """Test SLA class requirements are properly defined"""

        # Test S-class requirements
        s_requirements = sla_manager.get_sla_requirements(SLAClass.S)
        assert s_requirements.sla_class == SLAClass.S
        assert s_requirements.min_replicas == 3
        assert s_requirements.requires_attestation is True
        assert s_requirements.max_placement_latency_ms == 250
        assert s_requirements.min_success_rate == 0.999

        # Test A-class requirements
        a_requirements = sla_manager.get_sla_requirements(SLAClass.A)
        assert a_requirements.sla_class == SLAClass.A
        assert a_requirements.min_replicas == 2
        assert a_requirements.requires_attestation is False
        assert a_requirements.max_placement_latency_ms == 500
        assert a_requirements.min_success_rate == 0.99

        # Test B-class requirements
        b_requirements = sla_manager.get_sla_requirements(SLAClass.B)
        assert b_requirements.sla_class == SLAClass.B
        assert b_requirements.min_replicas == 1
        assert b_requirements.requires_attestation is False
        assert b_requirements.max_placement_latency_ms == 1000
        assert b_requirements.min_success_rate == 0.95

    def test_sla_class_validation(self):
        """Test SLA class string validation"""

        assert validate_sla_class("replicated_attested") == SLAClass.S
        assert validate_sla_class("replicated") == SLAClass.A
        assert validate_sla_class("best_effort") == SLAClass.B

        # Invalid class defaults to B
        assert validate_sla_class("invalid") == SLAClass.B
        assert validate_sla_class("") == SLAClass.B


class TestJobReplication:
    """Test job replication according to SLA requirements"""

    def test_s_class_job_replication(self, sla_manager):
        """Test S-class jobs require 3 replicas with attestation"""

        job = sla_manager.create_replicated_job("s-job-1", SLAClass.S, "production")

        assert job.sla_class == SLAClass.S
        assert job.requirements.min_replicas == 3
        assert job.requirements.requires_attestation is True
        assert len(job.replicas) == 0  # No replicas initially

        # Add required replicas
        replica_ids = []
        for i in range(3):
            replica_id = sla_manager.add_job_replica("s-job-1", f"node-{i}")
            replica_ids.append(replica_id)

        assert len(job.replicas) == 3
        assert all(r.status == "pending" for r in job.replicas)

    def test_a_class_job_replication(self, sla_manager):
        """Test A-class jobs require 2 replicas without attestation"""

        job = sla_manager.create_replicated_job("a-job-1", SLAClass.A, "staging")

        assert job.sla_class == SLAClass.A
        assert job.requirements.min_replicas == 2
        assert job.requirements.requires_attestation is False

        # Add required replicas
        for i in range(2):
            sla_manager.add_job_replica("a-job-1", f"node-{i}")

        assert len(job.replicas) == 2

    def test_b_class_job_single_placement(self, sla_manager):
        """Test B-class jobs require only 1 replica"""

        job = sla_manager.create_replicated_job("b-job-1", SLAClass.B, "development")

        assert job.sla_class == SLAClass.B
        assert job.requirements.min_replicas == 1
        assert job.requirements.requires_attestation is False

        # Add single replica
        sla_manager.add_job_replica("b-job-1", "node-0")

        assert len(job.replicas) == 1


class TestJobExecution:
    """Test job execution lifecycle with SLA enforcement"""

    def test_job_replica_lifecycle(self, sla_manager):
        """Test complete replica lifecycle from pending to completed"""

        job = sla_manager.create_replicated_job("lifecycle-job", SLAClass.A, "test")

        # Add replicas
        replica1_id = sla_manager.add_job_replica("lifecycle-job", "node-1")
        replica2_id = sla_manager.add_job_replica("lifecycle-job", "node-2")

        # Start replicas
        sla_manager.start_replica("lifecycle-job", replica1_id)
        sla_manager.start_replica("lifecycle-job", replica2_id)

        assert len(job.get_active_replicas()) == 2
        assert all(r.started_at is not None for r in job.get_active_replicas())

        # Complete replicas
        result_data = {"output": "success", "computation_time": 30.5}
        sla_manager.complete_replica("lifecycle-job", replica1_id, result_data)
        sla_manager.complete_replica("lifecycle-job", replica2_id, result_data)

        assert len(job.get_completed_replicas()) == 2
        assert job.is_consensus_reached()
        assert job.consensus_result is not None

    def test_replica_failure_handling(self, sla_manager):
        """Test replica failure handling and recovery"""

        job = sla_manager.create_replicated_job("failure-job", SLAClass.S, "production")

        # Add 3 replicas for S-class job
        replica_ids = []
        for i in range(3):
            replica_id = sla_manager.add_job_replica("failure-job", f"node-{i}")
            replica_ids.append(replica_id)
            sla_manager.start_replica("failure-job", replica_id)

        # Fail one replica
        sla_manager.fail_replica("failure-job", replica_ids[0], "node_failure")

        # Complete remaining replicas
        result_data = {"output": "success", "result": 42}
        sla_manager.complete_replica("failure-job", replica_ids[1], result_data)
        sla_manager.complete_replica("failure-job", replica_ids[2], result_data)

        # Should still reach consensus with majority
        assert job.is_consensus_reached()
        assert len(job.get_completed_replicas()) == 2

        # Check that one replica failed
        failed_replicas = [r for r in job.replicas if r.status == "failed"]
        assert len(failed_replicas) == 1

    def test_consensus_mechanism(self, sla_manager):
        """Test consensus mechanism with result validation"""

        job = sla_manager.create_replicated_job("consensus-job", SLAClass.S, "production")

        # Add 3 replicas
        replica_ids = []
        for i in range(3):
            replica_id = sla_manager.add_job_replica("consensus-job", f"node-{i}")
            replica_ids.append(replica_id)
            sla_manager.start_replica("consensus-job", replica_id)

        # Complete replicas with same result
        result_data = {"computation": "result", "value": 123}
        sla_manager.complete_replica("consensus-job", replica_ids[0], result_data)
        sla_manager.complete_replica("consensus-job", replica_ids[1], result_data)

        # Should reach consensus with 2/3 majority
        assert job.is_consensus_reached()
        assert job.consensus_result is not None
        assert job.consensus_result["participating_replicas"] == 2


class TestCryptographicAttestation:
    """Test cryptographic attestation for S-class jobs"""

    def test_s_class_attestation_generation(self, sla_manager):
        """Test attestation generation for S-class jobs"""

        job = sla_manager.create_replicated_job("attestation-job", SLAClass.S, "secure")

        # Add replica and complete it
        replica_id = sla_manager.add_job_replica("attestation-job", "secure-node")
        sla_manager.start_replica("attestation-job", replica_id)

        result_data = {"sensitive": "computation", "timestamp": time.time()}
        sla_manager.complete_replica("attestation-job", replica_id, result_data)

        # Check attestation was generated
        replica = job.replicas[0]
        assert replica.attestation_proof is not None
        assert "method" in replica.attestation_proof
        assert "timestamp" in replica.attestation_proof
        assert "execution_signature" in replica.attestation_proof
        assert "merkle_root" in replica.attestation_proof

        # Job should be marked as attestation complete when consensus reached
        # (Would need more replicas for full consensus in real scenario)

    def test_non_s_class_no_attestation(self, sla_manager):
        """Test that non-S-class jobs don't generate attestation"""

        # Test A-class job
        a_job = sla_manager.create_replicated_job("a-no-attest", SLAClass.A, "staging")
        replica_id = sla_manager.add_job_replica("a-no-attest", "node-1")
        sla_manager.start_replica("a-no-attest", replica_id)

        result_data = {"result": "no attestation needed"}
        sla_manager.complete_replica("a-no-attest", replica_id, result_data)

        replica = a_job.replicas[0]
        assert replica.attestation_proof is None

        # Test B-class job
        b_job = sla_manager.create_replicated_job("b-no-attest", SLAClass.B, "dev")
        replica_id = sla_manager.add_job_replica("b-no-attest", "node-2")
        sla_manager.start_replica("b-no-attest", replica_id)

        sla_manager.complete_replica("b-no-attest", replica_id, result_data)

        replica = b_job.replicas[0]
        assert replica.attestation_proof is None


class TestPlacementLatencySLA:
    """Test placement latency SLA enforcement"""

    def test_s_class_latency_violation(self, sla_manager, metrics_collector):
        """Test S-class latency violation detection"""

        job = sla_manager.create_replicated_job("slow-s-job", SLAClass.S, "production")
        replica_id = sla_manager.add_job_replica("slow-s-job", "slow-node")

        # Simulate slow placement (> 250ms for S-class)
        time.sleep(0.3)  # 300ms delay
        sla_manager.start_replica("slow-s-job", replica_id)

        # Check SLA violation was recorded
        assert len(job.sla_violations) > 0
        assert any("placement_latency" in violation for violation in job.sla_violations)

    def test_placement_latency_calculation(self, sla_manager):
        """Test placement latency calculation"""

        job = sla_manager.create_replicated_job("latency-job", SLAClass.B, "test")

        time.time()
        replica_id = sla_manager.add_job_replica("latency-job", "test-node")
        time.sleep(0.1)  # 100ms delay
        sla_manager.start_replica("latency-job", replica_id)

        latency = job.calculate_placement_latency()
        assert latency is not None
        assert latency >= 100  # At least 100ms
        assert latency < 200  # But reasonable


class TestSLACompliance:
    """Test SLA compliance monitoring and reporting"""

    def test_sla_compliance_report(self, sla_manager):
        """Test SLA compliance report generation"""

        # Create jobs with different SLA classes
        jobs = [
            ("s-job-1", SLAClass.S),
            ("s-job-2", SLAClass.S),
            ("a-job-1", SLAClass.A),
            ("a-job-2", SLAClass.A),
            ("a-job-3", SLAClass.A),
            ("b-job-1", SLAClass.B),
            ("b-job-2", SLAClass.B),
        ]

        for job_id, sla_class in jobs:
            sla_manager.create_replicated_job(job_id, sla_class, "compliance-test")

            # Add and complete replicas
            min_replicas = sla_manager.get_sla_requirements(sla_class).min_replicas
            for i in range(min_replicas):
                replica_id = sla_manager.add_job_replica(job_id, f"node-{i}")
                sla_manager.start_replica(job_id, replica_id)
                sla_manager.complete_replica(job_id, replica_id, {"result": "success"})

        # Generate compliance report
        report = sla_manager.get_sla_compliance_report()

        assert "total_jobs" in report
        assert "by_class" in report
        assert "overall_compliance" in report
        assert report["total_jobs"] == len(jobs)

        # Check class-specific data
        assert "replicated_attested" in report["by_class"]  # S-class
        assert "replicated" in report["by_class"]  # A-class
        assert "best_effort" in report["by_class"]  # B-class

    def test_sla_violation_tracking(self, sla_manager):
        """Test SLA violation counting and tracking"""

        initial_violations = sla_manager.violation_counts.copy()

        # Create job that will violate SLA
        sla_manager.create_replicated_job("violation-job", SLAClass.S, "test")
        replica_id = sla_manager.add_job_replica("violation-job", "node-1")

        # Force SLA violation by simulating long placement time
        time.sleep(0.3)  # > 250ms S-class limit
        sla_manager.start_replica("violation-job", replica_id)

        # Check violation was counted
        assert sla_manager.violation_counts[SLAClass.S] > initial_violations[SLAClass.S]


class TestResourceAllocation:
    """Test resource allocation policies by SLA class"""

    def test_sla_resource_limits(self, sla_manager):
        """Test SLA class resource allocation limits"""

        s_req = sla_manager.get_sla_requirements(SLAClass.S)
        a_req = sla_manager.get_sla_requirements(SLAClass.A)
        b_req = sla_manager.get_sla_requirements(SLAClass.B)

        # S-class should have highest priority but limited resources
        assert s_req.priority_weight > a_req.priority_weight > b_req.priority_weight
        assert s_req.max_resource_ratio < a_req.max_resource_ratio <= b_req.max_resource_ratio

    def test_priority_enforcement(self, sla_manager):
        """Test priority enforcement in resource allocation"""

        # Create mixed SLA class jobs
        s_job = sla_manager.create_replicated_job("priority-s", SLAClass.S, "prod")
        a_job = sla_manager.create_replicated_job("priority-a", SLAClass.A, "prod")
        b_job = sla_manager.create_replicated_job("priority-b", SLAClass.B, "prod")

        # S-class should have highest priority
        assert s_job.requirements.priority_weight > a_job.requirements.priority_weight
        assert a_job.requirements.priority_weight > b_job.requirements.priority_weight


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios"""

    @pytest.mark.asyncio
    async def test_mixed_workload_scenario(self, sla_manager, metrics_collector):
        """Test realistic mixed workload scenario"""

        # Simulate realistic workload mix
        jobs = []

        # 10% S-class (critical)
        for i in range(2):
            job = sla_manager.create_replicated_job(f"critical-{i}", SLAClass.S, "production")
            jobs.append(job)

        # 30% A-class (important)
        for i in range(6):
            job = sla_manager.create_replicated_job(f"important-{i}", SLAClass.A, "production")
            jobs.append(job)

        # 60% B-class (standard)
        for i in range(12):
            job = sla_manager.create_replicated_job(f"standard-{i}", SLAClass.B, "production")
            jobs.append(job)

        # Process all jobs
        for job in jobs:
            min_replicas = job.requirements.min_replicas

            for replica_idx in range(min_replicas):
                replica_id = sla_manager.add_job_replica(job.job_id, f"node-{replica_idx}")
                sla_manager.start_replica(job.job_id, replica_id)

                # Simulate successful completion
                result = {"computation": f"result-{job.job_id}-{replica_idx}"}
                sla_manager.complete_replica(job.job_id, replica_id, result)

        # Verify all jobs completed successfully
        assert all(job.is_consensus_reached() for job in jobs)

        # Generate final compliance report
        report = sla_manager.get_sla_compliance_report()
        assert report["total_jobs"] == 20
        assert report["overall_compliance"] >= 0.95  # 95% compliance target

    @pytest.mark.asyncio
    async def test_disaster_recovery_scenario(self, sla_manager):
        """Test disaster recovery with multiple node failures"""

        # Create S-class job with high replication
        job = sla_manager.create_replicated_job("disaster-job", SLAClass.S, "critical")

        # Add 5 replicas (more than minimum 3)
        replica_ids = []
        for i in range(5):
            replica_id = sla_manager.add_job_replica("disaster-job", f"node-{i}")
            replica_ids.append(replica_id)
            sla_manager.start_replica("disaster-job", replica_id)

        # Simulate disaster: fail 2 nodes
        sla_manager.fail_replica("disaster-job", replica_ids[0], "network_partition")
        sla_manager.fail_replica("disaster-job", replica_ids[1], "hardware_failure")

        # Complete remaining replicas
        result_data = {"disaster_recovery": True, "result": "survived"}
        for replica_id in replica_ids[2:]:
            sla_manager.complete_replica("disaster-job", replica_id, result_data)

        # Should still reach consensus with 3/5 replicas
        assert job.is_consensus_reached()
        assert len(job.get_completed_replicas()) == 3
        assert len([r for r in job.replicas if r.status == "failed"]) == 2


# Pytest configuration for SLA tests
@pytest.mark.sla
class TestSLAMarker:
    """Tests marked for SLA validation"""

    pass


if __name__ == "__main__":
    # Run SLA tests directly
    pytest.main([__file__, "-v", "-m", "not stress"])
