"""
Comprehensive Test Suite for Cryptographic Proof System

Tests all components of the fog computing proof system:
- Proof generation (PoE, PoA, PoSLA)
- Proof verification and validation
- Merkle tree construction and verification
- System integration and API endpoints
- Tokenomics integration and reward distribution
- Performance and security validation
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, Mock

import pytest

# Import proof system components
from infrastructure.fog.proofs import (
    AuditEvidence,
    MerkleProof,
    MerkleTree,
    ProofGenerator,
    ProofOfAudit,
    ProofOfExecution,
    ProofOfSLA,
    ProofReward,
    ProofSystemIntegration,
    ProofTokenomicsIntegration,
    ProofType,
    ProofVerifier,
    SLAMeasurement,
    TaskExecution,
    TaskProofRequest,
    VerificationResult,
)


class TestProofGenerator(unittest.TestCase):
    """Test proof generation functionality"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.node_id = "test_node_001"
        self.generator = ProofGenerator(node_id=self.node_id)

    def tearDown(self):
        """Clean up test environment"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_execution_proof_generation(self):
        """Test generation of Proof-of-Execution"""
        # Create mock task execution
        task_execution = TaskExecution(
            task_id="task_123",
            node_id=self.node_id,
            start_timestamp=time.time() - 100,
            end_timestamp=time.time(),
            input_hash="input_hash_123",
            output_hash="output_hash_123",
            exit_code=0,
            resource_usage={"cpu_percent": 45.5, "memory_percent": 32.1},
            environment_hash="env_hash_123",
            command_signature="python script.py",
        )

        # Generate proof
        proof = await self.generator.generate_proof_of_execution(
            task_execution=task_execution, computation_trace=["step1", "step2", "step3"], include_witness=True
        )

        # Validate proof structure
        self.assertIsInstance(proof, ProofOfExecution)
        self.assertEqual(proof.proof_type, ProofType.PROOF_OF_EXECUTION)
        self.assertEqual(proof.node_id, self.node_id)
        self.assertIsNotNone(proof.signature)
        self.assertIsNotNone(proof.data_hash)
        self.assertIsNotNone(proof.deterministic_hash)
        self.assertIsNotNone(proof.witness_data)
        self.assertEqual(len(proof.computation_trace), 3)

        # Validate metadata
        self.assertIn("execution_duration", proof.metadata)
        self.assertIn("resource_efficiency", proof.metadata)
        self.assertIn("verification_level", proof.metadata)

    @pytest.mark.asyncio
    async def test_audit_proof_generation(self):
        """Test generation of Proof-of-Audit"""
        # Create mock audit evidence
        audit_evidence = [
            AuditEvidence(
                audit_id="audit_1",
                auditor_id="auditor_a",
                task_id="task_123",
                timestamp=time.time(),
                verdict="pass",
                confidence_score=0.95,
                evidence_hashes=["hash1", "hash2"],
                consensus_weight=1.0,
            ),
            AuditEvidence(
                audit_id="audit_2",
                auditor_id="auditor_b",
                task_id="task_123",
                timestamp=time.time(),
                verdict="pass",
                confidence_score=0.88,
                evidence_hashes=["hash3", "hash4"],
                consensus_weight=0.8,
            ),
            AuditEvidence(
                audit_id="audit_3",
                auditor_id="auditor_c",
                task_id="task_123",
                timestamp=time.time(),
                verdict="fail",
                confidence_score=0.72,
                evidence_hashes=["hash5"],
                consensus_weight=0.6,
            ),
        ]

        # Generate proof
        proof = await self.generator.generate_proof_of_audit(audit_evidence=audit_evidence, consensus_threshold=0.67)

        # Validate proof structure
        self.assertIsInstance(proof, ProofOfAudit)
        self.assertEqual(proof.proof_type, ProofType.PROOF_OF_AUDIT)
        self.assertEqual(len(proof.audit_evidence), 3)
        self.assertGreaterEqual(proof.achieved_consensus, proof.consensus_threshold)
        self.assertIsNotNone(proof.consensus_proof)
        self.assertEqual(len(proof.auditor_signatures), 3)

        # Validate consensus calculation
        self.assertTrue(proof.metadata["consensus_achieved"])
        self.assertEqual(proof.metadata["auditor_count"], 3)

    @pytest.mark.asyncio
    async def test_sla_proof_generation(self):
        """Test generation of Proof-of-SLA"""
        # Create mock SLA measurements
        start_time = time.time() - 3600  # 1 hour ago
        end_time = time.time()

        sla_measurements = [
            SLAMeasurement(
                measurement_id="sla_1",
                node_id=self.node_id,
                timestamp=start_time + 300,
                metric_type="latency",
                measured_value=45.2,
                target_value=50.0,
                compliance_status="compliant",
                measurement_hash="hash1",
            ),
            SLAMeasurement(
                measurement_id="sla_2",
                node_id=self.node_id,
                timestamp=start_time + 600,
                metric_type="throughput",
                measured_value=105.8,
                target_value=100.0,
                compliance_status="compliant",
                measurement_hash="hash2",
            ),
            SLAMeasurement(
                measurement_id="sla_3",
                node_id=self.node_id,
                timestamp=start_time + 900,
                metric_type="availability",
                measured_value=99.95,
                target_value=99.9,
                compliance_status="compliant",
                measurement_hash="hash3",
            ),
        ]

        # Generate proof
        proof = await self.generator.generate_proof_of_sla(
            sla_measurements=sla_measurements, compliance_period=(start_time, end_time)
        )

        # Validate proof structure
        self.assertIsInstance(proof, ProofOfSLA)
        self.assertEqual(proof.proof_type, ProofType.PROOF_OF_SLA)
        self.assertEqual(len(proof.sla_measurements), 3)
        self.assertEqual(proof.compliance_period, (start_time, end_time))
        self.assertIsNotNone(proof.aggregated_metrics)
        self.assertIsNotNone(proof.attestation_signature)

        # Validate compliance calculation
        self.assertEqual(proof.metadata["compliance_percentage"], 100.0)
        self.assertEqual(proof.metadata["overall_status"], "compliant")

    @pytest.mark.asyncio
    async def test_batch_proof_generation(self):
        """Test generation of Merkle batch proof"""
        # Generate individual proofs first
        proofs = []

        # Create execution proof
        task_execution = TaskExecution(
            task_id=f"task_{i}",
            node_id=self.node_id,
            start_timestamp=time.time() - 100,
            end_timestamp=time.time(),
            input_hash=f"input_hash_{i}",
            output_hash=f"output_hash_{i}",
            exit_code=0,
            resource_usage={"cpu_percent": 45.5},
            environment_hash=f"env_hash_{i}",
            command_signature=f"python script_{i}.py",
        )

        for i in range(5):
            task_execution.task_id = f"task_{i}"
            task_execution.input_hash = f"input_hash_{i}"
            task_execution.output_hash = f"output_hash_{i}"

            proof = await self.generator.generate_proof_of_execution(task_execution)
            proofs.append(proof)

        # Create batch proof
        batch_proof = await self.generator.create_merkle_batch_proof(proofs)

        # Validate batch proof
        self.assertIsNotNone(batch_proof)
        self.assertEqual(batch_proof.proof_type, ProofType.MERKLE_BATCH)
        self.assertIsNotNone(batch_proof.merkle_root)
        self.assertEqual(batch_proof.metadata["batch_size"], 5)
        self.assertIn("merkle_tree_depth", batch_proof.verification_data)
        self.assertIn("leaf_hashes", batch_proof.verification_data)


class TestProofVerifier(unittest.TestCase):
    """Test proof verification functionality"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.node_id = "test_node_001"
        self.generator = ProofGenerator(node_id=self.node_id)
        self.verifier = ProofVerifier(
            verifier_id="test_verifier", trusted_keys_dir=str(Path(self.temp_dir) / "trusted_keys")
        )

        # Add generator's public key to verifier's trusted keys
        public_key_pem = self.generator.public_key.public_bytes(
            encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()
        self.verifier.add_trusted_key(public_key_pem, "test_node_001")

    def tearDown(self):
        """Clean up test environment"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_valid_proof_verification(self):
        """Test verification of valid proof"""
        # Generate a valid proof
        task_execution = TaskExecution(
            task_id="task_123",
            node_id=self.node_id,
            start_timestamp=time.time() - 100,
            end_timestamp=time.time(),
            input_hash="input_hash_123",
            output_hash="output_hash_123",
            exit_code=0,
            resource_usage={"cpu_percent": 45.5},
            environment_hash="env_hash_123",
            command_signature="python script.py",
        )

        proof = await self.generator.generate_proof_of_execution(task_execution)

        # Verify the proof
        report = await self.verifier.verify_proof(proof)

        # Validate verification results
        self.assertEqual(report.result, VerificationResult.VALID)
        self.assertTrue(report.signature_valid)
        self.assertTrue(report.timestamp_valid)
        self.assertTrue(report.data_integrity_valid)
        self.assertEqual(len(report.error_messages), 0)
        self.assertGreater(report.verification_time_ms, 0)

    @pytest.mark.asyncio
    async def test_invalid_signature_detection(self):
        """Test detection of invalid signatures"""
        # Generate a proof and corrupt its signature
        task_execution = TaskExecution(
            task_id="task_123",
            node_id=self.node_id,
            start_timestamp=time.time() - 100,
            end_timestamp=time.time(),
            input_hash="input_hash_123",
            output_hash="output_hash_123",
            exit_code=0,
            resource_usage={"cpu_percent": 45.5},
            environment_hash="env_hash_123",
            command_signature="python script.py",
        )

        proof = await self.generator.generate_proof_of_execution(task_execution)

        # Corrupt the signature
        proof.signature = "invalid_signature_data"

        # Verify the corrupted proof
        report = await self.verifier.verify_proof(proof)

        # Validate verification results
        self.assertNotEqual(report.result, VerificationResult.VALID)
        self.assertFalse(report.signature_valid)
        self.assertGreater(len(report.error_messages), 0)

    @pytest.mark.asyncio
    async def test_batch_verification(self):
        """Test batch verification of multiple proofs"""
        # Generate multiple proofs
        proofs = []
        for i in range(3):
            task_execution = TaskExecution(
                task_id=f"task_{i}",
                node_id=self.node_id,
                start_timestamp=time.time() - 100,
                end_timestamp=time.time(),
                input_hash=f"input_hash_{i}",
                output_hash=f"output_hash_{i}",
                exit_code=0,
                resource_usage={"cpu_percent": 45.5},
                environment_hash=f"env_hash_{i}",
                command_signature=f"python script_{i}.py",
            )

            proof = await self.generator.generate_proof_of_execution(task_execution)
            proofs.append(proof)

        # Batch verify
        reports = await self.verifier.batch_verify_proofs(proofs)

        # Validate batch results
        self.assertEqual(len(reports), 3)
        for report in reports:
            self.assertEqual(report.result, VerificationResult.VALID)
            self.assertTrue(report.signature_valid)


class TestMerkleTree(unittest.TestCase):
    """Test Merkle tree functionality"""

    def test_merkle_tree_construction(self):
        """Test Merkle tree construction"""
        # Create test data
        data_hashes = ["hash1", "hash2", "hash3", "hash4", "hash5"]

        # Build tree
        tree = MerkleTree(data_hashes)

        # Validate tree structure
        self.assertEqual(len(tree.get_leaf_hashes()), 5)
        self.assertIsNotNone(tree.get_root())
        self.assertGreater(tree.depth, 0)

        # Test tree info
        info = tree.get_tree_info()
        self.assertEqual(info["total_leaves"], 5)
        self.assertGreater(info["tree_depth"], 0)
        self.assertIsNotNone(info["root_hash"])

    def test_merkle_proof_generation(self):
        """Test Merkle inclusion proof generation"""
        data_hashes = ["hash1", "hash2", "hash3", "hash4"]
        tree = MerkleTree(data_hashes)

        # Generate proof for each leaf
        for i in range(len(data_hashes)):
            proof = tree.get_proof(i)

            self.assertIsNotNone(proof)
            self.assertIsInstance(proof, MerkleProof)
            self.assertEqual(proof.leaf_hash, data_hashes[i])
            self.assertEqual(proof.root_hash, tree.get_root())
            self.assertEqual(proof.leaf_index, i)
            self.assertTrue(proof.verify())

    def test_merkle_proof_verification(self):
        """Test Merkle proof verification"""
        data_hashes = ["hash1", "hash2", "hash3"]
        tree = MerkleTree(data_hashes)

        # Generate and verify proof
        proof = tree.get_proof(1)
        self.assertTrue(proof.verify())

        # Test invalid proof (corrupt leaf hash)
        invalid_proof = MerkleProof(
            leaf_hash="corrupted_hash",
            root_hash=proof.root_hash,
            leaf_index=proof.leaf_index,
            proof_path=proof.proof_path,
            tree_size=proof.tree_size,
        )
        self.assertFalse(invalid_proof.verify())


class TestProofIntegration(unittest.TestCase):
    """Test proof system integration"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.node_id = "test_node_001"

        # Create integration with auto proofs disabled for testing
        self.integration = ProofSystemIntegration(
            node_id=self.node_id, proof_storage_dir=str(Path(self.temp_dir) / "proofs"), enable_auto_proofs=False
        )

    def tearDown(self):
        """Clean up test environment"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_integration_startup_shutdown(self):
        """Test integration startup and shutdown"""
        # Test startup
        await self.integration.start()
        self.assertTrue(self.integration._running)

        # Test shutdown
        await self.integration.stop()
        self.assertFalse(self.integration._running)

    @pytest.mark.asyncio
    async def test_proof_request_queuing(self):
        """Test proof request queuing"""
        # Create test requests
        task_request = TaskProofRequest(
            task_id="task_123",
            node_id=self.node_id,
            input_data="test_input",
            output_data="test_output",
            command="python script.py",
            environment={"PATH": "/usr/bin"},
            resource_usage={"cpu_percent": 45.5},
            start_time=time.time() - 100,
            end_time=time.time(),
            exit_code=0,
        )

        # Queue request
        self.integration.queue_execution_proof(task_request)

        # Verify queuing (with auto proofs disabled, should remain in queue)
        self.assertEqual(len(self.integration.pending_execution_proofs), 1)

    @pytest.mark.asyncio
    async def test_proof_hook_execution(self):
        """Test proof hook execution"""
        hook_called = {"called": False, "proof": None}

        def test_hook(proof):
            hook_called["called"] = True
            hook_called["proof"] = proof

        # Add hook
        self.integration.add_task_hook(test_hook)

        # Generate proof (this should trigger the hook)
        task_request = TaskProofRequest(
            task_id="task_123",
            node_id=self.node_id,
            input_data="test_input",
            output_data="test_output",
            command="python script.py",
            environment={"PATH": "/usr/bin"},
            resource_usage={"cpu_percent": 45.5},
            start_time=time.time() - 100,
            end_time=time.time(),
            exit_code=0,
        )

        proof = await self.integration.generate_execution_proof(task_request)

        # Verify hook was called
        self.assertTrue(hook_called["called"])
        self.assertIsNotNone(hook_called["proof"])
        self.assertEqual(hook_called["proof"].proof_id, proof.proof_id)


class TestTokenomicsIntegration(unittest.TestCase):
    """Test tokenomics integration"""

    def setUp(self):
        """Set up test environment"""
        # Create mock fog token system
        self.fog_token_system = Mock()
        self.fog_token_system.record_contribution = AsyncMock(return_value=Mock())
        self.fog_token_system.transfer = AsyncMock(return_value=True)

        self.tokenomics = ProofTokenomicsIntegration(self.fog_token_system)

    @pytest.mark.asyncio
    async def test_execution_proof_reward_calculation(self):
        """Test reward calculation for execution proof"""
        # Create mock execution proof
        task_execution = TaskExecution(
            task_id="task_123",
            node_id="test_node",
            start_timestamp=time.time() - 300,  # 5 minutes ago
            end_timestamp=time.time(),
            input_hash="input_hash",
            output_hash="output_hash",
            exit_code=0,
            resource_usage={"cpu_percent": 85.0, "memory_percent": 60.0},
            environment_hash="env_hash",
            command_signature="python script.py",
        )

        proof = ProofOfExecution(
            proof_id="proof_123",
            proof_type=ProofType.PROOF_OF_EXECUTION,
            timestamp=datetime.now(timezone.utc),
            node_id="test_node",
            data_hash="data_hash",
            task_execution=task_execution,
            metadata={"resource_efficiency": 0.85, "execution_duration": 300.0},  # High efficiency  # Fast execution
        )

        # Calculate reward
        reward = await self.tokenomics.calculate_proof_reward(proof)

        # Validate reward calculation
        self.assertIsInstance(reward, ProofReward)
        self.assertEqual(reward.proof_id, "proof_123")
        self.assertGreater(reward.base_reward, 0)
        self.assertGreater(reward.quality_bonus, 0)  # Should get quality bonus
        self.assertGreaterEqual(reward.net_reward, reward.base_reward)

    @pytest.mark.asyncio
    async def test_audit_proof_reward_calculation(self):
        """Test reward calculation for audit proof"""
        # Create mock audit evidence
        audit_evidence = [
            AuditEvidence(
                audit_id="audit_1",
                auditor_id="auditor_a",
                task_id="task_123",
                timestamp=time.time(),
                verdict="pass",
                confidence_score=0.95,
                evidence_hashes=[],
                consensus_weight=1.0,
            )
        ]

        proof = ProofOfAudit(
            proof_id="audit_proof_123",
            proof_type=ProofType.PROOF_OF_AUDIT,
            timestamp=datetime.now(timezone.utc),
            node_id="test_node",
            data_hash="data_hash",
            audit_evidence=audit_evidence,
            consensus_threshold=0.67,
            achieved_consensus=0.95,  # Strong consensus
            auditor_signatures={},
        )

        # Calculate reward
        reward = await self.tokenomics.calculate_proof_reward(proof)

        # Validate reward calculation
        self.assertIsInstance(reward, ProofReward)
        self.assertGreater(reward.base_reward, 0)
        self.assertGreater(reward.consensus_bonus, 0)  # Should get consensus bonus

    @pytest.mark.asyncio
    async def test_reward_distribution(self):
        """Test reward distribution to account"""
        # Create mock reward
        reward = ProofReward(
            proof_id="proof_123",
            proof_type=ProofType.PROOF_OF_EXECUTION,
            base_reward=Decimal("10.0"),
            quality_bonus=Decimal("2.0"),
            verification_bonus=Decimal("1.0"),
            consensus_bonus=Decimal("0.5"),
            total_reward=Decimal("13.5"),
            penalty_amount=Decimal("0.0"),
            net_reward=Decimal("13.5"),
            reward_factors={},
        )

        # Distribute reward
        success = await self.tokenomics.distribute_proof_reward(reward, "test_account")

        # Verify distribution
        self.assertTrue(success)
        self.fog_token_system.record_contribution.assert_called_once()

    @pytest.mark.asyncio
    async def test_verification_reward_distribution(self):
        """Test verification reward distribution"""
        success = await self.tokenomics.distribute_verification_reward(
            "verifier_account", ProofType.PROOF_OF_EXECUTION, True
        )

        # Verify distribution
        self.assertTrue(success)
        self.fog_token_system.transfer.assert_called_once()

    def test_reward_statistics(self):
        """Test reward statistics collection"""
        stats = self.tokenomics.get_reward_statistics()

        # Validate statistics structure
        self.assertIn("total_rewards_calculated", stats)
        self.assertIn("total_rewards_distributed", stats)
        self.assertIn("reward_config", stats)
        self.assertIn("base_rewards", stats["reward_config"])


class TestProofSystemPerformance(unittest.TestCase):
    """Test proof system performance and security"""

    def setUp(self):
        """Set up performance test environment"""
        self.node_id = "perf_test_node"
        self.generator = ProofGenerator(node_id=self.node_id)

    @pytest.mark.asyncio
    async def test_proof_generation_performance(self):
        """Test proof generation performance"""
        num_proofs = 100
        start_time = time.time()

        # Generate multiple proofs
        tasks = []
        for i in range(num_proofs):
            task_execution = TaskExecution(
                task_id=f"perf_task_{i}",
                node_id=self.node_id,
                start_timestamp=time.time() - 100,
                end_timestamp=time.time(),
                input_hash=f"input_hash_{i}",
                output_hash=f"output_hash_{i}",
                exit_code=0,
                resource_usage={"cpu_percent": 45.5},
                environment_hash=f"env_hash_{i}",
                command_signature=f"python script_{i}.py",
            )

            task = self.generator.generate_proof_of_execution(task_execution)
            tasks.append(task)

        # Execute all proof generation tasks concurrently
        proofs = await asyncio.gather(*tasks)

        end_time = time.time()
        duration = end_time - start_time

        # Performance validation
        self.assertEqual(len(proofs), num_proofs)
        self.assertLess(duration, 60.0)  # Should complete within 60 seconds

        # Calculate throughput
        throughput = num_proofs / duration
        self.assertGreater(throughput, 1.0)  # Should generate at least 1 proof/second

        print(f"Generated {num_proofs} proofs in {duration:.2f}s (throughput: {throughput:.2f} proofs/s)")

    @pytest.mark.asyncio
    async def test_merkle_tree_performance(self):
        """Test Merkle tree construction performance"""
        # Create large dataset
        num_leaves = 10000
        data_hashes = [f"hash_{i}" for i in range(num_leaves)]

        start_time = time.time()

        # Build Merkle tree
        tree = MerkleTree(data_hashes)

        construction_time = time.time() - start_time

        # Test proof generation performance
        proof_start = time.time()

        # Generate proofs for sample of leaves
        sample_indices = list(range(0, num_leaves, 100))  # Every 100th leaf
        for i in sample_indices:
            proof = tree.get_proof(i)
            self.assertIsNotNone(proof)
            self.assertTrue(proof.verify())

        proof_time = time.time() - proof_start

        # Performance validation
        self.assertLess(construction_time, 10.0)  # Should construct within 10 seconds
        self.assertLess(proof_time, 5.0)  # Should generate sample proofs within 5 seconds

        print(f"Merkle tree with {num_leaves} leaves: construction={construction_time:.2f}s, ")
        print(f"proof generation for {len(sample_indices)} leaves: {proof_time:.2f}s")

    def test_proof_size_analysis(self):
        """Analyze proof sizes for efficiency"""

        # Create sample proofs of different types
        task_execution = TaskExecution(
            task_id="size_test",
            node_id=self.node_id,
            start_timestamp=time.time() - 100,
            end_timestamp=time.time(),
            input_hash="input_hash",
            output_hash="output_hash",
            exit_code=0,
            resource_usage={"cpu_percent": 45.5},
            environment_hash="env_hash",
            command_signature="python script.py",
        )

        # Measure proof sizes (in bytes when serialized)
        from dataclasses import asdict
        import json

        async def measure_proof_size():
            proof = await self.generator.generate_proof_of_execution(task_execution)
            proof_dict = asdict(proof)
            proof_dict["timestamp"] = proof.timestamp.isoformat()
            proof_dict["proof_type"] = proof.proof_type.value
            serialized = json.dumps(proof_dict)
            return len(serialized.encode("utf-8"))

        # Run size measurement
        import asyncio

        proof_size = asyncio.run(measure_proof_size())

        # Size validation (proofs should be reasonably compact)
        self.assertLess(proof_size, 10240)  # Less than 10KB
        self.assertGreater(proof_size, 100)  # At least 100 bytes

        print(f"Execution proof size: {proof_size} bytes ({proof_size/1024:.2f} KB)")


if __name__ == "__main__":
    # Run specific test categories
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "performance":
        # Run only performance tests
        suite = unittest.TestLoader().loadTestsFromTestCase(TestProofSystemPerformance)
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    else:
        # Run all tests
        unittest.main(verbosity=2)
