"""
Fog Computing Proof System Demonstration

Comprehensive example demonstrating all aspects of the cryptographic proof system:
- Proof-of-Execution generation and verification
- Proof-of-Audit consensus mechanisms
- Proof-of-SLA compliance validation
- Merkle tree batch aggregation
- Tokenomics integration and reward distribution
- REST API usage examples

This demo shows how to integrate the proof system with fog computing infrastructure
for verifiable, trustworthy distributed computing.
"""

import asyncio
import json
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import proof system components
from infrastructure.fog.proofs import (
    AuditEvidence,
    MerkleTree,
    ProofGenerator,
    ProofSystemIntegration,
    ProofTokenomicsIntegration,
    ProofVerifier,
    SLAMeasurement,
    TaskExecution,
    get_proof_system_info,
)
from infrastructure.fog.tokenomics.fog_token_system import FogTokenSystem


class ProofSystemDemo:
    """Comprehensive demonstration of the proof system"""

    def __init__(self):
        self.node_id = "demo_fog_node_001"
        self.demo_dir = Path("./proof_demo_data")
        self.demo_dir.mkdir(exist_ok=True)

        # Initialize components
        self.proof_generator = None
        self.proof_verifier = None
        self.proof_integration = None
        self.tokenomics_integration = None
        self.fog_token_system = None

        logger.info(f"Proof System Demo initialized for node {self.node_id}")

    async def setup_components(self):
        """Initialize all proof system components"""
        logger.info("Setting up proof system components...")

        # 1. Initialize Fog Token System
        self.fog_token_system = FogTokenSystem(
            initial_supply=1000000, reward_rate_per_hour=10, staking_apy=0.05  # 1M tokens
        )

        # Create test accounts
        await self.fog_token_system.create_account("demo_node_account", b"demo_public_key", initial_balance=1000.0)

        await self.fog_token_system.create_account("verifier_account", b"verifier_public_key", initial_balance=500.0)

        # 2. Initialize Proof Generator
        self.proof_generator = ProofGenerator(
            node_id=self.node_id, private_key_path=str(self.demo_dir / "node_private_key.pem")
        )

        # 3. Initialize Proof Verifier
        self.proof_verifier = ProofVerifier(
            verifier_id=f"{self.node_id}_verifier", trusted_keys_dir=str(self.demo_dir / "trusted_keys")
        )

        # Add our own key to trusted keys for demo
        public_key_pem = self.proof_generator.public_key.public_bytes(
            encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()
        self.proof_verifier.add_trusted_key(public_key_pem, self.node_id)

        # 4. Initialize Integration Layer
        self.proof_integration = ProofSystemIntegration(
            node_id=self.node_id,
            proof_storage_dir=str(self.demo_dir / "proofs"),
            enable_auto_proofs=False,  # Manual control for demo
        )

        # 5. Initialize Tokenomics Integration
        self.tokenomics_integration = ProofTokenomicsIntegration(fog_token_system=self.fog_token_system)

        logger.info("All components initialized successfully!")

    async def demonstrate_execution_proofs(self):
        """Demonstrate Proof-of-Execution generation and verification"""
        logger.info("\n=== Demonstrating Proof-of-Execution ===")

        # Simulate task execution
        start_time = time.time() - 120  # 2 minutes ago
        end_time = time.time()

        task_execution = TaskExecution(
            task_id="demo_ml_training_task",
            node_id=self.node_id,
            start_timestamp=start_time,
            end_timestamp=end_time,
            input_hash="sha256:a1b2c3d4e5f6...",  # Simulated input hash
            output_hash="sha256:z9y8x7w6v5u4...",  # Simulated output hash
            exit_code=0,
            resource_usage={
                "cpu_percent": 87.5,
                "memory_percent": 65.3,
                "gpu_utilization": 94.2,
                "disk_io_mb": 1250.8,
                "network_io_mb": 45.2,
            },
            environment_hash="sha256:env_hash_123...",
            command_signature="python train_model.py --epochs 100 --batch-size 64",
        )

        # Generate execution proof
        logger.info("Generating Proof-of-Execution...")
        execution_proof = await self.proof_generator.generate_proof_of_execution(
            task_execution=task_execution,
            computation_trace=[
                "Initialized neural network model",
                "Loaded training dataset (50,000 samples)",
                "Started training loop",
                "Completed epoch 50/100",
                "Completed epoch 100/100",
                "Model training completed",
                "Saved trained model to disk",
            ],
            include_witness=True,
        )

        logger.info(f"✓ Generated PoE proof: {execution_proof.proof_id}")
        logger.info(f"  - Data hash: {execution_proof.data_hash[:16]}...")
        logger.info(f"  - Signature: {execution_proof.signature[:16]}...")
        logger.info(f"  - Resource efficiency: {execution_proof.metadata['resource_efficiency']:.2%}")
        logger.info(f"  - Execution duration: {execution_proof.metadata['execution_duration']:.1f}s")

        # Verify the proof
        logger.info("Verifying Proof-of-Execution...")
        verification_report = await self.proof_verifier.verify_proof(execution_proof)

        logger.info(f"✓ Verification result: {verification_report.result.value}")
        logger.info(f"  - Signature valid: {verification_report.signature_valid}")
        logger.info(f"  - Timestamp valid: {verification_report.timestamp_valid}")
        logger.info(f"  - Data integrity: {verification_report.data_integrity_valid}")
        logger.info(f"  - Verification time: {verification_report.verification_time_ms:.1f}ms")

        # Calculate and demonstrate rewards
        logger.info("Calculating reward for execution proof...")
        proof_reward = await self.tokenomics_integration.calculate_proof_reward(execution_proof, verification_report)

        logger.info("✓ Reward calculation:")
        logger.info(f"  - Base reward: {proof_reward.base_reward} FOG")
        logger.info(f"  - Quality bonus: {proof_reward.quality_bonus} FOG")
        logger.info(f"  - Verification bonus: {proof_reward.verification_bonus} FOG")
        logger.info(f"  - Net reward: {proof_reward.net_reward} FOG")

        return execution_proof

    async def demonstrate_audit_proofs(self):
        """Demonstrate Proof-of-Audit consensus mechanism"""
        logger.info("\n=== Demonstrating Proof-of-Audit ===")

        # Simulate audit evidence from multiple AI auditors
        audit_evidence = [
            AuditEvidence(
                audit_id="audit_001",
                auditor_id="ai_auditor_alpha",
                task_id="demo_ml_training_task",
                timestamp=time.time(),
                verdict="pass",
                confidence_score=0.94,
                evidence_hashes=["model_architecture_hash", "training_data_hash", "hyperparameters_hash"],
                consensus_weight=1.0,
            ),
            AuditEvidence(
                audit_id="audit_002",
                auditor_id="ai_auditor_beta",
                task_id="demo_ml_training_task",
                timestamp=time.time(),
                verdict="pass",
                confidence_score=0.89,
                evidence_hashes=["output_validation_hash", "loss_curve_hash"],
                consensus_weight=0.8,
            ),
            AuditEvidence(
                audit_id="audit_003",
                auditor_id="ai_auditor_gamma",
                task_id="demo_ml_training_task",
                timestamp=time.time(),
                verdict="pass",
                confidence_score=0.91,
                evidence_hashes=["model_performance_hash", "resource_usage_hash"],
                consensus_weight=0.9,
            ),
            AuditEvidence(
                audit_id="audit_004",
                auditor_id="ai_auditor_delta",
                task_id="demo_ml_training_task",
                timestamp=time.time(),
                verdict="warning",  # One dissenting opinion
                confidence_score=0.76,
                evidence_hashes=["potential_overfitting_hash"],
                consensus_weight=0.7,
            ),
        ]

        # Generate audit consensus proof
        logger.info("Generating Proof-of-Audit with 4 AI auditors...")
        audit_proof = await self.proof_generator.generate_proof_of_audit(
            audit_evidence=audit_evidence, consensus_threshold=0.75
        )

        logger.info(f"✓ Generated PoA proof: {audit_proof.proof_id}")
        logger.info(f"  - Consensus achieved: {audit_proof.achieved_consensus:.1%}")
        logger.info(f"  - Consensus threshold: {audit_proof.consensus_threshold:.1%}")
        logger.info(f"  - Auditor count: {len(audit_proof.audit_evidence)}")
        logger.info(f"  - Consensus verdict: {audit_proof.metadata['consensus_verdict']}")
        logger.info(f"  - Weighted confidence: {audit_proof.metadata['weighted_confidence']:.2%}")

        # Verify audit proof
        logger.info("Verifying Proof-of-Audit...")
        audit_verification = await self.proof_verifier.verify_proof(audit_proof)

        logger.info(f"✓ Audit verification: {audit_verification.result.value}")
        logger.info(f"  - Consensus valid: {audit_verification.consensus_valid}")

        return audit_proof

    async def demonstrate_sla_proofs(self):
        """Demonstrate Proof-of-SLA compliance validation"""
        logger.info("\n=== Demonstrating Proof-of-SLA ===")

        # Simulate SLA measurements over time
        start_period = time.time() - 3600  # 1 hour ago
        end_period = time.time()

        sla_measurements = [
            # Latency measurements
            SLAMeasurement(
                measurement_id="latency_001",
                node_id=self.node_id,
                timestamp=start_period + 600,
                metric_type="latency",
                measured_value=45.2,  # 45.2ms
                target_value=50.0,  # Target: <50ms
                compliance_status="compliant",
                measurement_hash="lat_hash_001",
            ),
            SLAMeasurement(
                measurement_id="latency_002",
                node_id=self.node_id,
                timestamp=start_period + 1200,
                metric_type="latency",
                measured_value=48.7,
                target_value=50.0,
                compliance_status="compliant",
                measurement_hash="lat_hash_002",
            ),
            # Throughput measurements
            SLAMeasurement(
                measurement_id="throughput_001",
                node_id=self.node_id,
                timestamp=start_period + 900,
                metric_type="throughput",
                measured_value=125.8,  # 125.8 ops/sec
                target_value=100.0,  # Target: >100 ops/sec
                compliance_status="compliant",
                measurement_hash="thr_hash_001",
            ),
            # Availability measurements
            SLAMeasurement(
                measurement_id="availability_001",
                node_id=self.node_id,
                timestamp=start_period + 1800,
                metric_type="availability",
                measured_value=99.98,  # 99.98%
                target_value=99.9,  # Target: >99.9%
                compliance_status="compliant",
                measurement_hash="avail_hash_001",
            ),
            # One breach example
            SLAMeasurement(
                measurement_id="latency_003",
                node_id=self.node_id,
                timestamp=start_period + 2400,
                metric_type="latency",
                measured_value=55.3,  # Breach!
                target_value=50.0,
                compliance_status="breach",
                measurement_hash="lat_hash_003",
            ),
        ]

        # Generate SLA compliance proof
        logger.info(f"Generating Proof-of-SLA for {len(sla_measurements)} measurements...")
        sla_proof = await self.proof_generator.generate_proof_of_sla(
            sla_measurements=sla_measurements, compliance_period=(start_period, end_period)
        )

        logger.info(f"✓ Generated PoSLA proof: {sla_proof.proof_id}")
        logger.info(f"  - Compliance percentage: {sla_proof.metadata['compliance_percentage']:.1f}%")
        logger.info(f"  - Measurement count: {sla_proof.metadata['measurement_count']}")
        logger.info(f"  - Period duration: {sla_proof.metadata['period_duration']:.0f}s")
        logger.info(f"  - Overall status: {sla_proof.metadata['overall_status']}")

        # Show aggregated metrics
        if sla_proof.aggregated_metrics:
            logger.info("  - Aggregated metrics:")
            for metric, value in sla_proof.aggregated_metrics.items():
                logger.info(f"    {metric}: {value:.2f}")

        # Verify SLA proof
        logger.info("Verifying Proof-of-SLA...")
        sla_verification = await self.proof_verifier.verify_proof(sla_proof)

        logger.info(f"✓ SLA verification: {sla_verification.result.value}")

        return sla_proof

    async def demonstrate_merkle_batch_proofs(self, individual_proofs):
        """Demonstrate Merkle tree batch proof aggregation"""
        logger.info("\n=== Demonstrating Merkle Batch Proofs ===")

        # Create batch proof from individual proofs
        logger.info(f"Creating batch proof from {len(individual_proofs)} individual proofs...")
        batch_proof = await self.proof_generator.create_merkle_batch_proof(individual_proofs)

        logger.info(f"✓ Generated batch proof: {batch_proof.proof_id}")
        logger.info(f"  - Merkle root: {batch_proof.merkle_root[:16]}...")
        logger.info(f"  - Batch size: {batch_proof.metadata['batch_size']}")
        logger.info(f"  - Proof types: {batch_proof.metadata['proof_types']}")
        logger.info(f"  - Tree depth: {batch_proof.verification_data['merkle_tree_depth']}")

        # Verify batch proof
        logger.info("Verifying Merkle batch proof...")
        batch_verification = await self.proof_verifier.verify_proof(batch_proof)

        logger.info(f"✓ Batch verification: {batch_verification.result.value}")
        logger.info(f"  - Merkle validation: {batch_verification.merkle_valid}")

        # Demonstrate individual proof verification within batch
        logger.info("Demonstrating individual proof verification within batch...")

        # Create standalone Merkle tree for demonstration
        leaf_hashes = [proof.data_hash for proof in individual_proofs]
        merkle_tree = MerkleTree(leaf_hashes)

        # Verify a specific leaf in the tree
        test_index = 1
        merkle_proof = merkle_tree.get_proof(test_index)

        logger.info("✓ Individual proof verification:")
        logger.info(f"  - Leaf index: {test_index}")
        logger.info(f"  - Leaf hash: {merkle_proof.leaf_hash[:16]}...")
        logger.info(f"  - Proof valid: {merkle_proof.verify()}")
        logger.info(f"  - Proof path length: {len(merkle_proof.proof_path)}")

        return batch_proof

    async def demonstrate_tokenomics_integration(self, proofs):
        """Demonstrate comprehensive tokenomics integration"""
        logger.info("\n=== Demonstrating Tokenomics Integration ===")

        total_rewards = 0.0

        for i, proof in enumerate(proofs):
            logger.info(f"\nProcessing proof {i+1}/{len(proofs)}: {proof.proof_id}")

            # Calculate reward
            reward = await self.tokenomics_integration.calculate_proof_reward(proof)
            total_rewards += float(reward.net_reward)

            logger.info(f"  - Proof type: {proof.proof_type.value}")
            logger.info(f"  - Base reward: {reward.base_reward} FOG")
            logger.info(f"  - Quality bonus: {reward.quality_bonus} FOG")
            logger.info(f"  - Net reward: {reward.net_reward} FOG")

            # Distribute reward
            success = await self.tokenomics_integration.distribute_proof_reward(reward, "demo_node_account")

            if success:
                logger.info("  ✓ Reward distributed successfully")
            else:
                logger.info("  ✗ Reward distribution failed")

        # Show account balances
        logger.info("\n--- Account Balances After Rewards ---")
        node_balance = self.fog_token_system.get_account_balance("demo_node_account")
        verifier_balance = self.fog_token_system.get_account_balance("verifier_account")

        logger.info(f"Node account balance: {node_balance['balance']:.2f} FOG")
        logger.info(f"Verifier account balance: {verifier_balance['balance']:.2f} FOG")
        logger.info(f"Total rewards earned: {total_rewards:.2f} FOG")

        # Show network statistics
        logger.info("\n--- Network Statistics ---")
        network_stats = self.fog_token_system.get_network_stats()
        logger.info(f"Total supply: {network_stats['current_supply']:,.0f} FOG")
        logger.info(f"Total rewards distributed: {network_stats['total_rewards_distributed']:,.2f} FOG")
        logger.info(f"Active accounts: {network_stats['total_accounts']}")

    async def demonstrate_proof_system_statistics(self):
        """Show comprehensive proof system statistics"""
        logger.info("\n=== Proof System Statistics ===")

        # Generator statistics
        gen_stats = self.proof_generator.get_statistics()
        logger.info("\n--- Proof Generator Stats ---")
        logger.info(f"Total proofs generated: {gen_stats['total_proofs']}")
        logger.info(f"Recent activity (24h): {gen_stats['recent_activity']}")
        logger.info(f"Proofs by type: {gen_stats['proofs_by_type']}")

        # Verifier statistics
        ver_stats = self.proof_verifier.get_verification_stats()
        logger.info("\n--- Proof Verifier Stats ---")
        logger.info(f"Total verifications: {ver_stats['total_verifications']}")
        logger.info(f"Success rate: {ver_stats['success_rate_percent']:.1f}%")
        logger.info(f"Trusted keys: {ver_stats['trusted_keys_count']}")

        # Integration statistics
        int_stats = self.proof_integration.get_statistics()
        logger.info("\n--- Integration Stats ---")
        logger.info(f"Execution proofs: {int_stats['execution_proofs_generated']}")
        logger.info(f"Audit proofs: {int_stats['audit_proofs_generated']}")
        logger.info(f"SLA proofs: {int_stats['sla_proofs_generated']}")
        logger.info(f"Batch proofs: {int_stats['batch_proofs_generated']}")

        # Tokenomics statistics
        token_stats = self.tokenomics_integration.get_reward_statistics()
        logger.info("\n--- Tokenomics Stats ---")
        logger.info(f"Total rewards calculated: {token_stats['total_rewards_calculated']}")
        logger.info(f"Quality bonuses given: {token_stats['quality_bonuses_given']}")
        logger.info(f"Verification rewards: {token_stats['verification_rewards_given']}")

    async def save_demo_results(self, proofs):
        """Save demo results to files for inspection"""
        logger.info("\n=== Saving Demo Results ===")

        results_dir = self.demo_dir / "results"
        results_dir.mkdir(exist_ok=True)

        # Save all proofs to JSON files
        for i, proof in enumerate(proofs):
            filename = f"proof_{i+1}_{proof.proof_type.value}_{proof.proof_id}.json"
            filepath = results_dir / filename

            # Convert proof to dictionary
            from dataclasses import asdict

            proof_dict = asdict(proof)
            proof_dict["timestamp"] = proof.timestamp.isoformat()
            proof_dict["proof_type"] = proof.proof_type.value

            with open(filepath, "w") as f:
                json.dump(proof_dict, f, indent=2, default=str)

            logger.info(f"✓ Saved: {filename}")

        # Save system info
        info_file = results_dir / "proof_system_info.json"
        with open(info_file, "w") as f:
            json.dump(get_proof_system_info(), f, indent=2)

        logger.info(f"✓ Saved system info: {info_file}")
        logger.info(f"\nAll results saved to: {results_dir}")

    async def run_complete_demo(self):
        """Run the complete proof system demonstration"""
        logger.info("🚀 Starting Comprehensive Fog Computing Proof System Demo")
        logger.info("=" * 60)

        try:
            # Setup
            await self.setup_components()

            # Demonstrate each proof type
            execution_proof = await self.demonstrate_execution_proofs()
            audit_proof = await self.demonstrate_audit_proofs()
            sla_proof = await self.demonstrate_sla_proofs()

            # Collect all individual proofs
            individual_proofs = [execution_proof, audit_proof, sla_proof]

            # Demonstrate batch proofs
            batch_proof = await self.demonstrate_merkle_batch_proofs(individual_proofs)

            # All proofs including batch
            all_proofs = individual_proofs + [batch_proof]

            # Demonstrate tokenomics
            await self.demonstrate_tokenomics_integration(all_proofs)

            # Show statistics
            await self.demonstrate_proof_system_statistics()

            # Save results
            await self.save_demo_results(all_proofs)

            logger.info("\n🎉 Demo completed successfully!")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Demo failed with error: {e}")
            raise


async def main():
    """Main demonstration function"""
    demo = ProofSystemDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    # Print system information
    print("Fog Computing Cryptographic Proof System")
    print("=========================================\n")

    system_info = get_proof_system_info()
    print(f"Version: {system_info['version']}")
    print(f"Author: {system_info['author']}")
    print("\nSupported Proof Types:")
    for proof_type, description in system_info["proof_types"].items():
        print(f"  - {proof_type}: {description}")

    print("\nKey Features:")
    for feature in system_info["features"]:
        print(f"  - {feature}")

    print("\n" + "=" * 50 + "\n")

    # Run the demo
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback

        traceback.print_exc()
