"""
Comprehensive Test Suite for Constitutional Transparency System
Tests all transparency components with all constitutional tiers
"""

import pytest
import asyncio
import tempfile
import shutil
import time
from pathlib import Path

# Import transparency system components
from infrastructure.transparency import (
    ConstitutionalTransparencySystem,
    ConstitutionalTransparencyConfig,
    AuditLevel,
    ConstitutionalViolationType,
    ConstitutionalDecisionType,
    GovernanceLevel,
    PrivacyLevel,
    ZKProofType,
    ParticipationLevel,
)
from infrastructure.transparency.cryptographic_verification import (
    VerificationLevel,
)
from infrastructure.transparency.realtime_compliance_monitor import (
    RealTimeComplianceMonitor,
)


class TestConstitutionalTransparencySystem:
    """Test suite for the complete constitutional transparency system"""

    @pytest.fixture
    async def temp_storage(self):
        """Create temporary storage directory for tests"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    async def transparency_system(self, temp_storage):
        """Create transparency system for testing"""
        config = ConstitutionalTransparencyConfig(
            merkle_audit_storage=str(temp_storage / "merkle_audit"),
            decision_log_storage=str(temp_storage / "decisions"),
            privacy_audit_storage=str(temp_storage / "privacy_audit"),
            governance_audit_storage=str(temp_storage / "governance"),
            dashboard_config_path=str(temp_storage / "dashboard.json"),
            real_time_monitoring=False,  # Disable for testing
            dashboard_update_interval=5,
            democratic_participation_enabled=True,
        )

        system = ConstitutionalTransparencySystem(config)
        await system.initialize()

        yield system

        # Cleanup
        if system.is_initialized:
            await system.shutdown()

    @pytest.fixture
    def sample_decision_data(self):
        """Sample constitutional decision data for testing"""
        return {
            "decision_type": "content_moderation",
            "summary": "Content removed for constitutional harm violation",
            "outcome": "approved",
            "rationale_data": {
                "primary_reasoning": "Content contained H2-level harm that violates constitutional principles",
                "constitutional_principles": ["harm_prevention", "democratic_discourse"],
                "precedents": ["case_001", "case_012"],
                "harm_analysis": {"level": "H2", "confidence": 0.95, "categories": ["misinformation"]},
                "alternatives": ["warning_issued", "content_flagged"],
                "minority_opinions": [],
                "democratic_input": "Community reported content through democratic reporting system",
            },
            "evidence_data": [
                {
                    "type": "harm_classification",
                    "confidence": 0.95,
                    "data": {"classification": "H2", "categories": ["misinformation"]},
                    "source": "constitutional_classifier",
                    "documents": ["harm_policy_v2.1"],
                    "expert_opinions": ["constitutional_expert_001"],
                }
            ],
            "constitutional_context": {
                "policy_context": "content_moderation_policy_v1.0",
                "user_history": "no_previous_violations",
                "community_sentiment": "supportive_of_moderation",
            },
            "decision_maker": "constitutional_moderator_001",
            "policy_version": "1.0",
            "oversight_required": False,
        }


class TestTierBasedTransparency:
    """Test transparency across all constitutional tiers"""

    @pytest.mark.asyncio
    async def test_bronze_tier_full_transparency(self, transparency_system, sample_decision_data):
        """Test Bronze tier - full public transparency"""
        user_tier = "bronze"
        user_id = "bronze_user_123"

        # Log constitutional decision for Bronze tier
        decision_ids = await transparency_system.log_constitutional_decision_comprehensive(
            sample_decision_data,
            user_tier,
            user_id,
            ConstitutionalViolationType.HARM_CLASSIFICATION,
            GovernanceLevel.MODERATED,
        )

        # Verify all logging IDs returned
        assert "merkle_audit_id" in decision_ids
        assert "decision_log_id" in decision_ids

        # Verify Merkle audit entry exists with Bronze transparency
        merkle_entry = next(
            (
                e
                for e in transparency_system.merkle_audit.audit_entries
                if e.entry_id == decision_ids["merkle_audit_id"]
            ),
            None,
        )
        assert merkle_entry is not None
        assert merkle_entry.audit_level == AuditLevel.BRONZE
        assert merkle_entry.user_tier == user_tier
        assert merkle_entry.violation_type == ConstitutionalViolationType.HARM_CLASSIFICATION

        # Verify decision log entry exists
        decision_entry = transparency_system.decision_logger.decision_index.get(decision_ids["decision_log_id"])
        assert decision_entry is not None
        assert decision_entry.user_tier == user_tier
        assert decision_entry.decision_type == ConstitutionalDecisionType.CONTENT_MODERATION
        assert decision_entry.governance_level == GovernanceLevel.MODERATED

        # Verify public visibility (Bronze tier = full transparency)
        public_summary = transparency_system.decision_logger.get_public_decision_summary(
            decision_ids["decision_log_id"]
        )
        assert public_summary is not None
        assert "decision_summary" in public_summary
        assert "constitutional_principles" in public_summary

        # Verify integrity
        integrity_check = await transparency_system.verify_constitutional_decision_integrity(decision_ids)
        assert integrity_check["overall_integrity"]["verified"] is True

    @pytest.mark.asyncio
    async def test_silver_tier_selective_transparency(self, transparency_system, sample_decision_data):
        """Test Silver tier - selective disclosure transparency"""
        user_tier = "silver"
        user_id = "silver_user_456"

        # Log constitutional decision for Silver tier
        decision_ids = await transparency_system.log_constitutional_decision_comprehensive(
            sample_decision_data,
            user_tier,
            user_id,
            ConstitutionalViolationType.TIER_VIOLATION,
            GovernanceLevel.AUTOMATED,
        )

        # Verify Merkle audit entry with Silver transparency level
        merkle_entry = next(
            (
                e
                for e in transparency_system.merkle_audit.audit_entries
                if e.entry_id == decision_ids["merkle_audit_id"]
            ),
            None,
        )
        assert merkle_entry is not None
        assert merkle_entry.audit_level == AuditLevel.SILVER
        assert merkle_entry.user_tier == user_tier

        # Verify selective public disclosure (Silver tier = summary only)
        public_summary = transparency_system.decision_logger.get_public_decision_summary(
            decision_ids["decision_log_id"]
        )
        assert public_summary is not None
        assert "decision_type" in public_summary
        assert "decision_outcome" in public_summary
        # Should NOT contain full decision summary for Silver tier
        assert "decision_summary" not in public_summary or len(public_summary.get("decision_summary", "")) == 0

    @pytest.mark.asyncio
    async def test_gold_tier_privacy_preserving_transparency(self, transparency_system, sample_decision_data):
        """Test Gold tier - privacy-preserving transparency with ZK proofs"""
        user_tier = "gold"
        user_id = "gold_user_789"

        # Log constitutional decision for Gold tier
        decision_ids = await transparency_system.log_constitutional_decision_comprehensive(
            sample_decision_data, user_tier, user_id, None, GovernanceLevel.COMMUNITY  # No violation
        )

        # Verify ZK proof was generated for Gold tier
        assert "zk_proof_id" in decision_ids

        # Verify Merkle audit entry with Gold transparency level
        merkle_entry = next(
            (
                e
                for e in transparency_system.merkle_audit.audit_entries
                if e.entry_id == decision_ids["merkle_audit_id"]
            ),
            None,
        )
        assert merkle_entry is not None
        assert merkle_entry.audit_level == AuditLevel.GOLD
        assert merkle_entry.privacy_preserving_data is not None

        # Verify ZK proof exists and can be verified
        zk_proof = transparency_system.privacy_system.zk_proofs.get(decision_ids["zk_proof_id"])
        assert zk_proof is not None
        assert zk_proof.privacy_level == PrivacyLevel.PRIVACY_PRESERVING
        assert zk_proof.proof_type == ZKProofType.CONSTITUTIONAL_COMPLIANCE

        # Verify ZK proof verification
        zk_verification = await transparency_system.privacy_system.verify_zk_proof(decision_ids["zk_proof_id"])
        assert zk_verification["valid"] is True

        # Verify minimal public disclosure for Gold tier
        public_summary = transparency_system.decision_logger.get_public_decision_summary(
            decision_ids["decision_log_id"]
        )
        assert public_summary is not None
        # Should only contain privacy-preserving hash for Gold tier
        assert "privacy_preserving_hash" in public_summary

    @pytest.mark.asyncio
    async def test_platinum_tier_minimal_disclosure(self, transparency_system, sample_decision_data):
        """Test Platinum tier - minimal disclosure with cryptographic commitments"""
        user_tier = "platinum"
        user_id = "platinum_user_000"

        # Log constitutional decision for Platinum tier
        decision_ids = await transparency_system.log_constitutional_decision_comprehensive(
            sample_decision_data, user_tier, user_id, None, GovernanceLevel.CONSTITUTIONAL  # No violation
        )

        # Verify ZK proof was generated for Platinum tier
        assert "zk_proof_id" in decision_ids

        # Verify Merkle audit entry with Platinum transparency level
        merkle_entry = next(
            (
                e
                for e in transparency_system.merkle_audit.audit_entries
                if e.entry_id == decision_ids["merkle_audit_id"]
            ),
            None,
        )
        assert merkle_entry is not None
        assert merkle_entry.audit_level == AuditLevel.PLATINUM
        assert merkle_entry.privacy_preserving_data is not None

        # Verify ZK proof with minimal disclosure privacy level
        zk_proof = transparency_system.privacy_system.zk_proofs.get(decision_ids["zk_proof_id"])
        assert zk_proof is not None
        assert zk_proof.privacy_level == PrivacyLevel.MINIMAL_DISCLOSURE

        # Verify minimal public disclosure for Platinum tier
        public_summary = transparency_system.decision_logger.get_public_decision_summary(
            decision_ids["decision_log_id"]
        )
        assert public_summary is not None
        # Should only contain privacy-preserving hash and decision type
        assert "privacy_preserving_hash" in public_summary
        assert "decision_type" in public_summary
        assert len(public_summary) <= 3  # Very minimal information


class TestCryptographicVerification:
    """Test cryptographic verification and integrity"""

    @pytest.mark.asyncio
    async def test_digital_signatures(self, transparency_system, sample_decision_data):
        """Test digital signature generation and verification"""
        # Generate signature for decision data
        signature_id = await transparency_system.crypto_verifier.generate_constitutional_signature(
            sample_decision_data, "test_signer", "rsa_pss"
        )

        assert signature_id is not None

        # Verify signature
        verification = await transparency_system.crypto_verifier.verify_constitutional_signature(
            signature_id, sample_decision_data
        )

        assert verification["valid"] is True
        assert verification["signature_id"] == signature_id
        assert verification["signer_id"] == "test_signer"
        assert verification["algorithm"] == "rsa_pss"

    @pytest.mark.asyncio
    async def test_integrity_proofs(self, transparency_system, sample_decision_data):
        """Test integrity proof generation and verification"""
        # Test different verification levels
        verification_levels = [
            VerificationLevel.BASIC_HASH,
            VerificationLevel.DIGITAL_SIGNATURE,
            VerificationLevel.MERKLE_PROOF,
            VerificationLevel.ZERO_KNOWLEDGE,
            VerificationLevel.MULTI_SIGNATURE,
        ]

        for level in verification_levels:
            # Generate integrity proof
            proof_id = await transparency_system.crypto_verifier.generate_integrity_proof(sample_decision_data, level)

            assert proof_id is not None

            # Verify integrity proof
            proof_verification = await transparency_system.crypto_verifier.verify_integrity_proof(
                proof_id, sample_decision_data
            )

            assert proof_verification["valid"] is True
            assert proof_verification["verification_level"] == level.value

    @pytest.mark.asyncio
    async def test_audit_chains(self, transparency_system, sample_decision_data):
        """Test cryptographic audit chain creation and verification"""
        # Create audit chain
        chain_id = await transparency_system.crypto_verifier.create_audit_chain(sample_decision_data)
        assert chain_id is not None

        # Add additional data to chain
        additional_data = {
            "follow_up_decision": "Additional constitutional action",
            "timestamp": time.time(),
            "related_to": "original_decision",
        }

        new_link_hash = await transparency_system.crypto_verifier.add_to_audit_chain(chain_id, additional_data)
        assert new_link_hash is not None

        # Verify complete chain integrity
        chain_verification = await transparency_system.crypto_verifier.verify_audit_chain_integrity(chain_id)
        assert chain_verification["overall_valid"] is True
        assert chain_verification["chain_length"] == 2


class TestDemocraticGovernance:
    """Test democratic governance and participation tracking"""

    @pytest.mark.asyncio
    async def test_participant_registration(self, transparency_system):
        """Test democratic participant registration"""
        # Register participants across different tiers
        participants = []
        tiers = ["bronze", "silver", "gold", "platinum"]
        participation_levels = [
            ParticipationLevel.VOTER,
            ParticipationLevel.CONTRIBUTOR,
            ParticipationLevel.PROPOSER,
            ParticipationLevel.REPRESENTATIVE,
        ]

        for i, (tier, level) in enumerate(zip(tiers, participation_levels)):
            participant_id = await transparency_system.register_democratic_participant(f"participant_{i}", tier, level)
            participants.append((participant_id, tier, level))
            assert participant_id is not None

        # Verify participants were registered
        assert len(transparency_system.governance_audit.participants) == len(participants)

        for participant_id, tier, level in participants:
            participant = transparency_system.governance_audit.participants.get(participant_id)
            assert participant is not None
            assert participant.tier == tier
            assert participant.participation_level == level

    @pytest.mark.asyncio
    async def test_governance_proposal_workflow(self, transparency_system):
        """Test complete governance proposal and voting workflow"""
        # Register participants
        proposer_id = await transparency_system.register_democratic_participant(
            "proposer_001", "gold", ParticipationLevel.PROPOSER
        )

        voters = []
        for i in range(3):
            voter_id = await transparency_system.register_democratic_participant(
                f"voter_{i}", "silver", ParticipationLevel.VOTER
            )
            voters.append(voter_id)

        # Submit governance proposal
        proposal_data = {
            "proposal_type": "constitutional_improvement",
            "title": "Enhance Constitutional Transparency",
            "description": "Proposal to improve transparency mechanisms in constitutional governance",
            "implementation_plan": {"timeline": "60_days", "budget_required": False},
            "voting_period_days": 3,
            "quorum_requirement": 3,
            "approval_threshold": 0.6,
        }

        proposal_id = await transparency_system.submit_governance_proposal(proposer_id, proposal_data)
        assert proposal_id is not None

        # Verify proposal exists
        proposal = transparency_system.governance_audit.proposals.get(proposal_id)
        assert proposal is not None
        assert proposal.title == proposal_data["title"]
        assert proposal.current_status == "voting"

        # Cast votes
        vote_choices = ["approve", "approve", "reject"]
        vote_ids = []

        for voter_id, choice in zip(voters, vote_choices):
            vote_id = await transparency_system.governance_audit.cast_democratic_vote(
                voter_id, proposal_id, choice, f"Rationale for {choice} vote"
            )
            vote_ids.append(vote_id)
            assert vote_id is not None

        # Verify votes were cast
        for vote_id in vote_ids:
            vote = transparency_system.governance_audit.votes.get(vote_id)
            assert vote is not None
            assert vote.proposal_id == proposal_id

        # Check proposal vote counts
        updated_proposal = transparency_system.governance_audit.proposals.get(proposal_id)
        assert updated_proposal.votes_for == 2  # Two approve votes
        assert updated_proposal.votes_against == 1  # One reject vote
        assert updated_proposal.total_voting_power > 0


class TestPublicDashboard:
    """Test public accountability dashboard functionality"""

    @pytest.mark.asyncio
    async def test_dashboard_data_generation(self, transparency_system, sample_decision_data):
        """Test dashboard data generation for different user tiers"""
        # Log some decisions first
        for tier in ["bronze", "silver", "gold", "platinum"]:
            await transparency_system.log_constitutional_decision_comprehensive(
                sample_decision_data,
                tier,
                f"{tier}_user",
                ConstitutionalViolationType.HARM_CLASSIFICATION if tier == "bronze" else None,
                GovernanceLevel.MODERATED,
            )

        # Wait a moment for dashboard to update
        await asyncio.sleep(1)

        # Test dashboard data for different user tiers
        tiers_to_test = ["bronze", "silver", "gold", "platinum", None]  # None = public access

        for user_tier in tiers_to_test:
            dashboard_data = await transparency_system.dashboard.get_public_dashboard_data(user_tier)

            assert "dashboard_info" in dashboard_data
            assert "widgets" in dashboard_data
            assert "system_status" in dashboard_data

            # Verify system status
            assert dashboard_data["system_status"]["operational"] is True
            assert dashboard_data["system_status"]["constitutional_system_health"] == "operational"

            # Verify widgets exist (number may vary based on tier permissions)
            assert len(dashboard_data["widgets"]) > 0

    @pytest.mark.asyncio
    async def test_constitutional_metrics_summary(self, transparency_system, sample_decision_data):
        """Test constitutional metrics summary generation"""
        # Log decisions to generate metrics
        for i in range(5):
            await transparency_system.log_constitutional_decision_comprehensive(
                sample_decision_data,
                "bronze",
                f"user_{i}",
                ConstitutionalViolationType.HARM_CLASSIFICATION if i % 2 == 0 else None,
                GovernanceLevel.MODERATED,
            )

        # Wait for metrics to update
        await asyncio.sleep(1)

        # Get metrics summary
        metrics_summary = await transparency_system.dashboard.get_constitutional_metrics_summary()

        # Verify structure
        assert "constitutional_compliance" in metrics_summary
        assert "democratic_governance" in metrics_summary
        assert "transparency_and_privacy" in metrics_summary
        assert "system_integrity" in metrics_summary
        assert "summary_metadata" in metrics_summary

        # Verify data
        compliance_data = metrics_summary["constitutional_compliance"]
        assert "overall_rate" in compliance_data
        assert "total_decisions" in compliance_data
        assert compliance_data["total_decisions"] == 5

        # Verify public accountability score
        summary_metadata = metrics_summary["summary_metadata"]
        assert "public_accountability_score" in summary_metadata
        assert 0 <= summary_metadata["public_accountability_score"] <= 100


class TestRealTimeCompliance:
    """Test real-time compliance monitoring"""

    @pytest.mark.asyncio
    async def test_compliance_monitor_initialization(self, transparency_system):
        """Test compliance monitor initialization and basic functionality"""
        # Create compliance monitor
        monitor = RealTimeComplianceMonitor(
            transparency_system.merkle_audit,
            transparency_system.decision_logger,
            transparency_system.privacy_system,
            transparency_system.governance_audit,
            transparency_system.crypto_verifier,
            monitoring_interval=1,  # 1 second for testing
        )

        # Verify initialization
        assert monitor.is_monitoring is False
        assert len(monitor.compliance_thresholds) > 0
        assert len(monitor.metric_calculators) > 0

        # Test metric calculations
        compliance_rate = monitor._calculate_compliance_rate()
        assert isinstance(compliance_rate, float)
        assert 0 <= compliance_rate <= 100

        system_integrity = monitor._calculate_system_integrity()
        assert isinstance(system_integrity, float)
        assert 0 <= system_integrity <= 100

    @pytest.mark.asyncio
    async def test_system_health_snapshot(self, transparency_system, sample_decision_data):
        """Test system health snapshot generation"""
        # Log some decisions to have data
        for i in range(3):
            await transparency_system.log_constitutional_decision_comprehensive(
                sample_decision_data,
                "bronze",
                f"test_user_{i}",
                ConstitutionalViolationType.HARM_CLASSIFICATION if i == 0 else None,
                GovernanceLevel.MODERATED,
            )

        # Create compliance monitor
        monitor = RealTimeComplianceMonitor(
            transparency_system.merkle_audit,
            transparency_system.decision_logger,
            transparency_system.privacy_system,
            transparency_system.governance_audit,
            transparency_system.crypto_verifier,
            monitoring_interval=5,
        )

        # Create health snapshot
        health_snapshot = await monitor._create_health_snapshot()

        # Verify snapshot structure
        assert health_snapshot.timestamp > 0
        assert isinstance(health_snapshot.overall_compliance_rate, float)
        assert isinstance(health_snapshot.active_violations, int)
        assert isinstance(health_snapshot.system_integrity_score, float)
        assert isinstance(health_snapshot.component_status, dict)
        assert isinstance(health_snapshot.performance_metrics, dict)

        # Verify component status
        expected_components = [
            "merkle_audit",
            "decision_logger",
            "privacy_system",
            "governance_audit",
            "crypto_verifier",
        ]
        for component in expected_components:
            assert component in health_snapshot.component_status
            assert health_snapshot.component_status[component] in ["operational", "initializing"]


class TestIntegrationScenarios:
    """Test complete integration scenarios across the transparency system"""

    @pytest.mark.asyncio
    async def test_complete_constitutional_violation_workflow(self, transparency_system, sample_decision_data):
        """Test complete workflow for handling constitutional violation"""
        # Step 1: Log constitutional violation decision
        violation_data = sample_decision_data.copy()
        violation_data["violation_severity"] = "high"
        violation_data["constitutional_context"]["violation_type"] = "harm_classification_violation"

        decision_ids = await transparency_system.log_constitutional_decision_comprehensive(
            violation_data,
            "bronze",  # Bronze tier for full transparency
            "violating_user_123",
            ConstitutionalViolationType.HARM_CLASSIFICATION,
            GovernanceLevel.MODERATED,
        )

        # Step 2: Verify comprehensive logging
        assert len(decision_ids) >= 2  # At least Merkle and decision log

        # Step 3: Verify integrity across all systems
        integrity_results = await transparency_system.verify_constitutional_decision_integrity(decision_ids)
        assert integrity_results["overall_integrity"]["verified"] is True

        # Step 4: Check public transparency
        public_data = await transparency_system.get_public_transparency_data("bronze")
        assert "constitutional_metrics" in public_data
        assert "dashboard" in public_data

        # Step 5: Verify audit trail completeness
        audit_summary = transparency_system.merkle_audit.get_public_audit_summary()
        assert audit_summary["total_entries"] >= 1
        assert audit_summary["violation_statistics"]["harm_classification"] >= 1

    @pytest.mark.asyncio
    async def test_cross_tier_transparency_consistency(self, transparency_system, sample_decision_data):
        """Test that transparency maintains consistency across different user tiers"""
        tiers = ["bronze", "silver", "gold", "platinum"]
        all_decision_ids = {}

        # Log same decision type for all tiers
        for tier in tiers:
            decision_ids = await transparency_system.log_constitutional_decision_comprehensive(
                sample_decision_data,
                tier,
                f"{tier}_user_consistency_test",
                None,  # No violation
                GovernanceLevel.AUTOMATED,
            )
            all_decision_ids[tier] = decision_ids

        # Verify all tiers have logged decisions
        for tier in tiers:
            assert len(all_decision_ids[tier]) >= 2

            # Verify integrity for each tier
            integrity_check = await transparency_system.verify_constitutional_decision_integrity(all_decision_ids[tier])
            assert integrity_check["overall_integrity"]["verified"] is True

        # Verify privacy levels are appropriate for each tier
        for tier in tiers:
            merkle_entry = next(
                (
                    e
                    for e in transparency_system.merkle_audit.audit_entries
                    if e.entry_id == all_decision_ids[tier]["merkle_audit_id"]
                ),
                None,
            )
            assert merkle_entry is not None

            expected_audit_levels = {
                "bronze": AuditLevel.BRONZE,
                "silver": AuditLevel.SILVER,
                "gold": AuditLevel.GOLD,
                "platinum": AuditLevel.PLATINUM,
            }

            assert merkle_entry.audit_level == expected_audit_levels[tier]

    @pytest.mark.asyncio
    async def test_system_comprehensive_transparency_report(self, transparency_system, sample_decision_data):
        """Test comprehensive transparency report generation"""
        # Create diverse data across the system

        # 1. Log various constitutional decisions
        decision_types = [
            ("bronze", ConstitutionalViolationType.HARM_CLASSIFICATION),
            ("silver", ConstitutionalViolationType.TIER_VIOLATION),
            ("gold", None),
            ("platinum", None),
        ]

        for tier, violation_type in decision_types:
            await transparency_system.log_constitutional_decision_comprehensive(
                sample_decision_data, tier, f"{tier}_report_user", violation_type, GovernanceLevel.MODERATED
            )

        # 2. Create governance activity
        proposer_id = await transparency_system.register_democratic_participant(
            "report_proposer", "gold", ParticipationLevel.PROPOSER
        )

        proposal_data = {
            "title": "Test Transparency Proposal",
            "description": "Testing proposal for transparency report",
            "implementation_plan": {},
        }

        await transparency_system.submit_governance_proposal(proposer_id, proposal_data)

        # 3. Generate comprehensive transparency report
        comprehensive_report = await transparency_system.generate_comprehensive_transparency_report()

        # Verify report structure
        assert "comprehensive_transparency_report" in comprehensive_report
        assert "constitutional_compliance" in comprehensive_report
        assert "democratic_governance" in comprehensive_report
        assert "privacy_preservation" in comprehensive_report
        assert "public_accountability" in comprehensive_report
        assert "system_integrity_verification" in comprehensive_report
        assert "transparency_metrics_summary" in comprehensive_report

        # Verify report data
        report_info = comprehensive_report["comprehensive_transparency_report"]
        assert report_info["reporting_components"] == 4
        assert "generated_at" in report_info

        # Verify metrics summary
        metrics_summary = comprehensive_report["transparency_metrics_summary"]
        assert metrics_summary["total_constitutional_decisions"] >= 4
        assert metrics_summary["total_governance_participants"] >= 1
        assert metrics_summary["public_accountability_score"] > 0


# Performance and stress tests
class TestTransparencySystemPerformance:
    """Test system performance under load"""

    @pytest.mark.asyncio
    async def test_concurrent_decision_logging(self, transparency_system, sample_decision_data):
        """Test concurrent constitutional decision logging"""

        # Create multiple concurrent logging tasks
        async def log_decision(tier, user_id):
            return await transparency_system.log_constitutional_decision_comprehensive(
                sample_decision_data, tier, user_id, None, GovernanceLevel.AUTOMATED
            )

        # Create 20 concurrent logging tasks
        tasks = []
        for i in range(20):
            tier = ["bronze", "silver", "gold", "platinum"][i % 4]
            user_id = f"concurrent_user_{i}"
            tasks.append(log_decision(tier, user_id))

        # Execute all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        # Verify all succeeded
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 20

        # Verify performance (should complete within reasonable time)
        total_time = end_time - start_time
        assert total_time < 30  # Should complete within 30 seconds

        # Verify all decisions were logged
        assert len(transparency_system.merkle_audit.audit_entries) >= 20
        assert len(transparency_system.decision_logger.decision_logs) >= 20

    @pytest.mark.asyncio
    async def test_dashboard_performance_under_load(self, transparency_system, sample_decision_data):
        """Test dashboard performance with large amounts of data"""
        # Generate significant amount of data
        for i in range(100):
            tier = ["bronze", "silver", "gold", "platinum"][i % 4]
            violation = ConstitutionalViolationType.HARM_CLASSIFICATION if i % 5 == 0 else None

            await transparency_system.log_constitutional_decision_comprehensive(
                sample_decision_data, tier, f"load_test_user_{i}", violation, GovernanceLevel.AUTOMATED
            )

        # Test dashboard data generation performance
        start_time = time.time()
        dashboard_data = await transparency_system.get_public_transparency_data("bronze")
        end_time = time.time()

        # Verify response time is reasonable
        response_time = end_time - start_time
        assert response_time < 10  # Should respond within 10 seconds

        # Verify data integrity
        assert "dashboard" in dashboard_data
        assert "constitutional_metrics" in dashboard_data

        # Verify metrics reflect the load test data
        metrics = dashboard_data["constitutional_metrics"]
        assert metrics["constitutional_compliance"]["total_decisions"] >= 100


# Utility functions for tests
def run_transparency_system_tests():
    """Run all transparency system tests"""
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])


if __name__ == "__main__":
    run_transparency_system_tests()
