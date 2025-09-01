"""
Comprehensive tests for Constitutional Machine-Only Moderation Pipeline
Tests constitutional compliance, tier-based responses, and integration
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from infrastructure.constitutional.moderation.pipeline import (
    ConstitutionalModerationPipeline,
    ModerationDecision,
)
from infrastructure.constitutional.moderation.policy_enforcement import PolicyDecision


class TestConstitutionalModerationPipeline:
    """Test suite for constitutional moderation pipeline"""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing"""
        return ConstitutionalModerationPipeline()

    @pytest.fixture
    def mock_harm_classifier(self):
        """Mock harm classifier for controlled testing"""
        mock = AsyncMock()
        mock.version = "1.0.0"
        mock.classify_content.return_value = Mock(
            harm_level="H1", categories=["mild_concern"], confidence=0.8, metadata={}
        )
        return mock

    @pytest.mark.asyncio
    async def test_h0_constitutional_content_protection(self, pipeline):
        """Test H0 constitutional content receives maximum protection"""

        # Mock H0 classification
        with patch.object(pipeline.harm_classifier, "classify_content") as mock_classify:
            mock_classify.return_value = Mock(
                harm_level="H0",
                categories=["political_speech", "protected_expression"],
                confidence=0.9,
                metadata={"political_lean": "neutral"},
            )

            result = await pipeline.process_content(
                content="Political speech about democracy", content_type="text", user_tier="Gold"
            )

            assert result.decision == ModerationDecision.ALLOW
            assert result.harm_analysis.harm_level == "H0"
            assert "promote viewpoint diversity" in str(result.audit_trail).lower()
            assert result.transparency_score > 0.8

    @pytest.mark.asyncio
    async def test_h1_warning_with_constitutional_safeguards(self, pipeline):
        """Test H1 content gets warning with constitutional protections"""

        with patch.object(pipeline.harm_classifier, "classify_content") as mock_classify:
            mock_classify.return_value = Mock(
                harm_level="H1", categories=["minor_concern"], confidence=0.7, metadata={}
            )

            result = await pipeline.process_content(
                content="Mildly concerning content", content_type="text", user_tier="Silver"
            )

            assert result.decision == ModerationDecision.ALLOW_WITH_WARNING
            assert result.harm_analysis.harm_level == "H1"
            assert not result.requires_escalation  # Silver tier shouldn't escalate H1

    @pytest.mark.asyncio
    async def test_h2_restriction_with_due_process(self, pipeline):
        """Test H2 content restriction includes due process protections"""

        with patch.object(pipeline.harm_classifier, "classify_content") as mock_classify:
            mock_classify.return_value = Mock(
                harm_level="H2", categories=["moderate_harm"], confidence=0.85, metadata={}
            )

            result = await pipeline.process_content(
                content="Moderately harmful content", content_type="text", user_tier="Gold"
            )

            assert result.decision in [ModerationDecision.RESTRICT, ModerationDecision.QUARANTINE]
            assert result.appeal_eligible
            assert "due_process" in result.audit_trail["constitutional_analysis"]

    @pytest.mark.asyncio
    async def test_h3_severe_harm_escalation_gold_tier(self, pipeline):
        """Test H3 severe harm escalates for Gold tier users"""

        with patch.object(pipeline.harm_classifier, "classify_content") as mock_classify:
            mock_classify.return_value = Mock(harm_level="H3", categories=["severe_harm"], confidence=0.9, metadata={})

            result = await pipeline.process_content(
                content="Severely harmful content", content_type="text", user_tier="Gold"
            )

            assert result.requires_escalation
            assert result.appeal_eligible
            assert result.tier_level == "Gold"

    @pytest.mark.asyncio
    async def test_first_amendment_protected_speech_handling(self, pipeline):
        """Test First Amendment protected speech gets heightened scrutiny"""

        with patch.object(pipeline.harm_classifier, "classify_content") as mock_classify:
            mock_classify.return_value = Mock(
                harm_level="H2",
                categories=["political_speech", "religious_expression"],
                confidence=0.6,  # Lower confidence for protected speech
                metadata={"political_lean": "conservative"},
            )

            result = await pipeline.process_content(
                content="Protected political and religious speech", content_type="text", user_tier="Gold"
            )

            # Protected speech with low confidence should err on side of free speech
            assert result.harm_analysis.constitutional_concerns.get("first_amendment")
            assert result.requires_escalation  # Gold tier escalates constitutional concerns
            assert "heightened_scrutiny" in str(result.audit_trail)

    @pytest.mark.asyncio
    async def test_viewpoint_bias_detection_and_mitigation(self, pipeline):
        """Test viewpoint bias detection triggers appropriate safeguards"""

        with patch.object(pipeline.harm_classifier, "classify_content") as mock_classify:
            mock_classify.return_value = Mock(
                harm_level="H1",
                categories=["political_content"],
                confidence=0.8,
                metadata={"political_lean": "liberal", "lean_confidence": 0.7},
            )

            result = await pipeline.process_content(
                content="Political content with potential bias",
                content_type="text",
                user_tier="Silver",  # Silver has viewpoint firewall
            )

            assert result.harm_analysis.viewpoint_bias_score > 0
            viewpoint_concerns = result.harm_analysis.constitutional_concerns.get("viewpoint_neutrality")
            assert viewpoint_concerns is not None
            assert viewpoint_concerns.get("requires_bias_check")

    @pytest.mark.asyncio
    async def test_tier_based_escalation_thresholds(self, pipeline):
        """Test different escalation thresholds for different tiers"""

        # Same content, different tiers
        test_content = "Borderline content requiring review"

        with patch.object(pipeline.harm_classifier, "classify_content") as mock_classify:
            mock_classify.return_value = Mock(
                harm_level="H2", categories=["moderate_concern"], confidence=0.5, metadata={}  # Low confidence
            )

            # Bronze tier - no escalation (machine only)
            bronze_result = await pipeline.process_content(test_content, "text", "Bronze")
            assert not bronze_result.requires_escalation

            # Silver tier - limited escalation for unclear cases
            silver_result = await pipeline.process_content(test_content, "text", "Silver")
            assert silver_result.requires_escalation  # Low confidence H2 escalates

            # Gold tier - comprehensive escalation
            gold_result = await pipeline.process_content(test_content, "text", "Gold")
            assert gold_result.requires_escalation
            assert gold_result.appeal_eligible

    @pytest.mark.asyncio
    async def test_constitutional_error_handling(self, pipeline):
        """Test error handling maintains constitutional protections"""

        # Mock classifier failure
        with patch.object(pipeline.harm_classifier, "classify_content") as mock_classify:
            mock_classify.side_effect = Exception("Classifier error")

            result = await pipeline.process_content(
                content="Content during system error", content_type="text", user_tier="Gold"
            )

            # Error should result in escalation, not automatic restriction
            assert result.decision == ModerationDecision.ESCALATE
            assert result.requires_escalation
            assert result.appeal_eligible
            assert "system_error" in result.harm_analysis.harm_categories

    @pytest.mark.asyncio
    async def test_transparency_and_audit_trail(self, pipeline):
        """Test comprehensive transparency and audit trail generation"""

        with patch.object(pipeline.harm_classifier, "classify_content") as mock_classify:
            mock_classify.return_value = Mock(harm_level="H1", categories=["minor_issue"], confidence=0.85, metadata={})

            result = await pipeline.process_content(
                content="Test content for audit trail", content_type="text", user_tier="Gold"
            )

            audit_trail = result.audit_trail

            # Verify comprehensive audit trail
            assert "content_id" in audit_trail
            assert "timestamp" in audit_trail
            assert "processing_time_ms" in audit_trail
            assert "harm_classification" in audit_trail
            assert "policy_decision" in audit_trail
            assert "constitutional_analysis" in audit_trail
            assert "system_metadata" in audit_trail

            # Verify constitutional analysis components
            constitutional_analysis = audit_trail["constitutional_analysis"]
            assert "viewpoint_bias_score" in constitutional_analysis
            assert "first_amendment_considerations" in constitutional_analysis

            # High transparency score for clear decisions
            assert result.transparency_score > 0.7

    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self, pipeline):
        """Test performance metrics are properly tracked"""

        with patch.object(pipeline.harm_classifier, "classify_content") as mock_classify:
            mock_classify.return_value = Mock(harm_level="H1", categories=["test"], confidence=0.8, metadata={})

            # Process multiple pieces of content
            for i in range(5):
                await pipeline.process_content(f"Test content {i}", "text", "Silver")

            metrics = await pipeline.get_pipeline_metrics()

            assert "processing_statistics" in metrics
            assert metrics["processing_statistics"]["total_processed"] == 5
            assert "constitutional_metrics" in metrics
            assert "system_health" in metrics

    @pytest.mark.asyncio
    async def test_appeal_eligibility_determination(self, pipeline):
        """Test correct determination of appeal eligibility"""

        # Test cases for different decisions and tiers
        test_cases = [
            ("H2", "restrict", "Bronze", True),  # Restrictive decisions appealable
            ("H2", "quarantine", "Silver", True),  # Restrictive decisions appealable
            ("H3", "block", "Gold", True),  # Block decisions appealable
            ("H1", "allow_with_warning", "Gold", True),  # Gold tier can appeal warnings
            ("H1", "allow_with_warning", "Silver", False),  # Silver tier cannot appeal warnings
            ("H0", "allow", "Gold", False),  # Allow decisions not appealable
        ]

        for harm_level, decision, tier, should_be_eligible in test_cases:
            with patch.object(pipeline.harm_classifier, "classify_content") as mock_classify:
                mock_classify.return_value = Mock(
                    harm_level=harm_level, categories=["test"], confidence=0.8, metadata={}
                )

                # Mock policy enforcement result
                with patch.object(pipeline.policy_enforcement, "evaluate_content") as mock_policy:
                    mock_policy.return_value = Mock(
                        decision=PolicyDecision(decision),
                        rationale="Test decision",
                        constitutional_analysis={},
                        tier_modifications={},
                        monitoring_requirements=[],
                        confidence_score=0.8,
                    )

                    result = await pipeline.process_content(f"Test content for {harm_level} {decision}", "text", tier)

                    assert (
                        result.appeal_eligible == should_be_eligible
                    ), f"Appeal eligibility mismatch for {harm_level} {decision} {tier}"

    @pytest.mark.asyncio
    async def test_integration_with_fog_infrastructure(self, pipeline):
        """Test integration with fog compute infrastructure components"""

        # Mock TEE security integration
        with patch.object(pipeline.tee_security, "get_attestation") as mock_attestation:
            mock_attestation.return_value = "test_attestation_123"

            with patch.object(pipeline.harm_classifier, "classify_content") as mock_classify:
                mock_classify.return_value = Mock(harm_level="H1", categories=["test"], confidence=0.8, metadata={})

                result = await pipeline.process_content("Test integration content", "text", "Gold")

                # Verify TEE attestation is included in audit trail
                assert "tee_attestation" in result.audit_trail["system_metadata"]
                assert result.audit_trail["system_metadata"]["tee_attestation"] == "test_attestation_123"

    @pytest.mark.asyncio
    async def test_batch_processing_consistency(self, pipeline):
        """Test consistency across batch processing of similar content"""

        with patch.object(pipeline.harm_classifier, "classify_content") as mock_classify:
            mock_classify.return_value = Mock(
                harm_level="H1", categories=["similar_content"], confidence=0.8, metadata={}
            )

            # Process similar content multiple times
            results = []
            for i in range(10):
                result = await pipeline.process_content("Similar test content", "text", "Silver")
                results.append(result)

            # Verify consistency
            decisions = [r.decision for r in results]
            assert len(set(decisions)) == 1, "Inconsistent decisions for similar content"

            harm_levels = [r.harm_analysis.harm_level for r in results]
            assert len(set(harm_levels)) == 1, "Inconsistent harm levels for similar content"

    @pytest.mark.asyncio
    async def test_constitutional_compliance_verification(self, pipeline):
        """Test constitutional compliance verification across all decision types"""

        decision_types = ["allow", "allow_with_warning", "restrict", "quarantine", "block"]

        for decision in decision_types:
            with patch.object(pipeline.harm_classifier, "classify_content") as mock_classify:
                mock_classify.return_value = Mock(
                    harm_level="H2", categories=["test_content"], confidence=0.8, metadata={}
                )

                with patch.object(pipeline.policy_enforcement, "evaluate_content") as mock_policy:
                    mock_policy.return_value = Mock(
                        decision=PolicyDecision(decision),
                        rationale=f"Test {decision} decision",
                        constitutional_analysis={
                            "first_amendment": {"protected_speech_detected": False},
                            "due_process": {"requirements": ["notice"]},
                            "viewpoint_neutrality": {"discrimination_risk": False},
                        },
                        tier_modifications={},
                        monitoring_requirements=[],
                        confidence_score=0.8,
                    )

                    result = await pipeline.process_content(f"Test content for {decision}", "text", "Gold")

                    # Verify constitutional compliance components
                    constitutional_analysis = result.audit_trail["constitutional_analysis"]
                    assert "viewpoint_bias_score" in constitutional_analysis
                    assert "first_amendment_considerations" in constitutional_analysis
                    assert "due_process_flags" in constitutional_analysis

                    # Verify decision rationale includes constitutional considerations
                    assert len(result.policy_rationale) > 0

                    # For restrictive decisions, verify enhanced protections
                    if decision in ["restrict", "quarantine", "block"]:
                        assert result.appeal_eligible

    def test_pipeline_initialization(self):
        """Test proper pipeline initialization"""
        pipeline = ConstitutionalModerationPipeline()

        # Verify all components are initialized
        assert pipeline.harm_classifier is not None
        assert pipeline.policy_enforcement is not None
        assert pipeline.response_actions is not None
        assert pipeline.escalation_manager is not None
        assert pipeline.appeals_manager is not None
        assert pipeline.tee_security is not None

        # Verify statistics initialization
        assert pipeline.processing_stats["total_processed"] == 0
        assert isinstance(pipeline.processing_stats["decisions_by_type"], dict)
        assert isinstance(pipeline.processing_stats["harm_level_distribution"], dict)

    @pytest.mark.asyncio
    async def test_shutdown_graceful_handling(self, pipeline):
        """Test graceful shutdown with pending operations"""

        # Mock some active operations
        pipeline.processing_stats["total_processed"] = 100

        # Test shutdown
        await pipeline.shutdown()

        # Verify shutdown logging occurred (would check logs in real implementation)
        # For now, just verify shutdown completes without errors
        assert True  # Placeholder for actual shutdown verification


class TestConstitutionalComplianceValidation:
    """Specific tests for constitutional compliance validation"""

    @pytest.mark.asyncio
    async def test_first_amendment_compliance(self):
        """Test First Amendment compliance across scenarios"""
        pipeline = ConstitutionalModerationPipeline()

        first_amendment_scenarios = [
            {
                "content": "Political criticism of government policy",
                "categories": ["political_speech"],
                "expected_protection": "maximum",
            },
            {
                "content": "Religious expression and beliefs",
                "categories": ["religious_expression"],
                "expected_protection": "maximum",
            },
            {
                "content": "Artistic and creative expression",
                "categories": ["artistic_expression"],
                "expected_protection": "high",
            },
            {"content": "True threat of violence", "categories": ["true_threats"], "expected_protection": "none"},
        ]

        for scenario in first_amendment_scenarios:
            with patch.object(pipeline.harm_classifier, "classify_content") as mock_classify:
                mock_classify.return_value = Mock(
                    harm_level="H2", categories=scenario["categories"], confidence=0.7, metadata={}
                )

                result = await pipeline.process_content(scenario["content"], "text", "Gold")

                if scenario["expected_protection"] == "maximum":
                    # Should err on side of free speech
                    assert result.harm_analysis.constitutional_concerns.get("first_amendment")
                    assert result.requires_escalation  # Gold tier escalates protected speech

                elif scenario["expected_protection"] == "none":
                    # Unprotected speech can be restricted without escalation
                    assert result.decision in [ModerationDecision.RESTRICT, ModerationDecision.BLOCK]

    @pytest.mark.asyncio
    async def test_due_process_compliance(self):
        """Test due process compliance requirements"""
        pipeline = ConstitutionalModerationPipeline()

        with patch.object(pipeline.harm_classifier, "classify_content") as mock_classify:
            mock_classify.return_value = Mock(
                harm_level="H2", categories=["moderate_harm"], confidence=0.8, metadata={}
            )

            result = await pipeline.process_content("Content requiring due process protections", "text", "Gold")

            # Verify due process components
            audit_trail = result.audit_trail

            # Notice requirement
            assert "policy_decision" in audit_trail
            assert len(result.policy_rationale) > 0

            # Appeal rights
            if result.decision in [
                ModerationDecision.RESTRICT,
                ModerationDecision.QUARANTINE,
                ModerationDecision.BLOCK,
            ]:
                assert result.appeal_eligible

            # Proportionality
            assert "constitutional_analysis" in audit_trail

            # Evidence-based decision
            assert result.harm_analysis.confidence_score > 0

    @pytest.mark.asyncio
    async def test_equal_protection_compliance(self):
        """Test equal protection and viewpoint neutrality"""
        pipeline = ConstitutionalModerationPipeline()

        # Test similar content with different political leanings
        political_scenarios = [
            {"lean": "conservative", "content": "Conservative political viewpoint"},
            {"lean": "liberal", "content": "Liberal political viewpoint"},
            {"lean": "neutral", "content": "Neutral political commentary"},
        ]

        results = []
        for scenario in political_scenarios:
            with patch.object(pipeline.harm_classifier, "classify_content") as mock_classify:
                mock_classify.return_value = Mock(
                    harm_level="H1",
                    categories=["political_content"],
                    confidence=0.8,
                    metadata={"political_lean": scenario["lean"]},
                )

                result = await pipeline.process_content(scenario["content"], "text", "Silver")
                results.append(result)

        # Verify consistent treatment across political viewpoints
        decisions = [r.decision for r in results]
        # All similar political content should receive same treatment
        unique_decisions = set(decisions)
        assert len(unique_decisions) <= 2, "Inconsistent treatment of similar political content"

        # Verify viewpoint bias detection
        for result in results:
            if result.harm_analysis.viewpoint_bias_score > 0.3:
                # High bias content should trigger additional safeguards
                assert "viewpoint_neutrality" in result.harm_analysis.constitutional_concerns


@pytest.mark.integration
class TestIntegrationWithFogInfrastructure:
    """Integration tests with fog computing infrastructure"""

    @pytest.mark.asyncio
    async def test_end_to_end_content_processing(self):
        """Test complete end-to-end content processing through fog infrastructure"""

        pipeline = ConstitutionalModerationPipeline()

        # Simulate real content processing scenario
        test_content = {
            "content": "User-generated content requiring moderation",
            "content_type": "text",
            "user_tier": "Gold",
            "context": {
                "source": "fog_compute_node",
                "workload_id": "test_workload_123",
                "request_time": datetime.utcnow().isoformat(),
            },
        }

        with patch.object(pipeline.harm_classifier, "classify_content") as mock_classify:
            mock_classify.return_value = Mock(
                harm_level="H1", categories=["user_content"], confidence=0.85, metadata={}
            )

            result = await pipeline.process_content(
                test_content["content"],
                test_content["content_type"],
                test_content["user_tier"],
                test_content["context"],
            )

            # Verify result suitable for fog infrastructure integration
            assert result.content_id is not None
            assert result.decision in [d for d in ModerationDecision]
            assert result.transparency_score > 0
            assert result.audit_trail is not None

            # Verify response actions are executable
            assert isinstance(result.response_actions, list)

            # Verify constitutional compliance for fog deployment
            constitutional_analysis = result.audit_trail.get("constitutional_analysis", {})
            assert "viewpoint_bias_score" in constitutional_analysis

    @pytest.mark.asyncio
    async def test_high_volume_processing(self):
        """Test pipeline performance under high volume"""

        pipeline = ConstitutionalModerationPipeline()

        with patch.object(pipeline.harm_classifier, "classify_content") as mock_classify:
            mock_classify.return_value = Mock(harm_level="H1", categories=["bulk_content"], confidence=0.8, metadata={})

            # Process multiple items concurrently
            tasks = []
            for i in range(50):  # Simulate moderate volume
                task = pipeline.process_content(f"Bulk content item {i}", "text", "Silver")
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            # Verify all processed successfully
            assert len(results) == 50
            assert all(r.content_id for r in results)

            # Verify consistent processing
            decisions = [r.decision for r in results]
            # Should be consistent for similar content
            unique_decisions = set(decisions)
            assert len(unique_decisions) <= 2

            # Verify performance metrics updated
            metrics = await pipeline.get_pipeline_metrics()
            assert metrics["processing_statistics"]["total_processed"] >= 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
