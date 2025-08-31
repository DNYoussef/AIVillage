"""
Integration Tests for Constitutional H200-Hour Pricing System

Tests the comprehensive constitutional fog compute pricing enhancements:
- H200-hour equivalent calculations
- Constitutional tier pricing
- TEE-enhanced workload pricing 
- Audit trail compliance
- Governance pricing mechanisms
"""

import asyncio
from decimal import Decimal
import pytest
from unittest.mock import AsyncMock, MagicMock

from infrastructure.fog.market.pricing_manager import (
    DynamicPricingManager,
    UserSizeTier,
    ResourceLane,
    H200HourPricing,
    ConstitutionalTierMapping
)
from infrastructure.fog.market.constitutional_pricing import (
    ConstitutionalPricingEngine,
    H200EquivalentDevice,
    ConstitutionalLevel
)
from infrastructure.fog.market.audit_pricing import (
    AuditTrailManager,
    AuditEventType,
    ComplianceLevel
)


@pytest.fixture
async def pricing_manager():
    """Create pricing manager with constitutional features"""
    audit_manager = AuditTrailManager()
    manager = DynamicPricingManager(audit_manager=audit_manager)
    await manager.start()
    return manager


@pytest.fixture
def constitutional_engine():
    """Create constitutional pricing engine"""
    audit_manager = AuditTrailManager()
    return ConstitutionalPricingEngine(audit_manager)


@pytest.fixture
def sample_device():
    """Create sample H200 equivalent device"""
    return H200EquivalentDevice(
        device_id="test_gpu_001",
        device_type="gpu", 
        computing_power_tops=Decimal("500"),  # 500 TOPS
        memory_gb=Decimal("80"),
        power_efficiency=Decimal("0.7"),
        privacy_hardware=True,
        governance_participation=True,
        audit_logging_enabled=True
    )


class TestH200HourCalculations:
    """Test H200-hour equivalent calculations"""
    
    async def test_h200_hour_formula_accuracy(self, pricing_manager):
        """Test H200-hour calculation formula: H200h(d) = (TOPS_d × u × t) / T_ref"""
        
        # Test case: Device with 500 TOPS, 80% utilization, 2 hours
        device_tops = Decimal("500")
        utilization = Decimal("0.8")
        time_hours = Decimal("2")
        
        result = await pricing_manager.calculate_h200_hour_equivalent(
            device_tops, utilization, time_hours
        )
        
        # Expected: (500 × 0.8 × 2) / 989 = 0.809 H200-hours
        expected_h200_hours = (device_tops * utilization * time_hours) / Decimal("989")
        
        assert abs(Decimal(str(result["h200_hours_equivalent"])) - expected_h200_hours) < Decimal("0.001")
        assert result["calculation_formula"] == "H200h(d) = (TOPS_d × u × t) / T_ref"
        assert result["h200_reference_tops"] == 989.0
    
    async def test_h200_hour_edge_cases(self, pricing_manager):
        """Test H200-hour calculations with edge cases"""
        
        # Zero utilization case
        result_zero = await pricing_manager.calculate_h200_hour_equivalent(
            Decimal("1000"), Decimal("0"), Decimal("1")
        )
        assert result_zero["h200_hours_equivalent"] == 0.0
        
        # Full utilization case
        result_full = await pricing_manager.calculate_h200_hour_equivalent(
            Decimal("989"), Decimal("1.0"), Decimal("1")  # Exactly H200 specs
        )
        assert abs(result_full["h200_hours_equivalent"] - 1.0) < 0.001
        
        # High-performance device
        result_high = await pricing_manager.calculate_h200_hour_equivalent(
            Decimal("1978"), Decimal("0.5"), Decimal("1")  # 2x H200 performance
        )
        assert abs(result_high["h200_hours_equivalent"] - 1.0) < 0.001
    
    async def test_power_efficiency_calculation(self, pricing_manager):
        """Test power efficiency ratio calculations"""
        
        result = await pricing_manager.calculate_h200_hour_equivalent(
            Decimal("989"), Decimal("1.0"), Decimal("1"), "test_device"
        )
        
        # Power efficiency ratio should be computed
        assert "power_efficiency_ratio" in result
        assert isinstance(result["power_efficiency_ratio"], float)
        assert result["power_efficiency_ratio"] > 0


class TestConstitutionalTierPricing:
    """Test constitutional tier pricing with Bronze/Silver/Gold/Platinum"""
    
    async def test_bronze_tier_pricing(self, pricing_manager):
        """Test Bronze tier constitutional pricing"""
        
        quote = await pricing_manager.get_constitutional_h200_price(
            user_tier=UserSizeTier.BRONZE,
            device_computing_power_tops=Decimal("500"),
            utilization_rate=Decimal("0.8"),
            time_hours=Decimal("1"),
            constitutional_level="enhanced",
            tee_enabled=False
        )
        
        assert quote["user_tier"] == "bronze"
        assert quote["constitutional_level"] == "enhanced"
        assert "h200_calculation" in quote
        assert "pricing" in quote
        assert quote["pricing"]["base_rate_per_h200_hour"] == 0.5  # $0.50 per H200-hour
        
        # Should have constitutional discount
        assert "constitutional_discount" in quote["pricing"]["adjustments"]
        assert quote["pricing"]["adjustments"]["constitutional_discount"] < 0
    
    async def test_silver_tier_pricing(self, pricing_manager):
        """Test Silver tier constitutional pricing"""
        
        quote = await pricing_manager.get_constitutional_h200_price(
            user_tier=UserSizeTier.SILVER,
            device_computing_power_tops=Decimal("1000"),
            utilization_rate=Decimal("0.9"),
            time_hours=Decimal("2"),
            constitutional_level="full_audit"
        )
        
        assert quote["user_tier"] == "silver"
        assert quote["pricing"]["base_rate_per_h200_hour"] == 0.75  # $0.75 per H200-hour
        
        # Should have both constitutional and audit transparency bonuses
        adjustments = quote["pricing"]["adjustments"]
        assert "constitutional_discount" in adjustments
        assert "audit_transparency_bonus" in adjustments
        assert adjustments["constitutional_discount"] < 0
        assert adjustments["audit_transparency_bonus"] < 0
    
    async def test_gold_tier_pricing(self, pricing_manager):
        """Test Gold tier constitutional pricing"""
        
        quote = await pricing_manager.get_constitutional_h200_price(
            user_tier=UserSizeTier.GOLD,
            device_computing_power_tops=Decimal("750"),
            utilization_rate=Decimal("1.0"),
            time_hours=Decimal("4"),
            constitutional_level="constitutional"
        )
        
        assert quote["user_tier"] == "gold"  
        assert quote["pricing"]["base_rate_per_h200_hour"] == 1.0  # $1.00 per H200-hour
        
        # Gold tier should have highest constitutional benefits
        assert quote["tier_limits"]["constitutional_discount"] == 0.10  # 10%
        assert quote["tier_limits"]["audit_transparency_bonus"] == 0.08  # 8%
    
    async def test_platinum_tier_pricing(self, pricing_manager):
        """Test Platinum tier constitutional pricing"""
        
        quote = await pricing_manager.get_constitutional_h200_price(
            user_tier=UserSizeTier.PLATINUM,
            device_computing_power_tops=Decimal("2000"),
            utilization_rate=Decimal("0.7"),
            time_hours=Decimal("8"),
            constitutional_level="constitutional",
            tee_enabled=True
        )
        
        assert quote["user_tier"] == "platinum"
        assert quote["pricing"]["base_rate_per_h200_hour"] == 1.5  # $1.50 per H200-hour
        
        # Platinum should have all benefits plus TEE premium
        adjustments = quote["pricing"]["adjustments"]
        assert "constitutional_discount" in adjustments
        assert "audit_transparency_bonus" in adjustments  
        assert "tee_security_premium" in adjustments
        assert adjustments["tee_security_premium"] == 0.30  # 30% TEE premium
    
    async def test_tier_monthly_limits(self, pricing_manager):
        """Test constitutional tier monthly limits"""
        
        # Check each tier's monthly H200-hour limits
        bronze_quote = await pricing_manager.get_constitutional_h200_price(
            UserSizeTier.BRONZE, Decimal("100"), Decimal("1"), Decimal("1")
        )
        assert bronze_quote["tier_limits"]["max_h200_hours_monthly"] == 100
        
        silver_quote = await pricing_manager.get_constitutional_h200_price(
            UserSizeTier.SILVER, Decimal("100"), Decimal("1"), Decimal("1")
        )
        assert silver_quote["tier_limits"]["max_h200_hours_monthly"] == 500
        
        gold_quote = await pricing_manager.get_constitutional_h200_price(
            UserSizeTier.GOLD, Decimal("100"), Decimal("1"), Decimal("1")
        )
        assert gold_quote["tier_limits"]["max_h200_hours_monthly"] == 2000
        
        platinum_quote = await pricing_manager.get_constitutional_h200_price(
            UserSizeTier.PLATINUM, Decimal("100"), Decimal("1"), Decimal("1")
        )
        assert platinum_quote["tier_limits"]["max_h200_hours_monthly"] == 10000


class TestTEEEnhancedPricing:
    """Test TEE-enhanced workload pricing"""
    
    async def test_basic_tee_pricing(self, pricing_manager):
        """Test basic TEE security premium"""
        
        tee_quote = await pricing_manager.get_tee_enhanced_pricing(
            lane=ResourceLane.GPU,
            quantity=Decimal("1"),
            duration_hours=Decimal("1"),
            tee_level="basic"
        )
        
        assert tee_quote["tee_level"] == "basic"
        assert tee_quote["tee_enhanced_pricing"]["tee_premium_percentage"] == 20.0  # 20%
        assert tee_quote["tee_enhanced_pricing"]["constitutional_bonus"] == 5.0  # 5% bonus
        
        # Check TEE features
        features = tee_quote["tee_features"]
        assert features["hardware_security"] is True
        assert features["encrypted_computation"] is True
        assert features["constitutional_compliant"] is True
    
    async def test_enhanced_tee_pricing(self, pricing_manager):
        """Test enhanced TEE pricing with attestation"""
        
        tee_quote = await pricing_manager.get_tee_enhanced_pricing(
            lane=ResourceLane.FEDERATED_TRAINING,
            quantity=Decimal("2"),
            duration_hours=Decimal("4"),
            tee_level="enhanced"
        )
        
        assert tee_quote["tee_level"] == "enhanced"
        assert tee_quote["tee_enhanced_pricing"]["tee_premium_percentage"] == 35.0  # 35%
        
        # Enhanced should include attestation
        assert tee_quote["tee_features"]["attestation_available"] is True
        assert tee_quote["tee_features"]["confidential_computing"] is False
    
    async def test_confidential_computing_pricing(self, pricing_manager):
        """Test confidential computing premium pricing"""
        
        tee_quote = await pricing_manager.get_tee_enhanced_pricing(
            lane=ResourceLane.TEE_SECURE,
            quantity=Decimal("1"),
            duration_hours=Decimal("2"),
            tee_level="confidential"
        )
        
        assert tee_quote["tee_level"] == "confidential"
        assert tee_quote["tee_enhanced_pricing"]["tee_premium_percentage"] == 50.0  # 50%
        
        # Confidential should have all security features
        features = tee_quote["tee_features"]
        assert features["attestation_available"] is True
        assert features["confidential_computing"] is True
        assert features["constitutional_compliant"] is True
    
    async def test_tee_cost_calculation_accuracy(self, pricing_manager):
        """Test TEE cost calculation accuracy"""
        
        # Mock base pricing to control test
        base_cost = Decimal("100.00")
        
        tee_quote = await pricing_manager.get_tee_enhanced_pricing(
            lane=ResourceLane.CPU,
            quantity=Decimal("1"),
            duration_hours=Decimal("1"),
            tee_level="enhanced"
        )
        
        # Verify cost calculation: base * (1 + tee_premium) * (1 - constitutional_bonus)
        base_from_quote = tee_quote["tee_enhanced_pricing"]["base_cost"]
        tee_premium = Decimal("0.35")  # 35% for enhanced
        constitutional_bonus = Decimal("0.05")  # 5% constitutional bonus
        
        expected_tee_cost = base_from_quote * (Decimal("1.0") + tee_premium)
        expected_final = expected_tee_cost * (Decimal("1.0") - constitutional_bonus)
        
        assert abs(Decimal(str(tee_quote["tee_enhanced_pricing"]["final_cost"])) - expected_final) < Decimal("0.01")


class TestAuditTrailIntegration:
    """Test audit trail integration with pricing"""
    
    async def test_pricing_calculation_audit_logging(self, pricing_manager):
        """Test that pricing calculations are logged to audit trail"""
        
        # Perform H200-hour calculation
        result = await pricing_manager.calculate_h200_hour_equivalent(
            Decimal("500"), Decimal("0.8"), Decimal("2"), "test_device_123"
        )
        
        # Check audit trail
        audit_manager = pricing_manager.audit_manager
        assert len(audit_manager.audit_records) > 0
        
        # Find the calculation record
        calc_records = [
            r for r in audit_manager.audit_records 
            if r.event_type == AuditEventType.PRICING_CALCULATION
        ]
        assert len(calc_records) > 0
        
        record = calc_records[-1]  # Most recent
        assert record.device_id == "test_device_123"
        assert record.event_data["calculation_type"] == "h200_hour_calculation"
    
    async def test_pricing_quote_audit_logging(self, pricing_manager):
        """Test that pricing quotes are logged to audit trail"""
        
        # Generate constitutional quote
        quote = await pricing_manager.get_constitutional_h200_price(
            UserSizeTier.SILVER,
            Decimal("1000"),
            Decimal("0.9"),
            Decimal("1"),
            "enhanced",
            True
        )
        
        # Check audit trail for quote
        audit_manager = pricing_manager.audit_manager
        quote_records = [
            r for r in audit_manager.audit_records
            if r.event_type == AuditEventType.QUOTE_GENERATION
        ]
        assert len(quote_records) > 0
        
        record = quote_records[-1]  # Most recent
        assert record.transaction_id == quote["quote_id"]
        assert record.constitutional_features.get("transparency_enabled") is True
    
    async def test_tee_pricing_audit_logging(self, pricing_manager):
        """Test that TEE pricing is logged to audit trail"""
        
        # Generate TEE pricing quote
        tee_quote = await pricing_manager.get_tee_enhanced_pricing(
            ResourceLane.SPECIALIZED,
            Decimal("1"),
            Decimal("1"),
            "confidential",
            "tee_node_456"
        )
        
        # Check audit trail for TEE pricing calculation
        audit_manager = pricing_manager.audit_manager
        tee_records = [
            r for r in audit_manager.audit_records
            if r.event_data.get("calculation_type") == "tee_enhanced_pricing"
        ]
        assert len(tee_records) > 0
        
        record = tee_records[-1]
        assert record.device_id == "tee_node_456"
    
    async def test_audit_chain_integrity(self, pricing_manager):
        """Test audit chain integrity verification"""
        
        # Generate multiple pricing operations
        for i in range(5):
            await pricing_manager.calculate_h200_hour_equivalent(
                Decimal("100"), Decimal("0.5"), Decimal("1"), f"device_{i}"
            )
        
        # Verify audit chain integrity
        audit_manager = pricing_manager.audit_manager
        integrity_result = audit_manager.verify_audit_chain_integrity()
        
        assert integrity_result["chain_valid"] is True
        assert integrity_result["verified_records"] == integrity_result["total_records"]
        assert len(integrity_result["integrity_failures"]) == 0


class TestConstitutionalCompliance:
    """Test constitutional compliance mechanisms"""
    
    async def test_transparency_features(self, pricing_manager):
        """Test pricing transparency features"""
        
        quote = await pricing_manager.get_constitutional_h200_price(
            UserSizeTier.GOLD,
            Decimal("500"),
            Decimal("1.0"),
            Decimal("1"),
            "constitutional"
        )
        
        # Check constitutional features
        features = quote["constitutional_features"]
        assert features["transparency_enabled"] is True
        assert features["audit_trail"] is True
        assert features["governance_participation"] is True
    
    async def test_compliance_report_generation(self, pricing_manager):
        """Test constitutional compliance report generation"""
        
        # Generate various pricing operations
        await pricing_manager.get_constitutional_h200_price(
            UserSizeTier.BRONZE, Decimal("200"), Decimal("0.8"), Decimal("1")
        )
        await pricing_manager.get_tee_enhanced_pricing(
            ResourceLane.GPU, Decimal("1"), Decimal("1"), "enhanced"
        )
        
        # Generate compliance report
        audit_manager = pricing_manager.audit_manager
        compliance_report = audit_manager.generate_constitutional_compliance_report()
        
        assert compliance_report.pricing_transparency is True
        assert compliance_report.audit_trail_complete is True
        assert compliance_report.privacy_protections_active is True
        assert compliance_report.overall_compliance_score > Decimal("0.8")
    
    async def test_governance_participation_tracking(self, constitutional_engine):
        """Test governance participation in pricing"""
        
        # Create governance vote for pricing adjustment
        vote_id = await constitutional_engine.create_governance_vote(
            vote_type="pricing_adjustment",
            proposed_adjustment=Decimal("-5.0"),  # 5% decrease
            rationale="Community voted for lower pricing",
            proposer_id="community_dao",
            target_tier="silver"
        )
        
        assert vote_id in constitutional_engine.governance_votes
        
        # Cast vote
        result = await constitutional_engine.cast_governance_vote(
            vote_id, "voter_123", "for", Decimal("100.0")
        )
        
        assert result is True
        
        vote = constitutional_engine.governance_votes[vote_id]
        assert vote.votes_for == Decimal("100.0")


class TestPerformanceAndScalability:
    """Test performance and scalability of pricing system"""
    
    async def test_bulk_h200_calculations(self, pricing_manager):
        """Test bulk H200-hour calculations performance"""
        
        # Generate 100 H200-hour calculations
        tasks = []
        for i in range(100):
            task = pricing_manager.calculate_h200_hour_equivalent(
                Decimal(str(100 + i)),  # Varying TOPS
                Decimal("0.8"),
                Decimal("1"),
                f"device_{i}"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 100
        assert all("h200_hours_equivalent" in result for result in results)
        
        # Check audit trail can handle bulk operations
        audit_manager = pricing_manager.audit_manager
        assert len(audit_manager.audit_records) >= 100
    
    async def test_concurrent_pricing_requests(self, pricing_manager):
        """Test concurrent pricing request handling"""
        
        # Create concurrent pricing requests
        tasks = []
        for tier in ["bronze", "silver", "gold", "platinum"]:
            for i in range(10):
                task = pricing_manager.get_constitutional_h200_price(
                    UserSizeTier(tier),
                    Decimal("500"),
                    Decimal("0.8"),
                    Decimal("1"),
                    "enhanced"
                )
                tasks.append(task)
        
        quotes = await asyncio.gather(*tasks)
        
        assert len(quotes) == 40  # 4 tiers × 10 requests each
        assert all("quote_id" in quote for quote in quotes)
        assert len(set(quote["quote_id"] for quote in quotes)) == 40  # All unique


class TestErrorHandlingAndValidation:
    """Test error handling and input validation"""
    
    async def test_invalid_tier_handling(self, pricing_manager):
        """Test handling of invalid user tiers"""
        
        with pytest.raises(ValueError, match="Unknown constitutional tier"):
            await pricing_manager.get_constitutional_h200_price(
                "invalid_tier",  # This will fail at UserSizeTier construction
                Decimal("500"),
                Decimal("0.8"),
                Decimal("1")
            )
    
    async def test_negative_values_handling(self, pricing_manager):
        """Test handling of negative input values"""
        
        # Negative utilization should be handled gracefully
        result = await pricing_manager.calculate_h200_hour_equivalent(
            Decimal("500"),
            Decimal("-0.1"),  # Negative utilization
            Decimal("1")
        )
        
        # Should result in negative H200-hours (which indicates error)
        assert result["h200_hours_equivalent"] < 0
    
    async def test_zero_values_handling(self, pricing_manager):
        """Test handling of zero input values"""
        
        # Zero TOPS should result in zero H200-hours
        result = await pricing_manager.calculate_h200_hour_equivalent(
            Decimal("0"),
            Decimal("1.0"),
            Decimal("1")
        )
        
        assert result["h200_hours_equivalent"] == 0.0
    
    async def test_extreme_values_handling(self, pricing_manager):
        """Test handling of extreme input values"""
        
        # Very high TOPS value
        result = await pricing_manager.calculate_h200_hour_equivalent(
            Decimal("1000000"),  # 1M TOPS
            Decimal("1.0"),
            Decimal("1")
        )
        
        # Should handle gracefully
        assert result["h200_hours_equivalent"] > 1000
        assert "timestamp" in result


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Test complete end-to-end pricing workflows"""
    
    async def test_complete_constitutional_pricing_workflow(self, pricing_manager):
        """Test complete constitutional pricing workflow from calculation to audit"""
        
        # Step 1: Calculate H200-hour equivalent
        h200_calc = await pricing_manager.calculate_h200_hour_equivalent(
            Decimal("750"),  # 750 TOPS device
            Decimal("0.9"),  # 90% utilization
            Decimal("4"),    # 4 hours
            "production_gpu_001"
        )
        
        # Step 2: Get constitutional pricing quote
        quote = await pricing_manager.get_constitutional_h200_price(
            UserSizeTier.GOLD,
            Decimal("750"),
            Decimal("0.9"),
            Decimal("4"),
            "constitutional",
            True  # TEE enabled
        )
        
        # Step 3: Get TEE enhanced pricing for comparison
        tee_quote = await pricing_manager.get_tee_enhanced_pricing(
            ResourceLane.GPU,
            Decimal("1"),
            Decimal("4"),
            "confidential"
        )
        
        # Step 4: Verify audit trail completeness
        audit_manager = pricing_manager.audit_manager
        integrity_check = audit_manager.verify_audit_chain_integrity()
        
        # Step 5: Generate compliance report
        compliance_report = audit_manager.generate_constitutional_compliance_report()
        
        # Assertions
        assert h200_calc["h200_hours_equivalent"] > 0
        assert quote["user_tier"] == "gold"
        assert quote["constitutional_level"] == "constitutional"
        assert tee_quote["tee_level"] == "confidential"
        assert integrity_check["chain_valid"] is True
        assert compliance_report.overall_compliance_score >= Decimal("0.8")
    
    async def test_governance_driven_pricing_adjustment(self, constitutional_engine):
        """Test complete governance-driven pricing adjustment workflow"""
        
        # Step 1: Create pricing adjustment proposal
        vote_id = await constitutional_engine.create_governance_vote(
            vote_type="pricing_adjustment",
            proposed_adjustment=Decimal("-10.0"),  # 10% reduction
            rationale="Community cost reduction initiative",
            proposer_id="community_treasury",
            target_tier="gold",
            voting_duration_hours=24
        )
        
        # Step 2: Cast multiple votes to reach quorum
        await constitutional_engine.cast_governance_vote(vote_id, "voter_1", "for", Decimal("60.0"))
        await constitutional_engine.cast_governance_vote(vote_id, "voter_2", "for", Decimal("40.0"))
        await constitutional_engine.cast_governance_vote(vote_id, "voter_3", "against", Decimal("20.0"))
        
        # Step 3: Verify vote implementation
        vote = constitutional_engine.governance_votes[vote_id]
        assert vote.status in ["passed", "implemented"]
        
        # Step 4: Check if pricing adjustment was applied
        if vote.status == "implemented":
            # Pricing should be adjusted in constitutional_tiers
            gold_tier = constitutional_engine.constitutional_tiers["gold"]
            # Verify base rate was adjusted (this is approximate due to timing)
            assert gold_tier.h200_hour_base_rate != Decimal("1.00")  # Should be changed from original


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])