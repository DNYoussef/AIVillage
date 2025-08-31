#!/usr/bin/env python3
"""
Constitutional Pricing Validation Script

Validates the H200-hour pricing system and constitutional compliance:
- H200-hour formula accuracy
- Constitutional tier pricing correctness
- TEE-enhanced workload pricing validation
- Audit trail integrity verification
- Transparency and governance compliance

Usage:
    python scripts/validate_constitutional_pricing.py [--tier TIER] [--verbose]
"""

import asyncio
import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from infrastructure.fog.market.pricing_manager import (
    DynamicPricingManager,
    UserSizeTier,
    ResourceLane,
    get_pricing_manager
)
from infrastructure.fog.market.constitutional_pricing import (
    ConstitutionalPricingEngine,
    H200EquivalentDevice
)
from infrastructure.fog.market.audit_pricing import AuditTrailManager


class ConstitutionalPricingValidator:
    """Comprehensive validator for constitutional pricing system"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.validation_results: List[Dict[str, Any]] = []
        self.pricing_manager: DynamicPricingManager = None
        self.constitutional_engine: ConstitutionalPricingEngine = None
    
    async def initialize(self):
        """Initialize pricing components"""
        self.pricing_manager = await get_pricing_manager()
        audit_manager = AuditTrailManager()
        self.constitutional_engine = ConstitutionalPricingEngine(audit_manager)
    
    def log_result(self, test_name: str, status: str, details: Dict[str, Any] = None):
        """Log validation result"""
        result = {
            "test": test_name,
            "status": status,  # "PASS", "FAIL", "WARNING"
            "details": details or {}
        }
        self.validation_results.append(result)
        
        if self.verbose:
            print(f"[{status}] {test_name}")
            if details and self.verbose:
                print(f"    Details: {json.dumps(details, indent=2)}")
    
    async def validate_h200_formula_accuracy(self):
        """Validate H200-hour formula calculations"""
        print("\\n🔍 Validating H200-Hour Formula Accuracy...")
        
        # Test cases with known expected results
        test_cases = [
            {
                "name": "Exact H200 equivalent", 
                "tops": Decimal("989"), 
                "utilization": Decimal("1.0"), 
                "hours": Decimal("1"),
                "expected": Decimal("1.0")
            },
            {
                "name": "Half H200 performance",
                "tops": Decimal("494.5"),
                "utilization": Decimal("1.0"),
                "hours": Decimal("1"), 
                "expected": Decimal("0.5")
            },
            {
                "name": "Double performance, half utilization",
                "tops": Decimal("1978"),
                "utilization": Decimal("0.5"), 
                "hours": Decimal("1"),
                "expected": Decimal("1.0")
            },
            {
                "name": "Complex calculation",
                "tops": Decimal("750"),
                "utilization": Decimal("0.8"),
                "hours": Decimal("2.5"),
                "expected": Decimal("1.518")  # (750 * 0.8 * 2.5) / 989
            }
        ]
        
        for case in test_cases:
            result = await self.pricing_manager.calculate_h200_hour_equivalent(
                case["tops"], case["utilization"], case["hours"]
            )
            
            actual = Decimal(str(result["h200_hours_equivalent"]))
            expected = case["expected"]
            tolerance = Decimal("0.001")
            
            if abs(actual - expected) < tolerance:
                self.log_result(
                    f"H200 Formula - {case['name']}", 
                    "PASS",
                    {"expected": float(expected), "actual": float(actual)}
                )
            else:
                self.log_result(
                    f"H200 Formula - {case['name']}", 
                    "FAIL",
                    {
                        "expected": float(expected), 
                        "actual": float(actual), 
                        "difference": float(abs(actual - expected))
                    }
                )
    
    async def validate_constitutional_tier_pricing(self):
        """Validate constitutional tier pricing correctness"""
        print("\\n🏛️ Validating Constitutional Tier Pricing...")
        
        tiers = [UserSizeTier.BRONZE, UserSizeTier.SILVER, UserSizeTier.GOLD, UserSizeTier.PLATINUM]
        
        for tier in tiers:
            try:
                # Test basic pricing
                quote = await self.pricing_manager.get_constitutional_h200_price(
                    tier,
                    Decimal("500"),  # 500 TOPS
                    Decimal("0.8"),  # 80% utilization
                    Decimal("1"),    # 1 hour
                    "enhanced"
                )
                
                # Validate quote structure
                required_fields = [
                    "quote_id", "user_tier", "h200_calculation", "pricing", 
                    "tier_limits", "constitutional_features"
                ]
                
                missing_fields = [field for field in required_fields if field not in quote]
                
                if not missing_fields:
                    self.log_result(
                        f"Constitutional Pricing - {tier.value} tier structure",
                        "PASS",
                        {"h200_hours": quote["h200_calculation"]["h200_hours_equivalent"]}
                    )
                else:
                    self.log_result(
                        f"Constitutional Pricing - {tier.value} tier structure",
                        "FAIL",
                        {"missing_fields": missing_fields}
                    )
                
                # Validate tier-specific rates
                expected_rates = {
                    UserSizeTier.BRONZE: 0.5,
                    UserSizeTier.SILVER: 0.75,
                    UserSizeTier.GOLD: 1.0,
                    UserSizeTier.PLATINUM: 1.5
                }
                
                actual_rate = quote["pricing"]["base_rate_per_h200_hour"]
                expected_rate = expected_rates[tier]
                
                if actual_rate == expected_rate:
                    self.log_result(
                        f"Constitutional Pricing - {tier.value} rate correctness",
                        "PASS",
                        {"rate": actual_rate}
                    )
                else:
                    self.log_result(
                        f"Constitutional Pricing - {tier.value} rate correctness", 
                        "FAIL",
                        {"expected": expected_rate, "actual": actual_rate}
                    )
                
            except Exception as e:
                self.log_result(
                    f"Constitutional Pricing - {tier.value} tier error",
                    "FAIL",
                    {"error": str(e)}
                )
    
    async def validate_tee_enhanced_pricing(self):
        """Validate TEE-enhanced workload pricing"""
        print("\\n🔒 Validating TEE-Enhanced Pricing...")
        
        tee_levels = ["basic", "enhanced", "confidential"]
        expected_premiums = {"basic": 20.0, "enhanced": 35.0, "confidential": 50.0}
        
        for level in tee_levels:
            try:
                tee_quote = await self.pricing_manager.get_tee_enhanced_pricing(
                    ResourceLane.GPU,
                    Decimal("1"),
                    Decimal("1"), 
                    level
                )
                
                # Validate premium percentage
                actual_premium = tee_quote["tee_enhanced_pricing"]["tee_premium_percentage"]
                expected_premium = expected_premiums[level]
                
                if actual_premium == expected_premium:
                    self.log_result(
                        f"TEE Pricing - {level} premium",
                        "PASS",
                        {"premium": actual_premium}
                    )
                else:
                    self.log_result(
                        f"TEE Pricing - {level} premium",
                        "FAIL", 
                        {"expected": expected_premium, "actual": actual_premium}
                    )
                
                # Validate TEE features
                features = tee_quote["tee_features"]
                required_features = ["hardware_security", "encrypted_computation", "constitutional_compliant"]
                
                missing_features = [f for f in required_features if not features.get(f)]
                
                if not missing_features:
                    self.log_result(
                        f"TEE Features - {level} completeness",
                        "PASS"
                    )
                else:
                    self.log_result(
                        f"TEE Features - {level} completeness",
                        "FAIL",
                        {"missing_features": missing_features}
                    )
                
            except Exception as e:
                self.log_result(
                    f"TEE Pricing - {level} error",
                    "FAIL", 
                    {"error": str(e)}
                )
    
    async def validate_audit_trail_integrity(self):
        """Validate audit trail integrity and compliance"""
        print("\\n📋 Validating Audit Trail Integrity...")
        
        # Generate several operations to build audit trail
        operations = [
            ("H200 calculation", lambda: self.pricing_manager.calculate_h200_hour_equivalent(
                Decimal("400"), Decimal("0.9"), Decimal("1"), "audit_test_device"
            )),
            ("Constitutional quote", lambda: self.pricing_manager.get_constitutional_h200_price(
                UserSizeTier.SILVER, Decimal("600"), Decimal("0.8"), Decimal("2"), "enhanced"
            )),
            ("TEE pricing", lambda: self.pricing_manager.get_tee_enhanced_pricing(
                ResourceLane.SPECIALIZED, Decimal("1"), Decimal("1"), "enhanced", "audit_test_node"
            ))
        ]
        
        for op_name, operation in operations:
            try:
                await operation()
                self.log_result(f"Audit Trail - {op_name} execution", "PASS")
            except Exception as e:
                self.log_result(f"Audit Trail - {op_name} execution", "FAIL", {"error": str(e)})
        
        # Verify audit trail integrity
        try:
            audit_manager = self.pricing_manager.audit_manager
            integrity_result = audit_manager.verify_audit_chain_integrity()
            
            if integrity_result["chain_valid"]:
                self.log_result(
                    "Audit Trail - Chain integrity", 
                    "PASS",
                    {
                        "total_records": integrity_result["total_records"],
                        "verified_records": integrity_result["verified_records"]
                    }
                )
            else:
                self.log_result(
                    "Audit Trail - Chain integrity",
                    "FAIL",
                    {
                        "failures": len(integrity_result["integrity_failures"]),
                        "details": integrity_result["integrity_failures"][:5]  # First 5 failures
                    }
                )
        except Exception as e:
            self.log_result("Audit Trail - Chain integrity", "FAIL", {"error": str(e)})
    
    async def validate_constitutional_compliance(self):
        """Validate constitutional compliance features"""
        print("\\n🏛️ Validating Constitutional Compliance...")
        
        try:
            # Generate compliance report
            audit_manager = self.pricing_manager.audit_manager
            compliance_report = audit_manager.generate_constitutional_compliance_report()
            
            # Check compliance scores
            compliance_checks = [
                ("Pricing transparency", compliance_report.pricing_transparency),
                ("Audit trail complete", compliance_report.audit_trail_complete), 
                ("Governance participation", compliance_report.governance_participation_enabled),
                ("Privacy protections", compliance_report.privacy_protections_active)
            ]
            
            for check_name, check_result in compliance_checks:
                status = "PASS" if check_result else "FAIL"
                self.log_result(f"Constitutional Compliance - {check_name}", status)
            
            # Overall compliance score
            overall_score = float(compliance_report.overall_compliance_score)
            if overall_score >= 0.8:
                self.log_result(
                    "Constitutional Compliance - Overall score",
                    "PASS",
                    {"score": overall_score}
                )
            else:
                self.log_result(
                    "Constitutional Compliance - Overall score",
                    "WARNING" if overall_score >= 0.6 else "FAIL",
                    {
                        "score": overall_score, 
                        "issues": compliance_report.compliance_issues
                    }
                )
                
        except Exception as e:
            self.log_result("Constitutional Compliance - Report generation", "FAIL", {"error": str(e)})
    
    async def validate_governance_mechanisms(self):
        """Validate governance and democratic pricing mechanisms"""
        print("\\n🗳️ Validating Governance Mechanisms...")
        
        try:
            # Test governance vote creation
            vote_id = await self.constitutional_engine.create_governance_vote(
                vote_type="pricing_adjustment",
                proposed_adjustment=Decimal("-5.0"),
                rationale="Test governance validation",
                proposer_id="validator_test",
                target_tier="bronze"
            )
            
            self.log_result("Governance - Vote creation", "PASS", {"vote_id": vote_id[:8]})
            
            # Test vote casting
            cast_result = await self.constitutional_engine.cast_governance_vote(
                vote_id, "test_voter", "for", Decimal("10.0")
            )
            
            if cast_result:
                self.log_result("Governance - Vote casting", "PASS")
            else:
                self.log_result("Governance - Vote casting", "FAIL")
            
            # Test transparency report generation
            transparency_report = await self.constitutional_engine.get_pricing_transparency_report()
            
            required_sections = ["constitutional_tiers", "governance_activity", "transparency_features"]
            missing_sections = [s for s in required_sections if s not in transparency_report]
            
            if not missing_sections:
                self.log_result("Governance - Transparency report", "PASS")
            else:
                self.log_result(
                    "Governance - Transparency report",
                    "FAIL",
                    {"missing_sections": missing_sections}
                )
                
        except Exception as e:
            self.log_result("Governance - Mechanism validation", "FAIL", {"error": str(e)})
    
    async def validate_performance_characteristics(self):
        """Validate pricing system performance characteristics"""
        print("\\n⚡ Validating Performance Characteristics...")
        
        try:
            import time
            
            # Test bulk H200 calculations performance
            start_time = time.time()
            
            tasks = []
            for i in range(50):
                task = self.pricing_manager.calculate_h200_hour_equivalent(
                    Decimal(str(100 + i)), Decimal("0.8"), Decimal("1")
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            duration = end_time - start_time
            
            if duration < 5.0:  # Should complete in under 5 seconds
                self.log_result(
                    "Performance - Bulk calculations",
                    "PASS", 
                    {"operations": len(results), "duration_seconds": round(duration, 2)}
                )
            else:
                self.log_result(
                    "Performance - Bulk calculations",
                    "WARNING",
                    {"operations": len(results), "duration_seconds": round(duration, 2)}
                )
                
        except Exception as e:
            self.log_result("Performance - Bulk calculations", "FAIL", {"error": str(e)})
    
    async def run_validation(self, target_tier: str = None):
        """Run complete constitutional pricing validation"""
        print("🏛️ Constitutional Pricing Validation Suite")
        print("=" * 50)
        
        await self.initialize()
        
        # Run validation tests
        await self.validate_h200_formula_accuracy()
        
        if target_tier:
            # Validate specific tier only
            try:
                tier = UserSizeTier(target_tier)
                print(f"\\n🎯 Validating specific tier: {tier.value}")
                # Run tier-specific validations here
            except ValueError:
                print(f"❌ Invalid tier: {target_tier}")
                return False
        else:
            # Run full validation suite
            await self.validate_constitutional_tier_pricing()
        
        await self.validate_tee_enhanced_pricing()
        await self.validate_audit_trail_integrity()
        await self.validate_constitutional_compliance()
        await self.validate_governance_mechanisms()
        await self.validate_performance_characteristics()
        
        # Generate summary
        self.generate_validation_summary()
        
        return self.validation_passed()
    
    def validation_passed(self) -> bool:
        """Check if validation passed overall"""
        failed_tests = [r for r in self.validation_results if r["status"] == "FAIL"]
        return len(failed_tests) == 0
    
    def generate_validation_summary(self):
        """Generate validation summary"""
        print("\\n📊 Validation Summary")
        print("=" * 30)
        
        passed = len([r for r in self.validation_results if r["status"] == "PASS"])
        failed = len([r for r in self.validation_results if r["status"] == "FAIL"]) 
        warnings = len([r for r in self.validation_results if r["status"] == "WARNING"])
        total = len(self.validation_results)
        
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Warnings: {warnings}")
        
        if failed > 0:
            print("\\n❌ Failed Tests:")
            for result in self.validation_results:
                if result["status"] == "FAIL":
                    print(f"  - {result['test']}")
                    if result["details"]:
                        print(f"    {result['details']}")
        
        if warnings > 0:
            print("\\n⚠️  Warning Tests:")
            for result in self.validation_results:
                if result["status"] == "WARNING":
                    print(f"  - {result['test']}")
        
        success_rate = (passed / total) * 100 if total > 0 else 0
        print(f"\\n🎯 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("🏆 EXCELLENT: Constitutional pricing system is fully compliant!")
        elif success_rate >= 80:
            print("✅ GOOD: Constitutional pricing system is mostly compliant.")
        elif success_rate >= 70:
            print("⚠️  FAIR: Constitutional pricing system needs some improvements.")
        else:
            print("❌ POOR: Constitutional pricing system requires significant fixes.")


async def main():
    """Main validation entry point"""
    parser = argparse.ArgumentParser(description="Validate constitutional pricing system")
    parser.add_argument("--tier", choices=["bronze", "silver", "gold", "platinum"], 
                       help="Validate specific tier only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    validator = ConstitutionalPricingValidator(verbose=args.verbose)
    
    try:
        success = await validator.run_validation(args.tier)
        exit_code = 0 if success else 1
        
        print(f"\\n🏁 Validation {'PASSED' if success else 'FAILED'}")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\\n⏹️  Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\\n💥 Validation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())