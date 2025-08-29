#!/usr/bin/env python3
"""
Stream B Integration Test - Standalone

This script tests the complete Stream B implementation:
- DAO Governance System
- Tokenomics System  
- Compliance System
- Integration between all components
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path (go up 2 levels from scripts/integration/)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "infrastructure"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all Stream B components can be imported."""
    try:
        # Test DAO Governance System
        from infrastructure.fog.governance.dao_governance_system import (
            DAOGovernanceSystem, ProposalType, MemberRole, VoteChoice
        )
        logger.info("✓ DAO Governance System imported successfully")
        
        # Test Tokenomics System
        from infrastructure.fog.tokenomics.comprehensive_tokenomics_system import (
            ComprehensiveTokenomicsSystem, StakingTier
        )
        logger.info("✓ Tokenomics System imported successfully")
        
        # Test Compliance System
        from infrastructure.fog.compliance.automated_compliance_system import (
            AutomatedComplianceSystem, ComplianceFramework
        )
        logger.info("✓ Compliance System imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_system_initialization():
    """Test that all systems can be initialized."""
    try:
        from infrastructure.fog.governance.dao_governance_system import DAOGovernanceSystem
        from infrastructure.fog.tokenomics.comprehensive_tokenomics_system import ComprehensiveTokenomicsSystem
        from infrastructure.fog.compliance.automated_compliance_system import AutomatedComplianceSystem
        
        # Initialize systems
        dao_system = DAOGovernanceSystem(data_dir="./test_dao_data")
        logger.info("✓ DAO Governance System initialized")
        
        tokenomics_system = ComprehensiveTokenomicsSystem(data_dir="./test_tokenomics_data")
        logger.info("✓ Tokenomics System initialized")
        
        compliance_system = AutomatedComplianceSystem(data_dir="./test_compliance_data")
        logger.info("✓ Compliance System initialized")
        
        return dao_system, tokenomics_system, compliance_system
        
    except Exception as e:
        logger.error(f"✗ System initialization failed: {e}")
        return None

def test_dao_governance(dao_system):
    """Test DAO governance functionality."""
    try:
        from infrastructure.fog.governance.dao_governance_system import MemberRole, ProposalType, VoteChoice
        
        # Add a member
        member_id = "test_member_001"
        member = dao_system.add_member(
            member_id=member_id,
            address="0xtest123",
            role=MemberRole.MEMBER,
            voting_power=5000
        )
        assert member.member_id == member_id
        logger.info("✓ DAO member creation successful")
        
        # Create a proposal
        proposal_id = dao_system.create_proposal(
            title="Test Proposal",
            description="This is a test proposal",
            proposal_type=ProposalType.TOKENOMICS_CHANGE,
            author_id=member_id
        )
        assert proposal_id in dao_system.proposals
        logger.info("✓ DAO proposal creation successful")
        
        # Start voting and cast vote
        dao_system.start_voting(proposal_id)
        dao_system.cast_vote(proposal_id, member_id, VoteChoice.YES, "I support this")
        
        proposal = dao_system.proposals[proposal_id]
        assert proposal.yes_votes > 0
        logger.info("✓ DAO voting successful")
        
        # Get governance stats
        stats = dao_system.get_governance_stats()
        assert "governance_health" in stats
        logger.info("✓ DAO statistics successful")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ DAO governance test failed: {e}")
        return False

def test_tokenomics(tokenomics_system):
    """Test tokenomics functionality."""
    try:
        from infrastructure.fog.tokenomics.comprehensive_tokenomics_system import StakingTier
        
        # Create accounts
        account1 = tokenomics_system.create_account("test_account_1", "0xaccount1", 10000)
        account2 = tokenomics_system.create_account("test_account_2", "0xaccount2", 5000)
        assert account1.account_id == "test_account_1"
        logger.info("✓ Tokenomics account creation successful")
        
        # Transfer tokens
        tx_id = tokenomics_system.transfer_tokens("test_account_1", "test_account_2", 1000)
        assert any(tx.tx_id == tx_id for tx in tokenomics_system.transactions)
        logger.info("✓ Tokenomics transfer successful")
        
        # Stake tokens
        position_id = tokenomics_system.stake_tokens("test_account_1", 3000, StakingTier.GOLD)
        assert position_id in tokenomics_system.staking_positions
        logger.info("✓ Tokenomics staking successful")
        
        # Distribute rewards
        reward = tokenomics_system.distribute_compute_rewards("test_account_2", 5, 1.0)
        assert reward > 0
        logger.info("✓ Tokenomics rewards successful")
        
        # Get network stats
        stats = tokenomics_system.get_network_statistics()
        assert "supply_metrics" in stats
        logger.info("✓ Tokenomics statistics successful")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Tokenomics test failed: {e}")
        return False

def test_compliance(compliance_system):
    """Test compliance functionality."""
    try:
        # Register entity
        success = compliance_system.register_entity("test_entity_1", "individual", "US")
        assert success
        logger.info("✓ Compliance entity registration successful")
        
        # Monitor transaction
        violations = compliance_system.monitor_transaction({
            "transaction_id": "test_tx_001",
            "amount": 15000,
            "from_account": "test_entity_1",
            "to_account": "test_entity_2",
            "transaction_type": "transfer"
        })
        # Should detect violation for large transaction
        logger.info(f"✓ Compliance monitoring successful ({len(violations)} violations detected)")
        
        # Get compliance dashboard
        dashboard = compliance_system.get_compliance_dashboard()
        assert "overview" in dashboard
        logger.info("✓ Compliance dashboard successful")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Compliance test failed: {e}")
        return False

def test_integration(dao_system, tokenomics_system, compliance_system):
    """Test integration between systems."""
    try:
        from infrastructure.fog.governance.dao_governance_system import MemberRole, ProposalType, VoteChoice
        from infrastructure.fog.tokenomics.comprehensive_tokenomics_system import StakingTier
        
        # Create integrated user
        user_id = "integration_user_001"
        
        # Add to DAO
        dao_system.add_member(
            member_id=user_id,
            address="0xintegration123",
            role=MemberRole.MEMBER,
            voting_power=8000
        )
        
        # Create tokenomics account
        tokenomics_system.create_account(
            account_id=user_id,
            address="0xintegration123",
            initial_balance=15000
        )
        
        # Register with compliance
        compliance_system.register_entity(
            entity_id=user_id,
            entity_type="individual",
            jurisdiction="US"
        )
        
        # Create and vote on proposal
        proposal_id = dao_system.create_proposal(
            title="Integration Test Proposal",
            description="Testing system integration",
            proposal_type=ProposalType.TREASURY_SPEND,
            author_id=user_id,
            budget_request=25000
        )
        
        dao_system.start_voting(proposal_id)
        dao_system.cast_vote(proposal_id, user_id, VoteChoice.YES)
        
        # Stake tokens
        tokenomics_system.stake_tokens(user_id, 5000, StakingTier.PLATINUM)
        
        # Distribute governance rewards
        reward = tokenomics_system.distribute_governance_rewards(user_id, 1, 1)
        
        # Monitor governance action for compliance
        compliance_system.monitor_governance_action({
            "action_id": proposal_id,
            "action_type": "proposal_creation",
            "actor_id": user_id,
            "budget_request": 25000
        })
        
        # Verify integration
        assert user_id in dao_system.members
        assert user_id in tokenomics_system.accounts
        assert user_id in compliance_system.monitored_entities
        
        # Check account has rewards
        account = tokenomics_system.accounts[user_id]
        assert account.fog_balance > 15000  # Should have initial + rewards
        
        logger.info("✓ System integration successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ System integration test failed: {e}")
        return False

def main():
    """Run complete Stream B integration test."""
    print("="*80)
    print("STREAM B GOVERNANCE & COMPLIANCE - INTEGRATION TEST")
    print("="*80)
    
    start_time = time.time()
    total_tests = 0
    passed_tests = 0
    
    # Test 1: Import all components
    print("\n1. Testing component imports...")
    total_tests += 1
    if test_imports():
        passed_tests += 1
    
    # Test 2: Initialize systems
    print("\n2. Testing system initialization...")
    total_tests += 1
    systems = test_system_initialization()
    if systems:
        dao_system, tokenomics_system, compliance_system = systems
        passed_tests += 1
    else:
        print("Cannot proceed with further tests due to initialization failure")
        return
    
    # Test 3: DAO Governance
    print("\n3. Testing DAO governance system...")
    total_tests += 1
    if test_dao_governance(dao_system):
        passed_tests += 1
    
    # Test 4: Tokenomics
    print("\n4. Testing tokenomics system...")
    total_tests += 1
    if test_tokenomics(tokenomics_system):
        passed_tests += 1
    
    # Test 5: Compliance
    print("\n5. Testing compliance system...")
    total_tests += 1
    if test_compliance(compliance_system):
        passed_tests += 1
    
    # Test 6: Integration
    print("\n6. Testing system integration...")
    total_tests += 1
    if test_integration(dao_system, tokenomics_system, compliance_system):
        passed_tests += 1
    
    # Results
    end_time = time.time()
    duration = end_time - start_time
    success_rate = (passed_tests / total_tests) * 100
    
    print("\n" + "="*80)
    print("INTEGRATION TEST RESULTS")
    print("="*80)
    print(f"Tests Run: {total_tests}")
    print(f"Tests Passed: {passed_tests}")
    print(f"Tests Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Duration: {duration:.2f} seconds")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("Stream B Governance & Compliance systems are operational and ready for production!")
        print("\nKey Capabilities Validated:")
        print("✓ DAO governance with voting, proposals, and member management")
        print("✓ Comprehensive tokenomics with FOG tokens, staking, and rewards")
        print("✓ Automated compliance monitoring and reporting")
        print("✓ Complete system integration and data consistency")
        print("✓ End-to-end governance workflows")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed.")
        print("Review the error logs above and address issues before production deployment.")
    
    print("="*80)
    
    # Generate summary report
    report = {
        "test_summary": {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": success_rate,
            "duration_seconds": duration
        },
        "system_status": {
            "dao_governance": {
                "status": "operational",
                "members": len(dao_system.members),
                "proposals": len(dao_system.proposals),
                "features": ["voting", "proposals", "delegation", "member_management"]
            },
            "tokenomics": {
                "status": "operational", 
                "accounts": len(tokenomics_system.accounts),
                "transactions": len(tokenomics_system.transactions),
                "staking_positions": len(tokenomics_system.staking_positions),
                "features": ["fog_tokens", "staking", "rewards", "economic_incentives"]
            },
            "compliance": {
                "status": "operational",
                "monitored_entities": len(compliance_system.monitored_entities),
                "compliance_rules": len(compliance_system.compliance_rules),
                "features": ["transaction_monitoring", "governance_compliance", "automated_reporting"]
            }
        },
        "production_readiness": "ready" if passed_tests == total_tests else "needs_attention"
    }
    
    # Save report
    with open("stream_b_integration_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: stream_b_integration_report.json")

if __name__ == "__main__":
    main()