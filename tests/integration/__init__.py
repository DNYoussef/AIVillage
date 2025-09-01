"""
Integration Tests Package

This package contains comprehensive integration tests for the AI Village federated system.
These tests validate that all components work together as a cohesive system.

Test Categories:
- Complete Federated Pipeline: End-to-end inference workflow validation
- P2P Fog Integration: P2P networking with fog infrastructure integration
- Secure Training Workflow: Security-enhanced federated training pipeline
- Mobile Participation: Mobile device integration and optimization
- P2P Network Validation: Validation of proven Phase 1 P2P components
- Security Integration: Validation of security specialist enhancements
- Enhanced Fog Infrastructure: Fog specialist improvements validation
- Federated Training End-to-End: Ultimate system validation test

Success Criteria:
✅ All proven Phase 1 components still working
✅ New security layer functions correctly  
✅ Enhanced fog infrastructure operational
✅ Complete federated training pipeline functional
✅ End-to-end scenarios pass with real data
"""

__version__ = "1.0.0"
__author__ = "AI Village Integration Testing Team"

# Test execution priorities
TEST_EXECUTION_ORDER = [
    "test_p2p_network_validation",  # Validate Phase 1 foundation
    "test_security_integration_validation",  # Validate security enhancements
    "test_enhanced_fog_infrastructure",  # Validate fog improvements
    "test_p2p_fog_integration",  # Test P2P-fog integration
    "test_mobile_participation",  # Test mobile integration
    "test_secure_training_workflow",  # Test secure training
    "test_complete_federated_pipeline",  # Test complete inference pipeline
    "test_federated_training_end_to_end",  # Ultimate end-to-end validation
]

# Integration test success metrics
SUCCESS_METRICS = {
    "component_integration_rate": 0.95,  # 95% of components integrate successfully
    "performance_degradation_limit": 0.15,  # Max 15% performance impact from integration
    "security_breach_tolerance": 0,  # Zero security breaches allowed
    "fault_recovery_rate": 0.90,  # 90% of faults should be recoverable
    "mobile_participation_rate": 0.80,  # 80% mobile device participation success
    "end_to_end_success_rate": 0.95,  # 95% end-to-end scenario success rate
}

# Test configuration
INTEGRATION_TEST_CONFIG = {
    "parallel_execution": True,
    "timeout_seconds": 300,
    "retry_attempts": 2,
    "verbose_logging": True,
    "performance_monitoring": True,
    "security_validation": True,
    "mock_external_services": True,
}
