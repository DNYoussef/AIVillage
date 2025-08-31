"""
Unit tests for SecurityIntegrationManager using TDD London School methodology.

Following mock-driven development patterns to test behavior and interactions
between security components without depending on implementation details.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call
from dataclasses import dataclass
from typing import Dict, Any, Optional, Set, List

# Import the module under test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from infrastructure.security.security_integration_manager import (
    SecurityIntegrationManager,
    SecurityConfiguration,
    SecurityEvent,
    SecurityMetrics
)
from infrastructure.security.federated_auth_system import (
    NodeRole,
    AuthenticationMethod,
    AuthenticationStatus
)


class TestSecurityIntegrationManager:
    """Test suite for SecurityIntegrationManager using London School TDD approach."""
    
    @pytest.fixture
    def mock_auth_system(self):
        """Mock federated authentication system."""
        mock = AsyncMock()
        mock.initialize = AsyncMock(return_value=True)
        mock.health_check = AsyncMock(return_value={"healthy": True})
        mock.get_auth_stats = Mock(return_value={"active_sessions": 5})
        mock.register_node = AsyncMock()
        mock.authenticate_node = AsyncMock()
        mock.validate_session = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_transport_security(self):
        """Mock transport security manager."""
        mock = AsyncMock()
        mock.initialize = AsyncMock(return_value=True)
        mock.health_check = AsyncMock(return_value={"healthy": True})
        mock.establish_secure_channel = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_consensus_security(self):
        """Mock consensus security manager."""
        mock = Mock()
        mock.threshold_keys = {}
        mock.initialize_distributed_keys = AsyncMock(return_value=True)
        mock.detect_byzantine_behavior = AsyncMock(return_value=[])
        mock.mitigate_attacks = AsyncMock(return_value=True)
        mock.health_check = AsyncMock(return_value={"healthy": True})
        return mock
    
    @pytest.fixture
    def mock_threat_detection(self):
        """Mock threat detection system."""
        mock = AsyncMock()
        mock.ingest_security_event = AsyncMock()
        mock.detect_threats = AsyncMock(return_value=[])
        mock.mitigate_threat = AsyncMock(return_value=(True, ["action1", "action2"]))
        mock.health_check = AsyncMock(return_value={"healthy": True})
        mock.get_detection_stats = Mock(return_value={"threats_detected": 10})
        return mock
    
    @pytest.fixture
    def mock_trust_system(self):
        """Mock reputation trust system."""
        mock = AsyncMock()
        mock.initialize_node_trust = AsyncMock()
        mock.get_trust_decision = AsyncMock(return_value=(True, 0.8, "high_trust"))
        mock.record_interaction = AsyncMock()
        mock.report_byzantine_behavior = AsyncMock()
        mock.health_check = AsyncMock(return_value={"healthy": True})
        mock.get_trust_statistics = Mock(return_value={"total_nodes": 5})
        mock.get_node_trust_details = Mock(return_value={"trust_score": 0.8})
        mock.get_trusted_nodes = Mock(return_value=[])
        return mock
    
    @pytest.fixture
    def mock_secure_aggregation(self):
        """Mock secure aggregation protocol."""
        mock = AsyncMock()
        mock.setup_participant = AsyncMock()
        mock.create_secure_gradient = AsyncMock()
        mock.secure_aggregate = AsyncMock(return_value=(True, {"result": "aggregated"}, {"participants": 3}))
        mock.get_aggregation_stats = Mock(return_value={"aggregations": 15})
        return mock
    
    @pytest.fixture
    def security_manager(self, mock_auth_system, mock_transport_security, 
                        mock_consensus_security, mock_threat_detection,
                        mock_trust_system, mock_secure_aggregation):
        """Create SecurityIntegrationManager with mocked dependencies."""
        manager = SecurityIntegrationManager("test_node_001")
        
        # Inject mock dependencies
        manager.auth_system = mock_auth_system
        manager.transport_security = mock_transport_security
        manager.consensus_security = mock_consensus_security
        manager.threat_detection = mock_threat_detection
        manager.trust_system = mock_trust_system
        manager.secure_aggregation = mock_secure_aggregation
        
        return manager

    @pytest.mark.asyncio
    async def test_initialize_coordinates_all_components(self, security_manager):
        """Test that initialization properly coordinates all security components."""
        # Given a security manager with mocked components
        # When initialize is called
        success = await security_manager.initialize()
        
        # Then it should coordinate with all components
        security_manager.auth_system.initialize.assert_called_once()
        security_manager.transport_security.initialize.assert_called_once()
        security_manager.threat_detection.health_check.assert_called_once()
        security_manager.trust_system.health_check.assert_called_once()
        
        # And return success
        assert success is True
        assert security_manager.started is True
    
    @pytest.mark.asyncio
    async def test_register_federated_node_orchestrates_security_setup(self, security_manager):
        """Test node registration orchestrates complete security setup."""
        # Given mocked components return success
        mock_identity = Mock()
        mock_identity.metadata = {"totp_secret": "test_secret"}
        security_manager.auth_system.register_node.return_value = mock_identity
        
        # When registering a node
        success, totp_secret = await security_manager.register_federated_node(
            "participant_001",
            NodeRole.PARTICIPANT,
            "secure_password123!",
            {"cpu_cores": 4},
            initial_trust=0.7
        )
        
        # Then it should orchestrate with authentication system
        security_manager.auth_system.register_node.assert_called_once_with(
            "participant_001",
            NodeRole.PARTICIPANT,
            "secure_password123!",
            {"cpu_cores": 4}
        )
        
        # And coordinate trust initialization
        security_manager.trust_system.initialize_node_trust.assert_called_once_with(
            "participant_001",
            bootstrap_method="neutral"
        )
        
        # And setup secure aggregation
        security_manager.secure_aggregation.setup_participant.assert_called_once_with(
            "participant_001",
            {"cpu_cores": 4},
            {"epsilon_limit": 10.0}
        )
        
        # And return registration success
        assert success is True
        assert totp_secret == "test_secret"
    
    @pytest.mark.asyncio
    async def test_authenticate_federated_participant_verifies_trust_first(self, security_manager):
        """Test authentication verifies trust before proceeding with auth."""
        # Given trust system approves the participant
        security_manager.trust_system.get_trust_decision.return_value = (True, 0.8, "approved")
        
        # And authentication succeeds
        mock_session = Mock()
        mock_session.session_id = "session_123"
        security_manager.auth_system.authenticate_node.return_value = (True, mock_session)
        
        # When authenticating a participant
        success, session_id, metadata = await security_manager.authenticate_federated_participant(
            "participant_001",
            "secure_password123!",
            "123456",
            {"device": "mobile"}
        )
        
        # Then it should check trust first
        security_manager.trust_system.get_trust_decision.assert_called_once_with(
            "participant_001",
            security_manager.trust_system.get_trust_decision.call_args[0][1],  # TrustLevel.LOW
            {"device": "mobile"}
        )
        
        # And then proceed with authentication
        security_manager.auth_system.authenticate_node.assert_called_once_with(
            "participant_001",
            "secure_password123!",
            "123456"
        )
        
        # And record successful interaction
        security_manager.trust_system.record_interaction.assert_called_once_with(
            "participant_001",
            "authentication",
            "success"
        )
        
        # And return success
        assert success is True
        assert session_id == "session_123"
        assert metadata["trust_score"] == 0.8
    
    @pytest.mark.asyncio
    async def test_authenticate_rejects_untrusted_participants(self, security_manager):
        """Test authentication rejects participants with insufficient trust."""
        # Given trust system rejects the participant
        security_manager.trust_system.get_trust_decision.return_value = (False, 0.2, "low_trust")
        
        # When authenticating a participant
        success, session_id, metadata = await security_manager.authenticate_federated_participant(
            "malicious_node",
            "password",
            "123456"
        )
        
        # Then it should check trust first
        security_manager.trust_system.get_trust_decision.assert_called_once()
        
        # And NOT proceed with authentication
        security_manager.auth_system.authenticate_node.assert_not_called()
        
        # And return rejection
        assert success is False
        assert session_id is None
        assert metadata["reason"] == "low_trust"
        assert metadata["trust_score"] == 0.2
    
    @pytest.mark.asyncio
    async def test_secure_federated_aggregation_validates_participants(self, security_manager):
        """Test secure aggregation validates all participants before proceeding."""
        # Given valid sessions and trust decisions
        security_manager.auth_system.validate_session.return_value = (True, Mock())
        security_manager.trust_system.get_trust_decision.return_value = (True, 0.8, "approved")
        
        # And gradients from multiple participants
        participant_gradients = [
            {
                "participant_id": "participant_001",
                "gradients": {"layer1": [0.1, 0.2]},
                "session_id": "session_001"
            },
            {
                "participant_id": "participant_002", 
                "gradients": {"layer1": [0.3, 0.4]},
                "session_id": "session_002"
            }
        ]
        
        # When performing secure aggregation
        success, result, metadata = await security_manager.secure_federated_aggregation(
            "aggregation_001",
            participant_gradients
        )
        
        # Then it should validate each participant's session
        assert security_manager.auth_system.validate_session.call_count == 2
        security_manager.auth_system.validate_session.assert_any_call("session_001")
        security_manager.auth_system.validate_session.assert_any_call("session_002")
        
        # And check trust for each participant
        assert security_manager.trust_system.get_trust_decision.call_count == 2
        
        # And create secure gradients
        assert security_manager.secure_aggregation.create_secure_gradient.call_count == 2
        
        # And perform aggregation
        security_manager.secure_aggregation.secure_aggregate.assert_called_once()
        
        # And record successful interactions
        assert security_manager.trust_system.record_interaction.call_count == 2
        
        # And return success
        assert success is True
        assert result == {"result": "aggregated"}
    
    @pytest.mark.asyncio
    async def test_establish_secure_communication_checks_trust_first(self, security_manager):
        """Test secure channel establishment verifies remote node trust."""
        # Given remote node is trusted
        security_manager.trust_system.get_trust_decision.return_value = (True, 0.9, "high_trust")
        security_manager.transport_security.establish_secure_channel.return_value = (True, "channel_001")
        
        # When establishing secure communication
        success, channel_id = await security_manager.establish_secure_communication(
            "remote_node_001"
        )
        
        # Then it should check trust first
        security_manager.trust_system.get_trust_decision.assert_called_once_with(
            "remote_node_001",
            security_manager.trust_system.get_trust_decision.call_args[0][1]  # TrustLevel.MEDIUM
        )
        
        # And then establish secure channel
        security_manager.transport_security.establish_secure_channel.assert_called_once()
        
        # And return success
        assert success is True
        assert channel_id == "channel_001"
    
    @pytest.mark.asyncio
    async def test_secure_consensus_round_filters_untrusted_participants(self, security_manager):
        """Test consensus round filters out untrusted participants."""
        # Given mixed trust decisions
        def trust_decision_side_effect(node_id, *args):
            if node_id == "trusted_node":
                return (True, 0.8, "approved")
            else:
                return (False, 0.3, "untrusted")
        
        security_manager.trust_system.get_trust_decision.side_effect = trust_decision_side_effect
        security_manager.consensus_security.detect_byzantine_behavior.return_value = []
        
        # When running consensus with mixed participants
        participants = ["trusted_node", "untrusted_node", "malicious_node"]
        success, result = await security_manager.secure_consensus_round(
            "round_001",
            participants,
            {"action": "update_model"}
        )
        
        # Then it should check trust for all participants
        assert security_manager.trust_system.get_trust_decision.call_count == 3
        
        # But only proceed with trusted participants
        # (In this case, only 1 trusted node, so consensus should fail due to minimum requirement)
        assert success is False
    
    @pytest.mark.asyncio
    async def test_handle_security_incident_coordinates_threat_response(self, security_manager):
        """Test security incident handling coordinates comprehensive threat response."""
        # Given threat detection finds threats
        mock_threat = Mock()
        mock_threat.event_id = "threat_001"
        mock_threat.threat_category.value = "malicious_activity"
        mock_threat.source_nodes = ["malicious_node"]
        
        security_manager.threat_detection.detect_threats.return_value = [mock_threat]
        security_manager.threat_detection.mitigate_threat.return_value = (True, ["block_node", "alert_admin"])
        
        # When handling a security incident
        success = await security_manager.handle_security_incident(
            "suspicious_activity",
            "source_node",
            {"severity": "high", "details": "anomalous_behavior"}
        )
        
        # Then it should ingest the event
        security_manager.threat_detection.ingest_security_event.assert_called_once_with(
            "suspicious_activity",
            "source_node",
            "test_node_001",
            {"severity": "high", "details": "anomalous_behavior"}
        )
        
        # And detect threats
        security_manager.threat_detection.detect_threats.assert_called_once()
        
        # And mitigate threats
        security_manager.threat_detection.mitigate_threat.assert_called_once_with(mock_threat)
        
        # And update trust for malicious nodes
        security_manager.trust_system.record_interaction.assert_called_once_with(
            "malicious_node",
            "security_incident", 
            "threat_detected"
        )
        
        # And return success
        assert success is True
    
    @pytest.mark.asyncio
    async def test_health_check_aggregates_component_health(self, security_manager):
        """Test health check aggregates status from all components."""
        # Given all components report healthy
        component_health = {"healthy": True, "issues": [], "warnings": []}
        security_manager.auth_system.health_check.return_value = component_health
        security_manager.transport_security.health_check.return_value = component_health
        security_manager.consensus_security.health_check.return_value = component_health
        security_manager.threat_detection.health_check.return_value = component_health
        security_manager.trust_system.health_check.return_value = component_health
        
        # When performing health check
        health_result = await security_manager.health_check()
        
        # Then it should check all components
        security_manager.auth_system.health_check.assert_called_once()
        security_manager.transport_security.health_check.assert_called_once()
        security_manager.consensus_security.health_check.assert_called_once()
        security_manager.threat_detection.health_check.assert_called_once()
        security_manager.trust_system.health_check.assert_called_once()
        
        # And aggregate results
        assert health_result["healthy"] is True
        assert "components" in health_result
        assert "security_summary" in health_result
        assert len(health_result["components"]) == 5

    def test_security_metrics_tracking(self, security_manager):
        """Test security metrics are properly tracked and aggregated."""
        # Given initial metrics
        assert security_manager.security_metrics.total_authentications == 0
        assert security_manager.security_metrics.threats_detected == 0
        
        # When security events occur, metrics should be updated
        # This tests the metric tracking behavior patterns
        security_manager.security_metrics.total_authentications += 1
        security_manager.security_metrics.threats_detected += 2
        
        # Then metrics should reflect the changes
        summary = security_manager.get_security_summary()
        assert summary["security_metrics"]["total_authentications"] == 1
        assert summary["security_metrics"]["threats_detected"] == 2
        assert summary["node_id"] == "test_node_001"

    def test_component_coordination_contracts(self, security_manager):
        """Test that component coordination follows expected contracts."""
        # This test verifies the contracts between components
        # without testing implementation details
        
        # Authentication system contract
        assert hasattr(security_manager.auth_system, 'initialize')
        assert hasattr(security_manager.auth_system, 'register_node')
        assert hasattr(security_manager.auth_system, 'authenticate_node')
        assert hasattr(security_manager.auth_system, 'health_check')
        
        # Transport security contract
        assert hasattr(security_manager.transport_security, 'initialize')
        assert hasattr(security_manager.transport_security, 'establish_secure_channel')
        assert hasattr(security_manager.transport_security, 'health_check')
        
        # Trust system contract
        assert hasattr(security_manager.trust_system, 'get_trust_decision')
        assert hasattr(security_manager.trust_system, 'record_interaction')
        assert hasattr(security_manager.trust_system, 'health_check')
        
        # Threat detection contract
        assert hasattr(security_manager.threat_detection, 'ingest_security_event')
        assert hasattr(security_manager.threat_detection, 'detect_threats')
        assert hasattr(security_manager.threat_detection, 'health_check')

if __name__ == "__main__":
    pytest.main([__file__, "-v"])