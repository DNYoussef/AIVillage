"""
Unit tests for FederatedAuthenticationSystem using TDD London School methodology.

Tests focus on interactions and behavior verification through mocks,
ensuring proper coordination between authentication components.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from dataclasses import dataclass
from typing import Dict, Any, Optional, Set, List

# Import the module under test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from infrastructure.security.federated_auth_system import (
    FederatedAuthenticationSystem,
    NodeRole,
    AuthenticationMethod,
    AuthenticationStatus,
    NodeIdentity,
    AuthenticationChallenge,
    AuthenticationSession
)


class TestFederatedAuthenticationSystem:
    """Test suite for FederatedAuthenticationSystem using London School TDD."""
    
    @pytest.fixture
    def auth_system(self):
        """Create authentication system for testing."""
        return FederatedAuthenticationSystem(
            jwt_secret="test_secret_key",
            enable_mfa=True
        )
    
    @pytest.fixture
    def mock_bcrypt(self):
        """Mock bcrypt for password hashing."""
        with patch('infrastructure.security.federated_auth_system.bcrypt') as mock:
            mock.hashpw.return_value = b'hashed_password'
            mock.checkpw.return_value = True
            yield mock
    
    @pytest.fixture 
    def mock_pyotp(self):
        """Mock pyotp for MFA token handling."""
        with patch('infrastructure.security.federated_auth_system.pyotp') as mock:
            mock.random_base32.return_value = "JBSWY3DPEHPK3PXP"
            
            mock_totp = Mock()
            mock_totp.verify.return_value = True
            mock.TOTP.return_value = mock_totp
            
            yield mock
    
    @pytest.fixture
    def mock_rsa(self):
        """Mock RSA key generation."""
        with patch('infrastructure.security.federated_auth_system.rsa') as mock:
            mock_private_key = Mock()
            mock_public_key = Mock()
            mock_public_key.public_bytes.return_value = b'public_key_bytes'
            mock_private_key.public_key.return_value = mock_public_key
            mock_private_key.private_bytes.return_value = b'private_key_bytes'
            
            mock.generate_private_key.return_value = mock_private_key
            yield mock

    @pytest.mark.asyncio
    async def test_register_node_coordinates_security_setup(self, auth_system, mock_bcrypt, 
                                                          mock_pyotp, mock_rsa):
        """Test node registration coordinates complete security setup."""
        # When registering a coordinator node with strong password
        identity = await auth_system.register_node(
            "coordinator_001",
            NodeRole.COORDINATOR,
            "SecurePassword123!",
            capabilities={"encryption": True},
            enable_certificate_auth=True
        )
        
        # Then it should coordinate password hashing
        mock_bcrypt.hashpw.assert_called_once()
        
        # And generate cryptographic keys
        mock_rsa.generate_private_key.assert_called()
        
        # And setup MFA for coordinator role
        mock_pyotp.random_base32.assert_called_once()
        
        # And configure appropriate authentication methods
        expected_methods = {
            AuthenticationMethod.PASSWORD,
            AuthenticationMethod.CERTIFICATE,
            AuthenticationMethod.MULTI_FACTOR
        }
        assert identity.authentication_methods == expected_methods
        
        # And store node identity
        assert auth_system.node_identities["coordinator_001"] == identity
        assert identity.role == NodeRole.COORDINATOR
        assert identity.capabilities == {"encryption": True}
    
    @pytest.mark.asyncio
    async def test_register_participant_node_uses_basic_auth(self, auth_system, mock_bcrypt, 
                                                           mock_pyotp, mock_rsa):
        """Test participant node registration uses appropriate security level."""
        # When registering a participant node
        identity = await auth_system.register_node(
            "participant_001",
            NodeRole.PARTICIPANT, 
            "StrongPassword456!",
            enable_certificate_auth=False
        )
        
        # Then it should use password authentication only
        expected_methods = {AuthenticationMethod.PASSWORD}
        assert identity.authentication_methods == expected_methods
        
        # And not generate TOTP secret
        assert "totp_secret" not in identity.metadata
    
    def test_register_node_rejects_weak_passwords(self, auth_system):
        """Test registration rejects passwords that don't meet policy."""
        # When attempting to register with weak password
        with pytest.raises(ValueError, match="Password does not meet security requirements"):
            asyncio.run(auth_system.register_node(
                "weak_node",
                NodeRole.PARTICIPANT,
                "weak"  # Too short, no special chars, etc.
            ))
    
    def test_register_duplicate_node_raises_error(self, auth_system, mock_bcrypt, mock_rsa):
        """Test registration prevents duplicate node IDs."""
        # Given a node is already registered
        asyncio.run(auth_system.register_node(
            "existing_node",
            NodeRole.PARTICIPANT,
            "SecurePassword123!"
        ))
        
        # When attempting to register same node ID again
        with pytest.raises(ValueError, match="Node existing_node already registered"):
            asyncio.run(auth_system.register_node(
                "existing_node", 
                NodeRole.COORDINATOR,
                "AnotherPassword456!"
            ))
    
    @pytest.mark.asyncio
    async def test_authenticate_node_coordinates_multi_factor_verification(self, auth_system,
                                                                         mock_bcrypt, mock_pyotp, mock_rsa):
        """Test node authentication coordinates multi-factor verification."""
        # Given a coordinator node with MFA enabled
        await auth_system.register_node(
            "coordinator_001",
            NodeRole.COORDINATOR,
            "SecurePassword123!"
        )
        
        # When authenticating with password and MFA token
        success, session = await auth_system.authenticate_node(
            "coordinator_001",
            "SecurePassword123!",
            mfa_token="123456"
        )
        
        # Then it should verify password
        mock_bcrypt.checkpw.assert_called_once_with(
            "SecurePassword123!".encode(),
            mock_bcrypt.hashpw.return_value
        )
        
        # And verify MFA token
        mock_pyotp.TOTP.assert_called_once_with("JBSWY3DPEHPK3PXP")
        mock_pyotp.TOTP.return_value.verify.assert_called_once_with("123456", valid_window=1)
        
        # And create session with appropriate methods
        assert success is True
        assert session is not None
        assert AuthenticationMethod.PASSWORD in session.authenticated_methods
        assert AuthenticationMethod.MULTI_FACTOR in session.authenticated_methods
    
    @pytest.mark.asyncio
    async def test_authenticate_node_rejects_missing_mfa_for_coordinator(self, auth_system,
                                                                       mock_bcrypt, mock_rsa):
        """Test authentication rejects coordinators without MFA."""
        # Given a coordinator node requiring MFA
        await auth_system.register_node(
            "coordinator_001", 
            NodeRole.COORDINATOR,
            "SecurePassword123!"
        )
        
        # When authenticating without MFA token
        success, session = await auth_system.authenticate_node(
            "coordinator_001",
            "SecurePassword123!"
            # No MFA token provided
        )
        
        # Then authentication should fail
        assert success is False
        assert session is None
        assert auth_system.auth_stats["failed_authentications"] == 1
    
    @pytest.mark.asyncio
    async def test_authenticate_unknown_node_fails_gracefully(self, auth_system):
        """Test authentication handles unknown nodes gracefully."""
        # When authenticating non-existent node
        success, session = await auth_system.authenticate_node(
            "unknown_node",
            "AnyPassword123!"
        )
        
        # Then authentication should fail gracefully
        assert success is False
        assert session is None
        assert auth_system.auth_stats["failed_authentications"] == 1
    
    @pytest.mark.asyncio
    async def test_create_authentication_challenge_generates_proper_challenge(self, auth_system, mock_rsa):
        """Test challenge creation generates proper zero-knowledge challenge."""
        # Given a registered node
        await auth_system.register_node(
            "participant_001",
            NodeRole.PARTICIPANT,
            "SecurePassword123!"
        )
        
        # When creating authentication challenge
        challenge = await auth_system.create_authentication_challenge(
            "participant_001",
            AuthenticationMethod.ZERO_KNOWLEDGE
        )
        
        # Then it should create proper challenge structure
        assert challenge is not None
        assert challenge.node_id == "participant_001"
        assert challenge.method == AuthenticationMethod.ZERO_KNOWLEDGE
        assert len(challenge.challenge_data) == 32  # 32 random bytes
        assert challenge.challenge_id in auth_system.active_challenges
        
        # And set proper expiration
        assert challenge.expires_at > time.time()
        assert challenge.expires_at <= time.time() + 300  # 5 minutes
    
    @pytest.mark.asyncio
    async def test_validate_session_checks_expiration_and_activity(self, auth_system, mock_bcrypt, mock_rsa):
        """Test session validation checks expiration and updates activity."""
        # Given an authenticated session
        await auth_system.register_node(
            "participant_001",
            NodeRole.PARTICIPANT,
            "SecurePassword123!"
        )
        
        success, session = await auth_system.authenticate_node(
            "participant_001",
            "SecurePassword123!"
        )
        
        original_activity = session.last_activity
        
        # When validating session
        valid, validated_session = await auth_system.validate_session(session.session_id)
        
        # Then it should confirm validity
        assert valid is True
        assert validated_session == session
        
        # And update last activity
        assert validated_session.last_activity > original_activity
    
    @pytest.mark.asyncio
    async def test_validate_expired_session_revokes_automatically(self, auth_system, mock_bcrypt, mock_rsa):
        """Test expired session validation triggers automatic revocation."""
        # Given an authenticated session
        await auth_system.register_node(
            "participant_001",
            NodeRole.PARTICIPANT, 
            "SecurePassword123!"
        )
        
        success, session = await auth_system.authenticate_node(
            "participant_001",
            "SecurePassword123!"
        )
        
        # And session is expired
        session.expires_at = time.time() - 1  # 1 second ago
        
        # When validating expired session
        valid, validated_session = await auth_system.validate_session(session.session_id)
        
        # Then it should revoke automatically
        assert valid is False
        assert validated_session is None
        assert session.session_id not in auth_system.active_sessions
        assert session.session_id in auth_system.revoked_tokens
    
    @pytest.mark.asyncio
    async def test_revoke_session_coordinates_cleanup(self, auth_system, mock_bcrypt, mock_rsa):
        """Test session revocation coordinates proper cleanup."""
        # Given an active session
        await auth_system.register_node(
            "participant_001",
            NodeRole.PARTICIPANT,
            "SecurePassword123!"
        )
        
        success, session = await auth_system.authenticate_node(
            "participant_001",
            "SecurePassword123!"
        )
        
        session_id = session.session_id
        
        # When revoking session
        revoked = await auth_system.revoke_session(session_id)
        
        # Then it should coordinate cleanup
        assert revoked is True
        assert session_id not in auth_system.active_sessions
        assert session_id in auth_system.revoked_tokens
        assert session.is_active is False
        assert auth_system.auth_stats["revoked_sessions"] == 1
    
    @pytest.mark.asyncio
    async def test_update_node_reputation_adjusts_trust_level(self, auth_system, mock_bcrypt, mock_rsa):
        """Test reputation updates coordinate trust level adjustments."""
        # Given a registered node
        identity = await auth_system.register_node(
            "participant_001",
            NodeRole.PARTICIPANT,
            "SecurePassword123!"
        )
        
        assert identity.reputation_score == 0.5  # Initial score
        assert identity.trust_level == "basic"   # Initial trust
        
        # When updating reputation positively
        success = await auth_system.update_node_reputation("participant_001", 0.3)
        
        # Then it should coordinate trust level adjustment
        assert success is True
        assert identity.reputation_score == 0.8
        assert identity.trust_level == "medium"
        
        # When updating reputation to high level
        await auth_system.update_node_reputation("participant_001", 0.15)
        
        # Then trust level should increase
        assert identity.reputation_score == 0.95
        assert identity.trust_level == "high"
    
    @pytest.mark.asyncio
    async def test_get_node_permissions_coordinates_rbac(self, auth_system, mock_bcrypt, mock_rsa):
        """Test permission retrieval coordinates role-based access control."""
        # Given nodes with different roles and trust levels
        coordinator = await auth_system.register_node(
            "coordinator_001",
            NodeRole.COORDINATOR,
            "SecurePassword123!"
        )
        
        participant = await auth_system.register_node(
            "participant_001", 
            NodeRole.PARTICIPANT,
            "SecurePassword123!"
        )
        
        # And high trust coordinator
        await auth_system.update_node_reputation("coordinator_001", 0.45)  # -> 0.95 high trust
        
        # When getting permissions
        coordinator_perms = await auth_system.get_node_permissions("coordinator_001")
        participant_perms = await auth_system.get_node_permissions("participant_001")
        
        # Then it should coordinate role-based permissions
        assert "create_training_rounds" in coordinator_perms
        assert "manage_participants" in coordinator_perms
        assert "priority_participation" in coordinator_perms  # High trust bonus
        
        assert "participate_training" in participant_perms
        assert "create_training_rounds" not in participant_perms
        assert "priority_participation" not in participant_perms  # Basic trust
    
    @pytest.mark.asyncio
    async def test_health_check_coordinates_session_cleanup(self, auth_system, mock_bcrypt, mock_rsa):
        """Test health check coordinates automatic session cleanup."""
        # Given multiple sessions, some expired
        await auth_system.register_node(
            "participant_001",
            NodeRole.PARTICIPANT,
            "SecurePassword123!"
        )
        
        # Create active session
        success, active_session = await auth_system.authenticate_node(
            "participant_001",
            "SecurePassword123!"
        )
        
        # Create expired session
        success, expired_session = await auth_system.authenticate_node(
            "participant_001", 
            "SecurePassword123!"
        )
        expired_session.expires_at = time.time() - 1
        
        initial_sessions = len(auth_system.active_sessions)
        
        # When performing health check
        health_result = await auth_system.health_check()
        
        # Then it should coordinate cleanup
        assert health_result["healthy"] is True
        assert health_result["expired_sessions_cleaned"] == 1
        assert len(auth_system.active_sessions) == initial_sessions - 1
        assert expired_session.session_id not in auth_system.active_sessions
        assert active_session.session_id in auth_system.active_sessions

    def test_get_auth_stats_aggregates_metrics(self, auth_system):
        """Test statistics aggregation coordinates metric collection."""
        # Given some authentication activity
        auth_system.auth_stats["total_authentications"] = 10
        auth_system.auth_stats["failed_authentications"] = 2
        auth_system.auth_stats["mfa_verifications"] = 5
        
        # When getting statistics
        stats = auth_system.get_auth_stats()
        
        # Then it should coordinate metric aggregation
        assert stats["total_authentications"] == 10
        assert stats["failed_authentications"] == 2
        assert stats["mfa_verifications"] == 5
        assert stats["success_rate"] == 0.8  # (10-2)/10
        assert "registered_nodes" in stats
        assert "active_challenges" in stats
        assert "active_sessions" in stats

    def test_password_policy_validation_enforces_security(self, auth_system):
        """Test password policy validation enforces security requirements."""
        # Test various password policy violations
        test_cases = [
            ("short", False, "too short"),
            ("nouppercase123!", False, "no uppercase"),
            ("NOLOWERCASE123!", False, "no lowercase"), 
            ("NoNumbers!", False, "no numbers"),
            ("NoSpecialChars123", False, "no special chars"),
            ("ValidPassword123!", True, "meets all requirements")
        ]
        
        for password, expected_valid, description in test_cases:
            result = auth_system._validate_password_policy(password)
            assert result == expected_valid, f"Password '{password}' {description} should be {expected_valid}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])