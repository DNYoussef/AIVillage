"""Comprehensive Security Tests for AIVillage Enhanced Security Framework.

Tests AES-256-GCM encryption, MFA system, Redis session management,
and overall security improvements.
"""

import base64
from datetime import datetime, timedelta
import json
import os
import secrets
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Import security modules
from infrastructure.shared.security.enhanced_encryption import (
    DigitalTwinEncryptionError,
    EnhancedDigitalTwinEncryption,
    KeyRotationManager,
)
from infrastructure.shared.security.mfa_system import MFAError, MFAMethodType, MFASystem
from infrastructure.shared.security.redis_session_manager import DeviceInfo, RedisSessionManager, SessionData


class TestEnhancedEncryption:
    """Test AES-256-GCM encryption system."""

    def setup_method(self):
        """Set up test environment."""
        # Generate test master key
        self.test_master_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
        os.environ["DIGITAL_TWIN_MASTER_KEY"] = self.test_master_key

        # Clear any existing key version
        os.environ.pop("CURRENT_KEY_VERSION", None)

        self.encryption = EnhancedDigitalTwinEncryption()

    def teardown_method(self):
        """Clean up test environment."""
        os.environ.pop("DIGITAL_TWIN_MASTER_KEY", None)
        os.environ.pop("CURRENT_KEY_VERSION", None)

    def test_initialization(self):
        """Test encryption system initialization."""
        assert self.encryption.master_key is not None
        assert self.encryption.current_key_version is not None
        assert len(self.encryption.current_key_version) > 0

    def test_key_derivation(self):
        """Test key derivation from master key."""
        version1 = "v_20240101_120000_test1"
        version2 = "v_20240101_120000_test2"

        key1 = self.encryption._derive_key_from_master(version1)
        key2 = self.encryption._derive_key_from_master(version2)

        # Keys should be different for different versions
        assert key1 != key2
        assert len(key1) == 32  # AES-256 key length
        assert len(key2) == 32

        # Same version should produce same key
        key1_repeat = self.encryption._derive_key_from_master(version1)
        assert key1 == key1_repeat

    def test_encrypt_decrypt_cycle(self):
        """Test encryption and decryption of data."""
        test_data = "sensitive_user_data_12345"
        field_name = "test_field"

        # Encrypt data
        encrypted = self.encryption.encrypt_sensitive_field(test_data, field_name)
        assert isinstance(encrypted, bytes)
        assert len(encrypted) > len(test_data.encode())

        # Decrypt data
        decrypted = self.encryption.decrypt_sensitive_field(encrypted, field_name)
        assert decrypted == test_data

    def test_json_data_handling(self):
        """Test encryption of complex JSON data."""
        test_data = {
            "user_id": "user123",
            "preferences": {"theme": "dark", "notifications": True},
            "sensitive_info": "secret_value",
        }

        encrypted = self.encryption.encrypt_sensitive_field(test_data, "json_field")
        decrypted_str = self.encryption.decrypt_sensitive_field(encrypted, "json_field")
        decrypted_data = json.loads(decrypted_str)

        assert decrypted_data == test_data

    def test_encryption_package_format(self):
        """Test encryption package contains required fields."""
        test_data = "test_data"
        encrypted = self.encryption.encrypt_sensitive_field(test_data, "test")

        # Parse encrypted package
        package_str = encrypted.decode("utf-8")
        package = json.loads(package_str)

        required_fields = ["version", "iv", "tag", "data", "algorithm"]
        for field in required_fields:
            assert field in package

        assert package["algorithm"] == "AES-256-GCM"
        assert package["version"] == self.encryption.current_key_version

    def test_key_rotation(self):
        """Test key rotation functionality."""
        # Get initial key version
        initial_version = self.encryption.current_key_version

        # Encrypt data with initial key
        test_data = "data_before_rotation"
        encrypted_old = self.encryption.encrypt_sensitive_field(test_data, "test")

        # Rotate keys
        new_version = self.encryption.rotate_keys()
        assert new_version != initial_version
        assert self.encryption.current_key_version == new_version

        # New encryption should use new key
        encrypted_new = self.encryption.encrypt_sensitive_field(test_data, "test")

        # Both old and new encrypted data should decrypt correctly
        decrypted_old = self.encryption.decrypt_sensitive_field(encrypted_old, "test")
        decrypted_new = self.encryption.decrypt_sensitive_field(encrypted_new, "test")

        assert decrypted_old == test_data
        assert decrypted_new == test_data

    def test_key_status(self):
        """Test key status reporting."""
        status = self.encryption.get_key_status()

        required_fields = [
            "current_version",
            "created_at",
            "rotation_needed",
            "days_until_rotation",
            "active_versions",
            "algorithm",
        ]

        for field in required_fields:
            assert field in status

        assert status["algorithm"] == "AES-256-GCM"
        assert isinstance(status["rotation_needed"], bool)
        assert isinstance(status["active_versions"], list)

    def test_backward_compatibility(self):
        """Test backward compatibility with legacy Fernet encryption."""
        # Mock legacy Fernet cipher
        with patch("cryptography.fernet.Fernet") as mock_fernet:
            mock_cipher = Mock()
            mock_cipher.decrypt.return_value = b"legacy_decrypted_data"
            mock_fernet.return_value = mock_cipher

            # Set legacy key
            os.environ["DIGITAL_TWIN_ENCRYPTION_KEY"] = "test_legacy_key"

            # Create new encryption instance
            encryption_with_legacy = EnhancedDigitalTwinEncryption()

            # Test legacy decryption
            legacy_encrypted = b"legacy_encrypted_data"
            decrypted = encryption_with_legacy.decrypt_sensitive_field(legacy_encrypted, "test")

            assert decrypted == "legacy_decrypted_data"
            mock_cipher.decrypt.assert_called_once_with(legacy_encrypted)


class TestKeyRotationManager:
    """Test key rotation management."""

    def setup_method(self):
        self.rotation_manager = KeyRotationManager(rotation_days=30)

    def test_key_version_generation(self):
        """Test key version generation."""
        version1 = self.rotation_manager.generate_key_version()
        version2 = self.rotation_manager.generate_key_version()

        assert version1 != version2
        assert version1.startswith("v_")
        assert version2.startswith("v_")

    def test_rotation_needed_check(self):
        """Test rotation needed determination."""
        now = datetime.utcnow()

        # Recent key - no rotation needed
        recent_key = now - timedelta(days=10)
        assert not self.rotation_manager.is_rotation_needed(recent_key)

        # Old key - rotation needed
        old_key = now - timedelta(days=35)
        assert self.rotation_manager.is_rotation_needed(old_key)

    def test_active_keys_management(self):
        """Test active keys tracking."""
        now = datetime.utcnow()

        # Add some test keys
        self.rotation_manager.key_versions = {
            "v1": {"created_at": now - timedelta(days=10), "key": b"key1"},
            "v2": {"created_at": now - timedelta(days=40), "key": b"key2"},
            "v3": {"created_at": now - timedelta(days=70), "key": b"key3"},
        }

        active_keys = self.rotation_manager.get_active_keys()

        # Only v1 and v2 should be active (within 2x rotation period)
        assert "v1" in active_keys
        assert "v2" in active_keys
        assert "v3" not in active_keys


class TestMFASystem:
    """Test Multi-Factor Authentication system."""

    def setup_method(self):
        self.mfa = MFASystem()

    def test_totp_setup(self):
        """Test TOTP setup process."""
        user_id = "test_user"
        user_email = "test@aivillage.com"

        setup_data = self.mfa.setup_totp(user_id, user_email)

        required_fields = ["secret", "qr_code", "backup_codes", "method"]
        for field in required_fields:
            assert field in setup_data

        assert setup_data["method"] == MFAMethodType.TOTP
        assert isinstance(setup_data["backup_codes"], list)
        assert len(setup_data["backup_codes"]) == 10

    def test_totp_verification(self):
        """Test TOTP token verification."""
        secret = self.mfa.totp_manager.generate_secret()

        # Generate current token
        current_token = self.mfa.totp_manager.get_current_token(secret)

        # Verify token
        is_valid = self.mfa.verify_totp("test_user", current_token, secret)
        assert is_valid

        # Invalid token should fail
        is_valid = self.mfa.verify_totp("test_user", "123456", secret)
        assert not is_valid

    def test_backup_codes(self):
        """Test backup codes generation and verification."""
        codes = self.mfa.backup_codes.generate_backup_codes(count=5)

        assert len(codes) == 5
        for code in codes:
            assert "-" in code  # Format: XXXX-XXXX
            assert len(code) == 9  # 4 chars + dash + 4 chars

        # Test hashing and verification
        test_code = codes[0]
        hashed_code = self.mfa.backup_codes.hash_backup_code(test_code)

        # Correct code should verify
        assert self.mfa.backup_codes.verify_backup_code(test_code, hashed_code)

        # Incorrect code should not verify
        assert not self.mfa.backup_codes.verify_backup_code("WRONG-CODE", hashed_code)

    def test_sms_verification_flow(self):
        """Test SMS verification process."""
        user_id = "test_user"
        phone_number = "+1234567890"

        # Mock SMS provider
        with patch.object(self.mfa.sms_provider, "send_sms", return_value=True):
            # Send SMS
            sent = self.mfa.send_sms_verification(user_id, phone_number)
            assert sent

            # Get stored verification code (mock)
            stored_key = f"{user_id}:{MFAMethodType.SMS}"
            assert stored_key in self.mfa.verification_codes

    def test_email_verification_flow(self):
        """Test email verification process."""
        user_id = "test_user"
        email = "test@aivillage.com"

        # Mock email provider
        with patch.object(self.mfa.email_provider, "send_email", return_value=True):
            # Send email
            sent = self.mfa.send_email_verification(user_id, email)
            assert sent

            # Get stored verification code (mock)
            stored_key = f"{user_id}:{MFAMethodType.EMAIL}"
            assert stored_key in self.mfa.verification_codes

    def test_rate_limiting(self):
        """Test rate limiting for MFA attempts."""
        user_id = "test_user"
        method = MFAMethodType.SMS

        # First 5 attempts should be allowed
        for i in range(5):
            allowed = self.mfa.check_rate_limit(user_id, method)
            assert allowed

        # 6th attempt should be blocked
        blocked = self.mfa.check_rate_limit(user_id, method)
        assert not blocked

    def test_unified_verification(self):
        """Test unified MFA verification method."""
        user_id = "test_user"

        # Test TOTP verification
        secret = self.mfa.totp_manager.generate_secret()
        token = self.mfa.totp_manager.get_current_token(secret)

        verified = self.mfa.verify_mfa(user_id, MFAMethodType.TOTP, token, secret=secret)
        assert verified

        # Test invalid method
        verified = self.mfa.verify_mfa(user_id, "invalid_method", token)
        assert not verified


@pytest.mark.asyncio
class TestRedisSessionManager:
    """Test Redis session management system."""

    def setup_method(self):
        # Use mock Redis client for testing
        self.session_manager = RedisSessionManager()
        self.session_manager.redis_client = AsyncMock()

    async def test_session_creation(self):
        """Test session creation process."""
        user_id = "test_user"
        device_info = DeviceInfo(user_agent="Mozilla/5.0 Test Browser", ip_address="192.168.1.100")

        # Mock Redis operations
        self.session_manager.redis_client.pipeline.return_value.execute = AsyncMock()
        self.session_manager.redis_client.smembers.return_value = set()

        session_id = await self.session_manager.create_session(user_id, device_info)

        assert session_id.startswith("sess_")
        assert len(session_id) > 10

    async def test_device_info_handling(self):
        """Test device information processing."""
        device = DeviceInfo(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)", ip_address="10.0.0.100")

        assert device.device_fingerprint is not None
        assert len(device.device_fingerprint) == 16  # Truncated SHA256

        # Test serialization
        device_dict = device.to_dict()
        device_restored = DeviceInfo.from_dict(device_dict)

        assert device_restored.user_agent == device.user_agent
        assert device_restored.ip_address == device.ip_address
        assert device_restored.device_fingerprint == device.device_fingerprint

    async def test_token_session_mapping(self):
        """Test token to session mapping."""
        session_id = "test_session_123"
        jti = "token_jti_456"

        # Mock Redis operations
        self.session_manager.redis_client.hgetall.return_value = {
            "user_id": "test_user",
            "session_id": session_id,
            "device_info": json.dumps(
                {
                    "user_agent": "Test Browser",
                    "ip_address": "127.0.0.1",
                    "device_fingerprint": "test_fingerprint",
                    "first_seen": datetime.utcnow().isoformat(),
                    "last_seen": datetime.utcnow().isoformat(),
                }
            ),
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "access_tokens": json.dumps([]),
            "refresh_tokens": json.dumps([]),
            "is_active": True,
            "security_flags": json.dumps([]),
        }

        await self.session_manager.add_token_to_session(session_id, jti, "access")

        # Verify Redis operations were called
        self.session_manager.redis_client.hgetall.assert_called()
        self.session_manager.redis_client.setex.assert_called()

    async def test_token_revocation(self):
        """Test token revocation process."""
        jti = "test_token_123"
        session_id = "test_session_456"

        # Mock Redis responses
        self.session_manager.redis_client.get.return_value = session_id
        self.session_manager.redis_client.hgetall.return_value = {
            "user_id": "test_user",
            "session_id": session_id,
            "device_info": json.dumps(
                {
                    "user_agent": "Test Browser",
                    "ip_address": "127.0.0.1",
                    "device_fingerprint": "test_fingerprint",
                    "first_seen": datetime.utcnow().isoformat(),
                    "last_seen": datetime.utcnow().isoformat(),
                }
            ),
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "access_tokens": json.dumps([jti]),
            "refresh_tokens": json.dumps([]),
            "is_active": True,
            "security_flags": json.dumps([]),
        }

        revoked = await self.session_manager.revoke_token(jti)
        assert revoked

        # Verify revoked token is added to set
        self.session_manager.redis_client.sadd.assert_called()

    async def test_suspicious_activity_detection(self):
        """Test suspicious activity detection."""
        user_id = "test_user"
        device_info = DeviceInfo("New Browser", "192.168.1.200")

        # Mock existing sessions with different IPs
        mock_sessions = [
            SessionData(user_id, "sess1", DeviceInfo("Browser1", "10.0.0.1")),
            SessionData(user_id, "sess2", DeviceInfo("Browser2", "10.0.0.2")),
            SessionData(user_id, "sess3", DeviceInfo("Browser3", "10.0.0.3")),
            SessionData(user_id, "sess4", DeviceInfo("Browser4", "10.0.0.4")),
        ]

        # Update last activity to recent
        now = datetime.utcnow()
        for session in mock_sessions:
            session.last_activity = now - timedelta(minutes=30)

        with patch.object(self.session_manager, "get_user_sessions", return_value=mock_sessions):
            suspicious = await self.session_manager.detect_suspicious_activity(user_id, device_info)
            assert suspicious  # More than 3 IPs in last hour

    async def test_session_limits_enforcement(self):
        """Test session limits enforcement."""
        user_id = "test_user"

        # Mock existing sessions (at limit)
        existing_sessions = []
        for i in range(self.session_manager.max_sessions_per_user):
            session = SessionData(user_id, f"sess_{i}", DeviceInfo(f"Browser{i}", f"10.0.0.{i}"))
            session.last_activity = datetime.utcnow() - timedelta(minutes=i * 10)
            existing_sessions.append(session)

        with patch.object(self.session_manager, "get_user_sessions", return_value=existing_sessions):
            with patch.object(self.session_manager, "revoke_session", return_value=True) as mock_revoke:
                await self.session_manager._enforce_session_limits(user_id)

                # Should revoke oldest session
                mock_revoke.assert_called_once()

    async def test_health_check(self):
        """Test Redis health check."""
        # Mock successful ping
        self.session_manager.redis_client.ping = AsyncMock()
        self.session_manager.redis_client.info.return_value = {
            "connected_clients": 5,
            "used_memory_human": "1.5M",
            "redis_version": "6.2.0",
        }

        health = await self.session_manager.health_check()

        assert health["status"] == "healthy"
        assert "latency_ms" in health
        assert health["connected_clients"] == 5


class TestIntegratedSecurity:
    """Test integrated security system functionality."""

    def setup_method(self):
        """Set up integrated test environment."""
        # Set up encryption
        os.environ["DIGITAL_TWIN_MASTER_KEY"] = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
        self.encryption = EnhancedDigitalTwinEncryption()

        # Set up MFA
        self.mfa = MFASystem()

        # Set up session manager (mocked)
        self.session_manager = RedisSessionManager()
        self.session_manager.redis_client = AsyncMock()

    def teardown_method(self):
        os.environ.pop("DIGITAL_TWIN_MASTER_KEY", None)
        os.environ.pop("CURRENT_KEY_VERSION", None)

    def test_security_rating_improvements(self):
        """Test that security improvements meet B+ rating criteria."""
        # Check AES-256-GCM encryption
        key_status = self.encryption.get_key_status()
        assert key_status["algorithm"] == "AES-256-GCM"

        # Check key rotation capability
        assert not key_status["rotation_needed"] or key_status["days_until_rotation"] > 0

        # Check MFA availability
        mfa_status = self.mfa.get_user_mfa_status("test_user")
        expected_methods = [MFAMethodType.TOTP, MFAMethodType.SMS, MFAMethodType.EMAIL, MFAMethodType.BACKUP_CODES]
        for method in expected_methods:
            assert method in mfa_status["methods_available"]

        # Check session management capability
        assert self.session_manager.key_prefix == "aivillage:session"
        assert self.session_manager.max_sessions_per_user > 0

    @pytest.mark.asyncio
    async def test_complete_security_workflow(self):
        """Test complete security workflow from encryption to session management."""
        user_id = "workflow_test_user"

        # 1. Encrypt sensitive data
        sensitive_data = {"ssn": "123-45-6789", "credit_card": "4111-1111-1111-1111"}
        encrypted_data = self.encryption.encrypt_sensitive_field(sensitive_data, "user_pii")

        # 2. Set up MFA
        mfa_setup = self.mfa.setup_totp(user_id, "test@example.com")
        assert mfa_setup["method"] == MFAMethodType.TOTP

        # 3. Create session (mocked)
        device_info = DeviceInfo("Test Browser", "127.0.0.1")
        self.session_manager.redis_client.pipeline.return_value.execute = AsyncMock()
        self.session_manager.redis_client.smembers.return_value = set()

        session_id = await self.session_manager.create_session(user_id, device_info)

        # 4. Verify all components work together
        decrypted_data = self.encryption.decrypt_sensitive_field(encrypted_data, "user_pii")
        assert json.loads(decrypted_data) == sensitive_data

        totp_verified = self.mfa.verify_totp(
            user_id, self.mfa.totp_manager.get_current_token(mfa_setup["secret"]), mfa_setup["secret"]
        )
        assert totp_verified

        assert session_id.startswith("sess_")

    def test_backward_compatibility_preserved(self):
        """Test that backward compatibility is preserved."""
        # Test that legacy environment variables still work
        os.environ["DIGITAL_TWIN_ENCRYPTION_KEY"] = "test_legacy_key"

        with patch("cryptography.fernet.Fernet") as mock_fernet:
            mock_cipher = Mock()
            mock_fernet.return_value = mock_cipher

            encryption_with_legacy = EnhancedDigitalTwinEncryption()
            assert encryption_with_legacy.legacy_cipher is not None

    def test_security_error_handling(self):
        """Test proper error handling in security components."""
        # Test encryption errors
        with pytest.raises(DigitalTwinEncryptionError):
            # Invalid encrypted data
            self.encryption.decrypt_sensitive_field(b"invalid_data", "test")

        # Test MFA errors
        with pytest.raises(MFAError):
            # Invalid QR code generation
            self.mfa.totp_manager.generate_qr_code("invalid_secret", "test@example.com")

    @pytest.mark.asyncio
    async def test_session_security_features(self):
        """Test advanced session security features."""
        user_id = "security_test_user"
        DeviceInfo("Suspicious Browser", "192.168.1.100")

        # Mock session creation
        self.session_manager.redis_client.pipeline.return_value.execute = AsyncMock()
        self.session_manager.redis_client.smembers.return_value = set()

        # Test session analytics
        analytics = await self.session_manager.get_session_analytics(user_id)
        assert "user_id" in analytics or "message" in analytics

        # Test health monitoring
        self.session_manager.redis_client.ping = AsyncMock()
        self.session_manager.redis_client.info.return_value = {"redis_version": "6.2.0"}

        health = await self.session_manager.health_check()
        assert "status" in health


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
