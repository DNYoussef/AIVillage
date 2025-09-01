"""Tests for Authentication & Access Control System - Prompt H

Comprehensive validation of authentication and authorization including:
- Password hashing and validation
- Multi-factor authentication (MFA)
- Role-based access control (RBAC)
- Session management and token validation
- API key management and audit logging

Integration Point: Security validation for Phase 4 testing
"""

import os
from pathlib import Path
import sys
import tempfile
import time

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from security.auth_system import (
    ApiKey,
    AuthConfig,
    AuthenticationManager,
    AuthorizationManager,
    MFAManager,
    PasswordManager,
    Permission,
    SecurityLevel,
    TokenManager,
    User,
    UserRole,
)


class TestPasswordManager:  # pragma: allowlist secret
    """Test password management functionality."""

    def test_password_manager_initialization(self):  # pragma: allowlist secret
        """Test password manager initialization."""
        pm = PasswordManager()  # pragma: allowlist secret

        assert pm.salt_length == 32
        assert pm.iterations == 100000

    def test_password_hashing(self):  # pragma: allowlist secret
        """Test password hashing."""
        pm = PasswordManager()  # pragma: allowlist secret
        password = "test_password_hash_123!"  # nosec B106 - test password # pragma: allowlist secret

        hash1 = pm.hash_password(password)  # pragma: allowlist secret
        hash2 = pm.hash_password(password)  # pragma: allowlist secret

        # Hashes should be different due to salt
        assert hash1 != hash2
        assert len(hash1) > 0
        assert len(hash2) > 0

    def test_password_verification(self):  # pragma: allowlist secret
        """Test password verification."""
        pm = PasswordManager()  # pragma: allowlist secret
        password = "test_password_verify_123!"  # pragma: allowlist secret

        password_hash = pm.hash_password(password)  # pragma: allowlist secret

        # Correct password should verify
        assert pm.verify_password(password, password_hash) is True

        # Wrong password should not verify
        assert pm.verify_password("WrongPassword", password_hash) is False

    def test_password_strength_validation(self):  # pragma: allowlist secret
        """Test password strength validation."""
        pm = PasswordManager()  # pragma: allowlist secret
        config = AuthConfig()

        # Strong password
        strong_password = "test_strong_password_123!"  # pragma: allowlist secret
        is_valid, errors = pm.validate_password_strength(strong_password, config)  # pragma: allowlist secret
        assert is_valid is True
        assert len(errors) == 0

        # Weak password - too short
        weak_password = "test_short"  # pragma: allowlist secret
        is_valid, errors = pm.validate_password_strength(weak_password, config)  # pragma: allowlist secret
        assert is_valid is False
        assert len(errors) > 0
        assert any("at least" in error for error in errors)

        # Password missing uppercase
        no_upper = "lowercase123!"
        is_valid, errors = pm.validate_password_strength(no_upper, config)  # pragma: allowlist secret
        assert is_valid is False
        assert any("uppercase" in error for error in errors)

        # Password missing symbols
        no_symbols = "NoSymbols123"
        is_valid, errors = pm.validate_password_strength(no_symbols, config)  # pragma: allowlist secret
        assert is_valid is False
        assert any("special" in error for error in errors)


class TestTokenManager:
    """Test token management functionality."""

    def test_token_manager_initialization(self):
        """Test token manager initialization."""
        tm = TokenManager()

        assert tm.secret_key is not None
        assert len(tm.secret_key) > 0

    def test_token_creation_and_verification(self):
        """Test token creation and verification."""
        tm = TokenManager()
        user_id = "test_user"
        permissions = ["read", "write"]

        token = tm.create_token(user_id, permissions, expires_in_hours=1)

        assert isinstance(token, str)
        assert "." in token  # Should have payload.signature format

        # Verify token
        is_valid, payload = tm.verify_token(token)

        assert is_valid is True
        assert payload is not None
        assert payload["user_id"] == user_id
        assert payload["permissions"] == permissions

    def test_token_expiry(self):
        """Test token expiry functionality."""
        tm = TokenManager()

        # Create token with very short expiry
        token = tm.create_token("test_user", ["read"], expires_in_hours=-1)  # Already expired

        is_valid, payload = tm.verify_token(token)

        assert is_valid is False
        assert payload is None

    def test_token_signature_verification(self):
        """Test token signature verification."""
        tm = TokenManager()

        token = tm.create_token("test_user", ["read"])

        # Tamper with token
        parts = token.split(".")
        tampered_token = parts[0] + ".invalid_signature"

        is_valid, payload = tm.verify_token(tampered_token)

        assert is_valid is False
        assert payload is None


class TestMFAManager:
    """Test multi-factor authentication functionality."""

    def test_mfa_manager_initialization(self):
        """Test MFA manager initialization."""
        mfa = MFAManager()

        assert mfa.otp_window == 30
        assert mfa.otp_digits == 6

    def test_secret_generation(self):  # pragma: allowlist secret
        """Test MFA secret generation."""
        mfa = MFAManager()

        secret1 = mfa.generate_secret()  # pragma: allowlist secret
        secret2 = mfa.generate_secret()  # pragma: allowlist secret

        assert secret1 != secret2  # pragma: allowlist secret
        assert len(secret1) > 0
        assert len(secret2) > 0

    def test_otp_generation_and_verification(self):
        """Test OTP generation and verification."""
        mfa = MFAManager()
        secret = mfa.generate_secret()  # pragma: allowlist secret
        timestamp = int(time.time())

        # Generate OTP
        otp = mfa.generate_otp(secret, timestamp)  # pragma: allowlist secret

        assert len(otp) == 6
        assert otp.isdigit()

        # Verify OTP
        is_valid = mfa.verify_otp(secret, otp, timestamp)  # pragma: allowlist secret
        assert is_valid is True

        # Invalid OTP should fail
        is_valid = mfa.verify_otp(secret, "000000", timestamp)  # pragma: allowlist secret
        assert is_valid is False

    def test_otp_time_window(self):
        """Test OTP time window tolerance."""
        mfa = MFAManager()
        secret = mfa.generate_secret()  # pragma: allowlist secret

        base_time = int(time.time())

        # Generate OTP for base time
        otp = mfa.generate_otp(secret, base_time)  # pragma: allowlist secret

        # Should be valid within time window
        assert mfa.verify_otp(secret, otp, base_time) is True
        assert mfa.verify_otp(secret, otp, base_time - 30) is True  # Previous window
        assert mfa.verify_otp(secret, otp, base_time + 30) is True  # Next window

        # Should fail outside time window
        assert mfa.verify_otp(secret, otp, base_time - 90) is False


class TestAuthenticationManager:
    """Test authentication manager functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()

        self.config = AuthConfig(
            password_min_length=8, max_failed_attempts=3, lockout_duration_minutes=5
        )  # pragma: allowlist secret

        self.auth_manager = AuthenticationManager(config=self.config, db_path=self.temp_db.name)

    def teardown_method(self):
        """Cleanup after each test."""
        os.unlink(self.temp_db.name)

    def test_auth_manager_initialization(self):
        """Test authentication manager initialization."""
        assert self.auth_manager.config == self.config
        assert isinstance(self.auth_manager.password_manager, PasswordManager)
        assert isinstance(self.auth_manager.token_manager, TokenManager)
        assert isinstance(self.auth_manager.mfa_manager, MFAManager)

        # Check database tables exist
        with self.auth_manager._get_db() as conn:
            cursor = conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN ('users', 'api_keys', 'audit_logs')
            """
            )
            tables = [row[0] for row in cursor.fetchall()]

        assert "users" in tables
        assert "api_keys" in tables
        assert "audit_logs" in tables

    def test_user_creation(self):
        """Test user creation."""
        user = self.auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="test_user_password_123!",  # pragma: allowlist secret
            role=UserRole.DEVELOPER,
            security_level=SecurityLevel.CONFIDENTIAL,
        )

        assert isinstance(user, User)
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == UserRole.DEVELOPER
        assert user.security_level == SecurityLevel.CONFIDENTIAL
        assert user.enabled is True

        # Verify user is in database
        with self.auth_manager._get_db() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM users WHERE username = ?
            """,
                ("testuser",),
            )

            user_row = cursor.fetchone()
            assert user_row is not None
            assert user_row["username"] == "testuser"

    def test_user_creation_duplicate_username(self):
        """Test user creation with duplicate username."""
        # Create first user
        self.auth_manager.create_user(
            username="duplicate",
            email="first@example.com",
            password="test_duplicate_first_123!",  # pragma: allowlist secret
        )

        # Try to create user with same username
        with pytest.raises(ValueError):
            self.auth_manager.create_user(
                username="duplicate",
                email="second@example.com",
                password="test_duplicate_second_123!",  # pragma: allowlist secret
            )

    def test_user_creation_weak_password(self):  # pragma: allowlist secret
        """Test user creation with weak password."""
        with pytest.raises(ValueError, match="Password validation failed"):  # pragma: allowlist secret
            self.auth_manager.create_user(
                username="weakpass",
                email="weak@example.com",
                password="weak",  # nosec B106 - test password for security testing
            )

    def test_successful_authentication(self):
        """Test successful user authentication."""
        # Create user
        self.auth_manager.create_user(
            username="authuser",
            email="auth@example.com",
            password="test_auth_password_123!",  # pragma: allowlist secret
        )

        # Authenticate
        success, user, session_token = self.auth_manager.authenticate(
            username="authuser",
            password="test_auth_password_123!",  # pragma: allowlist secret
            ip_address="127.0.0.1",
            user_agent="test",
        )

        assert success is True
        assert user is not None
        assert user.username == "authuser"
        assert session_token is not None

    def test_failed_authentication_wrong_password(self):  # pragma: allowlist secret
        """Test failed authentication with wrong password."""
        # Create user
        self.auth_manager.create_user(
            username="authuser",
            email="auth@example.com",
            password="test_auth_password_123!",  # pragma: allowlist secret
        )

        # Try wrong password
        success, user, session_token = self.auth_manager.authenticate(
            username="authuser",
            password="test_wrong_password",  # pragma: allowlist secret
            ip_address="127.0.0.1",
            user_agent="test",
        )

        assert success is False
        assert user is not None  # User found but auth failed
        assert session_token is None

    def test_failed_authentication_nonexistent_user(self):
        """Test failed authentication with nonexistent user."""
        success, user, session_token = self.auth_manager.authenticate(
            username="nonexistent",
            password="test_auth_password_123!",  # pragma: allowlist secret
            ip_address="127.0.0.1",
            user_agent="test",
        )

        assert success is False
        assert user is None
        assert session_token is None

    def test_account_lockout(self):
        """Test account lockout after multiple failed attempts."""
        # Create user
        self.auth_manager.create_user(
            username="locktest",
            email="lock@example.com",
            password="test_auth_password_123!",  # pragma: allowlist secret
        )  # pragma: allowlist secret

        # Fail authentication multiple times
        for i in range(self.config.max_failed_attempts):
            success, user, _ = self.auth_manager.authenticate(
                username="locktest", password="WrongPassword", ip_address="127.0.0.1"  # pragma: allowlist secret
            )
            assert success is False

        # Account should now be locked
        success, user, _ = self.auth_manager.authenticate(
            username="locktest",
            password="test_auth_password_123!",  # Correct password  # pragma: allowlist secret
            ip_address="127.0.0.1",
        )

        assert success is False  # Should fail due to lockout

    def test_session_validation(self):
        """Test session validation."""
        # Create user and authenticate
        self.auth_manager.create_user(
            username="sessionuser",
            email="session@example.com",
            password="test_auth_password_123!",  # pragma: allowlist secret
        )

        success, user, session_token = self.auth_manager.authenticate(
            username="sessionuser",
            password="test_auth_password_123!",  # pragma: allowlist secret
            ip_address="127.0.0.1",  # pragma: allowlist secret
        )

        assert success is True
        assert session_token is not None

        # Validate session
        is_valid, session_user = self.auth_manager.validate_session(session_token)

        assert is_valid is True
        assert session_user is not None
        assert session_user.username == "sessionuser"

        # Invalid session
        is_valid, session_user = self.auth_manager.validate_session("invalid_token")

        assert is_valid is False
        assert session_user is None

    def test_api_key_creation_and_authentication(self):
        """Test API key creation and authentication."""
        # Create user
        user = self.auth_manager.create_user(
            username="apiuser", email="api@example.com", password="test_auth_password_123!"  # pragma: allowlist secret
        )

        # Create API key
        api_key, api_key_obj = self.auth_manager.create_api_key(
            user_id=user.user_id,
            name="Test API Key",
            permissions=[Permission.READ, Permission.WRITE],
        )

        assert isinstance(api_key, str)
        assert isinstance(api_key_obj, ApiKey)
        assert api_key_obj.name == "Test API Key"
        assert Permission.READ in api_key_obj.permissions
        assert Permission.WRITE in api_key_obj.permissions

        # Authenticate with API key
        success, auth_user = self.auth_manager.authenticate_api_key(api_key=api_key, ip_address="127.0.0.1")

        assert success is True
        assert auth_user is not None
        assert auth_user.user_id == user.user_id

    def test_api_key_revocation(self):
        """Test API key revocation."""
        # Create user and API key
        user = self.auth_manager.create_user(
            username="revokeuser",
            email="revoke@example.com",
            password="test_auth_password_123!",  # pragma: allowlist secret
        )

        api_key, api_key_obj = self.auth_manager.create_api_key(
            user_id=user.user_id, name="Test Key", permissions=[Permission.READ]
        )

        # Revoke API key
        success = self.auth_manager.revoke_api_key(key_id=api_key_obj.key_id, user_id=user.user_id)

        assert success is True

        # Authentication should fail with revoked key
        success, auth_user = self.auth_manager.authenticate_api_key(api_key=api_key, ip_address="127.0.0.1")

        assert success is False
        assert auth_user is None

    def test_mfa_enable_disable(self):
        """Test MFA enable/disable functionality."""
        # Create user
        user = self.auth_manager.create_user(
            username="mfauser", email="mfa@example.com", password="test_auth_password_123!"  # pragma: allowlist secret
        )

        # Enable MFA
        secret = self.auth_manager.enable_mfa(user.user_id)  # pragma: allowlist secret

        assert isinstance(secret, str)
        assert len(secret) > 0

        # Verify MFA is enabled in database
        with self.auth_manager._get_db() as conn:
            cursor = conn.execute(
                """
                SELECT mfa_enabled, mfa_secret FROM users WHERE user_id = ?  # pragma: allowlist secret
            """,
                (user.user_id,),
            )

            row = cursor.fetchone()
            assert row["mfa_enabled"] == 1
            assert row["mfa_secret"] == secret  # pragma: allowlist secret

        # Disable MFA
        self.auth_manager.disable_mfa(user.user_id)

        # Verify MFA is disabled
        with self.auth_manager._get_db() as conn:
            cursor = conn.execute(
                """
                SELECT mfa_enabled, mfa_secret FROM users WHERE user_id = ?  # pragma: allowlist secret
            """,
                (user.user_id,),
            )

            row = cursor.fetchone()
            assert row["mfa_enabled"] == 0
            assert row["mfa_secret"] is None

    def test_mfa_authentication(self):
        """Test authentication with MFA."""
        # Create user and enable MFA
        user = self.auth_manager.create_user(
            username="mfaauth",
            email="mfaauth@example.com",
            password="test_auth_password_123!",  # pragma: allowlist secret
        )

        secret = self.auth_manager.enable_mfa(user.user_id)  # pragma: allowlist secret

        # Generate OTP
        otp = self.auth_manager.mfa_manager.generate_otp(secret)  # pragma: allowlist secret

        # Authenticate with MFA
        success, auth_user, session_token = self.auth_manager.authenticate(
            username="mfaauth",
            password="test_auth_password_123!",  # pragma: allowlist secret
            mfa_code=otp,
            ip_address="127.0.0.1",
        )

        assert success is True
        assert auth_user is not None
        assert session_token is not None

        # Authentication without MFA should fail
        success, auth_user, response = self.auth_manager.authenticate(
            username="mfaauth", password="test_auth_password_123!", ip_address="127.0.0.1"  # nosec B106 - test password
        )

        assert success is False
        assert response == "mfa_required"

    def test_audit_logging(self):
        """Test audit logging functionality."""
        # Create user
        user = self.auth_manager.create_user(
            username="audituser",
            email="audit@example.com",
            password="test_auth_password_123!",  # pragma: allowlist secret
        )

        # Authenticate to generate logs
        self.auth_manager.authenticate(
            username="audituser",
            password="test_auth_password_123!",  # pragma: allowlist secret
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0",
        )

        # Get audit logs
        logs = self.auth_manager.get_audit_logs(user_id=user.user_id)

        assert len(logs) > 0

        # Should have user creation and login logs
        actions = [log.action for log in logs]
        assert "user_created" in actions
        assert "login_success" in actions

        # Check log details
        login_log = next(log for log in logs if log.action == "login_success")
        assert login_log.user_id == user.user_id
        assert login_log.ip_address == "192.168.1.100"
        assert login_log.user_agent == "Mozilla/5.0"
        assert login_log.success is True


class TestAuthorizationManager:
    """Test authorization manager functionality."""

    def test_authorization_manager_initialization(self):
        """Test authorization manager initialization."""
        auth_mgr = AuthorizationManager()

        assert len(auth_mgr.role_permissions) > 0
        assert UserRole.ADMIN in auth_mgr.role_permissions
        assert UserRole.VIEWER in auth_mgr.role_permissions

    def test_admin_permissions(self):
        """Test admin role permissions."""
        auth_mgr = AuthorizationManager()

        admin_user = User(
            user_id="admin",
            username="admin",
            email="admin@example.com",
            password_hash="hash",  # pragma: allowlist secret
            role=UserRole.ADMIN,
            security_level=SecurityLevel.TOP_SECRET,  # pragma: allowlist secret
        )

        # Admin should have all permissions
        for permission in Permission:
            assert auth_mgr.has_permission(admin_user, permission) is True

    def test_viewer_permissions(self):
        """Test viewer role permissions."""
        auth_mgr = AuthorizationManager()

        viewer_user = User(
            user_id="viewer",
            username="viewer",
            email="viewer@example.com",
            password_hash="hash",  # pragma: allowlist secret
            role=UserRole.VIEWER,
            security_level=SecurityLevel.INTERNAL,
        )

        # Viewer should only have read permission
        assert auth_mgr.has_permission(viewer_user, Permission.READ) is True
        assert auth_mgr.has_permission(viewer_user, Permission.WRITE) is False
        assert auth_mgr.has_permission(viewer_user, Permission.DELETE) is False
        assert auth_mgr.has_permission(viewer_user, Permission.ADMIN) is False

    def test_security_level_access_control(self):
        """Test security level-based access control."""
        auth_mgr = AuthorizationManager()

        # User with INTERNAL security level
        internal_user = User(
            user_id="internal",
            username="internal",
            email="internal@example.com",
            password_hash="hash",  # pragma: allowlist secret
            role=UserRole.DEVELOPER,
            security_level=SecurityLevel.INTERNAL,
        )

        # Should have access to INTERNAL and PUBLIC resources
        assert auth_mgr.check_access(internal_user, "resource", Permission.READ, SecurityLevel.INTERNAL) is True

        assert auth_mgr.check_access(internal_user, "resource", Permission.READ, SecurityLevel.PUBLIC) is True

        # Should NOT have access to CONFIDENTIAL resources
        assert auth_mgr.check_access(internal_user, "resource", Permission.READ, SecurityLevel.CONFIDENTIAL) is False

    def test_disabled_user_access_denial(self):
        """Test that disabled users are denied access."""
        auth_mgr = AuthorizationManager()

        disabled_user = User(
            user_id="disabled",
            username="disabled",
            email="disabled@example.com",
            password_hash="hash",  # pragma: allowlist secret
            role=UserRole.ADMIN,
            security_level=SecurityLevel.TOP_SECRET,  # pragma: allowlist secret
            enabled=False,
        )

        # Even admin users should be denied if disabled
        assert auth_mgr.check_access(disabled_user, "resource", Permission.READ, SecurityLevel.PUBLIC) is False


class TestIntegrationScenarios:
    """Test integration scenarios."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()

        self.auth_manager = AuthenticationManager(db_path=self.temp_db.name)
        self.authz_manager = AuthorizationManager()

    def teardown_method(self):
        """Cleanup after each test."""
        os.unlink(self.temp_db.name)

    def test_complete_user_workflow(self):
        """Test complete user workflow."""
        # 1. Create user
        user = self.auth_manager.create_user(
            username="workflow_user",
            email="workflow@example.com",
            password="test_secure_workflow_123!",  # pragma: allowlist secret
            role=UserRole.DEVELOPER,
            security_level=SecurityLevel.CONFIDENTIAL,
        )

        # 2. Enable MFA
        mfa_secret = self.auth_manager.enable_mfa(user.user_id)  # pragma: allowlist secret

        # 3. Create API key
        api_key, api_key_obj = self.auth_manager.create_api_key(
            user_id=user.user_id,
            name="Development Key",
            permissions=[Permission.READ, Permission.WRITE, Permission.EXECUTE],
        )

        # 4. Authenticate with password + MFA
        otp = self.auth_manager.mfa_manager.generate_otp(mfa_secret)  # pragma: allowlist secret
        success, auth_user, session_token = self.auth_manager.authenticate(
            username="workflow_user",
            password="test_secure_workflow_123!",  # pragma: allowlist secret
            mfa_code=otp,
            ip_address="10.0.0.1",
            user_agent="WorkflowTest/1.0",
        )

        assert success is True
        assert session_token is not None

        # 5. Validate session
        is_valid, session_user = self.auth_manager.validate_session(session_token)
        assert is_valid is True
        assert session_user.user_id == user.user_id

        # 6. Test authorization
        has_read = self.authz_manager.has_permission(session_user, Permission.READ)
        has_admin = self.authz_manager.has_permission(session_user, Permission.ADMIN)

        assert has_read is True  # Developer should have read
        assert has_admin is False  # Developer should not have admin

        # 7. Authenticate with API key
        api_success, api_user = self.auth_manager.authenticate_api_key(api_key=api_key, ip_address="10.0.0.1")

        assert api_success is True
        assert api_user.user_id == user.user_id

        # 8. Check audit logs
        logs = self.auth_manager.get_audit_logs(user_id=user.user_id)
        assert len(logs) >= 4  # Creation, MFA enable, login, API auth

    def test_security_breach_simulation(self):
        """Test system behavior during simulated security breach."""
        # Create victim user
        victim = self.auth_manager.create_user(
            username="victim",
            email="victim@example.com",
            password="test_victim_password_123!",  # pragma: allowlist secret
            role=UserRole.OPERATOR,
        )

        # Simulate brute force attack
        for i in range(10):
            success, _, _ = self.auth_manager.authenticate(
                username="victim",
                password=f"wrong_password_{i}",  # pragma: allowlist secret
                ip_address="192.168.1.100",
                user_agent="AttackBot/1.0",
            )
            assert success is False

        # Account should be locked after max attempts
        # Even correct password should fail
        success, user, _ = self.auth_manager.authenticate(
            username="victim",
            password="test_victim_password_123!",  # pragma: allowlist secret
            ip_address="192.168.1.100",  # pragma: allowlist secret
        )
        assert success is False

        # Check audit logs for attack pattern
        logs = self.auth_manager.get_audit_logs(user_id=victim.user_id)
        failed_logins = [log for log in logs if log.action == "login_failed"]

        assert len(failed_logins) >= 10

        # All failed attempts should be from same IP
        attacker_ips = set(log.ip_address for log in failed_logins)
        assert "192.168.1.100" in attacker_ips


if __name__ == "__main__":
    # Run authentication system validation
    print("=== Testing Authentication & Access Control System ===")

    # Test password management
    print("Testing password management...")
    pm = PasswordManager()  # pragma: allowlist secret
    password = "test_auth_password_123!"  # pragma: allowlist secret
    hash_result = pm.hash_password(password)  # pragma: allowlist secret
    verification = pm.verify_password(password, hash_result)  # pragma: allowlist secret
    print(f"OK Password hashing and verification: {verification}")  # pragma: allowlist secret

    # Test token management
    print("Testing token management...")
    tm = TokenManager()
    token = tm.create_token("test_user", ["read", "write"])
    is_valid, payload = tm.verify_token(token)
    print(f"OK Token creation and verification: {is_valid}")

    # Test MFA
    print("Testing multi-factor authentication...")
    mfa = MFAManager()
    secret = mfa.generate_secret()  # pragma: allowlist secret
    otp = mfa.generate_otp(secret)  # pragma: allowlist secret
    mfa_valid = mfa.verify_otp(secret, otp)  # pragma: allowlist secret
    print(f"OK MFA generation and verification: {mfa_valid}")

    # Test authentication manager
    print("Testing authentication manager...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        auth_mgr = AuthenticationManager(db_path=tmp.name)

        # Create and authenticate user
        user = auth_mgr.create_user(
            username="testuser",
            email="test@example.com",
            password="test_auth_password_123!",  # pragma: allowlist secret
        )

        success, auth_user, session = auth_mgr.authenticate(
            username="testuser", password="test_auth_password_123!", ip_address="127.0.0.1"  # pragma: allowlist secret
        )

        print(f"OK User creation and authentication: {success}")

        # Test API key
        api_key, api_obj = auth_mgr.create_api_key(user_id=user.user_id, name="Test Key", permissions=[Permission.READ])

        api_success, api_user = auth_mgr.authenticate_api_key(api_key)
        print(f"OK API key authentication: {api_success}")

        # Cleanup
        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass  # File still in use, will be cleaned up later

    # Test authorization
    print("Testing authorization manager...")
    authz_mgr = AuthorizationManager()

    admin_user = User(
        user_id="admin",
        username="admin",
        email="admin@example.com",
        password_hash="hash",  # pragma: allowlist secret
        role=UserRole.ADMIN,
        security_level=SecurityLevel.TOP_SECRET,  # pragma: allowlist secret
    )

    has_admin_perm = authz_mgr.has_permission(admin_user, Permission.ADMIN)
    has_access = authz_mgr.check_access(admin_user, "resource", Permission.READ, SecurityLevel.CONFIDENTIAL)

    print(f"OK Authorization: admin_perm={has_admin_perm}, access={has_access}")

    print("=== Authentication & access control system validation completed ===")
