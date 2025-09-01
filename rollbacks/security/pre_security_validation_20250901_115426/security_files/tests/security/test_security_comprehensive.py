#!/usr/bin/env python3
"""Comprehensive Security Test Suite.

Tests all security implementations including encryption, authentication,
authorization, compliance, and other CODEX security requirements.
"""

import base64
from datetime import datetime, timedelta
import os
from pathlib import Path

# Import our security modules
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from core.security.digital_twin_encryption import (
    DigitalTwinEncryption,
    DigitalTwinEncryptionError,
    generate_encryption_key,
)
from core.security.p2p_mtls_config import P2PMTLSConfig
from core.security.rbac_system import AccessDeniedException, Permission, RBACSystem, Role
from core.security.secure_api_server import AuthenticationError, InputValidator, JWTAuthenticator, RateLimiter
from core.security.secure_digital_twin_db import SecureDigitalTwinDB
from core.security.secure_file_upload import MaliciousFileError, SecureFileUploadValidator


class TestDigitalTwinEncryption(unittest.TestCase):
    """Test Digital Twin encryption functionality."""

    def setUp(self):
        """Set up test environment."""
        # Generate test encryption key
        self.test_key = generate_encryption_key()

        # Set environment variables
        os.environ["DIGITAL_TWIN_ENCRYPTION_KEY"] = self.test_key
        os.environ["DIGITAL_TWIN_COPPA_COMPLIANT"] = "true"
        os.environ["DIGITAL_TWIN_FERPA_COMPLIANT"] = "true"
        os.environ["DIGITAL_TWIN_GDPR_COMPLIANT"] = "true"

        self.encryption = DigitalTwinEncryption()

    def test_key_validation(self):
        """Test encryption key validation."""
        # Test valid key
        self.assertEqual(len(base64.b64decode(self.test_key)), 32)

        # Test invalid key
        with self.assertRaises(DigitalTwinEncryptionError):
            DigitalTwinEncryption("invalid_key")

    def test_field_encryption_decryption(self):
        """Test field-level encryption and decryption."""
        test_data = "sensitive learning data"
        field_name = "learning_style"

        # Encrypt data
        encrypted = self.encryption.encrypt_sensitive_field(test_data, field_name)
        self.assertIsInstance(encrypted, bytes)
        self.assertNotEqual(encrypted, test_data.encode())

        # Decrypt data
        decrypted = self.encryption.decrypt_sensitive_field(encrypted, field_name)
        self.assertEqual(decrypted, test_data)

    def test_user_id_hashing(self):
        """Test user ID hashing for privacy."""
        user_id = "test_user_123"
        hash1 = self.encryption.hash_user_id(user_id)
        hash2 = self.encryption.hash_user_id(user_id)

        # Hashes should be consistent
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)  # SHA256 hex length
        self.assertNotEqual(hash1, user_id)

    def test_compliance_flags(self):
        """Test compliance flag handling."""
        self.assertTrue(self.encryption.coppa_compliant)
        self.assertTrue(self.encryption.ferpa_compliant)
        self.assertTrue(self.encryption.gdpr_compliant)

    def test_data_retention_compliance(self):
        """Test data retention policy compliance."""
        # Test fresh data
        fresh_date = datetime.utcnow() - timedelta(days=1)
        status = self.encryption.check_data_retention_compliance(fresh_date)
        self.assertFalse(status["is_expired"])

        # Test expired data
        old_date = datetime.utcnow() - timedelta(days=400)
        status = self.encryption.check_data_retention_compliance(old_date)
        self.assertTrue(status["is_expired"])

    def test_profile_encryption(self):
        """Test full profile encryption."""
        profile_data = {
            "user_id": "test_user",
            "learning_style": "visual",
            "knowledge_domains": ["math", "science"],
            "learning_goals": ["improve_algebra"],
            "non_sensitive": "public_data",
        }

        # Encrypt profile
        encrypted_profile = self.encryption.encrypt_profile_data(profile_data)

        # Check that sensitive fields are encrypted
        self.assertIn("learning_style_encrypted", encrypted_profile)
        self.assertNotIn("learning_style", encrypted_profile)
        self.assertIn("non_sensitive", encrypted_profile)  # Non-sensitive remains

        # Decrypt profile
        decrypted_profile = self.encryption.decrypt_profile_data(encrypted_profile)
        self.assertEqual(decrypted_profile["learning_style"], profile_data["learning_style"])


class TestSecureDigitalTwinDB(unittest.TestCase):
    """Test secure database operations."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()

        # Set environment
        os.environ["DIGITAL_TWIN_ENCRYPTION_KEY"] = generate_encryption_key()
        os.environ["DIGITAL_TWIN_DB_PATH"] = self.db_path
        os.environ["DIGITAL_TWIN_COPPA_COMPLIANT"] = "true"
        os.environ["DIGITAL_TWIN_FERPA_COMPLIANT"] = "true"
        os.environ["DIGITAL_TWIN_GDPR_COMPLIANT"] = "true"

        self.secure_db = SecureDigitalTwinDB()

    def tearDown(self):
        """Clean up test environment."""
        os.unlink(self.db_path)

    def test_profile_creation(self):
        """Test encrypted profile creation."""
        profile_data = {
            "user_id": "test_user_123",
            "learning_style": "kinesthetic",
            "knowledge_domains": ["biology", "chemistry"],
            "preferred_difficulty": "advanced",
        }

        profile_id = self.secure_db.create_learning_profile(profile_data)
        self.assertIsNotNone(profile_id)
        self.assertTrue(profile_id.startswith("profile_"))

    def test_profile_retrieval_with_decryption(self):
        """Test profile retrieval and decryption."""
        # Create profile
        profile_data = {
            "user_id": "test_user_456",
            "learning_style": "auditory",
            "knowledge_domains": ["history", "literature"],
        }

        profile_id = self.secure_db.create_learning_profile(profile_data)

        # Retrieve with decryption
        retrieved = self.secure_db.get_learning_profile(profile_id, decrypt=True)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["learning_style"], profile_data["learning_style"])

    def test_gdpr_data_export(self):
        """Test GDPR data export functionality."""
        # Create profile
        profile_data = {"user_id": "gdpr_test_user", "learning_style": "visual"}
        self.secure_db.create_learning_profile(profile_data)

        # Export data
        user_hash = self.secure_db.encryption.hash_user_id("gdpr_test_user")
        export_data = self.secure_db.export_user_data(user_hash)

        self.assertIn("profiles", export_data)
        self.assertIn("audit_log", export_data)
        self.assertIn("compliance_status", export_data)
        self.assertTrue(export_data["compliance_status"]["gdpr_compliant"])

    def test_profile_deletion_gdpr(self):
        """Test GDPR-compliant profile deletion."""
        profile_data = {"user_id": "delete_test_user", "learning_style": "mixed"}
        profile_id = self.secure_db.create_learning_profile(profile_data)

        # Delete profile
        success = self.secure_db.delete_learning_profile(profile_id, "gdpr_request")
        self.assertTrue(success)

        # Verify deletion
        deleted_profile = self.secure_db.get_learning_profile(profile_id)
        self.assertIsNone(deleted_profile)

    def test_compliance_statistics(self):
        """Test compliance statistics generation."""
        # Create some test data
        for i in range(3):
            profile_data = {
                "user_id": f"stats_test_user_{i}",
                "learning_style": "visual",
            }
            self.secure_db.create_learning_profile(profile_data)

        stats = self.secure_db.get_compliance_stats()
        self.assertIn("profiles", stats)
        self.assertIn("encryption", stats)
        self.assertEqual(stats["profiles"]["total_profiles"], 3)


class TestSecureAPIServer(unittest.TestCase):
    """Test secure API server functionality."""

    def setUp(self):
        """Set up test environment."""
        os.environ["API_SECRET_KEY"] = "test_secret_key_32_characters_long"
        self.authenticator = JWTAuthenticator()
        self.rate_limiter = RateLimiter(max_requests=5, window_seconds=10)
        self.validator = InputValidator()

    def test_jwt_token_creation_and_verification(self):
        """Test JWT token creation and verification."""
        user_id = "test_user"
        roles = ["user", "student"]

        # Create token
        token = self.authenticator.create_access_token(user_id, roles)
        self.assertIsInstance(token, str)

        # Verify token
        payload = self.authenticator.verify_token(token)
        self.assertEqual(payload["user_id"], user_id)
        self.assertEqual(payload["roles"], roles)

    def test_token_expiration(self):
        """Test JWT token expiration."""
        # Create short-lived token
        original_expiry = self.authenticator.token_expiry_hours
        self.authenticator.token_expiry_hours = -1  # Expired

        token = self.authenticator.create_access_token("test_user")

        # Restore original expiry
        self.authenticator.token_expiry_hours = original_expiry

        # Verify expired token
        with self.assertRaises(AuthenticationError):
            self.authenticator.verify_token(token)

    def test_password_hashing(self):
        """Test secure password hashing."""
        password = "test_password_123"  # nosec B105 - test password

        # Hash password
        salt, hash_value = self.authenticator.hash_password(password)
        self.assertEqual(len(salt), 64)  # Hex encoded 32 bytes
        self.assertEqual(len(hash_value), 64)  # Hex encoded 32 bytes

        # Verify password
        is_valid = self.authenticator.verify_password(password, salt, hash_value)
        self.assertTrue(is_valid)

        # Verify wrong password
        is_invalid = self.authenticator.verify_password("wrong_password", salt, hash_value)
        self.assertFalse(is_invalid)

    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        client_id = "test_client"

        # Should allow initial requests
        for i in range(5):
            allowed = self.rate_limiter.is_allowed(client_id)
            self.assertTrue(allowed, f"Request {i} should be allowed")

        # Should deny 6th request
        denied = self.rate_limiter.is_allowed(client_id)
        self.assertFalse(denied, "6th request should be denied")

        # Check stats
        stats = self.rate_limiter.get_stats(client_id)
        self.assertEqual(stats["remaining"], 0)

    def test_input_validation(self):
        """Test input validation."""
        schema = {
            "username": {"required": True, "type": str, "min_length": 3},
            "age": {"type": int, "min_value": 0, "max_value": 150},
            "email": {"pattern": r"^[^@]+@[^@]+\.[^@]+$"},
        }

        # Valid data
        valid_data = {"username": "testuser", "age": 25, "email": "test@example.com"}
        validated = self.validator.validate_json(valid_data, schema)
        self.assertEqual(validated["username"], "testuser")

        # Invalid data
        invalid_data = {
            "username": "ab",  # Too short
            "age": 200,  # Too high
            "email": "invalid-email",
        }
        with self.assertRaises(Exception):
            self.validator.validate_json(invalid_data, schema)


class TestRBACSystem(unittest.TestCase):
    """Test Role-Based Access Control system."""

    def setUp(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()

        self.rbac = RBACSystem(self.db_path)

    def tearDown(self):
        """Clean up test environment."""
        os.unlink(self.db_path)

    def test_user_creation_and_role_assignment(self):
        """Test user creation and role assignment."""
        user_id = "test_user_rbac"
        username = "testuser"

        # Create user with student role
        success = self.rbac.create_user(
            user_id=user_id,
            username=username,
            email="test@example.com",
            roles=[Role.STUDENT],
        )
        self.assertTrue(success)

        # Check role assignment
        user_roles = self.rbac.get_user_roles(user_id)
        self.assertIn(Role.STUDENT, user_roles)

    def test_permission_inheritance(self):
        """Test permission inheritance from parent roles."""
        user_id = "inheritance_test_user"

        # Create user with educator role (inherits from user)
        self.rbac.create_user(user_id=user_id, username="educator", roles=[Role.EDUCATOR])

        # Check inherited permissions
        permissions = self.rbac.get_user_permissions(user_id)

        # Should have both educator and inherited user permissions
        self.assertIn(Permission.DIGITAL_TWIN_READ, permissions)  # From USER
        self.assertIn(Permission.DIGITAL_TWIN_WRITE, permissions)  # From EDUCATOR

    def test_permission_checking(self):
        """Test permission checking functionality."""
        user_id = "permission_test_user"

        # Create user with basic permissions
        self.rbac.create_user(user_id=user_id, username="basicuser", roles=[Role.USER])

        # Test permission checks
        has_read = self.rbac.check_permission(user_id, Permission.DIGITAL_TWIN_READ)
        has_admin = self.rbac.check_permission(user_id, Permission.SYSTEM_ADMIN)

        self.assertTrue(has_read)
        self.assertFalse(has_admin)

    def test_access_denial_exception(self):
        """Test access denial exception."""
        user_id = "denial_test_user"

        self.rbac.create_user(user_id=user_id, username="limiteduser", roles=[Role.GUEST])

        # Should raise exception for insufficient permissions
        with self.assertRaises(AccessDeniedException):
            self.rbac.require_permission(user_id, Permission.SYSTEM_ADMIN)

    def test_super_admin_permissions(self):
        """Test super admin gets all permissions."""
        user_id = "super_admin_test"

        self.rbac.create_user(user_id=user_id, username="superadmin", roles=[Role.SUPER_ADMIN])

        permissions = self.rbac.get_user_permissions(user_id)

        # Should have all permissions
        self.assertEqual(len(permissions), len(Permission))
        for permission in Permission:
            self.assertIn(permission, permissions)


class TestP2PMTLSConfig(unittest.TestCase):
    """Test P2P mTLS configuration."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.node_id = "test_node_123"
        self.mtls_config = P2PMTLSConfig(self.node_id, self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_certificate_generation(self):
        """Test certificate generation."""
        # Check that certificates were generated
        self.assertTrue(self.mtls_config.ca_cert_path.exists())
        self.assertTrue(self.mtls_config.ca_key_path.exists())
        self.assertTrue(self.mtls_config.node_cert_path.exists())
        self.assertTrue(self.mtls_config.node_key_path.exists())

    def test_ssl_context_creation(self):
        """Test SSL context creation."""
        # Create server context
        server_ctx = self.mtls_config.create_ssl_context_server()
        self.assertIsNotNone(server_ctx)
        self.assertEqual(server_ctx.minimum_version, server_ctx.maximum_version)

        # Create client context
        client_ctx = self.mtls_config.create_ssl_context_client()
        self.assertIsNotNone(client_ctx)
        self.assertEqual(client_ctx.minimum_version, client_ctx.maximum_version)

    def test_certificate_verification(self):
        """Test peer certificate verification."""
        # Get our own certificate for testing
        cert_der = self.mtls_config.get_node_certificate_der()

        # Verify certificate
        is_valid, node_id = self.mtls_config.verify_peer_certificate(cert_der)
        self.assertTrue(is_valid)
        self.assertEqual(node_id, self.node_id)

    def test_certificate_rotation(self):
        """Test certificate rotation."""
        # Get original certificate info
        original_info = self.mtls_config.get_certificate_info()

        # Rotate certificate
        self.mtls_config.rotate_node_certificate()

        # Check that new certificate was generated
        new_info = self.mtls_config.get_certificate_info()
        self.assertNotEqual(
            original_info["certificate"]["serial_number"],
            new_info["certificate"]["serial_number"],
        )


class TestSecureFileUpload(unittest.TestCase):
    """Test secure file upload validation."""

    def setUp(self):
        """Set up test environment."""
        self.validator = SecureFileUploadValidator()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_safe_file_validation(self):
        """Test validation of safe files."""
        # Create test text file
        test_file = Path(self.temp_dir) / "test.txt"
        test_content = b"This is a safe text file."

        with open(test_file, "wb") as f:
            f.write(test_content)

        # Validate file
        result = self.validator.validate_file(str(test_file), "test.txt")

        self.assertTrue(result["is_safe"])
        self.assertEqual(result["file_size"], len(test_content))
        self.assertIn("sha256", result["metadata"])

    def test_dangerous_file_detection(self):
        """Test detection of dangerous files."""
        # Create file with dangerous content
        dangerous_file = Path(self.temp_dir) / "malicious.txt"
        dangerous_content = b'<script>alert("XSS")</script>'

        with open(dangerous_file, "wb") as f:
            f.write(dangerous_content)

        # Validate file - should detect dangerous content
        with self.assertRaises(MaliciousFileError):
            self.validator.validate_file(str(dangerous_file), "malicious.txt")

    def test_filename_sanitization(self):
        """Test filename sanitization."""
        dangerous_filenames = [
            "../../../etc/passwd",
            'file<>:"*?.txt',
            "con.txt",  # Reserved name
            "file\x00name.txt",  # Null byte
        ]

        for dangerous in dangerous_filenames:
            sanitized = self.validator.sanitize_filename(dangerous)

            # Should not contain dangerous characters
            self.assertNotIn("..", sanitized)
            self.assertNotIn("/", sanitized)
            self.assertNotIn("\x00", sanitized)
            self.assertFalse(sanitized.startswith("."))

    def test_file_size_limits(self):
        """Test file size limit enforcement."""
        # Create oversized file
        large_file = Path(self.temp_dir) / "large.txt"
        large_content = b"A" * (20 * 1024 * 1024)  # 20MB

        with open(large_file, "wb") as f:
            f.write(large_content)

        # Should fail validation due to size
        with self.assertRaises(MaliciousFileError):
            self.validator.validate_file(str(large_file), "large.txt")


class TestSecurityIntegration(unittest.TestCase):
    """Test security system integration."""

    def setUp(self):
        """Set up integrated test environment."""
        # Set up encryption
        os.environ["DIGITAL_TWIN_ENCRYPTION_KEY"] = generate_encryption_key()
        os.environ["API_SECRET_KEY"] = "integration_test_secret_key_32_chars"

        # Create temporary databases
        self.dt_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.rbac_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)

        os.environ["DIGITAL_TWIN_DB_PATH"] = self.dt_db.name
        os.environ["RBAC_DB_PATH"] = self.rbac_db.name

        # Initialize systems
        self.secure_db = SecureDigitalTwinDB()
        self.rbac = RBACSystem(self.rbac_db.name)
        self.authenticator = JWTAuthenticator()

        self.dt_db.close()
        self.rbac_db.close()

    def tearDown(self):
        """Clean up test environment."""
        os.unlink(self.dt_db.name)
        os.unlink(self.rbac_db.name)

    def test_end_to_end_user_workflow(self):
        """Test complete user workflow with security."""
        # 1. Create user with RBAC
        user_id = "integration_test_user"
        username = "integrationuser"

        success = self.rbac.create_user(
            user_id=user_id,
            username=username,
            email="integration@test.com",
            roles=[Role.STUDENT],
        )
        self.assertTrue(success)

        # 2. Check user has required permissions
        has_read = self.rbac.check_permission(user_id, Permission.DIGITAL_TWIN_READ)
        has_write = self.rbac.check_permission(user_id, Permission.DIGITAL_TWIN_WRITE)
        self.assertTrue(has_read)
        self.assertTrue(has_write)

        # 3. Create JWT token for user
        token = self.authenticator.create_access_token(
            user_id=user_id,
            roles=["student"],
            permissions=["digital_twin:read", "digital_twin:write"],
        )
        self.assertIsInstance(token, str)

        # 4. Create encrypted profile
        profile_data = {
            "user_id": user_id,
            "learning_style": "visual",
            "knowledge_domains": ["mathematics"],
        }
        profile_id = self.secure_db.create_learning_profile(profile_data)
        self.assertIsNotNone(profile_id)

        # 5. Retrieve and verify profile
        retrieved = self.secure_db.get_learning_profile(profile_id, decrypt=True)
        self.assertEqual(retrieved["learning_style"], profile_data["learning_style"])

        # 6. Verify audit logs
        access_logs = self.rbac.get_access_log(user_id)
        self.assertGreater(len(access_logs), 0)

    def test_security_failure_scenarios(self):
        """Test security failure scenarios."""
        # 1. Test unauthorized access
        unauthorized_user = "unauthorized_user"
        self.rbac.create_user(
            user_id=unauthorized_user,
            username="unauthorized",
            roles=[Role.GUEST],  # Limited permissions
        )

        with self.assertRaises(AccessDeniedException):
            self.rbac.require_permission(unauthorized_user, Permission.SYSTEM_ADMIN)

        # 2. Test expired token
        expired_token = "invalid.token.here"
        with self.assertRaises(AuthenticationError):
            self.authenticator.verify_token(expired_token)

        # 3. Test compliance violations
        # (This would require specific compliance scenarios)


def run_security_tests():
    """Run all security tests."""
    # Create test suite
    test_classes = [
        TestDigitalTwinEncryption,
        TestSecureDigitalTwinDB,
        TestSecureAPIServer,
        TestRBACSystem,
        TestP2PMTLSConfig,
        TestSecureFileUpload,
        TestSecurityIntegration,
    ]

    suite = unittest.TestSuite()

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return test results
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success_rate": (
            (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun
            if result.testsRun > 0
            else 0
        ),
    }


if __name__ == "__main__":
    # Set up test environment
    print("Starting Comprehensive Security Test Suite...")
    print("=" * 60)

    # Run tests
    results = run_security_tests()

    # Print results
    print("\n" + "=" * 60)
    print("SECURITY TEST RESULTS:")
    print(f"Tests Run: {results['tests_run']}")
    print(f"Failures: {results['failures']}")
    print(f"Errors: {results['errors']}")
    print(f"Success Rate: {results['success_rate']:.1%}")

    if results["success_rate"] >= 0.95:
        print("✅ SECURITY TESTS PASSED")
    else:
        print("❌ SECURITY TESTS FAILED")

    print("=" * 60)
