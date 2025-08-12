#!/usr/bin/env python3
"""Standalone Security Test Suite.

Tests core security implementations independently without system dependencies.
"""

import base64
import hashlib
import os
import sqlite3
import tempfile
import time
import unittest


# Test the encryption key generation
def generate_encryption_key() -> str:
    """Generate a new 32-byte base64-encoded encryption key."""
    key_bytes = os.urandom(32)
    return base64.b64encode(key_bytes).decode("utf-8")


class TestEncryptionKeyGeneration(unittest.TestCase):
    """Test encryption key generation and validation."""

    def test_key_generation(self):
        """Test that encryption keys are properly generated."""
        key = generate_encryption_key()

        # Key should be base64 encoded
        self.assertIsInstance(key, str)

        # Decoded key should be exactly 32 bytes
        decoded = base64.b64decode(key)
        self.assertEqual(len(decoded), 32)

        # Two generated keys should be different
        key2 = generate_encryption_key()
        self.assertNotEqual(key, key2)

    def test_key_validation(self):
        """Test encryption key validation logic."""
        # Valid key
        valid_key = generate_encryption_key()
        try:
            decoded = base64.b64decode(valid_key)
            self.assertEqual(len(decoded), 32)
            validation_passed = True
        except Exception:
            validation_passed = False

        self.assertTrue(validation_passed)

        # Invalid key (wrong length)
        invalid_key = base64.b64encode(b"short").decode("utf-8")
        try:
            decoded = base64.b64decode(invalid_key)
            validation_passed = len(decoded) == 32
        except Exception:
            validation_passed = False

        self.assertFalse(validation_passed)


class TestPasswordHashing(unittest.TestCase):
    """Test secure password hashing."""

    def hash_password(self, password: str):
        """Hash password with salt using PBKDF2."""
        salt = os.urandom(32)
        password_hash = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, 100000
        )
        return salt.hex(), password_hash.hex()

    def verify_password(self, password: str, salt_hex: str, hash_hex: str) -> bool:
        """Verify password against stored hash."""
        try:
            salt = bytes.fromhex(salt_hex)
            stored_hash = bytes.fromhex(hash_hex)
            password_hash = hashlib.pbkdf2_hmac(
                "sha256", password.encode("utf-8"), salt, 100000
            )

            # Constant time comparison
            import hmac

            return hmac.compare_digest(password_hash, stored_hash)
        except Exception:
            return False

    def test_password_hashing(self):
        """Test password hashing functionality."""
        password = "test_password_123!@#"

        # Hash password
        salt, hash_value = self.hash_password(password)

        # Verify salt and hash lengths
        self.assertEqual(len(salt), 64)  # 32 bytes in hex
        self.assertEqual(len(hash_value), 64)  # 32 bytes in hex

        # Verify correct password
        is_valid = self.verify_password(password, salt, hash_value)
        self.assertTrue(is_valid)

        # Verify incorrect password
        is_invalid = self.verify_password("wrong_password", salt, hash_value)
        self.assertFalse(is_invalid)

    def test_password_hash_uniqueness(self):
        """Test that same password produces different hashes with different salts."""
        password = "same_password"

        salt1, hash1 = self.hash_password(password)
        salt2, hash2 = self.hash_password(password)

        # Salts should be different
        self.assertNotEqual(salt1, salt2)
        # Hashes should be different due to different salts
        self.assertNotEqual(hash1, hash2)

        # But both should verify correctly
        self.assertTrue(self.verify_password(password, salt1, hash1))
        self.assertTrue(self.verify_password(password, salt2, hash2))


class TestInputValidation(unittest.TestCase):
    """Test input validation and sanitization."""

    def validate_json_field(self, value, field_rules):
        """Basic JSON field validation."""
        # Check required
        if field_rules.get("required", False) and value is None:
            raise ValueError("Field is required")

        if value is not None:
            # Type check
            expected_type = field_rules.get("type")
            if expected_type and not isinstance(value, expected_type):
                raise ValueError(f"Invalid type: expected {expected_type.__name__}")

            # String length check
            if isinstance(value, str):
                min_len = field_rules.get("min_length", 0)
                max_len = field_rules.get("max_length", 10000)
                if not (min_len <= len(value) <= max_len):
                    raise ValueError(f"Invalid length: must be {min_len}-{max_len}")

            # Numeric range check
            if isinstance(value, (int, float)):
                min_val = field_rules.get("min_value")
                max_val = field_rules.get("max_value")
                if min_val is not None and value < min_val:
                    raise ValueError(f"Value too small: minimum {min_val}")
                if max_val is not None and value > max_val:
                    raise ValueError(f"Value too large: maximum {max_val}")

        return True

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        import re

        # Remove path components
        filename = os.path.basename(filename)

        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)

        # Limit length
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = name[: 200 - len(ext)] + ext

        # Ensure it doesn't start with dot or dash
        if filename.startswith(".") or filename.startswith("-"):
            filename = "file_" + filename

        return filename

    def test_json_validation(self):
        """Test JSON field validation."""
        # Valid values
        self.assertTrue(
            self.validate_json_field("test", {"type": str, "min_length": 3})
        )
        self.assertTrue(
            self.validate_json_field(
                25, {"type": int, "min_value": 0, "max_value": 100}
            )
        )

        # Invalid values
        with self.assertRaises(ValueError):
            self.validate_json_field("ab", {"type": str, "min_length": 3})

        with self.assertRaises(ValueError):
            self.validate_json_field(150, {"type": int, "max_value": 100})

        with self.assertRaises(ValueError):
            self.validate_json_field(None, {"required": True})

    def test_filename_sanitization(self):
        """Test filename sanitization."""
        dangerous_filenames = {
            "../../../etc/passwd": "etc_passwd",
            'file<>:"|?*.txt': "file_______.txt",
            "con.txt": "file_con.txt",
            "file\x00name.txt": "file_name.txt",
            ".hidden": "file_.hidden",
            "-dash": "file_-dash",
        }

        for dangerous, expected_safe in dangerous_filenames.items():
            sanitized = self.sanitize_filename(dangerous)

            # Should not contain dangerous patterns
            self.assertNotIn("..", sanitized)
            self.assertNotIn("/", sanitized)
            self.assertNotIn("\x00", sanitized)
            self.assertNotIn("<", sanitized)
            self.assertNotIn(">", sanitized)


class TestRateLimiting(unittest.TestCase):
    """Test rate limiting functionality."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_limits = {}  # client_id -> list of timestamps

    def is_rate_limited(
        self, client_id: str, max_requests: int = 5, window_seconds: int = 60
    ) -> bool:
        """Check if client is rate limited."""
        now = time.time()

        # Initialize client if not exists
        if client_id not in self.rate_limits:
            self.rate_limits[client_id] = []

        # Clean old requests
        self.rate_limits[client_id] = [
            timestamp
            for timestamp in self.rate_limits[client_id]
            if now - timestamp < window_seconds
        ]

        # Check if over limit
        if len(self.rate_limits[client_id]) >= max_requests:
            return True

        # Add current request
        self.rate_limits[client_id].append(now)
        return False

    def test_rate_limiting_basic(self):
        """Test basic rate limiting functionality."""
        client_id = "test_client"

        # First 5 requests should pass
        for i in range(5):
            is_limited = self.is_rate_limited(
                client_id, max_requests=5, window_seconds=10
            )
            self.assertFalse(is_limited, f"Request {i + 1} should not be rate limited")

        # 6th request should be rate limited
        is_limited = self.is_rate_limited(client_id, max_requests=5, window_seconds=10)
        self.assertTrue(is_limited, "6th request should be rate limited")

    def test_rate_limiting_window_reset(self):
        """Test that rate limiting resets after window expires."""
        client_id = "window_test_client"

        # Fill up the rate limit with very short window
        for i in range(3):
            is_limited = self.is_rate_limited(
                client_id, max_requests=3, window_seconds=1
            )
            self.assertFalse(is_limited)

        # Should be limited now
        is_limited = self.is_rate_limited(client_id, max_requests=3, window_seconds=1)
        self.assertTrue(is_limited)

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again
        is_limited = self.is_rate_limited(client_id, max_requests=3, window_seconds=1)
        self.assertFalse(is_limited)


class TestSecurityHeaders(unittest.TestCase):
    """Test security headers and configurations."""

    def get_security_headers(self):
        """Get recommended security headers."""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        }

    def test_security_headers_present(self):
        """Test that all required security headers are defined."""
        headers = self.get_security_headers()

        required_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Content-Security-Policy",
        ]

        for header in required_headers:
            self.assertIn(header, headers)
            self.assertIsInstance(headers[header], str)
            self.assertGreater(len(headers[header]), 0)

    def test_csp_policy(self):
        """Test Content Security Policy configuration."""
        headers = self.get_security_headers()
        csp = headers["Content-Security-Policy"]

        # Should restrict to self by default
        self.assertIn("default-src 'self'", csp)


class TestComplianceFlags(unittest.TestCase):
    """Test compliance flag handling."""

    def check_compliance_requirements(self, user_age: int, region: str = "US"):
        """Check compliance requirements based on user profile."""
        requirements = {
            "coppa_required": user_age < 13 and region == "US",
            "gdpr_required": region in ["EU", "UK"],
            "ferpa_applicable": True,  # Always applicable for educational records
            "data_retention_days": 365 if user_age >= 18 else 180,  # Shorter for minors
        }
        return requirements

    def test_coppa_compliance(self):
        """Test COPPA compliance requirements."""
        # Minor in US - COPPA required
        requirements = self.check_compliance_requirements(12, "US")
        self.assertTrue(requirements["coppa_required"])
        self.assertEqual(requirements["data_retention_days"], 180)

        # Adult in US - COPPA not required
        requirements = self.check_compliance_requirements(25, "US")
        self.assertFalse(requirements["coppa_required"])
        self.assertEqual(requirements["data_retention_days"], 365)

        # Minor outside US - COPPA not required
        requirements = self.check_compliance_requirements(12, "Canada")
        self.assertFalse(requirements["coppa_required"])

    def test_gdpr_compliance(self):
        """Test GDPR compliance requirements."""
        # EU user - GDPR required
        requirements = self.check_compliance_requirements(25, "EU")
        self.assertTrue(requirements["gdpr_required"])

        # UK user - GDPR required
        requirements = self.check_compliance_requirements(25, "UK")
        self.assertTrue(requirements["gdpr_required"])

        # US user - GDPR not required
        requirements = self.check_compliance_requirements(25, "US")
        self.assertFalse(requirements["gdpr_required"])

    def test_ferpa_compliance(self):
        """Test FERPA compliance (always applicable for educational records)."""
        requirements = self.check_compliance_requirements(20, "US")
        self.assertTrue(requirements["ferpa_applicable"])


class TestDatabaseSecurity(unittest.TestCase):
    """Test database security configurations."""

    def create_secure_connection(self, db_path: str):
        """Create secure SQLite connection with proper settings."""
        conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)

        # Security and performance settings
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA temp_store=MEMORY")

        return conn

    def test_parameterized_queries(self):
        """Test parameterized queries to prevent SQL injection."""
        with tempfile.NamedTemporaryFile(suffix=".db") as temp_db:
            conn = self.create_secure_connection(temp_db.name)

            # Create test table
            conn.execute(
                """
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL,
                    email TEXT
                )
            """
            )

            # Test parameterized insert (safe)
            username = "test_user"
            email = "test@example.com"
            conn.execute(
                "INSERT INTO users (username, email) VALUES (?, ?)", (username, email)
            )

            # Test parameterized select (safe)
            cursor = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()

            self.assertIsNotNone(result)
            self.assertEqual(result[1], username)
            self.assertEqual(result[2], email)

            conn.close()

    def test_database_permissions(self):
        """Test database file permissions."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
            db_path = temp_db.name

        try:
            # Create database
            conn = self.create_secure_connection(db_path)
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.close()

            # Check file exists
            self.assertTrue(os.path.exists(db_path))

            # Check file permissions (on Unix-like systems)
            if hasattr(os, "stat"):
                stat_info = os.stat(db_path)
                # File should be readable/writable by owner
                self.assertTrue(stat_info.st_mode & 0o600)

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


def run_security_tests():
    """Run all standalone security tests."""
    test_classes = [
        TestEncryptionKeyGeneration,
        TestPasswordHashing,
        TestInputValidation,
        TestRateLimiting,
        TestSecurityHeaders,
        TestComplianceFlags,
        TestDatabaseSecurity,
    ]

    suite = unittest.TestSuite()

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(getattr(result, "skipped", [])),
        "success_rate": (
            (result.testsRun - len(result.failures) - len(result.errors))
            / result.testsRun
            if result.testsRun > 0
            else 0
        ),
    }


if __name__ == "__main__":
    print("Starting Standalone Security Test Suite...")
    print("=" * 60)

    # Run tests
    results = run_security_tests()

    # Print results
    print("\n" + "=" * 60)
    print("STANDALONE SECURITY TEST RESULTS:")
    print(f"Tests Run: {results['tests_run']}")
    print(f"Failures: {results['failures']}")
    print(f"Errors: {results['errors']}")
    print(f"Skipped: {results['skipped']}")
    print(f"Success Rate: {results['success_rate']:.1%}")

    if results["success_rate"] >= 0.95:
        print("✅ SECURITY TESTS PASSED")
        exit_code = 0
    else:
        print("❌ SECURITY TESTS FAILED")
        exit_code = 1

    print("=" * 60)
    exit(exit_code)
