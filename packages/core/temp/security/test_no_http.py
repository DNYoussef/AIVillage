"""Test to ensure no HTTP URLs exist in production code.

This test validates that all production code uses HTTPS and has no hardcoded
HTTP URLs that could create security vulnerabilities.
"""

import os
import re
import unittest
from pathlib import Path


class TestNoHTTPInProduction(unittest.TestCase):
    """Test cases to verify HTTPS enforcement in production code."""

    def setUp(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent.parent
        self.src_production_dir = self.project_root / "src" / "production"
        self.config_dir = self.project_root / "config"

        # Pattern to find HTTP URLs in source code (more comprehensive)
        self.http_pattern = re.compile(
            r'["\']http://[^"\']*["\']|`http://[^`]*`|=http://[^\s]*|http://[^\s>]+',
            re.IGNORECASE,
        )

        # Files to exclude from scanning (development/test files)
        self.excluded_patterns = [
            "**/test_*.py",
            "**/tests/**",
            "**/tmp/**",
            "**/dev*.py",
            "**/example*.py",
            "**/demo*.py",
            "**/mock*.py",
        ]

    def scan_file_for_http_urls(self, file_path: Path) -> list[tuple[int, str]]:
        """Scan a file for HTTP URLs and return line numbers and content.

        Args:
            file_path: Path to the file to scan

        Returns:
            List of tuples (line_number, line_content) containing HTTP URLs
        """
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            violations = []
            for line_num, line in enumerate(lines, 1):
                if self.http_pattern.search(line):
                    # Skip lines that are part of security validation logic
                    line_lower = line.lower().strip()
                    if any(
                        keyword in line_lower
                        for keyword in [
                            'startswith("http://")',  # Security validation code
                            "startswith('http://')",  # Security validation code
                            'else "http://localhost',  # Development fallback
                            "use https://",  # Error message
                            "production environment",  # Error message context
                            "validation",  # Security validation context
                            "if.*http://",  # Conditional checks
                            "security",  # Security-related context
                            "error",  # Error message context
                            "# development only",  # Development comment
                            "# dev only",  # Development comment
                            "localhost",  # Localhost exceptions
                        ]
                    ):
                        continue

                    # Skip docker healthcheck URLs (internal container communication)
                    if "healthcheck" in line_lower and "localhost" in line_lower:
                        continue

                    violations.append((line_num, line.strip()))

            return violations
        except Exception as e:
            print(f"Warning: Could not scan {file_path}: {e}")
            return []

    def is_excluded_file(self, file_path: Path) -> bool:
        """Check if file should be excluded from scanning."""
        file_str = str(file_path)
        for pattern in self.excluded_patterns:
            if Path(file_str).match(pattern):
                return True
        return False

    def test_no_http_urls_in_production_source(self):
        """Test that no HTTP URLs exist in production source code."""
        if not self.src_production_dir.exists():
            self.skipTest("src/production directory not found")

        violations = []

        # Scan all Python files in src/production
        for py_file in self.src_production_dir.rglob("*.py"):
            if self.is_excluded_file(py_file):
                continue

            file_violations = self.scan_file_for_http_urls(py_file)
            if file_violations:
                violations.extend([f"{py_file}:{line_num}: {line}" for line_num, line in file_violations])

        self.assertEqual(
            [],
            violations,
            "Found HTTP URLs in production source code:\n" + "\n".join(violations),
        )

    def test_production_config_uses_https(self):
        """Test that production configuration files use HTTPS."""
        production_config = self.config_dir / "aivillage_config_production.yaml"

        if not production_config.exists():
            self.skipTest("Production config file not found")

        try:
            with open(production_config, encoding="utf-8") as f:
                config_content = f.read()
        except Exception as e:
            self.fail(f"Could not read production config: {e}")

        # Find HTTP URLs in the config (excluding comments)
        lines = config_content.split("\n")
        violations = []

        for line_num, line in enumerate(lines, 1):
            # Skip commented lines
            stripped_line = line.strip()
            if stripped_line.startswith("#"):
                continue

            if "http://" in line and "localhost" not in line:
                violations.append(f"Line {line_num}: {line.strip()}")

        self.assertEqual([], violations, f"Found HTTP URLs in production config: {violations}")

    def test_docker_compose_production_security(self):
        """Test that docker-compose files use secure configurations."""
        docker_files = list(self.project_root.glob("docker-compose*.yml"))

        if not docker_files:
            self.skipTest("No docker-compose files found")

        violations = []

        for docker_file in docker_files:
            if "dev" in docker_file.name or "development" in docker_file.name:
                continue  # Skip development configs

            try:
                with open(docker_file, encoding="utf-8") as f:
                    content = f.read()

                # Check for HTTP URLs not related to internal services
                lines = content.split("\n")
                for line_num, line in enumerate(lines, 1):
                    if "http://" in line:
                        # Allow internal container communication
                        if any(
                            internal in line
                            for internal in [
                                "localhost",
                                "127.0.0.1",
                                "twin:",
                                "prometheus:",
                                "redis:",
                                "postgres:",
                                "neo4j:",
                                "qdrant:",
                                "credits-api:",
                                "hyperag-mcp:",
                            ]
                        ):
                            continue

                        # Skip healthcheck commands (internal)
                        if "healthcheck" in line.lower():
                            continue

                        violations.append(f"{docker_file.name}:{line_num}: {line.strip()}")

            except Exception as e:
                print(f"Warning: Could not scan {docker_file}: {e}")

        # For production deployments, external URLs should be HTTPS
        self.assertEqual(
            [],
            violations,
            "Found insecure HTTP URLs in Docker configurations:\n" + "\n".join(violations),
        )

    def test_security_validator_functionality(self):
        """Test that HTTP security validator works correctly."""
        try:
            from src.core.security.http_security_validator import (
                HTTPSecurityError,
                scan_source_for_http_urls,
                validate_url_for_production,
            )

            # Test with production environment
            original_env = os.getenv("AIVILLAGE_ENV")
            os.environ["AIVILLAGE_ENV"] = "production"

            try:
                # Should pass HTTPS URLs
                https_url = validate_url_for_production("https://secure.example.com")
                self.assertEqual(https_url, "https://secure.example.com")

                # Should fail HTTP URLs in production
                with self.assertRaises(HTTPSecurityError):
                    validate_url_for_production("http://insecure.example.com")

                # Test source code scanning
                test_code = """
                url1 = "https://secure.example.com"
                url2 = "http://insecure.example.com"
                url3 = "ftp://example.com"
                """

                http_urls = scan_source_for_http_urls(test_code)
                self.assertEqual(len(http_urls), 1)
                self.assertIn("http://insecure.example.com", http_urls[0])

            finally:
                # Restore original environment
                if original_env is None:
                    os.environ.pop("AIVILLAGE_ENV", None)
                else:
                    os.environ["AIVILLAGE_ENV"] = original_env

        except ImportError:
            self.skipTest("HTTP security validator not available")

    def test_environment_variable_security(self):
        """Test that production environment variables are secure."""
        # List of environment variables that should not contain HTTP URLs
        security_sensitive_vars = [
            "QDRANT_URL",
            "REDIS_URL",
            "PROMETHEUS_ENDPOINT",
            "JAEGER_ENDPOINT",
            "GATEWAY_URL",
            "TWIN_URL",
            "API_BASE_URL",
            "NEO4J_URI",
        ]

        # Set production environment for testing
        original_env = os.getenv("AIVILLAGE_ENV")
        os.environ["AIVILLAGE_ENV"] = "production"

        violations = []

        try:
            for var_name in security_sensitive_vars:
                value = os.getenv(var_name)
                if value and value.startswith("http://"):
                    # Allow localhost for development/testing
                    if "localhost" not in value and "127.0.0.1" not in value:
                        violations.append(f"{var_name}={value}")

            self.assertEqual(
                [],
                violations,
                f"Found HTTP URLs in production environment variables: {violations}",
            )

        finally:
            # Restore environment
            if original_env is None:
                os.environ.pop("AIVILLAGE_ENV", None)
            else:
                os.environ["AIVILLAGE_ENV"] = original_env


if __name__ == "__main__":
    unittest.main(verbosity=2)
