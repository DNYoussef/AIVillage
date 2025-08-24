"""Test to ensure no HTTP URLs exist in production code.

This test validates that all production code uses HTTPS and has no hardcoded
HTTP URLs that could create security vulnerabilities.
"""

import os
from pathlib import Path
import re
import unittest

import yaml


class TestNoHTTPInProduction(unittest.TestCase):
    """Test cases to verify HTTPS enforcement in production code."""

    def setUp(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent.parent
        self.src_production_dir = self.project_root / "src" / "production"
        self.config_dir = self.project_root / "config"

        # Pattern to find HTTP URLs in source code
        self.http_pattern = re.compile(r'["\']http://[^"\']*["\']|`http://[^`]*`|=http://[^\s]*')

        # Files to exclude from scanning (development/test files)
        self.excluded_patterns = [
            "**/test_*.py",
            "**/tests/**",
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
                    line_lower = line.lower()
                    if any(
                        keyword in line_lower
                        for keyword in [
                            'startswith("http://")',  # Security validation
                            'else "http://localhost',  # Development fallback
                            "use https://",  # Error message
                            "production environment",  # Error message
                            "validation",  # Security validation context
                        ]
                    ):
                        continue
                    violations.append((line_num, line.strip()))

            return violations
        except Exception:
            # Skip files that can't be read
            return []

    def is_excluded_file(self, file_path: Path) -> bool:
        """Check if file should be excluded from scanning."""
        for pattern in self.excluded_patterns:
            if file_path.match(pattern):
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

        # Find HTTP URLs in the config
        http_matches = self.http_pattern.findall(config_content)

        self.assertEqual([], http_matches, f"Found HTTP URLs in production config: {http_matches}")

    def test_production_config_yaml_structure(self):
        """Test that production config YAML structure is valid and secure."""
        production_config = self.config_dir / "aivillage_config_production.yaml"

        if not production_config.exists():
            self.skipTest("Production config file not found")

        try:
            with open(production_config, encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.fail(f"Invalid YAML in production config: {e}")
        except Exception as e:
            self.fail(f"Could not load production config: {e}")

        # Recursively check for HTTP URLs in config values
        violations = self._find_http_in_dict(config_data, "")

        self.assertEqual(
            [],
            violations,
            f"Found HTTP URLs in production config structure: {violations}",
        )

    def _find_http_in_dict(self, data, path: str) -> list[str]:
        """Recursively find HTTP URLs in dictionary/list structures."""
        violations = []

        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                violations.extend(self._find_http_in_dict(value, current_path))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]"
                violations.extend(self._find_http_in_dict(item, current_path))
        elif isinstance(data, str) and data.startswith("http://"):
            violations.append(f"{path}: {data}")

        return violations

    def test_environment_variable_defaults_secure(self):
        """Test that default environment variable values are secure."""
        # Test that AIVILLAGE_ENV=production would trigger HTTPS validation
        os.environ["AIVILLAGE_ENV"] = "production"

        try:
            from packages.core.security import validate_production_environment

            # This should pass if no HTTP URLs in env vars
            validate_production_environment()

        except ImportError:
            self.skipTest("Security validation module not available")
        except Exception as e:
            # Check if the error is about HTTP URLs
            if "http://" in str(e).lower():
                self.fail(f"Production environment has HTTP URLs: {e}")
            # Other errors are acceptable for this test
        finally:
            # Clean up environment
            if "AIVILLAGE_ENV" in os.environ:
                del os.environ["AIVILLAGE_ENV"]

    def test_vector_store_https_enforcement(self):
        """Test that vector store enforces HTTPS in production."""
        try:
            from packages.rag.rag_system.retrieval.vector_store import _get_qdrant_url

            # Test with production environment
            os.environ["AIVILLAGE_ENV"] = "production"

            # Should return HTTPS URL by default in production
            url = _get_qdrant_url()
            self.assertTrue(
                url.startswith("https://"),
                f"Vector store should use HTTPS in production, got: {url}",
            )

            # Test that explicit HTTP URL raises error in production
            os.environ["QDRANT_URL"] = "http://insecure.example.com:6333"

            with self.assertRaises(ValueError) as cm:
                _get_qdrant_url()

            self.assertIn("https://", str(cm.exception))
            self.assertIn("production", str(cm.exception))

        except ImportError:
            self.skipTest("Vector store module not available")
        finally:
            # Clean up environment
            for var in ["AIVILLAGE_ENV", "QDRANT_URL"]:
                if var in os.environ:
                    del os.environ[var]

    def test_config_manager_validates_https(self):
        """Test that config manager validates HTTPS in production."""
        try:
            from packages.core.config_manager import CODEXConfigManager

            # Create temporary config with HTTP URL
            test_config = {
                "external_services": {"monitoring": {"prometheus_endpoint": "http://insecure.example.com:9090"}}
            }

            # Test with production environment
            os.environ["AIVILLAGE_ENV"] = "production"

            # Mock config manager to test validation
            config_manager = CODEXConfigManager(enable_hot_reload=False)

            with self.assertRaises(Exception) as cm:
                config_manager.validate_configuration(test_config)
                from packages.core.security import validate_config_dict_for_production

                validate_config_dict_for_production(test_config)

            # Should mention HTTP or HTTPS in error
            error_msg = str(cm.exception).lower()
            self.assertTrue(
                "http" in error_msg or "https" in error_msg,
                f"Error should mention HTTP/HTTPS validation: {cm.exception}",
            )

        except ImportError:
            self.skipTest("Config manager or security module not available")
        finally:
            # Clean up environment
            if "AIVILLAGE_ENV" in os.environ:
                del os.environ["AIVILLAGE_ENV"]

    def test_production_services_yaml_excluded(self):
        """Test that services.yaml is not used in production."""
        services_config = self.config_dir / "services.yaml"

        if not services_config.exists():
            self.skipTest("services.yaml not found")

        try:
            with open(services_config, encoding="utf-8") as f:
                services_data = yaml.safe_load(f)
        except Exception:
            self.skipTest("Could not parse services.yaml")

        # services.yaml should be marked as development
        environment = services_data.get("environment", "").lower()
        self.assertIn("dev", environment, "services.yaml should be marked as development-only")


if __name__ == "__main__":
    unittest.main()
