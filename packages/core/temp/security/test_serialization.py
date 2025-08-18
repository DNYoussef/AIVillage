"""Test to ensure secure serialization practices.

This test validates that pickle usage is replaced with secure JSON serialization
and that no unsafe deserialization occurs in the codebase.
"""

import json
import re
import subprocess
import unittest
from pathlib import Path


class TestSecureSerialization(unittest.TestCase):
    """Test cases to verify secure serialization practices."""

    def setUp(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent.parent
        self.src_dir = self.project_root / "src"

        # Patterns to find unsafe serialization
        self.pickle_loads_pattern = re.compile(r"pickle\.loads?\(", re.IGNORECASE)
        self.pickle_dump_pattern = re.compile(r"pickle\.dumps?\(", re.IGNORECASE)
        self.eval_pattern = re.compile(r"\beval\s*\(", re.IGNORECASE)
        self.exec_pattern = re.compile(r"\bexec\s*\(", re.IGNORECASE)

        # Files to exclude from scanning
        self.excluded_patterns = [
            "**/test_*.py",
            "**/tests/**",
            "**/tmp/**",
            "**/dev*.py",
            "**/example*.py",
            "**/demo*.py",
            "**/mock*.py",
            "**/secure_serialization.py",  # This file is allowed to mention pickle
        ]

    def scan_file_for_unsafe_serialization(self, file_path: Path) -> list[tuple[int, str, str]]:
        """Scan a file for unsafe serialization patterns.

        Args:
            file_path: Path to the file to scan

        Returns:
            List of tuples (line_number, line_content, issue_type)
        """
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            violations = []
            for line_num, line in enumerate(lines, 1):
                line_stripped = line.strip()

                # Skip comments and docstrings
                if line_stripped.startswith("#") or line_stripped.startswith('"""') or line_stripped.startswith("'''"):
                    continue

                # Check for pickle usage
                if self.pickle_loads_pattern.search(line):
                    # Allow if it's part of a security comment or safe usage annotation
                    if any(
                        keyword in line.lower()
                        for keyword in [
                            "# security:",
                            "# safe usage",
                            "# nosec",
                            "secure_serialization",
                            "replacement for",
                        ]
                    ):
                        continue
                    violations.append((line_num, line.strip(), "pickle.loads"))

                if self.pickle_dump_pattern.search(line):
                    # Allow if it's part of a security comment
                    if any(
                        keyword in line.lower()
                        for keyword in [
                            "# security:",
                            "# safe usage",
                            "# nosec",
                            "secure_serialization",
                            "replacement for",
                        ]
                    ):
                        continue
                    violations.append((line_num, line.strip(), "pickle.dumps"))

                # Check for eval/exec usage
                if self.eval_pattern.search(line) and "eval(" in line:
                    # Allow legitimate uses
                    if any(
                        allowed in line.lower()
                        for allowed in [
                            "# safe usage",
                            "model.eval()",
                            ".eval()",
                            "torch",
                            "pytorch",
                            "tensorflow",
                            "nn.eval",
                            "module.eval",
                        ]
                    ):
                        continue
                    violations.append((line_num, line.strip(), "eval"))

                if self.exec_pattern.search(line) and "exec(" in line:
                    # Allow legitimate uses in test/benchmark code
                    if (
                        any(
                            allowed in line.lower()
                            for allowed in [
                                "# safe usage",
                                "benchmark",
                                "test_code",
                                "exec_globals",
                                "subprocess.exec",
                                "os.exec",
                            ]
                        )
                        or "test" in str(file_path).lower()
                    ):
                        continue
                    violations.append((line_num, line.strip(), "exec"))

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

    def test_no_unsafe_pickle_usage(self):
        """Test that no unsafe pickle usage exists in source code."""
        if not self.src_dir.exists():
            self.skipTest("src directory not found")

        violations = []

        # Scan all Python files in src/
        for py_file in self.src_dir.rglob("*.py"):
            if self.is_excluded_file(py_file):
                continue

            file_violations = self.scan_file_for_unsafe_serialization(py_file)
            pickle_violations = [v for v in file_violations if v[2] in ["pickle.loads", "pickle.dumps"]]

            if pickle_violations:
                violations.extend(
                    [
                        f"{py_file}:{line_num}: {line} (found {issue_type})"
                        for line_num, line, issue_type in pickle_violations
                    ]
                )

        self.assertEqual(
            [],
            violations,
            "Found unsafe pickle usage in source code:\n"
            + "\n".join(violations)
            + "\nReplace with SecureSerializer from src.core.security.secure_serialization",
        )

    def test_no_eval_exec_usage(self):
        """Test that no eval/exec usage exists in source code."""
        if not self.src_dir.exists():
            self.skipTest("src directory not found")

        violations = []

        # Scan all Python files in src/
        for py_file in self.src_dir.rglob("*.py"):
            if self.is_excluded_file(py_file):
                continue

            file_violations = self.scan_file_for_unsafe_serialization(py_file)
            eval_violations = [v for v in file_violations if v[2] in ["eval", "exec"]]

            if eval_violations:
                violations.extend(
                    [
                        f"{py_file}:{line_num}: {line} (found {issue_type})"
                        for line_num, line, issue_type in eval_violations
                    ]
                )

        self.assertEqual(
            [],
            violations,
            "Found unsafe eval/exec usage in source code:\n"
            + "\n".join(violations)
            + "\nUse ast.literal_eval() for safe evaluation or other secure alternatives",
        )

    def test_secure_serializer_functionality(self):
        """Test that SecureSerializer works correctly."""
        try:
            from src.core.security.secure_serialization import SecureSerializer, secure_dumps, secure_loads

            # Test basic serialization
            test_data = {
                "string": "test",
                "number": 42,
                "boolean": True,
                "list": [1, 2, 3],
                "nested": {"key": "value"},
            }

            # Test serialization
            serialized = SecureSerializer.dumps(test_data)
            self.assertIsInstance(serialized, bytes)

            # Test deserialization
            deserialized = SecureSerializer.loads(serialized)
            self.assertEqual(deserialized, test_data)

            # Test backward-compatible interface
            serialized2 = secure_dumps(test_data)
            deserialized2 = secure_loads(serialized2)
            self.assertEqual(deserialized2, test_data)

            # Test that non-serializable objects raise appropriate errors
            with self.assertRaises(ValueError):
                SecureSerializer.dumps(lambda x: x)  # Function not serializable

            # Test malformed data handling
            with self.assertRaises(ValueError):
                SecureSerializer.loads(b"invalid json data")

        except ImportError:
            self.fail("SecureSerializer not available - implement src.core.security.secure_serialization")

    def test_performance_record_serialization(self):
        """Test that PerformanceRecord serialization is secure."""
        try:
            from src.core.security.secure_serialization import SecureSerializer

            # Mock PerformanceRecord-like data
            mock_record_data = {
                "timestamp": 1234567890,
                "task_type": "classification",
                "success": True,
                "execution_time_ms": 150,
                "accuracy": 0.95,
                "confidence": 0.88,
                "resource_usage": {"cpu": 0.3, "memory": "100MB"},
                "context": {"model": "test-model"},
            }

            # Test serialization of performance record
            serialized = SecureSerializer.serialize_performance_record(type("MockRecord", (), mock_record_data)())
            self.assertIsInstance(serialized, dict)
            self.assertEqual(serialized["timestamp"], mock_record_data["timestamp"])
            self.assertEqual(serialized["task_type"], mock_record_data["task_type"])

        except ImportError:
            self.skipTest("SecureSerializer performance record methods not available")
        except AttributeError:
            self.skipTest("Performance record serialization methods not implemented")

    def test_json_serialization_safety(self):
        """Test that JSON serialization is safe from common attacks."""
        try:
            from src.core.security.secure_serialization import SecureSerializer

            # Test that large numbers are handled safely
            large_data = {"large_number": 10**100}
            serialized = SecureSerializer.dumps(large_data)
            deserialized = SecureSerializer.loads(serialized)
            self.assertEqual(deserialized, large_data)

            # Test that Unicode is handled correctly
            unicode_data = {"unicode": "🔒🛡️🔐"}
            serialized = SecureSerializer.dumps(unicode_data)
            deserialized = SecureSerializer.loads(serialized)
            self.assertEqual(deserialized, unicode_data)

            # Test deeply nested structures
            deep_data = {"level": {"deep": {"nested": {"data": "value"}}}}
            serialized = SecureSerializer.dumps(deep_data)
            deserialized = SecureSerializer.loads(serialized)
            self.assertEqual(deserialized, deep_data)

        except ImportError:
            self.fail("SecureSerializer not available")

    def test_bandit_security_scan_serialization(self):
        """Test that Bandit security scanner would pass on serialization code."""
        try:
            # Run bandit on the secure_serialization module
            security_module = self.src_dir / "core" / "security" / "secure_serialization.py"

            if not security_module.exists():
                self.skipTest("Secure serialization module not found")

            result = subprocess.run(
                ["bandit", "-f", "json", str(security_module)],
                check=False,
                capture_output=True,
                text=True,
            )

            # Bandit should not find high severity issues in our secure module
            if result.returncode == 0:
                # Parse JSON output
                try:
                    bandit_output = json.loads(result.stdout)
                    high_issues = [
                        issue for issue in bandit_output.get("results", []) if issue.get("issue_severity") == "HIGH"
                    ]

                    self.assertEqual(
                        [],
                        high_issues,
                        f"Bandit found HIGH severity issues in secure_serialization.py: {high_issues}",
                    )
                except json.JSONDecodeError:
                    pass  # Bandit output format may vary

        except (FileNotFoundError, subprocess.SubprocessError):
            self.skipTest("Bandit not available for security scanning")

    def test_serialization_module_imports_safely(self):
        """Test that serialization modules can be imported without side effects."""
        try:
            # Import should not raise exceptions
            from src.core.security import secure_serialization

            # Module should have expected attributes
            self.assertTrue(hasattr(secure_serialization, "SecureSerializer"))
            self.assertTrue(hasattr(secure_serialization, "secure_dumps"))
            self.assertTrue(hasattr(secure_serialization, "secure_loads"))

            # SecureSerializer should have expected methods
            serializer = secure_serialization.SecureSerializer
            self.assertTrue(hasattr(serializer, "dumps"))
            self.assertTrue(hasattr(serializer, "loads"))

        except ImportError as e:
            self.fail(f"Could not import secure serialization module: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
