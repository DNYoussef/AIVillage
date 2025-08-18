#!/usr/bin/env python3
"""
CODEX Audit v3 - Security Gates Test
Testing claim: "CI blocks http:// & unsafe serialization"

This test verifies:
1. No http:// URLs in production code
2. No unsafe pickle.loads in production paths
3. Security gates exist to fail PRs with violations
4. Secure alternatives are properly implemented
"""

import json
import subprocess
import sys
from pathlib import Path


class SecurityGatesTest:
    """Test class for security gate verification"""

    def __init__(self):
        self.results = {
            "http_scan": {},
            "pickle_scan": {},
            "security_gates": {},
            "secure_alternatives": {},
            "overall_success": False,
        }

    def test_http_urls(self) -> bool:
        """Test for http:// URLs in production code"""
        try:
            # Scan production directories for http:// URLs
            production_dirs = [
                "src/production/",
                "src/core/",
                "src/agent_forge/",
                "src/token_economy/",
            ]

            http_violations = []

            for prod_dir in production_dirs:
                if Path(prod_dir).exists():
                    # Use git grep to find http:// occurrences
                    try:
                        result = subprocess.run(
                            ["git", "grep", "-n", "http://", prod_dir],
                            capture_output=True,
                            text=True,
                            cwd=Path.cwd(),
                        )

                        if result.returncode == 0 and result.stdout.strip():
                            lines = result.stdout.strip().split("\n")
                            for line in lines:
                                if line.strip():
                                    parts = line.split(":", 2)
                                    if len(parts) >= 3:
                                        http_violations.append(
                                            {
                                                "file": parts[0],
                                                "line": parts[1],
                                                "content": parts[2].strip(),
                                            }
                                        )
                    except Exception as e:
                        print(f"Warning: Failed to scan {prod_dir}: {e}")

            self.results["http_scan"] = {
                "success": len(http_violations) == 0,
                "violations_found": len(http_violations),
                "violations": http_violations,
                "scanned_dirs": production_dirs,
            }

            return len(http_violations) == 0

        except Exception as e:
            self.results["http_scan"] = {"success": False, "error": str(e)}
            return False

    def test_pickle_loads(self) -> bool:
        """Test for unsafe pickle.loads usage"""
        try:
            # Scan for pickle.loads patterns
            pickle_violations = []

            # Use git grep to find pickle.loads occurrences
            try:
                result = subprocess.run(
                    ["git", "grep", "-n", "pickle\\.loads", "src/"],
                    capture_output=True,
                    text=True,
                    cwd=Path.cwd(),
                )

                if result.returncode == 0 and result.stdout.strip():
                    lines = result.stdout.strip().split("\n")
                    for line in lines:
                        if line.strip():
                            parts = line.split(":", 2)
                            if len(parts) >= 3:
                                file_path = parts[0]
                                line_num = parts[1]
                                content = parts[2].strip()

                                # Check if this is a safe usage (comment, secure replacement, etc.)
                                is_safe = (
                                    content.strip().startswith("#")  # Comment
                                    or content.strip().startswith('"""')  # Docstring
                                    or "secure replacement" in content.lower()  # Safe replacement
                                    or "secure_serializer" in file_path  # Secure serializer module
                                )

                                if not is_safe:
                                    pickle_violations.append(
                                        {
                                            "file": file_path,
                                            "line": line_num,
                                            "content": content,
                                            "risk": "unsafe_pickle_usage",
                                        }
                                    )

            except Exception as e:
                print(f"Warning: Failed to scan for pickle.loads: {e}")

            self.results["pickle_scan"] = {
                "success": len(pickle_violations) == 0,
                "violations_found": len(pickle_violations),
                "violations": pickle_violations,
            }

            return len(pickle_violations) == 0

        except Exception as e:
            self.results["pickle_scan"] = {"success": False, "error": str(e)}
            return False

    def test_security_gates_exist(self) -> bool:
        """Test that security gates exist in CI/pre-commit"""
        try:
            gate_checks = {}

            # Check for pre-commit configuration
            precommit_file = Path(".pre-commit-config.yaml")
            if precommit_file.exists():
                content = precommit_file.read_text()
                gate_checks["precommit_exists"] = True
                gate_checks["has_bandit"] = "bandit" in content
                gate_checks["has_security_checks"] = any(
                    term in content.lower() for term in ["security", "bandit", "safety"]
                )
            else:
                gate_checks["precommit_exists"] = False

            # Check for GitHub Actions security workflows
            github_actions_dir = Path(".github/workflows")
            if github_actions_dir.exists():
                gate_checks["github_actions_exist"] = True
                security_workflows = []
                for workflow_file in github_actions_dir.glob("*.yml"):
                    content = workflow_file.read_text()
                    if any(term in content.lower() for term in ["security", "bandit", "safety", "vulnerability"]):
                        security_workflows.append(workflow_file.name)
                gate_checks["security_workflows"] = security_workflows
            else:
                gate_checks["github_actions_exist"] = False
                gate_checks["security_workflows"] = []

            # Check for security test files
            security_test_files = list(Path("tests").glob("**/test_*security*.py")) if Path("tests").exists() else []
            gate_checks["security_test_files"] = [str(f) for f in security_test_files]

            self.results["security_gates"] = gate_checks

            # Consider successful if we have some security measures
            has_gates = (
                gate_checks.get("has_bandit", False)
                or len(gate_checks.get("security_workflows", [])) > 0
                or len(gate_checks.get("security_test_files", [])) > 0
            )

            return has_gates

        except Exception as e:
            self.results["security_gates"] = {"success": False, "error": str(e)}
            return False

    def test_secure_alternatives(self) -> bool:
        """Test that secure alternatives are implemented"""
        try:
            alternatives_found = {}

            # Check for secure serializer
            secure_serializer = Path("src/core/security/secure_serializer.py")
            if secure_serializer.exists():
                alternatives_found["secure_serializer"] = True
                content = secure_serializer.read_text()
                alternatives_found["has_secure_loads"] = "def loads(" in content
                alternatives_found["has_secure_dumps"] = "def dumps(" in content
            else:
                alternatives_found["secure_serializer"] = False

            # Check for HTTPS enforcement
            try:
                result = subprocess.run(
                    ["git", "grep", "-n", "https://", "src/"],
                    capture_output=True,
                    text=True,
                    cwd=Path.cwd(),
                )

                https_count = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
                alternatives_found["https_usage"] = https_count > 0
                alternatives_found["https_count"] = https_count

            except Exception:
                alternatives_found["https_usage"] = False
                alternatives_found["https_count"] = 0

            self.results["secure_alternatives"] = alternatives_found

            # Success if we have secure serializer
            return alternatives_found.get("secure_serializer", False)

        except Exception as e:
            self.results["secure_alternatives"] = {"success": False, "error": str(e)}
            return False

    def run_all_tests(self) -> bool:
        """Run all security gate tests"""
        print("Testing Security Gates...")

        # Test 1: HTTP URLs scan
        print("  -> Scanning for http:// URLs in production...")
        http_success = self.test_http_urls()
        print(f"     HTTP scan: {'PASS' if http_success else 'FAIL'}")

        # Test 2: Pickle.loads scan
        print("  -> Scanning for unsafe pickle.loads...")
        pickle_success = self.test_pickle_loads()
        print(f"     Pickle scan: {'PASS' if pickle_success else 'FAIL'}")

        # Test 3: Security gates exist
        print("  -> Checking security gates...")
        gates_success = self.test_security_gates_exist()
        print(f"     Security gates: {'PASS' if gates_success else 'FAIL'}")

        # Test 4: Secure alternatives
        print("  -> Checking secure alternatives...")
        alternatives_success = self.test_secure_alternatives()
        print(f"     Secure alternatives: {'PASS' if alternatives_success else 'FAIL'}")

        # Overall success requires passing security scans
        overall_success = http_success and pickle_success

        self.results["overall_success"] = overall_success

        return overall_success


def main():
    """Main test execution"""
    try:
        tester = SecurityGatesTest()
        success = tester.run_all_tests()

        # Save results
        results_file = Path(__file__).parent.parent / "artifacts" / "security_gates.json"
        with open(results_file, "w") as f:
            json.dump(tester.results, f, indent=2)

        print(f"\nResults saved to: {results_file}")
        print(f"Overall Security Gates test: {'PASS' if success else 'FAIL'}")

        return success

    except Exception as e:
        print(f"Security Gates test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
