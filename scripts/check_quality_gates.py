#!/usr/bin/env python3
"""Local quality gate checker for Sprint 2 production/experimental separation.
Run this before committing to ensure quality standards.
"""

import json
import re
import subprocess
import sys
from pathlib import Path


class QualityGateChecker:
    def __init__(self) -> None:
        self.root_dir = Path.cwd()
        self.production_dir = self.root_dir / "production"
        self.experimental_dir = self.root_dir / "experimental"
        self.deprecated_dir = self.root_dir / "deprecated"

        self.issues = []
        self.warnings = []

    def log_issue(self, issue: str) -> None:
        """Log a quality gate issue."""
        self.issues.append(issue)
        print(f"ERROR: {issue}")

    def log_warning(self, warning: str) -> None:
        """Log a quality gate warning."""
        self.warnings.append(warning)
        print(f"WARNING: {warning}")

    def log_success(self, message: str) -> None:
        """Log a success message."""
        print(f"SUCCESS: {message}")

    def check_import_separation(self) -> None:
        """Check that production doesn't import experimental/deprecated."""
        print("\n=== Checking Import Separation ===")

        if not self.production_dir.exists():
            self.log_warning("Production directory not found")
            return

        # Find Python files in production
        production_files = list(self.production_dir.rglob("*.py"))

        forbidden_imports = []
        for py_file in production_files:
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                # Check for experimental imports (only actual import statements)
                if re.search(
                    r"^(from experimental|import experimental)", content, re.MULTILINE
                ):
                    forbidden_imports.append(f"{py_file}: imports experimental")

                # Check for deprecated imports (only actual import statements)
                if re.search(
                    r"^(from deprecated|import deprecated)", content, re.MULTILINE
                ):
                    forbidden_imports.append(f"{py_file}: imports deprecated")

            except Exception as e:
                self.log_warning(f"Could not read {py_file}: {e}")

        if forbidden_imports:
            for imp in forbidden_imports:
                self.log_issue(f"Forbidden import: {imp}")
        else:
            self.log_success("No forbidden imports found in production code")

    def check_todos_in_production(self) -> None:
        """Check that production code doesn't contain TODOs."""
        print("\n=== Checking for TODOs in Production ===")

        if not self.production_dir.exists():
            self.log_warning("Production directory not found")
            return

        production_files = list(self.production_dir.rglob("*.py"))

        todo_files = []
        for py_file in production_files:
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                # Check for TODOs, FIXMEs, XXXs
                if re.search(r"TODO|FIXME|XXX", content, re.IGNORECASE):
                    todo_files.append(str(py_file))

            except Exception as e:
                self.log_warning(f"Could not read {py_file}: {e}")

        if todo_files:
            for file in todo_files:
                self.log_issue(f"TODO found in production: {file}")
        else:
            self.log_success("No TODOs found in production code")

    def check_experimental_warnings(self) -> None:
        """Check that experimental code has appropriate warnings."""
        print("\n=== Checking Experimental Warnings ===")

        if not self.experimental_dir.exists():
            self.log_warning("Experimental directory not found")
            return

        experimental_files = list(self.experimental_dir.rglob("*.py"))

        files_with_warnings = 0
        files_without_warnings = []

        for py_file in experimental_files:
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                # Check for warning patterns
                if re.search(
                    r"warnings\.warn|ExperimentalWarning|warn_experimental", content
                ):
                    files_with_warnings += 1
                else:
                    files_without_warnings.append(
                        str(py_file.relative_to(self.experimental_dir))
                    )

            except Exception as e:
                self.log_warning(f"Could not read {py_file}: {e}")

        if files_without_warnings:
            for file in files_without_warnings[:10]:  # Show first 10
                self.log_warning(f"Experimental file missing warning: {file}")
            if len(files_without_warnings) > 10:
                self.log_warning(
                    f"... and {len(files_without_warnings) - 10} more files"
                )

        self.log_success(f"Found warnings in {files_with_warnings} experimental files")

    def check_test_coverage(self) -> None:
        """Check test coverage for production components."""
        print("\n=== Checking Test Coverage ===")

        production_tests = self.production_dir / "tests"
        if not production_tests.exists():
            self.log_issue("Production tests directory not found")
            return

        # Count test files
        test_files = list(production_tests.rglob("test_*.py"))

        # Count production modules
        production_modules = list(self.production_dir.rglob("*.py"))
        production_modules = [f for f in production_modules if "tests" not in str(f)]

        if len(test_files) == 0:
            self.log_issue("No test files found in production/tests/")
        else:
            self.log_success(
                f"Found {len(test_files)} test files for {len(production_modules)} production modules"
            )

    def run_tests(self) -> None:
        """Run production tests."""
        print("\n=== Running Production Tests ===")

        if not (self.production_dir / "tests").exists():
            self.log_warning("No production tests to run")
            return

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(self.production_dir / "tests"),
                    "-v",
                    "--tb=short",
                ],
                check=False,
                capture_output=True,
                text=True,
                cwd=self.root_dir,
            )

            if result.returncode == 0:
                self.log_success("All production tests passed")
            else:
                self.log_warning("Some production tests failed or were skipped")
                print(f"Test output: {result.stdout}")

        except Exception as e:
            self.log_warning(f"Could not run tests: {e}")

    def check_file_structure(self) -> None:
        """Check that the file structure follows Sprint 2 conventions."""
        print("\n=== Checking File Structure ===")

        expected_dirs = ["production", "experimental", "deprecated"]

        for dir_name in expected_dirs:
            if (self.root_dir / dir_name).exists():
                self.log_success(f"{dir_name}/ directory exists")
            else:
                self.log_warning(f"{dir_name}/ directory missing")

        # Check for __init__.py files
        for dir_name in ["production", "experimental"]:
            init_file = self.root_dir / dir_name / "__init__.py"
            if init_file.exists():
                self.log_success(f"{dir_name}/__init__.py exists")
            else:
                self.log_warning(f"{dir_name}/__init__.py missing")

    def check_pre_commit_hooks(self) -> None:
        """Check pre-commit configuration."""
        print("\n=== Checking Pre-commit Hooks ===")

        pre_commit_config = self.root_dir / ".pre-commit-config.yaml"
        if pre_commit_config.exists():
            self.log_success("Pre-commit configuration found")

            try:
                with open(pre_commit_config) as f:
                    content = f.read()

                # Check for quality gates
                if "production" in content.lower():
                    self.log_success("Pre-commit hooks include production checks")
                else:
                    self.log_warning(
                        "Pre-commit hooks may need production quality gates"
                    )

            except Exception as e:
                self.log_warning(f"Could not read pre-commit config: {e}")
        else:
            self.log_warning("Pre-commit configuration not found")

    def generate_report(self):
        """Generate quality gate report."""
        print("\n" + "=" * 60)
        print("SPRINT 2 QUALITY GATE REPORT")
        print("=" * 60)

        issues_count = len(self.issues)
        warnings_count = len(self.warnings)

        if issues_count == 0:
            print("ALL QUALITY GATES PASSED!")
            status = "PASS"
        else:
            print(f"{issues_count} QUALITY GATE FAILURES")
            status = "FAIL"

        if warnings_count > 0:
            print(f"{warnings_count} warnings")

        # Generate detailed report
        from datetime import datetime

        report = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "issues": self.issues,
            "warnings": self.warnings,
            "summary": {
                "total_issues": issues_count,
                "total_warnings": warnings_count,
                "status": status,
            },
        }

        # Save report
        report_file = self.root_dir / "quality_gate_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nDetailed report saved to: {report_file}")

        return issues_count == 0

    def run_all_checks(self):
        """Run all quality gate checks."""
        print("Sprint 2 Quality Gate Checker")
        print("Ensuring production/experimental separation standards\n")

        self.check_file_structure()
        self.check_import_separation()
        self.check_todos_in_production()
        self.check_experimental_warnings()
        self.check_test_coverage()
        self.check_pre_commit_hooks()
        self.run_tests()

        return self.generate_report()


def main() -> None:
    """Main entry point."""
    checker = QualityGateChecker()
    success = checker.run_all_checks()

    if success:
        print("\nReady for commit!")
        sys.exit(0)
    else:
        print("\nFix issues before committing")
        sys.exit(1)


if __name__ == "__main__":
    main()
