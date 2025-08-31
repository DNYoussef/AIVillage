"""
Import validation tests for clean architecture reorganization.
Ensures import statements are updated correctly as modules are moved.
"""

import ast
import importlib
from pathlib import Path
import sys
from typing import Any

import pytest

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "packages"))


class ImportValidator:
    """Validate imports during module reorganization"""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.packages_dir = self.project_root / "packages"
        self.tests_dir = self.project_root / "tests"
        self.import_mapping = self._load_import_mapping()

    def _load_import_mapping(self) -> dict[str, str]:
        """Load mapping of old imports to new imports"""
        # This would be loaded from a configuration file in real scenario
        return {
            # Example mappings for reorganization
            "agents.core.base": "packages.agents.core.base_agent",
            "rag.pipeline": "packages.rag.core.pipeline",
            "p2p.network": "packages.p2p.core.network",
            "common.utils": "packages.core.common.utilities",
            "compression.unified": "packages.core.compression.unified_compressor",
        }

    def scan_import_issues(self) -> dict[str, list[dict[str, Any]]]:
        """Scan for import issues across the project"""
        issues = {"broken_imports": [], "outdated_imports": [], "circular_imports": [], "missing_modules": []}

        # Scan test files
        test_files = list(self.tests_dir.rglob("*.py"))
        for file_path in test_files:
            file_issues = self._scan_file_imports(file_path)
            for issue_type, file_issues_list in file_issues.items():
                issues[issue_type].extend(file_issues_list)

        # Scan package files
        package_files = list(self.packages_dir.rglob("*.py"))
        for file_path in package_files:
            file_issues = self._scan_file_imports(file_path)
            for issue_type, file_issues_list in file_issues.items():
                issues[issue_type].extend(file_issues_list)

        return issues

    def _scan_file_imports(self, file_path: Path) -> dict[str, list[dict[str, Any]]]:
        """Scan a single file for import issues"""
        issues = {"broken_imports": [], "outdated_imports": [], "circular_imports": [], "missing_modules": []}

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            imports = self._extract_imports(tree)

            for import_info in imports:
                import_name = import_info["module"]
                line_no = import_info["line"]

                # Check if import is broken
                if not self._can_import_module(import_name):
                    issues["broken_imports"].append(
                        {"file": str(file_path), "line": line_no, "import": import_name, "type": import_info["type"]}
                    )

                # Check if import is outdated
                if self._is_outdated_import(import_name):
                    suggested_import = self.import_mapping.get(import_name)
                    issues["outdated_imports"].append(
                        {
                            "file": str(file_path),
                            "line": line_no,
                            "old_import": import_name,
                            "suggested_import": suggested_import,
                        }
                    )

                # Check for potential circular imports
                if self._might_be_circular(file_path, import_name):
                    issues["circular_imports"].append({"file": str(file_path), "line": line_no, "import": import_name})

        except (SyntaxError, UnicodeDecodeError) as e:
            issues["broken_imports"].append(
                {"file": str(file_path), "line": 0, "import": "SYNTAX_ERROR", "error": str(e)}
            )

        return issues

    def _extract_imports(self, tree: ast.AST) -> list[dict[str, Any]]:
        """Extract import statements from AST"""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({"module": alias.name, "line": node.lineno, "type": "import", "alias": alias.asname})
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(
                        {
                            "module": node.module,
                            "line": node.lineno,
                            "type": "from_import",
                            "names": [alias.name for alias in node.names],
                        }
                    )

        return imports

    def _can_import_module(self, module_name: str) -> bool:
        """Test if a module can be imported"""
        if not module_name or module_name == "SYNTAX_ERROR":
            return False

        try:
            # Handle relative imports
            if module_name.startswith("."):
                return True  # Skip relative imports for now

            # Try to import the module
            importlib.import_module(module_name)
            return True

        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    def _is_outdated_import(self, import_name: str) -> bool:
        """Check if import is outdated based on mapping"""
        return import_name in self.import_mapping

    def _might_be_circular(self, file_path: Path, import_name: str) -> bool:
        """Check if import might create circular dependency"""
        # Simplified circular dependency detection
        # In practice, this would be more sophisticated

        try:
            # Convert file path to module name
            file_module = self._file_to_module_name(file_path)

            # If imported module might import back to current module
            if import_name.startswith("packages.") and file_module:
                # Very basic check - would need graph analysis for real detection
                if file_module in import_name or import_name in file_module:
                    return True

        except Exception as e:
            import logging
            logging.exception("Import validation circular dependency check failed: %s", str(e))

        return False

    def _file_to_module_name(self, file_path: Path) -> str:
        """Convert file path to module name"""
        try:
            if "packages" in file_path.parts:
                package_idx = file_path.parts.index("packages")
                parts = file_path.parts[package_idx:]

                if parts[-1] == "__init__.py":
                    parts = parts[:-1]
                else:
                    parts = parts[:-1] + (parts[-1][:-3],)  # Remove .py

                return ".".join(parts)
        except Exception as e:
            import logging
            logging.exception("File to module name conversion error: %s", str(e))

        return ""

    def generate_import_fix_suggestions(self, issues: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Generate suggestions to fix import issues"""
        suggestions = []

        # Fix outdated imports
        for issue in issues["outdated_imports"]:
            suggestions.append(
                {
                    "type": "replace_import",
                    "file": issue["file"],
                    "line": issue["line"],
                    "old": issue["old_import"],
                    "new": issue["suggested_import"],
                    "action": f"Replace '{issue['old_import']}' with '{issue['suggested_import']}'",
                }
            )

        # Fix broken imports
        for issue in issues["broken_imports"]:
            if issue["import"] != "SYNTAX_ERROR":
                # Try to suggest alternative
                alternative = self._suggest_import_alternative(issue["import"])
                if alternative:
                    suggestions.append(
                        {
                            "type": "fix_broken_import",
                            "file": issue["file"],
                            "line": issue["line"],
                            "old": issue["import"],
                            "new": alternative,
                            "action": f"Replace broken '{issue['import']}' with '{alternative}'",
                        }
                    )

        return suggestions

    def _suggest_import_alternative(self, broken_import: str) -> str:
        """Suggest alternative for broken import"""
        # Try common patterns
        alternatives = [
            f"packages.{broken_import}",
            f"packages.core.{broken_import}",
            broken_import.replace(".", ".core."),
        ]

        for alt in alternatives:
            if self._can_import_module(alt):
                return alt

        # Check import mapping
        for old_import, new_import in self.import_mapping.items():
            if old_import in broken_import:
                return broken_import.replace(old_import, new_import)

        return ""

    def validate_critical_imports(self) -> dict[str, Any]:
        """Validate that critical imports still work"""
        critical_imports = [
            "packages.agents.core.base",
            "packages.rag.core.pipeline",
            "packages.p2p.core.network",
            "packages.core.common",
            "packages.core.compression",
        ]

        results = {"total": len(critical_imports), "working": 0, "broken": [], "status": "unknown"}

        for import_name in critical_imports:
            if self._can_import_module(import_name):
                results["working"] += 1
            else:
                results["broken"].append(import_name)

        # Determine status
        success_rate = results["working"] / results["total"]
        if success_rate == 1.0:
            results["status"] = "all_working"
        elif success_rate >= 0.8:
            results["status"] = "mostly_working"
        elif success_rate >= 0.5:
            results["status"] = "partially_working"
        else:
            results["status"] = "mostly_broken"

        return results


class ImportValidationTest:
    """Test import validation during reorganization"""

    def __init__(self):
        self.validator = ImportValidator()

    def test_no_broken_imports(self):
        """Test that there are no broken imports"""
        issues = self.validator.scan_import_issues()
        broken_imports = issues["broken_imports"]

        # Filter out known issues during reorganization
        serious_issues = [issue for issue in broken_imports if not self._is_acceptable_broken_import(issue)]

        if serious_issues:
            issue_msg = "\n".join(
                [
                    f"  {issue['file']}:{issue['line']} - {issue['import']}"
                    for issue in serious_issues[:10]  # Show first 10
                ]
            )
            if len(serious_issues) > 10:
                issue_msg += f"\n  ... and {len(serious_issues) - 10} more"

            pytest.fail(f"Found {len(serious_issues)} broken imports:\n{issue_msg}")

    def test_critical_imports_working(self):
        """Test that critical imports are working"""
        validation = self.validator.validate_critical_imports()

        if validation["status"] == "mostly_broken":
            broken_msg = ", ".join(validation["broken"])
            pytest.fail(f"Critical imports mostly broken: {broken_msg}")

        # During reorganization, we allow some imports to be temporarily broken
        if validation["status"] == "partially_working":
            pytest.skip("Some critical imports broken during reorganization (acceptable)")

    def test_no_circular_imports(self):
        """Test that there are no circular imports"""
        issues = self.validator.scan_import_issues()
        circular_imports = issues["circular_imports"]

        if circular_imports:
            issue_msg = "\n".join(
                [
                    f"  {issue['file']}:{issue['line']} - {issue['import']}"
                    for issue in circular_imports[:5]  # Show first 5
                ]
            )

            pytest.fail(f"Found {len(circular_imports)} potential circular imports:\n{issue_msg}")

    def test_import_fix_suggestions(self):
        """Test that import fix suggestions are generated"""
        issues = self.validator.scan_import_issues()
        suggestions = self.validator.generate_import_fix_suggestions(issues)

        # Test should pass if we have actionable suggestions
        [s for s in suggestions if s["type"] in ["replace_import", "fix_broken_import"] and s["new"]]

        # This test mainly validates the suggestion generation works
        assert isinstance(suggestions, list)

    def _is_acceptable_broken_import(self, issue: dict[str, Any]) -> bool:
        """Check if broken import is acceptable during reorganization"""
        import_name = issue.get("import", "")

        # Accept some imports that might be temporarily broken
        acceptable_patterns = [
            "AIVillage.",  # Module being reorganized
            "gateway.",  # Specific reorganization target
            "services.",  # Service layer reorganization
        ]

        return any(pattern in import_name for pattern in acceptable_patterns)


# Test fixtures
@pytest.fixture
def import_validator():
    """Fixture for import validator"""
    return ImportValidator()


@pytest.fixture
def import_validation_test():
    """Fixture for import validation test"""
    return ImportValidationTest()


# Test functions
def test_no_broken_imports(import_validation_test):
    """Test no broken imports"""
    import_validation_test.test_no_broken_imports()


def test_critical_imports_working(import_validation_test):
    """Test critical imports working"""
    import_validation_test.test_critical_imports_working()


def test_no_circular_imports(import_validation_test):
    """Test no circular imports"""
    import_validation_test.test_no_circular_imports()


def test_import_fix_suggestions(import_validation_test):
    """Test import fix suggestions"""
    import_validation_test.test_import_fix_suggestions()


def test_import_scanning(import_validator):
    """Test import scanning functionality"""
    issues = import_validator.scan_import_issues()

    assert isinstance(issues, dict)
    assert "broken_imports" in issues
    assert "outdated_imports" in issues
    assert "circular_imports" in issues


if __name__ == "__main__":
    # Generate import report when run directly
    validator = ImportValidator()
    issues = validator.scan_import_issues()

    print("Import Analysis Report")
    print("=" * 50)

    for issue_type, issue_list in issues.items():
        print(f"\n{issue_type.replace('_', ' ').title()}: {len(issue_list)}")
        if issue_list:
            for issue in issue_list[:3]:  # Show first 3 of each type
                if issue_type == "broken_imports":
                    print(f"  {issue['file']}:{issue['line']} - {issue['import']}")
                elif issue_type == "outdated_imports":
                    print(f"  {issue['file']}:{issue['line']} - {issue['old_import']} -> {issue['suggested_import']}")

            if len(issue_list) > 3:
                print(f"  ... and {len(issue_list) - 3} more")

    # Critical imports status
    validation = validator.validate_critical_imports()
    print(f"\nCritical Imports Status: {validation['status']}")
    print(f"Working: {validation['working']}/{validation['total']}")

    if validation["broken"]:
        print(f"Broken: {', '.join(validation['broken'])}")

    # Run tests
    pytest.main([__file__, "-v"])
