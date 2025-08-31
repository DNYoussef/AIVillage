#!/usr/bin/env python3
"""
Clean Architecture Validation Script - Refactored

Validates that the AIVillage codebase follows clean architecture principles
using modular components for maintainability and single responsibility.

Usage:
    python scripts/architecture/validate_clean_architecture.py [--fix] [--report]
"""

import argparse
import ast
from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path
import sys

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ViolationType(Enum):
    LAYER_BOUNDARY = "layer_boundary"
    DEPENDENCY_DIRECTION = "dependency_direction"
    INTERFACE_VIOLATION = "interface_violation"
    CONNASCENCE_VIOLATION = "connascence_violation"
    FILE_SIZE = "file_size"
    FUNCTION_COMPLEXITY = "function_complexity"
    PARAMETER_COUNT = "parameter_count"
    CIRCULAR_DEPENDENCY = "circular_dependency"


@dataclass
class Violation:
    type: ViolationType
    severity: str  # "error", "warning", "info"
    file_path: str
    line_number: int | None
    message: str
    suggestion: str | None = None


class ArchitectureRuleChecker:
    """Validates basic architecture rules and conventions."""

    def __init__(self, config: dict):
        self.config = config
        self.violations = []

    def check_file_size_limits(self, file_path: Path) -> list[Violation]:
        """Check if files exceed size limits."""
        violations = []
        max_lines = self.config.get("max_file_lines", 500)

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                line_count = sum(1 for _ in f)

            if line_count > max_lines:
                violations.append(
                    Violation(
                        type=ViolationType.FILE_SIZE,
                        severity="warning",
                        file_path=str(file_path),
                        line_number=None,
                        message=f"File has {line_count} lines (exceeds {max_lines} limit)",
                        suggestion="Consider refactoring into smaller, focused modules",
                    )
                )
        except Exception as e:
            logger.warning(f"Could not check file size for {file_path}: {e}")

        return violations

    def check_function_complexity(self, file_path: Path) -> list[Violation]:
        """Check function complexity and parameter counts."""
        violations = []
        max_params = self.config.get("max_function_params", 5)

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
                tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    param_count = len(node.args.args)
                    if param_count > max_params:
                        violations.append(
                            Violation(
                                type=ViolationType.PARAMETER_COUNT,
                                severity="warning",
                                file_path=str(file_path),
                                line_number=node.lineno,
                                message=f"Function '{node.name}' has {param_count} parameters (exceeds {max_params})",
                                suggestion="Use keyword-only parameters or group parameters into objects",
                            )
                        )

        except Exception as e:
            logger.warning(f"Could not analyze function complexity for {file_path}: {e}")

        return violations


class DependencyValidator:
    """Validates dependency relationships and layer boundaries."""

    def __init__(self, project_root: Path, config: dict):
        self.project_root = project_root
        self.config = config
        self.file_layers = self._build_file_layer_mapping()

    def _build_file_layer_mapping(self) -> dict[str, str]:
        """Build mapping of file paths to architectural layers."""
        mapping = {}
        layer_dirs = {
            "apps": ["apps"],
            "core": ["core"],
            "infrastructure": ["infrastructure", "gateway", "twin", "mcp", "p2p", "shared"],
            "devops": ["devops"],
            "libs": ["libs"],
            "integrations": ["integrations"],
        }

        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            relative_path = py_file.relative_to(self.project_root)

            for layer, dirs in layer_dirs.items():
                if any(str(relative_path).startswith(d) for d in dirs):
                    mapping[str(relative_path)] = layer
                    break

        return mapping

    def validate_layer_dependencies(self, file_path: Path) -> list[Violation]:
        """Validate that dependencies respect layer boundaries."""
        violations = []
        file_key = str(file_path.relative_to(self.project_root))
        file_layer = self.file_layers.get(file_key)

        if not file_layer:
            return violations

        try:
            imports = self._extract_imports(file_path)

            for imported_module in imports:
                if self._is_internal_import(imported_module):
                    imported_layer = self._get_layer_for_module(imported_module)

                    if imported_layer and not self._is_valid_dependency(file_layer, imported_layer):
                        violations.append(
                            Violation(
                                type=ViolationType.LAYER_BOUNDARY,
                                severity="error",
                                file_path=str(file_path),
                                line_number=None,
                                message=f"Invalid layer dependency: {file_layer} -> {imported_layer}",
                                suggestion="Use dependency injection or move to appropriate layer",
                            )
                        )
        except Exception as e:
            logger.warning(f"Could not validate dependencies for {file_path}: {e}")

        return violations

    def _extract_imports(self, file_path: Path) -> list[str]:
        """Extract import statements from a Python file."""
        imports = []
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except Exception as e:
            logging.debug(f"Failed to extract imports from {file_path}: {e}")

        return imports

    def _is_internal_import(self, module_name: str) -> bool:
        """Check if import is internal to the project."""
        return module_name.startswith(("core", "infrastructure", "apps", "integrations"))

    def _get_layer_for_module(self, module_name: str) -> str | None:
        """Get the layer for a given module."""
        if module_name.startswith("core"):
            return "core"
        elif module_name.startswith("infrastructure"):
            return "infrastructure"
        elif module_name.startswith("apps"):
            return "apps"
        elif module_name.startswith("integrations"):
            return "integrations"
        return None

    def _is_valid_dependency(self, from_layer: str, to_layer: str) -> bool:
        """Check if dependency between layers is valid according to clean architecture."""
        # Clean architecture dependency rules
        valid_dependencies = {
            "apps": ["core", "infrastructure", "integrations"],
            "infrastructure": ["core"],
            "core": [],  # Core should not depend on anything
            "integrations": ["core", "infrastructure"],
            "devops": ["core", "infrastructure", "apps"],
            "libs": [],
        }

        return to_layer in valid_dependencies.get(from_layer, [])


class LayerAnalyzer:
    """Analyzes layer structure and cohesion."""

    def __init__(self, project_root: Path, dependency_validator: DependencyValidator):
        self.project_root = project_root
        self.dependency_validator = dependency_validator

    def analyze_layer_cohesion(self) -> dict[str, dict]:
        """Analyze cohesion within each architectural layer."""
        layer_stats = {}

        for file_path, layer in self.dependency_validator.file_layers.items():
            if layer not in layer_stats:
                layer_stats[layer] = {"files": 0, "total_lines": 0, "violations": 0}

            layer_stats[layer]["files"] += 1

            # Count lines and violations
            try:
                full_path = self.project_root / file_path
                with open(full_path, encoding="utf-8", errors="ignore") as f:
                    lines = sum(1 for _ in f)
                layer_stats[layer]["total_lines"] += lines
            except Exception:
                continue

        return layer_stats

    def find_misplaced_components(self) -> list[Violation]:
        """Find components that might be in the wrong layer."""
        violations = []

        # Simple heuristic: files with "Repository" in name should be in infrastructure
        # files with "Service" should be in core, etc.

        for file_path, layer in self.dependency_validator.file_layers.items():
            full_path = self.project_root / file_path
            file_name = full_path.name

            # Check for common misplacements
            if "repository" in file_name.lower() and layer != "infrastructure":
                violations.append(
                    Violation(
                        type=ViolationType.LAYER_BOUNDARY,
                        severity="warning",
                        file_path=file_path,
                        line_number=None,
                        message=f"Repository pattern file in {layer} layer (should be in infrastructure)",
                        suggestion="Move repository implementations to infrastructure layer",
                    )
                )

            if "service" in file_name.lower() and layer not in ["core", "apps"]:
                violations.append(
                    Violation(
                        type=ViolationType.LAYER_BOUNDARY,
                        severity="warning",
                        file_path=file_path,
                        line_number=None,
                        message=f"Service file in {layer} layer (should be in core or apps)",
                        suggestion="Move business logic services to core layer",
                    )
                )

        return violations


class CleanArchitectureValidator:
    """Main validator facade - simplified and focused."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.violations: list[Violation] = []

        # Load configuration with defaults
        config_path = project_root / "config" / "architecture" / "clean_architecture_rules.yaml"
        if config_path.exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {"max_file_lines": 500, "max_function_params": 5, "architecture": {"layers": {}}}

        # Initialize component validators
        self.rule_checker = ArchitectureRuleChecker(self.config)
        self.dependency_validator = DependencyValidator(project_root, self.config)
        self.layer_analyzer = LayerAnalyzer(project_root, self.dependency_validator)

    def validate_project(self) -> list[Violation]:
        """Run comprehensive clean architecture validation."""
        logger.info("Starting clean architecture validation...")

        all_violations = []
        python_files = list(self.project_root.rglob("*.py"))

        # Filter out common non-source files
        source_files = [f for f in python_files if "__pycache__" not in str(f) and "deprecated" not in str(f)]

        logger.info(f"Analyzing {len(source_files)} Python files...")

        for file_path in source_files:
            try:
                # Check file-level rules
                all_violations.extend(self.rule_checker.check_file_size_limits(file_path))
                all_violations.extend(self.rule_checker.check_function_complexity(file_path))

                # Check layer dependencies
                all_violations.extend(self.dependency_validator.validate_layer_dependencies(file_path))

            except Exception as e:
                logger.warning(f"Error validating {file_path}: {e}")

        # Check for misplaced components
        all_violations.extend(self.layer_analyzer.find_misplaced_components())

        # Sort by severity
        severity_order = {"error": 0, "warning": 1, "info": 2}
        all_violations.sort(key=lambda v: (severity_order.get(v.severity, 3), v.file_path))

        self.violations = all_violations
        return all_violations

    def generate_report(self, output_file: Path | None = None) -> dict:
        """Generate validation report."""
        if not self.violations:
            self.validate_project()

        # Group violations by type and severity
        report = {
            "summary": {
                "total_violations": len(self.violations),
                "errors": len([v for v in self.violations if v.severity == "error"]),
                "warnings": len([v for v in self.violations if v.severity == "warning"]),
                "info": len([v for v in self.violations if v.severity == "info"]),
            },
            "violations_by_type": {},
            "violations_by_file": {},
            "layer_analysis": self.layer_analyzer.analyze_layer_cohesion(),
        }

        # Group violations
        for violation in self.violations:
            # By type
            violation_type = violation.type.value
            if violation_type not in report["violations_by_type"]:
                report["violations_by_type"][violation_type] = []
            report["violations_by_type"][violation_type].append(
                {
                    "severity": violation.severity,
                    "file": violation.file_path,
                    "line": violation.line_number,
                    "message": violation.message,
                    "suggestion": violation.suggestion,
                }
            )

            # By file
            if violation.file_path not in report["violations_by_file"]:
                report["violations_by_file"][violation.file_path] = []
            report["violations_by_file"][violation.file_path].append(
                {
                    "type": violation_type,
                    "severity": violation.severity,
                    "line": violation.line_number,
                    "message": violation.message,
                }
            )

        # Save report if requested
        if output_file:
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {output_file}")

        return report

    def print_summary(self):
        """Print validation summary to console."""
        if not self.violations:
            print("✅ No clean architecture violations found!")
            return

        print("\n🏗️  Clean Architecture Validation Summary")
        print("=" * 50)

        errors = [v for v in self.violations if v.severity == "error"]
        warnings = [v for v in self.violations if v.severity == "warning"]
        info = [v for v in self.violations if v.severity == "info"]

        print(f"❌ Errors: {len(errors)}")
        print(f"⚠️  Warnings: {len(warnings)}")
        print(f"ℹ️  Info: {len(info)}")

        # Show top violations
        if errors:
            print("\n🔴 Critical Issues:")
            for error in errors[:5]:
                print(f"  • {error.file_path}:{error.line_number or 'N/A'} - {error.message}")

        if warnings:
            print("\n🟡 Warnings:")
            for warning in warnings[:10]:
                print(f"  • {warning.file_path}:{warning.line_number or 'N/A'} - {warning.message}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Clean Architecture Validator")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root directory")
    parser.add_argument("--report", type=Path, help="Generate JSON report file")
    parser.add_argument("--summary", action="store_true", help="Print summary to console")

    args = parser.parse_args()

    # Run validation
    validator = CleanArchitectureValidator(args.project_root)
    violations = validator.validate_project()

    logger.info(f"Found {len(violations)} violations")

    # Generate report if requested
    if args.report:
        validator.generate_report(args.report)

    # Print summary if requested or no other output
    if args.summary or not args.report:
        validator.print_summary()

    # Return appropriate exit code
    errors = [v for v in violations if v.severity == "error"]
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
