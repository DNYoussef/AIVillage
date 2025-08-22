#!/usr/bin/env python3
"""
Clean Architecture Validation Script

Validates that the AIVillage codebase follows clean architecture principles:
- Layer boundaries are respected
- Dependencies flow in the correct direction
- Interface contracts are maintained
- Connascence rules are followed

Usage:
    python scripts/architecture/validate_clean_architecture.py [--fix] [--report]
"""

import argparse
import ast
import json
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

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


class CleanArchitectureValidator:
    """Validates clean architecture compliance"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.violations: list[Violation] = []

        # Load configuration
        config_path = project_root / "config" / "architecture" / "clean_architecture_rules.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Layer definitions
        self.layers = self.config["architecture"]["layers"]

        # Build file-to-layer mapping
        self.file_layers = self._build_file_layer_mapping()

        # Build dependency graph
        self.dependency_graph = self._build_dependency_graph()

    def _build_file_layer_mapping(self) -> dict[str, str]:
        """Build mapping of file paths to layers"""
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

    def _build_dependency_graph(self) -> dict[str, set[str]]:
        """Build dependency graph from imports"""
        graph = {}

        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            relative_path = str(py_file.relative_to(self.project_root))
            graph[relative_path] = set()

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import | ast.ImportFrom):
                        import_path = self._get_import_path(node)
                        if import_path:
                            # Convert import path to file path
                            file_path = self._import_to_file_path(import_path)
                            if file_path:
                                graph[relative_path].add(file_path)

            except Exception as e:
                logger.warning(f"Error parsing {py_file}: {e}")

        return graph

    def _get_import_path(self, node) -> str | None:
        """Extract import path from AST node"""
        if isinstance(node, ast.Import):
            return node.names[0].name if node.names else None
        elif isinstance(node, ast.ImportFrom):
            module = node.module if node.module else ""
            if node.names and node.names[0].name != "*":
                return f"{module}.{node.names[0].name}" if module else node.names[0].name
            return module
        return None

    def _import_to_file_path(self, import_path: str) -> str | None:
        """Convert import path to file path"""
        # Handle relative imports
        if import_path.startswith("."):
            return None

        # Convert module path to file path
        parts = import_path.split(".")

        # Try different combinations to find actual file
        for i in range(len(parts), 0, -1):
            potential_path = "/".join(parts[:i]) + ".py"
            if (self.project_root / potential_path).exists():
                return potential_path

            potential_path = "/".join(parts[:i]) + "/__init__.py"
            if (self.project_root / potential_path).exists():
                return potential_path

        return None

    def validate_all(self) -> list[Violation]:
        """Run all validation checks"""
        logger.info("Running clean architecture validation...")

        self.violations = []

        # Run validation checks
        self._validate_layer_boundaries()
        self._validate_dependency_direction()
        self._validate_file_sizes()
        self._validate_function_complexity()
        self._validate_parameter_counts()
        self._validate_circular_dependencies()
        self._validate_connascence_rules()
        self._validate_interface_compliance()

        # Sort violations by severity
        severity_order = {"error": 0, "warning": 1, "info": 2}
        self.violations.sort(key=lambda v: (severity_order[v.severity], v.file_path))

        return self.violations

    def _validate_layer_boundaries(self) -> None:
        """Validate layer boundary rules"""
        logger.info("Validating layer boundaries...")

        for file_path, dependencies in self.dependency_graph.items():
            file_layer = self.file_layers.get(file_path)
            if not file_layer:
                continue

            layer_config = self.layers.get(file_layer, {})
            forbidden_deps = layer_config.get("forbidden_dependencies", [])

            for dep_file in dependencies:
                dep_layer = self.file_layers.get(dep_file)
                if not dep_layer:
                    continue

                if dep_layer in forbidden_deps:
                    self.violations.append(
                        Violation(
                            type=ViolationType.LAYER_BOUNDARY,
                            severity="error",
                            file_path=file_path,
                            line_number=None,
                            message=f"Layer '{file_layer}' cannot depend on layer '{dep_layer}'",
                            suggestion="Move dependency to allowed layer or use interface",
                        )
                    )

    def _validate_dependency_direction(self) -> None:
        """Validate dependency direction follows clean architecture"""
        logger.info("Validating dependency direction...")

        # Define layer hierarchy (higher numbers cannot depend on lower)
        layer_hierarchy = {
            "libs": 0,
            "infrastructure": 1,
            "core": 2,
            "apps": 3,
            "devops": 4,
            "integrations": 1,  # Same level as infrastructure
        }

        for file_path, dependencies in self.dependency_graph.items():
            file_layer = self.file_layers.get(file_path)
            if not file_layer or file_layer not in layer_hierarchy:
                continue

            file_level = layer_hierarchy[file_layer]

            for dep_file in dependencies:
                dep_layer = self.file_layers.get(dep_file)
                if not dep_layer or dep_layer not in layer_hierarchy:
                    continue

                dep_level = layer_hierarchy[dep_layer]

                # Check if dependency violates hierarchy
                if file_level < dep_level:
                    self.violations.append(
                        Violation(
                            type=ViolationType.DEPENDENCY_DIRECTION,
                            severity="error",
                            file_path=file_path,
                            line_number=None,
                            message=f"Lower layer '{file_layer}' depends on higher layer '{dep_layer}'",
                            suggestion="Move code to appropriate layer or use dependency inversion",
                        )
                    )

    def _validate_file_sizes(self) -> None:
        """Validate file size limits"""
        logger.info("Validating file sizes...")

        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            relative_path = str(py_file.relative_to(self.project_root))
            file_layer = self.file_layers.get(relative_path)

            if not file_layer:
                continue

            layer_config = self.layers.get(file_layer, {})
            max_lines = layer_config.get("max_file_lines", 500)

            try:
                with open(py_file, encoding="utf-8") as f:
                    line_count = sum(1 for line in f if line.strip())

                if line_count > max_lines:
                    self.violations.append(
                        Violation(
                            type=ViolationType.FILE_SIZE,
                            severity="warning",
                            file_path=relative_path,
                            line_number=None,
                            message=f"File has {line_count} lines, exceeds limit of {max_lines}",
                            suggestion="Split file into smaller modules",
                        )
                    )

            except Exception as e:
                logger.warning(f"Error checking file size for {py_file}: {e}")

    def _validate_function_complexity(self) -> None:
        """Validate function complexity limits"""
        logger.info("Validating function complexity...")

        max_complexity = self.config["architecture"]["validation"]["max_complexity"]

        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            relative_path = str(py_file.relative_to(self.project_root))

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        complexity = self._calculate_cyclomatic_complexity(node)

                        if complexity > max_complexity:
                            self.violations.append(
                                Violation(
                                    type=ViolationType.FUNCTION_COMPLEXITY,
                                    severity="warning",
                                    file_path=relative_path,
                                    line_number=node.lineno,
                                    message=f"Function '{node.name}' has complexity {complexity}, exceeds limit of {max_complexity}",
                                    suggestion="Simplify function or split into smaller functions",
                                )
                            )

            except Exception as e:
                logger.warning(f"Error checking complexity for {py_file}: {e}")

    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Count decision points
            if isinstance(child, ast.If | ast.While | ast.For | ast.AsyncFor):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.And | ast.Or):
                complexity += 1

        return complexity

    def _validate_parameter_counts(self) -> None:
        """Validate function parameter limits"""
        logger.info("Validating parameter counts...")

        max_params = self.config["architecture"]["validation"]["max_parameters"]

        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            relative_path = str(py_file.relative_to(self.project_root))

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        param_count = len(node.args.args)

                        # Exclude 'self' parameter
                        if param_count > 0 and node.args.args[0].arg == "self":
                            param_count -= 1

                        if param_count > max_params:
                            self.violations.append(
                                Violation(
                                    type=ViolationType.PARAMETER_COUNT,
                                    severity="warning",
                                    file_path=relative_path,
                                    line_number=node.lineno,
                                    message=f"Function '{node.name}' has {param_count} parameters, exceeds limit of {max_params}",
                                    suggestion="Use parameter objects or keyword-only arguments",
                                )
                            )

            except Exception as e:
                logger.warning(f"Error checking parameters for {py_file}: {e}")

    def _validate_circular_dependencies(self) -> None:
        """Validate no circular dependencies exist"""
        logger.info("Validating circular dependencies...")

        visited = set()
        rec_stack = set()

        def has_cycle(node, path):
            if node in rec_stack:
                cycle = path[path.index(node) :]
                cycle_str = " -> ".join(cycle + [node])
                self.violations.append(
                    Violation(
                        type=ViolationType.CIRCULAR_DEPENDENCY,
                        severity="error",
                        file_path=node,
                        line_number=None,
                        message=f"Circular dependency detected: {cycle_str}",
                        suggestion="Refactor to break circular dependency",
                    )
                )
                return True

            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.dependency_graph.get(node, []):
                if has_cycle(neighbor, path + [node]):
                    return True

            rec_stack.remove(node)
            return False

        for node in self.dependency_graph:
            if node not in visited:
                has_cycle(node, [])

    def _validate_connascence_rules(self) -> None:
        """Validate connascence rules"""
        logger.info("Validating connascence rules...")

        connascence_config = self.config["connascence_rules"]

        # Check for algorithm connascence (duplicate implementations)
        self._check_algorithm_connascence(connascence_config["algorithm_connascence"])

        # Check for position connascence (parameter order dependencies)
        self._check_position_connascence(connascence_config["position_connascence"])

    def _check_algorithm_connascence(self, config: dict) -> None:
        """Check for algorithm connascence violations"""
        max_degree = config["max_degree"]

        # Find functions with similar implementations
        function_signatures = {}

        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            relative_path = str(py_file.relative_to(self.project_root))

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Create signature based on structure
                        signature = self._get_function_signature(node)

                        if signature not in function_signatures:
                            function_signatures[signature] = []

                        function_signatures[signature].append((relative_path, node.name, node.lineno))

            except Exception as e:
                logger.warning(f"Error checking algorithm connascence for {py_file}: {e}")

        # Report duplicates exceeding threshold
        for signature, occurrences in function_signatures.items():
            if len(occurrences) > max_degree:
                for file_path, func_name, line_no in occurrences:
                    self.violations.append(
                        Violation(
                            type=ViolationType.CONNASCENCE_VIOLATION,
                            severity="warning",
                            file_path=file_path,
                            line_number=line_no,
                            message=f"Algorithm connascence: function '{func_name}' has {len(occurrences)} similar implementations",
                            suggestion="Extract common algorithm to shared utility",
                        )
                    )

    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Get structural signature of function for similarity comparison"""
        # Simple heuristic: count different types of AST nodes
        node_counts = {}

        for child in ast.walk(node):
            node_type = type(child).__name__
            node_counts[node_type] = node_counts.get(node_type, 0) + 1

        # Create signature from node counts
        return str(sorted(node_counts.items()))

    def _check_position_connascence(self, config: dict) -> None:
        """Check for position connascence violations"""
        max_degree = config["max_degree"]

        # Check for functions with many positional parameters
        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            relative_path = str(py_file.relative_to(self.project_root))

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        positional_args = len([arg for arg in node.args.args if arg.arg != "self"])

                        if positional_args > max_degree:
                            self.violations.append(
                                Violation(
                                    type=ViolationType.CONNASCENCE_VIOLATION,
                                    severity="warning",
                                    file_path=relative_path,
                                    line_number=node.lineno,
                                    message=f"Position connascence: function '{node.name}' has {positional_args} positional parameters",
                                    suggestion="Use keyword-only arguments or parameter objects",
                                )
                            )

            except Exception as e:
                logger.warning(f"Error checking position connascence for {py_file}: {e}")

    def _validate_interface_compliance(self) -> None:
        """Validate interface compliance"""
        logger.info("Validating interface compliance...")

        # Find interface definitions
        interfaces = self._find_interfaces()

        # Find implementations
        implementations = self._find_implementations(interfaces)

        # Validate implementations comply with interfaces
        for interface, impls in implementations.items():
            interface_methods = self._get_interface_methods(interface)

            for impl_file, impl_class in impls:
                impl_methods = self._get_class_methods(impl_file, impl_class)

                # Check all interface methods are implemented
                missing_methods = interface_methods - impl_methods
                if missing_methods:
                    self.violations.append(
                        Violation(
                            type=ViolationType.INTERFACE_VIOLATION,
                            severity="error",
                            file_path=impl_file,
                            line_number=None,
                            message=f"Class '{impl_class}' missing interface methods: {missing_methods}",
                            suggestion="Implement all abstract methods from interface",
                        )
                    )

    def _find_interfaces(self) -> list[tuple[str, str]]:
        """Find interface definitions (ABC classes)"""
        interfaces = []

        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            relative_path = str(py_file.relative_to(self.project_root))

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check if class inherits from ABC
                        is_abstract = any(
                            isinstance(base, ast.Name)
                            and base.id == "ABC"
                            or isinstance(base, ast.Attribute)
                            and base.attr == "ABC"
                            for base in node.bases
                        )

                        if is_abstract:
                            interfaces.append((relative_path, node.name))

            except Exception as e:
                logger.warning(f"Error finding interfaces in {py_file}: {e}")

        return interfaces

    def _find_implementations(self, interfaces: list[tuple[str, str]]) -> dict[str, list[tuple[str, str]]]:
        """Find implementations of interfaces"""
        implementations = {interface[1]: [] for interface in interfaces}

        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            relative_path = str(py_file.relative_to(self.project_root))

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check if class implements any interface
                        for base in node.bases:
                            if isinstance(base, ast.Name):
                                interface_name = base.id
                                if interface_name in implementations:
                                    implementations[interface_name].append((relative_path, node.name))

            except Exception as e:
                logger.warning(f"Error finding implementations in {py_file}: {e}")

        return implementations

    def _get_interface_methods(self, interface_info: tuple[str, str]) -> set[str]:
        """Get abstract methods from interface"""
        file_path, class_name = interface_info
        methods = set()

        try:
            with open(self.project_root / file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            # Check if method has @abstractmethod decorator
                            is_abstract = any(
                                isinstance(decorator, ast.Name) and decorator.id == "abstractmethod"
                                for decorator in item.decorator_list
                            )

                            if is_abstract:
                                methods.add(item.name)

        except Exception as e:
            logger.warning(f"Error getting interface methods: {e}")

        return methods

    def _get_class_methods(self, file_path: str, class_name: str) -> set[str]:
        """Get methods from implementation class"""
        methods = set()

        try:
            with open(self.project_root / file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.add(item.name)

        except Exception as e:
            logger.warning(f"Error getting class methods: {e}")

        return methods

    def generate_report(self, output_path: Path | None = None) -> str:
        """Generate validation report"""
        if not output_path:
            output_path = self.project_root / "reports" / "architecture" / "validation_report.md"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Count violations by type and severity
        by_type = {}
        by_severity = {"error": 0, "warning": 0, "info": 0}

        for violation in self.violations:
            by_type[violation.type.value] = by_type.get(violation.type.value, 0) + 1
            by_severity[violation.severity] += 1

        # Generate report content
        report = f"""# Clean Architecture Validation Report

Generated: {__import__('datetime').datetime.now().isoformat()}

## Summary

- **Total Violations**: {len(self.violations)}
- **Errors**: {by_severity['error']}
- **Warnings**: {by_severity['warning']}
- **Info**: {by_severity['info']}

## Violations by Type

"""

        for violation_type, count in sorted(by_type.items()):
            report += f"- **{violation_type.replace('_', ' ').title()}**: {count}\n"

        report += "\n## Detailed Violations\n\n"

        current_type = None
        for violation in self.violations:
            if violation.type.value != current_type:
                current_type = violation.type.value
                report += f"\n### {current_type.replace('_', ' ').title()}\n\n"

            report += f"**{violation.severity.upper()}** - `{violation.file_path}`"
            if violation.line_number:
                report += f":{violation.line_number}"
            report += f"\n{violation.message}\n"

            if violation.suggestion:
                report += f"*Suggestion: {violation.suggestion}*\n"

            report += "\n"

        # Write report
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"Validation report written to: {output_path}")

        return report

    def auto_fix_violations(self) -> int:
        """Automatically fix violations where possible"""
        fixed_count = 0

        for violation in self.violations:
            if self._can_auto_fix(violation):
                if self._fix_violation(violation):
                    fixed_count += 1

        logger.info(f"Auto-fixed {fixed_count} violations")
        return fixed_count

    def _can_auto_fix(self, violation: Violation) -> bool:
        """Check if violation can be automatically fixed"""
        auto_fixable = {
            ViolationType.PARAMETER_COUNT,  # Can suggest parameter objects
            ViolationType.FILE_SIZE,  # Can suggest splitting
        }

        return violation.type in auto_fixable

    def _fix_violation(self, violation: Violation) -> bool:
        """Fix a specific violation"""
        # Placeholder for auto-fix implementations
        # For now, just log what would be fixed
        logger.info(f"Would fix: {violation.message}")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Validate clean architecture compliance")
    parser.add_argument("--fix", action="store_true", help="Automatically fix violations where possible")
    parser.add_argument("--report", type=Path, help="Output path for validation report")
    parser.add_argument("--format", choices=["json", "yaml", "markdown"], default="markdown", help="Report format")

    args = parser.parse_args()

    validator = CleanArchitectureValidator(PROJECT_ROOT)

    # Run validation
    violations = validator.validate_all()

    # Generate report
    if args.format == "markdown":
        report = validator.generate_report(args.report)
        print(report if not args.report else f"Report written to {args.report}")
    elif args.format == "json":
        report_data = {
            "violations": [
                {
                    "type": v.type.value,
                    "severity": v.severity,
                    "file_path": v.file_path,
                    "line_number": v.line_number,
                    "message": v.message,
                    "suggestion": v.suggestion,
                }
                for v in violations
            ]
        }

        output_path = args.report or PROJECT_ROOT / "reports" / "architecture" / "validation_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"JSON report written to: {output_path}")

    # Auto-fix if requested
    if args.fix:
        fixed_count = validator.auto_fix_violations()
        logger.info(f"Fixed {fixed_count} violations")

    # Exit with error code if violations found
    error_count = sum(1 for v in violations if v.severity == "error")
    if error_count > 0:
        logger.error(f"Found {error_count} errors")
        sys.exit(1)
    else:
        logger.info("Validation passed!")


if __name__ == "__main__":
    main()
