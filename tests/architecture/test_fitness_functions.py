"""
Architectural Fitness Functions for AIVillage
Based on connascence principles and architectural best practices
"""

import ast
from collections import Counter, defaultdict
import os
from pathlib import Path
import re
import sys

import networkx as nx
import pytest
import radon.complexity as cc
import yaml

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class ConnascenceViolation(Exception):
    """Raised when connascence rules are violated"""

    pass


class ArchitecturalViolation(Exception):
    """Raised when architectural rules are violated"""

    pass


class ArchitectureFitnessTests:
    """Comprehensive architectural fitness function test suite"""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.packages_dir = self.project_root / "packages"
        self.config_file = self.project_root / "config" / "architecture_rules.yaml"
        self.load_config()

    def load_config(self):
        """Load architecture rules configuration"""
        if self.config_file.exists():
            with open(self.config_file) as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                "max_file_lines": 500,
                "max_function_complexity": 10,
                "max_function_parameters": 3,
                "max_class_methods": 15,
                "allowed_dependencies": {
                    "core": ["common", "legacy"],
                    "agents": ["core", "common"],
                    "rag": ["core", "common"],
                    "p2p": ["core", "common"],
                    "fog": ["core", "common"],
                    "edge": ["core", "p2p", "common"],
                },
                "forbidden_patterns": [
                    "eval(",
                    "exec(",
                    "pickle.loads",
                    "__import__",
                    "globals(",
                    "locals(",
                ],
                "required_test_coverage": 80,
                "max_coupling_threshold": 0.3,
            }

    def get_python_files(self) -> list[Path]:
        """Get all Python files in packages directory"""
        python_files = []
        for root, dirs, files in os.walk(self.packages_dir):
            # Skip test directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith(("__pycache__", "test", ".")) and d != "codex-audit"]

            for file in files:
                if file.endswith(".py") and not file.startswith("."):
                    python_files.append(Path(root) / file)
        return python_files


class TestCircularDependencies:
    """Test for circular dependencies between modules and packages"""

    def __init__(self):
        self.fitness = ArchitectureFitnessTests()

    def build_dependency_graph(self) -> nx.DiGraph:
        """Build dependency graph from import statements"""
        graph = nx.DiGraph()
        python_files = self.fitness.get_python_files()

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Parse AST to extract imports
                tree = ast.parse(content)
                module_name = self.get_module_name(file_path)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self.add_dependency(graph, module_name, alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            self.add_dependency(graph, module_name, node.module)

            except (SyntaxError, UnicodeDecodeError) as e:
                print(f"Warning: Could not parse {file_path}: {e}")
                continue

        return graph

    def get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name"""
        rel_path = file_path.relative_to(self.fitness.project_root)
        parts = list(rel_path.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]  # Remove .py extension
        return ".".join(parts)

    def add_dependency(self, graph: nx.DiGraph, from_module: str, to_module: str):
        """Add dependency edge to graph"""
        # Normalize module names and filter internal dependencies only
        if to_module.startswith("."):
            # Relative import - resolve to absolute
            to_module = self.resolve_relative_import(from_module, to_module)

        # Only track dependencies within our packages
        if to_module.startswith("packages.") and from_module.startswith("packages.") and from_module != to_module:
            graph.add_edge(from_module, to_module)

    def resolve_relative_import(self, from_module: str, relative_import: str) -> str:
        """Resolve relative imports to absolute module names"""
        from_parts = from_module.split(".")

        # Count leading dots
        dots = 0
        for char in relative_import:
            if char == ".":
                dots += 1
            else:
                break

        # Remove the relative part
        import_part = relative_import[dots:]

        # Calculate base module
        if dots == 1:  # from .module
            base_parts = from_parts[:-1]
        else:  # from ..module (dots > 1)
            base_parts = from_parts[: -(dots - 1)]

        if import_part:
            return ".".join(base_parts + [import_part])
        else:
            return ".".join(base_parts)

    def test_no_circular_dependencies(self):
        """Test that there are no circular dependencies"""
        graph = self.build_dependency_graph()

        try:
            cycles = list(nx.simple_cycles(graph))
            if cycles:
                cycle_info = []
                for cycle in cycles:
                    cycle_info.append(" -> ".join(cycle + [cycle[0]]))

                raise CircularDependencyError(
                    f"Found {len(cycles)} circular dependencies:\n"
                    + "\n".join(f"  {i+1}. {cycle}" for i, cycle in enumerate(cycle_info))
                )
        except nx.NetworkXNoCycle:
            # No cycles found, test passes
            pass

        # Additional check for package-level cycles
        package_graph = self.build_package_dependency_graph(graph)
        try:
            package_cycles = list(nx.simple_cycles(package_graph))
            if package_cycles:
                raise CircularDependencyError(
                    f"Found {len(package_cycles)} package-level circular dependencies:\n"
                    + "\n".join(str(cycle) for cycle in package_cycles)
                )
        except nx.NetworkXNoCycle:
            pass

    def build_package_dependency_graph(self, module_graph: nx.DiGraph) -> nx.DiGraph:
        """Build package-level dependency graph"""
        package_graph = nx.DiGraph()

        for from_module, to_module in module_graph.edges():
            from_package = from_module.split(".")[1] if len(from_module.split(".")) > 1 else from_module
            to_package = to_module.split(".")[1] if len(to_module.split(".")) > 1 else to_module

            if from_package != to_package:
                package_graph.add_edge(from_package, to_package)

        return package_graph


class TestConnascencePrinciples:
    """Test connascence principles are followed"""

    def __init__(self):
        self.fitness = ArchitectureFitnessTests()

    def test_connascence_of_name_locality(self):
        """Test that strong connascence (name) is kept local to same class/function"""
        violations = []
        python_files = self.fitness.get_python_files()

        for file_path in python_files:
            try:
                violations.extend(self.check_name_connascence(file_path))
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")
                continue

        if violations:
            raise ConnascenceViolation(
                f"Found {len(violations)} connascence of name violations:\n" + "\n".join(violations)
            )

    def check_name_connascence(self, file_path: Path) -> list[str]:
        """Check for connascence of name violations"""
        violations = []

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            # Find all variable/attribute names used across class boundaries
            name_usage = defaultdict(list)

            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    context = self.get_context(node, tree)
                    name_usage[node.id].append(context)

            # Check for names used across multiple contexts (potential violation)
            for name, contexts in name_usage.items():
                if len(set(contexts)) > 1 and not self.is_acceptable_shared_name(name):
                    violations.append(f"{file_path}: Variable '{name}' used across multiple contexts: {set(contexts)}")

        except Exception as e:
            import logging

            logging.exception("File parsing error in connascence check: %s", str(e))

        return violations

    def get_context(self, node: ast.Name, tree: ast.AST) -> str:
        """Get the context (class/function) where a name is used"""
        # Walk up the AST to find containing class/function
        for parent in ast.walk(tree):
            if hasattr(parent, "body"):
                if node in ast.walk(parent):
                    if isinstance(parent, ast.ClassDef):
                        return f"class:{parent.name}"
                    elif isinstance(parent, ast.FunctionDef):
                        return f"function:{parent.name}"
        return "module"

    def is_acceptable_shared_name(self, name: str) -> bool:
        """Check if a shared name is acceptable (common patterns)"""
        acceptable_names = {
            # Common variable names that are acceptable to share
            "self",
            "cls",
            "logger",
            "config",
            "result",
            "data",
            "value",
            "i",
            "j",
            "k",
            "x",
            "y",
            "z",  # Common loop/math variables
            # Common modules/imports
            "os",
            "sys",
            "json",
            "yaml",
            "re",
            "typing",
        }
        return name in acceptable_names or name.startswith("_")

    def test_connascence_of_type_locality(self):
        """Test that connascence of type is kept local"""
        violations = []
        python_files = self.fitness.get_python_files()

        for file_path in python_files:
            try:
                violations.extend(self.check_type_connascence(file_path))
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")
                continue

        if violations:
            raise ConnascenceViolation(
                f"Found {len(violations)} connascence of type violations:\n" + "\n".join(violations)
            )

    def check_type_connascence(self, file_path: Path) -> list[str]:
        """Check for connascence of type violations"""
        violations = []

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Look for magic numbers and hardcoded types
            magic_number_pattern = r"\b\d{2,}\b"  # Numbers with 2+ digits
            matches = re.findall(magic_number_pattern, content)

            # Count occurrences of the same magic number
            number_counts = Counter(matches)
            for number, count in number_counts.items():
                if count > 1 and int(number) not in [0, 1, 100, 200, 404, 500]:  # Common acceptable numbers
                    violations.append(
                        f"{file_path}: Magic number '{number}' appears {count} times (possible connascence of value)"
                    )

        except Exception as e:
            import logging

            logging.exception("Type connascence check error: %s", str(e))

        return violations


class TestModuleSizeLimits:
    """Test module size constraints"""

    def __init__(self):
        self.fitness = ArchitectureFitnessTests()

    def test_file_size_limits(self):
        """Test that files don't exceed maximum line count"""
        violations = []
        max_lines = self.fitness.config.get("max_file_lines", 500)

        python_files = self.fitness.get_python_files()

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    line_count = sum(1 for _ in f)

                if line_count > max_lines:
                    violations.append(f"{file_path}: {line_count} lines (max allowed: {max_lines})")
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")
                continue

        if violations:
            raise ArchitecturalViolation(
                f"Found {len(violations)} files exceeding size limits:\n" + "\n".join(violations)
            )


class TestComplexityLimits:
    """Test function and class complexity limits"""

    def __init__(self):
        self.fitness = ArchitectureFitnessTests()

    def test_function_complexity_limits(self):
        """Test that functions don't exceed cyclomatic complexity limits"""
        violations = []
        max_complexity = self.fitness.config.get("max_function_complexity", 10)

        python_files = self.fitness.get_python_files()

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Calculate complexity using radon
                complexity_data = cc.cc_visit(content)

                for item in complexity_data:
                    if item.complexity > max_complexity:
                        violations.append(
                            f"{file_path}:{item.lineno} Function '{item.name}' "
                            f"complexity: {item.complexity} (max: {max_complexity})"
                        )
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")
                continue

        if violations:
            raise ArchitecturalViolation(
                f"Found {len(violations)} functions exceeding complexity limits:\n" + "\n".join(violations)
            )

    def test_function_parameter_limits(self):
        """Test that functions don't have too many parameters"""
        violations = []
        max_params = self.fitness.config.get("max_function_parameters", 3)

        python_files = self.fitness.get_python_files()

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Count parameters (excluding self, cls, *args, **kwargs)
                        param_count = 0
                        for arg in node.args.args:
                            if arg.arg not in ["self", "cls"]:
                                param_count += 1

                        # Don't count *args and **kwargs in the limit
                        if param_count > max_params:
                            violations.append(
                                f"{file_path}:{node.lineno} Function '{node.name}' "
                                f"has {param_count} parameters (max: {max_params})"
                            )
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")
                continue

        if violations:
            raise ArchitecturalViolation(
                f"Found {len(violations)} functions with too many parameters:\n" + "\n".join(violations)
            )


class TestLayeringViolations:
    """Test architectural layering principles"""

    def __init__(self):
        self.fitness = ArchitectureFitnessTests()

    def test_layering_violations(self):
        """Test that lower layers don't import higher layers"""
        violations = []
        allowed_deps = self.fitness.config.get("allowed_dependencies", {})

        python_files = self.fitness.get_python_files()

        for file_path in python_files:
            try:
                package_name = self.get_package_name(file_path)
                if package_name not in allowed_deps:
                    continue

                imports = self.get_imports(file_path)
                allowed = set(allowed_deps[package_name])

                for imported_module in imports:
                    imported_package = self.get_package_from_import(imported_module)

                    if imported_package and imported_package not in allowed and imported_package != package_name:
                        violations.append(
                            f"{file_path}: Package '{package_name}' "
                            f"imports from forbidden layer '{imported_package}'"
                        )
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")
                continue

        if violations:
            raise ArchitecturalViolation(f"Found {len(violations)} layering violations:\n" + "\n".join(violations))

    def get_package_name(self, file_path: Path) -> str:
        """Get package name from file path"""
        rel_path = file_path.relative_to(self.fitness.packages_dir)
        return rel_path.parts[0] if rel_path.parts else ""

    def get_imports(self, file_path: Path) -> set[str]:
        """Get all import statements from file"""
        imports = set()

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
        except Exception as e:
            import logging

            logging.exception("Import extraction error in fitness tests: %s", str(e))

        return imports

    def get_package_from_import(self, import_name: str) -> str | None:
        """Extract package name from import statement"""
        # Handle relative imports
        if import_name.startswith("."):
            return None

        # Handle packages.xxx imports
        if import_name.startswith("packages."):
            parts = import_name.split(".")
            return parts[1] if len(parts) > 1 else None

        # Handle direct package imports
        parts = import_name.split(".")
        if parts[0] in ["core", "agents", "rag", "p2p", "fog", "edge", "hrrm", "monitoring"]:
            return parts[0]

        return None


class TestGlobalState:
    """Test for global singleton/mutable state violations"""

    def __init__(self):
        self.fitness = ArchitectureFitnessTests()

    def test_no_global_singletons(self):
        """Test that there are no global singletons or mutable state"""
        violations = []
        python_files = self.fitness.get_python_files()

        for file_path in python_files:
            try:
                violations.extend(self.check_global_state(file_path))
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")
                continue

        if violations:
            raise ArchitecturalViolation(f"Found {len(violations)} global state violations:\n" + "\n".join(violations))

    def check_global_state(self, file_path: Path) -> list[str]:
        """Check for global state violations"""
        violations = []

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            # Look for module-level variables that are mutable
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            # Check if it's a mutable global variable
                            if (
                                not target.id.isupper()
                                and not target.id.startswith("_")  # Not a constant
                                and self.is_mutable_assignment(node.value)  # Not private
                            ):
                                violations.append(f"{file_path}:{node.lineno} Global mutable variable '{target.id}'")
        except Exception as e:
            import logging

            logging.exception("Global state check error: %s", str(e))

        return violations

    def is_mutable_assignment(self, node: ast.AST) -> bool:
        """Check if assignment creates mutable object"""
        if isinstance(node, ast.List | ast.Dict | ast.Set):
            return True
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                # Common mutable object constructors
                mutable_constructors = {"list", "dict", "set", "defaultdict", "Counter"}
                return node.func.id in mutable_constructors
        return False


class TestSecurityAntiPatterns:
    """Test for security anti-patterns"""

    def __init__(self):
        self.fitness = ArchitectureFitnessTests()

    def test_no_forbidden_patterns(self):
        """Test that forbidden security patterns are not used"""
        violations = []
        forbidden_patterns = self.fitness.config.get("forbidden_patterns", [])

        python_files = self.fitness.get_python_files()

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                for line_num, line in enumerate(content.splitlines(), 1):
                    for pattern in forbidden_patterns:
                        if pattern in line and not line.strip().startswith("#"):
                            violations.append(
                                f"{file_path}:{line_num} Forbidden pattern '{pattern}' found: {line.strip()}"
                            )
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")
                continue

        if violations:
            raise ArchitecturalViolation(
                f"Found {len(violations)} security anti-pattern violations:\n" + "\n".join(violations)
            )


# Custom exceptions
class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected"""

    pass


# Pytest fixtures and test functions
@pytest.fixture
def architecture_fitness():
    """Fixture for architecture fitness tests"""
    return ArchitectureFitnessTests()


def test_no_circular_dependencies():
    """Test that there are no circular dependencies"""
    test_instance = TestCircularDependencies()
    test_instance.test_no_circular_dependencies()


def test_connascence_of_name_locality():
    """Test connascence of name locality"""
    test_instance = TestConnascencePrinciples()
    test_instance.test_connascence_of_name_locality()


def test_connascence_of_type_locality():
    """Test connascence of type locality"""
    test_instance = TestConnascencePrinciples()
    test_instance.test_connascence_of_type_locality()


def test_file_size_limits():
    """Test file size limits"""
    test_instance = TestModuleSizeLimits()
    test_instance.test_file_size_limits()


def test_function_complexity_limits():
    """Test function complexity limits"""
    test_instance = TestComplexityLimits()
    test_instance.test_function_complexity_limits()


def test_function_parameter_limits():
    """Test function parameter limits"""
    test_instance = TestComplexityLimits()
    test_instance.test_function_parameter_limits()


def test_layering_violations():
    """Test architectural layering"""
    test_instance = TestLayeringViolations()
    test_instance.test_layering_violations()


def test_no_global_singletons():
    """Test for global singletons"""
    test_instance = TestGlobalState()
    test_instance.test_no_global_singletons()


def test_no_forbidden_patterns():
    """Test for security anti-patterns"""
    test_instance = TestSecurityAntiPatterns()
    test_instance.test_no_forbidden_patterns()


if __name__ == "__main__":
    # Run all tests when script is executed directly
    pytest.main([__file__, "-v"])
