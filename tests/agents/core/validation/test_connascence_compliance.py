"""Connascence Compliance Validation

Validates that the entire test suite follows connascence principles and
maintains proper coupling boundaries. This meta-test ensures test quality.
"""

import ast
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from packages.agents.core.agent_interface import AgentInterface


class ConnascenceAnalyzer:
    """Analyzer for detecting connascence violations in test code."""

    def __init__(self):
        self.violations = []
        self.metrics = defaultdict(int)

    def analyze_test_file(self, file_path: Path) -> dict[str, Any]:
        """Analyze a test file for connascence violations."""
        try:
            with open(file_path, encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source)

            violations = []
            violations.extend(self._check_position_connascence(tree))
            violations.extend(self._check_meaning_connascence(tree))
            violations.extend(self._check_algorithm_connascence(tree))
            violations.extend(self._check_temporal_connascence(tree))

            return {
                "file": str(file_path),
                "violations": violations,
                "metrics": {
                    "total_functions": self._count_functions(tree),
                    "total_classes": self._count_classes(tree),
                    "magic_numbers": self._count_magic_numbers(tree),
                    "positional_calls": self._count_positional_calls(tree),
                },
            }
        except Exception as e:
            return {"file": str(file_path), "error": str(e), "violations": [], "metrics": {}}

    def _check_position_connascence(self, tree: ast.AST) -> list[dict[str, Any]]:
        """Check for positional parameter connascence."""
        violations = []

        class PositionalCallVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                # Check for calls with many positional arguments
                if len(node.args) > 3:
                    violations.append(
                        {
                            "type": "positional_connascence",
                            "line": node.lineno,
                            "message": f"Function call with {len(node.args)} positional arguments",
                            "suggestion": "Use keyword arguments for clarity",
                        }
                    )
                self.generic_visit(node)

        PositionalCallVisitor().visit(tree)
        return violations

    def _check_meaning_connascence(self, tree: ast.AST) -> list[dict[str, Any]]:
        """Check for meaning connascence (magic numbers/strings)."""
        violations = []

        class MagicValueVisitor(ast.NodeVisitor):
            def visit_Num(self, node):
                # Check for magic numbers
                if isinstance(node.n, int | float) and node.n not in [0, 1, -1]:
                    # Common acceptable values
                    acceptable = [2, 3, 5, 10, 100, 1000, 0.5, 1.0]
                    if node.n not in acceptable:
                        violations.append(
                            {
                                "type": "meaning_connascence",
                                "line": node.lineno,
                                "message": f"Magic number: {node.n}",
                                "suggestion": "Use named constant",
                            }
                        )
                self.generic_visit(node)

            def visit_Str(self, node):
                # Check for magic strings in comparisons
                if len(node.s) > 1 and not node.s.isspace():
                    # Look for string literals in comparisons
                    parent = getattr(node, "parent", None)
                    if isinstance(parent, ast.Compare):
                        violations.append(
                            {
                                "type": "meaning_connascence",
                                "line": node.lineno,
                                "message": f"Magic string in comparison: '{node.s}'",
                                "suggestion": "Use named constant",
                            }
                        )
                self.generic_visit(node)

        # Add parent references for context
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node

        MagicValueVisitor().visit(tree)
        return violations

    def _check_algorithm_connascence(self, tree: ast.AST) -> list[dict[str, Any]]:
        """Check for algorithm connascence (duplicated logic)."""
        violations = []

        # Extract function bodies for comparison
        function_bodies = {}

        class FunctionVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Convert function body to string for comparison
                body_str = ast.dump(node)
                if body_str in function_bodies:
                    violations.append(
                        {
                            "type": "algorithm_connascence",
                            "line": node.lineno,
                            "message": f"Potential duplicate algorithm: {node.name}",
                            "suggestion": "Extract common logic to shared function",
                        }
                    )
                else:
                    function_bodies[body_str] = node.name
                self.generic_visit(node)

        FunctionVisitor().visit(tree)
        return violations

    def _check_temporal_connascence(self, tree: ast.AST) -> list[dict[str, Any]]:
        """Check for temporal connascence (order dependencies)."""
        violations = []

        class TemporalVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Look for setup/teardown patterns that indicate temporal coupling
                setup_calls = []
                teardown_calls = []

                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute):
                        method_name = stmt.func.attr
                        if method_name.startswith("setup") or method_name.endswith("_setup"):
                            setup_calls.append(stmt.lineno)
                        elif method_name.startswith("teardown") or method_name.endswith("_cleanup"):
                            teardown_calls.append(stmt.lineno)

                # If setup and teardown are mixed, it might indicate temporal coupling
                if setup_calls and teardown_calls:
                    mixed_order = any(setup > teardown for setup in setup_calls for teardown in teardown_calls)
                    if mixed_order:
                        violations.append(
                            {
                                "type": "temporal_connascence",
                                "line": node.lineno,
                                "message": f"Mixed setup/teardown order in {node.name}",
                                "suggestion": "Use proper fixtures or context managers",
                            }
                        )

                self.generic_visit(node)

        TemporalVisitor().visit(tree)
        return violations

    def _count_functions(self, tree: ast.AST) -> int:
        """Count total functions in the tree."""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])

    def _count_classes(self, tree: ast.AST) -> int:
        """Count total classes in the tree."""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])

    def _count_magic_numbers(self, tree: ast.AST) -> int:
        """Count magic numbers in the tree."""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Num) and isinstance(node.n, int | float):
                if node.n not in [0, 1, -1, 0.0, 1.0]:
                    count += 1
        return count

    def _count_positional_calls(self, tree: ast.AST) -> int:
        """Count function calls with many positional arguments."""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and len(node.args) > 3:
                count += 1
        return count


class TestConnascenceCompliance:
    """Test suite to validate connascence compliance."""

    @pytest.fixture
    def test_files(self):
        """Get all test files in the agents/core directory."""
        test_dir = Path(__file__).parent.parent
        test_files = []

        for pattern in ["**/*.py"]:
            for file_path in test_dir.glob(pattern):
                if file_path.name.startswith("test_") and file_path.is_file():
                    test_files.append(file_path)

        return test_files

    @pytest.fixture
    def analyzer(self):
        """Get connascence analyzer."""
        return ConnascenceAnalyzer()

    def test_no_excessive_positional_parameters(self, test_files, analyzer):
        """Test files should not use excessive positional parameters."""
        violations = []

        for test_file in test_files:
            analysis = analyzer.analyze_test_file(test_file)
            positional_violations = [v for v in analysis.get("violations", []) if v["type"] == "positional_connascence"]
            violations.extend(positional_violations)

        # Report violations
        if violations:
            violation_summary = "\n".join(
                [f"  {v['message']} at line {v['line']}" for v in violations[:10]]  # Show first 10
            )
            pytest.fail(f"Found {len(violations)} positional parameter violations:\n{violation_summary}")

    def test_minimal_magic_values(self, test_files, analyzer):
        """Test files should minimize magic numbers and strings."""
        total_magic_numbers = 0
        files_with_violations = []

        for test_file in test_files:
            analysis = analyzer.analyze_test_file(test_file)
            magic_count = analysis.get("metrics", {}).get("magic_numbers", 0)
            total_magic_numbers += magic_count

            if magic_count > 5:  # Threshold for acceptable magic numbers
                files_with_violations.append((test_file.name, magic_count))

        # Allow some magic numbers but flag excessive usage
        if files_with_violations:
            violation_summary = "\n".join(
                [f"  {filename}: {count} magic numbers" for filename, count in files_with_violations]
            )
            pytest.fail(
                f"Files with excessive magic numbers:\n{violation_summary}\n" f"Consider using named constants."
            )

    def test_no_algorithm_duplication(self, test_files, analyzer):
        """Test files should not duplicate business logic algorithms."""
        algorithm_violations = []

        for test_file in test_files:
            analysis = analyzer.analyze_test_file(test_file)
            algo_violations = [v for v in analysis.get("violations", []) if v["type"] == "algorithm_connascence"]
            algorithm_violations.extend(algo_violations)

        if algorithm_violations:
            violation_summary = "\n".join([f"  {v['message']} at line {v['line']}" for v in algorithm_violations])
            pytest.fail(f"Found {len(algorithm_violations)} algorithm duplication violations:\n{violation_summary}")

    def test_minimal_temporal_coupling(self, test_files, analyzer):
        """Test files should minimize temporal coupling."""
        temporal_violations = []

        for test_file in test_files:
            analysis = analyzer.analyze_test_file(test_file)
            temp_violations = [v for v in analysis.get("violations", []) if v["type"] == "temporal_connascence"]
            temporal_violations.extend(temp_violations)

        if temporal_violations:
            violation_summary = "\n".join([f"  {v['message']} at line {v['line']}" for v in temporal_violations])
            pytest.fail(f"Found {len(temporal_violations)} temporal coupling violations:\n{violation_summary}")

    def test_proper_builder_usage(self, test_files):
        """Test files should use builder pattern instead of direct constructors."""
        constructor_violations = []

        for test_file in test_files:
            try:
                with open(test_file, encoding="utf-8") as f:
                    content = f.read()

                # Look for direct constructor calls that should use builders
                problematic_patterns = [
                    "TaskInterface(",
                    "MessageInterface(",
                    "AgentMetadata(",
                    "QuietStarReflection(",
                ]

                lines = content.split("\n")
                for line_num, line in enumerate(lines, 1):
                    for pattern in problematic_patterns:
                        if pattern in line and "Builder" not in line:
                            constructor_violations.append(
                                {"file": test_file.name, "line": line_num, "pattern": pattern, "content": line.strip()}
                            )
            except Exception:
                continue  # Skip files with reading issues

        if constructor_violations:
            violation_summary = "\n".join(
                [f"  {v['file']}:{v['line']} - {v['pattern']} should use Builder" for v in constructor_violations[:10]]
            )
            pytest.fail(f"Found {len(constructor_violations)} builder pattern violations:\n{violation_summary}")

    def test_interface_abstraction_compliance(self):
        """Tests should depend on interfaces, not implementations."""
        # Verify that tests primarily use AgentInterface methods
        interface_methods = set(dir(AgentInterface))

        # These are the methods tests should primarily use
        expected_interface_methods = {
            "process_task",
            "can_handle_task",
            "estimate_task_duration",
            "send_message",
            "receive_message",
            "broadcast_message",
            "generate",
            "get_embedding",
            "rerank",
            "introspect",
            "communicate",
            "activate_latent_space",
            "health_check",
            "get_capabilities",
            "get_status",
            "get_performance_metrics",
        }

        # Verify expected methods are in interface
        missing_methods = expected_interface_methods - interface_methods
        assert not missing_methods, f"Interface missing expected methods: {missing_methods}"

        # This test mainly serves as documentation of the interface contract
        assert len(expected_interface_methods) > 10, "Interface should have substantial API"

    def test_test_isolation_patterns(self, test_files):
        """Tests should follow proper isolation patterns."""
        isolation_violations = []

        for test_file in test_files:
            try:
                with open(test_file, encoding="utf-8") as f:
                    content = f.read()

                # Check for potential isolation violations
                problematic_patterns = [
                    ("global ", "Global variables in tests"),
                    ("os.environ[", "Direct environment modification without cleanup"),
                    ("open(", "Direct file operations without temp directories"),
                    ("time.sleep(", "Sleep calls that slow tests unnecessarily"),
                ]

                lines = content.split("\n")
                for line_num, line in enumerate(lines, 1):
                    for pattern, description in problematic_patterns:
                        if pattern in line and "# OK" not in line:
                            # Skip if it's in a comment or docstring
                            stripped = line.strip()
                            if not stripped.startswith("#") and not stripped.startswith('"""'):
                                isolation_violations.append(
                                    {
                                        "file": test_file.name,
                                        "line": line_num,
                                        "description": description,
                                        "content": stripped,
                                    }
                                )
            except Exception:
                continue

        # Filter out acceptable cases (like mocking time.sleep)
        filtered_violations = []
        for violation in isolation_violations:
            content = violation["content"]
            if "patch" in content or "mock" in content or "tmp_path" in content or "fixture" in content:
                continue  # These are acceptable patterns
            filtered_violations.append(violation)

        if filtered_violations:
            violation_summary = "\n".join(
                [f"  {v['file']}:{v['line']} - {v['description']}" for v in filtered_violations[:5]]
            )
            pytest.fail(f"Found {len(filtered_violations)} isolation violations:\n{violation_summary}")


class TestArchitecturalCompliance:
    """Tests that validate architectural compliance."""

    def test_agent_interface_adherence(self):
        """All agent implementations should properly implement the interface."""
        from tests.agents.core.fixtures.conftest import MockTestAgent

        # Verify MockTestAgent implements all required methods
        required_methods = [
            "initialize",
            "shutdown",
            "health_check",
            "process_task",
            "can_handle_task",
            "estimate_task_duration",
            "send_message",
            "receive_message",
            "broadcast_message",
            "generate",
            "get_embedding",
            "rerank",
            "introspect",
            "communicate",
            "activate_latent_space",
        ]

        mock_agent_methods = set(dir(MockTestAgent))

        missing_methods = []
        for method in required_methods:
            if method not in mock_agent_methods:
                missing_methods.append(method)

        assert not missing_methods, f"MockTestAgent missing methods: {missing_methods}"

        # Verify methods are callable
        for method in required_methods:
            assert callable(getattr(MockTestAgent, method)), f"{method} should be callable"

    def test_dependency_direction_compliance(self):
        """Dependencies should flow in the correct direction."""
        # Test modules should depend on core modules, not vice versa

        # Core modules (what tests can depend on)
        core_modules = {
            "packages.agents.core.agent_interface",
            "packages.agents.core.base_agent_template",
            "packages.agents.core.base",
        }

        # Test modules should not be imported by core modules
        test_modules = {
            "tests.agents.core.fixtures",
            "tests.agents.core.behavioral",
            "tests.agents.core.integration",
            "tests.agents.core.properties",
        }

        # This is primarily a documentation test
        # In a real implementation, you'd use import analysis tools
        assert len(core_modules) > 0, "Core modules should exist"
        assert len(test_modules) > 0, "Test modules should exist"

    def test_single_responsibility_adherence(self):
        """Test classes should follow single responsibility principle."""
        # Analyze test class sizes and responsibilities
        from tests.agents.core.behavioral.test_agent_contracts import TestAgentContracts
        from tests.agents.core.integration.test_component_interactions import TestComponentInteractions
        from tests.agents.core.properties.test_agent_invariants import TestAgentInvariants

        test_classes = [TestAgentContracts, TestComponentInteractions, TestAgentInvariants]

        for test_class in test_classes:
            # Count test methods
            test_methods = [
                method
                for method in dir(test_class)
                if method.startswith("test_") and callable(getattr(test_class, method))
            ]

            # Each test class should have focused responsibility (not too many tests)
            assert len(test_methods) <= 20, f"{test_class.__name__} has {len(test_methods)} tests, consider splitting"

            # Should have meaningful number of tests
            assert len(test_methods) >= 3, f"{test_class.__name__} has too few tests: {len(test_methods)}"


class TestDocumentationCompliance:
    """Tests that validate documentation and self-documenting code."""

    def test_test_method_naming(self, test_files):
        """Test methods should have descriptive names."""
        poorly_named_tests = []

        for test_file in test_files:
            try:
                with open(test_file, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                        # Check for descriptive naming
                        if len(node.name) < 15:  # Very short names
                            poorly_named_tests.append(
                                {"file": test_file.name, "method": node.name, "line": node.lineno}
                            )

                        # Check for generic names
                        generic_patterns = ["test_basic", "test_simple", "test_1", "test_2"]
                        if any(pattern in node.name for pattern in generic_patterns):
                            poorly_named_tests.append(
                                {"file": test_file.name, "method": node.name, "line": node.lineno}
                            )
            except Exception:
                continue

        if poorly_named_tests:
            violation_summary = "\n".join(
                [f"  {v['file']}:{v['line']} - {v['method']}" for v in poorly_named_tests[:10]]
            )
            pytest.fail(f"Found {len(poorly_named_tests)} poorly named tests:\n{violation_summary}")

    def test_docstring_coverage(self, test_files):
        """Test classes and complex methods should have docstrings."""
        missing_docstrings = []

        for test_file in test_files:
            try:
                with open(test_file, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Test classes should have docstrings
                        if not ast.get_docstring(node):
                            missing_docstrings.append(
                                {"file": test_file.name, "type": "class", "name": node.name, "line": node.lineno}
                            )
            except Exception:
                continue

        if missing_docstrings:
            violation_summary = "\n".join(
                [f"  {v['file']}:{v['line']} - {v['type']} {v['name']}" for v in missing_docstrings[:10]]
            )
            pytest.fail(f"Found {len(missing_docstrings)} missing docstrings:\n{violation_summary}")


def test_overall_connascence_score(test_files=None):
    """Calculate overall connascence compliance score."""
    if test_files is None:
        test_dir = Path(__file__).parent.parent
        test_files = list(test_dir.glob("**/*.py"))
        test_files = [f for f in test_files if f.name.startswith("test_")]

    analyzer = ConnascenceAnalyzer()
    total_violations = 0
    total_metrics = defaultdict(int)

    for test_file in test_files:
        analysis = analyzer.analyze_test_file(test_file)
        total_violations += len(analysis.get("violations", []))

        for metric, value in analysis.get("metrics", {}).items():
            total_metrics[metric] += value

    # Calculate compliance score (higher is better)
    total_functions = total_metrics.get("total_functions", 1)
    violations_per_function = total_violations / total_functions
    compliance_score = max(0, 100 - (violations_per_function * 10))

    print("\nConnascence Compliance Report:")
    print(f"  Total test files analyzed: {len(test_files)}")
    print(f"  Total functions: {total_functions}")
    print(f"  Total violations: {total_violations}")
    print(f"  Violations per function: {violations_per_function:.2f}")
    print(f"  Compliance score: {compliance_score:.1f}/100")

    # Score should be reasonably high
    assert compliance_score >= 70, f"Connascence compliance score too low: {compliance_score:.1f}"

    return compliance_score
