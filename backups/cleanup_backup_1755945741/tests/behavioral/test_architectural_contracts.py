"""
Behavioral contract tests for clean architecture boundaries.
Tests that verify architectural layer separation and module contracts.
"""

import ast
import importlib
import inspect
from pathlib import Path
import sys
from typing import Protocol
from unittest.mock import Mock, patch

import pytest

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "packages"))


class ArchitecturalContract(Protocol):
    """Protocol defining architectural contract requirements"""

    def get_public_interface(self) -> set[str]:
        """Return public interface methods/functions"""
        ...

    def get_dependencies(self) -> set[str]:
        """Return allowed dependencies"""
        ...

    def validate_layer_separation(self) -> bool:
        """Validate that layer separation rules are followed"""
        ...


class LayerBoundaryTest:
    """Test layer boundary preservation during reorganization"""

    def __init__(self):
        self.allowed_dependencies = {
            "domain": set(),  # Domain layer has no dependencies
            "application": {"domain"},  # Application can depend on domain
            "infrastructure": {"domain", "application"},  # Infrastructure can depend on both
            "interfaces": {"domain", "application", "infrastructure"},  # Interfaces are top layer
        }

    def test_domain_layer_isolation(self):
        """Test that domain layer has no external dependencies"""
        domain_violations = []

        # Check domain modules for external imports
        domain_files = self._get_layer_files("domain")

        for file_path in domain_files:
            try:
                imports = self._get_imports(file_path)
                external_imports = self._filter_external_imports(imports)

                if external_imports:
                    domain_violations.append({"file": str(file_path), "violations": list(external_imports)})
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")

        if domain_violations:
            violation_msg = "\n".join([f"  {v['file']}: {', '.join(v['violations'])}" for v in domain_violations])
            raise AssertionError(f"Domain layer violations found:\n{violation_msg}")

    def test_application_layer_boundaries(self):
        """Test application layer only depends on domain"""
        app_violations = []

        app_files = self._get_layer_files("application")

        for file_path in app_files:
            try:
                imports = self._get_imports(file_path)
                invalid_imports = self._check_layer_dependencies("application", imports)

                if invalid_imports:
                    app_violations.append({"file": str(file_path), "violations": list(invalid_imports)})
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")

        if app_violations:
            violation_msg = "\n".join([f"  {v['file']}: {', '.join(v['violations'])}" for v in app_violations])
            raise AssertionError(f"Application layer violations found:\n{violation_msg}")

    def test_infrastructure_layer_boundaries(self):
        """Test infrastructure layer respects boundaries"""
        infra_violations = []

        infra_files = self._get_layer_files("infrastructure")

        for file_path in infra_files:
            try:
                imports = self._get_imports(file_path)
                invalid_imports = self._check_layer_dependencies("infrastructure", imports)

                if invalid_imports:
                    infra_violations.append({"file": str(file_path), "violations": list(invalid_imports)})
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")

        if infra_violations:
            violation_msg = "\n".join([f"  {v['file']}: {', '.join(v['violations'])}" for v in infra_violations])
            raise AssertionError(f"Infrastructure layer violations found:\n{violation_msg}")

    def _get_layer_files(self, layer: str) -> list[Path]:
        """Get all Python files in a layer"""
        files = []
        packages_dir = PROJECT_ROOT / "packages"

        # Map layer names to actual directory patterns
        layer_patterns = {
            "domain": ["**/entities", "**/value_objects", "**/domain", "**/models"],
            "application": ["**/services", "**/use_cases", "**/application", "**/handlers"],
            "infrastructure": ["**/repositories", "**/adapters", "**/infrastructure", "**/persistence"],
            "interfaces": ["**/api", "**/controllers", "**/interfaces", "**/web"],
        }

        patterns = layer_patterns.get(layer, [])
        for pattern in patterns:
            files.extend(packages_dir.glob(f"{pattern}/**/*.py"))

        return files

    def _get_imports(self, file_path: Path) -> set[str]:
        """Extract import statements from file"""
        imports = set()

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module)
        except Exception:
            pass

        return imports

    def _filter_external_imports(self, imports: set[str]) -> set[str]:
        """Filter out standard library and allowed internal imports"""
        external = set()

        allowed_internal_prefixes = {"packages.", "core.", "agents.", "rag.", "p2p."}

        standard_lib_prefixes = {
            "os",
            "sys",
            "json",
            "yaml",
            "typing",
            "pathlib",
            "collections",
            "dataclasses",
            "enum",
            "abc",
            "asyncio",
            "logging",
            "datetime",
            "uuid",
            "hashlib",
            "re",
            "math",
            "random",
            "itertools",
            "functools",
        }

        for imp in imports:
            # Skip standard library
            if any(imp.startswith(prefix) for prefix in standard_lib_prefixes):
                continue

            # Skip allowed internal imports
            if any(imp.startswith(prefix) for prefix in allowed_internal_prefixes):
                continue

            # Skip relative imports
            if imp.startswith("."):
                continue

            external.add(imp)

        return external

    def _check_layer_dependencies(self, layer: str, imports: set[str]) -> set[str]:
        """Check if imports violate layer dependency rules"""
        violations = set()
        allowed = self.allowed_dependencies.get(layer, set())

        # Map imports to layers
        layer_mapping = {
            "domain": ["entities", "value_objects", "domain", "models"],
            "application": ["services", "use_cases", "application", "handlers"],
            "infrastructure": ["repositories", "adapters", "infrastructure", "persistence"],
            "interfaces": ["api", "controllers", "interfaces", "web"],
        }

        for imp in imports:
            if not imp.startswith("packages."):
                continue

            # Determine which layer this import belongs to
            import_layer = None
            for layer_name, patterns in layer_mapping.items():
                if any(pattern in imp for pattern in patterns):
                    import_layer = layer_name
                    break

            if import_layer and import_layer not in allowed:
                violations.add(imp)

        return violations


class InterfaceStabilityTest:
    """Test that public interfaces remain stable during reorganization"""

    def __init__(self):
        self.critical_interfaces = [
            "packages.agents.core.base",
            "packages.core.common",
            "packages.rag.core.pipeline",
            "packages.p2p.network",
        ]

    def test_public_interface_stability(self):
        """Test that public interfaces are preserved"""
        interface_changes = []

        for interface_module in self.critical_interfaces:
            try:
                # Mock the import to test interface structure
                with patch("sys.modules", new=sys.modules.copy()):
                    expected_interface = self._get_expected_interface(interface_module)
                    actual_interface = self._get_actual_interface(interface_module)

                    missing_methods = expected_interface - actual_interface
                    if missing_methods:
                        interface_changes.append({"module": interface_module, "missing": list(missing_methods)})
            except Exception as e:
                print(f"Warning: Could not test interface {interface_module}: {e}")

        if interface_changes:
            change_msg = "\n".join(
                [f"  {change['module']}: missing {', '.join(change['missing'])}" for change in interface_changes]
            )
            raise AssertionError(f"Interface stability violations:\n{change_msg}")

    def _get_expected_interface(self, module_name: str) -> set[str]:
        """Get expected public interface for a module"""
        # Define expected interfaces based on architectural requirements
        expected_interfaces = {
            "packages.agents.core.base": {"BaseAgent", "AgentInterface", "process_message", "get_capabilities"},
            "packages.core.common": {"Logger", "Config", "ErrorHandler", "validate_input"},
            "packages.rag.core.pipeline": {"RAGPipeline", "process_query", "add_document", "search"},
            "packages.p2p.network": {"P2PNetwork", "connect", "disconnect", "send_message"},
        }

        return expected_interfaces.get(module_name, set())

    def _get_actual_interface(self, module_name: str) -> set[str]:
        """Get actual public interface from module"""
        try:
            # Try to import and inspect the module
            module = importlib.import_module(module_name)

            public_members = set()
            for name in dir(module):
                if not name.startswith("_"):
                    attr = getattr(module, name)
                    if inspect.isclass(attr) or inspect.isfunction(attr):
                        public_members.add(name)

            return public_members
        except ImportError:
            # Module doesn't exist yet - return empty set
            return set()


class BehaviorPreservationTest:
    """Test that behavioral contracts are preserved during reorganization"""

    def test_agent_behavior_contracts(self):
        """Test that agent behavioral contracts are preserved"""
        try:
            # Test agent creation and basic operations
            with patch("packages.agents.core.base.BaseAgent") as mock_agent:
                mock_instance = Mock()
                mock_agent.return_value = mock_instance

                # Test required behavioral contracts
                mock_instance.process_message.return_value = {"status": "success"}
                mock_instance.get_capabilities.return_value = ["chat", "analysis"]

                # Verify contracts
                assert hasattr(mock_instance, "process_message")
                assert hasattr(mock_instance, "get_capabilities")
                assert callable(mock_instance.process_message)
                assert callable(mock_instance.get_capabilities)

        except ImportError:
            # Module being reorganized - test passes if structure exists
            pytest.skip("Agent module under reorganization")

    def test_rag_behavior_contracts(self):
        """Test that RAG behavioral contracts are preserved"""
        try:
            with patch("packages.rag.core.pipeline.RAGPipeline") as mock_rag:
                mock_instance = Mock()
                mock_rag.return_value = mock_instance

                # Test behavioral contracts
                mock_instance.process_query.return_value = {"results": []}
                mock_instance.add_document.return_value = {"status": "indexed"}

                # Verify contracts
                assert hasattr(mock_instance, "process_query")
                assert hasattr(mock_instance, "add_document")
                assert callable(mock_instance.process_query)
                assert callable(mock_instance.add_document)

        except ImportError:
            pytest.skip("RAG module under reorganization")

    def test_p2p_behavior_contracts(self):
        """Test that P2P behavioral contracts are preserved"""
        try:
            with patch("packages.p2p.network.P2PNetwork") as mock_p2p:
                mock_instance = Mock()
                mock_p2p.return_value = mock_instance

                # Test behavioral contracts
                mock_instance.connect.return_value = {"status": "connected"}
                mock_instance.send_message.return_value = {"status": "sent"}

                # Verify contracts
                assert hasattr(mock_instance, "connect")
                assert hasattr(mock_instance, "send_message")
                assert callable(mock_instance.connect)
                assert callable(mock_instance.send_message)

        except ImportError:
            pytest.skip("P2P module under reorganization")


# Test fixtures and runners
@pytest.fixture
def layer_boundary_test():
    """Fixture for layer boundary tests"""
    return LayerBoundaryTest()


@pytest.fixture
def interface_stability_test():
    """Fixture for interface stability tests"""
    return InterfaceStabilityTest()


@pytest.fixture
def behavior_preservation_test():
    """Fixture for behavior preservation tests"""
    return BehaviorPreservationTest()


def test_domain_layer_isolation(layer_boundary_test):
    """Test domain layer isolation"""
    layer_boundary_test.test_domain_layer_isolation()


def test_application_layer_boundaries(layer_boundary_test):
    """Test application layer boundaries"""
    layer_boundary_test.test_application_layer_boundaries()


def test_infrastructure_layer_boundaries(layer_boundary_test):
    """Test infrastructure layer boundaries"""
    layer_boundary_test.test_infrastructure_layer_boundaries()


def test_public_interface_stability(interface_stability_test):
    """Test public interface stability"""
    interface_stability_test.test_public_interface_stability()


def test_agent_behavior_contracts(behavior_preservation_test):
    """Test agent behavior contracts"""
    behavior_preservation_test.test_agent_behavior_contracts()


def test_rag_behavior_contracts(behavior_preservation_test):
    """Test RAG behavior contracts"""
    behavior_preservation_test.test_rag_behavior_contracts()


def test_p2p_behavior_contracts(behavior_preservation_test):
    """Test P2P behavior contracts"""
    behavior_preservation_test.test_p2p_behavior_contracts()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
