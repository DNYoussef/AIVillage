"""Dependency analysis component for architectural analysis.

Handles module and package dependency graph construction and analysis.
Extracted from ArchitecturalAnalyzer following single responsibility principle.
"""

import ast
from collections import defaultdict
import logging
import os
from pathlib import Path

import networkx as nx

from .analysis_constants import AnalysisConstants

logger = logging.getLogger(__name__)


class DependencyAnalyzer:
    """Analyzes module and package dependencies.

    Single responsibility: Build and analyze dependency graphs
    for modules and packages within the project.
    """

    def __init__(self, project_root: Path, packages_dir: Path | None = None):
        """Initialize dependency analyzer.

        Args:
            project_root: Root directory of the project
            packages_dir: Directory containing packages (defaults to project_root/packages)
        """
        self.project_root = project_root
        self.packages_dir = packages_dir or project_root / "packages"

        # Dependency graphs
        self.dependency_graph = nx.DiGraph()
        self.package_graph = nx.DiGraph()

        logger.info(f"Dependency analyzer initialized for: {self.packages_dir}")

    def get_python_files(self) -> list[Path]:
        """Get all Python files in the packages directory.

        Returns:
            List of Python file paths
        """
        python_files = []

        for root, dirs, files in os.walk(self.packages_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not d.startswith(tuple(AnalysisConstants.EXCLUDED_DIRS))]

            for file in files:
                if file.endswith(AnalysisConstants.PYTHON_EXTENSION) and not file.startswith("."):
                    python_files.append(Path(root) / file)

        logger.info(f"Found {len(python_files)} Python files")
        return python_files

    def build_dependency_graphs(self) -> dict[str, any]:
        """Build module and package dependency graphs.

        Returns:
            Dictionary with analysis results
        """
        logger.info("Building dependency graphs...")

        python_files = self.get_python_files()
        module_dependencies = defaultdict(set)
        package_dependencies = defaultdict(set)

        # Analysis statistics
        analyzed_files = 0
        failed_files = 0

        for file_path in python_files:
            try:
                dependencies = self._analyze_file_dependencies(file_path)
                if dependencies:
                    module_name = dependencies["module_name"]
                    package_name = dependencies["package_name"]
                    imports = dependencies["imports"]

                    # Process imports
                    for imported_module in imports:
                        if imported_module.startswith("packages."):
                            # Module-level dependency
                            module_dependencies[module_name].add(imported_module)
                            self.dependency_graph.add_edge(module_name, imported_module)

                            # Package-level dependency
                            imported_package = self._extract_package_name(imported_module)
                            if imported_package and imported_package != package_name:
                                package_dependencies[package_name].add(imported_package)
                                self.package_graph.add_edge(package_name, imported_package)

                analyzed_files += 1

            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")
                failed_files += 1
                continue

        logger.info(f"Dependency analysis complete: {analyzed_files} analyzed, {failed_files} failed")

        return {
            "module_dependencies": dict(module_dependencies),
            "package_dependencies": dict(package_dependencies),
            "total_files": len(python_files),
            "analyzed_files": analyzed_files,
            "failed_files": failed_files,
            "total_modules": len(self.dependency_graph.nodes()),
            "total_packages": len(self.package_graph.nodes()),
            "total_module_edges": len(self.dependency_graph.edges()),
            "total_package_edges": len(self.package_graph.edges()),
        }

    def _analyze_file_dependencies(self, file_path: Path) -> dict[str, any] | None:
        """Analyze dependencies for a single file.

        Args:
            file_path: Path to Python file

        Returns:
            Dictionary with file analysis results or None if failed
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            module_name = self._get_module_name(file_path)
            package_name = self._get_package_name(file_path)

            # Extract imports
            imports = self._extract_imports(tree, module_name)

            return {
                "module_name": module_name,
                "package_name": package_name,
                "imports": imports,
                "file_path": str(file_path),
            }

        except Exception as e:
            logger.debug(f"Failed to analyze file {file_path}: {e}")
            return None

    def _extract_imports(self, tree: ast.AST, current_module: str) -> set[str]:
        """Extract all import statements from AST.

        Args:
            tree: AST of the file
            current_module: Current module name for relative import resolution

        Returns:
            Set of imported module names
        """
        imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    if node.module.startswith("."):
                        # Resolve relative import
                        resolved = self._resolve_relative_import(current_module, node.module)
                        if resolved:
                            imports.add(resolved)
                    else:
                        imports.add(node.module)

        return imports

    def _resolve_relative_import(self, from_module: str, relative_import: str) -> str | None:
        """Resolve relative imports to absolute module names.

        Args:
            from_module: Module making the import
            relative_import: Relative import string (e.g., "..", ".submodule")

        Returns:
            Absolute module name or None if resolution fails
        """
        try:
            from_parts = from_module.split(".")
            dots = len(relative_import) - len(relative_import.lstrip("."))
            import_part = relative_import[dots:]

            if dots == 1:
                # Single dot: same package
                base_parts = from_parts[:-1]
            else:
                # Multiple dots: go up levels
                base_parts = from_parts[: -(dots - 1)]

            if import_part:
                return ".".join(base_parts + [import_part])
            else:
                return ".".join(base_parts)

        except Exception:
            return None

    def _get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name.

        Args:
            file_path: Path to Python file

        Returns:
            Module name in dot notation
        """
        rel_path = file_path.relative_to(self.project_root)
        parts = list(rel_path.parts)

        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            # Remove .py extension
            parts[-1] = parts[-1][:-3]

        return ".".join(parts)

    def _get_package_name(self, file_path: Path) -> str:
        """Get package name from file path.

        Args:
            file_path: Path to Python file

        Returns:
            Package name
        """
        try:
            rel_path = file_path.relative_to(self.packages_dir)
            return rel_path.parts[0] if rel_path.parts else ""
        except ValueError:
            # File is not within packages directory
            return ""

    def _extract_package_name(self, module_name: str) -> str | None:
        """Extract package name from module name.

        Args:
            module_name: Full module name (e.g., "packages.core.security.encryption")

        Returns:
            Package name (e.g., "core") or None
        """
        if module_name.startswith("packages."):
            parts = module_name.split(".")
            return parts[1] if len(parts) > 1 else None
        return None

    def detect_circular_dependencies(self) -> list[list[str]]:
        """Detect circular dependencies in the dependency graph.

        Returns:
            List of cycles (each cycle is a list of module names)
        """
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))

            # Limit the number of cycles to report
            if len(cycles) > AnalysisConstants.MAX_CYCLES_TO_DETECT:
                logger.warning(f"Found {len(cycles)} cycles, limiting to {AnalysisConstants.MAX_CYCLES_TO_DETECT}")
                cycles = cycles[: AnalysisConstants.MAX_CYCLES_TO_DETECT]

            logger.info(f"Found {len(cycles)} circular dependencies")
            return cycles

        except Exception as e:
            logger.error(f"Failed to detect circular dependencies: {e}")
            return []

    def get_module_dependencies(self, module_name: str) -> dict[str, list[str]]:
        """Get dependencies for a specific module.

        Args:
            module_name: Name of the module

        Returns:
            Dictionary with incoming and outgoing dependencies
        """
        if module_name not in self.dependency_graph:
            return {"incoming": [], "outgoing": []}

        incoming = list(self.dependency_graph.predecessors(module_name))
        outgoing = list(self.dependency_graph.successors(module_name))

        return {
            "incoming": incoming,
            "outgoing": outgoing,
        }

    def get_package_dependencies(self, package_name: str) -> dict[str, list[str]]:
        """Get dependencies for a specific package.

        Args:
            package_name: Name of the package

        Returns:
            Dictionary with incoming and outgoing dependencies
        """
        if package_name not in self.package_graph:
            return {"incoming": [], "outgoing": []}

        incoming = list(self.package_graph.predecessors(package_name))
        outgoing = list(self.package_graph.successors(package_name))

        return {
            "incoming": incoming,
            "outgoing": outgoing,
        }

    def find_leaf_modules(self) -> list[str]:
        """Find modules with no outgoing dependencies (leaf nodes).

        Returns:
            List of leaf module names
        """
        return [node for node in self.dependency_graph.nodes() if self.dependency_graph.out_degree(node) == 0]

    def find_root_modules(self) -> list[str]:
        """Find modules with no incoming dependencies (root nodes).

        Returns:
            List of root module names
        """
        return [node for node in self.dependency_graph.nodes() if self.dependency_graph.in_degree(node) == 0]

    def get_dependency_stats(self) -> dict[str, any]:
        """Get comprehensive dependency statistics.

        Returns:
            Dictionary with dependency statistics
        """
        # Module graph stats
        module_nodes = len(self.dependency_graph.nodes())
        module_edges = len(self.dependency_graph.edges())

        # Package graph stats
        package_nodes = len(self.package_graph.nodes())
        package_edges = len(self.package_graph.edges())

        # Analyze connectivity
        strongly_connected = len(list(nx.strongly_connected_components(self.dependency_graph)))
        weakly_connected = len(list(nx.weakly_connected_components(self.dependency_graph)))

        # Find cycles
        cycles = self.detect_circular_dependencies()

        return {
            "modules": {
                "total_modules": module_nodes,
                "total_dependencies": module_edges,
                "avg_dependencies_per_module": module_edges / module_nodes if module_nodes > 0 else 0,
                "leaf_modules": len(self.find_leaf_modules()),
                "root_modules": len(self.find_root_modules()),
            },
            "packages": {
                "total_packages": package_nodes,
                "total_dependencies": package_edges,
                "avg_dependencies_per_package": package_edges / package_nodes if package_nodes > 0 else 0,
            },
            "connectivity": {
                "strongly_connected_components": strongly_connected,
                "weakly_connected_components": weakly_connected,
                "circular_dependencies": len(cycles),
            },
        }

    def export_graphs(self) -> dict[str, any]:
        """Export dependency graphs for external analysis.

        Returns:
            Dictionary with graph data
        """
        return {
            "module_graph": {
                "nodes": list(self.dependency_graph.nodes()),
                "edges": list(self.dependency_graph.edges()),
            },
            "package_graph": {
                "nodes": list(self.package_graph.nodes()),
                "edges": list(self.package_graph.edges()),
            },
        }
