#!/usr/bin/env python3
"""
Import Validation Script for AI Village
Analyzes import dependencies and identifies issues across the codebase.
"""

import ast
import os
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ImportAnalyzer:
    """Analyzes Python import dependencies and issues."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.import_graph: Dict[str, Set[str]] = {}
        self.import_errors: List[Tuple[str, str]] = []
        self.circular_deps: List[List[str]] = []

    def scan_directory(self, directory: str = None) -> None:
        """Scan directory for Python files and analyze imports."""
        scan_dir = Path(directory) if directory else self.root_dir

        for py_file in scan_dir.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue

            try:
                self._analyze_file(py_file)
            except Exception as e:
                self.import_errors.append((str(py_file), str(e)))
                logger.error(f"Error analyzing {py_file}: {e}")

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during analysis."""
        skip_dirs = {".git", "__pycache__", ".pytest_cache", "node_modules", "venv", ".venv"}
        return any(part in skip_dirs for part in file_path.parts)

    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file for imports."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse AST
            tree = ast.parse(content, filename=str(file_path))

            # Get module name from file path
            module_name = self._get_module_name(file_path)
            if module_name not in self.import_graph:
                self.import_graph[module_name] = set()

            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.import_graph[module_name].add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self.import_graph[module_name].add(node.module)

        except SyntaxError as e:
            self.import_errors.append((str(file_path), f"Syntax Error: {e}"))
        except Exception as e:
            self.import_errors.append((str(file_path), f"Analysis Error: {e}"))

    def _get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        rel_path = file_path.relative_to(self.root_dir)
        if rel_path.name == "__init__.py":
            parts = rel_path.parts[:-1]
        else:
            parts = rel_path.parts[:-1] + (rel_path.stem,)

        return ".".join(parts)

    def find_circular_dependencies(self) -> None:
        """Find circular dependencies using DFS."""
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]) -> bool:
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                if cycle not in self.circular_deps:
                    self.circular_deps.append(cycle)
                return True

            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.import_graph.get(node, []):
                if neighbor in self.import_graph:  # Only check internal modules
                    if dfs(neighbor, path):
                        return True

            rec_stack.remove(node)
            path.pop()
            return False

        for module in self.import_graph:
            if module not in visited:
                dfs(module, [])

    def validate_typing_imports(self) -> List[Tuple[str, str]]:
        """Check for missing or incorrect typing imports."""
        typing_issues = []

        for py_file in self.root_dir.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Check for typing usage without imports
                if any(pattern in content for pattern in ["Dict[", "List[", "Optional[", "Union[", "Tuple["]):
                    if "from typing import" not in content and "import typing" not in content:
                        typing_issues.append((str(py_file), "Uses typing annotations without importing typing"))

            except Exception as e:
                typing_issues.append((str(py_file), f"Error checking typing: {e}"))

        return typing_issues

    def test_imports(self) -> List[Tuple[str, str]]:
        """Test if modules can actually be imported."""
        import_test_results = []

        for module_name in self.import_graph.keys():
            try:
                # Convert module name to file path
                module_path = self.root_dir / "/".join(module_name.split("."))

                # Check if it's a package or module
                if (module_path / "__init__.py").exists():
                    spec_path = module_path / "__init__.py"
                elif (module_path.parent / f"{module_path.name}.py").exists():
                    spec_path = module_path.parent / f"{module_path.name}.py"
                else:
                    continue

                # Try to load the module spec
                spec = importlib.util.spec_from_file_location(module_name, spec_path)
                if spec and spec.loader:
                    importlib.util.module_from_spec(spec)
                    # Don't actually execute the module, just check if it can be loaded

            except Exception as e:
                import_test_results.append((module_name, str(e)))

        return import_test_results

    def generate_report(self) -> str:
        """Generate comprehensive import analysis report."""
        report = []
        report.append("=" * 80)
        report.append("IMPORT DEPENDENCY ANALYSIS REPORT")
        report.append("=" * 80)

        # Summary statistics
        report.append("\nSUMMARY:")
        report.append(f"- Total modules analyzed: {len(self.import_graph)}")
        report.append(f"- Import errors found: {len(self.import_errors)}")
        report.append(f"- Circular dependencies: {len(self.circular_deps)}")

        # Import errors
        if self.import_errors:
            report.append(f"\nIMPORT ERRORS ({len(self.import_errors)}):")
            report.append("-" * 40)
            for file_path, error in self.import_errors:
                report.append(f"• {file_path}")
                report.append(f"  Error: {error}")

        # Circular dependencies
        if self.circular_deps:
            report.append(f"\nCIRCULAR DEPENDENCIES ({len(self.circular_deps)}):")
            report.append("-" * 40)
            for i, cycle in enumerate(self.circular_deps, 1):
                report.append(f"Cycle {i}: {' → '.join(cycle)}")

        # Typing issues
        typing_issues = self.validate_typing_imports()
        if typing_issues:
            report.append(f"\nTYPING IMPORT ISSUES ({len(typing_issues)}):")
            report.append("-" * 40)
            for file_path, issue in typing_issues:
                report.append(f"• {file_path}: {issue}")

        # Module import graph (top dependencies)
        report.append("\nTOP MODULE DEPENDENCIES:")
        report.append("-" * 40)
        sorted_modules = sorted(self.import_graph.items(), key=lambda x: len(x[1]), reverse=True)[:10]

        for module, deps in sorted_modules:
            report.append(f"• {module} ({len(deps)} dependencies)")
            if len(deps) > 0:
                deps_list = sorted(list(deps))[:5]  # Show top 5
                report.append(f"  Dependencies: {', '.join(deps_list)}")
                if len(deps) > 5:
                    report.append(f"  ... and {len(deps) - 5} more")

        return "\n".join(report)


def main():
    """Main entry point for import validation."""
    root_dir = os.getcwd()
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]

    logger.info(f"Analyzing imports in: {root_dir}")

    analyzer = ImportAnalyzer(root_dir)

    # Focus on key directories
    key_dirs = ["infrastructure/p2p", "core/rag", "infrastructure/fog", "tests/communications"]

    for directory in key_dirs:
        dir_path = Path(root_dir) / directory
        if dir_path.exists():
            logger.info(f"Scanning {directory}...")
            analyzer.scan_directory(str(dir_path))

    # Find circular dependencies
    logger.info("Detecting circular dependencies...")
    analyzer.find_circular_dependencies()

    # Generate and print report
    report = analyzer.generate_report()
    print(report)

    # Save report to file
    report_file = Path(root_dir) / "docs" / "import_analysis_report.md"
    report_file.parent.mkdir(exist_ok=True)
    with open(report_file, "w") as f:
        f.write(report)

    logger.info(f"Report saved to: {report_file}")

    # Return exit code based on issues found
    if analyzer.import_errors or analyzer.circular_deps:
        logger.warning("Import issues detected!")
        return 1

    logger.info("No critical import issues found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
