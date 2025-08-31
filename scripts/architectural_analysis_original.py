#!/usr/bin/env python3
"""
Continuous Architecture Monitoring and Analysis Script

This script performs comprehensive architectural analysis including:
- Dependency graph generation and analysis
- Coupling metrics calculation
- Connascence detection and hotspot identification
- Architectural drift detection
- Technical debt assessment
- Health reporting

Usage:
    python scripts/architectural_analysis.py [--output-dir reports] [--format json|html|both]
"""

import argparse
import ast
from collections import defaultdict
from dataclasses import asdict, dataclass
import logging
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import radon.complexity as cc
import radon.metrics as rm
import seaborn as sns
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ConnascenceMetric:
    """Represents a connascence measurement"""

    type: str
    strength: str  # weak, strong
    locality: str  # same_function, same_class, same_module, cross_module
    instances: int
    files_affected: list[str]
    severity: str  # low, medium, high, critical


@dataclass
class CouplingMetric:
    """Represents coupling measurements"""

    module: str
    efferent_coupling: int  # Dependencies going out
    afferent_coupling: int  # Dependencies coming in
    instability: float  # Ce / (Ce + Ca)
    abstractness: float  # Abstract classes / Total classes
    distance: float  # Distance from main sequence


@dataclass
class ArchitecturalDrift:
    """Represents architectural drift measurement"""

    component: str
    drift_type: str
    severity: float
    description: str
    recommendation: str
    files_affected: list[str]


@dataclass
class TechnicalDebt:
    """Represents technical debt measurement"""

    category: str
    amount: float  # 0-100 scale
    location: str
    description: str
    effort_hours: float
    risk_level: str


@dataclass
class ArchitecturalReport:
    """Comprehensive architectural health report"""

    timestamp: str
    summary: dict[str, Any]
    dependency_metrics: dict[str, Any]
    coupling_metrics: list[CouplingMetric]
    connascence_metrics: list[ConnascenceMetric]
    architectural_drift: list[ArchitecturalDrift]
    technical_debt: list[TechnicalDebt]
    quality_gates: dict[str, bool]
    recommendations: list[str]
    trend_data: dict[str, list[float]]


class DependencyAnalyzer:
    """Handles dependency graph generation and analysis."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.dependency_graph = nx.DiGraph()

    def build_dependency_graph(self) -> nx.DiGraph:
        """Build module-level dependency graph."""
        python_files = list(self.project_root.rglob("*.py"))

        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue

            module_name = self._get_module_name(file_path)
            self.dependency_graph.add_node(module_name, file_path=str(file_path))

            try:
                imports = self._extract_imports(file_path)
                for imported_module in imports:
                    if self._is_internal_module(imported_module):
                        self.dependency_graph.add_edge(module_name, imported_module)
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")

        return self.dependency_graph

    def _should_skip_file(self, file_path: Path) -> bool:
        skip_patterns = ["__pycache__", ".git", "build", "dist", "deprecated"]
        return any(pattern in str(file_path) for pattern in skip_patterns)

    def _get_module_name(self, file_path: Path) -> str:
        relative_path = file_path.relative_to(self.project_root)
        return str(relative_path).replace("/", ".").replace("\\", ".").replace(".py", "")

    def _extract_imports(self, file_path: Path) -> list[str]:
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
        except Exception:
            pass
        return imports

    def _is_internal_module(self, module_name: str) -> bool:
        return module_name.startswith(("core", "infrastructure", "packages", "scripts"))


class ConnascenceAnalyzer:
    """Handles connascence detection and analysis."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.connascence_metrics = []

    def analyze_connascence(self) -> list[ConnascenceMetric]:
        """Analyze connascence across the codebase."""
        python_files = list(self.project_root.rglob("*.py"))

        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue

            try:
                self._analyze_file_connascence(file_path)
            except Exception as e:
                print(f"Warning: Could not analyze connascence in {file_path}: {e}")

        return self.connascence_metrics

    def _should_skip_file(self, file_path: Path) -> bool:
        skip_patterns = ["__pycache__", ".git", "build", "dist", "deprecated"]
        return any(pattern in str(file_path) for pattern in skip_patterns)

    def _analyze_file_connascence(self, file_path: Path):
        """Analyze connascence patterns in a single file."""
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
                tree = ast.parse(content)

            # Analyze different types of connascence
            self._detect_name_connascence(tree, file_path)
            self._detect_type_connascence(tree, file_path)
            self._detect_meaning_connascence(tree, file_path)
            self._detect_position_connascence(tree, file_path)

        except Exception as e:
            logging.warning(f"Failed to analyze connascence for {file_path}: {e}")

    def _detect_name_connascence(self, tree: ast.AST, file_path: Path):
        """Detect connascence of name."""
        names = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                name = node.id
                if name in names:
                    names[name] += 1
                else:
                    names[name] = 1

        high_usage_names = {name: count for name, count in names.items() if count > 10}
        if high_usage_names:
            self.connascence_metrics.append(
                ConnascenceMetric(
                    type="name",
                    strength="weak",
                    locality="same_file",
                    instances=len(high_usage_names),
                    files_affected=[str(file_path)],
                )
            )

    def _detect_type_connascence(self, tree: ast.AST, file_path: Path):
        """Detect connascence of type."""
        # Simplified type detection
        pass

    def _detect_meaning_connascence(self, tree: ast.AST, file_path: Path):
        """Detect connascence of meaning (magic numbers/strings)."""
        magic_literals = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant | ast.Num | ast.Str):
                if isinstance(node, ast.Constant) and isinstance(node.value, int | float | str):
                    if node.value not in (0, 1, True, False, None, "", []):
                        magic_literals += 1

        if magic_literals > 5:
            self.connascence_metrics.append(
                ConnascenceMetric(
                    type="meaning",
                    strength="strong",
                    locality="same_file",
                    instances=magic_literals,
                    files_affected=[str(file_path)],
                )
            )

    def _detect_position_connascence(self, tree: ast.AST, file_path: Path):
        """Detect connascence of position (argument order)."""
        functions_with_many_args = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.args.args) > 5:
                    functions_with_many_args += 1

        if functions_with_many_args > 0:
            self.connascence_metrics.append(
                ConnascenceMetric(
                    type="position",
                    strength="strong",
                    locality="same_file",
                    instances=functions_with_many_args,
                    files_affected=[str(file_path)],
                )
            )


class ArchitecturalAnalyzer:
    """Main architectural analysis engine"""

    def __init__(self, project_root: Path, config_file: Path | None = None):
        self.project_root = project_root
        self.packages_dir = project_root / "packages"
        self.reports_dir = project_root / "reports" / "architecture"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config_file = config_file or project_root / "config" / "architecture_rules.yaml"
        self.load_config()

        # Initialize analysis data
        self.dependency_graph = nx.DiGraph()
        self.package_graph = nx.DiGraph()
        self.coupling_metrics = []
        self.connascence_metrics = []
        self.architectural_drift = []
        self.technical_debt = []

    def load_config(self):
        """Load architectural analysis configuration"""
        if self.config_file.exists():
            with open(self.config_file) as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                "max_coupling_threshold": 0.3,
                "max_file_lines": 500,
                "max_function_complexity": 10,
                "quality_thresholds": {"maintainability_index": 70, "technical_debt_ratio": 5},
            }

    def get_python_files(self) -> list[Path]:
        """Get all Python files in the project"""
        python_files = []
        for root, dirs, files in os.walk(self.packages_dir):
            # Skip test directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith(("__pycache__", "test", ".")) and d != "codex-audit"]

            for file in files:
                if file.endswith(".py") and not file.startswith("."):
                    python_files.append(Path(root) / file)
        return python_files

    def build_dependency_graphs(self):
        """Build module and package dependency graphs"""
        print("Building dependency graphs...")

        python_files = self.get_python_files()
        module_dependencies = defaultdict(set)
        package_dependencies = defaultdict(set)

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)
                module_name = self.get_module_name(file_path)
                package_name = self.get_package_name(file_path)

                # Extract imports
                imports = self.extract_imports(tree, module_name)

                for imported_module in imports:
                    if imported_module.startswith("packages."):
                        # Module-level dependency
                        module_dependencies[module_name].add(imported_module)
                        self.dependency_graph.add_edge(module_name, imported_module)

                        # Package-level dependency
                        imported_package = imported_module.split(".")[1] if "." in imported_module else imported_module
                        if imported_package != package_name:
                            package_dependencies[package_name].add(imported_package)
                            self.package_graph.add_edge(package_name, imported_package)

            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")
                continue

    def extract_imports(self, tree: ast.AST, current_module: str) -> set[str]:
        """Extract all import statements from AST"""
        imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    if node.module.startswith("."):
                        # Resolve relative import
                        resolved = self.resolve_relative_import(current_module, node.module)
                        if resolved:
                            imports.add(resolved)
                    else:
                        imports.add(node.module)

        return imports

    def resolve_relative_import(self, from_module: str, relative_import: str) -> str | None:
        """Resolve relative imports to absolute module names"""
        try:
            from_parts = from_module.split(".")
            dots = len(relative_import) - len(relative_import.lstrip("."))
            import_part = relative_import[dots:]

            if dots == 1:
                base_parts = from_parts[:-1]
            else:
                base_parts = from_parts[: -(dots - 1)]

            if import_part:
                return ".".join(base_parts + [import_part])
            else:
                return ".".join(base_parts)
        except:
            return None

    def get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name"""
        rel_path = file_path.relative_to(self.project_root)
        parts = list(rel_path.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]  # Remove .py extension
        return ".".join(parts)

    def get_package_name(self, file_path: Path) -> str:
        """Get package name from file path"""
        rel_path = file_path.relative_to(self.packages_dir)
        return rel_path.parts[0] if rel_path.parts else ""

    def calculate_coupling_metrics(self):
        """Calculate coupling metrics for all modules"""
        print("Calculating coupling metrics...")

        for module in self.dependency_graph.nodes():
            if not module.startswith("packages."):
                continue

            # Efferent coupling (dependencies going out)
            efferent = len(list(self.dependency_graph.successors(module)))

            # Afferent coupling (dependencies coming in)
            afferent = len(list(self.dependency_graph.predecessors(module)))

            # Instability (Ce / (Ce + Ca))
            total_coupling = efferent + afferent
            instability = efferent / total_coupling if total_coupling > 0 else 0

            # Abstractness (would need additional analysis)
            abstractness = self.calculate_abstractness(module)

            # Distance from main sequence |A + I - 1|
            distance = abs(abstractness + instability - 1)

            metric = CouplingMetric(
                module=module,
                efferent_coupling=efferent,
                afferent_coupling=afferent,
                instability=instability,
                abstractness=abstractness,
                distance=distance,
            )

            self.coupling_metrics.append(metric)

    def calculate_abstractness(self, module: str) -> float:
        """Calculate abstractness of a module (simplified)"""
        try:
            # Find the corresponding file
            module_path = self.module_name_to_path(module)
            if not module_path or not module_path.exists():
                return 0.0

            with open(module_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            total_classes = 0
            abstract_classes = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    total_classes += 1

                    # Check if class has abstract methods or inherits from ABC
                    for decorator in node.decorator_list:
                        if (isinstance(decorator, ast.Name) and decorator.id == "abstractmethod") or any(
                            base.id == "ABC" for base in node.bases if isinstance(base, ast.Name)
                        ):
                            abstract_classes += 1
                            break

            return abstract_classes / total_classes if total_classes > 0 else 0.0

        except:
            return 0.0

    def module_name_to_path(self, module_name: str) -> Path | None:
        """Convert module name back to file path"""
        if not module_name.startswith("packages."):
            return None

        parts = module_name.split(".")[1:]  # Remove 'packages' prefix
        potential_paths = [
            self.packages_dir / Path(*parts).with_suffix(".py"),
            self.packages_dir / Path(*parts) / "__init__.py",
        ]

        for path in potential_paths:
            if path.exists():
                return path

        return None

    def detect_connascence_violations(self):
        """Detect connascence violations and hotspots"""
        print("Detecting connascence violations...")

        python_files = self.get_python_files()

        # Connascence of Name
        self.detect_name_connascence(python_files)

        # Connascence of Type
        self.detect_type_connascence(python_files)

        # Connascence of Position
        self.detect_position_connascence(python_files)

        # Connascence of Algorithm
        self.detect_algorithm_connascence(python_files)

    def detect_name_connascence(self, files: list[Path]):
        """Detect connascence of name violations"""
        name_usage = defaultdict(list)

        for file_path in files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and not node.id.startswith("_"):
                        context = self.get_context_type(node, tree)
                        name_usage[node.id].append((str(file_path), context))

            except:
                continue

        # Find names used across different modules
        for name, usages in name_usage.items():
            files_used = set(usage[0] for usage in usages)
            contexts_used = set(usage[1] for usage in usages)

            if len(files_used) > 1 and len(contexts_used) > 1:
                severity = self.calculate_connascence_severity(len(files_used), len(contexts_used))

                metric = ConnascenceMetric(
                    type="name",
                    strength="weak",
                    locality="cross_module" if len(files_used) > 1 else "same_module",
                    instances=len(usages),
                    files_affected=list(files_used),
                    severity=severity,
                )

                self.connascence_metrics.append(metric)

    def detect_type_connascence(self, files: list[Path]):
        """Detect connascence of type violations (magic numbers, hardcoded values)"""
        magic_values = defaultdict(list)

        for file_path in files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Num) and isinstance(node.n, int | float):
                        if node.n not in [0, 1, -1, 2]:  # Common acceptable values
                            magic_values[node.n].append(str(file_path))
                    elif isinstance(node, ast.Str):
                        if len(node.s) > 10 and not node.s.startswith(("http://", "https://")):
                            magic_values[f"string:{node.s[:20]}..."].append(str(file_path))

            except:
                continue

        # Find magic values used across multiple files
        for value, files_used in magic_values.items():
            if len(set(files_used)) > 1:
                severity = self.calculate_connascence_severity(len(set(files_used)), 1)

                metric = ConnascenceMetric(
                    type="type",
                    strength="weak",
                    locality="cross_module",
                    instances=len(files_used),
                    files_affected=list(set(files_used)),
                    severity=severity,
                )

                self.connascence_metrics.append(metric)

    def detect_position_connascence(self, files: list[Path]):
        """Detect connascence of position violations (parameter order dependencies)"""
        function_calls = defaultdict(list)

        for file_path in files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        if len(node.args) > 3:  # More than 3 positional arguments
                            function_calls[node.func.id].append(str(file_path))

            except:
                continue

        # Find functions with many positional arguments called from multiple places
        for func_name, files_used in function_calls.items():
            if len(set(files_used)) > 1:
                metric = ConnascenceMetric(
                    type="position",
                    strength="strong",
                    locality="cross_module",
                    instances=len(files_used),
                    files_affected=list(set(files_used)),
                    severity="high",
                )

                self.connascence_metrics.append(metric)

    def detect_algorithm_connascence(self, files: list[Path]):
        """Detect connascence of algorithm violations (duplicated complex logic)"""
        # Simplified: look for similar function structures
        function_signatures = defaultdict(list)

        for file_path in files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Create a simple signature based on structure
                        signature = self.create_function_signature(node)
                        if signature:
                            function_signatures[signature].append((str(file_path), node.name))

            except:
                continue

        # Find potentially duplicated algorithms
        for signature, occurrences in function_signatures.items():
            if len(occurrences) > 1:
                files_affected = list(set(occ[0] for occ in occurrences))
                if len(files_affected) > 1:  # Cross-module duplication
                    metric = ConnascenceMetric(
                        type="algorithm",
                        strength="strong",
                        locality="cross_module",
                        instances=len(occurrences),
                        files_affected=files_affected,
                        severity="medium",
                    )

                    self.connascence_metrics.append(metric)

    def create_function_signature(self, func_node: ast.FunctionDef) -> str | None:
        """Create a simplified signature for algorithm comparison"""
        try:
            # Count different types of statements
            stmt_counts = defaultdict(int)
            for stmt in ast.walk(func_node):
                stmt_counts[type(stmt).__name__] += 1

            # Create signature from statement counts
            if stmt_counts["If"] > 2 or stmt_counts["For"] > 1 or stmt_counts["While"] > 0:
                signature_parts = []
                for stmt_type in ["If", "For", "While", "Try", "Return"]:
                    if stmt_counts[stmt_type] > 0:
                        signature_parts.append(f"{stmt_type}:{stmt_counts[stmt_type]}")
                return ",".join(signature_parts)

            return None
        except:
            return None

    def get_context_type(self, node: ast.Name, tree: ast.AST) -> str:
        """Get the context type where a name is used"""
        # Simplified context detection
        return "unknown"  # Would need more sophisticated AST walking

    def calculate_connascence_severity(self, file_count: int, context_count: int) -> str:
        """Calculate severity of connascence violation"""
        score = file_count * context_count
        if score >= 10:
            return "critical"
        elif score >= 5:
            return "high"
        elif score >= 3:
            return "medium"
        else:
            return "low"

    def detect_architectural_drift(self):
        """Detect architectural drift patterns"""
        print("Detecting architectural drift...")

        # Analyze dependency violations
        self.analyze_dependency_violations()

        # Analyze complexity drift
        self.analyze_complexity_drift()

        # Analyze size drift
        self.analyze_size_drift()

    def analyze_dependency_violations(self):
        """Analyze violations of dependency rules"""
        allowed_deps = self.config.get("allowed_dependencies", {})
        self.config.get("forbidden_dependencies", {})

        for package, dependencies in allowed_deps.items():
            if package not in self.package_graph:
                continue

            actual_deps = set(self.package_graph.successors(package))
            allowed_set = set(dependencies)

            # Find violations
            violations = actual_deps - allowed_set - {package}  # Remove self-dependencies

            if violations:
                drift = ArchitecturalDrift(
                    component=package,
                    drift_type="dependency_violation",
                    severity=len(violations) / len(actual_deps) if actual_deps else 0,
                    description=f"Package {package} has unauthorized dependencies: {violations}",
                    recommendation=f"Remove dependencies on {violations} or update architecture rules",
                    files_affected=self.get_files_in_package(package),
                )
                self.architectural_drift.append(drift)

    def analyze_complexity_drift(self):
        """Analyze complexity drift over acceptable thresholds"""
        max_complexity = self.config.get("max_function_complexity", 10)
        python_files = self.get_python_files()

        high_complexity_files = []

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                complexity_data = cc.cc_visit(content)
                high_complexity_count = sum(1 for item in complexity_data if item.complexity > max_complexity)

                if high_complexity_count > 0:
                    high_complexity_files.append(str(file_path))

            except:
                continue

        if high_complexity_files:
            severity = len(high_complexity_files) / len(python_files)
            drift = ArchitecturalDrift(
                component="codebase",
                drift_type="complexity_drift",
                severity=severity,
                description=f"{len(high_complexity_files)} files exceed complexity thresholds",
                recommendation="Refactor complex functions into smaller, more focused units",
                files_affected=high_complexity_files,
            )
            self.architectural_drift.append(drift)

    def analyze_size_drift(self):
        """Analyze file size drift"""
        max_lines = self.config.get("max_file_lines", 500)
        python_files = self.get_python_files()

        large_files = []

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    line_count = sum(1 for _ in f)

                if line_count > max_lines:
                    large_files.append(str(file_path))

            except:
                continue

        if large_files:
            severity = len(large_files) / len(python_files)
            drift = ArchitecturalDrift(
                component="codebase",
                drift_type="size_drift",
                severity=severity,
                description=f"{len(large_files)} files exceed size limits",
                recommendation="Split large files into smaller, more cohesive modules",
                files_affected=large_files,
            )
            self.architectural_drift.append(drift)

    def get_files_in_package(self, package: str) -> list[str]:
        """Get list of files in a package"""
        package_dir = self.packages_dir / package
        files = []
        if package_dir.exists():
            for file_path in package_dir.rglob("*.py"):
                files.append(str(file_path))
        return files

    def calculate_technical_debt(self):
        """Calculate technical debt metrics"""
        print("Calculating technical debt...")

        python_files = self.get_python_files()

        total_maintainability = 0
        total_files = 0
        debt_items = []

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Calculate maintainability index
                mi = rm.mi_visit(content, multi=True)
                maintainability = mi.mi if hasattr(mi, "mi") else 0

                total_maintainability += maintainability
                total_files += 1

                # Low maintainability indicates technical debt
                if maintainability < self.config.get("quality_thresholds", {}).get("maintainability_index", 70):
                    effort_hours = (70 - maintainability) * 0.5  # Simplified calculation

                    debt = TechnicalDebt(
                        category="maintainability",
                        amount=70 - maintainability,
                        location=str(file_path),
                        description=f"Low maintainability index: {maintainability:.1f}",
                        effort_hours=effort_hours,
                        risk_level="medium" if maintainability > 50 else "high",
                    )
                    debt_items.append(debt)

            except:
                continue

        # Calculate overall technical debt ratio
        avg_maintainability = total_maintainability / total_files if total_files > 0 else 0
        debt_ratio = max(0, 100 - avg_maintainability)

        if debt_ratio > self.config.get("quality_thresholds", {}).get("technical_debt_ratio", 5):
            debt = TechnicalDebt(
                category="overall",
                amount=debt_ratio,
                location="codebase",
                description=f"Overall technical debt ratio: {debt_ratio:.1f}%",
                effort_hours=debt_ratio * 2,  # Simplified calculation
                risk_level="high" if debt_ratio > 20 else "medium",
            )
            debt_items.append(debt)

        self.technical_debt = debt_items

    def generate_visualizations(self, output_dir: Path):
        """Generate architectural visualizations"""
        print("Generating visualizations...")

        # Dependency graph visualization
        self.visualize_dependency_graph(output_dir / "dependency_graph.png")

        # Coupling metrics visualization
        self.visualize_coupling_metrics(output_dir / "coupling_metrics.png")

        # Connascence heatmap
        self.visualize_connascence_heatmap(output_dir / "connascence_heatmap.png")

        # Technical debt distribution
        self.visualize_technical_debt(output_dir / "technical_debt.png")

    def visualize_dependency_graph(self, output_path: Path):
        """Create dependency graph visualization"""
        if not self.package_graph.nodes():
            return

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.package_graph, k=2, iterations=50)

        # Draw nodes
        nx.draw_networkx_nodes(self.package_graph, pos, node_color="lightblue", node_size=1000, alpha=0.7)

        # Draw edges
        nx.draw_networkx_edges(self.package_graph, pos, edge_color="gray", arrows=True, arrowsize=20, alpha=0.5)

        # Draw labels
        nx.draw_networkx_labels(self.package_graph, pos, font_size=10, font_weight="bold")

        plt.title("Package Dependency Graph")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def visualize_coupling_metrics(self, output_path: Path):
        """Create coupling metrics visualization"""
        if not self.coupling_metrics:
            return

        # Create DataFrame
        df = pd.DataFrame([asdict(metric) for metric in self.coupling_metrics])

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Instability vs Abstractness (Martin's main sequence)
        axes[0, 0].scatter(df["abstractness"], df["instability"], alpha=0.7)
        axes[0, 0].plot([0, 1], [1, 0], "r--", alpha=0.5, label="Main Sequence")
        axes[0, 0].set_xlabel("Abstractness")
        axes[0, 0].set_ylabel("Instability")
        axes[0, 0].set_title("Main Sequence Diagram")
        axes[0, 0].legend()

        # Distance from main sequence
        axes[0, 1].hist(df["distance"], bins=20, alpha=0.7)
        axes[0, 1].set_xlabel("Distance from Main Sequence")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Distance Distribution")

        # Efferent vs Afferent coupling
        axes[1, 0].scatter(df["efferent_coupling"], df["afferent_coupling"], alpha=0.7)
        axes[1, 0].set_xlabel("Efferent Coupling (Ce)")
        axes[1, 0].set_ylabel("Afferent Coupling (Ca)")
        axes[1, 0].set_title("Coupling Scatter Plot")

        # Top modules by coupling
        top_coupled = df.nlargest(10, "efferent_coupling")
        axes[1, 1].barh(range(len(top_coupled)), top_coupled["efferent_coupling"])
        axes[1, 1].set_yticks(range(len(top_coupled)))
        axes[1, 1].set_yticklabels([m.split(".")[-1] for m in top_coupled["module"]])
        axes[1, 1].set_xlabel("Efferent Coupling")
        axes[1, 1].set_title("Most Coupled Modules")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def visualize_connascence_heatmap(self, output_path: Path):
        """Create connascence heatmap"""
        if not self.connascence_metrics:
            return

        # Create connascence matrix
        connascence_types = ["name", "type", "position", "algorithm"]
        severity_levels = ["low", "medium", "high", "critical"]

        matrix = np.zeros((len(connascence_types), len(severity_levels)))

        for metric in self.connascence_metrics:
            type_idx = connascence_types.index(metric.type) if metric.type in connascence_types else 0
            sev_idx = severity_levels.index(metric.severity) if metric.severity in severity_levels else 0
            matrix[type_idx, sev_idx] += metric.instances

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            matrix, xticklabels=severity_levels, yticklabels=connascence_types, annot=True, fmt="g", cmap="YlOrRd"
        )

        plt.title("Connascence Violations Heatmap")
        plt.xlabel("Severity Level")
        plt.ylabel("Connascence Type")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def visualize_technical_debt(self, output_path: Path):
        """Create technical debt visualization"""
        if not self.technical_debt:
            return

        df = pd.DataFrame([asdict(debt) for debt in self.technical_debt])

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Debt by category
        category_debt = df.groupby("category")["amount"].sum().sort_values(ascending=True)
        axes[0, 0].barh(category_debt.index, category_debt.values)
        axes[0, 0].set_xlabel("Technical Debt Amount")
        axes[0, 0].set_title("Technical Debt by Category")

        # Debt by risk level
        risk_counts = df["risk_level"].value_counts()
        axes[0, 1].pie(risk_counts.values, labels=risk_counts.index, autopct="%1.1f%%")
        axes[0, 1].set_title("Technical Debt by Risk Level")

        # Effort hours distribution
        axes[1, 0].hist(df["effort_hours"], bins=20, alpha=0.7)
        axes[1, 0].set_xlabel("Effort Hours")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Effort Hours Distribution")

        # Debt amount vs effort hours
        axes[1, 1].scatter(df["amount"], df["effort_hours"], alpha=0.7)
        axes[1, 1].set_xlabel("Debt Amount")
        axes[1, 1].set_ylabel("Effort Hours")
        axes[1, 1].set_title("Debt Amount vs Effort Required")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def evaluate_quality_gates(self) -> dict[str, bool]:
        """Evaluate architectural quality gates"""
        gates = {}

        # Coupling threshold gate
        high_coupling_count = sum(
            1 for metric in self.coupling_metrics if metric.instability > self.config.get("max_coupling_threshold", 0.3)
        )
        gates["coupling_threshold"] = high_coupling_count == 0

        # Circular dependencies gate
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            gates["no_circular_dependencies"] = len(cycles) == 0
        except:
            gates["no_circular_dependencies"] = True

        # Critical connascence gate
        critical_connascence = sum(1 for metric in self.connascence_metrics if metric.severity == "critical")
        gates["no_critical_connascence"] = critical_connascence == 0

        # Technical debt gate
        high_debt_count = sum(1 for debt in self.technical_debt if debt.risk_level == "high")
        gates["technical_debt_acceptable"] = high_debt_count < 5

        # Architectural drift gate
        critical_drift_count = sum(1 for drift in self.architectural_drift if drift.severity > 0.5)
        gates["no_critical_drift"] = critical_drift_count == 0

        return gates

    def generate_recommendations(self) -> list[str]:
        """Generate architectural improvement recommendations"""
        recommendations = []

        # Coupling recommendations
        high_coupling_modules = [m for m in self.coupling_metrics if m.instability > 0.7]
        if high_coupling_modules:
            recommendations.append(
                f"Reduce coupling in {len(high_coupling_modules)} highly unstable modules: "
                f"{', '.join([m.module.split('.')[-1] for m in high_coupling_modules[:3]])}"
            )

        # Connascence recommendations
        critical_connascence = [m for m in self.connascence_metrics if m.severity == "critical"]
        if critical_connascence:
            recommendations.append(
                f"Address {len(critical_connascence)} critical connascence violations, "
                f"particularly {critical_connascence[0].type} connascence"
            )

        # Technical debt recommendations
        high_debt = [d for d in self.technical_debt if d.risk_level == "high"]
        if high_debt:
            total_effort = sum(d.effort_hours for d in high_debt)
            recommendations.append(
                f"Address {len(high_debt)} high-risk technical debt items "
                f"(estimated {total_effort:.1f} hours effort)"
            )

        # Architectural drift recommendations
        for drift in self.architectural_drift:
            if drift.severity > 0.3:
                recommendations.append(drift.recommendation)

        return recommendations

    def run_full_analysis(self) -> ArchitecturalReport:
        """Run complete architectural analysis"""
        print("Starting comprehensive architectural analysis...")

        # Build dependency graphs
        self.build_dependency_graphs()

        # Calculate metrics
        self.calculate_coupling_metrics()
        self.detect_connascence_violations()
        self.detect_architectural_drift()
        self.calculate_technical_debt()

        # Evaluate quality gates
        quality_gates = self.evaluate_quality_gates()

        # Generate recommendations
        recommendations = self.generate_recommendations()

        # Create summary
        summary = {
            "total_modules": len(self.dependency_graph.nodes()),
            "total_packages": len(self.package_graph.nodes()),
            "total_dependencies": len(self.dependency_graph.edges()),
            "average_coupling": np.mean([m.instability for m in self.coupling_metrics]) if self.coupling_metrics else 0,
            "connascence_violations": len(self.connascence_metrics),
            "critical_violations": sum(1 for m in self.connascence_metrics if m.severity == "critical"),
            "technical_debt_items": len(self.technical_debt),
            "architectural_drift_items": len(self.architectural_drift),
            "quality_gates_passed": sum(quality_gates.values()),
            "quality_gates_total": len(quality_gates),
        }

        # Create report
        report = ArchitecturalReport(
            timestamp=datetime.now().isoformat(),
            summary=summary,
            dependency_metrics={
                "total_modules": summary["total_modules"],
                "total_dependencies": summary["total_dependencies"],
                "circular_dependencies": len(list(nx.simple_cycles(self.dependency_graph)))
                if self.dependency_graph.nodes()
                else 0,
            },
            coupling_metrics=self.coupling_metrics,
            connascence_metrics=self.connascence_metrics,
            architectural_drift=self.architectural_drift,
            technical_debt=self.technical_debt,
            quality_gates=quality_gates,
            recommendations=recommendations,
            trend_data={},  # Would be populated from historical data
        )

        return report

    def save_report(self, report: ArchitecturalReport, output_dir: Path, format_type: str = "json"):
        """Save architectural report in specified format"""
        output_dir.mkdir(parents=True, exist_ok=True)

        if format_type in ["json", "both"]:
            json_path = output_dir / f"architecture_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_path, "w") as f:
                json.dump(asdict(report), f, indent=2, default=str)
            print(f"JSON report saved to: {json_path}")

        if format_type in ["html", "both"]:
            html_path = output_dir / f"architecture_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            self.generate_html_report(report, html_path)
            print(f"HTML report saved to: {html_path}")

    def generate_html_report(self, report: ArchitecturalReport, output_path: Path):
        """Generate HTML report"""
        template_str = """
<!DOCTYPE html>
<html>
<head>
    <title>AIVillage Architecture Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }
        .violation { background-color: #ffebee; padding: 10px; margin: 5px 0; border-radius: 3px; }
        .recommendation { background-color: #e8f5e8; padding: 10px; margin: 5px 0; border-radius: 3px; }
        .passed { color: green; font-weight: bold; }
        .failed { color: red; font-weight: bold; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>AIVillage Architecture Report</h1>
        <p>Generated: {{ report.timestamp }}</p>
    </div>

    <div class="section">
        <h2>Summary</h2>
        <div class="metric">Total Modules: {{ report.summary.total_modules }}</div>
        <div class="metric">Total Packages: {{ report.summary.total_packages }}</div>
        <div class="metric">Dependencies: {{ report.summary.total_dependencies }}</div>
        <div class="metric">Quality Gates: {{ report.summary.quality_gates_passed }}/{{ report.summary.quality_gates_total }}</div>
    </div>

    <div class="section">
        <h2>Quality Gates</h2>
        {% for gate, passed in report.quality_gates.items() %}
        <div class="metric {% if passed %}passed{% else %}failed{% endif %}">
            {{ gate.replace('_', ' ').title() }}: {{ 'PASS' if passed else 'FAIL' }}
        </div>
        {% endfor %}
    </div>

    <div class="section">
        <h2>Coupling Metrics</h2>
        <table>
            <tr>
                <th>Module</th>
                <th>Efferent Coupling</th>
                <th>Afferent Coupling</th>
                <th>Instability</th>
                <th>Distance from Main Sequence</th>
            </tr>
            {% for metric in report.coupling_metrics[:10] %}
            <tr>
                <td>{{ metric.module.split('.')[-1] }}</td>
                <td>{{ metric.efferent_coupling }}</td>
                <td>{{ metric.afferent_coupling }}</td>
                <td>{{ "%.2f"|format(metric.instability) }}</td>
                <td>{{ "%.2f"|format(metric.distance) }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="section">
        <h2>Connascence Violations</h2>
        {% for violation in report.connascence_metrics %}
        <div class="violation">
            <strong>{{ violation.type.title() }} Connascence</strong> - {{ violation.severity.title() }} Severity<br>
            {{ violation.instances }} instances across {{ violation.files_affected|length }} files
        </div>
        {% endfor %}
    </div>

    <div class="section">
        <h2>Technical Debt</h2>
        {% for debt in report.technical_debt %}
        <div class="violation">
            <strong>{{ debt.category.title() }}</strong> - {{ debt.risk_level.title() }} Risk<br>
            {{ debt.description }}<br>
            Estimated effort: {{ "%.1f"|format(debt.effort_hours) }} hours
        </div>
        {% endfor %}
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        {% for rec in report.recommendations %}
        <div class="recommendation">{{ rec }}</div>
        {% endfor %}
    </div>
</body>
</html>
        """

        from jinja2 import Template

        template = Template(template_str)
        html_content = template.render(report=report)

        with open(output_path, "w") as f:
            f.write(html_content)


def main():
    parser = argparse.ArgumentParser(description="Architectural Analysis Tool")
    parser.add_argument("--output-dir", default="reports/architecture", help="Output directory for reports")
    parser.add_argument("--format", choices=["json", "html", "both"], default="both", help="Report format")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--visualizations", action="store_true", help="Generate visualization graphs")

    args = parser.parse_args()

    # Initialize analyzer
    config_file = Path(args.config) if args.config else None
    analyzer = ArchitecturalAnalyzer(PROJECT_ROOT, config_file)

    # Run analysis
    report = analyzer.run_full_analysis()

    # Save report
    output_dir = Path(args.output_dir)
    analyzer.save_report(report, output_dir, args.format)

    # Generate visualizations if requested
    if args.visualizations:
        analyzer.generate_visualizations(output_dir)

    # Print summary
    print("\nArchitectural Analysis Summary:")
    print(f"Quality Gates: {report.summary['quality_gates_passed']}/{report.summary['quality_gates_total']} passed")
    print(f"Critical Issues: {report.summary['critical_violations']} connascence violations")
    print(f"Technical Debt Items: {report.summary['technical_debt_items']}")
    print(f"Recommendations: {len(report.recommendations)}")

    # Exit code based on quality gates
    if report.summary["quality_gates_passed"] == report.summary["quality_gates_total"]:
        print("\n✅ All architectural quality gates passed!")
        sys.exit(0)
    else:
        print("\n❌ Some architectural quality gates failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
