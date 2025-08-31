#!/usr/bin/env python3
"""
Continuous Architecture Monitoring and Analysis Script - Refactored

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
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any

import networkx as nx
import radon.complexity as cc
import radon.metrics as rm

# Configure logging
logger = logging.getLogger(__name__)

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


@dataclass
class CouplingMetric:
    """Represents coupling measurement between modules"""

    from_module: str
    to_module: str
    coupling_type: str  # data, control, content, common, external
    strength: float
    violations: list[str]


@dataclass
class TechnicalDebt:
    """Represents technical debt item"""

    debt_type: str
    severity: str
    location: str
    description: str
    estimated_effort_hours: float
    impact: str


@dataclass
class ArchitecturalReport:
    """Complete architectural analysis report"""

    timestamp: str
    project_root: str
    summary: dict[str, Any]
    dependency_analysis: dict[str, Any]
    coupling_metrics: list[CouplingMetric]
    connascence_metrics: list[ConnascenceMetric]
    technical_debt: list[TechnicalDebt]
    quality_gates: dict[str, bool]
    recommendations: list[str]
    trend_data: dict[str, list[float]]


class DependencyAnalyzer:
    """Handles dependency graph generation and analysis."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.dependency_graph = nx.DiGraph()
        self.package_graph = nx.DiGraph()

    def build_dependency_graph(self) -> nx.DiGraph:
        """Build module-level dependency graph."""
        python_files = self._get_python_files()

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

    def analyze_circular_dependencies(self) -> list[list[str]]:
        """Find circular dependencies in the dependency graph."""
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            return cycles
        except nx.NetworkXError:
            return []

    def calculate_dependency_metrics(self) -> dict[str, Any]:
        """Calculate various dependency-related metrics."""
        if not self.dependency_graph:
            return {}

        return {
            "total_modules": self.dependency_graph.number_of_nodes(),
            "total_dependencies": self.dependency_graph.number_of_edges(),
            "average_dependencies_per_module": self.dependency_graph.number_of_edges()
            / max(1, self.dependency_graph.number_of_nodes()),
            "circular_dependencies": len(self.analyze_circular_dependencies()),
            "strongly_connected_components": len(list(nx.strongly_connected_components(self.dependency_graph))),
        }

    def _get_python_files(self) -> list[Path]:
        """Get all Python files in the project."""
        return list(self.project_root.rglob("*.py"))

    def _should_skip_file(self, file_path: Path) -> bool:
        skip_patterns = ["__pycache__", ".git", "build", "dist", "deprecated", "archive"]
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
        except Exception as e:
            logger.debug(f"Failed to parse imports from file {file_path}: {e}")
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

    def get_connascence_hotspots(self) -> list[dict[str, Any]]:
        """Identify files with high connascence violations."""
        file_violations = defaultdict(int)
        for metric in self.connascence_metrics:
            for file_path in metric.files_affected:
                file_violations[file_path] += metric.instances

        hotspots = [
            {"file": file_path, "violations": count}
            for file_path, count in sorted(file_violations.items(), key=lambda x: x[1], reverse=True)
            if count > 10
        ]

        return hotspots[:10]  # Top 10 hotspots

    def _should_skip_file(self, file_path: Path) -> bool:
        skip_patterns = ["__pycache__", ".git", "build", "dist", "deprecated", "archive"]
        return any(pattern in str(file_path) for pattern in skip_patterns)

    def _analyze_file_connascence(self, file_path: Path):
        """Analyze connascence patterns in a single file."""
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
                tree = ast.parse(content)

            # Analyze different types of connascence
            self._detect_name_connascence(tree, file_path)
            self._detect_meaning_connascence(tree, file_path)
            self._detect_position_connascence(tree, file_path)

        except Exception as e:
            logger.debug(f"Failed to analyze connascence in file {file_path}: {e}")

    def _detect_name_connascence(self, tree: ast.AST, file_path: Path):
        """Detect connascence of name."""
        names = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                name = node.id
                names[name] = names.get(name, 0) + 1

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


class MetricsCalculator:
    """Handles calculation of various code quality metrics."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def calculate_complexity_metrics(self) -> dict[str, Any]:
        """Calculate complexity metrics across the codebase."""
        python_files = list(self.project_root.rglob("*.py"))
        total_complexity = 0
        file_count = 0
        high_complexity_files = []

        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue

            try:
                complexity = self._calculate_file_complexity(file_path)
                if complexity:
                    total_complexity += complexity
                    file_count += 1
                    if complexity > 50:
                        high_complexity_files.append({"file": str(file_path), "complexity": complexity})
            except Exception:
                continue

        return {
            "average_complexity": total_complexity / max(1, file_count),
            "total_files_analyzed": file_count,
            "high_complexity_files": high_complexity_files[:10],
        }

    def calculate_maintainability_index(self) -> dict[str, Any]:
        """Calculate maintainability index for files."""
        python_files = list(self.project_root.rglob("*.py"))
        maintainability_scores = []

        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue

            try:
                score = self._calculate_file_maintainability(file_path)
                if score is not None:
                    maintainability_scores.append({"file": str(file_path), "score": score})
            except Exception:
                continue

        if maintainability_scores:
            scores = [item["score"] for item in maintainability_scores]
            return {
                "average_maintainability": sum(scores) / len(scores),
                "files_below_threshold": [item for item in maintainability_scores if item["score"] < 70][:10],
            }
        return {}

    def _should_skip_file(self, file_path: Path) -> bool:
        skip_patterns = ["__pycache__", ".git", "build", "dist", "deprecated", "archive"]
        return any(pattern in str(file_path) for pattern in skip_patterns)

    def _calculate_file_complexity(self, file_path: Path) -> float:
        """Calculate cyclomatic complexity for a file."""
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            complexity_results = cc.cc_visit(content)
            return sum(result.complexity for result in complexity_results)
        except Exception:
            return 0.0

    def _calculate_file_maintainability(self, file_path: Path) -> float:
        """Calculate maintainability index for a file."""
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            mi_results = rm.mi_visit(content, True)
            return mi_results
        except Exception:
            return None


class ReportGenerator:
    """Generates architectural analysis reports."""

    def __init__(self, project_root: Path, reports_dir: Path):
        self.project_root = project_root
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_comprehensive_report(
        self,
        dependency_metrics: dict,
        connascence_metrics: list[ConnascenceMetric],
        complexity_metrics: dict,
        maintainability_metrics: dict,
    ) -> ArchitecturalReport:
        """Generate comprehensive architectural analysis report."""

        # Generate summary
        summary = self._generate_summary(
            dependency_metrics, connascence_metrics, complexity_metrics, maintainability_metrics
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            dependency_metrics, connascence_metrics, complexity_metrics, maintainability_metrics
        )

        report = ArchitecturalReport(
            timestamp=datetime.now().isoformat(),
            project_root=str(self.project_root),
            summary=summary,
            dependency_analysis=dependency_metrics,
            coupling_metrics=[],
            connascence_metrics=connascence_metrics,
            technical_debt=[],
            quality_gates=self._evaluate_quality_gates(summary),
            recommendations=recommendations,
            trend_data={},
        )

        return report

    def save_report(self, report: ArchitecturalReport, format_type: str = "json"):
        """Save report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format_type in ["json", "both"]:
            json_file = self.reports_dir / f"architectural_analysis_{timestamp}.json"
            with open(json_file, "w") as f:
                json.dump(asdict(report), f, indent=2, default=str)
            print(f"JSON report saved to: {json_file}")

        if format_type in ["html", "both"]:
            html_file = self.reports_dir / f"architectural_analysis_{timestamp}.html"
            self._generate_html_report(report, html_file)
            print(f"HTML report saved to: {html_file}")

    def _generate_summary(
        self, dependency_metrics, connascence_metrics, complexity_metrics, maintainability_metrics
    ) -> dict:
        """Generate analysis summary."""
        return {
            "total_modules": dependency_metrics.get("total_modules", 0),
            "circular_dependencies": dependency_metrics.get("circular_dependencies", 0),
            "connascence_violations": len(connascence_metrics),
            "average_complexity": complexity_metrics.get("average_complexity", 0),
            "average_maintainability": maintainability_metrics.get("average_maintainability", 0),
            "files_needing_attention": len(complexity_metrics.get("high_complexity_files", [])),
        }

    def _generate_recommendations(
        self, dependency_metrics, connascence_metrics, complexity_metrics, maintainability_metrics
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if dependency_metrics.get("circular_dependencies", 0) > 0:
            recommendations.append("Break circular dependencies to improve maintainability")

        if len(connascence_metrics) > 50:
            recommendations.append("Reduce connascence violations by refactoring toward weaker forms")

        high_complexity_files = complexity_metrics.get("high_complexity_files", [])
        if len(high_complexity_files) > 10:
            recommendations.append("Refactor high-complexity files to improve readability")

        low_maintainability = maintainability_metrics.get("files_below_threshold", [])
        if len(low_maintainability) > 10:
            recommendations.append("Focus on improving maintainability index for critical files")

        return recommendations

    def _evaluate_quality_gates(self, summary: dict) -> dict[str, bool]:
        """Evaluate quality gates based on thresholds."""
        return {
            "no_circular_dependencies": summary.get("circular_dependencies", 1) == 0,
            "acceptable_complexity": summary.get("average_complexity", 100) < 20,
            "good_maintainability": summary.get("average_maintainability", 0) > 70,
            "limited_connascence_violations": summary.get("connascence_violations", 100) < 50,
        }

    def _generate_html_report(self, report: ArchitecturalReport, output_file: Path):
        """Generate HTML report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Architectural Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .quality-gate {{ margin: 10px 0; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Architectural Analysis Report</h1>
    <p>Generated: {report.timestamp}</p>
    <p>Project: {report.project_root}</p>

    <div class="summary">
        <h2>Summary</h2>
        <p>Total Modules: {report.summary.get('total_modules', 0)}</p>
        <p>Circular Dependencies: {report.summary.get('circular_dependencies', 0)}</p>
        <p>Connascence Violations: {report.summary.get('connascence_violations', 0)}</p>
        <p>Average Complexity: {report.summary.get('average_complexity', 0):.2f}</p>
        <p>Average Maintainability: {report.summary.get('average_maintainability', 0):.2f}</p>
    </div>

    <h2>Quality Gates</h2>
    {"".join([
        f'<div class="quality-gate"><span class="{"pass" if passed else "fail"}">'
        f'{"✅" if passed else "❌"} {gate_name}</span></div>'
        for gate_name, passed in report.quality_gates.items()
    ])}

    <h2>Recommendations</h2>
    <ul>
        {"".join([f"<li>{rec}</li>" for rec in report.recommendations])}
    </ul>
</body>
</html>
"""
        with open(output_file, "w") as f:
            f.write(html_content)


class ArchitecturalAnalyzer:
    """Main architectural analysis coordinator - simplified facade."""

    def __init__(self, project_root: Path, config_file: Path | None = None):
        self.project_root = project_root
        self.reports_dir = project_root / "reports" / "architecture"

        # Initialize component analyzers
        self.dependency_analyzer = DependencyAnalyzer(project_root)
        self.connascence_analyzer = ConnascenceAnalyzer(project_root)
        self.metrics_calculator = MetricsCalculator(project_root)
        self.report_generator = ReportGenerator(project_root, self.reports_dir)

    def run_full_analysis(self, output_format: str = "json") -> ArchitecturalReport:
        """Run comprehensive architectural analysis."""
        print("Starting comprehensive architectural analysis...")

        # Build dependency graph
        print("Analyzing dependencies...")
        self.dependency_analyzer.build_dependency_graph()
        dependency_metrics = self.dependency_analyzer.calculate_dependency_metrics()

        # Analyze connascence
        print("Analyzing connascence patterns...")
        connascence_metrics = self.connascence_analyzer.analyze_connascence()

        # Calculate complexity and maintainability metrics
        print("Calculating complexity metrics...")
        complexity_metrics = self.metrics_calculator.calculate_complexity_metrics()

        print("Calculating maintainability metrics...")
        maintainability_metrics = self.metrics_calculator.calculate_maintainability_index()

        # Generate comprehensive report
        print("Generating report...")
        report = self.report_generator.generate_comprehensive_report(
            dependency_metrics, connascence_metrics, complexity_metrics, maintainability_metrics
        )

        # Save report
        self.report_generator.save_report(report, output_format)

        print("Analysis complete!")
        return report


def main():
    """Main entry point for architectural analysis."""
    parser = argparse.ArgumentParser(description="Architectural Analysis Tool")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root directory")
    parser.add_argument("--output-format", choices=["json", "html", "both"], default="json", help="Output format")
    parser.add_argument("--config", type=Path, help="Configuration file path")

    args = parser.parse_args()

    # Run analysis
    analyzer = ArchitecturalAnalyzer(args.project_root, args.config)
    report = analyzer.run_full_analysis(args.output_format)

    # Print summary
    print("\n" + "=" * 60)
    print("ARCHITECTURAL ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total Modules: {report.summary.get('total_modules', 0)}")
    print(f"Circular Dependencies: {report.summary.get('circular_dependencies', 0)}")
    print(f"Connascence Violations: {report.summary.get('connascence_violations', 0)}")
    print(f"Files Needing Attention: {report.summary.get('files_needing_attention', 0)}")

    print("\nQuality Gates:")
    for gate_name, passed in report.quality_gates.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {gate_name}")

    if report.recommendations:
        print("\nTop Recommendations:")
        for i, rec in enumerate(report.recommendations[:5], 1):
            print(f"  {i}. {rec}")


if __name__ == "__main__":
    main()
