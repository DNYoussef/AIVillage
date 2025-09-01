#!/usr/bin/env python3
"""
Coupling Metrics Analyzer for AIVillage

This tool tracks various coupling metrics based on connascence principles:
- Positional parameter ratio (% functions with >3 positional params)
- Magic literal density (count of magic numbers/strings in conditionals)
- Duplicate algorithm count (implementations vs shared APIs)
- Global reference count (singletons/globals usage)
- God class detection (>500 LOC or >20 methods)
- Cross-module coupling violations

Provides quantitative metrics for tracking code quality improvements.
"""

import argparse
import ast
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any


@dataclass
class CouplingMetrics:
    """Container for all coupling metrics."""

    # File-level metrics
    file_path: str
    total_functions: int
    total_classes: int
    total_imports: int
    lines_of_code: int

    # Connascence metrics
    positional_param_violations: int
    positional_param_ratio: float
    magic_literal_count: int
    magic_literal_density: float  # per 100 LOC
    duplicate_algorithm_count: int
    global_usage_count: int
    god_classes: list[str]

    # Cross-module coupling
    import_coupling_score: float
    circular_dependency_risk: int

    # Quality indicators
    coupling_score: float  # 0-100, lower is better
    maintainability_index: float  # 0-100, higher is better


@dataclass
class ProjectMetrics:
    """Project-wide coupling metrics."""

    total_files: int
    total_loc: int
    average_coupling_score: float
    worst_coupled_files: list[tuple[str, float]]

    # Aggregate metrics
    total_positional_violations: int
    total_magic_literals: int
    total_god_classes: int
    total_global_usage: int

    # Ratios and densities
    overall_positional_ratio: float
    overall_magic_density: float
    god_class_ratio: float

    # Trends (if comparing to previous run)
    improvement_score: float | None = None
    regression_areas: list[str] = None


class CouplingAnalyzer(ast.NodeVisitor):
    """AST visitor for analyzing coupling metrics."""

    def __init__(self, file_path: str, source_lines: list[str]):
        self.file_path = file_path
        self.source_lines = source_lines
        self.lines_of_code = len([line for line in source_lines if line.strip() and not line.strip().startswith("#")])

        # Counters
        self.function_count = 0
        self.class_count = 0
        self.import_count = 0
        self.positional_violations = 0
        self.magic_literals = 0
        self.magic_in_conditionals = 0
        self.global_usage = 0
        self.god_classes = []

        # Algorithm tracking
        self.function_bodies = {}
        self.duplicate_algorithms = 0

        # Import tracking for coupling analysis
        self.imports = set()
        self.from_imports = defaultdict(list)

        # Complexity tracking
        self.complexity_scores = []

    def _get_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, ast.If | ast.While | ast.For | ast.AsyncFor):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.And | ast.Or):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += 1

        return complexity

    def _normalize_function_for_duplication(self, node: ast.FunctionDef) -> str:
        """Create a normalized representation for duplicate detection."""
        # Extract structural patterns
        patterns = []
        for stmt in node.body:
            if isinstance(stmt, ast.Return):
                patterns.append("RETURN")
            elif isinstance(stmt, ast.If):
                patterns.append("IF")
            elif isinstance(stmt, ast.For):
                patterns.append("FOR")
            elif isinstance(stmt, ast.While):
                patterns.append("WHILE")
            elif isinstance(stmt, ast.Assign):
                patterns.append("ASSIGN")
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                patterns.append("CALL")
            elif isinstance(stmt, ast.Try):
                patterns.append("TRY")
            elif isinstance(stmt, ast.With):
                patterns.append("WITH")

        # Include length information for better matching
        pattern_str = "|".join(patterns)
        return f"{len(patterns)}:{pattern_str}"

    def _is_magic_literal(self, value: Any) -> bool:
        """Determine if a literal value is a magic number/string."""
        if isinstance(value, int | float):
            # Common non-magic numbers
            return value not in [0, 1, -1, 2, 10, 100, 1000, 0.0, 1.0, -1.0]
        elif isinstance(value, str):
            # Skip empty strings, single chars, and simple patterns
            if len(value) <= 1:
                return False
            # Skip simple alphanumeric strings
            if re.match(r"^[a-zA-Z0-9_-]+$", value) and len(value) < 10:
                return False
            return True
        return False

    def _is_in_conditional_context(self, node: ast.AST) -> bool:
        """Check if node is in a conditional context (simplified)."""
        # In a real implementation, you'd walk up the AST
        line_content = self.source_lines[node.lineno - 1] if node.lineno <= len(self.source_lines) else ""
        return any(keyword in line_content for keyword in ["if ", "elif ", "while ", "assert ", "return "])

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Analyze function definitions for coupling metrics."""
        self.function_count += 1

        # Check positional parameters
        positional_args = [arg for arg in node.args.args if not arg.arg.startswith("_")]
        if len(positional_args) > 3:
            self.positional_violations += 1

        # Calculate complexity
        complexity = self._get_function_complexity(node)
        self.complexity_scores.append(complexity)

        # Check for algorithm duplication
        if len(node.body) > 2:  # Only check substantial functions
            normalized = self._normalize_function_for_duplication(node)
            if normalized in self.function_bodies:
                self.duplicate_algorithms += 1
            else:
                self.function_bodies[normalized] = node.name

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Analyze class definitions for God Object patterns."""
        self.class_count += 1

        # Count methods
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        method_count = len(methods)

        # Estimate lines of code
        if hasattr(node, "end_lineno") and node.end_lineno:
            class_loc = node.end_lineno - node.lineno
        else:
            # Fallback estimation
            class_loc = len(node.body) * 8

        # God class detection
        if method_count > 20 or class_loc > 500:
            self.god_classes.append(node.name)

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        """Track imports for coupling analysis."""
        for alias in node.names:
            self.imports.add(alias.name)
            self.import_count += 1
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track from imports for coupling analysis."""
        if node.module:
            self.from_imports[node.module].extend([alias.name for alias in node.names])
            self.import_count += len(node.names)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global):
        """Track global variable usage."""
        self.global_usage += len(node.names)
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal):
        """Track nonlocal variable usage (also coupling indicator)."""
        self.global_usage += len(node.names)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        """Detect magic literals."""
        if self._is_magic_literal(node.value):
            self.magic_literals += 1
            if self._is_in_conditional_context(node):
                self.magic_in_conditionals += 1
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        """Detect magic literals (Python 3.8+)."""
        if isinstance(node.value, (int, float, str)) and self._is_magic_literal(node.value):
            self.magic_literals += 1
            if self._is_in_conditional_context(node):
                self.magic_in_conditionals += 1
        self.generic_visit(node)

    def visit_Num(self, node: ast.Num):
        """Detect magic numbers (Python < 3.8 compatibility)."""
        if self._is_magic_literal(node.n):
            self.magic_literals += 1
            if self._is_in_conditional_context(node):
                self.magic_in_conditionals += 1
        self.generic_visit(node)

    def visit_Str(self, node: ast.Str):
        """Detect magic strings (Python < 3.8 compatibility)."""
        if self._is_magic_literal(node.s):
            self.magic_literals += 1
            if self._is_in_conditional_context(node):
                self.magic_in_conditionals += 1
        self.generic_visit(node)

    def calculate_metrics(self) -> CouplingMetrics:
        """Calculate final coupling metrics for the file."""
        # Calculate ratios and densities
        positional_ratio = (self.positional_violations / max(self.function_count, 1)) * 100
        magic_density = (self.magic_literals / max(self.lines_of_code, 1)) * 100

        # Calculate import coupling score
        # Higher scores indicate more coupling
        import_coupling = min(self.import_count * 2, 100)  # Cap at 100

        # Circular dependency risk (simplified heuristic)
        circular_risk = 0
        if len(self.from_imports) > 10:  # Many different modules imported
            circular_risk = min(len(self.from_imports) * 5, 50)

        # Overall coupling score (0-100, lower is better)
        coupling_score = (
            (positional_ratio * 0.25)
            + (magic_density * 0.25)
            + (import_coupling * 0.2)
            + (len(self.god_classes) * 20 * 0.15)
            + (self.global_usage * 5 * 0.1)
            + (circular_risk * 0.05)
        )
        coupling_score = min(coupling_score, 100)

        # Maintainability index (simplified version of Microsoft's formula)
        avg_complexity = sum(self.complexity_scores) / max(len(self.complexity_scores), 1)
        maintainability = max(
            0,
            171
            - 5.2 * avg_complexity
            - 0.23 * len(self.god_classes) * 10
            - 16.2 * (self.magic_literals / max(self.lines_of_code, 1)),
        )

        return CouplingMetrics(
            file_path=self.file_path,
            total_functions=self.function_count,
            total_classes=self.class_count,
            total_imports=self.import_count,
            lines_of_code=self.lines_of_code,
            positional_param_violations=self.positional_violations,
            positional_param_ratio=positional_ratio,
            magic_literal_count=self.magic_literals,
            magic_literal_density=magic_density,
            duplicate_algorithm_count=self.duplicate_algorithms,
            global_usage_count=self.global_usage,
            god_classes=self.god_classes,
            import_coupling_score=import_coupling,
            circular_dependency_risk=circular_risk,
            coupling_score=coupling_score,
            maintainability_index=maintainability,
        )


class ProjectCouplingAnalyzer:
    """Analyzes coupling metrics across a project."""

    def __init__(self, exclusions: list[str] | None = None):
        self.exclusions = exclusions or [
            "test_*",
            "tests/",
            "*_test.py",
            "conftest.py",
            "deprecated/",
            "archive/",
            "experimental/",
            "__pycache__/",
            ".git/",
            "build/",
            "dist/",
            "*.egg-info/",
            "venv*/",
            "*env*/",
        ]
        self.file_metrics: list[CouplingMetrics] = []

    def should_analyze_file(self, file_path: Path) -> bool:
        """Check if file should be analyzed based on exclusions."""
        path_str = str(file_path)
        for exclusion in self.exclusions:
            if exclusion.endswith("/"):
                if exclusion[:-1] in path_str:
                    return False
            elif "*" in exclusion:
                import fnmatch

                if fnmatch.fnmatch(path_str, exclusion):
                    return False
            elif exclusion in path_str:
                return False
        return True

    def analyze_file(self, file_path: Path) -> CouplingMetrics | None:
        """Analyze a single Python file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                source = f.read()
                source_lines = source.splitlines()

            tree = ast.parse(source, filename=str(file_path))
            analyzer = CouplingAnalyzer(str(file_path), source_lines)
            analyzer.visit(tree)

            return analyzer.calculate_metrics()

        except (SyntaxError, UnicodeDecodeError, Exception):
            # Return minimal metrics for problematic files
            return CouplingMetrics(
                file_path=str(file_path),
                total_functions=0,
                total_classes=0,
                total_imports=0,
                lines_of_code=0,
                positional_param_violations=0,
                positional_param_ratio=0.0,
                magic_literal_count=0,
                magic_literal_density=0.0,
                duplicate_algorithm_count=0,
                global_usage_count=0,
                god_classes=[],
                import_coupling_score=100.0,  # Penalize unparseable files
                circular_dependency_risk=50,
                coupling_score=100.0,
                maintainability_index=0.0,
            )

    def analyze_directory(self, directory: Path) -> ProjectMetrics:
        """Analyze all Python files in a directory."""
        for py_file in directory.rglob("*.py"):
            if self.should_analyze_file(py_file):
                metrics = self.analyze_file(py_file)
                if metrics:
                    self.file_metrics.append(metrics)

        return self._calculate_project_metrics()

    def _calculate_project_metrics(self) -> ProjectMetrics:
        """Calculate project-wide metrics from file metrics."""
        if not self.file_metrics:
            return ProjectMetrics(
                total_files=0,
                total_loc=0,
                average_coupling_score=0.0,
                worst_coupled_files=[],
                total_positional_violations=0,
                total_magic_literals=0,
                total_god_classes=0,
                total_global_usage=0,
                overall_positional_ratio=0.0,
                overall_magic_density=0.0,
                god_class_ratio=0.0,
            )

        # Aggregate totals
        total_files = len(self.file_metrics)
        total_loc = sum(m.lines_of_code for m in self.file_metrics)
        total_functions = sum(m.total_functions for m in self.file_metrics)
        total_classes = sum(m.total_classes for m in self.file_metrics)

        # Coupling metrics
        avg_coupling = sum(m.coupling_score for m in self.file_metrics) / total_files
        total_positional = sum(m.positional_param_violations for m in self.file_metrics)
        total_magic = sum(m.magic_literal_count for m in self.file_metrics)
        total_gods = sum(len(m.god_classes) for m in self.file_metrics)
        total_globals = sum(m.global_usage_count for m in self.file_metrics)

        # Calculate ratios
        overall_positional_ratio = (total_positional / max(total_functions, 1)) * 100
        overall_magic_density = (total_magic / max(total_loc, 1)) * 100
        god_class_ratio = (total_gods / max(total_classes, 1)) * 100

        # Find worst coupled files
        worst_files = sorted(
            [(m.file_path, m.coupling_score) for m in self.file_metrics], key=lambda x: x[1], reverse=True
        )[:10]

        return ProjectMetrics(
            total_files=total_files,
            total_loc=total_loc,
            average_coupling_score=avg_coupling,
            worst_coupled_files=worst_files,
            total_positional_violations=total_positional,
            total_magic_literals=total_magic,
            total_god_classes=total_gods,
            total_global_usage=total_globals,
            overall_positional_ratio=overall_positional_ratio,
            overall_magic_density=overall_magic_density,
            god_class_ratio=god_class_ratio,
        )

    def compare_with_baseline(self, baseline_file: Path) -> ProjectMetrics | None:
        """Compare current metrics with a baseline."""
        try:
            with open(baseline_file) as f:
                baseline_data = json.load(f)

            baseline_metrics = ProjectMetrics(**baseline_data)
            current_metrics = self._calculate_project_metrics()

            # Calculate improvement score
            improvement_score = baseline_metrics.average_coupling_score - current_metrics.average_coupling_score

            # Identify regression areas
            regression_areas = []
            if current_metrics.overall_positional_ratio > baseline_metrics.overall_positional_ratio:
                regression_areas.append("positional_parameters")
            if current_metrics.overall_magic_density > baseline_metrics.overall_magic_density:
                regression_areas.append("magic_literals")
            if current_metrics.god_class_ratio > baseline_metrics.god_class_ratio:
                regression_areas.append("god_classes")

            current_metrics.improvement_score = improvement_score
            current_metrics.regression_areas = regression_areas

            return current_metrics

        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return None

    def save_baseline(self, output_file: Path):
        """Save current metrics as baseline for future comparisons."""
        metrics = self._calculate_project_metrics()
        with open(output_file, "w") as f:
            json.dump(asdict(metrics), f, indent=2)

    def generate_report(self, project_metrics: ProjectMetrics, output_format: str = "text") -> str:
        """Generate a coupling metrics report."""
        if output_format == "json":
            return json.dumps(asdict(project_metrics), indent=2)

        # Text report
        lines = [
            "=" * 80,
            "COUPLING METRICS REPORT",
            "=" * 80,
            "",
            f"Total files analyzed: {project_metrics.total_files}",
            f"Total lines of code: {project_metrics.total_loc:,}",
            f"Average coupling score: {project_metrics.average_coupling_score:.1f}/100 (lower is better)",
            "",
            "CONNASCENCE METRICS",
            "-" * 40,
            f"Positional parameter violations: {project_metrics.total_positional_violations}",
            f"Positional parameter ratio: {project_metrics.overall_positional_ratio:.1f}%",
            f"Magic literal count: {project_metrics.total_magic_literals}",
            f"Magic literal density: {project_metrics.overall_magic_density:.2f} per 100 LOC",
            f"God classes detected: {project_metrics.total_god_classes}",
            f"God class ratio: {project_metrics.god_class_ratio:.1f}%",
            f"Global usage instances: {project_metrics.total_global_usage}",
            "",
        ]

        # Improvement tracking
        if project_metrics.improvement_score is not None:
            lines.extend(
                [
                    "IMPROVEMENT TRACKING",
                    "-" * 40,
                    f"Coupling score change: {project_metrics.improvement_score:+.1f}",
                    "Status: "
                    + (
                        "IMPROVED"
                        if project_metrics.improvement_score > 0
                        else "DEGRADED" if project_metrics.improvement_score < 0 else "STABLE"
                    ),
                    "",
                ]
            )

            if project_metrics.regression_areas:
                lines.extend(["Regression areas:", *[f"  - {area}" for area in project_metrics.regression_areas], ""])

        # Worst coupled files
        if project_metrics.worst_coupled_files:
            lines.extend(
                [
                    "MOST COUPLED FILES",
                    "-" * 40,
                ]
            )

            for file_path, score in project_metrics.worst_coupled_files[:10]:
                relative_path = os.path.relpath(file_path)
                lines.append(f"  {score:5.1f}: {relative_path}")

            lines.append("")

        # Quality recommendations
        lines.extend(
            [
                "RECOMMENDATIONS",
                "-" * 40,
            ]
        )

        if project_metrics.overall_positional_ratio > 20:
            lines.append("• HIGH: Reduce positional parameters (use keyword args, data classes)")
        if project_metrics.overall_magic_density > 5:
            lines.append("• HIGH: Replace magic literals with named constants")
        if project_metrics.god_class_ratio > 10:
            lines.append("• CRITICAL: Break down God classes using Single Responsibility Principle")
        if project_metrics.average_coupling_score > 50:
            lines.append("• CRITICAL: Overall coupling is very high - prioritize decoupling efforts")
        elif project_metrics.average_coupling_score > 30:
            lines.append("• MEDIUM: Consider architectural improvements to reduce coupling")
        else:
            lines.append("• GOOD: Coupling levels are within acceptable ranges")

        return "\n".join(lines)


def main():
    """Main entry point for coupling metrics analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze coupling metrics in Python code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python coupling_metrics.py .                         # Analyze current directory
  python coupling_metrics.py src/ --json               # JSON output
  python coupling_metrics.py . --baseline metrics.json # Compare with baseline
  python coupling_metrics.py . --save-baseline         # Save current as baseline
        """,
    )

    parser.add_argument("path", help="Path to analyze (file or directory)")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument(
        "--format", "-f", choices=["text", "json"], default="text", help="Output format (default: text)"
    )
    parser.add_argument("--baseline", "-b", help="Baseline file for comparison")
    parser.add_argument("--save-baseline", action="store_true", help="Save current metrics as baseline")
    parser.add_argument("--exclude", "-e", action="append", help="Additional exclusion patterns")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Initialize analyzer
    exclusions = None
    if args.exclude:
        exclusions = args.exclude

    analyzer = ProjectCouplingAnalyzer(exclusions)

    # Analyze path
    target_path = Path(args.path)
    if not target_path.exists():
        print(f"Error: Path '{target_path}' does not exist", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Analyzing coupling metrics for {target_path}...")

    start_time = time.time()

    # Run analysis
    if args.baseline:
        baseline_path = Path(args.baseline)
        if target_path.is_file():
            analyzer.analyze_file(target_path)
            project_metrics = analyzer._calculate_project_metrics()
        else:
            project_metrics = analyzer.analyze_directory(target_path)

        # Try to compare with baseline
        comparison_metrics = analyzer.compare_with_baseline(baseline_path)
        if comparison_metrics:
            project_metrics = comparison_metrics
    else:
        if target_path.is_file():
            analyzer.analyze_file(target_path)
            project_metrics = analyzer._calculate_project_metrics()
        else:
            project_metrics = analyzer.analyze_directory(target_path)

    elapsed = time.time() - start_time

    # Save baseline if requested
    if args.save_baseline:
        baseline_file = Path("coupling_baseline.json")
        analyzer.save_baseline(baseline_file)
        if args.verbose:
            print(f"Baseline saved to {baseline_file}")

    # Generate report
    report = analyzer.generate_report(project_metrics, args.format)

    # Output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        if args.verbose:
            print(f"Report saved to {args.output}")
    else:
        # Handle encoding issues on Windows
        try:
            print(report)
        except UnicodeEncodeError:
            print(report.encode("ascii", "replace").decode("ascii"))

    if args.verbose:
        print(f"\nAnalysis completed in {elapsed:.2f} seconds")
        print(f"Average coupling score: {project_metrics.average_coupling_score:.1f}/100")

    # Exit with error code if coupling is very high
    return 1 if project_metrics.average_coupling_score > 75 else 0


if __name__ == "__main__":
    sys.exit(main())
