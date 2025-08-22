#!/usr/bin/env python3
"""Magic Literal Analysis and Reduction Verification Script.

This script analyzes the codebase for magic literals and verifies the effectiveness
of our constants extraction efforts. It provides detailed reporting on:
- Magic literal count and distribution
- Constant coverage by package
- Remaining hotspots requiring attention
- Progress tracking against our 38% reduction target
"""

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MagicLiteral:
    """Represents a magic literal found in the code."""

    value: Any
    file_path: str
    line_number: int
    column: int
    context: str
    category: str = "unknown"
    severity: str = "medium"
    in_conditional: bool = False
    suggested_constant: str = ""


@dataclass
class PackageStats:
    """Statistics for a package's magic literal usage."""

    package_name: str
    total_literals: int = 0
    literals_by_category: dict[str, int] = field(default_factory=dict)
    literals_by_severity: dict[str, int] = field(default_factory=dict)
    files_with_literals: set[str] = field(default_factory=set)
    constants_modules: list[str] = field(default_factory=list)


class MagicLiteralAnalyzer(ast.NodeVisitor):
    """AST visitor to detect magic literals."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.literals: list[MagicLiteral] = []
        self.lines = []

        # Load file content for context
        try:
            with open(file_path, encoding="utf-8") as f:
                self.lines = f.readlines()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")

    def visit_Constant(self, node: ast.Constant) -> None:
        """Visit constant nodes (Python 3.8+)."""
        if self._should_analyze_literal(node.value):
            literal = self._create_magic_literal(node, node.value)
            self.literals.append(literal)
        self.generic_visit(node)

    def visit_Num(self, node: ast.Num) -> None:
        """Visit numeric literals (deprecated but still used)."""
        if self._should_analyze_literal(node.n):
            literal = self._create_magic_literal(node, node.n)
            self.literals.append(literal)
        self.generic_visit(node)

    def visit_Str(self, node: ast.Str) -> None:
        """Visit string literals (deprecated but still used)."""
        if self._should_analyze_literal(node.s):
            literal = self._create_magic_literal(node, node.s)
            self.literals.append(literal)
        self.generic_visit(node)

    def _should_analyze_literal(self, value: Any) -> bool:
        """Determine if a literal should be analyzed as a magic literal."""
        # Skip common non-magic values
        if value in (0, 1, -1, True, False, None, "", "utf-8"):
            return False

        # Skip very short strings that are likely not magic
        if isinstance(value, str) and len(value) <= 2:
            return False

        # Skip floats that are likely mathematical constants
        if isinstance(value, float) and value in (0.0, 1.0, -1.0, 0.5):
            return False

        # Analyze numbers >= 2 and meaningful strings
        if isinstance(value, int) and abs(value) >= 2:
            return True

        if isinstance(value, float) and abs(value) >= 2.0:
            return True

        if isinstance(value, str) and len(value) > 2:
            return True

        return False

    def _create_magic_literal(self, node: ast.AST, value: Any) -> MagicLiteral:
        """Create a MagicLiteral object from an AST node."""
        context = self._get_context(node.lineno)
        category = self._categorize_literal(value, context)
        severity = self._assess_severity(value, context, node)
        in_conditional = self._is_in_conditional(node)

        return MagicLiteral(
            value=value,
            file_path=self.file_path,
            line_number=node.lineno,
            column=node.col_offset,
            context=context,
            category=category,
            severity=severity,
            in_conditional=in_conditional,
            suggested_constant=self._suggest_constant_name(value, context),
        )

    def _get_context(self, line_number: int) -> str:
        """Get the line context for a literal."""
        if 1 <= line_number <= len(self.lines):
            return self.lines[line_number - 1].strip()
        return ""

    def _categorize_literal(self, value: Any, context: str) -> str:
        """Categorize the magic literal based on value and context."""
        context_lower = context.lower()

        # Security-related
        if any(
            keyword in context_lower for keyword in ["password", "auth", "token", "secret", "crypto", "hash", "encrypt"]
        ):
            return "security"

        # Time-related
        if any(
            keyword in context_lower
            for keyword in ["timeout", "interval", "delay", "sleep", "duration", "seconds", "minutes", "hours"]
        ):
            return "timing"

        # Configuration
        if any(
            keyword in context_lower for keyword in ["config", "setting", "option", "parameter", "limit", "max", "min"]
        ):
            return "configuration"

        # Network/API
        if any(
            keyword in context_lower for keyword in ["port", "host", "url", "endpoint", "api", "http", "connection"]
        ):
            return "network"

        # File/Path
        if any(keyword in context_lower for keyword in ["path", "file", "dir", "extension", "filename"]):
            return "file_system"

        # Business logic
        if any(keyword in context_lower for keyword in ["cost", "price", "rate", "budget", "threshold"]):
            return "business_logic"

        # Display/UI
        if any(keyword in context_lower for keyword in ["format", "message", "log", "print", "display"]):
            return "presentation"

        return "unknown"

    def _assess_severity(self, value: Any, context: str, node: ast.AST) -> str:
        """Assess the severity of the magic literal."""
        # Critical: Security-related magic literals
        if any(keyword in context.lower() for keyword in ["password", "secret", "key", "token", "auth"]):
            return "critical"

        # High: Configuration limits, timeouts in conditionals
        if self._is_in_conditional(node):
            return "high"

        # High: Large numbers that likely represent important thresholds
        if isinstance(value, int | float) and abs(value) > 100:
            return "high"

        # Medium: Most other magic literals
        return "medium"

    def _is_in_conditional(self, node: ast.AST) -> bool:
        """Check if the literal is used in a conditional statement."""
        # This is a simplified check - could be enhanced with parent node analysis
        context = self._get_context(node.lineno)
        return any(keyword in context for keyword in ["if ", "elif ", "while ", "for "])

    def _suggest_constant_name(self, value: Any, context: str) -> str:
        """Suggest a constant name for the magic literal."""
        # Extract meaningful words from context
        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", context)

        # Filter out common non-meaningful words
        meaningful_words = [
            w.upper()
            for w in words
            if w.lower() not in {"self", "def", "class", "if", "else", "for", "while", "in", "is", "and", "or", "not"}
        ]

        if meaningful_words:
            suggestion = "_".join(meaningful_words[:3])  # Max 3 words
            if isinstance(value, str):
                return f"{suggestion}_MESSAGE"
            elif "timeout" in context.lower():
                return f"{suggestion}_TIMEOUT_SECONDS"
            elif "interval" in context.lower():
                return f"{suggestion}_INTERVAL_SECONDS"
            else:
                return suggestion

        # Fallback based on value type
        if isinstance(value, str):
            return "DEFAULT_MESSAGE"
        elif isinstance(value, int | float):
            return "DEFAULT_VALUE"

        return "UNNAMED_CONSTANT"


def analyze_package(package_path: Path) -> PackageStats:
    """Analyze a package for magic literals."""
    package_name = package_path.name
    stats = PackageStats(package_name=package_name)

    # Find constants modules
    for constants_file in package_path.rglob("constants.py"):
        stats.constants_modules.append(str(constants_file.relative_to(package_path)))

    # Analyze Python files
    for py_file in package_path.rglob("*.py"):
        if py_file.name.startswith(".") or "test" in py_file.name:
            continue

        try:
            analyzer = MagicLiteralAnalyzer(str(py_file))
            with open(py_file, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(py_file))

            analyzer.visit(tree)

            if analyzer.literals:
                stats.files_with_literals.add(str(py_file.relative_to(package_path)))
                stats.total_literals += len(analyzer.literals)

                for literal in analyzer.literals:
                    # Update category counts
                    stats.literals_by_category[literal.category] = (
                        stats.literals_by_category.get(literal.category, 0) + 1
                    )

                    # Update severity counts
                    stats.literals_by_severity[literal.severity] = (
                        stats.literals_by_severity.get(literal.severity, 0) + 1
                    )

        except Exception as e:
            logger.warning(f"Error analyzing {py_file}: {e}")

    return stats


def generate_reduction_report(packages_stats: list[PackageStats]) -> dict[str, Any]:
    """Generate a comprehensive magic literal reduction report."""
    total_literals = sum(stats.total_literals for stats in packages_stats)
    total_files = sum(len(stats.files_with_literals) for stats in packages_stats)

    # Calculate category distribution
    category_totals = {}
    severity_totals = {}

    for stats in packages_stats:
        for category, count in stats.literals_by_category.items():
            category_totals[category] = category_totals.get(category, 0) + count

        for severity, count in stats.literals_by_severity.items():
            severity_totals[severity] = severity_totals.get(severity, 0) + count

    # Find packages with constants modules
    packages_with_constants = [stats for stats in packages_stats if stats.constants_modules]

    # Calculate reduction metrics
    target_reduction = 0.38  # 38% reduction target
    baseline_estimate = 32739  # From original analysis
    current_estimate = total_literals
    reduction_achieved = max(0, (baseline_estimate - current_estimate) / baseline_estimate)

    report = {
        "summary": {
            "total_magic_literals": total_literals,
            "files_with_literals": total_files,
            "packages_analyzed": len(packages_stats),
            "packages_with_constants": len(packages_with_constants),
            "baseline_estimate": baseline_estimate,
            "reduction_target": target_reduction,
            "reduction_achieved": reduction_achieved,
            "target_met": reduction_achieved >= target_reduction,
        },
        "category_distribution": category_totals,
        "severity_distribution": severity_totals,
        "package_breakdown": [
            {
                "package": stats.package_name,
                "literals": stats.total_literals,
                "files": len(stats.files_with_literals),
                "constants_modules": len(stats.constants_modules),
                "categories": stats.literals_by_category,
                "severities": stats.literals_by_severity,
            }
            for stats in sorted(packages_stats, key=lambda s: s.total_literals, reverse=True)
        ],
        "recommendations": generate_recommendations(packages_stats, category_totals),
    }

    return report


def generate_recommendations(packages_stats: list[PackageStats], category_totals: dict[str, int]) -> list[str]:
    """Generate actionable recommendations for magic literal reduction."""
    recommendations = []

    # High-priority categories
    if category_totals.get("security", 0) > 0:
        recommendations.append(
            f"CRITICAL: Replace {category_totals['security']} security-related magic literals immediately"
        )

    if category_totals.get("configuration", 0) > 0:
        recommendations.append(
            f"HIGH: Create configuration constants for {category_totals['configuration']} config literals"
        )

    # Packages without constants modules
    packages_without_constants = [
        stats for stats in packages_stats if not stats.constants_modules and stats.total_literals > 10
    ]

    for stats in packages_without_constants[:5]:  # Top 5
        recommendations.append(
            f"Create constants.py module for {stats.package_name} package ({stats.total_literals} literals)"
        )

    # Category-specific recommendations
    if category_totals.get("timing", 0) > 0:
        recommendations.append(
            f"Replace {category_totals['timing']} timing-related magic literals with named constants"
        )

    if category_totals.get("business_logic", 0) > 0:
        recommendations.append(
            f"Extract {category_totals['business_logic']} business logic literals to domain constants"
        )

    return recommendations


def main():
    """Main analysis function."""
    logger.info("Starting magic literal analysis...")

    # Analyze major packages
    packages_to_analyze = ["packages/core", "packages/agents", "packages/rag", "packages/fog", "packages/monitoring"]

    packages_stats = []

    for package_path_str in packages_to_analyze:
        package_path = Path(package_path_str)
        if package_path.exists():
            logger.info(f"Analyzing {package_path}...")
            stats = analyze_package(package_path)
            packages_stats.append(stats)
            logger.info(f"Found {stats.total_literals} magic literals in {len(stats.files_with_literals)} files")

    # Generate comprehensive report
    report = generate_reduction_report(packages_stats)

    # Save report
    report_path = Path("quality_reports/magic_literal_analysis.json")
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("MAGIC LITERAL REDUCTION ANALYSIS")
    print("=" * 60)
    print(f"Total magic literals found: {report['summary']['total_magic_literals']:,}")
    print(f"Files with literals: {report['summary']['files_with_literals']:,}")
    print(f"Reduction achieved: {report['summary']['reduction_achieved']:.1%}")
    print(f"Target (38%): {'MET' if report['summary']['target_met'] else 'NOT MET'}")

    print("\nTop categories:")
    for category, count in sorted(report["category_distribution"].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {category}: {count:,} literals")

    print("\nRecommendations:")
    for i, rec in enumerate(report["recommendations"][:5], 1):
        print(f"  {i}. {rec}")

    print(f"\nDetailed report saved to: {report_path}")

    logger.info("Magic literal analysis completed!")


if __name__ == "__main__":
    main()
