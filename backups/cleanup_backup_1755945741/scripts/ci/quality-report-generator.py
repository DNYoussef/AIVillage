#!/usr/bin/env python3
"""
Quality Report Generator for PR Comments
Generates comprehensive architectural quality reports for GitHub PR comments.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path


class QualityReportGenerator:
    """Generates quality reports for PR comments and dashboards."""

    def __init__(self):
        self.project_root = Path.cwd()

    def generate_pr_report(
        self, fitness_report: str, coupling_report: str, antipatterns_report: str, quality_gate: str, pr_number: int
    ) -> str:
        """Generate comprehensive PR quality report."""

        # Load report data
        quality_data = self._load_json_file(quality_gate)
        coupling_data = self._load_json_file(coupling_report)
        antipatterns_data = self._load_json_file(antipatterns_report)
        fitness_content = self._load_text_file(fitness_report)

        # Generate report sections
        report_sections = []

        # Header
        report_sections.append(self._generate_header(quality_data, pr_number))

        # Overall summary
        report_sections.append(self._generate_summary(quality_data))

        # Quality metrics breakdown
        report_sections.append(self._generate_metrics_breakdown(quality_data))

        # Coupling analysis
        if coupling_data:
            report_sections.append(self._generate_coupling_section(coupling_data))

        # Anti-patterns detection
        if antipatterns_data:
            report_sections.append(self._generate_antipatterns_section(antipatterns_data))

        # Fitness functions results
        if fitness_content:
            report_sections.append(self._generate_fitness_section(fitness_content))

        # Recommendations
        report_sections.append(self._generate_recommendations(quality_data))

        # Footer
        report_sections.append(self._generate_footer())

        return "\n\n".join(report_sections)

    def _generate_header(self, quality_data: dict, pr_number: int) -> str:
        """Generate report header."""
        passed = quality_data.get("passed", False)
        score = quality_data.get("overall_score", 0)

        status_emoji = "✅" if passed else "❌"
        status_text = "PASSED" if passed else "FAILED"

        score_emoji = self._get_score_emoji(score)

        return f"""## 🏗️ Architectural Quality Report

**PR #{pr_number} Quality Gate: {status_emoji} {status_text}**

{score_emoji} **Overall Score: {score:.1f}/100**

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC*"""

    def _generate_summary(self, quality_data: dict) -> str:
        """Generate quality summary section."""
        metrics = quality_data.get("metrics", [])
        passed_count = sum(1 for m in metrics if m.get("passed", False))
        total_count = len(metrics)

        violations = quality_data.get("violations", [])

        summary = f"""### 📊 Quality Summary

- **Metrics Passed:** {passed_count}/{total_count}
- **Quality Violations:** {len(violations)}"""

        if violations:
            summary += "\n\n**Critical Issues:**"
            for violation in violations[:3]:  # Show first 3 violations
                summary += f"\n- 🚨 {violation}"

            if len(violations) > 3:
                summary += f"\n- *... and {len(violations) - 3} more issues*"

        return summary

    def _generate_metrics_breakdown(self, quality_data: dict) -> str:
        """Generate detailed metrics breakdown."""
        metrics = quality_data.get("metrics", [])

        breakdown = "### 🎯 Quality Metrics\n\n"
        breakdown += "| Metric | Value | Threshold | Status | Level |\n"
        breakdown += "|--------|-------|-----------|--------|-------|\n"

        for metric in metrics:
            name = metric.get("name", "Unknown")
            value = metric.get("value", 0)
            threshold = metric.get("threshold", 0)
            passed = metric.get("passed", False)
            level = metric.get("level", "unknown")

            status_emoji = "✅" if passed else "❌"
            level_emoji = self._get_level_emoji(level)

            # Format value based on metric type
            if isinstance(value, float):
                value_str = f"{value:.1f}"
            else:
                value_str = str(value)

            if isinstance(threshold, float):
                threshold_str = f"{threshold:.1f}"
            else:
                threshold_str = str(threshold)

            breakdown += (
                f"| {name} | {value_str} | {threshold_str} | {status_emoji} | {level_emoji} {level.title()} |\n"
            )

        return breakdown

    def _generate_coupling_section(self, coupling_data: dict) -> str:
        """Generate coupling analysis section."""
        section = "### 🔗 Coupling Analysis\n\n"

        # Overall coupling metrics
        avg_coupling = coupling_data.get("average_coupling_score", 0)
        max_coupling = coupling_data.get("max_coupling_score", 0)

        section += f"- **Average Coupling Score:** {avg_coupling:.1f}\n"
        section += f"- **Maximum Coupling Score:** {max_coupling:.1f}\n\n"

        # High coupling modules
        high_coupling = coupling_data.get("high_coupling_modules", [])
        if high_coupling:
            section += "**High Coupling Modules:**\n"
            for module in high_coupling[:5]:  # Show top 5
                name = module.get("name", "Unknown")
                score = module.get("score", 0)
                section += f"- `{name}`: {score:.1f}\n"

            if len(high_coupling) > 5:
                section += f"- *... and {len(high_coupling) - 5} more modules*\n"

        return section

    def _generate_antipatterns_section(self, antipatterns_data: dict) -> str:
        """Generate anti-patterns detection section."""
        section = "### 🚩 Anti-patterns Detection\n\n"

        patterns = antipatterns_data.get("detected_patterns", [])

        if not patterns:
            section += "🎉 No anti-patterns detected!\n"
            return section

        section += f"**Detected Anti-patterns:** {len(patterns)}\n\n"

        # Group by pattern type
        pattern_groups = {}
        for pattern in patterns:
            pattern_type = pattern.get("type", "Unknown")
            if pattern_type not in pattern_groups:
                pattern_groups[pattern_type] = []
            pattern_groups[pattern_type].append(pattern)

        for pattern_type, instances in pattern_groups.items():
            section += f"**{pattern_type}** ({len(instances)} instances):\n"
            for instance in instances[:3]:  # Show first 3 instances
                location = instance.get("location", "Unknown")
                description = instance.get("description", "No description")
                section += f"- `{location}`: {description}\n"

            if len(instances) > 3:
                section += f"- *... and {len(instances) - 3} more instances*\n"
            section += "\n"

        return section

    def _generate_fitness_section(self, fitness_content: str) -> str:
        """Generate fitness functions section."""
        section = "### 🏋️ Fitness Functions Results\n\n"

        # Extract key information from fitness report
        lines = fitness_content.split("\n")

        # Look for summary lines or key metrics
        summary_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith("✅") or line.startswith("❌") or line.startswith("⚠️"):
                summary_lines.append(line)

        if summary_lines:
            section += "**Key Results:**\n"
            for line in summary_lines[:5]:  # Show first 5 results
                section += f"- {line}\n"

            if len(summary_lines) > 5:
                section += f"- *... and {len(summary_lines) - 5} more results*\n"
        else:
            section += "*Fitness functions executed successfully. See full report in artifacts.*\n"

        return section

    def _generate_recommendations(self, quality_data: dict) -> str:
        """Generate recommendations section."""
        recommendations = quality_data.get("recommendations", [])

        if not recommendations:
            return "### ✨ Recommendations\n\n🎉 No specific recommendations - architecture looks good!"

        section = "### ✨ Recommendations\n\n"

        # Prioritize recommendations
        priority_order = ["coupling", "complexity", "magic", "god", "connascence", "anti-pattern"]

        categorized_recs = {category: [] for category in priority_order}
        other_recs = []

        for rec in recommendations:
            categorized = False
            for category in priority_order:
                if category in rec.lower():
                    categorized_recs[category].append(rec)
                    categorized = True
                    break

            if not categorized:
                other_recs.append(rec)

        # Output prioritized recommendations
        priority_emojis = {
            "coupling": "🔗",
            "complexity": "🧩",
            "magic": "✨",
            "god": "🏗️",
            "connascence": "🔄",
            "anti-pattern": "🚩",
        }

        for category in priority_order:
            recs = categorized_recs[category]
            if recs:
                emoji = priority_emojis.get(category, "📌")
                section += f"{emoji} **{category.replace('-', ' ').title()}:**\n"
                for rec in recs:
                    section += f"   - {rec}\n"
                section += "\n"

        # Add other recommendations
        if other_recs:
            section += "📌 **General:**\n"
            for rec in other_recs:
                section += f"   - {rec}\n"

        return section

    def _generate_footer(self) -> str:
        """Generate report footer."""
        return """---

<details>
<summary>📈 How to Improve Quality Score</summary>

### Quality Improvement Guide

1. **Reduce Coupling** (25% weight):
   - Use dependency injection instead of direct imports
   - Implement facades for complex subsystems
   - Extract interfaces for external dependencies

2. **Lower Complexity** (20% weight):
   - Break down complex functions (>15 cyclomatic complexity)
   - Use early returns to reduce nesting
   - Extract methods for repeated logic

3. **Eliminate God Objects** (15% weight):
   - Apply Single Responsibility Principle
   - Extract related methods into separate classes
   - Use composition over inheritance

4. **Remove Magic Literals** (15% weight):
   - Replace magic numbers with named constants
   - Use enums for categorical values
   - Define configuration classes for settings

5. **Fix Connascence Issues** (15% weight):
   - Replace positional arguments with keyword arguments
   - Use type hints for better API contracts
   - Eliminate duplicate algorithms

6. **Address Anti-patterns** (10% weight):
   - Refactor Big Ball of Mud into modules
   - Remove Copy-Paste Programming
   - Eliminate database-as-IPC patterns

</details>

<details>
<summary>🔍 Understanding Quality Levels</summary>

### Quality Level Definitions

- **🟢 Excellent**: Best practices followed, minimal technical debt
- **🔵 Good**: Minor issues, generally well-structured
- **🟡 Acceptable**: Some technical debt, needs attention
- **🟠 Poor**: Significant issues, refactoring recommended
- **🔴 Critical**: Major problems, immediate action required

</details>

*This report was generated by the Architectural Quality Gate system.*
*For more details, check the workflow artifacts.*"""

    def _load_json_file(self, filepath: str) -> dict:
        """Load JSON file safely."""
        try:
            with open(filepath) as f:
                return json.load(f)
        except Exception:
            return {}

    def _load_text_file(self, filepath: str) -> str:
        """Load text file safely."""
        try:
            with open(filepath) as f:
                return f.read()
        except Exception:
            return ""

    def _get_score_emoji(self, score: float) -> str:
        """Get emoji based on quality score."""
        if score >= 90:
            return "🏆"
        elif score >= 80:
            return "🥇"
        elif score >= 70:
            return "🥈"
        elif score >= 60:
            return "🥉"
        elif score >= 50:
            return "⚠️"
        else:
            return "🚨"

    def _get_level_emoji(self, level: str) -> str:
        """Get emoji based on quality level."""
        level_emojis = {"excellent": "🟢", "good": "🔵", "acceptable": "🟡", "poor": "🟠", "critical": "🔴"}
        return level_emojis.get(level.lower(), "⚪")


def main():
    """Main entry point for report generation."""
    parser = argparse.ArgumentParser(description="Generate architectural quality report")
    parser.add_argument("--fitness-report", type=str, required=True, help="Path to fitness functions report")
    parser.add_argument("--coupling-report", type=str, required=True, help="Path to coupling analysis report")
    parser.add_argument("--antipatterns-report", type=str, required=True, help="Path to anti-patterns report")
    parser.add_argument("--quality-gate", type=str, required=True, help="Path to quality gate results")
    parser.add_argument("--pr-number", type=int, required=True, help="Pull request number")
    parser.add_argument("--output", type=str, default="pr_quality_report.md", help="Output file for the report")

    args = parser.parse_args()

    # Initialize report generator
    generator = QualityReportGenerator()

    # Generate PR report
    report = generator.generate_pr_report(
        fitness_report=args.fitness_report,
        coupling_report=args.coupling_report,
        antipatterns_report=args.antipatterns_report,
        quality_gate=args.quality_gate,
        pr_number=args.pr_number,
    )

    # Write report to file
    with open(args.output, "w") as f:
        f.write(report)

    print(f"Quality report generated: {args.output}")


if __name__ == "__main__":
    main()
