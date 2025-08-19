#!/usr/bin/env python3
"""
AIVillage Hotspot Analysis Tool

Identifies code hotspots using git-churn × complexity analysis to find:
- Files that change frequently AND are complex (high-risk areas)
- Bus factor issues (files with few contributors)
- Technical debt hotspots that need refactoring

Based on the approach from "Your Code as a Crime Scene" by Adam Tornhill.
"""

import argparse
import ast
import json
import logging
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class FileComplexity:
    """Complexity metrics for a single file."""

    file_path: str
    lines_of_code: int
    cyclomatic_complexity: int
    cognitive_complexity: int
    function_count: int
    class_count: int
    import_count: int
    complexity_score: float = 0.0


@dataclass
class FileChurn:
    """Git churn metrics for a single file."""

    file_path: str
    commit_count: int
    lines_added: int
    lines_deleted: int
    lines_modified: int
    unique_authors: int
    days_since_last_change: int
    churn_score: float = 0.0


@dataclass
class Hotspot:
    """A code hotspot combining churn and complexity."""

    file_path: str
    hotspot_score: float
    churn_score: float
    complexity_score: float
    commit_count: int
    unique_authors: int
    lines_of_code: int
    cyclomatic_complexity: int
    risk_level: str
    recommendations: list[str]


class ComplexityAnalyzer:
    """Analyzes code complexity using AST parsing."""

    def analyze_python_file(self, file_path: Path) -> FileComplexity:
        """Analyze complexity of a Python file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse AST
            tree = ast.parse(content)

            # Count lines of code (excluding comments and blank lines)
            lines = content.split("\n")
            loc = sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))

            # Analyze AST
            visitor = ComplexityVisitor()
            visitor.visit(tree)

            return FileComplexity(
                file_path=str(file_path),
                lines_of_code=loc,
                cyclomatic_complexity=visitor.cyclomatic_complexity,
                cognitive_complexity=visitor.cognitive_complexity,
                function_count=visitor.function_count,
                class_count=visitor.class_count,
                import_count=visitor.import_count,
            )

        except Exception as e:
            logging.warning(f"Could not analyze {file_path}: {e}")
            return FileComplexity(
                file_path=str(file_path),
                lines_of_code=0,
                cyclomatic_complexity=0,
                cognitive_complexity=0,
                function_count=0,
                class_count=0,
                import_count=0,
            )

    def calculate_complexity_score(self, complexity: FileComplexity) -> float:
        """Calculate normalized complexity score (0-100)."""
        if complexity.lines_of_code == 0:
            return 0.0

        # Weighted complexity score
        score = (
            complexity.cyclomatic_complexity * 0.4
            + complexity.cognitive_complexity * 0.3
            + complexity.lines_of_code * 0.0001
            + complexity.function_count * 0.1  # Normalize LOC
            + complexity.class_count * 0.2
        )

        # Normalize to 0-100 scale (adjust multiplier based on observations)
        return min(100.0, score * 2.0)


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor to calculate complexity metrics."""

    def __init__(self):
        self.cyclomatic_complexity = 1  # Start with 1 for the module
        self.cognitive_complexity = 0
        self.function_count = 0
        self.class_count = 0
        self.import_count = 0
        self.nesting_level = 0

    def visit_FunctionDef(self, node):
        self.function_count += 1
        # Each function adds to cyclomatic complexity
        self.cyclomatic_complexity += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.function_count += 1
        self.cyclomatic_complexity += 1
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.class_count += 1
        self.generic_visit(node)

    def visit_Import(self, node):
        self.import_count += len(node.names)

    def visit_ImportFrom(self, node):
        if node.names:
            self.import_count += len(node.names)
        else:
            self.import_count += 1

    def visit_If(self, node):
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += 1 + self.nesting_level
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1

    def visit_While(self, node):
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += 1 + self.nesting_level
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1

    def visit_For(self, node):
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += 1 + self.nesting_level
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1

    def visit_Try(self, node):
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += 1 + self.nesting_level
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += 1 + self.nesting_level
        self.generic_visit(node)


class ChurnAnalyzer:
    """Analyzes git churn metrics."""

    def __init__(self, repo_path: Path, days_back: int = 90):
        self.repo_path = repo_path
        self.days_back = days_back

    def get_git_log(self, file_path: str | None = None) -> list[str]:
        """Get git log output for analysis."""
        since_date = (datetime.now() - timedelta(days=self.days_back)).strftime("%Y-%m-%d")

        cmd = ["git", "log", f"--since={since_date}", "--numstat", "--format=format:%H|%an|%ad|%s", "--date=short"]

        if file_path:
            cmd.append("--")
            cmd.append(file_path)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_path, check=True)
            return result.stdout.strip().split("\n") if result.stdout.strip() else []
        except subprocess.CalledProcessError as e:
            logging.warning(f"Git command failed: {e}")
            return []

    def analyze_file_churn(self, file_path: str) -> FileChurn:
        """Analyze churn for a specific file."""
        log_lines = self.get_git_log(file_path)

        commit_count = 0
        lines_added = 0
        lines_deleted = 0
        authors = set()
        last_change_date = None

        i = 0
        while i < len(log_lines):
            line = log_lines[i]

            if "|" in line:  # Commit info line
                parts = line.split("|")
                if len(parts) >= 4:
                    commit_hash, author, date_str, subject = parts[0], parts[1], parts[2], "|".join(parts[3:])
                    commit_count += 1
                    authors.add(author.strip())

                    if not last_change_date:
                        try:
                            last_change_date = datetime.strptime(date_str.strip(), "%Y-%m-%d")
                        except ValueError:
                            pass

                # Look for numstat lines after this commit
                i += 1
                while i < len(log_lines) and log_lines[i] and "|" not in log_lines[i]:
                    numstat_line = log_lines[i]
                    if "\t" in numstat_line:
                        parts = numstat_line.split("\t")
                        if len(parts) >= 3:
                            try:
                                added = int(parts[0]) if parts[0] != "-" else 0
                                deleted = int(parts[1]) if parts[1] != "-" else 0
                                lines_added += added
                                lines_deleted += deleted
                            except ValueError:
                                pass
                    i += 1
            else:
                i += 1

        # Calculate days since last change
        days_since_last = 0
        if last_change_date:
            days_since_last = (datetime.now() - last_change_date).days

        return FileChurn(
            file_path=file_path,
            commit_count=commit_count,
            lines_added=lines_added,
            lines_deleted=lines_deleted,
            lines_modified=lines_added + lines_deleted,
            unique_authors=len(authors),
            days_since_last_change=days_since_last,
        )

    def calculate_churn_score(self, churn: FileChurn) -> float:
        """Calculate normalized churn score (0-100)."""
        if churn.commit_count == 0:
            return 0.0

        # Weighted churn score
        recency_weight = max(0.1, 1.0 - (churn.days_since_last_change / 365.0))

        score = (
            churn.commit_count * 10.0
            + churn.lines_modified * 0.1  # Frequency of changes
            + (10.0 / max(1, churn.unique_authors))  # Volume of changes
            + recency_weight * 20.0  # Bus factor (fewer authors = higher risk)  # Recent changes are more important
        )

        return min(100.0, score)


class HotspotAnalyzer:
    """Main hotspot analyzer combining churn and complexity."""

    def __init__(self, repo_path: Path = None, days_back: int = 90):
        self.repo_path = repo_path or Path.cwd()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.churn_analyzer = ChurnAnalyzer(self.repo_path, days_back)

    def find_python_files(self) -> list[Path]:
        """Find all Python files in the repository."""
        python_files = []

        # Define patterns to exclude
        exclude_patterns = {
            "/.git/",
            "/build/",
            "/__pycache__/",
            "/dist/",
            "/venv/",
            "/env/",
            ".pyc",
            "/deprecated/",
            "/archive/",
            "/.vscode/",
            "/.idea/",
            "/node_modules/",
            "/target/",
            "/.mypy_cache/",
            "/.pytest_cache/",
            "/coverage/",
            "/htmlcov/",
            "/.coverage",
        }

        for py_file in self.repo_path.rglob("*.py"):
            # Skip files matching exclude patterns
            file_str = str(py_file)
            if any(pattern in file_str for pattern in exclude_patterns):
                continue

            # Skip very small files (likely not significant)
            try:
                if py_file.stat().st_size < 100:  # Skip files < 100 bytes
                    continue
            except OSError:
                continue

            python_files.append(py_file)

        return python_files

    def analyze_hotspots(self, max_files: int = 50) -> list[Hotspot]:
        """Analyze hotspots in the repository."""
        logging.info(f"Finding Python files in {self.repo_path}")
        python_files = self.find_python_files()

        logging.info(f"Analyzing {len(python_files)} Python files...")
        hotspots = []

        for i, py_file in enumerate(python_files):
            if i % 10 == 0:
                logging.info(f"Progress: {i+1}/{len(python_files)} files")

            # Get relative path from repo root
            rel_path = py_file.relative_to(self.repo_path)

            # Analyze complexity
            complexity = self.complexity_analyzer.analyze_python_file(py_file)
            complexity.complexity_score = self.complexity_analyzer.calculate_complexity_score(complexity)

            # Analyze churn
            churn = self.churn_analyzer.analyze_file_churn(str(rel_path))
            churn.churn_score = self.churn_analyzer.calculate_churn_score(churn)

            # Calculate hotspot score (churn × complexity)
            hotspot_score = (churn.churn_score * complexity.complexity_score) / 100.0

            # Determine risk level
            risk_level = self._calculate_risk_level(hotspot_score, churn, complexity)

            # Generate recommendations
            recommendations = self._generate_recommendations(churn, complexity)

            hotspot = Hotspot(
                file_path=str(rel_path),
                hotspot_score=hotspot_score,
                churn_score=churn.churn_score,
                complexity_score=complexity.complexity_score,
                commit_count=churn.commit_count,
                unique_authors=churn.unique_authors,
                lines_of_code=complexity.lines_of_code,
                cyclomatic_complexity=complexity.cyclomatic_complexity,
                risk_level=risk_level,
                recommendations=recommendations,
            )

            hotspots.append(hotspot)

        # Sort by hotspot score and return top results
        hotspots.sort(key=lambda x: x.hotspot_score, reverse=True)
        return hotspots[:max_files]

    def _calculate_risk_level(self, hotspot_score: float, churn: FileChurn, complexity: FileComplexity) -> str:
        """Calculate risk level based on various factors."""
        if hotspot_score > 50 and churn.unique_authors < 2:
            return "CRITICAL"
        elif hotspot_score > 30:
            return "HIGH"
        elif hotspot_score > 15:
            return "MEDIUM"
        elif hotspot_score > 5:
            return "LOW"
        else:
            return "MINIMAL"

    def _generate_recommendations(self, churn: FileChurn, complexity: FileComplexity) -> list[str]:
        """Generate recommendations for improving the hotspot."""
        recommendations = []

        # High complexity recommendations
        if complexity.cyclomatic_complexity > 15:
            recommendations.append("Consider breaking down complex functions (high cyclomatic complexity)")

        if complexity.lines_of_code > 500:
            recommendations.append("Consider splitting large file into smaller modules")

        if complexity.function_count > 20:
            recommendations.append("Consider organizing functions into classes or modules")

        # High churn recommendations
        if churn.commit_count > 20:
            recommendations.append("Frequently changed file - investigate instability causes")

        if churn.unique_authors < 2 and churn.commit_count > 5:
            recommendations.append("Bus factor risk - add additional maintainers")

        if churn.lines_modified > 1000:
            recommendations.append("High volume of changes - consider architectural review")

        # Combined recommendations
        if churn.churn_score > 20 and complexity.complexity_score > 20:
            recommendations.append("High-priority refactoring candidate (high churn + complexity)")

        if not recommendations:
            recommendations.append("Well-maintained file - continue current practices")

        return recommendations


def generate_report(hotspots: list[Hotspot], output_format: str = "console") -> str:
    """Generate hotspot analysis report."""
    if output_format == "json":
        return json.dumps([asdict(h) for h in hotspots], indent=2)

    elif output_format == "markdown":
        md_lines = [
            "# Code Hotspot Analysis Report",
            f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## Summary",
            f"- **Files Analyzed**: {len(hotspots)} files",
            f"- **Critical Risk**: {len([h for h in hotspots if h.risk_level == 'CRITICAL'])} files",
            f"- **High Risk**: {len([h for h in hotspots if h.risk_level == 'HIGH'])} files",
            "",
            "## Top Hotspots",
            "",
            "| Rank | File | Risk | Score | Commits | Authors | LOC | Recommendations |",
            "|------|------|------|-------|---------|---------|-----|-----------------|",
        ]

        for i, hotspot in enumerate(hotspots[:10], 1):
            recommendations = "; ".join(hotspot.recommendations[:2])  # Limit for table
            md_lines.append(
                f"| {i} | `{hotspot.file_path}` | {hotspot.risk_level} | "
                f"{hotspot.hotspot_score:.1f} | {hotspot.commit_count} | "
                f"{hotspot.unique_authors} | {hotspot.lines_of_code} | {recommendations} |"
            )

        return "\n".join(md_lines)

    else:  # console format
        lines = []
        lines.append("=" * 80)
        lines.append("🔥 CODE HOTSPOT ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Files Analyzed: {len(hotspots)}")
        lines.append("")

        # Summary by risk level
        risk_counts = defaultdict(int)
        for h in hotspots:
            risk_counts[h.risk_level] += 1

        lines.append("📊 RISK LEVEL SUMMARY:")
        for risk in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL"]:
            count = risk_counts.get(risk, 0)
            if count > 0:
                emoji = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "⚡", "LOW": "💡", "MINIMAL": "✅"}[risk]
                lines.append(f"   {emoji} {risk}: {count} files")

        lines.append("")
        lines.append("🔥 TOP 10 HOTSPOTS:")
        lines.append("-" * 80)

        for i, hotspot in enumerate(hotspots[:10], 1):
            risk_emoji = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "⚡", "LOW": "💡", "MINIMAL": "✅"}[hotspot.risk_level]

            lines.append(f"{i:2}. {risk_emoji} {hotspot.file_path}")
            lines.append(
                f"    Score: {hotspot.hotspot_score:.1f} | "
                f"Churn: {hotspot.churn_score:.1f} | "
                f"Complexity: {hotspot.complexity_score:.1f}"
            )
            lines.append(
                f"    Commits: {hotspot.commit_count} | "
                f"Authors: {hotspot.unique_authors} | "
                f"LOC: {hotspot.lines_of_code}"
            )
            lines.append(f"    📋 {hotspot.recommendations[0]}")
            lines.append("")

        return "\n".join(lines)


def create_github_issues(hotspots: list[Hotspot], top_n: int = 5, dry_run: bool = True):
    """Create GitHub issues for top hotspots."""

    def create_issue_body(hotspot: Hotspot) -> str:
        return f"""## 🔥 Code Hotspot Detected

**File**: `{hotspot.file_path}`
**Risk Level**: {hotspot.risk_level}
**Hotspot Score**: {hotspot.hotspot_score:.1f}

### 📊 Metrics
- **Churn Score**: {hotspot.churn_score:.1f} (commits: {hotspot.commit_count}, authors: {hotspot.unique_authors})
- **Complexity Score**: {hotspot.complexity_score:.1f} (LOC: {hotspot.lines_of_code}, cyclomatic: {hotspot.cyclomatic_complexity})

### 🎯 Recommendations
{chr(10).join(f'- {rec}' for rec in hotspot.recommendations)}

### 📋 Next Steps
- [ ] Review code structure and identify refactoring opportunities
- [ ] Add additional maintainers if bus factor is low
- [ ] Consider breaking down complex functions or large files
- [ ] Add comprehensive tests for high-risk areas
- [ ] Update documentation for complex logic

### 📈 Analysis Details
This issue was automatically generated by hotspot analysis on {datetime.now().strftime('%Y-%m-%d')}.

The hotspot score combines git churn (frequency of changes) with code complexity to identify files that are both frequently modified AND structurally complex - these represent the highest risk areas in the codebase.

---
*Generated by AIVillage Hotspot Analysis*
"""

    print(f"🎟️ Creating GitHub issues for top {top_n} hotspots...")

    for i, hotspot in enumerate(hotspots[:top_n], 1):
        issue_title = f"Hotspot #{i}: Refactor {Path(hotspot.file_path).name} ({hotspot.risk_level} risk)"
        issue_body = create_issue_body(hotspot)

        if dry_run:
            print(f"\n{'='*60}")
            print(f"ISSUE {i}: {issue_title}")
            print(f"{'='*60}")
            print(issue_body)
        else:
            # Create actual GitHub issue (requires gh CLI)
            try:
                cmd = [
                    "gh",
                    "issue",
                    "create",
                    "--title",
                    issue_title,
                    "--body",
                    issue_body,
                    "--label",
                    "technical-debt,hotspot,refactoring",
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"✅ Created issue: {result.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to create issue: {e}")
            except FileNotFoundError:
                print("❌ GitHub CLI (gh) not found. Install with: brew install gh")
                break


def main():
    parser = argparse.ArgumentParser(description="Analyze code hotspots using git churn × complexity")
    parser.add_argument(
        "--repo-path", type=str, default=".", help="Path to Git repository (default: current directory)"
    )
    parser.add_argument("--days-back", type=int, default=90, help="Number of days to analyze git history (default: 90)")
    parser.add_argument("--max-files", type=int, default=50, help="Maximum number of files to analyze (default: 50)")
    parser.add_argument(
        "--output-format",
        choices=["console", "json", "markdown"],
        default="console",
        help="Output format (default: console)",
    )
    parser.add_argument("--output-file", type=str, help="Output file path (default: stdout)")
    parser.add_argument("--create-issues", action="store_true", help="Create GitHub issues for top hotspots")
    parser.add_argument("--top-issues", type=int, default=5, help="Number of GitHub issues to create (default: 5)")
    parser.add_argument("--dry-run", action="store_true", help="Show issue previews without creating them")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        repo_path = Path(args.repo_path).resolve()

        if not (repo_path / ".git").exists():
            print(f"❌ Error: {repo_path} is not a Git repository")
            sys.exit(1)

        # Run hotspot analysis
        analyzer = HotspotAnalyzer(repo_path, args.days_back)
        hotspots = analyzer.analyze_hotspots(args.max_files)

        # Generate report
        report = generate_report(hotspots, args.output_format)

        # Output report
        if args.output_file:
            with open(args.output_file, "w") as f:
                f.write(report)
            print(f"✅ Report saved to {args.output_file}")
        else:
            print(report)

        # Create GitHub issues if requested
        if args.create_issues:
            create_github_issues(hotspots, args.top_issues, args.dry_run)

    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        logging.exception("Detailed error information:")
        sys.exit(1)


if __name__ == "__main__":
    main()
