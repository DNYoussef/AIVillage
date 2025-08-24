#!/usr/bin/env python3
"""
Magic Literal Detector for Pre-commit Hook
Detects magic numbers and strings in Python code.
"""

import argparse
import ast
from pathlib import Path
import re
import sys


class MagicLiteralDetector:
    """Detects magic literals in Python code."""

    def __init__(self, threshold_percentage: float = 20.0):
        self.threshold_percentage = threshold_percentage
        self.violations = []

        # Common non-magic values
        self.allowed_numbers = {0, 1, -1, 2, 10, 100, 1000}
        self.allowed_strings = {
            "",
            " ",
            "\n",
            "\t",
            "\r\n",
            "utf-8",
            "utf8",
            "ascii",
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "PATCH",
            "http",
            "https",
            "localhost",
            "true",
            "false",
            "none",
            "null",
        }

        # Context patterns where literals are usually acceptable
        self.safe_contexts = {
            "test_",
            "assert",
            "raise",
            "logging",
            "print",
            "__str__",
            "__repr__",
            "__name__",
            "__main__",
            "format",
            "join",
            "split",
            "replace",
        }

    def check_file(self, filepath: Path) -> dict:
        """Check a single file for magic literals."""
        try:
            with open(filepath, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            # Collect all literals
            magic_literals = []
            total_literals = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.Constant | ast.Num | ast.Str):
                    total_literals += 1

                    if self._is_magic_literal(node, content):
                        magic_literals.append(
                            {
                                "value": self._get_literal_value(node),
                                "line": getattr(node, "lineno", 0),
                                "col": getattr(node, "col_offset", 0),
                                "type": self._get_literal_type(node),
                                "context": self._get_context(node, tree),
                            }
                        )

            # Calculate magic literal density
            density = 0.0
            if total_literals > 0:
                density = (len(magic_literals) / total_literals) * 100

            return {
                "file": str(filepath),
                "magic_literals": magic_literals,
                "total_literals": total_literals,
                "magic_count": len(magic_literals),
                "density": density,
                "threshold_exceeded": density > self.threshold_percentage,
                "severity": self._calculate_severity(density, len(magic_literals)),
            }

        except Exception as e:
            print(f"Warning: Could not parse {filepath}: {e}", file=sys.stderr)
            return {
                "file": str(filepath),
                "error": str(e),
                "magic_literals": [],
                "total_literals": 0,
                "magic_count": 0,
                "density": 0.0,
                "threshold_exceeded": False,
                "severity": "error",
            }

    def _is_magic_literal(self, node: ast.AST, content: str) -> bool:
        """Determine if a literal is 'magic' (should be extracted)."""
        value = self._get_literal_value(node)

        # Skip allowed values
        if isinstance(value, int | float) and value in self.allowed_numbers:
            return False

        if isinstance(value, str) and value.lower() in self.allowed_strings:
            return False

        # Skip very short strings (likely not business logic)
        if isinstance(value, str) and len(value) <= 2:
            return False

        # Skip numeric values in safe ranges
        if isinstance(value, int):
            if -10 <= value <= 10:  # Small integers often not magic
                return False
            if value in {24, 60, 365, 1024, 2048, 4096}:  # Common constants
                return False

        if isinstance(value, float):
            if -1.0 <= value <= 1.0:  # Small floats often not magic
                return False

        # Check context - skip if in safe context
        context = self._get_context(node, None)
        if any(safe in context.lower() for safe in self.safe_contexts):
            return False

        # Check if it's a docstring
        if self._is_docstring(node):
            return False

        # Check if it's in a configuration or constant assignment
        if self._is_configuration_value(node, content):
            return False

        return True

    def _get_literal_value(self, node: ast.AST):
        """Extract the literal value from an AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Str):
            return node.s
        else:
            return None

    def _get_literal_type(self, node: ast.AST) -> str:
        """Get the type of literal."""
        value = self._get_literal_value(node)
        if isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, bool):
            return "boolean"
        else:
            return "unknown"

    def _get_context(self, node: ast.AST, tree: ast.AST) -> str:
        """Get the context where the literal appears."""
        # Try to find the parent function or class name
        if hasattr(node, "parent"):
            parent = node.parent
            if isinstance(parent, ast.FunctionDef):
                return f"function:{parent.name}"
            elif isinstance(parent, ast.ClassDef):
                return f"class:{parent.name}"

        # Fallback: return the line content or a generic context
        return "unknown"

    def _is_docstring(self, node: ast.AST) -> bool:
        """Check if the literal is part of a docstring."""
        if not isinstance(node, ast.Str | ast.Constant):
            return False

        # Simple heuristic: if it's the first statement in a function/class
        # and it's a string, it's likely a docstring
        return False  # More complex logic would be needed here

    def _is_configuration_value(self, node: ast.AST, content: str) -> bool:
        """Check if the literal is in a configuration context."""
        # Check if the literal is in an assignment to an uppercase variable
        # (which often indicates a constant)
        line_num = getattr(node, "lineno", 0)
        if line_num > 0:
            try:
                lines = content.split("\n")
                line = lines[line_num - 1] if line_num <= len(lines) else ""

                # Look for uppercase variable assignments
                if re.search(r"^[A-Z_][A-Z0-9_]*\s*=", line.strip()):
                    return True

                # Look for configuration dictionary keys
                if re.search(r'[\'"][A-Za-z_][A-Za-z0-9_]*[\'"]:', line):
                    return True

            except (IndexError, AttributeError):
                pass

        return False

    def _calculate_severity(self, density: float, count: int) -> str:
        """Calculate severity based on density and count."""
        if density > 50 or count > 20:
            return "critical"
        elif density > 35 or count > 15:
            return "high"
        elif density > 20 or count > 10:
            return "medium"
        else:
            return "low"

    def check_files(self, filepaths: list[Path]) -> list[dict]:
        """Check multiple files for magic literals."""
        results = []

        for filepath in filepaths:
            result = self.check_file(filepath)
            results.append(result)

        return results

    def format_violations(self, results: list[dict], show_details: bool = True) -> str:
        """Format violations for output."""
        violations = [r for r in results if r.get("threshold_exceeded", False)]

        if not violations:
            return "[PASS] No magic literal violations detected!"

        output = []
        output.append(f"[FAIL] Found magic literal violations in {len(violations)} file(s):")
        output.append("")

        # Sort by severity and density
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        violations.sort(key=lambda x: (severity_order.get(x["severity"], 0), x["density"]), reverse=True)

        severity_emojis = {"critical": "[CRIT]", "high": "[HIGH]", "medium": "[MED]", "low": "[LOW]"}

        for result in violations:
            severity = result["severity"]
            emoji = severity_emojis.get(severity, "[LOW]")

            output.append(f"{emoji} {result['file']}")
            output.append(
                f"   Density: {result['density']:.1f}% ({result['magic_count']}/{result['total_literals']} literals)"
            )
            output.append(f"   Severity: {severity.upper()}")

            if show_details and result["magic_literals"]:
                output.append("   Magic literals found:")

                # Group by type
                by_type = {}
                for literal in result["magic_literals"][:10]:  # Show first 10
                    lit_type = literal["type"]
                    if lit_type not in by_type:
                        by_type[lit_type] = []
                    by_type[lit_type].append(literal)

                for lit_type, literals in by_type.items():
                    output.append(f"     {lit_type.title()}s:")
                    for literal in literals[:5]:  # Show first 5 per type
                        value = repr(literal["value"])
                        if len(value) > 50:
                            value = value[:47] + "..."
                        output.append(f"       Line {literal['line']}: {value}")

                    if len(literals) > 5:
                        output.append(f"       ... and {len(literals) - 5} more")

                if len(result["magic_literals"]) > 10:
                    output.append(f"   ... and {len(result['magic_literals']) - 10} more magic literals")

            output.append("")

        # Add refactoring suggestions
        output.append("[FIX] REFACTORING SUGGESTIONS:")
        output.append("   1. Extract magic numbers to named constants:")
        output.append("      [BAD] if timeout > 30:")
        output.append("      [GOOD] TIMEOUT_SECONDS = 30; if timeout > TIMEOUT_SECONDS:")
        output.append("")
        output.append("   2. Use enums for categorical values:")
        output.append("      [BAD] if status == 'active':")
        output.append("      [GOOD] class Status(Enum): ACTIVE = 'active'; if status == Status.ACTIVE:")
        output.append("")
        output.append("   3. Create configuration classes:")
        output.append("      [BAD] batch_size = 100; learning_rate = 0.001")
        output.append("      [GOOD] @dataclass class Config: batch_size: int = 100; learning_rate: float = 0.001")

        return "\n".join(output)


def main():
    """Main entry point for magic literal detection."""
    parser = argparse.ArgumentParser(description="Detect magic literals in Python code")
    parser.add_argument("files", nargs="*", help="Python files to check")
    parser.add_argument("--threshold", type=float, default=20.0, help="Magic literal density threshold percentage")
    parser.add_argument(
        "--severity", choices=["low", "medium", "high", "critical"], default="medium", help="Minimum severity to report"
    )
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--count", action="store_true", help="Output only the count of violations")
    parser.add_argument("--details", action="store_true", default=True, help="Show detailed violation information")

    args = parser.parse_args()

    # Get files to check
    if args.files:
        filepaths = [Path(f) for f in args.files if f.endswith(".py")]
    else:
        # Find all Python files in current directory
        filepaths = list(Path(".").rglob("*.py"))
        # Exclude common directories
        excluded_dirs = {"__pycache__", ".git", "venv", "env", "node_modules", "deprecated", "archive"}
        filepaths = [f for f in filepaths if not any(part in excluded_dirs for part in f.parts)]

    # Initialize detector
    detector = MagicLiteralDetector(threshold_percentage=args.threshold)

    # Check files
    results = detector.check_files(filepaths)

    # Filter by severity
    severity_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    min_level = severity_levels[args.severity]

    filtered_results = [
        r
        for r in results
        if r.get("threshold_exceeded", False) and severity_levels.get(r.get("severity", "low"), 1) >= min_level
    ]

    # Count total magic literals across all files
    total_violations = sum(len(r.get("magic_literals", [])) for r in filtered_results)

    # Output results
    if args.count:
        print(total_violations)
    elif args.json:
        import json

        print(json.dumps(filtered_results, indent=2, default=str))
    else:
        output = detector.format_violations(filtered_results, show_details=args.details)
        # Ensure ASCII-only output for Windows compatibility
        safe_output = output.encode("ascii", errors="replace").decode("ascii")
        print(safe_output)

    # Exit with error code if violations found
    if filtered_results:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
