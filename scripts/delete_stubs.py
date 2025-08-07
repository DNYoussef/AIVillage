#!/usr/bin/env python3
"""Script to identify and remove fake/stub code."""

import ast
from pathlib import Path


def analyze_file(filepath):
    """Analyze a Python file for stub patterns."""
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()

        # Skip files that are too small to be meaningful
        if len(content.strip()) < 50:
            return []

        issues = []

        # Check for fake success returns
        if 'return {"success": True}' in content:
            issues.append("Contains fake success return")

        # Check for empty functions with just pass
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function body is just pass
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    issues.append(f"Empty function: {node.name}")

                # Check for functions that just raise NotImplementedError
                if (
                    len(node.body) == 1
                    and isinstance(node.body[0], ast.Raise)
                    and isinstance(node.body[0].exc, ast.Call)
                    and hasattr(node.body[0].exc.func, "id")
                    and node.body[0].exc.func.id == "NotImplementedError"
                ):
                    issues.append(f"Stub function: {node.name}")

        return issues

    except Exception as e:
        return [f"Error analyzing file: {e}"]


def main() -> None:
    """Main function to scan for stub code."""
    src_dir = Path("src")

    print("Scanning for stub/fake code...")

    total_issues = 0

    for py_file in src_dir.rglob("*.py"):
        if py_file.name.startswith("test_"):
            continue

        issues = analyze_file(py_file)
        if issues:
            print(f"\n{py_file}:")
            for issue in issues:
                print(f"  - {issue}")
            total_issues += len(issues)

    if total_issues == 0:
        print("No obvious stub/fake code found!")
    else:
        print(f"\nFound {total_issues} potential issues")


if __name__ == "__main__":
    main()
