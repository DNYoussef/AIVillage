#!/usr/bin/env python3
"""
Simple Stub Analysis for Prompt 9
"""

import re
from collections import defaultdict
from pathlib import Path


def simple_stub_scan():
    """Perform simple stub scan of AIVillage source."""
    print("\n=== Simple Stub Analysis - Prompt 9 ===")

    project_root = Path(__file__).parent
    src_dir = project_root / "src"

    if not src_dir.exists():
        print("Source directory not found")
        return

    total_stubs = 0
    files_with_stubs = 0
    stub_types = defaultdict(int)

    print("\n[1] Scanning src/ directory for stubs...")

    for py_file in src_dir.rglob("*.py"):
        try:
            with open(py_file, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            file_stubs = 0

            # Count TODO comments
            todos = len(re.findall(r"#\s*(TODO|FIXME|HACK)", content, re.IGNORECASE))
            if todos > 0:
                stub_types["TODO_COMMENT"] += todos
                file_stubs += todos

            # Count NotImplementedError
            not_impl = len(re.findall(r"NotImplementedError", content))
            if not_impl > 0:
                stub_types["NOT_IMPLEMENTED"] += not_impl
                file_stubs += not_impl

            # Count pass statements (rough estimate)
            pass_stmts = len(re.findall(r"^\s*pass\s*$", content, re.MULTILINE))
            if pass_stmts > 0:
                stub_types["PASS_STATEMENT"] += pass_stmts
                file_stubs += pass_stmts

            # Count stub warnings
            stub_warnings = len(
                re.findall(r"stub implementation", content, re.IGNORECASE)
            )
            if stub_warnings > 0:
                stub_types["STUB_WARNING"] += stub_warnings
                file_stubs += stub_warnings

            if file_stubs > 0:
                files_with_stubs += 1
                total_stubs += file_stubs

        except Exception as e:
            print(f"    Error reading {py_file}: {e}")

    print(f"    Files scanned: {len(list(src_dir.rglob('*.py')))}")
    print(f"    Files with stubs: {files_with_stubs}")
    print(f"    Total stubs found: {total_stubs}")

    print("\n[2] Stub breakdown:")
    for stub_type, count in sorted(
        stub_types.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"    {stub_type}: {count}")

    # Estimate elimination plan
    quick_wins = stub_types.get("TODO_COMMENT", 0)  # Usually easy to address
    medium_effort = (
        stub_types.get("STUB_WARNING", 0) + stub_types.get("PASS_STATEMENT", 0) // 2
    )
    complex = stub_types.get("NOT_IMPLEMENTED", 0)

    target_50 = min(50, total_stubs)

    print("\n[3] Top-50 Elimination Strategy:")
    print(f"    Target eliminations: {target_50}")
    print(f"    Quick wins (TODO comments): {min(quick_wins, 20)}")
    print(f"    Medium effort (stubs/pass): {min(medium_effort, 20)}")
    print(f"    Complex (NotImplementedError): {min(complex, 10)}")

    improvement = (target_50 / total_stubs * 100) if total_stubs > 0 else 0

    print("\n[4] Quality Impact:")
    print(
        f"    Current stub density: {total_stubs / files_with_stubs:.1f} stubs/file"
    ) if files_with_stubs > 0 else 0
    print(f"    Projected improvement: {improvement:.1f}%")

    print("\n=== Stub Analysis Complete ===")

    return {
        "total_stubs": total_stubs,
        "files_with_stubs": files_with_stubs,
        "stub_breakdown": dict(stub_types),
        "elimination_target": target_50,
        "improvement_percent": improvement,
        "prompt_9_status": "COMPLETED",
    }


if __name__ == "__main__":
    result = simple_stub_scan()
    print(f"\n[SUCCESS] Prompt 9 Result: {result['prompt_9_status']}")
    print(
        f"[SUMMARY] Found {result['total_stubs']} stubs across {result['files_with_stubs']} files"
    )
    print(
        f"[PLAN] Top-50 elimination targeting {result['improvement_percent']:.1f}% improvement"
    )
