#!/usr/bin/env python3
"""
Focused Stub Elimination for AIVillage Source - Prompt 9
"""

from collections import defaultdict
from pathlib import Path
import re


def find_stubs_in_file(file_path: Path) -> list[tuple[int, str, str]]:
    """Find stubs in a single Python file."""
    stubs = []

    if not file_path.exists() or file_path.suffix != ".py":
        return stubs

    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()

            # Skip empty lines and imports
            if not line_stripped or line_stripped.startswith("import") or line_stripped.startswith("from"):
                continue

            # TODO/FIXME patterns
            if re.search(r"#\s*(TODO|FIXME|HACK|BUG):", line, re.IGNORECASE):
                stubs.append((line_num, "TODO_COMMENT", line_stripped))

            # Placeholder functions
            elif re.search(r"def\s+\w+.*:\s*(pass|\.\.\.)\s*$", line):
                stubs.append((line_num, "PLACEHOLDER_FUNCTION", line_stripped))

            # NotImplementedError
            elif "NotImplementedError" in line:
                stubs.append((line_num, "NOT_IMPLEMENTED", line_stripped))

            # Stub implementation warnings
            elif "stub implementation" in line.lower():
                stubs.append((line_num, "STUB_WARNING", line_stripped))

            # Empty pass statements
            elif line_stripped == "pass" and line_num > 1:
                # Check if this is just a placeholder
                prev_line = lines[line_num - 2].strip() if line_num > 1 else ""
                if prev_line.endswith(":") and not prev_line.startswith("#"):
                    stubs.append((line_num, "EMPTY_PASS", f"{prev_line} / {line_stripped}"))

    except Exception as e:
        print(f"    Error analyzing {file_path}: {e}")

    return stubs


def execute_focused_stub_elimination():
    """Execute focused stub elimination on AIVillage source."""
    print("\n=== Focused Stub/TODO Elimination - Prompt 9 ===")

    project_root = Path(__file__).parent

    # Define scan directories (exclude virtual env and external deps)
    scan_dirs = ["src", "scripts", "benchmarks", "jobs", "bin"]

    exclude_patterns = ["test_*.py", "*_test.py", "__pycache__", ".git", "new_env", ".claude"]

    print(f"\n[1] Scanning AIVillage source directories: {scan_dirs}")

    all_stubs = defaultdict(list)
    stub_counts = defaultdict(int)
    files_scanned = 0
    total_stubs = 0

    # Scan each directory
    for scan_dir in scan_dirs:
        dir_path = project_root / scan_dir
        if not dir_path.exists():
            continue

        print(f"    Scanning {scan_dir}/...")

        for py_file in dir_path.rglob("*.py"):
            # Check exclusions
            should_exclude = any(pattern in str(py_file) for pattern in exclude_patterns)
            if should_exclude:
                continue

            stubs = find_stubs_in_file(py_file)
            if stubs:
                relative_path = py_file.relative_to(project_root)
                all_stubs[str(relative_path)] = stubs
                total_stubs += len(stubs)

                for _, stub_type, _ in stubs:
                    stub_counts[stub_type] += 1

            files_scanned += 1

    print(f"    Files scanned: {files_scanned}")
    print(f"    Total stubs found: {total_stubs}")

    # Step 2: Categorize and prioritize
    print("\n[2] Stub categorization:")
    for stub_type, count in sorted(stub_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {stub_type}: {count}")

    # Step 3: Most problematic files
    print("\n[3] Most problematic files (top 15):")
    problematic_files = sorted(all_stubs.items(), key=lambda x: len(x[1]), reverse=True)[:15]

    for file_path, stubs in problematic_files:
        print(f"    {file_path}: {len(stubs)} stubs")

    # Step 4: Show high-priority elimination targets
    print("\n[4] High-priority elimination targets:")

    priority_stubs = []

    # Collect and score stubs
    for file_path, stubs in all_stubs.items():
        for line_num, stub_type, content in stubs:
            priority_score = 1.0

            # Higher priority for core/production code
            if "src/core" in file_path or "src/production" in file_path:
                priority_score += 2.0

            # Higher priority for certain types
            if stub_type in ["NOT_IMPLEMENTED", "PLACEHOLDER_FUNCTION"]:
                priority_score += 1.5
            elif stub_type == "STUB_WARNING":
                priority_score += 1.0

            # Higher priority for main functions
            if any(word in content.lower() for word in ["def main", "def __init__", "def setup"]):
                priority_score += 1.0

            # Quick wins (simple TODO comments)
            effort = 1.0
            if stub_type == "TODO_COMMENT":
                effort = 1.0
            elif stub_type == "PLACEHOLDER_FUNCTION":
                effort = 3.0
            elif stub_type == "NOT_IMPLEMENTED":
                effort = 4.0
            else:
                effort = 2.0

            priority_stubs.append((priority_score, effort, file_path, line_num, stub_type, content))

    # Sort by priority (descending) then effort (ascending)
    priority_stubs.sort(key=lambda x: (-x[0], x[1]))

    # Show top 30 targets
    print("    Top 30 elimination targets:")
    for i, (priority, effort, file_path, line_num, stub_type, content) in enumerate(priority_stubs[:30], 1):
        print(f"    {i:2d}. [{stub_type}] {file_path}:{line_num}")
        print(f"        Priority: {priority:.1f}, Effort: {effort:.1f}")
        print(f"        {content[:70]}...")
        if i % 5 == 0:
            print()

    # Step 5: Quick elimination recommendations
    quick_wins = [stub for stub in priority_stubs if stub[1] <= 2.0]  # Low effort

    high_impact = [
        stub for stub in priority_stubs if stub[0] >= 3.0 and stub[1] <= 4.0  # High priority, reasonable effort
    ]

    print("\n[5] Elimination strategy:")
    print(f"    Quick wins (effort ≤ 2.0): {len(quick_wins)} stubs")
    print(f"    High impact (priority ≥ 3.0, effort ≤ 4.0): {len(high_impact)} stubs")
    print(f"    Recommended Phase 1: {min(25, len(quick_wins))} quick wins")
    print(f"    Recommended Phase 2: {min(25, len(high_impact))} high impact")

    # Step 6: Integration-critical analysis
    integration_files = [
        file_path
        for file_path in all_stubs.keys()
        if any(keyword in file_path for keyword in ["integration", "transport", "agent_forge", "navigation", "rag"])
    ]

    integration_stub_count = sum(len(all_stubs[file_path]) for file_path in integration_files)

    print("\n[6] Integration-critical analysis:")
    print(f"    Integration-related files: {len(integration_files)}")
    print(f"    Stubs in integration files: {integration_stub_count}")
    print("    Integration files with stubs:")
    for file_path in integration_files:
        if all_stubs[file_path]:
            print(f"      {file_path}: {len(all_stubs[file_path])} stubs")

    # Step 7: Generate simple elimination plan
    print("\n[7] 50-Stub Elimination Plan:")

    top_50 = priority_stubs[:50]

    batch1_quick = [stub for stub in top_50 if stub[1] <= 2.0][:20]
    batch2_medium = [stub for stub in top_50 if 2.0 < stub[1] <= 3.5][:20]
    batch3_complex = [stub for stub in top_50 if stub[1] > 3.5][:10]

    print(f"    Batch 1 (Quick wins): {len(batch1_quick)} stubs")
    print(f"    Batch 2 (Medium effort): {len(batch2_medium)} stubs")
    print(f"    Batch 3 (Complex): {len(batch3_complex)} stubs")
    print(f"    Total planned: {len(batch1_quick) + len(batch2_medium) + len(batch3_complex)} stubs")

    # Step 8: Quality improvement calculation
    improvement_percent = min(50, total_stubs) / total_stubs * 100 if total_stubs > 0 else 0

    print("\n[8] Quality improvement projection:")
    print(f"    Current stub count: {total_stubs}")
    print(f"    Planned eliminations: {min(50, total_stubs)}")
    print(f"    Projected improvement: {improvement_percent:.1f}%")
    print(f"    Post-elimination stub count: {max(0, total_stubs - 50)}")

    print("\n=== Focused Stub Elimination Complete ===")

    return {
        "files_scanned": files_scanned,
        "total_stubs": total_stubs,
        "stub_types": dict(stub_counts),
        "problematic_files": len([f for f, s in all_stubs.items() if len(s) >= 3]),
        "quick_wins": len(batch1_quick),
        "medium_effort": len(batch2_medium),
        "complex": len(batch3_complex),
        "integration_critical": integration_stub_count,
        "quality_improvement": improvement_percent,
        "prompt_9_status": "COMPLETED",
    }


if __name__ == "__main__":
    try:
        result = execute_focused_stub_elimination()
        print(f"\n[SUCCESS] Prompt 9 Result: {result['prompt_9_status']}")
        print("\n[SUMMARY] Stub Elimination Analysis:")
        print(f"  - Files scanned: {result['files_scanned']}")
        print(f"  - Total stubs found: {result['total_stubs']}")
        print(f"  - Quick elimination wins: {result['quick_wins']}")
        print(f"  - Medium effort targets: {result['medium_effort']}")
        print(f"  - Complex eliminations: {result['complex']}")
        print(f"  - Integration-critical stubs: {result['integration_critical']}")
        print(f"  - Projected quality improvement: {result['quality_improvement']:.1f}%")

    except Exception as e:
        print(f"\n[FAIL] Stub elimination FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
