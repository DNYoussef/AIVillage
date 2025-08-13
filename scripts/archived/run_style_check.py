#!/usr/bin/env python3
"""Style Check Runner for Automation Scripts
Convenient wrapper for running comprehensive style checks on automation scripts.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_style_check(
    script_name: str = None, fix: bool = False, verbose: bool = False
) -> int:
    """Run style check on automation scripts"""
    cmd = [sys.executable, "scripts/enforce_style_guide.py"]

    if script_name:
        cmd.extend(["--script", script_name])

    if fix:
        cmd.append("--fix")

    if verbose:
        cmd.extend(["--report", "style_check_report.txt"])

    try:
        result = subprocess.run(cmd, check=False, cwd=Path.cwd())
        return result.returncode
    except Exception as e:
        print(f"Error running style check: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run style checks on automation scripts"
    )
    parser.add_argument("script", nargs="?", help="Specific script to check (optional)")
    parser.add_argument(
        "--fix", action="store_true", help="Auto-fix issues where possible"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Generate detailed report"
    )

    args = parser.parse_args()

    print("Running automation scripts style check...")

    exit_code = run_style_check(args.script, args.fix, args.verbose)

    if exit_code == 0:
        print("Style check passed!")
    else:
        print("Style check failed!")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
