#!/usr/bin/env python3
"""Verify Sprints 8-9 are actually complete."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=False
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def verify_sprint_8():
    print("Verifying Sprint 8: Emergency Stabilization")
    print("-" * 50)
    checks = []
    ci_file = Path(".github/workflows/basic-ci.yml")
    if ci_file.exists():
        content = ci_file.read_text()
        checks.append(("CI doesn't hide failures", "|| true" not in content))
    else:
        checks.append(("CI file exists", False))
    success, _, _ = run_command("python -m pytest --version")
    checks.append(("pytest installed", success))
    success, out, _ = run_command("grep -r 'NotImplementedError' src/ | wc -l")
    count = int(out.strip()) if out.strip().isdigit() else 999
    checks.append(("No NotImplementedError", count == 0))
    success, _, _ = run_command("python -m pytest tests/test_sanity.py -v")
    checks.append(("Sanity tests pass", success))
    passed = sum(1 for _, r in checks if r)
    print(f"\nSprint 8 Results: {passed}/{len(checks)} passed")
    for check, result in checks:
        print(f"  {'✓' if result else '✗'} {check}")
    return passed == len(checks)


def verify_sprint_9():
    print("\n\nVerifying Sprint 9: Working Compression")
    print("-" * 50)
    checks = []
    quantizer_path = Path("src/core/compression/simple_quantizer.py")
    checks.append(("SimpleQuantizer exists", quantizer_path.exists()))
    try:
        sys.path.insert(0, "src")
        from core.compression import SimpleQuantizer  # noqa: F401

        checks.append(("SimpleQuantizer imports", True))
    except Exception:
        checks.append(("SimpleQuantizer imports", False))
    success, _, _ = run_command(
        "PYTHONPATH=src python -m pytest tests/test_compression.py -v"
    )
    checks.append(("Compression tests pass", success))
    success, out, _ = run_command("python scripts/benchmark_compression.py")
    checks.append(("Achieves 4x compression", "4." in out))
    passed = sum(1 for _, r in checks if r)
    print(f"\nSprint 9 Results: {passed}/{len(checks)} passed")
    for check, result in checks:
        print(f"  {'✓' if result else '✗'} {check}")
    return passed == len(checks)


def main() -> int:
    sprint8_done = verify_sprint_8()
    sprint9_done = verify_sprint_9()
    print("\n" + "=" * 50)
    print("FINAL VERIFICATION")
    print("=" * 50)
    print(
        f"Sprint 8 (Emergency Stabilization): {'✓ COMPLETE' if sprint8_done else '✗ INCOMPLETE'}"
    )
    print(
        f"Sprint 9 (Working Compression): {'✓ COMPLETE' if sprint9_done else '✗ INCOMPLETE'}"
    )
    if sprint8_done and sprint9_done:
        print("\n🎉 Both sprints successfully completed!")
        return 0
    print("\n❌ Sprints incomplete - see failures above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
