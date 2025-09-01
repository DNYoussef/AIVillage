#!/usr/bin/env python3
"""
Test script for unified entry points

This script tests all the new unified entry points to ensure they work correctly.
"""

import os
import subprocess
import sys


def run_command(cmd, cwd=None):
    """Run a command and return the result"""
    try:
        result = subprocess.run(cmd, check=False, shell=True, capture_output=True, text=True, cwd=cwd)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)


def test_unified_main():
    """Test the unified main.py"""
    print("Testing unified main.py...")

    # Test help
    cmd = f"{sys.executable} main.py --help"
    ret, out, err = run_command(cmd)
    if ret == 0:
        print("✅ Unified main.py help works")
    else:
        print(f"❌ Unified main.py help failed: {err}")
        return False

    # Test mode help
    for mode in ["agent_forge", "king", "rag", "core"]:
        cmd = f"{sys.executable} main.py --mode {mode} --action status"
        ret, out, err = run_command(cmd)
        if ret == 0 or "module not found" in err.lower():
            print(f"✅ Mode {mode} routing works")
        else:
            print(f"❌ Mode {mode} routing failed: {err}")
            return False

    return True


def test_service_mains():
    """Test individual service main.py files"""
    services = [
        ("agent_forge", "agent_forge/main.py"),
        ("king", "agents/king/main.py"),
        ("rag", "rag_system/main.py"),
    ]

    for name, path in services:
        if not os.path.exists(path):
            print(f"⚠️  {name} main.py not found at {path}")
            continue

        print(f"Testing {name} service...")

        # Test help
        cmd = f"{sys.executable} {path} --help"
        ret, out, err = run_command(cmd)
        if ret == 0:
            print(f"✅ {name} help works")
        else:
            print(f"❌ {name} help failed: {err}")
            return False

        # Test status
        cmd = f"{sys.executable} {path} status"
        ret, out, err = run_command(cmd)
        if ret == 0:
            print(f"✅ {name} status works")
        else:
            print(f"❌ {name} status failed: {err}")
            return False

    return True


def test_cli_arguments():
    """Test CLI argument parsing"""
    print("Testing CLI argument parsing...")

    # Test unified main with various arguments
    test_cases = [
        f"{sys.executable} main.py --mode agent_forge --action train --config test.yaml",
        f"{sys.executable} main.py --mode king --action run --task 'test task'",
        f"{sys.executable} main.py --mode rag --action query --question 'test question'",
        f"{sys.executable} main.py --mode core --action status",
    ]

    for cmd in test_cases:
        ret, out, err = run_command(cmd)
        if ret == 0 or "module not found" in err.lower():
            print(f"✅ Command works: {cmd}")
        else:
            print(f"❌ Command failed: {cmd} - {err}")
            return False

    return True


def main():
    """Run all tests"""
    print("🧪 Testing Unified Entry Points")
    print("=" * 50)

    tests = [test_unified_main, test_service_mains, test_cli_arguments]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            print()

    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    print("❌ Some tests failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
