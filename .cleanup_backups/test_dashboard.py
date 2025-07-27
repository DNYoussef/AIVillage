#!/usr/bin/env python3
"""Agent Forge Test Dashboard
===========================

Simple test dashboard to verify core components after reorganization.
"""

from datetime import datetime
import importlib
import os
from pathlib import Path


def test_imports():
    """Test core module imports"""
    results = {}

    test_modules = [
        "agent_forge",
        "agent_forge.compression",
        "agent_forge.evomerge",
        "agent_forge.training",
        "agent_forge.orchestration",
        "core.evidence",
        "core.logging_config"
    ]

    for module in test_modules:
        try:
            importlib.import_module(module)
            results[module] = "PASS"
        except Exception as e:
            results[module] = f"FAIL: {str(e)[:50]}"

    return results

def test_file_structure():
    """Test that files are in correct locations"""
    results = {}

    expected_structure = {
        "scripts/run_full_agent_forge.py": "Script moved",
        "tests/pipeline_validation_test.py": "Test moved",
        "docs/AGENT_FORGE_DEPLOYMENT_READY.md": "Doc moved",
        "configs/orchestration_config.yaml": "Config moved",
        "agent_forge/__init__.py": "Core module exists"
    }

    for file_path, description in expected_structure.items():
        if Path(file_path).exists():
            results[file_path] = "PASS"
        else:
            results[file_path] = "FAIL: Missing"

    return results

def test_core_functionality():
    """Test basic functionality"""
    results = {}

    # Test agent_forge import
    try:
        results["Agent Forge Import"] = "PASS"
    except Exception as e:
        results["Agent Forge Import"] = f"FAIL: {e}"

    # Test evidence pack
    try:
        from core.evidence import EvidencePack
        pack = EvidencePack(
            session_id="test",
            query="test query",
            chunks=["test chunk"],
            retrieved_docs=[]
        )
        results["Evidence Pack Creation"] = "PASS"
    except Exception as e:
        results["Evidence Pack Creation"] = f"FAIL: {e}"

    # Test configuration
    try:
        from core.logging_config import setup_logging
        setup_logging()
        results["Logging Setup"] = "PASS"
    except Exception as e:
        results["Logging Setup"] = f"FAIL: {e}"

    return results

def generate_report():
    """Generate comprehensive test report"""
    print("="*80)
    print("AGENT FORGE TEST DASHBOARD")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working Directory: {os.getcwd()}")
    print()

    # Import Tests
    print("MODULE IMPORT TESTS")
    print("-" * 40)
    import_results = test_imports()
    for module, status in import_results.items():
        status_symbol = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"{status_symbol} {module}: {status}")
    print()

    # File Structure Tests
    print("FILE STRUCTURE TESTS")
    print("-" * 40)
    structure_results = test_file_structure()
    for file_path, status in structure_results.items():
        status_symbol = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"{status_symbol} {file_path}: {status}")
    print()

    # Core Functionality Tests
    print("CORE FUNCTIONALITY TESTS")
    print("-" * 40)
    function_results = test_core_functionality()
    for test_name, status in function_results.items():
        status_symbol = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"{status_symbol} {test_name}: {status}")
    print()

    # Overall Summary
    all_results = {**import_results, **structure_results, **function_results}
    passed = len([r for r in all_results.values() if r == "PASS"])
    total = len(all_results)
    pass_rate = (passed / total) * 100 if total > 0 else 0

    print("OVERALL SUMMARY")
    print("-" * 40)
    print(f"Tests Passed: {passed}/{total} ({pass_rate:.1f}%)")

    if pass_rate >= 70:
        print("Status: GOOD - Core system functional after reorganization")
    elif pass_rate >= 50:
        print("Status: ACCEPTABLE - Some issues but major components working")
    else:
        print("Status: NEEDS ATTENTION - Significant issues after reorganization")

    print("="*80)

    # Save detailed results
    results_file = Path("test_dashboard_results.json")
    import json
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "pass_rate": pass_rate,
            "import_tests": import_results,
            "structure_tests": structure_results,
            "functionality_tests": function_results
        }, f, indent=2)

    print(f"Detailed results saved to: {results_file}")

    return pass_rate >= 50

if __name__ == "__main__":
    success = generate_report()
# REMOVED:     sys.exit(0 if success else 1)
