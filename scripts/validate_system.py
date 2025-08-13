#!/usr/bin/env python3
"""System Validation Script.

Tests core functionality to ensure the system is working properly.
"""

import json
import sys
import traceback
from pathlib import Path


def test_imports():
    """Test that core modules can be imported."""
    print("Testing core imports...")

    sys.path.insert(0, "src")

    tests = [
        ("communications.protocol", "Communication protocol"),
        ("core.p2p.p2p_node", "P2P networking"),
        ("production.agent_forge.agent_factory", "Agent factory"),
        ("production.compression.compression_pipeline", "Compression pipeline"),
        ("production.evolution.evomerge_pipeline", "Evolution pipeline"),
    ]

    passed = 0
    failed = 0

    for module, description in tests:
        try:
            __import__(module)
            print(f"  PASS: {description}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {description} - {e}")
            failed += 1

    return passed, failed


def test_configurations():
    """Test that configuration files are valid."""
    print("\nTesting configuration files...")

    configs = [
        ("src/production/agent_forge/templates/master_config.json", "Agent templates"),
        ("pyproject.toml", "Project configuration"),
        (".pre-commit-config.yaml", "Pre-commit configuration"),
    ]

    passed = 0
    failed = 0

    for config_path, description in configs:
        try:
            if config_path.endswith(".json"):
                with open(config_path) as f:
                    json.load(f)
                print(f"  PASS: {description} - Valid JSON")
            elif config_path.endswith(".toml"):
                # Basic existence check for TOML
                if Path(config_path).exists():
                    print(f"  PASS: {description} - File exists")
                else:
                    msg = f"{config_path} not found"
                    raise FileNotFoundError(msg)
            elif config_path.endswith(".yaml"):
                if Path(config_path).exists():
                    print(f"  PASS: {description} - File exists")
                else:
                    msg = f"{config_path} not found"
                    raise FileNotFoundError(msg)
            passed += 1
        except Exception as e:
            print(f"  FAIL: {description} - {e}")
            failed += 1

    return passed, failed


def test_agent_templates():
    """Test that all 18 agent templates exist."""
    print("\nTesting agent templates...")

    try:
        with open("src/production/agent_forge/templates/master_config.json") as f:
            config = json.load(f)

        expected_agents = config["agent_types"]
        template_dir = Path("src/production/agent_forge/templates")

        passed = 0
        failed = 0

        for agent_type in expected_agents:
            template_file = template_dir / f"{agent_type}_template.json"
            if template_file.exists():
                try:
                    with open(template_file) as f:
                        json.load(f)
                    print(f"  PASS: {agent_type} template")
                    passed += 1
                except Exception as e:
                    print(f"  FAIL: {agent_type} template - Invalid JSON: {e}")
                    failed += 1
            else:
                print(f"  FAIL: {agent_type} template - File missing")
                failed += 1

        print(f"\nAgent templates: {passed}/{len(expected_agents)} found")
        return passed, failed

    except Exception as e:
        print(f"  FAIL: Could not load master config - {e}")
        return 0, 1


def test_project_structure():
    """Test that key directories and files exist."""
    print("\nTesting project structure...")

    required_paths = [
        "src/",
        "src/production/",
        "src/communications/",
        "src/core/",
        "tests/",
        "docs/",
        ".github/workflows/",
        "pyproject.toml",
        "README.md",
        "Makefile",
    ]

    passed = 0
    failed = 0

    for path in required_paths:
        if Path(path).exists():
            print(f"  PASS: {path}")
            passed += 1
        else:
            print(f"  FAIL: {path} missing")
            failed += 1

    return passed, failed


def main() -> int:
    """Run all validation tests."""
    print("=" * 50)
    print("AIVillage System Validation")
    print("=" * 50)

    total_passed = 0
    total_failed = 0

    # Run all tests
    tests = [
        test_imports,
        test_configurations,
        test_agent_templates,
        test_project_structure,
    ]

    for test_func in tests:
        try:
            passed, failed = test_func()
            total_passed += passed
            total_failed += failed
        except Exception as e:
            print(f"\nERROR in {test_func.__name__}: {e}")
            traceback.print_exc()
            total_failed += 1

    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Total tests passed: {total_passed}")
    print(f"Total tests failed: {total_failed}")

    if total_failed == 0:
        print("\nSUCCESS: All validation tests passed!")
        print("The system appears to be properly configured.")
        return 0
    print(f"\nFAILURE: {total_failed} tests failed.")
    print("Please address the issues above before proceeding.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
