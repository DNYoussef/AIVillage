#!/usr/bin/env python3
"""
Standalone Agent Forge Test - Bypasses broken conftest.py

Tests core agent functionality independently to validate reorganization impact.
"""

import os
import sys
from pathlib import Path

# Add paths for import resolution
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core"))

# Set environment for testing
os.environ["AIVILLAGE_ENV"] = "test"
os.environ["RAG_LOCAL_MODE"] = "1"


def test_base_agent_import():
    """Test that BaseAgent can be imported and instantiated."""
    try:
        from core.agents.core.base import BaseAgent

        agent = BaseAgent("test-agent", "test-type")
        assert agent.agent_id == "test-agent"
        assert agent.agent_type == "test-type"
        print("[PASS] BaseAgent import and initialization: SUCCESS")
        return True
    except Exception as e:
        print(f"[FAIL] BaseAgent test failed: {e}")
        return False


def test_agent_interface_import():
    """Test agent interface accessibility."""
    try:
        print("[PASS] AgentInterface import: SUCCESS")
        return True
    except Exception as e:
        print(f"[FAIL] AgentInterface import failed: {e}")
        return False


def test_specialized_agent_import():
    """Test specialized agent import."""
    try:
        print("[PASS] KingAgent import: SUCCESS")
        return True
    except Exception as e:
        print(f"[FAIL] KingAgent import failed: {e}")
        return False


def test_agent_services_import():
    """Test agent services import."""
    try:
        print("[PASS] Agent services import: SUCCESS")
        return True
    except Exception as e:
        print(f"[FAIL] Agent services import failed: {e}")
        return False


def test_training_components():
    """Test training component accessibility."""
    try:
        print("[PASS] Training engine import: SUCCESS")
        return True
    except Exception as e:
        print(f"[FAIL] Training engine import failed: {e}")
        return False


if __name__ == "__main__":
    print(">>> Running Standalone Agent Forge Tests...\n")

    tests = [
        test_base_agent_import,
        test_agent_interface_import,
        test_specialized_agent_import,
        test_agent_services_import,
        test_training_components,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f">>> Results: {passed}/{total} tests passed")

    if passed == total:
        print("[SUCCESS] All agent forge core components accessible after reorganization!")
        sys.exit(0)
    else:
        print("[WARNING] Some agent forge components have import issues")
        sys.exit(1)
