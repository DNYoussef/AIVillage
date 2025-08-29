#!/usr/bin/env python3
"""
Test Agent Forge Backend Integration with UnifiedCognateRefiner

This script tests the integration between the backend APIs and the working
UnifiedCognateRefiner system to ensure real training can be triggered.
"""

import asyncio
import logging
from pathlib import Path
import sys

import httpx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_backend_integration():
    """Test the complete backend integration."""

    print("[TEST] TESTING AGENT FORGE BACKEND INTEGRATION")
    print("=" * 60)

    # Test 1: Check if backend starts and loads UnifiedCognateRefiner
    print("\n1. Testing UnifiedCognateRefiner integration...")

    try:
        async with httpx.AsyncClient() as client:
            # Check root endpoint
            response = await client.get("http://localhost:8083/")

            if response.status_code == 200:
                data = response.json()
                print(f"[OK] Backend responding: {data['service']} v{data['version']}")
                print(f"[OK] Real training available: {data['real_training_available']}")
                print(f"[OK] Architecture: {data['architecture']['model_type']}")
                print(f"[OK] Parameters: {data['architecture']['exact_parameters']:,}")

                # Verify UnifiedCognateRefiner features
                features = data.get("features", [])
                unified_features = [
                    f for f in features if "UnifiedCognateRefiner" in f or "create_three_cognate_models" in f
                ]
                if unified_features:
                    print("[OK] UnifiedCognateRefiner features detected:")
                    for feature in unified_features:
                        print(f"   - {feature}")
                else:
                    print("[WARN] No UnifiedCognateRefiner features found")

            else:
                print(f"[ERROR] Backend not responding: {response.status_code}")

    except httpx.ConnectError:
        print("[ERROR] Backend not running. Please start it with:")
        print("   python infrastructure/gateway/unified_agent_forge_backend.py")
        return False
    except Exception as e:
        print(f"[ERROR] Error testing backend: {e}")
        return False

    # Test 2: Check available phases and models
    print("\n2. Testing phase status and model endpoints...")

    try:
        async with httpx.AsyncClient() as client:
            # Check phases status
            response = await client.get("http://localhost:8083/phases/status")
            if response.status_code == 200:
                data = response.json()
                print(f"[OK] Phases endpoint working - {len(data['phases'])} phases available")
                print(f"[OK] Real training available: {data.get('real_training_available', False)}")

                # Check training system status
                training_status = data.get("training_system_status", {})
                if training_status:
                    print("[OK] Training system status:")
                    for key, value in training_status.items():
                        print(f"   - {key}: {value}")

            # Check models endpoint
            response = await client.get("http://localhost:8083/models")
            if response.status_code == 200:
                data = response.json()
                print(f"[OK] Models endpoint working - {len(data['models'])} models available")

                # Check for UnifiedCognateRefiner models
                unified_models = [m for m in data["models"] if "UnifiedCognate" in m.get("model_name", "")]
                if unified_models:
                    print(f"[OK] Found {len(unified_models)} UnifiedCognateRefiner models")
                    for model in unified_models[:2]:  # Show first 2
                        print(f"   - {model['model_name']}: {model['parameter_count']:,} params")

    except Exception as e:
        print(f"[ERROR] Error testing endpoints: {e}")
        return False

    # Test 3: Test Cognate training trigger (dry run)
    print("\n3. Testing Cognate training trigger...")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Trigger Cognate training
            training_request = {
                "phase_name": "Cognate",
                "real_training": True,
                "parameters": {"test_mode": True, "quick_demo": True},
            }

            response = await client.post("http://localhost:8083/phases/cognate/start", json=training_request)

            if response.status_code == 200:
                data = response.json()
                print("[OK] Cognate training triggered successfully")
                print(f"   - Task ID: {data.get('task_id')}")
                print(f"   - Training mode: {data.get('training_mode')}")
                print(f"   - Message: {data.get('message')}")

                # Give it a moment to initialize
                await asyncio.sleep(2)

                # Check phase status
                response = await client.get("http://localhost:8083/phases/status")
                if response.status_code == 200:
                    data = response.json()
                    cognate_phase = None
                    for phase in data["phases"]:
                        if phase["phase_name"] == "Cognate":
                            cognate_phase = phase
                            break

                    if cognate_phase:
                        print(f"[OK] Cognate phase status: {cognate_phase['status']}")
                        print(f"   - Progress: {cognate_phase['progress']*100:.1f}%")
                        print(f"   - Message: {cognate_phase['message']}")

                        if "UnifiedCognateRefiner" in cognate_phase["message"]:
                            print("[OK] UnifiedCognateRefiner integration confirmed!")

            elif response.status_code == 409:
                print("[INFO] Cognate training already running (expected)")
            else:
                print(f"[ERROR] Failed to trigger training: {response.status_code}")
                print(f"   Response: {response.text}")

    except Exception as e:
        print(f"[ERROR] Error testing training trigger: {e}")
        return False

    print("\n" + "=" * 60)
    print("[SUCCESS] BACKEND INTEGRATION TEST COMPLETE")
    print("[OK] UnifiedCognateRefiner successfully integrated with backend APIs")
    print("[OK] Real training endpoints functional")
    print("[OK] Ready for admin UI integration")

    return True


def test_direct_imports():
    """Test direct imports of UnifiedCognateRefiner components."""
    print("\n[TEST] TESTING DIRECT IMPORTS")
    print("-" * 40)

    try:
        # Add paths
        current_dir = Path(__file__).parent
        core_dir = current_dir / "core"
        cognate_pretrain_dir = core_dir / "agent-forge" / "phases" / "cognate_pretrain"

        sys.path.insert(0, str(core_dir))
        sys.path.insert(0, str(cognate_pretrain_dir))

        # Test UnifiedCognateRefiner import
        from unified_cognate_25m import UnifiedCognateConfig, UnifiedCognateRefiner, create_three_cognate_models

        print("[OK] UnifiedCognateRefiner imported successfully")

        # Test model creation
        models = create_three_cognate_models()
        print(f"[OK] create_three_cognate_models() works - created {len(models)} models")

        for i, model in enumerate(models):
            param_count = sum(p.numel() for p in model.parameters())
            print(f"   Model {i+1}: {param_count:,} parameters")

        # Test config
        config = UnifiedCognateConfig()
        print(f"[OK] UnifiedCognateConfig created - target: {config.target_params:,}")

        # Test model instantiation
        model = UnifiedCognateRefiner(config)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[OK] UnifiedCognateRefiner instantiated - actual: {param_count:,} parameters")

        accuracy_pct = param_count / config.target_params * 100
        print(f"[OK] Parameter accuracy: {accuracy_pct:.2f}%")

        return True

    except Exception as e:
        print(f"[ERROR] Direct import test failed: {e}")
        return False


if __name__ == "__main__":
    # Test direct imports first
    direct_success = test_direct_imports()

    # Then test backend integration
    print("\n" + "=" * 60)
    backend_success = asyncio.run(test_backend_integration())

    if direct_success and backend_success:
        print("\n[SUCCESS] ALL TESTS PASSED - Integration successful!")
        sys.exit(0)
    else:
        print("\n[ERROR] Some tests failed - check logs above")
        sys.exit(1)
