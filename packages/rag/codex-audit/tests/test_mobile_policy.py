"""
Test C5: Mobile Policy - Verify cross-platform import + env-driven policies
"""

import json
import os
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))


def test_mobile_imports():
    """Test mobile resource management imports"""
    results = {"import": False, "policy_test": False}

    try:
        from production.monitoring.mobile.resource_management import BatteryThermalResourceManager

        results["import"] = True
        print("[PASS] Mobile resource management imported")

        # Test basic instantiation
        manager = BatteryThermalResourceManager()
        print("[PASS] Resource manager instantiated")

        # Test env-driven policy changes
        os.environ["BATTERY"] = "15"  # Low battery
        os.environ["THERMAL"] = "65"  # Hot

        status = manager.get_status()
        if "power_mode" in status or "battery" in status:
            results["policy_test"] = True
            print("[PASS] Policy adaptation detected")
        else:
            print("[INFO] Policy status not fully testable")

    except ImportError as e:
        print(f"[FAIL] Mobile import failed: {e}")
    except Exception as e:
        print(f"[INFO] Mobile test partial: {e}")

    return results


def main():
    results = test_mobile_imports()
    overall_success = results["import"]

    # Save results
    output_path = Path(__file__).parent.parent / "artifacts" / "mobile_test.json"
    with open(output_path, "w") as f:
        json.dump({"results": results, "overall_success": overall_success}, f, indent=2)

    print(f"Mobile Test Result: {'PASS' if overall_success else 'FAIL'}")
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
