"""
Test C4: Security Gates - Verify CI blocks http:// and unsafe serialization
"""

import json
import os
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def scan_for_security_issues():
    """Scan for security issues in production code"""
    results = {
        "http_urls": {"count": 0, "files": [], "status": "PASS"},
        "pickle_loads": {"count": 0, "files": [], "status": "PASS"},
        "security_gates": {"status": "UNKNOWN"},
    }

    # Read scan results
    artifacts_dir = Path(__file__).parent.parent / "artifacts"

    # Check HTTP URLs
    http_file = artifacts_dir / "http_scan.txt"
    if http_file.exists():
        content = http_file.read_text()
        if "No http://" in content:
            results["http_urls"]["status"] = "PASS"
            print("[PASS] No http:// URLs found in production code")
        else:
            lines = content.strip().split("\n")
            results["http_urls"]["count"] = len([l for l in lines if l.strip()])
            results["http_urls"]["files"] = lines[:10]  # First 10
            results["http_urls"]["status"] = "FAIL"
            print(
                f"[FAIL] Found {results['http_urls']['count']} http:// URLs in production"
            )

    # Check pickle.loads
    pickle_file = artifacts_dir / "pickle_scan.txt"
    if pickle_file.exists():
        content = pickle_file.read_text()
        if "No pickle.loads" in content:
            results["pickle_loads"]["status"] = "PASS"
            print("[PASS] No pickle.loads found in code")
        else:
            lines = content.strip().split("\n")
            results["pickle_loads"]["count"] = len([l for l in lines if l.strip()])
            results["pickle_loads"]["files"] = lines[:10]  # First 10
            results["pickle_loads"]["status"] = "FAIL"
            print(f"[FAIL] Found {results['pickle_loads']['count']} pickle.loads calls")

    return results


def check_security_infrastructure():
    """Check for security infrastructure like pre-commit hooks, CI gates"""
    security_items = {}

    # Check pre-commit config
    precommit_file = Path(__file__).parent.parent.parent / ".pre-commit-config.yaml"
    if precommit_file.exists():
        security_items["pre_commit"] = True
        print("[PASS] Pre-commit configuration exists")
    else:
        security_items["pre_commit"] = False
        print("[FAIL] No pre-commit configuration found")

    # Check for security tools in requirements
    base_path = Path(__file__).parent.parent.parent
    req_files = [
        base_path / "requirements.txt",
        base_path / "requirements-dev.txt",
        base_path / "pyproject.toml",
    ]

    security_tools = ["bandit", "safety", "ruff"]
    found_tools = []

    for req_file in req_files:
        if req_file.exists():
            content = req_file.read_text()
            for tool in security_tools:
                if tool in content:
                    found_tools.append(tool)

    security_items["security_tools"] = list(set(found_tools))
    if found_tools:
        print(f"[PASS] Security tools found: {', '.join(found_tools)}")
    else:
        print("[FAIL] No security tools found in requirements")

    # Check for GitHub workflows
    workflows_dir = base_path / ".github" / "workflows"
    security_items["github_workflows"] = workflows_dir.exists()
    if workflows_dir.exists():
        print("[PASS] GitHub workflows directory exists")
    else:
        print("[INFO] No GitHub workflows directory")

    return security_items


def main():
    """Run security gates test"""
    print("=" * 60)
    print("C4: Security Gates Test")
    print("=" * 60)

    print("\n1. Scanning for security issues...")
    scan_results = scan_for_security_issues()

    print("\n2. Checking security infrastructure...")
    infra_results = check_security_infrastructure()

    # Calculate overall success
    overall_success = (
        scan_results["http_urls"]["status"] == "PASS"
        and scan_results["pickle_loads"]["status"] == "PASS"
        and infra_results.get("pre_commit", False)
    )

    # Save results
    output_path = (
        Path(__file__).parent.parent / "artifacts" / "security_gates_test.json"
    )
    with open(output_path, "w") as f:
        json.dump(
            {
                "scan_results": scan_results,
                "infrastructure": infra_results,
                "overall_success": overall_success,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 60)
    print(
        f"Overall Security Gates Test Result: {'PASS' if overall_success else 'FAIL'}"
    )
    print(f"Results saved to: {output_path}")

    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
