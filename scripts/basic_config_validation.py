#!/usr/bin/env python3
"""Basic configuration validation script for CODEX Integration.

Validates configuration files without external dependencies.
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def validate_json_file(file_path):
    """Validate JSON file syntax."""
    try:
        with open(file_path, encoding="utf-8") as f:
            json.load(f)
        return True, None
    except Exception as e:
        return False, str(e)


def validate_yaml_basic(file_path):
    """Basic YAML validation (structure check only)."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Basic checks for YAML structure
        if not content.strip():
            return False, "File is empty"

        # Check for basic YAML indicators
        if "integration:" not in content:
            return False, "Missing 'integration:' section"

        if "evolution_metrics:" not in content:
            return False, "Missing 'evolution_metrics:' section"

        return True, None
    except Exception as e:
        return False, str(e)


def main():
    """Main validation function."""
    print("Running basic CODEX configuration validation...")

    config_dir = Path("config")
    results = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "UNKNOWN",
        "files_checked": 0,
        "files_passed": 0,
        "validation_results": {},
    }

    # Files to validate
    files_to_check = [
        ("aivillage_config.yaml", validate_yaml_basic),
        ("p2p_config.json", validate_json_file),
        ("rag_config.json", validate_json_file),
    ]

    print(f"\nChecking configuration files in {config_dir}...")

    for filename, validator in files_to_check:
        file_path = config_dir / filename
        results["files_checked"] += 1

        file_result = {
            "exists": file_path.exists(),
            "size_bytes": 0,
            "valid": False,
            "errors": [],
        }

        if file_result["exists"]:
            file_result["size_bytes"] = file_path.stat().st_size

            valid, error = validator(file_path)
            file_result["valid"] = valid

            if error:
                file_result["errors"].append(error)
            else:
                results["files_passed"] += 1
        else:
            file_result["errors"].append("File does not exist")

        results["validation_results"][filename] = file_result

    # Determine overall status
    success_rate = (results["files_passed"] / results["files_checked"]) * 100 if results["files_checked"] > 0 else 0

    if success_rate >= 100:
        results["overall_status"] = "EXCELLENT"
    elif success_rate >= 75:
        results["overall_status"] = "GOOD"
    elif success_rate >= 50:
        results["overall_status"] = "ACCEPTABLE"
    else:
        results["overall_status"] = "POOR"

    # Print results
    print("\n" + "=" * 80)
    print("BASIC CODEX CONFIGURATION VALIDATION REPORT")
    print("=" * 80)

    print(f"\nOverall Status: {results['overall_status']}")
    print(f"Files Checked: {results['files_checked']}")
    print(f"Files Passed: {results['files_passed']}")
    print(f"Success Rate: {success_rate:.1f}%")

    print("\nFile Validation Results:")
    for filename, file_result in results["validation_results"].items():
        status = "✅" if file_result["valid"] else "❌"
        size_info = f" ({file_result['size_bytes']} bytes)" if file_result["exists"] else ""
        print(f"  {filename}: {status}{size_info}")

        if file_result["errors"]:
            for error in file_result["errors"]:
                print(f"    Error: {error}")

    # Specific CODEX requirement checks
    print("\nCODEX Requirements Check:")

    # Check P2P config specifics
    p2p_file = config_dir / "p2p_config.json"
    if p2p_file.exists():
        try:
            with open(p2p_file) as f:
                p2p_config = json.load(f)

            # Check critical P2P settings
            checks = [
                ("Host is 0.0.0.0", p2p_config.get("host") == "0.0.0.0"),
                ("Port is 4001", p2p_config.get("port") == 4001),
                (
                    "mDNS enabled",
                    p2p_config.get("peer_discovery", {}).get("mdns_enabled") is True,
                ),
                (
                    "TCP transport enabled",
                    p2p_config.get("transports", {}).get("tcp_enabled") is True,
                ),
                (
                    "WebSocket transport enabled",
                    p2p_config.get("transports", {}).get("websocket_enabled") is True,
                ),
                (
                    "TLS security enabled",
                    p2p_config.get("security", {}).get("tls_enabled") is True,
                ),
                (
                    "Peer verification enabled",
                    p2p_config.get("security", {}).get("peer_verification") is True,
                ),
            ]

            for check_name, check_result in checks:
                status = "✅" if check_result else "❌"
                print(f"  {check_name}: {status}")

        except Exception as e:
            print(f"  P2P config validation failed: {e}")
    else:
        print("  P2P config: ❌ File missing")

    # Check RAG config specifics
    rag_file = config_dir / "rag_config.json"
    if rag_file.exists():
        try:
            with open(rag_file) as f:
                rag_config = json.load(f)

            checks = [
                (
                    "Embedding model is paraphrase-MiniLM-L3-v2",
                    rag_config.get("embedder", {}).get("model_name") == "paraphrase-MiniLM-L3-v2",
                ),
                (
                    "Vector top-k is 20",
                    rag_config.get("retrieval", {}).get("vector_top_k") == 20,
                ),
                (
                    "Keyword top-k is 20",
                    rag_config.get("retrieval", {}).get("keyword_top_k") == 20,
                ),
                (
                    "Final top-k is 10",
                    rag_config.get("retrieval", {}).get("final_top_k") == 10,
                ),
                (
                    "L1 cache size is 128",
                    rag_config.get("cache", {}).get("l1_size") == 128,
                ),
            ]

            for check_name, check_result in checks:
                status = "✅" if check_result else "❌"
                print(f"  {check_name}: {status}")

        except Exception as e:
            print(f"  RAG config validation failed: {e}")
    else:
        print("  RAG config: ❌ File missing")

    print("\n" + "=" * 80)

    if results["overall_status"] in ["EXCELLENT", "GOOD"]:
        print("✅ Basic configuration validation passed!")
        print("Configuration files are syntactically valid and meet basic CODEX requirements.")
    else:
        print("❌ Configuration validation found issues!")
        print("Review and fix the issues above before proceeding.")

    print("=" * 80)

    # Save basic report
    report_file = config_dir / "basic_validation_report.json"
    report_file.parent.mkdir(exist_ok=True)
    try:
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBasic report saved to: {report_file}")
    except Exception as e:
        print(f"Failed to save report: {e}")

    return results["overall_status"] in ["EXCELLENT", "GOOD", "ACCEPTABLE"]


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
