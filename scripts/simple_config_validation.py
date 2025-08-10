#!/usr/bin/env python3
"""Simple configuration validation script for CODEX Integration.

Validates configuration files without complex dependencies.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import yaml


def validate_yaml_file(file_path):
    """Validate YAML file syntax."""
    try:
        with open(file_path, encoding="utf-8") as f:
            yaml.safe_load(f)
        return True, None
    except Exception as e:
        return False, str(e)


def validate_json_file(file_path):
    """Validate JSON file syntax."""
    try:
        with open(file_path, encoding="utf-8") as f:
            json.load(f)
        return True, None
    except Exception as e:
        return False, str(e)


def check_codex_requirements():
    """Check configuration against CODEX requirements."""
    config_dir = Path("config")
    results = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "UNKNOWN",
        "file_validation": {},
        "codex_compliance": {},
        "recommendations": [],
    }

    # Check files exist and have valid syntax
    files_to_check = [
        ("aivillage_config.yaml", validate_yaml_file),
        ("p2p_config.json", validate_json_file),
        ("rag_config.json", validate_json_file),
    ]

    all_files_valid = True

    for filename, validator in files_to_check:
        file_path = config_dir / filename
        file_result = {
            "exists": file_path.exists(),
            "valid_syntax": False,
            "size_bytes": 0,
            "errors": [],
        }

        if file_result["exists"]:
            file_result["size_bytes"] = file_path.stat().st_size
            valid, error = validator(file_path)
            file_result["valid_syntax"] = valid
            if error:
                file_result["errors"].append(error)
                all_files_valid = False
        else:
            file_result["errors"].append("File does not exist")
            all_files_valid = False

        results["file_validation"][filename] = file_result

    if not all_files_valid:
        results["overall_status"] = "CRITICAL"
        results["recommendations"].append("Fix configuration file syntax errors")
        return results

    # Load and validate configuration content
    try:
        # Load main config
        with open(config_dir / "aivillage_config.yaml") as f:
            main_config = yaml.safe_load(f)

        # Load P2P config
        with open(config_dir / "p2p_config.json") as f:
            p2p_config = json.load(f)

        # Load RAG config
        with open(config_dir / "rag_config.json") as f:
            rag_config = json.load(f)

        # CODEX compliance checks
        compliance_checks = [
            # Main configuration checks
            (
                "integration.evolution_metrics.enabled",
                main_config.get("integration", {})
                .get("evolution_metrics", {})
                .get("enabled"),
                True,
            ),
            (
                "integration.evolution_metrics.backend",
                main_config.get("integration", {})
                .get("evolution_metrics", {})
                .get("backend"),
                "sqlite",
            ),
            (
                "integration.rag_pipeline.enabled",
                main_config.get("integration", {})
                .get("rag_pipeline", {})
                .get("enabled"),
                True,
            ),
            (
                "integration.rag_pipeline.embedding_model",
                main_config.get("integration", {})
                .get("rag_pipeline", {})
                .get("embedding_model"),
                "paraphrase-MiniLM-L3-v2",
            ),
            (
                "integration.rag_pipeline.chunk_size",
                main_config.get("integration", {})
                .get("rag_pipeline", {})
                .get("chunk_size"),
                512,
            ),
            (
                "integration.p2p_networking.enabled",
                main_config.get("integration", {})
                .get("p2p_networking", {})
                .get("enabled"),
                True,
            ),
            (
                "integration.p2p_networking.transport",
                main_config.get("integration", {})
                .get("p2p_networking", {})
                .get("transport"),
                "libp2p",
            ),
            (
                "integration.p2p_networking.discovery_method",
                main_config.get("integration", {})
                .get("p2p_networking", {})
                .get("discovery_method"),
                "mdns",
            ),
            (
                "integration.p2p_networking.max_peers",
                main_config.get("integration", {})
                .get("p2p_networking", {})
                .get("max_peers"),
                50,
            ),
            (
                "integration.digital_twin.enabled",
                main_config.get("integration", {})
                .get("digital_twin", {})
                .get("enabled"),
                True,
            ),
            (
                "integration.digital_twin.encryption_enabled",
                main_config.get("integration", {})
                .get("digital_twin", {})
                .get("encryption_enabled"),
                True,
            ),
            (
                "integration.digital_twin.privacy_mode",
                main_config.get("integration", {})
                .get("digital_twin", {})
                .get("privacy_mode"),
                "strict",
            ),
            (
                "integration.digital_twin.max_profiles",
                main_config.get("integration", {})
                .get("digital_twin", {})
                .get("max_profiles"),
                10000,
            ),
            # P2P configuration checks
            ("p2p_config.host", p2p_config.get("host"), "0.0.0.0"),
            ("p2p_config.port", p2p_config.get("port"), 4001),
            (
                "p2p_config.peer_discovery.mdns_enabled",
                p2p_config.get("peer_discovery", {}).get("mdns_enabled"),
                True,
            ),
            (
                "p2p_config.transports.tcp_enabled",
                p2p_config.get("transports", {}).get("tcp_enabled"),
                True,
            ),
            (
                "p2p_config.transports.websocket_enabled",
                p2p_config.get("transports", {}).get("websocket_enabled"),
                True,
            ),
            (
                "p2p_config.security.tls_enabled",
                p2p_config.get("security", {}).get("tls_enabled"),
                True,
            ),
            (
                "p2p_config.security.peer_verification",
                p2p_config.get("security", {}).get("peer_verification"),
                True,
            ),
            # RAG configuration checks
            (
                "rag_config.embedder.model_name",
                rag_config.get("embedder", {}).get("model_name"),
                "paraphrase-MiniLM-L3-v2",
            ),
            (
                "rag_config.retrieval.vector_top_k",
                rag_config.get("retrieval", {}).get("vector_top_k"),
                20,
            ),
            (
                "rag_config.retrieval.keyword_top_k",
                rag_config.get("retrieval", {}).get("keyword_top_k"),
                20,
            ),
            (
                "rag_config.retrieval.final_top_k",
                rag_config.get("retrieval", {}).get("final_top_k"),
                10,
            ),
            (
                "rag_config.cache.l1_size",
                rag_config.get("cache", {}).get("l1_size"),
                128,
            ),
        ]

        passed_checks = 0
        failed_checks = []

        for config_path, actual_value, expected_value in compliance_checks:
            if actual_value == expected_value:
                passed_checks += 1
            else:
                failed_checks.append(
                    {
                        "path": config_path,
                        "expected": expected_value,
                        "actual": actual_value,
                    }
                )

        compliance_score = (passed_checks / len(compliance_checks)) * 100

        results["codex_compliance"] = {
            "total_checks": len(compliance_checks),
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "compliance_score": compliance_score,
        }

        # Determine overall status
        if compliance_score >= 95:
            results["overall_status"] = "EXCELLENT"
        elif compliance_score >= 85:
            results["overall_status"] = "GOOD"
        elif compliance_score >= 70:
            results["overall_status"] = "ACCEPTABLE"
        else:
            results["overall_status"] = "POOR"

        # Generate recommendations
        if failed_checks:
            results["recommendations"].append(
                f"Fix {len(failed_checks)} CODEX compliance issues"
            )
            for failure in failed_checks[:3]:  # Show top 3
                results["recommendations"].append(
                    f"Set {failure['path']} to {failure['expected']} (currently {failure['actual']})"
                )

        if compliance_score == 100:
            results["recommendations"].append("Configuration is fully CODEX compliant")

    except Exception as e:
        results["overall_status"] = "CRITICAL"
        results["recommendations"].append(f"Failed to load configurations: {e}")

    return results


def main():
    """Main validation function."""
    print("Running CODEX configuration validation...")

    results = check_codex_requirements()

    # Print summary
    print("\n" + "=" * 80)
    print("CODEX CONFIGURATION VALIDATION REPORT")
    print("=" * 80)

    print(f"\nOverall Status: {results['overall_status']}")
    print(f"Timestamp: {results['timestamp']}")

    # File validation results
    print("\nFile Validation:")
    for filename, file_result in results["file_validation"].items():
        status = "✅" if file_result["valid_syntax"] else "❌"
        print(f"  {filename}: {status} ({file_result['size_bytes']} bytes)")
        if file_result["errors"]:
            for error in file_result["errors"]:
                print(f"    Error: {error}")

    # CODEX compliance results
    if "codex_compliance" in results:
        compliance = results["codex_compliance"]
        print(f"\nCODEX Compliance: {compliance.get('compliance_score', 0):.1f}%")
        print(
            f"  Passed: {compliance.get('passed_checks', 0)}/{compliance.get('total_checks', 0)}"
        )

        if compliance.get("failed_checks"):
            print("  Failed Requirements:")
            for failure in compliance["failed_checks"][:5]:  # Show top 5
                print(
                    f"    {failure['path']}: expected {failure['expected']}, got {failure['actual']}"
                )

    # Recommendations
    if results.get("recommendations"):
        print("\nRecommendations:")
        for rec in results["recommendations"]:
            print(f"  💡 {rec}")

    print("\n" + "=" * 80)

    if results["overall_status"] in ["EXCELLENT", "GOOD"]:
        print("✅ Configuration validation passed!")
        print("All configurations are ready for CODEX integration.")
    else:
        print("❌ Configuration validation found issues!")
        print("Review and fix the issues above before proceeding.")

    print("=" * 80)

    # Save report
    report_file = Path("config") / "configuration_validation_report.json"
    report_file.parent.mkdir(exist_ok=True)
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed report saved to: {report_file}")

    return results["overall_status"] in ["EXCELLENT", "GOOD", "ACCEPTABLE"]


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
