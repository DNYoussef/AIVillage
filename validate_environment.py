#!/usr/bin/env python3
"""CODEX Integration Requirements - Environment Variable Validation
Validates all environment variables per CODEX specifications
"""

import os
from typing import Any


def validate_codex_environment() -> dict[str, Any]:
    """Validate all environment variables per CODEX Integration Requirements."""
    print("[VALIDATION] Environment Variable Check Per CODEX Integration Requirements")
    print("=" * 80)

    # CODEX Required Environment Variables
    codex_vars = {
        # Debug System Variables (CODEX specified)
        "AIVILLAGE_DEBUG_MODE": {
            "required": False,
            "default": "false",
            "values": ["true", "false"],
        },
        "AIVILLAGE_LOG_LEVEL": {
            "required": False,
            "default": "INFO",
            "values": ["DEBUG", "INFO", "WARNING", "ERROR"],
        },
        "AIVILLAGE_PROFILE_PERFORMANCE": {
            "required": False,
            "default": "false",
            "values": ["true", "false"],
        },
        # Evolution Metrics System
        "AIVILLAGE_DB_PATH": {
            "required": False,
            "default": "./data/evolution_metrics.db",
        },
        "AIVILLAGE_STORAGE_BACKEND": {
            "required": False,
            "default": "sqlite",
            "values": ["sqlite", "redis", "file"],
        },
        "AIVILLAGE_REDIS_URL": {
            "required": False,
            "default": "redis://localhost:6379/0",
        },
        "AIVILLAGE_METRICS_FLUSH_THRESHOLD": {
            "required": False,
            "default": "50",
            "type": "int",
            "range": (1, 1000),
        },
        "AIVILLAGE_METRICS_FILE": {
            "required": False,
            "default": "evolution_metrics.json",
        },
        "AIVILLAGE_LOG_DIR": {"required": False, "default": "./evolution_logs"},
        # Redis Configuration
        "REDIS_HOST": {"required": False, "default": "localhost"},
        "REDIS_PORT": {
            "required": False,
            "default": "6379",
            "type": "int",
            "range": (1024, 65535),
        },
        "REDIS_DB": {
            "required": False,
            "default": "0",
            "type": "int",
            "range": (0, 15),
        },
        # RAG Pipeline System
        "RAG_CACHE_ENABLED": {
            "required": False,
            "default": "true",
            "values": ["true", "false"],
        },
        "RAG_L1_CACHE_SIZE": {
            "required": False,
            "default": "128",
            "type": "int",
            "range": (1, 1024),
        },
        "RAG_REDIS_URL": {"required": False, "default": "redis://localhost:6379/1"},
        "RAG_DISK_CACHE_DIR": {"required": False, "default": "/tmp/rag_disk_cache"},
        "RAG_EMBEDDING_MODEL": {
            "required": False,
            "default": "paraphrase-MiniLM-L3-v2",
        },
        "RAG_CROSS_ENCODER_MODEL": {
            "required": False,
            "default": "cross-encoder/ms-marco-MiniLM-L-2-v2",
        },
        "RAG_VECTOR_DIM": {
            "required": False,
            "default": "384",
            "type": "int",
            "range": (128, 1024),
        },
        "RAG_FAISS_INDEX_PATH": {"required": False, "default": "./data/faiss_index"},
        "RAG_BM25_CORPUS_PATH": {"required": False, "default": "./data/bm25_corpus"},
        "RAG_DEFAULT_K": {
            "required": False,
            "default": "10",
            "type": "int",
            "range": (1, 100),
        },
        "RAG_CHUNK_SIZE": {
            "required": False,
            "default": "512",
            "type": "int",
            "range": (128, 2048),
        },
        "RAG_CHUNK_OVERLAP": {
            "required": False,
            "default": "50",
            "type": "int",
            "range": (0, 256),
        },
        # P2P Networking
        "LIBP2P_HOST": {"required": False, "default": "0.0.0.0"},
        "LIBP2P_PORT": {
            "required": False,
            "default": "4001",
            "type": "int",
            "range": (4000, 4010),
        },
        "LIBP2P_PEER_ID_FILE": {"required": False, "default": "./data/peer_id"},
        "LIBP2P_PRIVATE_KEY_FILE": {"required": False, "default": "./data/private_key"},
        "MDNS_SERVICE_NAME": {"required": False, "default": "_aivillage._tcp"},
        "MDNS_DISCOVERY_INTERVAL": {
            "required": False,
            "default": "30",
            "type": "int",
            "range": (10, 300),
        },
        "MDNS_TTL": {
            "required": False,
            "default": "120",
            "type": "int",
            "range": (30, 600),
        },
        "MESH_MAX_PEERS": {
            "required": False,
            "default": "50",
            "type": "int",
            "range": (1, 100),
        },
        "MESH_HEARTBEAT_INTERVAL": {
            "required": False,
            "default": "10",
            "type": "int",
            "range": (1, 60),
        },
        "MESH_CONNECTION_TIMEOUT": {
            "required": False,
            "default": "30",
            "type": "int",
            "range": (5, 120),
        },
        "MESH_ENABLE_BLUETOOTH": {
            "required": False,
            "default": "true",
            "values": ["true", "false"],
        },
        "MESH_ENABLE_WIFI_DIRECT": {
            "required": False,
            "default": "true",
            "values": ["true", "false"],
        },
        "MESH_ENABLE_FILE_TRANSPORT": {
            "required": False,
            "default": "true",
            "values": ["true", "false"],
        },
        "MESH_FILE_TRANSPORT_DIR": {
            "required": False,
            "default": "/tmp/aivillage_mesh",
        },
        # Digital Twin System
        "DIGITAL_TWIN_ENCRYPTION_KEY": {
            "required": False,
            "default": None,
            "sensitive": True,
        },
        "DIGITAL_TWIN_VAULT_PATH": {"required": False, "default": "./data/vault"},
        "DIGITAL_TWIN_DB_PATH": {
            "required": False,
            "default": "./data/digital_twin.db",
        },
        "DIGITAL_TWIN_SQLITE_WAL": {
            "required": False,
            "default": "true",
            "values": ["true", "false"],
        },
        "DIGITAL_TWIN_COPPA_COMPLIANT": {
            "required": False,
            "default": "true",
            "values": ["true", "false"],
        },
        "DIGITAL_TWIN_FERPA_COMPLIANT": {
            "required": False,
            "default": "true",
            "values": ["true", "false"],
        },
        "DIGITAL_TWIN_GDPR_COMPLIANT": {
            "required": False,
            "default": "true",
            "values": ["true", "false"],
        },
        "DIGITAL_TWIN_MAX_PROFILES": {
            "required": False,
            "default": "10000",
            "type": "int",
            "range": (100, 50000),
        },
        "DIGITAL_TWIN_PROFILE_TTL_DAYS": {
            "required": False,
            "default": "365",
            "type": "int",
            "range": (30, 1095),
        },
        # Optional External Services
        "WANDB_API_KEY": {"required": False, "default": None, "sensitive": True},
    }

    # Check each variable
    results = {
        "valid": 0,
        "missing_required": 0,
        "invalid_values": 0,
        "out_of_range": 0,
        "security_issues": 0,
        "variables": {},
    }

    for var_name, config in codex_vars.items():
        value = os.getenv(var_name)
        status = "OK"
        issues = []

        # Check if required variable is missing
        if config.get("required", False) and value is None:
            status = "MISSING"
            results["missing_required"] += 1
            issues.append("Required variable is missing")

        # Use default if not set
        if value is None:
            value = config.get("default")
            if value is not None:
                status = "DEFAULT"

        if value is not None:
            # Check allowed values
            if "values" in config:
                if value not in config["values"]:
                    status = "INVALID"
                    results["invalid_values"] += 1
                    issues.append(f"Value must be one of: {config['values']}")

            # Check type and range for integers
            if config.get("type") == "int":
                try:
                    int_val = int(value)
                    if "range" in config:
                        min_val, max_val = config["range"]
                        if not (min_val <= int_val <= max_val):
                            status = "OUT_OF_RANGE"
                            results["out_of_range"] += 1
                            issues.append(
                                f"Value must be between {min_val} and {max_val}"
                            )
                except ValueError:
                    status = "INVALID"
                    results["invalid_values"] += 1
                    issues.append("Value must be an integer")

            # Check sensitive variables (don't log values)
            if config.get("sensitive", False):
                if value and len(value) < 8:
                    status = "WEAK"
                    results["security_issues"] += 1
                    issues.append("Sensitive value should be at least 8 characters")

        if status == "OK" or status == "DEFAULT":
            results["valid"] += 1

        # Store result (mask sensitive values)
        display_value = (
            "***MASKED***" if config.get("sensitive", False) and value else value
        )
        results["variables"][var_name] = {
            "value": display_value,
            "status": status,
            "issues": issues,
            "config": config,
        }

        # Print status
        status_icons = {
            "OK": "[OK]",
            "DEFAULT": "[DEFAULT]",
            "MISSING": "[MISSING]",
            "INVALID": "[INVALID]",
            "OUT_OF_RANGE": "[OUT_OF_RANGE]",
            "WEAK": "[WEAK]",
        }
        status_icon = status_icons.get(status, "[UNKNOWN]")
        print(f"{status_icon:<15} {var_name:<35} | {display_value or '(not set)'}")
        for issue in issues:
            print(f"              WARNING: {issue}")

    print()
    print("ENVIRONMENT VALIDATION SUMMARY:")
    print(f"Valid/Default Variables: {results['valid']}")
    print(f"Missing Required: {results['missing_required']}")
    print(f"Invalid Values: {results['invalid_values']}")
    print(f"Out of Range: {results['out_of_range']}")
    print(f"Security Issues: {results['security_issues']}")
    print(f"Total Variables: {len(codex_vars)}")

    # Final status
    total_issues = (
        results["missing_required"]
        + results["invalid_values"]
        + results["out_of_range"]
        + results["security_issues"]
    )
    print()
    if total_issues == 0:
        print("SUCCESS: ALL ENVIRONMENT VARIABLES PASS CODEX REQUIREMENTS")
    else:
        print(
            f"WARNING: {total_issues} ISSUES FOUND - Review and fix before production"
        )

    return results


if __name__ == "__main__":
    validate_codex_environment()
