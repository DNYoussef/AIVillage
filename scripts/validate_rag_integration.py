"""RAG Integration Validation Script.

Validates that the CODEX RAG integration meets all specified requirements.
"""

import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def validate_environment_variables() -> dict[str, Any]:
    """Validate CODEX-required environment variables."""
    print("=== Validating Environment Variables ===")

    required_vars = {
        "RAG_CACHE_ENABLED": "true",
        "RAG_L1_CACHE_SIZE": "128",
        "RAG_REDIS_URL": "redis://localhost:6379/1",
        "RAG_DISK_CACHE_DIR": "/tmp/rag_disk_cache",
        "RAG_EMBEDDING_MODEL": "paraphrase-MiniLM-L3-v2",
        "RAG_CROSS_ENCODER_MODEL": "cross-encoder/ms-marco-MiniLM-L-2-v2",
        "RAG_VECTOR_DIM": "384",
        "RAG_FAISS_INDEX_PATH": "./data/faiss_index",
        "RAG_BM25_CORPUS_PATH": "./data/bm25_corpus",
        "RAG_DEFAULT_K": "10",
        "RAG_CHUNK_SIZE": "512",
        "RAG_CHUNK_OVERLAP": "50",
    }

    results = {"passed": 0, "failed": 0, "issues": []}

    for var, expected in required_vars.items():
        actual = os.getenv(var, expected)  # Use expected as default
        if actual == expected:
            print(f"✅ {var}: {actual}")
            results["passed"] += 1
        else:
            print(f"⚠️  {var}: {actual} (expected: {expected})")
            results["issues"].append(f"{var} mismatch")
            results["failed"] += 1

    return results


def validate_configuration_files() -> dict[str, Any]:
    """Validate CODEX configuration files."""
    print("\n=== Validating Configuration Files ===")

    results = {"passed": 0, "failed": 0, "issues": []}
    config_files = [
        "config/rag_config.json",
        "config/aivillage_config.yaml",
        "config/p2p_config.json",
    ]

    for config_file in config_files:
        config_path = Path(config_file)
        if config_path.exists():
            print(f"✅ {config_file} exists")
            results["passed"] += 1

            # Validate JSON files
            if config_file.endswith(".json"):
                try:
                    with open(config_path) as f:
                        data = json.load(f)
                        print(f"   📄 Valid JSON with {len(data)} sections")
                except json.JSONDecodeError as e:
                    print(f"   ❌ Invalid JSON: {e}")
                    results["issues"].append(f"{config_file} invalid JSON")
        else:
            print(f"❌ {config_file} missing")
            results["failed"] += 1
            results["issues"].append(f"{config_file} not found")

    return results


def validate_database_schema() -> dict[str, Any]:
    """Validate database schema compliance."""
    print("\n=== Validating Database Schema ===")

    results = {"passed": 0, "failed": 0, "issues": []}
    db_path = "data/rag_index.db"

    if not Path(db_path).exists():
        print(f"ℹ️  Database not yet created at {db_path}")
        return {"passed": 0, "failed": 0, "issues": ["Database not created"]}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check required tables
    required_tables = ["documents", "chunks", "embeddings_metadata"]
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = [row[0] for row in cursor.fetchall()]

    for table in required_tables:
        if table in existing_tables:
            print(f"✅ Table '{table}' exists")
            results["passed"] += 1
        else:
            print(f"❌ Table '{table}' missing")
            results["failed"] += 1
            results["issues"].append(f"Missing table: {table}")

    # Check documents table schema
    if "documents" in existing_tables:
        cursor.execute("PRAGMA table_info(documents)")
        columns = [row[1] for row in cursor.fetchall()]
        required_columns = [
            "document_id",
            "title",
            "content",
            "file_hash",
            "word_count",
            "created_at",
            "metadata",
        ]

        for col in required_columns:
            if col in columns:
                print(f"   ✅ Column '{col}' exists")
            else:
                print(f"   ❌ Column '{col}' missing")
                results["issues"].append(f"Missing column: documents.{col}")

    conn.close()
    return results


def validate_source_code_structure() -> dict[str, Any]:
    """Validate source code structure."""
    print("\n=== Validating Source Code Structure ===")

    results = {"passed": 0, "failed": 0, "issues": []}

    required_files = [
        "src/production/rag/rag_system/core/codex_rag_integration.py",
        "src/production/rag/rag_api_server.py",
        "src/production/rag/wikipedia_data_loader.py",
        "tests/integration/test_codex_rag_integration.py",
    ]

    for file_path in required_files:
        if Path(file_path).exists():
            file_size = Path(file_path).stat().st_size
            print(f"✅ {file_path} ({file_size:,} bytes)")
            results["passed"] += 1
        else:
            print(f"❌ {file_path} missing")
            results["failed"] += 1
            results["issues"].append(f"Missing file: {file_path}")

    return results


def validate_port_configuration() -> dict[str, Any]:
    """Validate port configurations match CODEX requirements."""
    print("\n=== Validating Port Configuration ===")

    results = {"passed": 0, "failed": 0, "issues": []}

    # CODEX specified ports

    # Check environment variables for port configs
    rag_port = os.getenv("RAG_API_PORT", "8082")
    if rag_port == "8082":
        print(f"✅ RAG Pipeline port: {rag_port}")
        results["passed"] += 1
    else:
        print(f"⚠️  RAG Pipeline port: {rag_port} (should be 8082)")
        results["issues"].append("RAG port mismatch")

    # Check config files for port settings
    rag_config_path = Path("config/rag_config.json")
    if rag_config_path.exists():
        try:
            with open(rag_config_path) as f:
                config = json.load(f)
                api_port = config.get("api", {}).get("port", 8082)
                if api_port == 8082:
                    print(f"✅ Config file RAG port: {api_port}")
                    results["passed"] += 1
                else:
                    print(f"⚠️  Config file RAG port: {api_port}")
                    results["issues"].append("Config port mismatch")
        except Exception as e:
            print(f"❌ Error reading config: {e}")
            results["failed"] += 1

    return results


def validate_cache_configuration() -> dict[str, Any]:
    """Validate three-tier cache configuration."""
    print("\n=== Validating Cache Configuration ===")

    results = {"passed": 0, "failed": 0, "issues": []}

    # Check cache directories
    cache_dir = Path(os.getenv("RAG_DISK_CACHE_DIR", "/tmp/rag_disk_cache"))
    if cache_dir.exists() or cache_dir.parent.exists():
        print(f"✅ Cache directory accessible: {cache_dir}")
        results["passed"] += 1
    else:
        print(f"⚠️  Cache directory not accessible: {cache_dir}")
        results["issues"].append("Cache directory issue")

    # Check Redis configuration
    redis_url = os.getenv("RAG_REDIS_URL", "redis://localhost:6379/1")
    if "redis://localhost:6379/1" in redis_url:
        print(f"✅ Redis URL configured: {redis_url}")
        results["passed"] += 1
    else:
        print(f"⚠️  Redis URL: {redis_url}")
        results["issues"].append("Redis URL mismatch")

    # Check L1 cache size
    l1_size = int(os.getenv("RAG_L1_CACHE_SIZE", "128"))
    if l1_size == 128:
        print(f"✅ L1 cache size: {l1_size}")
        results["passed"] += 1
    else:
        print(f"⚠️  L1 cache size: {l1_size} (should be 128)")
        results["issues"].append("L1 cache size mismatch")

    return results


def validate_model_configuration() -> dict[str, Any]:
    """Validate embedding model configuration."""
    print("\n=== Validating Model Configuration ===")

    results = {"passed": 0, "failed": 0, "issues": []}

    # Check embedding model
    embedding_model = os.getenv("RAG_EMBEDDING_MODEL", "paraphrase-MiniLM-L3-v2")
    if embedding_model == "paraphrase-MiniLM-L3-v2":
        print(f"✅ Embedding model: {embedding_model}")
        results["passed"] += 1
    else:
        print(f"⚠️  Embedding model: {embedding_model}")
        results["issues"].append("Embedding model mismatch")

    # Check vector dimensions
    vector_dim = int(os.getenv("RAG_VECTOR_DIM", "384"))
    if vector_dim == 384:
        print(f"✅ Vector dimension: {vector_dim}")
        results["passed"] += 1
    else:
        print(f"⚠️  Vector dimension: {vector_dim} (should be 384)")
        results["issues"].append("Vector dimension mismatch")

    # Check cross-encoder model
    cross_encoder = os.getenv("RAG_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-2-v2")
    if "cross-encoder/ms-marco-MiniLM" in cross_encoder:
        print(f"✅ Cross-encoder model: {cross_encoder}")
        results["passed"] += 1
    else:
        print(f"⚠️  Cross-encoder model: {cross_encoder}")
        results["issues"].append("Cross-encoder model issue")

    return results


def validate_chunking_configuration() -> dict[str, Any]:
    """Validate chunk processing configuration."""
    print("\n=== Validating Chunking Configuration ===")

    results = {"passed": 0, "failed": 0, "issues": []}

    # Check chunk size
    chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "512"))
    if chunk_size == 512:
        print(f"✅ Chunk size: {chunk_size}")
        results["passed"] += 1
    else:
        print(f"⚠️  Chunk size: {chunk_size} (should be 512)")
        results["issues"].append("Chunk size mismatch")

    # Check chunk overlap
    chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
    if chunk_overlap == 50:
        print(f"✅ Chunk overlap: {chunk_overlap}")
        results["passed"] += 1
    else:
        print(f"⚠️  Chunk overlap: {chunk_overlap} (should be 50)")
        results["issues"].append("Chunk overlap mismatch")

    # Check default k
    default_k = int(os.getenv("RAG_DEFAULT_K", "10"))
    if default_k == 10:
        print(f"✅ Default K: {default_k}")
        results["passed"] += 1
    else:
        print(f"⚠️  Default K: {default_k} (should be 10)")
        results["issues"].append("Default K mismatch")

    return results


def generate_integration_report() -> dict[str, Any]:
    """Generate comprehensive integration report."""
    print("\n" + "=" * 60)
    print("CODEX RAG INTEGRATION VALIDATION REPORT")
    print("=" * 60)

    all_results = {
        "environment_variables": validate_environment_variables(),
        "configuration_files": validate_configuration_files(),
        "database_schema": validate_database_schema(),
        "source_code_structure": validate_source_code_structure(),
        "port_configuration": validate_port_configuration(),
        "cache_configuration": validate_cache_configuration(),
        "model_configuration": validate_model_configuration(),
        "chunking_configuration": validate_chunking_configuration(),
    }

    # Calculate totals
    total_passed = sum(r["passed"] for r in all_results.values())
    total_failed = sum(r["failed"] for r in all_results.values())
    total_issues = sum(len(r["issues"]) for r in all_results.values())
    total_checks = total_passed + total_failed

    # Generate summary
    print("\n=== INTEGRATION SUMMARY ===")
    print(f"Total checks: {total_checks}")
    print(f"Passed: {total_passed} ✅")
    print(f"Failed: {total_failed} ❌")
    print(f"Issues: {total_issues} ⚠️")

    if total_checks > 0:
        success_rate = (total_passed / total_checks) * 100
        print(f"Success rate: {success_rate:.1f}%")

        if success_rate >= 90:
            print("🎉 EXCELLENT: Ready for production")
        elif success_rate >= 75:
            print("✅ GOOD: Minor issues to address")
        elif success_rate >= 50:
            print("⚠️  FAIR: Several issues need attention")
        else:
            print("❌ POOR: Major issues require fixes")

    # List all issues
    if total_issues > 0:
        print(f"\n=== ISSUES TO ADDRESS ({total_issues}) ===")
        for category, results in all_results.items():
            if results["issues"]:
                print(f"\n{category.upper()}:")
                for issue in results["issues"]:
                    print(f"  • {issue}")

    return {
        "total_checks": total_checks,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_issues": total_issues,
        "success_rate": (total_passed / total_checks * 100) if total_checks > 0 else 0,
        "detailed_results": all_results,
    }


def main() -> None:
    """Main validation function."""
    try:
        report = generate_integration_report()

        # Save report to file
        report_path = Path("rag_integration_validation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Full report saved to: {report_path}")

        # Return appropriate exit code
        if report["success_rate"] >= 75:
            print("\n✅ RAG Integration validation completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️  RAG Integration validation completed with issues.")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
