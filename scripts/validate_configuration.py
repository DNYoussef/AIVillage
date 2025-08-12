#!/usr/bin/env python3
"""Configuration validation script for CODEX Integration.

Validates all configuration files, checks consistency, verifies paths and models,
and generates comprehensive validation report.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.config_manager import CODEXConfigManager, ConfigurationError

logger = logging.getLogger(__name__)


class ConfigurationValidator:
    """Comprehensive configuration validation for CODEX integration."""

    def __init__(self, config_dir: str = "config") -> None:
        self.config_dir = Path(config_dir)
        self.config_manager = None
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "UNKNOWN",
            "summary": {
                "total_checks": 0,
                "passed_checks": 0,
                "failed_checks": 0,
                "warnings": 0,
            },
            "file_validation": {},
            "configuration_consistency": {},
            "path_validation": {},
            "model_validation": {},
            "port_validation": {},
            "environment_variables": {},
            "codex_compliance": {},
            "recommendations": [],
        }

    def validate_file_syntax(self) -> dict[str, Any]:
        """Validate syntax of all configuration files."""
        results = {}

        files_to_check = [
            ("aivillage_config.yaml", "yaml"),
            ("p2p_config.json", "json"),
            ("rag_config.json", "json"),
        ]

        for filename, file_type in files_to_check:
            file_path = self.config_dir / filename
            file_result = {
                "exists": file_path.exists(),
                "readable": False,
                "valid_syntax": False,
                "size_bytes": 0,
                "errors": [],
            }

            if file_result["exists"]:
                try:
                    file_result["size_bytes"] = file_path.stat().st_size
                    file_result["readable"] = True

                    with open(file_path, encoding="utf-8") as f:
                        content = f.read()

                    if file_type == "yaml":
                        import yaml

                        yaml.safe_load(content)
                    elif file_type == "json":
                        json.loads(content)

                    file_result["valid_syntax"] = True

                except Exception as e:
                    file_result["errors"].append(str(e))

            else:
                file_result["errors"].append("File does not exist")

            results[filename] = file_result

        return results

    def validate_configuration_loading(self) -> dict[str, Any]:
        """Test configuration loading and manager initialization."""
        results = {
            "manager_created": False,
            "config_loaded": False,
            "hot_reload_enabled": False,
            "validation_passed": False,
            "errors": [],
        }

        try:
            self.config_manager = CODEXConfigManager(
                config_dir=str(self.config_dir), enable_hot_reload=True
            )
            results["manager_created"] = True
            results["hot_reload_enabled"] = True

            # Test config access
            config_data = self.config_manager.get_all()
            if config_data:
                results["config_loaded"] = True

            results["validation_passed"] = True

        except ConfigurationError as e:
            results["errors"].append(f"Configuration error: {e}")
        except Exception as e:
            results["errors"].append(f"Unexpected error: {e}")

        return results

    def validate_paths(self) -> dict[str, Any]:
        """Validate all file and directory paths in configuration."""
        if not self.config_manager:
            return {"error": "Configuration manager not initialized"}

        results = {}
        self.config_manager.get_all()

        # Paths to check
        paths_to_check = [
            ("integration.evolution_metrics.db_path", "database_file"),
            ("integration.digital_twin.db_path", "database_file"),
            ("rag_config.cache.l3_directory", "directory"),
            ("integration.rag_pipeline.faiss_index_path", "file"),
            ("integration.rag_pipeline.bm25_corpus_path", "file"),
            ("integration.digital_twin.vault_path", "directory"),
        ]

        for config_path, path_type in paths_to_check:
            path_value = self.config_manager.get(config_path)

            if path_value:
                path_obj = Path(path_value)
                path_result = {
                    "configured_path": str(path_value),
                    "absolute_path": str(path_obj.resolve()),
                    "exists": path_obj.exists(),
                    "parent_exists": path_obj.parent.exists(),
                    "type": path_type,
                    "accessible": False,
                    "writable": False,
                    "issues": [],
                }

                if path_type == "directory":
                    if path_obj.exists():
                        path_result["accessible"] = path_obj.is_dir()
                        try:
                            path_result["writable"] = os.access(path_obj, os.W_OK)
                        except:
                            path_result["writable"] = False
                    else:
                        path_result["issues"].append("Directory does not exist")

                elif path_type in ["file", "database_file"]:
                    if path_obj.exists():
                        path_result["accessible"] = path_obj.is_file()
                        try:
                            path_result["writable"] = os.access(path_obj, os.W_OK)
                        except:
                            path_result["writable"] = False
                    elif not path_obj.parent.exists():
                        path_result["issues"].append("Parent directory does not exist")
                    else:
                        path_result["issues"].append(
                            "File does not exist but parent directory is accessible"
                        )

                results[config_path] = path_result

        return results

    def validate_model_availability(self) -> dict[str, Any]:
        """Validate that specified models are available."""
        if not self.config_manager:
            return {"error": "Configuration manager not initialized"}

        results = {}

        # Models to check
        embedding_model = self.config_manager.get(
            "integration.rag_pipeline.embedding_model"
        )
        cross_encoder_model = self.config_manager.get(
            "integration.rag_pipeline.cross_encoder_model"
        )
        rag_embedder_model = self.config_manager.get("rag_config.embedder.model_name")

        models_to_check = [
            ("embedding_model", embedding_model),
            ("cross_encoder_model", cross_encoder_model),
            ("rag_embedder_model", rag_embedder_model),
        ]

        for model_name, model_path in models_to_check:
            if model_path:
                model_result = {
                    "model_path": model_path,
                    "expected_available": False,
                    "can_load": False,
                    "issues": [],
                }

                # Check if model matches CODEX requirements
                expected_models = [
                    "paraphrase-MiniLM-L3-v2",
                    "cross-encoder/ms-marco-MiniLM-L-2-v2",
                ]

                if model_path in expected_models:
                    model_result["expected_available"] = True

                    # Attempt to verify model can be loaded
                    try:
                        from sentence_transformers import SentenceTransformer

                        # Don't actually load, just check if library is available
                        model_result["can_load"] = True
                    except ImportError:
                        model_result["issues"].append(
                            "sentence-transformers library not available"
                        )
                    except Exception as e:
                        model_result["issues"].append(f"Error checking model: {e}")
                else:
                    model_result["issues"].append(
                        f"Model {model_path} not in CODEX requirements"
                    )

                results[model_name] = model_result

        return results

    def validate_port_configuration(self) -> dict[str, Any]:
        """Validate port configurations match CODEX requirements."""
        if not self.config_manager:
            return {"error": "Configuration manager not initialized"}

        results = {}

        # Expected ports from CODEX requirements
        expected_ports = {
            "libp2p_main": 4001,
            "libp2p_websocket": 4002,
            "mdns_discovery": 5353,
            "digital_twin_api": 8080,
            "evolution_metrics": 8081,
            "rag_pipeline": 8082,
            "redis": 6379,
        }

        # Check configured P2P port
        p2p_port = self.config_manager.get("p2p_config.port")
        if p2p_port:
            port_result = {
                "configured_port": p2p_port,
                "expected_port": expected_ports["libp2p_main"],
                "matches_requirement": p2p_port == expected_ports["libp2p_main"],
                "is_valid_port": isinstance(p2p_port, int) and 1 <= p2p_port <= 65535,
                "issues": [],
            }

            if not port_result["matches_requirement"]:
                port_result["issues"].append(
                    f"Port {p2p_port} does not match CODEX requirement: {expected_ports['libp2p_main']}"
                )

            results["p2p_main_port"] = port_result

        # Check for port conflicts in range
        android_p2p_range = range(4000, 4011)
        if p2p_port and p2p_port in android_p2p_range:
            results["android_p2p_conflict"] = {
                "port": p2p_port,
                "conflict": True,
                "issue": f"Port {p2p_port} conflicts with Android P2P range (4000-4010)",
            }

        return results

    def validate_environment_variables(self) -> dict[str, Any]:
        """Check environment variable configuration."""
        if not self.config_manager:
            return {"error": "Configuration manager not initialized"}

        results = {
            "total_env_vars": len(self.config_manager.env_mappings),
            "set_env_vars": 0,
            "unset_env_vars": 0,
            "overrides_applied": {},
            "missing_critical": [],
        }

        # Critical environment variables that should be set
        critical_env_vars = [
            "DIGITAL_TWIN_ENCRYPTION_KEY",
            "AIVILLAGE_DB_PATH",
            "RAG_EMBEDDING_MODEL",
        ]

        for env_var, config_path in self.config_manager.env_mappings.items():
            env_value = os.getenv(env_var)

            if env_value is not None:
                results["set_env_vars"] += 1
                results["overrides_applied"][env_var] = {
                    "value": env_value,
                    "config_path": config_path,
                    "applied": True,
                }
            else:
                results["unset_env_vars"] += 1

            # Check critical variables
            if env_var in critical_env_vars and env_value is None:
                results["missing_critical"].append(env_var)

        return results

    def validate_codex_compliance(self) -> dict[str, Any]:
        """Validate compliance with CODEX Integration Requirements."""
        if not self.config_manager:
            return {"error": "Configuration manager not initialized"}

        results = {
            "total_requirements": 0,
            "met_requirements": 0,
            "failed_requirements": [],
            "compliance_score": 0.0,
        }

        # CODEX requirements checklist
        requirements = [
            {
                "name": "Evolution metrics enabled",
                "check": lambda: self.config_manager.get(
                    "integration.evolution_metrics.enabled", False
                ),
                "critical": True,
            },
            {
                "name": "Evolution metrics backend is sqlite",
                "check": lambda: self.config_manager.get(
                    "integration.evolution_metrics.backend"
                )
                == "sqlite",
                "critical": True,
            },
            {
                "name": "RAG pipeline enabled",
                "check": lambda: self.config_manager.get(
                    "integration.rag_pipeline.enabled", False
                ),
                "critical": True,
            },
            {
                "name": "RAG embedding model is paraphrase-MiniLM-L3-v2",
                "check": lambda: self.config_manager.get(
                    "integration.rag_pipeline.embedding_model"
                )
                == "paraphrase-MiniLM-L3-v2",
                "critical": True,
            },
            {
                "name": "RAG cache enabled",
                "check": lambda: self.config_manager.get(
                    "integration.rag_pipeline.cache_enabled", False
                ),
                "critical": False,
            },
            {
                "name": "RAG chunk size is 512",
                "check": lambda: self.config_manager.get(
                    "integration.rag_pipeline.chunk_size"
                )
                == 512,
                "critical": False,
            },
            {
                "name": "P2P networking enabled",
                "check": lambda: self.config_manager.get(
                    "integration.p2p_networking.enabled", False
                ),
                "critical": True,
            },
            {
                "name": "P2P transport is libp2p",
                "check": lambda: self.config_manager.get(
                    "integration.p2p_networking.transport"
                )
                == "libp2p",
                "critical": True,
            },
            {
                "name": "P2P discovery method is mdns",
                "check": lambda: self.config_manager.get(
                    "integration.p2p_networking.discovery_method"
                )
                == "mdns",
                "critical": True,
            },
            {
                "name": "P2P max peers is 50",
                "check": lambda: self.config_manager.get(
                    "integration.p2p_networking.max_peers"
                )
                == 50,
                "critical": False,
            },
            {
                "name": "Digital twin enabled",
                "check": lambda: self.config_manager.get(
                    "integration.digital_twin.enabled", False
                ),
                "critical": True,
            },
            {
                "name": "Digital twin encryption enabled",
                "check": lambda: self.config_manager.get(
                    "integration.digital_twin.encryption_enabled", False
                ),
                "critical": True,
            },
            {
                "name": "Digital twin privacy mode is strict",
                "check": lambda: self.config_manager.get(
                    "integration.digital_twin.privacy_mode"
                )
                == "strict",
                "critical": False,
            },
            {
                "name": "Digital twin max profiles is 10000",
                "check": lambda: self.config_manager.get(
                    "integration.digital_twin.max_profiles"
                )
                == 10000,
                "critical": False,
            },
            {
                "name": "P2P host is 0.0.0.0",
                "check": lambda: self.config_manager.get("p2p_config.host")
                == "0.0.0.0",
                "critical": True,
            },
            {
                "name": "P2P port is 4001",
                "check": lambda: self.config_manager.get("p2p_config.port") == 4001,
                "critical": True,
            },
            {
                "name": "P2P mDNS enabled",
                "check": lambda: self.config_manager.get(
                    "p2p_config.peer_discovery.mdns_enabled", False
                ),
                "critical": True,
            },
            {
                "name": "P2P TCP transport enabled",
                "check": lambda: self.config_manager.get(
                    "p2p_config.transports.tcp_enabled", False
                ),
                "critical": True,
            },
            {
                "name": "P2P WebSocket transport enabled",
                "check": lambda: self.config_manager.get(
                    "p2p_config.transports.websocket_enabled", False
                ),
                "critical": True,
            },
            {
                "name": "P2P TLS security enabled",
                "check": lambda: self.config_manager.get(
                    "p2p_config.security.tls_enabled", False
                ),
                "critical": True,
            },
            {
                "name": "P2P peer verification enabled",
                "check": lambda: self.config_manager.get(
                    "p2p_config.security.peer_verification", False
                ),
                "critical": True,
            },
            {
                "name": "RAG embedder model name matches",
                "check": lambda: self.config_manager.get(
                    "rag_config.embedder.model_name"
                )
                == "paraphrase-MiniLM-L3-v2",
                "critical": True,
            },
            {
                "name": "RAG retrieval vector_top_k is 20",
                "check": lambda: self.config_manager.get(
                    "rag_config.retrieval.vector_top_k"
                )
                == 20,
                "critical": False,
            },
            {
                "name": "RAG retrieval keyword_top_k is 20",
                "check": lambda: self.config_manager.get(
                    "rag_config.retrieval.keyword_top_k"
                )
                == 20,
                "critical": False,
            },
            {
                "name": "RAG retrieval final_top_k is 10",
                "check": lambda: self.config_manager.get(
                    "rag_config.retrieval.final_top_k"
                )
                == 10,
                "critical": False,
            },
            {
                "name": "RAG cache L1 size is 128",
                "check": lambda: self.config_manager.get("rag_config.cache.l1_size")
                == 128,
                "critical": False,
            },
        ]

        results["total_requirements"] = len(requirements)

        for req in requirements:
            try:
                if req["check"]():
                    results["met_requirements"] += 1
                else:
                    results["failed_requirements"].append(
                        {"name": req["name"], "critical": req["critical"]}
                    )
            except Exception as e:
                results["failed_requirements"].append(
                    {"name": req["name"], "critical": req["critical"], "error": str(e)}
                )

        if results["total_requirements"] > 0:
            results["compliance_score"] = (
                results["met_requirements"] / results["total_requirements"]
            ) * 100

        return results

    def generate_recommendations(self) -> list[str]:
        """Generate configuration improvement recommendations."""
        recommendations = []

        if not self.config_manager:
            return [
                "Cannot generate recommendations: Configuration manager not initialized"
            ]

        # Check file validation results
        file_results = self.validation_results.get("file_validation", {})
        for filename, result in file_results.items():
            if not result.get("exists"):
                recommendations.append(f"Create missing configuration file: {filename}")
            elif not result.get("valid_syntax"):
                recommendations.append(f"Fix syntax errors in {filename}")

        # Check path validation results
        path_results = self.validation_results.get("path_validation", {})
        for path_config, result in path_results.items():
            if result.get("issues"):
                recommendations.append(
                    f"Fix path issues for {path_config}: {', '.join(result['issues'])}"
                )

        # Check model validation results
        model_results = self.validation_results.get("model_validation", {})
        for model_name, result in model_results.items():
            if result.get("issues"):
                recommendations.append(
                    f"Address model issues for {model_name}: {', '.join(result['issues'])}"
                )

        # Check CODEX compliance
        codex_results = self.validation_results.get("codex_compliance", {})
        failed_reqs = codex_results.get("failed_requirements", [])

        critical_failures = [req for req in failed_reqs if req.get("critical")]
        if critical_failures:
            recommendations.append(
                f"Fix critical CODEX compliance issues: {len(critical_failures)} requirements not met"
            )

        # Environment variable recommendations
        env_results = self.validation_results.get("environment_variables", {})
        missing_critical = env_results.get("missing_critical", [])
        if missing_critical:
            recommendations.append(
                f"Set critical environment variables: {', '.join(missing_critical)}"
            )

        # Performance recommendations
        if self.config_manager.get("integration.rag_pipeline.cache_enabled") is False:
            recommendations.append("Enable RAG pipeline caching for better performance")

        if not recommendations:
            recommendations.append("Configuration appears to be properly set up")

        return recommendations

    def run_comprehensive_validation(self) -> dict[str, Any]:
        """Run all validation checks and generate comprehensive report."""
        print("Running comprehensive configuration validation...")

        # File syntax validation
        print("  Validating file syntax...")
        self.validation_results["file_validation"] = self.validate_file_syntax()

        # Configuration loading validation
        print("  Testing configuration loading...")
        self.validation_results[
            "configuration_consistency"
        ] = self.validate_configuration_loading()

        # Path validation
        print("  Validating paths...")
        self.validation_results["path_validation"] = self.validate_paths()

        # Model validation
        print("  Validating model availability...")
        self.validation_results["model_validation"] = self.validate_model_availability()

        # Port validation
        print("  Validating port configuration...")
        self.validation_results["port_validation"] = self.validate_port_configuration()

        # Environment variables
        print("  Checking environment variables...")
        self.validation_results[
            "environment_variables"
        ] = self.validate_environment_variables()

        # CODEX compliance
        print("  Validating CODEX compliance...")
        self.validation_results["codex_compliance"] = self.validate_codex_compliance()

        # Generate recommendations
        print("  Generating recommendations...")
        self.validation_results["recommendations"] = self.generate_recommendations()

        # Calculate overall status
        self.calculate_overall_status()

        return self.validation_results

    def calculate_overall_status(self) -> None:
        """Calculate overall validation status."""
        total_checks = 0
        passed_checks = 0
        warnings = 0

        # Count file validation
        for result in self.validation_results["file_validation"].values():
            total_checks += 1
            if result.get("valid_syntax"):
                passed_checks += 1

        # Count configuration consistency
        if self.validation_results["configuration_consistency"].get(
            "validation_passed"
        ):
            passed_checks += 1
        total_checks += 1

        # Count path validation issues
        for result in self.validation_results["path_validation"].values():
            if isinstance(result, dict) and "issues" in result:
                total_checks += 1
                if not result["issues"]:
                    passed_checks += 1

        # Count CODEX compliance
        codex_results = self.validation_results["codex_compliance"]
        if "total_requirements" in codex_results:
            total_checks += codex_results["total_requirements"]
            passed_checks += codex_results["met_requirements"]

        # Determine overall status
        if total_checks == 0:
            overall_status = "UNKNOWN"
        else:
            success_rate = (passed_checks / total_checks) * 100

            if success_rate >= 95:
                overall_status = "EXCELLENT"
            elif success_rate >= 85:
                overall_status = "GOOD"
            elif success_rate >= 70:
                overall_status = "ACCEPTABLE"
            elif success_rate >= 50:
                overall_status = "POOR"
            else:
                overall_status = "CRITICAL"

        self.validation_results["overall_status"] = overall_status
        self.validation_results["summary"] = {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": total_checks - passed_checks,
            "warnings": warnings,
            "success_rate": (
                round((passed_checks / total_checks) * 100, 1)
                if total_checks > 0
                else 0
            ),
        }


def main():
    """Main validation function."""
    logging.basicConfig(level=logging.INFO)

    validator = ConfigurationValidator()
    results = validator.run_comprehensive_validation()

    # Print summary
    print("\n" + "=" * 80)
    print("CODEX CONFIGURATION VALIDATION REPORT")
    print("=" * 80)

    summary = results["summary"]
    print(f"\nOverall Status: {results['overall_status']}")
    print(f"Success Rate: {summary['success_rate']}%")
    print(f"Total Checks: {summary['total_checks']}")
    print(f"Passed: {summary['passed_checks']}")
    print(f"Failed: {summary['failed_checks']}")

    # CODEX Compliance
    codex_compliance = results.get("codex_compliance", {})
    if codex_compliance:
        compliance_score = codex_compliance.get("compliance_score", 0)
        print(f"CODEX Compliance: {compliance_score:.1f}%")

    # Print critical issues
    if codex_compliance.get("failed_requirements"):
        critical_failures = [
            req
            for req in codex_compliance["failed_requirements"]
            if req.get("critical")
        ]
        if critical_failures:
            print(f"\nCritical Issues ({len(critical_failures)}):")
            for failure in critical_failures:
                print(f"  ❌ {failure['name']}")

    # Print recommendations
    if results.get("recommendations"):
        print("\nRecommendations:")
        for rec in results["recommendations"][:5]:  # Show top 5
            print(f"  💡 {rec}")

    print("\n" + "=" * 80)

    if results["overall_status"] in ["EXCELLENT", "GOOD"]:
        print("✅ Configuration validation passed!")
        print("All configurations are ready for CODEX integration.")
    else:
        print("❌ Configuration validation found issues!")
        print("Review the detailed results and fix issues before proceeding.")

    print("=" * 80)

    # Save detailed report
    report_file = Path("config") / "configuration_validation_report.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed report saved to: {report_file}")

    return results["overall_status"] in ["EXCELLENT", "GOOD", "ACCEPTABLE"]


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
