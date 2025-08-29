#!/usr/bin/env python3
"""
Cognate Configuration Validation Script
Comprehensive validation suite for Cognate 25M configuration system
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

# Add config directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "config" / "cognate" / "25m"))

try:
    from config_loader import (
        CognateConfigLoader,
        load_cognate_config,
        validate_cognate_config,
    )
except ImportError as e:
    print(f"Error importing configuration loader: {e}")
    print("Please ensure you're running from the correct directory")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CognateConfigValidator:
    """Comprehensive validation suite for Cognate 25M configuration."""

    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self.validation_results: dict[str, Any] = {}

    def run_full_validation(self, environments: list[str] = None) -> dict[str, Any]:
        """Run complete validation suite across all environments."""
        if environments is None:
            environments = ["development", "staging", "production", "testing"]

        logger.info("Starting comprehensive Cognate configuration validation")

        overall_results = {
            "validation_summary": {"total_environments": len(environments), "passed": 0, "failed": 0, "warnings": 0},
            "environment_results": {},
            "cross_environment_analysis": {},
            "recommendations": [],
        }

        # Validate each environment
        for env in environments:
            logger.info(f"Validating {env} environment...")
            try:
                env_results = self.validate_environment(env)
                overall_results["environment_results"][env] = env_results

                # Update summary
                if env_results["validation_status"] == "passed":
                    overall_results["validation_summary"]["passed"] += 1
                elif env_results["validation_status"] == "failed":
                    overall_results["validation_summary"]["failed"] += 1
                else:
                    overall_results["validation_summary"]["warnings"] += 1

            except Exception as e:
                logger.error(f"Error validating {env} environment: {e}")
                overall_results["environment_results"][env] = {"validation_status": "error", "error": str(e)}
                overall_results["validation_summary"]["failed"] += 1

        # Cross-environment analysis
        overall_results["cross_environment_analysis"] = self.analyze_cross_environment_consistency(
            overall_results["environment_results"]
        )

        # Generate recommendations
        overall_results["recommendations"] = self.generate_recommendations(overall_results)

        # Print summary
        self.print_validation_summary(overall_results)

        return overall_results

    def validate_environment(self, environment: str) -> dict[str, Any]:
        """Validate configuration for a specific environment."""
        results = {
            "environment": environment,
            "validation_status": "passed",
            "config_loading": {},
            "parameter_validation": {},
            "functional_validation": {},
            "integration_validation": {},
            "performance_validation": {},
            "errors": [],
            "warnings": [],
            "info": [],
        }

        try:
            # Test configuration loading
            logger.info(f"Testing configuration loading for {environment}...")
            results["config_loading"] = self.test_config_loading(environment)

            # Test parameter validation
            logger.info(f"Testing parameter validation for {environment}...")
            results["parameter_validation"] = self.test_parameter_validation(environment)

            # Test functional validation
            logger.info(f"Testing functional validation for {environment}...")
            results["functional_validation"] = self.test_functional_validation(environment)

            # Test integration
            logger.info(f"Testing integration for {environment}...")
            results["integration_validation"] = self.test_integration_validation(environment)

            # Test performance characteristics
            logger.info(f"Testing performance characteristics for {environment}...")
            results["performance_validation"] = self.test_performance_validation(environment)

            # Determine overall status
            has_errors = any(
                results[section].get("errors", [])
                for section in [
                    "config_loading",
                    "parameter_validation",
                    "functional_validation",
                    "integration_validation",
                ]
            )

            has_warnings = any(
                results[section].get("warnings", [])
                for section in [
                    "config_loading",
                    "parameter_validation",
                    "functional_validation",
                    "integration_validation",
                ]
            )

            if has_errors:
                results["validation_status"] = "failed"
            elif has_warnings:
                results["validation_status"] = "warnings"
            else:
                results["validation_status"] = "passed"

        except Exception as e:
            results["validation_status"] = "error"
            results["errors"].append(f"Environment validation error: {e}")

        return results

    def test_config_loading(self, environment: str) -> dict[str, Any]:
        """Test configuration loading for environment."""
        results = {"status": "passed", "tests": {}, "errors": [], "warnings": []}

        try:
            # Test loading different config types
            config_types = ["complete", "model", "training", "deployment", "validation"]

            for config_type in config_types:
                try:
                    config = load_cognate_config(environment, config_type)
                    results["tests"][f"load_{config_type}"] = {
                        "status": "passed",
                        "config_size": len(str(config)),
                        "has_required_sections": self._check_required_sections(config, config_type),
                    }
                except Exception as e:
                    results["tests"][f"load_{config_type}"] = {"status": "failed", "error": str(e)}
                    results["errors"].append(f"Failed to load {config_type} config: {e}")

        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(f"Config loading test failed: {e}")

        return results

    def test_parameter_validation(self, environment: str) -> dict[str, Any]:
        """Test parameter count validation."""
        results = {"status": "passed", "tests": {}, "errors": [], "warnings": []}

        try:
            # Load configuration
            config = load_cognate_config(environment, "complete")

            # Create loader for parameter estimation
            loader = CognateConfigLoader(self.config_dir)
            loader.environment = environment

            # Estimate parameter count
            param_count = loader.estimate_parameter_count(config)
            results["tests"]["parameter_estimation"] = {"status": "passed", "estimated_count": param_count}

            # Validate against targets
            env_targets = {
                "development": (15_000_000, 5_000_000),  # 15M ±5M
                "staging": (25_000_000, 2_000_000),  # 25M ±2M
                "production": (25_000_000, 1_000_000),  # 25M ±1M
                "testing": (5_000_000, 2_000_000),  # 5M ±2M
            }

            if environment in env_targets:
                target, tolerance = env_targets[environment]
                diff = abs(param_count - target)

                if diff <= tolerance:
                    results["tests"]["parameter_target"] = {
                        "status": "passed",
                        "target": target,
                        "actual": param_count,
                        "difference": diff,
                        "within_tolerance": True,
                    }
                else:
                    results["tests"]["parameter_target"] = {
                        "status": "failed" if environment == "production" else "warning",
                        "target": target,
                        "actual": param_count,
                        "difference": diff,
                        "within_tolerance": False,
                    }

                    message = f"Parameter count {param_count:,} outside target {target:,} ±{tolerance:,}"
                    if environment == "production":
                        results["errors"].append(message)
                    else:
                        results["warnings"].append(message)

        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(f"Parameter validation failed: {e}")

        return results

    def test_functional_validation(self, environment: str) -> dict[str, Any]:
        """Test functional validation (model creation and basic operations)."""
        results = {"status": "passed", "tests": {}, "errors": [], "warnings": []}

        try:
            # Test configuration validation
            validation_report = validate_cognate_config(environment=environment)
            results["tests"]["config_validation"] = {
                "status": validation_report["status"],
                "errors": validation_report["errors"],
                "warnings": validation_report["warnings"],
                "checks_passed": len(validation_report["checks"]),
            }

            if validation_report["status"] == "failed":
                results["errors"].extend(validation_report["errors"])
            if validation_report["warnings"]:
                results["warnings"].extend(validation_report["warnings"])

            # Test model creation (only for non-production to avoid resource usage)
            if environment != "production":
                try:
                    # This is a basic test - just try to create the model
                    # We won't actually load it to avoid memory issues
                    config = load_cognate_config(environment, "complete")
                    loader = CognateConfigLoader(self.config_dir)
                    loader.environment = environment
                    cognate_config = loader.create_cognate_config_object(config)

                    results["tests"]["cognate_config_creation"] = {
                        "status": "passed",
                        "config_type": type(cognate_config).__name__,
                    }

                except Exception as e:
                    results["tests"]["cognate_config_creation"] = {"status": "failed", "error": str(e)}
                    results["errors"].append(f"CognateConfig creation failed: {e}")

        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(f"Functional validation failed: {e}")

        return results

    def test_integration_validation(self, environment: str) -> dict[str, Any]:
        """Test integration with Agent Forge pipeline."""
        results = {"status": "passed", "tests": {}, "errors": [], "warnings": []}

        try:
            # Test pipeline integration config loading
            try:
                pipeline_config_path = (
                    self.config_dir.parent.parent / "agent_forge" / "cognate" / "pipeline_integration.yaml"
                )
                if pipeline_config_path.exists():
                    from config_loader import CognateConfigLoader

                    loader = CognateConfigLoader(self.config_dir)
                    pipeline_config = loader._load_yaml_file(pipeline_config_path)

                    results["tests"]["pipeline_config_loading"] = {
                        "status": "passed",
                        "config_sections": list(pipeline_config.keys()),
                    }

                    # Check required pipeline sections
                    required_sections = ["pipeline", "cognate_phase", "integration", "monitoring"]
                    missing_sections = [s for s in required_sections if s not in pipeline_config]

                    if missing_sections:
                        results["warnings"].append(f"Missing pipeline sections: {missing_sections}")
                else:
                    results["warnings"].append("Pipeline integration config not found")

            except Exception as e:
                results["errors"].append(f"Pipeline config loading failed: {e}")

        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(f"Integration validation failed: {e}")

        return results

    def test_performance_validation(self, environment: str) -> dict[str, Any]:
        """Test performance characteristics validation."""
        results = {"status": "passed", "tests": {}, "errors": [], "warnings": []}

        try:
            # Load config and estimate resource requirements
            config = load_cognate_config(environment, "complete")

            # Estimate memory requirements
            param_count = CognateConfigLoader(self.config_dir).estimate_parameter_count(config)

            # Estimate memory usage (rough)
            # Parameters (4 bytes each) + gradients (4 bytes) + optimizer states (8 bytes) = 16 bytes per param
            estimated_training_memory_gb = (param_count * 16) / (1024**3)
            estimated_inference_memory_gb = (param_count * 4) / (1024**3)

            results["tests"]["memory_estimation"] = {
                "status": "passed",
                "estimated_training_memory_gb": round(estimated_training_memory_gb, 2),
                "estimated_inference_memory_gb": round(estimated_inference_memory_gb, 2),
            }

            # Check against environment expectations
            env_memory_limits = {
                "development": {"training": 4, "inference": 2},
                "staging": {"training": 8, "inference": 4},
                "production": {"training": 16, "inference": 8},
                "testing": {"training": 2, "inference": 1},
            }

            if environment in env_memory_limits:
                limits = env_memory_limits[environment]

                if estimated_training_memory_gb > limits["training"]:
                    results["warnings"].append(
                        f"Estimated training memory ({estimated_training_memory_gb:.2f}GB) exceeds "
                        f"{environment} limit ({limits['training']}GB)"
                    )

                if estimated_inference_memory_gb > limits["inference"]:
                    results["warnings"].append(
                        f"Estimated inference memory ({estimated_inference_memory_gb:.2f}GB) exceeds "
                        f"{environment} limit ({limits['inference']}GB)"
                    )

        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(f"Performance validation failed: {e}")

        return results

    def analyze_cross_environment_consistency(self, env_results: dict[str, Any]) -> dict[str, Any]:
        """Analyze consistency across environments."""
        analysis = {
            "consistency_checks": {},
            "parameter_count_comparison": {},
            "config_structure_comparison": {},
            "recommendations": [],
        }

        try:
            # Compare parameter counts across environments
            param_counts = {}
            for env, results in env_results.items():
                if "parameter_validation" in results and "tests" in results["parameter_validation"]:
                    param_test = results["parameter_validation"]["tests"].get("parameter_estimation")
                    if param_test and "estimated_count" in param_test:
                        param_counts[env] = param_test["estimated_count"]

            analysis["parameter_count_comparison"] = param_counts

            # Check parameter count consistency (development and testing should be different)
            if "staging" in param_counts and "production" in param_counts:
                if abs(param_counts["staging"] - param_counts["production"]) > 1000:
                    analysis["recommendations"].append(
                        "Staging and production parameter counts should be nearly identical"
                    )

        except Exception as e:
            analysis["error"] = str(e)

        return analysis

    def generate_recommendations(self, overall_results: dict[str, Any]) -> list[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        try:
            # Check overall success rate
            summary = overall_results["validation_summary"]
            if summary["failed"] > 0:
                recommendations.append(f"Fix configuration issues in {summary['failed']} failed environment(s)")

            if summary["warnings"] > summary["passed"]:
                recommendations.append("Address configuration warnings to improve system reliability")

            # Environment-specific recommendations
            for env, results in overall_results["environment_results"].items():
                if results["validation_status"] == "failed":
                    error_count = len(results.get("errors", []))
                    recommendations.append(
                        f"{env.capitalize()} environment has {error_count} critical issues that need fixing"
                    )

        except Exception as e:
            recommendations.append(f"Error generating recommendations: {e}")

        return recommendations

    def print_validation_summary(self, results: dict[str, Any]):
        """Print comprehensive validation summary."""
        print("\n" + "=" * 80)
        print("COGNATE 25M CONFIGURATION VALIDATION SUMMARY")
        print("=" * 80)

        # Overall summary
        summary = results["validation_summary"]
        print("\nOverall Results:")
        print(f"  Environments tested: {summary['total_environments']}")
        print(f"  Passed: {summary['passed']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Warnings: {summary['warnings']}")

        # Environment details
        print("\nEnvironment Details:")
        for env, env_results in results["environment_results"].items():
            status = env_results["validation_status"]
            status_emoji = {"passed": "✅", "warnings": "⚠️", "failed": "❌", "error": "💥"}
            print(f"  {status_emoji.get(status, '?')} {env.capitalize()}: {status}")

            # Show error/warning counts
            if "errors" in env_results and env_results["errors"]:
                print(f"    Errors: {len(env_results['errors'])}")
            if "warnings" in env_results and env_results["warnings"]:
                print(f"    Warnings: {len(env_results['warnings'])}")

        # Parameter count comparison
        if "cross_environment_analysis" in results:
            param_counts = results["cross_environment_analysis"].get("parameter_count_comparison", {})
            if param_counts:
                print("\nParameter Count Comparison:")
                for env, count in param_counts.items():
                    print(f"  {env.capitalize()}: {count:,} parameters")

        # Recommendations
        if results["recommendations"]:
            print("\nRecommendations:")
            for i, rec in enumerate(results["recommendations"], 1):
                print(f"  {i}. {rec}")

        print("\n" + "=" * 80)

    def _check_required_sections(self, config: dict[str, Any], config_type: str) -> dict[str, bool]:
        """Check if configuration has required sections."""
        required_sections = {
            "complete": ["model", "memory", "act", "training", "system"],
            "model": ["model"],
            "training": ["training"],
            "deployment": ["serving", "performance", "monitoring"],
            "validation": ["model_validation", "training_validation"],
        }

        required = required_sections.get(config_type, [])
        return {section: section in config for section in required}


def main():
    """Main validation script entry point."""
    parser = argparse.ArgumentParser(description="Validate Cognate 25M configuration system")
    parser.add_argument(
        "--config-dir",
        default=str(Path(__file__).parent.parent / "config" / "cognate" / "25m"),
        help="Configuration directory path",
    )
    parser.add_argument(
        "--environments",
        nargs="*",
        choices=["development", "staging", "production", "testing"],
        default=["development", "staging", "production", "testing"],
        help="Environments to validate",
    )
    parser.add_argument("--output", help="Save validation results to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", action="store_true", help="Quick validation (skip intensive tests)")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Create validator
        validator = CognateConfigValidator(args.config_dir)

        # Run validation
        results = validator.run_full_validation(args.environments)

        # Save results if requested
        if args.output:
            import json

            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Validation results saved to: {args.output}")

        # Exit with appropriate code
        if results["validation_summary"]["failed"] > 0:
            sys.exit(1)
        elif results["validation_summary"]["warnings"] > 0:
            sys.exit(2)  # Warnings
        else:
            sys.exit(0)  # Success

    except Exception as e:
        logger.error(f"Validation script failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
