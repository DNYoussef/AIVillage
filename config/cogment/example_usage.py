#!/usr/bin/env python3
"""
Cogment Configuration System Example Usage.

Demonstrates how to load, validate, and use the unified Cogment configuration system.
This example shows integration with Agent 1-4 components and parameter budget validation.
"""

import logging
from pathlib import Path
import sys

# Add the config directory to path for imports
config_dir = Path(__file__).parent
sys.path.insert(0, str(config_dir))

from config_loader import CogmentConfigLoader
from config_validation import CogmentConfigValidator

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Demonstrate Cogment configuration system usage."""

    print("Cogment Configuration System Demo")
    print("=" * 50)

    try:
        # 1. Initialize configuration loader
        print("\n1. Initializing Configuration Loader...")
        loader = CogmentConfigLoader()

        # 2. Load complete configuration
        print("\n2. Loading Complete Configuration...")
        complete_config = loader.load_complete_config()

        print(f"Loaded configuration from: {complete_config.loaded_from}")
        print("   - Model config loaded")
        print("   - Training config loaded")
        print("   - GrokFast config loaded")
        print("   - Deployment config loaded")
        print(f"   - {len(complete_config.stage_configs)} stage configs loaded")

        # 3. Validate configuration
        print("\n3. Validating Configuration...")
        validator = CogmentConfigValidator()
        validation_result = validator.validate_complete_config(complete_config)

        if validation_result.is_valid:
            print("Configuration validation PASSED")
        else:
            print("Configuration validation FAILED")
            print(f"   Errors: {len(validation_result.errors)}")
            for error in validation_result.errors[:3]:  # Show first 3 errors
                print(f"   - {error}")

        print(f"   Warnings: {len(validation_result.warnings)}")
        for warning in validation_result.warnings[:3]:  # Show first 3 warnings
            print(f"   - {warning}")

        # 4. Parameter budget analysis
        print("\n4. Parameter Budget Analysis...")
        param_analysis = validation_result.parameter_analysis

        print(f"   Target Budget: {param_analysis['target_budget']:,} parameters")
        print(f"   Estimated Total: {param_analysis['total_estimated']:,} parameters")
        print(f"   Utilization: {param_analysis['utilization_ratio']:.1%}")
        print(f"   Within Budget: {'YES' if param_analysis['within_budget'] else 'NO'}")

        print("\n   Component Breakdown:")
        for component, params in param_analysis["component_breakdown"].items():
            percentage = params / param_analysis["total_estimated"] * 100
            print(f"     {component}: {params:,} ({percentage:.1f}%)")

        # 5. Stage configuration examples
        print("\n5. Stage Configuration Examples...")
        loader.get_stage_names()

        for stage_id in [0, 1, 4]:  # Show sanity, ARC, and long-context stages
            stage_config = loader.load_stage_config(stage_id)
            print(f"\n   Stage {stage_id}: {stage_config.name}")
            print(f"     Description: {stage_config.description}")
            print(f"     Max Steps: {stage_config.training['max_steps']:,}")
            print(f"     Batch Size: {stage_config.training['batch_size']}")
            print(f"     Sequence Length: {stage_config.training['sequence_length']}")
            print(f"     Max Refinement Steps: {stage_config.model['max_refinement_steps']}")

            # Show GrokFast settings
            grokfast = stage_config.grokfast
            if grokfast.get("enabled", False):
                core_settings = grokfast.get("components", {}).get("refinement_core", {})
                if core_settings:
                    print(
                        f"     GrokFast: alpha={core_settings.get('alpha', 'N/A')}, lamb={core_settings.get('lamb', 'N/A')}"
                    )
            else:
                print("     GrokFast: Disabled")

        # 6. Agent 4 compatibility example
        print("\n6. Agent 4 Compatibility Example...")
        training_config = loader.load_training_config()

        print("   Multi-Optimizer Setup:")
        for optimizer_name, config in training_config.optimizers.items():
            print(f"     {optimizer_name}: lr={config['lr']}, weight_decay={config['weight_decay']}")

        print(f"   Curriculum Enabled: {training_config.curriculum['enabled']}")
        print(f"   Stages Available: {len([k for k in training_config.curriculum.keys() if k.startswith('stage_')])}")

        # 7. Configuration override example
        print("\n7. Configuration Override Example...")
        override_args = {"model": {"model": {"d_model": 480}}}  # Reduce from 512 to 480 (less drastic)

        overridden_config = loader.override_with_args(complete_config, override_args)
        new_param_analysis = validator.validate_parameter_budget(overridden_config)

        print("   Original d_model: 512")
        print("   Override d_model: 480")
        print(f"   New parameter count: {new_param_analysis.total_estimated:,}")
        print(f"   Parameter change: {new_param_analysis.total_estimated - param_analysis['total_estimated']:,}")

        # 8. Export configuration summary
        print("\n8. Exporting Configuration Summary...")
        summary = loader.export_config_summary(complete_config)

        print("   Summary includes:")
        print(f"     - Model summary: {len(summary['model_summary'])} key parameters")
        print(f"     - Training summary: {summary['training_summary']['curriculum_stages']} stages")
        print(f"     - Stage summary: {len(summary['stage_summary'])} detailed stage configs")

        # 9. Generate validation report
        print("\n9. Generating Validation Report...")
        report = validator.generate_validation_report(validation_result)

        # Save report to file
        report_path = config_dir / "validation_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"   Validation report saved to: {report_path}")
        print(f"   Report length: {len(report.split())} words")

        # 10. Integration readiness check
        print("\n10. Integration Readiness Check...")

        integration_checks = [
            ("Agent 1 (CogmentConfig)", "d_model" in complete_config.model_config["model"]),
            ("Agent 2 (GatedLTM)", "ltm_capacity" in complete_config.model_config["gated_ltm"]),
            ("Agent 3 (Heads)", "tie_embeddings" in complete_config.model_config["heads"]),
            ("Agent 4 (Training)", "curriculum" in complete_config.training_config),
            ("Parameter Budget", param_analysis["within_budget"]),
            ("Stage Progression", len(complete_config.stage_configs) == 5),
            ("GrokFast Config", "refinement_core" in complete_config.grokfast_config),
        ]

        all_passed = True
        for check_name, passed in integration_checks:
            status = "PASS" if passed else "FAIL"  # nosec B105 - status indicator, not password
            print(f"     {check_name}: {status}")
            if not passed:
                all_passed = False

        print(f"\n   Overall Integration Readiness: {'READY' if all_passed else 'NOT READY'}")

        # 11. Option A configuration confirmation
        print("\n11. Option A Configuration Confirmation...")
        model_config = complete_config.model_config

        option_a_params = {"d_model": 512, "d_kv": 64, "mem_slots": 2048, "ltm_capacity": 1024, "ltm_dim": 256}

        actual_params = {
            "d_model": model_config["model"]["d_model"],
            "d_kv": model_config["model"]["d_kv"],
            "mem_slots": model_config["gated_ltm"]["mem_slots"],
            "ltm_capacity": model_config["gated_ltm"]["ltm_capacity"],
            "ltm_dim": model_config["gated_ltm"]["ltm_dim"],
        }

        print("   Option A Parameter Verification:")
        for param, expected in option_a_params.items():
            actual = actual_params[param]
            status = "PASS" if actual == expected else "FAIL"  # nosec B105 - status indicator, not password
            print(f"     {param}: expected={expected}, actual={actual} {status}")

        # Final summary
        print("\nConfiguration System Demo Complete!")
        print(f"   Total Parameters: {param_analysis['total_estimated']:,}")
        print(f"   Budget Utilization: {param_analysis['utilization_ratio']:.1%}")
        print(f"   Configuration Valid: {'YES' if validation_result.is_valid else 'NO'}")
        print(f"   Agent Integration: {'YES' if all_passed else 'NO'}")

        return validation_result.is_valid and all_passed

    except Exception as e:
        logger.error(f"Configuration demo failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
