"""
Cogment Configuration Validation.

Provides comprehensive validation for Cogment configurations including parameter budget,
stage consistency, and integration compatibility with Agent 1-4 components.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from config_loader import CogmentCompleteConfig, StageConfig, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    parameter_analysis: Dict[str, Any]
    stage_analysis: Dict[int, Dict[str, Any]]


@dataclass
class ParameterBudgetAnalysis:
    """Analysis of parameter budget utilization."""

    total_estimated: int
    target_budget: int
    utilization_ratio: float
    within_budget: bool
    component_breakdown: Dict[str, int]
    optimization_suggestions: List[str]


class CogmentConfigValidator:
    """
    Comprehensive configuration validator for Cogment system.

    Validates parameter budgets, stage progression, component compatibility,
    and provides optimization suggestions for staying within the 25M parameter target.
    """

    def __init__(self):
        """Initialize configuration validator."""
        self.validation_rules = self._setup_validation_rules()
        logger.info("Initialized CogmentConfigValidator")

    def _setup_validation_rules(self) -> Dict[str, Any]:
        """Setup validation rules and constraints."""
        return {
            "parameter_budget": {
                "target_params": 25_000_000,
                "tolerance": 0.05,  # 5% tolerance
                "min_params": 23_750_000,
                "max_params": 26_250_000,
            },
            "model_constraints": {
                "min_d_model": 256,
                "max_d_model": 2048,
                "min_layers": 3,
                "max_layers": 24,
                "min_vocab_size": 8000,
                "max_vocab_size": 50000,
                "max_seq_len": 4096,
            },
            "memory_constraints": {
                "min_mem_slots": 512,
                "max_mem_slots": 16384,
                "min_ltm_dim": 128,
                "max_ltm_dim": 1024,
                "max_memory_ratio": 0.6,  # Memory shouldn't exceed 60% of total params
            },
            "stage_constraints": {
                "min_stages": 3,
                "max_stages": 6,
                "min_steps_per_stage": 100,
                "max_steps_per_stage": 50000,
                "progression_requirements": {
                    "increasing_complexity": True,
                    "increasing_max_steps": True,
                    "memory_utilization_growth": True,
                },
            },
        }

    def validate_complete_config(self, config: CogmentCompleteConfig) -> ValidationResult:
        """
        Perform comprehensive validation of complete configuration.

        Args:
            config: Complete configuration to validate

        Returns:
            Validation result with errors, warnings, and analysis
        """
        logger.info("Starting comprehensive configuration validation")

        errors = []
        warnings = []

        # 1. Validate parameter budget
        param_analysis = self.validate_parameter_budget(config)
        if not param_analysis.within_budget:
            if param_analysis.total_estimated > param_analysis.target_budget * 1.05:
                errors.append(
                    f"Parameter budget exceeded: {param_analysis.total_estimated:,} > {param_analysis.target_budget:,}"
                )
            # Under budget is fine, just note it

        # 2. Validate model configuration
        model_errors, model_warnings = self._validate_model_config(config.model_config)
        errors.extend(model_errors)
        warnings.extend(model_warnings)

        # 3. Validate stage configurations
        stage_analysis = {}
        for stage_id, stage_data in config.stage_configs.items():
            stage_config = StageConfig(**stage_data)
            stage_errors, stage_warnings, analysis = self._validate_stage_config(stage_config, stage_id)
            errors.extend([f"Stage {stage_id}: {err}" for err in stage_errors])
            warnings.extend([f"Stage {stage_id}: {warn}" for warn in stage_warnings])
            stage_analysis[stage_id] = analysis

        # 4. Validate stage progression
        progression_errors, progression_warnings = self._validate_stage_progression(config.stage_configs)
        errors.extend(progression_errors)
        warnings.extend(progression_warnings)

        # 5. Validate component integration
        integration_errors, integration_warnings = self._validate_component_integration(config)
        errors.extend(integration_errors)
        warnings.extend(integration_warnings)

        # 6. Validate GrokFast configuration
        grokfast_errors, grokfast_warnings = self._validate_grokfast_config(config.grokfast_config)
        errors.extend(grokfast_errors)
        warnings.extend(grokfast_warnings)

        is_valid = len(errors) == 0

        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            parameter_analysis=param_analysis.__dict__,
            stage_analysis=stage_analysis,
        )

        logger.info(
            f"Configuration validation completed: valid={is_valid}, " f"errors={len(errors)}, warnings={len(warnings)}"
        )

        return result

    def validate_parameter_budget(self, config: CogmentCompleteConfig) -> ParameterBudgetAnalysis:
        """
        Validate parameter budget and provide detailed analysis.

        Args:
            config: Configuration to analyze

        Returns:
            Parameter budget analysis
        """
        model_config = config.model_config
        rules = self.validation_rules["parameter_budget"]

        # Extract parameters
        d_model = model_config["model"]["d_model"]
        n_layers = model_config["model"]["n_layers"]
        vocab_size = model_config["model"]["vocab_size"]
        d_ff = model_config["model"]["d_ff"]
        mem_slots = model_config["gated_ltm"]["mem_slots"]
        ltm_dim = model_config["gated_ltm"]["ltm_dim"]

        # Calculate component parameters
        component_breakdown = {}

        # 1. Embeddings (potentially tied)
        embedding_params = vocab_size * d_model
        if model_config["heads"]["tie_embeddings"]:
            component_breakdown["embeddings"] = embedding_params
            component_breakdown["output_head"] = 0
        else:
            component_breakdown["embeddings"] = embedding_params
            component_breakdown["output_head"] = vocab_size * d_model

        # 2. Transformer backbone
        attention_per_layer = 4 * d_model * d_model  # QKV + output
        ff_per_layer = 2 * d_model * d_ff
        norm_per_layer = 4 * d_model  # 2 layer norms
        backbone_per_layer = attention_per_layer + ff_per_layer + norm_per_layer
        component_breakdown["backbone"] = n_layers * backbone_per_layer

        # 3. Memory system (main parameter consumer)
        memory_storage = mem_slots * ltm_dim
        memory_gates = d_model * ltm_dim + ltm_dim * d_model  # Read and write gates
        component_breakdown["memory_storage"] = memory_storage
        component_breakdown["memory_gates"] = memory_gates

        # 4. Refinement core (estimated)
        refinement_params = d_model * d_model * 2  # Simplified estimate
        component_breakdown["refinement_core"] = refinement_params

        # 5. ACT halting
        act_params = d_model * 2  # Halting head
        component_breakdown["act_halting"] = act_params

        # Total calculation
        total_estimated = sum(component_breakdown.values())
        target_budget = rules["target_params"]
        utilization_ratio = total_estimated / target_budget
        within_budget = rules["min_params"] <= total_estimated <= rules["max_params"]

        # Generate optimization suggestions
        suggestions = []
        if not within_budget:
            if total_estimated > rules["max_params"]:
                # Over budget - suggest reductions
                memory_ratio = (
                    component_breakdown["memory_storage"] + component_breakdown["memory_gates"]
                ) / total_estimated
                if memory_ratio > 0.6:
                    suggestions.append(
                        f"Reduce mem_slots from {mem_slots} to {int(mem_slots * 0.8)} (largest component)"
                    )

                if d_model > 1024:
                    suggestions.append(f"Reduce d_model from {d_model} to {int(d_model * 0.9)}")

                if vocab_size > 32000:
                    suggestions.append(f"Reduce vocab_size from {vocab_size} to 32000")

            else:
                # Under budget - suggest increases
                remaining_budget = rules["max_params"] - total_estimated
                suggestions.append(f"Can add {remaining_budget:,} parameters")
                suggestions.append(f"Consider increasing mem_slots to {int(mem_slots * 1.1)}")

        return ParameterBudgetAnalysis(
            total_estimated=total_estimated,
            target_budget=target_budget,
            utilization_ratio=utilization_ratio,
            within_budget=within_budget,
            component_breakdown=component_breakdown,
            optimization_suggestions=suggestions,
        )

    def _validate_model_config(self, model_config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate model configuration."""
        errors = []
        warnings = []
        rules = self.validation_rules["model_constraints"]

        # Validate basic model parameters
        d_model = model_config["model"]["d_model"]
        n_layers = model_config["model"]["n_layers"]
        vocab_size = model_config["model"]["vocab_size"]
        n_head = model_config["model"]["n_head"]

        # Check ranges
        if not (rules["min_d_model"] <= d_model <= rules["max_d_model"]):
            errors.append(f"d_model {d_model} outside valid range [{rules['min_d_model']}, {rules['max_d_model']}]")

        if not (rules["min_layers"] <= n_layers <= rules["max_layers"]):
            errors.append(f"n_layers {n_layers} outside valid range [{rules['min_layers']}, {rules['max_layers']}]")

        if not (rules["min_vocab_size"] <= vocab_size <= rules["max_vocab_size"]):
            errors.append(
                f"vocab_size {vocab_size} outside valid range [{rules['min_vocab_size']}, {rules['max_vocab_size']}]"
            )

        # Check divisibility
        if d_model % n_head != 0:
            errors.append(f"d_model ({d_model}) must be divisible by n_head ({n_head})")

        # Check memory constraints
        mem_rules = self.validation_rules["memory_constraints"]
        mem_slots = model_config["gated_ltm"]["mem_slots"]
        ltm_dim = model_config["gated_ltm"]["ltm_dim"]

        if not (mem_rules["min_mem_slots"] <= mem_slots <= mem_rules["max_mem_slots"]):
            errors.append(
                f"mem_slots {mem_slots} outside valid range [{mem_rules['min_mem_slots']}, {mem_rules['max_mem_slots']}]"
            )

        if not (mem_rules["min_ltm_dim"] <= ltm_dim <= mem_rules["max_ltm_dim"]):
            errors.append(
                f"ltm_dim {ltm_dim} outside valid range [{mem_rules['min_ltm_dim']}, {mem_rules['max_ltm_dim']}]"
            )

        # Warnings for suboptimal configurations
        if d_model < 512:
            warnings.append("d_model < 512 may limit model capacity")

        if mem_slots < 1024:
            warnings.append("mem_slots < 1024 may limit memory capacity")

        return errors, warnings

    def _validate_stage_config(
        self, stage_config: StageConfig, stage_id: int
    ) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Validate individual stage configuration."""
        errors = []
        warnings = []
        analysis = {}

        rules = self.validation_rules["stage_constraints"]

        # Validate training parameters
        max_steps = stage_config.training["max_steps"]
        batch_size = stage_config.training["batch_size"]
        sequence_length = stage_config.training["sequence_length"]

        if not (rules["min_steps_per_stage"] <= max_steps <= rules["max_steps_per_stage"]):
            errors.append(f"max_steps {max_steps} outside valid range")

        if batch_size < 1 or batch_size > 128:
            errors.append(f"batch_size {batch_size} outside valid range [1, 128]")

        if sequence_length < 64 or sequence_length > 4096:
            errors.append(f"sequence_length {sequence_length} outside valid range [64, 4096]")

        # Validate model parameters
        max_refinement_steps = stage_config.model["max_refinement_steps"]
        if max_refinement_steps < 1 or max_refinement_steps > 32:
            errors.append(f"max_refinement_steps {max_refinement_steps} outside valid range [1, 32]")

        # Validate convergence criteria
        min_accuracy = stage_config.convergence["min_accuracy"]
        if not (0.0 <= min_accuracy <= 1.0):
            errors.append(f"min_accuracy {min_accuracy} must be in [0.0, 1.0]")

        # Stage-specific validations
        if stage_id == 0:  # Sanity stage
            if max_steps > 1000:
                warnings.append("Sanity stage should be quick (<1000 steps)")
            if min_accuracy < 0.8:
                warnings.append("Sanity stage should have high accuracy target")

        elif stage_id in [1, 2]:  # ARC and Algorithmic stages
            if "augmentation" not in stage_config.data:
                warnings.append("Augmentation recommended for visual/algorithmic stages")

        elif stage_id in [3, 4]:  # Math and long-context stages
            if batch_size > 8:
                warnings.append("Large batch sizes may cause memory issues in complex stages")

        # Analysis
        analysis = {
            "complexity_score": self._calculate_stage_complexity(stage_config),
            "memory_requirements": batch_size * sequence_length,
            "computational_cost": max_steps * batch_size * sequence_length * max_refinement_steps,
        }

        return errors, warnings, analysis

    def _validate_stage_progression(self, stage_configs: Dict[int, Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """Validate progression across stages."""
        errors = []
        warnings = []

        # Check stage progression rules
        prev_max_steps = 0
        prev_complexity = 0

        for stage_id in sorted(stage_configs.keys()):
            stage_data = stage_configs[stage_id]

            max_steps = stage_data["training"]["max_steps"]
            sequence_length = stage_data["training"]["sequence_length"]
            max_refinement_steps = stage_data["model"]["max_refinement_steps"]

            # Calculate complexity score
            complexity = sequence_length * max_refinement_steps

            # Check progression
            if stage_id > 0:
                if max_steps < prev_max_steps:
                    warnings.append(f"Stage {stage_id} has fewer max_steps than stage {stage_id-1}")

                if complexity < prev_complexity:
                    warnings.append(f"Stage {stage_id} has lower complexity than stage {stage_id-1}")

            prev_max_steps = max_steps
            prev_complexity = complexity

        return errors, warnings

    def _validate_component_integration(self, config: CogmentCompleteConfig) -> Tuple[List[str], List[str]]:
        """Validate integration between different components."""
        errors = []
        warnings = []

        # Check Agent 1-4 compatibility
        model_config = config.model_config

        # Agent 1 compatibility (CogmentConfig structure)
        required_agent1_fields = ["d_model", "n_head", "n_layers", "vocab_size", "max_seq_len"]
        for field in required_agent1_fields:
            if field not in model_config["model"]:
                errors.append(f"Missing required field for Agent 1 compatibility: {field}")

        # Agent 2 compatibility (GatedLTM)
        required_agent2_fields = ["ltm_capacity", "ltm_dim", "mem_slots"]
        for field in required_agent2_fields:
            if field not in model_config["gated_ltm"]:
                errors.append(f"Missing required field for Agent 2 compatibility: {field}")

        # Agent 4 compatibility (TrainingConfig)
        training_config = config.training_config
        required_training_fields = ["curriculum", "optimizers", "training"]
        for field in required_training_fields:
            if field not in training_config:
                errors.append(f"Missing required field for Agent 4 compatibility: {field}")

        # Check optimizer configuration
        if "optimizers" in training_config:
            required_optimizers = ["refinement_core", "gated_ltm", "act_halting", "other_components"]
            for optimizer in required_optimizers:
                if optimizer not in training_config["optimizers"]:
                    warnings.append(f"Missing optimizer configuration for: {optimizer}")

        return errors, warnings

    def _validate_grokfast_config(self, grokfast_config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate GrokFast configuration."""
        errors = []
        warnings = []

        # Check component configurations
        required_components = ["refinement_core", "gated_ltm", "act_halting"]
        for component in required_components:
            if component not in grokfast_config:
                warnings.append(f"Missing GrokFast configuration for component: {component}")
                continue

            comp_config = grokfast_config[component]

            # Validate parameters
            if component != "act_halting":  # ACT halting should be disabled
                if "alpha" in comp_config and not (0.8 <= comp_config["alpha"] <= 0.99):
                    warnings.append(f"GrokFast alpha for {component} outside recommended range [0.8, 0.99]")

                if "lamb" in comp_config and not (0.5 <= comp_config["lamb"] <= 3.0):
                    warnings.append(f"GrokFast lamb for {component} outside recommended range [0.5, 3.0]")

            else:
                # ACT halting should be disabled
                if comp_config.get("enabled", True):
                    errors.append("GrokFast should be disabled for ACT halting component")

        return errors, warnings

    def _calculate_stage_complexity(self, stage_config: StageConfig) -> float:
        """Calculate complexity score for a stage."""
        training = stage_config.training
        model = stage_config.model

        # Factors contributing to complexity
        sequence_factor = training["sequence_length"] / 512  # Normalized to 512
        refinement_factor = model["max_refinement_steps"] / 4  # Normalized to 4
        steps_factor = training["max_steps"] / 2000  # Normalized to 2000

        complexity = sequence_factor * refinement_factor * steps_factor
        return complexity

    def generate_validation_report(
        self, validation_result: ValidationResult, output_path: Optional[Path] = None
    ) -> str:
        """
        Generate a comprehensive validation report.

        Args:
            validation_result: Validation result to report
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        report_lines = []

        # Header
        report_lines.append("=" * 60)
        report_lines.append("COGMENT CONFIGURATION VALIDATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")

        # Overall status
        status = "✅ VALID" if validation_result.is_valid else "❌ INVALID"
        report_lines.append(f"Overall Status: {status}")
        report_lines.append(f"Errors: {len(validation_result.errors)}")
        report_lines.append(f"Warnings: {len(validation_result.warnings)}")
        report_lines.append("")

        # Parameter analysis
        param_analysis = validation_result.parameter_analysis
        report_lines.append("PARAMETER BUDGET ANALYSIS")
        report_lines.append("-" * 30)
        report_lines.append(f"Target Budget: {param_analysis['target_budget']:,} parameters")
        report_lines.append(f"Estimated Total: {param_analysis['total_estimated']:,} parameters")
        report_lines.append(f"Utilization: {param_analysis['utilization_ratio']:.1%}")
        report_lines.append(f"Within Budget: {'✅' if param_analysis['within_budget'] else '❌'}")
        report_lines.append("")

        # Component breakdown
        report_lines.append("Component Breakdown:")
        for component, params in param_analysis["component_breakdown"].items():
            percentage = params / param_analysis["total_estimated"] * 100
            report_lines.append(f"  {component}: {params:,} ({percentage:.1f}%)")
        report_lines.append("")

        # Optimization suggestions
        if param_analysis["optimization_suggestions"]:
            report_lines.append("Optimization Suggestions:")
            for suggestion in param_analysis["optimization_suggestions"]:
                report_lines.append(f"  • {suggestion}")
            report_lines.append("")

        # Errors
        if validation_result.errors:
            report_lines.append("ERRORS")
            report_lines.append("-" * 30)
            for error in validation_result.errors:
                report_lines.append(f"❌ {error}")
            report_lines.append("")

        # Warnings
        if validation_result.warnings:
            report_lines.append("WARNINGS")
            report_lines.append("-" * 30)
            for warning in validation_result.warnings:
                report_lines.append(f"⚠️  {warning}")
            report_lines.append("")

        # Stage analysis
        report_lines.append("STAGE ANALYSIS")
        report_lines.append("-" * 30)
        for stage_id, analysis in validation_result.stage_analysis.items():
            report_lines.append(f"Stage {stage_id}: Complexity={analysis.get('complexity_score', 'N/A'):.2f}")

        report = "\n".join(report_lines)

        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Validation report saved to {output_path}")

        return report
