#!/usr/bin/env python3
"""
Phase Controller Base Classes

Defines the common interface and result structure for all Agent Forge phases.
Ensures consistent model passing and metrics collection across the pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class PhaseResult:
    """Result structure for Agent Forge phases."""

    # Core results
    success: bool
    model: nn.Module
    phase_name: str | None = None

    # Performance metrics
    metrics: dict[str, Any] | None = None
    duration_seconds: float = 0.0

    # Artifacts and outputs
    artifacts: dict[str, Any] | None = None

    # Configuration and metadata
    config: dict[str, Any] | None = None
    error: str | None = None

    # Timestamps
    start_time: datetime | None = None
    end_time: datetime | None = None

    def __post_init__(self):
        """Set timestamps if not provided."""
        if self.end_time is None:
            self.end_time = datetime.now()
        if self.start_time is None:
            self.start_time = self.end_time


class PhaseController(ABC):
    """
    Abstract base class for Agent Forge phase controllers.

    Defines the interface that all phases must implement to ensure
    consistent model passing and result reporting.
    """

    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def run(self, model: nn.Module) -> PhaseResult:
        """
        Execute the phase processing.

        Args:
            model: Input model from previous phase

        Returns:
            PhaseResult with processed model and metrics
        """
        pass

    def validate_input_model(self, model: nn.Module) -> bool:
        """
        Validate input model before processing.

        Args:
            model: Model to validate

        Returns:
            True if model is valid for this phase
        """
        if model is None:
            self.logger.error("Input model is None")
            return False

        if not isinstance(model, nn.Module):
            self.logger.error(f"Input model is not nn.Module, got {type(model)}")
            return False

        # Check if model has parameters
        param_count = sum(p.numel() for p in model.parameters())
        if param_count == 0:
            self.logger.warning("Input model has no parameters")

        self.logger.debug(f"Input model validated: {param_count:,} parameters")
        return True

    def create_success_result(
        self, model: nn.Module, metrics: dict[str, Any], artifacts: dict[str, Any] | None = None, duration: float = 0.0
    ) -> PhaseResult:
        """Create a successful phase result."""
        return PhaseResult(
            success=True,
            model=model,
            phase_name=self.__class__.__name__,
            metrics=metrics,
            artifacts=artifacts or {},
            config=self.config.dict() if hasattr(self.config, "dict") else None,
            duration_seconds=duration,
        )

    def create_failure_result(self, model: nn.Module, error: str, duration: float = 0.0) -> PhaseResult:
        """Create a failed phase result."""
        return PhaseResult(
            success=False,
            model=model,
            phase_name=self.__class__.__name__,
            error=error,
            metrics={"duration_seconds": duration},
            config=self.config.dict() if hasattr(self.config, "dict") else None,
            duration_seconds=duration,
        )


class ModelPassingValidator:
    """
    Validates model compatibility between phases.

    Ensures that models can be safely passed from one phase to another
    and that any required model properties are maintained.
    """

    @staticmethod
    def validate_model_transition(source_phase: str, target_phase: str, model: nn.Module) -> tuple[bool, str]:
        """
        Validate that a model can transition between phases.

        Args:
            source_phase: Name of source phase
            target_phase: Name of target phase
            model: Model to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Basic model validation
        if model is None:
            return False, "Model is None"

        if not isinstance(model, nn.Module):
            return False, f"Model is not nn.Module, got {type(model)}"

        # Phase-specific validations
        validations = {
            ("EvoMergePhase", "QuietSTaRPhase"): ModelPassingValidator._validate_evomerge_to_quietstar,
            ("QuietSTaRPhase", "BitNetCompressionPhase"): ModelPassingValidator._validate_quietstar_to_bitnet,
            ("BitNetCompressionPhase", "ForgeTrainingPhase"): ModelPassingValidator._validate_bitnet_to_training,
            ("ForgeTrainingPhase", "ToolPersonaBakingPhase"): ModelPassingValidator._validate_training_to_toolbaking,
            ("ToolPersonaBakingPhase", "ADASPhase"): ModelPassingValidator._validate_toolbaking_to_adas,
            ("ADASPhase", "FinalCompressionPhase"): ModelPassingValidator._validate_adas_to_final,
        }

        transition_key = (source_phase, target_phase)
        if transition_key in validations:
            return validations[transition_key](model)

        # Default validation for unknown transitions
        return ModelPassingValidator._default_validation(model)

    @staticmethod
    def _default_validation(model: nn.Module) -> tuple[bool, str]:
        """Default model validation."""
        try:
            # Check basic model properties
            param_count = sum(p.numel() for p in model.parameters())
            if param_count == 0:
                return False, "Model has no parameters"

            # Check model is not in a broken state
            if hasattr(model, "training"):
                # Try to set eval mode to ensure model is functional
                original_mode = model.training
                model.eval()
                if original_mode:
                    model.train()

            return True, ""

        except Exception as e:
            return False, f"Model validation failed: {str(e)}"

    @staticmethod
    def _validate_evomerge_to_quietstar(model: nn.Module) -> tuple[bool, str]:
        """Validate EvoMerge -> Quiet-STaR transition."""
        # EvoMerge output should be a merged model ready for reasoning enhancement
        if not hasattr(model, "config"):
            return False, "Model missing config attribute from EvoMerge"
        return ModelPassingValidator._default_validation(model)

    @staticmethod
    def _validate_quietstar_to_bitnet(model: nn.Module) -> tuple[bool, str]:
        """Validate Quiet-STaR -> BitNet transition."""
        # Quiet-STaR should have added reasoning capabilities
        # BitNet needs a model ready for quantization
        return ModelPassingValidator._default_validation(model)

    @staticmethod
    def _validate_bitnet_to_training(model: nn.Module) -> tuple[bool, str]:
        """Validate BitNet -> Training transition."""
        # BitNet should output a quantized model ready for training
        return ModelPassingValidator._default_validation(model)

    @staticmethod
    def _validate_training_to_toolbaking(model: nn.Module) -> tuple[bool, str]:
        """Validate Training -> Tool Baking transition."""
        # Training should output a well-trained model ready for specialization
        return ModelPassingValidator._default_validation(model)

    @staticmethod
    def _validate_toolbaking_to_adas(model: nn.Module) -> tuple[bool, str]:
        """Validate Tool Baking -> ADAS transition."""
        # Tool baking should output a model with baked capabilities
        return ModelPassingValidator._default_validation(model)

    @staticmethod
    def _validate_adas_to_final(model: nn.Module) -> tuple[bool, str]:
        """Validate ADAS -> Final Compression transition."""
        # ADAS should output an architecturally optimized model
        return ModelPassingValidator._default_validation(model)


class PhaseOrchestrator:
    """
    Orchestrates the execution of multiple phases with model passing validation.

    Handles the sequential execution of phases, validates model transitions,
    and provides comprehensive logging and error handling.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validator = ModelPassingValidator()

    async def run_phase_sequence(
        self, phases: list[tuple[str, PhaseController]], initial_model: nn.Module
    ) -> list[PhaseResult]:
        """
        Run a sequence of phases with model passing validation.

        Args:
            phases: List of (phase_name, controller) tuples
            initial_model: Initial model to start the pipeline

        Returns:
            List of PhaseResult objects from each phase
        """
        results = []
        current_model = initial_model

        for i, (phase_name, controller) in enumerate(phases):
            self.logger.info(f"Starting phase {i + 1}/{len(phases)}: {phase_name}")

            # Validate input model
            if not controller.validate_input_model(current_model):
                error_msg = f"Phase {phase_name} input validation failed"
                self.logger.error(error_msg)
                result = controller.create_failure_result(current_model, error_msg)
                results.append(result)
                break

            # Validate model transition if not first phase
            if i > 0:
                prev_phase_name = phases[i - 1][0]
                is_valid, error_msg = self.validator.validate_model_transition(
                    prev_phase_name, phase_name, current_model
                )

                if not is_valid:
                    self.logger.error(f"Model transition validation failed: {error_msg}")
                    result = controller.create_failure_result(current_model, error_msg)
                    results.append(result)
                    break

            # Run the phase
            try:
                result = await controller.run(current_model)
                results.append(result)

                if result.success:
                    current_model = result.model
                    self.logger.info(f"Phase {phase_name} completed successfully in {result.duration_seconds:.2f}s")
                else:
                    self.logger.error(f"Phase {phase_name} failed: {result.error}")
                    break

            except Exception as e:
                error_msg = f"Phase {phase_name} raised exception: {str(e)}"
                self.logger.exception(error_msg)
                result = controller.create_failure_result(current_model, error_msg)
                results.append(result)
                break

        return results

    def validate_phase_compatibility(self, phases: list[tuple[str, PhaseController]]) -> bool:
        """
        Validate that a sequence of phases can work together.

        Args:
            phases: List of (phase_name, controller) tuples

        Returns:
            True if phases are compatible
        """
        if not phases:
            return True

        # Check for required phase order
        expected_order = [
            "EvoMergePhase",
            "QuietSTaRPhase",
            "BitNetCompressionPhase",
            "ForgeTrainingPhase",
            "ToolPersonaBakingPhase",
            "ADASPhase",
            "FinalCompressionPhase",
        ]

        phase_names = [name for name, _ in phases]

        # Check if phases follow expected order (allowing for skipped phases)
        last_expected_index = -1
        for phase_name in phase_names:
            if phase_name in expected_order:
                current_index = expected_order.index(phase_name)
                if current_index < last_expected_index:
                    self.logger.error(f"Phase {phase_name} is out of order")
                    return False
                last_expected_index = current_index

        return True
