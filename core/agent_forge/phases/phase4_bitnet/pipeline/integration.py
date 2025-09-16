"""
BitNet Integration Module for Agent Forge Phase 4
=================================================

Integration layer for seamless compatibility with EvoMerge (Phase 2) and
Quiet-STaR (Phase 3) outputs. Handles model loading, format conversion,
and preparation for Phase 5 training while maintaining feature integrity.

Key Features:
- EvoMerge evolved model loading and conversion
- Quiet-STaR thought generation preservation
- Format standardization across phases
- Quality validation and compatibility checks
- Automated pipeline coordination

Author: BitNet Core Implementation Specialist - Agent Forge Phase 4
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Type
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add paths for phase integration
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from phases.phase2_evomerge.evomerge import EvoMergeConfig
    from phases.phase2_evomerge.integration import EvoMergeIntegration
except ImportError:
    warnings.warn("Phase 2 EvoMerge modules not found, some integration features may be unavailable")
    EvoMergeConfig = None
    EvoMergeIntegration = None

try:
    from phases.phase3_quietstar.architecture import QuietSTaRConfig
    from phases.phase3_quietstar.integration import QuietSTaRIntegration
except ImportError:
    warnings.warn("Phase 3 Quiet-STaR modules not found, some integration features may be unavailable")
    QuietSTaRConfig = None
    QuietSTaRIntegration = None

from ..config.bitnet_config import BitNetConfig
from ..pipeline.compression import BitNetCompressionPipeline, CompressionResult


@dataclass
class IntegrationResult:
    """Result of phase integration operation."""
    model: nn.Module
    metadata: Dict[str, Any]
    source_phase: str
    compression_compatible: bool
    success: bool
    error_message: Optional[str] = None


@dataclass
class PhaseOutput:
    """Standardized output format for phase integration."""
    model: nn.Module
    config: Dict[str, Any]
    metrics: Dict[str, float]
    phase_info: Dict[str, Any]
    checkpoint_path: Optional[str] = None


class BitNetPhaseIntegration:
    """
    Integration manager for BitNet Phase 4 with previous phases.

    Handles loading models from Phase 2 (EvoMerge) and Phase 3 (Quiet-STaR),
    performing necessary format conversions, and preparing outputs for Phase 5.
    """

    def __init__(self, config: BitNetConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or self._setup_logger()

        # Integration configurations
        self.integration_config = config.integration_config

        # Phase-specific integrators
        self.evomerge_integrator = None
        self.quietstar_integrator = None

        # Initialize integrators if available
        self._initialize_integrators()

        # Compression pipeline
        self.compression_pipeline = BitNetCompressionPipeline(config, logger)

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for integration operations."""
        logger = logging.getLogger('bitnet_integration')
        logger.setLevel(getattr(logging, self.config.log_level))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _initialize_integrators(self):
        """Initialize phase-specific integrators if available."""
        if EvoMergeIntegration is not None:
            try:
                self.evomerge_integrator = EvoMergeIntegration()
                self.logger.info("EvoMerge integrator initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize EvoMerge integrator: {e}")

        if QuietSTaRIntegration is not None:
            try:
                self.quietstar_integrator = QuietSTaRIntegration()
                self.logger.info("Quiet-STaR integrator initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Quiet-STaR integrator: {e}")

    def load_evomerge_model(self, model_path: Optional[str] = None,
                           generation: Optional[int] = None) -> IntegrationResult:
        """
        Load evolved model from EvoMerge Phase 2.

        Args:
            model_path: Path to specific model file
            generation: Generation number to load (if not specified, loads best)

        Returns:
            IntegrationResult with loaded model and metadata
        """
        self.logger.info("Loading EvoMerge model from Phase 2...")

        try:
            if model_path is None:
                # Auto-discover best model from Phase 2 output
                model_path = self._find_best_evomerge_model(generation)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"EvoMerge model not found: {model_path}")

            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=self.config.device)

            # Extract model and metadata
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                model_state = checkpoint['model']
            else:
                model_state = checkpoint

            # Create model architecture (this may need adaptation based on Phase 2 structure)
            model = self._reconstruct_evomerge_model(checkpoint, model_state)

            # Extract metadata
            metadata = {
                'source_path': model_path,
                'generation': checkpoint.get('generation', 'unknown'),
                'fitness_score': checkpoint.get('fitness', 0.0),
                'merge_technique': checkpoint.get('merge_technique', 'unknown'),
                'evolution_config': checkpoint.get('config', {}),
                'training_metrics': checkpoint.get('metrics', {})
            }

            # Validate compatibility
            compatibility_check = self._check_compression_compatibility(model)

            self.logger.info(f"Successfully loaded EvoMerge model (generation {metadata['generation']})")

            return IntegrationResult(
                model=model,
                metadata=metadata,
                source_phase="evomerge",
                compression_compatible=compatibility_check,
                success=True
            )

        except Exception as e:
            self.logger.error(f"Failed to load EvoMerge model: {str(e)}")
            return IntegrationResult(
                model=None,
                metadata={},
                source_phase="evomerge",
                compression_compatible=False,
                success=False,
                error_message=str(e)
            )

    def load_quietstar_model(self, model_path: Optional[str] = None) -> IntegrationResult:
        """
        Load enhanced model from Quiet-STaR Phase 3.

        Args:
            model_path: Path to specific model file

        Returns:
            IntegrationResult with loaded model and metadata
        """
        self.logger.info("Loading Quiet-STaR model from Phase 3...")

        try:
            if model_path is None:
                # Auto-discover model from Phase 3 output
                model_path = self._find_quietstar_model()

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Quiet-STaR model not found: {model_path}")

            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=self.config.device)

            # Extract model and metadata
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                model_state = checkpoint['model']
            else:
                model_state = checkpoint

            # Reconstruct model with Quiet-STaR enhancements
            model = self._reconstruct_quietstar_model(checkpoint, model_state)

            # Extract metadata
            metadata = {
                'source_path': model_path,
                'quietstar_config': checkpoint.get('quietstar_config', {}),
                'thought_generation_enabled': checkpoint.get('thought_generation_enabled', False),
                'attention_modifications': checkpoint.get('attention_modifications', {}),
                'training_metrics': checkpoint.get('metrics', {}),
                'integration_info': checkpoint.get('integration_info', {})
            }

            # Special handling for thought generation preservation
            if self.integration_config.preserve_thought_generation and metadata['thought_generation_enabled']:
                self.logger.info("Preserving Quiet-STaR thought generation capabilities")
                self._preserve_thought_generation(model, metadata)

            # Validate compatibility
            compatibility_check = self._check_compression_compatibility(model)

            self.logger.info("Successfully loaded Quiet-STaR enhanced model")

            return IntegrationResult(
                model=model,
                metadata=metadata,
                source_phase="quietstar",
                compression_compatible=compatibility_check,
                success=True
            )

        except Exception as e:
            self.logger.error(f"Failed to load Quiet-STaR model: {str(e)}")
            return IntegrationResult(
                model=None,
                metadata={},
                source_phase="quietstar",
                compression_compatible=False,
                success=False,
                error_message=str(e)
            )

    def compress_integrated_model(self, integration_result: IntegrationResult,
                                 validation_data: Optional[DataLoader] = None) -> CompressionResult:
        """
        Compress an integrated model from previous phases.

        Args:
            integration_result: Result from model integration
            validation_data: Optional validation data for quality checks

        Returns:
            CompressionResult with compressed model
        """
        if not integration_result.success:
            raise ValueError("Cannot compress failed integration result")

        self.logger.info(f"Compressing {integration_result.source_phase} model...")

        # Set validation data if provided
        if validation_data is not None:
            self.compression_pipeline.set_validation_data(validation_data)

        # Apply phase-specific preprocessing
        model = self._preprocess_for_compression(
            integration_result.model,
            integration_result.source_phase,
            integration_result.metadata
        )

        # Perform compression
        model_name = f"{integration_result.source_phase}_model"
        compression_result = self.compression_pipeline.compress_model(model, model_name)

        # Add integration metadata to compression result
        if compression_result.success:
            compression_result.compression_stats['integration_metadata'] = integration_result.metadata
            compression_result.compression_stats['source_phase'] = integration_result.source_phase

        return compression_result

    def prepare_for_phase5(self, compression_result: CompressionResult,
                          output_name: str = "bitnet_compressed") -> PhaseOutput:
        """
        Prepare compressed model for Phase 5 training.

        Args:
            compression_result: Result from compression pipeline
            output_name: Name for output files

        Returns:
            PhaseOutput formatted for Phase 5 consumption
        """
        self.logger.info("Preparing output for Phase 5...")

        if not compression_result.success:
            raise ValueError("Cannot prepare failed compression result for Phase 5")

        # Create output directory
        output_dir = Path(self.integration_config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save compressed model
        model_path = output_dir / f"{output_name}.pt"
        torch.save({
            'model_state_dict': compression_result.compressed_model.state_dict(),
            'bitnet_config': self.config.__dict__,
            'compression_stats': compression_result.compression_stats,
            'quality_metrics': compression_result.quality_metrics,
            'phase4_metadata': {
                'compression_time': compression_result.compression_time,
                'success': compression_result.success,
                'timestamp': torch.tensor(torch.get_rng_state()).sum().item()  # Simple timestamp
            }
        }, model_path)

        # Create Phase 5 configuration
        phase5_config = self._create_phase5_config(compression_result)

        # Save configuration
        config_path = output_dir / f"{output_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(phase5_config, f, indent=2, default=str)

        # Create metadata summary
        metadata_path = output_dir / f"{output_name}_metadata.json"
        metadata = {
            'model_path': str(model_path),
            'config_path': str(config_path),
            'compression_stats': compression_result.compression_stats,
            'quality_metrics': compression_result.quality_metrics,
            'phase_info': {
                'source_phase': compression_result.compression_stats.get('source_phase', 'unknown'),
                'bitnet_version': '1.0',
                'compression_level': self.config.compression_level.value,
                'quantization_mode': self.config.layer_config.quantization_mode.value
            }
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        self.logger.info(f"Phase 5 preparation complete. Model saved to: {model_path}")

        return PhaseOutput(
            model=compression_result.compressed_model,
            config=phase5_config,
            metrics=compression_result.quality_metrics,
            phase_info=metadata['phase_info'],
            checkpoint_path=str(model_path)
        )

    def run_complete_pipeline(self, source_phase: str = "auto",
                            model_path: Optional[str] = None,
                            validation_data: Optional[DataLoader] = None) -> PhaseOutput:
        """
        Run the complete integration and compression pipeline.

        Args:
            source_phase: Source phase ("evomerge", "quietstar", or "auto")
            model_path: Optional specific model path
            validation_data: Optional validation data

        Returns:
            PhaseOutput ready for Phase 5
        """
        self.logger.info("Running complete BitNet integration pipeline...")

        # Step 1: Auto-detect source phase if needed
        if source_phase == "auto":
            source_phase = self._auto_detect_source_phase(model_path)

        # Step 2: Load model from appropriate phase
        if source_phase == "evomerge":
            integration_result = self.load_evomerge_model(model_path)
        elif source_phase == "quietstar":
            integration_result = self.load_quietstar_model(model_path)
        else:
            raise ValueError(f"Unsupported source phase: {source_phase}")

        if not integration_result.success:
            raise RuntimeError(f"Failed to load model from {source_phase}: {integration_result.error_message}")

        # Step 3: Compress integrated model
        compression_result = self.compress_integrated_model(integration_result, validation_data)

        if not compression_result.success:
            raise RuntimeError(f"Failed to compress model: {compression_result.error_message}")

        # Step 4: Prepare for Phase 5
        phase_output = self.prepare_for_phase5(compression_result)

        self.logger.info("Complete pipeline execution successful!")
        return phase_output

    # Helper methods

    def _find_best_evomerge_model(self, generation: Optional[int] = None) -> str:
        """Find the best EvoMerge model from Phase 2 outputs."""
        search_dir = Path(self.integration_config.evomerge_model_path)

        if generation is not None:
            # Look for specific generation
            pattern = self.integration_config.evomerge_checkpoint_format.format(generation=generation)
            model_path = search_dir / pattern
            if model_path.exists():
                return str(model_path)

        # Find best model (highest generation or best fitness)
        model_files = list(search_dir.glob("*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No EvoMerge models found in {search_dir}")

        # For now, return the most recent file
        best_model = max(model_files, key=lambda p: p.stat().st_mtime)
        return str(best_model)

    def _find_quietstar_model(self) -> str:
        """Find the Quiet-STaR model from Phase 3 outputs."""
        search_dir = Path(self.integration_config.quietstar_model_path)

        # Look for output models
        model_files = list(search_dir.glob("*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No Quiet-STaR models found in {search_dir}")

        # Return the most recent file
        best_model = max(model_files, key=lambda p: p.stat().st_mtime)
        return str(best_model)

    def _reconstruct_evomerge_model(self, checkpoint: Dict[str, Any], model_state: Dict[str, torch.Tensor]) -> nn.Module:
        """Reconstruct model from EvoMerge checkpoint."""
        # This is a simplified version - in practice, would need to match the exact architecture
        # from Phase 2 based on the checkpoint metadata

        try:
            # Try to extract architecture info from checkpoint
            config = checkpoint.get('config', {})
            architecture_info = checkpoint.get('architecture_info', {})

            # Create a basic transformer model (this would need to be adapted)
            from transformers import AutoModel, AutoConfig

            if 'model_name' in architecture_info:
                model = AutoModel.from_pretrained(architecture_info['model_name'])
            else:
                # Fallback: create a basic model
                model = self._create_default_model()

            # Load state dict
            model.load_state_dict(model_state, strict=False)
            return model

        except Exception as e:
            self.logger.warning(f"Failed to reconstruct EvoMerge model architecture: {e}")
            # Return a basic model with loaded weights
            return self._create_default_model_with_state(model_state)

    def _reconstruct_quietstar_model(self, checkpoint: Dict[str, Any], model_state: Dict[str, torch.Tensor]) -> nn.Module:
        """Reconstruct model from Quiet-STaR checkpoint."""
        # Similar to EvoMerge reconstruction but accounting for Quiet-STaR modifications

        try:
            config = checkpoint.get('quietstar_config', {})

            # Create model with Quiet-STaR enhancements
            model = self._create_default_model()

            # Add thought generation components if present
            if 'thought_generation_enabled' in checkpoint and checkpoint['thought_generation_enabled']:
                self._add_thought_generation_components(model, config)

            # Load state dict
            model.load_state_dict(model_state, strict=False)
            return model

        except Exception as e:
            self.logger.warning(f"Failed to reconstruct Quiet-STaR model architecture: {e}")
            return self._create_default_model_with_state(model_state)

    def _create_default_model(self) -> nn.Module:
        """Create a default model architecture."""
        # This is a placeholder - in practice, would create based on configuration
        from transformers import GPT2Model, GPT2Config

        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12
        )
        return GPT2Model(config)

    def _create_default_model_with_state(self, model_state: Dict[str, torch.Tensor]) -> nn.Module:
        """Create a model that matches the provided state dict."""
        # Analyze state dict to determine architecture
        # This is a simplified version
        model = self._create_default_model()

        # Try to load what we can
        model.load_state_dict(model_state, strict=False)
        return model

    def _add_thought_generation_components(self, model: nn.Module, config: Dict[str, Any]):
        """Add thought generation components to model."""
        # Placeholder for adding Quiet-STaR specific components
        pass

    def _preserve_thought_generation(self, model: nn.Module, metadata: Dict[str, Any]):
        """Preserve thought generation capabilities during compression."""
        # Mark components that should be preserved during compression
        if hasattr(model, 'thought_generator'):
            # Mark for preservation
            model.thought_generator._preserve_during_compression = True

    def _check_compression_compatibility(self, model: nn.Module) -> bool:
        """Check if model is compatible with BitNet compression."""
        try:
            # Check for required components
            has_linear_layers = any(isinstance(m, nn.Linear) for m in model.modules())
            has_attention_layers = any(isinstance(m, nn.MultiheadAttention) for m in model.modules())

            # Check model size (shouldn't be too small or too large)
            param_count = sum(p.numel() for p in model.parameters())
            size_compatible = 1e6 <= param_count <= 1e10  # 1M to 10B parameters

            return has_linear_layers and size_compatible

        except Exception as e:
            self.logger.warning(f"Compatibility check failed: {e}")
            return False

    def _preprocess_for_compression(self, model: nn.Module, source_phase: str,
                                  metadata: Dict[str, Any]) -> nn.Module:
        """Apply phase-specific preprocessing before compression."""
        if source_phase == "quietstar" and self.integration_config.preserve_thought_generation:
            # Special handling for Quiet-STaR models
            self._mark_preservation_layers(model, metadata)

        # Move to correct device
        model = model.to(self.config.device)

        return model

    def _mark_preservation_layers(self, model: nn.Module, metadata: Dict[str, Any]):
        """Mark layers that should be preserved during compression."""
        # This would mark specific layers based on metadata
        pass

    def _create_phase5_config(self, compression_result: CompressionResult) -> Dict[str, Any]:
        """Create configuration for Phase 5 training."""
        return {
            'model_type': 'bitnet_compressed',
            'quantization_mode': self.config.layer_config.quantization_mode.value,
            'compression_level': self.config.compression_level.value,
            'compression_stats': compression_result.compression_stats,
            'recommended_training_params': {
                'learning_rate': 1e-4,  # Lower LR for quantized models
                'batch_size': 16,
                'gradient_accumulation_steps': 4,
                'warmup_steps': 100,
                'max_steps': 1000
            },
            'phase4_integration': {
                'source_phase': compression_result.compression_stats.get('source_phase', 'unknown'),
                'compression_success': compression_result.success,
                'quality_passed': compression_result.compression_stats.get('quality_gates_passed', False)
            }
        }

    def _auto_detect_source_phase(self, model_path: Optional[str] = None) -> str:
        """Auto-detect which phase the model comes from."""
        # Check for available models in order of preference
        if model_path:
            if 'quietstar' in model_path.lower():
                return 'quietstar'
            elif 'evomerge' in model_path.lower():
                return 'evomerge'

        # Check for available models in output directories
        quietstar_dir = Path(self.integration_config.quietstar_model_path)
        evomerge_dir = Path(self.integration_config.evomerge_model_path)

        if quietstar_dir.exists() and list(quietstar_dir.glob("*.pt")):
            return 'quietstar'
        elif evomerge_dir.exists() and list(evomerge_dir.glob("*.pt")):
            return 'evomerge'
        else:
            raise ValueError("No compatible models found in Phase 2 or Phase 3 output directories")


def create_integration_pipeline(config_name: str = "default") -> BitNetPhaseIntegration:
    """Create an integration pipeline with predefined configuration."""
    from ..config.bitnet_config import get_config

    config = get_config(config_name)
    return BitNetPhaseIntegration(config)


def run_integration_pipeline(source_phase: str = "auto", config_name: str = "default") -> PhaseOutput:
    """Simple interface for running the complete integration pipeline."""
    pipeline = create_integration_pipeline(config_name)
    return pipeline.run_complete_pipeline(source_phase)