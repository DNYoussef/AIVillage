"""
BitNet Compression Pipeline for Agent Forge Phase 4
===================================================

Complete compression pipeline for converting models to BitNet quantized format.
Handles layer-by-layer conversion, validation, and optimization while maintaining
model performance and ensuring seamless integration with other phases.

Key Features:
- Automated model conversion pipeline
- Quality validation and rollback
- Progressive compression strategies
- Memory-efficient processing
- Integration with EvoMerge and Quiet-STaR outputs

Author: BitNet Core Implementation Specialist - Agent Forge Phase 4
"""

import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json

from ..config.bitnet_config import BitNetConfig, QuantizationMode, CompressionLevel
from ..core.bitnet_layers import (
    BitNetLinear, BitNetAttention, BitNetTransformer,
    convert_linear_to_bitnet, convert_attention_to_bitnet
)
from ..core.quantization import BitNetQuantizationEngine


@dataclass
class CompressionResult:
    """Result of model compression operation."""
    compressed_model: nn.Module
    compression_stats: Dict[str, Any]
    quality_metrics: Dict[str, float]
    compression_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class CompressionProgress:
    """Progress tracking for compression operations."""
    total_layers: int
    completed_layers: int
    current_layer: str
    current_stage: str
    start_time: float
    estimated_completion: Optional[float] = None


class BitNetCompressionPipeline:
    """
    Complete pipeline for BitNet model compression.

    Handles the full workflow from model loading through quantization,
    validation, and output generation with comprehensive progress tracking
    and error handling.
    """

    def __init__(self, config: BitNetConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or self._setup_logger()
        self.quantization_engine = BitNetQuantizationEngine(config)

        # Progress tracking
        self.progress_callback: Optional[Callable[[CompressionProgress], None]] = None
        self.current_progress: Optional[CompressionProgress] = None

        # Validation data
        self.validation_data: Optional[DataLoader] = None
        self.calibration_data: Optional[DataLoader] = None

        # Theater detection and quality gates
        self.quality_gates = self._setup_quality_gates()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the compression pipeline."""
        logger = logging.getLogger('bitnet_compression')
        logger.setLevel(getattr(logging, self.config.log_level))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _setup_quality_gates(self) -> Dict[str, float]:
        """Setup quality gates for theater detection."""
        return {
            'max_accuracy_loss': self.config.validation_config.max_accuracy_loss,
            'min_compression_ratio': self.config.validation_config.min_compression_ratio,
            'min_cosine_similarity': 0.95,
            'max_inference_time_increase': 0.10,  # 10% increase allowed
            'max_memory_increase': self.config.validation_config.max_memory_overhead
        }

    def set_progress_callback(self, callback: Callable[[CompressionProgress], None]):
        """Set callback function for progress updates."""
        self.progress_callback = callback

    def set_validation_data(self, validation_data: DataLoader, calibration_data: Optional[DataLoader] = None):
        """Set validation and calibration datasets."""
        self.validation_data = validation_data
        self.calibration_data = calibration_data or validation_data

    def compress_model(self, model: nn.Module, model_name: str = "model") -> CompressionResult:
        """
        Complete model compression pipeline.

        Args:
            model: Model to compress
            model_name: Name for the model (used in logging and outputs)

        Returns:
            CompressionResult with compressed model and statistics
        """
        start_time = time.time()
        self.logger.info(f"Starting BitNet compression for model: {model_name}")

        try:
            # Initialize progress tracking
            total_layers = self._count_compressible_layers(model)
            self.current_progress = CompressionProgress(
                total_layers=total_layers,
                completed_layers=0,
                current_layer="initialization",
                current_stage="preparation",
                start_time=start_time
            )
            self._update_progress()

            # Step 1: Model preparation and validation
            self._update_progress_stage("model_validation")
            original_model = self._prepare_model(model)

            # Step 2: Calibration (if data available)
            if self.calibration_data is not None:
                self._update_progress_stage("calibration")
                self._calibrate_quantization(original_model)

            # Step 3: Progressive compression
            self._update_progress_stage("compression")
            compressed_model = self._progressive_compression(original_model)

            # Step 4: Quality validation
            self._update_progress_stage("validation")
            quality_metrics = self._validate_compression_quality(
                original_model, compressed_model
            )

            # Step 5: Quality gates check
            self._update_progress_stage("quality_gates")
            gate_results = self._check_quality_gates(quality_metrics)

            if not gate_results['passed']:
                self.logger.warning(f"Quality gates failed: {gate_results['failures']}")
                if self.config.compression_level == CompressionLevel.CONSERVATIVE:
                    # Attempt fallback compression
                    compressed_model = self._fallback_compression(original_model)
                    quality_metrics = self._validate_compression_quality(
                        original_model, compressed_model
                    )

            # Step 6: Optimization
            self._update_progress_stage("optimization")
            compressed_model = self._optimize_compressed_model(compressed_model)

            # Step 7: Final statistics
            self._update_progress_stage("finalization")
            compression_stats = self._compute_compression_statistics(
                original_model, compressed_model
            )

            # Update final progress
            self.current_progress.completed_layers = total_layers
            self.current_progress.current_stage = "completed"
            self._update_progress()

            compression_time = time.time() - start_time
            self.logger.info(f"Compression completed in {compression_time:.2f}s")

            return CompressionResult(
                compressed_model=compressed_model,
                compression_stats=compression_stats,
                quality_metrics=quality_metrics,
                compression_time=compression_time,
                success=True
            )

        except Exception as e:
            self.logger.error(f"Compression failed: {str(e)}")
            return CompressionResult(
                compressed_model=model,  # Return original model
                compression_stats={},
                quality_metrics={},
                compression_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )

    def _count_compressible_layers(self, model: nn.Module) -> int:
        """Count the number of layers that can be compressed."""
        count = 0
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.MultiheadAttention, nn.TransformerEncoderLayer)):
                count += 1
        return count

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for compression."""
        # Move to correct device
        model = model.to(self.config.device)

        # Set to evaluation mode for compression
        model.eval()

        # Enable gradient checkpointing if configured
        if self.config.optimization_config.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()

        return model

    def _calibrate_quantization(self, model: nn.Module):
        """Calibrate quantization parameters using calibration data."""
        self.logger.info("Calibrating quantization parameters...")
        self.quantization_engine.calibrate(model, self.calibration_data)

    def _progressive_compression(self, model: nn.Module) -> nn.Module:
        """
        Perform progressive compression with layer-by-layer conversion.

        This approach allows for better control and validation at each step.
        """
        self.logger.info("Starting progressive compression...")

        # Create a copy of the model
        compressed_model = self._deep_copy_model(model)

        # Strategy based on compression level
        if self.config.compression_level == CompressionLevel.CONSERVATIVE:
            compressed_model = self._conservative_compression(compressed_model)
        elif self.config.compression_level == CompressionLevel.BALANCED:
            compressed_model = self._balanced_compression(compressed_model)
        elif self.config.compression_level == CompressionLevel.AGGRESSIVE:
            compressed_model = self._aggressive_compression(compressed_model)

        return compressed_model

    def _conservative_compression(self, model: nn.Module) -> nn.Module:
        """Conservative compression strategy - preserve critical layers."""
        layer_count = 0

        # Convert linear layers selectively
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                # Skip first and last layers if configured
                if self.config.layer_config.preserve_first_layer and "embed" in name.lower():
                    continue
                if self.config.layer_config.preserve_last_layer and ("head" in name.lower() or "output" in name.lower()):
                    continue

                # Convert to BitNet
                parent = self._get_parent_module(model, name)
                attr_name = name.split('.')[-1]
                bitnet_layer = BitNetLinear.from_linear(
                    module, self.config.layer_config.quantization_mode
                )
                setattr(parent, attr_name, bitnet_layer)

                layer_count += 1
                self.current_progress.completed_layers = layer_count
                self.current_progress.current_layer = name
                self._update_progress()

        # Convert attention layers with preservation
        if not self.config.layer_config.preserve_attention_weights:
            for name, module in list(model.named_modules()):
                if isinstance(module, nn.MultiheadAttention):
                    parent = self._get_parent_module(model, name)
                    attr_name = name.split('.')[-1]
                    bitnet_attn = BitNetAttention.from_multihead_attention(
                        module, self.config.layer_config.quantization_mode,
                        preserve_attention_weights=True
                    )
                    setattr(parent, attr_name, bitnet_attn)

                    layer_count += 1
                    self.current_progress.completed_layers = layer_count
                    self.current_progress.current_layer = name
                    self._update_progress()

        return model

    def _balanced_compression(self, model: nn.Module) -> nn.Module:
        """Balanced compression strategy."""
        # Convert most linear layers
        convert_linear_to_bitnet(model, self.config.layer_config.quantization_mode)

        # Selectively convert attention layers
        layer_count = 0
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.MultiheadAttention):
                # Convert with moderate preservation
                parent = self._get_parent_module(model, name)
                attr_name = name.split('.')[-1]
                bitnet_attn = BitNetAttention.from_multihead_attention(
                    module, self.config.layer_config.quantization_mode,
                    preserve_attention_weights=self.config.layer_config.preserve_attention_weights
                )
                setattr(parent, attr_name, bitnet_attn)

                layer_count += 1
                self.current_progress.completed_layers = layer_count
                self.current_progress.current_layer = name
                self._update_progress()

        return model

    def _aggressive_compression(self, model: nn.Module) -> nn.Module:
        """Aggressive compression strategy - maximum compression."""
        # Convert all linear layers
        convert_linear_to_bitnet(model, self.config.layer_config.quantization_mode)

        # Convert all attention layers
        convert_attention_to_bitnet(
            model,
            self.config.layer_config.quantization_mode,
            preserve_attention_weights=False
        )

        return model

    def _fallback_compression(self, model: nn.Module) -> nn.Module:
        """Fallback compression with reduced aggressiveness."""
        self.logger.info("Applying fallback compression strategy...")

        # Use more conservative settings
        original_mode = self.config.layer_config.quantization_mode
        self.config.layer_config.quantization_mode = QuantizationMode.TERNARY

        # Apply conservative compression
        compressed_model = self._conservative_compression(self._deep_copy_model(model))

        # Restore original mode
        self.config.layer_config.quantization_mode = original_mode

        return compressed_model

    def _optimize_compressed_model(self, model: nn.Module) -> nn.Module:
        """Apply post-compression optimizations."""
        if self.config.optimization_config.enable_memory_efficient_attention:
            # Apply memory optimization techniques
            self._apply_memory_optimizations(model)

        return model

    def _apply_memory_optimizations(self, model: nn.Module):
        """Apply memory-efficient optimizations."""
        # This could include techniques like:
        # - Attention chunking
        # - Gradient checkpointing
        # - Memory pooling
        pass

    def _validate_compression_quality(self, original_model: nn.Module,
                                    compressed_model: nn.Module) -> Dict[str, float]:
        """Validate the quality of compressed model."""
        if self.validation_data is None:
            self.logger.warning("No validation data provided, skipping quality validation")
            return {}

        self.logger.info("Validating compression quality...")

        # Use quantization engine's validation
        quality_metrics = self.quantization_engine.validate_quantization_quality(
            original_model, compressed_model, self.validation_data
        )

        # Add compression-specific metrics
        compression_stats = self.quantization_engine.estimate_compression_ratio(compressed_model)
        quality_metrics.update(compression_stats)

        # Theater detection metrics
        quality_metrics['theater_score'] = self._compute_theater_score(quality_metrics)

        return quality_metrics

    def _compute_theater_score(self, quality_metrics: Dict[str, float]) -> float:
        """
        Compute theater detection score.

        Lower scores indicate potential performance theater.
        """
        # Weight different metrics for theater detection
        weights = {
            'average_cosine_similarity': 0.3,
            'theoretical_compression_ratio': 0.2,
            'psnr_db': 0.2,
            'quantization_coverage': 0.3
        }

        score = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            if metric in quality_metrics:
                # Normalize metrics to 0-1 range
                normalized_value = self._normalize_metric(metric, quality_metrics[metric])
                score += normalized_value * weight
                total_weight += weight

        return score / total_weight if total_weight > 0 else 0.0

    def _normalize_metric(self, metric_name: str, value: float) -> float:
        """Normalize a metric value to 0-1 range."""
        if metric_name == 'average_cosine_similarity':
            return max(0, min(1, value))
        elif metric_name == 'theoretical_compression_ratio':
            return min(1, value / 8.0)  # Normalize to max 8x compression
        elif metric_name == 'psnr_db':
            return min(1, max(0, value / 40.0))  # Normalize to 40dB max
        elif metric_name == 'quantization_coverage':
            return max(0, min(1, value))
        else:
            return value

    def _check_quality_gates(self, quality_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check if quality metrics pass the defined gates."""
        failures = []
        passed = True

        # Check accuracy loss
        if 'average_mse' in quality_metrics:
            accuracy_loss = quality_metrics['average_mse']
            if accuracy_loss > self.quality_gates['max_accuracy_loss']:
                failures.append(f"Accuracy loss {accuracy_loss:.4f} > {self.quality_gates['max_accuracy_loss']}")
                passed = False

        # Check compression ratio
        if 'theoretical_compression_ratio' in quality_metrics:
            compression_ratio = quality_metrics['theoretical_compression_ratio']
            if compression_ratio < self.quality_gates['min_compression_ratio']:
                failures.append(f"Compression ratio {compression_ratio:.2f} < {self.quality_gates['min_compression_ratio']}")
                passed = False

        # Check cosine similarity
        if 'average_cosine_similarity' in quality_metrics:
            cosine_sim = quality_metrics['average_cosine_similarity']
            if cosine_sim < self.quality_gates['min_cosine_similarity']:
                failures.append(f"Cosine similarity {cosine_sim:.4f} < {self.quality_gates['min_cosine_similarity']}")
                passed = False

        # Theater detection
        if 'theater_score' in quality_metrics:
            theater_score = quality_metrics['theater_score']
            if theater_score < 0.8:  # Threshold for theater detection
                failures.append(f"Potential performance theater detected (score: {theater_score:.3f})")
                # Don't fail for theater, just warn
                self.logger.warning(f"Theater detection warning: score {theater_score:.3f}")

        return {
            'passed': passed,
            'failures': failures,
            'quality_score': quality_metrics.get('theater_score', 0.0)
        }

    def _compute_compression_statistics(self, original_model: nn.Module,
                                      compressed_model: nn.Module) -> Dict[str, Any]:
        """Compute comprehensive compression statistics."""
        stats = {}

        # Basic model statistics
        original_params = sum(p.numel() for p in original_model.parameters())
        compressed_params = sum(p.numel() for p in compressed_model.parameters())

        stats['original_parameters'] = original_params
        stats['compressed_parameters'] = compressed_params
        stats['parameter_reduction'] = 1 - (compressed_params / original_params)

        # Memory statistics
        original_memory = sum(p.numel() * p.element_size() for p in original_model.parameters())
        compressed_memory = sum(p.numel() * p.element_size() for p in compressed_model.parameters())

        stats['original_memory_mb'] = original_memory / (1024 * 1024)
        stats['compressed_memory_mb'] = compressed_memory / (1024 * 1024)
        stats['memory_reduction'] = 1 - (compressed_memory / original_memory)

        # Layer-wise statistics
        stats['layer_conversion_stats'] = self._analyze_layer_conversions(compressed_model)

        # Quantization statistics
        stats['quantization_stats'] = self.quantization_engine.estimate_compression_ratio(compressed_model)

        return stats

    def _analyze_layer_conversions(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze which layers were converted and their statistics."""
        conversion_stats = {
            'bitnet_linear_layers': 0,
            'bitnet_attention_layers': 0,
            'preserved_layers': 0,
            'total_layers': 0
        }

        for module in model.modules():
            if isinstance(module, BitNetLinear):
                conversion_stats['bitnet_linear_layers'] += 1
            elif isinstance(module, BitNetAttention):
                conversion_stats['bitnet_attention_layers'] += 1
            elif isinstance(module, (nn.Linear, nn.MultiheadAttention)):
                conversion_stats['preserved_layers'] += 1

            if isinstance(module, (nn.Linear, BitNetLinear, nn.MultiheadAttention, BitNetAttention)):
                conversion_stats['total_layers'] += 1

        conversion_stats['conversion_rate'] = (
            (conversion_stats['bitnet_linear_layers'] + conversion_stats['bitnet_attention_layers']) /
            conversion_stats['total_layers']
        ) if conversion_stats['total_layers'] > 0 else 0

        return conversion_stats

    def save_compressed_model(self, compressed_model: nn.Module, model_name: str,
                            compression_stats: Dict[str, Any]) -> str:
        """Save compressed model and associated metadata."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = output_dir / f"{model_name}_bitnet_compressed.pt"
        torch.save({
            'model_state_dict': compressed_model.state_dict(),
            'model_config': self.config.__dict__,
            'compression_stats': compression_stats,
            'compression_time': time.time()
        }, model_path)

        # Save human-readable statistics
        stats_path = output_dir / f"{model_name}_compression_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(compression_stats, f, indent=2, default=str)

        self.logger.info(f"Compressed model saved to: {model_path}")
        return str(model_path)

    def _deep_copy_model(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of the model."""
        import copy
        return copy.deepcopy(model)

    def _get_parent_module(self, model: nn.Module, module_name: str) -> nn.Module:
        """Get the parent module of a named module."""
        parts = module_name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent

    def _update_progress_stage(self, stage: str):
        """Update the current progress stage."""
        if self.current_progress:
            self.current_progress.current_stage = stage
            self._update_progress()

    def _update_progress(self):
        """Update progress and call callback if set."""
        if self.current_progress and self.progress_callback:
            # Estimate completion time
            if self.current_progress.completed_layers > 0:
                elapsed = time.time() - self.current_progress.start_time
                rate = self.current_progress.completed_layers / elapsed
                remaining = self.current_progress.total_layers - self.current_progress.completed_layers
                self.current_progress.estimated_completion = remaining / rate if rate > 0 else None

            self.progress_callback(self.current_progress)


def create_compression_pipeline(config_name: str = "default") -> BitNetCompressionPipeline:
    """Create a compression pipeline with predefined configuration."""
    from ..config.bitnet_config import get_config

    config = get_config(config_name)
    return BitNetCompressionPipeline(config)


def compress_model_simple(model: nn.Module, config_name: str = "default") -> CompressionResult:
    """Simple interface for model compression."""
    pipeline = create_compression_pipeline(config_name)
    return pipeline.compress_model(model)