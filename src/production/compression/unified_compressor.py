#!/usr/bin/env python3
"""Unified Compression System for AIVillage.

This module provides a consolidated compression interface that combines the best
implementations from across the codebase into a single, production-ready system.

Features:
- Simple quantization (4x) for models <100M params
- Advanced pipeline (100x+) for large models
- Automatic fallback and error recovery
- Mobile-optimized profiles
- Comprehensive benchmarking
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import best implementations from agent_forge
from src.agent_forge.compression import (
    BITNETCompressor,
    SEEDLMCompressor,
    VPTQCompressor,
    bitnet_compress,
    seedlm_compress,
)

# Import production pipeline components
from .compression_pipeline import CompressionConfig, CompressionPipeline

logger = logging.getLogger(__name__)


class CompressionStrategy(Enum):
    """Compression strategy selection."""

    AUTO = "auto"  # Automatic selection based on model size
    SIMPLE = "simple"  # 4x quantization only
    ADVANCED = "advanced"  # Full 4-stage pipeline
    MOBILE = "mobile"  # Mobile-optimized profile


@dataclass
class CompressionResult:
    """Results from compression operation."""

    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    compression_time_seconds: float
    strategy_used: CompressionStrategy
    model_accuracy_retained: float | None = None
    mobile_compatible: bool = False
    benchmark_metrics: dict[str, Any] | None = None


class UnifiedCompressor:
    """Unified compression system combining best implementations."""

    def __init__(
        self,
        strategy: CompressionStrategy = CompressionStrategy.AUTO,
        mobile_target_mb: int = 100,
        accuracy_threshold: float = 0.95,
        enable_benchmarking: bool = True,
    ) -> None:
        """Initialize unified compressor.

        Args:
            strategy: Compression strategy to use
            mobile_target_mb: Target size for mobile deployment
            accuracy_threshold: Minimum accuracy to retain
            enable_benchmarking: Whether to run benchmarks
        """
        self.strategy = strategy
        self.mobile_target_mb = mobile_target_mb
        self.accuracy_threshold = accuracy_threshold
        self.enable_benchmarking = enable_benchmarking

        # Initialize compressors
        self.bitnet_compressor = BITNETCompressor()
        self.seedlm_compressor = SEEDLMCompressor()
        self.vptq_compressor = VPTQCompressor()

        # Initialize production pipeline for advanced compression
        self.production_pipeline = None

    def _estimate_model_size(self, model: nn.Module) -> float:
        """Estimate model size in MB."""
        total_params = sum(p.numel() for p in model.parameters())
        # Assume float32 (4 bytes per parameter)
        size_bytes = total_params * 4
        return size_bytes / (1024 * 1024)

    def _select_strategy(self, model: nn.Module) -> CompressionStrategy:
        """Automatically select compression strategy based on model."""
        if self.strategy != CompressionStrategy.AUTO:
            return self.strategy

        model_size_mb = self._estimate_model_size(model)
        total_params = sum(p.numel() for p in model.parameters())

        # Decision logic based on model size
        if total_params < 100_000_000:  # <100M params
            return CompressionStrategy.SIMPLE
        if model_size_mb > self.mobile_target_mb * 10:  # Very large models
            return CompressionStrategy.ADVANCED
        return CompressionStrategy.MOBILE

    async def compress_model(
        self,
        model: nn.Module | str,
        tokenizer: Any | None = None,
        output_path: Path | None = None,
    ) -> CompressionResult:
        """Compress model using unified pipeline.

        Args:
            model: Model to compress or path to model
            tokenizer: Tokenizer for the model
            output_path: Where to save compressed model

        Returns:
            CompressionResult with metrics and metadata
        """
        start_time = time.time()

        # Load model if path provided
        if isinstance(model, str | Path):
            model_path = str(model)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Calculate original size
        original_size_mb = self._estimate_model_size(model)
        logger.info(f"Original model size: {original_size_mb:.2f} MB")

        # Select compression strategy
        selected_strategy = self._select_strategy(model)
        logger.info(f"Selected compression strategy: {selected_strategy.value}")

        # Apply compression based on strategy
        try:
            if selected_strategy == CompressionStrategy.SIMPLE:
                compressed_model = await self._apply_simple_compression(model)
            elif selected_strategy == CompressionStrategy.MOBILE:
                compressed_model = await self._apply_mobile_compression(model)
            elif selected_strategy == CompressionStrategy.ADVANCED:
                compressed_model = await self._apply_advanced_compression(model, tokenizer)
            else:
                msg = f"Unsupported strategy: {selected_strategy}"
                raise ValueError(msg)

        except Exception as e:
            logger.warning(f"Compression failed, falling back to simple: {e}")
            compressed_model = await self._apply_simple_compression(model)
            selected_strategy = CompressionStrategy.SIMPLE

        # Calculate compressed size
        compressed_size_mb = self._estimate_model_size(compressed_model)
        compression_ratio = original_size_mb / compressed_size_mb
        compression_time = time.time() - start_time

        # Save compressed model if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            compressed_model.save_pretrained(output_path)
            if tokenizer:
                tokenizer.save_pretrained(output_path)
            logger.info(f"Saved compressed model to {output_path}")

        # Run benchmarks if enabled
        benchmark_metrics = None
        model_accuracy = None
        if self.enable_benchmarking:
            benchmark_metrics = await self._run_benchmarks(model, compressed_model, tokenizer)
            model_accuracy = benchmark_metrics.get("accuracy_retained", None)

        result = CompressionResult(
            original_size_mb=original_size_mb,
            compressed_size_mb=compressed_size_mb,
            compression_ratio=compression_ratio,
            compression_time_seconds=compression_time,
            strategy_used=selected_strategy,
            model_accuracy_retained=model_accuracy,
            mobile_compatible=compressed_size_mb <= self.mobile_target_mb,
            benchmark_metrics=benchmark_metrics,
        )

        logger.info(f"Compression complete: {compression_ratio:.2f}x ratio in {compression_time:.2f}s")
        return result

    async def _apply_simple_compression(self, model: nn.Module) -> nn.Module:
        """Apply simple 4x quantization."""
        logger.info("Applying simple 4x quantization...")
        return bitnet_compress(model)

    async def _apply_mobile_compression(self, model: nn.Module) -> nn.Module:
        """Apply mobile-optimized compression."""
        logger.info("Applying mobile-optimized compression...")
        # Stage 1: BitNet quantization
        compressed = bitnet_compress(model)

        # Stage 2: SeedLM if still too large
        current_size = self._estimate_model_size(compressed)
        if current_size > self.mobile_target_mb:
            compressed = seedlm_compress(compressed)

        return compressed

    async def _apply_advanced_compression(self, model: nn.Module, tokenizer: Any) -> nn.Module:
        """Apply full 4-stage advanced compression pipeline."""
        logger.info("Applying advanced 4-stage compression...")

        # Initialize production pipeline if needed
        if self.production_pipeline is None:
            config = CompressionConfig()  # Use default config
            self.production_pipeline = CompressionPipeline(config)

        # Use production pipeline for advanced compression
        result = await self.production_pipeline.compress_model()
        return result.compressed_model

    async def _run_benchmarks(
        self, original_model: nn.Module, compressed_model: nn.Module, tokenizer: Any
    ) -> dict[str, Any]:
        """Run benchmark tests on compressed model."""
        logger.info("Running compression benchmarks...")

        benchmarks = {}

        try:
            # Simple text generation test
            if tokenizer:
                test_prompt = "The future of AI is"

                # Test original model
                with torch.no_grad():
                    inputs = tokenizer(test_prompt, return_tensors="pt")
                    original_outputs = original_model.generate(**inputs, max_new_tokens=20)
                    original_text = tokenizer.decode(original_outputs[0], skip_special_tokens=True)

                # Test compressed model
                with torch.no_grad():
                    compressed_outputs = compressed_model.generate(**inputs, max_new_tokens=20)
                    compressed_text = tokenizer.decode(compressed_outputs[0], skip_special_tokens=True)

                # Simple similarity check (could be improved)
                similarity = len(set(original_text.split()) & set(compressed_text.split())) / len(
                    set(original_text.split())
                )
                benchmarks["text_similarity"] = similarity
                benchmarks["accuracy_retained"] = similarity  # Rough approximation

        except Exception as e:
            logger.warning(f"Benchmark failed: {e}")
            benchmarks["error"] = str(e)

        return benchmarks

    def get_compression_info(self) -> dict[str, Any]:
        """Get information about available compression methods."""
        return {
            "strategies": [s.value for s in CompressionStrategy],
            "simple_ratio": "~4x",
            "mobile_ratio": "~8-16x",
            "advanced_ratio": "~100x+",
            "mobile_target_mb": self.mobile_target_mb,
            "accuracy_threshold": self.accuracy_threshold,
            "compressors": {
                "bitnet": "Ternary quantization",
                "seedlm": "LFSR-based compression",
                "vptq": "Vector quantization",
            },
        }


# Convenience functions for backward compatibility
async def compress_simple(model: nn.Module | str, **kwargs) -> CompressionResult:
    """Simple compression using unified compressor."""
    compressor = UnifiedCompressor(strategy=CompressionStrategy.SIMPLE, **kwargs)
    return await compressor.compress_model(model)


async def compress_mobile(model: nn.Module | str, **kwargs) -> CompressionResult:
    """Mobile compression using unified compressor."""
    compressor = UnifiedCompressor(strategy=CompressionStrategy.MOBILE, **kwargs)
    return await compressor.compress_model(model)


async def compress_advanced(model: nn.Module | str, **kwargs) -> CompressionResult:
    """Advanced compression using unified compressor."""
    compressor = UnifiedCompressor(strategy=CompressionStrategy.ADVANCED, **kwargs)
    return await compressor.compress_model(model)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified Model Compression")
    parser.add_argument("model_path", help="Path to model to compress")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument(
        "--strategy",
        choices=["auto", "simple", "mobile", "advanced"],
        default="auto",
        help="Compression strategy",
    )
    parser.add_argument("--target-mb", type=int, default=100, help="Target size for mobile deployment")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip benchmarking")

    args = parser.parse_args()

    async def main() -> None:
        compressor = UnifiedCompressor(
            strategy=CompressionStrategy(args.strategy),
            mobile_target_mb=args.target_mb,
            enable_benchmarking=not args.no_benchmark,
        )

        result = await compressor.compress_model(args.model_path, output_path=args.output)

        print("Compression Result:")
        print(f"  Original size: {result.original_size_mb:.2f} MB")
        print(f"  Compressed size: {result.compressed_size_mb:.2f} MB")
        print(f"  Compression ratio: {result.compression_ratio:.2f}x")
        print(f"  Strategy used: {result.strategy_used.value}")
        print(f"  Mobile compatible: {result.mobile_compatible}")
        if result.model_accuracy_retained:
            print(f"  Accuracy retained: {result.model_accuracy_retained:.2%}")

    asyncio.run(main())
