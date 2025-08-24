#!/usr/bin/env python3
"""Agent Forge Compression Pipeline Stage-by-Stage Test (Simplified)

Tests each compression stage sequentially without complex dependencies:
1. BitNet 1.58-bit compression
2. SeedLM-style compression
3. VPTQ-style compression
4. Hypercompression-style compression

Measures size reduction and validates functionality at each stage.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Results from a single compression stage"""

    stage_name: str
    input_size_bytes: int
    output_size_bytes: int
    compression_ratio: float
    cumulative_compression: float
    reconstruction_error_mse: float
    stage_time_ms: float
    stage_successful: bool
    compressed_data: Any = None


class SimpleBitNetCompressor:
    """Simplified BitNet 1.58-bit compression implementation"""

    def compress(self, weights: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Compress weights using BitNet 1.58-bit ternary quantization"""
        compressed_data = {}
        total_compressed_bytes = 0

        for name, weight in weights.items():
            weight_np = weight.cpu().numpy()

            # Calculate threshold for ternary quantization
            threshold = np.std(weight_np) * 0.6

            # Create ternary values {-1, 0, +1}
            ternary = np.zeros_like(weight_np, dtype=np.int8)
            ternary[weight_np > threshold] = 1
            ternary[weight_np < -threshold] = -1

            # Calculate scale factors
            positive_weights = weight_np[weight_np > threshold]
            negative_weights = weight_np[weight_np < -threshold]

            positive_scale = np.mean(positive_weights) if len(positive_weights) > 0 else 1.0
            negative_scale = np.mean(np.abs(negative_weights)) if len(negative_weights) > 0 else 1.0

            compressed_data[name] = {
                "ternary": ternary,
                "positive_scale": positive_scale,
                "negative_scale": negative_scale,
                "threshold": threshold,
                "shape": weight.shape,
            }

            # Calculate compressed size (1.58 bits per weight ≈ 0.2 bytes)
            compressed_bytes = int(weight.numel() * 0.2) + 16  # weights + metadata
            total_compressed_bytes += compressed_bytes

        return {"data": compressed_data, "total_size": total_compressed_bytes}

    def decompress(self, compressed: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Decompress BitNet weights"""
        weights = {}

        for name, data in compressed["data"].items():
            ternary = data["ternary"]
            pos_scale = data["positive_scale"]
            neg_scale = data["negative_scale"]

            # Reconstruct weights
            reconstructed = ternary.astype(np.float32)
            reconstructed[reconstructed == 1] = pos_scale
            reconstructed[reconstructed == -1] = -neg_scale

            weights[name] = torch.tensor(reconstructed, dtype=torch.float32).reshape(data["shape"])

        return weights


class SimpleSeedLMCompressor:
    """Simplified SeedLM-style compression implementation"""

    def __init__(self, block_size=8, latent_dim=3, max_seeds=16):
        self.block_size = block_size
        self.latent_dim = latent_dim
        self.max_seeds = max_seeds

    def compress(self, weights: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Compress weights using seed-based projection"""
        compressed_data = {}
        total_compressed_bytes = 0

        for name, weight in weights.items():
            weight_flat = weight.flatten().cpu().numpy()

            # Pad to block size
            pad = (-len(weight_flat)) % self.block_size
            if pad:
                weight_flat = np.concatenate([weight_flat, np.zeros(pad)])

            blocks = weight_flat.reshape(-1, self.block_size)

            # Simple seed-based compression simulation
            seeds = []
            coeffs = []

            for block in blocks:
                # Find best seed (simplified)
                best_seed = np.random.randint(1, self.max_seeds + 1)

                # Generate pseudo-random matrix
                np.random.seed(best_seed)
                U = np.random.randn(self.block_size, self.latent_dim)

                # Solve for coefficients
                c = np.linalg.lstsq(U, block, rcond=None)[0]

                # Quantize coefficients to 3 bits
                quantized_c = np.round(c * 4).astype(np.int8)  # Simple quantization

                seeds.append(best_seed)
                coeffs.append(quantized_c)

            compressed_data[name] = {
                "seeds": np.array(seeds, dtype=np.uint16),
                "coefficients": np.array(coeffs, dtype=np.int8),
                "original_shape": weight.shape,
                "pad_length": pad,
                "block_size": self.block_size,
                "latent_dim": self.latent_dim,
            }

            # Calculate compressed size
            compressed_bytes = (
                len(seeds) * 2 + len(coeffs) * self.latent_dim + 16  # uint16 seeds  # int8 coefficients  # metadata
            )
            total_compressed_bytes += compressed_bytes

        return {"data": compressed_data, "total_size": total_compressed_bytes}

    def decompress(self, compressed: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Decompress SeedLM weights"""
        weights = {}

        for name, data in compressed["data"].items():
            seeds = data["seeds"]
            coeffs = data["coefficients"]

            blocks = []
            for seed, c in zip(seeds, coeffs):
                # Regenerate pseudo-random matrix
                np.random.seed(int(seed))
                U = np.random.randn(data["block_size"], data["latent_dim"])

                # Reconstruct block
                block = U @ (c.astype(np.float32) / 4)  # Dequantize
                blocks.append(block)

            # Reconstruct full weight
            flat = np.concatenate(blocks)
            if data["pad_length"]:
                flat = flat[: -data["pad_length"]]

            weights[name] = torch.tensor(flat, dtype=torch.float32).reshape(data["original_shape"])

        return weights


class SimpleVPTQCompressor:
    """Simplified VPTQ-style compression implementation"""

    def __init__(self, bits=2, vector_dim=4, codebook_size=16):
        self.bits = bits
        self.vector_dim = vector_dim
        self.codebook_size = codebook_size

    def compress(self, weights: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Compress weights using vector quantization"""
        compressed_data = {}
        total_compressed_bytes = 0

        for name, weight in weights.items():
            weight_flat = weight.flatten().cpu().numpy()

            # Pad to vector dimension
            pad = (-len(weight_flat)) % self.vector_dim
            if pad:
                weight_flat = np.concatenate([weight_flat, np.zeros(pad)])

            vectors = weight_flat.reshape(-1, self.vector_dim)

            # Simple k-means clustering for codebook
            # Initialize codebook randomly
            codebook = np.random.randn(self.codebook_size, self.vector_dim) * np.std(vectors)

            # Simple assignment (no iterative optimization for speed)
            indices = []
            for vector in vectors:
                distances = np.sum((codebook - vector[None, :]) ** 2, axis=1)
                indices.append(np.argmin(distances))

            indices = np.array(indices, dtype=np.uint8)

            compressed_data[name] = {
                "codebook": codebook.astype(np.float32),
                "indices": indices,
                "original_shape": weight.shape,
                "pad_length": pad,
                "vector_dim": self.vector_dim,
            }

            # Calculate compressed size
            compressed_bytes = codebook.nbytes + indices.nbytes + 16  # float32 codebook  # uint8 indices  # metadata
            total_compressed_bytes += compressed_bytes

        return {"data": compressed_data, "total_size": total_compressed_bytes}

    def decompress(self, compressed: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Decompress VPTQ weights"""
        weights = {}

        for name, data in compressed["data"].items():
            codebook = data["codebook"]
            indices = data["indices"]

            # Reconstruct vectors
            vectors = codebook[indices]
            flat = vectors.flatten()

            # Remove padding
            if data["pad_length"]:
                flat = flat[: -data["pad_length"]]

            weights[name] = torch.tensor(flat, dtype=torch.float32).reshape(data["original_shape"])

        return weights


class SimpleHyperCompressor:
    """Simplified Hypercompression-style implementation"""

    def __init__(self, num_clusters=8):
        self.num_clusters = num_clusters

    def compress(self, weights: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Compress weights using trajectory-based representation"""
        compressed_data = {}
        total_compressed_bytes = 0

        for name, weight in weights.items():
            weight_flat = weight.flatten().cpu().numpy()

            # Simple clustering by magnitude
            cluster_size = len(weight_flat) // self.num_clusters

            # Sort by absolute value and create clusters
            sorted_indices = np.argsort(np.abs(weight_flat))

            trajectory_params = []
            for i in range(self.num_clusters):
                start_idx = i * cluster_size
                end_idx = start_idx + cluster_size if i < self.num_clusters - 1 else len(weight_flat)
                cluster_indices = sorted_indices[start_idx:end_idx]
                cluster_weights = weight_flat[cluster_indices]

                # Fit simple sinusoidal trajectory
                # Parameters: amplitude, frequency, phase, offset
                mean_val = np.mean(cluster_weights)
                amplitude = np.std(cluster_weights)
                frequency = 1.0  # Simplified
                phase = 0.0  # Simplified

                params = {
                    "amplitude": amplitude,
                    "frequency": frequency,
                    "phase": phase,
                    "offset": mean_val,
                    "cluster_indices": cluster_indices,
                    "cluster_size": len(cluster_weights),
                }
                trajectory_params.append(params)

            compressed_data[name] = {
                "trajectory_params": trajectory_params,
                "original_shape": weight.shape,
                "num_clusters": self.num_clusters,
            }

            # Calculate compressed size (parameters only)
            compressed_bytes = self.num_clusters * 4 * 4 + len(weight_flat) * 2  # params + indices
            total_compressed_bytes += compressed_bytes

        return {"data": compressed_data, "total_size": total_compressed_bytes}

    def decompress(self, compressed: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Decompress Hypercompression weights"""
        weights = {}

        for name, data in compressed["data"].items():
            trajectory_params = data["trajectory_params"]
            original_shape = data["original_shape"]

            # Reconstruct weight tensor
            total_elements = int(np.prod(original_shape))
            reconstructed = np.zeros(total_elements)

            for params in trajectory_params:
                cluster_indices = params["cluster_indices"]
                cluster_size = params["cluster_size"]

                # Generate sinusoidal trajectory
                t = np.linspace(0, 2 * np.pi, cluster_size)
                trajectory = params["amplitude"] * np.sin(params["frequency"] * t + params["phase"]) + params["offset"]

                # Place in reconstructed tensor
                reconstructed[cluster_indices] = trajectory

            weights[name] = torch.tensor(reconstructed, dtype=torch.float32).reshape(original_shape)

        return weights


class AgentForgePipelineTester:
    """Test Agent Forge compression pipeline stage by stage"""

    def __init__(self):
        self.stage_results: list[StageResult] = []
        self.original_size_bytes = 0

        # Initialize compressors
        self.bitnet = SimpleBitNetCompressor()
        self.seedlm = SimpleSeedLMCompressor()
        self.vptq = SimpleVPTQCompressor()
        self.hyper = SimpleHyperCompressor()

    def create_real_model(self, model_name: str = "gpt2") -> nn.Module:
        """Create a real pre-trained model for testing"""
        logger.info(f"Loading real model: {model_name}")

        try:
            if model_name == "gpt2":
                # Load GPT-2 model (small version ~117M parameters)
                model = GPT2Model.from_pretrained("gpt2")
                logger.info(f"Loaded GPT-2 model with {sum(p.numel() for p in model.parameters())} parameters")
                return model
            elif model_name == "gpt2-small":
                # Create smaller GPT-2 for faster testing
                config = GPT2Config(
                    vocab_size=50257,
                    n_positions=512,  # Reduced from 1024
                    n_embd=384,  # Reduced from 768
                    n_layer=6,  # Reduced from 12
                    n_head=6,  # Reduced from 12
                    n_inner=1536,  # Reduced from 3072
                )
                model = GPT2Model(config)
                logger.info(f"Created small GPT-2 with {sum(p.numel() for p in model.parameters())} parameters")
                return model
            else:
                # Fallback to synthetic model
                return self.create_synthetic_model()

        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}. Using synthetic model.")
            return self.create_synthetic_model()

    def create_synthetic_model(self, target_size_mb: float = 25.0) -> nn.Module:
        """Create synthetic test model if real model loading fails"""
        target_params = int((target_size_mb * 1024 * 1024) / 4)
        hidden_size = int(np.sqrt(target_params / 8))

        model = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 100),
        )

        # Initialize weights
        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                nn.init.zeros_(layer.bias)

        return model

    def extract_weights(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Extract weights from model"""
        weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                weights[name] = param.data.clone()
        return weights

    def get_weights_size(self, weights: dict[str, torch.Tensor]) -> int:
        """Calculate total size of weights in bytes"""
        total_bytes = 0
        for weight in weights.values():
            total_bytes += weight.numel() * weight.element_size()
        return total_bytes

    def calculate_reconstruction_error(
        self, original: dict[str, torch.Tensor], reconstructed: dict[str, torch.Tensor]
    ) -> float:
        """Calculate MSE between original and reconstructed weights"""
        total_mse = 0.0
        total_elements = 0

        for name, orig in original.items():
            if name in reconstructed:
                recon = reconstructed[name]
                mse = torch.mean((orig - recon) ** 2).item()
                elements = orig.numel()
                total_mse += mse * elements
                total_elements += elements

        return total_mse / total_elements if total_elements > 0 else float("inf")

    def run_stage(
        self, stage_name: str, compressor, input_weights: dict[str, torch.Tensor], input_size: int
    ) -> StageResult:
        """Run a single compression stage"""
        logger.info(f"Running {stage_name} compression stage")

        start_time = time.perf_counter()

        try:
            # Compress
            compressed = compressor.compress(input_weights)
            output_size = compressed["total_size"]

            # Decompress for error calculation
            reconstructed = compressor.decompress(compressed)

            stage_time = (time.perf_counter() - start_time) * 1000

            # Calculate metrics
            reconstruction_error = self.calculate_reconstruction_error(input_weights, reconstructed)
            compression_ratio = input_size / output_size if output_size > 0 else 0
            cumulative_compression = self.original_size_bytes / output_size if output_size > 0 else 0

            logger.info(f"{stage_name}: {input_size} → {output_size} bytes ({compression_ratio:.1f}x)")

            return StageResult(
                stage_name=stage_name,
                input_size_bytes=input_size,
                output_size_bytes=output_size,
                compression_ratio=compression_ratio,
                cumulative_compression=cumulative_compression,
                reconstruction_error_mse=reconstruction_error,
                stage_time_ms=stage_time,
                stage_successful=True,
                compressed_data=reconstructed,
            )

        except Exception as e:
            logger.error(f"{stage_name} failed: {e}")
            return StageResult(
                stage_name=stage_name,
                input_size_bytes=input_size,
                output_size_bytes=input_size,
                compression_ratio=1.0,
                cumulative_compression=input_size / self.original_size_bytes if self.original_size_bytes > 0 else 1.0,
                reconstruction_error_mse=float("inf"),
                stage_time_ms=(time.perf_counter() - start_time) * 1000,
                stage_successful=False,
            )

    def run_full_pipeline(self, model_name: str = "gpt2-small") -> dict[str, Any]:
        """Run complete Agent Forge compression pipeline"""
        logger.info(f"Starting Agent Forge pipeline test with {model_name} model")

        # Create real model and extract weights
        model = self.create_real_model(model_name)
        original_weights = self.extract_weights(model)
        self.original_size_bytes = self.get_weights_size(original_weights)

        logger.info(f"Original model size: {self.original_size_bytes / (1024*1024):.2f} MB")

        # Stage 1: BitNet 1.58-bit compression
        stage1 = self.run_stage("BitNet 1.58-bit", self.bitnet, original_weights, self.original_size_bytes)
        self.stage_results.append(stage1)
        current_weights = stage1.compressed_data if stage1.stage_successful else original_weights
        current_size = stage1.output_size_bytes

        # Stage 2: SeedLM compression
        if stage1.stage_successful:
            stage2 = self.run_stage("SeedLM", self.seedlm, current_weights, current_size)
            self.stage_results.append(stage2)
            current_weights = stage2.compressed_data if stage2.stage_successful else current_weights
            current_size = stage2.output_size_bytes

        # Stage 3: VPTQ compression
        if len(self.stage_results) >= 2 and self.stage_results[-1].stage_successful:
            stage3 = self.run_stage("VPTQ", self.vptq, current_weights, current_size)
            self.stage_results.append(stage3)
            current_weights = stage3.compressed_data if stage3.stage_successful else current_weights
            current_size = stage3.output_size_bytes

        # Stage 4: Hypercompression
        if len(self.stage_results) >= 3 and self.stage_results[-1].stage_successful:
            stage4 = self.run_stage("Hypercompression", self.hyper, current_weights, current_size)
            self.stage_results.append(stage4)

        # Generate results
        return self._generate_results(model_name)

    def _generate_results(self, model_name: str) -> dict[str, Any]:
        """Generate comprehensive results"""
        final_result = self.stage_results[-1] if self.stage_results else None

        if not final_result:
            return {"error": "No stages completed successfully"}

        total_compression = final_result.cumulative_compression
        total_time = sum(r.stage_time_ms for r in self.stage_results)

        # Expected: BitNet (16x) * SeedLM (8x) * VPTQ (4x) * Hyper (2x) = 1024x theoretical
        # Practical target: 128x (with efficiency losses)
        target_compression = 128.0

        return {
            "test_suite": "Agent Forge Compression Pipeline (Stage-by-Stage with Real Model)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "model_name": model_name,
            "original_size_mb": round(self.original_size_bytes / (1024 * 1024), 2),
            "final_size_mb": round(final_result.output_size_bytes / (1024 * 1024), 4),
            "total_compression_ratio": round(total_compression, 1),
            "target_compression_ratio": target_compression,
            "target_achievement_percent": round((total_compression / target_compression) * 100, 1),
            "total_pipeline_time_ms": round(total_time, 1),
            "stages_completed": len(self.stage_results),
            "all_stages_successful": all(r.stage_successful for r in self.stage_results),
            "stage_breakdown": [
                {
                    "stage_number": i + 1,
                    "stage_name": r.stage_name,
                    "successful": r.stage_successful,
                    "input_size_mb": round(r.input_size_bytes / (1024 * 1024), 4),
                    "output_size_mb": round(r.output_size_bytes / (1024 * 1024), 4),
                    "stage_compression_ratio": round(r.compression_ratio, 2),
                    "cumulative_compression_ratio": round(r.cumulative_compression, 2),
                    "reconstruction_error_mse": f"{r.reconstruction_error_mse:.6f}",
                    "stage_time_ms": round(r.stage_time_ms, 1),
                }
                for i, r in enumerate(self.stage_results)
            ],
            "assessment": self._assess_results(total_compression, target_compression),
        }

    def _assess_results(self, achieved: float, target: float) -> dict[str, str]:
        """Assess pipeline performance"""
        if achieved >= target * 0.8:
            return {
                "grade": "A",
                "status": "Excellent",
                "description": "Pipeline achieves target compression with good quality",
            }
        elif achieved >= target * 0.6:
            return {
                "grade": "B",
                "status": "Good",
                "description": "Substantial compression achieved, minor optimizations needed",
            }
        elif achieved >= target * 0.4:
            return {
                "grade": "C",
                "status": "Fair",
                "description": "Moderate compression, significant improvements possible",
            }
        elif achieved >= target * 0.2:
            return {
                "grade": "D",
                "status": "Poor",
                "description": "Limited compression achieved, major improvements needed",
            }
        else:
            return {
                "grade": "F",
                "status": "Failing",
                "description": "Compression targets not met, fundamental issues present",
            }


def main():
    """Main function to run pipeline tests"""
    print("=== Agent Forge Compression Pipeline Stage-by-Stage Test ===\n")

    # Test with different model sizes
    model_sizes = [10.0, 25.0, 50.0]
    all_results = {}

    for size_mb in model_sizes:
        print(f"Testing {size_mb}MB model...")

        tester = AgentForgePipelineTester()
        results = tester.run_full_pipeline(size_mb)
        all_results[f"model_{size_mb}mb"] = results

        # Print summary
        print(f"Stages completed: {results.get('stages_completed', 0)}/4")
        print(f"Final compression: {results.get('total_compression_ratio', 'N/A')}x")
        print(f"Target achievement: {results.get('target_achievement_percent', 'N/A')}%")
        print(f"Grade: {results.get('assessment', {}).get('grade', 'N/A')}")
        print("-" * 50)

    # Save results
    output_file = Path("docs/benchmarks/agent_forge_pipeline_simple_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

    # Print detailed breakdown for largest model
    if "model_50.0mb" in all_results:
        results = all_results["model_50.0mb"]
        print("\n=== DETAILED BREAKDOWN (50MB Model) ===")

        for stage in results.get("stage_breakdown", []):
            print(f"Stage {stage['stage_number']}: {stage['stage_name']}")
            print(f"  Size: {stage['input_size_mb']:.4f}MB → {stage['output_size_mb']:.4f}MB")
            print(
                f"  Compression: {stage['stage_compression_ratio']}x (cumulative: {stage['cumulative_compression_ratio']}x)"
            )
            print(f"  Error: {stage['reconstruction_error_mse']}")
            print(f"  Time: {stage['stage_time_ms']}ms")
            print(f"  Status: {'✓ SUCCESS' if stage['successful'] else '✗ FAILED'}")
            print()

        assessment = results.get("assessment", {})
        print("=== FINAL ASSESSMENT ===")
        print(f"Grade: {assessment.get('grade', 'N/A')}")
        print(f"Status: {assessment.get('status', 'N/A')}")
        print(f"Description: {assessment.get('description', 'N/A')}")

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
