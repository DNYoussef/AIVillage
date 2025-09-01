#!/usr/bin/env python3
"""Agent Forge Compression Pipeline Stage-by-Stage Validation

Tests each compression stage in the Agent Forge pipeline sequentially:
1. BitNet 1.58-bit compression
2. SeedLM compression
3. VPTQ compression
4. Hypercompression

Measures size reduction and validates functionality at each stage.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Add packages to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "packages"))

from packages.agent_forge.phases.final_compression import HyperCompressionEncoder, SEEDLMCompressor, VPTQCompressor

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


class AgentForgePipelineTester:
    """Test each stage of the Agent Forge compression pipeline"""

    def __init__(self):
        self.stage_results: list[StageResult] = []
        self.original_model = None
        self.original_size_bytes = 0

    def create_test_model(self, target_size_mb: float = 25.0) -> nn.Module:
        """Create test model for pipeline testing"""
        # Calculate parameters for target size
        target_params = int((target_size_mb * 1024 * 1024) / 4)  # 4 bytes per float32

        # Design layers to hit target size
        hidden_size = int(np.sqrt(target_params / 8))  # Rough calculation

        model = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 100),  # Output layer
        )

        # Initialize with reasonable values
        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                nn.init.zeros_(layer.bias)

        return model

    def get_model_size_bytes(self, model: nn.Module) -> int:
        """Calculate model size in bytes"""
        total_bytes = 0
        for param in model.parameters():
            total_bytes += param.numel() * param.element_size()
        return total_bytes

    def extract_model_weights(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Extract all weight tensors from model"""
        weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                weights[name] = param.data.clone()
        return weights

    def calculate_reconstruction_error(
        self, original_weights: dict[str, torch.Tensor], reconstructed_weights: dict[str, torch.Tensor]
    ) -> float:
        """Calculate MSE between original and reconstructed weights"""
        total_mse = 0.0
        total_elements = 0

        for name, orig_weight in original_weights.items():
            if name in reconstructed_weights:
                recon_weight = reconstructed_weights[name]
                mse = torch.mean((orig_weight - recon_weight) ** 2).item()
                elements = orig_weight.numel()
                total_mse += mse * elements
                total_elements += elements

        return total_mse / total_elements if total_elements > 0 else float("inf")

    def stage_1_bitnet_compression(self, model: nn.Module) -> StageResult:
        """Stage 1: BitNet 1.58-bit compression"""
        logger.info("Stage 1: BitNet 1.58-bit compression")

        input_size = self.get_model_size_bytes(model)
        start_time = time.perf_counter()

        try:
            # Extract weights
            original_weights = self.extract_model_weights(model)

            # BitNet 1.58-bit compression: {-1, 0, +1} ternary quantization
            compressed_data = {}
            total_compressed_bytes = 0

            for name, weight in original_weights.items():
                weight_np = weight.cpu().numpy()

                # Calculate threshold for ternary quantization
                # Use 0.5 * std as threshold for balanced distribution
                threshold = np.std(weight_np) * 0.5

                # Create ternary values {-1, 0, +1}
                ternary = np.zeros_like(weight_np, dtype=np.int8)
                ternary[weight_np > threshold] = 1
                ternary[weight_np < -threshold] = -1

                # Calculate scale factor for reconstruction
                positive_weights = weight_np[weight_np > threshold]
                negative_weights = weight_np[weight_np < -threshold]

                if len(positive_weights) > 0:
                    positive_scale = np.mean(positive_weights)
                else:
                    positive_scale = 1.0

                if len(negative_weights) > 0:
                    negative_scale = np.mean(np.abs(negative_weights))
                else:
                    negative_scale = 1.0

                compressed_data[name] = {
                    "ternary": ternary,
                    "positive_scale": positive_scale,
                    "negative_scale": negative_scale,
                    "threshold": threshold,
                    "shape": weight.shape,
                }

                # Calculate compressed size
                # In practice, 1.58 bits per weight, but simplified to 2 bits (0.25 bytes)
                # Plus scales and threshold
                compressed_bytes = (ternary.size * 2) // 8 + 16  # 2 bits per weight + metadata
                total_compressed_bytes += compressed_bytes

            stage_time = (time.perf_counter() - start_time) * 1000

            # Test reconstruction
            reconstructed_weights = {}
            for name, data in compressed_data.items():
                ternary = data["ternary"]
                pos_scale = data["positive_scale"]
                neg_scale = data["negative_scale"]

                # Reconstruct weights
                reconstructed = ternary.astype(np.float32)
                reconstructed[reconstructed == 1] = pos_scale
                reconstructed[reconstructed == -1] = -neg_scale

                reconstructed_weights[name] = torch.tensor(reconstructed, dtype=torch.float32)

            # Calculate reconstruction error
            reconstruction_error = self.calculate_reconstruction_error(original_weights, reconstructed_weights)

            compression_ratio = input_size / total_compressed_bytes if total_compressed_bytes > 0 else 0

            logger.info(f"BitNet compression: {input_size} → {total_compressed_bytes} bytes ({compression_ratio:.1f}x)")

            return StageResult(
                stage_name="BitNet 1.58-bit",
                input_size_bytes=input_size,
                output_size_bytes=total_compressed_bytes,
                compression_ratio=compression_ratio,
                cumulative_compression=compression_ratio,
                reconstruction_error_mse=reconstruction_error,
                stage_time_ms=stage_time,
                stage_successful=True,
                compressed_data=compressed_data,
            )

        except Exception as e:
            logger.error(f"BitNet compression failed: {e}")
            return StageResult(
                stage_name="BitNet 1.58-bit",
                input_size_bytes=input_size,
                output_size_bytes=input_size,
                compression_ratio=1.0,
                cumulative_compression=1.0,
                reconstruction_error_mse=float("inf"),
                stage_time_ms=(time.perf_counter() - start_time) * 1000,
                stage_successful=False,
            )

    def stage_2_seedlm_compression(self, bitnet_result: StageResult) -> StageResult:
        """Stage 2: SeedLM compression on BitNet output"""
        logger.info("Stage 2: SeedLM compression")

        input_size = bitnet_result.output_size_bytes
        start_time = time.perf_counter()

        try:
            # Initialize SeedLM compressor
            seedlm = SEEDLMCompressor(bits_per_weight=3, max_candidates=12)

            # Apply SeedLM to reconstructed weights from BitNet
            bitnet_data = bitnet_result.compressed_data
            compressed_data = {}
            total_compressed_bytes = 0

            for name, data in bitnet_data.items():
                # Reconstruct weight from BitNet for SeedLM input
                ternary = data["ternary"]
                pos_scale = data["positive_scale"]
                neg_scale = data["negative_scale"]

                reconstructed = ternary.astype(np.float32)
                reconstructed[reconstructed == 1] = pos_scale
                reconstructed[reconstructed == -1] = -neg_scale

                weight_tensor = torch.tensor(reconstructed, dtype=torch.float32).reshape(data["shape"])

                # Apply SeedLM compression
                seedlm_compressed = seedlm.compress(weight_tensor)
                compressed_data[name] = seedlm_compressed

                # Calculate compressed size
                compressed_size = (
                    len(seedlm_compressed["seeds"]) * 2
                    + seedlm_compressed["coefficients"].nbytes  # uint16 seeds
                    + len(seedlm_compressed["shared_exponents"])  # int8 coefficients  # int8 exponents
                )
                total_compressed_bytes += compressed_size

            stage_time = (time.perf_counter() - start_time) * 1000

            # Test reconstruction
            reconstructed_weights = {}
            for name, seedlm_data in compressed_data.items():
                reconstructed_weights[name] = seedlm.decompress(seedlm_data)

            # Calculate reconstruction error vs original BitNet input
            bitnet_weights = {}
            for name, data in bitnet_data.items():
                ternary = data["ternary"]
                pos_scale = data["positive_scale"]
                neg_scale = data["negative_scale"]

                reconstructed = ternary.astype(np.float32)
                reconstructed[reconstructed == 1] = pos_scale
                reconstructed[reconstructed == -1] = -neg_scale

                bitnet_weights[name] = torch.tensor(reconstructed, dtype=torch.float32).reshape(data["shape"])

            reconstruction_error = self.calculate_reconstruction_error(bitnet_weights, reconstructed_weights)

            compression_ratio = input_size / total_compressed_bytes if total_compressed_bytes > 0 else 0
            cumulative_compression = (
                self.original_size_bytes / total_compressed_bytes if total_compressed_bytes > 0 else 0
            )

            logger.info(f"SeedLM compression: {input_size} → {total_compressed_bytes} bytes ({compression_ratio:.1f}x)")

            return StageResult(
                stage_name="SeedLM",
                input_size_bytes=input_size,
                output_size_bytes=total_compressed_bytes,
                compression_ratio=compression_ratio,
                cumulative_compression=cumulative_compression,
                reconstruction_error_mse=reconstruction_error,
                stage_time_ms=stage_time,
                stage_successful=True,
                compressed_data=compressed_data,
            )

        except Exception as e:
            logger.error(f"SeedLM compression failed: {e}")
            return StageResult(
                stage_name="SeedLM",
                input_size_bytes=input_size,
                output_size_bytes=input_size,
                compression_ratio=1.0,
                cumulative_compression=bitnet_result.cumulative_compression,
                reconstruction_error_mse=float("inf"),
                stage_time_ms=(time.perf_counter() - start_time) * 1000,
                stage_successful=False,
            )

    def stage_3_vptq_compression(self, seedlm_result: StageResult) -> StageResult:
        """Stage 3: VPTQ compression on SeedLM output"""
        logger.info("Stage 3: VPTQ compression")

        input_size = seedlm_result.output_size_bytes
        start_time = time.perf_counter()

        try:
            # Initialize VPTQ compressor
            vptq = VPTQCompressor(bits=2, vector_dim=4, iterations=10)

            # Apply VPTQ to SeedLM decompressed weights
            seedlm_data = seedlm_result.compressed_data
            compressed_data = {}
            total_compressed_bytes = 0

            # First decompress SeedLM to get weights for VPTQ
            seedlm = SEEDLMCompressor(bits_per_weight=3, max_candidates=12)
            seedlm_weights = {}
            for name, seedlm_compressed in seedlm_data.items():
                seedlm_weights[name] = seedlm.decompress(seedlm_compressed)

            # Apply VPTQ compression
            for name, weight in seedlm_weights.items():
                vptq_compressed = vptq.compress(weight)
                compressed_data[name] = vptq_compressed

                # Calculate compressed size
                compressed_size = (
                    vptq_compressed["codebook"].numel() * 4
                    + len(vptq_compressed["indices"])  # float32 codebook
                    + 8  # uint8 indices  # scale + offset (float32 each)
                )
                total_compressed_bytes += compressed_size

            stage_time = (time.perf_counter() - start_time) * 1000

            # Test reconstruction
            reconstructed_weights = {}
            for name, vptq_data in compressed_data.items():
                reconstructed_weights[name] = vptq.decompress(vptq_data)

            # Calculate reconstruction error vs SeedLM input
            reconstruction_error = self.calculate_reconstruction_error(seedlm_weights, reconstructed_weights)

            compression_ratio = input_size / total_compressed_bytes if total_compressed_bytes > 0 else 0
            cumulative_compression = (
                self.original_size_bytes / total_compressed_bytes if total_compressed_bytes > 0 else 0
            )

            logger.info(f"VPTQ compression: {input_size} → {total_compressed_bytes} bytes ({compression_ratio:.1f}x)")

            return StageResult(
                stage_name="VPTQ",
                input_size_bytes=input_size,
                output_size_bytes=total_compressed_bytes,
                compression_ratio=compression_ratio,
                cumulative_compression=cumulative_compression,
                reconstruction_error_mse=reconstruction_error,
                stage_time_ms=stage_time,
                stage_successful=True,
                compressed_data=compressed_data,
            )

        except Exception as e:
            logger.error(f"VPTQ compression failed: {e}")
            return StageResult(
                stage_name="VPTQ",
                input_size_bytes=input_size,
                output_size_bytes=input_size,
                compression_ratio=1.0,
                cumulative_compression=seedlm_result.cumulative_compression,
                reconstruction_error_mse=float("inf"),
                stage_time_ms=(time.perf_counter() - start_time) * 1000,
                stage_successful=False,
            )

    def stage_4_hypercompression(self, vptq_result: StageResult) -> StageResult:
        """Stage 4: Hypercompression on VPTQ output"""
        logger.info("Stage 4: Hypercompression")

        input_size = vptq_result.output_size_bytes
        start_time = time.perf_counter()

        try:
            # Initialize Hypercompression encoder
            hyper_encoder = HyperCompressionEncoder(
                num_clusters=8, trajectory_types=["sinusoidal", "spiral"]  # Smaller for final stage
            )

            # Apply Hypercompression to VPTQ decompressed weights
            vptq_data = vptq_result.compressed_data
            compressed_data = {}
            total_compressed_bytes = 0

            # First decompress VPTQ to get weights for Hypercompression
            vptq = VPTQCompressor(bits=2, vector_dim=4, iterations=10)
            vptq_weights = {}
            for name, vptq_compressed in vptq_data.items():
                vptq_weights[name] = vptq.decompress(vptq_compressed)

            # Apply Hypercompression
            for name, weight in vptq_weights.items():
                hyper_compressed = hyper_encoder.compress(weight)
                compressed_data[name] = hyper_compressed

                # Calculate compressed size (trajectory parameters)
                compressed_size = len(hyper_compressed["params"]) * 8 * 4  # 8 params * 4 bytes each
                total_compressed_bytes += compressed_size

            stage_time = (time.perf_counter() - start_time) * 1000

            # Test reconstruction
            reconstructed_weights = {}
            for name, hyper_data in compressed_data.items():
                reconstructed_weights[name] = hyper_encoder.decompress(hyper_data)

            # Calculate reconstruction error vs VPTQ input
            reconstruction_error = self.calculate_reconstruction_error(vptq_weights, reconstructed_weights)

            compression_ratio = input_size / total_compressed_bytes if total_compressed_bytes > 0 else 0
            cumulative_compression = (
                self.original_size_bytes / total_compressed_bytes if total_compressed_bytes > 0 else 0
            )

            logger.info(f"Hypercompression: {input_size} → {total_compressed_bytes} bytes ({compression_ratio:.1f}x)")

            return StageResult(
                stage_name="Hypercompression",
                input_size_bytes=input_size,
                output_size_bytes=total_compressed_bytes,
                compression_ratio=compression_ratio,
                cumulative_compression=cumulative_compression,
                reconstruction_error_mse=reconstruction_error,
                stage_time_ms=stage_time,
                stage_successful=True,
                compressed_data=compressed_data,
            )

        except Exception as e:
            logger.error(f"Hypercompression failed: {e}")
            return StageResult(
                stage_name="Hypercompression",
                input_size_bytes=input_size,
                output_size_bytes=input_size,
                compression_ratio=1.0,
                cumulative_compression=vptq_result.cumulative_compression,
                reconstruction_error_mse=float("inf"),
                stage_time_ms=(time.perf_counter() - start_time) * 1000,
                stage_successful=False,
            )

    def run_full_pipeline_test(self, model_size_mb: float = 25.0) -> dict[str, Any]:
        """Run complete Agent Forge compression pipeline test"""
        logger.info("Starting Agent Forge compression pipeline stage-by-stage test")

        # Create test model
        self.original_model = self.create_test_model(model_size_mb)
        self.original_size_bytes = self.get_model_size_bytes(self.original_model)

        logger.info(f"Original model size: {self.original_size_bytes / (1024*1024):.2f} MB")

        # Stage 1: BitNet 1.58-bit compression
        stage1_result = self.stage_1_bitnet_compression(self.original_model)
        self.stage_results.append(stage1_result)

        if not stage1_result.stage_successful:
            return self._generate_results_with_failure("BitNet compression failed")

        # Stage 2: SeedLM compression
        stage2_result = self.stage_2_seedlm_compression(stage1_result)
        self.stage_results.append(stage2_result)

        if not stage2_result.stage_successful:
            return self._generate_results_with_failure("SeedLM compression failed")

        # Stage 3: VPTQ compression
        stage3_result = self.stage_3_vptq_compression(stage2_result)
        self.stage_results.append(stage3_result)

        if not stage3_result.stage_successful:
            return self._generate_results_with_failure("VPTQ compression failed")

        # Stage 4: Hypercompression
        stage4_result = self.stage_4_hypercompression(stage3_result)
        self.stage_results.append(stage4_result)

        # Generate comprehensive results
        return self._generate_full_results()

    def _generate_results_with_failure(self, error_message: str) -> dict[str, Any]:
        """Generate results when pipeline fails partway through"""
        return {
            "test_suite": "Agent Forge Compression Pipeline (Stage-by-Stage)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "status": "PARTIAL_FAILURE",
            "error": error_message,
            "original_size_mb": self.original_size_bytes / (1024 * 1024),
            "stages_completed": len(self.stage_results),
            "total_stages": 4,
            "stage_results": [
                {
                    "stage": r.stage_name,
                    "successful": r.stage_successful,
                    "input_size_bytes": r.input_size_bytes,
                    "output_size_bytes": r.output_size_bytes,
                    "compression_ratio": round(r.compression_ratio, 2),
                    "cumulative_compression": round(r.cumulative_compression, 2),
                    "reconstruction_error": r.reconstruction_error_mse,
                    "stage_time_ms": round(r.stage_time_ms, 1),
                }
                for r in self.stage_results
            ],
        }

    def _generate_full_results(self) -> dict[str, Any]:
        """Generate complete results for successful pipeline run"""
        final_result = self.stage_results[-1]

        # Calculate overall pipeline metrics
        total_compression = (
            self.original_size_bytes / final_result.output_size_bytes if final_result.output_size_bytes > 0 else 0
        )
        total_time = sum(r.stage_time_ms for r in self.stage_results)

        # Check if pipeline meets expected targets
        target_compression = 128.0  # BitNet (16x) * SeedLM (8x) = 128x theoretical
        pipeline_successful = (
            all(r.stage_successful for r in self.stage_results)
            and total_compression >= target_compression * 0.3  # 30% of target
        )

        return {
            "test_suite": "Agent Forge Compression Pipeline (Stage-by-Stage)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "status": "SUCCESS" if pipeline_successful else "UNDERPERFORMING",
            "original_size_mb": round(self.original_size_bytes / (1024 * 1024), 2),
            "final_size_bytes": final_result.output_size_bytes,
            "final_size_mb": round(final_result.output_size_bytes / (1024 * 1024), 4),
            "total_compression_ratio": round(total_compression, 1),
            "target_compression_ratio": target_compression,
            "target_achievement_percent": round((total_compression / target_compression) * 100, 1),
            "total_pipeline_time_ms": round(total_time, 1),
            "stages_completed": len(self.stage_results),
            "all_stages_successful": all(r.stage_successful for r in self.stage_results),
            "stage_by_stage_results": [
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
                    "size_reduction_mb": round((r.input_size_bytes - r.output_size_bytes) / (1024 * 1024), 4),
                }
                for i, r in enumerate(self.stage_results)
            ],
            "compression_breakdown": {
                "bitnet_contribution": f"{self.stage_results[0].compression_ratio:.1f}x",
                "seedlm_contribution": (
                    f"{self.stage_results[1].compression_ratio:.1f}x" if len(self.stage_results) > 1 else "N/A"
                ),
                "vptq_contribution": (
                    f"{self.stage_results[2].compression_ratio:.1f}x" if len(self.stage_results) > 2 else "N/A"
                ),
                "hypercompression_contribution": (
                    f"{self.stage_results[3].compression_ratio:.1f}x" if len(self.stage_results) > 3 else "N/A"
                ),
            },
            "quality_analysis": {
                "final_reconstruction_error": f"{final_result.reconstruction_error_mse:.6f}",
                "error_accumulation": (
                    "Low"
                    if final_result.reconstruction_error_mse < 0.1
                    else "Moderate" if final_result.reconstruction_error_mse < 1.0 else "High"
                ),
                "quality_vs_compression_tradeoff": (
                    "Excellent"
                    if total_compression > 50 and final_result.reconstruction_error_mse < 0.1
                    else "Good" if total_compression > 20 else "Poor"
                ),
            },
            "pipeline_assessment": self._assess_pipeline_performance(
                total_compression, target_compression, pipeline_successful
            ),
        }

    def _assess_pipeline_performance(self, achieved: float, target: float, successful: bool) -> dict[str, str]:
        """Assess overall pipeline performance"""
        if achieved >= target * 0.8:
            grade = "A"
            assessment = "Excellent - Achieves target compression with good quality"
        elif achieved >= target * 0.6:
            grade = "B"
            assessment = "Good - Substantial compression achieved"
        elif achieved >= target * 0.4:
            grade = "C"
            assessment = "Fair - Moderate compression, room for improvement"
        elif achieved >= target * 0.2:
            grade = "D"
            assessment = "Poor - Limited compression achieved"
        else:
            grade = "F"
            assessment = "Failing - Compression targets not met"

        return {
            "overall_grade": grade,
            "assessment": assessment,
            "recommendation": self._get_recommendation(achieved, target),
            "production_readiness": "Ready" if achieved >= target * 0.6 else "Needs Development",
        }

    def _get_recommendation(self, achieved: float, target: float) -> str:
        """Get specific recommendations based on performance"""
        if achieved >= target * 0.8:
            return "Pipeline ready for production deployment with minor optimizations"
        elif achieved >= target * 0.6:
            return "Focus on bit-packing optimization and advanced clustering algorithms"
        elif achieved >= target * 0.4:
            return "Implement proper BitNet bit-packing and enhanced SeedLM/VPTQ algorithms"
        else:
            return "Significant algorithm improvements needed across all stages"


def main():
    """Main function to run Agent Forge pipeline stage tests"""
    print("=== Agent Forge Compression Pipeline Stage-by-Stage Test ===\n")

    # Test with different model sizes
    model_sizes = [10.0, 25.0, 50.0]  # MB

    all_results = {}

    for size_mb in model_sizes:
        print(f"Testing with {size_mb}MB model...")

        tester = AgentForgePipelineTester()
        results = tester.run_full_pipeline_test(size_mb)

        all_results[f"model_{size_mb}mb"] = results

        # Print summary for this model size
        print(f"\nResults for {size_mb}MB model:")
        print(f"Final compression: {results.get('total_compression_ratio', 'N/A')}x")
        print(f"Target achievement: {results.get('target_achievement_percent', 'N/A')}%")
        print(f"Overall grade: {results.get('pipeline_assessment', {}).get('overall_grade', 'N/A')}")
        print(f"Status: {results.get('status', 'N/A')}")
        print("-" * 50)

    # Save detailed results
    output_file = Path("docs/benchmarks/agent_forge_pipeline_stage_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

    # Print stage-by-stage breakdown for largest model
    if "model_50.0mb" in all_results:
        results = all_results["model_50.0mb"]
        print("\n=== STAGE-BY-STAGE BREAKDOWN (50MB Model) ===")

        if "stage_by_stage_results" in results:
            for stage in results["stage_by_stage_results"]:
                print(f"Stage {stage['stage_number']}: {stage['stage_name']}")
                print(f"  Size: {stage['input_size_mb']:.4f}MB → {stage['output_size_mb']:.4f}MB")
                print(
                    f"  Compression: {stage['stage_compression_ratio']}x (cumulative: {stage['cumulative_compression_ratio']}x)"
                )
                print(f"  Error: {stage['reconstruction_error_mse']}")
                print(f"  Time: {stage['stage_time_ms']}ms")
                print()

        print("=== FINAL ASSESSMENT ===")
        assessment = results.get("pipeline_assessment", {})
        print(f"Grade: {assessment.get('overall_grade', 'N/A')}")
        print(f"Assessment: {assessment.get('assessment', 'N/A')}")
        print(f"Recommendation: {assessment.get('recommendation', 'N/A')}")
        print(f"Production Readiness: {assessment.get('production_readiness', 'N/A')}")

    return 0


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
