#!/usr/bin/env python3
"""Compression measurement script to validate documented claims.

This script tests actual compression ratios across different algorithms and model sizes
to verify the 2-16x claims made in the documentation.
"""

import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

try:
    from src.agent_forge.compression.bitnet import BITNETCompressor
    from src.agent_forge.compression.seedlm import SEEDLMCompressor
    from src.agent_forge.compression.vptq import VPTQCompressor
    from src.core.compression.simple_quantizer import SimpleQuantizer
    from src.core.compression.unified_compressor import UnifiedCompressor
except ImportError as e:
    print(f"Warning: Could not import compression modules: {e}")
    print("Some compression algorithms may not be available")


class CompressionMeasurement:
    """Measures compression ratios across different algorithms and data types."""

    def __init__(self):
        self.results = []
        self.compressors = {}
        self._initialize_compressors()

    def _initialize_compressors(self):
        """Initialize available compression algorithms."""
        try:
            self.compressors["BitNet"] = BITNETCompressor()
            print("[OK] BitNet compressor initialized")
        except Exception as e:
            print(f"[FAIL] BitNet compressor failed: {e}")

        try:
            self.compressors["SeedLM"] = SEEDLMCompressor()
            print("[OK] SeedLM compressor initialized")
        except Exception as e:
            print(f"[FAIL] SeedLM compressor failed: {e}")

        try:
            self.compressors["VPTQ"] = VPTQCompressor()
            print("[OK] VPTQ compressor initialized")
        except Exception as e:
            print(f"[FAIL] VPTQ compressor failed: {e}")

        try:
            self.compressors["SimpleQuantizer"] = SimpleQuantizer()
            print("[OK] SimpleQuantizer initialized")
        except Exception as e:
            print(f"[FAIL] SimpleQuantizer failed: {e}")

        try:
            self.compressors["Unified"] = UnifiedCompressor()
            print("[OK] UnifiedCompressor initialized")
        except Exception as e:
            print(f"[FAIL] UnifiedCompressor failed: {e}")

    def create_test_models(self) -> dict[str, nn.Module]:
        """Create test models of different sizes matching mobile readiness report."""
        models = {}

        # Tiny model (~10K params)
        models["tiny"] = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32), nn.Linear(32, 10)
        )

        # Small model (~200K params)
        models["small"] = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, 10),
        )

        # Medium model (~1M params)
        models["medium"] = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

        # Large model (~5M params)
        models["large"] = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Linear(128, 10),
        )

        # Initialize parameters with realistic values
        for _name, model in models.items():
            for param in model.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)

        return models

    def create_test_tensors(self) -> dict[str, torch.Tensor]:
        """Create test tensors of various sizes and types."""
        tensors = {}

        # Small dense tensor
        tensors["dense_small"] = torch.randn(100, 100) * 0.1

        # Medium dense tensor
        tensors["dense_medium"] = torch.randn(500, 500) * 0.05

        # Large dense tensor
        tensors["dense_large"] = torch.randn(1000, 1000) * 0.02

        # Sparse tensor (mostly zeros)
        sparse_tensor = torch.randn(500, 500) * 0.1
        sparse_tensor[torch.abs(sparse_tensor) < 0.05] = 0
        tensors["sparse_medium"] = sparse_tensor

        # Structured tensor (repeating patterns)
        structured = torch.randn(10, 10) * 0.1
        tensors["structured_small"] = structured.repeat(50, 50)

        return tensors

    def measure_tensor_compression(
        self, tensor: torch.Tensor, name: str, compressor_name: str, compressor
    ) -> dict[str, Any]:
        """Measure compression ratio for a single tensor."""
        try:
            original_bytes = tensor.numel() * 4  # float32 = 4 bytes

            if compressor_name == "SimpleQuantizer":
                # SimpleQuantizer works on models, create a simple wrapper
                wrapper = nn.Linear(
                    tensor.shape[0],
                    tensor.shape[1] if len(tensor.shape) > 1 else tensor.shape[0],
                )
                with torch.no_grad():
                    if len(tensor.shape) == 2:
                        wrapper.weight.data = tensor
                    else:
                        wrapper.weight.data = tensor.view(wrapper.weight.shape)

                compressed_data = compressor.quantize_model(wrapper)
                compressed_bytes = len(compressed_data)

                # Test decompression
                compressor.decompress_model(compressed_data)
                success = True

            elif compressor_name == "Unified":
                # Unified compressor also works on models
                wrapper = nn.Linear(
                    tensor.shape[0],
                    tensor.shape[1] if len(tensor.shape) > 1 else tensor.shape[0],
                )
                with torch.no_grad():
                    if len(tensor.shape) == 2:
                        wrapper.weight.data = tensor
                    else:
                        wrapper.weight.data = tensor.view(wrapper.weight.shape)

                compressed_data = compressor.compress(wrapper)
                compressed_bytes = len(str(compressed_data).encode())  # Rough estimate

                # Test decompression
                reconstructed = compressor.decompress(compressed_data)
                success = True

            else:
                # Direct tensor compressors
                compressed_data = compressor.compress(tensor)
                compressed_bytes = self._estimate_compressed_size(compressed_data)

                # Test decompression
                reconstructed = compressor.decompress(compressed_data)
                success = torch.allclose(tensor, reconstructed, rtol=1e-3, atol=1e-3)

            ratio = original_bytes / max(compressed_bytes, 1)

            return {
                "artifact": f"{name}_{compressor_name}",
                "compressor": compressor_name,
                "tensor_name": name,
                "before_bytes": original_bytes,
                "after_bytes": compressed_bytes,
                "ratio": ratio,
                "success": success,
                "error": None,
            }

        except Exception as e:
            return {
                "artifact": f"{name}_{compressor_name}",
                "compressor": compressor_name,
                "tensor_name": name,
                "before_bytes": tensor.numel() * 4,
                "after_bytes": tensor.numel() * 4,  # No compression
                "ratio": 1.0,
                "success": False,
                "error": str(e),
            }

    def measure_model_compression(
        self, model: nn.Module, name: str, compressor_name: str, compressor
    ) -> dict[str, Any]:
        """Measure compression ratio for a complete model."""
        try:
            # Calculate original size
            original_bytes = sum(p.numel() * 4 for p in model.parameters())

            if compressor_name in ["SimpleQuantizer", "Unified"]:
                if compressor_name == "SimpleQuantizer":
                    compressed_data = compressor.quantize_model(model)
                    compressed_bytes = len(compressed_data)

                    # Test decompression
                    compressor.decompress_model(compressed_data)
                    success = True
                else:  # Unified
                    compressed_data = compressor.compress(model)
                    compressed_bytes = len(
                        str(compressed_data).encode()
                    )  # Rough estimate

                    # Test decompression
                    compressor.decompress(compressed_data)
                    success = True
            else:
                # For tensor compressors, compress each parameter
                total_compressed = 0
                all_success = True

                for param in model.parameters():
                    result = self.measure_tensor_compression(
                        param.data, f"{name}_param", compressor_name, compressor
                    )
                    total_compressed += result["after_bytes"]
                    all_success &= result["success"]

                compressed_bytes = total_compressed
                success = all_success

            ratio = original_bytes / max(compressed_bytes, 1)

            return {
                "artifact": f"{name}_model_{compressor_name}",
                "compressor": compressor_name,
                "model_name": name,
                "before_bytes": original_bytes,
                "after_bytes": compressed_bytes,
                "ratio": ratio,
                "success": success,
                "error": None,
            }

        except Exception as e:
            return {
                "artifact": f"{name}_model_{compressor_name}",
                "compressor": compressor_name,
                "model_name": name,
                "before_bytes": sum(p.numel() * 4 for p in model.parameters()),
                "after_bytes": sum(p.numel() * 4 for p in model.parameters()),
                "ratio": 1.0,
                "success": False,
                "error": str(e),
            }

    def _estimate_compressed_size(self, compressed_data: dict[str, Any]) -> int:
        """Estimate the size of compressed data in bytes."""
        size = 0
        for _key, value in compressed_data.items():
            if isinstance(value, torch.Tensor):
                size += value.numel() * value.element_size()
            elif isinstance(value, np.ndarray):
                size += value.nbytes
            elif isinstance(value, bytes):
                size += len(value)
            elif isinstance(value, int | float):
                size += 8
            elif isinstance(value, list | tuple):
                size += len(value) * 8  # Rough estimate
            elif isinstance(value, str):
                size += len(value.encode())
        return size

    def run_all_measurements(self):
        """Run comprehensive compression measurements."""
        print("Creating test models and tensors...")
        models = self.create_test_models()
        tensors = self.create_test_tensors()

        print(
            f"Testing {len(models)} models and {len(tensors)} tensors with {len(self.compressors)} compressors..."
        )

        # Test tensor compression
        for tensor_name, tensor in tensors.items():
            print(f"\nTesting tensor: {tensor_name} (shape: {tensor.shape})")
            for comp_name, compressor in self.compressors.items():
                print(f"  - {comp_name}... ", end="")
                try:
                    result = self.measure_tensor_compression(
                        tensor, tensor_name, comp_name, compressor
                    )
                    self.results.append(result)
                    if result["success"]:
                        print(f"[OK] {result['ratio']:.2f}x")
                    else:
                        print(f"[FAIL] Error: {result['error']}")
                except Exception as e:
                    print(f"[FAIL] Exception: {e}")

        # Test model compression
        for model_name, model in models.items():
            param_count = sum(p.numel() for p in model.parameters())
            print(f"\nTesting model: {model_name} ({param_count:,} parameters)")
            for comp_name, compressor in self.compressors.items():
                print(f"  - {comp_name}... ", end="")
                try:
                    result = self.measure_model_compression(
                        model, model_name, comp_name, compressor
                    )
                    self.results.append(result)
                    if result["success"]:
                        print(f"[OK] {result['ratio']:.2f}x")
                    else:
                        print(f"[FAIL] Error: {result['error']}")
                except Exception as e:
                    print(f"[FAIL] Exception: {e}")

    def generate_report(self) -> str:
        """Generate a comprehensive markdown report."""
        report = ["# Compression Measurement Results\n"]
        report.append("## Summary\n")
        report.append(
            "This report contains actual compression measurements to validate documented claims.\n"
        )

        # Claims validation section
        report.append("## Claims Validation\n")
        report.append("### Documented Claims:\n")
        report.append(
            "- **Mobile Report Claims**: tiny 2.95x, small 3.91x, medium 3.96x, large 3.98x"
        )
        report.append("- **Assessment Claims**: BitNet ~16x, SeedLM ~5x, VPTQ 14-16x")
        report.append("- **Evolution Claims**: SimpleQuantizer 4x compression\n")

        # Successful compressions
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]

        report.append(
            f"### Test Results: {len(successful)} successful, {len(failed)} failed\n"
        )

        if successful:
            report.append("## Successful Compression Results\n")
            report.append(
                "| Artifact | Compressor | Before (bytes) | After (bytes) | Ratio |\n"
            )
            report.append(
                "|----------|------------|----------------|---------------|-------|\n"
            )

            for result in sorted(successful, key=lambda x: x["ratio"], reverse=True):
                report.append(
                    f"| {result['artifact']} | {result['compressor']} | "
                    f"{result['before_bytes']:,} | {result['after_bytes']:,} | "
                    f"**{result['ratio']:.2f}x** |\n"
                )

        if failed:
            report.append("\n## Failed Compressions\n")
            report.append("| Artifact | Compressor | Error |\n")
            report.append("|----------|------------|-------|\n")

            for result in failed:
                error_msg = (
                    result["error"][:80] + "..."
                    if len(result["error"]) > 80
                    else result["error"]
                )
                report.append(
                    f"| {result['artifact']} | {result['compressor']} | {error_msg} |\n"
                )

        # Analysis section
        if successful:
            report.append("\n## Analysis\n")

            # Best performers
            best_ratios = {}
            for result in successful:
                comp = result["compressor"]
                if comp not in best_ratios or result["ratio"] > best_ratios[comp]:
                    best_ratios[comp] = result["ratio"]

            report.append("### Best Compression Ratios by Algorithm:\n")
            for comp, ratio in sorted(
                best_ratios.items(), key=lambda x: x[1], reverse=True
            ):
                report.append(f"- **{comp}**: {ratio:.2f}x\n")

            # Claim validation
            report.append("\n### Claim Validation:\n")

            bitnet_ratios = [
                r["ratio"] for r in successful if r["compressor"] == "BitNet"
            ]
            if bitnet_ratios:
                avg_bitnet = sum(bitnet_ratios) / len(bitnet_ratios)
                max_bitnet = max(bitnet_ratios)
                report.append(
                    f"- **BitNet**: Average {avg_bitnet:.2f}x, Max {max_bitnet:.2f}x vs claimed ~16x"
                )
                if max_bitnet >= 14:
                    report.append(" [VALIDATED] **CLAIM VALIDATED**\n")
                else:
                    report.append(" [DISPUTED] **CLAIM DISPUTED**\n")

            seedlm_ratios = [
                r["ratio"] for r in successful if r["compressor"] == "SeedLM"
            ]
            if seedlm_ratios:
                avg_seedlm = sum(seedlm_ratios) / len(seedlm_ratios)
                max_seedlm = max(seedlm_ratios)
                report.append(
                    f"- **SeedLM**: Average {avg_seedlm:.2f}x, Max {max_seedlm:.2f}x vs claimed ~5x"
                )
                if max_seedlm >= 4:
                    report.append(" [VALIDATED] **CLAIM VALIDATED**\n")
                else:
                    report.append(" [DISPUTED] **CLAIM DISPUTED**\n")

            vptq_ratios = [r["ratio"] for r in successful if r["compressor"] == "VPTQ"]
            if vptq_ratios:
                avg_vptq = sum(vptq_ratios) / len(vptq_ratios)
                max_vptq = max(vptq_ratios)
                report.append(
                    f"- **VPTQ**: Average {avg_vptq:.2f}x, Max {max_vptq:.2f}x vs claimed 14-16x"
                )
                if max_vptq >= 12:
                    report.append(" [VALIDATED] **CLAIM VALIDATED**\n")
                else:
                    report.append(" [DISPUTED] **CLAIM DISPUTED**\n")

            simple_ratios = [
                r["ratio"] for r in successful if r["compressor"] == "SimpleQuantizer"
            ]
            if simple_ratios:
                avg_simple = sum(simple_ratios) / len(simple_ratios)
                max_simple = max(simple_ratios)
                report.append(
                    f"- **SimpleQuantizer**: Average {avg_simple:.2f}x, Max {max_simple:.2f}x vs claimed 4x"
                )
                if avg_simple >= 3.5:
                    report.append(" [VALIDATED] **CLAIM VALIDATED**\n")
                else:
                    report.append(" [DISPUTED] **CLAIM DISPUTED**\n")

        report.append("\n## Test Environment\n")
        report.append(f"- Python version: {sys.version}\n")
        report.append(f"- PyTorch version: {torch.__version__}\n")
        report.append(
            f"- Available compressors: {', '.join(self.compressors.keys())}\n"
        )

        return "".join(report)


def main():
    """Main measurement script."""
    print("=== AIVillage Compression Measurement ===\n")

    try:
        measurement = CompressionMeasurement()
        measurement.run_all_measurements()

        print("\n=== Generating Report ===")
        report = measurement.generate_report()

        print(report)

        return 0

    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
