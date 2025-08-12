"""Stage 2 Compression Pipeline - VPTQ + HyperFn Implementation.

This module implements the Stage 2 compression pipeline that applies
VPTQ quantization and optional hyper-function compression to Stage 1 outputs.

Pipeline:
1. Load Stage 1 compressed model (.stage1.pt)
2. Apply VPTQ quantization with Hessian weighting
3. Optionally apply hyper-function compression
4. Evaluate compression ratio and accuracy
5. Save final compressed model (.stage2.pt or .gguf)
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .hyperfn import HyperCompressionEncoder
from .seedlm import SeedLMCompressor
from .vptq import VPTQQuantizer

logger = logging.getLogger(__name__)


class Stage2Compressor:
    """Main Stage 2 compression pipeline orchestrator."""

    def __init__(
        self,
        vptq_bits: float = 2.0,
        vptq_vector_length: int = 32,
        use_hyperfn: bool = True,
        hyperfn_clusters: int = 16,
    ) -> None:
        self.vptq = VPTQQuantizer(vptq_bits, vptq_vector_length)
        self.hyperfn = (
            HyperCompressionEncoder(hyperfn_clusters) if use_hyperfn else None
        )
        self.use_hyperfn = use_hyperfn

        # Setup logging
        logging.basicConfig(level=logging.INFO)

    def load_stage1_model(self, stage1_path: str) -> tuple[dict, dict]:
        """Load Stage 1 compressed model."""
        logger.info(f"Loading Stage 1 model from {stage1_path}")

        try:
            stage1_data = torch.load(stage1_path, map_location="cpu")

            # Extract compressed weights and metadata
            if "compressed_state" in stage1_data:
                compressed_weights = stage1_data["compressed_state"]
                metadata = {
                    "config": stage1_data.get("config", {}),
                    "compression_stats": stage1_data.get("compression_stats", {}),
                    "model_info": stage1_data.get("model_info", {}),
                }
            else:
                # Fallback for different Stage 1 formats
                compressed_weights = stage1_data
                metadata = {}

            return compressed_weights, metadata

        except Exception as e:
            logger.exception(f"Failed to load Stage 1 model: {e}")
            raise

    def decompress_stage1_weights(
        self, compressed_weights: dict
    ) -> dict[str, torch.Tensor]:
        """Decompress Stage 1 weights for Stage 2 processing."""
        logger.info("Decompressing Stage 1 weights")

        decompressed_weights = {}
        seedlm = SeedLMCompressor()

        for name, weight_data in compressed_weights.items():
            if isinstance(weight_data, dict) and "compressed_blocks" in weight_data:
                # This is SeedLM compressed data
                decompressed_weights[name] = seedlm.decompress_weight_matrix(
                    weight_data
                )
            else:
                # This is uncompressed data (biases, etc.)
                decompressed_weights[name] = weight_data

        return decompressed_weights

    def apply_vptq_quantization(
        self, weights: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        """Apply VPTQ quantization to decompressed weights."""
        logger.info("Applying VPTQ quantization")

        vptq_data = {}
        compression_stats = {}

        for name, weight in weights.items():
            if weight.dim() >= 2:  # Only quantize 2D+ tensors
                logger.debug(f"Quantizing parameter: {name}")

                quantized_data = self.vptq.quantize_weight_matrix(
                    weight, hessian_method="fisher"
                )
                vptq_data[name] = quantized_data

                compression_stats[name] = {
                    "compression_ratio": quantized_data["compression_ratio"],
                    "reconstruction_error": quantized_data["reconstruction_error"],
                }
            else:
                # Keep 1D parameters as-is
                vptq_data[name] = weight

        # Calculate overall compression statistics
        total_ratio = sum(
            stats["compression_ratio"] for stats in compression_stats.values()
        )
        avg_ratio = total_ratio / len(compression_stats) if compression_stats else 0

        total_error = sum(
            stats["reconstruction_error"] for stats in compression_stats.values()
        )

        logger.info(f"VPTQ compression completed. Average ratio: {avg_ratio:.2f}x")

        return {
            "vptq_data": vptq_data,
            "compression_stats": compression_stats,
            "average_compression_ratio": avg_ratio,
            "total_reconstruction_error": total_error,
        }

    def apply_hyperfn_compression(self, vptq_data: dict[str, Any]) -> dict[str, Any]:
        """Apply hyper-function compression to VPTQ data."""
        if not self.use_hyperfn:
            logger.info("Hyper-function compression disabled")
            return {"hyperfn_data": vptq_data, "compression_stats": {}}

        logger.info("Applying hyper-function compression")

        hyperfn_data = {}
        compression_stats = {}

        for name, data in vptq_data.items():
            if isinstance(data, dict) and "codebook" in data:
                # This is VPTQ quantized data - apply hyper-function to codebook
                logger.debug(f"Applying hyper-function to codebook: {name}")

                codebook = data["codebook"]
                hyperfn_compressed = self.hyperfn.compress_weight_matrix(codebook)

                # Replace codebook with hyper-function representation
                hyperfn_data[name] = {
                    **data,
                    "hyperfn_codebook": hyperfn_compressed,
                    "original_codebook_shape": codebook.shape,
                }

                compression_stats[name] = {
                    "compression_ratio": hyperfn_compressed["compression_ratio"],
                    "reconstruction_error": hyperfn_compressed["total_error"],
                }
            else:
                # Keep non-quantized data as-is
                hyperfn_data[name] = data

        # Calculate overall compression statistics
        if compression_stats:
            avg_ratio = sum(
                stats["compression_ratio"] for stats in compression_stats.values()
            ) / len(compression_stats)
            total_error = sum(
                stats["reconstruction_error"] for stats in compression_stats.values()
            )
        else:
            avg_ratio = 0
            total_error = 0

        logger.info(
            f"Hyper-function compression completed. Average ratio: {avg_ratio:.2f}x"
        )

        return {
            "hyperfn_data": hyperfn_data,
            "compression_stats": compression_stats,
            "average_compression_ratio": avg_ratio,
            "total_reconstruction_error": total_error,
        }

    def save_stage2_model(
        self, compressed_data: dict, metadata: dict, output_path: str
    ) -> None:
        """Save Stage 2 compressed model."""
        logger.info(f"Saving Stage 2 model to {output_path}")

        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Prepare final output
        final_output = {
            "stage2_compressed_data": compressed_data,
            "stage1_metadata": metadata,
            "compression_pipeline": "BitNet -> SeedLM -> VPTQ -> HyperFn",
            "timestamp": time.time(),
        }

        # Save based on file extension
        if output_path.endswith(".gguf"):
            # For GGUF format, we'd need additional conversion
            # For now, save as PT and log a warning
            logger.warning("GGUF format not fully implemented, saving as PT")
            torch.save(final_output, output_path.replace(".gguf", ".stage2.pt"))
        else:
            torch.save(final_output, output_path)

        logger.info("Stage 2 model saved successfully")

    def evaluate_compression(
        self, original_weights: dict, compressed_data: dict
    ) -> dict:
        """Evaluate Stage 2 compression quality."""
        logger.info("Evaluating Stage 2 compression")

        # Calculate overall compression statistics
        total_original_size = sum(
            w.numel() * 4 for w in original_weights.values()
        )  # 4 bytes per float32

        # Estimate compressed size (simplified)
        total_compressed_size = 0
        for data in compressed_data.values():
            if isinstance(data, dict):
                # Rough estimate of compressed size
                total_compressed_size += len(str(data)) * 4  # Very rough estimate
            else:
                total_compressed_size += (
                    data.numel() * 4 if hasattr(data, "numel") else 1000
                )

        overall_ratio = (
            total_original_size / total_compressed_size
            if total_compressed_size > 0
            else 0
        )

        # Calculate reconstruction error (simplified)
        total_error = 0
        if "compression_stats" in compressed_data:
            for stats in compressed_data["compression_stats"].values():
                total_error += stats.get("reconstruction_error", 0)

        return {
            "overall_compression_ratio": overall_ratio,
            "total_reconstruction_error": total_error,
            "original_size_mb": total_original_size / (1024 * 1024),
            "compressed_size_mb": total_compressed_size / (1024 * 1024),
        }

    def run_pipeline(self, stage1_path: str, output_path: str) -> dict:
        """Execute the complete Stage 2 compression pipeline."""
        start_time = time.time()

        try:
            # Step 1: Load Stage 1 model
            compressed_weights, metadata = self.load_stage1_model(stage1_path)

            # Step 2: Decompress Stage 1 weights
            decompressed_weights = self.decompress_stage1_weights(compressed_weights)

            # Step 3: Apply VPTQ quantization
            vptq_result = self.apply_vptq_quantization(decompressed_weights)

            # Step 4: Apply hyper-function compression
            hyperfn_result = self.apply_hyperfn_compression(vptq_result["vptq_data"])

            # Step 5: Evaluate compression
            eval_result = self.evaluate_compression(
                decompressed_weights, hyperfn_result["hyperfn_data"]
            )

            # Step 6: Save compressed model
            self.save_stage2_model(
                hyperfn_result["hyperfn_data"], metadata, output_path
            )

            # Compile results
            results = {
                "success": True,
                "vptq_compression_ratio": vptq_result["average_compression_ratio"],
                "hyperfn_compression_ratio": hyperfn_result[
                    "average_compression_ratio"
                ],
                "overall_compression_ratio": eval_result["overall_compression_ratio"],
                "total_reconstruction_error": eval_result["total_reconstruction_error"],
                "compressed_size_mb": eval_result["compressed_size_mb"],
                "total_time_seconds": time.time() - start_time,
                "output_path": output_path,
            }

            logger.info(
                f"Stage 2 compression completed successfully in {results['total_time_seconds']:.2f}s"
            )
            logger.info(
                f"Overall compression ratio: {results['overall_compression_ratio']:.2f}x"
            )

            return results

        except Exception as e:
            logger.exception(f"Stage 2 compression failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_time_seconds": time.time() - start_time,
            }


def main() -> None:
    """CLI entry point for Stage 2 compression."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 2 Compression Pipeline (VPTQ + HyperFn)"
    )
    parser.add_argument(
        "--input", required=True, help="Input Stage 1 compressed model path"
    )
    parser.add_argument(
        "--output", required=True, help="Output Stage 2 compressed model path"
    )
    parser.add_argument(
        "--vptq-bits", type=float, default=2.0, help="VPTQ bits per vector"
    )
    parser.add_argument(
        "--vptq-vector-length", type=int, default=32, help="VPTQ vector length"
    )
    parser.add_argument(
        "--disable-hyperfn",
        action="store_true",
        help="Disable hyper-function compression",
    )
    parser.add_argument(
        "--hyperfn-clusters",
        type=int,
        default=16,
        help="Number of hyper-function clusters",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level))

    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Create compressor
    compressor = Stage2Compressor(
        vptq_bits=args.vptq_bits,
        vptq_vector_length=args.vptq_vector_length,
        use_hyperfn=not args.disable_hyperfn,
        hyperfn_clusters=args.hyperfn_clusters,
    )

    # Run compression
    results = compressor.run_pipeline(args.input, args.output)

    # Print results
    if results["success"]:
        print("✅ Stage 2 compression completed successfully!")
        print(
            f"📊 Overall compression ratio: {results['overall_compression_ratio']:.2f}x"
        )
        print(f"🔧 VPTQ compression ratio: {results['vptq_compression_ratio']:.2f}x")
        if not args.disable_hyperfn:
            print(
                f"🌀 HyperFn compression ratio: {results['hyperfn_compression_ratio']:.2f}x"
            )
        print(f"💾 Compressed size: {results['compressed_size_mb']:.2f} MB")
        print(f"⏱️ Total time: {results['total_time_seconds']:.2f}s")
        print(f"📁 Output: {results['output_path']}")
    else:
        print(f"❌ Stage 2 compression failed: {results['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
