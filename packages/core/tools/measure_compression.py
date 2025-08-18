#!/usr/bin/env python3
"""
Compression measurement tool for EvoMerge models.
Measures model size, inference latency, and memory usage.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class CompressionMeasurer:
    """Measures compression metrics for models."""

    def __init__(self, device: str = "auto"):
        """Initialize compression measurer.

        Args:
            device: Target device (auto, cpu, cuda)
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def measure_model_size(self, model_path: str) -> dict[str, float]:
        """Measure model size on disk and in memory.

        Args:
            model_path: Path to model directory

        Returns:
            Dictionary with size metrics in MB
        """
        model_dir = Path(model_path)

        # Calculate disk size
        disk_size = 0
        for file_path in model_dir.rglob("*"):
            if file_path.is_file():
                disk_size += file_path.stat().st_size

        disk_size_mb = disk_size / (1024 * 1024)

        # Load model to measure memory size
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Estimate memory usage (rough approximation)
            # float16 = 2 bytes per parameter
            memory_mb = (total_params * 2) / (1024 * 1024)

            return {
                "disk_size_mb": disk_size_mb,
                "memory_size_mb": memory_mb,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "parameter_ratio": trainable_params / total_params if total_params > 0 else 0,
            }

        except Exception as e:
            print(f"Warning: Could not load model for memory measurement: {e}")
            return {
                "disk_size_mb": disk_size_mb,
                "memory_size_mb": 0,
                "total_parameters": 0,
                "trainable_parameters": 0,
                "parameter_ratio": 0,
            }

    def measure_inference_latency(
        self, model_path: str, num_samples: int = 10, max_length: int = 100
    ) -> dict[str, float]:
        """Measure inference latency.

        Args:
            model_path: Path to model directory
            num_samples: Number of inference samples
            max_length: Maximum generation length

        Returns:
            Dictionary with latency metrics in milliseconds
        """
        try:
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Test prompts
            test_prompts = [
                "What is the capital of France?",
                "Explain quantum computing in simple terms.",
                "Write a short story about a robot.",
                "Calculate 15 * 23 + 7.",
                "Describe the benefits of renewable energy.",
            ]

            latencies = []

            for i in range(num_samples):
                prompt = test_prompts[i % len(test_prompts)]
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

                # Measure generation time
                start_time = time.time()

                with torch.no_grad():
                    model.generate(
                        **inputs,
                        max_length=inputs.input_ids.shape[1] + max_length,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

            return {
                "mean_latency_ms": sum(latencies) / len(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "std_latency_ms": (sum((x - sum(latencies) / len(latencies)) ** 2 for x in latencies) / len(latencies))
                ** 0.5,
            }

        except Exception as e:
            print(f"Error measuring inference latency: {e}")
            return {
                "mean_latency_ms": 0,
                "min_latency_ms": 0,
                "max_latency_ms": 0,
                "std_latency_ms": 0,
            }

    def calculate_compression_ratio(self, original_path: str, compressed_path: str) -> dict[str, float]:
        """Calculate compression ratio between original and compressed models.

        Args:
            original_path: Path to original model
            compressed_path: Path to compressed model

        Returns:
            Dictionary with compression ratios
        """
        original_metrics = self.measure_model_size(original_path)
        compressed_metrics = self.measure_model_size(compressed_path)

        ratios = {}

        if original_metrics["disk_size_mb"] > 0:
            ratios["disk_compression_ratio"] = original_metrics["disk_size_mb"] / compressed_metrics["disk_size_mb"]
        else:
            ratios["disk_compression_ratio"] = 1.0

        if original_metrics["memory_size_mb"] > 0:
            ratios["memory_compression_ratio"] = (
                original_metrics["memory_size_mb"] / compressed_metrics["memory_size_mb"]
            )
        else:
            ratios["memory_compression_ratio"] = 1.0

        if original_metrics["total_parameters"] > 0:
            ratios["parameter_compression_ratio"] = (
                original_metrics["total_parameters"] / compressed_metrics["total_parameters"]
            )
        else:
            ratios["parameter_compression_ratio"] = 1.0

        return ratios

    def comprehensive_measurement(self, model_path: str, reference_path: str | None = None) -> dict:
        """Perform comprehensive compression measurement.

        Args:
            model_path: Path to model to measure
            reference_path: Optional path to reference model for comparison

        Returns:
            Complete measurement report
        """
        print(f"Measuring model: {model_path}")

        # Basic size metrics
        size_metrics = self.measure_model_size(model_path)
        print(f"Model size: {size_metrics['disk_size_mb']:.2f} MB")

        # Latency metrics
        latency_metrics = self.measure_inference_latency(model_path)
        print(f"Mean inference latency: {latency_metrics['mean_latency_ms']:.2f} ms")

        # Compilation metrics
        comparison_metrics = {}
        if reference_path:
            print(f"Comparing to reference: {reference_path}")
            comparison_metrics = self.calculate_compression_ratio(reference_path, model_path)
            print(f"Compression ratio: {comparison_metrics.get('disk_compression_ratio', 1.0):.2f}x")

        return {
            "model_path": model_path,
            "reference_path": reference_path,
            "size_metrics": size_metrics,
            "latency_metrics": latency_metrics,
            "comparison_metrics": comparison_metrics,
            "measurement_device": self.device,
        }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Measure compression metrics for models")
    parser.add_argument("model_path", help="Path to model directory")
    parser.add_argument("--reference", help="Path to reference model for comparison")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for inference testing",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of inference samples for latency measurement",
    )
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"Error: Model path does not exist: {args.model_path}")
        return 1

    if args.reference and not Path(args.reference).exists():
        print(f"Error: Reference path does not exist: {args.reference}")
        return 1

    # Perform measurement
    measurer = CompressionMeasurer(device=args.device)
    results = measurer.comprehensive_measurement(args.model_path, args.reference)

    # Output results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")
    elif args.verbose:
        print("\nDetailed Results:")
        print(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    exit(main())
