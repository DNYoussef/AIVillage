#!/usr/bin/env python3
"""Benchmark Sprint 9 SimpleQuantizer, Advanced pipeline and Unified compressor."""

import os
from pathlib import Path
import sys
import time

import psutil
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from compression.pipeline import UnifiedCompressor

from core.compression.advanced_pipeline import AdvancedCompressionPipeline
from core.compression.simple_quantizer import SimpleQuantizer


def create_test_models():
    return {
        "tiny": nn.Linear(100, 100),
        "small": nn.Sequential(nn.Linear(1000, 1000), nn.ReLU(), nn.Linear(1000, 1000)),
        "medium": nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
        ),
    }


def benchmark_all_methods() -> None:
    print("Comprehensive Compression Benchmark")
    print("=" * 80)
    print("Comparing: Sprint 9 SimpleQuantizer vs Advanced Pipeline vs Unified")
    print("=" * 80)

    models = create_test_models()
    compressors = {
        "Sprint9_Simple": SimpleQuantizer(),
        "Advanced_Pipeline": AdvancedCompressionPipeline(),
        "Unified_Auto": UnifiedCompressor(),
        "Unified_Force_Simple": UnifiedCompressor(target_compression=3.0),
        "Unified_Force_Advanced": UnifiedCompressor(target_compression=100.0),
    }

    results = []

    for model_name, model in models.items():
        param_count = sum(p.numel() for p in model.parameters())
        size_mb = param_count * 4 / 1024 / 1024
        print(f"\n{model_name.upper()} Model: {param_count:,} params ({size_mb:.1f}MB)")
        print("-" * 60)

        for comp_name, compressor in compressors.items():
            print(f"\n  {comp_name}:")
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / 1024 / 1024
            try:
                start_time = time.time()
                if comp_name == "Sprint9_Simple":
                    compressed = compressor.quantize_model(model)
                    compressed_size = len(compressed)
                elif comp_name == "Advanced_Pipeline":
                    compressed = compressor.compress_model(model)
                    compressed_size = len(compressed)
                else:
                    result = compressor.compress(model)
                    compressed_size = len(result["data"])
                elapsed = time.time() - start_time
                peak_memory = process.memory_info().rss / 1024 / 1024
                mem_used = peak_memory - start_memory
                ratio = (param_count * 4) / compressed_size
                print(f"    Compression ratio: {ratio:.1f}x")
                print(f"    Compressed size: {compressed_size / 1024:.1f}KB")
                print(f"    Time: {elapsed:.2f}s")
                print(f"    Memory used: {mem_used:.1f}MB")
                if "Unified" in comp_name:
                    print(f"    Method selected: {result['method']}")
                results.append(
                    {
                        "model": model_name,
                        "compressor": comp_name,
                        "ratio": ratio,
                        "time": elapsed,
                        "memory": mem_used,
                        "size_kb": compressed_size / 1024,
                    }
                )
            except Exception as e:  # pragma: no cover - diagnostic
                print(f"    FAILED: {e}")

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Model':<10} {'Compressor':<25} {'Ratio':<10} {'Time(s)':<10} {'Memory(MB)':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<10} {r['compressor']:<25} {r['ratio']:<10.1f} " f"{r['time']:<10.2f} {r['memory']:<12.1f}")

    print("\n" + "=" * 80)
    print("MOBILE DEPLOYMENT ANALYSIS (2GB limit)")
    print("=" * 80)
    for model_name, model in models.items():
        param_count = sum(p.numel() for p in model.parameters())
        original_mb = param_count * 4 / 1024 / 1024
        print(f"\n{model_name.upper()} ({original_mb:.1f}MB original):")
        for r in results:
            if r["model"] != model_name:
                continue
            final_mb = r["size_kb"] / 1024
            fits = final_mb < 1024
            print(f"  {r['compressor']:<25} -> {final_mb:>6.1f}MB " f"[{'\u2713 FITS' if fits else '\u2717 TOO BIG'}]")


if __name__ == "__main__":
    benchmark_all_methods()
